#!/usr/bin/env python3
"""External smoke tests for patched Amber/QMHub integrations.

The tests are intentionally hosted in QMHub rather than Amber's native test
tree. They expect an already patched AmberTools install and exercise the
patched executables through the public command-line interface.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import shlex
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
BOX = (20.0, 21.0, 22.0, 70.0, 80.0, 75.0)
MPI_LAUNCH_DEFAULT = "mpirun -np 2"


class TestFailure(RuntimeError):
    """Raised when one smoke-test case fails."""


@dataclass
class CommandResult:
    args: list[str]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


@dataclass
class CaseResult:
    name: str
    status: str
    detail: str


def tail(text: str, nlines: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-nlines:])


def format_command(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def run_command(
    args: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: int = 300,
) -> CommandResult:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    result = CommandResult(
        args=args,
        cwd=cwd,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if completed.returncode != 0:
        raise TestFailure(
            "\n".join(
                [
                    f"command failed: {format_command(args)}",
                    f"cwd: {cwd}",
                    f"return code: {completed.returncode}",
                    "stdout tail:",
                    tail(completed.stdout) or "<empty>",
                    "stderr tail:",
                    tail(completed.stderr) or "<empty>",
                ]
            )
        )
    return result


def resolve_command(name: str, override: str | None, amber_bin: Path | None = None) -> list[str] | None:
    if override:
        return shlex.split(override)

    if amber_bin is not None:
        candidate = amber_bin / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate)]

    found = shutil.which(name)
    if found:
        return [found]
    return None


def require_command(label: str, command: list[str] | None) -> list[str]:
    if command is None:
        raise TestFailure(
            f"missing required command: {label}\n"
            "Set AMBERHOME or pass an explicit command path to the runner."
        )
    return command


def write_command_shim(shim_dir: Path, name: str, command: str) -> None:
    command_args = shlex.split(command)
    shim = shim_dir / name
    shim.write_text(
        "#!/bin/sh\n"
        "exec "
        + " ".join(shlex.quote(arg) for arg in command_args)
        + ' "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def prepare_environment(args: argparse.Namespace, work_root: Path) -> tuple[dict[str, str], Path | None]:
    env = os.environ.copy()
    amber_home = args.amber_home or env.get("AMBERHOME")
    amber_bin = Path(amber_home).expanduser() / "bin" if amber_home else None

    shim_dir = work_root / "bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    qmhub_shim = shim_dir / "qmhub"
    qmhub_shim.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} -m qmhub \"$@\"\n",
        encoding="utf-8",
    )
    qmhub_shim.chmod(qmhub_shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # QMHub's SQM and Q-Chem backends invoke bare command names. When the
    # harness is given explicit paths, expose them through PATH-local shims so
    # sander and QMHub exercise the requested executables consistently.
    if args.sqm:
        write_command_shim(shim_dir, "sqm", args.sqm)
    if args.qchem_command:
        write_command_shim(shim_dir, "qchem", args.qchem_command)

    path_entries = [str(shim_dir)]
    if amber_bin is not None:
        path_entries.append(str(amber_bin))
    path_entries.append(env.get("PATH", ""))
    env["PATH"] = os.pathsep.join(path_entries)

    pythonpath_entries = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return env, amber_bin


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_pdb(path: Path) -> None:
    write_text(
        path,
        """\
CRYST1   20.000   21.000   22.000  70.00  80.00  75.00 P 1           1
ATOM      1  O   WAT A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  WAT A   1       0.957   0.000   0.000  1.00  0.00           H
ATOM      3  H2  WAT A   1      -0.239   0.927   0.000  1.00  0.00           H
ATOM      4  O   WAT A   2       2.900   0.100   0.000  1.00  0.00           O
ATOM      5  H1  WAT A   2       3.857   0.100   0.000  1.00  0.00           H
ATOM      6  H2  WAT A   2       2.661   1.027   0.000  1.00  0.00           H
TER
END
""",
    )


def write_tleap_input(path: Path) -> None:
    write_text(
        path,
        """\
source leaprc.water.tip3p
mol = loadpdb two_waters.pdb
set mol box { 20.0 21.0 22.0 }
saveamberparm mol system.prmtop system.rst7
quit
""",
    )


def rewrite_restart_box(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise TestFailure(f"restart file is unexpectedly short: {path}")

    box_line = "{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}{:12.7f}".format(*BOX)
    maybe_box = lines[-1].split()
    if len(maybe_box) in {3, 6}:
        lines[-1] = box_line
    else:
        lines.append(box_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_system(case_dir: Path, tleap_cmd: list[str], env: dict[str, str]) -> None:
    write_pdb(case_dir / "two_waters.pdb")
    write_tleap_input(case_dir / "tleap.in")
    run_command(tleap_cmd + ["-f", "tleap.in"], case_dir, env, timeout=120)
    rewrite_restart_box(case_dir / "system.rst7")


def write_qmhub_config(path: Path, engine: str) -> None:
    if engine == "sqm":
        engine_options = """\
[qm]
qm_theory = pm3
qmmm_int = 1
verbosity = 4
"""
    elif engine == "qchem":
        engine_options = """\
[qm]
jobtype = force
method = hf
basis = sto-3g
"""
    else:
        raise ValueError(engine)

    write_text(
        path,
        f"""\
[simulation]
protocol = md
save_input = true

[model]
switching_function = lrec
cutoff = 10.0
pbc = true

[engine]
qm = {engine}

{engine_options}""",
    )


def write_qmhub_mdin(path: Path, comm: int) -> None:
    write_text(
        path,
        f"""\
QMHub EXTERN comm={comm} smoke test
&cntrl
  imin=0, nstlim=1, dt=0.001,
  ntx=1, irest=0,
  ntb=1, cut=8.0,
  ntpr=1, ntwx=0, ntwr=0,
  ntt=0,
  ifqnt=1,
/
&qmmm
  qmmask=':1',
  qmcharge=0,
  spin=1,
  qm_theory='EXTERN',
  qmmm_int=1,
/
&qmhub
  config='qmhub.ini',
  basedir='qmhub_comm{comm}',
  comm={comm},
  debug=0,
/
""",
    )


def write_sqm_mdin(path: Path, qmmm_int: int) -> None:
    write_text(
        path,
        f"""\
Direct SQM qmmm_int={qmmm_int} field smoke test
&cntrl
  imin=0, nstlim=1, dt=0.001,
  ntx=1, irest=0,
  ntb=1, cut=8.0,
  ntpr=1, ntwx=0, ntwr=0,
  ntt=0,
  ifqnt=1,
/
&qmmm
  qmmask=':1',
  qmcharge=0,
  spin=1,
  qm_theory='PM3',
  qmmm_int={qmmm_int},
  verbosity=4,
/
""",
    )


def write_sinr_mdin(path: Path) -> None:
    write_text(
        path,
        """\
SINR ntt=12 smoke test
&cntrl
  imin=0, nstlim=2, dt=0.001,
  ntx=1, irest=0,
  ntb=1, cut=8.0,
  ntpr=1, ntwx=0, ntwr=0,
  ntc=1, ntf=1,
  ntt=12, gamma_ln=1.0,
  tempi=100.0, temp0=300.0,
  nkija=1, sinrtau=1.0,
/
""",
    )


def sander_args(sander_cmd: list[str], mdin: str, mdout: str, restart_out: str) -> list[str]:
    return sander_cmd + [
        "-O",
        "-i",
        mdin,
        "-o",
        mdout,
        "-p",
        "system.prmtop",
        "-c",
        "system.rst7",
        "-r",
        restart_out,
        "-ref",
        "system.rst7",
    ]


def assert_mdout_success(mdout: Path) -> str:
    text = mdout.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    if "error" in lower or "sander bomb" in lower:
        raise TestFailure(f"{mdout} contains an Amber error marker\n{tail(text)}")
    return text


def expected_cell_rows() -> list[list[float]]:
    a, b, c, alpha, beta, gamma = BOX
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)
    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * math.cos(gamma_r), b * math.sin(gamma_r), 0.0
    cx = c * math.cos(beta_r)
    cy = c * (math.cos(alpha_r) - math.cos(beta_r) * math.cos(gamma_r)) / math.sin(gamma_r)
    cz_sq = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz_sq, 0.0))
    return [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]


def assert_matrix_close(actual: list[list[float]], expected: list[list[float]], label: str) -> None:
    tolerance = 1.0e-5
    for i in range(3):
        for j in range(3):
            if abs(actual[i][j] - expected[i][j]) > tolerance:
                raise TestFailure(
                    f"{label} cell basis differs at row {i + 1}, column {j + 1}: "
                    f"got {actual[i][j]:.8f}, expected {expected[i][j]:.8f}\n"
                    f"actual matrix: {actual}\nexpected matrix: {expected}"
                )


def newest_saved_input(case_dir: Path, comm: int) -> Path:
    base = case_dir / f"qmhub_comm{comm}"
    matches = sorted(base.glob("**/qmmm.inp_*"))
    if not matches:
        raise TestFailure(f"no saved QMHub input exchange file found under {base}")
    return matches[-1]


def parse_text_cell(path: Path) -> list[list[float]]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if len(header) < 5:
            raise TestFailure(f"invalid QMHub text header in {path}")
        n_qm = int(header[0])
        n_mm = int(header[1])
        for _ in range(n_qm + n_mm):
            handle.readline()
        cell = []
        for _ in range(3):
            line = handle.readline()
            values = [float(value) for value in line.split()]
            if len(values) != 3:
                raise TestFailure(f"invalid text cell row in {path}: {line!r}")
            cell.append(values)
    return cell


def parse_bin_cell(path: Path) -> list[list[float]]:
    data = path.read_bytes()
    offset = 0
    if len(data) < 20:
        raise TestFailure(f"binary QMHub input is too short: {path}")
    n_qm, n_mm, _charge, _mult, _step = struct.unpack_from("=5i", data, offset)
    offset += struct.calcsize("=5i")
    offset += n_qm * struct.calcsize("=4di")
    offset += n_mm * struct.calcsize("=4d")
    need = struct.calcsize("=9d")
    if len(data) < offset + need:
        raise TestFailure(f"binary QMHub input is missing cell-basis data: {path}")
    values = struct.unpack_from("=9d", data, offset)
    return [list(values[i : i + 3]) for i in range(0, 9, 3)]


def run_qmhub_comm_case(
    case_dir: Path,
    env: dict[str, str],
    sander_cmd: list[str],
    comm: int,
    engine: str,
) -> None:
    write_qmhub_config(case_dir / "qmhub.ini", engine)
    mdin = f"qmhub_comm{comm}.in"
    mdout = f"qmhub_comm{comm}.out"
    write_qmhub_mdin(case_dir / mdin, comm)
    run_command(sander_args(sander_cmd, mdin, mdout, f"qmhub_comm{comm}.rst7"), case_dir, env)
    assert_mdout_success(case_dir / mdout)

    # The patched Amber interface writes rows for text, binary, and FIFO modes.
    # FIFO cannot be copied by save_input because it is a named pipe, so row-order
    # validation is limited to the saved text/bin exchange files.
    if comm == 0:
        assert_matrix_close(parse_text_cell(newest_saved_input(case_dir, comm)), expected_cell_rows(), "text")
    elif comm == 1:
        assert_matrix_close(parse_bin_cell(newest_saved_input(case_dir, comm)), expected_cell_rows(), "binary")


def assert_sqm_field_output(mdout: Path) -> None:
    text = assert_mdout_success(mdout)
    marker = "Electrostatic potential and field on MM atoms from QM Atoms"
    if marker not in text:
        raise TestFailure(f"{mdout} does not contain the SQM MM electrostatic-field marker")
    lines = [line for line in text.splitlines() if "QMMM: Atm" in line]
    if not lines:
        raise TestFailure(f"{mdout} does not contain printed MM electrostatic-field rows")
    for line in lines[:3]:
        numbers = [float(value) for value in re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", line)]
        if len(numbers) < 4 or not all(math.isfinite(value) for value in numbers[-4:]):
            raise TestFailure(f"invalid SQM MM electrostatic-field row in {mdout}: {line}")


def run_direct_sqm_case(
    case_dir: Path,
    env: dict[str, str],
    sander_cmd: list[str],
    qmmm_int: int,
) -> None:
    mdin = f"sqm_qmmm_int{qmmm_int}.in"
    mdout = f"sqm_qmmm_int{qmmm_int}.out"
    write_sqm_mdin(case_dir / mdin, qmmm_int)
    run_command(sander_args(sander_cmd, mdin, mdout, f"sqm_qmmm_int{qmmm_int}.rst7"), case_dir, env)
    assert_sqm_field_output(case_dir / mdout)


def run_sinr_case(case_dir: Path, env: dict[str, str], sander_cmd: list[str]) -> None:
    write_sinr_mdin(case_dir / "sinr_ntt12.in")
    run_command(sander_args(sander_cmd, "sinr_ntt12.in", "sinr_ntt12.out", "sinr_ntt12.rst7"), case_dir, env)
    text = assert_mdout_success(case_dir / "sinr_ntt12.out")
    if "Stochastic Isokinetic Nose-Hoover RESPA" not in text and "ntt=10/12" not in text:
        raise TestFailure("SINR ntt=12 run finished but did not print the expected SINR marker")


def run_case(results: list[CaseResult], name: str, func) -> None:
    try:
        func()
    except TestFailure as exc:
        results.append(CaseResult(name, "FAIL", str(exc)))
    except subprocess.TimeoutExpired as exc:
        results.append(CaseResult(name, "FAIL", f"command timed out: {format_command(exc.cmd)}"))
    else:
        results.append(CaseResult(name, "PASS", ""))


def should_run_qchem(mode: str, qchem_cmd: list[str] | None) -> tuple[bool, str]:
    value = mode.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False, "QMHUB_RUN_QCHEM disabled"
    if qchem_cmd is None:
        if value in {"1", "true", "yes", "on"}:
            raise TestFailure("Q-Chem tests were requested, but qchem was not found")
        return False, "qchem not found"
    return True, ""


def make_work_root(args: argparse.Namespace) -> tuple[Path, bool]:
    explicit = args.work_dir or os.environ.get("QMHUB_AMBER_TEST_WORK")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path, False
    return Path(tempfile.mkdtemp(prefix="qmhub-amber-tests-")), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["serial", "parallel"], required=True)
    parser.add_argument("--amber-home", help="AmberTools installation prefix")
    parser.add_argument("--work-dir", help="Working directory for generated test files")
    parser.add_argument("--keep-work", action="store_true", default=os.environ.get("QMHUB_KEEP_WORK") == "1")
    parser.add_argument("--qchem", choices=["auto", "0", "1"], default=os.environ.get("QMHUB_RUN_QCHEM", "auto"))
    parser.add_argument("--sander", help="Override serial sander command")
    parser.add_argument("--sander-mpi", help="Override parallel sander.MPI command")
    parser.add_argument("--tleap", help="Override tleap command")
    parser.add_argument("--sqm", help="Override sqm command used by QMHub")
    parser.add_argument("--qchem-command", help="Override qchem command")
    parser.add_argument("--mpi-launch", default=os.environ.get("MPI_LAUNCH", MPI_LAUNCH_DEFAULT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_root, temporary = make_work_root(args)
    keep_work = args.keep_work
    results: list[CaseResult] = []

    try:
        env, amber_bin = prepare_environment(args, work_root)
        tleap_cmd = require_command("tleap", resolve_command("tleap", args.tleap, amber_bin))
        require_command("sqm", resolve_command("sqm", args.sqm, amber_bin))
        serial_sander = require_command("sander", resolve_command("sander", args.sander, amber_bin))

        if args.mode == "parallel":
            sander_mpi = require_command(
                "sander.MPI",
                resolve_command("sander.MPI", args.sander_mpi, amber_bin),
            )
            sander_cmd = shlex.split(args.mpi_launch) + sander_mpi
        else:
            sander_cmd = serial_sander

        qchem_cmd = resolve_command("qchem", args.qchem_command)
        run_qchem, qchem_skip = should_run_qchem(args.qchem, qchem_cmd)

        case_dir = work_root / args.mode
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True)

        run_case(results, "prepare Amber system", lambda: prepare_system(case_dir, tleap_cmd, env))

        if results[-1].status == "PASS":
            for comm in (0, 1, 2):
                run_case(
                    results,
                    f"QMHub SQM comm={comm}",
                    lambda comm=comm: run_qmhub_comm_case(case_dir, env, sander_cmd, comm, "sqm"),
                )

            for qmmm_int in (6, 7):
                run_case(
                    results,
                    f"direct SQM qmmm_int={qmmm_int}",
                    lambda qmmm_int=qmmm_int: run_direct_sqm_case(case_dir, env, sander_cmd, qmmm_int),
                )

            run_case(results, "SINR ntt=12", lambda: run_sinr_case(case_dir, env, sander_cmd))

            if run_qchem:
                for comm in (0, 1, 2):
                    run_case(
                        results,
                        f"QMHub Q-Chem comm={comm}",
                        lambda comm=comm: run_qmhub_comm_case(case_dir, env, sander_cmd, comm, "qchem"),
                    )
            else:
                results.append(CaseResult("QMHub Q-Chem comm=0/1/2", "SKIP", qchem_skip))
        else:
            results.append(CaseResult("remaining cases", "SKIP", "Amber system setup failed"))

        print(f"Work directory: {work_root}")
        print("")
        for result in results:
            if result.status == "PASS":
                print(f"PASS {result.name}")
            elif result.status == "SKIP":
                print(f"SKIP {result.name}: {result.detail}")
            else:
                print(f"FAIL {result.name}")
                print(result.detail)
                print("")

        failures = [result for result in results if result.status == "FAIL"]
        if failures:
            keep_work = True
            print(f"Preserving work directory for debugging: {work_root}")
            return 1

        return 0
    except TestFailure as exc:
        print(f"Work directory: {work_root}")
        print("")
        print("FAIL setup")
        print(exc)
        return 2
    finally:
        if temporary and not keep_work:
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
