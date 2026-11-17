"""Inspect a selected QMHub/Sander/Q-Chem installation without running a job.

Run this opt-in developer tool with the intended environment's Python, preferably
with ``-I``. It is deliberately separate from QMHub's calculation path: native
linkage and environment diagnostics must not add work to individual QM/MM steps.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ENVIRONMENT_NAMES = (
    "AMBERHOME", "PMEMDHOME", "QC", "QCAUX", "QCSCRATCH", "QCPLATFORM",
    "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONNOUSERSITE", "QCTHREADS",
    "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "NCPUS", "PBS_NP",
    "SLURM_NTASKS", "OMP_PLACES", "OMP_PROC_BIND",
)


def _resolved(path):
    return Path(path).expanduser().resolve()


def _within(path, directory):
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _check_linkage(binary, env, checks):
    # Only inspect native binaries from the selected Amber installation. A
    # launcher script is not an ELF executable, and must never be run as a probe.
    try:
        with binary.open("rb") as stream:
            native = stream.read(4) == b"\x7fELF"
    except OSError as exc:
        checks.append({"status": "failed", "name": str(binary),
                       "message": f"Cannot inspect executable: {exc}"})
        return None
    if not native:
        checks.append({"status": "failed", "name": str(binary),
                       "message": "Expected an ELF executable; linkage check was skipped."})
        return None

    ldd = shutil.which("ldd", path=env.get("PATH", ""))
    if not ldd:
        checks.append({"status": "failed", "name": "linkage",
                       "message": "ldd is unavailable on PATH; native linkage could not be checked."})
        return None
    try:
        result = subprocess.run(
            [ldd, str(binary)], capture_output=True, text=True,
            check=False, timeout=30, env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks.append({"status": "failed", "name": str(binary),
                       "message": f"Could not inspect native linkage: {exc}"})
        return None

    output = result.stdout + result.stderr
    missing = [line.strip() for line in output.splitlines() if "not found" in line]
    if missing:
        checks.append({"status": "failed", "name": str(binary),
                       "message": "Missing runtime libraries: " + "; ".join(missing)
                       + ". Check the selected Sander/PLUMED module library paths."})
    elif result.returncode:
        checks.append({"status": "failed", "name": str(binary),
                       "message": f"ldd exited with status {result.returncode}: {output.strip()}"})
    else:
        checks.append({"status": "passed", "name": str(binary),
                       "message": "Native runtime libraries resolve."})
    return {"returncode": result.returncode, "missing": missing,
            "output": output.strip()}


def collect_report(amber_home=None):
    env = os.environ.copy()
    checks = []
    prefix = _resolved(sys.prefix)
    executables = {}
    for name in ("qmhub", "sander", "sander.MPI", "qchem"):
        found = shutil.which(name, path=env.get("PATH", ""))
        executables[name] = str(_resolved(found)) if found else None

    # Looking up the package spec avoids initializing NumPy, native PME, or a QM
    # backend merely to identify which package this interpreter would import.
    try:
        spec = importlib.util.find_spec("qmhub")
        origin = str(_resolved(spec.origin)) if spec and spec.origin else None
    except (ImportError, ValueError, OSError) as exc:
        origin = None
        checks.append({"status": "failed", "name": "qmhub import",
                       "message": f"Cannot resolve the QMHub package: {exc}"})
    if not origin:
        checks.append({"status": "failed", "name": "qmhub import",
                       "message": "QMHub is not discoverable by the selected interpreter."})
    elif not _within(Path(origin), prefix):
        checks.append({"status": "failed", "name": "qmhub import",
                       "message": f"QMHub resolves outside {prefix}: {origin}. "
                       "Use the intended environment's Python with -I."})
    else:
        checks.append({"status": "passed", "name": "qmhub import",
                       "message": "QMHub resolves inside the selected Python environment."})
    if not executables["qmhub"] or not _within(Path(executables["qmhub"]), prefix):
        checks.append({"status": "failed", "name": "qmhub launcher",
                       "message": f"The qmhub launcher on PATH must belong to {prefix}; "
                       "load the matching QMHub environment module."})

    chosen_home = amber_home or env.get("AMBERHOME")
    selected_home = _resolved(chosen_home) if chosen_home else None
    if selected_home is None or not selected_home.is_dir():
        checks.append({"status": "failed", "name": "AMBERHOME",
                       "message": "Select an existing Sander installation with AMBERHOME "
                       "or --amber-home; the Sander module should export AMBERHOME."})
    if selected_home is not None:
        # --amber-home describes the intended installation; this read-only tool
        # must still expose stale shell settings that a real job would inherit.
        inherited = env.get("AMBERHOME")
        if not inherited or _resolved(inherited) != selected_home:
            checks.append({"status": "failed", "name": "AMBERHOME",
                           "message": f"Shell AMBERHOME={inherited!r} does not select "
                           f"{selected_home}; reload the matching Sander module."})
        pmemd_home = env.get("PMEMDHOME")
        if pmemd_home and _resolved(pmemd_home) != selected_home:
            checks.append({"status": "failed", "name": "PMEMDHOME",
                           "message": f"PMEMDHOME={pmemd_home!r} conflicts with selected "
                           f"Sander prefix {selected_home}; check for mixed Amber modules."})

    linkage = {}
    for name in ("sander", "sander.MPI"):
        executable = executables[name]
        if not executable:
            checks.append({"status": "warning" if name == "sander.MPI" else "failed",
                           "name": name, "message": f"{name} is unavailable on PATH."})
            continue
        binary = Path(executable)
        if selected_home is None:
            continue
        if binary != (selected_home / "bin" / name).resolve():
            checks.append({"status": "failed", "name": name,
                           "message": f"PATH selects {binary}, but the selected Amber "
                           f"prefix expects {selected_home / 'bin' / name}."})
            continue
        if platform.system() == "Linux":
            linkage[name] = _check_linkage(binary, env, checks)
        else:
            checks.append({"status": "warning", "name": name,
                           "message": "Linux ELF linkage inspection is skipped on this platform."})

    qchem = executables["qchem"]
    qc = _resolved(env["QC"]) if env.get("QC") else None
    qcaux = _resolved(env["QCAUX"]) if env.get("QCAUX") else None
    if not qchem:
        checks.append({"status": "failed", "name": "qchem",
                       "message": "qchem is unavailable on PATH; load an explicit Q-Chem module."})
    if qc is None or not qc.is_dir():
        checks.append({"status": "failed", "name": "QC",
                       "message": "QC must identify an existing Q-Chem installation."})
    elif qchem and not _within(Path(qchem), qc):
        checks.append({"status": "failed", "name": "qchem",
                       "message": f"The qchem launcher {qchem} is outside QC={qc}; "
                       "check for mixed Q-Chem modules."})
    if qcaux is None or not qcaux.is_dir():
        checks.append({"status": "failed", "name": "QCAUX",
                       "message": "QCAUX must identify an existing Q-Chem auxiliary-data directory."})

    return {
        "ok": not any(item["status"] == "failed" for item in checks),
        "python": {"executable": sys.executable, "prefix": str(prefix),
                   "qmhub_origin": origin},
        "amber_home": str(selected_home) if selected_home else None,
        "executables": executables,
        "environment": {name: env.get(name) for name in ENVIRONMENT_NAMES},
        "linkage": linkage, "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amber-home", help="Intended Sander installation prefix")
    parser.add_argument("--json", action="store_true", help="Emit a structured JSON report")
    args = parser.parse_args(argv)
    report = collect_report(args.amber_home)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("QM/MM stack:", "PASS" if report["ok"] else "FAIL")
        print(f"Python: {report['python']['executable']}")
        print(f"Python prefix: {report['python']['prefix']}")
        print(f"QMHub origin: {report['python']['qmhub_origin']}")
        for name, path in report["executables"].items():
            print(f"{name}: {path}")
        for name, value in report["environment"].items():
            print(f"{name}: {value}")
        for item in report["checks"]:
            print(f"{item['status'].upper()} {item['name']}: {item['message']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
