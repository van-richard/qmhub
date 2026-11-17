"""The opt-in stack checker must inspect configuration without running jobs."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "devtools/scripts/check_qmmm_stack.py"
SPEC = importlib.util.spec_from_file_location("check_qmmm_stack", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def stack(tmp_path, monkeypatch):
    prefix = tmp_path / "python-env"
    amber = tmp_path / "sander"
    qc = tmp_path / "qchem"
    qcaux = qc / "qcaux"
    origin = prefix / "lib/site-packages/qmhub/__init__.py"
    origin.parent.mkdir(parents=True)
    origin.touch()
    qcaux.mkdir(parents=True)
    executables = {"qmhub": str(prefix / "bin/qmhub"),
                   "sander": str(amber / "bin/sander"),
                   "sander.MPI": str(amber / "bin/sander.MPI"),
                   "qchem": str(qc / "bin/qchem"), "ldd": "/usr/bin/ldd"}
    for name in ("qmhub", "sander", "sander.MPI", "qchem"):
        path = Path(executables[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELFfixture")
    env = {"PATH": "selected-path", "AMBERHOME": str(amber),
           "PMEMDHOME": str(amber), "QC": str(qc), "QCAUX": str(qcaux),
           "QCTHREADS": "3", "SLURM_NTASKS": "8"}
    monkeypatch.setattr(checker.os, "environ", env)
    monkeypatch.setattr(checker.sys, "prefix", str(prefix))
    monkeypatch.setattr(checker.sys, "executable", str(prefix / "bin/python"))
    monkeypatch.setattr(checker.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(origin)))
    monkeypatch.setattr(checker.shutil, "which", lambda name, path: executables.get(name))
    monkeypatch.setattr(checker.platform, "system", lambda: "Linux")
    calls = []

    def run(command, **kwargs):
        # Every subprocess must be a linkage inspection, never a backend launch.
        assert command[0] == "/usr/bin/ldd"
        assert command[1] in (executables.get("sander"), executables.get("sander.MPI"))
        assert kwargs["env"] == env
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="libplumed.so => /plumed/libplumed.so\n",
                               stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)
    return SimpleNamespace(prefix=prefix, amber=amber, qc=qc, origin=origin,
                           env=env, executables=executables, calls=calls)


def _messages(report, status="failed"):
    return "\n".join(item["message"] for item in report["checks"]
                     if item["status"] == status)


def test_selected_stack_passes_without_environment_mutation(stack):
    before = stack.env.copy()

    report = checker.collect_report()

    assert report["ok"]
    assert report["python"]["qmhub_origin"] == str(stack.origin)
    assert report["executables"]["sander"] == stack.executables["sander"]
    assert report["environment"]["QCTHREADS"] == "3"
    assert report["environment"]["SLURM_NTASKS"] == "8"
    assert len(stack.calls) == 2
    assert stack.env == before


@pytest.mark.parametrize("platform_name", ["Darwin", "Windows"])
def test_other_platforms_do_not_run_linux_linkage(stack, monkeypatch, platform_name):
    monkeypatch.setattr(checker.platform, "system", lambda: platform_name)

    report = checker.collect_report()

    assert report["ok"]
    assert not stack.calls
    assert "skipped on this platform" in _messages(report, "warning")


def test_missing_optional_mpi_warns(stack):
    del stack.executables["sander.MPI"]

    report = checker.collect_report()

    assert report["ok"]
    assert "sander.MPI is unavailable" in _messages(report, "warning")
    assert len(stack.calls) == 1


def test_missing_serial_is_an_error(stack):
    del stack.executables["sander"]

    report = checker.collect_report()

    assert not report["ok"]
    assert "sander is unavailable" in _messages(report)


def test_wrong_amber_executable_is_not_probed(stack):
    stack.executables["sander"] = str(stack.prefix / "bin/sander")

    report = checker.collect_report()

    assert not report["ok"]
    assert "PATH selects" in _messages(report)
    assert len(stack.calls) == 1
    assert stack.calls[0][1].endswith("sander.MPI")


def test_explicit_prefix_still_reports_stale_shell_home(stack):
    stack.env["AMBERHOME"] = str(stack.prefix)

    report = checker.collect_report(stack.amber)

    assert not report["ok"]
    assert report["amber_home"] == str(stack.amber)
    assert "Shell AMBERHOME" in _messages(report)


def test_missing_amber_home_cannot_validate_binary_origin(stack):
    del stack.env["AMBERHOME"]

    report = checker.collect_report()

    assert not report["ok"]
    assert "Select an existing Sander installation" in _messages(report)
    assert not stack.calls


def test_mixed_pmemd_home_is_an_error(stack):
    stack.env["PMEMDHOME"] = str(stack.prefix)

    report = checker.collect_report()

    assert not report["ok"]
    assert "check for mixed Amber modules" in _messages(report)


@pytest.mark.parametrize("name", ["sander", "sander.MPI"])
def test_missing_transitive_library_is_actionable(stack, monkeypatch, name):
    def run(command, **kwargs):
        output = "libplumedKernel.so => not found\n" if command[1].endswith(name) else ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)

    report = checker.collect_report()

    assert not report["ok"]
    assert "libplumedKernel.so => not found" in _messages(report)
    assert "Sander/PLUMED module library paths" in _messages(report)


@pytest.mark.parametrize("failure", [OSError("cannot execute ldd"),
                                     subprocess.TimeoutExpired("ldd", 30)])
def test_linkage_probe_failure_is_reported(stack, monkeypatch, failure):
    def run(command, **kwargs):
        raise failure

    monkeypatch.setattr(checker.subprocess, "run", run)

    report = checker.collect_report()

    assert not report["ok"]
    assert "Could not inspect native linkage" in _messages(report)


def test_non_elf_launcher_is_not_executed(stack):
    Path(stack.executables["sander"]).write_text("#!/bin/sh\nexit 99\n")

    report = checker.collect_report()

    assert not report["ok"]
    assert "Expected an ELF executable" in _messages(report)
    assert len(stack.calls) == 1


def test_ldd_missing_and_nonzero_exit_are_errors(stack, monkeypatch):
    del stack.executables["ldd"]
    assert "ldd is unavailable" in _messages(checker.collect_report())
    stack.executables["ldd"] = "/usr/bin/ldd"
    monkeypatch.setattr(checker.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr="bad ELF"))
    assert "ldd exited with status 1: bad ELF" in _messages(checker.collect_report())


def test_checkout_package_and_wrong_launcher_are_rejected(stack, monkeypatch):
    monkeypatch.setattr(checker.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin="/checkout/qmhub/__init__.py"))
    stack.executables["qmhub"] = str(stack.amber / "bin/qmhub")

    report = checker.collect_report()

    assert not report["ok"]
    assert "QMHub resolves outside" in _messages(report)
    assert "Use the intended environment's Python with -I" in _messages(report)
    assert "launcher on PATH must belong" in _messages(report)


def test_undiscoverable_package_is_reported(stack, monkeypatch):
    monkeypatch.setattr(checker.importlib.util, "find_spec", lambda name: None)

    report = checker.collect_report()

    assert not report["ok"]
    assert "QMHub is not discoverable" in _messages(report)


@pytest.mark.parametrize("setting", ["QC", "QCAUX"])
def test_missing_qchem_data_settings_are_errors(stack, setting):
    del stack.env[setting]

    report = checker.collect_report()

    assert not report["ok"]
    assert f"{setting} must identify an existing" in _messages(report)


def test_wrong_or_missing_qchem_launcher_is_reported(stack):
    stack.executables["qchem"] = str(stack.amber / "bin/qchem")
    assert "outside QC=" in _messages(checker.collect_report())
    del stack.executables["qchem"]
    assert "load an explicit Q-Chem module" in _messages(checker.collect_report())


def test_json_cli_returns_success_and_failure_status(stack, capsys):
    assert checker.main(["--json", "--amber-home", str(stack.amber)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"]
    assert report["amber_home"] == str(stack.amber)

    del stack.executables["sander"]
    assert checker.main(["--json"]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_text_cli_includes_identity_and_failures(stack, capsys):
    del stack.env["QCAUX"]

    assert checker.main([]) == 1

    output = capsys.readouterr().out
    assert "QM/MM stack: FAIL" in output
    assert f"QMHub origin: {stack.origin}" in output
    assert "QCTHREADS: 3" in output
    assert "FAILED QCAUX" in output
