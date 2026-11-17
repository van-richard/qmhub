"""Test smoke-harness setup without launching licensed QM/MM programs."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[2].joinpath(
        "devtools", "amber-qmhub-tests", "run_amber_qmhub_tests.py"
    )
    spec = importlib.util.spec_from_file_location("amber_harness_environment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def environment_args(**kwargs):
    values = dict(amber_home=None, qmhub_source="checkout", sqm=None, qchem_command=None)
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_explicit_relative_amber_home_overrides_inherited(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AMBERHOME", "/stale/amber")
    monkeypatch.setenv("PATH", "/existing/bin")
    before = os.environ.copy()

    env, amber_bin = runner.prepare_environment(
        environment_args(amber_home="selected-amber"), tmp_path / "work"
    )

    assert env["AMBERHOME"] == str(tmp_path / "selected-amber")
    assert amber_bin == tmp_path / "selected-amber" / "bin"
    assert env["PATH"].split(os.pathsep) == [
        str(tmp_path / "work" / "bin"), str(amber_bin), "/existing/bin"
    ]
    assert os.environ == before


def test_inherited_amber_home_expands_user(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("AMBERHOME", "~/amber-for-test")
    env, amber_bin = runner.prepare_environment(environment_args(), tmp_path)
    prefix = Path.home() / "amber-for-test"
    assert env["AMBERHOME"] == str(prefix.resolve())
    assert amber_bin == prefix.resolve() / "bin"


def test_missing_amber_prefix_keeps_path_lookup(runner, tmp_path, monkeypatch):
    monkeypatch.delenv("AMBERHOME", raising=False)
    monkeypatch.setenv("PATH", "/existing/bin")
    env, amber_bin = runner.prepare_environment(environment_args(), tmp_path)
    assert amber_bin is None
    assert "AMBERHOME" not in env
    assert env["PATH"].split(os.pathsep) == [str(tmp_path / "bin"), "/existing/bin"]
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/existing/bin/" + name)
    assert runner.resolve_command("sander", None, amber_bin) == ["/existing/bin/sander"]


def test_explicit_sander_command_retains_precedence(runner, tmp_path):
    sander = tmp_path / "sander"
    sander.write_text("unused")
    sander.chmod(0o755)
    assert runner.resolve_command("sander", "'/explicit path/sander' --flag", tmp_path) == [
        "/explicit path/sander", "--flag"
    ]
    assert runner.resolve_command("sander", None, tmp_path) == [str(sander)]


def test_checkout_mode_preserves_pythonpath(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing/python")
    env, _ = runner.prepare_environment(environment_args(), tmp_path)
    assert env["PYTHONPATH"].split(os.pathsep) == [str(runner.REPO_ROOT), "/existing/python"]
    shim = (tmp_path / "bin" / "qmhub").read_text().splitlines()[1]
    assert shlex.split(shim)[1:-1] == [str(Path(sys.executable).absolute()), "-m", "qmhub"]


def test_installed_mode_isolates_child_python(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(runner.REPO_ROOT))
    calls = []

    def validate(env, work_root):
        calls.append((env.copy(), work_root))
        return tmp_path / "prefix" / "qmhub" / "__init__.py"

    monkeypatch.setattr(runner, "validate_installed_qmhub", validate)
    env, _ = runner.prepare_environment(environment_args(qmhub_source="installed"), tmp_path)
    assert len(calls) == 1
    assert calls[0][1] == tmp_path
    assert "PYTHONPATH" not in calls[0][0]
    assert "PYTHONPATH" not in env
    assert os.environ["PYTHONPATH"] == str(runner.REPO_ROOT)
    shim = (tmp_path / "bin" / "qmhub").read_text().splitlines()[1]
    assert shlex.split(shim)[1:-1] == [str(Path(sys.executable).absolute()), "-I", "-m", "qmhub"]


def test_failed_installed_check_precedes_job_setup(runner, tmp_path, monkeypatch):
    def fail(*_args):
        raise runner.TestFailure("wrong installation")

    monkeypatch.setattr(runner, "validate_installed_qmhub", fail)
    with pytest.raises(runner.TestFailure, match="wrong installation"):
        runner.prepare_environment(environment_args(qmhub_source="installed"), tmp_path)
    assert not (tmp_path / "bin").exists()


def test_installed_import_checks_interpreter_and_origin(runner, tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    module = prefix / "lib" / "qmhub" / "__init__.py"
    calls = []

    def command(args, cwd, env, timeout):
        calls.append((args, cwd, env, timeout))
        return runner.CommandResult(args, cwd, 0, json.dumps({"module": str(module), "prefix": str(prefix)}), "")

    monkeypatch.setattr(runner, "run_command", command)
    env = {"PATH": "/bin"}
    assert runner.validate_installed_qmhub(env, tmp_path) == module
    args, cwd, passed_env, timeout = calls[0]
    assert args[:3] == [str(Path(sys.executable).absolute()), "-I", "-c"]
    assert "qmhub.__file__" in args[3]
    assert cwd == tmp_path
    assert passed_env is env
    assert timeout == 30


@pytest.mark.parametrize("location", ["checkout", "outside-prefix", "prefix-sibling"])
def test_installed_import_rejects_wrong_origin(runner, tmp_path, monkeypatch, location):
    prefix = tmp_path / "prefix"
    modules = {
        "checkout": runner.REPO_ROOT / "qmhub" / "__init__.py",
        "outside-prefix": tmp_path / "other" / "qmhub" / "__init__.py",
        "prefix-sibling": tmp_path / "prefix-other" / "qmhub" / "__init__.py",
    }
    output = json.dumps({"module": str(modules[location]), "prefix": str(prefix)})
    monkeypatch.setattr(
        runner, "run_command", lambda args, cwd, env, timeout: runner.CommandResult(args, cwd, 0, output, "")
    )
    with pytest.raises(runner.TestFailure, match="non-editable installation"):
        runner.validate_installed_qmhub({}, tmp_path)


def test_installed_import_rejects_symlink_outside_prefix(runner, tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = prefix / "qmhub"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    output = json.dumps({"module": str(link / "__init__.py"), "prefix": str(prefix)})
    monkeypatch.setattr(
        runner, "run_command", lambda args, cwd, env, timeout: runner.CommandResult(args, cwd, 0, output, "")
    )
    with pytest.raises(runner.TestFailure, match="non-editable installation"):
        runner.validate_installed_qmhub({}, tmp_path)


@pytest.mark.parametrize("output", ["not json", "{}", '{"module": null, "prefix": "/prefix"}'])
def test_installed_import_rejects_invalid_identity(runner, tmp_path, monkeypatch, output):
    monkeypatch.setattr(
        runner, "run_command", lambda args, cwd, env, timeout: runner.CommandResult(args, cwd, 0, output, "")
    )
    with pytest.raises(runner.TestFailure, match="invalid installed QMHub import identity"):
        runner.validate_installed_qmhub({}, tmp_path)


@pytest.mark.parametrize("failure", [OSError("unavailable interpreter"), subprocess.TimeoutExpired(["python"], 30)])
def test_installed_import_wraps_probe_failure(runner, tmp_path, monkeypatch, failure):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(runner, "run_command", fail)
    with pytest.raises(runner.TestFailure, match="installed QMHub import check failed") as caught:
        runner.validate_installed_qmhub({}, tmp_path)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("source", ["checkout", "installed"])
def test_source_option(runner, monkeypatch, source):
    args = ["run_amber_qmhub_tests.py", "--mode", "serial"]
    if source == "installed":
        args += ["--qmhub-source", "installed"]
    monkeypatch.setattr(sys, "argv", args)
    assert runner.parse_args().qmhub_source == source
