"""Optional MLPUI engine discovery without installing or evaluating a model."""

import importlib
from pathlib import Path
import subprocess
import sys
import types

import pytest


def _is_mlpui_module(name):
    return name == "qmhub.qmtools.mlpui" or name == "mlpui" or name.startswith("mlpui.")


@pytest.fixture
def isolated_mlpui():
    import qmhub.qmtools as qmtools

    # Cached adapters could bypass dependency checks; restore both import caches
    # and the parent's module attribute so these doubles cannot leak to other tests.
    saved_modules = {name: module for name, module in sys.modules.copy().items()
                     if _is_mlpui_module(name)}
    missing = object()
    saved_attribute = qmtools.__dict__.pop("mlpui", missing)
    for name in saved_modules:
        del sys.modules[name]
    try:
        yield qmtools
    finally:
        for name in list(sys.modules):
            if _is_mlpui_module(name):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        qmtools.__dict__.pop("mlpui", None)
        if saved_attribute is not missing:
            qmtools.mlpui = saved_attribute


def test_package_import_does_not_load_optional_mlpui():
    # A fresh interpreter exercises the first package import even when pytest
    # has already imported QMHub during collection of other test modules.
    script = """
import importlib.abc
import sys

class BlockMLPUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlpui' or fullname.startswith('mlpui.'):
            raise AssertionError('Unexpected optional dependency import: ' + fullname)

sys.meta_path.insert(0, BlockMLPUI())
import qmhub
import qmhub.qmtools
assert 'qmhub.qmtools.mlpui' not in sys.modules
assert not any(name == 'mlpui' or name.startswith('mlpui.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mlpui_factory_engine_receives_arguments(isolated_mlpui, tmp_path):
    class FakeEngine:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    factory_calls = []

    def make_engine_class():
        factory_calls.append(True)
        return FakeEngine

    package = types.ModuleType("mlpui")
    package.__path__ = []
    potentials = types.ModuleType("mlpui.potentials")
    potentials.__path__ = []
    adapter = types.ModuleType("mlpui.potentials.qmhub")
    adapter.make_engine_class = make_engine_class
    package.potentials = potentials
    potentials.qmhub = adapter
    sys.modules.update({"mlpui": package, "mlpui.potentials": potentials,
                        "mlpui.potentials.qmhub": adapter})

    positions = object()
    options = {"spec": "/path/to/potential.json", "threads": "8"}
    kwargs = {"charge": -2, "mult": 1, "cwd": tmp_path, "options": options}
    engine = isolated_mlpui.QM.create("mlpui", positions, **kwargs)

    assert isinstance(engine, FakeEngine)
    assert engine.args == (positions,)
    assert engine.kwargs == kwargs
    assert engine.kwargs["options"] is options
    module = importlib.import_module("qmhub.qmtools.mlpui")
    assert module.MLPUI is module.MLPUIEngine is FakeEngine
    assert len(factory_calls) == 1


def test_missing_mlpui_dependency_keeps_import_cause(isolated_mlpui):
    # A blocked import stays deterministic even if MLPUI is installed locally.
    sys.modules["mlpui"] = None
    with pytest.raises(ImportError, match="QMHub engine 'mlpui' requires the mlpui package") as error:
        isolated_mlpui.QM.create("mlpui")

    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert "mlpui" in str(error.value.__cause__)
