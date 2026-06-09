"""
Unit and regression test for the qmhub package.
"""

# Import package, test suite, and other packages as needed
import numpy as np
import qmhub
import pytest
import sys

from qmhub.utils.darray import DependArray
from qmhub.qmtools.qchem import QChem


def _clear_qchem_thread_env(monkeypatch):
    for name in (
        "QCTHREADS",
        "OMP_NUM_THREADS",
        "SLURM_CPUS_PER_TASK",
        "NCPUS",
        "PBS_NP",
        "SLURM_NTASKS",
        "PE_ENV",
        "CRAYPE_VERSION",
        "CRAY_CPU_TARGET",
        "CRAY_LD_LIBRARY_PATH",
        "OMP_PLACES",
        "OMP_PROC_BIND",
    ):
        monkeypatch.delenv(name, raising=False)


def _make_qchem(tmp_path):
    return QChem(
        qm_positions=DependArray(np.zeros((3, 1))),
        qm_elements=DependArray(np.array([1])),
        mm_positions=DependArray(np.zeros((3, 0))),
        mm_charges=DependArray(np.array([])),
        charge=0,
        mult=1,
        cwd=tmp_path,
    )


def test_qmhub_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qmhub" in sys.modules


def test_depend_array_operators():
    a = DependArray([1.0, 2.0, 4.0])
    b = DependArray([2.0, 2.0, 2.0])

    assert np.allclose(np.asarray(a / 2), [0.5, 1.0, 2.0])
    assert np.allclose(np.asarray(2 / a), [2.0, 1.0, 0.5])
    assert np.allclose(np.asarray(a + b), [3.0, 4.0, 6.0])
    assert np.allclose(np.asarray(a * b), [2.0, 4.0, 8.0])
    assert np.array_equal(np.asarray(a > 1), [False, True, True])

    a /= 2
    assert np.allclose(np.asarray(a), [0.5, 1.0, 2.0])


def test_depend_array_indexing_and_cache_invalidation():
    source = DependArray([1.0, 2.0, 4.0])
    dependent = DependArray(func=lambda array: np.asarray(array) * 2, dependencies=[source])

    assert source[1] == 2.0
    assert np.allclose(np.asarray(dependent), [2.0, 4.0, 8.0])

    source[1] = 3.0

    assert np.allclose(np.asarray(dependent), [2.0, 6.0, 8.0])


def test_qchem_cmdline_defaults_to_one_thread(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)

    qchem = _make_qchem(tmp_path)

    assert f"cd {tmp_path}; " in qchem.cmdline
    assert "QCTHREADS=1 OMP_NUM_THREADS=1 qchem -nt 1" in qchem.cmdline


def test_qchem_cmdline_uses_openmp_thread_count(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    qchem = _make_qchem(tmp_path)

    assert "QCTHREADS=4 OMP_NUM_THREADS=4 qchem -nt 4" in qchem.cmdline


def test_qchem_cmdline_uses_slurm_cpu_count(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    qchem = _make_qchem(tmp_path)

    assert "QCTHREADS=8 OMP_NUM_THREADS=8 qchem -nt 8" in qchem.cmdline


def test_qchem_cmdline_adds_cray_openmp_defaults(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("PE_ENV", "CRAY")
    monkeypatch.setenv("NCPUS", "16")

    qchem = _make_qchem(tmp_path)

    assert "QCTHREADS=16 OMP_NUM_THREADS=16" in qchem.cmdline
    assert "OMP_PLACES=cores" in qchem.cmdline
    assert "OMP_PROC_BIND=close" in qchem.cmdline
    assert "qchem -nt 16" in qchem.cmdline


def test_qchem_cmdline_preserves_user_cray_openmp_settings(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("CRAYPE_VERSION", "1")
    monkeypatch.setenv("NCPUS", "4")
    monkeypatch.setenv("OMP_PLACES", "threads")
    monkeypatch.setenv("OMP_PROC_BIND", "spread")

    qchem = _make_qchem(tmp_path)

    assert "QCTHREADS=4 OMP_NUM_THREADS=4 qchem -nt 4" in qchem.cmdline
    assert "OMP_PLACES=cores" not in qchem.cmdline
    assert "OMP_PROC_BIND=close" not in qchem.cmdline
