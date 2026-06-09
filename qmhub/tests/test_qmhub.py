"""
Unit and regression test for the qmhub package.
"""

# Import package, test suite, and other packages as needed
import os
import numpy as np
import qmhub
import pytest
import sys

from qmhub.utils.darray import DependArray
from qmhub.utils.sys import get_nproc
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


def _make_qchem(tmp_path, n_mm=0):
    return QChem(
        qm_positions=DependArray(np.zeros((3, 1))),
        qm_elements=DependArray(np.array([1])),
        mm_positions=DependArray(np.zeros((3, n_mm))),
        mm_charges=DependArray(np.ones(n_mm)),
        charge=0,
        mult=1,
        cwd=tmp_path,
    )


def _write_binary_mm_esp(tmp_path, output):
    potential = np.array([1.0, 2.0])
    field = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ])

    potential_path = tmp_path.joinpath(output[0])
    field_path = tmp_path.joinpath(output[1])
    potential_path.parent.mkdir(parents=True, exist_ok=True)
    field_path.parent.mkdir(parents=True, exist_ok=True)

    potential.astype("f8").tofile(potential_path)
    field.astype("f8").tofile(field_path)

    return potential, field


def _write_binary_output(tmp_path, output, values):
    output_path = tmp_path.joinpath(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="f8").tofile(output_path)
    return output_path


def _assert_mm_esp(qchem, potential, field, **kwargs):
    mm_esp = qchem._get_mm_esp(**kwargs)

    assert mm_esp.shape == (4, len(potential))
    assert np.allclose(mm_esp[0], potential)
    assert np.allclose(mm_esp[1:], -field.T)


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


def test_get_nproc_sanitizes_empty_openmp_threads(monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "")

    assert get_nproc() == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_get_nproc_uses_scheduler_threads_when_openmp_is_empty(monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    assert get_nproc() == 8
    assert os.environ["OMP_NUM_THREADS"] == "8"


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


def test_qchem_cache_reports_command_failure_with_logs(tmp_path, monkeypatch):
    qchem = _make_qchem(tmp_path)
    tmp_path.joinpath("qchem_run.log").write_text("launcher failed\n")
    tmp_path.joinpath("qchem.out").write_text("qchem stopped before writing save files\n")
    monkeypatch.setattr("qmhub.qmtools.qmbase.run_cmdline", lambda cmdline: 7)

    with pytest.raises(RuntimeError) as error:
        qchem._get_qm_cache()

    message = str(error.value)
    assert "Q-Chem command failed with exit code 7." in message
    assert "qchem_run.log" in message
    assert "launcher failed" in message
    assert "qchem.out" in message
    assert "qchem stopped before writing save files" in message


def test_qchem_energy_reads_valid_binary_and_removes(tmp_path):
    qchem = _make_qchem(tmp_path)
    output_path = _write_binary_output(tmp_path, "save/99.0", [0.0, -12.5])

    assert np.isclose(qchem._get_qm_energy(), -12.5)
    assert not output_path.exists()


def test_qchem_energy_reports_missing_binary(tmp_path):
    qchem = _make_qchem(tmp_path)

    with pytest.raises(FileNotFoundError) as error:
        qchem._get_qm_energy()

    message = str(error.value)
    assert "Q-Chem energy binary output was not produced" in message
    assert "save/99.0" in message


def test_qchem_energy_reports_incomplete_binary_without_removing(tmp_path):
    qchem = _make_qchem(tmp_path)
    output_path = _write_binary_output(tmp_path, "save/99.0", [0.0])

    with pytest.raises(ValueError) as error:
        qchem._get_qm_energy()

    message = str(error.value)
    assert "Q-Chem energy binary output is incomplete" in message
    assert "expected at least 16 bytes" in message
    assert output_path.exists()


def test_qchem_gradient_reads_valid_binary_and_removes(tmp_path):
    qchem = _make_qchem(tmp_path)
    output_path = _write_binary_output(tmp_path, "save/131.0", [1.0, 2.0, 3.0])

    assert np.allclose(qchem._get_qm_energy_gradient(), [[1.0], [2.0], [3.0]])
    assert not output_path.exists()


def test_qchem_gradient_reports_incomplete_binary_without_removing(tmp_path):
    qchem = _make_qchem(tmp_path)
    output_path = _write_binary_output(tmp_path, "save/131.0", [1.0])

    with pytest.raises(ValueError) as error:
        qchem._get_qm_energy_gradient()

    message = str(error.value)
    assert "Q-Chem energy gradient binary output is incomplete" in message
    assert "expected at least 24 bytes" in message
    assert output_path.exists()


def test_qchem_mm_esp_reads_existing_binary_pair(tmp_path):
    qchem = _make_qchem(tmp_path, n_mm=2)
    output = ("save/1521.0", "save/329.0")
    potential, field = _write_binary_mm_esp(tmp_path, output)

    _assert_mm_esp(qchem, potential, field)

    assert not tmp_path.joinpath(output[0]).exists()
    assert not tmp_path.joinpath(output[1]).exists()


def test_qchem_mm_esp_reads_new_binary_pair(tmp_path):
    qchem = _make_qchem(tmp_path, n_mm=2)
    output = ("save/5001.0", "save/5002.0")
    potential, field = _write_binary_mm_esp(tmp_path, output)

    _assert_mm_esp(qchem, potential, field)

    assert not tmp_path.joinpath(output[0]).exists()
    assert not tmp_path.joinpath(output[1]).exists()


def test_qchem_mm_esp_reads_explicit_binary_output(tmp_path):
    qchem = _make_qchem(tmp_path, n_mm=2)
    output = ("custom/potential.bin", "custom/field.bin")
    potential, field = _write_binary_mm_esp(tmp_path, output)

    _assert_mm_esp(qchem, potential, field, output=output)

    assert not tmp_path.joinpath(output[0]).exists()
    assert not tmp_path.joinpath(output[1]).exists()


def test_qchem_mm_esp_ignores_stale_text_outputs(tmp_path):
    qchem = _make_qchem(tmp_path, n_mm=2)

    # These text files can survive from a previous run, so binary outputs must
    # still be required for automatic Q-Chem MM ESP parsing.
    np.savetxt(tmp_path.joinpath("esp.dat"), np.array([1.0, 2.0]))
    np.savetxt(tmp_path.joinpath("efield.dat"), np.zeros((2, 3)))
    tmp_path.joinpath("plot.esp").write_text("0.0 0.0 0.0 1.0\n1.0 1.0 1.0 2.0\n")

    with pytest.raises(FileNotFoundError):
        qchem._get_mm_esp()

    assert tmp_path.joinpath("esp.dat").exists()
    assert tmp_path.joinpath("efield.dat").exists()
    assert tmp_path.joinpath("plot.esp").exists()


def test_qchem_mm_esp_reports_missing_sources(tmp_path):
    qchem = _make_qchem(tmp_path, n_mm=2)
    save_path = tmp_path.joinpath("save")
    save_path.mkdir()
    tmp_path.joinpath("save/123.0").write_bytes(b"not enough data")

    with pytest.raises(FileNotFoundError) as error:
        qchem._get_mm_esp()

    message = str(error.value)
    assert "Could not find valid Q-Chem MM ESP output." in message
    assert "save/1521.0" in message
    assert "save/5001.0" in message
    assert "123.0" in message
