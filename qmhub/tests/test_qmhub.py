"""
Unit and regression test for the qmhub package.
"""

# Import package, test suite, and other packages as needed
import io
import importlib
import importlib.util
import os
from pathlib import Path
import threading
import time
import warnings

import numpy as np
import qmhub
import pytest
import sys

from qmhub.electools.distance import get_dij_gradient
from qmhub.electools.distance import get_dij_inverse
from qmhub.electools.elec_near import ElecNear
from qmhub.electools.ewald import Ewald as DirectEwald
from qmhub.iotools.bin import IOBin
from qmhub.iotools.fifo import read_fifo_scalar
from qmhub.iotools.fifo import IOFifo
from qmhub.iotools.text import IOText
from qmhub.utils.darray import DependArray
from qmhub.utils.sys import get_nthreads
from qmhub.qmtools.orca import ORCA
from qmhub.qmtools.qchem import QChem


def _clear_qchem_thread_env(monkeypatch):
    for name in (
        "QCTHREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
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


def _make_orca(tmp_path, n_mm=0):
    return ORCA(
        qm_positions=DependArray(np.zeros((3, 1))),
        qm_elements=DependArray(np.array([1])),
        mm_positions=DependArray(np.zeros((3, n_mm))),
        mm_charges=DependArray(np.ones(n_mm)),
        charge=0,
        mult=1,
        cwd=tmp_path,
    )


def _load_amber_smoke_runner():
    path = Path(__file__).resolve().parents[2].joinpath(
        "devtools",
        "amber-qmhub-tests",
        "run_amber_qmhub_tests.py",
    )
    spec = importlib.util.spec_from_file_location("run_amber_qmhub_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_binary_mm_esp(tmp_path, output, potential=None, field=None):
    if potential is None:
        potential = np.array([1.0, 2.0])
    if field is None:
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


def _write_text_exchange(path):
    with open(path, "w") as f:
        f.write("2 2 0 1 5\n")
        f.write("0.0 0.0 0.0 -0.2 6\n")
        f.write("0.5 0.0 0.0 0.1 1\n")
        f.write("1.0 0.0 0.0 0.2\n")
        f.write("1.5 0.0 0.0 -0.1\n")
        np.savetxt(f, np.zeros((3, 3)))


def _write_bin_exchange(path):
    with open(path, "wb") as f:
        np.asarray([2, 2, 0, 1, 5], dtype="i4").tofile(f)
        np.asarray([
            (0.0, 0.0, 0.0, -0.2, 6),
            (0.5, 0.0, 0.0, 0.1, 1),
        ], dtype=[
            ("pos_x", "f8"),
            ("pos_y", "f8"),
            ("pos_z", "f8"),
            ("charge", "f8"),
            ("element", "i4"),
        ]).tofile(f)
        np.asarray([
            (1.0, 0.0, 0.0, 0.2),
            (1.5, 0.0, 0.0, -0.1),
        ], dtype=[
            ("pos_x", "f8"),
            ("pos_y", "f8"),
            ("pos_z", "f8"),
            ("charge", "f8"),
        ]).tofile(f)
        np.zeros((3, 3), dtype="f8").tofile(f)


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _at26_patch_paths():
    patch_dir = _repo_root().joinpath("patches")
    return [
        patch_dir.joinpath("at26_intel", "qmhub_at26.patch"),
        patch_dir.joinpath("at26_gnu", "qmhub_at26_gnu.patch"),
    ]


def _stream_f8_bytes(*arrays):
    return b"".join(np.asarray(array, dtype="f8").tobytes() for array in arrays)


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


def test_depend_array_electrostatic_style_operations():
    tensor = DependArray([[1.0, 0.2], [0.3, 1.5], [0.7, -0.4]])
    charges = DependArray([0.5, -0.25])
    offsets = DependArray([1.0, 2.0, 3.0])

    projected = tensor @ np.linalg.pinv(tensor) @ tensor
    assert np.allclose(np.asarray(projected), np.asarray(tensor))

    assert np.allclose(np.asarray(tensor @ charges), [0.45, -0.225, 0.45])
    assert np.allclose(np.asarray(np.sum(tensor, axis=0)), [2.0, 1.3])
    assert np.allclose(np.asarray(tensor + offsets[:, np.newaxis]), [[2.0, 1.2], [2.3, 3.5], [3.7, 2.6]])

    out = DependArray(np.zeros((3, 2)))
    np.add(tensor, 1.0, out=out)
    assert np.allclose(np.asarray(out), np.asarray(tensor) + 1.0)

    out *= 2.0
    assert np.allclose(np.asarray(out), (np.asarray(tensor) + 1.0) * 2.0)


def test_depend_array_derived_matrix_cache_invalidation():
    source = DependArray([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([1.0, -1.0])
    dependent = DependArray(func=lambda matrix: matrix @ weights, dependencies=[source])

    assert np.allclose(np.asarray(dependent), [-1.0, -1.0])

    source[0, 0] = 5.0

    assert np.allclose(np.asarray(dependent), [3.0, -1.0])


def test_distance_helpers_suppress_expected_zero_self_distance_warnings():
    rij = DependArray(np.zeros((3, 1, 1)))
    dij = DependArray(np.zeros((1, 1)))
    dij_gradient = DependArray(func=get_dij_gradient, dependencies=[rij, dij])
    dij_inverse = DependArray(func=get_dij_inverse, dependencies=[dij])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        gradient = np.asarray(dij_gradient)
        inverse = np.asarray(dij_inverse)

    assert np.all(np.isfinite(gradient))
    assert np.allclose(gradient, 0.0)
    assert np.isinf(inverse[0, 0])


def test_distance_helpers_keep_nonzero_distance_values():
    rij = np.array([[[3.0]], [[4.0]], [[0.0]]])
    dij = np.array([[5.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        gradient = get_dij_gradient(rij, dij)
        inverse = get_dij_inverse(dij)

    assert np.allclose(gradient[:, 0, 0], [0.6, 0.8, 0.0])
    assert np.allclose(inverse, [[0.2]])


def test_near_field_buffered_distances_ignore_masked_self_distance_warnings():
    dij = DependArray(np.array([[0.0, 2.0]]))
    mask = DependArray(
        func=ElecNear._get_near_field_buffered_mask,
        kwargs={"cutoff": 10.0},
        dependencies=[dij],
    )
    dij_min = DependArray(
        func=ElecNear._get_dij_min_buffered,
        dependencies=[dij, mask],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        value = np.asarray(dij_min)

    assert np.allclose(value, [2.0])


def test_direct_ewald_self_exclusions_do_not_warn():
    positions = np.zeros((3, 1))
    cell_basis = np.eye(3) * 20.0
    real_lattice = np.zeros((3, 1))
    recip_lattice = np.array([[1.0], [0.0], [0.0]])
    exclusion = np.array([0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        real = DirectEwald._get_ewald_real_tensor(
            positions,
            positions,
            real_lattice,
            alpha=0.3,
            exclusion=exclusion,
        )
        recip = DirectEwald._get_ewald_recip_tensor(
            positions,
            positions,
            recip_lattice,
            cell_basis,
            alpha=0.3,
            exclusion=exclusion,
        )

    assert np.all(np.isfinite(real))
    assert np.all(np.isfinite(recip))


def test_source_tree_helpmelib_extension_available_when_requested():
    if os.environ.get("QMHUB_REQUIRE_SOURCE_HELPME") != "1":
        pytest.skip("Set QMHUB_REQUIRE_SOURCE_HELPME=1 after building or copying qmhub.helpmelib into the source tree.")

    package_dir = Path(qmhub.__file__).resolve().parent
    extensions = list(package_dir.glob("helpmelib*.so"))

    assert extensions, f"No helpmelib extension found in source package directory: {package_dir}"

    module = importlib.import_module("qmhub.helpmelib")

    assert Path(module.__file__).resolve().parent == package_dir
    assert hasattr(module, "MatrixD")


def test_get_nthreads_sanitizes_empty_openmp_threads(monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "")

    assert get_nthreads() == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_get_nthreads_uses_qcthreads_when_openmp_is_empty(monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "")
    monkeypatch.setenv("QCTHREADS", "8")

    assert get_nthreads() == 8
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_get_nthreads_uses_mkl_threads_as_fallback(monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("MKL_NUM_THREADS", "6")

    assert get_nthreads() == 6


def test_qchem_nproc_uses_openmp_threads(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    qchem = _make_qchem(tmp_path)

    assert qchem.nproc == 4
    assert "QCTHREADS=4 OMP_NUM_THREADS=4 qchem -nt 4" in qchem.cmdline


def test_qchem_nproc_prefers_qcthreads(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("QCTHREADS", "8")

    qchem = _make_qchem(tmp_path)

    assert qchem.nproc == 8
    assert "QCTHREADS=8 OMP_NUM_THREADS=8 qchem -nt 8" in qchem.cmdline


def test_amber_mdout_scanner_allows_benign_error_estimates():
    runner = _load_amber_smoke_runner()

    text = """
 Ewald error estimate: 0.1234E-04
 A V E R A G E S   O V E R
"""

    assert not runner.contains_amber_mdout_failure_marker(text)


def test_amber_mdout_scanner_rejects_sander_bomb():
    runner = _load_amber_smoke_runner()

    assert runner.contains_amber_mdout_failure_marker(" SANDER BOMB in routine foo")


def test_orca_uses_thread_count(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    orca = _make_orca(tmp_path)
    orca.gen_input()

    assert orca.nproc == 4
    assert "%pal nprocs 4 end" in tmp_path.joinpath("orca.inp").read_text()


def test_orca_uses_only_openmp_thread_count(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("QCTHREADS", "8")

    orca = _make_orca(tmp_path)
    orca.gen_input()

    assert orca.nproc == 1
    assert "%pal nprocs 1 end" in tmp_path.joinpath("orca.inp").read_text()


def test_read_fifo_scalar_assigns_to_zero_dimensional_step():
    fin = io.BytesIO(np.asarray([42], dtype="i4").tobytes())
    step = np.asarray(0)

    step[()] = read_fifo_scalar(fin, dtype="i4")

    assert step.item() == 42


def test_text_and_binary_outputs_match_upstream_layout(tmp_path):
    energy = np.asarray(3.25)
    forces = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ])

    text_input = tmp_path.joinpath("exchange_text.txt")
    bin_input = tmp_path.joinpath("exchange_bin.bin")
    _write_text_exchange(text_input)
    _write_bin_exchange(bin_input)

    text_io = IOText()
    bin_io = IOBin()
    text_io.load_system(text_input)
    bin_io.load_system(bin_input)
    text_io.return_results(energy, forces)
    bin_io.return_results(energy, forces)

    text_lines = text_input.with_suffix(".out").read_text().splitlines()
    bin_output = np.fromfile(bin_input.with_suffix(".out"), dtype="f8")

    assert np.isclose(float(text_lines[0]), energy)
    assert np.allclose(np.loadtxt(text_lines[1:]).T, forces)
    assert np.isclose(bin_output[0], energy)
    assert np.allclose(bin_output[1:].reshape(4, 3).T, forces)


def test_fifo_round_trip_writes_energy_and_forces(tmp_path):
    input_fifo = tmp_path.joinpath("exchange.fifo")
    output_fifo = input_fifo.with_suffix(".out")
    os.mkfifo(input_fifo)

    energy = np.asarray(3.25)
    forces = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    output = {}

    def write_driver_input():
        with open(input_fifo, "wb") as f:
            f.write(np.asarray([2, 1, 0, 1, 0], dtype="i4").tobytes())
            f.write(np.asarray([0.1, -0.2], dtype="f8").tobytes())
            f.write(np.asarray([6], dtype="i4").tobytes())
            f.write(np.asarray([7], dtype="i4").tobytes())
            f.write(np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype="f8").tobytes())

    def read_driver_output():
        while not output_fifo.exists():
            time.sleep(0.01)

        with open(output_fifo, "rb") as f:
            output["energy"] = np.frombuffer(f.read(8), dtype="f8")[0]
            output["forces"] = np.frombuffer(f.read(6 * 8), dtype="f8").reshape(3, 2, order="F")

    writer = threading.Thread(target=write_driver_input)
    reader = threading.Thread(target=read_driver_output)
    writer.start()

    fifo_io = IOFifo()
    fifo_io.load_system(input_fifo)

    reader.start()
    fifo_io.return_results(energy, forces)
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert np.isclose(output["energy"], energy)
    assert np.allclose(output["forces"], forces)


def test_at26_patch_variants_match_current_fifo_cell_write():
    for patch in _at26_patch_paths():
        text = patch.read_text()
        assert text.count("write(iunit) (ucell(i,:), i=1,3)") == 2
        assert text.count("write(iunit) ucell(:,i)") == 1
        assert text.count("write(iunit,'(3(e21.15,1x))') ucell(:,i)") == 1


def test_at26_patch_variants_detect_qmhub_extern_namelist():
    for patch in _at26_patch_paths():
        text = patch.read_text()
        assert "! True when qm_theory='EXTERN' is paired with a &qmhub namelist." in text
        assert "logical :: qmhub_extern" in text
        assert "call mpi_bcast(self%qmhub_extern,         1, mpi_logical" in text
        assert "write(6,'(a,l)')     'qmhub_extern                = ', self%qmhub_extern" in text
        assert "qmhub_extern = .false." in text
        assert "call nmlsrc('qmhub',5,ifind)" in text
        assert text.count("rewind 5") >= 2
        assert "qmmm_nml%qmhub_extern = qmmm_nml%qmtheory%EXTERN .and. qmhub_extern" in text
        assert "noqmcutoff = qmmm_nml%qmhub_extern" in text


def test_at26_patch_variants_restore_at23_arithmetic_qm_center():
    for patch in _at26_patch_paths():
        text = patch.read_text()
        assert "! The modification below uses the arithmetic QM center," in text
        assert "! matching AmberTools23 pair-list imaging." in text
        assert "+  use constants, only : zero, one, half, two" in text
        assert "+  xtmp = xtmp * one_nquant" in text
        assert "+  ytmp = ytmp * one_nquant" in text
        assert "+  ztmp = ztmp * one_nquant" in text
        assert "+  frac(1) = atan2" not in text
        assert "+  frac(2) = atan2" not in text
        assert "+  frac(3) = atan2" not in text


def test_fifo_coordinate_packets_match_at23_and_at26_forms_byte_for_byte():
    qmcoords = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    clcoords = np.array([
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0],
        [30.0, 31.0, 32.0],
        [0.1, 0.2, 0.3],
    ])

    at23_loop_packet = _stream_f8_bytes(
        qmcoords[0, :], clcoords[0, :],
        qmcoords[1, :], clcoords[1, :],
        qmcoords[2, :], clcoords[2, :],
    )
    at26_implied_do_packet = _stream_f8_bytes(
        np.concatenate([
            qmcoords[0, :], clcoords[0, :],
            qmcoords[1, :], clcoords[1, :],
            qmcoords[2, :], clcoords[2, :],
        ])
    )

    assert at26_implied_do_packet == at23_loop_packet
    assert np.allclose(
        np.frombuffer(at26_implied_do_packet, dtype="f8"),
        [
            1.0, 2.0, 10.0, 11.0, 12.0,
            3.0, 4.0, 20.0, 21.0, 22.0,
            5.0, 6.0, 30.0, 31.0, 32.0,
        ],
    )


def test_fifo_cell_packets_match_at23_and_at26_forms_byte_for_byte():
    ucell = np.array([
        [10.0, 1.0, 2.0],
        [3.0, 20.0, 4.0],
        [5.0, 6.0, 30.0],
    ])

    at23_packet = _stream_f8_bytes(ucell[0, :], ucell[1, :], ucell[2, :])
    at26_packet = _stream_f8_bytes(np.concatenate([ucell[0, :], ucell[1, :], ucell[2, :]]))
    text_binary_packet = _stream_f8_bytes(ucell[:, 0], ucell[:, 1], ucell[:, 2])

    assert at26_packet == at23_packet
    assert np.allclose(np.frombuffer(at26_packet, dtype="f8"), ucell.reshape(9))
    assert text_binary_packet != at26_packet
    assert np.allclose(np.frombuffer(text_binary_packet, dtype="f8"), ucell.T.reshape(9))


def test_current_fifo_cell_packet_differs_from_text_binary_for_skewed_cell():
    # Amber keeps lattice vectors in ucell columns. Text/binary write ucell(:,i),
    # which QMHub reads as lattice-vector rows; current FIFO writes ucell(i,:).
    ucell = np.array([
        [10.0, 1.0, 2.0],
        [3.0, 20.0, 4.0],
        [5.0, 6.0, 30.0],
    ])

    text_binary_packet = np.concatenate([ucell[:, i] for i in range(3)])
    fifo_packet = np.concatenate([ucell[i, :] for i in range(3)])
    text_binary_basis = text_binary_packet.reshape(3, 3)
    fifo_basis = fifo_packet.reshape(3, 3)

    assert np.allclose(text_binary_basis, ucell.T)
    assert np.allclose(fifo_basis, ucell)
    assert not np.allclose(fifo_basis, text_binary_basis)


def test_pme_uses_openmp_thread_count_when_helpmelib_available(monkeypatch):
    pme_module = pytest.importorskip("qmhub.electools.pme")
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    calls = []

    class DummyPME:
        def __init__(self, *args):
            calls.append(args)

        def add_dependant(self, dependant):
            pass

    monkeypatch.setattr(pme_module, "DependPME", DummyPME)

    pme_module.Ewald(
        qm_positions=DependArray(np.zeros((3, 1))),
        positions=DependArray(np.zeros((3, 1))),
        charges=DependArray(np.ones(1)),
        cell_basis=DependArray(np.eye(3) * 20.0),
    )

    assert calls[0][-1] == 4


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


def test_qchem_cmdline_uses_qcthreads(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("QCTHREADS", "8")

    qchem = _make_qchem(tmp_path)

    assert "QCTHREADS=8 OMP_NUM_THREADS=8 qchem -nt 8" in qchem.cmdline


def test_qchem_cmdline_prefers_qcthreads_over_openmp(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("QCTHREADS", "8")

    qchem = _make_qchem(tmp_path)

    assert "QCTHREADS=8 OMP_NUM_THREADS=8 qchem -nt 8" in qchem.cmdline


def test_qchem_cmdline_adds_cray_openmp_defaults(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("PE_ENV", "CRAY")
    monkeypatch.setenv("OMP_NUM_THREADS", "16")

    qchem = _make_qchem(tmp_path)

    assert "QCTHREADS=16 OMP_NUM_THREADS=16" in qchem.cmdline
    assert "OMP_PLACES=cores" in qchem.cmdline
    assert "OMP_PROC_BIND=close" in qchem.cmdline
    assert "qchem -nt 16" in qchem.cmdline


def test_qchem_cmdline_preserves_user_cray_openmp_settings(tmp_path, monkeypatch):
    _clear_qchem_thread_env(monkeypatch)
    monkeypatch.setenv("CRAYPE_VERSION", "1")
    monkeypatch.setenv("QCTHREADS", "4")
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


def test_qchem_mm_esp_reads_trailing_rows_from_new_binary_pair(tmp_path):
    qchem = _make_qchem(tmp_path, n_mm=3)
    output = ("save/5001.0", "save/5002.0")
    potential = np.array([-100.0, -101.0, 1.0, 2.0, 3.0])
    field = np.array([
        [-10.0, -11.0, -12.0],
        [-13.0, -14.0, -15.0],
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])
    _write_binary_mm_esp(tmp_path, output, potential, field)

    _assert_mm_esp(qchem, potential[-3:], field[-3:])

    assert not tmp_path.joinpath(output[0]).exists()
    assert not tmp_path.joinpath(output[1]).exists()


def test_qchem_mm_esp_rejects_extra_rows_from_existing_binary_pair(tmp_path):
    qchem = _make_qchem(tmp_path, n_mm=3)
    output = ("save/1521.0", "save/329.0")
    potential = np.array([-100.0, -101.0, 1.0, 2.0, 3.0])
    field = np.zeros((5, 3))
    _write_binary_mm_esp(tmp_path, output, potential, field)

    with pytest.raises(FileNotFoundError) as error:
        qchem._get_mm_esp()

    assert "Could not find valid Q-Chem MM ESP output." in str(error.value)
    assert tmp_path.joinpath(output[0]).exists()
    assert tmp_path.joinpath(output[1]).exists()


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
