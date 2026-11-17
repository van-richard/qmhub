"""Driver-coordinate gradients through QMMM's real exchange boundaries."""

import os
import queue
import threading

import numpy as np
import pytest

from qmhub import QMMM
from qmhub.utils.darray import DependArray
from qmhub.utils.dlist import DependList


N_QM = 2
POSITIONS = np.array([
    [2.0, 3.0, 6.0, -2.0],
    [0.5, -1.0, 2.0, 4.0],
    [1.0, 0.0, -1.0, 2.0],
])
SECOND_POSITIONS = np.array([
    [-1.0, 2.0, 4.5, -3.0],
    [2.0, 1.0, -2.0, 3.0],
    [0.5, 2.5, 1.0, -0.5],
])
CHARGES = np.array([0.2, -0.2, 0.3, -0.1])
WEIGHTS = np.array([1.1, 2.3, 0.7, 1.4])
BIAS = np.array([
    [0.1, -0.3, 0.2, 0.5],
    [-0.4, 0.2, 0.3, -0.1],
    [0.3, 0.6, -0.2, 0.1],
])


def _analytic_values(positions, step):
    # An origin-dependent energy makes the centroid chain rule observable
    # without PME, model downloads, or an external QM executable.
    energy = np.sum(0.5 * WEIGHTS * positions**2 + BIAS * positions) + 0.125 * step
    gradient = WEIGHTS * positions + BIAS
    return float(energy), gradient


def _wrapped_positions(positions):
    return positions - positions[:, :N_QM].mean(axis=1, keepdims=True)


def _expected_driver_result(positions, step):
    def energy_at(coordinates):
        return _analytic_values(_wrapped_positions(coordinates), step)[0]

    # Differentiate energy in the original driver's coordinates independently
    # of System.map_energy_gradient, including the induced centroid movement.
    gradient = np.zeros_like(positions)
    displacement = 1e-4
    for axis, atom in np.ndindex(positions.shape):
        plus, minus = positions.copy(), positions.copy()
        plus[axis, atom] += displacement
        minus[axis, atom] -= displacement
        gradient[axis, atom] = (energy_at(plus) - energy_at(minus)) / (2 * displacement)
    return energy_at(positions), gradient


class AnalyticEngine:
    def __init__(self, qmmm):
        self.calls = []
        cache = DependList(
            func=self._evaluate,
            dependencies=[qmmm.system.atoms.positions, qmmm.simulation.step],
        )
        self.energy = DependArray(func=lambda values: values[0], dependencies=[cache])
        self.energy_gradient = DependArray(func=lambda values: values[1], dependencies=[cache])

    def _evaluate(self, positions, step):
        coordinates = np.array(positions, copy=True)
        step_value = int(step.item())
        energy, gradient = _analytic_values(coordinates, step_value)
        self.calls.append((step_value, coordinates, gradient.copy()))
        return energy, gradient


def _attach_engine(qmmm):
    engine = AnalyticEngine(qmmm)
    qmmm.simulation.add_engine("engine", engine)
    return engine


def _write_exchange(path, mode, positions, step):
    if mode == "text":
        with path.open("w") as stream:
            stream.write(f"{N_QM} {positions.shape[1] - N_QM} 0 1 {step}\n")
            for atom in range(N_QM):
                x, y, z = positions[:, atom]
                stream.write(f"{x} {y} {z} {CHARGES[atom]} 6\n")
            for atom in range(N_QM, positions.shape[1]):
                x, y, z = positions[:, atom]
                stream.write(f"{x} {y} {z} {CHARGES[atom]}\n")
            np.savetxt(stream, np.zeros((3, 3)))
        return

    qm_dtype = np.dtype([
        ("x", "f8"), ("y", "f8"), ("z", "f8"), ("charge", "f8"), ("element", "i4"),
    ])
    qm = np.zeros(N_QM, dtype=qm_dtype)
    for axis, name in enumerate(("x", "y", "z")):
        qm[name] = positions[axis, :N_QM]
    qm["charge"] = CHARGES[:N_QM]
    qm["element"] = 6
    with path.open("wb") as stream:
        np.asarray([N_QM, positions.shape[1] - N_QM, 0, 1, step], dtype="i4").tofile(stream)
        qm.tofile(stream)
        np.column_stack([positions[:, N_QM:].T, CHARGES[N_QM:]]).tofile(stream)
        np.zeros((3, 3), dtype="f8").tofile(stream)


def _read_result(path, mode):
    if mode == "text":
        lines = path.read_text().splitlines()
        return float(lines[0]), np.loadtxt(lines[1:]).T
    values = np.fromfile(path, dtype="f8")
    return values[0], values[1:].reshape(-1, 3).T


@pytest.mark.parametrize("mode", ["text", "bin"])
def test_qmmm_file_outputs_map_gradients_without_mutating_cached_results(tmp_path, mode):
    input_path = tmp_path / f"input.{mode}"
    step = 7
    _write_exchange(input_path, mode, POSITIONS, step)
    qmmm = QMMM(mode)
    qmmm.setup_simulation()
    qmmm.load_system(input_path)
    engine = _attach_engine(qmmm)
    assert not engine.calls

    qmmm.return_results()
    expected_energy, expected_gradient = _expected_driver_result(POSITIONS, step)
    energy, gradient = _read_result(input_path.with_suffix(".out"), mode)
    assert qmmm.simulation.energy.item() == expected_energy
    expected_output_energy = float(f"{expected_energy:22.14e}") if mode == "text" else expected_energy
    assert energy == expected_output_energy
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-8)
    assert len(engine.calls) == 1

    raw_gradient = _analytic_values(_wrapped_positions(POSITIONS), step)[1]
    assert np.max(np.abs(raw_gradient - gradient)) > 1e-2
    np.testing.assert_array_equal(qmmm.simulation.energy_gradient, raw_gradient)
    np.testing.assert_array_equal(engine.energy_gradient, raw_gradient)

    # Repeated output must reuse the raw cached evaluation and map a fresh
    # copy; it must neither accumulate corrections nor launch the engine again.
    repeated_path = tmp_path / "repeated.out"
    qmmm.return_results(repeated_path)
    repeated_energy, repeated_gradient = _read_result(repeated_path, mode)
    assert repeated_energy == energy
    np.testing.assert_array_equal(repeated_gradient, gradient)
    np.testing.assert_array_equal(engine.energy_gradient, raw_gradient)
    assert len(engine.calls) == 1


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Named FIFO exchange requires POSIX mkfifo.")
def test_qmmm_fifo_maps_each_frame_lazily_with_one_engine_evaluation(tmp_path):
    input_path = tmp_path / "exchange.fifo"
    output_path = input_path.with_suffix(".out")
    os.mkfifo(input_path)
    os.mkfifo(output_path)
    frames = [(7, POSITIONS), (11, SECOND_POSITIONS)]
    ready = threading.Event()
    errors = queue.Queue()
    shared = {}
    results = []

    def serve():
        qmmm = QMMM("fifo")
        qmmm.setup_simulation()
        shared["qmmm"] = qmmm
        try:
            qmmm.load_system(input_path)
            engine = _attach_engine(qmmm)
            shared["engine"] = engine
            # The header establishes atom counts; coordinates arrive later,
            # inside IOFifo.return_results, so gradient evaluation must wait.
            np.testing.assert_array_equal(qmmm.system.atoms.positions, np.zeros_like(POSITIONS))
            assert not engine.calls
            ready.set()
            qmmm.return_results()
        except BaseException as exc:
            errors.put(exc)
        finally:
            ready.set()
            for name in ("_fin", "_fout"):
                stream = getattr(qmmm.io, name, None)
                if stream is not None:
                    stream.close()

    def drive():
        output = None
        try:
            with input_path.open("wb") as stream:
                stream.write(np.asarray([4, N_QM, 0, 1, 0], dtype="i4").tobytes())
                stream.flush()
                assert ready.wait(timeout=5), "QMMM did not finish reading the FIFO header"
                stream.write(CHARGES.astype("f8").tobytes())
                stream.write(np.asarray([6, 6], dtype="i4").tobytes())
                for step, positions in frames:
                    stream.write(np.asarray([step], dtype="i4").tobytes())
                    stream.write(positions.astype("f8").tobytes())
                    stream.flush()
                    if output is None:
                        output = output_path.open("rb")
                    packet = output.read((1 + positions.size) * 8)
                    assert len(packet) == (1 + positions.size) * 8
                    values = np.frombuffer(packet, dtype="f8")
                    results.append((values[0], values[1:].reshape(positions.shape[1], 3).T))
        except BaseException as exc:
            errors.put(exc)
        finally:
            if output is not None:
                output.close()

    # Bounded daemon threads keep a broken exchange from hanging pytest.
    server = threading.Thread(target=serve, daemon=True)
    driver = threading.Thread(target=drive, daemon=True)
    server.start()
    driver.start()
    server.join(timeout=10)
    driver.join(timeout=10)
    assert not server.is_alive(), "QMMM FIFO exchange did not finish"
    assert not driver.is_alive(), "FIFO driver did not finish"
    if not errors.empty():
        raise errors.get()

    assert len(results) == len(frames)
    engine = shared["engine"]
    assert len(engine.calls) == len(frames)
    for (step, positions), (energy, gradient), call in zip(frames, results, engine.calls):
        expected_energy, expected_gradient = _expected_driver_result(positions, step)
        assert energy == expected_energy
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-8)
        assert call[0] == step
        np.testing.assert_array_equal(call[1], _wrapped_positions(positions))

    assert shared["qmmm"].simulation.step.item() == frames[-1][0]
    expected_raw = _analytic_values(_wrapped_positions(SECOND_POSITIONS), frames[-1][0])[1]
    np.testing.assert_array_equal(shared["qmmm"].simulation.energy_gradient, expected_raw)
    np.testing.assert_array_equal(engine.energy_gradient, expected_raw)
    assert len(engine.calls) == len(frames)
