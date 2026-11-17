"""Coordinate-chain-rule checks without external QM programs or trained models."""

from types import SimpleNamespace

import numpy as np
import pytest

from qmhub.model import Model
from qmhub.qmhub import QMMM
from qmhub.system import System
from qmhub.units import CODATA08_BOHR_TO_A
from qmhub.utils.darray import DependArray


def _system(coordinates, n_qm, cell=None):
    system = System(coordinates.shape[1], n_qm, qm_charge=0, qm_mult=1)
    system.atoms.positions[:] = coordinates
    if cell is not None:
        system.cell_basis[:] = np.eye(3) * cell
    return system


def _origin_dependent_energy(coordinates):
    # Different atom/axis coefficients make the absolute grid origin matter,
    # while retaining an independent, inexpensive analytic derivative.
    coefficient = 0.1 + np.arange(coordinates.size).reshape(coordinates.shape) / 37.0
    linear = np.array([0.4, -0.3, 0.7])[:, None]
    energy = np.sum(0.5 * coefficient * coordinates**2 + linear * coordinates)
    return float(energy), coefficient * coordinates + linear


def _wrapped_evaluation(coordinates, n_qm, cell):
    system = _system(coordinates, n_qm, cell)
    system.wrap_positions()
    energy, gradient = _origin_dependent_energy(np.array(system.atoms.positions))
    return system, energy, gradient


def _finite_difference(energy, coordinates, step=1e-5):
    gradient = np.empty_like(coordinates)
    for axis, atom in np.ndindex(coordinates.shape):
        plus, minus = coordinates.copy(), coordinates.copy()
        plus[axis, atom] += step
        minus[axis, atom] -= step
        gradient[axis, atom] = (energy(plus) - energy(minus)) / (2 * step)
    return gradient


@pytest.mark.parametrize("cell", [None, 24.0], ids=["nonperiodic", "periodic"])
@pytest.mark.parametrize("translated", [False, True], ids=["zero-centroid", "shifted-centroid"])
def test_mapping_matches_origin_dependent_energy_finite_difference(cell, translated):
    coordinates = np.array([
        [-1.25, 1.25, 14.1, -3.2],
        [0.5, -0.5, 4.2, 6.3],
        [-0.75, 0.75, -13.3, 2.1],
    ])
    if translated:
        coordinates += np.array([2.0, -3.0, 0.75])[:, None]
    system, _, raw = _wrapped_evaluation(coordinates, 2, cell)
    original = raw.copy()
    raw.flags.writeable = False

    mapped = system.map_energy_gradient(raw)
    numerical = _finite_difference(
        lambda positions: _wrapped_evaluation(positions, 2, cell)[1], coordinates)

    assert np.max(np.abs(original.sum(axis=1))) > 0.1
    np.testing.assert_allclose(mapped, numerical, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(mapped.sum(axis=1), 0.0, rtol=0.0, atol=3e-15)
    assert not np.shares_memory(mapped, raw)
    np.testing.assert_array_equal(raw, original)
    # The centering derivative affects QM columns only, including when the
    # present QM centroid is already exactly zero.
    assert mapped[:, 2:].tobytes() == original[:, 2:].tobytes()


@pytest.mark.parametrize("n_atoms,n_qm", [(1, 1), (3, 1), (3, 3)],
                         ids=["single-qm-only", "one-qm-with-mm", "qm-only"])
def test_mapping_handles_one_qm_and_no_mm(n_atoms, n_qm):
    coordinates = np.array([[0.3, 1.4, -0.8], [-0.6, 2.1, 1.2], [1.3, -0.7, 0.4]])[:, :n_atoms]
    system, _, raw = _wrapped_evaluation(coordinates, n_qm, None)
    mapped = system.map_energy_gradient(DependArray(raw.copy()))
    numerical = _finite_difference(
        lambda positions: _wrapped_evaluation(positions, n_qm, None)[1], coordinates)

    assert mapped.shape == coordinates.shape
    np.testing.assert_allclose(mapped, numerical, rtol=1e-8, atol=1e-9)
    if n_atoms == n_qm == 1:
        np.testing.assert_array_equal(mapped, np.zeros((3, 1)))


# Existing self-pair products are masked after evaluation; this control checks
# the coordinate contract rather than that unrelated warning policy.
@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply:RuntimeWarning")
def test_direct_model_keeps_fixed_origin_gradient():
    coordinates = np.array([[0.3, 1.4, 3.7], [-0.6, 0.1, 1.2], [1.3, -0.7, 2.4]])
    system = _system(coordinates, 2)
    model = Model(system.qm.atoms.positions, system.atoms.positions,
                  system.qm.atoms.charges, system.atoms.charges,
                  system.cell_basis, qm_total_charge=0, cutoff=8.0, pbc=False)
    energy = DependArray(func=lambda positions: 0.5 * np.square(positions).sum(),
                         dependencies=[system.qm.atoms.positions])
    gradient = DependArray(func=lambda positions: positions * CODATA08_BOHR_TO_A,
                           dependencies=[system.qm.atoms.positions])
    mm_esp = DependArray(np.zeros((4, len(model.elec.near_field.charges))))
    model.get_result("fixed_origin", energy, gradient, mm_esp)
    actual = np.array(model.fixed_origin.energy_gradient)

    def evaluate(positions):
        # No wrapping: the Model API differentiates the coordinates supplied
        # by its caller, so applying the input-frame map here would be wrong.
        system.atoms.positions[:] = positions
        return model.fixed_origin.energy.item()

    numerical = _finite_difference(evaluate, coordinates) * CODATA08_BOHR_TO_A
    np.testing.assert_allclose(actual, numerical, rtol=1e-9, atol=1e-10)
    assert np.max(np.abs(actual.sum(axis=1))) > 0.1
    np.testing.assert_array_equal(actual[:, 2:], np.zeros((3, 1)))


def _synthetic_group(frame, bases, energy):
    # Synthetic complete model results isolate MD/MTS ordering from the
    # electronic model. Two independently cached terms represent each group.
    terms = [DependArray(func=lambda current, base=base: base * (current.item() + 1),
                         dependencies=[frame]) for base in bases]
    return SimpleNamespace(
        energy=DependArray(np.array(energy)),
        energy_gradient=DependArray(func=lambda *values: sum(values), dependencies=terms),
        terms=terms,
    )


@pytest.mark.parametrize("protocol", ["md", "mts"])
@pytest.mark.parametrize("scaling", [None, 1.75], ids=["unscaled", "scaled"])
def test_driver_maps_lazy_aggregated_gradient_once_per_frame(protocol, scaling):
    qmmm = QMMM("text")
    qmmm.setup_simulation(protocol=protocol, nrespa=3, scaling_factor=scaling)
    qmmm.system = _system(np.zeros((3, 4)), 2)
    qmmm.system.wrap_positions()
    frame = DependArray(np.array(0))
    # Binary-exact coefficients keep aggregation checks independent of harmless
    # floating-point changes in the ordering of scaling and summation.
    high_bases = [np.arange(12, dtype=float).reshape(3, 4) / 8.0 + 0.25,
                  np.arange(12, dtype=float).reshape(3, 4) / 16.0 - 0.5]
    low_bases = [np.arange(12, dtype=float).reshape(3, 4) / 32.0 + 0.125,
                 np.arange(12, dtype=float).reshape(3, 4) / 64.0 + 0.375]
    high = _synthetic_group(frame, high_bases, 4.0)
    low = _synthetic_group(frame, low_bases, 1.5)
    qmmm.simulation.add_engine("engine", high)
    if protocol == "mts":
        qmmm.simulation.add_engine("engine2", low)

    calls = []
    original_map = qmmm.system.map_energy_gradient

    def record_mapping(raw):
        calls.append(np.array(raw))
        return original_map(raw)

    qmmm.system.map_energy_gradient = record_mapping
    records = []

    def consume(energy, mapped, output):
        assert output == "requested-output"
        assert isinstance(mapped, DependArray)
        assert not mapped._cache_valid
        assert calls == []
        # The first geometry is read after return_results receives the lazy
        # arrays, as in FIFO. Later frames invalidate the same mapped object.
        for index, step in enumerate([0, 3, 5, 6, 8]):
            frame[()] = index + 1
            qmmm.simulation.step[()] = step
            assert not mapped._cache_valid
            high_value = sum(high_bases) * (index + 2)
            low_value = sum(low_bases) * (index + 2)
            if protocol == "md" or step < 3:
                expected_raw = high_value
                expected_energy = 4.0
            elif (step + 1) % 3 == 0:
                expected_raw = low_value + 3 * (high_value - low_value)
                expected_energy = 4.0
            else:
                expected_raw = low_value
                expected_energy = 0.0
            factor = 1.0 if scaling is None else scaling
            expected_raw = expected_raw * factor
            expected_energy *= factor
            high_before = np.array(high.energy_gradient)
            low_before = np.array(low.energy_gradient) if protocol == "mts" else None
            simulation_before = np.array(qmmm.simulation.energy_gradient)
            result = np.array(mapped)
            np.testing.assert_allclose(calls[-1], expected_raw, rtol=1e-15, atol=1e-15)
            assert len(calls) == index + 1
            np.testing.assert_allclose(result.sum(axis=1), 0.0, rtol=0, atol=1e-14)
            assert result[:, 2:].tobytes() == simulation_before[:, 2:].tobytes()
            np.testing.assert_allclose(energy.item(), expected_energy, rtol=0, atol=0)
            # A cached second read must neither remap nor mutate the raw
            # simulation or either engine group's cached gradient.
            np.testing.assert_array_equal(np.array(mapped), result)
            assert len(calls) == index + 1
            np.testing.assert_allclose(np.asarray(qmmm.simulation.energy_gradient), expected_raw,
                                       rtol=1e-15, atol=1e-15)
            np.testing.assert_array_equal(np.asarray(high.energy_gradient), high_before)
            np.testing.assert_array_equal(high_before, high_value)
            if protocol == "mts":
                np.testing.assert_array_equal(np.asarray(low.energy_gradient), low_before)
                np.testing.assert_array_equal(low_before, low_value)
            records.append(result)

    qmmm.io = SimpleNamespace(return_results=consume)
    qmmm.return_results("requested-output")
    assert len(records) == len(calls) == 5
