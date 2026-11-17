"""Validate an installed QMHub/MLPUI environment without external QM or weights.

Run with the target environment's Python. CUDA checks use small analytic inputs;
they do not certify a trained potential or an Amber/Q-Chem execution path.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
import time
import traceback


EXPECTED = {
    "numpy": "2.4.6", "scipy": "1.17.1", "torch": "2.11.0+cu128",
    "mlpui": "0.1.0", "e3nn": "0.4.4", "opt_einsum_fx": "0.1.4",
}
OTHER_DEPENDENCIES = (
    "pytorch-lightning", "torch_geometric", "matplotlib", "pyyaml", "ase",
    "psutil", "omegaconf", "lightning-utilities", "qmhub",
)
IMPORT_NAMES = {"pytorch-lightning": "pytorch_lightning", "pyyaml": "yaml",
                "lightning-utilities": "lightning_utilities"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_versions(source):
    import qmhub

    versions = {name: metadata.version(name) for name in (*EXPECTED, *OTHER_DEPENDENCIES)}
    if platform.python_version() != "3.12.13":
        raise RuntimeError(f"Expected Python 3.12.13, found {platform.python_version()}")
    mismatches = {name: {"expected": expected, "actual": versions[name]}
                  for name, expected in EXPECTED.items() if versions[name] != expected}
    if mismatches:
        raise RuntimeError(f"Unexpected dependency versions: {mismatches}")
    package_file = Path(qmhub.__file__).resolve()
    if not package_file.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"QMHub import is outside the selected environment: {package_file}")
    origins = {name: importlib.import_module(IMPORT_NAMES.get(name, name)).__file__
               for name in versions}
    for name, origin in origins.items():
        if origin is None or not Path(origin).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"Dependency '{name}' import is outside the selected environment: {origin}")
    source_hashes = {}
    if source is not None:
        installed = Path(importlib.import_module("mlpui").__file__).resolve().parent
        # Package metadata alone cannot identify an archive revision. Compare
        # the complete runtime source set with the retained archive: matching
        # known hashes must not conceal injected imports or extra Python files.
        expected_files = {path.relative_to(source / "mlpui")
                          for path in source.joinpath("mlpui").rglob("*.py")}
        installed_files = {path.relative_to(installed) for path in installed.rglob("*.py")}
        if installed_files != expected_files:
            raise RuntimeError(
                f"Installed MLPUI Python file set differs: "
                f"missing={sorted(map(str, expected_files - installed_files))}, "
                f"extra={sorted(map(str, installed_files - expected_files))}"
            )
        for original in sorted(source.joinpath("mlpui").rglob("*.py")):
            relative = original.relative_to(source / "mlpui")
            expected_hash = sha256(original)
            if sha256(installed / relative) != expected_hash:
                raise RuntimeError(f"Installed MLPUI source differs: {relative}")
            source_hashes[str(relative)] = expected_hash
        if not source_hashes:
            raise RuntimeError(f"No MLPUI Python sources found under {source}")
    return {"python": platform.python_version(), "versions": versions,
            "origins": origins, "pinned_source_files_checked": len(source_hashes),
            "pinned_source_manifest_sha256": hashlib.sha256(
                json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()}


def check_numpy_torch():
    import numpy as np
    import torch

    original = np.arange(12, dtype=np.float64).reshape(3, 4)
    tensor = torch.from_numpy(original)
    np.testing.assert_array_equal(tensor.numpy(), original)
    tensor[0, 0] = 17.0
    if original[0, 0] != 17.0:
        raise RuntimeError("NumPy/Torch shared-array conversion failed")
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "shared_memory_roundtrip": True}


def check_native_pme(baseline):
    import numpy as np
    import qmhub.helpmelib as helpme
    from qmhub.utils.dpme import DependPME

    extension = Path(helpme.__file__).resolve()
    actual_hash, baseline_hash = sha256(extension), sha256(baseline)
    if actual_hash != baseline_hash:
        raise RuntimeError("Native helPME extension differs from the qmhub2 baseline")
    cell = np.diag([24.0, 25.0, 26.0])
    alpha = 0.28
    pme = DependPME(cell, np.array(alpha), 6, np.array([48, 48, 48]), 1)
    sources = np.array([[1.1, 2.2, 3.3], [8.4, 9.7, 11.6], [15., 17.5, 19.2]])
    probes = np.array([[2.3, 3.7, 5.4], [10.8, 6.2, 4.1]])
    charges = np.array([0.6, -0.4, -0.2])

    def evaluate(points):
        return pme.compute_recip_esp(points.T, sources.T, charges)

    actual = evaluate(probes)
    if not np.isfinite(actual).all() or np.max(np.abs(actual)) < 1e-6:
        raise RuntimeError("Native reciprocal PME returned invalid or trivial values")
    step = 1e-4
    finite_difference = np.zeros_like(actual[1:])
    for point in range(len(probes)):
        for axis in range(3):
            plus, minus = probes.copy(), probes.copy()
            plus[point, axis] += step
            minus[point, axis] -= step
            finite_difference[axis, point] = (
                evaluate(plus)[0, point] - evaluate(minus)[0, point]) / (2 * step)
    np.testing.assert_allclose(actual[1:], finite_difference, atol=1e-7, rtol=1e-4)

    # An independent reciprocal Fourier sum also detects a native call that
    # returns self-consistent but incorrect potentials and spatial derivatives.
    indices = np.mgrid[-12:13, -12:13, -12:13].reshape(3, -1).T
    indices = indices[np.any(indices != 0, axis=1)]
    vectors = indices @ (2 * np.pi * np.linalg.inv(cell))
    squared = np.square(vectors).sum(axis=1)
    weight = 4 * np.pi / np.linalg.det(cell) * np.exp(-squared / (4 * alpha**2)) / squared
    phase = np.einsum("pna,ka->pnk", sources[None] - probes[:, None], vectors)
    reference = np.empty_like(actual)
    reference[0] = np.einsum("pnk,n,k->p", np.cos(phase), charges, weight)
    reference[1:] = np.einsum("pnk,n,k,ka->ap", np.sin(phase), charges, weight, vectors)
    np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=2e-4)
    return {"extension": str(extension), "sha256": actual_hash,
            "baseline_extension": str(baseline), "baseline_sha256": baseline_hash,
            "reciprocal_esp": actual.tolist(), "probe_count": len(probes),
            "finite_difference_step_angstrom": step,
            "max_gradient_finite_difference_error": float(np.max(np.abs(actual[1:] - finite_difference))),
            "max_fourier_reference_error": float(np.max(np.abs(actual - reference)))}


def analytic_inputs():
    import numpy as np

    return (np.array([[0., 0., 0.], [1.3, .2, -.1], [-.4, 1.2, .3]]),
            np.array([[3., .7, -.3], [-2.1, 2.8, .6], [1.7, -2.3, 1.1]]),
            np.array([-.4, .1, .3]), np.array([.3, -.2, .15]))


def check_mlpui_factory(workdir):
    import numpy as np
    import qmhub.qmtools.mlpui as adapter
    from qmhub.qmtools import QM
    from qmhub.qmtools.qmbase import QMBase
    from qmhub.units import CODATA08_BOHR_TO_A, CODATA08_HARTREE_TO_EV
    from qmhub.utils.darray import DependArray
    from mlpui.potentials import FixedChargeCoulomb, QMMMInput

    if adapter.MLPUI is not adapter.MLPUIEngine or not issubclass(adapter.MLPUI, QMBase):
        raise RuntimeError("MLPUI factory did not provide the expected QMBase engine")
    positions, mm_positions, qm_charges, mm_charges = analytic_inputs()
    spec = {"name": "environment_validation", "charge": 0, "spin": 1, "terms": [
        {"type": "constant", "energy_eV": -0.125},
        {"type": "fixed_charge_coulomb", "qm_charges": qm_charges.tolist()}]}
    with TemporaryDirectory(prefix="analytic-", dir=workdir) as directory:
        spec_path = Path(directory) / "spec.json"
        spec_path.write_text(json.dumps(spec))
        engine = QM.create("mlpui", qm_positions=DependArray(positions.T),
            qm_elements=DependArray(np.array([6, 8, 1])),
            mm_positions=DependArray(mm_positions.T), mm_charges=DependArray(mm_charges),
            charge=0, mult=1, cwd=directory,
            options={"spec": str(spec_path), "threads": "1", "device": "cpu"})
        inputs = QMMMInput(np.array([6, 8, 1]), positions, 0, 1, mm_positions, mm_charges)
        reference = FixedChargeCoulomb(qm_charges).evaluate(inputs,
            ["energy", "forces", "mm_potential", "mm_potential_gradient"])
        energy = np.asarray(engine.qm_energy).item()
        gradient = np.asarray(engine.qm_energy_gradient)
        esp = np.asarray(engine.mm_esp)
        gradient_unit = CODATA08_HARTREE_TO_EV / CODATA08_BOHR_TO_A
        np.testing.assert_allclose(energy, (reference["energy"] - .125) / CODATA08_HARTREE_TO_EV)
        np.testing.assert_allclose(gradient, -reference["forces"].T / gradient_unit)
        np.testing.assert_allclose(esp[0], reference["mm_potential"] / CODATA08_HARTREE_TO_EV)
        np.testing.assert_allclose(esp[1:], reference["mm_potential_gradient"].T / gradient_unit)
    return {"adapter": adapter.__file__, "engine_class": engine.__class__.__qualname__,
            "analytic_spec": spec, "device": "cpu", "trained_weights_used": False,
            "energy_hartree": energy, "gradient_shape": list(gradient.shape),
            "mm_esp_shape": list(esp.shape)}


def check_cuda_coulomb():
    import numpy as np
    import torch
    from mlpui.potentials import FixedChargeCoulomb, QMMMInput
    from mlpui.potentials.terms import COULOMB_EV_ANGSTROM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with GPU access outside the restricted sandbox")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected the CUDA 12.8 PyTorch build, found {torch.version.cuda}")
    positions, mm_positions, qm_charges, mm_charges = analytic_inputs()
    device = torch.device("cuda:0")
    qm = torch.tensor(positions, dtype=torch.float64, device=device, requires_grad=True)
    mm = torch.tensor(mm_positions, dtype=torch.float64, device=device, requires_grad=True)
    charge = torch.tensor(mm_charges, dtype=torch.float64, device=device, requires_grad=True)
    fixed = torch.tensor(qm_charges, dtype=torch.float64, device=device)
    distances = torch.linalg.vector_norm(qm[:, None] - mm[None], dim=-1)
    energy = COULOMB_EV_ANGSTROM * (fixed[:, None] * charge[None] / distances).sum()
    qm_gradient, mm_gradient, potential = torch.autograd.grad(energy, (qm, mm, charge), create_graph=True)
    potential_gradient = torch.autograd.grad(potential.sum(), mm)[0]
    actual = {"energy": energy, "forces": -qm_gradient, "mm_forces": -mm_gradient,
              "mm_potential": potential, "mm_potential_gradient": potential_gradient}
    reference = FixedChargeCoulomb(qm_charges).evaluate(
        QMMMInput(np.array([6, 8, 1]), positions, 0, 1, mm_positions, mm_charges), actual.keys())
    errors = {}
    for name, value in actual.items():
        if value.device.type != "cuda":
            raise RuntimeError(f"{name} was not evaluated on CUDA")
        result = value.detach().cpu().numpy()
        np.testing.assert_allclose(result, reference[name], atol=1e-10, rtol=1e-10)
        errors[name] = float(np.max(np.abs(result - reference[name])))
    torch.testing.assert_close(-mm_gradient, -charge[:, None] * potential_gradient, atol=1e-10, rtol=1e-10)
    torch.cuda.synchronize()
    return {"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_runtime": torch.version.cuda, "compiled_architectures": torch.cuda.get_arch_list(),
            "dtype": "float64", "max_errors": errors,
            "trained_weights_used": False}


def check_cuda_neighbors():
    import numpy as np
    import torch
    from mlpui.models.torchmdnet.extensions import ops

    if ops.BACKEND != "python":
        raise RuntimeError(f"Expected the Python neighbor fallback, found {ops.BACKEND}")
    positions = torch.tensor([[0., 0., 0.], [1., .2, .3], [2., -.2, .1], [8., 0., 0.]],
                             dtype=torch.float64)

    def evaluate(device):
        coordinates = positions.to(device).detach().requires_grad_(True)
        outputs = ops.get_neighbor_pairs_kernel("brute", coordinates,
            torch.zeros(4, dtype=torch.long, device=device), torch.eye(3, device=device),
            False, .1, 2.5, 16, False, True)
        gradient = torch.autograd.grad(outputs[2].sum(), coordinates)[0]
        return (*outputs, gradient)

    cpu, gpu = evaluate("cpu"), evaluate("cuda:0")
    errors = []
    for reference, result in zip(cpu, gpu):
        if result.device.type != "cuda":
            raise RuntimeError("Neighbor fallback returned a non-CUDA tensor")
        first, second = reference.detach().numpy(), result.detach().cpu().numpy()
        np.testing.assert_allclose(first, second, atol=1e-12, rtol=1e-12)
        errors.append(float(np.max(np.abs(first - second))))
    return {"backend": ops.BACKEND, "pairs_found": int(gpu[3].item()),
            "outputs_and_position_gradient_max_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--baseline-helpmelib", type=Path, required=True,
                        help="Existing qmhub2 native extension whose hash must be preserved")
    parser.add_argument("--mlpui-source", type=Path,
                        help="Extracted pinned MLPUI archive for installed-source verification")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # This controls MLPUI's optional neighbor kernels, not QMHub's native PME.
    os.environ.setdefault("MLPUI_NEIGHBORS", "python")
    report = {"schema_version": 1, "executable": sys.executable, "prefix": sys.prefix,
              "host": platform.node(), "checks": {},
              "scope": {"trained_model_validation": False, "external_programs_launched": False,
                        "complete_periodic_qmmm_gradient_validation": False}}
    checks = {
        "versions_and_sources": lambda: check_versions(args.mlpui_source),
        "numpy_torch_roundtrip": check_numpy_torch,
        "native_pme": lambda: check_native_pme(args.baseline_helpmelib),
        "mlpui_factory_cpu_analytic": lambda: check_mlpui_factory(output.parent),
        "cuda_analytic_autograd": check_cuda_coulomb,
        "cuda_python_neighbors": check_cuda_neighbors,
    }
    for name, check in checks.items():
        started = time.perf_counter()
        try:
            # Preserve stdout for JSON even if an imported dependency prints.
            with redirect_stdout(sys.stderr):
                details = check()
            result = {"status": "pass", "details": details}
        except Exception as error:
            result = {"status": "fail", "error": str(error),
                      "error_type": type(error).__name__, "traceback": traceback.format_exc()}
        result["duration_seconds"] = time.perf_counter() - started
        report["checks"][name] = result
    report["success"] = all(check["status"] == "pass" for check in report["checks"].values())
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
