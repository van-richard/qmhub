"""Engine backed by an MLPUI potential (any mlpui.potentials PotentialTerm).

Requires MLPUI (https://github.com/MoriyaSuwako2524/MLPUI) on the Python path.
Configure with ``[engine] high = mlpui`` and ``[high] spec = <potential JSON>``;
see mlpui/potentials/qmhub.py for the options and the unit contract.
"""
try:
    from mlpui.potentials.qmhub import make_engine_class
except ImportError as exc:  # pragma: no cover
    raise ImportError("QMHub engine 'mlpui' requires the mlpui package (mlpui.potentials)") from exc

MLPUI = MLPUIEngine = make_engine_class()
