"""
tests/test_tape_resources.py
============================
Gate counts and depth computed from the tape (#344), against PennyLane's own
``resources_from_tape`` on the installed version.

``circuit_summary`` used to read ``tape.specs["resources"]``, whose fields
PennyLane 0.46 renames or drops (``num_gates``, ``gate_types``,
``gate_sizes``) and whose property it removes (#568).  It now counts itself;
these tests check that the counts are the ones ``resources_from_tape``
reports, using whichever spelling the installed PennyLane has, so they also
guard the next rename.
"""

from __future__ import annotations

from typing import Any

import pennylane as qml
import pytest
from packaging.version import Version

from hqnn_forge.diagnostics import circuit_summary
from hqnn_forge.diagnostics.circuit import _logical_tape, _tape_resources
from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    QuantumEncodingLayer,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}

PENNYLANE_046 = Version(qml.__version__) >= Version("0.46.0.dev0")

# Each layer with its number of gates on two or more wires, pinned by hand
# count: PennyLane 0.46 has no ``gate_sizes`` to compare it with.  Under
# backprop, the amplitude layer's gradient-carrying state preparation emits
# 12 CNOTs (plus 6 in the 2 entangling layers, for 18 total) on both 0.45
# and 0.46.
LAYERS = [
    pytest.param(lambda: QuantumEncodingLayer(n_qubits=4, n_layers=2, **CPU), 8, id="angle"),
    pytest.param(
        lambda: QuantumEncodingLayer(
            n_qubits=5, n_layers=3, entangler="strongly_entangling", readout="first", **CPU
        ),
        15,
        id="angle-strongly-first",
    ),
    pytest.param(
        lambda: QuantumEncodingLayer(n_qubits=5, n_layers=2, entangler="brickwork", **CPU),
        8,
        id="angle-brickwork",
    ),
    pytest.param(
        lambda: IQPEncodingLayer(n_qubits=4, n_layers=2, n_repeats=2, **CPU), 32, id="iqp"
    ),
    pytest.param(
        lambda: AmplitudeEncodingLayer(n_qubits=3, n_layers=2, **CPU),
        18,
        id="amplitude",
    ),
    pytest.param(
        lambda: DataReuploadingLayer(n_qubits=3, n_layers=3, trainable_input_scaling=True, **CPU),
        9,
        id="reuploading",
    ),
]


def _pennylane_resources(tape: qml.tape.QuantumScript) -> dict[str, Any]:
    """
    PennyLane's resources across supported API versions.

    PennyLane 0.46 renames and drops several fields of ``SpecsResources``:
    - ``gate_types`` (0.45) is renamed to ``counts`` (0.46+)
    - ``num_gates`` (0.45) is renamed to ``total_quantum_operations`` (0.46+)
    - ``depth`` is unchanged across versions
    - ``gate_sizes`` (0.45) is dropped in 0.46+
    """
    res = qml.resource.resources_from_tape(tape)
    counts = getattr(res, "gate_types", None)
    if counts is None:
        counts = res.counts
    total = getattr(res, "num_gates", None)
    if total is None:
        total = res.total_quantum_operations
    return {
        "depth": res.depth,
        "n_gates": total,
        "gate_counts": dict(counts),
        "gate_sizes": getattr(res, "gate_sizes", None),
    }


@pytest.mark.parametrize(("build", "n_two_qubit_gates"), LAYERS)
def test_counts_match_pennylane_resources(build: Any, n_two_qubit_gates: int) -> None:
    tape = _logical_tape(build())
    ours = _tape_resources(tape)
    theirs = _pennylane_resources(tape)
    assert ours.depth == theirs["depth"]
    assert ours.n_gates == theirs["n_gates"]
    assert ours.gate_counts == dict(sorted(theirs["gate_counts"].items()))
    assert ours.n_two_qubit_gates == n_two_qubit_gates
    gate_sizes = theirs["gate_sizes"]
    if gate_sizes is not None:  # dropped in PennyLane 0.46
        assert ours.n_two_qubit_gates == sum(c for size, c in gate_sizes.items() if size >= 2)


@pytest.mark.parametrize(("build", "n_two_qubit_gates"), LAYERS)
def test_the_summary_is_internally_consistent(build: Any, n_two_qubit_gates: int) -> None:
    summary = circuit_summary(build())
    assert summary.n_two_qubit_gates == n_two_qubit_gates
    assert summary.n_gates == sum(summary.gate_counts.values())
    assert 0 < summary.n_two_qubit_gates <= summary.n_gates
    assert summary.depth <= summary.n_gates


def test_depth_by_hand() -> None:
    # Layers H(0),RX(2) -> CNOT(0,1) -> RZ(1) -> CZ(1,2) -> Toffoli(0,1,2):
    # RX(2) runs alongside H(0), and the Toffoli waits for the CZ, so depth 5.
    ops = [
        qml.Hadamard(0),
        qml.RX(0.1, 2),
        qml.CNOT([0, 1]),
        qml.RZ(0.2, 1),
        qml.CZ([1, 2]),
        qml.Toffoli([0, 1, 2]),
    ]
    tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
    res = _tape_resources(tape)
    assert res.depth == 5
    assert res.n_gates == 6
    assert res.n_two_qubit_gates == 3  # CNOT, CZ, and the three-qubit Toffoli once
    assert res.gate_counts == {"CNOT": 1, "CZ": 1, "Hadamard": 1, "RX": 1, "RZ": 1, "Toffoli": 1}
    # Sorted by name, the order CircuitSummary prints; == on dicts ignores it.
    assert list(res.gate_counts) == sorted(res.gate_counts)


def test_empty_tape() -> None:
    res = _tape_resources(qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))]))
    assert res == (0, 0, 0, {})
