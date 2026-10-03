"""
tests/test_published_shnn_parity.py
====================================
Structural comparison with the SHNN published in the thesis and the
``hqnn-fraud-detection-benchmark`` repository.

What is checked, and how
------------------------
The reference model is rebuilt here from the benchmark's own source
(``src/models/quantum/shnn.py`` and ``vqc.py`` with the ``shnn`` block of
``configs/default.yaml``): ``Linear(8→8)`` + ``π·sigmoid``, ``AngleEmbedding``
with RY, ``StronglyEntanglingLayers`` (2 layers), readout ⟨Z_0⟩,
``Linear(1→1)``.  Its parameter count must equal the published 122, which
validates the rebuild.

Checked structurally (fast, no training):

* ``HybridBinaryClassifier.published_shnn()`` (#131) is the published model:
  122 parameters, identical circuit resources and CNOT pairs, and identical
  logits once the weights are copied across;
* the library's *default* ``HybridBinaryClassifier(8, 8, 2)`` shares qubits,
  layers, quantum parameter count and gate budget with it, and the defaults
  that differ (head width, embedding axis, entangler order and range, depth,
  encoder range) are pinned so a change to them shows up here.

Not checked here: the published MCC (0.5758 ± 0.0371) and MCC/kParam
(4.720) come from 5-fold CV on the 284,807-row Kaggle dataset with SMOTE and
100 epochs.  That run is ``tests/test_published_shnn_reproduction.py``, opt-in
because it needs the dataset (not redistributable) and days of simulation.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pennylane as qml
import pytest
import torch
from torch import nn

from hqnn_forge.diagnostics import LOGICAL_GATE_SET, CircuitSummary, circuit_summary
from hqnn_forge.diagnostics.circuit import _tape_resources
from hqnn_forge.models import HybridBinaryClassifier

pytestmark = pytest.mark.reproducibility

N_QUBITS = 8
N_LAYERS = 2
PUBLISHED_PARAMS = 122  # thesis / benchmark README results table
PUBLISHED_QUANTUM_PARAMS = 48  # configs/default.yaml: "48 quantum params → total ~122 params"


class _PiSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.pi * torch.sigmoid(x)


def _published_circuit() -> qml.QNode:
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> qml.measurements.ExpectationMP:
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    return circuit


def _published_shnn() -> tuple[nn.ModuleDict, qml.QNode]:
    circuit = _published_circuit()
    vqc = qml.qnn.TorchLayer(circuit, {"weights": (N_LAYERS, N_QUBITS, 3)})
    pre = nn.Sequential(nn.Linear(N_QUBITS, N_QUBITS), _PiSigmoid())
    post = nn.Linear(1, 1)  # followed by a parameter-free Sigmoid
    return nn.ModuleDict({"pre": pre, "vqc": vqc, "post": post}), circuit


def _logical_tape(circuit: qml.QNode, **weights: torch.Tensor) -> qml.tape.QuantumScript:
    """The tape ``circuit`` runs for one sample, decomposed to ``LOGICAL_GATE_SET``."""
    tape = qml.workflow.construct_tape(circuit, level="top")(torch.zeros(N_QUBITS), **weights)
    (decomposed,), _ = qml.transforms.decompose(tape, gate_set=LOGICAL_GATE_SET)
    return decomposed


def _first_entangler_gate(tape: qml.tape.QuantumScript) -> str:
    """Name of the first gate after the ``N_QUBITS`` embedding rotations."""
    return str(tape.operations[N_QUBITS].name)


def _published_summary(
    circuit: qml.QNode, n_quantum_params: int
) -> tuple[CircuitSummary, qml.tape.QuantumScript]:
    tape = _logical_tape(circuit, weights=torch.zeros(N_LAYERS, N_QUBITS, 3))
    res = _tape_resources(tape)
    summary = CircuitSummary(
        layer_type="published",
        n_qubits=N_QUBITS,
        n_trainable_params=n_quantum_params,
        depth=res.depth,
        n_gates=res.n_gates,
        n_two_qubit_gates=res.n_two_qubit_gates,
        gate_counts=res.gate_counts,
    )
    return summary, tape


def _our_tape(model: HybridBinaryClassifier) -> qml.tape.QuantumScript:
    qlayer = model.quantum_layer.qlayer
    weights = {name: param.detach() for name, param in qlayer.qnode_weights.items()}
    return _logical_tape(qlayer.qnode, **weights)


def _ours() -> HybridBinaryClassifier:
    return HybridBinaryClassifier(
        n_input_features=N_QUBITS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        device_name="default.qubit",
        diff_method="backprop",
    )


@pytest.fixture(scope="module")
def published() -> tuple[nn.ModuleDict, CircuitSummary, qml.tape.QuantumScript]:
    model, circuit = _published_shnn()
    # Counted from the rebuild rather than assumed: ``test_rebuild_quantum_parameters``
    # is what checks this number against the published table.
    n_quantum_params = sum(p.numel() for p in model["vqc"].parameters())
    summary, tape = _published_summary(circuit, n_quantum_params)
    return model, summary, tape


@pytest.fixture(scope="module")
def configured() -> tuple[HybridBinaryClassifier, CircuitSummary]:
    model = HybridBinaryClassifier.published_shnn(
        device_name="default.qubit", diff_method="backprop"
    )
    return model, circuit_summary(model)


@pytest.fixture(scope="module")
def ours() -> tuple[HybridBinaryClassifier, CircuitSummary, qml.tape.QuantumScript]:
    torch.manual_seed(0)  # the encoder's random init feeds test_encoder_output_range
    model = _ours()
    return model, circuit_summary(model), _our_tape(model)


class TestReferenceRebuild:
    def test_rebuild_has_the_published_parameter_count(self, published: tuple) -> None:
        model, _, _ = published
        assert sum(p.numel() for p in model.parameters()) == PUBLISHED_PARAMS

    def test_rebuild_quantum_parameters(self, published: tuple) -> None:
        model, _, _ = published
        assert sum(p.numel() for p in model["vqc"].parameters()) == PUBLISHED_QUANTUM_PARAMS


class TestPublishedConfigurationParity:
    """
    ``HybridBinaryClassifier.published_shnn()`` must *be* the published model:
    same parameter count, same circuit resources, same CNOT pairs, and the
    same logits when the weights are copied across.
    """

    def test_parameter_counts(self, configured: tuple) -> None:
        model, summary = configured
        assert model.count_parameters() == PUBLISHED_PARAMS
        assert summary.n_trainable_params == PUBLISHED_QUANTUM_PARAMS
        assert sum(p.numel() for p in model.classical_encoder.parameters()) == 8 * 8 + 8
        assert sum(p.numel() for p in model.head.parameters()) == 2

    def test_circuit_resources_are_identical(self, published: tuple, configured: tuple) -> None:
        _, ref, _ = published
        _, summary = configured
        assert summary.n_qubits == ref.n_qubits
        assert summary.depth == ref.depth == 15
        assert summary.n_gates == ref.n_gates == 40
        assert summary.n_two_qubit_gates == ref.n_two_qubit_gates == 16
        assert dict(summary.gate_counts) == dict(ref.gate_counts)
        assert summary.gate_counts["RY"] == 8 and "RX" not in summary.gate_counts

    def test_cnot_pairs_and_gate_order_are_identical(
        self, published: tuple, configured: tuple, cnot_pairs: Callable
    ) -> None:
        _, _, ref_tape = published
        model, _ = configured
        our_tape = _our_tape(model)
        assert cnot_pairs(our_tape) == cnot_pairs(ref_tape)
        assert _first_entangler_gate(our_tape) == _first_entangler_gate(ref_tape) == "Rot"

    def test_same_logits_with_the_same_weights(self, published: tuple, configured: tuple) -> None:
        """
        End-to-end numerical parity: copy the encoder, quantum and head
        weights from the rebuilt published model and compare logits.
        """
        reference, _, _ = published
        model, _ = configured
        with torch.no_grad():
            model.classical_encoder[0].weight.copy_(reference["pre"][0].weight)
            model.classical_encoder[0].bias.copy_(reference["pre"][0].bias)
            model.quantum_layer.qlayer.weights.copy_(reference["vqc"].weights)
            model.head.weight.copy_(reference["post"].weight)
            model.head.bias.copy_(reference["post"].bias)
        torch.manual_seed(0)
        x = torch.randn(6, N_QUBITS)
        with torch.no_grad():
            ours = model(x).squeeze(-1)
            angles = reference["pre"](x)
            theirs = reference["post"](reference["vqc"](angles).reshape(-1, 1)).squeeze(-1)
        torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-6)

    def test_quantum_init_is_normal_with_std_0_1(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier.published_shnn(
            device_name="default.qubit", diff_method="backprop"
        )
        weights = model.quantum_layer.qlayer.weights.detach()
        assert weights.std().item() == pytest.approx(0.1, rel=0.3)  # 48 draws
        assert abs(weights.mean().item()) < 0.1


class TestDefaultConfigurationIsAVariant:
    """
    The library's default ``HybridBinaryClassifier(8, 8, 2)`` shares the
    qubit count, layer count, quantum parameter count and gate budget with
    the published SHNN but is not that model.  Each assertion documents one
    default that differs, so a change to the defaults shows up here.
    """

    def test_qubits_layers_and_quantum_parameters(self, published: tuple, ours: tuple) -> None:
        _, ref, _ = published
        model, summary, _ = ours
        assert summary.n_qubits == ref.n_qubits == N_QUBITS
        assert model.n_layers == N_LAYERS
        assert summary.n_trainable_params == ref.n_trainable_params == PUBLISHED_QUANTUM_PARAMS

    def test_gate_budget(self, published: tuple, ours: tuple) -> None:
        _, ref, _ = published
        _, summary, _ = ours
        assert summary.n_gates == ref.n_gates == 40
        assert summary.n_two_qubit_gates == ref.n_two_qubit_gates == 16
        assert summary.gate_counts["Rot"] == ref.gate_counts["Rot"] == 16
        assert summary.gate_counts["CNOT"] == ref.gate_counts["CNOT"] == 16

    def test_classical_encoder_size(self, ours: tuple) -> None:
        model, _, _ = ours
        assert sum(p.numel() for p in model.classical_encoder.parameters()) == 8 * 8 + 8

    def test_total_parameters_differ_by_the_head(self, ours: tuple) -> None:
        model, _, _ = ours
        # Published head: Linear(1→1) = 2.  Default: Linear(8→1) = 9.
        assert model.count_parameters() == PUBLISHED_PARAMS - 2 + 9 == 129
        assert model.head.in_features == N_QUBITS  # readout="all"

    def test_embedding_axis(self, published: tuple, ours: tuple) -> None:
        _, ref, _ = published
        _, summary, _ = ours
        assert ref.gate_counts.get("RY") == 8 and "RX" not in ref.gate_counts
        assert summary.gate_counts.get("RX") == 8 and "RY" not in summary.gate_counts

    def test_entangler_range(self, published: tuple, ours: tuple, cnot_pairs: Callable) -> None:
        _, ref, ref_tape = published
        _, summary, our_tape = ours
        ring = [(i, (i + 1) % N_QUBITS) for i in range(N_QUBITS)]
        ring2 = [(i, (i + 2) % N_QUBITS) for i in range(N_QUBITS)]
        # StronglyEntanglingLayers uses range l mod (n-1) + 1: range 1, then range 2.
        assert cnot_pairs(ref_tape) == ring + ring2
        # Ours is a range-1 ring in every layer.
        assert cnot_pairs(our_tape) == ring + ring
        # The range, not the gate order, is what costs the depth: a range-2 ring on
        # 8 qubits splits into two independent 4-cycles, while a range-1 ring
        # serialises around all 8.  Swapping Rot and CNOT leaves both numbers alone.
        assert ref.depth == 15 and summary.depth == 19

    def test_entangler_order(self, published: tuple, ours: tuple) -> None:
        _, _, ref_tape = published
        _, _, our_tape = ours
        assert _first_entangler_gate(ref_tape) == "Rot"  # Rot, then the CNOT ring
        assert _first_entangler_gate(our_tape) == "CNOT"  # CNOT ring, then Rot

    def test_encoder_output_range(self, ours: tuple) -> None:
        model, _, _ = ours
        x = torch.linspace(-50, 50, 8 * 5).reshape(5, 8)
        with torch.no_grad():
            scaled = model.classical_encoder(x) * torch.pi
            published = _PiSigmoid()(x)
        assert scaled.min() < 0 <= published.min()  # (-π, π) vs (0, π)


class TestLiveParameters:
    """
    #234: with the ⟨Z_0⟩ readout 20 of the 48 quantum weights never move the
    output.  They are kept (the published shape); the live count is pinned
    here against autograd, and the structural count stays a lower bound.
    """

    #: Dead entries of the (layer, wire, angle) weight tensor: the whole
    #: last-layer Rot on wires 0, 1, 3, 5, 7, its ω on 2, 4, 6, and the
    #: first-layer ω on wires 0 and 1.
    DEAD = sorted(
        [(1, w, a) for w in (0, 1, 3, 5, 7) for a in range(3)]
        + [(1, w, 2) for w in (2, 4, 6)]
        + [(0, 0, 2), (0, 1, 2)]
    )

    def _always_zero(self) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model = HybridBinaryClassifier.published_shnn(
            device_name="default.qubit", diff_method="backprop"
        ).double()
        params = dict(model.named_parameters())
        zero = {name: torch.ones_like(p, dtype=torch.bool) for name, p in params.items()}
        g = torch.Generator().manual_seed(1)
        for _ in range(8):
            with torch.no_grad():
                for name, p in params.items():
                    low, high = (0.0, 2 * math.pi) if "qlayer" in name else (-1.0, 1.0)
                    p.uniform_(low, high, generator=g)
            x = torch.randn(4, N_QUBITS, generator=g, dtype=torch.float64)
            model.zero_grad()
            model(x).sum().backward()
            for name, p in params.items():
                assert p.grad is not None
                zero[name] &= p.grad.abs() < 1e-12
        return zero

    def test_dead_quantum_weights_and_the_live_count(self) -> None:
        zero = self._always_zero()
        quantum = zero["quantum_layer.qlayer.weights"]
        assert sorted(tuple(int(i) for i in idx) for idx in quantum.nonzero()) == self.DEAD
        # Every classical parameter is live: all eight wires reach the readout.
        assert all(not z.any() for name, z in zero.items() if "qlayer" not in name)
        live = sum(int((~z).sum()) for z in zero.values())
        assert (live, sum(z.numel() for z in zero.values())) == (102, PUBLISHED_PARAMS)

    def test_structural_count_is_a_lower_bound(self) -> None:
        model = HybridBinaryClassifier.published_shnn(
            device_name="default.qubit", diff_method="backprop"
        )
        summary = circuit_summary(model)
        assert summary.n_inert_params == 16
        assert summary.n_inert_params <= len(self.DEAD)
