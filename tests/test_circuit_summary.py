"""
tests/test_circuit_summary.py
==============================
hqnn_forge.diagnostics.circuit_summary against hand-computed gate counts.

Angle encoding, n qubits, L layers:
    RX n  ·  CNOT n·L  ·  Rot n·L        depth = 1 + L·(n + 1)
    (a CNOT ring on n wires has depth n: each gate shares a wire with the last)
IQP encoding, r repeats, then the same L entangling layers:
    H n·r  ·  RZ (n + C(n,2))·r  ·  CNOT 2·C(n,2)·r + n·L  ·  Rot n·L
Trainable parameters are 3·n·L for both.
"""

from __future__ import annotations

import math
from collections import Counter
from math import comb
from types import MappingProxyType
from typing import TypedDict

import pennylane as qml
import pytest
import torch

from hqnn_forge.diagnostics import (
    CircuitSummary,
    circuit_summary,
    draw_circuit,
    gradient_variance,
)
from hqnn_forge.diagnostics.circuit import _logical_tape, _written_tape, sample_input
from hqnn_forge.encoding import AmplitudeEncodingLayer, QuantumEncodingLayer, is_circuit_layer
from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier


class _Backend(TypedDict):
    device_name: DeviceName
    diff_method: DiffMethod


CPU: _Backend = {"device_name": "default.qubit", "diff_method": "backprop"}


class TestAngleEncodingCounts:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_gate_counts_depth_and_params(self, n_qubits: int, n_layers: int) -> None:
        s = circuit_summary(QuantumEncodingLayer(n_qubits=n_qubits, n_layers=n_layers, **CPU))
        assert s.layer_type == "QuantumEncodingLayer"
        assert s.n_qubits == n_qubits
        assert dict(s.gate_counts) == {
            "CNOT": n_qubits * n_layers,
            "RX": n_qubits,
            "Rot": n_qubits * n_layers,
        }
        assert s.n_gates == n_qubits + 2 * n_qubits * n_layers
        assert s.n_two_qubit_gates == n_qubits * n_layers
        assert s.depth == 1 + n_layers * (n_qubits + 1)
        assert s.n_trainable_params == 3 * n_qubits * n_layers

    def test_rotation_axis_is_reported(self) -> None:
        s = circuit_summary(QuantumEncodingLayer(n_qubits=2, n_layers=1, rotation="Y", **CPU))
        assert s.gate_counts["RY"] == 2 and "RX" not in s.gate_counts


class TestIQPEncodingCounts:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    @pytest.mark.parametrize("n_repeats", [1, 2])
    def test_gate_counts_and_params(self, n_qubits: int, n_repeats: int) -> None:
        n_layers = 1
        pairs = comb(n_qubits, 2)
        s = circuit_summary(
            IQPEncodingLayer(n_qubits=n_qubits, n_layers=n_layers, n_repeats=n_repeats, **CPU)
        )
        assert s.layer_type == "IQPEncodingLayer"
        assert dict(s.gate_counts) == {
            "CNOT": 2 * pairs * n_repeats + n_qubits * n_layers,
            "Hadamard": n_qubits * n_repeats,
            "RZ": (n_qubits + pairs) * n_repeats,
            "Rot": n_qubits * n_layers,
        }
        assert s.n_two_qubit_gates == 2 * pairs * n_repeats + n_qubits * n_layers
        assert s.n_gates == sum(s.gate_counts.values())
        assert s.n_trainable_params == 3 * n_qubits * n_layers

    def test_costs_more_two_qubit_gates_than_angle(self) -> None:
        angle = circuit_summary(QuantumEncodingLayer(n_qubits=4, n_layers=2, **CPU))
        iqp = circuit_summary(IQPEncodingLayer(n_qubits=4, n_layers=2, **CPU))
        assert iqp.n_two_qubit_gates > angle.n_two_qubit_gates
        assert iqp.depth > angle.depth


def _multirz_layer(n_wires: int, n_qubits: int = 4) -> torch.nn.Module:
    """RX embedding, one trainable ``MultiRZ`` on wires 0 … n_wires-1, ⟨Z_0⟩."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):  # type: ignore[no-untyped-def]
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.MultiRZ(weights[0], wires=range(n_wires))
        return [qml.expval(qml.PauliZ(0))]

    class MultiRZLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (1,)})

    return MultiRZLayer()


class TestMultiWireGateCost:
    """
    #170: a gate on k > 2 wires counts at its two-qubit cost.  MultiRZ on k
    wires is a CNOT ladder, 2(k-1) CNOTs around one RZ; on two wires it is a
    ZZ rotation and stays a single two-qubit gate.
    """

    @pytest.mark.parametrize("n_wires", [3, 4])
    def test_wide_multirz_counts_its_cnot_ladder(self, n_wires: int) -> None:
        s = circuit_summary(_multirz_layer(n_wires))
        assert s.n_two_qubit_gates == 2 * (n_wires - 1)
        assert s.gate_counts.get("CNOT") == 2 * (n_wires - 1)
        assert s.gate_counts.get("RZ") == 1
        assert "MultiRZ" not in s.gate_counts

    def test_two_wire_multirz_stays_one_gate(self) -> None:
        s = circuit_summary(_multirz_layer(2))
        assert s.n_two_qubit_gates == 1
        assert s.gate_counts.get("MultiRZ") == 1
        assert "CNOT" not in s.gate_counts

    @pytest.mark.parametrize("n_wires", [2, 3, 4])
    def test_the_decomposition_is_the_same_unitary(self, n_wires: int) -> None:
        """
        What is counted is the circuit that runs.  Compared as full unitaries:
        the layer's own ⟨Z_0⟩ commutes with a diagonal MultiRZ, so its output
        would not notice a dropped or mis-wired ladder.
        """
        layer = _multirz_layer(n_wires)
        assert is_circuit_layer(layer)
        qlayer = layer.qlayer
        x = torch.tensor([0.3, -1.1, 0.7, 2.0], dtype=torch.float64)
        with torch.no_grad():
            qlayer.weights.copy_(torch.tensor([0.9]))
        qnode = qlayer.qnode
        weights = dict(qlayer.qnode_weights.items())
        written = qml.workflow.construct_tape(qnode, level="top")(x, **weights)
        counted = _logical_tape(layer, inputs=x)
        wires = list(range(4))
        u_written = qml.matrix(written, wire_order=wires)
        u_counted = qml.matrix(counted, wire_order=wires)
        # The CNOT-ladder decomposition is exact, global phase included.
        assert torch.allclose(torch.as_tensor(u_counted), torch.as_tensor(u_written), atol=1e-6)

    def test_wide_gate_without_decomposition_still_counts(self) -> None:
        """An opaque 3-wire gate survives decomposition; it counts once, not zero."""

        class Opaque(qml.operation.Operation):
            num_wires = 3
            num_params = 0

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):  # type: ignore[no-untyped-def]
            qml.AngleEmbedding(inputs, wires=range(3))
            qml.RY(weights[0], wires=0)
            qml.CNOT(wires=[0, 1])
            Opaque(wires=[0, 1, 2])
            return [qml.expval(qml.PauliZ(0))]

        class OpaqueLayer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_qubits = 3
                self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (1,)})

        with pytest.warns(UserWarning):
            s = circuit_summary(OpaqueLayer())
        assert s.gate_counts.get("Opaque") == 1
        assert s.n_two_qubit_gates == 2

    def test_wide_multirz_does_not_hide_inert_parameters(self) -> None:
        """
        Inert parameters are counted on the circuit as written, where a wide
        MultiRZ is one diagonal gate.  Through its CNOT ladder, ⟨Z_1⟩ would
        reach wire 2 and the RY there would stop counting as inert.
        """
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):  # type: ignore[no-untyped-def]
            qml.AngleEmbedding(inputs, wires=range(3))
            qml.RY(weights[1], wires=2)
            qml.MultiRZ(weights[0], wires=[0, 1, 2])
            return [qml.expval(qml.PauliZ(1))]

        class Layer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_qubits = 3
                self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (2,)})

        layer = Layer()
        torch.manual_seed(0)
        x = torch.rand(3, dtype=torch.float64) * 2 * math.pi
        weights = layer.qlayer.qnode_weights["weights"]
        with torch.no_grad():
            weights.copy_(torch.tensor([0.7, 1.3]))
        layer.qlayer(x).sum().backward()
        assert weights.grad is not None
        assert torch.all(weights.grad.abs() < 1e-12)  # both are inert
        assert circuit_summary(layer).n_inert_params == 2


class TestGraphDecomposition:
    """
    With PennyLane's graph-based decomposition enabled, ``decompose`` refuses
    a call without ``gate_set``; the diagnostics must still work, and count
    the same circuit whenever every gate but templates is in the gate set.
    """

    @pytest.fixture
    def graph_enabled(self):  # type: ignore[no-untyped-def]
        was_enabled = qml.decomposition.enabled_graph()
        qml.decomposition.enable_graph()
        try:
            yield
        finally:
            if not was_enabled:
                qml.decomposition.disable_graph()

    @pytest.mark.parametrize(
        "make",
        [
            lambda: QuantumEncodingLayer(n_qubits=3, n_layers=1, **CPU),
            lambda: IQPEncodingLayer(n_qubits=3, n_layers=1, **CPU),
            lambda: _multirz_layer(3),
        ],
        ids=["angle", "iqp", "multirz3"],
    )
    def test_same_summary_and_drawing(self, make, graph_enabled) -> None:  # type: ignore[no-untyped-def]
        torch.manual_seed(0)
        layer = make()
        with_graph = (circuit_summary(layer), draw_circuit(layer))
        qml.decomposition.disable_graph()
        without_graph = (circuit_summary(layer), draw_circuit(layer))
        assert with_graph == without_graph

    # Without GlobalPhase in its gate set, the graph finds no decomposition
    # for these ops and PennyLane falls back with a DecompositionWarning.
    @pytest.mark.filterwarnings("error::pennylane.exceptions.DecompositionWarning")
    @pytest.mark.parametrize("prepare", ["mottonen", "unitary"])
    def test_global_phase_is_not_counted(self, prepare: str, graph_enabled) -> None:  # type: ignore[no-untyped-def]
        """
        Graph decomposition of state preparation and ``QubitUnitary`` emits
        ``GlobalPhase``, which ``op.decomposition()`` does not; left in, a
        two-wire one counted as a two-qubit gate and added to the depth.
        """
        state = torch.tensor([0.1, 0.5, -0.3, 0.8], dtype=torch.float64)
        state = state / state.norm()
        u = torch.as_tensor(qml.matrix(qml.QFT(wires=[0, 1])), dtype=torch.complex128)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):  # type: ignore[no-untyped-def]
            if prepare == "mottonen":
                qml.MottonenStatePreparation(state, wires=[0, 1])
            else:
                qml.QubitUnitary(u, wires=[0, 1])
            qml.RY(weights[0], wires=0)
            return [qml.expval(qml.PauliZ(0))]

        class Layer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_qubits = 2
                self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (1,)})

        layer = Layer()
        with_graph = circuit_summary(layer)
        qml.decomposition.disable_graph()
        without_graph = circuit_summary(layer)
        assert "GlobalPhase" not in with_graph.gate_counts
        assert with_graph.n_two_qubit_gates == without_graph.n_two_qubit_gates
        if prepare == "mottonen":
            # Same rules for every other gate here, so the same circuit.
            assert with_graph == without_graph


class TestDeviceIndependence:
    @pytest.mark.requires_lightning
    @pytest.mark.parametrize("layer_cls", [QuantumEncodingLayer, IQPEncodingLayer])
    def test_same_counts_on_lightning_adjoint(self, layer_cls: type) -> None:
        """lightning's adjoint path rewrites Rot as RZ·RY·RZ; the summary must not."""
        cpu = circuit_summary(layer_cls(n_qubits=3, n_layers=2, **CPU))
        lightning = circuit_summary(
            layer_cls(n_qubits=3, n_layers=2, device_name="lightning.qubit", diff_method="adjoint")
        )
        assert lightning.device_name == "lightning.qubit" and lightning.diff_method == "adjoint"
        assert cpu.device_name == "default.qubit" and cpu.diff_method == "backprop"
        for attr in ("depth", "n_gates", "n_two_qubit_gates", "n_trainable_params"):
            assert getattr(lightning, attr) == getattr(cpu, attr), attr
        assert dict(lightning.gate_counts) == dict(cpu.gate_counts)


class TestModelPassthrough:
    @pytest.mark.parametrize("model_cls", [HybridBinaryClassifier, ParallelHybridClassifier])
    @pytest.mark.parametrize("encoding_type", ["angle", "iqp"])
    def test_model_summary_is_its_quantum_layer(self, model_cls: type, encoding_type: str) -> None:
        model = model_cls(
            n_input_features=6, n_qubits=3, n_layers=2, encoding_type=encoding_type, **CPU
        )
        assert circuit_summary(model) == circuit_summary(model.quantum_layer)

    def test_published_configuration(self) -> None:
        """The thesis SHNN: 8 qubits, 2 layers, angle encoding."""
        s = circuit_summary(
            HybridBinaryClassifier(n_input_features=8, n_qubits=8, n_layers=2, **CPU)
        )
        assert s.n_qubits == 8 and s.n_trainable_params == 48
        assert s.gate_counts["CNOT"] == 16 and s.depth == 19

    def test_frozen_weights_are_not_counted(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=2, **CPU)
        layer.qlayer.weights.requires_grad_(False)
        assert circuit_summary(layer).n_trainable_params == 0

    def test_unsupported_target_raises(self) -> None:
        with pytest.raises(TypeError, match="expects an encoding layer.*got Linear"):
            circuit_summary(torch.nn.Linear(3, 1))


class TestPresentation:
    def test_str_lists_every_field(self) -> None:
        s = circuit_summary(QuantumEncodingLayer(n_qubits=3, n_layers=2, **CPU))
        text = str(s)
        assert text.splitlines()[0] == (
            "Circuit summary: QuantumEncodingLayer on default.qubit (backprop)"
        )
        expected = [
            ("qubits", 3),
            ("trainable params", 18),
            ("inert params", 3),
            ("depth", 9),
            ("gates", 15),
            ("two-qubit gates", 6),
            ("CNOT", 6),
            ("RX", 3),
            ("Rot", 6),
        ]
        for label, value in expected:
            assert any(
                line.strip().startswith(label) and line.rstrip().endswith(f": {value}")
                for line in text.splitlines()
            ), label

    def test_str_values_are_aligned(self) -> None:
        text = str(circuit_summary(IQPEncodingLayer(n_qubits=3, n_layers=1, **CPU)))
        colons = {line.index(" : ") for line in text.splitlines()[1:]}
        assert len(colons) == 1, text

    def test_to_dict_round_trips(self) -> None:
        s = circuit_summary(IQPEncodingLayer(n_qubits=3, n_layers=1, **CPU))
        d = s.to_dict()
        assert isinstance(d["gate_counts"], dict)
        assert d.pop("n_effective_params") == s.n_effective_params
        assert CircuitSummary(**d) == s

    def test_to_dict_accepts_any_mapping(self) -> None:
        """gate_counts is annotated Mapping, so a non-dict one must convert."""
        s = CircuitSummary(
            layer_type="QuantumEncodingLayer",
            n_qubits=2,
            n_trainable_params=6,
            depth=5,
            n_gates=8,
            n_two_qubit_gates=2,
            gate_counts=MappingProxyType({"CNOT": 2, "RX": 2, "Rot": 2}),
        )
        d = s.to_dict()
        assert type(d["gate_counts"]) is dict
        assert d["gate_counts"] == {"CNOT": 2, "RX": 2, "Rot": 2}

    def test_is_immutable(self) -> None:
        s = circuit_summary(QuantumEncodingLayer(n_qubits=2, n_layers=1, **CPU))
        with pytest.raises(AttributeError):
            s.depth = 0  # type: ignore[misc]

    def test_draw_shows_the_logical_gates(self) -> None:
        drawing = draw_circuit(QuantumEncodingLayer(n_qubits=2, n_layers=1, **CPU))
        assert "RX(0.00)" in drawing and "Rot(" in drawing and "<Z>" in drawing
        assert drawing.count("\n") >= 1  # one line per wire

    def test_draw_uses_the_given_inputs(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=2, n_layers=1, **CPU)
        drawing = draw_circuit(layer, inputs=torch.tensor([0.5, 1.25]))
        assert "RX(0.50)" in drawing and "RX(1.25)" in drawing
        assert "RX(0.00)" not in drawing

    def test_draw_accepts_a_model_and_decimals(self) -> None:
        model = HybridBinaryClassifier(n_input_features=2, n_qubits=2, n_layers=1, **CPU)
        drawing = draw_circuit(model, decimals=1)
        assert "RX(0.0)" in drawing


# ---------------------------------------------------------------------------
# Amplitude embedding: input width and state preparation (#218)
# ---------------------------------------------------------------------------


def _amplitude(n_qubits: int = 3, n_features: int | None = None) -> AmplitudeEncodingLayer:
    torch.manual_seed(0)
    return AmplitudeEncodingLayer(
        n_qubits=n_qubits,
        n_layers=1,
        n_features=n_features,
        device_name="default.qubit",
        diff_method="backprop",
    )


class TestAmplitudeEmbedding:
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_summary_counts_the_full_state_preparation(self, n_qubits: int) -> None:
        summary = circuit_summary(_amplitude(n_qubits))
        full = 2**n_qubits - 1
        assert summary.gate_counts["RY"] == full and summary.gate_counts["RZ"] == full
        assert summary.gate_counts["Rot"] == n_qubits

    def test_the_counts_bound_every_real_input(self) -> None:
        # PennyLane drops rotation blocks whose angles are all zero, so a
        # particular input can run fewer gates, never more.
        layer = _amplitude(3)
        bound = circuit_summary(layer).gate_counts
        g = torch.Generator().manual_seed(1)
        seen_rz = set()
        for _ in range(20):
            x = torch.randn(8, generator=g, dtype=torch.float64)
            counts = Counter(op.name for op in _logical_tape(layer, x).operations)
            assert all(counts[gate] <= bound[gate] for gate in counts), counts
            seen_rz.add(counts["RZ"])
        assert max(seen_rz) == bound["RZ"]
        positive = Counter(op.name for op in _logical_tape(layer, torch.ones(8)).operations)
        assert positive["RZ"] == 0  # why the default sample has a negative entry

    def test_padded_features(self) -> None:
        layer = _amplitude(3, n_features=5)
        summary = circuit_summary(layer)
        assert summary.gate_counts["RY"] > 0
        drawing = draw_circuit(layer, torch.tensor([0.3, -1.0, 2.0, 0.5, 1.5]))
        assert all(f"{w}: " in drawing for w in range(3)) and drawing.count("<Z>") == 3

    def test_the_circuit_gets_the_padded_normalised_state(self) -> None:
        """The state prepared is x zero-padded to 2**n and scaled to unit norm, by hand."""
        layer = _amplitude(3, n_features=5)
        x = torch.tensor([0.3, -1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
        padded = torch.cat([x, torch.zeros(3, dtype=torch.float64)])
        by_hand = padded / math.sqrt(float((padded**2).sum()))
        (state,) = _written_tape(layer, x).operations[0].data
        torch.testing.assert_close(state, by_hand)
        default = _written_tape(layer).operations[0].data[0]
        full = torch.tensor([-1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(default, full / math.sqrt(55.0))

    def test_gradient_variance_matches_hand_built_inputs(self) -> None:
        """
        Each draw is 2**n raw features uniform in [-π, π], padded and
        normalised, not n_qubits of them: recomputed here on the raw QNode.
        """
        layer = _amplitude(3)
        w0 = torch.linspace(0.1, 2.0, 9, dtype=torch.float32).reshape(1, 3, 3)

        def fixed(w: torch.Tensor) -> None:
            w.copy_(w0)

        n = 3
        result = gradient_variance(
            layer, n_samples=n, init=fixed, generator=torch.Generator().manual_seed(7)
        )
        gen = torch.Generator().manual_seed(7)
        grads = []
        for _ in range(n):
            x = (torch.rand(1, 8, generator=gen) * 2 - 1) * math.pi
            state = x[0].to(torch.float64) / torch.linalg.vector_norm(x[0].to(torch.float64))
            w = w0.clone().requires_grad_(True)
            (z0, *_) = layer.qlayer.qnode(state, weights=w)
            (g,) = torch.autograd.grad(z0, w)
            grads.append(g.to(torch.float64))
        torch.testing.assert_close(result.per_parameter, torch.stack(grads).var(dim=0))

    def test_draw_default_and_width_check(self) -> None:
        layer = _amplitude(3)
        drawing = draw_circuit(layer)
        assert all(f"{w}: " in drawing for w in range(3)) and "RZ(" in drawing
        with pytest.raises(ValueError, match=r"one sample of shape \(8,\)"):
            draw_circuit(layer, torch.zeros(3))

    def test_gradient_variance_runs_on_the_amplitude_width(self) -> None:
        layer = _amplitude(3)
        before = layer.qlayer.weights.detach().clone()
        a = gradient_variance(layer, n_samples=4, generator=torch.Generator().manual_seed(0))
        b = gradient_variance(layer, n_samples=4, generator=torch.Generator().manual_seed(0))
        assert a.per_parameter.shape == layer.qlayer.weights.shape
        assert a.total_variance > 0
        torch.testing.assert_close(a.per_parameter, b.per_parameter, rtol=0, atol=0)
        torch.testing.assert_close(layer.qlayer.weights, before, rtol=0, atol=0)
        with pytest.raises(ValueError):
            gradient_variance(layer, n_samples=2, input_scale=0.0)  # no state has zero norm

    def test_angle_layers_still_draw_zeros_by_default(self) -> None:
        layer = QuantumEncodingLayer(
            n_qubits=3, n_layers=1, device_name="default.qubit", diff_method="backprop"
        )
        assert "RX(0.00)" in draw_circuit(layer)
        assert sample_input(layer).tolist() == [0.0, 0.0, 0.0]
