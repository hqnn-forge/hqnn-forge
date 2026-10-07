"""
tests/test_hardware_efficient_entangler.py
==========================================
``entangler="hardware_efficient"``: the CZ-ladder + RY block of
``hqnn_forge.circuits.hardware_efficient_layer`` as a variational block of
every encoder and classifier (#304).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pennylane as qml
import pytest
import torch

from hqnn_forge.circuits import hardware_efficient_layer
from hqnn_forge.diagnostics import (
    circuit_summary,
    effective_dimension,
    gradient_variance,
)
from hqnn_forge.encoding import DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.angle_embedding import Readout, RotationAxis, variational_weight_shape
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.utils import load_checkpoint, save_checkpoint

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
HE: dict[str, Any] = {"entangler": "hardware_efficient", **CPU}


def _hand_written_block(w: torch.Tensor, n: int) -> None:
    for q in range(n - 1):
        qml.CZ(wires=[q, q + 1])
    for q in range(n):
        qml.RY(w[q], wires=q)


def _random_weights(layer: Any, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.rand(p.shape, generator=g) * 2 * torch.pi)


class TestCircuit:
    def test_weight_shape_is_one_angle_per_qubit_per_layer(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=4, n_layers=3, **HE)
        assert tuple(layer.qlayer.weights.shape) == (3, 4)
        assert variational_weight_shape("hardware_efficient", 4, 3) == (3, 4)

    @pytest.mark.parametrize("shape", [(3, 3), (4,), (3, 1), ()])
    def test_primitive_rejects_any_shape_but_one_angle_per_qubit(
        self, shape: tuple[int, ...]
    ) -> None:
        # (3, 3) is a Rot row: shape[0] matches n_qubits, and RY would broadcast over it.
        with pytest.raises(ValueError, match=r"n_qubits=3"):
            hardware_efficient_layer(torch.zeros(shape), 3)

    @pytest.mark.parametrize("rotation", ["X", "Y"])
    def test_angle_layer_matches_the_hand_written_circuit(self, rotation: RotationAxis) -> None:
        n, n_layers = 3, 2
        layer = QuantumEncodingLayer(n_qubits=n, n_layers=n_layers, rotation=rotation, **HE)
        _random_weights(layer, 0)
        w = layer.qlayer.weights.detach().double()
        x = torch.rand(4, n, generator=torch.Generator().manual_seed(1), dtype=torch.float64)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def reference(xi: torch.Tensor) -> list[Any]:
            qml.AngleEmbedding(xi, wires=range(n), rotation=rotation)
            for ell in range(n_layers):
                _hand_written_block(w[ell], n)
            return [qml.expval(qml.PauliZ(q)) for q in range(n)]

        expected = torch.stack([torch.stack(reference(xi)) for xi in x])
        with torch.no_grad():
            torch.testing.assert_close(layer(x.float()).double(), expected, atol=1e-6, rtol=0)

    def test_reuploading_interleaves_embedding_and_blocks(self) -> None:
        # The re-uploading layer applies the blocks one at a time with a
        # layer_offset; the hardware-efficient block ignores it, so the
        # circuit is embed, block 0, embed, block 1.
        n, n_layers = 3, 2
        layer = DataReuploadingLayer(n_qubits=n, n_layers=n_layers, **HE)
        _random_weights(layer, 2)
        w = layer.qlayer.weights.detach().double()
        x = torch.rand(3, n, generator=torch.Generator().manual_seed(3), dtype=torch.float64)

        @qml.qnode(qml.device("default.qubit", wires=n))
        def reference(xi: torch.Tensor) -> list[Any]:
            for ell in range(n_layers):
                qml.AngleEmbedding(xi, wires=range(n), rotation="X")
                _hand_written_block(w[ell], n)
            return [qml.expval(qml.PauliZ(q)) for q in range(n)]

        expected = torch.stack([torch.stack(reference(xi)) for xi in x])
        with torch.no_grad():
            torch.testing.assert_close(layer(x.float()).double(), expected, atol=1e-6, rtol=0)

    def test_iqp_layer_runs_the_block_after_the_embedding(self) -> None:
        layer = IQPEncodingLayer(n_qubits=3, n_layers=2, **HE)
        summary = circuit_summary(layer)
        assert summary.gate_counts["CZ"] == 2 * 2
        assert summary.gate_counts["RY"] == 3 * 2
        assert "Rot" not in summary.gate_counts


class TestCounts:
    @pytest.mark.parametrize("n_qubits, n_layers", [(2, 1), (4, 3)])
    def test_summary_counts(self, n_qubits: int, n_layers: int) -> None:
        summary = circuit_summary(QuantumEncodingLayer(n_qubits=n_qubits, n_layers=n_layers, **HE))
        assert summary.n_trainable_params == n_qubits * n_layers
        assert summary.n_two_qubit_gates == (n_qubits - 1) * n_layers
        assert summary.gate_counts["RY"] == n_qubits * n_layers

    @pytest.mark.parametrize("readout", ["all", "first"])
    def test_inert_count_is_the_numerically_dead_parameters(self, readout: Readout) -> None:
        # count_inert_parameters is structural; check it against the
        # parameters whose gradient is zero at every one of several random
        # points (inputs and weights).  With readout="all" every RY reaches a
        # measured qubit.  With "first", Z_0's light cone shrinks by one qubit
        # per layer counted back from the readout: CZ is diagonal, so a
        # neighbour acts on the qubit before it only through its Z
        # populations, which an RY must have set a layer earlier.  Counted
        # back, layer k leaves max(0, n - 1 - k) RY angles dead.
        n, n_layers = 4, 3
        layer = QuantumEncodingLayer(n_qubits=n, n_layers=n_layers, readout=readout, **HE)
        grads = []
        for seed in range(5):
            _random_weights(layer, seed)
            x = torch.rand(3, n, generator=torch.Generator().manual_seed(100 + seed)) * 6 - 3
            layer.qlayer.weights.grad = None
            layer(x).sum().backward()
            grads.append(layer.qlayer.weights.grad.abs())
        dead = int((torch.stack(grads).amax(0) < 1e-7).sum())
        assert dead == circuit_summary(layer).n_inert_params
        expected = 0 if readout == "all" else sum(max(0, n - 1 - k) for k in range(n_layers))
        assert dead == expected


class TestGradients:
    def test_every_weight_gets_a_gradient_and_parameter_shift_agrees(self) -> None:
        torch.manual_seed(0)
        backprop = QuantumEncodingLayer(n_qubits=3, n_layers=2, **HE)
        shift = QuantumEncodingLayer(
            n_qubits=3,
            n_layers=2,
            entangler="hardware_efficient",
            device_name="default.qubit",
            diff_method="parameter-shift",
        )
        _random_weights(backprop, 4)
        shift.load_state_dict(backprop.state_dict())
        x = torch.rand(4, 3, generator=torch.Generator().manual_seed(5))
        g_b = torch.autograd.grad(backprop(x).sum(), backprop.qlayer.weights)[0]
        g_s = torch.autograd.grad(shift(x).sum(), shift.qlayer.weights)[0]
        assert (g_b.abs() > 1e-6).all()
        torch.testing.assert_close(g_s, g_b, atol=2e-6, rtol=0)


@pytest.mark.parametrize(
    "cls, extra",
    [
        (HybridBinaryClassifier, {}),
        (ParallelHybridClassifier, {}),
        (MulticlassHybridClassifier, {"n_classes": 3}),
    ],
)
class TestClassifiers:
    @pytest.mark.parametrize("init_strategy", ["restricted", "block_local", "normal"])
    def test_every_init_strategy_fills_the_two_dimensional_weights(
        self, cls: type, extra: dict[str, Any], init_strategy: str
    ) -> None:
        torch.manual_seed(0)
        model = cls(
            n_input_features=5, n_qubits=3, n_layers=2, init_strategy=init_strategy, **HE, **extra
        )
        w = model.quantum_layer.qlayer.weights
        assert tuple(w.shape) == (2, 3) and w.std() > 0

    def test_trains_and_round_trips_a_checkpoint(
        self, cls: type, extra: dict[str, Any], tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        model = cls(n_input_features=5, n_qubits=3, n_layers=2, **HE, **extra)
        x = torch.randn(8, 5)
        before = model.quantum_layer.qlayer.weights.detach().clone()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        opt.zero_grad()
        model(x).pow(2).mean().backward()
        opt.step()
        assert not torch.equal(model.quantum_layer.qlayer.weights, before)
        save_checkpoint(model, tmp_path / "m.pt")
        loaded = load_checkpoint(tmp_path / "m.pt")
        assert loaded.get_config()["entangler"] == "hardware_efficient"
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model.eval()(x), rtol=0, atol=0)


def test_diagnostics_measure_the_block() -> None:
    layer = QuantumEncodingLayer(n_qubits=3, n_layers=2, **HE)
    result = gradient_variance(layer, n_samples=8)
    assert result.per_parameter.shape == (2, 3) and result.n_layers == 2
    dimension = effective_dimension(layer, torch.rand(4, 3), n_data=100, n_theta_samples=2)
    assert 0 < dimension.effective_dimension <= 6
