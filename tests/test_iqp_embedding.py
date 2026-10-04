"""
tests/test_iqp_embedding.py
===========================
Unit tests for hqnn_forge.encoding.iqp_embedding.IQPEncodingLayer.
"""

from __future__ import annotations

import math

import pytest
import torch

from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers.restricted_variance import restricted_normal_init_


class TestIQPEncodingLayer:
    def test_forward_shape_and_bounds(self) -> None:
        batch_size = 4
        n_qubits = 6
        n_layers = 2

        layer = IQPEncodingLayer(n_qubits=n_qubits, n_layers=n_layers)
        # Apply restricted initialization
        restricted_normal_init_(layer.qlayer.weights, n_qubits=n_qubits, n_layers=n_layers)

        x = torch.randn(batch_size, n_qubits)
        out = layer(x)

        assert out.shape == (batch_size, n_qubits)
        # Pauli-Z expectations must be in [-1, 1]
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_mismatched_feature_dim_raises(self) -> None:
        layer = IQPEncodingLayer(n_qubits=4)
        x_wrong = torch.randn(2, 5)
        with pytest.raises(ValueError, match="does not match n_qubits=4"):
            layer(x_wrong)

    def test_gradients_flow(self) -> None:
        layer = IQPEncodingLayer(n_qubits=3, n_layers=1)
        restricted_normal_init_(layer.qlayer.weights, n_qubits=3, n_layers=1)

        x = torch.randn(2, 3, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        # Check gradients flow back to the inputs
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0

        # Check gradients flow to the weights
        weights = layer.qlayer.weights
        assert weights.grad is not None
        assert weights.grad.abs().sum().item() > 0.0


class TestExplicitDecompositionMatchesTemplate:
    """
    The circuit writes qml.IQPEmbedding out gate by gate (with MultiRZ as
    CNOT·RZ·CNOT) so that a broadcasted batch only passes through
    single-parameter gates.
    Pin that it is still the same feature map, sample by sample.
    """

    @pytest.mark.parametrize("n_repeats", [1, 2])
    def test_matches_qml_iqp_embedding(self, n_repeats: int) -> None:
        import pennylane as qml

        from hqnn_forge.encoding.iqp_embedding import build_iqp_qnode

        n_qubits, n_layers = 4, 1
        ours = build_iqp_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_repeats=n_repeats,
            device_name="default.qubit",
            diff_method="backprop",
        )

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def reference(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            qml.IQPEmbedding(inputs, wires=range(n_qubits), n_repeats=n_repeats, pattern=None)
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
                for q in range(n_qubits):
                    qml.Rot(
                        weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q
                    )
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        torch.manual_seed(0)
        weights = torch.randn(n_layers, n_qubits, 3, dtype=torch.float64)
        for _ in range(3):
            x = torch.rand(n_qubits, dtype=torch.float64) * 2 * math.pi - math.pi
            with torch.no_grad():
                got = torch.stack(ours(x, weights))
                want = torch.stack(reference(x, weights))
            torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


class TestExplicitDecompositionUnderNoise:
    """
    Noiselessly the explicit CNOT·RZ·CNOT form and qml.IQPEmbedding agree
    (above), but qml.noise.insert reads the gates as written: 5 channels per
    ZZ term here, 2 for a template MultiRZ.  Pin the explicit form's noise
    model, so a switch to the template cannot pass silently (#230).
    """

    @pytest.mark.parametrize(("n_qubits", "n_repeats"), [(2, 1), (3, 1), (3, 2), (4, 1)])
    def test_channel_count_of_the_noisy_circuit(self, n_qubits: int, n_repeats: int) -> None:
        import pennylane as qml

        from hqnn_forge.noise import _noisy_qnode

        layer = IQPEncodingLayer(
            n_qubits=n_qubits,
            n_layers=1,
            n_repeats=n_repeats,
            device_name="default.qubit",
            diff_method="backprop",
        )
        noisy = _noisy_qnode(layer.qlayer.qnode, n_qubits, 0.05, "all", "depolarizing")
        weights = torch.zeros(1, n_qubits, 3, dtype=torch.float64)
        tape = qml.workflow.construct_tape(noisy, level="user")(
            torch.rand(n_qubits, dtype=torch.float64), weights
        )
        channels = sum(op.name == "DepolarizingChannel" for op in tape.operations)
        pairs = math.comb(n_qubits, 2)
        # Embedding: H and RZ on each wire, then CNOT·RZ·CNOT (2 + 1 + 2) per
        # pair, per repeat.  Ring ansatz: n two-wire CNOTs and n Rots.
        embedding = (2 * n_qubits + 5 * pairs) * n_repeats
        ansatz = 3 * n_qubits
        assert channels == embedding + ansatz
        assert not any(op.name == "MultiRZ" for op in tape.operations)

    def test_noisy_outputs_differ_from_the_template(self) -> None:
        import pennylane as qml

        from hqnn_forge.encoding.angle_embedding import apply_variational_layers, measure_z
        from hqnn_forge.noise import apply_depolarizing_noise

        n = 3
        x = torch.tensor([[0.3, -1.1, 0.7]], dtype=torch.float64)
        torch.manual_seed(0)
        layer = IQPEncodingLayer(
            n_qubits=n, n_layers=1, device_name="default.qubit", diff_method="backprop"
        )
        weights = layer.qlayer.weights.detach().to(torch.float64)
        dev = qml.device("default.mixed", wires=n)

        @qml.qnode(dev, interface="torch")
        def template(inputs: torch.Tensor) -> list:
            qml.IQPEmbedding(inputs, wires=range(n))
            apply_variational_layers(weights, n, 1)
            return measure_z(n)

        with apply_depolarizing_noise(layer, 0.05), torch.no_grad():
            explicit = layer(x)[0].to(torch.float64)
        noisy_template = qml.noise.insert(template, qml.DepolarizingChannel, 0.05, position="all")
        with torch.no_grad():
            reference = torch.stack(noisy_template(x[0]))
            noiseless = torch.stack(template(x[0]))
            torch.testing.assert_close(layer(x)[0].to(torch.float64), noiseless, rtol=0, atol=1e-6)
        # Same unitary, different noise model: the extra channels matter.
        assert (explicit - reference).abs().max() > 0.02
