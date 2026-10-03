"""
tests/test_data_reuploading.py
==============================
Unit tests for hqnn_forge.encoding.data_reuploading.DataReuploadingLayer.

The numerical checks pin what re-uploading is for: a single upload makes every
output a degree-one trigonometric polynomial in each feature, and ``L``
uploads raise that degree to ``L`` (Schuld, Sweke & Meyer 2021).  The Fourier
test below measures that degree directly, so a regression that silently drops
back to one upload fails it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import (
    DataReuploadingLayer,
    QuantumEncodingLayer,
    build_data_reuploading_qnode,
)

N_QUBITS = 3
N_LAYERS = 2
BATCH = 5


def _layer(
    n_layers: int = N_LAYERS, diff_method: str = "backprop", **kwargs: object
) -> DataReuploadingLayer:
    torch.manual_seed(0)
    return DataReuploadingLayer(
        n_qubits=N_QUBITS,
        n_layers=n_layers,
        device_name="default.qubit",
        diff_method=diff_method,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _random_batch(n: int = BATCH, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.rand(n, N_QUBITS, generator=generator) * 2 * math.pi - math.pi


def _loss_weights(n: int = BATCH) -> torch.Tensor:
    """Distinct weight per (sample, qubit): a plain sum would hide a permutation of samples."""
    return torch.linspace(0.1, 1.0, n * N_QUBITS).reshape(n, N_QUBITS)


# ---------------------------------------------------------------------------
# Shape, range, parameters
# ---------------------------------------------------------------------------


class TestForwardPassShape:
    def test_output_shape(self) -> None:
        assert _layer()(_random_batch()).shape == (BATCH, N_QUBITS)

    def test_single_sample(self) -> None:
        assert _layer()(_random_batch(1)).shape == (1, N_QUBITS)

    def test_expectation_values_in_range(self) -> None:
        with torch.no_grad():
            out = _layer()(_random_batch())
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_parameter_count_matches_angle_encoder_by_default(self) -> None:
        layer = _layer()
        assert sum(p.numel() for p in layer.parameters()) == N_LAYERS * N_QUBITS * 3
        assert f"n_params={N_LAYERS * N_QUBITS * 3}" in layer.extra_repr()

    def test_trainable_input_scaling_adds_one_parameter_per_upload_and_qubit(self) -> None:
        layer = _layer(trainable_input_scaling=True)
        expected = N_LAYERS * N_QUBITS * 3 + N_LAYERS * N_QUBITS
        assert sum(p.numel() for p in layer.parameters()) == expected
        assert layer.qlayer.input_scaling.shape == (N_LAYERS, N_QUBITS)
        assert torch.equal(layer.qlayer.input_scaling.detach(), torch.ones(N_LAYERS, N_QUBITS))
        assert f"n_params={expected}" in layer.extra_repr()

    def test_z_rotation_scales_every_upload_but_the_first(self) -> None:
        """The first RZ upload is a global phase, so it gets no scaling row."""
        layer = _layer(n_layers=3, rotation="Z", trainable_input_scaling=True)
        assert layer.qlayer.input_scaling.shape == (2, N_QUBITS)
        assert f"n_params={3 * N_QUBITS * 3 + 2 * N_QUBITS}" in layer.extra_repr()

    def test_readout_first_returns_qubit_zero_only(self) -> None:
        full = _layer()
        first = _layer(readout="first")
        assert first.n_outputs == 1 and full.n_outputs == N_QUBITS
        with torch.no_grad():
            first.qlayer.weights.copy_(full.qlayer.weights)
        x = _random_batch()
        with torch.no_grad():
            out = first(x)
            assert out.shape == (BATCH, 1)
            torch.testing.assert_close(out, full(x)[:, :1], rtol=0, atol=1e-6)
        assert "readout='first'" in first.extra_repr()


# ---------------------------------------------------------------------------
# Circuit structure
# ---------------------------------------------------------------------------


class TestCircuitStructure:
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_one_upload_per_layer(self, n_layers: int) -> None:
        """The tape holds n_layers AngleEmbedding ops, one before each block."""
        layer = _layer(n_layers)
        tape = qml.workflow.construct_tape(layer.qlayer.qnode)(
            _random_batch(1)[0], layer.qlayer.weights.detach()
        )
        names = [op.name for op in tape.operations]
        assert names.count("AngleEmbedding") == n_layers
        assert names.count("CNOT") == n_layers * N_QUBITS
        assert names.count("Rot") == n_layers * N_QUBITS
        # Order within each layer: embedding, ring, rotations.
        per_layer = ["AngleEmbedding"] + ["CNOT"] * N_QUBITS + ["Rot"] * N_QUBITS
        assert names == per_layer * n_layers

    @pytest.mark.parametrize("entangler", ["ring", "strongly_entangling"])
    def test_single_layer_equals_angle_encoder(self, entangler: str) -> None:
        """With one upload the circuit is QuantumEncodingLayer gate for gate."""
        ours = _layer(n_layers=1, entangler=entangler)
        reference = QuantumEncodingLayer(
            n_qubits=N_QUBITS,
            n_layers=1,
            device_name="default.qubit",
            diff_method="backprop",
            entangler=entangler,  # type: ignore[arg-type]
        )
        with torch.no_grad():
            reference.qlayer.weights.copy_(ours.qlayer.weights)
        x = _random_batch()
        with torch.no_grad():
            torch.testing.assert_close(ours(x), reference(x), rtol=0, atol=1e-6)

    @pytest.mark.parametrize("entangler", ["ring", "strongly_entangling"])
    def test_zero_input_reduces_to_the_angle_encoders_ansatz(self, entangler: str) -> None:
        """
        At x = 0 every RX upload is the identity, so the circuit is the angle
        encoder's ansatz alone.  Three layers on three qubits make the
        strongly-entangling ranges wrap (1, 2, 1), so this pins that the
        per-upload blocks carry the whole ansatz's layer index.
        """
        ours = _layer(n_layers=3, entangler=entangler)
        reference = QuantumEncodingLayer(
            n_qubits=N_QUBITS,
            n_layers=3,
            device_name="default.qubit",
            diff_method="backprop",
            entangler=entangler,  # type: ignore[arg-type]
        )
        with torch.no_grad():
            ours.qlayer.weights.uniform_(0, 2 * math.pi)
            reference.qlayer.weights.copy_(ours.qlayer.weights)
            x = torch.zeros(2, N_QUBITS)
            torch.testing.assert_close(ours(x), reference(x), rtol=0, atol=1e-6)

    def test_two_layers_differ_from_single_upload(self) -> None:
        """Same weights, same depth: only the extra upload distinguishes them."""
        ours = _layer(n_layers=2)
        reference = QuantumEncodingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
        )
        with torch.no_grad():
            reference.qlayer.weights.copy_(ours.qlayer.weights)
        x = _random_batch()
        with torch.no_grad():
            assert not torch.allclose(ours(x), reference(x), atol=1e-3)

    @pytest.mark.parametrize("trainable_input_scaling", [False, True])
    def test_matches_explicit_reference_circuit(self, trainable_input_scaling: bool) -> None:
        layer = _layer(trainable_input_scaling=trainable_input_scaling)
        weights = layer.qlayer.weights.detach()
        scaling = (
            torch.rand(N_LAYERS, N_QUBITS) * 2
            if trainable_input_scaling
            else torch.ones(N_LAYERS, N_QUBITS)
        )
        if trainable_input_scaling:
            with torch.no_grad():
                layer.qlayer.input_scaling.copy_(scaling)

        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev, interface="torch")
        def reference(x: torch.Tensor) -> list:
            for lyr in range(N_LAYERS):
                for q in range(N_QUBITS):
                    qml.RX(scaling[lyr, q] * x[q], wires=q)
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
                for q in range(N_QUBITS):
                    qml.Rot(weights[lyr, q, 0], weights[lyr, q, 1], weights[lyr, q, 2], wires=q)
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        x = _random_batch()
        with torch.no_grad():
            got = layer(x)
            want = torch.stack([torch.stack(reference(x[i])) for i in range(BATCH)])
        torch.testing.assert_close(got, want.to(got.dtype), rtol=1e-6, atol=1e-6)

    def test_z_rotation_matches_explicit_reference_circuit(self) -> None:
        """input_scaling row r scales upload r + 1; upload 0 sees the raw features."""
        n_layers = 3
        layer = _layer(n_layers=n_layers, rotation="Z", trainable_input_scaling=True)
        weights = layer.qlayer.weights.detach()
        scaling = torch.rand(n_layers - 1, N_QUBITS) * 2
        with torch.no_grad():
            layer.qlayer.input_scaling.copy_(scaling)

        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev, interface="torch")
        def reference(x: torch.Tensor) -> list:
            for lyr in range(n_layers):
                for q in range(N_QUBITS):
                    qml.RZ(x[q] if lyr == 0 else scaling[lyr - 1, q] * x[q], wires=q)
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
                for q in range(N_QUBITS):
                    qml.Rot(weights[lyr, q, 0], weights[lyr, q, 1], weights[lyr, q, 2], wires=q)
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        x = _random_batch()
        with torch.no_grad():
            got = layer(x)
            want = torch.stack([torch.stack(reference(x[i])) for i in range(BATCH)])
        torch.testing.assert_close(got, want.to(got.dtype), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Expressivity: Fourier degree grows with the number of uploads
# ---------------------------------------------------------------------------


def _fourier_magnitudes(model: torch.nn.Module, n_points: int = 16) -> torch.Tensor:
    """
    Fourier magnitudes |c_k|, k = 0 … n_points/2, of the outputs as functions
    of feature 0 with the other features held fixed, sampled on a uniform grid
    over one period; the maximum over the n_qubits outputs is taken per k.

    The maximum matters: with one layer, ⟨Z_0⟩ does not depend on feature 0
    at all (the cascaded CNOT ring maps Z_0 to Z_1⋯Z_{n-1} in the Heisenberg
    picture, and RX-embedded |0⟩ has ⟨X⟩ = 0), so probing a single output
    would under-count the degree.
    """
    fixed = _random_batch(1, generator=torch.Generator().manual_seed(1))
    grid = torch.arange(n_points, dtype=torch.float32) * 2 * math.pi / n_points - math.pi
    x = fixed.repeat(n_points, 1)
    x[:, 0] = grid
    with torch.no_grad():
        values = model(x).to(torch.float64)  # (n_points, n_qubits)
    return (torch.fft.rfft(values, dim=0).abs() / n_points).max(dim=1).values


class TestFourierDegree:
    FLOOR = 1e-3  # a present harmonic is far above float32 noise; an absent one far below

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_single_upload_has_degree_one(self, n_layers: int) -> None:
        """Sanity check of the probe: the angle encoder has no k ≥ 2 harmonics at any depth."""
        reference = QuantumEncodingLayer(
            n_qubits=N_QUBITS,
            n_layers=n_layers,
            device_name="default.qubit",
            diff_method="backprop",
        )
        torch.manual_seed(0)
        with torch.no_grad():
            reference.qlayer.weights.uniform_(0, 2 * math.pi)
        mags = _fourier_magnitudes(reference)
        assert mags[1] > self.FLOOR
        assert mags[2:].max() < self.FLOOR

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    @pytest.mark.parametrize("rotation", ["X", "Y"])
    def test_degree_equals_number_of_uploads(self, n_layers: int, rotation: str) -> None:
        self._assert_degree(_layer(n_layers, rotation=rotation), n_layers)

    @pytest.mark.parametrize("n_layers", [2, 3])
    def test_z_rotation_loses_the_first_upload(self, n_layers: int) -> None:
        """RZ on |0⟩ is a global phase, so L Z-uploads give degree L - 1."""
        self._assert_degree(_layer(n_layers, rotation="Z"), n_layers - 1)

    def _assert_degree(self, layer: DataReuploadingLayer, degree: int) -> None:
        torch.manual_seed(0)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        mags = _fourier_magnitudes(layer)
        assert mags[degree] > self.FLOOR, f"harmonic {degree} missing: {mags}"
        assert mags[degree + 1 :].max() < self.FLOOR, f"harmonics above {degree}: {mags}"


# ---------------------------------------------------------------------------
# Batching and gradients
# ---------------------------------------------------------------------------


def _per_sample(layer: DataReuploadingLayer, x: torch.Tensor) -> torch.Tensor:
    """One unbroadcasted 1-D call per sample: the reference the batched path must match."""
    return torch.stack([layer.qlayer(sample) for sample in x])


class TestBatchedMatchesPerSample:
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    @pytest.mark.parametrize("trainable_input_scaling", [False, True])
    def test_outputs_and_gradients(self, diff_method: str, trainable_input_scaling: bool) -> None:
        layer = _layer(diff_method=diff_method, trainable_input_scaling=trainable_input_scaling)
        if trainable_input_scaling:
            with torch.no_grad():
                layer.qlayer.input_scaling.copy_(torch.rand(N_LAYERS, N_QUBITS) + 0.5)
        x = _random_batch()
        loss_weights = _loss_weights()

        def run(forward: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, ...]:
            layer.zero_grad()
            xi = x.clone().requires_grad_(True)
            out = forward(xi)
            (out * loss_weights).sum().backward()
            grads = []
            for p in layer.parameters():
                assert p.grad is not None
                grads.append(p.grad.clone())
            assert xi.grad is not None
            return (out.detach(), xi.grad, *grads)

        batched = run(layer)
        looped = run(lambda xi: _per_sample(layer, xi))
        for got, want in zip(batched, looped, strict=True):
            assert got.abs().sum() > 0
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


class TestGradientFlow:
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_gradients_reach_weights_and_inputs(self, diff_method: str) -> None:
        layer = _layer(diff_method=diff_method)
        x = _random_batch().requires_grad_(True)
        (layer(x) * _loss_weights()).sum().backward()
        assert layer.qlayer.weights.grad is not None
        assert layer.qlayer.weights.grad.abs().sum().item() > 0.0
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # Every sample's input must receive gradient, not just the batch total.
        assert (x.grad.abs().sum(dim=1) > 0).all()

    def test_gradients_reach_input_scaling(self) -> None:
        layer = _layer(trainable_input_scaling=True)
        (layer(_random_batch()) * _loss_weights()).sum().backward()
        grad = layer.qlayer.input_scaling.grad
        assert grad is not None
        assert grad.abs().sum().item() > 0.0

    @pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
    def test_every_input_scaling_entry_gets_gradient(self, rotation: str) -> None:
        """
        No dead scaling parameters: checked per entry, since a whole-tensor sum
        would hide a row that never reaches the outputs (as a scale on the
        first RZ upload would not).
        """
        layer = _layer(n_layers=3, rotation=rotation, trainable_input_scaling=True)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        x = _random_batch(generator=torch.Generator().manual_seed(2))
        (layer(x) * _loss_weights()).sum().backward()
        assert (layer.qlayer.input_scaling.grad.abs() > 1e-6).all()

    @pytest.mark.parametrize(
        "device_name, diff_method",
        [
            pytest.param("default.qubit", "parameter-shift", id="parameter-shift"),
            pytest.param("lightning.qubit", "adjoint", id="lightning-adjoint"),
        ],
    )
    def test_z_scaling_matches_backprop_on_every_method(
        self, device_name: str, diff_method: str
    ) -> None:
        """
        backprop is pinned to the explicit RZ reference circuit above; every
        other method must agree with it in outputs and in all gradients, so a
        scaling row mis-indexed on one execution path cannot hide there.
        """
        n_layers = 3
        ref = _layer(n_layers=n_layers, rotation="Z", trainable_input_scaling=True)
        torch.manual_seed(0)
        other = DataReuploadingLayer(
            n_qubits=N_QUBITS,
            n_layers=n_layers,
            rotation="Z",
            device_name=device_name,  # type: ignore[arg-type]
            diff_method=diff_method,  # type: ignore[arg-type]
            trainable_input_scaling=True,
        )
        scaling = torch.linspace(0.5, 1.5, (n_layers - 1) * N_QUBITS).reshape(n_layers - 1, -1)
        with torch.no_grad():
            other.qlayer.weights.copy_(ref.qlayer.weights)
            ref.qlayer.input_scaling.copy_(scaling)
            other.qlayer.input_scaling.copy_(scaling)
        x = _random_batch()
        outs, grads = [], []
        for layer in (ref, other):
            xi = x.clone().requires_grad_(True)
            out = layer(xi)
            (out * _loss_weights()).sum().backward()
            outs.append(out.detach())
            grads.append((xi.grad, layer.qlayer.weights.grad, layer.qlayer.input_scaling.grad))
        torch.testing.assert_close(outs[1], outs[0].to(outs[1].dtype), rtol=1e-5, atol=1e-6)
        for got, want in zip(grads[1], grads[0], strict=True):
            torch.testing.assert_close(got, want.to(got.dtype), rtol=1e-4, atol=1e-5)

    def test_parameter_shift_matches_backprop(self) -> None:
        """Exact gradients agree across methods, including through the uploads."""
        a = _layer(diff_method="backprop", trainable_input_scaling=True)
        b = _layer(diff_method="parameter-shift", trainable_input_scaling=True)
        with torch.no_grad():
            b.qlayer.weights.copy_(a.qlayer.weights)
            scaling = torch.rand(N_LAYERS, N_QUBITS) + 0.5
            a.qlayer.input_scaling.copy_(scaling)
            b.qlayer.input_scaling.copy_(scaling)
        x = _random_batch()
        xa = x.clone().requires_grad_(True)
        xb = x.clone().requires_grad_(True)
        (a(xa) * _loss_weights()).sum().backward()
        (b(xb) * _loss_weights()).sum().backward()
        torch.testing.assert_close(
            a.qlayer.weights.grad, b.qlayer.weights.grad, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            a.qlayer.input_scaling.grad, b.qlayer.input_scaling.grad, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(xa.grad, xb.grad, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wrong_feature_dim_raises(self) -> None:
        with pytest.raises(ValueError, match=f"n_qubits={N_QUBITS}"):
            _layer()(torch.rand(BATCH, N_QUBITS + 1))

    def test_n_qubits_lt_2_raises(self) -> None:
        with pytest.raises(ValueError, match="n_qubits must be"):
            build_data_reuploading_qnode(n_qubits=1, device_name="default.qubit")

    def test_n_layers_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="n_layers must be"):
            build_data_reuploading_qnode(n_qubits=2, n_layers=0, device_name="default.qubit")

    @pytest.mark.parametrize("rotation", ["x", "W"])
    def test_invalid_rotation_raises_at_construction(self, rotation: str) -> None:
        with pytest.raises(ValueError, match="rotation must be"):
            _layer(rotation=rotation)

    def test_invalid_entangler_raises(self) -> None:
        with pytest.raises(ValueError, match="entangler must be"):
            _layer(entangler="ladder")

    def test_invalid_readout_raises(self) -> None:
        with pytest.raises(ValueError, match="readout must be"):
            _layer(readout="last")

    def test_single_z_upload_raises(self) -> None:
        """One RZ upload on |0⟩ is a global phase: the outputs would ignore the inputs."""
        with pytest.raises(ValueError, match='rotation="Z" needs n_layers'):
            build_data_reuploading_qnode(
                n_qubits=2, n_layers=1, rotation="Z", device_name="default.qubit"
            )
