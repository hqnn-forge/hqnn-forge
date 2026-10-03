"""
tests/test_amplitude_embedding.py
=================================
Unit tests for hqnn_forge.encoding.amplitude_embedding.AmplitudeEncodingLayer.

Beyond shape and gradient smoke tests, the numerical checks pin the encoding
itself: with the variational weights at zero the ansatz is a fixed permutation
of the computational basis (the CNOT ring), so the ⟨Z_i⟩ of a basis-state
input can be computed by hand, and the L2 normalisation makes the output
invariant to the scale of the input.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import AmplitudeEncodingLayer, build_amplitude_qnode

N_QUBITS = 3
N_AMPLITUDES = 2**N_QUBITS
N_LAYERS = 1
BATCH = 5


def _layer(
    diff_method: str = "backprop",
    device_name: str = "default.qubit",
    n_layers: int = N_LAYERS,
    **kwargs: object,
) -> AmplitudeEncodingLayer:
    torch.manual_seed(0)
    return AmplitudeEncodingLayer(
        n_qubits=N_QUBITS,
        n_layers=n_layers,
        device_name=device_name,  # type: ignore[arg-type]
        diff_method=diff_method,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Shape and range
# ---------------------------------------------------------------------------


class TestForwardPassShape:
    def test_full_width_input(self) -> None:
        """2**n_qubits features in, n_qubits expectation values out."""
        layer = _layer()
        out = layer(torch.randn(BATCH, N_AMPLITUDES))
        assert out.shape == (BATCH, N_QUBITS)

    def test_padded_input(self) -> None:
        """n_features < 2**n_qubits is zero-padded, output shape unchanged."""
        layer = _layer(n_features=5)
        out = layer(torch.randn(BATCH, 5))
        assert out.shape == (BATCH, N_QUBITS)

    def test_single_sample(self) -> None:
        layer = _layer()
        assert layer(torch.randn(1, N_AMPLITUDES)).shape == (1, N_QUBITS)

    def test_expectation_values_in_range(self) -> None:
        layer = _layer()
        with torch.no_grad():
            out = layer(torch.randn(BATCH, N_AMPLITUDES))
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_extra_repr_lists_n_features(self) -> None:
        assert "n_features=5" in _layer(n_features=5).extra_repr()


# ---------------------------------------------------------------------------
# Numerical correctness of the encoding
# ---------------------------------------------------------------------------


def _cnot_ring_permutation(bits: list[int]) -> list[int]:
    """Apply CNOT(i → i+1 mod n) for i = 0 … n-1 to a bit string, wire 0 first."""
    bits = list(bits)
    n = len(bits)
    for i in range(n):
        bits[(i + 1) % n] ^= bits[i]
    return bits


def _basis_index_to_bits(k: int, n: int) -> list[int]:
    """PennyLane orders amplitudes with wire 0 as the most significant bit."""
    return [(k >> (n - 1 - i)) & 1 for i in range(n)]


def _apply_1q(state: np.ndarray, gate: np.ndarray, wire: int) -> np.ndarray:
    state = np.moveaxis(state, wire, 0)
    state = np.tensordot(gate, state, axes=([1], [0]))
    return np.moveaxis(state, 0, wire)


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    state = state.copy()
    idx: list[slice | int] = [slice(None)] * state.ndim
    idx[control] = 1
    sub = state[tuple(idx)]
    # After fixing the control axis, the target axis shifts down by one if it came later.
    t = target - 1 if target > control else target
    state[tuple(idx)] = np.flip(sub, axis=t)
    return state


def _rot(phi: float, theta: float, omega: float) -> np.ndarray:
    """PennyLane's Rot(φ, θ, ω) = RZ(ω) · RY(θ) · RZ(φ)."""

    def rz(a: float) -> np.ndarray:
        return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])

    c, s = np.cos(theta / 2), np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]])
    return rz(omega) @ ry @ rz(phi)


def _numpy_expvals(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """⟨Z_i⟩ after amplitude-embedding x/‖x‖ and the CNOT-ring + Rot layers."""
    n = weights.shape[1]
    state = (x / np.linalg.norm(x)).astype(complex).reshape((2,) * n)  # wire 0 = axis 0 = MSB
    for layer_weights in weights:
        for q in range(n):
            state = _apply_cnot(state, q, (q + 1) % n)
        for q in range(n):
            state = _apply_1q(state, _rot(*layer_weights[q]), q)
    probs = np.abs(state) ** 2
    return np.array(
        [np.moveaxis(probs, q, 0)[0].sum() - np.moveaxis(probs, q, 0)[1].sum() for q in range(n)]
    )


class TestEncodingMatchesHandComputation:
    def test_basis_states_with_zero_weights(self) -> None:
        """
        With all Rot angles at zero, one layer is exactly the CNOT ring, which
        maps basis state |k⟩ to basis state |π(k)⟩.  ⟨Z_i⟩ is then
        1 - 2·bit_i(π(k)), computed here without PennyLane.
        """
        layer = _layer()
        with torch.no_grad():
            layer.qlayer.weights.zero_()
        x = torch.eye(N_AMPLITUDES)  # every basis state, one per row
        with torch.no_grad():
            got = layer(x)
        expected = torch.tensor(
            [
                [1.0 - 2.0 * b for b in _cnot_ring_permutation(_basis_index_to_bits(k, N_QUBITS))]
                for k in range(N_AMPLITUDES)
            ]
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)

    def test_superposition_with_zero_weights(self) -> None:
        """
        (|000⟩ + |001⟩)/√2 → CNOT ring → (|000⟩ + |101⟩)/√2 for 3 qubits
        (|001⟩: CNOT(0,1) and CNOT(1,2) leave it, CNOT(2,0) flips wire 0), so
        ⟨Z_0⟩ = ⟨Z_2⟩ = 0 and ⟨Z_1⟩ = 1.  Also checks that un-normalised input
        (norm √2 here, and 10·√2 in the second row) is normalised.
        """
        layer = _layer()
        with torch.no_grad():
            layer.qlayer.weights.zero_()
        x = torch.zeros(2, N_AMPLITUDES)
        x[0, 0] = x[0, 1] = 1.0
        x[1, 0] = x[1, 1] = 10.0
        with torch.no_grad():
            got = layer(x)
        torch.testing.assert_close(got, torch.tensor([[0.0, 1.0, 0.0]] * 2), rtol=0, atol=1e-6)

    def test_padding_is_on_the_right(self) -> None:
        """
        n_features=1 with the single feature non-zero must prepare |0…0⟩,
        whichever value the feature has: the remaining amplitudes are zero.
        """
        layer = _layer(n_features=1)
        with torch.no_grad():
            layer.qlayer.weights.zero_()
            got = layer(torch.tensor([[1.0], [-3.5]]))
        torch.testing.assert_close(got, torch.ones(2, N_QUBITS), rtol=0, atol=1e-6)

    def test_scale_invariance_with_random_weights(self) -> None:
        """Only the direction is encoded: x and c·x give the same output."""
        layer = _layer()
        x = torch.randn(BATCH, N_AMPLITUDES)
        with torch.no_grad():
            torch.testing.assert_close(layer(x), layer(7.5 * x), rtol=0, atol=1e-6)

    def test_huge_scale_is_encoded(self) -> None:
        """
        A float32 norm overflows from about 1.8e19; the layer scales by the
        largest feature first, so 1e20·x still encodes x instead of NaN.
        """
        layer = _layer()
        x = torch.randn(BATCH, N_AMPLITUDES)
        with torch.no_grad():
            torch.testing.assert_close(layer(1e20 * x), layer(x), rtol=0, atol=1e-6)

    def test_matches_numpy_state_vector_with_several_layers(self) -> None:
        """
        Two layers with random non-zero weights against a NumPy state-vector
        simulation written from the gate definitions, so a wrong layer index
        in the weight slice, or layers applied in the wrong order, fails.
        """
        n_layers = 2
        layer = _layer(n_layers=n_layers).double()
        weights = layer.qlayer.weights.detach().numpy()
        assert np.abs(weights[1] - weights[0]).min() > 1e-3  # the layers really differ
        x = torch.randn(BATCH, N_AMPLITUDES, dtype=torch.float64)
        with torch.no_grad():
            got = layer(x).numpy()
        want = np.stack([_numpy_expvals(x[i].numpy(), weights) for i in range(BATCH)])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)

    def test_matches_qml_amplitude_embedding_reference(self) -> None:
        """Per-sample agreement with a reference QNode built from the template."""
        n_features = 6
        layer = _layer(n_features=n_features)
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev, interface="torch")
        def reference(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), pad_with=0.0, normalize=True)
            for lyr in range(N_LAYERS):
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
                for q in range(N_QUBITS):
                    qml.Rot(weights[lyr, q, 0], weights[lyr, q, 1], weights[lyr, q, 2], wires=q)
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        weights = layer.qlayer.weights.detach()
        x = torch.randn(BATCH, n_features)
        with torch.no_grad():
            got = layer(x)
            want = torch.stack([torch.stack(reference(x[i], weights)) for i in range(BATCH)])
        torch.testing.assert_close(got, want.to(got.dtype), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatchedMatchesPerSample:
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_outputs(self, diff_method: str) -> None:
        layer = _layer(diff_method, n_features=6)
        x = torch.randn(BATCH, 6)
        with torch.no_grad():
            batched = layer(x)
            single = torch.cat([layer(x[i : i + 1]) for i in range(BATCH)])
        torch.testing.assert_close(batched, single, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_gradients_reach_quantum_weights(self, diff_method: str) -> None:
        layer = _layer(diff_method)
        layer(torch.randn(BATCH, N_AMPLITUDES)).sum().backward()
        grad = layer.qlayer.weights.grad
        assert grad is not None
        assert grad.abs().sum().item() > 0.0

    def test_gradients_reach_inputs_under_backprop(self) -> None:
        """Padding and normalisation are differentiable end to end."""
        layer = _layer("backprop", n_features=6)
        x = torch.randn(BATCH, 6, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum().item() > 0.0

    def test_input_gradient_is_orthogonal_to_the_input(self) -> None:
        """
        The output depends on x only through x/‖x‖, so the directional
        derivative along x itself is zero: ⟨∇_x f, x⟩ = 0 for every sample.
        """
        layer = _layer("backprop")
        x = torch.randn(BATCH, N_AMPLITUDES, dtype=torch.float64, requires_grad=True)
        layer.double()
        layer(x).sum().backward()
        assert x.grad is not None
        radial = (x.grad * x).sum(dim=-1)
        torch.testing.assert_close(
            radial, torch.zeros(BATCH, dtype=torch.float64), atol=1e-9, rtol=0
        )

    def test_backprop_input_gradient_near_zero_amplitude_in_float32(self) -> None:
        """
        An amplitude far below its partner is where the non-backprop methods
        return NaN; backprop differentiates the state vector instead, so its
        float32 gradient stays finite and agrees with the float64 one.
        """
        x = torch.randn(BATCH, N_AMPLITUDES, dtype=torch.float64)
        x[:, 0] = 1e-6 * x[:, 1]
        grads = []
        for dtype in (torch.float32, torch.float64):
            layer = _layer("backprop", n_layers=2).to(dtype)
            xi = x.to(dtype).requires_grad_(True)
            layer(xi).sum().backward()
            assert xi.grad is not None
            grads.append(xi.grad.double())
        assert torch.isfinite(grads[0]).all()
        torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=1e-4)

    @pytest.mark.parametrize(
        ("diff_method", "device_name"),
        [
            ("adjoint", "default.qubit"),
            ("parameter-shift", "default.qubit"),
            ("finite-diff", "default.qubit"),
            pytest.param("adjoint", "lightning.qubit", marks=pytest.mark.requires_lightning),
        ],
    )
    @pytest.mark.parametrize("n_features", [N_AMPLITUDES, 6])
    def test_input_gradient_refused_without_backprop(
        self, diff_method: str, device_name: str, n_features: int
    ) -> None:
        """
        Refused from the method alone, even for dense float64 inputs where
        the gradient would happen to be right: whether a batch contains a
        zero or small amplitude cannot be known in advance.
        """
        layer = _layer(diff_method, device_name, n_features=n_features).double()
        x = torch.randn(BATCH, n_features, dtype=torch.float64, requires_grad=True)
        with pytest.raises(RuntimeError, match="cannot differentiate with respect to its inputs"):
            layer(x)

    @pytest.mark.parametrize(
        ("diff_method", "device_name"),
        [
            ("parameter-shift", "default.qubit"),
            ("finite-diff", "default.qubit"),
            pytest.param("adjoint", "lightning.qubit", marks=pytest.mark.requires_lightning),
        ],
    )
    def test_weight_gradient_matches_backprop(self, diff_method: str, device_name: str) -> None:
        """
        The state-preparation angles do not depend on the weights, so weight
        gradients stay correct under every method, also with padding and a
        near-zero amplitude that would break the input gradient.
        """
        x = torch.randn(BATCH, 6, dtype=torch.float64)
        x[:, 0] = 1e-9 * x[:, 1]
        grads = []
        for method, device in (("backprop", "default.qubit"), (diff_method, device_name)):
            layer = _layer(method, device, n_layers=2, n_features=6).double()
            layer(x).sum().backward()
            assert layer.qlayer.weights.grad is not None
            grads.append(layer.qlayer.weights.grad)
        assert torch.isfinite(grads[1]).all()
        # finite-diff is first order in its step, so it gets a looser tolerance.
        atol = 1e-5 if diff_method == "finite-diff" else 1e-7
        torch.testing.assert_close(grads[1], grads[0], rtol=0, atol=atol)

    @pytest.mark.parametrize(
        ("diff_method", "device_name"),
        [
            ("parameter-shift", "default.qubit"),
            ("finite-diff", "default.qubit"),
            pytest.param("adjoint", "lightning.qubit", marks=requires_lightning),
        ],
    )
    def test_zero_amplitude_input_gradient_is_really_broken(
        self, diff_method: str, device_name: str
    ) -> None:
        """
        Pins the PennyLane behaviour the guard exists for: with an exactly
        zero amplitude, as zero padding produces, the non-backprop input
        gradient is NaN (PennyLane 0.45) or finite but wrong against
        backprop (0.46).  If this starts failing on some version, the guard
        may be relaxable there.
        """
        n = 3

        def make(method: str, device: str) -> Any:
            @qml.qnode(qml.device(device, wires=n), interface="torch", diff_method=method)
            def circuit(inputs: torch.Tensor) -> torch.Tensor:
                qml.AmplitudeEmbedding(inputs, wires=range(n))
                for q in range(n):
                    qml.CNOT(wires=[q, (q + 1) % n])
                for q in range(n):
                    qml.Rot(0.4 + q, 1.1 - q, 0.7 * q, wires=q)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

            return circuit

        x = torch.tensor([0.3, 0.1, 0.5, 0.2, 0.0, 0.6, 0.2, 0.1], dtype=torch.float64)
        grads = []
        for method, device in (("backprop", "default.qubit"), (diff_method, device_name)):
            xi = x.clone().requires_grad_(True)
            make(method, device)(xi / xi.norm()).backward()
            assert xi.grad is not None
            grads.append(xi.grad)
        reference, grad = grads
        assert torch.isfinite(reference).all()
        broken = not torch.isfinite(grad).all() or (grad - reference).abs().max() > 1e-2
        assert broken, f"{diff_method} input gradient matches backprop: {grad} vs {reference}"

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift", "finite-diff"])
    def test_inference_under_no_grad_is_allowed(self, diff_method: str) -> None:
        """No gradient is computed under no_grad, so nothing is refused."""
        layer = _layer(diff_method, n_features=6)
        x = torch.randn(BATCH, 6, requires_grad=True)
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (BATCH, N_QUBITS)

    def test_raw_qnode_refuses_too(self) -> None:
        """The check lives in the circuit, so direct QNode users are covered."""
        qnode = build_amplitude_qnode(
            n_qubits=N_QUBITS, n_layers=1, device_name="default.qubit", diff_method="adjoint"
        )
        x = torch.zeros(N_AMPLITUDES, requires_grad=True)
        with torch.no_grad():
            x[0] = 1.0
        with pytest.raises(RuntimeError, match="cannot differentiate with respect to its inputs"):
            qnode(x, torch.zeros(1, N_QUBITS, 3))

    def test_detached_input_is_fine_under_parameter_shift(self) -> None:
        layer = _layer("parameter-shift")
        x = torch.randn(BATCH, N_AMPLITUDES, requires_grad=True)
        layer(x.detach()).sum().backward()
        assert layer.qlayer.weights.grad is not None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wrong_feature_dim_raises(self) -> None:
        layer = _layer(n_features=6)
        with pytest.raises(ValueError, match="n_features=6"):
            layer(torch.randn(BATCH, 7))

    def test_more_features_than_amplitudes_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[1, 8\]"):
            _layer(n_features=N_AMPLITUDES + 1)

    def test_zero_features_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[1, 8\]"):
            _layer(n_features=0)

    def test_all_zero_sample_raises(self) -> None:
        layer = _layer()
        x = torch.randn(BATCH, N_AMPLITUDES)
        x[2] = 0.0
        with pytest.raises(ValueError, match="all-zero"):
            layer(x)

    @pytest.mark.parametrize("scale", [1e-13, 1e-40])
    def test_tiny_non_zero_scale_is_encoded(self, scale: float) -> None:
        """
        Only an exact zero has no direction: a tiny scale, down to float32
        subnormals (1e-40), encodes the same state as the unscaled input.
        """
        layer = _layer()
        x = torch.randn(BATCH, N_AMPLITUDES)
        with torch.no_grad():
            torch.testing.assert_close(layer(scale * x), layer(x), rtol=0, atol=1e-5)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_sample_raises(self, bad: float) -> None:
        layer = _layer()
        x = torch.randn(BATCH, N_AMPLITUDES)
        x[2, 0] = bad
        with pytest.raises(ValueError, match="NaN or ±inf"):
            layer(x)

    def test_n_qubits_lt_2_raises(self) -> None:
        with pytest.raises(ValueError, match="n_qubits must be"):
            build_amplitude_qnode(n_qubits=1, device_name="default.qubit", diff_method="backprop")

    @pytest.mark.parametrize("n_qubits", [-1, 0, 1])
    def test_layer_validates_n_qubits_first(self, n_qubits: int) -> None:
        """An invalid n_qubits is reported as such, not as a bad n_features."""
        with pytest.raises(ValueError, match="n_qubits must be"):
            AmplitudeEncodingLayer(
                n_qubits=n_qubits, device_name="default.qubit", diff_method="backprop"
            )

    def test_default_n_features_is_all_amplitudes(self) -> None:
        layer = _layer()
        assert layer.n_features == N_AMPLITUDES == 2**N_QUBITS
        assert math.log2(layer.n_amplitudes) == N_QUBITS
