"""
tests/test_kernels.py
=====================
Unit tests for hqnn_forge.kernels.

Two of the encoders have closed-form kernels, which serve as oracles:

* Angle embedding (RX, product state): ``k(x, y) = Π_i cos²((x_i − y_i) / 2)``.
* Amplitude embedding: ``k(x, y) = (x·y)² / (‖x‖² ‖y‖²)``.

The IQP and re-uploading kernels are checked two ways: against the overlap
circuit ``|⟨0| U†(x) U(y) |0⟩|²`` built from the layer's own tape, and against
states from circuits written out in this file from the documented topology,
which do not go through the replayed tape at all.  ``TestMatchesForward``
also checks that the replayed states reproduce ``layer(x)``: ⟨Z_i⟩ computed
from ``encoded_states`` must equal what the layer's own forward returns.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import pennylane as qml
import pytest
import torch

from hqnn_forge import kernels
from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    QuantumEncodingLayer,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.kernels import (
    encoded_states,
    kernel_from_states,
    kernel_target_alignment,
    quantum_kernel_matrix,
    train_kernel_alignment,
)

N_QUBITS = 3
M = 7  # samples


def _angles(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(n, N_QUBITS, dtype=torch.float64) * 2 * math.pi - math.pi


def _angle_layer(n_layers: int = 2) -> QuantumEncodingLayer:
    torch.manual_seed(0)
    return QuantumEncodingLayer(
        n_qubits=N_QUBITS, n_layers=n_layers, device_name="default.qubit", diff_method="backprop"
    )


ALL_LAYERS = [
    pytest.param(lambda: _angle_layer(), id="angle"),
    pytest.param(
        lambda: IQPEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, device_name="default.qubit", diff_method="backprop"
        ),
        id="iqp",
    ),
    pytest.param(
        lambda: AmplitudeEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, device_name="default.qubit", diff_method="backprop"
        ),
        id="amplitude",
    ),
    pytest.param(
        lambda: DataReuploadingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
        ),
        id="reuploading",
    ),
]


def _inputs_for(layer: torch.nn.Module) -> torch.Tensor:
    if isinstance(layer, AmplitudeEncodingLayer):
        torch.manual_seed(0)
        return torch.randn(M, layer.n_features, dtype=torch.float64)
    return _angles(M)


# ---------------------------------------------------------------------------
# Kernel matrix properties (the issue's acceptance criteria)
# ---------------------------------------------------------------------------


class TestKernelMatrixProperties:
    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_shape_symmetry_psd_and_unit_diagonal(self, build) -> None:
        layer = build()
        X = _inputs_for(layer)
        K = quantum_kernel_matrix(X, layer)
        assert K.shape == (M, M)
        assert K.dtype == torch.float64
        torch.testing.assert_close(K, K.T, rtol=0, atol=0)
        torch.testing.assert_close(
            K.diagonal(), torch.ones(M, dtype=torch.float64), atol=1e-10, rtol=0
        )
        eigenvalues = torch.linalg.eigvalsh(K)
        assert eigenvalues.min().item() >= -1e-10
        # No clamping in the implementation, so these bound the actual values.
        assert K.min().item() >= 0.0
        assert K.max().item() <= 1.0 + 1e-12

    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_rectangular_matrix_matches_square_blocks(self, build) -> None:
        """K(X, Y) is the off-diagonal block of K([X; Y])."""
        layer = build()
        XY = _inputs_for(layer)
        X, Y = XY[:4], XY[4:]
        full = quantum_kernel_matrix(XY, layer)
        rect = quantum_kernel_matrix(X, layer, Y=Y)
        assert rect.shape == (4, M - 4)
        torch.testing.assert_close(rect, full[:4, 4:], atol=1e-12, rtol=0)

    def test_identical_rows_give_kernel_one(self) -> None:
        layer = _angle_layer()
        X = _angles(3)
        X[2] = X[0]
        K = quantum_kernel_matrix(X, layer)
        assert K[0, 2].item() == pytest.approx(1.0, abs=1e-12)

    def test_states_are_normalised(self) -> None:
        for param in ALL_LAYERS:
            build = param.values[0]
            assert callable(build)
            layer = build()
            states = encoded_states(_inputs_for(layer), layer)
            assert states.shape == (M, 2**N_QUBITS)
            norms = torch.linalg.vector_norm(states, dim=1)
            torch.testing.assert_close(
                norms, torch.ones(M, dtype=torch.float64), atol=1e-12, rtol=0
            )


# ---------------------------------------------------------------------------
# The replayed circuit is the layer's circuit
# ---------------------------------------------------------------------------


def _z_expectations(states: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """⟨Z_i⟩ per row of a state batch; wire 0 is the most significant bit."""
    probs = states.abs().pow(2)
    index = torch.arange(2**n_qubits)
    signs = torch.stack(
        [
            1.0 - 2.0 * ((index >> (n_qubits - 1 - i)) & 1).to(torch.float64)
            for i in range(n_qubits)
        ]
    )
    return probs @ signs.T


def _reuploading_with_random_scaling(**kwargs) -> DataReuploadingLayer:
    torch.manual_seed(1)
    layer = DataReuploadingLayer(
        n_qubits=N_QUBITS, n_layers=2, trainable_input_scaling=True, **kwargs
    )
    with torch.no_grad():
        layer.qlayer.input_scaling.uniform_(0.5, 2.0)
    return layer


LIGHTNING: dict[str, Any] = {"device_name": "lightning.qubit", "diff_method": "adjoint"}

FORWARD_LAYERS = [
    *ALL_LAYERS,
    pytest.param(
        lambda: _reuploading_with_random_scaling(
            device_name="default.qubit", diff_method="backprop"
        ),
        id="reuploading-scaled",
    ),
    # lightning.qubit with adjoint (the default until #349, and what "auto"
    # picks above 12 qubits), whose QNodes are wrapped in a batch-expanding
    # transform the replay must see through.
    pytest.param(
        lambda: QuantumEncodingLayer(n_qubits=N_QUBITS, **LIGHTNING), id="angle-lightning"
    ),
    pytest.param(lambda: IQPEncodingLayer(n_qubits=N_QUBITS, **LIGHTNING), id="iqp-lightning"),
    pytest.param(
        lambda: AmplitudeEncodingLayer(n_qubits=N_QUBITS, n_features=5, **LIGHTNING),
        id="amplitude-lightning",
    ),
    pytest.param(
        lambda: _reuploading_with_random_scaling(**LIGHTNING), id="reuploading-lightning"
    ),
    # The library default, "auto": default.qubit with backprop at this size.
    pytest.param(lambda: QuantumEncodingLayer(n_qubits=N_QUBITS), id="angle-auto"),
]


class TestMatchesForward:
    @pytest.mark.parametrize("build", FORWARD_LAYERS)
    def test_states_reproduce_forward(self, build) -> None:
        layer = build()
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        X = _inputs_for(layer)
        with torch.no_grad():
            expected = layer(X.to(torch.float32)).to(torch.float64)
        actual = _z_expectations(encoded_states(X, layer), N_QUBITS)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Closed-form oracles
# ---------------------------------------------------------------------------


class TestClosedForms:
    def test_angle_kernel_is_product_of_cosines(self) -> None:
        """RX(x_i)|0⟩ is a product state, so the fidelity factorises per qubit."""
        X = _angles(M)
        K = quantum_kernel_matrix(X, _angle_layer())
        diff = X[:, None, :] - X[None, :, :]
        expected = torch.cos(diff / 2).pow(2).prod(dim=-1)
        torch.testing.assert_close(K, expected, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("n_layers", [1, 3])
    def test_angle_kernel_is_independent_of_the_ansatz(self, n_layers: int) -> None:
        """The variational unitary cancels in |⟨Φ(x)|V†V|Φ(y)⟩|²."""
        X = _angles(M)
        layer = _angle_layer(n_layers)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        K = quantum_kernel_matrix(X, layer)
        expected = torch.cos((X[:, None, :] - X[None, :, :]) / 2).pow(2).prod(dim=-1)
        torch.testing.assert_close(K, expected, atol=1e-12, rtol=0)

    def test_amplitude_kernel_is_squared_cosine_similarity(self) -> None:
        layer = AmplitudeEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, n_features=5, device_name="default.qubit"
        )
        torch.manual_seed(0)
        X = torch.randn(M, 5, dtype=torch.float64)
        K = quantum_kernel_matrix(X, layer)
        unit = X / torch.linalg.vector_norm(X, dim=1, keepdim=True)
        expected = (unit @ unit.T).pow(2)
        torch.testing.assert_close(K, expected, atol=1e-12, rtol=0)

    def test_amplitude_kernel_uses_the_layers_padding(self) -> None:
        """n_features < 2**n_qubits goes through prepare_inputs, not a raw QNode call."""
        layer = AmplitudeEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, n_features=2, device_name="default.qubit"
        )
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        K = quantum_kernel_matrix(X, layer)
        expected = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]], dtype=torch.float64
        )
        torch.testing.assert_close(K, expected, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Overlap-circuit reference for the entangling encoders
# ---------------------------------------------------------------------------


def _overlap_kernel(layer: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    """|⟨0|U†(x_i)U(x_j)|0⟩|² from the layer's own tape, via qml.adjoint."""
    qlayer = layer.qlayer
    assert isinstance(qlayer, qml.qnn.TorchLayer)
    weights = {k: p.detach().to(torch.float64) for k, p in qlayer.qnode_weights.items()}
    build = qml.workflow.construct_tape(qlayer.qnode, level=0)
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def overlap(x: torch.Tensor, y: torch.Tensor) -> qml.measurements.ProbabilityMP:
        for op in build(y, **weights).operations:
            qml.apply(op)
        for op in reversed(build(x, **weights).operations):
            qml.adjoint(op)
        return qml.probs(wires=range(N_QUBITS))

    K = torch.empty(M, M, dtype=torch.float64)
    for i in range(M):
        for j in range(M):
            K[i, j] = torch.as_tensor(overlap(X[i], X[j]))[0]
    return K


def _written_out_kernel(states: list[torch.Tensor]) -> torch.Tensor:
    S = torch.stack([torch.as_tensor(s).to(torch.complex128) for s in states])
    return (S @ S.conj().T).abs().pow(2)


def _iqp_reference(X: torch.Tensor) -> torch.Tensor:
    """qml.IQPEmbedding's own decomposition; the ansatz after it cancels."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def state(x: torch.Tensor) -> qml.measurements.StateMP:
        qml.IQPEmbedding(x, wires=range(N_QUBITS))
        return qml.state()

    return _written_out_kernel([state(x) for x in X])


def _reuploading_reference(X: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """RX(x) upload, CNOT ring, Rot on every qubit, repeated per layer."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def state(x: torch.Tensor) -> qml.measurements.StateMP:
        for layer in range(weights.shape[0]):
            for q in range(N_QUBITS):
                qml.RX(x[q], wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
            for q in range(N_QUBITS):
                qml.Rot(*weights[layer, q], wires=q)
        return qml.state()

    return _written_out_kernel([state(x) for x in X])


def _layers(*ids: str) -> list:
    selected = [p for p in ALL_LAYERS if p.id in ids]
    assert [p.id for p in selected] == list(ids)
    return selected


class TestOverlapCircuitReference:
    @pytest.mark.parametrize("build", _layers("iqp", "reuploading"))
    def test_matches_overlap_circuit(self, build) -> None:
        layer = build()
        X = _angles(M)
        K = quantum_kernel_matrix(X, layer)
        torch.testing.assert_close(K, _overlap_kernel(layer, X), atol=1e-10, rtol=0)

    def test_iqp_matches_a_written_out_circuit(self) -> None:
        layer = IQPEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, device_name="default.qubit", diff_method="backprop"
        )
        X = _angles(M)
        K = quantum_kernel_matrix(X, layer)
        torch.testing.assert_close(K, _iqp_reference(X), atol=1e-10, rtol=0)

    def test_reuploading_matches_a_written_out_circuit(self) -> None:
        layer = DataReuploadingLayer(
            n_qubits=N_QUBITS, n_layers=3, device_name="default.qubit", diff_method="backprop"
        )
        torch.manual_seed(2)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        weights = layer.qlayer.weights.detach().to(torch.float64)
        X = _angles(M)
        K = quantum_kernel_matrix(X, layer)
        torch.testing.assert_close(K, _reuploading_reference(X, weights), atol=1e-10, rtol=0)

    def test_reuploading_kernel_depends_on_the_weights(self) -> None:
        """Unlike the single-upload encoders, the weights sit between uploads."""
        X = _angles(M)
        layer = DataReuploadingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
        )
        torch.manual_seed(0)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        K1 = quantum_kernel_matrix(X, layer)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        K2 = quantum_kernel_matrix(X, layer)
        assert not torch.allclose(K1, K2, atol=1e-3)

    def test_reuploading_last_block_cancels(self) -> None:
        """weights[-1] follows the last upload, so it drops out as V†V does."""
        X = _angles(M)
        torch.manual_seed(0)
        layer = DataReuploadingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
        )
        K1 = quantum_kernel_matrix(X, layer)
        with torch.no_grad():
            layer.qlayer.weights[-1].uniform_(0, 2 * math.pi)
        torch.testing.assert_close(quantum_kernel_matrix(X, layer), K1, atol=1e-12, rtol=0)
        with torch.no_grad():
            layer.qlayer.weights[0].uniform_(0, 2 * math.pi)
        assert not torch.allclose(quantum_kernel_matrix(X, layer), K1, atol=1e-3)


# ---------------------------------------------------------------------------
# sklearn integration and validation
# ---------------------------------------------------------------------------


class TestUsage:
    def test_precomputed_svm_separates_a_toy_problem(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.svm import SVC

        torch.manual_seed(0)
        layer = _angle_layer(1)
        # Two clusters of angles, one around -π/2 and one around +π/2.
        X = torch.cat(
            [
                torch.randn(20, N_QUBITS) * 0.3 - math.pi / 2,
                torch.randn(20, N_QUBITS) * 0.3 + math.pi / 2,
            ]
        )
        y = torch.cat([torch.zeros(20), torch.ones(20)]).numpy()
        X_train, X_test = X[::2], X[1::2]
        y_train, y_test = y[::2], y[1::2]
        svm = SVC(kernel="precomputed")
        svm.fit(quantum_kernel_matrix(X_train, layer).numpy(), y_train)
        pred = svm.predict(quantum_kernel_matrix(X_test, layer, Y=X_train).numpy())
        assert (pred == y_test).mean() == 1.0

    def test_reused_states_match_the_rectangular_matrix(self) -> None:
        layer = _angle_layer()
        X_train, X_test = _angles(5, seed=0), _angles(3, seed=1)
        S_train = encoded_states(X_train, layer)
        torch.testing.assert_close(
            kernel_from_states(S_train), quantum_kernel_matrix(X_train, layer), atol=0, rtol=0
        )
        torch.testing.assert_close(
            kernel_from_states(encoded_states(X_test, layer), S_train),
            quantum_kernel_matrix(X_test, layer, Y=X_train),
            atol=0,
            rtol=0,
        )

    def test_unnormalised_state_is_not_hidden(self) -> None:
        """The diagonal is not clamped, so a bad state shows up as K[i, i] != 1."""
        states = encoded_states(_angles(3), _angle_layer())
        states[1] *= 1.1
        K = kernel_from_states(states)
        assert K[1, 1].item() == pytest.approx(1.1**4, abs=1e-12)
        assert K[0, 0].item() == pytest.approx(1.0, abs=1e-12)

    def test_does_not_touch_gradients_or_weights(self) -> None:
        layer = _angle_layer()
        before = layer.qlayer.weights.detach().clone()
        X = _angles(M).requires_grad_(True)
        K = quantum_kernel_matrix(X, layer)
        assert not K.requires_grad
        assert torch.equal(layer.qlayer.weights.detach(), before)

    def test_rejects_non_layers(self) -> None:
        with pytest.raises(TypeError, match="encoding layer"):
            quantum_kernel_matrix(_angles(2), torch.nn.Linear(3, 3))

    def test_rejects_wrong_shapes(self) -> None:
        layer = _angle_layer()
        with pytest.raises(ValueError, match="n_samples, n_features"):
            quantum_kernel_matrix(torch.zeros(N_QUBITS), layer)
        with pytest.raises(ValueError, match="no samples"):
            quantum_kernel_matrix(torch.zeros(0, N_QUBITS), layer)
        with pytest.raises(ValueError, match="state dimension"):
            kernel_from_states(torch.zeros(2, 4, dtype=torch.complex128), torch.zeros(2, 8))

    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_rejects_inputs_of_the_wrong_width_for_the_layer(self, build) -> None:
        """The layer's own width check applies, as it does in forward."""
        layer = build()
        width = _inputs_for(layer).shape[1]
        too_narrow = torch.rand(4, width - 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="does not match"):
            layer(too_narrow)
        with pytest.raises(ValueError, match="does not match"):
            quantum_kernel_matrix(too_narrow, layer)
        with pytest.raises(ValueError, match="does not match"):
            encoded_states(too_narrow, layer)
        with pytest.raises(ValueError, match="does not match"):
            quantum_kernel_matrix(_inputs_for(layer), layer, Y=too_narrow)

    def test_validates_y_before_simulating_x(self, monkeypatch) -> None:
        calls: list[int] = []
        real = kernels._simulate

        def counting(*args: Any) -> torch.Tensor:
            calls.append(1)
            return real(*args)

        monkeypatch.setattr(kernels, "_simulate", counting)
        with pytest.raises(ValueError, match="does not match"):
            quantum_kernel_matrix(_angles(2), _angle_layer(), Y=torch.zeros(2, N_QUBITS + 1))
        assert calls == []

    @pytest.mark.parametrize("build", ALL_LAYERS)
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_inputs(self, build, bad: float) -> None:
        layer = build()
        X = _inputs_for(layer)
        X[1, 0] = bad
        with pytest.raises(ValueError, match="NaN or ±inf"):
            quantum_kernel_matrix(X, layer)
        with pytest.raises(ValueError, match="NaN or ±inf"):
            quantum_kernel_matrix(_inputs_for(layer), layer, Y=X)

    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_forward_rejects_the_same_non_finite_inputs(self, build) -> None:
        layer = build()
        X = _inputs_for(layer)
        X[1, 0] = math.nan
        with pytest.raises(ValueError, match="NaN or ±inf"):
            layer(X)

    def test_refuses_a_foreign_transform_on_the_qnode(self) -> None:
        layer = _angle_layer()
        layer.qlayer.qnode = qml.transforms.cancel_inverses(layer.qlayer.qnode)
        with pytest.raises(RuntimeError, match="cancel_inverses"):
            quantum_kernel_matrix(_angles(3), layer)

    def test_allows_the_encoders_own_broadcast_expand(self) -> None:
        """Non-backprop layers carry broadcast_expand; the kernel is unchanged."""
        X = _angles(M)
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="adjoint"
        )
        assert len(layer.qlayer.qnode.compile_pipeline) == 1
        torch.testing.assert_close(
            quantum_kernel_matrix(X, layer),
            quantum_kernel_matrix(X, _angle_layer()),
            atol=1e-12,
            rtol=0,
        )

    def test_refuses_to_run_inside_the_noise_block(self) -> None:
        from hqnn_forge.noise import apply_depolarizing_noise

        layer = _angle_layer()
        X = _angles(3)
        with apply_depolarizing_noise(layer, 0.1):
            with pytest.raises(RuntimeError, match="apply_depolarizing_noise"):
                quantum_kernel_matrix(X, layer)
            with pytest.raises(RuntimeError, match="apply_depolarizing_noise"):
                encoded_states(X, layer)
        # p = 0 replaces nothing, and the layer is usable again after the block.
        with apply_depolarizing_noise(layer, 0.0):
            quantum_kernel_matrix(X, layer)
        quantum_kernel_matrix(X, layer)

    def test_promotes_lower_precision_and_real_states(self) -> None:
        S = encoded_states(_angles(M), _angle_layer())
        expected = kernel_from_states(S)
        S64 = S.to(torch.complex64)
        torch.testing.assert_close(kernel_from_states(S64, S), expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(kernel_from_states(S64), expected, atol=1e-6, rtol=0)
        real = S.real
        assert kernel_from_states(real, S).dtype == torch.float64
        torch.testing.assert_close(
            kernel_from_states(real, real), (real @ real.T).pow(2), atol=1e-12, rtol=0
        )

    def test_simulates_x_and_y_in_one_replay(self, monkeypatch) -> None:
        calls: list[int] = []
        real = kernels._simulate

        def counting(*args: Any) -> torch.Tensor:
            calls.append(1)
            return real(*args)

        monkeypatch.setattr(kernels, "_simulate", counting)
        quantum_kernel_matrix(_angles(3, seed=0), _angle_layer(), Y=_angles(4, seed=1))
        assert calls == [1]


# ---------------------------------------------------------------------------
# Trainable kernel: kernel-target alignment (#214)
# ---------------------------------------------------------------------------


def _reuploading(scaling: bool = True, n_layers: int = 3) -> DataReuploadingLayer:
    torch.manual_seed(0)
    layer = DataReuploadingLayer(
        n_qubits=N_QUBITS,
        n_layers=n_layers,
        trainable_input_scaling=scaling,
        device_name="default.qubit",
        diff_method="backprop",
    )
    return layer.to(torch.float64)


def _toy_task(n: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """Two classes that differ in the sign of feature 0 only."""
    g = torch.Generator().manual_seed(5)
    X = torch.rand(n, N_QUBITS, generator=g, dtype=torch.float64) * 2 - 1
    y = (X[:, 0] > 0).to(torch.float64)
    y[0], y[1] = 1.0, 0.0
    return X, y


class TestKernelTargetAlignment:
    def test_ideal_kernel_has_alignment_one(self) -> None:
        y = torch.tensor([1.0, -1, -1, 1, -1])
        assert float(kernel_target_alignment(torch.outer(y, y), y)) == pytest.approx(1.0)
        assert float(kernel_target_alignment(-torch.outer(y, y), y)) == pytest.approx(-1.0)

    def test_label_encodings_agree_and_centring_ignores_offsets(self) -> None:
        K = quantum_kernel_matrix(_angles(M), _angle_layer())
        y01 = torch.tensor([1, 0, 0, 1, 1, 0, 0])
        a = kernel_target_alignment(K, y01)
        assert float(a) == pytest.approx(float(kernel_target_alignment(K, 2 * y01 - 1)))
        # Centred alignment does not see a constant added to every entry.
        assert float(kernel_target_alignment(K + 3.0, y01)) == pytest.approx(float(a))

    @pytest.mark.parametrize(
        ("K", "y", "match"),
        [
            (torch.eye(3), torch.tensor([0, 1]), "3 labels"),
            (torch.eye(3), torch.tensor([1, 1, 1]), "both classes"),
            (torch.eye(3), torch.tensor([0, 1, 2]), "both classes"),
            (torch.ones(2, 3), torch.tensor([0, 1]), "square"),
        ],
    )
    def test_rejected(self, K: torch.Tensor, y: torch.Tensor, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            kernel_target_alignment(K, y)


class TestDifferentiableKernel:
    def test_values_equal_the_detached_path(self) -> None:
        layer = _reuploading()
        X = _angles(M)
        torch.testing.assert_close(
            quantum_kernel_matrix(X, layer, differentiable=True).detach(),
            quantum_kernel_matrix(X, layer),
            rtol=0,
            atol=1e-12,
        )
        assert not quantum_kernel_matrix(X, layer).requires_grad

    def test_alignment_gradient_matches_finite_differences(self) -> None:
        layer = _reuploading()
        X, y = _toy_task(8)

        def alignment() -> torch.Tensor:
            return kernel_target_alignment(quantum_kernel_matrix(X, layer, differentiable=True), y)

        alignment().backward()
        eps = 1e-6
        for name, param in layer.named_parameters():
            assert param.grad is not None, name
            flat, grad = param.data.view(-1), param.grad.view(-1)
            for i in (0, flat.numel() // 2, flat.numel() - 1):
                old = float(flat[i])
                with torch.no_grad():
                    flat[i] = old + eps
                    up = float(alignment())
                    flat[i] = old - eps
                    down = float(alignment())
                    flat[i] = old
                assert float(grad[i]) == pytest.approx((up - down) / (2 * eps), abs=1e-7), (
                    name,
                    i,
                )

    @pytest.mark.parametrize("build", _layers("angle", "iqp", "amplitude"))
    def test_single_upload_ansatz_cancels(self, build) -> None:
        layer = build().to(torch.float64)
        X = _inputs_for(layer)
        y = torch.tensor([1, 0, 1, 0, 0, 1, 0])
        kernel_target_alignment(quantum_kernel_matrix(X, layer, differentiable=True), y).backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None and param.grad.abs().max() < 1e-10, name

    def test_only_the_last_reuploading_block_cancels(self) -> None:
        layer = _reuploading(scaling=False)
        X, y = _toy_task(8)
        kernel_target_alignment(quantum_kernel_matrix(X, layer, differentiable=True), y).backward()
        grad = layer.qlayer.weights.grad
        assert grad is not None
        assert grad[-1].abs().max() < 1e-10
        assert grad[:-1].abs().max() > 1e-4


class TestTrainKernelAlignment:
    def test_alignment_increases(self) -> None:
        layer = _reuploading()
        X, y = _toy_task(12)
        before = float(kernel_target_alignment(quantum_kernel_matrix(X, layer), y))
        history = train_kernel_alignment(layer, X, y, steps=15, lr=0.1)
        after = float(kernel_target_alignment(quantum_kernel_matrix(X, layer), y))
        assert len(history) == 15 and history[0] == pytest.approx(before)
        assert after > before + 0.05, (before, after)

    def test_subsets_keep_both_classes(self) -> None:
        # 2 positives out of 12: an unstratified subset of 4 would often hold
        # none, and the alignment would refuse it.
        layer = _reuploading()
        X, _ = _toy_task(12)
        y = torch.zeros(12, dtype=torch.float64)
        y[:2] = 1.0
        history = train_kernel_alignment(
            layer, X, y, steps=10, subset_size=4, generator=torch.Generator().manual_seed(0)
        )
        assert len(history) == 10

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"steps": 0}, "steps must be >= 1"), ({"subset_size": 1}, "subset_size must lie")],
    )
    def test_rejected(self, kwargs: dict, match: str) -> None:
        X, y = _toy_task(6)
        with pytest.raises(ValueError, match=match):
            train_kernel_alignment(_reuploading(), X, y, **kwargs)


# ---------------------------------------------------------------------------
# Overlap-circuit estimate (#213)
# ---------------------------------------------------------------------------


def _count_executions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many tapes each qml.execute call inside kernels runs."""
    counts: list[int] = []
    real = kernels.qml.execute

    def counting(tapes, *args, **kwargs):  # type: ignore[no-untyped-def]
        counts.append(len(tapes))
        return real(tapes, *args, **kwargs)

    monkeypatch.setattr(kernels.qml, "execute", counting)
    return counts


class TestOverlapKernel:
    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_exact_probabilities_agree_with_the_state_vector_kernel(self, build) -> None:
        layer = build()
        X = _inputs_for(layer)
        exact = quantum_kernel_matrix(X, layer)
        estimate = kernels.overlap_kernel_matrix(X, layer)
        torch.testing.assert_close(estimate, exact, rtol=0, atol=1e-10)
        torch.testing.assert_close(estimate.diagonal(), torch.ones(M, dtype=torch.float64))
        assert torch.equal(estimate, estimate.T)

    @pytest.mark.parametrize("build", _layers("angle", "amplitude"))
    def test_rectangular_agrees_too(self, build) -> None:
        layer = build()
        X = _inputs_for(layer)
        Y = X[:3] * 0.5 + 0.1
        torch.testing.assert_close(
            kernels.overlap_kernel_matrix(X, layer, Y),
            quantum_kernel_matrix(X, layer, Y),
            rtol=0,
            atol=1e-10,
        )

    def test_circuit_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counts = _count_executions(monkeypatch)
        layer = _angle_layer()
        kernels.overlap_kernel_matrix(_angles(M), layer)
        kernels.overlap_kernel_matrix(_angles(M), layer, _angles(4, seed=1))
        assert counts == [M * (M - 1) // 2, M * 4]

    def test_finite_shots_are_binomial_estimates(self) -> None:
        layer = _angle_layer()
        X = _angles(M)
        exact = quantum_kernel_matrix(X, layer)
        shots = 4000
        estimate = kernels.overlap_kernel_matrix(X, layer, shots=shots, seed=11)
        upper = torch.triu(torch.ones(M, M, dtype=torch.bool), 1)
        k = exact[upper]
        se = torch.sqrt(k * (1 - k) / shots).clamp(min=1.0 / shots)
        z = (estimate[upper] - k).abs() / se
        assert z.max() < 5.0, z.max()
        # Estimates move with the shot noise: not the exact values, and on the
        # grid of multiples of 1/shots.
        assert not torch.allclose(estimate[upper], k, atol=1e-6)
        torch.testing.assert_close(
            estimate[upper] * shots, (estimate[upper] * shots).round(), rtol=0, atol=1e-6
        )

    def test_seeded_sampling_is_reproducible(self) -> None:
        layer = _angle_layer()
        X = _angles(4)
        a = kernels.overlap_kernel_matrix(X, layer, shots=100, seed=3)
        b = kernels.overlap_kernel_matrix(X, layer, shots=100, seed=3)
        c = kernels.overlap_kernel_matrix(X, layer, shots=100, seed=4)
        assert torch.equal(a, b) and not torch.equal(a, c)

    def test_projection_to_psd(self) -> None:
        layer = _angle_layer()
        X = _angles(10)
        rough = kernels.overlap_kernel_matrix(X, layer, shots=5, seed=0)
        assert torch.linalg.eigvalsh(rough).min() < -1e-6  # 5 shots: not PSD
        projected = kernels.overlap_kernel_matrix(X, layer, shots=5, seed=0, project_psd=True)
        assert torch.linalg.eigvalsh(projected).min() > -1e-12
        assert torch.equal(projected, projected.T)
        # The nearest PSD matrix: no worse than the rough matrix's own PSD part
        # from any other eigenvalue treatment, e.g. shifting the spectrum up.
        shift = rough - torch.linalg.eigvalsh(rough).min() * torch.eye(10, dtype=torch.float64)
        assert torch.linalg.norm(projected - rough) <= torch.linalg.norm(shift - rough)

    def test_nearest_psd_worked_example(self) -> None:
        # Eigenvalues 3 and -1; dropping -1 leaves 3·vvᵀ with v = (1, 1)/√2.
        K = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
        torch.testing.assert_close(
            kernels.nearest_psd(K), torch.full((2, 2), 1.5, dtype=torch.float64)
        )
        psd = quantum_kernel_matrix(_angles(5), _angle_layer())
        torch.testing.assert_close(kernels.nearest_psd(psd), psd, rtol=0, atol=1e-12)

    def test_user_device(self) -> None:
        layer = _angle_layer()
        X = _angles(4)
        on_device = kernels.overlap_kernel_matrix(
            X, layer, device=qml.device("default.qubit", wires=N_QUBITS)
        )
        torch.testing.assert_close(on_device, quantum_kernel_matrix(X, layer), rtol=0, atol=1e-10)

    def test_single_sample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counts = _count_executions(monkeypatch)
        K = kernels.overlap_kernel_matrix(_angles(1), _angle_layer())
        assert K.tolist() == [[1.0]] and counts == []

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"shots": 0}, "shots must be a positive integer"),
            ({"project_psd": True, "Y": "same"}, "square matrix only"),
        ],
    )
    def test_rejected_arguments(self, kwargs: dict, match: str) -> None:
        X = _angles(3)
        if kwargs.get("Y") == "same":
            kwargs = {**kwargs, "Y": X}
        with pytest.raises(ValueError, match=match):
            kernels.overlap_kernel_matrix(X, _angle_layer(), **kwargs)

    def test_transformed_qnode_is_refused(self) -> None:
        from hqnn_forge.noise import apply_depolarizing_noise

        layer = _angle_layer()
        with apply_depolarizing_noise(layer, 0.1):
            with pytest.raises(RuntimeError, match="apply_depolarizing_noise"):
                kernels.overlap_kernel_matrix(_angles(3), layer)


class TestConstantKernelIsRefused:
    """
    A single RZ embedding would map every input to one state, so its kernel
    would be all ones and a precomputed-kernel SVM a constant classifier
    (#212).  The layer is refused at construction instead.
    """

    def test_rotation_z_layer_cannot_be_built(self) -> None:
        with pytest.raises(ValueError, match="global phase"):
            QuantumEncodingLayer(
                n_qubits=N_QUBITS, n_layers=1, rotation="Z", device_name="default.qubit"
            )

    def test_accepted_axes_give_a_kernel_that_depends_on_the_data(self) -> None:
        for rotation in ("X", "Y"):
            layer = QuantumEncodingLayer(
                n_qubits=N_QUBITS,
                n_layers=1,
                rotation=rotation,  # type: ignore[arg-type]
                device_name="default.qubit",
            )
            K = quantum_kernel_matrix(_angles(M), layer)
            off_diagonal = K[~torch.eye(M, dtype=torch.bool)]
            assert off_diagonal.max() < 1 - 1e-6, rotation


# ---------------------------------------------------------------------------
# Simulating in batches (#222)
# ---------------------------------------------------------------------------


def _count_passes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Rows in each broadcast tape the kernels module executes."""
    rows: list[int] = []
    real = kernels.qml.execute

    def counting(tapes, *args, **kwargs):  # type: ignore[no-untyped-def]
        rows.extend(t.batch_size or 1 for t in tapes)
        return real(tapes, *args, **kwargs)

    monkeypatch.setattr(kernels.qml, "execute", counting)
    return rows


class TestBatchedSimulation:
    @pytest.mark.parametrize("build", ALL_LAYERS)
    @pytest.mark.parametrize(
        "batch_size", [1, 3, M, 50], ids=["one", "non-divisor", "M", "over-M"]
    )
    def test_states_and_kernel_are_unchanged(self, build, batch_size: int) -> None:
        layer = build()
        X = _inputs_for(layer)
        torch.testing.assert_close(
            encoded_states(X, layer, batch_size=batch_size),
            encoded_states(X, layer),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            quantum_kernel_matrix(X, layer, batch_size=batch_size),
            quantum_kernel_matrix(X, layer),
            rtol=0,
            atol=0,
        )
        Y = X[:2]
        torch.testing.assert_close(
            quantum_kernel_matrix(X, layer, Y, batch_size=batch_size),
            quantum_kernel_matrix(X, layer, Y),
            rtol=0,
            atol=0,
        )

    def test_rows_are_replayed_in_batches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = _count_passes(monkeypatch)
        encoded_states(_angles(M), _angle_layer(), batch_size=3)
        assert rows == [3, 3, 1]
        rows.clear()
        encoded_states(_angles(M), _angle_layer())
        assert rows == [M]

    def test_y_is_rejected_before_any_batch_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = _count_passes(monkeypatch)
        bad_y = torch.zeros(2, N_QUBITS + 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="does not match"):
            quantum_kernel_matrix(_angles(M), _angle_layer(), bad_y, batch_size=2)
        assert rows == []

    def test_gradients_are_unchanged(self) -> None:
        def grads(batch_size: int | None) -> torch.Tensor:
            layer = _reuploading()
            X, y = _toy_task(7)
            K = quantum_kernel_matrix(X, layer, differentiable=True, batch_size=batch_size)
            kernel_target_alignment(K, y).backward()
            assert layer.qlayer.weights.grad is not None
            return layer.qlayer.weights.grad

        torch.testing.assert_close(grads(2), grads(None), rtol=0, atol=1e-15)

    @pytest.mark.parametrize("batch_size", [0, -3, True, 2.5])
    def test_rejected_batch_size(self, batch_size: int) -> None:
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            encoded_states(_angles(3), _angle_layer(), batch_size=batch_size)
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            quantum_kernel_matrix(_angles(3), _angle_layer(), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Under depolarising noise (#221)
# ---------------------------------------------------------------------------

_PAULIS = [
    torch.eye(2, dtype=torch.complex128),
    torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
]


def _z_on(wire: int, n: int) -> torch.Tensor:
    op = torch.ones(1, 1, dtype=torch.complex128)
    for w in range(n):
        op = torch.kron(op, _PAULIS[3] if w == wire else _PAULIS[0])
    return op


class TestNoisyKernel:
    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_zero_noise_density_path_equals_the_noiseless_kernel(self, build) -> None:
        layer = build()
        X = _inputs_for(layer)
        rho = kernels.encoded_density_matrices(X, layer, noise_level=0.0)
        states = encoded_states(X, layer)
        torch.testing.assert_close(
            rho, torch.einsum("ia,ib->iab", states, states.conj()), rtol=0, atol=1e-12
        )
        torch.testing.assert_close(
            kernels.kernel_from_density_matrices(rho),
            quantum_kernel_matrix(X, layer),
            rtol=0,
            atol=1e-12,
        )
        # noise_level=0 on quantum_kernel_matrix is the state-vector path itself.
        assert torch.equal(
            quantum_kernel_matrix(X, layer, noise_level=0.0), quantum_kernel_matrix(X, layer)
        )

    @pytest.mark.parametrize("position", ["all", "end"])
    def test_the_models_noise_is_the_kernels_noise(self, position: str) -> None:
        # <Z_i> read off the kernel's density matrices equals the layer's own
        # output inside apply_depolarizing_noise: the same channels, inserted
        # the same way.
        from hqnn_forge.noise import apply_depolarizing_noise

        layer = _angle_layer()
        X = _angles(4)
        rho = kernels.encoded_density_matrices(
            X,
            layer,
            noise_level=0.1,
            noise_position=position,  # type: ignore[arg-type]
        )
        from_rho = torch.stack(
            [torch.einsum("iab,ba->i", rho, _z_on(w, N_QUBITS)).real for w in range(N_QUBITS)],
            dim=1,
        )
        with apply_depolarizing_noise(layer, 0.1, position=position), torch.no_grad():  # type: ignore[arg-type]
            expected = layer(X).to(torch.float64)
        torch.testing.assert_close(from_rho, expected, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("p", [0.03, 0.2])
    def test_end_position_closed_form(self, p: float) -> None:
        # Channels only before measurement: with λ = 1 − 4p/3 per qubit,
        # Tr[D(ρ)D(σ)] = 2⁻ⁿ Σ_P λ^(2·wt P) Tr[ρP] Tr[σP] over Pauli strings P,
        # computed here from the noiseless states.
        layer = _angle_layer()
        X = _angles(5)
        states = encoded_states(X, layer)
        lam = 1 - 4 * p / 3
        expected = torch.zeros(5, 5, dtype=torch.float64)
        for combo in itertools.product(range(4), repeat=N_QUBITS):
            P = torch.ones(1, 1, dtype=torch.complex128)
            for c in combo:
                P = torch.kron(P, _PAULIS[c])
            e = torch.einsum("ia,ab,ib->i", states.conj(), P, states).real
            weight = sum(c != 0 for c in combo)
            expected += lam ** (2 * weight) * torch.outer(e, e) / 2**N_QUBITS
        got = quantum_kernel_matrix(X, layer, noise_level=p, noise_position="end")
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)

    def test_symmetric_psd_with_purity_on_the_diagonal(self) -> None:
        layer = _reuploading()
        X = _angles(M)
        K = quantum_kernel_matrix(X, layer, noise_level=0.1)
        assert torch.equal(K, K.T)
        assert torch.linalg.eigvalsh(K).min() > -1e-12
        rho = kernels.encoded_density_matrices(X, layer, noise_level=0.1)
        purity = torch.einsum("iab,iba->i", rho, rho).real
        torch.testing.assert_close(K.diagonal(), purity, rtol=0, atol=1e-12)
        assert K.diagonal().max() < 1.0
        torch.testing.assert_close(
            torch.einsum("iaa->i", rho).real, torch.ones(M, dtype=torch.float64)
        )
        torch.testing.assert_close(rho, rho.conj().transpose(1, 2), rtol=0, atol=1e-12)

    def test_diagonal_falls_with_the_noise_level(self) -> None:
        layer = _angle_layer()
        X = _angles(4)
        diagonals = [
            quantum_kernel_matrix(X, layer, noise_level=p).diagonal()
            for p in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
        ]
        for lower, higher in itertools.pairwise(diagonals):
            assert torch.all(higher < lower)

    def test_rectangular_and_batched(self) -> None:
        layer = _angle_layer()
        X, Y = _angles(5), _angles(3, seed=1)
        full = quantum_kernel_matrix(torch.cat([X, Y]), layer, noise_level=0.1)
        rect = quantum_kernel_matrix(X, layer, Y, noise_level=0.1)
        torch.testing.assert_close(rect, full[:5, 5:], rtol=0, atol=1e-12)
        # Not bit-exact: the density matrices of a batch are evolved together,
        # so the batch size changes the contraction order and the last bits of
        # the result (5.6e-17 seen on CI).  The state-vector kernel, which
        # simulates every state on its own, is exact at any batch size.
        torch.testing.assert_close(
            quantum_kernel_matrix(X, layer, Y, noise_level=0.1, batch_size=2),
            rect,
            rtol=0,
            atol=1e-15,
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"noise_level": 0.8}, r"noise_level must lie in \[0, 0.75\]"),
            ({"noise_level": 0.1, "noise_position": "middle"}, "noise_position must be"),
        ],
    )
    def test_rejected_noise_arguments(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            quantum_kernel_matrix(_angles(3), _angle_layer(), **kwargs)
        with pytest.raises(ValueError, match=match):
            kernels.encoded_density_matrices(_angles(3), _angle_layer(), **kwargs)

    def test_inside_the_noise_block_it_points_to_noise_level(self) -> None:
        from hqnn_forge.noise import apply_depolarizing_noise

        layer = _angle_layer()
        with apply_depolarizing_noise(layer, 0.1):
            with pytest.raises(RuntimeError, match="pass noise_level="):
                quantum_kernel_matrix(_angles(3), layer, noise_level=0.1)

    def test_density_kernel_shape_checks(self) -> None:
        with pytest.raises(ValueError, match="stacks of equal square matrices"):
            kernels.kernel_from_density_matrices(torch.zeros(2, 4, 4), torch.zeros(2, 2, 2))
