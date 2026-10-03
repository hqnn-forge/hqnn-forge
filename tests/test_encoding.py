"""
tests/test_encoding.py
=======================
Smoke tests for hqnn_forge.encoding.QuantumEncodingLayer.

These tests validate:
1. Forward pass returns the correct shape.
2. Output values are within the valid expectation-value range [-1, 1].
3. Gradients flow back from the loss through the quantum layer to its weights.
4. The layer raises ValueError for mismatched input dimensions.
5. Restricted-variance and block-local init match their documented sigma laws,
   with assertions calibrated to reject a flat-sigma regression.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.initializers import block_local_init_, restricted_normal_init_
from hqnn_forge.initializers.restricted_variance import _not_restricting_ignored
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_QUBITS = 4  # keep small for test speed (real usage: 8)
N_LAYERS = 2
BATCH = 8


@pytest.fixture(scope="module")
def layer() -> QuantumEncodingLayer:
    """Shared QuantumEncodingLayer instance (uses default.qubit for portability)."""
    return QuantumEncodingLayer(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        device_name="default.qubit",  # lightning.qubit not required in CI
        diff_method="parameter-shift",  # universal; works on default.qubit
    )


@pytest.fixture
def random_batch() -> torch.Tensor:
    """Random input batch, values in (-π, π)."""
    return torch.rand(BATCH, N_QUBITS) * 2 * math.pi - math.pi


# ---------------------------------------------------------------------------
# Test 1: Output shape
# ---------------------------------------------------------------------------


class TestForwardPassShape:
    def test_output_shape(self, layer: QuantumEncodingLayer, random_batch: torch.Tensor) -> None:
        """Forward pass should return shape (batch_size, n_qubits)."""
        out = layer(random_batch)
        assert out.shape == (BATCH, N_QUBITS), (
            f"Expected shape ({BATCH}, {N_QUBITS}); got {out.shape}"
        )

    def test_single_sample(self, layer: QuantumEncodingLayer) -> None:
        """Single-sample batch (batch_size=1) should work correctly."""
        x = torch.rand(1, N_QUBITS)
        out = layer(x)
        assert out.shape == (1, N_QUBITS)


# ---------------------------------------------------------------------------
# Test 2: Output range
# ---------------------------------------------------------------------------


class TestOutputRange:
    def test_expectation_values_in_range(
        self, layer: QuantumEncodingLayer, random_batch: torch.Tensor
    ) -> None:
        """PauliZ expectation values must lie in [-1, 1]."""
        with torch.no_grad():
            out = layer(random_batch)
        assert out.min().item() >= -1.0 - 1e-5, f"Output below -1: {out.min().item()}"
        assert out.max().item() <= 1.0 + 1e-5, f"Output above  1: {out.max().item()}"


# ---------------------------------------------------------------------------
# Test 3: Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradients_reach_quantum_weights(
        self, layer: QuantumEncodingLayer, random_batch: torch.Tensor
    ) -> None:
        """
        A backward pass through a scalar loss must leave non-None, non-zero
        gradients on the variational weights.
        """
        # Reset any pre-existing gradients
        layer.zero_grad()

        out = layer(random_batch)  # (BATCH, N_QUBITS)
        loss = out.sum()  # scalar
        loss.backward()

        weights_param = layer.qlayer.weights
        assert weights_param.grad is not None, (
            "Gradient of quantum weights is None after backward pass."
        )
        assert weights_param.grad.abs().sum().item() > 0.0, (
            "Gradient of quantum weights is zero everywhere — training would stall."
        )

    def test_no_grad_inference(
        self, layer: QuantumEncodingLayer, random_batch: torch.Tensor
    ) -> None:
        """Inference under torch.no_grad() should not compute gradients."""
        with torch.no_grad():
            out = layer(random_batch)
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# Test 4: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wrong_feature_dim_raises(self, layer: QuantumEncodingLayer) -> None:
        """Input with wrong last-dim should raise ValueError."""
        wrong_input = torch.rand(BATCH, N_QUBITS + 1)
        with pytest.raises(ValueError, match="n_qubits"):
            layer(wrong_input)

    def test_n_qubits_lt_2_raises(self) -> None:
        """n_qubits < 2 is invalid (no meaningful entangling ring)."""
        from hqnn_forge.encoding.angle_embedding import build_encoding_qnode

        with pytest.raises(ValueError, match="n_qubits must be"):
            build_encoding_qnode(n_qubits=1)


# ---------------------------------------------------------------------------
# Test 5: Initialiser properties
# ---------------------------------------------------------------------------

# Statistical assertions on initialiser output draw n_qubits * 3 samples per
# layer.  At 16 qubits and 16 layers every tolerance below clears the worst deviation observed over 5000
# seeds, so the tests hold for any seed rather than relying on INIT_SEED alone:
#
#   restricted std, relative error   worst 0.091  tolerance 0.20  (2.2x)
#   log-std slope, restricted        worst 0.160  tolerance 0.25  (1.56x)
#
# block_local_init_ is pinned exactly instead: for the same seed it is
# restricted_normal_init_ times sqrt(L / (L + l)) on layer l, which a test
# checks to float precision.  The statistical tests below only confirm what
# the draws look like, at sizes where they cannot flake.
INIT_N_QUBITS = 16
INIT_N_LAYERS = 16
INIT_SEED = 0
STD_TOLERANCE = 0.20  # relative
SLOPE_TOLERANCE = 0.25  # in log-log space
# One layer's std from 768 draws (256 qubits): worst relative error over 5000
# seeds 0.100, so STD_TOLERANCE leaves 2x.
LAYER_STD_N_QUBITS = 256
# The first/last std ratio sqrt((2L - 1)/L) is only 1.39 at 16 layers, so it
# needs 3072 draws per layer (1024 qubits): over 5000 seeds its worst relative
# deviation is 0.063, and a flat tensor never comes closer than 0.236 to it.
# TAPER_TOLERANCE = 0.14 leaves 2.2x on the taper side and 1.7x on the power
# side.
TAPER_N_QUBITS = 1024
TAPER_TOLERANCE = 0.14  # relative


def _log_std_slope(tensor: torch.Tensor) -> float:
    """
    Least-squares slope of log(per-layer std) against log(layer_index + 1).

    A constant sigma across layers gives slope 0.
    """
    w = tensor.double()
    y = torch.log(w.flatten(start_dim=1).std(dim=1))
    x = torch.log(torch.arange(1, w.shape[0] + 1, dtype=torch.float64))
    x_mean, y_mean = x.mean(), y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    return slope.item()


def _restricted() -> torch.Tensor:
    torch.manual_seed(INIT_SEED)
    tensor = torch.empty(INIT_N_LAYERS, INIT_N_QUBITS, 3)
    return restricted_normal_init_(tensor, n_qubits=INIT_N_QUBITS, n_layers=INIT_N_LAYERS)


class TestRestrictedVarianceInit:
    def test_std_matches_documented_sigma(self) -> None:
        """sigma = pi / sqrt(n_qubits * n_layers), per the initialiser docstring."""
        expected_std = math.pi / math.sqrt(INIT_N_QUBITS * INIT_N_LAYERS)
        actual_std = _restricted().std().item()
        assert actual_std == pytest.approx(expected_std, rel=STD_TOLERANCE)

    def test_mean_approximately_zero(self) -> None:
        """Initialised weights should be zero-mean."""
        assert abs(_restricted().mean().item()) < 0.1

    def test_variance_is_flat_across_layers(self) -> None:
        """One shared sigma: log-std has zero slope in depth."""
        slope = _log_std_slope(_restricted())
        assert abs(slope) < SLOPE_TOLERANCE

    def test_returns_same_tensor(self) -> None:
        tensor = torch.empty(2, 4, 3)
        assert restricted_normal_init_(tensor, n_qubits=4, n_layers=2) is tensor


def _not_restricting(record: pytest.WarningsRecorder) -> warnings.WarningMessage:
    """The one init warning in *record*, whatever else torch or PennyLane emitted."""
    [match] = [w for w in record if "restricts nothing" in str(w.message)]
    return match


class TestWiderThanUniformWarning:
    """
    #167: the initialisers warn when the σ they draw is not narrower than a
    uniform draw over [0, 2π), std 2π/sqrt(12) ≈ 1.8138.  At scale = π that
    is n_qubits * n_layers <= 3, with n * L = 3 exactly on the boundary.
    """

    @pytest.mark.parametrize(("n_qubits", "n_layers"), [(1, 1), (2, 1), (3, 1), (1, 3)])
    def test_restricted_warns_when_not_narrower(self, n_qubits: int, n_layers: int) -> None:
        sigma = math.pi / math.sqrt(n_qubits * n_layers)
        with pytest.warns(UserWarning, match="restricts nothing") as record:
            restricted_normal_init_(torch.empty(n_layers, n_qubits, 3), n_qubits, n_layers)
        warning = _not_restricting(record)
        message = str(warning.message)
        assert f"σ = {sigma:.4f}" in message and "std 1.8138" in message
        assert "n_qubits * n_layers <= 3" in message
        assert warning.filename == __file__, "stacklevel should point at the caller"

    @pytest.mark.parametrize(("n_qubits", "n_layers"), [(3, 1), (1, 3), (2, 1)])
    def test_block_local_warns_on_its_widest_layer(self, n_qubits: int, n_layers: int) -> None:
        sigma = math.pi / math.sqrt(n_qubits * n_layers)
        with pytest.warns(UserWarning, match="block_local_init_") as record:
            block_local_init_(torch.empty(n_layers, n_qubits, 3), n_qubits=n_qubits)
        warning = _not_restricting(record)
        assert f"σ = {sigma:.4f}" in str(warning.message)
        assert warning.filename == __file__

    @pytest.mark.parametrize(("n_qubits", "n_layers"), [(4, 1), (2, 2), (8, 2)])
    def test_silent_once_narrower(self, n_qubits: int, n_layers: int) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            restricted_normal_init_(torch.empty(n_layers, n_qubits, 3), n_qubits, n_layers)
            block_local_init_(torch.empty(n_layers, n_qubits, 3), n_qubits=n_qubits)

    def test_a_smaller_scale_moves_the_boundary(self) -> None:
        """The check is on σ, not on n * L: scale = 1 is narrower even at 1 x 1."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            restricted_normal_init_(torch.empty(1, 1, 3), 1, 1, scale=1.0)

    def test_a_wider_scale_names_itself_not_the_pi_rule(self) -> None:
        """σ = 4/sqrt(4) = 2 > 1.81 at n * L = 4: the n * L <= 3 rule would mislead."""
        with pytest.warns(UserWarning, match="restricts nothing") as record:
            restricted_normal_init_(torch.empty(2, 2, 3), 2, 2, scale=4.0)
        message = str(_not_restricting(record).message)
        assert "scale=4" in message and "σ = 2.0000" in message
        assert "<= 3" not in message

    def test_block_local_passes_its_scale_through(self) -> None:
        """Layer 0 of 2 x 2 at scale = 4 is 4/sqrt(2 * 2) = 2, and at scale = 1 it is 0.5."""
        with pytest.warns(UserWarning, match="block_local_init_") as record:
            block_local_init_(torch.empty(2, 2, 3), n_qubits=2, scale=4.0)
        message = str(_not_restricting(record).message)
        assert "scale=4" in message and "σ = 2.0000" in message
        assert "<= 3" not in message
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            block_local_init_(torch.empty(1, 1, 3), n_qubits=1, scale=1.0)

    def test_suppression_leaves_other_warnings_and_the_filters_alone(self) -> None:
        """
        _not_restricting_ignored() silences only the init check, by flag rather
        than by message: other warnings, even with the same wording, still get
        through, and the global filter list is never touched.
        """
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            filters = list(warnings.filters)
            with _not_restricting_ignored():
                assert warnings.filters == filters
                restricted_normal_init_(torch.empty(1, 1, 3), 1, 1)
                warnings.warn("someone else's warning that restricts nothing", UserWarning)
        assert [str(w.message) for w in record] == [
            "someone else's warning that restricts nothing"
        ]
        with pytest.warns(UserWarning, match="restricts nothing"):
            restricted_normal_init_(torch.empty(1, 1, 3), 1, 1)

    @pytest.mark.parametrize(
        "cls", [HybridBinaryClassifier, ParallelHybridClassifier, MulticlassHybridClassifier]
    )
    @pytest.mark.parametrize("init_strategy", ["restricted", "block_local"])
    def test_classifier_warns_at_the_users_line(self, cls: type, init_strategy: str) -> None:
        kwargs = dict(
            n_input_features=2,
            n_qubits=2,
            n_layers=1,
            device_name="default.qubit",
            diff_method="backprop",
        )
        with pytest.warns(UserWarning, match="restricts nothing") as record:
            cls(init_strategy=init_strategy, **kwargs)
        warning = _not_restricting(record)
        assert warning.filename == __file__, "attributed inside hqnn_forge, not to the caller"
        assert "init_strategy='normal'" in str(warning.message)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            cls(init_strategy="normal", **kwargs)


class TestBlockLocalInit:
    """
    sigma_l = pi / sqrt(n_qubits * (L + l)) (#166): the restricted sigma in
    layer 0, tapering to it over sqrt((2L - 1)/L) in the last layer.
    """

    @pytest.mark.parametrize("shape", [(2, 3, 3), (7, 5, 3), (16, 16, 3), (1, 2, 3), (5,)])
    def test_is_restricted_init_tapered_by_layer(self, shape: tuple[int, ...]) -> None:
        """
        Exact, not statistical: for the same RNG state, layer l is the
        restricted draw times sqrt(L / (L + l)).  Together with the sigma
        pinned in TestRestrictedVarianceInit this fixes the whole schedule,
        including its dependence on the total depth L.
        """
        n_layers, n_qubits = shape[0], shape[1] if len(shape) > 1 else 4
        torch.manual_seed(INIT_SEED)
        restricted = restricted_normal_init_(torch.empty(shape), n_qubits, n_layers)
        torch.manual_seed(INIT_SEED)
        tapered = block_local_init_(torch.empty(shape), n_qubits=n_qubits)
        factor = torch.tensor([math.sqrt(n_layers / (n_layers + l)) for l in range(n_layers)])
        expected = restricted * factor.view(-1, *([1] * (len(shape) - 1)))
        torch.testing.assert_close(tapered, expected, rtol=1e-6, atol=0)

    def test_first_layer_narrows_with_total_depth(self) -> None:
        """
        Layer 0 is pi / sqrt(n * L): 4x narrower at 32 layers than at 2.  The
        old schedule, pi / sqrt(n * (l + 1)), drew it at pi / sqrt(n) for
        every depth and fails both bounds.
        """
        for n_layers in (2, 32):
            torch.manual_seed(INIT_SEED)
            tensor = block_local_init_(
                torch.empty(n_layers, LAYER_STD_N_QUBITS, 3), n_qubits=LAYER_STD_N_QUBITS
            )
            expected = math.pi / math.sqrt(LAYER_STD_N_QUBITS * n_layers)
            assert tensor[0].std().item() == pytest.approx(expected, rel=STD_TOLERANCE)

    def test_first_layer_wider_than_last(self) -> None:
        """The documented ratio sigma_0 / sigma_{L-1} = sqrt((2L - 1) / L)."""
        torch.manual_seed(INIT_SEED)
        tensor = torch.empty(INIT_N_LAYERS, TAPER_N_QUBITS, 3)
        block_local_init_(tensor, n_qubits=TAPER_N_QUBITS)
        ratio = tensor[0].std().item() / tensor[-1].std().item()
        expected = math.sqrt((2 * INIT_N_LAYERS - 1) / INIT_N_LAYERS)
        assert ratio == pytest.approx(expected, rel=TAPER_TOLERANCE)

    def test_taper_check_rejects_a_flat_tensor(self) -> None:
        """
        Power check for the assertion above: a constant-sigma tensor (what
        ``block_local_init_`` would produce if it silently degraded to
        ``restricted_normal_init_``) must *fail* the ratio criterion.
        """
        torch.manual_seed(INIT_SEED)
        tensor = torch.empty(INIT_N_LAYERS, TAPER_N_QUBITS, 3)
        restricted_normal_init_(tensor, n_qubits=TAPER_N_QUBITS, n_layers=INIT_N_LAYERS)
        ratio = tensor[0].std().item() / tensor[-1].std().item()
        expected = math.sqrt((2 * INIT_N_LAYERS - 1) / INIT_N_LAYERS)
        assert ratio != pytest.approx(expected, rel=TAPER_TOLERANCE)

    def test_returns_same_tensor(self) -> None:
        tensor = torch.empty(2, 4, 3)
        assert block_local_init_(tensor, n_qubits=4) is tensor
