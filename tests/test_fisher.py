"""
tests/test_fisher.py
====================
hqnn_forge.diagnostics.fisher: the Fisher matrix against hand-computed values
on a circuit small enough to work out on paper, the effective-dimension
formula against its closed forms, and the model-facing wrapper's mechanics.

Hand computation
----------------
Two qubits, one layer, input x = 0.  The CNOT ring leaves |00⟩ alone, so each
qubit sees only its own Rot(φ, θ, ω) = RZ(ω) RY(θ) RZ(φ) acting on |0⟩:
RZ(φ) is a phase, RY(θ) gives ⟨Z⟩ = cos θ, RZ(ω) commutes with Z.  Hence

    ∂⟨Z_i⟩/∂θ_i = −sin θ_i,   every other derivative = 0,

and the Gaussian (Gauss-Newton) Fisher matrix JᵀJ has eigenvalues
{sin²θ_0, sin²θ_1, 0, 0, 0, 0}.  For the classifier with logit
z = w·⟨Z⟩ + b the Bernoulli Fisher matrix is the rank-one
σ(z)(1−σ(z)) g gᵀ with g_θi = −w_i sin θ_i.
"""

from __future__ import annotations

import math
from typing import TypedDict

import pennylane as qml
import pytest
import torch

from hqnn_forge.diagnostics import (
    EffectiveDimensionResult,
    FisherSpectrum,
    effective_dimension,
    effective_dimension_from_spectra,
    fisher_information_matrix,
    fisher_information_spectrum,
)
from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier


class _Backend(TypedDict):
    device_name: DeviceName
    diff_method: DiffMethod


CPU: _Backend = {"device_name": "default.qubit", "diff_method": "backprop"}
WEIGHTS = torch.tensor([[[0.3, 0.7, 1.1], [0.2, 1.9, 0.5]]])  # (1 layer, 2 qubits, φ θ ω)
THETA = (0.7, 1.9)


def _layer(n_qubits: int = 2, n_layers: int = 1) -> QuantumEncodingLayer:
    torch.manual_seed(0)
    return QuantumEncodingLayer(n_qubits=n_qubits, n_layers=n_layers, **CPU)


def _two_qubit_layer() -> QuantumEncodingLayer:
    layer = _layer()
    with torch.no_grad():
        layer.qlayer.weights.copy_(WEIGHTS)
    return layer


def _two_qubit_classifier() -> HybridBinaryClassifier:
    torch.manual_seed(0)
    model = HybridBinaryClassifier(
        n_input_features=2, n_qubits=2, n_layers=1, use_classical_encoder=False, **CPU
    )
    with torch.no_grad():
        model.quantum_layer.qlayer.weights.copy_(WEIGHTS)
    return model


def _angles(n: int, n_qubits: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(n, n_qubits) * 2 * math.pi - math.pi


# ---------------------------------------------------------------------------
# Fisher matrix against hand computation
# ---------------------------------------------------------------------------


class TestFisherMatrixHandComputed:
    def test_layer_spectrum_at_zero_input(self) -> None:
        result = fisher_information_matrix(_two_qubit_layer(), torch.zeros(3, 2))
        assert result.likelihood == "gaussian"
        assert result.n_params == 6
        assert result.n_data == 3
        assert result.rank == 2
        expected = sorted((math.sin(t) ** 2 for t in THETA), reverse=True)
        torch.testing.assert_close(
            result.eigenvalues[:2], torch.tensor(expected, dtype=torch.float64), atol=1e-6, rtol=0
        )
        assert result.eigenvalues[2:].abs().max().item() < 1e-12

    def test_layer_matrix_is_diagonal_in_theta(self) -> None:
        """Only the two θ entries (index 1 of each Rot) are non-zero."""
        result = fisher_information_matrix(_two_qubit_layer(), torch.zeros(1, 2))
        expected = torch.zeros(6, 6, dtype=torch.float64)
        for qubit, theta in enumerate(THETA):
            i = qubit * 3 + 1
            expected[i, i] = math.sin(theta) ** 2
        torch.testing.assert_close(result.matrix, expected, atol=1e-6, rtol=0)

    def test_classifier_is_rank_one_bernoulli(self) -> None:
        model = _two_qubit_classifier()
        result = fisher_information_matrix(model, torch.zeros(2, 2))
        assert result.likelihood == "bernoulli"
        assert result.rank == 1
        w = model.head.weight.detach()[0].double()
        b = model.head.bias.detach()[0].double()
        z = w[0] * math.cos(THETA[0]) + w[1] * math.cos(THETA[1]) + b
        p = torch.sigmoid(z)
        expected = (
            p
            * (1 - p)
            * (w[0] ** 2 * math.sin(THETA[0]) ** 2 + w[1] ** 2 * math.sin(THETA[1]) ** 2)
        )
        assert result.eigenvalues[0].item() == pytest.approx(expected.item(), rel=1e-5)

    def test_classifier_rank_is_bounded_by_n_data(self) -> None:
        """Each sample contributes a rank-one term."""
        model = _two_qubit_classifier()
        assert fisher_information_matrix(model, _angles(2, 2)).rank <= 2
        assert fisher_information_matrix(model, _angles(4, 2)).rank <= 4


# ---------------------------------------------------------------------------
# Fisher matrix mechanics
# ---------------------------------------------------------------------------


MODELS = [
    pytest.param(lambda: _layer(3, 2), 3, id="angle-layer"),
    pytest.param(lambda: IQPEncodingLayer(n_qubits=3, n_layers=2, **CPU), 3, id="iqp-layer"),
    pytest.param(
        lambda: HybridBinaryClassifier(n_input_features=5, n_qubits=3, n_layers=2, **CPU),
        5,
        id="hybrid",
    ),
    pytest.param(
        lambda: ParallelHybridClassifier(n_input_features=5, n_qubits=3, n_layers=2, **CPU),
        5,
        id="parallel",
    ),
]


class TestFisherMatrixMechanics:
    @pytest.mark.parametrize(("build", "n_features"), MODELS)
    def test_shape_symmetry_psd_and_order(self, build, n_features: int) -> None:
        torch.manual_seed(0)
        model = build()
        result = fisher_information_matrix(model, _angles(4, n_features))
        d = (
            model.quantum_layer.qlayer.weights.numel()
            if hasattr(model, "quantum_layer")
            else model.qlayer.weights.numel()
        )
        assert isinstance(result, FisherSpectrum)
        assert result.matrix.shape == (d, d)
        assert result.matrix.dtype == torch.float64
        torch.testing.assert_close(result.matrix, result.matrix.T, rtol=0, atol=0)
        assert result.eigenvalues.shape == (d,)
        assert torch.all(result.eigenvalues[:-1] >= result.eigenvalues[1:])
        assert result.eigenvalues.min().item() >= 0.0
        assert result.trace == pytest.approx(result.matrix.trace().item(), rel=1e-9)
        assert 0 < result.rank <= d
        assert result.normalized_eigenvalues.sum().item() == pytest.approx(d, rel=1e-9)
        assert result.to_dict()["rank"] == result.rank

    def test_spectrum_helper_returns_the_eigenvalues(self) -> None:
        layer = _layer(3, 2)
        X = _angles(4, 3)
        torch.testing.assert_close(
            fisher_information_spectrum(layer, X), fisher_information_matrix(layer, X).eigenvalues
        )

    def test_leaves_model_untouched(self) -> None:
        layer = _layer(3, 2)
        before = layer.qlayer.weights.detach().clone()
        fisher_information_matrix(layer, _angles(4, 3))
        assert torch.equal(layer.qlayer.weights.detach(), before)
        assert layer.qlayer.weights.grad is None

    def test_last_layer_omega_carries_no_information(self) -> None:
        """
        Rot = RZ(ω) RY(θ) RZ(φ); the final RZ(ω) of the last layer commutes
        with every Z readout, so those n_qubits parameters have zero Fisher
        information at every input.  Pinned here because it is what the
        spectrum shows about this ansatz (see the dead-parameter issue).
        """
        layer = _layer(3, 2)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        result = fisher_information_matrix(layer, _angles(6, 3))
        diag = result.matrix.diagonal().reshape(2, 3, 3)
        assert diag[1, :, 2].abs().max().item() < 1e-12
        assert diag[0].min().item() > 1e-6
        assert diag[1, :, :2].min().item() > 1e-6

    def test_rejects_bad_data(self) -> None:
        layer = _layer()
        with pytest.raises(ValueError, match="n_samples, n_features"):
            fisher_information_matrix(layer, torch.zeros(2))
        with pytest.raises(ValueError, match="no rows"):
            fisher_information_matrix(layer, torch.zeros(0, 2))

    def test_rejects_models_without_quantum_weights(self) -> None:
        with pytest.raises(TypeError, match="fisher_information_matrix expects"):
            fisher_information_matrix(torch.nn.Linear(2, 1), torch.zeros(2, 2))

    def test_single_output_layer_is_gaussian_not_bernoulli(self) -> None:
        """
        readout="first" at x = 0 returns only ⟨Z_0⟩ = cos θ_0, so JᵀJ is
        sin²θ_0 in θ_0's slot.  The single output is an expectation value,
        not a logit, so no σ(z)(1 − σ(z)) factor may appear.
        """
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(n_qubits=2, n_layers=1, readout="first", **CPU)
        with torch.no_grad():
            layer.qlayer.weights.copy_(WEIGHTS)
        result = fisher_information_matrix(layer, torch.zeros(2, 2))
        assert result.likelihood == "gaussian"
        expected = torch.zeros(6, 6, dtype=torch.float64)
        expected[1, 1] = math.sin(THETA[0]) ** 2
        torch.testing.assert_close(result.matrix, expected, atol=1e-6, rtol=0)

    def test_frozen_weights_are_measured_and_stay_frozen(self) -> None:
        model = _two_qubit_classifier()
        X = _angles(3, 2)
        expected = fisher_information_matrix(model, X).matrix
        model.quantum_layer.qlayer.weights.requires_grad_(False)
        result = fisher_information_matrix(model, X)
        torch.testing.assert_close(result.matrix, expected, rtol=0, atol=0)
        assert not model.quantum_layer.qlayer.weights.requires_grad

    def test_results_compare_without_touching_the_tensors(self) -> None:
        layer = _two_qubit_layer()
        # Separately computed, so the tensors are equal but not identical and
        # tuple comparison cannot short-circuit on identity.
        a = fisher_information_matrix(layer, torch.zeros(1, 2))
        again = fisher_information_matrix(layer, torch.zeros(1, 2))
        other = fisher_information_matrix(layer, torch.zeros(2, 2))
        assert (a == again) is True
        assert (a == other) is False  # n_data differs


# ---------------------------------------------------------------------------
# Effective dimension: closed forms
# ---------------------------------------------------------------------------


def _kappa(n: int, gamma: float = 1.0) -> float:
    return gamma * n / (2 * math.pi * math.log(n))


def _identity_bound(d: int, n: int, gamma: float = 1.0) -> float:
    """d_eff of a flat spectrum, the most a single draw can reach: d·log(1+κ)/log κ."""
    return d * math.log(1 + _kappa(n, gamma)) / math.log(_kappa(n, gamma))


class TestEffectiveDimensionFormula:
    def test_identity_fisher(self) -> None:
        """F̂ = I_d gives d · log(1 + κ) / log κ exactly."""
        d, n = 6, 100
        got = effective_dimension_from_spectra(torch.ones(3, d), n_data=n)
        assert got == pytest.approx(d * math.log(1 + _kappa(n)) / math.log(_kappa(n)), rel=1e-12)

    def test_scale_invariance(self) -> None:
        """The trace normalisation removes the overall scale of F."""
        torch.manual_seed(0)
        spectra = torch.rand(5, 8, dtype=torch.float64)
        a = effective_dimension_from_spectra(spectra, n_data=500)
        b = effective_dimension_from_spectra(1e-3 * spectra, n_data=500)
        assert a == pytest.approx(b, rel=1e-12)

    def test_single_direction_is_below_full_rank(self) -> None:
        d, n = 8, 1000
        one_direction = torch.zeros(4, d)
        one_direction[:, 0] = 1.0
        low = effective_dimension_from_spectra(one_direction, n_data=n)
        full = effective_dimension_from_spectra(torch.ones(4, d), n_data=n)
        assert 0 < low < full
        # One normalised eigenvalue of size d: 2·½·log(1 + κd) / log κ.
        assert low == pytest.approx(math.log(1 + _kappa(n) * d) / math.log(_kappa(n)), rel=1e-12)

    def test_zero_spectra_give_zero(self) -> None:
        assert effective_dimension_from_spectra(torch.zeros(3, 5), n_data=500) == 0.0

    def test_accepts_a_sequence_of_spectra(self) -> None:
        spectra = [torch.ones(4), torch.ones(4)]
        assert effective_dimension_from_spectra(spectra, n_data=500) == pytest.approx(
            effective_dimension_from_spectra(torch.ones(2, 4), n_data=500)
        )

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="n_data"):
            effective_dimension_from_spectra(torch.ones(2, 3), n_data=1)
        with pytest.raises(ValueError, match="gamma"):
            effective_dimension_from_spectra(torch.ones(2, 3), n_data=500, gamma=0.0)
        with pytest.raises(ValueError, match="gamma"):
            effective_dimension_from_spectra(torch.ones(2, 3), n_data=500, gamma=1.5)
        with pytest.raises(ValueError, match="n_theta_samples, d"):
            effective_dimension_from_spectra(torch.ones(3), n_data=500)

    @pytest.mark.parametrize("n", [2, 7, 18, 19, 50, 73])
    def test_rejects_kappa_below_e(self, n: int) -> None:
        """
        log κ ≤ 0 flips the sign (n ≤ 18 at γ = 1) and 0 < log κ < 1 inflates
        the estimate far past d (n = 19 gave 159 for d = 6).
        """
        assert _kappa(n) < math.e
        with pytest.raises(ValueError, match=r"κ ≥ e, i\.e\. n_data ≥ 74"):
            effective_dimension_from_spectra(torch.ones(2, 6), n_data=n)

    def test_smaller_gamma_raises_the_minimum_n(self) -> None:
        with pytest.raises(ValueError, match="n_data ≥ 1213"):
            effective_dimension_from_spectra(torch.ones(2, 6), n_data=1000, gamma=0.1)

    def test_smallest_allowed_n_stays_close_to_d(self) -> None:
        d = 6
        got = effective_dimension_from_spectra(torch.ones(2, d), n_data=74)
        assert got == pytest.approx(_identity_bound(d, 74), rel=1e-12)
        assert d < got <= d * math.log(1 + math.e)


# ---------------------------------------------------------------------------
# Effective dimension: model-facing wrapper
# ---------------------------------------------------------------------------


class TestEffectiveDimension:
    def test_result_fields_and_bounds(self) -> None:
        layer = _layer(3, 2)
        result = effective_dimension(
            layer,
            _angles(6, 3),
            n_data=1000,
            n_theta_samples=4,
            generator=torch.Generator().manual_seed(1),
        )
        assert isinstance(result, EffectiveDimensionResult)
        assert result.layer_type == "QuantumEncodingLayer"
        assert result.init == "uniform"
        assert result.n_params == 18
        assert result.n_data == 1000
        assert result.n_theta_samples == 4
        assert 0.0 < result.effective_dimension <= _identity_bound(18, 1000)
        assert result.normalized_effective_dimension == pytest.approx(
            result.effective_dimension / 18
        )
        assert result.mean_normalized_spectrum.shape == (18,)
        assert result.mean_normalized_spectrum.sum().item() == pytest.approx(18, rel=1e-9)
        assert result.to_dict()["effective_dimension"] == result.effective_dimension

    def test_n_data_defaults_to_the_sample_size(self) -> None:
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(n_qubits=2, n_layers=1, readout="first", **CPU)
        result = effective_dimension(layer, _angles(80, 2), n_theta_samples=1)
        assert result.n_data == 80
        assert 0.0 <= result.effective_dimension <= _identity_bound(6, 80)

    def test_small_default_sample_asks_for_n_data(self) -> None:
        with pytest.raises(ValueError, match="pass it explicitly"):
            effective_dimension(_layer(), _angles(7, 2), n_theta_samples=2)

    def test_validates_before_computing_anything(self) -> None:
        """A Linear would fail weight resolution; the κ check must come first."""
        with pytest.raises(ValueError, match="gamma"):
            effective_dimension(torch.nn.Linear(2, 1), _angles(2, 2), n_data=1000, gamma=1.5)
        with pytest.raises(ValueError, match="κ ≥ e"):
            effective_dimension(torch.nn.Linear(2, 1), _angles(2, 2))

    def test_restores_weights_and_global_rng(self) -> None:
        layer = _layer(3, 2)
        X = _angles(4, 3)  # seeds the global RNG itself, so draw it first
        before = layer.qlayer.weights.detach().clone()
        torch.manual_seed(123)
        expected_next = torch.rand(3)
        torch.manual_seed(123)
        effective_dimension(layer, X, n_data=1000, n_theta_samples=3, init="restricted")
        assert torch.equal(layer.qlayer.weights.detach(), before)
        assert torch.equal(torch.rand(3), expected_next)

    def test_is_reproducible_with_a_generator(self) -> None:
        layer = _layer(3, 2)
        X = _angles(4, 3)
        a = effective_dimension(
            layer, X, n_data=1000, n_theta_samples=3, generator=torch.Generator().manual_seed(5)
        )
        b = effective_dimension(
            layer, X, n_data=1000, n_theta_samples=3, generator=torch.Generator().manual_seed(5)
        )
        assert a.effective_dimension == b.effective_dimension

    def test_dead_directions_lower_the_effective_dimension(self) -> None:
        """
        At x = 0 four of the six parameters of the two-qubit layer have no
        effect (see the hand computation), so the effective dimension is
        well below that with inputs spread over (-π, π).
        """
        layer = _layer()
        # Same seed for both, so the two runs see the same parameter draws
        # and differ only in their inputs.
        at_zero = effective_dimension(
            layer,
            torch.zeros(8, 2),
            n_data=1000,
            n_theta_samples=10,
            generator=torch.Generator().manual_seed(0),
        )
        spread = effective_dimension(
            layer,
            _angles(8, 2),
            n_data=1000,
            n_theta_samples=10,
            generator=torch.Generator().manual_seed(0),
        )
        assert at_zero.normalized_effective_dimension < 0.8 * spread.normalized_effective_dimension

    def test_works_on_a_classifier(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(n_input_features=4, n_qubits=3, n_layers=2, **CPU)
        result = effective_dimension(model, torch.randn(5, 4), n_data=200, n_theta_samples=2)
        assert result.n_params == 18
        assert 0.0 < result.effective_dimension <= _identity_bound(18, 200)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="n_theta_samples"):
            effective_dimension(_layer(), _angles(2, 2), n_theta_samples=0)
        with pytest.raises(ValueError, match="n_data"):
            effective_dimension(_layer(), _angles(1, 2), n_theta_samples=1)
        with pytest.raises(TypeError, match="effective_dimension expects"):
            effective_dimension(torch.nn.Linear(2, 1), _angles(2, 2), n_data=1000)

    def test_results_compare_without_touching_the_tensors(self) -> None:
        layer = _layer()
        X = _angles(3, 2)
        a, again = (
            effective_dimension(
                layer,
                X,
                n_data=1000,
                n_theta_samples=2,
                generator=torch.Generator().manual_seed(0),
            )
            for _ in range(2)
        )
        assert (a == again) is True


# ---------------------------------------------------------------------------
# Circuit executions per Fisher matrix (#223)
# ---------------------------------------------------------------------------


def _lightning_works() -> bool:
    try:
        qml.device("lightning.qubit", wires=1)
    except Exception:  # noqa: BLE001 - any failure means "not installed"
        return False
    return True


def _executions(model: torch.nn.Module, X: torch.Tensor) -> int:
    layer = getattr(model, "quantum_layer", model)
    qlayer = layer.qlayer
    assert isinstance(qlayer, qml.qnn.TorchLayer)
    with qml.Tracker(qlayer.qnode.device) as tracker:
        fisher_information_matrix(model, X)
    return int(tracker.totals["executions"])


class TestExecutionCount:
    """
    The k backward passes per row reuse the forward's Jacobian, so the cost in
    circuit executions does not grow with the number of outputs.
    """

    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
    @pytest.mark.parametrize("readout", ["all", "first"])
    def test_one_execution_per_row_on_default_qubit(self, diff_method: str, readout: str) -> None:
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(
            n_qubits=3,
            n_layers=2,
            readout=readout,  # type: ignore[arg-type]
            device_name="default.qubit",
            diff_method=diff_method,  # type: ignore[arg-type]
        )
        assert _executions(layer, torch.rand(5, 3)) == 5

    def test_classifier_on_default_qubit(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(4, 3, 2, device_name="default.qubit", diff_method="adjoint")
        assert _executions(model, torch.rand(5, 4)) == 5

    @pytest.mark.skipif(not _lightning_works(), reason="pennylane-lightning not installed")
    def test_two_executions_per_row_on_lightning(self) -> None:
        # The forward pass, and the adjoint Jacobian its three outputs share.
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(
            n_qubits=3, n_layers=2, device_name="lightning.qubit", diff_method="adjoint"
        )
        assert layer.qlayer.qnode.device.name == "lightning.qubit"
        assert _executions(layer, torch.rand(5, 3)) == 10

    def test_parameter_shift_repeats_its_shifts_per_output(self) -> None:
        # The one real k factor: per row, the forward plus 2d shifted circuits
        # for each of the k = 3 outputs, d = 2 layers x 3 qubits x 3 angles.
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(
            n_qubits=3, n_layers=2, device_name="default.qubit", diff_method="parameter-shift"
        )
        d = layer.qlayer.weights.numel()
        assert d == 18
        assert _executions(layer, torch.rand(5, 3)) == 5 * (1 + 3 * 2 * d)  # 545
