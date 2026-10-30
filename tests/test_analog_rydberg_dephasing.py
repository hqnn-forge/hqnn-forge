"""
tests/test_analog_rydberg_dephasing.py
======================================
``AnalogRydbergLayer(gamma=...)`` (#699): Markovian dephasing ``L_i = √γ n_i``
during the pulse, against values obtained another way:

* ``gamma = 0`` is the layer as it was, and a vanishing ``gamma`` on the
  density-matrix path gives the pure state of the unitary path;
* one resonant atom follows the damped Rabi oscillation of
  ``docs/rydberg-model.md``, typed in here, in all three damping regimes;
* two atoms under a piecewise pulse match the exponential of the full
  Lindblad generator of each segment, assembled here from Kronecker products
  (no ``rydberg_hamiltonian``, no eigendecomposition, no splitting);
* gradients match central finite differences.

Every layer is converted to ``float64`` (``.double()``) before a pulse is set,
as in ``tests/test_analog_rydberg_embedding.py``.
"""

from __future__ import annotations

import math
from functools import reduce

import numpy as np
import pytest
import torch

from hqnn_forge.encoding import AnalogRydbergLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.noise import apply_depolarizing_noise, noise_sweep
from hqnn_forge.rydberg import AtomRegister

F64 = torch.float64
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1j], [1j, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
N_OP = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)  # n = |r⟩⟨r|, |r⟩ = |1⟩
EYE = np.eye(2, dtype=complex)


def _layer(n_qubits: int, n_segments: int, **kwargs: object) -> AnalogRydbergLayer:
    torch.manual_seed(0)
    return AnalogRydbergLayer(n_qubits, n_segments, **kwargs).double()  # type: ignore[arg-type]


def _on(op: np.ndarray, i: int, n: int) -> np.ndarray:
    """``op`` on tensor factor ``i`` of ``n`` (atom 0 leftmost, the most significant bit)."""
    return reduce(np.kron, [op if j == i else EYE for j in range(n)])


def damped_rabi(omega: float, gamma: float, t: float) -> float:
    """
    ``⟨n⟩(t)`` of one resonant atom from ``|g⟩`` under ``L = √γ n``, typed in from
    the *Noise* section of ``docs/rydberg-model.md``: underdamped below
    ``γ = 4Ω``, critical at it, overdamped above.
    """
    if gamma < 4 * omega:
        lam = math.sqrt(omega**2 - gamma**2 / 16)
        bracket = math.cos(lam * t) + gamma / (4 * lam) * math.sin(lam * t)
        return 0.5 - 0.5 * math.exp(-gamma * t / 4) * bracket
    if gamma == 4 * omega:
        return 0.5 - 0.5 * math.exp(-omega * t) * (1 + omega * t)
    kappa = math.sqrt(gamma**2 / 16 - omega**2)
    bracket = math.cosh(kappa * t) + gamma / (4 * kappa) * math.sinh(kappa * t)
    return 0.5 - 0.5 * math.exp(-gamma * t / 4) * bracket


def _liouvillian(h: np.ndarray, gamma: float) -> np.ndarray:
    """
    The generator on ρ flattened row by row, ``vec(A ρ B) = (A ⊗ Bᵀ) vec(ρ)``::

        𝓛 = −i (H ⊗ 1 − 1 ⊗ Hᵀ) + γ Σ_i ( n_i ⊗ n_i − ½ n_i ⊗ 1 − ½ 1 ⊗ n_i )

    ``n_i`` is real and diagonal, so ``n_iᵀ = n_i``.
    """
    dim = h.shape[0]
    n = dim.bit_length() - 1
    eye = np.eye(dim)
    generator = -1j * (np.kron(h, eye) - np.kron(eye, h.T))
    for i in range(n):
        n_i = _on(N_OP, i, n)
        generator += gamma * (
            np.kron(n_i, n_i) - 0.5 * np.kron(n_i, eye) - 0.5 * np.kron(eye, n_i)
        )
    return generator


def _reference_density_matrix(
    layer: AnalogRydbergLayer, x: np.ndarray, gamma: float
) -> np.ndarray:
    """``ρ(T) = e^(𝓛_{K−1} τ) ⋯ e^(𝓛_0 τ) ρ(0)`` from the layer's pulse, by ``scipy.linalg.expm``."""
    linalg = pytest.importorskip("scipy.linalg")
    dim = 2**layer.n_qubits
    rho = np.zeros(dim * dim, dtype=complex)
    rho[0] = 1.0  # |g…g⟩⟨g…g|
    for h in _reference_hamiltonians(layer, x):
        rho = linalg.expm(_liouvillian(h, gamma) * layer.segment_duration) @ rho
    return rho.reshape(dim, dim)


def _reference_hamiltonians(layer: AnalogRydbergLayer, x: np.ndarray) -> list[np.ndarray]:
    """``H_k(x)`` of every segment, from the layer's pulse and register, built with ``np.kron``."""
    pulse = layer.pulse().numpy()
    local = layer.local_detuning_max / (1 + np.exp(-x))  # Δ_max,loc · σ(x_i)
    positions = layer.register.positions.numpy()
    n = layer.n_qubits
    dim = 2**n
    hamiltonians = []
    for omega, detuning, phase in pulse:
        h = np.zeros((dim, dim), dtype=complex)
        for i in range(n):
            drive = math.cos(phase) * X - math.sin(phase) * Y
            h += omega / 2 * _on(drive, i, n) - (detuning + local[i]) * _on(N_OP, i, n)
            for j in range(i + 1, n):
                v = layer.c6 / np.linalg.norm(positions[i] - positions[j]) ** 6
                h += v * _on(N_OP, i, n) @ _on(N_OP, j, n)
        hamiltonians.append(h)
    return hamiltonians


def _central_difference(function: object, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``∂f/∂x_j ≈ (f(x + ε e_j) − f(x − ε e_j)) / (2ε)`` for a scalar ``f``."""
    flat = x.detach().reshape(-1)
    grad = torch.zeros_like(flat)
    for j in range(flat.numel()):
        step = torch.zeros_like(flat)
        step[j] = eps
        plus = function((flat + step).reshape(x.shape))  # type: ignore[operator]
        minus = function((flat - step).reshape(x.shape))  # type: ignore[operator]
        grad[j] = (plus - minus) / (2 * eps)
    return grad.reshape(x.shape)


# ---------------------------------------------------------------------------
# gamma = 0
# ---------------------------------------------------------------------------


class TestWithoutDephasing:
    def test_gamma_zero_is_the_layer_without_it(self) -> None:
        """The unitary circuit on ``default.qubit``, bit for bit."""
        x = torch.randn(4, 3, dtype=F64, generator=torch.Generator().manual_seed(1))
        plain = _layer(3, 3, segment_duration=0.2)
        zero = _layer(3, 3, segment_duration=0.2, gamma=0.0, n_steps=7)
        for layer in (plain, zero):
            layer.set_pulse(omega=[6.0, 3.0, 9.0], detuning=[1.0, -2.0, 0.5], phase=[0, 1, 2])
        assert torch.equal(zero(x), plain(x))
        assert zero.qlayer.qnode.device.name == plain.qlayer.qnode.device.name == "default.qubit"
        assert zero.gamma == 0.0

    def test_a_vanishing_rate_on_the_density_path_gives_the_pure_state(self) -> None:
        """
        ``γ = 1e-12``: ``default.mixed`` with ``ρ(T)`` from the chained ``evolve``
        calls, against ``ψψ†`` of the unitary path and its ⟨Z_i⟩.  The
        dissipator changes ρ by ``O(γT) = 1e-12``.
        """
        x = torch.randn(4, 3, dtype=F64, generator=torch.Generator().manual_seed(2))
        pure = _layer(3, 3, segment_duration=0.2)
        mixed = _layer(3, 3, segment_duration=0.2, gamma=1e-12, n_steps=3)
        for layer in (pure, mixed):
            layer.set_pulse(omega=[6.0, 3.0, 9.0], detuning=[1.0, -2.0, 0.5], phase=[0, 1, 2])
        assert mixed.qlayer.qnode.device.name == "default.mixed"
        psi = pure.states(x).detach()
        rho = mixed.density_matrices(x).detach()
        torch.testing.assert_close(
            rho, psi[..., :, None] * psi[..., None, :].conj(), rtol=0, atol=1e-11
        )
        torch.testing.assert_close(mixed(x), pure(x), rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------
# The damped Rabi oscillation
# ---------------------------------------------------------------------------


class TestDampedRabi:
    """
    One atom at total detuning 0: the input ``x = −40`` puts the local
    detuning at ``Δ_max,loc · σ(−40) ≈ 9e-17`` rad/µs, and the global detuning
    is 0 exactly (raw weight 0).  Every segment has the same Ω and φ, so
    ``K`` segments are one constant pulse of duration ``K τ``.
    """

    OMEGA = 2.0
    #: As in ``tests/test_rydberg_lindblad.py``: 400 steps per unit of ``Ωt``
    #: up to ``γ = 4Ω``, per unit of ``γt/4`` above it, error below 1e-6.
    STEPS = 400

    @pytest.mark.parametrize("gamma_over_omega", [0.1, 1.0, 3.9, 4.0, 4.1, 8.0])
    @pytest.mark.parametrize(("n_segments", "omega_tau"), [(1, math.pi), (3, 2.0)])
    def test_closed_form_in_all_three_cases(
        self, gamma_over_omega: float, n_segments: int, omega_tau: float
    ) -> None:
        gamma = gamma_over_omega * self.OMEGA
        tau = omega_tau / self.OMEGA
        n_steps = math.ceil(self.STEPS * omega_tau * max(1.0, gamma_over_omega / 4))
        layer = _layer(
            1,
            n_segments,
            omega_max=2 * self.OMEGA,
            segment_duration=tau,
            gamma=gamma,
            n_steps=n_steps,
        )
        layer.set_pulse(omega=self.OMEGA, phase=0.7)  # a constant phase is a gauge
        x = torch.tensor([[-40.0]], dtype=F64)
        assert layer.local_detunings(x).item() < 1e-15
        assert layer.pulse()[:, 1].abs().max().item() == 0.0
        n_expected = damped_rabi(self.OMEGA, gamma, n_segments * tau)
        assert layer(x).item() == pytest.approx(1 - 2 * n_expected, abs=2e-6)


# ---------------------------------------------------------------------------
# Interactions and dephasing together
# ---------------------------------------------------------------------------


class TestAgainstTheLiouvillian:
    @pytest.mark.parametrize("gamma", [0.5, 4.0])
    def test_a_piecewise_pulse_on_two_atoms(self, gamma: float) -> None:
        """
        Two segments with different Ω, Δ, φ; different inputs; ``V ≈ 2.6 Ω``:
        nothing is symmetric, so a swapped atom, a sign of φ or a segment
        not starting from the previous state would show.
        """
        register = AtomRegister([[0.0, 0.0], [1.0, 0.3]])
        layer = _layer(
            2,
            2,
            register=register,
            c6=6.0,
            omega_max=6.0,
            segment_duration=0.4,
            gamma=gamma,
            n_steps=2000,
        )
        layer.set_pulse(omega=[2.0, 3.5], detuning=[0.8, -1.5], phase=[0.0, 1.3])
        x = torch.tensor([[0.3, -1.1]], dtype=F64)
        expected = _reference_density_matrix(layer, x[0].numpy(), gamma)
        rho = layer.density_matrices(x)[0].detach().numpy()
        np.testing.assert_allclose(rho, expected, rtol=0, atol=2e-6)
        z = [np.real(np.trace(expected @ _on(Z, i, 2))) for i in range(2)]
        np.testing.assert_allclose(layer(x)[0].detach().numpy(), z, rtol=0, atol=4e-6)
        # the dephasing matters at this tolerance
        assert np.abs(expected - _reference_density_matrix(layer, x[0].numpy(), 0.0)).max() > 1e-2

    def test_a_batch_equals_its_samples_one_by_one(self) -> None:
        layer = _layer(3, 2, segment_duration=0.2, gamma=1.5, n_steps=5)
        layer.set_pulse(omega=[6.0, 3.0], detuning=[1.0, -2.0], phase=[0.0, 1.0])
        x = torch.randn(2, 3, 3, dtype=F64, generator=torch.Generator().manual_seed(3))
        batched = layer(x)
        assert batched.shape == (2, 3, 3)
        for i in range(2):
            for j in range(3):
                assert torch.equal(batched[i, j], layer(x[i, j]))

    def test_the_readout_is_the_diagonal_of_the_density_matrix(self) -> None:
        """``⟨Z_i⟩ = 1 − 2 Σ_k b_i(k) ρ_kk``, atom 0 the most significant bit; also ``"first"``."""
        x = torch.tensor([[0.3, -0.5, 1.2]], dtype=F64)
        for readout in ("all", "first"):
            layer = _layer(3, 2, segment_duration=0.2, gamma=2.0, n_steps=4, readout=readout)
            layer.set_pulse(omega=[6.0, 3.0], detuning=[1.0, -2.0], phase=[0.0, 1.0])
            p = layer.density_matrices(x)[0].diagonal().real
            bits = torch.tensor(
                [[(k >> (2 - i)) & 1 for i in range(3)] for k in range(8)], dtype=F64
            )
            z = 1 - 2 * (p @ bits)
            expected = z if readout == "all" else z[:1]
            torch.testing.assert_close(layer(x)[0], expected, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


class TestGradients:
    def test_pulse_and_input_gradients_match_finite_differences(self) -> None:
        layer = _layer(2, 2, segment_duration=0.3, gamma=1.3, n_steps=4)
        layer.set_pulse(omega=[4.0, 6.0], detuning=[1.0, -2.0], phase=[0.0, 0.5])
        x = torch.tensor([[0.2, -0.7]], dtype=F64, requires_grad=True)
        layer(x).sum().backward()
        weights = layer.qlayer.weights
        assert weights.grad is not None and x.grad is not None

        def of_weights(value: torch.Tensor) -> float:
            with torch.no_grad():
                weights.copy_(value)
                return float(layer(x.detach()).sum())

        start = weights.detach().clone()
        expected_w = _central_difference(of_weights, start)
        with torch.no_grad():
            weights.copy_(start)

        def of_inputs(value: torch.Tensor) -> float:
            with torch.no_grad():
                return float(layer(value).sum())

        expected_x = _central_difference(of_inputs, x.detach().clone())
        torch.testing.assert_close(weights.grad, expected_w, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(x.grad, expected_x, rtol=1e-6, atol=1e-8)
        assert weights.grad.abs().min().item() > 1e-4  # every pulse parameter matters

    def test_a_degenerate_spectrum_still_has_the_right_gradient(self) -> None:
        """Equal inputs without interactions (#505): the two input gradients are equal."""
        layer = _layer(2, 2, interactions=False, segment_duration=0.3, gamma=0.8, n_steps=6)
        layer.set_pulse(omega=[4.0, 6.0], detuning=[1.0, -2.0], phase=[0.0, 0.5])
        x = torch.tensor([[0.4, 0.4]], dtype=F64, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert x.grad[0, 0].item() == pytest.approx(x.grad[0, 1].item(), rel=1e-10)

        def loss(value: torch.Tensor) -> float:
            with torch.no_grad():
                return float(layer(value).sum())

        torch.testing.assert_close(
            x.grad, _central_difference(loss, x.detach().clone()), rtol=1e-6, atol=1e-8
        )


# ---------------------------------------------------------------------------
# Options, validation and the gate-model noise
# ---------------------------------------------------------------------------


class TestOptions:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"gamma": 1.0}, "n_steps is required when gamma > 0"),
            ({"gamma": -0.1, "n_steps": 3}, "gamma must be a finite number >= 0"),
            ({"gamma": math.nan, "n_steps": 3}, "gamma must be a finite number >= 0"),
            ({"gamma": True, "n_steps": 3}, "gamma must be a finite number >= 0"),
            ({"gamma": 1.0, "n_steps": 0}, "n_steps must be an integer >= 1"),
            ({"gamma": 1.0, "n_steps": 2.0}, "n_steps must be an integer >= 1"),
            ({"gamma": 0.0, "n_steps": -1}, "n_steps must be an integer >= 1"),
            (
                {"gamma": 1.0, "n_steps": 3, "noise_level": 0.1},
                "noise_position='all' places a channel after every pulse segment",
            ),
        ],
    )
    def test_refuses_bad_options(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            AnalogRydbergLayer(2, 2, **kwargs)  # type: ignore[arg-type]

    def test_the_circuit_takes_one_sample_at_a_time(self) -> None:
        layer = _layer(2, 1, gamma=1.0, n_steps=2)
        weights = layer.qlayer.weights.detach()
        with pytest.raises(ValueError, match=r"one sample of shape \(2,\) at a time"):
            layer.qlayer.qnode(torch.zeros(3, 2, dtype=F64), weights)

    def test_repr_states_the_rate_and_the_steps(self) -> None:
        assert "gamma=1.5, n_steps=4" in repr(_layer(2, 1, gamma=1.5, n_steps=4))
        assert "gamma" not in repr(_layer(2, 1))

    def test_depolarizing_at_the_end_scales_every_readout(self) -> None:
        """``noise_position="end"`` acts on ``ρ(T)``: ⟨Z_i⟩ · (1 − 4p/3), in train mode only."""
        x = torch.tensor([[0.3, -0.5, 1.2]], dtype=F64)
        layer = _layer(
            3,
            2,
            segment_duration=0.2,
            gamma=2.0,
            n_steps=4,
            noise_level=0.15,
            noise_position="end",
        )
        layer.set_pulse(omega=[6.0, 3.0], detuning=[1.0, -2.0], phase=[0.0, 1.0])
        exact = layer.eval()(x)
        noisy = layer.train()(x)
        torch.testing.assert_close(noisy, exact * (1 - 4 * 0.15 / 3), rtol=1e-10, atol=1e-12)
        with apply_depolarizing_noise(layer.eval(), 0.15, position="end"):
            torch.testing.assert_close(layer(x), noisy, rtol=1e-10, atol=1e-12)

    def test_readout_error_is_the_affine_map_of_z(self) -> None:
        x = torch.tensor([[0.3, -0.5]], dtype=F64)
        layer = _layer(2, 2, gamma=2.0, n_steps=4, readout_error=(0.05, 0.1))
        exact = layer.eval()(x)
        torch.testing.assert_close(layer.train()(x), (1 - 0.15) * exact + (0.1 - 0.05))


class TestEveryPulseSegment:
    """
    ``position="all"`` means a channel after every pulse segment (#710).

    With ``gamma > 0`` the circuit has one preparation and no gate between
    segments, so ``"all"`` would act once, as ``"end"`` does, and is refused
    on every path; with ``gamma = 0`` it acts ``K`` times, as before.
    """

    MESSAGE = (
        "position='all' places a channel after every pulse segment, and with gamma > 0 the "
        "segments are evolved outside the circuit, which prepares only their final density "
        "matrix: there is no gate between segments to put the channel after, and the channel "
        "would act once, as at 'end'.  Use position='end' (once before measurement), or gamma=0."
    )

    def test_gamma_zero_acts_after_every_segment(self) -> None:
        """The hand-built ``ρ ← D^{⊗N}(U_k ρ U_k†)``, segment by segment, unchanged by #699."""
        linalg = pytest.importorskip("scipy.linalg")
        p = 0.1
        layer = _layer(2, 3, segment_duration=0.2)
        layer.set_pulse(omega=[6.0, 3.0, 4.5], detuning=[1.0, -2.0, 0.5], phase=[0.0, 1.0, 2.0])
        x = np.array([0.3, -0.5])
        n = layer.n_qubits
        rho = np.zeros((2**n, 2**n), dtype=complex)
        rho[0, 0] = 1.0
        for h in _reference_hamiltonians(layer, x):
            u = linalg.expm(-1j * h * layer.segment_duration)
            rho = u @ rho @ u.conj().T
            for i in range(n):
                paulis = [_on(op, i, n) for op in (X, Y, Z)]
                rho = (1 - p) * rho + p / 3 * sum(o @ rho @ o for o in paulis)
        expected = [np.trace(rho @ _on(Z, i, n)).real for i in range(n)]
        with apply_depolarizing_noise(layer, p, position="all"):
            every = layer(torch.tensor(x[None], dtype=F64))
        torch.testing.assert_close(
            every, torch.tensor([expected], dtype=F64), rtol=1e-10, atol=1e-12
        )

    def test_post_hoc_all_is_refused_with_dephasing(self) -> None:
        layer = _layer(2, 3, gamma=0.5, n_steps=4).eval()
        x = torch.tensor([[0.3, -0.5]], dtype=F64)
        qnode = layer.qlayer.qnode
        with (
            pytest.raises(ValueError) as error,
            apply_depolarizing_noise(layer, 0.05, position="all"),
        ):
            pass
        assert str(error.value) == self.MESSAGE
        # Refused before the layer was touched: the same QNode, no block left open.
        assert layer.qlayer.qnode is qnode
        assert getattr(layer.qlayer, "_hqnn_noise_depth", 0) == 0
        exact = layer(x)
        with apply_depolarizing_noise(layer, 0.05, position="end"):
            torch.testing.assert_close(layer(x), exact * (1 - 4 * 0.05 / 3), rtol=1e-10, atol=0)

    def test_post_hoc_all_at_p_zero_is_the_noiseless_layer(self) -> None:
        layer = _layer(2, 2, gamma=0.5, n_steps=4).eval()
        x = torch.tensor([[0.3, -0.5]], dtype=F64)
        exact = layer(x)
        with apply_depolarizing_noise(layer, 0.0, position="all"):
            assert torch.equal(layer(x), exact)

    def test_noise_sweep_refuses_before_the_first_evaluation(self) -> None:
        model = HybridBinaryClassifier(
            n_input_features=2,
            n_qubits=2,
            n_layers=2,
            encoding_type="analog",
            gamma=0.5,
            n_steps=4,
        )
        calls = []
        predict = model.predict_proba

        def counting(x: torch.Tensor) -> torch.Tensor:
            calls.append(x)
            return predict(x)

        model.predict_proba = counting  # type: ignore[method-assign]
        x = torch.zeros(1, 2)
        with pytest.raises(ValueError) as error:
            noise_sweep(model, x, [0.0, 0.05], position="all")
        assert str(error.value) == self.MESSAGE
        assert calls == []
        points = noise_sweep(model, x, [0.0, 0.05], position="end")
        assert [point.p for point in points] == [0.0, 0.05]
        assert len(calls) == 2

    def test_training_noise_names_its_own_argument(self) -> None:
        with pytest.raises(ValueError) as error:
            AnalogRydbergLayer(2, 2, gamma=1.0, n_steps=3, noise_level=0.1)
        assert str(error.value) == self.MESSAGE.replace("position=", "noise_position=")


class TestClassifiers:
    @pytest.mark.parametrize(
        "model_class",
        [HybridBinaryClassifier, ParallelHybridClassifier, MulticlassHybridClassifier],
    )
    def test_the_option_reaches_the_layer_and_the_config(self, model_class: type) -> None:
        extra = {"n_classes": 3} if model_class is MulticlassHybridClassifier else {}
        model = model_class(
            n_input_features=4,
            n_qubits=2,
            n_layers=2,
            encoding_type="analog",
            gamma=1.5,
            n_steps=3,
            **extra,
        )
        layer = model.quantum_layer
        assert isinstance(layer, AnalogRydbergLayer)
        assert (layer.gamma, layer.n_steps) == (1.5, 3)
        config = model.get_config()
        assert (config["gamma"], config["n_steps"]) == (1.5, 3)
        rebuilt = model_class(**config)
        rebuilt.load_state_dict(model.state_dict())
        x = torch.randn(3, 4)
        model.eval()
        rebuilt.eval()
        torch.testing.assert_close(rebuilt(x), model(x))

    @pytest.mark.parametrize("encoding_type", ["angle", "iqp", "amplitude", "reuploading"])
    def test_gamma_applies_to_the_analog_encoding_only(self, encoding_type: str) -> None:
        with pytest.raises(ValueError, match="gamma and n_steps apply to encoding_type='analog'"):
            HybridBinaryClassifier(
                n_input_features=2, n_qubits=2, encoding_type=encoding_type, gamma=1.0, n_steps=3
            )

    def test_n_steps_alone_applies_to_the_analog_encoding_only(self) -> None:
        with pytest.raises(ValueError, match="gamma and n_steps apply to encoding_type='analog'"):
            HybridBinaryClassifier(n_input_features=2, n_qubits=2, n_steps=3)

    def test_the_model_refuses_what_the_layer_refuses(self) -> None:
        with pytest.raises(ValueError, match="n_steps is required when gamma > 0"):
            HybridBinaryClassifier(
                n_input_features=2, n_qubits=2, encoding_type="analog", gamma=1.0
            )
