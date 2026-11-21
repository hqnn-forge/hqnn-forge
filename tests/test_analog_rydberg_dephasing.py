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
* gradients match central finite differences;
* the two-run estimate of a chain of segments with one shared step count
  (#760) is the error of the layer's chain against that exponential, and the
  count of ``AnalogRydbergLayer.step_count`` meets ``tol`` against it (#769).

Every layer is converted to ``float64`` (``.double()``) before a pulse is set,
as in ``tests/test_analog_rydberg_embedding.py``.
"""

from __future__ import annotations

import math
from functools import reduce
from typing import get_args

import numpy as np
import pytest
import torch

from hqnn_forge.encoding import AnalogRydbergLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.noise import Channel, Position, apply_depolarizing_noise, noise_sweep
from hqnn_forge.rydberg import AtomRegister
from hqnn_forge.rydberg import readout as rydberg_readout

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


#: The Kraus operators of every channel of ``hqnn_forge.noise.CHANNELS`` at
#: strength ``p``, typed in here from the channel definitions (PennyLane's
#: ``DepolarizingChannel``, ``AmplitudeDamping``, ``PhaseDamping``,
#: ``BitFlip``, ``PhaseFlip`` and ``ResetError(0, p)``), not taken from the
#: library.
KRAUS = {
    "depolarizing": lambda p: [math.sqrt(1 - p) * EYE] + [math.sqrt(p / 3) * P for P in (X, Y, Z)],
    "amplitude_damping": lambda p: [
        np.array([[1.0, 0.0], [0.0, math.sqrt(1 - p)]], dtype=complex),
        np.array([[0.0, math.sqrt(p)], [0.0, 0.0]], dtype=complex),
    ],
    "phase_damping": lambda p: [
        np.array([[1.0, 0.0], [0.0, math.sqrt(1 - p)]], dtype=complex),
        np.array([[0.0, 0.0], [0.0, math.sqrt(p)]], dtype=complex),
    ],
    "bit_flip": lambda p: [math.sqrt(1 - p) * EYE, math.sqrt(p) * X],
    "phase_flip": lambda p: [math.sqrt(1 - p) * EYE, math.sqrt(p) * Z],
    "atom_loss": lambda p: [
        math.sqrt(1 - p) * EYE,
        np.array([[0.0, 0.0], [math.sqrt(p), 0.0]], dtype=complex),  # |1⟩⟨0|
        np.array([[0.0, 0.0], [0.0, math.sqrt(p)]], dtype=complex),  # |1⟩⟨1|
    ],
}
CHANNEL_NAMES: list[Channel] = list(get_args(Channel))


def _channel_on_every_atom(rho: np.ndarray, channel: str, p: float) -> np.ndarray:
    """``ρ ← Σ_k K_k ρ K_k†`` on atom 0, then atom 1, …, with ``K_k`` on tensor factor ``i``."""
    n = rho.shape[0].bit_length() - 1
    for i in range(n):
        ops = [_on(k, i, n) for k in KRAUS[channel](p)]
        rho = sum(k @ rho @ k.conj().T for k in ops)
    return rho


def _reference_with_channel(
    layer: AnalogRydbergLayer, x: np.ndarray, gamma: float, channel: str, p: float
) -> np.ndarray:
    """
    ``ρ ← D^{⊗N}(e^(𝓛_k τ) ρ)`` for ``k = 0, …, K−1`` from ``|g…g⟩⟨g…g|``: the
    Lindblad generator of each segment exponentiated by ``scipy.linalg.expm``,
    then the channel's Kraus map on every atom (``γ = 0`` gives
    ``U_k ρ U_k†`` exactly).
    """
    linalg = pytest.importorskip("scipy.linalg")
    dim = 2**layer.n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    for h in _reference_hamiltonians(layer, x):
        step = linalg.expm(_liouvillian(h, gamma) * layer.segment_duration)
        rho = (step @ rho.reshape(-1)).reshape(dim, dim)  # row-major vec, as _liouvillian
        rho = _channel_on_every_atom(rho, channel, p)
    return rho


def _z(rho: np.ndarray) -> list[float]:
    n = rho.shape[0].bit_length() - 1
    return [float(np.trace(rho @ _on(Z, i, n)).real) for i in range(n)]


class TestEveryPulseSegment:
    """
    ``position="all"`` means a channel after every pulse segment on every
    atom, ``K`` times per atom, with and without dephasing (#710, #717).

    With ``gamma > 0`` the channel is applied to ρ in torch between the
    segments' ``evolve`` calls; with ``gamma = 0`` ``qml.noise.insert`` puts
    it after every ``QubitUnitary``.  Both are checked against the same
    reference, :func:`_reference_with_channel`.
    """

    P = 0.2
    #: Two atoms with a strong but not blockading interaction, three
    #: segments with different Ω, Δ, φ, different inputs: nothing symmetric.
    REGISTER = AtomRegister([[0.0, 0.0], [1.0, 0.3]])
    X_IN = np.array([0.3, -1.1])

    def _layer(self, gamma: float, n_steps: int | None = None, **kwargs: object):
        layer = _layer(
            2,
            3,
            register=self.REGISTER,
            c6=6.0,
            omega_max=6.0,
            segment_duration=0.4,
            gamma=gamma,
            n_steps=n_steps,
            **kwargs,
        )
        layer.set_pulse(omega=[2.0, 3.5, 5.0], detuning=[0.8, -1.5, 0.3], phase=[0.0, 1.3, -0.4])
        return layer

    def _x(self) -> torch.Tensor:
        return torch.tensor(self.X_IN[None], dtype=F64)

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

    @pytest.mark.parametrize("channel", CHANNEL_NAMES)
    def test_a_vanishing_rate_equals_all_on_the_unitary_layer(self, channel: Channel) -> None:
        """
        ``γ = 1e-12``: the channel applied in torch between the ``evolve``
        calls against the hand-built ``ρ ← D^{⊗N}(U_k ρ U_k†)`` and against
        PennyLane's channels inserted after every ``QubitUnitary`` of the
        unitary layer (``γ = 0``).  ``O(γT) ≈ 1e-12``.
        """
        x = self._x()
        reference = _z(_reference_with_channel(self._layer(0.0), self.X_IN, 0.0, channel, self.P))
        unitary = self._layer(0.0).eval()
        dephased = self._layer(1e-12, n_steps=3).eval()
        assert dephased.qlayer.qnode.device.name == "default.mixed"
        with apply_depolarizing_noise(unitary, self.P, position="all", channel=channel):
            on_gates = unitary(x)
        with apply_depolarizing_noise(dephased, self.P, position="all", channel=channel):
            in_torch = dephased(x)
        expected = torch.tensor([reference], dtype=F64)
        torch.testing.assert_close(on_gates, expected, rtol=0, atol=1e-10)
        torch.testing.assert_close(in_torch, expected, rtol=0, atol=1e-10)
        torch.testing.assert_close(in_torch, on_gates, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("channel", CHANNEL_NAMES)
    def test_dephasing_against_the_liouvillian_then_the_kraus_map(self, channel: Channel) -> None:
        """
        ``γ = 1.5``: per segment ``expm`` of the Lindblad generator, then the
        channel on every atom; ⟨Z_i⟩ through ``apply_depolarizing_noise``, and
        for the depolarising channel the whole ρ through
        ``encoded_density_matrices``.  ``n_steps = 2000`` puts the splitting
        error far below the tolerance.
        """
        from hqnn_forge.kernels import encoded_density_matrices

        gamma = 1.5
        layer = self._layer(gamma, n_steps=2000).eval()
        x = self._x()
        expected = _reference_with_channel(layer, self.X_IN, gamma, channel, self.P)
        with apply_depolarizing_noise(layer, self.P, position="all", channel=channel):
            z = layer(x)
        np.testing.assert_allclose(z[0].detach().numpy(), _z(expected), rtol=0, atol=4e-6)
        if channel == "depolarizing":  # the kernels insert depolarising channels only
            rho = encoded_density_matrices(x, layer, noise_level=self.P, noise_position="all")
            np.testing.assert_allclose(rho[0].numpy(), expected, rtol=0, atol=2e-6)
        # No channel, or one channel at the end only, is far from it.
        noiseless = _reference_with_channel(layer, self.X_IN, gamma, channel, 0.0)
        once = _channel_on_every_atom(noiseless, channel, self.P)
        for other in (noiseless, once):
            assert np.abs(expected - other).max() > 1e-3

    def test_training_noise_acts_after_every_segment(self) -> None:
        """``noise_position="all"`` at construction, in train mode only; same reference."""
        gamma = 1.5
        channel: Channel = "amplitude_damping"
        layer = self._layer(
            gamma,
            n_steps=2000,
            noise_level=self.P,
            noise_position="all",
            noise_channel=channel,
        )
        x = self._x()
        expected = _reference_with_channel(layer, self.X_IN, gamma, channel, self.P)
        noisy = layer.train()(x)
        np.testing.assert_allclose(noisy[0].detach().numpy(), _z(expected), rtol=0, atol=4e-6)
        noiseless = _reference_with_channel(layer, self.X_IN, gamma, channel, 0.0)
        np.testing.assert_allclose(
            layer.eval()(x)[0].detach().numpy(), _z(noiseless), rtol=0, atol=4e-6
        )
        assert "noise_position='all'" in repr(layer)

    def test_training_noise_gradients_match_finite_differences(self) -> None:
        layer = self._layer(1.3, n_steps=4, noise_level=0.1, noise_position="all").train()
        x = self._x()
        layer(x).sum().backward()
        weights = layer.qlayer.weights
        assert weights.grad is not None

        def of_weights(value: torch.Tensor) -> float:
            with torch.no_grad():
                weights.copy_(value)
                return float(layer(x).sum())

        start = weights.detach().clone()
        expected = _central_difference(of_weights, start)
        torch.testing.assert_close(weights.grad, expected, rtol=1e-6, atol=1e-8)
        assert weights.grad.abs().min().item() > 1e-4

    def test_post_hoc_all_at_p_zero_is_the_noiseless_layer(self) -> None:
        layer = _layer(2, 2, gamma=0.5, n_steps=4).eval()
        x = torch.tensor([[0.3, -0.5]], dtype=F64)
        exact = layer(x)
        with apply_depolarizing_noise(layer, 0.0, position="all"):
            assert torch.equal(layer(x), exact)

    def test_noise_sweep_reaches_every_segment(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            n_input_features=2,
            n_qubits=2,
            n_layers=3,
            encoding_type="analog",
            gamma=0.5,
            n_steps=4,
        ).double()
        x = torch.randn(3, 2, dtype=F64, generator=torch.Generator().manual_seed(5))
        points = noise_sweep(model, x, [0.0, 0.1], position="all")
        with apply_depolarizing_noise(model, 0.1, position="all"):
            every = model.predict_proba(x)
        with apply_depolarizing_noise(model, 0.1, position="end"):
            end = model.predict_proba(x)
        assert [point.p for point in points] == [0.0, 0.1]
        torch.testing.assert_close(points[0].probabilities, model.predict_proba(x))
        torch.testing.assert_close(points[1].probabilities, every)
        assert (every - end).abs().max() > 1e-4  # "all" is not "end"

    def test_cz_is_still_refused(self) -> None:
        layer = self._layer(0.5, n_steps=4).eval()
        with (
            pytest.raises(ValueError, match="this circuit has no CZ or CCZ"),
            apply_depolarizing_noise(layer, 0.1, position="cz"),
        ):
            layer(self._x())

    TRAJECTORIES_MESSAGE = (
        "noise_method='trajectories' at position 'all' is not available on a circuit that "
        "evolves its segments outside the circuit (AnalogRydbergLayer with gamma > 0): "
        "drawn after its one density-matrix preparation, the errors would act once, as "
        "at 'end', not after every segment, and the circuit holds the 4^N density matrix "
        "anyway, so sampling would save no memory.  Use noise_method='density', which "
        "applies the channel exactly after every segment at that same cost, or position "
        "'end'."
    )

    @pytest.mark.parametrize("channel", CHANNEL_NAMES)
    def test_trajectories_at_all_are_refused(self, channel: Channel) -> None:
        """
        Decided on #743 (#717): refused by ``trajectory_noise_qnode``, and so
        by the layer and the classifiers, for every channel.

        Drawn after ``ρ(T)`` the errors would act once, as at ``"end"``; drawn
        between the segments they would save no memory, since the circuit
        holds ρ anyway (``noise`` module docstring, *Circuits without gates
        between segments*).  The weighted-Kraus channels never reach this
        check on the layer: ``check_kraus_trajectories`` refuses them on
        ``default.mixed`` at any position, as before #717.
        """
        from hqnn_forge.noise import CHANNELS, trajectory_noise_qnode

        qnode = AnalogRydbergLayer(2, 2, gamma=1.0, n_steps=3).qlayer.qnode
        with pytest.raises(ValueError) as error:
            trajectory_noise_qnode(qnode, 0.1, position="all", channel=channel)
        assert str(error.value) == self.TRAJECTORIES_MESSAGE

        def layer(gamma: float, position: Position) -> AnalogRydbergLayer:
            return AnalogRydbergLayer(
                2,
                2,
                gamma=gamma,
                n_steps=3 if gamma > 0 else None,
                noise_level=0.1,
                noise_position=position,
                noise_method="trajectories",
                noise_channel=channel,
            )

        if CHANNELS[channel].paulis is None:
            with pytest.raises(ValueError, match="trajectories apply non-unitary"):
                layer(1.0, "end")
            return
        with pytest.raises(ValueError) as error:
            layer(1.0, "all")
        assert str(error.value) == self.TRAJECTORIES_MESSAGE
        with pytest.raises(ValueError) as error:
            HybridBinaryClassifier(
                n_input_features=2,
                n_qubits=2,
                encoding_type="analog",
                gamma=1.0,
                n_steps=3,
                noise_level=0.1,
                noise_position="all",
                noise_method="trajectories",
                noise_channel=channel,
            )
        assert str(error.value) == self.TRAJECTORIES_MESSAGE
        # Not refused: "end" with gamma > 0, and "all" with gamma = 0, where
        # the draws sit after every QubitUnitary.
        assert layer(1.0, "end").noise_method == "trajectories"
        assert layer(0.0, "all").noise_method == "trajectories"


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

    @pytest.mark.parametrize("use_classical_encoder", [True, False])
    def test_the_step_count_recipe_takes_the_layer_inputs(
        self, use_classical_encoder: bool
    ) -> None:
        """
        ``AnalogRydbergLayer.step_count`` names a model's layer inputs as
        ``model.classical_encoder(X) * π``, or ``X`` without the encoder
        (#769): what ``quantum_layer`` receives in ``forward``.  Its count
        then builds the model again from the config and the state dict, whose
        layer gives the same count (it holds the same weights) and whose
        outputs ``⟨Z_i⟩`` are within ``2 · tol`` of a run at 16 times the count.
        At ``γ = Ω_max`` the doubling sets the count (``4 n₀``; at ``γ = 1.5``
        ``2 n₀`` met even ``tol = 1e-4``), so half of it misses that bound:
        1.07 and 1.19 ``tol`` at the count, 4.2 and 4.6 ``tol`` at half.
        """
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            n_input_features=2,
            n_qubits=2,
            n_layers=2,
            encoding_type="analog",
            gamma=4 * math.pi,
            n_steps=3,
            use_classical_encoder=use_classical_encoder,
        ).eval()
        layer = model.quantum_layer
        assert isinstance(layer, AnalogRydbergLayer)
        x = torch.rand(3, 2) * 2 - 1
        seen: list[torch.Tensor] = []
        layer.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
        model(x)
        with torch.no_grad():
            inputs = model.classical_encoder(x) * math.pi if use_classical_encoder else x
        torch.testing.assert_close(inputs, seen[0], rtol=0, atol=0)
        tol = 1e-6
        n = layer.step_count(inputs, tol=tol)
        config = {**model.get_config(), "n_steps": n}
        rebuilt = HybridBinaryClassifier(**config)
        rebuilt.load_state_dict(model.state_dict())
        reference = HybridBinaryClassifier(**{**config, "n_steps": 16 * n})
        reference.load_state_dict(model.state_dict())
        rebuilt_layer, reference_layer = rebuilt.quantum_layer, reference.quantum_layer
        assert isinstance(rebuilt_layer, AnalogRydbergLayer)
        assert isinstance(reference_layer, AnalogRydbergLayer)
        assert rebuilt_layer.n_steps == n
        assert rebuilt_layer.step_count(inputs, tol=tol) == n
        with torch.no_grad():
            error = (rebuilt_layer(inputs).double() - reference_layer(inputs).double()).abs().max()
        assert 0 < error <= 2 * tol

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


class TestKernelsAndSummary:
    """The kernels and ``circuit_summary`` on the dephased layer (#711)."""

    GAMMA = 1.5

    def _dephased(self, n_qubits: int = 2, n_segments: int = 2, **kwargs: object):
        layer = _layer(n_qubits, n_segments, gamma=self.GAMMA, n_steps=2000, **kwargs)
        with torch.no_grad():
            layer.qlayer.weights.copy_(
                torch.tensor([[0.3, -0.4, 0.7], [-0.2, 0.5, -1.1]], dtype=F64)[:n_segments]
            )
        return layer

    @staticmethod
    def _inputs(n: int, n_qubits: int = 2) -> torch.Tensor:
        return torch.randn(n, n_qubits, generator=torch.Generator().manual_seed(3), dtype=F64)

    def test_kernel_is_tr_rho_rho_of_the_liouvillian_states(self) -> None:
        from hqnn_forge.kernels import quantum_kernel_matrix

        layer = self._dephased()
        x = self._inputs(3)
        rho = np.stack([_reference_density_matrix(layer, row.numpy(), self.GAMMA) for row in x])
        expected = np.einsum("aij,bji->ab", rho, rho).real
        kernel = quantum_kernel_matrix(x, layer)
        # n_steps = 2000: the splitting error of evolve is far below 1e-6 here.
        np.testing.assert_allclose(kernel.numpy(), expected, atol=1e-6)
        # The diagonal is the purity, below 1 under dephasing.
        purity = np.einsum("aij,aji->a", rho, rho).real
        np.testing.assert_allclose(kernel.diagonal().numpy(), purity, atol=1e-6)
        assert bool((kernel.diagonal() < 1 - 1e-3).all())

    def test_kernel_against_y_and_batches_matches_density_matrices(self) -> None:
        from hqnn_forge.kernels import encoded_density_matrices, quantum_kernel_matrix

        layer = self._dephased()
        x, y = self._inputs(3), self._inputs(5)[3:]
        rho_x = layer.density_matrices(x).detach()
        rho_y = layer.density_matrices(y).detach()
        expected = torch.einsum("aij,bji->ab", rho_x, rho_y).real
        torch.testing.assert_close(quantum_kernel_matrix(x, layer, Y=y), expected)
        torch.testing.assert_close(quantum_kernel_matrix(x, layer, Y=y, batch_size=2), expected)
        torch.testing.assert_close(encoded_density_matrices(x, layer, batch_size=2), rho_x)

    def test_gamma_to_zero_gives_the_fidelity_kernel_of_the_unitary_layer(self) -> None:
        from hqnn_forge.kernels import quantum_kernel_matrix

        unitary = _layer(2, 2)
        dephased = _layer(2, 2, gamma=1e-10, n_steps=4)
        with torch.no_grad():
            dephased.qlayer.weights.copy_(unitary.qlayer.weights)
        x = self._inputs(4)
        torch.testing.assert_close(
            quantum_kernel_matrix(x, dephased),
            quantum_kernel_matrix(x, unitary),
            atol=1e-8,
            rtol=0.0,
        )

    def test_depolarising_at_end_acts_once_on_every_atom(self) -> None:
        from hqnn_forge.kernels import encoded_density_matrices, quantum_kernel_matrix

        p = 0.1
        layer = self._dephased()
        x = self._inputs(2)
        rho = layer.density_matrices(x).detach().numpy()
        for i in range(2):  # DepolarizingChannel(p) on atom i, by its Kraus operators
            kraus = [math.sqrt(1 - p) * EYE] + [math.sqrt(p / 3) * P for P in (X, Y, Z)]
            rho = sum(_on(k, i, 2) @ rho @ _on(k, i, 2).conj().T for k in kraus)
        got = encoded_density_matrices(x, layer, noise_level=p, noise_position="end")
        np.testing.assert_allclose(got.numpy(), rho, atol=1e-12)
        kernel = quantum_kernel_matrix(x, layer, noise_level=p, noise_position="end")
        np.testing.assert_allclose(
            kernel.numpy(), np.einsum("aij,bji->ab", rho, rho).real, atol=1e-12
        )

    def test_differentiable_kernel_carries_the_gradient_of_tr_rho_rho(self) -> None:
        from hqnn_forge.kernels import quantum_kernel_matrix

        layer = self._dephased()
        x = self._inputs(2)
        quantum_kernel_matrix(x, layer, differentiable=True)[0, 1].backward()
        got = layer.qlayer.weights.grad.clone()
        layer.qlayer.weights.grad = None
        rho = layer.density_matrices(x)
        torch.einsum("ij,ji->", rho[0], rho[1]).real.backward()
        torch.testing.assert_close(got, layer.qlayer.weights.grad)
        assert bool((got != 0).any())

    def test_noise_at_all_acts_after_every_segment(self) -> None:
        """
        ``noise_position="all"``: the depolarising channel on every atom after
        every segment (#717), as in the layer's own noise; the kernel is
        ``Tr[ρ ρ]`` of those states.
        """
        from hqnn_forge.kernels import encoded_density_matrices, quantum_kernel_matrix

        p = 0.1
        layer = self._dephased()
        x = self._inputs(2)
        rho = np.stack(
            [
                _reference_with_channel(layer, row.numpy(), self.GAMMA, "depolarizing", p)
                for row in x
            ]
        )
        got = encoded_density_matrices(x, layer, noise_level=p, noise_position="all")
        np.testing.assert_allclose(got.numpy(), rho, atol=2e-6)
        kernel = quantum_kernel_matrix(x, layer, noise_level=p, noise_position="all")
        np.testing.assert_allclose(
            kernel.numpy(), np.einsum("aij,bji->ab", rho, rho).real, atol=4e-6
        )

    def test_pure_state_functions_refuse_the_layer(self) -> None:
        from hqnn_forge.kernels import encoded_states, overlap_kernel_matrix

        layer = self._dephased()
        x = self._inputs(2)
        for function in (encoded_states, overlap_kernel_matrix):
            with pytest.raises(ValueError, match="gamma > 0 prepares a mixed state"):
                function(x, layer)

    def test_circuit_summary_counts_the_preparation_and_the_pulse_values(self) -> None:
        from hqnn_forge.diagnostics import circuit_summary

        summary = circuit_summary(_layer(3, 4, gamma=self.GAMMA, n_steps=4))
        assert summary.gate_counts == {"QubitDensityMatrix": 1}
        assert (summary.n_gates, summary.depth, summary.n_two_qubit_gates) == (1, 1, 1)
        # Every pulse value shapes ρ(T), so none is inert.
        assert (summary.n_trainable_params, summary.n_inert_params) == (12, 0)
        assert summary.device_name == "default.mixed"


# ---------------------------------------------------------------------------
# One step count for the chain of segments (#769)
# ---------------------------------------------------------------------------

#: A layer case of #769: ``N`` atoms in a chain with ``V/Ω_max`` between
#: neighbours, ``γ/Ω_max``, and the pulse per segment as ``Ω_k/Ω_max``,
#: ``Δ_k/Ω_max`` and ``φ_k`` (radians), at the default ``Ω_max = 4π`` rad/µs,
#: ``τ = π/Ω_max`` and detuning bounds ``√3 Ω_max``.
LayerCase = tuple[int, float, float, list[float], list[float], list[float]]


class TestTheChainOfSegments:
    """
    The two-run estimate of :meth:`~hqnn_forge.rydberg.RydbergFeatureMap.step_count`
    (#760) on the layer's own chain (#769): ``K`` segments of equal τ, each
    with its own ``Ω_k``, ``Δ_k`` and ``φ_k``, every one evolved by ``evolve``
    with the same ``n_steps`` and chained through ``rho0``.

    #760 derives it for any chain whose segments share one count: segment
    ``k`` gives ``E_k + D_k/n² + O(n⁻⁴)``, so the chain is the exact chain
    plus ``1/n²`` times an ``n``-independent map, every linear feature is
    ``F_n = F + A/n² + O(n⁻⁴)``, and ``F_n − F_2n = 3 (F_2n − F)``.  A
    gate-model channel after every segment (``noise_position="all"``) is a
    fixed linear map between the segments and leaves this unchanged.  With
    ``n₀ = max_k ceil(2τ (W_k + Nγ/2))`` over the segments and samples
    (``W_k = E_max − E_min`` of the hand-built ``H_k``, τ in µs, rates in
    rad/µs), the features at ``2n₀`` steps per segment are off the chained
    exponential of the Liouvillian per segment (:func:`_reference_density_matrix`,
    :func:`_reference_with_channel`) by a third of their largest difference
    from ``n₀`` steps, within 10 %, as for one segment
    (``TestChoosingTheStepCount``) and the feature map's chain
    (``TestFeatureMapStepCount``).  This is measured at fixed weights, so it
    holds whatever count the layer ends up with; 126 cases (1 to 3 atoms,
    ``K`` 1, 2, 4, ``V/Ω_max`` 0 to 10, ``γ/Ω_max`` 0.1 to 10, the
    constructor's pulse and random weights) gave 0.993 to 1.007.

    :meth:`~hqnn_forge.encoding.AnalogRydbergLayer.step_count` doubles from
    that ``n₀`` at the weights the layer holds; its count is checked against
    the same exponential, and its target, ``⟨n_i⟩`` and ``⟨n_i n_j⟩`` of
    ``density_matrices``, against the alternatives it is not.
    """

    OMEGA_MAX = 4.0 * math.pi
    #: Inputs for up to 3 atoms, as the models feed them, in ``(−π, π)``.
    X_IN = torch.tensor([[-1.2, 0.4, 2.5], [0.3, -2.0, 0.0]], dtype=F64)

    CASES: list[LayerCase] = [
        # three segments, nothing symmetric, V ≈ Ω_max
        (2, 1.0, 1.0, [0.3, 0.9, 0.6], [0.5, -1.0, 1.5], [0.0, 1.3, -0.4]),
        # strong interactions, weak dephasing: W from V, the features need few steps
        (3, 10.0, 0.1, [0.8, 0.4], [-0.2, 1.2], [0.5, -0.7]),
        # one atom, strong dephasing, four segments
        (1, 0.0, 10.0, [0.2, 0.7, 0.95, 0.5], [1.0, -0.5, 0.0, 1.6], [0.0, 2.0, -1.0, 0.3]),
        # interactions and strong dephasing together
        (3, 1.0, 4.0, [0.5, 0.9, 0.3], [0.0, -1.5, 0.7], [0.2, -0.9, 1.1]),
    ]

    def build(self, case: LayerCase, n_steps: int, **kwargs: object) -> AnalogRydbergLayer:
        """The layer of ``case`` at ``n_steps`` per segment, with its pulse set."""
        n_atoms, v_over_omega, gamma_over_omega, omega, detuning, phase = case
        # V = C6/a⁶ between neighbours; with V = 0 the atoms sit 1 mm apart (V ≈ 5e-12)
        spacing = 1000.0 if v_over_omega == 0 else (6.0 / v_over_omega) ** (1 / 6)
        layer = _layer(
            n_atoms,
            len(omega),
            register=AtomRegister([[spacing * i, 0.0] for i in range(n_atoms)]),
            c6=6.0 * self.OMEGA_MAX,
            gamma=gamma_over_omega * self.OMEGA_MAX,
            n_steps=n_steps,
            **kwargs,
        )
        layer.set_pulse(
            omega=[w * self.OMEGA_MAX for w in omega],
            detuning=[d * self.OMEGA_MAX for d in detuning],
            phase=phase,
        )
        return layer

    def starting_count(self, layer: AnalogRydbergLayer, x: torch.Tensor) -> int:
        """``n₀ = max ceil(2τ (W_k + Nγ/2))`` over the hand-built ``H_k`` of every sample."""
        start = 1
        for row in x:
            for h in _reference_hamiltonians(layer, row.numpy()):
                energies = np.linalg.eigvalsh(h)
                rate = energies.max() - energies.min() + layer.n_qubits * layer.gamma / 2
                start = max(start, math.ceil(2 * layer.segment_duration * rate))
        return start

    @staticmethod
    def features(rho: torch.Tensor) -> torch.Tensor:
        """``⟨n_i⟩`` and ``⟨n_i n_j⟩`` of a batch of density matrices."""
        return rydberg_readout(rho, pairs=True)

    @pytest.mark.parametrize("case", CASES)
    def test_a_third_of_the_difference_is_the_error_of_the_chain(self, case: LayerCase) -> None:
        x = self.X_IN[:, : case[0]]
        start = self.starting_count(self.build(case, 1), x)
        coarse, fine = (
            self.features(self.build(case, n).density_matrices(x).detach())
            for n in (start, 2 * start)
        )
        reference = self.build(case, 1)
        exact = self.features(
            torch.stack(
                [
                    torch.as_tensor(
                        _reference_density_matrix(reference, row.numpy(), reference.gamma)
                    )
                    for row in x
                ]
            )
        )
        error = (fine - exact).abs().max().item()
        estimate = (coarse - fine).abs().max().item() / 3
        assert error > 1e-9  # above the rounding floor, so the ratio is not noise
        assert error / estimate == pytest.approx(1.0, rel=0.1)

    def test_a_channel_after_every_segment_keeps_the_estimate(self) -> None:
        """
        ``noise_position="all"``: the amplitude-damping map ``D^{⊗N}`` after
        every segment (#717), on the layer's outputs ``⟨Z_i⟩`` through
        ``apply_depolarizing_noise``, against ``e^(𝓛_k τ)`` then ``D^{⊗N}``
        per segment.
        """
        case = self.CASES[0]
        p = 0.1
        x = self.X_IN[:, : case[0]]
        start = self.starting_count(self.build(case, 1), x)

        def outputs(n_steps: int) -> torch.Tensor:
            layer = self.build(case, n_steps).eval()
            with apply_depolarizing_noise(layer, p, position="all", channel="amplitude_damping"):
                return layer(x).detach()

        coarse, fine = outputs(start), outputs(2 * start)
        reference = self.build(case, 1)
        exact = torch.tensor(
            [
                _z(
                    _reference_with_channel(
                        reference, row.numpy(), reference.gamma, "amplitude_damping", p
                    )
                )
                for row in x
            ],
            dtype=F64,
        )
        error = (fine - exact).abs().max().item()
        estimate = (coarse - fine).abs().max().item() / 3
        assert error > 1e-9
        assert error / estimate == pytest.approx(1.0, rel=0.1)
        # the channel matters at this accuracy
        noiseless = torch.tensor(
            [_z(_reference_density_matrix(reference, row.numpy(), reference.gamma)) for row in x],
            dtype=F64,
        )
        assert (exact - noiseless).abs().max().item() > 1e-2

    # -- AnalogRydbergLayer.step_count ------------------------------------

    #: ``⟨n_i⟩`` alone would meet ``tol = 1e-6`` at half the count here, the
    #: pairs do not (estimates 7.3e-7 and 1.15e-6 at 224 steps).
    PAIRS_CASE: LayerCase = (
        2,
        1.0,
        1.0,
        [0.76, 0.84, 0.59],
        [1.3, -0.7, -1.3],
        [0.9, -0.2, -1.9],
    )

    def exact_features(self, layer: AnalogRydbergLayer, x: torch.Tensor) -> torch.Tensor:
        """The features of the chained exponential of the Liouvillian per segment."""
        return self.features(
            torch.stack(
                [
                    torch.as_tensor(_reference_density_matrix(layer, row.numpy(), layer.gamma))
                    for row in x
                ]
            )
        )

    def estimate(self, case: LayerCase, x: torch.Tensor, n: int, part: slice) -> float:
        """A third of the largest difference of the features ``part`` at ``n/2`` and ``n`` steps."""
        coarse, fine = (
            self.features(self.build(case, m).density_matrices(x).detach())[:, part]
            for m in (n // 2, n)
        )
        return (coarse - fine).abs().max().item() / 3

    @pytest.mark.parametrize("case", CASES)
    def test_the_count_meets_tol_against_the_chained_liouvillian(self, case: LayerCase) -> None:
        """
        At the returned count the features of ``density_matrices`` are within
        ``tol`` of the chained exponential, and the count is ``2^k n₀`` with
        ``n₀`` from the hand-built ``H_k`` of the pulse that was set, not the
        constructor's: one doubling sequence for the whole chain.
        """
        tol = 1e-6
        x = self.X_IN[:, : case[0]]
        layer = self.build(case, 1)
        n = layer.step_count(x, tol=tol)
        doublings = math.log2(n / self.starting_count(layer, x))
        assert doublings == int(doublings) >= 1
        fine = self.features(self.build(case, n).density_matrices(x).detach())
        assert (fine - self.exact_features(layer, x)).abs().max().item() <= tol

    @pytest.mark.parametrize("case", [CASES[0], PAIRS_CASE])
    def test_it_is_the_smallest_count_tried(self, case: LayerCase) -> None:
        """
        The estimate at the returned count, from two runs of the chain at
        ``n/2`` and ``n``, meets ``tol``, and that at ``n/2`` does not.  In
        :attr:`PAIRS_CASE` the pairs set the count: ``⟨n_i⟩`` alone would
        meet ``tol`` at ``n/2``.
        """
        tol = 1e-6
        x = self.X_IN[:, : case[0]]
        layer = self.build(case, 1)
        n = layer.step_count(x, tol=tol)
        assert n > 2 * self.starting_count(layer, x)  # a count before it was tried
        every = slice(None)
        assert self.estimate(case, x, n, every) <= tol < self.estimate(case, x, n // 2, every)
        if case is self.PAIRS_CASE:
            assert self.estimate(case, x, n // 2, slice(0, case[0])) <= tol

    def test_the_target_is_the_excitations_not_the_outputs(self) -> None:
        """
        ``tol`` bounds ``⟨n_i⟩``, as for ``RydbergFeatureMap.step_count``, not
        the outputs ``⟨Z_i⟩ = 1 − 2⟨n_i⟩``: here the outputs' estimate at the
        count is above ``tol``, and their error lies between ``tol`` and
        ``2 tol``.
        """
        tol = 1e-6
        case = self.CASES[0]
        x = self.X_IN[:, : case[0]]
        layer = self.build(case, 1)
        n = layer.step_count(x, tol=tol)
        coarse, fine = (self.build(case, m).eval()(x).detach() for m in (n // 2, n))
        assert (coarse - fine).abs().max().item() / 3 > tol
        exact = torch.tensor(
            [_z(_reference_density_matrix(layer, row.numpy(), layer.gamma)) for row in x],
            dtype=F64,
        )
        assert tol < (fine - exact).abs().max().item() <= 2 * tol

    def test_settings_outside_the_state_do_not_change_it(self) -> None:
        """
        The layer's own ``n_steps``, ``readout``, ``readout_error`` and
        training noise do not enter: the target is the state of
        ``density_matrices``.
        """
        case = self.CASES[0]
        x = self.X_IN[:, : case[0]]
        n = self.build(case, 1).step_count(x, tol=1e-6)
        changes: list[dict[str, object]] = [
            {},
            {"readout": "first"},
            {"readout_error": (0.05, 0.1)},
            {"noise_level": 0.1, "noise_channel": "amplitude_damping"},
        ]
        for changed in changes:
            assert self.build(case, 977, **changed).step_count(x, tol=1e-6) == n

    def test_one_sample_is_a_batch_of_one(self) -> None:
        case = self.CASES[3]
        layer = self.build(case, 1)
        x = self.X_IN[1, : case[0]]
        assert layer.step_count(x, tol=1e-6) == layer.step_count(x[None], tol=1e-6)

    def test_without_dephasing_one_is_returned(self) -> None:
        layer = _layer(2, 3)
        assert layer.step_count(self.X_IN[:, :2], tol=1e-9) == 1

    def test_tol_is_required_and_keyword_only(self) -> None:
        layer = self.build(self.CASES[0], 1)
        x = self.X_IN[:, :2]
        with pytest.raises(TypeError, match="tol"):
            layer.step_count(x)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="positional"):
            layer.step_count(x, 1e-6)  # type: ignore[misc]

    @pytest.mark.parametrize("tol", [0.0, 1e-10, math.nan, "1e-6", True, None])
    def test_rejects_the_tol_step_count_rejects(self, tol: object) -> None:
        layer = self.build(self.CASES[0], 1)
        with pytest.raises(ValueError, match=r"tol must be a finite number >= 1e-09"):
            layer.step_count(self.X_IN[:, :2], tol=tol)  # type: ignore[arg-type]

    def test_rejects_inputs_forward_rejects(self) -> None:
        layer = self.build(self.CASES[0], 1)
        with pytest.raises(ValueError, match="does not match n_qubits=2"):
            layer.step_count(self.X_IN, tol=1e-6)
        with pytest.raises(ValueError, match="NaN"):
            layer.step_count(torch.full((1, 2), math.nan, dtype=F64), tol=1e-6)

    def test_rejects_an_empty_batch(self) -> None:
        layer = self.build(self.CASES[0], 1)
        with pytest.raises(ValueError, match="at least one sample"):
            layer.step_count(torch.zeros(0, 2, dtype=F64), tol=1e-6)
