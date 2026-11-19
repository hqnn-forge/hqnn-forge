"""
tests/test_rydberg_lindblad.py
==============================
``evolve`` and ``readout`` (#497) against the closed-form limits of
``docs/rydberg-model.md``: the detuned and the damped Rabi oscillation of one
atom, the product state without interactions, the blockade of two atoms, the
maximally mixed state under strong dephasing, and the order of the splitting;
the figures that the "Choosing ``n_steps``" note of ``evolve`` quotes, from
the points they come from and, opt-in, from the seeded sweep behind them
(#627); ``step_count`` (#598) by the feature error of the count it returns,
over a subset of #598's sweep, with its counts there pinned to the integer
(#759), and with ``target="state"`` by the trace distance of ρ over that
subset (the whole sweep opt-in, ``HQNN_FORGE_REPRODUCE=1``), and
``RydbergFeatureMap.step_count`` (#760) by that of a ramped drive and of a
jittered register, against the chained exponential of the Liouvillian per
segment.

Every expected value is obtained another way than the solver obtains it:

* a closed form typed in from the model page;
* ``scipy.linalg.expm`` applied to the state vector (``γ = 0``);
* :func:`liouvillian_reference`, the exponential of the full ``4^N × 4^N``
  Lindblad generator assembled from Kronecker products of the collapse
  operators ``√γ n_i``.  It uses neither an eigendecomposition, nor a
  splitting, nor the Hamming-distance form of the dissipator.

:func:`checked_evolve` is how the tests call the solver: it asserts trace 1,
Hermiticity and non-negative eigenvalues on every state it returns, so those
hold at every tested point.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import re
import time
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
import torch

from hqnn_forge.kernels import kernel_from_density_matrices
from hqnn_forge.rydberg import (
    DEFAULT_C6,
    MAX_ATOMS,
    AtomRegister,
    GlobalDrivePulseEncoding,
    PulseEncoding,
    RydbergFeatureMap,
    evolve,
    lindblad,
    readout,
    rydberg_hamiltonian,
    step_count,
)
from hqnn_forge.rydberg.lindblad import _populations_to_features

C128 = torch.complex128
F64 = torch.float64
ONE_ATOM = AtomRegister([[0.0, 0.0]])


def checked_evolve(
    hamiltonian: torch.Tensor,
    t: float,
    *,
    gamma: float = 0.0,
    n_steps: int | None = None,
    rho0: torch.Tensor | None = None,
) -> torch.Tensor:
    """``evolve``, with every returned state checked to be a density matrix."""
    rho = evolve(hamiltonian, t, gamma=gamma, n_steps=n_steps, rho0=rho0)
    assert rho.dtype == C128
    assert rho.shape == hamiltonian.shape
    trace = rho.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    # U = V e^(−iE dt) V† is unitary to rounding only, and the same U is applied
    # in every step, so the trace drifts by a few 1e-16 per step (measured:
    # −4e-16), not as a random walk.  ``evolve`` documents up to 1.4e-15 per step;
    # two atoms at V = γ = 100 Ω drift by 1.1e-15 per step over 10 096 steps.
    steps_taken = (n_steps or 0) if gamma > 0 else 0
    drift = 1e-12 + 2e-15 * steps_taken
    assert torch.allclose(trace, torch.ones_like(trace), rtol=0, atol=drift)
    assert torch.allclose(rho, rho.mH, rtol=0, atol=1e-12)
    # eigvalsh reads one triangle only; Hermiticity was asserted just above
    assert torch.linalg.eigvalsh(rho).min().item() >= -1e-12
    return rho


def number_operator(atom: int, n_atoms: int) -> torch.Tensor:
    """``n_i = 1 ⊗ … ⊗ |r⟩⟨r| ⊗ … ⊗ 1`` with the projector at factor ``atom`` from the left."""
    op = torch.ones(1, 1, dtype=C128)
    for j in range(n_atoms):
        factor = torch.tensor([[0.0, 0.0], [0.0, 1.0]]) if j == atom else torch.eye(2)
        op = torch.kron(op, factor.to(C128))
    return op


def liouvillian_reference(
    h: torch.Tensor, t: float, gamma: float, rho0: torch.Tensor | None = None
) -> torch.Tensor:
    """
    ``ρ(t) = exp(𝓛 t) ρ(0)`` from the full generator, for one Hamiltonian.

    With ρ flattened row by row, ``vec(A ρ B) = (A ⊗ Bᵀ) vec(ρ)``, so::

        𝓛 = −i (H ⊗ 1 − 1 ⊗ Hᵀ) + γ Σ_i ( n_i ⊗ n_i − ½ n_i ⊗ 1 − ½ 1 ⊗ n_i )

    which is ``−i[H, ρ] + Σ_i (L_i ρ L_i† − ½ {L_i† L_i, ρ})`` for
    ``L_i = √γ n_i`` (``n_i`` is real and diagonal, so ``n_iᵀ = n_i`` and
    ``n_i² = n_i``).  ``ρ(0)`` is ``rho0``, or all atoms in ``|g⟩`` without it.
    """
    dim = h.shape[-1]
    n_atoms = dim.bit_length() - 1
    eye = torch.eye(dim, dtype=C128)
    transpose = h.T.contiguous()
    generator = -1j * (torch.kron(h, eye) - torch.kron(eye, transpose))
    for atom in range(n_atoms):
        n = number_operator(atom, n_atoms)
        generator = generator + gamma * (
            torch.kron(n, n) - 0.5 * torch.kron(n, eye) - 0.5 * torch.kron(eye, n)
        )
    if rho0 is None:
        start = torch.zeros(dim * dim, dtype=C128)
        start[0] = 1.0
    else:
        start = rho0.to(C128).reshape(dim * dim)
    return (torch.linalg.matrix_exp(generator * t) @ start).reshape(dim, dim)


def single_atom(omega: float, delta: float) -> torch.Tensor:
    """``[[0, Ω/2], [Ω/2, −Δ]]`` in the basis ``(|g⟩, |r⟩)``, typed in."""
    return torch.tensor([[0.0, omega / 2], [omega / 2, -delta]], dtype=C128)


def kron_all(factors: list[torch.Tensor]) -> torch.Tensor:
    """``factors[0] ⊗ factors[1] ⊗ …``: atom 0 is the leftmost factor."""
    out = torch.ones(1, 1, dtype=C128)
    for factor in factors:
        out = torch.kron(out, factor)
    return out


def damped_rabi(omega: float, gamma: float, t: float) -> float:
    """``⟨n⟩(t)`` of one resonant atom under dephasing: the three cases of the model page."""
    if gamma < 4 * omega:
        lam = math.sqrt(omega**2 - gamma**2 / 16)
        bracket = math.cos(lam * t) + gamma / (4 * lam) * math.sin(lam * t)
        return 0.5 - 0.5 * math.exp(-gamma * t / 4) * bracket
    if gamma == 4 * omega:
        return 0.5 - 0.5 * math.exp(-omega * t) * (1 + omega * t)
    kappa = math.sqrt(gamma**2 / 16 - omega**2)
    bracket = math.cosh(kappa * t) + gamma / (4 * kappa) * math.sinh(kappa * t)
    return 0.5 - 0.5 * math.exp(-gamma * t / 4) * bracket


class TestReadout:
    """Hand-built diagonals: which basis state counts towards which atom and pair."""

    #: populations of |ggg⟩, |ggr⟩, |grg⟩, |grr⟩, |rgg⟩, |rgr⟩, |rrg⟩, |rrr⟩ (sum 1)
    P3 = (0.02, 0.03, 0.05, 0.07, 0.11, 0.13, 0.17, 0.42)

    def test_two_atoms_by_hand(self) -> None:
        """Order ``|gg⟩, |gr⟩, |rg⟩, |rr⟩``: atom 0 is excited in the last two states."""
        rho = torch.diag(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=C128))
        assert readout(rho).tolist() == pytest.approx([0.3 + 0.4, 0.2 + 0.4], rel=1e-15)
        assert readout(rho, pairs=True).tolist() == pytest.approx([0.7, 0.6, 0.4], rel=1e-15)

    def test_three_atoms_by_hand_with_the_pairs_in_the_order_of_the_model_page(self) -> None:
        """``⟨n_0⟩, ⟨n_1⟩, ⟨n_2⟩``, then the pairs (0,1), (0,2), (1,2)."""
        rho = torch.diag(torch.tensor(self.P3, dtype=C128))
        n0 = 0.11 + 0.13 + 0.17 + 0.42  # |r..⟩
        n1 = 0.05 + 0.07 + 0.17 + 0.42  # |.r.⟩
        n2 = 0.03 + 0.07 + 0.13 + 0.42  # |..r⟩
        n01 = 0.17 + 0.42  # |rr.⟩
        n02 = 0.13 + 0.42  # |r.r⟩
        n12 = 0.07 + 0.42  # |.rr⟩
        assert readout(rho).tolist() == pytest.approx([n0, n1, n2], rel=1e-15)
        assert readout(rho, pairs=True).tolist() == pytest.approx(
            [n0, n1, n2, n01, n02, n12], rel=1e-15
        )

    def test_equals_the_traces_with_the_number_operators(self) -> None:
        """``Tr[ρ n_i]`` and ``Tr[ρ n_i n_j]`` of a random full density matrix of 4 atoms."""
        generator = torch.Generator().manual_seed(3)
        a = torch.randn(16, 16, generator=generator, dtype=C128)
        rho = a @ a.mH
        rho = rho / torch.trace(rho)
        n = [number_operator(i, 4) for i in range(4)]
        singles = [torch.trace(rho @ n[i]).real.item() for i in range(4)]
        pairs = [
            torch.trace(rho @ n[i] @ n[j]).real.item() for i in range(4) for j in range(i + 1, 4)
        ]
        assert readout(rho).tolist() == pytest.approx(singles, rel=1e-13)
        assert readout(rho, pairs=True).tolist() == pytest.approx(singles + pairs, rel=1e-13)

    def test_only_the_diagonal_is_read(self) -> None:
        diagonal = torch.diag(torch.tensor(self.P3, dtype=C128))
        generator = torch.Generator().manual_seed(5)
        noise = torch.randn(8, 8, generator=generator, dtype=C128)
        off_diagonal = noise - torch.diag(noise.diagonal())
        assert torch.equal(
            readout(diagonal + off_diagonal, pairs=True), readout(diagonal, pairs=True)
        )

    def test_a_batch_is_read_sample_by_sample(self) -> None:
        first = torch.diag(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=C128))
        second = torch.diag(torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=C128))
        features = readout(torch.stack([first, second]), pairs=True)
        assert features.dtype == F64
        assert features.shape == (2, 3)
        assert features.tolist() == [
            pytest.approx([0.7, 0.6, 0.4], rel=1e-15),
            pytest.approx([0.3, 0.4, 0.1], rel=1e-15),
        ]

    def test_one_atom_has_no_pair(self) -> None:
        rho = torch.diag(torch.tensor([0.25, 0.75], dtype=C128))
        assert readout(rho).tolist() == [0.75]
        assert readout(rho, pairs=True).tolist() == [0.75]

    def test_a_real_matrix_is_accepted(self) -> None:
        rho = torch.diag(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=F64))
        assert readout(rho).tolist() == pytest.approx([0.7, 0.6], rel=1e-15)

    @pytest.mark.parametrize(
        "shape",
        [(4,), (3, 3), (4, 2), (2, 2, 4, 4), (1, 1), (2 ** (MAX_ATOMS + 1), 2 ** (MAX_ATOMS + 1))],
    )
    def test_rejects_a_matrix_of_the_wrong_shape(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"rho must have shape \(2\^n, 2\^n\)"):
            readout(torch.zeros(shape, dtype=torch.complex64))

    def test_rejects_something_that_is_not_a_tensor(self) -> None:
        with pytest.raises(TypeError, match="rho must be a torch.Tensor"):
            readout([[1.0, 0.0], [0.0, 0.0]])  # type: ignore[arg-type]

    def test_rejects_a_non_finite_matrix(self) -> None:
        rho = torch.diag(torch.tensor([math.nan, 1.0], dtype=C128))
        with pytest.raises(ValueError, match="rho must be finite"):
            readout(rho)

    def test_pairs_is_keyword_only(self) -> None:
        with pytest.raises(TypeError, match="positional"):
            readout(torch.eye(2, dtype=C128) / 2, True)  # type: ignore[misc]


class TestPopulationsToFeatures:
    """``_populations_to_features``: features from diagonal populations or integer counts."""

    @pytest.mark.parametrize("n_atoms", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("pairs", [False, True])
    def test_equivalence_with_readout_for_single_and_batched_states(
        self, n_atoms: int, pairs: bool
    ) -> None:
        dim = 2**n_atoms
        generator = torch.Generator().manual_seed(n_atoms)
        # 1D state
        a = torch.randn(dim, dim, generator=generator, dtype=C128)
        rho_1d = (a @ a.mH) / torch.trace(a @ a.mH)
        diag_1d = rho_1d.diagonal(dim1=-2, dim2=-1).real
        expected_1d = readout(rho_1d, pairs=pairs)
        actual_1d = _populations_to_features(diag_1d, pairs=pairs)
        assert torch.equal(actual_1d, expected_1d)

        # Batched 2D state
        b = torch.randn(3, dim, dim, generator=generator, dtype=C128)
        rho_2d = b @ b.mH
        diag_2d = rho_2d.diagonal(dim1=-2, dim2=-1).real
        expected_2d = readout(rho_2d, pairs=pairs)
        actual_2d = _populations_to_features(diag_2d, pairs=pairs)
        assert torch.equal(actual_2d, expected_2d)

    def test_exact_integer_counts_preserved(self) -> None:
        """Integer counts return exact integer feature counts with the same dtype."""
        counts = torch.zeros(8, dtype=torch.int64)
        counts[7] = 10  # state |rrr> (all atoms excited)
        counts[3] = 5  # state |grr> (atoms 1 and 2 excited)
        features = _populations_to_features(counts, pairs=True)
        assert features.dtype == torch.int64
        # singles: atom 0 -> 10, atom 1 -> 15, atom 2 -> 15
        # pairs: (0,1) -> 10, (0,2) -> 10, (1,2) -> 15
        assert features.tolist() == [10, 15, 15, 10, 10, 15]

    def test_one_atom_has_no_pair(self) -> None:
        counts = torch.tensor([4, 6], dtype=torch.int64)
        assert _populations_to_features(counts).tolist() == [6]
        assert _populations_to_features(counts, pairs=True).tolist() == [6]


class TestOneAtomWithoutDephasing:
    """``⟨n⟩(t) = Ω²/Ω_eff² · sin²(Ω_eff t / 2)`` with ``Ω_eff = √(Ω² + Δ²)``."""

    @pytest.mark.parametrize(
        ("omega", "delta"),
        [(2.0, 0.0), (2.0, 1.5), (2.0, -1.5), (4 * math.pi, 7.5), (0.8, 3.0)],
    )
    @pytest.mark.parametrize("omega_t", [0.3, 1.0, math.pi, 4.0, 11.0])
    def test_detuned_rabi_formula(self, omega: float, delta: float, omega_t: float) -> None:
        t = omega_t / omega
        rho = checked_evolve(rydberg_hamiltonian(ONE_ATOM, omega, [delta], c6=DEFAULT_C6), t)
        effective = math.sqrt(omega**2 + delta**2)
        expected = omega**2 / effective**2 * math.sin(effective * t / 2) ** 2
        assert readout(rho).item() == pytest.approx(expected, rel=1e-12, abs=1e-14)

    def test_omega_is_the_rabi_frequency_a_pi_pulse_excites_the_atom(self) -> None:
        """``ΩT = π`` on resonance: ``⟨n⟩ = sin²(π/2) = 1``.  With ``H = Ω X`` it would be 0."""
        omega = 4 * math.pi
        rho = checked_evolve(
            rydberg_hamiltonian(ONE_ATOM, omega, [0.0], c6=DEFAULT_C6), math.pi / omega
        )
        assert readout(rho).item() == pytest.approx(1.0, rel=1e-13)

    def test_the_pulse_of_the_model_page_has_its_first_zero_at_delta_max(self) -> None:
        """``ΩT = π``, ``Δ = √3 Ω``: ``s = 2`` and ``f(s) = sin²(π)/4 = 0``; 0.437 at half range."""
        omega = 4 * math.pi
        delta = torch.tensor([[math.sqrt(3) * omega], [math.sqrt(3) * omega / 2]], dtype=F64)
        rho = checked_evolve(
            rydberg_hamiltonian(ONE_ATOM, omega, delta, c6=DEFAULT_C6), math.pi / omega
        )
        features = readout(rho)[:, 0].tolist()
        assert features[0] == pytest.approx(0.0, abs=1e-13)
        s = math.sqrt(1 + 3 / 4)
        assert features[1] == pytest.approx(math.sin(math.pi * s / 2) ** 2 / s**2, rel=1e-12)
        assert features[1] == pytest.approx(0.437, abs=5e-4)  # the value quoted on the page

    @pytest.mark.parametrize("omega_t", [0.4, 2.0, 5.0])
    def test_time_runs_forward_the_coherence_is_plus_i_sin_omega_t_over_2(
        self, omega_t: float
    ) -> None:
        """
        On resonance ``ψ(t) = (cos(Ωt/2), −i sin(Ωt/2))``, so
        ``ρ_gr = ψ_g ψ_r* = +(i/2) sin(Ωt)``.  The populations cannot see the
        sign: ``exp(+iHt)`` gives the complex conjugate state and the same
        ``⟨n⟩``.  It is the sign that makes ``dp/dt = Ω · Im ρ_gr`` positive
        at the start of the pulse.
        """
        omega = 2.0
        rho = checked_evolve(
            rydberg_hamiltonian(ONE_ATOM, omega, [0.0], c6=DEFAULT_C6), omega_t / omega
        )
        assert rho[0, 1].real.item() == pytest.approx(0.0, abs=1e-14)
        assert rho[0, 1].imag.item() == pytest.approx(0.5 * math.sin(omega_t), rel=1e-12)

    def test_at_time_zero_every_atom_is_in_the_ground_state(self) -> None:
        h = rydberg_hamiltonian(AtomRegister.chain(3, 5.0), 2.0, [0.3, -0.2, 0.9], c6=1.0e4)
        expected = torch.zeros(8, 8, dtype=C128)
        expected[0, 0] = 1.0  # |ggg⟩⟨ggg|
        assert torch.allclose(checked_evolve(h, 0.0), expected, rtol=0, atol=1e-14)
        damped = checked_evolve(h, 0.0, gamma=0.7, n_steps=3)  # V V† = 1 to rounding only
        assert torch.allclose(damped, expected, rtol=0, atol=1e-14)


class TestDampedRabi:
    """One resonant atom: the damped oscillation of the model page, in all three cases."""

    OMEGA = 2.0
    #: Steps per unit of ``Ωt`` at ``γ ≤ 4Ω``, and of ``γt/4`` above it: the
    #: step has to resolve ``1/γ`` as well as ``1/Ω``.  The global error is of
    #: second order (``TestSecondOrder`` checks the order itself); with this
    #: step it was below ``7 × 10⁻⁷`` in every case below, against the
    #: tolerance of ``10⁻⁶``.
    STEPS = 400

    def n_of_t(self, gamma_over_omega: float, omega_t: float) -> float:
        h = rydberg_hamiltonian(ONE_ATOM, self.OMEGA, [0.0], c6=DEFAULT_C6)
        rho = checked_evolve(
            h,
            omega_t / self.OMEGA,
            gamma=gamma_over_omega * self.OMEGA,
            n_steps=math.ceil(self.STEPS * omega_t * max(1.0, gamma_over_omega / 4)),
        )
        return readout(rho).item()

    @pytest.mark.parametrize(
        "gamma_over_omega",
        [
            0.1,  # underdamped, many oscillations
            1.0,  # underdamped
            3.9,  # underdamped, λ small
            4.0,  # critical
            4.1,  # overdamped, κ small
            8.0,  # overdamped
        ],
    )
    @pytest.mark.parametrize("omega_t", [0.5, math.pi, 6.0])
    def test_closed_form_in_all_three_cases(self, gamma_over_omega: float, omega_t: float) -> None:
        expected = damped_rabi(self.OMEGA, gamma_over_omega * self.OMEGA, omega_t / self.OMEGA)
        assert self.n_of_t(gamma_over_omega, omega_t) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("gamma_over_omega", [0.1, 1.0, 3.9, 4.0, 4.1, 8.0, 100.0])
    @pytest.mark.parametrize("omega_t", [0.5, math.pi, 6.0, 10.0])
    def test_the_closed_forms_solve_the_lindblad_equation(
        self, gamma_over_omega: float, omega_t: float
    ) -> None:
        """
        Guards the expected values themselves: the three formulas as typed in
        ``damped_rabi`` against the exponential of the single-atom generator,
        which involves no step size.
        """
        gamma, t = gamma_over_omega * self.OMEGA, omega_t / self.OMEGA
        reference = liouvillian_reference(single_atom(self.OMEGA, 0.0), t, gamma)
        assert damped_rabi(self.OMEGA, gamma, t) == pytest.approx(
            reference[1, 1].real.item(), abs=1e-12
        )

    def test_a_weakly_damped_pi_pulse_loses_pi_gamma_over_8_omega(self) -> None:
        """
        ``γ ≪ Ω`` after a π pulse: ``⟨n⟩ ≈ 1 − πγ/(8Ω)``, the envelope
        ``e^(−γt/4)`` of a coherence decaying at ``γ/2``.  A dissipator twice
        or half as strong (a coherence decaying at ``γ``, as with
        ``√(γ/2) Z_i``, or at ``γ/4``) gives a slope of ``π/4`` or ``π/16``
        instead of ``π/8``; ``√γ Z_i`` is four times as strong.
        """
        g = 0.01
        deficit = 1 - self.n_of_t(g, math.pi)
        # the next order is (πg/8)²/2-sized: 0.4 % of the leading term at g = 0.01
        assert deficit == pytest.approx(math.pi * g / 8, rel=1e-2)

    def test_gamma_t_much_larger_than_1_does_not_mean_mixed(self) -> None:
        """
        ``γ = 100 Ω``, ``Ωt = 10``: ``γt = 1000`` and ``⟨n⟩ = 0.0905``, not ½.
        40 000 steps (``γ dt = 0.025``): the error was ``4.3 × 10⁻⁶`` at half
        as many, so about ``1.1 × 10⁻⁶`` here.
        """
        h = rydberg_hamiltonian(ONE_ATOM, self.OMEGA, [0.0], c6=DEFAULT_C6)
        rho = checked_evolve(h, 10 / self.OMEGA, gamma=100 * self.OMEGA, n_steps=40000)
        expected = damped_rabi(self.OMEGA, 100 * self.OMEGA, 10 / self.OMEGA)
        assert expected == pytest.approx(0.0905, abs=5e-5)  # the value quoted on the page
        assert readout(rho).item() == pytest.approx(expected, abs=5e-6)


class TestInteractionsOff:
    """Without the pair term ρ is the tensor product of single-atom states."""

    OMEGA = 2.0
    DELTA = (0.4, 1.7, 3.1)
    T = 1.3

    def hamiltonian(self) -> torch.Tensor:
        return rydberg_hamiltonian(
            AtomRegister.chain(3, 5.0), self.OMEGA, list(self.DELTA), c6=1.0e6, interactions=False
        )

    def test_without_dephasing_rho_is_the_product_of_single_atom_states(self) -> None:
        rho = checked_evolve(self.hamiltonian(), self.T)
        factors = [
            liouvillian_reference(single_atom(self.OMEGA, d), self.T, 0.0) for d in self.DELTA
        ]
        assert torch.allclose(rho, kron_all(factors), rtol=0, atol=1e-12)

    def test_with_dephasing_rho_is_the_product_of_single_atom_states(self) -> None:
        """Each factor solves the single-atom Lindblad equation for its own ``Δ_i``."""
        gamma = 0.7 * self.OMEGA
        rho = checked_evolve(self.hamiltonian(), self.T, gamma=gamma, n_steps=2000)
        factors = [
            liouvillian_reference(single_atom(self.OMEGA, d), self.T, gamma) for d in self.DELTA
        ]
        assert torch.allclose(rho, kron_all(factors), rtol=0, atol=1e-6)

    def test_the_splitting_itself_factorises_at_any_step_size(self) -> None:
        """
        Both halves of a step are products over the atoms (``U`` of a
        Kronecker sum, and ``exp(−γτ d_H/2) = Π_i exp(−γτ |a_i − b_i|/2)``),
        so three atoms in 7 coarse steps equal the product of three
        single-atom runs of 7 steps to rounding, not just to ``O(dt²)``.
        """
        gamma = 0.7 * self.OMEGA
        rho = checked_evolve(self.hamiltonian(), self.T, gamma=gamma, n_steps=7)
        factors = [
            checked_evolve(single_atom(self.OMEGA, d), self.T, gamma=gamma, n_steps=7)
            for d in self.DELTA
        ]
        assert torch.allclose(rho, kron_all(factors), rtol=0, atol=1e-14)

    def test_each_feature_is_the_rabi_formula_of_its_own_detuning(self) -> None:
        """Three different detunings: a permuted atom order in the readout would show."""
        features = readout(checked_evolve(self.hamiltonian(), self.T)).tolist()
        for feature, delta in zip(features, self.DELTA, strict=True):
            effective = math.sqrt(self.OMEGA**2 + delta**2)
            expected = self.OMEGA**2 / effective**2 * math.sin(effective * self.T / 2) ** 2
            assert feature == pytest.approx(expected, rel=1e-12)

    def test_pair_correlations_of_a_product_state_factorise(self) -> None:
        gamma = 0.7 * self.OMEGA
        rho = checked_evolve(self.hamiltonian(), self.T, gamma=gamma, n_steps=50)
        n0, n1, n2, n01, n02, n12 = readout(rho, pairs=True).tolist()
        assert [n01, n02, n12] == pytest.approx([n0 * n1, n0 * n2, n1 * n2], rel=1e-12)

    def test_with_interactions_the_state_is_not_that_product(self) -> None:
        """The control is not vacuous: at ``V = Ω`` the same pulse gives another state."""
        h = rydberg_hamiltonian(
            AtomRegister.chain(3, 1.0), self.OMEGA, list(self.DELTA), c6=self.OMEGA
        )
        rho = checked_evolve(h, self.T)
        assert (rho - checked_evolve(self.hamiltonian(), self.T)).abs().max().item() > 0.05


class TestBlockade:
    """Two resonant atoms with ``V ≫ Ω`` share one excitation, driven at ``√2 Ω``."""

    OMEGA = 2.0

    def states(self, v_over_omega: float, omega_t: torch.Tensor) -> torch.Tensor:
        """
        ``ρ`` at every ``Ωt`` of the grid in one call.  Without dephasing
        ``ρ(t)`` depends on ``H t`` only, so the batch holds ``H · t_k`` and is
        evolved for one unit of time.
        """
        # spacing 1 µm, so V = C6
        h = rydberg_hamiltonian(
            AtomRegister.chain(2, 1.0), self.OMEGA, [0.0, 0.0], c6=v_over_omega * self.OMEGA
        )
        return checked_evolve(h * (omega_t / self.OMEGA)[:, None, None], 1.0)

    @pytest.mark.parametrize(
        ("v_over_omega", "ratio"), [(10.0, 0.57), (30.0, 0.52), (100.0, 0.51)]
    )
    def test_doubly_excited_population_stays_below_omega_over_v_squared(
        self, v_over_omega: float, ratio: float
    ) -> None:
        """
        Over ``Ωt ≤ 20`` the maximum of ``ρ_rr,rr`` is the fraction of
        ``(Ω/V)²`` the model page states.  The population oscillates at about
        ``V``, so the grid resolves ``V t`` in steps of 0.02 rad.
        """
        omega_t = torch.linspace(0.0, 20.0, int(1000 * v_over_omega) + 1, dtype=F64)
        doubly_excited = self.states(v_over_omega, omega_t)[:, 3, 3].real
        bound = 1 / v_over_omega**2
        assert doubly_excited.max().item() < bound
        assert doubly_excited.max().item() / bound == pytest.approx(ratio, abs=5e-3)

    @pytest.mark.parametrize("v_over_omega", [30.0, 100.0, 1000.0])
    def test_total_excitation_oscillates_at_sqrt_2_omega(self, v_over_omega: float) -> None:
        """
        ``⟨n_0 + n_1⟩ → sin²(√2 Ωt / 2)`` as ``Ω/V → 0``.  The doubly excited
        state shifts the symmetric state by ``−Ω²/(2V)`` in second order, so
        the deviation is of order ``(Ω/V)²`` in amplitude plus a phase that
        grows as ``(Ω/V)² Ωt``: below ``(1 + Ωt) · (Ω/V)²`` over the two
        periods tested.
        """
        omega_t = torch.linspace(0.0, 2 * math.sqrt(2) * math.pi, 201, dtype=F64)
        total = readout(self.states(v_over_omega, omega_t)).sum(dim=-1)
        expected = torch.sin(math.sqrt(2) * omega_t / 2) ** 2
        tolerance = (1 + omega_t) / v_over_omega**2
        assert bool(((total - expected).abs() <= tolerance).all())

    def test_it_is_not_the_oscillation_of_two_independent_atoms(self) -> None:
        """
        At ``Ωt = π/√2`` the blockaded pair holds one excitation.  Two free
        atoms would hold ``2 sin²(π/(2√2)) = 1.61``, which the same call
        returns with the interaction switched off.
        """
        omega_t = math.pi / math.sqrt(2)
        total = readout(self.states(1000.0, torch.tensor([omega_t], dtype=F64))).sum().item()
        assert total == pytest.approx(1.0, abs=1e-5)
        free = rydberg_hamiltonian(
            AtomRegister.chain(2, 1.0), self.OMEGA, [0.0, 0.0], c6=1.0, interactions=False
        )
        free_total = readout(checked_evolve(free, omega_t / self.OMEGA)).sum().item()
        assert free_total == pytest.approx(2 * math.sin(omega_t / 2) ** 2, rel=1e-12)
        assert free_total == pytest.approx(1.61, abs=5e-3)

    def test_both_atoms_are_excited_equally(self) -> None:
        omega_t = torch.linspace(0.0, 5.0, 11, dtype=F64)
        features = readout(self.states(100.0, omega_t))
        assert torch.allclose(features[:, 0], features[:, 1], rtol=0, atol=1e-13)


class TestStrongDephasing:
    """``γ = 4Ω``, ``Ωt = 100``, the strong-dephasing row of the model page."""

    OMEGA = 2.0
    GAMMA = 4 * OMEGA
    T = 100 / OMEGA
    #: ``dt = 0.05/Ω``.  The maximally mixed state is a fixed point of every
    #: step at any ``dt`` (the damping factor is 1 on the diagonal and
    #: ``U 1 U† = 1``), so the step size only perturbs the rate of approach.
    N_STEPS = 2000

    def detunings(self, n_atoms: int) -> torch.Tensor:
        """The ends of ``[0, Δ_max]`` for every atom, a mixed row, and random rows."""
        delta_max = math.sqrt(3) * self.OMEGA
        generator = torch.Generator().manual_seed(n_atoms)
        random = torch.rand(3, n_atoms, generator=generator, dtype=F64) * delta_max
        ends = torch.tensor([[0.0] * n_atoms, [delta_max] * n_atoms], dtype=F64)
        mixed = torch.tensor([[delta_max * (i % 2) for i in range(n_atoms)]], dtype=F64)
        return torch.cat([ends, mixed, random])

    @pytest.mark.parametrize("v_over_omega", [0.0, 1.0])
    @pytest.mark.parametrize("n_atoms", [1, 2, 3, 4])
    def test_rho_is_maximally_mixed_whatever_the_input(
        self, n_atoms: int, v_over_omega: float
    ) -> None:
        """
        ``g ≥ 0.289 Ω`` over the detuning range at ``γ = 4Ω`` without
        interactions (model page), so ``e^(−gt) ≤ e^(−28.9) = 3 × 10⁻¹³``;
        the page reports every element within ``10⁻¹²`` at ``V/Ω`` of 0 and 1
        (largest distance here: ``6 × 10⁻¹³``, one atom).
        """
        h = rydberg_hamiltonian(
            AtomRegister.chain(n_atoms, 1.0),  # spacing 1 µm, so V = C6
            self.OMEGA,
            self.detunings(n_atoms),
            c6=v_over_omega * self.OMEGA,
            interactions=v_over_omega > 0,
        )
        rho = checked_evolve(h, self.T, gamma=self.GAMMA, n_steps=self.N_STEPS)
        mixed = torch.eye(2**n_atoms, dtype=C128) / 2**n_atoms
        assert (rho - mixed).abs().max().item() < 1e-12
        features = readout(rho, pairs=True)
        assert torch.allclose(
            features[:, :n_atoms], torch.full_like(features[:, :n_atoms], 0.5), rtol=0, atol=1e-10
        )
        assert torch.allclose(
            features[:, n_atoms:], torch.full_like(features[:, n_atoms:], 0.25), rtol=0, atol=1e-10
        )

    def test_early_in_the_same_evolution_the_state_is_not_mixed(self) -> None:
        """The limit is reached, not built in: at ``Ωt = 1`` the atom is at 0.264."""
        h = rydberg_hamiltonian(ONE_ATOM, self.OMEGA, [0.0], c6=DEFAULT_C6)
        rho = checked_evolve(h, 1 / self.OMEGA, gamma=self.GAMMA, n_steps=400)
        expected = 0.5 - 0.5 * math.exp(-1.0) * 2  # critical damping at Ωt = 1
        assert readout(rho).item() == pytest.approx(expected, abs=1e-6)


class TestAgainstScipyExpm:
    """``γ = 0``: ``ρ = ψψ†`` with ``ψ = expm(−iHt) |g…g⟩``."""

    @pytest.mark.parametrize("interactions", [True, False])
    @pytest.mark.parametrize("n_atoms", [1, 2, 3, 4, 5])
    def test_state_matches_expm_applied_to_the_state_vector(
        self, n_atoms: int, interactions: bool
    ) -> None:
        """The reference pulse in real units: ``Ω = 4π rad/µs``, ``T = 0.25 µs``, ``V ≈ Ω``."""
        linalg = pytest.importorskip("scipy.linalg")
        omega = 4 * math.pi
        generator = torch.Generator().manual_seed(n_atoms)
        delta = torch.rand(3, n_atoms, generator=generator, dtype=F64) * math.sqrt(3) * omega
        register = AtomRegister.chain(n_atoms, 8.69)
        h = rydberg_hamiltonian(register, omega, delta, c6=DEFAULT_C6, interactions=interactions)
        t = 0.25
        rho = checked_evolve(h, t)
        for b in range(3):
            psi = linalg.expm(-1j * h[b].numpy() * t)[:, 0]  # column 0: applied to |g…g⟩
            expected = torch.as_tensor(psi[:, None] * psi.conj()[None, :], dtype=C128)
            assert torch.allclose(rho[b], expected, rtol=0, atol=1e-12)

    def test_a_long_evolution_deep_in_the_blockade(self) -> None:
        """``V/Ω = 100``, ``Ωt = 20``: 2000 rad of phase on the doubly excited state."""
        linalg = pytest.importorskip("scipy.linalg")
        h = rydberg_hamiltonian(AtomRegister.chain(2, 1.0), 2.0, [0.3, -0.2], c6=200.0)
        rho = checked_evolve(h, 10.0)
        psi = linalg.expm(-1j * h.numpy() * 10.0)[:, 0]
        expected = torch.as_tensor(psi[:, None] * psi.conj()[None, :], dtype=C128)
        assert torch.allclose(rho, expected, rtol=0, atol=1e-10)


class TestAgainstTheFullLiouvillian:
    """Interactions and dephasing together, where no closed form exists."""

    @pytest.mark.parametrize("gamma_over_omega", [0.0, 0.3, 2.0])
    @pytest.mark.parametrize("n_atoms", [2, 3])
    def test_state_matches_the_exponential_of_the_generator(
        self, n_atoms: int, gamma_over_omega: float
    ) -> None:
        """Different Δ on every atom and an irregular register: nothing is symmetric."""
        omega = 2.0
        positions = [[0.0, 0.0], [1.0, 0.1], [0.4, 1.3]][:n_atoms]
        delta = [0.5, -1.2, 2.1][:n_atoms]
        h = rydberg_hamiltonian(AtomRegister(positions), omega, delta, c6=1.5 * omega)
        t = 2.5 / omega
        gamma = gamma_over_omega * omega
        rho = checked_evolve(h, t, gamma=gamma, n_steps=4000 if gamma > 0 else None)
        expected = liouvillian_reference(h, t, gamma)
        assert torch.allclose(rho, expected, rtol=0, atol=1e-6 if gamma > 0 else 1e-12)


class TestWhatGammaMeans:
    """
    A weak resonant drive: to lowest order in ``Ωt`` the atom stays in ``|g⟩``
    (``p = 0``) and the single-atom equation of the model page,
    ``dc/dt = −(γ/2) c − i (Ω/2)(2p − 1)`` for ``c = ρ_gr``, becomes
    ``dc/dt = −(γ/2) c + iΩ/2``, so::

        c(t) = (iΩ/γ) · (1 − e^(−γt/2))

    The coherence saturates at the rate ``γ/2`` at which it decays.  Read as a
    rate γ per coherence (collapse operators ``√(γ/2) Z_i``), it would be
    ``(iΩ/2γ)(1 − e^(−γt))``: 0.52 times this value at ``γt = 6``.
    """

    OMEGA = 1.0e-3  # Ωt = 10⁻³: the neglected terms are of relative order (Ωt)² = 10⁻⁶
    GAMMA = 6.0
    T = 1.0
    N_STEPS = 2000  # γ dt = 0.003

    def coherence(self) -> complex:
        return 1j * self.OMEGA / self.GAMMA * (1 - math.exp(-self.GAMMA * self.T / 2))

    def test_single_atom_coherence_saturates_at_rate_gamma_over_2(self) -> None:
        h = rydberg_hamiltonian(ONE_ATOM, self.OMEGA, [0.0], c6=DEFAULT_C6)
        rho = checked_evolve(h, self.T, gamma=self.GAMMA, n_steps=self.N_STEPS)
        expected = self.coherence()
        assert rho[0, 1].real.item() == pytest.approx(0.0, abs=1e-12)
        assert rho[0, 1].imag.item() == pytest.approx(expected.imag, rel=1e-5)
        # the other convention is far outside that tolerance
        other = self.OMEGA / (2 * self.GAMMA) * (1 - math.exp(-self.GAMMA * self.T))
        assert other / expected.imag == pytest.approx(0.52, abs=5e-3)

    def test_two_atoms_each_element_sits_where_the_basis_ordering_puts_it(self) -> None:
        """
        Without interactions ``ρ = ρ_0 ⊗ ρ_1``.  In the order
        ``|gg⟩, |gr⟩, |rg⟩, |rr⟩``: ``ρ[0, 1] = ρ[0, 2] = c`` (one atom
        differs) and ``ρ[0, 3] = c²``, ``ρ[1, 2] = |c|²`` (both differ).
        """
        h = rydberg_hamiltonian(
            AtomRegister.chain(2, 1.0), self.OMEGA, [0.0, 0.0], c6=1.0, interactions=False
        )
        rho = checked_evolve(h, self.T, gamma=self.GAMMA, n_steps=self.N_STEPS)
        c = self.coherence()
        assert rho[0, 1].imag.item() == pytest.approx(c.imag, rel=1e-5)
        assert rho[0, 2].imag.item() == pytest.approx(c.imag, rel=1e-5)
        assert rho[0, 3].real.item() == pytest.approx((c * c).real, rel=1e-5)
        assert rho[1, 2].real.item() == pytest.approx(abs(c) ** 2, rel=1e-5)


class TestSecondOrder:
    """Halving ``dt`` divides the error by about 4."""

    OMEGA = 2.0
    GAMMA = 1.0 * OMEGA
    T = math.pi / OMEGA

    def hamiltonian(self) -> torch.Tensor:
        # V = Ω between neighbours, a different detuning on every atom
        return rydberg_hamiltonian(
            AtomRegister.chain(3, 1.0), self.OMEGA, [0.5, 1.9, 3.1], c6=self.OMEGA
        )

    def errors(self, reference: torch.Tensor, steps: list[int]) -> list[float]:
        h = self.hamiltonian()
        return [
            (checked_evolve(h, self.T, gamma=self.GAMMA, n_steps=n) - reference).abs().max().item()
            for n in steps
        ]

    def test_error_against_a_fine_run_falls_by_4_per_halving(self) -> None:
        """Reference: 64 times the finest tested step count, so its own error is 1/4096 of that."""
        steps = [20, 40, 80, 160]
        reference = checked_evolve(
            self.hamiltonian(), self.T, gamma=self.GAMMA, n_steps=64 * steps[-1]
        )
        errors = self.errors(reference, steps)
        for coarse, fine in itertools.pairwise(errors):
            assert coarse / fine == pytest.approx(4.0, rel=0.01)

    def test_error_against_the_exact_generator_falls_by_4_per_halving(self) -> None:
        """The same against a reference that shares nothing with the solver."""
        steps = [20, 40, 80, 160]
        reference = liouvillian_reference(self.hamiltonian(), self.T, self.GAMMA)
        errors = self.errors(reference, steps)
        assert errors[0] > 1e-5  # the coarse run is measurably off: the ratios are not noise
        for coarse, fine in itertools.pairwise(errors):
            assert coarse / fine == pytest.approx(4.0, rel=0.01)

    def test_without_a_drive_the_splitting_is_exact_in_one_step(self) -> None:
        """
        ``Ω = 0``: ``H`` is diagonal, so the commutator and the dissipator
        both act elementwise and commute.  From ``|g…g⟩`` nothing moves.
        """
        h = rydberg_hamiltonian(AtomRegister.chain(2, 1.0), 0.0, [0.7, -0.4], c6=3.0)
        rho = checked_evolve(h, 5.0, gamma=2.0, n_steps=1)
        expected = torch.zeros(4, 4, dtype=C128)
        expected[0, 0] = 1.0
        assert torch.allclose(rho, expected, rtol=0, atol=1e-15)

    def test_n_steps_is_not_used_without_dephasing(self) -> None:
        """``γ = 0`` is exact from one eigendecomposition, whatever ``n_steps`` says."""
        h = self.hamiltonian()
        assert torch.equal(
            checked_evolve(h, self.T, n_steps=1), checked_evolve(h, self.T, n_steps=None)
        )


class TestChoosingTheStepCount:
    """
    The "Choosing ``n_steps``" note of ``evolve``: when two runs with ``n``
    and ``2n`` steps tell how far off the finer one is, and when they do not.
    """

    OMEGA = 4.0 * math.pi  # the reference Ω of the model page

    def test_coarse_runs_can_agree_while_both_are_wrong(self) -> None:
        """
        One resonant atom, ``ΩT = 16π``: with ``n <= 16`` steps ``Ω dt`` is a
        whole multiple of π.  At ``n = 1, 2, 4, 8`` it is an even multiple and
        ``U(dt) = ±1`` leaves ``|g⟩`` in place; at ``n = 16`` it is π and
        ``U(dt)`` swaps ``|g⟩`` and ``|r⟩``.  Either way the atom is in a
        basis state between steps, and the dephasing finds no coherence to
        act on.
        """
        gamma = 0.1 * self.OMEGA
        t = 16 * math.pi / self.OMEGA
        h = single_atom(self.OMEGA, 0.0)
        exact = damped_rabi(self.OMEGA, gamma, t)
        assert exact == pytest.approx(0.3578, abs=1e-4)
        for n in (1, 2, 4, 8, 16):
            assert readout(checked_evolve(h, t, gamma=gamma, n_steps=n)).item() < 1e-12
        # the first count that resolves the drive is close
        assert readout(checked_evolve(h, t, gamma=gamma, n_steps=32)).item() == pytest.approx(
            exact, abs=1e-4
        )

    @pytest.mark.parametrize(
        ("n_atoms", "v_over_omega", "gamma_over_omega", "omega_t", "delta_over_omega"),
        [
            (1, 0.0, 10.0, 16 * math.pi, [0.0]),
            (1, 0.0, 100.0, 10.0, [0.0]),
            (2, 100.0, 1.0, math.pi, [0.0, math.sqrt(3)]),
            (4, 10.0, 10.0, 4 * math.pi, [0.0, 0.6, math.sqrt(3), 1.1]),
            (4, 1.0, 0.1, math.pi, [0.3, 1.5, 0.9, 0.0]),
        ],
    )
    def test_once_dt_resolves_the_generator_a_third_of_the_difference_is_the_error(
        self,
        n_atoms: int,
        v_over_omega: float,
        gamma_over_omega: float,
        omega_t: float,
        delta_over_omega: list[float],
    ) -> None:
        """
        ``n = ceil(2t · (W + Nγ/2))``, so ``dt · (W + Nγ/2) <= 1/2``, ``W`` the
        spread of the energies of ``H``: the largest error of the ``2n``-step
        features against the exponential of the full generator is a third of
        the largest ``|F_n − F_2n|`` within 10 % at these five points.  The
        first point is the one that needs the factor 2: with
        ``dt · (W + Nγ/2) <= 1`` its ratio is 1.37, while the other four stay
        near 1 either way.  The range over the whole parameter domain is
        checked by :class:`TestTheSeededSweep` (#627).
        """
        omega = self.OMEGA
        gamma = gamma_over_omega * omega
        t = omega_t / omega
        # V = c6 / 1^6 between neighbours at unit spacing
        h = rydberg_hamiltonian(
            AtomRegister.chain(n_atoms, 1.0),
            omega,
            [d * omega for d in delta_over_omega],
            c6=v_over_omega * omega,
        )
        energies = torch.linalg.eigvalsh(h)
        spread = (energies.max() - energies.min()).item()
        n = math.ceil(2 * t * (spread + n_atoms * gamma / 2))
        pairs = n_atoms > 1
        exact = readout(liouvillian_reference(h, t, gamma), pairs=pairs)
        coarse = readout(checked_evolve(h, t, gamma=gamma, n_steps=n), pairs=pairs)
        fine = readout(checked_evolve(h, t, gamma=gamma, n_steps=2 * n), pairs=pairs)
        error = (fine - exact).abs().max().item()
        estimate = (coarse - fine).abs().max().item() / 3
        # 10x above the rounding floor of about 1e-10, so the ratio is not noise;
        # the smallest error at these points is 1.7e-8
        assert error > 1e-9
        assert error / estimate == pytest.approx(1.0, rel=0.1)


#: The reference Ω of the model page, in rad/µs, at which the seeded sweep runs.
#: Every axis of a case is relative to Ω, so Ω only scales the units.
SWEEP_OMEGA = 4.0 * math.pi
#: The seeds of the sweep behind the "Choosing ``n_steps``" note of ``evolve``
#: (#599, #627), in the order the note's figures were first measured.
SWEEP_SEEDS = (599, 1, 2, 3, 4)
#: Cases drawn per seed.
SWEEP_CASES_PER_SEED = 300
#: The rounding floor of the note: below a largest feature error of about this,
#: the error stops falling by 4 per halving of ``dt``, and the share within
#: 10 % and the extremes of the largest-error ratio count only the cases above it.
ROUNDING_FLOOR = 1e-10
#: The per-feature ratios count only the features whose own error is above
#: this, ten times the floor, as #627's script did.
FEATURE_FLOOR = 1e-9


def seeded_sweep(seed: int) -> list[SweepCase]:
    """
    The 300 cases of one seed of the sweep, in the order #627's script drew them.

    Per case, from ``random.Random(seed)`` and in this order: the number of
    atoms, from {1, 2, 4}, in a chain at unit spacing (so ``V`` is the
    interaction of neighbours, ``c6 = V``); ``V/Ω``, 0 with probability 0.15
    and log-uniform on [0.1, 100] otherwise; ``γ/Ω``, log-uniform on
    [0.1, 100]; ``ΩT``, uniform on [π, 16π]; one detuning ``Δ_i/Ω`` per
    atom, uniform on [−2, 2].  The order of the draws is part of what the
    seed means: drawing the same axes in another order gives other cases.
    """
    rng = random.Random(seed)
    cases: list[SweepCase] = []
    for _ in range(SWEEP_CASES_PER_SEED):
        n_atoms = rng.choice([1, 2, 4])
        v_over_omega = 0.0 if rng.random() < 0.15 else 10 ** rng.uniform(-1, 2)
        gamma_over_omega = 10 ** rng.uniform(-1, 2)
        omega_t = rng.uniform(math.pi, 16 * math.pi)
        delta_over_omega = [rng.uniform(-2, 2) for _ in range(n_atoms)]
        cases.append((n_atoms, v_over_omega, gamma_over_omega, omega_t, delta_over_omega))
    return cases


def sweep_problem(case: SweepCase) -> tuple[torch.Tensor, float, float]:
    """``(H, t, γ)`` of a case at ``Ω = SWEEP_OMEGA``: ``H`` and γ in rad/µs, ``t`` in µs."""
    n_atoms, v_over_omega, gamma_over_omega, omega_t, delta_over_omega = case
    h = rydberg_hamiltonian(
        AtomRegister.chain(n_atoms, 1.0),
        SWEEP_OMEGA,
        [d * SWEEP_OMEGA for d in delta_over_omega],
        c6=v_over_omega * SWEEP_OMEGA,  # V = c6 / 1⁶ between neighbours
    )
    return h, omega_t / SWEEP_OMEGA, gamma_over_omega * SWEEP_OMEGA


def resolving_count(case: SweepCase, bound: float) -> int:
    """
    The smallest ``n`` with ``dt · (W + Nγ/2) <= bound`` at ``dt = t/n``.

    ``n = ceil(t (W + Nγ/2) / bound)``, ``W = E_max − E_min`` from
    ``eigvalsh`` of ``H`` and ``N`` the number of atoms; ``W`` and γ in
    rad/µs, ``t`` in µs, so the product is a number.  ``bound = 1/2`` is the
    note's condition, ``ceil(2t (W + Nγ/2))`` as in
    :class:`TestChoosingTheStepCount` (dividing by 1/2 doubles exactly in
    floating point, so the two give the same count).
    """
    h, t, gamma = sweep_problem(case)
    energies = torch.linalg.eigvalsh(h)
    spread = (energies.max() - energies.min()).item()
    return math.ceil(t * (spread + case[0] * gamma / 2) / bound)


def two_run_errors(case: SweepCase, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``(|F_2n − F|, |F_n − F_2n| / 3)`` per feature, at ``n = n_steps``.

    ``F`` are ``⟨n_i⟩`` and ``⟨n_i n_j⟩`` (``readout(..., pairs=True)``),
    exact from :func:`liouvillian_reference`, ``F_n`` from ``evolve`` with
    ``n`` steps.  The first is the finer run's error, the second the note's
    estimate of it: with ``F_n = F + a/n² + O(n⁻⁴)``,
    ``F_n − F_2n = (3/4) a/n² = 3 (F_2n − F)``.  Both are probabilities, with
    no unit; their ratio is 1 in the second-order regime.
    """
    h, t, gamma = sweep_problem(case)
    exact = exact_features(h, t, gamma)[0]
    coarse = features_of(h, t, gamma, n_steps)[0]
    fine = features_of(h, t, gamma, 2 * n_steps)[0]
    return (fine - exact).abs(), (coarse - fine).abs() / 3


class TestTheRecordedPoints:
    """
    The four single-atom points behind "an estimate, not a bound" in the
    "Choosing ``n_steps``" note of ``evolve``: the two extremes of the seeded
    sweep at ``dt · (W + Nγ/2) <= 1/2`` (seed 2, case 24 and seed 4, case 31,
    the largest-error ratios furthest above 1), and two points within the
    sweep's ranges that it did not draw (#627), the one furthest below 1 and
    one more above it.  The note quotes the lowest and the highest of these,
    0.83 and 1.64.  The ratio is the finer run's largest feature error over a
    third of the largest difference of the two runs; at the next halving it
    moves toward 1 at every point, as the second order says it must.
    """

    @pytest.mark.parametrize(
        ("case", "n_steps", "ratio", "next_ratio"),
        [
            pytest.param(seeded_sweep(2)[24], 65, 1.587, 1.095, id="seed 2, case 24"),
            pytest.param(seeded_sweep(4)[31], 128, 1.642, 1.098, id="seed 4, case 31"),
            pytest.param((1, 0.0, 2.339, 4.49, [-0.312]), 20, 0.830, 0.949, id="not drawn, low"),
            pytest.param((1, 0.0, 0.222, 11.03, [-1.034]), 35, 1.183, 1.038, id="not drawn, high"),
        ],
    )
    def test_the_ratio_and_the_next_halving(
        self, case: SweepCase, n_steps: int, ratio: float, next_ratio: float
    ) -> None:
        assert case[0] == 1
        assert resolving_count(case, 1 / 2) == n_steps
        error, estimate = two_run_errors(case, n_steps)
        # 100x above the rounding floor (the smallest of the four is 6.6e-8),
        # so the ratio is the splitting's, not rounding's
        assert error.max().item() > 1e-8
        measured = error.max().item() / estimate.max().item()
        assert measured == pytest.approx(ratio, abs=1e-3)
        next_error, next_estimate = two_run_errors(case, 2 * n_steps)
        next_measured = next_error.max().item() / next_estimate.max().item()
        assert next_measured == pytest.approx(next_ratio, abs=1e-3)
        assert abs(next_measured - 1) < abs(measured - 1)


class SweepRow(NamedTuple):
    """One case of the seeded sweep at one bound: where it came from and what it gave."""

    seed: int
    case: int  # its index among the seed's cases
    n_atoms: int
    n_steps: int  # of the coarser run
    error: torch.Tensor  # |F_2n − F| per feature
    estimate: torch.Tensor  # |F_n − F_2n| / 3 per feature

    @property
    def largest_error(self) -> float:
        return self.error.max().item()

    @property
    def ratio(self) -> float:
        """The finer run's largest error over a third of the largest difference."""
        return self.largest_error / self.estimate.max().item()


def sweep_rows(bound: float) -> list[SweepRow]:
    """Every case of every seed, the coarser run at ``dt · (W + Nγ/2) <= bound``."""
    rows = []
    for seed in SWEEP_SEEDS:
        for index, case in enumerate(seeded_sweep(seed)):
            n_steps = resolving_count(case, bound)
            error, estimate = two_run_errors(case, n_steps)
            rows.append(SweepRow(seed, index, case[0], n_steps, error, estimate))
    return rows


def above_the_floor(rows: list[SweepRow]) -> list[SweepRow]:
    return [row for row in rows if row.largest_error > ROUNDING_FLOOR]


def share_within_10_percent(rows: list[SweepRow]) -> float:
    return sum(abs(row.ratio - 1) <= 0.1 for row in rows) / len(rows)


def feature_ratios(rows: list[SweepRow]) -> list[float]:
    """Each feature's error over a third of its own difference, where above ``FEATURE_FLOOR``."""
    return [
        error / estimate
        for row in rows
        for error, estimate in zip(row.error.tolist(), row.estimate.tolist(), strict=True)
        if error > FEATURE_FLOOR
    ]


def sweep_summary(bound: float, rows: list[SweepRow]) -> str:
    """The figures of one bound, as the note quotes them; printed with ``-s``."""
    above = above_the_floor(rows)
    below = [row for row in rows if row.largest_error <= ROUNDING_FLOOR]
    low, high = min(above, key=lambda r: r.ratio), max(above, key=lambda r: r.ratio)
    per_feature = feature_ratios(rows)
    within = sum(abs(row.ratio - 1) <= 0.1 for row in above)
    largest_n = max(row.n_steps for row in rows)
    lines = [
        f"dt (W + N gamma/2) <= {bound:g}: {len(rows)} cases, coarse n up to {largest_n}",
        f"  error > {ROUNDING_FLOOR:g}: {len(above)} cases, {within} within 10 %",
        f"  largest-error ratio {low.ratio:.3f} to {high.ratio:.3f}",
        f"  lowest: seed {low.seed}, case {low.case}, N = {low.n_atoms}",
        f"  highest: seed {high.seed}, case {high.case}, N = {high.n_atoms}",
        f"  per feature (error > {FEATURE_FLOOR:g}): {min(per_feature):.3f} to "
        + f"{max(per_feature):.3f}",
        f"  error <= {ROUNDING_FLOOR:g}: {len(below)} cases, ratio "
        + f"{min(row.ratio for row in below):.3f} to {max(row.ratio for row in below):.3f}",
    ]
    for seed in SWEEP_SEEDS:
        ratios = [row.ratio for row in above if row.seed == seed]
        within = sum(abs(ratio - 1) <= 0.1 for ratio in ratios)
        lines.append(
            f"  seed {seed}: {within} of {len(ratios)} within 10 %, "
            + f"ratio {min(ratios):.3f} to {max(ratios):.3f}"
        )
    return "\n".join(lines)


@pytest.fixture(scope="module")
def seeded_rows() -> dict[float, list[SweepRow]]:
    """:func:`sweep_rows` at both bounds, run once for :class:`TestTheSeededSweep`; printed."""
    rows = {bound: sweep_rows(bound) for bound in (1 / 2, 1.0)}
    for bound, bound_rows in rows.items():
        print(sweep_summary(bound, bound_rows))
    return rows


@pytest.mark.reproducibility
@pytest.mark.slow
@pytest.mark.may_skip  # opt-in: skips in CI even under HQNN_FORGE_FAIL_ON_SKIP=1
@pytest.mark.skipif(
    os.environ.get("HQNN_FORGE_REPRODUCE") != "1",
    reason="opt-in: set HQNN_FORGE_REPRODUCE=1 (about 5 minutes; see TestTheSeededSweep)",
)
class TestTheSeededSweep:
    """
    The sweep behind the figures of the "Choosing ``n_steps``" note of
    ``evolve`` (#599, #627): five seeds of :func:`seeded_sweep`, 300 cases
    each, every case at ``dt · (W + Nγ/2) <= 1/2`` (the note's condition) and
    at ``<= 1``, against :func:`liouvillian_reference`.

    Opt-in, as the reproduction of the published SHNN is: the 3000 pairs of
    runs reach 60 000 steps and exponentiate the ``256 × 256`` generator of
    four atoms, 5.4 minutes on a 16-core laptop, so the class skips
    unless ``HQNN_FORGE_REPRODUCE=1`` and is marked ``reproducibility``,
    ``slow`` and ``may_skip``.  Run it with::

        HQNN_FORGE_REPRODUCE=1 uv run --frozen --all-extras \\
            pytest tests/test_rydberg_lindblad.py -k TestTheSeededSweep -s

    ``-s`` prints the figures of both bounds and per seed.  Each test checks
    one figure of the note to the precision the note quotes it, so a change
    to ``evolve``, to :func:`liouvillian_reference` or to the sampler that
    moves a figure fails here, and the note has to follow.  The one
    exception is the spread below the floor: rounding sets it, so another
    torch or BLAS can move it, and only its being wide is checked.  Measured on
    2026-11-19 (torch 2.15.0, CPU): at ``<= 1/2``, 1452 of the 1463 cases above
    the floor within 10 % (99.2 %), the largest-error ratio 0.906 to 1.642,
    per feature 0.837 to 5.025, 0.572 to 11.66 below the floor; at ``<= 1``,
    1329 of 1470 (90.4 %) and 0.715 to 3.481.
    """

    def test_every_seed_and_case_ran(self, seeded_rows: dict[float, list[SweepRow]]) -> None:
        for bound_rows in seeded_rows.values():
            assert [(row.seed, row.case) for row in bound_rows] == [
                (seed, index) for seed in SWEEP_SEEDS for index in range(SWEEP_CASES_PER_SEED)
            ]

    def test_at_one_half_99_percent_are_within_10_percent(
        self, seeded_rows: dict[float, list[SweepRow]]
    ) -> None:
        """
        The note: "within 10 % ... in 99 % of the cases whose error was above
        the rounding floor ... (1452 of 1463)".  The counts are checked too,
        since the note quotes them.
        """
        above = above_the_floor(seeded_rows[1 / 2])
        assert share_within_10_percent(above) >= 0.99
        assert (sum(abs(row.ratio - 1) <= 0.1 for row in above), len(above)) == (1452, 1463)

    def test_at_one_half_the_extremes_are_single_atoms_at_0_91_and_1_64(
        self, seeded_rows: dict[float, list[SweepRow]]
    ) -> None:
        """The note: "the largest error was 0.91 to 1.64 times it, both extremes single atoms"."""
        above = above_the_floor(seeded_rows[1 / 2])
        low, high = min(above, key=lambda r: r.ratio), max(above, key=lambda r: r.ratio)
        assert round(low.ratio, 2) == 0.91
        assert round(high.ratio, 2) == 1.64
        assert low.n_atoms == high.n_atoms == 1
        # the highest is the second point of TestTheRecordedPoints
        assert (high.seed, high.case) == (4, 31)

    def test_at_one_half_a_single_feature_is_0_84_to_5_0(
        self, seeded_rows: dict[float, list[SweepRow]]
    ) -> None:
        """The note: a single feature's error was 0.84 to 5.0 times a third of its difference."""
        ratios = feature_ratios(seeded_rows[1 / 2])
        assert round(min(ratios), 2) == 0.84
        assert round(max(ratios), 1) == 5.0

    def test_below_the_floor_the_estimate_says_nothing(
        self, seeded_rows: dict[float, list[SweepRow]]
    ) -> None:
        """
        The note: below about ``1e-10`` "the estimate says nothing".  The
        cases under the floor leave the range that the cases above it keep, on
        both sides and not in one outlier only, while the cases above it stay
        within the note's 0.91 to 1.64 (the test before), so the floor sits
        where the note puts it.  The note's 0.57 and 11.7 are not pinned:
        they are ratios of rounding errors near ``1e-13`` (measured: 14 of
        the 37 cases outside the range above the floor, 7 beyond twice its
        largest ratio), which another torch or BLAS can move.
        """
        below = [row.ratio for row in seeded_rows[1 / 2] if row.largest_error <= ROUNDING_FLOOR]
        above = [row.ratio for row in above_the_floor(seeded_rows[1 / 2])]
        assert min(below) < min(above)
        assert max(below) > 2 * max(above)
        outside = [ratio for ratio in below if not min(above) <= ratio <= max(above)]
        assert len(outside) >= len(below) / 5

    def test_at_one_it_is_clearly_worse(self, seeded_rows: dict[float, list[SweepRow]]) -> None:
        """
        The condition itself, not only the ratios under it: at
        ``dt · (W + Nγ/2) <= 1`` the share within 10 % falls to 90 % and the
        ratios spread to 0.71 and 3.5, the note's figures, clearly worse than
        at ``<= 1/2``.  Dropping the factor 2 from the note's condition is
        therefore caught over the whole sweep, not only at the one point of
        :class:`TestChoosingTheStepCount` that fails without it.
        """
        half, one = above_the_floor(seeded_rows[1 / 2]), above_the_floor(seeded_rows[1.0])
        assert round(share_within_10_percent(one), 2) == 0.90
        assert round(min(row.ratio for row in one), 2) == 0.71
        assert round(max(row.ratio for row in one), 1) == 3.5
        assert share_within_10_percent(one) < share_within_10_percent(half) - 0.05
        assert max(row.ratio for row in one) > 1.5 * max(row.ratio for row in half)


def features_of(h: torch.Tensor, t: float, gamma: float, n_steps: int) -> torch.Tensor:
    """``⟨n_i⟩`` and ``⟨n_i n_j⟩`` of ``evolve`` at ``n_steps``, one row per sample."""
    batch = h if h.ndim == 3 else h[None]
    return readout(checked_evolve(batch, t, gamma=gamma, n_steps=n_steps), pairs=True)


def exact_features(h: torch.Tensor, t: float, gamma: float) -> torch.Tensor:
    """The same features from :func:`liouvillian_reference`, sample by sample."""
    batch = h if h.ndim == 3 else h[None]
    return torch.stack([readout(liouvillian_reference(s, t, gamma), pairs=True) for s in batch])


def two_run_estimate(h: torch.Tensor, t: float, gamma: float, n_steps: int) -> float:
    """A third of the largest ``|F_(n/2) − F_n|`` over samples and features."""
    coarse = features_of(h, t, gamma, n_steps // 2)
    return (coarse - features_of(h, t, gamma, n_steps)).abs().max().item() / 3


#: One case of the sweep of #598: atoms in a unit chain, ``V/Ω`` between
#: neighbours, ``γ/Ω``, ``ΩT`` and the detunings ``Δ_i/Ω``, at the reference
#: ``Ω = 4π`` rad/µs.
SweepCase = tuple[int, float, float, float, list[float]]

#: 23 of the 216 cases of #598's sweep.  Every value of every axis appears
#: (1, 2 and 4 atoms; ``V/Ω`` 0, 1, 10, 100; ``γ/Ω`` 0.1, 1, 10, 100; ``ΩT``
#: π, 4π, 16π; detunings all 0 or spread over ``[0, Δ_max = √3]``), and so do
#: the cases that make the design: the false agreement (first row), the point
#: where the estimate is furthest from the error (one atom, ``γ = 10 Ω``,
#: ``ΩT = 16π``, 1.08), the issue's four-atom counter-example (``V = γ = 10 Ω``,
#: ``ΩT = 4π``), and points where the starting count alone leaves an error
#: below the rounding floor (``V/Ω = 100``).  Left out for run time: the 61
#: cases that take more than half a second at ``tol = 1e-6`` (all of
#: ``γ = 100 Ω`` beyond ``ΩT = π`` and the longest ``V/Ω = 10`` runs), up to
#: 13 s and 400 000 steps each; one of them is the slow test below.  The full
#: sweep, run by hand for #598, met ``1e-6`` in all 216 cases.
SWEEP: list[SweepCase] = [
    (1, 0.0, 0.1, 16 * math.pi, [0.0]),
    (1, 0.0, 0.1, 4 * math.pi, [1.23]),
    (1, 0.0, 1.0, math.pi, [0.0]),
    (1, 0.0, 10.0, math.pi, [0.77]),
    (1, 0.0, 10.0, 16 * math.pi, [0.0]),
    (1, 0.0, 100.0, math.pi, [0.0]),
    (2, 0.0, 1.0, 4 * math.pi, [0.41, 1.57]),
    (2, 1.0, 0.1, 16 * math.pi, [1.05, 0.2]),
    (2, 1.0, 1.0, 16 * math.pi, [0.0, 0.0]),
    (2, 10.0, 0.1, math.pi, [0.41, 1.57]),
    (2, 10.0, 10.0, 4 * math.pi, [1.05, 0.2]),
    (2, 100.0, 1.0, math.pi, [0.0, 0.0]),
    (2, 100.0, 10.0, math.pi, [0.41, 1.57]),
    (2, 100.0, 100.0, math.pi, [0.0, 0.0]),
    (4, 0.0, 0.1, 16 * math.pi, [0.3, 1.5, 0.9, 1.7]),
    (4, 0.0, 10.0, math.pi, [0.0, 0.0, 0.0, 0.0]),
    (4, 1.0, 1.0, 4 * math.pi, [0.3, 1.5, 0.9, 1.7]),
    (4, 1.0, 10.0, 4 * math.pi, [0.0, 0.0, 0.0, 0.0]),
    (4, 10.0, 10.0, 4 * math.pi, [0.0, 0.0, 0.0, 0.0]),
    (4, 10.0, 1.0, math.pi, [0.3, 1.5, 0.9, 1.7]),
    (4, 100.0, 0.1, math.pi, [0.0, 0.0, 0.0, 0.0]),
    (4, 100.0, 10.0, 4 * math.pi, [0.3, 1.5, 0.9, 1.7]),
    (4, 100.0, 100.0, math.pi, [0.3, 1.5, 0.9, 1.7]),
]


class TestStepCount:
    """
    ``step_count`` (#598) against the exponential of the full Liouvillian and
    the closed forms of ``docs/rydberg-model.md``: the count it returns gives
    features within ``tol``, not only an integer.
    """

    OMEGA = 4.0 * math.pi  # the reference Ω of the model page

    def case(self, case: SweepCase) -> tuple[torch.Tensor, float, float]:
        """``(H, t, γ)`` of a sweep case; ``V = c6 / 1⁶`` between neighbours."""
        n_atoms, v_over_omega, gamma_over_omega, omega_t, delta_over_omega = case
        h = rydberg_hamiltonian(
            AtomRegister.chain(n_atoms, 1.0),
            self.OMEGA,
            [d * self.OMEGA for d in delta_over_omega],
            c6=v_over_omega * self.OMEGA,
        )
        return h, omega_t / self.OMEGA, gamma_over_omega * self.OMEGA

    @pytest.mark.parametrize("case", SWEEP)
    def test_the_count_meets_tol_over_the_sweep(self, case: SweepCase) -> None:
        tol = 1e-6
        h, t, gamma = self.case(case)
        n = step_count(h, t, gamma=gamma, tol=tol)
        error = (features_of(h, t, gamma, n) - exact_features(h, t, gamma)).abs().max().item()
        assert error <= tol

    @pytest.mark.slow
    def test_the_count_meets_tol_at_gamma_100_omega_over_four_pi_pulses(self) -> None:
        """One of the cases left out of :data:`SWEEP` for run time: 82 000 steps."""
        tol = 1e-6
        h, t, gamma = self.case((1, 0.0, 100.0, 4 * math.pi, [0.0]))
        n = step_count(h, t, gamma=gamma, tol=tol)
        error = (features_of(h, t, gamma, n) - exact_features(h, t, gamma)).abs().max().item()
        assert error <= tol

    def test_the_false_agreement_does_not_stop_it(self) -> None:
        """
        One resonant atom, ``ΩT = 16π``, ``γ = 0.1 Ω``: doubling from one step
        would stop at once, since 1 and 2 steps agree to rounding, at
        ``⟨n⟩ = 0`` where the damped-Rabi closed form gives 0.358.  The count
        returned starts from ``2t (W + γ/2)``, with ``W = Ω`` for one resonant
        atom: 106 steps, so at least 212 come back.
        """
        tol = 1e-4
        h = single_atom(self.OMEGA, 0.0)
        gamma = 0.1 * self.OMEGA
        t = 16 * math.pi / self.OMEGA
        exact = damped_rabi(self.OMEGA, gamma, t)
        assert exact == pytest.approx(0.3578, abs=1e-4)
        assert two_run_estimate(h, t, gamma, 2) < 1e-12  # the false agreement
        assert features_of(h, t, gamma, 2).item() < 1e-12
        n = step_count(h, t, gamma=gamma, tol=tol)
        assert n >= 2 * math.ceil(2 * t * (self.OMEGA + gamma / 2)) == 212
        assert features_of(h, t, gamma, n).item() == pytest.approx(exact, abs=tol)

    @pytest.mark.slow
    def test_the_counter_example_of_the_model_page(self) -> None:
        """
        ``γ = 100 Ω``, ``Ωt = 10``: ``⟨n⟩ = 0.0905`` by the overdamped closed
        form, 0.128 with 200 steps (#507).  At ``tol = 1e-6`` the count meets it.
        """
        tol = 1e-6
        h = single_atom(self.OMEGA, 0.0)
        gamma = 100 * self.OMEGA
        t = 10 / self.OMEGA
        exact = damped_rabi(self.OMEGA, gamma, t)
        assert exact == pytest.approx(0.0905, abs=1e-4)
        assert features_of(h, t, gamma, 200).item() == pytest.approx(0.128, abs=1e-3)
        n = step_count(h, t, gamma=gamma, tol=tol)
        assert features_of(h, t, gamma, n).item() == pytest.approx(exact, abs=tol)

    @pytest.mark.parametrize(
        ("delta_over_omega", "gamma_over_omega", "omega_t"),
        [(0.0, 1.0, math.pi), (0.77, 10.0, math.pi), (1.23, 0.1, 4 * math.pi)],
    )
    def test_it_is_the_smallest_count_tried_from_the_resolution_condition(
        self, delta_over_omega: float, gamma_over_omega: float, omega_t: float
    ) -> None:
        """
        One atom, ``H = [[0, Ω/2], [Ω/2, −Δ]]``: its energies are
        ``(−Δ ± √(Ω² + Δ²))/2``, so ``W = √(Ω² + Δ²)`` and the coarsest run has
        ``n₀ = ceil(2t (W + γ/2))`` steps.  The counts tried are ``2n₀, 4n₀, …``
        (the run with ``n₀`` steps is only ever the coarser one); with a
        ``tol`` every estimate meets, the first comes back.  With a tight
        ``tol``, the returned count's estimate meets it and that of the count
        before it, if one was tried, does not.
        """
        omega = self.OMEGA
        delta = delta_over_omega * omega
        gamma = gamma_over_omega * omega
        t = omega_t / omega
        h = single_atom(omega, delta)
        start = math.ceil(2 * t * (math.sqrt(omega**2 + delta**2) + gamma / 2))
        assert step_count(h, t, gamma=gamma, tol=1.0) == 2 * start
        tol = 1e-7
        n = step_count(h, t, gamma=gamma, tol=tol)
        doublings = math.log2(n / start)
        assert doublings == int(doublings) >= 1
        assert two_run_estimate(h, t, gamma, n) <= tol
        if n > 2 * start:
            assert two_run_estimate(h, t, gamma, n // 2) > tol

    def test_one_count_for_a_batch_meets_tol_for_every_sample(self) -> None:
        """Six samples of the four-atom counter-example of #598 with their own detunings."""
        tol = 1e-5
        generator = torch.Generator().manual_seed(598)
        delta = math.sqrt(3) * self.OMEGA * torch.rand(6, 4, generator=generator, dtype=F64)
        h = rydberg_hamiltonian(AtomRegister.chain(4, 1.0), self.OMEGA, delta, c6=10 * self.OMEGA)
        t = 4 * math.pi / self.OMEGA
        gamma = 10 * self.OMEGA
        n = step_count(h, t, gamma=gamma, tol=tol)
        errors = (features_of(h, t, gamma, n) - exact_features(h, t, gamma)).abs().amax(dim=-1)
        assert errors.shape == (6,)
        assert bool((errors <= tol).all())
        # the order of the samples does not matter: the count is a maximum over them
        assert step_count(h.flip(0), t, gamma=gamma, tol=tol) == n

    def test_a_count_from_all_inputs_gives_the_feature_map_features_within_tol(self) -> None:
        """
        The use the docstring prescribes: one call on the Hamiltonians of all
        inputs, the result passed to ``RydbergFeatureMap(n_steps=...)``, where
        ``get_config()`` records it.  The map's features are then within
        ``tol`` of the exponential of the Liouvillian, sample by sample.
        """
        tol = 1e-5
        omega = self.OMEGA
        register = AtomRegister.chain(3, AtomRegister.blockade_radius(DEFAULT_C6, omega))
        encoding = PulseEncoding(3, omega=omega)
        gamma = 2.0 * omega
        X = torch.tensor([[-1.0, 0.0, 2.0], [0.5, 0.5, 0.5], [3.0, -2.0, 0.1]], dtype=F64)
        h = rydberg_hamiltonian(register, omega, encoding.detunings(X), c6=DEFAULT_C6)
        n = step_count(h, encoding.t, gamma=gamma, tol=tol)
        feature_map = RydbergFeatureMap(
            register, encoding, c6=DEFAULT_C6, gamma=gamma, n_steps=n, correlations=True
        )
        assert feature_map.get_config()["n_steps"] == n
        features = feature_map.transform(X)
        exact = exact_features(h, encoding.t, gamma)
        assert (features - exact).abs().max().item() <= tol

    def test_without_dephasing_one_step_is_exact(self) -> None:
        """
        ``γ = 0``: ``evolve`` is exact and does not use ``n_steps``, so the
        smallest count, 1, meets any ``tol``; nothing is evolved for it.
        """
        h, t, _ = self.case((4, 10.0, 1.0, 4 * math.pi, [0.3, 1.5, 0.9, 1.7]))
        assert step_count(h, t, gamma=0.0, tol=1e-9) == 1
        exact = exact_features(h, t, 0.0)
        assert torch.allclose(features_of(h, t, 0.0, 1), exact, rtol=0, atol=1e-12)

    def test_it_raises_where_rounding_stops_the_estimate_falling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A solver whose rounding grows with the step count, ``−2e-12 · n`` in
        ``⟨n⟩``, as the trace drift of ``evolve`` does, only faster.  One
        resonant atom, ``γ = Ω``, ``ΩT = π``: the splitting's own error is
        positive (``+4.6e-4`` at 10 steps), so the drift adds to the
        difference of two runs, ``2e-12 · n`` for ``n`` and ``2n`` steps.  The
        estimate falls by about 4 per doubling up to 2560 steps (``8.0e-9``),
        by 2.3 to 5120 (``3.5e-9``), and grows after that (``3.9e-9``), so it
        never comes down to ``tol = 1e-9`` and no count is returned.  It has
        to raise at 10 240 steps; a solver asked for more fails the test,
        which would otherwise run forever without the check.
        """
        real_evolve = lindblad.evolve

        def drifting(h: torch.Tensor, t: float, *, gamma: float, n_steps: int) -> torch.Tensor:
            if n_steps > 10_240:
                pytest.fail(f"step_count went on to {n_steps} steps after the estimate rose")
            rho = real_evolve(h, t, gamma=gamma, n_steps=n_steps).clone()
            rho[..., 1, 1] -= 2e-12 * n_steps
            rho[..., 0, 0] += 2e-12 * n_steps
            return rho

        monkeypatch.setattr(lindblad, "evolve", drifting)
        h = single_atom(self.OMEGA, 0.0)
        with pytest.raises(RuntimeError, match="stopped falling"):
            step_count(h, math.pi / self.OMEGA, gamma=self.OMEGA, tol=1e-9)


#: ``step_count`` of the :data:`SWEEP` case at that index, at ``tol = 1e-4``
#: and ``1e-6``, as ``main`` returned them at 3f8ca86, before a target for ρ
#: (#759) was considered.  Every one is ``2^k n₀`` with ``k >= 1``.
FEATURE_COUNTS: dict[int, tuple[int, int]] = {
    0: (212, 424),
    1: (84, 672),
    2: (40, 320),
    3: (320, 2560),
    4: (1208, 1208),
    5: (1284, 10272),
    6: (200, 800),
    7: (474, 948),
    8: (666, 666),
    9: (126, 126),
    10: (1012, 8096),
    11: (1280, 1280),
    12: (1380, 2760),
    13: (2524, 10096),
    14: (1280, 2560),
    15: (302, 2416),
    16: (354, 708),
    17: (1274, 10192),
    18: (2602, 10408),
    19: (396, 396),
    20: (3830, 3830),
    21: (16162, 16162),
    22: (6304, 12608),
}

#: The same at ``tol = 1e-9``, for the cases that take less than half a
#: second there (the others reach 403 456 steps).
FEATURE_COUNTS_AT_THE_SMALLEST_TOL: dict[int, int] = {
    0: 13568,
    2: 10240,
    8: 666,
    9: 4032,
    11: 20480,
    19: 12672,
    20: 7660,
}


class TestStepCountKeepsItsResults:
    """
    The counts ``step_count`` returns for the features, pinned to the integer,
    so that a target for the state (#759) cannot change a call made today.
    """

    @pytest.mark.parametrize("index", sorted(FEATURE_COUNTS))
    def test_the_feature_counts_of_the_sweep_are_unchanged(self, index: int) -> None:
        h, t, gamma = TestStepCount().case(SWEEP[index])
        expected = FEATURE_COUNTS[index]
        assert step_count(h, t, gamma=gamma, tol=1e-4) == expected[0]
        assert step_count(h, t, gamma=gamma, tol=1e-6) == expected[1]
        # the default target is the features
        assert step_count(h, t, gamma=gamma, tol=1e-6, target="features") == expected[1]

    @pytest.mark.parametrize("index", sorted(FEATURE_COUNTS_AT_THE_SMALLEST_TOL))
    def test_the_feature_counts_at_the_smallest_tol_are_unchanged(self, index: int) -> None:
        h, t, gamma = TestStepCount().case(SWEEP[index])
        expected = FEATURE_COUNTS_AT_THE_SMALLEST_TOL[index]
        assert step_count(h, t, gamma=gamma, tol=1e-9) == expected


def states_of(h: torch.Tensor, t: float, gamma: float, n_steps: int) -> torch.Tensor:
    """ρ from ``evolve`` at ``n_steps``, one matrix per sample."""
    batch = h if h.ndim == 3 else h[None]
    return checked_evolve(batch, t, gamma=gamma, n_steps=n_steps)


def exact_states(h: torch.Tensor, t: float, gamma: float) -> torch.Tensor:
    """ρ from :func:`liouvillian_reference`, sample by sample."""
    batch = h if h.ndim == 3 else h[None]
    return torch.stack([liouvillian_reference(s, t, gamma) for s in batch])


def trace_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    ``½ ‖ρ − σ‖₁`` per sample, from the singular values (``ord="nuc"``), not
    from the eigenvalues ``step_count`` uses.
    """
    return 0.5 * torch.linalg.matrix_norm(rho - sigma, ord="nuc")


def two_run_state_estimate(h: torch.Tensor, t: float, gamma: float, n_steps: int) -> float:
    """A third of the largest ``D(ρ_(n/2), ρ_n)`` over the samples."""
    coarse = states_of(h, t, gamma, n_steps // 2)
    return trace_distance(coarse, states_of(h, t, gamma, n_steps)).max().item() / 3


def whole_sweep() -> list[SweepCase]:
    """
    The 216 cases of #598, in the units of :data:`SWEEP`: 1, 2 and 4 atoms;
    ``V/Ω`` 0, 1, 10, 100 (one atom has no pair, so 0 only); ``γ/Ω`` 0.1, 1,
    10, 100; ``ΩT`` π, 4π, 16π; detunings all 0, or uniform in ``[0, √3]``
    from one generator seeded 759, drawn in this order.
    """
    generator = torch.Generator().manual_seed(759)
    cases: list[SweepCase] = []
    for n_atoms in (1, 2, 4):
        for v in (0.0,) if n_atoms == 1 else (0.0, 1.0, 10.0, 100.0):
            for g in (0.1, 1.0, 10.0, 100.0):
                for omega_t in (math.pi, 4 * math.pi, 16 * math.pi):
                    cases.append((n_atoms, v, g, omega_t, [0.0] * n_atoms))
                    drawn = math.sqrt(3) * torch.rand(n_atoms, generator=generator, dtype=F64)
                    cases.append((n_atoms, v, g, omega_t, drawn.tolist()))
    return cases


class TestStepCountForTheState:
    """
    ``step_count(..., target="state")`` (#759) against the exponential of the
    full Liouvillian: the count it returns gives states within ``tol`` in the
    trace distance, and the inequalities its docstring derives hold.
    """

    OMEGA = TestStepCount.OMEGA

    @pytest.mark.parametrize("index", range(len(SWEEP)))
    def test_the_count_meets_tol_over_the_sweep(self, index: int) -> None:
        """
        Over :data:`SWEEP`, at ``tol = 1e-6``; the count is never below the
        features' at the same ``tol`` (:data:`FEATURE_COUNTS`), since ``D``
        bounds every feature's error.
        """
        tol = 1e-6
        h, t, gamma = TestStepCount().case(SWEEP[index])
        n = step_count(h, t, gamma=gamma, tol=tol, target="state")
        error = trace_distance(states_of(h, t, gamma, n), exact_states(h, t, gamma))
        assert error.max().item() <= tol
        assert n >= FEATURE_COUNTS[index][1]

    @pytest.mark.reproducibility
    @pytest.mark.slow
    @pytest.mark.may_skip  # opt-in: skips in CI even under HQNN_FORGE_FAIL_ON_SKIP=1
    def test_the_count_meets_tol_over_the_whole_sweep_of_598(self) -> None:
        """
        All 216 cases of :func:`whole_sweep` at ``tol = 1e-6``: 17 minutes
        on a laptop, so opt-in (``HQNN_FORGE_REPRODUCE=1``).  At the count
        returned the error is at most ``tol`` and, where the estimate is above
        the rounding floor, 0.9 to 1.2 times it (measured: 0.986 to 1.084).
        """
        if os.environ.get("HQNN_FORGE_REPRODUCE") != "1":
            pytest.skip("opt-in: set HQNN_FORGE_REPRODUCE=1 (the 216 cases of #598's sweep)")
        tol = 1e-6
        failures = []
        for case in whole_sweep():
            h, t, gamma = TestStepCount().case(case)
            n = step_count(h, t, gamma=gamma, tol=tol, target="state")
            error = trace_distance(states_of(h, t, gamma, n), exact_states(h, t, gamma))
            estimate = two_run_state_estimate(h, t, gamma, n)
            ratio = error.max().item() / estimate if estimate > 1e-10 else 1.0
            if error.max().item() > tol or not 0.9 <= ratio <= 1.2:
                failures.append((case, n, error.max().item(), estimate))
        assert failures == []

    @pytest.mark.parametrize(
        ("delta_over_omega", "gamma_over_omega", "omega_t"),
        [(0.0, 1.0, math.pi), (0.77, 10.0, math.pi), (1.23, 0.1, 4 * math.pi)],
    )
    def test_it_is_the_smallest_count_tried_from_the_resolution_condition(
        self, delta_over_omega: float, gamma_over_omega: float, omega_t: float
    ) -> None:
        """
        As for the features: one atom, ``W = √(Ω² + Δ²)``, the counts tried
        are ``2n₀, 4n₀, …``, and the one returned is the first whose estimate,
        recomputed here from the singular values, meets ``tol``.
        """
        omega = self.OMEGA
        delta = delta_over_omega * omega
        gamma = gamma_over_omega * omega
        t = omega_t / omega
        h = single_atom(omega, delta)
        start = math.ceil(2 * t * (math.sqrt(omega**2 + delta**2) + gamma / 2))
        assert step_count(h, t, gamma=gamma, tol=1.0, target="state") == 2 * start
        tol = 1e-7
        n = step_count(h, t, gamma=gamma, tol=tol, target="state")
        doublings = math.log2(n / start)
        assert doublings == int(doublings) >= 1
        assert two_run_state_estimate(h, t, gamma, n) <= tol
        if n > 2 * start:
            assert two_run_state_estimate(h, t, gamma, n // 2) > tol

    def test_one_count_for_a_batch_meets_tol_for_every_sample(self) -> None:
        """
        Two atoms at ``V = γ = Ω``, ``ΩT = 4π``: five resonant samples and one
        detuned (``Δ = 8 Ω, 9 Ω``) whose error is 17 times theirs.  The count
        is set by the largest error, not a typical one, so every sample meets
        ``tol``, and the order of the samples does not matter.
        """
        tol = 1e-6
        delta = torch.tensor([[0.0, 0.0]] * 5 + [[8.0, 9.0]], dtype=F64) * self.OMEGA
        h = rydberg_hamiltonian(AtomRegister.chain(2, 1.0), self.OMEGA, delta, c6=self.OMEGA)
        t = 4 * math.pi / self.OMEGA
        gamma = self.OMEGA
        n = step_count(h, t, gamma=gamma, tol=tol, target="state")
        errors = trace_distance(states_of(h, t, gamma, n), exact_states(h, t, gamma))
        assert errors.shape == (6,)
        assert bool((errors <= tol).all())
        assert errors[-1].item() > 10 * errors[0].item()
        assert step_count(h.flip(0), t, gamma=gamma, tol=tol, target="state") == n

    def test_the_trace_distance_bounds_the_features_the_entries_and_the_kernel(self) -> None:
        """
        The inequalities of *The state*: for ``Δ = ρ_n − ρ`` of each of four
        samples (four atoms, ``V = γ = 10 Ω``, ``ΩT = 4π``, the
        counter-example of #598), every feature error and every entry is at
        most ``D``, the Hilbert–Schmidt norm at most ``√2 D``, and the state
        kernel ``Tr[ρ ρ′]`` of every pair is off by at most ``D + D′``.  They
        need only that every ``ρ_n`` is a state, which it is at any count, so
        they are checked at 16 steps, far below the resolution condition
        (``n₀ = 1269``), where the errors are large.
        """
        generator = torch.Generator().manual_seed(759)
        delta = math.sqrt(3) * self.OMEGA * torch.rand(4, 4, generator=generator, dtype=F64)
        h = rydberg_hamiltonian(AtomRegister.chain(4, 1.0), self.OMEGA, delta, c6=10 * self.OMEGA)
        t = 4 * math.pi / self.OMEGA
        gamma = 10 * self.OMEGA
        rho = states_of(h, t, gamma, 16)
        exact = exact_states(h, t, gamma)
        d = trace_distance(rho, exact)
        assert bool((d > 1e-2).all())  # far above rounding
        feature_error = (readout(rho, pairs=True) - readout(exact, pairs=True)).abs().amax(-1)
        assert bool((feature_error <= d).all())
        assert bool(((rho - exact).abs().amax(dim=(-2, -1)) <= d).all())
        assert bool((torch.linalg.matrix_norm(rho - exact) <= math.sqrt(2) * d).all())
        kernel_error = (
            kernel_from_density_matrices(rho) - kernel_from_density_matrices(exact)
        ).abs()
        assert bool((kernel_error <= d[:, None] + d[None, :]).all())

    def test_the_hilbert_schmidt_norm_does_not_bound_the_features(self) -> None:
        """
        Why the target is not ``‖Δ‖₂``: four uncoupled atoms at ``γ = Ω``,
        ``ΩT = 16π`` with drawn detunings (case 131 of :func:`whole_sweep`),
        at ``2n₀ = 1484`` steps, the largest feature error (``4.0e-8``) is
        above the Hilbert–Schmidt norm of ``Δ = ρ_n − ρ`` (``2.5e-8``), so a
        count for ``‖Δ‖₂ <= tol`` can leave the features above ``tol``.  The
        trace distance is above both.
        """
        case = whole_sweep()[131]
        assert case[:4] == (4, 0.0, 1.0, 16 * math.pi)
        h, t, gamma = TestStepCount().case(case)
        n = 2 * math.ceil(2 * t * (lindblad._spread(h) + 4 * gamma / 2))
        assert n == 1484
        rho = states_of(h, t, gamma, n)
        exact = exact_states(h, t, gamma)
        feature_error = (readout(rho, pairs=True) - readout(exact, pairs=True)).abs().max()
        hilbert_schmidt = torch.linalg.matrix_norm(rho - exact).max()
        assert feature_error.item() > hilbert_schmidt.item() > 1e-9
        assert trace_distance(rho, exact).max().item() >= feature_error.item()

    def test_the_trace_distance_of_two_pure_states(self) -> None:
        """
        ``D(ψψ†, φφ†) = √(1 − |⟨ψ|φ⟩|²)`` for pure states, typed in; the
        largest over a batch comes back.
        """
        psi = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=C128)
        angles = torch.tensor([0.3, 1.1], dtype=F64)
        phi = torch.stack([torch.cos(angles), 1j * torch.sin(angles)], dim=-1).to(C128)
        rho = psi[:, :, None] * psi[:, None, :].conj()
        sigma = phi[:, :, None] * phi[:, None, :].conj()
        # |⟨ψ|φ⟩| = cos(angle), so D = sin(angle); the larger is sin(1.1)
        assert lindblad._largest_trace_distance(rho, sigma) == pytest.approx(
            math.sin(1.1), abs=1e-14
        )
        assert lindblad._largest_trace_distance(rho[:1], sigma[:1]) == pytest.approx(
            math.sin(0.3), abs=1e-14
        )

    def test_without_dephasing_one_step_is_exact(self) -> None:
        """``γ = 0``: 1 comes back for the state too, and ρ is exact at it."""
        h, t, _ = TestStepCount().case((4, 10.0, 1.0, 4 * math.pi, [0.3, 1.5, 0.9, 1.7]))
        assert step_count(h, t, gamma=0.0, tol=1e-9, target="state") == 1
        error = trace_distance(states_of(h, t, 0.0, 1), exact_states(h, t, 0.0))
        assert error.max().item() < 1e-12


class TestStepCountValidation:
    H = rydberg_hamiltonian(AtomRegister.chain(2, 1.0), 2.0, [0.1, 0.2], c6=1.0)

    @pytest.mark.parametrize("target", ["State", "trace", "", None, 1])
    def test_rejects_an_unknown_target(self, target: object) -> None:
        with pytest.raises(ValueError, match="target must be one of 'features', 'state'"):
            step_count(self.H, 1.0, gamma=0.5, tol=1e-6, target=target)  # type: ignore[arg-type]

    def test_tol_is_required_and_keyword_only(self) -> None:
        with pytest.raises(TypeError, match="tol"):
            step_count(self.H, 1.0, gamma=0.5)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="positional"):
            step_count(self.H, 1.0, 0.5, 1e-6)  # type: ignore[misc]

    @pytest.mark.parametrize("tol", [0.0, -1e-6, 1e-10, math.nan, math.inf, "1e-6", True, None])
    def test_rejects_a_tol_that_is_not_a_number_of_at_least_1e_9(self, tol: object) -> None:
        """Below about ``1e-10`` rounding decides the estimate; ``1e-9`` keeps a factor 10."""
        with pytest.raises(ValueError, match=r"tol must be a finite number >= 1e-09"):
            step_count(self.H, 1.0, gamma=0.5, tol=tol)  # type: ignore[arg-type]

    def test_the_smallest_tol_is_accepted(self) -> None:
        assert step_count(self.H, 0.1, gamma=0.5, tol=1e-9) >= 2

    @pytest.mark.parametrize("t", [-0.1, math.nan, "1.0", True])
    def test_rejects_a_bad_time(self, t: object) -> None:
        with pytest.raises(ValueError, match="t must be a finite number >= 0"):
            step_count(self.H, t, gamma=0.5, tol=1e-6)  # type: ignore[arg-type]

    @pytest.mark.parametrize("gamma", [-0.1, math.nan, None])
    def test_rejects_a_bad_rate(self, gamma: object) -> None:
        with pytest.raises(ValueError, match="gamma must be a finite number >= 0"):
            step_count(self.H, 1.0, gamma=gamma, tol=1e-6)  # type: ignore[arg-type]

    def test_rejects_what_evolve_rejects(self) -> None:
        with pytest.raises(TypeError, match="hamiltonian must be a torch.Tensor"):
            step_count([[0.0, 1.0], [1.0, 0.0]], 1.0, gamma=0.5, tol=1e-6)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"hamiltonian must have shape \(2\^n, 2\^n\)"):
            step_count(torch.zeros(3, 3, dtype=C128), 1.0, gamma=0.5, tol=1e-6)
        bad = self.H.clone()
        bad[0, 1] = bad[0, 1] + 0.5
        with pytest.raises(ValueError, match="hamiltonian must be Hermitian"):
            step_count(bad, 1.0, gamma=0.5, tol=1e-6)

    def test_rejects_an_empty_batch(self) -> None:
        """No sample, no count: the result is meant to be fixed by all inputs."""
        with pytest.raises(ValueError, match="at least one sample"):
            step_count(torch.zeros(0, 4, 4, dtype=C128), 1.0, gamma=0.5, tol=1e-6)


def trapezoid_segments(
    omega: float, t: float, t_ramp: float, n_segments: int | None
) -> list[tuple[float, float]]:
    """
    ``(Ω_k, τ_k)`` of the ramped drive, typed in from #732, in time order.

    ``n`` segments of ``t_ramp/n`` per edge at the midpoint values
    ``Ω (k + ½)/n`` rising and the same values in reverse falling, the plateau
    ``(Ω, T − t_ramp)`` between them where it is not empty; ``[(Ω, T)]``
    without a ramp.
    """
    if n_segments is None:
        return [(omega, t)]
    tau = t_ramp / n_segments
    rising = [(omega * (k + 0.5) / n_segments, tau) for k in range(n_segments)]
    plateau = [(omega, t - t_ramp)] if t > t_ramp else []
    return rising + plateau + rising[::-1]


def displaced_registers(
    register: AtomRegister, sigma: float, samples: int | None, seed: int | None
) -> list[AtomRegister]:
    """
    The registers of position jitter, drawn by hand as the feature map documents.

    One ``torch.randn((K, N, 2), float64)`` from a CPU generator seeded with
    ``seed``: entry ``[k, i, c]`` displaces coordinate ``c`` of atom ``i`` in
    realisation ``k`` by ``σ`` times it.  ``[register]`` without jitter.
    """
    if samples is None or seed is None:
        return [register]
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn((samples, register.n_atoms, 2), generator=generator, dtype=F64)
    return [AtomRegister(register.positions + sigma * z[k]) for k in range(samples)]


def chained_features(
    registers: list[AtomRegister],
    segments: list[tuple[float, float]],
    delta: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    ``⟨n_i⟩``, ``⟨n_i n_j⟩`` of the exact chained evolution, averaged over ``registers``.

    For every sample (row of ``delta``, in rad/µs) and register: the
    exponential of the full Liouvillian of each segment ``(Ω_k, τ_k)`` in
    turn (:func:`liouvillian_reference`), the first from ``|g…g⟩`` and each
    other from the state the one before left; then the mean over the
    registers of the final states' features.  No splitting, no
    eigendecomposition and no mixture of states, so nothing of the feature
    map's own path.
    """
    rows = []
    for detunings in delta:
        per_register = []
        for register in registers:
            rho: torch.Tensor | None = None
            for omega, tau in segments:
                h = rydberg_hamiltonian(register, omega, detunings, c6=DEFAULT_C6)
                rho = liouvillian_reference(h, tau, gamma, rho)
            assert rho is not None
            per_register.append(readout(rho, pairs=True))
        rows.append(torch.stack(per_register).mean(dim=0))
    return torch.stack(rows)


#: A feature map case of #760: atoms in a chain with ``V/Ω`` between
#: neighbours, ``γ/Ω``, the ramp as a fraction of ``T`` with its segments per
#: edge (``0.0, None``: the square pulse of a ``PulseEncoding``), and the
#: jitter ``σ`` in µm with ``K`` (``0.0, None``: none), at the reference
#: ``Ω = 4π`` rad/µs and ``T = π/Ω``.
MapCase = tuple[int, float, float, float, int | None, float, int | None]


class TestFeatureMapStepCount:
    """
    ``RydbergFeatureMap.step_count`` (#760): one count for a map that runs
    more than one ``evolve`` per sample, the segments of a ramped drive and
    the displaced registers of position jitter.  The expected features are
    those of :func:`chained_features`: the exponential of the full
    Liouvillian per segment, chained, and averaged over the registers drawn by
    hand.  The segments are typed in, the detunings are ``√3 Ω σ(x)``.
    """

    OMEGA = 4.0 * math.pi  # the reference Ω of the model page
    T = math.pi / OMEGA
    SEED = 760
    X = torch.tensor([[-1.0, 0.4, 2.0], [0.3, -0.6, 0.0]], dtype=F64)

    def build(self, case: MapCase) -> tuple[RydbergFeatureMap, torch.Tensor]:
        """The map of ``case`` (with a placeholder ``n_steps = 1``) and its inputs."""
        n_atoms, v_over_omega, gamma_over_omega, ramp, segments, sigma, samples = case
        omega = self.OMEGA
        spacing = (DEFAULT_C6 / (v_over_omega * omega)) ** (1 / 6)
        encoding: PulseEncoding | GlobalDrivePulseEncoding = (
            PulseEncoding(n_atoms, omega=omega)
            if segments is None
            else GlobalDrivePulseEncoding(
                n_atoms, omega=omega, levels=None, t_ramp=ramp * self.T, n_ramp_segments=segments
            )
        )
        jitter: dict[str, float | int | None] = (
            {}
            if samples is None
            else {"position_jitter": sigma, "jitter_samples": samples, "jitter_seed": self.SEED}
        )
        feature_map = RydbergFeatureMap(
            AtomRegister.chain(n_atoms, spacing),
            encoding,
            c6=DEFAULT_C6,
            gamma=gamma_over_omega * omega,
            n_steps=1,
            correlations=True,
            **jitter,  # type: ignore[arg-type]
        )
        return feature_map, self.X[:, :n_atoms]

    def exact(
        self, case: MapCase, feature_map: RydbergFeatureMap, x: torch.Tensor
    ) -> torch.Tensor:
        _, _, _, ramp, segments, sigma, samples = case
        registers = displaced_registers(feature_map.register, sigma, samples, self.SEED)
        drive = trapezoid_segments(self.OMEGA, self.T, ramp * self.T, segments)
        delta = math.sqrt(3) * self.OMEGA * torch.sigmoid(x)
        return chained_features(registers, drive, delta, feature_map.gamma)

    def starting_count(
        self, case: MapCase, feature_map: RydbergFeatureMap, x: torch.Tensor
    ) -> int:
        """
        ``n₀ = max ceil(2 τ_k (W + Nγ/2))`` over the typed-in segments, the
        registers drawn by hand and the samples, ``W`` from ``eigvalsh``.
        """
        n_atoms, _, _, ramp, segments, sigma, samples = case
        registers = displaced_registers(feature_map.register, sigma, samples, self.SEED)
        delta = math.sqrt(3) * self.OMEGA * torch.sigmoid(x)
        start = 1
        for omega, tau in trapezoid_segments(self.OMEGA, self.T, ramp * self.T, segments):
            for register in registers:
                energies = torch.linalg.eigvalsh(
                    rydberg_hamiltonian(register, omega, delta, c6=DEFAULT_C6)
                )
                spread = (energies.amax(dim=-1) - energies.amin(dim=-1)).max().item()
                rate = spread + n_atoms * feature_map.gamma / 2
                start = max(start, math.ceil(2 * tau * rate))
        return start

    @staticmethod
    def features_at(feature_map: RydbergFeatureMap, x: torch.Tensor, n_steps: int) -> torch.Tensor:
        rebuilt = RydbergFeatureMap(**{**feature_map.get_config(), "n_steps": n_steps})
        return rebuilt.transform(x)

    #: Ramped drives, a triangle, a jittered register under the square pulse,
    #: and both at once.
    CASES: list[MapCase] = [
        (2, 1.0, 1.0, 0.2, 3, 0.0, None),
        (3, 1.0, 0.1, 0.5, 2, 0.0, None),
        (2, 10.0, 10.0, 1.0, 2, 0.0, None),
        (3, 1.0, 1.0, 0.0, None, 0.5, 3),
        (2, 10.0, 4.0, 0.2, 2, 0.3, 2),
    ]

    @pytest.mark.parametrize("case", CASES)
    def test_the_count_meets_tol_against_the_chained_liouvillian(self, case: MapCase) -> None:
        tol = 1e-6
        feature_map, x = self.build(case)
        n = feature_map.step_count(x, tol=tol)
        doublings = math.log2(n / self.starting_count(case, feature_map, x))
        assert doublings == int(doublings) >= 1  # one doubling sequence for the whole pulse
        error = (self.features_at(feature_map, x, n) - self.exact(case, feature_map, x)).abs()
        assert error.max().item() <= tol

    @pytest.mark.parametrize("case", CASES)
    def test_a_third_of_the_difference_is_the_error_of_the_chain(self, case: MapCase) -> None:
        """
        The design of #760, measured: with ``n₀`` from the largest rate over
        every segment and register, the features of the whole map at ``2n₀``
        steps per segment are off the chained exponential by a third of their
        largest difference from ``n₀`` steps, within 10 %, as for one
        segment (``TestChoosingTheStepCount``).
        """
        feature_map, x = self.build(case)
        start = self.starting_count(case, feature_map, x)
        coarse = self.features_at(feature_map, x, start)
        fine = self.features_at(feature_map, x, 2 * start)
        error = (fine - self.exact(case, feature_map, x)).abs().max().item()
        estimate = (coarse - fine).abs().max().item() / 3
        assert error > 1e-9  # above the rounding floor, so the ratio is not noise
        assert error / estimate == pytest.approx(1.0, rel=0.1)

    # The plateau sets n₀, or, with a long ramp of one segment per edge, the rising edge.
    @pytest.mark.parametrize(
        ("ramp", "segments", "longest"), [(0.2, 2, 2), (0.0, None, 0), (0.9, 1, 0)]
    )
    def test_the_start_is_the_largest_over_the_segments(
        self, ramp: float, segments: int | None, longest: int
    ) -> None:
        """
        One atom, ``Δ = √3 σ(0.4) Ω ≈ 1.04 Ω``, ``γ = Ω``: segment ``k`` has
        ``W_k = √(Ω_k² + Δ²)``, so ``n₀ = max_k ceil(2 τ_k (W_k + γ/2))``,
        typed in; with a ``tol`` that every estimate meets, ``2 n₀`` comes
        back.  ``longest`` is the segment that sets it.
        """
        omega, gamma, x = self.OMEGA, self.OMEGA, 0.4
        delta = math.sqrt(3) * omega / (1 + math.exp(-x))
        counts = [
            math.ceil(2 * tau * (math.sqrt(omega_k**2 + delta**2) + gamma / 2))
            for omega_k, tau in trapezoid_segments(omega, self.T, ramp * self.T, segments)
        ]
        assert counts.index(max(counts)) == longest
        feature_map, _ = self.build((1, 1.0, 1.0, ramp, segments, 0.0, None))
        assert feature_map.step_count([[x]], tol=1.0) == 2 * max(counts)

    def test_the_start_is_the_largest_over_the_displaced_registers(self) -> None:
        """
        Two atoms 5 µm apart (``V ≈ 28 Ω``), ``σ = 0.3`` µm: the displaced
        registers have other couplings, hence other ``W``, and the start is
        the largest of their counts, not the undisplaced register's.
        """
        omega, gamma = self.OMEGA, self.OMEGA
        register = AtomRegister.chain(2, 5.0)
        feature_map = RydbergFeatureMap(
            register,
            PulseEncoding(2, omega=omega),
            c6=DEFAULT_C6,
            gamma=gamma,
            n_steps=1,
            position_jitter=0.3,
            jitter_samples=3,
            jitter_seed=self.SEED,
        )
        x = self.X[:1, :2]
        delta = math.sqrt(3) * omega * torch.sigmoid(x)

        def count(r: AtomRegister) -> int:
            energies = torch.linalg.eigvalsh(rydberg_hamiltonian(r, omega, delta, c6=DEFAULT_C6))
            return math.ceil(2 * self.T * ((energies.max() - energies.min()).item() + gamma))

        counts = [count(r) for r in displaced_registers(register, 0.3, 3, self.SEED)]
        assert max(counts) != count(register)
        assert feature_map.step_count(x, tol=1.0) == 2 * max(counts)

    def test_without_ramp_or_jitter_it_is_step_count_on_the_maps_hamiltonians(self) -> None:
        """The recipe of ``step_count``'s Notes, bit for bit: one segment, one register."""
        tol = 1e-6
        feature_map, x = self.build((3, 1.0, 1.0, 0.0, None, 0.0, None))
        encoding = feature_map.encoding
        h = rydberg_hamiltonian(
            feature_map.register, encoding.omega, encoding.detunings(x), c6=DEFAULT_C6
        )
        n = step_count(h, encoding.t, gamma=feature_map.gamma, tol=tol)
        assert feature_map.step_count(x, tol=tol) == n

    def test_the_chunk_size_does_not_change_it(self) -> None:
        feature_map, x = self.build(self.CASES[-1])
        whole = feature_map.step_count(x, tol=1e-4)
        assert feature_map.step_count(x, tol=1e-4, chunk_size=1) == whole

    @pytest.mark.parametrize("case", [CASES[0], CASES[3]])
    def test_it_is_the_smallest_count_tried_for_the_maps_features(self, case: MapCase) -> None:
        """
        The estimate at the returned count, from two runs of the whole map at
        ``n/2`` and ``n`` steps, meets ``tol``, and that of the count before
        it does not.  With jitter it is the estimate of the mixture, the
        features the map returns: in the second case bounding each displaced
        register alone, as a map of its own, would need twice the count (two of
        the three miss ``tol`` at it).
        """
        tol = 1e-6
        feature_map, x = self.build(case)
        n = feature_map.step_count(x, tol=tol)
        assert n > 2 * self.starting_count(case, feature_map, x)  # a count before it was tried

        def estimate(m: RydbergFeatureMap, steps: int) -> float:
            coarse = self.features_at(m, x, steps // 2)
            return (coarse - self.features_at(m, x, steps)).abs().max().item() / 3

        assert estimate(feature_map, n) <= tol < estimate(feature_map, n // 2)
        if case[-1] is not None:
            alone = [
                RydbergFeatureMap(
                    **{
                        **feature_map.get_config(),
                        "register": register.positions.tolist(),
                        "position_jitter": 0.0,
                        "jitter_samples": None,
                        "jitter_seed": None,
                    }
                )
                for register in displaced_registers(
                    feature_map.register, case[5], case[6], self.SEED
                )
            ]
            assert max(estimate(m, n) for m in alone) > tol

    def test_settings_after_the_state_do_not_change_it(self) -> None:
        """
        The map's own ``n_steps``, ``correlations`` and ``readout_error`` do not
        enter: the target is ``⟨n_i⟩``, ``⟨n_i n_j⟩`` of the state the map reads.
        In this case the pairs set the count (``⟨n_i⟩`` alone would meet
        ``tol`` at half of it), so ``correlations=False`` would show.
        """
        feature_map, x = self.build(self.CASES[1])
        n = feature_map.step_count(x, tol=1e-6)
        config = feature_map.get_config()
        for changed in (
            {"n_steps": 977},
            {"correlations": False},
            {"readout_error": (0.05, 0.1)},
        ):
            other = RydbergFeatureMap(**{**config, **changed})
            assert other.step_count(x, tol=1e-6) == n

    def test_without_dephasing_one_step_is_exact(self) -> None:
        """``γ = 0``: every segment is exact in one step, so 1 comes back, with jitter and ramp."""
        feature_map, x = self.build(self.CASES[-1])
        noiseless = RydbergFeatureMap(**{**feature_map.get_config(), "gamma": 0.0})
        assert noiseless.step_count(x, tol=1e-9) == 1
        exact = self.exact(self.CASES[-1], noiseless, x)
        torch.testing.assert_close(noiseless.transform(x), exact, rtol=0, atol=1e-12)

    def test_tol_is_required_and_keyword_only(self) -> None:
        feature_map, x = self.build(self.CASES[0])
        with pytest.raises(TypeError, match="tol"):
            feature_map.step_count(x)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="positional"):
            feature_map.step_count(x, 1e-6)  # type: ignore[misc]

    @pytest.mark.parametrize("tol", [0.0, 1e-10, math.nan, "1e-6", True, None])
    def test_rejects_the_tol_step_count_rejects(self, tol: object) -> None:
        feature_map, x = self.build(self.CASES[0])
        with pytest.raises(ValueError, match=r"tol must be a finite number >= 1e-09"):
            feature_map.step_count(x, tol=tol)  # type: ignore[arg-type]

    def test_rejects_inputs_transform_rejects(self) -> None:
        feature_map, x = self.build(self.CASES[0])
        with pytest.raises(ValueError, match=r"X must have shape \(n_samples, 2\)"):
            feature_map.step_count(self.X, tol=1e-6)
        with pytest.raises(ValueError, match="chunk_size must be an integer >= 1"):
            feature_map.step_count(x, tol=1e-6, chunk_size=0)

    def test_rejects_an_empty_batch(self) -> None:
        feature_map, _ = self.build(self.CASES[0])
        with pytest.raises(ValueError, match="at least one sample"):
            feature_map.step_count(torch.zeros(0, 2, dtype=F64), tol=1e-6)


def random_density_matrix(n_atoms: int, seed: int) -> torch.Tensor:
    """``A A† / Tr(A A†)`` for a complex Gaussian ``A``: a full-rank mixed state, nothing symmetric."""
    generator = torch.Generator().manual_seed(seed)
    dim = 2**n_atoms
    a = torch.randn(dim, dim, generator=generator, dtype=F64) + 1j * torch.randn(
        dim, dim, generator=generator, dtype=F64
    )
    rho = a @ a.mH
    return rho / rho.diagonal().sum()


class TestInitialState:
    """
    ``rho0``: the evolution starts from a given ``ρ(0)``, so a piecewise-constant
    pulse is a chain of calls, each starting from the previous segment's state.
    """

    REGISTER = AtomRegister([[0.0, 0.0], [1.0, 0.1], [0.4, 1.3]])
    OMEGA = 2.0

    def hamiltonian(self, delta: list[float], n_atoms: int = 3) -> torch.Tensor:
        register = AtomRegister(self.REGISTER.positions[:n_atoms].tolist())
        return rydberg_hamiltonian(register, self.OMEGA, delta[:n_atoms], c6=1.5 * self.OMEGA)

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (0.6, 40)])
    def test_the_ground_state_given_explicitly_is_the_default(
        self, gamma: float, n_steps: int | None
    ) -> None:
        h = self.hamiltonian([0.5, -1.2, 2.1])
        ground = torch.zeros(8, 8, dtype=C128)
        ground[0, 0] = 1.0
        default = checked_evolve(h, 1.3, gamma=gamma, n_steps=n_steps)
        given = checked_evolve(h, 1.3, gamma=gamma, n_steps=n_steps, rho0=ground)
        assert torch.allclose(given, default, rtol=0, atol=1e-13)

    @pytest.mark.parametrize("gamma_over_omega", [0.0, 0.3, 2.0])
    @pytest.mark.parametrize("n_atoms", [1, 2, 3])
    def test_a_mixed_initial_state_matches_the_exponential_of_the_generator(
        self, n_atoms: int, gamma_over_omega: float
    ) -> None:
        """A full-rank random ``ρ(0)``: every element of it is evolved, not only ``ρ_00``."""
        h = self.hamiltonian([0.5, -1.2, 2.1], n_atoms)
        rho0 = random_density_matrix(n_atoms, seed=n_atoms)
        t = 2.5 / self.OMEGA
        gamma = gamma_over_omega * self.OMEGA
        rho = checked_evolve(h, t, gamma=gamma, n_steps=4000 if gamma > 0 else None, rho0=rho0)
        expected = liouvillian_reference(h, t, gamma, rho0)
        assert torch.allclose(rho, expected, rtol=0, atol=1e-6 if gamma > 0 else 1e-12)

    @pytest.mark.parametrize("gamma_over_omega", [0.0, 0.3, 2.0])
    def test_two_segments_match_the_product_of_their_generators(
        self, gamma_over_omega: float
    ) -> None:
        """``ρ = e^(𝓛₂ τ₂) e^(𝓛₁ τ₁) ρ(0)``: the piecewise-constant pulse the analog layer runs."""
        first = self.hamiltonian([0.5, -1.2, 2.1])
        second = rydberg_hamiltonian(
            AtomRegister(self.REGISTER.positions.tolist()),
            0.7 * self.OMEGA,
            [-0.3, 0.8, 0.1],
            c6=1.5 * self.OMEGA,
            phase=1.1,
        )
        gamma = gamma_over_omega * self.OMEGA
        steps = 4000 if gamma > 0 else None
        middle = checked_evolve(first, 0.6, gamma=gamma, n_steps=steps)
        rho = checked_evolve(second, 0.9, gamma=gamma, n_steps=steps, rho0=middle)
        expected = liouvillian_reference(
            second, 0.9, gamma, liouvillian_reference(first, 0.6, gamma)
        )
        assert torch.allclose(rho, expected, rtol=0, atol=2e-6 if gamma > 0 else 1e-12)

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (0.6, 7)])
    def test_two_calls_with_the_same_step_are_one_call(
        self, gamma: float, n_steps: int | None
    ) -> None:
        """
        With the same ``dt``, the half steps where the calls meet merge into
        one full step, so ``S(dt)^7 S(dt)^7 = S(dt)^14`` term by term; without
        dephasing ``U(t₂) U(t₁) = U(t₁ + t₂)`` exactly.
        """
        h = self.hamiltonian([0.5, -1.2, 2.1])
        middle = checked_evolve(h, 0.7, gamma=gamma, n_steps=n_steps)
        chained = checked_evolve(h, 0.7, gamma=gamma, n_steps=n_steps, rho0=middle)
        whole = checked_evolve(h, 1.4, gamma=gamma, n_steps=None if n_steps is None else 14)
        assert torch.allclose(chained, whole, rtol=0, atol=1e-13)

    def test_a_batch_of_initial_states_is_evolved_sample_by_sample(self) -> None:
        h = torch.stack([self.hamiltonian([0.5, -1.2, 2.1]), self.hamiltonian([0.1, 0.2, 0.3])])
        rho0 = torch.stack([random_density_matrix(3, seed=1), random_density_matrix(3, seed=2)])
        rho = checked_evolve(h, 0.8, gamma=0.6, n_steps=9, rho0=rho0)
        for b in range(2):
            alone = checked_evolve(h[b], 0.8, gamma=0.6, n_steps=9, rho0=rho0[b])
            assert torch.allclose(rho[b], alone, rtol=0, atol=1e-14)

    def test_one_initial_state_is_shared_by_a_batch(self) -> None:
        h = torch.stack([self.hamiltonian([0.5, -1.2, 2.1]), self.hamiltonian([0.1, 0.2, 0.3])])
        rho0 = random_density_matrix(3, seed=1)
        rho = checked_evolve(h, 0.8, gamma=0.6, n_steps=9, rho0=rho0)
        for b in range(2):
            alone = checked_evolve(h[b], 0.8, gamma=0.6, n_steps=9, rho0=rho0)
            assert torch.allclose(rho[b], alone, rtol=0, atol=1e-14)

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (0.6, 5)])
    def test_gradients_reach_the_initial_state(self, gamma: float, n_steps: int | None) -> None:
        """
        ``⟨n_i⟩(t)`` is linear in ``ρ(0)``; its derivative along a traceless
        Hermitian direction matches central differences.
        """
        h = self.hamiltonian([0.5, -1.2], n_atoms=2)
        base = random_density_matrix(2, seed=3)
        direction = random_density_matrix(2, seed=4) - torch.eye(4, dtype=C128) / 4

        def features(s: torch.Tensor) -> torch.Tensor:
            return readout(evolve(h, 0.9, gamma=gamma, n_steps=n_steps, rho0=base + s * direction))

        s = torch.zeros((), dtype=F64, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(features, s)
        expected = central_differences(features, s)
        assert torch.allclose(jacobian, expected, rtol=0, atol=1e-9)
        assert jacobian.abs().max().item() > 1e-2

    def test_without_it_the_hamiltonian_alone_fixes_the_shape(self) -> None:
        """``rho0=None`` is the old call: ``|g…g⟩⟨g…g|``, batch from ``hamiltonian``."""
        h = torch.stack([self.hamiltonian([0.5, -1.2, 2.1])] * 2)
        assert checked_evolve(h, 0.3, rho0=None).shape == (2, 8, 8)


class TestInitialStateValidation:
    H = rydberg_hamiltonian(AtomRegister.chain(2, 1.0), 2.0, [0.1, 0.2], c6=1.0)

    def test_rejects_something_that_is_not_a_tensor(self) -> None:
        with pytest.raises(TypeError, match="rho0 must be a torch.Tensor"):
            evolve(self.H, 1.0, rho0=[[1.0, 0.0], [0.0, 0.0]])  # type: ignore[arg-type]

    def test_rejects_an_integer_matrix(self) -> None:
        with pytest.raises(TypeError, match="rho0 must be a real or complex floating"):
            evolve(self.H, 1.0, rho0=torch.eye(4, dtype=torch.int64))

    @pytest.mark.parametrize("shape", [(2, 2), (8, 8), (3, 4, 4), (1, 4, 4), (4,)])
    def test_rejects_a_state_of_another_shape(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="rho0 must have shape"):
            evolve(self.H, 1.0, rho0=torch.zeros(shape, dtype=C128))

    def test_a_batched_state_needs_a_batch_of_hamiltonians_of_its_size(self) -> None:
        batch = torch.stack([self.H, self.H])
        with pytest.raises(ValueError, match=r"rho0 must have shape \(4, 4\) or \(2, 4, 4\)"):
            evolve(batch, 1.0, rho0=torch.zeros(3, 4, 4, dtype=C128))
        with pytest.raises(
            ValueError, match=r"rho0 must have shape \(4, 4\), that of hamiltonian"
        ):
            evolve(self.H, 1.0, rho0=torch.zeros(2, 4, 4, dtype=C128))

    def test_rejects_a_non_finite_state(self) -> None:
        rho0 = torch.eye(4, dtype=C128) / 4
        rho0[1, 2] = math.nan
        with pytest.raises(ValueError, match="rho0 must be finite"):
            evolve(self.H, 1.0, rho0=rho0)

    def test_a_real_initial_state_is_accepted(self) -> None:
        rho0 = torch.eye(4, dtype=F64) / 4
        assert torch.equal(evolve(self.H, 0.4, rho0=rho0), evolve(self.H, 0.4, rho0=rho0.to(C128)))

    def test_rho0_is_keyword_only(self) -> None:
        with pytest.raises(TypeError, match="positional"):
            evolve(self.H, 1.0, 0.5, 10, torch.eye(4))  # type: ignore[misc]


class TestBatch:
    N = 3
    BATCH = 4

    def hamiltonians(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(11)
        delta = torch.rand(self.BATCH, self.N, generator=generator, dtype=F64) * 3
        return rydberg_hamiltonian(AtomRegister.chain(self.N, 1.0), 2.0, delta, c6=2.0)

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (1.5, 40)])
    def test_a_batch_equals_a_loop_over_its_samples(
        self, gamma: float, n_steps: int | None
    ) -> None:
        h = self.hamiltonians()
        rho = checked_evolve(h, 0.9, gamma=gamma, n_steps=n_steps)
        assert rho.shape == (self.BATCH, 8, 8)
        for b in range(self.BATCH):
            single = checked_evolve(h[b], 0.9, gamma=gamma, n_steps=n_steps)
            assert single.shape == (8, 8)
            assert torch.allclose(rho[b], single, rtol=0, atol=1e-13)

    def test_a_batch_of_one_keeps_its_batch_axis(self) -> None:
        h = self.hamiltonians()[:1]
        assert checked_evolve(h, 0.9, gamma=1.5, n_steps=5).shape == (1, 8, 8)

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (1.5, 5)])
    def test_an_empty_batch_comes_back_empty(self, gamma: float, n_steps: int | None) -> None:
        """``rydberg_hamiltonian`` returns ``(0, 2^N, 2^N)`` for no samples; so does this."""
        h = rydberg_hamiltonian(
            AtomRegister.chain(self.N, 1.0), 2.0, torch.zeros(0, self.N, dtype=F64), c6=2.0
        )
        assert h.shape == (0, 8, 8)
        rho = evolve(h, 0.9, gamma=gamma, n_steps=n_steps)
        assert rho.shape == (0, 8, 8)
        assert rho.dtype == C128
        assert readout(rho, pairs=True).shape == (0, 6)

    def test_a_single_precision_hamiltonian_is_evolved_in_double_precision(self) -> None:
        h = self.hamiltonians()[0]
        rho = checked_evolve(h.to(torch.complex64).to(C128), 0.9)
        assert torch.equal(evolve(h.to(torch.complex64), 0.9), rho)

    def test_a_hamiltonian_computed_in_single_precision_is_accepted(self) -> None:
        """
        A basis rotation carried out in ``complex64`` leaves ``H`` Hermitian to
        single-precision rounding only (here ``1e-7`` of its largest entry),
        far above the double-precision bound.
        """
        h = self.hamiltonians()[0].to(torch.complex64)
        generator = torch.Generator().manual_seed(0)
        q, _ = torch.linalg.qr(
            torch.complex(
                torch.randn(8, 8, generator=generator), torch.randn(8, 8, generator=generator)
            )
        )
        rotated = q @ h @ q.mH
        relative = ((rotated - rotated.mH).abs().max() / rotated.abs().max()).item()
        assert 1e-9 < relative < 1e-6
        rho = evolve(rotated, 0.9)
        assert rho.dtype == C128
        # the same matrix in double precision is held to the double-precision bound
        with pytest.raises(ValueError, match="hamiltonian must be Hermitian"):
            evolve(rotated.to(C128), 0.9)

    def test_a_real_symmetric_matrix_is_accepted(self) -> None:
        h = self.hamiltonians()[0]
        assert torch.equal(evolve(h.real, 0.9), evolve(h, 0.9))


def central_differences(
    function: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    ``∂f/∂x_j ≈ (f(x + ε e_j) − f(x − ε e_j)) / (2ε)``, with ``j`` on new trailing axes.

    The truncation error is ``ε² f‴/6``, about ``1e-12`` for the smooth features
    here, and rounding adds about ``1e-16/ε = 1e-10``.
    """
    flat = x.detach().reshape(-1)
    columns = []
    for j in range(flat.numel()):
        step = torch.zeros_like(flat)
        step[j] = eps
        plus = function((flat + step).reshape(x.shape))
        minus = function((flat - step).reshape(x.shape))
        columns.append((plus - minus) / (2 * eps))
    return torch.stack(columns, dim=-1).reshape(*columns[0].shape, *x.shape)


def uncoupled_pair(gamma: float, n_steps: int | None) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    ``⟨n_0⟩, ⟨n_1⟩, ⟨n_0 n_1⟩`` of #505's two atoms without interactions.

    ``H = h_0 ⊗ 1 + 1 ⊗ h_1``, so its energies are the sums ``e_0 + e_1`` of
    single-atom energies.  With equal detunings ``h_0 = h_1`` and the two
    mixed sums coincide: the spectrum is degenerate exactly where the two
    inputs are equal.
    """
    register = AtomRegister.chain(2, 1.0)

    def features(delta: torch.Tensor) -> torch.Tensor:
        h = rydberg_hamiltonian(register, 2.0, delta, c6=1.3, interactions=False)
        return readout(evolve(h, 0.8, gamma=gamma, n_steps=n_steps), pairs=True)

    return features


#: #505's degenerate point; a point ``1e-9`` away from it, whose two mixed
#: energies differ by about ``1e-9`` rad/µs, where a divided difference
#: ``(f(E_m) − f(E_k)) / (E_m − E_k)`` would cancel to a few digits; and a batch
#: holding both and a non-degenerate point.
DEGENERATE_POINTS = {
    "degenerate": [0.4, 0.4],
    "near-degenerate": [0.4, 0.4 + 1e-9],
    "batch": [[0.4, 0.4], [0.4, 0.4 + 1e-9], [0.4, 1.1]],
}
DEPHASING = [(0.0, None), (0.8, 6)]


class TestGradients:
    """
    Derivatives through ``evolve`` in every autograd mode, against central
    finite differences.  At a degenerate spectrum the backward pass of
    ``eigh`` returns a finite but wrong gradient (#505), and so did forward
    mode through it (#674).  ``evolve`` differentiates its propagator in the
    eigenbasis instead, by the Daleckii–Krein formula of the module docstring.
    """

    @pytest.mark.parametrize(("gamma", "n_steps"), DEPHASING)
    @pytest.mark.parametrize("pairs", [False, True])
    def test_gradient_at_a_degenerate_spectrum_matches_finite_differences(
        self, gamma: float, n_steps: int | None, pairs: bool
    ) -> None:
        """Equal detunings without interactions produce a degeneracy (#505)."""
        register = AtomRegister.chain(2, 1.0)

        def features(delta: torch.Tensor) -> torch.Tensor:
            h = rydberg_hamiltonian(register, 2.0, delta, c6=1.3, interactions=False)
            return readout(
                evolve(h, 0.8, gamma=gamma, n_steps=n_steps),
                pairs=pairs,
            )

        delta = torch.tensor([0.4, 0.4], dtype=F64, requires_grad=True)
        assert torch.autograd.gradcheck(features, (delta,), eps=1e-6, atol=1e-7)

    @pytest.mark.parametrize(("gamma", "n_steps"), DEPHASING)
    def test_gradient_of_the_features_matches_finite_differences(
        self, gamma: float, n_steps: int | None
    ) -> None:
        """Interacting atoms with different detunings: a non-degenerate spectrum."""
        register = AtomRegister.chain(2, 1.0)

        def features(delta: torch.Tensor) -> torch.Tensor:
            h = rydberg_hamiltonian(register, 2.0, delta, c6=1.3)
            return readout(evolve(h, 0.8, gamma=gamma, n_steps=n_steps), pairs=True)

        delta = torch.tensor([0.4, 1.1], dtype=F64, requires_grad=True)
        assert torch.autograd.gradcheck(features, (delta,), eps=1e-6, atol=1e-7)

    @pytest.mark.parametrize("point", DEGENERATE_POINTS)
    @pytest.mark.parametrize(("gamma", "n_steps"), DEPHASING)
    @pytest.mark.parametrize(
        "jacobian", [torch.func.jacfwd, torch.func.jacrev], ids=["jacfwd", "jacrev"]
    )
    def test_jacobian_at_a_degenerate_spectrum_matches_finite_differences(
        self,
        jacobian: Callable[..., Callable[[torch.Tensor], torch.Tensor]],
        gamma: float,
        n_steps: int | None,
        point: str,
    ) -> None:
        """``torch.func`` in forward and in reverse mode (#674)."""
        features = uncoupled_pair(gamma, n_steps)
        delta = torch.tensor(DEGENERATE_POINTS[point], dtype=F64)
        expected = central_differences(features, delta)
        torch.testing.assert_close(jacobian(features)(delta), expected, rtol=0, atol=1e-8)

    @pytest.mark.parametrize(
        ("gamma", "n_steps", "value"), [(0.0, None, -0.02279), (0.8, 6, -0.01892)]
    )
    def test_jacfwd_gives_the_finite_differences_of_the_issue(
        self, gamma: float, n_steps: int | None, value: float
    ) -> None:
        """
        ``∂(⟨n_0⟩ + ⟨n_1⟩)/∂Δ_i`` at ``Δ = (0.4, 0.4)``, to the 4 digits #505
        reports.  Through ``eigh``, ``jacfwd`` gave ``[−0.0209, −0.0287]`` and
        ``[+0.0120, −0.0017]`` (#674); the function is symmetric in the two
        atoms, so the two entries must be equal.
        """
        features = uncoupled_pair(gamma, n_steps)

        def total(delta: torch.Tensor) -> torch.Tensor:
            return features(delta)[:2].sum()

        grad = torch.func.jacfwd(total)(torch.tensor([0.4, 0.4], dtype=F64))
        torch.testing.assert_close(grad, torch.full((2,), value, dtype=F64), rtol=0, atol=5e-6)

    @pytest.mark.parametrize("point", ["degenerate", "near-degenerate"])
    @pytest.mark.parametrize(("gamma", "n_steps"), DEPHASING)
    def test_dual_tensors_match_finite_differences(
        self, gamma: float, n_steps: int | None, point: str
    ) -> None:
        """``torch.autograd.forward_ad``, which ``gradcheck`` drives, besides backward."""
        delta = torch.tensor(DEGENERATE_POINTS[point], dtype=F64, requires_grad=True)
        assert torch.autograd.gradcheck(
            uncoupled_pair(gamma, n_steps), (delta,), eps=1e-6, atol=1e-7, check_forward_ad=True
        )

    @pytest.mark.parametrize("point", ["degenerate", "near-degenerate"])
    @pytest.mark.parametrize(("gamma", "n_steps"), DEPHASING)
    def test_second_derivatives_match_finite_differences(
        self, gamma: float, n_steps: int | None, point: str
    ) -> None:
        """Reverse over reverse and forward over reverse, against differences of the gradient."""
        delta = torch.tensor(DEGENERATE_POINTS[point], dtype=F64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            uncoupled_pair(gamma, n_steps), (delta,), eps=1e-6, atol=1e-6, check_fwd_over_rev=True
        )

    @pytest.mark.parametrize(("gamma", "n_steps"), DEPHASING)
    def test_the_hessian_of_torch_func_matches_finite_differences(
        self, gamma: float, n_steps: int | None
    ) -> None:
        """``torch.func.hessian`` (forward over reverse) at #505's degenerate point."""
        features = uncoupled_pair(gamma, n_steps)

        def total(delta: torch.Tensor) -> torch.Tensor:
            return features(delta).sum()

        delta = torch.tensor([0.4, 0.4], dtype=F64)
        expected = central_differences(torch.func.grad(total), delta, eps=1e-5)
        torch.testing.assert_close(torch.func.hessian(total)(delta), expected, rtol=0, atol=1e-8)

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (0.5, 4)])
    def test_the_state_does_not_depend_on_the_autograd_mode(
        self, gamma: float, n_steps: int | None
    ) -> None:
        """
        A strong blockade, ``‖H‖₂ ≈ 6e8`` rad/µs (6 atoms 0.6 µm apart,
        ``DEFAULT_C6``), for ``t = 4`` µs: with ``matrix_exp`` on the gradient
        path, ``ρ`` differed from the ``no_grad`` one by ``3.6e-8`` and its
        trace from 1 by ``7e-8`` (#674).  Every mode runs the same
        eigendecomposition now, so the states are equal bit for bit.
        """
        register = AtomRegister.chain(6, 0.6)

        def state(delta: torch.Tensor) -> torch.Tensor:
            h = rydberg_hamiltonian(register, 2.0, delta, c6=DEFAULT_C6)
            return evolve(h, 4.0, gamma=gamma, n_steps=n_steps)

        delta = torch.linspace(-1.0, 1.0, 6, dtype=F64)
        with torch.no_grad():
            reference = state(delta)
        assert abs(reference.diagonal().sum().item() - 1) < 1e-12
        tangent = torch.ones_like(delta)
        with torch.autograd.forward_ad.dual_level():
            dual = state(torch.autograd.forward_ad.make_dual(delta, tangent))
            dual_primal = torch.autograd.forward_ad.unpack_dual(dual).primal.detach()
        states = {
            "requires_grad": state(delta.clone().requires_grad_(True)).detach(),
            "torch.func.jvp": torch.func.jvp(state, (delta,), (tangent,))[0],
            "torch.func.vjp": torch.func.vjp(state, delta)[0],
            "forward_ad": dual_primal,
        }
        for mode, rho in states.items():
            assert torch.equal(rho, reference), mode

    @pytest.mark.slow
    def test_a_backward_pass_at_max_atoms_costs_about_a_forward_pass(self) -> None:
        """
        The backward pass reuses the forward pass's eigendecomposition and adds
        four ``2^N × 2^N`` matrix products: forward and backward together took
        1.8 to 2.2 times the forward pass alone at 10 atoms (2 and 16 threads).
        With ``matrix_exp`` on the gradient path they took 22 to 45 times as
        long (#674).  The bound of 4 leaves room for a noisy machine.
        """
        register = AtomRegister.chain(MAX_ATOMS, 3.0)
        delta = torch.linspace(-1.0, 1.0, MAX_ATOMS, dtype=F64)

        def seconds(backward: bool) -> float:
            times = []
            for _ in range(3):
                x = delta.clone().requires_grad_(backward)
                start = time.perf_counter()
                h = rydberg_hamiltonian(register, 2.0, x, c6=DEFAULT_C6)
                total = readout(evolve(h, 4.0)).sum()
                if backward:
                    total.backward()
                times.append(time.perf_counter() - start)
            return min(times)

        assert seconds(backward=True) < 4 * seconds(backward=False)


class TestPropagatorDerivative:
    """
    The derivative of ``U = e^(−iHτ)`` that ``evolve`` uses (Daleckii–Krein),
    against ``scipy.linalg.expm_frechet``, which computes the same derivative
    by scaling and squaring, without an eigendecomposition.
    ``expm_frechet(A, E)`` is the derivative of ``e^A`` in direction ``E``, so
    ``d/dε e^(−iτ(H + εX)) = expm_frechet(−iτH, −iτX)``.
    """

    DIM = 8

    @staticmethod
    def jvp(
        function: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, tangent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``torch.func.jvp`` for one tensor in and out: ``(f(x), L(x, tangent))``."""
        output = torch.func.jvp(function, (x,), (tangent,))
        return output[0], output[1]

    def random_complex(self, generator: torch.Generator) -> torch.Tensor:
        shape = (self.DIM, self.DIM)
        return torch.complex(
            torch.randn(shape, generator=generator, dtype=F64),
            torch.randn(shape, generator=generator, dtype=F64),
        )

    def hamiltonian(self, scale: float) -> torch.Tensor:
        """
        Complex Hermitian, with a threefold cluster: two equal energies and one
        ``1e-9`` above them, in a random eigenbasis.
        """
        generator = torch.Generator().manual_seed(3)
        q, _ = torch.linalg.qr(self.random_complex(generator))
        energies = torch.tensor([-1.0, 0.3, 0.3, 0.3 + 1e-9, 1.2, 2.0, 2.5, 4.0], dtype=F64)
        h = (q * (scale * energies).to(C128)) @ q.mH
        return (h + h.mH) / 2

    @pytest.mark.parametrize(("tau", "scale"), [(0.7, 1.0), (300.0, 10.0)])
    def test_forward_and_reverse_mode_match_expm_frechet(self, tau: float, scale: float) -> None:
        """
        A non-Hermitian complex direction: the formula holds for any ``X``.
        ``‖Hτ‖ = 2.8`` and ``1.2e4``.  Both methods round the phases ``E τ``,
        so they agree to about ``ε ‖Hτ‖`` relative to the largest entry
        (measured: ``1e-15`` and ``2.4e-12``); the bound is 10 times that.
        The gradient is checked by the adjoint identity
        ``Re Tr(G† L(H, X)) = Re Tr(grad_H† X)`` with ``L`` from scipy, a sum
        of ``8 × 8`` products with entries of ``G`` of order 1, hence
        ``64 · atol`` there.
        """
        linalg = pytest.importorskip("scipy.linalg")
        h = self.hamiltonian(scale)
        generator = torch.Generator().manual_seed(4)
        direction, cotangent = self.random_complex(generator), self.random_complex(generator)
        _, frechet = linalg.expm_frechet(-1j * tau * h.numpy(), -1j * tau * direction.numpy())
        expected = torch.as_tensor(frechet)
        relative = 10 * torch.finfo(F64).eps * tau * torch.linalg.matrix_norm(h, ord=2).item()
        atol = relative * expected.abs().max().item()

        def propagator(matrix: torch.Tensor) -> torch.Tensor:
            return lindblad._Propagator.apply(matrix, tau, False)[0]

        u, tangent = self.jvp(propagator, h, direction)
        unitary = torch.as_tensor(linalg.expm(-1j * tau * h.numpy()))
        assert torch.allclose(u, unitary, rtol=0, atol=relative)
        assert torch.allclose(tangent, expected, rtol=0, atol=atol)
        (grad,) = torch.func.vjp(propagator, h)[1](cotangent)
        assert torch.sum(grad.conj() * direction).real.item() == pytest.approx(
            torch.sum(cotangent.conj() * expected).real.item(), rel=0, abs=self.DIM**2 * atol
        )

    def test_the_first_column_is_the_first_column_with_its_derivatives(self) -> None:
        """``U e_0``, which ``gamma = 0`` evolves, and its derivatives, from the full ones."""
        h = self.hamiltonian(1.0)
        generator = torch.Generator().manual_seed(5)
        direction = self.random_complex(generator)
        cotangent = self.random_complex(generator)[:, :1]

        def column(matrix: torch.Tensor) -> torch.Tensor:
            return lindblad._Propagator.apply(matrix, 0.7, True)[0]

        def full(matrix: torch.Tensor) -> torch.Tensor:
            return lindblad._Propagator.apply(matrix, 0.7, False)[0]

        u, tangent = self.jvp(full, h, direction)
        u0, tangent0 = self.jvp(column, h, direction)
        assert u0.shape == (self.DIM, 1)
        torch.testing.assert_close(u0, u[:, :1], rtol=0, atol=1e-15)
        torch.testing.assert_close(tangent0, tangent[:, :1], rtol=0, atol=1e-15)
        padded = torch.nn.functional.pad(cotangent, (0, self.DIM - 1))
        (grad0,) = torch.func.vjp(column, h)[1](cotangent)
        (grad,) = torch.func.vjp(full, h)[1](padded)
        torch.testing.assert_close(grad0, grad, rtol=0, atol=1e-15)

    def test_the_divided_differences_at_a_gap_of_1e_9(self) -> None:
        """
        ``(e^(−iE_mτ) − e^(−iE_kτ)) / (E_m − E_k)`` computed with ``mpmath`` at
        50 digits, where float64 subtraction would keep about 7 of them.
        """
        mpmath = pytest.importorskip("mpmath")  # a dependency of torch, through sympy
        energies = [0.3, 0.3 + 1e-9, 0.3, -2.0]
        tau = 0.7
        actual = lindblad._divided_differences(torch.tensor(energies, dtype=F64), tau)
        with mpmath.workdps(50):
            for m, k in itertools.product(range(4), repeat=2):
                e_m, e_k = mpmath.mpf(energies[m]), mpmath.mpf(energies[k])
                t = mpmath.mpf(tau)
                if e_m == e_k:
                    exact = -1j * t * mpmath.expj(-e_m * t)
                else:
                    exact = (mpmath.expj(-e_m * t) - mpmath.expj(-e_k * t)) / (e_m - e_k)
                assert abs(complex(actual[m, k]) - complex(exact)) < 1e-15


class TestValidation:
    H = rydberg_hamiltonian(AtomRegister.chain(2, 1.0), 2.0, [0.1, 0.2], c6=1.0)

    def test_rejects_something_that_is_not_a_tensor(self) -> None:
        with pytest.raises(TypeError, match="hamiltonian must be a torch.Tensor"):
            evolve([[0.0, 1.0], [1.0, 0.0]], 1.0)  # type: ignore[arg-type]

    def test_rejects_an_integer_or_boolean_matrix(self) -> None:
        with pytest.raises(TypeError, match="hamiltonian must be a real or complex floating"):
            evolve(torch.eye(2, dtype=torch.int64), 1.0)

    @pytest.mark.parametrize("shape", [(), (4,), (3, 3), (4, 2), (2, 2, 4, 4), (1, 1)])
    def test_rejects_a_matrix_of_the_wrong_shape(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"hamiltonian must have shape \(2\^n, 2\^n\)"):
            evolve(torch.zeros(shape, dtype=C128), 1.0)

    def test_rejects_more_atoms_than_the_dense_cap(self) -> None:
        dim = 2 ** (MAX_ATOMS + 1)
        with pytest.raises(ValueError, match=r"for 1 to 10 atoms; got shape \(2048, 2048\)"):
            evolve(torch.zeros(dim, dim, dtype=torch.complex64), 1.0)

    def test_rejects_a_matrix_that_is_not_hermitian(self) -> None:
        """``eigh`` reads one triangle only and would silently evolve another Hamiltonian."""
        h = self.H.clone()
        h[0, 1] = h[0, 1] + 0.5
        with pytest.raises(ValueError, match="hamiltonian must be Hermitian"):
            evolve(h, 1.0)

    def test_a_large_sample_does_not_hide_another_samples_asymmetry(self) -> None:
        """
        The bound is relative to each sample's own largest entry.  Two atoms
        1.5 µm apart have ``V = 4.8 × 10⁵``: against that scale an asymmetry of
        ``1e-8`` in another sample would pass, and a batch would not equal a
        loop over its samples.
        """
        strong = rydberg_hamiltonian(AtomRegister.chain(2, 1.5), 2.0, [0.1, 0.2], c6=DEFAULT_C6)
        assert strong.abs().max().item() == pytest.approx(4.76e5, rel=1e-2)
        for asymmetry, large in ((1e-8, strong), (0.5, self.H * 1e13)):
            bad = self.H.clone()
            bad[0, 1] = bad[0, 1] + asymmetry
            assert asymmetry < 1e-12 * large.abs().max().item()
            with pytest.raises(ValueError, match="hamiltonian must be Hermitian"):
                evolve(bad, 1.0)
            with pytest.raises(ValueError, match=r"\|H - H†\| entry of sample 1 is"):
                evolve(torch.stack([large, bad]), 1.0)

    def test_single_precision_does_not_excuse_a_real_asymmetry(self) -> None:
        """Largest entry 1, bound ``1.2e-5``: ten times that is refused."""
        h = self.H.to(torch.complex64)
        h[0, 1] = h[0, 1] + 1e-4
        with pytest.raises(ValueError, match="hamiltonian must be Hermitian"):
            evolve(h, 1.0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_gets_the_single_precision_bound(self, dtype: torch.dtype) -> None:
        """100 ε of ``bfloat16`` is 0.78: a drive entry doubled in one triangle would pass."""
        h = self.H.real.to(dtype)
        evolve(h, 1.0)
        h[0, 1] = h[0, 1] + 0.0625  # exact in both dtypes
        with pytest.raises(ValueError, match="hamiltonian must be Hermitian"):
            evolve(h, 1.0)

    def test_rejects_a_non_finite_matrix(self) -> None:
        h = self.H.clone()
        h[1, 1] = math.nan
        with pytest.raises(ValueError, match="hamiltonian must be finite"):
            evolve(h, 1.0)

    @pytest.mark.parametrize("t", [-0.1, math.nan, math.inf, "1.0", True, 10**400])
    def test_rejects_a_bad_time(self, t: object) -> None:
        with pytest.raises(ValueError, match="t must be a finite number >= 0"):
            evolve(self.H, t)  # type: ignore[arg-type]

    @pytest.mark.parametrize("gamma", [-0.1, math.nan, math.inf, None, 10**400])
    def test_rejects_a_bad_rate(self, gamma: object) -> None:
        with pytest.raises(ValueError, match="gamma must be a finite number >= 0"):
            evolve(self.H, 1.0, gamma=gamma, n_steps=10)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("values", "kind", "shape"),
        [
            (torch.tensor([0.5, 0.5], dtype=torch.float64), "tensor", (2,)),
            (torch.full((2, 1), 0.5), "tensor", (2, 1)),
            (np.full(2, 0.5), "NumPy array", (2,)),
        ],
        ids=["tensor (2,)", "tensor (2, 1)", "ndarray (2,)"],
    )
    @pytest.mark.parametrize("name", ["t", "gamma"])
    def test_a_time_or_rate_of_more_than_one_element_is_refused(
        self, name: str, values: object, kind: str, shape: tuple[int, ...]
    ) -> None:
        """
        ``t`` and ``gamma`` are shared by the batch (#530): one value per
        sample is not read as such, whatever the batch size, even when every
        element is the same valid number.  The message names the shape, not
        ``.item()``, which would fail on more than one element.
        """
        batch = torch.stack([self.H, self.H])
        args = {"t": 1.0, "gamma": 0.5, name: values}
        message = f"{name} must be a single finite number >= 0, not a {kind} of shape {shape}."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            evolve(batch, args["t"], gamma=args["gamma"], n_steps=10)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("make", "kind"),
        [
            (lambda: torch.tensor(0.5, dtype=torch.float64), "tensor"),
            (lambda: torch.tensor(0.5, dtype=torch.float64, requires_grad=True), "tensor"),
            (lambda: torch.tensor([0.5], dtype=torch.float64), "tensor"),
            (lambda: np.array(0.5), "NumPy array"),
        ],
        ids=["tensor ()", "tensor () requiring grad", "tensor (1,)", "ndarray ()"],
    )
    @pytest.mark.parametrize("name", ["t", "gamma"])
    def test_a_one_element_tensor_or_array_is_refused_naming_item(
        self, name: str, make: Callable[[], object], kind: str
    ) -> None:
        """
        Numbers only (#530, option A): ``float()`` would take a one-element
        tensor and drop its graph without a word, so it is refused, whether it
        requires grad or not, with a message naming ``.item()``.  What
        ``.item()`` returns is then accepted and evolves as the number does.
        """
        value = make()
        args: dict[str, object] = {"t": 0.5, "gamma": 0.5, name: value}
        message = (
            f"{name} must be a finite number >= 0, not a {kind}; got {value!r}. "
            f"Pass {name}.item() to use its value: no derivative with respect to "
            f"{name} is taken."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            evolve(self.H, args["t"], gamma=args["gamma"], n_steps=10)  # type: ignore[arg-type]
        args[name] = value.item()  # type: ignore[attr-defined]
        rho = evolve(self.H, args["t"], gamma=args["gamma"], n_steps=10)  # type: ignore[arg-type]
        assert torch.equal(rho, evolve(self.H, 0.5, gamma=0.5, n_steps=10))

    @pytest.mark.parametrize(
        ("make", "kind"),
        [
            (lambda: torch.tensor(math.nan, dtype=torch.float64), "tensor"),
            (lambda: torch.tensor(-1.0, dtype=torch.float64), "tensor"),
            (lambda: torch.tensor([math.inf], dtype=torch.float64), "tensor"),
            (lambda: torch.tensor(0.5 + 0j, dtype=torch.complex128), "tensor"),
            (lambda: torch.tensor(True), "tensor"),
            (lambda: torch.tensor(0.5, dtype=torch.float64, device="meta"), "tensor"),
            (lambda: np.array(math.nan), "NumPy array"),
        ],
        ids=["nan", "-1", "inf (1,)", "complex", "bool", "meta", "ndarray nan"],
    )
    @pytest.mark.parametrize("name", ["t", "gamma"])
    def test_a_one_element_tensor_or_array_of_a_bad_value_is_refused_without_item(
        self, name: str, make: Callable[[], object], kind: str
    ) -> None:
        """
        ``.item()`` is named only when its value would be accepted (#530):
        for a NaN, a negative, an infinite, a complex or a boolean element,
        or a meta tensor without a value, following the hint would only be
        refused again, so the message stops after the value.
        """
        value = make()
        args: dict[str, object] = {"t": 0.5, "gamma": 0.5, name: value}
        message = f"{name} must be a finite number >= 0, not a {kind}; got {value!r}."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            evolve(self.H, args["t"], gamma=args["gamma"], n_steps=10)  # type: ignore[arg-type]

    @pytest.mark.parametrize("scalar", [np.float64, np.float32, np.int64])
    def test_numpy_scalars_are_numbers(self, scalar: Callable[[float], object]) -> None:
        """
        A NumPy scalar is a ``numbers.Real`` and stays accepted: one resonant
        atom ends at ``⟨n⟩ = sin²(Ωt/2)``, and ``gamma`` gives the state of
        the same Python float.
        """
        one = rydberg_hamiltonian(ONE_ATOM, 2.0, [0.0], c6=1.0)
        t = scalar(1)
        population = readout(evolve(one, t)).item()  # type: ignore[arg-type]
        assert population == pytest.approx(math.sin(2.0 * 1 / 2) ** 2, abs=1e-14)
        damped = evolve(one, 1.0, gamma=scalar(2), n_steps=20)  # type: ignore[arg-type]
        assert torch.equal(damped, evolve(one, 1.0, gamma=2.0, n_steps=20))

    def test_a_missing_step_count_is_reported_before_the_eigendecomposition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cheapest argument error does not wait for the most expensive operation."""

        def no_eigh(*args: object, **kwargs: object) -> None:
            raise AssertionError("eigh ran before the arguments were checked")

        monkeypatch.setattr(torch.linalg, "eigh", no_eigh)
        with pytest.raises(ValueError, match="n_steps is required when gamma > 0"):
            evolve(self.H, 1.0, gamma=0.5)

    def test_dephasing_needs_a_step_count(self) -> None:
        """No default: the step the result is accurate at depends on γ and on ``H``."""
        with pytest.raises(ValueError, match="n_steps is required when gamma > 0"):
            evolve(self.H, 1.0, gamma=0.5)

    @pytest.mark.parametrize("n_steps", [0, -3, 2.0, True, "10"])
    @pytest.mark.parametrize("gamma", [0.0, 0.5])
    def test_rejects_a_bad_step_count_with_and_without_dephasing(
        self, gamma: float, n_steps: object
    ) -> None:
        with pytest.raises(ValueError, match="n_steps must be an integer >= 1"):
            evolve(self.H, 1.0, gamma=gamma, n_steps=n_steps)  # type: ignore[arg-type]

    def test_gamma_and_n_steps_are_keyword_only(self) -> None:
        with pytest.raises(TypeError, match="positional"):
            evolve(self.H, 1.0, 0.5, 10)  # type: ignore[misc]
