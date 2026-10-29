"""
tests/test_rydberg_lindblad.py
==============================
``evolve`` and ``readout`` (#497) against the closed-form limits of
``docs/rydberg-model.md``: the detuned and the damped Rabi oscillation of one
atom, the product state without interactions, the blockade of two atoms, the
maximally mixed state under strong dephasing, and the order of the splitting.

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
import time
from collections.abc import Callable

import pytest
import torch

from hqnn_forge.rydberg import (
    DEFAULT_C6,
    MAX_ATOMS,
    AtomRegister,
    evolve,
    lindblad,
    readout,
    rydberg_hamiltonian,
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
    # −4e-16), not as a random walk.
    steps_taken = (n_steps or 0) if gamma > 0 else 0
    drift = 1e-12 + 1e-15 * steps_taken
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
        near 1 either way.  The range over the whole parameter domain is the
        sweep's to check (#627).
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
