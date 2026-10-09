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

import pytest
import torch

from hqnn_forge.rydberg import (
    DEFAULT_C6,
    MAX_ATOMS,
    AtomRegister,
    evolve,
    readout,
    rydberg_hamiltonian,
)

C128 = torch.complex128
F64 = torch.float64
ONE_ATOM = AtomRegister([[0.0, 0.0]])


def checked_evolve(
    hamiltonian: torch.Tensor, t: float, *, gamma: float = 0.0, n_steps: int | None = None
) -> torch.Tensor:
    """``evolve``, with every returned state checked to be a density matrix."""
    rho = evolve(hamiltonian, t, gamma=gamma, n_steps=n_steps)
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


def liouvillian_reference(h: torch.Tensor, t: float, gamma: float) -> torch.Tensor:
    """
    ``ρ(t) = exp(𝓛 t) ρ(0)`` from the full generator, for one Hamiltonian.

    With ρ flattened row by row, ``vec(A ρ B) = (A ⊗ Bᵀ) vec(ρ)``, so::

        𝓛 = −i (H ⊗ 1 − 1 ⊗ Hᵀ) + γ Σ_i ( n_i ⊗ n_i − ½ n_i ⊗ 1 − ½ 1 ⊗ n_i )

    which is ``−i[H, ρ] + Σ_i (L_i ρ L_i† − ½ {L_i† L_i, ρ})`` for
    ``L_i = √γ n_i`` (``n_i`` is real and diagonal, so ``n_iᵀ = n_i`` and
    ``n_i² = n_i``).  ``ρ(0)`` is all atoms in ``|g⟩``.
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
    rho0 = torch.zeros(dim * dim, dtype=C128)
    rho0[0] = 1.0
    return (torch.linalg.matrix_exp(generator * t) @ rho0).reshape(dim, dim)


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
            features[:, :n_atoms], torch.full_like(features[:, :n_atoms], 0.5), atol=1e-10
        )
        assert torch.allclose(
            features[:, n_atoms:], torch.full_like(features[:, n_atoms:], 0.25), atol=1e-10
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

    def test_a_real_symmetric_matrix_is_accepted(self) -> None:
        h = self.hamiltonians()[0]
        assert torch.equal(evolve(h.real, 0.9), evolve(h, 0.9))


class TestGradients:
    """
    Nothing in the solver blocks autograd (#420): no in-place operation on the
    path.  At a degenerate spectrum ``eigh`` returns a wrong gradient (#505);
    the case here is not degenerate.
    """

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (0.8, 6)])
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

    def test_rejects_a_non_finite_matrix(self) -> None:
        h = self.H.clone()
        h[1, 1] = math.nan
        with pytest.raises(ValueError, match="hamiltonian must be finite"):
            evolve(h, 1.0)

    @pytest.mark.parametrize("t", [-0.1, math.nan, math.inf, "1.0", True])
    def test_rejects_a_bad_time(self, t: object) -> None:
        with pytest.raises(ValueError, match="t must be a finite number >= 0"):
            evolve(self.H, t)  # type: ignore[arg-type]

    @pytest.mark.parametrize("gamma", [-0.1, math.nan, math.inf, None])
    def test_rejects_a_bad_rate(self, gamma: object) -> None:
        with pytest.raises(ValueError, match="gamma must be a finite number >= 0"):
            evolve(self.H, 1.0, gamma=gamma, n_steps=10)  # type: ignore[arg-type]

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
