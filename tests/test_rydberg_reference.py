"""
tests/test_rydberg_reference.py
===============================
``hqnn_forge.rydberg.evolve`` and ``readout`` against QuTiP's ``mesolve``
(#498), for 2, 3 and 4 interacting, dephased atoms with random parameters.

The limits in ``tests/test_rydberg_lindblad.py`` have one or two atoms or a
special regime.  A wrong factor in the dephasing rate or a transposed tensor
factor still gives a valid density matrix, so the full dynamics are compared
here with an implementation that shares no code with the library: QuTiP
integrates the Lindblad equation with an adaptive ODE solver, without an
eigendecomposition, a splitting or the Hamming-distance form of the
dissipator.

The reference is built independently
------------------------------------
Nothing on the QuTiP side comes from ``hqnn_forge``: not the matrix of
``rydberg_hamiltonian``, not the couplings of ``AtomRegister``, and ``C6`` is
typed in.  :func:`reference` assembles, from ``qutip.tensor``, ``qutip.sigmax``
and ``qutip.num``, the model of ``docs/rydberg-model.md``::

    H   = Σ_i Ω/2 · X_i  −  Σ_i Δ_i n_i  +  Σ_{i<j} C6 / (|i − j| a)⁶ · n_i n_j
    L_i = √γ n_i                              (none for γ = 0)
    ρ(0) = |g…g⟩⟨g…g|

in rad/µs, µm and µs, for a chain of spacing ``a`` along one axis.

Conventions of the reference, each asserted in ``TestReferenceConventions``:

* ``qutip.basis(2, 0)`` is ``|g⟩`` and ``qutip.num(2) = diag(0, 1) = |r⟩⟨r|``;
* ``qutip.tensor([A, B, …])`` puts its first argument leftmost, so atom ``i``
  is tensor factor ``i`` from the left and atom 0 the most significant bit of
  the index, the ordering of :mod:`hqnn_forge.rydberg.hamiltonian`;
* ``mesolve`` takes collapse operators ``L`` with the rate inside,
  ``L ρ L† − ½ {L† L, ρ}``, so ``√γ n`` is the operator of the model page and a
  single-atom coherence decays at ``γ/2``.

Tolerances
----------
Both sides are approximate, and the tolerance of a comparison is the sum of
the two errors, each stated here and checked:

* **mesolve.**  ``MESOLVE_ATOL`` and ``MESOLVE_RTOL`` bound the *local* error
  of each step of its integrator, not the error of the result, which
  accumulates over the steps.  ``MESOLVE_ERROR`` is the allowance for the
  result.  It is measured where the library is exact: without dephasing
  ``evolve`` takes no steps, and the two then differ by at most 1.3e-11 over
  the cases below (QuTiP 5.3.1), about a hundredth of the allowance.  One
  dephased atom is within 8e-13 of its closed form.
* **The splitting.**  With dephasing ``evolve`` takes ``N_STEPS`` Strang steps
  of ``dt = t / N_STEPS``.  Its error has even powers of ``dt`` only (the step
  is symmetric), ``ρ_n − ρ = C/n² + O(1/n⁴)`` for ``n`` steps, so two runs
  give it without any reference::

      ρ_{n/2} − ρ_n = 3C/n²        ⇒        |ρ_n − ρ| = |ρ_{n/2} − ρ_n| / 3

  The tolerance takes twice that estimate for ``n = N_STEPS``.  So that the
  solver does not set its own tolerance freely, the estimate must stay below
  ``SPLITTING_CAP``: the draws below keep ``Ω dt ≤ 1.6e-3`` and
  ``γ dt ≤ 3.2e-3``, and the estimate is between 2.8e-9 and 1.3e-8.  A step
  that is not second order, or a dissipator that converges to another
  equation, cannot hide behind it: the first breaks the cap or the
  comparison, the second leaves a difference of the order of ``γ t`` whatever
  the step.

An element of ρ is therefore compared to ``MESOLVE_ERROR = 1e-9`` without
dephasing, and with it to at most ``MESOLVE_ERROR + 2 · SPLITTING_CAP = 1.01e-7``,
in the cases below to between 7e-9 and 2.6e-8, of which the measured
difference is about half.  A readout value is a sum of ``2^(N−1)``
(``⟨n_i⟩``) or ``2^(N−2)`` (``⟨n_i n_j⟩``) diagonal elements and gets that
many times the tolerance of one element.
"""

from __future__ import annotations

import functools
import math
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch

from hqnn_forge.rydberg import AtomRegister, evolve, readout, rydberg_hamiltonian

# From the dev extra; the lowest-floors CI job installs its floors by name.
# No may_skip: QuTiP has wheels for every interpreter of the CI matrix (3.11
# to 3.14), so a skip in CI means the reference stopped running, which
# HQNN_FORGE_FAIL_ON_SKIP=1 turns into a failure.
qutip = pytest.importorskip("qutip")

#: ``C6`` of two ⁸⁷Rb atoms in ``70S₁/₂`` in rad/µs · µm⁶, typed in from
#: ``docs/rydberg-model.md`` (``2π × 862 690 MHz µm⁶``), not imported.
C6 = 5_420_441.0

#: Local error tolerances of mesolve's integrator, per step.
MESOLVE_ATOL = 1e-14
MESOLVE_RTOL = 1e-12
#: Allowed error of an element of mesolve's ρ(t), accumulated over its steps.
MESOLVE_ERROR = 1e-9

#: Strang steps of the dephased runs; the error estimate also runs half of them.
N_STEPS = 4000
#: Largest splitting error estimate the comparison accepts, per element of ρ.
SPLITTING_CAP = 5e-8

ATOMS = (2, 3, 4)
SEEDS = (0, 1, 2)


class Case(NamedTuple):
    """One random array and pulse, in rad/µs, µm and µs."""

    n_atoms: int
    omega: float
    delta: tuple[float, ...]
    spacing: float
    gamma: float
    t: float
    interactions: bool


def draw_case(n_atoms: int, seed: int, *, dephasing: bool, interactions: bool) -> Case:
    """
    A case drawn in the dimensionless groups of the model page.

    * ``Ω`` between half and twice the reference ``4π rad/µs``;
    * ``Δ_i/Ω`` in ``[−√3, √3]``, one per atom and of either sign (the encoding
      of the page uses ``[0, √3]``; the sign is drawn so that the sign
      convention of Δ is compared too);
    * the spacing between 0.8 and 1.4 blockade radii ``R_b = (C6/Ω)^(1/6)``,
      which is ``V/Ω = (R_b/a)⁶`` from 0.13 to 3.8;
    * ``γ/Ω`` in ``[0.1, 2]``, or 0;
    * ``ΩT`` in ``[1, 2π]``.

    The four variants of one ``(n_atoms, seed)`` share every number but the
    two switched off, so they differ by exactly that term.
    """
    rng = np.random.default_rng([n_atoms, seed])
    omega = 4 * math.pi * rng.uniform(0.5, 2.0)
    delta = omega * rng.uniform(-math.sqrt(3), math.sqrt(3), size=n_atoms)
    spacing = (C6 / omega) ** (1 / 6) * rng.uniform(0.8, 1.4)
    gamma = omega * rng.uniform(0.1, 2.0)
    t = rng.uniform(1.0, 2 * math.pi) / omega
    return Case(
        n_atoms=n_atoms,
        omega=float(omega),
        delta=tuple(float(d) for d in delta),
        spacing=float(spacing),
        gamma=float(gamma) if dephasing else 0.0,
        t=float(t),
        interactions=interactions,
    )


def pair_order(n_atoms: int) -> list[tuple[int, int]]:
    """``(0,1), (0,2), …, (N−2,N−1)``: the order of the model page, typed as two loops."""
    return [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]


def on_atom(operator: Any, atom: int, n_atoms: int) -> Any:
    """``1 ⊗ … ⊗ operator ⊗ … ⊗ 1`` with ``operator`` at factor ``atom`` from the left."""
    return qutip.tensor([operator if j == atom else qutip.qeye(2) for j in range(n_atoms)])


def mesolve_options() -> dict[str, Any]:
    return {
        "atol": MESOLVE_ATOL,
        "rtol": MESOLVE_RTOL,
        # internal steps allowed between two output times; the default 2500
        # is not enough at these tolerances
        "nsteps": 1_000_000,
        "store_final_state": True,
        "progress_bar": "",
    }


class Reference(NamedTuple):
    rho: np.ndarray
    #: ``⟨n_0⟩, …, ⟨n_{N−1}⟩``, then ``⟨n_i n_j⟩`` in :func:`pair_order`
    features: np.ndarray


@functools.cache
def reference(case: Case) -> Reference:
    """``ρ(t)`` and the readout from ``qutip.mesolve``; see the module docstring."""
    n = case.n_atoms
    number = [on_atom(qutip.num(2), i, n) for i in range(n)]
    hamiltonian = 0
    for i in range(n):
        hamiltonian += case.omega / 2 * on_atom(qutip.sigmax(), i, n)
        hamiltonian -= case.delta[i] * number[i]
    if case.interactions:
        for i, j in pair_order(n):
            distance = (j - i) * case.spacing  # a chain along one axis
            hamiltonian += C6 / distance**6 * number[i] * number[j]
    collapse = [math.sqrt(case.gamma) * number[i] for i in range(n)] if case.gamma > 0 else []
    ground = qutip.tensor([qutip.basis(2, 0)] * n)
    result = qutip.mesolve(
        hamiltonian,
        # a density matrix, so that mesolve also returns one without collapse operators
        qutip.ket2dm(ground),
        [0.0, case.t],
        c_ops=collapse,
        e_ops=number + [number[i] * number[j] for i, j in pair_order(n)],
        options=mesolve_options(),
    )
    features = np.array([np.real(values[-1]) for values in result.expect])
    return Reference(rho=result.final_state.full(), features=features)


class Solved(NamedTuple):
    rho: torch.Tensor
    #: estimated error of an element of ``rho``: 0 without dephasing (exact)
    splitting_error: float


@functools.cache
def solved(case: Case) -> Solved:
    """``evolve`` on the case, with the step-doubling estimate of its splitting error."""
    hamiltonian = rydberg_hamiltonian(
        AtomRegister.chain(case.n_atoms, case.spacing),
        case.omega,
        list(case.delta),
        c6=C6,
        interactions=case.interactions,
    )
    if case.gamma == 0:
        return Solved(rho=evolve(hamiltonian, case.t), splitting_error=0.0)
    rho = evolve(hamiltonian, case.t, gamma=case.gamma, n_steps=N_STEPS)
    coarse = evolve(hamiltonian, case.t, gamma=case.gamma, n_steps=N_STEPS // 2)
    return Solved(rho=rho, splitting_error=float((coarse - rho).abs().max()) / 3)


def element_tolerance(case: Case) -> float:
    """Allowed difference of one element of ρ: see *Tolerances* in the module docstring."""
    estimate = solved(case).splitting_error
    assert estimate <= SPLITTING_CAP
    return MESOLVE_ERROR + 2 * estimate


all_cases = pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            draw_case(n, seed, dephasing=dephasing, interactions=interactions),
            id=f"{n} atoms, seed {seed}, "
            f"{'dephased' if dephasing else 'no dephasing'}, "
            f"{'interacting' if interactions else 'interactions off'}",
        )
        for n in ATOMS
        for seed in SEEDS
        for dephasing in (True, False)
        for interactions in (True, False)
    ],
)


class TestReferenceConventions:
    """What the QuTiP side means by an atom, a state and a rate, against typed-in values."""

    def test_the_first_tensor_factor_is_atom_0_the_most_significant_bit(self) -> None:
        """Order ``|gg⟩, |gr⟩, |rg⟩, |rr⟩``: ``n_0 = diag(0, 0, 1, 1)``, ``n_1 = diag(0, 1, 0, 1)``."""
        assert on_atom(qutip.num(2), 0, 2).full().tolist() == np.diag([0, 0, 1, 1]).tolist()
        assert on_atom(qutip.num(2), 1, 2).full().tolist() == np.diag([0, 1, 0, 1]).tolist()

    def test_the_initial_state_is_basis_index_0(self) -> None:
        ground = qutip.ket2dm(qutip.tensor([qutip.basis(2, 0)] * 3)).full()
        expected = np.zeros((8, 8))
        expected[0, 0] = 1.0
        assert ground.tolist() == expected.tolist()

    def test_the_drive_term_is_omega_over_2_times_x(self) -> None:
        """One atom, ``Ω = 3``, ``Δ = 0.5``: ``[[0, 1.5], [1.5, −0.5]]`` in ``(|g⟩, |r⟩)``."""
        matrix = (3.0 / 2 * qutip.sigmax() - 0.5 * qutip.num(2)).full()
        assert matrix.tolist() == [[0.0, 1.5], [1.5, -0.5]]

    @pytest.mark.parametrize("gamma_over_omega", [0.3, 1.5])
    def test_the_collapse_operator_gives_the_damped_rabi_oscillation_of_the_model_page(
        self, gamma_over_omega: float
    ) -> None:
        """
        One resonant atom, ``γ < 4Ω``::

            ⟨n⟩(t) = ½ − ½ e^(−γt/4) [cos λt + γ/(4λ) · sin λt],    λ = √(Ω² − γ²/16)

        With a collapse operator of rate ``2γ`` or ``γ/2`` (``√γ Z``, or the
        rate left outside the square root) the envelope would differ.
        """
        omega, t = 2.0, 2.2
        gamma = gamma_over_omega * omega
        case = Case(1, omega, (0.0,), 1.0, gamma, t, interactions=False)
        lam = math.sqrt(omega**2 - gamma**2 / 16)
        bracket = math.cos(lam * t) + gamma / (4 * lam) * math.sin(lam * t)
        expected = 0.5 - 0.5 * math.exp(-gamma * t / 4) * bracket
        assert reference(case).features[0] == pytest.approx(expected, rel=0, abs=MESOLVE_ERROR)
        assert reference(case).rho[1, 1].real == pytest.approx(expected, rel=0, abs=MESOLVE_ERROR)


class TestAgainstMesolve:
    """2, 3 and 4 atoms, with and without dephasing and interactions."""

    @all_cases
    def test_density_matrix(self, case: Case) -> None:
        """Every element of ``ρ(t)``, real and imaginary part."""
        difference = np.abs(solved(case).rho.numpy() - reference(case).rho).max()
        assert difference <= element_tolerance(case)

    @all_cases
    def test_readout(self, case: Case) -> None:
        """``⟨n_i⟩`` and ``⟨n_i n_j⟩`` against the expectation values mesolve returns."""
        n = case.n_atoms
        features = readout(solved(case).rho, pairs=True).numpy()
        expected = reference(case).features
        assert features.shape == expected.shape == (n + n * (n - 1) // 2,)
        # ⟨n_i⟩ sums the 2^(N−1) diagonal elements with b_i = 1, ⟨n_i n_j⟩ 2^(N−2).
        terms = np.array([2 ** (n - 1)] * n + [2 ** (n - 2)] * (n * (n - 1) // 2))
        assert np.all(np.abs(features - expected) <= terms * element_tolerance(case))

    @all_cases
    def test_the_cases_are_far_from_trivial(self, case: Case) -> None:
        """
        The comparison has something to tell apart: with the atoms of the
        reference in reverse order, or its dephasing rate doubled, ρ moves by
        thousands of tolerances.
        """
        rho = solved(case).rho.numpy()
        mirrored = reference(case._replace(delta=case.delta[::-1])).rho
        assert np.abs(rho - mirrored).max() > 1e4 * element_tolerance(case)
        if case.gamma > 0:
            doubled = reference(case._replace(gamma=2 * case.gamma)).rho
            assert np.abs(rho - doubled).max() > 1e4 * element_tolerance(case)
