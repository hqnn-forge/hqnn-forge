"""
tests/test_rydberg_hamiltonian.py
=================================
``rydberg_hamiltonian`` (#496) against the closed-form limits of
``docs/rydberg-model.md``: the single-atom matrix, the doubly excited
diagonal entry, the Kronecker sum without interactions, the ``1/r^6`` tail,
the basis ordering, and the batched call.

Every expected matrix is built another way than the function builds it: typed
in by hand, assembled from ``torch.kron`` of single-atom factors, or returned
by ``qml.matrix`` for the same operator written in PennyLane.
"""

from __future__ import annotations

import math

import pennylane as qml
import pytest
import torch

from hqnn_forge.rydberg import DEFAULT_C6, MAX_ATOMS, AtomRegister, rydberg_hamiltonian

C128 = torch.complex128


def single_atom(omega: float, delta: float) -> torch.Tensor:
    """``[[0, Ω/2], [Ω/2, −Δ]]`` in the basis ``(|g⟩, |r⟩)``, typed in."""
    return torch.tensor([[0.0, omega / 2], [omega / 2, -delta]], dtype=C128)


def kronecker_sum(factors: list[torch.Tensor]) -> torch.Tensor:
    """``Σ_i 1 ⊗ … ⊗ h_i ⊗ … ⊗ 1`` with factor ``i`` at position ``i`` from the left."""
    n = len(factors)
    total = torch.zeros(2**n, 2**n, dtype=C128)
    for i, factor in enumerate(factors):
        term = torch.ones(1, 1, dtype=C128)
        for j in range(n):
            term = torch.kron(term, factor if j == i else torch.eye(2, dtype=C128))
        total = total + term
    return total


def pennylane_matrix(
    register: AtomRegister,
    omega: list[float],
    delta: list[float],
    c6: float,
    interactions: bool,
) -> torch.Tensor:
    """The same operator written with PennyLane gates, atom ``i`` on wire ``i``."""
    n = register.n_atoms
    positions = register.positions.tolist()
    op = qml.s_prod(0.0, qml.Identity(0))
    for i in range(n):
        number = qml.Projector([1], wires=i)  # n_i = |r⟩⟨r|_i with |r⟩ = |1⟩
        op = op + qml.s_prod(omega[i] / 2, qml.PauliX(i)) - qml.s_prod(delta[i], number)
    if interactions:
        for i in range(n):
            for j in range(i + 1, n):
                v = c6 / math.dist(positions[i], positions[j]) ** 6
                pair = qml.Projector([1], wires=i) @ qml.Projector([1], wires=j)
                op = op + qml.s_prod(v, pair)
    return torch.as_tensor(qml.matrix(op, wire_order=range(n)), dtype=C128)


class TestOneAtom:
    @pytest.mark.parametrize(("omega", "delta"), [(2.0, 0.0), (4 * math.pi, 7.5), (1.3, -0.4)])
    def test_matrix_is_the_one_of_the_model_page(self, omega: float, delta: float) -> None:
        h = rydberg_hamiltonian(AtomRegister([[0.0, 0.0]]), omega, [delta], c6=DEFAULT_C6)
        assert h.dtype == C128
        assert h.shape == (2, 2)
        assert torch.equal(h, single_atom(omega, delta))

    def test_omega_is_the_rabi_frequency_not_half_of_it(self) -> None:
        """The splitting on resonance is Ω: eigenvalues ``±Ω/2``, not ``±Ω``."""
        h = rydberg_hamiltonian(AtomRegister([[0.0, 0.0]]), 3.0, [0.0], c6=DEFAULT_C6)
        assert torch.linalg.eigvalsh(h).tolist() == pytest.approx([-1.5, 1.5], rel=1e-14)

    def test_positive_detuning_lowers_the_rydberg_state(self) -> None:
        h = rydberg_hamiltonian(AtomRegister([[0.0, 0.0]]), 0.0, [2.5], c6=DEFAULT_C6)
        assert h.tolist() == [[0j, 0j], [0j, -2.5 + 0j]]


class TestTwoAtoms:
    OMEGA = 2.0
    DELTA = [0.7, -1.9]
    C6 = 1.0e6
    R = 5.0

    def test_doubly_excited_entry_is_minus_both_detunings_plus_the_interaction(self) -> None:
        """``⟨rr|H|rr⟩ = −Δ_0 − Δ_1 + C6 / r^6`` at index 3 = ``|rr⟩``."""
        register = AtomRegister([[1.0, 2.0], [1.0 + 3.0, 2.0 + 4.0]])  # 3-4-5: r = 5 µm
        h = rydberg_hamiltonian(register, self.OMEGA, self.DELTA, c6=self.C6)
        expected = -0.7 + 1.9 + 1.0e6 / 15625
        assert h[3, 3].real.item() == pytest.approx(expected, rel=1e-14)
        assert h[3, 3].imag.item() == 0.0

    def test_whole_matrix_by_hand(self) -> None:
        """Order ``|gg⟩, |gr⟩, |rg⟩, |rr⟩``: atom 0 is the left factor."""
        register = AtomRegister.chain(2, self.R)
        h = rydberg_hamiltonian(register, self.OMEGA, self.DELTA, c6=self.C6)
        v = 1.0e6 / 15625  # = 64
        expected = torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.9, 0.0, 1.0],  # |gr⟩: −Δ_1
                [1.0, 0.0, -0.7, 1.0],  # |rg⟩: −Δ_0
                [0.0, 1.0, 1.0, 1.2 + v],  # |rr⟩: −Δ_0 − Δ_1 + V
            ],
            dtype=C128,
        )
        assert torch.allclose(h, expected, rtol=1e-14, atol=0)

    def test_without_interactions_it_is_the_kronecker_sum(self) -> None:
        register = AtomRegister.chain(2, self.R)
        h = rydberg_hamiltonian(register, self.OMEGA, self.DELTA, c6=self.C6, interactions=False)
        expected = kronecker_sum([single_atom(self.OMEGA, d) for d in self.DELTA])
        assert torch.allclose(h, expected, rtol=1e-14, atol=0)

    def test_interactions_false_drops_the_pair_term_and_nothing_else(self) -> None:
        """The difference is ``V`` on ``|rr⟩⟨rr|`` and zero elsewhere."""
        register = AtomRegister.chain(2, self.R)
        on = rydberg_hamiltonian(register, self.OMEGA, self.DELTA, c6=self.C6)
        off = rydberg_hamiltonian(register, self.OMEGA, self.DELTA, c6=self.C6, interactions=False)
        expected = torch.zeros(4, 4, dtype=C128)
        expected[3, 3] = 1.0e6 / 15625
        assert torch.allclose(on - off, expected, rtol=1e-12, atol=0)

    def test_a_negative_c6_lowers_the_doubly_excited_state(self) -> None:
        h = rydberg_hamiltonian(AtomRegister.chain(2, 2.0), 1.0, [0.0, 0.0], c6=-64.0)
        assert h[3, 3] == -1.0


class TestThreeAtomChain:
    def test_next_nearest_coupling_is_one_64th_of_the_nearest(self) -> None:
        """Read off the diagonal at ``Δ = 0``: ``|rrg⟩`` = 6, ``|grr⟩`` = 3, ``|rgr⟩`` = 5."""
        h = rydberg_hamiltonian(AtomRegister.chain(3, 5.0), 2.0, [0.0, 0.0, 0.0], c6=1.0e6)
        diagonal = h.diagonal().real
        nearest = 1.0e6 / 5.0**6
        assert diagonal[0b110].item() == pytest.approx(nearest, rel=1e-14)
        assert diagonal[0b011].item() == pytest.approx(nearest, rel=1e-14)
        assert (diagonal[0b101] / diagonal[0b110]).item() == pytest.approx(1 / 64, rel=1e-13)
        # all three excited: both nearest pairs and the next-nearest one
        assert diagonal[0b111].item() == pytest.approx(nearest * (2 + 1 / 64), rel=1e-14)
        # at most one excitation: no pair, no interaction
        assert diagonal[[0b000, 0b001, 0b010, 0b100]].tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_without_interactions_it_is_the_kronecker_sum(self) -> None:
        omega, delta = [1.0, 2.0, 3.0], [0.5, -0.25, 1.5]
        h = rydberg_hamiltonian(
            AtomRegister.chain(3, 5.0), omega, delta, c6=1.0e6, interactions=False
        )
        expected = kronecker_sum([single_atom(o, d) for o, d in zip(omega, delta, strict=True)])
        assert torch.allclose(h, expected, rtol=1e-14, atol=0)


class TestBasisOrdering:
    @pytest.mark.parametrize("n_atoms", [1, 2, 3, 4])
    def test_a_detuning_on_one_atom_moves_exactly_the_entries_with_that_bit_set(
        self, n_atoms: int
    ) -> None:
        """Atom ``i`` is bit ``N − 1 − i`` of the index: atom 0 the most significant."""
        register = AtomRegister.chain(n_atoms, 6.0)
        zero = [0.0] * n_atoms
        base = rydberg_hamiltonian(register, 1.5, zero, c6=DEFAULT_C6)
        for atom in range(n_atoms):
            delta = list(zero)
            delta[atom] = 0.75
            moved = rydberg_hamiltonian(register, 1.5, delta, c6=DEFAULT_C6) - base
            expected = torch.zeros(2**n_atoms, dtype=torch.float64)
            for index in range(2**n_atoms):
                label = format(index, f"0{n_atoms}b")  # |b_0 b_1 … b_{N−1}⟩ as text
                if label[atom] == "1":
                    expected[index] = -0.75
            # −0.75 is added to entries that also hold V, so equal to rounding only
            assert torch.allclose(moved.diagonal().real, expected, rtol=0, atol=1e-12)
            off_diagonal = moved - torch.diag(moved.diagonal())
            assert torch.equal(off_diagonal, torch.zeros_like(off_diagonal))

    def test_two_atom_number_operator_of_the_model_page(self) -> None:
        """``n_0 = diag(0, 0, 1, 1)`` and ``n_1 = diag(0, 1, 0, 1)``: ``H = −n_i`` at Ω = 0."""
        register = AtomRegister.chain(2, 6.0)
        n0 = -rydberg_hamiltonian(register, 0.0, [1.0, 0.0], c6=DEFAULT_C6, interactions=False)
        n1 = -rydberg_hamiltonian(register, 0.0, [0.0, 1.0], c6=DEFAULT_C6, interactions=False)
        assert n0.diagonal().real.tolist() == [0.0, 0.0, 1.0, 1.0]
        assert n1.diagonal().real.tolist() == [0.0, 1.0, 0.0, 1.0]

    def test_a_drive_on_one_atom_flips_that_tensor_factor(self) -> None:
        """Ω on atom 0 of 3 only: ``(Ω/2) X ⊗ 1 ⊗ 1``.  An end atom: the middle one
        would sit on the same factor with the bit order reversed."""
        h = rydberg_hamiltonian(
            AtomRegister.chain(3, 6.0),
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            c6=DEFAULT_C6,
            interactions=False,
        )
        x = torch.tensor([[0.0, 1.5], [1.5, 0.0]], dtype=C128)
        eye = torch.eye(2, dtype=C128)
        assert torch.equal(h, torch.kron(torch.kron(x, eye), eye))

    @pytest.mark.parametrize("interactions", [True, False])
    @pytest.mark.parametrize("shape", ["chain", "ring"])
    @pytest.mark.parametrize("n_atoms", [1, 2, 3, 4, 5])
    def test_matches_pennylanes_matrix_in_its_wire_order(
        self, n_atoms: int, shape: str, interactions: bool
    ) -> None:
        """Per-atom Ω and Δ, all different, so a permuted atom would show."""
        register = getattr(AtomRegister, shape)(n_atoms, 5.5)
        omega = [1.0 + 0.37 * i for i in range(n_atoms)]
        delta = [0.9 - 0.61 * i for i in range(n_atoms)]
        h = rydberg_hamiltonian(register, omega, delta, c6=1.0e5, interactions=interactions)
        expected = pennylane_matrix(register, omega, delta, 1.0e5, interactions)
        assert h.shape == (2**n_atoms, 2**n_atoms)
        assert torch.allclose(h, expected, rtol=1e-12, atol=1e-12)

    def test_an_irregular_register_matches_pennylane(self) -> None:
        """No two distances equal, so every ``V_ij`` has to sit on its own pair."""
        register = AtomRegister([[0.0, 0.0], [4.1, 0.3], [1.2, 6.5], [9.0, 5.0]])
        omega = [2.0, 2.5, 3.0, 3.5]
        delta = [0.3, -1.1, 2.2, 0.8]
        h = rydberg_hamiltonian(register, omega, delta, c6=DEFAULT_C6)
        expected = pennylane_matrix(register, omega, delta, DEFAULT_C6, True)
        assert torch.allclose(h, expected, rtol=1e-12, atol=1e-12)


class TestHermiticity:
    @pytest.mark.parametrize("interactions", [True, False])
    @pytest.mark.parametrize("n_atoms", [1, 2, 3, 6])
    def test_equal_to_its_conjugate_transpose_exactly(
        self, n_atoms: int, interactions: bool
    ) -> None:
        generator = torch.Generator().manual_seed(n_atoms)
        omega = torch.rand(4, n_atoms, generator=generator, dtype=torch.float64) * 10
        delta = torch.randn(4, n_atoms, generator=generator, dtype=torch.float64) * 10
        h = rydberg_hamiltonian(
            AtomRegister.ring(n_atoms, 5.0), omega, delta, c6=DEFAULT_C6, interactions=interactions
        )
        assert torch.equal(h, h.mH)
        assert torch.equal(h.imag, torch.zeros_like(h.imag))  # φ = 0: a real matrix

    def test_trace_is_the_sum_of_the_diagonal_terms(self) -> None:
        """``Tr n_i = 2^(N−1)``, ``Tr n_i n_j = 2^(N−2)``, ``Tr X_i = 0``."""
        register = AtomRegister.chain(3, 5.0)
        delta = [0.5, -0.25, 1.5]
        h = rydberg_hamiltonian(register, 2.0, delta, c6=1.0e6)
        v = 1.0e6 / 5.0**6
        expected = -4 * sum(delta) + 2 * (v + v + v / 64)
        assert torch.trace(h).real.item() == pytest.approx(expected, rel=1e-13)


class TestOmega:
    def test_a_scalar_drives_every_atom(self) -> None:
        register = AtomRegister.chain(3, 5.0)
        delta = [0.5, -0.25, 1.5]
        scalar = rydberg_hamiltonian(register, 2.0, delta, c6=1.0e6)
        per_atom = rydberg_hamiltonian(register, [2.0, 2.0, 2.0], delta, c6=1.0e6)
        zero_dim = rydberg_hamiltonian(register, torch.tensor(2.0), delta, c6=1.0e6)
        assert torch.equal(scalar, per_atom)
        assert torch.equal(scalar, zero_dim)

    def test_python_floats_keep_double_precision(self) -> None:
        """``torch.as_tensor`` alone would read them as float32: 0.1 → 0.10000000149."""
        h = rydberg_hamiltonian(AtomRegister([[0.0, 0.0]]), 0.2, [0.1], c6=DEFAULT_C6)
        assert h.real.tolist() == [[0.0, 0.1], [0.1, -0.1]]

    def test_float32_and_integer_inputs_give_the_float64_matrix(self) -> None:
        register = AtomRegister.chain(2, 5.0)
        h = rydberg_hamiltonian(register, 2, torch.tensor([1, -2]), c6=1.0e6)
        expected = rydberg_hamiltonian(register, 2.0, [1.0, -2.0], c6=1.0e6)
        assert h.dtype == C128
        assert torch.equal(h, expected)
        h32 = rydberg_hamiltonian(register, 2.0, torch.tensor([0.5, 0.25]), c6=1.0e6)
        assert h32.dtype == C128
        assert h32[1, 1] == -0.25


class TestBatch:
    N = 3
    BATCH = 5

    def setup_method(self) -> None:
        generator = torch.Generator().manual_seed(7)
        self.register = AtomRegister.chain(self.N, 5.0)
        self.delta = torch.randn(self.BATCH, self.N, generator=generator, dtype=torch.float64)
        self.omega = torch.rand(self.BATCH, self.N, generator=generator, dtype=torch.float64) + 1

    def loop(self, omega: object, delta: torch.Tensor, **kwargs: bool) -> torch.Tensor:
        """One unbatched call per sample; ``omega`` is indexed where it is batched."""
        rows = []
        for b in range(delta.shape[0]):
            omega_b = omega[b] if isinstance(omega, torch.Tensor) and omega.ndim == 2 else omega
            rows.append(
                rydberg_hamiltonian(self.register, omega_b, delta[b], c6=1.0e6, **kwargs)  # type: ignore[arg-type]
            )
        return torch.stack(rows)

    @pytest.mark.parametrize("interactions", [True, False])
    def test_batched_delta_with_a_scalar_omega_equals_a_loop(self, interactions: bool) -> None:
        h = rydberg_hamiltonian(
            self.register, 2.0, self.delta, c6=1.0e6, interactions=interactions
        )
        assert h.shape == (self.BATCH, 8, 8)
        assert torch.equal(h, self.loop(2.0, self.delta, interactions=interactions))

    def test_batched_delta_with_a_per_atom_omega_equals_a_loop(self) -> None:
        omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        h = rydberg_hamiltonian(self.register, omega, self.delta, c6=1.0e6)
        assert torch.equal(h, self.loop(omega, self.delta))

    def test_both_batched_equals_a_loop(self) -> None:
        h = rydberg_hamiltonian(self.register, self.omega, self.delta, c6=1.0e6)
        assert torch.equal(h, self.loop(self.omega, self.delta))

    def test_one_global_omega_per_sample_has_shape_batch_by_1(self) -> None:
        omega = self.omega[:, :1]  # (batch, 1)
        h = rydberg_hamiltonian(self.register, omega, self.delta, c6=1.0e6)
        expected = torch.stack(
            [
                rydberg_hamiltonian(self.register, omega[b, 0].item(), self.delta[b], c6=1.0e6)
                for b in range(self.BATCH)
            ]
        )
        assert torch.equal(h, expected)

    def test_a_batched_omega_alone_batches_the_result(self) -> None:
        delta = self.delta[0]
        h = rydberg_hamiltonian(self.register, self.omega, delta, c6=1.0e6)
        expected = torch.stack(
            [
                rydberg_hamiltonian(self.register, self.omega[b], delta, c6=1.0e6)
                for b in range(self.BATCH)
            ]
        )
        assert h.shape == (self.BATCH, 8, 8)
        assert torch.equal(h, expected)

    def test_a_batch_of_one_keeps_its_batch_axis(self) -> None:
        h = rydberg_hamiltonian(self.register, 2.0, self.delta[:1], c6=1.0e6)
        assert h.shape == (1, 8, 8)
        assert torch.equal(h[0], rydberg_hamiltonian(self.register, 2.0, self.delta[0], c6=1.0e6))

    def test_a_one_dimensional_omega_is_per_atom_even_when_the_batch_has_that_size(self) -> None:
        """Batch of 3 on 3 atoms: ``omega`` of shape ``(3,)`` is read along the atoms."""
        omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        delta = self.delta[:3]
        h = rydberg_hamiltonian(self.register, omega, delta, c6=1.0e6)
        # sample 0, atom 2 (least significant bit): ⟨000|H|001⟩ = Ω_2 / 2
        assert h[0, 0b000, 0b001] == 1.5
        assert h[2, 0b000, 0b100] == 0.5


class TestGradients:
    def test_derivatives_are_the_operators_of_the_hamiltonian(self) -> None:
        """``∂H/∂Δ_i = −n_i`` and ``∂H/∂Ω = ½ Σ_i X_i``, here contracted with a fixed matrix."""
        register = AtomRegister.chain(2, 5.0)
        omega = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        delta = torch.tensor([0.3, -0.8], dtype=torch.float64, requires_grad=True)
        probe = torch.arange(16, dtype=torch.float64).reshape(4, 4)
        h = rydberg_hamiltonian(register, omega, delta, c6=1.0e6)
        (h.real * probe).sum().backward()
        # n_0 = diag(0,0,1,1), n_1 = diag(0,1,0,1); probe diagonal is 0, 5, 10, 15
        assert delta.grad is not None
        assert omega.grad is not None
        assert delta.grad.tolist() == [-(10.0 + 15.0), -(5.0 + 15.0)]
        # X_0 + X_1 has ones at (0,1),(0,2),(1,3),(2,3) and transposes: probe sums to 60
        assert omega.grad.item() == pytest.approx(0.5 * 60.0, rel=1e-14)


class TestAtomCount:
    def test_the_cap_is_the_documented_ten_atoms(self) -> None:
        assert MAX_ATOMS == 10

    def test_one_atom_over_the_cap_is_an_error_before_anything_is_allocated(self) -> None:
        register = AtomRegister.chain(MAX_ATOMS + 1, 5.0)
        with pytest.raises(ValueError, match=r"at most 10 atoms.*got 11"):
            rydberg_hamiltonian(register, 1.0, [0.0] * (MAX_ATOMS + 1), c6=DEFAULT_C6)

    def test_a_register_itself_may_be_larger(self) -> None:
        """The cap belongs to the dense matrix; a register of 30 atoms is fine (#418)."""
        assert AtomRegister.chain(30, 5.0).interaction_matrix(DEFAULT_C6).shape == (30, 30)

    def test_the_cap_itself_builds_and_keeps_the_ordering(self) -> None:
        n = MAX_ATOMS
        delta = [0.0] * n
        delta[0] = 1.0  # atom 0: the most significant bit, the upper half of the diagonal
        h = rydberg_hamiltonian(
            AtomRegister.chain(n, 5.0), 2.0, delta, c6=DEFAULT_C6, interactions=False
        )
        assert h.shape == (2**n, 2**n)
        diagonal = h.diagonal().real
        assert torch.equal(
            diagonal[: 2 ** (n - 1)], torch.zeros(2 ** (n - 1), dtype=diagonal.dtype)
        )
        assert torch.equal(
            diagonal[2 ** (n - 1) :], -torch.ones(2 ** (n - 1), dtype=diagonal.dtype)
        )
        # each row holds one entry Ω/2 per atom
        assert h.real.sum(dim=1)[0].item() == pytest.approx(n * 1.0, rel=1e-14)


class TestValidation:
    REGISTER = AtomRegister.chain(3, 5.0)

    def test_rejects_something_that_is_not_a_register(self) -> None:
        with pytest.raises(TypeError, match="register must be an AtomRegister"):
            rydberg_hamiltonian([[0.0, 0.0]], 1.0, [0.0], c6=DEFAULT_C6)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "delta",
        [
            0.5,  # a scalar: delta is per atom
            [0.0, 0.0],  # one atom short
            [0.0, 0.0, 0.0, 0.0],  # one too many
            [[0.0, 0.0]],  # batched, one atom short
            [[[0.0, 0.0, 0.0]]],  # two batch axes
        ],
    )
    def test_rejects_a_delta_of_the_wrong_shape(self, delta: object) -> None:
        with pytest.raises(ValueError, match=r"delta must have shape \(3,\) or \(batch, 3\)"):
            rydberg_hamiltonian(self.REGISTER, 1.0, delta, c6=DEFAULT_C6)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "omega",
        [
            [1.0, 2.0],  # neither one value nor one per atom
            [[1.0, 2.0, 3.0]] * 4,  # batch of 4 against a batch of 2
            [[[1.0]]],  # two batch axes
            [1.0],  # one-dimensional, so per-atom, and two atoms short
            [[1.0, 2.0, 3.0]],  # batch of 1 against a batch of 2: not broadcast
            [[1.0]],  # the same for one global omega per sample
        ],
    )
    def test_rejects_an_omega_of_the_wrong_shape(self, omega: object) -> None:
        delta = torch.zeros(2, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"omega must be a scalar.*\(batch, 1\)"):
            rydberg_hamiltonian(self.REGISTER, omega, delta, c6=DEFAULT_C6)  # type: ignore[arg-type]

    @pytest.mark.parametrize("delta_shape", [(3,), (1, 3)])
    def test_a_one_long_omega_is_not_a_global_one_even_for_a_batch_of_1(
        self, delta_shape: tuple[int, ...]
    ) -> None:
        """``(batch,)`` with ``batch = 1`` on 3 atoms: refused like any other batch size."""
        delta = torch.zeros(delta_shape, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"omega must be a scalar.*\(batch, 1\)"):
            rydberg_hamiltonian(self.REGISTER, torch.ones(1), delta, c6=DEFAULT_C6)

    def test_a_batch_of_1_delta_is_not_broadcast_to_the_batch_of_omega(self) -> None:
        delta = torch.zeros(1, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"omega must be a scalar.*\(batch, 1\)"):
            rydberg_hamiltonian(self.REGISTER, torch.ones(4, 1), delta, c6=DEFAULT_C6)

    def test_on_one_atom_a_one_long_omega_is_the_per_atom_shape(self) -> None:
        """``N = 1``: ``(1,)`` is ``(N,)`` and ``(batch, 1)`` is ``(batch, N)``; ``(batch,)`` raises."""
        one = AtomRegister([[0.0, 0.0]])
        delta = torch.tensor([[0.5], [0.25]], dtype=torch.float64)
        shared = rydberg_hamiltonian(one, [2.0], delta, c6=DEFAULT_C6)
        assert shared.real.tolist() == [[[0.0, 1.0], [1.0, -0.5]], [[0.0, 1.0], [1.0, -0.25]]]
        per_sample = rydberg_hamiltonian(one, [[2.0], [4.0]], delta, c6=DEFAULT_C6)
        assert per_sample.real.tolist() == [[[0.0, 1.0], [1.0, -0.5]], [[0.0, 2.0], [2.0, -0.25]]]
        with pytest.raises(ValueError, match=r"omega must be a scalar.*\(batch, 1\)"):
            rydberg_hamiltonian(one, [2.0, 4.0], delta, c6=DEFAULT_C6)

    def test_a_batch_long_omega_is_not_taken_for_one_value_per_sample(self) -> None:
        """Shape ``(batch,)`` is refused where it cannot be per-atom; ``(batch, 1)`` is meant."""
        delta = torch.zeros(4, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match=r"omega must be a scalar.*\(batch, 1\)"):
            rydberg_hamiltonian(self.REGISTER, torch.ones(4), delta, c6=DEFAULT_C6)

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_rejects_non_finite_pulse_parameters(self, bad: float) -> None:
        with pytest.raises(ValueError, match="omega must be finite"):
            rydberg_hamiltonian(self.REGISTER, bad, [0.0, 0.0, 0.0], c6=DEFAULT_C6)
        with pytest.raises(ValueError, match="delta must be finite"):
            rydberg_hamiltonian(self.REGISTER, 1.0, [0.0, bad, 0.0], c6=DEFAULT_C6)

    def test_rejects_complex_pulse_parameters(self) -> None:
        with pytest.raises(TypeError, match="omega must hold real numbers"):
            rydberg_hamiltonian(self.REGISTER, torch.tensor(1j), [0.0, 0.0, 0.0], c6=DEFAULT_C6)
        with pytest.raises(TypeError, match="delta must hold real numbers"):
            rydberg_hamiltonian(
                self.REGISTER, 1.0, torch.zeros(3, dtype=torch.complex128), c6=DEFAULT_C6
            )

    @pytest.mark.parametrize("interactions", [True, False])
    def test_rejects_a_bad_c6_with_and_without_interactions(self, interactions: bool) -> None:
        with pytest.raises(ValueError, match="c6 must be a finite number"):
            rydberg_hamiltonian(
                self.REGISTER, 1.0, [0.0, 0.0, 0.0], c6=math.nan, interactions=interactions
            )

    def test_c6_and_interactions_are_keyword_only(self) -> None:
        with pytest.raises(TypeError, match="positional"):
            rydberg_hamiltonian(self.REGISTER, 1.0, [0.0, 0.0, 0.0], DEFAULT_C6)  # type: ignore[misc]
