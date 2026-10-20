"""
tests/test_neutral_atom_layer.py
================================
The blockade interaction graph of an ``AtomRegister``, its parallel gate
schedule and the ``neutral_atom_layer`` primitive (#418).

Every expected value is obtained another way than the code computes it: the
edges, triangles and layers of hand-placed registers are worked out by hand
in each test's docstring, and the unitary of one layer is built with NumPy
from the diagonal phase of a product of CZ (and CCZ) gates and the 2x2 ``Rot``
matrices, not through PennyLane or the code under test.
"""

from __future__ import annotations

import math
from functools import reduce

import numpy as np
import pennylane as qml
import pytest
import torch

from hqnn_forge.circuits import blockade_schedule, neutral_atom_layer
from hqnn_forge.rydberg import AtomRegister

SIDE = 5.0  # µm, the spacing of every hand-placed register below

#: Three atoms on a line, ``SIDE`` apart: 0 — 1 — 2.
LINE_3 = AtomRegister([[0.0, 0.0], [SIDE, 0.0], [2 * SIDE, 0.0]])
#: An equilateral triangle of side ``SIDE``.
TRIANGLE = AtomRegister([[0.0, 0.0], [SIDE, 0.0], [SIDE / 2, SIDE * math.sqrt(3) / 2]])
#: A square of side ``SIDE``, counterclockwise from the origin: 0 (0,0), 1 (5,0), 2 (5,5),
#: 3 (0,5).  Sides 0-1, 1-2, 2-3, 3-0 are 5 µm, diagonals 0-2, 1-3 are 5√2 ≈ 7.07 µm.
SQUARE = AtomRegister([[0.0, 0.0], [SIDE, 0.0], [SIDE, SIDE], [0.0, SIDE]])
#: Between the square's side (5) and its diagonal (7.07).
R_SIDE = 6.0


# ---------------------------------------------------------------------------
# Hand-built reference unitaries (NumPy only)
# ---------------------------------------------------------------------------


def _rot(phi: float, theta: float, omega: float) -> np.ndarray:
    """``Rot(φ, θ, ω) = RZ(ω) · RY(θ) · RZ(φ)``, with ``RZ(a) = diag(e^{-ia/2}, e^{ia/2})``."""

    def rz(a: float) -> np.ndarray:
        return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])

    c, s = math.cos(theta / 2), math.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    return rz(omega) @ ry @ rz(phi)


def _diagonal_phase(
    n: int, pairs: list[tuple[int, int]], triples: list[tuple[int, int, int]]
) -> np.ndarray:
    """
    Diagonal of ``∏ CZ_e ∏ CCZ_t`` on ``n`` qubits, PennyLane's ordering (wire 0 the
    most significant bit): basis state ``b`` picks up ``(-1)^{b_i b_j}`` per pair and
    ``(-1)^{b_i b_j b_k}`` per triple.
    """
    diagonal = np.ones(2**n, dtype=complex)
    for index in range(2**n):
        bits = [(index >> (n - 1 - wire)) & 1 for wire in range(n)]
        for i, j in pairs:
            diagonal[index] *= (-1) ** (bits[i] * bits[j])
        for i, j, k in triples:
            diagonal[index] *= (-1) ** (bits[i] * bits[j] * bits[k])
    return diagonal


def _layer_unitary(
    weights: np.ndarray,
    pairs: list[tuple[int, int]],
    triples: list[tuple[int, int, int]] = [],  # noqa: B006 - never mutated
) -> np.ndarray:
    """``(⊗_i Rot_i) · diag(∏ CZ ∏ CCZ)``: the entangler first, then the rotations."""
    n = weights.shape[0]
    rotations = reduce(np.kron, [_rot(*weights[i]) for i in range(n)])
    return rotations @ np.diag(_diagonal_phase(n, pairs, triples))


def _primitive_unitary(
    weights: torch.Tensor, register: AtomRegister, radius: float, *, ccz: bool = False
) -> np.ndarray:
    n = register.n_atoms
    return np.asarray(
        qml.matrix(
            lambda w: neutral_atom_layer(w, register, radius, ccz=ccz),
            wire_order=range(n),
        )(weights)
    )


def _weights(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=g, dtype=torch.float64) * 2 * math.pi


# ---------------------------------------------------------------------------
# Interaction graph
# ---------------------------------------------------------------------------


class TestInteractionGraph:
    def test_line(self) -> None:
        """Neighbours are 5 µm apart, atoms 0 and 2 are 10 µm: only 0-1 and 1-2 at R_b = 6."""
        assert LINE_3.interaction_graph(R_SIDE) == [(0, 1), (1, 2)]
        assert LINE_3.blockade_triangles(R_SIDE) == []

    def test_line_with_next_nearest_inside_the_radius(self) -> None:
        """At R_b = 11 > 10 every pair interacts, and 0-1-2 is a blockade triangle."""
        assert LINE_3.interaction_graph(11.0) == [(0, 1), (0, 2), (1, 2)]
        assert LINE_3.blockade_triangles(11.0) == [(0, 1, 2)]

    def test_triangle(self) -> None:
        """All three sides are 5 µm: the complete graph on 3 atoms and one triangle."""
        assert TRIANGLE.interaction_graph(R_SIDE) == [(0, 1), (0, 2), (1, 2)]
        assert TRIANGLE.blockade_triangles(R_SIDE) == [(0, 1, 2)]

    def test_square_between_side_and_diagonal(self) -> None:
        """Sides 5 < 6 < 7.07 diagonals: the 4-cycle 0-1-2-3-0, no diagonal, no triangle."""
        assert SQUARE.interaction_graph(R_SIDE) == [(0, 1), (0, 3), (1, 2), (2, 3)]
        assert SQUARE.blockade_triangles(R_SIDE) == []

    def test_square_beyond_the_diagonal(self) -> None:
        """At R_b = 8 > 7.07 the diagonals join: K4, whose 4 triangles leave out one atom each."""
        assert SQUARE.interaction_graph(8.0) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert SQUARE.blockade_triangles(8.0) == [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

    def test_a_pair_exactly_one_radius_apart_does_not_interact(self) -> None:
        """
        The convention is strict: ``|r_i − r_j| < R_b``.  At exactly ``R_b`` the
        coupling equals Ω, which is not a blockade.  25 = 5² is exact in float64, so
        the comparison of squared distances meets the boundary exactly.
        """
        assert LINE_3.interaction_graph(SIDE) == []
        assert LINE_3.interaction_graph(math.nextafter(SIDE, math.inf)) == [(0, 1), (1, 2)]
        assert LINE_3.interaction_graph(math.nextafter(SIDE, 0.0)) == []

    def test_the_boundary_agrees_with_the_coupling(self) -> None:
        """``d < R_b`` exactly when ``V = C6 / d⁶ > Ω`` with ``R_b = (C6/Ω)^(1/6)``."""
        c6, omega = 64.0, 1.0  # R_b = 2
        radius = AtomRegister.blockade_radius(c6, omega)
        for spacing, inside in ((1.9, True), (2.1, False)):
            register = AtomRegister.chain(2, spacing)
            v = register.interaction_matrix(c6)[0, 1].item()
            assert (v > omega) is inside
            assert (register.interaction_graph(radius) == [(0, 1)]) is inside

    def test_does_not_depend_on_where_the_register_sits(self) -> None:
        moved = AtomRegister(SQUARE.positions + torch.tensor([100.0, -40.0], dtype=torch.float64))
        assert moved.interaction_graph(R_SIDE) == SQUARE.interaction_graph(R_SIDE)

    @pytest.mark.parametrize("radius", [0.0, -1.0, math.nan, math.inf, True, "6"])
    def test_rejects_a_bad_radius(self, radius: object) -> None:
        with pytest.raises(ValueError, match="blockade_radius must be a finite number > 0"):
            SQUARE.interaction_graph(radius)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="blockade_radius must be a finite number > 0"):
            SQUARE.blockade_triangles(radius)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Parallel schedule
# ---------------------------------------------------------------------------


class TestBlockadeSchedule:
    def test_line_of_three(self) -> None:
        """0-1 and 1-2 share atom 1, so they cannot run together: two layers."""
        assert blockade_schedule(LINE_3, R_SIDE) == [[(0, 1)], [(1, 2)]]

    def test_line_of_five_runs_the_far_ends_in_parallel(self) -> None:
        """
        Edges 0-1, 1-2, 2-3, 3-4.  0-1 opens layer 0; 1-2 shares atom 1 (layer 1);
        2-3 has atom 2 5 µm from atom 1 of 0-1 and shares atom 2 with 1-2 (layer 2);
        3-4 is 10 µm from 0-1 at its closest (atoms 1 and 3), so it joins layer 0.
        """
        line = AtomRegister.chain(5, SIDE)
        assert blockade_schedule(line, R_SIDE) == [[(0, 1), (3, 4)], [(1, 2)], [(2, 3)]]

    def test_an_idle_atom_inside_the_radius_does_not_block_a_gate(self) -> None:
        """
        Line of four at R_b = 6: 0-1 and 2-3 are disjoint, but atom 2 is 5 µm from
        atom 1, so they cannot share a layer; 1-2 shares atoms with both.  Three layers,
        even though atom 3 sits idle next to 1-2 in layer 1 and atom 0 next to 2-3.
        """
        line = AtomRegister.chain(4, SIDE)
        assert blockade_schedule(line, R_SIDE) == [[(0, 1)], [(1, 2)], [(2, 3)]]

    def test_triangle(self) -> None:
        """Every pair of the triangle's edges shares an atom: one CZ per layer."""
        assert blockade_schedule(TRIANGLE, R_SIDE) == [[(0, 1)], [(0, 2)], [(1, 2)]]

    def test_triangle_with_ccz(self) -> None:
        """The CCZ uses every atom, so it follows the three CZ layers alone."""
        assert blockade_schedule(TRIANGLE, R_SIDE, ccz=True) == [
            [(0, 1)],
            [(0, 2)],
            [(1, 2)],
            [(0, 1, 2)],
        ]

    def test_square_between_side_and_diagonal(self) -> None:
        """
        Opposite sides 0-1 and 2-3 are disjoint but 5 µm apart (atoms 1 and 2), and the
        same holds for 0-3 and 1-2: no two edges of the square may share a layer, so
        four layers in edge order, though a fixed coupling map would run two.
        """
        assert blockade_schedule(SQUARE, R_SIDE) == [[(0, 1)], [(0, 3)], [(1, 2)], [(2, 3)]]

    def test_square_without_triangles_is_unchanged_by_ccz(self) -> None:
        assert blockade_schedule(SQUARE, R_SIDE, ccz=True) == blockade_schedule(SQUARE, R_SIDE)

    def test_two_pairs_far_apart_share_a_layer(self) -> None:
        """
        Atoms numbered out of line order: 0 (0,0), 1 (5,0), 4 (10,0), 2 (15,0), 3 (20,0),
        so the edge order differs from the line order and the schedule must follow it.
        """
        register = AtomRegister([[0, 0], [5, 0], [15, 0], [20, 0], [10, 0]])
        # Edges sorted: (0,1), (1,4), (2,3), (2,4).  (0,1) -> L0; (1,4) shares 1 -> L1;
        # (2,3): closest to (0,1) is 1-2 at 10 µm -> L0; (2,4): shares 2 with (2,3) and
        # 4 with (1,4), atom 4 is 5 µm from atom 1 of (0,1) -> L2.
        assert register.interaction_graph(R_SIDE) == [(0, 1), (1, 4), (2, 3), (2, 4)]
        assert blockade_schedule(register, R_SIDE) == [[(0, 1), (2, 3)], [(1, 4)], [(2, 4)]]

    def test_every_gate_is_scheduled_once(self) -> None:
        register = AtomRegister.chain(6, SIDE)
        schedule = blockade_schedule(register, 11.0, ccz=True)
        flat = [gate for layer in schedule for gate in layer]
        expected = register.interaction_graph(11.0) + register.blockade_triangles(11.0)
        assert sorted(flat) == sorted(expected)
        assert len(flat) == len(set(flat))

    def test_no_layer_holds_two_gates_closer_than_the_radius(self) -> None:
        """The rule itself, checked on a 3x3 grid from the positions directly."""
        grid = AtomRegister([[x, y] for y in (0.0, 5.0, 10.0) for x in (0.0, 5.0, 10.0)])
        positions = grid.positions.numpy()
        for radius in (6.0, 7.5, 11.0):
            for layer in blockade_schedule(grid, radius, ccz=True):
                for a, g in enumerate(layer):
                    for h in layer[a + 1 :]:
                        gap = min(
                            float(np.linalg.norm(positions[i] - positions[j]))
                            for i in g
                            for j in h
                        )
                        assert gap >= radius

    def test_is_deterministic(self) -> None:
        grid = AtomRegister([[x, y] for y in (0.0, 5.0) for x in (0.0, 5.0, 10.0)])
        assert blockade_schedule(grid, 7.5, ccz=True) == blockade_schedule(grid, 7.5, ccz=True)

    def test_one_atom_has_an_empty_schedule(self) -> None:
        assert blockade_schedule(AtomRegister([[0.0, 0.0]]), R_SIDE) == []

    def test_a_disconnected_register_names_the_cut_off_atoms(self) -> None:
        """Atoms 2 and 3 form a second pair 30 µm away; atom 4 is alone."""
        register = AtomRegister([[0, 0], [5, 0], [35, 0], [40, 0], [0, 50]])
        with pytest.raises(
            ValueError, match=r"atoms \[2, 3, 4\] are not connected to atom 0.*isolated: \[4\]"
        ):
            blockade_schedule(register, R_SIDE)

    def test_a_radius_at_the_spacing_leaves_every_atom_isolated(self) -> None:
        with pytest.raises(ValueError, match=r"atoms \[1, 2\] are not connected"):
            blockade_schedule(LINE_3, SIDE)

    def test_rejects_a_bad_radius(self) -> None:
        with pytest.raises(ValueError, match="blockade_radius must be a finite number > 0"):
            blockade_schedule(LINE_3, -1.0)


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


class TestNeutralAtomLayer:
    @pytest.mark.parametrize(
        ("register", "pairs"),
        [
            (LINE_3, [(0, 1), (1, 2)]),
            (TRIANGLE, [(0, 1), (0, 2), (1, 2)]),
            (SQUARE, [(0, 1), (0, 3), (1, 2), (2, 3)]),
        ],
        ids=["line", "triangle", "square"],
    )
    def test_unitary_matches_a_hand_built_product_of_cz(
        self, register: AtomRegister, pairs: list[tuple[int, int]]
    ) -> None:
        weights = _weights(register.n_atoms, seed=register.n_atoms)
        expected = _layer_unitary(weights.numpy(), pairs)
        np.testing.assert_allclose(
            _primitive_unitary(weights, register, R_SIDE), expected, atol=1e-12
        )

    def test_unitary_with_ccz_on_the_triangle(self) -> None:
        weights = _weights(3, seed=7)
        expected = _layer_unitary(weights.numpy(), [(0, 1), (0, 2), (1, 2)], [(0, 1, 2)])
        np.testing.assert_allclose(
            _primitive_unitary(weights, TRIANGLE, R_SIDE, ccz=True), expected, atol=1e-12
        )

    def test_ccz_changes_the_triangle_but_not_the_square(self) -> None:
        w = _weights(3, seed=1)
        assert not np.allclose(
            _primitive_unitary(w, TRIANGLE, R_SIDE, ccz=True),
            _primitive_unitary(w, TRIANGLE, R_SIDE),
        )
        w = _weights(4, seed=1)
        np.testing.assert_allclose(
            _primitive_unitary(w, SQUARE, R_SIDE, ccz=True), _primitive_unitary(w, SQUARE, R_SIDE)
        )

    def test_entangler_at_zero_rotation_is_the_cz_phase(self) -> None:
        """With every angle 0 the layer is the diagonal ``(-1)^{Σ b_i b_j}`` alone."""
        weights = torch.zeros(4, 3, dtype=torch.float64)
        expected = np.diag(_diagonal_phase(4, [(0, 1), (0, 3), (1, 2), (2, 3)], []))
        np.testing.assert_allclose(
            _primitive_unitary(weights, SQUARE, R_SIDE), expected, atol=1e-15
        )

    def test_gates_are_recorded_in_schedule_order(self) -> None:
        line = AtomRegister.chain(5, SIDE)
        weights = _weights(5, seed=0)
        tape = qml.tape.make_qscript(neutral_atom_layer)(weights, line, R_SIDE)
        entangling = [tuple(op.wires.tolist()) for op in tape.operations if op.name == "CZ"]
        assert entangling == [(0, 1), (3, 4), (1, 2), (2, 3)]
        assert [op.name for op in tape.operations[4:]] == ["Rot"] * 5
        assert [op.wires.tolist() for op in tape.operations[4:]] == [[i] for i in range(5)]

    def test_records_one_ccz_per_triangle(self) -> None:
        tape = qml.tape.make_qscript(neutral_atom_layer)(_weights(4, 0), SQUARE, 8.0, ccz=True)
        names = [op.name for op in tape.operations]
        assert names.count("CZ") == 6
        assert names.count("CCZ") == 4

    def test_gradients_reach_every_weight(self) -> None:
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(w: torch.Tensor) -> list:
            qml.AngleEmbedding(torch.tensor([0.3, -0.7, 1.1, 0.5]), wires=range(4))
            neutral_atom_layer(w[0], SQUARE, R_SIDE)
            neutral_atom_layer(w[1], SQUARE, R_SIDE)
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        w = _weights(8, seed=3).reshape(2, 4, 3).requires_grad_(True)
        torch.stack(circuit(w)).sum().backward()
        assert w.grad is not None
        # The last layer's ω (RZ before a Z readout) is inert; everything else moves.
        assert bool((w.grad[0] != 0).all())
        assert bool((w.grad[1, :, :2] != 0).all())
        assert torch.allclose(w.grad[1, :, 2], torch.zeros(4, dtype=torch.float64))

    @pytest.mark.parametrize("shape", [(3,), (4,), (3, 3), (4, 2)])
    def test_rejects_the_wrong_weight_shape(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"weights must have shape \(n_atoms=4, 3\)"):
            qml.tape.make_qscript(neutral_atom_layer)(torch.zeros(shape), SQUARE, R_SIDE)

    def test_one_atom_is_a_rotation(self) -> None:
        register = AtomRegister([[0.0, 0.0]])
        w = _weights(1, seed=2)
        np.testing.assert_allclose(
            _primitive_unitary(w, register, R_SIDE), _rot(*w[0].numpy()), atol=1e-12
        )


class TestReadoutLightCone:
    """
    ⟨Z_0⟩ after ``L`` layers depends on the features of the atoms within graph
    distance ``L`` of atom 0, and on no other, under RX and RY embeddings.
    """

    @pytest.mark.parametrize("rotation", ["X", "Y"])
    @pytest.mark.parametrize(
        ("n_layers", "seen"), [(1, [0, 1]), (2, [0, 1, 2]), (3, [0, 1, 2, 3])]
    )
    def test_first_readout_sees_the_graph_ball(
        self, rotation: str, n_layers: int, seen: list[int]
    ) -> None:
        line = AtomRegister.chain(5, SIDE)  # graph distance from atom 0 is the index
        dev = qml.device("default.qubit", wires=5)
        w = _weights(5 * n_layers, seed=11).reshape(n_layers, 5, 3)

        @qml.qnode(dev, diff_method="backprop", interface="torch")
        def circuit(x: torch.Tensor) -> object:
            qml.AngleEmbedding(x, wires=range(5), rotation=rotation)
            for layer in range(n_layers):
                neutral_atom_layer(w[layer], line, R_SIDE)
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor([0.4, -0.9, 1.3, 0.7, -0.2], dtype=torch.float64, requires_grad=True)
        (grad,) = torch.autograd.grad(circuit(x), x)
        assert [i for i in range(5) if abs(grad[i].item()) > 1e-10] == seen
