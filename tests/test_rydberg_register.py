"""
tests/test_rydberg_register.py
==============================
``AtomRegister`` (#496): positions, the chain and ring constructors, the
interaction matrix ``C6 / r^6`` and the blockade radius.

Every expected number is obtained another way than the code computes it: a
distance worked out by hand, a ratio fixed by the geometry, or a figure
quoted in ``docs/rydberg-model.md``.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pennylane as qml
import pytest
import torch

from hqnn_forge.rydberg import DEFAULT_C6, AtomRegister

#: The reference Rabi frequency of ``docs/rydberg-model.md``, in rad/µs.
OMEGA_REF = 4 * math.pi


class TestPositions:
    def test_positions_are_stored_as_float64_rows(self) -> None:
        register = AtomRegister([[0, 0], [3, 4]])
        assert register.n_atoms == 2
        assert register.positions.dtype == torch.float64
        assert register.positions.tolist() == [[0.0, 0.0], [3.0, 4.0]]

    def test_python_floats_keep_double_precision(self) -> None:
        """``torch.as_tensor`` alone would read them as float32: 0.1 → 0.10000000149."""
        assert AtomRegister([[0.1, 0.2], [5.74, 0.3]]).positions.tolist() == [
            [0.1, 0.2],
            [5.74, 0.3],
        ]

    def test_accepts_numpy_and_tensors(self) -> None:
        rows = [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]
        from_numpy = AtomRegister(np.array(rows, dtype=np.float32))
        from_tensor = AtomRegister(torch.tensor(rows))
        assert from_numpy.positions.tolist() == rows
        assert from_tensor.positions.tolist() == rows

    def test_the_register_does_not_alias_its_input(self) -> None:
        source = torch.tensor([[0.0, 0.0], [5.0, 0.0]], dtype=torch.float64)
        register = AtomRegister(source)
        source[1, 0] = 9.0
        register.positions[1, 0] = 7.0
        assert register.positions.tolist() == [[0.0, 0.0], [5.0, 0.0]]

    def test_one_atom_is_a_register(self) -> None:
        register = AtomRegister([[1.5, -2.0]])
        assert register.n_atoms == 1
        assert register.interaction_matrix(DEFAULT_C6).tolist() == [[0.0]]

    @pytest.mark.parametrize(
        "positions",
        [
            [0.0, 1.0],  # one row given flat
            [[0.0, 1.0, 2.0]],  # three coordinates
            [[[0.0, 1.0]]],  # an extra axis
            [[0.0], [1.0]],  # one coordinate
        ],
    )
    def test_rejects_other_shapes(self, positions: list) -> None:
        with pytest.raises(ValueError, match=r"shape \(n_atoms, 2\)"):
            AtomRegister(positions)

    def test_rejects_an_empty_register(self) -> None:
        with pytest.raises(ValueError, match="at least one atom"):
            AtomRegister(torch.empty(0, 2))

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_positions_and_names_the_atom(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"finite.*atom 1"):
            AtomRegister([[0.0, 0.0], [bad, 1.0], [2.0, 0.0]])

    def test_rejects_duplicates_and_names_the_pair(self) -> None:
        with pytest.raises(ValueError, match=r"atoms 0 and 2 are both at \(1\.0, 2\.0\)"):
            AtomRegister([[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]])

    def test_rejects_complex_and_boolean_positions(self) -> None:
        with pytest.raises(TypeError, match="real numbers"):
            AtomRegister(torch.zeros(2, 2, dtype=torch.complex128))
        with pytest.raises(TypeError, match="real numbers"):
            AtomRegister(torch.zeros(2, 2, dtype=torch.bool))

    def test_repr_shows_the_positions(self) -> None:
        assert repr(AtomRegister([[0, 0], [5.5, 0]])) == (
            "AtomRegister(positions=[[0.0, 0.0], [5.5, 0.0]])"
        )


class TestChain:
    def test_atom_i_sits_at_i_times_the_spacing_on_the_x_axis(self) -> None:
        """The geometry of docs/rydberg-model.md: ``r_i = (i · a, 0)``."""
        register = AtomRegister.chain(4, 5.5)
        assert register.positions.tolist() == [[0.0, 0.0], [5.5, 0.0], [11.0, 0.0], [16.5, 0.0]]

    def test_a_chain_of_one(self) -> None:
        assert AtomRegister.chain(1, 5.0).positions.tolist() == [[0.0, 0.0]]

    @pytest.mark.parametrize("n_atoms", [0, -1, 2.0, True, "3"])
    def test_rejects_a_bad_atom_count(self, n_atoms: object) -> None:
        with pytest.raises(ValueError, match="n_atoms must be an integer >= 1"):
            AtomRegister.chain(n_atoms, 5.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("spacing", [0.0, -5.0, math.nan, math.inf, True, "5"])
    def test_rejects_a_bad_spacing(self, spacing: object) -> None:
        with pytest.raises(ValueError, match="spacing must be a finite number > 0"):
            AtomRegister.chain(3, spacing)  # type: ignore[arg-type]


class TestRing:
    @pytest.mark.parametrize("n_atoms", [2, 3, 4, 5, 6, 9])
    def test_neighbours_are_one_spacing_apart_around_the_ring(self, n_atoms: int) -> None:
        spacing = 6.0
        positions = AtomRegister.ring(n_atoms, spacing).positions.numpy()
        for i in range(n_atoms):
            j = (i + 1) % n_atoms  # includes the pair that closes the ring
            assert math.dist(positions[i], positions[j]) == pytest.approx(spacing, rel=1e-13)

    @pytest.mark.parametrize("n_atoms", [3, 4, 6])
    def test_atoms_lie_on_a_circle_around_the_origin(self, n_atoms: int) -> None:
        """Circumradius of a regular polygon of side ``a``: ``a / (2 sin(π/n))``."""
        spacing = 6.0
        radii = AtomRegister.ring(n_atoms, spacing).positions.norm(dim=1)
        expected = {3: spacing / math.sqrt(3), 4: spacing / math.sqrt(2), 6: spacing}[n_atoms]
        assert radii.tolist() == pytest.approx([expected] * n_atoms, rel=1e-13)

    def test_atom_0_is_on_the_positive_x_axis_and_the_order_is_counterclockwise(self) -> None:
        """A square of side ``a``: corners at ``(±a/√2, 0)`` and ``(0, ±a/√2)``."""
        r = 4.0 / math.sqrt(2)
        positions = AtomRegister.ring(4, 4.0).positions
        expected = torch.tensor([[r, 0.0], [0.0, r], [-r, 0.0], [0.0, -r]], dtype=torch.float64)
        assert torch.allclose(positions, expected, rtol=0, atol=1e-14)

    def test_a_ring_of_one_is_one_atom_at_the_origin(self) -> None:
        assert AtomRegister.ring(1, 5.0).positions.tolist() == [[0.0, 0.0]]

    def test_square_and_hexagon_couplings_follow_from_the_geometry(self) -> None:
        """Square diagonal ``a√2`` → ``V/8``; hexagon: ``a√3`` → ``V/27`` and ``2a`` → ``V/64``."""
        c6, spacing = 729.0, 3.0  # nearest-neighbour V = 729 / 3^6 = 1
        square = AtomRegister.ring(4, spacing).interaction_matrix(c6)
        assert square[0].tolist() == pytest.approx([0.0, 1.0, 1 / 8, 1.0], rel=1e-12)
        hexagon = AtomRegister.ring(6, spacing).interaction_matrix(c6)
        assert hexagon[0].tolist() == pytest.approx(
            [0.0, 1.0, 1 / 27, 1 / 64, 1 / 27, 1.0], rel=1e-12
        )

    def test_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="n_atoms must be an integer >= 1"):
            AtomRegister.ring(0, 5.0)
        with pytest.raises(ValueError, match="spacing must be a finite number > 0"):
            AtomRegister.ring(4, 0.0)


class TestInteractionMatrix:
    def test_a_3_4_5_triangle_by_hand(self) -> None:
        """Distances 3, 4 and 5 µm: ``V = C6 / 729``, ``C6 / 4096``, ``C6 / 15625``."""
        register = AtomRegister([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        v = register.interaction_matrix(1.0e6)
        expected = [
            [0.0, 1.0e6 / 729, 1.0e6 / 4096],
            [1.0e6 / 729, 0.0, 1.0e6 / 15625],
            [1.0e6 / 4096, 1.0e6 / 15625, 0.0],
        ]
        assert v.dtype == torch.float64
        assert torch.allclose(v, torch.tensor(expected, dtype=torch.float64), rtol=1e-14, atol=0)

    def test_three_atom_chain_next_nearest_is_one_64th_of_the_nearest(self) -> None:
        """The full tail of docs/rydberg-model.md: twice the distance, ``2^6 = 64``."""
        v = AtomRegister.chain(3, 5.7).interaction_matrix(DEFAULT_C6)
        nearest = DEFAULT_C6 / 5.7**6
        assert v[0, 1].item() == pytest.approx(nearest, rel=1e-13)
        assert v[1, 2].item() == pytest.approx(nearest, rel=1e-13)
        assert (v[0, 2] / v[0, 1]).item() == pytest.approx(1 / 64, rel=1e-13)

    def test_six_atom_chain_follows_one_over_separation_to_the_sixth(self) -> None:
        """``V_ij = V / |i − j|^6``: 1, 1/64, 1/729, 1/4096, 1/15625 along the first row."""
        v = AtomRegister.chain(6, 2.0).interaction_matrix(64.0)  # V = 64 / 2^6 = 1
        assert v[0].tolist() == pytest.approx(
            [0.0, 1.0, 1 / 64, 1 / 729, 1 / 4096, 1 / 15625], rel=1e-13
        )

    def test_symmetric_with_an_exactly_zero_diagonal(self) -> None:
        generator = torch.Generator().manual_seed(0)
        register = AtomRegister(torch.rand(6, 2, generator=generator, dtype=torch.float64) * 20)
        v = register.interaction_matrix(DEFAULT_C6)
        assert torch.equal(v, v.T)
        assert torch.equal(v.diagonal(), torch.zeros(6, dtype=torch.float64))
        assert bool((v[~torch.eye(6, dtype=torch.bool)] > 0).all())

    def test_does_not_depend_on_where_the_register_sits(self) -> None:
        """Only differences of positions enter: a translation and a 90° turn change nothing."""
        rows = torch.tensor([[0.0, 0.0], [4.0, 1.0], [2.0, 7.0]], dtype=torch.float64)
        turned = torch.stack([-rows[:, 1], rows[:, 0]], dim=1) + torch.tensor([10.0, -3.0])
        v = AtomRegister(rows).interaction_matrix(DEFAULT_C6)
        assert torch.allclose(AtomRegister(turned).interaction_matrix(DEFAULT_C6), v, rtol=1e-12)

    def test_the_sign_of_c6_is_kept(self) -> None:
        v = AtomRegister.chain(2, 2.0).interaction_matrix(-64.0)
        assert v.tolist() == [[0.0, -1.0], [-1.0, 0.0]]

    def test_default_c6_matches_the_measurement_quoted_on_the_model_page(self) -> None:
        """Bernien et al. (2017): ``2π × 24.4 MHz`` near 5.7 µm; the page states 24.1 at 5.74."""
        v = AtomRegister.chain(2, 5.74).interaction_matrix(DEFAULT_C6)[0, 1].item()
        assert v / (2 * math.pi) == pytest.approx(24.1, abs=0.05)

    @pytest.mark.parametrize("c6", [math.nan, math.inf, "1e6", True, None])
    def test_rejects_a_bad_c6(self, c6: object) -> None:
        with pytest.raises(ValueError, match="c6 must be a finite number"):
            AtomRegister.chain(2, 5.0).interaction_matrix(c6)  # type: ignore[arg-type]

    def test_refuses_an_interaction_that_overflows(self) -> None:
        register = AtomRegister([[0.0, 0.0], [1e-60, 0.0]])
        with pytest.raises(ValueError, match=r"not finite.*atoms 0 and 1"):
            register.interaction_matrix(DEFAULT_C6)


class TestDefaultC6:
    def test_value_stated_on_the_model_page(self) -> None:
        """``2π × 862 690 MHz µm⁶ = 5 420 441 rad/µs · µm⁶``."""
        assert pytest.approx(2 * math.pi * 862690, rel=1e-15) == DEFAULT_C6
        assert round(DEFAULT_C6) == 5_420_441

    def test_is_pennylanes_default_coefficient(self) -> None:
        """PennyLane takes it in MHz µm⁶ and multiplies by 2π itself."""
        signature = inspect.signature(qml.pulse.rydberg_interaction)
        in_mhz = signature.parameters["interaction_coeff"].default
        assert in_mhz == 862690
        assert pytest.approx(2 * math.pi * in_mhz, rel=1e-15) == DEFAULT_C6


class TestBlockadeRadius:
    def test_by_hand(self) -> None:
        """``(64 / 1)^(1/6) = 2`` and ``(729 / 1)^(1/6) = 3``."""
        assert AtomRegister.blockade_radius(64.0, 1.0) == pytest.approx(2.0, rel=1e-14)
        assert AtomRegister.blockade_radius(1458.0, 2.0) == pytest.approx(3.0, rel=1e-14)

    def test_callable_on_a_register_too(self) -> None:
        register = AtomRegister.chain(3, 5.0)
        assert register.blockade_radius(64.0, 1.0) == AtomRegister.blockade_radius(64.0, 1.0)

    def test_reference_value_of_the_model_page(self) -> None:
        """``R_b = 8.69 µm`` at ``Ω = 4π rad/µs`` with the default C6."""
        assert AtomRegister.blockade_radius(DEFAULT_C6, OMEGA_REF) == pytest.approx(8.69, abs=5e-3)

    def test_two_atoms_one_blockade_radius_apart_interact_with_omega(self) -> None:
        """The definition: ``V(R_b) = Ω``, the edge of the unit-disk graph of #418."""
        radius = AtomRegister.blockade_radius(DEFAULT_C6, OMEGA_REF)
        v = AtomRegister.chain(2, radius).interaction_matrix(DEFAULT_C6)[0, 1].item()
        assert v == pytest.approx(OMEGA_REF, rel=1e-12)

    @pytest.mark.parametrize(
        ("ratio", "spacing"),
        [(0.1, 12.76), (0.3, 10.62), (1, 8.69), (3, 7.24), (10, 5.92), (30, 4.93), (100, 4.03)],
    )
    def test_spacing_table_of_the_model_page(self, ratio: float, spacing: float) -> None:
        """``a = R_b · (V/Ω)^(−1/6)``, to the two decimals the page prints."""
        radius = AtomRegister.blockade_radius(DEFAULT_C6, OMEGA_REF)
        assert radius * ratio ** (-1 / 6) == pytest.approx(spacing, abs=5.1e-3)
        v = AtomRegister.chain(2, spacing).interaction_matrix(DEFAULT_C6)[0, 1].item()
        # A spacing rounded to 0.005 µm moves V by up to 6 · 0.005 / a, under 1 %.
        assert v / OMEGA_REF == pytest.approx(ratio, rel=1e-2)

    @pytest.mark.parametrize(
        ("c6", "omega", "name"),
        [
            (0.0, 1.0, "c6"),
            (-64.0, 1.0, "c6"),
            (math.nan, 1.0, "c6"),
            (64.0, 0.0, "omega"),
            (64.0, -1.0, "omega"),
            (64.0, math.inf, "omega"),
        ],
    )
    def test_rejects_non_positive_arguments(self, c6: float, omega: float, name: str) -> None:
        with pytest.raises(ValueError, match=f"{name} must be a finite number > 0"):
            AtomRegister.blockade_radius(c6, omega)
