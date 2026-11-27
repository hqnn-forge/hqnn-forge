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

import dataclasses
import inspect
import itertools
import math

import numpy as np
import pennylane as qml
import pytest
import scipy.spatial
import torch

from hqnn_forge.rydberg import AQUILA_LIMITS, DEFAULT_C6, AtomRegister, DeviceLimits

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

    def test_rejects_positions_that_require_grad(self) -> None:
        """Storing a detached copy would drop the gradient without a word."""
        positions = torch.tensor([[0.0, 0.0], [5.0, 0.0]], requires_grad=True)
        with pytest.raises(ValueError, match="must not require grad"):
            AtomRegister(positions)
        with pytest.raises(ValueError, match="must not require grad"):
            AtomRegister(positions * 2.0)  # a non-leaf of the graph as well
        with pytest.raises(ValueError, match="must not require grad"):
            AtomRegister(positions.detach().double().requires_grad_())  # no conversion
        with pytest.raises(ValueError, match="must not require grad"):
            AtomRegister(torch.nn.Parameter(positions.detach()))
        assert AtomRegister(positions.detach()).positions.tolist() == [[0.0, 0.0], [5.0, 0.0]]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rejects_positions_that_require_grad_under_no_grad(self, dtype: torch.dtype) -> None:
        """The float64 copy of a float32 tensor made under ``no_grad`` requires none."""
        positions = torch.tensor([[0.0, 0.0], [5.0, 0.0]], dtype=dtype, requires_grad=True)
        with torch.no_grad(), pytest.raises(ValueError, match="must not require grad"):
            AtomRegister(positions)

    def test_rejects_positions_with_a_forward_mode_tangent(self) -> None:
        """#531: a dual tensor does not require grad, and ``.detach()`` drops its tangent."""
        positions = torch.tensor([[0.0, 0.0], [5.0, 0.0]], dtype=torch.float64)

        def total(pos: torch.Tensor) -> torch.Tensor:
            return AtomRegister(pos).interaction_matrix(15625.0).sum()

        with pytest.raises(ValueError, match="must not require grad or carry a forward-mode"):
            torch.func.jacfwd(total)(positions)
        with pytest.raises(ValueError, match="must not require grad or carry a forward-mode"):
            torch.func.jvp(total, (positions,), (torch.ones_like(positions),))
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(positions, torch.ones_like(positions))
            with pytest.raises(ValueError, match="must not require grad or carry a forward-mode"):
                AtomRegister(dual)
            # Inside a dual level a tensor without a tangent is still a constant.
            assert AtomRegister(positions).positions.tolist() == positions.tolist()
        with pytest.raises(ValueError, match="must not require grad"):
            torch.func.jacrev(total)(positions)  # reverse mode, refused as before

    @pytest.mark.parametrize("mode", [torch.enable_grad, torch.no_grad, torch.inference_mode])
    @pytest.mark.parametrize("container", [list, tuple])
    def test_rejects_a_list_of_rows_that_require_grad(self, mode: object, container: type) -> None:
        """#531: under no_grad the rows used to be converted and their gradient dropped."""
        p = torch.tensor([[0.0, 0.0], [5.0, 0.0]], dtype=torch.float64, requires_grad=True)
        rows = container([p[0], p[1]])
        nested = container([[p[0, 0], 0.0], [5.0, 0.0]])  # one entry deep inside a row
        with mode():  # type: ignore[operator]
            for positions in (rows, nested):
                with pytest.raises(ValueError, match="must not require grad"):
                    AtomRegister(positions)

    def test_constant_inputs_are_accepted_as_before(self) -> None:
        expected = [[0.0, 0.0], [5.0, 0.0]]
        tensor = torch.tensor(expected, requires_grad=True)
        for positions in (
            expected,
            tuple(map(tuple, expected)),
            np.array(expected),
            tensor.detach(),
            [tensor[0].detach(), tensor[1].detach()],
        ):
            assert AtomRegister(positions).positions.tolist() == expected
        with torch.no_grad():
            assert AtomRegister([tensor[0] * 1, tensor[1] * 1]).positions.tolist() == expected
        assert AtomRegister.chain(2, 5.0).positions.tolist() == expected
        assert AtomRegister.ring(3, 5.0).n_atoms == 3

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


class TestGrid:
    def test_a_2_by_3_grid_is_row_major_by_hand(self) -> None:
        """Atom ``k = r · cols + c`` at ``(c · a, r · a)``: row 0 left to right, then row 1."""
        a = 5.5
        assert AtomRegister.grid(2, 3, a).positions.tolist() == [
            [0.0, 0.0],
            [5.5, 0.0],
            [11.0, 0.0],
            [0.0, 5.5],
            [5.5, 5.5],
            [11.0, 5.5],
        ]

    def test_a_square_has_v_on_its_edges_and_v_over_8_on_its_diagonals(self) -> None:
        """2 × 2: edges (0,1), (2,3), (0,2), (1,3) at ``a``; diagonals at ``√2 a``, ``(√2)⁶ = 8``."""
        c6, spacing = 15625.0, 5.0  # nearest-neighbour V = 15625 / 5^6 = 1
        v = AtomRegister.grid(2, 2, spacing).interaction_matrix(c6)
        expected = [
            [0.0, 1.0, 1.0, 1 / 8],
            [1.0, 0.0, 1 / 8, 1.0],
            [1.0, 1 / 8, 0.0, 1.0],
            [1 / 8, 1.0, 1.0, 0.0],
        ]
        torch.testing.assert_close(
            v, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0
        )

    def test_couplings_of_a_3_by_4_grid_from_the_lattice_offsets(self) -> None:
        """``V_kl = V / (Δr² + Δc²)³`` with ``(r, c) = divmod(k, cols)``, summed by hand."""
        rows, cols, c6, spacing = 3, 4, 729.0, 3.0  # V = 1
        v = AtomRegister.grid(rows, cols, spacing).interaction_matrix(c6).numpy()
        for k in range(rows * cols):
            for m in range(rows * cols):
                (rk, ck), (rm, cm) = divmod(k, cols), divmod(m, cols)
                squared = (rk - rm) ** 2 + (ck - cm) ** 2
                expected = 0.0 if k == m else 1.0 / squared**3
                assert v[k, m] == pytest.approx(expected, rel=1e-12)

    def test_one_row_is_the_chain(self) -> None:
        assert AtomRegister.grid(1, 4, 6.0).positions.tolist() == (
            AtomRegister.chain(4, 6.0).positions.tolist()
        )

    @pytest.mark.parametrize("name", ["rows", "cols"])
    @pytest.mark.parametrize("count", [0, -1, 2.0, True, "3"])
    def test_rejects_a_bad_count(self, name: str, count: object) -> None:
        shape = {"rows": 2, "cols": 2, name: count}
        with pytest.raises(ValueError, match=f"{name} must be an integer >= 1"):
            AtomRegister.grid(shape["rows"], shape["cols"], 5.0)  # type: ignore[arg-type]

    def test_rejects_a_bad_spacing(self) -> None:
        with pytest.raises(ValueError, match="spacing must be a finite number > 0"):
            AtomRegister.grid(2, 2, -1.0)


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

    # PennyLane 0.46 removed ``qml.pulse``; the value itself stays pinned by
    # test_value_stated_on_the_model_page, so only the cross-check goes (#569).
    @pytest.mark.may_skip  # the upstream job runs the PennyLane nightly
    @pytest.mark.skipif(
        not hasattr(qml, "pulse"),
        reason="qml.pulse is gone (PennyLane 0.46); nothing to cross-check DEFAULT_C6 against",
    )
    def test_is_pennylanes_default_coefficient(self) -> None:
        """PennyLane (up to 0.45) takes it in MHz µm⁶ and multiplies by 2π itself."""
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
        """The definition: ``V(R_b) = Ω``, the boundary of the unit-disk graph of #418."""
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


def _names(violations: list[str]) -> list[str]:
    """The field each message starts with."""
    return [v.split(":")[0] for v in violations]


class TestDeviceViolations:
    """
    ``AtomRegister.device_violations`` against the register bounds of a device (#783).

    Aquila's numbers are typed in from the datasheet of Wurtz et al. (2023): user-defined
    sites at least 4 µm apart, a site pattern at most 75 µm wide and 76 µm high, which the
    Braket device schema defines as the largest difference between the x coordinates, and
    between the y coordinates, of two sites.  Distances are worked out by hand or by
    ``scipy.spatial.distance.pdist``, extents by ``numpy.ptp``.
    """

    #: The register fields of a device alone; the pulse fields play no part here.
    LIMITS = DeviceLimits(
        omega_max=math.inf,
        detuning_max=math.inf,
        duration_max=math.inf,
        omega_slew_max=math.inf,
        min_spacing=4.0,
        region=(75.0, 76.0),
    )

    def test_aquila_is_these_bounds(self) -> None:
        assert (AQUILA_LIMITS.min_spacing, AQUILA_LIMITS.region) == (4.0, (75.0, 76.0))

    def test_a_pair_closer_than_the_spacing_is_named_with_its_distance(self) -> None:
        """Atoms 0 and 1 are 3.99 µm apart; 1 and 2 are 6.01 µm, 0 and 2 are 10 µm."""
        register = AtomRegister([[0.0, 0.0], [3.99, 0.0], [10.0, 0.0]])
        assert register.device_violations(self.LIMITS) == [
            "min_spacing: min |r_i − r_j| = 3.99 µm (atoms 0 and 1) is below 4 µm"
        ]

    def test_the_closest_pair_is_named_also_off_the_axes(self) -> None:
        """A 3-4-5 triangle scaled by 0.75: sides 2.25, 3 and 3.75 µm; atoms 0 and 1 closest."""
        register = AtomRegister([[10.0, 10.0], [12.25, 10.0], [10.0, 13.0]])
        assert register.device_violations(self.LIMITS) == [
            "min_spacing: min |r_i − r_j| = 2.25 µm (atoms 0 and 1) is below 4 µm"
        ]
        # 3² + 4² = 5², exact in float64: a diagonal pair at the bound passes; at (3, 3.99),
        # √(9 + 15.9201) = 4.99200 µm, it does not.
        five = dataclasses.replace(self.LIMITS, min_spacing=5.0)
        assert AtomRegister([[0.0, 0.0], [3.0, 4.0]]).device_violations(five) == []
        closer = AtomRegister([[0.0, 0.0], [3.0, 3.99]])
        assert closer.device_violations(five) == [
            "min_spacing: min |r_i − r_j| = 4.992 µm (atoms 0 and 1) is below 5 µm"
        ]

    def test_of_equally_close_pairs_the_first_is_named(self) -> None:
        """Pairs ``(i, j)``, ``i < j``, in lexicographic order: ``(0, 1)`` before ``(1, 2)``."""
        assert AtomRegister.chain(3, 3.5).device_violations(self.LIMITS) == [
            "min_spacing: min |r_i − r_j| = 3.5 µm (atoms 0 and 1) is below 4 µm"
        ]
        # Here the closest pair holds the last two rows.
        later = AtomRegister([[0.0, 0.0], [5.0, 0.0], [8.5, 0.0]])
        assert "(atoms 1 and 2)" in later.device_violations(self.LIMITS)[0]

    @pytest.mark.parametrize(
        ("spacing", "names"),
        [(4.0, []), (math.nextafter(4.0, 0.0), ["min_spacing"]), (math.nextafter(4.0, 5.0), [])],
    )
    def test_a_pair_exactly_at_the_spacing_passes(self, spacing: float, names: list[str]) -> None:
        """``d < min_spacing`` breaks the bound; the distances here are exact in float64."""
        assert _names(AtomRegister.chain(2, spacing).device_violations(self.LIMITS)) == names

    def test_too_wide_is_named_with_the_extent_in_x(self) -> None:
        register = AtomRegister([[-10.0, 0.0], [0.0, 30.0], [65.01, 5.0]])
        assert register.device_violations(self.LIMITS) == [
            "region: max x − min x = 75.01 µm exceeds the width 75 µm"
        ]

    def test_too_high_is_named_with_the_extent_in_y(self) -> None:
        register = AtomRegister([[0.0, -40.0], [5.0, 36.01], [-5.0, 0.0]])
        assert register.device_violations(self.LIMITS) == [
            "region: max y − min y = 76.01 µm exceeds the height 76 µm"
        ]

    @pytest.mark.parametrize(
        ("positions", "names"),
        [
            # Just inside and just outside each side, with both ends exact in float64.
            ([[0.0, 0.0], [75.0, 0.0]], []),
            ([[0.0, 0.0], [math.nextafter(75.0, 76.0), 0.0]], ["region"]),
            ([[0.0, 0.0], [0.0, 76.0]], []),
            ([[0.0, 0.0], [0.0, math.nextafter(76.0, 77.0)]], ["region"]),
            # Width is x and height is y: 75.5 µm fits along y but not along x.
            ([[0.0, 0.0], [0.0, 75.5]], []),
            ([[0.0, 0.0], [75.5, 0.0]], ["region"]),
            # Both sides at once: one message each, x first.
            ([[0.0, 0.0], [80.0, 80.0]], ["region", "region"]),
        ],
    )
    def test_each_side_of_the_region_alone(
        self, positions: list[list[float]], names: list[str]
    ) -> None:
        found = AtomRegister(positions).device_violations(self.LIMITS)
        assert _names(found) == names
        if len(found) == 2:
            assert "width" in found[0] and "height" in found[1]

    def test_every_bound_at_once_in_field_order(self) -> None:
        register = AtomRegister([[0.0, 0.0], [1.0, 0.0], [80.0, 80.0]])
        assert _names(register.device_violations(self.LIMITS)) == [
            "min_spacing",
            "region",
            "region",
        ]

    def test_a_shift_of_the_register_changes_nothing(self) -> None:
        """Both bounds are on differences of coordinates, so the origin does not matter."""
        rows = torch.tensor([[0.0, 0.0], [3.0, 0.0], [80.0, 80.0]], dtype=torch.float64)
        shifted = rows + torch.tensor([-1024.0, 512.0], dtype=torch.float64)
        found = AtomRegister(rows).device_violations(self.LIMITS)
        assert AtomRegister(shifted).device_violations(self.LIMITS) == found
        assert _names(found) == ["min_spacing", "region", "region"]

    def test_one_atom_has_no_pair_and_no_extent(self) -> None:
        tight = DeviceLimits(1.0, 1.0, 1.0, 1.0, min_spacing=1e9, region=(0.0, 0.0))
        assert AtomRegister([[3.0, -7.0]]).device_violations(tight) == []

    def test_without_register_fields_nothing_is_bounded(self) -> None:
        """The defaults, ``0`` and ``(inf, inf)``, keep any register."""
        loose = DeviceLimits(1.0, 1.0, 1.0, 1.0)
        register = AtomRegister([[0.0, 0.0], [1e-3, 0.0], [1e6, -1e6]])
        assert register.device_violations(loose) == []

    @pytest.mark.parametrize("seed", range(6))
    def test_against_scipy_on_random_registers(self, seed: int) -> None:
        """The flags and the numbers named, from ``pdist`` and ``ptp`` of the same positions."""
        rng = np.random.default_rng(seed)
        positions = rng.uniform(0.0, 90.0, size=(8, 2))
        found = AtomRegister(positions).device_violations(self.LIMITS)
        distances = scipy.spatial.distance.pdist(positions)
        pairs = list(itertools.combinations(range(8), 2))  # the order of pdist
        closest = int(np.argmin(distances))
        width, height = np.ptp(positions, axis=0)
        expected = []
        if distances[closest] < 4.0:
            i, j = pairs[closest]
            expected.append(
                f"min_spacing: min |r_i − r_j| = {distances[closest]:.6g} µm "
                f"(atoms {i} and {j}) is below 4 µm"
            )
        if width > 75.0:
            expected.append(f"region: max x − min x = {width:.6g} µm exceeds the width 75 µm")
        if height > 76.0:
            expected.append(f"region: max y − min y = {height:.6g} µm exceeds the height 76 µm")
        assert found == expected

    def test_the_random_registers_cover_both_outcomes_of_each_bound(self) -> None:
        """So that the comparison above is not one of empty lists."""
        flags = set()
        for seed in range(6):
            positions = np.random.default_rng(seed).uniform(0.0, 90.0, size=(8, 2))
            width, height = np.ptp(positions, axis=0)
            close = scipy.spatial.distance.pdist(positions).min() < 4.0
            flags |= {("spacing", close), ("width", width > 75.0), ("height", height > 76.0)}
        assert len(flags) == 6

    @pytest.mark.parametrize("geometry", ["chain", "ring", "grid"])
    @pytest.mark.parametrize("ratio", [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    def test_the_benchmark_grid_fits_aquila(self, geometry: str, ratio: float) -> None:
        """
        Four atoms at ``a = R_b · (V/Ω)^(−1/6)``, 12.76 to 4.03 µm (*Geometry* of the model
        page): every pair at least ``a`` apart, and at most ``3a = 38.3`` µm across.
        """
        a = AtomRegister.blockade_radius(DEFAULT_C6, OMEGA_REF) * ratio ** (-1 / 6)
        register = {
            "chain": AtomRegister.chain(4, a),
            "ring": AtomRegister.ring(4, a),
            "grid": AtomRegister.grid(2, 2, a),
        }[geometry]
        positions = register.positions.numpy()
        assert a > 4.03
        assert scipy.spatial.distance.pdist(positions).min() == pytest.approx(a, rel=1e-13)
        assert np.ptp(positions, axis=0).max() <= 3 * a + 1e-12 < 39.0
        assert register.device_violations(AQUILA_LIMITS) == []

    def test_the_model_pages_bounds_on_v_over_omega(self) -> None:
        """
        *Hardware realism*: ``0.04 ≤ V/Ω ≤ 105`` at the reference Ω for a row of 6 atoms.
        ``a = R_b (V/Ω)^(−1/6)`` is 4 µm at ``V/Ω = (R_b/4)⁶ = 105.3``, and the row of 6,
        ``5a`` long, is 75 µm at ``(R_b/15)⁶ = 0.0379``.
        """
        radius = AtomRegister.blockade_radius(DEFAULT_C6, OMEGA_REF)
        assert (radius / 4) ** 6 == pytest.approx(105.3, abs=0.05)
        assert (radius / 15) ** 6 == pytest.approx(0.0379, abs=5e-5)
        cases = [(105.0, []), (106.0, ["min_spacing"]), (0.04, []), (0.037, ["region"])]
        for ratio, names in cases:
            row = AtomRegister.chain(6, radius * ratio ** (-1 / 6))
            assert _names(row.device_violations(AQUILA_LIMITS)) == names
