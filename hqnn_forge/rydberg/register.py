"""
hqnn_forge.rydberg.register
===========================
The positions of a neutral-atom array and what follows from them alone: the
pairwise van der Waals couplings and the blockade radius.

Units are those of ``docs/rydberg-model.md``: ħ = 1, lengths in µm, angular
frequencies in rad/µs, so ``C6`` is in rad/µs · µm⁶.

One register serves both ways of using the array:

* the analog model (:func:`~hqnn_forge.rydberg.rydberg_hamiltonian`) takes the
  continuous couplings ``V_ij = C6 / |r_i − r_j|⁶`` from
  :meth:`AtomRegister.interaction_matrix`;
* a gate-based entangler connects the pairs closer than the blockade radius
  ``R_b = (C6/Ω)^(1/6)`` (:meth:`AtomRegister.blockade_radius`).  Since
  ``V_ij > Ω`` exactly when ``|r_i − r_j| < R_b``, that unit-disk graph is the
  set of pairs with ``V_ij > Ω``: the two views are the same geometry.  A
  pair at exactly ``R_b`` has ``V_ij = Ω`` and lies on the boundary of the
  graph.  The convention is strict: it is **not** an edge
  (:meth:`AtomRegister.interaction_graph`), since ``V_ij > Ω`` is what makes
  the doubly excited state blockaded.  :meth:`AtomRegister.blockade_triangles`
  lists the triples pairwise inside ``R_b``, and
  :func:`hqnn_forge.circuits.blockade_schedule` schedules gates on both.

A register holds positions and nothing else.  The atomic species (``C6``) and
the drive (Ω) are arguments of the methods that need them.

References
----------
* Saffman, Walker & Mølmer (2010) "Quantum information with Rydberg atoms",
  Reviews of Modern Physics 82, 2313.
* Bernien et al. (2017) "Probing many-body dynamics on a 51-atom quantum
  simulator", Nature 551, 579–584.
"""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.autograd import forward_ad

if TYPE_CHECKING:
    # Only for the annotation: feature_map imports this module.
    from hqnn_forge.rydberg.feature_map import DeviceLimits

__all__ = ["AtomRegister"]


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    """``value`` as a float, or a ``ValueError`` naming the argument."""
    number = (
        float(value)
        if isinstance(value, numbers.Real) and not isinstance(value, bool)
        else math.nan
    )
    if not math.isfinite(number) or (positive and number <= 0):
        requirement = "a finite number > 0" if positive else "a finite number"
        raise ValueError(f"{name} must be {requirement}; got {value!r}.")
    return number


def _real_tensor(value: Any, name: str) -> torch.Tensor:
    """
    ``value`` as a ``float64`` tensor, or a ``TypeError`` for complex or boolean data.

    A tensor keeps its device and its graph.  Anything else goes through NumPy
    first: ``torch.as_tensor`` reads Python floats as ``float32``, which would
    round a position or a detuning to 7 digits before the conversion to
    ``float64``; NumPy reads them as ``float64``.
    """
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must hold real numbers; got dtype {tensor.dtype}.")
    return tensor.to(torch.float64)


def _carries_derivative(value: Any) -> bool:
    """
    Whether ``value``, or a tensor in a list or tuple of it, would take a derivative along.

    A tensor that requires grad does for reverse mode (``backward``,
    ``torch.func.grad``, ``jacrev``), whatever the grad mode the register is
    built under.  A dual tensor of forward mode does not require grad, and
    ``.detach()`` strips its tangent, so it is recognised by its tangent:
    ``unpack_dual`` returns one for a ``torch.autograd.forward_ad.make_dual``
    tensor and for the inputs of ``torch.func.jvp`` and ``jacfwd``, and ``None``
    for any other tensor, also outside a dual level.  Lists and tuples are
    walked, since a container of tensors that require grad is converted
    without a graph under ``torch.no_grad()`` or ``torch.inference_mode()``.
    """
    if isinstance(value, torch.Tensor):
        return value.requires_grad or forward_ad.unpack_dual(value).tangent is not None
    if isinstance(value, (list, tuple)):
        return any(_carries_derivative(item) for item in value)
    return False


def _atom_count(n_atoms: Any, name: str = "n_atoms") -> int:
    """``n_atoms`` as an ``int``, or a ``ValueError`` naming the argument unless it is ``>= 1``."""
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, numbers.Integral) or n_atoms < 1:
        raise ValueError(f"{name} must be an integer >= 1; got {n_atoms!r}.")
    return int(n_atoms)


class AtomRegister:
    """
    The positions of the atoms of an array, in µm, in a plane.

    Atoms are counted from 0 in the order of the rows of ``positions``.  That
    order is the order of everything derived from the register: row and column
    ``i`` of :meth:`interaction_matrix`, and tensor factor (PennyLane wire)
    ``i`` of :func:`~hqnn_forge.rydberg.rydberg_hamiltonian`.

    Parameters
    ----------
    positions:
        Shape ``(n_atoms, 2)``, one ``(x, y)`` row per atom, in µm.  Any
        real-valued array-like; it is copied and stored as ``float64``.  The
        positions are constants: the copy carries no graph, so a tensor that
        requires grad, a dual tensor of forward mode (``torch.func.jvp``,
        ``jacfwd``, ``torch.autograd.forward_ad``), or a list or tuple
        holding either, is refused instead of losing its derivative
        unnoticed, whatever the grad mode.

    Raises
    ------
    TypeError
        If ``positions`` is complex or boolean.
    ValueError
        If ``positions`` does not have shape ``(n_atoms, 2)`` with at least
        one atom, carries a derivative (a tensor, or a list or tuple of
        tensors, that requires grad or has a forward-mode tangent), holds a
        NaN or an infinity,
        or places two atoms at the same point (their coupling ``C6 / 0``
        would be infinite).

    Notes
    -----
    Only exact duplicates are refused.  How close two atoms may sensibly be is
    a statement about the atom, not the geometry: the ``C6 / r⁶`` form is a
    long-distance one, and ``docs/rydberg-model.md`` does not use it below
    4 µm.  The register has no upper limit on the number of atoms; the dense
    Hamiltonian has (:data:`~hqnn_forge.rydberg.MAX_ATOMS`).  What a device
    allows, a smallest spacing and a region, is reported on request by
    :meth:`device_violations` and never refused here.

    Examples
    --------
    >>> register = AtomRegister.chain(3, spacing=5.0)
    >>> register.positions.tolist()
    [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]
    >>> v = register.interaction_matrix(c6=15625.0)  # V = C6 / 5^6 = 1
    >>> v[0, 1].item(), v[0, 2].item()
    (1.0, 0.015625)
    """

    def __init__(self, positions: Any) -> None:
        # Asked of the argument, before any conversion: under torch.no_grad() the
        # conversion of another dtype, or of a list of tensors, returns a tensor that no
        # longer requires grad, and .detach() strips a forward-mode tangent (#531).
        if _carries_derivative(positions):
            raise ValueError(
                "positions must not require grad or carry a forward-mode tangent "
                "(torch.func.jvp, jacfwd, forward_ad): the register stores them as constants, "
                "so no derivative would reach them. Pass detached values, such as "
                "positions.detach(), to use them."
            )
        tensor = _real_tensor(positions, "positions")
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError(
                f"positions must have shape (n_atoms, 2), one (x, y) row per atom in µm; "
                f"got shape {tuple(tensor.shape)}."
            )
        if tensor.shape[0] < 1:
            raise ValueError("positions must hold at least one atom; got shape (0, 2).")
        tensor = tensor.detach().to(device="cpu", dtype=torch.float64, copy=True)
        finite = torch.isfinite(tensor).all(dim=1)
        if not bool(finite.all()):
            atom = int((~finite).nonzero()[0, 0])
            raise ValueError(
                f"positions must be finite; atom {atom} is at {tuple(tensor[atom].tolist())}."
            )
        same = (tensor[:, None, :] == tensor[None, :, :]).all(dim=-1)
        duplicates = torch.triu(same, diagonal=1).nonzero()
        if len(duplicates):
            i, j = (int(k) for k in duplicates[0])
            raise ValueError(
                f"positions must not repeat: atoms {i} and {j} are both at "
                f"{tuple(tensor[i].tolist())}."
            )
        self._positions = tensor

    @classmethod
    def chain(cls, n_atoms: int, spacing: float) -> AtomRegister:
        """
        An open chain: atom ``i`` at ``r_i = (i · spacing, 0)``.

        This is the geometry of ``docs/rydberg-model.md``.  Atoms ``i`` and
        ``j`` are ``|i − j| · spacing`` apart, so with ``V = C6 / spacing⁶``
        the nearest-neighbour coupling, ``V_ij = V / |i − j|⁶``.

        Parameters
        ----------
        n_atoms:
            Number of atoms, at least 1.
        spacing:
            Distance between neighbouring atoms in µm, finite and positive.

        Raises
        ------
        ValueError
            If ``n_atoms`` is not an integer ``>= 1`` or ``spacing`` is not a
            finite number ``> 0``.
        """
        n = _atom_count(n_atoms)
        a = _finite_number(spacing, "spacing", positive=True)
        x = torch.arange(n, dtype=torch.float64) * a
        return cls(torch.stack([x, torch.zeros_like(x)], dim=1))

    @classmethod
    def ring(cls, n_atoms: int, spacing: float) -> AtomRegister:
        """
        A closed ring: the corners of a regular polygon of side ``spacing``.

        Atom ``i`` sits at ``R · (cos θ_i, sin θ_i)`` with ``θ_i = 2π i / n``,
        counterclockwise from the positive x axis around the origin.  Two
        atoms ``k`` steps apart around the ring subtend the angle ``2π k / n``
        and are the chord ``2 R sin(π k / n)`` apart.  Setting the ``k = 1``
        chord to ``spacing`` fixes the radius::

            R = spacing / (2 sin(π / n))

        so every atom has both ring neighbours at ``spacing``, including atoms
        ``n − 1`` and ``0``.  For ``n = 2`` this is two atoms ``spacing``
        apart (``R = spacing / 2``).  For ``n = 1`` there is no chord and the
        formula has no finite value; the single atom is placed at the origin.

        Parameters
        ----------
        n_atoms:
            Number of atoms, at least 1.
        spacing:
            Distance between neighbouring atoms in µm (the chord, not the
            arc), finite and positive.

        Raises
        ------
        ValueError
            If ``n_atoms`` is not an integer ``>= 1`` or ``spacing`` is not a
            finite number ``> 0``.
        """
        n = _atom_count(n_atoms)
        a = _finite_number(spacing, "spacing", positive=True)
        if n == 1:
            return cls(torch.zeros(1, 2, dtype=torch.float64))
        radius = a / (2 * math.sin(math.pi / n))
        theta = 2 * math.pi * torch.arange(n, dtype=torch.float64) / n
        return cls(radius * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1))

    @classmethod
    def grid(cls, rows: int, cols: int, spacing: float) -> AtomRegister:
        """
        A rectangular grid of ``rows × cols`` atoms on a square lattice of side ``spacing``.

        Atoms are counted row-major: atom ``k = r · cols + c``, for row
        ``0 <= r < rows`` and column ``0 <= c < cols``, sits at::

            r_k = (c · spacing, r · spacing)

        so row 0 runs along the positive x axis from the origin, atoms
        ``0, …, cols − 1``, and each further row lies one ``spacing`` higher
        in y.  Atoms ``k`` and ``l`` with ``(r, c) = divmod(k, cols)`` are
        ``spacing · √(Δr² + Δc²)`` apart, so with ``V = C6 / spacing⁶`` the
        nearest-neighbour coupling,

        ::

            V_kl = V / (Δr² + Δc²)³

        Each atom has its nearest neighbours (``V``) left, right, above and
        below: ``k ± 1`` in the same row and ``k ± cols``; two at a corner,
        three on an edge, four inside.  The next-nearest are the diagonals at
        ``√2 · spacing``, with ``V / 8``.  ``grid(1, n, a)`` is
        :meth:`chain` ``(n, a)``.

        Parameters
        ----------
        rows, cols:
            Number of rows and of columns, each at least 1.
        spacing:
            Distance between neighbouring atoms of a row or of a column in
            µm, finite and positive.

        Raises
        ------
        ValueError
            If ``rows`` or ``cols`` is not an integer ``>= 1`` or ``spacing``
            is not a finite number ``> 0``.

        Examples
        --------
        >>> AtomRegister.grid(2, 2, spacing=5.0).positions.tolist()
        [[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]
        """
        n_rows = _atom_count(rows, "rows")
        n_cols = _atom_count(cols, "cols")
        a = _finite_number(spacing, "spacing", positive=True)
        k = torch.arange(n_rows * n_cols)
        x = (k % n_cols).to(torch.float64) * a
        y = (k // n_cols).to(torch.float64) * a
        return cls(torch.stack([x, y], dim=1))

    @property
    def positions(self) -> torch.Tensor:
        """A copy of the positions: ``float64``, shape ``(n_atoms, 2)``, in µm."""
        return self._positions.clone()

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return self._positions.shape[0]

    def interaction_matrix(self, c6: float) -> torch.Tensor:
        """
        The van der Waals couplings ``V_ij = C6 / |r_i − r_j|⁶`` of every pair.

        With ``d_ij = r_i − r_j`` the squared distance is
        ``|d_ij|² = d_ij,x² + d_ij,y²``, and ``|d_ij|⁶ = (|d_ij|²)³``, which is
        how it is computed (no square root).  ``d_ij = −d_ji`` enters only
        squared, so the matrix is symmetric exactly, not just to rounding.  The
        diagonal is set to 0: an atom does not interact with itself, and the
        Hamiltonian sums over pairs ``i < j`` only.

        Parameters
        ----------
        c6:
            The ``C6`` coefficient in rad/µs · µm⁶.  Positive is repulsive
            (it raises the energy of a doubly excited pair), as for
            :data:`~hqnn_forge.rydberg.DEFAULT_C6`; the sign is kept.

        Returns
        -------
        torch.Tensor
            Shape ``(n_atoms, n_atoms)``, ``float64``, in rad/µs: entry
            ``[i, j]`` couples atoms ``i`` and ``j``.

        Raises
        ------
        ValueError
            If ``c6`` is not a finite number, or a coupling is not finite
            because two atoms are too close for ``float64``.
        """
        c6 = _finite_number(c6, "c6")
        difference = self._positions[:, None, :] - self._positions[None, :, :]
        squared = (difference**2).sum(dim=-1)
        diagonal = torch.eye(self.n_atoms, dtype=torch.bool)
        # 1 on the diagonal in place of 0, so that no 0/0 is formed there.
        v = (c6 / squared.masked_fill(diagonal, 1.0) ** 3).masked_fill(diagonal, 0.0)
        if not bool(torch.isfinite(v).all()):
            i, j = (int(k) for k in (~torch.isfinite(v)).nonzero()[0])
            raise ValueError(
                f"the interaction c6 / r^6 is not finite for atoms {i} and {j}, "
                f"{math.sqrt(float(squared[i, j])):.3g} µm apart (c6={c6!r})."
            )
        return v

    @staticmethod
    def blockade_radius(c6: float, omega: float) -> float:
        """
        The blockade radius ``R_b = (C6/Ω)^(1/6)``, in µm.

        It is the distance at which the coupling equals the Rabi frequency:
        ``C6 / R_b⁶ = Ω``.  Closer pairs have ``V_ij > Ω`` and cannot both be
        excited by a resonant drive; so ``V/Ω = (R_b/a)⁶`` for a pair at
        distance ``a``.  The radius depends on the atom and the drive, not on
        the positions, so it can be called on the class as well as on a
        register.  For :data:`~hqnn_forge.rydberg.DEFAULT_C6` and
        ``Ω = 4π rad/µs`` it is 8.69 µm.

        Parameters
        ----------
        c6:
            The ``C6`` coefficient in rad/µs · µm⁶, positive.
        omega:
            The Rabi frequency Ω in rad/µs, positive (the coefficient of
            ``X`` in the Hamiltonian is ``Ω/2``).

        Returns
        -------
        float
            ``R_b`` in µm.

        Raises
        ------
        ValueError
            If ``c6`` or ``omega`` is not a finite number ``> 0``.
        """
        c6 = _finite_number(c6, "c6", positive=True)
        omega = _finite_number(omega, "omega", positive=True)
        return (c6 / omega) ** (1 / 6)

    def _blockaded(self, blockade_radius: float) -> torch.Tensor:
        """
        Boolean ``(n_atoms, n_atoms)`` matrix of ``|r_i − r_j| < R_b`` for ``i ≠ j``.

        Compared as ``|r_i − r_j|² < R_b²``, without a square root, so a
        distance that is exactly ``R_b`` (``5 µm`` against ``R_b = 5``, whose
        squares are exact in ``float64``) falls outside, as the strict
        convention says.
        """
        radius = _finite_number(blockade_radius, "blockade_radius", positive=True)
        difference = self._positions[:, None, :] - self._positions[None, :, :]
        squared = (difference**2).sum(dim=-1)
        inside = squared < radius**2
        inside.fill_diagonal_(False)
        return inside

    def interaction_graph(self, blockade_radius: float) -> list[tuple[int, int]]:
        """
        The edges of the unit-disk graph at radius ``R_b``: every pair with ``|r_i − r_j| < R_b``.

        These are the pairs a blockade CZ can couple.  The comparison is
        **strict**: a pair exactly ``R_b`` apart has ``V_ij = C6 / R_b⁶ = Ω``
        (:meth:`blockade_radius`), the edge of the blockade, not inside it, so
        it is not an edge.  Equivalently, the edges are the pairs with
        ``V_ij > Ω`` for ``R_b = (C6/Ω)^(1/6)``.

        Parameters
        ----------
        blockade_radius:
            ``R_b`` in µm, a finite number ``> 0``; for instance
            ``AtomRegister.blockade_radius(c6, omega)``.

        Returns
        -------
        list of tuple of int
            The edges ``(i, j)`` with ``i < j``, sorted lexicographically.
            Atom indices are the rows of :attr:`positions` and the PennyLane
            wires of the gate-based circuits.

        Raises
        ------
        ValueError
            If ``blockade_radius`` is not a finite number ``> 0``.

        Examples
        --------
        >>> AtomRegister.chain(3, spacing=5.0).interaction_graph(6.0)
        [(0, 1), (1, 2)]
        """
        inside = torch.triu(self._blockaded(blockade_radius), diagonal=1)
        return [(int(i), int(j)) for i, j in inside.nonzero().tolist()]

    def blockade_triangles(self, blockade_radius: float) -> list[tuple[int, int, int]]:
        """
        The triangles of :meth:`interaction_graph`: triples of atoms pairwise closer than ``R_b``.

        Three atoms that blockade each other pairwise support a native CCZ
        (Levine et al. 2019, PRL 123, 170503).  The same strict ``< R_b``
        convention applies to each of the three pairs.

        Returns
        -------
        list of tuple of int
            The triples ``(i, j, k)`` with ``i < j < k``, sorted
            lexicographically.

        Raises
        ------
        ValueError
            If ``blockade_radius`` is not a finite number ``> 0``.
        """
        inside = self._blockaded(blockade_radius).tolist()
        n = self.n_atoms
        return [
            (i, j, k)
            for i in range(n)
            for j in range(i + 1, n)
            if inside[i][j]
            for k in range(j + 1, n)
            if inside[i][k] and inside[j][k]
        ]

    def device_violations(self, limits: DeviceLimits) -> list[str]:
        """
        The register bounds of a device these positions break, as messages naming each; empty if none.

        Checked, in the field order of
        :class:`~hqnn_forge.rydberg.DeviceLimits`, its two register fields
        (the pulse fields are the encoding's,
        :meth:`~hqnn_forge.rydberg.GlobalDrivePulseEncoding.device_violations`),
        in µm, the unit of :attr:`positions` and of the limits:

        * ``min_spacing``: the distance of the closest pair::

              d_min = min_{i<j} |r_i − r_j|,   |r_i − r_j|² = (x_i − x_j)² + (y_i − y_j)²

          broken when ``d_min < min_spacing``.  The squared distances are
          computed as in :meth:`interaction_matrix` and compared before the
          one square root, which is increasing, so the pair with the
          smallest square is the closest.  The message names it as atoms
          ``i`` and ``j``, ``i < j``, rows of :attr:`positions`; of pairs
          whose squared distances are equal in ``float64`` the first in
          lexicographic order (the bonds of :meth:`chain`, ``i · a`` apart
          in ``x``, can differ by rounding and are then not equal).  One
          atom has no pair and never breaks it.
        * ``region = (W, H)``: the extent along each axis::

              w = max_i x_i − min_i x_i,        h = max_i y_i − min_i y_i

          which is the largest difference between the ``x`` coordinates of
          two atoms (``max_ij (x_i − x_j)``), and in ``y``, the definition
          of the Braket device schema.  Broken when ``w > W`` (the width,
          along ``x``) or ``h > H`` (the height, along ``y``), one message
          for each.  Both are differences, so a shift of the register
          changes nothing; a rotation does, and the register is checked as
          given, in its own axes.

        A value equal to its limit passes, as in the encoding's check.  Each
        message starts with the field name and a colon.

        **These are the nominal positions.**  A device's register bounds are
        on the sites a program asks for; the shot-to-shot spread about them
        (``position_jitter`` of
        :class:`~hqnn_forge.rydberg.RydbergFeatureMap`) is the device's own
        and not a setting, so a map with jitter is checked through its
        :attr:`~hqnn_forge.rydberg.RydbergFeatureMap.register`, never its
        displaced realisations.  The check is never run by a map itself.

        Parameters
        ----------
        limits:
            The bounds to check against, such as
            :data:`~hqnn_forge.rydberg.AQUILA_LIMITS`.  Required, as ``c6``
            is for :meth:`interaction_matrix`: the register holds positions
            and nothing else.

        Examples
        --------
        >>> from hqnn_forge.rydberg import AQUILA_LIMITS
        >>> AtomRegister.chain(6, spacing=4.03).device_violations(AQUILA_LIMITS)
        []
        >>> AtomRegister.chain(6, spacing=16.0).device_violations(AQUILA_LIMITS)
        ['region: max x − min x = 80 µm exceeds the width 75 µm']
        """
        positions = self._positions
        violations = []
        n = self.n_atoms
        if n > 1:
            difference = positions[:, None, :] - positions[None, :, :]
            squared = (difference**2).sum(dim=-1)
            # Only the pairs i < j; argmin of the row-major flattening returns the first.
            pairs = torch.ones(n, n, dtype=torch.bool).triu(diagonal=1)
            closest = int(torch.argmin(squared.masked_fill(~pairs, math.inf)))
            i, j = divmod(closest, n)
            distance = math.sqrt(float(squared[i, j]))
            if distance < limits.min_spacing:
                violations.append(
                    f"min_spacing: min |r_i − r_j| = {distance:.6g} µm (atoms {i} and {j}) "
                    f"is below {limits.min_spacing:.6g} µm"
                )
        width, height = limits.region
        extent = positions.max(dim=0).values - positions.min(dim=0).values
        for axis, value, limit, side in (
            ("x", extent[0], width, "width"),
            ("y", extent[1], height, "height"),
        ):
            if float(value) > limit:
                violations.append(
                    f"region: max {axis} − min {axis} = {float(value):.6g} µm exceeds the "
                    f"{side} {limit:.6g} µm"
                )
        return violations

    def __repr__(self) -> str:
        return f"AtomRegister(positions={self._positions.tolist()})"
