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
  ``V_ij ≥ Ω`` exactly when ``|r_i − r_j| ≤ R_b``, that unit-disk graph is the
  set of pairs with ``V_ij ≥ Ω``: the two views are the same geometry.

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
from typing import Any

import numpy as np
import torch

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


def _atom_count(n_atoms: Any) -> int:
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, numbers.Integral) or n_atoms < 1:
        raise ValueError(f"n_atoms must be an integer >= 1; got {n_atoms!r}.")
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
        real-valued array-like; it is copied and stored as ``float64``.

    Raises
    ------
    TypeError
        If ``positions`` is complex or boolean.
    ValueError
        If ``positions`` does not have shape ``(n_atoms, 2)`` with at least
        one atom, holds a NaN or an infinity, or places two atoms at the same
        point (their coupling ``C6 / 0`` would be infinite).

    Notes
    -----
    Only exact duplicates are refused.  How close two atoms may sensibly be is
    a statement about the atom, not the geometry: the ``C6 / r⁶`` form is a
    long-distance one, and ``docs/rydberg-model.md`` does not use it below
    4 µm.  The register has no upper limit on the number of atoms; the dense
    Hamiltonian has (:data:`~hqnn_forge.rydberg.MAX_ATOMS`).

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

    def __repr__(self) -> str:
        return f"AtomRegister(positions={self._positions.tolist()})"
