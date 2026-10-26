"""
hqnn_forge.circuits.neutral_atom
================================
Neutral-atom entangler primitive: blockade CZ (and optionally CCZ) on the
interaction graph of an :class:`~hqnn_forge.rydberg.AtomRegister`, in parallel
layers, then a per-qubit ``Rot``.

Applies one block of:

    1. CZ on every edge of the unit-disk graph at the blockade radius ``R_b``
       (:meth:`~hqnn_forge.rydberg.AtomRegister.interaction_graph`), and with
       ``ccz=True`` a CCZ on every blockade triangle
       (:meth:`~hqnn_forge.rydberg.AtomRegister.blockade_triangles`), in the
       parallel layers of :func:`blockade_schedule`.
    2. Per-qubit SU(2) rotation: ``Rot(φ_i, θ_i, ω_i)`` on every atom.

Conventions
-----------
* **Wires.**  Atom ``i`` (row ``i`` of
  :attr:`~hqnn_forge.rydberg.AtomRegister.positions`) is PennyLane wire ``i``,
  as in :func:`~hqnn_forge.rydberg.rydberg_hamiltonian`.
* **Distance.**  Atoms ``i`` and ``j`` interact when ``|r_i − r_j| < R_b``,
  strictly: at ``|r_i − r_j| = R_b`` the coupling ``C6/R_b⁶`` equals Ω, the
  edge of the blockade rather than inside it.  Positions and ``R_b`` are in µm.
* **Order.**  Edges ``(i, j)`` have ``i < j`` and triangles ``(i, j, k)`` have
  ``i < j < k``; both are sorted lexicographically, and the schedule places
  them in that order, so it is a deterministic function of the positions.

Default register
----------------
``entangler="rydberg"`` in the encoders runs this block on a register that,
by default, depends on ``n_qubits`` alone (#418): :func:`rydberg_chain` at
:data:`RYDBERG_BLOCKADE_RADIUS`, the open chain
``AtomRegister.chain(n_qubits, RYDBERG_SPACING)``.  With ``a = 6 µm`` and
``R_b = 8.69 µm``, ``a < R_b ≤ 2a``: neighbours are blockaded
(``V/Ω = (R_b/a)⁶ = 9.25``) and next-nearest neighbours are not
(``V/Ω = 0.144``), so the interaction graph is the path
``0 — 1 — … — n−1`` and has no triangles.  Since the gates are ideal, every
spacing in ``[R_b/2, R_b)`` gives this graph and so the same circuit.  The
encoders, and the classifiers built on them, take another register, radius
and ``ccz`` through their ``atom_register``, ``blockade_radius`` and ``ccz``
arguments (#644); one register serves every layer of the stack.

The model is purely digital: each gate is the ideal CZ or CCZ, with no pulse,
no residual interaction and no atom motion.  Pulses and analog evolution live
in :mod:`hqnn_forge.rydberg`; shuttling atoms between layers is not modelled.

References
----------
* Jaksch et al. (2000) "Fast quantum gates for neutral atoms", PRL 85, 2208.
* Saffman, Walker & Mølmer (2010) "Quantum information with Rydberg atoms",
  Reviews of Modern Physics 82, 2313.
* Levine et al. (2019) "Parallel implementation of high-fidelity multiqubit
  gates with neutral atoms", PRL 123, 170503.
"""

from __future__ import annotations

import math

import pennylane as qml
import torch

from hqnn_forge.rydberg.hamiltonian import DEFAULT_C6
from hqnn_forge.rydberg.register import AtomRegister

__all__ = [
    "RYDBERG_BLOCKADE_RADIUS",
    "RYDBERG_SPACING",
    "blockade_schedule",
    "neutral_atom_layer",
    "rydberg_chain",
]

#: Blockade radius of ``entangler="rydberg"``, in µm: ``(C6/Ω)^(1/6)`` for
#: :data:`~hqnn_forge.rydberg.DEFAULT_C6` and the reference ``Ω = 4π rad/µs``
#: of ``docs/rydberg-model.md``, 8.69 µm.
RYDBERG_BLOCKADE_RADIUS: float = AtomRegister.blockade_radius(DEFAULT_C6, 4 * math.pi)
#: Atom spacing of :func:`rydberg_chain`, in µm: inside ``[R_b/2, R_b)``, so
#: only neighbours interact, and inside the 4 to 15 µm chains
#: ``docs/rydberg-model.md`` marks as supported.
RYDBERG_SPACING: float = 6.0

#: A scheduled gate: the atoms of a CZ ``(i, j)`` or a CCZ ``(i, j, k)``.
Gate = tuple[int, ...]


def _check_connected(edges: list[tuple[int, int]], n_atoms: int, radius: float) -> None:
    """``ValueError`` naming the atoms the graph does not connect to atom 0."""
    neighbours: list[list[int]] = [[] for _ in range(n_atoms)]
    for i, j in edges:
        neighbours[i].append(j)
        neighbours[j].append(i)
    reached = {0}
    frontier = [0]
    while frontier:
        atom = frontier.pop()
        for other in neighbours[atom]:
            if other not in reached:
                reached.add(other)
                frontier.append(other)
    if len(reached) < n_atoms:
        cut_off = [atom for atom in range(n_atoms) if atom not in reached]
        isolated = [atom for atom in range(n_atoms) if not neighbours[atom]]
        raise ValueError(
            f"the interaction graph at blockade_radius={radius!r} µm does not connect every "
            f"atom: atoms {cut_off} are not connected to atom 0 (isolated: {isolated}). "
            "Move the atoms closer or increase the blockade radius."
        )


def blockade_schedule(
    register: AtomRegister, blockade_radius: float, *, ccz: bool = False
) -> list[list[Gate]]:
    """
    The blockade gates of ``register``, grouped into layers that can run in parallel.

    Every edge of the interaction graph gets one CZ, and with ``ccz=True``
    every blockade triangle also gets one CCZ.  Layer ``ℓ`` holds the gates
    run at the same time in step ``ℓ``.

    The parallel rule
    -----------------
    Two gates ``g`` and ``h`` may share a layer exactly when every atom of
    ``g`` is at least ``R_b`` from every atom of ``h``::

        min_{a ∈ g, b ∈ h} |r_a − r_b| ≥ R_b.                          (*)

    Derivation.  A blockade CZ on ``(i, j)`` (Jaksch et al. 2000; Levine et
    al. 2019) drives both atoms between a computational state and the
    Rydberg state ``|r⟩``.  Its phase relies on ``|r_i r_j⟩`` being shifted
    by ``V_ij = C6/|r_i − r_j|⁶ > Ω``, i.e. ``|r_i − r_j| < R_b``, so that
    the pair cannot be doubly excited; that is why the gates sit on the
    edges of the unit-disk graph.  In the same idealisation, a pair with
    ``V ≤ Ω`` (distance ``≥ R_b``) does not interact at all.

    * An atom ``b`` of another gate ``h`` in the same layer is driven to
      ``|r⟩`` too.  If ``|r_a − r_b| < R_b`` for some ``a ∈ g``, then
      ``|r_a r_b⟩`` is blockaded as well: whether ``a`` can be excited, and
      so the phase ``g`` imprints, depends on the state of ``b``.  The two
      gates no longer act as ``g ⊗ h`` but as one entangling operation on
      ``g ∪ h``: the "third atom within ``R_b`` of a gate pair" that shifts
      the pair's blockade (Saffman, Walker & Mølmer 2010).  If (*) holds, no driven atom
      of ``h`` interacts with one of ``g``, and the layer is ``g ⊗ h``.
    * Gates sharing an atom violate (*) with ``a = b`` at distance 0, so (*)
      also forbids putting one atom in two gates of a layer.
    * An **idle** atom, one in no gate of the layer, is assumed to stay in
      its computational states, which have no Rydberg interaction: the drive
      is addressed to the gate atoms only.  So an idle atom within ``R_b`` of
      a gate does not block it.  (Under a global drive every atom within
      ``R_b`` of a pair would join the gate, and a chain of three atoms at
      spacing below ``R_b`` could run no CZ at all.)
    * A CCZ on a triangle ``(i, j, k)`` is the three-atom blockade gate
      (Levine et al. 2019); (*) applies to it with its three atoms.

    The schedule
    ------------
    The gates are taken in a fixed order: the edges sorted
    lexicographically, then (with ``ccz=True``) the triangles sorted
    lexicographically.  Each goes into the first layer where (*) holds
    against every gate already there, or opens a new layer.  This greedy
    colouring of the conflict graph is deterministic, but not always the
    fewest layers: the minimum is a graph-colouring problem.  The layer
    count is therefore an upper bound on the parallel depth of the device.

    The order of the gates does not change the circuit: CZ and CCZ are both
    diagonal in the computational basis, so they commute and the product
    over a block is the diagonal ``(-1)^{Σ_edges b_i b_j + Σ_triangles b_i
    b_j b_k}`` on basis state ``|b⟩`` whatever the schedule.  The schedule
    only fixes the depth.

    Parameters
    ----------
    register:
        The atom positions, in µm.
    blockade_radius:
        ``R_b`` in µm, a finite number ``> 0``, for instance
        ``AtomRegister.blockade_radius(c6, omega)``.
    ccz:
        Also schedule a CCZ on every blockade triangle.

    Returns
    -------
    list of list of tuple of int
        One list per layer, in time order; within a layer, the gates in the
        order they were placed.  A CZ is a pair ``(i, j)``, a CCZ a triple
        ``(i, j, k)``.  Empty for a single atom.

    Raises
    ------
    ValueError
        If ``blockade_radius`` is not a finite number ``> 0``, or the
        interaction graph does not connect every atom; the message names the
        atoms cut off from atom 0 and those with no neighbour at all.

    Examples
    --------
    >>> from hqnn_forge.rydberg import AtomRegister
    >>> blockade_schedule(AtomRegister.chain(5, spacing=5.0), 6.0)
    [[(0, 1), (3, 4)], [(1, 2)], [(2, 3)]]
    """
    edges = register.interaction_graph(blockade_radius)
    _check_connected(edges, register.n_atoms, float(blockade_radius))
    gates: list[Gate] = list(edges)
    if ccz:
        gates += register.blockade_triangles(blockade_radius)
    radius_sq = float(blockade_radius) ** 2
    positions = register.positions
    squared = ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(dim=-1).tolist()

    def compatible(g: Gate, h: Gate) -> bool:
        # (*) on squared distances; a shared atom has distance 0 < R_b².
        return all(squared[a][b] >= radius_sq for a in g for b in h)

    layers: list[list[Gate]] = []
    for gate in gates:
        for layer in layers:
            if all(compatible(gate, other) for other in layer):
                layer.append(gate)
                break
        else:
            layers.append([gate])
    return layers


def rydberg_chain(n_qubits: int) -> AtomRegister:
    """
    The register of ``entangler="rydberg"``: ``n_qubits`` atoms in an open chain.

    Atom ``i`` sits at ``(i · RYDBERG_SPACING, 0)``.  At
    :data:`RYDBERG_BLOCKADE_RADIUS` its interaction graph is the path
    ``(0, 1), (1, 2), …, (n−2, n−1)``, since ``a < R_b ≤ 2a``.

    Schedule.  Two chain CZs ``(i, i+1)`` and ``(j, j+1)``, ``i < j``, are
    closest at atoms ``i+1`` and ``j``, ``(j − i − 1)·a`` apart.  Rule (*) of
    :func:`blockade_schedule` lets them share a layer exactly when
    ``(j − i − 1)·a ≥ R_b``.  With ``a < R_b ≤ 2a`` that means
    ``j − i − 1 ≥ 2``, i.e. ``j ≥ i + 3``.  The greedy colouring, in edge
    order, therefore puts edge ``i`` in layer ``i mod 3``.  It needs
    ``min(3, n − 1)`` layers (none for one atom).  That is the minimum:
    any three consecutive edges conflict pairwise, and so do all edges when
    there are fewer than three.

    Raises
    ------
    ValueError
        If ``n_qubits`` is not an integer ``>= 1``.

    Examples
    --------
    >>> blockade_schedule(rydberg_chain(5), RYDBERG_BLOCKADE_RADIUS)
    [[(0, 1), (3, 4)], [(1, 2)], [(2, 3)]]
    """
    return AtomRegister.chain(n_qubits, RYDBERG_SPACING)


def neutral_atom_layer(
    weights: torch.Tensor,
    register: AtomRegister,
    blockade_radius: float,
    *,
    ccz: bool = False,
) -> None:
    """
    Apply one neutral-atom block to the current quantum circuit.

    The block is ``(⊗_i Rot(φ_i, θ_i, ω_i)) · ∏_{layers} ∏_{g ∈ layer} G_g``,
    with ``G_g`` the CZ of an edge or the CCZ of a triangle, applied in the
    order of :func:`blockade_schedule`.  Atom ``i`` is wire ``i``.

    Gate count per call, with ``E`` edges and ``T`` triangles:

    * ``E`` CZ gates, plus ``T`` CCZ gates with ``ccz=True``, in
      ``len(blockade_schedule(register, blockade_radius, ccz=ccz))`` parallel
      layers;
    * ``n_atoms`` ``Rot`` gates (3 trainable angles each).

    Readout light cone
    ------------------
    CZ and CCZ are diagonal, so they leave ``Z_i`` unchanged under
    conjugation and turn ``X_i`` (and ``Y_i``) into ``X_i`` times a diagonal
    operator on the neighbours of ``i`` (``∏_{j ~ i} Z_j`` for the CZs, a
    controlled phase on ``j, k`` for each CCZ triangle ``(i, j, k)``, whose
    atoms are neighbours of ``i`` already).  The ``Rot`` mixes ``Z_i`` with
    ``X_i`` and ``Y_i``.  In the Heisenberg picture one block therefore
    widens the support of a readout by one step of the interaction graph,
    and after ``L`` blocks on an angle-embedded product state ⟨Z_i⟩ sees the
    features of the atoms within graph distance ``L`` of atom ``i``, and no
    others.  Unlike ``hardware_efficient_layer`` (``RY`` only, which adds
    only ``X_i``, invisible under an RX embedding where ⟨X⟩ = 0), the ``Rot``
    also brings in ``Y_i`` (⟨Y⟩ = −sin x), so this holds from the first
    block under both RX and RY embeddings.  A readout of atom 0 alone
    covers the register only once ``L`` reaches the graph eccentricity of
    atom 0 (``n − 1`` for atom 0 at the end of a chain, 2 for a square).

    Inert parameters: in the last block the trailing ``RZ(ω_i)`` of each
    ``Rot`` commutes with a ⟨Z⟩ readout, as for the ``"ring"`` block, so
    those ``n_atoms`` angles cannot move the output.

    Parameters
    ----------
    weights:
        Rotation angles of shape ``(n_atoms, 3)``; row ``i`` is
        ``[φ_i, θ_i, ω_i]`` of ``Rot = RZ(ω) · RY(θ) · RZ(φ)`` on atom ``i``.
    register:
        The atom positions, in µm.
    blockade_radius:
        ``R_b`` in µm, a finite number ``> 0``.
    ccz:
        Also apply a CCZ on every blockade triangle.

    Raises
    ------
    ValueError
        If ``weights.shape != (n_atoms, 3)``, or for the reasons
        :func:`blockade_schedule` raises.
    """
    n_atoms = register.n_atoms
    if tuple(weights.shape) != (n_atoms, 3):
        raise ValueError(
            f"weights must have shape (n_atoms={n_atoms}, 3); got {tuple(weights.shape)}."
        )
    for layer in blockade_schedule(register, blockade_radius, ccz=ccz):
        for gate in layer:
            if len(gate) == 2:
                qml.CZ(wires=list(gate))
            else:
                qml.CCZ(wires=list(gate))
    for atom in range(n_atoms):
        qml.Rot(weights[atom, 0], weights[atom, 1], weights[atom, 2], wires=atom)
