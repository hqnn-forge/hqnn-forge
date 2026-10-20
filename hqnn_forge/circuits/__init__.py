"""
hqnn_forge.circuits
===================
Reusable, device-agnostic VQC ansatz primitives.

These functions are *pure quantum functions* — they are intended to be called
**inside** a QNode context (i.e. called within a function decorated with
``@qml.qnode``).  They apply in-place gate sequences and return nothing.

Exported symbols
----------------
strongly_entangling_layer   CNOT ring + per-qubit Rot(φ, θ, ω) block.
hardware_efficient_layer    CZ ladder + per-qubit RY(θ) block (lower CNOT depth).
neutral_atom_layer          Blockade CZ (optionally CCZ) on an atom register's interaction graph + Rot.
blockade_schedule           The parallel layers of those blockade gates for a register.
rydberg_chain               The register of entangler="rydberg": an open chain of n_qubits atoms.
RYDBERG_SPACING             Its atom spacing, 6 µm.
RYDBERG_BLOCKADE_RADIUS     Its blockade radius, (DEFAULT_C6 / 4π)^(1/6) = 8.69 µm.

These are the blocks the encoding layers run, selected by their ``entangler``
argument through :func:`hqnn_forge.encoding.angle_embedding.apply_variational_layers`:

* ``entangler="ring"`` (the default) is ``strongly_entangling_layer`` per layer.
  Despite the name, it is *not* ``entangler="strongly_entangling"``, which is
  PennyLane's ``qml.StronglyEntanglingLayers`` (Rot first, then a CNOT ring of
  growing range).
* ``entangler="hardware_efficient"`` is ``hardware_efficient_layer`` per layer.
* ``entangler="rydberg"`` is ``neutral_atom_layer`` per layer on
  ``rydberg_chain(n_qubits)`` at ``RYDBERG_BLOCKADE_RADIUS``: a chain whose
  only edges are neighbours, so no triangle and no CCZ.  Any other
  :class:`~hqnn_forge.rydberg.AtomRegister`, and ``ccz=True``, are for
  circuits that call ``neutral_atom_layer`` themselves (#418).
"""

from hqnn_forge.circuits.hardware_efficient import hardware_efficient_layer
from hqnn_forge.circuits.neutral_atom import (
    RYDBERG_BLOCKADE_RADIUS,
    RYDBERG_SPACING,
    blockade_schedule,
    neutral_atom_layer,
    rydberg_chain,
)
from hqnn_forge.circuits.strongly_entangling import strongly_entangling_layer

__all__: list[str] = [
    "RYDBERG_BLOCKADE_RADIUS",
    "RYDBERG_SPACING",
    "blockade_schedule",
    "hardware_efficient_layer",
    "neutral_atom_layer",
    "rydberg_chain",
    "strongly_entangling_layer",
]
