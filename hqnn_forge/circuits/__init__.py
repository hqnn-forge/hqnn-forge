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

These are the blocks the encoding layers run, selected by their ``entangler``
argument through :func:`hqnn_forge.encoding.angle_embedding.apply_variational_layers`:

* ``entangler="ring"`` (the default) is ``strongly_entangling_layer`` per layer.
  Despite the name, it is *not* ``entangler="strongly_entangling"``, which is
  PennyLane's ``qml.StronglyEntanglingLayers`` (Rot first, then a CNOT ring of
  growing range).
* ``entangler="hardware_efficient"`` is ``hardware_efficient_layer`` per layer.
"""

from hqnn_forge.circuits.hardware_efficient import hardware_efficient_layer
from hqnn_forge.circuits.strongly_entangling import strongly_entangling_layer

__all__: list[str] = [
    "hardware_efficient_layer",
    "strongly_entangling_layer",
]
