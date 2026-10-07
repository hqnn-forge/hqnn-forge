"""
hqnn_forge.circuits.hardware_efficient
=========================================
Hardware-efficient ansatz primitive (lower CNOT depth alternative).

Applies one block of:

    1. CZ nearest-neighbour ladder: CZ(i, i+1) for i in 0 … n-2.
    2. Per-qubit single-parameter rotation: RY(θ_i) for all i.

Compared to the strongly-entangling layer this ansatz:

* Uses a third of the parameters (1 angle per qubit vs. 3)
* Has CNOT depth O(n) vs O(n)  (same depth, but CZ on real devices is often
  native and cheaper than CNOT)
* Is better suited for near-term trapped-ion / superconducting devices with
  limited connectivity

Usage
-----
Same as :func:`strongly_entangling_layer` — call inside a QNode context.

References
----------
* Kandala et al. (2017) "Hardware-efficient variational quantum eigensolver for
  small molecules", Nature 549, 242–246.
"""

from __future__ import annotations

import pennylane as qml
import torch


def hardware_efficient_layer(
    weights: torch.Tensor,
    n_qubits: int,
) -> None:
    """
    Apply one hardware-efficient block to the current quantum circuit.

    Gate count per call:

    * n_qubits - 1 CZ gates  (nearest-neighbour ladder)
    * n_qubits RY gates       (1 trainable parameter each)

    Parameters
    ----------
    weights:
        Rotation angles tensor of shape ``(n_qubits,)``.
        ``weights[i]`` is the RY angle θ_i applied to qubit i.
    n_qubits:
        Number of qubits.  ``weights`` must have shape ``(n_qubits,)``.

    Raises
    ------
    ValueError
        If ``weights.shape != (n_qubits,)``.  A ``(n_qubits, 3)`` row of a
        ``Rot`` weight tensor would otherwise broadcast ``RY`` silently.
    """
    if weights.shape != (n_qubits,):
        raise ValueError(
            f"weights must have shape (n_qubits={n_qubits},); got {tuple(weights.shape)}."
        )

    # Nearest-neighbour CZ ladder
    for qubit in range(n_qubits - 1):
        qml.CZ(wires=[qubit, qubit + 1])

    # Single-axis RY rotation block
    for qubit in range(n_qubits):
        qml.RY(weights[qubit], wires=qubit)
