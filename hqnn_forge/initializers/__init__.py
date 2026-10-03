"""
hqnn_forge.initializers
=======================
Small-angle (restricted-variance) weight initialisation for variational quantum circuits.

Exported symbols
----------------
restricted_normal_init_     In-place initialiser; variance scaled as σ² ∝ 1/(n_qubits * n_layers).
block_local_init_           restricted_normal_init_ tapered by layer: σ_ℓ² ∝ 1/(n_qubits * (n_layers + ℓ)).
"""

from hqnn_forge.initializers.restricted_variance import (
    block_local_init_,
    restricted_normal_init_,
)

__all__: list[str] = [
    "block_local_init_",
    "restricted_normal_init_",
]
