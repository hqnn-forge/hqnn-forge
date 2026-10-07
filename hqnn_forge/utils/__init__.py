"""
hqnn_forge.utils
================
Imbalance-robust losses, checkpoint save/load, quantum-layer ablation and
module-mode helpers.

Exported symbols
----------------
FocalLoss               nn.Module implementing Focal Loss (Lin et al. 2017).
SoftmaxFocalLoss        Its K-class softmax version.
weighted_bce_loss       Functional helper: inverse-class-frequency weighted BCE.
compute_class_weights   Computes inverse-frequency class weights from a label tensor.
eval_mode               Context manager: eval mode for a block, submodule modes restored.
train_mode              Context manager: train mode for a block, frozen submodules left frozen.
save_checkpoint         Write a classifier's class, constructor arguments and weights.
load_checkpoint         Rebuild a classifier from such a file.
disable_quantum_layer   Context manager: replace the quantum layer's output with a constant.
classical_baseline      Untrained MLP matched in parameter count: the classical control.
permute_quantum_layer   Context manager: shuffle the quantum layer's output across the batch.
"""

from hqnn_forge.utils.ablation import (
    classical_baseline,
    disable_quantum_layer,
    permute_quantum_layer,
)
from hqnn_forge.utils.checkpoint import load_checkpoint, save_checkpoint
from hqnn_forge.utils.imbalance import (
    FocalLoss,
    SoftmaxFocalLoss,
    compute_class_weights,
    weighted_bce_loss,
)
from hqnn_forge.utils.modes import eval_mode, train_mode

__all__: list[str] = [
    "FocalLoss",
    "SoftmaxFocalLoss",
    "classical_baseline",
    "compute_class_weights",
    "disable_quantum_layer",
    "eval_mode",
    "load_checkpoint",
    "permute_quantum_layer",
    "save_checkpoint",
    "train_mode",
    "weighted_bce_loss",
]
