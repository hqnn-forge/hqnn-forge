"""
hqnn_forge.training
===================
A minimal, model-agnostic train/validate loop with early stopping.

Exported symbols
----------------
train_model       Mini-batch training with per-epoch validation and early stopping.
TrainingHistory   Per-epoch record returned by train_model.
EpochRecord       One epoch of that record.
SPSA              Gradient-free optimiser: two loss evaluations per step, for shot-based training.
"""

from hqnn_forge.training.spsa import SPSA
from hqnn_forge.training.trainer import EpochRecord, TrainingHistory, train_model

__all__: list[str] = [
    "SPSA",
    "EpochRecord",
    "TrainingHistory",
    "train_model",
]
