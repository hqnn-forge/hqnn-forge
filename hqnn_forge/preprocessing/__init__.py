"""
hqnn_forge.preprocessing
========================
Classical pre-processing pipeline — PCA dimensionality reduction and feature
standardisation — implemented in pure NumPy (no scikit-learn runtime dependency).

Exported symbols
----------------
PCANormalizer     Fits/applies PCA + per-feature standardisation; returns torch.Tensor.
stratified_kfold  Fold indices that keep class prevalence.
smote             SMOTE oversampling of a binary minority class.
oversample_fold   SMOTE one fold's training rows only; validation rows untouched.
iter_folds        Stratified folds with fold-safe oversampling.
Fold, SmoteResult  Result types of the above.
"""

from hqnn_forge.preprocessing.cv import (
    Fold,
    SmoteResult,
    iter_folds,
    oversample_fold,
    smote,
    stratified_kfold,
)
from hqnn_forge.preprocessing.pca_normalizer import PCANormalizer

__all__: list[str] = [
    "Fold",
    "PCANormalizer",
    "SmoteResult",
    "iter_folds",
    "oversample_fold",
    "smote",
    "stratified_kfold",
]
