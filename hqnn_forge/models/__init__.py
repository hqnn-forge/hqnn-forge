"""
hqnn_forge.models
=================
Full hybrid quantum-classical architectures.

Exported symbols
----------------
ClassifierBase            nn.Module base of every classifier: get_config, count_parameters.
BinaryClassifierBase      ClassifierBase + sigmoid predict_proba / thresholded predict.
HybridBinaryClassifier    Linear encoder → QuantumEncodingLayer → Linear head.
ParallelHybridClassifier  classical MLP branch ‖ QuantumEncodingLayer branch → Linear head.
MulticlassHybridClassifier  Linear encoder → QuantumEncodingLayer → n_classes heads (softmax / OvR).
ClassicalBaseline         Plain MLP with the same interface: the classical control of an ablation.
"""

from hqnn_forge.models.base import BinaryClassifierBase, ClassifierBase
from hqnn_forge.models.classical_baseline import ClassicalBaseline
from hqnn_forge.models.hybrid_classifier import HybridBinaryClassifier
from hqnn_forge.models.multiclass_hybrid_classifier import MulticlassHybridClassifier
from hqnn_forge.models.parallel_hybrid_classifier import ParallelHybridClassifier

__all__: list[str] = [
    "BinaryClassifierBase",
    "ClassicalBaseline",
    "ClassifierBase",
    "HybridBinaryClassifier",
    "MulticlassHybridClassifier",
    "ParallelHybridClassifier",
]
