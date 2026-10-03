"""
hqnn_forge.evaluation.multiclass
================================
Multiclass versions of the monitored metrics, on hard labels.

``train_model`` ranks epochs of a multiclass model by one of these (#309).
With more than two classes there is no single decision threshold to search,
so the metrics are computed on the argmax labels.

``multiclass_matthews_corrcoef``
    Gorodkin's ``R_K`` (2004), the generalisation of the MCC to ``K``
    classes: from the confusion matrix ``C`` with row sums ``t_k`` (true),
    column sums ``p_k`` (predicted), trace ``c`` and total ``s``::

        R_K = (c·s − Σ_k p_k t_k) / sqrt((s² − Σ_k p_k²)(s² − Σ_k t_k²))

    It is 1 for a perfect prediction, 0 at chance level and for any constant
    prediction, and reduces to the binary MCC for ``K = 2``.  It is what
    ``sklearn.metrics.matthews_corrcoef`` computes for multiclass input.
``macro_f1_score``
    The unweighted mean of the per-class F1 scores over the classes that
    occur in ``y_true`` or ``y_pred``, as ``sklearn.metrics.f1_score(...,
    average="macro")``: a class present in one but not the other scores 0.
``multiclass_balanced_accuracy``
    The mean per-class recall over the classes present in ``y_true``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

__all__ = [
    "MULTICLASS_METRICS",
    "confusion_counts",
    "macro_f1_score",
    "multiclass_balanced_accuracy",
    "multiclass_matthews_corrcoef",
]


def confusion_counts(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> torch.Tensor:
    """``(n_classes, n_classes)`` float64 counts, rows true, columns predicted."""
    t = torch.as_tensor(y_true).long().reshape(-1)
    p = torch.as_tensor(y_pred).long().reshape(-1)
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred differ in length: {t.numel()} vs {p.numel()}.")
    if t.numel() == 0:
        raise ValueError("y_true is empty.")
    for name, v in (("y_true", t), ("y_pred", p)):
        if v.min() < 0 or v.max() >= n_classes:
            raise ValueError(f"{name} must hold class indices in [0, {n_classes}).")
    counts = torch.bincount(t * n_classes + p, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes).to(torch.float64)


def _n_classes(y_true: torch.Tensor, y_pred: torch.Tensor) -> int:
    t, p = torch.as_tensor(y_true), torch.as_tensor(y_pred)
    if t.numel() == 0 or p.numel() == 0:
        # confusion_counts gives the specific message; any count will do here.
        return 1
    return int(max(int(t.max()), int(p.max()))) + 1


def multiclass_matthews_corrcoef(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Gorodkin's ``R_K``; 0 when either marginal is constant (undefined)."""
    c = confusion_counts(y_true, y_pred, _n_classes(y_true, y_pred))
    s = c.sum()
    t, p = c.sum(dim=1), c.sum(dim=0)
    cov = torch.trace(c) * s - (p * t).sum()
    denom = torch.sqrt((s**2 - (p**2).sum()) * (s**2 - (t**2).sum()))
    return float(cov / denom) if denom > 0 else 0.0


def macro_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Unweighted mean of per-class F1 over the classes seen in either argument."""
    c = confusion_counts(y_true, y_pred, _n_classes(y_true, y_pred))
    denom = c.sum(dim=0) + c.sum(dim=1)  # 2·tp + fp + fn
    seen = denom > 0  # a class index neither true nor predicted is no class at all
    return float((2 * torch.diag(c)[seen] / denom[seen]).mean())


def multiclass_balanced_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean recall over the classes present in ``y_true``."""
    c = confusion_counts(y_true, y_pred, _n_classes(y_true, y_pred))
    support = c.sum(dim=1)
    present = support > 0
    return float((torch.diag(c)[present] / support[present]).mean())


#: The multiclass counterparts of :data:`hqnn_forge.evaluation.METRICS`, under
#: the same names, so ``monitor="mcc"`` means the same thing for both.
MULTICLASS_METRICS: Mapping[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "mcc": multiclass_matthews_corrcoef,
    "f1": macro_f1_score,
    "balanced_accuracy": multiclass_balanced_accuracy,
}
