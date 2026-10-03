"""
hqnn_forge.utils.imbalance
============================
Imbalance-robust loss functions for heavily skewed binary classification.

In fraud detection datasets the positive class (fraud) often represents
< 0.1 % of all samples.  Standard BCE quickly degenerates to predicting the
majority class.  The two strategies implemented here are:

1. **Focal Loss** (Lin et al. 2017):  down-weights easy negatives so the model
   focuses training signal on hard, rare positives.

2. **Weighted BCE** (King & Zeng 2001): scales the per-sample loss by the
   inverse class frequency, giving rare events proportionally more influence.

Classes
-------
FocalLoss           nn.Module — Focal Loss with configurable α and γ.

Functions
---------
weighted_bce_loss       Functional interface for inverse-frequency BCE.
compute_class_weights   Compute inverse-frequency weights from a label tensor.

References
----------
* Lin, T.-Y., et al. (2017) "Focal Loss for Dense Object Detection",
  ICCV 2017.  arXiv:1708.02002.
* King, G. & Zeng, L. (2001) "Logistic Regression in Rare Events Data",
  Political Analysis 9(2), 137–163.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification on imbalanced data.

    Focal Loss modifies the standard BCE by adding a weighting factor
    ``(1 - p_t)^γ`` that down-weights well-classified (easy) examples
    and focuses learning on hard, misclassified examples:

        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    where:

    * ``p_t = p``   if ``y = 1`` (positive class probability)
    * ``p_t = 1-p`` if ``y = 0``
    * ``α_t = α``   if ``y = 1``, else ``α_t = 1 - α``

    Parameters
    ----------
    alpha:
        Weighting factor for the positive class ∈ (0, 1).
        Set higher for stronger minority-class emphasis.  Default: 0.25.
    gamma:
        Focusing parameter γ ≥ 0.  γ=0 recovers standard BCE.
        Typical values: 1.0 – 5.0.  Default: 2.0.
    reduction:
        ``"mean"`` | ``"sum"`` | ``"none"``.  Default: ``"mean"``.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.utils import FocalLoss
    >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    >>> logits = torch.tensor([0.8, -0.3, 1.2, -1.5])
    >>> targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    >>> loss = loss_fn(logits, targets)
    >>> round(loss.item(), 4)  # scalar
    0.0232
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
        if gamma < 0.0:
            raise ValueError(f"gamma must be ≥ 0; got {gamma}.")
        _check_reduction(reduction)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the focal loss.

        Parameters
        ----------
        logits:
            Raw model outputs (before sigmoid), shape ``(N,)`` or ``(N, 1)``.
        targets:
            Binary ground-truth labels (0.0 or 1.0), same shape as *logits*.

        Returns
        -------
        torch.Tensor
            Scalar loss if ``reduction != "none"``, else per-sample tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Binary cross-entropy (numerically stable via log-sum-exp)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: probability assigned to the correct class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # α_t: class-aware alpha weight
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal modulation
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss  # "none"

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction!r}"


# ---------------------------------------------------------------------------
# Inverse-frequency weighted BCE (functional)
# ---------------------------------------------------------------------------


class SoftmaxFocalLoss(nn.Module):
    """
    Focal loss for ``K``-class softmax classification (Lin et al. 2017).

    ``FL = −(1 − p_t)^γ · log p_t`` with ``p_t`` the softmax probability of the
    true class: cross-entropy that down-weights well-classified samples.
    ``γ = 0`` is ``nn.CrossEntropyLoss`` exactly.  (For one-vs-rest heads, the
    binary :class:`FocalLoss` applied to one-hot targets is the per-class
    version.)

    Parameters
    ----------
    gamma:
        Focusing parameter, ``≥ 0``.  Default: 2.0.

    Forward
    -------
    ``logits`` of shape ``(N, K)`` and integer class indices ``(N,)``; returns
    the mean loss.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"gamma must be ≥ 0; got {gamma}.")
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1).gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
        return (-((1 - log_p.exp()) ** self.gamma) * log_p).mean()


def compute_class_weights(
    labels: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for a binary label tensor.

    The weight for class c is:

        w_c = N / (2 * n_c + smooth)

    where N is total samples and n_c is the count of class c.
    The ``smooth`` term avoids zero division for classes absent in a mini-batch.

    Parameters
    ----------
    labels:
        Integer or float label tensor of shape ``(N,)`` containing 0s and 1s.
    smooth:
        Additive smoothing term.  Default: 1.0.

    Returns
    -------
    torch.Tensor
        1D tensor ``[w_neg, w_pos]`` — weights for class 0 and class 1.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.utils import compute_class_weights
    >>> y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 9:1 imbalance
    >>> compute_class_weights(y)  # [10 / (2*9 + 1), 10 / (2*1 + 1)]
    tensor([0.5263, 3.3333])
    >>> compute_class_weights(y, smooth=0.0)  # [10 / (2*9), 10 / (2*1)]
    tensor([0.5556, 5.0000])
    """
    labels = labels.view(-1).float()
    n_total = labels.numel()
    n_pos = labels.sum().item()
    n_neg = n_total - n_pos

    w_pos = n_total / (2.0 * n_pos + smooth)
    w_neg = n_total / (2.0 * n_neg + smooth)

    return torch.tensor([w_neg, w_pos], dtype=torch.float32)


def _check_reduction(reduction: str) -> None:
    """Refuse a *reduction* other than ``"mean"``, ``"sum"`` or ``"none"``."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}.")


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary cross-entropy loss with per-sample class weighting.

    Parameters
    ----------
    logits:
        Raw model outputs (pre-sigmoid), shape ``(N,)`` or ``(N, 1)``.
    targets:
        Binary labels (0.0 / 1.0), same shape as *logits*.
    class_weights:
        Tensor ``[w_neg, w_pos]`` from :func:`compute_class_weights`.
    reduction:
        ``"mean"`` | ``"sum"`` | ``"none"``.

    Returns
    -------
    torch.Tensor
        Weighted loss, scalar or per-sample depending on *reduction*.

    Raises
    ------
    ValueError
        If *reduction* is not one of the three values above.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.utils import compute_class_weights, weighted_bce_loss
    >>> y    = torch.tensor([0.0, 0.0, 0.0, 1.0])
    >>> pred = torch.tensor([0.1, -0.2, 0.05, 0.9])
    >>> cw   = compute_class_weights(y)
    >>> weighted_bce_loss(pred, y, cw)
    tensor(0.4081)
    """
    _check_reduction(reduction)
    logits = logits.view(-1)
    targets = targets.view(-1).float()

    # Map each sample's label to its class weight
    sample_weights = class_weights[0] * (1.0 - targets) + class_weights[1] * targets

    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted = loss * sample_weights

    if reduction == "mean":
        return weighted.mean()
    elif reduction == "sum":
        return weighted.sum()
    return weighted
