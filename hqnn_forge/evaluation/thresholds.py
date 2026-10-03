"""
hqnn_forge.evaluation.thresholds
================================
Data-driven decision thresholds and score-per-parameter for binary classifiers.

``predict`` takes a ``threshold`` and leaves its choice to the caller.  On
imbalanced data the default of 0.5 is rarely the best operating point, and
the usual remedy -- sweep thresholds on a validation split and keep the one
that maximises a rank-insensitive metric such as MCC -- is what
:func:`find_optimal_threshold` does.

The metrics in :data:`METRICS` take hard labels and are computed from the
confusion matrix with plain tensor arithmetic, so nothing depends on
scikit-learn at runtime.  :func:`pr_auc` is the exception in kind, not in
dependencies: it is threshold-free, scored from the probabilities over every
operating point, and so is deliberately not in :data:`METRICS`.  Inputs may
be ``torch.Tensor`` or anything ``torch.as_tensor`` accepts.

Conventions
-----------
* A sample is predicted positive when ``probability >= threshold``, matching
  ``BinaryClassifierBase.predict``.
* Metrics that are undefined for a labelling (MCC with a single predicted or
  true class, F1 with no positives anywhere) return ``0.0`` rather than NaN,
  the same convention scikit-learn uses, so a threshold search never picks a
  NaN over a number.
* :func:`balanced_accuracy` departs from scikit-learn deliberately when a
  class is absent from ``y_true``: scikit-learn averages recall over the
  classes that are present, which scores an all-negative labelling of an
  all-negative split 1.0, while this one returns 0.0 so that a search cannot
  rank a degenerate operating point top.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch
import torch.nn as nn

Metric = Callable[[torch.Tensor, torch.Tensor], float]

#: A metric expressed on confusion counts, elementwise over a batch of labellings.
CountMetric = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Confusion-matrix metrics
# ---------------------------------------------------------------------------


def _as_binary(y: object, name: str) -> torch.Tensor:
    t = torch.as_tensor(y).reshape(-1)
    if t.dtype == torch.bool:
        t = t.long()
    if t.numel() and not torch.all((t == 0) | (t == 1)):
        raise ValueError(
            f"{name} must contain only 0/1 labels; got values {torch.unique(t).tolist()}."
        )
    return t.long()


def _confusion(y_true: object, y_pred: object) -> tuple[float, float, float, float]:
    """Return (tp, tn, fp, fn) as floats."""
    t = _as_binary(y_true, "y_true")
    p = _as_binary(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred differ in length: {t.numel()} vs {p.numel()}.")
    tp = float(((t == 1) & (p == 1)).sum())
    tn = float(((t == 0) & (p == 0)).sum())
    fp = float(((t == 0) & (p == 1)).sum())
    fn = float(((t == 1) & (p == 0)).sum())
    return tp, tn, fp, fn


def _guarded_div(num: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """``num / denom`` elementwise, 0.0 wherever ``denom`` is zero."""
    safe = torch.where(denom == 0, torch.ones_like(denom), denom)
    return torch.where(denom == 0, torch.zeros_like(num), num / safe)


# The count metrics below are written elementwise over tensors of confusion
# counts, so one call scores a whole sweep of candidate thresholds.  The public
# scalar metrics are thin wrappers around them, so both paths share a formula.


def _mcc_counts(
    tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
) -> torch.Tensor:
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).sqrt()
    return _guarded_div(tp * tn - fp * fn, denom)


def _f1_counts(
    tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
) -> torch.Tensor:
    return _guarded_div(2 * tp, 2 * tp + fp + fn)


def _balanced_accuracy_counts(
    tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
) -> torch.Tensor:
    positives, negatives = tp + fn, tn + fp
    defined = (positives > 0) & (negatives > 0)
    recall = 0.5 * (_guarded_div(tp, positives) + _guarded_div(tn, negatives))
    return torch.where(defined, recall, torch.zeros_like(recall))


def _scalar(counts_metric: CountMetric, y_true: object, y_pred: object) -> float:
    counts = [torch.tensor(c, dtype=torch.float64) for c in _confusion(y_true, y_pred)]
    return float(counts_metric(*counts))


def matthews_corrcoef(y_true: object, y_pred: object) -> float:
    """
    Matthews correlation coefficient in [-1, 1].

    ``(tp·tn − fp·fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))``.  Returns 0.0 when
    any factor in the denominator is zero (only one class present in
    ``y_true`` or in ``y_pred``), as scikit-learn does.
    """
    return _scalar(_mcc_counts, y_true, y_pred)


def f1_score(y_true: object, y_pred: object) -> float:
    """F1 of the positive class; 0.0 when there are no positives in either vector."""
    return _scalar(_f1_counts, y_true, y_pred)


def balanced_accuracy(y_true: object, y_pred: object) -> float:
    """
    Mean of the recall on each class; 0.5 is chance.

    Returns 0.0 when a class is absent from ``y_true`` and the mean is
    therefore taken over one class only.  scikit-learn returns that one
    class's recall instead (1.0 for a perfect labelling of a single-class
    split); 0.0 is used here so that :func:`find_optimal_threshold` cannot
    prefer such a labelling.
    """
    return _scalar(_balanced_accuracy_counts, y_true, y_pred)


#: Metrics selectable by name in :func:`find_optimal_threshold`.
METRICS: Mapping[str, Metric] = {
    "mcc": matthews_corrcoef,
    "f1": f1_score,
    "balanced_accuracy": balanced_accuracy,
}

#: The same metrics on confusion counts, used to score a whole sweep at once.
_COUNT_METRICS: Mapping[str, CountMetric] = {
    "mcc": _mcc_counts,
    "f1": _f1_counts,
    "balanced_accuracy": _balanced_accuracy_counts,
}


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------


class ThresholdSearchResult(NamedTuple):
    """Best threshold found and the metric value it achieves."""

    threshold: float
    score: float


class _Counts(NamedTuple):
    """Confusion counts per distinct labelling, in order of increasing threshold."""

    unique: torch.Tensor  # the sorted unique scores, one fewer than the labellings
    tp: torch.Tensor
    tn: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor


class _Sweep(NamedTuple):
    """One entry per distinct labelling the probabilities can produce."""

    threshold: torch.Tensor  # representative threshold, the midpoint of the interval
    tp: torch.Tensor
    tn: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor


def _validate_scores(y_true: object, y_prob: object) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Binary labels and float64 scores, both flattened, checked for agreement.

    Checks what every scorer here needs: non-empty, equal lengths, 0/1
    labels.  Which score values are admissible (finite, within [0, 1]) is
    the caller's to decide.
    """
    t = _as_binary(y_true, "y_true")
    p = torch.as_tensor(y_prob, dtype=torch.float64).reshape(-1)
    if t.numel() == 0:
        raise ValueError("y_true is empty; nothing can be scored from no samples.")
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_prob differ in length: {t.numel()} vs {p.numel()}.")
    return t, p


def _counts(t: torch.Tensor, p: torch.Tensor) -> _Counts:
    """
    Confusion counts for every distinct labelling, in O(n log n).

    Thresholds in ``(u[i-1], u[i]]`` all produce the same labelling, where
    ``u`` are the sorted unique scores, so there are ``len(u) + 1``
    labellings: one per unique value, plus the all-negative one above the
    maximum.  Sorting the labels once and taking a cumulative sum gives the
    counts for all of them without re-scanning the vector per candidate.
    Only the order of ``p`` matters, so any real-valued score works.
    """
    n = p.numel()
    unique, counts = torch.unique(p, return_counts=True)  # sorted ascending
    labels = t[torch.argsort(p, stable=True)].to(torch.float64)

    # starts[i] is how many samples fall strictly below candidate i's interval
    starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    below = torch.cat([torch.zeros(1, dtype=torch.float64), labels.cumsum(0)])[starts]

    fn = below  # positives below the threshold
    tn = starts.to(torch.float64) - below
    tp = below[-1] - below  # positives at or above it
    fp = (n - starts).to(torch.float64) - tp
    return _Counts(unique, tp, tn, fp, fn)


def _sweep(t: torch.Tensor, p: torch.Tensor, dtype: torch.dtype) -> _Sweep:
    """
    :func:`_counts` with a representative threshold for each labelling.

    ``p`` must lie in [0, 1], which bounds the lowest and highest intervals.
    ``dtype`` is the precision the caller's probabilities came in.  The
    counts are exact in any case, but the thresholds are rounded to it so
    that ``prob >= threshold`` reproduces the scored labelling in that dtype
    too -- a float64 midpoint of two adjacent float32 probabilities rounds
    back onto one of them, which would relabel the samples sitting there.
    """
    unique, tp, tn, fp, fn = _counts(t, p)

    # Each candidate is reported as the midpoint of the interval of thresholds
    # that produce its labelling, not as the observed probability at its upper
    # end: the midpoint is the operating point furthest from the nearest
    # validation sample, and it is what makes the tie-break towards 0.5 mean
    # anything.  The lowest interval is bounded below by 0.0, and the
    # all-negative one above by 1.0, or by the next float above 1.0 when a
    # probability is exactly 1.0 (sigmoid in float32 saturates there).
    one = torch.ones(1, dtype=dtype)
    top = 1.0 if float(unique[-1]) < 1.0 else float(torch.nextafter(one, one * 2))
    lower = torch.cat([torch.zeros(1, dtype=torch.float64), unique])
    upper = torch.cat([unique, torch.tensor([top], dtype=torch.float64)])
    midpoint = ((lower + upper) / 2).to(dtype).to(torch.float64)
    # The endpoints are observed probabilities, exact in ``dtype``; the
    # midpoint may round down onto the lower one, or the interval may be too
    # narrow to have an interior point at all.  The upper end always produces
    # the intended labelling.
    threshold = torch.where(midpoint > lower, midpoint, upper)
    return _Sweep(threshold, tp, tn, fp, fn)


def find_optimal_threshold(
    y_true: object,
    y_prob: object,
    metric: str | Metric = "mcc",
) -> ThresholdSearchResult:
    """
    Threshold on ``y_prob`` that maximises ``metric`` against ``y_true``.

    Every distinct labelling the probabilities can produce is scored: with a
    sample positive when ``prob >= threshold``, all thresholds between two
    adjacent unique probabilities agree, so the candidates are those intervals
    plus the all-negative one above the largest probability.  That is
    exhaustive, so no grid resolution has to be chosen.

    The threshold returned is the *midpoint* of the winning interval rather
    than the probability at its upper end, so it sits as far as possible from
    the nearest validation sample: perfectly separable data with a gap between
    0.08 and 0.92 returns 0.5, not 0.92.  Ties between intervals are broken
    towards the candidate whose midpoint is closest to 0.5.

    Named metrics are scored from cumulative confusion counts, so the whole
    search costs one sort.  A callable metric is invoked once per candidate
    instead, since it is only defined on hard labels.

    Parameters
    ----------
    y_true:
        Binary labels, shape ``(n,)``.
    y_prob:
        Positive-class probabilities, shape ``(n,)``, e.g. from
        ``model.predict_proba(X_val)``.
    metric:
        ``"mcc"`` (default), ``"f1"``, ``"balanced_accuracy"``, or any callable
        ``(y_true, y_pred) -> float``.  A callable is always passed hard 0/1
        predictions, ``(y_prob >= threshold).long()``, never the
        probabilities.  Threshold-free metrics such as PR-AUC or ROC-AUC
        therefore do not belong here: given hard labels they score one
        degenerate operating point, not the ranking.  :func:`pr_auc` is
        refused outright; other such scorers are not detected.

    Returns
    -------
    ThresholdSearchResult
        ``(threshold, score)``.  The threshold may exceed 1.0 when labelling
        every sample negative is optimal and some probability is exactly 1.0.

    Raises
    ------
    ValueError
        If ``metric`` is an unknown name or is :func:`pr_auc` (by name or as
        the function), if the inputs are empty or differ in length, if
        ``y_true`` is not binary, or if ``y_prob`` contains NaN or lies
        outside ``[0, 1]``.

    Examples
    --------
    >>> import torch
    >>> y = torch.tensor([0, 0, 0, 1, 1])
    >>> p = torch.tensor([0.1, 0.2, 0.4, 0.45, 0.9])
    >>> result = find_optimal_threshold(y, p)
    >>> round(result.threshold, 3), result.score
    (0.425, 1.0)
    """
    scorer: Metric | None = None
    counts_metric: CountMetric | None = None
    if metric is pr_auc or metric == "pr_auc":
        raise ValueError(
            "pr_auc is threshold-free: it scores the probabilities over every "
            "threshold, so no threshold maximises it.  Search on 'mcc', 'f1' or "
            "'balanced_accuracy' and report pr_auc(y_true, y_prob) alongside."
        )
    if isinstance(metric, str):
        if metric not in METRICS:
            raise ValueError(
                f"unknown metric {metric!r}; choose from {sorted(METRICS)} or pass a callable."
            )
        counts_metric = _COUNT_METRICS[metric]
    else:
        scorer = metric

    t, p = _validate_scores(y_true, y_prob)
    # Counts are taken in float64 whatever came in, but the threshold is
    # reported in the precision the probabilities carry (see _sweep); a plain
    # Python sequence carries none, so it is taken at face value as float64.
    native = torch.as_tensor(y_prob).dtype if hasattr(y_prob, "dtype") else torch.float64
    dtype = native if native.is_floating_point else torch.float64
    # NaN passes both comparisons below and would be silently labelled
    # negative at every threshold, which a diverged model makes easy to hit.
    if torch.any(torch.isnan(p)):
        raise ValueError(f"y_prob contains {int(torch.isnan(p).sum())} NaN value(s).")
    if torch.any(p < 0) or torch.any(p > 1):
        raise ValueError(f"y_prob must lie in [0, 1]; got min {p.min():.4g}, max {p.max():.4g}.")

    sweep = _sweep(t, p, dtype)
    if counts_metric is not None:
        scores = counts_metric(sweep.tp, sweep.tn, sweep.fp, sweep.fn)
    else:
        assert scorer is not None
        scores = torch.tensor(
            [float(scorer(t, (p >= threshold).long())) for threshold in sweep.threshold.tolist()],
            dtype=torch.float64,
        )
    # A callable metric may return NaN; as with the plain maximum it once used,
    # the search never prefers one over a number.
    scores = torch.where(torch.isnan(scores), torch.full_like(scores, -math.inf), scores)

    tied = (scores == scores.max()).nonzero().flatten()
    best = tied[torch.argmin((sweep.threshold[tied] - 0.5).abs())]
    return ThresholdSearchResult(float(sweep.threshold[best]), float(scores[best]))


# ---------------------------------------------------------------------------
# Threshold-free metric
# ---------------------------------------------------------------------------


def pr_auc(y_true: object, y_prob: object) -> float:
    """
    Area under the precision-recall curve, as average precision.

    With the operating points taken in order of decreasing threshold,

        AP = Σ_k (R_k − R_{k−1}) · P_k

    the step interpolation of ``sklearn.metrics.average_precision_score``:
    each gain in recall is credited at the precision of the operating point
    that achieves it.  It is not the trapezoidal ``auc(recall, precision)``,
    which interpolates linearly between operating points and so credits
    precision no threshold actually attains.

    Samples that share a probability form one operating point: a threshold
    cannot separate them, so the result does not depend on the order in which
    tied samples arrive.

    PR-AUC is threshold-free, the usual companion of MCC on imbalanced data;
    it is not in :data:`METRICS`, and :func:`find_optimal_threshold` refuses
    it, because no threshold maximises it.  It depends only on the ranking,
    so any score monotone in the positive-class probability (a logit, say)
    gives the same value.

    Parameters
    ----------
    y_true:
        Binary labels, shape ``(n,)``.
    y_prob:
        Positive-class scores, shape ``(n,)``, e.g. from
        ``model.predict_proba(X_val)``.

    Returns
    -------
    float
        In ``[0, 1]``.  ``0.0`` when ``y_true`` has no positives, where recall
        is undefined; scikit-learn returns the same value, with a warning.

    Raises
    ------
    ValueError
        If the inputs are empty or differ in length, if ``y_true`` is not
        binary, or if ``y_prob`` contains NaN or ±inf.

    Examples
    --------
    >>> round(pr_auc([0, 1, 1, 0, 1], [0.1, 0.4, 0.35, 0.8, 0.9]), 4)  # 1/3 + 2/9 + 1/4
    0.8056
    """
    t, p = _validate_scores(y_true, y_prob)
    if not bool(torch.isfinite(p).all()):
        raise ValueError(
            f"y_prob contains {int((~torch.isfinite(p)).sum())} NaN or ±inf value(s)."
        )

    # One entry per distinct labelling, in order of increasing threshold: the
    # first predicts every sample positive, the last none.
    counts = _counts(t, p)
    positives = counts.tp[0]
    if positives == 0:
        return 0.0
    recall = counts.tp / positives
    precision = _guarded_div(counts.tp, counts.tp + counts.fp)
    # Lowering the threshold from entry k+1 to entry k gains recall
    # R_k - R_{k+1}, credited at precision P_k.  The all-negative entry has
    # no predictions and so no precision; it only anchors recall at 0.
    return float(((recall[:-1] - recall[1:]) * precision[:-1]).sum())


# ---------------------------------------------------------------------------
# Parameter efficiency
# ---------------------------------------------------------------------------


def parameter_efficiency(model: nn.Module | int, score: float) -> float:
    """
    ``score`` per thousand trainable parameters, e.g. MCC/kParam.

    A model's count includes every trainable parameter, also those that can
    never move the output (``circuit_summary(...).n_inert_params``; 20 of the
    published SHNN's 122).  That is the convention of the published results.
    To state efficiency over the live parameters instead, pass that count as
    an integer, and say which one was used.

    Parameters
    ----------
    model:
        A module exposing ``count_parameters()`` (the hybrid classifiers), any
        ``nn.Module`` (trainable parameters are counted directly), or an
        integer parameter count.
    score:
        The metric value achieved by that model.

    Raises
    ------
    ValueError
        If the parameter count is not positive.
    """
    if isinstance(model, int):
        n_params = model
    else:
        # nn.Module.__getattr__ is typed as returning Tensor | Module, so the
        # method is looked up and checked explicitly rather than via hasattr
        counter = getattr(model, "count_parameters", None)
        if callable(counter):
            n_params = int(counter())
        else:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params <= 0:
        raise ValueError(f"parameter count must be positive; got {n_params}.")
    return float(score) / (n_params / 1000.0)
