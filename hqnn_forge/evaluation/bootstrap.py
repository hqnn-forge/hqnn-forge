"""
hqnn_forge.evaluation.bootstrap
===============================
Bootstrap confidence intervals for a metric on one test set, and for the
difference between two models scored on the same samples.

With a rare positive class a test fold may hold a few dozen positives, and
its MCC moves noticeably with a handful of predictions; a point estimate
alone cannot say whether two models differ.

Resampling is **stratified by class**: each resample draws the positives
from the positives and the negatives from the negatives, with replacement,
so every resample keeps the original positive count.  A plain resample can
draw no positives at all, where MCC and F1 are undefined, and its varying
prevalence adds variance that the test set did not have.

Intervals
---------
* ``"percentile"`` -- the ``(1-c)/2`` and ``(1+c)/2`` quantiles of the
  bootstrap distribution.
* ``"bca"`` (default) -- bias-corrected and accelerated (Efron 1987): the
  quantile levels are shifted by the bias ``z0`` (from the share of resamples
  below the estimate, ties counted half) and the acceleration ``a`` (from a
  leave-one-out jackknife within each class, since the classes are resampled
  as separate samples).  Second-order accurate where the percentile
  interval is first-order, which matters for a bounded, skewed metric such as
  MCC near 1.  The normal quantile function is computed here (Acklam's
  approximation refined by Newton steps), so SciPy is not needed.

Metrics
-------
``"mcc"``, ``"f1"`` and ``"balanced_accuracy"`` take hard 0/1 predictions and
are computed from confusion counts for all resamples at once.  Any callable
``metric(y_true, y_score) -> float`` works too, e.g. a threshold-free metric
such as PR-AUC on probabilities; it is called once per resample (and once per
sample for the BCa jackknife), so it is slower.

References
----------
* Efron (1987) "Better bootstrap confidence intervals", JASA 82(397), 171–185.
* Efron & Tibshirani (1993) "An Introduction to the Bootstrap", Chapman & Hall.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from hqnn_forge.evaluation.thresholds import _COUNT_METRICS

Method = Literal["bca", "percentile"]
MetricArg = str | Callable[[npt.NDArray[np.int64], npt.NDArray[np.float64]], float]


class BootstrapResult(NamedTuple):
    """
    Attributes
    ----------
    estimate:
        The metric on the original samples (for the paired variant: model a
        minus model b).
    low, high:
        The interval bounds.
    confidence:
        Nominal coverage, e.g. 0.95.
    method:
        ``"bca"`` or ``"percentile"``.
    distribution:
        The ``n_resamples`` bootstrap values.
    """

    estimate: float
    low: float
    high: float
    confidence: float
    method: str
    distribution: npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Normal distribution without SciPy
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF: Acklam's rational approximation + 2 Newton steps."""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p must lie in [0, 1]; got {p}.")
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)  # fmt: skip
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)  # fmt: skip
    c = (-7.784894002430293e-03, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)  # fmt: skip
    d = (7.784695709041462e-03, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    low = 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p > 1.0 - low:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    for _ in range(2):  # Newton on Φ(x) = p
        x -= (_norm_cdf(x) - p) * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _check_inputs(
    y_true: npt.ArrayLike, *scores: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], list[npt.NDArray[np.float64]]]:
    y = np.asarray(y_true).reshape(-1)
    if y.size == 0:
        raise ValueError("y_true is empty.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0/1 labels.")
    y = y.astype(np.int64)
    if y.min() == y.max():
        raise ValueError("y_true must contain both classes to resample by class.")
    arrays = []
    for s in scores:
        a = np.asarray(s, dtype=np.float64).reshape(-1)
        if a.shape != y.shape:
            raise ValueError(f"scores must have y_true's length {y.size}; got {a.size}.")
        if not np.all(np.isfinite(a)):
            raise ValueError("scores must be finite.")
        arrays.append(a)
    return y, arrays


def _stratified_indices(
    y: npt.NDArray[np.int64], n_resamples: int, rng: np.random.Generator
) -> npt.NDArray[np.intp]:
    """``(n_resamples, n)`` row indices, each row keeping every class's count."""
    parts = []
    for cls in (0, 1):
        rows = np.flatnonzero(y == cls)
        parts.append(rows[rng.integers(0, rows.size, size=(n_resamples, rows.size))])
    return np.concatenate(parts, axis=1)


# Resample rows gathered per block in the count path, so a large test fold
# never materialises several (n_resamples, n) temporaries at once.
_BLOCK_ELEMENTS = 1 << 22


def _cells(
    name: str, y: npt.NDArray[np.int64], pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Each sample's confusion cell: 0 = tp, 1 = tn, 2 = fp, 3 = fn (the count-metric order)."""
    if not np.isin(pred, (0.0, 1.0)).all():
        raise ValueError(
            f"metric {name!r} needs hard 0/1 predictions; threshold the probabilities first, "
            f"or pass a threshold-free metric as a callable."
        )
    p = pred.astype(np.int64)
    # (y, p) = (1, 1) -> tp, (0, 0) -> tn, (0, 1) -> fp, (1, 0) -> fn.
    return np.array([1, 2, 3, 0], dtype=np.int64)[2 * y + p]


def _from_counts(name: str, counts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A named metric for every row of ``counts`` (columns tp, tn, fp, fn)."""
    cols = [torch.from_numpy(np.ascontiguousarray(counts[:, k])) for k in range(4)]
    return _COUNT_METRICS[name](*cols).numpy().astype(np.float64)


def _count_metric_over(
    name: str, y: npt.NDArray[np.int64], pred: npt.NDArray[np.float64], idx: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """A named metric for every row of ``idx`` from its confusion counts."""
    cell = _cells(name, y, pred)
    counts = np.empty((idx.shape[0], 4), dtype=np.float64)
    step = max(1, _BLOCK_ELEMENTS // max(1, idx.shape[1]))
    for start in range(0, idx.shape[0], step):
        block = cell[idx[start : start + step]]
        for k in range(4):
            counts[start : start + step, k] = np.sum(block == k, axis=1)
    return _from_counts(name, counts)


def _metric_over(
    metric: MetricArg,
    y: npt.NDArray[np.int64],
    score: npt.NDArray[np.float64],
    idx: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    if isinstance(metric, str):
        return _count_metric_over(metric, y, score, idx)
    return np.array([float(metric(y[row], score[row])) for row in idx], dtype=np.float64)


def _jackknife(
    metric: MetricArg, y: npt.NDArray[np.int64], score: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Leave-one-out values: entry ``i`` is the metric without sample ``i``.

    For a named metric, leaving sample ``i`` out takes one from its confusion
    cell, so there are only four distinct values and no ``(n, n - 1)`` index
    matrix is built (that would need ~26 GB for a 57,000-sample fold).  A
    callable is called ``n`` times on a boolean mask, ``O(n)`` memory each.
    """
    if isinstance(metric, str):
        cell = _cells(metric, y, score)
        full = np.bincount(cell, minlength=4).astype(np.float64)
        # Row k: the counts with one sample taken from cell k (clipped for an
        # empty cell, whose row no sample ever looks up).
        loo = np.clip(full[None, :] - np.eye(4), 0.0, None)
        return _from_counts(metric, loo)[cell]
    keep = np.ones(y.size, dtype=bool)
    out = np.empty(y.size, dtype=np.float64)
    for i in range(y.size):
        keep[i] = False
        out[i] = float(metric(y[keep], score[keep]))
        keep[i] = True
    return out


def _acceleration(theta: npt.NDArray[np.float64], y: npt.NDArray[np.int64]) -> float:
    """
    BCa acceleration from leave-one-out values ``theta[i]`` (sample ``i``
    left out), for a bootstrap stratified by ``y``: each class is its own
    sample, as in the multi-sample formula of Efron & Tibshirani (1993,
    §14.3).  With ``U_ji = (n_j - 1)(mean_j(theta) - theta_ji)``,
    ``a = Σ_j ΣU³/n_j³ / (6 (Σ_j ΣU²/n_j²)^(3/2))``.  Pooling both classes
    around one mean instead would count the gap between the classes' means as
    jackknife variation.
    """
    num = den = 0.0
    for cls in (0, 1):
        t = theta[y == cls]
        n = t.size
        u = (n - 1) * (t.mean() - t)
        num += float(np.sum(u**3)) / n**3
        den += float(np.sum(u**2)) / n**2
    return num / (6.0 * den**1.5) if den > 0 else 0.0


def _interval(
    estimate: float,
    distribution: npt.NDArray[np.float64],
    jackknife: Callable[[], npt.NDArray[np.float64]],
    y: npt.NDArray[np.int64],
    confidence: float,
    method: Method,
) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    if np.all(distribution == distribution[0]):
        return float(distribution[0]), float(distribution[0])
    if method == "percentile":
        lo, hi = np.quantile(distribution, [alpha, 1.0 - alpha])
        return float(lo), float(hi)
    # BCa.  Ties at the estimate count half, which keeps z0 finite and
    # unbiased for a discrete statistic such as MCC on a small test set.
    below = np.mean(distribution < estimate) + 0.5 * np.mean(distribution == estimate)
    z0 = _norm_ppf(float(below))
    accel = _acceleration(jackknife(), y)
    levels = []
    for z in (_norm_ppf(alpha), _norm_ppf(1.0 - alpha)):
        shifted = z0 + (z0 + z) / (1.0 - accel * (z0 + z))
        levels.append(_norm_cdf(shifted))
    lo, hi = np.quantile(distribution, levels)
    return float(lo), float(hi)


def _validate(n_resamples: int, confidence: float, method: str, metric: MetricArg) -> None:
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be >= 2; got {n_resamples}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1); got {confidence}.")
    if method not in ("bca", "percentile"):
        raise ValueError(f"method must be 'bca' or 'percentile'; got {method!r}.")
    if isinstance(metric, str) and metric not in _COUNT_METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; choose from {sorted(_COUNT_METRICS)} or pass a callable."
        )


def bootstrap_ci(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    metric: MetricArg = "mcc",
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    method: Method = "bca",
    rng: np.random.Generator | int | None = None,
) -> BootstrapResult:
    """
    Class-stratified bootstrap interval for ``metric(y_true, y_score)``.

    Parameters
    ----------
    y_true:
        0/1 labels; both classes must be present.
    y_score:
        Hard 0/1 predictions for the named metrics; whatever a callable
        ``metric`` takes otherwise (e.g. probabilities for PR-AUC).
    metric:
        ``"mcc"`` (default), ``"f1"``, ``"balanced_accuracy"``, or a callable
        ``metric(y_true, y_score) -> float``.
    n_resamples:
        Bootstrap resamples.  Default 2000.
    confidence:
        Nominal coverage.  Default 0.95.
    method:
        ``"bca"`` (default) or ``"percentile"``; see the module docstring.
    rng:
        A ``numpy.random.Generator`` or a seed.  Only this generator is used,
        never a global one, so equal seeds give equal intervals.

    Returns
    -------
    BootstrapResult
    """
    _validate(n_resamples, confidence, method, metric)
    y, (score,) = _check_inputs(y_true, y_score)
    gen = np.random.default_rng(rng)
    everything = np.arange(y.size)[None, :]
    estimate = float(_metric_over(metric, y, score, everything)[0])
    distribution = _metric_over(metric, y, score, _stratified_indices(y, n_resamples, gen))
    low, high = _interval(
        estimate,
        distribution,
        lambda: _jackknife(metric, y, score),
        y,
        confidence,
        method,
    )
    return BootstrapResult(estimate, low, high, confidence, method, distribution)


def paired_bootstrap_ci(
    y_true: npt.ArrayLike,
    score_a: npt.ArrayLike,
    score_b: npt.ArrayLike,
    metric: MetricArg = "mcc",
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    method: Method = "bca",
    rng: np.random.Generator | int | None = None,
) -> BootstrapResult:
    """
    Interval for ``metric(a) - metric(b)``, two models scored on the same
    samples.

    Both models are evaluated on the *same* resample each time, so what the
    samples have in common -- a hard case is hard for both -- cancels out of
    the difference.  Two separate intervals that overlap do not show that the
    models are alike; an interval of the difference that excludes 0 does show
    that they differ.  Parameters as for :func:`bootstrap_ci`.
    """
    _validate(n_resamples, confidence, method, metric)
    y, (a, b) = _check_inputs(y_true, score_a, score_b)
    gen = np.random.default_rng(rng)

    def difference(idx: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        return _metric_over(metric, y, a, idx) - _metric_over(metric, y, b, idx)

    estimate = float(difference(np.arange(y.size)[None, :])[0])
    distribution = difference(_stratified_indices(y, n_resamples, gen))
    low, high = _interval(
        estimate,
        distribution,
        lambda: _jackknife(metric, y, a) - _jackknife(metric, y, b),
        y,
        confidence,
        method,
    )
    return BootstrapResult(estimate, low, high, confidence, method, distribution)
