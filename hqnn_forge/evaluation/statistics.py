"""
hqnn_forge.evaluation.statistics
================================
Non-parametric comparison of models: two models across CV folds (Wilcoxon
signed-rank), and several models across several datasets (Friedman test with
Nemenyi or Holm post-hoc comparisons, following Demšar 2006).

With a handful of folds, parametric tests are not justified and the Wilcoxon
signed-rank test is the usual choice.  Its weakness at that size is that the
p-value is *discrete* with a floor: with n = 5 non-zero differences the
smallest two-sided p-value any data can produce is 2 / 2^5 = 0.0625, so
"not significant at 0.05" is guaranteed before a single fold is run.  The
result therefore carries ``min_p_value`` next to ``p_value``, and the
rank-biserial correlation is provided as the effect size to report instead.

Everything is computed in pure Python/NumPy: the exact null distribution of
the signed-rank statistic is built by dynamic programming over the ranks, so
tied absolute differences (average ranks) are handled exactly rather than by
switching to the normal approximation.

References
----------
* Wilcoxon (1945) "Individual comparisons by ranking methods", Biometrics
  Bulletin 1(6), 80–83.
* Kerby (2014) "The simple difference formula: an approach to teaching
  nonparametric correlation", Comprehensive Psychology 3, 11.IT.3.1.
* Demšar (2006) "Statistical comparisons of classifiers over multiple data
  sets", Journal of Machine Learning Research 7, 1–30.
* Friedman (1937) "The use of ranks to avoid the assumption of normality
  implicit in the analysis of variance", JASA 32(200), 675–701.
* Iman & Davenport (1980) "Approximations of the critical region of the
  Friedman statistic", Communications in Statistics – Theory and Methods
  9(6), 571–595.
* Holm (1979) "A simple sequentially rejective multiple test procedure",
  Scandinavian Journal of Statistics 6(2), 65–70.
* Nemenyi (1963) "Distribution-free multiple comparisons", PhD thesis,
  Princeton University.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

Alternative = Literal["two-sided", "greater", "less"]

#: Largest number of non-zero differences for which the exact distribution is
#: built; above it the normal approximation (with tie correction) is used.
EXACT_MAX_N: int = 50


class WilcoxonResult(NamedTuple):
    """
    Result of :func:`wilcoxon_signed_rank`.

    Attributes
    ----------
    statistic:
        ``W+``, the sum of the ranks of the positive differences ``a - b``.
    p_value:
        p-value for ``alternative``.
    n:
        Number of non-zero differences the test used.
    min_p_value:
        Smallest p-value attainable for this ``n``, rank pattern and
        ``alternative``.  If it exceeds your significance level, the test
        cannot reject whatever the data.
    method:
        ``"exact"`` or ``"normal"``.
    """

    statistic: float
    p_value: float
    n: int
    min_p_value: float
    method: str


def _differences(scores_a: npt.ArrayLike, scores_b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"paired scores must have the same length; got {a.size} and {b.size}.")
    if a.size == 0:
        raise ValueError("no scores given.")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("scores must be finite.")
    return a - b


def _average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """1-based ranks with ties sharing their mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _signed_ranks(
    diffs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    nonzero = diffs[diffs != 0.0]
    return _average_ranks(np.abs(nonzero)), nonzero > 0


def _exact_null_counts(ranks: npt.NDArray[np.float64]) -> dict[int, int]:
    """
    Number of sign assignments giving each value of 2·W+.

    Average ranks are multiples of 0.5, so doubling makes every rank an
    integer and the distribution can be counted exactly.
    """
    counts: dict[int, int] = {0: 1}
    for r in (int(round(2 * x)) for x in ranks):
        nxt: dict[int, int] = dict(counts)
        for total, c in counts.items():
            nxt[total + r] = nxt.get(total + r, 0) + c
        counts = nxt
    return counts


def _exact_p(counts: dict[int, int], w2: int, alternative: Alternative) -> float:
    total = sum(counts.values())
    upper = sum(c for v, c in counts.items() if v >= w2) / total
    lower = sum(c for v, c in counts.items() if v <= w2) / total
    if alternative == "greater":
        return upper
    if alternative == "less":
        return lower
    return min(1.0, 2.0 * min(upper, lower))


def _normal_p(ranks: npt.NDArray[np.float64], w_plus: float, alternative: Alternative) -> float:
    n = ranks.size
    mean = n * (n + 1) / 4.0
    # Tie correction: variance of W+ under H0 is sum(r_i^2) / 4
    sd = math.sqrt(float(np.sum(ranks**2)) / 4.0)
    if sd == 0.0:
        return 1.0
    z = (w_plus - mean) / sd
    upper = 0.5 * math.erfc(z / math.sqrt(2))
    lower = 0.5 * math.erfc(-z / math.sqrt(2))
    if alternative == "greater":
        return upper
    if alternative == "less":
        return lower
    return min(1.0, 2.0 * min(upper, lower))


def wilcoxon_signed_rank(
    scores_a: npt.ArrayLike,
    scores_b: npt.ArrayLike,
    alternative: Alternative = "two-sided",
) -> WilcoxonResult:
    """
    Wilcoxon signed-rank test on paired per-fold scores.

    Zero differences are dropped before ranking (Wilcoxon's original
    treatment).  Tied absolute differences get average ranks.

    Parameters
    ----------
    scores_a, scores_b:
        Paired metric values, one per fold, same length.
    alternative:
        ``"two-sided"`` (default); ``"greater"`` tests whether ``a`` tends to
        exceed ``b``; ``"less"`` the reverse.

    Returns
    -------
    WilcoxonResult

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty or non-finite, or if every
        difference is zero (the test is undefined; the rank-biserial
        correlation is 0 in that case).

    Examples
    --------
    >>> res = wilcoxon_signed_rank([0.58, 0.61, 0.55, 0.60, 0.57],
    ...                            [0.56, 0.57, 0.54, 0.55, 0.56])
    >>> res.statistic, res.p_value, res.min_p_value
    (15.0, 0.0625, 0.0625)
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"alternative must be 'two-sided', 'greater' or 'less'; got {alternative!r}."
        )
    ranks, positive = _signed_ranks(_differences(scores_a, scores_b))
    n = int(ranks.size)
    if n == 0:
        raise ValueError("every paired difference is zero, so the signed-rank test is undefined.")
    w_plus = float(np.sum(ranks[positive]))

    if n <= EXACT_MAX_N:
        counts = _exact_null_counts(ranks)
        p = _exact_p(counts, int(round(2 * w_plus)), alternative)
        extremes = (min(counts), max(counts))
        if alternative == "greater":
            min_p = _exact_p(counts, extremes[1], alternative)
        elif alternative == "less":
            min_p = _exact_p(counts, extremes[0], alternative)
        else:
            min_p = _exact_p(counts, extremes[1], alternative)
        return WilcoxonResult(w_plus, p, n, min_p, "exact")

    p = _normal_p(ranks, w_plus, alternative)
    total = float(np.sum(ranks))
    extreme = total if alternative != "less" else 0.0
    min_p = _normal_p(ranks, extreme, alternative)
    return WilcoxonResult(w_plus, p, n, min_p, "normal")


def rank_biserial_correlation(scores_a: npt.ArrayLike, scores_b: npt.ArrayLike) -> float:
    """
    Matched-pairs rank-biserial correlation, the effect size for the
    signed-rank test.

    ``r = (W+ − W−) / (W+ + W−)`` over the non-zero differences (Kerby 2014).
    It lies in [-1, 1]: ``+1`` when ``a`` beats ``b`` on every fold, ``-1``
    when it loses on every fold, ``0`` when wins and losses balance by rank.
    If every difference is zero the models are indistinguishable and ``0.0``
    is returned.

    Parameters
    ----------
    scores_a, scores_b:
        Paired metric values, one per fold, same length.
    """
    ranks, positive = _signed_ranks(_differences(scores_a, scores_b))
    total = float(np.sum(ranks))
    if total == 0.0:
        return 0.0
    w_plus = float(np.sum(ranks[positive]))
    return (2.0 * w_plus - total) / total


# ---------------------------------------------------------------------------
# Several models over several datasets (Demšar 2006)
# ---------------------------------------------------------------------------
#
# Comparing k models pairwise with one Wilcoxon test each inflates the
# family-wise false-positive rate, and a per-dataset p-value does not say
# whether a model is better *across* datasets.  Demšar's procedure: a
# Friedman test on the models' ranks within each dataset; if it rejects, the
# Nemenyi critical difference for all pairs, or Holm-corrected z-tests of
# every model against one control.  The distribution tails below (chi-square,
# F, normal, studentised range at infinite degrees of freedom) are computed
# here, NumPy-only, so SciPy stays out of the runtime dependencies.


class FriedmanResult(NamedTuple):
    """
    Result of :func:`friedman_test`.

    Attributes
    ----------
    statistic:
        Friedman's ``χ²_F`` on ``k - 1`` degrees of freedom.
    p_value:
        Upper tail of the chi-square distribution at ``statistic``.
    iman_davenport:
        Iman and Davenport's ``F_F = (N-1) χ²_F / (N(k-1) - χ²_F)`` on
        ``(k-1, (k-1)(N-1))`` degrees of freedom; less conservative than
        ``χ²_F``, and the variant Demšar recommends.
    iman_davenport_p_value:
        Upper tail of the F distribution at ``iman_davenport``.
    average_ranks:
        Mean rank of each model over the datasets, shape ``(k,)``; rank 1 is
        the best.
    n_datasets, n_models:
        ``N`` and ``k``.
    """

    statistic: float
    p_value: float
    iman_davenport: float
    iman_davenport_p_value: float
    average_ranks: npt.NDArray[np.float64]
    n_datasets: int
    n_models: int


class ControlComparison(NamedTuple):
    """
    One model against the control in :func:`compare_to_control`.

    Attributes
    ----------
    model:
        Column index of the model.
    rank_difference:
        ``R_model - R_control``; negative when the model ranks better.
    z:
        ``rank_difference / sqrt(k(k+1) / (6N))``.
    p_value:
        Two-sided normal p-value of ``z``.
    p_adjusted:
        ``p_value`` after Holm's correction over the ``k - 1`` comparisons.
    """

    model: int
    rank_difference: float
    z: float
    p_value: float
    p_adjusted: float


def _score_matrix(scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"scores must be a (n_datasets, n_models) matrix; got shape {matrix.shape}."
        )
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(
            f"need at least 2 datasets and 2 models; got {matrix.shape[0]} and {matrix.shape[1]}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scores must be finite.")
    return matrix


def average_ranks(
    scores: npt.ArrayLike, *, higher_is_better: bool = True
) -> npt.NDArray[np.float64]:
    """
    Mean rank of each model (column) over the datasets (rows) of ``scores``.

    Within a dataset the best model gets rank 1; tied scores share their
    average rank, as in Demšar (2006).
    """
    matrix = _score_matrix(scores)
    signed = -matrix if higher_is_better else matrix
    return np.mean([_average_ranks(row) for row in signed], axis=0)


def _check_ranks(ranks: npt.ArrayLike, n_datasets: int) -> npt.NDArray[np.float64]:
    r = np.asarray(ranks, dtype=np.float64).reshape(-1)
    k = r.size
    if k < 2 or n_datasets < 2:
        raise ValueError(f"need at least 2 datasets and 2 models; got {n_datasets} and {k}.")
    # Published ranks are rounded; half a unit in the second decimal per rank
    # accepts two-decimal tables and still catches a mistyped rank.
    if not math.isclose(float(r.sum()), k * (k + 1) / 2, rel_tol=0, abs_tol=0.005 * k):
        raise ValueError(
            f"average ranks of {k} models must sum to k(k+1)/2 = {k * (k + 1) / 2}; "
            f"got {float(r.sum())}."
        )
    return r


def friedman_from_ranks(average_ranks: npt.ArrayLike, n_datasets: int) -> FriedmanResult:
    """
    :func:`friedman_test` from the models' average ranks alone, as published
    tables usually report them.

    Parameters
    ----------
    average_ranks:
        Mean rank of each of the ``k`` models; they must sum to ``k(k+1)/2``,
        up to rounding to two decimals (``0.005 k``).
    n_datasets:
        Number of datasets ``N`` the ranks were averaged over.
    """
    r = _check_ranks(average_ranks, n_datasets)
    k, n = r.size, n_datasets
    chi2 = 12.0 * n / (k * (k + 1)) * (float(np.sum(r**2)) - k * (k + 1) ** 2 / 4.0)
    chi2 = max(chi2, 0.0)  # rounding below zero when all ranks are equal
    df1, df2 = k - 1, (k - 1) * (n - 1)
    denominator = n * (k - 1) - chi2
    if denominator <= 0.0:  # every dataset ranks the models identically
        f_stat, f_p = math.inf, 0.0
    else:
        f_stat = (n - 1) * chi2 / denominator
        f_p = _f_sf(f_stat, df1, df2)
    return FriedmanResult(chi2, _chi2_sf(chi2, df1), f_stat, f_p, r, n, k)


def friedman_test(scores: npt.ArrayLike, *, higher_is_better: bool = True) -> FriedmanResult:
    """
    Friedman test that ``k`` models perform alike over ``N`` datasets.

    Each dataset ranks the models (rank 1 best, ties averaged); under the null
    hypothesis every model's average rank is ``(k+1)/2``.  No tie correction
    is applied to ``χ²_F``, as in Demšar (2006); with ties that makes the
    test slightly conservative.

    Parameters
    ----------
    scores:
        ``(n_datasets, n_models)`` matrix, one score per model and dataset,
        e.g. the mean test MCC of each model on each dataset.
    higher_is_better:
        ``False`` for losses and error rates.

    Returns
    -------
    FriedmanResult
        Report ``iman_davenport_p_value``; if it rejects, follow with
        :func:`nemenyi_critical_difference` or :func:`compare_to_control`.

    Raises
    ------
    ValueError
        If ``scores`` is not a finite matrix with at least 2 rows and columns.
    """
    matrix = _score_matrix(scores)
    return friedman_from_ranks(
        average_ranks(matrix, higher_is_better=higher_is_better), matrix.shape[0]
    )


def nemenyi_critical_difference(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """
    Nemenyi critical difference ``CD = q_α sqrt(k(k+1) / (6N))``.

    Two models whose average ranks differ by at least ``CD`` differ
    significantly at family-wise level ``alpha`` over all ``k(k-1)/2`` pairs.
    ``q_α`` is the upper-``α`` point of the studentised range of ``k``
    normals at infinite degrees of freedom, divided by √2, computed here by
    numerical integration (Demšar 2006, Table 5a, tabulates it for k ≤ 10).
    """
    if n_models < 2 or n_datasets < 1:
        raise ValueError(f"need n_models >= 2 and n_datasets >= 1; got {n_models}, {n_datasets}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")
    q = _studentized_range_isf(alpha, n_models) / math.sqrt(2.0)
    return q * math.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets))


def holm_correction(p_values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Holm's step-down adjusted p-values, in the order given.

    The ``i``-th smallest of ``m`` p-values is multiplied by ``m - i + 1``,
    made non-decreasing along that order and capped at 1.  Rejecting every
    hypothesis whose adjusted p-value is at most ``α`` controls the
    family-wise error rate at ``α`` -- usable on any family, e.g. the
    per-dataset Wilcoxon p-values of one comparison.
    """
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    if p.size == 0:
        raise ValueError("no p-values given.")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("p-values must lie in [0, 1].")
    m = p.size
    order = np.argsort(p, kind="mergesort")
    stepped = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted


def compare_to_control(
    scores: npt.ArrayLike, control: int, *, higher_is_better: bool = True
) -> list[ControlComparison]:
    """
    Every model against ``control`` by the z-test on average ranks, with
    Holm's correction over the ``k - 1`` comparisons (Demšar 2006, §3.2.2).

    Parameters
    ----------
    scores:
        ``(n_datasets, n_models)`` matrix, as for :func:`friedman_test`.
    control:
        Column index of the control model, e.g. the classical baseline.

    Returns
    -------
    list of ControlComparison
        One per non-control model, in column order.
    """
    matrix = _score_matrix(scores)
    ranks = average_ranks(matrix, higher_is_better=higher_is_better)
    return _compare_ranks_to_control(ranks, matrix.shape[0], control)


def _compare_ranks_to_control(
    ranks: npt.NDArray[np.float64], n_datasets: int, control: int
) -> list[ControlComparison]:
    k = ranks.size
    if not 0 <= control < k:
        raise ValueError(f"control must be a column index in [0, {k}); got {control}.")
    se = math.sqrt(k * (k + 1) / (6.0 * n_datasets))
    others = [j for j in range(k) if j != control]
    diffs = [float(ranks[j] - ranks[control]) for j in others]
    zs = [d / se for d in diffs]
    ps = [math.erfc(abs(z) / math.sqrt(2.0)) for z in zs]
    adjusted = holm_correction(ps)
    return [
        ControlComparison(j, d, z, p, float(a))
        for j, d, z, p, a in zip(others, diffs, zs, ps, adjusted)
    ]


# ---------------------------------------------------------------------------
# Distribution tails (NumPy / math only)
# ---------------------------------------------------------------------------

_EPS = 1e-15
_MAX_ITER = 500


def _gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma ``Q(a, x)`` (series / continued fraction)."""
    if x <= 0.0:
        return 1.0
    log_prefactor = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:  # series for P(a, x)
        term = total = 1.0 / a
        ap = a
        for _ in range(_MAX_ITER):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefactor))
    # Lentz's continued fraction for Q(a, x)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(log_prefactor) * h


def _chi2_sf(x: float, df: int) -> float:
    return _gamma_q(df / 2.0, x / 2.0)


def _beta_cf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _beta_inc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_cf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_cf(b, a, 1.0 - x) / b


def _f_sf(f: float, df1: int, df2: int) -> float:
    """Upper tail of the F distribution with ``(df1, df2)`` degrees of freedom."""
    if f <= 0.0:
        return 1.0
    return _beta_inc(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f))


_erf = np.frompyfunc(math.erf, 1, 1)


def _normal_cdf(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 0.5 * (1.0 + np.asarray(_erf(t / math.sqrt(2.0)), dtype=np.float64))


def _studentized_range_cdf(w: float, k: int) -> float:
    """``P(range of k iid N(0,1) <= w)`` = k ∫ φ(x) [Φ(x+w) − Φ(x)]^(k−1) dx."""
    x = np.linspace(-9.0, 9.0, 3601)
    pdf = np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)
    integrand = k * pdf * (_normal_cdf(x + w) - _normal_cdf(x)) ** (k - 1)
    return float(np.trapezoid(integrand, x))


def _studentized_range_isf(alpha: float, k: int) -> float:
    """``w`` with ``P(range > w) = alpha``, by bisection."""
    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 1.0 - _studentized_range_cdf(mid, k) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
