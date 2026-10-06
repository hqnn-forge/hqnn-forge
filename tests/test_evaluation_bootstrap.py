"""
tests/test_evaluation_bootstrap.py
==================================
hqnn_forge.evaluation.bootstrap (#205): the stratification invariant, the
fast count path against the generic one, the paired variant, the BCa pieces
against closed forms and SciPy, and coverage close to nominal on a case with
a known population MCC.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hqnn_forge.evaluation import (
    BootstrapResult,
    balanced_accuracy,
    bootstrap_ci,
    f1_score,
    matthews_corrcoef,
    paired_bootstrap_ci,
    pr_auc,
)
from hqnn_forge.evaluation import bootstrap as bs


def _predictions(
    n_pos: int = 60, n_neg: int = 540, tpr: float = 0.7, fpr: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n_pos, np.int64), np.zeros(n_neg, np.int64)]
    pred = np.r_[rng.random(n_pos) < tpr, rng.random(n_neg) < fpr].astype(np.int64)
    return y, pred


def _population_mcc(n_pos: int, n_neg: int, tpr: float, fpr: float) -> float:
    tp, fn = n_pos * tpr, n_pos * (1 - tpr)
    fp, tn = n_neg * fpr, n_neg * (1 - fpr)
    return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


class TestStratification:
    def test_every_resample_keeps_each_class_count(self) -> None:
        y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        idx = bs._stratified_indices(y, 500, np.random.default_rng(0))
        assert idx.shape == (500, 10)
        assert np.all(y[idx].sum(axis=1) == 3)
        # Positives are drawn from positives only, negatives from negatives.
        assert set(idx[:, :7].ravel()) <= set(np.flatnonzero(y == 0))
        assert set(idx[:, 7:].ravel()) <= set(np.flatnonzero(y == 1))

    def test_resamples_draw_with_replacement(self) -> None:
        y, _ = _predictions()
        idx = bs._stratified_indices(y, 50, np.random.default_rng(0))
        assert any(np.unique(row).size < row.size for row in idx)


class TestMetricPaths:
    def test_named_metrics_match_the_generic_path(self) -> None:
        y, pred = _predictions()
        for name, fn in (("mcc", matthews_corrcoef), ("f1", f1_score)):
            fast = bootstrap_ci(y, pred, name, n_resamples=300, rng=4)
            slow = bootstrap_ci(y, pred, fn, n_resamples=300, rng=4)
            np.testing.assert_allclose(fast.distribution, slow.distribution, rtol=0, atol=1e-12)
            assert (fast.low, fast.high) == pytest.approx((slow.low, slow.high), abs=1e-12)
            assert fast.estimate == pytest.approx(fn(y, pred), abs=1e-12)

    def test_threshold_free_callable_on_probabilities(self) -> None:
        rng = np.random.default_rng(1)
        y = np.r_[np.ones(30, np.int64), np.zeros(170, np.int64)]
        prob = np.clip(0.3 * y + rng.random(200) * 0.7, 0, 1)

        res = bootstrap_ci(y, prob, pr_auc, n_resamples=200, rng=0)
        assert res.low <= res.estimate <= res.high
        assert res.estimate == pr_auc(y, prob)
        idx = bs._stratified_indices(y, 200, np.random.default_rng(0))
        np.testing.assert_array_equal(res.distribution, [pr_auc(y[r], prob[r]) for r in idx])

    def test_named_metric_refuses_probabilities(self) -> None:
        y, _ = _predictions()
        with pytest.raises(ValueError, match="needs hard 0/1 predictions"):
            bootstrap_ci(y, np.full(y.size, 0.3), "mcc", n_resamples=10)


class TestPaired:
    def test_both_models_are_scored_on_the_same_resamples(self) -> None:
        y, a = _predictions(seed=0)
        _, b = _predictions(seed=1)
        paired = paired_bootstrap_ci(y, a, b, n_resamples=400, rng=7)
        alone_a = bootstrap_ci(y, a, n_resamples=400, rng=7)
        alone_b = bootstrap_ci(y, b, n_resamples=400, rng=7)
        np.testing.assert_allclose(
            paired.distribution, alone_a.distribution - alone_b.distribution, atol=1e-12
        )
        assert paired.estimate == pytest.approx(alone_a.estimate - alone_b.estimate)

    def test_pairing_separates_models_whose_own_intervals_overlap(self) -> None:
        # b equals a except that it misses four positives a finds: each
        # model's own interval contains the other's, yet the paired interval
        # of the difference excludes 0 -- the case the paired variant is for.
        y, a = _predictions()
        b = a.copy()
        b[np.flatnonzero((y == 1) & (a == 1))[:4]] = 0
        alone_a = bootstrap_ci(y, a, n_resamples=2000, rng=0)
        alone_b = bootstrap_ci(y, b, n_resamples=2000, rng=0)
        assert alone_a.low < alone_b.high and alone_b.low < alone_a.high  # overlap
        paired = paired_bootstrap_ci(y, a, b, n_resamples=2000, rng=0)
        assert paired.low > 0
        assert paired.high - paired.low < alone_a.high - alone_a.low

    def test_identical_models(self) -> None:
        y, a = _predictions()
        res = paired_bootstrap_ci(y, a, a, n_resamples=100, rng=0)
        assert (res.estimate, res.low, res.high) == (0.0, 0.0, 0.0)


class TestBcaPieces:
    def test_normal_quantile(self) -> None:
        assert bs._norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-13)
        assert bs._norm_ppf(0.5) == pytest.approx(0.0, abs=1e-15)
        assert bs._norm_ppf(0.001) == pytest.approx(-3.090232306167813, abs=1e-12)
        assert bs._norm_ppf(0.0) == -math.inf and bs._norm_ppf(1.0) == math.inf
        for p in np.linspace(1e-6, 1 - 1e-6, 101):
            assert bs._norm_cdf(bs._norm_ppf(float(p))) == pytest.approx(p, abs=1e-13)

    def test_acceleration_matches_the_two_sample_closed_form(self) -> None:
        # For mean(positives) - mean(negatives) the multi-sample jackknife
        # gives a = [Σdx³/n1³ - Σdy³/n2³] / (6 [Σdx²/n1² + Σdy²/n2²]^1.5).
        rng = np.random.default_rng(2)
        y = np.r_[np.ones(25, np.int64), np.zeros(75, np.int64)]
        s = np.r_[rng.exponential(1.0, 25), rng.gamma(0.5, 1.0, 75)]
        theta = bs._jackknife(lambda t, v: v[t == 1].mean() - v[t == 0].mean(), y, s)
        dx, dy = s[y == 1] - s[y == 1].mean(), s[y == 0] - s[y == 0].mean()
        n1, n2 = dx.size, dy.size
        expected = (np.sum(dx**3) / n1**3 - np.sum(dy**3) / n2**3) / (
            6 * (np.sum(dx**2) / n1**2 + np.sum(dy**2) / n2**2) ** 1.5
        )
        assert bs._acceleration(theta, y) == pytest.approx(expected, rel=1e-10)

    def test_ties_at_the_estimate_count_half(self) -> None:
        # Symmetric around the estimate with ties on it: z0 = 0, and with a
        # constant jackknife the acceleration is 0 too, so BCa reduces to the
        # percentile interval.  Counting only values strictly below would give
        # z0 < 0 and shift the interval down.
        dist = np.array([0.0, 1.0, 1.0, 1.0, 2.0] * 40)
        y = np.array([1, 0, 0, 0])

        def jackknife() -> np.ndarray:
            return np.zeros(4)

        bca = bs._interval(1.0, dist, jackknife, y, 0.9, "bca")
        pct = bs._interval(1.0, dist, jackknife, y, 0.9, "percentile")
        assert bca == pytest.approx(pct, abs=1e-12)

    def test_percentile_interval_is_the_plain_quantiles(self) -> None:
        y, pred = _predictions()
        res = bootstrap_ci(y, pred, n_resamples=500, method="percentile", confidence=0.9, rng=0)
        assert (res.low, res.high) == tuple(np.quantile(res.distribution, [0.05, 0.95]))

    def test_jackknife_leaves_each_sample_out(self) -> None:
        y = np.array([1, 0, 1, 1, 0])
        s = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        loo = bs._jackknife(lambda t, v: float(v.sum() + 1000 * t.sum()), y, s)
        np.testing.assert_array_equal(loo, s.sum() + 1000 * y.sum() - s - 1000 * y)

    @pytest.mark.parametrize(
        ("name", "fn"),
        [("mcc", matthews_corrcoef), ("f1", f1_score), ("balanced_accuracy", balanced_accuracy)],
    )
    def test_count_jackknife_matches_brute_force(self, name: str, fn: bs.MetricArg) -> None:
        # The named-metric jackknife comes from the confusion counts, not from
        # n leave-one-out evaluations; it must give the same values.  The
        # second case has an empty cell (no false negatives).
        y, pred = _predictions(n_pos=12, n_neg=40)
        cases = [(y, pred), (y, np.where(y == 1, 1, pred))]
        for t, p in cases:
            np.testing.assert_allclose(
                bs._jackknife(name, t, p), bs._jackknife(fn, t, p), rtol=0, atol=1e-12
            )

    def test_bca_on_a_full_size_fold(self) -> None:
        # A credit-card test fold holds ~57,000 samples; the BCa jackknife
        # must not build an (n, n - 1) index matrix (~26 GB) to get there.
        rng = np.random.default_rng(0)
        n = 57_000
        y = np.zeros(n, np.int64)
        y[:100] = 1
        pred = np.where(y == 1, rng.random(n) < 0.7, rng.random(n) < 0.001).astype(np.int64)
        res = bootstrap_ci(y, pred, n_resamples=200, rng=0)
        assert res.low < res.estimate < res.high

    def test_bca_agrees_with_scipy(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        y = np.r_[np.ones(40, np.int64), np.zeros(160, np.int64)]
        s = np.r_[rng.exponential(1.0, 40) + 0.3, rng.exponential(1.0, 160)]
        ours = bootstrap_ci(
            y, s, lambda t, v: v[t == 1].mean() - v[t == 0].mean(), n_resamples=20000, rng=1
        )
        theirs = stats.bootstrap(
            (s[y == 1], s[y == 0]),
            lambda a, b, axis=-1: a.mean(axis) - b.mean(axis),
            n_resamples=20000,
            method="BCa",
            random_state=1,
            vectorized=True,
        ).confidence_interval
        # Different resample streams: agreement to Monte Carlo error only.
        assert ours.low == pytest.approx(theirs.low, abs=0.015)
        assert ours.high == pytest.approx(theirs.high, abs=0.015)


class TestCoverage:
    @pytest.mark.slow  # 400 bootstraps of 1000 resamples each
    @pytest.mark.parametrize("method", ["bca", "percentile"])
    def test_close_to_nominal_for_a_known_mcc(self, method: bs.Method) -> None:
        n_pos, n_neg, tpr, fpr = 60, 540, 0.7, 0.05
        truth = _population_mcc(n_pos, n_neg, tpr, fpr)
        reps = 400
        hits = 0
        for rep in range(reps):
            y, pred = _predictions(n_pos, n_neg, tpr, fpr, seed=1000 + rep)
            res = bootstrap_ci(y, pred, "mcc", n_resamples=1000, method=method, rng=rep)
            hits += res.low <= truth <= res.high
        # 400 repetitions: the standard error of the coverage is about 0.011.
        assert 0.92 <= hits / reps <= 0.98, hits / reps


class TestInterface:
    def test_result_fields(self) -> None:
        y, pred = _predictions()
        res = bootstrap_ci(y, pred, n_resamples=100, confidence=0.9, rng=0)
        assert isinstance(res, BootstrapResult)
        assert res.confidence == 0.9 and res.method == "bca" and res.distribution.shape == (100,)
        assert res.low <= res.estimate <= res.high

    def test_seeding(self) -> None:
        y, pred = _predictions()
        state = np.random.get_state()[1].copy()
        a = bootstrap_ci(y, pred, n_resamples=200, rng=3)
        b = bootstrap_ci(y, pred, n_resamples=200, rng=np.random.default_rng(3))
        c = bootstrap_ci(y, pred, n_resamples=200, rng=4)
        np.testing.assert_array_equal(a.distribution, b.distribution)
        assert not np.array_equal(a.distribution, c.distribution)
        np.testing.assert_array_equal(np.random.get_state()[1], state)  # global RNG untouched

    def test_higher_confidence_is_wider(self) -> None:
        y, pred = _predictions()
        narrow = bootstrap_ci(y, pred, n_resamples=2000, confidence=0.8, rng=0)
        wide = bootstrap_ci(y, pred, n_resamples=2000, confidence=0.99, rng=0)
        assert wide.low < narrow.low and wide.high > narrow.high

    def test_constant_metric_gives_a_point_interval(self) -> None:
        y = np.r_[np.ones(10, np.int64), np.zeros(30, np.int64)]
        res = bootstrap_ci(y, y, "mcc", n_resamples=50, rng=0)  # perfect on every resample
        assert (res.estimate, res.low, res.high) == (1.0, 1.0, 1.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_resamples": 1}, "n_resamples must be >= 2"),
            ({"confidence": 1.0}, r"confidence must lie in \(0, 1\)"),
            ({"method": "basic"}, "method must be 'bca' or 'percentile'"),
            ({"metric": "auc"}, "unknown metric 'auc'"),
        ],
    )
    def test_rejected_arguments(self, kwargs: dict, match: str) -> None:
        y, pred = _predictions()
        with pytest.raises(ValueError, match=match):
            bootstrap_ci(y, pred, **kwargs)

    @pytest.mark.parametrize(
        ("y", "s", "match"),
        [
            ([], [], "empty"),
            ([0, 2, 1], [0, 1, 1], "only 0/1 labels"),
            ([1, 1, 1], [0, 1, 1], "both classes"),
            ([0, 1, 1], [0, 1], "length 3"),
            ([0, 1, 1], [0, np.nan, 1], "finite"),
        ],
    )
    def test_rejected_data(self, y: list, s: list, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            bootstrap_ci(y, s, n_resamples=10)
