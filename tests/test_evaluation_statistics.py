"""
tests/test_evaluation_statistics.py
====================================
Unit tests for hqnn_forge.evaluation.statistics.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from hqnn_forge.evaluation import (
    WilcoxonResult,
    average_ranks,
    compare_to_control,
    friedman_from_ranks,
    friedman_test,
    holm_correction,
    nemenyi_critical_difference,
    rank_biserial_correlation,
    statistics,
    wilcoxon_signed_rank,
)
from hqnn_forge.evaluation.statistics import EXACT_MAX_N, Alternative, _average_ranks


def _brute_force_p(diffs: np.ndarray, alternative: Alternative) -> float:
    """Enumerate all 2^n sign flips of the observed |d| ranks."""
    d = diffs[diffs != 0]
    ranks = _average_ranks(np.abs(d))
    observed = ranks[d > 0].sum()
    stats = [
        sum(r for r, s in zip(ranks, signs) if s)
        for signs in itertools.product([0, 1], repeat=len(d))
    ]
    null = np.array(stats)
    upper = np.mean(null >= observed - 1e-9)
    lower = np.mean(null <= observed + 1e-9)
    return {"greater": upper, "less": lower, "two-sided": min(1.0, 2 * min(upper, lower))}[
        alternative
    ]


class TestAverageRanks:
    """Mid-ranks checked against references that do not use _average_ranks."""

    def test_hand_computed_mid_ranks(self) -> None:
        values = np.array([0.5, 0.25, 0.5, 0.5, 0.125, 0.25])
        # sorted: 0.125 | 0.25 0.25 | 0.5 0.5 0.5
        # ranks:      1 | 2.5  2.5  |   5   5   5
        assert _average_ranks(values).tolist() == [5.0, 2.5, 5.0, 5.0, 1.0, 2.5]

    def test_no_ties_are_plain_ordinal_ranks(self) -> None:
        values = np.array([0.4, 0.1, 0.3, 0.2])
        assert _average_ranks(values).tolist() == [4.0, 1.0, 3.0, 2.0]

    def test_matches_scipy_rankdata_on_heavily_tied_data(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(3)
        values = rng.integers(1, 5, size=20).astype(float)
        np.testing.assert_allclose(_average_ranks(values), stats.rankdata(values))


class TestRankBiserial:
    def test_identical_arrays_give_zero(self) -> None:
        a = [0.5, 0.6, 0.7]
        assert rank_biserial_correlation(a, a) == 0.0

    def test_strict_domination_gives_plus_minus_one(self) -> None:
        a = [0.60, 0.62, 0.65, 0.58, 0.61]
        b = [0.50, 0.52, 0.55, 0.48, 0.51]
        assert rank_biserial_correlation(a, b) == pytest.approx(1.0)
        assert rank_biserial_correlation(b, a) == pytest.approx(-1.0)

    def test_hand_computed_value(self) -> None:
        # d = [+1, -2, +3, +4]  ranks 1..4, W+ = 8, W- = 2, r = 6/10
        a = [1.0, 0.0, 3.0, 4.0]
        b = [0.0, 2.0, 0.0, 0.0]
        assert rank_biserial_correlation(a, b) == pytest.approx(0.6)

    def test_is_antisymmetric(self) -> None:
        rng = np.random.default_rng(0)
        a, b = rng.random(7), rng.random(7)
        assert rank_biserial_correlation(a, b) == pytest.approx(-rank_biserial_correlation(b, a))


class TestWilcoxon:
    def test_returns_named_result(self) -> None:
        res = wilcoxon_signed_rank([1, 2, 3], [0, 0, 0])
        assert isinstance(res, WilcoxonResult) and res.method == "exact" and res.n == 3

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
    def test_minimum_attainable_p_for_small_n(self, n: int) -> None:
        """All folds favour a: the p-value is the floor, 2 / 2^n two-sided."""
        a = np.arange(1, n + 1, dtype=float)
        b = np.zeros(n)
        res = wilcoxon_signed_rank(a, b)
        assert res.statistic == n * (n + 1) / 2
        assert res.p_value == pytest.approx(min(1.0, 2 / 2**n))
        assert res.min_p_value == pytest.approx(min(1.0, 2 / 2**n))
        greater = wilcoxon_signed_rank(a, b, alternative="greater")
        assert greater.p_value == pytest.approx(1 / 2**n)
        assert greater.min_p_value == pytest.approx(1 / 2**n)
        # "less" is the mirror image: this data is as far from it as possible,
        # but its floor is still 1 / 2^n, reached when every fold favours b.
        less = wilcoxon_signed_rank(a, b, alternative="less")
        assert less.p_value == pytest.approx(1.0)
        assert less.min_p_value == pytest.approx(1 / 2**n)
        assert wilcoxon_signed_rank(b, a, alternative="less").p_value == pytest.approx(1 / 2**n)

    def test_five_folds_cannot_reach_005(self) -> None:
        """The thesis case: n=5 can never reject at 0.05 two-sided."""
        res = wilcoxon_signed_rank([0.58, 0.61, 0.55, 0.60, 0.57], [0.56, 0.57, 0.54, 0.55, 0.56])
        assert res.min_p_value == pytest.approx(0.0625)
        assert res.min_p_value > 0.05

    def test_zero_differences_are_dropped(self) -> None:
        res = wilcoxon_signed_rank([1.0, 2.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0])
        assert res.n == 2

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_matches_brute_force_with_ties(self, alternative: Alternative) -> None:
        # Eighths, so the differences are exact in binary floating point and
        # the intended tie group really forms: six |d| = 0.25, one 0.5, one zero.
        a = np.array([0.375, 0.625, 0.250, 1.000, 0.500, 0.125, 0.875, 0.750])
        b = np.array([0.125, 0.875, 0.500, 0.750, 0.500, 0.375, 0.625, 0.250])
        d = a - b
        nonzero = np.abs(d[d != 0])
        assert np.count_nonzero(nonzero == 0.25) == 6 and np.unique(nonzero).size == 2
        res = wilcoxon_signed_rank(a, b, alternative=alternative)
        assert res.p_value == pytest.approx(_brute_force_p(a - b, alternative), abs=1e-12)

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_matches_scipy_without_ties(self, alternative: Alternative) -> None:
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(1)
        for n in (5, 9, 20):
            a, b = rng.random(n), rng.random(n)
            ours = wilcoxon_signed_rank(a, b, alternative=alternative)
            theirs = stats.wilcoxon(a, b, alternative=alternative, method="exact")
            assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-10)
            if alternative != "two-sided":
                assert ours.statistic == pytest.approx(theirs.statistic)

    def test_normal_approximation_above_exact_limit(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(2)
        n = EXACT_MAX_N + 10
        a, b = rng.random(n) + 0.05, rng.random(n)
        ours = wilcoxon_signed_rank(a, b)
        theirs = stats.wilcoxon(a, b, method="approx", correction=False)
        assert ours.method == "normal"
        assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-8)
        assert ours.min_p_value < 1e-10

    def test_normal_approximation_corrects_for_ties(self) -> None:
        """Above the exact limit the variance must be sum(r^2)/4, not n(n+1)(2n+1)/24."""
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(4)
        n = EXACT_MAX_N + 11
        diffs = rng.choice(np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), size=n)
        a, b = np.zeros(n), -diffs
        ours = wilcoxon_signed_rank(a, b)
        theirs = stats.wilcoxon(a, b, method="approx", correction=False)
        assert ours.method == "normal" and ours.n == n
        assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-8)

        # The tie correction is not cosmetic here: dropping it moves the p-value.
        ranks = _average_ranks(np.abs(diffs))
        w_plus = float(ranks[diffs > 0].sum())
        uncorrected_z = (w_plus - n * (n + 1) / 4) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        uncorrected_p = min(1.0, 2 * 0.5 * math.erfc(abs(uncorrected_z) / math.sqrt(2)))
        assert uncorrected_p != pytest.approx(theirs.pvalue, rel=1e-3)

        # min_p_value floors in this branch too, for both one-sided directions.
        assert wilcoxon_signed_rank(a, b, alternative="less").min_p_value < 1e-10
        assert wilcoxon_signed_rank(a, b, alternative="greater").min_p_value < 1e-10

    def test_p_value_is_symmetric_in_argument_order(self) -> None:
        a, b = [0.3, 0.5, 0.2, 0.8, 0.45], [0.1, 0.6, 0.4, 0.7, 0.4]
        assert wilcoxon_signed_rank(a, b).p_value == pytest.approx(
            wilcoxon_signed_rank(b, a).p_value
        )
        assert wilcoxon_signed_rank(a, b, alternative="greater").p_value == pytest.approx(
            wilcoxon_signed_rank(b, a, alternative="less").p_value
        )

    def test_all_zero_differences_raise(self) -> None:
        with pytest.raises(ValueError, match="every paired difference is zero"):
            wilcoxon_signed_rank([0.5, 0.6], [0.5, 0.6])

    @pytest.mark.parametrize(
        "a, b, match",
        [
            ([1, 2], [1], "same length"),
            ([], [], "no scores"),
            ([1, float("nan")], [0, 0], "finite"),
        ],
    )
    def test_input_validation(self, a: list, b: list, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            wilcoxon_signed_rank(a, b)
        with pytest.raises(ValueError, match=match):
            rank_biserial_correlation(a, b)

    def test_bad_alternative(self) -> None:
        with pytest.raises(ValueError, match="alternative must be"):
            wilcoxon_signed_rank([1], [0], alternative="both")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Several models over several datasets (#204)
# ---------------------------------------------------------------------------

# Demšar (2006), Section 3.2.4: four C4.5 variants over 14 datasets.  The paper
# publishes the average ranks to three decimals; as ranks over 14 datasets
# they are multiples of 1/28, which recovers them exactly and they sum to
# k(k+1)/2 = 10 as they must.
DEMSAR_RANKS = [44 / 14, 28 / 14, 40.5 / 14, 27.5 / 14]
DEMSAR_N = 14

# Demšar (2006), Table 5(a): Nemenyi q_α (studentised range / sqrt 2), k = 2..10.
DEMSAR_Q = {
    0.05: [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
    0.10: [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920],
}


class TestFriedmanAgainstDemsar:
    def test_published_ranks_round_as_printed(self) -> None:
        assert [round(r, 3) for r in DEMSAR_RANKS] == [3.143, 2.0, 2.893, 1.964]

    def test_statistics_of_the_worked_example(self) -> None:
        res = friedman_from_ranks(DEMSAR_RANKS, DEMSAR_N)
        assert round(res.statistic, 2) == 9.28  # χ²_F in the paper
        assert round(res.iman_davenport, 2) == 3.69  # F_F in the paper
        assert (res.n_datasets, res.n_models) == (14, 4)
        # F(3, 39) critical value at 0.05 is 2.85: the paper rejects the null.
        # Tails on (k-1, (k-1)(N-1)) = (3, 39) and on k-1 = 3 degrees of freedom:
        assert res.iman_davenport_p_value == pytest.approx(0.019823, abs=1e-6)
        assert res.p_value == pytest.approx(0.025807, abs=1e-6)

    def test_ranks_as_printed_in_the_paper(self) -> None:
        # The docstring's use case: ranks copied from a table, rounded.
        res = friedman_from_ranks([3.143, 2.000, 2.893, 1.964], DEMSAR_N)
        assert round(res.statistic, 2) == 9.28
        assert round(res.iman_davenport, 2) == 3.69
        # Two decimals, summing to 5.99 rather than 6, are still accepted.
        friedman_from_ranks([1.33, 2.33, 2.33], 10)

    def test_nemenyi_critical_difference(self) -> None:
        assert round(nemenyi_critical_difference(4, DEMSAR_N, 0.05), 2) == 1.25

    @pytest.mark.parametrize("alpha", [0.05, 0.10])
    def test_nemenyi_q_table(self, alpha: float) -> None:
        for k, q in zip(range(2, 11), DEMSAR_Q[alpha]):
            cd = nemenyi_critical_difference(k, 6, alpha)
            computed_q = cd / math.sqrt(k * (k + 1) / 36)
            # The printed table is rounded to three decimals.
            assert computed_q == pytest.approx(q, abs=1.1e-3), k

    def test_holm_against_the_control(self) -> None:
        # The paper's post-hoc example: C4.5 (column 0) as the control.
        rows = [
            c._asdict()
            for c in statistics._compare_ranks_to_control(np.array(DEMSAR_RANKS), DEMSAR_N, 0)
        ]
        by_model = {r["model"]: r for r in rows}
        assert (
            round(abs(by_model[3]["z"]), 3) == 2.415 and round(by_model[3]["p_value"], 3) == 0.016
        )
        assert (
            round(abs(by_model[1]["z"]), 3) == 2.342 and round(by_model[1]["p_value"], 3) == 0.019
        )
        assert (
            round(abs(by_model[2]["z"]), 3) == 0.512 and round(by_model[2]["p_value"], 3) == 0.608
        )
        # Holm at 0.05: 0.016 < 0.05/3 and 0.019 < 0.05/2 reject; 0.608 does not.
        assert by_model[3]["p_adjusted"] <= 0.05 and by_model[1]["p_adjusted"] <= 0.05
        assert by_model[2]["p_adjusted"] > 0.05


class TestFriedmanFromScores:
    def test_ranks_best_first_with_ties_averaged(self) -> None:
        scores = [[0.9, 0.8, 0.8], [0.5, 0.7, 0.6]]
        np.testing.assert_allclose(
            average_ranks(scores), [(1 + 3) / 2, (2.5 + 1) / 2, (2.5 + 2) / 2]
        )
        np.testing.assert_allclose(
            average_ranks(scores, higher_is_better=False),
            [(3 + 1) / 2, (1.5 + 3) / 2, (1.5 + 2) / 2],
        )

    def test_scores_and_ranks_agree(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.random((9, 5))
        from_scores = friedman_test(scores)
        from_ranks = friedman_from_ranks(average_ranks(scores), 9)
        assert from_scores.statistic == from_ranks.statistic
        assert from_scores.iman_davenport_p_value == from_ranks.iman_davenport_p_value
        comparisons = compare_to_control(scores, 2)
        assert [c.model for c in comparisons] == [0, 1, 3, 4]

    def test_identical_rankings_everywhere(self) -> None:
        scores = np.tile([0.9, 0.7, 0.5], (6, 1))
        res = friedman_test(scores)
        assert res.iman_davenport == math.inf and res.iman_davenport_p_value == 0.0
        assert res.statistic == pytest.approx(12.0) and res.p_value < 0.01

    def test_no_difference_at_all(self) -> None:
        res = friedman_test(np.full((5, 3), 0.4))
        assert res.statistic == 0.0 and res.p_value == 1.0
        assert res.iman_davenport == 0.0 and res.iman_davenport_p_value == 1.0

    def test_matches_scipy_without_ties(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(3)
        for n, k in [(5, 3), (14, 4), (30, 6)]:
            scores = rng.random((n, k))
            ours = friedman_test(scores)
            theirs = stats.friedmanchisquare(*scores.T)  # tie-corrected; no ties here
            assert ours.statistic == pytest.approx(theirs.statistic, rel=1e-12)
            assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-10)

    @pytest.mark.parametrize(
        ("scores", "match"),
        [
            (np.zeros(4), r"\(n_datasets, n_models\) matrix"),
            (np.zeros((1, 3)), "at least 2 datasets and 2 models"),
            (np.zeros((3, 1)), "at least 2 datasets and 2 models"),
            (np.array([[0.1, np.nan], [0.2, 0.3]]), "finite"),
        ],
    )
    def test_rejected_inputs(self, scores: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            friedman_test(scores)

    def test_ranks_must_sum_correctly(self) -> None:
        with pytest.raises(ValueError, match=r"sum to k\(k\+1\)/2 = 6.0"):
            friedman_from_ranks([1.0, 2.0, 2.0], 5)
        with pytest.raises(ValueError, match="must sum"):
            friedman_from_ranks([1.33, 2.33, 2.32], 10)  # off by 0.02 > 0.015

    def test_control_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="control must be a column index"):
            compare_to_control(np.random.default_rng(0).random((4, 3)), 3)


class TestHolm:
    def test_hand_worked_example(self) -> None:
        # Sorted: 0.005·4 = 0.02, 0.01·3 = 0.03, 0.03·2 = 0.06, 0.04·1 = 0.04 → 0.06.
        adjusted = holm_correction([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])

    def test_capped_at_one_and_order_preserved(self) -> None:
        np.testing.assert_allclose(holm_correction([0.6, 0.2, 0.9]), [1.0, 0.6, 1.0])

    def test_single_p_value_is_unchanged(self) -> None:
        assert holm_correction([0.037]).tolist() == [0.037]

    @pytest.mark.parametrize("p", [[], [0.2, 1.2], [-0.1]])
    def test_rejected(self, p: list) -> None:
        with pytest.raises(ValueError):
            holm_correction(p)


class TestDistributionTails:
    """The NumPy-only tails against SciPy, where SciPy is installed."""

    def test_chi_square(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        for df in (1, 3, 9, 40):
            for x in (0.2, 1.0, 5.0, 9.28, 30.0):
                assert statistics._chi2_sf(x, df) == pytest.approx(stats.chi2.sf(x, df), rel=1e-10)

    def test_f(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        for d1, d2 in [(1, 5), (3, 39), (4, 100), (9, 13)]:
            for f in (0.1, 1.0, 3.69, 12.0):
                assert statistics._f_sf(f, d1, d2) == pytest.approx(
                    stats.f.sf(f, d1, d2), rel=1e-10
                )

    def test_studentized_range(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        for k in (2, 4, 7, 12):
            want = stats.studentized_range.isf(0.05, k, np.inf)
            assert statistics._studentized_range_isf(0.05, k) == pytest.approx(want, abs=1e-8)
