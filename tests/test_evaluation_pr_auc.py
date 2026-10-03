"""
tests/test_evaluation_pr_auc.py
===============================
hqnn_forge.evaluation.pr_auc (#171): average precision with step
interpolation, ties as one operating point, no scikit-learn at runtime.

Hand-computed cases list the operating points by decreasing threshold as
(recall, precision); AP = Σ (R_k − R_{k−1}) · P_k.
"""

from __future__ import annotations

import math

import pytest
import torch

from hqnn_forge.evaluation import METRICS, find_optimal_threshold, pr_auc


class TestHandComputed:
    def test_distinct_probabilities(self) -> None:
        """
        0.9 (+), 0.8 (−), 0.4 (+), 0.35 (+), 0.1 (−), 3 positives:
        (1/3, 1), (1/3, 1/2), (2/3, 2/3), (1, 3/4), (1, 3/5)
        → 1/3·1 + 0 + 1/3·2/3 + 1/3·3/4 + 0 = 29/36.
        """
        assert pr_auc([0, 1, 1, 0, 1], [0.1, 0.4, 0.35, 0.8, 0.9]) == pytest.approx(29 / 36)

    def test_tied_samples_are_one_operating_point(self) -> None:
        """
        0.9 (+), then 0.5 shared by one positive and one negative:
        (1/2, 1), (1, 2/3) → 1/2 + 1/2·2/3 = 5/6.  Taken one sample at a time
        with the positive first, it would be 1/2 + 1/2·1 = 1.0.
        """
        assert pr_auc([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1]) == pytest.approx(5 / 6)

    def test_tie_order_does_not_matter(self) -> None:
        y = torch.tensor([1, 0, 1, 0, 1, 0])
        p = torch.tensor([0.5, 0.5, 0.5, 0.2, 0.9, 0.9])
        expected = pr_auc(y, p)
        for seed in range(5):
            perm = torch.randperm(6, generator=torch.Generator().manual_seed(seed))
            assert pr_auc(y[perm], p[perm]) == pytest.approx(expected, abs=1e-15)

    def test_step_not_trapezoid(self) -> None:
        """
        Operating points (1/2, 1), (1/2, 1/2), (1, 2/3): step AP is
        1/2 + 0 + 1/2·2/3 = 5/6.  The trapezoid over the same points, anchored
        at (0, 1), averages the precisions either side of the last recall gain
        instead: 1/2 + 0 + 1/2·(1/2 + 2/3)/2 = 19/24.  Precision changes across
        that gain, which is what lets the two rules disagree.
        """
        assert pr_auc([1, 0, 1], [0.9, 0.7, 0.5]) == pytest.approx(5 / 6)

    def test_perfect_ranking_is_one(self) -> None:
        assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_all_tied_is_the_prevalence(self) -> None:
        assert pr_auc([1, 0, 0, 0, 1], [0.3] * 5) == pytest.approx(2 / 5)

    def test_only_the_ranking_matters(self) -> None:
        y = [0, 1, 1, 0, 1]
        prob = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.9])
        assert pr_auc(y, torch.logit(prob)) == pytest.approx(pr_auc(y, prob))


class TestSingleClass:
    def test_no_positives_is_zero(self) -> None:
        assert pr_auc([0, 0, 0], [0.2, 0.5, 0.9]) == 0.0

    def test_only_positives_is_one(self) -> None:
        assert pr_auc([1, 1, 1], [0.2, 0.5, 0.9]) == 1.0


class TestAgainstScikitLearn:
    @pytest.mark.parametrize("seed", range(10))
    def test_matches_average_precision_score(self, seed: int) -> None:
        """Random labels and probabilities, rounded to one decimal so ties are common."""
        sk = pytest.importorskip("sklearn.metrics")
        gen = torch.Generator().manual_seed(seed)
        y = (torch.rand(60, generator=gen) < 0.2).long()
        y[0] = 1  # at least one positive
        p = torch.round(torch.rand(60, generator=gen), decimals=1)
        expected = sk.average_precision_score(y.numpy(), p.numpy())
        assert pr_auc(y, p) == pytest.approx(expected, abs=1e-12)

    def test_no_positives_matches(self) -> None:
        sk = pytest.importorskip("sklearn.metrics")
        with pytest.warns(UserWarning):
            expected = sk.average_precision_score([0, 0, 0], [0.2, 0.5, 0.9])
        assert pr_auc([0, 0, 0], [0.2, 0.5, 0.9]) == pytest.approx(expected)


class TestValidation:
    @pytest.mark.parametrize(
        ("y", "p", "match"),
        [
            ([], [], "empty"),
            ([0, 1], [0.5], "differ in length"),
            ([0, 2], [0.1, 0.5], "0/1 labels"),
            ([0, 1], [0.1, math.nan], "NaN"),
            ([0, 1], [0.1, math.inf], "inf"),
        ],
    )
    def test_bad_inputs_raise(self, y: list, p: list, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            pr_auc(y, p)


class TestKeptOutOfTheThresholdSearch:
    """Every METRICS entry scores hard labels; a threshold-free metric must fail loudly."""

    def test_not_in_metrics(self) -> None:
        assert "pr_auc" not in METRICS
        assert pr_auc not in METRICS.values()

    @pytest.mark.parametrize("metric", ["pr_auc", pr_auc])
    def test_find_optimal_threshold_refuses_it(self, metric: object) -> None:
        with pytest.raises(ValueError, match="threshold-free"):
            find_optimal_threshold([0, 1, 1], [0.2, 0.6, 0.9], metric=metric)  # type: ignore[arg-type]
