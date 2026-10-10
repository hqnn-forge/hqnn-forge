"""
tests/test_separation.py
========================
The separation measures of ``hqnn_forge.diagnostics.separation`` (#500) on
constructed data with known answers.

Every expected value is obtained another way than the module obtains it:

* typed in from a hand calculation on a handful of points (distances, the
  pair order, the pooled variance, the Gram matrix);
* closed forms for two blobs whose class means and class covariances are set
  exactly (:func:`exact_blob`), derived in the test that uses them;
* the population values of Gaussian blobs (chi and folded-normal means), to
  the sampling error of the draw;
* an independent route: NumPy's ``cov`` with SciPy's ``entropy``,
  ``mahalanobis`` and ``minimize``, and for the Rydberg features the state
  kernel of ``hqnn_forge.kernels``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
import torch

from hqnn_forge.diagnostics import (
    PairwiseDistances,
    SeparationMeasures,
    effective_rank,
    fisher_discriminant_ratio,
    linear_feature_kernel,
    pairwise_distances,
    separation_measures,
)
from hqnn_forge.kernels import kernel_from_density_matrices, kernel_target_alignment
from hqnn_forge.rydberg import AtomRegister, PulseEncoding, RydbergFeatureMap, readout

F64 = torch.float64


def normal(n_samples: int, n_features: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_samples, n_features, generator=generator, dtype=F64)


def exact_blob(n_samples: int, centre: list[float], sigma: float, seed: int) -> torch.Tensor:
    """
    A Gaussian draw moved so that its sample mean is ``centre`` and its sample
    covariance (denominator ``n_samples − 1``) is ``sigma² · 1``, to rounding.

    With ``cov = L Lᵀ`` the rows ``z = L⁻¹ x`` of the centred draw have the
    covariance ``L⁻¹ cov L⁻ᵀ = 1``.
    """
    draw = normal(n_samples, len(centre), seed)
    draw = draw - draw.mean(dim=0)
    lower = torch.linalg.cholesky(draw.T @ draw / (n_samples - 1))
    white = torch.linalg.solve_triangular(lower, draw.T, upper=False).T
    return sigma * white + torch.tensor(centre, dtype=F64)


def two_blobs(
    n0: int, n1: int, delta: list[float], sigma: float, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class 0 at the origin and class 1 at ``delta``, both with covariance ``sigma² · 1``."""
    features = torch.cat(
        [
            exact_blob(n0, [0.0] * len(delta), sigma, seed),
            exact_blob(n1, delta, sigma, seed + 1),
        ]
    )
    labels = torch.cat([torch.zeros(n0, dtype=torch.int64), torch.ones(n1, dtype=torch.int64)])
    return features, labels


#: Four points in the plane; rows 0 and 2 are class 0, rows 1 and 3 class 1.
FOUR_POINTS = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0], [6.0, 8.0]], dtype=F64)
FOUR_LABELS = torch.tensor([0, 1, 0, 1])


# ---------------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------------


class TestPairwiseDistances:
    def test_four_points_by_hand_in_the_documented_pair_order(self) -> None:
        """
        The pairs ``(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`` have the
        distances ``5, 1, 10, √18, 5, √85``; ``(0,2)`` and ``(1,3)`` are the
        pairs within a class.
        """
        result = pairwise_distances(FOUR_POINTS, FOUR_LABELS)
        assert isinstance(result, PairwiseDistances)
        assert result.within.tolist() == [1.0, 5.0]
        assert result.between.tolist() == pytest.approx(
            [5.0, 10.0, math.sqrt(18.0), math.sqrt(85.0)], abs=1e-14
        )
        assert result.within.dtype == F64 and result.between.dtype == F64
        assert result.mean_within == 3.0
        mean_between = (15.0 + math.sqrt(18.0) + math.sqrt(85.0)) / 4
        assert result.mean_between == pytest.approx(mean_between, abs=1e-14)
        assert result.ratio == pytest.approx(mean_between / 3.0, abs=1e-14)

    def test_the_pair_counts_are_those_of_the_class_sizes(self) -> None:
        """``n0(n0−1)/2 + n1(n1−1)/2`` pairs within a class and ``n0 · n1`` between."""
        features = normal(11, 3, seed=0)
        labels = torch.tensor([0] * 4 + [1] * 7)
        result = pairwise_distances(features, labels)
        assert result.within.numel() == 4 * 3 // 2 + 7 * 6 // 2
        assert result.between.numel() == 4 * 7

    def test_separated_on_a_line_the_between_mean_is_the_difference_of_the_class_means(
        self,
    ) -> None:
        """
        One feature, every class-1 value above every class-0 value: each
        between-class distance is ``b − a``, so their mean is ``μ1 − μ0``.
        Within a class the mean is Gini's mean difference,
        ``2 Σ_k (2k − n − 1) x_(k) / (n(n − 1))`` over the sorted values.
        """
        low = torch.tensor([0.1, -0.4, 0.3, 0.25, -0.05], dtype=F64)
        high = torch.tensor([2.0, 2.6, 1.9], dtype=F64)
        features = torch.cat([low, high]).unsqueeze(1)
        labels = torch.tensor([0] * 5 + [1] * 3)
        result = pairwise_distances(features, labels)
        assert result.mean_between == pytest.approx(float(high.mean() - low.mean()), abs=1e-14)

        def gini_sum(values: torch.Tensor) -> float:
            ordered = sorted(values.tolist())
            n = len(ordered)
            return sum((2 * k - n - 1) * x for k, x in enumerate(ordered, start=1))

        within = (gini_sum(low) + gini_sum(high)) / (5 * 4 // 2 + 3 * 2 // 2)
        assert result.mean_within == pytest.approx(within, abs=1e-14)

    def test_gaussian_blobs_have_the_chi_and_noncentral_chi_means(self) -> None:
        """
        Two blobs ``N(0, σ² 1)`` and ``N(δ, σ² 1)`` in 3 dimensions.  A
        difference within a class is ``N(0, 2σ² 1)``, whose norm has the mean
        ``√2 σ · E[χ_3] = 2σ Γ(2)/Γ(3/2)``; a difference between the classes
        is ``N(δ, 2σ² 1)``, whose squared norm over ``2σ²`` is noncentral
        chi-squared with 3 degrees of freedom and noncentrality
        ``|δ|²/(2σ²)``.  The sampled means follow the sample's scale, which
        1500 samples of 3 features per class fix to about
        ``1/√(2 · 4500) = 1 %``: over six seeds they were within 1.5 % of both
        values, and the tolerance is 4 %.
        """
        stats = pytest.importorskip("scipy.stats")
        sigma, delta, n = 0.7, [1.5, -0.5, 1.0], 1500
        features = torch.cat(
            [sigma * normal(n, 3, seed=1), sigma * normal(n, 3, seed=2) + torch.tensor(delta)]
        )
        labels = torch.tensor([0] * n + [1] * n)
        result = pairwise_distances(features, labels)

        within = 2 * sigma * math.gamma(2.0) / math.gamma(1.5)
        noncentrality = sum(x * x for x in delta) / (2 * sigma**2)
        between = math.sqrt(2) * sigma * stats.ncx2(3, noncentrality).expect(np.sqrt)
        assert result.mean_within == pytest.approx(within, rel=4e-2)
        assert result.mean_between == pytest.approx(between, rel=4e-2)
        assert result.ratio == pytest.approx(between / within, rel=4e-2)
        assert between / within > 1.3  # 1.38: the classes overlap

    def test_labels_that_carry_no_information_give_a_ratio_of_one(self) -> None:
        """
        One blob with labels drawn independently of the features: a pair is
        within a class or between the classes whatever its distance, so both
        means estimate the same mean distance (here to 1 %).
        """
        features = normal(2000, 4, seed=3)
        generator = torch.Generator().manual_seed(4)
        labels = (torch.rand(2000, generator=generator) < 0.3).to(torch.int64)
        assert pairwise_distances(features, labels).ratio == pytest.approx(1.0, abs=1e-2)

    def test_identical_features_have_zero_distances_and_no_ratio(self) -> None:
        """Every distance is exactly 0, and ``0/0`` is reported as NaN, not as a number."""
        features = torch.tensor([[0.1, 0.7, 0.437]], dtype=F64).repeat(7, 1)
        result = pairwise_distances(features, torch.tensor([0, 1, 0, 1, 1, 0, 1]))
        assert result.mean_between == 0.0
        assert result.mean_within == 0.0
        assert math.isnan(result.ratio)

    def test_classes_collapsed_to_two_points_have_an_infinite_ratio(self) -> None:
        features = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=F64)
        result = pairwise_distances(features, torch.tensor([0, 0, 1, 1]))
        assert result.mean_within == 0.0
        assert result.mean_between == 1.0
        assert result.ratio == math.inf

    def test_one_sample_per_class_has_no_within_class_pair(self) -> None:
        result = pairwise_distances(torch.tensor([[0.0], [2.0]], dtype=F64), torch.tensor([0, 1]))
        assert result.within.numel() == 0
        assert result.between.tolist() == [2.0]
        assert math.isnan(result.mean_within) and math.isnan(result.ratio)

    def test_distances_of_nearly_equal_rows_keep_their_relative_precision(self) -> None:
        """
        Rows ``½ + ε g`` with ``ε = 1e-12``: the distances are ``ε`` times
        those of ``g``, to the rounding of the rows themselves (``1e-16``
        against ``1e-12``), which a distance computed from squared norms
        would lose entirely.
        """
        base = normal(6, 3, seed=5)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        reference = pairwise_distances(base, labels)
        scaled = pairwise_distances(0.5 + 1e-12 * base, labels)
        assert torch.allclose(scaled.between, 1e-12 * reference.between, rtol=1e-3, atol=0)
        assert scaled.ratio == pytest.approx(reference.ratio, rel=1e-3)


# ---------------------------------------------------------------------------
# Fisher discriminant ratio
# ---------------------------------------------------------------------------


def fisher_reference(features: torch.Tensor, labels: torch.Tensor) -> float:
    """``dᵀ S_w⁻¹ d`` with the pooled covariance assembled from ``numpy.cov`` per class."""
    x, y = features.numpy(), labels.numpy()
    x0, x1 = x[y == y.min()], x[y == y.max()]
    pooled = (
        (len(x0) - 1) * np.atleast_2d(np.cov(x0, rowvar=False))
        + (len(x1) - 1) * np.atleast_2d(np.cov(x1, rowvar=False))
    ) / (len(x) - 2)
    difference = x1.mean(axis=0) - x0.mean(axis=0)
    return float(difference @ np.linalg.solve(pooled, difference))


class TestFisherDiscriminantRatio:
    def test_one_feature_by_hand(self) -> None:
        """
        Class 0 is ``{−1, 1}`` and class 1 ``{3, 5}``: the means differ by 4,
        each class has the scatter ``1 + 1 = 2``, the pooled variance is
        ``(2 + 2)/(4 − 2) = 2``, and the ratio ``4²/2 = 8``.
        """
        features = torch.tensor([[-1.0], [1.0], [3.0], [5.0]], dtype=F64)
        assert fisher_discriminant_ratio(features, torch.tensor([0, 0, 1, 1])) == pytest.approx(
            8.0, abs=1e-13
        )

    def test_unequal_class_sizes_by_hand_and_not_the_other_two_conventions(self) -> None:
        """
        Class 0 is ``{−1, 1}`` and class 1 ``{2, 4, 6}``: the means differ by
        4, the sums of squares within the classes are 2 and 8, the pooled
        variance is ``(2 + 8)/(5 − 2)`` and the ratio ``16 · 3/10 = 4.8``.
        The sums themselves in the denominator would give ``16/10 = 1.6``,
        the sum of the class variances ``16/(2/1 + 8/2) = 2.67``.
        """
        features = torch.tensor([[-1.0], [1.0], [2.0], [4.0], [6.0]], dtype=F64)
        labels = torch.tensor([0, 0, 1, 1, 1])
        assert fisher_discriminant_ratio(features, labels) == pytest.approx(4.8, abs=1e-13)

    @pytest.mark.parametrize("copies", [1, 2, 10, 1000])
    def test_repeating_every_sample_approaches_a_limit_instead_of_scaling(
        self, copies: int
    ) -> None:
        """
        The five points above, each ``k`` times: the means are unchanged, the
        sums of squares are ``10 k`` and ``M − 2 = 5k − 2``, so the ratio is
        ``16 (5k − 2)/(10 k)``, which rises from 4.8 to the limit 8 (the
        ratio with the variance over ``M``).  It is not proportional to
        ``M`` or to ``1/M``.
        """
        features = torch.tensor([[-1.0], [1.0], [2.0], [4.0], [6.0]], dtype=F64).repeat(copies, 1)
        labels = torch.tensor([0, 0, 1, 1, 1]).repeat(copies)
        expected = 16.0 * (5 * copies - 2) / (10 * copies)
        assert fisher_discriminant_ratio(features, labels) == pytest.approx(expected, rel=1e-12)

    def test_unequal_class_covariances_are_weighted_by_class_size(self) -> None:
        """
        Class covariances exactly ``σ0² 1`` and ``σ1² 1``: the pooled one is
        ``((n0 − 1) σ0² + (n1 − 1) σ1²)/(M − 2) · 1``, so the larger class
        counts for more, and exchanging the two sizes changes the ratio
        (here by ``825/305``, a factor of 2.7).  With the two covariances summed it would
        be ``|δ|²/(σ0² + σ1²)`` for both.
        """
        delta, norm2 = [1.2, -0.5, 2.0], 1.2**2 + 0.5**2 + 2.0**2
        values = []
        for n0, n1 in [(25, 90), (90, 25)]:
            features = torch.cat(
                [exact_blob(n0, [0.0] * 3, 1.0, seed=20), exact_blob(n1, delta, 3.0, seed=21)]
            )
            labels = torch.tensor([0] * n0 + [1] * n1)
            pooled = ((n0 - 1) * 1.0 + (n1 - 1) * 9.0) / (n0 + n1 - 2)
            values.append(fisher_discriminant_ratio(features, labels))
            assert values[-1] == pytest.approx(norm2 / pooled, rel=1e-10)
        assert values[1] / values[0] == pytest.approx(825 / 305, rel=1e-10)

    @pytest.mark.parametrize(("n0", "n1", "delta"), [(45, 5, 0.0), (20, 20, 0.0), (45, 5, 1.0)])
    def test_the_mean_over_gaussian_samples_is_the_documented_one(
        self, n0: int, n1: int, delta: float
    ) -> None:
        """
        ``E[J] = (M − 2)/(M − d − 3) · (Δ² + d M/(n0 n1))`` for Gaussian
        classes of covariance 1 whose means differ by ``delta`` along the
        first of ``d = 4`` features: 0.99 for 45 against 5 samples without
        any class difference and 0.46 for 20 against 20, not 0.  The mean of
        3000 samples was within 3.2 % of it over eight seeds in each of the
        three cases; the tolerance is 6 %, and the factor in front alone is
        12 % at ``M = 50``.
        """
        d, draws = 4, 3000
        m = n0 + n1
        expected = (m - 2) / (m - d - 3) * (delta**2 + d * m / (n0 * n1))
        labels = torch.tensor([0] * n0 + [1] * n1)
        samples = normal(draws * m, d, seed=22).reshape(draws, m, d)
        samples[:, n0:, 0] += delta
        mean = sum(fisher_discriminant_ratio(sample, labels) for sample in samples) / draws
        assert mean == pytest.approx(expected, rel=6e-2)

    @pytest.mark.parametrize(("n0", "n1"), [(40, 40), (25, 90)])
    def test_two_blobs_give_the_squared_distance_in_units_of_sigma(self, n0: int, n1: int) -> None:
        """
        Both class covariances are exactly ``σ² 1``, so the pooled one is
        too, and ``dᵀ S_w⁻¹ d = |δ|²/σ²`` whatever the class sizes.
        """
        delta, sigma = [1.2, -0.5, 2.0], 0.8
        features, labels = two_blobs(n0, n1, delta, sigma)
        expected = sum(x * x for x in delta) / sigma**2
        assert fisher_discriminant_ratio(features, labels) == pytest.approx(expected, rel=1e-10)

    def test_matches_the_pooled_covariance_from_numpy_on_correlated_classes(self) -> None:
        """Unequal class sizes, unequal and correlated class covariances."""
        mix0 = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.5, 0.0], [-0.3, 0.2, 2.0]], dtype=F64)
        mix1 = torch.tensor([[0.4, 0.0, 0.0], [0.1, 1.5, 0.0], [0.6, -0.7, 0.9]], dtype=F64)
        features = torch.cat(
            [normal(30, 3, seed=6) @ mix0.T, normal(75, 3, seed=7) @ mix1.T + 0.9]
        )
        labels = torch.tensor([0] * 30 + [1] * 75)
        expected = fisher_reference(features, labels)
        assert fisher_discriminant_ratio(features, labels) == pytest.approx(expected, rel=1e-11)

    def test_is_the_squared_mahalanobis_distance_between_the_class_means(self) -> None:
        distance = pytest.importorskip("scipy.spatial.distance")
        features, labels = two_blobs(20, 35, [0.5, 1.0], 1.0, seed=8)
        features = features @ torch.tensor([[2.0, 0.3], [-0.4, 0.5]], dtype=F64)
        x, y = features.numpy(), labels.numpy()
        pooled = (19 * np.cov(x[y == 0], rowvar=False) + 34 * np.cov(x[y == 1], rowvar=False)) / 53
        mahalanobis = distance.mahalanobis(
            x[y == 0].mean(axis=0), x[y == 1].mean(axis=0), np.linalg.inv(pooled)
        )
        assert fisher_discriminant_ratio(features, labels) == pytest.approx(
            mahalanobis**2, rel=1e-11
        )

    def test_no_direction_separates_better(self) -> None:
        """
        The ratio is the maximum over directions ``w`` of
        ``(wᵀ(μ1 − μ0))² / (pooled variance of the projection on w)``, with
        each projected ratio computed on one feature: a numerical maximiser
        reaches it and random directions stay below it.
        """
        optimize = pytest.importorskip("scipy.optimize")
        features = torch.cat([normal(40, 3, seed=9), 0.6 * normal(50, 3, seed=10) + 0.8])
        labels = torch.tensor([0] * 40 + [1] * 50)
        best = fisher_discriminant_ratio(features, labels)

        def projected(direction: Any) -> float:
            w = torch.as_tensor(np.asarray(direction), dtype=F64)
            return fisher_discriminant_ratio((features @ w).unsqueeze(1), labels)

        directions = normal(200, 3, seed=11)
        assert max(projected(w) for w in directions) <= best * (1 + 1e-12)
        found = optimize.minimize(lambda w: -projected(w), x0=np.ones(3), method="Nelder-Mead")
        assert -found.fun == pytest.approx(best, rel=1e-6)

    def test_unchanged_by_an_invertible_affine_map_and_by_the_label_coding(self) -> None:
        features, labels = two_blobs(30, 45, [1.0, 0.2, -0.7], 1.3, seed=12)
        value = fisher_discriminant_ratio(features, labels)
        mix = torch.tensor([[2.0, 0.1, 0.0], [-0.5, 0.7, 0.3], [0.0, 1.1, -3.0]], dtype=F64)
        moved = features @ mix.T + torch.tensor([5.0, -2.0, 0.3], dtype=F64)
        assert fisher_discriminant_ratio(moved, labels) == pytest.approx(value, rel=1e-10)
        assert fisher_discriminant_ratio(features, 2 * labels - 1) == pytest.approx(value)
        assert fisher_discriminant_ratio(features, 1 - labels) == pytest.approx(value, rel=1e-12)

    def test_equal_class_means_give_zero(self) -> None:
        features = torch.tensor([[-1.0], [1.0], [-2.0], [2.0]], dtype=F64)
        assert fisher_discriminant_ratio(features, torch.tensor([0, 0, 1, 1])) == 0.0

    def test_a_repeated_feature_adds_nothing(self) -> None:
        """
        A second, identical column makes the within-class covariance singular
        along ``(1, −1)``, where the class means do not differ either; the
        ratio stays that of one feature (8, as by hand above).
        """
        column = torch.tensor([[-1.0], [1.0], [3.0], [5.0]], dtype=F64)
        features = torch.cat([column, column], dim=1)
        assert fisher_discriminant_ratio(features, torch.tensor([0, 0, 1, 1])) == pytest.approx(
            8.0, abs=1e-12
        )

    def test_singular_within_class_scatter_on_the_line_of_the_means_by_hand(self) -> None:
        """
        All points on the line along ``(1, 1)/√2``: class 0 at ``0`` and
        ``2√2``, class 1 at ``3√2`` and ``5√2`` on it.  The means differ by
        ``3√2``, the pooled variance is ``(4 + 4)/2 = 4``, the ratio 4.5.
        """
        features = torch.tensor([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0], [5.0, 5.0]], dtype=F64)
        assert fisher_discriminant_ratio(features, torch.tensor([0, 0, 1, 1])) == pytest.approx(
            4.5, abs=1e-12
        )

    def test_a_direction_without_within_class_spread_separates_perfectly(self) -> None:
        """
        Both classes spread along ``(1, 1)`` only and the means differ along
        ``(1, 0)``: projected on ``(1, −1)`` each class is a point and the
        two points differ, so the ratio is infinite.
        """
        features = torch.tensor([[0.0, 0.0], [2.0, 2.0], [3.0, 0.0], [5.0, 2.0]], dtype=F64)
        assert fisher_discriminant_ratio(features, torch.tensor([0, 0, 1, 1])) == math.inf

    def test_features_equal_to_the_labels_separate_perfectly(self) -> None:
        labels = torch.tensor([0, 1, 1, 0, 1])
        assert fisher_discriminant_ratio(labels.to(F64).unsqueeze(1), labels) == math.inf

    def test_identical_features_have_no_ratio(self) -> None:
        """No mean difference and no spread: ``0/0`` in every direction, reported as NaN."""
        features = torch.tensor([[0.1, 0.7, 0.437]], dtype=F64).repeat(7, 1)
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 1])
        assert math.isnan(fisher_discriminant_ratio(features, labels))


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------


class TestEffectiveRank:
    def test_an_isotropic_cloud_has_the_full_rank(self) -> None:
        """The rows ``±e_k``: the covariance is a multiple of the identity, ``p_k = 1/4``."""
        eye = torch.eye(4, dtype=F64)
        assert effective_rank(torch.cat([eye, -eye])) == pytest.approx(4.0, abs=1e-12)

    def test_two_eigenvalues_by_hand(self) -> None:
        """
        The rows ``(±√3, 0), (0, ±1)`` have the scatter ``diag(6, 2)``, so
        ``p = (¾, ¼)`` and ``exp(−¾ ln ¾ − ¼ ln ¼) = 4 / 3^(3/4)``.
        """
        root = math.sqrt(3.0)
        features = torch.tensor([[root, 0.0], [-root, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=F64)
        assert effective_rank(features) == pytest.approx(4 / 3**0.75, abs=1e-12)

    def test_features_on_one_line_have_rank_one(self) -> None:
        """Every row is ``c + t v``: five features, one direction."""
        t = normal(20, 1, seed=13)
        direction = torch.tensor([[0.3, -1.0, 2.0, 0.0, 0.5]], dtype=F64)
        assert effective_rank(0.5 + t @ direction) == pytest.approx(1.0, abs=1e-9)

    def test_identical_features_have_rank_one(self) -> None:
        """The same row for every sample: the covariance is zero, and the rank is reported as 1."""
        features = torch.tensor([[0.1, 0.7, 0.437]], dtype=F64).repeat(7, 1)
        assert effective_rank(features) == 1.0
        assert effective_rank(torch.full((5, 4), 0.5, dtype=F64)) == 1.0

    def test_matches_the_entropy_of_the_eigenvalues_of_numpy_cov(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        mix = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.3, 0.0, 0.0],
                [0.0, 0.2, 0.05, 0.0],
                [1.0, 1.0, 1.0, 2.0],
            ],
            dtype=F64,
        )
        features = normal(60, 4, seed=14) @ mix.T + 3.0
        eigenvalues = np.linalg.eigvalsh(np.cov(features.numpy(), rowvar=False))
        expected = math.exp(stats.entropy(eigenvalues))  # entropy normalises to sum 1
        assert effective_rank(features) == pytest.approx(expected, rel=1e-10)
        assert 1.0 < expected < 4.0

    def test_unchanged_by_translation_rotation_and_scale(self) -> None:
        features = normal(30, 2, seed=15) * torch.tensor([2.0, 0.5], dtype=F64)
        value = effective_rank(features)
        angle = 0.7
        rotation = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=F64
        )
        moved = 1e-9 * (features @ rotation.T) + torch.tensor([0.5, 0.5], dtype=F64)
        assert effective_rank(moved) == pytest.approx(value, rel=1e-6)

    def test_at_most_one_less_than_the_number_of_samples(self) -> None:
        """Three samples in five dimensions span a plane: the rank is in ``[1, 2]``."""
        value = effective_rank(normal(3, 5, seed=16))
        assert 1.0 <= value <= 2.0 + 1e-12

    def test_two_blobs_have_the_rank_of_their_closed_form_spectrum(self) -> None:
        """
        The total scatter is the within-class scatter plus that of the class
        means, ``(M − 2) σ² 1 + (n0 n1 / M) δ δᵀ``: the eigenvalue
        ``a = (M − 2) σ²`` twice and ``a + n0 n1 |δ|²/M`` along δ.
        """
        n0, n1, delta, sigma = 30, 50, [3.0, 0.0, 4.0], 0.5
        features, _ = two_blobs(n0, n1, delta, sigma)
        m = n0 + n1
        a = (m - 2) * sigma**2
        eigenvalues = [a, a, a + n0 * n1 * 25.0 / m]
        p = [value / sum(eigenvalues) for value in eigenvalues]
        expected = math.exp(-sum(q * math.log(q) for q in p))
        assert effective_rank(features) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Linear feature kernel and its alignment
# ---------------------------------------------------------------------------


class TestLinearFeatureKernel:
    def test_gram_matrix_by_hand(self) -> None:
        """Entry ``[i, j]`` is the inner product of rows ``i`` and ``j``: samples, not features."""
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=F64)
        kernel = linear_feature_kernel(features)
        expected = [[5.0, 11.0, 17.0], [11.0, 25.0, 39.0], [17.0, 39.0, 61.0]]
        assert kernel.tolist() == expected
        assert kernel.dtype == F64

    @pytest.mark.parametrize("coding", ["0/1", "-1/+1"])
    def test_features_equal_to_the_labels_have_alignment_one(self, coding: str) -> None:
        labels = torch.tensor([0, 1, 1, 0, 1, 0, 0])
        if coding == "-1/+1":
            labels = 2 * labels - 1
        kernel = linear_feature_kernel(labels.to(F64).unsqueeze(1))
        assert float(kernel_target_alignment(kernel, labels)) == pytest.approx(1.0, abs=1e-12)

    def test_features_affine_in_the_label_have_alignment_one(self) -> None:
        """Every feature is ``c_k + v_k y``: the centred kernel is a multiple of the ideal one."""
        labels = torch.tensor([0, 1, 1, 0, 1, 0, 0])
        features = torch.tensor([0.5, 0.2, -1.0], dtype=F64) + labels.to(F64).unsqueeze(
            1
        ) * torch.tensor([0.3, -2.0, 0.0], dtype=F64)
        alignment = kernel_target_alignment(linear_feature_kernel(features), labels)
        assert float(alignment) == pytest.approx(1.0, abs=1e-12)

    def test_one_feature_along_the_labels_and_one_across_gives_one_over_root_two(self) -> None:
        """
        ``F = [y, u]`` with ``y = (−1, −1, 1, 1)`` and ``u = (1, −1, 1, −1)``,
        both centred, orthogonal and of squared norm 4: the centred kernel is
        ``y yᵀ + u uᵀ`` with Frobenius norm ``√(16 + 16)``, its inner product
        with ``y yᵀ`` is 16, and ``|y yᵀ| = 4``, so the alignment is
        ``16 / (4 √32) = 1/√2``.  A constant added to a feature is removed by
        the centring.
        """
        labels = torch.tensor([-1, -1, 1, 1])
        across = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=F64)
        features = torch.stack([labels.to(F64), across], dim=1)
        alignment = kernel_target_alignment(linear_feature_kernel(features), labels)
        assert float(alignment) == pytest.approx(1 / math.sqrt(2), abs=1e-12)
        shifted = kernel_target_alignment(linear_feature_kernel(features + 3.0), labels)
        assert float(shifted) == pytest.approx(1 / math.sqrt(2), abs=1e-12)

    def test_two_blobs_have_the_closed_form_alignment(self) -> None:
        """
        With centred labels ``ỹ`` (``y = ±1``) and centred features ``F_c``,
        the alignment is ``|F_cᵀ ỹ|² / (|F_cᵀ F_c| · |ỹ|²)``.  Here
        ``F_cᵀ ỹ = 2 n0 n1 δ / M``, ``|ỹ|² = 4 n0 n1 / M`` and ``F_cᵀ F_c`` is
        the total scatter, with the eigenvalue ``a = (M − 2) σ²`` twice and
        ``a + s`` once, ``s = n0 n1 |δ|²/M``; the alignment is
        ``s / √(2 a² + (a + s)²)``.
        """
        n0, n1, delta, sigma = 30, 50, [3.0, 0.0, 4.0], 2.0
        features, labels = two_blobs(n0, n1, delta, sigma)
        m = n0 + n1
        a, s = (m - 2) * sigma**2, n0 * n1 * 25.0 / m
        expected = s / math.sqrt(2 * a**2 + (a + s) ** 2)
        alignment = kernel_target_alignment(linear_feature_kernel(features), labels)
        assert float(alignment) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# All measures at once
# ---------------------------------------------------------------------------


class TestSeparationMeasures:
    def test_collects_the_single_measures(self) -> None:
        features, labels = two_blobs(30, 50, [1.0, 0.5, -0.3], 0.9, seed=17)
        measures = separation_measures(features, labels)
        assert isinstance(measures, SeparationMeasures)
        distances = pairwise_distances(features, labels)
        assert measures.n_samples == 80 and measures.n_features == 3
        assert measures.mean_within_distance == distances.mean_within
        assert measures.mean_between_distance == distances.mean_between
        assert measures.distance_ratio == distances.ratio
        assert measures.fisher_ratio == fisher_discriminant_ratio(features, labels)
        assert measures.effective_rank == effective_rank(features)
        alignment = float(kernel_target_alignment(linear_feature_kernel(features), labels))
        assert measures.kernel_alignment == pytest.approx(alignment, rel=1e-12)

    def test_two_separated_blobs(self) -> None:
        """
        Blobs of ``σ = 0.1`` at distance 5: the within-class distances are of
        order σ, the between-class ones within ``|δ| ± 6σ`` or so, the Fisher
        ratio ``|δ|²/σ² = 2500``; the rank and the alignment are the closed
        forms of the tests above with ``a = (M − 2) σ²``,
        ``s = n0 n1 |δ|²/M``.
        """
        n0 = n1 = 40
        features, labels = two_blobs(n0, n1, [3.0, 0.0, 4.0], 0.1)
        measures = separation_measures(features, labels)
        assert measures.fisher_ratio == pytest.approx(2500.0, rel=1e-9)
        assert measures.mean_between_distance == pytest.approx(5.0, abs=0.05)
        assert 0.1 < measures.mean_within_distance < 0.4
        assert measures.distance_ratio > 12.0
        a, s = 78 * 0.01, 40 * 40 * 25.0 / 80
        assert measures.kernel_alignment == pytest.approx(
            s / math.sqrt(2 * a**2 + (a + s) ** 2), rel=1e-9
        )
        p = [a / (3 * a + s), a / (3 * a + s), (a + s) / (3 * a + s)]
        assert measures.effective_rank == pytest.approx(
            math.exp(-sum(q * math.log(q) for q in p)), rel=1e-9
        )
        assert measures.effective_rank < 1.03  # nearly all variance lies between the classes

    def test_identical_features(self) -> None:
        """
        Zero between-class distance and rank 1; the two ratios are ``0/0``
        (NaN), and the alignment of a kernel without any variation is 0, as
        ``kernel_target_alignment`` defines it.
        """
        features = torch.tensor([[0.1, 0.7, 0.437]], dtype=F64).repeat(9, 1)
        measures = separation_measures(features, torch.tensor([0, 1, 0, 1, 1, 0, 1, 1, 0]))
        assert measures.mean_between_distance == 0.0
        assert measures.mean_within_distance == 0.0
        assert measures.effective_rank == 1.0
        assert math.isnan(measures.distance_ratio)
        assert math.isnan(measures.fisher_ratio)
        assert measures.kernel_alignment == 0.0

    def test_features_equal_to_the_labels(self) -> None:
        labels = torch.tensor([0, 1, 1, 0, 1, 0, 0])
        measures = separation_measures(labels.to(F64).unsqueeze(1), labels)
        assert measures.kernel_alignment == pytest.approx(1.0, abs=1e-12)
        assert measures.effective_rank == 1.0
        assert measures.fisher_ratio == math.inf
        assert measures.distance_ratio == math.inf
        assert measures.mean_between_distance == 1.0

    def test_the_alignment_survives_features_that_differ_in_the_twelfth_digit(self) -> None:
        """
        ``½ + ε g`` with ``ε = 1e-12`` has the alignment, the Fisher ratio
        and the rank of ``g``: none of them depends on an offset or a scale.
        The Gram matrix of the features as they are is ``d/4`` to 12 digits
        and its centred alignment is that of its rounding, which is why
        ``separation_measures`` forms the kernel of ``F − F[0]``.
        """
        base, labels = two_blobs(20, 30, [1.0, -0.5, 0.2], 1.0, seed=18)
        reference = separation_measures(base, labels)
        squeezed = 0.5 + 1e-12 * base
        measures = separation_measures(squeezed, labels)
        assert measures.kernel_alignment == pytest.approx(reference.kernel_alignment, rel=1e-3)
        assert measures.fisher_ratio == pytest.approx(reference.fisher_ratio, rel=1e-3)
        assert measures.effective_rank == pytest.approx(reference.effective_rank, rel=1e-3)
        assert measures.mean_between_distance == pytest.approx(
            1e-12 * reference.mean_between_distance, rel=1e-3
        )

    def test_to_dict_holds_every_scalar_and_survives_json(self) -> None:
        features, labels = two_blobs(12, 15, [1.0, 0.5], 1.0, seed=19)
        measures = separation_measures(features, labels)
        record = measures.to_dict()
        assert list(record) == [
            "n_samples",
            "n_features",
            "mean_within_distance",
            "mean_between_distance",
            "distance_ratio",
            "fisher_ratio",
            "effective_rank",
            "kernel_alignment",
        ]
        assert all(type(value) in (int, float) for value in record.values())
        assert SeparationMeasures(**json.loads(json.dumps(record, allow_nan=False))) == measures

    def test_accepts_numpy_arrays_and_lists(self) -> None:
        features, labels = two_blobs(12, 15, [1.0, 0.5], 1.0, seed=20)
        expected = separation_measures(features, labels)
        assert separation_measures(features.numpy(), labels.numpy()) == expected
        assert separation_measures(features.tolist(), labels.tolist()) == expected
        assert separation_measures(features.to(torch.float32), labels).fisher_ratio == (
            pytest.approx(expected.fisher_ratio, rel=1e-4)
        )


# ---------------------------------------------------------------------------
# On the features and states of the Rydberg feature map
# ---------------------------------------------------------------------------

OMEGA = 4 * math.pi


def feature_map(n_atoms: int, gamma_over_omega: float, **settings: Any) -> RydbergFeatureMap:
    """A chain at ``V = Ω`` (interaction ``C6 = Ω`` at spacing 1 µm) with the default pulse."""
    return RydbergFeatureMap(
        AtomRegister.chain(n_atoms, 1.0),
        PulseEncoding(n_atoms, omega=OMEGA, **settings.pop("encoding", {})),
        c6=OMEGA,
        gamma=gamma_over_omega * OMEGA,
        **settings,
    )


class TestOnRydbergFeatures:
    @pytest.mark.parametrize("gamma_over_omega", [0.0, 1.0])
    def test_feature_distances_are_bounded_by_the_state_kernel(
        self, gamma_over_omega: float
    ) -> None:
        """
        ``|F(x) − F(x′)| ≤ ½ √(2^N) · |ρ(x) − ρ(x′)|`` in the Hilbert–Schmidt
        norm, which the state kernel gives as
        ``√(K(x,x) + K(x′,x′) − 2 K(x,x′))`` (``docs/rydberg-model.md``).
        The bound holds for every pair and is not empty: the largest ratio of
        the two sides is above 0.3.
        """
        n_atoms = 3
        rydberg = feature_map(n_atoms, gamma_over_omega, n_steps=200 if gamma_over_omega else None)
        inputs = 1.5 * normal(12, n_atoms, seed=21)
        labels = torch.tensor([0, 1] * 6)
        kernel = kernel_from_density_matrices(rydberg.states(inputs))
        purity = kernel.diagonal()
        state_distance = (purity[:, None] + purity[None, :] - 2 * kernel).clamp(min=0).sqrt()
        bound = 0.5 * math.sqrt(2**n_atoms) * state_distance

        features = rydberg.transform(inputs)
        distances = pairwise_distances(features, labels)
        rows, columns = torch.triu_indices(12, 12, offset=1)
        same = labels[rows] == labels[columns]
        assert bool((distances.within <= bound[rows, columns][same] + 1e-12).all())
        assert bool((distances.between <= bound[rows, columns][~same] + 1e-12).all())
        assert float((distances.between / bound[rows, columns][~same]).max()) > 0.3

    def test_the_bound_is_attained_by_states_that_differ_in_one_excitation_probability(
        self,
    ) -> None:
        """
        One atom, ``ρ = diag(1 − p, p)``: the feature is ``p``, and two such
        states differ by ``|p − p′|`` in the feature and by ``√2 |p − p′|``
        in the Hilbert–Schmidt norm, so ``½ √2 · √2 |p − p′|`` is an equality.
        """
        populations = torch.tensor([0.2, 0.9], dtype=F64)
        rho = torch.diag_embed(torch.stack([1 - populations, populations], dim=1)).to(
            torch.complex128
        )
        kernel = kernel_from_density_matrices(rho)
        state_distance = math.sqrt(float(kernel[0, 0] + kernel[1, 1] - 2 * kernel[0, 1]))
        distances = pairwise_distances(readout(rho), torch.tensor([0, 1]))
        assert distances.between.item() == pytest.approx(0.7, abs=1e-14)
        assert 0.5 * math.sqrt(2) * state_distance == pytest.approx(0.7, abs=1e-14)


class TestStrongDephasing:
    """
    ``γ = 4Ω``, the strong-dephasing row of ``docs/rydberg-model.md``: the
    state tends to the maximally mixed one and every ``⟨n_i⟩`` to ½, whatever
    the input, so the features lose what they separated.
    """

    N_ATOMS = 3

    def measures(self, v_over_omega: float, omega_t: float, n_steps: int) -> SeparationMeasures:
        """24 samples labelled by the sign of the first input, on a chain with ``V = v · Ω``."""
        omega = 2.0
        rydberg = RydbergFeatureMap(
            AtomRegister.chain(self.N_ATOMS, 1.0),
            PulseEncoding(self.N_ATOMS, omega=omega, t=omega_t / omega),
            c6=v_over_omega * omega,
            gamma=4 * omega,
            n_steps=n_steps,
            interactions=v_over_omega > 0,
        )
        inputs = 1.5 * normal(24, self.N_ATOMS, seed=24)
        return separation_measures(rydberg.transform(inputs), inputs[:, 0] > 0)

    @pytest.mark.parametrize("v_over_omega", [0.0, 1.0])
    def test_every_distance_falls_to_zero(self, v_over_omega: float) -> None:
        """
        At ``Ωt = 100`` every element of ρ is within ``10⁻¹²`` of ``1/2^N``
        (model page), so every feature is within ``4 × 10⁻¹²`` of ½ (a sum of
        4 populations) and no two rows are further apart than
        ``2 √3 · 4 × 10⁻¹² < 2 × 10⁻¹¹``.  After the π pulse of the same
        map they are 0.03 or more apart on average.
        """
        pulse = self.measures(v_over_omega, math.pi, 200)
        assert pulse.mean_within_distance > 0.03 and pulse.mean_between_distance > 0.03
        mixed = self.measures(v_over_omega, 100.0, 2000)
        assert mixed.mean_within_distance < 2e-11
        assert mixed.mean_between_distance < 2e-11

    @pytest.mark.parametrize("v_over_omega", [0.0, 1.0])
    def test_on_the_way_the_other_measures_move_towards_no_information(
        self, v_over_omega: float
    ) -> None:
        """
        The ratios, the rank and the alignment do not depend on the scale of
        the features, so a collapse alone would not change them; what moves
        them is that the features lose their dependence on the input at
        different rates.  Compared at ``Ωt = 30``, where the features still
        differ by ``10⁻⁶`` to ``10⁻⁵`` (far above rounding), with the π pulse
        of the same map: the Fisher ratio is below a third (19 → 1.6 and
        17 → 1.1 for ``V/Ω`` of 0 and 1, measured), the alignment below half
        (0.61 → 0.26, 0.66 → 0.22), the distance ratio less than half as far
        from 1 (1.47 → 1.19, 1.62 → 1.15), and the rank lower (2.91 → 2.50,
        2.65 → 1.62).  This is an observation on one seeded sample, not a
        bound: whether dephasing contracts these measures in general is open
        (#500).  At ``Ωt = 100`` the rows differ by the solver's rounding, and
        these four numbers describe that.
        """
        pulse = self.measures(v_over_omega, math.pi, 200)
        later = self.measures(v_over_omega, 30.0, 800)
        assert 1e-7 < later.mean_between_distance < 1e-3 * pulse.mean_between_distance
        assert later.fisher_ratio < pulse.fisher_ratio / 3
        assert later.kernel_alignment < pulse.kernel_alignment / 2
        assert 0 < later.distance_ratio - 1 < (pulse.distance_ratio - 1) / 2
        assert later.effective_rank < pulse.effective_rank


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    LABELS = torch.tensor([0, 1, 0, 1])

    @pytest.mark.parametrize(
        "function", [pairwise_distances, fisher_discriminant_ratio, separation_measures]
    )
    def test_functions_of_features_and_labels_reject_bad_input(self, function: Any) -> None:
        features = normal(4, 2, seed=22)
        with pytest.raises(ValueError, match=r"F must have shape \(n_samples, n_features\)"):
            function(features[:, 0], self.LABELS)
        with pytest.raises(ValueError, match="F must be finite"):
            function(torch.tensor([[0.0], [math.nan], [1.0], [2.0]]), self.LABELS)
        with pytest.raises(TypeError, match="F must hold real numbers"):
            function(features.to(torch.complex128), self.LABELS)
        with pytest.raises(ValueError, match="y must have 4 labels; got 3"):
            function(features, self.LABELS[:3])
        with pytest.raises(ValueError, match="y must hold both classes as 0/1 or -1/\\+1"):
            function(features, torch.zeros(4))
        with pytest.raises(ValueError, match="y must hold both classes as 0/1 or -1/\\+1"):
            function(features, torch.tensor([0, 1, 2, 1]))

    @pytest.mark.parametrize("function", [effective_rank, linear_feature_kernel])
    def test_functions_of_features_reject_bad_input(self, function: Any) -> None:
        with pytest.raises(ValueError, match=r"F must have shape \(n_samples, n_features\)"):
            function(torch.zeros(4))
        with pytest.raises(ValueError, match=r"F must have shape \(n_samples, n_features\)"):
            function(torch.zeros(4, 0))
        with pytest.raises(ValueError, match="F must be finite"):
            function(torch.tensor([[0.0], [math.inf]]))
        with pytest.raises(TypeError, match="F must hold real numbers"):
            function(torch.tensor([[True], [False]]))

    def test_the_fisher_ratio_needs_three_samples(self) -> None:
        with pytest.raises(ValueError, match="at least 3 samples"):
            fisher_discriminant_ratio(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1]))
        with pytest.raises(ValueError, match="at least 3 samples"):
            separation_measures(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1]))

    def test_the_effective_rank_needs_two_samples(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            effective_rank(torch.tensor([[0.0, 1.0]]))

    def test_results_carry_no_gradient(self) -> None:
        features = normal(6, 2, seed=23).requires_grad_()
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        assert not pairwise_distances(features, labels).between.requires_grad
        assert not linear_feature_kernel(features).requires_grad
