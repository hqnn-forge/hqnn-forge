"""
tests/test_calibration.py
=========================
Calibration metrics, temperature and Platt scaling, the history's
temperature and the benchmark's calibration columns (#319).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hqnn_forge.evaluation import (
    MulticlassTemperatureScaler,
    PlattScaler,
    TemperatureScaler,
    brier_score,
    classwise_ece,
    expected_calibration_error,
    multiclass_brier_score,
    reliability_curve,
    top_label_ece,
)
from hqnn_forge.training import train_model


def _synthetic(
    n: int, seed: int, a: float = 1.0, b: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and labels drawn from σ(a·z + b): calibrated after that map."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, generator=g, dtype=torch.float64) * 3
    y = (torch.rand(n, generator=g, dtype=torch.float64) < torch.sigmoid(a * z + b)).double()
    return z, y


class TestMetrics:
    def test_against_scikit_learn(self) -> None:
        metrics = pytest.importorskip("sklearn.metrics")
        from sklearn.calibration import calibration_curve

        rng = np.random.default_rng(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(200):
                n = int(rng.integers(5, 200))
                p = rng.beta(0.5, 2, n)
                y = (rng.random(n) < p).astype(int)
                if y.min() == y.max():
                    continue
                assert brier_score(y, p) == pytest.approx(
                    metrics.brier_score_loss(y, p), abs=1e-12
                )
                confidence, frequency, _ = reliability_curve(y, p, 10)
                prob_true, prob_pred = calibration_curve(y, p, n_bins=10)
                np.testing.assert_allclose(frequency.numpy(), prob_true, atol=1e-12)
                np.testing.assert_allclose(confidence.numpy(), prob_pred, atol=1e-12)

    def test_known_values(self) -> None:
        # A constant 0.5 on a 20 % base rate: one bin, gap 0.3; Brier 0.25.
        y = torch.tensor([1.0] * 2 + [0.0] * 8)
        p = torch.full((10,), 0.5)
        assert expected_calibration_error(y, p) == pytest.approx(0.3)
        assert brier_score(y, p) == pytest.approx(0.25)
        assert brier_score(y, y) == 0.0 and expected_calibration_error(y, y) == 0.0

    def test_ece_weights_bins_by_their_counts(self) -> None:
        # Bin at 0.1: 8 samples, 2 positive (gap 0.15); bin at 0.9: 2 samples,
        # both positive (gap 0.1).  Weighted 0.8·0.15 + 0.2·0.1 = 0.14; an
        # unweighted mean would give 0.125.
        p = torch.tensor([0.1] * 8 + [0.9] * 2)
        y = torch.tensor([1.0, 1.0] + [0.0] * 6 + [1.0, 1.0])
        assert expected_calibration_error(y, p) == pytest.approx(0.14)

    def test_a_calibrated_predictor_has_small_ece(self) -> None:
        z, y = _synthetic(40_000, 0)
        for strategy in ("uniform", "quantile"):
            assert expected_calibration_error(y, torch.sigmoid(z), 10, strategy) < 0.01  # type: ignore[arg-type]

    def test_quantile_bins_hold_equal_counts(self) -> None:
        p = torch.rand(1000, generator=torch.Generator().manual_seed(1), dtype=torch.float64) ** 4
        y = (
            torch.rand(1000, generator=torch.Generator().manual_seed(2), dtype=torch.float64) < p
        ).double()
        _, _, counts = reliability_curve(y, p, 10, "quantile")
        assert counts.tolist() == [100.0] * 10
        _, _, uniform = reliability_curve(y, p, 10, "uniform")
        assert uniform[0] > 400  # skewed probabilities crowd the lowest uniform bin

    @pytest.mark.parametrize(
        "y, p, match",
        [
            ([0, 1], [0.5], "differ in length"),
            ([0, 2], [0.5, 0.5], "0/1 labels"),
            ([0, 1], [0.5, float("nan")], r"probabilities in \[0, 1\]"),
            ([0, 1], [0.5, 1.2], r"probabilities in \[0, 1\]"),
            ([], [], "empty"),
        ],
    )
    def test_bad_input(self, y: list[float], p: list[float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            brier_score(y, p)

    def test_bool_labels_score_as_0_and_1(self) -> None:
        assert brier_score([False, True], [0.25, 0.5]) == pytest.approx(0.15625)

    @pytest.mark.may_skip  # no CUDA device on the CI runners
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
    def test_cuda_inputs(self) -> None:
        z, y = _synthetic(500, 9, a=0.5)
        zc, yc = z.cuda(), y.cuda()
        p = torch.sigmoid(z)
        assert brier_score(yc, p.cuda()) == pytest.approx(brier_score(y, p))
        assert expected_calibration_error(yc, p.cuda(), 10, "quantile") == pytest.approx(
            expected_calibration_error(y, p, 10, "quantile")
        )
        assert TemperatureScaler.fit(zc, yc).temperature == pytest.approx(
            TemperatureScaler.fit(z, y).temperature
        )
        assert PlattScaler.fit(zc, yc).a == pytest.approx(PlattScaler.fit(z, y).a)

    def test_bad_bins(self) -> None:
        with pytest.raises(ValueError, match="strategy must be"):
            reliability_curve([0, 1], [0.1, 0.9], strategy="log")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_bins must be"):
            expected_calibration_error([0, 1], [0.1, 0.9], n_bins=0)
        with pytest.raises(ValueError, match="n_bins must be"):
            reliability_curve([0, 1], [0.1, 0.9], n_bins=0)
        with pytest.raises(ValueError, match="strategy must be"):
            expected_calibration_error([0, 1], [0.1, 0.9], strategy="log")  # type: ignore[arg-type]


def _nll(z: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.binary_cross_entropy_with_logits(z, y))


class TestTemperatureScaling:
    def test_fits_the_nll_optimum(self) -> None:
        z, y = _synthetic(5000, 3, a=0.4)  # the logits are over-confident by 2.5x
        t = TemperatureScaler.fit(z, y).temperature
        assert t > 1
        # A minimum of the NLL in T: the derivative vanishes and both
        # neighbours are worse.
        tt = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        (grad,) = torch.autograd.grad(F.binary_cross_entropy_with_logits(z / tt, y), tt)
        assert abs(float(grad)) < 1e-6
        assert _nll(z / t, y) <= min(_nll(z / (1.05 * t), y), _nll(z / (t / 1.05), y))

    def test_keeps_the_ranking_and_never_raises_the_nll(self) -> None:
        z, y = _synthetic(3000, 4, a=0.5)
        scaler = TemperatureScaler.fit(z, y)
        calibrated = scaler(z)
        assert torch.equal(torch.argsort(calibrated), torch.argsort(torch.sigmoid(z)))
        assert _nll(z / scaler.temperature, y) <= _nll(z, y)
        assert expected_calibration_error(y, calibrated) < expected_calibration_error(
            y, torch.sigmoid(z)
        )

    def test_needs_both_classes(self) -> None:
        with pytest.raises(ValueError, match="both classes"):
            TemperatureScaler.fit(torch.randn(5), torch.zeros(5))

    @pytest.mark.parametrize("scaler", [TemperatureScaler, PlattScaler])
    @pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
    def test_rejects_non_finite_logits(
        self, scaler: type[TemperatureScaler | PlattScaler], bad: float
    ) -> None:
        with pytest.raises(ValueError, match="finite"):
            scaler.fit([-1.0, 0.5, bad, 1.0], [0, 1, 0, 1])

    def test_no_finite_temperature_for_logits_that_separate_the_classes(self) -> None:
        # Every margin positive (0 counts as not wrong): the NLL falls as T -> 0.
        for z in ([-2.0, -1.0, 1.0, 2.0], [-2.0, 0.0, 1.0, 2.0]):
            with pytest.raises(ValueError, match="separate the classes"):
                TemperatureScaler.fit(z, [0, 0, 1, 1])

    def test_no_finite_temperature_for_logits_no_better_than_chance(self) -> None:
        # Sum of signed margins <= 0: the NLL falls as T -> infinity.
        for z in ([2.0, 1.0, -1.0, -2.0], [1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]):
            with pytest.raises(ValueError, match="no better than chance"):
                TemperatureScaler.fit(z, [0, 0, 1, 1])

    def test_one_wrong_margin_is_enough_for_a_finite_temperature(self) -> None:
        z = torch.tensor([-2.0, 0.5, -0.5, 2.0], dtype=torch.float64)
        y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
        t = TemperatureScaler.fit(z, y).temperature
        tt = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        (grad,) = torch.autograd.grad(F.binary_cross_entropy_with_logits(z / tt, y), tt)
        assert 0 < t < math.inf and abs(float(grad)) < 1e-6


class TestPlattScaling:
    def test_recovers_a_and_b_at_the_nll_optimum(self) -> None:
        z, y = _synthetic(40_000, 5, a=0.4, b=-1.0)
        scaler = PlattScaler.fit(z, y)
        assert scaler.a == pytest.approx(0.4, abs=0.03) and scaler.b == pytest.approx(
            -1.0, abs=0.05
        )
        ab = torch.tensor([scaler.a, scaler.b], dtype=torch.float64, requires_grad=True)
        (grad,) = torch.autograd.grad(F.binary_cross_entropy_with_logits(ab[0] * z + ab[1], y), ab)
        assert grad.abs().max() < 1e-6

    @pytest.mark.parametrize(
        "z",
        [
            [-2.0, -1.0, 1.0, 2.0],  # separated in order
            [2.0, 1.0, -1.0, -2.0],  # separated in reverse: a -> -infinity
            [-2.0, 0.0, 0.0, 2.0],  # quasi-complete: tied only at the boundary
        ],
    )
    def test_no_finite_fit_when_a_threshold_separates_the_classes(self, z: list[float]) -> None:
        with pytest.raises(ValueError, match="separates the classes"):
            PlattScaler.fit(z, [0, 0, 1, 1])

    def test_reverse_ranked_overlapping_logits_fit_a_negative_a(self) -> None:
        z, y = _synthetic(4000, 8, a=-0.6, b=0.2)
        assert PlattScaler.fit(z, y).a == pytest.approx(-0.6, abs=0.08)

    def test_monotone_for_positive_a(self) -> None:
        z, y = _synthetic(2000, 6, a=0.7, b=0.5)
        scaler = PlattScaler.fit(z, y)
        assert scaler.a > 0
        assert torch.equal(torch.argsort(scaler(z)), torch.argsort(z))


class TestTrainingHistory:
    def _data(self) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(200, 3, generator=g)
        return x, (x[:, 0] + 0.5 * torch.randn(200, generator=g) > 0).float()

    def test_records_the_validation_temperature_of_the_returned_weights(self) -> None:
        x, y = self._data()
        torch.manual_seed(0)
        model = nn.Linear(3, 1)
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.Adam(model.parameters(), lr=0.1),
            x[:150],
            y[:150],
            x[150:],
            y[150:],
            max_epochs=20,
        )
        with torch.no_grad():
            logits = model(x[150:]).squeeze(-1)
        assert history.temperature is not None
        assert history.temperature == pytest.approx(
            TemperatureScaler.fit(logits, y[150:]).temperature
        )

    def test_none_without_validation(self) -> None:
        x, y = self._data()
        model = nn.Linear(3, 1)
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            x,
            y,
            max_epochs=2,
        )
        assert history.temperature is None

    def test_none_for_soft_validation_targets(self) -> None:
        x, y = self._data()
        soft = y * 0.9 + 0.05
        model = nn.Linear(3, 1)
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            x[:150],
            soft[:150],
            x[150:],
            soft[150:],
            monitor="val_loss",
            max_epochs=2,
        )
        assert history.n_epochs == 2 and history.temperature is None

    def test_none_when_the_validation_logits_separate_the_classes(self) -> None:
        x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
        y = torch.tensor([0.0, 0.0, 1.0, 1.0])
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.zero_()
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.0),
            x,
            y,
            x,
            y,
            max_epochs=1,
        )
        assert history.n_epochs == 1 and history.temperature is None


class TestBenchmark:
    def test_fit_and_score_reports_the_test_probabilities_calibration(self) -> None:
        from hqnn_forge import benchmark
        from hqnn_forge.models import HybridBinaryClassifier
        from hqnn_forge.utils import FocalLoss

        rng = np.random.default_rng(0)
        X = rng.standard_normal((90, 3))
        y = (X[:, 0] + 0.5 * rng.standard_normal(90) > 0).astype(np.int64)
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            3, 2, 1, device_name="default.qubit", diff_method="backprop"
        )
        fit = benchmark._fit_and_score(
            model,
            FocalLoss,
            X[:50],
            y[:50],
            X[50:70],
            y[50:70],
            X[70:],
            y[70:],
            lr=0.05,
            max_epochs=3,
            batch_size=16,
            patience=None,
            batch_seed=0,
        )
        prob = model.predict_proba(torch.tensor(X[70:], dtype=torch.float32))
        assert fit.brier == pytest.approx(brier_score(y[70:], prob))
        assert fit.ece == pytest.approx(
            expected_calibration_error(y[70:], prob, benchmark.ECE_BINS, "quantile")
        )

    def test_a_diverged_fold_scores_nan_calibration_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hqnn_forge import benchmark
        from hqnn_forge.models import HybridBinaryClassifier
        from hqnn_forge.utils import FocalLoss

        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 3))
        y = (X[:, 0] > 0).astype(np.int64)
        model = HybridBinaryClassifier(
            3, 2, 1, device_name="default.qubit", diff_method="backprop"
        )
        # What a model with NaN weights predicts.
        monkeypatch.setattr(model, "predict_proba", lambda x: torch.full((len(x),), math.nan))
        fit = benchmark._fit_and_score(
            model,
            FocalLoss,
            X[:30],
            y[:30],
            X[30:45],
            y[30:45],
            X[45:],
            y[45:],
            lr=0.05,
            max_epochs=1,
            batch_size=16,
            patience=None,
            batch_seed=0,
        )
        assert fit.mcc == 0.0
        assert math.isnan(fit.brier) and math.isnan(fit.ece)

    def test_records_average_the_folds(self) -> None:
        from hqnn_forge import benchmark
        from hqnn_forge.models import HybridBinaryClassifier

        rng = np.random.default_rng(1)
        X = rng.standard_normal((120, 3))
        y = (X[:, 0] + 0.5 * rng.standard_normal(120) > 0.5).astype(int)
        result = benchmark.run_benchmark(
            {"toy": (X, y)},
            lambda d: HybridBinaryClassifier(
                d, 2, 1, device_name="default.qubit", diff_method="backprop"
            ),
            n_splits=3,
            max_epochs=2,
            batch_size=32,
            smote_kwargs={"k_neighbors": 3},
        )
        for record in result.records:
            folds = [f for f in result.folds if f.model == record["model"]]
            assert len(folds) == 3
            assert all(0 <= f.brier <= 1 and 0 <= f.ece <= 1 for f in folds)
            assert record["brier_mean"] == pytest.approx(np.mean([f.brier for f in folds]))
            assert record["ece_mean"] == pytest.approx(np.mean([f.ece for f in folds]))


# ---------------------------------------------------------------------------
# Multiclass (#360)
# ---------------------------------------------------------------------------


def _multiclass(n: int = 300, k: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * 2
    prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in prob])
    return y, prob


class TestMulticlassBrier:
    def test_matches_scikit_learn(self) -> None:
        from sklearn.metrics import brier_score_loss

        # scikit-learn takes multiclass input only from 1.7 on, above the 1.6
        # floor, so compare with the sum of its one-vs-rest binary losses,
        # which is the same quantity on every supported version.
        y, prob = _multiclass()
        one_vs_rest = sum(brier_score_loss(y == k, prob[:, k]) for k in range(prob.shape[1]))
        assert multiclass_brier_score(y, prob) == pytest.approx(one_vs_rest)

    @pytest.mark.may_skip  # scikit-learn below 1.7 on the lowest-floors job
    def test_matches_scikit_learn_on_multiclass_input(self) -> None:
        import sklearn
        from sklearn.metrics import brier_score_loss

        if tuple(int(part) for part in sklearn.__version__.split(".")[:2]) < (1, 7):
            pytest.skip("brier_score_loss takes (n, K) input from scikit-learn 1.7 on")
        y, prob = _multiclass()
        assert multiclass_brier_score(y, prob) == pytest.approx(brier_score_loss(y, prob))
        # Two classes: scikit-learn halves the score unless told not to.
        y, prob = _multiclass(k=2)
        ours = multiclass_brier_score(y, prob)
        assert ours == pytest.approx(brier_score_loss(y, prob, scale_by_half=False))
        assert ours == pytest.approx(2 * brier_score_loss(y, prob))

    def test_ignores_the_autograd_graph(self) -> None:
        y, prob = _multiclass(n=20)
        logits = torch.from_numpy(np.log(prob)).requires_grad_()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            score = multiclass_brier_score(y, torch.softmax(logits, dim=1))
        assert score == pytest.approx(multiclass_brier_score(y, prob))

    def test_known_values(self) -> None:
        k = 4
        y = np.array([0, 1, 2, 3, 1])
        assert multiclass_brier_score(y, np.eye(k)[y]) == 0.0
        assert multiclass_brier_score(y, np.full((5, k), 1 / k)) == pytest.approx(1 - 1 / k)
        # Certain and wrong: two squared errors of 1.
        assert multiclass_brier_score(y, np.eye(k)[(y + 1) % k]) == 2.0

    def test_two_classes_count_both_columns(self) -> None:
        y, prob = _multiclass(k=2)
        assert multiclass_brier_score(y, prob) == pytest.approx(2 * brier_score(y, prob[:, 1]))


class TestMulticlassECE:
    def test_top_label_by_hand(self) -> None:
        # Confidence 0.9 right once and wrong once (frequency 0.5), and 0.6
        # right twice (frequency 1): |0.5 − 0.9| · ½ + |1 − 0.6| · ½ = 0.4.
        prob = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        y = np.array([0, 1, 1, 2])
        assert top_label_ece(y, prob) == pytest.approx(0.4)

    def test_top_label_with_quantile_bins(self) -> None:
        # The same four samples in two bins.  Equal-count bins split 0.6 from
        # 0.9 and give 0.4 again; uniform bins put all four above 0.5, where
        # the mean confidence 0.75 equals the frequency 3/4.
        prob = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        y = np.array([0, 1, 1, 2])
        assert top_label_ece(y, prob, n_bins=2, strategy="quantile") == pytest.approx(0.4)
        assert top_label_ece(y, prob, n_bins=2) == pytest.approx(0.0)

    def test_top_label_is_zero_when_confidence_is_frequency(self) -> None:
        # Two samples at confidence 0.5 in two classes, one right and one wrong.
        prob = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
        assert top_label_ece(np.array([0, 1]), prob) == pytest.approx(0.0)

    def test_classwise_is_the_mean_of_one_vs_rest_eces(self) -> None:
        y, prob = _multiclass()
        per_class = [
            expected_calibration_error((y == k).astype(int), prob[:, k]) for k in range(4)
        ]
        assert classwise_ece(y, prob) == pytest.approx(np.mean(per_class))
        quantile = [
            expected_calibration_error((y == k).astype(int), prob[:, k], strategy="quantile")
            for k in range(4)
        ]
        assert classwise_ece(y, prob, strategy="quantile") == pytest.approx(np.mean(quantile))

    def test_classwise_by_hand(self) -> None:
        # Class 0: p = 0.8, 0.8 with labels 1, 0 → |0.5 − 0.8| = 0.3; p = 0.2,
        # 0.2 with labels 0, 0 → 0.2; ECE 0.25.  Class 1 mirrors it: 0.25.
        prob = np.array([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]])
        y = np.array([0, 1, 1, 1])
        assert classwise_ece(y, prob) == pytest.approx(0.25)

    def test_calibrated_sampling_gives_a_small_ece(self) -> None:
        # Labels drawn from the probabilities themselves: calibrated by
        # construction, so the ECEs are sampling noise only.
        y, prob = _multiclass(n=20000)
        assert top_label_ece(y, prob) < 0.02
        assert classwise_ece(y, prob) < 0.02


class TestMulticlassTemperature:
    def test_recovers_an_over_confidence_factor(self) -> None:
        # Labels drawn from softmax(z), logits reported as 3z: T ≈ 3 undoes it.
        rng = np.random.default_rng(1)
        z = rng.normal(size=(20000, 3))
        prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        y = np.array([rng.choice(3, p=row) for row in prob])
        scaler = MulticlassTemperatureScaler.fit(3 * z, y)
        assert scaler.temperature == pytest.approx(3.0, rel=0.05)

    def test_keeps_the_argmax_and_never_raises_the_nll(self) -> None:
        y, prob = _multiclass(seed=2)
        logits = torch.from_numpy(np.log(prob) * 0.4)  # under-confident
        scaler = MulticlassTemperatureScaler.fit(logits, y)
        calibrated = scaler(logits)
        assert torch.equal(calibrated.argmax(1), logits.argmax(1))
        target = torch.from_numpy(y)
        before = torch.nn.functional.cross_entropy(logits, target)
        after = torch.nn.functional.nll_loss(calibrated.log(), target)
        assert after <= before + 1e-12
        assert scaler.temperature < 1

    def test_at_the_nll_optimum(self) -> None:
        y, prob = _multiclass(seed=3)
        logits = torch.from_numpy(np.log(prob) * 2.5)
        scaler = MulticlassTemperatureScaler.fit(logits, y)
        t = torch.tensor(scaler.temperature, dtype=torch.float64, requires_grad=True)
        torch.nn.functional.cross_entropy(logits / t, torch.from_numpy(y)).backward()
        assert t.grad is not None and abs(t.grad.item()) < 1e-6

    def test_needs_two_classes(self) -> None:
        with pytest.raises(ValueError, match="at least two classes"):
            MulticlassTemperatureScaler.fit(np.zeros((4, 3)), np.array([1, 1, 1, 1]))

    def test_call_is_the_softmax_of_the_scaled_logits(self) -> None:
        z = np.random.default_rng(4).normal(size=(50, 4))
        calibrated = MulticlassTemperatureScaler(2.5)(z)
        expected = np.exp(z / 2.5) / np.exp(z / 2.5).sum(1, keepdims=True)
        assert calibrated.dtype == torch.float64
        np.testing.assert_allclose(calibrated.numpy(), expected, rtol=1e-12)
        np.testing.assert_allclose(calibrated.sum(1).numpy(), 1.0, rtol=1e-12)

    def test_call_rejects_a_single_row_without_its_batch_axis(self) -> None:
        with pytest.raises(ValueError, match=r"logits must have shape \(n, K\)"):
            MulticlassTemperatureScaler(2.0)(np.array([1.0, 2.0, 3.0]))

    def test_no_finite_temperature_when_every_true_class_has_the_top_logit(self) -> None:
        # A tie for the top (the last row) counts as not wrong: the NLL falls as T -> 0.
        z = np.random.default_rng(5).normal(size=(200, 3))
        z[-1] = [1.0, 1.0, 0.0]
        with pytest.raises(ValueError, match="top logit"):
            MulticlassTemperatureScaler.fit(z, z.argmax(1))

    def test_no_finite_temperature_for_logits_no_better_than_chance(self) -> None:
        z = np.random.default_rng(5).normal(size=(200, 3))
        with pytest.raises(ValueError, match="no better than chance"):
            MulticlassTemperatureScaler.fit(z, (z.argmax(1) + 1) % 3)
        # Exactly chance: every row constant, so the slope at 1/T = 0 is 0.
        with pytest.raises(ValueError, match="no better than chance"):
            MulticlassTemperatureScaler.fit(np.ones((4, 3)), np.array([0, 1, 2, 0]))

    def test_one_wrong_sample_is_enough_for_a_finite_temperature(self) -> None:
        # Both conditions hold by the smallest margin: the fit is interior, and
        # the slope of the cross-entropy in T vanishes there.
        z = torch.from_numpy(np.random.default_rng(5).normal(size=(200, 3)))
        y = z.argmax(1)
        y[0] = (y[0] + 1) % 3
        scaler = MulticlassTemperatureScaler.fit(z, y)
        assert 0 < scaler.temperature < 1
        t = torch.tensor(scaler.temperature, dtype=torch.float64, requires_grad=True)
        F.cross_entropy(z / t, y).backward()
        assert t.grad is not None and abs(t.grad.item()) < 1e-6

    @pytest.mark.parametrize(
        ("logits", "y", "match"),
        [
            (np.array([0.5, -0.5]), np.array([0, 1]), r"logits must have shape \(n, K\)"),
            (np.array([[0.5], [-0.5]]), np.array([0, 0]), r"logits must have shape \(n, K\)"),
            (np.array([[np.inf, 0.0], [0.0, 1.0]]), np.array([0, 1]), "logits must be finite"),
            (np.array([[np.nan, 0.0], [0.0, 1.0]]), np.array([0, 1]), "logits must be finite"),
            (np.array([[0.5, -0.5]]), np.array([0, 1]), "y_true and logits differ in length"),
            (np.empty((0, 3)), np.array([]), "y_true is empty"),
            (np.array([[0.5, -0.5], [0.0, 1.0]]), np.array([0, 2]), "class labels 0 … 1"),
            (np.array([[0.5, -0.5], [0.0, 1.0]]), np.array([0.0, 0.5]), "integer class labels"),
        ],
    )
    def test_fit_names_the_logits_in_its_errors(
        self, logits: np.ndarray, y: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            MulticlassTemperatureScaler.fit(logits, y)


class TestMulticlassDevices:
    @pytest.mark.may_skip  # no CUDA device on the CI runners
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
    def test_cuda_inputs(self) -> None:
        labels, prob = _multiclass(seed=6)
        y, p = torch.from_numpy(labels), torch.from_numpy(prob)
        for fn in (multiclass_brier_score, top_label_ece, classwise_ece):
            assert fn(y.cuda(), p.cuda()) == pytest.approx(fn(y, p))
        logits = p.log() * 2.5
        on_cpu = MulticlassTemperatureScaler.fit(logits, y).temperature
        assert MulticlassTemperatureScaler.fit(logits.cuda(), y.cuda()).temperature == (
            pytest.approx(on_cpu)
        )
        assert MulticlassTemperatureScaler.fit(logits.cuda(), labels).temperature == (
            pytest.approx(on_cpu)
        )


class TestMulticlassValidation:
    @pytest.mark.parametrize(
        ("y", "prob", "match"),
        [
            (np.array([0, 1]), np.array([0.2, 0.8]), r"shape \(n, K\)"),
            (np.array([0, 1]), np.array([[0.2, 0.8]]), "differ in length"),
            (np.array([0, 3]), np.array([[0.2, 0.8], [0.5, 0.5]]), "class labels 0"),
            (np.array([0.0, 0.5]), np.array([[0.2, 0.8], [0.5, 0.5]]), "integer class labels"),
            (np.array([0, 1]), np.array([[0.2, 0.7], [0.5, 0.5]]), "sum to 1"),
            (np.array([0, 1]), np.array([[-0.2, 1.2], [0.5, 0.5]]), r"in \[0, 1\]"),
        ],
    )
    @pytest.mark.parametrize("fn", [multiclass_brier_score, top_label_ece, classwise_ece])
    def test_bad_input(self, fn: Any, y: np.ndarray, prob: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            fn(y, prob)

    @pytest.mark.parametrize("fn", [top_label_ece, classwise_ece])
    def test_bad_bins(self, fn: Any) -> None:
        y, prob = _multiclass(k=3)
        with pytest.raises(ValueError, match="n_bins must be"):
            fn(y, prob, n_bins=0)
        with pytest.raises(ValueError, match="strategy must be"):
            fn(y, prob, strategy="log")

    @pytest.mark.parametrize("fn", [top_label_ece, classwise_ece])
    def test_does_not_repeat_binary_validation(
        self, fn: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        y, prob = _multiclass(k=3)

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("_pair was unexpectedly invoked")

        monkeypatch.setattr("hqnn_forge.evaluation.calibration._pair", _boom)
        val = fn(y, prob)
        assert isinstance(val, float)
