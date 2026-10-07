"""
tests/test_sklearn_estimator.py
================================
hqnn_forge.sklearn.HybridClassifierEstimator against scikit-learn tooling.

Skipped where scikit-learn is not installed (optional dependency).

Everything is seeded (``random_state=0``), so the runs are deterministic.  The
accuracy bars (> 0.7 on a linearly separable task) test that the wrapper
trains the model, not how well a 2-qubit model optimises.  The tests that
assert learning use ``LEARN``, whose bars were measured to hold across seeds
(see below), rather than a seed that happens to work at ``FAST``.
"""

from __future__ import annotations

import functools
import inspect
import math
import pickle
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import estimator_checks_generator

import hqnn_forge.noise as noise_module
from hqnn_forge.evaluation.calibration import (
    PlattScaler,
    TemperatureScaler,
    expected_calibration_error,
)
from hqnn_forge.evaluation.thresholds import find_optimal_threshold
from hqnn_forge.sklearn import HybridClassifierEstimator
from hqnn_forge.training import train_model

FAST: dict[str, Any] = dict(
    n_qubits=2,
    n_layers=1,
    device_name="default.qubit",
    diff_method="backprop",
    max_epochs=15,
    batch_size=16,
    lr=0.05,
    loss="bce",
    random_state=0,
)


# The tests that assert learning, not only plumbing, need more than FAST's
# budget to hold across seeds, because the 2-qubit, 1-layer model sometimes
# stalls.  Over random_state 0-9 at FAST, the cross-validation check
# (mean MCC > 0.3) failed for 3 seeds on main and 4 here, the pipeline check
# (accuracy > 0.7) for 1 on main and 2 here, and the serial fit (> 0.7) for
# 2 here; seed 0 is among the failures here.  With LEARN, over random_state
# 0-19: serial accuracy 0.95-1.00, parallel 1.00, 3-fold mean MCC 0.20-0.93
# (19 of 20 above 0.41) and pipeline accuracy 0.74-1.00.  Every bar sits below
# the worst seed and well above chance.
LEARN = {**FAST, "n_layers": 2, "max_epochs": 40}


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


class TestParams:
    def test_get_set_params_round_trip(self) -> None:
        est = HybridClassifierEstimator(**FAST)
        params = est.get_params()
        assert params["n_qubits"] == 2 and params["model"] == "serial"
        est.set_params(n_layers=3, model="parallel")
        assert est.get_params()["n_layers"] == 3 and est.model == "parallel"
        cloned = clone(est)
        assert cloned.get_params() == est.get_params() and cloned is not est

    def test_init_stores_arguments_unchanged(self) -> None:
        est = HybridClassifierEstimator(threshold=0.3, patience=2)
        assert est.threshold == 0.3 and est.patience == 2
        assert not hasattr(est, "model_")


class TestFitPredict:
    @pytest.mark.parametrize("model", ["serial", "parallel"])
    def test_shapes_and_learning(self, data: tuple, model: Literal["serial", "parallel"]) -> None:
        X, y = data
        est = HybridClassifierEstimator(model=model, **LEARN).fit(X, y)
        proba = est.predict_proba(X)
        assert proba.shape == (80, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        pred = est.predict(X)
        assert pred.shape == (80,) and set(pred) <= {0, 1}
        assert est.score(X, y) > 0.7
        assert est.n_features_in_ == 3 and list(est.classes_) == [0, 1]
        assert est.history_.n_epochs == LEARN["max_epochs"]

    def test_string_labels_and_positive_class(self, data: tuple) -> None:
        X, y = data
        labels = np.where(y == 1, "fraud", "legit")
        est = HybridClassifierEstimator(**FAST).fit(X, labels)
        assert list(est.classes_) == ["fraud", "legit"]
        # classes_[1] ("legit") is the positive column
        assert est.predict_proba(X).shape == (80, 2)
        assert set(est.predict(X)) <= {"fraud", "legit"}
        assert est.score(X, labels) > 0.7

    def test_reproducible_with_random_state(self, data: tuple) -> None:
        X, y = data
        a = HybridClassifierEstimator(**FAST).fit(X, y).predict_proba(X)
        b = HybridClassifierEstimator(**FAST).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(a, b)

    def test_validation_split_sets_optimal_threshold(self, data: tuple) -> None:
        X, y = data
        est = HybridClassifierEstimator(
            **{**FAST, "validation_fraction": 0.25, "patience": 5}
        ).fit(X, y)
        assert est.history_.best_threshold is not None
        assert est.threshold_ is not None
        assert est.threshold_ == pytest.approx(est.history_.best_threshold)
        np.testing.assert_array_equal(
            est.predict(X),
            est.classes_[(est.predict_proba(X)[:, 1] >= est.threshold_).astype(int)],
        )

    def test_fixed_threshold_and_default_without_validation(self, data: tuple) -> None:
        X, y = data
        default = HybridClassifierEstimator(**FAST).fit(X, y)
        assert default.threshold_ == 0.5
        strict = clone(default).set_params(threshold=0.9).fit(X, y)
        assert strict.threshold_ == 0.9
        # Same seed, same weights: only the threshold differs
        np.testing.assert_array_equal(strict.predict_proba(X), default.predict_proba(X))
        assert strict.predict(X).sum() <= default.predict(X).sum()

    def test_numpy_scalar_threshold_accepted(self, data: tuple) -> None:
        # bool is rejected as a threshold, but a numpy float is a real number
        est = HybridClassifierEstimator(**{**FAST, "threshold": np.float32(0.3)}).fit(*data)
        assert est.threshold_ == pytest.approx(0.3, abs=1e-7)

    def test_patience_default_matches_train_model(self) -> None:
        # The wrapper's own default must not quietly disable the early stopping
        # that validation_fraction pays training samples for.
        assert (
            inspect.signature(HybridClassifierEstimator).parameters["patience"].default
            == inspect.signature(train_model).parameters["patience"].default
        )

    def test_focal_loss_default(self, data: tuple) -> None:
        X, y = data
        est = HybridClassifierEstimator(**{**FAST, "loss": "focal"}).fit(X, y)
        assert est.history_.train_loss[-1] < est.history_.train_loss[0]


class TestSklearnTooling:
    @pytest.mark.slow  # three full fits
    def test_cross_val_score(self, data: tuple) -> None:
        X, y = data
        scores = cross_val_score(
            HybridClassifierEstimator(**LEARN), X, y, cv=3, scoring="matthews_corrcoef"
        )
        assert scores.shape == (3,) and scores.mean() > 0.15

    def test_pipeline(self, data: tuple) -> None:
        X, y = data
        pipe = make_pipeline(StandardScaler(), HybridClassifierEstimator(**LEARN)).fit(
            X * 50 + 7, y
        )
        assert pipe.score(X * 50 + 7, y) > 0.7

    def test_grid_search(self, data: tuple) -> None:
        X, y = data
        search = GridSearchCV(
            HybridClassifierEstimator(**{**FAST, "max_epochs": 3}),
            {"n_layers": [1, 2]},
            cv=2,
            scoring="accuracy",
        ).fit(X, y)
        assert search.best_params_["n_layers"] in (1, 2)
        assert search.best_estimator_.model_.n_layers == search.best_params_["n_layers"]


class TestErrors:
    def test_not_fitted(self, data: tuple) -> None:
        X, _ = data
        with pytest.raises(NotFittedError):
            HybridClassifierEstimator(**FAST).predict(X)

    def test_feature_count_mismatch(self, data: tuple) -> None:
        X, y = data
        est = HybridClassifierEstimator(**{**FAST, "max_epochs": 1}).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            est.predict(X[:, :2])

    def test_one_class_rejected(self, data: tuple) -> None:
        # More than two classes are supported since #309; one is not.
        X, _ = data
        with pytest.raises(ValueError, match="needs at least two classes; got 1"):
            HybridClassifierEstimator(**FAST).fit(X, np.zeros(80, dtype=int))

    @pytest.mark.parametrize(
        "params, match",
        [
            (dict(model="deep"), "model must be 'serial' or 'parallel'"),
            (dict(loss="hinge"), "loss must be 'focal' or 'bce'"),
            (dict(validation_fraction=1.0), r"validation_fraction must lie in \[0, 1\)"),
            (dict(validation_fraction=0.001), "leaves no training or no validation samples"),
            (dict(threshold="best"), "threshold must be 'optimal' or a real number"),
            (dict(threshold=True), "threshold must be 'optimal' or a real number"),
            (dict(threshold=1.5), r"threshold must lie in \[0, 1\]"),
            (dict(threshold=-0.1), r"threshold must lie in \[0, 1\]"),
        ],
    )
    def test_bad_parameters_fail_in_fit(self, data: tuple, params: dict, match: str) -> None:
        X, y = data
        with pytest.raises(ValueError, match=match):
            HybridClassifierEstimator(**{**FAST, **params}).fit(X, y)

    def test_failed_refit_keeps_the_previous_fit(self, data: tuple) -> None:
        # The refit fails inside the validation split, after the new model is
        # built: the estimator must not be left holding that untrained model.
        X, y = data
        est = HybridClassifierEstimator(**FAST).fit(X, y)
        trained, history, predictions = est.model_, est.history_, est.predict(X)
        with pytest.raises(ValueError, match="leaves no training or no validation samples"):
            est.set_params(validation_fraction=0.001).fit(X, y)
        assert est.model_ is trained and est.history_ is history
        np.testing.assert_array_equal(est.predict(X), predictions)

    def test_nan_input_rejected(self, data: tuple) -> None:
        X, y = data
        X = X.copy()
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            HybridClassifierEstimator(**FAST).fit(X, y)


class TestPickle:
    """A fitted model holds a QNode around a local function; pickle rebuilds it."""

    def test_fitted_pipeline_round_trips(self, data: tuple) -> None:
        X, y = data
        pipe = make_pipeline(StandardScaler(), HybridClassifierEstimator(**FAST)).fit(X, y)
        loaded = pickle.loads(pickle.dumps(pipe))
        np.testing.assert_array_equal(loaded.predict_proba(X), pipe.predict_proba(X))
        est = loaded[-1]
        assert est.threshold_ == pipe[-1].threshold_
        assert not est.model_.training
        assert type(est.model_) is type(pipe[-1].model_)

    @pytest.mark.parametrize("model", ["serial", "parallel"])
    def test_both_models_and_the_original_is_untouched(self, data: tuple, model: str) -> None:
        X, y = data
        est = HybridClassifierEstimator(**{**FAST, "model": model}).fit(X, y)
        trained = est.model_
        payload = pickle.dumps(est)
        assert est.model_ is trained  # pickling must not strip the live estimator
        loaded = pickle.loads(payload)
        np.testing.assert_array_equal(loaded.predict(X), est.predict(X))

    def test_unpickling_leaves_the_global_rng_alone(self, data: tuple) -> None:
        X, y = data
        payload = pickle.dumps(HybridClassifierEstimator(**FAST).fit(X, y))
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        pickle.loads(payload)
        torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)

    def test_unfitted_estimator_round_trips(self) -> None:
        loaded = pickle.loads(pickle.dumps(HybridClassifierEstimator(**FAST)))
        assert loaded.get_params() == HybridClassifierEstimator(**FAST).get_params()
        assert not hasattr(loaded, "model_")


# ---------------------------------------------------------------------------
# scikit-learn's own conformance checks
# ---------------------------------------------------------------------------

#: Checks this estimator is expected to fail, each with the reason.  Empty:
#: every check that runs passes.
EXPECTED_FAILED_CHECKS: dict[str, str] = {}

#: Checks that pass or fail by float32 rounding, depending on the CPU and on an
#: unseeded permutation the check draws, so a strict xfail cannot express them.
#: check_methods_sample_order_invariance compares predict_proba on a permuted
#: batch at rtol=1e-7, below float32 resolution (eps ~1.2e-7): the model runs
#: in float32, and on some CI runners a permuted batch moves an output by one
#: ulp.  test_sample_order_invariance_at_float32 checks the same property at
#: the model's own precision.
FLOAT32_TOLERANCE_CHECKS: dict[str, str] = {
    "check_methods_sample_order_invariance": (
        "rtol=1e-7 is below float32 resolution; one-ulp differences on some CPUs"
    ),
}

#: Checks scikit-learn skips itself when an optional package is absent, which
#: this project does not install.  They carry ``may_skip`` so that
#: HQNN_FORGE_FAIL_ON_SKIP=1 in CI does not turn the skip into a failure.
MAY_SKIP_CHECKS: dict[str, str] = {
    "check_classifier_data_not_an_array": "needs pandas",
    "check_array_api_input": "needs SCIPY_ARRAY_API=1 and array-api-strict",
}


def _conformance_estimator() -> HybridClassifierEstimator:
    # check_classifiers_train requires training accuracy > 0.83 on its own
    # toy problems, binary and (with the multiclass tag) 3-class blobs; 2
    # epochs fall short, 30 pass with margin at this seed.  Two qubits reach
    # only 0.77 on the 3-class problem (0.82 at 60 epochs); three reach 0.91.
    return HybridClassifierEstimator(
        n_qubits=3,
        n_layers=1,
        device_name="default.qubit",
        diff_method="backprop",
        max_epochs=30,
        batch_size=32,
        random_state=0,
    )


def _check_id(value: Any) -> str:
    if isinstance(value, HybridClassifierEstimator):
        return "HybridClassifierEstimator"
    if isinstance(value, functools.partial):
        kwargs = ",".join(f"{k}={v}" for k, v in value.keywords.items())
        return f"{value.func.__name__}({kwargs})" if kwargs else value.func.__name__
    return getattr(value, "__name__", repr(value))


def _check_name(check: Any) -> str:
    return getattr(check, "func", check).__name__


# Checks that fit to convergence several times over: 3-4 s per case (#323).
SLOW_CHECKS = frozenset({"check_classifiers_train"})


def _conformance_params() -> list[Any]:
    # The strict xfail marks are applied here rather than through
    # estimator_checks_generator(mark="xfail", xfail_strict=True): xfail_strict
    # only exists from scikit-learn 1.8, and the floor is 1.6.  Strict, so a
    # check that starts passing has to leave EXPECTED_FAILED_CHECKS.
    params = []
    for estimator, check in estimator_checks_generator(_conformance_estimator()):
        name = _check_name(check)
        reason = EXPECTED_FAILED_CHECKS.get(name)
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        if name in FLOAT32_TOLERANCE_CHECKS:
            # raises=AssertionError: only the tolerance comparison may fail;
            # an exception from fit or predict still fails the test.
            marks.append(
                pytest.mark.xfail(
                    reason=FLOAT32_TOLERANCE_CHECKS[name], strict=False, raises=AssertionError
                )
            )
        if name in MAY_SKIP_CHECKS:
            marks.append(pytest.mark.may_skip)
        if name in SLOW_CHECKS:
            marks.append(pytest.mark.slow)
        params.append(pytest.param(estimator, check, marks=marks))
    return params


@pytest.mark.parametrize(("estimator", "check"), _conformance_params(), ids=_check_id)
def test_scikit_learn_conformance(
    estimator: HybridClassifierEstimator, check: Callable[[HybridClassifierEstimator], None]
) -> None:
    check(estimator)


def test_sample_order_invariance_at_float32() -> None:
    # check_methods_sample_order_invariance at the model's float32 precision:
    # permuting the batch permutes the outputs, up to a few float32 ulps of a
    # probability (absolute, since 1 - p near p = 1 has no relative scale).
    # The check covers predict too, which the xfail would otherwise hide: labels
    # must match exactly, since threshold_ is the midpoint between two
    # validation probabilities, so an ulp-sized move does not cross it here.
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    y = (X[:, 0] > 1.5).astype(int)
    est = _conformance_estimator().fit(X, y)
    proba = est.predict_proba(X)
    labels = est.predict(X)
    eps = float(np.finfo(np.float32).eps)
    for seed in range(5):
        idx = np.random.RandomState(seed).permutation(X.shape[0])
        np.testing.assert_allclose(est.predict_proba(X[idx]), proba[idx], rtol=0, atol=4 * eps)
        np.testing.assert_array_equal(est.predict(X[idx]), labels[idx])


class TestNoiseAwareTraining:
    """#228: the model's training-noise options, exposed on the estimator."""

    NOISY = {**FAST, "max_epochs": 3, "noise_level": 0.1, "noise_position": "end"}

    def test_defaults_train_exactly_as_before(self, data: tuple) -> None:
        X, y = data
        explicit = HybridClassifierEstimator(
            **FAST,
            noise_level=0.0,
            noise_position="all",
            noise_method="density",
            noise_trajectories=1,
        ).fit(X, y)
        default = HybridClassifierEstimator(**FAST).fit(X, y)
        np.testing.assert_array_equal(explicit.predict_proba(X), default.predict_proba(X))
        assert default.model_.quantum_layer._training_noise_qnode is None

    @pytest.mark.parametrize("model", ["serial", "parallel"])
    def test_fit_runs_the_noisy_circuit_and_predict_does_not(
        self, data: tuple, model: Literal["serial", "parallel"], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        X, y = data
        calls = {"n": 0}
        original = noise_module.run_with_training_noise

        def spy(*args: object, **kwargs: object) -> torch.Tensor:
            calls["n"] += 1
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(noise_module, "run_with_training_noise", spy)
        est = HybridClassifierEstimator(model=model, **self.NOISY).fit(X, y)
        assert calls["n"] > 0
        layer = est.model_.quantum_layer
        assert (layer.noise_level, layer.noise_position) == (0.1, "end")
        calls["n"] = 0
        proba = est.predict_proba(X)
        assert calls["n"] == 0
        # predict is the noiseless model: the same weights evaluated clean.
        est.model_.eval()
        with torch.no_grad():
            clean = torch.sigmoid(est.model_(torch.from_numpy(X.astype(np.float32)))).squeeze(-1)
        np.testing.assert_allclose(proba[:, 1], clean.numpy(), rtol=1e-6)

    def test_noise_changes_what_is_learned(self, data: tuple) -> None:
        X, y = data
        clean = HybridClassifierEstimator(**{**FAST, "max_epochs": 3}).fit(X, y)
        noisy = HybridClassifierEstimator(**self.NOISY).fit(X, y)
        assert not np.allclose(clean.predict_proba(X), noisy.predict_proba(X))

    def test_trajectories_are_reproducible_with_random_state(self, data: tuple) -> None:
        X, y = data
        params = {**self.NOISY, "noise_method": "trajectories", "noise_trajectories": 2}
        a = HybridClassifierEstimator(**params).fit(X, y).predict_proba(X)
        b = HybridClassifierEstimator(**params).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(a, b)

    def test_params_round_trip_and_clone(self) -> None:
        est = HybridClassifierEstimator(noise_level=0.2, noise_method="trajectories")
        params = est.get_params()
        assert params["noise_level"] == 0.2 and params["noise_method"] == "trajectories"
        assert params["noise_position"] == "all" and params["noise_trajectories"] == 1
        cloned = clone(est.set_params(noise_trajectories=3))
        assert cloned.get_params() == est.get_params()

    def test_grid_search_tunes_the_noise_level(self, data: tuple) -> None:
        X, y = data
        search = GridSearchCV(
            HybridClassifierEstimator(**{**FAST, "max_epochs": 2}),
            {"noise_level": [0.0, 0.05]},
            cv=2,
            scoring="accuracy",
        ).fit(X, y)
        best = search.best_params_["noise_level"]
        assert search.best_estimator_.model_.quantum_layer.noise_level == best

    @pytest.mark.parametrize(
        "params, match",
        [
            ({"noise_level": 0.9}, r"noise_level must lie in \[0, 0.75\]"),
            ({"noise_position": "middle"}, "noise_position must be 'all' or 'end'"),
            ({"noise_method": "kraus"}, "noise_method must be"),
            ({"noise_trajectories": 2}, "needs noise_method='trajectories'"),
        ],
    )
    def test_bad_values_fail_in_fit_not_in_the_constructor(
        self, data: tuple, params: dict, match: str
    ) -> None:
        est = HybridClassifierEstimator(**{**FAST, **params})  # does not raise
        with pytest.raises(ValueError, match=match):
            est.fit(*data)


class TestCalibration:
    """#359: calibration fitted on the validation split, applied in predict_proba."""

    CAL = dict(FAST, loss="focal", validation_fraction=0.3, patience=5)

    @staticmethod
    def _noisy_data() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((240, 3))
        y = (X[:, 0] - X[:, 1] + 0.5 * rng.standard_normal(240) > 0).astype(int)
        return X, y

    def _validation(self, est: HybridClassifierEstimator, y: np.ndarray, seed: int) -> np.ndarray:
        # fit draws the split first from default_rng(random_state).
        _, va = est._stratified_holdout(y, 0.3, np.random.default_rng(seed))
        return va

    @staticmethod
    def _nll(y: np.ndarray, p: np.ndarray) -> float:
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # Any: the tests read calibrator_ and threshold_ as the type each one fitted.
    def _fit(self, calibration: str | None, seed: int = 0, **kw: Any) -> Any:
        X, y = self._noisy_data()
        params = {**self.CAL, "random_state": seed, "calibration": calibration, **kw}
        return HybridClassifierEstimator(**params).fit(X, y)

    def test_no_calibration_by_default(self) -> None:
        est = self._fit(None)
        assert est.calibration is None and est.calibrator_ is None

    @pytest.mark.parametrize(
        ("calibration", "cls"), [("temperature", TemperatureScaler), ("platt", PlattScaler)]
    )
    def test_fitted_calibrator(self, calibration: str, cls: type) -> None:
        assert isinstance(self._fit(calibration).calibrator_, cls)

    @pytest.mark.parametrize("calibration", ["temperature", "platt"])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_validation_log_loss_never_rises(self, calibration: str, seed: int) -> None:
        # Guaranteed: the fit minimises the validation NLL over a family that
        # contains the identity (T = 1, or a = 1 and b = 0).
        X, y = self._noisy_data()
        raw, cal = self._fit(None, seed), self._fit(calibration, seed)
        va = self._validation(cal, y, seed)
        p_raw, p_cal = raw.predict_proba(X[va])[:, 1], cal.predict_proba(X[va])[:, 1]
        assert self._nll(y[va], p_cal) <= self._nll(y[va], p_raw) + 1e-9

    @pytest.mark.parametrize("calibration", ["temperature", "platt"])
    def test_focal_loss_under_confidence_is_corrected(self, calibration: str) -> None:
        # Measured on this seed: validation ECE 0.194 uncalibrated, 0.101 with
        # temperature scaling (T = 0.37: the focal-loss model is
        # under-confident) and 0.067 with Platt scaling.
        X, y = self._noisy_data()
        raw, cal = self._fit(None, 2), self._fit(calibration, 2)
        va = self._validation(cal, y, 2)
        ece_raw = expected_calibration_error(y[va], raw.predict_proba(X[va])[:, 1])
        ece_cal = expected_calibration_error(y[va], cal.predict_proba(X[va])[:, 1])
        # The margin is well inside the smaller measured drop (0.093): ECE is
        # binned over 72 points, so a few of them changing bin moves it.
        assert ece_cal < ece_raw - 0.02, (ece_cal, ece_raw)

    @pytest.mark.parametrize("calibration", ["temperature", "platt"])
    def test_rankings_and_predictions_are_unchanged(self, calibration: str) -> None:
        X, _ = self._noisy_data()
        raw, cal = self._fit(None), self._fit(calibration)
        p_raw, p_cal = raw.predict_proba(X)[:, 1], cal.predict_proba(X)[:, 1]
        assert not np.allclose(p_raw, p_cal)
        np.testing.assert_array_equal(
            np.argsort(p_raw, kind="stable"), np.argsort(p_cal, kind="stable")
        )
        np.testing.assert_array_equal(raw.predict(X), cal.predict(X))

    @pytest.mark.parametrize("calibration", ["temperature", "platt"])
    def test_threshold_is_mapped_through_the_calibrator(self, calibration: str) -> None:
        raw, cal = self._fit(None), self._fit(calibration)
        assert 0.0 < raw.threshold_ < 1.0
        z = math.log(raw.threshold_ / (1.0 - raw.threshold_))
        if calibration == "temperature":
            z /= cal.calibrator_.temperature
        else:
            # The offset moves the threshold as well as the slope.
            assert cal.calibrator_.a > 0 and abs(cal.calibrator_.b) > 0.1
            z = cal.calibrator_.a * z + cal.calibrator_.b
        assert cal.threshold_ == pytest.approx(1.0 / (1.0 + math.exp(-z)), abs=1e-12)

    def test_the_one_half_fallback_of_a_loss_monitor_is_mapped_too(self) -> None:
        # val_loss searches no threshold, so "optimal" is 0.5 on the
        # uncalibrated probabilities.  An increasing map keeps that decision,
        # as it keeps a searched one, and threshold_ is the image of 0.5:
        # σ(a·0 + b) = σ(b), not 0.5 on the calibrated probabilities.
        X, _ = self._noisy_data()
        raw, cal = self._fit(None, monitor="val_loss"), self._fit("platt", monitor="val_loss")
        a, b = cal.calibrator_.a, cal.calibrator_.b
        assert raw.threshold_ == 0.5 and a > 0 and abs(b) > 0.1
        assert cal.threshold_ == pytest.approx(1.0 / (1.0 + math.exp(-b)), abs=1e-12)
        np.testing.assert_array_equal(cal.predict(X), raw.predict(X))
        at_one_half = cal.classes_[(cal.predict_proba(X)[:, 1] >= 0.5).astype(int)]
        assert not np.array_equal(at_one_half, cal.predict(X))

    def test_a_decreasing_platt_map_searches_the_threshold_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hqnn_forge import sklearn as est_module

        monkeypatch.setattr(
            est_module.PlattScaler, "fit", classmethod(lambda cls, z, y: cls(-1.0, 0.0))
        )
        X, y = self._noisy_data()
        est = self._fit("platt")
        va = self._validation(est, y, 0)
        expected = find_optimal_threshold(
            torch.from_numpy(y[va]), torch.from_numpy(est.predict_proba(X[va])[:, 1])
        ).threshold
        assert est.threshold_ == pytest.approx(expected)
        # predict thresholds the calibrated probabilities, which now run the
        # other way: it is not the uncalibrated decision any more.
        np.testing.assert_array_equal(
            est.predict(X),
            est.classes_[(est.predict_proba(X)[:, 1] >= est.threshold_).astype(int)],
        )
        assert not np.array_equal(est.predict(X), self._fit(None).predict(X))

    def test_a_decreasing_platt_map_under_a_loss_monitor_keeps_one_half(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # val_loss searches no threshold, so there is no metric to search again
        # with: the 0.5 fallback applies to the calibrated probabilities.
        from hqnn_forge import sklearn as est_module

        monkeypatch.setattr(
            est_module.PlattScaler, "fit", classmethod(lambda cls, z, y: cls(-1.0, 0.0))
        )
        X, _ = self._noisy_data()
        est = self._fit("platt", monitor="val_loss")
        assert est.threshold_ == 0.5
        np.testing.assert_array_equal(
            est.predict(X), est.classes_[(est.predict_proba(X)[:, 1] >= 0.5).astype(int)]
        )

    def test_pickle_keeps_the_fitted_calibrator(self) -> None:
        X, _ = self._noisy_data()
        est = self._fit("platt")
        loaded = pickle.loads(pickle.dumps(est))
        assert (loaded.calibrator_.a, loaded.calibrator_.b) == (
            est.calibrator_.a,
            est.calibrator_.b,
        )
        np.testing.assert_array_equal(loaded.predict_proba(X), est.predict_proba(X))
        np.testing.assert_array_equal(loaded.predict(X), est.predict(X))

    @staticmethod
    def _replace_logits(
        monkeypatch: pytest.MonkeyPatch,
        logits: Callable[[Callable, torch.Tensor], torch.Tensor],
        threshold: float | None = None,
    ) -> None:
        """Train as usual, then let ``logits(forward, x)`` stand in for the model's output."""
        from hqnn_forge import sklearn as est_module
        from hqnn_forge.training.trainer import _validation_temperature

        def train(model: Any, *args: Any, **kw: Any) -> Any:
            history = train_model(model, *args, **kw)
            forward = model.forward
            model.forward = lambda x: logits(forward, x)  # type: ignore[method-assign]
            # What train_model would have reported for these logits.
            x_val, y_val = args[4], args[5]
            if threshold is not None:
                history.best_threshold = threshold
            elif kw["monitor"] != "val_loss":
                history.best_threshold = find_optimal_threshold(
                    y_val.long(), model.predict_proba(x_val)
                ).threshold
            history.temperature = _validation_temperature(model, (x_val, y_val))
            return history

        monkeypatch.setattr(est_module, "train_model", train)

    def test_predictions_are_unchanged_where_float32_sigmoid_saturates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Near a logit of 13 float32 sigmoid takes one value over a stretch of
        # logits a few hundredths wide.  The threshold is one such value, as
        # find_optimal_threshold returns when two validation probabilities
        # are adjacent floats.  Uncalibrated, every logit rounding onto it is
        # positive; thresholding the calibrated float64 probabilities would
        # split that stretch at the threshold's own logit.
        t = float(torch.sigmoid(torch.tensor(13.0)))
        self._replace_logits(monkeypatch, lambda forward, x: x[:, :1], threshold=t)
        rng = np.random.default_rng(0)
        X = np.zeros((240, 3), dtype=np.float32)
        X[:, 0] = np.linspace(12.9, 13.1, 240)
        y = (X[:, 0] + 0.05 * rng.standard_normal(240) > 13.0).astype(int)
        params = {**self.CAL, "random_state": 0}
        raw = HybridClassifierEstimator(**params).fit(X, y)
        cal: Any = HybridClassifierEstimator(**params, calibration="platt").fit(X, y)
        assert raw.threshold_ == t and cal.calibrator_.a > 0
        assert len(np.unique(raw.predict(X))) == 2
        np.testing.assert_array_equal(raw.predict(X), cal.predict(X))
        from_calibrated = cal.classes_[(cal.predict_proba(X)[:, 1] >= cal.threshold_).astype(int)]
        assert not np.array_equal(from_calibrated, cal.predict(X))

    @pytest.mark.parametrize(
        ("calibration", "match"),
        [("temperature", "separate the classes at 0"), ("platt", "separate")],
    )
    def test_a_split_with_no_finite_fit_warns_and_stays_uncalibrated(
        self, monkeypatch: pytest.MonkeyPatch, calibration: str, match: str
    ) -> None:
        self._replace_logits(monkeypatch, lambda forward, x: 5.0 * x[:, :1])
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 3))
        y = (X[:, 0] > 0).astype(int)
        params = {**self.CAL, "random_state": 0}
        raw = HybridClassifierEstimator(**params).fit(X, y)
        with pytest.warns(UserWarning, match=f"has no fit on the validation split.*{match}"):
            cal = HybridClassifierEstimator(**{**params, "calibration": calibration}).fit(X, y)
        assert cal.calibrator_ is None
        assert cal.threshold_ == raw.threshold_
        np.testing.assert_array_equal(cal.predict_proba(X), raw.predict_proba(X))
        np.testing.assert_array_equal(cal.predict(X), raw.predict(X))

    def test_an_all_negative_threshold_above_one_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # find_optimal_threshold returns the next float32 above 1.0 when
        # labelling everything negative wins and a probability is exactly 1.0.
        # It has no logit to map; it labels everything negative on any scale.
        top = float(torch.nextafter(torch.ones(()), torch.tensor(2.0)))
        self._replace_logits(monkeypatch, lambda forward, x: forward(x), threshold=top)
        X, _ = self._noisy_data()
        est = self._fit("platt")
        assert est.calibrator_ is not None
        assert est.threshold_ == top
        assert (est.predict(X) == est.classes_[0]).all()

    def test_non_finite_validation_logits_warn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._replace_logits(monkeypatch, lambda forward, x: forward(x) * float("nan"))
        with pytest.warns(UserWarning, match="logits must be finite"):
            est = self._fit("platt", monitor="val_loss")
        assert est.calibrator_ is None

    @pytest.mark.parametrize("calibration", ["temperature", "platt"])
    def test_calibrated_probabilities_ignore_the_model_s_train_mode(
        self, calibration: str
    ) -> None:
        # no_grad leaves dropout active; the calibrated path must run in eval
        # mode as the model's own predict_proba does, and restore the mode.
        X, _ = self._noisy_data()
        est = self._fit(calibration, dropout_p=0.5)
        expected = est.predict_proba(X)
        est.model_.train()
        np.testing.assert_array_equal(est.predict_proba(X), expected)
        np.testing.assert_array_equal(est.predict_proba(X), expected)
        assert est.model_.training

    def test_the_temperature_is_the_one_train_model_fitted(self) -> None:
        est = self._fit("temperature")
        assert est.calibrator_.temperature == est.history_.temperature

    def test_calibration_is_refused_for_more_than_two_classes(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((90, 3))
        y = np.arange(90) % 3
        with pytest.raises(ValueError, match="applies to two classes; got 3"):
            HybridClassifierEstimator(**{**self.CAL, "calibration": "temperature"}).fit(X, y)

    def test_fixed_threshold_applies_to_calibrated_probabilities(self) -> None:
        X, _ = self._noisy_data()
        est = self._fit("temperature", threshold=0.5)
        assert est.threshold_ == 0.5
        np.testing.assert_array_equal(
            est.predict(X), est.classes_[(est.predict_proba(X)[:, 1] >= 0.5).astype(int)]
        )

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            (dict(calibration="isotonic"), "calibration must be None, 'temperature' or 'platt'"),
            (dict(calibration="temperature", validation_fraction=0.0), "validation_fraction > 0"),
        ],
    )
    def test_bad_settings_fail_in_fit(self, params: dict, match: str) -> None:
        X, y = self._noisy_data()
        with pytest.raises(ValueError, match=match):
            HybridClassifierEstimator(**{**self.CAL, **params}).fit(X, y)

    def test_params_clone_and_grid_search(self) -> None:
        est = HybridClassifierEstimator(**{**self.CAL, "calibration": "platt"})
        assert est.get_params()["calibration"] == "platt"
        assert clone(est).get_params()["calibration"] == "platt"
        X, y = self._noisy_data()
        search = GridSearchCV(
            HybridClassifierEstimator(**{**self.CAL, "max_epochs": 3}),
            {"calibration": [None, "temperature"]},
            cv=2,
            scoring="neg_log_loss",
        ).fit(X, y)
        assert search.best_params_["calibration"] in (None, "temperature")
        assert len(search.cv_results_["params"]) == 2
