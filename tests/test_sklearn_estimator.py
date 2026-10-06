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
