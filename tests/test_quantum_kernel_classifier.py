"""
tests/test_quantum_kernel_classifier.py
=======================================
QuantumKernelClassifier, the QSVM as a scikit-learn estimator (#317).
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import estimator_checks_generator

from hqnn_forge import kernels
from hqnn_forge.kernels import quantum_kernel_matrix
from hqnn_forge.sklearn import QuantumKernelClassifier


def _data(n: int = 40, d: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, (n, d))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({}, id="angle"),
        pytest.param({"encoding": "iqp"}, id="iqp"),
        pytest.param(
            {"encoding": "reuploading", "trainable_input_scaling": True}, id="reuploading"
        ),
        pytest.param({"encoding": "amplitude"}, id="amplitude"),
        pytest.param({"noise_level": 0.1}, id="noisy"),
        # svc_ is then the calibrator's full-data refit, which must equal the plain SVM.
        pytest.param({"probability": True}, id="probability"),
    ],
)
def test_predictions_equal_a_hand_built_precomputed_svc(params: dict[str, Any]) -> None:
    X, y = _data()
    Xt, _ = _data(15, seed=1)
    est = QuantumKernelClassifier(**params, C=2.0, random_state=0).fit(X, y)
    noise = {"noise_level": params.get("noise_level", 0.0)}
    K_train = quantum_kernel_matrix(torch.tensor(X), est.layer_, **noise).numpy()
    K_test = quantum_kernel_matrix(torch.tensor(Xt), est.layer_, torch.tensor(X), **noise).numpy()
    assert K_test.shape == (15, 40)  # rows new samples, columns the training set
    svc = SVC(kernel="precomputed", C=2.0, random_state=0).fit(K_train, y)
    np.testing.assert_array_equal(est.predict(Xt), svc.predict(K_test))
    np.testing.assert_allclose(
        est.decision_function(Xt), svc.decision_function(K_test), atol=1e-10
    )


def test_learns_a_nonlinear_problem() -> None:
    X, y = _data(80)
    Xt, yt = _data(40, seed=2)
    est = QuantumKernelClassifier(C=5.0, random_state=0).fit(X, y)
    assert est.score(Xt, yt) > 0.75  # XOR-like signs; a linear model is at chance


def test_three_classes_and_string_labels() -> None:
    X, _ = _data(60)
    labels = np.array(["a", "b", "c"])[
        (np.arctan2(X[:, 1], X[:, 0]) // (2 * np.pi / 3)).astype(int) % 3
    ]
    est = QuantumKernelClassifier(random_state=0, probability=True).fit(X, labels)
    assert list(est.classes_) == ["a", "b", "c"]
    assert set(est.predict(X)) <= {"a", "b", "c"}
    proba = est.predict_proba(X)
    assert proba.shape == (60, 3)
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-8)


def test_predict_proba_needs_probability() -> None:
    X, y = _data()
    est = QuantumKernelClassifier().fit(X, y)
    assert not hasattr(est, "predict_proba")  # as SVC(probability=False)
    with pytest.raises(AttributeError, match="predict_proba"):
        est.predict_proba(X)


class TestAlignment:
    def test_alignment_trains_the_layer_and_raises_the_alignment(self) -> None:
        X, y = _data(30)
        est = QuantumKernelClassifier(
            encoding="reuploading",
            trainable_input_scaling=True,
            align_steps=15,
            align_lr=0.1,
            random_state=0,
        ).fit(X, y)
        history = est.alignment_history_
        assert len(history) == 15 and history[-1] > history[0]
        untrained = QuantumKernelClassifier(
            encoding="reuploading", trainable_input_scaling=True, random_state=0
        ).fit(X, y)
        assert untrained.alignment_history_ == []
        trained_layer: Any = est.layer_
        untrained_layer: Any = untrained.layer_
        assert not torch.equal(
            trained_layer.qlayer.input_scaling, untrained_layer.qlayer.input_scaling
        )

    def test_alignment_needs_two_classes(self) -> None:
        X, _ = _data(30)
        with pytest.raises(ValueError, match="defined for two classes"):
            QuantumKernelClassifier(align_steps=2).fit(X, np.arange(30) % 3)


class TestTooling:
    def test_cross_validation_grid_search_and_pipeline(self) -> None:
        X, y = _data(60)
        assert cross_val_score(QuantumKernelClassifier(random_state=0), X, y, cv=3).shape == (3,)
        search = GridSearchCV(
            QuantumKernelClassifier(random_state=0), {"C": [0.5, 5.0]}, cv=2
        ).fit(X, y)
        assert search.best_params_["C"] in (0.5, 5.0)
        pipe = make_pipeline(StandardScaler(), QuantumKernelClassifier(random_state=0)).fit(
            X * 10 + 3, y
        )
        assert pipe.score(X * 10 + 3, y) > 0.6

    def test_reproducible(self) -> None:
        X, y = _data()
        a = QuantumKernelClassifier(encoding="reuploading", random_state=3).fit(X, y)
        b = QuantumKernelClassifier(encoding="reuploading", random_state=3).fit(X, y)
        np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))


class TestValidation:
    def test_amplitude_infers_its_qubits(self) -> None:
        X, y = _data(20, d=5)
        est = QuantumKernelClassifier(encoding="amplitude").fit(X, y)
        assert est.layer_.n_qubits == 3 and est.layer_.n_features == 5

    @pytest.mark.parametrize(
        "params, match",
        [
            ({"n_qubits": 2}, "one feature per qubit"),
            ({"encoding": "kernel"}, "encoding must be"),
            ({"trainable_input_scaling": True}, "'reuploading' only"),
            ({"encoding": "amplitude", "trainable_input_scaling": True}, "'reuploading' only"),
            ({"align_steps": -1}, "align_steps must be >= 0"),
            ({"align_steps": 2}, "only changes the kernel of encoding='reuploading'"),
            ({"encoding": "iqp", "align_steps": 2}, "only changes the kernel"),
            (
                {"encoding": "reuploading", "align_steps": 2, "noise_level": 0.1},
                "cannot be combined with noise_level",
            ),
        ],
    )
    def test_bad_parameters_fail_in_fit(self, params: dict[str, Any], match: str) -> None:
        X, y = _data()
        est = QuantumKernelClassifier(**params)  # construction never validates
        with pytest.raises(ValueError, match=match):
            est.fit(X, y)

    @pytest.mark.parametrize("encoding", ["angle", "iqp", "reuploading", "amplitude"])
    def test_one_feature_is_rejected(self, encoding: str) -> None:
        # Amplitude encoding would normalise every sample to ±|0⟩: a kernel of ones.
        X, y = _data()
        with pytest.raises(ValueError, match="at least two features; got n_features = 1"):
            QuantumKernelClassifier(encoding=encoding).fit(X[:, :1], y)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "params, labels, match",
        [
            ({"probability": True}, np.r_[np.zeros(39, int), 1], "two samples of every class"),
            ({}, np.linspace(0, 1, 40), "Unknown label type"),
        ],
    )
    def test_label_checks_run_before_any_simulation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        params: dict[str, Any],
        labels: np.ndarray,
        match: str,
    ) -> None:
        def no_simulation(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("simulated before validating the labels")

        monkeypatch.setattr(kernels, "encoded_states", no_simulation)
        monkeypatch.setattr(kernels, "encoded_density_matrices", no_simulation)
        X, _ = _data()
        with pytest.raises(ValueError, match=match):
            QuantumKernelClassifier(**params).fit(X, labels)

    def test_width_is_checked_at_predict(self) -> None:
        X, y = _data()
        est = QuantumKernelClassifier().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            est.predict(X[:, :2])


def test_probabilities_equal_a_hand_built_calibrated_svc() -> None:
    from sklearn.calibration import CalibratedClassifierCV

    X, y = _data(50)
    Xt, _ = _data(10, seed=4)
    est = QuantumKernelClassifier(probability=True, random_state=0).fit(X, y)
    K_train = quantum_kernel_matrix(torch.tensor(X), est.layer_).numpy()
    K_test = quantum_kernel_matrix(torch.tensor(Xt), est.layer_, torch.tensor(X)).numpy()
    manual = CalibratedClassifierCV(
        SVC(kernel="precomputed", random_state=0), method="sigmoid", ensemble=False, cv=5
    ).fit(K_train, y)
    np.testing.assert_allclose(est.predict_proba(Xt), manual.predict_proba(K_test), atol=1e-10)


# ---------------------------------------------------------------------------
# scikit-learn conventions: conformance, pickling, the caller's RNG
# ---------------------------------------------------------------------------

#: Checks scikit-learn skips itself when an optional package is absent.
MAY_SKIP_CHECKS = {"check_classifier_data_not_an_array", "check_array_api_input"}


def _conformance_params() -> list[Any]:
    params = []
    for probability in (False, True):
        est = QuantumKernelClassifier(probability=probability, random_state=0)
        for estimator, check in estimator_checks_generator(est):
            name = getattr(check, "func", check).__name__
            marks = [pytest.mark.may_skip] if name in MAY_SKIP_CHECKS else []
            params.append(
                pytest.param(estimator, check, marks=marks, id=f"probability={probability}-{name}")
            )
    return params


@pytest.mark.parametrize(("estimator", "check"), _conformance_params())
def test_scikit_learn_conformance(
    estimator: QuantumKernelClassifier, check: Callable[[QuantumKernelClassifier], None]
) -> None:
    check(estimator)


def test_unfitted_raises_not_fitted() -> None:
    X, _ = _data()
    est = QuantumKernelClassifier(probability=True)
    for method in (est.predict, est.decision_function, est.predict_proba):
        with pytest.raises(NotFittedError):
            method(X)


@pytest.mark.parametrize("params", [{}, {"encoding": "reuploading", "noise_level": 0.1}])
def test_pickle_round_trip_predicts_the_same(params: dict[str, Any]) -> None:
    X, y = _data()
    Xt, _ = _data(10, seed=5)
    est = QuantumKernelClassifier(**params, probability=True, random_state=0).fit(X, y)
    torch.manual_seed(123)
    state = torch.random.get_rng_state()
    loaded = pickle.loads(pickle.dumps(est))
    assert torch.equal(torch.random.get_rng_state(), state)  # unpickling leaves the RNG alone
    assert hasattr(est, "layer_")  # pickling does not strip the original
    np.testing.assert_array_equal(loaded.decision_function(Xt), est.decision_function(Xt))
    np.testing.assert_array_equal(loaded.predict_proba(Xt), est.predict_proba(Xt))


def test_seeded_fit_leaves_the_global_rng_alone() -> None:
    X, y = _data()
    torch.manual_seed(7)
    state = torch.random.get_rng_state()
    seed: Any = np.int64(3)  # scikit-learn tools pass NumPy integers
    QuantumKernelClassifier(encoding="reuploading", align_steps=2, random_state=seed).fit(X, y)
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize(
    "fitted, change",
    [
        pytest.param({}, {"noise_level": 0.1}, id="noise-on"),
        pytest.param({"noise_level": 0.1}, {"noise_level": 0.2}, id="noise-stronger"),
        pytest.param({"noise_level": 0.1}, {"noise_position": "end"}, id="noise-position"),
    ],
)
def test_set_params_after_fit_does_not_change_predictions(
    fitted: dict[str, Any], change: dict[str, Any]
) -> None:
    X, y = _data()
    Xt, _ = _data(10, seed=6)
    est = QuantumKernelClassifier(**fitted, random_state=0).fit(X, y)
    before = est.decision_function(Xt)
    est.set_params(**change)  # takes effect at the next fit, as in any estimator
    np.testing.assert_array_equal(est.decision_function(Xt), before)


def test_pickle_rebuilds_the_fitted_layer_not_the_current_parameters() -> None:
    X, y = _data()
    Xt, _ = _data(10, seed=7)
    est = QuantumKernelClassifier(encoding="reuploading", random_state=0).fit(X, y)
    before = est.decision_function(Xt)
    # IQP has the same weight shapes, so load_state_dict alone would not notice.
    est.set_params(encoding="iqp")
    loaded = pickle.loads(pickle.dumps(est))
    assert type(loaded.layer_) is type(est.layer_)
    np.testing.assert_array_equal(loaded.decision_function(Xt), before)


@pytest.mark.parametrize("width", [3, 4])
def test_failed_refit_keeps_the_previous_model(width: int) -> None:
    X, y = _data()
    est = QuantumKernelClassifier(encoding="amplitude", random_state=0).fit(X, y)
    layer, svc, before = est.layer_, est.svc_, est.decision_function(X)
    X_bad, _ = _data(d=width, seed=8)
    X_bad[0] = 0.0  # amplitude encoding cannot normalise an all-zero sample
    with pytest.raises(ValueError, match="zero"):
        est.set_params(random_state=1).fit(X_bad, y)
    assert est.layer_ is layer and est.svc_ is svc and est.n_features_in_ == 3
    np.testing.assert_array_equal(est.decision_function(X), before)


def test_predict_proba_follows_the_fit_not_set_params() -> None:
    X, y = _data()
    est = QuantumKernelClassifier(random_state=0).fit(X, y)
    est.set_params(probability=True)
    assert not hasattr(est, "predict_proba")  # unchanged until the next fit
    est = QuantumKernelClassifier(probability=True, random_state=0).fit(X, y)
    proba = est.predict_proba(X)
    est.set_params(probability=False)
    np.testing.assert_array_equal(est.predict_proba(X), proba)
