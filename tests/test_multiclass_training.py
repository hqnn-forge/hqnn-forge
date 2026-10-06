"""
tests/test_multiclass_training.py
=================================
Multiclass metrics, the softmax focal loss, ``train_model`` on a multiclass
model, and the scikit-learn estimator with more than two classes (#309).
"""

from __future__ import annotations

import math
import pickle
import warnings
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from hqnn_forge.evaluation import (
    MULTICLASS_METRICS,
    macro_f1_score,
    multiclass_balanced_accuracy,
    multiclass_matthews_corrcoef,
)
from hqnn_forge.training import train_model
from hqnn_forge.utils import SoftmaxFocalLoss

sklearn_metrics = pytest.importorskip("sklearn.metrics")


class TestMetricsAgainstScikitLearn:
    def test_random_cases(self) -> None:
        rng = np.random.default_rng(0)
        worst = dict.fromkeys(("mcc", "f1", "ba"), 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn warns on absent classes
            for _ in range(500):
                k, n = int(rng.integers(2, 6)), int(rng.integers(1, 40))
                t = rng.integers(0, k, n)
                p = np.where(rng.random(n) < 0.5, t, rng.integers(0, k, n))
                tt, pp = torch.tensor(t), torch.tensor(p)
                pairs = {
                    "mcc": (
                        multiclass_matthews_corrcoef(tt, pp),
                        sklearn_metrics.matthews_corrcoef(t, p),
                    ),
                    "f1": (
                        macro_f1_score(tt, pp),
                        sklearn_metrics.f1_score(t, p, average="macro", zero_division=0),
                    ),
                    "ba": (
                        multiclass_balanced_accuracy(tt, pp),
                        sklearn_metrics.balanced_accuracy_score(t, p),
                    ),
                }
                for name, (ours, theirs) in pairs.items():
                    worst[name] = max(worst[name], abs(ours - theirs))
        assert max(worst.values()) < 1e-12, worst

    def test_known_values(self) -> None:
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        assert multiclass_matthews_corrcoef(y, y) == pytest.approx(1.0)
        assert multiclass_matthews_corrcoef(y, torch.zeros(6, dtype=torch.long)) == 0.0
        assert macro_f1_score(y, y) == pytest.approx(1.0)
        # Two of three classes fully right, the third never predicted.
        assert multiclass_balanced_accuracy(y, torch.tensor([0, 1, 0, 0, 1, 0])) == pytest.approx(
            2 / 3
        )

    def test_binary_case_is_the_binary_mcc(self) -> None:
        from hqnn_forge.evaluation import matthews_corrcoef

        t = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0])
        p = torch.tensor([0, 1, 0, 0, 1, 1, 1, 0])
        assert multiclass_matthews_corrcoef(t, p) == pytest.approx(matthews_corrcoef(t, p))

    def test_names_match_the_binary_monitors(self) -> None:
        from hqnn_forge.evaluation import METRICS

        assert set(MULTICLASS_METRICS) == set(METRICS)

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            macro_f1_score(torch.tensor([0, 1]), torch.tensor([0]))
        with pytest.raises(ValueError, match="empty"):
            multiclass_matthews_corrcoef(torch.tensor([]), torch.tensor([]))


class TestSoftmaxFocalLoss:
    def test_gamma_zero_is_cross_entropy(self) -> None:
        logits = torch.randn(9, 4, generator=torch.Generator().manual_seed(0))
        target = torch.randint(0, 4, (9,), generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(
            SoftmaxFocalLoss(0.0)(logits, target), nn.CrossEntropyLoss()(logits, target)
        )

    def test_focusing_down_weights_easy_samples(self) -> None:
        # Per sample the focal term (1 − p_t)^γ ≤ 1, so the loss is below
        # cross-entropy, and much below it for a confident correct sample.
        logits = torch.tensor([[8.0, 0.0, 0.0], [0.2, 0.1, 0.0]])
        target = torch.tensor([0, 0])
        ce = nn.CrossEntropyLoss(reduction="none")(logits, target)
        fl = torch.stack(
            [SoftmaxFocalLoss(2.0)(logits[i : i + 1], target[i : i + 1]) for i in range(2)]
        )
        assert (fl <= ce).all() and fl[0] < 1e-3 * ce[0]

    def test_negative_gamma(self) -> None:
        with pytest.raises(ValueError, match="gamma must be ≥ 0"):
            SoftmaxFocalLoss(-1.0)


def _blobs(n: int = 90, k: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.array([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7], [0.0, 3.0]])[:k]
    y = np.arange(n) % k
    X = centres[y] + 0.5 * rng.standard_normal((n, 2))
    return X, y


class TestTrainModel:
    @pytest.mark.parametrize("monitor", ["mcc", "f1", "balanced_accuracy"])
    def test_multiclass_monitor_scores_argmax_labels(self, monitor: str) -> None:
        # Overlapping classes and one epoch, so the three metrics differ and a
        # monitor scored with the wrong one is caught.
        rng = np.random.default_rng(1)
        y = np.arange(90) % 3
        X = rng.standard_normal((90, 2)) + 0.6 * np.eye(3)[y][:, :2]
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
        torch.manual_seed(0)
        model = nn.Linear(2, 3)
        history = train_model(
            model,
            nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            Xt,
            yt,
            Xt,
            yt,
            max_epochs=1,
            monitor=monitor,
            restore_best=False,
        )
        with torch.no_grad():
            labels = model(Xt).argmax(-1)
        scores = {name: fn(yt, labels) for name, fn in MULTICLASS_METRICS.items()}
        assert len({round(v, 9) for v in scores.values()}) == 3, scores
        assert history.epochs[-1].val_score == pytest.approx(scores[monitor])
        assert history.best_threshold is None and history.epochs[-1].val_threshold is None


sklearn = pytest.importorskip("sklearn")
from sklearn.model_selection import GridSearchCV, cross_val_score

from hqnn_forge.models import MulticlassHybridClassifier
from hqnn_forge.sklearn import HybridClassifierEstimator

FAST: dict[str, Any] = dict(
    n_qubits=3,
    n_layers=1,
    device_name="default.qubit",
    diff_method="backprop",
    max_epochs=30,
    batch_size=32,
    lr=0.05,
    random_state=0,
)


@pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
@pytest.mark.parametrize("loss", ["bce", "focal"])
class TestEstimator:
    def test_fits_three_classes(self, strategy: Any, loss: Any) -> None:
        X, y = _blobs()
        est = HybridClassifierEstimator(**FAST, strategy=strategy, loss=loss).fit(X, y)
        assert isinstance(est.model_, MulticlassHybridClassifier)
        assert est.model_.strategy == strategy and est.model_.n_classes == 3
        proba = est.predict_proba(X)
        assert proba.shape == (90, 3)
        np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-6)
        assert est.threshold_ is None
        assert est.score(X, y) > 0.8  # separable blobs; chance is 1/3


def test_string_labels_and_class_order() -> None:
    X, y = _blobs()
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    est = HybridClassifierEstimator(**FAST).fit(X, names)
    assert list(est.classes_) == ["setosa", "versicolor", "virginica"]
    pred = est.predict(X)
    assert set(pred) <= set(names)
    # With softmax, predict is the argmax of predict_proba, column by column.
    np.testing.assert_array_equal(pred, est.classes_[est.predict_proba(X).argmax(1)])


def test_validation_split_keeps_every_class() -> None:
    X, y = _blobs(n=120, k=4)
    est = HybridClassifierEstimator(
        **{**FAST, "max_epochs": 5}, validation_fraction=0.25, patience=None
    ).fit(X, y)
    assert est.history_.best_epoch is not None
    assert est.history_.best_threshold is None


def test_sklearn_tooling_with_three_classes() -> None:
    X, y = _blobs()
    fast = {**FAST, "max_epochs": 5}
    scores = cross_val_score(HybridClassifierEstimator(**fast), X, y, cv=3)
    assert scores.shape == (3,)
    search = GridSearchCV(
        HybridClassifierEstimator(**fast), {"strategy": ["softmax", "one_vs_rest"]}, cv=2
    ).fit(X, y)
    assert search.best_params_["strategy"] in ("softmax", "one_vs_rest")
    assert HybridClassifierEstimator().__sklearn_tags__().classifier_tags.multi_class


@pytest.mark.parametrize(
    "params, match",
    [
        ({"model": "parallel"}, "model='parallel' is a binary topology"),
        ({"threshold": 0.4}, "applies to two classes"),
        ({"strategy": "ovo"}, "strategy must be 'softmax' or 'one_vs_rest'"),
    ],
)
def test_multiclass_errors(params: dict[str, Any], match: str) -> None:
    X, y = _blobs()
    with pytest.raises(ValueError, match=match):
        HybridClassifierEstimator(**{**FAST, **params}).fit(X, y)


def test_two_classes_still_train_the_binary_model() -> None:
    X, y = _blobs(k=2)
    est = HybridClassifierEstimator(**FAST).fit(X, y)
    assert type(est.model_).__name__ == "HybridBinaryClassifier"
    assert isinstance(est.threshold_, float) and est.predict_proba(X).shape == (90, 2)


@pytest.mark.parametrize(
    "strategy, loss, expected",
    [
        ("softmax", "bce", nn.CrossEntropyLoss),
        ("softmax", "focal", SoftmaxFocalLoss),
        ("one_vs_rest", "bce", nn.BCEWithLogitsLoss),
        ("one_vs_rest", "focal", "FocalLoss"),
    ],
)
def test_each_strategy_trains_with_its_own_likelihood(
    strategy: str, loss: str, expected: Any
) -> None:
    est = HybridClassifierEstimator(strategy=strategy, loss=loss)  # type: ignore[arg-type]
    fn = est._loss(3)
    logits = torch.randn(5, 3, generator=torch.Generator().manual_seed(0))
    y = torch.tensor([0, 2, 1, 1, 0])
    if strategy == "softmax":
        assert isinstance(fn, expected)
    else:
        # One binary loss per class, on one-hot targets.
        inner: Any = fn.loss
        assert type(inner).__name__ == (
            expected if isinstance(expected, str) else expected.__name__
        )
        one_hot = nn.functional.one_hot(y, 3).float()
        torch.testing.assert_close(fn(logits, y), inner(logits, one_hot))


def test_the_holdout_stratifies_every_class() -> None:
    y = np.repeat(np.arange(4), [20, 12, 8, 40])
    train, val = HybridClassifierEstimator._stratified_holdout(y, 0.25, np.random.default_rng(0))
    assert np.intersect1d(train, val).size == 0 and train.size + val.size == y.size
    np.testing.assert_array_equal(np.bincount(y[val]), [5, 3, 2, 10])


def test_non_contiguous_integer_labels_round_trip() -> None:
    X, y = _blobs()
    labels = np.array([10, -3, 7])[y]
    est = HybridClassifierEstimator(**FAST).fit(X, labels)
    np.testing.assert_array_equal(est.classes_, [-3, 7, 10])
    assert est.predict(X).dtype == labels.dtype
    assert est.score(X, labels) > 0.8


@pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
def test_fitted_multiclass_estimator_pickles(strategy: Any) -> None:
    X, y = _blobs()
    est = HybridClassifierEstimator(**{**FAST, "max_epochs": 2}, strategy=strategy).fit(X, y)
    loaded = pickle.loads(pickle.dumps(est))
    assert isinstance(loaded.model_, MulticlassHybridClassifier)
    assert loaded.model_.strategy == strategy
    np.testing.assert_array_equal(loaded.predict_proba(X), est.predict_proba(X))
    np.testing.assert_array_equal(loaded.predict(X), est.predict(X))


def test_train_model_casts_integer_binary_labels_for_bce() -> None:
    # Labels are cast per loss call; a binary model given integer labels must
    # still train (BCEWithLogitsLoss rejects integer targets) and match the
    # same run on float labels exactly.
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.standard_normal((40, 2)), dtype=torch.float32)
    y = (X[:, 0] > 0).long()
    runs = []
    for labels in (y, y.float()):
        torch.manual_seed(0)
        model = nn.Linear(2, 1)
        history = train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            X,
            labels,
            X,
            labels,
            max_epochs=3,
            generator=torch.Generator().manual_seed(0),
        )
        runs.append([(r.train_loss, r.val_loss) for r in history.epochs])
    assert runs[0] == runs[1]


_NON_INTEGER = "integer class labels; {split} holds non-integer"
_OUT_OF_RANGE = r"labels in \[0, 2\]; {split} holds"


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (1.7, _NON_INTEGER),  # .long() would truncate it to class 1
        (0.5, _NON_INTEGER),
        (math.nan, _NON_INTEGER),
        (math.inf, _NON_INTEGER),
        (3, _OUT_OF_RANGE),  # the losses raise only in the batch holding it
        (-1, _OUT_OF_RANGE),
        (-100, _OUT_OF_RANGE),  # CrossEntropyLoss's ignore_index: silently skipped
    ],
)
@pytest.mark.parametrize("split", ["y_train", "y_val"])
@pytest.mark.parametrize("loss_name", ["cross_entropy", "softmax_focal", "one_vs_rest"])
def test_train_model_rejects_bad_multiclass_labels_before_any_step(
    bad: float, match: str, split: str, loss_name: str
) -> None:
    # A single bad label in the last sample of either split, so outside the
    # first batch of y_train and seen by validation only after an epoch.  The
    # raise must come before any optimiser step, whatever the loss would do.
    from hqnn_forge.sklearn import _OneHotLoss

    losses = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "softmax_focal": SoftmaxFocalLoss(),
        "one_vs_rest": _OneHotLoss(nn.BCEWithLogitsLoss(), 3),
    }
    X = torch.randn(30, 2)
    model = nn.Linear(2, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    y = (torch.arange(30) % 3).float()
    y_bad = y.clone()
    y_bad[-1] = bad
    y_train, y_val = (y_bad, y) if split == "y_train" else (y, y_bad)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match=match.format(split=split)):
        train_model(
            model, losses[loss_name], opt, X, y_train, X, y_val, max_epochs=1, batch_size=8
        )
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.float32, torch.float64])
def test_train_model_accepts_class_index_labels_of_any_dtype(dtype: torch.dtype) -> None:
    # Integer and whole-valued float labels are the same class indices and
    # train identically.
    X = torch.randn(30, 2)
    y = torch.arange(30) % 3
    losses = []
    for labels in (y, y.to(dtype)):
        torch.manual_seed(0)
        model = nn.Linear(2, 3)
        history = train_model(
            model,
            nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            X,
            labels,
            X,
            labels,
            max_epochs=2,
            batch_size=8,
            generator=torch.Generator().manual_seed(0),
        )
        losses.append([(r.train_loss, r.val_loss) for r in history.epochs])
    assert losses[0] == losses[1]


def test_train_model_leaves_binary_labels_unchecked() -> None:
    # The multiclass check does not reach binary models: soft and bool labels
    # train as before.
    X = torch.randn(30, 2)
    for labels in (torch.linspace(0, 1, 30), torch.arange(30) % 2 == 0):
        model = nn.Linear(2, 1)
        train_model(
            model,
            nn.BCEWithLogitsLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            X,
            labels,
            max_epochs=1,
        )
