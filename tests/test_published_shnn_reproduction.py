"""
tests/test_published_shnn_reproduction.py
=========================================
Does this library reproduce the published SHNN's *numbers*, not only its
structure (#184)?  ``test_published_shnn_parity.py`` pins the architecture;
the thesis and ``hqnn-fraud-detection-benchmark`` (@ cade1ea) report

    MCC        0.5758 ± 0.0371   (5 folds, held-out test set)
    MCC/kParam 4.720             (= 0.5758 / 0.122, 122 trainable parameters)

This module runs the benchmark's recipe with the library's own components and
checks the mean MCC lands inside the published ±0.0371 band, and MCC/kParam
inside the same band divided by 0.122 kParam.  If it does not, the outcome to
record is the discrepancy, not a wider tolerance.

Opt-in only
-----------
The run needs the Kaggle dataset (not redistributable) and days of state-vector
simulation at ~2 s per 256-sample batch, so it is skipped unless
``HQNN_FORGE_REPRODUCE=1`` is set, skipped without the dataset, and marked
``reproducibility``, ``slow`` and ``may_skip`` (so the skip does not fail CI
under ``HQNN_FORGE_FAIL_ON_SKIP=1``).  Run it deliberately with::

    HQNN_FORGE_REPRODUCE=1 HQNN_FORGE_DATA=data/raw \\
        pytest tests/test_published_shnn_reproduction.py -m reproducibility -s

``-s`` shows the per-fold log.  A small synthetic run of the same recipe
(``test_recipe_runs_end_to_end``) is part of the normal suite, so the pipeline
code cannot rot between deliberate runs.

The recipe (benchmark ``configs/default.yaml``, ``src/data/cv.py``,
``src/training/trainer.py``)
------------------------------------------------------------------------------
* ``Time`` dropped; 80/20 stratified split, ``random_state=42``: the 20% is
  the held-out test set every fold is scored on.
* 5-fold stratified CV on the 80%, shuffled, seed 42.  Per fold, fitted on
  its training part only: ``RobustScaler`` → ``MinMaxScaler`` to [0, π] →
  ``PCA(8)``; then SMOTE (k = 5) on the training part only.
* ``published_shnn()``, Adam (lr 0.01, weight decay 1e-5), batch 256, up to
  100 epochs, early stopping on validation MCC with patience 20 (strict
  improvement), best epoch restored.  Validation MCC is taken at the
  threshold that maximises it on the benchmark's grid, ``np.arange(0.05,
  0.95, 0.01)`` (first maximum wins), for the early-stopping signal and again
  on the restored model for the test threshold.  The grid tops out at 0.94,
  and a SMOTE-balanced model on 0.17% fraud often wants a higher one (the
  benchmark's own trainer notes a fold tuned to 0.94), so the grid is part of
  the recipe: the library's exhaustive ``find_optimal_threshold`` would score
  a different, typically higher, MCC.  That is why the epochs are driven here
  one ``train_model`` call at a time rather than by its own ``monitor``.

Deviations, each deliberate
---------------------------
* Folds and SMOTE come from :mod:`hqnn_forge.preprocessing` (the question is
  whether the *library* reproduces the result), so the synthetic rows and the
  fold assignment differ from imblearn's and scikit-learn's draws.  The
  scalers, PCA and the held-out split are scikit-learn's, as in the benchmark.
* ``BCEWithLogitsLoss`` on logits instead of ``Sigmoid`` + ``BCELoss``: the
  same objective, numerically stabler in the tails.
* The classical layers are reset to PyTorch's default ``nn.Linear`` init
  (``reset_parameters``), as the benchmark's SHNN leaves them, instead of the
  classifiers' Xavier init.  The quantum weights keep the preset's
  ``N(0, 0.1²)``, which already matches.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn as nn

from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.preprocessing import smote, stratified_kfold
from hqnn_forge.training import train_model

REPRODUCE_ENV = "HQNN_FORGE_REPRODUCE"
PUBLISHED_MCC = 0.5758
PUBLISHED_MCC_STD = 0.0371
PUBLISHED_PARAMS = 122
PUBLISHED_MCC_PER_KPARAM = 4.720
SEED = 42


def benchmark_threshold(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> float:
    """The benchmark's ``find_optimal_threshold``: first MCC maximum on its 0.05..0.94 grid."""
    from sklearn.metrics import matthews_corrcoef

    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    best_mcc, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        mcc = matthews_corrcoef(y_true, (y_prob >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return float(best_t)


def run_published_recipe(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    *,
    n_folds: int = 5,
    max_epochs: int = 100,
    patience: int = 20,
    batch_size: int = 256,
    log: Callable[[str], None] = print,
    **model_overrides: object,
) -> list[float]:
    """Test-set MCC of each fold, following the benchmark recipe above."""
    sk_model_selection = pytest.importorskip("sklearn.model_selection")
    sk_preprocessing = pytest.importorskip("sklearn.preprocessing")
    sk_decomposition = pytest.importorskip("sklearn.decomposition")
    sk_pipeline = pytest.importorskip("sklearn.pipeline")
    from sklearn.metrics import matthews_corrcoef

    X_dev, X_test, y_dev, y_test = sk_model_selection.train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    scores = []
    for fold, (tr, va) in enumerate(stratified_kfold(y_dev, n_folds, random_state=SEED)):
        pre = sk_pipeline.make_pipeline(
            sk_preprocessing.RobustScaler(),
            sk_preprocessing.MinMaxScaler(feature_range=(0.0, math.pi)),
            sk_decomposition.PCA(n_components=8, random_state=SEED),
        )
        x_tr = pre.fit_transform(X_dev[tr])
        x_va, x_te = pre.transform(X_dev[va]), pre.transform(X_test)
        oversampled = smote(x_tr, y_dev[tr], k_neighbors=5, random_state=SEED)

        torch.manual_seed(SEED + fold)
        model = HybridBinaryClassifier.published_shnn(**model_overrides)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()  # the benchmark's default nn.Linear init

        def tensor(a: npt.ArrayLike) -> torch.Tensor:
            return torch.as_tensor(np.asarray(a), dtype=torch.float32)

        x_val_t, y_val_np = tensor(x_va), y_dev[va]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        generator = torch.Generator().manual_seed(SEED + fold)
        best_mcc, best_epoch, stale = -1.0, 0, 0
        best_state = copy.deepcopy(model.state_dict())
        for epoch in range(1, max_epochs + 1):
            # One epoch per call, no validation inside: the stopping signal is
            # the benchmark's grid-thresholded MCC, computed below.
            (record,) = train_model(
                model,
                nn.BCEWithLogitsLoss(),
                optimizer,
                tensor(oversampled.X),
                tensor(oversampled.y),
                max_epochs=1,
                batch_size=batch_size,
                generator=generator,
            ).epochs
            val_prob = model.predict_proba(x_val_t).numpy()
            val_pred = (val_prob >= benchmark_threshold(y_val_np, val_prob)).astype(int)
            val_mcc = float(matthews_corrcoef(y_val_np, val_pred))
            log(f"fold {fold} epoch {epoch:3d} loss {record.train_loss:.4f} val_mcc {val_mcc:.4f}")
            if val_mcc > best_mcc:
                best_mcc, best_epoch, stale = val_mcc, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        threshold = benchmark_threshold(y_val_np, model.predict_proba(x_val_t).numpy())
        pred = model.predict(tensor(x_te), threshold=threshold).numpy()
        scores.append(float(matthews_corrcoef(y_test, pred)))
        log(
            f"fold {fold}: best epoch {best_epoch}, threshold {threshold:.2f}, "
            f"test MCC {scores[-1]:.4f}"
        )
    return scores


def test_benchmark_threshold_is_capped_and_takes_the_first_maximum() -> None:
    """The grid, not the exhaustive search: 0.94 at most, and the lowest tied threshold."""
    pytest.importorskip("sklearn")
    from hqnn_forge.evaluation import find_optimal_threshold

    # Separable only in (0.965, 0.985]: the exhaustive search finds MCC 1 at
    # 0.975, while the grid stops at 0.94, so its best is the labelling of
    # (0.905, 0.965] -- one negative still called positive -- first reached at 0.91.
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.105, 0.505, 0.905, 0.965, 0.985, 0.995])
    assert find_optimal_threshold(y, p).threshold == pytest.approx(0.975)
    assert benchmark_threshold(y, p) == pytest.approx(0.91)
    # Every threshold in (0.305, 0.705] separates these; the grid keeps the first.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.205, 0.305, 0.705, 0.805])
    assert benchmark_threshold(y, p) == pytest.approx(0.31)


def test_recipe_runs_end_to_end() -> None:
    """The recipe on a small synthetic imbalanced set, so the code above stays runnable."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 30))
    y = (X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=400) > 2.2).astype(np.int64)
    assert 10 < y.sum() < 60
    scores = run_published_recipe(
        X,
        y,
        n_folds=2,
        max_epochs=2,
        log=lambda _: None,
        device_name="default.qubit",
        diff_method="backprop",
    )
    assert len(scores) == 2 and all(-1.0 <= s <= 1.0 for s in scores)


@pytest.mark.reproducibility
@pytest.mark.slow
@pytest.mark.may_skip  # opt-in: skips in CI even under HQNN_FORGE_FAIL_ON_SKIP=1
def test_published_mcc_is_reproduced() -> None:
    if os.environ.get(REPRODUCE_ENV) != "1":
        pytest.skip(f"opt-in: set {REPRODUCE_ENV}=1 (days of simulation; see module docstring)")
    from hqnn_forge.data import DatasetNotFoundError, load_credit_card_fraud

    try:
        data = load_credit_card_fraud(drop_time=True, strict=True)
    except DatasetNotFoundError as exc:
        pytest.skip(f"dataset not available: {exc}")

    assert HybridBinaryClassifier.published_shnn().count_parameters() == PUBLISHED_PARAMS
    scores = run_published_recipe(data.X, data.y)
    mean, std = float(np.mean(scores)), float(np.std(scores, ddof=1))
    per_kparam = mean / (PUBLISHED_PARAMS / 1000)
    print(f"MCC {mean:.4f} ± {std:.4f} over {len(scores)} folds, MCC/kParam {per_kparam:.3f}")
    assert abs(mean - PUBLISHED_MCC) <= PUBLISHED_MCC_STD, scores
    assert abs(per_kparam - PUBLISHED_MCC_PER_KPARAM) <= PUBLISHED_MCC_STD / (
        PUBLISHED_PARAMS / 1000
    )
