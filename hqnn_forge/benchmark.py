"""
hqnn_forge.benchmark
====================
Train a hybrid model and its matched classical control on identical folds of
several datasets, and report one row per dataset and model.

This answers the library's question -- does a small quantum layer earn its
parameters? -- the same way every time, instead of by a comparison assembled
by hand.

What both models share
----------------------
For each dataset, the outer folds come from :func:`stratified_kfold`, and in
every fold the hybrid model and its control (:func:`classical_baseline`,
matched to the hybrid's live parameter count) get exactly the same:

* **Scaling.**  Features are standardised with the mean and standard deviation
  of the fold's training part only, then applied to every row of the fold.
* **Validation split.**  The training part is split once more, stratified: one
  ``validation_folds``-th is held out for early stopping and threshold tuning.
  It holds only real rows.
* **Oversampling.**  With ``oversample=True``, SMOTE runs on the remaining
  training rows only, after scaling, since its neighbour search measures
  distances.  Test and validation rows never contribute a synthetic sample.
* **Training.**  The same loss, optimiser, learning rate, batch size, epoch
  budget and early stopping, the same batch order (one generator seed per
  fold), and the same initialisation seed.  Each model is built and trained
  inside :func:`torch.random.fork_rng`, so the run neither depends on nor
  disturbs the caller's global RNG.
* **Threshold.**  The threshold that maximises MCC on the validation split at
  the best epoch (``TrainingHistory.best_threshold``), applied unchanged to
  the fold's test rows.  The test rows are used for nothing else.

Reported per dataset and model
------------------------------
``mcc_mean``/``mcc_std`` over the test folds (with ``n_seeds > 1``, of each
fold's mean over its initialisation seeds, and ``mcc_seed_std`` the mean
across-seed standard deviation), ``n_parameters`` (the total trainable count,
``count_parameters()``, including circuit weights that can never move the
output) and MCC per 1,000 of them (:func:`parameter_efficiency` of the mean).
The control is matched to the hybrid's live count, so the two rows'
``n_parameters`` differ by the hybrid's inert weights as well as by the width
rounding: for the published SHNN, 122 (102 live) against 101 (#234).  Then the
wall-clock
training time summed over folds (simulating the circuit is part of an honest
efficiency comparison), and the paired Wilcoxon signed-rank test of hybrid
against control over the per-fold MCCs, with its effect size.  The test
columns are the same on both rows of a dataset.

``wilcoxon_min_p`` is the smallest p-value the test could have produced with
this many folds: with 5 folds, two-sided, it is 0.0625, so no outcome reaches
0.05.  A difference then needs more folds (``n_splits``) to be declared, not
a lower threshold.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from hqnn_forge.evaluation import (
    brier_score,
    expected_calibration_error,
    matthews_corrcoef,
    parameter_efficiency,
    rank_biserial_correlation,
    wilcoxon_signed_rank,
)
from hqnn_forge.models import BinaryClassifierBase, HybridBinaryClassifier
from hqnn_forge.noise import Position, apply_depolarizing_noise, validate_noise
from hqnn_forge.preprocessing import oversample_fold, stratified_kfold
from hqnn_forge.training import train_model
from hqnn_forge.utils import FocalLoss, classical_baseline

HybridBuilder = Callable[[int], nn.Module]
LossBuilder = Callable[[], nn.Module]

#: The two models of every dataset, in report order.
MODELS: tuple[str, str] = ("hybrid", "control")

#: Columns of :attr:`BenchmarkResult.records` and of the CSV, in order.
COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "architecture",
    "n_samples",
    "n_positives",
    "n_folds",
    "n_parameters",
    "mcc_mean",
    "mcc_std",
    "n_seeds",
    "mcc_seed_std",
    "mcc_per_kparam",
    "brier_mean",
    "ece_mean",
    "fold_mcc",
    "train_seconds",
    "wilcoxon_p",
    "wilcoxon_min_p",
    "rank_biserial",
)


def default_hybrid(n_input_features: int) -> nn.Module:
    """``HybridBinaryClassifier(n_input_features, n_qubits=8, n_layers=2)``."""
    return HybridBinaryClassifier(n_input_features, 8, 2)


@dataclass(frozen=True)
class FoldResult:
    """
    One model on one outer fold.  The index arrays point into the dataset's
    rows, so the two models of a fold can be checked to have seen the same
    data.
    """

    dataset: str
    model: str
    fold: int
    train_idx: npt.NDArray[np.intp]
    """Rows the model was trained on (before SMOTE added synthetic ones)."""
    val_idx: npt.NDArray[np.intp]
    """Rows used for early stopping and threshold tuning."""
    test_idx: npt.NDArray[np.intp]
    """Rows the reported score comes from."""
    n_synthetic: int
    split_seed: int
    """Seed of the dataset's outer stratified split (the same for every fold)."""
    inner_seed: int
    """Seed of the split of the fold's training rows into train and validation."""
    smote_seed: int
    """Seed of the fold's SMOTE draw (unused with ``oversample=False``)."""
    init_seed: int
    batch_seed: int
    threshold: float
    mcc: float
    train_seconds: float
    epochs: int
    device: str | None
    """PennyLane device the circuit actually ran on (after any fallback); ``None`` for the control."""
    seed_index: int = 0
    """Which of the ``n_seeds`` initialisations of this fold (``init_seed`` is its seed)."""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    """The training settings tuning chose for this fold and model; empty without tuning."""
    noise_mcc: dict[float, float] = field(default_factory=dict)
    """Hybrid only: test MCC under depolarising noise of each swept probability."""
    brier: float = math.nan
    """Brier score of the test probabilities (#319)."""
    ece: float = math.nan
    """Expected calibration error of the test probabilities, ``ECE_BINS`` quantile bins."""


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Attributes
    ----------
    records:
        One dict per dataset and model with the keys in :data:`COLUMNS`;
        ``fold_mcc`` is a tuple of the per-fold scores.
    folds:
        Every :class:`FoldResult`, in dataset, fold, model order.
    settings:
        The ``run_benchmark`` arguments other than the datasets and the
        builder, with ``loss`` as its import path.
    models:
        Per dataset: ``{"hybrid": {"class": ..., "config": ...}, "control":
        {...}}``, each model's class name and ``get_config()``.
    datasets:
        Per dataset: ``n_samples``, ``n_features``, ``n_positives`` and the
        SHA-256 of the float64 features and int64 labels, so a re-run can
        check it was given the same data.
    """

    records: list[dict[str, Any]]
    folds: list[FoldResult]
    settings: dict[str, Any]
    models: dict[str, dict[str, dict[str, Any]]]
    datasets: dict[str, dict[str, Any]]
    noise: list[dict[str, Any]] = field(default_factory=list)
    """
    With ``noise_levels``: one row per dataset and noise level -- the hybrid's
    and the noise-free control's mean MCC over folds, and the one-sided paired
    Wilcoxon test that the hybrid is better (``p_hybrid_better``,
    ``min_p``).
    """
    noise_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    """
    Per dataset: whether the hybrid is significantly better than the control
    without noise (``better_noiseless``), and the first swept noise level at
    which it no longer is (``lost_at``; ``None`` if it never was, or never
    stopped being within the sweep).
    """


def fingerprint(X: npt.NDArray[np.float64], y: npt.NDArray[np.int64]) -> str:
    """SHA-256 over the shape and bytes of ``X`` (float64) and ``y`` (int64)."""
    digest = hashlib.sha256()
    for array, dtype in ((X, np.float64), (y, np.int64)):
        contiguous = np.ascontiguousarray(array, dtype=dtype)
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _device_name(model: nn.Module) -> str | None:
    """The device the model's QNode is bound to, or ``None`` without one."""
    layer = getattr(model, "quantum_layer", None)
    qnode = getattr(getattr(layer, "qlayer", None), "qnode", None)
    device = getattr(qnode, "device", None)
    return None if device is None else str(device.name)


def _import_path(obj: object) -> str:
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}"


def _standardise(
    X: npt.NDArray[np.float64], rows: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """``X`` scaled by the mean and std of ``X[rows]``; constant columns only centred."""
    mean = X[rows].mean(axis=0)
    std = X[rows].std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def _seeds(root: np.random.SeedSequence, n: int) -> list[int]:
    return [int(s.generate_state(1)[0]) for s in root.spawn(n)]


#: Bins of the per-fold expected calibration error.  Equal-count
#: ("quantile") bins: on imbalanced data equal-width ones put nearly every
#: sample in the bin nearest 0 and leave the rest near-empty.
ECE_BINS = 10


class FitScore(NamedTuple):
    """One trained model scored on its test rows."""

    mcc: float
    threshold: float
    seconds: float
    epochs: int
    brier: float
    ece: float


def _fit_and_score(
    model: BinaryClassifierBase,
    loss: LossBuilder,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    X_val: npt.NDArray[np.float64],
    y_val: npt.NDArray[np.int64],
    X_test: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.int64],
    *,
    lr: float,
    max_epochs: int,
    batch_size: int,
    patience: int | None,
    batch_seed: int,
) -> FitScore:
    """Train, then score the test rows at the validation threshold."""
    as_tensor = torch.from_numpy
    start = time.perf_counter()
    history = train_model(
        model,
        loss(),
        torch.optim.Adam(model.parameters(), lr=lr),
        as_tensor(X_train.astype(np.float32)),
        as_tensor(y_train.astype(np.float32)),
        as_tensor(X_val.astype(np.float32)),
        as_tensor(y_val.astype(np.float32)),
        max_epochs=max_epochs,
        batch_size=batch_size,
        monitor="mcc",
        patience=patience,
        generator=torch.Generator().manual_seed(batch_seed),
    )
    seconds = time.perf_counter() - start
    threshold = history.best_threshold if history.best_threshold is not None else 0.5
    prob = model.predict_proba(as_tensor(X_test.astype(np.float32)))
    mcc = matthews_corrcoef(y_test, (prob >= threshold).long())
    # A diverged model's NaN probabilities still score an MCC (every comparison
    # is False, so all negative); they have no calibration, and must not end
    # the run.
    finite = bool(torch.isfinite(prob).all())
    return FitScore(
        mcc=float(mcc),
        threshold=float(threshold),
        seconds=seconds,
        epochs=history.n_epochs,
        brier=brier_score(y_test, prob) if finite else math.nan,
        ece=expected_calibration_error(y_test, prob, ECE_BINS, "quantile") if finite else math.nan,
    )


#: Training settings a search space may vary.  The architecture is not
#: tunable: the control's size is matched to the hybrid's, and tuning one
#: model's architecture would break that match.
TUNABLE: frozenset[str] = frozenset({"lr", "batch_size", "max_epochs", "patience"})


@dataclass(frozen=True)
class Tuning:
    """
    The same random-search budget for the hybrid model and its control.

    Per outer fold, each model gets ``n_trials`` configurations drawn from its
    own search space (all of them if the space has fewer), and each is scored
    by the mean MCC over the same ``inner_folds`` stratified folds of the outer
    training part -- never the outer test rows.  The best configuration is
    then trained as usual.  The budget is counted in trials, not seconds, so
    the slower simulated model gets as many configurations as the control.

    Attributes
    ----------
    n_trials:
        Configurations per model and outer fold.
    search_spaces:
        ``{"hybrid": {...}, "control": {...}}``, each mapping a setting in
        :data:`TUNABLE` to the values to draw from.
    inner_folds:
        Stratified folds of the outer training part a configuration is scored
        on.  Default 3.
    """

    n_trials: int
    search_spaces: Mapping[str, Mapping[str, Sequence[Any]]]
    inner_folds: int = 3

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1; got {self.n_trials}.")
        if self.inner_folds < 2:
            raise ValueError(f"inner_folds must be >= 2; got {self.inner_folds}.")
        if set(self.search_spaces) != set(MODELS):
            raise ValueError(
                f"search_spaces needs exactly the keys {sorted(MODELS)}; "
                f"got {sorted(self.search_spaces)}."
            )
        for model, space in self.search_spaces.items():
            unknown = sorted(set(space) - TUNABLE)
            if unknown:
                raise ValueError(
                    f"{model}: {unknown} cannot be tuned; choose from {sorted(TUNABLE)}."
                )
            empty = sorted(name for name, values in space.items() if len(values) == 0)
            if empty:
                raise ValueError(f"{model}: no values to draw from for {empty}.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "inner_folds": self.inner_folds,
            "search_spaces": {
                m: {k: list(v) for k, v in sp.items()} for m, sp in self.search_spaces.items()
            },
        }


def _sample_configs(
    space: Mapping[str, Sequence[Any]], n_trials: int, seed: int
) -> list[dict[str, Any]]:
    """``n_trials`` distinct configurations from the grid of ``space`` (all if fewer)."""
    keys = sorted(space)
    grid = [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]
    if len(grid) <= n_trials:
        return grid
    chosen = np.random.default_rng(seed).choice(len(grid), size=n_trials, replace=False)
    return [grid[int(i)] for i in sorted(chosen)]


def _build(model_name: str, hybrid: HybridBuilder, n_features: int) -> BinaryClassifierBase:
    built = hybrid(n_features)
    model = classical_baseline(built) if model_name == "control" else built
    if not isinstance(model, BinaryClassifierBase):
        raise TypeError(f"hybrid must return a hybrid classifier; got {type(model).__name__}.")
    return model


def _evaluate_config(
    model_name: str,
    hybrid: HybridBuilder,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    fit_rows: npt.NDArray[np.intp],
    val_rows: npt.NDArray[np.intp],
    settings: Mapping[str, Any],
    *,
    loss: LossBuilder,
    seed: int,
    oversample: bool,
    smote_options: Mapping[str, Any],
) -> float:
    """
    MCC of one tuning trial on one inner fold: fitted on ``fit_rows``,
    scored on ``val_rows``.  The only place tuning touches data, and it gets
    row indices, so what it sees can be checked.  Early stopping and the
    threshold use ``val_rows`` too; that favours neither model.
    """
    X_fold = _standardise(X, fit_rows)
    if oversample:
        fold = oversample_fold(X_fold, y, fit_rows, val_rows, random_state=seed, **smote_options)
        X_fit, y_fit = fold.X_train, np.asarray(fold.y_train, dtype=np.int64)
    else:
        X_fit, y_fit = X_fold[fit_rows], y[fit_rows]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _build(model_name, hybrid, X.shape[1])
        mcc = _fit_and_score(
            model,
            loss,
            X_fit,
            y_fit,
            X_fold[val_rows],
            y[val_rows],
            X_fold[val_rows],
            y[val_rows],
            lr=settings["lr"],
            max_epochs=settings["max_epochs"],
            batch_size=settings["batch_size"],
            patience=settings["patience"],
            batch_seed=seed,
        ).mcc
    return mcc


def _tune(
    model_name: str,
    tuning: Tuning,
    hybrid: HybridBuilder,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    train_part: npt.NDArray[np.intp],
    defaults: Mapping[str, Any],
    *,
    loss: LossBuilder,
    seed: int,
    oversample: bool,
    smote_options: Mapping[str, Any],
) -> dict[str, Any]:
    """The configuration with the best mean inner-fold MCC (the first, on ties)."""
    inner = stratified_kfold(y[train_part], tuning.inner_folds, random_state=seed)
    best: tuple[float, dict[str, Any]] | None = None
    for config in _sample_configs(tuning.search_spaces[model_name], tuning.n_trials, seed):
        settings = {**defaults, **config}
        score = float(
            np.mean(
                [
                    _evaluate_config(
                        model_name,
                        hybrid,
                        X,
                        y,
                        train_part[fit],
                        train_part[val],
                        settings,
                        loss=loss,
                        seed=seed,
                        oversample=oversample,
                        smote_options=smote_options,
                    )
                    for fit, val in inner
                ]
            )
        )
        if best is None or score > best[0]:
            best = (score, config)
    assert best is not None
    return best[1]


def _noise_scores(
    model: BinaryClassifierBase,
    X_test: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.int64],
    threshold: float,
    levels: Sequence[float],
    position: Position,
) -> dict[float, float]:
    """Test MCC of the trained model under inference-time depolarising noise, per level."""
    x = torch.from_numpy(X_test.astype(np.float32))
    scores: dict[float, float] = {}
    for p in levels:
        with apply_depolarizing_noise(model, p, position=position):
            prob = model.predict_proba(x)
        scores[float(p)] = float(matthews_corrcoef(y_test, (prob >= threshold).long()))
    return scores


def _one_sided_better(hybrid: Sequence[float], control: Sequence[float]) -> tuple[float, float]:
    """(p, attainable minimum p) that the hybrid scores higher; NaN if every fold ties."""
    try:
        test = wilcoxon_signed_rank(hybrid, control, alternative="greater")
    except ValueError:
        return math.nan, math.nan
    return test.p_value, test.min_p_value


def _noise_comparison(
    dataset: str,
    folds: Sequence[FoldResult],
    n_splits: int,
    levels: Sequence[float],
    hybrid_noiseless: Sequence[float],
    control: Sequence[float],
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-level rows and the summary for one dataset (fold scores averaged over seeds)."""
    rows = []
    lost_at: float | None = None
    p0, _ = _one_sided_better(hybrid_noiseless, control)
    better_noiseless = bool(p0 < alpha)
    for level in sorted(float(p) for p in levels):
        per_fold = [
            float(
                np.mean(
                    [
                        f.noise_mcc[level]
                        for f in folds
                        if f.dataset == dataset and f.model == "hybrid" and f.fold == k
                    ]
                )
            )
            for k in range(n_splits)
        ]
        p_value, min_p = _one_sided_better(per_fold, control)
        rows.append(
            {
                "dataset": dataset,
                "noise_level": level,
                "hybrid_mcc_mean": float(np.mean(per_fold)),
                "control_mcc_mean": float(np.mean(control)),
                "hybrid_fold_mcc": tuple(per_fold),
                "p_hybrid_better": p_value,
                "min_p": min_p,
            }
        )
        if better_noiseless and lost_at is None and not p_value < alpha:
            lost_at = level
    return rows, {"better_noiseless": better_noiseless, "lost_at": lost_at}


def run_benchmark(
    datasets: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]],
    hybrid: HybridBuilder = default_hybrid,
    *,
    n_splits: int = 5,
    validation_folds: int = 5,
    oversample: bool = True,
    loss: LossBuilder = FocalLoss,
    lr: float = 0.01,
    max_epochs: int = 100,
    batch_size: int = 256,
    patience: int | None = 10,
    random_state: int = 0,
    smote_kwargs: Mapping[str, Any] | None = None,
    record_path: str | os.PathLike[str] | None = None,
    n_seeds: int = 1,
    tuning: Tuning | None = None,
    noise_levels: Sequence[float] | None = None,
    noise_position: Position = "all",
    alpha: float = 0.05,
) -> BenchmarkResult:
    """
    Compare ``hybrid`` with its matched classical control on every dataset.

    See the module docstring for what the two models share in each fold.

    Parameters
    ----------
    datasets:
        Name → ``(X, y)``, with ``y`` binary 0/1 and 1 the rare class, e.g.
        ``{"credit-card": load_credit_card_fraud()[:2]}``.
    hybrid:
        ``hybrid(n_input_features)`` returns a fresh, untrained
        ``HybridBinaryClassifier`` or ``ParallelHybridClassifier``; the control
        is ``classical_baseline`` of it.  Called once per fold.
    n_splits:
        Outer folds per dataset.  At least 2; see the module docstring for
        the smallest p-value a given number of folds allows.
    validation_folds:
        One ``validation_folds``-th of each training part is held out for
        validation.  Default 5, i.e. 20 %.
    oversample:
        SMOTE the training rows (``smote_kwargs`` are passed on).
    loss:
        Called once per model and fold for a fresh loss, default
        :class:`FocalLoss`.
    lr, max_epochs, batch_size, patience:
        Adam learning rate and the :func:`train_model` settings, the same for
        both models.
    random_state:
        Root seed.  Every fold split, SMOTE draw, initialisation and batch
        order derives from it, so a run is repeatable; the seeds used are in
        :attr:`BenchmarkResult.folds`.
    n_seeds:
        Train each model this many times per fold, with distinct recorded
        initialisation seeds (split, SMOTE and batch order stay the fold's).
        The fold's score is the mean over its seeds, so the paired test still
        pairs folds, and ``mcc_seed_std`` reports how much a model's score
        moves with the initialisation alone: per fold the sample standard
        deviation (``ddof=1``) of its seeds' MCCs, averaged over folds.
        Default 1: one seed, as before.  With more than one, ``hybrid`` must
        build its model with ``init_seed=None`` (a model's own seed would
        override the runner's and repeat the same weights), and a sampling one
        (``shots``) with ``seed=None`` too (a device seed would repeat the same
        shot noise); a ``ValueError`` is raised otherwise.
    noise_levels, noise_position:
        Also score each trained hybrid model on its test rows under
        depolarising noise of each probability (inserted as
        :func:`hqnn_forge.noise.apply_depolarizing_noise` does, at
        ``noise_position``), at the fold's threshold, and compare it with the
        noise-free control per level (:attr:`BenchmarkResult.noise`).  This
        is **inference-time** noise on a model trained without it: it asks
        whether an advantage measured in noiseless simulation survives on a
        noisy device, not what training under noise would give.
    alpha:
        Significance level for :attr:`BenchmarkResult.noise_summary`.
    tuning:
        Tune both models' training settings with the same budget in every
        outer fold before training them; see :class:`Tuning`.  Default: no
        tuning, the settings above for both.
    record_path:
        Also write an experiment record (config, seeds, fold indices,
        dependency versions, devices, metrics) there as JSON; see
        :func:`hqnn_forge.experiment.save_record`.

    Returns
    -------
    BenchmarkResult
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2; got {n_splits}.")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1; got {n_seeds}.")
    if validation_folds < 2:
        raise ValueError(f"validation_folds must be >= 2; got {validation_folds}.")
    if not datasets:
        raise ValueError("datasets is empty.")
    smote_options = dict(smote_kwargs or {})
    settings: dict[str, Any] = {
        "n_splits": n_splits,
        "validation_folds": validation_folds,
        "oversample": oversample,
        "loss": _import_path(loss),
        "lr": lr,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "patience": patience,
        "random_state": random_state,
        "smote_kwargs": smote_options,
        "hybrid_builder": _import_path(hybrid),
        "n_seeds": n_seeds,
        "tuning": None if tuning is None else tuning.as_dict(),
        "noise_levels": None if noise_levels is None else [float(p) for p in noise_levels],
        "noise_position": noise_position,
        "alpha": alpha,
    }
    if noise_levels is not None:
        validate_noise(0.0, noise_position, position_name="noise_position")
        for p in noise_levels:
            validate_noise(float(p), noise_position, p_name="noise level")
    noise_rows: list[dict[str, Any]] = []
    noise_summary: dict[str, dict[str, Any]] = {}

    records: list[dict[str, Any]] = []
    folds: list[FoldResult] = []
    models: dict[str, dict[str, dict[str, Any]]] = {}
    data_info: dict[str, dict[str, Any]] = {}
    for d, (name, (X_raw, y_raw)) in enumerate(datasets.items()):
        X = np.asarray(X_raw, dtype=np.float64)
        y = np.asarray(y_raw)
        if X.ndim != 2 or y.shape != (X.shape[0],):
            raise ValueError(
                f"{name}: X must be (n_samples, n_features) and y (n_samples,); "
                f"got {X.shape} and {y.shape}."
            )
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f"{name}: y must be binary 0/1.")
        y = y.astype(np.int64)
        if not np.isfinite(X).all():
            raise ValueError(f"{name}: X contains NaN or infinite values.")
        data_info[name] = {
            "n_samples": int(y.size),
            "n_features": int(X.shape[1]),
            "n_positives": int(y.sum()),
            "sha256": fingerprint(X, y),
        }

        root = np.random.SeedSequence([random_state, d])
        split_root, *fold_roots = root.spawn(n_splits + 1)
        split_seed = int(split_root.generate_state(1)[0])
        outer = stratified_kfold(y, n_splits, random_state=split_seed)
        scores: dict[str, list[float]] = {m: [] for m in MODELS}
        seed_stds: dict[str, list[float]] = {m: [] for m in MODELS}
        briers: dict[str, list[float]] = {m: [] for m in MODELS}
        eces: dict[str, list[float]] = {m: [] for m in MODELS}
        seconds: dict[str, float] = {m: 0.0 for m in MODELS}
        n_parameters: dict[str, int] = {}
        architecture: dict[str, str] = {}

        for k, ((train_part, test_idx), fold_root) in enumerate(zip(outer, fold_roots)):
            inner_seed, smote_seed, init_seed, batch_seed = _seeds(fold_root, 4)
            # The first seed is the one a single-seed run uses, so n_seeds=1
            # reproduces earlier results; the others come from a child stream.
            init_seeds = [init_seed]
            if n_seeds > 1:
                extra = np.random.SeedSequence([init_seed, 1])
                init_seeds += _seeds(extra, n_seeds - 1)
            inner_tr, inner_va = stratified_kfold(
                y[train_part], validation_folds, random_state=inner_seed
            )[0]
            train_idx, val_idx = train_part[inner_tr], train_part[inner_va]
            X_fold = _standardise(X, train_part)
            if oversample:
                fold = oversample_fold(
                    X_fold, y, train_idx, val_idx, random_state=smote_seed, **smote_options
                )
                X_train, y_train, n_synthetic = fold.X_train, fold.y_train, len(fold.sources)
            else:
                X_train, y_train, n_synthetic = X_fold[train_idx], y[train_idx], 0

            defaults = {
                "lr": lr,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "patience": patience,
            }
            chosen: dict[str, dict[str, Any]] = {m: {} for m in MODELS}
            if tuning is not None:
                (tune_seed,) = _seeds(fold_root, 1)
                for model_name in MODELS:
                    chosen[model_name] = _tune(
                        model_name,
                        tuning,
                        hybrid,
                        X,
                        y,
                        train_part,
                        defaults,
                        loss=loss,
                        seed=tune_seed,
                        oversample=oversample,
                        smote_options=smote_options,
                    )

            for model_name in MODELS:
                train_settings = {**defaults, **chosen[model_name]}
                seed_scores: list[float] = []
                seed_briers: list[float] = []
                seed_eces: list[float] = []
                for seed_index, seed in enumerate(init_seeds):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed)
                        model = _build(model_name, hybrid, X.shape[1])
                        if n_seeds > 1 and model.get_config().get("init_seed") is not None:
                            # The model's own seed overrides the runner's, so
                            # every repeat would start from the same weights.
                            raise ValueError(
                                f"n_seeds={n_seeds} needs a hybrid built with "
                                f"init_seed=None; got init_seed="
                                f"{model.get_config()['init_seed']}, which gives "
                                f"every seed the same initial weights."
                            )
                        config = model.get_config()
                        if (
                            n_seeds > 1
                            and config.get("seed") is not None
                            and config.get("shots") is not None
                        ):
                            # Likewise for a seeded sampling device: every
                            # repeat would replay the same shot noise.
                            raise ValueError(
                                f"n_seeds={n_seeds} needs a sampling hybrid built "
                                f"with seed=None; got seed={config['seed']}, which "
                                f"gives every seed the same shot noise."
                            )
                        fit = _fit_and_score(
                            model,
                            loss,
                            X_train,
                            np.asarray(y_train, dtype=np.int64),
                            X_fold[val_idx],
                            y[val_idx],
                            X_fold[test_idx],
                            y[test_idx],
                            lr=train_settings["lr"],
                            max_epochs=train_settings["max_epochs"],
                            batch_size=train_settings["batch_size"],
                            patience=train_settings["patience"],
                            batch_seed=batch_seed,
                        )
                        noisy: dict[float, float] = {}
                        if noise_levels is not None and model_name == "hybrid":
                            noisy = _noise_scores(
                                model,
                                X_fold[test_idx],
                                y[test_idx],
                                fit.threshold,
                                noise_levels,
                                noise_position,
                            )
                    n_parameters[model_name] = model.count_parameters()
                    models.setdefault(name, {})[model_name] = {
                        "class": type(model).__name__,
                        "config": model.get_config(),
                    }
                    architecture[model_name] = type(model).__name__
                    seed_scores.append(fit.mcc)
                    seed_briers.append(fit.brier)
                    seed_eces.append(fit.ece)
                    seconds[model_name] += fit.seconds
                    folds.append(
                        FoldResult(
                            dataset=name,
                            model=model_name,
                            fold=k,
                            train_idx=np.sort(train_idx),
                            val_idx=np.sort(val_idx),
                            test_idx=np.sort(test_idx),
                            n_synthetic=n_synthetic,
                            split_seed=split_seed,
                            inner_seed=inner_seed,
                            smote_seed=smote_seed,
                            init_seed=seed,
                            batch_seed=batch_seed,
                            threshold=fit.threshold,
                            mcc=fit.mcc,
                            train_seconds=fit.seconds,
                            epochs=fit.epochs,
                            device=_device_name(model),
                            seed_index=seed_index,
                            hyperparameters=dict(chosen[model_name]),
                            noise_mcc=noisy,
                            brier=fit.brier,
                            ece=fit.ece,
                        )
                    )
                scores[model_name].append(float(np.mean(seed_scores)))
                briers[model_name].extend(seed_briers)
                eces[model_name].extend(seed_eces)
                if n_seeds > 1:
                    seed_stds[model_name].append(float(np.std(seed_scores, ddof=1)))

        hybrid_scores, control_scores = scores["hybrid"], scores["control"]
        if noise_levels is not None:
            noise_rows_here, summary = _noise_comparison(
                name, folds, n_splits, noise_levels, hybrid_scores, control_scores, alpha
            )
            noise_rows.extend(noise_rows_here)
            noise_summary[name] = summary
        try:
            test = wilcoxon_signed_rank(hybrid_scores, control_scores)
            p, min_p = test.p_value, test.min_p_value
        except ValueError:  # every fold tied: the test is undefined
            p = min_p = math.nan
        effect = rank_biserial_correlation(hybrid_scores, control_scores)

        for model_name in MODELS:
            fold_scores = np.asarray(scores[model_name])
            mean = float(fold_scores.mean())
            records.append(
                {
                    "dataset": name,
                    "model": model_name,
                    "architecture": architecture[model_name],
                    "n_samples": int(y.size),
                    "n_positives": int(y.sum()),
                    "n_folds": n_splits,
                    "n_parameters": n_parameters[model_name],
                    "mcc_mean": mean,
                    "mcc_std": float(fold_scores.std(ddof=1)),
                    "n_seeds": n_seeds,
                    # Undefined with one seed: None, not NaN, so records compare equal.
                    "mcc_seed_std": (
                        float(np.mean(seed_stds[model_name])) if n_seeds > 1 else None
                    ),
                    "mcc_per_kparam": parameter_efficiency(n_parameters[model_name], mean),
                    # NaN when any fold diverged: a mean over the rest would
                    # flatter the model.
                    "brier_mean": float(np.mean(briers[model_name])),
                    "ece_mean": float(np.mean(eces[model_name])),
                    "fold_mcc": tuple(float(s) for s in fold_scores),
                    "train_seconds": seconds[model_name],
                    "wilcoxon_p": p,
                    "wilcoxon_min_p": min_p,
                    "rank_biserial": effect,
                }
            )
    result = BenchmarkResult(
        records, folds, settings, models, data_info, noise_rows, noise_summary
    )
    if record_path is not None:
        from hqnn_forge.experiment import save_record  # imports this module

        save_record(result, record_path)
    return result


def write_csv(records: Sequence[Mapping[str, Any]], path: str | os.PathLike[str]) -> None:
    """
    Write ``records`` to ``path`` with the columns in :data:`COLUMNS`;
    ``fold_mcc`` is written as its scores joined by ``;``.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for record in records:
            row = {column: record[column] for column in COLUMNS}
            row["fold_mcc"] = ";".join(repr(s) for s in record["fold_mcc"])
            writer.writerow(row)
