"""
hqnn_forge.training.trainer
===========================
Mini-batch training with per-epoch validation and early stopping.

The loop is deliberately small and makes no assumption about the model beyond
``model(x) -> logits`` of shape ``(batch,)`` or ``(batch, 1)`` (binary), or
``(batch, n_classes)`` (multiclass, with integer labels), so it works for the
hybrid classifiers and for any classical baseline alike.  Loss and optimiser
are passed in.

Monitoring
----------
``monitor`` selects what early stopping watches on the validation split:

* ``"val_loss"`` -- the mean validation loss, lower is better.
* ``"mcc"``, ``"f1"``, ``"balanced_accuracy"`` -- the metric at the threshold
  that maximises it on the validation probabilities
  (:func:`hqnn_forge.evaluation.find_optimal_threshold`), higher is better.
  This is the default (``"mcc"``) because on imbalanced data a fixed 0.5
  threshold makes the monitored value mostly a function of calibration.

The threshold found at the best epoch is recorded, so the operating point that
produced the best score travels with the history instead of being re-derived.

A multiclass model has no single threshold: the metric monitors score its
argmax labels with the multiclass version of the same metric
(:data:`hqnn_forge.evaluation.MULTICLASS_METRICS`: Gorodkin's ``R_K`` for
``"mcc"``, macro-F1 for ``"f1"``), and no threshold is recorded.
"""

from __future__ import annotations

import copy
import functools
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn

from hqnn_forge.evaluation import (
    METRICS,
    MULTICLASS_METRICS,
    TemperatureScaler,
    find_optimal_threshold,
)
from hqnn_forge.training.spsa import SPSA
from hqnn_forge.utils.modes import _modes, _restore, eval_mode, train_mode

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EpochRecord:
    """Statistics for one epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    val_score: float | None = None
    val_threshold: float | None = None


@dataclass
class TrainingHistory:
    """
    What ``train_model`` did.

    Attributes
    ----------
    epochs:
        One ``EpochRecord`` per completed epoch, in order.
    monitor:
        The monitored quantity.
    best_epoch:
        1-based epoch with the best monitored value, or ``None`` without
        validation data.
    best_value:
        The monitored value at ``best_epoch``.
    best_threshold:
        Decision threshold at ``best_epoch`` (metric monitors only).
    stopped_early:
        ``True`` if patience ran out before ``max_epochs``.
    temperature:
        The :class:`~hqnn_forge.evaluation.TemperatureScaler` temperature fitted
        on the validation split for the returned weights, or ``None`` without a
        validation split, for a multiclass model, or when it cannot be fitted
        (non-binary targets such as smoothed labels, one class, non-finite
        logits, or logits that separate the validation classes or rank them no
        better than chance, where the NLL has no finite optimum in ``T``).
        ``T > 1`` means the model is
        over-confident on held-out data; ``σ(logits / T)`` is the calibrated
        probability (#319).
    restored_best:
        ``True`` if the model's weights were rolled back to ``best_epoch``.
    """

    monitor: str
    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int | None = None
    best_value: float | None = None
    best_threshold: float | None = None
    stopped_early: bool = False
    restored_best: bool = False
    temperature: float | None = None

    @property
    def train_loss(self) -> list[float]:
        return [e.train_loss for e in self.epochs]

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``(batch,)`` logits of a binary model, or ``(batch, n_classes)`` of a multiclass one."""
    out = model(x)
    if out.ndim == 2 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    if out.ndim != 1 and not (out.ndim == 2 and out.shape[-1] >= 2):
        raise ValueError(
            f"model output must have shape (batch,) or (batch, 1) for a binary model, or "
            f"(batch, n_classes) for a multiclass one; got {tuple(out.shape)}."
        )
    return out


def _target(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Labels as the loss expects them: class indices for multiclass logits, else float."""
    return y.long() if logits.ndim == 2 else y.float()


def _check_class_labels(labels: dict[str, torch.Tensor], n_classes: int) -> None:
    """Raise unless every label of every split is a class index in ``[0, n_classes)``.

    Run once over the whole of ``y_train`` and ``y_val`` before the first
    optimiser step.  The losses would otherwise catch a bad label only in the
    batch holding it -- after earlier batches have stepped the model -- or not
    at all: ``.long()`` truncates a soft label to another class, and
    ``CrossEntropyLoss`` silently skips ``-100`` (its ``ignore_index``).
    """
    for name, y in labels.items():
        if y.is_floating_point() and not bool(
            torch.all(torch.isfinite(y) & (y == torch.trunc(y)))
        ):
            raise ValueError(
                f"a multiclass model needs integer class labels; {name} holds non-integer "
                f"values (soft or probabilistic targets are not supported)."
            )
        low, high = int(y.min()), int(y.max())
        if low < 0 or high >= n_classes:
            raise ValueError(
                f"a multiclass model with {n_classes} outputs needs class labels in "
                f"[0, {n_classes - 1}]; {name} holds labels in [{low}, {high}]."
            )


def _check_pair(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X_{name} and y_{name} differ in length: {x.shape[0]} vs {y.shape[0]}.")
    if x.shape[0] == 0:
        raise ValueError(f"X_{name} is empty.")


def train_model(
    model: nn.Module,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    *,
    max_epochs: int = 100,
    batch_size: int = 256,
    monitor: str = "mcc",
    patience: int | None = 10,
    min_delta: float = 0.0,
    restore_best: bool = True,
    generator: torch.Generator | None = None,
    on_epoch_end: Callable[[EpochRecord], None] | None = None,
) -> TrainingHistory:
    """
    Train ``model`` with mini-batches, validating and early-stopping per epoch.

    Parameters
    ----------
    model:
        Any module mapping ``(batch, n_features)`` to logits of shape
        ``(batch,)`` or ``(batch, 1)`` (binary), or ``(batch, n_classes)``
        (multiclass).
    loss_fn:
        ``loss_fn(logits, targets) -> scalar``, e.g. ``FocalLoss()`` or
        ``nn.BCEWithLogitsLoss()``.  Targets are passed as float for a binary
        model and as integer class indices for a multiclass one.  Mean
        reduction is assumed for the reported ``train_loss``, which averages
        the batch losses weighted by batch size; with ``reduction="sum"``
        training is unaffected but ``train_loss`` is comparable neither
        across batch sizes nor with the full-batch ``val_loss``.
    optimizer:
        Optimiser already bound to the parameters to train.  A gradient-free
        one (``optimizer.gradient_free``, e.g.
        :class:`~hqnn_forge.training.SPSA`) gets ``step(closure)`` with a
        closure that returns the batch loss tensor, and no ``backward`` pass
        of its own (SPSA's ``gradient_optimizer`` backpropagates that loss to
        the classical head only).  An SPSA built without ``model=`` is given
        ``model``, so its two evaluations share the devices' shot noise.
    X_train, y_train:
        Training split.  For a multiclass model every label of ``y_train``
        and ``y_val`` must be a class index in ``[0, n_classes)`` (integer,
        or a whole-valued float); anything else raises ``ValueError`` before
        the first optimiser step.  A binary model's labels are passed to the
        loss as float, unchecked, as before.
    X_val, y_val:
        Validation split.  Without it the loop runs ``max_epochs`` epochs and
        ``monitor``, ``patience`` and ``restore_best`` have no effect.
    max_epochs:
        Upper bound on the number of epochs.  Default: 100.
    batch_size:
        Mini-batch size.  The last batch may be smaller, but for
        ``batch_size > 1`` never a single sample unless the training set is
        one: a remainder of one is merged into the batch before it, which then
        holds ``batch_size + 1``, because batch norm in train mode fails on one
        sample.  With ``batch_size=1`` every batch is one sample, as asked.
        Default: 256.
    monitor:
        ``"mcc"`` (default), ``"f1"``, ``"balanced_accuracy"`` or ``"val_loss"``.
    patience:
        Stop after this many consecutive epochs without an improvement larger
        than ``min_delta``.  ``None`` disables early stopping.  Default: 10.
    min_delta:
        Minimum change that counts as an improvement.  Default: 0.0.
    restore_best:
        Load the weights of the best epoch before returning.  Default: True.
    generator:
        Generator for the per-epoch shuffle, for reproducible batch order.
    on_epoch_end:
        Called with each ``EpochRecord``, e.g. for logging.

    Returns
    -------
    TrainingHistory

    Notes
    -----
    Training runs under :func:`~hqnn_forge.utils.modes.train_mode`: a
    submodule the caller put in eval mode -- a frozen batch-norm layer, say --
    stays in eval mode, with its statistics untouched, unless the whole model
    arrived in eval mode, which is then trained in train mode throughout.
    Validation runs in eval mode through ``eval_mode``.  Every epoch starts
    from the modes training began with, so an ``on_epoch_end`` callback that
    puts the model in eval mode does not carry over into the next epoch.  On
    return every submodule has the mode it had on entry.
    """
    if monitor != "val_loss" and monitor not in METRICS:
        raise ValueError(
            f"unknown monitor {monitor!r}; choose 'val_loss' or one of {sorted(METRICS)}."
        )
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1; got {max_epochs}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}.")
    if patience is not None and patience < 1:
        raise ValueError(f"patience must be >= 1 or None; got {patience}.")
    if (X_val is None) != (y_val is None):
        raise ValueError("pass both X_val and y_val, or neither.")

    X_train = torch.as_tensor(X_train)
    # Cast per loss call (_target): float for a binary model's BCE-style loss,
    # class indices for a multiclass model's cross-entropy.
    y_train = torch.as_tensor(y_train).reshape(-1)
    _check_pair(X_train, y_train, "train")
    val: tuple[torch.Tensor, torch.Tensor] | None = None
    if X_val is not None and y_val is not None:
        val = (torch.as_tensor(X_val), torch.as_tensor(y_val).reshape(-1))
        _check_pair(*val, "val")
        # A single-class split scores the same degenerate value at every
        # threshold, so epoch 1 wins, patience expires and restore_best hands
        # back the initial weights -- silently, on data the model does learn.
        if monitor != "val_loss" and torch.unique(val[1]).numel() < 2:
            raise ValueError(
                f"y_val contains a single class, so the {monitor!r} monitor cannot rank "
                f"epochs on it.  Pass a validation split holding both classes (e.g. a "
                f"stratified one), or monitor='val_loss'."
            )
    has_val = val is not None
    # Multiclass labels are checked once, over both splits, at the first
    # multiclass logits (the number of classes is known only then) -- before
    # any optimiser step touches the caller's model.
    unchecked_labels: dict[str, torch.Tensor] | None = {"y_train": y_train} | (
        {"y_val": val[1]} if val is not None else {}
    )

    lower_is_better = monitor == "val_loss"
    history = TrainingHistory(monitor=monitor)
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    n = X_train.shape[0]

    # Batch boundaries; a trailing batch of one sample is merged into the one
    # before it, unless every batch is meant to be one (see batch_size above).
    bounds = [*range(0, n, batch_size), n]
    if batch_size > 1 and len(bounds) > 2 and bounds[-1] - bounds[-2] == 1:
        del bounds[-2]

    # SPSA synchronises the two evaluations' shot noise only through the model
    # (its model=); the model trained here is the one the closure evaluates.
    if isinstance(optimizer, SPSA) and optimizer.model is None:
        optimizer.model = model

    def train_loss(rows: torch.Tensor) -> torch.Tensor:
        """The loss on the training rows ``rows``, with its autograd graph."""
        # Labels are checked at the first evaluation, which precedes the first
        # parameter update on either path.
        nonlocal unchecked_labels
        logits = _logits(model, X_train[rows])
        if unchecked_labels is not None and logits.ndim == 2:
            _check_class_labels(unchecked_labels, logits.shape[-1])
            unchecked_labels = None
        return loss_fn(logits, _target(logits, y_train[rows]))

    # The caller's per-submodule modes, not model.train(), which recurses and
    # would unfreeze a submodule the caller put in eval mode (#174).
    with train_mode(model):
        # Validation restores its own modes, but a callback need not, so every
        # epoch starts from these, as the per-epoch model.train() used to.
        epoch_modes = _modes(model)
        for epoch in range(1, max_epochs + 1):
            _restore(epoch_modes)
            # ── train ────────────────────────────────────────────────────────
            perm = torch.randperm(n, generator=generator)
            total, seen = 0.0, 0
            for start, stop in itertools.pairwise(bounds):
                idx = perm[start:stop]
                if getattr(optimizer, "gradient_free", False):
                    # SPSA and the like evaluate the loss themselves, twice, with
                    # no backward pass of their own (see hqnn_forge.training.spsa).
                    # The tensor, not a float: SPSA's gradient_optimizer
                    # backpropagates it to the classical head.  torch types the
                    # closure as returning float, but its own optimisers (LBFGS)
                    # take one returning the loss tensor.
                    closure = functools.partial(train_loss, idx)
                    batch_loss = float(optimizer.step(cast("Callable[[], float]", closure)))
                else:
                    optimizer.zero_grad()
                    loss = train_loss(idx)
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item()
                total += batch_loss * idx.numel()
                seen += idx.numel()
            record = EpochRecord(epoch=epoch, train_loss=total / seen)

            # ── validate ─────────────────────────────────────────────────────
            if val is not None:
                x_v, y_v = val
                with torch.no_grad(), eval_mode(model):
                    val_logits = _logits(model, x_v)
                    val_loss = float(loss_fn(val_logits, _target(val_logits, y_v)))
                if lower_is_better:
                    value, threshold = val_loss, None
                elif val_logits.ndim == 2:
                    # Multiclass: no threshold to search; score the argmax labels.
                    if torch.any(torch.isnan(val_logits)):
                        value, threshold = math.nan, None
                    else:
                        value = MULTICLASS_METRICS[monitor](y_v.long(), val_logits.argmax(dim=-1))
                        threshold = None
                else:
                    val_prob = torch.sigmoid(val_logits)
                    if torch.any(torch.isnan(val_prob)):
                        # find_optimal_threshold rejects NaN probabilities rather
                        # than label them negative.  Diverging must not take the
                        # history and the best-epoch snapshot down with it, so
                        # score the epoch NaN, as the val_loss path already does:
                        # it never improves, and patience ends the run.
                        value, threshold = math.nan, None
                    else:
                        search = find_optimal_threshold(y_v.long(), val_prob, metric=monitor)
                        value, threshold = search.score, search.threshold
                record = EpochRecord(
                    epoch=epoch,
                    train_loss=record.train_loss,
                    val_loss=val_loss,
                    val_score=None if lower_is_better else value,
                    val_threshold=threshold,
                )

                if history.best_value is None or math.isnan(history.best_value):
                    improved = not math.isnan(value)
                elif lower_is_better:
                    improved = value < history.best_value - min_delta
                else:
                    improved = value > history.best_value + min_delta

                if improved:
                    history.best_epoch, history.best_value = epoch, value
                    history.best_threshold = threshold
                    epochs_without_improvement = 0
                    if restore_best:
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    epochs_without_improvement += 1

            history.epochs.append(record)
            if on_epoch_end is not None:
                on_epoch_end(record)

            if has_val and patience is not None and epochs_without_improvement >= patience:
                history.stopped_early = epoch < max_epochs
                break

        if restore_best and best_state is not None and history.best_epoch != history.n_epochs:
            model.load_state_dict(best_state)
            history.restored_best = True
        if val is not None:
            history.temperature = _validation_temperature(model, val)
    return history


def _validation_temperature(
    model: nn.Module, val: tuple[torch.Tensor, torch.Tensor]
) -> float | None:
    """The temperature fitted on the validation logits of the final weights, if it can be."""
    x_v, y_v = val
    with torch.no_grad(), eval_mode(model):
        logits = _logits(model, x_v)
    if logits.ndim != 1:
        return None
    try:
        return TemperatureScaler.fit(logits, y_v).temperature
    except ValueError:
        # Soft targets, one class, non-finite logits or a split the logits
        # separate (no finite temperature) all train fine; they just have no
        # temperature to report, so the finished run must not fail here.
        return None
