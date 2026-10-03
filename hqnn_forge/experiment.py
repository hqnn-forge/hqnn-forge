"""
hqnn_forge.experiment
=====================
Experiment records: what is needed to check or repeat a benchmark result,
written as JSON beside it.

A checkpoint (:mod:`hqnn_forge.utils.checkpoint`) is enough to reload a model,
not to reproduce a result.  A record written by :func:`save_record` (or by
``run_benchmark(..., record_path=...)``) holds:

* **config** -- every ``run_benchmark`` setting, and per dataset the class and
  ``get_config()`` of the hybrid model and of its control;
* **seeds** -- the root ``random_state`` and, per fold, every seed derived
  from it: the outer split, the train/validation split, SMOTE, initialisation
  and batch order.  A builder that fixes its own ``init_seed`` overrides the
  per-fold initialisation seed; that ``init_seed`` is then in the model config;
* **folds** -- the train, validation and test row indices of every fold, and
  the threshold, MCC, epochs and seconds per model;
* **environment** -- Python, platform, ``hqnn_forge`` and the versions of
  PennyLane, pennylane-lightning, torch, NumPy, scikit-learn and SciPy, and
  per fold the PennyLane device the circuit really ran on, since the device
  factory falls back silently apart from a warning;
* **datasets** -- size, positives and a SHA-256 of the data, not the data;
* **metrics** -- the per-dataset, per-model rows of the result.

:func:`load_record` reads one back and reports which recorded versions differ
from the running environment.  :func:`rerun_benchmark` repeats the run from a
record on the same data: on CPU it reproduces the per-fold scores exactly.

The file is strict JSON: NaN (an undefined Wilcoxon p-value) is written as
``null``.
"""

from __future__ import annotations

import datetime
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

import hqnn_forge

if TYPE_CHECKING:
    from hqnn_forge.benchmark import BenchmarkResult

#: Bumped whenever the layout below changes incompatibly.
RECORD_FORMAT_VERSION = 1

#: Distributions whose installed version is recorded.  ``None`` means absent.
RECORDED_PACKAGES: tuple[str, ...] = (
    "pennylane",
    "pennylane-lightning",
    "torch",
    "numpy",
    "scikit-learn",
    "scipy",
)


def environment() -> dict[str, Any]:
    """Python, platform, ``hqnn_forge`` and :data:`RECORDED_PACKAGES` versions."""
    packages: dict[str, str | None] = {}
    for name in RECORDED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "hqnn_forge": hqnn_forge.__version__,
        "packages": packages,
    }


def _plain(value: Any) -> Any:
    """``value`` as JSON-compatible data: NaN → None, NumPy → Python, tuples → lists."""
    if isinstance(value, float | np.floating):
        return None if math.isnan(value) else float(value)
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, str):
        return value
    raise TypeError(
        f"cannot record a {type(value).__name__} ({value!r}) as JSON; experiment records "
        f"hold plain data only."
    )


def to_record(result: BenchmarkResult) -> dict[str, Any]:
    """The JSON-ready experiment record of ``result`` in the current environment."""
    folds = [
        {
            "dataset": f.dataset,
            "model": f.model,
            "fold": f.fold,
            "train_idx": f.train_idx,
            "val_idx": f.val_idx,
            "test_idx": f.test_idx,
            "n_synthetic": f.n_synthetic,
            "split_seed": f.split_seed,
            "inner_seed": f.inner_seed,
            "smote_seed": f.smote_seed,
            "init_seed": f.init_seed,
            "seed_index": f.seed_index,
            "batch_seed": f.batch_seed,
            "device": f.device,
            "hyperparameters": f.hyperparameters,
            "noise_mcc": {repr(level): mcc for level, mcc in f.noise_mcc.items()},
            "threshold": f.threshold,
            "mcc": f.mcc,
            "brier": f.brier,
            "ece": f.ece,
            "epochs": f.epochs,
            "train_seconds": f.train_seconds,
        }
        for f in result.folds
    ]
    return _plain(
        {
            "format_version": RECORD_FORMAT_VERSION,
            "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "environment": environment(),
            "config": {"settings": result.settings, "models": result.models},
            "seeds": {
                "random_state": result.settings["random_state"],
                "folds": [
                    {
                        "dataset": f.dataset,
                        "model": f.model,
                        "fold": f.fold,
                        "seed_index": f.seed_index,
                        "split_seed": f.split_seed,
                        "inner_seed": f.inner_seed,
                        "smote_seed": f.smote_seed,
                        "init_seed": f.init_seed,
                        "batch_seed": f.batch_seed,
                    }
                    for f in result.folds
                ],
            },
            "datasets": result.datasets,
            "folds": folds,
            "metrics": result.records,
            "noise": result.noise,
            "noise_summary": result.noise_summary,
        }
    )


def save_record(result: BenchmarkResult, path: str | os.PathLike[str]) -> None:
    """
    Write ``result``'s experiment record to ``path`` as JSON.

    Raises
    ------
    TypeError
        If a model config holds something that is not plain data, e.g. a
        custom encoder module; such a run cannot be recorded.
    """
    record = to_record(result)
    text = json.dumps(record, indent=1, allow_nan=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def compare_environment(recorded: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """
    ``{name: (recorded, current)}`` for every recorded version that differs
    from the running environment: ``python``, ``hqnn_forge`` and each package.
    The platform string is informational and not compared.
    """
    current = environment()
    differences: dict[str, tuple[Any, Any]] = {}
    for key in ("python", "implementation", "hqnn_forge"):
        if recorded.get(key) != current[key]:
            differences[key] = (recorded.get(key), current[key])
    recorded_packages = recorded.get("packages", {})
    for name in sorted(set(recorded_packages) | set(current["packages"])):
        old, new = recorded_packages.get(name), current["packages"].get(name)
        if old != new:
            differences[name] = (old, new)
    return differences


def load_record(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    """
    Read a record written by :func:`save_record`.

    Returns
    -------
    (record, differences)
        The record as a dict, and :func:`compare_environment` of its
        environment: empty when every recorded version matches.

    Raises
    ------
    ValueError
        If the file is not an experiment record of a known format version.
    """
    with open(path, encoding="utf-8") as fh:
        record = json.load(fh)
    if not isinstance(record, dict) or "format_version" not in record:
        raise ValueError(f"{path} is not an experiment record.")
    if record["format_version"] != RECORD_FORMAT_VERSION:
        raise ValueError(
            f"{path} has record format {record['format_version']}; this version of "
            f"hqnn_forge reads format {RECORD_FORMAT_VERSION}."
        )
    return record, compare_environment(record["environment"])


def _resolve(path: str) -> Any:
    """Import ``module.qualname`` and return the object."""
    module_name, _, qualname = path.rpartition(".")
    while module_name:
        try:
            obj: Any = importlib.import_module(module_name)
        except ImportError:
            module_name, _, head = module_name.rpartition(".")
            qualname = f"{head}.{qualname}"
            continue
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    raise ImportError(path)


def rerun_benchmark(
    record: Mapping[str, Any],
    datasets: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    loss: Any = None,
) -> BenchmarkResult:
    """
    Repeat the run described by ``record`` on the same ``datasets``.

    The hybrid model is rebuilt from the recorded class and config (the
    builder function itself is not recorded), with every recorded setting and
    the root seed, so the folds, SMOTE draws, initialisations and batch
    orders are the ones recorded.  On CPU the per-fold scores then match
    exactly.

    Parameters
    ----------
    record:
        As returned by :func:`load_record`.
    datasets:
        The same datasets under the same names and in the same order; each is
        checked against its recorded SHA-256.
    loss:
        The loss builder, if the recorded import path cannot be imported
        (a loss defined inside a function or a script).

    Raises
    ------
    ValueError
        If a dataset is missing, extra, reordered or differs from the recorded
        one, or if the rebuilt models' configs differ from the recorded ones.
    """
    from hqnn_forge import models as model_module
    from hqnn_forge.benchmark import Tuning, fingerprint, run_benchmark

    recorded_data = record["datasets"]
    if list(datasets) != list(recorded_data):
        raise ValueError(
            f"record has datasets {list(recorded_data)}, in that order; got {list(datasets)}."
        )
    for name, (X, y) in datasets.items():
        digest = fingerprint(np.asarray(X, dtype=np.float64), np.asarray(y).astype(np.int64))
        if digest != recorded_data[name]["sha256"]:
            raise ValueError(f"dataset {name!r} differs from the recorded one (SHA-256).")

    settings = dict(record["config"]["settings"])
    recorded_models = record["config"]["models"]
    first = next(iter(recorded_models.values()))["hybrid"]
    cls = getattr(model_module, first["class"])
    base_config = {k: v for k, v in first["config"].items() if k != "n_input_features"}

    def hybrid(n_input_features: int) -> Any:
        return cls(n_input_features=n_input_features, **base_config)

    if loss is None:
        try:
            loss = _resolve(settings["loss"])
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"the recorded loss {settings['loss']!r} cannot be imported; pass loss=."
            ) from exc

    result = run_benchmark(
        datasets,
        hybrid,
        n_splits=settings["n_splits"],
        validation_folds=settings["validation_folds"],
        oversample=settings["oversample"],
        loss=loss,
        lr=settings["lr"],
        max_epochs=settings["max_epochs"],
        batch_size=settings["batch_size"],
        patience=settings["patience"],
        random_state=settings["random_state"],
        smote_kwargs=settings["smote_kwargs"],
        n_seeds=settings.get("n_seeds", 1),
        tuning=None if settings.get("tuning") is None else Tuning(**settings["tuning"]),
        noise_levels=settings.get("noise_levels"),
        noise_position=settings.get("noise_position", "all"),
        alpha=settings.get("alpha", 0.05),
    )
    rebuilt = _plain(result.models)
    if rebuilt != recorded_models:
        raise ValueError(
            "the rebuilt models' configs differ from the recorded ones; the hybrid "
            "builder of the original run did not depend on the input width alone."
        )
    return result
