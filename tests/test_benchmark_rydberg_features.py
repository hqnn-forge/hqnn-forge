"""
tests/test_benchmark_rydberg_features.py
========================================
``examples/benchmark_rydberg_features.py`` (#501): the runner that compares
Rydberg features with and without interactions against classical baselines.

What the numbers are checked against:

* the model page's closed forms, typed in: the spacing table of
  ``docs/rydberg-model.md``, the single-atom feature ``f(√(1 + 3 σ(x)²))``
  and the damped Rabi oscillation after the pulse;
* a single-atom map, for arm B under dephasing;
* a fold rebuilt by hand from the library's own functions
  (``stratified_kfold``, ``PCANormalizer``, ``smote``, ``train_model``), not
  through the script, for the recorded indices, seeds, scores and separation
  measures of a ``--quick`` run;
* a circuit written out gate by gate in PennyLane, for arm D;
* hand computations, for the positive control's labels, the parameter counts
  and the JSON mapping;
* ``hashlib`` over files listed with ``os.walk``, for the code digest in the
  feature cache's key (#522), and ``compare_environment`` for which entries
  of the environment the key holds;
* a matrix standardised by hand, scikit-learn's ``StandardScaler`` and the
  standardisation of ``g`` for features ``½ + ε g``, for
  ``standardise_features`` (#518).
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import lzma
import math
import os
import re
import socket
import struct
import sys
import warnings
import zipfile
import zlib
from concurrent.futures import Future
from pathlib import Path
from threading import Thread
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pennylane as qml
import pytest
import torch
import torch.nn as nn

import hqnn_forge
from hqnn_forge import benchmark, experiment
from hqnn_forge.benchmark import fingerprint, sample_configs
from hqnn_forge.data import load_iranian_churn
from hqnn_forge.diagnostics import separation_measures
from hqnn_forge.evaluation import matthews_corrcoef, pr_auc
from hqnn_forge.experiment import (
    COMPARED_VERSIONS,
    RECORDED_PACKAGES,
    compare_environment,
    environment,
)
from hqnn_forge.models import ClassicalBaseline, HybridBinaryClassifier, LinearClassifier
from hqnn_forge.preprocessing import PCANormalizer, smote, stratified_kfold
from hqnn_forge.rydberg import (
    AQUILA_LIMITS,
    DEFAULT_C6,
    AtomRegister,
    GlobalDrivePulseEncoding,
    PulseEncoding,
    RydbergFeatureMap,
)
from hqnn_forge.training import train_model

ROOT = Path(__file__).resolve().parent.parent
OMEGA = 4.0 * math.pi  # the page's reference Rabi frequency, rad/µs


def _script() -> ModuleType:
    path = ROOT / "examples" / "benchmark_rydberg_features.py"
    spec = importlib.util.spec_from_file_location("benchmark_rydberg_features", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


bench = _script()
QUICK = bench.QUICK
#: Arms that need no PennyLane circuit, for runs that only check the plumbing.
FAST = ("A", "B", "C")


def _replace(settings: Any, **changes: Any) -> Any:
    import dataclasses

    return dataclasses.replace(settings, **changes)


# ---------------------------------------------------------------------------
# A fold rebuilt by hand, from the library and not from the script
# ---------------------------------------------------------------------------


def _draw(root: np.random.SeedSequence, n: int) -> list[int]:
    return [int(child.generate_state(1)[0]) for child in root.spawn(n)]


def _fold_by_hand(X: np.ndarray, y: np.ndarray, settings: Any, dataset_index: int) -> Any:
    """Fold 0 of ``settings``: seeds, row indices, inputs and the oversampled training rows."""
    root = np.random.SeedSequence([settings.random_state, dataset_index])
    split_root, *fold_roots = root.spawn(settings.n_splits + 1)
    split_seed = int(split_root.generate_state(1)[0])
    inner_seed, smote_seed, init_seed, batch_seed, tune_seed = _draw(fold_roots[0], 5)
    train_part, test_idx = stratified_kfold(y, settings.n_splits, random_state=split_seed)[0]
    tr, va = stratified_kfold(y[train_part], settings.validation_folds, random_state=inner_seed)[0]
    train_idx, val_idx = train_part[tr], train_part[va]

    # Step 2's constant rule is that of ``standardise_features`` (#580),
    # tested on its own; the run's inputs are compared exactly, so the same
    # rounding is needed here.
    scaled = bench.standardise_features(X, train_part)
    pca = PCANormalizer(n_components=settings.n_atoms, scale_to_pi=True).fit(scaled[train_part])
    inputs = pca.transform(scaled).double().numpy()
    oversampled = smote(inputs[train_idx], y[train_idx], random_state=smote_seed)
    return SimpleNamespace(
        seeds=dict(
            split_seed=split_seed,
            inner_seed=inner_seed,
            smote_seed=smote_seed,
            init_seed=init_seed,
            batch_seed=batch_seed,
            tune_seed=tune_seed,
        ),
        train_part=train_part,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        inputs=inputs,
        train_inputs=oversampled.X,
        train_labels=np.asarray(oversampled.y, dtype=np.int64),
        n_synthetic=len(oversampled.sources),
    )


def _map_by_hand(n_atoms: int, v: float, gamma_ratio: float, n_steps: int, **kwargs: Any) -> Any:
    """The page's map at ``(V/Ω, γ/Ω)``: ``a = (C6/Ω)^(1/6) (V/Ω)^(−1/6)``, ``ΩT = π``."""
    a = (DEFAULT_C6 / OMEGA) ** (1 / 6) * v ** (-1 / 6)
    return RydbergFeatureMap(
        AtomRegister.chain(n_atoms, a),
        PulseEncoding(n_atoms, omega=OMEGA),
        c6=DEFAULT_C6,
        gamma=gamma_ratio * OMEGA,
        n_steps=n_steps if gamma_ratio > 0 else None,
        **kwargs,
    )


def _circuit_by_hand(X: np.ndarray, circuit_seed: int, n_layers: int = 2) -> np.ndarray:
    """
    Arm D's features, gate by gate: ``RX(x_i)`` on qubit ``i``, then per layer
    a CNOT ring ``i → i+1`` and ``Rot`` on every qubit, then ``⟨Z_i⟩``.  The
    angles are ``U(−π, π)`` draws of ``circuit_seed``, held in float32.
    """
    n = X.shape[1]
    draw = torch.rand(
        (n_layers, n, 3),
        generator=torch.Generator().manual_seed(circuit_seed),
        dtype=torch.float64,
    )
    weights = ((2 * draw - 1) * math.pi).to(torch.float32).double().numpy()

    @qml.qnode(qml.device("default.qubit", wires=n))
    def circuit(x: np.ndarray) -> Any:
        for i in range(n):
            qml.RX(x[i], wires=i)
        for layer in range(n_layers):
            for i in range(n):
                qml.CNOT(wires=[i, (i + 1) % n])
            for i in range(n):
                qml.Rot(*weights[layer, i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return np.array([[float(v) for v in circuit(row)] for row in X])


def _single_atom_feature(x: np.ndarray) -> np.ndarray:
    """``f(√(1 + 3 σ(x)²))`` with ``f(s) = sin²(π s/2)/s²``: the page's closed form."""
    s = np.sqrt(1.0 + 3.0 / (1.0 + np.exp(-x)) ** 2)
    return np.asarray(np.sin(math.pi * s / 2) ** 2 / s**2)


@pytest.fixture(scope="module")
def quick(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One ``--quick`` run through the command line, and its fold rebuilt by hand."""
    directory = tmp_path_factory.mktemp("quick")
    out = directory / "quick.jsonl"
    records = bench.main(["--quick", "--out", str(out), "--cache-dir", str(directory / "cache")])
    X, y, data_seed = bench.load_dataset("positive-control", QUICK)
    return SimpleNamespace(
        records=records,
        lines=bench.read_records(out),
        text=out.read_text(encoding="utf-8"),
        cache=directory / "cache",
        X=X,
        y=y,
        data_seed=data_seed,
        fold=_fold_by_hand(X, y, QUICK, 0),
    )


def _line(run: Any, arm: str, head: str) -> dict[str, Any]:
    (found,) = [r for r in run.lines if r["arm"] == arm and r["head"] == head]
    return dict(found)


def _standardised_by_hand(raw: np.ndarray, fold: Any) -> np.ndarray:
    """
    ``raw`` (the fold's rows stacked: training with synthetic, validation,
    test) with every column centred and divided by its mean and population
    standard deviation over the real training rows.  ``smote`` puts those
    first, in their order, so no synthetic, validation or test row enters.
    """
    assert fold.train_inputs.shape[0] == fold.train_idx.size + fold.n_synthetic
    fitting = raw[: fold.train_idx.size]
    return np.asarray((raw - fitting.mean(axis=0)) / fitting.std(axis=0))


def _assert_scores_by_hand(
    record: dict[str, Any], fold: Any, y: np.ndarray, features: np.ndarray, head: str
) -> None:
    """
    MCC, PR-AUC, threshold and epochs of ``record``, from a head built and
    trained here on ``features`` (the fold's rows stacked as in
    :func:`_standardised_by_hand`) with the recorded hyperparameters and the
    fold's seeds, through ``train_model`` and not through the script.
    """
    x = torch.from_numpy(features.astype(np.float32))
    n_train, n_val = fold.train_inputs.shape[0], fold.val_idx.size
    x_train, x_val, x_test = x.split([n_train, n_val, fold.test_idx.size])
    seed = fold.seeds["init_seed"]
    d = features.shape[1]
    model: nn.Module = (
        LinearClassifier(d, init_seed=seed)
        if head == "linear"
        else ClassicalBaseline(d, [QUICK.hidden], init_seed=seed)
    )
    settings = record["hyperparameters"]
    history = train_model(
        model,
        nn.BCEWithLogitsLoss(),
        torch.optim.Adam(model.parameters(), lr=settings["lr"]),
        x_train,
        torch.from_numpy(fold.train_labels.astype(np.float32)),
        x_val,
        torch.from_numpy(y[fold.val_idx].astype(np.float32)),
        max_epochs=settings["max_epochs"],
        batch_size=settings["batch_size"],
        monitor="mcc",
        patience=settings["patience"],
        generator=torch.Generator().manual_seed(fold.seeds["batch_seed"]),
    )
    assert isinstance(model, LinearClassifier | ClassicalBaseline)
    probability = model.predict_proba(x_test)
    truth = y[fold.test_idx]
    assert history.best_threshold is not None
    assert record["threshold"] == pytest.approx(history.best_threshold, abs=1e-6)
    prediction = (probability >= history.best_threshold).long()
    assert record["mcc"] == pytest.approx(matthews_corrcoef(truth, prediction), abs=1e-9)
    assert record["pr_auc"] == pytest.approx(pr_auc(truth, probability), abs=1e-6)
    assert record["epochs"] == history.n_epochs
    assert 0.0 <= record["pr_auc"] <= 1.0


# ---------------------------------------------------------------------------
# The grid: spacing, dephasing rate, pulse
# ---------------------------------------------------------------------------


class TestGrid:
    @pytest.mark.parametrize(
        ("v", "a"),
        [(0.1, 12.76), (0.3, 10.62), (1.0, 8.69), (3.0, 7.24), (10.0, 5.92), (30.0, 4.93)],
    )
    def test_spacing_is_the_table_of_the_model_page(self, v: float, a: float) -> None:
        assert bench.spacing(v, OMEGA, DEFAULT_C6) == pytest.approx(a, abs=0.006)

    @pytest.mark.parametrize("v", [0.1, 1.0, 7.5, 100.0])
    def test_spacing_gives_the_nearest_neighbour_interaction_asked_for(self, v: float) -> None:
        """``C6/a⁶ = (V/Ω) Ω``, and the next-nearest coupling is ``1/64`` of it."""
        settings = bench.Settings(n_atoms=4)
        fmap = bench.feature_map(settings, v, 0.0, interactions=True)
        a = bench.spacing(v, settings.omega, settings.c6)
        assert a == pytest.approx((DEFAULT_C6 / OMEGA) ** (1 / 6) * v ** (-1 / 6), rel=1e-13)
        assert DEFAULT_C6 / a**6 / OMEGA == pytest.approx(v, rel=1e-12)
        couplings = fmap.register.interaction_matrix(DEFAULT_C6)
        assert float(couplings[0, 1]) / OMEGA == pytest.approx(v, rel=1e-12)
        assert float(couplings[0, 2]) / OMEGA == pytest.approx(v / 64, rel=1e-12)
        # an open chain in the order of the inputs: atom i at (i a, 0)
        expected = np.array([[i * a, 0.0] for i in range(4)])
        np.testing.assert_allclose(fmap.register.positions.numpy(), expected, rtol=1e-13)

    def test_spacing_refuses_a_non_positive_ratio(self) -> None:
        with pytest.raises(ValueError, match="v_over_omega must be > 0"):
            bench.spacing(0.0, OMEGA, DEFAULT_C6)

    def test_the_grid_is_the_product_and_holds_gamma_zero_for_every_spacing(self) -> None:
        settings = bench.Settings()
        points = bench.grid(settings)
        assert len(points) == len(settings.v_over_omega) * len(settings.gamma_over_omega) == 28
        assert set(points) == {
            (v, g) for v in settings.v_over_omega for g in settings.gamma_over_omega
        }
        assert {v for v, g in points if g == 0.0} == set(settings.v_over_omega)
        assert bench.grid(QUICK) == [(1.0, 0.1)]

    def test_the_pulse_area_is_the_same_at_every_grid_point(self) -> None:
        """``T = ΩT/Ω``: 0.25 µs at the reference Ω, whatever ``V`` and γ."""
        settings = bench.Settings(n_atoms=2)
        assert settings.evolution_time == pytest.approx(0.25, rel=1e-15)
        for v, gamma in bench.grid(settings):
            for interactions in (True, False):
                fmap = bench.feature_map(settings, v, gamma, interactions=interactions)
                assert fmap.evolution_time * fmap.encoding.omega == pytest.approx(math.pi)
                assert fmap.encoding.delta_max == pytest.approx(math.sqrt(3) * OMEGA)
        longer = bench.Settings(n_atoms=2, omega_t=2 * math.pi)
        assert bench.feature_map(longer, 1.0, 0.0, interactions=True).evolution_time == (
            pytest.approx(0.5)
        )

    def test_gamma_is_in_units_of_omega_and_zero_needs_no_steps(self) -> None:
        settings = bench.Settings(n_atoms=2, n_steps=123)
        noiseless = bench.feature_map(settings, 1.0, 0.0, interactions=True)
        assert noiseless.gamma == 0.0
        assert noiseless.n_steps is None
        noisy = bench.feature_map(settings, 1.0, 0.25, interactions=True)
        assert noisy.gamma == pytest.approx(math.pi)  # 0.25 · 4π
        assert noisy.n_steps == 123

    @pytest.mark.parametrize("ratio", [0.1, 1.0])
    def test_dephasing_sweep_against_the_damped_rabi_oscillation(self, ratio: float) -> None:
        """
        A resonant atom (``x → −∞``, so ``Δ → 0``) after the π pulse, from the
        page: ``⟨n⟩ = ½ − ½ e^(−γT/4) [cos λT + γ/(4λ) sin λT]``,
        ``λ = √(Ω² − γ²/16)``.  At ``γ = 0`` the same atom is fully excited.
        """
        settings = bench.Settings(n_atoms=2, n_steps=400)
        x = torch.full((1, 2), -40.0, dtype=torch.float64)
        gamma, t = ratio * OMEGA, math.pi / OMEGA
        lam = math.sqrt(OMEGA**2 - gamma**2 / 16)
        expected = 0.5 - 0.5 * math.exp(-gamma * t / 4) * (
            math.cos(lam * t) + gamma / (4 * lam) * math.sin(lam * t)
        )
        noisy = bench.feature_map(settings, 1.0, ratio, interactions=False).transform(x)
        assert noisy[0].tolist() == pytest.approx([expected, expected], abs=1e-6)
        assert expected < 1 - 0.03  # the dephasing is visible, not a rounding effect
        noiseless = bench.feature_map(settings, 1.0, 0.0, interactions=False).transform(x)
        assert noiseless[0].tolist() == pytest.approx([1.0, 1.0], abs=1e-12)

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            (dict(n_atoms=0), "n_atoms"),
            (dict(n_atoms=11), "n_atoms"),
            (dict(n_atoms=11, arms=("B", "C")), "n_atoms"),
            (dict(n_atoms=0, arms=("C",)), "n_atoms"),
            (dict(shots=0), "shots"),
            (dict(shots=-3, arms=("A", "B")), "shots"),
            (dict(hidden=0), "hidden"),
            (dict(circuit_layers=0), "circuit_layers"),
            (dict(max_epochs=0), "max_epochs"),
            (dict(patience=0), "patience"),
            (dict(v_over_omega=(1.0, 0.0)), "v_over_omega"),
            (dict(v_over_omega=()), "v_over_omega"),
            (dict(gamma_over_omega=(-0.1,)), "gamma_over_omega"),
            (dict(arms=("A", "E")), "arms"),
            (dict(arms=()), "arms"),
            (dict(loss="hinge"), "loss"),
            (dict(n_splits=1), "n_splits"),
            (dict(max_folds=0), "max_folds"),
            (dict(n_trials=-1), "n_trials"),
            (dict(inner_folds=1), "inner_folds"),
            (dict(search_space={"hidden": (4, 8)}), "cannot be tuned"),
            (dict(search_space={"lr": ()}), "no values"),
        ],
    )
    def test_invalid_settings(self, changes: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            bench.Settings(**changes)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            # arm D runs default.qubit with backprop, bounded at AUTO_BACKPROP_MAX_QUBITS (12)
            (
                dict(n_atoms=13, arms=("D",)),
                "n_atoms must be >= 1, <= 10 with arm A or B and <= 12 with arm D; got 13.",
            ),
            (
                dict(n_atoms=13, arms=("C", "D")),
                "n_atoms must be >= 1, <= 10 with arm A or B and <= 12 with arm D; got 13.",
            ),
            (
                dict(n_atoms=11, arms=("A", "D")),
                "n_atoms must be >= 1, <= 10 with arm A or B and <= 12 with arm D; got 11.",
            ),
            (dict(max_epochs=math.nan), "max_epochs must be >= 1; got nan."),
            (dict(batch_size=math.nan), "batch_size must be >= 1; got nan."),
            (dict(patience=math.nan), "patience must be >= 1 or None; got nan."),
            (dict(batch_size=2.5), "batch_size must be >= 1; got 2.5."),
            (dict(batch_size=True), "batch_size must be >= 1; got True."),
            (dict(batch_size=0), "batch_size must be >= 1; got 0."),
            (dict(batch_size=-4, n_trials=0), "batch_size must be >= 1; got -4."),
            (dict(batch_size=None), "batch_size must be >= 1; got None."),
            (dict(batch_size="64"), "batch_size must be >= 1; got '64'."),
            (dict(lr=0.0), "lr must be finite and > 0; got 0.0."),
            (dict(lr=-0.01, n_trials=0), "lr must be finite and > 0; got -0.01."),
            (dict(lr=math.nan), "lr must be finite and > 0; got nan."),
            (dict(lr=math.inf), "lr must be finite and > 0; got inf."),
            (dict(lr=True), "lr must be finite and > 0; got True."),
            (dict(lr=None), "lr must be finite and > 0; got None."),
            (dict(lr="0.01"), "lr must be finite and > 0; got '0.01'."),
            (dict(max_epochs=None), "max_epochs must be >= 1; got None."),
            (dict(max_epochs="64"), "max_epochs must be >= 1; got '64'."),
            (dict(max_epochs=2.5), "max_epochs must be >= 1; got 2.5."),
            (dict(max_epochs=True), "max_epochs must be >= 1; got True."),
            (dict(patience=0), "patience must be >= 1 or None; got 0."),
            (dict(patience=2.5), "patience must be >= 1 or None; got 2.5."),
            (dict(patience=True), "patience must be >= 1 or None; got True."),
            (
                dict(search_space={"batch_size": (32, 0)}),
                "search_space['batch_size'] values must be >= 1; got [32, 0].",
            ),
            (
                dict(search_space={"batch_size": (None,)}),
                "search_space['batch_size'] values must be >= 1; got [None].",
            ),
            (
                dict(search_space={"batch_size": ("64",)}),
                "search_space['batch_size'] values must be >= 1; got ['64'].",
            ),
            (
                dict(search_space={"lr": (0.01, 0.0)}),
                "search_space['lr'] values must be finite and > 0; got [0.01, 0.0].",
            ),
            (
                dict(search_space={"lr": (-0.1,)}),
                "search_space['lr'] values must be finite and > 0; got [-0.1].",
            ),
            (
                dict(search_space={"lr": (0.01, math.inf)}),
                "search_space['lr'] values must be finite and > 0; got [0.01, inf].",
            ),
            (
                dict(search_space={"lr": (True,)}),
                "search_space['lr'] values must be finite and > 0; got [True].",
            ),
            (
                dict(search_space={"lr": (None,)}),
                "search_space['lr'] values must be finite and > 0; got [None].",
            ),
            (
                dict(search_space={"lr": ("0.01",)}),
                "search_space['lr'] values must be finite and > 0; got ['0.01'].",
            ),
            (
                dict(search_space={"max_epochs": (0,)}),
                "search_space['max_epochs'] values must be >= 1; got [0].",
            ),
            (
                dict(search_space={"max_epochs": (None,)}),
                "search_space['max_epochs'] values must be >= 1; got [None].",
            ),
            (
                dict(search_space={"max_epochs": ("64",)}),
                "search_space['max_epochs'] values must be >= 1; got ['64'].",
            ),
            (
                dict(search_space={"max_epochs": (10, -1)}),
                "search_space['max_epochs'] values must be >= 1; got [10, -1].",
            ),
            (
                dict(search_space={"max_epochs": (True,)}),
                "search_space['max_epochs'] values must be >= 1; got [True].",
            ),
            (
                dict(search_space={"patience": (0,)}),
                "search_space['patience'] values must be >= 1 or None; got [0].",
            ),
            (
                dict(search_space={"patience": (5, -3, None)}),
                "search_space['patience'] values must be >= 1 or None; got [5, -3, None].",
            ),
            (
                dict(search_space={"max_epochs": (math.nan,)}),
                "search_space['max_epochs'] values must be >= 1; got [nan].",
            ),
            (
                dict(search_space={"max_epochs": (2.5,)}),
                "search_space['max_epochs'] values must be >= 1; got [2.5].",
            ),
            (
                dict(search_space={"batch_size": (math.nan,)}),
                "search_space['batch_size'] values must be >= 1; got [nan].",
            ),
            (
                dict(search_space={"batch_size": (2.5,)}),
                "search_space['batch_size'] values must be >= 1; got [2.5].",
            ),
            (
                dict(search_space={"patience": (math.nan,)}),
                "search_space['patience'] values must be >= 1 or None; got [nan].",
            ),
            (
                dict(search_space={"patience": (2.5,)}),
                "search_space['patience'] values must be >= 1 or None; got [2.5].",
            ),
        ],
    )
    def test_refused_settings_name_the_setting_and_the_value(
        self, changes: dict[str, Any], message: str
    ) -> None:
        """
        #540, #605: values ``train_model`` or Adam would take late, or not at all, are refused
        here.
        """
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.Settings(**changes)

    @pytest.mark.parametrize(
        "space",
        [
            {"max_epochs": (1, 10)},
            {"patience": (1, None)},
            {"patience": (None,)},
            {"max_epochs": (1,), "patience": (1, 5, None)},
        ],
    )
    def test_valid_search_space_max_epochs_and_patience(self, space: dict[str, Any]) -> None:
        """#605: candidates on the bounds, and ``None`` for ``patience``, are accepted."""
        settings = bench.Settings(search_space=space)
        for key, expected in space.items():
            assert settings.search_space[key] == expected

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_epochs", np.int64(10)),
            ("batch_size", np.int64(32)),
            ("patience", np.int64(5)),
            ("lr", np.float32(0.01)),
        ],
    )
    @pytest.mark.parametrize("as_search_space", [False, True])
    def test_numpy_numeric_training_settings_are_accepted(
        self, key: str, value: Any, as_search_space: bool
    ) -> None:
        """NumPy integer and real scalars are valid training-setting values."""
        if as_search_space:
            settings = bench.Settings(search_space={key: (value,)})
            assert settings.search_space[key] == (value,)
        else:
            settings = bench.Settings(**{key: value})
            assert getattr(settings, key) == value

    def test_a_limit_binds_only_where_its_arm_runs(self) -> None:
        """``MAX_ATOMS`` is the Rydberg maps' limit and ``circuit_layers`` is arm D's depth."""
        assert bench.Settings(n_atoms=11, arms=("C", "D")).n_atoms == 11
        assert bench.Settings(n_atoms=12, arms=("C", "D")).n_atoms == 12
        assert bench.Settings(n_atoms=13, arms=("C",)).n_atoms == 13
        assert bench.Settings(n_atoms=10).n_atoms == 10
        assert bench.Settings(circuit_layers=0, arms=("A", "C")).circuit_layers == 0
        assert bench.Settings(shots=1, patience=None, max_epochs=1).shots == 1


# ---------------------------------------------------------------------------
# Arms A and B: the features
# ---------------------------------------------------------------------------


class TestRydbergArms:
    X = np.random.default_rng(3).uniform(-math.pi, math.pi, size=(9, 4))

    def _features(self, settings: Any, arm: str, v: float, gamma: float) -> np.ndarray:
        (rep,) = [
            r
            for r in bench.representations(_replace(settings, arms=(arm,)))
            if (r.v_over_omega, r.gamma_over_omega) == (v, gamma)
        ]
        assert rep.compute is not None
        return np.asarray(rep.compute(self.X))

    def test_arm_b_is_the_closed_form_single_atom_feature(self) -> None:
        """Feature ``i`` is ``f(√(1 + 3 σ(x_i)²))`` of input ``i`` alone, at every spacing."""
        settings = bench.Settings(v_over_omega=(0.1, 1.0, 30.0), gamma_over_omega=(0.0,))
        for v in settings.v_over_omega:
            features = self._features(settings, "B", v, 0.0)
            np.testing.assert_allclose(features, _single_atom_feature(self.X), rtol=0, atol=1e-12)

    def test_arm_b_under_dephasing_is_a_single_atom_evolved_alone(self) -> None:
        """Column ``i`` equals a one-atom map (same pulse, γ and steps) applied to input ``i``."""
        settings = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.5,), n_steps=60)
        features = self._features(settings, "B", 1.0, 0.5)
        alone = RydbergFeatureMap(
            AtomRegister.chain(1, 1.0),
            PulseEncoding(1, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.5 * OMEGA,
            n_steps=60,
        )
        for i in range(4):
            column = alone.transform(self.X[:, [i]]).numpy()[:, 0]
            np.testing.assert_allclose(features[:, i], column, rtol=0, atol=1e-12)
        # and it differs from the noiseless closed form by far more than that
        assert np.abs(features - _single_atom_feature(self.X)).max() > 0.05

    def test_arm_a_is_the_interacting_map_and_mixes_neighbouring_inputs(self) -> None:
        settings = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.0,))
        features = self._features(settings, "A", 1.0, 0.0)
        by_hand = _map_by_hand(4, 1.0, 0.0, 1).transform(self.X).numpy()
        np.testing.assert_allclose(features, by_hand, rtol=0, atol=1e-12)
        control = self._features(settings, "B", 1.0, 0.0)
        assert np.abs(features - control).max() > 0.1
        # Input 1 moves the feature of atom 0 in A and not in B.
        moved = self.X.copy()
        moved[:, 1] += 0.7
        (rep_a,) = bench.representations(_replace(settings, arms=("A",)))
        (rep_b,) = bench.representations(_replace(settings, arms=("B",)))
        assert np.abs(rep_a.compute(moved)[:, 0] - features[:, 0]).max() > 1e-2
        assert np.abs(rep_b.compute(moved)[:, 0] - control[:, 0]).max() < 1e-12

    def test_the_two_arms_differ_in_the_interactions_flag_only(self) -> None:
        settings = bench.Settings(v_over_omega=(3.0,), gamma_over_omega=(1.0,))
        rep_a, rep_b = bench.representations(_replace(settings, arms=("A", "B")))
        assert (rep_a.arm, rep_b.arm) == ("A", "B")
        config_a, config_b = rep_a.config["config"], rep_b.config["config"]
        assert config_a["interactions"] is True
        assert config_b["interactions"] is False
        assert {k: v for k, v in config_a.items() if k != "interactions"} == {
            k: v for k, v in config_b.items() if k != "interactions"
        }
        assert config_a == _map_by_hand(4, 3.0, 1.0, 200).get_config()
        assert rep_a.heads == rep_b.heads == ("linear", "mlp")

    def test_shots_are_drawn_with_the_seed_of_the_matrix(self) -> None:
        """
        From a generator seeded with the matrix's seed (#523): the rule of
        :func:`shot_stream_seed` over the config, which holds ``shot_seed``,
        and the inputs, by hand.
        """
        settings = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.0,), shots=64)
        features = self._features(settings, "A", 1.0, 0.0)
        shot_seed = bench.run_seeds(settings.random_state)["shot_seed"]
        config = _rydberg_config_by_hand(_map_by_hand(4, 1.0, 0.0, 1).get_config(), 64, shot_seed)
        (rep,) = bench.representations(_replace(settings, arms=("A",)))
        assert rep.config == config
        seed = _stream_seed_by_hand(config, self.X)
        assert bench.shot_stream_seed(rep.config, self.X) == seed
        assert 0 <= seed < 2**32 and seed != shot_seed
        by_hand = _map_by_hand(4, 1.0, 0.0, 1).transform(
            self.X, shots=64, generator=torch.Generator().manual_seed(seed)
        )
        np.testing.assert_array_equal(features, by_hand.numpy())
        # multiples of 1/64, and within the binomial error of the exact features
        np.testing.assert_allclose(features * 64, np.round(features * 64), atol=1e-9)
        exact = _map_by_hand(4, 1.0, 0.0, 1).transform(self.X).numpy()
        assert 0 < np.abs(features - exact).max() < 5 * 0.5 / math.sqrt(64)


# ---------------------------------------------------------------------------
# Arm D: the frozen circuit
# ---------------------------------------------------------------------------


class TestCircuitArm:
    def test_weights_are_uniform_draws_from_the_circuit_seed_and_frozen(self) -> None:
        settings = bench.Settings(n_atoms=3, circuit_layers=2)
        layer = bench.frozen_circuit(settings, 11)
        weights = layer.qlayer.weights
        draw = torch.rand(
            (2, 3, 3), generator=torch.Generator().manual_seed(11), dtype=torch.float64
        )
        expected = ((2 * draw - 1) * math.pi).to(weights.dtype)
        assert torch.equal(weights.detach(), expected)
        assert float(weights.abs().max()) <= math.pi
        assert all(not p.requires_grad for p in layer.parameters())
        assert not torch.equal(weights, bench.frozen_circuit(settings, 12).qlayer.weights)

    def test_features_are_the_z_expectations_of_the_circuit_written_out(self) -> None:
        """
        ``RX(x_i)`` on qubit ``i``, then per layer a CNOT ring ``i → i+1`` and
        ``Rot`` on every qubit, then ``⟨Z_i⟩``: feature ``i`` is qubit ``i``.
        """
        settings = bench.Settings(n_atoms=3, circuit_layers=2)
        (rep,) = [
            r for r in bench.representations(_replace(settings, arms=("D",))) if r.arm == "D"
        ]
        seed = bench.run_seeds(settings.random_state)["circuit_seed"]
        weights = bench.frozen_circuit(settings, seed).qlayer.weights.detach().double().numpy()
        device = qml.device("default.qubit", wires=3)

        @qml.qnode(device)
        def circuit(x: np.ndarray) -> Any:
            for i in range(3):
                qml.RX(x[i], wires=i)
            for layer in range(2):
                for i in range(3):
                    qml.CNOT(wires=[i, (i + 1) % 3])
                for i in range(3):
                    qml.Rot(*weights[layer, i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        X = np.random.default_rng(5).uniform(-math.pi, math.pi, size=(6, 3))
        by_hand = np.array([[float(v) for v in circuit(row)] for row in X])
        features = rep.compute(X)
        np.testing.assert_allclose(features, by_hand, rtol=0, atol=1e-9)
        assert np.abs(features).max() <= 1.0
        assert rep.n_frozen_parameters == 2 * 3 * 3
        assert rep.config["config"]["circuit_seed"] == seed
        np.testing.assert_allclose(features, _circuit_by_hand(X, seed), rtol=0, atol=1e-9)

    def test_the_recorded_config_is_that_of_the_circuit_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Qubits, layers, entangler and readout in the config, which is also the
        cache key, are read off the layer: a circuit built otherwise is
        recorded otherwise.
        """
        settings = bench.Settings(n_atoms=3, circuit_layers=1, arms=("D",))
        seed = bench.run_seeds(settings.random_state)["circuit_seed"]
        (rep, _) = bench.representations(settings)
        assert rep.config["config"] == {
            "n_qubits": 3,
            "n_layers": 1,
            "rotation": "X",
            "entangler": "ring",
            "readout": "all",
            "device_name": "default.qubit",
            "weights": "uniform(-pi, pi)",
            "circuit_seed": seed,
        }

        def other(settings: Any, circuit_seed: int) -> Any:
            return bench.QuantumEncodingLayer(
                4, 2, device_name="default.qubit", entangler="brickwork", readout="first"
            )

        monkeypatch.setattr(bench, "frozen_circuit", other)
        (rep, _) = bench.representations(settings)
        assert rep.config["config"] == {
            "n_qubits": 4,
            "n_layers": 2,
            "rotation": "X",
            "entangler": "brickwork",
            "readout": "first",
            "device_name": "default.qubit",
            "weights": "uniform(-pi, pi)",
            "circuit_seed": seed,
        }

    def test_sampled_features_depend_on_the_inputs_and_the_seed_alone(self) -> None:
        """
        With shots, a matrix is drawn by a device seeded with the matrix's
        seed (#523), whatever was sampled before it: the cache key (inputs
        and config) then determines the features, also when a run is resumed.
        """
        settings = bench.Settings(n_atoms=3, circuit_layers=2, shots=64)
        (rep,) = [
            r for r in bench.representations(_replace(settings, arms=("D",))) if r.arm == "D"
        ]
        seeds = bench.run_seeds(settings.random_state)
        assert rep.config["shots"] == 64
        assert rep.config["shot_seed"] == seeds["shot_seed"]
        assert rep.config["shot_stream"] == "per-matrix"
        X = np.random.default_rng(6).uniform(-math.pi, math.pi, size=(7, 3))
        first = rep.compute(X)
        rep.compute(X[:3])  # another matrix in between moves no stream
        np.testing.assert_array_equal(rep.compute(X), first)
        # by hand: a circuit built here with the matrix's seed, evaluated once
        seed = _stream_seed_by_hand(rep.config, X)
        assert seed != seeds["shot_seed"]
        fresh = bench.frozen_circuit(settings, seeds["circuit_seed"], seed)
        with torch.no_grad():
            by_hand = fresh(torch.from_numpy(X)).double().numpy()
            again = fresh(torch.from_numpy(X)).double().numpy()
        np.testing.assert_array_equal(first, by_hand)
        assert not np.array_equal(again, by_hand)  # the stream of one circuit does run on
        # ⟨Z⟩ from 64 shots: (64 − 2k)/64, within the binomial error of the exact value
        np.testing.assert_allclose((first + 1) * 32, np.round((first + 1) * 32), atol=1e-9)
        (exact_rep,) = [
            r
            for r in bench.representations(_replace(settings, arms=("D",), shots=None))
            if r.arm == "D"
        ]
        assert 0 < np.abs(first - exact_rep.compute(X)).max() < 5 / math.sqrt(64)

    def test_the_hybrid_row_is_a_trained_circuit_on_the_inputs(self) -> None:
        settings = bench.Settings(n_atoms=3, circuit_layers=2)
        reps = bench.representations(_replace(settings, arms=("D",)))
        assert [(r.arm, r.kind, r.heads) for r in reps] == [
            ("D", "frozen-circuit", ("linear", "mlp")),
            ("D-hybrid", "inputs", ("hybrid",)),
        ]
        assert reps[1].compute is None
        assert reps[1].n_frozen_parameters == 0


# ---------------------------------------------------------------------------
# Heads and parameter counts
# ---------------------------------------------------------------------------


class TestModels:
    @pytest.mark.parametrize(("d", "hidden"), [(4, 8), (6, 5), (10, 3)])
    def test_parameter_counts_by_hand(self, d: int, hidden: int) -> None:
        settings = bench.Settings(hidden=hidden, circuit_layers=2)
        linear = bench.build_model("linear", d, settings, 0)
        mlp = bench.build_model("mlp", d, settings, 0)
        assert isinstance(linear, LinearClassifier)
        assert isinstance(mlp, ClassicalBaseline)
        assert linear.count_parameters() == d + 1
        assert mlp.count_parameters() == hidden * (d + 2) + 1 == d * hidden + hidden + hidden + 1
        assert mlp.get_config()["hidden_dims"] == [hidden]

    def test_hybrid_parameter_count_by_hand(self) -> None:
        model = bench.build_model("hybrid", 4, bench.Settings(circuit_layers=2), 0)
        assert isinstance(model, HybridBinaryClassifier)
        assert model.count_parameters() == 3 * 2 * 4 + 4 + 1
        config = model.get_config()
        assert config["use_classical_encoder"] is False
        assert (config["n_input_features"], config["n_qubits"], config["n_layers"]) == (4, 4, 2)

    def test_the_same_seed_gives_every_arm_the_same_initial_head(self) -> None:
        a = bench.build_model("mlp", 4, bench.Settings(), 17)
        b = bench.build_model("mlp", 4, bench.Settings(), 17)
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            assert torch.equal(p, q)

    def test_unknown_head(self) -> None:
        with pytest.raises(ValueError, match="unknown head"):
            bench.build_model("svm", 4, bench.Settings(), 0)

    def test_the_shared_helpers_are_the_library_ones(self) -> None:
        """
        One implementation each, the one ``run_benchmark`` uses (#524): the tuning draw,
        fit-and-predict and the JSON mapping are imported, not written again.
        """
        assert bench.sample_configs is benchmark.sample_configs
        assert bench.fit_and_predict is benchmark.fit_and_predict
        assert bench.plain_data is experiment.plain_data

    def test_fit_and_score_is_fit_and_predict_on_the_blocks_of_the_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rng = np.random.default_rng(3)
        labels = np.tile([0, 1], 20)
        features = rng.normal(size=(40, 2)) + labels[:, None]
        # training rows 0-19 (16 real, 4 synthetic), validation 20-29, test 30-39
        split = bench.Split(
            inputs=features, labels=labels, n_train=20, n_synthetic=4, n_val=10, n_test=10
        )
        calls: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []
        real = bench.fit_and_predict

        def spy(*args: Any, **kwargs: Any) -> Any:
            out = real(*args, **kwargs)
            calls.append((args, kwargs, out))
            return out

        monkeypatch.setattr(bench, "fit_and_predict", spy)
        config = {"lr": 0.05, "max_epochs": 5, "batch_size": 8, "patience": None}
        model = LinearClassifier(2, init_seed=1)
        fit = bench.fit_and_score(model, features, split, config, loss="bce", batch_seed=7)
        ((args, kwargs, out),) = calls
        assert args[0] is model and args[1] is nn.BCEWithLogitsLoss
        for got, rows in zip(
            args[2::2], (slice(0, 20), slice(20, 30), slice(30, 40)), strict=True
        ):
            np.testing.assert_array_equal(got, features[rows])
        for got, rows in zip(
            args[3::2], (slice(0, 20), slice(20, 30), slice(30, 40)), strict=True
        ):
            np.testing.assert_array_equal(got, labels[rows])
        assert kwargs == {**config, "batch_seed": 7}
        assert (fit.mcc, fit.threshold, fit.seconds, fit.epochs) == (
            out.mcc,
            out.threshold,
            out.seconds,
            out.epochs,
        )
        assert fit.pr_auc == pr_auc(labels[30:], out.probability)


# ---------------------------------------------------------------------------
# Data: the positive control and a UCI loader
# ---------------------------------------------------------------------------


class TestPositiveControl:
    def test_labels_by_hand(self) -> None:
        latent = [
            [1.0, 2.0, 3.0, 4.0],  # 2 + 6 + 12 = 20
            [1.0, -2.0, 3.0, -4.0],  # −2 − 6 − 12 = −20
            [1.0, 1.0, -3.0, 0.5],  # 1 − 3 − 1.5 = −3.5
            [-1.0, -1.0, 0.1, 5.0],  # 1 − 0.1 + 0.5 = 1.4
            [5.0, 0.1, -1.0, 9.0],  # 0.5 − 0.1 − 9 = −8.6: not the first bond alone
            [1.0, 1.0, -1.0, 1.0],  # 1 − 1 − 1 = −1
        ]
        assert bench.product_labels(latent).tolist() == [1, 0, 0, 1, 0, 0]
        assert bench.product_labels([[2.0, 3.0], [2.0, -3.0]]).tolist() == [1, 0]

    def test_only_neighbours_enter(self) -> None:
        """Atoms 0 and 2 are not neighbours: their product alone decides nothing."""
        assert bench.product_labels([[1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]).tolist() == [0, 0]
        with pytest.raises(ValueError, match="N >= 2"):
            bench.product_labels([[1.0], [2.0]])

    def test_flipping_one_parity_flips_every_label(self) -> None:
        """The symmetry behind ``P(y = 1 | z_i) = ½``: the even latents stay as they are."""
        control = bench.positive_control(5, 400, seed=2)
        flipped = control.latent * np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        np.testing.assert_array_equal(bench.product_labels(flipped), 1 - control.y)

    def test_columns_are_noisy_copies_in_blocks_of_halving_size(self) -> None:
        control = bench.positive_control(4, 3000, seed=0, noise=0.1)
        assert control.X.shape == (3000, 15)  # 8 + 4 + 2 + 1
        assert control.X.dtype == np.float64
        assert control.y.dtype == np.int64
        np.testing.assert_array_equal(control.y, bench.product_labels(control.latent))
        owner = [0] * 8 + [1] * 4 + [2] * 2 + [3]
        residual = control.X - control.latent[:, owner]
        assert residual.std(axis=0) == pytest.approx(np.full(15, 0.1), rel=0.1)
        assert abs(control.y.mean() - 0.5) < 0.05
        again = bench.positive_control(4, 3000, seed=0, noise=0.1)
        np.testing.assert_array_equal(again.X, control.X)

    def test_leading_eigenvalues_are_the_closed_form(self) -> None:
        """``λ_i = 1 + (m_i − 1)/(1 + noise²)`` for the standardised columns."""
        noise = 0.3
        control = bench.positive_control(4, 20000, seed=1, noise=noise)
        scaled = (control.X - control.X.mean(0)) / control.X.std(0)
        pca = PCANormalizer(n_components=4).fit(scaled)
        rho = 1 / (1 + noise**2)
        expected = [1 + (m - 1) * rho for m in (8, 4, 2, 1)]
        assert pca.explained_variance_.tolist() == pytest.approx(expected, rel=0.03)

    @pytest.mark.parametrize("n_atoms", [4, 6])
    def test_latent_i_ends_up_on_atom_i_with_a_positive_sign(self, n_atoms: int) -> None:
        """Input column ``i`` of the pipeline follows latent ``i`` and no other."""
        control = bench.positive_control(n_atoms, 600, seed=0)
        rows = np.arange(600)
        inputs = bench.preprocess(control.X, rows, n_atoms)
        correlation = np.corrcoef(inputs.T, control.latent.T)[:n_atoms, n_atoms:]
        assert np.diag(correlation).min() > 0.93  # π tanh of a normal variable: about 0.96
        off_diagonal = correlation - np.diag(np.diag(correlation))
        assert np.abs(off_diagonal).max() < 0.15
        # increasing: the ranks agree almost perfectly
        for i in range(n_atoms):
            order = np.argsort(control.latent[:, i])
            ranked = np.argsort(np.argsort(inputs[order, i]))
            assert np.corrcoef(ranked, np.arange(600))[0, 1] > 0.98

    def test_arm_a_separates_the_classes_and_arm_b_cannot(self) -> None:
        """
        The Fisher ratio of arm B's features, and of the inputs, sits at its
        no-information level ``(M − 2)/(M − d − 3) · d M/(n0 n1)`` (the class
        means coincide); with interactions at ``V = Ω`` it is far above it.
        """
        m, d = 600, 4
        control = bench.positive_control(d, m, seed=0)
        inputs = bench.preprocess(control.X, np.arange(m), d)
        n1 = int(control.y.sum())
        null = (m - 2) / (m - d - 3) * d * m / (n1 * (m - n1))
        fisher = {
            "inputs": separation_measures(inputs, control.y).fisher_ratio,
            "A": separation_measures(
                _map_by_hand(d, 1.0, 0.0, 1).transform(inputs), control.y
            ).fisher_ratio,
            "B": separation_measures(
                _map_by_hand(d, 1.0, 0.0, 1, interactions=False).transform(inputs), control.y
            ).fisher_ratio,
        }
        assert fisher["B"] < 3 * null
        assert fisher["inputs"] < 3 * null
        assert fisher["A"] > 20 * null
        assert fisher["A"] > 20 * fisher["B"]

    def test_load_dataset_draws_the_control_from_the_run_seed(self) -> None:
        settings = _replace(QUICK, random_state=5)
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        assert data_seed == bench.run_seeds(5)["data_seed"]
        by_hand = bench.positive_control(settings.n_atoms, settings.n_samples, data_seed)
        np.testing.assert_array_equal(X, by_hand.X)
        np.testing.assert_array_equal(y, by_hand.y)
        with pytest.raises(ValueError, match="unknown dataset"):
            bench.load_dataset("mnist", settings)

    @pytest.mark.parametrize("arms", [("C",), ("C", "D")])
    def test_n_atoms_above_its_bound_is_refused_with_any_arms(self, arms: tuple[str, ...]) -> None:
        """#540: ``2^N − 1`` columns, 2.6 GiB per PCA fit at 13; 12 is the bound."""
        assert bench.POSITIVE_CONTROL_MAX_ATOMS == 12
        message = "n_atoms must be <= 12 with the positive control (2^N - 1 raw columns); got 13."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.positive_control(13, 3, seed=0)
        # Settings leaves it to the dataset: with arm C alone nothing else bounds n_atoms.
        settings = bench.Settings(n_atoms=13, arms=("C",), n_samples=3)
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.load_dataset("positive-control", settings)
        # 12 is built, with every arm that can run at 12
        settings = bench.Settings(n_atoms=12, arms=arms, n_samples=3)
        X, y, _ = bench.load_dataset("positive-control", settings)
        assert X.shape == (3, 2**12 - 1)
        assert y.shape == (3,)


def _bonds_by_hand(n_atoms: int, geometry: str, cols: int | None = None) -> list[tuple[int, int]]:
    """The nearest-neighbour pairs written out from the constructors' index conventions."""
    if geometry == "chain":
        return [(i, i + 1) for i in range(n_atoms - 1)]
    if geometry == "ring":
        ends = [(i, (i + 1) % n_atoms) for i in range(n_atoms)]
        return sorted((min(i, j), max(i, j)) for i, j in ends)
    assert cols is not None
    across = [(k, k + 1) for k in range(n_atoms) if k % cols != cols - 1]
    up = [(k, k + cols) for k in range(n_atoms - cols)]
    return sorted(across + up)


def _colouring(n_atoms: int, geometry: str, cols: int | None = None) -> np.ndarray:
    """``±1`` per atom, opposite at the two ends of every bond: index parity, ``r + c`` on a grid."""
    k = np.arange(n_atoms)
    parity = (k // cols + k % cols) % 2 if cols is not None else k % 2
    return np.where(parity == 0, 1.0, -1.0)


#: Geometries of the label tests: ``(n_atoms, geometry, grid_cols)``.
LABEL_GEOMETRIES = [
    (4, "chain", None),
    (5, "chain", None),
    (3, "ring", None),
    (4, "ring", None),
    (5, "ring", None),
    (6, "ring", None),
    (4, "grid", 2),
    (6, "grid", 3),
    (9, "grid", 3),
    (8, "grid", 4),  # an even number of columns: r + c, not k, is the colouring
]


class TestGeometryLabel:
    """
    #738: the positive control's label over the nearest-neighbour pairs of a geometry.

    Checked against the constructors' index conventions written out by hand, the
    chain formula ``Σ z_i z_{i+1}`` as it was (label by label, and the data
    fingerprints of ``main`` at 6080cde), the colouring that flips every bond, and,
    for the triangle, ``P(y = 1) = 1 − 1/√3`` and SciPy quadrature of
    ``P(y = 1 | z_0 = t)``.
    """

    @pytest.mark.parametrize(("n_atoms", "geometry", "cols"), LABEL_GEOMETRIES)
    def test_the_bonds_are_the_nearest_neighbours_written_out_by_hand(
        self, n_atoms: int, geometry: str, cols: int | None
    ) -> None:
        assert bench.nearest_neighbour_bonds(n_atoms, geometry, cols) == _bonds_by_hand(
            n_atoms, geometry, cols
        )

    def test_the_bonds_of_small_shapes_typed_in(self) -> None:
        """The cases the issue names: the ring's closing bond, the grid's columns."""
        assert bench.nearest_neighbour_bonds(3, "ring") == [(0, 1), (0, 2), (1, 2)]
        assert bench.nearest_neighbour_bonds(4, "ring") == [(0, 1), (0, 3), (1, 2), (2, 3)]
        # 2 × 2: atoms 1 = (a, 0) and 2 = (0, a) are a diagonal, √2 a apart, not a bond
        assert bench.nearest_neighbour_bonds(4, "grid") == [(0, 1), (0, 2), (1, 3), (2, 3)]
        assert bench.nearest_neighbour_bonds(6, "grid", 3) == [
            (0, 1),
            (0, 3),
            (1, 2),
            (1, 4),
            (2, 5),
            (3, 4),
            (4, 5),
        ]
        # without grid_cols, the default shape of grid_shape: 6 atoms are 2 × 3
        assert bench.nearest_neighbour_bonds(6, "grid") == bench.nearest_neighbour_bonds(
            6, "grid", 3
        )

    @pytest.mark.parametrize("geometry", ["chain", "ring", "grid"])
    @pytest.mark.parametrize("n_atoms", [3, 4, 5, 6, 8, 9, 10, 12])
    def test_bonds_are_the_pairs_at_the_spacing_and_no_other_pair_is_closer_than_root_2(
        self, geometry: str, n_atoms: int
    ) -> None:
        """
        So the radius ``(1 + √2)/2 · a`` lies between the two in every geometry the
        benchmark builds (``2a cos(π/N) >= √2 a`` on a ring of ``N >= 4``).
        """
        cols: int | None = None
        if geometry == "grid":
            if n_atoms in (3, 5):
                return  # a prime number of atoms has no grid shape
            rows, cols = bench.grid_shape(n_atoms)
            register = AtomRegister.grid(rows, cols, 3.0)
        else:
            register = getattr(AtomRegister, geometry)(n_atoms, 3.0)
        positions = register.positions.numpy()
        distance = np.linalg.norm(positions[:, None] - positions[None], axis=-1) / 3.0
        bonds = bench.nearest_neighbour_bonds(n_atoms, geometry, cols)
        assert bonds == _bonds_by_hand(n_atoms, geometry, cols)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if (i, j) in bonds:
                    assert distance[i, j] == pytest.approx(1.0, abs=1e-12)
                else:
                    assert distance[i, j] >= math.sqrt(2) - 1e-12

    @pytest.mark.parametrize("n_atoms", range(2, 13))
    def test_the_chain_label_is_bit_identical_to_the_formula_before_the_geometry(
        self, n_atoms: int
    ) -> None:
        """Existing chain results stay comparable: the formula of ``main`` at 6080cde."""
        latent = np.random.default_rng(n_atoms).standard_normal((5000, n_atoms))
        before = ((latent[:, :-1] * latent[:, 1:]).sum(axis=1) > 0).astype(np.int64)
        bonds = bench.nearest_neighbour_bonds(n_atoms, "chain")
        np.testing.assert_array_equal(bench.product_labels(latent, bonds), before)
        np.testing.assert_array_equal(bench.product_labels(latent), before)

    @pytest.mark.parametrize(
        ("n_atoms", "n_samples", "random_state", "sha256"),
        [
            (4, 600, 0, "0f76549d09e84e908f062cdaf914bbcc0470a9cde46fbea4a27bccd17695e593"),
            (6, 600, 0, "2076ebf7b1d914d07e2f4b66a1fb7d1bdfda936767412cc39e100aea6740d8f1"),
            (4, 160, 0, "9893f5fabce42b3762849c354a1a6bd5d98efebd7d2037f8239c62b14874b4b8"),
            (5, 600, 3, "50437b04fdbad1c0a28aca815f02e5fbe48b620f6574bfe0a534375514c048a5"),
        ],
    )
    def test_the_chain_control_has_the_data_fingerprint_from_before(
        self, n_atoms: int, n_samples: int, random_state: int, sha256: str
    ) -> None:
        """Fingerprints of ``positive_control`` computed on ``main`` at 6080cde."""
        seed = bench.run_seeds(random_state)["data_seed"]
        chain = bench.nearest_neighbour_bonds(n_atoms, "chain")
        for control in (
            bench.positive_control(n_atoms, n_samples, seed),
            bench.positive_control(n_atoms, n_samples, seed, bonds=chain),
        ):
            assert fingerprint(control.X, control.y) == sha256

    def test_the_label_sums_the_given_bonds_by_hand(self) -> None:
        z = [
            [1.0, 2.0, 3.0, 4.0],  # ring: 2 + 4 + 6 + 12 = 24; grid: 2 + 3 + 8 + 12 = 25
            [1.0, 1.0, -1.0, 1.0],  # ring: 1 + 1 − 1 − 1 = 0; grid: 1 − 1 + 1 − 1 = 0
            [2.0, -1.0, 1.0, 0.5],  # ring: −2 + 1 − 1 + 0.5 = −1.5; grid: −2 + 2 − 0.5 + 0.5 = 0
            [1.0, -1.0, -3.0, 2.0],  # ring: −1 + 2 + 3 − 6 = −2; grid: −1 − 3 − 2 − 6 = −12
            [-1.0, 1.0, 3.0, 2.0],  # ring: −1 − 2 + 3 + 6 = 6; grid: −1 − 3 + 2 + 6 = 4
        ]
        ring = bench.nearest_neighbour_bonds(4, "ring")
        grid = bench.nearest_neighbour_bonds(4, "grid")
        assert bench.product_labels(z, ring).tolist() == [1, 0, 0, 0, 1]
        assert bench.product_labels(z, grid).tolist() == [1, 0, 0, 0, 1]
        # the closing bond decides: chain 0 + 0 − 1 = −1, the ring adds z_3 z_0 = 2
        assert bench.product_labels([[2.0, 0.0, -1.0, 1.0]]).tolist() == [0]
        assert bench.product_labels([[2.0, 0.0, -1.0, 1.0]], ring).tolist() == [1]
        # the 2 × 2 grid's bond (0, 2) is a column, the chain's (1, 2) a diagonal
        assert bench.product_labels([[1.0, -1.0, 2.0, 0.0]]).tolist() == [0]  # −1 − 2 + 0
        assert bench.product_labels([[1.0, -1.0, 2.0, 0.0]], grid).tolist() == [1]  # −1 + 2

    def test_bonds_out_of_range_or_degenerate_are_refused(self) -> None:
        z = np.ones((2, 3))
        for bonds, match in [
            ([], "at least one bond"),
            ([(0, 3)], r"pairs of distinct atoms in \[0, 3\)"),
            ([(1, 1)], r"pairs of distinct atoms in \[0, 3\)"),
            ([(-1, 0)], r"pairs of distinct atoms in \[0, 3\)"),
            ([(0, 1, 2)], r"pairs of distinct atoms in \[0, 3\)"),
        ]:
            with pytest.raises(ValueError, match=match):
                bench.product_labels(z, bonds)
        with pytest.raises(ValueError, match="unknown geometry"):
            bench.nearest_neighbour_bonds(4, "triangle")
        with pytest.raises(ValueError, match="does not divide"):
            bench.nearest_neighbour_bonds(6, "grid", 4)
        with pytest.raises(ValueError, match="needs n_atoms >= 3"):
            bench.nearest_neighbour_bonds(2, "ring")

    @pytest.mark.parametrize(
        ("n_atoms", "geometry", "cols"),
        [g for g in LABEL_GEOMETRIES if not (g[1] == "ring" and g[0] % 2)],
    )
    def test_the_colouring_flips_every_label_where_the_bonds_are_bipartite(
        self, n_atoms: int, geometry: str, cols: int | None
    ) -> None:
        """Every bond joins the two colours, so ``Σ z_j z_k`` changes sign and the label flips."""
        bonds = bench.nearest_neighbour_bonds(n_atoms, geometry, cols)
        control = bench.positive_control(n_atoms, 400, seed=2, bonds=bonds)
        colour = _colouring(n_atoms, geometry, cols)
        for keep in (1.0, -1.0):  # the atoms of either colour stay as they are
            flipped = bench.product_labels(control.latent * np.where(colour == keep, 1, -1), bonds)
            np.testing.assert_array_equal(flipped, 1 - control.y)

    @pytest.mark.parametrize("n_atoms", [3, 5, 7, 9, 11])
    def test_no_sign_flip_negates_every_bond_of_an_odd_ring(self, n_atoms: int) -> None:
        """Not bipartite: each of the ``2^N`` sign patterns keeps at least one bond's sign."""
        bonds = np.array(bench.nearest_neighbour_bonds(n_atoms, "ring"))
        patterns = 1 - 2 * ((np.arange(2**n_atoms)[:, None] >> np.arange(n_atoms)) & 1)
        negated = patterns[:, bonds[:, 0]] * patterns[:, bonds[:, 1]] == -1
        assert not negated.all(axis=1).any()

    @pytest.mark.parametrize(("n_atoms", "geometry", "cols"), LABEL_GEOMETRIES)
    def test_the_class_means_of_every_single_input_coincide(
        self, n_atoms: int, geometry: str, cols: int | None
    ) -> None:
        """
        The inputs of the pipeline (standardisation, PCA, ``π tanh``), class by class,
        agree within 4 standard errors on every atom: by the colouring where the bonds
        are bipartite, and on an odd ring by the global flip ``z → −z``, which keeps
        every label and turns each input, an odd function of its latent, around.
        """
        m = 20000
        bonds = bench.nearest_neighbour_bonds(n_atoms, geometry, cols)
        control = bench.positive_control(n_atoms, m, seed=11, bonds=bonds)
        inputs = bench.preprocess(control.X, np.arange(m), n_atoms)
        one, zero = inputs[control.y == 1], inputs[control.y == 0]
        difference = one.mean(axis=0) - zero.mean(axis=0)
        error = np.sqrt(one.var(axis=0) / len(one) + zero.var(axis=0) / len(zero))
        assert np.abs(difference / error).max() < 4.0
        # input i is latent i (otherwise this would check the wrong atoms)
        correlation = np.corrcoef(inputs.T, control.latent.T)[:n_atoms, n_atoms:]
        assert np.diag(correlation).min() > 0.9

    @pytest.mark.parametrize(
        ("n_atoms", "geometry", "cols"),
        [(4, "ring", None), (6, "ring", None), (4, "grid", 2), (6, "grid", 3)],
    )
    def test_where_the_bonds_are_bipartite_one_latent_leaves_the_label_at_a_half(
        self, n_atoms: int, geometry: str, cols: int | None
    ) -> None:
        """``P(y = 1 | z_i = t) = ½`` for every atom and value, so every function of it is blind."""
        bonds = bench.nearest_neighbour_bonds(n_atoms, geometry, cols)
        rng = np.random.default_rng(5)
        m = 40000  # standard error of a rate of ½: 0.0025
        for i in range(n_atoms):
            for t in (0.0, 0.5, 2.0):
                latent = rng.standard_normal((m, n_atoms))
                latent[:, i] = t
                assert abs(bench.product_labels(latent, bonds).mean() - 0.5) < 0.0125

    def test_the_triangle_has_fewer_positives_and_one_latent_moves_the_label(self) -> None:
        """
        A ring of three: ``S = w_1² − (w_2² + w_3²)/2`` in the eigenbasis of the
        adjacency (eigenvalues 2, −1, −1), so ``P(y = 1) = 1 − E[e^{−w²}] = 1 − 1/√3``.
        Given ``z_0 = t`` the label is ``(z_1 + t)(z_2 + t) > t²``; its rate, by SciPy
        quadrature, is ½ at ``t = 0`` and below it elsewhere, so ``P(y | z_0)`` is not
        constant.  The class means of ``z_0`` and of ``z_0²`` still coincide.
        """
        stats = pytest.importorskip("scipy.stats")
        integrate = pytest.importorskip("scipy.integrate")

        def rate(t: float) -> float:
            # P(pq > t²) with p, q ~ N(t, 1): q > t²/p for p > 0, q < t²/p for p < 0
            above = integrate.quad(
                lambda p: stats.norm.pdf(p - t) * stats.norm.cdf(t - t * t / p), 0, np.inf
            )[0]
            below = integrate.quad(
                lambda p: stats.norm.pdf(p - t) * stats.norm.cdf(t * t / p - t), -np.inf, 0
            )[0]
            return float(above + below)

        ring = bench.nearest_neighbour_bonds(3, "ring")
        rng = np.random.default_rng(3)
        latent = rng.standard_normal((400000, 3))
        y = bench.product_labels(latent, ring)
        assert y.mean() == pytest.approx(1 - 1 / math.sqrt(3), abs=0.004)  # 0.4226; SE 0.0008
        for t in (0.5, 1.0, 2.0):
            assert rate(t) < 0.45  # 0.397, 0.399, 0.438
            fixed = rng.standard_normal((200000, 3))
            fixed[:, 0] = t
            assert bench.product_labels(fixed, ring).mean() == pytest.approx(rate(t), abs=0.005)
        fixed = rng.standard_normal((200000, 3))
        fixed[:, 0] = 0.0
        assert bench.product_labels(fixed, ring).mean() == pytest.approx(0.5, abs=0.005)
        # z_0 and z_0²: the global flip, and a label that depends on the direction of z only
        for g in (latent[:, 0], latent[:, 0] ** 2):
            one, zero = g[y == 1], g[y == 0]
            error = math.sqrt(one.var() / len(one) + zero.var() / len(zero))
            assert abs(one.mean() - zero.mean()) < 4 * error


CHURN_HEADER = [
    "Call  Failure",
    "Complains",
    "Subscription  Length",
    "Charge  Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
    "Status",
    "Age",
    "Customer Value",
    "Churn",
]


def _write_churn(
    directory: Path, n: int = 96, n_pos: int = 24, repeated: slice | None = None
) -> Path:
    """
    A small file in the format of UCI's ``Customer Churn.csv``; the dataset is not fetched.

    ``repeated`` appends an exact copy of those rows at the end, as the published
    file repeats 300 earlier rows.
    """
    rng = np.random.default_rng(7)
    data = rng.integers(0, 500, (n, 13)).astype(float)
    labels = np.zeros(n, dtype=int)
    labels[:n_pos] = 1
    data[labels == 1, :3] += 150.0  # churners differ, so there is something to learn
    if repeated is not None:
        data = np.concatenate([data, data[repeated]])
        labels = np.concatenate([labels, labels[repeated]])
    lines = [",".join(CHURN_HEADER)]
    lines += [",".join(f"{v:g}" for v in row) + f",{label}" for row, label in zip(data, labels)]
    path = directory / "Customer Churn.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestUciData:
    def test_a_uci_loader_runs_through_the_command_line(self, tmp_path: Path) -> None:
        _write_churn(tmp_path)
        out = tmp_path / "churn.jsonl"
        arguments = ["--dataset", "iranian-churn", "--data-path", str(tmp_path), "--no-cache"]
        arguments += ["--arms", "C", "--folds", "3", "--max-folds", "1", "--trials", "0"]
        arguments += ["--epochs", "3", "--out", str(out)]
        records = bench.main(arguments)
        assert [(r["arm"], r["head"]) for r in records] == [("C", "linear"), ("C", "mlp")]
        assert not (tmp_path / "rydberg_features_cache").exists()

        data = load_iranian_churn(tmp_path)
        settings = bench.Settings(n_splits=3)
        fold = _fold_by_hand(data.X, data.y, settings, bench.DATASETS.index("iranian-churn"))
        for record in bench.read_records(out):
            assert record["dataset"] == "iranian-churn"
            assert record["data_sha256"] == fingerprint(data.X, data.y)
            assert (record["n_samples"], record["n_positives"]) == (96, 24)
            assert record["n_raw_features"] == 13
            assert record["seeds"]["data_seed"] is None
            assert record["test_idx"] == np.sort(fold.test_idx).tolist()
            assert record["tuning"] is None
            assert record["hyperparameters"]["max_epochs"] == 3
            # The 13 columns reduced to 4 bounded inputs, fitted on the training part.
            by_hand = separation_measures(
                fold.inputs[fold.test_idx], data.y[fold.test_idx]
            ).to_dict()
            assert record["separation"]["test"] == pytest.approx(by_hand, rel=1e-9)

    def test_drop_duplicates_reaches_the_churn_loader_only(self, tmp_path: Path) -> None:
        """#704: rows 20 to 27 (4 churners, 4 not) are repeated at the end of the file."""
        _write_churn(tmp_path, repeated=slice(20, 28))
        X, y, _ = bench.load_dataset("iranian-churn", bench.Settings(), path=tmp_path)
        assert (y.size, int(y.sum())) == (104, 28)
        settings = bench.Settings(drop_duplicates=True)
        X, y, _ = bench.load_dataset("iranian-churn", settings, path=tmp_path)
        assert (y.size, int(y.sum())) == (96, 24)
        unique = load_iranian_churn(tmp_path, drop_duplicates=True)
        np.testing.assert_array_equal(X, unique.X)
        np.testing.assert_array_equal(y, unique.y)
        # The positive control has no duplicates to drop and is drawn as before.
        control, _, _ = bench.load_dataset("positive-control", settings)
        default_control, _, _ = bench.load_dataset("positive-control", bench.Settings())
        np.testing.assert_array_equal(control, default_control)

    @pytest.mark.parametrize("drop", [False, True])
    def test_drop_duplicates_on_the_command_line_changes_the_lines(
        self, tmp_path: Path, drop: bool
    ) -> None:
        _write_churn(tmp_path, repeated=slice(20, 28))
        out = tmp_path / "churn.jsonl"
        arguments = ["--dataset", "iranian-churn", "--data-path", str(tmp_path), "--no-cache"]
        arguments += ["--arms", "C", "--folds", "3", "--max-folds", "1", "--trials", "0"]
        arguments += ["--epochs", "2", "--out", str(out)]
        bench.main(arguments + (["--drop-duplicates"] if drop else []))
        data = load_iranian_churn(tmp_path, drop_duplicates=drop)
        records = bench.read_records(out)
        assert records
        for record in records:
            assert record["settings"]["drop_duplicates"] is drop
            assert (record["n_samples"], record["n_positives"]) == (
                (96, 24) if drop else (104, 28)
            )
            assert record["data_sha256"] == fingerprint(data.X, data.y)

    def test_a_dataset_name_outside_datasets_is_refused_with_the_names_allowed(self) -> None:
        """#540: it used to raise ``tuple.index(x): x not in tuple`` from the seeds."""
        message = (
            "unknown dataset 'mnist'; choose from "
            "['positive-control', 'iranian-churn', 'taiwanese-bankruptcy', "
            "'cervical-cancer-risk']."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.fold_seeds(0, "mnist", 3)
        X = np.random.default_rng(0).normal(size=(40, 5))
        y = np.array([0, 1] * 20)
        records = bench.run_dataset(
            "mnist", X, y, bench.Settings(arms=("C",)), bench.FeatureCache(None)
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            next(records)
        # The names in DATASETS keep the seeds their position gives them.
        root = np.random.SeedSequence([0, bench.DATASETS.index("iranian-churn")])
        assert bench.fold_seeds(0, "iranian-churn", 3)[0] == int(
            root.spawn(4)[0].generate_state(1)[0]
        )

    def test_too_few_columns_for_the_atoms(self) -> None:
        X = np.random.default_rng(0).normal(size=(40, 3))
        y = np.array([0, 1] * 20)
        records = bench.run_dataset(
            "iranian-churn", X, y, bench.Settings(), bench.FeatureCache(None)
        )
        with pytest.raises(ValueError, match="cannot be reduced to n_atoms=4"):
            next(records)


# ---------------------------------------------------------------------------
# The command line: what a failing or noisy run leaves behind
# ---------------------------------------------------------------------------


@pytest.fixture
def probe_prefixes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the prefix of each temporary file the writability probe creates."""
    prefixes: list[str] = []
    temporary_file = bench.tempfile.NamedTemporaryFile

    def probe(*args: Any, **kwargs: Any) -> Any:
        prefixes.append(kwargs["prefix"])
        return temporary_file(*args, **kwargs)

    monkeypatch.setattr(bench.tempfile, "NamedTemporaryFile", probe)
    return prefixes


class TestCheckWritable:
    @pytest.mark.may_skip
    @pytest.mark.skipif(os.name != "posix", reason="named pipes require POSIX")
    def test_a_fifo_probe_returns_without_a_reader(self, tmp_path: Path) -> None:
        """Exercise the real probe, with a deadline even if it starts opening the FIFO."""
        out = tmp_path / "log"
        os.mkfifo(out)
        result: Future[None] = Future()

        def check() -> None:
            try:
                bench.check_writable(out)
            except BaseException as error:  # noqa: BLE001 - re-raise in the test thread
                result.set_exception(error)
            else:
                result.set_result(None)

        worker = Thread(target=check, daemon=True)
        worker.start()
        try:
            result.result(timeout=2)
        finally:
            if not result.done():
                # Release a writer blocked by a regression so it cannot leak into later tests.
                reader = os.open(out, os.O_RDONLY | os.O_NONBLOCK)
                try:
                    worker.join(timeout=2)
                finally:
                    os.close(reader)
            else:
                worker.join(timeout=2)
        assert not worker.is_alive()

    @pytest.mark.may_skip
    @pytest.mark.skipif(os.name != "posix", reason="named pipes require POSIX")
    @pytest.mark.parametrize("symlink", [False, True])
    @pytest.mark.parametrize("writable", [False, True])
    @pytest.mark.parametrize("effective_ids", [False, True])
    def test_a_fifo_is_checked_without_opening_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        symlink: bool,
        writable: bool,
        effective_ids: bool,
    ) -> None:
        """#625: opening a FIFO without a reader would hang instead of failing the test."""
        fifo = tmp_path / "log"
        os.mkfifo(fifo)
        out = fifo
        if symlink:
            out = tmp_path / "link"
            out.symlink_to(fifo)

        def refuse_open(*args: Any, **kwargs: Any) -> None:
            pytest.fail("the writability probe opened a FIFO")

        access: list[tuple[Path, int, bool]] = []

        def can_write(path: Path, mode: int, *, effective_ids: bool) -> bool:
            access.append((path, mode, effective_ids))
            return writable

        monkeypatch.setattr(bench, "open", refuse_open, raising=False)
        monkeypatch.setattr(bench.os, "access", can_write)
        monkeypatch.setattr(
            bench.os, "supports_effective_ids", {can_write} if effective_ids else set()
        )
        if writable:
            bench.check_writable(out)
        else:
            message = f"out {str(out)!r} cannot be written: {os.strerror(errno.EACCES)}."
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                bench.check_writable(out)
        assert access == [(out, os.W_OK, effective_ids)]

    @pytest.mark.may_skip
    @pytest.mark.skipif(os.name != "posix", reason="Unix sockets require POSIX")
    def test_a_socket_is_refused_before_any_dataset_is_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "socket"
        loaded = TestCommandLine._spy_on_loading(monkeypatch)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(out))
            out.chmod(0o666)
            message = f"out {str(out)!r} cannot be written:"
            with pytest.raises(ValueError, match=f"^{re.escape(message)}"):
                bench.main(["--quick", "--out", str(out), "--no-cache"])
        assert loaded == []

    @pytest.mark.may_skip
    @pytest.mark.skipif(os.name != "posix", reason="requires a POSIX null device")
    def test_a_device_is_checked_without_opening_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse_open(*args: Any, **kwargs: Any) -> None:
            pytest.fail("the writability probe opened a device")

        monkeypatch.setattr(bench, "open", refuse_open, raising=False)
        bench.check_writable(os.devnull)

    @pytest.mark.may_skip
    def test_a_250_byte_filename_is_accepted(self, tmp_path: Path) -> None:
        out = tmp_path / ("a" * 250)
        # Verify that this filesystem supports the name before testing the probe.
        try:
            out.touch()
        except OSError as error:
            if error.errno == errno.ENAMETOOLONG:
                pytest.skip("the filesystem does not support a 250-byte basename")
            raise
        out.unlink()
        bench.check_writable(out)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("name", ["a" * 15, "a" * 16, "a" * 17, "é" * 9])
    def test_the_filename_limit_counts_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        calls: list[tuple[Path, str]] = []

        def pathconf(path: Path, key: str) -> int:
            calls.append((path, key))
            return 16

        monkeypatch.setattr(bench.os, "pathconf", pathconf, raising=False)
        out = tmp_path / name
        if len(os.fsencode(name)) <= 16:
            bench.check_writable(out)
        else:
            message = f"out {str(out)!r} cannot be written: {os.strerror(errno.ENAMETOOLONG)}."
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                bench.check_writable(out)
        assert calls == [(tmp_path, "PC_NAME_MAX")]
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        ("name", "prefix"),
        [("log", ".log."), ("a" * 6, "." + "a" * 6 + "."), ("a" * 7, ".out."), ("é" * 4, ".out.")],
    )
    def test_the_probe_keeps_the_basename_when_it_fits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        probe_prefixes: list[str],
        name: str,
        prefix: str,
    ) -> None:
        monkeypatch.setattr(bench.os, "pathconf", lambda path, key: 16, raising=False)
        bench.check_writable(tmp_path / name)
        assert probe_prefixes == [prefix]
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "limit", [None, OSError(errno.EIO, "unsupported"), ValueError(), -1, 0]
    )
    def test_an_unavailable_filename_limit_still_probes_the_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        probe_prefixes: list[str],
        limit: Exception | int | None,
    ) -> None:
        def pathconf(path: Path, key: str) -> int:
            if isinstance(limit, Exception):
                raise limit
            assert isinstance(limit, int)
            return limit

        if limit is None:
            monkeypatch.delattr(bench.os, "pathconf", raising=False)
        else:
            monkeypatch.setattr(bench.os, "pathconf", pathconf, raising=False)
        bench.check_writable(tmp_path / "new.jsonl")
        assert probe_prefixes == [".new.jsonl."]
        assert list(tmp_path.iterdir()) == []
        with pytest.raises(ValueError, match="cannot be written"):
            bench.check_writable(tmp_path / "missing" / "new.jsonl")
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        ("name", "prefix"),
        [
            ("a" * 250, ".out."),
            # An unpaired surrogate must not crash the measurement.
            ("x\udc80" + "a" * 240, "." + "x\udc80" + "a" * 240 + "."),
        ],
    )
    def test_windows_without_pathconf_limits_names_to_255_utf16_units(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        probe_prefixes: list[str],
        name: str,
        prefix: str,
    ) -> None:
        monkeypatch.setattr(bench, "_WINDOWS", True)
        monkeypatch.delattr(bench.os, "pathconf", raising=False)
        bench.check_writable(tmp_path / name)
        assert probe_prefixes == [prefix]
        assert list(tmp_path.iterdir()) == []

    def test_windows_measures_names_in_utf16_code_units(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bench, "_WINDOWS", True)
        # 250 UTF-16 code units, although 500 bytes in UTF-8.
        assert bench._filename_length("é" * 250) == 250
        assert bench._filename_length("\U0001f600" * 100) == 200
        assert bench._filename_length("x\udc80") == 2

    def test_windows_without_pathconf_refuses_a_name_over_255_utf16_units(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe_prefixes: list[str]
    ) -> None:
        monkeypatch.setattr(bench, "_WINDOWS", True)
        monkeypatch.delattr(bench.os, "pathconf", raising=False)
        out = tmp_path / ("a" * 256)
        message = f"out {str(out)!r} cannot be written: {os.strerror(errno.ENAMETOOLONG)}."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.check_writable(out)
        assert probe_prefixes == []
        assert list(tmp_path.iterdir()) == []


class TestCommandLine:
    @pytest.mark.parametrize(
        ("flags", "match"),
        [
            (["--shots", "0"], "shots"),
            (["--shots", "0", "--arms", "A", "B"], "shots"),
            (["--n-steps", "0"], "n_steps"),
            (["--hidden", "0"], "hidden"),
            (["--circuit-layers", "0"], "circuit_layers"),
            (["--n-atoms", "11"], "n_atoms"),
            (["--epochs", "0"], "max_epochs"),
            # with tuning on, the search space used to override both and the run completed
            (["--batch-size", "0"], "batch_size"),
            (["--lr", "0"], "lr must be finite and > 0"),
            (["--lr", "inf"], "lr must be finite and > 0"),
            (["--n-atoms", "13", "--arms", "C", "D"], "n_atoms"),
            # refused for the positive control before any dataset is loaded (2^N - 1 columns)
            (["--n-atoms", "13", "--arms", "C"], "n_atoms must be <= 12 with the positive"),
            # refused by the splits, with the data in hand: no record, so no new file
            (["--folds", "100"], "n_splits"),
            (["--inner-folds", "100"], "n_splits"),
        ],
    )
    def test_a_refused_setting_leaves_an_earlier_results_file_alone(
        self, tmp_path: Path, flags: list[str], match: str
    ) -> None:
        out = tmp_path / "results.jsonl"
        out.write_text("an earlier run\n", encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            bench.main([*flags, "--samples", "40", "--out", str(out), "--no-cache"])
        assert out.read_text(encoding="utf-8") == "an earlier run\n"

    def test_a_run_that_fails_later_keeps_the_records_it_wrote(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The file is opened with the first record and holds every record up to the failure."""
        out = tmp_path / "results.jsonl"
        out.write_text("an earlier run\n", encoding="utf-8")

        def run_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
            yield {"dataset": name, "mcc": 0.5}
            raise RuntimeError("the second fold failed")

        monkeypatch.setattr(bench, "run_dataset", run_dataset)
        with pytest.raises(RuntimeError, match="second fold"):
            bench.run({"a": (np.zeros((2, 2)), np.zeros(2), None)}, QUICK, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == [{"dataset": "a", "mcc": 0.5}]

    @staticmethod
    def _spy_on_loading(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        loaded: list[str] = []
        load = bench.load_dataset

        def load_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
            loaded.append(name)
            return load(name, *args, **kwargs)

        monkeypatch.setattr(bench, "load_dataset", load_dataset)
        return loaded

    @pytest.mark.parametrize(
        "case",
        [
            "missing directory",
            "a directory",
            # skips where the user is root, who may write to it anyway
            pytest.param("read-only file", marks=pytest.mark.may_skip),
        ],
    )
    def test_an_unwritable_out_is_refused_before_any_dataset_is_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
    ) -> None:
        """#540: ``--quick --out missing/x.jsonl`` used to train the first arm, then raise."""
        if case == "missing directory":
            out, reason = tmp_path / "missing" / "x.jsonl", "No such file or directory"
        elif case == "a directory":
            out, reason = tmp_path / "results", "Is a directory"
            out.mkdir()
            (out / "earlier.jsonl").write_text("an earlier run\n", encoding="utf-8")
        else:
            if os.name != "posix" or os.geteuid() == 0:
                pytest.skip("root writes to a read-only file")
            out, reason = tmp_path / "results.jsonl", "Permission denied"
            out.write_text("an earlier run\n", encoding="utf-8")
            out.chmod(0o444)
        before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
        loaded = self._spy_on_loading(monkeypatch)
        message = f"out {str(out)!r} cannot be written: {reason}."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.main(["--quick", "--out", str(out), "--no-cache"])
        assert loaded == []
        # Nothing was created, removed or changed: no stray probe file, the earlier file intact.
        assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before
        if case == "a directory":
            assert (out / "earlier.jsonl").read_text(encoding="utf-8") == "an earlier run\n"
        if case == "read-only file":
            assert out.read_text(encoding="utf-8") == "an earlier run\n"

    def test_run_refuses_an_unwritable_out_before_the_first_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started: list[str] = []

        def run_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
            started.append(name)
            yield {"dataset": name}

        monkeypatch.setattr(bench, "run_dataset", run_dataset)
        out = tmp_path / "missing" / "x.jsonl"
        message = f"out {str(out)!r} cannot be written: No such file or directory."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.run({"a": (np.zeros((2, 2)), np.zeros(2), None)}, QUICK, out)
        assert started == []
        assert list(tmp_path.iterdir()) == []

    def test_checking_a_writable_out_changes_nothing(self, tmp_path: Path) -> None:
        """The probe neither creates ``out`` nor leaves a file beside it, nor touches an earlier one."""
        new = tmp_path / "new.jsonl"
        bench.check_writable(new)
        assert list(tmp_path.iterdir()) == []
        earlier = tmp_path / "earlier.jsonl"
        earlier.write_bytes(b"an earlier run\n")
        stat = earlier.stat()
        bench.check_writable(earlier)
        assert earlier.read_bytes() == b"an earlier run\n"
        assert earlier.stat().st_mtime_ns == stat.st_mtime_ns
        assert sorted(p.name for p in tmp_path.iterdir()) == ["earlier.jsonl"]

    # Skips only when Windows requires a symlink privilege the user lacks.
    @pytest.mark.may_skip
    def test_a_broken_symlink_out_is_refused(self, tmp_path: Path) -> None:
        """Its target's missing directory is checked, not the symlink's own one."""
        out = tmp_path / "results.jsonl"
        try:
            out.symlink_to(tmp_path / "gone" / "results.jsonl")
        except NotImplementedError as error:
            pytest.skip(f"symlink creation is unavailable: {error}")
        except OSError as error:
            if getattr(error, "winerror", None) != 1314:
                raise
            pytest.skip("creating a symlink needs Developer Mode or SeCreateSymbolicLinkPrivilege")

        message = f"out {str(out)!r} cannot be written: No such file or directory."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.check_writable(out)

    def test_arm_d_above_its_bound_is_refused_before_any_dataset_is_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded = self._spy_on_loading(monkeypatch)
        out = tmp_path / "results.jsonl"
        out.write_text("an earlier run\n", encoding="utf-8")
        message = "n_atoms must be >= 1, <= 10 with arm A or B and <= 12 with arm D; got 13."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.main(["--arms", "C", "D", "--n-atoms", "13", "--out", str(out), "--no-cache"])
        assert loaded == []
        assert out.read_text(encoding="utf-8") == "an earlier run\n"

    def test_the_positive_control_above_its_bound_is_refused_before_anything_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """#540: with arm C alone no arm bounds ``n_atoms``; the control's 2^N − 1 columns do."""
        started: list[str] = []

        def run_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
            started.append(name)
            yield from ()

        monkeypatch.setattr(bench, "run_dataset", run_dataset)
        out = tmp_path / "results.jsonl"
        out.write_text("an earlier run\n", encoding="utf-8")
        message = "n_atoms must be <= 12 with the positive control (2^N - 1 raw columns); got 13."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.main(
                ["--arms", "C", "--n-atoms", "13", "--samples", "40", "--out", str(out)]
                + ["--no-cache"]
            )
        assert started == []
        assert out.read_text(encoding="utf-8") == "an earlier run\n"

    def test_the_positive_control_bound_is_checked_before_a_dataset_listed_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A UCI file listed before the control is not read, or downloaded, for a refused run."""
        loaded = self._spy_on_loading(monkeypatch)
        message = "n_atoms must be <= 12 with the positive control (2^N - 1 raw columns); got 13."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bench.main(
                ["--arms", "C", "--n-atoms", "13", "--dataset", "iranian-churn"]
                + ["positive-control", "--download", "--out", str(tmp_path / "r.jsonl")]
                + ["--no-cache"]
            )
        assert loaded == []

    @pytest.mark.parametrize(("flag", "patience"), [("0", None), ("1", 1), ("7", 7)])
    def test_a_patience_of_0_is_no_early_stopping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, patience: int | None
    ) -> None:
        """As ``--trials`` and ``--max-separation-rows``: ``0`` is "none", ``None`` here."""
        captured: dict[str, Any] = {}

        def run(datasets: Any, settings: Any, out: Any, **kwargs: Any) -> list[dict[str, Any]]:
            captured["settings"] = settings
            return []

        monkeypatch.setattr(bench, "run", run)
        bench.main(["--patience", flag, "--out", str(tmp_path / "out.jsonl"), "--no-cache"])
        assert captured["settings"].patience == patience

    @pytest.mark.parametrize("quick", [[], ["--quick"]])
    @pytest.mark.parametrize("value", ["-1", "-10"])
    def test_a_negative_patience_is_refused_before_any_dataset_is_loaded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        quick: list[str],
        value: str,
    ) -> None:
        """0 is the one spelling of "no early stopping": a negative value is a typo."""
        loaded: list[str] = []

        # Fails fast if the value gets through: without --quick a load would be the full benchmark.
        def load_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
            loaded.append(name)
            raise AssertionError(f"{name} was loaded")

        monkeypatch.setattr(bench, "load_dataset", load_dataset)
        out = tmp_path / "results.jsonl"
        out.write_text("an earlier run\n", encoding="utf-8")
        with pytest.raises(SystemExit) as raised:
            bench.main([*quick, "--patience", value, "--out", str(out), "--no-cache"])
        assert raised.value.code == 2
        error = capsys.readouterr().err.splitlines()[-1]
        assert error.endswith(
            f"error: argument --patience: must be 0 (no early stopping) or an integer >= 1; "
            f"got {value}"
        )
        assert loaded == []
        assert out.read_text(encoding="utf-8") == "an earlier run\n"

    def test_without_early_stopping_every_epoch_runs_and_patience_is_recorded_as_null(
        self, tmp_path: Path
    ) -> None:
        # More epochs than the default patience (10), which stops these heads after 11.
        assert bench.Settings().patience == 10
        out = tmp_path / "results.jsonl"
        arguments = ["--patience", "0", "--arms", "C", "--trials", "0", "--folds", "2"]
        arguments += ["--max-folds", "1", "--epochs", "15", "--samples", "80"]
        records = bench.main([*arguments, "--out", str(out), "--no-cache"])
        assert [(r["arm"], r["head"]) for r in records] == [("C", "linear"), ("C", "mlp")]
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        for line in lines:
            assert line["settings"]["patience"] is None
            assert line["hyperparameters"]["patience"] is None
            assert line["epochs"] == 15  # train_model stops early only with a patience
        assert '"patience": null' in out.read_text(encoding="utf-8")

    def test_main_hides_no_warning_of_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A warning is how a suspect number announces itself: NumPy's
        invalid-value and overflow notes, and ``PCANormalizer``'s degenerate
        eigenvalues, after which the component that drives an atom is
        arbitrary.  ``main`` shows each of them, and a fit with distinct
        eigenvalues adds none.
        """
        cross = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]).repeat(3, axis=0)
        distinct = cross * np.array([2.0, 1.0])

        def run(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            warnings.warn("a device fallback", UserWarning, stacklevel=1)
            assert math.isnan(np.log(np.float64(-1.0)))
            assert math.isinf(np.float64(1e308) * np.float64(10.0))
            PCANormalizer(n_components=2).fit(distinct)
            PCANormalizer(n_components=2).fit(cross)  # both eigenvalues are 6/11
            return []

        monkeypatch.setattr(bench, "run", run)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bench.main(["--quick", "--out", str(tmp_path / "out.jsonl"), "--no-cache"])
        shown = [(w.category, str(w.message)) for w in caught]
        assert [category for category, _ in shown] == [UserWarning] + [RuntimeWarning] * 3
        assert shown[0][1] == "a device fallback"
        assert shown[1][1] == "invalid value encountered in log"
        assert "overflow encountered" in shown[2][1]
        assert "are degenerate" in shown[3][1]
        assert "not reproducible" in shown[3][1]

    def test_the_parser_builds_without_a_docstring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``python -OO`` strips the module docstring the description is taken from."""
        assert bench._parser().description.startswith("``RydbergFeatureMap``")
        monkeypatch.setattr(bench, "__doc__", None)
        parser = bench._parser()
        assert parser.description is None
        assert parser.parse_args(["--quick"]).quick is True

    def test_every_parser_option_has_help(self) -> None:
        """Every optional argument of _parser() must carry a non-empty help string (#526).

        ``ArgumentDefaultsHelpFormatter`` only prints a default next to an option
        that already has a ``help=`` string; without one, both the description and
        the default are invisible to users running ``--help``.
        """
        parser = bench._parser()
        missing = [
            action.option_strings
            for action in parser._actions
            if action.option_strings  # skip the positional catch-all action
            and action.help in (None, argparse.SUPPRESS, "")
        ]
        assert missing == [], f"Options without a help string: {missing}"

    def test_quick_help_clarifies_overrides_and_effective_options(self) -> None:
        """--quick help must document overrides and preserved options (#549)."""
        parser = bench._parser()
        # Inspect the help text belonging specifically to the --quick action,
        # so assertions do not pass merely because options appear elsewhere in --help.
        quick_action = next(a for a in parser._actions if "--quick" in a.option_strings)
        assert isinstance(quick_action.help, str)
        quick_help = quick_action.help

        # Predefined QUICK smoke-test configuration
        assert "QUICK" in quick_help
        assert "smoke test" in quick_help

        # Positive control
        assert "positive control" in quick_help

        # Other applicable command-line settings are overridden or ignored
        assert "overrid" in quick_help.lower() or "ignor" in quick_help.lower()

        # Options that remain effective are explicitly listed in --quick's help text
        effective_options = [
            "--out",
            "--cache-dir",
            "--no-cache",
            "--seed",
            "--max-separation-rows",
            "--separation-seed",
        ]
        for opt in effective_options:
            assert opt in quick_help

    def test_quick_behavior_overrides_conflicting_options_and_preserves_effective_ones(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--quick must select positive-control and predefined QUICK settings (#549)."""
        captured: dict[str, Any] = {}

        def mock_run(
            datasets: Any,
            settings: Any,
            out: Any,
            *,
            cache_dir: Any = None,
        ) -> list[dict[str, Any]]:
            captured["datasets"] = datasets
            captured["settings"] = settings
            captured["out"] = out
            captured["cache_dir"] = cache_dir
            return []

        monkeypatch.setattr(bench, "run", mock_run)

        out_path = tmp_path / "custom_out.jsonl"
        cache_dir = tmp_path / "custom_cache"

        bench.main(
            [
                "--quick",
                "--n-atoms",
                "6",
                "--patience",
                "5",
                "--dataset",
                "iranian-churn",
                "--seed",
                "7",
                "--separation-seed",
                "13",
                "--out",
                str(out_path),
                "--cache-dir",
                str(cache_dir),
            ]
        )

        # Selected dataset must only be positive-control
        assert list(captured["datasets"].keys()) == ["positive-control"]

        # Settings must equal predefined QUICK with only preserved seed fields changed
        expected_settings = _replace(QUICK, random_state=7, separation_seed=13)
        assert captured["settings"] == expected_settings

        # Conflicting options must not have been honored
        assert captured["settings"].n_atoms == QUICK.n_atoms
        assert captured["settings"].n_atoms != 6
        assert captured["settings"].patience == QUICK.patience
        assert captured["settings"].patience != 5

        # Explicit out and cache_dir paths are preserved
        assert captured["out"] == out_path
        assert captured["cache_dir"] == cache_dir


# ---------------------------------------------------------------------------
# Inputs: fitted on the training rows, oversampled once
# ---------------------------------------------------------------------------


class TestInputs:
    def _data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(4)
        X = rng.normal(size=(120, 7)) * np.array([1.0, 50.0, 0.01, 3.0, 1.0, 1.0, 9.0])
        X[:, 5] = 2.5  # a constant column, as in the bankruptcy data
        y = (rng.random(120) < 0.25).astype(np.int64)
        return X, y

    def test_inputs_are_bounded_and_fitted_on_the_given_rows_only(self) -> None:
        X, _ = self._data()
        fit_rows = np.arange(0, 90)
        inputs = bench.preprocess(X, fit_rows, 4)
        assert inputs.shape == (120, 4)
        assert inputs.dtype == np.float64
        assert np.abs(inputs).max() <= float(np.float32(math.pi))
        # Changing rows outside the fit changes no other row's inputs.
        changed = X.copy()
        changed[90:] = changed[90:] * 3.0 + 1.0
        again = bench.preprocess(changed, fit_rows, 4)
        np.testing.assert_array_equal(again[:90], inputs[:90])
        assert np.abs(again[90:] - inputs[90:]).max() > 0.1
        # ... and a change inside it does.
        changed = X.copy()
        changed[:10] += 1.0
        assert np.abs(bench.preprocess(changed, fit_rows, 4)[90:] - inputs[90:]).max() > 1e-3

    def test_inputs_are_standardised_components_squashed_to_pi(self) -> None:
        """By hand: standardise, project on the leading eigenvectors, ``π tanh(· / std)``."""
        from sklearn.preprocessing import StandardScaler

        X, _ = self._data()
        rows = np.arange(0, 90)
        # These columns have enough variation that the tighter bound of
        # ``standardise_features`` and ``StandardScaler`` agree (#580).
        scaled = StandardScaler().fit(X[rows]).transform(X)
        centred = scaled[rows] - scaled[rows].mean(0)
        values, vectors = np.linalg.eigh(np.cov(centred, rowvar=False))
        leading = vectors[:, np.argsort(values)[::-1][:4]]
        leading = leading * np.sign(leading[np.abs(leading).argmax(0), np.arange(4)])
        projected = (scaled - scaled[rows].mean(0)) @ leading
        by_hand = math.pi * np.tanh(projected / projected[rows].std(0, ddof=1))
        np.testing.assert_allclose(bench.preprocess(X, rows, 4), by_hand, rtol=0, atol=2e-6)

    def test_a_column_constant_up_to_rounding_is_only_centred(self) -> None:
        """
        A raw column constant on the fitting rows up to rounding enters no component (#580).

        Column 3 holds ``1.5`` and its float64 neighbour ``1.5 + 2⁻⁵²``, the
        values a sum that is 1.5 in exact arithmetic leaves after rounding,
        written out so they depend on no random stream or summation order:
        a σ of about ``1e-16``, not 0, and a variance (about ``1e-32``) below
        the rounding bound of :func:`standardise_features`
        (``(2 ceil(log2(n)) ε μ)² ≈ 3e-29`` for 150 rows).
        The other three columns, standardised, span a plane plus a direction
        of variance ``2.4e-5``, so with three components the third is that
        small direction.  The rounding noise, scaled to unit variance by the
        ``σ == 0`` rule, would take that slot instead: an independent column
        of variance 1 against ``2.4e-5``.

        Expected value by hand: the columns scaled by scikit-learn's
        ``StandardScaler`` (which centres the rounding column only), then the
        principal components with ``numpy.linalg.eigh``.  It must also equal
        :func:`preprocess` on the three other columns alone.
        """
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(7)
        a, b = rng.standard_normal((2, 200))
        rounding = np.where(np.arange(200) % 3 == 0, np.nextafter(1.5, 2.0), 1.5)
        X = np.column_stack([a, b, a + b + 0.01 * rng.standard_normal(200), rounding])
        rows = np.arange(150)
        # The premise: σ is not 0 (so ``σ == 0`` scales it), but the column is
        # constant to scikit-learn as well.
        assert 0 < X[rows, 3].std() < 1e-15
        assert StandardScaler().fit(X[rows]).scale_[3] == 1.0

        scaled = StandardScaler().fit(X[rows]).transform(X)
        centred = scaled[rows] - scaled[rows].mean(0)
        values, vectors = np.linalg.eigh(np.cov(centred, rowvar=False))
        leading = vectors[:, np.argsort(values)[::-1][:3]]
        leading = leading * np.sign(leading[np.abs(leading).argmax(0), np.arange(3)])
        assert np.abs(leading[3]).max() < 1e-12  # the column loads on no component
        projected = (scaled - scaled[rows].mean(0)) @ leading
        # ``PCANormalizer`` adds 1e-8 to each component's σ; on the third,
        # σ ≈ 4.9e-3, that is a relative 2e-6, above the float32 tolerance.
        by_hand = math.pi * np.tanh(projected / (projected[rows].std(0, ddof=1) + 1e-8))

        inputs = bench.preprocess(X, rows, 3)
        np.testing.assert_allclose(inputs, by_hand, rtol=0, atol=2e-6)
        np.testing.assert_allclose(inputs, bench.preprocess(X[:, :3], rows, 3), rtol=0, atol=2e-6)

    def test_a_column_constant_up_to_rounding_adds_no_rank(self) -> None:
        """
        Only centred, such a column adds nothing to the centred rank (#580).

        Two independent columns and one of ``1.5`` and ``1.5 + 2⁻⁵²`` have
        centred rank 2, so three components are refused rather than filled
        with the rounding noise, which the ``σ == 0`` rule scaled to unit
        variance and took as the third input.  Two components still work.
        """
        rng = np.random.default_rng(7)
        a, b = rng.standard_normal((2, 200))
        rounding = np.where(np.arange(200) % 3 == 0, np.nextafter(1.5, 2.0), 1.5)
        X = np.column_stack([a, b, rounding])
        rows = np.arange(150)
        assert 0 < X[rows, 2].std() < 1e-15
        with pytest.raises(ValueError, match=r"rank 2 < n_components=3"):
            bench.preprocess(X, rows, 3)
        np.testing.assert_allclose(
            bench.preprocess(X, rows, 2), bench.preprocess(X[:, :2], rows, 2), rtol=0, atol=2e-6
        )

    def test_split_blocks_and_synthetic_rows(self) -> None:
        X, y = self._data()
        train, val, test = np.arange(0, 70), np.arange(70, 90), np.arange(90, 120)
        fit_rows = np.arange(0, 90)
        split = bench.make_split(X, y, fit_rows, train, val, test, n_components=4, smote_seed=3)
        inputs = bench.preprocess(X, fit_rows, 4)
        n_pos = int(y[train].sum())
        n_synthetic = (70 - n_pos) - n_pos  # SMOTE balances the training rows
        assert (split.n_train, split.n_synthetic) == (70 + n_synthetic, n_synthetic)
        assert (split.n_val, split.n_test) == (20, 30)
        np.testing.assert_array_equal(split.inputs[:70], inputs[train])
        np.testing.assert_array_equal(split.labels[:70], y[train])
        np.testing.assert_array_equal(split.inputs[split.val], inputs[val])
        np.testing.assert_array_equal(split.labels[split.val], y[val])
        np.testing.assert_array_equal(split.inputs[split.test], inputs[test])
        np.testing.assert_array_equal(split.labels[split.test], y[test])
        assert split.labels[70 : split.n_train].tolist() == [1] * n_synthetic
        by_hand = smote(inputs[train], y[train], random_state=3)
        np.testing.assert_array_equal(split.inputs[split.train], by_hand.X)
        # The real rows of the training part: training rows, then validation rows.
        np.testing.assert_array_equal(split.inputs[split.real], inputs[fit_rows])
        # A tuning fold scores its validation rows.
        tuning = bench.make_split(X, y, train, train, val, None, n_components=4, smote_seed=3)
        assert tuning.n_test == 0
        assert tuning.test == tuning.val
        assert tuning.inputs.shape[0] == tuning.n_train + 20

    @pytest.mark.parametrize(
        "change",
        [
            lambda y: np.where(y == 1, np.where(np.arange(y.size) % 2 == 0, 1.0, 0.9), 0.2),
            lambda y: np.where(y == 1, 1.0, 0.5),
            lambda y: 2 * y,
            lambda y: np.where(np.arange(y.size) == 0, np.nan, y.astype(np.float64)),
        ],
        ids=["soft", "one-half", "0-and-2", "nan"],
    )
    def test_labels_other_than_0_and_1_are_refused_and_not_truncated(self, change: Any) -> None:
        """
        A cast to integers turns 0.9, 0.5 and 0.2 into class 0, and the
        first of these cases would then run on labels nobody passed.
        """
        X, y = self._data()
        with pytest.raises(ValueError, match="y binary 0/1"):
            next(
                bench.run_dataset(
                    "positive-control", X, change(y), QUICK, bench.FeatureCache(None)
                )
            )

    @pytest.mark.parametrize("dtype", [np.float64, np.bool_, np.int8])
    def test_labels_of_0_and_1_in_another_dtype_are_those_labels(self, dtype: Any) -> None:
        X, y = self._data()
        record = next(
            bench.run_dataset(
                "positive-control", X, y.astype(dtype), QUICK, bench.FeatureCache(None)
            )
        )
        assert record["data_sha256"] == fingerprint(X, y)
        assert record["n_positives"] == int(y.sum())


# ---------------------------------------------------------------------------
# The feature cache
# ---------------------------------------------------------------------------


def _sources(package: Path) -> list[Path]:
    """Every ``*.py`` file below ``package``, found with ``os.walk``."""
    return [
        Path(directory) / name
        for directory, _, names in os.walk(package)
        for name in names
        if name.endswith(".py")
    ]


def _code_by_hand() -> str:
    """The code digest of the cache key: file names and content hashes, hashed."""
    package = Path(hqnn_forge.__file__).parent
    hashes = {
        f"hqnn_forge/{p.relative_to(package).as_posix()}": hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in _sources(package)
    }
    script = ROOT / "examples" / "benchmark_rydberg_features.py"
    hashes[script.name] = hashlib.sha256(script.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _change_id(change: dict[str, Any]) -> str:
    packages = change.get("packages", {})
    names = [*(k for k in change if k != "packages"), *packages]
    return "-".join(names) + ("-absent" if None in packages.values() else "")


class TestCache:
    inputs = np.random.default_rng(8).uniform(-3, 3, size=(12, 4))
    labels = np.array([0, 1] * 6)

    def _counting(self) -> tuple[Any, list[int]]:
        calls: list[int] = []
        fmap = _map_by_hand(4, 1.0, 0.0, 1)

        def compute(x: np.ndarray) -> np.ndarray:
            calls.append(x.shape[0])
            return np.asarray(fmap.transform(x).numpy())

        return compute, calls

    def test_a_second_request_reads_the_same_features_without_computing(
        self, tmp_path: Path
    ) -> None:
        compute, calls = self._counting()
        config = {
            "feature_map": "RydbergFeatureMap",
            "config": _map_by_hand(4, 1.0, 0.0, 1).get_config(),
        }
        cache = bench.FeatureCache(tmp_path / "cache")
        first = cache.fetch(self.inputs, self.labels, config, compute)
        assert (first.hit, calls) == (False, [12])
        np.testing.assert_array_equal(
            first.features, _map_by_hand(4, 1.0, 0.0, 1).transform(self.inputs)
        )
        second = bench.FeatureCache(tmp_path / "cache").fetch(
            self.inputs, self.labels, config, compute
        )
        assert (second.hit, calls) == (True, [12])
        np.testing.assert_array_equal(second.features, first.features)
        assert second.features.dtype == np.float64
        # the simulation time of the first computation travels with the entry
        assert second.seconds == first.seconds > 0
        assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
        assert not list((tmp_path / "cache").glob("*.part"))

    @pytest.mark.parametrize(
        "corruption",
        [
            "empty",
            "invalid",
            "missing_features",
            "missing_environment",
            "invalid_seconds",
            "non_object_environment",
        ],
    )
    def test_unreadable_cache_entries_are_recomputed(
        self, tmp_path: Path, corruption: str
    ) -> None:
        compute, calls = self._counting()
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        cache = bench.FeatureCache(tmp_path)
        first_path = cache.path(self.inputs, self.labels, config)
        assert first_path is not None

        # Create a valid entry, then damage it.
        cache.fetch(self.inputs, self.labels, config, compute)
        if corruption == "empty":
            first_path.write_bytes(b"")
        elif corruption == "invalid":
            first_path.write_bytes(b"not a valid npz archive")
        elif corruption == "missing_features":
            np.savez(
                first_path,
                seconds=np.float64(1.0),
                environment=np.array(json.dumps(cache.environment)),
            )
        elif corruption == "invalid_seconds":
            np.savez(
                first_path,
                features=np.zeros((12, 4)),
                seconds=np.array([1.0, 2.0]),
                environment=np.array(json.dumps(cache.environment)),
            )
        elif corruption == "non_object_environment":
            np.savez(
                first_path,
                features=np.zeros((12, 4)),
                seconds=np.float64(1.0),
                environment=np.array("[1]"),
            )
        else:
            np.savez(
                first_path,
                features=np.zeros((12, 4)),
                seconds=np.float64(1.0),
            )

        with pytest.warns(RuntimeWarning, match="unreadable"):
            recovered = cache.fetch(self.inputs, self.labels, config, compute)

        assert recovered.hit is False
        assert calls == [12, 12]

        np.testing.assert_array_equal(
            recovered.features,
            _map_by_hand(4, 1.0, 0.0, 1).transform(self.inputs),
        )

        again = cache.fetch(self.inputs, self.labels, config, compute)
        assert again.hit is True
        assert calls == [12, 12]
        np.testing.assert_array_equal(again.features, recovered.features)

    def test_cache_disappearing_during_load_is_recomputed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        compute, calls = self._counting()
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        cache = bench.FeatureCache(tmp_path)
        cache.fetch(self.inputs, self.labels, config, compute)

        original_load = bench.np.load
        disappeared = False

        def load_with_disappearance(file: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal disappeared
            if not disappeared:
                disappeared = True
                raise FileNotFoundError("simulated cache removal during load")
            return original_load(file, *args, **kwargs)

        monkeypatch.setattr(bench.np, "load", load_with_disappearance)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            recovered = cache.fetch(self.inputs, self.labels, config, compute)

        assert recovered.hit is False
        assert calls == [12, 12]
        monkeypatch.setattr(bench.np, "load", original_load)
        assert cache.fetch(self.inputs, self.labels, config, compute).hit is True
        assert calls == [12, 12]
        np.testing.assert_array_equal(
            recovered.features,
            _map_by_hand(4, 1.0, 0.0, 1).transform(self.inputs),
        )

    def test_a_change_of_the_data_is_a_miss(self, tmp_path: Path) -> None:
        compute, calls = self._counting()
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        cache = bench.FeatureCache(tmp_path)
        cache.fetch(self.inputs, self.labels, config, compute)
        moved = self.inputs.copy()
        moved[7, 2] = np.nextafter(moved[7, 2], np.inf)  # one bit of one entry
        assert cache.fetch(moved, self.labels, config, compute).hit is False
        other_labels = self.labels.copy()
        other_labels[0] = 1
        assert cache.fetch(self.inputs, other_labels, config, compute).hit is False
        assert cache.fetch(self.inputs[:-1], self.labels[:-1], config, compute).hit is False
        assert calls == [12, 12, 12, 11]
        assert cache.fetch(self.inputs, self.labels, config, compute).hit is True

    def test_a_change_of_the_config_is_a_miss(self, tmp_path: Path) -> None:
        """Every setting ``get_config()`` records, and the shots, enter the key."""
        compute, calls = self._counting()
        cache = bench.FeatureCache(tmp_path)
        maps = [
            _map_by_hand(4, 1.0, 0.0, 1),
            _map_by_hand(4, 1.0, 0.0, 1, interactions=False),
            _map_by_hand(4, 3.0, 0.0, 1),  # another spacing
            _map_by_hand(4, 1.0, 0.5, 50),  # another γ
            _map_by_hand(4, 1.0, 0.5, 100),  # another step count
            _map_by_hand(4, 1.0, 0.0, 1, correlations=True),
        ]
        configs = [{"config": m.get_config(), "shots": None, "shot_seed": None} for m in maps]
        configs.append({**configs[0], "shots": 100, "shot_seed": 1})
        configs.append({**configs[0], "shots": 100, "shot_seed": 2})
        paths = {cache.path(self.inputs, self.labels, c) for c in configs}
        assert len(paths) == len(configs)
        for config in configs:
            assert cache.fetch(self.inputs, self.labels, config, compute).hit is False
        assert len(calls) == len(configs)
        # The same settings rebuilt from scratch are the same entry.
        rebuilt = {
            "config": _map_by_hand(4, 1.0, 0.0, 1).get_config(),
            "shots": None,
            "shot_seed": None,
        }
        assert cache.fetch(self.inputs, self.labels, rebuilt, compute).hit is True

    def test_the_key_is_the_hash_of_the_data_the_config_the_versions_and_the_code(
        self, tmp_path: Path
    ) -> None:
        """The key by hand: canonical JSON of the four parts, hashed."""
        config = {"config": _map_by_hand(4, 1.0, 0.5, 50).get_config()}
        env = environment()
        text = json.dumps(
            {
                "data": fingerprint(self.inputs, self.labels),
                "config": config,
                "environment": {
                    "python": env["python"],
                    "implementation": env["implementation"],
                    "hqnn_forge": env["hqnn_forge"],
                    "packages": env["packages"],
                },
                "code": _code_by_hand(),
            },
            sort_keys=True,
        )
        expected = hashlib.sha256(text.encode()).hexdigest()
        path = bench.FeatureCache(tmp_path).path(self.inputs, self.labels, config)
        assert path == tmp_path / f"{expected}.npz"
        assert (
            bench.cache_key(fingerprint(self.inputs, self.labels), config, env, _code_by_hand())
            == expected
        )

    @pytest.mark.parametrize(
        "change",
        [
            {"hqnn_forge": "0.0.0+other"},
            {"python": "3.0.0"},
            {"implementation": "pypy"},
            *({"packages": {name: "0.0.0+other"}} for name in RECORDED_PACKAGES),
            {"packages": {"torch": None}},
            {"code_sha256": "0" * 64},
        ],
        ids=_change_id,
    )
    def test_an_entry_written_under_another_version_or_other_code_is_a_miss(
        self, tmp_path: Path, change: dict[str, Any]
    ) -> None:
        """
        Each entry of the environment that ``compare_environment`` compares,
        and the code digest: features stored under another value are not
        served, and stay the entry of that other value.
        """
        compute, calls = self._counting()
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        env = environment()
        other_env = {
            **env,
            **{k: v for k, v in change.items() if k not in ("packages", "code_sha256")},
            "packages": {**env["packages"], **change.get("packages", {})},
        }
        other_code = change.get("code_sha256")
        # the change is one a record's reader would be told about, or one of the code
        assert bool(compare_environment(other_env)) is (other_code is None)

        def other() -> Any:
            return bench.FeatureCache(tmp_path, env=other_env, code_sha256=other_code)

        marked = np.full((12, 4), 7.0)
        assert other().fetch(self.inputs, self.labels, config, lambda x: marked).hit is False
        here = bench.FeatureCache(tmp_path).fetch(self.inputs, self.labels, config, compute)
        assert (here.hit, calls) == (False, [12])
        np.testing.assert_array_equal(
            here.features, _map_by_hand(4, 1.0, 0.0, 1).transform(self.inputs)
        )
        again = other().fetch(self.inputs, self.labels, config, compute)
        assert (again.hit, calls) == (True, [12])
        np.testing.assert_array_equal(again.features, marked)
        assert len(list(tmp_path.glob("*.npz"))) == 2

    def test_an_entry_under_the_key_without_versions_is_a_miss(self, tmp_path: Path) -> None:
        """A cache directory filled before #522 (data and config only) is not read."""
        compute, calls = self._counting()
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        text = json.dumps(
            {"data": fingerprint(self.inputs, self.labels), "config": config}, sort_keys=True
        )
        old = tmp_path / f"{hashlib.sha256(text.encode()).hexdigest()}.npz"
        np.savez(old, features=np.full((12, 4), 7.0), seconds=np.float64(1.0))
        fetched = bench.FeatureCache(tmp_path).fetch(self.inputs, self.labels, config, compute)
        assert (fetched.hit, calls) == (False, [12])
        np.testing.assert_array_equal(
            fetched.features, _map_by_hand(4, 1.0, 0.0, 1).transform(self.inputs)
        )

    def test_the_platform_is_not_in_the_key_and_a_hit_says_where_it_was_computed(
        self, tmp_path: Path
    ) -> None:
        """
        ``compare_environment`` does not compare the platform string and the
        key does not hold it: such a hit carries the environment of the entry.
        """
        compute, calls = self._counting()
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        env = environment()
        elsewhere = {**env, "platform": "Elsewhere-1.0-riscv64"}
        assert compare_environment(elsewhere) == {}
        first = bench.FeatureCache(tmp_path, env=elsewhere).fetch(
            self.inputs, self.labels, config, compute
        )
        assert (first.hit, first.environment) == (False, elsewhere)
        second = bench.FeatureCache(tmp_path).fetch(self.inputs, self.labels, config, compute)
        assert (second.hit, calls) == (True, [12])
        assert second.environment == elsewhere != env
        # a miss, and a request without a directory, carry the running environment
        other_config = {"config": _map_by_hand(4, 3.0, 0.0, 1).get_config()}
        third = bench.FeatureCache(tmp_path).fetch(self.inputs, self.labels, other_config, compute)
        assert (third.hit, third.environment) == (False, env)
        fourth = bench.FeatureCache(None).fetch(self.inputs, self.labels, config, compute)
        assert (fourth.hit, fourth.environment) == (False, env)

    def test_the_key_holds_exactly_what_compare_environment_compares(self) -> None:
        """Each entry of ``environment()`` is in the key if and only if a change of it is a difference."""
        env = environment()
        assert set(bench.key_environment(env)) == {*COMPARED_VERSIONS, "packages"}
        for name in sorted(set(env) - {"packages"}):
            changed = {**env, name: "other"}
            in_key = bench.key_environment(changed) != bench.key_environment(env)
            assert in_key is bool(compare_environment(changed)), name

    def test_the_code_digest_by_hand(self, tmp_path: Path) -> None:
        """Every ``*.py`` below the package and the script, by name and content."""
        package = tmp_path / "pkg"
        (package / "sub").mkdir(parents=True)
        (package / "a.py").write_bytes(b"A = 1\n")
        (package / "sub" / "b.py").write_bytes(b"B = 2\n")
        (package / "notes.txt").write_bytes(b"not code\n")
        script = tmp_path / "elsewhere" / "run.py"
        script.parent.mkdir()
        script.write_bytes(b"print(1)\n")

        def digest() -> str:
            return str(bench.code_fingerprint(bench.code_files(package, script)))

        assert bench.code_files(package, script) == {
            "pkg/a.py": package / "a.py",
            "pkg/sub/b.py": package / "sub" / "b.py",
            "run.py": script,
        }
        by_hand = {
            "pkg/a.py": hashlib.sha256(b"A = 1\n").hexdigest(),
            "pkg/sub/b.py": hashlib.sha256(b"B = 2\n").hexdigest(),
            "run.py": hashlib.sha256(b"print(1)\n").hexdigest(),
        }
        first = digest()
        assert first == hashlib.sha256(json.dumps(by_hand, sort_keys=True).encode()).hexdigest()
        # what is not Python source does not enter
        (package / "notes.txt").write_bytes(b"other notes\n")
        assert digest() == first
        # one byte of a module, one byte of the script, a new module, a module that moves
        seen = {first}
        (package / "sub" / "b.py").write_bytes(b"B = 3\n")
        seen.add(digest())
        script.write_bytes(b"print(2)\n")
        seen.add(digest())
        (package / "c.py").write_bytes(b"")
        seen.add(digest())
        (package / "a.py").rename(package / "sub" / "a.py")
        seen.add(digest())
        assert len(seen) == 5
        # bytecode Python wrote beside the source is not a module of its own
        (package / "__pycache__").mkdir()
        (package / "__pycache__" / "c.cpython-312.pyc").write_bytes(b"")
        assert digest() in seen

    @pytest.mark.parametrize("name", ["sub/d.pyc", "e.cpython-312-x86_64-linux-gnu.so", "f.pyd"])
    def test_a_module_that_is_not_source_is_refused(self, tmp_path: Path, name: str) -> None:
        """Its code would not enter the digest, so a change of it would be a hit."""
        package = tmp_path / "pkg"
        (package / "sub").mkdir(parents=True)
        (package / "a.py").write_bytes(b"A = 1\n")
        (package / name).write_bytes(b"")
        with pytest.raises(RuntimeError, match="not Python source.*--no-cache"):
            bench.code_files(package, tmp_path / "run.py")

    def test_the_code_digest_covers_the_package_and_the_script(self) -> None:
        """The files listed another way, among them the modules the features come from."""
        package = Path(hqnn_forge.__file__).parent
        files = bench.code_files()
        assert files == {
            **{f"hqnn_forge/{p.relative_to(package).as_posix()}": p for p in _sources(package)},
            "benchmark_rydberg_features.py": Path(str(bench.__file__)),
        }
        hashed = {p.resolve() for p in files.values()}
        for obj in (RydbergFeatureMap, bench.QuantumEncodingLayer, PCANormalizer, bench):
            module = obj if isinstance(obj, ModuleType) else sys.modules[obj.__module__]
            assert module.__file__ is not None
            assert Path(module.__file__).resolve() in hashed, module.__name__
        # every module of the package this process has imported
        loaded = {
            Path(str(m.__file__)).resolve()
            for name, m in list(sys.modules.items())
            if name.split(".")[0] == "hqnn_forge" and getattr(m, "__file__", None)
        }
        assert len(loaded) > 20 and loaded <= hashed
        assert bench.code_fingerprint() == bench.code_fingerprint(files) == _code_by_hand()
        assert bench.CODE_SHA256 == _code_by_hand()

    def test_an_edit_after_the_import_does_not_rename_the_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The digest in the key is the one read when the script was imported:
        a file that changes before the cache is built (while the datasets
        load) is not the code this process computes with, and its digest
        would file the features of the old code under the name of the new.
        """
        package = tmp_path / "hqnn_forge"
        package.mkdir()
        (package / "__init__.py").write_bytes(b"MARK = 1\n")
        script = ROOT / "examples" / "benchmark_rydberg_features.py"
        by_hand = {
            "hqnn_forge/__init__.py": hashlib.sha256(b"MARK = 1\n").hexdigest(),
            script.name: hashlib.sha256(script.read_bytes()).hexdigest(),
        }
        at_import = hashlib.sha256(json.dumps(by_hand, sort_keys=True).encode()).hexdigest()
        monkeypatch.setattr(hqnn_forge, "__file__", str(package / "__init__.py"))
        monkeypatch.setitem(sys.modules, "benchmark_rydberg_features", bench)  # put back after
        fresh = _script()
        assert fresh.CODE_SHA256 == at_import

        (package / "__init__.py").write_bytes(b"MARK = 2\n")
        assert fresh.code_fingerprint() != at_import
        cache = fresh.FeatureCache(tmp_path / "cache")
        assert cache.code_sha256 == at_import
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        key = fresh.cache_key(
            fingerprint(self.inputs, self.labels), config, environment(), at_import
        )
        assert cache.path(self.inputs, self.labels, config) == tmp_path / "cache" / f"{key}.npz"

    def test_a_package_without_source_files_is_refused_unless_nothing_is_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Compiled files only: no digest, so no key; without a directory it still runs."""
        package = tmp_path / "hqnn_forge"
        package.mkdir()
        (package / "__init__.pyc").write_bytes(b"")
        monkeypatch.setattr(hqnn_forge, "__file__", str(package / "__init__.pyc"))
        monkeypatch.setitem(sys.modules, "benchmark_rydberg_features", bench)  # put back after
        sourceless = _script()  # imported from such a package: the import itself still works
        assert sourceless.CODE_SHA256 is None
        with pytest.raises(RuntimeError, match="--no-cache"):
            sourceless.FeatureCache(tmp_path / "cache")
        assert not (tmp_path / "cache").exists()
        compute, calls = self._counting()
        fetched = sourceless.FeatureCache(None).fetch(self.inputs, self.labels, {}, compute)
        assert (fetched.hit, calls) == (False, [12])

    def test_a_digest_that_could_not_be_read_at_the_import_is_not_read_later(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        No digest when the script was imported, files that can be read when
        the cache is built: their digest is of the files as they are then,
        not of the code the process has loaded, so the directory is refused
        instead of being filled under it.
        """
        package = tmp_path / "hqnn_forge"
        package.mkdir()
        monkeypatch.setattr(hqnn_forge, "__file__", str(package / "__init__.py"))
        monkeypatch.setitem(sys.modules, "benchmark_rydberg_features", bench)  # put back after
        late = _script()  # no __init__.py yet: nothing to fingerprint
        assert late.CODE_SHA256 is None

        (package / "__init__.py").write_bytes(b"MARK = 2\n")
        assert len(late.code_fingerprint()) == 64  # readable now
        with pytest.raises(RuntimeError, match="when the script was imported.*--no-cache"):
            late.FeatureCache(tmp_path / "cache")
        assert not (tmp_path / "cache").exists()
        assert late.FeatureCache(tmp_path / "cache", code_sha256="c" * 64).code_sha256 == "c" * 64
        assert late.FeatureCache(None).path(self.inputs, self.labels, {}) is None

    def test_files_that_cannot_be_read_are_refused_with_a_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreadable module: no digest at the import, and the error names ``--no-cache``."""
        package = tmp_path / "hqnn_forge"
        package.mkdir()
        (package / "__init__.py").write_bytes(b"MARK = 1\n")
        monkeypatch.setattr(hqnn_forge, "__file__", str(package / "__init__.py"))
        monkeypatch.setitem(sys.modules, "benchmark_rydberg_features", bench)  # put back after

        def unreadable(path: Path) -> bytes:
            raise PermissionError(13, "Permission denied", str(path))

        monkeypatch.setattr(Path, "read_bytes", unreadable)
        locked = _script()
        assert locked.CODE_SHA256 is None
        with pytest.raises(RuntimeError, match="cannot be read.*--no-cache") as raised:
            locked.FeatureCache(tmp_path / "cache")
        assert isinstance(raised.value.__cause__, PermissionError)
        assert not (tmp_path / "cache").exists()

    def test_without_a_directory_every_request_computes(self) -> None:
        compute, calls = self._counting()
        cache = bench.FeatureCache(None)
        assert cache.path(self.inputs, self.labels, {}) is None
        assert cache.fetch(self.inputs, self.labels, {}, compute).hit is False
        assert cache.fetch(self.inputs, self.labels, {}, compute).hit is False
        assert calls == [12, 12]

    def test_a_repeated_run_reads_every_feature_matrix_and_scores_the_same(
        self, tmp_path: Path
    ) -> None:
        settings = _replace(QUICK, arms=("A", "B"))
        data = {"positive-control": bench.load_dataset("positive-control", settings)}
        first = bench.run(data, settings, tmp_path / "a.jsonl", cache_dir=tmp_path / "cache")
        stored = sorted(p.name for p in (tmp_path / "cache").iterdir())
        # per arm: the fold's rows and one matrix per tuning fold
        assert len(stored) == 2 * (1 + settings.inner_folds)
        second = bench.run(data, settings, tmp_path / "b.jsonl", cache_dir=tmp_path / "cache")
        assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == stored
        assert not any(r["feature_cache_hit"] for r in first)
        assert all(r["feature_cache_hit"] for r in second)
        for before, after in zip(first, second, strict=True):
            assert after["feature_seconds"] == before["feature_seconds"] > 0
            assert (after["mcc"], after["pr_auc"], after["threshold"]) == (
                before["mcc"],
                before["pr_auc"],
                before["threshold"],
            )
            assert after["separation"] == before["separation"]

    def test_a_run_under_another_version_computes_and_one_elsewhere_records_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Through ``run``: after a change of a package version no line is a
        hit; after a change of the platform string alone every line is, and
        carries the environment its features were computed in.
        """
        settings = _replace(QUICK, arms=FAST, n_trials=0)
        data = {"positive-control": bench.load_dataset("positive-control", settings)}
        cache = tmp_path / "cache"
        env = environment()
        first = bench.run(data, settings, tmp_path / "a.jsonl", cache_dir=cache)
        assert len(list(cache.iterdir())) == 2  # arms A and B, no tuning folds
        for record in first:
            assert record["feature_cache_hit"] is False
            assert record["feature_environment"] == (None if record["arm"] == "C" else env)

        upgraded = {**env, "packages": {**env["packages"], "torch": "0.0.0+other"}}
        monkeypatch.setattr(bench, "environment", lambda: upgraded)
        second = bench.run(data, settings, tmp_path / "b.jsonl", cache_dir=cache)
        assert len(list(cache.iterdir())) == 4
        for record in second:
            assert record["feature_cache_hit"] is False
            assert record["environment"] == upgraded
            assert record["feature_environment"] == (None if record["arm"] == "C" else upgraded)

        moved = {**env, "platform": "Elsewhere-1.0-riscv64"}
        monkeypatch.setattr(bench, "environment", lambda: moved)
        third = bench.run(data, settings, tmp_path / "c.jsonl", cache_dir=cache)
        assert len(list(cache.iterdir())) == 4
        for record, line in zip(third, bench.read_records(tmp_path / "c.jsonl"), strict=True):
            assert record["feature_cache_hit"] is (record["arm"] != "C")
            assert record["environment"] == moved
            assert record["feature_environment"] == (None if record["arm"] == "C" else env)
            assert line["feature_environment"] == record["feature_environment"]
        for before, after in zip(first, third, strict=True):
            assert (after["mcc"], after["pr_auc"]) == (before["mcc"], before["pr_auc"])

    def test_the_tuning_folds_record_their_hits_and_environments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The features of the tuning folds come through the cache too: a hit
        computed elsewhere is recorded per tuning fold in ``tuning``, not
        only for the fold the head is scored on.
        """
        settings = _replace(QUICK, arms=("A", "C"), n_trials=2, search_space={"lr": (0.01, 0.1)})
        data = {"positive-control": bench.load_dataset("positive-control", settings)}
        cache = tmp_path / "cache"
        env = environment()
        moved = {**env, "platform": "Elsewhere-1.0-riscv64"}
        monkeypatch.setattr(bench, "environment", lambda: moved)
        first = bench.run(data, settings, tmp_path / "a.jsonl", cache_dir=cache)
        monkeypatch.setattr(bench, "environment", lambda: env)
        second = bench.run(data, settings, tmp_path / "b.jsonl", cache_dir=cache)
        assert first and second
        for records, hit, where in ((first, False, moved), (second, True, env)):
            for record in records:
                fixed = record["arm"] != "C"
                tuning = record["tuning"]
                assert tuning["feature_cache_hits"] == [hit and fixed] * settings.inner_folds
                assert tuning["feature_environments"] == (
                    [moved if fixed else None] * settings.inner_folds
                )
                assert record["environment"] == where
        for line, record in zip(bench.read_records(tmp_path / "b.jsonl"), second, strict=True):
            assert line["tuning"] == record["tuning"]

    @pytest.mark.slow
    def test_the_tuning_folds_record_the_seconds_stored_with_their_entries(
        self, tmp_path: Path
    ) -> None:
        """
        #785: ``tuning["feature_seconds"]`` holds one value per tuning fold, in
        the order of the folds, each the ``seconds`` stored with that fold's
        cache entry: on a cold run the time just measured, on a warm one the
        time read back with the features.  Between the two runs every entry
        gets a value of its own planted in place of its seconds, so the warm
        run shows which entry each value was read from.  The entries are
        found from the tuning folds built by hand (:func:`inner_splits` on
        fold 0's training part and ``tune_seed``).  Arm C simulates nothing
        (``0.0`` per fold), and a reused arm B line carries its source's
        values, as it does ``feature_cache_hits``, beside its own
        ``feature_seconds`` of 0.
        """
        settings = _replace(QUICK, arms=FAST, v_over_omega=(1.0, 3.0))
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        fold = _fold_by_hand(X, y, settings, 0)
        own = bench.make_split(
            X,
            y,
            fold.train_part,
            fold.train_idx,
            fold.val_idx,
            fold.test_idx,
            n_components=settings.n_atoms,
            smote_seed=fold.seeds["smote_seed"],
        )
        tuning_splits = bench.inner_splits(
            X, y, fold.train_part, settings, fold.seeds["tune_seed"]
        )
        assert len(tuning_splits) == settings.inner_folds == 2
        directory = tmp_path / "cache"
        cache = bench.FeatureCache(directory)  # the run's environment and code digest

        def entries(record: dict[str, Any]) -> tuple[Path, list[Path]]:
            config = record["feature_map"]
            paths = [cache.path(s.inputs, s.labels, config) for s in (own, *tuning_splits)]
            assert all(p is not None and p.is_file() for p in paths)
            first, *rest = paths
            assert first is not None
            return first, [p for p in rest if p is not None]

        def seconds(path: Path) -> float:
            with np.load(path, allow_pickle=False) as stored:
                return float(stored["seconds"])

        data = {"positive-control": (X, y, data_seed)}
        cold = bench.run(data, settings, tmp_path / "cold.jsonl", cache_dir=directory)
        # A at two V/Ω, B computed at the first, each with the fold and two tuning folds.
        assert len(list(directory.iterdir())) == (2 + 1) * (1 + settings.inner_folds)
        measured = {path: seconds(path) for path in directory.iterdir()}
        assert all(s > 0 for s in measured.values())
        planted: dict[Path, float] = {}
        for i, path in enumerate(sorted(directory.iterdir())):
            with np.load(path, allow_pickle=False) as stored:
                members = {name: stored[name] for name in stored.files}
            planted[path] = 1000.0 + 0.25 * i  # exact in binary, one per entry
            members["seconds"] = np.float64(planted[path])
            with path.open("wb") as fh:
                np.savez(fh, **members)
        warm = bench.run(data, settings, tmp_path / "warm.jsonl", cache_dir=directory)

        for records, name, stored_seconds in ((cold, "cold", measured), (warm, "warm", planted)):
            lines = bench.read_records(tmp_path / f"{name}.jsonl")
            assert len(records) == len(lines) == 2 * (2 + 2 + 1)  # heads × (A, B, C lines)
            for record, line in zip(records, lines, strict=True):
                np.testing.assert_array_equal(record["train_idx"], np.sort(fold.train_idx))
                recorded = record["tuning"]["feature_seconds"]
                assert line["tuning"]["feature_seconds"] == recorded
                if record["arm"] == "C":
                    assert recorded == [0.0] * settings.inner_folds
                elif record["reused_from_v_over_omega"] is not None:
                    (source,) = [
                        r
                        for r in records
                        if r["arm"] == "B"
                        and r["head"] == record["head"]
                        and r["reused_from_v_over_omega"] is None
                    ]
                    assert record["feature_seconds"] == 0.0
                    assert recorded == source["tuning"]["feature_seconds"]
                else:
                    fold_entry, tuning_entries = entries(record)
                    assert record["feature_seconds"] == stored_seconds[fold_entry]
                    assert recorded == [stored_seconds[p] for p in tuning_entries]
            # Every entry is read by exactly one matrix of the run (per head).
            values = [
                s
                for r in records
                if r["arm"] != "C"
                and r["reused_from_v_over_omega"] is None
                and r["head"] == "linear"
                for s in (r["feature_seconds"], *r["tuning"]["feature_seconds"])
            ]
            assert sorted(values) == sorted(stored_seconds.values())

    def test_prune_removes_entries_under_another_code_digest_and_keeps_current_entries(
        self, tmp_path: Path
    ) -> None:
        """Pruning deletes entries of other code and leaves current entries intact (#575)."""
        compute, _ = self._counting()
        config1 = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        config2 = {"config": _map_by_hand(4, 3.0, 0.0, 1).get_config()}

        # One entry written under other code digest, one written under current code
        other_code = "0" * 64
        cache_other = bench.FeatureCache(tmp_path, code_sha256=other_code)
        entry_other = cache_other.fetch(self.inputs, self.labels, config1, compute)
        assert entry_other.hit is False

        cache_current = bench.FeatureCache(tmp_path)
        entry_current = cache_current.fetch(self.inputs, self.labels, config2, compute)
        assert entry_current.hit is False

        entries_before = list(tmp_path.glob("*.npz"))
        assert len(entries_before) == 2
        path_other = cache_other.path(self.inputs, self.labels, config1)
        path_current = cache_current.path(self.inputs, self.labels, config2)
        assert path_other is not None and path_current is not None
        stale_size = path_other.stat().st_size

        report = cache_current.prune()
        assert report.files == 1
        assert report.bytes == stale_size
        assert report.unreadable == ()

        assert not path_other.exists()
        assert path_current.exists()

        # The kept entry is still a hit
        hit = cache_current.fetch(self.inputs, self.labels, config2, compute)
        assert hit.hit is True
        np.testing.assert_array_equal(hit.features, entry_current.features)

    def test_prune_removes_entries_under_another_environment(self, tmp_path: Path) -> None:
        """Pruning deletes entries whose key_environment differs (#575)."""
        compute, _ = self._counting()
        config1 = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        config2 = {"config": _map_by_hand(4, 3.0, 0.0, 1).get_config()}

        env = environment()
        other_env = {**env, "packages": {**env["packages"], "torch": "0.0.0+other"}}
        cache_other = bench.FeatureCache(tmp_path, env=other_env)
        cache_other.fetch(self.inputs, self.labels, config1, compute)

        cache_current = bench.FeatureCache(tmp_path)
        cache_current.fetch(self.inputs, self.labels, config2, compute)

        path_other = cache_other.path(self.inputs, self.labels, config1)
        path_current = cache_current.path(self.inputs, self.labels, config2)
        assert path_other is not None and path_current is not None
        stale_size = path_other.stat().st_size

        report = cache_current.prune()
        assert report.files == 1
        assert report.bytes == stale_size
        assert not path_other.exists()
        assert path_current.exists()

        hit = cache_current.fetch(self.inputs, self.labels, config2, compute)
        assert hit.hit is True

    def test_prune_preserves_entries_whose_platform_alone_differs(self, tmp_path: Path) -> None:
        """Entries from another platform share key_environment and are preserved (#575)."""
        compute, _ = self._counting()
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}

        env = environment()
        elsewhere = {**env, "platform": "Elsewhere-1.0-riscv64"}
        cache_elsewhere = bench.FeatureCache(tmp_path, env=elsewhere)
        entry = cache_elsewhere.fetch(self.inputs, self.labels, config, compute)
        assert entry.hit is False

        cache_current = bench.FeatureCache(tmp_path)
        path = cache_current.path(self.inputs, self.labels, config)
        assert path is not None and path.exists()

        report = cache_current.prune()
        assert report.files == 0
        assert report.bytes == 0
        assert report.unreadable == ()
        assert path.exists()

        # The entry is still a hit for the running process
        hit = cache_current.fetch(self.inputs, self.labels, config, compute)
        assert hit.hit is True

    def test_prune_leaves_unreadable_legacy_and_partial_files_untouched(
        self, tmp_path: Path
    ) -> None:
        """Files that cannot be read as valid entries are reported, not removed (#575)."""
        compute, _ = self._counting()
        config = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        cache = bench.FeatureCache(tmp_path)
        cache.fetch(self.inputs, self.labels, config, compute)

        # Legacy entry without code_sha256 or environment (pre-#543)
        legacy_path = tmp_path / "legacy.npz"
        np.savez(legacy_path, features=np.full((12, 4), 1.0), seconds=np.float64(0.5))

        # Partial file
        part_path = tmp_path / "in_flight.part"
        part_path.write_bytes(b"partial content")

        # Corrupt npz file
        corrupt_path = tmp_path / "corrupt.npz"
        corrupt_path.write_bytes(b"not a valid zip file")

        # Zero-byte npz file
        empty_npz_path = tmp_path / "empty.npz"
        empty_npz_path.touch()

        # Non-npz file
        txt_path = tmp_path / "notes.txt"
        txt_path.write_bytes(b"some notes")

        report = cache.prune()
        assert report.files == 0
        assert report.bytes == 0
        assert set(report.unreadable) == {
            legacy_path,
            part_path,
            corrupt_path,
            empty_npz_path,
            txt_path,
        }

        # All files still exist
        assert legacy_path.exists()
        assert part_path.exists()
        assert corrupt_path.exists()
        assert empty_npz_path.exists()
        assert txt_path.exists()
        assert cache.fetch(self.inputs, self.labels, config, compute).hit is True

    def test_prune_empty_or_nonexistent_or_disabled_directory(self, tmp_path: Path) -> None:
        """prune() handles None, nonexistent, or empty directories cleanly (#575)."""
        assert bench.FeatureCache(None).prune() == bench.PruneReport(0, 0, ())
        assert bench.FeatureCache(tmp_path / "empty").prune() == bench.PruneReport(0, 0, ())
        cache_missing = bench.FeatureCache(tmp_path / "missing")
        (tmp_path / "missing").rmdir()
        assert cache_missing.prune() == bench.PruneReport(0, 0, ())

    def test_prune_removes_stale_entry_with_obsolete_environment_schema(
        self, tmp_path: Path
    ) -> None:
        """Entries under another code digest are pruned even with an obsolete environment (#575)."""
        cache = bench.FeatureCache(tmp_path)
        stale_path = tmp_path / "stale_obsolete.npz"
        np.savez(
            stale_path,
            features=np.full((12, 4), 1.0),
            seconds=np.float64(0.5),
            code_sha256=np.array("0" * 64),
            environment=np.array(json.dumps({"obsolete_schema": True})),
        )
        report = cache.prune()
        assert report.files == 1
        assert not stale_path.exists()
        assert report.unreadable == ()

    def test_prune_cli_option(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--prune-cache removes stale entries and exits without running the benchmark (#575)."""
        compute, _ = self._counting()
        config_stale = {"config": _map_by_hand(4, 1.0, 0.0, 1).get_config()}
        config_current = {"config": _map_by_hand(4, 3.0, 0.0, 1).get_config()}
        cache_dir = tmp_path / "cache"

        # Stale entry under another code digest
        cache_other = bench.FeatureCache(cache_dir, code_sha256="1" * 64)
        cache_other.fetch(self.inputs, self.labels, config_stale, compute)
        path_stale = cache_other.path(self.inputs, self.labels, config_stale)
        assert path_stale is not None and path_stale.exists()
        stale_size = path_stale.stat().st_size

        # Current valid entry
        cache_current = bench.FeatureCache(cache_dir)
        cache_current.fetch(self.inputs, self.labels, config_current, compute)
        path_current = cache_current.path(self.inputs, self.labels, config_current)
        assert path_current is not None and path_current.exists()

        assert len(list(cache_dir.glob("*.npz"))) == 2

        out_file = tmp_path / "out.jsonl"
        records = bench.main(
            ["--quick", "--prune-cache", "--cache-dir", str(cache_dir), "--out", str(out_file)]
        )

        assert records == []
        assert not out_file.exists()
        assert not path_stale.exists()
        assert path_current.exists()
        assert cache_current.fetch(self.inputs, self.labels, config_current, compute).hit is True

        captured = capsys.readouterr().out
        assert f"pruned cache ({cache_dir}): removed 1 file ({stale_size} bytes)" in captured
        assert "wrote" not in captured

    def test_prune_cli_conflicts_with_no_cache(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--prune-cache cannot be combined with --no-cache (#575, #634)."""
        with pytest.raises(SystemExit):
            bench.main(["--no-cache", "--prune-cache", "--out", str(tmp_path / "out.jsonl")])
        assert "--prune-cache cannot be used with --no-cache" in capsys.readouterr().err

    def test_prune_cli_refuses_a_missing_cache_dir_and_creates_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A mistyped --cache-dir is an error, not an already-clean cache (#634)."""
        missing = tmp_path / "x" / "typo"
        with pytest.raises(SystemExit):
            bench.main(["--prune-cache", "--cache-dir", str(missing)])
        assert f"{missing} is not an existing directory" in capsys.readouterr().err
        assert not missing.exists()
        assert not missing.parent.exists()

    def test_prune_cli_refuses_without_a_code_digest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without CODE_SHA256 the prune says why, instead of advising --no-cache (#634)."""
        monkeypatch.setattr(bench, "CODE_SHA256", None)
        with pytest.raises(SystemExit):
            bench.main(["--prune-cache", "--cache-dir", str(tmp_path)])
        error_line = capsys.readouterr().err.strip().splitlines()[-1]
        assert "--prune-cache cannot tell stale entries from current ones" in error_line
        assert "run it again" in error_line
        assert "--no-cache" not in error_line

    def test_prune_cli_names_unfingerprintable_code(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Code that can never be fingerprinted is not told to run again (#634)."""

        def compiled(*args: object, **kwargs: object) -> dict[str, Path]:
            raise RuntimeError("hqnn_forge holds a module that is not Python source")

        monkeypatch.setattr(bench, "CODE_SHA256", None)
        monkeypatch.setattr(bench, "code_files", compiled)
        with pytest.raises(SystemExit):
            bench.main(["--prune-cache", "--cache-dir", str(tmp_path)])
        error_line = capsys.readouterr().err.strip().splitlines()[-1]
        assert "hqnn_forge is not imported from Python source files" in error_line
        assert "run it again" not in error_line

    def test_prune_cli_ignores_the_run_options(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--prune-cache prunes even when a run option such as --patience is invalid (#634)."""
        assert (
            bench.main(["--prune-cache", "--cache-dir", str(tmp_path), "--patience", "-1"]) == []
        )
        assert f"pruned cache ({tmp_path}): removed 0 files" in capsys.readouterr().out

    def test_prune_cli_lists_unreadable_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The CLI prints the unreadable files it left in place (#634)."""
        (tmp_path / "corrupt.npz").write_bytes(b"not a valid zip file")
        assert bench.main(["--prune-cache", "--cache-dir", str(tmp_path)]) == []
        out = capsys.readouterr().out
        assert "unreadable cache files (not removed): corrupt.npz" in out
        assert (tmp_path / "corrupt.npz").exists()

    def test_prune_reports_a_stale_entry_whose_unlink_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale entry that cannot be removed is reported as such, not as unreadable (#634)."""
        stale = tmp_path / "stale.npz"
        np.savez(
            stale,
            features=np.full((12, 4), 1.0),
            seconds=np.float64(0.5),
            code_sha256=np.array("0" * 64),
            environment=np.array(json.dumps({})),
        )
        real_unlink = Path.unlink

        def refuse(path: Path, missing_ok: bool = False) -> None:
            if path == stale:
                raise PermissionError(13, "Permission denied", str(path))
            real_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", refuse)
        report = bench.FeatureCache(tmp_path).prune()
        assert report.files == 0
        assert report.bytes == 0
        assert report.unreadable == ()
        assert len(report.not_removed) == 1
        path, error = report.not_removed[0]
        assert path == stale
        assert "Permission denied" in error
        assert stale.exists()

        assert bench.main(["--prune-cache", "--cache-dir", str(tmp_path)]) == []
        out = capsys.readouterr().out
        assert (
            "stale cache files that could not be removed: stale.npz (Permission denied)\n" in out
        )
        assert "unreadable cache files" not in out

    def test_prune_skips_a_stale_entry_removed_meanwhile(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An entry removed by someone else before the unlink is neither removed nor failed (#634)."""
        stale = tmp_path / "stale.npz"
        np.savez(
            stale,
            features=np.full((12, 4), 1.0),
            seconds=np.float64(0.5),
            code_sha256=np.array("0" * 64),
            environment=np.array(json.dumps({})),
        )
        real_unlink = Path.unlink

        def vanish(path: Path, missing_ok: bool = False) -> None:
            real_unlink(path, missing_ok=missing_ok)
            if path == stale:
                raise FileNotFoundError(2, "No such file or directory", str(path))

        monkeypatch.setattr(Path, "unlink", vanish)
        report = bench.FeatureCache(tmp_path).prune()
        assert report == bench.PruneReport(0, 0, (), ())
        assert not stale.exists()

    def test_prune_skips_an_entry_removed_before_it_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An entry removed by someone else before it is read is not listed as unreadable."""
        stale = tmp_path / "stale.npz"
        np.savez(stale, code_sha256=np.array("0" * 64))
        real_load = np.load

        def vanish(path: Path, *args: Any, **kwargs: Any) -> Any:
            Path(path).unlink()
            return real_load(path, *args, **kwargs)

        monkeypatch.setattr(bench.np, "load", vanish)
        assert bench.FeatureCache(tmp_path).prune() == bench.PruneReport(0, 0, (), ())

    def test_prune_cli_names_the_unreadable_source_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A source file that cannot be read is named in the no-digest message (#634)."""
        missing = tmp_path / "gone.py"
        monkeypatch.setattr(bench, "CODE_SHA256", None)
        monkeypatch.setattr(bench, "code_files", lambda *a, **k: {"gone.py": missing})
        with pytest.raises(SystemExit):
            bench.main(["--prune-cache", "--cache-dir", str(tmp_path)])
        error_line = capsys.readouterr().err.strip().splitlines()[-1]
        assert "its source files cannot be read" in error_line
        assert str(missing) in error_line

    @pytest.mark.may_skip
    @pytest.mark.skipif(sys.version_info < (3, 14), reason="ZIP_ZSTANDARD needs Python 3.14+")
    def test_prune_lists_a_corrupt_zstd_entry_as_unreadable(self, tmp_path: Path) -> None:
        """A Zstandard member that fails to decode does not abort the prune (#634)."""
        path = tmp_path / "zstd.npz"
        members = {
            "features": np.full((12, 4), 1.0),
            "seconds": np.float64(0.5),
            "code_sha256": np.array("0" * 64),
            "environment": np.array(json.dumps({})),
        }
        # getattr: ZIP_ZSTANDARD only exists on 3.14+, which mypy and vermin -t=3.10- check against.
        zstd = getattr(zipfile, "ZIP_ZSTANDARD")  # noqa: B009
        with zipfile.ZipFile(path, "w", zstd) as zf:
            for name, array in members.items():
                with zf.open(f"{name}.npy", "w") as member:
                    np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=False)
        # The first four bytes of a Zstandard frame are its magic number.
        with zipfile.ZipFile(path) as zf:
            offset = zf.getinfo("code_sha256.npy").header_offset
        raw = bytearray(path.read_bytes())
        name_len, extra_len = struct.unpack("<HH", raw[offset + 26 : offset + 30])
        start = offset + 30 + name_len + extra_len
        raw[start : start + 4] = b"\xff\xff\xff\xff"
        path.write_bytes(bytes(raw))

        report = bench.FeatureCache(tmp_path).prune()
        assert report.unreadable == (path,)
        assert report.files == 0
        assert path.exists()

    def test_prune_lists_a_corrupt_lzma_entry_as_unreadable(self, tmp_path: Path) -> None:
        """An LZMA member that fails to decode does not abort the prune (#634)."""
        path = tmp_path / "lzma.npz"
        members = {
            "features": np.full((12, 4), 1.0),
            "seconds": np.float64(0.5),
            "code_sha256": np.array("0" * 64),
            "environment": np.array(json.dumps({})),
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_LZMA) as zf:
            for name, array in members.items():
                with zf.open(f"{name}.npy", "w") as member:
                    np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=False)
        # Byte 4 of a zip LZMA stream is the lc/lp/pb properties byte; 0xFF is invalid.
        with zipfile.ZipFile(path) as zf:
            offset = zf.getinfo("code_sha256.npy").header_offset
        raw = bytearray(path.read_bytes())
        name_len, extra_len = struct.unpack("<HH", raw[offset + 26 : offset + 30])
        raw[offset + 30 + name_len + extra_len + 4] = 0xFF
        path.write_bytes(bytes(raw))
        with pytest.raises(lzma.LZMAError), np.load(path, allow_pickle=False) as stored:
            stored["code_sha256"]

        report = bench.FeatureCache(tmp_path).prune()
        assert report.unreadable == (path,)
        assert report.files == 0
        assert path.exists()

    def test_prune_lists_a_corrupt_compressed_entry_as_unreadable(self, tmp_path: Path) -> None:
        """A compressed member that fails to inflate does not abort the prune (#634)."""
        path = tmp_path / "compressed.npz"
        np.savez_compressed(
            path,
            features=np.full((12, 4), 1.0),
            seconds=np.float64(0.5),
            code_sha256=np.array("0" * 64),
            environment=np.array(json.dumps({})),
        )
        # Overwrite the first byte of the code_sha256 member's deflate stream with
        # the reserved block type, so reading it raises zlib.error.
        with zipfile.ZipFile(path) as zf:
            offset = zf.getinfo("code_sha256.npy").header_offset
        raw = bytearray(path.read_bytes())
        name_len, extra_len = struct.unpack("<HH", raw[offset + 26 : offset + 30])
        raw[offset + 30 + name_len + extra_len] = 0xFF
        path.write_bytes(bytes(raw))
        with pytest.raises(zlib.error), np.load(path, allow_pickle=False) as stored:
            stored["code_sha256"]

        report = bench.FeatureCache(tmp_path).prune()
        assert report.unreadable == (path,)
        assert report.files == 0
        assert path.exists()


# ---------------------------------------------------------------------------
# Standardising a feature matrix (#518)
# ---------------------------------------------------------------------------


def _collapsed(epsilon: float, n_synthetic: int = 8) -> tuple[Any, np.ndarray, np.ndarray]:
    """
    A split of 200 rows with features ``½ + ε g`` and the rows a fit may see.

    ``g`` has four columns of unit normal noise, shifted by ``±0.8`` with
    the label in the first two, so a head can learn it.  Blocks: 120
    training rows whose last ``n_synthetic`` count as synthetic, 40
    validation rows, 40 test rows.
    """
    rng = np.random.default_rng(518)
    labels = (np.arange(200) % 2).astype(np.int64)
    g = rng.standard_normal((200, 4))
    g[:, :2] += 0.8 * (2.0 * labels[:, None] - 1.0)
    split = bench.Split(
        inputs=g, labels=labels, n_train=120, n_synthetic=n_synthetic, n_val=40, n_test=40
    )
    return split, 0.5 + epsilon * g, np.arange(120 - n_synthetic)


class TestStandardiseFeatures:
    def test_large_mean_with_real_variation_is_scaled(self) -> None:
        rows = np.arange(5000)
        values = 1e9 + np.where(rows % 2 == 0, -1e-4, 1e-4)
        features = values[:, None]
        assert 0 < values.std() < rows.size * np.finfo(np.float64).eps * abs(values.mean())
        assert values.var() <= (rows.size * np.finfo(np.float64).eps * values.mean()) ** 2
        result = bench.standardise_features(features, rows)
        np.testing.assert_allclose(result.std(), 1.0, rtol=1e-12)
        scaled, centred = bench._standardise(features, rows)
        np.testing.assert_array_equal(scaled, result)
        assert centred == []

    def test_the_column_means_are_summed_pairwise(self) -> None:
        # 5000 copies of one float64 next to a second column: summed row by row,
        # the mean of column 0 is off by about 1.8e-5, above the pairwise bound,
        # and the column would be scaled to -1 instead of only centred.
        rows = np.arange(5000)
        features = np.column_stack([np.full(rows.size, 1e9 + 1 / 3), rows.astype(float)])
        result = bench.standardise_features(features, rows)
        np.testing.assert_array_equal(result[:, 0], 0.0)

    def test_large_mean_float64_neighbours_are_only_centred_and_reported(self) -> None:
        rows = np.arange(5000)
        mean = 1e9
        neighbours = mean + np.where(rows % 2 == 0, 0.0, np.spacing(mean))
        features = neighbours[:, None]
        assert neighbours.std() > 0
        result = bench.standardise_features(features, rows)
        np.testing.assert_array_equal(result[:, 0], neighbours - neighbours.mean())
        scaled, centred = bench._standardise(features, rows)
        np.testing.assert_array_equal(scaled, result)
        assert centred == [0]

    def test_a_matrix_by_hand(self) -> None:
        # Column 0 on the fitting rows: 1, 2, 3, so μ = 2 and σ² = (1 + 0 + 1)/3.
        # Column 1 is constant there and is only centred.
        features = np.array([[1.0, 0.5], [2.0, 0.5], [3.0, 0.5], [5.0, 0.7]])
        result = bench.standardise_features(features, np.array([0, 1, 2]))
        s = math.sqrt(2.0 / 3.0)
        expected = np.array([[-1 / s, 0.0], [0.0, 0.0], [1 / s, 0.0], [3 / s, 0.2]])
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)
        assert result.dtype == np.float64

    def test_it_agrees_with_scikit_learn_fitted_on_the_fitting_rows(self) -> None:
        from sklearn.preprocessing import StandardScaler

        _, features, fit_rows = _collapsed(0.01)
        expected = StandardScaler().fit(features[fit_rows]).transform(features)
        result = bench.standardise_features(features, fit_rows)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-11)
        # On the fitting rows every column then has mean 0 and variance 1 (1/n).
        np.testing.assert_allclose(result[fit_rows].mean(axis=0), 0.0, atol=1e-11)
        np.testing.assert_allclose(result[fit_rows].std(axis=0), 1.0, atol=1e-11)

    def test_rows_outside_the_fit_change_no_other_row(self) -> None:
        split, features, fit_rows = _collapsed(0.01)
        features = features.copy()
        scale = np.array([0.125, 0.0625, 0.03125, 0.015625])
        signs = 2 * (np.arange(fit_rows.size) % 2) - 1
        features[fit_rows] = 0.5 + signs[:, None] * scale
        before = bench.standardise_features(features, fit_rows)
        # Synthetic, validation and test rows: none of them is a fitting row.
        outside = np.setdiff1d(np.arange(features.shape[0]), fit_rows)
        assert outside[0] == split.n_train - split.n_synthetic
        changed = features.copy()
        changed[outside] = 0.5 + 7.0 * (changed[outside] - 0.5)
        after = bench.standardise_features(changed, fit_rows)
        np.testing.assert_array_equal(after[fit_rows], before[fit_rows])
        columns = np.ascontiguousarray(features[fit_rows].T)
        mean, std = columns.mean(axis=1), columns.std(axis=1)
        np.testing.assert_allclose(after[outside], (changed[outside] - mean) / std, rtol=1e-12)
        assert np.abs(after[outside] - before[outside]).max() > 1.0
        # A fitting row does enter: the same change there moves every row.
        changed = features.copy()
        changed[fit_rows[:20]] = 0.5 + 7.0 * (changed[fit_rows[:20]] - 0.5)
        moved = bench.standardise_features(changed, fit_rows)
        assert np.abs(moved[outside] - before[outside]).max() > 0.1

    @pytest.mark.parametrize("epsilon", [1e-1, 1e-3, 1e-6])
    def test_the_result_does_not_depend_on_offset_and_scale(self, epsilon: float) -> None:
        # ½ + ε g has μ = ½ + ε μ_g and σ = ε σ_g, so its standardisation is that of g.
        split, features, fit_rows = _collapsed(epsilon)
        g = split.inputs
        expected = (g - g[fit_rows].mean(axis=0)) / g[fit_rows].std(axis=0)
        result = bench.standardise_features(features, fit_rows)
        # ½ + ε g is rounded to 1.1e-16, which is 1.1e-16/(ε σ_g) after the division.
        np.testing.assert_allclose(result, expected, rtol=0, atol=4 * 1.1e-16 / epsilon + 1e-13)
        # A negative factor only flips the sign.
        np.testing.assert_allclose(
            bench.standardise_features(0.5 - epsilon * g, fit_rows),
            -expected,
            rtol=0,
            atol=4 * 1.1e-16 / epsilon + 1e-13,
        )

    def test_a_constant_column_is_only_centred(self) -> None:
        # σ = 0 on the fitting rows: no division by zero, the column becomes F − μ.
        features = np.array([[0.5, 0.0], [0.5, 1.0], [0.5, 2.0], [0.75, 3.0], [0.25, 4.0]])
        result = bench.standardise_features(features, np.array([0, 1, 2]))
        assert np.isfinite(result).all()
        np.testing.assert_array_equal(result[:, 0], [0.0, 0.0, 0.0, 0.25, -0.25])
        np.testing.assert_allclose(result[:, 1], (features[:, 1] - 1.0) / math.sqrt(2 / 3))
        # Computed variance exactly 0: only centred by either rule, so not listed.
        assert bench._standardise(features, np.array([0, 1, 2]))[1] == []

    def test_a_column_constant_up_to_rounding_is_only_centred(self) -> None:
        """
        ½ and ½ ± one unit in the last place (2⁻⁵³): a computed variance of
        about 8e-33, below the rounding bound
        ``(2 ceil(log2(n)) ε |μ|)² ≈ 3.2e-30`` for
        150 rows, so the column is only centred, as scikit-learn's
        ``StandardScaler`` treats it; divided by its σ it would be ±1.2.
        """
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        ulp = 0.5 + rng.integers(-1, 2, size=(200, 1)) * 2.0**-53
        spread = 0.5 + 1e-6 * rng.standard_normal((200, 1))  # small, far above rounding
        features = np.hstack([ulp, spread])
        rows = np.arange(150)
        result = bench.standardise_features(features, rows)
        np.testing.assert_allclose(result[:, 0], ulp[:, 0] - ulp[rows, 0].mean(), atol=0)
        assert np.abs(result[:, 0]).max() < 2e-16
        assert np.abs(ulp[:, 0] - ulp[rows, 0].mean()).max() / ulp[rows, 0].std() > 1.0
        np.testing.assert_allclose(result[rows, 1].std(), 1.0, rtol=1e-9)
        expected = StandardScaler().fit(features[rows]).transform(features)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-9)
        # Only centred although its computed variance is not 0: the one column listed.
        assert ulp[rows, 0].var() > 0
        assert bench._standardise(features, rows)[1] == [0]

    @pytest.mark.parametrize(
        ("features", "rows", "message"),
        [
            (np.zeros(4), np.array([0, 1]), "features must have shape"),
            (np.zeros((4, 2)), np.array([], dtype=np.intp), "fit_rows must name at least one row"),
            (
                np.array([[0.0, 1.0], [np.nan, 2.0], [1.0, 3.0]]),
                np.array([0, 1]),
                "not finite on the fitting rows",
            ),
        ],
    )
    def test_what_is_refused(self, features: np.ndarray, rows: np.ndarray, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            bench.standardise_features(features, rows)

    @pytest.mark.parametrize("head", ["linear", "mlp"])
    def test_scores_on_standardised_features_do_not_depend_on_epsilon(self, head: str) -> None:
        settings = bench.Settings()
        config = {"lr": 0.05, "max_epochs": 30, "batch_size": 32, "patience": None}

        def score(epsilon: float, *, standardise: bool) -> Any:
            split, features, fit_rows = _collapsed(epsilon)
            if standardise:
                features = bench.standardise_features(features, fit_rows)
            model = bench.build_model(head, 4, settings, init_seed=3)
            return bench.fit_and_score(model, features, split, config, loss="bce", batch_seed=5)

        reference = score(1.0, standardise=True)
        # The task is learnable, so equal scores are not equal failures.
        assert reference.mcc > 0.5
        assert reference.pr_auc > 0.8
        for epsilon in (1e-2, 1e-4, 1e-6):
            fit = score(epsilon, standardise=True)
            assert fit.mcc == reference.mcc
            assert fit.epochs == reference.epochs
            assert fit.pr_auc == pytest.approx(reference.pr_auc, abs=1e-6)
            assert fit.threshold == pytest.approx(reference.threshold, abs=1e-4)
        # Without the step the same data at another ε is another training problem:
        # the head starts from the same weights but its logits are ε times as wide.
        raw = score(1.0, standardise=False)
        small = score(1e-4, standardise=False)
        assert small.pr_auc < raw.pr_auc - 0.05


def _collapsed_representation(epsilon: float) -> Any:
    """A fixed map with features ``½ + ε tanh(x)``: the shape of a collapse, at any ε."""

    def compute(inputs: np.ndarray) -> np.ndarray:
        return np.asarray(0.5 + epsilon * np.tanh(inputs))

    return bench.Representation(
        arm="A",
        kind="collapsed",
        heads=bench.HEADS,
        config={"feature_map": "collapsed", "epsilon": epsilon},
        compute=compute,
    )


class TestStandardisationInARun:
    """Step 4 of the script's docstring through ``run_dataset`` (#518)."""

    def _run(
        self, monkeypatch: pytest.MonkeyPatch, epsilon: float, standardise: bool
    ) -> tuple[list[dict[str, Any]], Any, np.ndarray]:
        settings = _replace(QUICK, arms=("A",), standardise=standardise)
        monkeypatch.setattr(
            bench, "representations", lambda s: [_collapsed_representation(epsilon)]
        )
        X, y, _ = bench.load_dataset("positive-control", settings)
        records = list(
            bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None))
        )
        return records, _fold_by_hand(X, y, settings, 0), y

    def test_scores_do_not_depend_on_epsilon_and_are_those_of_a_head_trained_by_hand(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Features ``½ + ε tanh(x)``: every line, tuning included, is the same
        for ε of 1, 1e-3 and 1e-6, and each is the score of a head trained by
        hand on ``tanh(x)`` standardised with numpy on the real training rows
        of the fold rebuilt by hand.  The separation measures stay on the
        features the map returns: their distances scale with ε, the Fisher
        ratio does not, and both are those of the library on those features.
        """
        lines = {}
        for epsilon in (1.0, 1e-3, 1e-6):
            records, fold, y = self._run(monkeypatch, epsilon, standardise=True)
            stacked = np.concatenate(
                [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
            )
            # The standardisation of ½ + ε g is that of g, so the head is trained on g.
            expected = _standardised_by_hand(np.tanh(stacked), fold)
            real = np.concatenate([fold.train_idx, fold.val_idx])
            separation = separation_measures(
                0.5 + epsilon * np.tanh(fold.inputs[real]), y[real]
            ).to_dict()
            for record in records:
                assert record["settings"]["standardise"] is True
                assert record["tuning"] is not None
                _assert_scores_by_hand(record, fold, y, expected, record["head"])
                assert record["separation"]["train"] == pytest.approx(separation, rel=1e-6)
            lines[epsilon] = {r["head"]: r for r in records}
        assert set(lines[1.0]) == {"linear", "mlp"}
        assert lines[1.0]["linear"]["mcc"] > 0.0
        for epsilon in (1e-3, 1e-6):
            for head, reference in lines[1.0].items():
                line = lines[epsilon][head]
                assert line["mcc"] == reference["mcc"]
                assert line["epochs"] == reference["epochs"]
                assert line["hyperparameters"] == reference["hyperparameters"]
                assert line["tuning"]["scores"] == reference["tuning"]["scores"]
                assert line["pr_auc"] == pytest.approx(reference["pr_auc"], abs=1e-6)
                assert line["threshold"] == pytest.approx(reference["threshold"], abs=1e-4)
                distance = line["separation"]["train"]["mean_between_distance"]
                assert distance == pytest.approx(
                    epsilon * reference["separation"]["train"]["mean_between_distance"],
                    rel=1e-6,
                )
                assert line["separation"]["train"]["fisher_ratio"] == pytest.approx(
                    reference["separation"]["train"]["fisher_ratio"], rel=1e-6
                )

    def test_without_standardisation_the_head_sees_the_features_as_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``standardise=False``: the scores of a head trained by hand on ``½ + ε tanh(x)``."""
        scores = {}
        for epsilon in (1.0, 1e-4):
            records, fold, y = self._run(monkeypatch, epsilon, standardise=False)
            stacked = np.concatenate(
                [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
            )
            for record in records:
                assert record["settings"]["standardise"] is False
                _assert_scores_by_hand(
                    record, fold, y, 0.5 + epsilon * np.tanh(stacked), record["head"]
                )
            scores[epsilon] = [
                (r["pr_auc"], r["threshold"], r["tuning"]["scores"]) for r in records
            ]
        assert scores[1e-4] != scores[1.0]

    def test_statistics_are_those_of_the_real_training_rows_of_every_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Each call fits on the first rows of its split's training block, as
        many as the split has real training rows (counted here from the
        folds rebuilt by hand): no synthetic, validation or test row.  Arm C
        is standardised too; ``D-hybrid`` never is.  :func:`fit_inputs`
        standardises the raw columns with the same function (step 2, #580):
        its calls are told apart by a flag it sets, and fit on the outer
        fold's training part and on each tuning fold's fitting rows.
        """
        calls: list[tuple[int, np.ndarray]] = []
        raw_calls: list[np.ndarray] = []
        in_preprocess = False
        original = bench._standardise
        original_preprocess = bench._preprocess

        def spy(features: Any, fit_rows: np.ndarray) -> Any:
            if in_preprocess:
                raw_calls.append(np.asarray(fit_rows))
            else:
                calls.append((np.asarray(features).shape[0], np.asarray(fit_rows)))
            return original(features, fit_rows)

        def preprocess_spy(X: Any, fit_rows: np.ndarray, n_components: int) -> Any:
            nonlocal in_preprocess
            in_preprocess = True
            try:
                return original_preprocess(X, fit_rows, n_components)
            finally:
                in_preprocess = False

        settings = _replace(QUICK, arms=("A", "C"))
        X, y, _ = bench.load_dataset("positive-control", settings)
        fold = _fold_by_hand(X, y, settings, 0)  # before the spies: it standardises too
        # Every standardisation, the public function included, goes through these two.
        monkeypatch.setattr(bench, "_standardise", spy)
        monkeypatch.setattr(bench, "_preprocess", preprocess_spy)
        list(bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None)))
        n_outer = fold.train_inputs.shape[0] + fold.val_idx.size + fold.test_idx.size
        inner = stratified_kfold(
            y[fold.train_part], settings.inner_folds, random_state=fold.seeds["tune_seed"]
        )
        expected = [(n_outer, fold.train_idx.size)]
        for fit, held_out in inner:
            n_synthetic = len(smote(np.zeros((fit.size, 1)), y[fold.train_part][fit]).sources)
            expected.append((fit.size + n_synthetic + held_out.size, fit.size))
        assert [(n, rows.size) for n, rows in calls] == expected * 2  # arm A, then arm C
        for _, rows in calls:
            np.testing.assert_array_equal(rows, np.arange(rows.size))
        expected_raw = [fold.train_part] + [fold.train_part[fit] for fit, _ in inner]
        assert len(raw_calls) == len(expected_raw)
        for rows, want in zip(raw_calls, expected_raw, strict=True):
            np.testing.assert_array_equal(rows, want)

        calls.clear()
        raw_calls.clear()
        settings = _replace(QUICK, arms=("D",), n_trials=0, max_epochs=1)
        list(bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None)))
        assert [(n, rows.size) for n, rows in calls] == [(n_outer, fold.train_idx.size)]
        assert len(raw_calls) == 1
        np.testing.assert_array_equal(raw_calls[0], fold.train_part)

    def test_the_cache_holds_the_features_as_returned_and_serves_either_setting(
        self, tmp_path: Path
    ) -> None:
        """A run without the step reads every entry of a run with it, and scores otherwise."""
        settings = _replace(QUICK, arms=("A", "B"), n_trials=0)
        data = {"positive-control": bench.load_dataset("positive-control", settings)}
        cache = tmp_path / "cache"
        on = bench.run(data, settings, tmp_path / "on.jsonl", cache_dir=cache)
        stored = sorted(p.name for p in cache.iterdir())
        off = bench.run(
            data, _replace(settings, standardise=False), tmp_path / "off.jsonl", cache_dir=cache
        )
        assert sorted(p.name for p in cache.iterdir()) == stored
        assert not any(r["feature_cache_hit"] for r in on)
        assert all(r["feature_cache_hit"] for r in off)
        for a, b in zip(on, off, strict=True):
            assert (a["settings"]["standardise"], b["settings"]["standardise"]) == (True, False)
            assert a["separation"] == b["separation"]
        assert [(r["pr_auc"], r["threshold"]) for r in on] != [
            (r["pr_auc"], r["threshold"]) for r in off
        ]

    def test_the_command_line_switches_it_and_defaults_to_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert bench.Settings().standardise is True
        assert QUICK.standardise is True
        seen: list[Any] = []

        def run(datasets: Any, settings: Any, *args: Any, **kwargs: Any) -> list[Any]:
            seen.append(settings)
            return []

        monkeypatch.setattr(bench, "run", run)
        out = str(tmp_path / "out.jsonl")
        for flags, expected in (
            ([], True),
            (["--no-standardise"], False),
            (["--standardise"], True),
        ):
            bench.main([*flags, "--samples", "40", "--out", out, "--no-cache"])
            assert seen[-1].standardise is expected
            assert seen[-1].as_dict()["standardise"] is expected
            # --quick replaces the settings by QUICK but keeps this switch.
            bench.main(["--quick", *flags, "--out", out, "--no-cache"])
            assert seen[-1].standardise is expected
            assert seen[-1].n_atoms == QUICK.n_atoms


def _spy_on_make_split(monkeypatch: pytest.MonkeyPatch) -> list[tuple[np.ndarray, Any]]:
    """Every ``(fit_rows, split)`` that :func:`make_split` is called with and returns, in order."""
    made: list[tuple[np.ndarray, Any]] = []
    original = bench.make_split

    def spy(X: Any, y: Any, fit_rows: Any, *args: Any, **kwargs: Any) -> Any:
        split = original(X, y, fit_rows, *args, **kwargs)
        made.append((np.asarray(fit_rows), split))
        return split

    monkeypatch.setattr(bench, "make_split", spy)
    return made


class TestRecordedConstantColumns:
    """
    ``raw_constant_columns`` and ``feature_constant_columns`` are the
    columns of the standardisation that was applied (#692): each split's
    own, outer fold and tuning folds alike, on the rows it was fitted on.
    """

    def test_a_split_keeps_the_raw_columns_only_centred_on_its_fitting_rows(self) -> None:
        """
        Column 0 holds ``1.5`` and its float64 neighbour ``1.5 + 2⁻⁵²`` on
        rows 0 to 89, a variance of about ``1e-32``, nonzero but below the
        rounding bound ``(2 ceil(log2(n)) ε μ)² ≈ 2.2e-29`` for 90 rows, and
        ``2.5`` on row 100.  Fitted on rows 0 to 89 it is only centred and
        listed; fitted on rows that include row 100 it is scaled and not.
        """
        rng = np.random.default_rng(692)
        X = rng.standard_normal((120, 6))
        X[:, 0] = np.where(np.arange(120) % 3 == 0, np.nextafter(1.5, 2.0), 1.5)
        X[100, 0] = 2.5
        y = (np.arange(120) % 4 == 0).astype(np.int64)
        assert 0 < X[:90, 0].var() < (2 * 7 * np.finfo(np.float64).eps * 1.5) ** 2
        train, val, test = np.arange(0, 70), np.arange(70, 90), np.arange(90, 120)
        split = bench.make_split(
            X, y, np.arange(90), train, val, test, n_components=4, smote_seed=3
        )
        assert split.raw_constant_columns == (0,)
        wider = bench.make_split(
            X, y, np.arange(110), train, val, None, n_components=4, smote_seed=3
        )
        assert wider.raw_constant_columns == ()

    def test_each_tuning_fold_records_the_raw_columns_of_the_rows_it_was_fitted_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Raw column ``j`` (``j = 0 … inner_folds − 1``) is ``1e9`` or its
        float64 neighbour on every row except one row ``r_j`` of the
        training part, held out by tuning fold ``j``, where it is
        ``1e9 + 1e3``.  A set of fitting rows that holds ``r_j`` scales
        column ``j``, one without ``r_j`` only centres it: so the outer fold,
        fitted on every ``r_j``, records ``[]``, and tuning fold ``j``, fitted
        on every ``r_c`` but its own, records ``[j]``, which a list taken
        from another fold would not.  Which fold holds which row out follows
        from the fitting rows :func:`make_split` was actually called with,
        not from a second computation of the folds: :func:`inner_splits` is
        made to build its folds from another seed than the run's
        ``tune_seed``, so a second ``stratified_kfold`` call with that seed
        would name other folds.
        """
        settings = _replace(QUICK, arms=("C",), max_epochs=1, inner_folds=3)
        X, y, _ = bench.load_dataset("positive-control", settings)
        fold = _fold_by_hand(X, y, settings, 0)
        train_part = fold.train_part
        shifted_seed = fold.seeds["tune_seed"] + 1
        # One row held out by each fold of the shifted seed; checked below
        # against the fitting rows the run used.
        marks = [
            int(train_part[held_out][0])
            for _, held_out in stratified_kfold(
                y[train_part], settings.inner_folds, random_state=shifted_seed
            )
        ]
        assert X.shape[1] >= settings.inner_folds
        noise = np.where(np.arange(X.shape[0]) % 2 == 0, 0.0, np.spacing(1e9))
        for j, r in enumerate(marks):
            X[:, j] = 1e9 + noise
            X[r, j] = 1e9 + 1e3
        original_inner_splits = bench.inner_splits

        def shifted_inner_splits(
            X: Any, y: Any, train_part: Any, settings: Any, tune_seed: int
        ) -> Any:
            return original_inner_splits(X, y, train_part, settings, tune_seed + 1)

        monkeypatch.setattr(bench, "inner_splits", shifted_inner_splits)
        made = _spy_on_make_split(monkeypatch)
        records = list(
            bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None))
        )
        (outer_fit, _), *tuning = made
        np.testing.assert_array_equal(outer_fit, train_part)
        assert len(tuning) == settings.inner_folds

        def only_centred(fit: np.ndarray) -> list[int]:
            return [j for j, r in enumerate(marks) if r not in fit]

        expected = [only_centred(fit) for fit, _ in tuning]
        assert expected == [[j] for j in range(settings.inner_folds)]
        # The premise: folds rebuilt with the run's own tune_seed give other lists.
        rebuilt = stratified_kfold(
            y[train_part], settings.inner_folds, random_state=fold.seeds["tune_seed"]
        )
        assert [only_centred(train_part[fit]) for fit, _ in rebuilt] != expected
        assert records
        for record in records:
            assert record["raw_constant_columns"] == []
            assert record["tuning"]["raw_constant_columns"] == expected
            line = json.loads(bench.to_json_line(record))
            assert line["tuning"]["raw_constant_columns"] == expected

    def test_a_feature_constant_up_to_rounding_on_the_real_training_rows_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A map whose feature ``m`` on the ``m``-th split prepared (0 the
        outer fold, ``1 + i`` tuning fold ``i``) is ``½`` or ``½ + 2⁻⁵³`` on
        every real row and ``¾`` on every synthetic SMOTE row; its other
        first ``1 + inner_folds`` features vary as ``½ + ¼ tanh`` of an
        input, and the rest are ``tanh`` of the inputs.  On the ``n`` real
        training rows, ``k`` of them at ``½ + 2⁻⁵³``, feature ``m`` has
        variance ``k (n − k)/n² · 2⁻¹⁰⁶ ≤ 2⁻¹⁰⁸ ≈ 3.1e-33``, nonzero and
        below the rounding bound ``(2 ceil(log2(n)) ε · ½)²``, at least
        ``ε² ≈ 4.9e-32``: only centred, so every line records ``[0]`` and
        tuning fold ``i`` records ``[1 + i]``, which a list taken from
        another fold would not.  Fitted with the synthetic rows, whose ``¾``
        gives a variance of order ``1e-2``, it would be scaled and nothing
        recorded.

        The data are the positive control with two thirds of its positive
        rows dropped, so SMOTE adds synthetic rows to every split.
        """
        settings = _replace(QUICK, arms=("A",), max_epochs=1)
        X, y, _ = bench.load_dataset("positive-control", settings)
        keep = (y == 0) | (np.cumsum(y) % 3 == 0)
        X, y = X[keep], y[keep]
        made = _spy_on_make_split(monkeypatch)

        n_marked = 1 + settings.inner_folds

        def compute(inputs: np.ndarray) -> np.ndarray:
            m, split = next((m, s) for m, (_, s) in enumerate(made) if s.inputs is inputs)
            synthetic = np.zeros(inputs.shape[0], dtype=bool)
            synthetic[split.n_train - split.n_synthetic : split.n_train] = True
            rounding = 0.5 + np.where(inputs[:, 0] > 0, 2.0**-53, 0.0)
            marked = [
                np.where(synthetic, 0.75, rounding)
                if c == m
                else 0.5 + 0.25 * np.tanh(inputs[:, c % inputs.shape[1]])
                for c in range(n_marked)
            ]
            return np.column_stack([*marked, np.tanh(inputs)])

        rep = bench.Representation(
            arm="A",
            kind="rounding",
            heads=bench.HEADS,
            config={"feature_map": "rounding"},
            compute=compute,
        )
        monkeypatch.setattr(bench, "representations", lambda s: [rep])
        records = list(
            bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None))
        )

        eps = np.finfo(np.float64).eps
        assert len(made) == n_marked
        for m, (_, split) in enumerate(made):
            assert split.n_synthetic > 0
            column = compute(split.inputs)[:, m]
            real = column[split.real_train]
            n, k = real.size, int((real > 0.5).sum())
            assert 0 < k < n
            variance = k * (n - k) / n**2 * 2.0**-106
            assert variance < (2 * math.ceil(math.log2(n)) * eps * 0.5) ** 2
            assert column[split.train].var() > 1e-3
        assert records
        for record in records:
            assert record["feature_constant_columns"] == [0]
            tuning = [[1 + i] for i in range(settings.inner_folds)]
            assert record["tuning"]["feature_constant_columns"] == tuning
            line = json.loads(bench.to_json_line(record))
            assert line["feature_constant_columns"] == [0]
            assert line["tuning"]["feature_constant_columns"] == tuning


# ---------------------------------------------------------------------------
# Strict JSON
# ---------------------------------------------------------------------------


def _strict_loads(line: str) -> Any:
    def refuse(token: str) -> Any:
        raise AssertionError(f"not strict JSON: {token}")

    return json.loads(line, parse_constant=refuse)


class TestJson:
    def test_infinity_and_nan_have_a_stated_mapping(self) -> None:
        record = {
            "a": math.inf,
            "b": -math.inf,
            "c": math.nan,
            "d": np.float64(np.inf),
            "e": [1.5, math.inf, (np.float32(-np.inf), math.nan)],
            "f": {"g": np.int64(3), "h": np.array([1, 2]), "i": None, "j": "inf", "k": True},
        }
        line = bench.to_json_line(record)
        assert "Infinity" not in line
        assert "NaN" not in line
        assert _strict_loads(line) == {
            "a": "inf",
            "b": "-inf",
            "c": None,
            "d": "inf",
            "e": [1.5, "inf", ["-inf", None]],
            "f": {"g": 3, "h": [1, 2], "i": None, "j": "inf", "k": True},
        }

    def test_the_separation_measures_survive_the_file(self, tmp_path: Path) -> None:
        """Perfect separation (``inf``) and identical rows (NaN), as #500 returns them."""
        perfect = separation_measures([[0.0], [0.0], [1.0], [1.0]], [0, 0, 1, 1]).to_dict()
        identical = separation_measures([[0.5], [0.5], [0.5], [0.5]], [0, 0, 1, 1]).to_dict()
        assert math.isinf(perfect["fisher_ratio"])
        assert math.isinf(perfect["distance_ratio"])
        assert math.isnan(identical["fisher_ratio"])
        assert math.isnan(identical["distance_ratio"])
        record = {
            "arm": "C",
            "v_over_omega": None,
            "pr_auc": 0.75,
            "separation": {"train": perfect, "test": identical},
        }
        path = tmp_path / "records.jsonl"
        path.write_text(bench.to_json_line(record) + "\n\n", encoding="utf-8")
        written = _strict_loads(path.read_text(encoding="utf-8").strip())
        assert written["separation"]["train"]["fisher_ratio"] == "inf"
        assert written["separation"]["test"]["fisher_ratio"] is None
        (read,) = bench.read_records(path)
        assert read["v_over_omega"] is None  # "does not apply" stays None
        assert read["pr_auc"] == 0.75
        for part, original in (("train", perfect), ("test", identical)):
            restored = read["separation"][part]
            assert set(restored) == set(original)
            assert type(restored["n_samples"]) is int
            for key, value in original.items():
                if isinstance(value, float) and math.isnan(value):
                    assert math.isnan(restored[key]), key
                else:
                    assert restored[key] == value, key

    def test_an_xz_file_reads_back_as_the_plain_one(self, tmp_path: Path) -> None:
        """``read_records`` opens ``.xz`` with ``lzma`` and returns the same records (#502)."""
        perfect = separation_measures([[0.0], [0.0], [1.0], [1.0]], [0, 0, 1, 1]).to_dict()
        identical = separation_measures([[0.5], [0.5], [0.5], [0.5]], [0, 0, 1, 1]).to_dict()
        records: list[dict[str, Any]] = [
            {
                "arm": "A",
                "v_over_omega": 1.0,
                "pr_auc": 0.75,
                "mcc": 0.5,
                "separation": {"train": perfect, "test": identical},
            },
            {
                "arm": "C",
                "v_over_omega": None,
                "pr_auc": math.nan,
                "mcc": -0.25,
                "separation": {"train": identical, "test": perfect},
            },
        ]
        text = "".join(bench.to_json_line(r) + "\n" for r in records)
        plain = tmp_path / "records.jsonl"
        plain.write_text(text, encoding="utf-8")
        packed = tmp_path / "records.jsonl.xz"
        packed.write_bytes(lzma.compress(text.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME))
        assert packed.read_bytes()[:6] == b"\xfd7zXZ\x00"  # really compressed
        from_plain = bench.read_records(plain)
        from_xz = bench.read_records(str(packed))
        assert len(from_xz) == len(records)

        def canonical(rs: list[dict[str, Any]]) -> str:
            # NaN != NaN, so compare the strict-JSON lines the records write back to.
            return "".join(bench.to_json_line(r) for r in rs)

        assert canonical(from_xz) == canonical(from_plain) == canonical(records)
        assert math.isinf(from_xz[0]["separation"]["train"]["fisher_ratio"])
        assert math.isnan(from_xz[0]["separation"]["test"]["fisher_ratio"])
        assert math.isnan(from_xz[1]["pr_auc"])
        assert from_xz[1]["v_over_omega"] is None

    def test_a_pr_auc_that_is_nan_survives_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A model whose probabilities are not finite scores a ``pr_auc`` of
        NaN, written as ``null``; read back it is NaN again, so the lines of
        a file summarise as the records ``run`` returned do.
        """
        rng = np.random.default_rng(0)
        labels = np.tile([0, 1], 20)
        inputs = rng.normal(size=(40, 2)) + labels[:, None]
        split = bench.Split(
            inputs=inputs, labels=labels, n_train=20, n_synthetic=0, n_val=10, n_test=10
        )
        model = LinearClassifier(2)
        trained = model.predict_proba

        def predict_proba(x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(trained(x), math.nan)

        monkeypatch.setattr(model, "predict_proba", predict_proba)
        fit = bench.fit_and_score(
            model, split.inputs, split, QUICK.defaults, loss="bce", batch_seed=0
        )
        assert math.isnan(fit.pr_auc)

        separation = separation_measures([[0.0], [0.5], [1.0], [1.5]], [0, 0, 1, 1]).to_dict()
        record = {
            "dataset": "positive-control",
            "v_over_omega": None,
            "gamma_over_omega": None,
            "arm": "C",
            "head": "linear",
            "mcc": fit.mcc,
            "pr_auc": fit.pr_auc,
            "feature_dim": 2,
            "n_parameters": 3,
            "feature_seconds": 0.0,
            "train_seconds": fit.seconds,
            "separation": {"train": separation, "test": separation},
        }
        path = tmp_path / "records.jsonl"
        path.write_text(bench.to_json_line(record) + "\n", encoding="utf-8")
        assert _strict_loads(path.read_text(encoding="utf-8"))["pr_auc"] is None
        (read,) = bench.read_records(path)
        assert math.isnan(read["pr_auc"])
        assert bench.summarise([read]) == bench.summarise([record])
        assert " nan " in bench.summarise([read])


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


class TestTuning:
    def test_every_arm_gets_the_same_trials_on_folds_of_the_training_part(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _replace(
            QUICK, arms=FAST, n_trials=3, search_space={"lr": (0.01, 0.03, 0.1, 0.3)}
        )
        X, y, _ = bench.load_dataset("positive-control", settings)
        fold = _fold_by_hand(X, y, settings, 0)

        splits: list[tuple[Any, ...]] = []
        original_split = bench.make_split

        def spy_split(
            X_: Any, y_: Any, fit: Any, train: Any, val: Any, test: Any, **kw: Any
        ) -> Any:
            splits.append((fit, train, val, test, kw["smote_seed"]))
            return original_split(X_, y_, fit, train, val, test, **kw)

        fits_per_tune: list[int] = []
        fits: list[int] = [0]
        original_fit, original_tune = bench.fit_and_score, bench.tune

        def spy_fit(*args: Any, **kw: Any) -> Any:
            fits[0] += 1
            return original_fit(*args, **kw)

        def spy_tune(*args: Any, **kw: Any) -> Any:
            before = fits[0]
            result = original_tune(*args, **kw)
            fits_per_tune.append(fits[0] - before)
            return result

        monkeypatch.setattr(bench, "make_split", spy_split)
        monkeypatch.setattr(bench, "fit_and_score", spy_fit)
        monkeypatch.setattr(bench, "tune", spy_tune)
        records = list(
            bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None))
        )

        # The budget: the same number of trainings for each of the six models.
        assert len(records) == 6
        assert fits_per_tune == [3 * settings.inner_folds] * 6
        assert fits[0] == 6 * (3 * settings.inner_folds + 1)
        expected = sample_configs({"lr": (0.01, 0.03, 0.1, 0.3)}, 3, fold.seeds["tune_seed"])
        for record in records:
            assert record["tuning"]["n_trials"] == 3
            assert record["tuning"]["inner_folds"] == settings.inner_folds
            assert record["tuning"]["candidates"] == expected
            scores = record["tuning"]["scores"]
            assert len(scores) == 3
            chosen = expected[scores.index(max(scores))]  # the first of the best
            assert record["hyperparameters"] == {**settings.defaults, **chosen}
            assert record["tuning_seconds"] > 0

        # The data: one outer split, then the tuning folds, which never touch a test row.
        outer, *inner = splits
        np.testing.assert_array_equal(outer[0], fold.train_part)
        np.testing.assert_array_equal(outer[3], fold.test_idx)
        assert len(inner) == settings.inner_folds
        by_hand = stratified_kfold(
            y[fold.train_part], settings.inner_folds, random_state=fold.seeds["tune_seed"]
        )
        held_out: list[int] = []
        for (fit, train, val, test, seed), (fit_pos, val_pos) in zip(inner, by_hand, strict=True):
            assert test is None
            assert seed == fold.seeds["tune_seed"]
            np.testing.assert_array_equal(fit, fold.train_part[fit_pos])
            np.testing.assert_array_equal(train, fit)  # fitted and trained on the same rows
            np.testing.assert_array_equal(val, fold.train_part[val_pos])
            assert not set(fit.tolist()) & set(val.tolist())
            assert not (set(fit.tolist()) | set(val.tolist())) & set(fold.test_idx.tolist())
            held_out += val.tolist()
        assert sorted(held_out) == fold.train_part.tolist()

    @pytest.mark.slow
    def test_every_arm_is_fitted_on_the_same_rows_with_the_same_seeds_and_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        What each of the nine models of a fold is handed, arm D and the
        hybrid row included: the same split object, initialisation seed,
        batch seed and loss for the final fit, and the same tuning folds,
        candidates, seed and training settings for every trial.

        The search space holds more configurations than the budget of two
        trials, so a model that was given one trial more would get a longer
        list; with the two values of ``QUICK`` every budget of two or more
        is the whole grid.
        """
        space = {"lr": (0.01, 0.02, 0.05, 0.1)}
        settings = _replace(QUICK, search_space=space)
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        fold = _fold_by_hand(X, y, settings, 0)

        seed_of: dict[int, int] = {}
        fits: list[SimpleNamespace] = []
        tunes: list[SimpleNamespace] = []
        tuning = [False]
        original_build, original_fit, original_tune = (
            bench.build_model,
            bench.fit_and_score,
            bench.tune,
        )

        def spy_build(head: str, n_features: int, settings_: Any, seed: int) -> Any:
            model = original_build(head, n_features, settings_, seed)
            seed_of[id(model)] = seed
            return model

        def spy_fit(model: Any, features: Any, split: Any, config: Any, **kw: Any) -> Any:
            fits.append(
                SimpleNamespace(
                    model=model,  # kept alive, so that no id is reused
                    init_seed=seed_of[id(model)],
                    n_rows=features.shape[0],
                    split=split,
                    config=dict(config),
                    loss=kw["loss"],
                    batch_seed=kw["batch_seed"],
                    tuning=tuning[0],
                )
            )
            return original_fit(model, features, split, config, **kw)

        def spy_tune(
            head: str, splits: Any, features: Any, candidates: Any, settings_: Any, seed: int
        ) -> Any:
            tunes.append(
                SimpleNamespace(splits=list(splits), candidates=list(candidates), seed=seed)
            )
            tuning[0] = True
            try:
                return original_tune(head, splits, features, candidates, settings_, seed)
            finally:
                tuning[0] = False

        monkeypatch.setattr(bench, "build_model", spy_build)
        monkeypatch.setattr(bench, "fit_and_score", spy_fit)
        monkeypatch.setattr(bench, "tune", spy_tune)
        records = list(
            bench.run_dataset(
                "positive-control", X, y, settings, bench.FeatureCache(None), data_seed=data_seed
            )
        )
        assert [(r["arm"], r["head"]) for r in records] == ROWS

        # The final fit of every line.
        final = [f for f in fits if not f.tuning]
        assert len(final) == len(ROWS)
        stacked = np.concatenate(
            [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
        )
        np.testing.assert_array_equal(final[0].split.inputs, stacked)
        for fit, record in zip(final, records, strict=True):
            row = (record["arm"], record["head"])
            assert fit.split is final[0].split, row
            assert fit.n_rows == stacked.shape[0], row
            assert fit.init_seed == fold.seeds["init_seed"], row
            assert fit.batch_seed == fold.seeds["batch_seed"], row
            assert fit.loss == settings.loss, row
            assert fit.config == record["hyperparameters"], row

        # The tuning of every line: the same folds, candidates and seed ...
        assert len(tunes) == len(ROWS)
        candidates = sample_configs(space, settings.n_trials, fold.seeds["tune_seed"])
        assert len(candidates) == settings.n_trials == 2 < len(space["lr"])
        for tune in tunes:
            assert len(tune.splits) == settings.inner_folds
            assert all(a is b for a, b in zip(tune.splits, tunes[0].splits, strict=True))
            assert tune.candidates == candidates
            assert tune.seed == fold.seeds["tune_seed"]
        # ... and every trial of every line with the same settings.
        trials = [f for f in fits if f.tuning]
        per_line = len(candidates) * settings.inner_folds
        assert len(trials) == len(ROWS) * per_line
        expected = [
            ({**settings.defaults, **candidate}, split)
            for candidate in candidates
            for split in tunes[0].splits
        ]
        for start in range(0, len(trials), per_line):
            for trial, (config, split) in zip(
                trials[start : start + per_line], expected, strict=True
            ):
                assert trial.config == config
                assert trial.split is split
                assert trial.init_seed == trial.batch_seed == fold.seeds["tune_seed"]
                assert trial.loss == settings.loss
                assert trial.n_rows == split.inputs.shape[0]

    def test_the_best_mean_score_wins_and_the_first_wins_a_tie(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table = {0.1: [0.2, 0.4], 0.2: [0.5, 0.3], 0.3: [0.4, 0.4], 0.4: [0.1, 0.1]}
        calls: list[float] = []

        def fake_fit(model: Any, features: Any, split: Any, config: Any, **kw: Any) -> Any:
            calls.append(config["lr"])
            return bench.Fit(table[config["lr"]][calls.count(config["lr"]) - 1], 0.0, 0.5, 0.0, 1)

        monkeypatch.setattr(bench, "fit_and_score", fake_fit)
        candidates = [{"lr": lr} for lr in table]
        features = [np.zeros((4, 2)), np.zeros((4, 2))]
        chosen, scores = bench.tune("linear", [None, None], features, candidates, QUICK, 0)
        assert scores == pytest.approx([0.3, 0.4, 0.4, 0.1])
        assert chosen == {"lr": 0.2}
        assert calls == [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]

    def test_no_trials_means_the_default_settings(self, tmp_path: Path) -> None:
        settings = _replace(QUICK, arms=("C",), n_trials=0)
        X, y, _ = bench.load_dataset("positive-control", settings)
        records = list(
            bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None))
        )
        for record in records:
            assert record["tuning"] is None
            assert record["tuning_seconds"] == 0.0
            assert record["hyperparameters"] == settings.defaults


# ---------------------------------------------------------------------------
# --quick: one fold of one small grid point, every field of every line
# ---------------------------------------------------------------------------

ROWS = [
    ("A", "linear"),
    ("A", "mlp"),
    ("B", "linear"),
    ("B", "mlp"),
    ("C", "linear"),
    ("C", "mlp"),
    ("D", "linear"),
    ("D", "mlp"),
    ("D-hybrid", "hybrid"),
]
FIELDS = {
    "dataset",
    "data_sha256",
    "label_bonds",
    "n_samples",
    "n_positives",
    "n_raw_features",
    "raw_constant_columns",
    "feature_constant_columns",
    "arm",
    "head",
    "representation",
    "feature_map_trained",
    "geometry",
    "grid_shape",
    "v_over_omega",
    "gamma_over_omega",
    "spacing_um",
    "omega",
    "omega_t",
    "evolution_time_us",
    "n_steps",
    "feature_map",
    "reused_from_v_over_omega",
    "reused_from_geometry",
    "measurement",
    "shots",
    "shot_stream_seed",
    "fold",
    "n_splits",
    "train_idx",
    "val_idx",
    "test_idx",
    "n_synthetic",
    "pca",
    "seeds",
    "feature_dim",
    "n_parameters",
    "n_frozen_parameters",
    "model",
    "hyperparameters",
    "tuning",
    "mcc",
    "pr_auc",
    "threshold",
    "epochs",
    "feature_seconds",
    "feature_cache_hit",
    "feature_environment",
    "train_seconds",
    "tuning_seconds",
    "separation",
    "settings",
    "environment",
}


@pytest.mark.slow
class TestQuick:
    def test_raw_rounding_column_is_listed_in_result_line(self) -> None:
        settings = _replace(QUICK, arms=("C",), n_trials=0, max_epochs=1)
        X, y, _ = bench.load_dataset("positive-control", settings)
        X[:, 0] = 1e9 + np.where(np.arange(X.shape[0]) % 2 == 0, 0.0, np.spacing(1e9))
        records = list(
            bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None))
        )
        assert records
        for record in records:
            assert 0 in record["raw_constant_columns"]
            assert 0 in json.loads(bench.to_json_line(record))["raw_constant_columns"]

    def test_one_line_per_arm_and_head_of_one_fold_and_one_grid_point(self, quick: Any) -> None:
        assert [(r["arm"], r["head"]) for r in quick.lines] == ROWS
        assert {r["fold"] for r in quick.lines} == {0}
        assert {(r["v_over_omega"], r["gamma_over_omega"]) for r in quick.lines[:4]} == {
            (1.0, 0.1)
        }
        assert {(r["v_over_omega"], r["gamma_over_omega"]) for r in quick.lines[4:]} == {
            (None, None)
        }
        lines = quick.text.splitlines()
        assert len(lines) == len(ROWS)
        # The control's raw columns are continuous draws, none of them near
        # constant (σ ≥ 0.9 against values of at most 3.8 here), so no fitting
        # rows of the run leave one only centred.  A nonzero list is tested in
        # ``TestRecordedConstantColumns``.
        assert quick.X.std(axis=0).min() > 0.1 * np.abs(quick.X).max()
        for line, record in zip(lines, quick.lines, strict=True):
            assert set(_strict_loads(line)) == FIELDS == set(record)
            assert record["raw_constant_columns"] == []
            assert record["feature_constant_columns"] == [] or all(
                0 <= column < record["feature_dim"]
                for column in record["feature_constant_columns"]
            )
            tuning = record["tuning"]
            assert tuning["raw_constant_columns"] == [[]] * QUICK.inner_folds
            assert len(tuning["feature_constant_columns"]) == QUICK.inner_folds
            assert len(tuning["pca"]) == QUICK.inner_folds

    def test_every_arm_has_the_same_fold_rows_and_seeds(self, quick: Any) -> None:
        """Compared between the lines, and against the fold rebuilt by hand."""
        fold = quick.fold
        first = quick.lines[0]
        for record in quick.lines:
            for key in ("train_idx", "val_idx", "test_idx", "n_synthetic", "seeds", "n_splits"):
                assert record[key] == first[key], (record["arm"], key)
        assert first["train_idx"] == np.sort(fold.train_idx).tolist()
        assert first["val_idx"] == np.sort(fold.val_idx).tolist()
        assert first["test_idx"] == np.sort(fold.test_idx).tolist()
        assert first["n_synthetic"] == fold.n_synthetic
        rows = first["train_idx"] + first["val_idx"] + first["test_idx"]
        assert sorted(rows) == list(range(QUICK.n_samples))  # a partition of the dataset
        # a quarter of each class, the first fold taking a remainder: 40 or 41 of 160 rows
        assert len(first["test_idx"]) in (40, 41)
        assert set(first["train_idx"]).isdisjoint(first["val_idx"])
        run_root = np.random.SeedSequence([QUICK.random_state, 1 << 16])
        data_seed, shot_seed, circuit_seed = _draw(run_root, 3)
        assert first["seeds"] == {
            "random_state": 0,
            **fold.seeds,
            "data_seed": data_seed,
            "shot_seed": shot_seed,
            "circuit_seed": circuit_seed,
        }
        assert quick.data_seed == data_seed
        assert len(set(first["seeds"].values())) == len(first["seeds"])

    def test_dataset_settings_and_environment(self, quick: Any) -> None:
        for record in quick.lines:
            assert record["dataset"] == "positive-control"
            assert record["data_sha256"] == fingerprint(quick.X, quick.y)
            assert record["n_samples"] == 160
            assert record["n_positives"] == int(quick.y.sum())
            assert record["n_raw_features"] == 15
            assert record["environment"] == environment()
            # computed in this run, in this environment; arms without a fixed map have none
            assert record["feature_environment"] == (
                environment() if record["arm"] in ("A", "B", "D") else None
            )
            assert record["settings"] == json.loads(json.dumps(QUICK.as_dict()))
            assert record["measurement"] == "exact"
            assert record["shots"] is None

    def test_grid_fields(self, quick: Any) -> None:
        for record in quick.lines[:4]:
            assert record["omega"] == pytest.approx(OMEGA)
            assert record["omega_t"] == pytest.approx(math.pi)
            assert record["evolution_time_us"] == pytest.approx(0.25)  # π / 4π
            assert record["spacing_um"] == pytest.approx(8.69, abs=0.005)  # R_b of the page
            assert record["n_steps"] == 40
            by_hand = _map_by_hand(4, 1.0, 0.1, 40, interactions=record["arm"] == "A")
            assert record["feature_map"] == {
                "feature_map": "RydbergFeatureMap",
                "config": by_hand.get_config(),
                "shots": None,
                "shot_seed": None,
            }
            assert record["representation"] == (
                "rydberg" if record["arm"] == "A" else "rydberg-noninteracting"
            )
        for record in quick.lines[4:]:
            for key in ("spacing_um", "omega", "omega_t", "evolution_time_us", "n_steps"):
                assert record[key] is None
        assert [r["feature_map"] for r in quick.lines[4:6]] == [None, None]
        assert quick.lines[6]["feature_map"]["feature_map"] == "QuantumEncodingLayer"
        assert quick.lines[8]["feature_map"] is None

    def test_dimensions_and_parameter_counts(self, quick: Any) -> None:
        """Linear ``d + 1 = 5``; MLP ``hidden (d + 2) + 1 = 25``; hybrid ``24 + 5``."""
        expected = {"linear": 5, "mlp": 25, "hybrid": 29}
        classes = {
            "linear": "LinearClassifier",
            "mlp": "ClassicalBaseline",
            "hybrid": "HybridBinaryClassifier",
        }
        for record in quick.lines:
            assert record["feature_dim"] == 4
            assert record["n_parameters"] == expected[record["head"]]
            assert record["model"]["class"] == classes[record["head"]]
            assert record["model"]["config"]["init_seed"] == record["seeds"]["init_seed"]
            assert record["n_frozen_parameters"] == (24 if record["arm"] == "D" else 0)
            assert record["feature_map_trained"] is (record["arm"] == "D-hybrid")
        # The MLP on the inputs has the parameter count of the MLP head in A.
        assert _line(quick, "C", "mlp")["n_parameters"] == _line(quick, "A", "mlp")["n_parameters"]
        assert _line(quick, "C", "mlp")["model"] == _line(quick, "A", "mlp")["model"]
        assert _line(quick, "C", "linear")["model"] == _line(quick, "B", "linear")["model"]

    def test_tuning_budget_is_the_same_for_every_line(self, quick: Any) -> None:
        for record in quick.lines:
            assert record["tuning"]["n_trials"] == 2
            assert record["tuning"]["inner_folds"] == 2
            assert record["tuning"]["candidates"] == [{"lr": 0.01}, {"lr": 0.05}]
            scores = record["tuning"]["scores"]
            assert len(scores) == 2
            assert all(-1.0 <= s <= 1.0 for s in scores)
            best = 0.01 if scores[0] >= scores[1] else 0.05
            assert record["hyperparameters"] == {
                "lr": best,
                "max_epochs": 6,
                "batch_size": 32,
                "patience": None,
            }

    def test_seconds_are_recorded_separately(self, quick: Any) -> None:
        for record in quick.lines:
            assert record["train_seconds"] > 0
            assert record["tuning_seconds"] > 0
            assert record["epochs"] == 6
            simulated = record["arm"] in ("A", "B", "D")
            assert (record["feature_seconds"] > 0) is simulated
            assert record["feature_cache_hit"] is False  # a fresh cache directory
        # Both heads of an arm read the same features: the same simulation time.
        for arm in ("A", "B", "D"):
            assert (
                _line(quick, arm, "linear")["feature_seconds"]
                == _line(quick, arm, "mlp")["feature_seconds"]
            )
        # per simulated arm: the fold's rows and one matrix per tuning fold
        assert len(list(quick.cache.glob("*.npz"))) == 3 * (1 + QUICK.inner_folds)

    def _features(self, quick: Any, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The arm's features of the training part's real rows and of the test rows, by hand."""
        fold = quick.fold
        real = np.concatenate([fold.train_idx, fold.val_idx])
        if arm in ("C", "D-hybrid"):
            return fold.inputs[real], fold.inputs[fold.test_idx], real
        if arm == "D":
            seed = _draw(np.random.SeedSequence([QUICK.random_state, 1 << 16]), 3)[2]
            return (
                _circuit_by_hand(fold.inputs[real], seed),
                _circuit_by_hand(fold.inputs[fold.test_idx], seed),
                real,
            )
        fmap = _map_by_hand(4, 1.0, 0.1, 40, interactions=arm == "A")
        return (
            fmap.transform(fold.inputs[real]).numpy(),
            fmap.transform(fold.inputs[fold.test_idx]).numpy(),
            real,
        )

    @pytest.mark.parametrize("arm", ["A", "B", "C", "D", "D-hybrid"])
    def test_separation_measures_of_the_features_the_map_returns(
        self, quick: Any, arm: str
    ) -> None:
        """On the unstandardised features (#518), although the heads see them standardised."""
        train, test, real = self._features(quick, arm)
        expected = {
            "train": separation_measures(train, quick.y[real]).to_dict(),
            "test": separation_measures(test, quick.y[quick.fold.test_idx]).to_dict(),
        }
        for head in ("hybrid",) if arm == "D-hybrid" else ("linear", "mlp"):
            recorded = _line(quick, arm, head)["separation"]
            assert set(recorded) == {"train", "test"}
            for part in ("train", "test"):
                assert recorded[part] == pytest.approx(expected[part], rel=1e-7, abs=1e-12)
        n_test = quick.fold.test_idx.size
        assert expected["train"]["n_samples"] == 160 - n_test  # no synthetic row among them
        assert expected["test"]["n_samples"] == n_test
        assert expected["train"]["n_features"] == 4

    def test_arm_b_separation_is_that_of_single_atom_features(self, quick: Any) -> None:
        """The recorded B line, from four one-atom maps: no interaction entered it."""
        fold = quick.fold
        alone = RydbergFeatureMap(
            AtomRegister.chain(1, 1.0),
            PulseEncoding(1, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.1 * OMEGA,
            n_steps=40,
        )
        test = fold.inputs[fold.test_idx]
        features = np.column_stack([alone.transform(test[:, [i]]).numpy()[:, 0] for i in range(4)])
        expected = separation_measures(features, quick.y[fold.test_idx]).to_dict()
        recorded = _line(quick, "B", "linear")["separation"]["test"]
        assert recorded == pytest.approx(expected, rel=1e-7, abs=1e-12)
        other = _line(quick, "A", "linear")["separation"]["test"]
        assert other["fisher_ratio"] != pytest.approx(expected["fisher_ratio"], rel=1e-3)

    @pytest.mark.parametrize(
        ("arm", "head"),
        [("C", "linear"), ("A", "mlp"), ("B", "linear"), ("D", "linear"), ("D", "mlp")],
    )
    def test_scores_of_a_model_trained_by_hand(self, quick: Any, arm: str, head: str) -> None:
        """
        MCC, PR-AUC, threshold and epochs of a line, from a model built and
        trained here with the recorded hyperparameters on the fold rebuilt by
        hand.  For arm C with the linear head this is the logistic regression.
        The features are standardised by hand with the mean and population
        standard deviation of the real training rows (#518).
        """
        fold = quick.fold
        record = _line(quick, arm, head)
        assert record["settings"]["standardise"] is True

        # One matrix for the fold's rows, as the script evolves them: a batch
        # of another size may round differently in the last digit (#509).
        stacked = np.concatenate(
            [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
        )
        if arm == "C":
            raw = stacked
        elif arm == "D":
            circuit_seed = _draw(np.random.SeedSequence([QUICK.random_state, 1 << 16]), 3)[2]
            raw = _circuit_by_hand(stacked, circuit_seed)
        else:
            fmap = _map_by_hand(4, 1.0, 0.1, 40, interactions=arm == "A")
            raw = fmap.transform(stacked).numpy()
        _assert_scores_by_hand(record, fold, quick.y, _standardised_by_hand(raw, fold), head)

    def test_records_returned_are_the_lines_written(self, quick: Any) -> None:
        assert len(quick.records) == len(quick.lines)
        for returned, read in zip(quick.records, quick.lines, strict=True):
            assert returned["mcc"] == read["mcc"]
            assert returned["test_idx"].tolist() == read["test_idx"]
        table = bench.summarise(quick.records)
        assert len(table.splitlines()) == 1 + len(ROWS)


# ---------------------------------------------------------------------------
# Seeded runs, against numbers recorded before the helpers were shared (#524)
# ---------------------------------------------------------------------------

_QUICK_CHOSEN = {"lr": 0.05, "max_epochs": 6, "batch_size": 32, "patience": None}
_QUICK_CANDIDATES = [{"lr": 0.01}, {"lr": 0.05}]

#: ``(arm, head, mcc, pr_auc, threshold, epochs, hyperparameters, tuning candidates,
#: tuning scores)`` of every line of the ``--quick`` run (the ``quick`` fixture), in
#: order, as read back from the file.  Recorded on c6ce325 (``main`` before #524, whose
#: own ``fit_and_score`` and ``sample_configs`` the run went through) with Python 3.13 and
#: the locked dependencies, as the ``repr`` of each value of ``read_records`` of a
#: ``main(["--quick", ...])`` file.  On the declared floors (Python 3.11, torch 2.3.0,
#: PennyLane 0.45) the same run differs only in the threshold of ``D-hybrid``, by 1.1e-7
#: relative (float32 rounding in its trained circuit); with the newest dependencies on
#: Python 3.14 it is equal.  Hence ``rel=1e-6`` for thresholds and PR-AUCs and equality for
#: the rest.
PINNED_QUICK: list[tuple[Any, ...]] = [
    ("A", "linear", -0.20558916985752212, 0.5329118467546871, 0.3602352738380432, 6, _QUICK_CHOSEN, _QUICK_CANDIDATES, [0.2990380270712763, 0.3766888457297747]),
    ("A", "mlp", 0.4002254054659389, 0.820174388605423, 0.7212197780609131, 6, _QUICK_CHOSEN, _QUICK_CANDIDATES, [0.4635477431867409, 0.5250485620506561]),
    ("B", "linear", 0.14815570653432425, 0.5465996863888339, 0.6279727220535278, 6, _QUICK_CHOSEN, _QUICK_CANDIDATES, [0.3431302575779871, 0.3730147033864202]),
    ("B", "mlp", 0.4552444556393538, 0.7846675347791451, 0.4923308491706848, 6, _QUICK_CHOSEN, _QUICK_CANDIDATES, [0.3323368410689924, 0.4683972478385392]),
    ("C", "linear", 0.23018043421059298, 0.6799227754498263, 0.6437177062034607, 6, _QUICK_CHOSEN, _QUICK_CANDIDATES, [0.4321224662491732, 0.458441142352223]),
    ("C", "mlp", 0.5683600322805923, 0.7793636709763081, 0.5460516214370728, 6, _QUICK_CHOSEN, _QUICK_CANDIDATES, [0.38242515233508956, 0.3964050669890513]),
    ("D", "linear", 0.3651561712358987, 0.6299677873719636, 0.5444587469100952, 6, _QUICK_CHOSEN, _QUICK_CANDIDATES, [0.20787000637835967, 0.22999276535327426]),
    ("D", "mlp", 0.17996712863233494, 0.6731643962297735, 0.5894492864608765, 6, _QUICK_CHOSEN, _QUICK_CANDIDATES, [0.2642957782445727, 0.3462264656229034]),
    ("D-hybrid", "hybrid", 0.223367231551576, 0.6580915539517214, 0.556437075138092, 6, _QUICK_CHOSEN, _QUICK_CANDIDATES, [0.25875104190884707, 0.3656388625821353]),
]  # fmt: skip

_DRAWN_CANDIDATES = [
    {"batch_size": 16, "lr": 0.03},
    {"batch_size": 32, "lr": 0.01},
    {"batch_size": 32, "lr": 0.03},
]

#: The same fields for the lines of :func:`_drawn_run`, recorded with :data:`PINNED_QUICK`
#: (``run_dataset`` of that run's settings, without a cache).  The linear head's first and
#: third candidates tie, and the first is chosen.
PINNED_DRAWN: list[tuple[Any, ...]] = [
    ("C", "linear", 0.23018043421059298, 0.6780477754498262, 0.6350880861282349, 3, {"lr": 0.03, "max_epochs": 6, "batch_size": 16, "patience": 2}, _DRAWN_CANDIDATES, [0.4453757000449171, 0.41655436046835304, 0.4453757000449171]),
    ("C", "mlp", 0.42516542614512887, 0.7093699492764305, 0.5214287042617798, 4, {"lr": 0.03, "max_epochs": 6, "batch_size": 32, "patience": 2}, _DRAWN_CANDIDATES, [0.38049407261126644, 0.38242515233508956, 0.39746943174633065]),
]  # fmt: skip


def _drawn_run() -> list[dict[str, Any]]:
    """
    Arm C with a search space of six configurations and a budget of three, so tuning
    draws them (``--quick``'s two candidates are its whole grid), and early stopping.
    """
    settings = _replace(
        QUICK,
        arms=("C",),
        search_space={"lr": (0.01, 0.03, 0.05), "batch_size": (16, 32)},
        n_trials=3,
        patience=2,
    )
    X, y, _ = bench.load_dataset("positive-control", settings)
    return list(bench.run_dataset("positive-control", X, y, settings, bench.FeatureCache(None)))


def _assert_pinned(records: list[dict[str, Any]], pinned: list[tuple[Any, ...]]) -> None:
    assert len(records) == len(pinned)
    for record, row in zip(records, pinned, strict=True):
        arm, head, mcc, pr, threshold, epochs, chosen, candidates, scores = row
        assert (record["arm"], record["head"]) == (arm, head)
        assert (record["mcc"], record["epochs"], record["hyperparameters"]) == (
            mcc,
            epochs,
            chosen,
        )
        assert record["tuning"]["candidates"] == candidates
        assert record["tuning"]["scores"] == scores
        assert record["threshold"] == pytest.approx(threshold, rel=1e-6)
        assert record["pr_auc"] == pytest.approx(pr, rel=1e-6)


class TestSeededRunsAreUnchanged:
    """
    The script's numbers are those of ``main`` before its tuning draw and its
    fit-and-score were shared with ``hqnn_forge.benchmark`` (#524).

    Every line goes through the parts that moved: the candidates of tuning, each trial's
    training and validation threshold, and the final fit's MCC at that threshold, PR-AUC
    and epochs.
    """

    def test_the_quick_run_is_as_recorded(self, quick: Any) -> None:
        _assert_pinned(quick.lines, PINNED_QUICK)

    def test_a_run_whose_tuning_draws_from_the_grid_is_as_recorded(self) -> None:
        _assert_pinned(_drawn_run(), PINNED_DRAWN)


class TestShotsRun:
    def test_the_measurement_assumption_is_recorded_per_line(self, tmp_path: Path) -> None:
        settings = _replace(QUICK, arms=("A", "C"), shots=32, n_trials=0)
        data = {"positive-control": bench.load_dataset("positive-control", settings)}
        records = bench.run(data, settings, tmp_path / "shots.jsonl", cache_dir=tmp_path / "c")
        read = bench.read_records(tmp_path / "shots.jsonl")
        assert [(r["arm"], r["measurement"], r["shots"]) for r in read] == [
            ("A", "shots", 32),
            ("A", "shots", 32),
            ("C", "exact", None),
            ("C", "exact", None),
        ]
        X, y, _ = data["positive-control"]
        fold = _fold_by_hand(X, y, settings, 0)
        shot_seed = records[0]["seeds"]["shot_seed"]
        assert read[0]["feature_map"]["shots"] == 32
        assert read[0]["feature_map"]["shot_seed"] == shot_seed
        # The script's features of the fold's rows, drawn by hand with the seed of that matrix.
        stacked = np.concatenate(
            [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
        )
        seed = _stream_seed_by_hand(read[0]["feature_map"], stacked)
        assert read[0]["shot_stream_seed"] == read[1]["shot_stream_seed"] == seed
        assert read[2]["shot_stream_seed"] is None  # arm C draws nothing
        sampled = _map_by_hand(4, 1.0, 0.1, 40).transform(
            stacked, shots=32, generator=torch.Generator().manual_seed(seed)
        )
        test = sampled[-fold.test_idx.size :].numpy()
        expected = separation_measures(test, y[fold.test_idx]).to_dict()
        assert read[0]["separation"]["test"] == pytest.approx(expected, rel=1e-9)
        exact = _map_by_hand(4, 1.0, 0.1, 40).transform(fold.inputs[fold.test_idx]).numpy()
        assert np.abs(test - exact).max() > 0.01  # sampled, not exact


# ---------------------------------------------------------------------------
# Shot streams: one per feature matrix (#523)
# ---------------------------------------------------------------------------


def _stream_seed_by_hand(config: dict[str, Any], inputs: np.ndarray) -> int:
    """
    The seed of a matrix's shot stream, written out: the first four bytes,
    big-endian, of the SHA-256 of the canonical JSON of the matrix's config
    and of the SHA-256 of its inputs' shape and float64 bytes.
    """
    contiguous = np.ascontiguousarray(inputs, dtype=np.float64)
    inputs_sha256 = hashlib.sha256(repr(contiguous.shape).encode() + contiguous.tobytes())
    text = json.dumps({"config": config, "inputs": inputs_sha256.hexdigest()}, sort_keys=True)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def _rydberg_config_by_hand(map_config: dict[str, Any], shots: int, shot_seed: int) -> Any:
    """``Representation.config`` of a chain map with shots, typed in."""
    return {
        "feature_map": "RydbergFeatureMap",
        "config": map_config,
        "shots": shots,
        "shot_seed": shot_seed,
        "shot_stream": "per-matrix",
    }


def _torch_uniforms(state: torch.Tensor, n: int = 256) -> np.ndarray:
    """The next ``n`` float64 uniforms of a torch generator in ``state``."""
    generator = torch.Generator()
    generator.set_state(state)
    return torch.rand(n, generator=generator, dtype=torch.float64).numpy()


def _numpy_uniforms(seed: int, n: int = 256) -> np.ndarray:
    """The first ``n`` uniforms of ``numpy.random.default_rng(seed)``, a sampling device's generator."""
    return np.random.default_rng(seed).random(n)


def _share_no_draw(streams: list[np.ndarray]) -> bool:
    """Whether no two of ``streams`` hold the same value at any one position."""
    return all(
        not np.any(streams[i] == streams[j])
        for i in range(len(streams))
        for j in range(i + 1, len(streams))
    )


def _spy_on_draws(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """
    Every sampled ``RydbergFeatureMap.transform`` call (the map's config, the
    inputs, the seed and the state of its generator on entry) and every seed
    arm D's sampling device is built with, in order.
    """
    spied = SimpleNamespace(transforms=[], circuit_seeds=[])
    transform = RydbergFeatureMap.transform
    frozen_circuit = bench.frozen_circuit

    def spy_transform(self: Any, X: Any, *args: Any, **kwargs: Any) -> Any:
        generator = kwargs.get("generator")
        if generator is not None:
            spied.transforms.append(
                SimpleNamespace(
                    config=self.get_config(),
                    inputs=torch.as_tensor(X).numpy().astype(np.float64),
                    seed=generator.initial_seed(),
                    state=generator.get_state(),
                )
            )
        return transform(self, X, *args, **kwargs)

    def spy_circuit(settings: Any, circuit_seed: int, shot_seed: int | None = None) -> Any:
        if shot_seed is not None:
            spied.circuit_seeds.append(shot_seed)
        return frozen_circuit(settings, circuit_seed, shot_seed)

    monkeypatch.setattr(RydbergFeatureMap, "transform", spy_transform)
    monkeypatch.setattr(bench, "frozen_circuit", spy_circuit)
    return spied


#: Two folds with tuning, two interaction strengths, arms A, B (reused at the
#: second ``V/Ω``) and D, with shots.
STREAM_SETTINGS = _replace(
    QUICK,
    arms=("A", "B", "D"),
    shots=16,
    v_over_omega=(1.0, 3.0),
    max_folds=2,
    n_trials=1,
)


@pytest.fixture(scope="class")
def stream_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A run of :data:`STREAM_SETTINGS` without a cache, with every draw it made."""
    settings = STREAM_SETTINGS
    data = {"positive-control": bench.load_dataset("positive-control", settings)}
    out = tmp_path_factory.mktemp("streams") / "streams.jsonl"
    with pytest.MonkeyPatch.context() as monkeypatch:
        spied = _spy_on_draws(monkeypatch)
        records = bench.run(data, settings, out, cache_dir=None)
    return SimpleNamespace(
        records=records,
        lines=bench.read_records(out),
        transforms=spied.transforms,
        circuit_seeds=spied.circuit_seeds,
        shot_seed=bench.run_seeds(settings.random_state)["shot_seed"],
    )


@pytest.mark.slow
class TestShotStreams:
    """
    Every feature matrix is drawn from a stream of its own, seeded from
    ``shot_seed``, the matrix's config and its inputs (#523): folds, tuning
    folds, grid points and arms share no draw, and the same matrix is drawn
    from the same stream whenever it is computed.
    """

    X = np.random.default_rng(7).uniform(-math.pi, math.pi, size=(9, 4))

    def _with_another_tail(self) -> np.ndarray:
        """:attr:`X` with its first row kept and every other row redrawn: another fold's matrix."""
        other = np.random.default_rng(8).uniform(-math.pi, math.pi, size=self.X.shape)
        other[0] = self.X[0]
        return other

    def test_two_folds_do_not_share_draws_on_a_row_they_have_in_common(self) -> None:
        """
        Row 0 has the same populations in both matrices.  Drawn from one
        stream, as before #523, it would get the same bitstrings and the same
        estimates in both (checked by hand below); from two streams it does
        not, and the two streams share no uniform.
        """
        settings = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.0,), shots=64)
        other = self._with_another_tail()
        reps = bench.representations(_replace(settings, arms=("A", "D")))
        rep_a, rep_d = reps[0], reps[1]
        assert (rep_a.arm, rep_d.arm) == ("A", "D")
        for rep in (rep_a, rep_d):
            first, second = rep.compute(self.X), rep.compute(other)
            assert not np.array_equal(first[0], second[0])
            seeds = [_stream_seed_by_hand(rep.config, m) for m in (self.X, other)]
            assert seeds[0] != seeds[1]
            if rep.arm == "A":
                streams = [
                    _torch_uniforms(torch.Generator().manual_seed(s).get_state()) for s in seeds
                ]
            else:
                streams = [_numpy_uniforms(s) for s in seeds]
            assert _share_no_draw(streams)

        # What one shared stream gives: the same estimates for the shared row.
        fmap = _map_by_hand(4, 1.0, 0.0, 1)
        shared = [
            fmap.transform(m, shots=64, generator=torch.Generator().manual_seed(5)).numpy()
            for m in (self.X, other)
        ]
        np.testing.assert_array_equal(shared[0][0], shared[1][0])
        circuit_seed = bench.run_seeds(settings.random_state)["circuit_seed"]
        shared_d = []
        for m in (self.X, other):
            with torch.no_grad():
                circuit = bench.frozen_circuit(settings, circuit_seed, 5)
                shared_d.append(circuit(torch.from_numpy(m)).double().numpy())
        np.testing.assert_array_equal(shared_d[0][0], shared_d[1][0])

    def test_every_matrix_of_a_run_has_a_stream_of_its_own(self, stream_run: Any) -> None:
        """
        Folds, tuning folds, grid points and arms: every matrix the run
        sampled was drawn from the seed of the rule, and no two share a
        uniform at any position, in arms A and B (one torch generator per
        matrix) as in arm D (one sampling device per matrix).
        """
        settings = STREAM_SETTINGS
        transforms = stream_run.transforms
        # 2 folds × (1 + 2 tuning folds) × (A at two V/Ω + B once): B is reused at V/Ω = 3.
        per_matrix = settings.max_folds * (1 + settings.inner_folds)
        assert len(transforms) == per_matrix * 3
        assert sum(t.config["interactions"] for t in transforms) == per_matrix * 2
        for t in transforms:
            config = _rydberg_config_by_hand(t.config, 16, stream_run.shot_seed)
            assert t.seed == _stream_seed_by_hand(config, t.inputs)
        assert len({t.seed for t in transforms}) == len(transforms)
        assert _share_no_draw([_torch_uniforms(t.state) for t in transforms])

        circuit_seeds = stream_run.circuit_seeds
        assert len(circuit_seeds) == per_matrix
        assert len(set(circuit_seeds)) == per_matrix
        assert _share_no_draw([_numpy_uniforms(s) for s in circuit_seeds])
        # and A, B and D draw from different seeds too
        assert not {t.seed for t in transforms} & set(circuit_seeds)

    def test_the_line_records_the_seed_of_its_matrices(self, stream_run: Any) -> None:
        """
        ``shot_stream_seed`` is the seed of the fold's matrix and
        ``tuning["shot_stream_seeds"]`` those of its tuning folds' matrices,
        so the seeds on the lines are exactly the seeds the run drew from.
        """
        by_arm: dict[str, set[int]] = {"A": set(), "B": set(), "D": set()}
        for line in stream_run.lines:
            if line["arm"] in ("C", "D-hybrid"):
                assert line["shot_stream_seed"] is None
                assert line["tuning"]["shot_stream_seeds"] == [None] * STREAM_SETTINGS.inner_folds
                continue
            assert line["feature_map"]["shot_stream"] == "per-matrix"
            seeds = [line["shot_stream_seed"], *line["tuning"]["shot_stream_seeds"]]
            by_arm[line["arm"]].update(seeds)
        assert by_arm["A"] | by_arm["B"] == {t.seed for t in stream_run.transforms}
        assert by_arm["D"] == set(stream_run.circuit_seeds)
        # The fold's own matrix: the seed of the transform call on the fold's stacked inputs.
        fold_a = [
            r
            for r in stream_run.lines
            if r["arm"] == "A" and r["head"] == "linear" and r["v_over_omega"] == 1.0
        ]
        assert [r["fold"] for r in fold_a] == [0, 1]
        sizes = {t.seed: t.inputs.shape[0] for t in stream_run.transforms}
        for r in fold_a:
            n_rows = (
                r["n_synthetic"] + len(r["train_idx"]) + len(r["val_idx"]) + len(r["test_idx"])
            )
            assert sizes[r["shot_stream_seed"]] == n_rows
        # fold 0 by hand: the seed of its stacked inputs (training with synthetic, validation, test)
        X, y, _ = bench.load_dataset("positive-control", STREAM_SETTINGS)
        fold = _fold_by_hand(X, y, STREAM_SETTINGS, 0)
        stacked = np.concatenate(
            [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
        )
        map_config = _map_by_hand(4, 1.0, 0.1, 40).get_config()
        config = _rydberg_config_by_hand(map_config, 16, stream_run.shot_seed)
        assert fold_a[0]["feature_map"] == config
        assert fold_a[0]["shot_stream_seed"] == _stream_seed_by_hand(config, stacked)

    def test_a_reused_arm_b_line_shares_the_draws_of_its_source(self, stream_run: Any) -> None:
        """
        Arm B at ``V/Ω = 3`` is the matrix computed at ``V/Ω = 1``, reused:
        the same draws, and the line records its source's seeds.
        """
        for fold in (0, 1):
            source, reused = [
                r
                for r in stream_run.lines
                if r["arm"] == "B" and r["head"] == "linear" and r["fold"] == fold
            ]
            assert reused["reused_from_v_over_omega"] == source["v_over_omega"] == 1.0
            assert reused["shot_stream_seed"] == source["shot_stream_seed"]
            assert reused["tuning"]["shot_stream_seeds"] == source["tuning"]["shot_stream_seeds"]
            assert reused["separation"] == source["separation"]

    def test_the_same_matrix_is_drawn_from_the_same_stream(self, tmp_path: Path) -> None:
        """
        The units meant to share draws are the requests of one matrix: the
        same inputs under the same config, from maps built anew, give the
        same stream and the same features bit for bit.  The labels do not
        enter: two cache entries that differ in their labels only hold the
        same features.
        """
        settings = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.0,), shots=64)
        for arm in ("A", "D"):
            first, second = (
                next(
                    r
                    for r in bench.representations(_replace(settings, arms=(arm,)))
                    if r.arm == arm
                )
                for _ in range(2)
            )
            assert first.config == second.config
            assert bench.shot_stream_seed(first.config, self.X) == _stream_seed_by_hand(
                second.config, self.X.copy()
            )
            np.testing.assert_array_equal(first.compute(self.X), second.compute(self.X.copy()))
        (rep,) = bench.representations(_replace(settings, arms=("A",)))
        cache = bench.FeatureCache(tmp_path)
        labels = np.array([0, 1] * 4 + [0], dtype=np.int64)
        other_labels = 1 - labels
        assert cache.path(self.X, labels, rep.config) != cache.path(
            self.X, other_labels, rep.config
        )
        one = cache.fetch(self.X, labels, rep.config, rep.compute)
        two = cache.fetch(self.X, other_labels, rep.config, rep.compute)
        assert (one.hit, two.hit) == (False, False)
        np.testing.assert_array_equal(one.features, two.features)

    def test_a_stream_needs_shots_and_a_seed(self) -> None:
        """Exact features draw nothing, and a sampling circuit without a seed is refused."""
        exact = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.0,), arms=("A",))
        (rep,) = bench.representations(exact)
        assert "shot_stream" not in rep.config
        with pytest.raises(ValueError, match="shots and an integer shot_seed"):
            bench.shot_stream_seed(rep.config, self.X)
        with pytest.raises(ValueError, match="shot_seed"):
            bench.frozen_circuit(bench.Settings(n_atoms=3, shots=8), 1)

    def test_a_repeated_and_a_resumed_run_reproduce_every_matrix_bit_for_bit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream_run: Any
    ) -> None:
        """
        Every matrix the run fetched, again without a cache, and once more
        resumed from a cache that a run of arm B alone filled first, so that
        arm A's matrices are sampled after fewer others: bit for bit the same,
        in the same order, with the same seeds on the lines.
        """
        settings = STREAM_SETTINGS
        data = {"positive-control": bench.load_dataset("positive-control", settings)}
        fetched: list[np.ndarray] = []
        fetch = bench.FeatureCache.fetch

        def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = fetch(self, *args, **kwargs)
            fetched.append(result.features)
            return result

        monkeypatch.setattr(bench.FeatureCache, "fetch", spy)
        runs = []
        for cache_dir in (None, tmp_path / "cache"):
            if cache_dir is not None:
                bench.run(
                    data,
                    _replace(settings, arms=("B",)),
                    tmp_path / "b.jsonl",
                    cache_dir=cache_dir,
                )
            fetched.clear()
            records = bench.run(data, settings, tmp_path / "run.jsonl", cache_dir=cache_dir)
            runs.append((list(fetched), records))
        (plain, plain_records), (resumed, resumed_records) = runs
        # Per fold and tuning fold: A at two V/Ω, B once (reused at the second), D.
        assert len(plain) == len(resumed) == 2 * (1 + 2) * 4
        for a, b in zip(plain, resumed, strict=True):
            assert a.tobytes() == b.tobytes()

        def seeds(records: list[dict[str, Any]]) -> list[Any]:
            return [(r["shot_stream_seed"], r["tuning"]["shot_stream_seeds"]) for r in records]

        assert seeds(plain_records) == seeds(resumed_records) == seeds(stream_run.records)
        assert [r["mcc"] for r in plain_records] == [r["mcc"] for r in stream_run.records]
        assert any(r["feature_cache_hit"] for r in resumed_records if r["arm"] == "B")


# ---------------------------------------------------------------------------
# Arm B reuse across interaction-grid values and separation-measure subsampling
# ---------------------------------------------------------------------------


class TestArmBReuse:
    def test_arm_b_computed_once_per_gamma_and_fold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Regression test for #519: Arm B feature map has no interactions, so its
        features, tuning, training and separation measures do not depend on V/Ω.
        Arm B should be computed once per (gamma, fold) and reused across all
        interaction-grid points.
        """
        settings = _replace(
            QUICK,
            arms=("A", "B"),
            n_atoms=2,
            n_trials=0,
            n_splits=2,
            max_folds=2,
            n_steps=10,
            v_over_omega=(0.5, 1.0, 2.0),
            gamma_over_omega=(0.0, 0.1),
        )
        rng = np.random.RandomState(42)
        X = rng.randn(40, 2)
        y = np.array([0, 1] * 20)
        data = {"positive-control": (X, y, 0)}

        calls: list[str] = []
        orig_features_of = bench.features_of

        def spy_features_of(rep: Any, split: Any, cache: Any) -> Any:
            calls.append(rep.arm)
            return orig_features_of(rep, split, cache)

        monkeypatch.setattr(bench, "features_of", spy_features_of)

        out = tmp_path / "reuse.jsonl"
        records = bench.run(data, settings, out, cache_dir=tmp_path / "cache")
        read = bench.read_records(out)
        assert len(records) == len(read)

        # 2 folds, 3 V values, 2 gamma values:
        # Arm A must run for every V and gamma: 2 folds * 2 gammas * 3 V = 12 times.
        # Arm B must run only once per (gamma, fold): 2 folds * 2 gammas = 4 times.
        assert calls.count("A") == 12
        assert calls.count("B") == 4

        # Verify output records:
        # Each fold has 2 gammas * 3 V values * 2 arms * 2 heads = 24 records per fold -> 48 records total
        assert len(records) == 2 * 2 * 3 * 2 * 2

        # Arm A records differ across V values (at least for training / separation)
        a_records = [
            r
            for r in records
            if r["arm"] == "A"
            and r["head"] == "linear"
            and r["fold"] == 0
            and r["gamma_over_omega"] == 0.1
        ]
        assert len(a_records) == 3
        assert [r["v_over_omega"] for r in a_records] == [0.5, 1.0, 2.0]

        # Arm B records are yielded for every V value to preserve schema and A/B pairing
        for gamma in (0.0, 0.1):
            b_records = [
                r
                for r in records
                if r["arm"] == "B"
                and r["head"] == "linear"
                and r["fold"] == 0
                and r["gamma_over_omega"] == gamma
            ]
            assert len(b_records) == 3
            assert [r["v_over_omega"] for r in b_records] == [0.5, 1.0, 2.0]

            # Each record retains the register of its own grid point
            reg0 = b_records[0]["feature_map"]["config"]["register"]
            reg1 = b_records[1]["feature_map"]["config"]["register"]
            reg2 = b_records[2]["feature_map"]["config"]["register"]
            assert reg0 != reg1 and reg1 != reg2
            assert b_records[0]["spacing_um"] != b_records[1]["spacing_um"]

            # First line is original; subsequent lines are marked as reused with zero timings
            assert b_records[0]["reused_from_v_over_omega"] is None
            assert b_records[1]["reused_from_v_over_omega"] == 0.5
            assert b_records[2]["reused_from_v_over_omega"] == 0.5
            assert b_records[1]["feature_seconds"] == 0.0
            assert b_records[1]["train_seconds"] == 0.0
            assert b_records[1]["tuning_seconds"] == 0.0
            assert b_records[2]["feature_seconds"] == 0.0
            assert b_records[2]["train_seconds"] == 0.0
            assert b_records[2]["tuning_seconds"] == 0.0

            # But Arm B results are mathematically identical across all V values for this gamma
            for k in ("mcc", "pr_auc", "threshold", "separation", "hyperparameters"):
                assert b_records[0][k] == b_records[1][k] == b_records[2][k]

        # Verify Arm B is not incorrectly reused across different dephasing rates
        b_g0 = [
            r
            for r in records
            if r["arm"] == "B"
            and r["head"] == "linear"
            and r["fold"] == 0
            and r["gamma_over_omega"] == 0.0
        ]
        b_g1 = [
            r
            for r in records
            if r["arm"] == "B"
            and r["head"] == "linear"
            and r["fold"] == 0
            and r["gamma_over_omega"] == 0.1
        ]
        assert b_g0[0]["separation"] != b_g1[0]["separation"]

    def test_arm_b_features_independent_of_spacing(self) -> None:
        """Non-interacting features (Arm B) do not depend on atom spacing or V/Omega."""
        settings = _replace(QUICK, n_atoms=3)
        map1 = bench.feature_map(
            settings, v_over_omega=0.5, gamma_over_omega=0.1, interactions=False
        )
        map2 = bench.feature_map(
            settings, v_over_omega=2.0, gamma_over_omega=0.1, interactions=False
        )
        assert map1.register != map2.register

        X = torch.tensor([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.2]], dtype=torch.float64)
        F1 = map1.transform(X)
        F2 = map2.transform(X)
        torch.testing.assert_close(F1, F2, atol=1e-12, rtol=1e-12)

        # Also verify exact noiseless evolution (gamma=0)
        map_nl1 = bench.feature_map(
            settings, v_over_omega=0.5, gamma_over_omega=0.0, interactions=False
        )
        map_nl2 = bench.feature_map(
            settings, v_over_omega=2.0, gamma_over_omega=0.0, interactions=False
        )
        assert map_nl1.register != map_nl2.register
        F_nl1 = map_nl1.transform(X)
        F_nl2 = map_nl2.transform(X)
        torch.testing.assert_close(F_nl1, F_nl2, atol=1e-12, rtol=1e-12)

    def test_increasing_v_grid_does_not_increase_arm_b_computation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arm B computation count depends only on gamma and folds, not the number of V grid points."""
        rng = np.random.RandomState(42)
        X = rng.randn(40, 2)
        y = np.array([0, 1] * 20)
        data = {"positive-control": (X, y, 0)}

        b_calls = 0
        orig_features_of = bench.features_of

        def spy_features_of(rep: Any, split: Any, cache: Any) -> Any:
            nonlocal b_calls
            if rep.arm == "B":
                b_calls += 1
            return orig_features_of(rep, split, cache)

        monkeypatch.setattr(bench, "features_of", spy_features_of)

        # 1 V value
        s1 = _replace(
            QUICK,
            arms=("B",),
            n_atoms=2,
            n_trials=0,
            n_splits=2,
            max_folds=2,
            n_steps=10,
            v_over_omega=(1.0,),
            gamma_over_omega=(0.1,),
        )
        bench.run(data, s1, tmp_path / "run1.jsonl", cache_dir=tmp_path / "cache1")
        assert b_calls == 2  # 2 folds

        # 4 V values
        b_calls = 0
        s4 = _replace(
            QUICK,
            arms=("B",),
            n_atoms=2,
            n_trials=0,
            n_splits=2,
            max_folds=2,
            n_steps=10,
            v_over_omega=(0.2, 0.5, 1.0, 2.0),
            gamma_over_omega=(0.1,),
        )
        bench.run(data, s4, tmp_path / "run4.jsonl", cache_dir=tmp_path / "cache4")
        assert b_calls == 2  # Still 2 folds, despite 4x V values!

    def test_arm_b_reused_results_match_single_grid_results(self, tmp_path: Path) -> None:
        """Reusing Arm B gives identical metrics and separation measures as running it alone."""
        rng = np.random.RandomState(42)
        X = rng.randn(40, 2)
        y = np.array([0, 1] * 20)
        data = {"positive-control": (X, y, 0)}

        s_single = _replace(
            QUICK,
            arms=("B",),
            n_atoms=2,
            n_trials=0,
            n_splits=2,
            n_steps=10,
            v_over_omega=(2.0,),
            gamma_over_omega=(0.1,),
        )
        r_single = bench.run(data, s_single, tmp_path / "s.jsonl", cache_dir=tmp_path / "cs")

        s_multi = _replace(
            QUICK,
            arms=("B",),
            n_atoms=2,
            n_trials=0,
            n_splits=2,
            n_steps=10,
            v_over_omega=(0.5, 1.0, 2.0),
            gamma_over_omega=(0.1,),
        )
        r_multi = bench.run(data, s_multi, tmp_path / "m.jsonl", cache_dir=tmp_path / "cm")

        # For V=2.0 in multi-grid, Arm B was reused from V=0.5 and matches single-grid run exactly
        matching = [r for r in r_multi if r["v_over_omega"] == 2.0]
        assert len(matching) == len(r_single)
        for r1, r2 in zip(r_single, matching, strict=True):
            assert r2["reused_from_v_over_omega"] == 0.5
            assert r2["feature_seconds"] == 0.0
            assert r2["train_seconds"] == 0.0
            assert r2["tuning_seconds"] == 0.0
            assert r1["mcc"] == r2["mcc"]
            assert r1["pr_auc"] == pytest.approx(r2["pr_auc"])
            assert r1["threshold"] == pytest.approx(r2["threshold"])
            assert r1["separation"] == r2["separation"]
            assert r1["hyperparameters"] == r2["hyperparameters"]

    def test_a_reused_line_records_the_cache_hit_of_its_source_line(self, tmp_path: Path) -> None:
        """
        A reused arm B line fetches nothing itself: its ``feature_cache_hit``,
        like its ``feature_environment`` and the hits of its tuning folds, is
        that of the line it was taken from, as the module docstring states
        (#541).  Cold cache: ``False`` on both; warm: ``True`` on both.
        """
        rng = np.random.RandomState(7)
        data = {"positive-control": (rng.randn(40, 2), np.array([0, 1] * 20), 0)}
        settings = _replace(
            QUICK,
            arms=("B",),
            n_atoms=2,
            n_trials=2,
            n_splits=2,
            n_steps=10,
            v_over_omega=(0.5, 1.0),
            gamma_over_omega=(0.1,),
        )
        cache = tmp_path / "cache"
        for name, hit in (("cold", False), ("warm", True)):
            records = bench.run(data, settings, tmp_path / f"{name}.jsonl", cache_dir=cache)
            source = [r for r in records if r["reused_from_v_over_omega"] is None]
            reused = [r for r in records if r["reused_from_v_over_omega"] == 0.5]
            assert len(source) == len(reused) == 2  # two heads, one fold
            for first, again in zip(source, reused, strict=True):
                assert first["v_over_omega"] == 0.5 and again["v_over_omega"] == 1.0
                assert first["feature_cache_hit"] is again["feature_cache_hit"] is hit
                assert again["feature_environment"] == first["feature_environment"]
                assert again["tuning"]["feature_cache_hits"] == [hit] * settings.inner_folds
        # entries for the 0.5 register only: the fold and its tuning folds
        assert len(list(cache.iterdir())) == 1 + settings.inner_folds


class TestGeometry:
    """#733: ``--geometry`` runs the chain, a ring and a 2D grid at the same ``V/Ω``."""

    #: Keys of the chain's ``--quick`` maps, computed on ``main`` before #733 (0840dae, which
    #: already records ``readout_error`` from #734 and the position jitter from #735) with
    #: ``cache_key("0" * 64, config, KEY_ENV, "c" * 64)``: the config is unchanged, so the
    #: entries a chain run wrote before the option existed are still hit.
    CHAIN_KEYS: ClassVar[dict[str, str]] = {
        "A": "ac15c7f711f29923fd098f6a212528cc68b9fb83205650a2a33fb70297fdb7e0",
        "B": "f1d5115943e52fec3e823fd7cc07e32453dff88a5dc2975487913c1bb68e800e",
    }
    KEY_ENV: ClassVar[dict[str, Any]] = {
        "python": "3.12.0",
        "implementation": "CPython",
        "hqnn_forge": "0.0.0",
        "packages": {"torch": "2.0"},
    }

    @staticmethod
    def _neighbours(geometry: str, n_atoms: int, cols: int = 0) -> set[tuple[int, int]]:
        """The nearest-neighbour pairs ``i < j`` of each geometry, written out by index."""
        if geometry == "chain":
            return {(i, i + 1) for i in range(n_atoms - 1)}
        if geometry == "ring":
            return {tuple(sorted((i, (i + 1) % n_atoms))) for i in range(n_atoms)}  # type: ignore[misc]
        across = {(k, k + 1) for k in range(n_atoms - 1) if (k + 1) % cols}
        down = {(k, k + cols) for k in range(n_atoms - cols)}
        return across | down

    @pytest.mark.parametrize(
        ("n_atoms", "cols", "shape"),
        [
            (4, None, (2, 2)),
            (6, None, (2, 3)),
            (8, None, (2, 4)),
            (9, None, (3, 3)),
            (10, None, (2, 5)),
            (12, None, (3, 4)),
            (6, 2, (3, 2)),
            (12, 6, (2, 6)),
        ],
    )
    def test_the_grid_shape_rule(self, n_atoms: int, cols: int | None, shape: tuple) -> None:
        """Default: the smallest ``cols`` dividing ``N`` with ``cols² >= N``; else as given."""
        assert bench.grid_shape(n_atoms, cols) == shape

    @pytest.mark.parametrize(
        ("n_atoms", "cols", "match"),
        [
            (6, 4, "grid_cols=4 does not divide n_atoms=6"),
            (5, None, "a grid of 1 × 5 atoms is a chain"),
            (6, 6, "a grid of 1 × 6 atoms is a chain"),
            (6, 1, "a grid of 6 × 1 atoms is a chain"),
            (6, 0, "grid_cols must be an integer >= 1"),
        ],
    )
    def test_a_shape_that_is_no_grid_is_refused(
        self, n_atoms: int, cols: int | None, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            bench.grid_shape(n_atoms, cols)

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"geometries": ()}, "geometries must be distinct values"),
            ({"geometries": ("chain", "chain")}, "geometries must be distinct values"),
            ({"geometries": ("square",)}, "geometries must be distinct values"),
            ({"grid_cols": 2}, "grid_cols is the shape of the grid geometry"),
            ({"geometries": ("grid",), "grid_cols": 3}, "grid_cols=3 does not divide"),
            ({"geometries": ("grid",), "n_atoms": 5}, "a grid of 1 × 5 atoms is a chain"),
            ({"geometries": ("ring",), "n_atoms": 2}, "the ring geometry needs n_atoms >= 3"),
        ],
    )
    def test_invalid_geometry_settings(self, changes: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _replace(QUICK, **changes)

    def test_a_shape_binds_only_where_a_rydberg_map_is_built(self) -> None:
        settings = _replace(QUICK, arms=("C",), n_atoms=5, geometries=("grid", "ring"))
        assert settings.geometries == ("grid", "ring")

    @pytest.mark.parametrize("geometry", ["chain", "ring", "grid"])
    @pytest.mark.parametrize("v", [0.1, 1.0, 30.0])
    @pytest.mark.parametrize("n_atoms", [4, 6])
    def test_the_nearest_neighbour_coupling_is_the_ratio_asked_for(
        self, geometry: str, v: float, n_atoms: int
    ) -> None:
        """``V_ij/Ω = V/Ω`` on every nearest-neighbour pair, and no pair couples more."""
        settings = _replace(QUICK, n_atoms=n_atoms, geometries=(geometry,))
        register = bench.atom_register(settings, geometry, v)
        ratio = register.interaction_matrix(DEFAULT_C6).numpy() / OMEGA
        cols = 0 if settings.grid_shape is None else settings.grid_shape[1]
        pairs = self._neighbours(geometry, n_atoms, cols)
        assert (
            len(pairs)
            == {"chain": n_atoms - 1, "ring": n_atoms, "grid": 4 if n_atoms == 4 else 7}[geometry]
        )
        for i, j in pairs:
            assert ratio[i, j] == pytest.approx(v, rel=1e-12)
        assert ratio.max() == pytest.approx(v, rel=1e-12)
        others = [ratio[i, j] for i in range(n_atoms) for j in range(i + 1, n_atoms)]
        assert sum(r == pytest.approx(v, rel=1e-12) for r in others) == len(pairs)

    @pytest.mark.parametrize(
        ("geometry", "n_atoms", "pair", "factor"),
        [
            ("chain", 4, (0, 2), 64.0),  # 2a
            ("ring", 4, (0, 2), 8.0),  # the square's diagonal, 2a cos(π/4) = √2 a
            ("ring", 6, (0, 2), 27.0),  # 2a cos(π/6) = √3 a
            ("grid", 4, (0, 3), 8.0),  # √2 a
            ("grid", 6, (0, 4), 8.0),  # 2 × 3: atom 4 is (1, 1)
        ],
    )
    def test_the_next_nearest_coupling_of_each_geometry(
        self, geometry: str, n_atoms: int, pair: tuple[int, int], factor: float
    ) -> None:
        settings = _replace(QUICK, n_atoms=n_atoms, geometries=(geometry,))
        register = bench.atom_register(settings, geometry, 10.0)
        ratio = register.interaction_matrix(DEFAULT_C6).numpy() / OMEGA
        assert ratio[pair] == pytest.approx(10.0 / factor, rel=1e-12)

    def test_the_next_nearest_table_of_the_model_page(self) -> None:
        """Each entry of the page's table, to its two or three digits, from the register."""
        text = (ROOT / "docs" / "rydberg-model.md").read_text(encoding="utf-8")
        table = {
            cells[0]: [float(x) for x in cells[1:]]
            for line in text.splitlines()
            if line.lstrip().startswith("|")
            and (cells := [c.strip() for c in line.strip().strip("|").split("|")])[0].startswith(
                ("chain, ", "ring of ")
            )
        }
        pairs = {
            "chain, `2a`": [("chain", 4, (0, 2))],
            "ring of 6, `√3 a`": [("ring", 6, (0, 2))],
            "ring of 4 and grid, `√2 a`": [("ring", 4, (0, 2)), ("grid", 4, (0, 3))],
        }
        assert set(table) == set(pairs)
        values = bench.Settings().v_over_omega
        for row, cases in pairs.items():
            assert len(table[row]) == len(values)
            for geometry, n_atoms, pair in cases:
                settings = _replace(QUICK, n_atoms=n_atoms, geometries=(geometry,))
                for v, shown in zip(values, table[row], strict=True):
                    register = bench.atom_register(settings, geometry, v)
                    exact = float(register.interaction_matrix(DEFAULT_C6)[pair]) / OMEGA
                    assert shown == pytest.approx(exact, rel=0.035)  # rounded to 2 digits

    def test_the_registers_are_the_constructors_at_the_spacing_of_the_ratio(self) -> None:
        settings = _replace(QUICK, n_atoms=6, geometries=("chain", "ring", "grid"), grid_cols=2)
        a = bench.spacing(3.0, OMEGA, DEFAULT_C6)
        expected = {
            "chain": AtomRegister.chain(6, a),
            "ring": AtomRegister.ring(6, a),
            "grid": AtomRegister.grid(3, 2, a),
        }
        for geometry, register in expected.items():
            built = bench.atom_register(settings, geometry, 3.0)
            assert built.positions.tolist() == register.positions.tolist()

    def test_the_chain_reproduces_the_cache_key_from_before_the_option(self) -> None:
        settings = _replace(QUICK, geometries=("chain", "ring", "grid"))
        reps = [r for r in bench.representations(settings) if r.geometry == "chain"]
        for rep in reps:
            assert "geometry" not in rep.config
            key = bench.cache_key("0" * 64, rep.config, self.KEY_ENV, "c" * 64)
            assert key == self.CHAIN_KEYS[rep.arm]
        assert [r.arm for r in reps] == ["A", "B"]

    def test_a_chain_a_ring_and_a_grid_get_different_keys_at_one_grid_point(self) -> None:
        settings = _replace(QUICK, geometries=("chain", "ring", "grid"))
        reps = bench.representations(settings)
        assert [(r.arm, r.geometry) for r in reps[:6]] == [
            ("A", "chain"),
            ("B", "chain"),
            ("A", "ring"),
            ("B", "ring"),
            ("A", "grid"),
            ("B", "grid"),
        ]
        assert {(r.v_over_omega, r.gamma_over_omega) for r in reps[:6]} == {(1.0, 0.1)}
        keys = {bench.cache_key("0" * 64, r.config, self.KEY_ENV, "c" * 64) for r in reps[:6]}
        assert len(keys) == 6
        assert reps[2].config["geometry"] == "ring" and "grid_shape" not in reps[2].config
        assert reps[4].config["geometry"] == "grid" and reps[4].config["grid_shape"] == [2, 2]
        assert reps[4].grid_shape == (2, 2)

    @pytest.mark.parametrize("gamma", [0.0, 0.1])
    def test_arm_b_features_are_identical_across_geometries(self, gamma: float) -> None:
        """Without interactions feature ``i`` depends on input ``i`` alone, wherever atom i is."""
        settings = _replace(QUICK, geometries=("chain", "ring", "grid"))
        X = torch.tensor(
            [[-2.0, -0.5, 0.3, 2.9], [0.1, 1.2, -3.0, 0.0], [1.5, 1.5, -1.5, -1.5]],
            dtype=torch.float64,
        )
        features = {
            g: bench.feature_map(settings, 1.0, gamma, interactions=False, geometry=g).transform(X)
            for g in ("chain", "ring", "grid")
        }
        torch.testing.assert_close(features["ring"], features["chain"], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(features["grid"], features["chain"], rtol=1e-12, atol=1e-12)
        if gamma == 0.0:  # the closed form of the module docstring
            expected = torch.from_numpy(_single_atom_feature(X.numpy()))
            torch.testing.assert_close(features["chain"], expected, rtol=1e-10, atol=1e-12)
        # Arm A is not: the ring couples atoms 0 and 3, which the chain barely does.
        interacting = {
            g: bench.feature_map(settings, 10.0, gamma, interactions=True, geometry=g).transform(X)
            for g in ("chain", "ring")
        }
        assert (interacting["ring"] - interacting["chain"]).abs().max() > 1e-3

    def test_arm_b_is_computed_once_for_every_geometry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _replace(
            QUICK,
            arms=("A", "B"),
            n_trials=0,
            n_splits=2,
            max_folds=1,
            n_steps=10,
            v_over_omega=(1.0, 10.0),
            gamma_over_omega=(0.1,),
            geometries=("chain", "ring", "grid"),
        )
        rng = np.random.RandomState(3)
        # Random data, not a control drawn for these settings: no data_seed, so no label bonds.
        data = {"iranian-churn": (rng.randn(40, 4), np.array([0, 1] * 20), None)}
        calls: list[tuple[str, str]] = []
        original = bench.features_of

        def spy(rep: Any, split: Any, cache: Any) -> Any:
            calls.append((rep.arm, rep.geometry))
            return original(rep, split, cache)

        monkeypatch.setattr(bench, "features_of", spy)
        records = bench.run(data, settings, tmp_path / "out.jsonl", cache_dir=None)
        # A at 3 geometries × 2 V/Ω; B once, at the first geometry.
        assert sorted(calls) == sorted(
            [("A", g) for g in ("chain", "ring", "grid") for _ in range(2)] + [("B", "chain")]
        )
        b = [r for r in records if r["arm"] == "B" and r["head"] == "linear"]
        assert [(r["geometry"], r["v_over_omega"]) for r in b] == [
            (g, v) for g in ("chain", "ring", "grid") for v in (1.0, 10.0)
        ]
        assert [(r["reused_from_geometry"], r["reused_from_v_over_omega"]) for r in b] == [
            (None, None),
            *[("chain", 1.0)] * 5,
        ]
        for r in b[1:]:
            for k in ("mcc", "pr_auc", "threshold", "separation", "hyperparameters"):
                assert r[k] == b[0][k]
            assert r["feature_seconds"] == r["train_seconds"] == 0.0
        grid_lines = [r for r in records if r["geometry"] == "grid"]
        assert {tuple(r["grid_shape"]) for r in grid_lines} == {(2, 2)}
        assert all(r["grid_shape"] is None for r in records if r["geometry"] != "grid")
        assert all(
            r["feature_map"]["config"]["register"]
            == bench.atom_register(settings, r["geometry"], r["v_over_omega"]).positions.tolist()
            for r in records
        )

    @pytest.mark.slow
    def test_quick_settings_with_every_geometry_in_one_run(self, tmp_path: Path) -> None:
        """
        One run of all three at ``QUICK``, on a dataset whose label does not depend on the
        geometry (the positive control's does, so it is run once per geometry, #741): the
        lines carry the geometry, the keys differ.
        """
        out, cache = tmp_path / "quick.jsonl", tmp_path / "cache"
        X, y, _ = bench.load_dataset("positive-control", QUICK)
        settings = _replace(QUICK, geometries=("chain", "ring", "grid"))
        records = bench.run({"iranian-churn": (X, y, None)}, settings, out, cache_dir=cache)
        lines = bench.read_records(out)
        assert len(lines) == len(records) == 3 * 4 + 5  # A and B × 2 heads per geometry; C, D
        rydberg = [r for r in lines if r["arm"] in ("A", "B")]
        assert [r["geometry"] for r in rydberg] == [
            g for g in ("chain", "ring", "grid") for _ in range(4)
        ]
        assert [r["grid_shape"] for r in rydberg] == [None] * 8 + [[2, 2]] * 4
        assert all(r["geometry"] is None for r in lines if r["arm"] not in ("A", "B"))
        assert all(r["settings"]["geometries"] == ["chain", "ring", "grid"] for r in lines)
        # Three A maps, one B map, arm D: each with its fold and its tuning folds.
        assert len(list(cache.iterdir())) == 5 * (1 + QUICK.inner_folds)
        a_configs = {
            json.dumps(r["feature_map"], sort_keys=True) for r in rydberg if r["arm"] == "A"
        }
        assert len(a_configs) == 3
        chain_a = next(r for r in rydberg if r["arm"] == "A")
        assert "geometry" not in chain_a["feature_map"]
        key = bench.cache_key("0" * 64, chain_a["feature_map"], self.KEY_ENV, "c" * 64)
        assert key == self.CHAIN_KEYS["A"]
        table = bench.summarise(lines)
        assert {line.split()[1] for line in table.splitlines()[1:]} == {
            "chain",
            "ring",
            "grid-2x2",
            "-",
        }

    def test_the_command_line_passes_the_geometry_also_under_quick(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[Any] = []

        def capture(datasets: Any, settings: Any, out: Any, **kwargs: Any) -> list[Any]:
            captured.append(settings)
            return []

        monkeypatch.setattr(bench, "run", capture)
        out = str(tmp_path / "out.jsonl")
        bench.main(["--quick", "--geometry", "grid", "grid", "--out", out, "--no-cache"])
        bench.main(["--geometry", "grid", "--grid-cols", "2", "--out", out, "--no-cache"])
        assert captured[0] == _replace(QUICK, geometries=("grid",))
        assert (captured[1].geometries, captured[1].grid_cols) == (("grid",), 2)
        # Several geometries, on a dataset whose label does not depend on them.
        monkeypatch.setattr(bench, "load_dataset", lambda *args, **kwargs: (None, None, None))
        bench.main(
            ["--dataset", "iranian-churn", "--geometry", "grid", "ring", "grid"]
            + ["--out", out, "--no-cache"]
        )
        assert captured[2].geometries == ("grid", "ring")
        assert bench.Settings().geometries == ("chain",)
        with pytest.raises(SystemExit):
            bench._parser().parse_args(["--geometry", "square"])


class TestControlFollowsTheGeometry:
    """
    #738: a run labels the positive control over its geometry's bonds and records them.

    The labels are checked against the sign of the bond sums written out by hand on
    latents drawn independently of the runner; the chain's against the fingerprint of
    ``main`` at 6080cde.
    """

    def test_the_bonds_of_a_run_typed_in(self) -> None:
        def bonds(**changes: Any) -> list[tuple[int, int]]:
            return bench.control_bonds(_replace(QUICK, **changes))

        assert bonds() == [(0, 1), (1, 2), (2, 3)]
        assert bonds(geometries=("ring",)) == [(0, 1), (0, 3), (1, 2), (2, 3)]
        assert bonds(geometries=("grid",)) == [(0, 1), (0, 2), (1, 3), (2, 3)]
        # 3 × 2: k = 2r + c, rows (0, 1), (2, 3), (4, 5) and columns (k, k + 2)
        assert bonds(geometries=("grid",), n_atoms=6, grid_cols=2) == [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 5),
            (4, 5),
        ]

    @pytest.mark.parametrize(
        "geometries", [("chain", "ring"), ("ring", "grid"), ("grid", "chain")]
    )
    def test_a_control_of_several_geometries_is_labelled_over_the_first(
        self, geometries: tuple[str, ...]
    ) -> None:
        """#741: ``control_bonds`` and ``load_dataset`` are those of the first geometry alone."""
        settings = _replace(QUICK, geometries=geometries)
        alone = _replace(QUICK, geometries=geometries[:1])
        assert bench.control_bonds(settings) == bench.control_bonds(alone)
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        X1, y1, seed1 = bench.load_dataset("positive-control", alone)
        np.testing.assert_array_equal(X, X1)
        np.testing.assert_array_equal(y, y1)
        assert data_seed == seed1

    @pytest.mark.parametrize(
        ("geometry", "bond_sum"),
        [
            ("chain", lambda z: z[:, 0] * z[:, 1] + z[:, 1] * z[:, 2] + z[:, 2] * z[:, 3]),
            (
                "ring",
                lambda z: (
                    z[:, 0] * z[:, 1] + z[:, 1] * z[:, 2] + z[:, 2] * z[:, 3] + z[:, 3] * z[:, 0]
                ),
            ),
            # 2 × 2: rows (0, 1), (2, 3), columns (0, 2), (1, 3)
            (
                "grid",
                lambda z: (
                    z[:, 0] * z[:, 1] + z[:, 2] * z[:, 3] + z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]
                ),
            ),
        ],
    )
    def test_the_loaded_control_is_labelled_over_the_geometry(
        self, geometry: str, bond_sum: Any
    ) -> None:
        settings = _replace(QUICK, geometries=(geometry,))
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        assert data_seed == bench.run_seeds(QUICK.random_state)["data_seed"]
        # The latents as positive_control documents them: the first draw of its generator.
        latent = np.random.default_rng(data_seed).standard_normal((QUICK.n_samples, 4))
        np.testing.assert_array_equal(y, (bond_sum(latent) > 0).astype(np.int64))
        chain_X, chain_y, _ = bench.load_dataset("positive-control", QUICK)
        np.testing.assert_array_equal(X, chain_X)  # only the label follows the geometry
        sha = fingerprint(X, y)
        if geometry == "chain":
            # positive_control(4, 160, ·) on main at 6080cde, before the geometry
            assert sha == "9893f5fabce42b3762849c354a1a6bd5d98efebd7d2037f8239c62b14874b4b8"
        else:
            assert (y != chain_y).any() and sha != fingerprint(chain_X, chain_y)

    def test_every_line_records_the_bonds_of_its_label(self, tmp_path: Path) -> None:
        settings = _replace(QUICK, arms=("C",), n_trials=0, max_epochs=1, geometries=("ring",))
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        rng = np.random.RandomState(0)
        datasets = {
            "positive-control": (X, y, data_seed),
            "iranian-churn": (rng.randn(80, 5), np.array([0, 1] * 40), None),
        }
        out = tmp_path / "out.jsonl"
        records = bench.run(datasets, settings, out, cache_dir=None)
        lines = bench.read_records(out)
        assert len(lines) == len(records) == 4  # arm C, 2 heads, one fold, 2 datasets
        control = [r for r in lines if r["dataset"] == "positive-control"]
        assert [r["label_bonds"] for r in control] == [[[0, 1], [0, 3], [1, 2], [2, 3]]] * 2
        assert {r["data_sha256"] for r in control} == {fingerprint(X, y)}
        assert {r["seeds"]["data_seed"] for r in control} == {data_seed}
        assert all(r["label_bonds"] is None for r in lines if r["dataset"] == "iranian-churn")

    @pytest.mark.parametrize(
        ("geometries", "grid_cols"), [(("chain", "ring"), None), (("ring", "grid"), 2)]
    )
    def test_the_labels_of_a_control_of_several_geometries(
        self, geometries: tuple[str, ...], grid_cols: int | None
    ) -> None:
        """
        #741: one label per geometry, in order, each over its geometry's bonds and run
        with the settings of that geometry alone, on the same inputs.  The labels are
        checked against the sign of the bond sums written out by hand on latents drawn
        independently of the runner, as in
        ``test_the_loaded_control_is_labelled_over_the_geometry``.
        """
        settings = _replace(QUICK, geometries=geometries, grid_cols=grid_cols)
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        labels = bench.labels_of("positive-control", X, y, data_seed, settings)
        z = np.random.default_rng(data_seed).standard_normal((QUICK.n_samples, 4))
        by_hand = {
            "chain": z[:, 0] * z[:, 1] + z[:, 1] * z[:, 2] + z[:, 2] * z[:, 3],
            "ring": z[:, 0] * z[:, 1] + z[:, 1] * z[:, 2] + z[:, 2] * z[:, 3] + z[:, 3] * z[:, 0],
            # 2 × 2: rows (0, 1), (2, 3), columns (0, 2), (1, 3)
            "grid": z[:, 0] * z[:, 1] + z[:, 2] * z[:, 3] + z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3],
        }
        bonds = {
            "chain": [(0, 1), (1, 2), (2, 3)],
            "ring": [(0, 1), (0, 3), (1, 2), (2, 3)],
            "grid": [(0, 1), (0, 2), (1, 3), (2, 3)],
        }
        assert [s.geometries for _, s, _ in labels] == [(g,) for g in geometries]
        for geometry, (labels_y, alone, label_bonds) in zip(geometries, labels, strict=True):
            np.testing.assert_array_equal(labels_y, (by_hand[geometry] > 0).astype(np.int64))
            assert label_bonds == bonds[geometry]
            assert alone == _replace(
                settings,
                geometries=(geometry,),
                grid_cols=grid_cols if geometry == "grid" else None,
            )
        np.testing.assert_array_equal(labels[0][0], y)
        assert not np.array_equal(labels[0][0], labels[1][0])
        # One geometry, or a dataset without a data_seed: the entry as it is, once.
        alone = _replace(QUICK, geometries=geometries[:1])
        X1, y1, seed1 = bench.load_dataset("positive-control", alone)
        ((only_y, only_settings, only_bonds),) = bench.labels_of(
            "positive-control", X1, y1, seed1, alone
        )
        assert only_y is y1 and only_settings is alone and only_bonds == bonds[geometries[0]]
        ((uci_y, uci_settings, uci_bonds),) = bench.labels_of(
            "iranian-churn", X, y, None, settings
        )
        assert uci_y is y and uci_settings is settings and uci_bonds is None

    @pytest.mark.parametrize("entry", ["other inputs", "the second label"])
    def test_a_run_of_several_geometries_refuses_a_control_it_cannot_redraw(
        self, entry: str, tmp_path: Path
    ) -> None:
        """
        The second label is drawn from ``data_seed``, so ``(X, y)`` must be the control
        drawn at it, labelled over the first geometry; a file is not opened.
        """
        settings = _replace(QUICK, arms=("C",), geometries=("chain", "ring"))
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        if entry == "other inputs":
            X = X + 1.0
        else:
            _, y, _ = bench.load_dataset("positive-control", _replace(QUICK, geometries=("ring",)))
        out = tmp_path / "out.jsonl"
        with pytest.raises(ValueError, match="must be load_dataset's control"):
            bench.run({"positive-control": (X, y, data_seed)}, settings, out)
        assert not out.exists()

    @pytest.mark.parametrize(
        ("geometries", "grid_cols"), [(("chain", "ring"), None), (("ring", "grid"), 2)]
    )
    def test_a_run_of_two_geometries_writes_the_lines_of_two_runs_of_one(
        self, geometries: tuple[str, ...], grid_cols: int | None, tmp_path: Path
    ) -> None:
        """
        #741: the control of a two-geometry run against two runs of one geometry each,
        and a UCI-like dataset beside it whose arm B is still computed once.  Every line
        of a label equals that of its single run but for ``settings``, which is the run's,
        and the timings; B and C are trained per label, on the label's own folds.
        """
        settings = _replace(
            QUICK, arms=FAST, n_trials=0, geometries=geometries, grid_cols=grid_cols
        )
        multi = bench.run(
            {"positive-control": bench.load_dataset("positive-control", settings)},
            settings,
            tmp_path / "multi.jsonl",
        )
        multi_lines = bench.read_records(tmp_path / "multi.jsonl")
        timings = {"feature_seconds", "train_seconds", "tuning_seconds"}
        singles: list[dict[str, Any]] = []
        for geometry in geometries:
            alone = _replace(
                settings,
                geometries=(geometry,),
                grid_cols=grid_cols if geometry == "grid" else None,
            )
            out = tmp_path / f"{geometry}.jsonl"
            bench.run(
                {"positive-control": bench.load_dataset("positive-control", alone)}, alone, out
            )
            lines = bench.read_records(out)
            assert all(r["settings"] == json.loads(json.dumps(alone.as_dict())) for r in lines)
            singles.extend(lines)
        # A, B (linear, mlp) and C (linear, mlp) per label, one fold.
        assert len(multi) == len(multi_lines) == len(singles) == 2 * 6
        for got, want in zip(multi_lines, singles, strict=True):
            assert list(got) == list(want)
            assert got["settings"] == json.loads(json.dumps(settings.as_dict()))
            assert {k for k in got if got[k] != want[k]} <= {"settings"} | timings
        first, second = multi_lines[:6], multi_lines[6:]
        assert all(r["label_bonds"] != first[0]["label_bonds"] for r in second)
        assert first[0]["test_idx"] != second[0]["test_idx"]  # the folds follow the label
        b_lines = [r for r in multi_lines if r["arm"] == "B"]
        assert [r["reused_from_geometry"] for r in b_lines] == [None] * 4
        assert [r["geometry"] for r in b_lines] == [g for g in geometries for _ in range(2)]
        # Beside it, a dataset without a data_seed: one label, B computed once.
        rng = np.random.RandomState(0)
        mixed = bench.run(
            {
                "positive-control": bench.load_dataset("positive-control", settings),
                "iranian-churn": (rng.randn(80, 5), np.array([0, 1] * 40), None),
            },
            settings,
            tmp_path / "mixed.jsonl",
        )
        churn = [r for r in mixed if r["dataset"] == "iranian-churn"]
        churn_b = [r["reused_from_geometry"] for r in churn if r["arm"] == "B"]
        assert churn_b == [None, None, geometries[0], geometries[0]]
        assert all(r["label_bonds"] is None for r in churn)
        # The printed summary keeps the labels apart: one row per label and arm C head.
        table = bench.summarise(multi_lines)
        names = ["grid-2x2" if g == "grid" else g for g in geometries]
        c_rows = [line.split() for line in table.splitlines()[1:] if line.split()[4] == "C"]
        assert [(r[1], r[-1]) for r in c_rows] == [(f"({g})", "1") for g in names for _ in "ab"]
        alone_table = bench.summarise(singles[:6])
        assert [line.split()[1] for line in alone_table.splitlines()[1:]] == [names[0]] * 4 + [
            "-"
        ] * 2

    def test_the_command_line_loads_the_control_once_for_several_geometries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[Any] = []

        def capture(datasets: Any, settings: Any, out: Any, **kwargs: Any) -> list[Any]:
            captured.append((datasets, settings))
            return []

        monkeypatch.setattr(bench, "run", capture)
        argv = ["--quick", "--geometry", "chain", "ring", "--no-cache"]
        bench.main([*argv, "--out", str(tmp_path / "out.jsonl")])
        ((datasets, settings),) = captured
        assert settings.geometries == ("chain", "ring") and list(datasets) == ["positive-control"]
        X, y, data_seed = datasets["positive-control"]
        X1, y1, seed1 = bench.load_dataset("positive-control", QUICK)
        np.testing.assert_array_equal(X, X1)
        np.testing.assert_array_equal(y, y1)
        assert data_seed == seed1


class TestSummary:
    @staticmethod
    def _line(v: float | None, arm: str, feature: float, train: float, **extra: Any) -> dict:
        return {
            "dataset": "positive-control",
            "v_over_omega": v,
            "gamma_over_omega": None if v is None else 0.1,
            "arm": arm,
            "head": "linear",
            "mcc": 0.5,
            "pr_auc": 0.6,
            "feature_dim": 2,
            "n_parameters": 3,
            "feature_seconds": feature,
            "train_seconds": train,
            **extra,
        }

    def _row(self, table: str, v: str, arm: str) -> list[str]:
        (row,) = [
            line.split()
            for line in table.splitlines()[1:]
            if line.split()[2] == v and line.split()[4] == arm
        ]
        return row

    def test_reused_lines_show_no_time_and_are_left_out_of_the_mean(self) -> None:
        """
        ``feat s`` and ``fit s`` are means over the lines that were computed:
        a group of reused arm B lines only prints ``-`` (#541), and the zero
        timings of reused lines do not lower the mean of a group that also
        has computed ones.  Lines without the field (an older file) count as
        computed.
        """
        records = [
            self._line(0.5, "B", 2.0, 0.5, reused_from_v_over_omega=None),
            self._line(0.5, "B", 4.0, 1.5, reused_from_v_over_omega=None),
            self._line(1.0, "B", 0.0, 0.0, reused_from_v_over_omega=0.5),
            self._line(1.0, "B", 0.0, 0.0, reused_from_v_over_omega=0.5),
            self._line(1.0, "A", 3.0, 0.25, reused_from_v_over_omega=None),
            # a group with one computed and one reused line
            self._line(2.0, "B", 6.0, 1.0, reused_from_v_over_omega=None),
            self._line(2.0, "B", 0.0, 0.0, reused_from_v_over_omega=0.5),
            self._line(None, "C", 0.0, 0.75),
        ]
        table = bench.summarise(records)
        # columns: ..., dim, params, feat s, fit s, n
        assert self._row(table, "0.5", "B")[-3:] == ["3.00", "1.00", "2"]
        assert self._row(table, "1", "B")[-3:] == ["-", "-", "2"]
        assert self._row(table, "1", "A")[-3:] == ["3.00", "0.25", "1"]
        assert self._row(table, "2", "B")[-3:] == ["6.00", "1.00", "2"]
        assert self._row(table, "-", "C")[-3:] == ["0.00", "0.75", "1"]
        widths = {len(line) for line in table.splitlines()[1:]}
        assert len(widths) == 1  # a "-" keeps the columns aligned

    def test_the_total_adds_the_seconds_of_the_tuning_folds(self) -> None:
        """
        #785: once a line records ``tuning["feature_seconds"]``, ``tot feat s``
        follows ``feat s``: the mean over the computed lines of
        ``feature_seconds + Σ_j tuning["feature_seconds"][j]``.  A line without
        tuning adds nothing to its ``feature_seconds``, a reused arm B line is
        left out although it carries its source's tuning seconds, and a row
        with a line written before #785 (``tuning`` without the field) shows
        ``-``.  Without a line that records it, there is no such column.
        """

        def timed(*seconds: float) -> dict[str, Any]:
            return {"feature_seconds": list(seconds)}

        before = {"scores": [0.5]}  # a tuning record from before #785
        records = [
            # 2 + 1 + 0.5 = 3.5 and 4 + 0.25 + 0.25 = 4.5: mean 4
            self._line(0.5, "A", 2.0, 1.0, tuning=timed(1.0, 0.5)),
            self._line(0.5, "A", 4.0, 1.0, tuning=timed(0.25, 0.25)),
            # computed: 1 + 2 + 3 = 6; the reused line carries 2 and 3 and is left out
            self._line(0.5, "B", 1.0, 0.5, reused_from_v_over_omega=None, tuning=timed(2.0, 3.0)),
            self._line(1.0, "B", 0.0, 0.0, reused_from_v_over_omega=0.5, tuning=timed(2.0, 3.0)),
            self._line(2.0, "B", 0.0, 0.0, reused_from_v_over_omega=0.5, tuning=timed(2.0, 3.0)),
            self._line(2.0, "B", 5.0, 0.5, reused_from_v_over_omega=None, tuning=timed(0.5, 0.5)),
            # no trials: nothing to add
            self._line(None, "C", 0.0, 0.75, tuning=None),
            self._line(None, "D", 1.5, 0.75, tuning=None),
            # one line from before #785 in the row
            self._line(2.0, "A", 1.0, 1.0, tuning=timed(1.0)),
            self._line(2.0, "A", 3.0, 1.0, tuning=before),
        ]
        table = bench.summarise(records)
        header = table.splitlines()[0]
        assert header.endswith(" feat s tot feat s  fit s  n")
        # columns: ..., feat s, tot feat s, fit s, n
        assert self._row(table, "0.5", "A")[-4:] == ["3.00", "4.00", "1.00", "2"]
        assert self._row(table, "0.5", "B")[-4:] == ["1.00", "6.00", "0.50", "1"]
        assert self._row(table, "1", "B")[-4:] == ["-", "-", "-", "1"]
        assert self._row(table, "2", "B")[-4:] == ["5.00", "6.00", "0.50", "2"]
        assert self._row(table, "-", "C")[-4:] == ["0.00", "0.00", "0.75", "1"]
        assert self._row(table, "-", "D")[-4:] == ["1.50", "1.50", "0.75", "1"]
        assert self._row(table, "2", "A")[-4:] == ["2.00", "-", "1.00", "2"]
        assert len({len(line) for line in table.splitlines()}) == 1
        assert [bench.total_feature_seconds(r) for r in records] == [
            3.5,
            4.5,
            6.0,
            5.0,
            5.0,
            6.0,
            0.0,
            1.5,
            2.0,
            None,
        ]
        # Lines that do not record it: the table of before, without the column.
        old = bench.summarise([r for r in records if not bench._times_tuning(r)])
        assert old.splitlines()[0].endswith(" t-deg  feat s  fit s  n")
        assert self._row(old, "2", "A")[-3:] == ["3.00", "1.00", "1"]

    def test_degenerate_folds_are_counted_per_row(self) -> None:
        """
        ``deg`` counts the lines (folds) of a row whose own PCA fit was
        degenerate, ``t-deg`` those with a degenerate tuning fold (#525);
        lines without the field (an older file) are not counted, and a row
        none of whose lines records it shows ``-``.
        """

        def pca(degenerate: bool) -> dict[str, Any]:
            return {"degenerate": degenerate}

        def tuning(*flags: bool) -> dict[str, Any]:
            return {"pca": [pca(flag) for flag in flags]}

        records = [
            # three folds of A: one degenerate, one with a degenerate tuning fold
            self._line(1.0, "A", 1.0, 1.0, pca=pca(True), tuning=tuning(False, False)),
            self._line(1.0, "A", 1.0, 1.0, pca=pca(False), tuning=tuning(False, True)),
            self._line(1.0, "A", 1.0, 1.0, pca=pca(False), tuning=tuning(False, False)),
            # two folds of B, both degenerate, without tuning
            self._line(1.0, "B", 1.0, 1.0, pca=pca(True), tuning=None),
            self._line(1.0, "B", 1.0, 1.0, pca=pca(True), tuning=None),
            # an older file: no field
            self._line(None, "C", 0.0, 1.0),
            # a line without the field next to one with it
            self._line(2.0, "A", 1.0, 1.0),
            self._line(2.0, "A", 1.0, 1.0, pca=pca(True), tuning=tuning(True, True)),
        ]
        table = bench.summarise(records)
        header = table.splitlines()[0].split()
        assert header[header.index("params") + 1 : header.index("params") + 3] == ["deg", "t-deg"]
        # columns: ..., dim, params, deg, t-deg, feat s, fit s, n
        assert self._row(table, "1", "A")[-5:-3] == ["1", "1"]
        assert self._row(table, "1", "B")[-5:-3] == ["2", "-"]
        assert self._row(table, "-", "C")[-5:-3] == ["-", "-"]
        assert self._row(table, "2", "A")[-5:-3] == ["1", "1"]
        assert len({len(line) for line in table.splitlines()}) == 1


#: ``(measurement, shots, shot_stream)`` of a line of arm A, B or D with ``--shots 16`` since
#: #762 (a stream per matrix), of one written before it (no ``shot_stream``), and of exact
#: features, which arms C and ``D-hybrid`` always have.
PER_MATRIX = ("shots", 16, "per-matrix")
SHARED = ("shots", 16, None)
EXACT = ("exact", None, None)
#: The tail of every refusal of a row measured in more than one way (#763).
MEASUREMENT_ADVICE = (
    "Its mean and spread would pool lines whose features were measured differently; "
    "summarise each run's lines on their own."
)


@pytest.fixture(scope="module")
def shots_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """
    The lines, as read back, of a quick positive-control run with ``--shots 16`` and every
    arm (A, B, C, D and with D ``D-hybrid``): two folds, no tuning.
    """
    settings = _replace(QUICK, shots=16, n_trials=0, max_folds=2)
    out = tmp_path_factory.mktemp("shots") / "shots.jsonl"
    data = {"positive-control": bench.load_dataset("positive-control", settings)}
    bench.run(data, settings, out, cache_dir=None)
    lines: list[dict[str, Any]] = bench.read_records(out)
    return lines


#: The committed result files and the SHA-256 of their :func:`summarise` on ``main`` at
#: 9c60224, before #763.
COMMITTED_SUMMARIES = (
    (
        "rydberg_features_benchmark.jsonl.xz",
        "89c150dc9ece159497717cba5927f061a9346c0b943c10a6132ee95ff6cad5b7",
    ),
    (
        "rydberg_features_geometries_control_chain.jsonl.xz",
        "ffc8c7fa9abddcd2c20ae6a5ef40cf48eff7f98b4d425af81330a24f669e402a",
    ),
    (
        "rydberg_features_geometries_control_ring.jsonl.xz",
        "4d084ad9216f7242f33c5122a65e3b5156725b751f23b4a85d70ac15d84ce06b",
    ),
    (
        "rydberg_features_geometries_control_grid.jsonl.xz",
        "019193cce2734f21c8432b942805b86f9768059a897886841734d05716a60a0e",
    ),
    pytest.param(
        "rydberg_features_geometries_iranian_churn.jsonl.xz",
        "7fc97ed184689b1479a3c18781c76d2a649a3585b741c6051dbecd50d0cb7ea0",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "rydberg_features_geometries_iranian_churn_15_folds.jsonl.xz",
        "25ea34a397d8c5d59fadb64f5ba6acc8cad32e4c9d94cd1a1d6b14509c5cd9b9",
        marks=pytest.mark.slow,
    ),
)


class TestSummaryMeasurement:
    """
    #763: a row of :func:`summarise` takes the mean and spread over its lines, so every line
    of a row must have measured its features alike, ``(measurement, shots,
    feature_map["shot_stream"])``; the check is per row, not per dataset.
    """

    @staticmethod
    def _line(v: float | None, arm: str, sampled: tuple[Any, ...], **extra: Any) -> dict:
        """A line of :class:`TestSummary` measured as ``sampled``; arm C exact, no config."""
        measurement, shots, stream = sampled
        config = None if arm == "C" else {"shots": shots}
        if config is not None and stream is not None:
            config["shot_stream"] = stream
        return TestSummary._line(
            v, arm, 1.0, 1.0, measurement=measurement, shots=shots, feature_map=config, **extra
        )

    def _run(self, sampled: tuple[Any, ...], **extra: Any) -> list[dict[str, Any]]:
        """Two folds of A and B at ``V/Ω = 1`` measured as ``sampled``, and of C (exact)."""
        return [
            self._line(v, arm, EXACT if arm == "C" else sampled, **extra)
            for arm, v in (("A", 1.0), ("B", 1.0), ("C", None))
            for _ in range(2)
        ]

    @staticmethod
    def _refusal(records: list[dict[str, Any]]) -> str:
        with pytest.raises(ValueError) as caught:
            bench.summarise(records)
        return str(caught.value)

    def test_an_old_and_a_new_shots_run_of_one_row_are_refused(self) -> None:
        """
        A shots run from before #762 and one since, joined: the rows of A and B would pool
        both kinds of line (``n`` doubled), not that of C, exact in both.  Refused at the
        first such row, with each value and its number of lines in the order first seen.
        """
        new, old = self._run(PER_MATRIX), self._run(SHARED)
        assert self._refusal(new + old) == (
            "dataset 'positive-control' has lines of 2 feature measurements (measurement, "
            "shots, shot_stream) in row A/linear chain V/Ω=1 γ/Ω=0.1: measurement shots, "
            "shots 16, shot_stream per-matrix (2 lines); measurement shots, shots 16, "
            "shot_stream not recorded (2 lines) (rows with more than one: 2 of 3).  "
            + MEASUREMENT_ADVICE
        )
        assert "shot_stream not recorded (2 lines); measurement shots" in self._refusal(old + new)

    def test_an_exact_and_a_shots_run_of_one_row_are_refused_with_their_label_bonds(
        self,
    ) -> None:
        """
        The same for exact against shots lines, naming the label bonds of a control line, and
        for two shots runs that differ only in ``--shots``; a line without the fields counts
        as one more value, ``not recorded``.
        """
        bonds = {"label_bonds": [[0, 1], [1, 2], [2, 3]]}
        exact, shots = self._run(EXACT, **bonds), self._run(PER_MATRIX, **bonds)
        assert self._refusal(exact + shots) == (
            "dataset 'positive-control' (label_bonds [[0, 1], [1, 2], [2, 3]]) has lines of 2 "
            "feature measurements (measurement, shots, shot_stream) in row A/linear chain "
            "V/Ω=1 γ/Ω=0.1: measurement exact, shots not recorded, shot_stream not recorded "
            "(2 lines); measurement shots, shots 16, shot_stream per-matrix (2 lines) (rows "
            "with more than one: 2 of 3).  " + MEASUREMENT_ADVICE
        )
        # Two shots runs that differ only in --shots.
        assert (
            "shots 16, shot_stream per-matrix (2 lines); measurement shots, shots 32, "
            "shot_stream per-matrix (2 lines) (rows with more than one: 2 of 3)."
        ) in self._refusal(shots + self._run(("shots", 32, "per-matrix"), **bonds))
        unrecorded = [
            {k: v for k, v in r.items() if k not in ("measurement", "shots", "feature_map")}
            for r in exact
        ]
        assert (
            "shot_stream not recorded (2 lines); measurement not recorded, shots not recorded, "
            "shot_stream not recorded (2 lines) (rows with more than one: 3 of 3)."
        ) in self._refusal(exact + unrecorded)

    def test_rows_of_other_measurements_print_as_before(self) -> None:
        """
        Per row, not per dataset: A and B sampled and C exact in one run print what the same
        lines print without the fields, the output from before #763.
        """
        run = self._run(PER_MATRIX)
        before = [
            {k: v for k, v in r.items() if k not in ("measurement", "shots", "feature_map")}
            for r in run
        ]
        assert bench.summarise(run) == bench.summarise(before)
        assert len(bench.summarise(run).splitlines()) == 4

    @pytest.mark.slow
    def test_one_shots_run_with_every_arm_summarises(
        self, shots_run: list[dict[str, Any]]
    ) -> None:
        """
        A run with shots holds both kinds of line in one dataset and still summarises: each
        row has both folds of its arm and head, and the mean and spread of their MCCs and
        the mean PR-AUC computed here.  The same run with ``shot_stream`` removed from every
        config, as a run from before #762 writes it, joined to the first is refused in the 6
        rows of A, B and D, not in the 3 of C and ``D-hybrid``.
        """

        def measured(r: dict[str, Any]) -> tuple[Any, ...]:
            return r["measurement"], r["shots"], (r["feature_map"] or {}).get("shot_stream")

        found = {(r["arm"], r["head"], measured(r)) for r in shots_run}
        assert found == {
            *((arm, head, PER_MATRIX) for arm in ("A", "B", "D") for head in ("linear", "mlp")),
            ("C", "linear", EXACT),
            ("C", "mlp", EXACT),
            ("D-hybrid", "hybrid", EXACT),
        }
        table = bench.summarise(shots_run).splitlines()
        assert len(table) == 1 + len(found)
        for row in table[1:]:
            cells = row.split()
            lines = [r for r in shots_run if (r["arm"], r["head"]) == (cells[4], cells[5])]
            mcc = np.array([r["mcc"] for r in lines])
            pr_auc = np.mean([r["pr_auc"] for r in lines])
            assert mcc.size == 2
            assert cells[6:10] == [
                f"{mcc.mean():.3f}",
                "±",
                f"{mcc.std(ddof=1):.3f}",
                f"{pr_auc:.3f}",
            ]
            assert cells[-1] == "2"
        old = [
            {**r, "feature_map": {k: v for k, v in r["feature_map"].items() if k != "shot_stream"}}
            if r["feature_map"] is not None
            else r
            for r in shots_run
        ]
        assert (
            "in row A/linear chain V/Ω=1 γ/Ω=0.1: measurement shots, shots 16, shot_stream "
            "per-matrix (2 lines); measurement shots, shots 16, shot_stream not recorded (2 "
            "lines) (rows with more than one: 6 of 9)."
        ) in self._refusal(shots_run + old)

    @pytest.mark.parametrize(("name", "digest"), COMMITTED_SUMMARIES)
    def test_the_committed_results_summarise_as_before(self, name: str, digest: str) -> None:
        """Every committed result file prints byte for byte what it printed before #763."""
        records = bench.read_records(ROOT / "docs" / "results" / name)
        assert {r["measurement"] for r in records} == {"exact"}
        text = bench.summarise(records)
        assert hashlib.sha256(text.encode()).hexdigest() == digest


class TestSeparationSubsamplingIntegration:
    def test_parser_options(self) -> None:
        parser = bench._parser()
        args = parser.parse_args(["--max-separation-rows", "500", "--separation-seed", "42"])
        assert args.max_separation_rows == 500
        assert args.separation_seed == 42

        # Disable subsampling via 0
        args_zero = parser.parse_args(["--max-separation-rows", "0"])
        assert args_zero.max_separation_rows == 0

    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            (2, "2"),
            (0, "0"),
            (-10, "-10"),
            (30.0, "30.0"),
            (30.5, "30.5"),
            (True, "True"),
            ("2000", "'2000'"),
        ],
    )
    def test_settings_validates_max_separation_rows(self, value: Any, shown: str) -> None:
        """An integer ``>= 3`` or ``None``; a float, a bool or a string is refused (#541)."""
        message = rf"^max_separation_rows must be an integer >= 3 or None; got {shown}\.$"
        with pytest.raises(ValueError, match=message):
            bench.Settings(max_separation_rows=value)

    def test_settings_accepts_an_integer_max_separation_rows(self) -> None:
        assert bench.Settings(max_separation_rows=3).max_separation_rows == 3
        assert bench.Settings(max_separation_rows=np.int64(500)).max_separation_rows == 500
        assert bench.Settings(max_separation_rows=None).max_separation_rows is None

    @pytest.mark.parametrize("value", ["1", "2", "-1", "-2000"])
    def test_the_command_line_refuses_a_limit_below_three_other_than_zero(
        self,
        value: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """
        ``0`` is the one spelling of "no limit"; 1, 2 and a negative value end
        in an argparse error naming the option, before anything runs (#541).
        """

        def run(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise AssertionError("the run must not start")

        monkeypatch.setattr(bench, "run", run)
        for quick in ([], ["--quick"]):
            with pytest.raises(SystemExit) as raised:
                bench.main(
                    [*quick, "--max-separation-rows", value, "--out", str(tmp_path / "o.jsonl")]
                )
            assert raised.value.code == 2
            error = capsys.readouterr().err.splitlines()[-1]
            assert error.endswith(
                "error: argument --max-separation-rows: must be 0 (no limit) or an integer "
                f">= 3; got {value}"
            )

    def test_the_help_text_names_the_spelling_of_no_limit(self) -> None:
        parser = bench._parser()
        (action,) = [a for a in parser._actions if "--max-separation-rows" in a.option_strings]
        assert action.help == (
            "rows of a training or test part above which its separation measures are computed "
            "on a class-stratified subsample of that many rows: an integer >= 3, or 0 for no "
            "limit"
        )
        assert action.default == 2000

    def test_quick_respects_separation_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_settings: list[Any] = []

        def spy_run_dataset(
            name: str, X: Any, y: Any, settings: Any, *args: Any, **kwargs: Any
        ) -> Any:
            captured_settings.append(settings)
            return []

        monkeypatch.setattr(bench, "run_dataset", spy_run_dataset)
        bench.main(
            [
                "--quick",
                "--max-separation-rows",
                "50",
                "--separation-seed",
                "12",
                "--out",
                str(tmp_path / "o1.jsonl"),
            ]
        )
        assert len(captured_settings) == 1
        assert captured_settings[0].max_separation_rows == 50
        assert captured_settings[0].separation_seed == 12

        captured_settings.clear()
        bench.main(["--quick", "--max-separation-rows", "0", "--out", str(tmp_path / "o2.jsonl")])
        assert len(captured_settings) == 1
        assert captured_settings[0].max_separation_rows is None

    def test_bounded_separation_records_and_training_unaffected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Verify that max_separation_rows bounds separation calculation rows,
        records the metadata in settings, and leaves model training using the full dataset.
        """
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = np.array([0, 1] * 50)
        data = {"positive-control": (X, y, 0)}

        # Capture model training inputs to verify full dataset was used.  The script trains
        # through hqnn_forge.benchmark.fit_and_predict (#524), so the spy goes there.
        train_row_counts: list[int] = []
        orig_train_model = benchmark.train_model

        def spy_train_model(
            model: Any,
            loss_fn: Any,
            optimizer: Any,
            x_train: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            train_row_counts.append(x_train.shape[0])
            return orig_train_model(model, loss_fn, optimizer, x_train, *args, **kwargs)

        monkeypatch.setattr(benchmark, "train_model", spy_train_model)

        settings = _replace(
            QUICK,
            arms=("C",),
            n_atoms=2,
            n_trials=0,
            n_splits=2,
            max_separation_rows=20,
            separation_seed=7,
        )
        records = bench.run(data, settings, tmp_path / "sub.jsonl", cache_dir=tmp_path / "csub")

        # In fold 0: 100 samples total, split into 50 train_part, 50 test.
        # With 4 validation folds, inner_train has ~37-38 rows.
        # Model training was NOT subsampled to 20 rows!
        assert len(train_row_counts) > 0
        assert all(count >= 35 for count in train_row_counts)

        # Separation measures for train must be bounded by max_separation_rows = 20
        for r in records:
            assert r["settings"]["max_separation_rows"] == 20
            assert r["settings"]["separation_seed"] == 7
            assert r["separation"]["train"]["n_samples"] == 20
            # Test partition has 50 samples, which also exceeds 20
            assert r["separation"]["test"]["n_samples"] == 20

    def test_seed_changes_separation_samples(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(80, 2)
        y = np.array([0, 1] * 40)
        data = {"positive-control": (X, y, 0)}

        s1 = _replace(
            QUICK,
            arms=("C",),
            n_atoms=2,
            n_trials=0,
            n_splits=2,
            max_separation_rows=25,
            separation_seed=1,
        )
        r1 = bench.run(data, s1, tmp_path / "r1.jsonl", cache_dir=tmp_path / "c1")

        s2 = _replace(
            QUICK,
            arms=("C",),
            n_atoms=2,
            n_trials=0,
            n_splits=2,
            max_separation_rows=25,
            separation_seed=2,
        )
        r2 = bench.run(data, s2, tmp_path / "r2.jsonl", cache_dir=tmp_path / "c2")

        # Different separation seeds produce different separation results on the subsample
        sep1 = r1[0]["separation"]["train"]["mean_within_distance"]
        sep2 = r2[0]["separation"]["train"]["mean_within_distance"]
        assert sep1 != sep2


# ---------------------------------------------------------------------------
# PCA degeneracy per fold (#525)
# ---------------------------------------------------------------------------


def _relative_gaps_by_hand(X: np.ndarray, fit_rows: np.ndarray, n_components: int) -> np.ndarray:
    """
    The relative eigenvalue gaps of step 2's PCA fit on ``fit_rows``, by SVD.

    The raw columns standardised by hand with their mean and population
    standard deviation on ``fit_rows`` (none of the data here has a constant
    column); the eigenvalues of the fit are the squared singular values of
    the centred rows, up to the common factor ``1/(n − 1)`` that the
    relative gaps divide out.
    """
    rows = X[fit_rows]
    scaled = (rows - rows.mean(axis=0)) / rows.std(axis=0)
    lam = np.linalg.svd(scaled - scaled.mean(axis=0), compute_uv=False) ** 2
    return np.asarray((lam[:n_components] - lam[1 : n_components + 1]) / lam[0])


def _with_correlation(rng: np.random.Generator, n: int, corr: np.ndarray) -> np.ndarray:
    """
    ``n`` rows whose sample covariance is exactly ``corr`` (unit diagonal).

    Orthonormal centred columns (QR of centred Gaussians) times
    ``√(n − 1) · corr^½``: their covariance is ``corr`` up to rounding, and
    with unit variances the column standardisation of step 2 changes the
    spectrum by one common factor only.
    """
    G = rng.standard_normal((n, corr.shape[0]))
    Q, _ = np.linalg.qr(G - G.mean(axis=0))
    w, V = np.linalg.eigh(corr)
    return np.asarray(Q @ (V * np.sqrt((n - 1) * w)) @ V.T)


class TestPcaDegeneracy:
    """
    Every line records the degeneracy of its fold's PCA fit, read off
    ``PCANormalizer.eigenvalue_gaps_`` and not from the warning (#525).

    The degenerate data: two blocks of two columns with correlation ½ and
    one with ⅕, so the spectrum of the correlation matrix is ``1 ± ½``
    twice and ``1 ± ⅕``: ``1.5, 1.5, 1.2, 0.8, 0.5, 0.5``.  With four
    components the relative gaps among them and at the cutoff are
    ``0, 0.3, 0.4, 0.3`` over ``1.5``: the leading pair is degenerate.
    """

    CORR = np.array(
        [
            [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.2, 1.0],
        ]
    )
    GAPS: ClassVar[list[float]] = [0.0, 0.3 / 1.5, 0.4 / 1.5, 0.3 / 1.5]
    SETTINGS = _replace(QUICK, arms=("C",), max_epochs=1)

    def _data(self, degenerate_rows: Any) -> tuple[np.ndarray, np.ndarray, Any]:
        """
        Labels, raw features with the degenerate spectrum on the rows
        ``degenerate_rows(fold, y)`` picks from fold 0 rebuilt by hand and
        Gaussian rows elsewhere, and that fold.
        """
        rng = np.random.default_rng(525)
        n = QUICK.n_samples
        y = (np.arange(n) % 2).astype(np.int64)
        dataset_index = bench.DATASETS.index("iranian-churn")
        fold = _fold_by_hand(rng.standard_normal((n, 6)), y, self.SETTINGS, dataset_index)
        rows = degenerate_rows(fold, y)
        X = rng.standard_normal((n, 6))
        X[rows] = _with_correlation(rng, rows.size, self.CORR)
        return X, y, fold

    @staticmethod
    def _tuning_fit_rows(fold: Any, y: np.ndarray) -> list[np.ndarray]:
        inner = stratified_kfold(
            y[fold.train_part], QUICK.inner_folds, random_state=fold.seeds["tune_seed"]
        )
        return [fold.train_part[fit] for fit, _ in inner]

    def _run(self, X: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
        # The warning is hidden here: the flag must not depend on it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return list(
                bench.run_dataset("iranian-churn", X, y, self.SETTINGS, bench.FeatureCache(None))
            )

    def test_the_construction_has_the_spectrum(self) -> None:
        X, _, fold = self._data(lambda fold, y: fold.train_part)
        np.testing.assert_allclose(
            _relative_gaps_by_hand(X, fold.train_part, 4), self.GAPS, rtol=0, atol=1e-12
        )

    def test_two_equal_leading_eigenvalues_set_the_flag_on_every_line(self) -> None:
        X, y, fold = self._data(lambda fold, y: fold.train_part)
        records = self._run(X, y)
        assert [(r["arm"], r["head"]) for r in records] == [("C", "linear"), ("C", "mlp")]
        for record in records:
            pca = record["pca"]
            assert pca["degenerate"] is True
            assert pca["gap_threshold"] == 1e-12
            np.testing.assert_allclose(pca["relative_gaps"], self.GAPS, rtol=0, atol=1e-12)
            assert pca["min_relative_gap"] == min(pca["relative_gaps"]) < 1e-12
            # Only the outer fit is degenerate: a tuning fold fits a subset of its rows.
            for tuning_pca, rows in zip(
                record["tuning"]["pca"], self._tuning_fit_rows(fold, y), strict=True
            ):
                assert tuning_pca["degenerate"] is False
                np.testing.assert_allclose(
                    tuning_pca["relative_gaps"],
                    _relative_gaps_by_hand(X, rows, 4),
                    rtol=1e-8,
                    atol=1e-14,
                )
            line = json.loads(bench.to_json_line(record))
            assert line["pca"]["degenerate"] is True
            assert line["tuning"]["pca"][0]["degenerate"] is False
        # deg and t-deg, then feat s, tot feat s (#785), fit s and n
        assert bench.summarise(records).splitlines()[1].split()[-6:-4] == ["1", "0"]

    def test_the_run_warns_as_well(self) -> None:
        X, y, _ = self._data(lambda fold, y: fold.train_part)
        with pytest.warns(RuntimeWarning, match=r"eigenvalues 0 and 1 are degenerate"):
            list(bench.run_dataset("iranian-churn", X, y, self.SETTINGS, bench.FeatureCache(None)))

    def test_a_degenerate_tuning_fold_is_recorded_in_tuning_only(self) -> None:
        """
        Tuning fold 0's fitting rows carry the degenerate spectrum; the outer
        fit, over those rows and the held-out ones, does not.
        """
        X, y, fold = self._data(lambda fold, y: self._tuning_fit_rows(fold, y)[0])
        records = self._run(X, y)
        _, second_fit = self._tuning_fit_rows(fold, y)
        for record in records:
            assert record["pca"]["degenerate"] is False
            np.testing.assert_allclose(
                record["pca"]["relative_gaps"],
                _relative_gaps_by_hand(X, fold.train_part, 4),
                rtol=1e-8,
                atol=1e-14,
            )
            first, second = record["tuning"]["pca"]
            assert first["degenerate"] is True
            np.testing.assert_allclose(first["relative_gaps"], self.GAPS, rtol=0, atol=1e-12)
            assert second["degenerate"] is False
            np.testing.assert_allclose(
                second["relative_gaps"],
                _relative_gaps_by_hand(X, second_fit, 4),
                rtol=1e-8,
                atol=1e-14,
            )
        assert bench.summarise(records).splitlines()[1].split()[-6:-4] == ["0", "1"]

    @pytest.mark.slow
    def test_the_positive_control_is_not_degenerate(self, quick: Any) -> None:
        """
        On the ``--quick`` run (every arm), each line's gaps are those of an
        SVD of the fold's training part and of its tuning folds' fitting
        rows, and none is near the threshold.
        """
        fold = quick.fold
        outer = _relative_gaps_by_hand(quick.X, fold.train_part, QUICK.n_atoms)
        tuning = [
            _relative_gaps_by_hand(quick.X, rows, QUICK.n_atoms)
            for rows in self._tuning_fit_rows(fold, quick.y)
        ]
        assert min(outer.min(), *(g.min() for g in tuning)) > 1e-6
        for record in quick.lines:
            assert record["pca"]["degenerate"] is False
            np.testing.assert_allclose(record["pca"]["relative_gaps"], outer, rtol=1e-8)
            assert record["pca"]["min_relative_gap"] == pytest.approx(outer.min(), rel=1e-8)
            for got, want in zip(record["tuning"]["pca"], tuning, strict=True):
                assert got["degenerate"] is False
                np.testing.assert_allclose(got["relative_gaps"], want, rtol=1e-8)
        assert bench.summarise(quick.records).splitlines()[1].split()[-6:-4] == ["0", "0"]


# ---------------------------------------------------------------------------
# Device realism: detection error, position jitter and the global drive (#775)
# ---------------------------------------------------------------------------

#: Aquila's detection error ``(p01, p10)``, *Hardware realism* of ``docs/rydberg-model.md``.
AQUILA_ERROR = (0.01, 0.08)
#: Inputs typed in, with both bounds of the pattern (``h = 0`` and ``h = 1`` at ``±40``).
REALISM_X = np.array(
    [
        [-2.0, -0.5, 0.3, 2.9],
        [0.1, 1.2, -3.0, 0.0],
        [1.5, 1.5, -1.5, -1.5],
        [40.0, -40.0, 0.0, 0.7],
    ]
)


def _realistic(**changes: Any) -> Any:
    """``QUICK`` with all three options: the detection error, jitter and a ramped drive."""
    options: dict[str, Any] = {
        "readout_error": AQUILA_ERROR,
        "position_jitter": 0.2,
        "jitter_samples": 2,
        "encoding": "global-drive",
        "levels": 3,
        "t_ramp": 0.0503,
        "n_ramp_segments": 2,
    }
    return _replace(QUICK, **{**options, **changes})


def _rep(settings: Any, arm: str, v: float | None = None) -> Any:
    """The representation of ``arm`` in ``settings`` (at ``V/Ω = v`` for A and B)."""
    (found,) = [
        r
        for r in bench.representations(settings)
        if r.arm == arm and (v is None or r.v_over_omega == v)
    ]
    return found


class TestDetectionError:
    """A line's exact features are ``p01 + (1 − p01 − p10) F`` of the error-free map."""

    @pytest.mark.parametrize("arm", ["A", "B"])
    def test_a_rydberg_arm_reads_through_the_confusion_matrix(self, arm: str) -> None:
        p01, p10 = AQUILA_ERROR
        noisy = _rep(_replace(QUICK, readout_error=AQUILA_ERROR), arm).compute(REALISM_X)
        clean = _map_by_hand(4, 1.0, 0.1, 40, interactions=arm == "A").transform(REALISM_X)
        expected = p01 + (1 - p01 - p10) * clean.numpy()
        np.testing.assert_allclose(noisy, expected, rtol=0, atol=1e-15)
        assert np.abs(noisy - clean.numpy()).max() > 0.01

    @pytest.mark.slow
    def test_the_lines_of_a_run_are_those_of_the_error_free_map_read_through_it(self) -> None:
        """The recorded separation measures, from the error-free map's features mapped by hand."""
        p01, p10 = AQUILA_ERROR
        settings = _replace(QUICK, arms=("A", "B"), n_trials=0, readout_error=AQUILA_ERROR)
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        records = list(
            bench.run_dataset(
                "positive-control", X, y, settings, bench.FeatureCache(None), data_seed=data_seed
            )
        )
        fold = _fold_by_hand(X, y, settings, 0)
        test = fold.inputs[fold.test_idx]
        assert len(records) == 4
        for record in records:
            clean = _map_by_hand(4, 1.0, 0.1, 40, interactions=record["arm"] == "A")
            read = p01 + (1 - p01 - p10) * clean.transform(test).numpy()
            expected = separation_measures(read, y[fold.test_idx]).to_dict()
            assert record["separation"]["test"] == pytest.approx(expected, rel=1e-7, abs=1e-12)
            assert record["feature_map"]["config"]["readout_error"] == list(AQUILA_ERROR)
            assert record["measurement"] == "exact"

    def test_arm_d_reads_through_the_same_confusion_matrix(self) -> None:
        """
        ``⟨Z⟩ → (1 − p01 − p10) ⟨Z⟩ + (p10 − p01)`` of the circuit written out gate by gate,
        which is ``⟨n̂⟩ = p01 + (1 − p01 − p10) ⟨n⟩`` for ``n = (1 − Z)/2``: one model for A,
        B and D.
        """
        p01, p10 = AQUILA_ERROR
        circuit_seed = bench.run_seeds(QUICK.random_state)["circuit_seed"]
        rep = _rep(_replace(QUICK, readout_error=AQUILA_ERROR), "D")
        z = _circuit_by_hand(REALISM_X, circuit_seed)
        read = rep.compute(REALISM_X)
        np.testing.assert_allclose(read, (1 - p01 - p10) * z + (p10 - p01), atol=1e-6)
        np.testing.assert_allclose((1 - read) / 2, p01 + (1 - p01 - p10) * (1 - z) / 2, atol=1e-6)
        assert rep.config["config"]["readout_error"] == list(AQUILA_ERROR)
        # Without the option arm D's config holds no such key: its cache keys are those of main.
        assert "readout_error" not in _rep(QUICK, "D").config["config"]

    @pytest.mark.slow
    def test_with_shots_arm_d_flips_every_shot_and_repeats_from_its_stream(self) -> None:
        """
        Each estimate of ``S`` flipped shots has mean ``⟨Z'⟩ = c ⟨Z⟩ + (p10 − p01)`` and
        variance ``(1 − ⟨Z'⟩²)/S``: over 64 copies of the inputs, the mean is within 5
        standard errors of the exact error-mapped value; the matrix's stream repeats it.
        Flipped, not rescaled: each estimate is ``(2k' − S)/S`` with ``k'`` the whole number
        of shots read 0, which ``c ẑ + (p10 − p01)`` of a sampled ``ẑ`` is not (it would
        pass the mean).
        """
        p01, p10 = AQUILA_ERROR
        shots = 256
        settings = _replace(QUICK, readout_error=AQUILA_ERROR, shots=shots)
        rep = _rep(settings, "D")
        copies = np.tile(REALISM_X, (64, 1))
        sampled = rep.compute(copies).reshape(64, *REALISM_X.shape)
        zeros_read = (1 + sampled) * shots / 2
        np.testing.assert_allclose(zeros_read, np.round(zeros_read), rtol=0, atol=1e-9)
        circuit_seed = bench.run_seeds(QUICK.random_state)["circuit_seed"]
        exact = (1 - p01 - p10) * _circuit_by_hand(REALISM_X, circuit_seed) + (p10 - p01)
        error = np.sqrt((1 - exact**2) / (shots * 64))
        assert np.all(np.abs(sampled.mean(axis=0) - exact) < 5 * error)
        np.testing.assert_array_equal(rep.compute(copies), sampled.reshape(copies.shape))
        assert np.abs(sampled - exact).max() > 0.01  # sampled, not exact


class TestGlobalDrive:
    @pytest.mark.parametrize("gamma", [0.0, 0.1])
    @pytest.mark.parametrize("interactions", [True, False])
    def test_continuous_levels_without_a_ramp_are_the_square_pulse_bit_for_bit(
        self, gamma: float, interactions: bool
    ) -> None:
        """The library's guarantee, here through the script's builder."""
        drive = _replace(QUICK, encoding="global-drive", levels="continuous", t_ramp=0.0)
        X = torch.from_numpy(REALISM_X)
        ramped = bench.feature_map(drive, 1.0, gamma, interactions=interactions)
        square = bench.feature_map(QUICK, 1.0, gamma, interactions=interactions)
        assert isinstance(ramped.encoding, GlobalDrivePulseEncoding)
        assert ramped.encoding.levels is None
        assert torch.equal(ramped.transform(X), square.transform(X))

    def test_the_builder_passes_every_setting(self) -> None:
        fmap = bench.feature_map(_realistic(), 1.0, 0.1, interactions=True)
        by_hand = GlobalDrivePulseEncoding(
            4, omega=OMEGA, levels=3, t_ramp=0.0503, n_ramp_segments=2, t=math.pi / OMEGA
        )
        assert fmap.encoding.get_config() == by_hand.get_config()
        assert fmap.readout_error == AQUILA_ERROR
        assert (fmap.position_jitter, fmap.jitter_samples) == (0.2, 2)
        assert fmap.n_steps == QUICK.n_steps

    def test_a_line_records_the_whole_pulse_and_its_device_violations(self) -> None:
        """
        ``evolution_time_us`` is ``T + t_ramp``; every global-drive map records its
        violations of ``AQUILA_LIMITS``, also outside the preset (here none), and a square
        pulse none at all (its config, and the cache key, are those of ``main``).
        """
        settings = _realistic(arms=("A", "B"))
        for rep in bench.representations(settings):
            fields = bench._grid_fields(rep, settings)
            assert fields["evolution_time_us"] == pytest.approx(0.25 + 0.0503, abs=1e-15)
            assert rep.config["device_violations"] == []
        for rep in bench.representations(_replace(QUICK, arms=("A", "B"))):
            assert bench._grid_fields(rep, QUICK)["evolution_time_us"] == 0.25
            assert "device_violations" not in rep.config

    def test_a_drive_outside_the_limits_records_them_without_the_preset(self) -> None:
        """Without ``--device`` the run is simulated as given and the list says what breaks."""
        no_ramp = _replace(QUICK, encoding="global-drive", levels=4, t_ramp=0.0)
        assert _rep(no_ramp, "A").config["device_violations"] == [
            "omega_slew_max: Ω / t_ramp = inf rad/µs² exceeds 250 rad/µs²"
        ]
        short = _replace(QUICK, encoding="global-drive", levels=4, t_ramp=0.05, n_ramp_segments=1)
        assert _rep(short, "B").config["device_violations"] == [
            "omega_slew_max: Ω / t_ramp = 251.327 rad/µs² exceeds 250 rad/µs²"
        ]


class TestPositionJitter:
    #: The run's seed sequence, ``[random_state, 2^16]``, written out (:func:`bench.run_seeds`).
    RUN_ROOT = (0, 1 << 16)

    def test_the_seed_is_the_fourth_draw_of_the_run_and_a_and_b_share_it(self) -> None:
        """The three seeds of ``run_seeds`` stay what they were; A and B get the fourth."""
        children = np.random.SeedSequence(list(self.RUN_ROOT)).spawn(4)
        seeds = [int(child.generate_state(1)[0]) for child in children]
        assert list(bench.run_seeds(0).values()) == seeds[:3]
        assert bench.jitter_seed(0) == seeds[3]
        settings = _realistic(v_over_omega=(0.5, 2.0), gamma_over_omega=(0.0, 0.1))
        configs = [r.config["config"] for r in bench.representations(settings) if r.arm in "AB"]
        assert len(configs) == 8
        for config in configs:
            assert (config["position_jitter"], config["jitter_samples"]) == (0.2, 2)
            assert config["jitter_seed"] == seeds[3]

    def test_arm_a_is_the_mean_of_maps_on_registers_displaced_by_hand(self) -> None:
        """
        The displacements of the run's seed, drawn here as ``N(0, σ²)`` per coordinate
        (one ``randn((K, N, 2))``), and the features of a map on each displaced chain,
        averaged: the mixture the library reads (*Geometry* of the model page).
        """
        settings = _replace(QUICK, position_jitter=0.2, jitter_samples=3)
        generator = torch.Generator().manual_seed(bench.jitter_seed(QUICK.random_state))
        z = torch.randn((3, 4, 2), generator=generator, dtype=torch.float64)
        chain = AtomRegister.chain(4, bench.spacing(1.0, OMEGA, DEFAULT_C6)).positions
        features = [
            RydbergFeatureMap(
                AtomRegister(chain + 0.2 * z[k]),
                PulseEncoding(4, omega=OMEGA),
                c6=DEFAULT_C6,
                gamma=0.1 * OMEGA,
                n_steps=40,
            ).transform(REALISM_X)
            for k in range(3)
        ]
        expected = (features[0] + features[1] + features[2]).numpy() / 3
        jittered = _rep(settings, "A").compute(REALISM_X)
        np.testing.assert_allclose(jittered, expected, rtol=0, atol=1e-14)
        fixed = _rep(QUICK, "A").compute(REALISM_X)
        assert np.abs(jittered - fixed).max() > 1e-4

    def test_a_line_records_the_jitter_seed_only_with_jitter(self) -> None:
        settings = _replace(
            QUICK, arms=("A", "C"), n_trials=0, position_jitter=0.2, jitter_samples=2
        )
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        records = list(
            bench.run_dataset(
                "positive-control", X, y, settings, bench.FeatureCache(None), data_seed=data_seed
            )
        )
        assert {r["seeds"]["jitter_seed"] for r in records} == {bench.jitter_seed(0)}
        default = list(
            bench.run_dataset(
                "positive-control",
                X,
                y,
                _replace(settings, position_jitter=0.0, jitter_samples=None),
                bench.FeatureCache(None),
                data_seed=data_seed,
            )
        )
        assert all("jitter_seed" not in r["seeds"] for r in default)
        assert [r["seeds"] for r in default] == [
            {k: v for k, v in r["seeds"].items() if k != "jitter_seed"} for r in records
        ]


class TestArmBUnderRealism:
    """Arm B keeps its definition: A's encoding, detection error and jitter, no interactions."""

    def test_b_differs_from_a_in_the_interactions_flag_only(self) -> None:
        settings = _realistic()
        a, b = (_rep(settings, arm).config for arm in ("A", "B"))
        assert a["config"]["interactions"] is True and b["config"]["interactions"] is False
        assert {**a["config"], "interactions": None} == {**b["config"], "interactions": None}
        assert {k: v for k, v in a.items() if k != "config"} == {
            k: v for k, v in b.items() if k != "config"
        }

    @pytest.mark.parametrize("gamma", [0.0, 0.1])
    def test_each_feature_of_b_is_a_one_atom_map_of_its_own_input(self, gamma: float) -> None:
        """
        The derivation behind the reuse: without interactions the state stays a product and
        feature ``i`` is a one-atom map of input ``i`` under the same drive, levels and
        detection error, wherever the atom sits and however it is displaced.
        """
        settings = _realistic()
        b = bench.feature_map(settings, 1.0, gamma, interactions=False).transform(REALISM_X)
        alone = RydbergFeatureMap(
            AtomRegister.chain(1, 1.0),
            GlobalDrivePulseEncoding(1, omega=OMEGA, levels=3, t_ramp=0.0503, n_ramp_segments=2),
            c6=DEFAULT_C6,
            gamma=gamma * OMEGA,
            n_steps=40 if gamma else None,
            readout_error=AQUILA_ERROR,
        )
        by_hand = np.column_stack(
            [alone.transform(REALISM_X[:, [i]]).numpy()[:, 0] for i in range(4)]
        )
        # Equal up to the rounding of the 16 × 16 and 2 × 2 products over 5 segments (40
        # steps on the plateau, 5 on each edge segment since #782): 1.3e-13 on the
        # declared floors (torch 2.3) with 40 on every one, as across geometries above.
        np.testing.assert_allclose(b.numpy(), by_hand, rtol=0, atol=1e-12)

    @pytest.mark.slow
    def test_a_reused_b_matrix_is_the_one_recomputed_at_another_v(self) -> None:
        """
        Under all three options: the B line of the second ``V/Ω``, reused from the first,
        carries the features of B recomputed at the second ``V/Ω``, while A's differ.
        """
        settings = _realistic(
            arms=("A", "B"), n_trials=0, v_over_omega=(0.5, 2.0), gamma_over_omega=(0.1,)
        )
        X, y, data_seed = bench.load_dataset("positive-control", settings)
        records = list(
            bench.run_dataset(
                "positive-control", X, y, settings, bench.FeatureCache(None), data_seed=data_seed
            )
        )
        (reused,) = [
            r
            for r in records
            if r["arm"] == "B" and r["head"] == "linear" and r["v_over_omega"] == 2.0
        ]
        assert reused["reused_from_v_over_omega"] == 0.5
        fold = _fold_by_hand(X, y, settings, 0)
        stacked = np.concatenate(
            [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
        )
        first = _rep(settings, "B", 0.5).compute(stacked)
        again = _rep(settings, "B", 2.0).compute(stacked)
        np.testing.assert_array_equal(again, first)
        test = again[-fold.test_idx.size :]
        expected = separation_measures(test, y[fold.test_idx]).to_dict()
        assert reused["separation"]["test"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert reused["feature_map"] == _rep(settings, "B", 2.0).config
        a_first = _rep(settings, "A", 0.5).compute(stacked)
        a_again = _rep(settings, "A", 2.0).compute(stacked)
        assert np.abs(a_again - a_first).max() > 1e-3


class TestDevicePreset:
    def test_the_preset_sets_every_documented_value(self) -> None:
        preset = bench.device_preset("aquila", OMEGA)
        assert preset == {
            "readout_error": (0.01, 0.08),
            "position_jitter": 0.2,
            "encoding": "global-drive",
            "t_ramp": 0.0503,
            "n_ramp_segments": bench.AQUILA_RAMP_SEGMENTS,
            "shots": 100,
        }
        # The edge at the slew bound, rounded up to 1e-4 µs: Ω/250 = 0.050265 µs.
        assert OMEGA / 0.0503 <= 250.0 < OMEGA / 0.0502
        assert AQUILA_LIMITS.omega_slew_max == 250.0

    @pytest.mark.parametrize(
        ("omega", "t_ramp"),
        [
            (4.0 * math.pi, 0.0503),
            (10.0, 0.04),
            (15.8, 0.0632),
            (1.0, 0.004),
            # Ω/250 = 0.050030 µs, below the middle of its step: the nearest would be 0.0500,
            # a slope of 250.15 rad/µs².
            (12.5075, 0.0501),
            # 10⁴ Ω / 250 = 173 exactly, but Ω / 0.0173 rounds to 250.00000000000003.
            (4.325, 0.0174),
            # 10⁴ Ω / 250 rounds to 122.00000000000001, and Ω / 0.0122 to 250 exactly.
            (math.nextafter(3.05, 4.0), 0.0122),
        ],
    )
    def test_the_edge_is_the_shortest_multiple_of_the_resolution_within_the_slew(
        self, omega: float, t_ramp: float
    ) -> None:
        """``k / 10⁴`` µs with the smallest ``k`` for which ``Ω / t_ramp <= 250``."""
        found = bench.ramp_at_slew_bound(omega, 250.0)
        assert found == t_ramp
        k = round(found * 1e4)
        assert omega / (k / 1e4) <= 250.0
        assert omega / ((k - 1) / 1e4) > 250.0
        encoding = GlobalDrivePulseEncoding(
            1, omega=omega, levels=None, t_ramp=found, n_ramp_segments=1
        )
        assert "omega_slew_max" not in " ".join(encoding.device_violations(AQUILA_LIMITS))

    def test_its_maps_report_no_violation(self) -> None:
        settings = _replace(
            QUICK,
            **bench.device_preset("aquila", OMEGA),
            device="aquila",
            levels=8,
            jitter_samples=2,
        )
        reps = [r for r in bench.representations(settings) if r.arm in ("A", "B")]
        assert len(reps) == 2
        for rep in reps:
            assert rep.config["device_violations"] == []
            fmap = bench.feature_map(settings, 1.0, 0.1, interactions=rep.arm == "A")
            assert fmap.encoding.device_violations(AQUILA_LIMITS) == []
            assert fmap.encoding.duration == pytest.approx(0.3003, abs=1e-15)

    @pytest.mark.parametrize(
        ("changes", "named"),
        [
            ({"omega": 16.0}, "omega_max: Ω = 16 rad/µs exceeds 15.8 rad/µs"),
            ({"omega_t": 52.0}, "duration_max: T + t_ramp = 4.18833 µs exceeds 4 µs"),
        ],
    )
    def test_a_pulse_that_violates_is_refused_with_the_bound_named(
        self, changes: dict[str, float], named: str
    ) -> None:
        omega = changes.get("omega", OMEGA)
        with pytest.raises(ValueError) as caught:
            _replace(
                QUICK,
                **bench.device_preset("aquila", omega),
                device="aquila",
                levels=8,
                jitter_samples=2,
                **changes,
            )
        message = str(caught.value)
        assert message.startswith("device='aquila' refuses the pulse of every grid point")
        assert message.endswith(named + ".")

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"levels": None}, "levels is required with encoding='global-drive'"),
            ({"jitter_samples": None}, "jitter_samples is required with position_jitter > 0"),
            ({"shots": 1000}, "device='aquila' sets shots=100; got shots=1000."),
            (
                {"t_ramp": 0.06},
                re.escape("device='aquila' sets t_ramp=0.0503; got t_ramp=0.06."),
            ),
            (
                {"device": "rubidium"},
                "device must be None or one of \\['aquila'\\]; got 'rubidium'",
            ),
        ],
    )
    def test_the_preset_still_requires_levels_and_k_and_keeps_its_values(
        self, changes: dict[str, Any], match: str
    ) -> None:
        values = {
            **bench.device_preset("aquila", OMEGA),
            "device": "aquila",
            "levels": 8,
            "jitter_samples": 2,
        }
        with pytest.raises(ValueError, match=match):
            _replace(QUICK, **{**values, **changes})

    def test_the_command_line_fills_in_the_preset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def mock_run(datasets: Any, settings: Any, out: Any, *, cache_dir: Any = None) -> list:
            captured["settings"] = settings
            return []

        monkeypatch.setattr(bench, "run", mock_run)
        out = str(tmp_path / "o.jsonl")
        bench.main(["--device", "aquila", "--levels", "8", "--jitter-samples", "4", "--out", out])
        expected = bench.Settings(
            device="aquila", levels=8, jitter_samples=4, **bench.device_preset("aquila", OMEGA)
        )
        assert captured["settings"] == expected
        bench.main(
            ["--quick", "--device", "aquila", "--levels", "continuous", "--jitter-samples", "2"]
            + ["--out", out]
        )
        assert captured["settings"] == _replace(
            QUICK,
            device="aquila",
            levels="continuous",
            jitter_samples=2,
            **bench.device_preset("aquila", OMEGA),
        )

    @pytest.mark.parametrize(
        "option",
        [
            ["--shots", "100"],
            ["--readout-error", "0.01", "0.08"],
            ["--position-jitter", "0.2"],
            ["--encoding", "global-drive"],
            ["--t-ramp", "0.06"],
            ["--n-ramp-segments", "8"],
        ],
    )
    def test_an_option_the_preset_sets_is_refused_beside_it(
        self, option: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = ["--device", "aquila", "--levels", "8", "--jitter-samples", "4", *option]
        with pytest.raises(SystemExit):
            bench.main([*argv, "--out", str(tmp_path / "o.jsonl")])
        error = capsys.readouterr().err
        assert f"argument {option[0]}: --device aquila sets it to" in error
        assert not (tmp_path / "o.jsonl").exists()


class TestRealismSettings:
    """Each option is validated as the existing fields are: the field and the value named."""

    DRIVE: ClassVar[dict[str, Any]] = {"encoding": "global-drive", "levels": 4, "t_ramp": 0.0}

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"readout_error": (0.1,)}, "readout_error must be None or a pair"),
            ({"readout_error": (0.1, 1.5)}, "readout_error: p10 must be a number in \\[0, 1\\]"),
            ({"readout_error": (math.nan, 0.1)}, "readout_error: p01 must be a number"),
            ({"position_jitter": -0.1}, "position_jitter must be a finite number >= 0"),
            ({"position_jitter": math.inf}, "position_jitter must be a finite number >= 0"),
            ({"position_jitter": True}, "position_jitter must be a finite number >= 0"),
            ({"position_jitter": 0.2}, "jitter_samples is required with position_jitter > 0"),
            ({"jitter_samples": 4}, "jitter_samples is the number of displaced registers"),
            (
                {"position_jitter": 0.2, "jitter_samples": 0},
                "jitter_samples must be an integer >= 1; got 0",
            ),
            (
                {"position_jitter": 0.2, "jitter_samples": 2.0},
                "jitter_samples must be an integer >= 1; got 2.0",
            ),
            ({"encoding": "trapezoid"}, "encoding must be one of \\['square', 'global-drive'\\]"),
            ({"levels": 4}, "levels is a setting of the global-drive encoding; got levels=4"),
            ({"t_ramp": 0.05}, "t_ramp is a setting of the global-drive encoding"),
            ({"n_ramp_segments": 4}, "n_ramp_segments is a setting of the global-drive encoding"),
            ({**DRIVE, "levels": None}, "levels is required with encoding='global-drive'"),
            ({**DRIVE, "levels": 1}, "levels must be an integer >= 2 or 'continuous'; got 1"),
            ({**DRIVE, "levels": "none"}, "levels must be an integer >= 2 or 'continuous'"),
            ({**DRIVE, "t_ramp": None}, "t_ramp is required with encoding='global-drive'"),
            ({**DRIVE, "t_ramp": -0.01}, "t_ramp must be a finite number in \\[0, T\\] µs"),
            ({**DRIVE, "t_ramp": 0.26}, "T = omega_t / omega = 0.25, .*got 0.26"),
            ({**DRIVE, "t_ramp": 0.05}, "n_ramp_segments is required with t_ramp > 0"),
            (
                {**DRIVE, "n_ramp_segments": 4},
                "n_ramp_segments is used only with t_ramp > 0; got n_ramp_segments=4",
            ),
            (
                {**DRIVE, "t_ramp": 0.05, "n_ramp_segments": 0},
                "n_ramp_segments must be an integer >= 1; got 0",
            ),
        ],
    )
    def test_invalid_options(self, changes: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _replace(QUICK, **changes)

    def test_valid_options_and_their_record(self) -> None:
        settings = _realistic(readout_error=[0.0, 0.0], levels="continuous")
        assert settings.readout_error == (0.0, 0.0)
        recorded = json.loads(json.dumps(settings.as_dict()))
        assert {k: recorded[k] for k in self.REALISM_DEFAULTS} == {
            "readout_error": [0.0, 0.0],
            "position_jitter": 0.2,
            "jitter_samples": 2,
            "encoding": "global-drive",
            "levels": "continuous",
            "t_ramp": 0.0503,
            "n_ramp_segments": 2,
            "device": None,
        }

    #: The fields of #775 and their defaults: a run without the options records these.
    REALISM_DEFAULTS: ClassVar[dict[str, Any]] = {
        "readout_error": None,
        "position_jitter": 0.0,
        "jitter_samples": None,
        "encoding": "square",
        "levels": None,
        "t_ramp": None,
        "n_ramp_segments": None,
        "device": None,
    }

    def test_the_command_line_passes_every_option(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def mock_run(datasets: Any, settings: Any, out: Any, *, cache_dir: Any = None) -> list:
            captured["settings"] = settings
            return []

        monkeypatch.setattr(bench, "run", mock_run)
        argv = [
            "--readout-error", "0.02", "0.05",
            "--position-jitter", "0.3",
            "--jitter-samples", "6",
            "--encoding", "global-drive",
            "--levels", "5",
            "--t-ramp", "0.06",
            "--n-ramp-segments", "8",
            "--out", str(tmp_path / "o.jsonl"),
        ]  # fmt: skip
        bench.main(argv)
        assert captured["settings"] == bench.Settings(
            readout_error=(0.02, 0.05),
            position_jitter=0.3,
            jitter_samples=6,
            encoding="global-drive",
            levels=5,
            t_ramp=0.06,
            n_ramp_segments=8,
        )
        bench.main(["--quick", *argv])
        assert captured["settings"] == _replace(
            QUICK,
            readout_error=(0.02, 0.05),
            position_jitter=0.3,
            jitter_samples=6,
            encoding="global-drive",
            levels=5,
            t_ramp=0.06,
            n_ramp_segments=8,
        )

    def test_levels_on_the_command_line_and_every_new_option_has_help(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        options = (
            "--readout-error",
            "--position-jitter",
            "--jitter-samples",
            "--encoding",
            "--levels",
            "--t-ramp",
            "--n-ramp-segments",
            "--device",
        )
        quick = next(a for a in bench._parser()._actions if "--quick" in a.option_strings)
        for option in options:
            assert option in str(quick.help)
            (action,) = [a for a in bench._parser()._actions if option in a.option_strings]
            assert action.help
        parser = bench._parser()
        assert parser.parse_args(["--levels", "continuous"]).levels == "continuous"
        assert parser.parse_args(["--levels", "8"]).levels == 8
        with pytest.raises(SystemExit):
            parser.parse_args(["--levels", "none"])
        assert "must be an integer >= 2 or 'continuous'; got 'none'" in capsys.readouterr().err


#: The inputs the features of :class:`TestDefaultRunIsUnchanged` were recorded on.
PIN_X = np.array(
    [
        [-2.0, -0.5, 0.3, 2.9],
        [0.1, 1.2, -3.0, 0.0],
        [1.5, 1.5, -1.5, -1.5],
        [0.0, 0.0, 0.0, 0.0],
        [3.0, -3.0, 2.0, -1.0],
    ]
)


class TestDefaultRunIsUnchanged:
    """
    #775: without any device option a run writes the cache keys, features and lines of the
    runner before it.  Every value below was recorded on 26ba88f (the head of #779, which #775
    is built on) with the unmodified script, Python 3.14 and the locked dependencies; the
    scores of the seeded ``--quick`` run are pinned by :class:`TestSeededRunsAreUnchanged`.
    """

    #: ``(number, SHA-256)`` of the newline-joined ``cache_key("0" * 64, config, KEY_ENV,
    #: "c" * 64)`` of every representation with a config, in the order of
    #: ``representations``: ``--quick``'s A, B and D, the default grid's 56 maps and D, and
    #: ``--quick`` with ``shots=16``.
    KEYS: ClassVar[dict[str, tuple[int, str]]] = {
        "quick": (3, "898aa236823b30621e6a3d81dc6d09eed2c99482983741c9f510696c24d19072"),
        "default": (57, "2df2ebb873890af1f11b1622845f983034ad6671c04dc773ba0c79816552e8d5"),
        "quick-shots": (3, "997a81c4bf886e0faa0812b46af46a5931fa6da6f0abc22e585d43b9c38d88de"),
    }
    #: ``Representation.compute(PIN_X)`` of ``--quick``'s arms A, B and D, as ``repr``.
    FEATURES: ClassVar[dict[str, list[list[float]]]] = {
        "A": [[0.7823144645181488, 0.7126316586422141, 0.735503279092411, 0.2698203916289493], [0.6713341430207072, 0.6530310727032648, 0.6059405649435567, 0.7302477536532479], [0.3366214315097612, 0.5825652620176092, 0.6621254784015489, 0.807948335704195], [0.7360294093617219, 0.7511591130154893, 0.7511591130154889, 0.7360294093617211], [0.3079979422773812, 0.719182013798775, 0.621626873641162, 0.8333868049334938]],
        "B": [[0.9241697913267485, 0.6318029034189906, 0.34138953871832695, 0.037038639525952294], [0.41150335436299595, 0.12546183076898657, 0.9561161390676821, 0.4482419014409549], [0.09029116224253067, 0.09029116224253031, 0.8749974506439965, 0.8749974506439943], [0.4482419014409601, 0.44824190144096043, 0.448241901440961, 0.4482419014409613], [0.03612885632138398, 0.956116139067614, 0.057389916555876976, 0.7807997942635748]],
        "D": [[-0.025561936667844065, -0.33979846179866224, 0.23722236056501622, 0.39196538403145287], [-0.08532456230171642, 0.5838278875603452, 0.2631775270714224, 0.6121502776216614], [0.1262263234727608, 0.06057662552962867, -0.1507656484074572, 0.00460068980629913], [-0.17365210205867593, -0.5612156967161119, 0.09041265840228518, -0.5851968409115809], [-0.10055419121184817, 0.02046605633735976, 0.22777767127756682, -0.2215988076815098]],
    }  # fmt: skip
    #: ``json.dumps(QUICK.as_dict(), sort_keys=True)``: a ``--quick`` line's ``settings``.
    QUICK_SETTINGS: ClassVar[dict[str, Any]] = json.loads(
        '{"arms": ["A", "B", "C", "D"], "batch_size": 32, "c6": 5420441.132650757, '
        '"circuit_layers": 2, "drop_duplicates": false, "gamma_over_omega": [0.1], '
        '"geometries": ["chain"], "grid_cols": null, "hidden": 4, "inner_folds": 2, "loss": '
        '"bce", "lr": 0.01, "max_epochs": 6, "max_folds": 1, "max_separation_rows": 2000, '
        '"n_atoms": 4, "n_samples": 160, "n_splits": 4, "n_steps": 40, "n_trials": 2, "omega": '
        '12.566370614359172, "omega_t": 3.141592653589793, "patience": null, "random_state": 0, '
        '"search_space": {"lr": [0.01, 0.05]}, "separation_seed": 0, "shots": null, '
        '"standardise": true, "v_over_omega": [1.0], "validation_folds": 4}'
    )

    @pytest.mark.parametrize("name", ["quick", "default", "quick-shots"])
    def test_the_cache_keys_are_those_of_main(self, name: str) -> None:
        settings = {
            "quick": QUICK,
            "default": bench.Settings(),
            "quick-shots": _replace(QUICK, shots=16),
        }[name]
        keys = [
            bench.cache_key("0" * 64, r.config, TestGeometry.KEY_ENV, "c" * 64)
            for r in bench.representations(settings)
            if r.config is not None
        ]
        assert (len(keys), hashlib.sha256("\n".join(keys).encode()).hexdigest()) == self.KEYS[name]

    @pytest.mark.parametrize(("arm", "atol"), [("A", 1e-13), ("B", 1e-13), ("D", 1e-6)])
    def test_the_features_are_those_of_main(self, arm: str, atol: float) -> None:
        """A and B in float64 to rounding; arm D's circuit holds its angles in float32."""
        np.testing.assert_allclose(
            _rep(QUICK, arm).compute(PIN_X), self.FEATURES[arm], rtol=0, atol=atol
        )

    def test_a_line_gains_only_the_new_settings_at_their_defaults(self, quick: Any) -> None:
        """
        Every field of a ``--quick`` line is there as before (``FIELDS``), its seeds and
        configs too (:class:`TestQuick`); ``settings`` gains the fields of #775, at their
        defaults, and keeps every other value.
        """
        defaults = TestRealismSettings.REALISM_DEFAULTS
        for record in quick.lines:
            assert set(record) == FIELDS
            settings = record["settings"]
            assert {k: v for k, v in settings.items() if k not in defaults} == self.QUICK_SETTINGS
            assert {k: settings[k] for k in defaults} == defaults
            assert "jitter_seed" not in record["seeds"]
            assert "device_violations" not in (record["feature_map"] or {})

    #: ``(number, SHA-256)`` of the newline-joined ``shot_stream_seed(config, PIN_X)`` of
    #: every representation with a config, recorded on 26ba88f with the unmodified script:
    #: ``--quick`` with ``shots=16`` and the default grid with ``shots=100``.
    STREAMS: ClassVar[dict[str, tuple[int, str]]] = {
        "quick-shots": (3, "f9bfd9fc051a3a0bb10b7179a63151bd3ddd44e39266762bcc04fa2964c29177"),
        "default-shots": (57, "2396f15ebde8bdd1462abda0084d3914bb14f9dbf8a9e46c088f861bf152a7ae"),
    }

    @pytest.mark.parametrize("name", ["quick-shots", "default-shots"])
    def test_a_square_pulse_records_no_device_violations_and_keeps_its_streams(
        self, name: str
    ) -> None:
        """
        #781's decision: a square pulse records no ``device_violations``, so the shot stream of
        every matrix of a default run with shots is the one of ``main``, seed for seed.
        """
        settings = {"quick-shots": _replace(QUICK, shots=16), "default-shots": bench.Settings(shots=100)}[name]  # fmt: skip
        reps = [r for r in bench.representations(settings) if r.config is not None]
        assert all("device_violations" not in r.config for r in reps)
        seeds = [str(bench.shot_stream_seed(r.config, PIN_X)) for r in reps]
        digest = hashlib.sha256("\n".join(seeds).encode()).hexdigest()
        assert (len(seeds), digest) == self.STREAMS[name]

    @pytest.mark.parametrize("omega", [1.0, OMEGA, 15.8])
    def test_the_square_pulse_as_a_global_drive_never_passes_the_slew_bound(
        self, omega: float
    ) -> None:
        """
        Why a square pulse records no list: it is the global drive with continuous levels and
        ``t_ramp = 0`` (the same detunings, and the same features bit for bit,
        :class:`TestGlobalDrive`), whose slope ``Ω/t_ramp`` is infinite, so at every Ω within
        the other bounds its list is this one entry and never the empty one.  The equivalent
        run, ``--encoding global-drive --levels continuous --t-ramp 0``, records it.
        """
        square = PulseEncoding(4, omega=omega, t=math.pi / omega)
        drive = GlobalDrivePulseEncoding(4, omega=omega, levels=None, t_ramp=0.0, t=square.t)
        X = torch.from_numpy(PIN_X)
        assert torch.equal(drive.detunings(X), square.detunings(X))
        slew = ["omega_slew_max: Ω / t_ramp = inf rad/µs² exceeds 250 rad/µs²"]
        assert drive.device_violations(AQUILA_LIMITS) == slew
        # dataclasses.replace runs Settings' validation, as the command line does.
        equivalent = _replace(
            QUICK, omega=omega, encoding="global-drive", levels="continuous", t_ramp=0.0
        )
        assert _rep(equivalent, "A").config["device_violations"] == slew
        assert "device_violations" not in _rep(QUICK, "A").config


class TestSummaryRealism:
    """
    #775: a row of :func:`summarise` must not pool lines of different device models, as #763
    refuses lines measured differently.
    """

    ADVICE = (
        "Its mean and spread would pool lines whose features come from different models; "
        "summarise each run's lines on their own."
    )
    #: The map config of an ideal line (``get_config()`` keys of #775 at their defaults).
    IDEAL: ClassVar[dict[str, Any]] = {
        "readout_error": None,
        "position_jitter": 0.0,
        "jitter_samples": None,
        "jitter_seed": None,
        "encoding": {"n_features": 4, "omega": OMEGA, "t": 0.25, "delta_max": 21.77},
    }

    def _run(self, **config: Any) -> list[dict[str, Any]]:
        """Two folds of A and B at ``V/Ω = 1`` whose map config is ``IDEAL`` with ``config``."""
        feature_map = {"feature_map": "RydbergFeatureMap", "config": {**self.IDEAL, **config}}
        lines = [
            TestSummary._line(v, arm, 1.0, 1.0, feature_map=feature_map)
            for arm, v in (("A", 1.0), ("B", 1.0))
            for _ in range(2)
        ]
        return lines + [TestSummary._line(None, "C", 0.0, 1.0, feature_map=None)] * 2

    def _refusal(self, records: list[dict[str, Any]]) -> str:
        with pytest.raises(ValueError) as caught:
            bench.summarise(records)
        return str(caught.value)

    def test_a_detection_error_is_refused_beside_lines_without_it(self) -> None:
        ideal, noisy = self._run(), self._run(readout_error=[0.01, 0.08])
        assert self._refusal(ideal + noisy) == (
            "dataset 'positive-control' has lines of 2 device models (readout_error, "
            "position_jitter, jitter_samples, encoding) in row A/linear chain V/Ω=1 γ/Ω=0.1: "
            "readout_error none, position_jitter 0, jitter_samples none, encoding square (2 "
            "lines); readout_error [0.01, 0.08], position_jitter 0, jitter_samples none, "
            "encoding square (2 lines) (rows with more than one: 2 of 3).  " + self.ADVICE
        )
        assert "readout_error [0.01, 0.08], position_jitter 0" in self._refusal(
            noisy + self._run(readout_error=[0.02, 0.08])
        )

    def test_jitter_is_refused_and_so_is_another_number_of_registers(self) -> None:
        jittered = self._run(position_jitter=0.2, jitter_samples=8, jitter_seed=5)
        assert (
            "readout_error none, position_jitter 0, jitter_samples none, encoding square (2 "
            "lines); readout_error none, position_jitter 0.2, jitter_samples 8, encoding "
            "square (2 lines) (rows with more than one: 2 of 3)."
        ) in self._refusal(self._run() + jittered)
        assert "position_jitter 0.2, jitter_samples 16, encoding square (2 lines)" in (
            self._refusal(jittered + self._run(position_jitter=0.2, jitter_samples=16))
        )
        # Another seed of one model (another --seed) is the same model, as for the data.
        reseeded = self._run(position_jitter=0.2, jitter_samples=8, jitter_seed=6)
        assert len(bench.summarise(jittered + reseeded).splitlines()) == 4

    def test_another_encoding_is_refused(self) -> None:
        drive = {"n_features": 4, "omega": OMEGA, "t": 0.25, "delta_local": 21.77}
        ramped = self._run(
            encoding={**drive, "levels": 8, "t_ramp": 0.0503, "n_ramp_segments": 32}
        )
        assert (
            "encoding square (2 lines); readout_error none, position_jitter 0, jitter_samples "
            "none, encoding global-drive levels 8 t_ramp 0.0503 n_ramp_segments 32 (2 lines)"
        ) in self._refusal(self._run() + ramped)
        continuous = self._run(
            encoding={**drive, "levels": None, "t_ramp": 0.0, "n_ramp_segments": None}
        )
        assert (
            "encoding global-drive levels 8 t_ramp 0.0503 n_ramp_segments 32 (2 lines); "
            "readout_error none, position_jitter 0, jitter_samples none, encoding global-drive "
            "levels continuous t_ramp 0 n_ramp_segments none (2 lines)"
        ) in self._refusal(ramped + continuous)

    def test_lines_from_before_the_options_are_the_ideal_model(self) -> None:
        """
        A line without the keys was computed without the options (they did not exist), so it
        joins an ideal line of today and prints what the lines print without any config;
        beside a line with the detection error it is refused as the ideal one would be.
        """
        old = [
            {**r, "feature_map": {"config": {"encoding": self.IDEAL["encoding"]}}}
            if r["feature_map"] is not None
            else r
            for r in self._run()
        ]
        bare = [{k: v for k, v in r.items() if k != "feature_map"} for r in self._run()]
        assert bench.summarise(old + self._run()) == bench.summarise(bare + bare)
        assert "readout_error none, position_jitter 0, jitter_samples none, encoding square" in (
            self._refusal(old + self._run(readout_error=[0.01, 0.08]))
        )
        assert bench.feature_realism(bare[0]) == (None, 0.0, None, None)

    def test_arm_d_records_its_detection_error(self) -> None:
        circuit = {"feature_map": "QuantumEncodingLayer", "config": {"n_qubits": 4}}
        noisy = {**circuit, "config": {"n_qubits": 4, "readout_error": [0.01, 0.08]}}
        lines = [TestSummary._line(None, "D", 1.0, 1.0, feature_map=circuit)] * 2
        read = [TestSummary._line(None, "D", 1.0, 1.0, feature_map=noisy)] * 2
        assert (
            "in row D/linear: readout_error none, position_jitter 0, jitter_samples none, "
            "encoding none (2 lines); readout_error [0.01, 0.08], position_jitter 0, "
            "jitter_samples none, encoding none (2 lines) (rows with more than one: 1 of 1)."
        ) in self._refusal(lines + read)

    def test_one_run_of_the_preset_with_every_arm_summarises(self) -> None:
        """
        The representations of a preset run give every A, B and D line one model, and C none:
        the check is per row, and one run never fails it.
        """
        settings = _replace(
            QUICK,
            **bench.device_preset("aquila", OMEGA),
            device="aquila",
            levels=8,
            jitter_samples=2,
        )
        models = {
            r.arm: bench.feature_realism({"feature_map": r.config})
            for r in bench.representations(settings)
        }
        ramp = ("global-drive", 8, 0.0503, bench.AQUILA_RAMP_SEGMENTS)
        assert models == {
            "A": ((0.01, 0.08), 0.2, 2, ramp),
            "B": ((0.01, 0.08), 0.2, 2, ramp),
            "C": (None, 0.0, None, None),
            "D": ((0.01, 0.08), 0.0, None, None),
            "D-hybrid": (None, 0.0, None, None),
        }
