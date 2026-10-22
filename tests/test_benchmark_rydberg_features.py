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
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import torch
import torch.nn as nn

import hqnn_forge
from hqnn_forge.benchmark import _sample_configs, fingerprint
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
from hqnn_forge.rydberg import DEFAULT_C6, AtomRegister, PulseEncoding, RydbergFeatureMap
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

    def test_shots_are_drawn_with_the_recorded_seed(self) -> None:
        settings = bench.Settings(v_over_omega=(1.0,), gamma_over_omega=(0.0,), shots=64)
        features = self._features(settings, "A", 1.0, 0.0)
        shot_seed = bench.run_seeds(settings.random_state)["shot_seed"]
        by_hand = _map_by_hand(4, 1.0, 0.0, 1).transform(
            self.X, shots=64, generator=torch.Generator().manual_seed(shot_seed)
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
        With shots, a matrix is drawn from the start of the ``shot_seed``
        stream whatever was sampled before it: the cache key (inputs and
        config) then determines the features, also when a run is resumed.
        """
        settings = bench.Settings(n_atoms=3, circuit_layers=2, shots=64)
        (rep,) = [
            r for r in bench.representations(_replace(settings, arms=("D",))) if r.arm == "D"
        ]
        seeds = bench.run_seeds(settings.random_state)
        assert rep.config["shots"] == 64
        assert rep.config["shot_seed"] == seeds["shot_seed"]
        X = np.random.default_rng(6).uniform(-math.pi, math.pi, size=(7, 3))
        first = rep.compute(X)
        rep.compute(X[:3])  # another matrix in between moves no stream
        np.testing.assert_array_equal(rep.compute(X), first)
        # by hand: a circuit built here, evaluated once
        fresh = bench.frozen_circuit(settings, seeds["circuit_seed"])
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

    def test_candidates_are_drawn_as_the_benchmark_tuning_draws_them(self) -> None:
        space = {"lr": (0.003, 0.01, 0.03), "batch_size": (32, 128)}
        for seed in (0, 1, 99):
            assert bench.sample_configs(space, 4, seed) == _sample_configs(space, 4, seed)
            assert len(bench.sample_configs(space, 4, seed)) == 4
        assert len(bench.sample_configs(space, 10, 0)) == 6  # the whole grid, no more


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


def _write_churn(directory: Path, n: int = 96, n_pos: int = 24) -> Path:
    """A small file in the format of UCI's ``Customer Churn.csv``; the dataset is not fetched."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 500, (n, 13)).astype(float)
    labels = np.zeros(n, dtype=int)
    labels[:n_pos] = 1
    data[labels == 1, :3] += 150.0  # churners differ, so there is something to learn
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
        # ``StandardScaler`` has the constant rule of ``standardise_features``
        # (#580): a variance within its rounding bound is only centred.
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
        (``(n ε μ)² ≈ 2.5e-27`` for 150 rows).
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
        before = bench.standardise_features(features, fit_rows)
        # Synthetic, validation and test rows: none of them is a fitting row.
        outside = np.setdiff1d(np.arange(features.shape[0]), fit_rows)
        assert outside[0] == split.n_train - split.n_synthetic
        changed = features.copy()
        changed[outside] = 0.5 + 7.0 * (changed[outside] - 0.5)
        after = bench.standardise_features(changed, fit_rows)
        np.testing.assert_array_equal(after[fit_rows], before[fit_rows])
        mean, std = features[fit_rows].mean(axis=0), features[fit_rows].std(axis=0)
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

    def test_a_column_constant_up_to_rounding_is_only_centred(self) -> None:
        """
        ½ and ½ ± one unit in the last place (2⁻⁵³): a computed variance of
        about 8e-33, below the rounding bound ``(n ε |μ|)² ≈ 2.8e-28`` for
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
        is standardised too; ``D-hybrid`` never is.  :func:`preprocess`
        standardises the raw columns with the same function (step 2, #580):
        its calls are told apart by a flag it sets, and fit on the outer
        fold's training part and on each tuning fold's fitting rows.
        """
        calls: list[tuple[int, np.ndarray]] = []
        raw_calls: list[np.ndarray] = []
        in_preprocess = False
        original = bench.standardise_features
        original_preprocess = bench.preprocess

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
        monkeypatch.setattr(bench, "standardise_features", spy)
        monkeypatch.setattr(bench, "preprocess", preprocess_spy)
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
        expected = _sample_configs({"lr": (0.01, 0.03, 0.1, 0.3)}, 3, fold.seeds["tune_seed"])
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
        candidates = _sample_configs(space, settings.n_trials, fold.seeds["tune_seed"])
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
    "n_samples",
    "n_positives",
    "n_raw_features",
    "arm",
    "head",
    "representation",
    "feature_map_trained",
    "v_over_omega",
    "gamma_over_omega",
    "spacing_um",
    "omega",
    "omega_t",
    "evolution_time_us",
    "n_steps",
    "feature_map",
    "reused_from_v_over_omega",
    "measurement",
    "shots",
    "fold",
    "n_splits",
    "train_idx",
    "val_idx",
    "test_idx",
    "n_synthetic",
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
        for line, record in zip(lines, quick.lines, strict=True):
            assert set(_strict_loads(line)) == FIELDS == set(record)

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
        # The script's features of the fold's rows, drawn by hand with that seed.
        stacked = np.concatenate(
            [fold.train_inputs, fold.inputs[fold.val_idx], fold.inputs[fold.test_idx]]
        )
        sampled = _map_by_hand(4, 1.0, 0.1, 40).transform(
            stacked, shots=32, generator=torch.Generator().manual_seed(shot_seed)
        )
        test = sampled[-fold.test_idx.size :].numpy()
        expected = separation_measures(test, y[fold.test_idx]).to_dict()
        assert read[0]["separation"]["test"] == pytest.approx(expected, rel=1e-9)
        exact = _map_by_hand(4, 1.0, 0.1, 40).transform(fold.inputs[fold.test_idx]).numpy()
        assert np.abs(test - exact).max() > 0.01  # sampled, not exact


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
            if line.split()[1] == v and line.split()[3] == arm
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

        # Capture model training inputs to verify full dataset was used
        train_row_counts: list[int] = []
        orig_train_model = bench.train_model

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

        monkeypatch.setattr(bench, "train_model", spy_train_model)

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
