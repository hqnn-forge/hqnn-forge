"""Numerical and pairing contracts for the frozen robustness study additions."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from sklearn.metrics import matthews_corrcoef as sklearn_mcc

from hqnn_forge import benchmark
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import Channel


def _hybrid(width: int) -> HybridBinaryClassifier:
    return HybridBinaryClassifier(width, 2, 1, init_strategy="normal", device_name="default.qubit")


def test_inner_pca_matches_independent_covariance_eigenvectors() -> None:
    X = np.array(
        [[1.0, 8.0, 2.0], [2.0, 6.0, 4.0], [4.0, 3.0, 3.0], [8.0, 1.0, 9.0], [90.0, -20.0, 50.0]]
    )
    train = np.arange(4)
    got, parameters = benchmark._pca_standardise(X, train, 2)
    Z = (X - X[train].mean(0)) / (X[train].std(0) + 1e-12)
    _, eigenvectors = np.linalg.eigh(Z[train].T @ Z[train])
    expected = Z @ eigenvectors[:, ::-1][:, :2]
    signs = np.sign((got[train] * expected[train]).sum(0))
    np.testing.assert_allclose(got * signs, expected, atol=1e-12)
    changed = X.copy()
    changed[-1] *= 100
    _, other = benchmark._pca_standardise(changed, train, 2)
    assert parameters == other


def test_noisy_threshold_uses_validation_not_test(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext

    model = _hybrid(1)
    # The noisy validation threshold is .25, clean is .8, and illegal test
    # optimisation would use .15 and obtain a perfect MCC.
    values = [torch.tensor([0.1, 0.3, 0.2, 0.4]), torch.tensor([0.1, 0.2, 0.24, 0.35])]
    monkeypatch.setattr(benchmark, "apply_depolarizing_noise", lambda *a, **k: nullcontext())
    monkeypatch.setattr(model, "predict_proba", lambda x: values.pop(0))
    details: list[dict[str, Any]] = []
    fixed = benchmark._noise_scores(
        model,
        np.zeros((4, 1)),
        np.array([0, 1, 1, 1]),
        0.8,
        [0.2],
        "all",
        channel="amplitude_damping",
        X_val=np.zeros((4, 1)),
        y_val=np.array([0, 1, 0, 1]),
        retune=True,
        results=details,
    )
    assert fixed[0.2] == 0
    assert details[0]["retuned_threshold"] == pytest.approx(0.25)
    assert details[0]["retuned_mcc"] == pytest.approx(1 / 3)
    assert details[0]["retuned_mcc"] == sklearn_mcc([0, 1, 1, 1], [0, 0, 0, 1])
    assert details[0]["fixed_confusion"] == {"tn": 1, "fp": 0, "fn": 3, "tp": 0}
    assert details[0]["retuned_confusion"] == {"tn": 1, "fp": 0, "fn": 2, "tp": 1}


@pytest.mark.parametrize("failed", [False, True])
def test_extension_restarts_original_state_once_before_test(
    monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    from hqnn_forge.training import TrainingHistory
    from hqnn_forge.training.trainer import EpochRecord

    model = _hybrid(1)
    starts, optimizers, options, order = [], [], [], []

    def train(m: Any, loss: Any, optimizer: Any, *args: Any, **kwargs: Any) -> Any:
        order.append("train")
        starts.append({key: value.clone() for key, value in m.state_dict().items()})
        optimizers.append(optimizer)
        assert not optimizer.state
        options.append(kwargs)
        with torch.no_grad():
            next(m.parameters()).add_(10.0)
        return TrainingHistory(
            monitor="mcc",
            epochs=[EpochRecord(1, 0.123, 0.234, 0.2, 0.5)],
            best_epoch=1,
            best_threshold=0.5,
        )

    monkeypatch.setattr(benchmark, "train_model", train)
    diagnostics = {
        "finite": True,
        "threshold": 0.5,
        "mcc": 0.01 if failed else 0.4,
        "pr_auc": 0.1 if failed else 0.9,
        "prevalence": 0.25,
    }
    monkeypatch.setattr(
        benchmark,
        "_training_validation",
        lambda *a: {
            **diagnostics,
            "mcc": -0.2 if failed and len(starts) > 1 else diagnostics["mcc"],
        },
    )

    def predict(X: torch.Tensor) -> torch.Tensor:
        if float(X[0, 0]) == 9.0:
            order.append("test")
        return torch.tensor([0.1, 0.9, 0.2, 0.8])

    monkeypatch.setattr(model, "predict_proba", predict)
    details: dict[str, Any] = {}
    score = benchmark._fit_and_score(
        model,
        torch.nn.BCEWithLogitsLoss,
        np.zeros((4, 1)),
        np.array([0, 1, 0, 1]),
        np.ones((4, 1)),
        np.array([0, 1, 0, 1]),
        np.full((4, 1), 9.0),
        np.array([0, 1, 0, 1]),
        lr=0.05,
        max_epochs=30,
        batch_size=2,
        patience=10,
        batch_seed=23,
        validation_extension=benchmark.ValidationExtension(),
        details=details,
    )
    assert order == (["train", "train", "test"] if failed else ["train", "test"])
    assert score.mcc == 1.0
    assert len(starts) == 1 + failed
    assert details["selected_attempt"] == int(failed)
    assert len(details["attempts"]) == 1 + failed
    assert score.seconds == sum(a["seconds"] for a in details["attempts"])
    assert all(a["history"]["epochs"][0]["train_loss"] == 0.123 for a in details["attempts"])
    if failed:
        for key in starts[0]:
            torch.testing.assert_close(starts[0][key], starts[1][key], rtol=0, atol=0)
        assert optimizers[0] is not optimizers[1]
        assert options[1]["max_epochs"] == 60 and options[1]["patience"] is None
        assert torch.equal(
            options[0]["generator"].get_state(), options[1]["generator"].get_state()
        )
        assert details["extension_reasons"] == ["validation_mcc", "validation_pr_auc"]
        assert details["attempts"][1]["training_noise_validation"]["mcc"] == -0.2


@pytest.mark.parametrize(
    "diagnostics, reasons",
    [
        ({"finite": False}, ["nonfinite_validation"]),
        ({"finite": True, "mcc": 0.1, "pr_auc": 0.2, "prevalence": 0.1}, []),
        ({"finite": True, "mcc": 0.09, "pr_auc": 0.2, "prevalence": 0.1}, ["validation_mcc"]),
    ],
)
def test_extension_guard_is_validation_only(
    diagnostics: dict[str, Any], reasons: list[str]
) -> None:
    assert benchmark._extension_reasons(diagnostics, benchmark.ValidationExtension()) == reasons


@pytest.mark.slow
def test_shards_replay_and_callback_rng_isolation(tmp_path: Any) -> None:
    import random

    from hqnn_forge.experiment import load_record, rerun_benchmark

    rng = np.random.default_rng(4)
    data = {"tiny": (rng.normal(size=(60, 3)), np.tile([0, 0, 1], 20))}
    settings: dict[str, Any] = dict(
        n_splits=3,
        validation_folds=2,
        n_seeds=2,
        max_epochs=1,
        batch_size=16,
        oversample=False,
        pca_components=2,
        noise_channel="amplitude_damping",
        noise_levels=[0.0, 0.2],
        retune_noise_threshold=True,
        noise_batch_size=8,
    )
    complete = benchmark.run_benchmark(data, _hybrid, **settings)

    def observer(*args: Any) -> None:
        torch.rand(100)
        np.random.random(100)
        random.random()

    shard = benchmark.run_benchmark(
        data,
        _hybrid,
        fold_indices=[2, 0],
        on_fold_end=observer,
        record_path=tmp_path / "record.json",
        **settings,
    )
    for got, want in zip(
        shard.folds, [f for f in complete.folds if f.fold in (0, 2)], strict=True
    ):
        assert (got.fold, got.init_seed, got.batch_seed, got.mcc, got.threshold) == (
            want.fold,
            want.init_seed,
            want.batch_seed,
            want.mcc,
            want.threshold,
        )
        assert got.noise_results == want.noise_results
        assert got.diagnostics["preprocessing"] == want.diagnostics["preprocessing"]
    assert all(r["n_folds"] == 2 for r in shard.records)
    record, _ = load_record(tmp_path / "record.json")
    replay = rerun_benchmark(record, data)
    assert [f.noise_results for f in replay.folds] == [f.noise_results for f in shard.folds]
    assert replay.settings["fold_indices"] == [0, 2]


@pytest.mark.parametrize("channel", ["depolarizing", "amplitude_damping", "phase_damping"])
def test_end_channel_closed_form_and_chunk_equivalence(channel: str) -> None:
    from hqnn_forge.noise import apply_depolarizing_noise

    torch.manual_seed(12)
    model = _hybrid(2)
    X = np.array([[0.1, 0.2], [0.7, -0.3], [-0.2, 0.9], [0.3, 0.4]])
    x = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        encoded = x
        clean = model.quantum_layer(encoded)
        with apply_depolarizing_noise(
            model, 0.2, position="end", channel=cast("Channel", channel)
        ):
            noisy = model.quantum_layer(encoded)
            chunked = benchmark._predict(model, X, 2)
            whole = benchmark._predict(model, X, None)
    expected = {
        "depolarizing": (1 - 4 * 0.2 / 3) * clean,
        "amplitude_damping": 0.8 * clean + 0.2,
        "phase_damping": clean,
    }[channel]
    torch.testing.assert_close(noisy, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(chunked, whole, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    "option",
    [
        dict(pca_components=0),
        dict(pca_components=True),
        dict(noise_batch_size=0),
        dict(fold_indices=[]),
        dict(fold_indices=[1, 1]),
        dict(fold_indices=[3]),
        dict(noise_channel="unknown"),
        dict(noise_levels=[1.1], noise_channel="amplitude_damping"),
    ],
)
def test_new_options_validate_before_fit(option: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        benchmark.run_benchmark(
            {"a": (np.zeros((12, 2)), np.tile([0, 1], 6))}, _hybrid, n_splits=3, **option
        )


def test_nonfinite_noisy_predictions_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext

    model = _hybrid(1)
    monkeypatch.setattr(benchmark, "apply_depolarizing_noise", lambda *a, **k: nullcontext())
    monkeypatch.setattr(model, "predict_proba", lambda X: torch.tensor([float("nan"), 0.2]))
    rows: list[dict[str, Any]] = []
    result = benchmark._noise_scores(
        model,
        np.zeros((2, 1)),
        np.array([0, 1]),
        0.5,
        [0.2],
        "all",
        retune=True,
        X_val=np.zeros((2, 1)),
        y_val=np.array([0, 1]),
        results=rows,
    )
    # Legacy field unchanged from main: NaN scores as negative, so predictions
    # [0, 0] against labels [0, 1] give the all-negative MCC of 0.
    assert result[0.2] == 0.0
    assert rows[0]["fixed_mcc"] is None and rows[0]["retuned_mcc"] is None
    assert rows[0]["retuned_threshold"] is None and rows[0]["finite"] is False


@pytest.mark.parametrize("options", [dict(shots=10), dict(noise_method="trajectories")])
def test_extension_rejects_unvalidated_estimators_before_fit(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    model = _hybrid(1)
    real_config = model.get_config()
    monkeypatch.setattr(model, "get_config", lambda: {**real_config, **options})
    monkeypatch.setattr(benchmark, "train_model", lambda *a, **k: pytest.fail("unvalidated fit"))
    with pytest.raises(ValueError, match="analytic density/noiseless"):
        benchmark._fit_and_score(
            model,
            torch.nn.BCEWithLogitsLoss,
            np.zeros((4, 1)),
            np.array([0, 1, 0, 1]),
            np.zeros((4, 1)),
            np.array([0, 1, 0, 1]),
            np.zeros((4, 1)),
            np.array([0, 1, 0, 1]),
            lr=0.05,
            max_epochs=1,
            batch_size=2,
            patience=1,
            batch_seed=1,
            validation_extension=benchmark.ValidationExtension(),
        )


@pytest.mark.slow
def test_old_record_replay_defaults_and_inner_pca_no_validation_leakage(tmp_path: Any) -> None:
    from hqnn_forge.experiment import rerun_benchmark, to_record

    X = np.random.default_rng(7).normal(size=(60, 3))
    y = np.tile([0, 0, 1], 20)
    defaults: dict[str, Any] = dict(
        n_splits=3,
        validation_folds=2,
        max_epochs=1,
        batch_size=16,
        oversample=False,
    )
    result = benchmark.run_benchmark({"a": (X, y)}, _hybrid, **defaults)
    record = to_record(result)
    # Old format-1 records have no additive settings. Keep all three folds
    # selected to make omission of fold_indices reproduce its old semantics.
    old_settings = [
        "pca_components",
        "noise_channel",
        "retune_noise_threshold",
        "noise_batch_size",
        "validation_extension",
        "fold_indices",
    ]
    for option in old_settings:
        del record["config"]["settings"][option]
    replay = rerun_benchmark(record, {"a": (X, y)})
    assert [f.mcc for f in replay.folds] == [f.mcc for f in result.folds]
    assert [f.threshold for f in replay.folds] == [f.threshold for f in result.folds]
    pca = benchmark.run_benchmark({"a": (X, y)}, _hybrid, pca_components=2, **defaults)
    changed = X.copy()
    changed[np.concatenate([pca.folds[0].test_idx, pca.folds[0].val_idx])] *= 100
    altered = benchmark.run_benchmark({"a": (changed, y)}, _hybrid, pca_components=2, **defaults)
    assert (
        pca.folds[0].diagnostics["preprocessing"] == altered.folds[0].diagnostics["preprocessing"]
    )
    assert (
        pca.folds[0].diagnostics["selected_state_sha256"]
        == altered.folds[0].diagnostics["selected_state_sha256"]
    )


def test_nonfinite_attempt_serialization_is_additive(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from hqnn_forge.experiment import _plain, to_record

    def fit(*args: Any, **kwargs: Any) -> Any:
        kwargs["details"].update(
            attempts=[{"loss": float("inf"), "nested": [-float("inf"), float("nan")]}]
        )
        return benchmark.FitScore(0.0, 0.5, 0.0, 0, 0.0, 0.0)

    monkeypatch.setattr(benchmark, "_fit_and_score", fit)
    result = benchmark.run_benchmark(
        {"a": (np.arange(60).reshape(20, 3), np.tile([0, 1], 10))},
        _hybrid,
        n_splits=2,
        validation_folds=2,
        oversample=False,
    )
    record = to_record(result)
    assert record["folds"][0]["diagnostics"]["attempts"] == [
        {"loss": None, "nested": [None, None]}
    ]
    json.dumps(record, allow_nan=False)
    assert _plain(float("inf")) == float("inf")
    with pytest.raises(ValueError):
        json.dumps(_plain({"legacy_configuration": float("inf")}), allow_nan=False)


@pytest.mark.parametrize("channel", ["depolarizing", "amplitude_damping", "phase_damping"])
def test_benchmark_scoring_forwards_actual_channel_and_mismatch(channel: str) -> None:
    from hqnn_forge.noise import apply_depolarizing_noise

    torch.manual_seed(18)
    model = HybridBinaryClassifier(
        2,
        2,
        1,
        init_strategy="normal",
        device_name="default.qubit",
        noise_level=0.2,
        noise_channel=cast("Channel", channel),
    )
    X = np.array([[0.1, 0.2], [0.7, -0.3], [-0.2, 0.9], [0.3, 0.4]])
    y = np.array([0, 1, 0, 1])
    with apply_depolarizing_noise(model, 0.01, position="end", channel=cast("Channel", channel)):
        expected = (
            model.predict_proba(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)
        )
    rows: list[dict[str, Any]] = []
    benchmark._noise_scores(
        model,
        X,
        y,
        0.5,
        [0.01],
        "end",
        channel=cast("Channel", channel),
        retune=True,
        X_val=X,
        y_val=y,
        batch_size=2,
        results=rows,
    )
    assert rows[0]["brier"] == pytest.approx(np.mean((expected - y) ** 2), abs=1e-10)
    assert rows[0]["fixed_mcc"] == sklearn_mcc(y, expected >= 0.5)
    assert rows[0]["channel"] == channel and rows[0]["probability"] == 0.01
