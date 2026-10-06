"""
tests/test_experiment.py
========================
Experiment records (#200): what ``run_benchmark(..., record_path=...)`` writes,
that it reads back, that environment differences are reported, and that a
re-run from a record on the same synthetic data reproduces the recorded
per-fold scores exactly on CPU.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch.nn as nn

import hqnn_forge
from hqnn_forge import benchmark
from hqnn_forge.benchmark import BenchmarkResult, fingerprint, run_benchmark
from hqnn_forge.experiment import (
    RECORD_FORMAT_VERSION,
    RECORDED_PACKAGES,
    compare_environment,
    environment,
    load_record,
    rerun_benchmark,
    save_record,
)
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.preprocessing import stratified_kfold

SETTINGS: dict[str, Any] = dict(
    n_splits=3, max_epochs=2, batch_size=32, smote_kwargs={"k_neighbors": 3}, random_state=4
)


def _hybrid(n_input_features: int) -> nn.Module:
    return HybridBinaryClassifier(
        n_input_features,
        2,
        1,
        device_name="default.qubit",
        diff_method="backprop",
        init_strategy="normal",
    )


def _data(seed: int = 0, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 1.0).astype(int)
    return X, y


def _datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {"first": _data(0), "second": _data(1, n=90)}


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> tuple[BenchmarkResult, Path]:
    path = tmp_path_factory.mktemp("record") / "run.json"
    result = run_benchmark(_datasets(), _hybrid, record_path=path, **SETTINGS)
    return result, path


def _no_seconds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "train_seconds"} for r in rows]


class TestContents:
    def test_strict_json_with_every_section(self, recorded: tuple) -> None:
        _, path = recorded

        def refuse(constant: str) -> None:
            raise AssertionError(f"non-standard JSON constant {constant}")

        record = json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)
        assert set(record) == {
            "format_version",
            "created",
            "environment",
            "config",
            "seeds",
            "datasets",
            "folds",
            "metrics",
            "noise",
            "noise_summary",
        }
        assert record["format_version"] == RECORD_FORMAT_VERSION

    def test_config_seeds_and_folds(self, recorded: tuple) -> None:
        result, path = recorded
        record, _ = load_record(path)
        settings = record["config"]["settings"]
        assert settings["n_splits"] == 3 and settings["random_state"] == 4
        assert settings["loss"] == "hqnn_forge.utils.imbalance.FocalLoss"
        hybrid = record["config"]["models"]["first"]["hybrid"]
        assert hybrid["class"] == "HybridBinaryClassifier"
        assert hybrid["config"]["n_input_features"] == 3
        assert record["config"]["models"]["first"]["control"]["class"] == "ClassicalBaseline"

        assert len(record["folds"]) == len(result.folds) == 2 * 3 * 2
        for saved, fold in zip(record["folds"], result.folds):
            assert saved["train_idx"] == fold.train_idx.tolist()
            assert saved["val_idx"] == fold.val_idx.tolist()
            assert saved["test_idx"] == fold.test_idx.tolist()
            assert (saved["init_seed"], saved["batch_seed"]) == (fold.init_seed, fold.batch_seed)
            assert saved["mcc"] == fold.mcc and saved["threshold"] == fold.threshold
        seeds = record["seeds"]["folds"]
        for name in ("split_seed", "inner_seed", "smote_seed", "init_seed", "batch_seed"):
            assert [s[name] for s in seeds] == [getattr(f, name) for f in result.folds]
            assert [s[name] for s in seeds] == [s[name] for s in record["folds"]]

    def test_recorded_split_seeds_regenerate_the_recorded_folds(self, recorded: tuple) -> None:
        # The split seeds alone, with the recorded settings, give back every
        # fold's test, train and validation rows.
        _, path = recorded
        record, _ = load_record(path)
        settings = record["config"]["settings"]
        for fold in record["folds"]:
            _, y = _datasets()[fold["dataset"]]
            outer = stratified_kfold(y, settings["n_splits"], random_state=fold["split_seed"])
            train_part, test_idx = outer[fold["fold"]]
            assert sorted(test_idx.tolist()) == fold["test_idx"]
            inner_tr, inner_va = stratified_kfold(
                y[train_part], settings["validation_folds"], random_state=fold["inner_seed"]
            )[0]
            assert sorted(train_part[inner_tr].tolist()) == fold["train_idx"]
            assert sorted(train_part[inner_va].tolist()) == fold["val_idx"]
        split_seeds = {(f["dataset"], f["split_seed"]) for f in record["folds"]}
        assert len(split_seeds) == 2  # one outer split per dataset

    def test_recorded_smote_seed_is_the_one_drawn_with(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No rows of the SMOTE draw are recorded, so check the seed at the call.
        drawn: list[tuple[list[int], int]] = []
        real = benchmark.oversample_fold

        def spy(X: Any, y: Any, train_idx: Any, val_idx: Any, **kwargs: Any) -> Any:
            drawn.append((sorted(train_idx.tolist()), kwargs["random_state"]))
            return real(X, y, train_idx, val_idx, **kwargs)

        monkeypatch.setattr(benchmark, "oversample_fold", spy)
        monkeypatch.setattr(
            benchmark,
            "_fit_and_score",
            lambda *a, **k: benchmark.FitScore(0.3, 0.5, 0.01, 1, 0.2, 0.1),
        )
        path = tmp_path / "run.json"
        run_benchmark(_datasets(), _hybrid, record_path=path, **SETTINGS)
        record, _ = load_record(path)
        hybrid_folds = [f for f in record["folds"] if f["model"] == "hybrid"]
        assert drawn == [(f["train_idx"], f["smote_seed"]) for f in hybrid_folds]
        assert len({seed for _, seed in drawn}) == len(drawn)

    def test_devices_actually_used(self, recorded: tuple) -> None:
        _, path = recorded
        record, _ = load_record(path)
        devices = {(f["model"], f["device"]) for f in record["folds"]}
        assert devices == {("hybrid", "default.qubit"), ("control", None)}

    def test_environment_and_datasets(self, recorded: tuple) -> None:
        _, path = recorded
        record, differences = load_record(path)
        assert differences == {}
        env = record["environment"]
        assert env["hqnn_forge"] == hqnn_forge.__version__
        assert env["packages"]["torch"] == importlib.metadata.version("torch")
        assert set(env["packages"]) == set(RECORDED_PACKAGES)
        X, y = _data(0)
        assert record["datasets"]["first"] == {
            "n_samples": 120,
            "n_features": 3,
            "n_positives": int(y.sum()),
            "sha256": fingerprint(X, y),
        }

    def test_metrics_are_the_result_rows(self, recorded: tuple) -> None:
        result, path = recorded
        record, _ = load_record(path)
        rows = [{**r, "fold_mcc": tuple(r["fold_mcc"])} for r in record["metrics"]]
        assert rows == result.records


class TestRerun:
    def test_reproduces_the_recorded_scores_exactly(self, recorded: tuple) -> None:
        _, path = recorded
        record, _ = load_record(path)
        again = rerun_benchmark(record, _datasets())
        assert [f.mcc for f in again.folds] == [f["mcc"] for f in record["folds"]]
        assert [f.threshold for f in again.folds] == [f["threshold"] for f in record["folds"]]
        assert [f.epochs for f in again.folds] == [f["epochs"] for f in record["folds"]]
        recorded_rows = [{**r, "fold_mcc": tuple(r["fold_mcc"])} for r in record["metrics"]]
        assert _no_seconds(again.records) == _no_seconds(recorded_rows)

    def test_changed_data_is_refused(self, recorded: tuple) -> None:
        _, path = recorded
        record, _ = load_record(path)
        data = _datasets()
        X, y = data["second"]
        X = X.copy()
        X[-1, -1] += 1e-9  # the last value, so a partial hash would miss it
        data["second"] = (X, y)
        with pytest.raises(ValueError, match="'second' differs from the recorded one"):
            rerun_benchmark(record, data)

    @pytest.mark.parametrize(
        "names", [["first"], ["second", "first"], ["first", "second", "third"]]
    )
    def test_datasets_must_match_by_name_and_order(self, recorded: tuple, names: list) -> None:
        _, path = recorded
        record, _ = load_record(path)
        data = {n: _data(i) for i, n in enumerate(names)}
        with pytest.raises(ValueError, match="in that order"):
            rerun_benchmark(record, data)

    def test_an_unimportable_loss_must_be_passed(self, tmp_path: Path) -> None:
        def local_loss() -> nn.Module:
            return nn.BCEWithLogitsLoss()

        path = tmp_path / "run.json"
        data = {"first": _data(0)}
        result = run_benchmark(data, _hybrid, loss=local_loss, record_path=path, **SETTINGS)
        record, _ = load_record(path)
        assert record["config"]["settings"]["loss"].endswith("<locals>.local_loss")
        with pytest.raises(ValueError, match="cannot be imported; pass loss="):
            rerun_benchmark(record, data)
        again = rerun_benchmark(record, data, loss=local_loss)
        assert [f.mcc for f in again.folds] == [f.mcc for f in result.folds]


class TestReading:
    def test_environment_differences_are_reported(self, recorded: tuple) -> None:
        _, path = recorded
        record, _ = load_record(path)
        env = record["environment"]
        env["python"] = "3.9.0"
        env["packages"]["torch"] = "1.0.0"
        env["packages"]["pennylane-lightning"] = None
        diff = compare_environment(env)
        current = environment()
        assert diff["python"] == ("3.9.0", current["python"])
        assert diff["torch"] == ("1.0.0", current["packages"]["torch"])
        assert diff["pennylane-lightning"] == (None, current["packages"]["pennylane-lightning"])
        assert "platform" not in diff and "numpy" not in diff

    def test_other_format_version_is_refused(self, recorded: tuple, tmp_path: Path) -> None:
        _, path = recorded
        record = json.loads(path.read_text(encoding="utf-8"))
        record["format_version"] = RECORD_FORMAT_VERSION + 1
        other = tmp_path / "future.json"
        other.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="record format 2"):
            load_record(other)

    def test_not_a_record(self, tmp_path: Path) -> None:
        path = tmp_path / "x.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="not an experiment record"):
            load_record(path)


class TestWriting:
    def test_undefined_p_value_is_written_as_null(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            benchmark,
            "_fit_and_score",
            lambda *a, **k: benchmark.FitScore(0.3, 0.5, 0.01, 1, 0.2, 0.1),
        )
        path = tmp_path / "tied.json"
        run_benchmark({"first": _data(0)}, _hybrid, record_path=path, **SETTINGS)
        record, _ = load_record(path)
        assert record["metrics"][0]["wilcoxon_p"] is None

    def test_unrecordable_config_is_refused(self, recorded: tuple, tmp_path: Path) -> None:
        result, _ = recorded
        models = {"first": {"hybrid": {"class": "X", "config": {"encoder": nn.Linear(2, 2)}}}}
        broken = BenchmarkResult(
            result.records, result.folds, result.settings, models, result.datasets
        )
        path = tmp_path / "broken.json"
        with pytest.raises(TypeError, match="cannot record a Linear"):
            save_record(broken, path)
        assert not path.exists()


class TestSeveralSeedsInTheRecord:
    def test_seed_indices_are_recorded_and_rerun(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.json"
        data = {"first": _data(0)}
        run_benchmark(data, _hybrid, record_path=path, n_seeds=2, **SETTINGS)
        record, _ = load_record(path)
        assert record["config"]["settings"]["n_seeds"] == 2
        assert [f["seed_index"] for f in record["folds"]][:4] == [0, 1, 0, 1]
        assert len({f["init_seed"] for f in record["seeds"]["folds"]}) > 1
        again = rerun_benchmark(record, data)
        assert [f.mcc for f in again.folds] == [f["mcc"] for f in record["folds"]]


class TestTuningInTheRecord:
    def test_budget_and_choices_are_recorded_and_rerun(self, tmp_path: Path) -> None:
        from hqnn_forge.benchmark import Tuning

        spaces = {"hybrid": {"lr": [0.01, 0.05]}, "control": {"lr": [0.01, 0.05]}}
        tuning = Tuning(n_trials=2, search_spaces=spaces, inner_folds=2)
        path = tmp_path / "tuned.json"
        data = {"first": _data(0)}
        run_benchmark(data, _hybrid, record_path=path, tuning=tuning, **SETTINGS)
        record, _ = load_record(path)
        assert record["config"]["settings"]["tuning"] == {
            "n_trials": 2,
            "inner_folds": 2,
            "search_spaces": spaces,
        }
        assert all(f["hyperparameters"]["lr"] in (0.01, 0.05) for f in record["folds"])
        again = rerun_benchmark(record, data)
        assert [f.mcc for f in again.folds] == [f["mcc"] for f in record["folds"]]
        assert [f.hyperparameters for f in again.folds] == [
            f["hyperparameters"] for f in record["folds"]
        ]


class TestNoiseInTheRecord:
    def test_noise_results_are_recorded_and_rerun(self, tmp_path: Path) -> None:
        path = tmp_path / "noise.json"
        data = {"first": _data(0)}
        run_benchmark(
            data,
            _hybrid,
            record_path=path,
            noise_levels=[0.0, 0.3],
            noise_position="end",
            **SETTINGS,
        )
        record, _ = load_record(path)
        assert record["config"]["settings"]["noise_levels"] == [0.0, 0.3]
        assert [row["noise_level"] for row in record["noise"]] == [0.0, 0.3]
        assert set(record["noise_summary"]) == {"first"}
        hybrid = [f for f in record["folds"] if f["model"] == "hybrid"]
        assert all(set(f["noise_mcc"]) == {"0.0", "0.3"} for f in hybrid)
        again = rerun_benchmark(record, data)
        assert [row["hybrid_mcc_mean"] for row in again.noise] == [
            row["hybrid_mcc_mean"] for row in record["noise"]
        ]
