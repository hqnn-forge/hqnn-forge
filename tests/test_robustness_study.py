"""Independent arithmetic for the study-local inference and data policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hqnn_forge.data import BinaryDataset

_spec = importlib.util.spec_from_file_location(
    "study_noise_robustness",
    Path(__file__).resolve().parents[1] / "examples/study_noise_robustness.py",
)
assert _spec is not None and _spec.loader is not None
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


@pytest.mark.parametrize("pairs, floor", [(10, 2 / 1024), (9, 2 / 512), (8, 2 / 256)])
def test_exact_floor_and_fixed_twelve_slot_family(pairs: int, floor: float) -> None:
    values = [[0.1, 0.1]] * pairs + [[0.0, 0.0]] * (10 - pairs)
    stats = study._primary_statistics(values)
    assert stats["wilcoxon_p"] == floor
    assert stats["wilcoxon_min_p"] == floor
    assert stats["effective_pairs"] == pairs
    assert stats["rank_biserial"] == 1.0
    rows = [dict(stats)] + [study._primary_statistics([[0.0, 0.0]] * 10) for _ in range(11)]
    study._adjust_family(rows)
    assert rows[0]["wilcoxon_p_holm"] == 12 * floor
    assert rows[0]["holm_first_step_bound"] == 12 * floor
    assert all(r["wilcoxon_p_holm"] is None for r in rows[1:])


def test_seed_offsets_cancel_within_folds_and_local_tolerance() -> None:
    stats = study._primary_statistics([[0.2, -0.2 + 1e-13]] * 10)
    assert stats["effective_pairs"] == 0
    assert stats["wilcoxon_p"] is None
    assert stats["mean_difference"] != 0.0
    stats = study._primary_statistics([[2e-12, 2e-12]] * 10)
    assert stats["effective_pairs"] == 10


def test_nonfinite_and_missing_pairs_never_shrink_family() -> None:
    stats = study._primary_statistics([[float("nan"), 0.1]] + [[0.1, 0.1]] * 9)
    assert stats["wilcoxon_p"] is None
    with pytest.raises(ValueError, match="ten folds"):
        study._primary_statistics([[0.1, 0.1]] * 9)
    with pytest.raises(ValueError, match="twelve"):
        study._adjust_family([stats] * 11)


@pytest.mark.parametrize("inverse_shape", ["flat", "column"])
def test_unique_features_representative_is_source_first_not_label(
    monkeypatch: pytest.MonkeyPatch, inverse_shape: str
) -> None:
    unique = np.unique

    def shaped_unique(*args: Any, **kwargs: Any) -> Any:
        values, first, inverse, counts = unique(*args, **kwargs)
        inverse = inverse.reshape(-1) if inverse_shape == "flat" else inverse.reshape(-1, 1)
        return values, first, inverse, counts

    monkeypatch.setattr(study.np, "unique", shaped_unique)
    data = BinaryDataset(
        np.array([[1.0, 2.0], [2.0, 3.0], [1.0, 2.0], [4.0, 5.0], [2.0, 3.0]]),
        np.array([0, 0, 1, 1, 0]),
        ("a", "b"),
    )
    result, metadata = study._unique_data(data)
    np.testing.assert_array_equal(result.X, [[1, 2], [2, 3], [4, 5]])
    np.testing.assert_array_equal(result.y, [0, 0, 1])
    assert metadata["conflicting_groups"] == 1 and metadata["conflicting_source_rows"] == 2
    assert metadata["duplicate_rows_removed"] == 2


def test_incomplete_scientific_inventory_cannot_be_analyzed(tmp_path: Path) -> None:
    study._atomic_json(
        tmp_path / "manifest.json",
        {
            "smoke": False,
            "execution": {"protocol_sha256": study._hash(study._read(study.PROTOCOL_PATH))},
        },
    )
    with pytest.raises(ValueError, match="240"):
        study._analyze(tmp_path)


def test_protocol_hash_is_committed_and_recipe_agrees() -> None:
    protocol = study._read(study.PROTOCOL_PATH)
    assert study._hash(protocol) == study.PROTOCOL_PATH.with_suffix(".sha256").read_text().strip()
    settings = study._recipe()
    assert settings["n_splits"] == 10 and settings["n_seeds"] == 2
    assert settings["batch_size"] == protocol["training"]["batch_size"] == 32
    assert settings["max_epochs"] == protocol["training"]["max_epochs"] == 30
    assert len(study._inventory()) == 240


def test_completed_block_identity_and_inventory_are_enforced() -> None:
    with pytest.raises(ValueError, match="identity"):
        study._validate_block(
            {"identity": {"wrong": True}}, {"cell": ["taiwan", "depolarizing", 0.2, 0]}
        )
    block: dict[str, Any] = {
        "identity": {"cell": ["taiwan", "depolarizing", 0.2, 0]},
        "record": {"folds": []},
    }
    with pytest.raises(ValueError, match="replicas"):
        study._validate_block(block, block["identity"])


@pytest.fixture
def artifacts(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Tiny synthetic saved records; no model fits or real observations."""
    from dataclasses import asdict

    manifest = {
        "smoke": False,
        "execution": {"protocol_sha256": study._hash(study._read(study.PROTOCOL_PATH))},
        "datasets": {name: {"fingerprint": name} for name in study.DATASETS},
    }
    study._atomic_json(tmp_path / "manifest.json", manifest)
    for dataset, channel, training, fold in study._inventory():
        roots = np.random.SeedSequence([348, 0]).spawn(11)
        split = int(roots[0].generate_state(1)[0])
        inner, smote, initial, batch = [
            int(root.generate_state(1)[0]) for root in roots[fold + 1].spawn(4)
        ]
        second = int(np.random.SeedSequence([initial, 1]).spawn(1)[0].generate_state(1)[0])
        rows = []
        for model in ("hybrid", "control"):
            for seed, init in enumerate((initial, second)):
                score = (0.3 + training) if model == "hybrid" else -0.8
                row = {
                    "dataset": dataset,
                    "model": model,
                    "fold": fold,
                    "seed_index": seed,
                    "split_seed": split,
                    "inner_seed": inner,
                    "smote_seed": smote,
                    "init_seed": init,
                    "batch_seed": batch,
                    "train_idx": [0, 1],
                    "val_idx": [2, 3],
                    "test_idx": [4, 5],
                    "mcc": score,
                    "diagnostics": {
                        "training_noise": {"probability": training, "channel": channel},
                        "clean_validation": {
                            "finite": True,
                            "mcc": 0.5,
                            "pr_auc": 0.9,
                            "prevalence": 0.1,
                        },
                        "initial_state_sha256": str(init),
                        "selected_state_sha256": str((init, training)),
                        "preprocessing": {"sha256": "transform"},
                    },
                    "noise_results": [
                        {
                            "channel": channel,
                            "probability": p,
                            "placement": "all",
                            "retuned_mcc": score,
                        }
                        for p in study.LEVELS
                    ],
                }
                rows.append(row)
        settings = study._recipe()
        settings["validation_extension"] = asdict(settings["validation_extension"])
        settings.update(fold_indices=[fold], noise_channel=channel)
        record = {
            "folds": rows,
            "config": {"settings": study._plain(settings)},
            "datasets": {dataset: {"sha256": dataset}},
        }
        cell = [dataset, channel, training, fold]
        block = {
            "identity": {
                "cell": cell,
                "execution_sha256": study._hash(manifest["execution"]),
                "data_sha256": dataset,
            },
            "record": record,
            "record_sha256": study._hash(record),
            "wall_seconds": 1.0,
        }
        study._atomic_json(tmp_path / "blocks" / f"{study._block_id(*cell)}.json", block)
    return tmp_path, manifest


def test_saved_artifact_analysis_uses_noiseless_hybrid_not_classical(
    artifacts: tuple[Path, dict[str, Any]],
) -> None:
    path, _ = artifacts
    analysis = study._analyze(path)
    assert len(analysis["primary"]) == 12
    assert len(analysis["secondary_descriptive"]) == 48
    assert len(analysis["classical_context"]) == 480
    for row in analysis["primary"]:
        assert row["mean_difference"] == pytest.approx(0.2)
        assert row["wilcoxon_p"] == 2 / 1024
        assert row["wilcoxon_p_holm"] == 24 / 1024
    assert all(row["passing_folds"] == 10 for row in analysis["learning"].values())


def test_completed_block_resumed_without_training(
    artifacts: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = artifacts
    monkeypatch.setattr(study, "_prepare", lambda *a: manifest)
    monkeypatch.setattr(study, "_load", lambda *a: pytest.fail("completed block reloaded/trained"))
    study._run(path, Path("unused"), Path("unused"), "taiwan", "phase_damping", 0.2, 9)


@pytest.mark.parametrize("defect", ["seed", "duplicate", "missing_level", "initial_state"])
def test_mispaired_or_incomplete_artifacts_rejected(
    artifacts: tuple[Path, dict[str, Any]], defect: str
) -> None:
    path, _ = artifacts
    target = path / "blocks" / f"{study._block_id('taiwan', 'phase_damping', 0.2, 9)}.json"
    block = study._read(target)
    row = block["record"]["folds"][0]
    if defect == "seed":
        row["init_seed"] += 1
    elif defect == "duplicate":
        block["record"]["folds"][1] = dict(row)
    elif defect == "missing_level":
        row["noise_results"].pop()
    else:
        row["diagnostics"]["initial_state_sha256"] = "different"
    block["record_sha256"] = study._hash(block["record"])
    study._atomic_json(target, block)
    with pytest.raises(ValueError, match="seeds|replicas|levels|Mispaired"):
        study._analyze(path)


def test_actual_training_arms_share_initial_weights_inputs_and_batch_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from hqnn_forge import benchmark
    from hqnn_forge.models import HybridBinaryClassifier

    seen: list[dict[str, Any]] = []

    def fit(model: Any, loss: Any, *arrays: Any, **options: Any) -> Any:
        seen.append(
            {
                "hybrid": isinstance(model, HybridBinaryClassifier),
                "state": {key: value.clone() for key, value in model.state_dict().items()},
                "arrays": [array.copy() for array in arrays],
                "batch_seed": options["batch_seed"],
            }
        )
        return benchmark.FitScore(0.0, 0.5, 0.0, 0, 0.0, 0.0)

    monkeypatch.setattr(benchmark, "_fit_and_score", fit)
    X = np.random.default_rng(9).normal(size=(200, 5))
    y = np.tile([0, 0, 0, 1], 50)
    reference: list[dict[str, Any]] | None = None
    for channel in study.CHANNELS:
        for training in (0.0, 0.2):
            seen.clear()
            settings = study._recipe()
            settings["noise_levels"] = None
            benchmark.run_benchmark(
                {"synthetic": (X, y)},
                study._builder(channel, training),
                fold_indices=[9],
                **settings,
            )
            if reference is None:
                reference = list(seen)
                continue
            for got, want in zip(seen, reference, strict=True):
                assert got["batch_seed"] == want["batch_seed"]
                for key in want["state"]:
                    torch.testing.assert_close(
                        got["state"][key], want["state"][key], rtol=0, atol=0
                    )
                for actual, expected in zip(got["arrays"], want["arrays"], strict=True):
                    np.testing.assert_array_equal(actual, expected)
