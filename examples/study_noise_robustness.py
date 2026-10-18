"""Frozen #348 study: prepare, smoke, resumable exact-density blocks, then analyze.

No real fit runs without the explicit ``run`` subcommand. ``prepare`` loads
strict sources and freezes metadata, protocol and execution hashes. Each run
block is a single-entry run_benchmark mapping with two paired initializations
and native classical controls. Public JSON contains no raw observations or
private paths; local .pt sidecars retain selected model/optimizer/preprocessor
states for verification. Completed blocks are immutable and hash-checked.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from hqnn_forge.benchmark import FoldResult, ValidationExtension, fingerprint, run_benchmark
from hqnn_forge.data import (
    BinaryDataset,
    load_cervical_cancer_risk,
    load_credit_card_fraud,
    load_iranian_churn,
    load_taiwanese_bankruptcy,
)
from hqnn_forge.evaluation import holm_correction, rank_biserial_correlation, wilcoxon_signed_rank
from hqnn_forge.experiment import _finite_diagnostics as _plain
from hqnn_forge.experiment import environment
from hqnn_forge.models import BinaryClassifierBase, HybridBinaryClassifier
from hqnn_forge.noise import Channel

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs/results/noise_aware_robustness_protocol.json"
DATASETS = ("credit_card", "taiwan", "churn", "cervical")
CHANNELS = ("depolarizing", "amplitude_damping", "phase_damping")
LEVELS = (0.0, 0.01, 0.05, 0.1, 0.2)
EXPECTED = {
    "credit_card": "6b052f9aab74d9f7e555d0bf4f9c12b3cc0aa38d23ea125f1f0552b39fb194db",
    "taiwan": "5ce94f3a91d14aa8adb63af19b5bb06c4fae18c30282bd46794e81a4c82aac8a",
    "churn": "1b647a1ba17d26e06cf1248d3c1b2d09f378f5ed163b5d3eb05b06c9c26fccf4",
    "cervical": "7733872af350750a43b5f2c0a0e653450ed490451e4a9204932305f2aff70faf",
}


def _hash(value: Any) -> str:
    """Canonical strict-JSON SHA-256; preserves full precision finite floats."""
    return hashlib.sha256(
        json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(_plain(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _unique_data(
    data: BinaryDataset, *, credit_card: bool = False
) -> tuple[BinaryDataset, dict[str, Any]]:
    """Keep first feature-identical representative in source order, independent of y.

    Credit card draws negatives then positives with RNG 348 and sorts the
    selected indices, retaining 200 positives and round(200*275190/473)
    negatives. Twenty test positives/fold imply 5 percentage-point recall
    resolution, not a precision/power guarantee. Conflicts are counted before
    representative selection; they never influence that selection.
    """
    X, y = data.X, data.y
    _, first, inverse, counts = np.unique(
        X, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    # NumPy 2.0 retains a singleton axis in inverse indices for axis=0.
    positive_counts = np.bincount(inverse.reshape(-1), weights=y, minlength=len(first))
    conflicting = (positive_counts > 0) & (positive_counts < counts)
    keep = np.sort(first)
    unique_X, unique_y = X[keep], y[keep]
    selected = np.arange(len(keep))
    if credit_card:
        rng = np.random.default_rng(348)
        selected = np.sort(
            np.concatenate(
                (
                    rng.choice(np.flatnonzero(unique_y == 0), 116359, replace=False),
                    rng.choice(np.flatnonzero(unique_y == 1), 200, replace=False),
                )
            )
        )
    result = BinaryDataset(unique_X[selected], unique_y[selected], data.feature_names)
    return result, {
        "eligible_rows": len(y),
        "eligible_positives": int(y.sum()),
        "eligible_fingerprint": fingerprint(X, y),
        "unique_rows": len(keep),
        "unique_positives": int(unique_y.sum()),
        "duplicate_rows_removed": len(y) - len(keep),
        "duplicate_policy": "first_source_feature_vector_label_independent",
        "conflicting_groups": int(conflicting.sum()),
        "conflicting_source_rows": int(counts[conflicting].sum()),
        "sample_rows": len(result.y),
        "sample_positives": int(result.y.sum()),
        "prevalence": float(result.y.mean()),
        "feature_names": result.feature_names,
        "fingerprint": fingerprint(result.X, result.y),
        "selected_source_indices_sha256": hashlib.sha256(
            keep[selected].astype("<i8").tobytes()
        ).hexdigest(),
        "sampling": {"seed": 348, "negative_first": True, "positives": 200, "negatives": 116359}
        if credit_card
        else None,
    }


def _load(name: str, credit_card: Path, uci: Path) -> tuple[BinaryDataset, dict[str, Any]]:
    from hqnn_forge.data.uci import CERVICAL_CANCER_RISK, IRANIAN_CHURN, TAIWANESE_BANKRUPTCY

    if name == "credit_card":
        loaded = load_credit_card_fraud(credit_card, strict=True, drop_time=True)
        data = BinaryDataset(loaded.X, loaded.y, loaded.feature_names)
        source = credit_card / "creditcard.csv" if credit_card.is_dir() else credit_card
        raw_rows, raw_positive = 284807, 492
        options: dict[str, Any] = {"strict": True, "drop_time": True}
    else:
        loaders = {
            "taiwan": load_taiwanese_bankruptcy,
            "churn": load_iranian_churn,
            "cervical": load_cervical_cancer_risk,
        }
        archives = {
            "taiwan": TAIWANESE_BANKRUPTCY,
            "churn": IRANIAN_CHURN,
            "cervical": CERVICAL_CANCER_RISK,
        }
        data = loaders[name](uci, strict=True)
        source = uci / archives[name].file_name
        raw_rows, raw_positive = {
            "taiwan": (6819, 220),
            "churn": (3150, 495),
            "cervical": (858, 55),
        }[name]
        options = {
            "strict": True,
            **(
                {"missing": "drop"}
                if name == "cervical"
                else {"drop_duplicates": False}
                if name == "churn"
                else {}
            ),
        }
    selected, metadata = _unique_data(data, credit_card=name == "credit_card")
    metadata.update(
        {
            "raw_rows": raw_rows,
            "raw_positives": raw_positive,
            "loader_options": options,
            "source_sha256": _file_hash(source),
        }
    )
    if metadata["fingerprint"] != EXPECTED[name]:
        raise ValueError(f"{name}: data differs from the prospectively verified fingerprint.")
    return selected, metadata


def _builder(channel: str, training: float) -> Any:
    def build(width: int) -> HybridBinaryClassifier:
        return HybridBinaryClassifier(
            width,
            4,
            2,
            device_name="default.qubit",
            diff_method="backprop",
            noise_method="density",
            noise_channel=cast("Channel", channel),
            noise_level=training,
            noise_position="all",
        )  # type: ignore[arg-type]

    return build


def _recipe() -> dict[str, Any]:
    return dict(
        n_splits=10,
        validation_folds=5,
        n_seeds=2,
        random_state=348,
        oversample=False,
        pca_components=4,
        lr=0.05,
        batch_size=32,
        max_epochs=30,
        patience=10,
        noise_levels=LEVELS,
        noise_position="all",
        retune_noise_threshold=True,
        noise_batch_size=256,
        validation_extension=ValidationExtension(),
    )


def _execution() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("Scientific execution requires clean tracked source.")
    return {
        "source_sha": head,
        "lock_sha256": _file_hash(ROOT / "uv.lock"),
        "protocol_sha256": _hash(_read(PROTOCOL_PATH)),
        "protocol": _read(PROTOCOL_PATH),
        "environment": environment(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "dtype": "float32",
        "device": "cpu",
    }


def _block_id(dataset: str, channel: str, training: float, fold: int) -> str:
    return f"{dataset}--{channel}--train-{training:g}--fold-{fold}"


def _inventory() -> list[tuple[str, str, float, int]]:
    return list(itertools.product(DATASETS, CHANNELS, (0.0, 0.2), range(10)))


@contextlib.contextmanager
def _ownership(directory: Path) -> Iterator[None]:
    """Process-held advisory lock releases on interruption; metadata survives."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "owner.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this execution directory.") from exc
        _atomic_json(
            directory / "local-process.json", {"pid": os.getpid(), "started": time.time()}
        )
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _validate_block(block: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if block.get("identity") != expected:
        raise ValueError("Completed block identity differs; use a fresh execution directory.")
    record = block["record"]
    dataset, channel, training, fold = expected["cell"]
    entries = record["folds"]
    inventory = [(row["model"], row["seed_index"]) for row in entries]
    if sorted(inventory) != [("control", 0), ("control", 1), ("hybrid", 0), ("hybrid", 1)]:
        raise ValueError("Incomplete or duplicated block replicas.")
    for row in entries:
        if row["dataset"] != dataset or row["fold"] != fold:
            raise ValueError("Misidentified block fold.")
        if row["model"] == "hybrid":
            if [
                (r["channel"], r["probability"], r["placement"]) for r in row["noise_results"]
            ] != [(channel, p, "all") for p in LEVELS]:
                raise ValueError("Incomplete evaluation levels.")
            noise = row["diagnostics"]["training_noise"]
            if noise["probability"] != training or noise["channel"] != channel:
                raise ValueError("Training noise differs from block identity.")
    settings = record["config"]["settings"]
    expected_settings = _recipe()
    expected_settings["validation_extension"] = asdict(expected_settings["validation_extension"])
    for key, value in expected_settings.items():
        if settings[key] != _plain(value):
            raise ValueError(f"Recorded recipe differs: {key}.")
    if settings["fold_indices"] != [fold] or settings["noise_channel"] != channel:
        raise ValueError("Recorded fold/channel differs.")
    if record["datasets"][dataset]["sha256"] != expected["data_sha256"]:
        raise ValueError("Recorded data fingerprint differs.")
    # Derive all ten roots before selecting this fold, as the runner does.
    roots = np.random.SeedSequence([348, 0]).spawn(11)
    split_seed = int(roots[0].generate_state(1)[0])
    inner, smote, initial, batch = [
        int(root.generate_state(1)[0]) for root in roots[fold + 1].spawn(4)
    ]
    second = int(np.random.SeedSequence([initial, 1]).spawn(1)[0].generate_state(1)[0])
    for row in entries:
        seeds = (
            row["split_seed"],
            row["inner_seed"],
            row["smote_seed"],
            row["init_seed"],
            row["batch_seed"],
        )
        if seeds != (split_seed, inner, smote, [initial, second][row["seed_index"]], batch):
            raise ValueError("Recorded seeds differ from complete ten-fold seed tree.")
    if block.get("record_sha256") != _hash(record):
        raise ValueError("Completed record hash mismatch.")


def _prepare(directory: Path, credit_card: Path, uci: Path) -> dict[str, Any]:
    execution = _execution()
    proof_path = directory / "smoke-proof.json"
    if not proof_path.exists():
        raise ValueError(
            "Run synthetic smoke separately, then prepare with --smoke-proof <smoke.json>."
        )
    proof = _read(proof_path)
    if (
        not proof.get("smoke")
        or proof["code_sha256"] != _code_hash()
        or proof["protocol_sha256"] != execution["protocol_sha256"]
    ):
        raise ValueError("Smoke proof does not match this code and protocol.")
    data = {name: _load(name, credit_card, uci)[1] for name in DATASETS}
    manifest = {
        "execution": execution,
        "datasets": data,
        "planned_cells": [_block_id(*cell) for cell in _inventory()],
        "smoke": False,
        "smoke_proof": proof,
    }
    target = directory / "manifest.json"
    if target.exists():
        if _read(target) != _plain(manifest):
            raise ValueError("Manifest differs; completed execution is immutable.")
    else:
        _atomic_json(target, manifest)
    return manifest


def _run(
    directory: Path,
    credit_card: Path,
    uci: Path,
    dataset: str | None,
    channel: str | None,
    training: float | None,
    fold: int | None,
) -> None:
    with _ownership(directory):
        manifest = _prepare(directory, credit_card, uci)
        completed = []
        for cell in _inventory():
            name, c, t, f = cell
            if (
                (dataset is not None and dataset != name)
                or (channel is not None and channel != c)
                or (training is not None and training != t)
                or (fold is not None and fold != f)
            ):
                continue
            block_id = _block_id(*cell)
            target = directory / "blocks" / f"{block_id}.json"
            identity = {
                "cell": list(cell),
                "execution_sha256": _hash(manifest["execution"]),
                "data_sha256": manifest["datasets"][name]["fingerprint"],
            }
            if target.exists():
                _validate_block(_read(target), identity)
                completed.append(block_id)
                print(f"reused {block_id}", flush=True)
                continue
            data, _ = _load(name, credit_card, uci)
            local = directory / "local" / block_id
            local.mkdir(parents=True, exist_ok=True)

            def observe(
                model: BinaryClassifierBase,
                result: FoldResult,
                history: dict[str, Any],
                local: Path = local,
                block_id: str = block_id,
            ) -> None:
                key = f"{result.model}-seed-{result.seed_index}"
                temporary = local / f"{key}.pt.tmp"
                torch.save(
                    {
                        "config": model.get_config(),
                        "state_dict": model.state_dict(),
                        "optimizer_state": history.get("optimizer_state"),
                        "diagnostics": result.diagnostics,
                    },
                    temporary,
                )
                temporary.replace(local / f"{key}.pt")
                for attempt in result.diagnostics["attempts"]:
                    _atomic_json(local / f"{key}-attempt-{attempt['attempt']}.json", attempt)
                _atomic_json(
                    directory / "progress.json",
                    {
                        "active": block_id,
                        "last_model": key,
                        "completed_this_invocation": completed,
                    },
                )
                print(
                    f"scored {block_id} {key} attempts={len(result.diagnostics['attempts'])} epochs={result.epochs}",
                    flush=True,
                )

            start = time.perf_counter()
            record_temp = local / "record.json.tmp"
            run_benchmark(
                {name: (data.X, data.y)},
                _builder(c, t),
                fold_indices=[f],
                noise_channel=cast("Channel", c),
                on_fold_end=observe,
                record_path=record_temp,
                **_recipe(),
            )  # type: ignore[arg-type]
            record = _read(record_temp)
            block = {
                "identity": identity,
                "record": record,
                "record_sha256": _hash(record),
                "wall_seconds": time.perf_counter() - start,
            }
            _validate_block(block, identity)
            _atomic_json(target, block)
            completed.append(block_id)
            _atomic_json(
                directory / "progress.json",
                {"active": None, "completed_this_invocation": completed},
            )


def _signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        _hash(row[key])
        for key in (
            "train_idx",
            "val_idx",
            "test_idx",
            "split_seed",
            "inner_seed",
            "init_seed",
            "batch_seed",
        )
    )


def _primary_statistics(seed_differences: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Average two seed MCC differences within each of ten folds, then test.

    Positive difference means density-trained minus noiseless-trained hybrid.
    Only study-local |mean|<=1e-12 differences become zero for ranks/tests.
    Raw effects retain full precision. For n effective nonzero pairs the
    exact two-sided floor is 2/2**n; seed replicas never enlarge n.
    """
    values = np.asarray(seed_differences, dtype=float)
    if values.shape != (10, 2):
        raise ValueError("Primary inference needs ten folds and two paired seeds.")
    raw = values.mean(axis=1)
    finite = bool(np.isfinite(raw).all())
    differences = np.where(np.abs(raw) <= 1e-12, 0.0, raw)
    n = int(np.count_nonzero(differences)) if finite else 0
    p = effect = floor = None
    if n:
        test = wilcoxon_signed_rank(differences, np.zeros(10))
        p, floor = test.p_value, test.min_p_value
        effect = rank_biserial_correlation(differences, np.zeros(10))
    return {
        "seed_differences": values.tolist(),
        "fold_differences": raw.tolist(),
        "mean_difference": float(raw.mean()),
        "median_difference": float(np.median(raw)),
        "sample_std": float(raw.std(ddof=1)),
        "effective_pairs": n,
        "wilcoxon_p": p,
        "wilcoxon_min_p": floor,
        "rank_biserial": effect if effect is not None else (0.0 if finite else None),
    }


def _adjust_family(rows: list[dict[str, Any]]) -> None:
    """Keep all 12 planned slots: undefined tests occupy p=1, reported null.

    Ordered p_(i), i=1..12, adjust to min(1,max_{j<=i}(13-j)*p_(j)).
    Attainable adjusted floors apply the same ordering to each test's exact
    floor (undefined slots remain 1); the 12*floor first-step bound is also
    reported to show what the most significant slot could attain alone.
    """
    if len(rows) != 12:
        raise ValueError("The confirmatory family must contain twelve planned slots.")
    adjusted = holm_correction(
        [row["wilcoxon_p"] if row["wilcoxon_p"] is not None else 1.0 for row in rows]
    )
    floors = holm_correction(
        [row["wilcoxon_min_p"] if row["wilcoxon_min_p"] is not None else 1.0 for row in rows]
    )
    for row, p, floor in zip(rows, adjusted, floors, strict=True):
        row["wilcoxon_p_holm"] = float(p) if row["wilcoxon_p"] is not None else None
        row["holm_attainable_floor"] = float(floor) if row["wilcoxon_min_p"] is not None else None
        row["holm_first_step_bound"] = (
            min(1.0, 12 * row["wilcoxon_min_p"]) if row["wilcoxon_min_p"] is not None else None
        )


def _analyze(directory: Path) -> dict[str, Any]:
    manifest = _read(directory / "manifest.json")
    if manifest.get("smoke") or manifest["execution"]["protocol_sha256"] != _hash(
        _read(PROTOCOL_PATH)
    ):
        raise ValueError("Analysis requires the frozen scientific protocol, never smoke data.")
    expected_ids = {_block_id(*cell) for cell in _inventory()}
    actual_ids = {path.stem for path in (directory / "blocks").glob("*.json")}
    if actual_ids != expected_ids:
        raise ValueError("Analysis requires exactly all 240 planned blocks.")
    entries: dict[tuple[str, str, float, int, int], dict[str, Any]] = {}
    controls = []
    wall_seconds = 0.0
    for cell in _inventory():
        name, c, t, f = cell
        block = _read(directory / "blocks" / f"{_block_id(*cell)}.json")
        _validate_block(
            block,
            {
                "cell": list(cell),
                "execution_sha256": _hash(manifest["execution"]),
                "data_sha256": manifest["datasets"][name]["fingerprint"],
            },
        )
        wall_seconds += block["wall_seconds"]
        for row in block["record"]["folds"]:
            if row["model"] == "control":
                controls.append(row)
            else:
                entries[name, c, t, f, row["seed_index"]] = row
    primary, secondary, learning = [], [], {}
    for name in DATASETS:
        passes = []
        clean: list[float] = []
        for f in range(10):
            baseline_rows = [entries[name, CHANNELS[0], 0.0, f, seed] for seed in range(2)]
            diagnostics = [row["diagnostics"]["clean_validation"] for row in baseline_rows]
            prevalence = diagnostics[0]["prevalence"]
            passes.append(
                all(d["finite"] for d in diagnostics)
                and np.median([d["mcc"] if d["mcc"] is not None else -1 for d in diagnostics])
                >= 0.1
                and np.median(
                    [d["pr_auc"] if d["pr_auc"] is not None else -1 for d in diagnostics]
                )
                >= 2 * prevalence
            )
            clean.extend(row["mcc"] for row in baseline_rows)
            for seed, reference in enumerate(baseline_rows):
                for c in CHANNELS:
                    row = entries[name, c, 0.0, f, seed]
                    if (
                        _signature(row) != _signature(reference)
                        or row["mcc"] != reference["mcc"]
                        or row["diagnostics"]["selected_state_sha256"]
                        != reference["diagnostics"]["selected_state_sha256"]
                        or row["diagnostics"]["clean_validation"]
                        != reference["diagnostics"]["clean_validation"]
                    ):
                        raise ValueError("Repeated noiseless baselines disagree across channels.")
        learning[name] = {
            "passing_folds": int(sum(passes)),
            "fold_passes": [bool(p) for p in passes],
            "clean_baseline_mcc": clean,
            "validation_learned": sum(passes) >= 8,
            "clean_baseline_median_mcc": float(np.median(clean)),
            "interpretation": "learning-limited/inconclusive"
            if sum(passes) < 8
            else "validation guard passes; assess held-out clean learning before recommendation",
        }
        for c in CHANNELS:
            for level in LEVELS:
                differences, seed_scores = [], []
                for f in range(10):
                    paired = []
                    for seed in range(2):
                        base, noisy = (entries[name, c, t, f, seed] for t in (0.0, 0.2))
                        if (
                            _signature(base) != _signature(noisy)
                            or base["diagnostics"]["initial_state_sha256"]
                            != noisy["diagnostics"]["initial_state_sha256"]
                            or base["diagnostics"]["preprocessing"]
                            != noisy["diagnostics"]["preprocessing"]
                        ):
                            raise ValueError("Mispaired training arms.")
                        b, n = (
                            next(r for r in row["noise_results"] if r["probability"] == level)
                            for row in (base, noisy)
                        )
                        diff = (
                            n["retuned_mcc"] - b["retuned_mcc"]
                            if n["retuned_mcc"] is not None and b["retuned_mcc"] is not None
                            else math.nan
                        )
                        paired.append(diff)
                        seed_scores.append(
                            {
                                "fold": f,
                                "seed_index": seed,
                                "init_seed": base["init_seed"],
                                "noiseless": b,
                                "density": n,
                                "clean_noiseless_mcc": base["mcc"],
                                "clean_density_mcc": noisy["mcc"],
                            }
                        )
                    differences.append(paired)
                if level == 0.2:
                    primary.append(
                        {
                            "dataset": name,
                            "channel": c,
                            **_primary_statistics(differences),
                            "seed_scores": seed_scores,
                        }
                    )
                else:
                    secondary.append(
                        {
                            "dataset": name,
                            "channel": c,
                            "evaluation_probability": level,
                            "fold_differences": np.mean(differences, axis=1).tolist(),
                            "seed_scores": seed_scores,
                        }
                    )
    _adjust_family(primary)
    return _plain(
        {
            "protocol_sha256": manifest["execution"]["protocol_sha256"],
            "primary": primary,
            "secondary_descriptive": secondary,
            "learning": learning,
            "classical_context": controls,
            "summed_block_wall_seconds": wall_seconds,
            "manifest": manifest,
            "limitations": "overlapping training folds; paired-CV convention; limited power; unique feature vectors; capped credit card; cervical complete cases and very few positives; non-rejection does not establish equality",
        }
    )


def _code_hash() -> str:
    return _hash(
        {
            name: _file_hash(ROOT / name)
            for name in (
                "hqnn_forge/benchmark.py",
                "hqnn_forge/experiment.py",
                "examples/study_noise_robustness.py",
                "uv.lock",
            )
        }
    )


def _smoke(directory: Path) -> None:
    """Synthetic only: exercise all three channels and the actual study API."""
    rng = np.random.default_rng(348)
    X = rng.normal(size=(80, 5))
    y = np.tile([0, 0, 0, 1], 20)
    for channel in CHANNELS:
        run_benchmark(
            {"SYNTHETIC_SMOKE": (X, y)},
            _builder(channel, 0.2),
            n_splits=2,
            validation_folds=2,
            n_seeds=2,
            max_epochs=1,
            patience=1,
            oversample=False,
            pca_components=4,
            batch_size=32,
            lr=0.05,
            noise_levels=[0.0, 0.2],
            noise_channel=cast("Channel", channel),
            retune_noise_threshold=True,
            noise_batch_size=16,
            fold_indices=[1],
            validation_extension=ValidationExtension(max_epochs=1),
            record_path=directory / f"smoke-{channel}.json",
        )  # type: ignore[arg-type]
    _atomic_json(
        directory / "smoke.json",
        {
            "smoke": True,
            "protocol_sha256": _hash(_read(PROTOCOL_PATH)),
            "code_sha256": _code_hash(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credit-card", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--uci", type=Path, default=Path("data/raw"))
    parser.add_argument("--smoke-proof", type=Path)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--channel", choices=CHANNELS)
    parser.add_argument("--training", type=float, choices=(0.0, 0.2))
    parser.add_argument("--fold", type=int, choices=range(10))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    status = 1
    try:
        if args.action == "prepare":
            with _ownership(args.output):
                if args.smoke_proof is not None:
                    target = args.output / "smoke-proof.json"
                    if target.exists() and _read(target) != _read(args.smoke_proof):
                        raise ValueError("Existing smoke proof is immutable.")
                    _atomic_json(target, _read(args.smoke_proof))
                _prepare(args.output, args.credit_card, args.uci)
        elif args.action == "smoke":
            with _ownership(args.output):
                _smoke(args.output)
        elif args.action == "run":
            _run(
                args.output,
                args.credit_card,
                args.uci,
                args.dataset,
                args.channel,
                args.training,
                args.fold,
            )
        else:
            with _ownership(args.output):
                _atomic_json(args.output / "analysis.json", _analyze(args.output))
        status = 0
    finally:
        _atomic_json(
            args.output / "exit-status.json",
            {"exit_status": status, "finished": time.time(), "action": args.action},
        )


if __name__ == "__main__":
    main()
