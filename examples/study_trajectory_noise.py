"""
examples/study_trajectory_noise.py
==================================
Does noise-aware training by Pauli trajectories work as well as the exact
channel?  The study behind #311.

``noise_method="trajectories"`` (#229) samples the depolarizing channel at
pure-state cost; its gradient is unbiased but noisier than the exact
``"density"`` gradient.  This script trains the same hybrid classifier
noiselessly, with the density channel, and with trajectories at ``k = 1`` and
``k = 4`` draws per sample, and scores each on held-out data: clean (threshold
from the clean validation rows) and under the noise it was trained for
(threshold re-tuned on the validation rows under that noise, as a model
deployed on a noisy device would be).

Data: scikit-learn's breast-cancer set (569 samples, 63 % positive), which is
bundled, so the study runs offline.  Per seed: a stratified 60/20/20
train/validation/test split, standardisation and PCA to ``n_qubits`` fitted on
the training rows, and an ``HybridBinaryClassifier(n_qubits, n_qubits, 2)`` on
``default.qubit`` with backprop, trained with focal loss and MCC early stopping
(``train_model``, 30 epochs, patience 10).  The same seed fixes the split and
the initial weights of every method, so methods are compared on identical
starting points.

Writes one JSON line per run to ``--out`` (default
``trajectory_noise_study.jsonl``) and prints a summary table.  ``--quick`` runs
one seed at 4 qubits, for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hqnn_forge.evaluation import find_optimal_threshold, matthews_corrcoef
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import apply_depolarizing_noise
from hqnn_forge.training import train_model
from hqnn_forge.utils import FocalLoss

#: (noise_level, noise_position) the noise-aware runs train with.
NOISE = [(0.01, "all"), (0.05, "all"), (0.05, "end")]
#: name -> constructor options; the density runs are the exact reference.
METHODS: dict[str, dict[str, Any]] = {
    "density": {"noise_method": "density"},
    "trajectories-k1": {"noise_method": "trajectories", "noise_trajectories": 1},
    "trajectories-k4": {"noise_method": "trajectories", "noise_trajectories": 4},
}


def _split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 60/20/20 train/validation/test indices."""
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[], [], []]
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        a, b = int(0.6 * idx.size), int(0.8 * idx.size)
        for part, chunk in zip(parts, (idx[:a], idx[a:b], idx[b:]), strict=True):
            part.append(chunk)
    train, val, test = (np.sort(np.concatenate(p)) for p in parts)
    return train, val, test


def _prepare(X: np.ndarray, train: np.ndarray, n_components: int) -> np.ndarray:
    """Standardise and project onto the training rows' first principal components."""
    mean, std = X[train].mean(0), X[train].std(0) + 1e-12
    Z = (X - mean) / std
    _, _, vt = np.linalg.svd(Z[train], full_matrices=False)
    return Z @ vt[:n_components].T


def run_one(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_qubits: int,
    method: str,
    noise_level: float,
    noise_position: str,
    epochs: int,
    settings: list[tuple[float, str]],
) -> list[dict[str, Any]]:
    train, val, test = _split(y, seed)
    Z = torch.tensor(_prepare(X, train, n_qubits), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    options = (
        {}
        if method == "noiseless"
        else {
            "noise_level": noise_level,
            "noise_position": noise_position,
            **METHODS[method],
        }
    )
    torch.manual_seed(seed)
    model = HybridBinaryClassifier(
        n_qubits, n_qubits, 2, device_name="default.qubit", diff_method="backprop", **options
    )
    start = time.perf_counter()
    history = train_model(
        model,
        FocalLoss(),
        torch.optim.Adam(model.parameters(), lr=0.05),
        Z[train],
        yt[train],
        Z[val],
        yt[val],
        max_epochs=epochs,
        batch_size=32,
        monitor="mcc",
        patience=10,
        generator=torch.Generator().manual_seed(seed),
    )
    seconds = time.perf_counter() - start
    threshold = history.best_threshold if history.best_threshold is not None else 0.5
    labels = y[test]
    clean = model.predict_proba(Z[test])
    base = {
        "seed": seed,
        "n_qubits": n_qubits,
        "method": method,
        "epochs": history.n_epochs,
        "seconds": seconds,
        "mcc_clean": matthews_corrcoef(labels, (clean >= threshold).long()),
    }
    rows = []
    for p, pos in settings:
        # Under noise the probabilities shift, so the clean validation
        # threshold can land anywhere.  Tune it on the validation rows under
        # the same noise, then apply it to the noisy test rows: the deployed
        # model would be tuned on the device it runs on.
        with apply_depolarizing_noise(model, p, position=pos):  # type: ignore[arg-type]
            val_noisy = model.predict_proba(Z[val])
            test_noisy = model.predict_proba(Z[test])
        t_noisy = find_optimal_threshold(torch.tensor(y[val]), val_noisy).threshold
        rows.append(
            {
                **base,
                "noise_level": p,
                "noise_position": pos,
                "mcc_noisy": matthews_corrcoef(labels, (test_noisy >= t_noisy).long()),
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> str:
    keys = sorted({(r["n_qubits"], r["noise_level"], r["noise_position"]) for r in rows})
    methods = ["noiseless", *METHODS]
    lines = [
        (
            f"{'qubits':>6s} {'p':>5s} {'pos':>4s}  {'method':16s} {'MCC clean':>16s} "
            f"{'MCC noisy':>16s} {'s/run':>7s}  n"
        )
    ]
    for n, p, pos in keys:
        for method in methods:
            sel = [
                r
                for r in rows
                if r["n_qubits"] == n
                and r["method"] == method
                and (r["noise_level"], r["noise_position"]) == (p, pos)
            ]
            if not sel:
                continue
            clean = np.array([r["mcc_clean"] for r in sel])
            noisy = np.array([r["mcc_noisy"] for r in sel])
            secs = np.mean([r["seconds"] for r in sel])
            lines.append(
                f"{n:6d} {p:5.2f} {pos:>4s}  {method:16s} "
                f"{clean.mean():7.3f} ± {clean.std(ddof=1) if len(sel) > 1 else 0:.3f} "
                f"{noisy.mean():7.3f} ± {noisy.std(ddof=1) if len(sel) > 1 else 0:.3f} "
                f"{secs:7.1f}  {len(sel)}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--out", type=Path, default=Path("trajectory_noise_study.jsonl"))
    parser.add_argument("--quick", action="store_true", help="one seed, 4 qubits, 5 epochs")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    seeds, qubits, epochs = ((0,), (4,), 5) if args.quick else (range(5), (4, 6), 30)
    rows: list[dict[str, Any]] = []

    def record(new: list[dict[str, Any]], fh: Any) -> None:
        for row in new:
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        fh.flush()

    with args.out.open("w") as fh:
        for n in qubits:
            for seed in seeds:
                # One noiseless model per seed, scored under every noise setting.
                record(
                    run_one(
                        X,
                        y,
                        seed=seed,
                        n_qubits=n,
                        method="noiseless",
                        noise_level=0.0,
                        noise_position="all",
                        epochs=epochs,
                        settings=NOISE,
                    ),
                    fh,
                )
                for p, pos in NOISE:
                    for method in METHODS:
                        record(
                            run_one(
                                X,
                                y,
                                seed=seed,
                                n_qubits=n,
                                method=method,
                                noise_level=p,
                                noise_position=pos,
                                epochs=epochs,
                                settings=[(p, pos)],
                            ),
                            fh,
                        )
    print(summarise(rows))


if __name__ == "__main__":
    main()
