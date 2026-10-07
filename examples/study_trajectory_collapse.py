"""
examples/study_trajectory_collapse.py
=====================================
Can the occasional collapse of trajectory-noise training be prevented?  The
study behind #347.

``study_trajectory_noise.py`` (#311) found that training with
``noise_method="trajectories"`` at ``p = 0.05`` after every gate left 3 of 20
runs below a test MCC of 0.5 (0.13 to 0.42) within its 30 epochs, with Adam at
lr 0.05 for every method, while the exact ``"density"`` channel always
trained.  This script re-runs that
setting on the same data, splits and model, over 10 seeds (0 to 4 are the
original ones), with these changes to the trajectory runs:

* ``lr`` 0.02 and 0.01 instead of 0.05;
* ``clip``: gradient-norm clipping at 0.1 (lr 0.05).  The median unclipped
  norm was 0.05 at ``k = 1`` and 0.10 to 0.12 at ``k = 4`` and ``8``.  The
  bound acted on 6.5 to 8.8 % of the steps on average at ``k = 1`` (0 to 20 %
  per run) and on 43 to 58 % at ``k = 4`` and ``8`` (0.6 to 63 % per run);
* ``warmup``: the noise ramps linearly from ``p/5`` to ``p`` over the first
  5 epochs (lr 0.05);

each at ``k = 1``, ``4`` and ``8`` draws per sample.  The density channel is
trained at every learning rate, so each trajectory run has a density run from
the same seed and at the same learning rate to be paired with.

A run *collapses* when its test MCC under the training noise is below 0.5; the
successful runs in #311 all scored at least 0.68.  Training is at most 30
epochs with a patience of 10, as in #311, so a collapse means "had not trained
within that budget": a run that starts late counts the same as one that never
trains.  Separating the two is #480.

Writes one JSON line per run to ``--out`` (default
``trajectory_collapse_study.jsonl``), in parallel over ``--workers``
processes, and prints a summary.  ``--quick`` runs one seed at 4 qubits for
5 epochs, for a smoke test.  ``--settings`` re-runs the given methods at the
other two noise settings of #311 (p = 0.01 after every gate, p = 0.05 before
measurement), to check that a mitigation does not cost accuracy there;
``--settings-only`` skips the main grid, and ``--no-density`` with
``--reference`` pairs them with an earlier file's density runs instead of
training new ones.  The results in ``docs/results/trajectory-collapse-study.md``
came from::

    cd examples
    python study_trajectory_collapse.py --workers 14 \
        --out trajectory_collapse_study.jsonl
    python study_trajectory_collapse.py --workers 14 --settings-only --no-density \
        --settings trajectories:1:lr0.05 trajectories:4:lr0.05 trajectories:8:lr0.05 \
        --reference ../docs/results/trajectory_noise_study.jsonl \
        --out trajectory_collapse_settings.jsonl

Each worker runs one training at a time on one thread; at 6 qubits one
density run then takes about half an hour.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from study_trajectory_noise import _prepare, _split

from hqnn_forge.evaluation import find_optimal_threshold, matthews_corrcoef
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import apply_depolarizing_noise, trajectory_noise_qnode
from hqnn_forge.training import EpochRecord, train_model
from hqnn_forge.utils import FocalLoss

#: A run whose test MCC under the training noise is below this collapsed.
COLLAPSE_MCC = 0.5
#: Gradient-norm bound of the ``clip`` variant.  The per-run median of the
#: unclipped global norm was 0.02 to 0.07 at k = 1 and 0.03 to 0.13 at k = 4
#: and 8.  A bound of 1.0 would never have acted at k = 1 (per-run maxima at
#: most 0.43) and only on the largest steps at k = 4 and 8 (maxima above 1.0
#: in 18 of the 40 runs, up to 1.26 at k = 4 and 3.78 at k = 8).
CLIP_NORM = 0.1
#: Epochs over which the ``warmup`` variant ramps the noise up to ``p``.
WARMUP_EPOCHS = 5


class _ClippedAdam(torch.optim.Adam):
    """Adam that clips the global gradient norm first and records it unclipped."""

    def __init__(self, params: Any, lr: float, max_norm: float) -> None:
        super().__init__(params, lr=lr)
        self.max_norm = max_norm
        self.norms: list[float] = []

    def step(self, closure: Any = None) -> Any:
        params = [p for group in self.param_groups for p in group["params"]]
        self.norms.append(float(torch.nn.utils.clip_grad_norm_(params, self.max_norm)))
        return super().step(closure)


def _set_trajectory_noise(model: torch.nn.Module, p: float, position: str) -> None:
    """Rebuild every noisy layer's trajectory QNode at strength ``p``."""
    for layer in model.modules():
        if getattr(layer, "_training_noise_qnode", None) is not None:
            layer._training_noise_qnode = trajectory_noise_qnode(  # type: ignore[assignment]
                layer.qlayer.qnode,  # type: ignore[union-attr]
                p,
                position,  # type: ignore[arg-type]
                channel=layer.noise_channel,  # type: ignore[arg-type]
            )


def _main_grid(base: dict[str, Any]) -> list[dict[str, Any]]:
    """The runs at p = 0.05 after every gate, for one seed and size."""
    runs = [{**base, "method": "noiseless", "k": 0, "variant": "lr0.05"}]
    for lr in ("0.05", "0.02", "0.01"):
        runs.append({**base, "method": "density", "k": 0, "variant": f"lr{lr}"})
    for k in (1, 4, 8):
        for variant in ("lr0.05", "lr0.02", "lr0.01", "clip", "warmup"):
            runs.append({**base, "method": "trajectories", "k": k, "variant": variant})
    return runs


def configurations(
    quick: bool, settings: list[str], *, main: bool = True, density: bool = True
) -> list[dict[str, Any]]:
    """
    Every run of the study, as keyword arguments of :func:`run_one`.

    ``main=False`` leaves out the grid at p = 0.05 after every gate, and
    ``density=False`` the density references at the ``settings``.
    """
    seeds, qubits = ((0,), (4,)) if quick else (range(10), (4, 6))
    runs = []
    for n in qubits:
        for seed in seeds:
            base: dict[str, Any] = {
                "seed": seed,
                "n_qubits": n,
                "noise_level": 0.05,
                "noise_position": "all",
            }
            if main:
                runs.extend(_main_grid(base))
            for p, pos in ((0.01, "all"), (0.05, "end")) if settings and density else ():
                # The density reference to pair the extra settings with.
                runs.append(
                    {
                        **base,
                        "noise_level": p,
                        "noise_position": pos,
                        "method": "density",
                        "k": 0,
                        "variant": "lr0.05",
                    }
                )
            for spec in settings:
                # "trajectories:4:warmup" -> k=4 trajectories with the warm-up.
                method, draws, variant = spec.split(":")
                for p, pos in ((0.01, "all"), (0.05, "end")):
                    runs.append(
                        {
                            **base,
                            "noise_level": p,
                            "noise_position": pos,
                            "method": method,
                            "k": int(draws),
                            "variant": variant,
                        }
                    )
    return runs


def run_one(
    *,
    seed: int,
    n_qubits: int,
    method: str,
    k: int,
    variant: str,
    noise_level: float,
    noise_position: str,
    epochs: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    train, val, test = _split(y, seed)
    Z = torch.tensor(_prepare(X, train, n_qubits), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    options: dict[str, Any] = {}
    if method != "noiseless":
        options = {"noise_level": noise_level, "noise_position": noise_position}
        options["noise_method"] = method
        if method == "trajectories":
            options["noise_trajectories"] = k
    torch.manual_seed(seed)
    model = HybridBinaryClassifier(
        n_qubits, n_qubits, 2, device_name="default.qubit", diff_method="backprop", **options
    )
    lr = float(variant[2:]) if variant.startswith("lr") else 0.05
    optimizer: torch.optim.Optimizer
    if variant == "clip":
        optimizer = _ClippedAdam(model.parameters(), lr=lr, max_norm=CLIP_NORM)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    on_epoch_end = None
    if variant == "warmup":
        # Epoch e (train_model counts from 1) trains at p·min(1, e/WARMUP_EPOCHS).
        def ramp(epoch: int) -> None:
            scale = min(1.0, epoch / WARMUP_EPOCHS)
            _set_trajectory_noise(model, noise_level * scale, noise_position)

        ramp(1)

        def on_epoch_end(record: EpochRecord) -> None:
            ramp(record.epoch + 1)

    start = time.perf_counter()
    history = train_model(
        model,
        FocalLoss(),
        optimizer,
        Z[train],
        yt[train],
        Z[val],
        yt[val],
        max_epochs=epochs,
        batch_size=32,
        monitor="mcc",
        patience=10,
        generator=torch.Generator().manual_seed(seed),
        on_epoch_end=on_epoch_end,
    )
    seconds = time.perf_counter() - start
    threshold = history.best_threshold if history.best_threshold is not None else 0.5
    labels = y[test]
    clean = model.predict_proba(Z[test])
    # Threshold re-tuned on the validation rows under the training noise, as
    # in study_trajectory_noise.py.
    with apply_depolarizing_noise(model, noise_level, position=noise_position):  # type: ignore[arg-type]
        val_noisy = model.predict_proba(Z[val])
        test_noisy = model.predict_proba(Z[test])
    t_noisy = find_optimal_threshold(torch.tensor(y[val]), val_noisy).threshold
    row = {
        "seed": seed,
        "n_qubits": n_qubits,
        "method": method,
        "k": k,
        "variant": variant,
        "noise_level": noise_level,
        "noise_position": noise_position,
        "epochs": history.n_epochs,
        "seconds": seconds,
        "mcc_clean": matthews_corrcoef(labels, (clean >= threshold).long()),
        "mcc_noisy": matthews_corrcoef(labels, (test_noisy >= t_noisy).long()),
    }
    if isinstance(optimizer, _ClippedAdam):
        norms = np.array(optimizer.norms)
        row["grad_norm_median"] = float(np.median(norms))
        row["grad_norm_p99"] = float(np.quantile(norms, 0.99))
        row["grad_norm_max"] = float(norms.max())
        row["clipped_fraction"] = float((norms > CLIP_NORM).mean())
    return row


def _lr(variant: str) -> str:
    return variant if variant.startswith("lr") else "lr0.05"


def summarise(rows: list[dict[str, Any]]) -> str:
    """Collapse count and MCC paired against density at the same seed and lr."""
    density = {
        (r["n_qubits"], r["noise_level"], r["noise_position"], r["seed"], r["variant"]): r
        for r in rows
        if r["method"] == "density"
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["noise_level"], r["noise_position"], r["n_qubits"], r["method"], r["k"])
        groups.setdefault((*key, r["variant"]), []).append(r)
    lines = [
        (
            f"{'p':>5s} {'pos':>4s} {'n':>2s}  {'method':12s} {'k':>2s} {'variant':7s} "
            f"{'collapsed':>9s}  {'MCC noisy':>13s}  {'Δ density':>15s}  {'s/run':>6s}"
        )
    ]
    for (p, pos, n, method, k, variant), sel in sorted(groups.items()):
        noisy = np.array([r["mcc_noisy"] for r in sel])
        collapsed = int((noisy < COLLAPSE_MCC).sum())
        paired = [
            r["mcc_noisy"] - density[(n, p, pos, r["seed"], _lr(variant))]["mcc_noisy"]
            for r in sel
            if (n, p, pos, r["seed"], _lr(variant)) in density
        ]
        delta = (
            f"{np.mean(paired):+.3f} ({np.median(paired):+.3f})"
            if paired and method != "density"
            else ""
        )
        lines.append(
            f"{p:5.2f} {pos:>4s} {n:2d}  {method:12s} {k:2d} {variant:7s} "
            f"{collapsed:4d} / {len(sel):<2d}  {noisy.mean():.3f} ± {noisy.std():.3f}  "
            f"{delta:>15s}  {np.mean([r['seconds'] for r in sel]):6.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--out", type=Path, default=Path("trajectory_collapse_study.jsonl"))
    parser.add_argument("--quick", action="store_true", help="one seed, 4 qubits, 5 epochs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--settings",
        nargs="*",
        default=[],
        metavar="METHOD:K:VARIANT",
        help="also run these at the other noise settings, e.g. trajectories:4:warmup",
    )
    parser.add_argument(
        "--settings-only", action="store_true", help="run only the --settings, not the main grid"
    )
    parser.add_argument(
        "--no-density",
        action="store_true",
        help="no density references at the --settings; pair with --reference instead",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        help="JSON lines whose density runs (lr 0.05) the summary also pairs with, e.g. "
        "the output of study_trajectory_noise.py or of an earlier run of this script",
    )
    parser.add_argument("--summarise", action="store_true", help="only print the summary of --out")
    args = parser.parse_args()
    reference = [
        {"k": 0, "variant": "lr0.05", **row}
        for path in args.reference
        for row in map(json.loads, path.open())
        if row["method"] == "density"
    ]
    if args.summarise:
        print(summarise([json.loads(line) for line in args.out.open()] + reference))
        return
    epochs = 5 if args.quick else 30
    runs = configurations(
        args.quick, args.settings, main=not args.settings_only, density=not args.no_density
    )
    # Longest first, so the slow density runs do not straggle at the end.
    runs.sort(key=lambda r: (r["method"] != "density", -r["n_qubits"]))
    rows: list[dict[str, Any]] = []
    with args.out.open("w") as fh, ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, epochs=epochs, **run) for run in runs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(json.dumps(row), flush=True)
    print(summarise(rows + reference))


if __name__ == "__main__":
    main()
