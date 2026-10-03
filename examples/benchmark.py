"""
examples/benchmark.py
=====================
Hybrid model against its matched classical control, per dataset, with
:func:`hqnn_forge.benchmark.run_benchmark`.

By default it runs on a synthetic imbalanced dataset, so it works out of the
box.  ``--credit-card`` adds the Kaggle credit-card data (see
``hqnn_forge.data.load_credit_card_fraud``); ``--max-rows`` keeps a
stratified random subset of it, since training the hybrid on all 284,807 rows
takes days of circuit simulation.  A subset keeps the positive rate.

Usage
-----
    python examples/benchmark.py
    python examples/benchmark.py --credit-card --max-rows 20000 --csv results.csv
    python examples/benchmark.py --record run.json   # see hqnn_forge.experiment

The default run takes about four minutes on a laptop CPU, nearly all of it
simulating the circuit; the control trains in under a second.
"""

from __future__ import annotations

import argparse

import numpy as np

from hqnn_forge.benchmark import run_benchmark, write_csv
from hqnn_forge.data import load_credit_card_fraud
from hqnn_forge.models import HybridBinaryClassifier


def synthetic(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` rows, 10 features, about 5 % positives from a noisy linear rule."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10))
    score = X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + 0.5 * rng.standard_normal(n)
    y = (score > np.quantile(score, 0.95)).astype(np.int64)
    return X, y


def stratified_subset(
    X: np.ndarray, y: np.ndarray, max_rows: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """At most ``max_rows`` rows, drawn per class so the positive rate is kept."""
    if y.size <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    keep = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == c), round(max_rows * np.mean(y == c)), replace=False)
            for c in (0, 1)
        ]
    )
    keep.sort()
    return X[keep], y[keep]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid model against its matched classical control."
    )
    parser.add_argument(
        "--credit-card", action="store_true", help="add the Kaggle credit-card data"
    )
    parser.add_argument("--max-rows", type=int, default=20_000, help="rows kept per real dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--seeds", type=int, default=1, help="initialisation seeds per fold (fold score = mean)"
    )
    parser.add_argument(
        "--noise",
        type=float,
        nargs="*",
        help="also score the hybrid under these inference-time depolarising probabilities",
    )
    parser.add_argument("--csv", help="also write the table to this file")
    parser.add_argument(
        "--record", help="write an experiment record (JSON) for reproducing the run here"
    )
    args = parser.parse_args()

    datasets = {"synthetic": synthetic()}
    if args.credit_card:
        data = load_credit_card_fraud(drop_time=True)
        datasets["credit-card"] = stratified_subset(data.X, data.y, args.max_rows)

    def hybrid(n_input_features: int) -> HybridBinaryClassifier:
        return HybridBinaryClassifier(n_input_features, args.qubits, args.layers)

    result = run_benchmark(
        datasets,
        hybrid,
        n_splits=args.folds,
        max_epochs=args.epochs,
        record_path=args.record,
        n_seeds=args.seeds,
        noise_levels=args.noise,
    )

    print(f"{'dataset':<12} {'model':<8} {'params':>6} {'MCC':>14} {'MCC/kP':>7} {'train s':>8}")
    for r in result.records:
        mcc = f"{r['mcc_mean']:.3f} ± {r['mcc_std']:.3f}"
        print(
            f"{r['dataset']:<12} {r['model']:<8} {r['n_parameters']:>6} {mcc:>14} "
            f"{r['mcc_per_kparam']:>7.2f} {r['train_seconds']:>8.1f}"
        )
    for r in result.records[::2]:
        print(
            f"{r['dataset']}: Wilcoxon p = {r['wilcoxon_p']:.4f} (smallest possible with "
            f"{r['n_folds']} folds: {r['wilcoxon_min_p']:.4f}), rank-biserial r = "
            f"{r['rank_biserial']:+.2f}"
        )
    for row in result.noise:
        print(
            f"{row['dataset']}: noise p = {row['noise_level']:.2f}: hybrid MCC "
            f"{row['hybrid_mcc_mean']:.3f} vs control {row['control_mcc_mean']:.3f}, "
            f"p(hybrid better) = {row['p_hybrid_better']:.4f}"
        )
    for dataset, summary in result.noise_summary.items():
        if not summary["better_noiseless"]:
            print(f"{dataset}: the hybrid is not significantly better even without noise.")
        elif summary["lost_at"] is None:
            print(f"{dataset}: the hybrid stays significantly better across the sweep.")
        else:
            print(
                f"{dataset}: the hybrid stops being significantly better at p = {summary['lost_at']}."
            )
    if args.csv:
        write_csv(result.records, args.csv)
        print(f"written to {args.csv}")


if __name__ == "__main__":
    main()
