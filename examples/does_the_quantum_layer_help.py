"""
examples/does_the_quantum_layer_help.py
=======================================
Does a quantum layer help on *your* dataset, per parameter?

This walks through the comparison step by step: a hybrid model and a purely
classical control with the same number of parameters, trained the same way
on the same stratified folds, scored by MCC at a tuned threshold, and
compared with a paired test that reports a verdict in plain words.
``hqnn_forge.benchmark.run_benchmark`` runs the same procedure (with early
stopping) over several datasets in one call; here every step is spelled out.

It runs out of the box on a synthetic imbalanced dataset.  To use your own
data, replace ``load_data`` (step 1).

Requires the scikit-learn extra:

    pip install "hqnn-forge[sklearn]"

Usage
-----
    python examples/does_the_quantum_layer_help.py

About three minutes on a laptop CPU, almost all of it simulating the circuit.
"""

from __future__ import annotations

import numpy as np
import torch

from hqnn_forge.evaluation import (
    find_optimal_threshold,
    matthews_corrcoef,
    parameter_efficiency,
    rank_biserial_correlation,
    wilcoxon_signed_rank,
)
from hqnn_forge.preprocessing import smote, stratified_kfold
from hqnn_forge.sklearn import HybridClassifierEstimator
from hqnn_forge.training import train_model
from hqnn_forge.utils import FocalLoss, classical_baseline

SEED = 0
# With n untied folds the test below cannot go under p = 2 / 2**n: 6 folds are
# the fewest that can reach 0.05, and a fold where both models score the same
# drops out.  8 leaves room for ties.
N_FOLDS = 8
ALPHA = 0.05
LR, MAX_EPOCHS, BATCH_SIZE = 0.02, 12, 64  # the same for both models


# ── 1. Your data ───────────────────────────────────────────────────────────
def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Replace this with your own data: ``X`` of shape (n_samples, n_features),
    ``y`` of 0/1 labels with 1 the rare class.  Tabular features, or
    embeddings from a pretrained model for images or text.

    The stand-in: 600 rows, 6 features, about 8 % positives, from a rule with
    an interaction term, so it is learnable but not trivially linear.
    """
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((600, 6))
    score = X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3] + 0.5 * rng.standard_normal(600)
    y = (score > np.quantile(score, 0.92)).astype(np.int64)
    return X, y


def standardise(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    # Fitted on the training rows only: statistics computed over test rows
    # would leak them into training, and the test score would be optimistic.
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1.0
    return [(a - mean) / std for a in (train, *others)]


def score_at_tuned_threshold(
    prob_val: np.ndarray, y_val: np.ndarray, prob_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, float]:
    # Why not 0.5: with 8 % positives a model's probabilities rarely cross
    # 0.5 for the rare class, and predicting "negative" for everything is
    # already 92 % accurate.  The threshold that maximises MCC is found on the
    # validation rows and then applied, unchanged, to the test rows -- tuning
    # it on the test rows would report the best case, not an estimate.
    threshold = find_optimal_threshold(y_val, prob_val, "mcc").threshold
    return matthews_corrcoef(y_test, (prob_test >= threshold).astype(np.int64)), threshold


def main() -> None:
    X, y = load_data()
    print(f"{len(y)} samples, {X.shape[1]} features, {y.mean():.1%} positive\n")

    scores: dict[str, list[float]] = {"hybrid": [], "control": []}
    n_params: dict[str, int] = {}

    # ── 2. The same stratified folds for both models ──────────────────────
    # Stratified, so every test fold keeps the rare class; the same folds for
    # both, so each fold gives one *paired* difference for the test in step 4.
    for k, (train, test) in enumerate(stratified_kfold(y, N_FOLDS, random_state=SEED)):
        # A validation split inside the training part tunes the threshold; the
        # test fold is touched once, for the final score.
        inner_train, val = stratified_kfold(y[train], 5, random_state=SEED + k)[0]
        fit_idx, val_idx = train[inner_train], train[val]
        X_fit, X_val, X_test = standardise(X[fit_idx], X[val_idx], X[test])

        # SMOTE only the rows the models learn from, after scaling (it measures
        # distances).  Oversampling before the split would put synthetic
        # copies of test positives into training.
        X_fit, y_fit, _ = smote(X_fit, y[fit_idx], k_neighbors=5, random_state=SEED + k)

        # ── The hybrid model ──
        hybrid = HybridClassifierEstimator(
            n_qubits=4,
            n_layers=2,
            validation_fraction=0.0,  # the validation split is ours, shared with the control
            random_state=SEED + k,
            lr=LR,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
        ).fit(X_fit, y_fit)
        mcc_h, thr_h = score_at_tuned_threshold(
            hybrid.predict_proba(X_val)[:, 1],
            y[val_idx],
            hybrid.predict_proba(X_test)[:, 1],
            y[test],
        )

        # ── The classical control ──
        # Same parameter count as the hybrid, built fresh and trained from
        # scratch on the same data with the same settings.  Re-using the
        # hybrid's trained classical layers, or just switching the circuit off,
        # would measure a model trained *with* the circuit -- not what a
        # classical model achieves on its own.  Its initial weights are seeded
        # from the hybrid's init_seed (random_state), so the fold reproduces.
        control = classical_baseline(hybrid.model_)
        train_model(
            control,
            FocalLoss(),  # the estimator's default loss
            torch.optim.Adam(control.parameters(), lr=LR),
            torch.from_numpy(X_fit.astype(np.float32)),
            torch.from_numpy(y_fit.astype(np.float32)),
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            generator=torch.Generator().manual_seed(SEED + k),  # a seeded batch order
        )
        control.eval()

        prob_val, prob_test = (
            control.predict_proba(torch.from_numpy(rows.astype(np.float32))).numpy()
            for rows in (X_val, X_test)
        )
        mcc_c, thr_c = score_at_tuned_threshold(prob_val, y[val_idx], prob_test, y[test])

        scores["hybrid"].append(mcc_h)
        scores["control"].append(mcc_c)
        n_params = {
            "hybrid": hybrid.model_.count_parameters(),
            "control": control.count_parameters(),
        }
        print(
            f"fold {k + 1}/{N_FOLDS}: MCC hybrid {mcc_h:+.3f} (threshold {thr_h:.2f}), "
            f"control {mcc_c:+.3f} (threshold {thr_c:.2f})"
        )

    # ── 3. Score, parameters, score per parameter ─────────────────────────
    print(f"\n{'model':<8} {'params':>6} {'MCC':>15} {'MCC per 1k params':>18}")
    for name in ("hybrid", "control"):
        s = np.array(scores[name])
        print(
            f"{name:<8} {n_params[name]:>6} {s.mean():>8.3f} ± {s.std(ddof=1):.3f} "
            f"{parameter_efficiency(n_params[name], s.mean()):>18.2f}"
        )

    # ── 4. Paired test and verdict ─────────────────────────────────────────
    # Paired over folds, because both models were scored on the same rows: a
    # hard fold is hard for both, and pairing removes that shared variation.
    diff = np.array(scores["hybrid"]) - np.array(scores["control"])
    print(f"\nmean MCC difference, hybrid − control: {diff.mean():+.3f}")
    if np.all(diff == 0):
        print("Verdict: identical scores on every fold; the data cannot tell the models apart.")
        return
    wilcoxon = wilcoxon_signed_rank(scores["hybrid"], scores["control"])
    effect = rank_biserial_correlation(scores["hybrid"], scores["control"])
    print(
        f"Wilcoxon signed-rank p = {wilcoxon.p_value:.4f} "
        f"(smallest possible with {wilcoxon.n} untied folds: {wilcoxon.min_p_value:.4f}), "
        f"rank-biserial r = {effect:+.2f}"
    )
    if wilcoxon.min_p_value >= ALPHA:
        verdict = (
            f"inconclusive by construction -- with {wilcoxon.n} untied folds the test "
            f"cannot reach p < {ALPHA} whatever the scores; use more folds."
        )
    elif wilcoxon.p_value < ALPHA:
        better = "hybrid" if diff.mean() > 0 else "classical control"
        verdict = (
            f"the {better} scores higher on these folds (p = {wilcoxon.p_value:.3f} < {ALPHA}), "
            f"at the same parameter budget."
        )
    else:
        verdict = (
            f"no detectable difference (p = {wilcoxon.p_value:.3f} ≥ {ALPHA}).  That is not "
            f"evidence that the models are equal -- only that {N_FOLDS} folds of this data "
            f"do not separate them.  At equal MCC the simpler, faster classical model wins."
        )
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
