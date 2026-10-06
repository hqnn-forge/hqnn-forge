"""
examples/noise_aware_training.py
================================
Does training *under* depolarizing noise give a model that is more robust
when deployed on a noisy device?

Two copies of the same ``HybridBinaryClassifier`` are trained on the same
synthetic imbalanced data from the same initial weights: one noiseless, one
with ``noise_level=TRAIN_NOISE`` (the channel is applied in train mode only,
so both are evaluated noiselessly by default).  Both are then evaluated under
the post-hoc noise sweep from ``hqnn_forge.noise`` over a range of ``p`` and
the MCC-at-0.5 of each is printed side by side.

The run is deliberately small (4 qubits, one layer, a few hundred samples) so
it finishes in well under a minute on a laptop; the shape of the comparison is the
point, not the numbers.

Run with:
    python examples/noise_aware_training.py
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from hqnn_forge.evaluation import matthews_corrcoef
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import noise_sweep
from hqnn_forge.training import EpochRecord, train_model
from hqnn_forge.utils import FocalLoss

SEED = 0
N_FEATURES = 6
N_QUBITS = 4
N_LAYERS = 1
N_SAMPLES = 240
POSITIVE_RATE = 0.2
EPOCHS = 12
BATCH_SIZE = 24
TRAIN_NOISE = 0.05
SWEEP = (0.0, 0.02, 0.05, 0.1, 0.2)


def make_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two Gaussian classes in N_FEATURES dimensions, POSITIVE_RATE positives."""
    g = torch.Generator().manual_seed(seed)
    n_pos = int(n * POSITIVE_RATE)
    neg = torch.randn(n - n_pos, N_FEATURES, generator=g)
    pos = torch.randn(n_pos, N_FEATURES, generator=g) + 1.2
    X = torch.cat([neg, pos])
    y = torch.cat([torch.zeros(n - n_pos), torch.ones(n_pos)])
    perm = torch.randperm(n, generator=g)
    return X[perm], y[perm]


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, seed: int) -> None:
    """Focal-loss Adam training for EPOCHS epochs; the batch order is fixed by ``seed``."""

    def log(record: EpochRecord) -> None:
        print(f"    epoch {record.epoch:2d}/{EPOCHS}  loss {record.train_loss:.4f}")

    train_model(
        model,
        FocalLoss(alpha=0.25, gamma=2.0),
        torch.optim.Adam(model.parameters(), lr=0.05),
        X,
        y,
        max_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        generator=torch.Generator().manual_seed(seed),
        on_epoch_end=log,
    )


def main() -> None:
    torch.manual_seed(SEED)
    X_train, y_train = make_data(N_SAMPLES, SEED)
    X_test, y_test = make_data(N_SAMPLES // 2, SEED + 1)

    # Same initial weights for both models: build the noiseless one, deep-copy
    # its state into the noisy one.
    torch.manual_seed(SEED)
    # Training noise runs on default.mixed with backprop; the noiseless path
    # uses default.qubit with backprop too, so both models differentiate alike.
    clean = HybridBinaryClassifier(
        N_FEATURES, N_QUBITS, N_LAYERS, device_name="default.qubit", diff_method="backprop"
    )
    noisy = HybridBinaryClassifier(
        N_FEATURES,
        N_QUBITS,
        N_LAYERS,
        noise_level=TRAIN_NOISE,
        device_name="default.qubit",
        diff_method="backprop",
    )
    noisy.load_state_dict(copy.deepcopy(clean.state_dict()))

    print(f"Training noiseless model ({clean.count_parameters()} parameters)")
    train(clean, X_train, y_train, SEED)
    print(f"Training noise-aware model (noise_level={TRAIN_NOISE})")
    train(noisy, X_train, y_train, SEED)

    def score(y_true: torch.Tensor, probs: torch.Tensor) -> float:
        return matthews_corrcoef(y_true, probs >= 0.5)

    clean_points = noise_sweep(clean, X_test, SWEEP, y=y_test, score_fn=score)
    noisy_points = noise_sweep(noisy, X_test, SWEEP, y=y_test, score_fn=score)

    print()
    print("Test MCC under post-hoc depolarizing noise (position='all'):")
    print(f"  {'p':>6}  {'noiseless-trained':>18}  {'noise-aware-trained':>20}")
    for c, n in zip(clean_points, noisy_points):
        print(f"  {c.p:>6.2f}  {c.score:>18.3f}  {n.score:>20.3f}")


if __name__ == "__main__":
    main()
