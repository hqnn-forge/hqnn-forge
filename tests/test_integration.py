"""
tests/test_integration.py
==========================
End-to-end integration test for the hqnn-forge pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.optim as optim

from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier
from hqnn_forge.preprocessing import PCANormalizer
from hqnn_forge.utils import FocalLoss, compute_class_weights, weighted_bce_loss

N_QUBITS = 4
N_LAYERS = 2
N_RAW_FEATURES = 12


@pytest.fixture
def synthetic_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_train, n_test = 80, 20

    X_train = rng.standard_normal((n_train, N_RAW_FEATURES))
    X_test = rng.standard_normal((n_test, N_RAW_FEATURES))

    y_train = (rng.random(n_train) < 0.10).astype(np.float64)
    y_test = (rng.random(n_test) < 0.10).astype(np.float64)
    y_train[0] = 1.0
    y_test[0] = 1.0

    return X_train, y_train, X_test, y_test


class TestEndToEndPipeline:
    @pytest.mark.slow
    def test_training_loss_decreases(
        self, synthetic_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X_train_np, y_train_np, _, _ = synthetic_dataset

        pca = PCANormalizer(n_components=N_QUBITS, scale_to_pi=True)
        X_train = pca.fit_transform(X_train_np)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)

        model = HybridBinaryClassifier(
            n_input_features=N_QUBITS,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
            init_strategy="restricted",
        )

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=0.05)

        losses = []
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            logits = model(X_train).squeeze(-1)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]

    @pytest.mark.slow
    def test_class_weighted_training_loss_decreases(
        self, synthetic_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X_train_np, y_train_np, _, _ = synthetic_dataset

        pca = PCANormalizer(n_components=N_QUBITS, scale_to_pi=True)
        X_train = pca.fit_transform(X_train_np)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)
        class_weights = compute_class_weights(y_train)
        assert class_weights[1] > class_weights[0]

        model = HybridBinaryClassifier(
            n_input_features=N_QUBITS,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
            init_strategy="restricted",
        )
        optimizer = optim.Adam(model.parameters(), lr=0.05)

        losses = []
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            logits = model(X_train).squeeze(-1)
            loss = weighted_bce_loss(logits, y_train, class_weights)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0]

    @pytest.mark.slow
    def test_predict_proba_after_training(
        self, synthetic_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X_train_np, y_train_np, X_test_np, _ = synthetic_dataset

        pca = PCANormalizer(n_components=N_QUBITS, scale_to_pi=True)
        X_train = pca.fit_transform(X_train_np)
        X_test = pca.transform(X_test_np)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)

        model = HybridBinaryClassifier(
            n_input_features=N_QUBITS,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
        )

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=0.05)

        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            loss = loss_fn(model(X_train).squeeze(-1), y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        probs = model.predict_proba(X_test)

        assert probs.min().item() >= 0.0 - 1e-6
        assert probs.max().item() <= 1.0 + 1e-6
        assert probs.shape == (len(X_test_np),)

    @pytest.mark.parametrize("model_cls", [HybridBinaryClassifier, ParallelHybridClassifier])
    def test_pca_scaled_angles_stay_within_pi(
        self,
        synthetic_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        model_cls: type[HybridBinaryClassifier | ParallelHybridClassifier],
    ) -> None:
        """PCANormalizer(scale_to_pi=True) output must reach a bypassed circuit within ±π."""
        X_train_np, _, _, _ = synthetic_dataset
        X_train = PCANormalizer(n_components=N_QUBITS, scale_to_pi=True).fit_transform(X_train_np)

        model = model_cls(
            n_input_features=N_QUBITS,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
        )

        captured: list[torch.Tensor] = []
        handle = model.quantum_layer.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        try:
            model(X_train)
        finally:
            handle.remove()

        assert captured[0].abs().max().item() <= torch.pi
