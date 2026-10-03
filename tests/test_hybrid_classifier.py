"""
tests/test_hybrid_classifier.py
================================
Tests specific to hqnn_forge.models.HybridBinaryClassifier.  The inference,
bypass and encoding-type tests shared with every classifier live in
tests/test_binary_classifiers.py.
"""

from __future__ import annotations

import math

import torch

from hqnn_forge.models import HybridBinaryClassifier


class TestInitStrategies:
    def test_restricted_strategy(self) -> None:
        model = HybridBinaryClassifier(
            n_input_features=4,
            n_qubits=4,
            n_layers=2,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
            init_strategy="restricted",
        )
        assert model is not None

    def test_block_local_is_restricted_tapered_by_layer(self) -> None:
        """
        sigma_l = pi / sqrt(n_qubits * (L + l)) (#166): from the same seed,
        exactly the ``restricted`` weights times sqrt(L / (L + l)) on layer l.
        This checks the model passes the initialiser its whole weight tensor,
        whose first dimension is the depth L.
        """
        n_layers = 3

        def weights(strategy: str) -> torch.Tensor:
            torch.manual_seed(0)
            model = HybridBinaryClassifier(
                n_input_features=4,
                n_qubits=4,
                n_layers=n_layers,
                use_classical_encoder=False,
                device_name="default.qubit",
                diff_method="parameter-shift",
                init_strategy=strategy,
            )
            return model.quantum_layer.qlayer.weights.detach()

        factor = torch.tensor([math.sqrt(n_layers / (n_layers + l)) for l in range(n_layers)])
        torch.testing.assert_close(
            weights("block_local"),
            weights("restricted") * factor.view(-1, 1, 1),
            rtol=1e-6,
            atol=0,
        )
