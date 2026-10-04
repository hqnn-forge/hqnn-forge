"""
tests/test_multiclass_hybrid_classifier.py
==========================================
hqnn_forge.models.MulticlassHybridClassifier on a synthetic 3-class problem:
output shapes, probability semantics under both strategies, gradient flow to
every class head and to the quantum weights, and a short training run that
has to beat chance.  Also: numerical stability of the one-vs-rest
probabilities at saturated logits, init validation, and a checkpoint round trip.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hqnn_forge.models import MulticlassHybridClassifier
from hqnn_forge.utils.checkpoint import load_checkpoint, save_checkpoint

N_FEATURES = 6
N_QUBITS = 4
N_LAYERS = 1
N_CLASSES = 3
BATCH = 9
CPU = {"device_name": "default.qubit", "diff_method": "backprop"}


def _model(**kwargs: object) -> MulticlassHybridClassifier:
    torch.manual_seed(0)
    options: dict[str, object] = {"n_classes": N_CLASSES, **CPU, **kwargs}
    return MulticlassHybridClassifier(
        n_input_features=N_FEATURES,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        **options,  # type: ignore[arg-type]
    )


def _three_class_dataset(
    n_per_class: int = 12, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Three Gaussian blobs at distinct centres in a 6-D feature space."""
    torch.manual_seed(seed)
    centres = torch.tensor(
        [[2.0, 0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0, 0], [0, 0, 2.0, 0, 0, 0]], dtype=torch.float32
    )
    X = torch.cat([centres[c] + 0.4 * torch.randn(n_per_class, N_FEATURES) for c in range(3)])
    y = torch.repeat_interleave(torch.arange(3), n_per_class)
    perm = torch.randperm(X.shape[0])
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Shapes and semantics
# ---------------------------------------------------------------------------


class TestOutputs:
    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_shapes(self, strategy: str) -> None:
        model = _model(strategy=strategy)
        x = torch.randn(BATCH, N_FEATURES)
        assert model(x).shape == (BATCH, N_CLASSES)
        assert model.predict_proba(x).shape == (BATCH, N_CLASSES)
        assert model.predict(x).shape == (BATCH,)

    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_probabilities_are_a_distribution(self, strategy: str) -> None:
        model = _model(strategy=strategy)
        probs = model.predict_proba(torch.randn(BATCH, N_FEATURES))
        assert probs.min().item() >= 0.0
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-6, rtol=0)

    def test_softmax_probabilities_are_the_softmax_of_the_logits(self) -> None:
        model = _model()
        x = torch.randn(BATCH, N_FEATURES)
        with torch.no_grad():
            expected = torch.softmax(model(x), dim=-1)
        torch.testing.assert_close(model.predict_proba(x), expected)

    def test_one_vs_rest_probabilities_are_normalised_sigmoids(self) -> None:
        model = _model(strategy="one_vs_rest")
        x = torch.randn(BATCH, N_FEATURES)
        with torch.no_grad():
            scores = torch.sigmoid(model(x))
        torch.testing.assert_close(model.predict_proba(x), scores / scores.sum(-1, keepdim=True))

    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_predict_is_the_argmax(self, strategy: str) -> None:
        model = _model(strategy=strategy)
        x = torch.randn(BATCH, N_FEATURES)
        labels = model.predict(x)
        assert labels.dtype == torch.long
        assert torch.equal(labels, model.predict_proba(x).argmax(dim=-1))
        assert labels.min().item() >= 0 and labels.max().item() < N_CLASSES

    def test_one_hot_targets(self) -> None:
        model = _model()
        y = torch.tensor([0, 2, 1])
        expected = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
        assert torch.equal(model.one_hot(y), expected)

    def test_one_hot_accepts_integer_valued_floats(self) -> None:
        model = _model()
        assert torch.equal(
            model.one_hot(torch.tensor([0.0, 2.0])), model.one_hot(torch.tensor([0, 2]))
        )

    def test_one_hot_rejects_non_integer_labels(self) -> None:
        with pytest.raises(ValueError, match="integer class labels"):
            _model().one_hot(torch.tensor([0.9, 1.6]))

    def test_one_hot_follows_the_model_dtype(self) -> None:
        model = _model(strategy="one_vs_rest").double()
        X, y = _three_class_dataset(n_per_class=2)
        target = model.one_hot(y)
        assert target.dtype == torch.float64
        loss = nn.BCEWithLogitsLoss()(model(X.double()), target)
        assert loss.dtype == torch.float64

    @pytest.mark.parametrize("method", ["predict_proba", "predict"])
    def test_prediction_runs_in_eval_mode_and_restores_it(
        self, method: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _model(dropout_p=0.5)
        model.train()
        # Record every submodule's mode inside the forward pass, not a
        # function of its output: argmax labels can survive active dropout.
        modes_inside: list[list[bool]] = []
        forward = model.forward

        def recording_forward(x: torch.Tensor) -> torch.Tensor:
            modes_inside.append([m.training for m in model.modules()])
            return forward(x)

        monkeypatch.setattr(model, "forward", recording_forward)
        getattr(model, method)(torch.randn(BATCH, N_FEATURES))
        assert modes_inside and not any(modes_inside[0])
        assert all(m.training for m in model.modules())


class TestOneVsRestSaturation:
    """``predict_proba`` and ``predict`` at logits a trained OvR model reaches."""

    @staticmethod
    def _with_logits(logits: torch.Tensor) -> MulticlassHybridClassifier:
        """A one-vs-rest model whose output is ``logits`` for every input row."""
        model = _model(strategy="one_vs_rest")
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.copy_(logits)
        return model

    def test_all_negative_logits_stay_finite(self) -> None:
        # Every sigmoid underflows to 0 in float32 below about -88.
        logits = torch.tensor([-110.0, -120.0, -130.0])
        probs = self._with_logits(logits).predict_proba(torch.randn(2, N_FEATURES))
        assert torch.isfinite(probs).all()
        # Exact value: sigmoid(l) ≈ exp(l) here, so the ratio is softmax(l).
        expected = torch.softmax(logits.double(), dim=-1).float().expand(2, -1)
        torch.testing.assert_close(probs, expected)

    def test_matches_normalised_sigmoids_in_float64(self) -> None:
        logits = torch.tensor([-3.0, 0.5, 4.0])
        probs = self._with_logits(logits).predict_proba(torch.randn(1, N_FEATURES))
        scores = torch.sigmoid(logits.double())
        torch.testing.assert_close(probs[0].double(), scores / scores.sum(), rtol=1e-6, atol=0)

    def test_predict_uses_the_logits_when_probabilities_saturate(self) -> None:
        model = self._with_logits(torch.tensor([17.0, 20.0, 30.0]))
        x = torch.randn(2, N_FEATURES)
        # In float32 classes 1 and 2 tie in predict_proba, so its argmax
        # returns 1; predict works on the logits and returns 2.
        probs = model.predict_proba(x)
        assert torch.equal(probs[:, 1], probs[:, 2])
        assert torch.equal(probs.argmax(dim=-1), torch.tensor([1, 1]))
        assert torch.equal(model.predict(x), torch.tensor([2, 2]))


# ---------------------------------------------------------------------------
# Parameters and gradients
# ---------------------------------------------------------------------------


class TestParametersAndGradients:
    def test_parameter_count(self) -> None:
        model = _model()
        encoder = N_FEATURES * N_QUBITS + N_QUBITS
        quantum = N_LAYERS * N_QUBITS * 3
        heads = N_CLASSES * N_QUBITS + N_CLASSES
        assert model.count_parameters() == encoder + quantum + heads
        assert model.head.weight.shape == (N_CLASSES, N_QUBITS)
        assert f"n_classes={N_CLASSES}" in model.extra_repr()

    def test_quantum_parameters_do_not_grow_with_n_classes(self) -> None:
        three = _model()
        seven = _model(n_classes=7)
        assert (
            three.quantum_layer.qlayer.weights.numel()
            == seven.quantum_layer.qlayer.weights.numel()
        )
        assert seven.count_parameters() - three.count_parameters() == 4 * (N_QUBITS + 1)

    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_gradients_reach_every_class_head_and_the_quantum_weights(self, strategy: str) -> None:
        model = _model(strategy=strategy)
        X, y = _three_class_dataset(n_per_class=3)
        logits = model(X)
        if strategy == "softmax":
            loss = nn.CrossEntropyLoss()(logits, y)
        else:
            loss = nn.BCEWithLogitsLoss()(logits, model.one_hot(y))
        loss.backward()
        head_grad = model.head.weight.grad
        assert head_grad is not None
        # Every class head row receives a non-zero gradient.
        assert torch.all(head_grad.abs().sum(dim=1) > 0)
        assert model.head.bias.grad is not None and torch.all(model.head.bias.grad != 0)
        q_grad = model.quantum_layer.qlayer.weights.grad
        assert q_grad is not None and q_grad.abs().sum().item() > 0.0
        encoder = model.classical_encoder
        assert isinstance(encoder, nn.Sequential)
        enc_linear = encoder[0]
        assert isinstance(enc_linear, nn.Linear)
        enc_grad = enc_linear.weight.grad
        assert enc_grad is not None and enc_grad.abs().sum().item() > 0.0

    def test_quantum_init_matches_documented_sigma(self) -> None:
        torch.manual_seed(0)
        model = MulticlassHybridClassifier(
            n_input_features=16,
            n_qubits=16,
            n_layers=16,
            n_classes=3,
            **CPU,  # type: ignore[arg-type]
        )
        std = model.quantum_layer.qlayer.weights.detach().std().item()
        assert std == pytest.approx(math.pi / math.sqrt(16 * 16), rel=0.2)

    def test_block_local_init_matches_its_per_layer_sigma(self) -> None:
        """
        sigma_l = pi / sqrt(n_qubits * (L + l)) (#166): from the same seed,
        exactly the ``restricted`` weights times sqrt(L / (L + l)) on layer l.
        """
        n_qubits, n_layers = 16, 4

        def weights(strategy: str) -> torch.Tensor:
            torch.manual_seed(0)
            model = MulticlassHybridClassifier(
                n_input_features=n_qubits,
                n_qubits=n_qubits,
                n_layers=n_layers,
                n_classes=3,
                init_strategy=strategy,
                **CPU,  # type: ignore[arg-type]
            )
            return model.quantum_layer.qlayer.weights.detach()

        factor = torch.tensor([math.sqrt(n_layers / (n_layers + l)) for l in range(n_layers)])
        torch.testing.assert_close(
            weights("block_local"),
            weights("restricted") * factor.view(-1, 1, 1),
            rtol=1e-6,
            atol=0,
        )

    def test_normal_init_uses_init_std(self) -> None:
        torch.manual_seed(0)
        model = MulticlassHybridClassifier(
            n_input_features=16,
            n_qubits=16,
            n_layers=16,
            n_classes=3,
            init_strategy="normal",
            init_std=0.05,
            **CPU,  # type: ignore[arg-type]
        )
        std = model.quantum_layer.qlayer.weights.detach().std().item()
        assert std == pytest.approx(0.05, rel=0.2)


# ---------------------------------------------------------------------------
# Training on the synthetic problem
# ---------------------------------------------------------------------------


class TestTraining:
    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_short_training_beats_chance(self, strategy: str) -> None:
        """
        Three well-separated blobs: 25 Adam steps must take the loss down and
        the training accuracy well above 1/3.
        """
        model = _model(strategy=strategy)
        X, y = _three_class_dataset(n_per_class=12)
        loss_fn = nn.CrossEntropyLoss() if strategy == "softmax" else nn.BCEWithLogitsLoss()
        target = y if strategy == "softmax" else model.one_hot(y)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
        losses = []
        for _ in range(25):
            optimiser.zero_grad()
            loss = loss_fn(model(X), target)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        assert losses[-1] < 0.6 * losses[0]
        accuracy = (model.predict(X) == y).float().mean().item()
        assert accuracy > 0.8


# ---------------------------------------------------------------------------
# Options and validation
# ---------------------------------------------------------------------------


class TestOptions:
    def test_iqp_encoding(self) -> None:
        model = _model(encoding_type="iqp")
        assert model(torch.randn(BATCH, N_FEATURES)).shape == (BATCH, N_CLASSES)

    def test_encoder_bypass(self) -> None:
        torch.manual_seed(0)
        model = MulticlassHybridClassifier(
            n_input_features=N_QUBITS,
            n_qubits=N_QUBITS,
            n_layers=1,
            n_classes=3,
            use_classical_encoder=False,
            **CPU,  # type: ignore[arg-type]
        )
        assert isinstance(model.classical_encoder, nn.Identity)
        assert model(torch.rand(BATCH, N_QUBITS) * 2 - 1).shape == (BATCH, 3)

    def test_bypass_requires_matching_widths(self) -> None:
        with pytest.raises(ValueError, match="must equal n_qubits"):
            MulticlassHybridClassifier(
                n_input_features=5,
                n_qubits=4,
                n_classes=3,
                use_classical_encoder=False,
                **CPU,  # type: ignore[arg-type]
            )

    def test_rejects_fewer_than_two_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes"):
            _model(n_classes=1)

    def test_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            _model(strategy="argmax")

    def test_rejects_unknown_encoding(self) -> None:
        with pytest.raises(ValueError, match="encoding_type"):
            _model(encoding_type="kernel")

    def test_rejects_unknown_init_strategy(self) -> None:
        with pytest.raises(ValueError, match="init_strategy"):
            _model(init_strategy="block-local")

    def test_rejects_init_std_it_would_ignore(self) -> None:
        with pytest.raises(ValueError, match="init_std applies"):
            _model(init_std=0.3)

    @pytest.mark.parametrize("dropout_p", [-0.3, 1.0, 1.5])
    def test_rejects_dropout_outside_unit_interval(self, dropout_p: float) -> None:
        with pytest.raises(ValueError, match="dropout_p"):
            _model(dropout_p=dropout_p)

    def test_rejects_non_positive_init_std(self) -> None:
        with pytest.raises(ValueError, match="init_std must be > 0"):
            _model(init_strategy="normal", init_std=0.0)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_config_rebuilds_the_architecture(self) -> None:
        model = _model(strategy="one_vs_rest", n_classes=5, dropout_p=0.2)
        rebuilt = MulticlassHybridClassifier(**model.get_config())
        assert rebuilt.get_config() == model.get_config()
        assert rebuilt.count_parameters() == model.count_parameters()

    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_round_trip(self, strategy: str, tmp_path: Path) -> None:
        model = _model(strategy=strategy)
        x = torch.randn(BATCH, N_FEATURES)
        path = tmp_path / "multiclass.pt"
        save_checkpoint(model, path)
        loaded = load_checkpoint(path)
        assert isinstance(loaded, MulticlassHybridClassifier)
        assert loaded.get_config() == model.get_config()
        torch.testing.assert_close(loaded.predict_proba(x), model.predict_proba(x))
        assert torch.equal(loaded.predict(x), model.predict(x))
