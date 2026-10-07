"""
tests/test_classifier_trunk.py
==============================
The shared trunk and ClassifierBase (#226): every classifier builds the same
encoder → quantum layer → dropout from one piece of code, so the multiclass
model accepts every circuit option the binary one does and behaves the same
way under it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from hqnn_forge.models import (
    BinaryClassifierBase,
    ClassifierBase,
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.models._trunk import QuantumTrunk

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
MODELS = [HybridBinaryClassifier, ParallelHybridClassifier, MulticlassHybridClassifier]

TRUNK_OPTIONS = [
    pytest.param({}, id="defaults"),
    pytest.param({"embedding_rotation": "Y"}, id="rotation-y"),
    pytest.param({"entangler": "strongly_entangling"}, id="strongly-entangling"),
    pytest.param({"readout": "first"}, id="readout-first"),
    pytest.param({"encoder_activation": "sigmoid"}, id="sigmoid"),
    pytest.param({"encoding_type": "iqp", "entangler": "strongly_entangling"}, id="iqp"),
    pytest.param(
        {"use_classical_encoder": False, "n_input_features": 3}, id="no-classical-encoder"
    ),
    pytest.param({"init_strategy": "normal", "init_std": 0.3}, id="normal-init"),
    pytest.param({"noise_level": 0.1, "noise_position": "end"}, id="noise-density"),
    pytest.param(
        {"noise_level": 0.1, "noise_method": "trajectories", "noise_trajectories": 2},
        id="noise-trajectories",
    ),
]


def _build(cls: type, **options: Any) -> Any:
    torch.manual_seed(0)
    return cls(**{"n_input_features": 5, "n_qubits": 3, "n_layers": 2, **CPU, **options})


def _input(model: Any) -> torch.Tensor:
    return torch.randn(4, model.n_input_features, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("options", TRUNK_OPTIONS)
class TestMulticlassHasTheBinaryTrunk:
    def test_same_quantum_features_for_the_same_weights(self, options: dict[str, Any]) -> None:
        # Same circuit, encoder and scaling: with the binary model's trunk
        # weights copied in, the features the heads read are bit-identical.
        binary = _build(HybridBinaryClassifier, **options)
        multi = _build(MulticlassHybridClassifier, **options)
        multi.classical_encoder.load_state_dict(binary.classical_encoder.state_dict())
        multi.quantum_layer.load_state_dict(binary.quantum_layer.state_dict())
        binary.eval()
        multi.eval()
        x = _input(binary)
        with torch.no_grad():
            torch.testing.assert_close(
                multi._quantum_features(x), binary._quantum_features(x), rtol=0, atol=0
            )

    def test_same_config_for_the_trunk_options(self, options: dict[str, Any]) -> None:
        binary = _build(HybridBinaryClassifier, **options).get_config()
        multi = _build(MulticlassHybridClassifier, **options).get_config()
        shared = set(binary) & set(multi)
        assert set(binary) - shared == set()  # every binary option exists on multiclass
        assert {k: binary[k] for k in shared} == {k: multi[k] for k in shared}

    def test_the_head_reads_the_readout_width(self, options: dict[str, Any]) -> None:
        multi = _build(MulticlassHybridClassifier, n_classes=4, **options)
        width = 1 if options.get("readout") == "first" else 3
        assert multi.head.in_features == multi.quantum_layer.n_outputs == width
        x = _input(multi)
        assert multi(x).shape == (4, 4)
        assert multi.predict_proba(x).sum(-1).allclose(torch.ones(4))

    def test_same_quantum_init_from_the_same_seed(self, options: dict[str, Any]) -> None:
        binary = _build(HybridBinaryClassifier, **options)
        multi = _build(MulticlassHybridClassifier, **options)
        for model in (binary, multi):
            torch.manual_seed(5)
            model._initialise_quantum_weights()
        torch.testing.assert_close(
            multi.quantum_layer.qlayer.weights, binary.quantum_layer.qlayer.weights, rtol=0, atol=0
        )


class TestMulticlassTrainingNoise:
    def test_train_mode_runs_the_noisy_circuit_and_eval_mode_does_not(self) -> None:
        noisy = _build(MulticlassHybridClassifier, noise_level=0.2, noise_position="end")
        clean = _build(MulticlassHybridClassifier)
        clean.load_state_dict(noisy.state_dict())
        x = _input(noisy)
        with torch.no_grad():
            noisy.eval()
            clean.eval()
            torch.testing.assert_close(noisy(x), clean(x), rtol=0, atol=0)
            noisy.train()
            # "end" damps every <Z_i> by exactly 1 - 4p/3 before the heads.
            torch.testing.assert_close(
                noisy._quantum_features(x), (1 - 4 * 0.2 / 3) * clean._quantum_features(x)
            )


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize(
    "options, match",
    [
        ({"encoder_activation": "relu"}, "encoder_activation must be 'tanh' or 'sigmoid'"),
        ({"init_strategy": "xavier"}, "init_strategy must be"),
        ({"init_std": 0.3}, "init_std applies to init_strategy='normal' only"),
        ({"init_strategy": "normal", "init_std": 0.0}, "init_std must be > 0"),
        ({"use_classical_encoder": False}, "must equal n_qubits"),
        (
            {
                "use_classical_encoder": False,
                "n_input_features": 3,
                "encoder_activation": "sigmoid",
            },
            "encoder_activation applies with use_classical_encoder=True only",
        ),
        ({"encoding_type": "iqp", "embedding_rotation": "Y"}, "IQP embedding has no rotation"),
        ({"encoding_type": "kernel"}, "Unsupported encoding_type"),
        ({"noise_method": "kraus"}, "noise_method must be"),
    ],
)
def test_every_model_rejects_the_same_trunk_options(
    cls: type, options: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _build(cls, **options)


@pytest.mark.parametrize("cls", [HybridBinaryClassifier, ParallelHybridClassifier])
def test_binary_models_keep_their_unvalidated_dropout_p(cls: type) -> None:
    # The refactor leaves dropout_p exactly as the binary models took it
    # before: handed to nn.Dropout unchecked, so 1.0 and negative values
    # build (and such checkpoints load).  Tightening this is #404.
    assert isinstance(_build(cls, dropout_p=1.0).dropout, nn.Dropout)
    assert isinstance(_build(cls, dropout_p=-0.1).dropout, nn.Identity)
    with pytest.raises(ValueError, match="dropout probability has to be between 0 and 1"):
        _build(cls, dropout_p=1.5)


@pytest.mark.parametrize("dropout_p", [1.0, -0.1])
def test_multiclass_keeps_its_dropout_range(dropout_p: float) -> None:
    with pytest.raises(ValueError, match=r"dropout_p must be in \[0, 1\)"):
        _build(MulticlassHybridClassifier, dropout_p=dropout_p)


class TestOneImplementation:
    @pytest.mark.parametrize("cls", MODELS)
    def test_the_trunk_and_bookkeeping_are_inherited_not_copied(self, cls: type) -> None:
        assert issubclass(cls, QuantumTrunk) and issubclass(cls, ClassifierBase)
        for name in (
            "_build_trunk",
            "_initialise_quantum_weights",
            "_quantum_features",
            "_eval_logits",
            "count_parameters",
            "get_config",
        ):
            assert name not in vars(cls), f"{cls.__name__} defines its own {name}"

    def test_binary_heads_share_the_binary_predict(self) -> None:
        for cls in (HybridBinaryClassifier, ParallelHybridClassifier):
            assert issubclass(cls, BinaryClassifierBase)
        assert not issubclass(MulticlassHybridClassifier, BinaryClassifierBase)

    @pytest.mark.parametrize("cls", MODELS)
    def test_eval_logits_restores_the_training_flags(self, cls: type) -> None:
        model = _build(cls, dropout_p=0.5)
        model.train()
        model.dropout.eval()  # a mixed state that a blanket train() would lose
        a = model._eval_logits(_input(model))
        b = model._eval_logits(_input(model))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert model.training and not model.dropout.training
        assert not a.requires_grad

    def test_state_dict_keys_are_unchanged(self) -> None:
        # Checkpoints load by these keys; the trunk must not nest them.
        for cls in MODELS:
            top = {k.split(".")[0] for k in _build(cls).state_dict()}
            expected = {"classical_encoder", "quantum_layer", "head"}
            if cls is ParallelHybridClassifier:
                expected.add("classical_branch")
            assert top == expected, cls.__name__

    def test_no_dropout_module_without_dropout(self) -> None:
        for cls in MODELS:
            assert isinstance(_build(cls).dropout, nn.Identity)
            assert isinstance(_build(cls, dropout_p=0.3).dropout, nn.Dropout)


@pytest.mark.parametrize("cls", MODELS)
def test_every_model_keeps_a_custom_encoder_as_given(cls: type) -> None:
    # The trunk owns the custom encoder, so the multiclass model gets it too,
    # and no model's classical init overwrites its (possibly pretrained) weights.
    encoder = nn.Linear(5, 3)
    with torch.no_grad():
        encoder.weight.fill_(0.25)
        encoder.bias.fill_(-0.5)
    model = _build(cls, classical_encoder=encoder, init_seed=3)
    assert model.classical_encoder[0] is encoder
    assert torch.equal(encoder.weight, torch.full((3, 5), 0.25))
    assert torch.equal(encoder.bias, torch.full((3,), -0.5))
    x = _input(model)
    with torch.no_grad():
        expected = model.quantum_layer(torch.tanh(encoder(x)) * torch.pi)
        torch.testing.assert_close(model._quantum_features(x), expected, rtol=0, atol=0)
