"""
tests/test_classifier_encodings.py
==================================
``encoding_type="amplitude"`` and ``"reuploading"`` in every classifier (#307).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.utils import load_checkpoint, save_checkpoint

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
N_QUBITS = 3

CLASSIFIERS = [
    pytest.param(HybridBinaryClassifier, {}, id="serial"),
    pytest.param(ParallelHybridClassifier, {}, id="parallel"),
    pytest.param(MulticlassHybridClassifier, {"n_classes": 3}, id="multiclass"),
]

ENCODINGS = [
    pytest.param(
        {"encoding_type": "amplitude"}, AmplitudeEncodingLayer, 2**N_QUBITS, id="amplitude"
    ),
    pytest.param(
        {"encoding_type": "amplitude", "entangler": "hardware_efficient", "readout": "first"},
        AmplitudeEncodingLayer,
        2**N_QUBITS,
        id="amplitude-he-first",
    ),
    pytest.param(
        {"encoding_type": "reuploading"}, DataReuploadingLayer, N_QUBITS, id="reuploading"
    ),
    pytest.param(
        {
            "encoding_type": "reuploading",
            "trainable_input_scaling": True,
            "embedding_rotation": "Y",
            "entangler": "strongly_entangling",
        },
        DataReuploadingLayer,
        N_QUBITS,
        id="reuploading-scaled-y",
    ),
]


def _model(cls: type, extra: dict[str, Any], options: dict[str, Any], **kw: Any) -> Any:
    torch.manual_seed(0)
    args = {"n_input_features": 6, "n_qubits": N_QUBITS, "n_layers": 2, **CPU, **extra, **options}
    return cls(**{**args, **kw})


def _x(n_features: int = 6) -> torch.Tensor:
    return torch.randn(5, n_features, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("cls, extra", CLASSIFIERS)
@pytest.mark.parametrize("options, layer_cls, width", ENCODINGS)
class TestEveryClassifier:
    def test_builds_the_layer_with_the_options(
        self,
        cls: type,
        extra: dict[str, Any],
        options: dict[str, Any],
        layer_cls: type,
        width: int,
    ) -> None:
        model = _model(cls, extra, options)
        assert type(model.quantum_layer) is layer_cls
        layer: Any = model.quantum_layer
        assert model.classical_encoder[0].out_features == width
        assert layer.entangler == options.get("entangler", "ring")
        assert layer.readout == options.get("readout", "all")
        if layer_cls is DataReuploadingLayer:
            assert layer.rotation == options.get("embedding_rotation", "X")
            assert layer.trainable_input_scaling == options.get("trainable_input_scaling", False)
        expected = 1 if options.get("readout") == "first" else N_QUBITS
        assert layer.n_outputs == expected

    def test_quantum_features_are_the_layer_on_the_scaled_encoder_output(
        self,
        cls: type,
        extra: dict[str, Any],
        options: dict[str, Any],
        layer_cls: type,
        width: int,
    ) -> None:
        model = _model(cls, extra, options).eval()
        x = _x()
        with torch.no_grad():
            encoded = model.classical_encoder(x)
            torch.testing.assert_close(
                model._quantum_features(x), model.quantum_layer(encoded * torch.pi), rtol=0, atol=0
            )
            if layer_cls is AmplitudeEncodingLayer:
                # The amplitude layer normalises, so the ·π scaling is irrelevant.
                torch.testing.assert_close(
                    model.quantum_layer(encoded * torch.pi), model.quantum_layer(encoded)
                )

    def test_every_parameter_gets_a_gradient(
        self,
        cls: type,
        extra: dict[str, Any],
        options: dict[str, Any],
        layer_cls: type,
        width: int,
    ) -> None:
        model = _model(cls, extra, options)
        with torch.no_grad():  # away from the init's small angles
            model.quantum_layer.qlayer.weights.uniform_(0, 2 * torch.pi)
        out = model(_x())
        # Positive, unequal weights: a symmetric weighting would sum to zero and
        # leave the head's bias without a gradient by construction.
        (out * torch.linspace(0.5, 1.5, out.numel()).reshape(out.shape)).sum().backward()
        dead = [name for name, p in model.named_parameters() if p.grad is None or not p.grad.any()]
        # Only parameters count_inert_parameters would also call dead may have
        # no gradient: the last layer's angles cut off from a single readout.
        if options.get("readout") == "first":
            dead = [name for name in dead if "quantum_layer" not in name]
        assert not dead, dead

    def test_trains_and_round_trips_a_checkpoint(
        self,
        cls: type,
        extra: dict[str, Any],
        options: dict[str, Any],
        layer_cls: type,
        width: int,
        tmp_path: Path,
    ) -> None:
        model = _model(cls, extra, options)
        x = _x()
        target = torch.zeros_like(model(x))
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        losses = []
        for _ in range(15):
            opt.zero_grad()
            loss = (model(x) - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]
        save_checkpoint(model, tmp_path / "m.pt")
        loaded = load_checkpoint(tmp_path / "m.pt")
        assert loaded.get_config() == model.get_config()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model.eval()(x), rtol=0, atol=0)


@pytest.mark.parametrize("cls, extra", CLASSIFIERS)
class TestAmplitudeWithoutAnEncoder:
    @pytest.mark.parametrize("n_features", [1, 5, 8])
    def test_raw_features_are_padded(
        self, cls: type, extra: dict[str, Any], n_features: int
    ) -> None:
        model = _model(
            cls,
            extra,
            {"encoding_type": "amplitude"},
            use_classical_encoder=False,
            n_input_features=n_features,
        )
        assert model.quantum_layer.n_features == n_features
        model(torch.rand(4, n_features) + 0.1)

    def test_any_diff_method_works_without_input_gradients(
        self, cls: type, extra: dict[str, Any]
    ) -> None:
        model = _model(
            cls,
            extra,
            {"encoding_type": "amplitude"},
            use_classical_encoder=False,
            n_input_features=5,
            diff_method="parameter-shift",
        )
        model(torch.rand(4, 5) + 0.1).sum().backward()
        assert model.quantum_layer.qlayer.weights.grad is not None


@pytest.mark.parametrize("cls, extra", CLASSIFIERS)
@pytest.mark.parametrize(
    "options, match",
    [
        (
            {"encoding_type": "amplitude", "diff_method": "adjoint"},
            "only correct under diff_method='backprop'",
        ),
        (
            {"encoding_type": "amplitude", "diff_method": "parameter-shift"},
            "only correct under diff_method='backprop'",
        ),
        (
            # "auto" on an explicit lightning device resolves to adjoint; the
            # error names what the user passed and what to change.
            {
                "encoding_type": "amplitude",
                "device_name": "lightning.qubit",
                "diff_method": "auto",
            },
            (
                "got diff_method='auto', which resolved to 'adjoint' on "
                "device_name='lightning.qubit'.  Leave device_name='auto'"
            ),
        ),
        (
            {"encoding_type": "amplitude", "use_classical_encoder": False, "n_input_features": 9},
            "takes 1 to 2\\*\\*n_qubits = 8 features",
        ),
        (
            {"encoding_type": "amplitude", "embedding_rotation": "Y"},
            "amplitude embedding has no rotation",
        ),
        ({"encoding_type": "iqp", "embedding_rotation": "Y"}, "IQP embedding has no rotation"),
        ({"encoding_type": "angle", "trainable_input_scaling": True}, "'reuploading' only"),
        ({"encoding_type": "amplitude", "trainable_input_scaling": True}, "'reuploading' only"),
        (
            {"encoding_type": "reuploading", "use_classical_encoder": False},
            "must equal n_qubits",
        ),
        ({"encoding_type": "kernel"}, "choose 'angle', 'iqp', 'amplitude' or 'reuploading'"),
    ],
)
def test_inconsistent_options_raise_at_construction(
    cls: type, extra: dict[str, Any], options: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _model(cls, extra, {}, **options)


@pytest.mark.parametrize("cls, extra", CLASSIFIERS)
@pytest.mark.parametrize("encoding_type", ["amplitude", "reuploading"])
def test_training_noise_reaches_the_layer(
    cls: type, extra: dict[str, Any], encoding_type: str
) -> None:
    model = _model(
        cls, extra, {"encoding_type": encoding_type}, noise_level=0.2, noise_position="end"
    )
    x = _x()
    with torch.no_grad():
        encoded = model.classical_encoder(x) * torch.pi
        model.eval()
        clean = model.quantum_layer(encoded)
        model.train()
        torch.testing.assert_close(model.quantum_layer(encoded), (1 - 4 * 0.2 / 3) * clean)


def test_the_estimator_selects_the_new_encodings() -> None:
    pytest.importorskip("sklearn")
    from hqnn_forge.sklearn import HybridClassifierEstimator

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4))
    y = (X[:, 0] > 0).astype(int)
    for encoding_type in ("amplitude", "reuploading"):
        est = HybridClassifierEstimator(
            n_qubits=2,
            n_layers=1,
            encoding_type=encoding_type,
            device_name="default.qubit",
            diff_method="backprop",
            max_epochs=2,
            random_state=0,
        ).fit(X, y)
        assert est.model_.get_config()["encoding_type"] == encoding_type
        assert est.predict_proba(X).shape == (40, 2)


@pytest.mark.parametrize("cls, extra", CLASSIFIERS)
@pytest.mark.parametrize(
    "encoding, width", [("amplitude", 2**N_QUBITS), ("reuploading", N_QUBITS)]
)
def test_a_custom_encoder_maps_to_the_encoding_width(
    cls: type, extra: dict[str, Any], encoding: str, width: int
) -> None:
    # The custom encoder (#298) feeds the circuit exactly as the built-in one
    # does: amplitude takes 2**n_qubits features, the others n_qubits.
    torch.manual_seed(3)
    module = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.ReLU(), torch.nn.Linear(5, width))
    model = _model(cls, extra, {"encoding_type": encoding}, classical_encoder=module).eval()
    x = _x()
    with torch.no_grad():
        expected = model.quantum_layer(torch.tanh(module(x)) * torch.pi)
        torch.testing.assert_close(model._quantum_features(x), expected, rtol=0, atol=0)
    wrong = torch.nn.Linear(6, N_QUBITS if width != N_QUBITS else 2**N_QUBITS)
    with pytest.raises(ValueError, match=f"to \\(batch, .*{width}\\)"):
        _model(cls, extra, {"encoding_type": encoding}, classical_encoder=wrong)
