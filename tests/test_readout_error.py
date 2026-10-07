"""
tests/test_readout_error.py
===========================
The asymmetric readout error (#358): its affine effect on ⟨Z⟩ against an
independent density-matrix reference, post hoc, in a sweep, and in training.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import torch
from torch import nn

from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    IQPEncodingLayer,
    QuantumEncodingLayer,
)
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.noise import (
    apply_depolarizing_noise,
    apply_readout_error,
    noise_sweep,
    readout_error_map,
    validate_readout_error,
)
from hqnn_forge.utils import load_checkpoint, save_checkpoint

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
P01, P10 = 0.02, 0.09


def _layer(**kwargs: Any) -> QuantumEncodingLayer:
    torch.manual_seed(0)
    return QuantumEncodingLayer(n_qubits=3, n_layers=2, **{**CPU, **kwargs})


def _x() -> torch.Tensor:
    return torch.rand(5, 3, generator=torch.Generator().manual_seed(1)) * 4 - 2


def _confusion_kraus(p01: float, p10: float) -> list[torch.Tensor]:
    # The classical confusion matrix as a channel: 0 → 1 with p01, 1 → 0 with p10.
    return [
        torch.tensor([[math.sqrt(1 - p01), 0.0], [0.0, 0.0]], dtype=torch.complex128),
        torch.tensor([[0.0, 0.0], [math.sqrt(p01), 0.0]], dtype=torch.complex128),
        torch.tensor([[0.0, 0.0], [0.0, math.sqrt(1 - p10)]], dtype=torch.complex128),
        torch.tensor([[0.0, math.sqrt(p10)], [0.0, 0.0]], dtype=torch.complex128),
    ]


def _density_reference(layer: QuantumEncodingLayer, x: torch.Tensor, p01: float, p10: float):
    """The layer's circuit on default.mixed with the confusion channel on every wire at the end."""
    kraus = [k.numpy() for k in _confusion_kraus(p01, p10)]
    base = qml.QNode(
        layer.qlayer.qnode.func,
        qml.device("default.mixed", wires=3),
        diff_method="backprop",
        interface="torch",
    )

    def confusion(wires: object) -> None:
        qml.QubitChannel(kraus, wires=wires)

    noisy = qml.noise.insert(base, confusion, (), position="end")
    weights = layer.qlayer.weights.detach().double()
    return torch.stack([torch.stack(noisy(xi.double(), weights)) for xi in x])


class TestMap:
    def test_matches_the_confusion_channel_on_every_qubit(self) -> None:
        layer = _layer()
        x = _x()
        with torch.no_grad(), apply_readout_error(layer, P01, P10):
            out = layer(x).double()
        reference = _density_reference(layer, x, P01, P10)
        torch.testing.assert_close(out, reference, atol=1e-6, rtol=0)

    def test_the_map_is_the_stated_affine_function(self) -> None:
        z = torch.linspace(-1, 1, 11, dtype=torch.float64)
        expected = (1 - P01 - P10) * z + (P10 - P01)
        torch.testing.assert_close(readout_error_map(z, P01, P10), expected)
        # The extremes: a certain 0 reads 1 with p01, a certain 1 reads 0 with p10.
        assert readout_error_map(torch.tensor(1.0), P01, P10).item() == pytest.approx(1 - 2 * P01)
        assert readout_error_map(torch.tensor(-1.0), P01, P10).item() == pytest.approx(
            -1 + 2 * P10
        )

    def test_symmetric_case_is_bit_flip_at_the_end(self) -> None:
        p = 0.07
        layer = _layer()
        x = _x()
        with torch.no_grad():
            with apply_readout_error(layer, p, p):
                readout = layer(x)
            with apply_depolarizing_noise(layer, p, position="end", channel="bit_flip"):
                flipped = layer(x)
        torch.testing.assert_close(readout, flipped.to(readout.dtype), atol=1e-6, rtol=0)

    def test_first_qubit_readout(self) -> None:
        layer = _layer(readout="first")
        x = _x()
        with torch.no_grad():
            clean = layer(x)
            with apply_readout_error(layer, P01, P10):
                noisy = layer(x)
        torch.testing.assert_close(noisy, readout_error_map(clean, P01, P10))


class TestPostHoc:
    def test_composes_with_a_noise_block(self) -> None:
        layer = _layer()
        x = _x()
        with torch.no_grad(), apply_depolarizing_noise(layer, 0.05):
            noisy = layer(x)
            with apply_readout_error(layer, P01, P10):
                both = layer(x)
        torch.testing.assert_close(both, readout_error_map(noisy, P01, P10))

    def test_block_replaces_the_layer_s_own_training_readout(self) -> None:
        layer = _layer(readout_error=(0.3, 0.3))
        layer.train()
        x = _x()
        with torch.no_grad():
            layer.eval()
            clean = layer(x)
            layer.train()
            with apply_readout_error(layer, P01, P10):
                out = layer(x)
        torch.testing.assert_close(out, readout_error_map(clean, P01, P10))

    def test_restores_the_layer(self) -> None:
        layer = _layer()
        x = _x()
        with torch.no_grad():
            before = layer(x)
            with apply_readout_error(layer, P01, P10):
                pass
            torch.testing.assert_close(layer(x), before, atol=0, rtol=0)

    def test_cannot_nest(self) -> None:
        layer = _layer()
        with (
            apply_readout_error(layer, P01, P10),
            pytest.raises(RuntimeError, match="nested"),
            apply_readout_error(layer, 0.1, 0.1),
        ):
            pass

    @pytest.mark.parametrize(("p01", "p10"), [(-0.1, 0.0), (0.0, 1.5), (True, 0.0)])
    def test_invalid_probabilities(self, p01: Any, p10: Any) -> None:
        with pytest.raises(ValueError, match="number in"), apply_readout_error(_layer(), p01, p10):
            pass

    def test_refuses_a_layer_that_would_not_apply_it(self) -> None:
        # A CircuitLayer without TrainingNoiseMixin: its forward never reaches
        # _run_circuit, so the block would leave every output clean.
        inner = _layer()

        class Bare(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qlayer = inner.qlayer
                self.n_qubits = inner.n_qubits

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.qlayer(x)

        bare = Bare()
        with apply_depolarizing_noise(bare, 0.05):  # the QNode-level block does work on it
            pass
        with (
            pytest.raises(TypeError, match="TrainingNoiseMixin"),
            apply_readout_error(bare, P01, P10),
        ):
            pass
        assert getattr(bare.qlayer, "_hqnn_readout_error", None) is None


class TestInsideANoiseBlock:
    """The layer's own readout error is training noise: a noise block suppresses it."""

    @pytest.mark.parametrize("noise_level", [0.0, 0.05])
    def test_p_zero_block_is_the_noiseless_model(self, noise_level: float) -> None:
        layer = _layer(readout_error=(P01, P10), noise_level=noise_level)
        x = _x()
        with torch.no_grad():
            layer.eval()
            clean = layer(x)
            layer.train()
            assert not torch.allclose(layer(x), clean)
            with apply_depolarizing_noise(layer, 0.0):
                torch.testing.assert_close(layer(x), clean, atol=0, rtol=0)

    def test_p_positive_block_is_the_same_in_train_and_eval_mode(self) -> None:
        layer = _layer(readout_error=(P01, P10))
        x = _x()
        with torch.no_grad(), apply_depolarizing_noise(layer, 0.05):
            layer.eval()
            evaluated = layer(x)
            layer.train()
            torch.testing.assert_close(layer(x), evaluated, atol=0, rtol=0)

    def test_an_explicit_readout_block_still_applies_inside_it(self) -> None:
        layer = _layer(readout_error=(0.3, 0.3))
        layer.train()
        x = _x()
        with torch.no_grad(), apply_depolarizing_noise(layer, 0.05):
            noisy = layer(x)
            with apply_readout_error(layer, P01, P10):
                torch.testing.assert_close(layer(x), readout_error_map(noisy, P01, P10))


class TestSweep:
    def test_numbers_are_symmetric_and_pairs_asymmetric(self) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
        x = _x()
        points = noise_sweep(model, x, [0.0, 0.05, (P01, P10)], channel="readout")
        assert [pt.p for pt in points] == [(0.0, 0.0), (0.05, 0.05), (P01, P10)]
        torch.testing.assert_close(points[0].probabilities, model.predict_proba(x))
        with apply_readout_error(model, P01, P10):
            torch.testing.assert_close(points[2].probabilities, model.predict_proba(x))

    def test_every_error_is_checked_before_the_first_evaluation(self) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
        calls = {"n": 0}
        original = model.predict_proba

        def counted(inp: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            return original(inp)

        model.predict_proba = counted  # type: ignore[method-assign, assignment]
        with pytest.raises(ValueError, match="readout error"):
            noise_sweep(model, _x(), [0.01, (0.1, 2.0)], channel="readout")
        assert calls["n"] == 0

    @pytest.mark.parametrize(
        "ps",
        [torch.linspace(0.0, 0.1, 3), np.linspace(0.0, 0.1, 3, dtype=np.float32)],
        ids=["tensor", "float32"],
    )
    def test_accepts_the_scalar_types_the_other_channels_accept(self, ps: Any) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
        x = _x()
        points = noise_sweep(model, x, ps, channel="readout")
        assert [pt.p for pt in points] == [pytest.approx((p, p)) for p in (0.0, 0.05, 0.1)]
        assert all(type(v) is float for pt in points for v in pt.p)  # type: ignore[union-attr]
        flipped = noise_sweep(model, x, ps, channel="bit_flip", position="end")
        for readout, flip in zip(points, flipped, strict=True):
            torch.testing.assert_close(
                readout.probabilities, flip.probabilities, atol=1e-6, rtol=0
            )

    @pytest.mark.parametrize("bad", [None, True, "ab"])
    def test_invalid_entry(self, bad: Any) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
        with pytest.raises(ValueError, match="readout error"):
            noise_sweep(model, _x(), [0.01, bad], channel="readout")

    def test_position_is_validated_as_for_the_other_channels(self) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
        with pytest.raises(ValueError, match="position"):
            noise_sweep(model, _x(), [0.01], channel="readout", position="ends")  # type: ignore[arg-type]


_LAYERS: dict[str, Any] = {
    "angle": QuantumEncodingLayer,
    "iqp": IQPEncodingLayer,
    "amplitude": AmplitudeEncodingLayer,
    "reuploading": DataReuploadingLayer,
}
_CLASSIFIERS: dict[str, Any] = {
    "binary": HybridBinaryClassifier,
    "multiclass": MulticlassHybridClassifier,
    "parallel": ParallelHybridClassifier,
}


def _assert_train_mode_only(layer: Any) -> None:
    """``layer``'s train-mode output is the readout map of its eval-mode output."""
    assert layer.readout_error == (P01, P10)
    x = torch.rand(5, layer.n_features, generator=torch.Generator().manual_seed(1)) + 0.1
    with torch.no_grad():
        layer.eval()
        clean = layer(x)
        layer.train()
        trained = layer(x)
    torch.testing.assert_close(trained, readout_error_map(clean, P01, P10))
    assert not torch.allclose(trained, clean)


class TestTraining:
    def test_applied_in_train_mode_only(self) -> None:
        layer = _layer(readout_error=(P01, P10))
        x = _x()
        with torch.no_grad():
            layer.eval()
            clean = layer(x)
            layer.train()
            trained = layer(x)
        torch.testing.assert_close(trained, readout_error_map(clean, P01, P10))

    @pytest.mark.parametrize("encoding", list(_LAYERS))
    def test_every_layer_applies_it(self, encoding: str) -> None:
        torch.manual_seed(0)
        _assert_train_mode_only(
            _LAYERS[encoding](n_qubits=3, n_layers=2, readout_error=(P01, P10), **CPU)
        )

    @pytest.mark.parametrize("encoding_type", list(_LAYERS))
    @pytest.mark.parametrize("classifier", list(_CLASSIFIERS))
    def test_every_classifier_and_encoding_forwards_it(
        self, classifier: str, encoding_type: str
    ) -> None:
        model = _CLASSIFIERS[classifier](
            n_input_features=3,
            n_qubits=3,
            n_layers=1,
            encoding_type=encoding_type,
            readout_error=(P01, P10),
            **CPU,
        )
        assert isinstance(model.quantum_layer, _LAYERS[encoding_type])
        assert model.get_config()["readout_error"] == (P01, P10)
        _assert_train_mode_only(model.quantum_layer)

    def test_gradient_is_scaled_by_the_contrast(self) -> None:
        x = _x()
        grads = []
        for readout_error in (None, (P01, P10)):
            layer = _layer(readout_error=readout_error)
            layer.train()
            layer(x).sum().backward()
            grads.append(layer.qlayer.weights.grad)
        torch.testing.assert_close(grads[1], (1 - P01 - P10) * grads[0])

    @pytest.mark.parametrize("bad", [(0.1,), (0.1, 0.2, 0.3), "ab", (0.1, -0.2), 0.1])
    def test_invalid_option(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="readout_error"):
            _layer(readout_error=bad)

    @pytest.mark.parametrize(
        "bad",
        [{0.02, 0.09}, {0.02: "a", 0.09: "b"}, "01", (0.1, "0.2"), (0.1, float("nan"))],
        ids=["set", "dict", "str", "str-element", "nan"],
    )
    def test_unordered_or_non_numeric_pair_is_a_value_error(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="readout_error"):
            _layer(readout_error=bad)

    @pytest.mark.parametrize(
        "pair",
        [
            [P01, P10],
            np.array([P01, P10]),
            (np.float64(P01), np.float64(P10)),
            torch.tensor([P01, P10], dtype=torch.float64),
            (v for v in (P01, P10)),
            (0, 1),
        ],
        ids=["list", "ndarray", "np.float64", "tensor", "generator", "ints"],
    )
    def test_normalised_to_a_tuple_of_plain_floats(self, pair: Any) -> None:
        expected = (0.0, 1.0) if isinstance(pair, tuple) and pair == (0, 1) else (P01, P10)
        stored = validate_readout_error(pair)
        assert stored == expected
        assert type(stored) is tuple
        assert all(type(v) is float for v in stored)

    def test_layer_pickled_before_the_option_existed(self) -> None:
        # Such an object has no readout_error attribute at all.
        layer = _layer()
        del layer.readout_error
        layer.train()
        x = _x()
        with torch.no_grad():
            out = layer(x)
            layer.eval()
            torch.testing.assert_close(out, layer(x), atol=0, rtol=0)
        assert "readout_error" not in layer.extra_repr()

    def test_shown_in_the_repr(self) -> None:
        assert "readout_error=(0.02, 0.09)" in _layer(readout_error=(P01, P10)).extra_repr()
        assert "readout_error" not in _layer().extra_repr()

    def test_classifier_config_and_weight_safe_checkpoint(self, tmp_path) -> None:
        model = HybridBinaryClassifier(
            n_input_features=3, n_qubits=3, n_layers=1, readout_error=(P01, P10), **CPU
        )
        assert model.get_config()["readout_error"] == (P01, P10)
        assert model.quantum_layer.readout_error == (P01, P10)
        save_checkpoint(model, tmp_path / "m.pt")
        loaded = load_checkpoint(tmp_path / "m.pt", readout_error=(0.0, 0.1))
        assert loaded.quantum_layer.readout_error == (0.0, 0.1)  # type: ignore[union-attr]

    @pytest.mark.parametrize("classifier", list(_CLASSIFIERS))
    def test_numpy_pair_survives_a_checkpoint(self, classifier: str, tmp_path) -> None:
        # The config is written as it is, and the loader refuses NumPy objects.
        given = np.array([P01, P10])
        model = _CLASSIFIERS[classifier](
            n_input_features=3, n_qubits=3, n_layers=1, readout_error=given, **CPU
        )
        stored = model.get_config()["readout_error"]
        assert stored == (P01, P10)
        assert all(type(v) is float for v in stored)
        given[0] = 0.5  # the config holds its own copy
        assert model.get_config()["readout_error"] == (P01, P10)
        save_checkpoint(model, tmp_path / "m.pt")
        loaded = load_checkpoint(tmp_path / "m.pt")
        assert loaded.get_config()["readout_error"] == (P01, P10)
        assert loaded.quantum_layer.readout_error == (P01, P10)  # type: ignore[union-attr]
