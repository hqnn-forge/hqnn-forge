"""
tests/test_readout_error.py
===========================
The asymmetric readout error (#358): its affine effect on ⟨Z⟩ against an
independent density-matrix reference, post hoc, in a sweep, and in training;
on a sampled layer, flipped shots (#486).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
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
    _binomial_by_inversion,
    _flip_generator,
    apply_depolarizing_noise,
    apply_readout_error,
    apply_shots,
    noise_sweep,
    readout_error_map,
    shot_sweep,
    validate_readout_error,
)
from hqnn_forge.training import SPSA
from hqnn_forge.utils import load_checkpoint, save_checkpoint

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
SHIFT: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "parameter-shift"}
P01, P10 = 0.02, 0.09
#: Standard errors a statistical test allows; with a fixed seed the tests are
#: deterministic, and this keeps them from depending on a lucky seed.
Z = 5.0


def _layer(**kwargs: Any) -> QuantumEncodingLayer:
    torch.manual_seed(0)
    return QuantumEncodingLayer(n_qubits=3, n_layers=2, **{**CPU, **kwargs})


def _x() -> torch.Tensor:
    return torch.rand(5, 3, generator=torch.Generator().manual_seed(1)) * 4 - 2


def _on_shot_lattice(values: torch.Tensor, shots: int) -> bool:
    """Whether every ⟨Z⟩ in ``values`` is ``(2k − shots) / shots``: a sampled estimate."""
    # Counts are integers; the float32 rounding of a layer's output moves them
    # by up to about 3e-8 · shots, far below the 1e-3 allowed.
    k = (values.detach().double() * shots + shots) / 2
    return bool(torch.allclose(k, k.round(), rtol=0, atol=1e-3))


def _sampled(pair: tuple[float, float] | None, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
    """
    A ``seed=7`` layer's output inside ``apply_shots(layer, 200)``, and the layer.

    With ``pair`` the block also has ``apply_readout_error(layer, *pair)``;
    with ``None`` it has none, so the output is the plain sampled estimate the
    flips start from (the shots are the same either way).
    """
    layer = _layer(seed=7)
    readout = nullcontext() if pair is None else apply_readout_error(layer, *pair)
    with torch.no_grad(), apply_shots(layer, 200), readout:
        return layer(x), layer


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

    def test_composes_with_a_shots_block(self) -> None:
        # Inside apply_shots the shots are flipped, not the estimate mapped:
        # the output stays on the 200-shot lattice, which the map (scale
        # 0.89) leaves, and differs from the same shots without the error.
        # TestFlippedShots checks the distribution.
        x = _x()
        both, _ = _sampled((P01, P10), x)
        sampled, _ = _sampled(None, x)
        assert _on_shot_lattice(both, 200)
        assert not torch.equal(both, sampled)
        with torch.no_grad():
            assert not torch.allclose(sampled, _layer()(x), atol=1e-3)  # really sampled

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

    def test_a_bare_pair_is_two_symmetric_levels(self) -> None:
        # ps is the list of levels; one asymmetric error is [(p01, p10)].
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
        x = _x()
        assert [pt.p for pt in noise_sweep(model, x, (P01, P10), channel="readout")] == [
            (P01, P01),
            (P10, P10),
        ]
        assert [pt.p for pt in noise_sweep(model, x, [(P01, P10)], channel="readout")] == [
            (P01, P10)
        ]

    def test_unknown_channel_error_names_readout(self) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
        with pytest.raises(ValueError, match="'phase_flip', 'readout'; got 'readouts'"):
            noise_sweep(model, _x(), [0.01], channel="readouts")  # type: ignore[arg-type]

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

    @pytest.mark.parametrize("bad", [None, True, np.True_, "ab"])
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

    def test_with_shots_it_flips_the_shots_as_the_block_does(self) -> None:
        # Same weights and device seed: the layer's own train-mode error draws
        # the same shots and the same flips as an apply_readout_error block on
        # a layer without one (TestFlippedShots checks the block's statistics).
        x = _x()
        outputs = []
        for readout_error in (None, (P01, P10)):
            layer = _layer(shots=200, seed=7, readout_error=readout_error, **SHIFT)
            with torch.no_grad():
                if readout_error is None:
                    with apply_readout_error(layer, P01, P10):
                        outputs.append(layer(x))
                else:
                    layer.train()
                    outputs.append(layer(x))
        assert _on_shot_lattice(outputs[1], 200)
        torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
        # Both really flip: the same seeded shots without an error differ.
        unflipped = _layer(shots=200, seed=7, **SHIFT)
        with torch.no_grad():
            assert not torch.equal(unflipped(x), outputs[0])

    @pytest.mark.parametrize("channel", ["depolarizing", "amplitude_damping"])
    def test_with_trajectories_it_maps_the_mean_over_draws(self, channel: str) -> None:
        # The draws come from torch's generator: reseeded, both layers take the same.
        x = _x()
        outputs = []
        for readout_error in (None, (P01, P10)):
            layer = _layer(
                noise_level=0.2,
                noise_channel=channel,
                noise_method="trajectories",
                noise_trajectories=4,
                readout_error=readout_error,
            )
            layer.train()
            torch.manual_seed(11)
            with torch.no_grad():
                outputs.append(layer(x))
        layer.eval()
        with torch.no_grad():
            assert not torch.allclose(outputs[0], layer(x), atol=1e-3)  # really noisy
        torch.testing.assert_close(outputs[1], readout_error_map(outputs[0], P01, P10))

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
        "bad",
        [
            (True, 0),
            (np.True_, 0),
            np.array([True, False]),
            torch.tensor([True, False]),
            np.array([[0.1], [0.2]]),
            torch.tensor([[0.1], [0.2]]),
            ([0.1], [0.2]),
        ],
        ids=["bool", "np.bool_", "bool-array", "bool-tensor", "2x1-array", "2x1-tensor", "lists"],
    )
    def test_boolean_or_non_scalar_element_is_refused(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="readout_error"):
            validate_readout_error(bad)

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


class TestFlippedShots:
    """
    On a sampled layer the readout error flips each shot (#486).

    ``N`` shots with ``k`` zeros flip to ``Binomial(k, 1 − p01) +
    Binomial(N − k, p10)`` zeros, each flipped shot reading 0 with probability
    ``(1 + ⟨Z'⟩)/2``: the estimate has mean ``⟨Z'⟩ = (1 − p01 − p10)⟨Z⟩ +
    (p10 − p01)`` and variance ``(1 − ⟨Z'⟩²)/N``.  The map of #372 applied to
    the estimate had the same mean and variance ``(1 − p01 − p10)²(1 − ⟨Z⟩²)/N``
    instead, zero at ``p01 + p10 = 1``.
    """

    X = torch.tensor([[0.3, -1.1, 0.7]])

    @pytest.mark.slow
    @pytest.mark.parametrize("pair", [(0.0, 0.0), (0.1, 0.3), (0.3, 0.7), (0.5, 0.5)], ids=str)
    def test_mean_and_variance_are_those_of_flipped_shots(self, pair: tuple[float, float]) -> None:
        # Each row of the batch is one independent run of the same input.  The
        # sample variance of 600 near-Gaussian runs has relative standard error
        # sqrt(2/599) = 5.8%, so Z of them is 29%.  The scaled shot noise of
        # #372 is 62% too small at (0.1, 0.3) (measured: std 0.059 against
        # 0.097) and zero at (0.3, 0.7) and (0.5, 0.5), so all three fail it;
        # (0, 0) is the control, where nothing is flipped.
        shots, repeats = 100, 600
        layer = _layer(seed=5)
        with torch.no_grad():
            z_flipped = readout_error_map(layer(self.X)[0].double(), *pair)
            with apply_shots(layer, shots), apply_readout_error(layer, *pair):
                runs = layer(self.X.expand(repeats, 3)).double()
        assert _on_shot_lattice(runs, shots)
        expected_var = (1 - z_flipped**2) / shots
        se_mean = (expected_var / repeats).sqrt()
        assert ((runs.mean(0) - z_flipped).abs() <= Z * se_mean).all(), runs.mean(0)
        ratio = runs.var(0) / expected_var
        assert ((ratio - 1).abs() <= Z * math.sqrt(2 / (repeats - 1))).all(), ratio

    def test_which_probability_flips_which_outcome(self) -> None:
        # Hand-built limits, with the shots shared through the device seed:
        # p01 = p10 = 1 flips every shot, so the count of zeros k becomes
        # N − k and the estimate its exact negative; (1, 0) turns every 0 into
        # a 1 (⟨Z'⟩ = −1) and (0, 1) every 1 into a 0 (⟨Z'⟩ = +1).  Swapping
        # p01 and p10 in the draw swaps the last two.
        x = _x()
        outputs = {
            pair: _sampled(pair, x)[0] for pair in (None, (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))
        }
        sampled = outputs[None]
        with torch.no_grad():
            assert not torch.allclose(sampled, _layer()(x), atol=1e-3)  # really sampled
        torch.testing.assert_close(outputs[(1.0, 1.0)], -sampled, atol=0, rtol=0)
        assert (outputs[(1.0, 0.0)] == -1).all()
        assert (outputs[(0.0, 1.0)] == 1).all()

    def test_no_error_draws_nothing(self) -> None:
        # (0, 0) is the identity on every shot: the output is the sampled
        # estimate itself, and the device's generator is not advanced.
        x = _x()
        outputs, states = [], []
        for pair in (None, (0.0, 0.0)):
            out, layer = _sampled(pair, x)
            outputs.append(out)
            states.append(layer.qlayer.qnode.device._rng.bit_generator.state)
        torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
        assert states[1] == states[0]

    def test_the_exact_path_is_unchanged_bit_for_bit(self) -> None:
        # shots=None keeps the map of #372, readout_error_map of the clean
        # output, in train mode, in a block, and inside apply_shots(None)
        # (whose parameter-shift QNode differs from backprop in the last bit,
        # so its own clean output is the reference there).
        layer = _layer(readout_error=(P01, P10))
        x = _x()
        with torch.no_grad():
            layer.eval()
            expected = readout_error_map(layer(x), P01, P10)
            with apply_readout_error(layer, P01, P10):
                torch.testing.assert_close(layer(x), expected, atol=0, rtol=0)
            with apply_shots(layer, None):
                expected_shift = readout_error_map(layer(x), P01, P10)
                with apply_readout_error(layer, P01, P10):
                    torch.testing.assert_close(layer(x), expected_shift, atol=0, rtol=0)
            layer.train()
            torch.testing.assert_close(layer(x), expected, atol=0, rtol=0)

    def test_a_seeded_layer_repeats_whatever_torch_s_seed(self) -> None:
        # The flips are drawn from the device's generator, as the shots are:
        # the device seed alone fixes them, and torch's seed does not enter.
        def run(seed: int, torch_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
            layer = _layer(seed=seed)
            torch.manual_seed(torch_seed)
            with torch.no_grad(), apply_shots(layer, 100), apply_readout_error(layer, 0.1, 0.3):
                return layer(_x()), layer(_x())

        first, again = run(7, 0), run(7, 1)
        for a, b in zip(first, again, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert not torch.equal(first[0], first[1])  # each call draws afresh
        assert not torch.equal(first[0], run(8, 0)[0])

    def test_without_a_device_generator_torch_s_seed_repeats_the_draw(self) -> None:
        # A device that keeps no NumPy generator (hardware, a plugin) gets one
        # seeded from torch's global generator, so torch.manual_seed fixes it.
        def draw(torch_seed: int) -> np.ndarray:
            torch.manual_seed(torch_seed)
            return _flip_generator(object()).integers(0, 2**31, size=4)

        np.testing.assert_array_equal(draw(3), draw(3))
        assert not np.array_equal(draw(3), draw(4))
        device = _layer(seed=7).qlayer.qnode.device
        assert _flip_generator(device) is device._rng

    @pytest.mark.slow
    def test_trajectories_flip_the_pooled_shots(self) -> None:
        # The output is the mean of k = 4 draws of N = 50 shots each, so it is
        # an estimate from k·N shots.  At p01 + p10 = 1 a flipped shot reads 0
        # with probability p10 whatever was measured, so the pooled count of
        # zeros is Binomial(k·N, p10) exactly: mean 2·p10 − 1, variance
        # 4·p10·(1 − p10)/(k·N).  Flipping N shots instead would give k = 4
        # times that variance, and #372's map zero; with 300 runs, Z standard
        # errors of the variance are 41%.
        shots, k, repeats, (p01, p10) = 50, 4, 300, (0.3, 0.7)
        layer = _layer(
            shots=shots,
            seed=5,
            noise_level=0.1,
            noise_method="trajectories",
            noise_trajectories=k,
            readout_error=(p01, p10),
            **SHIFT,
        )
        layer.train()
        with torch.no_grad():
            runs = layer(self.X.expand(repeats, 3)).double()
        assert _on_shot_lattice(runs, k * shots)
        mean, var = 2 * p10 - 1, 4 * p10 * (1 - p10) / (k * shots)
        assert ((runs.mean(0) - mean).abs() <= Z * math.sqrt(var / repeats)).all()
        ratio = runs.var(0) / var
        assert ((ratio - 1).abs() <= Z * math.sqrt(2 / (repeats - 1))).all(), ratio

    def test_parameter_shift_gradient_is_that_of_the_mapped_estimate(self) -> None:
        # The gradient of a flipped output is (1 − p01 − p10) times the
        # sampled one, as before #486.  Both layers draw the same forward
        # shots; the second also draws its flips, so its generator is reset to
        # the first's state for the backward pass, whose shifted circuits then
        # draw the same shots too.
        x = _x()
        grads = []
        state = None
        for pair in (None, (P01, P10)):
            layer = _layer(shots=200, seed=7, **SHIFT)
            rng = layer.qlayer.qnode.device._rng
            if pair is None:
                out = layer(x)
                state = rng.bit_generator.state
            else:
                with apply_readout_error(layer, *pair):
                    out = layer(x)
                assert _on_shot_lattice(out, 200)
                rng.bit_generator.state = state
            grads.append(torch.autograd.grad(out.sum(), layer.qlayer.weights)[0])
        assert grads[0].abs().max() > 0
        torch.testing.assert_close(grads[1], (1 - P01 - P10) * grads[0])

    def test_spsa_evaluations_share_the_flips(self) -> None:
        # SPSA restores the device generators between its two evaluations
        # (model=), and the flips come from them: with only the head
        # perturbed, both evaluations see the same flipped outputs.  Without
        # model= they draw independently.
        def outputs(sync: bool) -> list[torch.Tensor]:
            model = HybridBinaryClassifier(
                n_input_features=3,
                n_qubits=3,
                n_layers=1,
                shots=100,
                seed=3,
                readout_error=(0.1, 0.3),
                **SHIFT,
            )
            model.train()
            seen: list[torch.Tensor] = []
            model.quantum_layer.register_forward_hook(
                lambda _m, _i, out: seen.append(out.detach().clone())
            )
            x, y = _x(), torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
            opt = SPSA(model.head.parameters(), perturbation=0.05, model=model if sync else None)
            opt.gradient_estimate(
                lambda: nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
            )
            assert len(seen) == 2
            assert _on_shot_lattice(seen[0], 100)
            return seen

        plus, minus = outputs(sync=True)
        torch.testing.assert_close(plus, minus, atol=0, rtol=0)
        plus, minus = outputs(sync=False)
        assert not torch.equal(plus, minus)

    def test_spsa_evaluations_stay_in_step_when_the_counts_differ(self) -> None:
        # With the quantum weights perturbed the two evaluations see different
        # counts of zeros.  Each flip still takes one uniform per binomial, so
        # both leave the shared device generator in the same state, and the
        # draws after the flips stay common.  numpy's binomial sampler takes a
        # count-dependent number of uniforms at large n·p, which fails this.
        model = HybridBinaryClassifier(
            n_input_features=3,
            n_qubits=3,
            n_layers=1,
            shots=1000,
            seed=3,
            readout_error=(0.1, 0.3),
            **SHIFT,
        )
        model.train()
        rng = model.quantum_layer.qlayer.qnode.device._rng
        seen: list[tuple[torch.Tensor, dict[str, Any]]] = []
        model.quantum_layer.register_forward_hook(
            lambda _m, _i, out: seen.append((out.detach().clone(), rng.bit_generator.state))
        )
        x, y = _x(), torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        opt = SPSA(model.quantum_layer.parameters(), perturbation=0.2, model=model)
        opt.gradient_estimate(
            lambda: nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
        )
        (plus, plus_state), (minus, minus_state) = seen
        assert not torch.equal(plus, minus)  # the counts really differ
        assert plus_state == minus_state

    def test_binomial_by_inversion_takes_one_uniform_per_element(self) -> None:
        # Whatever the counts and p, exactly counts.size uniforms; p = 0 and
        # p = 1 are exact even at a uniform of 1 (rng.random() returning 0).
        counts = np.array([[0, 3, 400], [1000, 7, 0]])
        for p in (0.0, 0.02, 0.5, 0.97, 1.0):
            rng, reference = np.random.default_rng(1), np.random.default_rng(1)
            draws = _binomial_by_inversion(counts, p, rng)
            reference.random(counts.shape)
            assert rng.bit_generator.state == reference.bit_generator.state
            assert draws.dtype == np.int64
            assert ((draws >= 0) & (draws <= counts)).all()

        class Zero:
            def random(self, shape: tuple[int, ...]) -> np.ndarray:
                return np.zeros(shape)

        top = _binomial_by_inversion(counts, 0.0, Zero())  # type: ignore[arg-type]
        np.testing.assert_array_equal(top, 0)
        np.testing.assert_array_equal(_binomial_by_inversion(counts, 1.0, Zero()), counts)  # type: ignore[arg-type]
        np.testing.assert_array_equal(_binomial_by_inversion(counts, 0.3, Zero()), counts)  # type: ignore[arg-type]

    @pytest.mark.slow
    def test_shot_sweep_inside_the_block(self) -> None:
        # At p01 + p10 = 1 the flipped output is 2·Binomial(N, p10)/N − 1
        # whatever the circuit: mean p10 − p01 and variance
        # 4·p10·(1 − p10)/N, where #372's map gave the constant p10 − p01.
        # With the head set to the identity, predict_proba is sigmoid(⟨Z'_0⟩).
        # 600 values: Z standard errors of the variance are 29%.  The device
        # seed fixes the shots and the flips, and torch.manual_seed the weights
        # the shots are drawn from, so the test is deterministic.
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            n_input_features=3, n_qubits=3, n_layers=1, readout="first", seed=5, **CPU
        )
        with torch.no_grad():
            model.head.weight.fill_(1.0)
            model.head.bias.zero_()
        shots, rows, (p01, p10) = 100, 300, (0.3, 0.7)
        x = _x()[:1].expand(rows, 3)
        with apply_readout_error(model, p01, p10):
            exact, sampled = shot_sweep(model, x, [None, shots], n_repeats=2)
        torch.testing.assert_close(
            exact.probabilities.double(),
            torch.full((2, rows), p10 - p01, dtype=torch.float64).sigmoid(),
        )
        z = torch.logit(sampled.probabilities.double()).flatten()
        mean, var = p10 - p01, 4 * p10 * (1 - p10) / shots
        n = z.numel()
        assert abs(z.mean().item() - mean) <= Z * math.sqrt(var / n)
        assert abs(z.var().item() / var - 1) <= Z * math.sqrt(2 / (n - 1)), z.var()
