"""
tests/test_encoding_contract.py
===============================
Every encoding layer against the contract in hqnn_forge._encoding_contract (#215).

``ENCODERS`` is typed ``Callable[[], EncodingLayer]``, so mypy (which CI runs
on tests/) checks each encoder class against the protocol statically; the
tests check the same contract at run time, and that ``forward`` does no
classical input handling of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import pennylane as qml
import pytest
import torch
import torch.nn as nn

from hqnn_forge._resolve import require_prepare_inputs, resolve_encoding_layer
from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    CircuitLayer,
    DataReuploadingLayer,
    EncodingLayer,
    QuantumEncodingLayer,
    is_circuit_layer,
    is_encoding_layer,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer

N_QUBITS = 3


def _angle() -> EncodingLayer:
    return QuantumEncodingLayer(
        n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
    )


def _angle_y_first() -> EncodingLayer:
    return QuantumEncodingLayer(
        n_qubits=N_QUBITS,
        n_layers=2,
        rotation="Y",
        entangler="strongly_entangling",
        readout="first",
        device_name="default.qubit",
        diff_method="backprop",
    )


def _iqp() -> EncodingLayer:
    return IQPEncodingLayer(
        n_qubits=N_QUBITS,
        n_layers=2,
        n_repeats=2,
        device_name="default.qubit",
        diff_method="backprop",
    )


def _amplitude_padded() -> EncodingLayer:
    # 5 features on 3 qubits: prepare_inputs pads to 8 and normalises, the one
    # encoder whose classical step transforms the input rather than only checking it.
    return AmplitudeEncodingLayer(
        n_qubits=N_QUBITS,
        n_layers=2,
        n_features=5,
        device_name="default.qubit",
        diff_method="backprop",
    )


def _reuploading_scaled() -> EncodingLayer:
    return DataReuploadingLayer(
        n_qubits=N_QUBITS,
        n_layers=2,
        trainable_input_scaling=True,
        device_name="default.qubit",
        diff_method="backprop",
    )


ENCODERS: dict[str, Callable[[], EncodingLayer]] = {
    "angle": _angle,
    "angle-y-first": _angle_y_first,
    "iqp": _iqp,
    "amplitude-padded": _amplitude_padded,
    "reuploading-scaled": _reuploading_scaled,
}


def _build(name: str) -> EncodingLayer:
    torch.manual_seed(0)
    return ENCODERS[name]()


def _inputs(layer: EncodingLayer) -> torch.Tensor:
    width = layer.n_features
    # Unnormalised, wider than [-1, 1], and float32 as a caller would pass
    # them: the amplitude layer's normalisation has something to do.
    return 3 * torch.randn(4, width, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("name", ENCODERS)
class TestEveryEncoder:
    def test_satisfies_the_runtime_check(self, name: str) -> None:
        layer = _build(name)
        assert is_circuit_layer(layer) and is_encoding_layer(layer)
        assert isinstance(layer.qlayer, qml.qnn.TorchLayer)
        assert layer.n_qubits == N_QUBITS
        expected_width = 5 if name == "amplitude-padded" else N_QUBITS
        assert type(layer.n_features) is int and layer.n_features == expected_width

    def test_prepare_inputs_accepts_n_features(self, name: str) -> None:
        # n_features is the width prepare_inputs takes: the contract member
        # callers size their own inputs by (the width-rejection test below
        # checks that one less or one more is refused).
        layer = _build(name)
        x = torch.randn(2, layer.n_features, generator=torch.Generator().manual_seed(3))
        assert layer.prepare_inputs(x).shape[0] == 2

    @pytest.mark.parametrize("mode", ["eval", "train"])
    def test_forward_is_qlayer_of_prepare_inputs(self, name: str, mode: str) -> None:
        # Bit-identical, not close: forward must do nothing between the two.
        layer = _build(name)
        assert isinstance(layer, nn.Module)
        layer.train(mode == "train")
        x = _inputs(layer)
        with torch.no_grad():
            torch.testing.assert_close(
                layer(x), layer.qlayer(layer.prepare_inputs(x)), rtol=0, atol=0
            )

    def test_prepare_inputs_rejects_what_forward_rejects(self, name: str) -> None:
        # The kernels validate through prepare_inputs alone, so it has to be
        # where forward's own validation lives.
        layer = _build(name)
        x = _inputs(layer)
        x[0, 0] = float("nan")
        with pytest.raises(ValueError) as from_forward:
            layer(x)
        with pytest.raises(ValueError) as from_prepare:
            layer.prepare_inputs(x)
        assert str(from_prepare.value) == str(from_forward.value)

    @pytest.mark.parametrize("delta", [-1, 1])
    def test_prepare_inputs_rejects_the_wrong_width_as_forward_does(
        self, name: str, delta: int
    ) -> None:
        # The width check is the other half of the validation the contract
        # puts in prepare_inputs; the kernels rely on it to refuse bad X.
        layer = _build(name)
        width = _inputs(layer).shape[1] + delta
        x = torch.randn(4, width, generator=torch.Generator().manual_seed(2))
        with pytest.raises(ValueError) as from_forward:
            layer(x)
        with pytest.raises(ValueError) as from_prepare:
            layer.prepare_inputs(x)
        assert str(from_prepare.value) == str(from_forward.value)

    def test_forward_returns_expectation_values(self, name: str) -> None:
        # One row per sample, one ⟨Z⟩ per read-out wire, each in [-1, 1].
        layer = _build(name)
        x = _inputs(layer)
        n_outputs = 1 if getattr(layer, "readout", "all") == "first" else N_QUBITS
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (x.shape[0], n_outputs)
        assert torch.isfinite(out).all() and out.abs().max() <= 1 + 1e-6

    def test_the_resolver_accepts_it(self, name: str) -> None:
        layer = _build(name)
        found, qlayer, n_qubits = resolve_encoding_layer(layer, "f", allow_model=False)
        assert found is layer and qlayer is layer.qlayer and n_qubits == N_QUBITS
        assert require_prepare_inputs(found, "f") is layer


@pytest.mark.parametrize(
    "cls",
    [
        QuantumEncodingLayer,
        IQPEncodingLayer,
        partial(AmplitudeEncodingLayer, n_features=N_QUBITS),
        DataReuploadingLayer,
    ],
    ids=["angle", "iqp", "amplitude", "reuploading"],
)
def test_training_noise_is_the_one_sanctioned_difference(cls: Callable[..., Any]) -> None:
    # The contract's documented exception: with noise_level > 0, forward in
    # train() mode runs a noisy circuit; in eval() mode it is the noiseless one.
    torch.manual_seed(0)
    layer = cls(
        n_qubits=N_QUBITS,
        n_layers=2,
        noise_level=0.2,
        device_name="default.qubit",
        diff_method="backprop",
    )
    x = torch.randn(4, N_QUBITS, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        clean = layer.qlayer(layer.prepare_inputs(x))
        layer.eval()
        torch.testing.assert_close(layer(x), clean, rtol=0, atol=0)
        layer.train()
        assert not torch.allclose(layer(x), clean)


def _bare_circuit_layer() -> nn.Module:
    """qlayer and n_qubits, no prepare_inputs: enough for the diagnostics only."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):  # type: ignore[no-untyped-def]
        qml.AngleEmbedding(inputs, wires=range(2))
        qml.RY(weights[0], wires=0)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    class Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_qubits = 2
            self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (1,)})

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.qlayer(x)  # type: ignore[no-any-return]

    return Bare()


class TestTheChecks:
    def test_a_circuit_layer_without_prepare_inputs(self) -> None:
        bare = _bare_circuit_layer()
        assert is_circuit_layer(bare) and not is_encoding_layer(bare)
        found, _, _ = resolve_encoding_layer(bare, "circuit_summary")
        assert found is bare
        with pytest.raises(TypeError, match="f expects an encoding layer with a prepare_inputs"):
            require_prepare_inputs(found, "f")

    @pytest.mark.parametrize("n_features", [None, True, 3.0, "3"])
    def test_n_features_must_be_a_plain_int(self, n_features: object) -> None:
        layer = _build("angle")
        if n_features is None:
            del layer.n_features
        else:
            layer.n_features = n_features  # type: ignore[assignment]
        assert is_circuit_layer(layer) and not is_encoding_layer(layer)
        with pytest.raises(TypeError, match="and an int n_features"):
            require_prepare_inputs(layer, "f")

    def test_a_non_callable_prepare_inputs(self) -> None:
        bare = _bare_circuit_layer()
        bare.prepare_inputs = 3  # type: ignore[assignment]
        assert not is_encoding_layer(bare)

    def test_not_a_module(self) -> None:
        layer = _build("angle")

        class Impostor:
            qlayer = layer.qlayer
            n_qubits = N_QUBITS
            prepare_inputs = layer.prepare_inputs

        assert not is_circuit_layer(Impostor()) and not is_encoding_layer(Impostor())

    @pytest.mark.parametrize("n_qubits", [True, 3.0, "3", None])
    def test_n_qubits_must_be_a_plain_int(self, n_qubits: object) -> None:
        layer = _build("angle")
        layer.n_qubits = n_qubits  # type: ignore[assignment]
        assert not is_circuit_layer(layer) and not is_encoding_layer(layer)

    def test_qlayer_must_be_a_torch_layer(self) -> None:
        layer = _build("angle")
        assert isinstance(layer, nn.Module)
        layer.qlayer = nn.Linear(3, 3)  # type: ignore[assignment]
        assert not is_circuit_layer(layer)

    def test_submodule_qlayer_is_seen(self) -> None:
        # The reason for is_circuit_layer over a runtime_checkable protocol:
        # qlayer lives in nn.Module._modules, which getattr_static cannot see.
        layer = _build("angle")
        assert isinstance(layer, nn.Module)
        assert "qlayer" not in vars(layer) and "qlayer" in layer._modules
        assert is_circuit_layer(layer)


def test_circuit_layer_is_the_weaker_contract() -> None:
    # Statically: an EncodingLayer is usable wherever a CircuitLayer is.
    weaker: CircuitLayer = _build("angle")
    assert is_circuit_layer(weaker)
