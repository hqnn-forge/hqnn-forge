"""
tests/test_resolve.py
=====================
hqnn_forge._resolve.resolve_encoding_layer, the one walk from "a model or a
layer" to its encoding layer that every diagnostic shares (#182).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hqnn_forge._resolve import resolve_encoding_layer
from hqnn_forge.diagnostics import circuit_summary, draw_circuit, gradient_variance
from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.kernels import quantum_kernel_matrix
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import apply_depolarizing_noise

CPU = dict(device_name="default.qubit", diff_method="backprop")


@pytest.fixture
def layer() -> QuantumEncodingLayer:
    return QuantumEncodingLayer(n_qubits=2, n_layers=1, **CPU)  # type: ignore[arg-type]


class _Holder(nn.Module):
    """A 'classifier' whose quantum_layer is whatever the test puts there."""

    def __init__(self, inner: object) -> None:
        super().__init__()
        object.__setattr__(self, "quantum_layer", inner)


class TestResolution:
    def test_a_layer_resolves_to_itself(self, layer: QuantumEncodingLayer) -> None:
        assert resolve_encoding_layer(layer, "f") == (layer, layer.qlayer, 2)

    def test_a_classifier_resolves_to_its_quantum_layer(self) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=2, n_layers=1, **CPU)  # type: ignore[arg-type]
        found, qlayer, n_qubits = resolve_encoding_layer(model, "f")
        assert found is model.quantum_layer and qlayer is model.quantum_layer.qlayer
        assert n_qubits == 2


class TestRejection:
    def test_names_the_caller_and_the_type(self) -> None:
        with pytest.raises(TypeError, match=r"^my_function expects .*got Linear\.$"):
            resolve_encoding_layer(nn.Linear(2, 1), "my_function")

    def test_a_non_module_quantum_layer_is_refused(self, layer: QuantumEncodingLayer) -> None:
        """The check #127 had to restore: duck-typed attributes on a non-module."""

        class Impostor:
            qlayer = layer.qlayer
            n_qubits = 2

        with pytest.raises(TypeError, match="got _Holder"):
            resolve_encoding_layer(_Holder(Impostor()), "f")

    def test_qlayer_must_be_a_torch_layer(self) -> None:
        bad = nn.Module()
        bad.qlayer = nn.Linear(2, 2)
        bad.n_qubits = 2  # type: ignore[assignment]
        with pytest.raises(TypeError):
            resolve_encoding_layer(bad, "f")

    @pytest.mark.parametrize("n_qubits", [2.0, "2", True, None])
    def test_n_qubits_must_be_an_int(self, layer: QuantumEncodingLayer, n_qubits: object) -> None:
        layer.n_qubits = n_qubits  # type: ignore[assignment]
        with pytest.raises(TypeError):
            resolve_encoding_layer(layer, "f")

    def test_without_allow_model_a_classifier_is_refused_with_the_reason(self) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=2, n_layers=1, **CPU)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="classical encoder runs before"):
            resolve_encoding_layer(model, "f", allow_model=False)


class TestEveryCallerNamesItself:
    """Each public function's error message names that function, not a sibling."""

    @pytest.mark.parametrize(
        ("call", "name"),
        [
            (lambda t: circuit_summary(t), "circuit_summary"),
            (lambda t: draw_circuit(t), "draw_circuit"),
            (lambda t: gradient_variance(t), "gradient_variance"),
            (lambda t: apply_depolarizing_noise(t, 0.1).__enter__(), "apply_depolarizing_noise"),
            (lambda t: quantum_kernel_matrix(torch.zeros(2, 2), t), "quantum_kernel_matrix"),
        ],
    )
    def test_message(self, call: object, name: str) -> None:
        with pytest.raises(TypeError, match=rf"^{name} expects"):
            call(nn.Linear(2, 1))  # type: ignore[operator]
