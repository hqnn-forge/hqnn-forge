"""
tests/test_variational_weight_shape.py
======================================
One definition of the variational weight shape (#305): every encoder registers
``weights`` with ``variational_weight_shape``, and ``extra_repr`` counts the
parameters the layer really has.
"""

from __future__ import annotations

import re
from typing import Any, get_args

import pytest

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.angle_embedding import Entangler, variational_weight_shape
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
ENTANGLERS: list[Entangler] = list(get_args(Entangler))


def _layers(entangler: Entangler) -> list[Any]:
    kw: dict[str, Any] = {"n_qubits": 3, "n_layers": 2, **CPU}
    layers = [
        QuantumEncodingLayer(entangler=entangler, **kw),
        IQPEncodingLayer(entangler=entangler, **kw),
        DataReuploadingLayer(entangler=entangler, **kw),
        DataReuploadingLayer(entangler=entangler, trainable_input_scaling=True, **kw),
    ]
    if entangler == "ring":  # the amplitude layer always uses the ring block
        layers.append(AmplitudeEncodingLayer(n_features=5, **kw))
    return layers


@pytest.mark.parametrize("entangler", ENTANGLERS)
def test_every_encoder_registers_the_shared_shape(entangler: Entangler) -> None:
    for layer in _layers(entangler):
        assert tuple(layer.qlayer.weights.shape) == variational_weight_shape(
            entangler,
            layer.n_qubits,
            layer.n_layers,
        ), type(layer).__name__


@pytest.mark.parametrize("entangler", ENTANGLERS)
def test_extra_repr_counts_the_real_parameters(entangler: Entangler) -> None:
    for layer in _layers(entangler):
        (n_params,) = re.findall(r"n_params=(\d+)", layer.extra_repr())
        assert int(n_params) == sum(p.numel() for p in layer.parameters()), type(layer).__name__


def test_dim_zero_is_the_layer_index() -> None:
    # block_local_init_ and the diagnostics' n_layers fallback read dim 0.
    for entangler in ENTANGLERS:
        assert variational_weight_shape(entangler, 5, 7)[0] == 7


def test_unknown_entangler() -> None:
    with pytest.raises(ValueError, match="entangler must be"):
        variational_weight_shape("ladder", 3, 2)  # type: ignore[arg-type]
