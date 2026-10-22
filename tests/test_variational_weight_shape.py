"""
tests/test_variational_weight_shape.py
======================================
One definition of the variational weight shape (#305): every encoder registers
``weights`` with ``variational_weight_shape``, and ``extra_repr`` counts the
parameters the layer really has.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, get_args

import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding._common import apply_variational_layers
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


# n_qubits=4 differs from the Rot dim 3, so a swapped axis is caught.
EXPECTED_SHAPES: dict[Entangler, tuple[int, ...]] = {
    "ring": (2, 4, 3),
    "strongly_entangling": (2, 4, 3),
    "brickwork": (2, 4, 3),
    "hardware_efficient": (2, 4),
}


def test_expected_shapes_cover_every_entangler() -> None:
    assert set(EXPECTED_SHAPES) == set(ENTANGLERS)


@pytest.mark.parametrize(("entangler", "expected_shape"), EXPECTED_SHAPES.items())
def test_shapes_for_each_entangler(entangler: Entangler, expected_shape: tuple[int, ...]) -> None:
    assert variational_weight_shape(entangler, 4, 2) == expected_shape


def _gates(tape: qml.tape.QuantumScript) -> Iterator[qml.operation.Operator]:
    """The tape's gates, with the one template (``StronglyEntanglingLayers``) unpacked."""
    for op in tape.operations:
        if isinstance(op, qml.StronglyEntanglingLayers):
            yield from op.decomposition()
        else:
            yield op


# At least two qubits: the library refuses fewer, and the ring's CNOT needs a partner.
@pytest.mark.parametrize("entangler", ENTANGLERS)
@pytest.mark.parametrize(("n_qubits", "n_layers"), [(2, 1), (2, 3), (3, 2), (4, 3), (5, 1)])
def test_every_weight_reaches_exactly_one_gate_angle(
    entangler: Entangler, n_qubits: int, n_layers: int
) -> None:
    # Distinct values, so a weight that no gate reads, or that two gates read, shows up in the
    # list of angles; a block that reads the wrong shape broadcasts instead of raising.
    shape = variational_weight_shape(entangler, n_qubits, n_layers)
    weights = torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(shape)
    with qml.tape.QuantumTape() as tape:
        apply_variational_layers(weights, n_qubits, n_layers, entangler)

    assert tape.batch_size is None
    angles = [angle for gate in _gates(tape) for angle in gate.data]
    assert all(getattr(angle, "ndim", 0) == 0 for angle in angles)
    assert sorted(float(angle) for angle in angles) == weights.flatten().tolist()
