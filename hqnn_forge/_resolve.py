"""
hqnn_forge._resolve
===================
Find the encoding layer inside the object a public function was given.

Every diagnostic or wrapper that takes "a model or a layer" -- circuit_summary,
gradient_variance, the Fisher diagnostics, apply_depolarizing_noise, the
quantum kernels -- needs the same walk: unwrap a hybrid classifier's
``quantum_layer``, then check for a ``qlayer`` TorchLayer and an integer
``n_qubits``.  Each used to carry its own copy, and the copies drifted (#182):
one skipped the ``nn.Module`` check, and each hardcoded a function name into
its error message, which then named the wrong function when a second caller
reused it.  Callers keep only what is specific to them on top of the result.

What an encoding layer is, is :mod:`hqnn_forge._encoding_contract`:
:func:`resolve_encoding_layer` finds a :class:`CircuitLayer`, and
:func:`require_prepare_inputs` narrows it to an :class:`EncodingLayer` for a
caller that replays the circuit on its own inputs.
"""

from __future__ import annotations

import pennylane as qml

from hqnn_forge._encoding_contract import (
    CircuitLayer,
    EncodingLayer,
    is_circuit_layer,
    is_encoding_layer,
)


def resolve_encoding_layer(
    target: object, caller: str, *, allow_model: bool = True
) -> tuple[CircuitLayer, qml.qnn.TorchLayer, int]:
    """
    ``(layer, qlayer, n_qubits)`` for the encoding layer ``target`` is or holds.

    Parameters
    ----------
    target:
        A :class:`~hqnn_forge.encoding.CircuitLayer` (an ``nn.Module``
        with a ``qlayer`` TorchLayer and an integer ``n_qubits``) or, with ``allow_model``, a hybrid classifier
        holding one as ``quantum_layer``.
    caller:
        The public function's name, for the error message.
    allow_model:
        Unwrap ``quantum_layer``.  Off for a caller defined by the encoder
        alone, which refuses a classifier with a message saying why.

    Raises
    ------
    TypeError
        If no encoding layer is found.
    """
    if not allow_model and hasattr(target, "quantum_layer"):
        raise TypeError(
            f"{caller} expects an encoding layer, not a hybrid classifier "
            f"({type(target).__name__}): the classifier's classical encoder runs before "
            f"its quantum layer, so the result for its quantum_layer on raw inputs would "
            f"describe a different feature map.  Pass model.quantum_layer, with inputs "
            f"already encoded, if that is what you mean."
        )
    layer = getattr(target, "quantum_layer", target) if allow_model else target
    if not is_circuit_layer(layer):
        expected = (
            "an encoding layer or a hybrid classifier with a quantum_layer attribute"
            if allow_model
            else "an encoding layer"
        )
        raise TypeError(
            f"{caller} expects {expected} (QuantumEncodingLayer, IQPEncodingLayer, "
            f"AmplitudeEncodingLayer, DataReuploadingLayer); got {type(target).__name__}."
        )
    return layer, layer.qlayer, layer.n_qubits


def require_prepare_inputs(layer: CircuitLayer, caller: str) -> EncodingLayer:
    """
    ``layer`` as an :class:`EncodingLayer`, or raise.

    For a caller that runs the circuit on inputs of its own, which has to apply
    the layer's classical step exactly as ``forward`` does.

    Raises
    ------
    TypeError
        If ``layer`` has no callable ``prepare_inputs`` or no ``int``
        ``n_features``.
    """
    if not is_encoding_layer(layer):
        raise TypeError(
            f"{caller} expects an encoding layer with a prepare_inputs method and an "
            f"int n_features, which {type(layer).__name__} does not have."
        )
    return layer
