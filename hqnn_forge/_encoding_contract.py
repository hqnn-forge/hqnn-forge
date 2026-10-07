"""
hqnn_forge._encoding_contract
=============================
The interface every encoding layer implements, and what relies on it.

Public as ``hqnn_forge.encoding.EncodingLayer`` and friends.  It lives outside
the ``encoding`` package because ``hqnn_forge._resolve`` needs it, and
importing anything under ``hqnn_forge.encoding`` runs the package's
``__init__``, whose encoders import ``hqnn_forge.noise``, which imports
``hqnn_forge._resolve``: a cycle.

An encoding layer is an ``nn.Module`` with

``qlayer``
    The ``qml.qnn.TorchLayer`` holding the circuit.  Its QNode's first
    argument is ``inputs``; everything else is a trainable weight.
``n_qubits``
    The circuit's width, an ``int``.
``n_features``
    The input's width, an ``int``: the number of features per sample that
    ``prepare_inputs`` accepts.  ``n_qubits`` for the angle-type layers, up to
    ``2**n_qubits`` for the amplitude layer.
``prepare_inputs(x)``
    The whole classical step between a batch ``x`` and the QNode: input
    validation (width, finiteness) and any transform (the amplitude layer's
    padding and normalisation).  It is the *only* place for classical input
    handling, so that, noise aside, ``forward(x)`` is exactly
    ``qlayer(prepare_inputs(x))``.

:class:`CircuitLayer` (``qlayer`` and ``n_qubits``) is the minimum that
:func:`hqnn_forge._resolve.resolve_encoding_layer` accepts, and all that
``circuit_summary``, ``gradient_variance``, the Fisher diagnostics and
``apply_depolarizing_noise`` require.  A caller that feeds the circuit inputs
of its own, such as :mod:`hqnn_forge.kernels`, requires
:class:`EncodingLayer`: ``n_features`` to know how wide those inputs are, and
``prepare_inputs`` to process them exactly as ``forward`` does.  A layer that
validated or transformed inputs inline in ``forward`` would have the kernel
skip that step and compute the kernel of a different feature map, without any
error; keeping the step in ``prepare_inputs`` is what makes the replay
faithful.

The one sanctioned difference between ``forward(x)`` and
``qlayer(prepare_inputs(x))`` is noise, in two forms.  Training-time noise: an
encoding layer built with ``noise_level > 0`` runs a noisy circuit in
``train()`` mode, and one built with ``readout_error`` passes the circuit's
output through that readout error in ``train()`` mode.  And an open
:func:`hqnn_forge.noise.apply_readout_error` block, which does the same in
either mode.  In ``eval()`` mode outside such a block, or without any of
these, the two are identical.  A readout error acts on the output, not in the
circuit, so a caller that replays ``qlayer(prepare_inputs(x))`` never sees
it.

Why not ``isinstance``
----------------------
These are static ``Protocol`` classes, not ``runtime_checkable`` ones.  From
Python 3.12, ``isinstance`` against a runtime protocol looks attributes up with
``inspect.getattr_static``, which does not see an ``nn.Module``'s submodules:
``qlayer`` lives in ``_modules`` and is reached through ``__getattr__``, so
every encoder in this package would fail the check.  :func:`is_encoding_layer`
does the runtime check with ordinary attribute access instead, and narrows the
type for the checker.  A protocol cannot also say "is an ``nn.Module``"; a
caller that needs the module API as well narrows once more with
``isinstance(layer, nn.Module)``, which the check has already guaranteed.
"""

from __future__ import annotations

from typing import Protocol, TypeGuard

import pennylane as qml
import torch
import torch.nn as nn

__all__ = ["CircuitLayer", "EncodingLayer", "is_circuit_layer", "is_encoding_layer"]


class CircuitLayer(Protocol):
    """An ``nn.Module`` with a ``qlayer`` TorchLayer on ``n_qubits`` wires."""

    qlayer: qml.qnn.TorchLayer
    n_qubits: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class EncodingLayer(CircuitLayer, Protocol):
    """
    A :class:`CircuitLayer` taking ``n_features`` inputs through ``prepare_inputs``.

    ``forward(x)`` is ``qlayer(prepare_inputs(x))``, except for training-time
    noise in ``train()`` mode and a readout error applied to the output (see
    the module docstring).
    """

    n_features: int

    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and transform a batch exactly as ``forward`` does before the QNode."""
        ...


def is_circuit_layer(obj: object) -> TypeGuard[CircuitLayer]:
    """
    Whether ``obj`` is an ``nn.Module`` with a TorchLayer ``qlayer`` and an ``int`` ``n_qubits``.

    ``bool`` is refused as ``n_qubits`` although it is an ``int`` subclass.
    """
    n_qubits = getattr(obj, "n_qubits", None)
    return (
        isinstance(obj, nn.Module)
        and isinstance(getattr(obj, "qlayer", None), qml.qnn.TorchLayer)
        and isinstance(n_qubits, int)
        and not isinstance(n_qubits, bool)
    )


def is_encoding_layer(obj: object) -> TypeGuard[EncodingLayer]:
    """
    Whether ``obj`` is a :class:`CircuitLayer` with ``n_features`` and ``prepare_inputs``.

    ``n_features`` must be an ``int`` (``bool`` refused, as for ``n_qubits``)
    and ``prepare_inputs`` callable.
    """
    n_features = getattr(obj, "n_features", None)
    return (
        is_circuit_layer(obj)
        and isinstance(n_features, int)
        and not isinstance(n_features, bool)
        and callable(getattr(obj, "prepare_inputs", None))
    )
