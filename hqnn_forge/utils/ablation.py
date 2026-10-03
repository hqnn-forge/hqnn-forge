"""
hqnn_forge.utils.ablation
=========================
Switch off a hybrid model's quantum layer to measure what it contributes.

The question "is the circuit carrying signal, or would the classical layers
around it do as well on their own?" is answered by comparing a model with its
quantum layer against the same model with that layer replaced by a constant.
:func:`disable_quantum_layer` does the replacement for the duration of a
``with`` block, without retraining and without touching the weights.

The replacement output is ``fill`` (default 0.0) for every qubit.  0 is the
midpoint of the ⟨Z⟩ range [-1, 1], i.e. the readout of a maximally
uninformative qubit, so the head sees "no quantum information" rather than an
out-of-distribution value.  The circuit is not executed at all inside the
block, so an ablated evaluation is also much faster.

:func:`permute_quantum_layer` is the other null: the circuit runs, and its
readouts are shuffled across the batch, so they keep their distribution and
lose only their correspondence to the input.  Comparing the full model against
both says whether the head needs the readout's content or only its scale.

Example
-------
::

    from hqnn_forge.evaluation import find_optimal_threshold
    from hqnn_forge.models import ParallelHybridClassifier
    from hqnn_forge.utils import disable_quantum_layer

    model = ParallelHybridClassifier(n_input_features=30, n_qubits=8, n_layers=2)
    # ... train model ...
    full = find_optimal_threshold(y_val, model.predict_proba(X_val)).score
    with disable_quantum_layer(model):
        ablated = find_optimal_threshold(y_val, model.predict_proba(X_val)).score
    print(f"MCC with circuit {full:.3f}, without {ablated:.3f}")

Which topology
--------------
Only a model with a classical path around the circuit gives a meaningful
comparison.  In :class:`~hqnn_forge.models.ParallelHybridClassifier` the
classical branch still reaches the head, so the ablated score is the score of
that branch on its own.  In :class:`~hqnn_forge.models.HybridBinaryClassifier`
the quantum layer is the only path from input to head, so ablating it leaves
``head(full((n_qubits,), fill))``: one probability for every sample, a constant
predictor scoring 0 MCC by construction.  That number restates the topology and
measures nothing.  A classical baseline for the serial model has to be a
separately trained classical model, not this context manager.

For either topology, the classical control a reviewer asks for is a model of
comparable capacity trained *from scratch*: a head trained alongside the
circuit has adapted to its outputs, so ablating after training measures
dependence, not the best achievable classical performance.  (Training a serial
model from scratch inside the block trains its head's bias and nothing else.)
:func:`classical_baseline` builds that control.

Matched capacity
----------------
:func:`classical_baseline` returns an untrained
:class:`~hqnn_forge.models.ClassicalBaseline` whose trainable parameter count
is the closest it can get to the hybrid model's **live** count:
``count_parameters()`` minus the circuit weights that can never move the
output, ``circuit_summary(model).n_inert_params`` (#234).  Every other
rotation angle counts as one parameter.  A dead weight adds no capacity, so a
control matched on the total would get a live parameter for each one the
hybrid cannot use: the published SHNN has 122 trainable parameters, 102 of
them live, and a control matched on 122 would have 121, 19% more than the
hybrid's live count.  It does not claim an angle is worth a ``Linear``
weight; it holds the count fixed so that the comparison is about what the
parameters are, not how many there are.

``n_inert_params`` is structural and a lower bound: for strongly-entangling
layers with a ⟨Z_0⟩ readout it finds 16 of the 20 dead weights.  The target is
therefore an upper bound on the live count, and the control is never smaller
than one matched to the exact live count.  For the published SHNN both
targets, 106 and 102, give the same 101-parameter control.  Efficiency figures
(MCC/kParam) still divide by the total, the published convention; only the
match uses the live count.

The architecture keeps the hybrid's classical shape and replaces the circuit's
capacity with width:

* ``HybridBinaryClassifier`` → ``Linear(n_in → h) → tanh → Linear(h → 1)``,
  the serial model with its encoder, circuit and head replaced by one hidden
  layer, ``h`` chosen to match.
* ``ParallelHybridClassifier`` → its classical branch plus a head,
  ``Linear(n_in → w) → ReLU → Linear(w → w) → ReLU → Linear(w → 1)``, the
  branch widened from ``classical_hidden_dim`` to the ``w`` that matches.

Widths are integers, so the match is to within half the parameters one more
unit adds: ``(n_in + 2) / 2`` for the serial control, ``(2w + n_in + 4) / 2``
for the parallel one.  At 30 features, 8 qubits and 2 layers that bound is
5.4% of the serial model's 297 live parameters (305 in total: with the
all-qubit readout the 8 last-layer ``ω`` are inert; the control has 289) and
3.4% of the parallel model's 1081 (1089 in total; the control has 1061).  ``dropout_p`` is carried over, and so is
``init_seed``: a seeded hybrid gets a control seeded with the same seed, so the
pair is reproducible without touching the global torch RNG (#175), and an
unseeded one gets a control drawn from the global RNG as before.  For another
seed, rebuild it: ``ClassicalBaseline(**{**control.get_config(), "init_seed": s})``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    # Type-only: hqnn_forge.models imports hqnn_forge.utils.
    from hqnn_forge.models import ClassicalBaseline


def classical_baseline(model: nn.Module) -> ClassicalBaseline:
    """
    Untrained classical control for ``model``, matched in parameter count.

    See "Matched capacity" above for the architecture and the matching rule.
    Train it on the same data with the same recipe as the hybrid model; its
    score is the classical reference the hybrid's has to beat.

    Parameters
    ----------
    model:
        A ``HybridBinaryClassifier`` or ``ParallelHybridClassifier``.

    Returns
    -------
    ClassicalBaseline
        With ``count_parameters()`` within half a width step of ``model``'s
        live count, ``model.count_parameters() -
        circuit_summary(model).n_inert_params``, and ``model``'s
        ``n_input_features``,
        ``dropout_p`` and ``init_seed``.

    Raises
    ------
    TypeError
        If ``model`` is not one of the two binary hybrid classifiers.
    """
    # Imported here: hqnn_forge.models and hqnn_forge.diagnostics import
    # hqnn_forge.utils.
    from hqnn_forge.diagnostics import circuit_summary
    from hqnn_forge.models import (
        ClassicalBaseline,
        HybridBinaryClassifier,
        ParallelHybridClassifier,
    )
    from hqnn_forge.models.classical_baseline import mlp_parameter_count

    if isinstance(model, ParallelHybridClassifier):

        def shape(width: int) -> list[int]:
            return [width, width]

        activation = "relu"
    elif isinstance(model, HybridBinaryClassifier):

        def shape(width: int) -> list[int]:
            return [width]

        activation = "tanh"
    else:
        raise TypeError(
            f"classical_baseline expects a HybridBinaryClassifier or "
            f"ParallelHybridClassifier; got {type(model).__name__}."
        )

    config = model.get_config()
    n_in = config["n_input_features"]
    # Matched on the live count (#234): a weight that can never move the output
    # adds no capacity for the control to match.
    target = model.count_parameters() - circuit_summary(model).n_inert_params

    def count(width: int) -> int:
        return mlp_parameter_count(n_in, shape(width))

    # count is increasing in width: step up to the first width at or past the
    # target, then keep whichever of it and the one below is closer (the
    # smaller on a tie, so the control is never the larger model by choice).
    width = 1
    while count(width) < target:
        width += 1
    if width > 1 and target - count(width - 1) <= count(width) - target:
        width -= 1
    return ClassicalBaseline(
        n_in,
        shape(width),
        activation=activation,
        dropout_p=config["dropout_p"],
        init_seed=config["init_seed"],
    )


@contextmanager
def disable_quantum_layer(model: nn.Module, fill: float = 0.0) -> Iterator[nn.Module]:
    """
    Replace ``model.quantum_layer``'s output with the constant ``fill``.

    Inside the block the layer returns a tensor of shape
    ``(*batch_dims, n_outputs)`` filled with ``fill``, in the input's dtype and
    device, without running the circuit.  ``n_outputs`` is the layer's readout
    width -- ``n_qubits`` for ``readout="all"``, 1 for ``readout="first"`` -- so
    the head downstream sees the width it was built for.  The output therefore
    carries no gradient to the quantum weights or to anything upstream of it.  The
    original ``forward`` is restored on exit, including when the block raises.

    Training inside the block is allowed and trains only the parameters that
    still influence the loss.

    Parameters
    ----------
    model:
        A hybrid classifier with a ``quantum_layer`` attribute whose layer has
        ``n_qubits``.
    fill:
        Constant output per qubit.  Must lie in [-1, 1], the range of a Pauli-Z
        expectation.  Default: 0.0.

    Yields
    ------
    nn.Module
        The disabled quantum layer.

    Raises
    ------
    TypeError
        If ``model`` has no suitable ``quantum_layer``.
    ValueError
        If ``fill`` is outside [-1, 1], or, inside the block, if the layer is
        called with an input whose last dimension is not ``n_qubits``.
    RuntimeError
        If the layer's ``forward`` is already replaced (nested use on the same
        model, including inside :func:`permute_quantum_layer`).
    """
    if not -1.0 <= fill <= 1.0:
        raise ValueError(f"fill must lie in [-1, 1], the range of <Z>; got {fill}.")

    def constant(layer: nn.Module, n_qubits: int, n_outputs: int) -> Forward:
        def constant_forward(x: torch.Tensor) -> torch.Tensor:
            # The encoding layers reject an input whose width is not n_qubits; the
            # replacement has to reject it too, or an ablated run silently returns
            # numbers for input the full model refuses.
            if x.shape[-1] != n_qubits:
                raise ValueError(
                    f"Input feature dimension {x.shape[-1]} does not match n_qubits={n_qubits}."
                )
            return torch.full((*x.shape[:-1], n_outputs), fill, dtype=x.dtype, device=x.device)

        return constant_forward

    with _replace_forward(model, "disable_quantum_layer", constant) as layer:
        yield layer


@contextmanager
def permute_quantum_layer(
    model: nn.Module, generator: torch.Generator | None = None
) -> Iterator[nn.Module]:
    """
    Shuffle ``model.quantum_layer``'s output across the batch.

    The permutation null of feature-importance work.  Inside the block the
    circuit still runs, and its ``(batch, n_outputs)`` output is returned with
    the rows permuted: every sample receives some other sample's quantum
    readout.  The readouts keep their distribution -- the same rows, the same
    spread -- and lose only their correspondence to the input, so the head
    sees in-distribution values that carry no information about *this*
    sample.  :func:`disable_quantum_layer` removes both at once: the
    correspondence and all variance.  A model that scores the same under both
    nulls depends on the readout's scale, not its content; one that recovers
    under the permutation needs something non-constant in those slots.

    Unlike the constant null, this one is meaningful for
    ``HybridBinaryClassifier``: a permuted serial model is not a constant
    predictor.  Each forward pass draws a fresh permutation of its batch, so
    score a validation set in one batch, or accept that the null is drawn per
    batch.

    The permuted output is detached: no gradient reaches the quantum weights
    or anything upstream of the layer.  The original ``forward`` is restored
    on exit, including when the block raises.

    Parameters
    ----------
    model:
        A hybrid classifier with a ``quantum_layer`` attribute whose layer has
        ``n_qubits``.
    generator:
        Source of the permutations.  Pass a seeded one: the ablated score is a
        random variable, and without a seed it cannot be reproduced.  ``None``
        draws from the global torch RNG.

    Yields
    ------
    nn.Module
        The permuted quantum layer.

    Warns
    -----
    RuntimeWarning
        When a forward pass leaves the output unchanged: a batch of one sample
        (or an unbatched input), or a draw of the identity permutation.  The
        "ablated" output is then the real one, and a score computed from it
        says nothing about the null.

    Raises
    ------
    TypeError
        If ``model`` has no suitable ``quantum_layer``.
    RuntimeError
        If the layer's ``forward`` is already replaced (nested use on the same
        model, including inside :func:`disable_quantum_layer`).
    """

    def permuted(layer: nn.Module, n_qubits: int, n_outputs: int) -> Forward:
        # Looked up on the class: the instance attribute is about to shadow it.
        real_forward = type(layer).forward.__get__(layer)

        def permuted_forward(x: torch.Tensor) -> torch.Tensor:
            out = real_forward(x).detach()
            batch = out.shape[0] if out.ndim > 1 else 1
            perm = torch.randperm(batch, generator=generator)
            if batch < 2 or torch.equal(perm, torch.arange(batch)):
                warnings.warn(
                    f"permute_quantum_layer left a batch of {batch} unchanged "
                    f"({'one sample has nothing to swap with' if batch < 2 else 'the identity permutation was drawn'}); "
                    f"the ablated output is the real one for this forward pass.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return out if out.ndim < 2 else out[perm.to(out.device)]

        return permuted_forward

    with _replace_forward(model, "permute_quantum_layer", permuted) as layer:
        yield layer


Forward = Callable[[torch.Tensor], torch.Tensor]


@contextmanager
def _replace_forward(
    model: nn.Module, caller: str, make_forward: Callable[[nn.Module, int, int], Forward]
) -> Iterator[nn.Module]:
    """
    Shadow ``model.quantum_layer.forward`` with ``make_forward(layer, n_qubits,
    n_outputs)`` for the block, and restore it on exit.

    The two ablations share this, so they check the model, refuse nesting and
    restore ``forward`` identically.
    """
    layer = getattr(model, "quantum_layer", None)
    n_qubits = getattr(layer, "n_qubits", None)
    if not isinstance(layer, nn.Module) or not isinstance(n_qubits, int):
        raise TypeError(
            f"{caller} expects a model with a quantum_layer attribute; got {type(model).__name__}."
        )
    # The readout decides how wide the layer's output is, and the head is built
    # for that width; filling n_qubits wide would break readout="first".
    n_outputs = getattr(layer, "n_outputs", None)
    if not isinstance(n_outputs, int):
        n_outputs = n_qubits
    if "forward" in vars(layer):
        raise RuntimeError(
            f"{type(layer).__name__}.forward is already overridden on this instance; "
            f"{caller} cannot be nested with itself or with the other ablation."
        )

    # nn.Module.__call__ dispatches to self.forward, so an instance attribute
    # shadows the class method for this layer only.
    layer.forward = make_forward(layer, n_qubits, n_outputs)  # type: ignore[method-assign]
    try:
        yield layer
    finally:
        del layer.forward
