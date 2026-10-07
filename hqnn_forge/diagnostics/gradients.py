"""
hqnn_forge.diagnostics.gradients
================================
Empirical gradient variance of an encoding layer, to see barren plateaus.

A barren plateau shows up as the variance of the cost gradient over random
parameter draws shrinking exponentially with the number of qubits.
:func:`gradient_variance` estimates that variance for one layer
configuration and one initialisation scheme; :func:`gradient_variance_sweep`
repeats it over qubit and layer counts so the trend is visible, and
:func:`format_sweep` prints the result as a table.

Estimator
---------
For each of ``n_samples`` draws the layer's rotation angles (its ``weights``
tensor) are re-initialised with ``init`` -- any other trainable tensor, such
as ``input_scaling``, keeps its values -- an input is drawn uniformly from
``[-input_scale, input_scale]^n``, the cost is evaluated for that single
sample and its gradient with respect to every quantum weight, in every
trainable tensor, is recorded.  The sample variance is taken per weight;
``total_variance`` is its sum over weights (the variance of the gradient
vector, which does not shrink just because a larger circuit has more
parameters) and ``mean_variance`` its mean.  The default cost is ⟨Z_0⟩, a
single-qubit observable; whether it acts as a local cost in the sense of
Cerezo et al. (2021) depends on the circuit's light cone (see below).

What the library's own circuits show
------------------------------------
Measured with this tool on ``QuantumEncodingLayer`` (2 layers, 100–200
samples, ``default.qubit``):

* ``total_variance`` falls by roughly 5x from 2 to 6 qubits under uniform
  init, even with the local cost at 2 layers.  The CNOT ring is a cascade, so
  the backward light cone of Z_0 covers every qubit within one layer and the
  "local" cost behaves like a global one.  ``entangler="brickwork"`` is the
  exception: its light cone does not grow with the register, and its
  ``total_variance`` stays flat from 4 to 8 qubits at 2 layers (see
  :mod:`hqnn_forge.initializers.restricted_variance`).
* With inputs spread over (-π, π) -- what both classifiers produce, and what
  ``PCANormalizer(scale_to_pi=True)`` produces -- the restricted-variance init
  gives the same gradient variance as uniform init: the angle embedding
  already randomises the state.  Only with inputs near 0 does the restricted
  init retain more variance (1.67–1.89x at 8 qubits over 5 seeds, 1.09x at
  4; see :mod:`hqnn_forge.initializers.restricted_variance`).

The layer is run in eval mode, so a layer built with ``noise_level > 0``
is measured on its noiseless circuit, not the train-mode ``default.mixed``
one; its mode and weights are restored when the estimate finishes.  To
measure a noise-induced plateau, run the estimate inside
:func:`hqnn_forge.noise.apply_depolarizing_noise`.  The Fisher diagnostics
follow the same rule.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from hqnn_forge._encoding_contract import CircuitLayer
from hqnn_forge._resolve import resolve_encoding_layer
from hqnn_forge.diagnostics.circuit import input_width
from hqnn_forge.initializers import block_local_init_, restricted_normal_init_
from hqnn_forge.initializers.restricted_variance import _not_restricting_ignored
from hqnn_forge.utils.modes import eval_mode
from hqnn_forge.utils.rng import seeded_rng

InitName = Literal["uniform", "restricted", "block_local"]
InitFn = Callable[[torch.Tensor], Any]
CostFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GradientVarianceResult:
    """
    Gradient-variance estimate for one layer configuration.

    ``per_parameter`` and ``per_tensor`` are excluded from ``==`` and
    ``hash``: comparing them would return a Tensor rather than a bool, so two
    results compare on their scalar fields only.

    Attributes
    ----------
    layer_type, n_qubits, n_layers:
        What was measured.
    init:
        Name of the initialisation scheme (``"custom"`` for a callable).
    input_scale:
        Inputs were drawn from ``[-input_scale, input_scale]``.
    n_samples:
        Number of random draws.
    total_variance:
        Sum of the per-entry gradient variance over every trainable tensor of
        the layer, i.e. over the whole gradient vector.
    mean_variance:
        ``total_variance`` divided by the number of trainable entries.
    per_parameter:
        Per-entry variance.  For a layer with one trainable tensor, the
        common case, it has that tensor's shape.  With several, it is the
        flat concatenation of ``per_tensor``'s values, in that mapping's
        order.
    per_tensor:
        Per-entry variance per trainable tensor, keyed by the TorchLayer
        argument name (``"weights"``, ``"input_scaling"``), each shaped like
        its tensor.
    """

    layer_type: str
    n_qubits: int
    n_layers: int
    init: str
    input_scale: float
    n_samples: int
    total_variance: float
    mean_variance: float
    per_parameter: torch.Tensor = field(compare=False)
    per_tensor: Mapping[str, torch.Tensor] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Scalar fields only, for logging."""
        return {
            "layer_type": self.layer_type,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "init": self.init,
            "input_scale": self.input_scale,
            "n_samples": self.n_samples,
            "total_variance": self.total_variance,
            "mean_variance": self.mean_variance,
        }


def _resolve_layer(
    target: nn.Module, caller: str = "gradient_variance"
) -> tuple[CircuitLayer, dict[str, torch.Tensor], int]:
    """
    Return ``(layer, tensors, n_qubits)`` for the layer inside *target*.
    *caller* names the public function in the error messages.

    ``tensors`` is the TorchLayer's ``qnode_weights`` mapping, every trainable
    argument by name, read the same way
    :func:`~hqnn_forge.diagnostics.circuit.circuit_summary` resolves a layer.
    Which tensor holds the rotation angles is not decided here, so a caller
    that draws none (the Fisher matrix) accepts any set of tensors.
    """
    layer, qlayer, n_qubits = resolve_encoding_layer(target, caller)
    tensors = dict(qlayer.qnode_weights.items())
    return layer, tensors, n_qubits


def _resolve_tensors(
    target: nn.Module, caller: str = "gradient_variance"
) -> tuple[CircuitLayer, dict[str, torch.Tensor], str, int, int]:
    """
    Return ``(layer, tensors, angles, n_qubits, n_layers)`` for the layer
    inside *target*, as :func:`_resolve_layer` does, plus the angle tensor.

    ``angles`` names the rotation-angle tensor the init strategies draw:
    ``"weights"``, or the only tensor there is.  A layer with several tensors
    and none named ``weights`` is refused, since which of them is the angles
    would be a guess.  ``n_layers`` falls back to the angle tensor's first
    dimension when the layer has no ``n_layers`` attribute.
    """
    layer, tensors, n_qubits = _resolve_layer(target, caller)
    if len(tensors) == 1:
        (angles,) = tensors
    elif "weights" in tensors:
        angles = "weights"
    else:
        raise ValueError(
            f"{caller}: {type(layer).__name__} has {len(tensors)} trainable tensors "
            f"({', '.join(sorted(tensors))}) and none named 'weights', so which of them "
            f"holds the rotation angles the init draws is ambiguous."
        )
    n_layers = getattr(layer, "n_layers", None)
    return (
        layer,
        tensors,
        angles,
        n_qubits,
        n_layers if isinstance(n_layers, int) else int(tensors[angles].shape[0]),
    )


def _make_init(
    init: InitName | InitFn, n_qubits: int, n_layers: int, generator: torch.Generator
) -> tuple[str, InitFn]:
    if callable(init):
        return "custom", init
    if init == "uniform":
        return init, lambda w: w.copy_(
            torch.rand(w.shape, generator=generator, dtype=w.dtype) * 2 * math.pi
        )
    if init == "restricted":

        def restricted(w: torch.Tensor) -> None:
            # Small sizes are measured on purpose here, not a misuse.
            with _seeded(generator), _not_restricting_ignored():
                restricted_normal_init_(w, n_qubits=n_qubits, n_layers=n_layers)

        return init, restricted
    if init == "block_local":

        def block_local(w: torch.Tensor) -> None:
            # Small sizes are measured on purpose here, not a misuse.
            with _seeded(generator), _not_restricting_ignored():
                block_local_init_(w, n_qubits=n_qubits)

        return init, block_local
    raise ValueError(
        f"unknown init {init!r}; choose 'uniform', 'restricted', 'block_local' or pass a callable."
    )


@contextmanager
def _seeded(generator: torch.Generator) -> Iterator[None]:
    """
    Run the library initialisers (which use the global RNG) from ``generator``:
    a seed drawn from it drives :func:`hqnn_forge.utils.rng.seeded_rng`, which
    restores the caller's CPU and CUDA RNG state afterwards.
    """
    seed = int(torch.randint(0, 2**62, (1,), generator=generator))
    with seeded_rng(seed):
        yield


def _local_z0(outputs: torch.Tensor) -> torch.Tensor:
    return outputs[..., 0].sum()


def gradient_variance(
    target: nn.Module,
    n_samples: int = 100,
    *,
    init: InitName | InitFn = "uniform",
    input_scale: float = math.pi,
    cost_fn: CostFn | None = None,
    generator: torch.Generator | None = None,
) -> GradientVarianceResult:
    """
    Estimate the variance of the cost gradient over random weight draws.

    Parameters
    ----------
    target:
        An encoding layer, or a hybrid classifier (its ``quantum_layer`` is
        used and inputs are fed to it directly, bypassing the classical
        encoder).
    n_samples:
        Random draws.  At least 2.  Default: 100.
    init:
        ``"uniform"`` over [0, 2π) (the barren-plateau reference),
        ``"restricted"``, ``"block_local"``, or a callable that fills the
        weight tensor in place.  Only the rotation angles are drawn: the
        tensor named ``weights`` (or the layer's only tensor).  Any other
        trainable tensor -- ``DataReuploadingLayer``'s ``input_scaling`` --
        keeps its current values for every draw, since an angle distribution
        means nothing for a scale factor, but its gradient is measured and
        counted in ``total_variance`` like the rest of the gradient vector.
    input_scale:
        Inputs are uniform in ``[-input_scale, input_scale]``, one per input
        feature (``n_features`` for the amplitude encoder, else one per
        qubit), and go through the layer's ``prepare_inputs``.  ``π`` matches
        what the classifiers feed the circuit; ``0`` feeds zeros, which the
        amplitude encoder refuses (no state has zero norm).
    cost_fn:
        Maps the layer output of shape ``(1, n_qubits)`` to a scalar.
        Default: ⟨Z_0⟩.
    generator:
        Source of randomness for weights and inputs.  The global RNG state is
        left untouched either way.

    Returns
    -------
    GradientVarianceResult
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2 to estimate a variance; got {n_samples}.")
    if input_scale < 0:
        raise ValueError(f"input_scale must be >= 0; got {input_scale}.")
    layer, tensors, angles, n_qubits, n_layers = _resolve_tensors(target)
    weights = tensors[angles]
    # One value per input feature, which is not one per qubit for the
    # amplitude encoder; forward's prepare_inputs pads and normalises them.
    width = input_width(layer)
    gen = generator if generator is not None else torch.Generator().manual_seed(0)
    init_name, init_fn = _make_init(init, n_qubits, n_layers, gen)
    cost = cost_fn if cost_fn is not None else _local_z0

    names = list(tensors)
    originals = {name: (t.detach().clone(), t.grad) for name, t in tensors.items()}
    grads = {
        name: torch.empty((n_samples, *t.shape), dtype=torch.float64)
        for name, t in tensors.items()
    }
    # resolve_encoding_layer checked this; the narrowing adds the module API
    # the CircuitLayer protocol cannot declare.
    assert isinstance(layer, nn.Module)
    try:
        with eval_mode(layer):
            for s in range(n_samples):
                with torch.no_grad():
                    init_fn(weights)
                x = (torch.rand(1, width, generator=gen) * 2 - 1) * input_scale
                for t in tensors.values():
                    t.grad = None
                value = cost(layer(x))
                if not isinstance(value, torch.Tensor):
                    # existing behaviour; switching to TypeError is not a style change
                    raise ValueError(  # noqa: TRY004
                        f"cost_fn must return a 0-d Tensor to differentiate; "
                        f"got {type(value).__name__}."
                    )
                if value.ndim != 0:
                    raise ValueError(
                        f"cost_fn must return a scalar; got shape {tuple(value.shape)}."
                    )
                # allow_unused: a tensor the cost cannot reach has gradient 0
                # for this draw, not an undefined one.
                sample = torch.autograd.grad(
                    value, [tensors[name] for name in names], allow_unused=True
                )
                for name, grad in zip(names, sample, strict=True):
                    grads[name][s] = 0.0 if grad is None else grad.detach().to(torch.float64)
    finally:
        with torch.no_grad():
            for name, (value_before, grad_before) in originals.items():
                tensors[name].copy_(value_before)
                tensors[name].grad = grad_before

    per_tensor = {name: g.var(dim=0) for name, g in grads.items()}
    per_parameter = (
        per_tensor[names[0]]
        if len(names) == 1
        else torch.cat([v.reshape(-1) for v in per_tensor.values()])
    )
    return GradientVarianceResult(
        layer_type=type(layer).__name__,
        n_qubits=n_qubits,
        n_layers=n_layers,
        init=init_name,
        input_scale=float(input_scale),
        n_samples=n_samples,
        total_variance=float(per_parameter.sum()),
        mean_variance=float(per_parameter.mean()),
        per_parameter=per_parameter,
        per_tensor=per_tensor,
    )


def gradient_variance_sweep(
    build: Callable[[int, int], nn.Module],
    qubit_counts: Iterable[int],
    layer_counts: Iterable[int] = (2,),
    **kwargs: Any,
) -> list[GradientVarianceResult]:
    """
    :func:`gradient_variance` for every (n_qubits, n_layers) combination.

    Parameters
    ----------
    build:
        ``build(n_qubits, n_layers)`` returns a fresh layer or model, e.g.
        ``lambda q, l: QuantumEncodingLayer(q, l, device_name="default.qubit",
        diff_method="backprop")``.
    qubit_counts, layer_counts:
        Grid to sweep.
    **kwargs:
        Passed to :func:`gradient_variance` (``init``, ``n_samples``, ...).
        A ``generator`` is shared across the sweep.

    Returns
    -------
    list[GradientVarianceResult]
        One per combination, qubits in the outer loop.
    """
    layer_counts = list(layer_counts)
    return [gradient_variance(build(q, l), **kwargs) for q in qubit_counts for l in layer_counts]


def format_sweep(results: Sequence[GradientVarianceResult]) -> str:
    """
    Plain-text table of a sweep, with the ratio to the previous row of the
    same init and layer count so exponential decay is readable at a glance.
    """
    header = (
        f"{'init':<12} {'qubits':>6} {'layers':>6} {'total var':>12} {'mean var':>12} {'ratio':>7}"
    )
    lines = [header, "-" * len(header)]
    previous: dict[tuple[str, int], float] = {}
    for r in results:
        key = (r.init, r.n_layers)
        prev = previous.get(key)
        # `is not None`, not truthiness: a previous row of exactly zero
        # variance (a stationary point) is still a previous row.  Dividing by
        # it is undefined, so that one case prints a marker; only a genuinely
        # absent previous row leaves the column blank.
        if prev is None:
            ratio = f"{'':>7}"
        elif prev == 0.0:
            ratio = f"{'n/a':>7}"
        else:
            ratio = f"{r.total_variance / prev:7.3f}"
        previous[key] = r.total_variance
        lines.append(
            f"{r.init:<12} {r.n_qubits:>6} {r.n_layers:>6} "
            f"{r.total_variance:>12.4e} {r.mean_variance:>12.4e} {ratio}"
        )
    return "\n".join(lines)
