"""
hqnn_forge.diagnostics.expressibility
=====================================
Expressibility and entangling capability of an encoding layer (Sim et al. 2019).

Both are properties of the *family* of states a circuit reaches as its
parameters vary, measured before any training, and are the standard way to
compare ansätze and encodings of the same size.

Expressibility
--------------
Draw pairs of parameter settings, prepare both states and record the fidelity
``F = |⟨ψ(θ)|ψ(φ)⟩|²``.  For Haar-random states on ``N = 2^n`` amplitudes, ``F``
has density ``P_Haar(F) = (N − 1)(1 − F)^(N − 2)``.  Expressibility is the KL
divergence of the circuit's fidelity histogram from that distribution::

    Expr = D_KL(P_circuit ‖ P_Haar) = Σ_i p_i · ln(p_i / q_i)

over ``n_bins`` equal bins of ``[0, 1]``.  **Lower is more expressive**: 0 is
Haar-like, and a circuit that always prepares the same state (every ``F = 1``)
scores the maximum ``(N − 1) · ln(n_bins)``.  ``q_i`` is the exact Haar
probability of bin ``i``, ``(1 − a)^(N−1) − (1 − b)^(N−1)`` for bin ``[a, b)``,
rather than the density at the bin centre, so the maximum is exact and nothing
is lost in the narrow high-fidelity bins where the Haar density is steep.

A histogram estimate of a KL divergence is biased upwards by roughly
``(k − 1) / (2 · n_pairs)`` for ``k`` occupied bins, so compare circuits at the
same ``n_pairs`` and ``n_bins``.  Sim et al. use 5000 pairs and 75 bins, the
defaults here.

Entangling capability
---------------------
The mean Meyer–Wallach measure over sampled states::

    Q(ψ) = 2 · (1 − (1/n) Σ_k Tr ρ_k²)

where ``ρ_k`` is the reduced state of qubit ``k``.  ``Q = 0`` exactly for
product states and ``Q = 1`` for, e.g., the GHZ state.  For Haar-random states
``E[Q] = (N − 2) / (N + 1)`` (Scott 2004), the reference a circuit's value is
usually read against.

Sampling
--------
Every trainable tensor of the layer (``weights``, and ``input_scaling`` for a
re-uploading layer) is drawn uniformly from ``[0, 2π)`` for each state, Sim et
al.'s convention.  The inputs are either drawn uniformly from ``(−π, π)`` per
state (``inputs="random"``, the default: the family the model can reach over
its data and weights), or one fixed vector for every state (Sim et al.'s
data-free setting: the ansatz's expressibility around that input).  Inputs go
through the layer's ``prepare_inputs``, and the states are replayed on a
state-vector simulator exactly as :mod:`hqnn_forge.kernels` does; the layer's
own weights are left untouched.  Each state's circuit is built as its own
tape, with its own draws in the same order as ever, and each chunk of tapes
is then run as one broadcast tape: the tapes differ only in their parameters,
so stacking those gives one circuit over the whole chunk (#361).  That skips
PennyLane's per-tape execution overhead, about 3 ms per state on
``default.qubit``: the default 5000 pairs of a 2-layer angle layer went from
35 s to 8 s at 4 qubits and from 78 s to 15 s at 8, the rest being the
building of the tapes and the comparison of their structure.  A chunk is
merged only when its tapes agree in everything but the values of their
parameters (operations, wires, parameter shapes and hyperparameters) and every
operation is known to apply a stacked parameter as a batch, itself or through
its decomposition; the states are then those of running the tapes one by one
up to round-off.  For every encoding layer of this package they are checked to
be bit-identical at 3 qubits; at 12 qubits an IQP layer's differ by a few
1e-17.  Any other chunk, and one whose broadcast run raises, runs tape by tape
as before.

References
----------
* Sim et al. (2019) "Expressibility and entangling
  capability of parameterized quantum circuits for hybrid quantum-classical
  algorithms", Adv. Quantum Technol. 2, 1900070.
* Meyer & Wallach (2002) "Global entanglement in multiparticle systems",
  J. Math. Phys. 43, 4273.
* Brennen (2003) "An observable measure of entanglement for pure states of
  multi-qubit systems", Quantum Inf. Comput. 3, 619 (the purity form of Q).
* Scott (2004) "Multipartite entanglement, quantum-error-correcting codes,
  and entangling power of quantum evolutions", PRA 69, 052330.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import pennylane as qml
import torch
from pennylane.operation import Operator
from pennylane.ops.functions import bind_new_parameters
from torch import nn

from hqnn_forge.kernels import _prepare, _resolve_layer

__all__ = [
    "EntanglingCapabilityResult",
    "ExpressibilityResult",
    "entangling_capability",
    "expressibility",
    "expressibility_from_fidelities",
    "haar_fidelity_bin_probabilities",
    "meyer_wallach",
]

Inputs = Literal["random"] | torch.Tensor

#: Tapes executed per call to the simulator: bounds memory, not the result.
_CHUNK = 512


@dataclass(frozen=True)
class ExpressibilityResult:
    """
    Attributes
    ----------
    expressibility:
        ``D_KL(P_circuit ‖ P_Haar)`` in nats; lower is more expressive.
    maximum:
        ``(N − 1) · ln(n_bins)``, the value of a circuit whose state never
        changes, for reading ``expressibility`` on a scale.
    fidelities:
        The ``n_pairs`` sampled fidelities, float64.
    """

    layer_type: str
    n_qubits: int
    n_pairs: int
    n_bins: int
    expressibility: float
    maximum: float
    fidelities: torch.Tensor


@dataclass(frozen=True)
class EntanglingCapabilityResult:
    """
    Attributes
    ----------
    entangling_capability:
        Mean Meyer–Wallach ``Q`` over the sampled states, in ``[0, 1]``.
    standard_error:
        Standard error of that mean.
    haar_reference:
        ``(N − 2) / (N + 1)``, the expected ``Q`` of a Haar-random state.
    values:
        ``Q`` of every sampled state, float64.
    """

    layer_type: str
    n_qubits: int
    n_samples: int
    entangling_capability: float
    standard_error: float
    haar_reference: float
    values: torch.Tensor


# ---------------------------------------------------------------------------
# The formulas, on fidelities and states
# ---------------------------------------------------------------------------


def haar_fidelity_bin_probabilities(n_qubits: int, n_bins: int) -> torch.Tensor:
    """
    Exact probability of each of ``n_bins`` equal bins of ``[0, 1]`` under
    ``P_Haar(F) = (N − 1)(1 − F)^(N − 2)``, ``N = 2^n_qubits``.  Sums to 1.

    From about 8 qubits the high-fidelity bins are below the smallest float64
    and come out as 0; the KL divergence uses their logarithms instead.
    """
    return _haar_log_bin_probabilities(n_qubits, n_bins).exp()


def _haar_log_bin_probabilities(n_qubits: int, n_bins: int) -> torch.Tensor:
    """
    ``ln q_i`` for the bins of :func:`haar_fidelity_bin_probabilities`, finite
    at any width: ``q_i = S(a) − S(b)`` with ``S(F) = (1 − F)^(N−1)``, so
    ``ln q_i = ln S(a) + ln(1 − S(b)/S(a))``.
    """
    if n_qubits < 1 or n_bins < 1:
        raise ValueError(f"n_qubits and n_bins must be ≥ 1; got {n_qubits}, {n_bins}.")
    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    log_survival = (2**n_qubits - 1) * torch.log1p(-edges)  # ln P(F ≥ edge); -inf at 1
    return log_survival[:-1] + torch.log1p(-torch.exp(log_survival[1:] - log_survival[:-1]))


def expressibility_from_fidelities(
    fidelities: torch.Tensor, n_qubits: int, n_bins: int = 75
) -> float:
    """
    ``D_KL`` of the histogram of ``fidelities`` from the Haar fidelity
    distribution on ``n_qubits`` qubits (see the module docstring).

    Raises
    ------
    ValueError
        If ``fidelities`` is empty or has values outside ``[0, 1]`` beyond
        round-off.
    """
    f = torch.as_tensor(fidelities, dtype=torch.float64).flatten()
    if f.numel() == 0:
        raise ValueError("fidelities is empty.")
    if f.min() < -1e-9 or f.max() > 1 + 1e-9:
        raise ValueError(f"fidelities must lie in [0, 1]; got [{f.min():.3g}, {f.max():.3g}].")
    # F = 1 belongs to the last bin, not to a bin past the end.
    index = (f.clamp(0.0, 1.0) * n_bins).long().clamp(max=n_bins - 1)
    p = torch.bincount(index, minlength=n_bins).to(torch.float64) / f.numel()
    log_q = _haar_log_bin_probabilities(n_qubits, n_bins)
    occupied = p > 0
    return float((p[occupied] * (torch.log(p[occupied]) - log_q[occupied])).sum())


def meyer_wallach(states: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """
    ``Q`` of each state in ``states`` (shape ``(batch, 2^n_qubits)``), as
    ``2 · (1 − mean_k Tr ρ_k²)``.  States are normalised first.
    """
    psi = torch.as_tensor(states).to(torch.complex128)
    if psi.ndim != 2 or psi.shape[1] != 2**n_qubits:
        raise ValueError(f"states must have shape (batch, {2**n_qubits}); got {tuple(psi.shape)}.")
    psi = psi / psi.norm(dim=1, keepdim=True)
    tensor = psi.reshape(psi.shape[0], *([2] * n_qubits))
    purities = []
    for k in range(n_qubits):
        # Qubit k first, everything else flattened: ρ_k = M M†.
        m = tensor.movedim(k + 1, 1).reshape(psi.shape[0], 2, -1)
        rho = m @ m.conj().transpose(1, 2)
        purities.append((rho.abs() ** 2).sum(dim=(1, 2)))
    return 2.0 * (1.0 - torch.stack(purities).mean(dim=0))


# ---------------------------------------------------------------------------
# Sampling states from a layer
# ---------------------------------------------------------------------------


def _sample_states(
    layer: nn.Module, n_states: int, inputs: Inputs, generator: torch.Generator, caller: str
) -> tuple[torch.Tensor, int]:
    """``n_states`` states of ``layer``, each with its own weight (and input) draw."""
    encoder = _resolve_layer(layer, caller)
    qlayer, n_qubits = encoder.qlayer, encoder.n_qubits
    shapes = {name: tuple(p.shape) for name, p in qlayer.qnode_weights.items()}
    width = encoder.n_features
    if isinstance(inputs, str):
        if inputs != "random":
            raise ValueError(f"inputs must be 'random' or a tensor; got {inputs!r}.")
        raw = (
            torch.rand(n_states, width, generator=generator, dtype=torch.float64) * 2 - 1
        ) * math.pi
    else:
        x = torch.as_tensor(inputs, dtype=torch.float64)
        if x.ndim != 1:
            raise ValueError(
                f"a fixed input must be one vector of {width} features; got {tuple(x.shape)}."
            )
        raw = x.expand(n_states, -1)
    prepared = _prepare(raw, encoder, "inputs")

    device = qml.device("default.qubit", wires=n_qubits)
    states: list[torch.Tensor] = []
    for start in range(0, n_states, _CHUNK):
        tapes = []
        for i in range(start, min(start + _CHUNK, n_states)):
            weights = {
                name: torch.rand(shape, generator=generator, dtype=torch.float64) * 2 * math.pi
                for name, shape in shapes.items()
            }
            # level=0: the circuit as written, as in hqnn_forge.kernels.
            tape = qml.workflow.construct_tape(qlayer.qnode, level=0)(prepared[i], **weights)
            tapes.append(tape.copy(measurements=[qml.state()]))
        states.extend(_execute(tapes, device))
    return torch.stack(states).reshape(n_states, 2**n_qubits), n_qubits


def _execute(tapes: list[qml.tape.QuantumScript], device: Any) -> list[torch.Tensor]:
    """The state of every tape: from one broadcast run if they merge, else one by one."""
    merged = _broadcast(tapes)
    if merged is not None:
        # An operation can accept a batch when it is built and still fail to
        # run one (GlobalPhase with torch parameters on PennyLane 0.45).  The
        # tapes then run alone, where a genuine error still surfaces.  So do
        # they when the result is not one state per tape: the reshape raises.
        with contextlib.suppress(Exception):
            (result,) = qml.execute([merged], device)
            rows = torch.as_tensor(result).reshape(len(tapes), 2 ** len(device.wires))
            return [r.to(torch.complex128) for r in rows]
    return [torch.as_tensor(r).to(torch.complex128) for r in qml.execute(tapes, device)]


def _alike(a: object, b: object) -> bool:
    """
    Whether two operations differ in nothing but the values of their parameters.

    Name, wires, parameter shapes and every hyperparameter must agree: a Pauli
    word, a template's ``rotation`` or ``ranges`` and the like are not stacked,
    so the merged tape would silently take them from the first tape.  An
    operation held as a hyperparameter (the base of a controlled operation) is
    compared the same way; anything that cannot be compared counts as differing.
    """
    if a is b:
        return True
    if isinstance(a, Operator):
        return (
            isinstance(b, Operator)
            and a.name == b.name
            and a.wires == b.wires
            and [qml.math.shape(d) for d in a.data] == [qml.math.shape(d) for d in b.data]
            and _alike(a.hyperparameters, b.hyperparameters)
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_alike(v, b[k]) for k, v in a.items())
        )
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(_alike(x, y) for x, y in zip(a, b, strict=True))
        )
    try:
        return bool(qml.math.shape(a) == qml.math.shape(b) and qml.math.all(a == b))
    except Exception:  # noqa: BLE001
        return False


def _as_batch(op: Operator, batch: int) -> list[Operator] | None:
    """
    ``op``, holding stacked parameters, as operations that each run ``batch`` circuits.

    That is ``op`` itself when it reports the batch.  A template such as
    ``StronglyEntanglingLayers`` reports none for a stacked weight tensor but
    decomposes into gates that do, and is replaced by them.  None when neither
    holds: PennyLane would then read the stacked parameter as one circuit's
    (``BasisEmbedding`` of a ``(batch, n)`` tensor) and return wrong states
    without raising.
    """
    if op.batch_size == batch:
        return [op]
    if not op.has_decomposition:
        return None
    ops: list[Operator] = []
    parametrised = False
    for part in op.decomposition():
        if not part.data:
            ops.append(part)
            continue
        expanded = _as_batch(part, batch)
        if expanded is None:
            return None
        ops.extend(expanded)
        parametrised = True
    # Parameters that vanish in the decomposition were not applied per circuit.
    return ops if parametrised else None


def _broadcast(tapes: list[qml.tape.QuantumScript]) -> qml.tape.QuantumScript | None:
    """
    One tape running every tape in ``tapes`` as a parameter batch, or None.

    None when there are fewer than two tapes, when they differ in their
    operations, wires, parameter shapes or hyperparameters, when one is
    already broadcast, or when an operation does not take a batch of
    parameters: the caller then runs them one by one.
    """
    if len(tapes) < 2 or any(t.batch_size is not None for t in tapes):
        return None
    first = tapes[0]
    for tape in tapes[1:]:
        if len(tape.operations) != len(first.operations) or not all(
            _alike(a, b) for a, b in zip(first.operations, tape.operations, strict=True)
        ):
            return None
    ops: list[Operator] = []
    for k, op in enumerate(first.operations):
        if not op.data:
            ops.append(op)
            continue
        stacked = [
            qml.math.stack([t.operations[k].data[j] for t in tapes]) for j in range(len(op.data))
        ]
        try:
            batched = _as_batch(bind_new_parameters(op, stacked), len(tapes))
        except Exception:  # noqa: BLE001
            return None
        if batched is None:
            return None
        ops.extend(batched)
    merged = qml.tape.QuantumScript(ops, [qml.state()])
    return merged if merged.batch_size == len(tapes) else None


def _generator(generator: torch.Generator | None) -> torch.Generator:
    return generator if generator is not None else torch.Generator().manual_seed(0)


# ---------------------------------------------------------------------------
# Public diagnostics
# ---------------------------------------------------------------------------


def expressibility(
    layer: nn.Module,
    n_pairs: int = 5000,
    n_bins: int = 75,
    *,
    inputs: Inputs = "random",
    generator: torch.Generator | None = None,
) -> ExpressibilityResult:
    """
    Expressibility of ``layer``: KL divergence of its fidelity distribution
    from the Haar one (see the module docstring).

    Parameters
    ----------
    layer:
        An encoding layer.  A hybrid classifier is refused; pass its
        ``quantum_layer``.
    n_pairs:
        Pairs of independently drawn states.  Default: 5000 (Sim et al.).
    n_bins:
        Histogram bins on ``[0, 1]``.  Default: 75 (Sim et al.).
    inputs:
        ``"random"`` (default): each state gets its own input from
        ``(−π, π)``.  A 1-D tensor: that input for every state.
    generator:
        Source of every draw.  Default: a generator seeded with 0, so repeated
        calls agree.

    Returns
    -------
    ExpressibilityResult
    """
    if n_pairs < 1 or n_bins < 1:
        raise ValueError(f"n_pairs and n_bins must be ≥ 1; got {n_pairs}, {n_bins}.")
    gen = _generator(generator)
    states, n_qubits = _sample_states(layer, 2 * n_pairs, inputs, gen, "expressibility")
    first, second = states[:n_pairs], states[n_pairs:]
    fidelities = ((first.conj() * second).sum(dim=1).abs() ** 2).clamp(0.0, 1.0)
    return ExpressibilityResult(
        layer_type=type(layer).__name__,
        n_qubits=n_qubits,
        n_pairs=n_pairs,
        n_bins=n_bins,
        expressibility=expressibility_from_fidelities(fidelities, n_qubits, n_bins),
        maximum=(2**n_qubits - 1) * math.log(n_bins),
        fidelities=fidelities,
    )


def entangling_capability(
    layer: nn.Module,
    n_samples: int = 1000,
    *,
    inputs: Inputs = "random",
    generator: torch.Generator | None = None,
) -> EntanglingCapabilityResult:
    """
    Entangling capability of ``layer``: mean Meyer–Wallach ``Q`` over sampled
    states (see the module docstring).  Arguments as for :func:`expressibility`.

    Returns
    -------
    EntanglingCapabilityResult
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be ≥ 2 for a standard error; got {n_samples}.")
    states, n_qubits = _sample_states(
        layer, n_samples, inputs, _generator(generator), "entangling_capability"
    )
    values = meyer_wallach(states, n_qubits)
    n = 2**n_qubits
    return EntanglingCapabilityResult(
        layer_type=type(layer).__name__,
        n_qubits=n_qubits,
        n_samples=n_samples,
        entangling_capability=float(values.mean()),
        standard_error=float(values.std() / math.sqrt(n_samples)),
        haar_reference=(n - 2) / (n + 1),
        values=values,
    )
