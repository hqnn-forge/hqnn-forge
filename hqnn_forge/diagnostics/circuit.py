"""
hqnn_forge.diagnostics.circuit
==============================
Depth, gate counts and parameter counts for the library's quantum circuits.

The encoding layers build their QNodes internally, so answering "how many
CNOTs does this configuration use?" otherwise means reading the source.
``circuit_summary`` asks PennyLane instead: it constructs the tape the layer
would execute and counts the resources on it.

Counting convention
-------------------
Resources are counted on the *logical* circuit, i.e. after decomposing
templates (``AngleEmbedding`` → one ``RX`` per qubit) but **before** any
device-specific decomposition.  ``lightning.qubit``'s adjoint path, for
example, rewrites every ``Rot`` as ``RZ·RY·RZ``, which would make the same
model report different counts depending on the simulator it happens to run
on.  The gate set counted against is ``LOGICAL_GATE_SET``; everything is
decomposed until only those gates remain, except that a ``MultiRZ`` on more
than two wires is decomposed further, into one- and two-qubit gates, even
though its name is in the set (a gate with no decomposition at all is left
as it is and counts once).  Two-qubit gates are
counted separately because they are what NISQ feasibility is usually judged
by, so ``n_two_qubit_gates`` is the two-qubit cost of the circuit: a k-wire
``MultiRZ`` contributes the 2(k-1) CNOTs of its ladder, not 1, while a
two-wire ``MultiRZ`` (a ZZ rotation, native on some hardware) stays one gate.

With PennyLane's graph-based decomposition enabled
(``qml.decomposition.enable_graph()``) the library's own layers count the
same, since every gate they emit is in the set.  A gate outside it
(``CRX``, ``Toffoli``, ...) is decomposed by whichever rule the graph finds
cheapest rather than by ``op.decomposition()``, so its counts can differ
between the two modes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, NamedTuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from hqnn_forge._encoding_contract import CircuitLayer
from hqnn_forge._resolve import resolve_encoding_layer

#: Gate names a circuit is decomposed to before its resources are counted.
#: Every gate the library's circuits emit is in here, so the count is of the
#: circuit as written; a template such as ``AngleEmbedding`` is expanded.
#: ``circuit_summary`` does not decompose to exactly this set: it keeps
#: ``MultiRZ`` only on at most two wires and decomposes a wider one to its
#: CNOT ladder (see _decompose_logical), so ``decompose(gate_set=...)`` with
#: this set alone counts a wide ``MultiRZ`` once and does not reproduce it.
LOGICAL_GATE_SET: frozenset[str] = frozenset(
    {"Hadamard", "RX", "RY", "RZ", "Rot", "PhaseShift", "CNOT", "CZ", "MultiRZ"}
)


@dataclass(frozen=True)
class CircuitSummary:
    """
    Resource summary of one quantum encoding layer's circuit.

    Attributes
    ----------
    layer_type:
        Class name of the summarised layer, e.g. ``"QuantumEncodingLayer"``.
    n_qubits:
        Wires the circuit acts on.
    n_trainable_params:
        Trainable entries in the layer's weight tensors (``requires_grad``).
    depth:
        Longest path of gates through the logical circuit.
    n_gates:
        Total gate count after decomposition to ``LOGICAL_GATE_SET`` (with
        ``MultiRZ`` kept only on at most two wires).
    n_two_qubit_gates:
        Two-qubit gates (CNOT, CZ, two-wire MultiRZ) after gates on more
        than two wires have been decomposed into them, i.e. the circuit's
        two-qubit cost.  A wider gate with no decomposition counts once.
    gate_counts:
        Count per gate name, sorted by name.  This field is a mapping, so the
        dataclass is frozen for immutability but is **not** hashable.
    device_name:
        PennyLane device the layer's QNode is bound to, after any fallback.
    diff_method:
        Differentiation method the QNode is configured with.
    n_inert_params:
        Trainable parameters that cannot affect any measurement for **any**
        input or weight values, found structurally by
        :func:`count_inert_parameters`.  The typical case is the ``ω`` of a
        ``Rot`` whose wire carries only ``Z``-type content downstream, such as
        a last-layer ``Rot`` followed at most by CNOTs and a ``⟨Z⟩`` readout:
        ``Rot = RZ(ω)·RY(θ)·RZ(φ)`` and the final ``RZ`` commutes with ``Z``.
        The count is of **weight entries** (:func:`count_inert_weights`): an
        entry is inert when every gate parameter it feeds is inert, so it is
        on the same footing as ``n_trainable_params``.
    """

    layer_type: str
    n_qubits: int
    n_trainable_params: int
    depth: int
    n_gates: int
    n_two_qubit_gates: int
    gate_counts: Mapping[str, int] = field(default_factory=dict)
    device_name: str = ""
    diff_method: str = ""
    n_inert_params: int = 0

    @property
    def n_effective_params(self) -> int:
        """
        ``n_trainable_params - n_inert_params``: the trainable weight entries
        that can move the output.  Both count weight entries, so the
        difference holds for a layer that reuses an entry in several gates or
        computes an angle from several entries, and it is never negative.
        """
        return self.n_trainable_params - self.n_inert_params

    def to_dict(self) -> dict[str, Any]:
        """
        Plain-dict form, for logging frameworks and JSON.

        Besides the fields it holds the derived ``n_effective_params``; drop
        that key to rebuild a ``CircuitSummary`` from the dict.
        """
        # Not dataclasses.asdict: it deep-copies, which raises for a
        # gate_counts that is a Mapping but not a dict (e.g. a mappingproxy).
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["gate_counts"] = dict(self.gate_counts)
        d["n_effective_params"] = self.n_effective_params
        return d

    def __str__(self) -> str:
        rows = (
            ("qubits", self.n_qubits),
            ("trainable params", self.n_trainable_params),
            ("inert params", self.n_inert_params),
            ("effective params", self.n_effective_params),
            ("depth", self.depth),
            ("gates", self.n_gates),
            ("two-qubit gates", self.n_two_qubit_gates),
        )
        # Gate names are indented two columns further, so they get two less padding
        width = max(
            max(len(label) for label, _ in rows),
            max((len(name) + 2 for name in self.gate_counts), default=0),
        )
        lines = [
            f"Circuit summary: {self.layer_type} on {self.device_name} ({self.diff_method})",
        ]
        lines += [f"  {label:<{width}} : {value}" for label, value in rows]
        lines += [f"    {name:<{width - 2}} : {count}" for name, count in self.gate_counts.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resolving what to inspect
# ---------------------------------------------------------------------------


def input_width(layer: CircuitLayer) -> int:
    """
    Number of features ``layer`` takes per sample: ``n_features`` where the
    layer has one (the amplitude encoder, up to ``2**n_qubits``), else one
    per qubit.
    """
    width = getattr(layer, "n_features", None)
    if isinstance(width, int):
        return width
    n_qubits = getattr(layer, "n_qubits", None)
    if not isinstance(n_qubits, int):
        raise TypeError(f"{type(layer).__name__} has neither n_features nor n_qubits.")
    return n_qubits


def _prepare(layer: CircuitLayer, x: torch.Tensor) -> torch.Tensor:
    """
    ``layer.prepare_inputs(x)``, the classical step ``forward`` runs before the QNode.

    A layer without one gets ``x`` unchanged: the diagnostics accept any
    module with a ``qlayer`` and an integer ``n_qubits``, and such a layer
    feeds its QNode its raw inputs.
    """
    prepare = getattr(layer, "prepare_inputs", None)
    if not callable(prepare):
        return x
    out = prepare(x)
    assert isinstance(out, torch.Tensor)
    return out


def sample_input(layer: CircuitLayer) -> torch.Tensor:
    """
    One raw input for drawing and counting the layer's circuit.

    Zeros where the layer's ``prepare_inputs`` accepts them: the angle-type
    embeddings run the same gates for every input, so only the printed angles
    depend on it.

    Amplitude embedding has no state for the zero vector, and the gates of
    its Möttönen state preparation *do* depend on the input: PennyLane leaves
    out a block of rotations when all its angles are zero.  A uniform or
    all-positive vector has no phases and so shows no ``RZ`` at all, and
    other sign patterns drop some ``RZ`` blocks.  It gets
    ``[-1, 2, 3, …, n_features]`` instead: distinct magnitudes and one
    negative entry make every block non-zero, so the counts are the most any
    input needs: ``2**n - 1`` ``RY`` and ``2**n - 1`` ``RZ`` rotations when
    ``n_features == 2**n``.  With fewer features the zero padding leaves out
    the ``RY`` rotations that act only on padded amplitudes (for ``n = 3``,
    6 with 4 features, 4 with 2), since no input can make those non-zero.
    """
    width = input_width(layer)
    zeros = torch.zeros(1, width, dtype=torch.float64)
    try:
        _prepare(layer, zeros)
    except ValueError:
        full = torch.arange(1, width + 1, dtype=torch.float64)
        full[0] = -1.0
        return full
    return zeros[0]


def _written_tape(
    layer: CircuitLayer, inputs: torch.Tensor | None = None
) -> qml.tape.QuantumScript:
    """
    The tape the layer executes for one sample, as written.

    ``inputs`` is one raw sample of the layer's input width (default:
    :func:`sample_input`); it goes through the layer's ``prepare_inputs``
    first, as ``forward`` does, so the raw QNode receives what it would in
    training -- for the amplitude encoder, a padded and normalised state.

    The weight tensors keep their ``requires_grad`` flag, so the gate
    parameters that come from trainable weights can be told apart from the
    (non-trainable) inputs by :func:`count_inert_parameters`.
    """
    qlayer = getattr(layer, "qlayer", None)
    if not isinstance(qlayer, qml.qnn.TorchLayer):
        raise TypeError(f"{type(layer).__name__} has no qlayer TorchLayer.")
    raw = sample_input(layer) if inputs is None else torch.as_tensor(inputs, dtype=torch.float64)
    if raw.ndim != 1 or raw.shape[0] != input_width(layer):
        raise ValueError(
            f"inputs must be one sample of shape ({input_width(layer)},); got {tuple(raw.shape)}."
        )
    inputs = _prepare(layer, raw[None, :])[0]
    weights = dict(qlayer.qnode_weights.items())
    # level="top": the circuit as written, before the QNode's own transforms
    # (batch expansion) and before the device rewrites gates it cannot run.
    return qml.workflow.construct_tape(qlayer.qnode, level="top")(inputs, **weights)


def _logical_tape(
    layer: CircuitLayer, inputs: torch.Tensor | None = None
) -> qml.tape.QuantumScript:
    """The tape the layer executes for one sample, decomposed by _decompose_logical."""
    return _decompose_logical(_written_tape(layer, inputs))


def _decompose_logical(tape: qml.tape.QuantumScript) -> qml.tape.QuantumScript:
    """
    ``tape`` decomposed to ``LOGICAL_GATE_SET``, with ``MultiRZ`` kept only on
    at most two wires, so a wider one counts at its two-qubit cost.

    The wire rule is a ``stopping_condition`` next to a ``gate_set`` without
    ``MultiRZ``, rather than a stopping condition alone: PennyLane keeps an op
    that satisfies either, and graph-based decomposition
    (``qml.decomposition.enable_graph()``) refuses a call without ``gate_set``.

    Graph-based decomposition also emits ``GlobalPhase`` (from state
    preparation or ``QubitUnitary``, say) where ``op.decomposition()`` does
    not.  It is kept, so it is not decomposed further, then dropped: a global
    phase is not a gate and would otherwise add to ``depth`` and, on two
    wires, to ``n_two_qubit_gates``.
    """
    (decomposed,), _ = qml.transforms.decompose(
        tape,
        gate_set=(LOGICAL_GATE_SET - {"MultiRZ"}) | {"GlobalPhase"},
        stopping_condition=_is_two_wire_multirz,
    )
    return decomposed.copy(
        operations=[op for op in decomposed.operations if op.name != "GlobalPhase"]
    )


def _is_two_wire_multirz(op: qml.operation.Operator) -> bool:
    return op.name == "MultiRZ" and len(op.wires) <= 2


class _TapeResources(NamedTuple):
    """Depth and gate counts of a tape's operations (measurements excluded)."""

    depth: int
    n_gates: int
    n_two_qubit_gates: int
    gate_counts: dict[str, int]


def _tape_resources(tape: qml.tape.QuantumScript) -> _TapeResources:
    """
    Depth and gate counts of a tape from _decompose_logical, computed from
    its operations.

    Counted here rather than read from ``tape.specs["resources"]``, whose
    layout PennyLane changes between releases: 0.46 drops ``num_gates``,
    ``gate_types`` and ``gate_sizes`` (#344).  The depth is the usual one, the
    number of layers when every gate starts as soon as all its wires are free.
    A gate on two or more wires counts once towards ``n_two_qubit_gates``:
    after _decompose_logical only a gate with no decomposition can still act
    on more than two, and it counts once rather than dropping out of the count.

    On a tape from _decompose_logical these are the numbers ``specs`` reports
    in both 0.45 and 0.46.  On other tapes they can differ: ``specs`` prefixes
    a gate with several control wires with their number (``2C(RX)``, here
    ``C(RX)``), expands a ``ResourcesOperation`` into its declared resources,
    and gives an operation without wires (``Barrier()``, ``Snapshot``) or one
    conditioned on a mid-circuit measurement a different layer.
    """
    free_at: dict[object, int] = {}
    depth = 0
    for op in tape.operations:
        layer = 1 + max((free_at.get(w, 0) for w in op.wires), default=0)
        for w in op.wires:
            free_at[w] = layer
        depth = max(depth, layer)
    names = Counter(op.name for op in tape.operations)
    return _TapeResources(
        depth=depth,
        n_gates=len(tape.operations),
        n_two_qubit_gates=sum(1 for op in tape.operations if len(op.wires) >= 2),
        gate_counts=dict(sorted(names.items())),
    )


# ---------------------------------------------------------------------------
# Inert parameters: what can never reach a measurement
# ---------------------------------------------------------------------------

# Backward-propagated support of the measured observables on each wire:
# nothing measured touches the wire / only Z-type (diagonal) content / may
# carry X or Y.  Only the last one anticommutes with a Z rotation.
_NONE, _Z, _XY = 0, 1, 2

# Gates diagonal in the computational basis: they commute with every Z-string.
_DIAGONAL = frozenset(
    {"RZ", "PhaseShift", "MultiRZ", "IsingZZ", "CZ", "PauliZ", "S", "T", "Identity"}
)


def _has_scalar_parameters(op: qml.operation.Operator) -> bool:
    return all(qml.math.ndim(value) == 0 for value in op.data)


def _z_only(pauli_rep: Any) -> bool:
    """Every Pauli word of a ``PauliSentence`` is a product of ``Z`` (or identity)."""
    return all(set(word.values()) <= {"Z"} for word in pauli_rep)


#: Widest operator whose dense matrix the diagonality fallback builds: a
#: ``2**n``-square matrix of a parameterless ``QFT`` on 16 wires alone would
#: take 64 GiB.  A wider operator counts as not diagonal, which keeps the
#: inert count a lower bound.
_MATRIX_CHECK_MAX_WIRES = 6


def _off_diagonal_zero(op: qml.operation.Operator) -> bool:
    """
    Every entry off the diagonal of the fixed (parameterless) matrix of ``op``
    is zero; ``False`` when ``op`` has no matrix or is too wide to build one.
    """
    if len(op.wires) > _MATRIX_CHECK_MAX_WIRES:
        return False
    try:
        matrix = qml.matrix(op, wire_order=op.wires)
    except (qml.exceptions.MatrixUndefinedError, NotImplementedError):
        return False
    arr = np.asarray(qml.math.to_numpy(matrix))
    return bool(np.all(np.abs(arr - np.diag(np.diag(arr))) < 1e-12))


def _is_diagonal_gate(op: qml.operation.Operator) -> bool:
    """
    Whether ``op`` is diagonal in the computational basis for every value of
    its parameters, decided structurally rather than by name:

    * one of the known diagonal gates (``_DIAGONAL``);
    * a symbolic wrapper (``Adjoint``, ``Controlled``, ``Conditional``,
      ``Pow``, ``Exp``) of a diagonal gate, since each keeps diagonality;
    * a gate whose generator has only ``Z``/identity Pauli words
      (``CRZ``, ``ControlledPhaseShift``, ...), since then ``exp(-iθG)`` is
      diagonal for every ``θ``;
    * a gate without parameters whose matrix is diagonal (``S``, ``CCZ``),
      checked only up to ``_MATRIX_CHECK_MAX_WIRES`` wires.

    A parametrised gate is never judged by its matrix: at a particular value
    it can be diagonal by coincidence (``RX(0)`` is the identity), which says
    nothing about the other values.  Anything undecided counts as mixing, so
    the inert count stays a lower bound.
    """
    if op.name in _DIAGONAL:
        return True
    base = getattr(op, "base", None)
    if isinstance(op, qml.ops.op_math.SymbolicOp) and isinstance(base, qml.operation.Operator):
        return _is_diagonal_gate(base)
    if op.num_params > 0:
        try:
            rep = op.generator().pauli_rep
        except (qml.exceptions.GeneratorUndefinedError, NotImplementedError, AttributeError):
            return False
        return rep is not None and _z_only(rep)
    return _off_diagonal_zero(op)


#: Measurements in the computational basis without an observable: they read
#: only the diagonal of the state, so they count as Z content on their wires.
#: ``state``/``density_matrix`` and the entropy-type measurements read
#: coherences and stay X/Y content.
_BASIS_MEASUREMENTS = (
    qml.measurements.ProbabilityMP,
    qml.measurements.SampleMP,
    qml.measurements.CountsMP,
)


def _is_diagonal_measurement(measurement: qml.measurements.MeasurementProcess) -> bool:
    obs = getattr(measurement, "obs", None)
    if obs is None:
        return isinstance(measurement, _BASIS_MEASUREMENTS) and not isinstance(
            measurement, qml.measurements.StateMP
        )
    rep = obs.pauli_rep
    if rep is not None:
        return _z_only(rep)
    return _off_diagonal_zero(obs)


def _inert_slots(tape: qml.tape.QuantumScript) -> list[tuple[Any, bool]]:
    """
    ``(value, inert)`` for every trainable gate-parameter slot of ``tape``,
    after decomposing it to scalar-parameter gates; see
    :func:`count_inert_parameters` for the rules and the errors raised.
    """
    if tape.batch_size is not None:
        raise ValueError(
            "count_inert_parameters needs an unbroadcast tape: with parameter "
            f"broadcasting (batch size {tape.batch_size}) one gate parameter holds "
            "several values, so a count of gate parameters has no clear meaning."
        )
    # Every gate in the set has scalar parameters, so passing it does not
    # change what stops; graph-based decomposition requires a gate_set, and
    # emits GlobalPhase (see _decompose_logical).
    (tape,), _ = qml.transforms.decompose(
        tape,
        gate_set=LOGICAL_GATE_SET | {"GlobalPhase"},
        stopping_condition=_has_scalar_parameters,
    )
    unexpanded = sorted({op.name for op in tape.operations if not _has_scalar_parameters(op)})
    if unexpanded:
        raise ValueError(
            "count_inert_parameters could not decompose these gates to gates with "
            f"scalar parameters: {', '.join(unexpanded)}."
        )
    support: dict[Any, int] = dict.fromkeys(tape.wires, _NONE)
    for measurement in tape.measurements:
        wires = list(measurement.wires) if len(measurement.wires) else list(tape.wires)
        diagonal = _is_diagonal_measurement(measurement)
        for wire in wires:
            support[wire] = max(support[wire], _Z if diagonal else _XY)

    slots: list[tuple[Any, bool]] = []
    for op in reversed(tape.operations):
        wires = list(op.wires)
        if isinstance(op, qml.ops.MidMeasure):
            for w in wires:
                support[w] = _XY
            continue
        if op.name == "GlobalPhase":
            # Commutes with everything, so no wire's support changes.  Not
            # counted: under state() the phase does reach the measurement.
            # Still a live slot, so count_inert_weights does not take an entry
            # that feeds only a GlobalPhase for one that feeds nothing.
            slots.extend((value, False) for value in op.data if qml.math.requires_grad(value))
            continue
        trainable = [value for value in op.data if qml.math.requires_grad(value)]
        if all(support[w] == _NONE for w in wires):
            # nothing measured downstream ever sees this gate
            slots.extend((value, True) for value in trainable)
            continue
        if _is_diagonal_gate(op):
            commutes = all(support[w] != _XY for w in wires)
            # inert when it commutes with every observable it meets
            slots.extend((value, commutes) for value in trainable)
            if not commutes and len(wires) > 1:
                # X on one wire of a diagonal multi-qubit gate spreads Z to the others
                for w in wires:
                    support[w] = max(support[w], _Z)
        elif op.name == "CNOT":
            slots.extend((value, False) for value in trainable)
            control, target = wires
            new_control, new_target = support[control], support[target]
            if support[target] != _NONE:
                new_control = max(new_control, _Z)  # Z_t → Z_c Z_t, Y_t → Z_c Y_t
            if support[control] == _XY:
                new_target = _XY  # X_c → X_c X_t
            support[control], support[target] = new_control, new_target
        elif op.name == "Rot":
            (wire,) = wires
            # Rot = RZ(ω)·RY(θ)·RZ(φ), ω applied last: it commutes with a
            # diagonal observable, so ω is inert whenever the wire carries no
            # X/Y content.  RY(θ) then mixes Z into X/Y for the earlier gates.
            omega_inert = support[wire] != _XY
            for index, value in enumerate(op.data):
                if qml.math.requires_grad(value):
                    slots.append((value, index == 2 and omega_inert))
            support[wire] = _XY
        else:
            # Any other gate (RX, RY, Hadamard, ...): content on its wires may
            # become X/Y, and a multi-qubit gate may spread it across its wires.
            slots.extend((value, False) for value in trainable)
            for w in wires:
                support[w] = _XY
    return slots


def count_inert_parameters(tape: qml.tape.QuantumScript) -> int:
    """
    Number of trainable gate parameters that cannot affect any measurement of
    ``tape``, for any input or weight values.

    The measured observables are propagated backwards through the circuit in
    the Heisenberg picture, keeping per wire only whether the observable's
    content there is nothing, diagonal (``Z``-type) or possibly ``X``/``Y``:
    CNOT moves ``Z`` content from target to control and ``X``/``Y`` content
    from control to target; a diagonal gate leaves ``Z`` content alone, and a
    diagonal multi-qubit gate (``CZ``, ``MultiRZ``, ``IsingZZ``) spreads
    ``Z`` content to all its wires once one of them carries ``X``/``Y``; any
    other gate turns the content on its wires into ``X``/``Y``.  A trainable
    parameter is inert when its gate acts on wires with no content at all,
    or when the gate is diagonal (``RZ``, ``PhaseShift``, ``MultiRZ``,
    ``CZ``, ``IsingZZ``, ...) or the final ``RZ(ω)`` of a ``Rot`` and every
    wire it touches carries only diagonal content: the gate then commutes
    with everything measured.

    The propagation over-approximates the ``X``/``Y`` content, so the count
    is a lower bound on the parameters that are dead for structural reasons:
    everything it counts has an exactly zero gradient for every input, and
    parameters that are dead only for particular inputs or weights (say, a
    rotation of ``|0⟩`` about ``Z``) are not counted.

    Templates and other gates with tensor-valued parameters (such as
    ``StronglyEntanglingLayers``) are first decomposed until every gate
    parameter is a scalar, so each counted parameter is one gate-parameter
    slot.  A parameter-broadcast tape is rejected, since one slot there
    stands for a whole batch of values.

    A measurement counts as diagonal when every Pauli word of its observable
    is a product of ``Z`` (``Z(0)``, ``2 * Z(0)``, ``Z(0) + Z(1)``, nested
    products, ``Z(0) @ I(1)``; for an observable without a Pauli
    representation, when its matrix is diagonal and it acts on at most
    six wires), and when it is a
    computational-basis measurement without an observable (``probs``,
    ``sample``, ``counts``).  ``state``, ``density_matrix`` and any other
    measurement mark their wires as ``X``/``Y`` content, and a measurement
    without wires (``state``, ``probs`` over all wires) marks every wire.
    A mid-circuit measurement counts as a measurement of arbitrary content on
    its wire, since its outcome may drive a conditional gate or be returned.
    Whether a gate is diagonal is decided structurally
    (:func:`_is_diagonal_gate`: its generator, a symbolic wrapper of a
    diagonal gate, or the matrix of a gate without parameters on at most
    six wires, so a wide ``QFT`` never builds its dense matrix), so
    ``CRZ``, ``ControlledPhaseShift``, ``Adjoint(RZ)`` or a conditional
    ``RZ`` count as diagonal.  Gates that are neither diagonal nor ``CNOT``
    nor ``Rot`` are treated as fully mixing, which keeps the count a lower
    bound for any gate.

    Raises
    ------
    ValueError
        If ``tape`` uses parameter broadcasting, or a gate with a
        tensor-valued parameter cannot be decomposed to scalar-parameter gates.
    """
    return sum(inert for _, inert in _inert_slots(tape))


def count_inert_weights(tape: qml.tape.QuantumScript, weights: Sequence[torch.Tensor]) -> int:
    """
    Number of entries of ``weights`` that cannot affect any measurement of
    ``tape``: an entry is inert when every gate parameter it feeds is inert
    (:func:`count_inert_parameters`), or when it feeds none.

    This counts weight entries, the unit of ``n_trainable_params``, rather
    than gate-parameter slots.  The two differ for a layer that reuses one
    entry in several gates (an inert ``RZ(w)`` and a live ``RX(w)``: the
    entry is live) or computes one angle from several entries (an inert
    ``RZ(w1·w2)``: both entries are inert).

    Which entries a slot depends on is read off autograd: the entries with a
    nonzero gradient of the slot's value.  Build ``tape`` from generic inputs
    (not zeros), so that no dependence vanishes by accident, e.g. through an
    input-times-weight angle at input 0.  The same holds for the weights
    themselves: a live angle at a stationary point of its function of the
    weights (``RX(w1·w2)`` at ``w1 = w2 = 0``) reads as depending on neither
    entry, so both are counted inert.  Only weights with ``requires_grad``
    are counted.

    Parameters
    ----------
    tape:
        As for :func:`count_inert_parameters`, with gate parameters computed
        from ``weights`` by torch operations.
    weights:
        The weight tensors the gate parameters are computed from.
    """
    leaves = [w for w in weights if w.requires_grad]
    live = [torch.zeros(w.shape, dtype=torch.bool) for w in leaves]
    for value, inert in _inert_slots(tape):
        if inert or not isinstance(value, torch.Tensor) or not value.requires_grad:
            continue
        grads = torch.autograd.grad(value, leaves, allow_unused=True, retain_graph=True)
        for k, grad in enumerate(grads):
            if grad is not None:
                live[k] |= grad.detach().cpu() != 0
    return sum(int((~mask).sum()) for mask in live)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def circuit_summary(target: nn.Module) -> CircuitSummary:
    """
    Summarise the circuit of an encoding layer or of a hybrid classifier.

    Parameters
    ----------
    target:
        A ``QuantumEncodingLayer`` / ``IQPEncodingLayer``, or a classifier with
        a ``quantum_layer`` attribute (``HybridBinaryClassifier``,
        ``ParallelHybridClassifier``).

    Returns
    -------
    CircuitSummary
        Depth, gate counts, parameter count and inert-parameter count of the
        logical circuit.  The result does not depend on the device: the same
        layer configuration gives the same summary on ``default.qubit`` and
        ``lightning.qubit``.

    Examples
    --------
    >>> from hqnn_forge.encoding import QuantumEncodingLayer
    >>> from hqnn_forge.diagnostics import circuit_summary
    >>> summary = circuit_summary(QuantumEncodingLayer(n_qubits=4, n_layers=2))
    >>> summary.n_two_qubit_gates
    8
    >>> print(summary)  # doctest: +SKIP
    Circuit summary: QuantumEncodingLayer on lightning.qubit (adjoint)
      qubits           : 4
      ...
    """
    layer, qlayer, n_qubits = resolve_encoding_layer(target, "circuit_summary")
    written = _written_tape(layer)
    tape = _decompose_logical(written)
    resources = _tape_resources(tape)
    qnode = qlayer.qnode
    # Counted on the tape as written, which count_inert_parameters decomposes
    # to LOGICAL_GATE_SET itself: a wide MultiRZ stays one diagonal gate there.
    # Its CNOT ladder would copy Z content between wires the MultiRZ leaves
    # untouched, and parameters on them would stop counting as inert.
    # Inertness is structural and does not depend on the input, but which
    # weight entries a gate parameter depends on is read off autograd, where
    # an input of 0 could hide a dependence (an input-times-weight angle).
    # Distinct magnitudes and one negative entry, as in sample_input, so the
    # amplitude encoder's state preparation keeps every rotation block.
    generic = 0.5 + torch.rand(
        input_width(layer), generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    generic[0] = -generic[0]
    n_inert = count_inert_weights(
        _written_tape(layer, generic), list(qlayer.qnode_weights.values())
    )
    return CircuitSummary(
        layer_type=type(layer).__name__,
        n_qubits=n_qubits,
        n_trainable_params=sum(
            p.numel() for p in qlayer.qnode_weights.values() if p.requires_grad
        ),
        depth=resources.depth,
        n_gates=resources.n_gates,
        n_two_qubit_gates=resources.n_two_qubit_gates,
        gate_counts=resources.gate_counts,
        device_name=str(qnode.device.name),
        diff_method=str(qnode.diff_method),
        n_inert_params=n_inert,
    )


def draw_circuit(target: nn.Module, inputs: torch.Tensor | None = None, decimals: int = 2) -> str:
    """
    Text drawing of the logical circuit for one sample, with the layer's
    current weights.  Suitable for ``print`` or a log line alongside a
    training run.

    Parameters
    ----------
    target:
        Same as for :func:`circuit_summary`.
    inputs:
        The one raw sample to draw the embedding for, shape
        ``(n_features,)`` -- one value per qubit, except for the amplitude
        encoder.  It goes through the layer's ``prepare_inputs`` as in
        ``forward``.  Default: :func:`sample_input`, zeros where the layer
        accepts them (every embedding rotation drawn as ``RX(0.00)``).  Pass
        a real sample to see the feature map it produces.
    decimals:
        Digits shown for gate parameters.  Default: 2.

    Notes
    -----
    Only the printed angles depend on ``inputs``; the gates and the wiring do
    not, which is why :func:`circuit_summary` does not take one.
    """
    layer, _, _ = resolve_encoding_layer(target, "draw_circuit")
    tape = _logical_tape(layer, inputs)
    return qml.drawer.tape_text(tape, decimals=decimals)
