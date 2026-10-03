"""
tests/test_inert_parameters.py
==============================
hqnn_forge.diagnostics.count_inert_parameters: the structural count of
trainable parameters that can never reach a measurement, checked against
autograd (an inert parameter has an exactly zero gradient for every input and
weight draw) and against hand-built tapes with a known answer.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import pennylane as qml
import pytest
import torch

from hqnn_forge.diagnostics import circuit_summary, count_inert_parameters, count_inert_weights
from hqnn_forge.diagnostics.circuit import _logical_tape, _written_tape
from hqnn_forge.encoding import DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod, Entangler, Readout
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier


class _DeviceKwargs(TypedDict):
    device_name: DeviceName
    diff_method: DiffMethod


CPU: _DeviceKwargs = {"device_name": "default.qubit", "diff_method": "backprop"}


ZERO = 1e-6  # float32 backprop leaves ~1e-8 on dead entries; live ones are ~1e-2


def _zero_gradient_entries(
    layer: QuantumEncodingLayer | IQPEncodingLayer, n_draws: int = 4
) -> int:
    """
    Weight entries whose gradient is zero (below ``ZERO``) for every one of
    ``n_draws`` random (input, weight) draws: the autograd view of "inert".
    """
    torch.manual_seed(0)
    weights: torch.Tensor = layer.qlayer.qnode_weights["weights"]
    always_zero = torch.ones_like(weights, dtype=torch.bool)
    for _ in range(n_draws):
        with torch.no_grad():
            weights.uniform_(0, 2 * math.pi)
        x = torch.rand(3, layer.n_qubits) * 2 * math.pi - math.pi
        weights.grad = None
        out = layer(x)
        (out * torch.rand(3, out.shape[1])).sum().backward()
        assert weights.grad is not None
        always_zero &= weights.grad.abs() < ZERO
    return int(always_zero.sum())


class TestAgainstAutograd:
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_ring_last_layer_omega(self, n_layers: int) -> None:
        """Exactly n_qubits inert parameters: the ω of each last-layer Rot."""
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=n_layers, **CPU)
        summary = circuit_summary(layer)
        assert summary.n_inert_params == 3
        assert summary.n_effective_params == 9 * n_layers - 3
        # At two or more layers the structural count is the whole story.
        if n_layers >= 2:
            assert _zero_gradient_entries(layer) == 3

    def test_structural_count_is_a_lower_bound_at_one_layer(self) -> None:
        """
        With one layer and RX embedding more entries are dead for the actual
        inputs (the φ of some Rots, #150), which the structural count does
        not claim: autograd finds 5 zeros against the structural 3.
        """
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, **CPU)
        assert circuit_summary(layer).n_inert_params == 3
        assert _zero_gradient_entries(layer) == 5

    def test_iqp_layer(self) -> None:
        layer = IQPEncodingLayer(n_qubits=3, n_layers=2, **CPU)
        assert circuit_summary(layer).n_inert_params == 3
        assert _zero_gradient_entries(layer) == 3

    @pytest.mark.parametrize("cls", [HybridBinaryClassifier, ParallelHybridClassifier])
    def test_default_eight_qubit_models_carry_eight_dead_weights(self, cls: type) -> None:
        model = cls(n_input_features=8, n_qubits=8, n_layers=2, **CPU)
        summary = circuit_summary(model)
        assert summary.n_trainable_params == 48
        assert summary.n_inert_params == 8
        assert summary.n_effective_params == 40
        assert summary.to_dict()["n_effective_params"] == 40
        lines = str(summary).splitlines()
        assert any(
            line.strip().startswith("inert params") and line.rstrip().endswith(": 8")
            for line in lines
        )
        assert any(
            line.strip().startswith("effective params") and line.rstrip().endswith(": 40")
            for line in lines
        )

    @pytest.mark.parametrize(
        ("entangler", "expected", "dead"),
        [("ring", 12, 12), ("strongly_entangling", 8, 12), ("brickwork", 16, 17)],
    )
    def test_first_readout(self, entangler: Entangler, expected: int, dead: int) -> None:
        """
        With only ⟨Z_0⟩ measured, the last layer's Rot on wires 1..n-1 is
        dead as a whole, far beyond the n_qubits ω of readout="all".  For the
        strongly entangling ansatz autograd finds more: the Z_0 content
        cancels through the range-2 CNOTs, which the per-wire propagation
        cannot see, so the structural count stays a lower bound.  Brickwork's
        light cone is the narrowest: the structural 16 are the last layer's
        Rot off wire 0 and wire 0's ω, and layer 0's Rot on wires 2 and 3.
        Autograd adds layer 0's φ on wire 1, which CNOT(1, 2) leaves acting
        on a state diagonal in Z_1 under RX embedding (#150).
        """
        layer = QuantumEncodingLayer(
            n_qubits=4, n_layers=2, entangler=entangler, readout="first", **CPU
        )
        assert circuit_summary(layer).n_inert_params == expected
        assert _zero_gradient_entries(layer) == dead

    @pytest.mark.parametrize("entangler", ["ring", "strongly_entangling", "brickwork"])
    @pytest.mark.parametrize("readout", ["all", "first"])
    def test_one_gate_slot_per_weight_entry(self, entangler: Entangler, readout: Readout) -> None:
        """
        n_effective_params subtracts gate-parameter slots from weight entries;
        that is only sound while the two line up one to one, as they do here.
        """
        layer = QuantumEncodingLayer(
            n_qubits=3, n_layers=2, entangler=entangler, readout=readout, **CPU
        )
        tape = _logical_tape(layer)
        slots = sum(1 for op in tape.operations for v in op.data if qml.math.requires_grad(v))
        assert slots == circuit_summary(layer).n_trainable_params

    def test_inert_omegas_have_zero_gradient_through_the_classifier(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(n_input_features=5, n_qubits=3, n_layers=2, **CPU)
        x = torch.randn(4, 5)
        model(x).sum().backward()
        grad = model.quantum_layer.qlayer.weights.grad
        assert grad is not None
        assert grad[-1, :, 2].abs().max().item() < ZERO
        assert (grad[-1, :, :2].abs() > ZERO).all()


def _tape(ops_fn, measurements) -> qml.tape.QuantumScript:
    with qml.queuing.AnnotatedQueue() as q:
        ops_fn()
        for m in measurements:
            qml.apply(m)
    return qml.tape.QuantumScript.from_queue(q)


def _p(value: float) -> torch.Tensor:
    return torch.tensor(value, requires_grad=True)


class TestHandBuiltTapes:
    def test_rz_before_z_readout_is_inert_but_rx_is_not(self) -> None:
        tape = _tape(lambda: (qml.RZ(_p(0.3), 0), qml.RX(_p(0.4), 0)), [qml.expval(qml.PauliZ(0))])
        # RZ is applied first, RX after it: the RZ acts on Z content already
        # turned into X/Y by the later RX, so it is live; only a trailing RZ would be inert.
        assert count_inert_parameters(tape) == 0
        tape = _tape(lambda: (qml.RX(_p(0.4), 0), qml.RZ(_p(0.3), 0)), [qml.expval(qml.PauliZ(0))])
        assert count_inert_parameters(tape) == 1

    def test_gate_on_an_unmeasured_wire_is_inert(self) -> None:
        tape = _tape(
            lambda: (qml.RX(_p(0.1), 1), qml.Rot(_p(0.1), _p(0.2), _p(0.3), 1)),
            [qml.expval(qml.PauliZ(0))],
        )
        assert count_inert_parameters(tape) == 4

    def test_cnot_moves_z_content_to_the_control_and_xy_to_the_target(self) -> None:
        # Z_1 is measured; CNOT(0,1) makes it Z_0 Z_1, so an RX on wire 0
        # before the CNOT is live, while a trailing RZ on wire 0 stays inert.
        tape = _tape(
            lambda: (qml.RX(_p(0.1), 0), qml.CNOT([0, 1]), qml.RZ(_p(0.2), 0)),
            [qml.expval(qml.PauliZ(1))],
        )
        assert count_inert_parameters(tape) == 1
        # X/Y content on the control spreads to the target (Y_0 → Y_0 X_1), so
        # an RZ on the target before the CNOT is live once the control has
        # been rotated out of the Z basis after it: ⟨Z_0⟩ then contains
        # sin(b)·⟨Y_0 (cos(a) X_1 + sin(a) Y_1)⟩, which depends on a.
        tape = _tape(
            lambda: (qml.RZ(_p(0.1), 1), qml.CNOT([0, 1]), qml.RX(_p(0.2), 0)),
            [qml.expval(qml.PauliZ(0))],
        )
        assert count_inert_parameters(tape) == 0
        # Without the RX the control keeps Z content only and the RZ is inert.
        tape = _tape(
            lambda: (qml.RZ(_p(0.1), 1), qml.CNOT([0, 1])),
            [qml.expval(qml.PauliZ(0))],
        )
        assert count_inert_parameters(tape) == 1
        tape = _tape(
            lambda: (qml.RZ(_p(0.1), 1), qml.CNOT([0, 1]), qml.RX(_p(0.2), 1)),
            [qml.expval(qml.PauliZ(1))],
        )
        assert count_inert_parameters(tape) == 0  # X_1 → X_1 through the CNOT: the RZ is live

    def test_global_phase_neither_mixes_nor_counts(self) -> None:
        # Graph-based decomposition emits GlobalPhase.  It commutes with the
        # measurement, so the RZ before it stays inert; its own parameter is
        # not counted, since under state() a global phase is observable.
        tape = _tape(
            lambda: (qml.RX(_p(0.4), 0), qml.RZ(_p(0.3), 0), qml.GlobalPhase(_p(0.2), wires=0)),
            [qml.expval(qml.PauliZ(0))],
        )
        assert count_inert_parameters(tape) == 1

    def test_non_z_measurement_disables_the_shortcut(self) -> None:
        tape = _tape(lambda: (qml.RZ(_p(0.3), 0),), [qml.expval(qml.PauliX(0))])
        assert count_inert_parameters(tape) == 0
        # probs reads the diagonal only, so a trailing RZ is inert under it (#236).
        tape = _tape(lambda: (qml.RZ(_p(0.3), 0),), [qml.probs(wires=[0])])
        assert count_inert_parameters(tape) == 1

    def test_non_trainable_parameters_are_not_counted(self) -> None:
        tape = _tape(lambda: (qml.RZ(0.3, 0),), [qml.expval(qml.PauliZ(0))])
        assert count_inert_parameters(tape) == 0

    def test_mid_circuit_measurement_feeding_a_conditional_is_live(self) -> None:
        """
        RX(a) on wire 1 reaches ⟨Z_0⟩ only through the measurement of wire 1
        and the X it conditions on wire 0; the gradient is nonzero, so the
        parameter must not be counted.  A gate on a wire nothing measures
        stays inert.
        """

        def ops() -> None:
            qml.RX(_p(0.7), 1)
            qml.RX(_p(0.2), 2)
            m = qml.measure(1)
            qml.cond(m, qml.PauliX)(0)

        tape = _tape(ops, [qml.expval(qml.PauliZ(0))])
        assert count_inert_parameters(tape) == 1

        @qml.qnode(qml.device("default.qubit"), interface="torch", diff_method="backprop")
        def circuit(a: torch.Tensor) -> Any:
            qml.RX(a, 1)
            m = qml.measure(1)
            qml.cond(m, qml.PauliX)(0)
            return qml.expval(qml.PauliZ(0))

        a = _p(0.7)
        circuit(a).backward()
        assert a.grad is not None and abs(a.grad.item()) > 0.1

    def test_template_is_decomposed_before_counting(self) -> None:
        """
        A template carries its weights as one tensor; counted as is it would
        be one slot, and the Rots inside it would never be seen.
        """
        weights = torch.rand(2, 2, 3, requires_grad=True)
        tape = _tape(
            lambda: qml.StronglyEntanglingLayers(weights, wires=[1, 2]),
            [qml.expval(qml.PauliZ(0))],
        )
        assert count_inert_parameters(tape) == 12  # every weight, on unmeasured wires
        weights = torch.rand(1, 3, 3, requires_grad=True)
        tape = _tape(
            lambda: qml.StronglyEntanglingLayers(weights, wires=[0, 1, 2]),
            [qml.expval(qml.PauliZ(w)) for w in range(3)],
        )
        assert count_inert_parameters(tape) == 3  # the ω of each Rot

    def test_broadcast_tape_is_rejected(self) -> None:
        tape = _tape(
            lambda: (qml.RX(torch.rand(4, requires_grad=True), 1),), [qml.expval(qml.PauliZ(0))]
        )
        with pytest.raises(ValueError, match="broadcast"):
            count_inert_parameters(tape)


# ---------------------------------------------------------------------------
# Diagonal gates and measurements recognised structurally (#236)
# ---------------------------------------------------------------------------

Ops = Callable[[torch.Tensor], None]
Measure = Callable[[], list[Any]]


def _prepare() -> None:
    for w in range(3):
        qml.RX(0.3 + 0.2 * w, wires=w)
        qml.RY(0.5 - 0.1 * w, wires=w)
    qml.CNOT([0, 1])
    qml.CNOT([1, 2])


def _gradient_is_zero(ops: Ops, measure: Measure, n_draws: int = 4) -> bool:
    """Autograd: the gradient w.r.t. the one trainable angle vanishes at every draw."""
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(a: torch.Tensor) -> Any:
        _prepare()
        ops(a)
        return measure()

    g = torch.Generator().manual_seed(3)
    for _ in range(n_draws):
        a = (torch.rand((), generator=g, dtype=torch.float64) * 2 * math.pi).requires_grad_(True)
        out = circuit(a)
        parts = out if isinstance(out, (list, tuple)) else [out]
        value = torch.zeros((), dtype=torch.float64)
        for k, part in enumerate(parts):
            value = value + torch.as_tensor(part).to(torch.float64).sum() * (k + 1.3)
        (grad,) = torch.autograd.grad(value, a)
        if abs(float(grad)) > 1e-10:
            return False
    return True


def _count(ops: Ops, measure: Measure) -> int:
    def circuit() -> None:
        _prepare()
        ops(_p(0.4))

    return count_inert_parameters(_tape(circuit, measure()))


def _rz0(a: torch.Tensor) -> None:
    qml.RZ(a, wires=0)


CASES: list[Any] = [
    pytest.param(_rz0, lambda: [qml.expval(2.0 * qml.PauliZ(0))], id="sprod"),
    pytest.param(_rz0, lambda: [qml.expval(qml.PauliZ(0) + qml.PauliZ(1))], id="sum"),
    pytest.param(
        _rz0,
        lambda: [qml.expval(qml.prod(qml.PauliZ(0), qml.prod(qml.PauliZ(1), qml.PauliZ(2))))],
        id="nested-prod",
    ),
    pytest.param(
        _rz0, lambda: [qml.expval(qml.PauliZ(0) @ qml.Identity(1))], id="z-times-identity"
    ),
    pytest.param(_rz0, lambda: [qml.probs(wires=[0])], id="probs"),
    pytest.param(_rz0, lambda: [qml.probs(wires=[0, 1])], id="probs-two-wires"),
    pytest.param(
        lambda a: qml.CRZ(a, wires=[0, 1]),
        lambda: [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))],
        id="crz",
    ),
    pytest.param(
        lambda a: qml.ControlledPhaseShift(a, wires=[1, 2]),
        lambda: [qml.expval(qml.PauliZ(2))],
        id="controlled-phase-shift",
    ),
    pytest.param(
        lambda a: qml.adjoint(qml.RZ(a, wires=0)),
        lambda: [qml.expval(qml.PauliZ(0))],
        id="adjoint-rz",
    ),
]


class TestStructuralDiagonals:
    @pytest.mark.parametrize(("ops", "measure"), CASES)
    def test_counted_inert_and_zero_gradient(self, ops: Ops, measure: Measure) -> None:
        assert _count(ops, measure) == 1
        assert _gradient_is_zero(ops, measure)

    def test_sample_and_counts_are_basis_measurements(self) -> None:
        for measure in (lambda: [qml.sample(wires=[0])], lambda: [qml.counts(wires=[0])]):
            assert _count(_rz0, measure) == 1
        # Their distribution is probs, whose gradient vanishes (probs case above).

    def test_conditional_rz(self) -> None:
        def ops(a: torch.Tensor) -> None:
            m = qml.measure(2)
            qml.cond(m, qml.RZ)(a, wires=0)

        def measure() -> list[Any]:
            return [qml.expval(qml.PauliZ(0))]

        def whole() -> None:
            _prepare()
            ops(_p(0.4))

        tape = _tape(whole, measure())
        assert any(op.name.startswith("Conditional") for op in tape.operations)
        assert count_inert_parameters(tape) == 1
        assert _gradient_is_zero(ops, measure)

    def test_state_still_reads_coherences(self) -> None:
        def measure() -> list[Any]:
            return [qml.state()]

        assert _count(_rz0, measure) == 0

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(a: torch.Tensor) -> Any:
            _prepare()
            _rz0(a)
            return qml.state()

        a = _p(0.4).to(torch.float64).detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(circuit(a).real.sum(), a)
        assert abs(float(grad)) > 1e-6  # the phase is visible in the state

    def test_x_readout_still_disables_it(self) -> None:
        assert _count(_rz0, lambda: [qml.expval(qml.PauliX(0))]) == 0
        assert _count(lambda a: qml.CRZ(a, wires=[0, 1]), lambda: [qml.expval(qml.PauliX(1))]) == 0

    def test_a_parametrised_gate_is_not_judged_by_its_matrix(self) -> None:
        # RX(0) is the identity; its matrix at that value is diagonal, the gate is not.
        def circuit() -> None:
            _prepare()
            qml.RX(torch.tensor(0.0, requires_grad=True), 0)

        tape = _tape(circuit, [qml.expval(qml.PauliZ(0))])
        assert count_inert_parameters(tape) == 0

    def test_a_wide_parameterless_gate_never_builds_its_matrix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A 20-wire QFT's dense matrix would take 16 TiB; it counts as mixing.
        widths: list[int] = []
        real_matrix = qml.matrix

        def spy(op: Any, *args: Any, **kwargs: Any) -> Any:
            widths.append(len(op.wires))
            return real_matrix(op, *args, **kwargs)

        monkeypatch.setattr(qml, "matrix", spy)
        tape = _tape(
            lambda: (qml.RZ(_p(0.3), 0), qml.QFT(wires=range(20)), qml.RZ(_p(0.4), 0)),
            [qml.expval(qml.PauliZ(0))],
        )
        assert count_inert_parameters(tape) == 1  # the trailing RZ only
        assert all(w <= 6 for w in widths)
        # A narrow parameterless diagonal gate is still judged by its matrix.
        tape = _tape(lambda: (qml.RZ(_p(0.3), 0), qml.CCZ([0, 1, 2])), [qml.probs(wires=[0])])
        assert count_inert_parameters(tape) == 1


_GATES_1 = ["RX", "RY", "RZ", "PhaseShift", "Rot", "Hadamard", "S", "T"]
_GATES_2 = ["CNOT", "CZ", "IsingZZ", "MultiRZ", "SWAP", "CRX", "CRZ", "ControlledPhaseShift"]
_READOUTS: list[Measure] = [
    lambda: [qml.expval(qml.PauliZ(0))],
    lambda: [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))],
    lambda: [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))],
    lambda: [qml.expval(qml.PauliX(1))],
    lambda: [qml.probs(wires=[0])],
    lambda: [qml.expval(2.0 * qml.PauliZ(2))],
    lambda: [qml.expval(qml.PauliZ(0) + qml.PauliZ(1))],
]


def _random_circuit(rng: np.random.Generator) -> list[tuple[str, list[int], int]]:
    """(gate name, wires, number of angles) for 8 random gates on 3 wires."""
    gates = []
    for _ in range(8):
        if rng.random() < 0.6:
            name = str(rng.choice(_GATES_1))
            wires = [int(rng.integers(3))]
        else:
            name = str(rng.choice(_GATES_2))
            wires = [int(w) for w in rng.choice(3, size=2, replace=False)]
        n_params = {
            "Rot": 3,
            "RX": 1,
            "RY": 1,
            "RZ": 1,
            "PhaseShift": 1,
            "IsingZZ": 1,
            "MultiRZ": 1,
            "CRX": 1,
            "CRZ": 1,
            "ControlledPhaseShift": 1,
        }.get(name, 0)
        gates.append((name, wires, n_params))
    return gates


def _apply(gates: list[tuple[str, list[int], int]], params: list[torch.Tensor]) -> None:
    it = iter(params)
    for name, wires, n in gates:
        getattr(qml, name)(*[next(it) for _ in range(n)], wires=wires)


def _check_random_circuit(
    gates: list[tuple[str, list[int], int]],
    readout: Measure,
    rng: np.random.Generator,
    dev: Any,
) -> tuple[int, int]:
    """(structural inert count, parameters whose gradient is zero at every draw)."""
    n = sum(k for _, _, k in gates)
    params = [_p(float(v)) for v in rng.uniform(0, 2 * math.pi, n)]
    inert = count_inert_parameters(_tape(lambda: _apply(gates, params), readout()))

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(*ps: torch.Tensor) -> Any:
        _apply(gates, list(ps))
        return readout()

    always_zero = torch.ones(n, dtype=torch.bool)
    for _ in range(3):
        draws = rng.uniform(0, 2 * math.pi, n)
        ps = [torch.tensor(float(v), dtype=torch.float64, requires_grad=True) for v in draws]
        out = circuit(*ps)
        out = out[0] if isinstance(out, (list, tuple)) else out
        weights = torch.linspace(1.0, 2.0, out.numel(), dtype=torch.float64)
        grads = torch.autograd.grad((out.reshape(-1) * weights).sum(), ps, allow_unused=True)
        always_zero &= torch.tensor([g is None or abs(float(g)) < 1e-9 for g in grads])
    return inert, int(always_zero.sum())


class TestSoundnessOnRandomCircuits:
    def test_nothing_counted_inert_has_a_gradient(self) -> None:
        """
        The #165 review's check, with the new diagonal gates and readouts: the
        count never exceeds the parameters whose gradient is zero at every draw.
        """
        rng = np.random.default_rng(0)
        dev = qml.device("default.qubit", wires=3)
        checked = counted = 0
        for trial in range(400):
            gates = _random_circuit(rng)
            if sum(k for _, _, k in gates) == 0:
                continue
            inert, zero = _check_random_circuit(gates, _READOUTS[trial % len(_READOUTS)], rng, dev)
            assert inert <= zero, (gates, trial)
            checked += 1
            counted += inert
        assert checked > 350 and counted > 50  # the check has teeth


# ---------------------------------------------------------------------------
# Counting per weight entry (#235)
# ---------------------------------------------------------------------------


def _entry_tape(ops: Callable[[torch.Tensor], None], w: torch.Tensor) -> qml.tape.QuantumScript:
    def circuit() -> None:
        qml.RY(0.4, wires=0)
        ops(w)

    return _tape(circuit, [qml.expval(qml.PauliZ(0))])


class TestPerWeightEntry:
    def test_entry_shared_by_an_inert_and_a_live_slot_is_live(self) -> None:
        w = torch.tensor([0.3], requires_grad=True)

        def ops(v: torch.Tensor) -> None:
            qml.RX(v[0], wires=0)  # live
            qml.RZ(v[0], wires=0)  # inert: trailing RZ before a Z readout

        tape = _entry_tape(ops, w)
        assert count_inert_parameters(tape) == 1  # one inert slot
        assert count_inert_weights(tape, [w]) == 0  # but the entry moves the output

    def test_angle_from_two_entries_makes_both_inert(self) -> None:
        w = torch.tensor([0.3, 0.7], requires_grad=True)

        def ops(v: torch.Tensor) -> None:
            qml.RZ(v[0] * v[1], wires=0)

        tape = _entry_tape(ops, w)
        assert count_inert_parameters(tape) == 1
        assert count_inert_weights(tape, [w]) == 2

    def test_entry_in_two_inert_slots_counts_once(self) -> None:
        w = torch.tensor([0.3, 0.9], requires_grad=True)

        def ops(v: torch.Tensor) -> None:
            qml.RX(v[1], wires=0)
            qml.RZ(v[0], wires=0)
            qml.PhaseShift(v[0], wires=0)

        tape = _entry_tape(ops, w)
        assert count_inert_parameters(tape) == 2  # slots: more than the one dead entry
        assert count_inert_weights(tape, [w]) == 1  # so n_trainable - inert >= 0 again

    def test_unused_and_frozen_entries(self) -> None:
        w = torch.tensor([0.3, 0.5], requires_grad=True)
        frozen = torch.tensor([0.1])

        def ops(v: torch.Tensor) -> None:
            qml.RX(v[0], wires=0)
            qml.RX(frozen[0], wires=0)

        tape = _entry_tape(ops, w)
        assert (
            count_inert_weights(tape, [w, frozen]) == 1
        )  # v[1] feeds nothing; frozen not counted

    def test_entry_feeding_only_a_global_phase_is_live(self) -> None:
        # count_inert_parameters does not count a GlobalPhase slot (it reaches
        # state()), so its entry must not read as feeding nothing.
        w = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)

        def circuit() -> None:
            qml.RX(w[0], wires=0)
            qml.GlobalPhase(w[1], wires=0)

        tape = _tape(circuit, [qml.state()])
        assert count_inert_parameters(tape) == 0
        assert count_inert_weights(tape, [w]) == 0

    def test_counted_entries_have_zero_gradient(self) -> None:
        w = torch.tensor([0.3, 0.7, 1.1], dtype=torch.float64, requires_grad=True)

        def ops(v: torch.Tensor) -> None:
            qml.RX(v[2], wires=0)
            qml.RZ(v[0] * v[1], wires=0)

        assert count_inert_weights(_entry_tape(ops, w), [w]) == 2

        @qml.qnode(qml.device("default.qubit", wires=1), interface="torch")
        def circuit(v: torch.Tensor) -> Any:
            qml.RY(0.4, wires=0)
            ops(v)
            return qml.expval(qml.PauliZ(0))

        circuit(w).backward()
        assert w.grad is not None
        assert abs(float(w.grad[0])) < 1e-12 and abs(float(w.grad[1])) < 1e-12
        assert abs(float(w.grad[2])) > 1e-3

    @pytest.mark.parametrize(
        "build",
        [
            lambda: QuantumEncodingLayer(n_qubits=4, n_layers=2, **CPU),
            lambda: QuantumEncodingLayer(
                n_qubits=4, n_layers=2, entangler="strongly_entangling", readout="first", **CPU
            ),
            lambda: IQPEncodingLayer(n_qubits=4, n_layers=2, **CPU),
            lambda: DataReuploadingLayer(
                n_qubits=3, n_layers=3, trainable_input_scaling=True, **CPU
            ),
        ],
        ids=["ring", "strongly-first", "iqp", "reuploading-scaled"],
    )
    def test_built_in_layers_count_the_same_either_way(self, build: Callable[[], Any]) -> None:
        # Each weight entry feeds exactly one slot in the library's layers.
        layer = build()
        x = torch.rand(layer.n_qubits, dtype=torch.float64) + 0.5
        tape = _written_tape(layer, x)
        weights = list(layer.qlayer.qnode_weights.values())
        assert count_inert_weights(tape, weights) == count_inert_parameters(tape)
        summary = circuit_summary(layer)
        assert summary.n_inert_params == count_inert_parameters(tape)
        assert summary.n_effective_params >= 0
