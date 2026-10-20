"""
tests/test_rydberg_entangler.py
===============================
``entangler="rydberg"``: :func:`hqnn_forge.circuits.neutral_atom_layer` on the
fixed chain register :func:`hqnn_forge.circuits.rydberg_chain` as a
variational block of every encoder and classifier (#418).

Expected values come from somewhere other than the code under test: the
register's numbers from the closed form ``R_b = (C6/Ω)^(1/6)``, the schedules
from the ``i mod 3`` rule derived in ``rydberg_chain``'s docstring and written
out by hand, the circuit outputs from a NumPy state vector built from 2x2
gate matrices, and the inert-parameter counts from autograd in float64.
"""

from __future__ import annotations

import math
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import torch

from hqnn_forge.circuits import (
    RYDBERG_BLOCKADE_RADIUS,
    RYDBERG_SPACING,
    blockade_schedule,
    rydberg_chain,
)
from hqnn_forge.diagnostics import CircuitSummary, circuit_summary
from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    QuantumEncodingLayer,
)
from hqnn_forge.encoding.angle_embedding import (
    Entangler,
    Readout,
    RotationAxis,
    validate_circuit_options,
    variational_weight_shape,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.utils import load_checkpoint, save_checkpoint

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
RY_: dict[str, Any] = {"entangler": "rydberg", **CPU}


def _random_weights(layer: Any, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.rand(p.shape, generator=g, dtype=p.dtype) * 2 * torch.pi)


# ---------------------------------------------------------------------------
# NumPy reference: state vector of embed + (CZ chain, Rot) blocks
# ---------------------------------------------------------------------------


def _rz(a: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a / 2), math.sin(a / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a / 2), math.sin(a / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _rot(phi: float, theta: float, omega: float) -> np.ndarray:
    return _rz(omega) @ _ry(theta) @ _rz(phi)


def _chain_cz_phase(n: int) -> np.ndarray:
    """``∏_i CZ(i, i+1)``: ``(-1)^{Σ_i b_i b_{i+1}}``, wire 0 the most significant bit."""
    phase = np.ones(2**n, dtype=complex)
    for index in range(2**n):
        bits = [(index >> (n - 1 - wire)) & 1 for wire in range(n)]
        phase[index] = (-1) ** sum(bits[i] * bits[i + 1] for i in range(n - 1))
    return phase


def _expval_z(state: np.ndarray, n: int) -> np.ndarray:
    probs = np.abs(state) ** 2
    out = np.zeros(n)
    for index, p in enumerate(probs):
        for wire in range(n):
            out[wire] += p * (1 - 2 * ((index >> (n - 1 - wire)) & 1))
    return out


def _reference(
    x: np.ndarray, w: np.ndarray, rotation: str, *, reupload: bool = False
) -> np.ndarray:
    """``⟨Z_i⟩`` of RX/RY(x) on |0…0⟩, then per layer CZ(i, i+1) for all i and Rot on all."""
    n, n_layers = x.shape[0], w.shape[0]
    gate = _rx if rotation == "X" else _ry
    embed = reduce(np.kron, [gate(float(x[i])) for i in range(n)])
    phase = _chain_cz_phase(n)
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    for ell in range(n_layers):
        if ell == 0 or reupload:
            state = embed @ state
        state = phase * state
        state = reduce(np.kron, [_rot(*w[ell, i]) for i in range(n)]) @ state
    return _expval_z(state, n)


# ---------------------------------------------------------------------------
# The register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_blockade_radius_is_the_closed_form_at_the_reference_omega(self) -> None:
        """``(C6/Ω)^(1/6)`` with ``C6 = 2π · 862690`` and ``Ω = 4π``: ``431345^(1/6)``."""
        assert RYDBERG_BLOCKADE_RADIUS == pytest.approx(431345.0 ** (1 / 6), rel=1e-15)
        assert round(RYDBERG_BLOCKADE_RADIUS, 2) == 8.69

    def test_only_neighbours_are_blockaded(self) -> None:
        """``a < R_b ≤ 2a``; ``V/Ω = (R_b/a)⁶`` is 9.25 for neighbours and 9.25/64 beyond."""
        a, r = RYDBERG_SPACING, RYDBERG_BLOCKADE_RADIUS
        assert a == 6.0 and a < r <= 2 * a
        assert (r / a) ** 6 == pytest.approx(9.2452, abs=1e-4)
        assert (r / (2 * a)) ** 6 == pytest.approx(0.14446, abs=1e-5)

    def test_positions_are_a_chain_at_the_spacing(self) -> None:
        expected = torch.tensor([[0.0, 0.0], [6.0, 0.0], [12.0, 0.0], [18.0, 0.0]])
        torch.testing.assert_close(rydberg_chain(4).positions, expected.double(), rtol=0, atol=0)

    def test_graph_is_the_path_without_triangles(self) -> None:
        register = rydberg_chain(5)
        assert register.interaction_graph(RYDBERG_BLOCKADE_RADIUS) == [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ]
        assert register.blockade_triangles(RYDBERG_BLOCKADE_RADIUS) == []

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1, []),
            (2, [[(0, 1)]]),
            (3, [[(0, 1)], [(1, 2)]]),
            (4, [[(0, 1)], [(1, 2)], [(2, 3)]]),
            (5, [[(0, 1), (3, 4)], [(1, 2)], [(2, 3)]]),
            (7, [[(0, 1), (3, 4)], [(1, 2), (4, 5)], [(2, 3), (5, 6)]]),
        ],
    )
    def test_schedule_puts_edge_i_in_layer_i_mod_3(
        self, n: int, expected: list[list[tuple[int, int]]]
    ) -> None:
        """
        Hand-counted: CZs (i, i+1) and (j, j+1) are (j − i − 1)·6 µm apart at their
        closest, ≥ 8.69 µm only for j ≥ i + 3.  So 0 | 1 | 2 | 3 … go to layers
        0 | 1 | 2 | 0 …, and with ccz=True nothing changes (no triangle).
        """
        register = rydberg_chain(n)
        assert blockade_schedule(register, RYDBERG_BLOCKADE_RADIUS) == expected
        assert blockade_schedule(register, RYDBERG_BLOCKADE_RADIUS, ccz=True) == expected

    @pytest.mark.parametrize("n", [0, -1, 2.0, True])
    def test_rejects_a_bad_size(self, n: object) -> None:
        with pytest.raises(ValueError, match=r"n_atoms must be an integer >= 1"):
            rydberg_chain(n)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class TestOptions:
    @pytest.mark.parametrize(("n_qubits", "n_layers"), [(2, 1), (4, 3), (5, 2)])
    def test_weight_shape_is_one_rot_per_qubit_per_layer(
        self, n_qubits: int, n_layers: int
    ) -> None:
        assert variational_weight_shape("rydberg", n_qubits, n_layers) == (n_layers, n_qubits, 3)
        layer = QuantumEncodingLayer(n_qubits=n_qubits, n_layers=n_layers, **RY_)
        assert tuple(layer.qlayer.weights.shape) == (n_layers, n_qubits, 3)

    @pytest.mark.parametrize("readout", ["all", "first"])
    @pytest.mark.parametrize("rotation", ["X", "Y", None])
    def test_validate_circuit_options_accepts_it(
        self, readout: Readout, rotation: RotationAxis | None
    ) -> None:
        validate_circuit_options(3, "rydberg", readout, rotation)  # does not raise
        # The positive check that it is *covered*: the error lists it among the choices
        with pytest.raises(ValueError, match=r"'hardware_efficient', 'rydberg'; got 'x'"):
            validate_circuit_options(3, "x", readout, rotation)  # type: ignore[arg-type]

    def test_an_unknown_entangler_lists_rydberg(self) -> None:
        message = (
            "entangler must be one of 'ring', 'strongly_entangling', 'brickwork', "
            "'hardware_efficient', 'rydberg'; got 'neutral_atom'."
        )
        with pytest.raises(ValueError) as excinfo:
            validate_circuit_options(3, "neutral_atom", "all")  # type: ignore[arg-type]
        assert str(excinfo.value) == message
        with pytest.raises(ValueError) as excinfo:
            variational_weight_shape("neutral_atom", 3, 1)  # type: ignore[arg-type]
        assert str(excinfo.value) == message

    def test_repr_names_it(self) -> None:
        assert "entangler='rydberg'" in repr(QuantumEncodingLayer(n_qubits=2, n_layers=1, **RY_))


# ---------------------------------------------------------------------------
# The circuit against NumPy
# ---------------------------------------------------------------------------


class TestCircuit:
    @pytest.mark.parametrize("rotation", ["X", "Y"])
    def test_angle_layer_matches_numpy(self, rotation: RotationAxis) -> None:
        n, n_layers = 4, 2
        layer = QuantumEncodingLayer(
            n_qubits=n, n_layers=n_layers, rotation=rotation, **RY_
        ).double()
        _random_weights(layer, 0)
        w = layer.qlayer.weights.detach().numpy()
        x = torch.rand(3, n, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        expected = np.stack([_reference(xi, w, rotation) for xi in x.numpy()])
        with torch.no_grad():
            out = layer(x).numpy()
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)

    def test_reuploading_interleaves_embedding_and_blocks(self) -> None:
        # The block ignores layer_offset, so one block at a time is the same chain.
        n, n_layers = 3, 3
        layer = DataReuploadingLayer(n_qubits=n, n_layers=n_layers, **RY_).double()
        _random_weights(layer, 2)
        w = layer.qlayer.weights.detach().numpy()
        x = torch.rand(3, n, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        expected = np.stack([_reference(xi, w, "X", reupload=True) for xi in x.numpy()])
        with torch.no_grad():
            out = layer(x).numpy()
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)

    def test_two_qubits_is_one_cz_then_rot(self) -> None:
        """
        n = 2, one layer, RX(x) embedding, Rot = identity on wire 1 and RY(π/2) on wire 0.

        CZ is a phase on |11⟩ and leaves ⟨Z_1⟩ = cos x_1.  ⟨Z_0⟩ after RY(π/2) is
        −⟨X_0⟩ before it, and CZ maps X_0 to X_0 Z_1.  On the RX-embedded product
        state ⟨X_0⟩ = 0, so ⟨Z_0⟩ = 0, whatever x.
        """
        layer = QuantumEncodingLayer(n_qubits=2, n_layers=1, **RY_).double()
        with torch.no_grad():
            layer.qlayer.weights.zero_()
            layer.qlayer.weights[0, 0, 1] = math.pi / 2
        x = torch.tensor([[0.3, 1.1], [-2.0, 0.4]], dtype=torch.float64)
        with torch.no_grad():
            out = layer(x)
        torch.testing.assert_close(out[:, 1], torch.cos(x[:, 1]), rtol=0, atol=1e-12)
        torch.testing.assert_close(
            out[:, 0], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
        )

    @pytest.mark.parametrize("cls", [IQPEncodingLayer, AmplitudeEncodingLayer])
    def test_iqp_and_amplitude_run_the_chain(self, cls: type) -> None:
        n, n_layers = 3, 2
        summary = circuit_summary(cls(n_qubits=n, n_layers=n_layers, **RY_))
        # IQP's embedding has no CZ, amplitude's state preparation neither:
        # every CZ is the block's.
        assert summary.gate_counts["CZ"] == (n - 1) * n_layers
        assert summary.gate_counts["Rot"] == n * n_layers
        assert summary.n_blockade_layers == 2 * n_layers


# ---------------------------------------------------------------------------
# circuit_summary
# ---------------------------------------------------------------------------


class TestSummary:
    @pytest.mark.parametrize(
        # Hand-counted per block: n − 1 CZs in min(3, n − 1) layers
        ("n_qubits", "per_block"),
        [(2, 1), (3, 2), (4, 3), (5, 3), (6, 3)],
    )
    @pytest.mark.parametrize("n_layers", [1, 2])
    def test_blockade_layers_and_counts(
        self, n_qubits: int, per_block: int, n_layers: int
    ) -> None:
        summary = circuit_summary(
            QuantumEncodingLayer(n_qubits=n_qubits, n_layers=n_layers, **RY_)
        )
        assert summary.n_blockade_layers == per_block * n_layers
        assert summary.n_two_qubit_gates == (n_qubits - 1) * n_layers
        assert summary.gate_counts == {
            "CZ": (n_qubits - 1) * n_layers,
            "RX": n_qubits,
            "Rot": n_qubits * n_layers,
        }
        assert summary.n_trainable_params == 3 * n_qubits * n_layers

    def test_tape_depth_follows_the_recorded_schedule(self) -> None:
        """
        n = 5, one layer.  The CZs are recorded in schedule order (0,1), (3,4) |
        (1,2) | (2,3), so as soon as possible they take moments 1, 1, 2, 3: three,
        as many as the blockade layers, plus RX before and Rot after, depth 5.
        Wire-disjoint packing alone would allow (0,1), (2,3) | (1,2), (3,4), two
        moments, which the blockade forbids ((0,1) and (2,3) are 6 µm apart).
        """
        summary = circuit_summary(QuantumEncodingLayer(n_qubits=5, n_layers=1, **RY_))
        assert summary.depth == 5
        assert summary.n_blockade_layers == 3

    @pytest.mark.parametrize(
        "entangler", ["ring", "strongly_entangling", "brickwork", "hardware_efficient"]
    )
    def test_other_entanglers_report_none(self, entangler: Entangler) -> None:
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, entangler=entangler, **CPU)
        assert circuit_summary(layer).n_blockade_layers is None
        assert "blockade layers" not in str(circuit_summary(layer))

    def test_str_and_dict_carry_it(self) -> None:
        summary = circuit_summary(QuantumEncodingLayer(n_qubits=4, n_layers=2, **RY_))
        assert any(
            line.strip().startswith("blockade layers") and line.rstrip().endswith(": 6")
            for line in str(summary).splitlines()
        )
        d = summary.to_dict()
        assert d["n_blockade_layers"] == 6
        d.pop("n_effective_params")
        assert CircuitSummary(**d) == summary

    @pytest.mark.parametrize("readout", ["all", "first"])
    @pytest.mark.parametrize(("n", "n_layers"), [(4, 3), (5, 2), (2, 3)])
    def test_inert_count_is_the_derived_and_the_numerically_dead_parameters(
        self, readout: Readout, n: int, n_layers: int
    ) -> None:
        """
        Under "all" only the last-layer ω are dead: n.  Under "first", counted
        back from the readout, block k < n keeps Rot live on wires 0 … k − 1,
        wire k's ω dead and every Rot beyond wire k dead, 1 + 3(n − 1 − k);
        blocks with k ≥ n leave none dead.  Autograd in float64 agrees, where
        float32 rounding leaves some dead gradients near 1e-7.
        """
        layer = QuantumEncodingLayer(n_qubits=n, n_layers=n_layers, readout=readout, **RY_)
        layer = layer.double()
        grads = []
        for seed in range(5):
            _random_weights(layer, seed)
            x = torch.rand(3, n, generator=torch.Generator().manual_seed(100 + seed)) * 6 - 3
            layer.qlayer.weights.grad = None
            layer(x.double()).sum().backward()
            grads.append(layer.qlayer.weights.grad.abs())
        dead = int((torch.stack(grads).amax(0) < 1e-10).sum())
        if readout == "all":
            expected = n
        else:
            expected = sum(1 + 3 * (n - 1 - k) for k in range(min(n_layers, n)))
        assert dead == expected
        assert circuit_summary(layer).n_inert_params == expected


# ---------------------------------------------------------------------------
# Gradients and models
# ---------------------------------------------------------------------------


class TestGradients:
    def test_parameter_shift_agrees_with_backprop(self) -> None:
        backprop = QuantumEncodingLayer(n_qubits=3, n_layers=2, **RY_)
        shift = QuantumEncodingLayer(
            n_qubits=3,
            n_layers=2,
            entangler="rydberg",
            device_name="default.qubit",
            diff_method="parameter-shift",
        )
        _random_weights(backprop, 4)
        shift.load_state_dict(backprop.state_dict())
        x = torch.rand(4, 3, generator=torch.Generator().manual_seed(5))
        g_b = torch.autograd.grad(backprop(x).sum(), backprop.qlayer.weights)[0]
        g_s = torch.autograd.grad(shift(x).sum(), shift.qlayer.weights)[0]
        # Every weight but the 3 last-layer ω moves the output
        assert int((g_b.abs() > 1e-6).sum()) == 3 * 3 * 2 - 3
        torch.testing.assert_close(g_s, g_b, atol=2e-6, rtol=0)


@pytest.mark.parametrize(
    "cls, extra",
    [
        (HybridBinaryClassifier, {}),
        (ParallelHybridClassifier, {}),
        (MulticlassHybridClassifier, {"n_classes": 3}),
    ],
)
class TestClassifiers:
    @pytest.mark.parametrize("init_strategy", ["restricted", "block_local", "normal"])
    def test_every_init_strategy_fills_the_rot_weights(
        self, cls: type, extra: dict[str, Any], init_strategy: str
    ) -> None:
        torch.manual_seed(0)
        model = cls(
            n_input_features=5, n_qubits=3, n_layers=2, init_strategy=init_strategy, **RY_, **extra
        )
        w = model.quantum_layer.qlayer.weights
        assert tuple(w.shape) == (2, 3, 3) and w.std() > 0

    def test_trains_and_round_trips_a_checkpoint(
        self, cls: type, extra: dict[str, Any], tmp_path: Path
    ) -> None:
        torch.manual_seed(0)
        model = cls(n_input_features=5, n_qubits=3, n_layers=2, **RY_, **extra)
        x = torch.randn(8, 5)
        before = model.quantum_layer.qlayer.weights.detach().clone()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        opt.zero_grad()
        model(x).pow(2).mean().backward()
        opt.step()
        assert not torch.equal(model.quantum_layer.qlayer.weights, before)
        save_checkpoint(model, tmp_path / "m.pt")
        loaded = load_checkpoint(tmp_path / "m.pt")
        assert loaded.get_config()["entangler"] == "rydberg"
        assert circuit_summary(loaded).n_blockade_layers == 2 * 2
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model.eval()(x), rtol=0, atol=0)


def test_primitive_and_encoder_record_the_same_gates() -> None:
    """The encoder's block is neutral_atom_layer on rydberg_chain, gate for gate."""
    from hqnn_forge.circuits import neutral_atom_layer
    from hqnn_forge.encoding.angle_embedding import apply_variational_layers

    w = torch.rand(2, 4, 3, generator=torch.Generator().manual_seed(7))
    with qml.tape.QuantumTape() as encoder_tape:
        apply_variational_layers(w, 4, 2, "rydberg")
    with qml.tape.QuantumTape() as primitive_tape:
        for ell in range(2):
            neutral_atom_layer(w[ell], rydberg_chain(4), RYDBERG_BLOCKADE_RADIUS)
    assert [(op.name, op.wires.tolist()) for op in encoder_tape.operations] == [
        (op.name, op.wires.tolist()) for op in primitive_tape.operations
    ]
    # Schedule order: (0,1) | (1,2) | (2,3), then Rot on every wire
    assert [op.wires.tolist() for op in encoder_tape.operations[:3]] == [[0, 1], [1, 2], [2, 3]]
