"""
tests/test_cz_noise_position.py
===============================
``position="cz"``: a channel after every CZ and CCZ only (#700), for gate-dependent
noise such as atom loss during the Rydberg blockade gates (#419) of
``entangler="rydberg"`` (#418), including the CCZ of a custom register (#644).

Every expected value is obtained another way than through the code under test:
the sites of a hand-placed register are listed by hand from its blockade
schedule, the density result is compared with a density matrix built by hand in
NumPy from Kraus operators written out here, and the trajectory mean with the
density result within a stated number of standard errors.

Index conventions, as in PennyLane and ``hqnn_forge.noise``: wire ``i`` is atom
``i`` of the register and bit ``i`` of a basis index counted from the most
significant one (wire 0 leftmost in every Kronecker product), ``|0⟩`` has
``Z = +1``.  A *site* is one (gate, wire) pair, in tape order and, within a
gate, in the gate's own wire order.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.noise import (
    Channel,
    apply_depolarizing_noise,
    training_noise_qnode,
    trajectory_noise_qnode,
    validate_noise,
)
from hqnn_forge.rydberg import AtomRegister

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
#: Monte Carlo tolerance, in standard errors of the mean (fixed seed, so the
#: check is deterministic; 5 SE fails by chance with probability ≈ 6e-7).
Z_SE = 5.0

SIDE = 5.0  # µm
#: An equilateral triangle of side 5 µm, as in tests/test_neutral_atom_layer.py.
#: At R_b = 6 µm every pair interacts and (0, 1, 2) is a blockade triangle, so
#: the schedule with ccz=True is [(0, 1)], [(0, 2)], [(1, 2)], [(0, 1, 2)]
#: (that file's ``test_triangle_with_ccz``).
TRIANGLE = AtomRegister([[0.0, 0.0], [SIDE, 0.0], [SIDE / 2, SIDE * math.sqrt(3) / 2]])
R_B = 6.0
#: The blockade gates of one block on TRIANGLE with ccz=True, in schedule order.
TRIANGLE_GATES: list[tuple[int, ...]] = [(0, 1), (0, 2), (1, 2), (0, 1, 2)]

X_IN = (0.7, -1.2, 2.1)
#: Rot angles (φ, θ, ω) per atom, one block.
ROT = ((0.3, 1.1, -0.4), (-0.8, 0.5, 0.9), (1.4, -0.6, 0.2))

CHANNEL_NAMES: list[Channel] = [
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "bit_flip",
    "phase_flip",
    "atom_loss",
]


def _triangle_layer(**kwargs: Any) -> QuantumEncodingLayer:
    """The angle encoder on TRIANGLE with CCZ: RX(x_i), the four gates, Rot per atom."""
    return QuantumEncodingLayer(
        n_qubits=3,
        n_layers=1,
        entangler="rydberg",
        atom_register=TRIANGLE,
        blockade_radius=R_B,
        ccz=True,
        **CPU,
        **kwargs,
    )


def _set_rot(layer: QuantumEncodingLayer) -> None:
    with torch.no_grad():
        layer.qlayer.weights.copy_(torch.tensor([ROT], dtype=torch.float64))


# ---------------------------------------------------------------------------
# Hand-built density matrices (NumPy, independent of PennyLane's channels)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)


def _kraus(channel: str, p: float) -> list[np.ndarray]:
    """Each channel's Kraus operators at strength ``p``, written out by hand."""
    if channel == "depolarizing":
        return [math.sqrt(1 - p) * _I] + [math.sqrt(p / 3) * s for s in (_X, _Y, _Z)]
    if channel == "amplitude_damping":
        return [
            np.array([[1, 0], [0, math.sqrt(1 - p)]], dtype=complex),
            np.array([[0, math.sqrt(p)], [0, 0]], dtype=complex),
        ]
    if channel == "phase_damping":
        return [
            np.array([[1, 0], [0, math.sqrt(1 - p)]], dtype=complex),
            np.array([[0, 0], [0, math.sqrt(p)]], dtype=complex),
        ]
    if channel == "bit_flip":
        return [math.sqrt(1 - p) * _I, math.sqrt(p) * _X]
    if channel == "phase_flip":
        return [math.sqrt(1 - p) * _I, math.sqrt(p) * _Z]
    if channel == "atom_loss":
        # Reset to the dark |1⟩: sqrt(1 − p) I, sqrt(p)|1⟩⟨0|, sqrt(p)|1⟩⟨1|.
        return [
            math.sqrt(1 - p) * _I,
            np.array([[0, 0], [math.sqrt(p), 0]], dtype=complex),
            np.array([[0, 0], [0, math.sqrt(p)]], dtype=complex),
        ]
    raise AssertionError(channel)


def _on(op: np.ndarray, wire: int, n: int = 3) -> np.ndarray:
    """A one-qubit ``op`` on ``wire`` of ``n`` (wire 0 most significant)."""
    return functools.reduce(np.kron, [op if w == wire else _I for w in range(n)])


def _cz_like(wires: tuple[int, ...], n: int = 3) -> np.ndarray:
    """CZ (two wires) or CCZ (three): −1 on the basis states with every wire at 1."""
    diag = np.ones(2**n, dtype=complex)
    for index in range(2**n):
        bits = [(index >> (n - 1 - w)) & 1 for w in range(n)]
        if all(bits[w] for w in wires):
            diag[index] = -1.0
    return np.diag(diag)


def _rx(t: float) -> np.ndarray:
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _rot(phi: float, theta: float, omega: float) -> np.ndarray:
    """Rot(φ, θ, ω) = RZ(ω) · RY(θ) · RZ(φ)."""

    def rz(t: float) -> np.ndarray:
        return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])

    c, s = math.cos(theta / 2), math.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    return rz(omega) @ ry @ rz(phi)


def _hand_built(channel: str, p: float, position: str) -> np.ndarray:
    """
    ⟨Z_0⟩, ⟨Z_1⟩, ⟨Z_2⟩ of :func:`_triangle_layer` with ``channel`` at ``position``.

    ``"cz"``: after each of the four blockade gates, on each of its wires.
    ``"all"`` and ``"end"`` are built too, to show the test can tell them apart.
    """
    kraus = _kraus(channel, p)

    def noise(rho: np.ndarray, wire: int) -> np.ndarray:
        return sum(_on(k, wire) @ rho @ _on(k, wire).conj().T for k in kraus)  # type: ignore[return-value]

    rho = np.zeros((8, 8), dtype=complex)
    rho[0, 0] = 1.0
    gates: list[tuple[np.ndarray, tuple[int, ...], bool]] = [
        (_on(_rx(x), w), (w,), False) for w, x in enumerate(X_IN)
    ]
    gates += [(_cz_like(g), g, True) for g in TRIANGLE_GATES]
    gates += [(_on(_rot(*ROT[w]), w), (w,), False) for w in range(3)]
    for unitary, wires, blockade in gates:
        rho = unitary @ rho @ unitary.conj().T
        if position == "all" or (position == "cz" and blockade):
            for w in wires:
                rho = noise(rho, w)
    if position == "end":
        for w in range(3):
            rho = noise(rho, w)
    return np.array([np.trace(rho @ _on(_Z, w)).real for w in range(3)])


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


def _sites(
    tape: qml.tape.QuantumScript, inserted: Callable[[qml.operation.Operator], bool]
) -> list[tuple[str, tuple[int, ...], int]]:
    """``(gate name, gate wires, noise wire)`` for every inserted op, in tape order."""
    sites = []
    last: qml.operation.Operator | None = None
    for op in tape.operations:
        if inserted(op):
            assert last is not None, "a channel before every gate"
            sites.append((last.name, tuple(last.wires.tolist()), int(op.wires[0])))
        else:
            last = op
    return sites


#: The sites of one TRIANGLE block with ccz=True, listed by hand.
TRIANGLE_SITES = [
    ("CZ", (0, 1), 0),
    ("CZ", (0, 1), 1),
    ("CZ", (0, 2), 0),
    ("CZ", (0, 2), 2),
    ("CZ", (1, 2), 1),
    ("CZ", (1, 2), 2),
    ("CCZ", (0, 1, 2), 0),
    ("CCZ", (0, 1, 2), 1),
    ("CCZ", (0, 1, 2), 2),
]


def _tape(qnode: Any, layer: QuantumEncodingLayer) -> qml.tape.QuantumScript:
    x = torch.tensor(X_IN, dtype=torch.float64)
    weights = layer.qlayer.weights.detach().double()
    return qml.workflow.construct_tape(qnode, level="user")(x, weights=weights)


class TestSites:
    def test_density_sites_on_a_hand_placed_register(self) -> None:
        layer = _triangle_layer()
        noisy = training_noise_qnode(layer.qlayer.qnode, 3, 0.1, "cz", channel="atom_loss")
        tape = _tape(noisy, layer)
        sites = _sites(tape, lambda op: isinstance(op, qml.ResetError))
        assert sites == TRIANGLE_SITES
        # The channel itself: ResetError(0, p) on that wire, nothing else added.
        resets = [op for op in tape.operations if isinstance(op, qml.ResetError)]
        assert all([float(v) for v in op.parameters] == [0.0, 0.1] for op in resets)
        assert len(tape.operations) == 3 + 4 + 3 + len(TRIANGLE_SITES)

    def test_every_block_of_a_stack_gets_its_sites(self) -> None:
        layer = QuantumEncodingLayer(
            n_qubits=3,
            n_layers=2,
            entangler="rydberg",
            atom_register=TRIANGLE,
            blockade_radius=R_B,
            ccz=True,
            **CPU,
        )
        noisy = training_noise_qnode(layer.qlayer.qnode, 3, 0.1, "cz", channel="depolarizing")
        tape = _tape(noisy, layer)
        sites = _sites(tape, lambda op: isinstance(op, qml.DepolarizingChannel))
        assert sites == TRIANGLE_SITES * 2

    def test_kraus_trajectories_put_their_branches_on_the_same_sites(self) -> None:
        layer = _triangle_layer()
        noisy = trajectory_noise_qnode(layer.qlayer.qnode, 0.1, "cz", channel="atom_loss")
        tape = _tape(noisy, layer)
        assert _sites(tape, lambda op: isinstance(op, qml.QubitUnitary)) == TRIANGLE_SITES

    def test_pauli_trajectories_put_their_errors_on_the_same_sites(self) -> None:
        # Each Pauli site is RZ(π·z) then RX(π·x).  The circuit's own RX are the
        # embedding, before the first CZ, and its rotations are Rot, so every
        # RZ / RX after a blockade gate is inserted.
        layer = _triangle_layer()
        noisy = trajectory_noise_qnode(layer.qlayer.qnode, 0.1, "cz", channel="depolarizing")
        tape = _tape(noisy, layer)
        ops = tape.operations
        first_cz = next(i for i, op in enumerate(ops) if isinstance(op, qml.CZ))
        pairs = []
        for i, op in enumerate(ops[first_cz:], start=first_cz):
            if isinstance(op, qml.RZ):
                assert isinstance(ops[i + 1], qml.RX) and ops[i + 1].wires == op.wires
                pairs.append(i)
        rx_count = sum(isinstance(op, qml.RX) for op in ops[first_cz:])
        assert rx_count == len(pairs)
        marker = set(pairs) | {i + 1 for i in pairs}
        sites = []
        last = None
        for i, op in enumerate(ops):
            if i in pairs:
                assert last is not None
                sites.append((last.name, tuple(last.wires.tolist()), int(op.wires[0])))
            elif i not in marker:
                last = op
        assert sites == TRIANGLE_SITES

    def test_hardware_efficient_cz_are_sites_too(self) -> None:
        # The tape cannot tell the rydberg entangler's CZ from the CZ ladder of
        # "hardware_efficient": CZ(0, 1), CZ(1, 2) per block, 2 blocks.
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=2, entangler="hardware_efficient", **CPU)
        noisy = training_noise_qnode(layer.qlayer.qnode, 3, 0.1, "cz", channel="bit_flip")
        tape = _tape(noisy, layer)
        sites = _sites(tape, lambda op: isinstance(op, qml.BitFlip))
        block = [("CZ", (0, 1), 0), ("CZ", (0, 1), 1), ("CZ", (1, 2), 1), ("CZ", (1, 2), 2)]
        assert sites == block * 2

    def test_gates_inside_a_template_are_found(self) -> None:
        # qml.noise.insert decomposes templates before placing channels; "cz"
        # sees the same decomposed tape, so a CZ inside a template is a site.
        device = qml.device("default.qubit", wires=2)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(theta: torch.Tensor) -> Any:
            qml.RY(theta, wires=0)
            qml.RY(theta, wires=1)
            qml.BasicEntanglerLayers(torch.zeros(1, 2), wires=[0, 1], rotation=qml.RX)
            qml.Hadamard(wires=1)
            qml.CZ(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        noisy = training_noise_qnode(circuit, 2, 0.2, "cz", channel="phase_flip")
        tape = qml.workflow.construct_tape(noisy, level="user")(torch.tensor(0.4))
        sites = _sites(tape, lambda op: isinstance(op, qml.PhaseFlip))
        # BasicEntanglerLayers on two wires: RX, RX, CNOT -- no CZ.
        assert sites == [("CZ", (0, 1), 0), ("CZ", (0, 1), 1)]


# ---------------------------------------------------------------------------
# The density channel
# ---------------------------------------------------------------------------


class TestDensity:
    @pytest.mark.parametrize("channel", CHANNEL_NAMES)
    def test_matches_a_hand_built_density_matrix(self, channel: Channel) -> None:
        p = 0.3
        layer = _triangle_layer()
        _set_rot(layer)
        noisy = training_noise_qnode(layer.qlayer.qnode, 3, p, "cz", channel=channel)
        x = torch.tensor(X_IN, dtype=torch.float64)
        got = torch.stack(noisy(x, weights=torch.tensor([ROT], dtype=torch.float64)))
        expected = _hand_built(channel, p, "cz")
        np.testing.assert_allclose(got.numpy(), expected, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("channel", CHANNEL_NAMES)
    def test_the_hand_built_reference_tells_the_positions_apart(self, channel: Channel) -> None:
        # Without this, a "cz" that put channels at "all" or "end" could pass.
        # The smallest gap is phase_flip against "all" (0.007): a Z commutes
        # with CZ and CCZ, so only the flips after the embedding's RX differ.
        # Any gap far above the 1e-12 of the comparison above will do.
        cz = _hand_built(channel, 0.3, "cz")
        for other in ("all", "end"):
            assert np.abs(cz - _hand_built(channel, 0.3, other)).max() > 1e-3

    def test_hand_built_noiseless_limit_is_the_layer(self) -> None:
        # p = 0 in the reference is the clean circuit, which checks the
        # reference's gates, order and wire convention on their own.
        layer = _triangle_layer()
        _set_rot(layer)
        with torch.no_grad():
            got = layer.qlayer.qnode(
                torch.tensor(X_IN, dtype=torch.float64),
                weights=torch.tensor([ROT], dtype=torch.float64),
            )
        np.testing.assert_allclose(
            torch.stack(got).numpy(), _hand_built("atom_loss", 0.0, "cz"), atol=1e-12, rtol=0
        )

    def test_post_hoc_and_training_noise_agree_with_the_reference(self) -> None:
        p = 0.2
        x = torch.tensor([X_IN], dtype=torch.float32)
        expected = torch.tensor(_hand_built("atom_loss", p, "cz"), dtype=torch.float32)
        layer = _triangle_layer()
        _set_rot(layer)
        layer.eval()
        with (
            torch.no_grad(),
            apply_depolarizing_noise(layer, p, position="cz", channel="atom_loss"),
        ):
            post_hoc = layer(x)[0]
        torch.testing.assert_close(post_hoc, expected, atol=1e-5, rtol=0)
        trained = _triangle_layer(noise_level=p, noise_position="cz", noise_channel="atom_loss")
        _set_rot(trained)
        with torch.no_grad():
            torch.testing.assert_close(trained.train()(x)[0], expected, atol=1e-5, rtol=0)
            clean = _hand_built("atom_loss", 0.0, "cz")
            torch.testing.assert_close(
                trained.eval()(x)[0], torch.tensor(clean, dtype=torch.float32), atol=1e-5, rtol=0
            )

    def test_atom_loss_at_one_on_the_last_gate_reads_dark_before_the_rot(self) -> None:
        # p = 1: after the CCZ, every atom is |1⟩, whatever came before; only
        # the final Rot acts on it, so ⟨Z_w⟩ = ⟨1|Rot† Z Rot|1⟩ = −cos θ_w.
        layer = _triangle_layer()
        noisy = training_noise_qnode(layer.qlayer.qnode, 3, 1.0, "cz", channel="atom_loss")
        got = torch.stack(
            noisy(
                torch.tensor(X_IN, dtype=torch.float64),
                weights=torch.tensor([ROT], dtype=torch.float64),
            )
        )
        expected = torch.tensor([-math.cos(ROT[w][1]) for w in range(3)], dtype=torch.float64)
        torch.testing.assert_close(got, expected, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------


class TestTrajectories:
    @pytest.mark.parametrize("channel", CHANNEL_NAMES)
    def test_mean_matches_the_density_channel(self, channel: Channel) -> None:
        p, draws = 0.3, 4000
        layer = _triangle_layer(
            noise_level=p, noise_position="cz", noise_channel=channel, noise_method="trajectories"
        )
        _set_rot(layer)
        x = torch.tensor([X_IN], dtype=torch.float32)
        exact = torch.tensor(_hand_built(channel, p, "cz"), dtype=torch.float64)
        with torch.no_grad():
            layer.train()
            torch.manual_seed(3)
            runs = layer(x.repeat(draws, 1)).double()
        se = runs.std(0) / math.sqrt(draws)
        assert (se > 0).all(), "a draw that never varies would make this test vacuous"
        assert ((runs.mean(0) - exact).abs() <= Z_SE * se + 1e-6).all()


# ---------------------------------------------------------------------------
# Circuits without CZ or CCZ, and validation
# ---------------------------------------------------------------------------


class TestWithoutBlockadeGates:
    def test_post_hoc_refuses_a_circuit_without_cz(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, entangler="ring", **CPU)
        with (
            pytest.raises(ValueError, match="no CZ or CCZ"),
            apply_depolarizing_noise(layer, 0.1, position="cz"),
        ):
            layer(torch.zeros(1, 3))

    @pytest.mark.parametrize("method", ["density", "trajectories"])
    @pytest.mark.parametrize("channel", ["depolarizing", "atom_loss"])
    def test_training_noise_refuses_a_circuit_without_cz(self, method: str, channel: str) -> None:
        layer = QuantumEncodingLayer(
            n_qubits=3,
            n_layers=1,
            entangler="brickwork",
            noise_level=0.1,
            noise_position="cz",
            noise_method=method,  # type: ignore[arg-type]
            noise_channel=channel,  # type: ignore[arg-type]
            **CPU,
        )
        with pytest.raises(ValueError, match="no CZ or CCZ"):
            layer.train()(torch.zeros(2, 3))
        # Eval mode runs the noiseless circuit, which has no sites to miss.
        assert layer.eval()(torch.zeros(2, 3)).shape == (2, 3)


class TestValidation:
    def test_cz_is_accepted(self) -> None:
        validate_noise(0.1, "cz", channel="atom_loss")

    def test_unknown_position_lists_the_choices(self) -> None:
        with pytest.raises(ValueError, match=r"position must be one of 'all', 'end', 'cz'"):
            validate_noise(0.1, "entangling")
