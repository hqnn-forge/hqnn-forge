"""
tests/test_rydberg_custom_register.py
=====================================
``entangler="rydberg"`` on a register other than the default chain: the
``atom_register``, ``blockade_radius`` and ``ccz`` options of every encoder,
the trunk and the three classifiers (#644).

Expected values come from somewhere other than the code under test: the
edges, triangles and schedules of the hand-placed registers are written out
by hand (they are those of ``tests/test_neutral_atom_layer.py``, worked out in
its docstrings), the circuit outputs come from a NumPy state vector built
from the diagonal phase of those gates and 2x2 ``Rot`` matrices, and the
stored positions are the literal coordinates the registers were built from.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import torch

from hqnn_forge.diagnostics import circuit_summary
from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    QuantumEncodingLayer,
)
from hqnn_forge.encoding.angle_embedding import (
    Entangler,
    apply_variational_layers,
    build_encoding_qnode,
    validate_circuit_options,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.rydberg import AtomRegister
from hqnn_forge.utils import checkpoint as ckpt
from hqnn_forge.utils import load_checkpoint, save_checkpoint

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}

SIDE = 5.0  # µm, as in tests/test_neutral_atom_layer.py
R_SIDE = 6.0  # between the square's side (5) and its diagonal (7.07)

#: The registers of tests/test_neutral_atom_layer.py, as the positions a user
#: would pass: three atoms on a line, an equilateral triangle, and a square
#: counterclockwise from the origin.
LINE_3 = [[0.0, 0.0], [SIDE, 0.0], [2 * SIDE, 0.0]]
TRIANGLE = [[0.0, 0.0], [SIDE, 0.0], [SIDE / 2, SIDE * math.sqrt(3) / 2]]
SQUARE = [[0.0, 0.0], [SIDE, 0.0], [SIDE, SIDE], [0.0, SIDE]]

#: Hand-derived at R_b = 6 µm: every side (5 µm) is an edge, the square's
#: diagonals (7.07 µm) and the line's ends (10 µm) are not.
EDGES = {
    "line": [(0, 1), (1, 2)],
    "triangle": [(0, 1), (0, 2), (1, 2)],
    "square": [(0, 1), (0, 3), (1, 2), (2, 3)],
}
TRIANGLES = {"line": [], "triangle": [(0, 1, 2)], "square": []}
POSITIONS = {"line": LINE_3, "triangle": TRIANGLE, "square": SQUARE}

ENCODERS = [QuantumEncodingLayer, IQPEncodingLayer, AmplitudeEncodingLayer, DataReuploadingLayer]
CLASSIFIERS = [
    (HybridBinaryClassifier, {}),
    (ParallelHybridClassifier, {}),
    (MulticlassHybridClassifier, {"n_classes": 3}),
]


def _random_weights(layer: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.rand(p.shape, generator=g, dtype=p.dtype) * 2 * torch.pi)


# ---------------------------------------------------------------------------
# NumPy reference: embed, then per layer the diagonal blockade phase and Rot
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


def _phase(
    n: int, pairs: Sequence[tuple[int, int]], triples: Sequence[tuple[int, ...]]
) -> np.ndarray:
    """``∏ CZ ∏ CCZ``: ``(-1)^{Σ b_i b_j + Σ b_i b_j b_k}``, wire 0 the most significant bit."""
    out = np.ones(2**n, dtype=complex)
    for index in range(2**n):
        b = [(index >> (n - 1 - wire)) & 1 for wire in range(n)]
        exponent = sum(b[i] * b[j] for i, j in pairs)
        exponent += sum(b[t[0]] * b[t[1]] * b[t[2]] for t in triples)
        out[index] = (-1) ** exponent
    return out


def _reference(
    x: np.ndarray,
    w: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    triples: Sequence[tuple[int, ...]],
    *,
    reupload: bool = False,
) -> np.ndarray:
    """``⟨Z_i⟩`` of RX(x) on |0…0⟩, then per layer the CZ/CCZ phase and Rot on every wire."""
    n = x.shape[0]
    embed = reduce(np.kron, [_rx(float(x[i])) for i in range(n)])
    phase = _phase(n, pairs, triples)
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    for ell in range(w.shape[0]):
        if ell == 0 or reupload:
            state = embed @ state
        state = reduce(np.kron, [_rot(*w[ell, i]) for i in range(n)]) @ (phase * state)
    probs = np.abs(state) ** 2
    signs = np.array(
        [[1 - 2 * ((k >> (n - 1 - wire)) & 1) for wire in range(n)] for k in range(2**n)]
    )
    return probs @ signs


def _block_gates(name: str, ccz: bool) -> list[tuple[str, list[int]]]:
    """One block as written, by hand: CZ on the edges, CCZ on the triangles, Rot on every atom."""
    gates = [("CZ", list(e)) for e in EDGES[name]]
    if ccz:
        gates += [("CCZ", list(t)) for t in TRIANGLES[name]]
    return gates + [("Rot", [i]) for i in range(len(POSITIONS[name]))]


# ---------------------------------------------------------------------------
# The circuit runs on the register it was given
# ---------------------------------------------------------------------------


class TestCircuit:
    @pytest.mark.parametrize(
        ("name", "ccz"), [("triangle", True), ("triangle", False), ("square", False)]
    )
    def test_angle_layer_matches_numpy(self, name: str, ccz: bool) -> None:
        n, n_layers = len(POSITIONS[name]), 2
        layer = QuantumEncodingLayer(
            n_qubits=n,
            n_layers=n_layers,
            entangler="rydberg",
            atom_register=POSITIONS[name],
            blockade_radius=R_SIDE,
            ccz=ccz,
            **CPU,
        ).double()
        _random_weights(layer, 0)
        w = layer.qlayer.weights.detach().numpy()
        x = torch.rand(3, n, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        triples = TRIANGLES[name] if ccz else []
        expected = np.stack([_reference(xi, w, EDGES[name], triples) for xi in x.numpy()])
        with torch.no_grad():
            out = layer(x).numpy()
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)

    def test_ccz_changes_the_triangle_output(self) -> None:
        """Not vacuous: on the triangle the CCZ is a different circuit from the CZs alone."""
        x = torch.rand(4, 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
        outs = []
        for ccz in (False, True):
            layer = QuantumEncodingLayer(
                n_qubits=3,
                n_layers=1,
                entangler="rydberg",
                atom_register=TRIANGLE,
                blockade_radius=R_SIDE,
                ccz=ccz,
                **CPU,
            ).double()
            _random_weights(layer, 3)
            with torch.no_grad():
                outs.append(layer(x))
        assert (outs[0] - outs[1]).abs().max() > 1e-3

    def test_reuploading_runs_every_block_on_the_register(self) -> None:
        layer = DataReuploadingLayer(
            n_qubits=3,
            n_layers=3,
            entangler="rydberg",
            atom_register=TRIANGLE,
            blockade_radius=R_SIDE,
            ccz=True,
            **CPU,
        ).double()
        _random_weights(layer, 4)
        w = layer.qlayer.weights.detach().numpy()
        x = torch.rand(3, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        expected = np.stack(
            [
                _reference(xi, w, EDGES["triangle"], TRIANGLES["triangle"], reupload=True)
                for xi in x.numpy()
            ]
        )
        with torch.no_grad():
            out = layer(x).numpy()
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("cls", ENCODERS)
    @pytest.mark.parametrize(("name", "ccz"), [("triangle", True), ("square", False)])
    def test_every_encoder_records_the_hand_written_blocks(
        self, cls: type, name: str, ccz: bool
    ) -> None:
        """After the embedding, each of the 2 blocks is CZ on the edges, CCZ on the triangles, Rot."""
        n = len(POSITIONS[name])
        layer = cls(
            n_qubits=n,
            n_layers=2,
            entangler="rydberg",
            atom_register=AtomRegister(POSITIONS[name]),
            blockade_radius=R_SIDE,
            ccz=ccz,
            **CPU,
        )
        x = layer.prepare_inputs(torch.full((1, layer.n_features), 0.3))[0]
        tape = qml.workflow.construct_tape(layer.qlayer.qnode, level="top")(
            x, **dict(layer.qlayer.qnode_weights.items())
        )
        recorded = [
            (op.name, op.wires.tolist())
            for op in tape.operations
            if op.name in ("CZ", "CCZ", "Rot")
        ]
        assert recorded == _block_gates(name, ccz) * 2

    def test_a_register_given_as_positions_or_as_an_atom_register_is_the_same(self) -> None:
        w = torch.rand(1, 4, 3, generator=torch.Generator().manual_seed(6))
        tapes = []
        for register in (SQUARE, AtomRegister(SQUARE), torch.tensor(SQUARE)):
            with qml.tape.QuantumTape() as tape:
                apply_variational_layers(
                    w, 4, 1, "rydberg", atom_register=register, blockade_radius=R_SIDE
                )
            tapes.append([(op.name, op.wires.tolist()) for op in tape.operations])
        assert tapes[0] == tapes[1] == tapes[2] == _block_gates("square", False)

    def test_none_is_the_default_chain(self) -> None:
        """``atom_register=None`` and ``blockade_radius=None`` are rydberg_chain at 8.69 µm."""
        x = torch.rand(2, 4, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
        default = QuantumEncodingLayer(n_qubits=4, n_layers=2, entangler="rydberg", **CPU).double()
        explicit = QuantumEncodingLayer(
            n_qubits=4,
            n_layers=2,
            entangler="rydberg",
            atom_register=[[0.0, 0.0], [6.0, 0.0], [12.0, 0.0], [18.0, 0.0]],
            blockade_radius=(2 * math.pi * 862690 / (4 * math.pi)) ** (1 / 6),
            **CPU,
        ).double()
        _random_weights(default, 8)
        explicit.load_state_dict(default.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(explicit(x), default(x), rtol=0, atol=0)
        assert default.atom_register is None and default.blockade_radius is None

    def test_repr_names_the_options(self) -> None:
        layer = QuantumEncodingLayer(
            n_qubits=3,
            n_layers=1,
            entangler="rydberg",
            atom_register=TRIANGLE,
            blockade_radius=R_SIDE,
            ccz=True,
            **CPU,
        )
        text = repr(layer)
        assert f"atom_register={AtomRegister(TRIANGLE).positions.tolist()}" in text
        assert "blockade_radius=6.0" in text and "ccz=True" in text
        assert "atom_register" not in repr(
            QuantumEncodingLayer(n_qubits=3, n_layers=1, entangler="rydberg", **CPU)
        )


# ---------------------------------------------------------------------------
# circuit_summary counts the layer's own register
# ---------------------------------------------------------------------------


class TestBlockadeLayers:
    @pytest.mark.parametrize(
        ("positions", "radius", "ccz", "per_block"),
        [
            # 0-1 and 1-2 share atom 1: one CZ per layer
            pytest.param(LINE_3, R_SIDE, False, 2, id="line"),
            # Every pair of the triangle's edges shares an atom: 3 layers
            pytest.param(TRIANGLE, R_SIDE, False, 3, id="triangle"),
            # The CCZ uses all three atoms, so it follows alone: 4 layers
            pytest.param(TRIANGLE, R_SIDE, True, 4, id="triangle-ccz"),
            # Opposite sides are disjoint but 5 µm < R_b apart: 4 layers
            pytest.param(SQUARE, R_SIDE, False, 4, id="square"),
            # No triangle at 6 µm, so ccz adds nothing
            pytest.param(SQUARE, R_SIDE, True, 4, id="square-ccz"),
            # At 8 µm every pair is inside R_b (diagonal 7.07), so every gate runs
            # alone: 6 edges + 4 triangles
            pytest.param(SQUARE, 8.0, True, 10, id="square-8um-ccz"),
        ],
    )
    @pytest.mark.parametrize("n_layers", [1, 2])
    def test_hand_counted_schedules(
        self,
        positions: list[list[float]],
        radius: float,
        ccz: bool,
        per_block: int,
        n_layers: int,
    ) -> None:
        layer = QuantumEncodingLayer(
            n_qubits=len(positions),
            n_layers=n_layers,
            entangler="rydberg",
            atom_register=positions,
            blockade_radius=radius,
            ccz=ccz,
            **CPU,
        )
        summary = circuit_summary(layer)
        assert summary.n_blockade_layers == per_block * n_layers

    @pytest.mark.parametrize(("cls", "extra"), CLASSIFIERS)
    def test_a_classifier_reports_its_register(self, cls: type, extra: dict[str, Any]) -> None:
        model = cls(
            n_input_features=5,
            n_qubits=3,
            n_layers=2,
            entangler="rydberg",
            atom_register=TRIANGLE,
            blockade_radius=R_SIDE,
            ccz=True,
            **CPU,
            **extra,
        )
        assert circuit_summary(model).n_blockade_layers == 4 * 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("cls", ENCODERS)
    def test_the_register_needs_one_atom_per_qubit(self, cls: type) -> None:
        with pytest.raises(ValueError, match=r"atom_register has 3 atoms.*n_qubits=4"):
            cls(n_qubits=4, n_layers=1, entangler="rydberg", atom_register=TRIANGLE, **CPU)

    @pytest.mark.parametrize("cls", ENCODERS)
    def test_a_disconnected_graph_is_refused_at_construction(self, cls: type) -> None:
        # The line's neighbours are exactly 5 µm apart: at R_b = 5 no pair interacts
        with pytest.raises(
            ValueError, match=r"does not connect every atom.*isolated: \[0, 1, 2\]"
        ):
            cls(
                n_qubits=3,
                n_layers=1,
                entangler="rydberg",
                atom_register=LINE_3,
                blockade_radius=SIDE,
                **CPU,
            )

    def test_a_radius_below_the_default_spacing_disconnects_the_default_chain(self) -> None:
        with pytest.raises(ValueError, match=r"does not connect every atom"):
            QuantumEncodingLayer(
                n_qubits=3, n_layers=1, entangler="rydberg", blockade_radius=5.9, **CPU
            )

    @pytest.mark.parametrize("radius", [0.0, -1.0, math.nan, math.inf, "6", True])
    def test_a_bad_radius_is_refused(self, radius: object) -> None:
        with pytest.raises(ValueError, match=r"blockade_radius must be a finite number > 0"):
            QuantumEncodingLayer(
                n_qubits=3,
                n_layers=1,
                entangler="rydberg",
                blockade_radius=radius,  # type: ignore[arg-type]
                **CPU,
            )

    @pytest.mark.parametrize("ccz", [1, "yes", None])
    def test_ccz_must_be_a_bool(self, ccz: object) -> None:
        with pytest.raises(TypeError, match=r"ccz must be True or False"):
            QuantumEncodingLayer(
                n_qubits=3,
                n_layers=1,
                entangler="rydberg",
                ccz=ccz,  # type: ignore[arg-type]
                **CPU,
            )

    def test_bad_positions_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"positions must not repeat"):
            QuantumEncodingLayer(
                n_qubits=2,
                n_layers=1,
                entangler="rydberg",
                atom_register=[[0.0, 0.0], [0.0, 0.0]],
                **CPU,
            )

    @pytest.mark.parametrize(
        "entangler", ["ring", "strongly_entangling", "brickwork", "hardware_efficient"]
    )
    @pytest.mark.parametrize(
        ("options", "named"),
        [
            ({"atom_register": SQUARE}, "atom_register applies"),
            ({"blockade_radius": R_SIDE}, "blockade_radius applies"),
            ({"ccz": True}, "ccz applies"),
            ({"atom_register": SQUARE, "ccz": True}, "atom_register, ccz apply"),
        ],
    )
    def test_any_other_entangler_refuses_the_options(
        self, entangler: Entangler, options: dict[str, Any], named: str
    ) -> None:
        pattern = rf"{named} to entangler='rydberg' only; got entangler='{entangler}'"
        with pytest.raises(ValueError, match=pattern):
            validate_circuit_options(4, entangler, "all", **options)
        with pytest.raises(ValueError, match=pattern):
            build_encoding_qnode(n_qubits=4, n_layers=1, entangler=entangler, **options, **CPU)
        with pytest.raises(ValueError, match=pattern):
            QuantumEncodingLayer(n_qubits=4, n_layers=1, entangler=entangler, **options, **CPU)
        with pytest.raises(ValueError, match=pattern), qml.tape.QuantumTape():
            apply_variational_layers(torch.zeros(1, 4, 3), 4, 1, entangler, **options)

    @pytest.mark.parametrize(("cls", "extra"), CLASSIFIERS)
    def test_classifiers_refuse_the_same(self, cls: type, extra: dict[str, Any]) -> None:
        common = {"n_input_features": 5, "n_qubits": 3, "n_layers": 1, **CPU, **extra}
        with pytest.raises(ValueError, match=r"atom_register applies to entangler='rydberg'"):
            cls(atom_register=TRIANGLE, **common)
        with pytest.raises(ValueError, match=r"atom_register has 4 atoms.*n_qubits=3"):
            cls(entangler="rydberg", atom_register=SQUARE, **common)
        with pytest.raises(ValueError, match=r"does not connect every atom"):
            cls(entangler="rydberg", atom_register=LINE_3, blockade_radius=SIDE, **common)


# ---------------------------------------------------------------------------
# get_config and checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "extra"), CLASSIFIERS)
class TestConfigAndCheckpoint:
    @staticmethod
    def _model(model_cls: type, extra: dict[str, Any], **options: Any) -> Any:
        torch.manual_seed(0)
        return model_cls(
            n_input_features=5,
            n_qubits=4,
            n_layers=2,
            entangler="rydberg",
            **options,
            **CPU,
            **extra,
        )

    def test_config_holds_the_positions_as_lists(self, cls: type, extra: dict[str, Any]) -> None:
        model = self._model(
            cls, extra, atom_register=AtomRegister(SQUARE), blockade_radius=6, ccz=True
        )
        config = model.get_config()
        assert config["atom_register"] == [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]]
        assert type(config["atom_register"]) is list
        assert all(type(v) is float for row in config["atom_register"] for v in row)
        assert config["blockade_radius"] == 6.0 and type(config["blockade_radius"]) is float
        assert config["ccz"] is True
        # A fresh copy: editing it leaves the model's config as it was
        config["atom_register"][0][0] = 99.0
        assert model.get_config()["atom_register"][0][0] == 0.0

    def test_the_default_config_records_none(self, cls: type, extra: dict[str, Any]) -> None:
        config = self._model(cls, extra).get_config()
        assert config["atom_register"] is None
        assert config["blockade_radius"] is None
        assert config["ccz"] is False

    def test_rebuilds_from_its_config(self, cls: type, extra: dict[str, Any]) -> None:
        model = self._model(cls, extra, atom_register=SQUARE, blockade_radius=R_SIDE)
        rebuilt = cls(**model.get_config())
        rebuilt.load_state_dict(model.state_dict())
        x = torch.randn(3, 5)
        model.eval()
        rebuilt.eval()
        with torch.no_grad():
            torch.testing.assert_close(rebuilt(x), model(x), rtol=0, atol=0)
        assert rebuilt.get_config() == model.get_config()

    def test_checkpoint_round_trip_with_weights_only(
        self, cls: type, extra: dict[str, Any], tmp_path: Path
    ) -> None:
        model = self._model(
            cls, extra, atom_register=TRIANGLE + [[SIDE, -SIDE]], blockade_radius=R_SIDE
        )
        path = tmp_path / "m.pt"
        save_checkpoint(model, path)
        payload = torch.load(path, weights_only=True)
        assert (
            payload["config"]["atom_register"]
            == AtomRegister(TRIANGLE + [[SIDE, -SIDE]]).positions.tolist()
        )
        loaded = load_checkpoint(path)
        layer = loaded.quantum_layer
        assert isinstance(layer, QuantumEncodingLayer) and layer.atom_register is not None
        torch.testing.assert_close(
            layer.atom_register.positions,
            torch.tensor(TRIANGLE + [[SIDE, -SIDE]], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        x = torch.randn(3, 5)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)
        assert (
            circuit_summary(loaded).n_blockade_layers == circuit_summary(model).n_blockade_layers
        )

    def test_a_rydberg_checkpoint_from_before_the_options_loads_as_the_chain(
        self, cls: type, extra: dict[str, Any], tmp_path: Path
    ) -> None:
        """
        A file written by #418's tree has entangler="rydberg" but none of the three
        arguments, in its config or its known_args.  _LEGACY_DEFAULTS fills them
        with the default chain, which is what that tree ran.
        """
        model = self._model(cls, extra)
        path = tmp_path / "new.pt"
        save_checkpoint(model, path)
        payload = torch.load(path, weights_only=True)
        for name in ("atom_register", "blockade_radius", "ccz"):
            del payload["config"][name]
            payload["known_args"].remove(name)
        torch.save(payload, tmp_path / "old.pt")

        with pytest.warns(RuntimeWarning, match="predates") as record:
            loaded = load_checkpoint(tmp_path / "old.pt")
        message = str(record[0].message)
        assert "atom_register" in message and "blockade_radius" in message and "ccz" in message
        assert loaded.get_config() == model.get_config()
        x = torch.randn(3, 5)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)
        # min(3, n − 1) = 3 blockade layers per block on the 4-atom chain
        assert circuit_summary(loaded).n_blockade_layers == 3 * 2


def test_the_legacy_defaults_are_the_default_chain() -> None:
    assert ckpt._LEGACY_DEFAULTS["atom_register"] is None
    assert ckpt._LEGACY_DEFAULTS["blockade_radius"] is None
    assert ckpt._LEGACY_DEFAULTS["ccz"] is False
    for name in ("atom_register", "blockade_radius", "ccz"):
        assert name not in ckpt.WEIGHT_SAFE_ARGS


# ---------------------------------------------------------------------------
# Inert parameters follow the graph distance from atom 0
# ---------------------------------------------------------------------------

#: Graph distances from atom 0 at R_b = 6 µm, by hand.  Square: the opposite
#: corner 2 is two sides away.  Kite: the square plus atom 4 at (10, 0), a
#: neighbour of atom 1 only.  Wheel: the square plus atom 4 at its centre,
#: 3.54 µm from every corner.
DISTANCES = {
    "square": (SQUARE, [0, 1, 2, 1]),
    "kite": ([*SQUARE, [10.0, 0.0]], [0, 1, 2, 1, 2]),
    "wheel": ([*SQUARE, [2.5, 2.5]], [0, 1, 2, 1, 1]),
}


@pytest.mark.parametrize("name", list(DISTANCES))
@pytest.mark.parametrize("readout", ["all", "first"])
@pytest.mark.parametrize("ccz", [False, True])
def test_inert_count_follows_the_graph_distance(name: str, readout: str, ccz: bool) -> None:
    """
    Under "all" only the n last-layer ω are dead.  Under "first", block k counted
    back from the readout leaves dead the ω of each atom at distance k from atom 0
    and all three angles of each atom farther away (step 3 of
    _make_angle_embedding_circuit): square 10 + 5 + 1, kite 13 + 8 + 2, wheel
    13 + 6 + 1 at 3 layers.  Autograd in float64 agrees with circuit_summary.
    """
    positions, distance = DISTANCES[name]
    n, n_layers = len(positions), 3
    layer = QuantumEncodingLayer(
        n_qubits=n,
        n_layers=n_layers,
        entangler="rydberg",
        readout=readout,  # type: ignore[arg-type]
        atom_register=positions,
        blockade_radius=R_SIDE,
        ccz=ccz,
        **CPU,
    ).double()
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
        expected = sum(
            sum(1 if d == k else 3 if d > k else 0 for d in distance) for k in range(n_layers)
        )
    assert dead == expected
    assert circuit_summary(layer).n_inert_params == expected
