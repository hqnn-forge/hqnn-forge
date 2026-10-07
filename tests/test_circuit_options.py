"""
tests/test_circuit_options.py
=============================
The circuit options added for the published SHNN configuration (#131):
``entangler``, ``readout`` and ``rotation`` on the encoding layers, and
``embedding_rotation``, ``entangler``, ``readout``, ``encoder_activation``
and ``init_strategy="normal"`` on the classifiers.

The ``strongly_entangling`` block is pinned against a hand-written
Rot-then-CNOT circuit with the template's range rule ``r = ℓ mod (n-1) + 1``,
so the option is checked against the documented topology, not just against
the template it wraps.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, TypedDict, TypeVar

import pennylane as qml
import pytest
import torch
import torch.nn as nn

from hqnn_forge.encoding import DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.angle_embedding import (
    DeviceName,
    DiffMethod,
    apply_variational_layers,
    build_encoding_qnode,
    measure_z,
    readout_wires,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier


class _Backend(TypedDict):
    device_name: DeviceName
    diff_method: DiffMethod


CPU: _Backend = {"device_name": "default.qubit", "diff_method": "backprop"}
N_QUBITS = 4
N_LAYERS = 3  # > n-1 so the range rule wraps: ranges 1, 2, 3 for 4 qubits
BATCH = 5


def _angles(n: int = BATCH, n_qubits: int = N_QUBITS) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(n, n_qubits) * 2 * math.pi - math.pi


# ---------------------------------------------------------------------------
# Encoding layers
# ---------------------------------------------------------------------------


class TestEntangler:
    def test_strongly_entangling_matches_hand_written_rot_then_cnot(self) -> None:
        torch.manual_seed(0)
        weights = torch.randn(N_LAYERS, N_QUBITS, 3, dtype=torch.float64)
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev, interface="torch")
        def reference(x: torch.Tensor) -> list:
            qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="X")
            for layer in range(N_LAYERS):
                for q in range(N_QUBITS):
                    qml.Rot(
                        weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q
                    )
                r = layer % (N_QUBITS - 1) + 1
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + r) % N_QUBITS])
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        ours = build_encoding_qnode(
            n_qubits=N_QUBITS, n_layers=N_LAYERS, entangler="strongly_entangling", **CPU
        )
        for x in _angles(3).double():
            with torch.no_grad():
                torch.testing.assert_close(
                    torch.stack(ours(x, weights)), torch.stack(reference(x)), rtol=0, atol=1e-12
                )

    def test_layer_offset_continues_the_range_rule(self) -> None:
        """
        Blocks applied in two calls, 1 then 2 at layer_offset=1, give the same
        circuit as all 3 in one call: ranges 1, 2, 3 on 4 qubits, not a
        restart at 1 or one range repeated across the second call.
        """
        torch.manual_seed(0)
        weights = torch.randn(N_LAYERS, N_QUBITS, 3, dtype=torch.float64)
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev, interface="torch")
        def at_once() -> list:
            apply_variational_layers(weights, N_QUBITS, N_LAYERS, "strongly_entangling")
            return measure_z(N_QUBITS)

        @qml.qnode(dev, interface="torch")
        def split() -> list:
            apply_variational_layers(weights[:1], N_QUBITS, 1, "strongly_entangling")
            apply_variational_layers(
                weights[1:], N_QUBITS, N_LAYERS - 1, "strongly_entangling", layer_offset=1
            )
            return measure_z(N_QUBITS)

        torch.testing.assert_close(
            torch.stack(split()), torch.stack(at_once()), rtol=0, atol=1e-12
        )

    def test_strongly_entangling_on_one_qubit(self) -> None:
        """One wire has no CNOT partner; the block must not divide by n - 1 = 0."""
        weights = torch.randn(2, 1, 3, dtype=torch.float64)
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="torch")
        def ours() -> list:
            apply_variational_layers(weights, 1, 2, "strongly_entangling")
            return measure_z(1)

        @qml.qnode(dev, interface="torch")
        def reference() -> list:
            for layer in range(2):
                qml.Rot(*weights[layer, 0], wires=0)
            return [qml.expval(qml.PauliZ(0))]

        torch.testing.assert_close(
            torch.stack(ours()), torch.stack(reference()), rtol=0, atol=1e-12
        )

    def test_ring_is_unchanged(self) -> None:
        """The default entangler is still CNOT ring then Rot, as documented."""
        qnode = build_encoding_qnode(n_qubits=N_QUBITS, n_layers=1, **CPU)
        tape = qml.workflow.construct_tape(qnode, level=0)(
            torch.zeros(N_QUBITS), torch.zeros(1, N_QUBITS, 3)
        )
        names = [op.name for op in tape.operations]
        assert names == ["AngleEmbedding"] + ["CNOT"] * N_QUBITS + ["Rot"] * N_QUBITS

    def test_two_entanglers_differ_but_share_the_parameter_count(self) -> None:
        torch.manual_seed(0)
        ring = QuantumEncodingLayer(n_qubits=N_QUBITS, n_layers=2, **CPU)
        sel = QuantumEncodingLayer(
            n_qubits=N_QUBITS, n_layers=2, entangler="strongly_entangling", **CPU
        )
        with torch.no_grad():
            sel.qlayer.weights.copy_(ring.qlayer.weights)
        assert ring.qlayer.weights.shape == sel.qlayer.weights.shape
        x = _angles()
        with torch.no_grad():
            assert not torch.allclose(ring(x), sel(x), atol=1e-3)
        assert "entangler='strongly_entangling'" in sel.extra_repr()
        assert "entangler" not in ring.extra_repr()

    def test_gradients_flow_through_the_template(
        self, grad_of: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        layer = QuantumEncodingLayer(
            n_qubits=N_QUBITS, n_layers=2, entangler="strongly_entangling", **CPU
        )
        x = _angles().requires_grad_(True)
        layer(x).sum().backward()
        assert grad_of(_tensor(layer.qlayer.weights)).abs().sum().item() > 0
        assert grad_of(x).abs().sum().item() > 0

    def test_unknown_entangler_raises(self) -> None:
        with pytest.raises(ValueError, match="entangler"):
            QuantumEncodingLayer(n_qubits=N_QUBITS, n_layers=1, entangler="ladder", **CPU)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="entangler"):
            apply_variational_layers(torch.zeros(1, 2, 3), 2, 1, "ladder")  # type: ignore[arg-type]


class TestBrickwork:
    """``entangler="brickwork"`` (#161): even, then odd nearest-neighbour CNOTs, then Rot."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_matches_hand_written_brickwork(self, n_qubits: int) -> None:
        torch.manual_seed(0)
        weights = torch.randn(2, n_qubits, 3, dtype=torch.float64)
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def reference(x: torch.Tensor) -> list:
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="X")
            for layer in range(2):
                for q in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(n_qubits):
                    qml.Rot(*weights[layer, q], wires=q)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        ours = build_encoding_qnode(n_qubits=n_qubits, n_layers=2, entangler="brickwork", **CPU)
        torch.manual_seed(1)
        for x in (torch.rand(3, n_qubits, dtype=torch.float64) * 2 - 1) * math.pi:
            with torch.no_grad():
                torch.testing.assert_close(
                    torch.stack(ours(x, weights)), torch.stack(reference(x)), rtol=0, atol=1e-12
                )

    def test_gate_sequence_has_no_wraparound(self) -> None:
        """n - 1 CNOTs per layer, never CNOT(n-1, 0), and the Rot block after them."""
        qnode = build_encoding_qnode(n_qubits=5, n_layers=2, entangler="brickwork", **CPU)
        tape = qml.workflow.construct_tape(qnode, level=0)(torch.zeros(5), torch.zeros(2, 5, 3))
        ops = [(op.name, op.wires.tolist()) for op in tape.operations[1:]]
        cnots = [("CNOT", [0, 1]), ("CNOT", [2, 3]), ("CNOT", [1, 2]), ("CNOT", [3, 4])]
        rots = [("Rot", [q]) for q in range(5)]
        assert ops == (cnots + rots) * 2

    @pytest.mark.parametrize(
        ("n_layers", "light_cone"), [(1, [0]), (2, [0, 1]), (3, [0, 1, 2, 3])]
    )
    def test_z0_light_cone_does_not_grow_with_the_register(
        self, n_layers: int, light_cone: list[int]
    ) -> None:
        """
        The qubits whose weights ⟨Z_0⟩ has a gradient for: {0}, {0, 1},
        {0, …, 3} after 1, 2, 3 layers, at 8 qubits as at 10.  The ring
        reaches every qubit after 2 layers.
        """
        for n_qubits in (8, 10):
            layer = QuantumEncodingLayer(
                n_qubits=n_qubits, n_layers=n_layers, entangler="brickwork", **CPU
            )
            torch.manual_seed(0)
            with torch.no_grad():
                layer.qlayer.weights.uniform_(0, 2 * math.pi)
            x = (torch.rand(1, n_qubits) * 2 - 1) * math.pi
            (grad,) = torch.autograd.grad(layer(x)[0, 0], layer.qlayer.weights)
            reached = [q for q in range(n_qubits) if grad[:, q].abs().max() > 1e-7]
            assert reached == light_cone, (n_qubits, reached)

    def test_every_encoder_and_classifier_accepts_it(self) -> None:
        x = _angles()
        for cls in (QuantumEncodingLayer, IQPEncodingLayer, DataReuploadingLayer):
            layer = cls(n_qubits=N_QUBITS, n_layers=2, entangler="brickwork", **CPU)
            assert layer.qlayer.weights.shape == (2, N_QUBITS, 3)
            assert "entangler='brickwork'" in layer.extra_repr()
            layer(x).sum().backward()
            assert layer.qlayer.weights.grad.abs().sum().item() > 0, cls.__name__
        for model_cls in (HybridBinaryClassifier, ParallelHybridClassifier):
            model = model_cls(
                n_input_features=6, n_qubits=N_QUBITS, n_layers=2, entangler="brickwork", **CPU
            )
            assert model.quantum_layer.entangler == "brickwork"
            model(torch.randn(3, 6)).sum().backward()
            grad = model.quantum_layer.qlayer.weights.grad
            assert grad is not None and grad.abs().sum().item() > 0, model_cls.__name__


_ALL = [0, 1, 2, 3, 4]
_CASCADE_ONE_LAYER = [[1, 2, 3, 4], [0, 1], [0, 1, 2], [0, 1, 2, 3], _ALL]


class TestReadoutFeatures:
    """
    Which features each readout sees (#150), as derived in step 2 of
    ``_make_angle_embedding_circuit``: under the default ring and RX
    embedding, after one layer, ⟨Z_0⟩ = c_0(w)·cos x_1⋯cos x_{n-1}, for
    0 < i < n-1 ⟨Z_i⟩ = c_i(w)·cos x_0⋯cos x_i, and ⟨Z_{n-1}⟩ adds a
    sin x_0 sin x_1 cos x_2⋯cos x_{n-2} sin x_{n-1} term to the full product.
    """

    N = 5

    def _layer(self, entangler: str = "ring", rotation: str = "X", n_layers: int = 1):
        layer = QuantumEncodingLayer(
            n_qubits=self.N,
            n_layers=n_layers,
            entangler=entangler,  # type: ignore[arg-type]
            rotation=rotation,  # type: ignore[arg-type]
            **CPU,  # type: ignore[arg-type]
        )
        torch.manual_seed(0)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        return layer

    def _ring_readouts(self) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self._layer()
        torch.manual_seed(1)
        # |x| < 1.2 keeps every cosine away from zero, so the ratios are well defined
        x = (torch.rand(8, self.N, dtype=torch.float64) * 2 - 1) * 1.2
        with torch.no_grad():
            return x, layer(x.float()).double()

    def test_ring_readouts_are_the_derived_cosine_products(self) -> None:
        """Each ratio to its cosine product is a constant c_i(w), whatever the input."""
        x, out = self._ring_readouts()
        cos = torch.cos(x)
        products = [cos[:, 1:].prod(dim=1)]
        products += [cos[:, : i + 1].prod(dim=1) for i in range(1, self.N - 1)]
        for i, product in enumerate(products):
            ratio = out[:, i] / product
            torch.testing.assert_close(
                ratio, ratio[:1].expand_as(ratio), rtol=1e-4, atol=1e-5, msg=f"readout {i}"
            )
            assert ratio[0].abs() > 1e-2, f"readout {i}: c_i(w) vanished, the check is vacuous"

    def test_ring_last_readout_has_the_wraparound_sine_term(self) -> None:
        """⟨Z_{n-1}⟩ is exactly a·∏cos x_j + b·sin x_0 sin x_1 ∏cos x_{2..n-2} sin x_{n-1}."""
        x, out = self._ring_readouts()
        cos, sin = torch.cos(x), torch.sin(x)
        basis = torch.stack(
            [cos.prod(dim=1), sin[:, 0] * sin[:, 1] * cos[:, 2:-1].prod(dim=1) * sin[:, -1]], dim=1
        )
        coef = torch.linalg.lstsq(basis, out[:, -1:]).solution
        torch.testing.assert_close(basis @ coef, out[:, -1:], rtol=0, atol=1e-5)
        assert coef.abs().min() > 1e-2, f"a or b vanished, the check is vacuous: {coef.flatten()}"

    @pytest.mark.parametrize(
        ("entangler", "rotation", "n_layers", "seen"),
        [
            ("ring", "X", 1, _CASCADE_ONE_LAYER),
            ("ring", "Y", 1, [_ALL, [0, 1, 2], [0, 1, 2, 3], _ALL, _ALL]),
            ("strongly_entangling", "X", 1, _CASCADE_ONE_LAYER),
            ("strongly_entangling", "Y", 1, _CASCADE_ONE_LAYER),
            ("ring", "X", 2, [_ALL] * 5),
            ("ring", "Y", 2, [_ALL] * 5),
            ("strongly_entangling", "X", 2, [_ALL] * 5),
            ("strongly_entangling", "Y", 2, [_ALL] * 5),
            # Each ⟨Z_i⟩ sees x_i, but ⟨Z_0⟩'s narrow light cone misses most features
            ("brickwork", "X", 1, [[0], [0, 1], [0, 1, 2], [2, 3], [2, 3, 4]]),
            ("brickwork", "Y", 1, [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3], [2, 3, 4], [2, 3, 4]]),
            ("brickwork", "X", 2, [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3], _ALL, _ALL]),
            ("brickwork", "Y", 2, [[0, 1, 2, 3], _ALL, _ALL, _ALL, _ALL]),
            # CZ is diagonal: ⟨Z_i⟩ sees a neighbour only through the X_i the RY mixes
            # in, which the CZs dress with Z_{i±1}.  After L layers ⟨Z_i⟩ sees
            # x_{i-L} … x_{i+L}, except that one layer under RX (⟨X⟩ = 0) sees x_i alone
            ("hardware_efficient", "X", 1, [[0], [1], [2], [3], [4]]),
            ("hardware_efficient", "Y", 1, [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4]]),
            (
                "hardware_efficient",
                "X",
                2,
                [[0, 1, 2], [0, 1, 2, 3], _ALL, [1, 2, 3, 4], [2, 3, 4]],
            ),
            (
                "hardware_efficient",
                "Y",
                2,
                [[0, 1, 2], [0, 1, 2, 3], _ALL, [1, 2, 3, 4], [2, 3, 4]],
            ),
            # "Z" sees nothing at any depth and is refused; see TestRotationZIsRefused
        ],
    )
    def test_features_each_readout_sees(
        self, entangler: str, rotation: str, n_layers: int, seen: list[list[int]]
    ) -> None:
        """
        The first harmonic of ⟨Z_i⟩ in x_j, the other features fixed: above
        1e-5 where readout i sees feature j (products of cosines can be as
        small as 2e-4 here), below 1e-7 where it is blind.
        """
        layer = self._layer(entangler, rotation, n_layers)
        n_points = 16
        torch.manual_seed(1)
        base = (torch.rand(1, self.N) * 2 - 1) * math.pi
        for j in range(self.N):
            x = base.repeat(n_points, 1)
            x[:, j] = torch.arange(n_points) * 2 * math.pi / n_points - math.pi
            with torch.no_grad():
                values = layer(x).to(torch.float64)
            harmonic = torch.fft.rfft(values, dim=0).abs()[1] / n_points
            for i in range(self.N):
                if j in seen[i]:
                    assert harmonic[i] > 1e-5, f"⟨Z_{i}⟩ should see x_{j}: {harmonic[i]:.2e}"
                else:
                    assert harmonic[i] < 1e-7, (
                        f"⟨Z_{i}⟩ should be blind to x_{j}: {harmonic[i]:.2e}"
                    )


class TestReadout:
    @pytest.mark.parametrize("cls", [QuantumEncodingLayer, IQPEncodingLayer])
    def test_first_is_column_zero_of_all(
        self, cls: type[QuantumEncodingLayer | IQPEncodingLayer]
    ) -> None:
        torch.manual_seed(0)
        every = cls(n_qubits=N_QUBITS, n_layers=2, **CPU)
        first = cls(n_qubits=N_QUBITS, n_layers=2, readout="first", **CPU)
        with torch.no_grad():
            _tensor(first.qlayer.weights).copy_(_tensor(every.qlayer.weights))
        x = _angles()
        with torch.no_grad():
            out_first = first(x)
            out_all = every(x)
        assert out_first.shape == (BATCH, 1)
        assert first.n_outputs == 1 and every.n_outputs == N_QUBITS
        torch.testing.assert_close(out_first[:, 0], out_all[:, 0], rtol=0, atol=1e-6)

    def test_readout_wires(self) -> None:
        assert readout_wires(4, "all") == [0, 1, 2, 3]
        assert readout_wires(4, "first") == [0]
        with pytest.raises(ValueError, match="readout"):
            readout_wires(4, "last")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="readout"):
            QuantumEncodingLayer(n_qubits=N_QUBITS, n_layers=1, readout="last", **CPU)  # type: ignore[arg-type]

    def test_unknown_rotation_raises_at_construction(self) -> None:
        # qml.AngleEmbedding only rejects the axis when the circuit first runs,
        # a forward pass away from the constructor that was handed it.
        with pytest.raises(ValueError, match="rotation"):
            QuantumEncodingLayer(n_qubits=N_QUBITS, n_layers=1, rotation="W", **CPU)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="rotation"):
            build_encoding_qnode(n_qubits=N_QUBITS, n_layers=1, rotation="W", **CPU)  # type: ignore[arg-type]

    def test_measure_z_returns_one_expectation_per_wire(self) -> None:
        with qml.queuing.AnnotatedQueue() as q:
            measurements = measure_z(3, "all")
        assert len(measurements) == 3 and len(q.queue) == 3
        assert [m.wires.tolist() for m in measurements] == [[0], [1], [2]]


class TestRotationZIsRefused:
    """
    One RZ embedding on |0…0⟩ is a global phase, so the layer would return
    the same outputs for every input (#212).
    """

    def test_layer_and_qnode_builder_refuse_it(self) -> None:
        with pytest.raises(ValueError, match="global phase"):
            QuantumEncodingLayer(n_qubits=N_QUBITS, n_layers=2, rotation="Z", **CPU)
        with pytest.raises(ValueError, match='rotation="Z" would make the layer ignore'):
            build_encoding_qnode(n_qubits=N_QUBITS, n_layers=2, rotation="Z", **CPU)

    @pytest.mark.parametrize("cls", [HybridBinaryClassifier, ParallelHybridClassifier])
    def test_classifiers_refuse_it(self, cls: type) -> None:
        with pytest.raises(ValueError, match="global phase"):
            _classifier(cls, embedding_rotation="Z")

    def test_why_it_is_refused(self) -> None:
        # The circuit the layer would have run, built by hand: the outputs do
        # not move with the inputs and the input gradients vanish.
        dev = qml.device("default.qubit", wires=N_QUBITS)
        torch.manual_seed(0)
        weights = torch.randn(3, N_QUBITS, 3, dtype=torch.float64)

        @qml.qnode(dev, interface="torch")
        def circuit(x: torch.Tensor) -> list:
            qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Z")
            apply_variational_layers(weights, N_QUBITS, 3)
            return measure_z(N_QUBITS)

        x = (torch.rand(N_QUBITS, dtype=torch.float64) * 2 - 1) * math.pi
        x.requires_grad_(True)
        out = torch.stack(circuit(x))
        (grad,) = torch.autograd.grad(out.sum(), x)
        other = torch.stack(circuit(-x.detach()))
        torch.testing.assert_close(out.detach(), other, rtol=0, atol=1e-12)
        assert grad.abs().max() < 1e-12

    @pytest.mark.parametrize("rotation", ["X", "Y"])
    def test_the_accepted_axes_depend_on_the_input(self, rotation: str) -> None:
        layer = QuantumEncodingLayer(
            n_qubits=N_QUBITS,
            n_layers=2,
            rotation=rotation,  # type: ignore[arg-type]
            **CPU,
        )
        x = _angles()
        x.requires_grad_(True)
        (grad,) = torch.autograd.grad(layer(x).sum(), x)
        assert grad.abs().max() > 1e-3


class TestRotationPassThrough:
    def test_rotation_y_changes_the_embedding_gate(self) -> None:
        qnode = build_encoding_qnode(n_qubits=N_QUBITS, n_layers=1, rotation="Y", **CPU)
        tape = qml.workflow.construct_tape(qnode, level=0)(
            torch.zeros(N_QUBITS), torch.zeros(1, N_QUBITS, 3)
        )
        embedding = next(op for op in tape.operations if op.name == "AngleEmbedding")
        assert embedding.hyperparameters["rotation"] is qml.RY


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


CLASSIFIERS = [
    pytest.param(HybridBinaryClassifier, id="serial"),
    pytest.param(ParallelHybridClassifier, id="parallel"),
]


Classifier = HybridBinaryClassifier | ParallelHybridClassifier
C = TypeVar("C", bound=Classifier)


def _classifier(cls: type[C], **kwargs: Any) -> C:
    torch.manual_seed(0)
    return cls(n_input_features=6, n_qubits=N_QUBITS, n_layers=2, **CPU, **kwargs)


def _tensor(value: object) -> torch.Tensor:
    """A parameter reached through ``nn.Module.__getattr__``, narrowed to a tensor."""
    assert isinstance(value, torch.Tensor)
    return value


def _encoder_part(model: Classifier, index: int) -> nn.Module:
    """``model.classical_encoder[index]``: the Linear (0) or the activation (1)."""
    encoder = model.classical_encoder
    assert isinstance(encoder, nn.Sequential)
    return encoder[index]


class TestClassifierOptions:
    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_defaults_are_unchanged(self, cls: type[Classifier]) -> None:
        model = _classifier(cls)
        assert model.quantum_layer.entangler == "ring"
        assert model.quantum_layer.readout == "all"
        assert model.encoder_activation == "tanh"
        assert isinstance(_encoder_part(model, 1), nn.Tanh)
        expected_in = N_QUBITS if cls is HybridBinaryClassifier else 16 + N_QUBITS
        assert model.head.in_features == expected_in

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_readout_first_narrows_the_head(self, cls: type[Classifier]) -> None:
        model = _classifier(cls, readout="first")
        expected_in = 1 if cls is HybridBinaryClassifier else 16 + 1
        assert model.head.in_features == expected_in
        assert model(torch.randn(BATCH, 6)).shape == (BATCH, 1)
        assert model.predict_proba(torch.randn(BATCH, 6)).shape == (BATCH,)

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_sigmoid_encoder_maps_into_zero_to_pi(self, cls: type[Classifier]) -> None:
        model = _classifier(cls, encoder_activation="sigmoid")
        assert isinstance(_encoder_part(model, 1), nn.Sigmoid)
        x = torch.linspace(-50, 50, 6 * 10).reshape(10, 6)
        with torch.no_grad():
            angles = model.classical_encoder(x) * torch.pi
        assert 0.0 <= angles.min().item()
        assert angles.max().item() <= math.pi + 1e-6  # float32 π rounds up
        # And the quantum layer receives exactly those angles.
        seen: list[torch.Tensor] = []
        handle = model.quantum_layer.register_forward_pre_hook(lambda _m, inp: seen.append(inp[0]))
        with torch.no_grad():
            model(x)
        handle.remove()
        torch.testing.assert_close(seen[0], angles)

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_embedding_rotation_and_entangler_reach_the_quantum_layer(
        self, cls: type[Classifier]
    ) -> None:
        model = _classifier(cls, embedding_rotation="Y", entangler="strongly_entangling")
        tape = qml.workflow.construct_tape(model.quantum_layer.qlayer.qnode, level=0)(
            torch.zeros(N_QUBITS), torch.zeros(2, N_QUBITS, 3)
        )
        names = [op.name for op in tape.operations]
        assert names == ["AngleEmbedding", "StronglyEntanglingLayers"]
        assert tape.operations[0].hyperparameters["rotation"] is qml.RY

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_normal_init(self, cls: type[Classifier]) -> None:
        torch.manual_seed(0)
        model = cls(
            n_input_features=16,
            n_qubits=16,
            n_layers=16,
            init_strategy="normal",
            init_std=0.05,
            **CPU,
        )
        weights = _tensor(model.quantum_layer.qlayer.weights).detach()
        assert weights.std().item() == pytest.approx(0.05, rel=0.1)
        assert abs(weights.mean().item()) < 0.01

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_options_train(
        self, cls: type[Classifier], grad_of: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        model = _classifier(
            cls,
            embedding_rotation="Y",
            entangler="strongly_entangling",
            readout="first",
            encoder_activation="sigmoid",
            init_strategy="normal",
        )
        x = torch.randn(BATCH, 6)
        model(x).sum().backward()
        assert grad_of(_tensor(model.quantum_layer.qlayer.weights)).abs().sum().item() > 0
        assert grad_of(_tensor(_encoder_part(model, 0).weight)).abs().sum().item() > 0
        assert grad_of(model.head.weight).abs().sum().item() > 0

    def test_iqp_rejects_embedding_rotation(self) -> None:
        with pytest.raises(ValueError, match="embedding_rotation"):
            _classifier(HybridBinaryClassifier, encoding_type="iqp", embedding_rotation="Y")

    def test_iqp_accepts_entangler_and_readout(self) -> None:
        model = _classifier(
            HybridBinaryClassifier,
            encoding_type="iqp",
            entangler="strongly_entangling",
            readout="first",
        )
        assert model.head.in_features == 1
        assert model(torch.randn(BATCH, 6)).shape == (BATCH, 1)

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_validation(self, cls: type[Classifier]) -> None:
        with pytest.raises(ValueError, match="encoder_activation"):
            _classifier(cls, encoder_activation="relu")
        with pytest.raises(ValueError, match="init_strategy"):
            _classifier(cls, init_strategy="xavier")
        with pytest.raises(ValueError, match="embedding_rotation|rotation"):
            _classifier(cls, embedding_rotation="W")
        with pytest.raises(ValueError, match="init_std"):
            _classifier(cls, init_std=0.0)

    @pytest.mark.parametrize("cls", CLASSIFIERS)
    def test_an_option_that_would_be_ignored_raises_instead(self, cls: type[Classifier]) -> None:
        """
        Both options below are inert outside the configuration that uses them.
        Accepting one there would record it in get_config() and in a checkpoint,
        so the config would describe a model the library never built.
        """
        with pytest.raises(ValueError, match="encoder_activation"):
            cls(
                n_input_features=N_QUBITS,
                n_qubits=N_QUBITS,
                n_layers=1,
                use_classical_encoder=False,
                encoder_activation="sigmoid",
                **CPU,
            )
        with pytest.raises(ValueError, match="init_std"):
            _classifier(cls, init_strategy="restricted", init_std=0.05)

        # The defaults stay accepted in both places: they say nothing.
        no_encoder = cls(
            n_input_features=N_QUBITS,
            n_qubits=N_QUBITS,
            n_layers=1,
            use_classical_encoder=False,
            **CPU,
        )
        assert isinstance(no_encoder.classical_encoder, nn.Identity)
        assert _classifier(cls, init_strategy="normal", init_std=0.05).init_std == 0.05


class TestPublishedPreset:
    def test_serial_preset_has_122_parameters(self) -> None:
        model = HybridBinaryClassifier.published_shnn(**CPU)
        assert model.count_parameters() == 122
        assert model.quantum_layer.qlayer.weights.numel() == 48
        assert model.head.in_features == 1
        assert model.encoder_activation == "sigmoid"
        assert model.quantum_layer.readout == "first"
        assert model.quantum_layer.entangler == "strongly_entangling"

    def test_overrides_apply(self) -> None:
        model = HybridBinaryClassifier.published_shnn(n_layers=3, **CPU)
        assert model.n_layers == 3
        assert model.count_parameters() == 122 + 8 * 3

    def test_parallel_preset_uses_the_same_quantum_branch(self) -> None:
        model = ParallelHybridClassifier.published_shnn(**CPU)
        assert model.quantum_layer.qlayer.weights.numel() == 48
        assert model.quantum_layer.readout == "first"
        assert model.head.in_features == model.classical_hidden_dim + 1
