"""
tests/test_expressibility.py
============================
Expressibility and entangling capability (hqnn_forge.diagnostics, #318).

The formulas are checked against exact values (the Haar bin probabilities,
the maximum for a constant state, Meyer–Wallach of product, GHZ and Bell
states, and the Haar expectation (N − 2)/(N + 1)); the layer-level functions
against small layers built so that the answer is known in closed form, and
against the layer's own states.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import pennylane as qml
import pytest
import torch
import torch.nn as nn

from hqnn_forge.diagnostics import (
    entangling_capability,
    expressibility,
    meyer_wallach,
)
from hqnn_forge.diagnostics.expressibility import (
    expressibility_from_fidelities,
    haar_fidelity_bin_probabilities,
)
from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.kernels import encoded_states
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import apply_depolarizing_noise

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
# The module, not the function of the same name that hqnn_forge.diagnostics
# exports and that shadows it as an attribute.
expr_module = importlib.import_module("hqnn_forge.diagnostics.expressibility")
Z = 5.0


def _haar_states(n: int, n_qubits: int, seed: int) -> torch.Tensor:
    """Haar-random states: normalised complex Gaussian vectors."""
    g = torch.Generator().manual_seed(seed)
    shape = (n, 2**n_qubits)
    z = torch.complex(
        torch.randn(shape, generator=g, dtype=torch.float64),
        torch.randn(shape, generator=g, dtype=torch.float64),
    )
    return z / z.norm(dim=1, keepdim=True)


class TestHaarBins:
    @pytest.mark.parametrize("n_qubits, n_bins", [(1, 10), (2, 75), (4, 75), (6, 20)])
    def test_probabilities_are_the_integrated_density(self, n_qubits: int, n_bins: int) -> None:
        q = haar_fidelity_bin_probabilities(n_qubits, n_bins)
        assert q.shape == (n_bins,)
        torch.testing.assert_close(q.sum(), torch.tensor(1.0, dtype=torch.float64))
        # Midpoint rule on a fine grid inside each bin, against the closed form.
        n = 2**n_qubits
        fine = 2000
        for i in (0, n_bins // 2, n_bins - 1):
            f = (i + (torch.arange(fine, dtype=torch.float64) + 0.5) / fine) / n_bins
            numeric = ((n - 1) * (1 - f) ** (n - 2)).mean() / n_bins
            torch.testing.assert_close(q[i], numeric, rtol=1e-5, atol=1e-12)

    def test_one_qubit_haar_fidelity_is_uniform(self) -> None:
        torch.testing.assert_close(
            haar_fidelity_bin_probabilities(1, 8), torch.full((8,), 1 / 8, dtype=torch.float64)
        )


class TestExpressibilityFormula:
    @pytest.mark.parametrize("n_qubits, n_bins", [(2, 75), (3, 20), (8, 75), (10, 75)])
    def test_a_constant_state_scores_the_maximum(self, n_qubits: int, n_bins: int) -> None:
        value = expressibility_from_fidelities(torch.ones(100), n_qubits, n_bins)
        assert value == pytest.approx((2**n_qubits - 1) * math.log(n_bins), rel=1e-12)

    def test_haar_states_score_near_zero(self) -> None:
        n_qubits, n_pairs, n_bins = 3, 5000, 75
        states = _haar_states(2 * n_pairs, n_qubits, seed=0)
        f = (states[:n_pairs].conj() * states[n_pairs:]).sum(1).abs() ** 2
        value = expressibility_from_fidelities(f, n_qubits, n_bins)
        # Only the histogram bias, about (occupied bins - 1) / (2 n_pairs).
        assert 0 <= value < 0.03

    def test_rejects_bad_fidelities(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            expressibility_from_fidelities(torch.tensor([]), 2)
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            expressibility_from_fidelities(torch.tensor([0.5, 1.2]), 2)


class TestMeyerWallach:
    def test_product_states_are_zero(self) -> None:
        singles = [_haar_states(5, 1, seed=s) for s in range(3)]
        product = torch.stack(
            [torch.kron(torch.kron(singles[0][i], singles[1][i]), singles[2][i]) for i in range(5)]
        )
        torch.testing.assert_close(
            meyer_wallach(product, 3), torch.zeros(5, dtype=torch.float64), atol=1e-12, rtol=0
        )

    def test_known_entangled_states(self) -> None:
        s = 1 / math.sqrt(2)
        ghz3 = torch.zeros(8, dtype=torch.complex128)
        ghz3[0] = ghz3[7] = s
        bell = torch.tensor([s, 0, 0, s], dtype=torch.complex128)
        zero_bell = torch.kron(torch.tensor([1, 0], dtype=torch.complex128), bell)
        torch.testing.assert_close(
            meyer_wallach(ghz3[None], 3), torch.tensor([1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            meyer_wallach(bell[None], 2), torch.tensor([1.0], dtype=torch.float64)
        )
        # Purities 1, 1/2, 1/2: Q = 2 (1 − 2/3) = 2/3.
        torch.testing.assert_close(
            meyer_wallach(zero_bell[None], 3), torch.tensor([2 / 3], dtype=torch.float64)
        )

    def test_invariant_under_permuting_qubits(self) -> None:
        states = _haar_states(20, 3, seed=2)
        permuted = states.reshape(20, 2, 2, 2).permute(0, 3, 1, 2).reshape(20, 8)
        torch.testing.assert_close(meyer_wallach(permuted, 3), meyer_wallach(states, 3))

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_haar_mean_is_scott_s_value(self, n_qubits: int) -> None:
        q = meyer_wallach(_haar_states(20_000, n_qubits, seed=n_qubits), n_qubits)
        n = 2**n_qubits
        se = q.std() / math.sqrt(q.numel())
        assert abs(q.mean() - (n - 2) / (n + 1)) <= Z * se

    def test_rejects_the_wrong_width(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(batch, 8\)"):
            meyer_wallach(torch.zeros(3, 4, dtype=torch.complex128), 3)


# ---------------------------------------------------------------------------
# Layers whose answer is known in closed form
# ---------------------------------------------------------------------------


class _CustomLayer(nn.Module):
    """A minimal EncodingLayer (qlayer, n_qubits, n_features, prepare_inputs) around ``circuit``."""

    def __init__(self, circuit: Any, n_qubits: int, weight_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = n_qubits
        qnode = qml.QNode(circuit, qml.device("default.qubit", wires=n_qubits), interface="torch")
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": weight_shape})

    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlayer(self.prepare_inputs(x))  # type: ignore[no-any-return]


def _product_layer() -> _CustomLayer:
    # Rotations only: every state is a product state, Q = 0 exactly.
    def circuit(inputs, weights):  # type: ignore[no-untyped-def]
        for i in range(3):
            qml.RX(inputs[..., i], wires=i)
            qml.Rot(*weights[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    return _CustomLayer(circuit, 3, (3, 3))


def _ghz_layer() -> _CustomLayer:
    # GHZ, then Z phases on each qubit: still |0…0⟩ + e^{iφ}|1…1⟩, so Q = 1.
    def circuit(inputs, weights):  # type: ignore[no-untyped-def]
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        for i in range(3):
            qml.RZ(weights[i] + inputs[..., i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    return _CustomLayer(circuit, 3, (3,))


def _constant_layer() -> _CustomLayer:
    # Z rotations on |000⟩ are global phases: one state for every draw.
    def circuit(inputs, weights):  # type: ignore[no-untyped-def]
        for i in range(3):
            qml.RZ(weights[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    return _CustomLayer(circuit, 3, (3,))


class TestKnownLayers:
    def test_product_layer_has_zero_entangling_capability(self) -> None:
        result = entangling_capability(_product_layer(), n_samples=50)
        assert result.entangling_capability == pytest.approx(0.0, abs=1e-12)
        assert result.values.abs().max() < 1e-12

    def test_ghz_layer_has_entangling_capability_one(self) -> None:
        result = entangling_capability(_ghz_layer(), n_samples=50)
        torch.testing.assert_close(result.values, torch.ones(50, dtype=torch.float64))
        assert result.standard_error == pytest.approx(0.0, abs=1e-12)

    def test_constant_layer_scores_the_maximum_expressibility(self) -> None:
        result = expressibility(_constant_layer(), n_pairs=200, n_bins=75)
        torch.testing.assert_close(result.fidelities, torch.ones(200, dtype=torch.float64))
        assert result.expressibility == pytest.approx(result.maximum, rel=1e-9)
        assert result.maximum == pytest.approx(7 * math.log(75))

    def test_haar_reference_is_reported(self) -> None:
        result = entangling_capability(_product_layer(), n_samples=10)
        assert result.haar_reference == pytest.approx(6 / 9)


# ---------------------------------------------------------------------------
# The library's encoders
# ---------------------------------------------------------------------------


ENCODERS = [
    pytest.param(lambda: QuantumEncodingLayer(n_qubits=3, n_layers=2, **CPU), id="angle"),
    pytest.param(lambda: IQPEncodingLayer(n_qubits=3, n_layers=2, **CPU), id="iqp"),
    pytest.param(
        lambda: AmplitudeEncodingLayer(n_qubits=3, n_layers=2, n_features=5, **CPU), id="amplitude"
    ),
    pytest.param(
        lambda: DataReuploadingLayer(n_qubits=3, n_layers=2, trainable_input_scaling=True, **CPU),
        id="reuploading-scaled",
    ),
    pytest.param(
        lambda: QuantumEncodingLayer(
            n_qubits=3, n_layers=2, entangler="strongly_entangling", **CPU
        ),
        id="angle-strongly-entangling",
    ),
]


@pytest.mark.parametrize("build", ENCODERS)
@pytest.mark.parametrize("inputs", ["random", "fixed"])
def test_broadcast_states_are_bit_identical_to_tape_by_tape(
    build: Any, inputs: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #361: every chunk runs as one broadcast tape; without it (_broadcast
    # returning None) each tape runs alone.  Same draws, same states, bit for
    # bit.  40 states in chunks of 32: one full chunk and a partial one.
    monkeypatch.setattr(expr_module, "_CHUNK", 32)
    layer = build()
    width = int(getattr(layer, "n_features", layer.n_qubits))
    x: Any = "random" if inputs == "random" else torch.linspace(-1.0, 1.0, width)

    def sample() -> torch.Tensor:
        gen = torch.Generator().manual_seed(3)
        states, _ = expr_module._sample_states(layer, 40, x, gen, "test")
        return states

    # What reaches the simulator, not what _broadcast returns: a merged tape
    # whose run raises falls back silently, and would compare tape by tape
    # with itself below.
    runs: list[tuple[int, int | None]] = []
    real = qml.execute

    def spy(batch: Any, *args: Any, **kwargs: Any) -> Any:
        runs.append((len(batch), batch[0].batch_size))
        return real(batch, *args, **kwargs)

    monkeypatch.setattr(qml, "execute", spy)
    broadcast = sample()
    assert runs == [(1, 32), (1, 8)], "every chunk should run as one broadcast tape"
    runs.clear()
    monkeypatch.setattr(expr_module, "_broadcast", lambda tapes: None)
    assert torch.equal(broadcast, sample())
    assert runs == [(32, None), (8, None)]


def _theta(i: int) -> torch.Tensor:
    return torch.tensor(0.3 + 0.7 * i, dtype=torch.float64)


def _tapes(make: Any, n: int = 3) -> list[qml.tape.QuantumScript]:
    return [qml.tape.QuantumScript(make(i), [qml.state()]) for i in range(n)]


class TestBroadcastFallback:
    @staticmethod
    def _tape(*ops: Any) -> qml.tape.QuantumScript:
        return qml.tape.QuantumScript(list(ops), [qml.state()])

    def test_differing_structure_is_not_merged(self) -> None:
        a = self._tape(qml.RX(torch.tensor(0.1), 0), qml.CNOT([0, 1]))
        b = self._tape(qml.RY(torch.tensor(0.2), 0), qml.CNOT([0, 1]))
        c = self._tape(qml.RX(torch.tensor(0.2), 1), qml.CNOT([0, 1]))
        d = self._tape(qml.RX(torch.tensor(0.2), 0))
        assert expr_module._broadcast([a, b]) is None
        assert expr_module._broadcast([a, c]) is None
        assert expr_module._broadcast([a, d]) is None
        assert expr_module._broadcast([d, a]) is None

    def test_differing_hyperparameters_are_not_merged(self) -> None:
        # A Pauli word is no parameter, so it is not stacked: merging would
        # apply tape 0's word to every state (measured: off by 0.9, no error).
        words = _tapes(lambda i: [qml.PauliRot(_theta(i), "XY" if i == 0 else "ZZ", [0, 1])])
        assert expr_module._broadcast(words) is None
        # Likewise a template's rotation, and the base of a controlled operation.
        axes = _tapes(
            lambda i: [qml.AngleEmbedding(_theta(i).expand(2), [0, 1], rotation="XY"[i % 2])]
        )
        assert expr_module._broadcast(axes) is None
        bases = _tapes(
            lambda i: [qml.ctrl(qml.PauliRot(_theta(i), "XY" if i == 0 else "ZZ", [1, 2]), 0)]
        )
        assert expr_module._broadcast(bases) is None
        # ... while equal ones still merge.
        same = _tapes(lambda i: [qml.PauliRot(_theta(i), "XY", [0, 1])])
        assert expr_module._broadcast(same) is not None

    def test_a_single_or_already_broadcast_tape_is_not_merged(self) -> None:
        one = self._tape(qml.RX(torch.tensor(0.1), 0))
        batched = self._tape(qml.RX(torch.tensor([0.1, 0.2]), 0))
        assert expr_module._broadcast([]) is None
        assert expr_module._broadcast([one]) is None
        assert expr_module._broadcast([batched, batched]) is None

    def test_equal_structure_is_merged_into_one_batch(self) -> None:
        tapes = [self._tape(qml.RX(torch.tensor(t), 0), qml.CNOT([0, 1])) for t in (0.1, 0.2, 0.3)]
        merged = expr_module._broadcast(tapes)
        assert merged is not None and merged.batch_size == 3
        torch.testing.assert_close(merged.operations[0].data[0], torch.tensor([0.1, 0.2, 0.3]))

    def test_an_operation_that_keeps_its_stack_as_one_circuit_is_not_merged(self) -> None:
        # The tape reports a batch because of the RX, but the stacked operation
        # beside it does not take one and would run as a single circuit's.
        class Unbatched(qml.operation.Operation):
            num_wires = 1
            ndim_params = (1,)

            @property
            def batch_size(self) -> None:
                return None

        tapes = _tapes(lambda i: [Unbatched(_theta(i).expand(2), wires=0), qml.RX(_theta(i), 0)])
        assert expr_module._broadcast(tapes) is None

    def test_a_template_without_a_batch_of_its_own_is_merged_as_its_gates(self) -> None:
        # StronglyEntanglingLayers reports no batch for stacked weights; its
        # Rot gates do, so they stand in for it and the chunk still merges.
        def make(i: int) -> list[Any]:
            w = torch.rand(
                2, 2, 3, generator=torch.Generator().manual_seed(i), dtype=torch.float64
            )
            return [qml.StronglyEntanglingLayers(w, wires=[0, 1])]

        merged = expr_module._broadcast(_tapes(make))
        assert merged is not None and merged.batch_size == 3
        assert all(op.batch_size == 3 for op in merged.operations if op.data)


def _bits(i: int) -> torch.Tensor:
    return torch.tensor([i % 2, 1])


# Circuits outside this package's encoders, as a layer written against the
# encoding contract may hold them.  Whether each merges is PennyLane's business
# and may change with its version; that the states are right must not.
FOREIGN_CIRCUITS = [
    pytest.param(
        lambda i: [qml.BasisEmbedding(_bits(0), [0, 1]), qml.RX(_theta(i), 0)],
        id="basis-embedding-same-bits",
    ),
    pytest.param(
        lambda i: [qml.BasisEmbedding(_bits(i), [0, 1]), qml.RX(_theta(i), 0)],
        id="basis-embedding-differing-bits",
    ),
    pytest.param(
        lambda i: [qml.PauliRot(_theta(i), "XY" if i == 0 else "ZZ", [0, 1])], id="pauli-words"
    ),
    pytest.param(lambda i: [qml.RX(_theta(i), 0), qml.GlobalPhase(_theta(i))], id="global-phase"),
    pytest.param(
        lambda i: [qml.Hadamard(0), qml.MultiRZ(_theta(i), [0, 1]), qml.RY(_theta(i), 1)],
        id="multi-rz",
    ),
    pytest.param(
        lambda i: [
            qml.Hadamard(0),
            qml.ctrl(qml.RY(_theta(i), 1), 0),
            qml.IsingXX(_theta(i), [0, 1]),
        ],
        id="controlled",
    ),
    pytest.param(
        lambda i: [
            qml.BasicEntanglerLayers(_theta(i) * torch.ones(2, 2, dtype=torch.float64), [0, 1])
        ],
        id="basic-entangler",
    ),
    pytest.param(
        lambda i: [
            qml.SimplifiedTwoDesign(
                _theta(i).expand(2), _theta(i) * torch.ones(1, 1, 2, dtype=torch.float64), [0, 1]
            )
        ],
        id="simplified-two-design",
    ),
    pytest.param(
        lambda i: [
            qml.ArbitraryStatePreparation(_theta(i) * torch.ones(6, dtype=torch.float64), [0, 1])
        ],
        id="arbitrary-state-preparation",
    ),
]


@pytest.mark.parametrize("make", FOREIGN_CIRCUITS)
def test_foreign_circuits_give_the_tape_by_tape_states(make: Any) -> None:
    tapes = _tapes(make)
    device = qml.device("default.qubit", wires=2)
    expected = torch.stack(
        [torch.as_tensor(r).to(torch.complex128) for r in qml.execute(tapes, device)]
    )
    torch.testing.assert_close(
        torch.stack(expr_module._execute(tapes, device)), expected, rtol=0, atol=1e-12
    )


def test_a_broadcast_run_that_raises_falls_back_to_tape_by_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tapes = _tapes(lambda i: [qml.RX(_theta(i), 0)])
    assert expr_module._broadcast(tapes) is not None
    device = qml.device("default.qubit", wires=1)
    expected = torch.stack([torch.as_tensor(r) for r in qml.execute(tapes, device)])
    real = qml.execute
    sizes: list[int] = []

    def execute(batch: Any, *args: Any, **kwargs: Any) -> Any:
        sizes.append(len(batch))
        if len(batch) == 1:
            raise TypeError("cannot run a batch")
        return real(batch, *args, **kwargs)

    monkeypatch.setattr(qml, "execute", execute)
    states = torch.stack(expr_module._execute(tapes, device))
    assert sizes == [1, 3]
    torch.testing.assert_close(states, expected.to(torch.complex128), rtol=0, atol=0)


def test_a_broadcast_result_of_the_wrong_shape_falls_back_to_tape_by_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two tapes on two qubits: one unbatched state has 4 amplitudes, which
    # would also split into two rows of 2 if the width were not checked.
    tapes = _tapes(lambda i: [qml.RX(_theta(i), 0), qml.RY(_theta(i), 1)], n=2)
    assert expr_module._broadcast(tapes) is not None
    device = qml.device("default.qubit", wires=2)
    expected = torch.stack([torch.as_tensor(r) for r in qml.execute(tapes, device)])
    real = qml.execute
    sizes: list[int] = []

    def execute(batch: Any, *args: Any, **kwargs: Any) -> Any:
        sizes.append(len(batch))
        if len(batch) == 1:
            return (torch.zeros(4, dtype=torch.complex128),)
        return real(batch, *args, **kwargs)

    monkeypatch.setattr(qml, "execute", execute)
    states = torch.stack(expr_module._execute(tapes, device))
    assert sizes == [1, 2]
    torch.testing.assert_close(states, expected.to(torch.complex128), rtol=0, atol=0)


@pytest.mark.parametrize("build", ENCODERS)
class TestEncoders:
    def test_sampled_states_are_the_layer_s_own(self, build: Any) -> None:
        # Replay the first draw by hand: the input, then every trainable
        # tensor, uniform in [0, 2π), in qnode_weights order.  With those
        # weights in the layer, its own encoded state must be the first
        # sampled state.
        layer = build()
        before = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        result = entangling_capability(
            layer, n_samples=2, generator=torch.Generator().manual_seed(4)
        )
        for k, v in layer.state_dict().items():  # untouched
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

        gen = torch.Generator().manual_seed(4)
        width = layer.n_features
        x = (torch.rand(2, width, generator=gen, dtype=torch.float64) * 2 - 1) * math.pi
        with torch.no_grad():
            for p in layer.qlayer.qnode_weights.values():
                p.copy_(torch.rand(p.shape, generator=gen, dtype=torch.float64) * 2 * math.pi)
        state = encoded_states(x[:1], layer)
        torch.testing.assert_close(meyer_wallach(state, 3), result.values[:1], rtol=0, atol=1e-6)

    def test_values_are_in_range_and_reproducible(self, build: Any) -> None:
        a = expressibility(build(), n_pairs=100, n_bins=20)
        b = expressibility(build(), n_pairs=100, n_bins=20)
        assert a.expressibility == b.expressibility and 0 <= a.expressibility <= a.maximum
        e = entangling_capability(build(), n_samples=20)
        assert 0 <= e.entangling_capability <= 1


class TestSettings:
    def test_depth_increases_expressibility_of_the_ansatz(self) -> None:
        # Sim et al.'s trend, with the input fixed: one ring layer on 3 qubits
        # measured 0.065, four layers 0.017 -- the level of the histogram bias
        # at 1500 pairs, i.e. indistinguishable from Haar here.
        x = torch.full((3,), 0.3)
        shallow = expressibility(
            QuantumEncodingLayer(n_qubits=3, n_layers=1, **CPU), 1500, inputs=x
        )
        deep = expressibility(QuantumEncodingLayer(n_qubits=3, n_layers=4, **CPU), 1500, inputs=x)
        assert shallow.expressibility > 2 * deep.expressibility
        assert shallow.expressibility - deep.expressibility > 0.03

    def test_fidelities_of_a_deep_layer_have_the_haar_moments(self) -> None:
        # E[F] = 1/N and E[F²] = 2/(N(N+1)) for Haar pairs.  Checks that the
        # fidelity is |⟨ψ|φ⟩|² of independent draws: |⟨ψ|φ⟩| alone, or pairs
        # sharing a draw, would be far off.
        f = expressibility(QuantumEncodingLayer(n_qubits=3, n_layers=4, **CPU), 800).fidelities
        n = 8
        for moment, expected in ((f, 1 / n), (f**2, 2 / (n * (n + 1)))):
            se = moment.std() / math.sqrt(moment.numel())
            assert abs(moment.mean() - expected) <= Z * se

    def test_fixed_and_random_inputs_are_different_families(self) -> None:
        layer = IQPEncodingLayer(n_qubits=3, n_layers=1, **CPU)
        fixed = expressibility(layer, 200, inputs=torch.zeros(3))
        random = expressibility(layer, 200)
        assert not torch.equal(fixed.fidelities, random.fidelities)

    def test_a_classifier_is_refused_by_name(self) -> None:
        model = HybridBinaryClassifier(n_input_features=4, n_qubits=3, **CPU)
        with pytest.raises(TypeError, match="expressibility expects an encoding layer"):
            expressibility(model, 10)
        with pytest.raises(TypeError, match="entangling_capability expects an encoding layer"):
            entangling_capability(model, 10)

    def test_a_noise_transformed_layer_is_refused(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, **CPU)
        with apply_depolarizing_noise(layer, 0.1), pytest.raises(RuntimeError, match="replay"):
            expressibility(layer, 10)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n_pairs": 0}, "n_pairs and n_bins must be ≥ 1"),
            ({"inputs": "zeros"}, "inputs must be 'random' or a tensor"),
            ({"inputs": torch.zeros(2, 3)}, "one vector of 3 features"),
        ],
    )
    def test_bad_arguments(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            expressibility(QuantumEncodingLayer(n_qubits=3, n_layers=1, **CPU), **kwargs)

    def test_entangling_capability_needs_two_samples(self) -> None:
        with pytest.raises(ValueError, match="n_samples must be ≥ 2"):
            entangling_capability(QuantumEncodingLayer(n_qubits=3, n_layers=1, **CPU), 1)
