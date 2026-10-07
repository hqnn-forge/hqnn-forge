"""
tests/test_pauli_trajectories.py
================================
Training-time noise by Pauli-trajectory sampling (``noise_method="trajectories"``, #229).

The trajectories output is random, so the checks against the exact
``"density"`` output are statistical: the mean of ``N`` independent draws must
lie within ``Z`` standard errors of the density value, with the standard error
estimated from the draws themselves.  Every test is seeded, so each run is
deterministic; ``Z = 5`` leaves margin for the seed not to matter.  Where the
effect is exact rather than statistical (``position="end"``, where every draw
is ``±`` the clean output), the test is exact.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import warnings
from functools import partial
from typing import Any

import pennylane as qml
import pytest
import torch
import torch.nn as nn

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier
from hqnn_forge.noise import (
    Position,
    _pauli_trajectories,
    apply_depolarizing_noise,
    run_with_training_noise,
    training_noise_qnode,
    trajectory_noise_qnode,
)
from hqnn_forge.utils import load_checkpoint, save_checkpoint

N_QUBITS = 3
P = 0.2
Z = 5.0
FLOAT32_ATOL = 1e-6
CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
# Every encoding layer; the amplitude one takes N_QUBITS features, padded.
LAYERS = [
    pytest.param(QuantumEncodingLayer, id="angle"),
    pytest.param(IQPEncodingLayer, id="iqp"),
    pytest.param(partial(AmplitudeEncodingLayer, n_features=N_QUBITS), id="amplitude"),
    pytest.param(partial(DataReuploadingLayer, trainable_input_scaling=True), id="reuploading"),
]


def _lightning_available() -> bool:
    try:
        qml.device("lightning.qubit", wires=1)
    except Exception:  # noqa: BLE001 - any failure means "not installed"
        return False
    return True


requires_lightning = pytest.mark.skipif(
    not _lightning_available(), reason="pennylane-lightning not installed"
)


def _layer(cls: type = QuantumEncodingLayer, **kwargs: Any) -> Any:
    torch.manual_seed(0)
    return cls(n_qubits=N_QUBITS, n_layers=2, **{**CPU, **kwargs})


def _x() -> torch.Tensor:
    return torch.randn(2, N_QUBITS, generator=torch.Generator().manual_seed(1))


def _density(layer: Any, x: torch.Tensor, position: Position) -> torch.Tensor:
    """The exact channel's output for ``layer``'s weights, one sample at a time."""
    noisy = training_noise_qnode(layer.qlayer.qnode, N_QUBITS, P, position, channel="depolarizing")
    weights = dict(layer.qlayer.qnode_weights)
    prepared = layer.prepare_inputs(x)
    return torch.stack([torch.stack(noisy(xi, **weights)) for xi in prepared]).detach()


def _draws(layer: Any, x: torch.Tensor, n: int) -> torch.Tensor:
    """``n`` independent train-mode outputs per sample, shape ``(n, batch, n_out)``."""
    layer.train()
    with torch.no_grad():
        out = layer(x.repeat(n, 1))
    return out.reshape(n, *x.shape[:1], -1)


def _assert_mean_within(draws: torch.Tensor, exact: torch.Tensor) -> None:
    mean = draws.mean(0)
    se = draws.std(0) / draws.shape[0] ** 0.5
    # FLOAT32_ATOL covers entries whose spread is round-off: a parameter that
    # cannot reach the readout has gradient 0 in every draw, up to float32
    # error of order 1e-8, and a standard error of the same order.
    ok = (mean - exact.to(mean.dtype)).abs() <= Z * se + FLOAT32_ATOL
    assert ok.all(), (mean - exact).abs() / se


PAULIS = {
    "I": torch.eye(2, dtype=torch.complex128),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
}


def _pauli_of(z_bit: int, x_bit: int) -> str:
    """The Pauli that ``RZ(π·z)`` then ``RX(π·x)`` applies, up to a global phase."""
    m = torch.as_tensor(
        qml.matrix(qml.RX(torch.pi * x_bit, 0)) @ qml.matrix(qml.RZ(torch.pi * z_bit, 0)),
        dtype=torch.complex128,
    )
    # |tr(P† M)| = 2 exactly when M = phase · P.
    (name,) = [n for n, p in PAULIS.items() if abs(abs(torch.trace(p.conj().T @ m)) - 2) < 1e-9]
    return name


def _tapes(position: Position, batch: int) -> tuple[qml.tape.QuantumScript, ...]:
    """The layer's batched tape with channels inserted, and with one trajectory drawn."""
    layer = _layer(IQPEncodingLayer)
    x = torch.randn(batch, N_QUBITS, generator=torch.Generator().manual_seed(7))
    tape = qml.tape.make_qscript(layer.qlayer.qnode.func)(x, layer.qlayer.weights)
    (density,), _ = qml.noise.insert(tape, qml.DepolarizingChannel, P, position=position)
    (trajectory,), _ = _pauli_trajectories(tape, p=P, position=position, channel="depolarizing")
    return density, trajectory


def _error_sites(
    density: qml.tape.QuantumScript, trajectory: qml.tape.QuantumScript
) -> list[tuple[qml.operation.Operator, qml.operation.Operator]]:
    """The ``(RZ, RX)`` pair standing where each density channel stands, or fail."""
    ops, i, pairs = trajectory.operations, 0, []
    for op in density.operations:
        if isinstance(op, qml.DepolarizingChannel):
            rz, rx = ops[i], ops[i + 1]
            assert (rz.name, rx.name) == ("RZ", "RX")
            assert rz.wires == rx.wires == op.wires
            pairs.append((rz, rx))
            i += 2
        else:
            assert (ops[i].name, ops[i].wires) == (op.name, op.wires)
            i += 1
    assert i == len(ops)
    return pairs


@pytest.mark.parametrize("position", ["all", "end"])
class TestTheSampler:
    """
    Exact checks on the transformed tape, with no simulation: the error sites
    are the density method's channel sites, each applies I, X, Y or Z, with
    frequencies (1 − p, p/3, p/3, p/3), drawn independently per sample.
    Together these make the trajectory average *be* the depolarizing channel.
    The end-to-end statistical tests below confirm that but cannot see, for
    instance, Y replaced by X on a circuit this small.
    """

    def test_sites_are_the_density_channel_sites(self, position: Position) -> None:
        density, trajectory = _tapes(position, batch=4)
        n_sites = len(_error_sites(density, trajectory))
        gates = [op for op in density.operations if not isinstance(op, qml.DepolarizingChannel)]
        # "end": one site per wire; "all": one per wire of every gate.
        expected = N_QUBITS if position == "end" else sum(len(op.wires) for op in gates)
        assert n_sites == expected > 0

    def test_each_site_applies_a_pauli_with_the_depolarizing_frequencies(
        self, position: Position
    ) -> None:
        batch = 20_000
        torch.manual_seed(8)
        pairs = _error_sites(*_tapes(position, batch=batch))
        z = torch.stack([rz.data[0] for rz, _ in pairs])
        x = torch.stack([rx.data[0] for _, rx in pairs])
        assert z.shape == x.shape == (len(pairs), batch)
        # Angles are 0 or π only: a bit per gate.
        for angles in (z, x):
            assert set(angles.unique().tolist()) <= {0.0, torch.pi}
        z_bits, x_bits = (z == torch.pi).long(), (x == torch.pi).long()
        counts = {name: 0 for name in PAULIS}
        for zb in (0, 1):
            for xb in (0, 1):
                counts[_pauli_of(zb, xb)] += int(((z_bits == zb) & (x_bits == xb)).sum())
        total = z.numel()
        for name, prob in zip("IXYZ", (1 - P, P / 3, P / 3, P / 3), strict=True):
            se = (prob * (1 - prob) / total) ** 0.5
            assert abs(counts[name] / total - prob) <= Z * se, (name, counts)

    def test_samples_draw_independently(self, position: Position) -> None:
        # A draw shared by the whole batch would still average right over
        # forward passes, but correlate every sample within one.  With
        # p = 0.2 and 2000 samples, a site where all agree has probability ~0.
        torch.manual_seed(9)
        for rz, rx in _error_sites(*_tapes(position, batch=2_000)):
            codes = 2 * (rz.data[0] == torch.pi).long() + (rx.data[0] == torch.pi).long()
            assert codes.shape == (2_000,) and codes.unique().numel() == 4

    def test_one_unbatched_sample_draws_scalars(self, position: Position) -> None:
        layer = _layer(IQPEncodingLayer)
        tape = qml.tape.make_qscript(layer.qlayer.qnode.func)(_x()[0], layer.qlayer.weights)
        (density,), _ = qml.noise.insert(tape, qml.DepolarizingChannel, P, position=position)
        (trajectory,), _ = _pauli_trajectories(
            tape, p=P, position=position, channel="depolarizing"
        )
        for rz, rx in _error_sites(density, trajectory):
            assert rz.data[0].shape == rx.data[0].shape == ()


@pytest.mark.parametrize("cls", LAYERS)
@pytest.mark.parametrize("position", ["all", "end"])
class TestMatchesTheDensityMatrix:
    def test_mean_output(self, cls: type, position: Position) -> None:
        layer = _layer(cls, noise_level=P, noise_position=position, noise_method="trajectories")
        x = _x()
        torch.manual_seed(2)
        _assert_mean_within(_draws(layer, x, 3000), _density(layer, x, position))

    def test_mean_gradient(self, cls: type, position: Position) -> None:
        # The loss gradient of one draw is an unbiased estimate of the density
        # gradient.  Chunks of draws give independent gradient estimates, whose
        # spread is the standard error of their mean.
        layer = _layer(cls, noise_level=P, noise_position=position, noise_method="trajectories")
        layer.train()
        x = _x()
        density = training_noise_qnode(
            layer.qlayer.qnode, N_QUBITS, P, position, channel="depolarizing"
        )
        w = layer.qlayer.weights
        weights = dict(layer.qlayer.qnode_weights)
        prepared = layer.prepare_inputs(x)
        total = torch.stack([torch.stack(density(xi, **weights)).sum() for xi in prepared]).sum()
        exact = torch.autograd.grad(total, w)[0].detach()
        torch.manual_seed(3)
        chunks, per_chunk = 20, 150
        grads = []
        for _ in range(chunks):
            out = layer(x.repeat(per_chunk, 1))
            (g,) = torch.autograd.grad(out.sum() / per_chunk, w)
            grads.append(g.detach())
        _assert_mean_within(torch.stack(grads), exact)


@pytest.mark.parametrize("cls", LAYERS)
class TestEndPositionIsExact:
    def test_every_draw_is_plus_or_minus_the_clean_output(self, cls: type) -> None:
        # With one channel per wire before measurement, an X or Y error flips
        # that wire's ⟨Z⟩ and I or Z leaves it: each draw is exactly ±clean,
        # flipped with probability 2p/3, so the mean is (1 − 4p/3)·clean.
        layer = _layer(cls, noise_level=P, noise_position="end", noise_method="trajectories")
        x = _x()
        layer.eval()
        with torch.no_grad():
            clean = layer(x)
        torch.manual_seed(4)
        n = 4000
        draws = _draws(layer, x, n)
        torch.testing.assert_close(draws.abs(), clean.abs().expand_as(draws))
        flipped = (torch.sign(draws) != torch.sign(clean)).double().mean().item()
        rate, se = 2 * P / 3, (2 * P / 3 * (1 - 2 * P / 3) / draws.numel()) ** 0.5
        assert abs(flipped - rate) <= Z * se
        _assert_mean_within(draws, (1 - 4 * P / 3) * clean)


class TestBehaviour:
    def test_eval_mode_is_the_noiseless_layer(self) -> None:
        noisy = _layer(noise_level=P, noise_method="trajectories")
        clean = _layer()
        noisy.eval()
        x = _x()
        with torch.no_grad():
            torch.testing.assert_close(noisy(x), clean(x), rtol=0, atol=0)

    def test_train_mode_draws_afresh_and_the_seed_reproduces_it(self) -> None:
        layer = _layer(noise_level=0.5, noise_method="trajectories")
        layer.train()
        x = _x().repeat(4, 1)
        with torch.no_grad():
            torch.manual_seed(5)
            first = layer(x)
            second = layer(x)
            torch.manual_seed(5)
            again = layer(x)
        assert not torch.equal(first, second)
        torch.testing.assert_close(again, first, rtol=0, atol=0)

    def test_zero_noise_builds_nothing(self) -> None:
        layer = _layer(noise_level=0.0, noise_method="trajectories")
        assert layer._training_noise_qnode is None

    def test_n_trajectories_averages_that_many_draws(self) -> None:
        # Same seed, same draw order: k draws per sample inside the layer are
        # the k repeated rows run by hand, averaged.
        k = 4
        layer = _layer(noise_level=P, noise_method="trajectories", noise_trajectories=k)
        layer.train()
        x = _x()
        with torch.no_grad():
            torch.manual_seed(6)
            averaged = layer(x)
            torch.manual_seed(6)
            rows = run_with_training_noise(
                layer.qlayer, layer._training_noise_qnode, x.repeat_interleave(k, dim=0)
            )
        assert averaged.shape == (2, N_QUBITS)
        torch.testing.assert_close(averaged, rows.reshape(2, k, -1).mean(1), rtol=0, atol=0)

    def test_n_trajectories_on_one_unbatched_sample(self) -> None:
        layer = _layer(noise_level=P, noise_method="trajectories", noise_trajectories=3)
        layer.train()
        assert layer(_x()[0]).shape == (N_QUBITS,)

    def test_post_hoc_wrapper_wins_in_train_mode(self) -> None:
        layer = _layer(noise_level=0.5, noise_method="trajectories")
        layer.train()
        x = _x()
        with torch.no_grad(), apply_depolarizing_noise(layer, P, position="end"):
            inside = layer(x)
        torch.testing.assert_close(inside, _density(layer, x, "end").to(inside.dtype))

    def test_extra_repr(self) -> None:
        text = repr(_layer(noise_level=P, noise_method="trajectories", noise_trajectories=2))
        assert "noise_method='trajectories', noise_trajectories=2" in text
        assert "noise_method" not in repr(_layer(noise_level=P))

    def test_trajectory_qnode_rejects_zero(self) -> None:
        layer = _layer()
        with pytest.raises(ValueError, match="needs p > 0"):
            trajectory_noise_qnode(layer.qlayer.qnode, 0.0, channel="depolarizing")


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(noise_method="kraus"), "noise_method must be 'density' or 'trajectories'"),
            (dict(noise_method="trajectories", noise_trajectories=0), "positive int"),
            (dict(noise_method="trajectories", noise_trajectories=True), "positive int"),
            (dict(noise_method="trajectories", noise_trajectories=2.0), "positive int"),
            (dict(noise_trajectories=2), "needs noise_method='trajectories'"),
        ],
    )
    @pytest.mark.parametrize("noise_level", [0.0, P])
    def test_rejected_at_construction(
        self, kwargs: dict[str, object], match: str, noise_level: float
    ) -> None:
        # Also at noise_level=0: a bad option must not wait for the day the
        # noise is switched on.
        with pytest.raises(ValueError, match=match):
            _layer(QuantumEncodingLayer, noise_level=noise_level, **kwargs)

    def test_classifier_validates_too(self) -> None:
        with pytest.raises(ValueError, match="noise_method must be"):
            HybridBinaryClassifier(
                n_input_features=4,
                n_qubits=N_QUBITS,
                noise_method="kraus",  # type: ignore[arg-type]
                **CPU,
            )

    def test_no_memory_warning_for_trajectories(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            QuantumEncodingLayer(n_qubits=8, noise_level=P, noise_method="trajectories", **CPU)

    def test_the_density_warning_points_at_trajectories(self) -> None:
        with pytest.warns(RuntimeWarning, match="noise_method='trajectories'"):
            QuantumEncodingLayer(n_qubits=8, noise_level=P, **CPU)


@pytest.mark.parametrize("cls", [HybridBinaryClassifier, ParallelHybridClassifier])
class TestClassifiers:
    def test_options_reach_the_quantum_layer_and_the_config(self, cls: type) -> None:
        model = cls(
            n_input_features=4,
            n_qubits=N_QUBITS,
            n_layers=1,
            noise_level=P,
            noise_method="trajectories",
            noise_trajectories=2,
            **CPU,
        )
        assert model.quantum_layer.noise_method == "trajectories"
        assert model.quantum_layer.noise_trajectories == 2
        config = model.get_config()
        assert config["noise_method"] == "trajectories" and config["noise_trajectories"] == 2

    def test_checkpoint_round_trip(self, cls: type, tmp_path: object) -> None:
        torch.manual_seed(0)
        model = cls(
            n_input_features=4,
            n_qubits=N_QUBITS,
            n_layers=1,
            noise_level=P,
            noise_method="trajectories",
            **CPU,
        )
        path = tmp_path / "m.pt"  # type: ignore[operator]
        save_checkpoint(model, path)
        loaded = load_checkpoint(path)
        assert loaded.get_config() == model.get_config()
        x = torch.randn(3, 4)
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model.eval()(x), rtol=0, atol=0)

    def test_a_training_step_reduces_the_loss_on_average(self, cls: type) -> None:
        torch.manual_seed(0)
        model = cls(
            n_input_features=4,
            n_qubits=N_QUBITS,
            n_layers=1,
            noise_level=0.05,
            noise_method="trajectories",
            **CPU,
        )
        x = torch.randn(32, 4, generator=torch.Generator().manual_seed(1))
        y = (x[:, 0] > 0).float()
        loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.05)

        def eval_loss() -> float:
            model.eval()
            with torch.no_grad():
                return float(loss_fn(model(x).squeeze(-1), y))

        before = eval_loss()
        model.train()
        for _ in range(30):
            opt.zero_grad()
            loss_fn(model(x).squeeze(-1), y).backward()
            opt.step()
        assert eval_loss() < before


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        # The library default since #349: "auto" is default.qubit/backprop at 8 qubits.
        ("", ("default.qubit", "backprop")),
        # The default before #349, and what "auto" picks above 12 qubits.
        pytest.param(
            ', device_name="lightning.qubit", diff_method="adjoint"',
            ("lightning.qubit", "adjoint"),
            marks=requires_lightning,
        ),
    ],
)
def test_trajectories_train_within_the_noiseless_memory(
    backend: str, expected: tuple[str, str]
) -> None:
    # The point of #229: 8 qubits, 2 layers, noise after every gate, batch
    # 64.  The density method peaked at +2.7 GB here; trajectories measured
    # +34 MB on lightning with adjoint (+18 MB noiseless) and +23 MB on
    # default.qubit with backprop.  A fresh interpreter so the peak is this
    # step's, not the test session's.
    script = textwrap.dedent(
        f"""
        import json, resource, torch
        from hqnn_forge.encoding import QuantumEncodingLayer
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(
            n_qubits=8, n_layers=2, noise_level=0.1, noise_method="trajectories"{backend}
        )
        x = torch.randn(64, 8)
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        layer.train()
        layer(x).sum().backward()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(json.dumps({{
            "mb": (peak - base) / 1024,
            "grad": bool(layer.qlayer.weights.grad.abs().sum() > 0),
            "backend": [layer.qlayer.qnode.device.name, str(layer.qlayer.qnode.diff_method)],
        }}))
        """
    )
    pytest.importorskip("resource")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=300
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert tuple(report["backend"]) == expected and report["grad"]
    assert report["mb"] < 300, report
