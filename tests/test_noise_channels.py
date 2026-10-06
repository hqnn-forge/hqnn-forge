"""
tests/test_noise_channels.py
============================
Amplitude damping, phase damping, bit flip and phase flip beside the
depolarizing channel, post hoc and for training (#313).

At ``position="end"`` every channel's effect on a ⟨Z⟩ readout is known in
closed form (the table in ``hqnn_forge.noise``), so those checks are exact.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.noise import (
    CHANNELS,
    Channel,
    Position,
    apply_depolarizing_noise,
    noise_sweep,
    training_noise_qnode,
    trajectory_noise_qnode,
)
from hqnn_forge.utils import load_checkpoint, save_checkpoint

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
Z = 5.0

END_EFFECT = {
    "depolarizing": lambda z, p: (1 - 4 * p / 3) * z,
    "amplitude_damping": lambda z, p: (1 - p) * z + p,
    "phase_damping": lambda z, p: z,
    "bit_flip": lambda z, p: (1 - 2 * p) * z,
    "phase_flip": lambda z, p: z,
}
CHANNEL_NAMES: list[Channel] = [
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "bit_flip",
    "phase_flip",
]


def _layer(**kwargs: Any) -> Any:
    torch.manual_seed(0)
    return QuantumEncodingLayer(n_qubits=3, n_layers=2, **{**CPU, **kwargs})


def _qnode(device_name: str, diff_method: str | None, shots: int | None = None) -> qml.QNode:
    """A one-qubit QNode built directly, as a caller outside the layers would."""
    device = qml.device(device_name, wires=1)

    def circuit(theta: torch.Tensor) -> Any:
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, device, diff_method=diff_method, interface="torch")
    return qnode if shots is None else qml.set_shots(qnode, shots=shots)


def _x() -> torch.Tensor:
    return torch.rand(4, 3, generator=torch.Generator().manual_seed(1)) * 4 - 2


def test_every_channel_is_tabulated() -> None:
    assert set(CHANNELS) == set(END_EFFECT)


@pytest.mark.parametrize("channel", CHANNEL_NAMES)
class TestPostHoc:
    def test_end_effect_on_every_readout(self, channel: Channel) -> None:
        layer = _layer()
        x = _x()
        p = 0.3
        with torch.no_grad():
            clean = layer(x)
            with apply_depolarizing_noise(layer, p, position="end", channel=channel):
                noisy = layer(x)
        torch.testing.assert_close(noisy, END_EFFECT[channel](clean, p), atol=1e-5, rtol=0)

    def test_gate_noise_moves_every_channel(self, channel: Channel) -> None:
        # With channels after every gate even the dephasing ones act, through
        # the gates that follow them.
        layer = _layer()
        x = _x()
        with torch.no_grad():
            clean = layer(x)
            with apply_depolarizing_noise(layer, 0.3, position="all", channel=channel):
                noisy = layer(x)
        assert (noisy - clean).abs().max() > 1e-2

    def test_sweep_follows_the_end_formula(self, channel: Channel) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(n_input_features=4, n_qubits=3, n_layers=1, **CPU)
        x = torch.randn(6, 4)
        points = noise_sweep(model, x, [0.0, 0.2, 0.5], position="end", channel=channel)
        clean = points[0].probabilities
        assert [pt.p for pt in points] == [0.0, 0.2, 0.5]
        if channel in ("phase_damping", "phase_flip"):
            for pt in points:
                torch.testing.assert_close(pt.probabilities, clean, atol=1e-5, rtol=0)
        else:
            assert (points[2].probabilities - clean).abs().max() > 1e-3


@pytest.mark.parametrize("channel", CHANNEL_NAMES)
def test_training_noise_density_has_the_end_effect(channel: Channel) -> None:
    p = 0.2
    noisy = _layer(noise_level=p, noise_position="end", noise_channel=channel)
    clean = _layer()
    x = _x()
    with torch.no_grad():
        noisy.train()
        torch.testing.assert_close(noisy(x), END_EFFECT[channel](clean(x), p), atol=1e-5, rtol=0)
        noisy.eval()
        torch.testing.assert_close(noisy(x), clean(x), rtol=0, atol=0)


class TestTrajectories:
    @pytest.mark.parametrize("channel", CHANNEL_NAMES)
    def test_mean_matches_the_density_channel(self, channel: Channel) -> None:
        p, draws = 0.2, 3000
        layer = _layer(noise_level=p, noise_channel=channel, noise_method="trajectories")
        x = _x()[:2]
        density = training_noise_qnode(layer.qlayer.qnode, 3, p, "all", channel)
        with torch.no_grad():
            exact = torch.stack(
                [torch.stack(density(xi, layer.qlayer.weights)) for xi in x]
            ).float()
            layer.train()
            torch.manual_seed(2)
            runs = layer(x.repeat(draws, 1)).reshape(draws, 2, -1).double()
        se = runs.std(0) / math.sqrt(draws)
        assert ((runs.mean(0) - exact).abs() <= Z * se + 1e-6).all()

    def test_bit_flip_at_the_end_is_plus_or_minus_clean(self) -> None:
        p = 0.25
        layer = _layer(
            noise_level=p,
            noise_position="end",
            noise_channel="bit_flip",
            noise_method="trajectories",
        )
        x = _x()
        with torch.no_grad():
            layer.eval()
            clean = layer(x)
            layer.train()
            torch.manual_seed(3)
            runs = torch.stack([layer(x) for _ in range(400)])
        torch.testing.assert_close(runs.abs(), clean.abs().expand_as(runs))
        rate = (torch.sign(runs) != torch.sign(clean)).double().mean().item()
        assert abs(rate - p) <= Z * math.sqrt(p * (1 - p) / runs.numel())

    def test_phase_flip_at_the_end_changes_nothing(self) -> None:
        layer = _layer(
            noise_level=0.4,
            noise_position="end",
            noise_channel="phase_flip",
            noise_method="trajectories",
        )
        x = _x()
        with torch.no_grad():
            layer.eval()
            clean = layer(x)
            layer.train()
            for _ in range(20):
                torch.testing.assert_close(layer(x), clean, atol=1e-6, rtol=0)


class TestDampingTrajectories:
    """#357: phase damping as a phase flip, amplitude damping by weighted Kraus branches."""

    def test_phase_damping_is_the_mapped_phase_flip(self) -> None:
        gamma = 0.3
        layer = _layer()
        flip = (1 - math.sqrt(1 - gamma)) / 2
        damping = training_noise_qnode(layer.qlayer.qnode, 3, gamma, "all", "phase_damping")
        flipping = training_noise_qnode(layer.qlayer.qnode, 3, flip, "all", "phase_flip")
        with torch.no_grad():
            for xi in _x():
                a = torch.stack(damping(xi, layer.qlayer.weights))
                b = torch.stack(flipping(xi, layer.qlayer.weights))
                torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
        assert CHANNELS["phase_damping"].paulis is not None
        assert CHANNELS["phase_damping"].paulis(gamma) == (1 - flip, 0.0, 0.0, flip)

    @pytest.mark.parametrize("position", ["all", "end"])
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
    def test_every_branch_weighted_is_the_density_channel_exactly(
        self, diff_method: str, position: Position, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force every combination of branches at once, one per batch row, and
        # weight each by its probability: the sum must be the channel's output
        # and gradient to rounding, not merely within sampling error.
        gamma = 0.3
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(
            n_qubits=2,
            n_layers=1,
            device_name="default.qubit",
            diff_method=diff_method,  # type: ignore[arg-type]
        )
        layer.double()
        x = torch.tensor([0.4, -1.1], dtype=torch.float64)
        weights = layer.qlayer.weights
        noisy = trajectory_noise_qnode(layer.qlayer.qnode, gamma, position, "amplitude_damping")

        real_rand = torch.rand
        calls: list[int] = []

        def count(*shape: Any, **kwargs: Any) -> torch.Tensor:
            calls.append(0)
            return real_rand(*shape, **kwargs)

        monkeypatch.setattr(torch, "rand", count)
        noisy(x, weights)
        n_sites = len(calls)
        # At the end there is one site per qubit; after every gate, more.
        assert n_sites == 2 if position == "end" else n_sites >= 6

        combos = torch.tensor(list(itertools.product([0, 1], repeat=n_sites)))
        q = gamma / 2
        jump = torch.tensor(q, dtype=torch.float64)
        prob = torch.where(combos == 1, jump, 1 - jump).prod(1)
        drawn = iter(range(n_sites * len(combos)))

        def forced(*shape: Any, **kwargs: Any) -> torch.Tensor:
            # 0 < q draws the jump (branch 1), 1 > q the no-jump branch.  With
            # backprop the batch is one tape and each site draws a column; the
            # other methods split it into one tape per sample first, whose
            # sites then draw one entry each, sample by sample.
            k = next(drawn)
            if shape and shape[0] == (len(combos),):
                return 1.0 - combos[:, k].double()
            return 1.0 - combos[k // n_sites, k % n_sites].double()

        monkeypatch.setattr(torch, "rand", forced)
        weights.grad = None
        out = torch.stack(noisy(x.expand(len(combos), 2), weights), -1)
        estimate = (prob[:, None] * out).sum(0)
        estimate.sum().backward()
        grad = weights.grad.clone()
        monkeypatch.setattr(torch, "rand", real_rand)

        density = training_noise_qnode(layer.qlayer.qnode, 2, gamma, position, "amplitude_damping")
        weights.grad = None
        exact = torch.stack(density(x, weights))
        exact.sum().backward()
        torch.testing.assert_close(estimate, exact.to(estimate.dtype), atol=1e-12, rtol=0)
        # A finite difference is exact only to its step's truncation error.
        grad_atol = 1e-5 if diff_method == "finite-diff" else 1e-12
        torch.testing.assert_close(grad, weights.grad, atol=grad_atol, rtol=0)

    @pytest.mark.parametrize("noise_level", [0.0, 0.1])
    def test_amplitude_damping_refuses_adjoint(self, noise_level: float) -> None:
        # Refused whatever noise_level is, like every other bad option.
        with pytest.raises(ValueError, match="not known to differentiate correctly"):
            _layer(
                noise_level=noise_level,
                noise_channel="amplitude_damping",
                noise_method="trajectories",
                diff_method="adjoint",
            )

    def test_amplitude_damping_refuses_shots(self) -> None:
        with pytest.raises(ValueError, match="unnormalised"):
            _layer(
                noise_level=0.1,
                noise_channel="amplitude_damping",
                noise_method="trajectories",
                diff_method="parameter-shift",
                shots=100,
            )

    # The same refusals where trajectory_noise_qnode makes them itself, on a
    # QNode no layer constructor has looked at first.
    @pytest.mark.parametrize(
        ("device_name", "diff_method"),
        [
            ("default.qubit", "adjoint"),
            ("lightning.qubit", "adjoint"),
            # PennyLane resolves "best" to adjoint on lightning.
            ("lightning.qubit", "best"),
            ("default.qubit", "spsa"),
        ],
    )
    def test_trajectory_qnode_refuses_other_diff_methods(
        self, device_name: str, diff_method: str
    ) -> None:
        with pytest.raises(ValueError, match="not known to differentiate correctly"):
            trajectory_noise_qnode(
                _qnode(device_name, diff_method), 0.1, "all", "amplitude_damping"
            )

    def test_trajectory_qnode_refuses_shots(self) -> None:
        sampled = _qnode("default.qubit", "parameter-shift", shots=100)
        with pytest.raises(ValueError, match="unnormalised"):
            trajectory_noise_qnode(sampled, 0.1, "all", "amplitude_damping")

    def test_trajectory_qnode_refuses_an_unchecked_device(self) -> None:
        mixed = _qnode("default.mixed", "backprop")
        with pytest.raises(ValueError, match="decomposes QubitUnitary"):
            trajectory_noise_qnode(mixed, 0.1, "all", "amplitude_damping")

    @pytest.mark.parametrize(
        ("device_name", "diff_method", "shots"),
        [
            ("default.qubit", "adjoint", None),
            ("default.qubit", "parameter-shift", 100),
            ("default.mixed", "backprop", None),
        ],
    )
    def test_pauli_channels_have_no_such_restriction(
        self, device_name: str, diff_method: str, shots: int | None
    ) -> None:
        qnode = _qnode(device_name, diff_method, shots=shots)
        for channel in ("depolarizing", "phase_damping"):
            noisy = trajectory_noise_qnode(qnode, 0.1, "all", channel)  # type: ignore[arg-type]
            assert torch.isfinite(noisy(torch.tensor(0.3, dtype=torch.float64)))

    def test_trajectory_qnode_runs_without_a_diff_method(self) -> None:
        # diff_method=None computes no gradient, so there is none to get wrong.
        noisy = trajectory_noise_qnode(
            _qnode("default.qubit", None), 0.1, "all", "amplitude_damping"
        )
        assert torch.isfinite(noisy(torch.tensor(0.3, dtype=torch.float64)))

    def test_a_channel_without_a_sampler_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Neither Pauli probabilities nor Kraus branches: refused by name when
        # the layer is built, whatever noise_level is, not by an assertion in
        # the first training step.
        bare = CHANNELS["amplitude_damping"]._replace(kraus=None)
        monkeypatch.setitem(CHANNELS, "amplitude_damping", bare)
        for noise_level in (0.0, 0.1):
            with pytest.raises(ValueError, match="no trajectory sampler"):
                _layer(
                    noise_level=noise_level,
                    noise_channel="amplitude_damping",
                    noise_method="trajectories",
                )
        with pytest.raises(ValueError, match="no trajectory sampler"):
            trajectory_noise_qnode(
                _qnode("default.qubit", "backprop"), 0.1, "all", "amplitude_damping"
            )
        # The density method does not need one.
        _layer(noise_level=0.1, noise_channel="amplitude_damping")

    def test_phase_damping_runs_under_adjoint(self) -> None:
        # A Pauli channel in disguise: unitary draws, so adjoint is fine.
        layer = _layer(
            noise_level=0.1,
            noise_channel="phase_damping",
            noise_method="trajectories",
            diff_method="adjoint",
        )
        layer.train()
        layer(_x()).sum().backward()
        assert torch.isfinite(layer.qlayer.weights.grad).all()

    def test_classifier_trains_with_amplitude_damping_trajectories(self) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            n_input_features=3,
            n_qubits=3,
            n_layers=1,
            noise_level=0.05,
            noise_channel="amplitude_damping",
            noise_method="trajectories",
            noise_trajectories=4,
            **CPU,
        )
        model.train()
        model(_x()).sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)


class TestValidation:
    def test_ranges_are_per_channel(self) -> None:
        _layer(noise_level=0.9, noise_channel="amplitude_damping")  # a full decay is 1
        with pytest.raises(ValueError, match=r"noise_level must lie in \[0, 0.75\]"):
            _layer(noise_level=0.9)
        with pytest.raises(ValueError, match=r"must lie in \[0, 1.0\] for 'bit_flip'"):
            with apply_depolarizing_noise(_layer(), 1.2, channel="bit_flip"):
                pass

    def test_unknown_channel_is_named(self) -> None:
        with pytest.raises(ValueError, match="noise_channel must be one of"):
            _layer(noise_channel="thermal")
        with pytest.raises(ValueError, match="channel must be one of"):
            model = HybridBinaryClassifier(n_input_features=3, n_qubits=3, n_layers=1, **CPU)
            noise_sweep(model, _x(), [0.1], channel="thermal")  # type: ignore[arg-type]

    def test_repr_names_a_non_default_channel(self) -> None:
        assert "noise_channel='bit_flip'" in repr(
            _layer(noise_level=0.1, noise_channel="bit_flip")
        )
        assert "noise_channel" not in repr(_layer(noise_level=0.1))


@pytest.mark.parametrize(
    "cls, extra",
    [
        (HybridBinaryClassifier, {}),
        (ParallelHybridClassifier, {}),
        (MulticlassHybridClassifier, {"n_classes": 3}),
    ],
)
def test_classifiers_record_and_restore_the_channel(
    cls: type, extra: dict[str, Any], tmp_path: Path
) -> None:
    torch.manual_seed(0)
    model = cls(
        n_input_features=4,
        n_qubits=3,
        n_layers=1,
        noise_level=0.1,
        noise_channel="amplitude_damping",
        **CPU,
        **extra,
    )
    assert model.quantum_layer.noise_channel == "amplitude_damping"
    save_checkpoint(model, tmp_path / "m.pt")
    loaded = load_checkpoint(tmp_path / "m.pt")
    assert loaded.get_config()["noise_channel"] == "amplitude_damping"
    # Weight-safe: a channel can be swapped at load time for fine-tuning.
    swapped: Any = load_checkpoint(tmp_path / "m.pt", noise_channel="bit_flip")
    assert swapped.quantum_layer.noise_channel == "bit_flip"
