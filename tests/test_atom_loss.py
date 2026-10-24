"""
tests/test_atom_loss.py
=======================
Neutral-atom atom loss read out as ``|1⟩`` (``channel="atom_loss"``, #419).

Every expected value here is obtained another way than through the code under
test: the single-qubit closed form ``⟨Z⟩ → (1 − p)⟨Z⟩ − p``, its two-qubit
correlator, a density matrix built by hand in NumPy, an enumeration of every
trajectory branch weighted by its probability, and a Monte Carlo mean within
a stated number of standard errors.  The erasure derivation in the module
docstring (#702) is checked the same way: the kept fraction ``(1 − p)^k``
computed by hand, and the no-loss branch against the noiseless circuit.

Index conventions, as in PennyLane: wire 0 is the most significant bit of a
basis index, ``|0⟩`` has ``Z = +1`` and ``|1⟩`` (the dark state a lost atom
reads as) ``Z = −1``.  A *site* is one (gate, wire) pair at ``position="all"``,
in tape order and, within a gate, in the gate's wire order; at ``"end"`` it is
one wire, in wire order.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import (
    CHANNELS,
    Position,
    apply_depolarizing_noise,
    noise_sweep,
    training_noise_qnode,
    trajectory_noise_qnode,
)

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
#: Monte Carlo tolerance, in standard errors of the mean.  With a fixed seed
#: the check is deterministic; 5 SE would fail by chance with probability
#: about 6e-7 per compared value under a fresh seed.
Z = 5.0


def _one_qubit(device_name: str = "default.qubit", diff_method: str | None = "backprop") -> Any:
    device = qml.device(device_name, wires=1)

    def circuit(theta: torch.Tensor) -> Any:
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    return qml.QNode(circuit, device, diff_method=diff_method, interface="torch")


def _two_qubit(diff_method: str = "backprop") -> Any:
    """RY(a)⊗RY(b), CNOT(0, 1), RX(c) on 1: five sites at ``"all"``."""
    device = qml.device("default.qubit", wires=2)

    def circuit(theta: torch.Tensor) -> Any:
        qml.RY(theta[..., 0], wires=0)
        qml.RY(theta[..., 1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(theta[..., 2], wires=1)
        return (
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        )

    qnode = qml.QNode(circuit, device, diff_method=diff_method, interface="torch")
    # Parameter-shift and finite-diff cannot differentiate a broadcast tape
    # with respect to its broadcast parameters; split it per sample first, as
    # the encoding layers do.
    return qnode if diff_method == "backprop" else qml.transforms.broadcast_expand(qnode)


THETA = (0.7, -1.3, 0.4)


# ---------------------------------------------------------------------------
# Hand-built density matrices (NumPy, independent of PennyLane's channels)
# ---------------------------------------------------------------------------

_I = np.eye(2)
_Z = np.diag([1.0, -1.0])
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def _ry(t: float) -> np.ndarray:
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rx(t: float) -> np.ndarray:
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _on(op: np.ndarray, wire: int) -> np.ndarray:
    """A one-qubit ``op`` on ``wire`` of two (wire 0 most significant)."""
    return np.kron(op, _I) if wire == 0 else np.kron(_I, op)


def _lose(rho: np.ndarray, wire: int, p: float) -> np.ndarray:
    """
    Atom loss read out as |1⟩ on ``wire``, written out as the reset it is:
    ρ → (1 − p) ρ + p · |1⟩⟨1|_wire ⊗ Tr_wire ρ.  The reset part is the sum
    over the two outcomes k of |1⟩⟨k| ρ |k⟩⟨1| on that wire.
    """
    reset = np.zeros_like(rho)
    for k in (0, 1):
        jump = np.zeros((2, 2))
        jump[1, k] = 1.0
        op = _on(jump, wire)
        reset += op @ rho @ op.conj().T
    return (1 - p) * rho + p * reset


def _hand_built(theta: tuple[float, float, float], p: float, position: str) -> np.ndarray:
    """⟨Z0⟩, ⟨Z1⟩, ⟨Z0 Z1⟩ of :func:`_two_qubit` with loss at ``position``."""
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    gates = [
        (_on(_ry(theta[0]), 0), (0,)),
        (_on(_ry(theta[1]), 1), (1,)),
        (_CNOT, (0, 1)),
        (_on(_rx(theta[2]), 1), (1,)),
    ]
    for unitary, wires in gates:
        rho = unitary @ rho @ unitary.conj().T
        if position == "all":
            for w in wires:
                rho = _lose(rho, w, p)
    if position == "end":
        for w in (0, 1):
            rho = _lose(rho, w, p)
    observables = [np.kron(_Z, _I), np.kron(_I, _Z), np.kron(_Z, _Z)]
    return np.array([np.trace(rho @ o).real for o in observables])


# ---------------------------------------------------------------------------
# The density channel
# ---------------------------------------------------------------------------


class TestDensity:
    @pytest.mark.parametrize("p", [0.0, 0.1, 0.35, 1.0])
    @pytest.mark.parametrize("theta", [0.0, 0.9, math.pi / 2, 2.5, math.pi])
    def test_single_qubit_closed_form(self, theta: float, p: float) -> None:
        # ⟨Z⟩ = cos θ before the loss; (1 − p) cos θ − p after it.  One gate,
        # so "all" and "end" place the same single site.
        qnode = _one_qubit()
        for position in ("all", "end"):
            if p == 0.0:
                continue  # training_noise_qnode needs p > 0
            noisy = training_noise_qnode(qnode, 1, p, position, channel="atom_loss")
            got = float(noisy(torch.tensor(theta, dtype=torch.float64)))
            assert got == pytest.approx((1 - p) * math.cos(theta) - p, abs=1e-12)

    @pytest.mark.parametrize("p", [0.05, 0.3, 1.0])
    def test_two_qubit_correlator_at_the_end(self, p: float) -> None:
        # Heisenberg picture, each wire independently: Z_i → (1 − p) Z_i − p, so
        # Z0 Z1 → (1 − p)² Z0 Z1 − p (1 − p) (Z0 + Z1) + p².
        theta = torch.tensor(THETA, dtype=torch.float64)
        z0, z1, zz = (float(v) for v in _two_qubit()(theta))
        noisy = training_noise_qnode(_two_qubit(), 2, p, "end", channel="atom_loss")
        n0, n1, nzz = (float(v) for v in noisy(theta))
        assert n0 == pytest.approx((1 - p) * z0 - p, abs=1e-12)
        assert n1 == pytest.approx((1 - p) * z1 - p, abs=1e-12)
        expected = (1 - p) ** 2 * zz - p * (1 - p) * (z0 + z1) + p**2
        assert nzz == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("position", ["all", "end"])
    @pytest.mark.parametrize("p", [0.1, 0.6])
    def test_matches_a_hand_built_density_matrix(self, position: str, p: float) -> None:
        noisy = training_noise_qnode(_two_qubit(), 2, p, position, channel="atom_loss")  # type: ignore[arg-type]
        got = torch.stack(noisy(torch.tensor(THETA, dtype=torch.float64))).numpy()
        np.testing.assert_allclose(got, _hand_built(THETA, p, position), atol=1e-12, rtol=0)

    def test_is_exactly_reset_error_to_one(self) -> None:
        # The issue's definition: qml.ResetError(p0=0, p1=p) on default.mixed.
        p = 0.27
        op = CHANNELS["atom_loss"].op(p, wires=0)
        assert isinstance(op, qml.ResetError)
        assert [float(v) for v in op.parameters] == [0.0, p]

    def test_a_lost_atom_reads_dark_whatever_its_state(self) -> None:
        # p = 1: every wire reads |1⟩, so every ⟨Z⟩ is −1 and ⟨Z0 Z1⟩ is +1.
        noisy = training_noise_qnode(_two_qubit(), 2, 1.0, "all", channel="atom_loss")
        got = torch.stack(noisy(torch.tensor(THETA, dtype=torch.float64)))
        torch.testing.assert_close(got, torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))

    def test_post_hoc_on_a_layer_and_training_mode_agree_with_the_formula(self) -> None:
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=2, **CPU)
        x = torch.rand(5, 3, generator=torch.Generator().manual_seed(1)) * 4 - 2
        p = 0.15
        with torch.no_grad():
            clean = layer(x)
            with apply_depolarizing_noise(layer, p, position="end", channel="atom_loss"):
                post_hoc = layer(x)
        torch.testing.assert_close(post_hoc, (1 - p) * clean - p, atol=1e-5, rtol=0)
        torch.manual_seed(0)
        trained = QuantumEncodingLayer(
            n_qubits=3,
            n_layers=2,
            noise_level=p,
            noise_position="end",
            noise_channel="atom_loss",
            **CPU,
        )
        with torch.no_grad():
            torch.testing.assert_close(trained.train()(x), (1 - p) * clean - p, atol=1e-5, rtol=0)
            torch.testing.assert_close(trained.eval()(x), clean, atol=0, rtol=0)

    def test_sweep_over_p_loss_follows_the_closed_form(self) -> None:
        # The expected probabilities come from the clean model with its
        # quantum layer's output replaced by (1 − p)⟨Z⟩ − p through a hook.
        torch.manual_seed(0)
        model = HybridBinaryClassifier(n_input_features=4, n_qubits=3, n_layers=2, **CPU)
        x = torch.randn(6, 4, generator=torch.Generator().manual_seed(2))
        ps = [0.0, 0.1, 0.4, 1.0]
        points = noise_sweep(model, x, ps, position="end", channel="atom_loss")
        assert [pt.p for pt in points] == ps
        for pt in points:
            p = float(pt.p)  # type: ignore[arg-type]
            handle = model.quantum_layer.register_forward_hook(
                lambda _m, _i, out, p=p: (1 - p) * out - p
            )
            try:
                expected = model.predict_proba(x)
            finally:
                handle.remove()
            torch.testing.assert_close(pt.probabilities, expected, atol=1e-5, rtol=0)

    def test_gradient_matches_the_closed_form(self) -> None:
        # d/dθ [(1 − p) cos θ − p] = −(1 − p) sin θ.
        p, theta = 0.3, torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
        noisy = training_noise_qnode(_one_qubit(), 1, p, "end", channel="atom_loss")
        noisy(theta).backward()
        assert theta.grad is not None
        assert float(theta.grad) == pytest.approx(-(1 - p) * math.sin(0.8), abs=1e-12)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------


def _forced_branches(combos: torch.Tensor, q1: float, q2: float) -> tuple[torch.Tensor, Any]:
    """
    ``torch.rand`` replacement that makes site ``k`` of row ``r`` draw branch
    ``combos[r, k]`` (0 no loss, 1 the |0⟩ → |1⟩ jump, 2 the |1⟩ → |1⟩ one).

    The sampler takes branch 1 for a uniform below ``q1``, branch 2 below
    ``q1 + q2`` and branch 0 otherwise, so each branch gets a uniform from
    the middle of its interval.
    """
    values = torch.tensor([(1 + q1 + q2) / 2, q1 / 2, q1 + q2 / 2], dtype=torch.float64)
    n_rows, n_sites = combos.shape
    drawn = iter(range(n_rows * n_sites))

    def forced(*shape: Any, **kwargs: Any) -> torch.Tensor:
        # With backprop the batch is one tape and each site draws a column;
        # the other methods split it into one tape per sample, whose sites
        # then draw one entry each, sample by sample.
        k = next(drawn)
        if shape and shape[0] == (n_rows,):
            return values[combos[:, k]]
        return values[combos[k // n_sites, k % n_sites]]

    return values, forced


class TestTrajectories:
    @pytest.mark.parametrize(
        ("position", "diff_method"),
        [
            ("end", "backprop"),
            ("end", "parameter-shift"),
            ("end", "finite-diff"),
            ("all", "backprop"),
            ("all", "parameter-shift"),
        ],
    )
    def test_every_branch_weighted_is_the_density_channel_exactly(
        self, position: Position, diff_method: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Three weighted branches per site (no loss with 1 − p, each jump with
        # p/2).  Forcing every combination once and weighting it by its
        # probability must give the density output and gradient to rounding.
        p = 0.3
        q = (1 - p, p / 2, p / 2)
        n_sites = {"end": 2, "all": 5}[position]
        noisy = trajectory_noise_qnode(_two_qubit(diff_method), p, position, channel="atom_loss")
        real_rand = torch.rand
        calls: list[int] = []

        def count(*shape: Any, **kwargs: Any) -> torch.Tensor:
            calls.append(0)
            return real_rand(*shape, **kwargs)

        monkeypatch.setattr(torch, "rand", count)
        noisy(torch.tensor(THETA, dtype=torch.float64))
        assert len(calls) == n_sites  # one uniform per site

        combos = torch.tensor(list(itertools.product(range(3), repeat=n_sites)))
        prob = torch.tensor(q, dtype=torch.float64)[combos].prod(1)
        assert float(prob.sum()) == pytest.approx(1.0, abs=1e-12)
        _, forced = _forced_branches(combos, q[1], q[2])
        monkeypatch.setattr(torch, "rand", forced)
        theta = torch.tensor(THETA, dtype=torch.float64, requires_grad=True)
        out = torch.stack(noisy(theta.expand(len(combos), 3)), -1)
        estimate = (prob[:, None] * out).sum(0)
        estimate.sum().backward()
        assert theta.grad is not None
        grad = theta.grad.clone()
        monkeypatch.setattr(torch, "rand", real_rand)

        exact_theta = torch.tensor(THETA, dtype=torch.float64, requires_grad=True)
        density = training_noise_qnode(_two_qubit(), 2, p, position, channel="atom_loss")
        exact = torch.stack(density(exact_theta))
        exact.sum().backward()
        torch.testing.assert_close(
            estimate, torch.tensor(_hand_built(THETA, p, position)), atol=1e-12, rtol=0
        )
        torch.testing.assert_close(estimate, exact, atol=1e-12, rtol=0)
        grad_atol = 1e-5 if diff_method == "finite-diff" else 1e-12
        torch.testing.assert_close(grad, exact_theta.grad, atol=grad_atol, rtol=0)

    def test_a_lost_draw_reads_the_weighted_dark_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # One qubit, θ = 0.9, one site.  Branch 1 applies √2 |1⟩⟨0|: the state
        # √2 cos(θ/2) |1⟩, so ⟨Z⟩ = −2 cos²(θ/2).  Branch 2 applies √2 |1⟩⟨1|:
        # ⟨Z⟩ = −2 sin²(θ/2).  Branch 0 is the identity: cos θ.  Weighted by
        # (1 − p, p/2, p/2) they sum to (1 − p) cos θ − p.
        p, theta = 0.4, 0.9
        noisy = trajectory_noise_qnode(_one_qubit(), p, "end", channel="atom_loss")
        combos = torch.tensor([[0], [1], [2]])
        _, forced = _forced_branches(combos, p / 2, p / 2)
        monkeypatch.setattr(torch, "rand", forced)
        out = noisy(torch.full((3,), theta, dtype=torch.float64))
        c2, s2 = math.cos(theta / 2) ** 2, math.sin(theta / 2) ** 2
        torch.testing.assert_close(
            out, torch.tensor([math.cos(theta), -2 * c2, -2 * s2], dtype=torch.float64)
        )

    @pytest.mark.parametrize("position", ["all", "end"])
    def test_monte_carlo_mean_matches_density(self, position: Position) -> None:
        # Unbiased: the mean of 4000 independent draws (fixed seed) lies
        # within Z standard errors of the density value for every readout.
        p, draws = 0.2, 4000
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(
            n_qubits=3,
            n_layers=1,
            noise_level=p,
            noise_position=position,
            noise_channel="atom_loss",
            noise_method="trajectories",
            **CPU,
        )
        x = torch.rand(2, 3, generator=torch.Generator().manual_seed(1)) * 4 - 2
        density = training_noise_qnode(layer.qlayer.qnode, 3, p, position, channel="atom_loss")
        with torch.no_grad():
            exact = torch.stack([torch.stack(density(xi, layer.qlayer.weights)) for xi in x])
            layer.train()
            torch.manual_seed(5)
            runs = layer(x.repeat(draws, 1)).reshape(draws, 2, -1).double()
        se = runs.std(0) / math.sqrt(draws)
        assert (se > 0).all()
        assert ((runs.mean(0) - exact.double()).abs() <= Z * se).all()

    def test_refuses_adjoint_like_amplitude_damping(self) -> None:
        # The jumps are non-unitary weighted branches, so the #357 allow-lists apply.
        with pytest.raises(ValueError, match="not known to differentiate correctly"):
            QuantumEncodingLayer(
                n_qubits=2,
                n_layers=1,
                device_name="default.qubit",
                diff_method="adjoint",
                noise_level=0.0,
                noise_channel="atom_loss",
                noise_method="trajectories",
            )


class TestErasureDerivation:
    """
    The module docstring's erasure derivation (#702): post-selecting on no
    loss at any of the ``k`` sites keeps a fraction ``(1 − p)^k`` of the
    samples, and a kept sample's output and gradient are the noiseless ones.
    """

    @pytest.mark.parametrize("position", ["all", "end"])
    def test_the_no_loss_branch_is_the_noiseless_circuit(
        self, position: Position, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The circuit has 5 sites at "all" (1 + 1 + 2 + 1 gate wires) and 2 at
        # "end".  At p = 0.3, by hand: 0.7^5 = 0.16807 and 0.7^2 = 0.49.
        p = 0.3
        kept = {"all": 0.16807, "end": 0.49}[position]
        n_sites = {"all": 5, "end": 2}[position]
        combos = torch.tensor(list(itertools.product(range(3), repeat=n_sites)))
        prob = torch.tensor((1 - p, p / 2, p / 2), dtype=torch.float64)[combos].prod(1)
        no_loss = (combos == 0).all(1)
        assert int(no_loss.sum()) == 1
        assert float(prob[no_loss].sum()) == pytest.approx(kept, abs=1e-12)

        noisy = trajectory_noise_qnode(_two_qubit(), p, position, channel="atom_loss")
        _, forced = _forced_branches(combos, p / 2, p / 2)
        monkeypatch.setattr(torch, "rand", forced)
        theta = torch.tensor(THETA, dtype=torch.float64, requires_grad=True)
        out = torch.stack(noisy(theta.expand(len(combos), 3)), -1)
        # The renormalised post-selected output: Σ over no-loss rows of
        # prob · out, divided by the kept fraction.
        conditional = (prob[no_loss, None] * out[no_loss]).sum(0) / prob[no_loss].sum()
        conditional.sum().backward()
        assert theta.grad is not None
        monkeypatch.undo()

        clean_theta = torch.tensor(THETA, dtype=torch.float64, requires_grad=True)
        clean = torch.stack(_two_qubit()(clean_theta))
        clean.sum().backward()
        torch.testing.assert_close(
            conditional, torch.tensor(_hand_built(THETA, 0.0, position)), atol=1e-12, rtol=0
        )
        torch.testing.assert_close(conditional, clean, atol=1e-12, rtol=0)
        torch.testing.assert_close(theta.grad, clean_theta.grad, atol=1e-12, rtol=0)

    def test_monte_carlo_kept_fraction_and_conditional_mean(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 4000 draws at "all" (5 sites), p = 0.1.  A draw is flagged when any
        # site took a jump (its uniform is below p).  The kept fraction lies
        # within Z binomial standard errors of 0.9^5 = 0.59049, and the mean
        # output of the kept draws equals the noiseless value: they are the
        # noiseless circuit, so not a Monte Carlo estimate but exact.
        p, draws, n_sites = 0.1, 4000, 5
        noisy = trajectory_noise_qnode(_two_qubit(), p, "all", channel="atom_loss")
        real_rand = torch.rand
        uniforms: list[torch.Tensor] = []

        def recording(*shape: Any, **kwargs: Any) -> torch.Tensor:
            r = real_rand(*shape, **kwargs)
            uniforms.append(r)
            return r

        monkeypatch.setattr(torch, "rand", recording)
        torch.manual_seed(3)
        with torch.no_grad():
            out = torch.stack(noisy(torch.tensor(THETA, dtype=torch.float64).expand(draws, 3)), -1)
        monkeypatch.undo()
        assert len(uniforms) == n_sites
        flagged = (torch.stack(uniforms, 1) < p).any(1)
        kept = 1 - flagged.double().mean()
        expected = 0.9**5
        assert expected == pytest.approx(0.59049, abs=1e-15)
        se = math.sqrt(expected * (1 - expected) / draws)
        assert abs(float(kept) - expected) <= Z * se
        clean = torch.tensor(_hand_built(THETA, 0.0, "all"))
        kept_out = out[~flagged]
        # Every kept draw is the noiseless value to rounding (zero spread), so
        # the conditional mean is held to 1e-12 rather than to Z SE.
        torch.testing.assert_close(kept_out, clean.expand_as(kept_out), atol=1e-12, rtol=0)
        # The flagged draws are not: their mean is far from the clean value,
        # so the post-selection is what makes the difference.
        assert ((out[flagged].mean(0) - clean).abs() > 0.05).any()


class TestValidation:
    @pytest.mark.parametrize("p", [-0.1, 1.01])
    def test_p_loss_outside_the_unit_interval_is_refused(self, p: float) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[0, 1.0\] for 'atom_loss'"):
            training_noise_qnode(_one_qubit(), 1, p, "end", channel="atom_loss")
        with pytest.raises(
            ValueError, match=r"noise_level must lie in \[0, 1.0\] for 'atom_loss'"
        ):
            QuantumEncodingLayer(
                n_qubits=2, n_layers=1, noise_level=p, noise_channel="atom_loss", **CPU
            )
