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
computed by hand, and the no-loss branch against the noiseless circuit; the
site count ``k`` behind it against gate widths counted by hand.

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

from hqnn_forge.encoding import AnalogRydbergLayer, QuantumEncodingLayer
from hqnn_forge.models import ClassicalBaseline, HybridBinaryClassifier
from hqnn_forge.noise import (
    CHANNELS,
    Position,
    _erasure_kept_fraction,
    _noise_site_count,
    apply_depolarizing_noise,
    apply_shots,
    erasure_kept_fraction,
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
    @pytest.mark.parametrize("p", [0.1, 0.35, 1.0])
    @pytest.mark.parametrize("theta", [0.0, 0.9, math.pi / 2, 2.5, math.pi])
    def test_single_qubit_closed_form(self, theta: float, p: float) -> None:
        # ⟨Z⟩ = cos θ before the loss; (1 − p) cos θ − p after it.  One gate,
        # so "all" and "end" place the same single site.  p = 0 inserts no
        # channel (training_noise_qnode refuses it); the sweep test below
        # checks that p = 0 gives the clean model.
        qnode = _one_qubit()
        for position in ("all", "end"):
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

    def test_amplitude_damping_branches_as_before_the_generalisation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Before #419 the jump was ``torch.rand(shape) < q``: a float32 uniform
        # against q rounded to float32.  At γ = 0.9, q = 0.45 rounds down to
        # r = 0.449999988..., a value torch.rand can return; r < q in float64
        # but not in float32, so the old draw took branch 0 (no jump), and the
        # generalised sampler must too.  Branch 0 applies
        # diag(1, sqrt(1 − γ)) / sqrt(1 − q) to RY(θ)|0⟩:
        # ⟨Z⟩ = (cos²(θ/2) − (1 − γ) sin²(θ/2)) / (1 − q).  The jump would read
        # γ sin²(θ/2) / q = 2 sin²(θ/2) instead.
        gamma, theta = 0.9, 0.9
        q = gamma / 2
        r = torch.tensor(q, dtype=torch.float32)
        assert float(r) < q and not bool(r < q)
        noisy = trajectory_noise_qnode(_one_qubit(), gamma, "end", channel="amplitude_damping")
        monkeypatch.setattr(torch, "rand", lambda *shape, **kwargs: r.clone())
        got = float(noisy(torch.tensor(theta, dtype=torch.float64)))
        c2, s2 = math.cos(theta / 2) ** 2, math.sin(theta / 2) ** 2
        assert got == pytest.approx((c2 - (1 - gamma) * s2) / (1 - q), abs=1e-12)

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


def _three_qubit() -> Any:
    """RX on 0, CNOT(0, 1), Toffoli, CZ(1, 2), CCZ: 1 + 2 + 3 + 2 + 3 = 11 sites at "all"."""
    device = qml.device("default.qubit", wires=3)

    def circuit(theta: torch.Tensor) -> Any:
        qml.RX(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.Toffoli(wires=[0, 1, 2])
        qml.CZ(wires=[1, 2])
        qml.CCZ(wires=[0, 1, 2])
        return [qml.expval(qml.PauliZ(w)) for w in range(3)]

    return qml.QNode(circuit, device, diff_method="backprop", interface="torch")


class TestErasureSiteCount:
    """
    ``_noise_site_count`` and ``_erasure_kept_fraction`` (#702): ``k`` against
    gate widths counted by hand and against the number of uniforms the
    trajectory sampler draws (one per site), and ``(1 − p)^k`` against values
    computed by hand.
    """

    @pytest.mark.parametrize(("position", "k"), [("all", 11), ("end", 3), ("cz", 5)])
    def test_hand_counted_gate_widths(
        self, position: Position, k: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "all": 1 + 2 + 3 + 2 + 3 = 11; "end": 3 wires; "cz": CZ 2 + CCZ 3 = 5.
        theta = torch.tensor(0.4, dtype=torch.float64)
        assert _noise_site_count(_three_qubit(), position, theta) == k
        # The trajectory sampler draws one uniform per site, at the sites the
        # same insertion places.
        noisy = trajectory_noise_qnode(_three_qubit(), 0.2, position, channel="atom_loss")
        real_rand = torch.rand
        calls: list[torch.Tensor] = []

        def recording(*shape: Any, **kwargs: Any) -> torch.Tensor:
            r = real_rand(*shape, **kwargs)
            calls.append(r)
            return r

        monkeypatch.setattr(torch, "rand", recording)
        noisy(theta)
        monkeypatch.undo()
        assert len(calls) == k

    @pytest.mark.parametrize(("position", "k"), [("all", 5), ("end", 2)])
    def test_the_two_qubit_circuit(self, position: Position, k: int) -> None:
        # 1 + 1 + 2 + 1 gate wires at "all", 2 wires at "end", as in
        # TestErasureDerivation.
        theta = torch.tensor(THETA, dtype=torch.float64)
        assert _noise_site_count(_two_qubit(), position, theta) == k

    def test_templates_are_expanded_and_a_barrier_is_a_site(self) -> None:
        # BasicEntanglerLayers, one layer on 3 wires, expands to RX on each
        # wire (3 sites) and the CNOT ring (0,1), (1,2), (2,0) (6 sites); the
        # Barrier on 2 wires adds 2.  As one gate of width 3 it would be 3 + 2.
        device = qml.device("default.qubit", wires=3)

        def circuit(weights: torch.Tensor) -> Any:
            qml.BasicEntanglerLayers(weights, wires=range(3))
            qml.Barrier(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, device, interface="torch")
        weights = torch.tensor([[0.1, 0.2, 0.3]])
        assert _noise_site_count(qnode, "all", weights) == 11
        assert _noise_site_count(qnode, "end", weights) == 3

    def test_end_counts_the_wires_of_the_tape(self) -> None:
        # A 4-wire device, a circuit on wire 0 only: one site, not four.
        device = qml.device("default.qubit", wires=4)

        def circuit(theta: torch.Tensor) -> Any:
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, device, interface="torch")
        assert _noise_site_count(qnode, "end", torch.tensor(0.1)) == 1

    def test_refusals(self) -> None:
        theta = torch.tensor(THETA, dtype=torch.float64)
        with pytest.raises(ValueError, match="no CZ or CCZ"):
            _noise_site_count(_two_qubit(), "cz", theta)
        with pytest.raises(ValueError, match="position must be one of"):
            _noise_site_count(_two_qubit(), "start", theta)  # type: ignore[arg-type]
        layer = AnalogRydbergLayer(2, 2, gamma=0.1, n_steps=2)
        with pytest.raises(ValueError, match="not on its tape"):
            _noise_site_count(layer.qlayer.qnode, "all")

    @pytest.mark.parametrize(
        ("p", "k", "kept"),
        [
            # By hand: 0.9^11 = 0.31381059609, 0.7^5 = 0.16807, 0.75^2 = 0.5625.
            (0.1, 11, 0.31381059609),
            (0.3, 5, 0.16807),
            (0.25, 2, 0.5625),
            (0.0, 7, 1.0),
            (1.0, 3, 0.0),
            (1.0, 0, 1.0),
        ],
    )
    def test_kept_fraction_by_hand(self, p: float, k: int, kept: float) -> None:
        assert _erasure_kept_fraction(p, k) == pytest.approx(kept, abs=1e-12)

    @pytest.mark.parametrize(
        ("p", "k", "match"),
        [
            (-0.1, 3, r"p must lie in \[0, 1.0\]"),
            (1.01, 3, r"p must lie in \[0, 1.0\]"),
            (0.1, -1, "n_sites must be a non-negative int"),
            (0.1, 2.0, "n_sites must be a non-negative int"),
            (0.1, True, "n_sites must be a non-negative int"),
        ],
    )
    def test_kept_fraction_refusals(self, p: float, k: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _erasure_kept_fraction(p, k)


def _classifier(**kwargs: Any) -> HybridBinaryClassifier:
    torch.manual_seed(0)
    return HybridBinaryClassifier(n_input_features=4, n_qubits=3, n_layers=2, **CPU, **kwargs)


#: The site count of each classifier, counted by hand from the documented
#: circuits (3 qubits, 2 layers; AngleEmbedding expands to one RX per wire):
#:
#: * default (CNOT ring + Rot): "all" 3 RX + 2 × (3 CNOT × 2 wires + 3 Rot)
#:   = 3 + 2 × 9 = 21; "end" 3 wires.
#: * ``entangler="rydberg"`` (chain: CZ(0, 1), CZ(1, 2) + Rot): "all"
#:   3 + 2 × (2 CZ × 2 + 3) = 17; "cz" 2 × 2 CZ × 2 = 8.
#: * ``encoding_type="iqp"``: H and RZ per wire (6), each of the 3 pairs as
#:   CNOT, RZ, CNOT (2 + 1 + 2 = 5 each, 15), then the default ansatz (18):
#:   "all" 6 + 15 + 18 = 39.
HAND_COUNTED: list[tuple[dict[str, Any], Position, int]] = [
    ({}, "all", 21),
    ({}, "end", 3),
    ({"entangler": "rydberg"}, "all", 17),
    ({"entangler": "rydberg"}, "cz", 8),
    ({"encoding_type": "iqp"}, "all", 39),
]


class TestErasureKeptFraction:
    """
    ``erasure_kept_fraction`` (#702, option A): ``(1 − p)^k`` per row of
    ``X``, with ``k`` counted by hand from the circuit and, independently,
    as the number of ``ResetError`` channels the density path's
    ``qml.noise.insert`` puts on the tape.  That a kept sample's output and
    gradient are the clean ones is :class:`TestErasureDerivation`.
    """

    X = torch.randn(4, 4, generator=torch.Generator().manual_seed(5))

    @pytest.mark.parametrize(("kwargs", "position", "k"), HAND_COUNTED)
    def test_hand_counted_kept_fraction(
        self, kwargs: dict[str, Any], position: Position, k: int
    ) -> None:
        p = 0.1
        kept = erasure_kept_fraction(_classifier(**kwargs), self.X, p, position=position)
        assert kept.dtype == torch.float64
        expected = torch.full((len(self.X),), (1 - p) ** k, dtype=torch.float64)
        torch.testing.assert_close(kept, expected, atol=1e-15, rtol=0)

    def test_hand_computed_decimals(self) -> None:
        # 0.8^3 = 0.512 at "end"; 0.9^21 = 0.109418989131512359 at "all"
        # (0.9^10 = 0.3486784401, squared 0.12157665459056928801, times 0.9).
        model = _classifier()
        end = erasure_kept_fraction(model, self.X, 0.2, position="end")
        every = erasure_kept_fraction(model, self.X, 0.1, position="all")
        torch.testing.assert_close(end, torch.full((4,), 0.512, dtype=torch.float64))
        torch.testing.assert_close(
            every, torch.full((4,), 0.109418989131512359, dtype=torch.float64)
        )

    @pytest.mark.parametrize(("kwargs", "position", "k"), HAND_COUNTED)
    def test_k_is_the_number_of_channels_the_insertion_places(
        self, kwargs: dict[str, Any], position: Position, k: int
    ) -> None:
        # The independent route: open the density path's atom-loss block, take
        # the noisy QNode it put on the layer, and count the single-wire
        # ResetError channels on its tape for each row of the circuit inputs.
        # erasure_kept_fraction is called inside the same block, so it also
        # has to find the layer's original QNode under the noisy one.
        p = 0.3
        model = _classifier(**kwargs)
        qlayer = model.quantum_layer.qlayer
        inputs: list[torch.Tensor] = []
        handle = qlayer.register_forward_pre_hook(lambda _m, args: inputs.append(args[0]))
        try:
            model.predict_proba(self.X)
        finally:
            handle.remove()
        weights = dict(qlayer.qnode_weights.items())
        with apply_depolarizing_noise(model, p, position=position, channel="atom_loss"):
            channels = [
                [
                    op
                    for op in qml.workflow.construct_tape(qlayer.qnode)(row, **weights).operations
                    if isinstance(op, qml.ResetError)
                ]
                for row in inputs[0]
            ]
            inside = erasure_kept_fraction(model, self.X, p, position=position)
        assert all(len(op.wires) == 1 for row in channels for op in row)
        assert [len(row) for row in channels] == [k] * len(self.X)
        outside = erasure_kept_fraction(model, self.X, p, position=position)
        expected = torch.tensor([(1 - p) ** len(row) for row in channels], dtype=torch.float64)
        torch.testing.assert_close(outside, expected, atol=1e-15, rtol=0)
        torch.testing.assert_close(inside, expected, atol=1e-15, rtol=0)

    def test_inside_apply_shots(self) -> None:
        model = _classifier()
        with apply_shots(model, 50):
            kept = erasure_kept_fraction(model, self.X, 0.1, position="all")
        torch.testing.assert_close(kept, torch.full((4,), 0.9**21, dtype=torch.float64))

    @pytest.mark.parametrize("position", ["all", "end"])
    def test_limits(self, position: Position) -> None:
        # p = 0 keeps everything, p = 1 nothing (k ≥ 1 on every layer, so
        # k = 0, which keeps everything at any p, is not reachable here; the
        # helper's k = 0 is TestErasureSiteCount.test_kept_fraction_by_hand).
        model = _classifier()
        none_lost = erasure_kept_fraction(model, self.X, 0.0, position=position)
        all_lost = erasure_kept_fraction(model, self.X, 1.0, position=position)
        assert torch.equal(none_lost, torch.ones(4, dtype=torch.float64))
        assert torch.equal(all_lost, torch.zeros(4, dtype=torch.float64))

    def test_dephased_analog_layer_at_the_end(self) -> None:
        # Its circuit prepares ρ(T) on 2 wires, one sample at a time: k = 2.
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            n_input_features=4, n_qubits=2, encoding_type="analog", gamma=0.1, n_steps=2
        )
        kept = erasure_kept_fraction(model, self.X[:2], 0.25, position="end")
        torch.testing.assert_close(kept, torch.full((2,), 0.5625, dtype=torch.float64))
        with pytest.raises(ValueError, match="not on its tape"):
            erasure_kept_fraction(model, self.X[:2], 0.25, position="all")

    @pytest.mark.parametrize(
        ("p", "position", "match"),
        [
            (-0.1, "all", r"p must lie in \[0, 1.0\] for 'atom_loss'"),
            (1.01, "end", r"p must lie in \[0, 1.0\] for 'atom_loss'"),
            (0.1, "start", "position must be one of 'all', 'end', 'cz'"),
        ],
    )
    def test_refusals(self, p: float, position: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            erasure_kept_fraction(_classifier(), self.X, p, position=position)

    def test_refuses_cz_on_a_circuit_without_cz(self) -> None:
        with pytest.raises(ValueError, match="no CZ or CCZ"):
            erasure_kept_fraction(_classifier(), self.X, 0.1, position="cz")

    def test_refuses_a_model_without_predict_proba_or_quantum_layer(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, **CPU)
        with pytest.raises(TypeError, match="needs a model with predict_proba"):
            erasure_kept_fraction(layer, self.X, 0.1)
        with pytest.raises(TypeError, match="erasure_kept_fraction expects"):
            erasure_kept_fraction(ClassicalBaseline(4, [3]), self.X, 0.1)


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
