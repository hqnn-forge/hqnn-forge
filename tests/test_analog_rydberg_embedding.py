"""
tests/test_analog_rydberg_embedding.py
======================================
``AnalogRydbergLayer`` (#420) against values obtained another way: the
closed-form single-atom Rabi oscillation, the blockade bound, the product of
single-atom solutions for far-apart atoms, ``scipy.linalg.expm`` of the same
piecewise Hamiltonian built here from Kronecker products of Pauli matrices,
and central finite differences for the gradients.

Every layer is converted to ``float64`` (``.double()``) before a pulse is set,
so the pulse that runs is the requested one to rounding and the comparisons
can be tight.
"""

from __future__ import annotations

import math
from functools import reduce

import numpy as np
import pytest
import scipy.linalg
import torch

from hqnn_forge.diagnostics import circuit_summary, gradient_variance
from hqnn_forge.encoding import AnalogRydbergLayer
from hqnn_forge.noise import apply_depolarizing_noise, apply_readout_error, apply_shots
from hqnn_forge.rydberg import DEFAULT_C6, MAX_ATOMS, AtomRegister

F64 = torch.float64
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1j], [1j, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
N_OP = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)  # n = |r⟩⟨r|, |r⟩ = |1⟩
EYE = np.eye(2, dtype=complex)


def _layer(n_qubits: int, n_segments: int, **kwargs: object) -> AnalogRydbergLayer:
    torch.manual_seed(0)
    return AnalogRydbergLayer(n_qubits, n_segments, **kwargs).double()  # type: ignore[arg-type]


def _on(op: np.ndarray, i: int, n: int) -> np.ndarray:
    """``op`` on tensor factor ``i`` of ``n`` (atom 0 leftmost, the most significant bit)."""
    return reduce(np.kron, [op if j == i else EYE for j in range(n)])


def _reference_hamiltonian(
    positions: np.ndarray,
    c6: float,
    omega: float,
    phase: float,
    detunings: np.ndarray,
    interactions: bool = True,
) -> np.ndarray:
    """``Ω/2 Σ (cos φ X_i − sin φ Y_i) − Σ Δ_i n_i + Σ_{i<j} C6/r^6 n_i n_j`` by Kronecker products."""
    n = len(positions)
    h = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        drive = math.cos(phase) * X - math.sin(phase) * Y
        h += omega / 2 * _on(drive, i, n) - detunings[i] * _on(N_OP, i, n)
    if interactions:
        for i in range(n):
            for j in range(i + 1, n):
                v = c6 / np.linalg.norm(positions[i] - positions[j]) ** 6
                h += v * _on(N_OP, i, n) @ _on(N_OP, j, n)
    return h


def _reference_state(layer: AnalogRydbergLayer, x: np.ndarray) -> np.ndarray:
    """``ψ = e^(−i H_K τ) ⋯ e^(−i H_1 τ) |g…g⟩`` with ``scipy.linalg.expm``, from the layer's pulse."""
    pulse = layer.pulse().detach().numpy()
    local = layer.local_detuning_max / (1 + np.exp(-x))  # Δ_max,loc · σ(x_i)
    positions = layer.register.positions.numpy()
    psi = np.zeros(2**layer.n_qubits, dtype=complex)
    psi[0] = 1.0
    for omega, detuning, phase in pulse:
        h = _reference_hamiltonian(
            positions, layer.c6, omega, phase, detuning + local, layer.interactions
        )
        psi = scipy.linalg.expm(-1j * h * layer.segment_duration) @ psi
    return psi


def _z_expectations(psi: np.ndarray) -> np.ndarray:
    n = int(math.log2(len(psi)))
    return np.array([np.real(psi.conj() @ _on(Z, i, n) @ psi) for i in range(n)])


def _single_atom_propagator(omega: float, delta: float, phase: float, tau: float) -> np.ndarray:
    """
    Closed form of ``e^(−i H_1 τ)``, ``H_1 = Ω/2 (cos φ X − sin φ Y) − Δ n``.

    ``−Δ n = −Δ/2 + Δ Z/2``, so ``H_1 = −Δ/2 + A/2`` with
    ``A = Ω (cos φ X − sin φ Y) + Δ Z`` and ``A² = (Ω² + Δ²) 1``.
    """
    omega_eff = math.hypot(omega, delta)
    a = omega * (math.cos(phase) * X - math.sin(phase) * Y) + delta * Z
    return np.exp(1j * delta * tau / 2) * (
        math.cos(omega_eff * tau / 2) * EYE - 1j * math.sin(omega_eff * tau / 2) * a / omega_eff
    )


# ---------------------------------------------------------------------------
# Closed-form limits
# ---------------------------------------------------------------------------


class TestRabiOscillation:
    @pytest.mark.parametrize("n_segments", [1, 3])
    @pytest.mark.parametrize("omega_t", [0.4, math.pi / 2, math.pi, 2.5 * math.pi])
    def test_one_resonant_atom_gives_sin_squared(self, n_segments: int, omega_t: float) -> None:
        """``⟨n⟩ = sin²(Ωt/2)``, i.e. ``⟨Z⟩ = cos(Ωt)``, at total detuning 0."""
        omega = 3.0
        layer = _layer(
            1, n_segments, omega_max=6.0, segment_duration=omega_t / (omega * n_segments)
        )
        x = torch.tensor([[0.25]], dtype=F64)
        local = layer.local_detunings(x)[0, 0].item()
        layer.set_pulse(omega=omega, detuning=-local, phase=0.0)  # Δ_k + Δ_loc = 0
        n_expected = math.sin(omega_t / 2) ** 2
        assert layer(x).item() == pytest.approx(1 - 2 * n_expected, abs=1e-12)

    def test_a_constant_phase_does_not_change_the_readout(self) -> None:
        """One segment from ``|g⟩``: φ is removed by ``e^(−iφ n)``, which commutes with the readout."""
        layer = _layer(1, 1, omega_max=6.0, segment_duration=0.7)
        x = torch.tensor([[0.0]], dtype=F64)
        layer.set_pulse(omega=3.0, detuning=0.4, phase=0.0)
        before = layer(x).item()
        layer.set_pulse(omega=3.0, detuning=0.4, phase=1.9)
        assert layer(x).item() == pytest.approx(before, abs=1e-13)

    def test_a_detuned_atom_gives_the_detuned_rabi_formula(self) -> None:
        """``⟨n⟩ = Ω²/Ω_eff² · sin²(Ω_eff t/2)``, with the local and global detunings added."""
        omega, global_delta, t = 2.0, 0.7, 1.3
        layer = _layer(1, 2, omega_max=4.0, segment_duration=t / 2)
        x = torch.tensor([[-0.4]], dtype=F64)
        layer.set_pulse(omega=omega, detuning=global_delta, phase=0.0)
        delta = global_delta + layer.local_detuning_max / (1 + math.exp(0.4))
        omega_eff = math.hypot(omega, delta)
        n_expected = omega**2 / omega_eff**2 * math.sin(omega_eff * t / 2) ** 2
        assert layer(x).item() == pytest.approx(1 - 2 * n_expected, abs=1e-12)


class TestBlockade:
    @pytest.mark.parametrize("v_over_omega", [10.0, 100.0])
    def test_the_doubly_excited_population_stays_below_omega_over_v_squared(
        self, v_over_omega: float
    ) -> None:
        """Two resonant atoms with ``V ≫ Ω``: ``P(|rr⟩) ≤ (Ω/V)²`` over ``Ωt ≤ 20``."""
        omega = 4 * math.pi
        spacing = (DEFAULT_C6 / (v_over_omega * omega)) ** (1 / 6)
        n_segments = 40
        layer = _layer(
            2,
            n_segments,
            register=AtomRegister.chain(2, spacing),
            omega_max=2 * omega,
            segment_duration=20 / omega / n_segments,
        )
        x = torch.zeros(1, 2, dtype=F64)
        local = layer.local_detunings(x)[0, 0].item()
        layer.set_pulse(omega=omega, detuning=-local, phase=0.0)
        # ψ after every segment: the bound must hold at each time, not only at the end.
        psi = torch.zeros(4, dtype=torch.complex128)
        psi[0] = 1.0
        largest = 0.0
        sum_error = 0.0
        for k, u in enumerate(layer.unitaries(x)[:, 0]):
            psi = u @ psi
            largest = max(largest, abs(psi[3].item()) ** 2)
            t = (k + 1) * layer.segment_duration
            n_total = (abs(psi[1]) ** 2 + abs(psi[2]) ** 2 + 2 * abs(psi[3]) ** 2).item()
            # The blockaded pair oscillates between |gg⟩ and (|gr⟩ + |rg⟩)/√2 at √2 Ω.
            sum_error = max(sum_error, abs(n_total - math.sin(math.sqrt(2) * omega * t / 2) ** 2))
        assert largest <= 1 / v_over_omega**2
        assert largest > 0.1 / v_over_omega**2  # the bound is of the right order, not loose
        assert sum_error < 5 / v_over_omega


class TestFarApartAtoms:
    @pytest.mark.parametrize("interactions", [True, False])
    def test_far_apart_atoms_evolve_as_a_product_of_single_atom_solutions(
        self, interactions: bool
    ) -> None:
        """200 µm apart, ``V = C6/200⁶ ≈ 8.5e-8 rad/µs``: ψ = ψ_0 ⊗ ψ_1, each from the closed form."""
        layer = _layer(
            2,
            3,
            register=AtomRegister.chain(2, 200.0),
            omega_max=8.0,
            segment_duration=0.4,
            interactions=interactions,
        )
        layer.set_pulse(omega=[2.0, 5.0, 3.5], detuning=[0.3, -1.2, 2.0], phase=[0.0, 1.1, -0.6])
        x = torch.tensor([[-0.8, 1.3]], dtype=F64)
        psi = layer.states(x)[0].detach().numpy()
        local = layer.local_detunings(x)[0].numpy()
        singles = []
        for i in range(2):
            factor = np.array([1.0, 0.0], dtype=complex)
            for omega, detuning, phase in layer.pulse().numpy():
                factor = _single_atom_propagator(omega, detuning + local[i], phase, 0.4) @ factor
            singles.append(factor)
        expected = np.kron(singles[0], singles[1])
        # Without interactions exactly; with them to the size of V · t ≈ 1e-7.
        np.testing.assert_allclose(psi, expected, atol=1e-12 if not interactions else 1e-6)
        assert not np.allclose(singles[0], singles[1], atol=1e-3)  # the atoms do differ


class TestAgainstScipyExpm:
    @pytest.mark.parametrize(
        "register",
        [
            AtomRegister.chain(3, 7.0),
            AtomRegister([[0.0, 0.0], [4.1, 0.3], [1.2, 6.5], [9.0, 5.0]]),
        ],
        ids=["chain-3", "irregular-4"],
    )
    def test_states_and_readout_match_expm_of_the_piecewise_hamiltonian(
        self, register: AtomRegister
    ) -> None:
        n = register.n_atoms
        layer = _layer(n, 3, register=register, segment_duration=0.11)
        layer.set_pulse(omega=[3.0, 9.0, 6.0], detuning=[-4.0, 10.0, 2.5], phase=[0.2, -1.4, 2.9])
        x = torch.linspace(-2.0, 2.5, n, dtype=F64).reshape(1, n)
        psi = _reference_state(layer, x[0].numpy())
        np.testing.assert_allclose(layer.states(x)[0].detach().numpy(), psi, atol=1e-11)
        np.testing.assert_allclose(layer(x)[0].detach().numpy(), _z_expectations(psi), atol=1e-11)

    def test_a_batch_equals_its_samples_one_by_one(self) -> None:
        layer = _layer(3, 2, segment_duration=0.2)
        layer.set_pulse(omega=[5.0, 7.0], detuning=[1.0, -3.0], phase=[0.0, 0.8])
        x = torch.randn(5, 3, generator=torch.Generator().manual_seed(4), dtype=F64)
        batched = layer(x)
        for b in range(5):
            np.testing.assert_allclose(
                batched[b].detach().numpy(),
                _z_expectations(_reference_state(layer, x[b].numpy())),
                atol=1e-11,
            )

    def test_readout_order_follows_the_atoms(self) -> None:
        """Without interactions an input moves its own atom's ⟨Z⟩ only: column i is atom i."""
        layer = _layer(3, 2, interactions=False, segment_duration=0.15)
        layer.set_pulse(omega=[6.0, 4.0], detuning=[0.5, 0.5], phase=[0.0, 0.3])
        x = torch.zeros(1, 3, dtype=F64)
        base = layer(x)[0]
        for i in range(3):
            moved = x.clone()
            moved[0, i] = 2.0
            change = (layer(moved)[0] - base).abs()
            assert change[i] > 1e-3
            assert torch.all(change[torch.arange(3) != i] < 1e-14)


# ---------------------------------------------------------------------------
# Pulse parameterisation
# ---------------------------------------------------------------------------


class TestPulseBounds:
    def test_the_bounds_hold_for_any_raw_weight(self) -> None:
        layer = _layer(2, 4, omega_max=5.0, detuning_max=7.0)
        with torch.no_grad():
            layer.qlayer.weights.copy_(
                torch.tensor(
                    [[-30.0, -30.0, 100.0], [30.0, 30.0, -7.0], [0.0, 0.0, 0.0], [2.0, -1.0, 3.0]]
                )
            )
        omega, detuning, phase = layer.pulse().T
        assert torch.all((omega >= 0) & (omega <= 5.0))
        assert torch.all(detuning.abs() <= 7.0)
        assert phase.tolist() == [100.0, -7.0, 0.0, 3.0]  # unbounded, enters through cos and sin
        assert omega[2].item() == pytest.approx(2.5, abs=1e-15)  # σ(0) = ½
        assert detuning[2].item() == 0.0  # tanh(0) = 0

    def test_set_pulse_is_the_inverse_of_pulse(self) -> None:
        layer = _layer(2, 3, omega_max=5.0, detuning_max=7.0)
        requested = torch.tensor(
            [[0.01, -6.9, 0.3], [2.5, 0.0, -2.0], [4.99, 6.5, 3.0]], dtype=F64
        )
        layer.set_pulse(*requested.T)
        torch.testing.assert_close(layer.pulse(), requested, rtol=1e-12, atol=1e-12)

    def test_the_default_pulse_is_half_omega_no_detuning_no_phase(self) -> None:
        torch.manual_seed(5)
        layer = AnalogRydbergLayer(3, 2)
        assert torch.equal(layer.qlayer.weights, torch.zeros(2, 3))
        omega, detuning, phase = layer.pulse().T
        assert omega.tolist() == [2 * math.pi] * 2  # Ω_max/2 with Ω_max = 4π
        assert detuning.tolist() == [0.0, 0.0]
        assert phase.tolist() == [0.0, 0.0]

    @pytest.mark.parametrize(
        "pulse, match",
        [
            ({"omega": 5.0}, r"omega must lie in \(0, omega_max\) = \(0, 5.0\)"),
            ({"omega": 0.0}, r"omega must lie in \(0, omega_max\)"),
            ({"detuning": -7.0}, r"detuning must lie in \(-detuning_max, detuning_max\)"),
            ({"phase": math.nan}, "phase must be finite"),
            ({"omega": [1.0, 2.0]}, r"one value per segment \(3\)"),
        ],
    )
    def test_set_pulse_refuses_values_the_reparameterisation_cannot_reach(
        self, pulse: dict[str, object], match: str
    ) -> None:
        layer = _layer(2, 3, omega_max=5.0, detuning_max=7.0)
        with pytest.raises(ValueError, match=match):
            layer.set_pulse(**pulse)  # type: ignore[arg-type]

    def test_gradients_stay_defined_at_the_edge_of_the_range(self) -> None:
        """σ and tanh saturate but never clip: a raw weight of 12 still has a nonzero gradient."""
        layer = _layer(2, 1)
        with torch.no_grad():
            layer.qlayer.weights.copy_(torch.tensor([[12.0, 12.0, 0.5]]))
        layer(torch.tensor([[0.3, -0.2]], dtype=F64)).sum().backward()
        grad = layer.qlayer.weights.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert torch.all(grad[0, :2].abs() > 0)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def _central_difference(f, value: torch.Tensor, step: float = 1e-6) -> torch.Tensor:  # type: ignore[no-untyped-def]
    grad = torch.zeros_like(value)
    for index in np.ndindex(*value.shape):
        plus, minus = value.clone(), value.clone()
        plus[index] += step
        minus[index] -= step
        grad[index] = (f(plus) - f(minus)) / (2 * step)
    return grad


class TestGradients:
    PROBE = torch.tensor([0.7, -1.3, 0.4], dtype=F64)  # a generic linear readout of ⟨Z⟩

    def _setup(self) -> tuple[AnalogRydbergLayer, torch.Tensor]:
        layer = _layer(3, 3, register=AtomRegister.chain(3, 7.5), segment_duration=0.12)
        layer.set_pulse(omega=[4.0, 9.0, 6.5], detuning=[2.0, -5.0, 1.0], phase=[0.3, 1.7, -0.9])
        x = torch.tensor([[0.4, -1.1, 2.0], [-0.3, 0.2, 0.9]], dtype=F64)
        return layer, x

    def test_pulse_parameter_gradients_match_finite_differences(self) -> None:
        layer, x = self._setup()
        weights = layer.qlayer.weights
        (layer(x) @ self.PROBE).sum().backward()
        assert weights.grad is not None

        def loss(w: torch.Tensor) -> float:
            with torch.no_grad():
                return float((torch.stack(layer.qlayer.qnode(x, w), -1) @ self.PROBE).sum())

        expected = _central_difference(loss, weights.detach().clone())
        torch.testing.assert_close(weights.grad, expected, rtol=1e-6, atol=1e-8)
        # A common shift of every phase is a gauge (module docstring): the
        # phase gradients sum to zero, while each one alone is not zero.
        assert abs(weights.grad[:, 2].sum().item()) < 1e-12
        assert torch.all(weights.grad[:, 2].abs() > 1e-4)

    def test_input_gradients_match_finite_differences(self) -> None:
        layer, x = self._setup()
        x = x.clone().requires_grad_(True)
        (layer(x) @ self.PROBE).sum().backward()

        def loss(value: torch.Tensor) -> float:
            with torch.no_grad():
                return float((layer(value) @ self.PROBE).sum())

        assert x.grad is not None
        torch.testing.assert_close(
            x.grad, _central_difference(loss, x.detach().clone()), rtol=1e-6, atol=1e-8
        )

    def test_a_degenerate_spectrum_still_has_the_right_gradient(self) -> None:
        """Equal inputs without interactions: degenerate H, where an ``eigh`` backward is wrong (#505)."""
        layer = _layer(2, 2, interactions=False, segment_duration=0.3)
        layer.set_pulse(omega=[4.0, 6.0], detuning=[1.0, -2.0], phase=[0.0, 0.5])
        x = torch.tensor([[0.4, 0.4]], dtype=F64, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert x.grad[0, 0].item() == pytest.approx(x.grad[0, 1].item(), rel=1e-10)

        def loss(value: torch.Tensor) -> float:
            with torch.no_grad():
                return float(layer(value).sum())

        torch.testing.assert_close(
            x.grad, _central_difference(loss, x.detach().clone()), rtol=1e-6, atol=1e-8
        )


# ---------------------------------------------------------------------------
# Settings, validation and noise options
# ---------------------------------------------------------------------------


class TestSettings:
    def test_c6_is_configurable_and_defaults_to_rubidium_70s(self) -> None:
        assert _layer(2, 1).c6 == DEFAULT_C6
        assert DEFAULT_C6 == pytest.approx(5.42e6, rel=1e-3)  # the issue's 5.42 × 10⁶
        x = torch.tensor([[0.1, -0.3]], dtype=F64)
        outputs = []
        for c6 in (DEFAULT_C6, DEFAULT_C6 / 50):
            layer = _layer(2, 2, c6=c6, segment_duration=0.2)
            layer.set_pulse(omega=8.0, detuning=1.0, phase=[0.0, 0.4])
            np.testing.assert_allclose(
                layer(x)[0].detach().numpy(),
                _z_expectations(_reference_state(layer, x[0].numpy())),
                atol=1e-11,
            )
            outputs.append(layer(x))
        assert not torch.allclose(outputs[0], outputs[1], atol=1e-3)

    def test_the_default_register_is_the_rydberg_entanglers_chain(self) -> None:
        layer = _layer(4, 1)
        assert torch.equal(layer.register.positions, AtomRegister.chain(4, 6.0).positions)

    def test_the_cap_is_max_atoms(self) -> None:
        with pytest.raises(ValueError, match=rf"at most {MAX_ATOMS} atoms.*got {MAX_ATOMS + 1}"):
            AnalogRydbergLayer(MAX_ATOMS + 1, 1)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"register": AtomRegister.chain(2, 6.0)}, r"register has 2 atoms.*n_qubits=3"),
            ({"n_segments": 0}, "n_segments must be an integer >= 1"),
            ({"omega_max": -1.0}, "omega_max must be a finite number > 0"),
            ({"segment_duration": 0.0}, "segment_duration must be a finite number > 0"),
            ({"shots": 100}, "AnalogRydbergLayer computes exact expectation values"),
            ({"diff_method": "adjoint"}, "differentiated by backprop on default.qubit"),
            ({"device_name": "lightning.qubit"}, "differentiated by backprop on default.qubit"),
            ({"readout": "middle"}, "readout must be 'all' or 'first'"),
        ],
    )
    def test_refuses_bad_options(self, kwargs: dict[str, object], match: str) -> None:
        options = {"n_qubits": 3, "n_segments": 2, **kwargs}
        with pytest.raises(ValueError, match=match):
            AnalogRydbergLayer(**options)  # type: ignore[arg-type]

    def test_rejects_bad_inputs(self) -> None:
        layer = _layer(3, 1)
        with pytest.raises(ValueError, match="does not match n_qubits=3"):
            layer(torch.zeros(2, 4, dtype=F64))
        with pytest.raises(ValueError, match="NaN"):
            layer(torch.tensor([[0.0, math.nan, 0.0]], dtype=F64))


class TestNoiseOptions:
    def test_depolarizing_at_the_end_scales_every_readout(self) -> None:
        """``position="end"``: one channel per atom before measurement, ⟨Z_i⟩ · (1 − 4p/3)."""
        layer = _layer(3, 2, segment_duration=0.2)
        layer.set_pulse(omega=[6.0, 3.0], detuning=[1.0, -2.0], phase=[0.0, 1.0])
        x = torch.tensor([[0.3, -0.5, 1.2]], dtype=F64)
        exact = layer(x)
        with apply_depolarizing_noise(layer, 0.15, position="end"):
            noisy = layer(x)
        torch.testing.assert_close(noisy, exact * (1 - 4 * 0.15 / 3), rtol=1e-10, atol=1e-12)

    def test_depolarizing_after_every_segment_differs_from_the_end(self) -> None:
        layer = _layer(2, 3, segment_duration=0.2)
        layer.set_pulse(omega=6.0, detuning=[1.0, -2.0, 0.5], phase=[0.0, 1.0, 2.0])
        x = torch.tensor([[0.3, -0.5]], dtype=F64)
        with apply_depolarizing_noise(layer, 0.1, position="all"):
            every = layer(x)
        with apply_depolarizing_noise(layer, 0.1, position="end"):
            end = layer(x)
        assert not torch.allclose(every, end, atol=1e-3)

    def test_readout_error_is_the_affine_map_of_z(self) -> None:
        layer = _layer(2, 1)
        x = torch.tensor([[0.3, -0.5]], dtype=F64)
        exact = layer(x)
        with apply_readout_error(layer, 0.02, 0.08):
            mapped = layer(x)
        torch.testing.assert_close(mapped, (1 - 0.1) * exact + (0.08 - 0.02), rtol=0, atol=1e-12)

    @pytest.mark.parametrize("method", ["density", "trajectories"])
    def test_training_noise_runs_in_train_mode_only(self, method: str) -> None:
        layer = _layer(2, 2, noise_level=0.2, noise_position="end", noise_method=method)
        x = torch.tensor([[0.3, -0.5]], dtype=F64).repeat(20, 1)
        clean = layer.eval()(x)
        torch.manual_seed(3)
        noisy = layer.train()(x)
        if method == "density":
            torch.testing.assert_close(noisy, clean * (1 - 4 * 0.2 / 3), rtol=1e-6, atol=1e-7)
        else:
            assert not torch.allclose(noisy, clean, atol=1e-6)


class TestDiagnostics:
    def test_circuit_summary_counts_one_unitary_per_segment(self) -> None:
        s = circuit_summary(AnalogRydbergLayer(3, 4))
        assert s.layer_type == "AnalogRydbergLayer"
        assert s.gate_counts == {"QubitUnitary": 4}
        assert (s.n_gates, s.depth, s.n_trainable_params) == (4, 4, 12)
        # Every segment mixes every atom; the common-phase redundancy is not a
        # per-entry inertness, so nothing is counted inert.
        assert s.n_inert_params == 0

    def test_gradient_variance_runs_on_the_pulse_weights(self) -> None:
        result = gradient_variance(
            AnalogRydbergLayer(2, 2), n_samples=4, generator=torch.Generator().manual_seed(0)
        )
        assert result.per_parameter.shape == (2, 3)
        assert bool((result.per_parameter > 0).all())

    def test_apply_shots_samples_around_the_exact_readout(self) -> None:
        layer = _layer(2, 2).eval()
        x = torch.zeros(1, 2, dtype=F64)
        with torch.no_grad():
            exact = layer(x)
            with apply_shots(layer, 20000):
                sampled = layer(x)
        # Standard error sqrt((1 − ⟨Z⟩²)/S) ≤ 0.0071; five of them.
        assert torch.all((sampled - exact).abs() < 0.036)
        assert not torch.equal(sampled, exact)
