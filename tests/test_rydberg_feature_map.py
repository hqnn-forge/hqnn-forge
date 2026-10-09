"""
tests/test_rydberg_feature_map.py
=================================
``PulseEncoding`` and ``RydbergFeatureMap`` (#499) against the model of
``docs/rydberg-model.md``.

Every expected value is obtained another way than the feature map obtains it:

* the closed forms of the model page, typed in: the detuned Rabi formula, the
  pulse function ``f(s) = sin²(π s / 2) / s²`` and the logistic bound;
* :func:`dephased_atom`, the page's two single-atom dephasing equations
  ``dp/dt = Ω Im c`` and ``dc/dt = −(γ/2 + iΔ) c − i (Ω/2)(2p − 1)`` solved as
  a real linear system with ``torch.linalg.matrix_exp``.  It uses neither a
  Hamiltonian matrix, nor an eigendecomposition, nor a splitting;
* :func:`reference_state`, ``exp(−iHT)`` applied to ``|g…g⟩`` with ``H``
  assembled from Kronecker products of typed-in 2 × 2 matrices, and the
  readout as traces with Kronecker-built number operators;
* for the shots, the binomial variance ``p(1 − p)/S`` of the exact features.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
import torch

import hqnn_forge.rydberg.feature_map as feature_map_module
from hqnn_forge._encoding_contract import is_circuit_layer
from hqnn_forge.rydberg import (
    DEFAULT_C6,
    MAX_ATOMS,
    AtomRegister,
    PulseEncoding,
    RydbergFeatureMap,
    readout,
)

C128 = torch.complex128
F64 = torch.float64

#: The reference Rabi frequency of the model page, 2π × 2 MHz, in rad/µs.
OMEGA = 4 * math.pi
#: The spacing at which the nearest-neighbour interaction equals OMEGA (8.69 µm).
BLOCKADE_RADIUS = (DEFAULT_C6 / OMEGA) ** (1 / 6)


def chain(n_atoms: int, v_over_omega: float = 1.0) -> AtomRegister:
    """An open chain with nearest-neighbour interaction ``V = v_over_omega · Ω``."""
    return AtomRegister.chain(n_atoms, BLOCKADE_RADIUS * v_over_omega ** (-1 / 6))


def inputs(n_samples: int, n_features: int, seed: int = 0) -> torch.Tensor:
    """Standard normal inputs scaled by 1.5, so the detunings cover most of their range."""
    generator = torch.Generator().manual_seed(seed)
    return 1.5 * torch.randn(n_samples, n_features, generator=generator, dtype=F64)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rabi(omega: float, delta: float, t: float) -> float:
    """``⟨n⟩(t) = Ω²/Ω_eff² · sin²(Ω_eff t / 2)``, the detuned Rabi formula of the model page."""
    effective = math.sqrt(omega**2 + delta**2)
    return omega**2 / effective**2 * math.sin(effective * t / 2) ** 2


def pulse_function(s: float) -> float:
    """``f(s) = sin²(π s / 2) / s²`` of the model page."""
    return math.sin(math.pi * s / 2) ** 2 / s**2


def dephased_atom(omega: float, delta: float, gamma: float, t: float) -> float:
    """
    ``⟨n⟩(t)`` of one atom from the dephasing equations of the model page.

    With ``c = u + iv``, ``dp/dt = Ω Im c`` and
    ``dc/dt = −(γ/2 + iΔ) c − i (Ω/2)(2p − 1)`` read::

        dp/dt =  Ω v
        du/dt = −(γ/2) u + Δ v
        dv/dt = −Δ u − (γ/2) v − Ω p + Ω/2

    a linear system with a constant term, solved exactly by the exponential of
    the 4 × 4 matrix acting on ``(p, u, v, 1)``, from ``(0, 0, 0, 1)``.
    """
    generator = torch.tensor(
        [
            [0.0, 0.0, omega, 0.0],
            [0.0, -gamma / 2, delta, 0.0],
            [-omega, -delta, -gamma / 2, omega / 2],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=F64,
    )
    start = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=F64)
    return float((torch.linalg.matrix_exp(generator * t) @ start)[0])


def number_operator(atom: int, n_atoms: int) -> torch.Tensor:
    """``n_i`` with the projector ``|r⟩⟨r|`` at tensor factor ``atom`` from the left."""
    op = torch.ones(1, 1, dtype=C128)
    for j in range(n_atoms):
        factor = torch.tensor([[0.0, 0.0], [0.0, 1.0]]) if j == atom else torch.eye(2)
        op = torch.kron(op, factor.to(C128))
    return op


def pauli_x(atom: int, n_atoms: int) -> torch.Tensor:
    op = torch.ones(1, 1, dtype=C128)
    for j in range(n_atoms):
        factor = torch.tensor([[0.0, 1.0], [1.0, 0.0]]) if j == atom else torch.eye(2)
        op = torch.kron(op, factor.to(C128))
    return op


def reference_state(
    positions: list[float], omega: float, deltas: list[float], c6: float, t: float
) -> torch.Tensor:
    """
    ``ψ(T) = exp(−iHT) |g…g⟩`` for atoms on a line at ``positions`` (µm).

    ``H = Ω/2 · Σ_i X_i − Σ_i Δ_i n_i + Σ_{i<j} C6/|x_i − x_j|⁶ · n_i n_j`` from
    Kronecker products, exponentiated with ``torch.linalg.matrix_exp``.
    """
    n = len(positions)
    h = torch.zeros(2**n, 2**n, dtype=C128)
    for i in range(n):
        h = h + omega / 2 * pauli_x(i, n) - deltas[i] * number_operator(i, n)
        for j in range(i + 1, n):
            coupling = c6 / abs(positions[i] - positions[j]) ** 6
            h = h + coupling * number_operator(i, n) @ number_operator(j, n)
    return torch.linalg.matrix_exp(-1j * h * t)[:, 0]


class TestPulseEncoding:
    """``Δ_i = Δ_max · σ(x_i)``: the bound, the defaults and what is refused."""

    def test_defaults_are_those_of_the_model_page(self) -> None:
        """``ΩT = π`` and ``Δ_max = √3 Ω``; at the reference Ω, 0.25 µs and 21.8 rad/µs."""
        encoding = PulseEncoding(4, omega=OMEGA)
        assert encoding.n_features == 4
        assert encoding.omega == OMEGA
        assert encoding.omega * encoding.t == pytest.approx(math.pi, rel=1e-15)
        assert encoding.t == pytest.approx(0.25, rel=1e-15)
        assert encoding.delta_max / encoding.omega == pytest.approx(math.sqrt(3), rel=1e-15)
        assert encoding.delta_max == pytest.approx(21.8, abs=0.05)

    def test_detunings_are_the_logistic_function_of_each_input(self) -> None:
        """``σ(0) = 1/2`` and ``σ(±ln 3) = 3/4, 1/4``; column ``i`` belongs to input ``i``."""
        encoding = PulseEncoding(3, omega=2.0, delta_max=8.0)
        x = [[0.0, math.log(3), -math.log(3)], [-math.log(3), 0.0, math.log(3)]]
        delta = encoding.detunings(x)
        assert delta.dtype == F64
        assert delta.shape == (2, 3)
        assert delta.tolist() == [
            pytest.approx([4.0, 6.0, 2.0], rel=1e-14),
            pytest.approx([2.0, 4.0, 6.0], rel=1e-14),
        ]

    @pytest.mark.parametrize("x", [-30.0, -3.0, -0.2, 0.0, 0.7, 5.0, 30.0])
    def test_matches_the_formula_and_stays_inside_the_open_interval(self, x: float) -> None:
        encoding = PulseEncoding(1, omega=1.0, delta_max=2.5)
        delta = encoding.detunings([[x]]).item()
        assert delta == pytest.approx(2.5 * sigmoid(x), rel=1e-13)
        assert 0.0 < delta < 2.5

    def test_the_bound_is_not_a_clip(self) -> None:
        """Inputs far beyond any clipping range still map to different detunings, in order."""
        encoding = PulseEncoding(1, omega=1.0)
        x = torch.tensor([[-20.0], [-10.0], [-5.0], [5.0], [10.0], [20.0]], dtype=F64)
        delta = encoding.detunings(x)[:, 0]
        assert bool((delta[1:] > delta[:-1]).all())
        assert delta[0].item() > 0.0
        assert delta[-1].item() < encoding.delta_max
        # 1 − σ(20) = σ(−20) = 2.06e-9: the distance to the bound, not zero.
        assert (encoding.delta_max - delta[-1].item()) / encoding.delta_max == pytest.approx(
            sigmoid(-20.0), rel=1e-6
        )

    def test_accepts_numpy_and_float32_inputs_in_double_precision(self) -> None:
        encoding = PulseEncoding(2, omega=1.0, delta_max=1.0)
        from_numpy = encoding.detunings(np.array([[0.3, -1.2]]))
        from_float32 = encoding.detunings(torch.tensor([[0.5, -1.0]], dtype=torch.float32))
        assert from_numpy.tolist()[0] == pytest.approx([sigmoid(0.3), sigmoid(-1.2)], rel=1e-14)
        assert from_float32.dtype == F64
        assert from_float32.tolist()[0] == pytest.approx([sigmoid(0.5), sigmoid(-1.0)], rel=1e-14)

    def test_no_samples_give_no_detunings(self) -> None:
        assert PulseEncoding(3, omega=1.0).detunings(torch.zeros(0, 3)).shape == (0, 3)

    @pytest.mark.parametrize("shape", [(3,), (2, 2), (2, 4), (2, 3, 1), ()])
    def test_rejects_inputs_of_the_wrong_width(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"X must have shape \(n_samples, 3\)"):
            PulseEncoding(3, omega=1.0).detunings(torch.zeros(shape))

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_inputs(self, bad: float) -> None:
        with pytest.raises(ValueError, match="X must be finite"):
            PulseEncoding(2, omega=1.0).detunings([[0.0, bad]])

    def test_rejects_complex_inputs(self) -> None:
        with pytest.raises(TypeError, match="X must hold real numbers"):
            PulseEncoding(1, omega=1.0).detunings(torch.zeros(1, 1, dtype=C128))

    @pytest.mark.parametrize("n_features", [0, -1, 2.0, True, "3", None])
    def test_rejects_a_bad_width(self, n_features: Any) -> None:
        with pytest.raises(ValueError, match="n_features must be an integer >= 1"):
            PulseEncoding(n_features, omega=1.0)

    @pytest.mark.parametrize("omega", [0.0, -1.0, math.nan, math.inf, "1", True])
    def test_rejects_a_bad_omega(self, omega: Any) -> None:
        with pytest.raises(ValueError, match="omega must be a finite number > 0"):
            PulseEncoding(2, omega=omega)

    @pytest.mark.parametrize("t", [-0.1, math.nan, math.inf, "1", True])
    def test_rejects_a_bad_time(self, t: Any) -> None:
        with pytest.raises(ValueError, match="t must be a finite number >= 0"):
            PulseEncoding(2, omega=1.0, t=t)

    @pytest.mark.parametrize("delta_max", [0.0, -1.0, math.nan, math.inf, "1", True])
    def test_rejects_a_bad_bound(self, delta_max: Any) -> None:
        with pytest.raises(ValueError, match="delta_max must be a finite number > 0"):
            PulseEncoding(2, omega=1.0, delta_max=delta_max)

    def test_config_holds_the_resolved_pulse_and_rebuilds_it(self) -> None:
        encoding = PulseEncoding(5, omega=2.0)
        config = encoding.get_config()
        assert config == {
            "n_features": 5,
            "omega": 2.0,
            "t": math.pi / 2.0,
            "delta_max": math.sqrt(3) * 2.0,
        }
        rebuilt = PulseEncoding(**json.loads(json.dumps(config)))
        assert rebuilt.get_config() == config
        x = inputs(4, 5)
        assert torch.equal(rebuilt.detunings(x), encoding.detunings(x))


class TestInteractionsOff:
    """The control: feature ``i`` is the single-atom value of input ``i`` and of nothing else."""

    def test_default_pulse_gives_the_pulse_function_of_the_own_input(self) -> None:
        """``F_i = f(√(1 + 3 σ(x_i)²))``, the last row of the page's closed-form limits."""
        x = inputs(6, 5, seed=1)
        feature_map = RydbergFeatureMap(
            chain(5),
            PulseEncoding(5, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.0,
            interactions=False,
        )
        features = feature_map.transform(x)
        assert features.dtype == F64
        assert features.shape == (6, 5)
        for sample in range(6):
            expected = [
                pulse_function(math.sqrt(1 + 3 * sigmoid(x[sample, i].item()) ** 2))
                for i in range(5)
            ]
            assert features[sample].tolist() == pytest.approx(expected, abs=1e-12)

    def test_the_page_quotes_0437_at_a_zero_input(self) -> None:
        feature_map = RydbergFeatureMap(
            chain(4),
            PulseEncoding(4, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.0,
            interactions=False,
        )
        assert feature_map.transform(torch.zeros(1, 4)).tolist()[0] == pytest.approx(
            [0.437] * 4, abs=5e-4
        )

    @pytest.mark.parametrize(
        ("omega", "t", "delta_max"), [(1.3, 2.9, 4.0), (OMEGA, 0.11, 30.0), (0.7, 9.0, 0.5)]
    )
    def test_any_pulse_gives_the_detuned_rabi_formula(
        self, omega: float, t: float, delta_max: float
    ) -> None:
        """Ω, ``T`` and ``Δ_max`` are settings: the reference uses the ones passed."""
        x = inputs(5, 3, seed=2)
        feature_map = RydbergFeatureMap(
            chain(3),
            PulseEncoding(3, omega=omega, t=t, delta_max=delta_max),
            c6=DEFAULT_C6,
            gamma=0.0,
            interactions=False,
        )
        features = feature_map.transform(x)
        for sample in range(5):
            expected = [rabi(omega, delta_max * sigmoid(x[sample, i].item()), t) for i in range(3)]
            assert features[sample].tolist() == pytest.approx(expected, abs=1e-12)

    def test_the_dephasing_reference_reproduces_the_damped_oscillation(self) -> None:
        """:func:`dephased_atom` at ``Δ = 0`` against the page's underdamped closed form."""
        omega, gamma, t = 2.0, 1.0, 1.7
        lam = math.sqrt(omega**2 - gamma**2 / 16)
        bracket = math.cos(lam * t) + gamma / (4 * lam) * math.sin(lam * t)
        closed_form = 0.5 - 0.5 * math.exp(-gamma * t / 4) * bracket
        assert dephased_atom(omega, 0.0, gamma, t) == pytest.approx(closed_form, abs=1e-13)
        assert dephased_atom(omega, 1.1, 0.0, t) == pytest.approx(rabi(omega, 1.1, t), abs=1e-13)

    @pytest.mark.parametrize("gamma_over_omega", [0.1, 1.0, 4.0])
    def test_with_dephasing_gives_the_single_atom_dephasing_solution(
        self, gamma_over_omega: float
    ) -> None:
        """Each feature solves the page's two dephasing equations for its own ``Δ_i``."""
        x = inputs(4, 4, seed=3)
        encoding = PulseEncoding(4, omega=OMEGA)
        gamma = gamma_over_omega * OMEGA
        feature_map = RydbergFeatureMap(
            chain(4), encoding, c6=DEFAULT_C6, gamma=gamma, n_steps=4000, interactions=False
        )
        features = feature_map.transform(x)
        for sample in range(4):
            expected = [
                dephased_atom(
                    OMEGA, encoding.delta_max * sigmoid(x[sample, i].item()), gamma, encoding.t
                )
                for i in range(4)
            ]
            # second-order splitting at 4000 steps: the measured error is 1.5e-9,
            # 2.7e-9 and 5.9e-8 at γ/Ω = 0.1, 1 and 4
            assert features[sample].tolist() == pytest.approx(expected, abs=5e-7)

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (0.5 * OMEGA, 60)])
    @pytest.mark.parametrize("correlations", [False, True])
    def test_a_feature_does_not_change_with_another_atoms_input(
        self, gamma: float, n_steps: int | None, correlations: bool
    ) -> None:
        """Perturbing every input but ``x_i`` leaves ``⟨n_i⟩`` unchanged to 1e-12."""
        n = 4
        feature_map = RydbergFeatureMap(
            chain(n),
            PulseEncoding(n, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=gamma,
            n_steps=n_steps,
            interactions=False,
            correlations=correlations,
        )
        x = inputs(3, n, seed=4)
        base = feature_map.transform(x)
        for atom in range(n):
            perturbed = x + 0.9
            perturbed[:, atom] = x[:, atom]
            changed = feature_map.transform(perturbed)
            assert torch.allclose(changed[:, atom], base[:, atom], rtol=0, atol=1e-12)
            # and its own input does move it: the control is informative
            own = x.clone()
            own[:, atom] += 0.9
            moved = (feature_map.transform(own)[:, atom] - base[:, atom]).abs()
            assert moved.min().item() > 1e-3

    def test_pair_features_are_products_without_interactions(self) -> None:
        """A product state has ``⟨n_i n_j⟩ = ⟨n_i⟩⟨n_j⟩``, in the order (0,1), (0,2), (1,2)."""
        feature_map = RydbergFeatureMap(
            chain(3),
            PulseEncoding(3, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.0,
            interactions=False,
            correlations=True,
        )
        x = torch.tensor([[-1.0, 0.2, 1.4]], dtype=F64)
        p = [pulse_function(math.sqrt(1 + 3 * sigmoid(v) ** 2)) for v in (-1.0, 0.2, 1.4)]
        expected = [*p, p[0] * p[1], p[0] * p[2], p[1] * p[2]]
        assert feature_map.transform(x).tolist()[0] == pytest.approx(expected, abs=1e-12)


class TestInteractionsOn:
    """With the ``n_i n_j`` term the features are those of the full Hamiltonian."""

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (0.5 * OMEGA, 60)])
    def test_a_feature_changes_with_another_atoms_input(
        self, gamma: float, n_steps: int | None
    ) -> None:
        """
        The perturbation that left the control unchanged to 1e-12 moves every feature.

        At ``V = Ω`` the smallest change over the 3 samples and 4 atoms is
        ``3.7e-4`` (with dephasing), 8 orders above the control's bound.
        """
        n = 4
        feature_map = RydbergFeatureMap(
            chain(n), PulseEncoding(n, omega=OMEGA), c6=DEFAULT_C6, gamma=gamma, n_steps=n_steps
        )
        x = inputs(3, n, seed=4)
        base = feature_map.transform(x)
        for atom in range(n):
            perturbed = x + 0.9
            perturbed[:, atom] = x[:, atom]
            moved = (feature_map.transform(perturbed)[:, atom] - base[:, atom]).abs()
            assert moved.min().item() > 1e-4

    def test_two_atoms_against_the_exponential_of_a_hand_built_hamiltonian(self) -> None:
        """Ω, ``T``, the detunings and ``C6/r⁶`` all reach the Hamiltonian as documented."""
        omega, t, delta_max, c6, spacing = 1.3, 2.1, 2.0, 50.0, 2.0
        x = [[0.4, -0.9], [-1.5, 2.0]]
        feature_map = RydbergFeatureMap(
            AtomRegister.chain(2, spacing),
            PulseEncoding(2, omega=omega, t=t, delta_max=delta_max),
            c6=c6,
            gamma=0.0,
            correlations=True,
        )
        features = feature_map.transform(x)
        states = feature_map.states(x)
        assert states.dtype == C128
        assert states.shape == (2, 4, 4)
        for sample, row in enumerate(x):
            deltas = [delta_max * sigmoid(v) for v in row]
            psi = reference_state([0.0, spacing], omega, deltas, c6, t)
            # |gg⟩, |gr⟩, |rg⟩, |rr⟩: atom 0 is excited in the last two
            p = (psi.abs() ** 2).tolist()
            expected = [p[2] + p[3], p[1] + p[3], p[3]]
            assert features[sample].tolist() == pytest.approx(expected, abs=1e-12)
            assert torch.allclose(
                states[sample], psi[:, None] * psi.conj()[None, :], rtol=0, atol=1e-12
            )

    def test_three_atoms_with_pairs_in_the_order_of_the_model_page(self) -> None:
        """``⟨n_0⟩, ⟨n_1⟩, ⟨n_2⟩``, then (0,1), (0,2), (1,2), as traces with ``n_i n_j``."""
        omega, t, delta_max, c6 = 2.0, 1.4, 3.0, 300.0
        positions = [0.0, 2.0, 5.0]  # unequal spacings: no two pairs are alike
        x = [[0.3, -0.7, 1.1]]
        feature_map = RydbergFeatureMap(
            AtomRegister([[p, 0.0] for p in positions]),
            PulseEncoding(3, omega=omega, t=t, delta_max=delta_max),
            c6=c6,
            gamma=0.0,
            correlations=True,
        )
        psi = reference_state(positions, omega, [delta_max * sigmoid(v) for v in x[0]], c6, t)
        n = [number_operator(i, 3) for i in range(3)]
        singles = [(psi.conj() @ n[i] @ psi).real.item() for i in range(3)]
        pairs = [(psi.conj() @ n[i] @ n[j] @ psi).real.item() for i, j in [(0, 1), (0, 2), (1, 2)]]
        assert len({round(v, 6) for v in singles + pairs}) == 6
        assert feature_map.transform(x).tolist()[0] == pytest.approx(singles + pairs, abs=1e-12)

    def test_states_are_the_density_matrices_the_features_are_read_from(self) -> None:
        feature_map = RydbergFeatureMap(
            chain(3),
            PulseEncoding(3, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.3 * OMEGA,
            n_steps=40,
            correlations=True,
        )
        x = inputs(5, 3, seed=5)
        states = feature_map.states(x)
        assert states.shape == (5, 8, 8)
        trace = states.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        assert torch.allclose(trace, torch.ones_like(trace), rtol=0, atol=1e-12)
        assert torch.allclose(states, states.mH, rtol=0, atol=1e-12)
        # dephasing leaves a mixed state: Tr ρ² < 1
        purity = (states @ states).diagonal(dim1=-2, dim2=-1).sum(dim=-1).real
        assert purity.max().item() < 0.99
        assert torch.equal(readout(states, pairs=True), feature_map.transform(x))


class TestWhichInputFeedsWhichAtom:
    """
    Inputs that excite chosen atoms with certainty, without interactions.

    ``x = −40`` gives ``Δ = Δ_max σ(−40) = 4e-18 Δ_max``, a resonant π pulse and
    ``⟨n⟩ = 1`` to rounding; ``x = +40`` gives ``σ = 1`` in ``float64``,
    ``Δ = Δ_max`` and ``⟨n⟩ = sin²(π)/4 = 0`` to rounding.
    """

    ON, OFF = -40.0, 40.0

    def build(self, n: int) -> RydbergFeatureMap:
        return RydbergFeatureMap(
            chain(n),
            PulseEncoding(n, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.0,
            interactions=False,
            correlations=True,
        )

    @pytest.mark.parametrize("atom", [0, 1, 2, 3])
    def test_one_input_excites_one_atom(self, atom: int) -> None:
        x = torch.full((1, 4), self.OFF, dtype=F64)
        x[0, atom] = self.ON
        expected = [0.0] * 10
        expected[atom] = 1.0
        assert self.build(4).transform(x).tolist()[0] == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize(
        ("excited", "pair_index"),
        [((0, 1), 0), ((0, 2), 1), ((0, 3), 2), ((1, 2), 3), ((1, 3), 4), ((2, 3), 5)],
    )
    def test_two_inputs_excite_one_pair(self, excited: tuple[int, int], pair_index: int) -> None:
        """Pair columns follow (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) after the 4 atoms."""
        x = torch.full((1, 4), self.OFF, dtype=F64)
        x[0, list(excited)] = self.ON
        expected = [0.0] * 10
        for atom in excited:
            expected[atom] = 1.0
        expected[4 + pair_index] = 1.0
        feature_map = self.build(4)
        assert feature_map.transform(x).tolist()[0] == pytest.approx(expected, abs=1e-12)
        # the sampled bitstrings are read in the same order: the outcome is certain
        sampled = feature_map.transform(x, shots=50, generator=torch.Generator().manual_seed(0))
        assert sampled.tolist()[0] == expected


class TestZeroTime:
    """``T = 0``: nothing has happened, every atom is still in ``|g⟩``."""

    def test_without_dephasing_the_features_are_exactly_zero(self) -> None:
        feature_map = RydbergFeatureMap(
            chain(4),
            PulseEncoding(4, omega=OMEGA, t=0.0),
            c6=DEFAULT_C6,
            gamma=0.0,
            correlations=True,
        )
        features = feature_map.transform(inputs(5, 4))
        assert features.shape == (5, 10)
        # ψ(0) = V V† e_0 is e_0 to rounding only: the largest feature is 1.2e-30
        assert features.abs().max().item() < 1e-15
        assert feature_map.evolution_time == 0.0

    @pytest.mark.parametrize("n_steps", [1, 7])
    def test_with_dephasing_the_features_are_zero_to_rounding(self, n_steps: int) -> None:
        """
        A step count is still required; the steps are identities up to rounding.

        The largest feature is 1.3e-30 after 1 step and 6.1e-29 after 7: zero
        to rounding, not bit for bit.
        """
        feature_map = RydbergFeatureMap(
            chain(4),
            PulseEncoding(4, omega=OMEGA, t=0.0),
            c6=DEFAULT_C6,
            gamma=OMEGA,
            n_steps=n_steps,
            correlations=True,
        )
        features = feature_map.transform(inputs(5, 4))
        assert features.shape == (5, 10)
        assert features.abs().max().item() < 1e-15

    def test_the_same_map_with_a_pulse_is_not_zero(self) -> None:
        """The zeros above come from ``T = 0``, not from the map."""
        feature_map = RydbergFeatureMap(
            chain(4), PulseEncoding(4, omega=OMEGA), c6=DEFAULT_C6, gamma=OMEGA, n_steps=7
        )
        assert feature_map.transform(inputs(5, 4)).min().item() > 1e-3


class TestChunks:
    """Chunking bounds the working memory and changes no number."""

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (OMEGA, 25)])
    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 6, 7, 50])
    def test_chunked_and_unchunked_features_are_identical(
        self, gamma: float, n_steps: int | None, chunk_size: int
    ) -> None:
        """Bit for bit, also for a last chunk as long as the register (4 of 7 with size 3)."""
        feature_map = RydbergFeatureMap(
            chain(4),
            PulseEncoding(4, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=gamma,
            n_steps=n_steps,
            correlations=True,
        )
        x = inputs(7, 4, seed=6)
        whole = feature_map.transform(x, chunk_size=7)
        assert torch.equal(feature_map.transform(x, chunk_size=chunk_size), whole)
        assert torch.equal(feature_map.transform(x), whole)
        assert torch.equal(
            feature_map.states(x, chunk_size=chunk_size), feature_map.states(x, chunk_size=7)
        )

    def test_chunked_and_unchunked_shot_estimates_are_identical(self) -> None:
        """Bitstrings are drawn sample by sample, so the chunks do not move the stream."""
        feature_map = RydbergFeatureMap(
            chain(3), PulseEncoding(3, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0, correlations=True
        )
        x = inputs(7, 3, seed=7)
        whole = feature_map.transform(
            x, shots=200, generator=torch.Generator().manual_seed(11), chunk_size=7
        )
        for chunk_size in (1, 3):
            chunked = feature_map.transform(
                x, shots=200, generator=torch.Generator().manual_seed(11), chunk_size=chunk_size
            )
            assert torch.equal(chunked, whole)

    @pytest.mark.parametrize("method", ["transform", "states"])
    def test_the_solver_sees_at_most_chunk_size_samples(
        self, method: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """7 samples in chunks of 3 reach ``evolve`` as batches of 3, 3 and 1."""
        seen: list[int] = []
        solver = feature_map_module.evolve

        def spy(hamiltonian: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            seen.append(hamiltonian.shape[0])
            return solver(hamiltonian, *args, **kwargs)

        monkeypatch.setattr(feature_map_module, "evolve", spy)
        feature_map = RydbergFeatureMap(
            chain(2), PulseEncoding(2, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0
        )
        getattr(feature_map, method)(inputs(7, 2), chunk_size=3)
        assert seen == [3, 3, 1]

    @pytest.mark.parametrize(("n_atoms", "expected"), [(1, 2**18), (4, 4096), (6, 256), (10, 1)])
    def test_the_default_chunk_holds_at_most_2_to_the_20_matrix_entries(
        self, n_atoms: int, expected: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``2^20 / 4^N`` samples, 16 MiB of ``complex128`` per matrix of the chunk."""
        seen: list[int] = []

        def spy(hamiltonian: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            seen.append(hamiltonian.shape[0])
            dim = hamiltonian.shape[-1]
            return torch.zeros(hamiltonian.shape[0], dim, dim, dtype=C128)

        # The Hamiltonians of a whole default chunk of 1 atom would be built for
        # nothing here; a stand-in keeps the test at the chunk arithmetic.
        monkeypatch.setattr(feature_map_module, "evolve", spy)
        monkeypatch.setattr(
            feature_map_module,
            "rydberg_hamiltonian",
            lambda register, omega, delta, **kwargs: torch.zeros(
                delta.shape[0], 2**register.n_atoms, 2**register.n_atoms, dtype=C128
            ),
        )
        feature_map = RydbergFeatureMap(
            chain(n_atoms), PulseEncoding(n_atoms, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0
        )
        feature_map.transform(torch.zeros(expected + 1, n_atoms))
        assert seen == [expected, 1]

    @pytest.mark.parametrize("chunk_size", [0, -2, 2.0, True, "4"])
    @pytest.mark.parametrize("method", ["transform", "states"])
    def test_rejects_a_bad_chunk_size(self, method: str, chunk_size: Any) -> None:
        feature_map = RydbergFeatureMap(
            chain(2), PulseEncoding(2, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0
        )
        with pytest.raises(ValueError, match="chunk_size must be an integer >= 1 or None"):
            getattr(feature_map, method)(inputs(3, 2), chunk_size=chunk_size)

    @pytest.mark.parametrize("correlations", [False, True])
    def test_no_samples_give_no_features(self, correlations: bool) -> None:
        feature_map = RydbergFeatureMap(
            chain(3),
            PulseEncoding(3, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.0,
            correlations=correlations,
        )
        empty = torch.zeros(0, 3)
        assert feature_map.transform(empty).shape == (0, 6 if correlations else 3)
        assert feature_map.states(empty).shape == (0, 8, 8)


class TestShots:
    """Features estimated from bitstrings sampled from the diagonal of ρ."""

    def build(self, n: int = 4, v_over_omega: float = 1.0) -> RydbergFeatureMap:
        return RydbergFeatureMap(
            chain(n, v_over_omega),
            PulseEncoding(n, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.0,
            correlations=True,
        )

    def test_the_same_seed_repeats_the_estimates_and_another_does_not(self) -> None:
        feature_map = self.build()
        x = inputs(6, 4, seed=8)
        first = feature_map.transform(x, shots=300, generator=torch.Generator().manual_seed(5))
        again = feature_map.transform(x, shots=300, generator=torch.Generator().manual_seed(5))
        other = feature_map.transform(x, shots=300, generator=torch.Generator().manual_seed(6))
        assert first.dtype == F64
        assert first.shape == (6, 10)
        assert torch.equal(first, again)
        assert not torch.equal(first, other)

    def test_a_generator_is_advanced_not_reset(self) -> None:
        """Two calls with one generator are two independent estimates."""
        feature_map = self.build()
        x = inputs(6, 4, seed=8)
        generator = torch.Generator().manual_seed(5)
        first = feature_map.transform(x, shots=300, generator=generator)
        second = feature_map.transform(x, shots=300, generator=generator)
        assert not torch.equal(first, second)

    def test_estimates_are_multiples_of_one_over_shots_and_pairs_never_exceed_atoms(self) -> None:
        """Each estimate is a count of bitstrings divided by ``S``."""
        feature_map = self.build(3)
        shots = 40
        estimates = feature_map.transform(
            inputs(8, 3, seed=9), shots=shots, generator=torch.Generator().manual_seed(1)
        )
        counts = estimates * shots
        assert torch.allclose(counts, counts.round(), rtol=0, atol=1e-9)
        assert estimates.min().item() >= 0.0
        assert estimates.max().item() <= 1.0
        # b_i b_j <= b_i and <= b_j in every bitstring: columns (0,1), (0,2), (1,2)
        for column, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)], start=3):
            assert bool((estimates[:, column] <= estimates[:, i]).all())
            assert bool((estimates[:, column] <= estimates[:, j]).all())

    @pytest.mark.parametrize("shots", [50, 800, 12800])
    def test_the_error_has_the_binomial_variance(self, shots: int) -> None:
        """
        ``Σ (F̂ − F)² / Σ F(1 − F)/S`` over all features and samples is 1 on average.

        Its expectation is exactly 1 for an unbiased estimator of variance
        ``F(1 − F)/S``, at every ``S``: the error falls as ``1/√S`` with the
        binomial constant.  Over 100 seeds per ``S`` the ratio had mean 1.00
        and standard deviation 0.03 (768 samples, 10 features each, between
        0.92 and 1.09), so the band is 5 standard deviations wide and does
        not rest on the seed.
        """
        feature_map = self.build()
        x = inputs(768, 4, seed=10)
        exact = feature_map.transform(x)
        estimates = feature_map.transform(
            x, shots=shots, generator=torch.Generator().manual_seed(shots)
        )
        ratio = ((estimates - exact) ** 2).sum() / (exact * (1 - exact) / shots).sum()
        assert ratio.item() == pytest.approx(1.0, abs=0.15)

    def test_the_estimates_converge_to_the_exact_features(self) -> None:
        """At ``S = 4 · 10⁵`` every estimate is within 6 standard errors (plus 5 counts)."""
        feature_map = self.build()
        x = inputs(8, 4, seed=12)
        exact = feature_map.transform(x)
        shots = 400_000
        estimates = feature_map.transform(
            x, shots=shots, generator=torch.Generator().manual_seed(2)
        )
        standard_error = (exact * (1 - exact) / shots).sqrt()
        assert bool(((estimates - exact).abs() <= 6 * standard_error + 5 / shots).all())
        # 1/(2√S) = 7.9e-4 bounds the standard error of any feature
        assert (estimates - exact).abs().max().item() < 6 * 0.5 / math.sqrt(shots)

    def test_the_mean_over_repeated_estimates_is_unbiased(self) -> None:
        """200 estimates of 20 shots each average to the exact features at ``1/√(200·20)``."""
        feature_map = self.build(3)
        x = inputs(4, 3, seed=13)
        exact = feature_map.transform(x)
        generator = torch.Generator().manual_seed(3)
        mean = torch.stack(
            [feature_map.transform(x, shots=20, generator=generator) for _ in range(200)]
        ).mean(dim=0)
        standard_error = (exact * (1 - exact) / (200 * 20)).sqrt()
        assert bool(((mean - exact).abs() <= 6 * standard_error + 5 / 4000).all())

    def test_shots_none_is_exact_and_ignores_the_generator(self) -> None:
        feature_map = self.build(3)
        x = inputs(3, 3)
        generator = torch.Generator().manual_seed(0)
        state = generator.get_state()
        assert torch.equal(
            feature_map.transform(x, shots=None, generator=generator), feature_map.transform(x)
        )
        assert torch.equal(generator.get_state(), state)

    @pytest.mark.parametrize("shots", [0, -5, 10.0, True, "100"])
    def test_rejects_a_bad_shot_count(self, shots: Any) -> None:
        with pytest.raises(ValueError, match="shots must be an integer >= 1 or None"):
            self.build(2).transform(
                inputs(2, 2), shots=shots, generator=torch.Generator().manual_seed(0)
            )

    def test_shots_need_a_generator(self) -> None:
        with pytest.raises(ValueError, match="generator is required when shots is given"):
            self.build(2).transform(inputs(2, 2), shots=10)

    @pytest.mark.parametrize("generator", [0, np.random.default_rng(0), "seed"])
    def test_rejects_another_kind_of_generator(self, generator: Any) -> None:
        with pytest.raises(TypeError, match=r"generator must be a torch\.Generator"):
            self.build(2).transform(inputs(2, 2), shots=10, generator=generator)


class TestConfig:
    """``get_config()`` holds every setting and rebuilds the map."""

    def test_holds_every_setting_as_plain_data(self) -> None:
        feature_map = RydbergFeatureMap(
            AtomRegister.chain(3, 6.0),
            PulseEncoding(3, omega=2.0, t=1.5, delta_max=3.0),
            c6=1000.0,
            gamma=0.25,
            n_steps=30,
            interactions=False,
            correlations=True,
        )
        assert feature_map.get_config() == {
            "register": [[0.0, 0.0], [6.0, 0.0], [12.0, 0.0]],
            "encoding": {"n_features": 3, "omega": 2.0, "t": 1.5, "delta_max": 3.0},
            "c6": 1000.0,
            "gamma": 0.25,
            "n_steps": 30,
            "interactions": False,
            "correlations": True,
        }

    @pytest.mark.parametrize(("gamma", "n_steps"), [(0.0, None), (0.7 * OMEGA, 35)])
    @pytest.mark.parametrize("interactions", [True, False])
    @pytest.mark.parametrize("correlations", [True, False])
    def test_a_map_rebuilt_from_its_config_through_json_reproduces_the_features(
        self, gamma: float, n_steps: int | None, interactions: bool, correlations: bool
    ) -> None:
        feature_map = RydbergFeatureMap(
            AtomRegister.ring(4, 0.93 * BLOCKADE_RADIUS),
            PulseEncoding(4, omega=OMEGA, t=0.31, delta_max=17.0),
            c6=DEFAULT_C6,
            gamma=gamma,
            n_steps=n_steps,
            interactions=interactions,
            correlations=correlations,
        )
        config = json.loads(json.dumps(feature_map.get_config()))
        rebuilt = type(feature_map)(**config)
        assert rebuilt.get_config() == feature_map.get_config()
        x = inputs(5, 4, seed=14)
        assert torch.equal(rebuilt.transform(x), feature_map.transform(x))
        assert torch.equal(rebuilt.states(x), feature_map.states(x))

    def test_a_changed_setting_changes_the_features(self) -> None:
        """Each recorded setting matters: a config that dropped one could not round-trip."""
        base: dict[str, Any] = RydbergFeatureMap(
            chain(3),
            PulseEncoding(3, omega=OMEGA),
            c6=DEFAULT_C6,
            gamma=0.5 * OMEGA,
            n_steps=6,
            correlations=True,
        ).get_config()
        x = inputs(3, 3, seed=15)
        reference = RydbergFeatureMap(**base).transform(x)
        encoding = base["encoding"]
        changes: list[dict[str, Any]] = [
            {"register": [[0.0, 0.0], [9.5, 0.0], [17.4, 0.0]]},
            {"encoding": {**encoding, "omega": 1.1 * OMEGA}},
            {"encoding": {**encoding, "t": 1.1 * encoding["t"]}},
            {"encoding": {**encoding, "delta_max": 1.1 * encoding["delta_max"]}},
            {"c6": 1.5 * DEFAULT_C6},
            {"gamma": OMEGA},
            {"n_steps": 3},
            {"interactions": False},
        ]
        for change in changes:
            changed = RydbergFeatureMap(**{**base, **change}).transform(x)
            assert (changed - reference).abs().max().item() > 1e-5, change
        without_pairs = RydbergFeatureMap(**{**base, "correlations": False}).transform(x)
        assert torch.equal(without_pairs, reference[:, :3])

    def test_attributes(self) -> None:
        register = AtomRegister.chain(5, 7.0)
        encoding = PulseEncoding(5, omega=OMEGA)
        feature_map = RydbergFeatureMap(register, encoding, c6=DEFAULT_C6, gamma=0.0)
        assert feature_map.n_features_in == 5
        assert feature_map.n_features_out == 5
        assert feature_map.evolution_time == pytest.approx(0.25, rel=1e-15)
        assert feature_map.register is register
        assert feature_map.encoding is encoding
        assert feature_map.c6 == DEFAULT_C6
        assert feature_map.gamma == 0.0
        assert feature_map.n_steps is None
        assert feature_map.interactions is True
        assert feature_map.correlations is False
        with_pairs = RydbergFeatureMap(
            register, encoding, c6=DEFAULT_C6, gamma=0.0, correlations=True
        )
        assert with_pairs.n_features_out == 5 + 10

    def test_one_atom_has_no_pair_feature(self) -> None:
        feature_map = RydbergFeatureMap(
            AtomRegister([[0.0, 0.0]]),
            PulseEncoding(1, omega=2.0, t=1.0, delta_max=1.0),
            c6=1.0,
            gamma=0.0,
            correlations=True,
        )
        assert feature_map.n_features_out == 1
        assert feature_map.transform([[0.0]]).tolist() == [
            pytest.approx([rabi(2.0, 0.5, 1.0)], abs=1e-13)
        ]


class TestNotAnEncodingLayer:
    def test_it_is_a_preprocessing_step_without_parameters_or_gradients(self) -> None:
        feature_map = RydbergFeatureMap(
            chain(2), PulseEncoding(2, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0
        )
        assert not is_circuit_layer(feature_map)
        assert not isinstance(feature_map, torch.nn.Module)
        x = inputs(2, 2).requires_grad_()
        assert not feature_map.transform(x).requires_grad
        assert not feature_map.states(x).requires_grad


class TestValidation:
    def test_rejects_an_encoding_of_another_width_than_the_register(self) -> None:
        with pytest.raises(ValueError, match="one input per atom"):
            RydbergFeatureMap(chain(3), PulseEncoding(4, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0)

    def test_rejects_a_register_above_the_cap(self) -> None:
        n = MAX_ATOMS + 1
        with pytest.raises(ValueError, match=f"at most {MAX_ATOMS} atoms"):
            RydbergFeatureMap(chain(n), PulseEncoding(n, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0)

    def test_dephasing_needs_a_step_count_at_construction(self) -> None:
        with pytest.raises(ValueError, match="n_steps is required when gamma > 0"):
            RydbergFeatureMap(chain(2), PulseEncoding(2, omega=OMEGA), c6=DEFAULT_C6, gamma=0.1)

    @pytest.mark.parametrize("n_steps", [0, -3, 2.0, True, "10"])
    @pytest.mark.parametrize("gamma", [0.0, 0.5])
    def test_rejects_a_bad_step_count(self, gamma: float, n_steps: Any) -> None:
        with pytest.raises(ValueError, match="n_steps must be an integer >= 1"):
            RydbergFeatureMap(
                chain(2),
                PulseEncoding(2, omega=OMEGA),
                c6=DEFAULT_C6,
                gamma=gamma,
                n_steps=n_steps,
            )

    def test_the_step_count_reaches_the_solver_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every chunk is evolved with the ``t``, ``gamma`` and ``n_steps`` of the map."""
        seen: list[tuple[float, dict[str, Any]]] = []
        solver = feature_map_module.evolve

        def spy(hamiltonian: torch.Tensor, t: float, **kwargs: Any) -> torch.Tensor:
            seen.append((t, kwargs))
            return solver(hamiltonian, t, **kwargs)

        monkeypatch.setattr(feature_map_module, "evolve", spy)
        feature_map = RydbergFeatureMap(
            chain(2),
            PulseEncoding(2, omega=OMEGA, t=0.2),
            c6=DEFAULT_C6,
            gamma=0.5,
            n_steps=13,
        )
        feature_map.transform(inputs(5, 2), chunk_size=2)
        assert seen == [(0.2, {"gamma": 0.5, "n_steps": 13})] * 3

    @pytest.mark.parametrize("gamma", [-0.1, math.nan, math.inf, None, "0"])
    def test_rejects_a_bad_gamma(self, gamma: Any) -> None:
        with pytest.raises(ValueError, match="gamma must be a finite number >= 0"):
            RydbergFeatureMap(
                chain(2), PulseEncoding(2, omega=OMEGA), c6=DEFAULT_C6, gamma=gamma, n_steps=5
            )

    @pytest.mark.parametrize("c6", [math.nan, math.inf, None, "1"])
    def test_rejects_a_bad_c6(self, c6: Any) -> None:
        with pytest.raises(ValueError, match="c6 must be a finite number"):
            RydbergFeatureMap(chain(2), PulseEncoding(2, omega=OMEGA), c6=c6, gamma=0.0)

    @pytest.mark.parametrize("name", ["interactions", "correlations"])
    @pytest.mark.parametrize("value", [1, 0, None, "yes"])
    def test_rejects_a_flag_that_is_not_a_bool(self, name: str, value: Any) -> None:
        with pytest.raises(TypeError, match=f"{name} must be a bool"):
            RydbergFeatureMap(
                chain(2), PulseEncoding(2, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0, **{name: value}
            )

    @pytest.mark.parametrize("encoding", [None, 4, "logistic", [2, 1.0]])
    def test_rejects_another_kind_of_encoding(self, encoding: Any) -> None:
        with pytest.raises(TypeError, match="encoding must be a PulseEncoding"):
            RydbergFeatureMap(chain(2), encoding, c6=DEFAULT_C6, gamma=0.0)

    def test_positions_in_place_of_a_register_are_validated_as_a_register(self) -> None:
        with pytest.raises(ValueError, match="positions must not repeat"):
            RydbergFeatureMap(
                [[0.0, 0.0], [0.0, 0.0]], PulseEncoding(2, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0
            )

    @pytest.mark.parametrize("method", ["transform", "states"])
    def test_rejects_inputs_of_the_wrong_width_or_not_finite(self, method: str) -> None:
        feature_map = RydbergFeatureMap(
            chain(3), PulseEncoding(3, omega=OMEGA), c6=DEFAULT_C6, gamma=0.0
        )
        call = getattr(feature_map, method)
        for shape in [(3,), (2, 2), (2, 4)]:
            with pytest.raises(ValueError, match=r"X must have shape \(n_samples, 3\)"):
                call(torch.zeros(shape))
        with pytest.raises(ValueError, match="X must be finite"):
            call([[0.0, math.nan, 0.0]])
