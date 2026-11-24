"""
tests/test_rydberg_global_drive.py
==================================
``GlobalDrivePulseEncoding`` and its use in ``RydbergFeatureMap`` (#732)
against the model of ``docs/rydberg-model.md``.

Every expected value is obtained another way than the feature map obtains it:

* the resonant closed form ``⟨n⟩ = sin²(A/2)`` with the pulse area
  ``A = ∫Ω dt = ΩT``, typed in;
* ``scipy.integrate.solve_ivp`` of the time-dependent Schrödinger equation of
  one atom under the trapezoidal ``Ω(t)`` itself, not its segments;
* the Liouvillian of two atoms as a ``16 × 16`` matrix built from typed-in
  Kronecker products, and its ``scipy.linalg.expm`` per segment;
* the level boundaries ``ln((2m − 1)/(2L − 2m − 1))``, typed in;
* :class:`PulseEncoding`, for the limit without ramp and quantisation;
* the device limits, typed in from the model page's *Hardware realism*.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from fractions import Fraction

import numpy as np
import pytest
import scipy.integrate
import scipy.linalg
import torch

from hqnn_forge.rydberg import (
    AQUILA_LIMITS,
    DEFAULT_C6,
    AtomRegister,
    DeviceLimits,
    GlobalDrivePulseEncoding,
    PulseEncoding,
    RydbergFeatureMap,
    evolve,
    readout,
    rydberg_hamiltonian,
)

F64 = torch.float64

#: The reference Rabi frequency of the model page, 2π × 2 MHz, in rad/µs.
OMEGA = 4 * math.pi
#: The spacing at which the nearest-neighbour interaction equals OMEGA (8.69 µm).
BLOCKADE_RADIUS = (DEFAULT_C6 / OMEGA) ** (1 / 6)
ONE_ATOM = AtomRegister([[0.0, 0.0]])
#: Two atoms with ``V = Ω``.
TWO_ATOMS = AtomRegister.chain(2, BLOCKADE_RADIUS)


def trapezoid(omega: float, t: float, t_ramp: float) -> list[tuple[float, float, object]]:
    """The three pieces of ``Ω(t)`` as ``(start, end, Ω(·))``, typed in from the issue."""
    pieces: list[tuple[float, float, object]] = []
    if t_ramp > 0:
        pieces.append((0.0, t_ramp, lambda s: omega * s / t_ramp))
    pieces.append((t_ramp, t, lambda s: omega))
    if t_ramp > 0:
        pieces.append((t, t + t_ramp, lambda s: omega * (t + t_ramp - s) / t_ramp))
    return pieces


def schroedinger_one_atom(omega: float, delta: float, t: float, t_ramp: float) -> np.ndarray:
    """
    ``ρ(T + t_ramp)`` of one atom from ``i dψ/dt = H(t) ψ``, ``H = [[0, Ω(t)/2], [Ω(t)/2, −Δ]]``.

    Integrated piece by piece over the trapezoid, so that no step straddles a
    kink of ``Ω(t)``, with ``rtol = atol = 1e-12``.
    """
    psi = np.array([1.0, 0.0], dtype=complex)
    for start, end, rabi in trapezoid(omega, t, t_ramp):
        if end <= start:
            continue

        def rhs(s: float, y: np.ndarray, rabi: object = rabi) -> np.ndarray:
            half = rabi(s) / 2  # type: ignore[operator]
            h = np.array([[0.0, half], [half, -delta]], dtype=complex)
            return np.asarray(-1j * (h @ y))

        solution = scipy.integrate.solve_ivp(
            rhs, (start, end), psi, method="DOP853", rtol=1e-12, atol=1e-12
        )
        assert solution.success
        psi = solution.y[:, -1]
    return np.outer(psi, psi.conj())


def two_atom_liouvillian(omega: float, delta: np.ndarray, v: float, gamma: float) -> np.ndarray:
    """
    The Lindblad generator of two atoms acting on ``ρ.reshape(-1)`` (row-major).

    ``vec(A ρ B) = (A ⊗ Bᵀ) vec(ρ)`` for row-major vectorisation, so
    ``−i[H, ρ] → −i (H ⊗ 1 − 1 ⊗ Hᵀ)`` and
    ``γ (n ρ n − ½{n, ρ}) → γ (n ⊗ n − ½ n ⊗ 1 − ½ 1 ⊗ n)`` for a real diagonal ``n``.
    """
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    n = np.diag([0.0, 1.0])
    one = np.eye(2)
    n0, n1 = np.kron(n, one), np.kron(one, n)
    h = (
        omega / 2 * (np.kron(x, one) + np.kron(one, x))
        - delta[0] * n0
        - delta[1] * n1
        + v * n0 @ n1
    ).astype(complex)
    eye = np.eye(4)
    generator = -1j * (np.kron(h, eye) - np.kron(eye, h.T))
    for number in (n0, n1):
        generator += gamma * (
            np.kron(number, number) - 0.5 * np.kron(number, eye) - 0.5 * np.kron(eye, number)
        )
    return generator


def device_encoding(n_features: int = 1, /, **kwargs: object) -> GlobalDrivePulseEncoding:
    settings: dict[str, object] = {"omega": OMEGA, "levels": None, "t_ramp": 0.0}
    settings.update(kwargs)
    return GlobalDrivePulseEncoding(n_features, **settings)  # type: ignore[arg-type]


def device_map(
    register: AtomRegister,
    encoding: GlobalDrivePulseEncoding | PulseEncoding,
    **kwargs: object,
) -> RydbergFeatureMap:
    settings: dict[str, object] = {"c6": DEFAULT_C6, "gamma": 0.0}
    settings.update(kwargs)
    return RydbergFeatureMap(register, encoding, **settings)  # type: ignore[arg-type]


class TestPulseArea:
    """The pulse area ``∫Ω dt = ΩT`` is kept, so a resonant atom ends at ``sin²(ΩT/2)``."""

    @pytest.mark.parametrize("omega_t", [math.pi, 0.6 * math.pi, 2.5])
    @pytest.mark.parametrize(
        ("ramp_fraction", "segments"), [(0.0, None), (0.2, 1), (0.2, 3), (1.0, 4)]
    )
    def test_a_resonant_atom_rotates_by_the_pulse_area(
        self, omega_t: float, ramp_fraction: float, segments: int | None
    ) -> None:
        # One atom at Δ = 0: every H(t) is a multiple of X, they commute, and the
        # rotation angle is ∫Ω dt whatever the ramp or the segmentation.
        t = omega_t / OMEGA
        encoding = device_encoding(
            t=t, levels=2, t_ramp=ramp_fraction * t, n_ramp_segments=segments
        )
        # levels=2 sends every input below 0 to h = 0 exactly: Δ = 0.
        features = device_map(ONE_ATOM, encoding).transform([[-3.0]])
        assert encoding.detunings([[-3.0]]).item() == 0.0
        assert features.item() == pytest.approx(math.sin(omega_t / 2) ** 2, abs=1e-12)

    def test_the_segments_have_the_pulse_area_and_the_total_duration(self) -> None:
        t, t_ramp, n = 0.25, 0.05, 4
        encoding = device_encoding(t=t, t_ramp=t_ramp, n_ramp_segments=n)
        segments = encoding.segments()
        assert len(segments) == 2 * n + 1
        assert sum(tau for _, tau in segments) == pytest.approx(t + t_ramp, abs=1e-15)
        assert sum(o * tau for o, tau in segments) == pytest.approx(OMEGA * t, abs=1e-14)
        # The midpoints of the linear ramps, typed in.
        tau = t_ramp / n
        up = [OMEGA * (k + 0.5) * tau / t_ramp for k in range(n)]
        down = [OMEGA * (t + t_ramp - (t + (k + 0.5) * tau)) / t_ramp for k in range(n)]
        expected = [*zip(up, [tau] * n), (OMEGA, t - t_ramp), *zip(down, [tau] * n)]
        np.testing.assert_allclose(np.array(segments), np.array(expected), rtol=1e-14)
        assert encoding.duration == t + t_ramp

    def test_a_triangle_has_no_plateau(self) -> None:
        segments = device_encoding(t=0.25, t_ramp=0.25, n_ramp_segments=2).segments()
        assert len(segments) == 4

    def test_the_maps_evolution_time_is_the_whole_pulse(self) -> None:
        encoding = device_encoding(t=0.25, t_ramp=0.05, n_ramp_segments=2)
        assert device_map(ONE_ATOM, encoding).evolution_time == 0.3


class TestRampDiscretisation:
    """The midpoint segments against the time-dependent Schrödinger equation."""

    def test_the_error_falls_as_the_square_of_the_segment_count(self) -> None:
        t, t_ramp, x = math.pi / OMEGA, 0.1, 0.3
        delta = math.sqrt(3.0) * OMEGA / (1 + math.exp(-x))
        exact = schroedinger_one_atom(OMEGA, delta, t, t_ramp)
        errors = []
        for n in (2, 4, 8, 16):
            encoding = device_encoding(t=t, t_ramp=t_ramp, n_ramp_segments=n)
            rho = device_map(ONE_ATOM, encoding).states([[x]])[0].numpy()
            errors.append(np.abs(rho - exact).max())
        ratios = [coarse / fine for coarse, fine in itertools.pairwise(errors)]
        # Second order in 1/n: doubling n divides the error by 4.
        assert ratios == pytest.approx([4.0, 4.0, 4.0], rel=0.05), (errors, ratios)
        # Far above the reference's own error, so the ratios measure the segments.
        assert errors[-1] > 1e-6
        # The docstring's leading order: each edge's propagator is off by
        # t_ramp² ΔΩ/(24 n²) in the operator norm, and ρ = UρU† by at most twice the
        # propagator's error, so by t_ramp² ΔΩ/(6 n²) for the two edges.  Measured: 0.29 of it.
        for n, error in zip((2, 4, 8, 16), errors):
            bound = t_ramp**2 * OMEGA * delta / (6 * n**2)
            assert 0.1 * bound < error <= bound

    def test_the_ramped_feature_differs_from_the_square_pulse(self) -> None:
        # The comparison above is not of two square pulses: off resonance the ramp
        # changes the feature by far more than the reference's tolerance.
        t, x = math.pi / OMEGA, 0.3
        delta = math.sqrt(3.0) * OMEGA / (1 + math.exp(-x))
        square = schroedinger_one_atom(OMEGA, delta, t, 0.0)[1, 1].real
        ramped = schroedinger_one_atom(OMEGA, delta, t, 0.1)[1, 1].real
        encoding = device_encoding(t=t, t_ramp=0.1, n_ramp_segments=16)
        feature = device_map(ONE_ATOM, encoding).transform([[x]]).item()
        assert abs(ramped - square) > 1e-2
        assert feature == pytest.approx(ramped, abs=1e-3)


class TestDephasedRamp:
    """With γ > 0: the splitting per segment against the Liouvillian exponential per segment."""

    def test_two_atoms_against_the_liouvillian(self) -> None:
        gamma, t, t_ramp, segments = OMEGA / 2, math.pi / OMEGA, 0.08, 2
        x = torch.tensor([[0.4, -0.7]], dtype=F64)
        encoding = device_encoding(2, t=t, levels=5, t_ramp=t_ramp, n_ramp_segments=segments)
        delta = encoding.detunings(x)[0].numpy()
        v = DEFAULT_C6 / BLOCKADE_RADIUS**6
        rho = np.zeros(16, dtype=complex)
        rho[0] = 1.0
        for omega_k, tau_k in encoding.segments():
            rho = scipy.linalg.expm(two_atom_liouvillian(omega_k, delta, v, gamma) * tau_k) @ rho
        exact = rho.reshape(4, 4)
        errors = []
        # The edge segments of 0.04 µs have 2^2 τ <= T = 0.25 < 2^3 τ, so ceil(n/4) steps:
        # counts that 4 divides double every segment's count together (#782).
        for n_steps in (16, 32, 64):
            fmap = device_map(TWO_ATOMS, encoding, gamma=gamma, n_steps=n_steps)
            assert fmap.segment_steps == (n_steps // 4,) * 2 + (n_steps,) + (n_steps // 4,) * 2
            errors.append(np.abs(fmap.states(x)[0].numpy() - exact).max())
        # Strang splitting, per segment: second order in dt.
        assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05), errors
        assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.05), errors
        assert errors[2] < 1e-4
        # The features are read from that state.
        fmap = device_map(TWO_ATOMS, encoding, gamma=gamma, n_steps=64, correlations=True)
        torch.testing.assert_close(
            fmap.transform(x), readout(fmap.states(x), pairs=True), rtol=0, atol=0
        )


class TestSquarePulseLimit:
    """``levels=None, t_ramp=0`` is :class:`PulseEncoding`, bit for bit."""

    @pytest.mark.parametrize(
        "extra",
        [
            {"gamma": 0.0},
            {"gamma": OMEGA / 2, "n_steps": 7},
            {"gamma": 0.0, "position_jitter": 0.3, "jitter_samples": 2, "jitter_seed": 1},
            {"gamma": 0.0, "readout_error": (0.01, 0.08)},
        ],
    )
    def test_equal_to_pulse_encoding(self, extra: dict[str, object]) -> None:
        x = 1.5 * torch.randn(4, 2, generator=torch.Generator().manual_seed(0), dtype=F64)
        settings: dict[str, object] = {"c6": DEFAULT_C6, "correlations": True, **extra}
        square = RydbergFeatureMap(
            TWO_ATOMS,
            PulseEncoding(2, omega=OMEGA, t=0.3, delta_max=20.0),
            **settings,  # type: ignore[arg-type]
        )
        device = RydbergFeatureMap(
            TWO_ATOMS,
            GlobalDrivePulseEncoding(
                2, omega=OMEGA, t=0.3, delta_local=20.0, levels=None, t_ramp=0.0
            ),
            **settings,  # type: ignore[arg-type]
        )
        assert torch.equal(device.transform(x), square.transform(x))
        assert torch.equal(device.states(x), square.states(x))
        generator = torch.Generator().manual_seed(3)
        shots = device.transform(x, shots=50, generator=generator)
        generator = torch.Generator().manual_seed(3)
        assert torch.equal(shots, square.transform(x, shots=50, generator=generator))


class TestComposition:
    """Position jitter (#731) and detection error (#730) act on the ramped pulse."""

    def test_every_realisation_is_evolved_under_the_ramp(self) -> None:
        # The mixture is the mean of maps without jitter, one per drawn register, each
        # with the ramped encoding; the detection error then acts on that mixture.
        p01, p10 = 0.02, 0.1
        encoding = device_encoding(2, levels=None, t_ramp=0.08, n_ramp_segments=2)
        settings: dict[str, object] = {
            "position_jitter": 0.5,
            "jitter_samples": 3,
            "jitter_seed": 4,
            "readout_error": (p01, p10),
        }
        jittered = device_map(TWO_ATOMS, encoding, **settings)
        x = torch.tensor([[0.4, -0.7], [1.2, 0.3]], dtype=F64)
        drawn = jittered._jitter_registers
        assert len(drawn) == 3
        first, second, third = (device_map(r, encoding).states(x) for r in drawn)
        expected = (first + second + third) / 3
        torch.testing.assert_close(jittered.states(x), expected, rtol=0, atol=1e-14)
        torch.testing.assert_close(
            jittered.transform(x),
            p01 + (1 - p01 - p10) * readout(expected),
            rtol=0,
            atol=1e-14,
        )
        # The ramp matters here: the same draws under the square pulse differ.
        square = device_map(TWO_ATOMS, device_encoding(2, levels=None, t_ramp=0.0), **settings)
        assert (jittered.states(x) - square.states(x)).abs().max() > 1e-3


class TestQuantisation:
    """``h_i = round(σ(x_i)(L − 1))/(L − 1)``."""

    @pytest.mark.parametrize("levels", [2, 5])
    def test_the_level_boundaries(self, levels: int) -> None:
        encoding = device_encoding(levels=levels)
        top = levels - 1
        for m in range(1, levels):
            # σ(x) (L − 1) = m − ½  ⇔  x = ln((2m − 1)/(2L − 2m − 1)), typed in.
            boundary = math.log((2 * m - 1) / (2 * levels - 2 * m - 1))
            below, above = encoding.detunings([[boundary - 1e-9], [boundary + 1e-9]])[:, 0]
            assert below.item() == encoding.delta_local * ((m - 1) / top)
            assert above.item() == encoding.delta_local * (m / top)
        ends = encoding.detunings([[-40.0], [40.0]])[:, 0].tolist()
        assert ends == [0.0, encoding.delta_local]

    def test_inputs_in_the_same_level_give_the_same_features(self) -> None:
        encoding = device_encoding(2, levels=4, t_ramp=0.05, n_ramp_segments=2)
        fmap = device_map(TWO_ATOMS, encoding, correlations=True)
        # Level boundaries at L = 4: ln(1/5), ln(3/3) = 0, ln(5/1).
        x = torch.tensor([[-1.0, 0.1], [-0.3, 1.5], [0.1, 0.1]], dtype=F64)
        features = fmap.transform(x)
        assert torch.equal(features[0], features[1])
        assert (features[0] - features[2]).abs().max() > 1e-3

    def test_none_is_the_logistic_itself(self) -> None:
        x = torch.tensor([[-2.0], [0.0], [0.7]], dtype=F64)
        encoding = device_encoding(levels=None, delta_local=10.0)
        expected = [10.0 / (1 + math.exp(-v)) for v in (-2.0, 0.0, 0.7)]
        assert encoding.detunings(x)[:, 0].tolist() == pytest.approx(expected, rel=1e-15)


class TestDeviceLimits:
    """Each limit is flagged by a configuration that breaks it alone."""

    def test_aquilas_numbers(self) -> None:
        # docs/rydberg-model.md, Hardware realism.
        assert AQUILA_LIMITS == DeviceLimits(
            omega_max=15.8, detuning_max=125.0, duration_max=4.0, omega_slew_max=250.0
        )

    def test_the_reference_configuration_passes(self) -> None:
        # Ω = 4π, Δ_local = √3 Ω = 21.8, T = 0.25 µs and the shortest ramp,
        # t_ramp = Ω / 250 = 0.0503 µs, rounded up.
        encoding = device_encoding(4, t_ramp=0.0503, n_ramp_segments=4, levels=8)
        assert encoding.device_violations() == []

    @pytest.mark.parametrize(
        ("limit", "settings"),
        [
            ("omega_max", {"omega": 16.0, "t_ramp": 0.08}),
            ("detuning_max", {"delta_local": 130.0, "t_ramp": 0.0503}),
            ("duration_max", {"t": 4.0, "t_ramp": 0.0503}),
            # 0.05 µs, 0.0503 rounded down, is a slew of 251 rad/µs².
            ("omega_slew_max", {"t_ramp": 0.05}),
            ("omega_slew_max", {"t_ramp": 0.0}),
        ],
    )
    def test_each_limit_alone(self, limit: str, settings: dict[str, object]) -> None:
        if settings["t_ramp"]:
            settings = {**settings, "n_ramp_segments": 2}
        violations = device_encoding(**settings).device_violations()
        assert len(violations) == 1, violations
        assert violations[0].startswith(f"{limit}:")

    def test_every_limit_at_once_in_field_order(self) -> None:
        encoding = device_encoding(omega=20.0, delta_local=200.0, t=5.0, t_ramp=0.0)
        names = [v.split(":")[0] for v in encoding.device_violations()]
        assert names == ["omega_max", "detuning_max", "duration_max", "omega_slew_max"]

    def test_a_value_at_its_limit_passes(self) -> None:
        encoding = device_encoding(t_ramp=0.0503, n_ramp_segments=2)
        exact = DeviceLimits(
            omega_max=encoding.omega,
            detuning_max=encoding.delta_local,
            duration_max=encoding.duration,
            omega_slew_max=encoding.omega / encoding.t_ramp,
        )
        assert encoding.device_violations(exact) == []

    def test_other_limits(self) -> None:
        loose = DeviceLimits(
            omega_max=100.0, detuning_max=1e3, duration_max=10.0, omega_slew_max=math.inf
        )
        assert device_encoding(t_ramp=0.0).device_violations(loose) == []


def doubled_halvings(tau: float, t: float) -> int:
    """
    ``p``, the largest integer ``>= 0`` with ``2^p τ <= T`` (#782), by doubling ``τ``.

    A float times a power of two is exact in binary, so each comparison is of
    the exact ``2^(p+1) τ``; not the library's ``frexp`` form.
    """
    p = 0
    while tau > 0 and 2 ** (p + 1) * tau <= t:
        p += 1
    return p


def chain_by_hand(
    register: AtomRegister,
    encoding: GlobalDrivePulseEncoding | PulseEncoding,
    x: torch.Tensor,
    gamma: float,
    counts: tuple[int | None, ...],
) -> torch.Tensor:
    """
    ``ρ(T)``: one :func:`evolve` per segment in time order, with the step counts given.

    The segments are the encoding's (``(Ω, T)`` for a :class:`PulseEncoding`),
    the first from ``|g…g⟩``, each other from the state the one before left,
    as #732 documents the map's chain.
    """
    segments = (
        encoding.segments()
        if isinstance(encoding, GlobalDrivePulseEncoding)
        else ((encoding.omega, encoding.t),)
    )
    delta = encoding.detunings(x)
    rho: torch.Tensor | None = None
    for (omega_k, tau_k), n_k in zip(segments, counts, strict=True):
        h = rydberg_hamiltonian(register, omega_k, delta, c6=DEFAULT_C6)
        if rho is None:
            rho = evolve(h, tau_k, gamma=gamma, n_steps=n_k)
        else:
            rho = evolve(h, tau_k, gamma=gamma, n_steps=n_k, rho0=rho)
    assert rho is not None
    return rho


class TestSolverStepsPerSegment:
    """
    #782: with dephasing, segment ``k`` of ``τ_k`` µs takes ``ceil(n / 2^p_k)``
    steps, ``p_k`` the largest integer ``>= 0`` with ``2^p_k τ_k <= T``, so no
    step is longer than ``T/n``; the square pulse keeps ``n``.  The counts are
    typed in, or from :func:`doubled_halvings`, and the chain is rebuilt by
    hand from :func:`evolve` calls.
    """

    @pytest.mark.parametrize(
        ("t", "t_ramp", "segments", "n_steps", "expected"),
        [
            # The benchmark's Aquila preset: τ = 0.0503/32 µs, T/τ = 159.0, p = 7.
            (math.pi / OMEGA, 0.0503, 32, 200, (2,) * 32 + (200,) + (2,) * 32),
            # τ = 0.05 µs: T/τ = 5, p = 2; the plateau of 0.15 µs is longer than T/2.
            (0.25, 0.1, 2, 10, (3, 3, 10, 3, 3)),
            # A plateau of 0.1 µs (p = 1) between edges of 0.15 µs (p = 0).
            (0.25, 0.15, 1, 7, (7, 4, 7)),
            # Triangles: τ = T keeps n, τ = T/2 exactly is in the band p = 1.
            (0.25, 0.25, 1, 7, (7, 7)),
            (0.25, 0.25, 2, 7, (4, 4, 4, 4)),
            (0.25, 0.0, None, 7, (7,)),
        ],
    )
    def test_the_counts_typed_in(
        self,
        t: float,
        t_ramp: float,
        segments: int | None,
        n_steps: int,
        expected: tuple[int, ...],
    ) -> None:
        encoding = device_encoding(t=t, t_ramp=t_ramp, n_ramp_segments=segments)
        fmap = device_map(ONE_ATOM, encoding, gamma=1.0, n_steps=n_steps)
        assert fmap.segment_steps == expected
        assert device_map(ONE_ATOM, encoding).segment_steps is None  # no n_steps
        square = RydbergFeatureMap(
            ONE_ATOM, PulseEncoding(1, omega=OMEGA, t=t), c6=DEFAULT_C6, gamma=1.0, n_steps=9
        )
        assert square.segment_steps == (9,)

    def test_the_bands_are_closed_above(self) -> None:
        """
        ``τ = T/2`` exactly is in the band ``p = 1``, the next float above it
        in ``p = 0``: ``T = 0.25`` and ``t_ramp = 0.125`` are exact in binary,
        and so is the plateau ``T − t_ramp``.  The rule reads the segments'
        floats: with ``t_ramp`` one float below ``0.125`` the plateau
        ``0.25 − t_ramp`` is ``0.125 + 2^-56``, a tie that rounds to even,
        back to ``0.125``, so both are in ``p = 1``.
        """
        below = math.nextafter(0.125, 0.0)
        assert 0.25 - below == 0.125
        cases = [
            (0.125, (3, 3, 3)),
            (math.nextafter(0.125, 1.0), (5, 3, 5)),
            (below, (3, 3, 3)),
        ]
        for t_ramp, expected in cases:
            encoding = device_encoding(t=0.25, t_ramp=t_ramp, n_ramp_segments=1)
            fmap = device_map(ONE_ATOM, encoding, gamma=1.0, n_steps=5)
            assert fmap.segment_steps == expected, t_ramp

    def test_the_rule_and_its_bounds_on_random_drives(self) -> None:
        """
        On 300 random drives and counts: the counts are ``ceil(n / 2^p_k)``
        with ``p_k`` from :func:`doubled_halvings`; every step ``τ_k / n_k``
        is at most ``T/n`` (in exact rational arithmetic); and where ``2^P``
        divides ``n``, every step is longer than ``T/(2n)`` and doubling ``n``
        doubles every count, which the step count's estimate needs.
        """
        rng = random.Random(782)
        for _ in range(300):
            t = rng.uniform(0.01, 2.0)
            t_ramp = rng.choice([t, rng.uniform(0.0, t), rng.uniform(0.0, t / 50)])
            segments = rng.randint(1, 64)
            encoding = device_encoding(t=t, t_ramp=t_ramp, n_ramp_segments=segments)
            drive = encoding.segments()
            p = [doubled_halvings(tau, t) for _, tau in drive]
            n = rng.choice([rng.randint(1, 1000), rng.randint(1, 20) * 2 ** max(p)])
            counts = device_map(ONE_ATOM, encoding, gamma=1.0, n_steps=n).segment_steps
            assert counts == tuple(math.ceil(n / 2**p_k) for p_k in p)
            assert counts is not None
            for (_, tau), n_k in zip(drive, counts, strict=True):
                assert Fraction(tau) * n <= Fraction(t) * n_k  # τ_k / n_k <= T / n
            if n % 2 ** max(p) == 0:
                doubled = device_map(ONE_ATOM, encoding, gamma=1.0, n_steps=2 * n).segment_steps
                assert doubled == tuple(2 * n_k for n_k in counts)
                for (_, tau), n_k in zip(drive, counts, strict=True):
                    assert 2 * Fraction(tau) * n > Fraction(t) * n_k  # τ_k / n_k > T / (2n)

    def test_each_segment_is_evolved_with_its_count(self) -> None:
        """
        The map's states are the chain of ``evolve`` calls with ``(3, 3, 10,
        3, 3)`` bit for bit, and not the chain with 10 steps everywhere, the
        rule before #782.
        """
        gamma = OMEGA / 2
        encoding = device_encoding(2, t=0.25, t_ramp=0.1, n_ramp_segments=2)
        fmap = device_map(TWO_ATOMS, encoding, gamma=gamma, n_steps=10, correlations=True)
        x = torch.tensor([[0.4, -0.7], [1.2, 0.3]], dtype=F64)
        expected = chain_by_hand(TWO_ATOMS, encoding, x, gamma, (3, 3, 10, 3, 3))
        assert torch.equal(fmap.states(x), expected)
        assert torch.equal(fmap.transform(x), readout(expected, pairs=True))
        before = chain_by_hand(TWO_ATOMS, encoding, x, gamma, (10,) * 5)
        assert (fmap.states(x) - before).abs().max() > 1e-6

    @pytest.mark.parametrize(
        "encoding",
        [
            PulseEncoding(2, omega=OMEGA, t=0.3),
            device_encoding(2, t=0.3, t_ramp=0.0),
            device_encoding(2, t=0.3, t_ramp=0.3, n_ramp_segments=1),  # a triangle, τ = T
        ],
        ids=["square", "no-ramp", "triangle"],
    )
    @pytest.mark.parametrize("jitter", [False, True])
    def test_drives_without_short_segments_are_unchanged_bit_for_bit(
        self, encoding: PulseEncoding | GlobalDrivePulseEncoding, jitter: bool
    ) -> None:
        """
        No segment is shorter than ``T/2``, so every one keeps ``n_steps``: the
        map makes the calls of the rule before #782 (``n_steps`` on every
        segment), bit for bit, with and without position jitter.
        """
        gamma, n_steps = OMEGA / 2, 7
        settings: dict[str, object] = (
            {"position_jitter": 0.3, "jitter_samples": 2, "jitter_seed": 5} if jitter else {}
        )
        fmap = device_map(TWO_ATOMS, encoding, gamma=gamma, n_steps=n_steps, **settings)
        x = torch.tensor([[0.4, -0.7], [1.2, 0.3]], dtype=F64)
        counts = (n_steps,) * len(fmap._segments())
        assert fmap.segment_steps == counts
        registers = fmap._jitter_registers if jitter else (TWO_ATOMS,)
        states = [chain_by_hand(r, encoding, x, gamma, counts) for r in registers]
        expected = states[0] if len(states) == 1 else (states[0] + states[1]) / 2
        assert torch.equal(fmap.states(x), expected)

    def test_without_dephasing_the_counts_change_nothing(self) -> None:
        """``γ = 0``: ``evolve`` is exact and uses no steps, so a ramp is the chain of before."""
        encoding = device_encoding(2, t=0.25, t_ramp=0.0503, n_ramp_segments=32)
        fmap = device_map(TWO_ATOMS, encoding, gamma=0.0, n_steps=200)
        x = torch.tensor([[0.4, -0.7], [1.2, 0.3]], dtype=F64)
        assert fmap.segment_steps == (2,) * 32 + (200,) + (2,) * 32
        assert torch.equal(fmap.states(x), chain_by_hand(TWO_ATOMS, encoding, x, 0.0, (200,) * 65))

    def test_the_counts_are_read_from_the_config(self) -> None:
        encoding = device_encoding(2, t=0.25, t_ramp=0.1, n_ramp_segments=2)
        fmap = device_map(TWO_ATOMS, encoding, gamma=OMEGA / 2, n_steps=10)
        rebuilt = RydbergFeatureMap(**json.loads(json.dumps(fmap.get_config())))
        assert rebuilt.segment_steps == fmap.segment_steps == (3, 3, 10, 3, 3)
        x = torch.tensor([[0.4, -0.7], [1.2, 0.3]], dtype=F64)
        assert torch.equal(rebuilt.transform(x), fmap.transform(x))

    def test_the_chunking_changes_the_features_by_rounding_only(self) -> None:
        """As for the square pulse (``TestChunks.ROUNDING`` in the feature map's tests)."""
        encoding = device_encoding(3, t=0.25, t_ramp=0.1, n_ramp_segments=3)
        register = AtomRegister.chain(3, BLOCKADE_RADIUS)
        fmap = device_map(register, encoding, gamma=OMEGA / 2, n_steps=12, correlations=True)
        x = 1.5 * torch.randn(7, 3, generator=torch.Generator().manual_seed(2), dtype=F64)
        whole = fmap.transform(x, chunk_size=7)
        for chunk_size in (1, 2, 3):
            assert (fmap.transform(x, chunk_size=chunk_size) - whole).abs().max() <= 1e-12

    @pytest.mark.parametrize(("gamma_over_omega", "measured"), [(1.0, 1.59e-6), (10.0, 5.58e-5)])
    def test_the_preset_drive_against_the_chained_liouvillian(
        self, gamma_over_omega: float, measured: float
    ) -> None:
        """
        The Aquila preset's drive at the default ``n_steps = 200``, two atoms
        at ``V = Ω``: the edges take 2 steps per segment.  Against the
        exponential of the Liouvillian per segment, chained, the features are
        off by ``measured`` (to 5 %; the rule before #782, 200 steps on every
        segment, gave 2.3e-6 and 5.1e-5), within the preset's ``tol = 1e-4``,
        and a third of their difference from ``n_steps = 100`` (1 and 100
        steps: every count doubles) is that error to 5 %.
        """
        gamma = gamma_over_omega * OMEGA
        encoding = device_encoding(2, t=math.pi / OMEGA, t_ramp=0.0503, n_ramp_segments=32)
        x = torch.tensor([[0.4, -0.7], [2.0, 1.1], [-1.5, 0.2]], dtype=F64)
        v = DEFAULT_C6 / BLOCKADE_RADIUS**6
        exact = []
        for delta in encoding.detunings(x).numpy():
            rho = np.zeros(16, dtype=complex)
            rho[0] = 1.0
            for omega_k, tau_k in encoding.segments():
                rho = (
                    scipy.linalg.expm(two_atom_liouvillian(omega_k, delta, v, gamma) * tau_k) @ rho
                )
            exact.append(rho.reshape(4, 4))
        features = readout(torch.tensor(np.array(exact)), pairs=True)
        fmap = device_map(TWO_ATOMS, encoding, gamma=gamma, n_steps=200, correlations=True)
        coarse = device_map(TWO_ATOMS, encoding, gamma=gamma, n_steps=100, correlations=True)
        assert fmap.segment_steps == (2,) * 32 + (200,) + (2,) * 32
        assert coarse.segment_steps == (1,) * 32 + (100,) + (1,) * 32
        error = (fmap.transform(x) - features).abs().max().item()
        assert error == pytest.approx(measured, rel=0.05)
        assert error <= 1e-4
        estimate = (coarse.transform(x) - fmap.transform(x)).abs().max().item() / 3
        assert estimate == pytest.approx(error, rel=0.05)


class TestConfig:
    def test_round_trip_through_json_and_the_map(self) -> None:
        encoding = device_encoding(2, t=0.3, levels=6, t_ramp=0.07, n_ramp_segments=3)
        fmap = device_map(TWO_ATOMS, encoding, gamma=1.0, n_steps=5, correlations=True)
        config = json.loads(json.dumps(fmap.get_config()))
        rebuilt = RydbergFeatureMap(**config)
        assert isinstance(rebuilt.encoding, GlobalDrivePulseEncoding)
        assert rebuilt.get_config() == fmap.get_config()
        x = torch.tensor([[0.3, -1.2], [2.0, 0.0]], dtype=F64)
        assert torch.equal(rebuilt.transform(x), fmap.transform(x))
        assert config["encoding"] == {
            "n_features": 2,
            "omega": OMEGA,
            "t": 0.3,
            "delta_local": math.sqrt(3.0) * OMEGA,
            "levels": 6,
            "t_ramp": 0.07,
            "n_ramp_segments": 3,
        }

    def test_a_pulse_encoding_config_is_unchanged(self) -> None:
        fmap = RydbergFeatureMap(TWO_ATOMS, PulseEncoding(2, omega=OMEGA), c6=DEFAULT_C6, gamma=0)
        rebuilt = RydbergFeatureMap(**json.loads(json.dumps(fmap.get_config())))
        assert type(rebuilt.encoding) is PulseEncoding


class TestValidation:
    @pytest.mark.parametrize(
        ("settings", "match"),
        [
            ({"levels": 1}, "levels"),
            ({"levels": True}, "levels"),
            ({"levels": 2.0}, "levels"),
            ({"t_ramp": -0.01}, "t_ramp"),
            ({"t_ramp": 0.3, "n_ramp_segments": 2}, "t_ramp"),
            ({"t_ramp": 0.05}, "n_ramp_segments"),
            ({"t_ramp": 0.05, "n_ramp_segments": 0}, "n_ramp_segments"),
            ({"t_ramp": 0.0, "n_ramp_segments": 2}, "n_ramp_segments"),
            ({"delta_local": 0.0}, "delta_local"),
        ],
    )
    def test_refused_settings_name_the_argument(
        self, settings: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            device_encoding(**settings)

    def test_levels_and_t_ramp_have_no_default(self) -> None:
        with pytest.raises(TypeError, match="levels"):
            GlobalDrivePulseEncoding(1, omega=OMEGA, t_ramp=0.0)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="t_ramp"):
            GlobalDrivePulseEncoding(1, omega=OMEGA, levels=None)  # type: ignore[call-arg]
