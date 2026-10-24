"""
hqnn_forge.encoding.analog_rydberg
==================================
A trainable analog encoding layer: a neutral-atom register evolving under a
piecewise-constant global laser pulse, with the data in a local detuning
(#420).  No gates: the trainable parameters are the pulse's Rabi frequency,
detuning and phase in each time segment, and the atoms are entangled by the
Rydberg interaction of their geometry alone (Henriet et al. 2020).

Units: ħ = 1, angular frequencies (Ω, Δ, ``C6/r⁶``) in rad/µs, lengths in µm,
times in µs.  The model, its sign conventions and its basis ordering are
those of ``docs/rydberg-model.md`` and :mod:`hqnn_forge.rydberg`; the drive
phase φ is the one the page writes out for this layer.

The Hamiltonian of segment k
----------------------------
The pulse has ``K`` segments of equal duration τ.  In segment ``k``, for a
sample ``x ∈ ℝ^N`` on ``N`` atoms::

    H_k(x) = Ω_k/2 · Σ_i (cos φ_k X_i − sin φ_k Y_i)
             − Σ_i (Δ_k + δ_i(x_i)) n_i
             + Σ_{i<j} V_ij n_i n_j

    n_i = |r⟩⟨r|_i,        V_ij = C6 / |r_i − r_j|⁶,        δ_i(x_i) = δ_max · σ(x_i)

* **Ω_k ≥ 0** is the Rabi frequency (not half of it: a resonant atom has
  ``⟨n⟩(t) = sin²(Ωt/2)``), global and trainable.
* **Δ_k** is the global detuning, trainable; **δ_i(x_i)** the local detuning
  of atom ``i``, which carries input ``x_i`` and nothing else, in every
  segment.  Δ is the laser frequency minus the transition frequency and
  enters as ``−Δ n``: a positive detuning lowers ``|r⟩``.  The local term is
  the encoding of ``docs/rydberg-model.md`` (``Δ_i = Δ_max · σ(x_i)``,
  one-signed, ``δ_max = √3 Ω_max`` by default), here added to a trainable
  global detuning.
* **φ_k** is the drive phase, global and trainable:
  ``cos φ X − sin φ Y = e^{iφ} |g⟩⟨r| + e^{−iφ} |r⟩⟨g|``.
* ``C6 > 0`` is repulsive, :data:`~hqnn_forge.rydberg.DEFAULT_C6` (⁸⁷Rb
  ``70S₁/₂``, ``2π × 862 690 MHz µm⁶ = 5.42 × 10⁶ rad/µs · µm⁶``) unless
  given.  ``interactions=False`` drops the last term and nothing else.

Each ``H_k`` is built by :func:`~hqnn_forge.rydberg.rydberg_hamiltonian`
(with its ``phase`` argument); no second Hamiltonian builder exists here.

Evolution
---------
``H`` is constant within a segment, so the Schrödinger equation
``i dψ/dt = H ψ`` is solved exactly segment by segment::

    U_k = e^(−i H_k τ),        ψ(T) = U_{K−1} ⋯ U_1 U_0 |g…g⟩,        T = K τ

``U_k`` is the propagator of :func:`~hqnn_forge.rydberg.evolve`
(``hqnn_forge.rydberg.lindblad._propagator``): from ``torch.linalg.eigh``
when no gradient is tracked, and from ``torch.linalg.matrix_exp`` when one
is, whose backward pass is correct at a degenerate spectrum (#505).  Each
``U_k`` is a dense ``2^N × 2^N`` matrix per sample, built in ``complex128``.

Basis, wires and readout
------------------------
Atom ``i`` is row ``i`` of the register's positions, tensor factor ``i`` from
the left, PennyLane wire ``i`` and input column ``i``; atom 0 is the most
significant bit of a basis index, which is the order ``qml.QubitUnitary``
reads a matrix on wires ``0, …, N−1``.  ``|g⟩ = |0⟩`` and ``|r⟩ = |1⟩``, so
``Z_i = 1 − 2 n_i`` and the outputs are::

    ⟨Z_i⟩ = 1 − 2 ⟨n_i⟩ ∈ [−1, 1],        i = 0, …, N − 1  (readout="all")

or ``⟨Z_0⟩`` alone (``readout="first"``), in the order of the atoms.

Bounds by reparameterisation
----------------------------
The trainable tensor ``weights`` of shape ``(K, 3)`` holds unbounded raw
values ``(u_k, v_k, w_k)``; the pulse is::

    Ω_k = Ω_max · σ(u_k)       ∈ (0, Ω_max)          σ(u) = 1 / (1 + e^(−u))
    Δ_k = Δ_max · tanh(v_k)    ∈ (−Δ_max, Δ_max)
    φ_k = w_k                  (any real; it enters through cos and sin)

and its inverse, which :meth:`AnalogRydbergLayer.set_pulse` uses to start
from a stated pulse::

    u_k = log(Ω_k) − log(Ω_max − Ω_k)        (the logit of Ω_k/Ω_max)
    v_k = atanh(Δ_k / Δ_max)

defined exactly on the open ranges above.  The bounds are not clipped:
``dΩ/du = Ω (Ω_max − Ω)/Ω_max`` and ``dΔ/dv = Δ_max (1 − tanh² v)`` are
positive for every finite raw value, so a gradient never vanishes because a
bound was hit (in double precision σ saturates only beyond ``|u| ≈ 37``, and
tanh beyond ``|v| ≈ 19``).  The input encoding ``δ_max · σ(x_i)`` is
bounded the same way.  Without any training (the constructor's pulse),
every raw value is 0: ``Ω_k = Ω_max/2``, ``Δ_k = 0`` and ``φ_k = 0``.

Only phase differences matter
-----------------------------
Let ``R(α) = Π_i e^{−iα n_i}``.  It commutes with every ``n_i``, so with the
detuning and interaction terms and with each ``Z_i``, and
``R(α)† |g⟩⟨r| R(α) = e^{−iα} |g⟩⟨r|``, so ``R(α)† H_k(φ_k) R(α) =
H_k(φ_k − α)`` for every segment.  Then, writing ``U(φ)`` for the whole
evolution with the phases ``φ = (φ_0, …, φ_{K−1})``,
``U(φ) = R(α) U(φ − α) R(α)†`` and ``R(α)† |g…g⟩ = |g…g⟩``, so ``⟨Z_i⟩`` is
the same for the phases ``φ_k`` and ``φ_k − α``: **shifting every phase by
the same amount changes no output**, only the differences ``φ_k − φ_0``
reach the readout.  Hence the phase gradients sum to zero,
``Σ_k ∂⟨Z_i⟩/∂φ_k = 0``, and with a single segment its phase is inert.  All
``K`` phases are kept so that every segment has the same three parameters,
as the issue's pulse parameterisation asks.

Defaults
--------
``Ω_max = 4π rad/µs`` is the reference Rabi frequency of
``docs/rydberg-model.md``; ``Δ_max = δ_max = √3 Ω_max`` its detuning bound;
``τ = π/Ω_max``, so a segment at full amplitude is a π pulse; and the
register is :func:`hqnn_forge.circuits.rydberg_chain`, the open chain at
6 µm of ``entangler="rydberg"``, where nearest neighbours interact with
``V/Ω_max = 9.25``.  These are inside QuEra Aquila's bounds
(``Ω ≤ 15.8 rad/µs``, ``|Δ| ≤ 125 rad/µs``, at most 4 µs, i.e. 16
segments at the default τ), but the per-atom local detuning is an
idealisation (see the *Hardware realism* section of
``docs/rydberg-model.md``).

Size
----
The state and every ``U_k`` are dense: ``2^N`` amplitudes and ``4^N``
matrix entries per sample and segment.  The layer accepts at most
:data:`~hqnn_forge.rydberg.MAX_ATOMS` (10) atoms, the cap of the package's
Hamiltonian, and refuses more at construction (16 MiB per matrix at 10
atoms, 4 times more per further atom).  Training keeps every ``U_k`` of the
batch for the backward pass.

The circuit
-----------
The QNode applies one ``qml.QubitUnitary(U_k)`` per segment on all wires and
returns ``⟨Z_i⟩``, so the layer satisfies the encoding-layer contract
(:mod:`hqnn_forge._encoding_contract`): ``qlayer`` is a TorchLayer whose
QNode takes ``inputs`` and ``weights``, and ``forward(x)`` is
``qlayer(prepare_inputs(x))``.  The propagators are computed in torch inside
the QNode, so it runs on ``default.qubit`` with backprop only.  Through
that circuit the library's noise options apply: a channel at
``noise_position="all"`` acts after every pulse segment on every atom, at
``"end"`` once before measurement; a readout error maps each ``⟨Z_i⟩``.
They are gate-model channels placed between segments, not the continuous
dephasing of ``docs/rydberg-model.md``.

References
----------
* Henriet et al. (2020)
  "Quantum computing with neutral atoms", Quantum 4, 327.
* Wurtz et al. (2023) "Aquila: QuEra's 256-qubit neutral-atom quantum
  computer", arXiv:2306.11727.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import pennylane as qml
import torch
import torch.nn as nn

from hqnn_forge.circuits import rydberg_chain
from hqnn_forge.encoding._common import (
    DeviceName,
    DiffMethod,
    Readout,
    backend_repr,
    check_inputs,
    measure_z,
    readout_wires,
    resolve_device,
    shots_repr,
    validate_seed,
)
from hqnn_forge.noise import Channel, NoiseMethod, Position, TrainingNoiseMixin
from hqnn_forge.rydberg.hamiltonian import DEFAULT_C6, MAX_ATOMS, rydberg_hamiltonian
from hqnn_forge.rydberg.lindblad import _propagator
from hqnn_forge.rydberg.register import AtomRegister, _finite_number

__all__ = ["AnalogRydbergLayer", "build_analog_rydberg_qnode"]

#: Default upper bound of the Rabi frequency, rad/µs: the reference
#: ``Ω = 4π`` (2π × 2 MHz) of ``docs/rydberg-model.md``.
_DEFAULT_OMEGA_MAX: float = 4 * math.pi


class _AnalogPulse(NamedTuple):
    """The fixed settings the circuit function closes over."""

    register: AtomRegister
    c6: float
    interactions: bool
    omega_max: float
    detuning_max: float
    local_detuning_max: float
    segment_duration: float


def _bounded_pulse(
    weights: torch.Tensor, omega_max: float, detuning_max: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(Ω_k, Δ_k, φ_k)``, each of shape ``(K,)`` in ``float64``, from raw ``weights`` ``(K, 3)``."""
    raw = weights.to(torch.float64)
    omega = omega_max * torch.sigmoid(raw[:, 0])
    detuning = detuning_max * torch.tanh(raw[:, 1])
    return omega, detuning, raw[:, 2]


def _segment_unitaries(
    pulse: _AnalogPulse, inputs: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """``U_k = e^(−i H_k(x) τ)`` for every segment: shape ``(K, *batch, 2^N, 2^N)``."""
    omega, detuning, phase = _bounded_pulse(weights, pulse.omega_max, pulse.detuning_max)
    local = pulse.local_detuning_max * torch.sigmoid(inputs.to(torch.float64))
    unitaries = []
    for k in range(weights.shape[0]):
        h = rydberg_hamiltonian(
            pulse.register,
            omega[k],
            detuning[k] + local,
            c6=pulse.c6,
            interactions=pulse.interactions,
            phase=phase[k],
        )
        unitaries.append(_propagator(h, pulse.segment_duration))
    return torch.stack(unitaries)


def _make_analog_circuit(
    pulse: _AnalogPulse, readout: Readout
) -> Callable[[torch.Tensor, torch.Tensor], list[qml.measurements.ExpectationMP]]:
    """
    The bare quantum function: one ``QubitUnitary(U_k)`` per segment, then ⟨Z_i⟩.

    ``inputs`` has shape ``(N,)`` or ``(batch, N)``, ``weights`` ``(K, 3)``;
    see the module docstring for what they become.
    """
    wires = list(range(pulse.register.n_atoms))

    def circuit(
        inputs: torch.Tensor, weights: torch.Tensor
    ) -> list[qml.measurements.ExpectationMP]:
        for unitary in _segment_unitaries(pulse, inputs, weights):
            qml.QubitUnitary(unitary, wires=wires)
        return measure_z(len(wires), readout)

    return circuit


def build_analog_rydberg_qnode(
    register: AtomRegister,
    *,
    c6: float = DEFAULT_C6,
    interactions: bool = True,
    omega_max: float = _DEFAULT_OMEGA_MAX,
    detuning_max: float | None = None,
    local_detuning_max: float | None = None,
    segment_duration: float | None = None,
    readout: Readout = "all",
    seed: int | None = None,
) -> qml.QNode:
    """
    The QNode of :class:`AnalogRydbergLayer`, on ``default.qubit`` with backprop.

    Its arguments are ``inputs`` and ``weights`` (shape ``(K, 3)``, any
    ``K ≥ 1``); the keyword options are those of the layer, which documents
    them and their defaults.

    Raises
    ------
    ValueError
        As :class:`AnalogRydbergLayer` for the same options.
    """
    pulse = _pulse_settings(
        register, c6, interactions, omega_max, detuning_max, local_detuning_max, segment_duration
    )
    readout_wires(register.n_atoms, readout)  # validates readout
    device = resolve_device("default.qubit", register.n_atoms, seed=seed)
    return qml.QNode(
        func=_make_analog_circuit(pulse, readout),
        device=device,
        diff_method="backprop",
        interface="torch",
    )


def _pulse_settings(
    register: AtomRegister,
    c6: float,
    interactions: bool,
    omega_max: float,
    detuning_max: float | None,
    local_detuning_max: float | None,
    segment_duration: float | None,
) -> _AnalogPulse:
    """Validate the fixed settings and fill in the defaults of the module docstring."""
    if not isinstance(register, AtomRegister):
        raise TypeError(f"register must be an AtomRegister; got {type(register).__name__}.")
    if register.n_atoms > MAX_ATOMS:
        raise ValueError(
            f"AnalogRydbergLayer simulates a dense 2^n state and accepts at most "
            f"{MAX_ATOMS} atoms; got {register.n_atoms} (4^{register.n_atoms} matrix "
            f"entries per sample and segment)."
        )
    omega_max = _finite_number(omega_max, "omega_max", positive=True)
    default_detuning = math.sqrt(3) * omega_max
    return _AnalogPulse(
        register=register,
        c6=_finite_number(c6, "c6"),
        interactions=bool(interactions),
        omega_max=omega_max,
        detuning_max=(
            default_detuning
            if detuning_max is None
            else _finite_number(detuning_max, "detuning_max", positive=True)
        ),
        local_detuning_max=(
            default_detuning
            if local_detuning_max is None
            else _finite_number(local_detuning_max, "local_detuning_max", positive=True)
        ),
        segment_duration=(
            math.pi / omega_max
            if segment_duration is None
            else _finite_number(segment_duration, "segment_duration", positive=True)
        ),
    )


def _per_segment(value: Any, name: str, n_segments: int) -> torch.Tensor:
    """``value`` as a finite ``float64`` tensor of shape ``(n_segments,)``; a scalar is repeated."""
    tensor = torch.as_tensor(value, dtype=torch.float64).detach()
    if tensor.ndim == 0:
        tensor = tensor.expand(n_segments)
    if tensor.shape != (n_segments,):
        raise ValueError(
            f"{name} must be a scalar or hold one value per segment ({n_segments}); "
            f"got shape {tuple(tensor.shape)}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite; got {tensor.tolist()}.")
    return tensor


class AnalogRydbergLayer(TrainingNoiseMixin, nn.Module):
    """
    Analog neutral-atom encoding: a trainable piecewise-constant pulse on a register.

    ``n_qubits`` atoms evolve from ``|g…g⟩`` under ``n_segments`` segments of
    the Rydberg Hamiltonian, each with a trainable global Rabi frequency
    ``Ω_k ∈ (0, omega_max)``, global detuning ``Δ_k ∈ (−detuning_max,
    detuning_max)`` and drive phase ``φ_k``, while input ``x_i`` sets the
    local detuning ``local_detuning_max · σ(x_i)`` of atom ``i``.  The
    outputs are ``⟨Z_i⟩ = 1 − 2⟨n_i⟩``.  The Hamiltonian, the bounds and their
    inverse, the readout order and the defaults are derived in the module
    docstring (:mod:`hqnn_forge.encoding.analog_rydberg`).

    Input range: any finite real; σ maps it into ``(0, 1)``.  The models feed
    ``tanh(·)·π ∈ (−π, π)``, which uses 4 % to 96 % of the local range.

    Parameters
    ----------
    n_qubits:
        Number of atoms ``N``, 1 to :data:`~hqnn_forge.rydberg.MAX_ATOMS`;
        also the input width.
    n_segments:
        Number of pulse segments ``K``, an integer ``>= 1``.  ``weights`` has
        shape ``(n_segments, 3)``; ``n_layers`` is the same number.
    register:
        Atom positions in µm, an :class:`~hqnn_forge.rydberg.AtomRegister` of
        ``n_qubits`` atoms.  Default: :func:`hqnn_forge.circuits.rydberg_chain`,
        an open chain at 6 µm.
    c6:
        ``C6`` in rad/µs · µm⁶.  Default :data:`~hqnn_forge.rydberg.DEFAULT_C6`.
    interactions:
        ``False`` drops the ``V_ij n_i n_j`` term: the non-interacting control.
    omega_max:
        Upper bound of ``Ω_k`` in rad/µs, ``> 0``.  Default ``4π``.
    detuning_max:
        Bound of ``|Δ_k|`` in rad/µs, ``> 0``.  Default ``√3 · omega_max``.
    local_detuning_max:
        Range of the data-carrying local detuning in rad/µs, ``> 0``.
        Default ``√3 · omega_max``.
    segment_duration:
        Duration τ of each segment in µs, ``> 0``.  Default ``π/omega_max``.
    readout:
        ``"all"`` (default): ``⟨Z_i⟩`` of every atom.  ``"first"``: atom 0 only.
    device_name, diff_method:
        ``"auto"`` or ``"default.qubit"``, and ``"auto"`` or ``"backprop"``:
        the propagators are computed in torch inside the QNode, so only
        backprop on ``default.qubit`` differentiates them.  Accepted for the
        models' sake, which pass their resolved names; anything else raises.
    noise_level, noise_position, noise_method, noise_trajectories, noise_channel, readout_error:
        Training-time noise as in
        :class:`~hqnn_forge.encoding.QuantumEncodingLayer`; a channel at
        ``"all"`` acts after every pulse segment (see the module docstring).
    shots:
        Must be ``None``: the layer computes exact expectation values, since
        backprop, the only method that differentiates its propagators,
        cannot run on samples.
    seed:
        Seed of the device, as for the other layers.

    Attributes
    ----------
    qlayer : qml.qnn.TorchLayer
        The circuit, with the trainable raw pulse ``weights`` ``(n_segments, 3)``.
    n_qubits, n_features, n_outputs, n_segments, n_layers : int
    register : AtomRegister
    c6, omega_max, detuning_max, local_detuning_max, segment_duration, duration : float
        ``duration = n_segments · segment_duration``, in µs.
    interactions : bool

    Raises
    ------
    TypeError
        If ``register`` is not an ``AtomRegister``.
    ValueError
        For more than :data:`~hqnn_forge.rydberg.MAX_ATOMS` atoms, a register
        of another size than ``n_qubits``, a bound or duration that is not a
        finite number ``> 0``, ``n_segments < 1``, ``shots``, a device or
        differentiation method other than the above, or an unknown
        ``readout`` or noise option.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.encoding import AnalogRydbergLayer
    >>> layer = AnalogRydbergLayer(n_qubits=3, n_segments=2)
    >>> layer.pulse().shape  # (Ω_k, Δ_k, φ_k) per segment
    torch.Size([2, 3])
    >>> layer(torch.zeros(4, 3)).shape
    torch.Size([4, 3])
    """

    n_qubits: int
    n_features: int
    n_segments: int
    n_layers: int
    n_outputs: int
    readout: Readout
    register: AtomRegister
    c6: float
    interactions: bool
    omega_max: float
    detuning_max: float
    local_detuning_max: float
    segment_duration: float
    duration: float
    seed: int | None
    _pulse: _AnalogPulse

    def __init__(
        self,
        n_qubits: int = 4,
        n_segments: int = 2,
        *,
        register: AtomRegister | None = None,
        c6: float = DEFAULT_C6,
        interactions: bool = True,
        omega_max: float = _DEFAULT_OMEGA_MAX,
        detuning_max: float | None = None,
        local_detuning_max: float | None = None,
        segment_duration: float | None = None,
        readout: Readout = "all",
        device_name: DeviceName = "auto",
        diff_method: DiffMethod = "auto",
        noise_level: float = 0.0,
        noise_position: Position = "all",
        noise_method: NoiseMethod = "density",
        noise_trajectories: int = 1,
        shots: int | None = None,
        noise_channel: Channel = "depolarizing",
        seed: int | None = None,
        readout_error: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(n_qubits, bool)
            or not isinstance(n_qubits, numbers.Integral)
            or n_qubits < 1
        ):
            raise ValueError(f"n_qubits must be an integer >= 1; got {n_qubits!r}.")
        if (
            isinstance(n_segments, bool)
            or not isinstance(n_segments, numbers.Integral)
            or n_segments < 1
        ):
            raise ValueError(f"n_segments must be an integer >= 1; got {n_segments!r}.")
        if n_qubits > MAX_ATOMS:
            raise ValueError(
                f"AnalogRydbergLayer simulates a dense 2^n state and accepts at most "
                f"{MAX_ATOMS} atoms; got {n_qubits} (4^{n_qubits} matrix entries per sample "
                f"and segment)."
            )
        if shots is not None:
            raise ValueError(
                f"AnalogRydbergLayer computes exact expectation values: its propagators are "
                f"built in torch and differentiated by backprop, which cannot run on samples; "
                f"got shots={shots!r}."
            )
        if device_name not in ("auto", "default.qubit") or diff_method not in (
            "auto",
            "backprop",
        ):
            raise ValueError(
                f"AnalogRydbergLayer builds its propagators in torch inside the QNode, so it is "
                f"differentiated by backprop on default.qubit only; got "
                f"device_name={device_name!r}, diff_method={diff_method!r}."
            )
        register = rydberg_chain(int(n_qubits)) if register is None else register
        if isinstance(register, AtomRegister) and register.n_atoms != n_qubits:
            raise ValueError(
                f"register has {register.n_atoms} atoms; n_qubits={n_qubits} needs one atom "
                f"per qubit."
            )
        pulse = _pulse_settings(
            register,
            c6,
            interactions,
            omega_max,
            detuning_max,
            local_detuning_max,
            segment_duration,
        )

        self.n_qubits = int(n_qubits)
        self.n_features = self.n_qubits
        self.n_segments = int(n_segments)
        self.n_layers = self.n_segments
        self.readout = readout
        self.n_outputs = len(readout_wires(self.n_qubits, readout))
        self.register = pulse.register
        self.c6 = pulse.c6
        self.interactions = pulse.interactions
        self.omega_max = pulse.omega_max
        self.detuning_max = pulse.detuning_max
        self.local_detuning_max = pulse.local_detuning_max
        self.segment_duration = pulse.segment_duration
        self.duration = self.n_segments * self.segment_duration
        self._pulse = pulse

        qnode = build_analog_rydberg_qnode(
            pulse.register,
            c6=pulse.c6,
            interactions=pulse.interactions,
            omega_max=pulse.omega_max,
            detuning_max=pulse.detuning_max,
            local_detuning_max=pulse.local_detuning_max,
            segment_duration=pulse.segment_duration,
            readout=readout,
            seed=seed,
        )
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (self.n_segments, 3)})
        # The constructor's pulse: every raw value 0, i.e. Ω_k = Ω_max/2, Δ_k = 0, φ_k = 0.
        with torch.no_grad():
            self.qlayer.weights.zero_()
        self._init_training_noise(
            qnode,
            self.n_qubits,
            noise_level,
            noise_position,
            noise_method,
            noise_trajectories,
            shots=None,
            noise_channel=noise_channel,
            readout_error=readout_error,
        )
        self.seed = validate_seed(seed)

    # ------------------------------------------------------------------
    def pulse(self) -> torch.Tensor:
        """
        The pulse the layer runs: ``(Ω_k, Δ_k, φ_k)`` per segment, shape ``(K, 3)``.

        ``float64``, detached, in rad/µs and radians, computed from the raw
        ``weights`` by the reparameterisation of the module docstring.
        """
        omega, detuning, phase = _bounded_pulse(
            self.qlayer.weights.detach(), self.omega_max, self.detuning_max
        )
        return torch.stack([omega, detuning, phase], dim=-1)

    def set_pulse(
        self,
        omega: float | Sequence[float] | torch.Tensor | None = None,
        detuning: float | Sequence[float] | torch.Tensor | None = None,
        phase: float | Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """
        Set the raw ``weights`` so that the layer runs this pulse.

        Each argument is a scalar for every segment, one value per segment,
        or ``None`` to keep the current one.  The raw values come from the
        inverse of the reparameterisation (module docstring),
        ``u = log Ω − log(Ω_max − Ω)`` and ``v = atanh(Δ/Δ_max)``, and are
        stored in the dtype of ``weights``: in ``float32`` (the default) the
        pulse that runs, :meth:`pulse`, is the requested one to about 1e-7
        relative.

        Raises
        ------
        ValueError
            If a value is not finite, has another length than
            ``n_segments``, or lies outside the open range the
            reparameterisation reaches: ``0 < Ω < omega_max`` and
            ``|Δ| < detuning_max``.
        """
        weights = self.qlayer.weights
        k = self.n_segments
        columns: list[tuple[int, torch.Tensor]] = []
        if omega is not None:
            value = _per_segment(omega, "omega", k)
            if not bool(((value > 0) & (value < self.omega_max)).all()):
                raise ValueError(
                    f"omega must lie in (0, omega_max) = (0, {self.omega_max}); "
                    f"got {value.tolist()}."
                )
            columns.append((0, torch.log(value) - torch.log(self.omega_max - value)))
        if detuning is not None:
            value = _per_segment(detuning, "detuning", k)
            if not bool((value.abs() < self.detuning_max).all()):
                raise ValueError(
                    f"detuning must lie in (-detuning_max, detuning_max) = "
                    f"({-self.detuning_max}, {self.detuning_max}); got {value.tolist()}."
                )
            columns.append((1, torch.atanh(value / self.detuning_max)))
        if phase is not None:
            columns.append((2, _per_segment(phase, "phase", k)))
        with torch.no_grad():
            for column, raw in columns:
                weights[:, column] = raw.to(weights.dtype)

    def local_detunings(self, x: torch.Tensor) -> torch.Tensor:
        """``δ_i(x_i) = local_detuning_max · σ(x_i)`` in rad/µs, ``float64``, the shape of ``x``."""
        x = self.prepare_inputs(x)
        return self.local_detuning_max * torch.sigmoid(x.to(torch.float64))

    def unitaries(self, x: torch.Tensor) -> torch.Tensor:
        """
        The segment propagators ``U_k = e^(−i H_k(x) τ)``: shape ``(K, *batch, 2^N, 2^N)``.

        ``complex128``, with gradients to the weights and ``x``; segment 0
        acts first.  Exactly the matrices the circuit applies.
        """
        return _segment_unitaries(self._pulse, self.prepare_inputs(x), self.qlayer.weights)

    def states(self, x: torch.Tensor) -> torch.Tensor:
        """
        ``ψ(T) = U_{K−1} ⋯ U_0 |g…g⟩`` in torch, shape ``(*batch, 2^N)``, ``complex128``.

        The state the circuit prepares before its readout, without noise;
        basis ordering as in the module docstring (atom 0 the most
        significant bit).
        """
        unitaries = self.unitaries(x)
        psi = unitaries[0][..., :, 0]  # U_0 |g…g⟩ is the first column of U_0
        for unitary in unitaries[1:]:
            psi = (unitary @ psi.unsqueeze(-1)).squeeze(-1)
        return psi

    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check that ``x`` has ``n_qubits`` finite features; the values are used as given.

        This is the classical step ``forward`` applies before the QNode; the
        local detuning ``local_detuning_max · σ(x_i)`` is part of the circuit.
        """
        check_inputs(x, self.n_qubits)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evolve a batch of inputs under the pulse and return ⟨Z_i⟩."""
        x = self.prepare_inputs(x)
        return self._run_circuit(x)

    def extra_repr(self) -> str:
        options = ""
        if not self.interactions:
            options += ", interactions=False"
        if self.readout != "all":
            options += f", readout={self.readout!r}"
        options += self._noise_repr()
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_segments={self.n_segments}, "
            f"segment_duration={self.segment_duration:.4g}, "
            f"omega_max={self.omega_max:.4g}, "
            f"n_params={sum(p.numel() for p in self.parameters())}{options}"
            f"{shots_repr(self.shots, self.seed)}{backend_repr(self.qlayer)}"
        )
