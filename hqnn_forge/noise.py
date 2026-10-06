"""
hqnn_forge.noise
================
Post-hoc depolarizing noise for evaluating a trained model's NISQ robustness.

A model trained on a noiseless simulator will run on hardware whose gates
depolarize.  :func:`apply_depolarizing_noise` re-executes the model's quantum
layer on ``default.mixed`` with a ``DepolarizingChannel`` of probability ``p``
inserted into the circuit, for the duration of a ``with`` block and without
touching the weights; :func:`noise_sweep` repeats that over a range of ``p``
and collects predictions (and a score, if asked).

Channel
-------
``qml.DepolarizingChannel(p)`` maps ρ → (1−p)ρ + p/3 (XρX + YρY + ZρZ), so on
its own it scales every Pauli expectation by ``1 − 4p/3``; ``p = 3/4`` is fully
depolarizing, and ``p`` is restricted to [0, 3/4].

``position`` chooses where channels go, as in ``qml.noise.insert``:

* ``"all"`` (default) -- after every gate, on every wire the gate acts on: a
  simple gate-noise model whose effect grows with circuit size.
* ``"end"`` -- once on every wire before measurement: readout-style noise that
  damps each ⟨Z_i⟩ by exactly ``1 − 4p/3``.

Other channels
--------------
``channel`` (``noise_channel`` on the layers) selects the channel ``p`` is the
strength of.  What each does to a ⟨Z⟩ readout with ``position="end"``:

========================  =============================  ===================
channel                   effect on ⟨Z⟩ at ``"end"``      ``p`` range
========================  =============================  ===================
``"depolarizing"``        ``(1 − 4p/3) ⟨Z⟩``             [0, 3/4]
``"amplitude_damping"``   ``(1 − p) ⟨Z⟩ + p`` (T1)       [0, 1]
``"phase_damping"``       unchanged (T2 dephasing)       [0, 1]
``"bit_flip"``            ``(1 − 2p) ⟨Z⟩``               [0, 1]
``"phase_flip"``          unchanged                      [0, 1]
========================  =============================  ===================

Dephasing and phase flips commute with a Z measurement, so they act only
through gates that follow them (``position="all"``).  A symmetric readout
error -- each measured bit flipped with probability ``p`` -- is exactly
``"bit_flip"`` at ``"end"``; an asymmetric one (``p(0→1) ≠ p(1→0)``) is not
modelled.  Training noise by Pauli trajectories covers the Pauli channels
(depolarizing, bit flip, phase flip); amplitude and phase damping are not
mixtures of Pauli errors and need ``noise_method="density"``.

``p = 0`` replaces nothing: the original QNode stays in place, so the output
is bit-identical to the noiseless model rather than merely close to it.  It
still counts as an active wrapper, so a layer's training-time noise (below) is
suppressed inside it just as for ``p > 0``, but it does not count as a
replaced QNode: a ``p > 0`` block may be opened inside it.

Training-time noise
-------------------
The encoding layers and hybrid classifiers also take ``noise_level``,
``noise_position`` and ``noise_channel`` at construction.  With ``noise_level > 0`` the layer runs
the same noisy QNode (built by :func:`training_noise_qnode`) whenever it is
in **train mode**, so gradients are computed through the noisy circuit, and
the noiseless QNode in eval mode, like dropout.  Evaluation under noise is
then done with :func:`apply_depolarizing_noise` / :func:`noise_sweep`, which
take precedence over the training-time channel if both are active at once.
``noise_level=0`` (the default) leaves the layer exactly as before.

Two methods, chosen with ``noise_method``:

``"density"`` (default)
    The exact channel: mixed-state simulation on ``default.mixed``,
    differentiated with backprop.  It costs ``O(4^n)`` memory, and training
    keeps a ``batch × 4^n`` complex density matrix for every gate and every
    inserted channel for the backward pass.  At 8 qubits, batch 64 and
    ``position="all"``, one step measured +2.7 GB peak and 50 s, so this is
    practical up to about 6 qubits.
``"trajectories"``
    Pauli-trajectory (Monte Carlo) sampling on the layer's own device and
    differentiation method.  At every channel site each sample independently
    gets one Pauli, drawn with the channel's weights: for depolarizing ``I``
    with probability ``1 − p`` and ``X``, ``Y`` or ``Z`` with ``p/3`` each;
    for bit flip ``I`` with ``1 − p`` and ``X`` with ``p``; for phase flip
    ``I`` with ``1 − p`` and ``Z`` with ``p`` (the ``paulis`` weights in
    :data:`CHANNELS`).  Each mixture *is* its channel, so the output averaged
    over draws equals the ``"density"`` output, and so does the gradient:
    each step's loss gradient is an unbiased estimate of the one
    ``"density"`` computes, at pure-state cost.  The same step measured
    +34 MB and 0.8 s on ``lightning.qubit`` with adjoint (+18 MB and 0.3 s
    noiseless).  The price is gradient variance, as with dropout, which a
    fresh draw every forward pass resembles.  ``noise_trajectories = k``
    averages ``k`` draws per sample, at ``k`` times the cost, to reduce it.

    Measured on one small proxy dataset (#311, breast cancer, 4 and 6
    qubits, 5 seeds; ``docs/results/trajectory-noise-study.md``; the
    benchmark datasets are #414): at
    ``p = 0.01``, and with noise only before measurement, trajectory
    training matched the density channel; at ``p = 0.05`` after every gate
    it was slower to leave the initial loss plateau, so that 3 of 20 runs
    scored a test MCC below 0.5 within 30 epochs and a patience of 10,
    density none.  Re-run for 60 epochs without early stopping, two of the
    three trained (one of them at ``k = 4``) and one, at ``k = 1``, still
    had not (#480).  So keep ``"density"`` where it fits (up to about 6
    qubits), and beyond that use ``"trajectories"`` with
    ``noise_trajectories ≥ 4``, a longer training budget than density
    needs, and check the runs.

    The Pauli at a site is applied as ``RZ(π·z)`` then ``RX(π·x)`` with bits
    ``(x, z)``: ``(0, 0)`` is ``I``, ``(1, 0)`` is ``X``, ``(0, 1)`` is ``Z``
    and ``(1, 1)`` is ``Y`` up to a global phase.  Every sample therefore
    runs the same gate sequence with different angles, so parameter
    broadcasting and the per-sample batch split of the adjoint path apply
    unchanged.  The sites are placed by ``qml.noise.insert`` itself, so they
    are exactly the sites the ``"density"`` path puts channels on.  The
    draws use torch's global RNG, like dropout, so ``torch.manual_seed``
    makes them reproducible -- on an exact layer.  With ``shots`` the
    readouts are also sampled by the device's own generator, which torch does
    not seed, so such a layer repeats only when built with ``seed``.

Shot noise
----------
The other error a device adds is sampling: every expectation value is
estimated from a finite number of measurements.  A layer or classifier built
with ``shots=N`` is sampled that way throughout (training with
``parameter-shift``).  :func:`apply_shots` evaluates a model trained on exact
values with ``N`` shots for the duration of a ``with`` block, and
:func:`shot_sweep` repeats its predictions over a range of shot counts, like
:func:`noise_sweep` over noise levels.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Literal, NamedTuple

import pennylane as qml
import torch
from torch import nn

from hqnn_forge._resolve import resolve_encoding_layer

Position = Literal["all", "end"]
NoiseMethod = Literal["density", "trajectories"]
Channel = Literal["depolarizing", "amplitude_damping", "phase_damping", "bit_flip", "phase_flip"]
MAX_P = 0.75


class _ChannelSpec(NamedTuple):
    op: type[qml.operation.Channel]
    #: Largest meaningful strength: 3/4 is fully depolarizing, 1 is a full
    #: decay, dephasing or flip.
    max_p: float
    #: (I, X, Y, Z) probabilities of a Pauli channel at strength p, or None:
    #: the channels the trajectory sampler can draw.
    paulis: Callable[[float], tuple[float, float, float, float]] | None


CHANNELS: dict[str, _ChannelSpec] = {
    "depolarizing": _ChannelSpec(
        qml.DepolarizingChannel, MAX_P, lambda p: (1 - p, p / 3, p / 3, p / 3)
    ),
    "amplitude_damping": _ChannelSpec(qml.AmplitudeDamping, 1.0, None),
    "phase_damping": _ChannelSpec(qml.PhaseDamping, 1.0, None),
    "bit_flip": _ChannelSpec(qml.BitFlip, 1.0, lambda p: (1 - p, p, 0.0, 0.0)),
    "phase_flip": _ChannelSpec(qml.PhaseFlip, 1.0, lambda p: (1 - p, 0.0, 0.0, p)),
}


def qnode_shots(qnode: object) -> int | None:
    """The shot count ``qnode`` samples with, or ``None`` for exact values."""
    shots = getattr(getattr(qnode, "shots", None), "total_shots", None)
    return shots if isinstance(shots, int) else None


def validate_noise(
    p: float,
    position: str,
    *,
    p_name: str = "p",
    position_name: str = "position",
    channel: str = "depolarizing",
    channel_name: str = "channel",
) -> None:
    """
    Raise ``ValueError`` unless ``channel`` is known, ``0 <= p`` is at most the
    channel's maximum, and ``position`` is known.

    The ``*_name`` arguments are the names the messages use, so a caller that
    exposes these under other names (``noise_level``) is quoted.
    """
    if channel not in CHANNELS:
        raise ValueError(
            f"{channel_name} must be one of {', '.join(map(repr, CHANNELS))}; got {channel!r}."
        )
    max_p = CHANNELS[channel].max_p
    if not 0.0 <= p <= max_p:
        raise ValueError(f"{p_name} must lie in [0, {max_p}] for {channel!r}; got {p}.")
    if position not in ("all", "end"):
        raise ValueError(f"{position_name} must be 'all' or 'end'; got {position!r}.")


def _require_pauli_channel(channel: Channel, label: str) -> None:
    """Raise ``ValueError`` unless ``channel`` can be sampled as Pauli trajectories."""
    if CHANNELS[channel].paulis is None:
        raise ValueError(
            f"{label} is not a mixture of Pauli errors, so it cannot be sampled as "
            f"Pauli trajectories; use noise_method='density'."
        )


def _noisy_qnode(
    qnode: qml.QNode, n_qubits: int, p: float, position: Position, channel: Channel
) -> qml.QNode:
    device = qml.device("default.mixed", wires=n_qubits)
    base = qml.QNode(qnode.func, device, diff_method="backprop", interface="torch")
    return qml.noise.insert(base, CHANNELS[channel].op, p, position=position)


def training_noise_qnode(
    qnode: qml.QNode,
    n_qubits: int,
    p: float,
    position: Position = "all",
    *,
    channel: Channel,
) -> qml.QNode:
    """
    The noisy counterpart of ``qnode`` an encoding layer runs in train mode.

    Same construction as :func:`apply_depolarizing_noise` uses: the layer's
    circuit function on ``default.mixed`` with ``channel`` of strength ``p``
    inserted at ``position``, differentiated with backprop.  The layer's own
    ``device_name`` and ``diff_method`` apply to its noiseless path only;
    mixed-state simulation costs ``O(4^n)`` memory per sample, and training
    through it keeps one such state per operation for backprop (see the module
    docstring).

    Raises
    ------
    ValueError
        If ``channel`` is unknown, ``p`` is 0 or above the channel's maximum,
        or ``position`` is unknown.
    """
    validate_noise(p, position, channel=channel)
    if p == 0.0:
        raise ValueError("training_noise_qnode needs p > 0; p = 0 is the noiseless QNode itself.")
    return _noisy_qnode(qnode, n_qubits, p, position, channel)


@qml.transform
def _pauli_trajectories(
    tape: qml.tape.QuantumScript, p: float, position: Position, channel: Channel
) -> tuple[qml.tape.QuantumScriptBatch, Callable[..., object]]:
    """
    ``tape`` with one randomly drawn Pauli per sample at every channel site.

    The draw happens here, when the tape is built, so every forward pass
    draws afresh; the backward pass differentiates the tape that ran.
    """
    shape = () if tape.batch_size is None else (tape.batch_size,)
    n_draws = 1 if tape.batch_size is None else tape.batch_size
    paulis = CHANNELS[channel].paulis
    assert paulis is not None  # trajectory_noise_qnode refuses the others
    weights = torch.tensor(paulis(p), dtype=torch.float64)

    def pauli_error(wires: object) -> None:
        # 0 = I, 1 = X, 2 = Y, 3 = Z; Y = i·X·Z, so Y sets both bits.
        u = torch.multinomial(weights, n_draws, replacement=True).reshape(shape)
        qml.RZ(torch.pi * ((u == 2) | (u == 3)).to(torch.float64), wires=wires)
        qml.RX(torch.pi * ((u == 1) | (u == 2)).to(torch.float64), wires=wires)

    return qml.noise.insert(tape, pauli_error, (), position=position)


def trajectory_noise_qnode(
    qnode: qml.QNode, p: float, position: Position = "all", *, channel: Channel
) -> qml.QNode:
    """
    ``qnode`` with a Pauli ``channel`` sampled as Pauli trajectories.

    Runs on ``qnode``'s own device and differentiation method; each call draws
    a fresh error pattern per sample (see the module docstring).  The average
    over draws equals :func:`training_noise_qnode`'s output.  Only the Pauli
    channels -- depolarizing, bit flip, phase flip -- are mixtures of Pauli
    errors; amplitude and phase damping are not, and are refused.

    Raises
    ------
    ValueError
        If ``p`` is 0 or out of range, ``position`` is unknown, or ``channel``
        is not a Pauli channel.
    """
    validate_noise(p, position, channel=channel)
    _require_pauli_channel(channel, repr(channel))
    if p == 0.0:
        raise ValueError(
            "trajectory_noise_qnode needs p > 0; p = 0 is the noiseless QNode itself."
        )
    return _pauli_trajectories(qnode, p=p, position=position, channel=channel)


@contextmanager
def _swapped_qnode(qlayer: qml.qnn.TorchLayer, target: qml.QNode) -> Iterator[None]:
    """Temporarily replace ``qlayer.qnode`` with ``target`` for execution."""
    original = qlayer.qnode
    qlayer.qnode = target
    try:
        yield
    finally:
        qlayer.qnode = original


def run_with_training_noise(
    qlayer: qml.qnn.TorchLayer,
    noisy_qnode: qml.QNode,
    x: torch.Tensor,
    n_trajectories: int = 1,
) -> torch.Tensor:
    """
    Evaluate ``qlayer`` on ``x`` with ``noisy_qnode`` in place of its QNode.

    ``n_trajectories > 1`` (the ``"trajectories"`` method only) runs each
    sample that many times and returns the mean, each run with its own draw.

    If :func:`apply_depolarizing_noise` is active on the layer, its channel
    (none at ``p = 0``) is kept and the training-time one is not applied: the
    post-hoc wrapper is the evaluation instrument and wins.  The original
    QNode is restored afterwards, including when the forward pass raises.
    """
    if getattr(qlayer, "_hqnn_noise_depth", 0) > 0:
        return qlayer(x)
    with _swapped_qnode(qlayer, noisy_qnode):
        if n_trajectories == 1:
            return qlayer(x)
        # Sample-major repeat: rows k·i … k·i + k − 1 are sample i's draws.
        batched = x.ndim > 1
        repeated = (
            x.repeat_interleave(n_trajectories, dim=0)
            if batched
            else x.expand(n_trajectories, *x.shape)
        )
        out = qlayer(repeated)
        out = out.reshape(-1, n_trajectories, *out.shape[1:]).mean(dim=1)
        return out if batched else out[0]


def validate_noise_method(method: str, n_trajectories: object) -> None:
    """
    Raise ``ValueError`` unless ``method`` is known and ``n_trajectories`` fits it.

    ``n_trajectories`` must be a positive ``int`` (not ``bool``), and 1 for
    ``"density"``, which is exact and has nothing to average.
    """
    if method not in ("density", "trajectories"):
        raise ValueError(f"noise_method must be 'density' or 'trajectories'; got {method!r}.")
    if (
        isinstance(n_trajectories, bool)
        or not isinstance(n_trajectories, int)
        or n_trajectories < 1
    ):
        raise ValueError(f"noise_trajectories must be a positive int; got {n_trajectories!r}.")
    if method == "density" and n_trajectories != 1:
        raise ValueError(
            f"noise_trajectories={n_trajectories} needs noise_method='trajectories'; the "
            f"density method is exact and has nothing to average."
        )


MAX_TRAINING_NOISE_QUBITS = 6
"""Above this many qubits, a layer built with ``noise_method="density"`` warns."""


class TrainingNoiseMixin:
    """
    Training-time noise for an encoding layer: construction, dispatch and repr.

    The one implementation behind every layer's ``noise_level``,
    ``noise_position``, ``noise_method`` and ``noise_trajectories``.  A layer
    calls :meth:`_init_training_noise` once its QNode exists, runs its circuit
    through :meth:`_run_circuit` (train mode with noise: the noisy QNode, else
    ``qlayer`` itself; eval passes under ``torch.no_grad()`` dispatch to an
    undifferentiated QNode when ``diff_method="adjoint"``) and appends
    :meth:`_noise_repr` to ``extra_repr``.
    Everything else -- validation at construction, the memory warning, the
    QNode swap and its restore, the precedence of
    :func:`apply_depolarizing_noise` -- happens here.
    """

    qlayer: qml.qnn.TorchLayer
    training: bool
    noise_level: float
    noise_channel: Channel
    noise_position: Position
    noise_method: NoiseMethod
    noise_trajectories: int
    _training_noise_qnode: qml.QNode | None
    _eval_qnode_cache: tuple[qml.QNode, qml.QNode] | None

    def _init_training_noise(
        self,
        qnode: qml.QNode,
        n_qubits: int,
        noise_level: float,
        noise_position: Position,
        noise_method: NoiseMethod,
        noise_trajectories: int,
        *,
        shots: int | None = None,
        noise_channel: Channel = "depolarizing",
    ) -> None:
        """
        Validate the options and build the train-mode QNode (``None`` without noise).

        ``noise_channel`` is the channel ``noise_level`` is the strength of; the
        trajectory method samples the Pauli channels only.

        With ``shots``, only ``noise_method="trajectories"`` is accepted: it runs
        the layer's own sampled QNode, while the density method runs the exact
        channel on ``default.mixed``, which would train on exact expectation
        values in a layer meant to see sampled ones.

        Raises ``ValueError`` for an option out of range, whatever
        ``noise_level`` is, so a bad value does not wait for the day the noise
        is switched on.  Warns when the density method is asked for past
        :data:`MAX_TRAINING_NOISE_QUBITS`.
        """
        validate_noise(
            noise_level,
            noise_position,
            p_name="noise_level",
            position_name="noise_position",
            channel=noise_channel,
            channel_name="noise_channel",
        )
        validate_noise_method(noise_method, noise_trajectories)
        if noise_method == "trajectories":
            _require_pauli_channel(noise_channel, f"noise_channel={noise_channel!r}")
        self.noise_channel = noise_channel
        self.noise_level = noise_level
        self.noise_position = noise_position
        self.noise_method = noise_method
        self.noise_trajectories = noise_trajectories
        self._training_noise_qnode = None
        if noise_level == 0.0:
            return
        if shots is not None and noise_method == "density":
            raise ValueError(
                f"noise_method='density' simulates the exact channel and ignores shots={shots}; "
                f"use noise_method='trajectories', which samples the noise on the layer's own "
                f"shot-based QNode."
            )
        if noise_method == "trajectories":
            self._training_noise_qnode = trajectory_noise_qnode(
                qnode, noise_level, noise_position, channel=noise_channel
            )
            return
        if n_qubits > MAX_TRAINING_NOISE_QUBITS:
            warnings.warn(
                f"noise_level > 0 trains on default.mixed, which keeps a batch × 4^n density "
                f"matrix per operation for backprop; at n_qubits={n_qubits} (practical limit "
                f"about {MAX_TRAINING_NOISE_QUBITS}) a training step may run out of memory.  "
                f"noise_method='trajectories' samples the same noise at pure-state cost; "
                f"see hqnn_forge.noise.",
                RuntimeWarning,
                stacklevel=3,
            )
        self._training_noise_qnode = training_noise_qnode(
            qnode, n_qubits, noise_level, noise_position, channel=noise_channel
        )

    @property
    def shots(self) -> int | None:
        """
        The shot count the layer samples with now; ``None`` for exact values.

        Read from the QNode the layer runs, so it follows
        :func:`apply_shots` (and is ``None`` inside a ``p > 0``
        :func:`apply_depolarizing_noise` block, which simulates the exact
        channel) instead of repeating the construction argument.
        """
        return qnode_shots(self.qlayer.qnode)

    def _eval_qnode_for(self, qnode: qml.QNode) -> qml.QNode:
        """
        Return an undifferentiated (``diff_method=None``) clone of *qnode* for
        adjoint passes under ``torch.no_grad()`` (#426, #439).

        Preserves the execution configuration and transforms pipeline (including
        ``broadcast_expand``) while skipping unused adjoint Jacobian evaluations.
        Caches ``(source_qnode, eval_qnode)`` on the layer instance so that QNode
        copies produced by transforms invalidate the cache.
        """
        if getattr(qnode, "diff_method", None) != "adjoint":
            return qnode
        cached = getattr(self, "_eval_qnode_cache", None)
        if cached is not None and cached[0] is qnode:
            return cached[1]
        eval_qnode = qnode.update(diff_method=None)
        self._eval_qnode_cache = (qnode, eval_qnode)
        return eval_qnode

    def _run_circuit(self, x: torch.Tensor) -> torch.Tensor:
        """
        ``qlayer(x)``, through the noisy QNode in train mode when there is one;
        dispatches to an undifferentiated QNode under ``torch.no_grad()`` when
        ``diff_method='adjoint'``.
        """
        if self.training and self._training_noise_qnode is not None:
            if (
                self.noise_method == "density"
                and getattr(self.qlayer, "_hqnn_shots_original", None) is not None
                and getattr(self.qlayer, "_hqnn_noise_depth", 0) == 0
            ):
                # The density QNode simulates the exact channel and would
                # train on exact values inside the shot block.
                raise RuntimeError(
                    "noise_method='density' training noise simulates the exact channel and "
                    "would ignore apply_shots; evaluate in eval mode inside apply_shots, or "
                    "build the layer with noise_method='trajectories', which samples."
                )
            target = self._training_noise_qnode
            if not torch.is_grad_enabled():
                target = self._eval_qnode_for(target)
            return run_with_training_noise(self.qlayer, target, x, self.noise_trajectories)
        target = self.qlayer.qnode
        if not torch.is_grad_enabled():
            target = self._eval_qnode_for(target)
        if target is not self.qlayer.qnode:
            with _swapped_qnode(self.qlayer, target):
                return self.qlayer(x)  # type: ignore[no-any-return]
        return self.qlayer(x)  # type: ignore[no-any-return]

    def _noise_repr(self) -> str:
        """The ``extra_repr`` fragment for the training noise; empty without it."""
        if not self.noise_level:
            return ""
        text = f", noise_level={self.noise_level}, noise_position={self.noise_position!r}"
        if self.noise_channel != "depolarizing":
            text += f", noise_channel={self.noise_channel!r}"
        if self.noise_method != "density":
            text += (
                f", noise_method={self.noise_method!r}, "
                f"noise_trajectories={self.noise_trajectories}"
            )
        return text


@contextmanager
def apply_depolarizing_noise(
    model: nn.Module,
    p: float,
    *,
    position: Position = "all",
    channel: Channel = "depolarizing",
) -> Iterator[nn.Module]:
    """
    Run ``model``'s quantum layer with noise inside the block.

    Named for its default channel; ``channel`` selects any of :data:`CHANNELS`.

    Parameters
    ----------
    model:
        A hybrid classifier or an encoding layer.
    p:
        Strength of each channel: the depolarizing probability in [0, 0.75],
        or the damping / flip probability in [0, 1].
    position:
        ``"all"`` or ``"end"``; see the module docstring.
    channel:
        ``"depolarizing"`` (default), ``"amplitude_damping"``,
        ``"phase_damping"``, ``"bit_flip"`` or ``"phase_flip"``; see the module
        docstring.

    Yields
    ------
    nn.Module
        ``model`` itself, for convenience.

    Raises
    ------
    TypeError
        If ``model`` has no quantum layer.
    ValueError
        If ``p`` or ``position`` is out of range.
    RuntimeError
        If the layer is already running a replaced QNode (nested use).

    Notes
    -----
    The original QNode is restored on exit, including when the block raises.
    Gradients flow through the noisy circuit, so the block can also be used to
    fine-tune under noise.
    """
    validate_noise(p, position, channel=channel)
    _, qlayer, n_qubits = resolve_encoding_layer(model, "apply_depolarizing_noise")
    # Two separate markers.  _hqnn_noise_depth counts open blocks of any p and
    # is what tells run_with_training_noise to skip a layer's train-mode
    # channel.  _hqnn_noise_original is set only while a p > 0 block has
    # replaced the QNode, and is the nesting guard.  p = 0 touches only the
    # depth, so it neither raises inside a p > 0 block (whose channel stays in
    # charge) nor blocks a p > 0 block opened inside it.
    if p > 0.0 and getattr(qlayer, "_hqnn_noise_original", None) is not None:
        raise RuntimeError("apply_depolarizing_noise cannot be nested on the same layer.")
    original = qlayer.qnode
    # The noisy QNode simulates the exact channel on default.mixed, so a
    # sampled layer (built with shots, or inside apply_shots) would silently
    # return exact values here -- the reason density training noise refuses
    # shots too.
    shots = qnode_shots(original)
    if p > 0.0 and shots is not None:
        raise RuntimeError(
            f"apply_depolarizing_noise simulates the exact channel and would ignore the "
            f"layer's shots={shots}; evaluate the noise without shots, or the shots "
            f"without the noise block."
        )
    # Build the replacement before touching the layer. default.mixed refuses
    # more than 23 wires, and a failure here has to leave the layer as it was:
    # arming the guard first would leave it armed with no block to disarm it,
    # and every later call on that layer would raise "cannot be nested".
    noisy = _noisy_qnode(original, n_qubits, p, position, channel) if p > 0.0 else None
    qlayer._hqnn_noise_depth = getattr(qlayer, "_hqnn_noise_depth", 0) + 1
    if noisy is not None:
        qlayer._hqnn_noise_original = original
        qlayer.qnode = noisy
    try:
        yield model
    finally:
        if noisy is not None:
            qlayer.qnode = original
            qlayer._hqnn_noise_original = None
        qlayer._hqnn_noise_depth -= 1


class NoiseSweepPoint(NamedTuple):
    """One noise level of a sweep."""

    p: float
    probabilities: torch.Tensor
    score: float | None


def noise_sweep(
    model: nn.Module,
    X: torch.Tensor,
    ps: Iterable[float],
    *,
    position: Position = "all",
    channel: Channel = "depolarizing",
    y: torch.Tensor | None = None,
    score_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
) -> list[NoiseSweepPoint]:
    """
    Predict ``X`` at each noise level in ``ps``, without retraining.

    Parameters
    ----------
    model:
        A classifier with ``predict_proba`` (the hybrid classifiers).
    X:
        Inputs to evaluate.
    ps:
        Channel strengths, each in the channel's range; consumed once.
    position, channel:
        Passed to :func:`apply_depolarizing_noise`.
    y, score_fn:
        If both are given, ``score_fn(y, probabilities)`` is recorded per level,
        e.g. ``lambda y, p: find_optimal_threshold(y, p).score``.

    Returns
    -------
    list[NoiseSweepPoint]
        In the order of ``ps``.

    Raises
    ------
    ValueError
        If only one of ``y`` and ``score_fn`` is given, or if any level is out
        of range -- checked before the first evaluation, not as the sweep
        reaches it.
    TypeError
        If ``model`` has no ``predict_proba``.
    """
    if (y is None) != (score_fn is None):
        raise ValueError("pass both y and score_fn, or neither.")
    predict = getattr(model, "predict_proba", None)
    if not callable(predict):
        raise TypeError(
            f"noise_sweep needs a model with predict_proba; got {type(model).__name__}."
        )
    # Materialised and range-checked up front: the levels may arrive as a
    # generator, and a bad one at the end would otherwise be found only after
    # every earlier (O(4^n)) evaluation had already been paid for.
    levels = [float(p) for p in ps]
    validate_noise(0.0, position, channel=channel)
    max_p = CHANNELS[channel].max_p
    invalid = [p for p in levels if not 0.0 <= p <= max_p]
    if invalid:
        raise ValueError(f"every p must lie in [0, {max_p}] for {channel!r}; got {invalid}.")
    points = []
    for p in levels:
        with apply_depolarizing_noise(model, p, position=position, channel=channel):
            probs = predict(X)
        score = float(score_fn(y, probs)) if score_fn is not None and y is not None else None
        points.append(NoiseSweepPoint(float(p), probs, score))
    return points


# ---------------------------------------------------------------------------
# Shot noise
# ---------------------------------------------------------------------------


@contextmanager
def apply_shots(model: nn.Module, shots: int | None) -> Iterator[nn.Module]:
    """
    Run ``model``'s quantum layer with ``shots`` samples per circuit inside the block.

    Every expectation value is then estimated from ``shots`` measurements, as
    on hardware: a readout with exact value ``⟨Z⟩`` comes back with standard
    deviation ``sqrt((1 − ⟨Z⟩²) / shots)``.  This evaluates a model trained on
    exact values under sampling, without rebuilding it or touching its
    weights; ``shots=None`` gives exact values again, for a reference point.

    The layer's circuit runs on its own device through a ``parameter-shift``
    QNode, the one differentiation method that samples unbiased gradients, so
    gradients inside the block are sampled too.  A layer with
    ``noise_method="trajectories"`` training noise samples its train-mode
    trajectories with the block's shots as well; one with ``"density"``
    training noise, which simulates the exact channel, raises in train mode
    (eval mode, where training noise is off, is unaffected).  The original
    QNodes are restored on exit, including when the block raises.

    Raises
    ------
    TypeError
        If ``model`` has no quantum layer.
    ValueError
        If ``shots`` is not ``None`` or a positive ``int``.
    RuntimeError
        If the layer is already inside ``apply_shots`` or a ``p > 0``
        :func:`apply_depolarizing_noise` block, whose exact channel shots
        would not apply to.
    """
    # Imported here: hqnn_forge.encoding imports this module.
    from hqnn_forge.encoding._common import expand_batch_dimension, validate_shots

    validate_shots(shots, "parameter-shift")
    layer, qlayer, _ = resolve_encoding_layer(model, "apply_shots")
    if getattr(qlayer, "_hqnn_shots_original", None) is not None:
        raise RuntimeError("apply_shots cannot be nested on the same layer.")
    if getattr(qlayer, "_hqnn_noise_original", None) is not None:
        raise RuntimeError(
            "apply_shots cannot run inside apply_depolarizing_noise: the noise block "
            "simulates the exact channel on default.mixed."
        )
    original = qlayer.qnode
    func = original.func
    # A circuit function that checks its own input gradients (amplitude
    # embedding) holds the construction-time diff_method; under backprop its
    # check would let parameter-shift differentiate the inputs here.
    check = getattr(layer, "_input_gradient_check", None)
    if check is not None:
        circuit = func

        @functools.wraps(circuit)
        def func(inputs: torch.Tensor, *args: object, **kwargs: object) -> object:
            check(inputs, "parameter-shift")
            return circuit(inputs, *args, **kwargs)

    sampled = expand_batch_dimension(
        qml.QNode(
            func,
            original.device,
            interface="torch",
            diff_method="parameter-shift",
            shots=shots,
        ),
        "parameter-shift",
    )
    # Train-mode trajectory noise runs its own copy of the QNode; it is
    # rebuilt on the sampled one, so it samples with the block's shots.
    noise_original = getattr(layer, "_training_noise_qnode", None)
    noise_sampled = (
        trajectory_noise_qnode(
            sampled,
            layer.noise_level,  # type: ignore[attr-defined]
            layer.noise_position,  # type: ignore[attr-defined]
            channel=layer.noise_channel,  # type: ignore[attr-defined]
        )
        if noise_original is not None and getattr(layer, "noise_method", None) == "trajectories"
        else noise_original
    )
    qlayer._hqnn_shots_original = original
    qlayer.qnode = sampled
    if noise_sampled is not noise_original:
        layer._training_noise_qnode = noise_sampled  # type: ignore[attr-defined]
    try:
        yield model
    finally:
        qlayer.qnode = original
        qlayer._hqnn_shots_original = None
        if noise_sampled is not noise_original:
            layer._training_noise_qnode = noise_original  # type: ignore[attr-defined]


class ShotSweepPoint(NamedTuple):
    """One shot count of a sweep."""

    shots: int | None
    #: ``(n_repeats, *predict_proba(X).shape)``: ``(n_repeats, n_samples)``
    #: for a binary classifier, ``(n_repeats, n_samples, n_classes)`` for
    #: :class:`~hqnn_forge.models.MulticlassHybridClassifier`.
    probabilities: torch.Tensor
    #: Scores of the repeated evaluations, or None without ``score_fn``.
    scores: list[float] | None


def shot_sweep(
    model: nn.Module,
    X: torch.Tensor,
    shots: Iterable[int | None],
    *,
    n_repeats: int = 5,
    y: torch.Tensor | None = None,
    score_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
) -> list[ShotSweepPoint]:
    """
    Predict ``X`` ``n_repeats`` times at each shot count, without retraining.

    The repeats show the spread shot noise alone puts on the predictions and
    on the score; ``None`` in ``shots`` gives the exact reference (its repeats
    agree).  Arguments as for :func:`noise_sweep`.

    Raises
    ------
    ValueError
        If ``n_repeats < 1``, only one of ``y`` and ``score_fn`` is given, or a
        shot count is invalid -- checked before the first evaluation.
    TypeError
        If ``model`` has no ``predict_proba``.
    """
    from hqnn_forge.encoding._common import validate_shots

    if (y is None) != (score_fn is None):
        raise ValueError("pass both y and score_fn, or neither.")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be ≥ 1; got {n_repeats}.")
    predict = getattr(model, "predict_proba", None)
    if not callable(predict):
        raise TypeError(
            f"shot_sweep needs a model with predict_proba; got {type(model).__name__}."
        )
    counts = list(shots)
    for count in counts:
        validate_shots(count, "parameter-shift")
    points = []
    for count in counts:
        with apply_shots(model, count):
            runs = torch.stack([predict(X) for _ in range(n_repeats)])
        scores = (
            [float(score_fn(y, row)) for row in runs]
            if score_fn is not None and y is not None
            else None
        )
        points.append(ShotSweepPoint(count, runs, scores))
    return points
