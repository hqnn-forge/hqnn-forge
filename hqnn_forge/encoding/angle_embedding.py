"""
hqnn_forge.encoding.angle_embedding
====================================
Core quantum encoding module for projecting classical tabular feature vectors
into an n-qubit Hilbert space via angle (rotation) embedding followed by a
strongly-entangled variational ansatz.

Design Rationale
----------------
* **Angle Embedding** maps each input feature x_i ∈ ℝ to a Pauli-rotation angle
  (default: RX) on qubit i.  This keeps the encoding linear in feature values and
  avoids the exponential "Hilbert-space crowding" of more aggressive embeddings.

* **Strongly-Entangling Ansatz** — after embedding, L layers of a CNOT ring
  followed by per-qubit SU(2) Rot(φ, θ, ω) gates are applied.  This produces a
  high-expressibility ansatz while keeping the depth O(n * L).

* **Adjoint Differentiation** — the QNode is configured for the `adjoint` method
  on a `lightning.qubit` device.  Adjoint diff computes exact gradients in a
  single forward + backward pass and scales as O(p) in the number of parameters p,
  making it strictly superior to the parameter-shift rule for state-vector sims.
  The GPU-accelerated ``lightning.gpu`` and ``lightning.kokkos`` devices support
  the same method; :func:`~hqnn_forge.encoding._common.resolve_device` falls back
  through ``lightning.qubit`` to ``default.qubit`` when a backend is not
  installed or has no usable hardware.

* **Initialisation** — weights are *not* initialised here; callers should use
  `hqnn_forge.initializers.restricted_normal_init_` on the returned layer.  Note
  what that buys for this circuit (measured, see `hqnn_forge.initializers`):
  more initial gradient variance only for inputs near zero, and no escape from
  its exponential decay with qubit count.  The cascaded CNOT ring puts every
  qubit in the backward light cone of each ⟨Z_i⟩ within two layers (of ⟨Z_0⟩
  and ⟨Z_{n-1}⟩ within one), so at the default depth the per-qubit readouts
  are global costs in the sense of Cerezo et al. (2021).  The escape is the
  entangler: with ``entangler="brickwork"`` the readouts stay local at
  shallow depth and the total gradient variance does not fall from 4 to 8
  qubits (see :func:`apply_variational_layers`).

References
----------
* Schuld et al. (2020) "Circuit-centric quantum classifiers", PRA 101, 032308.
* Sim et al. (2019) "Expressibility and entangling capability of PQCs", Adv. Quantum
  Technol. 2, 1900070.
* Jones & Gacon (2020) "Efficient calculation of gradients in classical simulations
  of variational quantum algorithms" arXiv:2009.02823.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pennylane as qml
import torch
import torch.nn as nn

from hqnn_forge.encoding._common import (
    DEVICE_FAILURES,
    FALLBACK_CHAIN,
    KNOWN_DEVICES,
    DeviceName,
    DiffMethod,
    Entangler,
    Readout,
    RotationAxis,
    apply_variational_layers,
    backend_repr,
    check_inputs,
    expand_batch_dimension,
    is_out_of_memory,
    measure_z,
    readout_wires,
    reset_device_fallback,
    resolve_backend,
    resolve_device,
    shots_repr,
    validate_circuit_options,
    validate_device_shots,
    validate_seed,
    validate_shots,
    variational_weight_shape,
)
from hqnn_forge.noise import Channel, NoiseMethod, Position, TrainingNoiseMixin

logger = logging.getLogger(__name__)

# QuantumEncodingLayer and build_encoding_qnode are defined here; the rest are
# re-exported from hqnn_forge.encoding._common, where they live since #306.
# KNOWN_DEVICES was always there; it is re-exported only so the API reference
# can render it at a public path (docs/api/encoding.md).
__all__ = [
    "FALLBACK_CHAIN",
    "KNOWN_DEVICES",
    "AngleEmbeddingQNode",
    "DeviceName",
    "DiffMethod",
    "Entangler",
    "QuantumEncodingLayer",
    "Readout",
    "RotationAxis",
    "apply_variational_layers",
    "build_encoding_qnode",
    "check_inputs",
    "measure_z",
    "readout_wires",
    "reset_device_fallback",
    "validate_circuit_options",
    "variational_weight_shape",
]

# Re-exported: these lived here until #306.  The underscored names are the old
# spellings, kept for one release for code that imported them.
_resolve_device = resolve_device
_expand_batch_dimension = expand_batch_dimension
_is_out_of_memory = is_out_of_memory
_DEVICE_FAILURES = DEVICE_FAILURES

# ---------------------------------------------------------------------------
# Raw QNode function
# ---------------------------------------------------------------------------


def _make_angle_embedding_circuit(
    n_qubits: int,
    n_layers: int,
    rotation: RotationAxis,
    entangler: Entangler = "ring",
    readout: Readout = "all",
) -> Callable[[torch.Tensor, torch.Tensor], list[qml.measurements.ExpectationMP]]:
    """
    Factory returning the *bare quantum function* (not yet a QNode) that
    implements the angle-embedding feature map + strongly-entangled ansatz.

    The returned function has the signature::

        circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[ExpectationMP]

    where

    * ``inputs``  — shape ``(n_qubits,)`` — the pre-processed feature vector.
    * ``weights`` — shape :func:`variational_weight_shape`: ``(n_layers,
                    n_qubits, 3)``, the ``qml.Rot`` angles (φ, θ, ω) per layer
                    and qubit, or ``(n_layers, n_qubits)``, the ``RY`` angles,
                    for ``entangler="hardware_efficient"``.

    Called inside a QNode it records one ``qml.expval(PauliZ)`` measurement per
    readout wire; the QNode turns them into the expectation values.

    Circuit structure (per layer ℓ = 0 … L-1)
    ------------------------------------------
    1. **Feature embedding** (applied before the first layer only):
       ``AngleEmbedding(inputs, wires, rotation=rotation)``
       → RX(x_i) on wire i, ∀ i ∈ {0, …, n_qubits-1}.

    2. **CNOT entangling ring**:
       CNOT(i → i+1 mod n) for i ∈ {0, …, n_qubits-1}, applied as a cascade.
       This creates a cyclic entanglement graph with all-to-all reachability
       within a single layer.  The flip side is the backward light cone of
       the readouts: through one ring it covers qubits {0, …, i+1} for ⟨Z_i⟩
       with 0 < i < n-1, and all n qubits for ⟨Z_0⟩ (Z_0 ↦ Z_1⋯Z_{n-1} in the
       Heisenberg picture) and ⟨Z_{n-1}⟩; through two rings it covers all n
       qubits for every i.  From 2 layers on, the ⟨Z_i⟩ readouts therefore do
       not enjoy the local-cost gradient bounds of Cerezo et al. (2021); see
       `hqnn_forge.initializers` for what is measured instead.

       The same conjugation decides which *features* a readout sees.  After
       one layer with the default ``rotation="X"``, the X and Y terms the
       ``Rot`` mixes in land, for i < n-1, on operators with an ``X`` on some
       wire, whose expectation in the RX-embedded product state is 0, so only
       the Z image survives:

           ⟨Z_0⟩ = c_0(w) · cos x_1 ⋯ cos x_{n-1}      (no x_0)
           ⟨Z_i⟩ = c_i(w) · cos x_0 ⋯ cos x_i          (0 < i < n-1)

       For i = n-1 the wrap-around CNOT(n-1, 0) carries Y_{n-1} to
       ∝ Y_0 Y_1 Z_2 ⋯ Z_{n-2} Y_{n-1}, and ⟨Y⟩ = -sin x, so ⟨Z_{n-1}⟩ =
       a(w) · cos x_0 ⋯ cos x_{n-1} + b(w) · sin x_0 sin x_1 cos x_2 ⋯
       cos x_{n-2} sin x_{n-1}.

       Readout 0 is blind to its own feature, and carries a product of n-1
       cosines, which is small for inputs spread over (-π, π).  With
       ``rotation="Y"`` the X terms survive (⟨X⟩ = sin x) and ⟨Z_0⟩ does see
       x_0; with ``entangler="strongly_entangling"`` the image of Z_0 leaves
       wire 0 before its ``Rot`` is reached, so ⟨Z_0⟩ ignores x_0 under either
       rotation.  For both cascades, from two layers on every readout sees
       every feature under RX or RY.  (Under ``rotation="Z"`` no readout sees
       any feature at any depth: RZ on |0⟩ is only a phase, which is why
       :func:`build_encoding_qnode` refuses it, #212.)

       Under ``readout="all"`` the blind spot costs nothing, since readouts
       1 … n-1 together cover x_0.  Under ``readout="first"`` use
       ``n_layers >= 2`` with either cascade (#150).  ``entangler="brickwork"``
       is *not* a remedy there: its CNOT(0, 1) has wire 0 as control, so Z_0
       keeps its own wire and each ⟨Z_i⟩ sees x_i after one layer, but the
       same narrow light cone leaves ⟨Z_0⟩ seeing only x_0 (RX) or x_0, x_1
       (RY), and at 5 qubits still missing x_2 … x_4 (RX) or x_4 (RY) after
       two layers.  Nor is ``entangler="hardware_efficient"``: its ``CZ``
       ladder is diagonal, so ⟨Z_i⟩ reaches a neighbour only through the
       X_i its ``RY`` mixes in, which the ``CZ`` gates dress with Z_{i±1}.
       After L layers ⟨Z_i⟩ sees x_{i-L} … x_{i+L}, except that under RX
       (⟨X⟩ = 0) a single layer leaves it seeing x_i alone; under
       ``readout="first"``, ⟨Z_0⟩ thus sees L + 1 features (one at L = 1
       under RX).

    3. **Per-qubit SU(2) rotation block**:
       ``qml.Rot(φ, θ, ω, wires=i)`` applies Rz(ω)·Ry(θ)·Rz(φ), covering the
       full Bloch sphere.  This is the most expressive single-qubit gate.
       In the **last** layer the trailing Rz(ω) commutes with the ⟨Z⟩
       readouts, so with ``readout="all"`` those ``n_qubits`` angles can
       never change the output.  With ``readout="first"`` far more is dead:
       for this ring ansatz the last layer's ``Rot`` on every wire but 0
       acts after anything that reaches ⟨Z_0⟩, and so do some earlier ω.
       At 4 qubits and 2 layers that is 4 of 24 weights under ``"all"`` and
       12 under ``"first"``; ``circuit_summary`` reports the count as
       ``n_inert_params``.  With ``entangler="strongly_entangling"`` the
       last-layer ``Rot`` comes before that layer's CNOTs, which carry Z_0
       onto other wires (at 4 qubits, to wire 2, whose last-layer ``Rot`` is
       then live while wire 0's is dead), and the count under ``"first"`` is
       only a lower bound: 8 of the 12 weights autograd finds dead at 4
       qubits and 2 layers.  With ``entangler="brickwork"`` the narrow light
       cone of ⟨Z_0⟩ leaves more dead under ``"first"``: 16 of 24 at 4
       qubits and 2 layers (the last-layer ``Rot`` off wire 0, wire 0's ω,
       and layer 0's ``Rot`` on wires 2 and 3), 17 by autograd.  The weight
       tensor keeps its ``(n_layers, n_qubits, 3)`` shape for every
       ``Rot`` entangler; ``"hardware_efficient"`` has one ``RY`` angle per
       qubit, shape ``(n_layers, n_qubits)``.  None of its angles is inert
       under ``"all"``; under ``"first"`` the k-th layer counted back from
       the readout (k = 0 the last) leaves max(0, n − 1 − k) dead, 6 of 12
       at 4 qubits and 3 layers, and ``n_inert_params`` matches autograd.

    4. **Measurement**:
       Returns ``[qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]``.
       Each output is ∈ [-1, 1], giving an n_qubits–dimensional real vector
       suitable as input to a classical head.

    Parameters
    ----------
    n_qubits:
        Number of qubits (= number of input features).
    n_layers:
        Number of variational layers L.  Depth = O(n_qubits * n_layers).
    rotation:
        Pauli axis used by AngleEmbedding: ``"X"`` | ``"Y"``.
        :func:`build_encoding_qnode` refuses ``"Z"``, a phase on ``|0⟩``.
    entangler:
        ``"ring"`` (steps 2 and 3 above), ``"strongly_entangling"``
        (``qml.StronglyEntanglingLayers``: Rot first, then a CNOT ring of
        range ``ℓ mod (n-1) + 1``), ``"brickwork"`` (nearest-neighbour
        CNOT pairs, no wrap-around, then Rot) or ``"hardware_efficient"`` (a
        CZ ladder, then ``RY``; ``weights`` of shape ``(n_layers, n_qubits)``).
        See :func:`apply_variational_layers`.
    readout:
        ``"all"`` (step 4 above) or ``"first"`` (``[⟨Z_0⟩]`` only, as in the
        published SHNN).

    Returns
    -------
    callable
        A plain Python function suitable for ``@qml.qnode`` decoration.
    """
    validate_circuit_options(n_qubits, entangler, readout, rotation)

    def circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[qml.measurements.ExpectationMP]:
        # ── 1. Angle embedding: map x_i → RX(x_i)|0⟩ on wire i ──────────
        qml.AngleEmbedding(
            features=inputs,
            wires=range(n_qubits),
            rotation=rotation,
        )

        # ── 2 & 3. Variational layers ────────────────────────────────────
        apply_variational_layers(weights, n_qubits, n_layers, entangler)

        # ── 4. Measurement: Pauli-Z expectation on the readout wires ─────
        return measure_z(n_qubits, readout)

    return circuit


# ---------------------------------------------------------------------------
# Public QNode factory
# ---------------------------------------------------------------------------


def build_encoding_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation: RotationAxis = "X",
    device_name: DeviceName = "auto",
    diff_method: DiffMethod = "auto",
    entangler: Entangler = "ring",
    readout: Readout = "all",
    shots: int | None = None,
    seed: int | None = None,
) -> qml.QNode:
    """
    Build and return a PennyLane QNode for the angle-embedding feature map.

    The QNode is bound to the device :func:`resolve_device` returns for
    *device_name* and configured for the specified differentiation method.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  Must equal the dimensionality of the input feature
        vector after PCA reduction.  Default: 8.
    n_layers:
        Number of entangling + rotation layers in the VQC ansatz.
        More layers increase expressibility but deepen the circuit.  Default: 2.
    rotation:
        Pauli rotation axis for AngleEmbedding: ``"X"`` (default) or ``"Y"``.
        ``"Z"`` raises: a single ``RZ`` on ``|0⟩`` is only a phase, so the
        layer would not depend on its inputs.
    device_name:
        PennyLane device name.  Default ``"auto"``: ``default.qubit`` up to
        12 qubits, ``lightning.qubit`` above (see
        :func:`~hqnn_forge.encoding.resolve_backend`).  The simulators in
        :data:`~hqnn_forge.encoding.angle_embedding.KNOWN_DEVICES` fall back along
        ``lightning.qubit → default.qubit`` with a warning per step when
        unavailable; any other name (a plugin or hardware) is constructed as
        given, and PennyLane's error surfaces if it cannot be.  Hardware
        needs ``shots`` and ``diff_method="parameter-shift"``.
    diff_method:
        Differentiation strategy:

        - ``"auto"``            — the default; chosen by device, see
          :func:`~hqnn_forge.encoding.resolve_backend`.
        - ``"adjoint"``         — exact, O(p) memory; fastest on lightning.
        - ``"parameter-shift"`` — exact, hardware-compatible, O(p) circuit evals.
        - ``"backprop"``        — auto-diff through simulator; requires default.qubit.
        - ``"finite-diff"``     — approximate; avoid for training.
    entangler:
        ``"ring"`` (default), ``"strongly_entangling"``, ``"brickwork"`` or
        ``"hardware_efficient"``; see :func:`apply_variational_layers`.
    readout:
        ``"all"`` (default): ⟨Z_i⟩ on every qubit.  ``"first"``: ⟨Z_0⟩ only.
    shots, seed:
        Finite-shot sampling and the device seed, as for
        :class:`QuantumEncodingLayer`.

    Returns
    -------
    qml.QNode
        A callable QNode with signature
        ``(inputs: Tensor, weights: Tensor) -> Tensor``
        where outputs are ⟨Z_i⟩ expectation values, shape ``(n_qubits,)``
        (or ``(1,)`` with ``readout="first"``).

    Raises
    ------
    ValueError
        If ``n_qubits < 2`` (minimum for a meaningful entangling ring), or if
        ``rotation``, ``entangler`` or ``readout`` is not one of the values
        above, or if ``rotation="Z"`` -- all checked here, before the circuit
        first runs.

    Examples
    --------
    >>> qnode = build_encoding_qnode(n_qubits=8, n_layers=2)
    >>> x = torch.rand(8)
    >>> w = torch.zeros(2, 8, 3)
    >>> result = qnode(x, w)  # list of 8 expectation values
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2 for the CNOT entangling ring; got {n_qubits}.")
    if rotation == "Z":
        # The inputs are embedded once, on |0…0⟩, where RZ(x) only multiplies
        # each wire by a phase: the state entering the ansatz is the same for
        # every x, so the layer would be a constant.  DataReuploadingLayer
        # can use "Z" from its second upload on.
        raise ValueError(
            'rotation="Z" would make the layer ignore its inputs: a single RZ embedding '
            "acts on |0…0⟩, where it is only a global phase, so every input gives the same "
            'state and the input gradients are zero.  Use "X" or "Y", or '
            'DataReuploadingLayer(rotation="Z", n_layers >= 2).'
        )

    device_name, diff_method = resolve_backend(device_name, diff_method, n_qubits, shots=shots)
    validate_shots(shots, diff_method)
    device = resolve_device(device_name, n_qubits, seed=seed)
    validate_device_shots(device, shots)
    circuit_fn = _make_angle_embedding_circuit(n_qubits, n_layers, rotation, entangler, readout)

    qnode = qml.QNode(
        func=circuit_fn,
        device=device,
        diff_method=diff_method,
        interface="torch",  # enables PyTorch autograd interop
        shots=shots,
    )
    qnode = expand_batch_dimension(qnode, diff_method)

    logger.info(
        "QNode built | device=%s | qubits=%d | layers=%d | diff=%s | rotation=%s | "
        "entangler=%s | readout=%s",
        device.name,
        n_qubits,
        n_layers,
        diff_method,
        rotation,
        entangler,
        readout,
    )
    return qnode


# ---------------------------------------------------------------------------
# PyTorch nn.Module wrapper
# ---------------------------------------------------------------------------


class QuantumEncodingLayer(TrainingNoiseMixin, nn.Module):
    """
    A PyTorch ``nn.Module`` that wraps the angle-embedding QNode as a fully
    differentiable layer via ``pennylane.qnn.TorchLayer``.

    The layer owns the variational weights as ``nn.Parameter`` objects.  During
    the forward pass the classical ``inputs`` tensor is embedded into the quantum
    circuit and the Pauli-Z expectation values are returned as a real-valued
    tensor, enabling direct composition with classical ``nn.Linear`` layers.

    Weight Shapes
    -------------
    The internal ``TorchLayer`` registers one trainable parameter:

    +-----------+------------------------------------+
    | Name      | Shape                              |
    +===========+====================================+
    | ``weights``| ``(n_layers, n_qubits, 3)``       |
    +-----------+------------------------------------+

    ``(n_layers, n_qubits)`` for ``entangler="hardware_efficient"``; see
    :func:`variational_weight_shape`.

    **Important**: Call ``hqnn_forge.initializers.restricted_normal_init_``
    on ``layer.qlayer.weights`` immediately after construction to obtain
    small-angle initial values (see :mod:`hqnn_forge.initializers` for what
    that heuristic does and does not guarantee).

    Parameters
    ----------
    n_qubits:
        Number of qubits / input feature dimensions.  Default: 8.
    n_layers:
        Number of entangling + rotation blocks in the VQC ansatz.  Default: 2.
        At 1, ⟨Z_0⟩ does not see feature 0 under the default ring and RX,
        so ``readout="first"`` wants 2 or more; see step 2 of
        :func:`_make_angle_embedding_circuit`.
    rotation:
        Pauli axis for AngleEmbedding: ``"X"`` | ``"Y"``; ``"Z"`` raises, see
        :func:`build_encoding_qnode`.
    device_name:
        PennyLane device name.  Default ``"auto"``: ``default.qubit`` up to
        12 qubits, ``lightning.qubit`` above (see
        :func:`~hqnn_forge.encoding.resolve_backend`).  The simulators in
        :data:`~hqnn_forge.encoding.angle_embedding.KNOWN_DEVICES` fall back along
        ``lightning.qubit → default.qubit`` with a warning per step when
        unavailable; any other name (a plugin or hardware) is constructed as
        given, and PennyLane's error surfaces if it cannot be.  Hardware
        needs ``shots`` and ``diff_method="parameter-shift"``.
    diff_method:
        ``"auto"`` (default) picks by device: ``"backprop"`` on
        ``default.qubit``, ``"adjoint"`` on lightning, ``"parameter-shift"``
        with ``shots`` or on any other device.  Or one of ``"adjoint"``,
        ``"parameter-shift"``, ``"backprop"``, ``"finite-diff"``.
    entangler:
        ``"ring"`` (default), ``"strongly_entangling"``, ``"brickwork"`` (the
        same parameter count) or ``"hardware_efficient"`` (a third of it: one
        ``RY`` angle per qubit per layer); see :func:`apply_variational_layers`.
    readout:
        ``"all"`` (default): the layer returns ``(batch, n_qubits)``.
        ``"first"``: ⟨Z_0⟩ only, ``(batch, 1)``, the published SHNN readout.
    noise_level:
        Strength of the ``noise_channel`` applied to the circuit in **train
        mode**: the depolarizing probability in ``[0, 0.75]``, or the damping
        or flip probability in ``[0, 1]`` for the other channels (the table in
        :mod:`hqnn_forge.noise`); ``0`` (default) is the plain noiseless layer.
        With ``noise_level > 0`` the train-mode forward pass runs the circuit
        on ``default.mixed`` with that channel inserted, so gradients are
        computed through the noisy circuit (noise-aware training); eval mode
        is always noiseless, like dropout.  With the
        default ``noise_method``, backprop keeps a ``batch × 4^n`` density
        matrix per operation, so this is practical up to about 6 qubits.  See
        :mod:`hqnn_forge.noise`.
    noise_position:
        ``"all"`` (after every gate, default) or ``"end"`` (before
        measurement), as in :func:`hqnn_forge.noise.apply_depolarizing_noise`.
    noise_method:
        ``"density"`` (default): the exact channel on ``default.mixed``.
        ``"trajectories"``: Pauli-trajectory sampling on this layer's own
        device and ``diff_method``, at pure-state memory; the train-mode
        output is then random, and equal to the ``"density"`` output on
        average.  See :mod:`hqnn_forge.noise`.
    noise_trajectories:
        Draws averaged per sample with ``noise_method="trajectories"``.
        Default 1; must be 1 for ``"density"``.  Use 8 or more at noise of a
        few percent per gate: fewer draws occasionally made training collapse
        on the breast-cancer proxy (#347; see :mod:`hqnn_forge.noise`).
    noise_channel:
        The channel ``noise_level`` is the strength of: ``"depolarizing"``
        (default), ``"amplitude_damping"`` (T1), ``"phase_damping"`` (T2),
        ``"bit_flip"`` (also a symmetric readout error at ``"end"``) or
        ``"phase_flip"``; see :mod:`hqnn_forge.noise`.  With
        ``noise_method="trajectories"``, amplitude damping needs
        ``diff_method="backprop"``, ``"parameter-shift"`` or ``"finite-diff"``,
        ``default.qubit`` or ``lightning.qubit``, and no ``shots``.
    shots:
        ``None`` (default): exact expectation values.  An ``int``: every
        readout is estimated from that many samples, as on hardware.  Needs
        ``diff_method="parameter-shift"``; with training noise, only
        ``noise_method="trajectories"``.  The samples come from the device's
        own generator, which ``torch.manual_seed`` does not reach: pass
        ``seed`` for samples that repeat run to run.  The ``shots`` attribute
        reads the QNode the layer runs, so it follows
        :func:`hqnn_forge.noise.apply_shots`.
    seed:
        Seed of the device's generator, which draws the shot samples; a
        non-negative ``int`` (NumPy integers are converted) or ``None``
        (default: unseeded).  Inert for exact simulation.  Shown in the repr;
        see :func:`~hqnn_forge.encoding.angle_embedding.resolve_device`.

    Attributes
    ----------
    n_qubits : int
    n_features : int
        Width of the input, one feature per qubit: ``n_qubits``.
    n_layers : int
    n_outputs : int
        Width of the output: ``n_qubits`` or 1.
    entangler : str
    readout : str
    noise_level : float
    noise_channel : str
    noise_position : str
    noise_method : str
    noise_trajectories : int
    qlayer : pennylane.qnn.TorchLayer
        The underlying differentiable quantum layer.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.encoding import QuantumEncodingLayer
    >>> from hqnn_forge.initializers import restricted_normal_init_
    >>>
    >>> layer = QuantumEncodingLayer(n_qubits=8, n_layers=2)
    >>> _ = restricted_normal_init_(layer.qlayer.weights, n_qubits=8, n_layers=2)
    >>>
    >>> x = torch.randn(4, 8)   # batch of 4 samples
    >>> out = layer(x)           # shape: (4, 8)
    >>> out.shape
    torch.Size([4, 8])
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation: RotationAxis = "X",
        device_name: DeviceName = "auto",
        diff_method: DiffMethod = "auto",
        entangler: Entangler = "ring",
        readout: Readout = "all",
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

        self.n_qubits = n_qubits
        self.n_features = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        self.readout = readout
        self.n_outputs = len(readout_wires(n_qubits, readout))

        # Build the QNode ─────────────────────────────────────────────────
        qnode = build_encoding_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation=rotation,
            device_name=device_name,
            diff_method=diff_method,
            entangler=entangler,
            readout=readout,
            shots=shots,
            seed=seed,
        )

        # Declare the trainable weight tensor shape for TorchLayer ─────────
        # Shape: variational_weight_shape(entangler, ...)
        #   dim-0: layer index ℓ ∈ {0, …, n_layers-1}
        #   dim-1: qubit  index i ∈ {0, …, n_qubits-1}
        #   dim-2: Euler angles (φ, θ, ω) for qml.Rot; absent for
        #          "hardware_efficient", whose RY takes one angle
        weight_shapes: dict[str, tuple[int, ...]] = {
            "weights": variational_weight_shape(entangler, n_qubits, n_layers),
        }

        # Wrap QNode as an nn.Module with registered Parameters ───────────
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Training-time noise (see hqnn_forge.noise) ──────────────────────
        self._init_training_noise(
            qnode,
            n_qubits,
            noise_level,
            noise_position,
            noise_method,
            noise_trajectories,
            shots=shots,
            noise_channel=noise_channel,
            readout_error=readout_error,
        )
        self.seed = validate_seed(seed)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check that ``x`` has ``n_qubits`` finite features; the angles are used as given.

        This is the classical step ``forward`` applies before the QNode.  Every
        encoding layer has one, so tools which replay the circuit
        (:mod:`hqnn_forge.kernels`) validate and transform inputs exactly as
        ``forward`` does.
        """
        check_inputs(
            x,
            self.n_qubits,
            hint=f"  Apply PCA to reduce to {self.n_qubits} features before passing "
            f"to QuantumEncodingLayer.",
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of feature vectors into Pauli-Z expectation values.

        Parameters
        ----------
        x : torch.Tensor
            Classical input tensor of shape ``(batch_size, n_qubits)``.
            Values should be in ``[-π, π]`` for meaningful angle embedding
            (apply ``torch.tanh(x) * π`` or similar normalisation upstream).

        Returns
        -------
        torch.Tensor
            Quantum expectation values of shape ``(batch_size, n_outputs)``
            (``n_qubits``, or 1 with ``readout="first"``), each ∈ [-1, 1].

        Raises
        ------
        ValueError
            If the last dimension of ``x`` does not equal ``self.n_qubits``.
        """
        x = self.prepare_inputs(x)

        # TorchLayer hands the whole batch to the QNode in one call and reshapes
        # the result to (batch, n_qubits).  Whether the batch is executed as one
        # broadcasted tape or split into one tape per sample is decided in
        # build_encoding_qnode (see expand_batch_dimension); the outputs and
        # gradients are the same either way.
        return self._run_circuit(x)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        options = ""
        if self.entangler != "ring":
            options += f", entangler={self.entangler!r}"
        if self.readout != "all":
            options += f", readout={self.readout!r}"
        options += (
            self._noise_repr() + shots_repr(self.shots, self.seed) + backend_repr(self.qlayer)
        )
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_params={sum(p.numel() for p in self.parameters())}{options}"
        )


# ---------------------------------------------------------------------------
# Re-export convenience alias
# ---------------------------------------------------------------------------
AngleEmbeddingQNode = build_encoding_qnode
"""Alias: ``build_encoding_qnode`` — returns the raw QNode without an nn.Module wrapper."""
