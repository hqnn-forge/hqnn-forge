"""
hqnn_forge.encoding.amplitude_embedding
=======================================
Quantum encoding module projecting classical tabular feature vectors into an
n-qubit Hilbert space via amplitude embedding, followed by the same
strongly-entangled variational ansatz the angle and IQP encoders use.

Design Rationale
----------------
* **Amplitude Embedding** writes the (zero-padded, L2-normalised) feature
  vector directly into the probability amplitudes of the state:
  ``|ψ(x)⟩ = Σ_k x_k |k⟩ / ‖x‖``.  ``n_qubits`` qubits therefore take up to
  ``2**n_qubits`` features, where angle and IQP embedding take ``n_qubits``.
  It is the qubit-efficient choice: 8 qubits carry 256 features instead of 8.

* **State-preparation cost** is the price of that efficiency.  An arbitrary
  amplitude vector needs a state-preparation circuit that is exponential in
  ``n_qubits`` on real hardware (O(2^n) gates; Möttönen et al. 2005), and on
  a state-vector simulator the embedding is one O(2^n) vector write instead
  of ``n`` rotations.  Angle and IQP embedding cost O(n) and O(n²) gates
  respectively.  Amplitude embedding is the right tool when the number of
  features, not the gate budget, is the constraint.

* **Normalisation removes one degree of freedom.**  Because only the
  direction of ``x`` survives, ``x`` and ``c·x`` encode the same state for any
  ``c > 0``, and an all-zero vector has no valid state at all (``forward``
  raises for it, and for NaN or infinite features).  Feature scaling
  upstream matters less than for angle embedding, where the absolute value
  of each feature is a rotation angle.

* **Differentiation.**  Under ``backprop`` on ``default.qubit`` gradients
  flow to both the weights and the inputs through the state vector itself.
  The ``adjoint``, ``parameter-shift`` and ``finite-diff`` methods instead
  differentiate the Möttönen rotation-gate decomposition of the state
  preparation, whose angles are ``arcsin`` of amplitude ratios.  That
  derivative is infinite when an amplitude is zero and ill-conditioned when
  it is merely small next to its partner.  How PennyLane fails there
  depends on its version, but it never raises.  Measured on 0.45.1 and on
  the 0.46.0.dev114 nightly (0.46 itself is not released yet), on the
  three-qubit circuit the tests pin, where amplitudes ``2k`` and ``2k + 1``
  form a pair:

  - On 0.45 the input gradient is **NaN** in every component as soon as one
    amplitude is exactly zero, which zero padding makes true of every
    sample.  It is also NaN when the first amplitude of a pair is merely
    small next to the second: below about 1e-4 of it in float32, and in
    float64 somewhere between 1e-6 and 1e-8 of it.  Short of that edge it is
    **finite but wrong**: in float32 off by 1e-4 at 1e-2 of the second and
    by 0.2 at 3e-4, in float64 by 5e-5 at 1e-6.  A small second amplitude
    does no harm.
  - On the 0.46 pre-releases one small amplitude next to a larger partner
    differentiates correctly, and what an exactly-zero amplitude does
    depends on its partner.  If the partner is zero too, as with two or more
    padded amplitudes, the gradient is still **NaN** in every component.  If
    it is not, the gradient is **finite but wrong**: the zero amplitude's
    own component comes back as 0, which it is not, and the others are
    right.  A single padded amplitude is therefore harmless there, its
    component being discarded, but a feature that is exactly zero, such as
    a ReLU output of a classical encoder, silently gets no gradient, which
    is worse than a NaN.
  - On both, a pair that is small as a whole breaks the gradient without
    any exact zero, if it is the first pair of its group of four (amplitudes
    ``4k`` and ``4k + 1``) and small next to the second.  Measured on
    amplitudes 4 and 5 under ``parameter-shift``: in float32 the gradient is
    **finite but wrong** (off by 5e-2) when the pair is about 1e-4 of the
    others and **NaN** at 1e-5; in float64 it is off by 9e-5 at 1e-7 and
    NaN at 1e-10.  A small second pair leaves ``parameter-shift`` right.
  - On both, ``parameter-shift``, ``finite-diff`` and lightning's
    ``adjoint`` agree with each other on single small and zero amplitudes.
    Around a small pair, first or second and in either dtype, the latter
    two lose accuracy where ``parameter-shift`` is still right: with the
    pair at 1e-5 of the others they are off by between 1e-5 and 2e-3.
    ``adjoint`` on ``default.qubit`` returns **zero** for every input they
    handle and NaN wherever they return NaN.

  Which inputs are affected depends on the data, so no per-batch check can
  catch them reliably.

  The circuit therefore raises ``RuntimeError`` whenever the inputs require
  a gradient under any method other than ``backprop``, decided from the
  method alone.  The weight gradients are correct under every method: the
  state-preparation angles do not depend on the weights.

* **Strongly-Entangling Ansatz** and **Measurement** are identical to
  :mod:`hqnn_forge.encoding.angle_embedding`: L layers of a CNOT ring
  followed by per-qubit ``Rot``, then ⟨Z_i⟩ on every qubit.

References
----------
* Möttönen et al. (2005) "Transformation of quantum states using uniformly
  controlled rotations", Quantum Inf. Comput. 5, 467.
* Schuld & Petruccione (2018) "Supervised Learning with Quantum Computers",
  Springer, §5.2 (amplitude encoding).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pennylane as qml
import torch
import torch.nn as nn

from hqnn_forge.encoding._common import (
    DeviceName,
    DiffMethod,
    Entangler,
    Readout,
    apply_variational_layers,
    backend_repr,
    check_inputs,
    expand_batch_dimension,
    measure_z,
    readout_wires,
    resolve_backend,
    resolve_device,
    shots_repr,
    validate_circuit_options,
    validate_device_shots,
    validate_shots,
    variational_weight_shape,
)
from hqnn_forge.noise import Channel, NoiseMethod, Position, TrainingNoiseMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw QNode function
# ---------------------------------------------------------------------------


def _check_input_gradient(inputs: torch.Tensor, diff_method: str) -> None:
    """
    Raise if a gradient with respect to ``inputs`` would be computed under a
    method other than ``backprop``.

    Only relevant when ``inputs`` requires grad and autograd is recording.
    The rule depends on the method alone, not on the values in ``inputs``:
    the non-backprop input gradient is NaN or silently wrong for zero and
    small amplitudes, and which batches contain those cannot be told in
    advance.  See the module docstring, *Differentiation*.
    """
    if diff_method == "backprop" or not isinstance(inputs, torch.Tensor):
        return
    if not (inputs.requires_grad and torch.is_grad_enabled()):
        return
    raise RuntimeError(
        f"Amplitude embedding cannot differentiate with respect to its inputs under "
        f"diff_method={diff_method!r}: that method differentiates the state-preparation "
        f"decomposition, whose input gradient is NaN or silently wrong whenever an "
        f"amplitude is zero or small.  Use diff_method='backprop' on default.qubit, or "
        f"detach the inputs (weight gradients are unaffected)."
    )


def _make_amplitude_embedding_circuit(
    n_qubits: int,
    n_layers: int,
    diff_method: str,
    entangler: Entangler = "ring",
    readout: Readout = "all",
) -> Callable[[torch.Tensor, torch.Tensor], list[qml.measurements.ExpectationMP]]:
    """
    Factory returning the bare quantum function for the amplitude feature map.

    The returned function has the signature::

        circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[ExpectationMP]

    where ``inputs`` has shape ``(2**n_qubits,)`` (or ``(batch, 2**n_qubits)``
    when broadcasted) and is already zero-padded and L2-normalised, and
    ``weights`` has shape ``(n_layers, n_qubits, 3)``.  ``diff_method`` is
    used only by the input-gradient check.

    Called inside a QNode it records one ``qml.expval(PauliZ)`` measurement on
    every wire; the QNode turns them into the expectation values.

    Circuit structure
    -----------------
    0. :func:`_check_input_gradient` refuses input gradients under every
       method other than ``backprop``.
    1. ``AmplitudeEmbedding(inputs)``: prepares ``Σ_k inputs_k |k⟩``.  The
       template does not re-normalise: ``inputs`` must already have unit norm,
       which the template checks.
    2. :func:`~hqnn_forge.encoding.angle_embedding.apply_variational_layers`
       with the ``"ring"`` block: per layer a CNOT ring ``CNOT(i → i+1 mod n)``,
       then ``Rot`` on each qubit.
    3. ``[⟨Z_i⟩ for i in range(n_qubits)]``.
    """

    def circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[qml.measurements.ExpectationMP]:
        # ── 0. Refuse input gradients PennyLane would get wrong ──────────
        _check_input_gradient(inputs, diff_method)

        # ── 1. Amplitude embedding ───────────────────────────────────────
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits))

        # ── 2 & 3. Variational layers, then ⟨Z⟩ on every wire ──────────────
        apply_variational_layers(weights, n_qubits, n_layers, entangler)
        return measure_z(n_qubits, readout)

    return circuit


# ---------------------------------------------------------------------------
# Public QNode factory
# ---------------------------------------------------------------------------


def build_amplitude_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    device_name: DeviceName = "auto",
    diff_method: DiffMethod = "auto",
    entangler: Entangler = "ring",
    readout: Readout = "all",
    shots: int | None = None,
    seed: int | None = None,
) -> qml.QNode:
    """
    Build and return a PennyLane QNode for the amplitude feature map.

    The QNode expects unit-norm ``inputs`` of shape ``(2**n_qubits,)`` or
    ``(batch, 2**n_qubits)``; use :class:`AmplitudeEncodingLayer` for the
    padding and normalisation of raw feature vectors.

    When ``inputs`` requires a gradient under a method other than
    ``"backprop"``, calling the QNode raises ``RuntimeError``: PennyLane would
    return NaN, zero or a silently wrong value for that gradient on some
    inputs.  See the module docstring.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  The state has ``2**n_qubits`` amplitudes.
    n_layers:
        Number of entangling + rotation layers in the VQC ansatz.
    device_name, diff_method, entangler, readout:
        As for :func:`hqnn_forge.encoding.build_encoding_qnode`.

    Raises
    ------
    ValueError
        If ``n_qubits < 2``, or ``entangler`` or ``readout`` is unknown.
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2 for the CNOT entangling ring; got {n_qubits}.")
    validate_circuit_options(n_qubits, entangler, readout)

    device_name, diff_method = resolve_backend(device_name, diff_method, n_qubits, shots=shots)
    validate_shots(shots, diff_method)
    device = resolve_device(device_name, n_qubits, seed=seed)
    validate_device_shots(device, shots)
    circuit_fn = _make_amplitude_embedding_circuit(
        n_qubits, n_layers, diff_method, entangler, readout
    )

    qnode = qml.QNode(
        func=circuit_fn,
        device=device,
        diff_method=diff_method,
        interface="torch",
        shots=shots,
    )
    qnode = expand_batch_dimension(qnode, diff_method)

    logger.info(
        "Amplitude QNode built | device=%s | qubits=%d | layers=%d | diff=%s",
        device.name,
        n_qubits,
        n_layers,
        diff_method,
    )
    return qnode


# ---------------------------------------------------------------------------
# PyTorch nn.Module wrapper
# ---------------------------------------------------------------------------


class AmplitudeEncodingLayer(TrainingNoiseMixin, nn.Module):
    """
    A PyTorch ``nn.Module`` wrapping the amplitude-embedding QNode.

    API-consistent with :class:`~hqnn_forge.encoding.QuantumEncodingLayer` and
    :class:`~hqnn_forge.encoding.iqp_embedding.IQPEncodingLayer`: same
    constructor shape, same ``qlayer.weights`` parameter of shape
    ``(n_layers, n_qubits, 3)``, same ``(batch, n_qubits)`` output of ⟨Z_i⟩.
    The difference is the input width: up to ``2**n_qubits`` features per
    sample instead of exactly ``n_qubits``.

    Input handling in ``forward``
    -----------------------------
    1. The last dimension must equal ``n_features``.
    2. If ``n_features < 2**n_qubits`` the vector is zero-padded on the right
       to ``2**n_qubits`` amplitudes.
    3. The padded vector is L2-normalised, after dividing by its largest
       absolute feature so the norm cannot overflow or underflow.  A sample
       containing NaN or ±inf, or one that is exactly all zeros, raises
       ``ValueError``: the zero vector has no direction, so there is no
       state to prepare.  Any non-zero scale, however small, is accepted.

    Both steps are differentiable, so the gradient can reach whatever
    produced the features (see *Differentiation methods* for which methods
    support that).

    Cost versus the other encoders
    ------------------------------
    Angle embedding uses ``n`` single-qubit rotations for ``n`` features; IQP
    uses O(n²) gates for ``n`` features; amplitude embedding takes up to
    ``2**n`` features but needs an O(2^n)-gate state-preparation circuit on
    hardware (Möttönen et al. 2005).  On a simulator the difference is one
    vector write against ``n`` rotations, so the wall-clock gap is small; on
    hardware it is exponential.  Choose amplitude embedding when the feature
    count is the constraint, not when the gate budget is.

    Differentiation methods
    -----------------------
    Weight gradients are correct under every method.  The input gradient is
    only supported under ``"backprop"`` on ``default.qubit``; every other
    method raises ``RuntimeError`` when ``x`` requires a gradient, because
    PennyLane returns NaN or a silently wrong value for zero and small
    amplitudes (see the module docstring).

    The check only applies when a gradient will actually be computed:
    detached inputs, and any input under ``torch.no_grad()``, are fine with
    every method.  With a classical encoder upstream, use ``backprop``.  The
    default ``diff_method="auto"`` is ``backprop`` on ``default.qubit`` up to
    12 qubits, where input gradients work; above that it is ``adjoint`` on
    lightning, and input gradients are refused.  The hybrid classifiers pick
    ``backprop`` whatever the size when their encoder feeds this layer.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  Default: 8 (256 amplitudes).
    n_layers:
        Entangling + rotation blocks in the VQC ansatz.  Default: 2.
    n_features:
        Width of the input vectors, ``1 ≤ n_features ≤ 2**n_qubits``.
        Default: ``2**n_qubits`` (no padding).
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
        Gradient method, ``"auto"`` by default.  See *Differentiation
        methods* above.
    entangler:
        The variational block: ``"ring"`` (default), ``"strongly_entangling"``,
        ``"brickwork"`` or ``"hardware_efficient"``; see
        :func:`~hqnn_forge.encoding.angle_embedding.apply_variational_layers`.
    readout:
        ``"all"`` (default): the layer returns ``(batch, n_qubits)``.
        ``"first"``: ⟨Z_0⟩ only, ``(batch, 1)``.
    noise_level, noise_position, noise_method, noise_trajectories, noise_channel:
        Training-time noise, exactly as for
        :class:`~hqnn_forge.encoding.QuantumEncodingLayer`: ``noise_level``
        is the strength of ``noise_channel`` (default ``"depolarizing"``), in
        ``[0, 0.75]`` for depolarizing and ``[0, 1]`` for the damping and flip
        channels (default 0, noiseless), applied in train mode only, at
        ``"all"`` gates or at the ``"end"``, simulated exactly (``"density"``)
        or by Pauli trajectories (the Pauli channels only).  See
        :mod:`hqnn_forge.noise`.
    shots:
        Finite-shot sampling, exactly as for
        :class:`~hqnn_forge.encoding.QuantumEncodingLayer`.

    Attributes
    ----------
    n_qubits : int
    n_layers : int
    n_features : int
        Width of the input, before padding to ``n_amplitudes``.
    n_amplitudes : int
        ``2**n_qubits``.
    qlayer : pennylane.qnn.TorchLayer

    Methods
    -------
    prepare_inputs(x)
        The padding and normalisation ``forward`` applies before the QNode.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.encoding import AmplitudeEncodingLayer
    >>> layer = AmplitudeEncodingLayer(
    ...     n_qubits=3, n_layers=1, n_features=6,
    ...     device_name="default.qubit", diff_method="backprop",
    ... )
    >>> x = torch.randn(4, 6)   # 6 features → padded to 8 amplitudes
    >>> layer(x).shape
    torch.Size([4, 3])
    """

    #: Re-applied by :func:`hqnn_forge.noise.apply_shots`, whose
    #: parameter-shift QNode replays this circuit function: the check built
    #: into it holds the construction-time ``diff_method``, which under
    #: ``backprop`` would let parameter-shift differentiate the inputs.
    _input_gradient_check = staticmethod(_check_input_gradient)

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_features: int | None = None,
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
    ) -> None:
        super().__init__()

        # Validated here, before 2**n_qubits is used to check n_features.
        if n_qubits < 2:
            raise ValueError(f"n_qubits must be ≥ 2 for the CNOT entangling ring; got {n_qubits}.")
        n_amplitudes = 2**n_qubits
        if n_features is None:
            n_features = n_amplitudes
        if not 1 <= n_features <= n_amplitudes:
            raise ValueError(
                f"n_features must lie in [1, 2**n_qubits] = [1, {n_amplitudes}]; got {n_features}."
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_amplitudes = n_amplitudes
        self.entangler = entangler
        self.readout = readout
        self.n_outputs = len(readout_wires(n_qubits, readout))

        qnode = build_amplitude_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name,
            diff_method=diff_method,
            entangler=entangler,
            readout=readout,
            shots=shots,
            seed=seed,
        )

        weight_shapes: dict[str, tuple[int, ...]] = {
            "weights": variational_weight_shape(entangler, n_qubits, n_layers),
        }
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self._init_training_noise(
            qnode,
            n_qubits,
            noise_level,
            noise_position,
            noise_method,
            noise_trajectories,
            shots=shots,
            noise_channel=noise_channel,
        )
        self.seed = seed

    # ------------------------------------------------------------------
    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zero-pad ``x`` to ``2**n_qubits`` and L2-normalise each sample.

        This is the classical step ``forward`` applies before the QNode.  Every
        encoding layer has one, so tools which replay the circuit
        (:mod:`hqnn_forge.kernels`) validate and transform inputs exactly as
        ``forward`` does.
        """
        check_inputs(
            x,
            self.n_features,
            name="n_features",
            hint=f"  At most 2**n_qubits={self.n_amplitudes} features fit in "
            f"{self.n_qubits} qubits.",
        )
        # Dividing by the largest |x_k| first keeps the norm in [1, √n]: a raw
        # float32 norm overflows to inf from about 1.8e19 and underflows to 0
        # below about 1e-19.  After the division the largest entry is ±1, so
        # any non-zero finite scale is safe and only an exact zero is refused.
        # x/‖x‖ does not depend on the scale, so neither does its gradient,
        # and the scale can be detached.
        scale = x.detach().abs().amax(dim=-1, keepdim=True)
        if not bool((scale > 0).all()):
            raise ValueError(
                "Amplitude embedding cannot encode an all-zero feature vector: "
                "it has no direction, so there is no state to prepare."
            )
        x = x / scale
        pad = self.n_amplitudes - self.n_features
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of feature vectors into Pauli-Z expectation values.

        Parameters
        ----------
        x : torch.Tensor
            Classical input tensor of shape ``(batch_size, n_features)``.
            Any real scale; only the direction is encoded.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, n_outputs)`` (``n_qubits``, or 1 with
            ``readout="first"``), each element ∈ [-1, 1].

        Raises
        ------
        ValueError
            If the last dimension of ``x`` is not ``n_features``, or a sample
            is all zeros or contains NaN or ±inf.
        RuntimeError
            If ``x`` requires a gradient and ``diff_method`` is not
            ``"backprop"`` (see *Differentiation methods*).
        """
        amplitudes = self.prepare_inputs(x)
        # Whole batch in one call; see QuantumEncodingLayer.forward.
        return self._run_circuit(amplitudes)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        options = ""
        if self.entangler != "ring":
            options += f", entangler={self.entangler!r}"
        if self.readout != "all":
            options += f", readout={self.readout!r}"
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_features={self.n_features}, "
            f"n_params={sum(p.numel() for p in self.parameters())}{options}"
            f"{self._noise_repr()}{shots_repr(self.shots, self.seed)}{backend_repr(self.qlayer)}"
        )
