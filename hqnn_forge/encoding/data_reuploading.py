"""
hqnn_forge.encoding.data_reuploading
====================================
Quantum encoding module that re-embeds the classical feature vector before
**every** variational layer instead of once at the start of the circuit
(data re-uploading, Pérez-Salinas et al. 2020).

Design Rationale
----------------
* **Why re-upload.**  A circuit that embeds ``x`` once and then applies a
  data-independent ansatz computes, as a function of each feature, a
  trigonometric polynomial of degree one: the ⟨Z_i⟩ outputs are of the form
  ``a + b·cos(x_j) + c·sin(x_j)`` in every ``x_j``.  Embedding the same
  features ``L`` times raises the accessible degree to ``L`` (Schuld, Sweke
  & Meyer 2021): the frequency spectrum of the model grows with the number
  of uploads, not with the number of qubits.  Re-uploading is therefore the
  way to increase a fixed-qubit-count circuit's expressivity without adding
  qubits.

  This counts every upload only for ``rotation="X"`` or ``"Y"``.  With
  ``rotation="Z"`` the first upload is an ``RZ`` acting on ``|0⟩``, which is a
  global phase, so ``L`` uploads give degree ``L - 1``.  With a single
  ``"Z"`` upload the outputs would not depend on ``x`` at all, so that
  combination is rejected.

* **Structure.**  Per layer ℓ = 0 … L-1:

  1. ``AngleEmbedding(x, rotation)``: one rotation per qubit, angle ``x_i``
     (or ``s_{ℓ,i}·x_i`` with trainable input scaling, see below);
  2. variational block ℓ of the angle encoder's ansatz
     (:func:`~hqnn_forge.encoding.angle_embedding.apply_variational_layers`):
     by default a CNOT ring ``CNOT(i → i+1 mod n)`` then per-qubit
     ``Rot(φ, θ, ω)``.

  Step 2 is exactly one layer of the angle encoder, so with ``n_layers=1``
  this layer is :class:`~hqnn_forge.encoding.QuantumEncodingLayer` gate for
  gate, for every ``entangler``.  The measurement is ⟨Z_i⟩ on the readout
  wires (:func:`~hqnn_forge.encoding.angle_embedding.measure_z`), as in the
  other encoders.

* **Trainable input scaling** (``trainable_input_scaling=True``) multiplies
  the features by a learned ``(n_layers, n_qubits)`` tensor before each
  upload, as in the original proposal, where the classical data enters
  through trainable weights.  This lets the model choose which frequencies
  to use per layer at the cost of ``n_layers·n_qubits`` extra parameters.
  With ``rotation="Z"`` the first upload is a global phase whatever it is
  scaled by, so only the other ``n_layers - 1`` uploads are scaled and the
  tensor is ``(n_layers - 1, n_qubits)``.  Scaling is off by default so
  that the parameter count matches the other encoders.

* **Cost.**  The angle encoder embeds once (``n`` rotations) and then runs
  ``L`` blocks of ``2n`` gates, ``n + 2Ln`` in total; re-uploading embeds
  before each block, ``3Ln`` in total.  The circuit overhead is the ``L - 1``
  extra uploads: ``(L - 1)·n`` single-qubit gates and ``L - 1`` units of
  depth, which is all that ``adjoint`` and ``backprop`` pay for.

  Under ``parameter-shift`` and ``finite-diff`` the gradient cost is
  counted in trainable gate parameters, and every upload rotation whose
  angle needs a gradient is one.  When the inputs need gradients (a
  classical encoder upstream), all ``L·n`` upload angles do; with trainable
  input scaling alone, only the scaled ones, ``L·n`` or ``(L - 1)·n`` for
  ``rotation="Z"``; inputs that need no gradient, without input scaling,
  add nothing.  Per sample, ``parameter-shift`` costs two circuit
  evaluations per such angle, ``2Ln`` against ``2n`` for the angle
  encoder's single upload (``L`` times as many for the embedding), on top
  of ``6Ln`` for ``weights``; ``finite-diff`` (first-order forward
  difference, PennyLane's default) costs one per parameter plus one
  unshifted evaluation, ``Ln + 3Ln + 1`` against ``n + 3Ln + 1``.

References
----------
* Pérez-Salinas et al. (2020) "Data re-uploading for a universal quantum
  classifier", Quantum 4, 226.
* Schuld, Sweke & Meyer (2021) "Effect of data encoding on the expressive
  power of variational quantum-machine-learning models", PRA 103, 032430.
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
    RotationAxis,
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
    validate_seed,
    validate_shots,
    variational_weight_shape,
)
from hqnn_forge.noise import Channel, NoiseMethod, Position, TrainingNoiseMixin

logger = logging.getLogger(__name__)


def _first_scaled_upload(rotation: RotationAxis) -> int:
    """
    Index of the first upload that ``input_scaling`` multiplies.

    The first ``RZ`` upload acts on ``|0⟩`` as a global phase, so a scale on
    it would never reach the outputs and would get zero gradient; for
    ``rotation="Z"`` scaling starts at the second upload.
    """
    return 1 if rotation == "Z" else 0


def input_scaling_shape(n_qubits: int, n_layers: int, rotation: RotationAxis) -> tuple[int, int]:
    """Shape of ``input_scaling``: one row per scaled upload, one column per qubit."""
    return (n_layers - _first_scaled_upload(rotation), n_qubits)


# ---------------------------------------------------------------------------
# Raw QNode function
# ---------------------------------------------------------------------------


def _make_data_reuploading_circuit(
    n_qubits: int,
    n_layers: int,
    rotation: RotationAxis,
    trainable_input_scaling: bool,
    entangler: Entangler = "ring",
    readout: Readout = "all",
) -> Callable[..., list[qml.measurements.ExpectationMP]]:
    """
    Factory returning the bare quantum function for the re-uploading circuit.

    The returned function has the signature::

        circuit(inputs, weights)                 # trainable_input_scaling=False
        circuit(inputs, weights, input_scaling)  # trainable_input_scaling=True

    with ``inputs`` of shape ``(n_qubits,)`` (or ``(batch, n_qubits)`` when
    broadcasted), ``weights`` of shape ``(n_layers, n_qubits, 3)``
    (``(n_layers, n_qubits)`` for ``entangler="hardware_efficient"``) and
    ``input_scaling`` of shape :func:`input_scaling_shape`.

    Circuit structure (per layer ℓ = 0 … L-1)
    ------------------------------------------
    1. ``AngleEmbedding(inputs · input_scaling[ℓ - f], rotation)`` — the
       upload, with ``f = 1`` for ``rotation="Z"`` (whose upload 0 is left
       unscaled) and ``f = 0`` otherwise.
    2. ``apply_variational_layers(weights[ℓ], entangler, layer_offset=ℓ)``,
       so the ``"strongly_entangling"`` ranges follow the whole ansatz.

    Then :func:`~hqnn_forge.encoding.angle_embedding.measure_z` on the
    readout wires.
    """
    first_scaled = _first_scaled_upload(rotation)

    def circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        input_scaling: torch.Tensor | None = None,
    ) -> list[qml.measurements.ExpectationMP]:
        for layer in range(n_layers):
            # ``inputs`` may carry a leading batch axis; scaling is per
            # (upload, qubit) and broadcasts over it.
            if input_scaling is None or layer < first_scaled:
                features = inputs
            else:
                features = inputs * input_scaling[layer - first_scaled]
            qml.AngleEmbedding(features=features, wires=range(n_qubits), rotation=rotation)
            apply_variational_layers(
                weights[layer : layer + 1], n_qubits, 1, entangler, layer_offset=layer
            )
        return measure_z(n_qubits, readout)

    # TorchLayer requires a weight shape for every non-input parameter in the
    # signature, so each mode gets a signature naming exactly its weights.
    if trainable_input_scaling:

        def scaled_circuit(
            inputs: torch.Tensor,
            weights: torch.Tensor,
            input_scaling: torch.Tensor,
        ) -> list[qml.measurements.ExpectationMP]:
            return circuit(inputs, weights, input_scaling)

        return scaled_circuit

    def unscaled_circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[qml.measurements.ExpectationMP]:
        return circuit(inputs, weights)

    return unscaled_circuit


# ---------------------------------------------------------------------------
# Public QNode factory
# ---------------------------------------------------------------------------


def build_data_reuploading_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation: RotationAxis = "X",
    device_name: DeviceName = "auto",
    diff_method: DiffMethod = "auto",
    trainable_input_scaling: bool = False,
    entangler: Entangler = "ring",
    readout: Readout = "all",
    shots: int | None = None,
    seed: int | None = None,
) -> qml.QNode:
    """
    Build and return a PennyLane QNode for the data re-uploading circuit.

    Parameters
    ----------
    n_qubits:
        Number of qubits (= number of input features).
    n_layers:
        Number of uploads, each followed by one entangling + rotation block.
    rotation:
        Pauli axis of the embedding rotations: ``"X"`` (default), ``"Y"``, ``"Z"``.
    device_name, diff_method:
        As for :func:`hqnn_forge.encoding.build_encoding_qnode`.
    trainable_input_scaling:
        If ``True`` the QNode takes a third argument ``input_scaling`` of shape
        :func:`input_scaling_shape` that multiplies the features before each
        scaled upload.
    entangler, readout:
        As for :func:`hqnn_forge.encoding.build_encoding_qnode`.

    Raises
    ------
    ValueError
        If ``n_qubits < 2``, ``n_layers < 1``, ``rotation``, ``entangler`` or
        ``readout`` is not one of the values above, or ``rotation="Z"`` with
        ``n_layers=1`` (a lone ``RZ`` upload on ``|0⟩`` is a global phase, so
        the outputs would not depend on the inputs).  All are checked here,
        before the circuit first runs.
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2 for the CNOT entangling ring; got {n_qubits}.")
    if n_layers < 1:
        raise ValueError(f"n_layers must be ≥ 1 (one upload per layer); got {n_layers}.")
    validate_circuit_options(n_qubits, entangler, readout, rotation)
    if rotation == "Z" and n_layers < 2:
        raise ValueError(
            'rotation="Z" needs n_layers ≥ 2: the first RZ upload acts on |0⟩ as a '
            "global phase, so a single upload leaves the outputs independent of the inputs."
        )

    device_name, diff_method = resolve_backend(device_name, diff_method, n_qubits, shots=shots)
    validate_shots(shots, diff_method)
    device = resolve_device(device_name, n_qubits, seed=seed)
    validate_device_shots(device, shots)
    circuit_fn = _make_data_reuploading_circuit(
        n_qubits, n_layers, rotation, trainable_input_scaling, entangler, readout
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
        "Re-uploading QNode built | device=%s | qubits=%d | layers=%d | diff=%s | "
        "rotation=%s | trainable_input_scaling=%s | entangler=%s | readout=%s",
        device.name,
        n_qubits,
        n_layers,
        diff_method,
        rotation,
        trainable_input_scaling,
        entangler,
        readout,
    )
    return qnode


# ---------------------------------------------------------------------------
# PyTorch nn.Module wrapper
# ---------------------------------------------------------------------------


class DataReuploadingLayer(TrainingNoiseMixin, nn.Module):
    """
    A PyTorch ``nn.Module`` wrapping the data re-uploading QNode.

    API-consistent with :class:`~hqnn_forge.encoding.QuantumEncodingLayer`:
    same constructor arguments (plus ``trainable_input_scaling``), same
    ``qlayer.weights`` shape for each ``entangler``, same
    ``(batch, n_qubits)`` input and ``(batch, n_outputs)`` output.
    The difference is inside the circuit: the features are embedded before
    every variational layer, not only before the first.

    Expressivity versus depth
    -------------------------
    With one upload the model's dependence on each feature is a degree-one
    trigonometric polynomial, whatever ``n_layers`` is; with ``L`` uploads
    it is of degree ``L`` (Schuld, Sweke & Meyer 2021), or ``L - 1`` for
    ``rotation="Z"``, whose first upload is a phase on ``|0⟩``.  The price
    over the single-upload encoder at the same ``n_layers`` is the ``L - 1``
    extra uploads: ``(L - 1)·n_qubits`` single-qubit gates, ``L - 1`` units
    of depth and, on hardware, ``L`` times the data-loading cost.  For
    ``n_layers=1`` without input scaling the two layers are identical.

    Weight Shapes
    -------------
    ``qlayer.weights`` has shape ``(n_layers, n_qubits, 3)``
    (``(n_layers, n_qubits)`` for ``entangler="hardware_efficient"``) and, like the
    other encoders, starts from ``TorchLayer``'s default Uniform(0, 2π).
    **Call** ``hqnn_forge.initializers.restricted_normal_init_`` **on it
    immediately after construction** for the library's small-angle initial
    values (see :mod:`hqnn_forge.initializers` for what they do and do not
    guarantee).
    Leave ``qlayer.input_scaling`` at its ones-initialisation, which starts
    the layer as the plain re-uploading circuit.

    Parameters
    ----------
    n_qubits:
        Number of qubits / input features.  Default: 8.
    n_layers:
        Number of uploads, each followed by one variational block of
        ``entangler``.  Default: 2.
    rotation:
        Pauli axis of the embedding rotations.  Default: ``"X"``.
        ``"Z"`` requires ``n_layers ≥ 2``.
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
    trainable_input_scaling:
        Add a trainable ``qlayer.input_scaling`` of shape
        ``(n_layers, n_qubits)``, initialised to ones, that multiplies the
        features before each upload (Pérez-Salinas et al. 2020); for
        ``rotation="Z"`` it is ``(n_layers - 1, n_qubits)`` and leaves the
        first upload, a global phase, unscaled.  Default: ``False``, so the
        parameter count matches the other encoders.
    entangler:
        ``"ring"`` (default), ``"strongly_entangling"``, ``"brickwork"`` or
        ``"hardware_efficient"``; see
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
        or by trajectories (amplitude damping with restrictions).  See
        :mod:`hqnn_forge.noise`.
    readout_error:
        A train-mode readout error ``(p01, p10)`` on the outputs, exactly as
        for :class:`~hqnn_forge.encoding.QuantumEncodingLayer`: train mode
        only, and with ``shots`` it rescales the sampled estimate.  Default
        ``None``.
    shots, seed:
        Finite-shot sampling and the device seed, exactly as for
        :class:`~hqnn_forge.encoding.QuantumEncodingLayer`.

    Attributes
    ----------
    n_qubits, n_layers : int
    n_features : int
        Width of the input, one feature per qubit: ``n_qubits``.
    n_outputs : int
        Width of the output: ``n_qubits`` or 1.
    rotation : str
    trainable_input_scaling : bool
    entangler, readout : str
    qlayer : pennylane.qnn.TorchLayer
        Owns ``weights`` and, if enabled, ``input_scaling``.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.encoding import DataReuploadingLayer
    >>> from hqnn_forge.initializers import restricted_normal_init_
    >>> layer = DataReuploadingLayer(n_qubits=4, n_layers=3)
    >>> _ = restricted_normal_init_(layer.qlayer.weights, n_qubits=4, n_layers=3)
    >>> layer(torch.rand(2, 4)).shape
    torch.Size([2, 4])
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation: RotationAxis = "X",
        device_name: DeviceName = "auto",
        diff_method: DiffMethod = "auto",
        trainable_input_scaling: bool = False,
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
        self.rotation = rotation
        self.trainable_input_scaling = trainable_input_scaling
        self.entangler = entangler
        self.readout = readout
        self.n_outputs = len(readout_wires(n_qubits, readout))

        qnode = build_data_reuploading_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation=rotation,
            device_name=device_name,
            diff_method=diff_method,
            trainable_input_scaling=trainable_input_scaling,
            entangler=entangler,
            readout=readout,
            shots=shots,
            seed=seed,
        )

        weight_shapes: dict[str, tuple[int, ...]] = {
            "weights": variational_weight_shape(entangler, n_qubits, n_layers),
        }
        if trainable_input_scaling:
            weight_shapes["input_scaling"] = input_scaling_shape(n_qubits, n_layers, rotation)

        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        if trainable_input_scaling:
            # Start as the plain re-uploading circuit: every upload sees x.
            with torch.no_grad():
                self.qlayer.input_scaling.fill_(1.0)
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
    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check that ``x`` has ``n_qubits`` finite features; the angles are used as given.

        The optional ``input_scaling`` is a circuit parameter, applied inside
        the QNode, not here.

        This is the classical step ``forward`` applies before the QNode.  Every
        encoding layer has one, so tools which replay the circuit
        (:mod:`hqnn_forge.kernels`) validate and transform inputs exactly as
        ``forward`` does.
        """
        check_inputs(x, self.n_qubits)
        return x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of feature vectors, re-uploading before every layer.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_qubits)``, values in ``[-π, π]`` as for
            the angle encoder.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, n_outputs)``, each element ∈ [-1, 1].

        Raises
        ------
        ValueError
            If the last dimension of ``x`` is not ``n_qubits``.
        """
        # Whole batch in one call; see QuantumEncodingLayer.forward.
        return self._run_circuit(self.prepare_inputs(x))

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        # rotation is always shown: it sets the input_scaling shape, so two
        # layers that differ only in it can differ in n_params.
        options = f", rotation={self.rotation!r}"
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
            f"trainable_input_scaling={self.trainable_input_scaling}, "
            f"n_params={n_params}{options}"
        )
