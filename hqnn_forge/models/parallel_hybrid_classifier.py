"""
hqnn_forge.models.parallel_hybrid_classifier
==============================================
Parallel-topology Hybrid Quantum-Classical Binary Classifier.

Architecture
------------
::

    Input (batch, n_input_features)
         │
         ├─────────────────────────────┬──────────────────────────────┐
         ▼                             ▼
    [Classical branch]            [Quantum branch]
    Linear → ReLU →               [Classical encoder] Linear(→n_qubits) + Tanh
    Linear → ReLU                      │
    → (batch, classical_hidden_dim)    ▼
                                   QuantumEncodingLayer(n_qubits, n_layers)
                                        │
                                        ▼
                                   ⟨Z_i⟩, (batch, n_qubits)
         │                             │
         └──────────────┬──────────────┘
                         ▼
                   [Concatenate]  (batch, classical_hidden_dim + n_qubits)
                         │
                         ▼
                   [Dropout]
                         │
                         ▼
                   [Classical head]  Linear(→ 1)
                         │
                         ▼
                   Raw logit (batch, 1)   ← apply sigmoid for probability

Design Notes
------------
* The two branches process the *same* raw input independently and are fused
  by concatenation before the final classification head.  This is the
  standard architecture for testing whether added classical capacity can
  substitute for, or extend, what the quantum layer contributes — compare
  ``ParallelHybridClassifier.count_parameters()`` against an equivalently
  configured ``HybridBinaryClassifier`` to quantify the trade-off.

* The quantum branch mirrors ``HybridBinaryClassifier`` in *topology* (same
  classical encoder + ``QuantumEncodingLayer`` / ``IQPEncodingLayer`` choice,
  same small-angle initialisation scheme).  Note that seeding the two
  architectures identically does **not** give them identical quantum weights:
  this model builds more classical layers before the quantum init runs, so it
  draws from a different RNG state.  To compare the two topologies fairly,
  copy the quantum weights across after construction — see
  ``examples/quick_start.py``.

* The classical branch is a small two-layer MLP (``classical_hidden_dim``
  units) with ReLU activations.

Parameters
----------
n_input_features:
    Dimensionality of the raw / PCA-reduced input.
n_qubits:
    Number of qubits in the quantum branch.
n_layers:
    Number of variational layers in the quantum circuit.
classical_hidden_dim:
    Width of the classical MLP branch.
use_classical_encoder:
    If ``True`` (default), prepend a ``Linear + Tanh`` to project input to
    ``n_qubits`` dims for the quantum branch.  If ``False``, input must already
    lie in (-π, π) (e.g. ``PCANormalizer(scale_to_pi=True)``); the quantum
    branch then passes it to the circuit unscaled.
device_name:
    PennyLane device name.  Default ``"auto"``: ``default.qubit`` up to
    12 qubits, ``lightning.qubit`` above (see
    :func:`~hqnn_forge.encoding.resolve_backend`).  The four simulators
    fall back along ``lightning.qubit → default.qubit`` with a warning
    when a backend cannot be initialised; any other name (a plugin or
    hardware) is used as given.
diff_method:
    ``"auto"`` (default) picks by device: ``"backprop"`` on
    ``default.qubit``, ``"adjoint"`` on lightning, ``"parameter-shift"``
    with ``shots`` or on any other device.  Or one of ``"adjoint"``,
    ``"parameter-shift"``, ``"backprop"``, ``"finite-diff"``.
init_strategy:
    ``"restricted"`` (default) — global restricted-normal init.
    ``"block_local"``           — per-layer decreasing variance.
encoding_type:
    ``"angle"`` (default), ``"iqp"``, ``"reuploading"`` or ``"amplitude"``; see
    the class docstring.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import (
    DeviceName,
    DiffMethod,
    Entangler,
    Readout,
    RotationAxis,
)
from hqnn_forge.models._trunk import (
    DEFAULT_ENCODER_ACTIVATION,
    DEFAULT_INIT_STD,
    QuantumTrunk,
    validate_encoder_activation,
    validate_init,
)
from hqnn_forge.models.base import BinaryClassifierBase
from hqnn_forge.models.hybrid_classifier import _PUBLISHED_SHNN
from hqnn_forge.noise import Channel, NoiseMethod, Position, validate_readout_error
from hqnn_forge.utils.rng import as_seed, seeded_rng


class ParallelHybridClassifier(QuantumTrunk, BinaryClassifierBase):
    """
    Parallel-topology hybrid quantum-classical binary classifier.

    See module docstring for architecture overview.  API-consistent with
    ``HybridBinaryClassifier`` — same ``encoding_type`` / ``init_strategy``
    options and ``predict_proba`` / ``predict`` / ``count_parameters``
    interface.

    The published SHNN (thesis / ``hqnn-fraud-detection-benchmark``) is
    ``embedding_rotation="Y"``, ``entangler="strongly_entangling"``,
    ``readout="first"``, ``encoder_activation="sigmoid"``,
    ``init_strategy="normal"``; see :meth:`published_shnn`.

    Parameters
    ----------
    n_input_features:
        Number of raw (or PCA-reduced) input features.  Default: 8.
    n_qubits:
        Number of qubits in the quantum branch.  Default: 8.
    n_layers:
        VQC ansatz layers.  Default: 2.  At 1, with angle encoding and
        ``readout="first"``, ⟨Z_0⟩ misses the first encoded angle under the
        default ring and RX embedding, and most of them under
        ``entangler="brickwork"``; under ``"hardware_efficient"`` it sees at most
        ``n_layers + 1`` of them at any depth.  Use 2 or more with a ring.  See step 2 of
        :func:`hqnn_forge.encoding.angle_embedding._make_angle_embedding_circuit`.
    classical_hidden_dim:
        Width of the classical MLP branch.  Default: 16.
    use_classical_encoder:
        Prepend ``Linear(n_input_features → n_qubits) + Tanh`` to the quantum
        branch.  Default: True.  If ``False``, input must already lie in
        (-π, π); it is not rescaled.
    dropout_p:
        Dropout probability applied to the fused branch outputs.  Default: 0.0.
    device_name:
        PennyLane device name.  Default ``"auto"``: ``default.qubit`` up to
        12 qubits, ``lightning.qubit`` above (see
        :func:`~hqnn_forge.encoding.resolve_backend`).  The four simulators
        fall back along ``lightning.qubit → default.qubit`` with a warning
        when a backend cannot be initialised; any other name (a plugin or
        hardware) is used as given.
    diff_method:
        ``"auto"`` (default) picks by device: ``"backprop"`` on
        ``default.qubit``, ``"adjoint"`` on lightning, ``"parameter-shift"``
        with ``shots`` or on any other device.  Or one of ``"adjoint"``,
        ``"parameter-shift"``, ``"backprop"``, ``"finite-diff"``.
    init_strategy:
        ``"restricted"`` (default), ``"block_local"``, or ``"normal"``
        (``N(0, init_std²)``, the published SHNN's init).
    encoding_type:
        The quantum embedding.  Default: ``"angle"``.

        * ``"angle"``: one rotation per feature
          (:class:`~hqnn_forge.encoding.QuantumEncodingLayer`).
        * ``"iqp"``: Hadamards, ``RZ(x_i)`` and pairwise ``x_i x_j`` phases
          (:class:`~hqnn_forge.encoding.iqp_embedding.IQPEncodingLayer`).
        * ``"reuploading"``: the angle embedding repeated before every
          variational layer (:class:`~hqnn_forge.encoding.DataReuploadingLayer`),
          optionally with ``trainable_input_scaling``.
        * ``"amplitude"``: the features as the ``2**n_qubits`` amplitudes of
          the state (:class:`~hqnn_forge.encoding.AmplitudeEncodingLayer`).  The
          classical encoder then maps to ``2**n_qubits`` features, and the
          ``·π`` scaling is irrelevant because the layer normalises.  Its
          input gradient is only correct under backprop, so with a classical
          encoder ``"auto"`` picks ``default.qubit``/backprop at any size, and
          an explicit choice that resolves to another method raises.  Without
          one, 1 to ``2**n_qubits`` raw features are zero-padded.
    embedding_rotation:
        Pauli axis of the angle embedding, ``"X"`` (default), ``"Y"`` or ``"Z"``.
        Angle and re-uploading encodings only.  ``"Z"`` raises under angle
        encoding, since a single ``RZ`` embedding on ``|0⟩`` ignores the input;
        under re-uploading it needs ``n_layers ≥ 2``.
    entangler:
        ``"ring"`` (default: CNOT ring then ``Rot``), ``"strongly_entangling"``
        (``qml.StronglyEntanglingLayers``: ``Rot`` then a CNOT ring of growing
        range), ``"brickwork"`` (nearest-neighbour CNOT pairs, so each ⟨Z_i⟩
        keeps a local light cone at shallow depth) or ``"hardware_efficient"``
        (a CZ ladder then ``RY``: a third of the circuit parameters).  See
        :func:`hqnn_forge.encoding.angle_embedding.apply_variational_layers`.
    readout:
        ``"all"`` (default): the head reads every ⟨Z_i⟩.  ``"first"``: ⟨Z_0⟩
        only, so the head is ``Linear(1 → 1)``.
    encoder_activation:
        ``"tanh"`` (default): encoder output ``tanh(·)·π`` in (-π, π).
        ``"sigmoid"``: ``π·sigmoid(·)`` in (0, π).  Requires
        ``use_classical_encoder=True``; there is no activation without an
        encoder, so a non-default value raises rather than being ignored.
    init_std:
        Standard deviation for ``init_strategy="normal"``.  Default: 0.1.
        Raises under the other strategies, which derive their own sigma.

    noise_level:
        Training-time strength of ``noise_channel`` for the quantum layer: the
        depolarizing probability in ``[0, 0.75]``, or the damping or flip
        probability in ``[0, 1]`` for the other channels.  Default: ``0.0``
        (noiseless).  Applied in train mode
        only.  With the default ``noise_method`` it runs on ``default.mixed``
        with backprop, whose memory grows as ``batch × 4^n_qubits`` per
        operation: practical up to about 6 qubits.  See :mod:`hqnn_forge.noise`.
    noise_position:
        ``"all"`` (default) or ``"end"``; where the channel is inserted.
    noise_method:
        ``"density"`` (default, exact) or ``"trajectories"`` (Pauli-trajectory
        sampling on the layer's own device, at pure-state memory; equal to
        ``"density"`` on average).  See
        :class:`~hqnn_forge.encoding.QuantumEncodingLayer`.
    noise_trajectories:
        Draws averaged per sample with ``"trajectories"``.  Default: 1.  Use 8
        or more at noise of a few percent per gate: with fewer draws some runs
        on the breast-cancer proxy had not started to train within 30 epochs
        (#347, #480; see :mod:`hqnn_forge.noise`).
    init_seed:
        Seed for weight initialisation.  ``None`` (default) draws the initial
        weights from the global torch RNG; an int draws them from a private RNG
        seeded with it, so the same seed gives the same weights and the global
        RNG is left exactly as it was.
    classical_encoder:
        Your own module in place of the built-in ``Linear(n_input_features →
        n_qubits)``, trained together with the quantum layer: a small MLP,
        or a CNN or sequence model that reshapes the flat
        ``(batch, n_input_features)`` input itself.  It must return
        ``(batch, n_qubits)`` (``(batch, 2**n_qubits)`` with
        ``encoding_type="amplitude"``), which is checked here with one forward pass.
        The model owns the angle range: it applies ``encoder_activation`` and
        the factor π on top of the module, exactly as for the built-in
        encoder, so the module should output unbounded features and not end
        in ``Tanh`` or ``Sigmoid`` (that warns).  The module is used as given
        and never re-initialised, so pretrained weights are kept.  Requires
        ``use_classical_encoder=True``.  ``save_checkpoint`` refuses a model
        with a custom encoder; save its ``state_dict`` instead.  Default:
        ``None``, the built-in encoder.
    trainable_input_scaling:
        With ``encoding_type="reuploading"`` only: a trainable per-upload
        scale on the features, initialised to 1.  Default: ``False``.
    shots:
        ``None`` (default): exact expectation values.  An ``int``: each circuit
        is sampled that many times, as on hardware, so predictions carry shot
        noise.  ``diff_method="auto"`` then picks ``"parameter-shift"``, the
        only method that works: ``adjoint`` and ``backprop`` need the exact
        state, and ``finite-diff``'s tiny step turns the shot noise into
        gradients of order 1e6.  The samples come from the device's own
        generator, which ``torch.manual_seed`` does not reach: pass ``seed``
        for a model with shots that repeats run to run.
        :func:`hqnn_forge.noise.apply_shots` evaluates a model with a finite
        shot count without rebuilding it.
    noise_channel:
        The channel ``noise_level`` is the strength of: ``"depolarizing"``
        (default), ``"amplitude_damping"``, ``"phase_damping"``,
        ``"bit_flip"`` or ``"phase_flip"``; see :mod:`hqnn_forge.noise`.  With
        ``noise_method="trajectories"``, amplitude damping needs
        ``diff_method="backprop"``, ``"parameter-shift"`` or ``"finite-diff"``,
        ``default.qubit`` or ``lightning.qubit``, and no ``shots``.
    seed:
        Seed of the device's random generator, which draws the shot samples,
        so a shot-based model gives the same samples on every run.  Default
        ``None``: unseeded, and ``torch.manual_seed`` does not reach it.  See
        :func:`hqnn_forge.encoding._common.resolve_device`.
    readout_error:
        ``(p01, p10)``: in train mode, each measured bit reads 1 instead of 0
        with probability ``p01`` and 0 instead of 1 with ``p10``, applied
        exactly to the ⟨Z⟩ readouts (see
        :func:`hqnn_forge.noise.readout_error_map`).  It shifts every readout
        by ``p10 − p01``, which the head learns, and ``predict_proba`` runs
        without it: evaluate inside
        :func:`hqnn_forge.noise.apply_readout_error` with the same pair to
        keep the shift (#488).  Default ``None``.

    Attributes
    ----------
    classical_branch  : nn.Sequential
    classical_encoder : nn.Sequential or nn.Identity
        ``Sequential(Linear, activation)``, ``Sequential(custom module,
        activation)`` with a custom ``classical_encoder``, or ``Identity``.
    quantum_layer     : QuantumEncodingLayer or IQPEncodingLayer
    dropout           : nn.Dropout
    head              : nn.Linear

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.models import ParallelHybridClassifier
    >>> model = ParallelHybridClassifier(n_input_features=30, n_qubits=8, n_layers=2)
    >>> x = torch.randn(4, 30)
    >>> logits = model(x)           # shape (4, 1)
    >>> probs  = model.predict_proba(x)  # shape (4,), values in [0, 1]
    """

    def __init__(
        self,
        n_input_features: int = 8,
        n_qubits: int = 8,
        n_layers: int = 2,
        *,
        classical_hidden_dim: int = 16,
        use_classical_encoder: bool = True,
        dropout_p: float = 0.0,
        device_name: DeviceName = "auto",
        diff_method: DiffMethod = "auto",
        init_strategy: str = "restricted",
        encoding_type: str = "angle",
        embedding_rotation: RotationAxis = "X",
        entangler: Entangler = "ring",
        readout: Readout = "all",
        encoder_activation: str = DEFAULT_ENCODER_ACTIVATION,
        init_std: float = DEFAULT_INIT_STD,
        noise_level: float = 0.0,
        noise_position: Position = "all",
        init_seed: int | None = None,
        classical_encoder: nn.Module | None = None,
        noise_method: NoiseMethod = "density",
        noise_trajectories: int = 1,
        trainable_input_scaling: bool = False,
        shots: int | None = None,
        noise_channel: Channel = "depolarizing",
        seed: int | None = None,
        readout_error: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        init_seed = as_seed(init_seed)
        seed = as_seed(seed, "seed")
        # Plain floats: _config goes into the checkpoint as it is.
        readout_error = validate_readout_error(readout_error)
        # Building the layers draws from the global RNG (nn.Linear and
        # TorchLayer defaults), all of it overwritten by _initialise_weights.
        # With init_seed the whole build runs inside seeded_rng, so the
        # caller's stream is exactly where it was afterwards -- also when a
        # check below raises after some layers were built.
        with seeded_rng(init_seed) as reseed:
            self._config = dict(
                n_input_features=n_input_features,
                n_qubits=n_qubits,
                n_layers=n_layers,
                classical_hidden_dim=classical_hidden_dim,
                use_classical_encoder=use_classical_encoder,
                dropout_p=dropout_p,
                device_name=device_name,
                diff_method=diff_method,
                init_strategy=init_strategy,
                encoding_type=encoding_type,
                embedding_rotation=embedding_rotation,
                entangler=entangler,
                readout=readout,
                encoder_activation=encoder_activation,
                init_std=init_std,
                noise_level=noise_level,
                noise_position=noise_position,
                init_seed=init_seed,
                classical_encoder=classical_encoder,
                noise_method=noise_method,
                noise_trajectories=noise_trajectories,
                trainable_input_scaling=trainable_input_scaling,
                shots=shots,
                noise_channel=noise_channel,
                seed=seed,
                readout_error=readout_error,
            )

            # Validated before the classical branch is built, as before the shared
            # trunk: a bad option fails without drawing from the RNG.
            validate_encoder_activation(encoder_activation)
            validate_init(init_strategy, init_std)
            self.classical_hidden_dim = classical_hidden_dim

            # ── Classical branch (MLP) ────────────────────────────────────────
            self.classical_branch = nn.Sequential(
                nn.Linear(n_input_features, classical_hidden_dim),
                nn.ReLU(),
                nn.Linear(classical_hidden_dim, classical_hidden_dim),
                nn.ReLU(),
            )

            # ── Quantum branch: encoder, circuit and the fused dropout ─────────
            n_readouts = self._build_trunk(
                n_input_features=n_input_features,
                n_qubits=n_qubits,
                n_layers=n_layers,
                use_classical_encoder=use_classical_encoder,
                dropout_p=dropout_p,
                device_name=device_name,
                diff_method=diff_method,
                init_strategy=init_strategy,
                encoding_type=encoding_type,
                embedding_rotation=embedding_rotation,
                entangler=entangler,
                readout=readout,
                encoder_activation=encoder_activation,
                init_std=init_std,
                noise_level=noise_level,
                noise_position=noise_position,
                noise_method=noise_method,
                noise_trajectories=noise_trajectories,
                classical_encoder=classical_encoder,
                trainable_input_scaling=trainable_input_scaling,
                shots=shots,
                noise_channel=noise_channel,
                seed=seed,
                readout_error=readout_error,
            )

            # ── Classical head ────────────────────────────────────────────────
            self.head = nn.Linear(classical_hidden_dim + n_readouts, 1)

            # ── Small-angle restricted-variance initialisation ─────────────────
            reseed()
            self._initialise_weights()

    # ------------------------------------------------------------------
    @classmethod
    def published_shnn(cls, **overrides: object) -> ParallelHybridClassifier:
        """
        The published SHNN's quantum branch and encoder (see
        :meth:`HybridBinaryClassifier.published_shnn`) alongside the classical
        MLP branch.  ``overrides`` are passed to the constructor.
        """
        options: dict[str, object] = {**_PUBLISHED_SHNN, **overrides}
        return cls(**options)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    def _initialise_weights(self) -> None:
        """
        Initialise each block with the scheme derived for its non-linearity:
        He for the ReLU branch, Xavier for the Tanh encoder and linear head,
        restricted-variance for the quantum weights.
        """
        # Classical MLP branch: He/Kaiming — derived for ReLU, which Xavier
        # under-scales by sqrt(2) per layer.
        for module in self.classical_branch.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Tanh encoder and linear head: Xavier uniform.  A custom encoder is
        # left as given.
        for module in (*self.classical_encoder.modules(), self.head):
            if isinstance(module, nn.Linear) and id(module) not in self._custom_encoder_ids:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self._initialise_quantum_weights()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical branch and quantum branch process ``x``
        independently, are concatenated, and fed through the classification
        head.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.

        Returns
        -------
        torch.Tensor
            Raw logits, shape ``(batch_size, 1)``.  Apply ``torch.sigmoid``
            for probabilities, or pass directly to ``FocalLoss``.
        """
        # Classical branch
        classical_out = self.classical_branch(x)  # (B, classical_hidden_dim)

        # Quantum branch
        quantum_out = self._quantum_features(x)  # (B, n_outputs), values ∈ [-1, 1]

        # Fuse branches
        fused = torch.cat([classical_out, quantum_out], dim=-1)
        fused = self.dropout(fused)

        # Classification head
        return self.head(fused)  # (B, 1)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n_input_features={self.n_input_features}, "
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"classical_hidden_dim={self.classical_hidden_dim}, "
            f"total_params={self.count_parameters()}"
        )
