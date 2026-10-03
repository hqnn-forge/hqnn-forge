"""
hqnn_forge.models._trunk
========================
The classical-encoder → quantum-layer → dropout trunk every hybrid classifier shares.

:class:`HybridBinaryClassifier`, :class:`ParallelHybridClassifier`'s quantum
branch and :class:`MulticlassHybridClassifier` all build the same trunk and
differ only in what surrounds it (the head's width, the parallel model's
classical branch).  :class:`QuantumTrunk` builds it once, so a circuit option
or a fix reaches every model at the same time, instead of the multiclass model
falling behind the binary one as it had (#226).

The trunk's modules are assigned to the model itself (``classical_encoder``,
``quantum_layer``, ``dropout``), not to a sub-module, so state-dict keys and
therefore checkpoints are unchanged.  So is the order in which they are
built, which fixes how many draws each takes from the global RNG: a seeded
model has the same weights as before the refactor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hqnn_forge.encoding.amplitude_embedding import AmplitudeEncodingLayer
from hqnn_forge.encoding.angle_embedding import (
    DeviceName,
    DiffMethod,
    Entangler,
    QuantumEncodingLayer,
    Readout,
    RotationAxis,
)
from hqnn_forge.encoding.data_reuploading import DataReuploadingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers.restricted_variance import (
    block_local_init_,
    restricted_normal_init_,
)
from hqnn_forge.models.base import custom_encoder
from hqnn_forge.noise import Channel, NoiseMethod, Position

#: Constructor defaults that mark an option as "not asked for".  Both options
#: are inert outside the configuration that uses them, so a non-default value
#: there is a mistake worth naming rather than a setting to record and ignore.
DEFAULT_ENCODER_ACTIVATION = "tanh"
DEFAULT_INIT_STD = 0.1

INIT_STRATEGIES = ("restricted", "block_local", "normal")


def validate_init(init_strategy: str, init_std: float) -> None:
    """Raise ``ValueError`` for an unknown ``init_strategy`` or an ``init_std`` it would ignore."""
    if init_strategy not in INIT_STRATEGIES:
        raise ValueError(
            f"init_strategy must be 'restricted', 'block_local' or 'normal'; got {init_strategy!r}."
        )
    if init_std <= 0.0:
        raise ValueError(f"init_std must be > 0; got {init_std}.")
    if init_strategy != "normal" and init_std != DEFAULT_INIT_STD:
        raise ValueError(
            f"init_std applies to init_strategy='normal' only; "
            f"'{init_strategy}' derives its own sigma from the circuit size, so "
            f"init_std={init_std} would be recorded in the config and ignored."
        )


def validate_encoder_activation(encoder_activation: str) -> None:
    """Raise ``ValueError`` unless ``encoder_activation`` is ``"tanh"`` or ``"sigmoid"``."""
    if encoder_activation not in ("tanh", "sigmoid"):
        raise ValueError(
            f"encoder_activation must be 'tanh' or 'sigmoid'; got {encoder_activation!r}."
        )


ENCODING_TYPES = ("angle", "iqp", "amplitude", "reuploading")


def _check_encoding_options(
    encoding_type: str,
    *,
    n_input_features: int,
    n_qubits: int,
    use_classical_encoder: bool,
    diff_method: str,
    embedding_rotation: str,
    trainable_input_scaling: bool,
) -> int:
    """
    Raise ``ValueError`` for an option ``encoding_type`` cannot use; return the
    width of what enters the circuit (the classical encoder's output).

    The angle, IQP and re-uploading layers take one feature per qubit.  The
    amplitude layer takes up to ``2**n_qubits``: the classical encoder maps to
    all of them, and without it the raw features are zero-padded.
    """
    if encoding_type not in ENCODING_TYPES:
        raise ValueError(
            f"Unsupported encoding_type: {encoding_type!r}; choose 'angle', 'iqp', "
            f"'amplitude' or 'reuploading'."
        )
    if embedding_rotation != "X" and encoding_type in ("iqp", "amplitude"):
        raise ValueError(
            f"embedding_rotation applies to encoding_type='angle' or 'reuploading' only; "
            f"{'IQP' if encoding_type == 'iqp' else 'amplitude'} embedding has no rotation axis."
        )
    if trainable_input_scaling and encoding_type != "reuploading":
        raise ValueError(
            f"trainable_input_scaling applies to encoding_type='reuploading' only; the "
            f"{encoding_type} embedding uploads its inputs once."
        )
    if encoding_type == "amplitude":
        width = 2**n_qubits
        if use_classical_encoder and diff_method != "backprop":
            # The encoder is trained through the amplitudes, and PennyLane's
            # input gradient of the state preparation is NaN or silently wrong
            # under every other method (see hqnn_forge.encoding.amplitude_embedding).
            raise ValueError(
                f"encoding_type='amplitude' with a classical encoder trains the encoder "
                f"through the amplitude embedding, whose input gradient is only correct "
                f"under diff_method='backprop' (on default.qubit); got "
                f"diff_method={diff_method!r}."
            )
        if not use_classical_encoder and not 1 <= n_input_features <= width:
            raise ValueError(
                f"When use_classical_encoder=False, encoding_type='amplitude' takes "
                f"1 to 2**n_qubits = {width} features; got n_input_features={n_input_features}."
            )
        return width
    if not use_classical_encoder and n_input_features != n_qubits:
        raise ValueError(
            f"When use_classical_encoder=False, n_input_features "
            f"({n_input_features}) must equal n_qubits ({n_qubits})."
        )
    return n_qubits


class QuantumTrunk(nn.Module):
    """
    Mixin for a classifier built around the shared trunk.

    A subclass validates its own options, calls :meth:`_build_trunk` where the
    trunk belongs in its construction order, adds its head, and in ``forward``
    calls :meth:`_quantum_features`.
    """

    classical_encoder: nn.Module
    quantum_layer: (
        QuantumEncodingLayer | IQPEncodingLayer | AmplitudeEncodingLayer | DataReuploadingLayer
    )
    dropout: nn.Module
    n_input_features: int
    n_qubits: int
    n_layers: int
    init_strategy: str
    init_std: float
    use_classical_encoder: bool
    encoder_activation: str
    _custom_encoder_ids: frozenset[int]

    def _build_trunk(
        self,
        *,
        n_input_features: int,
        n_qubits: int,
        n_layers: int,
        use_classical_encoder: bool,
        dropout_p: float,
        device_name: DeviceName,
        diff_method: DiffMethod,
        init_strategy: str,
        encoding_type: str,
        embedding_rotation: RotationAxis,
        entangler: Entangler,
        readout: Readout,
        encoder_activation: str,
        init_std: float,
        noise_level: float,
        noise_position: Position,
        noise_method: NoiseMethod,
        noise_trajectories: int,
        classical_encoder: nn.Module | None,
        trainable_input_scaling: bool = False,
        shots: int | None = None,
        noise_channel: Channel = "depolarizing",
    ) -> int:
        """
        Build ``classical_encoder``, ``quantum_layer`` and ``dropout`` on ``self``.

        A custom ``classical_encoder`` module is wrapped with the activation
        and recorded in ``_custom_encoder_ids``, so the model's classical init
        leaves it as given (it may be pretrained).

        Returns the quantum layer's readout width, which the head reads.
        Raises ``ValueError`` for an inconsistent option.
        """
        validate_encoder_activation(encoder_activation)
        validate_init(init_strategy, init_std)
        # dropout_p is deliberately not validated here: the binary models have
        # always handed it to nn.Dropout unchecked, and tightening that changes
        # which checkpoints load, so it is its own change (#404).  The
        # multiclass model keeps its own [0, 1) check.
        if classical_encoder is not None and not use_classical_encoder:
            raise ValueError(
                "classical_encoder replaces the built-in encoder and needs "
                "use_classical_encoder=True; use_classical_encoder=False feeds the "
                "input to the circuit directly, with no encoder at all."
            )
        width = _check_encoding_options(
            encoding_type,
            n_input_features=n_input_features,
            n_qubits=n_qubits,
            use_classical_encoder=use_classical_encoder,
            diff_method=diff_method,
            embedding_rotation=embedding_rotation,
            trainable_input_scaling=trainable_input_scaling,
        )

        self.n_input_features = n_input_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.init_strategy = init_strategy
        self.use_classical_encoder = use_classical_encoder
        self.encoder_activation = encoder_activation
        self.init_std = init_std

        # ── Classical encoder ─────────────────────────────────────────────
        if classical_encoder is not None:
            self.classical_encoder = custom_encoder(
                classical_encoder, n_input_features, n_qubits, encoder_activation, width
            )
        elif use_classical_encoder:
            self.classical_encoder = nn.Sequential(
                nn.Linear(n_input_features, width),
                nn.Tanh() if encoder_activation == "tanh" else nn.Sigmoid(),
            )
        else:
            if encoder_activation != DEFAULT_ENCODER_ACTIVATION:
                raise ValueError(
                    f"encoder_activation applies with use_classical_encoder=True only; "
                    f"without the encoder the features enter the circuit as given, so "
                    f"{encoder_activation!r} would be recorded in the config and ignored."
                )
            self.classical_encoder = nn.Identity()

        # ── Quantum encoding layer ────────────────────────────────────────
        if encoding_type == "angle":
            self.quantum_layer = QuantumEncodingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                rotation=embedding_rotation,
                device_name=device_name,
                diff_method=diff_method,
                entangler=entangler,
                readout=readout,
                noise_level=noise_level,
                noise_position=noise_position,
                noise_method=noise_method,
                noise_trajectories=noise_trajectories,
                shots=shots,
                noise_channel=noise_channel,
            )
        elif encoding_type == "iqp":
            self.quantum_layer = IQPEncodingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                n_repeats=1,
                device_name=device_name,
                diff_method=diff_method,
                entangler=entangler,
                readout=readout,
                noise_level=noise_level,
                noise_position=noise_position,
                noise_method=noise_method,
                noise_trajectories=noise_trajectories,
                shots=shots,
                noise_channel=noise_channel,
            )
        elif encoding_type == "amplitude":
            self.quantum_layer = AmplitudeEncodingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                n_features=width if use_classical_encoder else n_input_features,
                device_name=device_name,
                diff_method=diff_method,
                entangler=entangler,
                readout=readout,
                noise_level=noise_level,
                noise_position=noise_position,
                noise_method=noise_method,
                noise_trajectories=noise_trajectories,
                shots=shots,
                noise_channel=noise_channel,
            )
        else:
            self.quantum_layer = DataReuploadingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                rotation=embedding_rotation,
                device_name=device_name,
                diff_method=diff_method,
                trainable_input_scaling=trainable_input_scaling,
                entangler=entangler,
                readout=readout,
                noise_level=noise_level,
                noise_position=noise_position,
                noise_method=noise_method,
                noise_trajectories=noise_trajectories,
                shots=shots,
                noise_channel=noise_channel,
            )

        # ── Dropout ───────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        self._custom_encoder_ids = frozenset(
            map(id, classical_encoder.modules()) if classical_encoder is not None else ()
        )
        return self.quantum_layer.n_outputs

    # ------------------------------------------------------------------
    def _initialise_quantum_weights(self) -> None:
        """Draw the circuit weights with ``init_strategy`` (the models' classical init is their own)."""
        weights = self.quantum_layer.qlayer.weights  # dim 0: layer (variational_weight_shape)
        if self.init_strategy == "block_local":
            block_local_init_(weights.data, n_qubits=self.n_qubits)
        elif self.init_strategy == "normal":
            # The published SHNN's init: N(0, init_std²), independent of size.
            with torch.no_grad():
                weights.normal_(mean=0.0, std=self.init_std)
        else:
            restricted_normal_init_(weights.data, n_qubits=self.n_qubits, n_layers=self.n_layers)

    # ------------------------------------------------------------------
    def _quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder, angle scaling and circuit: ``x`` → ``(batch, n_outputs)`` ⟨Z⟩ values."""
        x = self.classical_encoder(x)  # (B, n_qubits)
        # Tanh output (-1, 1) → (-π, π), or sigmoid output (0, 1) → (0, π).
        # Bypassed input is already in (-π, π); scaling it again would alias
        # angles mod 2π.
        if self.use_classical_encoder:
            x = x * torch.pi
        return self.quantum_layer(x)
