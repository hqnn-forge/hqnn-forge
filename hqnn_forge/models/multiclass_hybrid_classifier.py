"""
hqnn_forge.models.multiclass_hybrid_classifier
==============================================
Hybrid quantum-classical classifier for ``n_classes ≥ 2`` classes.

Architecture
------------
::

    Input (batch, n_input_features)
         │
         ▼
    [Classical encoder]  nn.Linear(n_input_features → n_qubits) + Tanh (or Sigmoid), · π
         │
         ▼
    [Quantum encoding]   QuantumEncodingLayer / IQPEncodingLayer (n_qubits, n_layers)
         │                  → ⟨Z_i⟩, shape (batch, n_outputs)
         ▼
    [Dropout]
         │
         ▼
    [Class heads]        nn.Linear(n_outputs → n_classes)
         │
         ▼
    Raw logits (batch, n_classes)

Design Notes
------------
* The quantum layer is shared; the class heads are the rows of one
  ``nn.Linear(n_outputs, n_classes)``, each a binary head reading the same
  ``n_outputs`` expectation values (``n_qubits``, or 1 with
  ``readout="first"``).  This is the "ensemble of binary heads sharing the
  quantum layer" option: the quantum parameter count does not grow with
  ``n_classes``, only the head does (``n_outputs + 1`` per class).

* ``strategy`` selects how the ``n_classes`` logits are turned into
  probabilities and, by implication, which loss to train with:

  - ``"softmax"`` (default): ``predict_proba`` is the softmax over classes.
    Train with ``nn.CrossEntropyLoss`` on integer labels.
  - ``"one_vs_rest"``: each head is an independent binary classifier
    (class ``c`` against the rest); ``predict_proba`` applies a sigmoid to
    each logit and normalises the ``n_classes`` scores to sum to one, as
    ``sklearn.multiclass.OneVsRestClassifier`` does.  Train with
    ``nn.BCEWithLogitsLoss`` on one-hot targets (see :meth:`one_hot`).

  ``forward`` returns raw logits in both cases, and ``predict`` is the
  argmax in both cases; only ``predict_proba`` differs.

* With ``n_classes=2`` the softmax model is a two-logit form of the binary
  classifier.  :class:`~hqnn_forge.models.HybridBinaryClassifier` with its
  single logit is the smaller model for that case and works with
  :class:`~hqnn_forge.utils.FocalLoss`; this class exists for ``n_classes >
  2``.

* The classical encoder, quantum layer and dropout are the trunk
  :class:`~hqnn_forge.models.HybridBinaryClassifier` builds, from the same
  code, so every circuit, initialisation and training-noise option behaves
  exactly as there.  Only the head differs.

Parameters
----------
n_input_features:
    Dimensionality of the raw / PCA-reduced input.
n_qubits:
    Number of qubits.
n_layers:
    Number of variational layers in the quantum circuit.
n_classes:
    Number of classes, ``≥ 2``.
strategy:
    ``"softmax"`` or ``"one_vs_rest"``; see above.
use_classical_encoder, dropout_p, device_name, diff_method, init_strategy,
encoding_type, embedding_rotation, entangler, readout, encoder_activation,
init_std, noise_level, noise_position, noise_method, noise_trajectories,
trainable_input_scaling, init_seed, classical_encoder, shots, noise_channel, seed,
readout_error:
    As for :class:`~hqnn_forge.models.HybridBinaryClassifier`.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import (
    DeviceName,
    DiffMethod,
    Entangler,
    Readout,
    RotationAxis,
)
from hqnn_forge.models._trunk import DEFAULT_ENCODER_ACTIVATION, DEFAULT_INIT_STD, QuantumTrunk
from hqnn_forge.models.base import ClassifierBase
from hqnn_forge.noise import Channel, NoiseMethod, Position
from hqnn_forge.utils.rng import as_seed, seeded_rng

MulticlassStrategy = Literal["softmax", "one_vs_rest"]


class MulticlassHybridClassifier(QuantumTrunk, ClassifierBase):
    """
    Hybrid quantum-classical multiclass classifier.

    See the module docstring for the architecture and the two strategies.

    Parameters
    ----------
    n_input_features:
        Number of raw (or PCA-reduced) input features.  Default: 8.
    n_qubits:
        Number of qubits in the quantum encoding layer.  Default: 8.
    n_layers:
        VQC ansatz layers.  Default: 2.
    n_classes:
        Number of classes.  Default: 3.  Must be ``≥ 2``.
    strategy:
        ``"softmax"`` (default) or ``"one_vs_rest"``.
    use_classical_encoder, dropout_p, device_name, diff_method,
    init_strategy, encoding_type, embedding_rotation, entangler, readout,
    encoder_activation, init_std, noise_level, noise_position, noise_method,
    noise_trajectories, classical_encoder, trainable_input_scaling, shots, noise_channel, seed,
    readout_error:
        The trunk's options, exactly as for
        :class:`~hqnn_forge.models.HybridBinaryClassifier`.  With
        ``readout="first"`` every class head reads ⟨Z_0⟩ alone.
    init_seed:
        Seed for weight initialisation.  ``None`` (default) draws the initial
        weights from the global torch RNG; an int draws them from a private RNG
        seeded with it, so the same seed gives the same weights and the global
        RNG is left exactly as it was.

    Attributes
    ----------
    classical_encoder : nn.Sequential or nn.Identity
    quantum_layer     : QuantumEncodingLayer or IQPEncodingLayer
    dropout           : nn.Dropout or nn.Identity
    head              : nn.Linear
        ``weight[c]`` and ``bias[c]`` are class ``c``'s head.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.models import MulticlassHybridClassifier
    >>> model = MulticlassHybridClassifier(n_input_features=6, n_qubits=4, n_layers=1, n_classes=3)
    >>> x = torch.randn(5, 6)
    >>> model(x).shape                 # logits
    torch.Size([5, 3])
    >>> model.predict_proba(x).shape   # rows sum to 1
    torch.Size([5, 3])
    >>> model.predict(x).shape         # argmax labels in {0, 1, 2}
    torch.Size([5])
    """

    def __init__(
        self,
        n_input_features: int = 8,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_classes: int = 3,
        *,
        strategy: MulticlassStrategy = "softmax",
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
        noise_method: NoiseMethod = "density",
        noise_trajectories: int = 1,
        init_seed: int | None = None,
        classical_encoder: nn.Module | None = None,
        trainable_input_scaling: bool = False,
        shots: int | None = None,
        noise_channel: Channel = "depolarizing",
        seed: int | None = None,
        readout_error: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        init_seed = as_seed(init_seed)
        seed = as_seed(seed, "seed")
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
                n_classes=n_classes,
                strategy=strategy,
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
                init_seed=init_seed,
                classical_encoder=classical_encoder,
                trainable_input_scaling=trainable_input_scaling,
                shots=shots,
                noise_channel=noise_channel,
                seed=seed,
                readout_error=readout_error,
            )

            if n_classes < 2:
                raise ValueError(f"n_classes must be ≥ 2; got {n_classes}.")
            if strategy not in ("softmax", "one_vs_rest"):
                raise ValueError(f"strategy must be 'softmax' or 'one_vs_rest'; got {strategy!r}.")
            if not 0.0 <= dropout_p < 1.0:
                raise ValueError(f"dropout_p must be in [0, 1); got {dropout_p}.")
            self.n_classes = n_classes
            self.strategy = strategy

            # ── Shared trunk: encoder → quantum layer → dropout ───────────────
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

            # ── Class heads: one row per class ────────────────────────────────
            self.head = nn.Linear(n_readouts, n_classes)

            # ── Small-angle restricted-variance initialisation ─────────────────
            reseed()
            self._initialise_weights()

    # ------------------------------------------------------------------
    def _initialise_weights(self) -> None:
        """Xavier on the encoder and heads; ``init_strategy`` on the quantum weights."""
        # A custom encoder is left as given (it may be pretrained).
        for module in self.modules():
            if isinstance(module, nn.Linear) and id(module) not in self._custom_encoder_ids:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self._initialise_quantum_weights()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical encode → quantum encode → class heads.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.

        Returns
        -------
        torch.Tensor
            Raw logits, shape ``(batch_size, n_classes)``.  Feed to
            ``nn.CrossEntropyLoss`` (``strategy="softmax"``) or to
            ``nn.BCEWithLogitsLoss`` with :meth:`one_hot` targets
            (``strategy="one_vs_rest"``).
        """
        x = self._quantum_features(x)  # (B, n_outputs), values ∈ [-1, 1]
        x = self.dropout(x)
        return self.head(x)  # (B, n_classes)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-class probabilities, shape ``(batch_size, n_classes)``, rows
        summing to one.  Softmax of the logits under ``"softmax"``; sigmoid
        of each logit, normalised across classes, under ``"one_vs_rest"``.
        The latter is computed as ``softmax(logsigmoid(logits))``, which is
        the same quantity but stays finite when every sigmoid in a row
        underflows (all logits below about -88 in float32), where the direct
        ratio would be ``0 / 0``.

        Runs in eval mode whatever mode the model is in (dropout off) and
        restores every submodule's ``training`` flag afterwards.
        """
        logits = self._eval_logits(x)
        if self.strategy == "softmax":
            return torch.softmax(logits, dim=-1)
        return torch.softmax(nn.functional.logsigmoid(logits), dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicted class labels, shape ``(batch_size,)``, dtype ``torch.long``:
        the argmax of the logits, under both strategies.

        This is the argmax of :meth:`predict_proba` in exact arithmetic, since
        both normalisations are monotone per row, but not always in floating
        point: under ``"one_vs_rest"`` large positive logits saturate the
        sigmoid, so e.g. logits ``[17, 20, 30]`` give float32 probabilities in
        which classes 1 and 2 tie, and ``predict_proba(x).argmax(-1)`` returns
        class 1 while ``predict`` returns class 2.  Use this method, not
        ``predict_proba(x).argmax(-1)``, for labels.
        """
        return self._eval_logits(x).argmax(dim=-1)

    # ------------------------------------------------------------------
    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        """
        Integer labels ``(batch_size,)`` → one-hot ``(batch_size, n_classes)``
        in the dtype of the class heads, the target format of
        ``nn.BCEWithLogitsLoss`` for the one-vs-rest strategy.  Matching the
        heads' dtype keeps the loss of a ``model.double()`` in float64.

        Float labels are accepted only if integer-valued; anything else (e.g.
        smoothed targets) raises instead of being truncated.
        """
        if y.is_floating_point() and not torch.equal(y, y.round()):
            raise ValueError("one_hot expects integer class labels; got non-integer values.")
        one_hot = nn.functional.one_hot(y.long(), num_classes=self.n_classes)
        return one_hot.to(self.head.weight.dtype)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n_input_features={self.n_input_features}, "
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_classes={self.n_classes}, "
            f"strategy={self.strategy!r}, "
            f"total_params={self.count_parameters()}"
        )
