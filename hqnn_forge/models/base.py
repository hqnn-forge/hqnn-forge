"""
hqnn_forge.models.base
======================
Shared inference and bookkeeping API for the classifiers.

:class:`ClassifierBase` holds what does not depend on the head: the recorded
constructor arguments (``get_config``, which checkpoints rely on), parameter
counting, and the eval-mode, gradient-free forward pass every ``predict``
starts from.  :class:`BinaryClassifierBase` adds the single-logit head's
sigmoid ``predict_proba`` and thresholded ``predict``;
:class:`~hqnn_forge.models.MulticlassHybridClassifier` adds its softmax and
one-vs-rest ones.  Each piece lives here once, so a fix applies to every model
rather than to whichever copy happened to be found (cf. #59, which had to be
fixed twice).

Subclasses implement ``__init__`` and ``forward`` only.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import torch
import torch.nn as nn

from hqnn_forge.utils.modes import eval_mode


def custom_encoder(
    module: nn.Module,
    n_input_features: int,
    n_qubits: int,
    activation: str,
    width: int | None = None,
) -> nn.Sequential:
    """
    ``module`` followed by the model's ``activation``, after checking that it
    maps ``(batch, n_input_features)`` to ``(batch, width)``.

    ``width`` is what enters the circuit: ``n_qubits`` (the default) for the
    angle, IQP and re-uploading encodings, ``2**n_qubits`` for amplitude.

    The width is checked with one forward pass on zeros, in eval mode and
    without gradients, so it neither trains nor updates running statistics.
    ``module`` is used as given -- not copied and not re-initialised -- so a
    pretrained extractor keeps its weights and trains with the model.

    Raises
    ------
    TypeError
        If ``module`` is not an ``nn.Module``.
    ValueError
        If the forward pass fails or returns another shape.

    Warns
    -----
    UserWarning
        If ``module`` ends in ``nn.Tanh`` or ``nn.Sigmoid``: the model applies
        ``activation`` on top, so the angles would be squashed twice.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"classical_encoder must be an nn.Module; got {type(module).__name__}.")
    probe = torch.zeros(2, n_input_features)
    try:
        with torch.no_grad(), eval_mode(module):
            out = module(probe)
    except Exception as exc:
        raise ValueError(
            f"classical_encoder failed on an input of shape {tuple(probe.shape)}, i.e. "
            f"(batch, n_input_features={n_input_features}): {type(exc).__name__}: {exc}"
        ) from exc
    shape = tuple(out.shape) if isinstance(out, torch.Tensor) else None
    expected = n_qubits if width is None else width
    if shape != (2, expected):
        what = f"n_qubits={n_qubits}" if expected == n_qubits else f"2**n_qubits={expected}"
        raise ValueError(
            f"classical_encoder must map (batch, n_input_features={n_input_features}) to "
            f"(batch, {what}); on a batch of 2 it returned "
            f"{shape if shape is not None else type(out).__name__}."
        )
    last = list(module.modules())[-1]
    if isinstance(last, (nn.Tanh, nn.Sigmoid)):
        warnings.warn(
            f"classical_encoder ends in {type(last).__name__}, and the model applies "
            f"encoder_activation={activation!r} on top of it, so the angles are squashed "
            f"twice.  Drop the final activation from the module; the model bounds the "
            f"angles itself.",
            UserWarning,
            stacklevel=3,
        )
    return nn.Sequential(module, nn.Tanh() if activation == "tanh" else nn.Sigmoid())


class ClassifierBase(nn.Module):
    """
    Head-agnostic base class for the classifiers.

    Methods
    -------
    count_parameters(trainable_only=True)
        Total number of (trainable) parameters, quantum and classical.
    get_config()
        The constructor arguments, so ``type(model)(**model.get_config())``
        rebuilds an equivalent architecture.  Subclasses record them in
        ``self._config`` at the top of ``__init__``.
    """

    _config: dict[str, Any] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement forward(x) -> logits.")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        ``forward(x)`` in eval mode and without gradients.

        ``no_grad`` alone leaves ``nn.Dropout`` (and training-time circuit
        noise) active: they check ``self.training``, not grad mode.  Every
        submodule's ``training`` flag is restored afterwards, so calling this
        mid-training leaves the model exactly as it was.
        """
        with eval_mode(self):
            return self.forward(x)

    # ------------------------------------------------------------------
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total parameter count (quantum + classical)."""
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        """
        Constructor arguments of this model, as a fresh dict.

        ``type(model)(**model.get_config())`` builds a model with the same
        architecture (load a ``state_dict`` for the trained weights).  Its
        initial weights are fresh draws only if ``init_seed`` is ``None``: a
        model built with ``init_seed`` rebuilds the *same* initial weights, so
        for restarts or ensemble members pass ``init_seed=None`` or a new seed.
        Used by ``hqnn_forge.utils.checkpoint``.

        A module argument (a custom ``classical_encoder``) is deep-copied, so
        a model built from the config does not share it with this one; the
        copy carries the module's current weights, since the model never
        re-initialises a custom encoder.
        """
        if self._config is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not record its constructor arguments; "
                f"set self._config in __init__."
            )
        return {
            name: copy.deepcopy(value) if isinstance(value, nn.Module) else value
            for name, value in self._config.items()
        }


class BinaryClassifierBase(ClassifierBase):
    """
    Base class for hybrid binary classifiers.

    Contract for subclasses
    -----------------------
    ``forward(x)`` takes ``(batch, n_input_features)`` and returns raw logits of
    shape ``(batch, 1)``.  ``predict_proba`` and ``predict`` are derived from it
    and must not be overridden to keep the two models interchangeable.

    Methods
    -------
    predict_proba(x)
        Sigmoid of the logits, shape ``(batch,)``, computed in eval mode.
    predict(x, threshold=0.5)
        ``predict_proba(x) >= threshold`` as ``torch.long``.
    count_parameters(), get_config()
        From :class:`ClassifierBase`.

    Attributes
    ----------
    head : nn.Linear
        The output layer producing the single logit.  Every subclass assigns it
        in ``__init__``, so code that only reads it can be typed against this
        class.  ``classical_encoder`` and ``quantum_layer`` are deliberately not
        declared here: the hybrid models have them, but ``ClassicalBaseline``
        does not, so a declaration on the base would let mypy accept an access
        that raises ``AttributeError`` at runtime.
    """

    # Declaration only: nn.Module registers the submodule when a subclass
    # assigns it, so this changes neither state_dict keys nor checkpoints.
    head: nn.Linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(x) -> logits of shape (batch, 1)."
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute positive-class probabilities (inference mode, no gradients).

        Runs in eval mode whatever mode the model is in, so dropout is off and
        repeated calls on the same input agree (except under finite ``shots``,
        whose readouts are sampled afresh on every call).  Every submodule's ``training``
        flag is restored afterwards, so calling this mid-training leaves the
        model exactly as it was.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.

        Returns
        -------
        torch.Tensor
            Probability of class 1, shape ``(batch_size,)``, values ∈ [0, 1].
        """
        return torch.sigmoid(self._eval_logits(x)).squeeze(-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary labels.  Runs in eval mode, like ``predict_proba``.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.
        threshold:
            Decision threshold.  Default: 0.5.
            For imbalanced datasets consider tuning via ROC/PR curves.

        Returns
        -------
        torch.Tensor
            Binary label tensor of shape ``(batch_size,)``, dtype ``torch.long``.
        """
        return (self.predict_proba(x) >= threshold).long()
