"""
hqnn_forge.models.classical_baseline
====================================
A plain MLP binary classifier: the classical control of a hybrid model.

An ablation that switches a trained model's circuit off
(:func:`hqnn_forge.utils.disable_quantum_layer`) measures how much that model
depends on it; it cannot say whether a classical model of the same size,
trained from scratch on the same data, would do as well.  ``ClassicalBaseline``
is that model, and :func:`hqnn_forge.utils.ablation.classical_baseline` builds
one matched in parameter count to a given hybrid classifier.

Architecture
------------
``Linear(n_input_features → h_1) → act → … → Linear(h_{k-1} → h_k) → act →
Dropout → Linear(h_k → 1)``, returning one raw logit per sample like every
classifier in this package, so it trains under
:func:`hqnn_forge.training.train_model` and scores through ``predict_proba``.
Linear layers are initialised as in the hybrid classifiers: Xavier-uniform
weights, zero biases, drawn from a private RNG when ``init_seed`` is given.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
import torch.nn as nn

from hqnn_forge.models.base import BinaryClassifierBase
from hqnn_forge.utils.rng import as_seed, seeded_rng

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


def mlp_parameter_count(n_input_features: int, hidden_dims: Sequence[int]) -> int:
    """Trainable parameters of ``ClassicalBaseline(n_input_features, hidden_dims)``."""
    widths = [n_input_features, *hidden_dims, 1]
    return sum(w_in * w_out + w_out for w_in, w_out in itertools.pairwise(widths))


class ClassicalBaseline(BinaryClassifierBase):
    """
    MLP binary classifier with the hybrid classifiers' interface.

    Parameters
    ----------
    n_input_features:
        Input width.
    hidden_dims:
        Width of each hidden layer, at least one.
    activation:
        ``"relu"`` (default) or ``"tanh"``, after every hidden layer.
    dropout_p:
        Dropout before the output layer, in ``[0, 1)``.  Default: 0.0.
    init_seed:
        Seed for weight initialisation, as on the hybrid classifiers.  ``None``
        (default) draws the initial weights from the global torch RNG; an int
        draws them from a private RNG seeded with it, so the same seed gives
        the same weights and the global RNG is left exactly as it was.

    Examples
    --------
    >>> model = ClassicalBaseline(n_input_features=30, hidden_dims=[10])
    >>> model.count_parameters()
    321
    """

    def __init__(
        self,
        n_input_features: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "relu",
        dropout_p: float = 0.0,
        init_seed: int | None = None,
    ) -> None:
        super().__init__()
        init_seed = as_seed(init_seed)
        # As in the hybrid classifiers: the whole build runs inside seeded_rng,
        # so a seeded model leaves the caller's stream where it was, also when
        # a check below raises.
        with seeded_rng(init_seed) as reseed:
            hidden = [int(h) for h in hidden_dims]
            self._config = dict(
                n_input_features=n_input_features,
                hidden_dims=hidden,
                activation=activation,
                dropout_p=dropout_p,
                init_seed=init_seed,
            )
            if not hidden or min(hidden) < 1:
                raise ValueError(f"hidden_dims must hold at least one width >= 1; got {hidden}.")
            if activation not in _ACTIVATIONS:
                raise ValueError(
                    f"activation must be one of {sorted(_ACTIVATIONS)}; got {activation!r}."
                )
            if not 0.0 <= dropout_p < 1.0:
                raise ValueError(f"dropout_p must lie in [0, 1); got {dropout_p}.")

            self.n_input_features = n_input_features
            layers: list[nn.Module] = []
            for w_in, w_out in itertools.pairwise([n_input_features, *hidden]):
                layers += [nn.Linear(w_in, w_out), _ACTIVATIONS[activation]()]
            self.body = nn.Sequential(*layers)
            self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
            self.head = nn.Linear(hidden[-1], 1)

            # nn.Linear's own init draws are all overwritten below; reseeding
            # first makes the seeded weights independent of how many it took.
            reseed()
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits, shape ``(batch_size, 1)``."""
        return self.head(self.dropout(self.body(x)))
