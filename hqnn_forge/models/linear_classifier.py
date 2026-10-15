"""
hqnn_forge.models.linear_classifier
===================================
A linear binary classifier: one affine map from the inputs to a logit.

It is the smallest head a fixed feature map can be followed by, such as
:class:`~hqnn_forge.rydberg.RydbergFeatureMap`
(``examples/benchmark_rydberg_features.py`` trains it on those features), and
on raw inputs it is the linear baseline.
:class:`~hqnn_forge.models.ClassicalBaseline` cannot stand in for it: it
needs at least one hidden layer.

The model
---------
For an input ``x ∈ ℝ^n`` (one row of the batch; the inputs carry whatever
unit the caller gives them, the logit is dimensionless)::

    z(x) = Σ_j w_j x_j + b                  logit, shape (batch, 1)
    P(y = 1 | x) = σ(z) = 1 / (1 + e^(−z))  predict_proba

with ``n`` weights ``w`` and one bias ``b``: ``n + 1`` trainable parameters.
A positive weight raises the probability of class 1 as its input grows.  The
decision boundary at threshold ½ is the hyperplane ``w · x + b = 0``.

Logistic regression
-------------------
The model above with the Bernoulli likelihood is logistic regression.  For
labels ``y ∈ {0, 1}`` the negative mean log-likelihood is::

    −(1/M) Σ_m [ y_m ln σ(z_m) + (1 − y_m) ln(1 − σ(z_m)) ]

which is what ``torch.nn.BCEWithLogitsLoss`` computes from the logits, so
training this model with that loss fits a logistic regression by maximum
likelihood, without a penalty.  With another loss (``FocalLoss``) it is a
linear classifier, but not that estimator.

Initialisation
--------------
As in the other classifiers: Xavier-uniform weights, drawn from
``U(−a, a)`` with ``a = √(6 / (n + 1))`` (fan-in ``n``, fan-out 1), and a
zero bias, from a private RNG when ``init_seed`` is given.
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn

from hqnn_forge.models.base import BinaryClassifierBase
from hqnn_forge.utils.rng import as_seed, seeded_rng


class LinearClassifier(BinaryClassifierBase):
    """
    Linear binary classifier, ``z = w · x + b``, with the classifiers' interface.

    One raw logit per sample, like every classifier in this package, so it
    trains under :func:`hqnn_forge.training.train_model` and scores through
    ``predict_proba``.  The module docstring gives the model and says when it
    is a logistic regression.

    Parameters
    ----------
    n_input_features:
        Input width ``n``, an integer ``>= 1``.
    init_seed:
        Seed for weight initialisation, as on the other classifiers.  ``None``
        (default) draws the initial weights from the global torch RNG; an int
        draws them from a private RNG seeded with it, so the same seed gives
        the same weights and the global RNG is left exactly as it was.

    Attributes
    ----------
    head : nn.Linear
        ``Linear(n_input_features → 1)``: the whole model.
    n_input_features : int

    Raises
    ------
    TypeError
        If ``init_seed`` is neither an int nor ``None``.
    ValueError
        If ``n_input_features`` is not an integer ``>= 1``.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.models import LinearClassifier
    >>> model = LinearClassifier(n_input_features=4)
    >>> model.count_parameters()  # 4 weights and a bias
    5
    >>> with torch.no_grad():
    ...     _ = model.head.weight.copy_(torch.tensor([[1.0, 0.0, -1.0, 2.0]]))
    ...     _ = model.head.bias.fill_(0.5)
    >>> model(torch.tensor([[2.0, 7.0, 1.0, -1.0]])).tolist()  # 2 − 1 − 2 + 0.5
    [[-0.5]]
    """

    n_input_features: int

    def __init__(self, n_input_features: int, *, init_seed: int | None = None) -> None:
        super().__init__()
        init_seed = as_seed(init_seed)
        # As in the other classifiers: the whole build runs inside seeded_rng,
        # so a seeded model leaves the caller's stream where it was, also when
        # the check below raises.
        with seeded_rng(init_seed) as reseed:
            self._config = dict(n_input_features=n_input_features, init_seed=init_seed)
            if (
                isinstance(n_input_features, bool)
                or not isinstance(n_input_features, numbers.Integral)
                or n_input_features < 1
            ):
                raise ValueError(
                    f"n_input_features must be an integer >= 1; got {n_input_features!r}."
                )
            self.n_input_features = int(n_input_features)
            self._config["n_input_features"] = self.n_input_features
            self.head = nn.Linear(self.n_input_features, 1)

            # nn.Linear's own init draws are overwritten below; reseeding
            # first makes the seeded weights independent of how many it took.
            reseed()
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits ``w · x + b``, shape ``(batch_size, 1)``."""
        logits: torch.Tensor = self.head(x)
        return logits
