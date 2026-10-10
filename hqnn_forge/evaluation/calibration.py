"""
hqnn_forge.evaluation.calibration
=================================
Calibration of a classifier's probabilities, and post-hoc fixes (#319; the
multiclass measures #360).

For risk scoring -- fraud, credit, churn -- the predicted probability is used as
a probability, so it has to mean what it says: of the samples scored 0.2, about
20 % should be positive.  Ranking (AUC) and thresholded performance (MCC) do
not see this, and training choices move it: focal loss, the library default,
is known to under-state confidence (Mukhoti et al. 2020), and resampling for
imbalance shifts the base rate the model learns.

Measures
--------
``brier_score``
    Mean squared error ``mean((p − y)²)`` of the probabilities: calibration
    and sharpness together; 0 is perfect, 0.25 is a constant 0.5.
``expected_calibration_error``
    ``Σ_b (n_b / n) · |mean(y)_b − mean(p)_b|`` over probability bins: the
    average gap between confidence and frequency (Naeini et al. 2015; Guo et
    al. 2017), here on the positive-class probability.  ``strategy="uniform"``
    uses ``n_bins`` equal-width bins of [0, 1]; ``"quantile"`` equal-count
    bins, which on imbalanced data keeps the bins near 0 from swallowing
    every sample.
``reliability_curve``
    Per non-empty bin, the mean probability and the observed frequency, the
    points of a reliability diagram (``plots.plot_reliability_diagram``).

For ``K`` classes, with ``(n, K)`` probabilities whose rows sum to 1 and
integer labels ``0 … K−1``:

``multiclass_brier_score``
    ``mean_i Σ_k (p_ik − y_ik)²`` with ``y`` one-hot.  0 is perfect; it
    ranges up to 2, and a constant ``1/K`` scores ``1 − 1/K``.  For
    ``K ≥ 3`` it is scikit-learn's ``brier_score_loss`` on ``(n, K)`` input
    (scikit-learn 1.7 on).  For ``K = 2`` it is twice the binary
    ``brier_score``, which counts one class only, and twice scikit-learn's
    default, which halves the two-class score (``scale_by_half="auto"``;
    ``scale_by_half=False`` gives this value).
``top_label_ece``
    The ECE of the confidence ``max_k p_ik`` against whether the top class
    is right, Guo et al.'s definition: does a 70 % prediction come true
    70 % of the time?
``classwise_ece``
    The mean over classes of each class's one-vs-rest ECE: every class's
    probability, not only the top one, must be calibrated (Kull et al. 2019).

Post-hoc calibration
--------------------
All three are fitted on a validation split by minimising the negative
log-likelihood of the labels, and applied to logits:

``TemperatureScaler``
    ``σ(z / T)``: one parameter.  Monotone, so the ranking -- AUC, and the
    MCC at a correspondingly moved threshold -- is unchanged; only
    confidence is rescaled.  ``T > 1`` softens over-confident predictions.
``PlattScaler``
    ``σ(a·z + b)`` (Platt 1999): also shifts the base rate, e.g. after
    training on oversampled data.  Monotone for ``a > 0``.
``MulticlassTemperatureScaler``
    ``softmax(z / T)`` for ``(n, K)`` logits, one ``T`` shared by every
    class (Guo et al. 2017): dividing every logit by the same positive
    number keeps their order, so the predicted class never changes.

References
----------
* Platt (1999) "Probabilistic outputs for support vector machines".
* Naeini, Cooper & Hauskrecht (2015) "Obtaining well calibrated
  probabilities using Bayesian binning", AAAI.
* Guo, Pleiss, Sun & Weinberger (2017) "On calibration of modern neural
  networks", ICML.
* Mukhoti et al. (2020) "Calibrating deep neural networks using focal loss",
  NeurIPS.
* Kull et al. (2019) "Beyond temperature scaling: obtaining well-calibrated
  multi-class probabilities with Dirichlet calibration", NeurIPS (classwise
  ECE).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from hqnn_forge.evaluation.thresholds import _validate_scores

__all__ = [
    "MulticlassTemperatureScaler",
    "PlattScaler",
    "TemperatureScaler",
    "brier_score",
    "classwise_ece",
    "expected_calibration_error",
    "multiclass_brier_score",
    "reliability_curve",
    "top_label_ece",
]

BinStrategy = Literal["uniform", "quantile"]


def _pair(y_true: object, prob: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Float64 labels and probabilities on the CPU, whatever device they came from."""
    t, p = _validate_scores(y_true, prob)
    # contiguous: a column of a 2-D array (predict_proba(X)[:, 1]) is a strided
    # view, which torch.searchsorted in the binning copies with a warning.
    y, p = t.cpu().to(torch.float64), p.cpu().contiguous()
    if torch.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError("prob must hold probabilities in [0, 1].")
    return y, p


def brier_score(y_true: object, prob: object) -> float:
    """``mean((prob − y_true)²)``."""
    y, p = _pair(y_true, prob)
    return float(((p - y) ** 2).mean())


def _validate_bins(n_bins: int, strategy: BinStrategy) -> None:
    """Raise ValueError on non-positive bin count or an unrecognised strategy."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}.")
    if strategy not in ("uniform", "quantile"):
        raise ValueError(f"strategy must be 'uniform' or 'quantile'; got {strategy!r}.")


def _bins(p: torch.Tensor, n_bins: int, strategy: BinStrategy) -> torch.Tensor:
    """Bin index of every probability; the top edge belongs to the last bin."""
    _validate_bins(n_bins, strategy)
    if strategy == "uniform":
        edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    elif strategy == "quantile":
        edges = torch.quantile(p, torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64))
    else:
        raise AssertionError(strategy)  # _validate_bins and this list are out of sync
    # Right-closed at the top, as sklearn.calibration.calibration_curve does.
    return torch.searchsorted(edges[1:-1], p, right=True)


def _reliability_curve(
    y: torch.Tensor, p: torch.Tensor, n_bins: int, strategy: BinStrategy
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binned curve of validated float64 CPU tensors; see ``reliability_curve``."""
    # contiguous: a column of a 2-D tensor (p[:, k]) is a strided view, which
    # torch.searchsorted in the binning copies with a warning.
    p = p.contiguous()
    idx = _bins(p, n_bins, strategy)
    counts = torch.bincount(idx, minlength=n_bins).to(torch.float64)
    sum_p = torch.bincount(idx, weights=p, minlength=n_bins)
    sum_y = torch.bincount(idx, weights=y, minlength=n_bins)
    keep = counts > 0
    return sum_p[keep] / counts[keep], sum_y[keep] / counts[keep], counts[keep]


def reliability_curve(
    y_true: object, prob: object, n_bins: int = 10, strategy: BinStrategy = "uniform"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ``(mean probability, observed frequency, count)`` per non-empty bin.

    With the default uniform bins this is ``sklearn.calibration.calibration_curve``
    (which returns the frequency first and the mean probability second).
    """
    y, p = _pair(y_true, prob)
    return _reliability_curve(y, p, n_bins, strategy)


def _ece(y: torch.Tensor, p: torch.Tensor, n_bins: int, strategy: BinStrategy) -> float:
    """Expected calibration error of validated float64 CPU tensors."""
    confidence, frequency, counts = _reliability_curve(y, p, n_bins, strategy)
    return float((counts / counts.sum() * (frequency - confidence).abs()).sum())


def expected_calibration_error(
    y_true: object, prob: object, n_bins: int = 10, strategy: BinStrategy = "uniform"
) -> float:
    """``Σ_b (n_b / n) · |frequency_b − mean probability_b|`` over the bins."""
    y, p = _pair(y_true, prob)
    return _ece(y, p, n_bins, strategy)


def _class_labels(y_true: object, name: str, n: int, k: int) -> torch.Tensor:
    """Integer class labels ``(n,)`` in ``0 … k−1`` on the CPU, for ``n`` rows of ``name``."""
    y = torch.as_tensor(y_true).detach().reshape(-1).cpu()
    if y.numel() != n:
        raise ValueError(f"y_true and {name} differ in length: {y.numel()} vs {n}.")
    if y.numel() == 0:
        raise ValueError("y_true is empty.")
    if y.is_floating_point() and not torch.equal(y, y.round()):
        raise ValueError("y_true must hold integer class labels 0 … K−1.")
    y = y.long()
    if y.min() < 0 or y.max() >= k:
        raise ValueError(f"y_true must hold class labels 0 … {k - 1}.")
    return y


def _multiclass_pair(y_true: object, prob: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer labels ``(n,)`` and float64 probabilities ``(n, K)`` on the CPU, validated."""
    p = torch.as_tensor(prob).detach().cpu().to(torch.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f"prob must have shape (n, K) with K ≥ 2; got {tuple(p.shape)}.")
    y = _class_labels(y_true, "prob", p.shape[0], p.shape[1])
    if torch.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError("prob must hold probabilities in [0, 1].")
    if not torch.allclose(p.sum(1), torch.ones(p.shape[0], dtype=torch.float64), atol=1e-6):
        raise ValueError("each row of prob must sum to 1.")
    return y, p


def multiclass_brier_score(y_true: object, prob: object) -> float:
    """``mean_i Σ_k (p_ik − onehot(y_i)_k)²``; see the module docstring."""
    y, p = _multiclass_pair(y_true, prob)
    onehot = F.one_hot(y, p.shape[1]).to(torch.float64)
    return float(((p - onehot) ** 2).sum(1).mean())


def top_label_ece(
    y_true: object, prob: object, n_bins: int = 10, strategy: BinStrategy = "uniform"
) -> float:
    """ECE of the top-class confidence against top-class correctness."""
    y, p = _multiclass_pair(y_true, prob)
    _validate_bins(n_bins, strategy)
    confidence, predicted = p.max(1)
    return _ece((predicted == y).to(torch.float64), confidence, n_bins, strategy)


def classwise_ece(
    y_true: object, prob: object, n_bins: int = 10, strategy: BinStrategy = "uniform"
) -> float:
    """Mean over classes of the one-vs-rest ECE of each class's probability."""
    y, p = _multiclass_pair(y_true, prob)
    _validate_bins(n_bins, strategy)
    per_class = [
        _ece((y == k).to(torch.float64), p[:, k], n_bins, strategy) for k in range(p.shape[1])
    ]
    return float(sum(per_class) / len(per_class))


def _logit_pair(logits: object, y_true: object) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.as_tensor(logits).detach().reshape(-1).cpu().to(torch.float64)
    if not torch.isfinite(z).all():
        raise ValueError("logits must be finite.")
    y, _ = _pair(y_true, torch.sigmoid(z))
    if torch.unique(y).numel() < 2:
        raise ValueError("fitting a calibration needs both classes in y_true.")
    return z, y


def _minimise_nll(params: torch.Tensor, nll: Callable[[], torch.Tensor]) -> None:
    """Minimise ``nll()`` over ``params`` in place, by L-BFGS."""
    optimiser = torch.optim.LBFGS(
        [params],
        lr=1.0,
        max_iter=500,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    optimiser.step(closure)


@dataclass(frozen=True)
class TemperatureScaler:
    """``σ(z / temperature)``; see the module docstring."""

    temperature: float

    @classmethod
    def fit(cls, logits: object, y_true: object) -> TemperatureScaler:
        """
        The temperature minimising the validation NLL (L-BFGS on ``log T``, so ``T > 0``).

        The NLL is convex in ``s = 1/T`` with slope ``−Σ m_i σ(−s·m_i)``, where
        ``m_i = (2y_i − 1)·z_i`` is the signed margin.  At ``s → 0`` the slope is
        ``−Σ m_i / 2`` and at ``s → ∞`` it is ``−Σ_{m_i<0} m_i``, so a finite
        ``T`` exists only if ``Σ m_i > 0`` (the logits rank better than chance)
        and some ``m_i < 0`` (the classes are not separated at 0).  Otherwise the
        optimum is ``T → ∞`` or ``T → 0`` and fitting raises ``ValueError``.
        """
        z, y = _logit_pair(logits, y_true)
        margin = (2 * y - 1) * z
        if margin.sum() <= 0:
            raise ValueError(
                "the logits rank no better than chance (sum of signed margins <= 0), so the "
                "NLL falls as the temperature grows without bound; there is no finite fit."
            )
        if not (margin < 0).any():
            raise ValueError(
                "the logits separate the classes at 0, so the NLL falls as the temperature "
                "shrinks to 0; there is no finite fit."
            )
        log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        _minimise_nll(log_t, lambda: F.binary_cross_entropy_with_logits(z / log_t.exp(), y))
        return cls(float(log_t.detach().exp()))

    def __call__(self, logits: object) -> torch.Tensor:
        """Calibrated positive-class probabilities, ``float64``."""
        z = torch.as_tensor(logits).detach().to(torch.float64)
        return torch.sigmoid(z / self.temperature)


@dataclass(frozen=True)
class PlattScaler:
    """``σ(a·z + b)``; see the module docstring."""

    a: float
    b: float

    @classmethod
    def fit(cls, logits: object, y_true: object) -> PlattScaler:
        """
        ``(a, b)`` minimising the validation NLL: logistic regression on the logit.

        Its maximum-likelihood estimate is finite exactly when no threshold on
        ``z`` separates the classes, ties allowed (quasi-complete separation);
        otherwise ``|a| → ∞`` and fitting raises ``ValueError``.
        """
        z, y = _logit_pair(logits, y_true)
        neg, pos = z[y == 0], z[y == 1]
        if neg.max() <= pos.min() or pos.max() <= neg.min():
            raise ValueError(
                "a threshold on the logits separates the classes, so the NLL falls as |a| "
                "grows without bound; there is no finite fit."
            )
        ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
        _minimise_nll(ab, lambda: F.binary_cross_entropy_with_logits(ab[0] * z + ab[1], y))
        a, b = ab.detach().tolist()
        return cls(float(a), float(b))

    def __call__(self, logits: object) -> torch.Tensor:
        """Calibrated positive-class probabilities, ``float64``."""
        z = torch.as_tensor(logits).detach().to(torch.float64)
        return torch.sigmoid(self.a * z + self.b)


@dataclass(frozen=True)
class MulticlassTemperatureScaler:
    """``softmax(z / temperature)`` for ``(n, K)`` logits; see the module docstring."""

    temperature: float

    @classmethod
    def fit(cls, logits: object, y_true: object) -> MulticlassTemperatureScaler:
        """
        The temperature minimising the validation cross-entropy (L-BFGS on ``log T``).

        The cross-entropy is convex in ``s = 1/T`` with slope
        ``Σ_i (E_i[z] − z_iy)``, where ``z_iy`` is the true-class logit of
        sample ``i`` and ``E_i`` the mean of its logits under ``softmax(s·z_i)``.
        At ``s → 0`` that mean is the plain row mean and at ``s → ∞`` the row
        maximum, so a finite ``T`` exists only if ``Σ_i (z_iy − mean_k z_ik) > 0``
        (the logits rank the true class better than chance) and some
        ``z_iy < max_k z_ik`` (not every true class has the top logit).
        Otherwise the optimum is ``T → ∞`` or ``T → 0`` and fitting raises
        ``ValueError``.
        """
        z = torch.as_tensor(logits).detach().cpu().to(torch.float64)
        if z.ndim != 2 or z.shape[1] < 2:
            raise ValueError(f"logits must have shape (n, K) with K ≥ 2; got {tuple(z.shape)}.")
        if not torch.isfinite(z).all():
            raise ValueError("logits must be finite.")
        y = _class_labels(y_true, "logits", z.shape[0], z.shape[1])
        if torch.unique(y).numel() < 2:
            raise ValueError("fitting a calibration needs at least two classes in y_true.")
        true = z.gather(1, y[:, None]).squeeze(1)
        if (true - z.mean(1)).sum() <= 0:
            raise ValueError(
                "the logits rank the true class no better than chance (sum of true-class "
                "logits minus row means <= 0), so the NLL falls as the temperature grows "
                "without bound; there is no finite fit."
            )
        if not (true < z.max(1).values).any():
            raise ValueError(
                "every true class has the top logit, so the NLL falls as the temperature "
                "shrinks to 0; there is no finite fit."
            )
        log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        _minimise_nll(log_t, lambda: F.cross_entropy(z / log_t.exp(), y))
        return cls(float(log_t.detach().exp()))

    def __call__(self, logits: object) -> torch.Tensor:
        """Calibrated class probabilities ``(n, K)``, ``float64``."""
        z = torch.as_tensor(logits).detach().to(torch.float64)
        if z.ndim != 2:
            raise ValueError(f"logits must have shape (n, K); got {tuple(z.shape)}.")
        return torch.softmax(z / self.temperature, dim=-1)
