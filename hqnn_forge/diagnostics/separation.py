"""
hqnn_forge.diagnostics.separation
=================================
How well a fixed feature matrix separates inputs, and classes, before any
head is trained on it.

A classification score does not say why a fixed feature map helps or fails.
These measures are properties of the features themselves: of a matrix ``F``
with one row per sample and one column per feature, and binary labels ``y``.
Nothing here is specific to one feature map.  They were written for the
readout of :class:`~hqnn_forge.rydberg.RydbergFeatureMap`, which is
``n_atoms`` numbers per sample and can lose most of what the state
distinguishes; which statement each measure supports there, and how they
relate to the state kernel
(:func:`~hqnn_forge.kernels.kernel_from_density_matrices`), is in
``docs/rydberg-model.md``.

Conventions
-----------
* **Rows are samples, columns are features.**  ``F`` has shape ``(M, d)``;
  ``F[i]`` is the feature vector of sample ``i``.  The features carry
  whatever unit the map gives them (excitation probabilities are
  dimensionless); distances are in that unit, the other measures are
  dimensionless.
* **Labels** are ``{0, 1}`` or ``{−1, +1}`` with both classes present, as
  for :func:`~hqnn_forge.kernels.kernel_target_alignment`.  Class 1 is the
  larger value; ``n0`` and ``n1`` are the class sizes, ``μ0`` and ``μ1`` the
  class means (vectors of length ``d``).  No measure changes when the two
  classes are swapped.
* **Pairs** of samples are ``(i, j)`` with ``i < j``, in the order
  ``(0,1), (0,2), …, (0,M−1), (1,2), …, (M−2,M−1)``.
* **Undefined is not a number.**  A measure that is ``0/0`` on the given
  data is returned as NaN and one whose defining supremum is unbounded as
  ``inf``; neither is replaced by a finite value.  The one exception is
  stated under "Effective rank".

Pairwise distances
------------------
The Euclidean distance of every pair::

    D_ij = |F[i] − F[j]| = √( Σ_k (F[i,k] − F[j,k])² ),        i < j

split into the ``n0(n0−1)/2 + n1(n1−1)/2`` pairs within a class
(``y_i = y_j``) and the ``n0 · n1`` pairs between the classes.  The two sets
of distances are the distribution; their means and the ratio ::

    R = mean of D over the pairs between the classes
        ───────────────────────────────────────────
        mean of D over the pairs within a class

summarise it.  Each mean is over pairs, so within a class the larger class
has the larger weight (it has more pairs).

* **Range.**  ``D ≥ 0`` and ``R ∈ (0, ∞]``.  ``R = ∞`` when each class is a
  single point and the two points differ.  ``R`` is undefined (NaN) when
  there is no pair within a class, or when every row is the same: all
  distances are then 0, and a between-class mean of 0 occurs in no other
  case.
* **No information.**  If the labels are independent of the features, a pair
  is within a class or between the classes whatever its distance, so both
  means estimate the same mean distance and ``R ≈ 1``.  ``R`` below 1 is
  possible: members of a class can be further apart than the classes.
* **Scale.**  The distances scale with the features; ``R`` does not.

Fisher discriminant ratio
-------------------------
Project the features on a direction ``w`` (a vector of length ``d``).  The
projected class means differ by ``wᵀ δ`` with ``δ = μ1 − μ0``, and the
pooled variance of the projection within the classes is ``wᵀ S_w w`` with
the pooled within-class covariance ::

    S_w = ( Σ_{i: y_i = 0} (F[i] − μ0)(F[i] − μ0)ᵀ + Σ_{i: y_i = 1} (F[i] − μ1)(F[i] − μ1)ᵀ ) / (M − 2)

(``M − 2`` because two means were estimated).  Fisher's ratio for that
direction is the squared difference of the means over the variance, and the
measure is its value along the best direction::

    J(w) = (wᵀ δ)² / (wᵀ S_w w),        J = sup_w J(w)

By the Cauchy–Schwarz inequality in the inner product ``⟨a, b⟩ = aᵀ S_w b``,
for an invertible ``S_w``::

    (wᵀ δ)² = ⟨w, S_w⁻¹ δ⟩² ≤ ⟨w, w⟩ · ⟨S_w⁻¹ δ, S_w⁻¹ δ⟩ = (wᵀ S_w w) · (δᵀ S_w⁻¹ δ)

with equality for ``w ∝ S_w⁻¹ δ``, so ::

    J = δᵀ S_w⁻¹ δ

the squared Mahalanobis distance between the class means.  With the singular
value decomposition ``F_w = U diag(s) Vᵀ`` of the features centred class by
class (row ``i`` of ``F_w`` is ``F[i] − μ_{y_i}``), ``S_w = V diag(s²) Vᵀ /
(M − 2)`` and ``J = (M − 2) Σ_a (v_aᵀ δ)² / s_a²``, which is how it is
computed.

* **Range.**  ``J ∈ [0, ∞]``, dimensionless, and unchanged by any invertible
  affine map of the features (``F → F Aᵀ + b`` maps ``δ → A δ`` and
  ``S_w → A S_w Aᵀ``).  ``J = 0`` when the class means coincide.
* **Singular** ``S_w``.  A direction ``v_a`` without within-class spread
  (``s_a = 0``) has ``J(v_a) = (v_aᵀ δ)²/0``: if the means differ along it
  the classes are separated perfectly and ``J = ∞``; if they do not, the
  direction says nothing and is left out of the sum.  ``J`` is infinite in
  particular for features equal to the labels, and whenever ``M − 2 < d``
  and the means differ outside the span of the within-class differences,
  which is the rule with fewer samples than features.  If every row is the
  same, no direction has a ratio and ``J`` is NaN.
* **Reading.**  For two Gaussian classes with a common covariance, the best
  linear rule at equal priors errs with probability ``Φ(−√J / 2)``; for
  other distributions ``J`` gives no error rate.
* **Sample size.**  ``δ`` and ``S_w`` estimate the difference of the
  population means and the within-class covariance whatever ``M`` is, so
  ``J`` estimates one number, ``Δ² = δᵀ Σ⁻¹ δ`` of the populations, and
  neither grows nor shrinks in proportion to ``M``.  It does overestimate
  ``Δ²`` on a finite sample.  ``n0 n1 J / M`` is Hotelling's ``T²``, and for
  two Gaussian classes with a common covariance ``T² (M − d − 1)/((M − 2) d)``
  has a noncentral F distribution with ``d`` and ``M − d − 1`` degrees of
  freedom and noncentrality ``n0 n1 Δ² / M``, whose mean gives, for
  ``M > d + 3``::

      E[J] = (M − 2)/(M − d − 3) · ( Δ² + d · M/(n0 n1) )

  For classes that do not differ at all (``Δ² = 0``) ``J`` is therefore
  about ``d · M/(n0 n1)`` on average, not 0: ``4 d / M`` for classes of equal
  size.  Values of ``J`` from samples of different size are comparable only
  where they are large against that level.
* **Class imbalance.**  Swapping the classes changes nothing, and for a
  common covariance the population value ``Δ²`` does not depend on the class
  sizes.  Two things do.  The level above is ``d · M/(n0 n1)``, about ``d``
  over the size of the smaller class when the other is much larger, so a
  rare class raises ``J`` without any class difference behind it.  And
  ``S_w`` weights the covariance of each class by its size (by ``n_k − 1``):
  if the classes have different covariances ``Σ0`` and ``Σ1``, ``S_w`` has
  the mean ``((n0 − 1) Σ0 + (n1 − 1) Σ1)/(M − 2)``, in which the larger
  class dominates, so ``J`` changes with the class proportions for the same
  two populations.
* **Which normalisation.**  ``S_w`` is the pooled covariance: the
  within-class sums of squares and products over their ``M − 2`` degrees of
  freedom, as in the two-sample ``T²``.  Two other conventions carry the
  same name and give other numbers.  With the sums themselves in the
  denominator (Fisher 1936, and ``S_W`` in most textbooks) the ratio is
  ``J/(M − 2)``: the best direction is the same, and the value falls as
  ``1/M`` for the same populations.  With the sum of the two class
  covariances, ``S_0 + S_1`` (each over ``n_k − 1``), the ratio is ``J/2``
  when the two classes have the same size, and otherwise the classes are
  weighted equally instead of by size, which in general also changes the
  best direction.  Class 0 ``{−1, 1}`` and class 1 ``{2, 4, 6}`` (``δ = 4``,
  sums of squares 2 and 8): ``J = 16/(10/3) = 4.8`` here, against
  ``16/10 = 1.6`` and ``16/(2 + 4) = 2.67``.

Effective rank
--------------
With the eigenvalues ``λ_1, …, λ_d ≥ 0`` of the covariance of the features
(the rows centred on their mean; its normalisation cancels)::

    p_a = λ_a / Σ_b λ_b,        erank = exp( − Σ_a p_a ln p_a ),        0 ln 0 = 0

the exponential of the entropy of the normalised spectrum (Roy & Vetterli
2007, applied to the covariance).  The eigenvalues are the squared singular
values of the centred features, which is how they are computed.

* **Range.**  ``1 ≤ erank ≤ min(d, M − 1)``: 1 when all variance lies along
  one direction, ``k`` when it is spread evenly over ``k`` directions.  It
  is unchanged by a translation, a rotation and a common scale of the
  features, but not by rescaling one feature against another.
* **Constant features.**  If every row is the same the covariance is zero
  and has no normalised spectrum.  The function then returns 1.0, the lower
  end of the range, so that a collapse onto a point does not read as a
  higher rank than a collapse onto a line.  This is the one value in this
  module that is set by convention and not by the definition.

Linear feature kernel
---------------------
::

    K = F Fᵀ,        K[i, j] = Σ_k F[i,k] F[j,k]

an ``M × M`` Gram matrix over the samples, symmetric and positive
semi-definite.  Passed to :func:`~hqnn_forge.kernels.kernel_target_alignment`
it gives the centred alignment of the readout with the labels, to compare
with that of the state kernel.  With ``H = 1 − 11ᵀ/M``, the centred features
``F_c = H F`` and the centred labels ``ỹ = H y`` (``y = ±1``)::

    A = ⟨H K H, ỹ ỹᵀ⟩ / ( |H K H| · |ỹ ỹᵀ| ) = |F_cᵀ ỹ|² / ( |F_c F_cᵀ| · |ỹ|² )

in the Frobenius inner product and norm.

* **Range.**  ``A ∈ [0, 1]`` for this kernel: the numerator is a squared
  norm.  ``A = 1`` exactly when every feature is an affine function of the
  label (``F_c = ỹ vᵀ``), as for features equal to the labels; ``A = 0``
  when no feature is correlated with the label.  If every row is the same,
  ``H K H = 0``, for which ``kernel_target_alignment`` returns 0.
* ``A`` is unchanged by a translation and a common scale of the features.

Rounding, and features that have collapsed
------------------------------------------
Every measure is unchanged when the same vector is added to every row, and
:func:`separation_measures`, :func:`fisher_discriminant_ratio` and
:func:`effective_rank` use that: they subtract the first row from all rows
before anything is squared or averaged.
Rows that are equal then give exact zeros, so the constant case above is
detected exactly, and features of the form ``c + ε g`` with a small ``ε``
(excitation probabilities near ½ under strong dephasing) keep the relative
precision of ``ε g``.  The distances are computed from differences, not from
squared norms, for the same reason.

The ratios, the rank and the alignment do not depend on the scale of the
features, so **a collapse onto a point does not by itself move them towards
a no-information value**: features ``c + ε g`` have the ratios, rank and
alignment of ``g`` for every ``ε > 0``, down to where ``ε g`` is rounding
error.  What shows the collapse is the distances; the other measures change
only if ``g`` does.  Whether a small difference is resolvable is a question
about the measurement (the number of shots), not one these measures answer.

Costs
-----
The distances and the kernel hold ``M²`` numbers; the Fisher ratio and the
rank need one singular value decomposition of an ``M × d`` matrix.  All
results are ``float64`` on the CPU and carry no gradient.

References
----------
* Fisher (1936) "The use of multiple measurements in taxonomic problems",
  Annals of Eugenics 7, 179–188.
* Roy & Vetterli (2007) "The effective rank: a measure of effective
  dimensionality", 15th European Signal Processing Conference, 606–610.
* Cortes, Mohri & Rostamizadeh (2012) "Algorithms for learning kernels based
  on centered alignment", JMLR 13, 795–828.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hqnn_forge.kernels import kernel_target_alignment

__all__ = [
    "PairwiseDistances",
    "SeparationMeasures",
    "effective_rank",
    "fisher_discriminant_ratio",
    "linear_feature_kernel",
    "pairwise_distances",
    "separation_measures",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _features(F: Any) -> torch.Tensor:
    """
    ``F`` as a finite ``float64`` tensor of shape ``(M, d)`` on the CPU, detached.

    Anything but a tensor goes through NumPy first: ``torch.as_tensor`` reads
    Python floats as ``float32``, which would round the features to 7 digits.
    """
    tensor = F if isinstance(F, torch.Tensor) else torch.as_tensor(np.asarray(F))
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"F must hold real numbers; got dtype {tensor.dtype}.")
    tensor = tensor.detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(
            f"F must have shape (n_samples, n_features), one row per sample, with at least "
            f"one of each; got shape {tuple(tensor.shape)}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("F must be finite; got a NaN or an infinity.")
    return tensor


def _classes(y: Any, n_samples: int) -> torch.Tensor:
    """Boolean mask of class 1, from labels checked as ``kernel_target_alignment`` checks them."""
    labels = y if isinstance(y, torch.Tensor) else torch.as_tensor(np.asarray(y))
    labels = labels.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if labels.numel() != n_samples:
        raise ValueError(f"y must have {n_samples} labels; got {labels.numel()}.")
    values = set(labels.unique().tolist())
    if values not in ({0.0, 1.0}, {-1.0, 1.0}):
        raise ValueError(f"y must hold both classes as 0/1 or -1/+1; got values {sorted(values)}.")
    return labels > 0


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` for non-negative floats: ``x/0 = inf``, ``0/0 = NaN``."""
    if math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    if denominator == 0.0:
        return math.nan if numerator == 0.0 else math.inf
    return numerator / denominator


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PairwiseDistances:
    """
    The Euclidean distances of every pair of samples, split by class membership.

    Two results are equal when both sets of distances are, element by
    element; a result is not hashable.

    Attributes
    ----------
    within:
        ``D_ij`` of the pairs ``i < j`` with ``y_i = y_j``, ``float64``,
        length ``n0(n0−1)/2 + n1(n1−1)/2``, in the pair order
        ``(0,1), (0,2), …, (M−2,M−1)`` with the other pairs left out.
    between:
        ``D_ij`` of the pairs with ``y_i ≠ y_j``, length ``n0 · n1``, in the
        same order.
    """

    within: torch.Tensor
    between: torch.Tensor

    # Written out: the generated __eq__ would call bool() on an element-wise
    # tensor comparison and raise, and with both fields left out of it (the
    # only fields there are) any two results would compare equal.  Defining
    # __eq__ without __hash__ makes the class unhashable, as tensors compared
    # by value have to be.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PairwiseDistances):
            return NotImplemented
        return torch.equal(self.within, other.within) and torch.equal(self.between, other.between)

    @property
    def mean_within(self) -> float:
        """Mean distance within a class, over pairs; NaN if no class has two samples."""
        return float(self.within.mean()) if self.within.numel() else math.nan

    @property
    def mean_between(self) -> float:
        """Mean distance between the classes, over pairs; 0 only if every row is the same."""
        return float(self.between.mean())

    @property
    def ratio(self) -> float:
        """
        ``mean_between / mean_within``, in ``(0, ∞]``; about 1 for labels
        that are independent of the features.  ``inf`` if each class is one
        point and the points differ; NaN if all distances are 0 or no class
        has two samples.
        """
        return _ratio(self.mean_between, self.mean_within)


@dataclass(frozen=True)
class SeparationMeasures:
    """
    The scalar separation measures of one feature matrix; see the module docstring.

    Attributes
    ----------
    n_samples, n_features:
        Shape ``(M, d)`` of the feature matrix.
    mean_within_distance, mean_between_distance:
        Means of the Euclidean distances over the pairs within a class and
        between the classes, in the unit of the features.
    distance_ratio:
        ``mean_between_distance / mean_within_distance``, in ``(0, ∞]``;
        about 1 without class information.
    fisher_ratio:
        Fisher discriminant ratio along the best linear direction,
        ``δᵀ S_w⁻¹ δ`` with the pooled within-class covariance (denominator
        ``n_samples − 2``), in ``[0, ∞]``.
    effective_rank:
        Exponential of the entropy of the normalised eigenvalues of the
        feature covariance, in ``[1, min(d, M − 1)]``.
    kernel_alignment:
        Centred alignment of the linear feature kernel ``F Fᵀ`` with the
        labels, in ``[0, 1]``.

    The two ratios can be ``inf`` (perfect separation) or NaN (undefined, as
    for identical rows), which strict JSON has no value for.
    """

    n_samples: int
    n_features: int
    mean_within_distance: float
    mean_between_distance: float
    distance_ratio: float
    fisher_ratio: float
    effective_rank: float
    kernel_alignment: float

    def to_dict(self) -> dict[str, Any]:
        """Every field as a plain ``int`` or ``float``, in the order above, for logging."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "mean_within_distance": self.mean_within_distance,
            "mean_between_distance": self.mean_between_distance,
            "distance_ratio": self.distance_ratio,
            "fisher_ratio": self.fisher_ratio,
            "effective_rank": self.effective_rank,
            "kernel_alignment": self.kernel_alignment,
        }


# ---------------------------------------------------------------------------
# The measures, on validated tensors
# ---------------------------------------------------------------------------


def _distances(features: torch.Tensor, positive: torch.Tensor) -> PairwiseDistances:
    n_samples = features.shape[0]
    # From the differences, not from |a|² + |b|² − 2 a·b: that form loses the
    # distance of nearly equal rows to cancellation and is not exactly 0 for
    # equal ones.
    distances = torch.cdist(features, features, compute_mode="donot_use_mm_for_euclid_dist")
    rows, columns = torch.triu_indices(n_samples, n_samples, offset=1)
    values = distances[rows, columns]
    same = positive[rows] == positive[columns]
    return PairwiseDistances(within=values[same], between=values[~same])


def _fisher(features: torch.Tensor, positive: torch.Tensor) -> float:
    n_samples, n_features = features.shape
    if n_samples < 3:
        raise ValueError(
            f"the Fisher discriminant ratio needs at least 3 samples: the pooled covariance "
            f"divides by n_samples - 2; got {n_samples}."
        )
    shifted = features - features[0]
    mean0 = shifted[~positive].mean(dim=0)
    mean1 = shifted[positive].mean(dim=0)
    difference = mean1 - mean0  # δ = μ1 − μ0
    centred = shifted - torch.where(positive.unsqueeze(1), mean1, mean0)  # F_w
    # |shifted|² = |F_w|² + n0 |mean0|² + n1 |mean1|²: zero exactly when every
    # row equals the first, and otherwise the size against which a spread or a
    # mean difference counts as rounding error.
    scale = float(torch.linalg.norm(shifted))
    if scale == 0.0:
        return math.nan
    tolerance = max(n_samples, n_features) * torch.finfo(torch.float64).eps * scale
    _, singular, vh = torch.linalg.svd(centred, full_matrices=False)
    spread = singular > tolerance
    components = vh @ difference  # v_aᵀ δ
    # What is left of δ outside the directions with within-class spread
    # (including, for M < d, the directions the decomposition does not list).
    outside = difference - vh[spread].T @ components[spread]
    if float(torch.linalg.norm(outside)) > tolerance:
        return math.inf
    return float((n_samples - 2) * ((components[spread] / singular[spread]) ** 2).sum())


def _effective_rank(features: torch.Tensor) -> float:
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError(
            f"the effective rank needs at least 2 samples for a covariance; got {n_samples}."
        )
    shifted = features - features[0]
    centred = shifted - shifted.mean(dim=0)
    eigenvalues = torch.linalg.svdvals(centred) ** 2  # of the scatter matrix F_cᵀ F_c
    total = eigenvalues.sum()
    if float(total) == 0.0:
        return 1.0
    p = eigenvalues / total
    p = p[p > 0]  # 0 ln 0 = 0
    return float(torch.exp(-(p * torch.log(p)).sum()))


def _gram(features: torch.Tensor) -> torch.Tensor:
    kernel = features @ features.T
    return torch.triu(kernel) + torch.triu(kernel, 1).T  # exactly symmetric


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def pairwise_distances(F: Any, y: Any) -> PairwiseDistances:
    """
    The distribution of pairwise Euclidean feature distances, within and between classes.

    ``D_ij = |F[i] − F[j]|`` for every pair ``i < j``, split into the pairs
    with equal and with different labels; see "Pairwise distances" in the
    module docstring.

    Parameters
    ----------
    F:
        Features of shape ``(n_samples, n_features)``, any real-valued
        array-like, one row per sample.  Converted to ``float64``.
    y:
        Binary labels, ``{0, 1}`` or ``{-1, +1}``, one per row, with both
        classes present.

    Returns
    -------
    PairwiseDistances
        The two sets of distances, with their means and the ratio of the
        between-class to the within-class mean as properties.

    Raises
    ------
    TypeError
        If ``F`` is complex or boolean.
    ValueError
        If ``F`` is not a matrix with at least one row and column or holds a
        NaN or an infinity, or ``y`` has another length than ``F`` has rows
        or does not hold both classes as 0/1 or -1/+1.

    Notes
    -----
    The full ``n_samples × n_samples`` distance matrix is formed: 8 bytes per
    entry, 200 MB at 5000 samples.

    Examples
    --------
    >>> from hqnn_forge.diagnostics import pairwise_distances
    >>> result = pairwise_distances([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0], [6.0, 8.0]], [0, 1, 0, 1])
    >>> result.within.tolist()  # the pairs (0, 2) and (1, 3)
    [1.0, 5.0]
    >>> result.mean_within
    3.0
    >>> round(result.ratio, 3)
    2.372
    """
    features = _features(F)
    return _distances(features, _classes(y, features.shape[0]))


def fisher_discriminant_ratio(F: Any, y: Any) -> float:
    """
    Fisher discriminant ratio along the best linear direction: ``δᵀ S_w⁻¹ δ``.

    The largest value, over directions ``w``, of the squared difference of
    the projected class means over the pooled within-class variance of the
    projection.  ``δ = μ1 − μ0`` and ``S_w`` is the pooled within-class
    covariance with denominator ``n_samples − 2``; the result is the squared
    Mahalanobis distance between the class means.  See "Fisher discriminant
    ratio" in the module docstring for the derivation and the conventions.

    Parameters
    ----------
    F, y:
        As for :func:`pairwise_distances`; at least 3 samples.

    Returns
    -------
    float
        In ``[0, ∞]``, dimensionless.  0 when the class means coincide;
        ``inf`` when the means differ along a direction in which neither
        class has any spread (features equal to the labels, or fewer samples
        than ``n_features + 2`` in general position); NaN when every row is
        the same.  It estimates the same population value at every sample
        size, with an upward bias of about ``n_features · M/(n0 n1)`` (larger
        for a small sample and for a rare class), and with unequal class
        covariances it weights the larger class more; see "Sample size" and
        "Class imbalance" in the module docstring.

    Raises
    ------
    TypeError, ValueError
        As :func:`pairwise_distances`; also ``ValueError`` for fewer than 3
        samples.

    Notes
    -----
    A within-class spread or a mean difference along a direction counts as
    zero when it is below ``max(n_samples, n_features) · ε · |F − F[0]|``,
    with ε the ``float64`` machine epsilon and the Frobenius norm of the
    features after the first row is subtracted from every row: the size of
    the rounding error of the centred features.  A repeated or linearly
    dependent feature therefore neither changes the ratio nor makes it
    infinite.

    Examples
    --------
    Class 0 is ``{-1, 1}`` and class 1 ``{3, 5}``: the means differ by 4 and
    the pooled variance is ``(2 + 2)/(4 − 2) = 2``.

    >>> from hqnn_forge.diagnostics import fisher_discriminant_ratio
    >>> round(fisher_discriminant_ratio([[-1.0], [1.0], [3.0], [5.0]], [0, 0, 1, 1]), 12)
    8.0

    Unequal class sizes, class 1 ``{2, 4, 6}``: the means differ by 4 again
    and the pooled variance is ``(2 + 8)/(5 − 2)``.

    >>> round(fisher_discriminant_ratio([[-1.0], [1.0], [2.0], [4.0], [6.0]], [0, 0, 1, 1, 1]), 12)
    4.8
    """
    features = _features(F)
    return _fisher(features, _classes(y, features.shape[0]))


def effective_rank(F: Any) -> float:
    """
    Effective rank of the feature covariance: ``exp(−Σ_a p_a ln p_a)``.

    ``p_a`` are the eigenvalues of the covariance of the features divided by
    their sum.  It shows features collapsing onto fewer directions: 1 when
    all variance lies along one direction, ``k`` when it is spread evenly
    over ``k``.  See "Effective rank" in the module docstring.

    Parameters
    ----------
    F:
        Features of shape ``(n_samples, n_features)``, any real-valued
        array-like, with at least 2 samples.  No labels are used.

    Returns
    -------
    float
        In ``[1, min(n_features, n_samples − 1)]``.  1.0 by convention when
        every row is the same and the covariance is zero.

    Raises
    ------
    TypeError
        If ``F`` is complex or boolean.
    ValueError
        If ``F`` is not a matrix with at least 2 rows and one column, or
        holds a NaN or an infinity.

    Examples
    --------
    The rows ``(±√3, 0), (0, ±1)`` have three quarters of their variance
    along the first feature: ``exp(−¾ ln ¾ − ¼ ln ¼) = 4 / 3^(3/4)``.

    >>> from hqnn_forge.diagnostics import effective_rank
    >>> root = 3**0.5
    >>> round(effective_rank([[root, 0.0], [-root, 0.0], [0.0, 1.0], [0.0, -1.0]]), 6)
    1.754765
    """
    return _effective_rank(_features(F))


def linear_feature_kernel(F: Any) -> torch.Tensor:
    """
    The linear kernel of the features, ``K = F Fᵀ``.

    ``K[i, j] = Σ_k F[i,k] F[j,k]``, the Gram matrix over the samples.  Pass
    it to :func:`~hqnn_forge.kernels.kernel_target_alignment` for the
    alignment of the features with the labels, in ``[0, 1]`` for this kernel
    (see "Linear feature kernel" in the module docstring), next to that of
    the state kernel of the same inputs.

    Parameters
    ----------
    F:
        Features of shape ``(n_samples, n_features)``, any real-valued
        array-like.

    Returns
    -------
    torch.Tensor
        Shape ``(n_samples, n_samples)``, ``float64``, exactly symmetric,
        positive semi-definite to rounding.

    Raises
    ------
    TypeError
        If ``F`` is complex or boolean.
    ValueError
        If ``F`` is not a matrix with at least one row and column, or holds
        a NaN or an infinity.

    Notes
    -----
    The alignment centres the kernel, which subtracts numbers of the size of
    ``|F[i]|²`` from one another.  For features whose rows differ only in
    their last digits (``½ + 1e-12 · g``) the result is that of the rounding
    error; :func:`separation_measures` avoids this by forming the kernel of
    ``F − F[0]``, which has the same centred alignment.

    Examples
    --------
    >>> from hqnn_forge.diagnostics import linear_feature_kernel
    >>> from hqnn_forge.kernels import kernel_target_alignment
    >>> linear_feature_kernel([[1.0, 2.0], [3.0, 4.0]]).tolist()
    [[5.0, 11.0], [11.0, 25.0]]
    >>> labels = [0, 0, 1, 1]
    >>> features = [[0.0], [0.0], [1.0], [1.0]]  # equal to the labels
    >>> round(float(kernel_target_alignment(linear_feature_kernel(features), labels)), 12)
    1.0
    """
    return _gram(_features(F))


def separation_measures(F: Any, y: Any) -> SeparationMeasures:
    """
    Every scalar separation measure of a feature matrix, in one record.

    The means and the ratio of :func:`pairwise_distances`,
    :func:`fisher_discriminant_ratio`, :func:`effective_rank` and the
    alignment of :func:`linear_feature_kernel` with the labels
    (:func:`~hqnn_forge.kernels.kernel_target_alignment`).  None of them
    involves a trained model, and none predicts the score of one.

    Parameters
    ----------
    F, y:
        As for :func:`pairwise_distances`; at least 3 samples.

    Returns
    -------
    SeparationMeasures
        Its ``to_dict()`` gives plain numbers for a log line.

    Raises
    ------
    TypeError, ValueError
        As :func:`fisher_discriminant_ratio`.

    Notes
    -----
    The alignment is that of the Gram matrix of ``F − F[0]`` (the first row
    subtracted from every row).  Centring removes a common offset, so this
    is the centred alignment of ``F Fᵀ``, computed without the cancellation
    described in :func:`linear_feature_kernel`.

    Examples
    --------
    >>> from hqnn_forge.diagnostics import separation_measures
    >>> measures = separation_measures([[-1.0], [1.0], [3.0], [5.0]], [0, 0, 1, 1])
    >>> measures.mean_within_distance, measures.mean_between_distance
    (2.0, 4.0)
    >>> round(measures.fisher_ratio, 12), measures.effective_rank
    (8.0, 1.0)
    >>> round(measures.kernel_alignment, 12)  # 8² / (20 · 4)
    0.8
    """
    features = _features(F)
    n_samples, n_features = features.shape
    positive = _classes(y, n_samples)
    fisher = _fisher(features, positive)  # first: it refuses fewer than 3 samples
    distances = _distances(features, positive)
    alignment = kernel_target_alignment(_gram(features - features[0]), positive.to(torch.float64))
    return SeparationMeasures(
        n_samples=n_samples,
        n_features=n_features,
        mean_within_distance=distances.mean_within,
        mean_between_distance=distances.mean_between,
        distance_ratio=distances.ratio,
        fisher_ratio=fisher,
        effective_rank=_effective_rank(features),
        kernel_alignment=float(alignment),
    )
