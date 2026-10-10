"""
examples/benchmark_rydberg_features.py
======================================
Do the interactions of a Rydberg feature map add anything a head can use?
The benchmark runner behind #501.

``RydbergFeatureMap`` (``docs/rydberg-model.md``) turns each sample into the
excitation probabilities of a small simulated atom array.  This script
trains small classical heads on those features, on the same features with
the interactions switched off, on the inputs themselves, and on a
gate-based feature map, on identical folds, and writes one JSON line per
arm, head, grid point and fold.  It produces records; it draws no
conclusion.

Arms
----
Every arm starts from the same inputs ``x ∈ (−π, π)^N`` (below), ``N`` the
number of atoms.

``A``
    ``RydbergFeatureMap`` with interactions → head.
``B``
    The same map, register, pulse and dephasing with ``interactions=False``
    → the same head.
``C``
    The inputs themselves → the same two heads: the linear one is then a
    logistic regression (with the default loss), and the MLP has the
    parameter count of the MLP head in A, because inputs and features both
    have width ``N``.
``D``
    ``⟨Z_i⟩`` of a frozen angle-embedding circuit of ``N`` qubits → the same
    head: the like-for-like gate-based feature map.  Its rotation angles are
    drawn once, uniformly from ``(−π, π)``, and never trained.
``D-hybrid``
    A ``HybridBinaryClassifier`` (the same circuit shape, trained end to end
    with a linear head on its ``⟨Z_i⟩``).  **Less comparable than every
    other row**: its feature map is trained, the others are fixed.  Its lines
    carry ``"feature_map_trained": true``.

Heads: ``linear`` (:class:`~hqnn_forge.models.LinearClassifier`, ``d + 1``
parameters for ``d`` features) and ``mlp``
(``ClassicalBaseline(d, [hidden])``, ``hidden · (d + 2) + 1`` parameters),
each on A, B, C and D.

What ``A − B`` measures, and what it does not
---------------------------------------------
**Arm B is a classical model.**  With interactions off each feature is a
closed-form function of the input on its own atom: at ``γ = 0``,
``f(√(1 + 3 σ(x_i)²))`` with ``f(s) = sin²(π s/2)/s²``, and with dephasing
the solution of that atom's two equations (``docs/rydberg-model.md``).  The
difference between A and B is therefore **what the interaction-induced
mixing of inputs adds to the features**: in A, feature ``i`` also depends on
the inputs of the other atoms.  It is not a comparison of anything "quantum"
against "classical".  Arm A is itself a classical simulation of at most a
handful of atoms, and no line of the output supports a statement about
computational advantage.

**Parameter counts are recorded, not offered as proof of a fair
comparison.**  The heads of A, B, C and D have the same count by
construction; ``D-hybrid`` has more (its circuit weights), and the fixed
maps differ in what they compute before the first trainable parameter.

One fold, step by step
----------------------
For a dataset ``(X, y)`` and one outer fold of :func:`stratified_kfold`
(training part and test rows):

1. **Validation split.**  The training part is split once more, stratified;
   one ``validation_folds``-th is held out for early stopping and the
   threshold.
2. **Inputs.**  Every raw column is standardised with the mean and standard
   deviation of the training part, and a :class:`PCANormalizer` fitted on
   the training part reduces the result to ``N`` components, each squashed
   to ``(−π, π)`` (``π · tanh`` of the standardised component).  Test rows
   enter neither fit.  **Component ``i`` is input ``i``, and input ``i``
   sets the detuning of atom ``i``**, the atom at position ``(i · a, 0)`` of
   the chain: neighbouring components sit on neighbouring atoms.
3. **Oversampling.**  :func:`smote` on the training rows' inputs.  The
   synthetic rows are made once, in input space, so every arm trains on the
   same samples; validation and test rows are real rows only.
4. **Features.**  Each arm maps those inputs (training rows with the
   synthetic ones, validation rows, test rows) to its features.  A head
   sees the features as the map returns them; nothing is rescaled in
   between.
5. **Tuning** (below), then **training** with :func:`train_model`: Adam,
   the loss of ``--loss``, MCC early stopping on the validation rows.
6. **Scoring.**  The threshold is the one that maximises MCC on the
   validation rows at the best epoch, applied unchanged to the test rows
   for ``mcc``; ``pr_auc`` is threshold-free.

The outer folds, the validation split, the SMOTE draw, the initialisation
seed, the batch order and the tuning seed belong to the fold, not to the
arm: every line of a fold records the same row indices and seeds.

The grid
--------
A grid point is ``(V/Ω, γ/Ω)`` at a fixed pulse area ``ΩT`` (default π);
``--v-over-omega`` and ``--gamma-over-omega`` list the values and the grid is
their product.  The default γ list starts with ``γ = 0``, the noiseless map.

* **``V/Ω`` is set through the spacing.**  The nearest-neighbour interaction
  of a chain at spacing ``a`` is ``V = C6/a⁶``, and the blockade radius
  ``R_b = (C6/Ω)^(1/6)`` is the distance with ``V = Ω``, so

  ::

      V/Ω = (R_b/a)⁶,        a = R_b · (V/Ω)^(−1/6)        [µm]

  (:func:`spacing`).  The full ``1/r⁶`` tail is kept.
* **``γ/Ω``** is the dephasing rate of the collapse operators ``√γ n_i`` in
  units of Ω: ``γ = (γ/Ω) · Ω`` rad/µs.
* **Solver steps.**  With ``γ > 0`` the Lindblad solver needs an explicit
  step count and has no default (#507).  ``--n-steps`` (default 200) is
  passed to every map and recorded in its config.  At ``ΩT = π``, 200 steps
  left the features within ``2.8e-6`` of a reference at ``γ = Ω`` and within
  ``1.0e-4`` at ``γ = 10 Ω`` (#507); the error grows about as ``γ²``, so
  raise the count for larger rates.  With ``γ = 0`` the evolution is exact
  and the count is not used.
* **Arms C and D have no grid.**  They are run once per fold, with ``null``
  in the grid fields.  Arm B does not depend on ``V/Ω`` (its Hamiltonian has
  no interaction term); it is still evaluated at every grid point, with that
  point's register, so every A line has its B line.

Units: ħ = 1, angular frequencies (Ω, γ, ``C6/r⁶``) in rad/µs, lengths in
µm, times in µs; inputs and features are dimensionless.

Measurement
-----------
By default every feature is an exact expectation value
(``"measurement": "exact"``).  With ``--shots S`` the features of A, B and D
are estimated from ``S`` sampled bitstrings per sample
(``"measurement": "shots"``), drawn with the recorded ``shot_seed``.
``D-hybrid`` is always trained and scored on exact expectation values, and
arm C has nothing to measure.

Feature cache
-------------
A fixed map has nothing to fit, so its features are a function of the input
matrix and the map's settings alone.  They are computed once and stored
under ``--cache-dir``, keyed by the SHA-256 of

* the data fingerprint, :func:`hqnn_forge.benchmark.fingerprint` of the
  input matrix handed to the map and its labels, and
* the map's ``get_config()``, with the number of shots and the shot seed.

A later request with the same inputs and settings (another head, another
tuning trial, a repeated run) reads the file; a change in either part of the
key is a miss.  The inputs are fitted per fold (step 2), so each fold has
its own entries: "once" is once per grid point and input matrix.  The
recorded ``feature_seconds`` is the simulation time measured when the
features were computed, also on a cache hit.

Tuning
------
Every arm and head gets the same random-search budget: the same
``--trials`` configurations, drawn once per fold from one search space of
training settings (``lr``, ``batch_size``), each scored by the mean MCC
over the same ``--inner-folds`` stratified folds of the fold's training
part.  Inside an inner fold everything is refitted on its fitting rows
(standardisation, PCA, SMOTE), as ``run_benchmark``'s ``Tuning`` does, and
the outer test rows are never used.  The budget is counted in trials, not
seconds.  The architecture is not tuned.

Data
----
``positive-control`` (:func:`positive_control`)
    A synthetic task whose label is the sign of
    ``Σ_i z_i z_{i+1}``, a sum of products of latent variables that end up,
    in this order, on neighbouring atoms.  No single input carries any
    information about the label, so the class means of arm B's features
    (and of the inputs) coincide, while those of arm A's features do not:
    arm A can in principle beat arm B there.  It tells a null result on real
    data ("interactions do not help") from a pipeline that could not have
    shown an effect.
``iranian-churn``, ``taiwanese-bankruptcy``, ``cervical-cancer-risk``
    The UCI loaders of :mod:`hqnn_forge.data`, read from ``--data-path``
    (``--download`` fetches a missing file).

What a line records
-------------------
``arm``, ``head``, ``representation``, ``feature_map_trained``; the grid
point (``v_over_omega``, ``gamma_over_omega``, ``spacing_um``, ``omega``,
``omega_t``, ``evolution_time_us``, ``n_steps``) and the map's
``get_config()`` (``feature_map``); ``measurement`` and ``shots``; the fold
(``fold``, ``n_splits``, ``train_idx``, ``val_idx``, ``test_idx`` as sorted
row indices into the dataset, ``n_synthetic``); every seed (``seeds``);
``feature_dim``, ``n_parameters`` (trainable) and ``n_frozen_parameters``;
the model's class and config; the chosen ``hyperparameters`` and the
``tuning`` candidates with their scores; ``mcc``, ``pr_auc``, ``threshold``
and ``epochs``; ``feature_seconds`` (simulating the features of the fold's
rows), ``train_seconds`` (the final fit) and ``tuning_seconds``;
``separation``, the measures of
:func:`hqnn_forge.diagnostics.separation_measures` on the matrix the head
receives, for the real rows of the training part (``train``) and for the
test rows (``test``); the dataset's size and SHA-256; ``settings`` and
``environment``.

**The file is strict JSON.**  A NaN is written as ``null`` (as
:mod:`hqnn_forge.experiment` does) and an infinity as the string ``"inf"``
or ``"-inf"``; both occur in ``separation``, where a ratio is ``inf`` for
perfectly separated features and NaN when it is ``0/0``.
:func:`read_records` turns them back into floats there.

Usage
-----
::

    python examples/benchmark_rydberg_features.py --quick
    python examples/benchmark_rydberg_features.py \\
        --dataset positive-control iranian-churn --data-path data/raw --download

``--quick`` runs one fold of one small grid point on a small positive
control, for a smoke test.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from hqnn_forge.benchmark import fingerprint
from hqnn_forge.data import (
    load_cervical_cancer_risk,
    load_iranian_churn,
    load_taiwanese_bankruptcy,
)
from hqnn_forge.diagnostics import separation_measures
from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.evaluation import matthews_corrcoef, pr_auc
from hqnn_forge.experiment import _plain, environment
from hqnn_forge.models import (
    BinaryClassifierBase,
    ClassicalBaseline,
    HybridBinaryClassifier,
    LinearClassifier,
)
from hqnn_forge.preprocessing import PCANormalizer, smote, stratified_kfold
from hqnn_forge.rydberg import DEFAULT_C6, AtomRegister, PulseEncoding, RydbergFeatureMap
from hqnn_forge.training import train_model
from hqnn_forge.utils import FocalLoss

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
RowArray = npt.NDArray[np.intp]

#: Arms that can be selected with ``--arms``; ``D`` includes the ``D-hybrid`` row.
ARMS: tuple[str, ...] = ("A", "B", "C", "D")
#: The two heads of every fixed representation.
HEADS: tuple[str, ...] = ("linear", "mlp")
#: Datasets by name; the position seeds the dataset's folds, so it is fixed.
DATASETS: tuple[str, ...] = (
    "positive-control",
    "iranian-churn",
    "taiwanese-bankruptcy",
    "cervical-cancer-risk",
)
#: Losses by name.  ``bce`` makes the linear head on the inputs a logistic regression.
LOSSES: dict[str, Callable[[], nn.Module]] = {"bce": nn.BCEWithLogitsLoss, "focal": FocalLoss}
#: Training settings the shared search space may vary, as in ``hqnn_forge.benchmark``.
TUNABLE: frozenset[str] = frozenset({"lr", "batch_size", "max_epochs", "patience"})
#: Rows a frozen circuit evaluates at a time.
_CIRCUIT_CHUNK = 1024
#: Second entry of the seed sequence of the seeds that belong to the run, not to a fold.
_RUN_STREAM = 1 << 16


@dataclass(frozen=True)
class Settings:
    """
    Every setting of a run; recorded in each line as ``settings``.

    Attributes
    ----------
    n_atoms:
        Atoms ``N`` of the chain: the number of PCA components, of inputs
        and of qubits of arm D.
    omega:
        Rabi frequency Ω in rad/µs.  Default: the reference ``4π``.
    omega_t:
        Pulse area ``ΩT``, the same at every grid point.  Default π.
    c6:
        ``C6`` in rad/µs · µm⁶.
    v_over_omega, gamma_over_omega:
        The values whose product is the grid; ``V/Ω > 0``, ``γ/Ω >= 0``.
    n_steps:
        Solver steps of every map with ``γ > 0``.
    shots:
        ``None`` for exact features, or the bitstrings per sample.
    arms:
        A subset of :data:`ARMS`.
    hidden:
        Hidden width of the MLP head.
    circuit_layers:
        Variational layers of arm D's circuit.
    n_splits, max_folds:
        Outer folds, and how many of them are run (``None``: all).
    validation_folds:
        One ``validation_folds``-th of a training part is the validation split.
    loss:
        A key of :data:`LOSSES`.
    lr, max_epochs, batch_size, patience:
        Adam learning rate and the ``train_model`` settings, unless tuned.
    n_trials, inner_folds, search_space:
        The tuning budget (0 trials: no tuning) and its search space.
    n_samples:
        Samples of the positive control.
    random_state:
        Root seed of every split, draw, initialisation and batch order.
    """

    n_atoms: int = 4
    omega: float = 4.0 * math.pi
    omega_t: float = math.pi
    c6: float = DEFAULT_C6
    v_over_omega: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    gamma_over_omega: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)
    n_steps: int = 200
    shots: int | None = None
    arms: tuple[str, ...] = ARMS
    hidden: int = 8
    circuit_layers: int = 2
    n_splits: int = 10
    max_folds: int | None = None
    validation_folds: int = 5
    loss: str = "bce"
    lr: float = 0.01
    max_epochs: int = 100
    batch_size: int = 64
    patience: int | None = 10
    n_trials: int = 4
    inner_folds: int = 3
    search_space: Mapping[str, Sequence[Any]] = field(
        default_factory=lambda: {"lr": (0.003, 0.01, 0.03), "batch_size": (32, 128)}
    )
    n_samples: int = 600
    random_state: int = 0

    def __post_init__(self) -> None:
        if not self.v_over_omega or min(self.v_over_omega) <= 0:
            raise ValueError(f"v_over_omega needs values > 0; got {self.v_over_omega}.")
        if not self.gamma_over_omega or min(self.gamma_over_omega) < 0:
            raise ValueError(f"gamma_over_omega needs values >= 0; got {self.gamma_over_omega}.")
        unknown = sorted(set(self.arms) - set(ARMS))
        if unknown or not self.arms:
            raise ValueError(f"arms must be a non-empty subset of {list(ARMS)}; got {self.arms}.")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {sorted(LOSSES)}; got {self.loss!r}.")
        if self.n_splits < 2 or self.validation_folds < 2:
            raise ValueError("n_splits and validation_folds must be >= 2.")
        if self.max_folds is not None and not 1 <= self.max_folds <= self.n_splits:
            raise ValueError(f"max_folds must lie in [1, {self.n_splits}]; got {self.max_folds}.")
        if self.n_trials < 0:
            raise ValueError(f"n_trials must be >= 0; got {self.n_trials}.")
        if self.n_trials and self.inner_folds < 2:
            raise ValueError(f"inner_folds must be >= 2; got {self.inner_folds}.")
        untunable = sorted(set(self.search_space) - TUNABLE)
        if untunable:
            raise ValueError(f"{untunable} cannot be tuned; choose from {sorted(TUNABLE)}.")
        empty = sorted(k for k, values in self.search_space.items() if len(values) == 0)
        if empty:
            raise ValueError(f"no values to draw from for {empty}.")

    @property
    def evolution_time(self) -> float:
        """The pulse duration ``T = ΩT / Ω`` in µs."""
        return self.omega_t / self.omega

    @property
    def defaults(self) -> dict[str, Any]:
        """The training settings a tuning trial overrides."""
        return {
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
        }

    def as_dict(self) -> dict[str, Any]:
        """Every field as plain data."""
        plain: dict[str, Any] = dataclasses.asdict(self)
        plain["search_space"] = {k: list(v) for k, v in self.search_space.items()}
        return plain


#: ``--quick``: one fold of one small grid point, with dephasing so that the
#: solver's stepping runs, on a small positive control.
QUICK = Settings(
    v_over_omega=(1.0,),
    gamma_over_omega=(0.1,),
    n_steps=40,
    hidden=4,
    n_splits=4,
    max_folds=1,
    validation_folds=4,
    max_epochs=6,
    batch_size=32,
    patience=None,
    n_trials=2,
    inner_folds=2,
    search_space={"lr": (0.01, 0.05)},
    n_samples=160,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class PositiveControl(NamedTuple):
    """
    Attributes
    ----------
    X:
        Raw features, shape ``(n_samples, 2^N − 1)``, float64.
    y:
        Labels, shape ``(n_samples,)``, int64.
    latent:
        The latent variables ``z``, shape ``(n_samples, N)``: column ``i``
        is the one that ends up on atom ``i``.
    """

    X: FloatArray
    y: IntArray
    latent: FloatArray


def product_labels(latent: npt.ArrayLike) -> IntArray:
    """
    ``y = 1`` where ``Σ_{i=0}^{N−2} z_i z_{i+1} > 0``, else 0.

    The sum runs over the bonds of an open chain: each term is the product
    of two latent variables that sit on neighbouring atoms.

    Parameters
    ----------
    latent:
        Shape ``(n_samples, N)`` with ``N >= 2``; column ``i`` belongs to
        atom ``i``.
    """
    z = np.asarray(latent, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError(f"latent must have shape (n_samples, N >= 2); got {z.shape}.")
    bonds: FloatArray = (z[:, :-1] * z[:, 1:]).sum(axis=1)
    return (bonds > 0).astype(np.int64)


def positive_control(
    n_atoms: int, n_samples: int, seed: int, *, noise: float = 0.1
) -> PositiveControl:
    """
    A task that needs products of inputs on neighbouring atoms.

    Construction
    ------------
    ``z`` holds ``N`` independent standard normal latent variables per
    sample, and the label is :func:`product_labels` of them.  The raw
    features are noisy copies: latent ``i`` appears in ``m_i = 2^(N−1−i)``
    columns, each ``z_i + noise · ε`` with its own standard normal ε.  The
    columns are in the order of the latents (``m_0`` copies of ``z_0``
    first), ``2^N − 1`` in total.

    Why latent ``i`` ends up on atom ``i``
    --------------------------------------
    The pipeline standardises the columns and keeps the first ``N``
    principal components, in order of decreasing variance.  Two copies of
    the same latent correlate with ``ρ = 1/(1 + noise²)`` and copies of
    different latents not at all, so the correlation matrix is block
    diagonal, and block ``i`` has the leading eigenvalue

    ::

        λ_i = 1 + (m_i − 1) ρ

    with a constant eigenvector (every other eigenvalue is ``1 − ρ``).
    ``m_i`` halves from one latent to the next, so ``λ_0 > λ_1 > …``:
    component ``i`` is the mean of the copies of latent ``i``, with a
    positive sign (``PCANormalizer`` makes the largest entry of a component
    positive).  Input ``i`` is ``π · tanh`` of that standardised mean, a
    strictly increasing function of ``z_i`` up to the noise, and it sets the
    detuning of atom ``i``.  On a finite sample the components are these up
    to sampling error.

    Why a single input says nothing about the label
    -----------------------------------------------
    Fix an atom ``i`` and flip the sign of every latent whose index has the
    other parity.  Each bond ``(j, j+1)`` joins one index of each parity, so
    every product ``z_j z_{j+1}`` changes sign and the label flips, while
    ``z_i`` is unchanged and the distribution of ``z`` is the same.  Hence
    ``P(y = 1 | z_i) = ½`` for every ``i``: the label is independent of each
    single latent, and so of each single input.  For any functions ``g_i``,

    ::

        Cov( y, Σ_i g_i(x_i) ) = 0

    Arm B's features are each a function of one input, so their class means
    coincide, as those of the inputs do: the Fisher discriminant ratio of
    either is 0 in the population (on ``M`` samples it sits at its
    no-information level, about ``d · M/(n0 n1)`` for ``d`` features), and
    no logit of the form ``Σ_i g_i(x_i)`` correlates with the label.  That
    is less than "at chance": a decision ``w · x > t`` is not a sum of
    single-input functions, and a half-space far from the centre can isolate
    a corner in which two neighbouring inputs share a sign, so a linear head
    on such features can still score above zero.  The interactions of arm A
    make feature ``i`` depend on ``x_{i−1}`` and ``x_{i+1}`` as well, and
    the class means of its features differ.  The classes are balanced, so
    SMOTE adds at most a few rows.

    Parameters
    ----------
    n_atoms:
        ``N >= 2``.
    n_samples:
        Number of samples.
    seed:
        Seed of the generator the latents and the noise are drawn from.
    noise:
        Standard deviation of the noise on each copy, ``> 0``.

    Returns
    -------
    PositiveControl
    """
    if n_atoms < 2:
        raise ValueError(f"n_atoms must be >= 2 for a chain with a bond; got {n_atoms}.")
    if noise <= 0:
        raise ValueError(f"noise must be > 0; got {noise}.")
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_samples, n_atoms))
    copies = [2 ** (n_atoms - 1 - i) for i in range(n_atoms)]
    owner = np.repeat(np.arange(n_atoms), copies)
    X = latent[:, owner] + noise * rng.standard_normal((n_samples, owner.size))
    return PositiveControl(np.ascontiguousarray(X), product_labels(latent), latent)


def load_dataset(
    name: str,
    settings: Settings,
    *,
    path: str | os.PathLike[str] | None = None,
    download: bool = False,
) -> tuple[FloatArray, IntArray, int | None]:
    """
    ``(X, y, data_seed)`` of a dataset in :data:`DATASETS`.

    ``data_seed`` is the seed the positive control was drawn with, derived
    from ``settings.random_state``, and ``None`` for a UCI dataset, which is
    read from ``path`` as its loader documents.
    """
    if name == "positive-control":
        data_seed = run_seeds(settings.random_state)["data_seed"]
        control = positive_control(settings.n_atoms, settings.n_samples, data_seed)
        return control.X, control.y, data_seed
    loaders = {
        "iranian-churn": load_iranian_churn,
        "taiwanese-bankruptcy": load_taiwanese_bankruptcy,
        "cervical-cancer-risk": load_cervical_cancer_risk,
    }
    if name not in loaders:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}.")
    data = loaders[name](path, download=download)
    return data.X, data.y, None


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def _draw(root: np.random.SeedSequence, n: int) -> list[int]:
    return [int(child.generate_state(1)[0]) for child in root.spawn(n)]


def run_seeds(random_state: int) -> dict[str, int]:
    """The seeds that belong to the run: positive control, shots, arm D's circuit."""
    data_seed, shot_seed, circuit_seed = _draw(
        np.random.SeedSequence([random_state, _RUN_STREAM]), 3
    )
    return {"data_seed": data_seed, "shot_seed": shot_seed, "circuit_seed": circuit_seed}


def fold_seeds(random_state: int, dataset: str, n_splits: int) -> tuple[int, list[dict[str, int]]]:
    """
    The outer split seed of a dataset and, per fold, the seeds every arm shares.

    As in ``run_benchmark``: one seed sequence per dataset, from the root
    seed and the dataset's position in :data:`DATASETS`; its first child
    seeds the outer split, and each further child one fold's validation
    split (``inner_seed``), SMOTE draw, initialisation, batch order and
    tuning.
    """
    root = np.random.SeedSequence([random_state, DATASETS.index(dataset)])
    split_root, *fold_roots = root.spawn(n_splits + 1)
    names = ("inner_seed", "smote_seed", "init_seed", "batch_seed", "tune_seed")
    per_fold = [dict(zip(names, _draw(fold_root, len(names)))) for fold_root in fold_roots]
    return int(split_root.generate_state(1)[0]), per_fold


# ---------------------------------------------------------------------------
# Inputs: one fold's rows, preprocessed and oversampled
# ---------------------------------------------------------------------------


def preprocess(X: FloatArray, fit_rows: RowArray, n_components: int) -> FloatArray:
    """
    The bounded inputs of every row of ``X``, fitted on ``fit_rows`` only.

    Each column is standardised with the mean and standard deviation of
    ``X[fit_rows]`` (a constant column is only centred), and a
    :class:`PCANormalizer` fitted on those rows projects onto their first
    ``n_components`` principal components, standardises each and squashes it
    with ``π · tanh``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, n_components)``, float64, in ``(−π, π)`` (at
        the float32 precision ``PCANormalizer`` returns).  Column ``i`` is
        component ``i``, in order of decreasing variance.
    """
    mean = X[fit_rows].mean(axis=0)
    std = X[fit_rows].std(axis=0)
    std[std == 0] = 1.0
    scaled = (X - mean) / std
    normalizer = PCANormalizer(n_components=n_components, scale_to_pi=True)
    normalizer.fit(scaled[fit_rows])
    inputs: FloatArray = normalizer.transform(scaled).double().numpy()
    return inputs


@dataclass(frozen=True)
class Split:
    """
    The inputs of one fold, stacked: training rows, validation rows, test rows.

    Attributes
    ----------
    inputs:
        Shape ``(n_train + n_val + n_test, N)``.  The training block holds
        the real training rows in their order, then the ``n_synthetic``
        SMOTE rows.
    labels:
        The label of each row of ``inputs``, int64.
    n_train, n_synthetic, n_val, n_test:
        Block sizes; ``n_train`` includes the synthetic rows.  With
        ``n_test = 0`` (a tuning fold) the validation rows are scored.
    """

    inputs: FloatArray
    labels: IntArray
    n_train: int
    n_synthetic: int
    n_val: int
    n_test: int

    @property
    def train(self) -> slice:
        return slice(0, self.n_train)

    @property
    def val(self) -> slice:
        return slice(self.n_train, self.n_train + self.n_val)

    @property
    def test(self) -> slice:
        """The rows that are scored: the test block, or the validation block without one."""
        if self.n_test == 0:
            return self.val
        return slice(self.n_train + self.n_val, self.n_train + self.n_val + self.n_test)

    @property
    def real(self) -> RowArray:
        """Positions of the real rows of the training part: training rows, then validation."""
        return np.concatenate(
            [
                np.arange(self.n_train - self.n_synthetic),
                np.arange(self.n_train, self.n_train + self.n_val),
            ]
        )


def make_split(
    X: FloatArray,
    y: IntArray,
    fit_rows: RowArray,
    train_rows: RowArray,
    val_rows: RowArray,
    test_rows: RowArray | None,
    *,
    n_components: int,
    smote_seed: int,
) -> Split:
    """
    Steps 2 and 3 of the module docstring for one set of rows.

    The inputs are fitted on ``fit_rows`` (:func:`preprocess`) and the
    training rows oversampled with :func:`smote` in input space.  An outer
    fold passes its training part as ``fit_rows``; a tuning fold passes its
    fitting rows as both ``fit_rows`` and ``train_rows`` and no test rows.
    """
    inputs = preprocess(X, fit_rows, n_components)
    oversampled = smote(inputs[train_rows], y[train_rows], random_state=smote_seed)
    blocks = [oversampled.X, inputs[val_rows]]
    labels = [np.asarray(oversampled.y, dtype=np.int64), y[val_rows]]
    if test_rows is not None:
        blocks.append(inputs[test_rows])
        labels.append(y[test_rows])
    return Split(
        inputs=np.ascontiguousarray(np.concatenate(blocks)),
        labels=np.concatenate(labels).astype(np.int64),
        n_train=int(oversampled.X.shape[0]),
        n_synthetic=int(oversampled.sources.shape[0]),
        n_val=int(val_rows.size),
        n_test=0 if test_rows is None else int(test_rows.size),
    )


# ---------------------------------------------------------------------------
# The grid and the feature maps
# ---------------------------------------------------------------------------


def spacing(v_over_omega: float, omega: float, c6: float) -> float:
    """
    The chain spacing ``a`` in µm with nearest-neighbour interaction ``V = (V/Ω) · Ω``.

    ``V = C6/a⁶`` and ``R_b = (C6/Ω)^(1/6)`` give ``V/Ω = (R_b/a)⁶``, so
    ``a = R_b · (V/Ω)^(−1/6)``.  At the reference ``Ω = 4π`` rad/µs and the
    default ``C6``, ``V/Ω = 1`` is ``a = R_b = 8.69`` µm and ``V/Ω = 10`` is
    5.92 µm.

    Parameters
    ----------
    v_over_omega:
        ``V/Ω > 0``, dimensionless.
    omega:
        Ω in rad/µs.
    c6:
        ``C6`` in rad/µs · µm⁶, positive.
    """
    if v_over_omega <= 0:
        raise ValueError(f"v_over_omega must be > 0; got {v_over_omega}.")
    return AtomRegister.blockade_radius(c6, omega) * v_over_omega ** (-1.0 / 6.0)


def feature_map(
    settings: Settings, v_over_omega: float, gamma_over_omega: float, *, interactions: bool
) -> RydbergFeatureMap:
    """
    The map of one grid point: arm A with ``interactions=True``, arm B without.

    An open chain of ``n_atoms`` atoms at :func:`spacing`, the pulse
    ``ΩT = settings.omega_t`` with the default detuning bound ``√3 Ω``, and
    ``γ = (γ/Ω) · Ω``.  The two arms differ in the ``interactions`` flag and
    in nothing else.  ``n_steps`` is passed only with ``γ > 0``.
    """
    gamma = gamma_over_omega * settings.omega
    return RydbergFeatureMap(
        AtomRegister.chain(settings.n_atoms, spacing(v_over_omega, settings.omega, settings.c6)),
        PulseEncoding(settings.n_atoms, omega=settings.omega, t=settings.evolution_time),
        c6=settings.c6,
        gamma=gamma,
        n_steps=settings.n_steps if gamma > 0 else None,
        interactions=interactions,
    )


def grid(settings: Settings) -> list[tuple[float, float]]:
    """The grid points ``(V/Ω, γ/Ω)``: the product of the two lists, γ varying fastest."""
    return list(itertools.product(settings.v_over_omega, settings.gamma_over_omega))


def frozen_circuit(settings: Settings, circuit_seed: int) -> QuantumEncodingLayer:
    """
    Arm D's fixed gate-based feature map.

    A :class:`~hqnn_forge.encoding.QuantumEncodingLayer` of ``n_atoms``
    qubits: input ``i`` is the angle of an ``RX`` rotation on qubit ``i``
    (it lies in ``(−π, π)``, the range the layer expects), followed by
    ``circuit_layers`` layers of the ring ansatz and a ``⟨Z_i⟩`` readout of
    every qubit, so feature ``i`` belongs to qubit ``i``, in ``[−1, 1]``.

    The rotation angles of the ansatz are drawn once from ``U(−π, π)`` with
    a generator seeded by ``circuit_seed`` and frozen.  The full range, not
    the small-angle initialisation of the trainable models: near the
    identity the circuit would hardly mix the inputs, and this arm is the
    gate-based counterpart of a map that does.  With ``settings.shots`` the
    readout is estimated from that many shots, seeded with the run's
    ``shot_seed``.
    """
    sampling: dict[str, Any] = (
        {"diff_method": "backprop"}
        if settings.shots is None
        else {"shots": settings.shots, "seed": run_seeds(settings.random_state)["shot_seed"]}
    )
    layer = QuantumEncodingLayer(
        settings.n_atoms, settings.circuit_layers, device_name="default.qubit", **sampling
    )
    weights = layer.qlayer.weights
    generator = torch.Generator().manual_seed(circuit_seed)
    with torch.no_grad():
        draw = torch.rand(weights.shape, generator=generator, dtype=torch.float64)
        weights.copy_((2.0 * draw - 1.0) * math.pi)
    layer.requires_grad_(False)
    layer.eval()
    return layer


@dataclass(frozen=True)
class Representation:
    """
    What one arm hands to its heads.

    Attributes
    ----------
    arm, kind:
        ``"A"``/``"rydberg"``, ``"B"``/``"rydberg-noninteracting"``,
        ``"C"``/``"inputs"``, ``"D"``/``"frozen-circuit"`` or
        ``"D-hybrid"``/``"inputs"``.
    heads:
        The models trained on it.
    v_over_omega, gamma_over_omega:
        The grid point, ``None`` for the arms without one.
    config:
        The settings the features are cached under: the map's
        ``get_config()`` (or arm D's circuit settings) with the shots.
        ``None`` for the inputs themselves.
    compute:
        Inputs ``(n, N)`` to features ``(n, d)``, float64; ``None`` for the
        inputs themselves.
    n_frozen_parameters:
        Parameters of the map that are fixed and never trained.
    """

    arm: str
    kind: str
    heads: tuple[str, ...]
    v_over_omega: float | None = None
    gamma_over_omega: float | None = None
    config: dict[str, Any] | None = None
    compute: Callable[[FloatArray], FloatArray] | None = None
    n_frozen_parameters: int = 0


def _rydberg_representation(
    settings: Settings, arm: str, v: float, gamma: float, shot_seed: int
) -> Representation:
    fmap = feature_map(settings, v, gamma, interactions=arm == "A")
    shots = settings.shots

    def compute(inputs: FloatArray) -> FloatArray:
        generator = None if shots is None else torch.Generator().manual_seed(shot_seed)
        features = fmap.transform(torch.from_numpy(inputs), shots=shots, generator=generator)
        result: FloatArray = features.numpy()
        return result

    return Representation(
        arm=arm,
        kind="rydberg" if arm == "A" else "rydberg-noninteracting",
        heads=HEADS,
        v_over_omega=v,
        gamma_over_omega=gamma,
        config={
            "feature_map": "RydbergFeatureMap",
            "config": fmap.get_config(),
            "shots": shots,
            "shot_seed": None if shots is None else shot_seed,
        },
        compute=compute,
    )


def _circuit_representation(
    settings: Settings, circuit_seed: int, shot_seed: int
) -> Representation:
    layer = frozen_circuit(settings, circuit_seed)

    def compute(inputs: FloatArray) -> FloatArray:
        x = torch.from_numpy(inputs)
        with torch.no_grad():
            chunks = [
                layer(x[start : start + _CIRCUIT_CHUNK])
                for start in range(0, x.shape[0], _CIRCUIT_CHUNK)
            ]
        result: FloatArray = torch.cat(chunks).to(torch.float64).numpy()
        return result

    return Representation(
        arm="D",
        kind="frozen-circuit",
        heads=HEADS,
        config={
            "feature_map": "QuantumEncodingLayer",
            "config": {
                "n_qubits": settings.n_atoms,
                "n_layers": settings.circuit_layers,
                "rotation": "X",
                "entangler": "ring",
                "readout": "all",
                "device_name": "default.qubit",
                "weights": "uniform(-pi, pi)",
                "circuit_seed": circuit_seed,
            },
            "shots": settings.shots,
            "shot_seed": None if settings.shots is None else shot_seed,
        },
        compute=compute,
        n_frozen_parameters=int(layer.qlayer.weights.numel()),
    )


def representations(settings: Settings) -> list[Representation]:
    """
    Every representation of a run, in output order: A and B at each grid
    point, then C, D and D-hybrid, as far as ``settings.arms`` selects them.
    """
    seeds = run_seeds(settings.random_state)
    result: list[Representation] = []
    for v, gamma in grid(settings):
        for arm in ("A", "B"):
            if arm in settings.arms:
                result.append(_rydberg_representation(settings, arm, v, gamma, seeds["shot_seed"]))
    if "C" in settings.arms:
        result.append(Representation(arm="C", kind="inputs", heads=HEADS))
    if "D" in settings.arms:
        result.append(_circuit_representation(settings, seeds["circuit_seed"], seeds["shot_seed"]))
        result.append(Representation(arm="D-hybrid", kind="inputs", heads=("hybrid",)))
    return result


# ---------------------------------------------------------------------------
# The feature cache
# ---------------------------------------------------------------------------


def cache_key(data_sha256: str, config: Mapping[str, Any]) -> str:
    """SHA-256 of the data fingerprint and the feature map's settings, as canonical JSON."""
    text = json.dumps(
        {"data": data_sha256, "config": _plain(config)}, sort_keys=True, allow_nan=False
    )
    return hashlib.sha256(text.encode()).hexdigest()


class CachedFeatures(NamedTuple):
    """Features, the seconds their computation took, and whether they were read from disk."""

    features: FloatArray
    seconds: float
    hit: bool


class FeatureCache:
    """
    Features on disk, one ``.npz`` file per input matrix and feature-map config.

    The file name is :func:`cache_key` of
    :func:`hqnn_forge.benchmark.fingerprint` of the inputs and their labels
    and of the config.  A file holds the features and the seconds their
    computation took; it is written to a temporary name and renamed, so an
    interrupted run leaves no partial entry.  With ``directory=None``
    nothing is stored and every request computes.
    """

    def __init__(self, directory: str | os.PathLike[str] | None) -> None:
        self.directory = None if directory is None else Path(directory)
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, inputs: FloatArray, labels: IntArray, config: Mapping[str, Any]) -> Path | None:
        """Where the features of these inputs under this config are stored."""
        if self.directory is None:
            return None
        return self.directory / f"{cache_key(fingerprint(inputs, labels), config)}.npz"

    def fetch(
        self,
        inputs: FloatArray,
        labels: IntArray,
        config: Mapping[str, Any],
        compute: Callable[[FloatArray], FloatArray],
    ) -> CachedFeatures:
        """The stored features, or ``compute(inputs)``, timed and stored."""
        path = self.path(inputs, labels, config)
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as stored:
                return CachedFeatures(
                    np.asarray(stored["features"], dtype=np.float64),
                    float(stored["seconds"]),
                    True,
                )
        start = time.perf_counter()
        features = np.ascontiguousarray(compute(inputs), dtype=np.float64)
        seconds = time.perf_counter() - start
        if path is not None:
            partial = path.with_name(f"{path.stem}.{os.getpid()}.part")
            with partial.open("wb") as fh:
                np.savez(fh, features=features, seconds=np.float64(seconds))
            os.replace(partial, path)
        return CachedFeatures(features, seconds, False)


def features_of(rep: Representation, split: Split, cache: FeatureCache) -> CachedFeatures:
    """The matrix ``rep`` hands to its heads for the rows of ``split``, row for row."""
    if rep.compute is None or rep.config is None:
        return CachedFeatures(split.inputs, 0.0, False)
    return cache.fetch(split.inputs, split.labels, rep.config, rep.compute)


# ---------------------------------------------------------------------------
# Models, training, tuning
# ---------------------------------------------------------------------------


def build_model(
    head: str, n_features: int, settings: Settings, init_seed: int
) -> BinaryClassifierBase:
    """
    A fresh model for ``n_features`` columns, initialised from ``init_seed``.

    ``linear``
        ``LinearClassifier(n_features)``: ``n_features + 1`` parameters.
    ``mlp``
        ``ClassicalBaseline(n_features, [hidden])``, one ReLU hidden layer:
        ``hidden · (n_features + 2) + 1`` parameters.
    ``hybrid``
        ``HybridBinaryClassifier`` with ``n_features`` qubits and
        ``circuit_layers`` layers, without a classical encoder (the inputs
        already lie in ``(−π, π)``), on ``default.qubit`` with backprop:
        ``3 · circuit_layers · n_features`` rotation angles and a linear
        head of ``n_features + 1``.
    """
    if head == "linear":
        return LinearClassifier(n_features, init_seed=init_seed)
    if head == "mlp":
        return ClassicalBaseline(n_features, [settings.hidden], init_seed=init_seed)
    if head == "hybrid":
        return HybridBinaryClassifier(
            n_features,
            n_features,
            settings.circuit_layers,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="backprop",
            init_seed=init_seed,
        )
    raise ValueError(f"unknown head {head!r}.")


class Fit(NamedTuple):
    """One trained model scored on the scored rows of its split."""

    mcc: float
    pr_auc: float
    threshold: float
    seconds: float
    epochs: int


def fit_and_score(
    model: BinaryClassifierBase,
    features: FloatArray,
    split: Split,
    config: Mapping[str, Any],
    *,
    loss: str,
    batch_seed: int,
) -> Fit:
    """
    Train on the training block, stop and choose the threshold on the
    validation block, score the test block.

    ``features`` has one row per row of ``split.inputs``.  The threshold is
    the one that maximises MCC on the validation rows at the best epoch
    (0.5 if none was recorded) and is applied unchanged to the scored rows.
    ``pr_auc`` is NaN if the model's probabilities are not finite.
    """
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(split.labels.astype(np.float32))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(batch_seed)
        start = time.perf_counter()
        history = train_model(
            model,
            LOSSES[loss](),
            torch.optim.Adam(model.parameters(), lr=config["lr"]),
            x[split.train],
            y[split.train],
            x[split.val],
            y[split.val],
            max_epochs=config["max_epochs"],
            batch_size=config["batch_size"],
            monitor="mcc",
            patience=config["patience"],
            generator=torch.Generator().manual_seed(batch_seed),
        )
        seconds = time.perf_counter() - start
        probability = model.predict_proba(x[split.test])
    threshold = history.best_threshold if history.best_threshold is not None else 0.5
    truth = split.labels[split.test]
    finite = bool(torch.isfinite(probability).all())
    return Fit(
        mcc=float(matthews_corrcoef(truth, (probability >= threshold).long())),
        pr_auc=pr_auc(truth, probability) if finite else math.nan,
        threshold=float(threshold),
        seconds=seconds,
        epochs=history.n_epochs,
    )


def sample_configs(
    space: Mapping[str, Sequence[Any]], n_trials: int, seed: int
) -> list[dict[str, Any]]:
    """
    ``n_trials`` distinct configurations from the grid of ``space`` (all of
    it if it has no more), in grid order, as ``run_benchmark``'s tuning
    draws them.  Every arm of a fold gets this same list.
    """
    keys = sorted(space)
    full = [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]
    if len(full) <= n_trials:
        return full
    chosen = np.random.default_rng(seed).choice(len(full), size=n_trials, replace=False)
    return [full[int(i)] for i in sorted(chosen)]


def inner_splits(
    X: FloatArray, y: IntArray, train_part: RowArray, settings: Settings, tune_seed: int
) -> list[Split]:
    """
    The tuning folds of one outer fold: stratified folds of its training part.

    Each is prepared from its fitting rows alone (:func:`make_split` with no
    test rows), so a trial is trained on the fitting rows and scored on the
    held-out rows of the training part.  Rows outside ``train_part`` are
    never indexed.
    """
    folds = stratified_kfold(y[train_part], settings.inner_folds, random_state=tune_seed)
    return [
        make_split(
            X,
            y,
            train_part[fit],
            train_part[fit],
            train_part[held_out],
            None,
            n_components=settings.n_atoms,
            smote_seed=tune_seed,
        )
        for fit, held_out in folds
    ]


def tune(
    head: str,
    splits: Sequence[Split],
    features: Sequence[FloatArray],
    candidates: Sequence[Mapping[str, Any]],
    settings: Settings,
    tune_seed: int,
) -> tuple[dict[str, Any], list[float]]:
    """
    The candidate with the best mean MCC over the tuning folds (the first, on ties).

    Returns the chosen configuration and every candidate's score, in order.
    Each trial builds a fresh model from ``tune_seed``.
    """
    scores: list[float] = []
    for candidate in candidates:
        config = {**settings.defaults, **candidate}
        scores.append(
            float(
                np.mean(
                    [
                        fit_and_score(
                            build_model(head, matrix.shape[1], settings, tune_seed),
                            matrix,
                            split,
                            config,
                            loss=settings.loss,
                            batch_seed=tune_seed,
                        ).mcc
                        for split, matrix in zip(splits, features, strict=True)
                    ]
                )
            )
        )
    best = max(range(len(scores)), key=lambda i: (scores[i], -i))
    return dict(candidates[best]), scores


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _strict(value: Any) -> Any:
    """``value`` with every infinite float replaced by the string ``"inf"`` or ``"-inf"``."""
    if isinstance(value, float | np.floating) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, Mapping):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_strict(item) for item in value]
    return value


def to_json_line(record: Mapping[str, Any]) -> str:
    """
    One record as a line of strict JSON.

    NumPy values become Python ones and NaN becomes ``null``, as in
    :mod:`hqnn_forge.experiment`; ``+inf`` and ``−inf``, which that module
    refuses, become the strings ``"inf"`` and ``"-inf"``.
    """
    return json.dumps(_plain(_strict(record)), allow_nan=False)


def _restore(value: Any) -> float:
    if value is None:
        return math.nan
    if value in ("inf", "-inf"):
        return math.inf if value == "inf" else -math.inf
    return float(value)


def read_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """
    The records of a file written by this script, one dict per line.

    In ``separation`` the two float mappings of :func:`to_json_line` are
    undone: ``null`` is NaN and ``"inf"`` is infinity (the two integer
    fields stay integers).  Elsewhere a ``null`` stays ``None``, because
    there it also stands for "does not apply" (the grid fields of arm C).
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            for measures in record["separation"].values():
                for key, value in measures.items():
                    if key not in ("n_samples", "n_features"):
                        measures[key] = _restore(value)
            records.append(record)
    return records


def _grid_fields(rep: Representation, settings: Settings) -> dict[str, Any]:
    """The physical settings of a line; ``None`` throughout for an arm without a Rydberg map."""
    if rep.v_over_omega is None or rep.gamma_over_omega is None or rep.config is None:
        return dict.fromkeys(
            (
                "v_over_omega",
                "gamma_over_omega",
                "spacing_um",
                "omega",
                "omega_t",
                "evolution_time_us",
                "n_steps",
            )
        )
    return {
        "v_over_omega": rep.v_over_omega,
        "gamma_over_omega": rep.gamma_over_omega,
        "spacing_um": spacing(rep.v_over_omega, settings.omega, settings.c6),
        "omega": settings.omega,
        "omega_t": settings.omega_t,
        "evolution_time_us": settings.evolution_time,
        "n_steps": rep.config["config"]["n_steps"],
    }


def run_dataset(
    name: str,
    X: FloatArray,
    y: IntArray,
    settings: Settings,
    cache: FeatureCache,
    *,
    data_seed: int | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Every record of one dataset: per fold, each representation and head.

    Parameters
    ----------
    name:
        A name in :data:`DATASETS`; it selects the dataset's seeds.
    X, y:
        Raw features ``(n_samples, n_features)`` and binary labels.
    settings, cache:
        The run's settings and feature cache.
    data_seed:
        The seed the data was generated with, recorded with the other seeds.

    Yields
    ------
    dict
        One record per arm, head, grid point and fold, in that nesting from
        the inside out; see the module docstring for its fields.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    if X.ndim != 2 or y.shape != (X.shape[0],) or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{name}: X must be (n_samples, n_features) and y binary 0/1.")
    if X.shape[1] < settings.n_atoms:
        raise ValueError(
            f"{name}: {X.shape[1]} features cannot be reduced to n_atoms={settings.n_atoms}."
        )
    data_info = {
        "dataset": name,
        "data_sha256": fingerprint(X, y),
        "n_samples": int(y.size),
        "n_positives": int(y.sum()),
        "n_raw_features": int(X.shape[1]),
    }
    env = environment()
    run = run_seeds(settings.random_state)
    split_seed, per_fold = fold_seeds(settings.random_state, name, settings.n_splits)
    outer = stratified_kfold(y, settings.n_splits, random_state=split_seed)
    reps = representations(settings)
    n_folds = settings.n_splits if settings.max_folds is None else settings.max_folds

    for k in range(n_folds):
        train_part, test_idx = outer[k]
        seeds = per_fold[k]
        inner_train, inner_val = stratified_kfold(
            y[train_part], settings.validation_folds, random_state=seeds["inner_seed"]
        )[0]
        train_idx, val_idx = train_part[inner_train], train_part[inner_val]
        split = make_split(
            X,
            y,
            train_part,
            train_idx,
            val_idx,
            test_idx,
            n_components=settings.n_atoms,
            smote_seed=seeds["smote_seed"],
        )
        tuning_splits: list[Split] = []
        candidates: list[dict[str, Any]] = []
        if settings.n_trials:
            tuning_splits = inner_splits(X, y, train_part, settings, seeds["tune_seed"])
            candidates = sample_configs(
                settings.search_space, settings.n_trials, seeds["tune_seed"]
            )
        fold_info = {
            "fold": k,
            "n_splits": settings.n_splits,
            "train_idx": np.sort(train_idx),
            "val_idx": np.sort(val_idx),
            "test_idx": np.sort(test_idx),
            "n_synthetic": split.n_synthetic,
            "seeds": {
                "random_state": settings.random_state,
                "split_seed": split_seed,
                **seeds,
                **run,
                "data_seed": data_seed,
            },
        }

        for rep in reps:
            cached = features_of(rep, split, cache)
            matrix = cached.features
            separation = {
                "train": separation_measures(
                    matrix[split.real], split.labels[split.real]
                ).to_dict(),
                "test": separation_measures(
                    matrix[split.test], split.labels[split.test]
                ).to_dict(),
            }
            tuning_features = [features_of(rep, s, cache).features for s in tuning_splits]
            for head in rep.heads:
                chosen: dict[str, Any] = {}
                tuning: dict[str, Any] | None = None
                tuning_seconds = 0.0
                if candidates:
                    start = time.perf_counter()
                    chosen, scores = tune(
                        head,
                        tuning_splits,
                        tuning_features,
                        candidates,
                        settings,
                        seeds["tune_seed"],
                    )
                    tuning_seconds = time.perf_counter() - start
                    tuning = {
                        "n_trials": len(candidates),
                        "inner_folds": settings.inner_folds,
                        "candidates": candidates,
                        "scores": scores,
                    }
                hyperparameters = {**settings.defaults, **chosen}
                model = build_model(head, matrix.shape[1], settings, seeds["init_seed"])
                fit = fit_and_score(
                    model,
                    matrix,
                    split,
                    hyperparameters,
                    loss=settings.loss,
                    batch_seed=seeds["batch_seed"],
                )
                sampled = settings.shots is not None and rep.arm in ("A", "B", "D")
                yield {
                    **data_info,
                    "arm": rep.arm,
                    "head": head,
                    "representation": rep.kind,
                    "feature_map_trained": head == "hybrid",
                    **_grid_fields(rep, settings),
                    "feature_map": rep.config,
                    "measurement": "shots" if sampled else "exact",
                    "shots": settings.shots if sampled else None,
                    **fold_info,
                    "feature_dim": int(matrix.shape[1]),
                    "n_parameters": model.count_parameters(),
                    "n_frozen_parameters": rep.n_frozen_parameters,
                    "model": {"class": type(model).__name__, "config": model.get_config()},
                    "hyperparameters": hyperparameters,
                    "tuning": tuning,
                    "mcc": fit.mcc,
                    "pr_auc": fit.pr_auc,
                    "threshold": fit.threshold,
                    "epochs": fit.epochs,
                    "feature_seconds": cached.seconds,
                    "feature_cache_hit": cached.hit,
                    "train_seconds": fit.seconds,
                    "tuning_seconds": tuning_seconds,
                    "separation": separation,
                    "settings": settings.as_dict(),
                    "environment": env,
                }


def summarise(records: Sequence[Mapping[str, Any]]) -> str:
    """Mean and fold spread of MCC and PR-AUC per dataset, arm, head and grid point."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            record["dataset"],
            record["v_over_omega"],
            record["gamma_over_omega"],
            record["arm"],
            record["head"],
        )
        groups.setdefault(key, []).append(record)
    header = (
        f"{'dataset':22s} {'V/Ω':>6s} {'γ/Ω':>6s} {'arm':8s} {'head':6s} "
        f"{'MCC':>15s} {'PR-AUC':>7s} {'dim':>4s} {'params':>6s} {'feat s':>7s} {'fit s':>6s}  n"
    )
    lines = [header]
    for (dataset, v, gamma, arm, head), rows in groups.items():
        mcc = np.array([r["mcc"] for r in rows], dtype=np.float64)
        spread = mcc.std(ddof=1) if len(rows) > 1 else 0.0
        ap = float(np.mean([r["pr_auc"] for r in rows]))
        lines.append(
            f"{dataset:22s} {'-' if v is None else format(v, 'g'):>6s} "
            f"{'-' if gamma is None else format(gamma, 'g'):>6s} {arm:8s} {head:6s} "
            f"{mcc.mean():7.3f} ± {spread:5.3f} {ap:7.3f} {rows[0]['feature_dim']:4d} "
            f"{rows[0]['n_parameters']:6d} "
            f"{np.mean([r['feature_seconds'] for r in rows]):7.2f} "
            f"{np.mean([r['train_seconds'] for r in rows]):6.2f}  {len(rows)}"
        )
    return "\n".join(lines)


def run(
    datasets: Mapping[str, tuple[FloatArray, IntArray, int | None]],
    settings: Settings,
    out: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Run every dataset and write one JSON line per record to ``out``.

    Parameters
    ----------
    datasets:
        Name → ``(X, y, data_seed)``, as :func:`load_dataset` returns them.
    settings:
        The run's settings.
    out:
        The JSON-lines file; overwritten.
    cache_dir:
        Directory of the feature cache; ``None`` computes without storing.

    Returns
    -------
    list of dict
        The records as written, before the JSON mapping.
    """
    cache = FeatureCache(cache_dir)
    records: list[dict[str, Any]] = []
    with open(out, "w", encoding="utf-8") as fh:
        for name, (X, y, data_seed) in datasets.items():
            for record in run_dataset(name, X, y, settings, cache, data_seed=data_seed):
                records.append(record)
                fh.write(to_json_line(record) + "\n")
                fh.flush()
    return records


def _parser() -> argparse.ArgumentParser:
    default = Settings()
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("rydberg_features_benchmark.jsonl"))
    parser.add_argument("--quick", action="store_true", help="one fold, one small grid point")
    parser.add_argument("--dataset", nargs="+", choices=DATASETS, default=["positive-control"])
    parser.add_argument("--data-path", type=Path, default=None, help="file or directory (UCI)")
    parser.add_argument("--download", action="store_true", help="fetch a missing UCI file")
    parser.add_argument("--cache-dir", type=Path, default=Path("rydberg_features_cache"))
    parser.add_argument("--no-cache", action="store_true", help="do not store features")
    parser.add_argument("--n-atoms", type=int, default=default.n_atoms)
    parser.add_argument(
        "--v-over-omega", type=float, nargs="+", default=list(default.v_over_omega)
    )
    parser.add_argument(
        "--gamma-over-omega", type=float, nargs="+", default=list(default.gamma_over_omega)
    )
    parser.add_argument("--n-steps", type=int, default=default.n_steps)
    parser.add_argument("--shots", type=int, default=None, help="default: exact features")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(default.arms))
    parser.add_argument("--hidden", type=int, default=default.hidden)
    parser.add_argument("--circuit-layers", type=int, default=default.circuit_layers)
    parser.add_argument("--folds", type=int, default=default.n_splits)
    parser.add_argument("--max-folds", type=int, default=None, help="default: every fold")
    parser.add_argument("--validation-folds", type=int, default=default.validation_folds)
    parser.add_argument("--trials", type=int, default=default.n_trials, help="0: no tuning")
    parser.add_argument("--inner-folds", type=int, default=default.inner_folds)
    parser.add_argument("--loss", choices=sorted(LOSSES), default=default.loss)
    parser.add_argument("--lr", type=float, default=default.lr)
    parser.add_argument("--epochs", type=int, default=default.max_epochs)
    parser.add_argument("--batch-size", type=int, default=default.batch_size)
    parser.add_argument("--patience", type=int, default=default.patience)
    parser.add_argument("--samples", type=int, default=default.n_samples)
    parser.add_argument("--seed", type=int, default=default.random_state)
    return parser


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """
    Parse the command line, run, print the summary and return the records.

    ``--quick`` replaces the grid, the budget and the dataset by those of
    :data:`QUICK`; only ``--out``, the cache options and ``--seed`` still
    apply.
    """
    args = _parser().parse_args(argv)
    if args.quick:
        settings = dataclasses.replace(QUICK, random_state=args.seed)
        names = ["positive-control"]
    else:
        settings = Settings(
            n_atoms=args.n_atoms,
            v_over_omega=tuple(args.v_over_omega),
            gamma_over_omega=tuple(args.gamma_over_omega),
            n_steps=args.n_steps,
            shots=args.shots,
            arms=tuple(args.arms),
            hidden=args.hidden,
            circuit_layers=args.circuit_layers,
            n_splits=args.folds,
            max_folds=args.max_folds,
            validation_folds=args.validation_folds,
            loss=args.loss,
            lr=args.lr,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            n_trials=args.trials,
            inner_folds=args.inner_folds,
            n_samples=args.samples,
            random_state=args.seed,
        )
        names = list(dict.fromkeys(args.dataset))
    datasets = {
        name: load_dataset(name, settings, path=args.data_path, download=args.download)
        for name in names
    }
    with warnings.catch_warnings():
        # PennyLane's device fallbacks and PCANormalizer's notes, once per fold and trial.
        warnings.simplefilter("ignore")
        records = run(
            datasets, settings, args.out, cache_dir=None if args.no_cache else args.cache_dir
        )
    print(summarise(records))
    print(f"wrote {len(records)} records to {args.out}")
    return records


if __name__ == "__main__":
    main()
