"""
hqnn_forge.sklearn
==================
A scikit-learn estimator around the hybrid classifiers.

``QuantumKernelClassifier`` is the quantum-kernel SVM (QSVM) as an estimator:
the same bookkeeping around ``SVC(kernel="precomputed")`` that
:mod:`hqnn_forge.kernels` otherwise leaves to the caller.

``HybridClassifierEstimator`` implements ``fit`` / ``predict`` /
``predict_proba`` / ``get_params`` / ``set_params``, so a hybrid model can be
dropped into ``cross_val_score``, ``GridSearchCV`` and ``Pipeline`` like any
other classifier.  Training is delegated to
:func:`hqnn_forge.training.train_model`.

scikit-learn (>= 1.6) is an optional dependency, declared by the ``sklearn``
extra; importing this module without it raises an ``ImportError`` saying how to
install it.

Example
-------
::

    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from hqnn_forge.sklearn import HybridClassifierEstimator

    clf = make_pipeline(
        StandardScaler(),
        HybridClassifierEstimator(n_qubits=4, n_layers=2, max_epochs=20),
    )
    scores = cross_val_score(clf, X, y, cv=5, scoring="matthews_corrcoef")
"""

from __future__ import annotations

import numbers
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    from sklearn.utils.metaestimators import available_if
    from sklearn.utils.multiclass import check_classification_targets, unique_labels
    from sklearn.utils.validation import check_is_fitted, validate_data
except ImportError as exc:  # pragma: no cover - exercised only without scikit-learn
    raise ImportError(
        'hqnn_forge.sklearn needs scikit-learn >= 1.6: pip install "scikit-learn>=1.6" '
        '(or pip install "hqnn-forge[sklearn]").  validate_data and __sklearn_tags__ '
        "were added in 1.6, so an older install fails this import too."
    ) from exc

import math

from hqnn_forge import kernels
from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.training import TrainingHistory, train_model
from hqnn_forge.utils import FocalLoss, SoftmaxFocalLoss
from hqnn_forge.utils.rng import as_seed, seeded_rng

ModelName = Literal["serial", "parallel"]
LossName = Literal["focal", "bce"]
StrategyName = Literal["softmax", "one_vs_rest"]


class _OneHotLoss(torch.nn.Module):
    """A per-class binary loss on ``(N, K)`` logits and integer labels, via one-hot targets."""

    def __init__(self, loss: torch.nn.Module, n_classes: int) -> None:
        super().__init__()
        self.loss = loss
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        target = torch.nn.functional.one_hot(y.long(), self.n_classes).to(logits.dtype)
        return self.loss(logits, target)  # type: ignore[no-any-return]


class HybridClassifierEstimator(ClassifierMixin, BaseEstimator):
    """
    scikit-learn classifier training a hybrid quantum-classical model.

    Binary or multiclass, decided by ``y`` in ``fit``, like any scikit-learn
    classifier: two classes train a ``HybridBinaryClassifier`` (or the parallel
    model), three or more a ``MulticlassHybridClassifier`` with the same
    circuit options.

    Parameters
    ----------
    model:
        ``"serial"`` (``HybridBinaryClassifier``) or ``"parallel"``
        (``ParallelHybridClassifier``).  With more than two classes only
        ``"serial"`` exists (``MulticlassHybridClassifier``); ``"parallel"``
        raises in ``fit``.
    n_qubits, n_layers, encoding_type, init_strategy, use_classical_encoder,
    dropout_p, device_name, diff_method:
        Passed to the model constructor.  ``n_input_features`` is taken from
        the data in ``fit``.
    classical_hidden_dim:
        MLP width; used by the parallel model only.
    loss:
        ``"focal"`` (``FocalLoss()``, the library default for imbalanced data)
        or ``"bce"`` (``BCEWithLogitsLoss``).
    lr:
        Adam learning rate.
    max_epochs, batch_size, patience, monitor:
        Passed to ``train_model``.  Early stopping needs a validation split:
        with ``validation_fraction=0`` the run always lasts ``max_epochs``,
        whatever ``patience`` says.  ``patience=None`` disables early stopping
        even when there is a split.
    validation_fraction:
        Share of the training data held out (stratified) for early stopping
        and threshold selection.  ``0`` trains on everything.
    strategy:
        More than two classes only: ``"softmax"`` (default; trained with
        cross-entropy, or its focal version) or ``"one_vs_rest"`` (one binary
        head per class, trained with BCE, or the focal loss, on one-hot
        targets).  See :class:`~hqnn_forge.models.MulticlassHybridClassifier`.
    threshold:
        Binary only.  ``"optimal"`` uses the validation-optimal threshold found by
        ``train_model``; a float in ``[0, 1]`` fixes it.  ``train_model``
        searches a threshold for the metric monitors only, so ``"optimal"``
        falls back to 0.5 both without a validation split and under
        ``monitor="val_loss"``.
    random_state:
        Seeds weight initialisation, dropout, the validation split and batch
        order.  The initial weights are the model's ``init_seed=random_state``
        draws; dropout and batch order use seeds spawned from it, so they are
        independent of the init.  A seeded ``fit`` is reproducible and leaves
        the global torch RNG exactly as it was; ``None`` draws everything from
        the global RNG.  The draws of ``noise_method="trajectories"`` share
        the dropout stream, so a seeded noisy ``fit`` is reproducible too.
    noise_level, noise_position, noise_method, noise_trajectories:
        Noise-aware training, passed to the model: depolarizing noise of
        probability ``noise_level`` (in ``[0, 0.75]``) applied to the circuit
        in train mode only, so ``fit`` trains through the noisy circuit and
        ``predict`` / ``predict_proba`` are noiseless.  ``noise_position`` is
        ``"all"`` or ``"end"``; ``noise_method`` is ``"density"`` (exact,
        practical up to about 6 qubits) or ``"trajectories"`` (sampled, at
        pure-state cost), with ``noise_trajectories`` draws per sample.  The
        defaults (no noise) train exactly as without these parameters.  Like
        every other parameter they are validated in ``fit``, so they can be
        tuned with ``GridSearchCV``.  See :mod:`hqnn_forge.noise`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The labels, sorted; with two, ``classes_[1]`` is the positive class.
    n_features_in_ : int
    model_ : torch.nn.Module
        The trained model.
    history_ : TrainingHistory
    threshold_ : float or None
        Decision threshold used by ``predict``; ``None`` with more than two
        classes, where ``predict`` is the argmax.
    """

    def __init__(
        self,
        model: ModelName = "serial",
        n_qubits: int = 8,
        n_layers: int = 2,
        encoding_type: str = "angle",
        init_strategy: str = "restricted",
        use_classical_encoder: bool = True,
        classical_hidden_dim: int = 16,
        dropout_p: float = 0.0,
        device_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
        loss: LossName = "focal",
        lr: float = 0.01,
        max_epochs: int = 20,
        batch_size: int = 64,
        validation_fraction: float = 0.0,
        patience: int | None = 10,
        monitor: str = "mcc",
        threshold: float | Literal["optimal"] = "optimal",
        random_state: int | None = None,
        noise_level: float = 0.0,
        noise_position: str = "all",
        noise_method: str = "density",
        noise_trajectories: int = 1,
        strategy: StrategyName = "softmax",
    ) -> None:
        self.model = model
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        self.init_strategy = init_strategy
        self.use_classical_encoder = use_classical_encoder
        self.classical_hidden_dim = classical_hidden_dim
        self.dropout_p = dropout_p
        self.device_name = device_name
        self.diff_method = diff_method
        self.loss = loss
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.monitor = monitor
        self.threshold = threshold
        self.random_state = random_state
        self.noise_level = noise_level
        self.noise_position = noise_position
        self.noise_method = noise_method
        self.noise_trajectories = noise_trajectories
        self.strategy = strategy

    # ------------------------------------------------------------------
    def _build(
        self, n_features: int, n_classes: int, init_seed: int | None = None
    ) -> HybridBinaryClassifier | ParallelHybridClassifier | MulticlassHybridClassifier:
        common: dict[str, Any] = dict(
            n_input_features=n_features,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            use_classical_encoder=self.use_classical_encoder,
            dropout_p=self.dropout_p,
            device_name=self.device_name,
            diff_method=self.diff_method,
            init_strategy=self.init_strategy,
            encoding_type=self.encoding_type,
            noise_level=self.noise_level,
            noise_position=self.noise_position,
            noise_method=self.noise_method,
            noise_trajectories=self.noise_trajectories,
            init_seed=init_seed,
        )
        if self.model not in ("serial", "parallel"):
            raise ValueError(f"model must be 'serial' or 'parallel'; got {self.model!r}.")
        if n_classes > 2:
            if self.model != "serial":
                raise ValueError(
                    f"model='parallel' is a binary topology; {n_classes} classes need "
                    f"model='serial' (MulticlassHybridClassifier)."
                )
            return MulticlassHybridClassifier(
                n_classes=n_classes, strategy=self.strategy, **common
            )
        if self.model == "serial":
            return HybridBinaryClassifier(**common)
        return ParallelHybridClassifier(classical_hidden_dim=self.classical_hidden_dim, **common)

    def _loss(self, n_classes: int) -> torch.nn.Module:
        if self.loss not in ("focal", "bce"):
            raise ValueError(f"loss must be 'focal' or 'bce'; got {self.loss!r}.")
        if n_classes == 2:
            return FocalLoss() if self.loss == "focal" else torch.nn.BCEWithLogitsLoss()
        if self.strategy == "softmax":
            return SoftmaxFocalLoss() if self.loss == "focal" else torch.nn.CrossEntropyLoss()
        per_class = FocalLoss() if self.loss == "focal" else torch.nn.BCEWithLogitsLoss()
        return _OneHotLoss(per_class, n_classes)

    @staticmethod
    def _stratified_holdout(
        y: npt.NDArray[np.int64], fraction: float, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        val_parts, train_parts = [], []
        for cls in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == cls))
            n_val = int(round(fraction * idx.size))
            if n_val == 0 or n_val == idx.size:
                raise ValueError(
                    f"validation_fraction={fraction} leaves no training or no validation "
                    f"samples of class {cls} ({idx.size} available)."
                )
            val_parts.append(idx[:n_val])
            train_parts.append(idx[n_val:])
        return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))

    @staticmethod
    def _check_threshold(threshold: float | Literal["optimal"]) -> None:
        if threshold == "optimal":
            return
        # bool is a subclass of int, and threshold=True would silently mean 1.0.
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
            # ValueError, as for any other threshold outside 'optimal' or [0, 1]
            raise ValueError(  # noqa: TRY004
                f"threshold must be 'optimal' or a real number; got {threshold!r}."
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                f"threshold must lie in [0, 1], the range of a probability; got {threshold!r}."
            )

    # ------------------------------------------------------------------
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> HybridClassifierEstimator:
        """Build the model for ``X``'s width and train it on ``(X, y)``."""
        X_arr, y_arr = validate_data(self, X, y, dtype=np.float32)
        classes = unique_labels(y_arr)
        if classes.size < 2:
            # "1 class" is one of the phrasings scikit-learn's conformance
            # checks (check_fit2d_1sample) match on.
            raise ValueError(
                f"HybridClassifierEstimator needs at least two classes; got 1 class: "
                f"{classes.tolist()}."
            )
        if self.strategy not in ("softmax", "one_vs_rest"):
            raise ValueError(
                f"strategy must be 'softmax' or 'one_vs_rest'; got {self.strategy!r}."
            )
        n_classes = int(classes.size)
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must lie in [0, 1); got {self.validation_fraction}."
            )
        self._check_threshold(self.threshold)
        if n_classes > 2 and self.threshold != "optimal":
            raise ValueError(
                f"threshold={self.threshold!r} applies to two classes; with {n_classes}, "
                f"predict is the argmax."
            )
        # Class indices in classes_ order; for two classes, 1 is the positive one.
        y01 = np.searchsorted(classes, y_arr).astype(np.int64)

        # A NumPy integer, as scikit-learn tools pass, is taken as the int it is.
        seed = as_seed(self.random_state, "random_state")
        rng = np.random.default_rng(seed)
        # Three independent torch streams, none of them the caller's (#175):
        # the model seeds its initial weights from random_state itself, and
        # training -- the dropout masks and the batch order -- runs on seeds
        # spawned from it, so neither replays the numbers the init drew.  The
        # global RNG is restored afterwards, so a seeded fit leaves it exactly
        # where it was.
        model = self._build(X_arr.shape[1], n_classes, init_seed=seed)
        loss_fn = self._loss(n_classes)
        # BCE-style losses take float targets, cross-entropy class indices.
        as_target = (lambda t: t.float()) if n_classes == 2 else (lambda t: t.long())
        if seed is None:
            dropout_seed: int | None = None
            generator = None
        else:
            dropout_seed, batch_seed = (
                int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(2)
            )
            generator = torch.Generator().manual_seed(batch_seed)

        X_t = torch.from_numpy(X_arr)
        if self.validation_fraction > 0:
            tr, va = self._stratified_holdout(y01, self.validation_fraction, rng)
            val: tuple[torch.Tensor, torch.Tensor] | tuple[None, None] = (
                X_t[va],
                as_target(torch.from_numpy(y01[va])),
            )
        else:
            tr = np.arange(y01.size)
            val = (None, None)

        with seeded_rng(dropout_seed):
            history = train_model(
                model,
                loss_fn,
                torch.optim.Adam(model.parameters(), lr=self.lr),
                X_t[tr],
                as_target(torch.from_numpy(y01[tr])),
                val[0],
                val[1],
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                monitor=self.monitor,
                patience=self.patience,
                generator=generator,
            )
        threshold: float | None
        if n_classes > 2:
            threshold = None
        elif self.threshold == "optimal":
            best = history.best_threshold
            threshold = float(best) if best is not None else 0.5
        else:
            threshold = float(self.threshold)
        model.eval()

        # Fitted attributes are published only once training has succeeded, so a
        # failed refit leaves the estimator on its previous fit rather than on an
        # untrained model that ``check_is_fitted`` would wave through.
        self.classes_ = classes
        self.history_: TrainingHistory = history
        self.threshold_ = threshold
        self.model_ = model
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities, shape ``(n_samples, n_classes)``, columns in ``classes_`` order."""
        check_is_fitted(self, "model_")
        X_arr = validate_data(self, X, dtype=np.float32, reset=False)
        proba = self.model_.predict_proba(torch.from_numpy(X_arr)).numpy().astype(np.float64)
        if proba.ndim == 2:  # multiclass: already one column per class
            return proba
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """
        Labels from ``classes_``: the positive probability thresholded at
        ``threshold_`` for two classes, the argmax of the logits for more.
        """
        check_is_fitted(self, "model_")
        if self.threshold_ is None:
            X_arr = validate_data(self, X, dtype=np.float32, reset=False)
            return self.classes_[self.model_.predict(torch.from_numpy(X_arr)).numpy()]
        positive = self.predict_proba(X)[:, 1]
        return self.classes_[(positive >= self.threshold_).astype(np.intp)]

    # ------------------------------------------------------------------
    # Pickling: the fitted model holds a PennyLane QNode built around a local
    # function, which pickle cannot serialise.  The model is stored as its
    # class, constructor arguments and weights instead, and rebuilt on load.
    def __getstate__(self) -> dict[str, Any]:
        # BaseEstimator returns the live __dict__ on Python 3.11+; copy it so
        # pickling does not strip model_ from the estimator itself.
        state = dict(super().__getstate__())
        model = state.pop("model_", None)
        if model is not None:
            state["_model_state"] = {
                "class_name": type(model).__name__,
                "config": model.get_config(),
                "state_dict": model.state_dict(),
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        saved = state.pop("_model_state", None)
        super().__setstate__(state)
        if saved is not None:
            classes = {
                cls.__name__: cls
                for cls in (
                    HybridBinaryClassifier,
                    ParallelHybridClassifier,
                    MulticlassHybridClassifier,
                )
            }
            # Construction initialises weights from the global torch RNG before
            # load_state_dict overwrites them; fork it so unpickling leaves the
            # caller's random stream untouched.
            with torch.random.fork_rng(devices=[]):
                model = classes[saved["class_name"]](**saved["config"])
            model.load_state_dict(saved["state_dict"])
            model.eval()
            self.model_ = model

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = True
        tags.non_deterministic = self.random_state is None
        return tags


# ---------------------------------------------------------------------------
# Quantum-kernel SVM
# ---------------------------------------------------------------------------

KernelEncoding = Literal["angle", "iqp", "reuploading", "amplitude"]

#: Set by validate_data in fit; restored if the fit fails.
_INPUT_ATTRIBUTES = ("n_features_in_", "feature_names_in_")


class QuantumKernelClassifier(ClassifierMixin, BaseEstimator):
    """
    Support vector classifier on a quantum fidelity kernel (QSVM).

    ``fit`` builds an encoding layer for the width of ``X``, optionally trains
    it by kernel-target alignment, computes the Gram matrix
    ``K[i, j] = |⟨Φ(x_i)|Φ(x_j)⟩|²`` with :func:`hqnn_forge.kernels.quantum_kernel_matrix`
    and fits ``SVC(kernel="precomputed")`` on it.  ``predict`` and friends
    compute the kernel between new samples and the training set -- rows new,
    columns training, the orientation ``SVC`` needs -- from the training
    states cached at ``fit``, so only the new samples are simulated.  So the
    estimator works in ``cross_val_score``, ``GridSearchCV`` and ``Pipeline``
    like any other.

    Parameters
    ----------
    encoding:
        ``"angle"`` (default), ``"iqp"``, ``"reuploading"`` or ``"amplitude"``.
    n_qubits:
        Default: one qubit per feature, or ``max(2, ceil(log2(n_features)))``
        for ``"amplitude"`` (whose features are zero-padded to ``2**n_qubits``).
        With one feature per qubit it must equal ``n_features``.
    n_layers:
        Variational layers of the encoding layer.  For the single-upload
        encoders the ansatz cancels in the fidelity kernel (see
        :mod:`hqnn_forge.kernels`); it matters for ``"reuploading"``, where
        the weights sit between uploads.  Default: 2.
    trainable_input_scaling:
        ``"reuploading"`` only: a trainable scale per upload.  Default: False.
    align_steps, align_lr, align_subset_size:
        With ``align_steps > 0``, train the layer by kernel-target alignment
        before fitting the SVM (:func:`hqnn_forge.kernels.train_kernel_alignment`).
        Only for ``"reuploading"`` (the single-upload ansatz cancels in the
        kernel, so there is nothing to train), two classes and
        ``noise_level=0`` (alignment trains the noiseless kernel).  Default: 0.
    noise_level, noise_position:
        Estimate the kernel under depolarising noise from density matrices;
        ``0`` (default) is the exact state-vector kernel.
    C, class_weight:
        Passed to ``SVC``.
    probability:
        Enable ``predict_proba``: Platt (sigmoid) scaling of the SVM's
        decision values, fitted on out-of-fold predictions by
        ``CalibratedClassifierCV(..., ensemble=False)``, which splits the
        precomputed kernel by rows *and* columns.  (``SVC(probability=True)``,
        which did the same internally, is deprecated from scikit-learn 1.9.)
        ``predict`` and ``decision_function`` stay those of the plain SVM.
        Default: False.
    batch_size:
        Samples simulated at a time, to bound memory.  Default: all.
    random_state:
        Seeds the layer's initialisation, the alignment subsets and ``SVC``,
        from private streams: the caller's global torch RNG is left untouched.

    Attributes
    ----------
    classes_ : ndarray
    n_features_in_ : int
    layer_ : nn.Module
        The (possibly aligned) encoding layer.
    svc_ : sklearn.svm.SVC
    calibrator_ : sklearn.calibration.CalibratedClassifierCV or None
        The probability model, with ``probability=True``.
    alignment_history_ : list of float
        The alignment at each training step; empty without alignment.
    """

    def __init__(
        self,
        encoding: KernelEncoding = "angle",
        n_qubits: int | None = None,
        n_layers: int = 2,
        trainable_input_scaling: bool = False,
        align_steps: int = 0,
        align_lr: float = 0.05,
        align_subset_size: int | None = None,
        noise_level: float = 0.0,
        noise_position: str = "all",
        C: float = 1.0,
        class_weight: dict[Any, float] | str | None = None,
        probability: bool = False,
        batch_size: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.trainable_input_scaling = trainable_input_scaling
        self.align_steps = align_steps
        self.align_lr = align_lr
        self.align_subset_size = align_subset_size
        self.noise_level = noise_level
        self.noise_position = noise_position
        self.C = C
        self.class_weight = class_weight
        self.probability = probability
        self.batch_size = batch_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Everything the fitted model depends on is copied into _fit_config at fit
    # (batch_size, which only bounds memory, stays live),
    # so a set_params after fit cannot change how predict encodes new samples
    # or how unpickling rebuilds layer_.
    def _config(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "trainable_input_scaling": self.trainable_input_scaling,
            "noise_level": self.noise_level,
            "noise_position": self.noise_position,
        }

    @staticmethod
    def _build_layer(config: dict[str, Any], n_features: int) -> torch.nn.Module:
        encoding = config["encoding"]
        if encoding not in ("angle", "iqp", "reuploading", "amplitude"):
            raise ValueError(
                f"encoding must be 'angle', 'iqp', 'reuploading' or 'amplitude'; got {encoding!r}."
            )
        if config["trainable_input_scaling"] and encoding != "reuploading":
            raise ValueError("trainable_input_scaling applies to encoding='reuploading' only.")
        common: dict[str, Any] = dict(
            n_layers=config["n_layers"], device_name="default.qubit", diff_method="backprop"
        )
        if encoding == "amplitude":
            n_qubits = (
                config["n_qubits"]
                if config["n_qubits"] is not None
                else max(2, math.ceil(math.log2(n_features)))
            )
            return AmplitudeEncodingLayer(n_qubits=n_qubits, n_features=n_features, **common)
        n_qubits = config["n_qubits"] if config["n_qubits"] is not None else n_features
        if n_qubits != n_features:
            raise ValueError(
                f"encoding={encoding!r} takes one feature per qubit: n_qubits={n_qubits} "
                f"but X has {n_features} features.  Reduce the features (e.g. PCA) or use "
                f"encoding='amplitude'."
            )
        if encoding == "angle":
            return QuantumEncodingLayer(n_qubits=n_qubits, **common)
        if encoding == "iqp":
            return IQPEncodingLayer(n_qubits=n_qubits, **common)
        return DataReuploadingLayer(
            n_qubits=n_qubits, trainable_input_scaling=config["trainable_input_scaling"], **common
        )

    def _encode(
        self, X: torch.Tensor, layer: torch.nn.Module, config: dict[str, Any]
    ) -> torch.Tensor:
        """States, or density matrices under noise, of ``X`` through ``layer``."""
        if config["noise_level"]:
            return kernels.encoded_density_matrices(
                X,
                layer,
                noise_level=config["noise_level"],
                noise_position=config["noise_position"],
                batch_size=self.batch_size,
            )
        return kernels.encoded_states(X, layer, batch_size=self.batch_size)

    @staticmethod
    def _kernel(
        config: dict[str, Any], encoded_x: torch.Tensor, encoded_y: torch.Tensor | None = None
    ) -> np.ndarray:
        K = (
            kernels.kernel_from_density_matrices(encoded_x, encoded_y)
            if config["noise_level"]
            else kernels.kernel_from_states(encoded_x, encoded_y)
        )
        return K.detach().numpy()

    # ------------------------------------------------------------------
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> QuantumKernelClassifier:
        """Build (and optionally align) the layer, then fit the SVM on the Gram matrix."""
        # validate_data resets n_features_in_ and feature_names_in_, the latter
        # even before it rejects X; put them back if the fit fails, so a failed
        # refit leaves the previous model whole.
        previous = {k: v for k, v in vars(self).items() if k in _INPUT_ATTRIBUTES}
        try:
            X_arr, y_arr = validate_data(self, X, y, dtype=np.float64)
            return self._fit(X_arr, y_arr)
        except BaseException:
            for name in _INPUT_ATTRIBUTES:
                if name in previous:
                    setattr(self, name, previous[name])
                else:
                    self.__dict__.pop(name, None)
            raise

    def _fit(self, X_arr: np.ndarray, y_arr: np.ndarray) -> QuantumKernelClassifier:
        # Every check that needs no simulation runs before the first circuit.
        check_classification_targets(y_arr)
        classes, counts = np.unique(y_arr, return_counts=True)
        if classes.size < 2:
            # "1 class" is the wording scikit-learn's conformance checks match on.
            raise ValueError(
                f"QuantumKernelClassifier needs at least two classes; got {classes.size} class."
            )
        if self.probability and counts.min() < 2:
            raise ValueError(
                "probability=True calibrates on cross-validated predictions and needs "
                "at least two samples of every class."
            )
        if self.align_steps < 0:
            raise ValueError(f"align_steps must be >= 0; got {self.align_steps}.")
        if self.align_steps:
            if classes.size != 2:
                raise ValueError(
                    f"kernel-target alignment is defined for two classes; got {classes.size}."
                )
            if self.encoding != "reuploading":
                # The single-upload ansatz cancels in the fidelity kernel, so the
                # alignment gradient is zero (see kernels.train_kernel_alignment).
                raise ValueError(
                    f"kernel-target alignment only changes the kernel of "
                    f"encoding='reuploading'; got encoding={self.encoding!r}."
                )
            if self.noise_level:
                raise ValueError(
                    "kernel-target alignment trains the noiseless kernel, so it cannot be "
                    "combined with noise_level > 0 yet."
                )
        if X_arr.shape[1] < 2:
            # "n_features = 1" is the wording scikit-learn's conformance checks match on.
            raise ValueError(
                f"QuantumKernelClassifier needs at least two features; got n_features = "
                f"{X_arr.shape[1]}.  Angle, IQP and re-uploading need one qubit per feature "
                f"and two qubits at least, and amplitude encoding normalises a single "
                f"feature to ±|0⟩, a constant kernel."
            )
        config = self._config()
        # A NumPy integer, as scikit-learn tools pass, is taken as the int it is.
        # The layer is initialised inside seeded_rng, so a seeded fit leaves the
        # caller's global torch RNG exactly where it was (#175).
        seed = as_seed(self.random_state, "random_state")
        with seeded_rng(seed):
            layer = self._build_layer(config, X_arr.shape[1])
        X_t = torch.tensor(X_arr)
        history: list[float] = []
        if self.align_steps:
            labels = torch.from_numpy(np.searchsorted(classes, y_arr))
            generator = torch.Generator().manual_seed(seed) if seed is not None else None
            history = kernels.train_kernel_alignment(
                layer,
                X_t,
                labels,
                steps=self.align_steps,
                lr=self.align_lr,
                subset_size=self.align_subset_size,
                generator=generator,
            )
        encoded = self._encode(X_t, layer, config)
        K = self._kernel(config, encoded)

        def svm() -> SVC:
            return SVC(
                kernel="precomputed",
                C=self.C,
                class_weight=self.class_weight,
                random_state=seed,
            )

        calibrator = None
        if self.probability:
            calibrator = CalibratedClassifierCV(
                svm(), method="sigmoid", ensemble=False, cv=min(5, int(counts.min()))
            ).fit(K, y_arr)
            # With ensemble=False the calibrator refits a clone of svm() on all of
            # K, which is exactly the plain SVM; reuse it rather than fit it twice.
            svc = calibrator.calibrated_classifiers_[0].estimator
        else:
            svc = svm().fit(K, y_arr)
        # Published together, after everything that can fail.
        self._fit_config = config
        self.layer_ = layer
        self._train_encoded = encoded
        self.svc_ = svc
        self.calibrator_ = calibrator
        self.classes_ = svc.classes_
        self.alignment_history_ = history
        return self

    def _test_kernel(self, X: npt.ArrayLike) -> np.ndarray:
        check_is_fitted(self, "svc_")
        X_arr = validate_data(self, X, dtype=np.float64, reset=False)
        encoded = self._encode(torch.tensor(X_arr), self.layer_, self._fit_config)
        return self._kernel(self._fit_config, encoded, self._train_encoded)

    # The kernel is computed before svc_ is touched, so an unfitted estimator
    # raises NotFittedError (from check_is_fitted), not AttributeError.
    def decision_function(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """``SVC.decision_function`` on the kernel between ``X`` and the training set."""
        K = self._test_kernel(X)
        return self.svc_.decision_function(K)  # type: ignore[no-any-return]

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Labels from ``classes_``."""
        K = self._test_kernel(X)
        return self.svc_.predict(K)  # type: ignore[no-any-return]

    # Fitted, it follows the fit (calibrator_), not a later set_params.
    @available_if(
        lambda self: (
            self.calibrator_ is not None if hasattr(self, "calibrator_") else self.probability
        )
    )
    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Platt-scaled probabilities; only present with ``probability=True``, as in ``SVC``."""
        K = self._test_kernel(X)
        return self.calibrator_.predict_proba(K)  # type: ignore[no-any-return,union-attr]

    # ------------------------------------------------------------------
    # Pickling: layer_ holds a PennyLane QNode built around a local function,
    # which pickle cannot serialise.  It is stored as its weights and rebuilt
    # from the configuration saved at fit on load.
    def __getstate__(self) -> dict[str, Any]:
        state = dict(super().__getstate__())
        layer = state.pop("layer_", None)
        if layer is not None:
            state["_layer_state"] = layer.state_dict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        saved = state.pop("_layer_state", None)
        super().__setstate__(state)
        if saved is not None:
            # Construction draws initial weights that load_state_dict then
            # overwrites; fork the RNG so unpickling leaves the caller's alone.
            with torch.random.fork_rng(devices=[]):
                layer = self._build_layer(self._fit_config, self.n_features_in_)
            layer.load_state_dict(saved)
            self.layer_ = layer

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.non_deterministic = self.random_state is None
        return tags
