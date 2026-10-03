"""
hqnn_forge.preprocessing.pca_normalizer
==========================================
Classical pre-processing: PCA dimensionality reduction + per-feature
standardisation, implemented in **pure NumPy** (no scikit-learn runtime
dependency).

Workflow
--------
1. Fit on training data     → ``PCANormalizer.fit(X_train)``
2. Transform train/test     → ``PCANormalizer.transform(X)`` → ``torch.Tensor``
3. Pass to QuantumEncLayer  → features should lie in ``[-π, π]`` after scaling

Encoding Scaling
----------------
Raw PCA components are standardised (zero mean, unit variance) and then
optionally rescaled into ``[-π, π]`` via a tanh squeeze:

    x̂_i = tanh(x_std_i) * π

This keeps all features within the valid range for angle embedding while
preventing wrap-around aliasing for large outliers.

Notes
-----
* Eigendecomposition uses ``numpy.linalg.eigh`` (symmetric covariance matrix),
  which is numerically more stable than ``numpy.linalg.eig`` for this use case.
* Only the top ``n_components`` eigenvectors (by descending eigenvalue) are kept.
* Eigenvector signs are canonicalised so that the largest-magnitude entry of
  each component is positive.  ``eigh`` gives no guarantee about which of the
  two valid signs it returns, so without this the fitted basis -- and every
  encoded feature -- would differ between LAPACK builds and platforms.  The
  guarantee is conditional; ``PCANormalizer.components_`` states the conditions.
"""

from __future__ import annotations

import operator
import warnings

import numpy as np
import numpy.typing as npt
import torch

# Eigenvalue gap, relative to the largest eigenvalue, below which fit warns that the retained basis is not
# reproducible.  Chosen from measurement, not intuition: perturbing the
# covariance by 1e-16 relative (the scale on which two LAPACK builds disagree
# about the same matrix, i.e. relative to the largest eigenvalue) rotates the affected components by ~0.002 deg at a gap
# of 1e-12, ~3 deg at 1e-14 and ~20 deg when exactly degenerate, and by nothing
# measurable at 1e-8.  Rotation scales as ||E|| / gap, and ||E|| scales with
# the largest eigenvalue, so the gap is measured against that rather than
# against the pair itself -- otherwise a close pair of small eigenvalues would
# look well separated.  1e-12 sits safely inside the flat region.
EIGENVALUE_GAP_WARN: float = 1e-12


class PCANormalizer:
    """
    Fit/apply PCA + per-feature standardisation without scikit-learn.

    The fitted arrays (``mean_``, ``components_``, ``explained_variance_`` and
    ``std_``) are set by ``fit`` and absent until then, as in scikit-learn;
    ``is_fitted_`` exists from construction.

    Parameters
    ----------
    n_components:
        Number of principal components to retain.  Must be an integer ≥ 1
        (``int`` or a NumPy integer; ``bool`` and floats, even integral ones
        such as ``3.0``, are rejected rather than coerced); ``fit`` raises
        ``ValueError`` otherwise.  Default: 8.
    scale_to_pi:
        If ``True`` (default), rescale standardised components into ``[-π, π]``
        via ``tanh(x) * π`` before returning.  Ensures valid angle-embedding
        range without hard clipping.

    Attributes
    ----------
    mean_ : np.ndarray, shape (n_features,)
        Per-feature mean computed during ``fit``.
    components_ : np.ndarray, shape (n_components, n_features)
        Principal component matrix (rows = eigenvectors, sorted by descending
        explained variance).  Each row's sign is canonicalised so its
        largest-magnitude entry is positive, making ``components_`` -- and
        therefore ``transform`` -- a deterministic function of the input data
        rather than of the LAPACK build.  This is part of the public contract,
        and holds whenever the eigenvalues are well separated.  Entries tied for
        largest magnitude are resolved by column order, so mirrored feature
        pairs are covered; near-degenerate eigenvalues, however, leave the basis
        itself build-dependent, which no sign convention can repair.  ``fit``
        checks for this and emits a ``RuntimeWarning`` when two consecutive
        eigenvalues among the retained ones (or at the cutoff) differ by less
        than ``EIGENVALUE_GAP_WARN`` times the largest eigenvalue, which in
        practice means exactly degenerate up to rounding.
        The convention is the one scikit-learn's ``svd_flip`` applies with
        ``u_based_decision=False``, reimplemented here rather than depended on.
    explained_variance_ : np.ndarray, shape (n_components,)
        Eigenvalues corresponding to retained components.
    std_ : np.ndarray, shape (n_components,)
        Per-component standard deviation (computed on training projections).
    is_fitted_ : bool
        ``True`` after ``fit`` has been called.

    Notes
    -----
    Input is converted with ``np.asarray``, so ``float64`` input is used without
    copying and ``fit``/``transform`` may hold a view of the caller's buffer for
    the duration of the call.  Neither method writes into it, and no view of it
    is kept in the fitted attributes, so the caller is free to modify or discard
    the array afterwards.

    Examples
    --------
    >>> import numpy as np
    >>> from hqnn_forge.preprocessing import PCANormalizer
    >>> rng = np.random.default_rng(42)
    >>> X_train = rng.standard_normal((1000, 30))  # 1000 samples, 30 raw features
    >>> pca = PCANormalizer(n_components=8)
    >>> pca.fit(X_train)  # returns the fitted normaliser
    PCANormalizer(n_components=8, scale_to_pi=True, status=fitted)
    >>> X_enc = pca.transform(X_train)  # torch.Tensor, shape (1000, 8)
    >>> X_enc.shape
    torch.Size([1000, 8])
    """

    # Set by fit() and absent until then, as in scikit-learn; is_fitted_ guards
    # every read, so they are typed as always present.
    mean_: npt.NDArray[np.float64]
    components_: npt.NDArray[np.float64]
    explained_variance_: npt.NDArray[np.float64]
    std_: npt.NDArray[np.float64]

    def __init__(
        self,
        n_components: int | np.integer = 8,
        *,
        scale_to_pi: bool = True,
    ) -> None:
        self.n_components = n_components
        self.scale_to_pi = scale_to_pi

        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    def fit(self, X: npt.ArrayLike) -> PCANormalizer:
        """
        Compute PCA basis and per-component statistics on training data.

        Parameters
        ----------
        X:
            Training data array-like of shape ``(n_samples, n_features)``.
            ``n_features`` must be ≥ 2 and ≥ ``n_components``, and
            ``n_samples`` must be > ``n_components``.  Not copied when already
            ``float64``, and never modified.

        Returns
        -------
        self
            The fitted transformer (for method chaining).

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, if ``n_features < 2`` (a single column has no
            covariance to decompose), if ``n_components`` is not an integer or
            is ``< 1``, if ``n_features < n_components``, or if
            ``n_samples <= n_components``
            (centred data then has rank below ``n_components``, so some
            components have zero variance).  Also if the centred data has rank
            below ``n_components`` for any other reason -- collinear features,
            duplicated samples, constant data -- which is detected from
            ``numpy.linalg.matrix_rank`` of the centred data, before any
            component is retained.

        Notes
        -----
        ``fit`` is all-or-nothing: every check above raises before the first
        fitted attribute is assigned, so a rejected fit leaves the instance
        exactly as it found it.  An instance that was already fitted keeps that
        fit and stays usable; one that was not stays unfitted.
        """
        return self._fit(X, stacklevel=3)

    def _fit(self, X: npt.ArrayLike, stacklevel: int) -> PCANormalizer:
        # Body of fit, shared with fit_transform so the degeneracy warning's
        # stacklevel points at the caller of either public method.
        # asarray, not array: float64 input is used as-is rather than copied, so
        # X_arr may share memory with the caller.  Never write into it in place.
        X_arr: npt.NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        self._check_2d(X_arr)

        n_samples, n_features = X_arr.shape
        # A single column clears every check below and then fails inside
        # numpy: np.cov(..., rowvar=False) returns a 0-d array for it, which
        # eigh rejects with a message about the array, not about the data
        if n_features < 2:
            raise ValueError(
                f"n_features={n_features} < 2.  PCA needs at least two features "
                f"to have a covariance to decompose; a single feature has "
                f"nothing to project."
            )

        # Checked here rather than in __init__ because the attribute can be
        # reassigned afterwards.  The comparisons below all accept a float, so
        # a non-integer would clear them and fail on the top-k slice with a
        # TypeError about slice indices.  operator.index accepts int and NumPy
        # integers and rejects floats, integral ones included: 3.0 is not
        # coerced, the caller casts.  bool is an int subclass and would keep
        # one component for True; nobody means that, so it is rejected too.
        # Everything below uses the plain int it returns, never the attribute:
        # a fixed-width NumPy integer would wrap in the arithmetic of the error
        # messages (np.uint8(255) + 1 == 0)
        if isinstance(self.n_components, bool):
            # a value check, not a type check: bool is an int subclass
            raise ValueError(  # noqa: TRY004
                f"n_components={self.n_components!r} is a bool, not an integer.  "
                f"Provide the number of components to retain."
            )
        try:
            n_components = operator.index(self.n_components)
        except TypeError:
            raise ValueError(
                f"n_components={self.n_components!r} is not an integer.  "
                f"Provide an int (or NumPy integer); a float such as 3.0 is "
                f"rejected rather than coerced."
            ) from None
        # Values <= 0 pass both shape checks below and then silently slice off
        # components from the end (-1 keeps all but one)
        if n_components < 1:
            raise ValueError(
                f"n_components={n_components} < 1.  Provide a positive "
                f"number of components to retain."
            )
        if n_features < n_components:
            raise ValueError(
                f"n_features={n_features} < n_components={n_components}.  "
                f"Reduce n_components or provide higher-dimensional data."
            )
        # Centred data has rank <= n_samples - 1, so fewer rows leave some
        # kept components with zero variance
        if n_samples <= n_components:
            raise ValueError(
                f"n_samples={n_samples} <= n_components={n_components}.  "
                f"Reduce n_components or provide at least "
                f"{n_components + 1} samples."
            )

        # 1. Centre the data.  mean_ is assigned only once the rank check below
        # has passed -- together with the checks above raising before any other
        # attribute is written, that makes fit all-or-nothing: a rejected fit
        # leaves the instance untouched, so a failed re-fit keeps the previous
        # fit intact and usable rather than half-replacing it
        mean = X_arr.mean(axis=0)
        X_centered = X_arr - mean

        # 2. Covariance matrix (unbiased estimator, ddof=1)
        cov = np.cov(X_centered, rowvar=False)  # shape (n_features, n_features)

        # 3. Eigendecomposition (eigh: exploit symmetry for stability + speed)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 4. Sort by descending eigenvalue and keep top-k components
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # 5. Reject data whose centred rank is below n_components.  The extra
        # components have a numerically zero eigenvalue, their directions are an
        # arbitrary basis of the null space, and std_ falls back to the 1e-8
        # added for division safety -- so transform divides new data's
        # projection on them by 1e-8 and returns ~1e8 (or a saturated +-pi).
        # Nothing about that is detectable from the output, hence the check.
        #
        # The rank is read off the centred data rather than off the eigenvalues
        # above, because cov squares the singular values and so halves the
        # digits available to separate signal from the noise floor.  A tolerance
        # relative to eigenvalues[0] == s_max**2 squares the condition number
        # with it, rejecting genuinely full-rank data whose feature scales
        # differ by more than ~1e8; and taking sqrt afterwards does not undo
        # that, since a null direction's eigenvalue sits near lambda_max * eps,
        # whose root is s_max * sqrt(eps) -- ~1e-8 relative, far above any
        # eps-scaled threshold.  matrix_rank thresholds the singular values
        # themselves, which is where a relative tolerance belongs.  An absolute
        # one is not an option either: it would reject full-rank data that
        # merely has a small overall scale.
        rank = int(np.linalg.matrix_rank(X_centered))
        if rank < n_components:
            remedy = (
                "Provide data that varies: every feature is constant, so the "
                "centred data is all zeros."
                if rank == 0
                else f"Reduce n_components to at most {rank}, or provide data "
                f"of higher rank (collinear features and duplicated samples "
                f"both lower it)."
            )
            raise ValueError(
                f"centred data has rank {rank} < n_components="
                f"{n_components}, so components {rank}.."
                f"{n_components - 1} have zero variance and transform "
                f"would divide their projections by the 1e-8 epsilon.  {remedy}"
            )
        kept = eigenvalues[:n_components]

        # 5b. Warn when the retained basis is not reproducible.  Within a
        # degenerate eigenspace eigh may return any orthonormal basis, and a
        # rotation is not a sign flip, so step 6 cannot repair it; degeneracy
        # *at* the cutoff additionally makes it arbitrary which component is
        # kept at all.  Hence the gaps between consecutive eigenvalues among the
        # kept ones and the first excluded one are all checked.  Gaps are
        # relative to eigenvalues[0], the scale of the rounding that rotates the
        # basis (see EIGENVALUE_GAP_WARN); the rank check above guarantees it is
        # > 0.  The first offending pair is reported, so that n_components=first
        # keeps only well-separated eigenvalues, cutoff included.  A warning and
        # not an error: the fit is still a valid PCA, it just is not the same
        # one on every platform, which only matters to callers who rely on the
        # components_ contract.
        kept_and_next = eigenvalues[: n_components + 1]
        if kept_and_next.shape[0] > 1:
            gaps = (kept_and_next[:-1] - kept_and_next[1:]) / eigenvalues[0]
            degenerate = np.flatnonzero(gaps < EIGENVALUE_GAP_WARN)
            if degenerate.size:
                first = int(degenerate[0])
                where = (
                    "at the n_components cutoff, so which component is retained is arbitrary"
                    if first == n_components - 1
                    else "among the retained components"
                )
                remedy = (
                    f"Reduce n_components to {first}, or accept"
                    if first > 0
                    else "No smaller n_components avoids this; accept"
                )
                warnings.warn(
                    f"eigenvalues {first} and {first + 1} are degenerate "
                    f"(gap {gaps[first]:.1e} of the largest eigenvalue < "
                    f"{EIGENVALUE_GAP_WARN:.0e}) {where}: components_ and "
                    f"transform are not reproducible across platforms for this "
                    f"data.  {remedy} a basis that depends on the LAPACK build.",
                    RuntimeWarning,
                    stacklevel=stacklevel,
                )

        self.mean_ = mean
        self.explained_variance_ = kept
        # rows = components (shape: n_components × n_features)
        components = eigenvectors[:, :n_components].T

        # 6. Canonicalise the sign of each component.  eigh returns eigenvectors
        # up to an arbitrary sign, so another LAPACK build may hand back a
        # component negated -- components_ and the encoded features would not be
        # reproducible across platforms.  Convention (matching scikit-learn's
        # svd_flip): the largest-magnitude entry of each component is positive.
        #
        # Which entry that is has to be decided with a tolerance, because exact
        # ties are structural rather than a coincidence of continuous data: if
        # two columns are exact mirrors (x_j == -x_i, as in a two-level one-hot,
        # a share/1-share pair, or a +/- sensor pair), then (e_i + e_j)/sqrt(2)
        # is an exact null eigenvector of the covariance, so every retained
        # component satisfies v_j == -v_i to the last ulp.  Where that pair
        # carries the largest entry, a strict argmax keys the whole row's sign
        # on ~1e-16 rounding noise.  Taking the first entry within a relative
        # tolerance of the maximum instead makes the choice a function of column
        # order alone; where the maximum is unique it selects what argmax would.
        # np.sign cannot return 0 here: the largest-magnitude entry of a
        # unit-norm vector is at least 1/sqrt(n_features).
        magnitudes = np.abs(components)
        tied = magnitudes >= magnitudes.max(axis=1, keepdims=True) * (1 - 1e-12)
        leading = tied.argmax(axis=1)
        signs = np.sign(components[np.arange(components.shape[0]), leading])
        # Out-of-place: components is a non-contiguous view over the full
        # (n_features, n_features) eigenvector matrix, which this also drops
        self.components_ = components * signs[:, np.newaxis]

        # 7. Project training data → compute per-component std for standardisation
        projections = X_centered @ self.components_.T  # (n_samples, n_components)
        self.std_ = projections.std(axis=0, ddof=1) + 1e-8  # avoid div-by-zero

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    def transform(self, X: npt.ArrayLike) -> torch.Tensor:
        """
        Project and standardise (and optionally scale to [-π, π]).

        Only statistics learned in ``fit`` (``mean_``, ``components_``,
        ``std_``) are used, so each row is encoded independently of the other
        rows in ``X``.  Standardised output has zero mean and unit variance on
        the training data; other data keeps its offset from the training
        distribution.

        Parameters
        ----------
        X:
            Data array-like of shape ``(n_samples, n_features)``.
            Must have the same ``n_features`` as the training data.
            Not copied when already ``float64``, and never modified.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape ``(n_samples, n_components)``,
            dtype ``float32``, values in ``[-π, π]`` if ``scale_to_pi=True``.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        ValueError
            If ``X`` is not 2-D, or if its feature dimension doesn't match
            training data.
        """
        self._check_is_fitted()

        # asarray, not array: float64 input is used as-is rather than copied, so
        # X_arr may share memory with the caller.  Never write into it in place.
        X_arr: npt.NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        self._check_2d(X_arr)

        if X_arr.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Input has {X_arr.shape[1]} features but PCANormalizer was "
                f"fitted on {self.mean_.shape[0]} features."
            )

        # Centre → project → standardise, using training statistics only (no
        # per-batch mean), so each row's encoding is independent of the batch
        X_centered = X_arr - self.mean_
        projections = X_centered @ self.components_.T
        standardised = projections / self.std_

        if self.scale_to_pi:
            # Soft-clip to (-π, π) preserving relative magnitudes of outliers
            standardised = np.tanh(standardised) * np.pi

        return torch.tensor(standardised, dtype=torch.float32)

    # ------------------------------------------------------------------
    def fit_transform(self, X: npt.ArrayLike) -> torch.Tensor:
        """
        Fit and transform in a single call (convenience method).

        Parameters
        ----------
        X:
            Training data, shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Transformed tensor, shape ``(n_samples, n_components)``.
        """
        return self._fit(X, stacklevel=3).transform(X)

    # ------------------------------------------------------------------
    @property
    def explained_variance_ratio_(self) -> npt.NDArray[np.float64]:
        """
        Fraction of total variance explained by each retained component.

        Returns
        -------
        np.ndarray, shape (n_components,)
        """
        self._check_is_fitted()
        total_var = self.explained_variance_.sum()
        return self.explained_variance_ / total_var

    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("PCANormalizer is not fitted.  Call .fit(X_train) first.")

    # ------------------------------------------------------------------
    @staticmethod
    def _check_2d(X_arr: npt.NDArray[np.float64]) -> None:
        # 1-D input is ambiguous (one sample or one feature), so reject it
        # rather than guess a reshape
        if X_arr.ndim != 2:
            # The reshape hint only fits 1-D input; for higher ndim it would mislead
            hint = "  For a single sample, use X.reshape(1, -1)." if X_arr.ndim == 1 else ""
            raise ValueError(
                f"Expected 2-D input of shape (n_samples, n_features), got "
                f"shape {X_arr.shape}.{hint}"
            )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"PCANormalizer("
            f"n_components={self.n_components}, "
            f"scale_to_pi={self.scale_to_pi}, "
            f"status={status})"
        )
