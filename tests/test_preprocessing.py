"""
tests/test_preprocessing.py
=============================
Unit tests for hqnn_forge.preprocessing.PCANormalizer.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch

from hqnn_forge.preprocessing import PCANormalizer
from hqnn_forge.preprocessing.pca_normalizer import EIGENVALUE_GAP_WARN

N_SAMPLES = 100
N_FEATURES = 12
N_COMPONENTS = 4


@pytest.fixture
def training_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_SAMPLES, N_FEATURES))


@pytest.fixture
def fitted_pca(training_data: np.ndarray) -> PCANormalizer:
    # Fitted on the training_data fixture rather than an inlined copy of it:
    # test_negated_eigenvectors_give_identical_fit compares a fit on
    # training_data against this one, so the two must be the same array by
    # construction and not by both happening to use seed 0 and the same shape
    pca = PCANormalizer(n_components=N_COMPONENTS, scale_to_pi=True)
    pca.fit(training_data)
    return pca


class TestFitAttributes:
    def test_is_fitted(self, fitted_pca: PCANormalizer) -> None:
        assert fitted_pca.is_fitted_ is True

    def test_components_shape(self, fitted_pca: PCANormalizer) -> None:
        assert fitted_pca.components_.shape == (N_COMPONENTS, N_FEATURES)

    def test_explained_variance_sorted_descending(self, fitted_pca: PCANormalizer) -> None:
        ev = fitted_pca.explained_variance_
        for i in range(len(ev) - 1):
            assert ev[i] >= ev[i + 1]

    def test_components_have_positive_leading_entry(self, fitted_pca: PCANormalizer) -> None:
        # The documented sign convention, asserted directly: it is what makes
        # components_ and the pinned golden values reproducible across platforms.
        # A strict argmax states the property independently of how fit computes
        # it, which is sound precisely because the fixture has one clearly
        # largest entry per component -- guarded by
        # TestGoldenFixtureIsWellConditioned::test_sign_convention_is_unambiguous
        components = fitted_pca.components_
        leading = np.abs(components).argmax(axis=1)
        assert np.all(components[np.arange(components.shape[0]), leading] > 0)

    def test_negated_eigenvectors_give_identical_fit(
        self,
        fitted_pca: PCANormalizer,
        training_data: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The signs cannot be made to differ on one machine, so stand in for a
        # LAPACK build that returns the other (equally valid) eigenvectors.  This
        # is the property the convention exists for; the assertion above only
        # checks the convention is self-consistent, not that it is reached from
        # both starting points.
        #
        # The flip is alternating, not uniform: negating *every* eigenvector is
        # undone by symmetry by any global sign rule, so a per-row bug -- one
        # component's sign broadcast to all rows -- would leave a uniform flip
        # passing.  Flipping a subset is what makes this assert the per-component
        # property.
        real_eigh = np.linalg.eigh

        def flipped_eigh(a):  # type: ignore[no-untyped-def]
            eigenvalues, eigenvectors = real_eigh(a)
            pattern = np.ones(eigenvectors.shape[1])
            pattern[::2] = -1.0
            return eigenvalues, eigenvectors * pattern  # columns are eigenvectors

        monkeypatch.setattr(np.linalg, "eigh", flipped_eigh)
        flipped = PCANormalizer(n_components=N_COMPONENTS, scale_to_pi=True).fit(training_data)

        np.testing.assert_allclose(flipped.components_, fitted_pca.components_)

    def test_mirrored_feature_pair_keeps_a_stable_sign(self) -> None:
        # A tie for the largest-magnitude entry is structural, not a freak of
        # continuous data.  With x_5 == -x_0 exactly, (e_0 + e_5)/sqrt(2) is a
        # null eigenvector of the covariance, so every retained component has
        # v_5 == -v_0 to the last ulp; scaling column 0 up puts that pair in the
        # leading position, where a strict argmax decides the row's sign on
        # rounding noise.  Reordering rows perturbs the covariance by far less
        # than a different LAPACK build would, so a sign that survives it is the
        # weaker of the two claims -- and the strict argmax did not survive it.
        rng = np.random.default_rng(0)
        base = rng.standard_normal((50, 5))
        base[:, 0] *= 3.0
        X = np.column_stack([base, -base[:, 0]])

        reference = PCANormalizer(n_components=3, scale_to_pi=True).fit(X).components_

        # Guard the premise: if the fixture ever stops producing a tie, this
        # test silently stops covering the tie-break rather than failing
        magnitudes = np.sort(np.abs(reference), axis=1)
        margins = (magnitudes[:, -1] - magnitudes[:, -2]) / magnitudes[:, -1]
        assert margins.min() < 1e-12, (
            f"fixture no longer has a component whose two largest entries are "
            f"tied (smallest relative margin {margins.min():.2e}), so it no "
            f"longer exercises the tie-break"
        )

        for seed in range(8):
            perm = np.random.default_rng(seed).permutation(X.shape[0])
            permuted = PCANormalizer(n_components=3, scale_to_pi=True).fit(X[perm]).components_
            np.testing.assert_allclose(permuted, reference, atol=1e-8)


class TestTransformOutput:
    def test_output_shape(self, fitted_pca: PCANormalizer, training_data: np.ndarray) -> None:
        result = fitted_pca.transform(training_data)
        assert result.shape == (N_SAMPLES, N_COMPONENTS)

    def test_output_dtype(self, fitted_pca: PCANormalizer, training_data: np.ndarray) -> None:
        result = fitted_pca.transform(training_data)
        assert result.dtype == torch.float32

    def test_matches_golden_values(
        self, fitted_pca: PCANormalizer, held_out_data: np.ndarray
    ) -> None:
        # Pins the actual numbers, not just the shape, so a refactor of the
        # conversion or projection path cannot quietly change the encoding.
        # Signs need no normalisation here: fit canonicalises them, so these
        # values are pinned as-is to 1e-6 on any platform.
        expected = torch.tensor(
            [
                [0.26675245, -1.63643660, 2.29188750, 3.09272840],
                [3.00168420, 0.01153241, 2.76722460, 2.86429880],
                [2.91354400, 1.08601160, 3.04996010, -1.35583290],
            ]
        )
        result = fitted_pca.transform(held_out_data)[:3]
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=0.0)


class TestGoldenFixtureIsWellConditioned:
    """
    test_matches_golden_values and the sign assertions above both assume the
    fixture is unambiguous: well-separated eigenvalues, and one clearly largest
    entry per component.  Assert that directly, so a future change to the
    fixture fails here with a stated reason rather than as an inscrutable golden
    mismatch on someone else's platform.

    fit itself no longer needs the second assumption -- it resolves exact ties by
    column order, which test_mirrored_feature_pair_keeps_a_stable_sign covers.
    The assertions that restate the convention with a strict argmax still do.

    The thresholds are canaries, not descriptions of the current fixture: they
    sit several times below what it actually has, so an innocuous tweak won't
    trip them, and ~12 orders of magnitude above the ~1e-15 spread between
    LAPACK builds, so tripping one means real ambiguity rather than noise.
    """

    def test_eigenvalues_are_well_separated(self, training_data: np.ndarray) -> None:
        # Includes the gap at the cutoff (ev[n_components-1] -> ev[n_components]):
        # degeneracy there permutes which components are kept at all, and eigh may
        # return an arbitrarily rotated basis within a degenerate subspace —
        # neither of which sign normalisation can repair
        eigenvalues = np.linalg.eigh(np.cov(training_data, rowvar=False))[0][::-1]
        kept_and_next = eigenvalues[: N_COMPONENTS + 1]
        gaps = (kept_and_next[:-1] - kept_and_next[1:]) / kept_and_next[:-1]
        assert gaps.min() > 0.01, (
            f"fixture eigenvalues are nearly degenerate (smallest relative gap "
            f"{gaps.min():.2%}), so the component basis is not stable across "
            f"platforms and the golden values cannot be pinned"
        )

    def test_sign_convention_is_unambiguous(self, fitted_pca: PCANormalizer) -> None:
        magnitudes = np.sort(np.abs(fitted_pca.components_), axis=1)
        margins = (magnitudes[:, -1] - magnitudes[:, -2]) / magnitudes[:, -1]
        assert margins.min() > 0.001, (
            f"a component's two largest entries are nearly equal (smallest "
            f"relative margin {margins.min():.2%}); fit resolves exact ties by "
            f"column order, but the assertions that restate the convention with "
            f"a strict argmax need a clear winner to key on"
        )


class TestScaleToPi:
    def test_values_within_pi(self, fitted_pca: PCANormalizer, training_data: np.ndarray) -> None:
        result = fitted_pca.transform(training_data)
        assert result.min().item() >= -math.pi - 1e-6
        assert result.max().item() <= math.pi + 1e-6


@pytest.fixture
def held_out_data() -> np.ndarray:
    # Different seed and a mean offset, so held-out rows differ from the training mean.
    # Kept small so tanh stays out of saturation and assertions on scaled output bite.
    rng = np.random.default_rng(1)
    return rng.standard_normal((30, N_FEATURES)) + 1.0


class TestTransformUsesFitStatistics:
    """transform must depend only on statistics learned in fit, never on the batch."""

    def test_matches_manual_projection(
        self, fitted_pca: PCANormalizer, held_out_data: np.ndarray
    ) -> None:
        z = ((held_out_data - fitted_pca.mean_) @ fitted_pca.components_.T) / fitted_pca.std_
        expected = torch.tensor(np.tanh(z) * np.pi, dtype=torch.float32)
        torch.testing.assert_close(fitted_pca.transform(held_out_data), expected)

    def test_single_row_matches_row_in_batch(
        self, fitted_pca: PCANormalizer, held_out_data: np.ndarray
    ) -> None:
        batch = fitted_pca.transform(held_out_data)
        for i in range(len(held_out_data)):
            torch.testing.assert_close(fitted_pca.transform(held_out_data[i : i + 1])[0], batch[i])

    def test_held_out_batch_independent_of_split(
        self, fitted_pca: PCANormalizer, held_out_data: np.ndarray
    ) -> None:
        whole = fitted_pca.transform(held_out_data)
        chunks = torch.cat(
            [fitted_pca.transform(held_out_data[s : s + 7]) for s in range(0, 30, 7)]
        )
        torch.testing.assert_close(chunks, whole)

    def test_single_non_mean_row_is_not_all_zeros(
        self, fitted_pca: PCANormalizer, held_out_data: np.ndarray
    ) -> None:
        assert fitted_pca.transform(held_out_data[:1]).abs().max().item() > 0.1

    def test_train_test_shift_is_preserved(
        self, training_data: np.ndarray, held_out_data: np.ndarray
    ) -> None:
        pca = PCANormalizer(n_components=N_COMPONENTS, scale_to_pi=False).fit(training_data)
        k = 2.0
        # Shift every row along the first principal axis; components are orthonormal,
        # so only the first standardised column should move, by exactly k / std_[0].
        delta = pca.transform(held_out_data + k * pca.components_[0]) - pca.transform(
            held_out_data
        )
        expected = torch.zeros_like(delta)
        expected[:, 0] = k / pca.std_[0]
        torch.testing.assert_close(delta, expected, atol=1e-5, rtol=0.0)

    def test_training_output_is_standardised(self, training_data: np.ndarray) -> None:
        z = PCANormalizer(n_components=N_COMPONENTS, scale_to_pi=False).fit_transform(
            training_data
        )
        torch.testing.assert_close(z.mean(dim=0), torch.zeros(N_COMPONENTS), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(z.std(dim=0), torch.ones(N_COMPONENTS), atol=1e-5, rtol=0.0)


class TestExplainedVarianceRatio:
    def test_sums_to_approximately_one(self, fitted_pca: PCANormalizer) -> None:
        ratio_sum = fitted_pca.explained_variance_ratio_.sum()
        assert 0.0 < ratio_sum <= 1.0 + 1e-6


class TestInputNotModified:
    """
    fit/transform convert with ``np.asarray``, so X_arr can share memory with the
    caller's array.  Every step must allocate rather than write in place.
    """

    def test_fit_leaves_input_unchanged(self, training_data: np.ndarray) -> None:
        pristine = training_data.copy()
        PCANormalizer(n_components=N_COMPONENTS).fit(training_data)
        np.testing.assert_array_equal(training_data, pristine)

    def test_transform_leaves_input_unchanged(
        self, fitted_pca: PCANormalizer, held_out_data: np.ndarray
    ) -> None:
        pristine = held_out_data.copy()
        fitted_pca.transform(held_out_data)
        np.testing.assert_array_equal(held_out_data, pristine)

    def test_non_contiguous_float32_input_unchanged(self) -> None:
        # float32 forces a dtype conversion and the strided slice makes the input
        # non-contiguous, so the converting path is exercised as well as the
        # zero-copy float64 one above
        rng = np.random.default_rng(2)
        X = rng.standard_normal((N_SAMPLES, 2 * N_FEATURES)).astype(np.float32)[:, ::2]
        assert X.dtype == np.float32 and not X.flags["C_CONTIGUOUS"]

        pristine = X.copy()
        PCANormalizer(n_components=N_COMPONENTS).fit_transform(X)
        np.testing.assert_array_equal(X, pristine)


class TestCopyOptionRemoved:
    def test_copy_keyword_rejected(self) -> None:
        # `copy` never had an effect; it was removed rather than deprecated
        with pytest.raises(TypeError, match="copy"):
            PCANormalizer(n_components=N_COMPONENTS, copy=True)  # type: ignore[call-arg]


class TestErrors:
    def test_transform_before_fit(self) -> None:
        pca = PCANormalizer(n_components=4)
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.transform(np.zeros((10, 12)))

    def test_too_few_features(self) -> None:
        pca = PCANormalizer(n_components=10)
        X_small = np.random.randn(50, 5)
        with pytest.raises(ValueError, match="n_features"):
            pca.fit(X_small)

    # Warnings as errors: on too few rows, np.cov/eigh warn and then raise LinAlgError
    # (a ValueError subclass), so the check must fire before any of that
    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("n_rows", [0, 1, N_COMPONENTS])
    def test_too_few_samples(self, n_rows: int) -> None:
        pca = PCANormalizer(n_components=N_COMPONENTS)
        X = np.random.default_rng(0).standard_normal((n_rows, N_FEATURES))
        with pytest.raises(
            ValueError,
            match=rf"n_samples={n_rows} <= n_components={N_COMPONENTS}\..*at least {N_COMPONENTS + 1} samples",
        ):
            pca.fit(X)

    # Warnings as errors: constant data makes explained_variance_ratio_ divide
    # 0 by 0, so the check must fire before any component is retained
    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize(
        "name, rank, remedy",
        [
            ("collinear_features", 2, r"at most 2"),
            ("duplicated_samples", 2, r"at most 2"),
            # Rank 0 gets prose, not "at most 0" -- a value fit() itself rejects
            ("constant", 0, r"Provide data that varies"),
        ],
    )
    def test_rank_below_n_components(self, name: str, rank: int, remedy: str) -> None:
        # Enough rows to clear the n_samples check, but a centred rank below
        # n_components.  Without the guard these fit silently and transform
        # returns ~1e8 (or a saturated +-pi) on the zero-variance components,
        # because std_ falls back to the 1e-8 added for division safety
        rng = np.random.default_rng(0)
        X = {
            "collinear_features": rng.standard_normal((N_SAMPLES, 2))
            @ rng.standard_normal((2, N_FEATURES)),
            "duplicated_samples": np.tile(rng.standard_normal((3, N_FEATURES)), (34, 1)),
            "constant": np.ones((N_SAMPLES, N_FEATURES)),
        }[name]
        assert len(X) > N_COMPONENTS, "must clear the n_samples check to reach the rank check"

        pca = PCANormalizer(n_components=N_COMPONENTS)
        with pytest.raises(
            ValueError,
            match=rf"rank {rank} < n_components={N_COMPONENTS}\b.*{remedy}",
        ):
            pca.fit(X)
        # A rejected fit leaves no attribute populated
        for attr in ("mean_", "components_", "explained_variance_", "std_"):
            assert not hasattr(pca, attr), attr
        assert pca.is_fitted_ is False

    @pytest.mark.filterwarnings("error")
    def test_failed_refit_leaves_previous_fit_intact(self) -> None:
        # fit is all-or-nothing: every check raises before the first attribute
        # is assigned, so a rejected re-fit is a no-op rather than a partial
        # overwrite.  Pinned because the obvious "tidy-up" -- resetting the
        # fitted attributes at the top of fit -- would silently break it, and
        # would destroy a working fit in response to one bad batch
        rng = np.random.default_rng(0)
        X = rng.standard_normal((N_SAMPLES, N_FEATURES))
        pca = PCANormalizer(n_components=N_COMPONENTS).fit(X)
        before = (
            pca.mean_.copy(),
            pca.components_.copy(),
            pca.explained_variance_.copy(),
            pca.std_.copy(),
        )
        expected = pca.transform(X)

        # One rejection per data-dependent guard, in declaration order
        # (the n_components < 1 guard keys off the attribute, not off X)
        rejected = [
            rng.standard_normal(N_SAMPLES),  # not 2-D
            rng.standard_normal((N_SAMPLES, N_COMPONENTS - 1)),  # too few features
            rng.standard_normal((N_COMPONENTS, N_FEATURES)),  # too few samples
            np.ones((N_SAMPLES, N_FEATURES)),  # rank deficient
        ]
        for X_bad in rejected:
            with pytest.raises(ValueError):
                pca.fit(X_bad)

        assert pca.is_fitted_ is True
        for name, old, new_ in zip(
            ("mean_", "components_", "explained_variance_", "std_"),
            before,
            (pca.mean_, pca.components_, pca.explained_variance_, pca.std_),
        ):
            assert np.array_equal(old, new_), f"{name} changed across a failed re-fit"
        # and the fit is still usable, not merely still present
        assert torch.equal(pca.transform(X), expected)

    def test_full_rank_at_tiny_scale_still_fits(self) -> None:
        # Guards the tolerance against being absolute.  These eigenvalues are
        # ~1e-14, below the ~4e-14 an absolute tolerance would need in order to
        # reject the rank-deficient cases above -- so an absolute threshold
        # would reject this genuinely full-rank data, and a relative one must not
        X = 1e-7 * np.random.default_rng(0).standard_normal((N_SAMPLES, N_FEATURES))
        pca = PCANormalizer(n_components=N_COMPONENTS).fit(X)
        assert pca.is_fitted_ is True
        assert pca.explained_variance_.max() < 1e-13
        # and the components carry real variance, not the 1e-8 epsilon.
        # std_ is std + 1e-8, so compare the excess: a true std of 1e-12 would
        # still clear a bare "> 1e-8" while being 99.99% epsilon
        assert np.all(pca.std_ - 1e-8 > 1e-8)

    def test_full_rank_at_heterogeneous_scales_still_fits(self) -> None:
        # Guards the tolerance against living in eigenvalue space.  Thresholding
        # the covariance eigenvalues relative to their maximum squares the
        # condition number, so full-rank data whose feature scales differ by
        # more than ~1e8 gets rejected as rank-deficient
        X = np.random.default_rng(0).standard_normal((N_SAMPLES, N_FEATURES))
        X[:, 0] *= 1e9
        assert np.linalg.matrix_rank(X - X.mean(axis=0)) == N_FEATURES

        # Full rank, but the remaining eigenvalues (~1) sit ~1e-18 apart
        # relative to the largest (~1e18), well inside the rounding that
        # rotates them, so the degeneracy warning is expected alongside the fit
        with pytest.warns(RuntimeWarning, match=r"degenerate"):
            pca = PCANormalizer(n_components=N_COMPONENTS).fit(X)
        assert pca.is_fitted_ is True
        # every retained component carries real variance, not the 1e-8 epsilon
        assert np.all(pca.std_ - 1e-8 > 1e-8)

    def test_minimum_samples_fit_has_nonzero_variance(self) -> None:
        X = np.random.default_rng(0).standard_normal((N_COMPONENTS + 1, N_FEATURES))
        pca = PCANormalizer(n_components=N_COMPONENTS).fit(X)
        # Degenerate components fall back to std_ == 1e-8; every one must be far above it
        assert pca.std_.shape == (N_COMPONENTS,)
        assert np.all(pca.std_ > 0.1)
        np.testing.assert_allclose(pca.std_, np.sqrt(pca.explained_variance_) + 1e-8, rtol=1e-10)

    # Warnings as errors: with a single row np.cov/eigh warn and then raise
    # LinAlgError (a ValueError subclass), so the check must fire before any of that
    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("n_rows", [N_SAMPLES, 1])
    @pytest.mark.parametrize("n_components", [0, -1])
    def test_non_positive_n_components(self, n_components: int, n_rows: int) -> None:
        pca = PCANormalizer(n_components=n_components)
        X = np.random.default_rng(0).standard_normal((n_rows, N_FEATURES))
        with pytest.raises(
            ValueError,
            match=rf"n_components={n_components} < 1\..*positive",
        ):
            pca.fit(X)

    # Warnings as errors: without the guard the top-k slice raises a bare
    # TypeError about slice indices, so the check must fire before it
    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("n_components", [2.5, 3.0, "4", None])
    def test_non_integer_n_components(self, n_components: object) -> None:
        pca = PCANormalizer(n_components=n_components)  # type: ignore[arg-type]
        X = np.random.default_rng(0).standard_normal((N_SAMPLES, N_FEATURES))
        with pytest.raises(
            ValueError,
            match=rf"n_components={n_components!r} is not an integer\..*rejected rather than coerced",
        ):
            pca.fit(X)
        assert pca.is_fitted_ is False

    @pytest.mark.filterwarnings("error")
    def test_bool_n_components(self) -> None:
        # bool subclasses int, so True would otherwise fit silently with one component
        pca = PCANormalizer(n_components=True)
        X = np.random.default_rng(0).standard_normal((N_SAMPLES, N_FEATURES))
        with pytest.raises(ValueError, match=r"n_components=True is a bool, not an integer"):
            pca.fit(X)
        assert pca.is_fitted_ is False

    @pytest.mark.parametrize("n_components", [np.int64(4), np.int32(4), np.uint8(4)])
    def test_numpy_integer_n_components_fits(self, n_components: np.integer) -> None:
        X = np.random.default_rng(0).standard_normal((N_SAMPLES, N_FEATURES))
        pca = PCANormalizer(n_components=n_components).fit(X)
        assert pca.components_.shape == (4, N_FEATURES)
        assert pca.transform(X).shape == (N_SAMPLES, 4)

    # Warnings as errors: a fixed-width NumPy integer at its maximum would wrap
    # in the "at least n_components + 1 samples" message (255 + 1 == 0 for
    # uint8) and raise an overflow RuntimeWarning instead of the ValueError
    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("n_components", [np.uint8(255), np.int8(127)])
    def test_numpy_integer_n_components_does_not_overflow(self, n_components: np.integer) -> None:
        n = int(n_components)
        pca = PCANormalizer(n_components=n_components)
        X = np.random.default_rng(0).standard_normal((100, n + 45))
        with pytest.raises(
            ValueError,
            match=rf"n_samples=100 <= n_components={n}\..*at least {n + 1} samples",
        ):
            pca.fit(X)
        assert pca.is_fitted_ is False

    # Warnings as errors: with a single column np.cov returns a 0-d array and
    # eigh raises LinAlgError (a ValueError subclass) about the array's
    # dimensionality, so the check must fire before it
    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("n_components", [1, N_COMPONENTS])
    def test_single_feature_input(self, n_components: int) -> None:
        pca = PCANormalizer(n_components=n_components)
        X = np.random.default_rng(0).standard_normal((10, 1))
        with pytest.raises(ValueError, match=r"n_features=1 < 2\..*at least two features"):
            pca.fit(X)
        assert pca.is_fitted_ is False

    def test_two_features_still_fit(self) -> None:
        # The lower bound is 2, not higher: the smallest decomposable case must work
        X = np.random.default_rng(0).standard_normal((10, 2))
        pca = PCANormalizer(n_components=2).fit(X)
        assert pca.components_.shape == (2, 2)

    def test_fit_1d_input(self) -> None:
        pca = PCANormalizer(n_components=N_COMPONENTS)
        with pytest.raises(
            ValueError, match=rf"2-D input.*got shape \({N_FEATURES},\).*reshape\(1, -1\)"
        ):
            pca.fit(np.zeros(N_FEATURES))

    def test_fit_3d_input(self) -> None:
        pca = PCANormalizer(n_components=N_COMPONENTS)
        # Anchored at the end: the 1-D reshape hint must not follow
        with pytest.raises(ValueError, match=rf"2-D input.*got shape \(2, 3, {N_FEATURES}\)\.$"):
            pca.fit(np.zeros((2, 3, N_FEATURES)))

    def test_transform_1d_input(self, fitted_pca: PCANormalizer) -> None:
        with pytest.raises(
            ValueError, match=rf"2-D input.*got shape \({N_FEATURES},\).*reshape\(1, -1\)"
        ):
            fitted_pca.transform(np.zeros(N_FEATURES))

    def test_transform_3d_input(self, fitted_pca: PCANormalizer) -> None:
        # Last axis matches the training features, so this must get the ndim error,
        # not a misleading feature-count mismatch
        with pytest.raises(ValueError, match=rf"2-D input.*got shape \(2, 3, {N_FEATURES}\)\.$"):
            fitted_pca.transform(np.zeros((2, 3, N_FEATURES)))


def _data_with_spectrum(
    eigenvalues: list[float], n_samples: int = 200, seed: int = 0
) -> np.ndarray:
    """
    Data whose *sample* covariance has exactly the given eigenvalues.

    Orthonormalise centred Gaussian columns with QR (they stay centred, since
    the column space is orthogonal to the ones vector), scale each to the chosen
    variance, then mix with a random rotation so the components are not axis-
    aligned.  cov = R.T @ diag(eigenvalues) @ R up to rounding.
    """
    rng = np.random.default_rng(seed)
    n_features = len(eigenvalues)
    G = rng.standard_normal((n_samples, n_features))
    Q, _ = np.linalg.qr(G - G.mean(axis=0))
    R, _ = np.linalg.qr(rng.standard_normal((n_features, n_features)))
    return Q @ np.diag(np.sqrt((n_samples - 1) * np.asarray(eigenvalues))) @ R


class TestDegenerateEigenvalueWarning:
    """
    components_ is reproducible only when the eigenvalues are well separated.
    fit surfaces the exception with a RuntimeWarning instead of leaving the
    caller with a silently build-dependent basis.
    """

    WELL_SEPARATED = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0]

    def test_construction_hits_the_requested_spectrum(self) -> None:
        X = _data_with_spectrum(self.WELL_SEPARATED)
        got = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
        np.testing.assert_allclose(got, self.WELL_SEPARATED, rtol=1e-10)

    @pytest.mark.filterwarnings("error")
    def test_well_separated_spectrum_does_not_warn(self) -> None:
        PCANormalizer(n_components=4).fit(_data_with_spectrum(self.WELL_SEPARATED))

    @pytest.mark.filterwarnings("error")
    def test_golden_fixture_does_not_warn(self, training_data: np.ndarray) -> None:
        PCANormalizer(n_components=N_COMPONENTS).fit(training_data)

    def test_degenerate_pair_among_kept_components_warns(self) -> None:
        X = _data_with_spectrum([8.0, 6.0, 6.0, 4.0, 2.0, 1.0])
        with pytest.warns(
            RuntimeWarning,
            match=r"eigenvalues 1 and 2 are degenerate.*among the retained components.*Reduce n_components to 1",
        ):
            pca = PCANormalizer(n_components=4).fit(X)
        # The fit itself is still valid: pinned so the behaviour at exact
        # degeneracy is recorded rather than discovered later
        assert pca.is_fitted_ is True
        assert pca.components_.shape == (4, 6)
        np.testing.assert_allclose(pca.explained_variance_, [8.0, 6.0, 6.0, 4.0], rtol=1e-10)

    def test_degenerate_pair_at_cutoff_warns(self) -> None:
        # ev[3] == ev[4] with n_components=4: which of the two is kept is arbitrary
        X = _data_with_spectrum([8.0, 6.0, 5.0, 3.0, 3.0, 1.0])
        with pytest.warns(
            RuntimeWarning, match=r"eigenvalues 3 and 4 are degenerate.*at the n_components cutoff"
        ):
            PCANormalizer(n_components=4).fit(X)

    @pytest.mark.filterwarnings("error")
    def test_degeneracy_beyond_cutoff_is_ignored(self) -> None:
        # ev[4] == ev[5] are both discarded; the retained basis is unaffected
        PCANormalizer(n_components=4).fit(_data_with_spectrum([8.0, 6.0, 5.0, 3.0, 1.0, 1.0]))

    @pytest.mark.filterwarnings("error")
    def test_all_features_kept_checks_only_retained_gaps(self) -> None:
        # n_components == n_features: no cutoff gap exists, and the spectrum is separated
        PCANormalizer(n_components=6).fit(_data_with_spectrum(self.WELL_SEPARATED))

    @pytest.mark.parametrize("gap", [1e-8, 1e-10, 10 * EIGENVALUE_GAP_WARN])
    @pytest.mark.filterwarnings("error")
    def test_gap_above_threshold_does_not_warn(self, gap: float) -> None:
        # Separation the LAPACK spread cannot rotate measurably: no warning
        X = _data_with_spectrum([8.0, 6.0, 6.0 * (1 - gap), 4.0, 2.0, 1.0])
        PCANormalizer(n_components=4).fit(X)

    def test_gap_below_threshold_warns(self) -> None:
        X = _data_with_spectrum([8.0, 6.0, 6.0 * (1 - EIGENVALUE_GAP_WARN / 10), 4.0, 2.0, 1.0])
        with pytest.warns(RuntimeWarning, match=r"eigenvalues 1 and 2"):
            PCANormalizer(n_components=4).fit(X)

    def test_close_pair_of_small_eigenvalues_warns(self) -> None:
        # Relative to each other ev[1] and ev[2] differ by 1e-9, but LAPACK
        # rounding scales with ev[0], against which they are 1e-15 apart
        X = _data_with_spectrum([1.0, 1e-6, 1e-6 * (1 - 1e-9), 1e-7, 1e-8, 1e-9])
        with pytest.warns(RuntimeWarning, match=r"eigenvalues 1 and 2 are degenerate"):
            PCANormalizer(n_components=3).fit(X)

    def test_first_degenerate_pair_is_reported(self) -> None:
        # Two degenerate pairs: the suggestion must avoid both, not only the later one
        X = _data_with_spectrum([8.0, 6.0, 6.0, 4.0, 4.0, 1.0])
        with pytest.warns(
            RuntimeWarning, match=r"eigenvalues 1 and 2 .*Reduce n_components to 1,"
        ) as record:
            PCANormalizer(n_components=4).fit(X)
        assert len(record) == 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PCANormalizer(n_components=1).fit(X)

    def test_degeneracy_at_first_eigenvalue_suggests_no_smaller_n_components(self) -> None:
        X = _data_with_spectrum([6.0, 6.0, 4.0, 1.0])
        with pytest.warns(
            RuntimeWarning, match=r"eigenvalues 0 and 1 .*No smaller n_components avoids this"
        ) as record:
            PCANormalizer(n_components=2).fit(X)
        assert "Reduce n_components to 0" not in str(record[0].message)

    @pytest.mark.parametrize("method", ["fit", "fit_transform"])
    def test_warning_points_at_the_caller(self, method: str) -> None:
        X = _data_with_spectrum([8.0, 6.0, 6.0, 4.0, 2.0, 1.0])
        with pytest.warns(RuntimeWarning, match=r"degenerate") as record:
            getattr(PCANormalizer(n_components=4), method)(X)
        assert record[0].filename == __file__


class TestEigenvalueGaps:
    """
    ``eigenvalue_gaps_`` holds the gaps the degeneracy warning checks, so a
    caller can record them without parsing the warning (#525): entry ``i`` is
    ``(λ_i − λ_{i+1}) / λ_0`` for the eigenvalues ``λ`` of the covariance in
    descending order, over the retained ones and the first excluded one.
    """

    def test_gaps_of_a_known_spectrum(self) -> None:
        # Kept 8, 6, 6, 4; first excluded 2 (the cutoff gap); 1 is not looked at.
        X = _data_with_spectrum([8.0, 6.0, 6.0, 4.0, 2.0, 1.0])
        with pytest.warns(RuntimeWarning, match=r"eigenvalues 1 and 2"):
            pca = PCANormalizer(n_components=4).fit(X)
        np.testing.assert_allclose(
            pca.eigenvalue_gaps_, [2 / 8, 0.0, 2 / 8, 2 / 8], rtol=0, atol=1e-12
        )
        assert pca.eigenvalue_gaps_[1] < EIGENVALUE_GAP_WARN

    @pytest.mark.filterwarnings("error")
    def test_degeneracy_beyond_the_cutoff_is_not_among_them(self) -> None:
        pca = PCANormalizer(n_components=4).fit(
            _data_with_spectrum([8.0, 6.0, 5.0, 3.0, 1.0, 1.0])
        )
        np.testing.assert_allclose(
            pca.eigenvalue_gaps_, [2 / 8, 1 / 8, 2 / 8, 2 / 8], rtol=0, atol=1e-12
        )

    @pytest.mark.filterwarnings("error")
    def test_all_features_kept_has_no_cutoff_gap(self) -> None:
        pca = PCANormalizer(n_components=6).fit(
            _data_with_spectrum(TestDegenerateEigenvalueWarning.WELL_SEPARATED)
        )
        np.testing.assert_allclose(pca.eigenvalue_gaps_, [1 / 8] * 5, rtol=0, atol=1e-12)

    def test_gaps_match_an_svd_of_the_centred_data(self, training_data: np.ndarray) -> None:
        # Another route to the eigenvalues: the squared singular values over n - 1.
        pca = PCANormalizer(n_components=N_COMPONENTS).fit(training_data)
        s = np.linalg.svd(training_data - training_data.mean(axis=0), compute_uv=False)
        lam = s**2 / (N_SAMPLES - 1)
        expected = (lam[:N_COMPONENTS] - lam[1 : N_COMPONENTS + 1]) / lam[0]
        np.testing.assert_allclose(pca.eigenvalue_gaps_, expected, rtol=1e-10, atol=0)

    @pytest.mark.parametrize(
        ("spectrum", "warns"),
        [
            ([8.0, 6.0, 6.0, 4.0, 2.0, 1.0], True),
            ([8.0, 6.0, 5.0, 3.0, 3.0, 1.0], True),
            ([8.0, 6.0, 5.0, 3.0, 1.0, 1.0], False),
            ([8.0, 6.0, 6.0 * (1 - 10 * EIGENVALUE_GAP_WARN), 4.0, 2.0, 1.0], False),
        ],
    )
    def test_the_warning_fires_exactly_when_a_gap_is_below_the_threshold(
        self, spectrum: list[float], warns: bool
    ) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pca = PCANormalizer(n_components=4).fit(_data_with_spectrum(spectrum))
        assert bool(caught) is warns
        assert bool((pca.eigenvalue_gaps_ < EIGENVALUE_GAP_WARN).any()) is warns

    def test_a_failed_refit_keeps_the_gaps(self) -> None:
        X = _data_with_spectrum(TestDegenerateEigenvalueWarning.WELL_SEPARATED)
        pca = PCANormalizer(n_components=4).fit(X)
        before = pca.eigenvalue_gaps_.copy()
        with pytest.raises(ValueError):
            pca.fit(np.ones((50, 6)))
        np.testing.assert_array_equal(pca.eigenvalue_gaps_, before)
