"""
tests/test_cv.py
================
Stratified k-fold and fold-safe SMOTE (hqnn_forge.preprocessing.cv).

The three properties the issue asks for:
(a) class prevalence is preserved per fold,
(b) validation partitions are never oversampled,
(c) no synthetic sample is derived from a row outside its training fold.
"""

from __future__ import annotations

import numpy as np
import pytest

from hqnn_forge.preprocessing import (
    Fold,
    iter_folds,
    oversample_fold,
    smote,
    stratified_kfold,
)


@pytest.fixture
def imbalanced() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, n_pos = 203, 23
    X = rng.standard_normal((n, 4))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    X[y == 1] += 2.0
    return X, y


class TestStratifiedKFold:
    @pytest.mark.parametrize("n_splits", [2, 3, 5, 7])
    def test_prevalence_preserved_per_fold(self, imbalanced: tuple, n_splits: int) -> None:
        _, y = imbalanced
        n_pos = int(y.sum())
        for train, val in stratified_kfold(y, n_splits, random_state=0):
            # Each class is dealt into near-equal chunks: counts differ by at most 1
            assert abs(int(y[val].sum()) - n_pos / n_splits) < 1
            assert abs(int((y[val] == 0).sum()) - (y.size - n_pos) / n_splits) < 1

    def test_partition_properties(self, imbalanced: tuple) -> None:
        _, y = imbalanced
        folds = stratified_kfold(y, 5, random_state=0)
        all_val = np.concatenate([val for _, val in folds])
        assert np.array_equal(np.sort(all_val), np.arange(y.size))  # each sample validated once
        for train, val in folds:
            assert np.intersect1d(train, val).size == 0
            assert train.size + val.size == y.size
            assert np.all(np.diff(train) > 0) and np.all(np.diff(val) > 0)

    def test_shuffle_and_seed(self, imbalanced: tuple) -> None:
        _, y = imbalanced
        a = stratified_kfold(y, 5, random_state=1)
        b = stratified_kfold(y, 5, random_state=1)
        c = stratified_kfold(y, 5, random_state=2)
        assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))
        assert not all(np.array_equal(x[1], z[1]) for x, z in zip(a, c))

    def test_no_shuffle_is_ordered(self) -> None:
        y = np.array([0, 0, 0, 0, 1, 1])
        folds = stratified_kfold(y, 2, shuffle=False)
        assert folds[0][1].tolist() == [0, 1, 4]
        assert folds[1][1].tolist() == [2, 3, 5]

    def test_multiclass_and_string_labels(self) -> None:
        y = np.array(["a"] * 6 + ["b"] * 9 + ["c"] * 3)
        for _, val in stratified_kfold(y, 3, random_state=0):
            assert sorted(np.unique(y[val], return_counts=True)[1].tolist()) == [1, 2, 3]

    def test_class_smaller_than_n_splits(self) -> None:
        with pytest.raises(ValueError, match=r"class 1 has 2"):
            stratified_kfold([0, 0, 0, 0, 1, 1], 3)

    @pytest.mark.parametrize(
        "y, n_splits, match",
        [
            ([0, 1], 1, "n_splits must be >= 2"),
            ([[0, 1]], 2, "must be 1-D"),
            ([], 2, "y is empty"),
        ],
    )
    def test_bad_arguments(self, y: list, n_splits: int, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            stratified_kfold(y, n_splits)


class TestSmote:
    def test_balances_and_keeps_originals_first(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        res = smote(X, y, random_state=0)
        assert np.array_equal(res.X[: len(X)], X) and np.array_equal(res.y[: len(y)], y)
        assert int((res.y == 1).sum()) == int((res.y == 0).sum())
        assert np.all(res.y[len(y) :] == 1)
        assert res.sources.shape == (len(res.y) - len(y), 2)

    def test_synthetic_points_lie_between_their_sources(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        res = smote(X, y, random_state=0)
        synth = res.X[len(X) :]
        a, b = X[res.sources[:, 0]], X[res.sources[:, 1]]
        assert np.all(y[res.sources] == 1)
        assert np.all(res.sources[:, 0] != res.sources[:, 1])
        direction = b - a
        u = np.sum((synth - a) * direction, axis=1) / np.sum(direction**2, axis=1)
        assert np.all((u >= 0) & (u <= 1))
        np.testing.assert_allclose(a + u[:, None] * direction, synth, atol=1e-12)

    def test_neighbours_are_among_the_k_nearest(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        k = 3
        res = smote(X, y, k_neighbors=k, random_state=0)
        M_idx = np.flatnonzero(y == 1)
        for src, nb in res.sources:
            d = np.linalg.norm(X[M_idx] - X[src], axis=1)
            d[M_idx == src] = np.inf
            nearest = set(M_idx[np.argsort(d, kind="stable")[:k]].tolist())
            assert nb in nearest

    def test_sampling_ratio(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        res = smote(X, y, sampling_ratio=0.5, random_state=0)
        assert int((res.y == 1).sum()) == int(np.ceil(0.5 * (y == 0).sum()))

    def test_already_balanced_generates_nothing(self) -> None:
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.array([0, 1] * 5)
        res = smote(X, y, k_neighbors=2)
        assert np.array_equal(res.X, X) and res.sources.shape == (0, 2)

    def test_balanced_no_op_does_not_need_k_neighbors_samples(self) -> None:
        """Nothing is interpolated, so the neighbourhood size cannot apply."""
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.array([0, 1] * 5)  # 5 minority rows, default k_neighbors=5
        res = smote(X, y)
        assert np.array_equal(res.X, X) and res.sources.shape == (0, 2)

    def test_minority_is_whichever_label_is_rarer(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 2))
        y = np.array([1] * 24 + [0] * 6)
        res = smote(X, y, k_neighbors=2, random_state=0)
        assert np.all(res.y[30:] == 0)

    def test_reproducible(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        a, b = smote(X, y, random_state=5), smote(X, y, random_state=5)
        assert np.array_equal(a.X, b.X)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"k_neighbors": 30}, "needs at least 31"),
            ({"k_neighbors": 0}, "k_neighbors must be >= 1"),
            ({"sampling_ratio": 0.0}, "sampling_ratio must lie"),
            ({"sampling_ratio": 1.5}, "sampling_ratio must lie"),
        ],
    )
    def test_argument_errors(self, imbalanced: tuple, kwargs: dict, match: str) -> None:
        X, y = imbalanced
        with pytest.raises(ValueError, match=match):
            smote(X, y, **kwargs)

    def test_shape_and_class_errors(self) -> None:
        with pytest.raises(ValueError, match="X must be 2-D"):
            smote(np.zeros(4), np.zeros(4))
        with pytest.raises(ValueError, match="to match X"):
            smote(np.zeros((4, 2)), np.zeros(3))
        with pytest.raises(ValueError, match="exactly two classes"):
            smote(np.zeros((6, 2)), np.array([0, 1, 2, 0, 1, 2]))


class TestFoldSafety:
    def test_validation_rows_are_untouched(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        for fold in iter_folds(X, y, 5, random_state=0, k_neighbors=3):
            assert isinstance(fold, Fold)
            assert np.array_equal(fold.X_val, X[fold.val_idx])
            assert np.array_equal(fold.y_val, y[fold.val_idx])
            # prevalence of the validation split is the original one, not balanced
            assert float(fold.y_val.mean()) < 0.2

    def test_training_split_is_oversampled(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        for fold in iter_folds(X, y, 5, random_state=0, k_neighbors=3):
            assert (fold.y_train == 1).sum() == (fold.y_train == 0).sum()
            assert np.array_equal(fold.X_train[: fold.train_idx.size], X[fold.train_idx])

    def test_no_synthetic_sample_comes_from_outside_its_training_fold(
        self, imbalanced: tuple
    ) -> None:
        X, y = imbalanced
        for fold in iter_folds(X, y, 5, random_state=0, k_neighbors=3):
            assert fold.sources.size > 0
            assert np.all(np.isin(fold.sources, fold.train_idx))
            assert not np.any(np.isin(fold.sources, fold.val_idx))
            # and the stored sources really are the pair each synthetic row was built from
            synth = fold.X_train[fold.train_idx.size :]
            a, b = X[fold.sources[:, 0]], X[fold.sources[:, 1]]
            u = np.sum((synth - a) * (b - a), axis=1) / np.sum((b - a) ** 2, axis=1)
            np.testing.assert_allclose(a + u[:, None] * (b - a), synth, atol=1e-12)

    def test_smote_on_full_data_would_leak(self, imbalanced: tuple) -> None:
        """The failure this module prevents, shown on the same data."""
        X, y = imbalanced
        leaked = smote(X, y, k_neighbors=3, random_state=0)
        _, val = stratified_kfold(y, 5, random_state=0)[0]
        assert np.any(np.isin(leaked.sources, val))

    def test_without_oversampling(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        for fold in iter_folds(X, y, 4, random_state=0, oversample=False):
            assert np.array_equal(fold.X_train, X[fold.train_idx])
            assert fold.sources.shape == (0, 2)

    def test_folds_are_reproducible(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        a = list(iter_folds(X, y, 3, random_state=7, k_neighbors=3))
        b = list(iter_folds(X, y, 3, random_state=7, k_neighbors=3))
        for fa, fb in zip(a, b):
            assert np.array_equal(fa.X_train, fb.X_train)
            assert np.array_equal(fa.sources, fb.sources)

    def test_a_different_seed_changes_the_folds(self, imbalanced: tuple) -> None:
        """random_state drives both the shuffle and the SMOTE streams."""
        X, y = imbalanced
        a = list(iter_folds(X, y, 3, random_state=7, k_neighbors=3))
        b = list(iter_folds(X, y, 3, random_state=8, k_neighbors=3))
        assert not np.array_equal(a[0].train_idx, b[0].train_idx)

    def test_the_smote_stream_is_what_varies_within_a_fold(self, imbalanced: tuple) -> None:
        """
        Folds differ in their training data, so comparing two folds cannot
        show that their SMOTE streams differ.  Hold the fold fixed instead and
        vary only the generator.
        """
        X, y = imbalanced
        tr, va = stratified_kfold(y, 3, random_state=0)[0]
        one = oversample_fold(X, y, tr, va, k_neighbors=3, random_state=np.random.default_rng(1))
        two = oversample_fold(X, y, tr, va, k_neighbors=3, random_state=np.random.default_rng(2))
        assert np.array_equal(one.X_train[: tr.size], two.X_train[: tr.size])
        assert not np.allclose(one.X_train[tr.size :], two.X_train[tr.size :])

    def test_each_fold_draws_from_its_own_smote_stream(self, imbalanced: tuple) -> None:
        """
        Comparing raw synthetic rows across folds proves nothing: the training
        data differs anyway.  Folds 0 and 1 of this fixture draw the same
        number of synthetic rows from minority sets of the same size, so the
        *sequence of positions* they pick is comparable, and is identical
        whenever the two folds share one stream.
        """
        X, y = imbalanced
        folds = list(iter_folds(X, y, 3, random_state=7, k_neighbors=3))
        picks = []
        for f in folds[:2]:
            minority = f.train_idx[y[f.train_idx] == 1]
            picks.append(np.searchsorted(minority, f.sources[:, 0]))
        assert minority.size and picks[0].size == picks[1].size  # comparable streams
        assert not np.array_equal(picks[0], picks[1])

    def test_accepts_a_generator_as_random_state(self, imbalanced: tuple) -> None:
        """stratified_kfold and smote both take a Generator; so must iter_folds."""
        X, y = imbalanced
        a = list(iter_folds(X, y, 3, random_state=np.random.default_rng(5), k_neighbors=3))
        b = list(iter_folds(X, y, 3, random_state=np.random.default_rng(5), k_neighbors=3))
        assert len(a) == 3
        for fa, fb in zip(a, b):
            assert np.array_equal(fa.X_train, fb.X_train)

    def test_a_fold_too_small_for_smote_names_the_fold(self) -> None:
        """
        stratified_kfold only promises each class reaches every fold; SMOTE
        needs k_neighbors + 1 minority rows in each *training* split.
        """
        X = np.arange(120, dtype=float).reshape(60, 2)
        y = np.zeros(60, dtype=int)
        y[:6] = 1
        with pytest.raises(ValueError, match=r"fold 0 of 5: SMOTE failed"):
            list(iter_folds(X, y, 5, random_state=0, k_neighbors=5))

    def test_overlapping_indices_are_rejected(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        with pytest.raises(ValueError, match="overlap"):
            oversample_fold(X, y, np.arange(0, 150), np.arange(100, 203))

    def test_negative_indices_are_rejected(self, imbalanced: tuple) -> None:
        """
        -1 names the last row but does not intersect 202, so a negative
        train_idx would pass the overlap check and then train on the
        validation rows, with Fold.sources reporting the negatives as safe.
        """
        X, y = imbalanced
        val = np.arange(198, 203)
        train = np.concatenate([np.arange(0, 150), -np.arange(1, 6)])
        with pytest.raises(ValueError, match=r"train_idx must index rows of X"):
            oversample_fold(X, y, train, val, k_neighbors=3)

    def test_out_of_range_indices_are_rejected(self, imbalanced: tuple) -> None:
        X, y = imbalanced
        with pytest.raises(ValueError, match=r"val_idx must index rows of X"):
            oversample_fold(X, y, np.arange(0, 150), np.array([203]))

    def test_boolean_masks_select_rows_not_zeros_and_ones(self, imbalanced: tuple) -> None:
        """A mask cast to intp would silently become indices 0 and 1."""
        X, y = imbalanced
        tr, va = stratified_kfold(y, 4, random_state=0)[0]
        train_mask = np.zeros(y.size, dtype=bool)
        train_mask[tr] = True
        val_mask = np.zeros(y.size, dtype=bool)
        val_mask[va] = True
        masked = oversample_fold(X, y, train_mask, val_mask, k_neighbors=3, random_state=0)
        indexed = oversample_fold(X, y, tr, va, k_neighbors=3, random_state=0)
        assert np.array_equal(masked.train_idx, indexed.train_idx)
        assert np.array_equal(masked.X_train, indexed.X_train)
