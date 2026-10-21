"""
tests/test_example_benchmark.py
===============================
``examples/benchmark.py`` (#269): the subset it keeps of a large dataset.

``stratified_subset`` keeps the rows of
:func:`hqnn_forge.diagnostics.stratified_subsample_indices` (#541), checked
here against that function called directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from hqnn_forge.diagnostics import stratified_subsample_indices

ROOT = Path(__file__).resolve().parent.parent


def _script() -> ModuleType:
    path = ROOT / "examples" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("example_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _script()


def _data(counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Rows numbered ``0, 1, …`` in their first column, labels shuffled with a fixed seed."""
    y = np.random.default_rng(0).permutation(np.repeat(np.arange(len(counts)), counts))
    X = np.stack([np.arange(y.size, dtype=np.float64), y.astype(np.float64)], axis=1)
    return X, y


class TestStratifiedSubset:
    def test_keeps_the_rows_of_stratified_subsample_indices(self) -> None:
        X, y = _data([950, 50])
        for max_rows, seed in ((200, 0), (37, 3)):
            idx = stratified_subsample_indices(y, max_rows, seed=seed)
            X_sub, y_sub = example.stratified_subset(X, y, max_rows, seed=seed)
            assert np.array_equal(X_sub, X[idx])
            assert np.array_equal(y_sub, y[idx])
            assert np.array_equal(X_sub[:, 0].astype(int), idx)  # the rows, in their order

    def test_the_credit_card_subset_is_the_one_drawn_before(self) -> None:
        """
        On the class counts of the credit-card data (492 frauds in 284,807
        rows) the rows are those the earlier per-class rounding drew, typed
        in here, at the default ``--max-rows`` and three others (#541).
        """
        y = np.zeros(284_807, dtype=np.int64)
        y[np.random.default_rng(5).choice(y.size, 492, replace=False)] = 1
        X = np.arange(y.size, dtype=np.float64)[:, None]
        for max_rows in (1000, 5000, 20_000, 50_000):
            rng = np.random.default_rng(0)
            before = np.sort(
                np.concatenate(
                    [
                        rng.choice(
                            np.flatnonzero(y == c),
                            round(max_rows * np.mean(y == c)),
                            replace=False,
                        )
                        for c in (0, 1)
                    ]
                )
            )
            X_sub, _ = example.stratified_subset(X, y, max_rows)
            assert np.array_equal(X_sub[:, 0].astype(np.int64), before)

    def test_keeps_exactly_max_rows(self) -> None:
        """
        ``round(5 · ½) = 2`` for each of two equal classes kept 4 of 5 rows,
        and ``round(5 · 0.3) + round(5 · 0.7) = 2 + 4`` kept 6; the
        largest-remainder quotas keep 5 (``[3, 2]`` and ``[2, 3]``, the tie
        to the lower label).
        """
        for counts, kept in (([50, 50], [3, 2]), ([30, 70], [2, 3])):
            X, y = _data(counts)
            X_sub, y_sub = example.stratified_subset(X, y, 5)
            assert len(X_sub) == len(y_sub) == 5
            assert np.bincount(y_sub, minlength=2).tolist() == kept

    def test_keeps_every_class(self) -> None:
        """Three classes, where the old draw kept only classes 0 and 1."""
        X, y = _data([60, 30, 10])
        _, y_sub = example.stratified_subset(X, y, 10)
        assert np.bincount(y_sub).tolist() == [6, 3, 1]

    def test_a_small_dataset_is_kept_whole(self) -> None:
        X, y = _data([8, 2])
        X_sub, y_sub = example.stratified_subset(X, y, 10)
        assert X_sub is X and y_sub is y

    def test_a_tie_goes_to_the_lower_label(self) -> None:
        """Quotas 2.5 and 1.5: the old rounding kept 2 and 2, the quotas give class 0 the row."""
        X, y = _data([5, 3])
        _, y_sub = example.stratified_subset(X, y, 4)
        assert np.bincount(y_sub).tolist() == [3, 1]
