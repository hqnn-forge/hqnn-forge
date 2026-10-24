"""
tests/test_summarise_rydberg_benchmark.py
=========================================
``examples/summarise_rydberg_benchmark.py`` (#502): the statistics behind the
tables of ``docs/results/rydberg-features-benchmark.md``.

The t quantiles are checked against tabulated values, the Wilcoxon p-value of
a comparison won in every one of 13 folds against its exact floor ``2/2^13``,
and the Holm family against :func:`holm_correction` called directly.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from hqnn_forge.evaluation import holm_correction

ROOT = Path(__file__).resolve().parent.parent


def _script() -> ModuleType:
    path = ROOT / "examples" / "summarise_rydberg_benchmark.py"
    spec = importlib.util.spec_from_file_location("summarise_rydberg_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


summary = _script()

N_FOLDS = 13
MEASURES = {
    "n_samples": 10,
    "n_features": 4,
    "distance_ratio": 1.0,
    "fisher_ratio": 0.5,
    "effective_rank": 3.0,
    "kernel_alignment": 0.1,
}


def _line(
    arm: str,
    head: str,
    fold: int,
    mcc: float,
    *,
    v: float | None = None,
    gamma: float | None = None,
    reused: float | None = None,
    **measures: float,
) -> dict[str, Any]:
    return {
        "dataset": "toy",
        "arm": arm,
        "head": head,
        "v_over_omega": v,
        "gamma_over_omega": gamma,
        "fold": fold,
        "reused_from_v_over_omega": reused,
        "mcc": mcc,
        "pr_auc": 0.5,
        "feature_dim": 4,
        "n_parameters": 5,
        "feature_seconds": 0.0 if reused is not None else 1.0,
        "train_seconds": 0.1,
        "separation": {"train": {**MEASURES, **measures}, "test": dict(MEASURES)},
    }


def _records() -> list[dict[str, Any]]:
    """A at two V/Ω and one γ/Ω, B (computed at the first V/Ω, reused at the second), C."""
    rng = np.random.default_rng(1)
    records = []
    for fold in range(N_FOLDS):
        base = float(rng.uniform(0.0, 0.2))
        # A wins every fold at V/Ω = 1 and is indistinguishable from B at V/Ω = 10.
        records.append(
            _line("A", "linear", fold, base + 0.3 + 0.01 * fold, v=1.0, gamma=0.0, fisher_ratio=2)
        )
        records.append(
            _line("A", "linear", fold, base + (-1) ** fold * 0.01 * (fold + 1), v=10.0, gamma=0.0)
        )
        records.append(_line("B", "linear", fold, base, v=1.0, gamma=0.0))
        records.append(_line("B", "linear", fold, base, v=10.0, gamma=0.0, reused=1.0))
        records.append(_line("C", "linear", fold, base + 0.05))
        records.append(_line("C", "mlp", fold, base + 0.05))
    return records


@pytest.mark.parametrize(
    ("df", "expected"), [(1, 12.706205), (4, 2.776445), (12, 2.178813), (30, 2.042272)]
)
def test_t_quantile_matches_tables(df: int, expected: float) -> None:
    assert summary.t_quantile(0.975, df) == pytest.approx(expected, abs=1e-5)


def test_interval_is_the_t_interval() -> None:
    mean, sd, low, high = summary.interval([1.0, 2.0, 3.0])
    half = 4.302653 / math.sqrt(3)
    assert (mean, sd) == pytest.approx((2.0, 1.0))
    assert (low, high) == pytest.approx((2.0 - half, 2.0 + half), abs=1e-5)


def test_spearman_of_a_monotone_map_is_one_and_ties_average() -> None:
    rho, n, p = summary.spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25], permutations=999, seed=0)
    assert (rho, n) == (pytest.approx(1.0), 5)
    # Only the identity and nothing else reaches |rho| = 1 among the 120 orders but its
    # reverse, so about 2/120 of the draws plus the observed one.
    assert p == pytest.approx((1 + 999 * 2 / 120) / 1000, abs=0.02)
    assert summary._ranks(np.array([3.0, 1.0, 3.0])).tolist() == [2.5, 1.0, 2.5]


def test_b_lines_reused_across_v_count_once() -> None:
    rows = summary._rows(_records())
    b = rows[("B", "linear", None, 0.0)]
    assert len(b) == N_FOLDS
    assert all(r["reused_from_v_over_omega"] is None for r in b.values())


def test_comparisons_form_one_holm_family() -> None:
    family = summary.comparisons(summary._rows(_records()))
    # A − B at two grid points; A at two, B at one, each against two baselines.
    assert [c["kind"] for c in family].count("A − B") == 2
    assert [c["kind"] for c in family].count("arm − baseline") == (2 + 1) * 2
    raw = [c["p"] for c in family]
    assert [c["p_holm"] for c in family] == pytest.approx(holm_correction(raw).tolist())
    won = next(c for c in family if c["kind"] == "A − B" and c["a"][2] == 1.0)
    # Won in all 13 folds: the exact two-sided floor, 2/2^13, and it survives Holm over 8.
    assert won["n_pairs"] == N_FOLDS
    assert won["p"] == pytest.approx(2 / 2**N_FOLDS)
    assert won["min_p"] == pytest.approx(2 / 2**N_FOLDS)
    assert won["effect"] == pytest.approx(1.0)
    assert won["p_holm"] == pytest.approx(len(family) * 2 / 2**N_FOLDS)


def test_a_tie_in_every_fold_is_undefined_and_counts_as_one() -> None:
    records = [r for r in _records() if r["arm"] in ("A", "B") and r["v_over_omega"] == 1.0]
    for r in records:
        if r["arm"] == "A":
            r["mcc"] = next(
                b["mcc"] for b in records if b["arm"] == "B" and b["fold"] == r["fold"]
            )
    (only,) = summary.comparisons(summary._rows(records))
    assert math.isnan(only["p"]) and only["n_pairs"] == 0
    assert only["p_holm"] == 1.0


def test_summarise_prints_the_tracking_correlation() -> None:
    text = summary.summarise(_records(), permutations=99)
    assert "Holm family: 8 comparisons" in text
    # The A − B grid: ΔMCC 0.36 over the 13 folds, Holm p 8 · 2/2^13, significant.
    assert "| 1 | +0.360 (0.0020 ✓) |" in text
    # Two grid points are too few for a rank correlation: it is reported as missing, with n.
    assert "| linear | fisher_ratio | – | 2 | – |" in text
    # The Fisher ratio of A at V/Ω = 1 (2 in every fold) beside that of B (0.5).
    assert "| 1 | 0 | 1.000 | 1.000 | 2.000 | 0.500 |" in text
