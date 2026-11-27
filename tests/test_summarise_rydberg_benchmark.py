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

import dataclasses
import hashlib
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
    b = rows[("B", "linear", None, None, 0.0)]
    assert len(b) == N_FOLDS
    assert all(r["reused_from_v_over_omega"] is None for r in b.values())


def _score_cells(text: str, row: str) -> list[str]:
    """The cells of the scores row that starts with ``row``, e.g. ``"| A | linear | 1 | 0 |"``."""
    (line,) = [line for line in text.splitlines() if line.startswith(row)]
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_the_scores_add_the_feature_seconds_of_the_tuning_folds() -> None:
    """
    #785: once a line records ``tuning["feature_seconds"]``, the scores table has a
    ``total feature s`` column after ``feature s``, the mean over the row's folds of
    ``feature_seconds + Σ_j tuning["feature_seconds"][j]``; arm B's from its computed
    lines, as ``feature s`` is.  A line without tuning adds nothing, and a row with a line
    from before #785 (``tuning`` without the field) shows ``–``.  Without such a line the
    table has no such column.
    """
    records = _records()  # feature_seconds 1.0, 0.0 on the reused B lines; no "tuning"
    plain = summary.summarise(records, permutations=99)
    assert "total feature s" not in plain
    for r in records:
        fold = r["fold"]
        if r["arm"] == "A" and r["v_over_omega"] == 1.0:
            # 1 + 0.25 + 0.25 · fold: mean 1.25 + 0.25 · 6 = 2.75 over folds 0 … 12
            r["tuning"] = {"feature_seconds": [0.25, 0.25 * fold]}
        elif r["arm"] == "A":
            r["tuning"] = {"feature_seconds": [1.0]} if fold else {"scores": [0.5]}
        elif r["arm"] == "B":
            # the computed line: 1 + 2 + 1 = 4; a reused one carries the same tuning seconds
            r["tuning"] = {"feature_seconds": [2.0, 1.0]}
        elif r["head"] == "linear":
            r["tuning"] = None
    text = summary.summarise(records, permutations=99)
    assert "| dim | params | feature s | total feature s | fit s |" in text
    assert "\n|" + "---|" * 14 + "\n" in text
    # cells: ..., dim, params, feature s, total feature s, fit s
    assert _score_cells(text, "| A | linear | 1 | 0 |")[-3:] == ["1.00", "2.75", "0.10"]
    assert _score_cells(text, "| A | linear | 10 | 0 |")[-3:] == ["1.00", "–", "0.10"]
    assert _score_cells(text, "| B | linear | – | 0 |")[-3:] == ["1.00", "4.00", "0.10"]
    assert _score_cells(text, "| C | linear |")[-3:] == ["1.00", "1.00", "0.10"]
    assert _score_cells(text, "| C | mlp |")[-3:] == ["1.00", "1.00", "0.10"]

    def outside_scores(text: str) -> str:
        above, scores = text.split("### Scores", 1)
        return above + scores.split("### A against B", 1)[1]

    assert outside_scores(text) == outside_scores(plain)  # only the scores table changes


def test_comparisons_form_one_holm_family() -> None:
    family = summary.comparisons(summary._rows(_records()))
    # A − B at two grid points; A at two, B at one, each against two baselines.
    assert [c["kind"] for c in family].count("A − B") == 2
    assert [c["kind"] for c in family].count("arm − baseline") == (2 + 1) * 2
    raw = [c["p"] for c in family]
    assert [c["p_holm"] for c in family] == pytest.approx(holm_correction(raw).tolist())
    won = next(c for c in family if c["kind"] == "A − B" and c["a"][3] == 1.0)
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


def test_tracking_pairs_a_minus_b_of_measure_and_mcc_per_grid_point() -> None:
    """ρ is +1 where A − B of a measure rises with A − B of MCC, −1 where it falls."""
    records = []
    for fold in range(N_FOLDS):
        records.append(_line("B", "linear", fold, 0.1, v=1.0, gamma=0.0))
        for k, v in enumerate((1.0, 3.0, 10.0, 30.0)):
            # ΔMCC = 0.1 k; ΔFisher = (1 + k) − 0.5 rises with it, Δrank = −0.5 k falls.
            records.append(
                _line(
                    "A",
                    "linear",
                    fold,
                    0.1 + 0.1 * k,
                    v=v,
                    gamma=0.0,
                    fisher_ratio=1.0 + k,
                    effective_rank=3.0 - 0.5 * k,
                )
            )
    text = summary.mechanism_tables(
        summary._rows(records), part="train", permutations=99, seed=0, heads=["linear"]
    )
    assert "| linear | fisher_ratio | 1.00 | 4 |" in text
    assert "| linear | effective_rank | -1.00 | 4 |" in text


def test_summarise_prints_the_tracking_correlation() -> None:
    text = summary.summarise(_records(), permutations=99)
    assert "Holm family: 8 comparisons" in text
    # The A − B grid: ΔMCC 0.36 over the 13 folds, Holm p 8 · 2/2^13, significant.
    assert "| 1 | +0.360 (0.0020 ✓) |" in text
    # Two grid points are too few for a rank correlation: it is reported as missing, with n.
    assert "| linear | fisher_ratio | – | 2 | – |" in text
    # The Fisher ratio of A at V/Ω = 1 (2 in every fold) beside that of B (0.5).
    assert "| 1 | 0 | 1.000 | 1.000 | 2.000 | 0.500 |" in text


def _two_geometries() -> list[dict[str, Any]]:
    """
    :func:`_records` (chain lines without the field, as before #733) and arm A on a 2 × 2
    grid at the same grid points, with B reused for the grid lines as the runner does.
    """
    records = _records()
    for r in list(records):
        if r["arm"] == "A":
            # The grid wins every fold at both V/Ω, by more than the chain at V/Ω = 1.
            records.append({**r, "geometry": "grid", "grid_shape": [2, 2], "mcc": r["mcc"] + 1})
        if r["arm"] == "B":
            records.append(
                {
                    **r,
                    "geometry": "grid",
                    "grid_shape": [2, 2],
                    "reused_from_v_over_omega": 1.0,
                    "reused_from_geometry": "chain",
                }
            )
    return records


def test_lines_without_a_geometry_are_the_chain_and_a_grid_is_labelled_by_its_shape() -> None:
    assert summary.geometry_label({"arm": "A"}) == "chain"
    assert summary.geometry_label({"arm": "A", "geometry": "ring", "grid_shape": None}) == "ring"
    assert summary.geometry_label({"geometry": "grid", "grid_shape": [2, 3]}) == "grid-2x3"


def test_arm_a_is_grouped_by_geometry_and_b_counts_once() -> None:
    rows = summary._rows(_two_geometries())
    a_keys = sorted((k[2], k[3]) for k in rows if k[0] == "A")
    assert a_keys == [("chain", 1.0), ("chain", 10.0), ("grid-2x2", 1.0), ("grid-2x2", 10.0)]
    # One B row, its computed lines only, although the grid lines repeat it.
    (b_key,) = [k for k in rows if k[0] == "B"]
    assert b_key == ("B", "linear", None, None, 0.0)
    assert all(r.get("geometry") is None for r in rows[b_key].values())
    assert summary._geometries(rows) == ["chain", "grid-2x2"]


def test_the_geometries_of_a_dataset_form_one_holm_family() -> None:
    family = summary.comparisons(summary._rows(_two_geometries()))
    # A − B at four rows (two geometries × two V/Ω); A at four, B at one against two baselines.
    assert [c["kind"] for c in family].count("A − B") == 4
    assert [c["kind"] for c in family].count("arm − baseline") == (4 + 1) * 2
    raw = [c["p"] for c in family]
    assert [c["p_holm"] for c in family] == pytest.approx(holm_correction(raw).tolist())
    grid = [c for c in family if c["kind"] == "A − B" and c["a"][2] == "grid-2x2"]
    assert all(c["b"] == ("B", "linear", None, None, 0.0) for c in grid)
    # MCC + 1 over the chain's: A − B of the grid at V/Ω = 1 is that of the chain plus 1.
    chain = next(c for c in family if c["kind"] == "A − B" and c["a"][2:4] == ("chain", 1.0))
    assert grid[0]["a"][3] == 1.0
    assert grid[0]["mean_diff"] == pytest.approx(chain["mean_diff"] + 1.0)


def test_one_geometry_prints_what_it_printed_before_and_two_print_each() -> None:
    one = summary.summarise(_records(), permutations=99, long=True)
    assert "geometry" not in one
    assert "Head `linear`:" in one
    text = summary.summarise(_two_geometries(), permutations=99, long=True)
    assert "Holm family: 14 comparisons" in text
    assert "| arm | head | geometry | V/Ω | γ/Ω |" in text
    assert "| A | linear | grid-2x2 | 1 | 0 |" in text
    assert "| B | linear | – | – | 0 |" in text
    assert text.count("Head `linear`, geometry `chain`:") == 1
    assert text.count("Head `linear`, geometry `grid-2x2`:") == 1
    assert text.count("Geometry `grid-2x2`:") == 1
    assert "| A/linear grid-2x2 V/Ω=1 γ/Ω=0 | B/linear γ/Ω=0 |" in text
    # The grid's A − B table, ΔMCC 1.36 at V/Ω = 1, its Holm p 14 · 2/2^13.
    grid_section = text.split("Head `linear`, geometry `grid-2x2`:")[1]
    assert grid_section.split("\n\n")[1].splitlines()[2].startswith("| 1 | +1.360 (0.0034 ✓) |")


# ---------------------------------------------------------------------------
# One dataset, one data fingerprint (#742)
# ---------------------------------------------------------------------------

bench = summary._runner()

#: ``fingerprint`` of the default positive control (4 atoms, 600 samples, random_state 0,
#: chain), as pinned in tests/test_benchmark_rydberg_features.py and recorded in every
#: control line of docs/results/rydberg_features_benchmark.jsonl.xz.
CHAIN_CONTROL_SHA = "0f76549d09e84e908f062cdaf914bbcc0470a9cde46fbea4a27bccd17695e593"
#: Lines per fold of :func:`_records`: A at two V/Ω, B computed and reused, C with two heads.
LINES_PER_FOLD = 6


def _control_run(**changes: Any) -> tuple[list[dict[str, Any]], str]:
    """
    :func:`_records` as the lines of one positive-control run of ``bench.Settings(**changes)``.

    Each line carries the fingerprint of the control that run draws (``load_dataset``) and its
    ``label_bonds``; arm A carries the run's geometry, as the runner writes them.
    """
    settings = bench.Settings(**changes)
    X, y, _ = bench.load_dataset("positive-control", settings)
    sha: str = bench.fingerprint(X, y)
    (geometry,) = settings.geometries
    bonds = [[j, k] for j, k in bench.control_bonds(settings)]
    lines = []
    for r in _records():
        line = {**r, "dataset": "positive-control", "data_sha256": sha, "label_bonds": bonds}
        if r["arm"] == "A":
            line["geometry"] = geometry
        lines.append(line)
    return lines, sha


def test_two_control_runs_of_different_geometries_in_one_input_are_two_groups() -> None:
    """
    #741: lines of two labels of one dataset are two groups, keyed by ``label_bonds``, each
    with its one fingerprint.  Each prints what it prints on its own, under a heading that
    names its bonds, in the order first seen.
    """
    chain, chain_sha = _control_run(geometries=("chain",))
    ring, ring_sha = _control_run(geometries=("ring",))
    # The chain's is the committed control's fingerprint; the ring's label differs.
    assert chain_sha == CHAIN_CONTROL_SHA
    assert ring_sha != chain_sha
    assert chain[0]["label_bonds"] != ring[0]["label_bonds"]
    alone = {
        "chain": summary.summarise(chain, permutations=99),
        "ring": summary.summarise(ring, permutations=99),
    }
    assert all(text.startswith("## positive-control\n") for text in alone.values())
    bonds = {"chain": "[[0, 1], [1, 2], [2, 3]]", "ring": "[[0, 1], [0, 3], [1, 2], [2, 3]]"}
    headed = {
        name: text.replace(
            "## positive-control\n", f"## positive-control, label_bonds {bonds[name]}\n", 1
        )
        for name, text in alone.items()
    }
    assert summary.summarise(chain + ring, permutations=99) == "\n".join(
        (headed["chain"], headed["ring"])
    )
    assert summary.summarise(ring + chain, permutations=99) == "\n".join(
        (headed["ring"], headed["chain"])
    )
    groups = summary.groups(ring + chain)
    assert list(groups) == [
        ("positive-control", ((0, 1), (0, 3), (1, 2), (2, 3))),
        ("positive-control", ((0, 1), (1, 2), (2, 3))),
    ]
    assert [len(g) for g in groups.values()] == [LINES_PER_FOLD * N_FOLDS] * 2


def test_two_control_runs_of_different_seeds_in_one_input_are_refused() -> None:
    first, first_sha = _control_run(geometries=("chain",))
    second, second_sha = _control_run(geometries=("chain",), random_state=1)
    assert first_sha == CHAIN_CONTROL_SHA != second_sha
    # Same geometry, so the same label_bonds: only the fingerprint tells the two apart.
    assert first[0]["label_bonds"] == second[0]["label_bonds"]
    with pytest.raises(ValueError, match="'positive-control'") as caught:
        summary.summarise(first + second, permutations=99)
    message = str(caught.value)
    assert first_sha in message and second_sha in message
    assert "(label_bonds [[0, 1], [1, 2], [2, 3]])" in message


def test_lines_without_a_fingerprint_and_lines_with_one_are_refused() -> None:
    """A line written without ``data_sha256`` is not known to share the others' data."""
    run, sha = _control_run(geometries=("chain",))
    unrecorded = [
        {
            **{k: v for k, v in r.items() if k != "data_sha256"},
            "dataset": "positive-control",
            "label_bonds": run[0]["label_bonds"],
        }
        for r in _records()
    ]
    with pytest.raises(ValueError, match="not recorded") as caught:
        summary.summarise(run + unrecorded, permutations=99)
    assert sha in str(caught.value)
    assert summary.data_fingerprints(unrecorded) == {None: LINES_PER_FOLD * N_FOLDS}


def test_a_line_without_label_bonds_is_a_group_of_its_own() -> None:
    """
    A line without ``label_bonds`` (missing or ``null``: a UCI dataset, or a control written
    before #738, whose label is the chain's) is keyed ``None``.  It is not merged with control
    lines that record the chain's bonds, even of the same data, and the guard applies within
    the ``None`` group too.
    """
    run, sha = _control_run(geometries=("chain",))
    old = [{**r, "dataset": "positive-control", "data_sha256": sha} for r in _records()]
    null = [{**r, "label_bonds": None} for r in old]
    assert "label_bonds" not in old[0]
    assert list(summary.groups(run + old + null)) == [
        ("positive-control", ((0, 1), (1, 2), (2, 3))),
        ("positive-control", None),
    ]
    assert len(summary.groups(old + null)[("positive-control", None)]) == 2 * len(old)
    text = summary.summarise(run + old, permutations=99)
    assert text.count("## positive-control, label_bonds [[0, 1], [1, 2], [2, 3]]\n") == 1
    assert text.count("## positive-control, label_bonds not recorded\n") == 1
    assert text.count("Holm family: 8 comparisons") == 2
    # A dataset of one group keeps the heading from before #741, whatever its bonds.
    assert summary.summarise(old, permutations=99).startswith("## positive-control\n")
    other = [{**r, "data_sha256": "f" * 64} for r in old]
    with pytest.raises(ValueError, match="'positive-control' has lines of 2"):
        summary.summarise(old + other, permutations=99)


def _family_line(fold: int, arm: str, head: str, mcc: float, bonds: Any, sha: str) -> dict:
    v, gamma = (1.0, 0.1) if arm in ("A", "B") else (None, None)
    # The chain of 4 atoms has 3 bonds, the ring 4.
    geometry = None if arm != "A" else ("chain" if len(bonds) == 3 else "ring")
    return {
        **_line(arm, head, fold, mcc, v=v, gamma=gamma),
        "dataset": "positive-control",
        "data_sha256": sha,
        "label_bonds": bonds,
        "geometry": geometry,
    }


def test_two_labels_form_two_holm_families_by_hand() -> None:
    """
    Two labels of one control, six folds each, arm A and B linear at one grid point and both
    baselines at MCC 0, so each group has m = 5 comparisons: A − B, A and B against C/linear
    and C/mlp.  The exact two-sided signed-rank p-values over 6 untied pairs, by hand:

    * every pair won: one sign pattern of 2^6 reaches W− = 0, ``p = 2 · 1/64 = 0.03125``;
    * one lost, of rank 1: W− ≤ 1 for ∅ and {1}, ``p = 2 · 2/64 = 0.0625``;
    * every pair tied: undefined, a slot of ``p = 1`` in its family.

    Chain: A = 0.1 … 0.6, B = A − (−0.01, 0.02, …, 0.06): A − B 0.0625, the four against
    the baselines 0.03125.  Holm over m = 5, ascending: 5 · 0.03125, then
    max(0.15625, 4 · 0.03125), … and max(0.15625, 1 · 0.0625), so 0.15625 throughout.
    Ring: A = B = 0.6 … 0.1: A − B undefined (1), the others 0.03125, so 0.15625 four times
    and max(0.15625, 1 · 1) = 1.  One family of the ten would instead give the chain's
    smallest 10 · 0.03125 = 0.3125.
    """
    chain_bonds = [[0, 1], [1, 2], [2, 3]]
    ring_bonds = [[0, 1], [0, 3], [1, 2], [2, 3]]
    lines = []
    for fold in range(6):
        a = 0.1 * (fold + 1)
        b = a - (-0.01 if fold == 0 else 0.01 * (fold + 1))
        ring_a = 0.1 * (6 - fold)
        for bonds, sha, mccs in (
            (chain_bonds, "a" * 64, {"A": a, "B": b}),
            (ring_bonds, "b" * 64, {"A": ring_a, "B": ring_a}),
        ):
            lines.append(_family_line(fold, "A", "linear", mccs["A"], bonds, sha))
            lines.append(_family_line(fold, "B", "linear", mccs["B"], bonds, sha))
            lines.append(_family_line(fold, "C", "linear", 0.0, bonds, sha))
            lines.append(_family_line(fold, "C", "mlp", 0.0, bonds, sha))
    groups = summary.groups(lines)
    families = {key[1]: summary.comparisons(summary._rows(recs)) for key, recs in groups.items()}
    chain, ring = families[((0, 1), (1, 2), (2, 3))], families[((0, 1), (0, 3), (1, 2), (2, 3))]

    def by_name(family: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[float, float]]:
        return {(c["a"][0], c["b"][0] + c["b"][1]): (c["p"], c["p_holm"]) for c in family}

    assert by_name(chain) == {
        ("A", "Blinear"): (pytest.approx(0.0625), pytest.approx(0.15625)),
        ("A", "Clinear"): (pytest.approx(0.03125), pytest.approx(0.15625)),
        ("A", "Cmlp"): (pytest.approx(0.03125), pytest.approx(0.15625)),
        ("B", "Clinear"): (pytest.approx(0.03125), pytest.approx(0.15625)),
        ("B", "Cmlp"): (pytest.approx(0.03125), pytest.approx(0.15625)),
    }
    ring_p = by_name(ring)
    assert math.isnan(ring_p.pop(("A", "Blinear"))[0])
    assert next(c["p_holm"] for c in ring if c["kind"] == "A − B") == 1.0
    assert ring_p == dict.fromkeys(
        [("A", "Clinear"), ("A", "Cmlp"), ("B", "Clinear"), ("B", "Cmlp")],
        (pytest.approx(0.03125), pytest.approx(0.15625)),
    )
    # As printed: one family of 5 per label, the chain's A − B at Holm p 0.15625.
    text = summary.summarise(lines, permutations=99, long=True)
    assert text.count("Holm family: 5 comparisons") == 2
    chain_text, ring_text = text.split("## positive-control, label_bonds ")[1:]
    assert chain_text.startswith("[[0, 1], [1, 2], [2, 3]]\n")
    # Mean ΔMCC 0.19/6; floor 2/2^6; r_rb = (20 − 1)/21 = +0.90.
    row = "| A/linear V/Ω=1 γ/Ω=0.1 | B/linear γ/Ω=0.1 | 6 |"
    assert f"{row} +0.032 | 6 | 0.0625 | 0.0312 | 0.1562 | +0.90 |" in chain_text
    assert f"{row} +0.000 | 0 | – | – | 1.0000 | +0.00 |" in ring_text


def test_each_dataset_may_have_its_own_fingerprint() -> None:
    """The guard is per dataset: two datasets of one fingerprint each are summarised as before."""
    run, sha = _control_run(geometries=("chain",))
    other = [{**r, "data_sha256": "f" * 64} for r in _records()]
    assert summary.data_fingerprints(run) == {sha: LINES_PER_FOLD * N_FOLDS}
    text = summary.summarise(run + other, permutations=99)
    assert text.count("Holm family: 8 comparisons") == 2
    assert text.endswith(summary.summarise(other, permutations=99))


#: The line under the scores table of every committed results file (#747): their lines
#: predate the runner's ``pca`` record (#525).
NOT_RECORDED = "Degenerate PCA fits (#525): fold fit not recorded, a tuning fit not recorded."


def _without_degeneracy(text: str, sections: int) -> str:
    """
    ``text`` with its ``sections`` :data:`NOT_RECORDED` lines removed, each with the blank line
    after it: the output from before #747.  Each must sit right under a scores table.
    """
    block = f"|\n\n{NOT_RECORDED}\n\n### A against B"
    assert text.count(NOT_RECORDED) == text.count(block) == sections
    return text.replace(f"{NOT_RECORDED}\n\n", "")


def test_the_committed_results_summarise_as_before() -> None:
    """
    The summary of the committed results, ``--long`` with 99 permutations, is byte for byte
    that of ``main`` at 29c8931, before the fingerprint guard (#742), apart from the line of
    #747 under each scores table, which says the PCA degeneracy is not recorded: the SHA-256
    below is that of the script's output there.
    """
    records = summary.read_records(ROOT / "docs/results/rydberg_features_benchmark.jsonl.xz")
    assert {r["dataset"]: r["data_sha256"] for r in records}["positive-control"] == (
        CHAIN_CONTROL_SHA
    )
    assert not any("pca" in r for r in records)
    text = summary.summarise(records, permutations=99, long=True)
    assert (
        hashlib.sha256(_without_degeneracy(text, 2).encode()).hexdigest()
        == "171f37e71e738a6378621c86e241243d6d5f451610ba6cc83d5ebd19c818c315"
    )


#: The committed results of the ring and grid run (#737): one file per geometry of the
#: positive control (a run with the control takes one geometry, #738) and one of Iranian Churn
#: over all three.  Per file: its datasets, each with its data fingerprint and label bonds
#: (``None`` for a UCI dataset), the geometries of its A lines, and the SHA-256 of its
#: ``--long`` summary with 99 permutations.
GEOMETRY_RESULTS: tuple[tuple[str, dict[str, tuple[str, Any]], set[str], str], ...] = (
    (
        "rydberg_features_geometries_control_chain.jsonl.xz",
        {"positive-control": (CHAIN_CONTROL_SHA, [[0, 1], [1, 2], [2, 3]])},
        {"chain"},
        "ea7f09e360aa4bb89889bfd78c4e07a4e86a0bf05a3c84ca758707d79e0deb36",
    ),
    (
        "rydberg_features_geometries_control_ring.jsonl.xz",
        {
            "positive-control": (
                "c76aa50e12d80c25a2b3c58f7575f656fdfa46f3c499bcd554dcddf9d77994a2",
                [[0, 1], [0, 3], [1, 2], [2, 3]],
            )
        },
        {"ring"},
        "250d77f21b6e75d806bbd9f9ce38f2a477df582e92404ce1733fdf7d1d12b336",
    ),
    (
        "rydberg_features_geometries_control_grid.jsonl.xz",
        {
            "positive-control": (
                "c0da2581bc06e975bc4cddacc8b93247b7614b47c0e64cfe20ce39909eaccfa9",
                [[0, 1], [0, 2], [1, 3], [2, 3]],
            )
        },
        {"grid-2x2"},
        "934726139f32e050bb120c9bac0d8de0c1e45afc2f9854b5516b3904dcdf1752",
    ),
    (
        "rydberg_features_geometries_iranian_churn.jsonl.xz",
        {
            "iranian-churn": (
                "1c24f5e361662ad22e66d998438378c873a403193804090212314ff0d7acae63",
                None,
            )
        },
        {"chain", "ring", "grid-2x2"},
        "daf2f4cf7c25c4c6bf97d69a3354bd129fabb961faeb872982bd5ef0956b5369",
    ),
)


@pytest.mark.parametrize(("name", "datasets", "geometries", "digest"), GEOMETRY_RESULTS)
def test_the_committed_geometry_results_summarise(
    name: str, datasets: dict[str, tuple[str, Any]], geometries: set[str], digest: str
) -> None:
    """
    Each results file of the ring and grid run (#737) holds one data fingerprint per dataset
    (the guard of #742 refuses anything else), the label bonds of its geometry, A lines of
    the stated geometries and 13 folds, and summarises without error to a fixed output: the
    output from before #747, with a line under the scores table that says the PCA degeneracy
    is not recorded.
    """
    records = summary.read_records(ROOT / "docs/results" / name)
    found = {(r["dataset"], r["data_sha256"], repr(r.get("label_bonds"))) for r in records}
    assert found == {(d, sha, repr(bonds)) for d, (sha, bonds) in datasets.items()}
    assert {bench.geometry_label(r) for r in records if r["arm"] == "A"} == geometries
    assert {r["n_splits"] for r in records} == {13}
    assert not any("pca" in r for r in records)
    text = summary.summarise(records, permutations=99, long=True)
    assert hashlib.sha256(_without_degeneracy(text, 1).encode()).hexdigest() == digest


def test_the_three_committed_controls_read_together_are_three_families() -> None:
    """
    #741: the three control files of #737 read as one input (refused by #742's guard before)
    print each file's own summary, pinned above, one after the other, each headed by its
    label bonds: three families of 190, not one of 570.
    """
    controls = GEOMETRY_RESULTS[:3]
    records = [
        r for name, *_ in controls for r in summary.read_records(ROOT / "docs/results" / name)
    ]
    text = _without_degeneracy(summary.summarise(records, permutations=99, long=True), 3)
    sections = text.split("\n## positive-control, label_bonds ")
    assert sections[0].startswith("## positive-control, label_bonds ")
    sections[0] = sections[0].removeprefix("## positive-control, label_bonds ")
    assert len(sections) == 3
    for section, (_, datasets, _, digest) in zip(sections, controls, strict=True):
        bonds = datasets["positive-control"][1]
        assert section.startswith(f"{bonds}\n")
        alone = "## positive-control" + section.removeprefix(str(bonds))
        assert hashlib.sha256(alone.encode()).hexdigest() == digest
    assert text.count("Holm family: 190 comparisons") == 3


# ---------------------------------------------------------------------------
# Iranian Churn with 15 folds (#746)
# ---------------------------------------------------------------------------

#: The rerun of Iranian Churn on the chain, the ring and the ``2 × 2`` grid with 15 folds
#: (#746), every other option as in the 13-fold file above.
CHURN_15_FOLDS = "rydberg_features_geometries_iranian_churn_15_folds.jsonl.xz"
CHURN_SHA = "1c24f5e361662ad22e66d998438378c873a403193804090212314ff0d7acae63"
#: Per geometry, the A − B comparisons won or lost in all 15 folds and the significant ones
#: (Holm, α = 0.05), as the note's *What the results show* states them: every one a loss.
CHURN_15_A_MINUS_B = {"chain": (3, 3), "ring": (14, 14), "grid-2x2": (17, 17)}
#: Significant comparisons of the whole family of 526, as the note's *Statistics* states.
CHURN_15_SIGNIFICANT = 109
#: SHA-256 of the file's ``--long`` summary with 99 permutations.
CHURN_15_DIGEST = "8e08785df76289cdcc1f31e2f7ab5916984e87b84611491317b6ac2073866119"


def _signed_rank_p(diffs: np.ndarray, tol: float = 1e-12) -> float:
    """
    Two-sided exact Wilcoxon signed-rank p-value of ``diffs``, by enumerating every sign.

    Differences with ``|d| <= tol`` are dropped.  The ``n`` left get the rank of ``|d_i|``
    among them, ties (within ``tol``) sharing their mean rank:
    ``r_i = #{j : |d_j| < |d_i| − tol} + (#{j : ||d_j| − |d_i|| <= tol} + 1) / 2``.
    Under H0 each sign is + or − with probability 1/2, independently, so each of the
    ``2^n`` assignments ``s ∈ {0, 1}^n`` (bit ``i`` of the row index: pair ``i`` positive) is
    equally likely and gives ``W+ = Σ_i s_i r_i``.  With ``w`` the observed ``W+``,
    ``p = min(1, 2 · min(P(W+ >= w), P(W+ <= w)))``, the convention of
    :func:`~hqnn_forge.evaluation.wilcoxon_signed_rank`; ranks are multiples of 1/2, so the
    comparisons are exact in floating point.  An independent route to the value the
    summariser gets from that function's counting recursion.
    """
    d = diffs[np.abs(diffs) > tol]
    a = np.abs(d)
    below = (a[None, :] < a[:, None] - tol).sum(axis=1)
    tied = (np.abs(a[None, :] - a[:, None]) <= tol).sum(axis=1)
    ranks = below + (tied + 1) / 2.0
    n = d.size
    signs = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
    null = signs @ ranks
    w = float(ranks[d > 0].sum())
    return min(1.0, 2.0 * min(float(np.mean(null >= w)), float(np.mean(null <= w))))


def _holm(p: np.ndarray) -> np.ndarray:
    """
    Holm's step-down adjusted p-values: sorted ascending, the ``k``-th (from 0) becomes
    ``max_{j <= k} min(1, (m − j) · p_(j))``.
    """
    m = p.size
    order = np.argsort(p, kind="stable")
    steps = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.maximum.accumulate(steps)
    return out


@pytest.mark.slow
def test_the_committed_15_fold_churn_results_reach_alpha() -> None:
    """
    #746: the 15-fold Iranian Churn file holds one data fingerprint (Iranian Churn's, as in
    #502), A lines of the three geometries, 15 folds of every row and the PCA record of #525.
    Its one Holm family has ``m = 3 · 56 + 2 · 179 = 526`` comparisons, for which
    ``2/2^15 < α/m``, so a comparison won or lost in all 15 folds is significant.  Every
    comparison's exact p-value and Holm p are recomputed from the raw MCCs, paired by fold
    here, with :func:`_signed_rank_p` and :func:`_holm`, and the significant ones counted.
    The ``--long`` summary with 99 permutations is pinned, as for the files above.
    """
    records = summary.read_records(ROOT / "docs/results" / CHURN_15_FOLDS)
    assert {(r["dataset"], r["data_sha256"], r["label_bonds"]) for r in records} == {
        ("iranian-churn", CHURN_SHA, None)
    }
    assert {bench.geometry_label(r) for r in records if r["arm"] == "A"} == {
        "chain",
        "ring",
        "grid-2x2",
    }
    assert {r["n_splits"] for r in records} == {15}
    assert all(r["pca"] is not None for r in records)

    # MCC per row and fold from the raw lines, keyed as the summariser keys its rows:
    # (arm, head, geometry, V/Ω, γ/Ω), with None where an arm has no such coordinate.  Arm B
    # is written once per geometry and V/Ω with the same scores, so every copy must agree.
    mcc: dict[tuple[Any, ...], dict[int, float]] = {}
    for r in records:
        arm, head = r["arm"], r["head"]
        if arm == "A":
            geometry = bench.geometry_label(r)
            key = (arm, head, geometry, r["v_over_omega"], r["gamma_over_omega"])
        elif arm == "B":
            key = (arm, head, None, None, r["gamma_over_omega"])
        else:
            key = (arm, head, None, None, None)
        folds = mcc.setdefault(key, {})
        assert folds.setdefault(r["fold"], r["mcc"]) == r["mcc"]
    assert {len(f) for f in mcc.values()} == {15}
    a_rows = [k for k in mcc if k[0] == "A"]
    others = [k for k in mcc if k[0] in ("B", "D", "D-hybrid")]
    assert (len(a_rows), len(others)) == (3 * 2 * 7 * 4, 2 * 4 + 2 + 1)

    bases = [("C", head, None, None, None) for head in ("linear", "mlp")]
    pairs = [(k, ("B", k[1], None, None, k[4])) for k in a_rows]
    pairs += [(k, base) for k in a_rows + others for base in bases]
    m = len(pairs)
    assert m == 526
    assert 2 / 2**15 < summary.ALPHA / m < 2 / 2**14

    diffs = [np.array([mcc[a][f] - mcc[b][f] for f in range(15)]) for a, b in pairs]
    p = np.array([_signed_rank_p(d) for d in diffs])
    holm = _holm(p)
    ours = {pair: (pv, q) for pair, pv, q in zip(pairs, p, holm, strict=True)}

    family = summary.comparisons(summary._rows(records))
    assert {(c["a"], c["b"]) for c in family} == set(ours)
    for c in family:
        pv, q = ours[(c["a"], c["b"])]
        assert c["p"] == pytest.approx(pv, rel=1e-12, abs=0)
        assert c["p_holm"] == pytest.approx(q, rel=1e-12, abs=0)

    # Won or lost in all 15 folds: exact p at the floor 2/2^15, Holm p m · 2/2^15 < α.
    unanimous = np.array([bool(np.all(d > 0) or np.all(d < 0)) for d in diffs])
    assert np.allclose(p[unanimous], 2 / 2**15, rtol=1e-12, atol=0)
    assert np.allclose(holm[unanimous], m * 2 / 2**15, rtol=1e-12, atol=0)
    significant = holm <= summary.ALPHA
    # The note's *Statistics*: every significant comparison is won or lost in all 15 folds.
    assert np.array_equal(significant, unanimous)
    a_minus_b = np.arange(m) < len(a_rows)
    counts = {
        geometry: (
            int((unanimous & a_minus_b & in_geometry).sum()),
            int((significant & a_minus_b & in_geometry).sum()),
        )
        for geometry in ("chain", "ring", "grid-2x2")
        for in_geometry in [np.array([a[2] == geometry for a, _ in pairs])]
    }
    assert counts == CHURN_15_A_MINUS_B
    assert int(significant.sum()) == CHURN_15_SIGNIFICANT
    # No arm is significantly better than B or than a baseline: every significant one is a loss.
    assert all(float(d.mean()) < 0 for d, sig in zip(diffs, significant, strict=True) if sig)

    text = summary.summarise(records, permutations=99, long=True)
    assert hashlib.sha256(text.encode()).hexdigest() == CHURN_15_DIGEST


# ---------------------------------------------------------------------------
# One group, one split (#753)
# ---------------------------------------------------------------------------

#: The tail of every refusal of :func:`summary._check_one_split`.
SPLIT_ADVICE = (
    "Its folds and Holm family would pair lines of different splits by fold index; "
    "summarise each run's lines on their own."
)


def _refusal(records: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]) -> str:
    """The message ``main`` refuses ``records`` with, read as one input, after printing nothing."""
    with pytest.raises(ValueError) as caught:
        summary.main(["joined.jsonl", "--permutations", "99"], read=lambda _: records)
    assert capsys.readouterr() == ("", "")
    return str(caught.value)


@pytest.mark.slow
def test_the_two_committed_churn_runs_read_together_are_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    #753: the 13-fold (#745) and the 15-fold (#746) Iranian Churn files have the same data
    and no label bonds, so #742's guard lets them through as one group, whose fold ``k`` would
    pair a fold of one split with a fold of another.  Read as one input, in either order, they
    are refused before anything is printed, naming the dataset and each ``n_splits`` with its
    number of lines (the files' lengths, counted here).
    """
    thirteen = summary.read_records(ROOT / "docs/results" / GEOMETRY_RESULTS[3][0])
    fifteen = summary.read_records(ROOT / "docs/results" / CHURN_15_FOLDS)
    assert (len(thirteen), len(fifteen)) == (4433, 5115)
    assert {r["n_splits"] for r in thirteen} == {13} and {r["n_splits"] for r in fifteen} == {15}
    # One fingerprint and one label over both: only the split tells the two runs apart.
    assert summary.data_fingerprints(thirteen + fifteen) == {CHURN_SHA: 4433 + 5115}
    assert list(summary.groups(thirteen + fifteen)) == [("iranian-churn", None)]
    for first, second in ((thirteen, fifteen), (fifteen, thirteen)):
        found = ", ".join(f"{r[0]['n_splits']} ({len(r)} lines)" for r in (first, second))
        assert _refusal(first + second, capsys) == (
            f"dataset 'iranian-churn' has lines of 2 fold counts (n_splits): {found}.  "
            + SPLIT_ADVICE
        )
    assert "13 (4433 lines), 15 (5115 lines)" in _refusal(thirteen + fifteen, capsys)


def _with_split(
    lines: list[dict[str, Any]], n_splits: int | None, test_idx: Any
) -> list[dict[str, Any]]:
    """
    ``lines`` with ``n_splits`` and, per line, ``test_idx(fold)`` as the runner writes them.

    ``None`` drops the field, as on a line written without it.
    """
    out = []
    for r in lines:
        split = {"n_splits": n_splits, "test_idx": test_idx(r["fold"])}
        line = {k: v for k, v in r.items() if k not in split}
        out.append({**line, **{k: v for k, v in split.items() if v is not None}})
    return out


def _partition(fold: int) -> list[int]:
    """Fold ``fold`` of 13 over 78 rows: rows ``fold, fold + 13, …, fold + 65``."""
    return list(range(fold, 78, 13))


def test_two_control_runs_of_different_fold_counts_are_refused_with_their_label_bonds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    The same refusal in a group with label bonds, which it names as #742's guard does; a line
    without ``n_splits`` counts as one more fold count, ``not recorded``.
    """
    run, sha = _control_run(geometries=("chain",))
    thirteen = _with_split(run, 13, _partition)
    fifteen = _with_split(run, 15, lambda k: list(range(k, 78, 15)))
    assert summary.data_fingerprints(thirteen + fifteen) == {sha: 2 * LINES_PER_FOLD * N_FOLDS}
    lines = LINES_PER_FOLD * N_FOLDS
    assert _refusal(thirteen + fifteen, capsys) == (
        "dataset 'positive-control' (label_bonds [[0, 1], [1, 2], [2, 3]]) has lines of 2 fold "
        f"counts (n_splits): 13 ({lines} lines), 15 ({lines} lines).  " + SPLIT_ADVICE
    )
    old = _with_split(run, None, _partition)
    assert "n_splits" not in old[0]
    assert f"13 ({lines} lines), not recorded ({lines} lines)." in _refusal(thirteen + old, capsys)


def test_different_splits_in_different_groups_are_not_refused() -> None:
    """
    The check is per group: a 13-fold chain control and a 15-fold ring control, two labels and
    so two groups, differ in ``n_splits`` and in every fold's ``test_idx`` but are each one
    split, so they are summarised as each run on its own under a heading naming its bonds.
    A ``test_idx`` written in another order is the same test set.
    """
    chain, _ = _control_run(geometries=("chain",))
    ring, _ = _control_run(geometries=("ring",))
    thirteen = _with_split(chain, 13, _partition)
    fifteen = _with_split(ring, 15, lambda k: list(range(k, 78, 15)))
    alone = {
        "[[0, 1], [1, 2], [2, 3]]": summary.summarise(thirteen, permutations=99),
        "[[0, 1], [0, 3], [1, 2], [2, 3]]": summary.summarise(fifteen, permutations=99),
    }
    headed = [
        text.replace("## positive-control\n", f"## positive-control, label_bonds {bonds}\n", 1)
        for bonds, text in alone.items()
    ]
    assert summary.summarise(thirteen + fifteen, permutations=99) == "\n".join(headed)
    shuffled = _with_split(chain, 13, lambda k: _partition(k)[::-1])
    assert shuffled[0]["test_idx"] != thirteen[0]["test_idx"]
    assert summary.summarise(thirteen + shuffled, permutations=99) == next(iter(alone.values()))


def _digest(rows: list[int]) -> str:
    """The first 12 hex digits of the SHA-256 of ``rows`` written as ``"3,16,29"``."""
    return hashlib.sha256(",".join(map(str, rows)).encode()).hexdigest()[:12]


def test_lines_of_one_fold_count_but_other_test_sets_are_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Two runs of one UCI dataset (one fingerprint, no label bonds) with the same ``n_splits``
    but another split, as two ``--seed`` values draw: here the second swaps rows 68 and 72
    between folds 3 and 7.  Refused, naming the folds whose lines disagree and, for the first,
    each test set by its size and digest with its number of lines; a line without
    ``test_idx`` counts as one more test set, ``not recorded``.
    """
    first = _with_split([{**r, "data_sha256": "f" * 64} for r in _records()], 13, _partition)
    swap = {3: [3, 16, 29, 42, 55, 72], 7: [7, 20, 33, 46, 59, 68]}
    second = _with_split(first, 13, lambda k: swap.get(k, _partition(k)))
    assert _partition(3) == [3, 16, 29, 42, 55, 68] and _partition(7)[-1] == 72
    # Each run's folds partition the same 78 rows: only the assignment to folds differs.
    for run in (first, second):
        rows = sorted(i for k in range(N_FOLDS) for i in run[LINES_PER_FOLD * k]["test_idx"])
        assert rows == list(range(78))
    assert summary.data_fingerprints(first + second) == {"f" * 64: 2 * LINES_PER_FOLD * N_FOLDS}
    a, b = _digest(_partition(3)), _digest(swap[3])
    assert a != b
    assert _refusal(first + second, capsys) == (
        "dataset 'toy' has lines of 2 test sets (test_idx) for fold 3: "
        f"6 rows, sha256 {a} ({LINES_PER_FOLD} lines), "
        f"6 rows, sha256 {b} ({LINES_PER_FOLD} lines) "
        "(folds with more than one: [3, 7]).  " + SPLIT_ADVICE
    )
    # The same split, written twice, is one test set per fold: summarised as each run alone.
    assert summary.summarise(first + first, permutations=99) == summary.summarise(
        first, permutations=99
    )
    old = _with_split(first, 13, lambda k: None)
    assert "test_idx" not in old[0]
    assert (
        f"for fold 0: 6 rows, sha256 {_digest(_partition(0))} ({LINES_PER_FOLD} lines), "
        f"not recorded ({LINES_PER_FOLD} lines) (folds with more than one: "
        f"{list(range(N_FOLDS))})."
    ) in _refusal(first + old, capsys)


# ---------------------------------------------------------------------------
# One row, one measurement (#763)
# ---------------------------------------------------------------------------

#: The tail of every refusal of :func:`summary._check_one_measurement`.
MEASUREMENT_ADVICE = (
    "Its spread, intervals and paired tests would pool lines whose features were measured "
    "differently; summarise each run's lines on their own."
)
#: ``(measurement, shots, shot_stream)`` of a line of arm A, B or D with ``--shots 16`` since
#: #762 (a stream per matrix), of one written before it (no ``shot_stream``), and of exact
#: features, which arms C and ``D-hybrid`` always have.
PER_MATRIX = ("shots", 16, "per-matrix")
SHARED = ("shots", 16, None)
EXACT = ("exact", None, None)


def _measured(lines: list[dict[str, Any]], sampled: tuple[Any, ...]) -> list[dict[str, Any]]:
    """
    ``lines`` as a run writes them whose arms A and B were measured as ``sampled``.

    ``sampled`` is ``(measurement, shots, shot_stream)``; a ``shot_stream`` of ``None`` is left
    out of ``feature_map``, as before #762 and in every exact config.  Arm C is exact with a
    ``feature_map`` of ``null``, as the runner writes it.
    """
    measurement, shots, stream = sampled
    out = []
    for r in lines:
        if r["arm"] == "C":
            fields: dict[str, Any] = {"measurement": "exact", "shots": None, "feature_map": None}
        else:
            config = {"n_atoms": 4, "shots": shots}
            if stream is not None:
                config["shot_stream"] = stream
            fields = {"measurement": measurement, "shots": shots, "feature_map": config}
        out.append({**r, **fields})
    return out


def test_an_old_and_a_new_shots_run_of_one_row_are_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    #763: a shots run from before #762 (every fold from the one ``shot_seed`` stream) and one
    since (a stream per matrix), of the same data, label and folds, pass the guards of #742 and
    #753, and every row of A and B would pool both kinds of line.  Refused before anything is
    printed, at the first such row of the scores table, naming each value with its number of
    lines in the order first seen and how many of the 5 rows have more than one: A at two
    ``V/Ω`` and B (keyed by ``γ/Ω`` alone), not C, exact in both runs.
    """
    new, old = _measured(_records(), PER_MATRIX), _measured(_records(), SHARED)
    # One fingerprint and one split (neither recorded): only the measurement tells them apart.
    assert summary.data_fingerprints(new + old) == {None: 2 * LINES_PER_FOLD * N_FOLDS}
    assert list(summary.groups(new + old)) == [("toy", None)]
    found = (
        f"measurement shots, shots 16, shot_stream per-matrix ({N_FOLDS} lines); "
        f"measurement shots, shots 16, shot_stream not recorded ({N_FOLDS} lines)"
    )
    assert _refusal(new + old, capsys) == (
        "dataset 'toy' has lines of 2 feature measurements (measurement, shots, shot_stream) "
        f"in row A/linear chain V/Ω=1 γ/Ω=0: {found} (rows with more than one: 3 of 5).  "
        + MEASUREMENT_ADVICE
    )
    assert (
        f"shot_stream not recorded ({N_FOLDS} lines); measurement shots, shots 16, "
        f"shot_stream per-matrix ({N_FOLDS} lines)"
    ) in _refusal(old + new, capsys)


def test_an_exact_and_a_shots_run_of_one_row_are_refused_with_their_label_bonds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    The same for exact against shots lines, in a group with label bonds, which the refusal
    names as the guards of #742 and #753 do, and for two shots runs that differ only in
    ``--shots``.  A line without the fields counts as one more value, ``not recorded``, here
    beside exact lines: then C differs too, so all 5 rows do.
    """
    run, _ = _control_run(geometries=("chain",))
    exact, shots = _measured(run, EXACT), _measured(run, PER_MATRIX)
    assert _refusal(exact + shots, capsys) == (
        "dataset 'positive-control' (label_bonds [[0, 1], [1, 2], [2, 3]]) has lines of 2 "
        "feature measurements (measurement, shots, shot_stream) in row A/linear chain V/Ω=1 "
        f"γ/Ω=0: measurement exact, shots not recorded, shot_stream not recorded ({N_FOLDS} "
        f"lines); measurement shots, shots 16, shot_stream per-matrix ({N_FOLDS} lines) "
        "(rows with more than one: 3 of 5).  " + MEASUREMENT_ADVICE
    )
    # Two shots runs that differ only in --shots.
    assert (
        f"shots 16, shot_stream per-matrix ({N_FOLDS} lines); measurement shots, shots 32, "
        f"shot_stream per-matrix ({N_FOLDS} lines) (rows with more than one: 3 of 5)."
    ) in _refusal(shots + _measured(run, ("shots", 32, "per-matrix")), capsys)
    unrecorded = [
        {k: v for k, v in r.items() if k not in ("measurement", "shots", "feature_map")}
        for r in exact
    ]
    assert (
        f"shot_stream not recorded ({N_FOLDS} lines); measurement not recorded, shots not "
        f"recorded, shot_stream not recorded ({N_FOLDS} lines) (rows with more than one: 5 of 5)."
    ) in _refusal(exact + unrecorded, capsys)


def test_the_check_is_per_row_and_changes_no_output() -> None:
    """
    In one shots run arm C is exact and arms A and B are sampled, so a check over the whole
    group would refuse every shots run.  Per row it passes, and the run prints what the same
    lines print without the fields, as before #763; so does a run read twice.
    """
    run = _measured(_records(), PER_MATRIX)
    assert {summary.feature_measurement(r) for r in run} == {PER_MATRIX, EXACT}
    before = summary.summarise(_records(), permutations=99, long=True)
    assert summary.summarise(run, permutations=99, long=True) == before
    assert summary.summarise(run + run, permutations=99, long=True) == before


@pytest.fixture(scope="module")
def shots_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """
    The lines, as read back, of a quick positive-control run with ``--shots 16`` and every
    arm (A, B, C, D and with D ``D-hybrid``): two folds, no tuning.
    """
    settings = dataclasses.replace(bench.QUICK, shots=16, n_trials=0, max_folds=2)
    out = tmp_path_factory.mktemp("shots") / "shots.jsonl"
    data = {"positive-control": bench.load_dataset("positive-control", settings)}
    bench.run(data, settings, out, cache_dir=None)
    lines: list[dict[str, Any]] = bench.read_records(out)
    return lines


@pytest.mark.slow
def test_one_shots_run_with_every_arm_summarises(
    shots_run: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A run of the runner with shots holds both kinds of line in one group and still
    summarises: each row of the scores table has both folds of its arm and head, its mean
    and sample spread those of their MCCs computed here.  The same run with ``shot_stream``
    removed from every config, as a run from before #762 writes it, keeps its fingerprint
    and split, and joined to the first is refused in the 6 rows of A, B and D, not in the 3
    of C and ``D-hybrid``.
    """
    measured = {(r["arm"], r["head"], summary.feature_measurement(r)) for r in shots_run}
    assert measured == {
        *((arm, head, PER_MATRIX) for arm in ("A", "B", "D") for head in ("linear", "mlp")),
        ("C", "linear", EXACT),
        ("C", "mlp", EXACT),
        ("D-hybrid", "hybrid", EXACT),
    }
    text = summary.summarise(shots_run, permutations=99)
    for arm, head, _ in measured:
        mcc = np.array([r["mcc"] for r in shots_run if (r["arm"], r["head"]) == (arm, head)])
        assert mcc.size == 2
        (row,) = [line for line in text.splitlines() if line.startswith(f"| {arm} | {head} |")]
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert cells[4:6] == ["2", f"{mcc.mean():.3f} ± {mcc.std(ddof=1):.3f}"]
    old = [
        {**r, "feature_map": {k: v for k, v in r["feature_map"].items() if k != "shot_stream"}}
        if r["feature_map"] is not None
        else r
        for r in shots_run
    ]
    assert summary.data_fingerprints(shots_run + old) == {
        shots_run[0]["data_sha256"]: 2 * len(shots_run)
    }
    assert (
        "in row A/linear chain V/Ω=1 γ/Ω=0.1: measurement shots, shots 16, shot_stream "
        "per-matrix (2 lines); measurement shots, shots 16, shot_stream not recorded (2 lines) "
        "(rows with more than one: 6 of 9)."
    ) in _refusal(shots_run + old, capsys)


# ---------------------------------------------------------------------------
# PCA degeneracy (#747)
# ---------------------------------------------------------------------------


def _pca_records() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    ``(degenerate, regular)``: the runner's own ``pca`` record (``fit_inputs``) of two fits.

    The degenerate fit is on columns 1 to 6 of the ``8 × 8`` Sylvester-Hadamard matrix: each
    sums to 0 and has the same norm, and they are pairwise orthogonal, so after the runner's
    standardisation the covariance is the identity, every eigenvalue is 1 and every relative
    gap is 0.  The regular one is fold 0's training part of the default positive control,
    which #748 found far from degenerate.
    """
    h = np.array([[1.0]])
    for _ in range(3):
        h = np.block([[h, h], [h, -h]])
    with pytest.warns(RuntimeWarning, match="degenerate"):
        _, degenerate = bench.fit_inputs(h[:, 1:7], np.arange(8), 4)
    settings = bench.Settings()
    X, y, _ = bench.load_dataset("positive-control", settings)
    split_seed, _ = bench.fold_seeds(settings.random_state, "positive-control", 13)
    train_part, _ = bench.stratified_kfold(y, 13, random_state=split_seed)[0]
    _, regular = bench.fit_inputs(X, train_part, settings.n_atoms)
    return degenerate, regular


def test_the_pca_records_are_one_degenerate_and_one_regular_fit() -> None:
    degenerate, regular = _pca_records()
    assert degenerate["degenerate"] is True
    assert degenerate["relative_gaps"] == pytest.approx([0.0] * 4, abs=1e-12)
    assert regular["degenerate"] is False
    assert regular["min_relative_gap"] > 1e-6


def _with_pca(
    degenerate_fold: int | None, degenerate_tuning: int | None, dataset: str = "toy"
) -> list[dict[str, Any]]:
    """
    :func:`_records` (13 folds, 6 lines per fold) with ``pca`` and ``tuning["pca"]`` as the
    runner writes them: every line of a fold carries the fold's fit and its two tuning folds'
    fits.  Fold ``degenerate_fold`` has a degenerate fit of its own, fold
    ``degenerate_tuning`` a degenerate second tuning fit; every other fit is regular.
    """
    degenerate, regular = _pca_records()
    lines = []
    for r in _records():
        fold = r["fold"]
        tuning = [regular, degenerate if fold == degenerate_tuning else regular]
        lines.append(
            {
                **r,
                "dataset": dataset,
                "pca": degenerate if fold == degenerate_fold else regular,
                "tuning": {"pca": tuning},
            }
        )
    return lines


def test_one_degenerate_fold_is_counted_once_not_once_per_line() -> None:
    """
    Fold 3's own fit and fold 7's second tuning fit are degenerate.  Each is on all six lines
    of its fold, so a count of lines would give 6; of the 13 folds, one each is affected.
    The line sits right under the scores table.
    """
    lines = _with_pca(degenerate_fold=3, degenerate_tuning=7)
    assert summary.degenerate_folds(lines) == {"folds": 13, "fold": (1, 13), "tuning": (1, 13)}
    text = summary.summarise(lines, permutations=99)
    expected = (
        "Degenerate PCA fits (#525): fold fit in 1 of 13 folds, a tuning fit in 1 of 13 folds."
    )
    assert text.count(expected) == 1
    assert f"|\n\n{expected}\n\n### A against B" in text
    # The tables are those of the same lines without the field.
    assert text.replace(f"{expected}\n\n", "") == _without_degeneracy(
        summary.summarise(_records(), permutations=99), 1
    )


@pytest.mark.parametrize("position", range(6))
def test_a_fold_is_degenerate_if_any_of_its_lines_says_so(position: int) -> None:
    """
    Fold 5's own fit, and its second tuning fit, flagged on one of its six lines alone, first,
    last or in between, still make one degenerate fold each: neither the first nor the last
    line of a fold decides for it.
    """
    degenerate, regular = _pca_records()
    lines = _with_pca(degenerate_fold=None, degenerate_tuning=None)
    fold_5 = [r for r in lines if r["fold"] == 5]
    assert len(fold_5) == 6
    fold_5[position]["pca"] = degenerate
    fold_5[position]["tuning"] = {"pca": [regular, degenerate]}
    assert summary.degenerate_folds(lines) == {"folds": 13, "fold": (1, 13), "tuning": (1, 13)}


def test_no_degenerate_fold_counts_zero() -> None:
    lines = _with_pca(degenerate_fold=None, degenerate_tuning=None)
    assert summary.degenerate_folds(lines) == {"folds": 13, "fold": (0, 13), "tuning": (0, 13)}
    assert (
        "Degenerate PCA fits (#525): fold fit in 0 of 13 folds, a tuning fit in 0 of 13 folds."
        in summary.summarise(lines, permutations=99)
    )


def test_lines_before_525_are_not_recorded_and_partial_records_say_so() -> None:
    """
    Lines without ``pca`` (written before #525) are not counted, and a fold none of whose lines
    records it is named as not recorded.  A run without tuning (``tuning`` null) records no
    tuning fit.
    """
    old = _records()
    assert summary.degenerate_folds(old) == {"folds": 13, "fold": None, "tuning": None}
    assert NOT_RECORDED in summary.summarise(old, permutations=99)
    # Folds 0 to 9 record their own fit (fold 3 degenerate) and ran no tuning; 10 to 12
    # predate the field.
    mixed: list[dict[str, Any]] = []
    for r in _with_pca(degenerate_fold=3, degenerate_tuning=None):
        if r["fold"] < 10:
            mixed.append({**r, "tuning": None})
        else:
            mixed.append({k: v for k, v in r.items() if k not in ("pca", "tuning")})
    assert summary.degenerate_folds(mixed) == {"folds": 13, "fold": (1, 10), "tuning": None}
    assert (
        "Degenerate PCA fits (#525): fold fit in 1 of 10 folds (3 folds not recorded), "
        "a tuning fit not recorded." in summary.summarise(mixed, permutations=99)
    )
    # Only fold 12 predates the field: one fold, in the singular.
    one_old = [
        r if r["fold"] < 12 else {k: v for k, v in r.items() if k not in ("pca", "tuning")}
        for r in _with_pca(degenerate_fold=None, degenerate_tuning=None)
    ]
    assert summary.degeneracy_line(one_old) == (
        "Degenerate PCA fits (#525): fold fit in 0 of 12 folds (1 fold not recorded), "
        "a tuning fit in 0 of 12 folds (1 fold not recorded)."
    )


def test_each_group_counts_its_own_folds() -> None:
    """Two datasets, only the second with degenerate fits: each section has its own line."""
    toy = _with_pca(degenerate_fold=None, degenerate_tuning=None)
    other = _with_pca(degenerate_fold=0, degenerate_tuning=12, dataset="other")
    first, second = summary.summarise(toy + other, permutations=99).split("## other")
    assert "fold fit in 0 of 13 folds, a tuning fit in 0 of 13 folds." in first
    assert "fold fit in 1 of 13 folds, a tuning fit in 1 of 13 folds." in second


# ---------------------------------------------------------------------------
# One row, one device model (#775)
# ---------------------------------------------------------------------------

#: The tail of every refusal of :func:`summary._check_one_realism`.
REALISM_ADVICE = (
    "Its spread, intervals and paired tests would pool lines whose features come from "
    "different models; summarise each run's lines on their own."
)
#: The encoding config of a square pulse and of a global drive, as ``get_config()`` writes them.
SQUARE = {"n_features": 4, "omega": 4 * math.pi, "t": 0.25, "delta_max": 21.77}
RAMP = {
    "n_features": 4,
    "omega": 4 * math.pi,
    "t": 0.25,
    "delta_local": 21.77,
    "levels": 8,
    "t_ramp": 0.0503,
    "n_ramp_segments": 32,
}


def _modelled(lines: list[dict[str, Any]], **config: Any) -> list[dict[str, Any]]:
    """
    ``lines`` whose arms A and B carry a map config of the ideal model with ``config``; arm C
    has a ``feature_map`` of ``null``, as the runner writes it.
    """
    ideal = {
        "readout_error": None,
        "position_jitter": 0.0,
        "jitter_samples": None,
        "jitter_seed": None,
        "encoding": SQUARE,
    }
    out = []
    for r in lines:
        feature_map = (
            None
            if r["arm"] == "C"
            else {"feature_map": "RydbergFeatureMap", "config": {**ideal, **config}}
        )
        out.append({**r, "feature_map": feature_map})
    return out


IDEAL_TEXT = "readout_error none, position_jitter 0, jitter_samples none, encoding square"


def test_a_detection_error_beside_an_ideal_run_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    An ideal run and one with ``--readout-error`` of the same data, label and folds pass every
    guard of #742, #753 and #763; each of the 3 rows of A and B would pool both.  Refused at
    the first row of the scores table, naming each value with its number of lines.
    """
    ideal = _modelled(_records())
    noisy = _modelled(_records(), readout_error=[0.01, 0.08])
    assert _refusal(ideal + noisy, capsys) == (
        "dataset 'toy' has lines of 2 device models (readout_error, position_jitter, "
        f"jitter_samples, encoding) in row A/linear chain V/Ω=1 γ/Ω=0: {IDEAL_TEXT} "
        f"({N_FOLDS} lines); readout_error [0.01, 0.08], position_jitter 0, jitter_samples "
        f"none, encoding square ({N_FOLDS} lines) (rows with more than one: 3 of 5).  "
        + REALISM_ADVICE
    )


def test_jitter_beside_an_ideal_run_is_refused_with_the_label_bonds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run, _ = _control_run(geometries=("chain",))
    jittered = _modelled(run, position_jitter=0.2, jitter_samples=8, jitter_seed=3)
    assert _refusal(_modelled(run) + jittered, capsys) == (
        "dataset 'positive-control' (label_bonds [[0, 1], [1, 2], [2, 3]]) has lines of 2 "
        "device models (readout_error, position_jitter, jitter_samples, encoding) in row "
        f"A/linear chain V/Ω=1 γ/Ω=0: {IDEAL_TEXT} ({N_FOLDS} lines); readout_error none, "
        f"position_jitter 0.2, jitter_samples 8, encoding square ({N_FOLDS} lines) (rows with "
        "more than one: 3 of 5).  " + REALISM_ADVICE
    )
    # Another K is another model; another seed of the same model is not refused.
    assert "position_jitter 0.2, jitter_samples 4, encoding square" in _refusal(
        jittered + _modelled(run, position_jitter=0.2, jitter_samples=4, jitter_seed=3), capsys
    )
    reseeded = _modelled(run, position_jitter=0.2, jitter_samples=8, jitter_seed=4)
    assert summary.summarise(jittered + reseeded, permutations=99) == summary.summarise(
        jittered, permutations=99
    )


def test_another_encoding_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    ramped = _modelled(_records(), encoding=RAMP)
    assert (
        f"{IDEAL_TEXT} ({N_FOLDS} lines); readout_error none, position_jitter 0, "
        "jitter_samples none, encoding global-drive levels 8 t_ramp 0.0503 n_ramp_segments 32 "
        f"({N_FOLDS} lines) (rows with more than one: 3 of 5)."
    ) in _refusal(_modelled(_records()) + ramped, capsys)
    coarse = _modelled(_records(), encoding={**RAMP, "levels": 4})
    assert "encoding global-drive levels 4 t_ramp 0.0503 n_ramp_segments 32" in _refusal(
        ramped + coarse, capsys
    )


def test_lines_from_before_the_options_are_the_ideal_model_and_print_as_before() -> None:
    """
    A line written before #730, #731 and #732 has none of the keys: it was computed without
    the options, so it joins an ideal line of today, and the output is that of the lines
    without any config (the committed results file, read twice, checks the same on real
    lines in :func:`test_the_committed_results_summarise_as_before`).
    """
    old = [
        {**r, "feature_map": {"config": {"encoding": SQUARE}}} if r["arm"] != "C" else r
        for r in _modelled(_records())
    ]
    before = summary.summarise(_records(), permutations=99, long=True)
    assert summary.summarise(old + _modelled(_records()), permutations=99, long=True) == before
    assert {summary.feature_realism(r) for r in old} == {
        (None, 0.0, None, "square"),
        (None, 0.0, None, None),
    }


def test_the_runner_and_the_summariser_read_one_model_alike() -> None:
    """The two readings of a line agree on lines of a preset run's every representation."""
    settings = dataclasses.replace(
        bench.QUICK,
        **bench.device_preset("aquila", 4 * math.pi),
        device="aquila",
        levels="continuous",
        jitter_samples=2,
    )
    for rep in bench.representations(settings):
        line = {"feature_map": rep.config}
        assert summary.feature_realism(line) == bench.feature_realism(line)
    (a,) = [r for r in bench.representations(settings) if r.arm == "A"]
    assert summary.feature_realism({"feature_map": a.config}) == (
        (0.01, 0.08),
        0.2,
        2,
        ("global-drive", None, 0.0503, bench.AQUILA_RAMP_SEGMENTS),
    )
