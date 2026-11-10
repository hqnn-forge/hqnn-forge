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


def test_the_committed_results_summarise_as_before() -> None:
    """
    The summary of the committed results, ``--long`` with 99 permutations, is byte for byte
    that of ``main`` at 29c8931, before the fingerprint guard (#742): the SHA-256 below is that
    of the script's output there.
    """
    records = summary.read_records(ROOT / "docs/results/rydberg_features_benchmark.jsonl.xz")
    assert {r["dataset"]: r["data_sha256"] for r in records}["positive-control"] == (
        CHAIN_CONTROL_SHA
    )
    text = summary.summarise(records, permutations=99, long=True)
    assert (
        hashlib.sha256(text.encode()).hexdigest()
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
    the stated geometries and 13 folds, and summarises without error to a fixed output.
    """
    records = summary.read_records(ROOT / "docs/results" / name)
    found = {(r["dataset"], r["data_sha256"], repr(r.get("label_bonds"))) for r in records}
    assert found == {(d, sha, repr(bonds)) for d, (sha, bonds) in datasets.items()}
    assert {bench.geometry_label(r) for r in records if r["arm"] == "A"} == geometries
    assert {r["n_splits"] for r in records} == {13}
    text = summary.summarise(records, permutations=99, long=True)
    assert hashlib.sha256(text.encode()).hexdigest() == digest


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
    text = summary.summarise(records, permutations=99, long=True)
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
