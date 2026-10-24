"""
examples/summarise_rydberg_benchmark.py
=======================================
The tables of ``docs/results/rydberg-features-benchmark.md``, computed from
the JSON lines of ``examples/benchmark_rydberg_features.py``.

Every number in that note is printed by this script from the committed
results file; nothing is typed in by hand.  For each dataset it prints, as
Markdown:

* **Scores**: test MCC and PR-AUC per arm, head and grid point over the outer
  folds, as mean, sample standard deviation and a two-sided 95 % Student-t
  interval over the folds, ``mean ± t(0.975, n − 1) · sd / √n``.  Beside them
  the feature dimension, the trainable parameters and the mean seconds of
  feature simulation and of the final fit over the lines that computed them.
* **A against B** paired by fold, per head and grid point, and **every arm
  against each classical baseline** (arm C with the linear and with the MLP
  head), paired by fold: mean MCC difference, the exact two-sided Wilcoxon
  signed-rank p-value (:func:`hqnn_forge.evaluation.wilcoxon_signed_rank`),
  the rank-biserial effect and the Holm-adjusted p-value
  (:func:`hqnn_forge.evaluation.holm_correction`).  **All comparisons of one
  dataset form one Holm family**; an undefined test (every pair tied)
  counts as ``p = 1`` in it, as ``docs/methodology.md`` prescribes.
* **Mechanism**: the separation measures of arms A and B on the real
  training rows, averaged over folds per grid point, and for each measure
  the Spearman rank correlation across the grid points between its A − B
  difference and the A − B difference of mean MCC, with the number of grid
  points and a two-sided permutation p-value (``--permutations`` draws,
  seeded).

Usage::

    python examples/summarise_rydberg_benchmark.py \\
        docs/results/rydberg_features_benchmark.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hqnn_forge.evaluation import holm_correction, rank_biserial_correlation, wilcoxon_signed_rank

FloatArray = npt.NDArray[np.float64]

#: Family-wise error rate of the Holm family of one dataset.
ALPHA = 0.05
#: Separation measures of the mechanism table, in the order printed.
MEASURES: tuple[str, ...] = (
    "distance_ratio",
    "fisher_ratio",
    "effective_rank",
    "kernel_alignment",
)
#: The classical baselines: arm C with each head.
BASELINES: tuple[tuple[str, str], ...] = (("C", "linear"), ("C", "mlp"))

_RUNNER = Path(__file__).with_name("benchmark_rydberg_features.py")


def _runner() -> Any:
    spec = importlib.util.spec_from_file_location("benchmark_rydberg_features", _RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_records(path: Path) -> list[dict[str, Any]]:
    """The records of ``path``, read with the runner's own :func:`read_records`."""
    records: list[dict[str, Any]] = _runner().read_records(path)
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _t_cdf(x: float, df: int) -> float:
    """Student-t CDF at ``x >= 0`` by Simpson's rule on its density (2000 panels)."""
    norm = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))

    def pdf(t: float) -> float:
        return norm * (1.0 + t * t / df) ** (-(df + 1) / 2)

    panels = 2000
    h = x / panels
    total = pdf(0.0) + pdf(x)
    total += sum((4 if k % 2 else 2) * pdf(k * h) for k in range(1, panels))
    return 0.5 + total * h / 3.0


def t_quantile(p: float, df: int) -> float:
    """The ``p`` quantile (``p > 0.5``) of Student's t with ``df`` degrees of freedom."""
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if _t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2.0


def interval(values: Sequence[float]) -> tuple[float, float, float, float]:
    """``(mean, sd, low, high)``: the two-sided 95 % t-interval of the mean."""
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    if a.size < 2:
        return mean, math.nan, math.nan, math.nan
    sd = float(a.std(ddof=1))
    half = t_quantile(0.975, a.size - 1) * sd / math.sqrt(a.size)
    return mean, sd, mean - half, mean + half


def _ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    for value in np.unique(values):  # average ranks over ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(
    x: Sequence[float], y: Sequence[float], *, permutations: int, seed: int
) -> tuple[float, int, float]:
    """``(rho, n, p)``: Spearman's rho over the finite pairs and its permutation p-value."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = int(a.size)
    if n < 3:
        return math.nan, n, math.nan
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return math.nan, n, math.nan
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(permutations)])
    # The observed statistic counts as one draw of the null (Phipson & Smyth 2010).
    p = (1 + int(np.sum(np.abs(null) >= abs(rho) - 1e-12))) / (permutations + 1)
    return rho, n, p


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

Key = tuple[str, str, Any, Any]  # arm, head, V/Ω, γ/Ω


def _rows(records: Iterable[Mapping[str, Any]]) -> dict[Key, dict[int, Mapping[str, Any]]]:
    """``{(arm, head, V/Ω, γ/Ω): {fold: record}}``; arm B is keyed by γ/Ω alone."""
    rows: dict[Key, dict[int, Mapping[str, Any]]] = {}
    for r in records:
        v = None if r["arm"] == "B" else r["v_over_omega"]
        key = (r["arm"], r["head"], v, r["gamma_over_omega"])
        folds = rows.setdefault(key, {})
        # Arm B has one line per V/Ω, all with the same scores (reused); keep the computed one.
        if r["fold"] in folds and r.get("reused_from_v_over_omega") is not None:
            continue
        folds[r["fold"]] = r
    return rows


def _label(key: Key) -> str:
    arm, head, v, gamma = key
    grid = (
        "" if gamma is None else (f" γ/Ω={gamma:g}" if v is None else f" V/Ω={v:g} γ/Ω={gamma:g}")
    )
    return f"{arm}/{head}{grid}"


def _sort(key: Key) -> tuple[Any, ...]:
    arm, head, v, gamma = key
    return (arm, head, -1.0 if gamma is None else gamma, -1.0 if v is None else v)


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "–"
    return f"{value:.{digits}f}"


def _p(value: float) -> str:
    if not math.isfinite(value):
        return "–"
    return f"{value:.2e}" if value < 1e-3 else f"{value:.4f}"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def scores_table(rows: Mapping[Key, Mapping[int, Mapping[str, Any]]]) -> str:
    lines = [
        (
            "| arm | head | V/Ω | γ/Ω | n | MCC mean ± sd | MCC 95 % CI | PR-AUC mean ± sd "
            "| PR-AUC 95 % CI | dim | params | feature s | fit s |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(rows, key=_sort):
        arm, head, v, gamma = key
        folds = rows[key]
        recs = [folds[f] for f in sorted(folds)]
        mcc = interval([r["mcc"] for r in recs])
        ap = interval([r["pr_auc"] for r in recs])
        dims = sorted({r["feature_dim"] for r in recs})
        params = sorted({r["n_parameters"] for r in recs})
        feat = np.mean([r["feature_seconds"] for r in recs])
        fit = np.mean([r["train_seconds"] for r in recs])
        lines.append(
            f"| {arm} | {head} | {'–' if v is None else f'{v:g}'} | "
            f"{'–' if gamma is None else f'{gamma:g}'} | {len(recs)} | "
            f"{_fmt(mcc[0])} ± {_fmt(mcc[1])} | [{_fmt(mcc[2])}, {_fmt(mcc[3])}] | "
            f"{_fmt(ap[0])} ± {_fmt(ap[1])} | [{_fmt(ap[2])}, {_fmt(ap[3])}] | "
            f"{'/'.join(map(str, dims))} | {'/'.join(map(str, params))} | "
            f"{feat:.2f} | {fit:.2f} |"
        )
    return "\n".join(lines)


def _paired(
    a: Mapping[int, Mapping[str, Any]], b: Mapping[int, Mapping[str, Any]]
) -> tuple[FloatArray, FloatArray]:
    folds = sorted(set(a) & set(b))
    return (
        np.array([a[f]["mcc"] for f in folds], dtype=np.float64),
        np.array([b[f]["mcc"] for f in folds], dtype=np.float64),
    )


def comparisons(
    rows: Mapping[Key, Mapping[int, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Every A − B and arm − baseline comparison of one dataset, Holm-adjusted as one family."""
    out: list[dict[str, Any]] = []
    for key in sorted(rows, key=_sort):
        arm, head, _, gamma = key
        if arm == "A":
            b_key: Key = ("B", head, None, gamma)
            if b_key in rows:
                out.append({"kind": "A − B", "a": key, "b": b_key})
        if arm != "C":
            for base in BASELINES:
                base_key: Key = (*base, None, None)
                if base_key in rows:
                    out.append({"kind": "arm − baseline", "a": key, "b": base_key})
    for c in out:
        a, b = _paired(rows[c["a"]], rows[c["b"]])
        c["n_folds"] = int(a.size)
        c["mean_diff"] = float(np.mean(a - b))
        try:
            res = wilcoxon_signed_rank(a, b)
        except ValueError:  # every pair tied: undefined
            c.update(p=math.nan, n_pairs=0, min_p=math.nan, effect=0.0)
        else:
            c.update(
                p=res.p_value,
                n_pairs=res.n,
                min_p=res.min_p_value,
                effect=rank_biserial_correlation(a, b),
            )
    if out:
        raw = np.array([1.0 if math.isnan(c["p"]) else c["p"] for c in out])
        for c, adjusted in zip(out, holm_correction(raw), strict=True):
            c["p_holm"] = float(adjusted)
    return out


def comparison_table(family: Sequence[Mapping[str, Any]], kind: str) -> str:
    lines = [
        "| comparison | against | n | mean ΔMCC | W pairs | p (exact) | p floor | p Holm | r_rb |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in family:
        if c["kind"] != kind:
            continue
        mark = " **✓**" if c["p_holm"] <= ALPHA else ""
        lines.append(
            f"| {_label(c['a'])} | {_label(c['b'])} | {c['n_folds']} | "
            f"{c['mean_diff']:+.3f} | {c['n_pairs']} | {_p(c['p'])} | {_p(c['min_p'])} | "
            f"{_p(c['p_holm'])}{mark} | {c['effect']:+.2f} |"
        )
    return "\n".join(lines)


def _mean_measure(folds: Mapping[int, Mapping[str, Any]], measure: str, part: str) -> float:
    values = [float(r["separation"][part][measure]) for r in folds.values()]
    finite = [x for x in values if math.isfinite(x)]
    return float(np.mean(finite)) if len(finite) == len(values) else math.nan


def mechanism_tables(
    rows: Mapping[Key, Mapping[int, Mapping[str, Any]]],
    *,
    part: str,
    permutations: int,
    seed: int,
    heads: Sequence[str],
) -> str:
    """Separation measures of A and B per grid point and their rank correlation with ΔMCC."""
    out: list[str] = []
    cells = sorted({(k[2], k[3]) for k in rows if k[0] == "A"}, key=lambda c: (c[1], c[0]))
    # The measures are properties of the features, not of the head: read them from one head.
    head0 = heads[0]
    out.append("| V/Ω | γ/Ω | " + " | ".join(f"{m} A | {m} B" for m in MEASURES) + " |")
    out.append("|---|---|" + "---|---|" * len(MEASURES))
    for v, gamma in cells:
        a = rows[("A", head0, v, gamma)]
        b = rows.get(("B", head0, None, gamma), {})
        vals = []
        for m in MEASURES:
            vals.append(_fmt(_mean_measure(a, m, part)))
            vals.append(_fmt(_mean_measure(b, m, part)) if b else "–")
        out.append(f"| {v:g} | {gamma:g} | " + " | ".join(vals) + " |")
    out.append("")
    out.append("| head | measure | Spearman ρ (Δmeasure, ΔMCC) | n grid points | permutation p |")
    out.append("|---|---|---|---|---|")
    for head in heads:
        d_mcc: list[float] = []
        d_measure: dict[str, list[float]] = {m: [] for m in MEASURES}
        for v, gamma in cells:
            a_rows = rows.get(("A", head, v, gamma))
            b_rows = rows.get(("B", head, None, gamma))
            if not a_rows or not b_rows:
                continue
            ma, mb = _paired(a_rows, b_rows)
            d_mcc.append(float(np.mean(ma - mb)))
            for m in MEASURES:
                d_measure[m].append(
                    _mean_measure(a_rows, m, part) - _mean_measure(b_rows, m, part)
                )
        for m in MEASURES:
            rho, n, p = spearman(d_measure[m], d_mcc, permutations=permutations, seed=seed)
            out.append(f"| {head} | {m} | {_fmt(rho, 2)} | {n} | {_p(p)} |")
    return "\n".join(out)


def summarise(
    records: Sequence[Mapping[str, Any]], *, permutations: int = 10_000, seed: int = 0
) -> str:
    """The Markdown tables of every dataset in ``records``."""
    by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        by_dataset.setdefault(r["dataset"], []).append(r)
    sections: list[str] = []
    for dataset, recs in by_dataset.items():
        rows = _rows(recs)
        family = comparisons(rows)
        n_folds = {len(f) for f in rows.values()}
        heads = sorted({k[1] for k in rows if k[0] == "A"})
        m = len(family)
        sections += [
            f"## {dataset}",
            "",
            (
                f"Folds per row: {sorted(n_folds)}.  Holm family: {m} comparisons at "
                f"α = {ALPHA}, first threshold α/m = {_p(ALPHA / m) if m else '–'}."
            ),
            "",
            "### Scores",
            "",
            scores_table(rows),
            "",
            "### A against B (MCC, paired by fold)",
            "",
            comparison_table(family, "A − B"),
            "",
            "### Every arm against the classical baselines (MCC, paired by fold)",
            "",
            comparison_table(family, "arm − baseline"),
            "",
            "### Separation measures (real training rows, mean over folds)",
            "",
            mechanism_tables(rows, part="train", permutations=permutations, seed=seed, heads=heads)
            if heads
            else "(no arm A lines)",
            "",
        ]
    return "\n".join(sections)


def main(
    argv: Sequence[str] | None = None,
    *,
    read: Callable[[Path], list[dict[str, Any]]] = read_records,
) -> str:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[1])
    parser.add_argument("results", type=Path, help="JSON lines of benchmark_rydberg_features.py")
    parser.add_argument(
        "--permutations", type=int, default=10_000, help="draws of the permutation p-values"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the permutation draws")
    args = parser.parse_args(argv)
    text = summarise(read(args.results), permutations=args.permutations, seed=args.seed)
    print(text)
    return text


if __name__ == "__main__":
    main()
