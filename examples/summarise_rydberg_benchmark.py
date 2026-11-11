"""
examples/summarise_rydberg_benchmark.py
=======================================
The tables of ``docs/results/rydberg-features-benchmark.md``, computed from
the JSON lines of ``examples/benchmark_rydberg_features.py``.

Every number in that note is printed by this script from the committed
results file; nothing is typed in by hand.  For each dataset, and within
it each label (*Groups* below), it prints, as Markdown:

* **Scores**: test MCC and PR-AUC per arm, head and grid point over the outer
  folds, as mean, sample standard deviation and a two-sided 95 % Student-t
  interval over the folds, ``mean ± t(0.975, n − 1) · sd / √n``.  Beside them
  the feature dimension, the trainable parameters and the mean seconds of
  feature simulation and of the final fit over the lines that computed them.
  Under the table, one line (:func:`degeneracy_line`, #747) says in how many
  folds the PCA fit, and in how many a tuning fold's fit, was degenerate
  (the runner's ``pca`` and ``tuning["pca"]``, #525): in such a fold the
  assignment of components to atoms, and with it ``A − B``, is arbitrary.
  Lines written before #525 do not record it, and the line says ``not
  recorded``.
* **A against B** paired by fold, per head and grid point, and **every arm
  against each classical baseline** (arm C with the linear and with the MLP
  head), paired by fold, as ``V/Ω`` × ``γ/Ω`` grids of the mean MCC
  difference and its Holm-adjusted p-value
  (:func:`hqnn_forge.evaluation.holm_correction` over the exact two-sided
  Wilcoxon signed-rank p-values of
  :func:`hqnn_forge.evaluation.wilcoxon_signed_rank`).  ``--long`` adds one
  row per comparison with the pairs left after ties, the exact p-value, its
  floor and the rank-biserial effect.  **All comparisons of one group (one
  dataset, one label) form one Holm family**; an undefined test (every pair
  tied) counts as ``p = 1`` in it, as ``docs/methodology.md`` prescribes.
* **Mechanism**: the separation measures of arms A and B on the real rows
  of the training part (as the runner records them under ``"train"``),
  averaged over folds per grid point, and for each measure
  the Spearman rank correlation across the grid points between its A − B
  difference and the A − B difference of mean MCC, with the number of grid
  points and a two-sided permutation p-value (``--permutations`` draws,
  seeded).

**Geometries** (``--geometry`` of the runner, #733).  Arm A is grouped by
geometry as well (``chain``, ``ring``, ``grid-RxC``); a line written before
the option existed is a chain line.  Arms B, C and D do not depend on the
geometry and appear once per group: B's features do not depend on where its
atoms sit, so every A line of a group, whatever its geometry, is paired with
the same B line.  A file of a single geometry prints exactly the tables it
printed before the option existed.  With several in one group (a UCI
dataset, whose label does not depend on the geometry), the scores table
gains a ``geometry`` column, every grid and mechanism table is printed once
per geometry, and the comparisons of all its geometries still form **one**
Holm family.

**Groups: one dataset, one label** (#741).  The positive control's label
follows the geometry (#738), so one dataset can hold lines of several
labels on the same inputs, each with its own folds (stratified on the
label) and its own arms B, C and D.  Lines are therefore grouped by
``(dataset, label_bonds)``, in the order first seen (:func:`groups`), and
each group gets its own section and its own Holm family; the comparisons
of two labels are never pooled.  A line without ``label_bonds`` (the field
missing or ``null``) is keyed ``(dataset, None)``: a UCI dataset, whose
lines record ``null``, or a positive control written before #738, whose
label is the chain's.  Such a control is *not* keyed as the chain's bonds,
so beside lines that record them (every control line since #738, the chain
as ``[[0, 1], [1, 2], …]``) it is a group of its own, even on the same
data.  A dataset with one group keeps the heading ``## <dataset>`` it had
before; with several, each heading adds ``, label_bonds [[j, k], …]`` (or
``, label_bonds not recorded``).

**One group, one data** (#742).  Every line of a group must carry the same
``data_sha256``: two positive-control runs of one geometry but different
``--seed`` or ``--samples`` have the same ``label_bonds``, and read
together they would otherwise be paired by fold index across data in one
Holm family.  The script refuses such input with a ``ValueError`` naming
the dataset, its label bonds and each fingerprint with its number of lines
(a line without the field counts as one more, ``not recorded``) before
printing anything; summarise each run on its own.  Lines of different
labels carry different fingerprints too, but are in different groups.

Usage::

    python examples/summarise_rydberg_benchmark.py \\
        docs/results/rydberg_features_benchmark.jsonl.xz

The committed file is the runner's JSONL, xz-compressed; the runner's
``read_records``, which this script reads through, opens a ``.xz`` path with
:mod:`lzma` and any other path as plain text.
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

#: Family-wise error rate of the Holm family of one group (dataset and label).
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

Key = tuple[str, str, Any, Any, Any]  # arm, head, geometry, V/Ω, γ/Ω


def data_fingerprints(records: Iterable[Mapping[str, Any]]) -> dict[str | None, int]:
    """
    ``{data_sha256: number of lines}`` over ``records``, in the order first seen.

    ``None`` stands for lines without the field, whose data is not recorded.
    """
    counts: dict[str | None, int] = {}
    for r in records:
        sha = r.get("data_sha256")
        counts[sha] = counts.get(sha, 0) + 1
    return counts


Bonds = tuple[tuple[int, int], ...]
GroupKey = tuple[str, Bonds | None]  # dataset, label_bonds


def label_bonds(record: Mapping[str, Any]) -> Bonds | None:
    """
    The ``label_bonds`` of a line as a tuple of pairs, in the order recorded.

    ``None`` for a line without them, the field missing or ``null``: a UCI
    dataset, or a positive control written before #738 (whose label is the
    chain's, but which is not keyed as the chain's bonds; see the module
    docstring).
    """
    bonds = record.get("label_bonds")
    if bonds is None:
        return None
    return tuple((int(j), int(k)) for j, k in bonds)


def _bonds_text(bonds: Bonds | None) -> str:
    """``label_bonds`` as the runner writes them, ``[[0, 1], …]``, or ``not recorded``."""
    if bonds is None:
        return "not recorded"
    return str([list(pair) for pair in bonds])


def groups(records: Iterable[Mapping[str, Any]]) -> dict[GroupKey, list[Mapping[str, Any]]]:
    """
    ``{(dataset, label_bonds): lines}``, in the order first seen.

    One group per dataset and label (:func:`label_bonds`): the unit of the
    fingerprint check, of the tables and of the Holm family.
    """
    out: dict[GroupKey, list[Mapping[str, Any]]] = {}
    for r in records:
        out.setdefault((r["dataset"], label_bonds(r)), []).append(r)
    return out


def _check_one_fingerprint(
    dataset: str, records: Sequence[Mapping[str, Any]], bonds: Bonds | None = None
) -> None:
    """
    Refuse ``records`` of ``dataset`` unless they all carry the same ``data_sha256``.

    ``records`` are one group of :func:`groups`, ``bonds`` its
    ``label_bonds``.  The folds, arm B's row of each γ/Ω and the Holm family
    are keyed by the group, so lines of other data in it, e.g. two
    positive-control runs of one geometry with different seeds, would be
    paired by fold index across data.  Every line without the field counts
    as one more fingerprint, ``not recorded``, so it is not merged with
    lines that have one.
    """
    counts = data_fingerprints(records)
    if len(counts) > 1:
        found = ", ".join(
            f"{'not recorded' if sha is None else sha} ({n} lines)" for sha, n in counts.items()
        )
        label = "" if bonds is None else f" (label_bonds {_bonds_text(bonds)})"
        raise ValueError(
            f"dataset {dataset!r}{label} has lines of {len(counts)} data fingerprints "
            f"(data_sha256): {found}.  Its folds and Holm family would pair lines of "
            "different data; summarise each run's lines on their own."
        )


def geometry_label(record: Mapping[str, Any]) -> str:
    """
    The geometry of an arm A line as one word: ``chain``, ``ring`` or ``grid-RxC``.

    A line without the field was written before ``--geometry`` existed, on
    the chain.
    """
    geometry = str(record.get("geometry") or "chain")
    if geometry == "grid" and record.get("grid_shape") is not None:
        rows, cols = record["grid_shape"]
        return f"grid-{rows}x{cols}"
    return geometry


def _rows(records: Iterable[Mapping[str, Any]]) -> dict[Key, dict[int, Mapping[str, Any]]]:
    """
    ``{(arm, head, geometry, V/Ω, γ/Ω): {fold: record}}``.

    Arm B is keyed by γ/Ω alone, and only arm A by its geometry
    (:func:`geometry_label`); the other arms have ``None`` there.
    """
    rows: dict[Key, dict[int, Mapping[str, Any]]] = {}
    for r in records:
        v = None if r["arm"] == "B" else r["v_over_omega"]
        geometry = geometry_label(r) if r["arm"] == "A" else None
        key = (r["arm"], r["head"], geometry, v, r["gamma_over_omega"])
        folds = rows.setdefault(key, {})
        # Arm B has one line per geometry and V/Ω, all with the same scores (reused); keep the
        # computed one.
        if r["fold"] in folds and r.get("reused_from_v_over_omega") is not None:
            continue
        folds[r["fold"]] = r
    return rows


def _label(key: Key, *, geometry: bool = False) -> str:
    """``arm/head`` and the grid point; with ``geometry``, arm A's geometry first."""
    arm, head, shape, v, gamma = key
    grid = (
        "" if gamma is None else (f" γ/Ω={gamma:g}" if v is None else f" V/Ω={v:g} γ/Ω={gamma:g}")
    )
    where = f" {shape}" if geometry and shape is not None else ""
    return f"{arm}/{head}{where}{grid}"


#: The runner's order of the geometries, in which the tables print them.
GEOMETRY_ORDER: tuple[str, ...] = ("chain", "ring", "grid")


def _geometry_rank(shape: str) -> tuple[int, str]:
    """Sort key of a :func:`geometry_label`: the runner's order, a shape after its kind."""
    kind = shape.split("-")[0]
    rank = GEOMETRY_ORDER.index(kind) if kind in GEOMETRY_ORDER else len(GEOMETRY_ORDER)
    return rank, shape


def _geometries(rows: Iterable[Key]) -> list[str]:
    """The geometries of arm A in ``rows``, in the order of :func:`_geometry_rank`."""
    return sorted({k[2] for k in rows if k[0] == "A"}, key=_geometry_rank)


def _sort(key: Key) -> tuple[Any, ...]:
    arm, head, shape, v, gamma = key
    return (
        arm,
        head,
        (-1, "") if shape is None else _geometry_rank(shape),
        -1.0 if gamma is None else gamma,
        -1.0 if v is None else v,
    )


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


def scores_table(
    rows: Mapping[Key, Mapping[int, Mapping[str, Any]]], *, geometry: bool = False
) -> str:
    """One row per arm, head and grid point; with ``geometry``, a column for arm A's geometry."""
    where = ("geometry | ", "---|") if geometry else ("", "")
    lines = [
        (
            f"| arm | head | {where[0]}V/Ω | γ/Ω | n | MCC mean ± sd | MCC 95 % CI "
            "| PR-AUC mean ± sd | PR-AUC 95 % CI | dim | params | feature s | fit s |"
        ),
        f"|---|---|{where[1]}---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(rows, key=_sort):
        arm, head, shape, v, gamma = key
        folds = rows[key]
        recs = [folds[f] for f in sorted(folds)]
        mcc = interval([r["mcc"] for r in recs])
        ap = interval([r["pr_auc"] for r in recs])
        dims = sorted({r["feature_dim"] for r in recs})
        params = sorted({r["n_parameters"] for r in recs})
        feat = np.mean([r["feature_seconds"] for r in recs])
        fit = np.mean([r["train_seconds"] for r in recs])
        cell = f"{'–' if shape is None else shape} | " if geometry else ""
        lines.append(
            f"| {arm} | {head} | {cell}{'–' if v is None else f'{v:g}'} | "
            f"{'–' if gamma is None else f'{gamma:g}'} | {len(recs)} | "
            f"{_fmt(mcc[0])} ± {_fmt(mcc[1])} | [{_fmt(mcc[2])}, {_fmt(mcc[3])}] | "
            f"{_fmt(ap[0])} ± {_fmt(ap[1])} | [{_fmt(ap[2])}, {_fmt(ap[3])}] | "
            f"{'/'.join(map(str, dims))} | {'/'.join(map(str, params))} | "
            f"{feat:.2f} | {fit:.2f} |"
        )
    return "\n".join(lines)


def degenerate_folds(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """
    How many folds of one group had a degenerate PCA fit, as the runner records it (#525).

    ``records`` are one group of :func:`groups`.  Every line of a fold, of
    whatever arm, head or grid point, is trained and scored on the inputs of
    the same PCA fit, on the fold's training part, and its tuning folds on
    those of the same tuning fits, so the count is over folds (``fold``), not
    lines.  Returns ``{"folds": n, "fold": f, "tuning": t}``:

    * ``n``: the number of distinct folds in ``records``.
    * ``f``: ``(degenerate, recorded)`` for the fold's own fit, ``pca`` on
      the line: ``recorded`` is the number of folds with a line carrying it
      (not ``null``), ``degenerate`` of those the folds with a line whose
      ``pca["degenerate"]`` is true (there the assignment of components to
      atoms, and with it ``A − B``, is arbitrary).  ``None`` if no line
      records it: every line written before #525.
    * ``t``: the same for the tuning fits, ``tuning["pca"]`` (one record per
      tuning fold): a fold is degenerate if any of them is (its choice of
      hyperparameters may then depend on the platform).  A line without
      tuning (``tuning`` null), or written before #525, records none.

    A fold counts as degenerate if any of its lines says so; the runner
    writes the same record on all of them.
    """
    folds: set[int] = set()
    fold_fit: dict[int, bool] = {}
    tuning_fit: dict[int, bool] = {}
    for r in records:
        k = int(r["fold"])
        folds.add(k)
        if r.get("pca") is not None:
            fold_fit[k] = fold_fit.get(k, False) or bool(r["pca"]["degenerate"])
        tuning = r.get("tuning")
        if tuning and tuning.get("pca"):
            any_fit = any(bool(fit["degenerate"]) for fit in tuning["pca"])
            tuning_fit[k] = tuning_fit.get(k, False) or any_fit
    return {
        "folds": len(folds),
        "fold": (sum(fold_fit.values()), len(fold_fit)) if fold_fit else None,
        "tuning": (sum(tuning_fit.values()), len(tuning_fit)) if tuning_fit else None,
    }


def degeneracy_line(records: Iterable[Mapping[str, Any]]) -> str:
    """
    :func:`degenerate_folds` of one group as the sentence under its scores table.

    ``Degenerate PCA fits (#525): fold fit in d of r folds, a tuning fit in
    d of r folds.``, each part ``not recorded`` when no line records it, and
    with ``(m folds not recorded)`` after it when ``m`` of the group's folds
    have no line that does.
    """
    counts = degenerate_folds(records)

    def part(found: tuple[int, int] | None) -> str:
        if found is None:
            return "not recorded"
        degenerate, recorded = found
        text = f"in {degenerate} of {recorded} folds"
        missing = counts["folds"] - recorded
        return text + (f" ({missing} folds not recorded)" if missing else "")

    return (
        f"Degenerate PCA fits (#525): fold fit {part(counts['fold'])}, "
        f"a tuning fit {part(counts['tuning'])}."
    )


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
    """
    Every A − B and arm − baseline comparison of one group, Holm-adjusted as one family.

    ``rows`` are the :func:`_rows` of one group of :func:`groups`, one
    dataset and label.  The family spans every geometry of the group.  Arm A
    of each geometry is compared with the one B row of its γ/Ω, and B, C and
    D, which have no geometry, are compared with the baselines once.
    """
    out: list[dict[str, Any]] = []
    for key in sorted(rows, key=_sort):
        arm, head, _, _, gamma = key
        if arm == "A":
            b_key: Key = ("B", head, None, None, gamma)
            if b_key in rows:
                out.append({"kind": "A − B", "a": key, "b": b_key})
        if arm != "C":
            for base in BASELINES:
                base_key: Key = (*base, None, None, None)
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


def comparison_table(
    family: Sequence[Mapping[str, Any]], kind: str, *, geometry: bool = False
) -> str:
    lines = [
        "| comparison | against | n | mean ΔMCC | W pairs | p (exact) | p floor | p Holm | r_rb |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in family:
        if c["kind"] != kind:
            continue
        mark = " **✓**" if c["p_holm"] <= ALPHA else ""
        lines.append(
            f"| {_label(c['a'], geometry=geometry)} | {_label(c['b'])} | {c['n_folds']} | "
            f"{c['mean_diff']:+.3f} | {c['n_pairs']} | {_p(c['p'])} | {_p(c['min_p'])} | "
            f"{_p(c['p_holm'])}{mark} | {c['effect']:+.2f} |"
        )
    return "\n".join(lines)


def _cell(c: Mapping[str, Any]) -> str:
    mark = " ✓" if c["p_holm"] <= ALPHA else ""
    return f"{c['mean_diff']:+.3f} ({_p(c['p_holm'])}{mark})"


def grid_table(
    family: Sequence[Mapping[str, Any]],
    kind: str,
    head: str,
    against: Key,
    geometry: str | None = None,
) -> str:
    """
    Arm A's comparisons of one head as a ``V/Ω`` × ``γ/Ω`` grid: ΔMCC (Holm p).

    ``geometry`` (a :func:`geometry_label`) restricts it to arm A of that
    geometry; ``None`` takes every A row, as for a file of one geometry.
    """
    cells = {
        (c["a"][3], c["a"][4]): c
        for c in family
        if c["kind"] == kind
        and c["a"][0] == "A"
        and c["a"][1] == head
        and (geometry is None or c["a"][2] == geometry)
        and c["b"][:2] == against[:2]
    }
    vs = sorted({v for v, _ in cells})
    gammas = sorted({g for _, g in cells})
    lines = [
        "| V/Ω | " + " | ".join(f"γ/Ω = {g:g}" for g in gammas) + " |",
        "|---|" + "---|" * len(gammas),
    ]
    for v in vs:
        entries = [_cell(cells[(v, g)]) if (v, g) in cells else "–" for g in gammas]
        lines.append(f"| {v:g} | " + " | ".join(entries) + " |")
    return "\n".join(lines)


def other_arms_table(family: Sequence[Mapping[str, Any]]) -> str:
    """Every arm row except A against each baseline: ΔMCC (Holm p)."""
    rows: dict[Key, dict[Key, Mapping[str, Any]]] = {}
    for c in family:
        if c["kind"] == "arm − baseline" and c["a"][0] != "A":
            rows.setdefault(c["a"], {})[c["b"]] = c
    bases: list[Key] = [(*b, None, None, None) for b in BASELINES]
    lines = [
        "| arm | " + " | ".join(f"against {_label(b)}" for b in bases) + " |",
        "|---|" + "---|" * len(bases),
    ]
    for key in sorted(rows, key=_sort):
        entries = [_cell(rows[key][b]) if b in rows[key] else "–" for b in bases]
        lines.append(f"| {_label(key)} | " + " | ".join(entries) + " |")
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
    geometry: str | None = None,
) -> str:
    """
    Separation measures of A and B per grid point and their rank correlation with ΔMCC.

    ``geometry`` restricts arm A to that geometry (the rank correlation then
    runs over its grid points only); ``None`` takes every A row.
    """
    out: list[str] = []
    a_keys = {k for k in rows if k[0] == "A" and (geometry is None or k[2] == geometry)}
    shapes = {k[3:]: k[2] for k in a_keys}
    cells = sorted(shapes, key=lambda c: (c[1], c[0]))
    # The measures are properties of the features, not of the head: read them from one head.
    head0 = heads[0]
    out.append("| V/Ω | γ/Ω | " + " | ".join(f"{m} A | {m} B" for m in MEASURES) + " |")
    out.append("|---|---|" + "---|---|" * len(MEASURES))
    for v, gamma in cells:
        a = rows[("A", head0, shapes[(v, gamma)], v, gamma)]
        b = rows.get(("B", head0, None, None, gamma), {})
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
            a_rows = rows.get(("A", head, shapes[(v, gamma)], v, gamma))
            b_rows = rows.get(("B", head, None, None, gamma))
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
    records: Sequence[Mapping[str, Any]],
    *,
    permutations: int = 10_000,
    seed: int = 0,
    long: bool = False,
) -> str:
    """
    The Markdown tables of every dataset in ``records``; ``long`` adds one row per test.

    One section per group of :func:`groups`, ``(dataset, label_bonds)``, in
    the order first seen, each its own Holm family.  Its heading is the
    dataset's name, followed by ``, label_bonds [[j, k], …]`` (or
    ``, label_bonds not recorded``) when the dataset has more than one group.
    A group with arm A in several geometries gets a ``geometry`` column in
    its scores, and its grid and mechanism tables once per geometry; with one
    geometry the output is that of a file from before ``--geometry``.  Under
    each scores table, :func:`degeneracy_line` counts the group's folds with
    a degenerate PCA fit; apart from that line the output is that of a file
    from before #747.

    Raises
    ------
    ValueError
        If the lines of one group carry more than one ``data_sha256``
        (a line without it counting as one more), naming the dataset, its
        label bonds and the fingerprints; nothing is printed then.
    """
    by_group = groups(records)
    for (dataset, bonds), recs in by_group.items():
        _check_one_fingerprint(dataset, recs, bonds)
    labels_per_dataset: dict[str, int] = {}
    for dataset, _ in by_group:
        labels_per_dataset[dataset] = labels_per_dataset.get(dataset, 0) + 1
    sections: list[str] = []
    for (dataset, bonds), recs in by_group.items():
        heading = (
            dataset
            if labels_per_dataset[dataset] == 1
            else f"{dataset}, label_bonds {_bonds_text(bonds)}"
        )
        rows = _rows(recs)
        family = comparisons(rows)
        n_folds = {len(f) for f in rows.values()}
        heads = sorted({k[1] for k in rows if k[0] == "A"})
        found = _geometries(rows)
        several = len(found) > 1
        # Per geometry, with its name, when there are several; else once, as before #733.
        per_geometry: list[tuple[str | None, str]] = (
            [(g, f", geometry `{g}`") for g in found] if several else [(None, "")]
        )
        m = len(family)
        sections += [
            f"## {heading}",
            "",
            (
                f"Folds per row: {sorted(n_folds)}.  Holm family: {m} comparisons at "
                f"α = {ALPHA}, first threshold α/m = {_p(ALPHA / m) if m else '–'}."
            ),
            "",
            "### Scores",
            "",
            scores_table(rows, geometry=several),
            "",
            degeneracy_line(recs),
            "",
            "### A against B (MCC, paired by fold)",
            "",
            "Mean ΔMCC = A − B over the folds (Holm-adjusted p; ✓: at most α).",
            "",
        ]
        for head in heads:
            for shape, named in per_geometry:
                sections += [
                    f"Head `{head}`{named}:",
                    "",
                    grid_table(family, "A − B", head, ("B", head, None, None, None), shape),
                    "",
                ]
        sections += [
            "### Every arm against the classical baselines (MCC, paired by fold)",
            "",
            "Mean ΔMCC = arm − baseline (Holm-adjusted p; ✓: at most α).",
            "",
        ]
        for head in heads:
            for base in BASELINES:
                for shape, named in per_geometry:
                    sections += [
                        f"Arm A, head `{head}`{named}, against {base[0]}/{base[1]}:",
                        "",
                        grid_table(
                            family, "arm − baseline", head, (*base, None, None, None), shape
                        ),
                        "",
                    ]
        sections += [
            "The other arms:",
            "",
            other_arms_table(family),
            "",
        ]
        if long:
            sections += [
                "### Every comparison",
                "",
                comparison_table(family, "A − B", geometry=several),
                "",
                comparison_table(family, "arm − baseline", geometry=several),
                "",
            ]
        sections += [
            "### Separation measures (real rows of the training part, mean over folds)",
            "",
        ]
        if not heads:
            sections += ["(no arm A lines)", ""]
        for shape, _ in per_geometry:
            if heads and several:
                sections += [f"Geometry `{shape}`:", ""]
            if heads:
                sections += [
                    mechanism_tables(
                        rows,
                        part="train",
                        permutations=permutations,
                        seed=seed,
                        heads=heads,
                        geometry=shape,
                    ),
                    "",
                ]
    return "\n".join(sections)


def main(
    argv: Sequence[str] | None = None,
    *,
    read: Callable[[Path], list[dict[str, Any]]] = read_records,
) -> str:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[1])
    parser.add_argument(
        "results", type=Path, help="JSON lines of benchmark_rydberg_features.py, or .jsonl.xz"
    )
    parser.add_argument(
        "--permutations", type=int, default=10_000, help="draws of the permutation p-values"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the permutation draws")
    parser.add_argument(
        "--long",
        action="store_true",
        help="also print every comparison as a row: n, W pairs, exact p, its floor, effect",
    )
    args = parser.parse_args(argv)
    text = summarise(
        read(args.results), permutations=args.permutations, seed=args.seed, long=args.long
    )
    print(text)
    return text


if __name__ == "__main__":
    main()
