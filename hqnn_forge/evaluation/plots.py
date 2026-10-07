"""
hqnn_forge.evaluation.plots
===========================
Figures for benchmarking hybrid models: confusion matrix, reliability diagram, per-fold metric
distributions, and the score-versus-parameter-count frontier.

Every function returns the ``matplotlib.figure.Figure`` it drew on and never
calls ``plt.show()``, so the figures compose with notebooks and with
``fig.savefig`` pipelines.  Pass ``ax`` to draw into an existing axes; the
returned figure is then that axes' root figure, the top-level ``Figure`` even
when the axes sits on a ``fig.subfigures()`` sub-figure.

matplotlib is an optional dependency (the ``examples`` extra); it is imported
when a plotting function is first called, and a missing install raises an
``ImportError`` that says how to add it.  :func:`pareto_frontier` needs no
matplotlib.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from hqnn_forge.evaluation.calibration import BinStrategy


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "hqnn_forge.evaluation.plots needs matplotlib: "
            'pip install "hqnn-forge[examples]" (or pip install matplotlib).'
        ) from exc
    return plt


def _axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is not None:
        # For an axes on a sub-figure, ax.figure is the SubFigure, which has no
        # savefig. Its .figure is the root Figure (and a Figure's .figure is
        # itself), which works on every matplotlib with sub-figures, unlike
        # ax.get_figure(root=True), new in 3.10.
        parent = ax.figure
        if parent is None:  # a removed or never-attached axes; the stubs leave out None
            raise ValueError("ax is not attached to a figure.")
        return parent.figure, ax
    fig, new_ax = _pyplot().subplots(figsize=figsize)
    return fig, new_ax


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def _as_binary(y: npt.ArrayLike, name: str) -> npt.NDArray[np.int64]:
    """
    Flatten to 1-D int64 after checking that every value is a 0/1 label.

    The check runs before the cast, as in ``thresholds._as_binary``: casting
    first would turn probabilities into a plausible-looking matrix of zeros
    instead of raising.
    """
    a = np.asarray(y).reshape(-1)
    if a.size and not np.isin(a, (0, 1)).all():
        raise ValueError(
            f"{name} must contain only binary 0/1 labels; got values {np.unique(a).tolist()}."
        )
    return a.astype(np.int64)


def confusion_matrix(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """
    2×2 count matrix, rows = true class (0, 1), columns = predicted class.

    ``[[tn, fp], [fn, tp]]``, the scikit-learn layout.
    """
    t = _as_binary(y_true, "y_true")
    p = _as_binary(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred differ in length: {t.size} vs {p.size}.")
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


def plot_confusion_matrix(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    labels: tuple[str, str] = ("negative", "positive"),
    normalize: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """
    Heat-map of the binary confusion matrix with the value in every cell.

    Parameters
    ----------
    y_true, y_pred:
        Binary labels, e.g. ``model.predict(X)``.
    labels:
        Names for class 0 and class 1.
    normalize:
        Show each row as fractions of its true class (recall per class)
        instead of counts.  Rows with no samples show 0.
    title:
        Axes title.  Default: ``"Confusion matrix"``.
    ax:
        Draw into this axes.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        totals = cm.sum(axis=1, keepdims=True)
        values = np.divide(cm, totals, out=np.zeros_like(cm, dtype=np.float64), where=totals > 0)
    else:
        values = cm.astype(np.float64)

    fig, axes = _axes(ax, (4.2, 3.6))
    vmax = 1.0 if normalize else max(float(values.max()), 1.0)
    image = axes.imshow(values, cmap="Blues", vmin=0.0, vmax=vmax)
    fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    threshold = vmax / 2.0
    for i in range(2):
        for j in range(2):
            text = f"{values[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
            axes.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
            )
    axes.set_xticks([0, 1], labels=list(labels))
    axes.set_yticks([0, 1], labels=list(labels))
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title(title if title is not None else "Confusion matrix")
    return fig


def plot_reliability_diagram(
    y_true: npt.ArrayLike,
    prob: npt.ArrayLike,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """
    Reliability diagram: observed positive frequency against mean predicted
    probability per bin, with the diagonal of perfect calibration and the
    expected calibration error in the legend.

    Parameters
    ----------
    y_true, prob:
        Binary labels and positive-class probabilities.
    n_bins, strategy:
        As for :func:`hqnn_forge.evaluation.calibration.reliability_curve`;
        ``"quantile"`` suits imbalanced data.
    title:
        Axes title.  Default: ``"Reliability diagram"``.
    ax:
        Draw into this axes.
    """
    from hqnn_forge.evaluation.calibration import reliability_curve

    confidence, frequency, counts = reliability_curve(y_true, prob, n_bins, strategy)
    # expected_calibration_error, from the curve already computed.
    ece = float((counts / counts.sum() * (frequency - confidence).abs()).sum())
    fig, axes = _axes(ax, (4.2, 4.0))
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    axes.plot(confidence.numpy(), frequency.numpy(), marker="o", label=f"model (ECE {ece:.3f})")
    for x, yv, n in zip(confidence.tolist(), frequency.tolist(), counts.tolist(), strict=True):
        axes.annotate(
            f"{int(n)}", (x, yv), textcoords="offset points", xytext=(4, -10), fontsize=7
        )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_xlabel("Mean predicted probability")
    axes.set_ylabel("Observed frequency")
    axes.legend(loc="upper left")
    axes.set_title(title if title is not None else "Reliability diagram")
    return fig


# ---------------------------------------------------------------------------
# Per-fold distributions
# ---------------------------------------------------------------------------


def plot_fold_metric_boxplot(
    model_scores: Mapping[str, Sequence[float]],
    *,
    metric_name: str = "MCC",
    show_points: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """
    One box per model over its per-fold scores, in the mapping's order.

    With a handful of folds a box hides how few points it summarises, so the
    individual fold scores are overlaid by default.

    Parameters
    ----------
    model_scores:
        ``{model_name: [score_fold_1, ...]}``.
    metric_name:
        Y-axis label.
    show_points:
        Overlay each fold's score.
    ax:
        Draw into this axes.
    """
    if not model_scores:
        raise ValueError("model_scores is empty.")
    names = list(model_scores)
    data = [np.asarray(model_scores[n], dtype=np.float64) for n in names]
    empty = [n for n, d in zip(names, data) if d.size == 0]
    if empty:
        raise ValueError(f"no scores for: {empty}.")
    nonfinite = [n for n, d in zip(names, data) if not np.isfinite(d).all()]
    if nonfinite:
        # A NaN fold score makes the box statistics NaN, so the box silently
        # disappears while its tick label stays behind.
        raise ValueError(f"non-finite scores for: {nonfinite}.")

    fig, axes = _axes(ax, (max(4.0, 1.1 * len(names) + 1.5), 3.8))
    positions = np.arange(1, len(names) + 1)
    axes.boxplot(data, positions=positions, widths=0.5, showmeans=True)
    if show_points:
        for pos, d in zip(positions, data):
            jitter = np.linspace(-0.08, 0.08, d.size) if d.size > 1 else np.zeros(1)
            axes.scatter(pos + jitter, d, s=14, color="black", alpha=0.7, zorder=3)
    axes.set_xticks(positions, labels=names)
    axes.set_ylabel(metric_name)
    axes.set_title(f"{metric_name} across folds")
    axes.grid(axis="y", alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Efficiency frontier
# ---------------------------------------------------------------------------


def pareto_frontier(models: Mapping[str, tuple[float, int]]) -> list[str]:
    """
    Names of the models no other model beats on both axes.

    A model is dominated if another has at most as many parameters and at
    least as high a score, and is strictly better on one of the two.  The
    result is ordered by increasing parameter count.
    """
    items = [(name, float(score), int(params)) for name, (score, params) in models.items()]
    frontier = []
    for name, score, params in items:
        dominated = any(
            p2 <= params and s2 >= score and (p2 < params or s2 > score)
            for n2, s2, p2 in items
            if n2 != name
        )
        if not dominated:
            frontier.append((params, -score, name))
    return [name for _, _, name in sorted(frontier)]


def plot_efficiency_frontier(
    models: Mapping[str, tuple[float, int]],
    *,
    metric_name: str = "MCC",
    ax: Axes | None = None,
) -> Figure:
    """
    Score against trainable-parameter count (log x), with the Pareto frontier.

    Parameters
    ----------
    models:
        ``{model_name: (score, n_parameters)}``, e.g.
        ``{"SHNN": (0.576, 122), "SNN": (0.563, 3201)}``.
    metric_name:
        Y-axis label.
    ax:
        Draw into this axes.

    Notes
    -----
    Points on the frontier are joined by a step line: moving right along it,
    each additional parameter buys a higher score.  Points below the line are
    dominated.
    """
    if not models:
        raise ValueError("models is empty.")
    bad = [n for n, (_, p) in models.items() if int(p) <= 0]
    if bad:
        raise ValueError(f"parameter counts must be positive for a log axis: {bad}.")

    fig, axes = _axes(ax, (5.2, 3.8))
    frontier = pareto_frontier(models)
    for name, (score, params) in models.items():
        on_front = name in frontier
        axes.scatter(
            params,
            score,
            s=40 if on_front else 28,
            color="tab:blue" if on_front else "tab:gray",
            zorder=3,
        )
        axes.annotate(name, (params, score), xytext=(4, 4), textcoords="offset points", fontsize=8)
    xs = [models[n][1] for n in frontier]
    ys = [models[n][0] for n in frontier]
    axes.step(xs, ys, where="post", color="tab:blue", alpha=0.6, label="Pareto frontier")
    axes.set_xscale("log")
    axes.set_xlabel("Trainable parameters (log scale)")
    axes.set_ylabel(metric_name)
    axes.set_title(f"{metric_name} vs parameter count")
    axes.grid(alpha=0.3, which="both")
    axes.legend(loc="lower right", fontsize=8)
    return fig
