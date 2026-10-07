"""
tests/test_evaluation_plots.py
===============================
hqnn_forge.evaluation.plots: each figure is built from the right numbers.

Skipped where matplotlib is not installed (it is an optional extra).
"""

from __future__ import annotations

import io

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.text import Annotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from hqnn_forge.evaluation import plots

THESIS = {  # (MCC, params) from the benchmark README
    "SHNN": (0.5758, 122),
    "PHNN": (0.5688, 489),
    "SNN": (0.5633, 3201),
    "TabNet": (0.4824, 6176),
    "ResNet": (0.6933, 8897),
    "FT-T": (0.6934, 14869),
    "SAINT": (0.6975, 29357),
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestConfusionMatrix:
    Y_TRUE = [0, 0, 0, 0, 1, 1, 1]
    Y_PRED = [0, 0, 1, 0, 1, 0, 1]  # tn=3 fp=1 fn=1 tp=2

    def test_counts(self) -> None:
        cm = plots.confusion_matrix(self.Y_TRUE, self.Y_PRED)
        assert cm.tolist() == [[3, 1], [1, 2]]

    def test_matches_scikit_learn(self) -> None:
        sk = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        t, p = rng.integers(0, 2, 40), rng.integers(0, 2, 40)
        assert np.array_equal(
            plots.confusion_matrix(t, p), sk.confusion_matrix(t, p, labels=[0, 1])
        )

    def test_figure_shows_every_count(self) -> None:
        fig = plots.plot_confusion_matrix(self.Y_TRUE, self.Y_PRED, labels=("legit", "fraud"))
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert sorted(t.get_text() for t in ax.texts) == ["1", "1", "2", "3"]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["legit", "fraud"]
        assert ax.get_xlabel() == "Predicted" and ax.get_ylabel() == "True"

    def test_normalized_rows(self) -> None:
        fig = plots.plot_confusion_matrix(self.Y_TRUE, self.Y_PRED, normalize=True)
        texts = sorted(t.get_text() for t in fig.axes[0].texts)
        assert texts == ["0.25", "0.33", "0.67", "0.75"]

    def test_normalize_with_empty_row(self) -> None:
        fig = plots.plot_confusion_matrix([0, 0], [0, 1], normalize=True)
        assert sorted(t.get_text() for t in fig.axes[0].texts) == ["0.00", "0.00", "0.50", "0.50"]

    def test_draws_into_given_axes(self) -> None:
        fig, (a, b) = plt.subplots(1, 2)
        assert plots.plot_confusion_matrix([0, 1], [0, 1], ax=b) is fig
        assert len(b.texts) == 4 and len(a.texts) == 0

    @pytest.mark.parametrize(
        "t, p, match",
        [
            ([0, 1], [0], "differ in length"),
            ([0, 2], [0, 1], "binary 0/1"),
            # Probabilities must raise, not be truncated to zeros by the cast.
            ([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.95], "binary 0/1"),
        ],
    )
    def test_errors(self, t: list, p: list, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            plots.plot_confusion_matrix(t, p)

    def test_probabilities_are_rejected_not_truncated(self) -> None:
        with pytest.raises(ValueError, match="y_pred"):
            plots.confusion_matrix([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.95])


class TestFoldBoxplot:
    SCORES = {"SHNN": [0.55, 0.58, 0.60, 0.57, 0.59], "SNN": [0.56, 0.55, 0.57, 0.56, 0.58]}

    def test_one_box_per_model_in_order(self) -> None:
        fig = plots.plot_fold_metric_boxplot(self.SCORES, metric_name="MCC")
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["SHNN", "SNN"]
        assert ax.get_ylabel() == "MCC"
        # Tie each median to the box it belongs to: a median line spans the
        # full box width (0.5) centred on the box position, while the whisker
        # caps span half that, so the span picks out one line per box.
        segments = [
            (np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float))
            for line in ax.lines
        ]
        medians = {
            round(float(xs.mean()), 6): float(ys[0])
            for xs, ys in segments
            if xs.size == 2 and abs(float(np.ptp(xs)) - 0.5) < 1e-9
        }
        ticks = {
            t.get_text(): round(float(x), 6) for t, x in zip(ax.get_xticklabels(), ax.get_xticks())
        }
        assert medians == {ticks["SHNN"]: pytest.approx(0.58), ticks["SNN"]: pytest.approx(0.56)}

    def test_points_overlaid(self) -> None:
        ax = plots.plot_fold_metric_boxplot(self.SCORES).axes[0]
        offsets = np.concatenate([c.get_offsets() for c in ax.collections])
        assert offsets.shape == (10, 2)
        assert sorted(offsets[:, 1].tolist()) == sorted(
            s for scores in self.SCORES.values() for s in scores
        )

    def test_points_can_be_hidden(self) -> None:
        ax = plots.plot_fold_metric_boxplot(self.SCORES, show_points=False).axes[0]
        assert len(ax.collections) == 0

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plots.plot_fold_metric_boxplot({})
        with pytest.raises(ValueError, match=r"no scores for: \['B'\]"):
            plots.plot_fold_metric_boxplot({"A": [0.1], "B": []})
        with pytest.raises(ValueError, match=r"non-finite scores for: \['B'\]"):
            plots.plot_fold_metric_boxplot({"A": [0.1, 0.2], "B": [0.1, float("nan")]})


class TestEfficiencyFrontier:
    def test_pareto_frontier_on_thesis_numbers(self) -> None:
        # SHNN has the fewest parameters.  PHNN, SNN and TabNet score below it
        # with more parameters, so they are dominated.  ResNet, FT-T (0.6934 >
        # 0.6933) and SAINT each buy a higher score with more parameters.
        assert plots.pareto_frontier(THESIS) == ["SHNN", "ResNet", "FT-T", "SAINT"]

    def test_pareto_ties(self) -> None:
        models = {"a": (0.5, 10), "b": (0.5, 10), "c": (0.5, 20)}
        # a and b tie exactly: neither is strictly better, both stay; c is dominated
        assert plots.pareto_frontier(models) == ["a", "b"]

    def test_figure(self) -> None:
        fig = plots.plot_efficiency_frontier(THESIS)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert sorted(t.get_text() for t in ax.texts) == sorted(THESIS)
        (step,) = [line for line in ax.lines if line.get_label() == "Pareto frontier"]
        assert list(np.asarray(step.get_xdata())) == [122, 8897, 14869, 29357]
        assert list(np.asarray(step.get_ydata())) == [
            THESIS[n][0] for n in ["SHNN", "ResNet", "FT-T", "SAINT"]
        ]
        # Each label is anchored on its own model's point, and each point is
        # drawn at that model's (params, score).
        labels = [t for t in ax.texts if isinstance(t, Annotation)]
        assert len(labels) == len(ax.texts)
        assert {t.get_text(): tuple(t.xy) for t in labels} == {
            name: (float(params), score) for name, (score, params) in THESIS.items()
        }
        points = np.concatenate([c.get_offsets() for c in ax.collections])
        assert sorted(map(tuple, points.tolist())) == sorted(
            (float(params), score) for score, params in THESIS.values()
        )

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plots.plot_efficiency_frontier({})
        with pytest.raises(ValueError, match=r"must be positive.*\['bad'\]"):
            plots.plot_efficiency_frontier({"bad": (0.5, 0)})

    def test_never_shows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*_: object, **__: object) -> None:
            raise AssertionError("plt.show() called")

        monkeypatch.setattr(plt, "show", fail)
        plots.plot_efficiency_frontier(THESIS)
        plots.plot_fold_metric_boxplot({"a": [0.1, 0.2]})
        plots.plot_confusion_matrix([0, 1], [1, 1])


class TestSubfigureAxes:
    """
    #173: an axes on a ``fig.subfigures()`` sub-figure has ``ax.figure`` set
    to the SubFigure, which has no ``savefig``.  Every plot returns the root
    Figure instead, the object the caller can save.
    """

    @staticmethod
    def _subfigures(parent: Figure | SubFigure, nrows: int = 1, ncols: int = 2) -> np.ndarray:
        """``parent.subfigures``, narrowed: more than one sub-figure comes as an array."""
        subs = parent.subfigures(nrows, ncols)
        assert isinstance(subs, np.ndarray)
        return subs

    @staticmethod
    def _draw(kind: str, ax: Axes) -> Figure:
        if kind == "confusion":
            return plots.plot_confusion_matrix([0, 1, 1], [0, 1, 0], ax=ax)
        if kind == "boxplot":
            return plots.plot_fold_metric_boxplot({"a": [0.5, 0.6], "b": [0.4, 0.5]}, ax=ax)
        return plots.plot_efficiency_frontier({"a": (0.5, 10), "b": (0.6, 20)}, ax=ax)

    @pytest.mark.parametrize("kind", ["confusion", "boxplot", "frontier"])
    def test_returns_the_root_figure_which_can_be_saved(self, kind: str) -> None:
        root = plt.figure()
        sub = self._subfigures(root)[1]
        ax = sub.subplots()
        out = self._draw(kind, ax)
        assert out is root and type(out) is Figure
        out.savefig(io.BytesIO(), format="png")

    def test_nested_subfigures_reach_the_root(self) -> None:
        root = plt.figure()
        inner = self._subfigures(self._subfigures(root)[0], 2, 1)[1]
        assert self._draw("confusion", inner.subplots()) is root

    def test_confusion_contents_land_on_the_subfigure(self) -> None:
        # The root is returned, but the colorbar is stolen from the sub-figure
        # axes and lives on that sub-figure, not on the root beside it.
        root = plt.figure()
        sub = self._subfigures(root)[1]
        ax = sub.subplots()
        self._draw("confusion", ax)
        assert len(sub.axes) == 2 and ax in sub.axes
        assert root.axes == sub.axes  # the root lists its sub-figures' axes, none of its own
        assert [t.get_text() for t in ax.texts] == ["1", "0", "1", "1"]

    def test_detached_axes_raises(self) -> None:
        ax = plt.figure().add_subplot()
        ax.remove()
        with pytest.raises(ValueError, match="not attached"):
            plots.plot_confusion_matrix([0, 1], [0, 1], ax=ax)


def test_reliability_diagram_plots_the_reliability_curve() -> None:
    # Here rather than in test_calibration.py: the lowest-floors CI job has no
    # matplotlib and leaves exactly this module out.
    from hqnn_forge.evaluation import expected_calibration_error, reliability_curve

    rng = np.random.default_rng(7)
    prob = 1 / (1 + np.exp(-3 * rng.standard_normal(500)))
    y = (rng.random(500) < prob).astype(float)
    fig = plots.plot_reliability_diagram(y, prob, strategy="quantile")
    (ax,) = fig.axes
    diagonal, model = ax.get_lines()
    confidence, frequency, counts = reliability_curve(y, prob, 10, "quantile")
    np.testing.assert_allclose(np.asarray(diagonal.get_xydata(), dtype=float), [[0, 0], [1, 1]])
    np.testing.assert_allclose(np.asarray(model.get_xdata(), dtype=float), confidence.numpy())
    np.testing.assert_allclose(np.asarray(model.get_ydata(), dtype=float), frequency.numpy())
    assert [t.get_text() for t in ax.texts] == [str(int(n)) for n in counts.tolist()]
    ece = expected_calibration_error(y, prob, 10, "quantile")
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_texts()[1].get_text() == f"model (ECE {ece:.3f})"
