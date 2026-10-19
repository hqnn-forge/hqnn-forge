"""
hqnn_forge.evaluation
=====================
Post-hoc evaluation helpers for binary classifiers on imbalanced data.

The plotting helpers live in the submodule ``hqnn_forge.evaluation.plots``
(needs matplotlib; import it explicitly).

Exported symbols
----------------
METRICS                  The binary metrics by name, each mapped to its function.
MULTICLASS_METRICS       The multiclass metrics by name, each mapped to its function.
BootstrapResult          Result of both bootstrap confidence-interval functions.
ControlComparison        One row of compare_to_control.
FriedmanResult           Result of friedman_test.
MulticlassTemperatureScaler  Post-hoc temperature scaler for multiclass predictions.
PlattScaler              Post-hoc Platt calibration scaler.
TemperatureScaler        Post-hoc temperature scaler.
ThresholdSearchResult    (threshold, score) pair returned by find_optimal_threshold.
WilcoxonResult           Result of wilcoxon_signed_rank.
average_ranks             Mean rank of each model over the datasets (1 = best).
balanced_accuracy        Mean of recall over the two classes.
bootstrap_ci             Class-stratified bootstrap interval (BCa or percentile) for a metric.
brier_score              Brier score for probability predictions.
classwise_ece            Classwise expected calibration error.
compare_to_control       Holm-corrected z-tests of every model against a control.
expected_calibration_error  Expected calibration error for probability predictions.
f1_score                 F1 for the positive class.
find_optimal_threshold   Decision threshold that maximises a metric on held-out probabilities.
friedman_from_ranks      The same Friedman test from published average ranks.
friedman_test            Friedman / Iman–Davenport test of k models over N datasets.
holm_correction          Holm step-down adjusted p-values for any family.
macro_f1_score           Macro F1 score for multiclass predictions.
matthews_corrcoef        MCC from labels, pure torch/NumPy.
multiclass_balanced_accuracy  Balanced accuracy for multiclass predictions.
multiclass_brier_score   Brier score for multiclass probabilities.
multiclass_matthews_corrcoef  MCC for multiclass predictions.
nemenyi_critical_difference  Nemenyi CD for all-pairs comparison of average ranks.
paired_bootstrap_ci      Bootstrap interval for the difference between two models on the same samples.
parameter_efficiency     Score per thousand trainable parameters.
pr_auc                   Average precision (PR-AUC), threshold-free; not in METRICS.
rank_biserial_correlation  Effect size for the paired comparison.
reliability_curve        Calibration data for probability predictions.
top_label_ece            Top-label expected calibration error.
wilcoxon_signed_rank     Paired signed-rank test with its attainable p-value floor.
"""

from hqnn_forge.evaluation.bootstrap import (
    BootstrapResult,
    bootstrap_ci,
    paired_bootstrap_ci,
)
from hqnn_forge.evaluation.calibration import (
    MulticlassTemperatureScaler,
    PlattScaler,
    TemperatureScaler,
    brier_score,
    classwise_ece,
    expected_calibration_error,
    multiclass_brier_score,
    reliability_curve,
    top_label_ece,
)
from hqnn_forge.evaluation.multiclass import (
    MULTICLASS_METRICS,
    macro_f1_score,
    multiclass_balanced_accuracy,
    multiclass_matthews_corrcoef,
)
from hqnn_forge.evaluation.statistics import (
    ControlComparison,
    FriedmanResult,
    WilcoxonResult,
    average_ranks,
    compare_to_control,
    friedman_from_ranks,
    friedman_test,
    holm_correction,
    nemenyi_critical_difference,
    rank_biserial_correlation,
    wilcoxon_signed_rank,
)
from hqnn_forge.evaluation.thresholds import (
    METRICS,
    ThresholdSearchResult,
    balanced_accuracy,
    f1_score,
    find_optimal_threshold,
    matthews_corrcoef,
    parameter_efficiency,
    pr_auc,
)

__all__: list[str] = [
    "METRICS",
    "MULTICLASS_METRICS",
    "BootstrapResult",
    "ControlComparison",
    "FriedmanResult",
    "MulticlassTemperatureScaler",
    "PlattScaler",
    "TemperatureScaler",
    "ThresholdSearchResult",
    "WilcoxonResult",
    "average_ranks",
    "balanced_accuracy",
    "bootstrap_ci",
    "brier_score",
    "classwise_ece",
    "compare_to_control",
    "expected_calibration_error",
    "f1_score",
    "find_optimal_threshold",
    "friedman_from_ranks",
    "friedman_test",
    "holm_correction",
    "macro_f1_score",
    "matthews_corrcoef",
    "multiclass_balanced_accuracy",
    "multiclass_brier_score",
    "multiclass_matthews_corrcoef",
    "nemenyi_critical_difference",
    "paired_bootstrap_ci",
    "parameter_efficiency",
    "pr_auc",
    "rank_biserial_correlation",
    "reliability_curve",
    "top_label_ece",
    "wilcoxon_signed_rank",
]
