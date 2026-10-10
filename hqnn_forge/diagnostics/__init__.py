"""
hqnn_forge.diagnostics
======================
Tools for inspecting the quantum side of a hybrid model without reading the
circuit source: what the circuit costs on hardware, and later, how it trains.
The separation measures apply to any fixed feature matrix, whatever produced
it.

Exported symbols
----------------
CircuitSummary           Frozen record: qubits, depth, gate counts, trainable parameters.
circuit_summary          Build a CircuitSummary from an encoding layer or a hybrid classifier.
draw_circuit             Text drawing of the same circuit, for logs and notebooks.
count_inert_parameters   Trainable gate parameters that can never reach a measurement.
count_inert_weights      The same counted per weight entry, as circuit_summary reports it.
LOGICAL_GATE_SET         Gate names circuits are decomposed to before counting; circuit_summary
                         also decomposes a MultiRZ on more than two wires.
gradient_variance        Variance of the cost gradient over random weight draws.
gradient_variance_sweep  The same over a grid of qubit and layer counts.
GradientVarianceResult   Result of gradient_variance.
format_sweep             Text table of a sweep.
fisher_information_matrix     Fisher matrix and spectrum of the quantum weights.
fisher_information_spectrum   Its eigenvalues only, descending.
FisherSpectrum                Result of fisher_information_matrix.
effective_dimension           Effective dimension (Abbas et al. 2021) over random draws.
effective_dimension_from_spectra  The formula alone, from Fisher eigenvalues.
EffectiveDimensionResult      Result of effective_dimension.
expressibility                KL divergence of the fidelity distribution from Haar (Sim et al. 2019).
entangling_capability         Mean Meyer–Wallach entanglement over sampled states.
meyer_wallach                 Meyer–Wallach Q of given state vectors.
ExpressibilityResult, EntanglingCapabilityResult  Their results.
separation_measures           Every scalar separation measure of a feature matrix and its labels.
SeparationMeasures            Result of separation_measures.
pairwise_distances            Pairwise feature distances, within and between classes, and their ratio.
PairwiseDistances             Result of pairwise_distances.
fisher_discriminant_ratio     Fisher discriminant ratio along the best linear direction.
effective_rank                Effective rank of the feature covariance (exponential spectral entropy).
linear_feature_kernel         The linear kernel F Fᵀ, for kernel_target_alignment.
"""

from hqnn_forge.diagnostics.circuit import (
    LOGICAL_GATE_SET,
    CircuitSummary,
    circuit_summary,
    count_inert_parameters,
    count_inert_weights,
    draw_circuit,
)
from hqnn_forge.diagnostics.expressibility import (
    EntanglingCapabilityResult,
    ExpressibilityResult,
    entangling_capability,
    expressibility,
    meyer_wallach,
)
from hqnn_forge.diagnostics.fisher import (
    EffectiveDimensionResult,
    FisherSpectrum,
    effective_dimension,
    effective_dimension_from_spectra,
    fisher_information_matrix,
    fisher_information_spectrum,
)
from hqnn_forge.diagnostics.gradients import (
    GradientVarianceResult,
    format_sweep,
    gradient_variance,
    gradient_variance_sweep,
)
from hqnn_forge.diagnostics.separation import (
    PairwiseDistances,
    SeparationMeasures,
    effective_rank,
    fisher_discriminant_ratio,
    linear_feature_kernel,
    pairwise_distances,
    separation_measures,
)

__all__: list[str] = [
    "LOGICAL_GATE_SET",
    "CircuitSummary",
    "EffectiveDimensionResult",
    "EntanglingCapabilityResult",
    "ExpressibilityResult",
    "FisherSpectrum",
    "GradientVarianceResult",
    "PairwiseDistances",
    "SeparationMeasures",
    "circuit_summary",
    "count_inert_parameters",
    "count_inert_weights",
    "draw_circuit",
    "effective_dimension",
    "effective_dimension_from_spectra",
    "effective_rank",
    "entangling_capability",
    "expressibility",
    "fisher_discriminant_ratio",
    "fisher_information_matrix",
    "fisher_information_spectrum",
    "format_sweep",
    "gradient_variance",
    "gradient_variance_sweep",
    "linear_feature_kernel",
    "meyer_wallach",
    "pairwise_distances",
    "separation_measures",
]
