# Noise-aware robustness: prospective protocol

**PROSPECTIVE — NO ROBUSTNESS RESULTS COLLECTED.** This is the decision brief
for [#348](https://github.com/hqnn-forge/hqnn-forge/issues/348), not its results
note. Dataset checks and bounded timing probes below establish feasibility
only. They contain no evidence that noise-aware training improves MCC. The
completed study must replace or complete this note with actual measurements,
saved records and a recommendation or a documented negative result.

## Required evidence

The issue has eight acceptance criteria; none is complete in this draft:

1. Extend the method of `examples/study_trajectory_noise.py` with a documented,
   matched training comparison.
2. Measure stronger noise through strength 0.2.
3. Include amplitude damping and phase damping alongside depolarizing noise.
4. Use the real credit-card data and all three UCI loaders through `run_benchmark`.
5. Measure mismatch between training and evaluation noise levels.
6. Report actual test MCC in `docs/results/`, answering whether, when and by
   how much noise-aware training improves robustness.
7. Pair seeds and use more than five fold pairs for a defensible Wilcoxon
   analysis, reporting its attainable floor and limitations.
8. Give a recommendation for use or document a negative result.

**Status for each criterion: not done, pending protocol decision,
implementation and measurement.** A protocol plan does not satisfy a
measurement criterion.

## What the existing evidence settles

The reference study used breast-cancer data, training-fitted standardisation
and PCA to 4 or 6 inputs, `HybridBinaryClassifier(n, n, 2)`, focal loss,
Adam at 0.05, batch 32, 30 epochs and patience 10, without SMOTE. It selected
checkpoints on clean validation MCC and re-tuned thresholds on noisy validation
rows before noisy test scoring (`examples/study_trajectory_noise.py:16`,
`:108`, `:135`). The
[accepted decision on #343](https://github.com/hqnn-forge/hqnn-forge/pull/343#issuecomment-5917638593)
scoped its conclusions to that proxy and deferred benchmark confirmation to
[#414](https://github.com/hqnn-forge/hqnn-forge/issues/414).

The benchmark's defaults are different: raw feature inputs, training-only
standardisation and SMOTE, Adam at 0.01, batch 256 and up to 100 epochs
(`hqnn_forge/benchmark.py:576`). Using those defaults silently would change
the reference method. Training-fitted PCA inside each fold, the recipe and any
longer-budget control therefore need an explicit choice. The
[#414 discussion](https://github.com/hqnn-forge/hqnn-forge/issues/414#issuecomment-6027010714)
also warns that the 30-epoch early-stopping budget confounds slow starts with
failure to train.

All prerequisite stacks are on the inspected base,
`8bea1c003e42a9686b4d1a8a7afca246d08f811b`: channels (#313/#336), loaders
(#266), benchmark runner (#269) and the inherited study (#343). Training
supports all three requested channels with density simulation and trajectories.
Phase-damping trajectories use an equivalent Pauli channel; amplitude damping
uses weighted, unnormalised Kraus branches. Its device, differentiation and
analytic-shot restrictions and increased variance must be respected
(`hqnn_forge/noise.py:145`, `:427`). A faster trajectory step does not establish
equal training quality.

The runner's built-in sweep is depolarizing-only and uses the clean threshold
(`hqnn_forge/benchmark.py:505`). Its results contain model configurations,
not trained fold models (`:184`). Channel-aware noisy-validation threshold
tuning, mismatch scoring and recorded replay settings need implementation.
[#428](https://github.com/hqnn-forge/hqnn-forge/issues/428) tracks broader
channel plumbing; #414 tracks the broader trajectory comparison. This study
may implement their necessary subset without claiming either issue complete.
The fold scoring and record API remains a decision, not an implemented hook.

## Bounded probes, not study results

The accompanying [sanitized feasibility metadata](noise_aware_robustness_feasibility.json)
records checks from the inspected base. No dataset rows, predictions, private
paths or patient/card records are included. The source files passed the
loaders' `strict=True` checks; features were finite.

| Dataset / loader options | Rows | Positives | Features |
|---|---:|---:|---:|
| Credit card, `drop_time=True` | 284,807 | 492 | 29 |
| Taiwanese bankruptcy | 6,819 | 220 | 95 |
| Iranian churn, published rows | 3,150 | 495 | 13 |
| Cervical cancer, complete cases | 668 | 45 | 30 |

Churn's optional duplicate removal leaves 2,850 rows and 446 positives.
Duplicate policy must be recorded because duplicates across folds can inflate
scores. Cervical's 45 positives limit fold-level precision; complete-case
selection also limits generalisation.

Synthetic timing probes used CPU, four Torch threads, one interop thread,
Python 3.13.15, PennyLane 0.45.1 and Torch 2.14.1. Each configuration used
`default.qubit`, backprop, 29 inputs, two layers, focal loss, Adam at 0.05,
batch 256, strength 0.2 and placement `all`. After one warm-up optimizer step,
two steps were timed; all final losses were finite.

| Width | Method | Depolarizing seconds/step | Amplitude damping | Phase damping |
|---|---|---:|---:|---:|
| 4 | density | 0.221–0.301 | 0.167–0.176 | 0.149–0.160 |
| 6 | density | 9.285–10.698 | 7.316–7.352 | 7.355–7.458 |
| 4 | trajectories, k=8 | 0.064–0.067 | 0.038–0.038 | 0.063–0.063 |
| 6 | trajectories, k=8 | 0.143–0.151 | 0.086–0.089 | 0.143–0.191 |

These are bounded synthetic probes, not fitted study models or grid wall
times. Two measurements do not characterise runtime variance, convergence or
memory. They do not justify truncating a required benchmark and calling it
complete. A separate bounded runner probe on deduplicated churn completed
six folds at one epoch in 14.48 s; it was a plumbing/timing check, not a
robustness comparison. With the reference input width and batch 32, two warm
depolarizing density steps took 0.061–0.063 s at 4 qubits and 0.454–0.464 s at
6 qubits, showing why the recipe matters to cost.

#414 explicitly permits a documented stratified credit-card subsample or a
density width restriction. It does not specify a cap. As planning examples,
seed-0 stratified samples of 20,000 and 100,000 contain 35 and 173 positives;
ten outer folds then have only 3–4 and 17–18 test positives, respectively,
versus 49–50 using all 492 positives. No cap is selected here.

## Protocol decision required before implementation

**Question:** which explicit data/recipe/method grid, paired inference plan
and fold scoring/record API will answer #348 while preserving the reference
comparison and making any cost restriction scientifically visible?

| Option | Concrete scope | Unresolved consequences |
|---|---|---|
| A | Reference PCA recipe, full credit-card and UCI data, 4/6 qubits, density and inherited trajectory methods | Largest compute requirement; budget and convergence controls must be explicit. |
| B | Reference recipe with the transparent credit-card cap and/or 4-qubit density restriction allowed by #414 | Exact cap and retained width coverage need selection; fewer positives weaken inference. |
| C | Retain 6-qubit credit-card coverage with recorded trajectories, omitting its density cell | Select k and variance checks, especially for weighted Kraus amplitude damping; acknowledge the missing exact reference. |
| D | Narrow to settled documentation/tooling and file the study as follow-up | Leaves #348's measurements and completion unmet; cannot be presented as its answer. |

For A–C, settle PCA versus raw inputs, SMOTE, loss, learning rate, batch size,
epochs, checkpoint selection and extended-budget controls together. Select
channels, placements, widths, trajectory counts and any cap before comparing
scores. Keep initialization, splits and batch order paired across methods;
test data must never choose thresholds, hyperparameters or the grid.

One **unapproved candidate** is five paired initializations averaged within
ten fixed outer folds, training strengths 0.05 and 0.2, evaluation strengths
0, 0.01, 0.05, 0.1 and 0.2, all three channels and explicit off-diagonal
train/evaluation mismatches. The seed average keeps the fold as the test
unit, following `docs/methodology.md`; 50 seed-fold observations are not 50
independent pairs. A candidate primary family is the 24 density-versus-noiseless
comparisons at training/evaluation strength 0.2 and placement `all`
(four datasets × two widths × three channels), with Holm correction; other
cells would be descriptive. With ten nonzero pairs the two-sided exact floor
is 2/1024 = 0.001953125, whose first Holm threshold over 24 comparisons is
attainable (24 × 2/1024 = 0.046875). Zeros reduce the effective pairs; ties,
overlapping training folds, limited power and small positive counts must be
reported. A restricted width grid changes that family and must be declared.

Report fixed-clean-threshold MCC beside noisy-validation-retuned MCC if
`end` placement is retained. With Z readouts and a linear head, common
end-channel affine transformations can preserve ranking, so retuned MCC alone
can conceal calibration changes. The #414 discussion establishes this problem
for depolarizing noise. Mark `end` descriptive or justify another primary
endpoint before seeing the study scores. The final decision must also specify
how channel, training/evaluation strengths, thresholds, preprocessing, seeds,
dataset fingerprints, devices and replayable settings are saved.

No option, cap, architecture, recipe, scoring hook or hypothesis family above
has been adopted by this draft. A subsequent evidence-backed decision and
implementation must precede the scientific grid and the final results note.
