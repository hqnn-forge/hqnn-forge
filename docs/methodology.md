# How hqnn-forge compares models

This document states the rules the library's comparisons follow, so that a result produced
with it can be checked without reading the source. Each rule names the function that
implements it.

## Question

The comparisons answer one question: **on an imbalanced binary classification task, does a
small quantum layer add enough per parameter to justify it?** A hybrid model is compared with a
purely classical control that has the same number of parameters and is trained the same way on
the same data.

What they do **not** claim:

- **No quantum advantage.** A hybrid model that beats its control on a dataset shows that this
  architecture used its parameters better on that data, in simulation. It says nothing about
  classical models of other shapes, or about computational hardness.
- **Simulation unless stated.** Every result is from a classical simulator (`default.qubit`,
  `lightning.qubit`, `default.mixed`). The noise sweep below is a model of hardware noise, not a
  measurement on hardware.
- **Parameter count is the budget, not a claim about value.** One rotation angle counts as one
  parameter, like one weight of a `Linear` layer. Matching the count does not claim the two are
  worth the same.

## Controls

`classical_baseline(model)` (`hqnn_forge.utils`) builds an **untrained** multilayer perceptron
for a given hybrid model, which is trained from scratch with the same recipe. Reusing the
hybrid's trained classical layers, or switching its circuit off (`disable_quantum_layer`),
would measure a model trained *with* the circuit, not what a classical model achieves alone.

- **Serial hybrid** (`HybridBinaryClassifier`): `Linear(n_in → h) → tanh → Linear(h → 1)`, one
  hidden layer in place of encoder, circuit and head.
- **Parallel hybrid** (`ParallelHybridClassifier`): its classical branch plus a head,
  `Linear(n_in → w) → ReLU → Linear(w → w) → ReLU → Linear(w → 1)`, the branch widened to the
  matching width.
- **Matching rule.** The width is the integer whose parameter count is closest to the hybrid's
  live count (next bullet), ties to the smaller model, so the two differ by at most half a
  width step. `dropout_p` and `init_seed` are carried over.
- **Which count.** Matching uses the structurally *live* count,
  `count_parameters() - circuit_summary(model).n_inert_params`: a circuit weight that can never
  reach the measurement adds no capacity. For the published SHNN that is 106 of 122 (16
  structurally inert; by gradient, 102 are live, #234), halfway between widths 10 and 11, so
  the tie rule gives a 101-parameter control; the exact live count 102 gives the same one.
  Efficiency figures (MCC/kParam) and the benchmark's `n_parameters` keep the total.

## Data handling

For each dataset, `run_benchmark` uses **stratified outer folds** (`stratified_kfold`), so every
test fold keeps the rare class. In each fold, the hybrid and its control receive exactly the
same data:

1. **Scaling** is fitted on the fold's training part only (mean and standard deviation) and
   applied to every row of the fold. Statistics over test rows would leak them into training.
2. **Validation split.** The training part is split once more, stratified, and one
   `validation_folds`-th is held out for early stopping and threshold tuning. It holds real
   rows only.
3. **Oversampling.** With `oversample=True` (the default), SMOTE (Chawla et al. 2002;
   `oversample_fold`) runs on the remaining training rows only, after scaling, since its
   neighbour search measures distances. No synthetic row is derived from a validation or test
   row.
4. **Training.** The same loss, optimiser, learning rate, batch size, epoch budget, early
   stopping, batch order and initialisation seed for both models.
5. **Test rows** are used only to score the trained models: the final score, and with
   `noise_levels` the hybrid's score under each noise level (see Noise).

`hqnn_forge.data` loaders return data as published, without scaling; their docstrings state any
cleaning (for example, dropped leakage columns or rows with missing values) and the resulting
positive rate.

## Thresholds

With a rare positive class, a classifier's probabilities often never cross 0.5 for positives,
and predicting "negative" for everything is already highly accurate. A fixed threshold of 0.5
therefore mostly measures calibration.

The runner instead uses the threshold that maximises **MCC on the validation rows**, at the
epoch early stopping keeps (`find_optimal_threshold`, via `train_model(monitor="mcc")`), and
applies it **unchanged to the test rows**. Tuning the threshold on test rows would report the
best case, not an estimate. MCC is the primary metric because it stays informative under
imbalance, where accuracy does not.

## Calibration

MCC at a tuned threshold says nothing about whether a predicted probability means what it
says, and for risk scoring it has to. Each fold therefore also records the Brier score and
the expected calibration error of the test probabilities (`brier_mean` and `ece_mean` in the
records, per fold in `FoldResult`). The ECE uses 10 equal-count bins: on imbalanced data,
equal-width bins put nearly every sample in the lowest bin. Focal loss, the default, is
known to change calibration, so compare these columns before reading a probability as a
probability. `TemperatureScaler` and `PlattScaler` (`hqnn_forge.evaluation`) fix
calibration after training, fitted on the validation split; `train_model` records the
validation temperature in its history. Temperature scaling is monotone, so it leaves every
ranking-based number above unchanged. `HybridClassifierEstimator(calibration="temperature")`
(or `"platt"`) fits one on its validation split and applies it in `predict_proba`; `predict`
keeps deciding on the uncalibrated probabilities, so it does not change, and `threshold_`
reports the threshold mapped through the calibrator (#359). With focal loss the fitted
temperature was 0.37 to 0.62 over four seeds of the synthetic data of the estimator's tests
(240 samples, 30 % validation): there the model was under-confident, and calibration
sharpened it. That is one small dataset, not a general result about focal loss.

## Statistics

| Question | Test | Where |
|---|---|---|
| Does the hybrid differ from its control on one dataset? | Paired Wilcoxon signed-rank test over folds (Wilcoxon 1945), with the rank-biserial correlation as effect size (Kerby 2014) | `wilcoxon_signed_rank`, `rank_biserial_correlation`; reported by `run_benchmark` |
| Do several models differ across several datasets? | Friedman test with the Iman–Davenport correction, then Nemenyi or Holm-corrected comparisons against a control (Demšar 2006) | `friedman_test`, `nemenyi_critical_difference`, `compare_to_control`, `holm_correction` (#273) |
| How uncertain is one fold's score? | Class-stratified bootstrap interval, BCa or percentile (Efron 1987), and the paired interval of the difference between two models | `bootstrap_ci`, `paired_bootstrap_ci` (#274) |
| How much does a score depend on initialisation? | Repetition over seeds | `run_benchmark(n_seeds=...)` |

Rules that follow from the tests:

- **Pairing is by fold.** Both models are scored on the same rows, so a hard fold is hard for
  both, and pairing removes that shared variation.
- **The attainable p-value has a floor.** With `n` untied folds, the two-sided exact test
  cannot go below `2 / 2^n`: 5 folds cannot reach 0.05 whatever the scores. `run_benchmark`
  reports this floor as `wilcoxon_min_p`; a result above 0.05 at 5 folds is inconclusive by
  construction, and more folds are the remedy, not a looser threshold. Folds where both
  models score the same drop out.
- **Two models across datasets:** with exactly two models, the Wilcoxon test over per-dataset
  scores is the appropriate test (Demšar 2006); Friedman applies from three models on.
- **Seeds.** With `n_seeds > 1`, each fold is trained once per seed and its score is the mean
  over seeds, so the paired test still pairs folds. `mcc_seed_std` reports the across-seed
  spread: a model whose score depends strongly on its initialisation shows it.
- **Undefined is not zero.** If every fold ties, the test is undefined and the p-value is
  reported as missing, not as 1.

## Tuning

A hybrid model tuned carefully against an untuned control proves nothing. With
`run_benchmark(tuning=Tuning(...))`:

- **Same budget, in trials.** Each model gets the same number of configurations per outer fold,
  counted in trials rather than seconds, so the slower simulated model is not penalised.
- **Same selection.** Every configuration is scored by the mean MCC over the same inner
  stratified folds of the outer training part, for both models.
- **No test data.** Tuning sees only the outer training part; the outer test rows are used
  once, after the configuration is chosen.
- **Training settings only.** The search spaces may vary `lr`, `batch_size`, `max_epochs` and
  `patience`. Architecture is not tuned: the control's size is matched to the hybrid's, and
  tuning either architecture would break that match.
- The spaces, the budget and every chosen configuration are recorded.

## Noise

`run_benchmark(noise_levels=...)` re-scores each trained hybrid model on its test rows with a
`DepolarizingChannel(p)` inserted after every gate (`noise_position="all"`) or before the
measurements (`"end"`), at the fold's threshold, and compares it per level with the noise-free
control. The report states the first level at which the hybrid is no longer significantly
better, if it ever was.

What this models, and what it does not:

- **Inference-time noise on a model trained without it.** It answers whether an advantage
  measured in noiseless simulation survives on a noisy device. Training under noise
  (`noise_level=` on the models) is a different question.
- **Uniform depolarising noise** as a simple gate-noise model. It is not a device's calibrated
  noise model: no coherent errors, crosstalk or readout-specific errors.
- **Mixed-state simulation** (`default.mixed`), which costs `4^n` memory per sample and limits
  the sweep to small circuits.

## Reproducibility

`run_benchmark(record_path=...)` writes an **experiment record** (`hqnn_forge.experiment`,
strict JSON) with:

- every setting of the run, and per dataset the class and constructor arguments of the hybrid
  and its control;
- the root seed and, per fold, model and seed, the initialisation and batch-order seeds used,
  with the train, validation and test row indices;
- the Python, platform and package versions (PennyLane, pennylane-lightning, torch, NumPy,
  scikit-learn, SciPy), and the PennyLane device each circuit actually ran on, since the
  device factory can fall back;
- a SHA-256 fingerprint of each dataset (not the data), tuning choices, noise results and the
  metrics.

`load_record` reports which recorded versions differ from the running environment, and
`rerun_benchmark` repeats a run from its record on the same data. On CPU it reproduces the
per-fold scores exactly.

## References

- Chawla, Bowyer, Hall & Kegelmeyer (2002), "SMOTE: Synthetic Minority Over-sampling
  Technique", *Journal of Artificial Intelligence Research* 16, 321–357.
- Demšar (2006), "Statistical Comparisons of Classifiers over Multiple Data Sets", *Journal of
  Machine Learning Research* 7, 1–30.
- Efron (1987), "Better Bootstrap Confidence Intervals", *Journal of the American Statistical
  Association* 82(397), 171–185.
- Kerby (2014), "The Simple Difference Formula: An Approach to Teaching Nonparametric
  Correlation", *Comprehensive Psychology* 3, 11.IT.3.1.
- Wilcoxon (1945), "Individual Comparisons by Ranking Methods", *Biometrics Bulletin* 1(6),
  80–83.
