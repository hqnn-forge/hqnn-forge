# Do the interactions of a Rydberg feature map add anything a head can use?

A worked run of the Rydberg benchmark runner (#501), behind #502. Each sample's inputs set the
detunings of a small simulated atom chain, and the excitation probabilities after one pulse are
the features of a small classical head. This note shows how to read one run: scores over the
interaction and dephasing grid, the paired comparisons, the separation measures of #500 beside
them, and the positive control that says whether the pipeline could have shown an effect at
all. It also checks that the model (#495), the solver (#496–#498, #507), the feature map
(#499), the separation measures (#500) and the runner work together end to end.

## Setup

- **Script:** `examples/benchmark_rydberg_features.py` as of commit `2024e49`; this note's pull
  request changes only its `read_records`, which now also opens `.xz` files, not what a run
  writes.
- **Raw results:** `docs/results/rydberg_features_benchmark.jsonl.xz`, one JSON line per arm,
  head, grid point and fold, both datasets. It is the runner's exact JSONL output, every field
  of every line, xz-compressed only because the repository refuses files above 1000 KB (#220);
  nothing was removed. `read_records` and the summary script read it as it is, and
  `xz -dk` gives back the plain file. Every table and number below is printed by
  `examples/summarise_rydberg_benchmark.py` from that file.
- **Reproduce:** the two datasets were run as two invocations and concatenated:

    ```bash
    uv run python examples/benchmark_rydberg_features.py --dataset positive-control \
        --folds 13 --out pc.jsonl
    uv run python examples/benchmark_rydberg_features.py --dataset iranian-churn \
        --data-path data/raw --download --folds 13 --out churn.jsonl
    cat pc.jsonl churn.jsonl > rydberg_features_benchmark.jsonl
    xz -9e rydberg_features_benchmark.jsonl
    mv rydberg_features_benchmark.jsonl.xz docs/results/
    uv run python examples/summarise_rydberg_benchmark.py \
        docs/results/rydberg_features_benchmark.jsonl.xz
    ```

    Every other option is the runner's default and is recorded in each line as `settings`.
    The compressed bytes can differ between xz versions; the decompressed lines are what a
    rerun reproduces, apart from the measured seconds and whether a feature came from the
    cache.
- **Model and units** (`docs/rydberg-model.md`): the two-level ground/Rydberg Hamiltonian
  `H = Ω/2 Σ_i X_i − Σ_i Δ_i n_i + Σ_{i<j} C6/|r_i − r_j|⁶ n_i n_j`, ħ = 1, angular frequencies
  in rad/µs, lengths in µm, times in µs. Ω = 4π rad/µs (the page's reference), `C6 = 2π × 862 690`
  MHz µm⁶ (`70S` of ⁸⁷Rb), a constant pulse of area `ΩT = π` (T = 0.25 µs). Results are
  reported in `V/Ω` and `γ/Ω`.
- **Geometry:** an open chain of N = 4 atoms with the full `1/r⁶` tail. `V/Ω` is set through the
  spacing, `a = R_b · (V/Ω)^(−1/6)` with `R_b = (C6/Ω)^(1/6) = 8.69 µm`: `V/Ω` = 0.1, 0.3, 1, 3,
  10, 30, 100 at `a` = 12.76, 10.62, 8.69, 7.24, 5.92, 4.93, 4.03 µm.
- **Encoding:** one input per atom, `Δ_i = Δ_max σ(x_i)` with `Δ_max = √3 Ω`. The inputs are the
  first four principal components of the standardised raw columns, fitted on each fold's
  training part and squashed to `(−π, π)`; component `i` sets the detuning of atom `i`.
- **Readout:** the four excitation probabilities `⟨n_i⟩` as exact expectation values (no
  shots), no pair correlations. Each feature column is standardised on the fold's real
  training rows before the head (#518).
- **Noise:** Markovian dephasing during the pulse, collapse operators `√γ n_i` (a single-atom
  coherence decays at `γ/2`), at `γ/Ω` = 0, 0.1, 1, 10; no Rydberg decay, no detection error.
  With `γ > 0` the solver takes 200 Strang steps; at `ΩT = π` that left the features within
  `2.8e-6` of a reference at `γ = Ω` and `1.0e-4` at `γ = 10 Ω` (#507). The features were
  computed under `torch.no_grad()`, so `evolve` took its exact `eigh` propagator, not the
  `matrix_exp` path it uses when gradients are needed.
- **Arms**, all on the same folds, validation splits, SMOTE rows and seeds:
    - **A**: interacting Rydberg features → head;
    - **B**: the same map with `interactions=False` → the same head. B is a classical model
      (each feature is a closed-form function of its own input), so **A − B measures what the
      interaction-induced mixing of inputs adds**, not anything quantum against classical;
    - **C**: the four inputs themselves → the same heads (the linear head is then a logistic
      regression): the classical baselines;
    - **D**: `⟨Z_i⟩` of a frozen, randomly initialised 4-qubit angle-embedding circuit (2
      layers) → the same heads; and **D-hybrid**, a `HybridBinaryClassifier` trained end to end,
      less comparable than every other row because its feature map is trained.
- **Heads:** `linear` (`LinearClassifier`, 5 parameters on 4 features) and `mlp`
  (`ClassicalBaseline(4, [8])`, 49 parameters), trained with BCE, Adam, at most 100 epochs, MCC
  early stopping on the validation rows (patience 10). The decision threshold maximises MCC on
  the validation rows and is applied unchanged to the test rows.
- **Tuning budget:** 4 random-search trials per arm, head and fold over `lr` ∈ {0.003, 0.01,
  0.03} and `batch_size` ∈ {32, 128}, each scored by mean MCC over 3 inner stratified folds of
  the training part (everything refitted inside them). The architecture is not tuned.
- **Data:**
    - `positive-control`: the runner's synthetic task, 600 samples, label the sign of
      `Σ_i z_i z_{i+1}` over four standard-normal latents that end up on neighbouring atoms; 15
      noisy raw columns. No single input carries information about the label, so arm B and the
      inputs have no class-mean difference, while arm A's features can.
    - `iranian-churn`: UCI dataset 563 (doi:10.24432/C5JW3Z, CC BY 4.0), `Customer Churn.csv`
      with SHA-256 `90d5fb6b…f7dbe1` as `hqnn_forge.data` pins it, 3,150 rows, 13 columns, 495
      churned (15.7 %), loaded as published (its 300 exact duplicate rows kept).
- **Splits:** 13 stratified outer folds per dataset (why 13: *Statistics* below); in each,
  one fifth of the training part is the validation split, SMOTE on the remaining training
  rows only.
- **Environment:** Python 3.13.15, hqnn-forge 0.1.0, PennyLane 0.45.1, pennylane-lightning
  0.45.0, torch 2.14.1, NumPy 2.5.3, scikit-learn 1.9.1, SciPy 1.18.1, Linux on one AMD Ryzen 7
  250 laptop CPU (16 threads). Recorded in every line as `environment`.
- **Validation the solver passed:** the closed-form limits of `docs/rydberg-model.md` (one-atom
  Rabi and detuned Rabi formulas, the Kronecker-sum and tensor-product structure with
  interactions off, the blockade bound, the damped Rabi oscillation in all three regimes, the
  strong-dephasing limit) in `tests/test_rydberg_hamiltonian.py` and
  `tests/test_rydberg_lindblad.py`; an independent QuTiP `mesolve` reference for 2 to 4 atoms
  with random Ω, Δ, spacing and γ in `tests/test_rydberg_reference.py` (#498); and the feature
  map's single-atom formula and locality checks in `tests/test_rydberg_feature_map.py`. All of
  them passed in this environment at the commit of the run.

### Statistics

- **Paired test:** the exact two-sided Wilcoxon signed-rank test over the outer folds
  (`wilcoxon_signed_rank`), on test MCC, the primary metric (`docs/methodology.md`). PR-AUC is
  reported but not tested.
- **α and correction:** α = 0.05, family-wise, with Holm's step-down correction
  (`holm_correction`), as in `docs/methodology.md`. **All comparisons of one dataset are one
  family:**
    - A against B, per head and grid point: 2 × 7 × 4 = 56;
    - every arm row against each of the two classical baselines (C with the linear head, C with
      the MLP head): A 56 rows, B 2 × 4 = 8 (B does not depend on `V/Ω`), D 2, D-hybrid 1, so
      67 × 2 = 134;
    - m = 190 per dataset.
- **Why 13 folds.** With n untied pairs the smallest two-sided exact p-value is `2/2^n`. Holm's
  first threshold is `α/m = 0.05/190 = 2.63e-4`. 12 folds give `2/4096 = 4.88e-4`, above it;
  **13 give `2/8192 = 2.44e-4`, below it**, so the strongest possible result can be declared
  significant after the correction. With 5 folds the floor would be 0.0625, above even the
  uncorrected α. A fold where two rows tie drops out and raises that pair's floor; the tables
  show each comparison's own floor.
- **Intervals:** a two-sided 95 % Student-t interval of the mean over the 13 folds,
  `mean ± t(0.975, 12) · sd/√13` with `t = 2.179`. It assumes the fold scores are independent
  draws from a roughly normal distribution. They are not independent (the training parts of
  two folds share 11/12 of their rows), and no unbiased estimator of the variance of a k-fold
  mean exists (Bengio & Grandvalet 2004), so the intervals are narrower than the true
  uncertainty and are a guide to the fold spread, not a test.
- **Tracking:** whether the separation measures track the scores is measured by Spearman's
  rank correlation, across the 28 grid points, between a measure's A − B difference (mean over
  folds, on the real rows of the training part; see *Mechanism*) and the A − B difference of mean
  MCC, with a two-sided
  permutation p-value (10,000 draws, seed 0). The 28 grid points are not independent either
  (they share folds, and B is shared across `V/Ω`).

## Results

The tables are the output of `examples/summarise_rydberg_benchmark.py` on the committed file,
unchanged. In the score tables `n` is the number of folds, the interval is the 95 % t-interval of
the fold mean (see *Statistics*), `dim` the feature dimension, `params` the trainable parameters,
`feature s` the mean seconds of feature simulation per fold (on the whole fold, all its rows,
measured when the features were computed) and `fit s` the mean seconds of the final fit. A grid
cell of a comparison is the mean MCC difference over the 13 folds with its Holm-adjusted p-value;
✓ marks the ones at most α = 0.05. Rows of arm B do not depend on `V/Ω` and are shown once per
`γ/Ω`. `--long` prints every comparison with its pairs, exact p-value, floor and rank-biserial
effect.

### Positive control

Folds per row: [13].  Holm family: 190 comparisons at α = 0.05, first threshold α/m = 2.63e-04.

#### Scores

| arm | head | V/Ω | γ/Ω | n | MCC mean ± sd | MCC 95 % CI | PR-AUC mean ± sd | PR-AUC 95 % CI | dim | params | feature s | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | linear | 0.1 | 0 | 13 | 0.187 ± 0.133 | [0.106, 0.267] | 0.557 ± 0.079 | [0.509, 0.605] | 4 | 5 | 0.02 | 0.06 |
| A | linear | 0.3 | 0 | 13 | 0.274 ± 0.138 | [0.191, 0.357] | 0.604 ± 0.070 | [0.562, 0.646] | 4 | 5 | 0.02 | 0.07 |
| A | linear | 1 | 0 | 13 | 0.511 ± 0.098 | [0.452, 0.570] | 0.845 ± 0.025 | [0.830, 0.860] | 4 | 5 | 0.02 | 0.06 |
| A | linear | 3 | 0 | 13 | 0.112 ± 0.150 | [0.022, 0.202] | 0.570 ± 0.084 | [0.519, 0.620] | 4 | 5 | 0.02 | 0.05 |
| A | linear | 10 | 0 | 13 | 0.337 ± 0.119 | [0.265, 0.409] | 0.737 ± 0.094 | [0.680, 0.794] | 4 | 5 | 0.02 | 0.05 |
| A | linear | 30 | 0 | 13 | 0.422 ± 0.130 | [0.344, 0.501] | 0.763 ± 0.075 | [0.718, 0.809] | 4 | 5 | 0.02 | 0.08 |
| A | linear | 100 | 0 | 13 | 0.095 ± 0.165 | [-0.005, 0.194] | 0.541 ± 0.054 | [0.509, 0.573] | 4 | 5 | 0.02 | 0.05 |
| A | linear | 0.1 | 0.1 | 13 | 0.108 ± 0.192 | [-0.008, 0.224] | 0.543 ± 0.091 | [0.488, 0.598] | 4 | 5 | 0.10 | 0.06 |
| A | linear | 0.3 | 0.1 | 13 | 0.250 ± 0.130 | [0.171, 0.328] | 0.600 ± 0.066 | [0.560, 0.640] | 4 | 5 | 0.11 | 0.06 |
| A | linear | 1 | 0.1 | 13 | 0.488 ± 0.083 | [0.438, 0.538] | 0.834 ± 0.040 | [0.810, 0.859] | 4 | 5 | 0.13 | 0.08 |
| A | linear | 3 | 0.1 | 13 | 0.117 ± 0.164 | [0.018, 0.216] | 0.567 ± 0.096 | [0.509, 0.625] | 4 | 5 | 0.10 | 0.05 |
| A | linear | 10 | 0.1 | 13 | 0.317 ± 0.136 | [0.235, 0.399] | 0.728 ± 0.092 | [0.672, 0.784] | 4 | 5 | 0.09 | 0.07 |
| A | linear | 30 | 0.1 | 13 | 0.388 ± 0.125 | [0.313, 0.464] | 0.760 ± 0.080 | [0.712, 0.809] | 4 | 5 | 0.09 | 0.08 |
| A | linear | 100 | 0.1 | 13 | 0.152 ± 0.165 | [0.053, 0.252] | 0.540 ± 0.070 | [0.498, 0.583] | 4 | 5 | 0.09 | 0.04 |
| A | linear | 0.1 | 1 | 13 | 0.159 ± 0.179 | [0.051, 0.267] | 0.570 ± 0.070 | [0.527, 0.612] | 4 | 5 | 0.10 | 0.05 |
| A | linear | 0.3 | 1 | 13 | 0.204 ± 0.166 | [0.103, 0.304] | 0.592 ± 0.069 | [0.551, 0.634] | 4 | 5 | 0.10 | 0.07 |
| A | linear | 1 | 1 | 13 | 0.227 ± 0.122 | [0.153, 0.301] | 0.671 ± 0.119 | [0.599, 0.743] | 4 | 5 | 0.10 | 0.08 |
| A | linear | 3 | 1 | 13 | 0.158 ± 0.115 | [0.088, 0.227] | 0.552 ± 0.078 | [0.505, 0.599] | 4 | 5 | 0.10 | 0.04 |
| A | linear | 10 | 1 | 13 | 0.304 ± 0.147 | [0.215, 0.393] | 0.694 ± 0.088 | [0.641, 0.747] | 4 | 5 | 0.10 | 0.07 |
| A | linear | 30 | 1 | 13 | 0.296 ± 0.184 | [0.185, 0.407] | 0.672 ± 0.120 | [0.599, 0.744] | 4 | 5 | 0.09 | 0.07 |
| A | linear | 100 | 1 | 13 | 0.260 ± 0.154 | [0.167, 0.352] | 0.615 ± 0.103 | [0.553, 0.678] | 4 | 5 | 0.10 | 0.07 |
| A | linear | 0.1 | 10 | 13 | 0.243 ± 0.151 | [0.152, 0.335] | 0.620 ± 0.086 | [0.568, 0.671] | 4 | 5 | 0.10 | 0.07 |
| A | linear | 0.3 | 10 | 13 | 0.262 ± 0.115 | [0.193, 0.331] | 0.630 ± 0.084 | [0.579, 0.681] | 4 | 5 | 0.11 | 0.06 |
| A | linear | 1 | 10 | 13 | 0.221 ± 0.144 | [0.134, 0.308] | 0.629 ± 0.064 | [0.590, 0.667] | 4 | 5 | 0.10 | 0.06 |
| A | linear | 3 | 10 | 13 | 0.177 ± 0.188 | [0.064, 0.291] | 0.604 ± 0.095 | [0.546, 0.661] | 4 | 5 | 0.09 | 0.06 |
| A | linear | 10 | 10 | 13 | 0.184 ± 0.140 | [0.100, 0.269] | 0.611 ± 0.103 | [0.548, 0.673] | 4 | 5 | 0.10 | 0.06 |
| A | linear | 30 | 10 | 13 | 0.177 ± 0.155 | [0.083, 0.271] | 0.563 ± 0.090 | [0.509, 0.617] | 4 | 5 | 0.10 | 0.06 |
| A | linear | 100 | 10 | 13 | 0.196 ± 0.168 | [0.094, 0.297] | 0.596 ± 0.091 | [0.541, 0.651] | 4 | 5 | 0.09 | 0.05 |
| A | mlp | 0.1 | 0 | 13 | 0.742 ± 0.091 | [0.687, 0.798] | 0.936 ± 0.049 | [0.906, 0.965] | 4 | 49 | 0.02 | 0.13 |
| A | mlp | 0.3 | 0 | 13 | 0.715 ± 0.098 | [0.655, 0.774] | 0.915 ± 0.056 | [0.882, 0.949] | 4 | 49 | 0.02 | 0.11 |
| A | mlp | 1 | 0 | 13 | 0.597 ± 0.120 | [0.525, 0.669] | 0.877 ± 0.045 | [0.849, 0.904] | 4 | 49 | 0.02 | 0.13 |
| A | mlp | 3 | 0 | 13 | 0.485 ± 0.124 | [0.410, 0.560] | 0.805 ± 0.067 | [0.764, 0.846] | 4 | 49 | 0.02 | 0.12 |
| A | mlp | 10 | 0 | 13 | 0.653 ± 0.072 | [0.609, 0.697] | 0.908 ± 0.040 | [0.884, 0.932] | 4 | 49 | 0.02 | 0.13 |
| A | mlp | 30 | 0 | 13 | 0.691 ± 0.090 | [0.637, 0.746] | 0.922 ± 0.043 | [0.896, 0.948] | 4 | 49 | 0.02 | 0.13 |
| A | mlp | 100 | 0 | 13 | 0.528 ± 0.123 | [0.454, 0.603] | 0.819 ± 0.067 | [0.778, 0.860] | 4 | 49 | 0.02 | 0.15 |
| A | mlp | 0.1 | 0.1 | 13 | 0.751 ± 0.101 | [0.690, 0.812] | 0.934 ± 0.052 | [0.903, 0.966] | 4 | 49 | 0.10 | 0.11 |
| A | mlp | 0.3 | 0.1 | 13 | 0.675 ± 0.111 | [0.608, 0.742] | 0.918 ± 0.055 | [0.884, 0.951] | 4 | 49 | 0.11 | 0.11 |
| A | mlp | 1 | 0.1 | 13 | 0.584 ± 0.109 | [0.518, 0.650] | 0.862 ± 0.055 | [0.829, 0.895] | 4 | 49 | 0.13 | 0.14 |
| A | mlp | 3 | 0.1 | 13 | 0.480 ± 0.184 | [0.368, 0.591] | 0.781 ± 0.081 | [0.733, 0.830] | 4 | 49 | 0.10 | 0.11 |
| A | mlp | 10 | 0.1 | 13 | 0.669 ± 0.129 | [0.591, 0.746] | 0.917 ± 0.035 | [0.895, 0.938] | 4 | 49 | 0.09 | 0.14 |
| A | mlp | 30 | 0.1 | 13 | 0.698 ± 0.096 | [0.640, 0.755] | 0.927 ± 0.040 | [0.903, 0.951] | 4 | 49 | 0.09 | 0.14 |
| A | mlp | 100 | 0.1 | 13 | 0.531 ± 0.123 | [0.456, 0.605] | 0.823 ± 0.062 | [0.785, 0.860] | 4 | 49 | 0.09 | 0.15 |
| A | mlp | 0.1 | 1 | 13 | 0.765 ± 0.059 | [0.729, 0.800] | 0.936 ± 0.045 | [0.908, 0.963] | 4 | 49 | 0.10 | 0.10 |
| A | mlp | 0.3 | 1 | 13 | 0.705 ± 0.111 | [0.638, 0.772] | 0.919 ± 0.054 | [0.886, 0.952] | 4 | 49 | 0.10 | 0.13 |
| A | mlp | 1 | 1 | 13 | 0.491 ± 0.141 | [0.406, 0.576] | 0.828 ± 0.070 | [0.786, 0.871] | 4 | 49 | 0.10 | 0.10 |
| A | mlp | 3 | 1 | 13 | 0.491 ± 0.141 | [0.406, 0.576] | 0.817 ± 0.070 | [0.775, 0.859] | 4 | 49 | 0.10 | 0.13 |
| A | mlp | 10 | 1 | 13 | 0.720 ± 0.102 | [0.659, 0.782] | 0.932 ± 0.037 | [0.910, 0.955] | 4 | 49 | 0.10 | 0.13 |
| A | mlp | 30 | 1 | 13 | 0.718 ± 0.111 | [0.651, 0.785] | 0.927 ± 0.046 | [0.899, 0.954] | 4 | 49 | 0.09 | 0.13 |
| A | mlp | 100 | 1 | 13 | 0.629 ± 0.111 | [0.562, 0.696] | 0.906 ± 0.051 | [0.875, 0.937] | 4 | 49 | 0.10 | 0.15 |
| A | mlp | 0.1 | 10 | 13 | 0.699 ± 0.098 | [0.640, 0.758] | 0.915 ± 0.045 | [0.888, 0.942] | 4 | 49 | 0.10 | 0.13 |
| A | mlp | 0.3 | 10 | 13 | 0.676 ± 0.090 | [0.622, 0.731] | 0.914 ± 0.042 | [0.889, 0.939] | 4 | 49 | 0.11 | 0.13 |
| A | mlp | 1 | 10 | 13 | 0.637 ± 0.090 | [0.582, 0.691] | 0.906 ± 0.041 | [0.881, 0.930] | 4 | 49 | 0.10 | 0.11 |
| A | mlp | 3 | 10 | 13 | 0.551 ± 0.103 | [0.488, 0.613] | 0.853 ± 0.054 | [0.820, 0.885] | 4 | 49 | 0.09 | 0.10 |
| A | mlp | 10 | 10 | 13 | 0.646 ± 0.065 | [0.607, 0.685] | 0.900 ± 0.044 | [0.874, 0.926] | 4 | 49 | 0.10 | 0.11 |
| A | mlp | 30 | 10 | 13 | 0.707 ± 0.081 | [0.658, 0.755] | 0.919 ± 0.047 | [0.891, 0.947] | 4 | 49 | 0.10 | 0.12 |
| A | mlp | 100 | 10 | 13 | 0.686 ± 0.082 | [0.636, 0.736] | 0.916 ± 0.046 | [0.888, 0.943] | 4 | 49 | 0.09 | 0.12 |
| B | linear | – | 0 | 13 | 0.135 ± 0.171 | [0.032, 0.239] | 0.536 ± 0.082 | [0.486, 0.585] | 4 | 5 | 0.02 | 0.05 |
| B | linear | – | 0.1 | 13 | 0.128 ± 0.197 | [0.009, 0.248] | 0.543 ± 0.078 | [0.496, 0.589] | 4 | 5 | 0.10 | 0.05 |
| B | linear | – | 1 | 13 | 0.230 ± 0.153 | [0.138, 0.323] | 0.574 ± 0.083 | [0.524, 0.624] | 4 | 5 | 0.10 | 0.07 |
| B | linear | – | 10 | 13 | 0.265 ± 0.168 | [0.163, 0.366] | 0.622 ± 0.079 | [0.575, 0.670] | 4 | 5 | 0.10 | 0.05 |
| B | mlp | – | 0 | 13 | 0.746 ± 0.097 | [0.687, 0.804] | 0.937 ± 0.041 | [0.913, 0.962] | 4 | 49 | 0.02 | 0.11 |
| B | mlp | – | 0.1 | 13 | 0.769 ± 0.074 | [0.725, 0.814] | 0.934 ± 0.047 | [0.906, 0.962] | 4 | 49 | 0.10 | 0.14 |
| B | mlp | – | 1 | 13 | 0.756 ± 0.072 | [0.713, 0.800] | 0.939 ± 0.047 | [0.911, 0.968] | 4 | 49 | 0.10 | 0.10 |
| B | mlp | – | 10 | 13 | 0.702 ± 0.095 | [0.644, 0.759] | 0.924 ± 0.042 | [0.898, 0.949] | 4 | 49 | 0.10 | 0.12 |
| C | linear | – | – | 13 | 0.215 ± 0.166 | [0.115, 0.316] | 0.573 ± 0.106 | [0.508, 0.637] | 4 | 5 | 0.00 | 0.05 |
| C | mlp | – | – | 13 | 0.791 ± 0.097 | [0.733, 0.850] | 0.966 ± 0.034 | [0.945, 0.986] | 4 | 49 | 0.00 | 0.14 |
| D | linear | – | – | 13 | 0.061 ± 0.117 | [-0.009, 0.132] | 0.565 ± 0.078 | [0.518, 0.612] | 4 | 5 | 0.01 | 0.05 |
| D | mlp | – | – | 13 | 0.006 ± 0.144 | [-0.081, 0.093] | 0.544 ± 0.080 | [0.496, 0.593] | 4 | 49 | 0.01 | 0.08 |
| D-hybrid | hybrid | – | – | 13 | 0.378 ± 0.114 | [0.309, 0.447] | 0.787 ± 0.077 | [0.740, 0.833] | 4 | 29 | 0.00 | 2.20 |

#### A against B

Mean ΔMCC = A − B over the folds (Holm-adjusted p; ✓: at most α).

Head `linear`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.052 (1.0000) | -0.020 (1.0000) | -0.072 (1.0000) | -0.021 (1.0000) |
| 0.3 | +0.139 (1.0000) | +0.121 (1.0000) | -0.027 (1.0000) | -0.002 (1.0000) |
| 1 | +0.376 (0.0464 ✓) | +0.360 (0.0542) | -0.003 (1.0000) | -0.043 (1.0000) |
| 3 | -0.023 (1.0000) | -0.011 (1.0000) | -0.073 (1.0000) | -0.087 (1.0000) |
| 10 | +0.202 (0.8503) | +0.189 (1.0000) | +0.074 (1.0000) | -0.080 (1.0000) |
| 30 | +0.287 (0.0754) | +0.260 (0.2344) | +0.066 (1.0000) | -0.088 (1.0000) |
| 100 | -0.041 (1.0000) | +0.024 (1.0000) | +0.029 (1.0000) | -0.069 (1.0000) |

Head `mlp`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.003 (1.0000) | -0.018 (1.0000) | +0.008 (1.0000) | -0.002 (1.0000) |
| 0.3 | -0.031 (1.0000) | -0.094 (0.5215) | -0.051 (1.0000) | -0.025 (1.0000) |
| 1 | -0.149 (0.2344) | -0.185 (0.1221) | -0.265 (0.0464 ✓) | -0.065 (1.0000) |
| 3 | -0.261 (0.0464 ✓) | -0.290 (0.0464 ✓) | -0.265 (0.0464 ✓) | -0.151 (0.3213) |
| 10 | -0.093 (1.0000) | -0.101 (1.0000) | -0.036 (1.0000) | -0.056 (1.0000) |
| 30 | -0.054 (1.0000) | -0.072 (1.0000) | -0.038 (1.0000) | +0.005 (1.0000) |
| 100 | -0.217 (0.0464 ✓) | -0.239 (0.0464 ✓) | -0.127 (0.0986) | -0.015 (1.0000) |

#### Every arm against the classical baselines

Mean ΔMCC = arm − baseline (Holm-adjusted p; ✓: at most α).

Arm A, head `linear`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.028 (1.0000) | -0.107 (1.0000) | -0.057 (1.0000) | +0.028 (1.0000) |
| 0.3 | +0.059 (1.0000) | +0.034 (1.0000) | -0.012 (1.0000) | +0.047 (1.0000) |
| 1 | +0.296 (0.0542) | +0.273 (0.0464 ✓) | +0.012 (1.0000) | +0.006 (1.0000) |
| 3 | -0.103 (1.0000) | -0.098 (1.0000) | -0.058 (1.0000) | -0.038 (1.0000) |
| 10 | +0.122 (1.0000) | +0.102 (1.0000) | +0.089 (1.0000) | -0.031 (1.0000) |
| 30 | +0.207 (0.6768) | +0.173 (0.4221) | +0.081 (1.0000) | -0.038 (1.0000) |
| 100 | -0.121 (1.0000) | -0.063 (1.0000) | +0.044 (1.0000) | -0.020 (1.0000) |

Arm A, head `linear`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.605 (0.0464 ✓) | -0.683 (0.0464 ✓) | -0.633 (0.0464 ✓) | -0.548 (0.0464 ✓) |
| 0.3 | -0.517 (0.0464 ✓) | -0.542 (0.0464 ✓) | -0.588 (0.0464 ✓) | -0.529 (0.0464 ✓) |
| 1 | -0.280 (0.0542) | -0.303 (0.0464 ✓) | -0.564 (0.0464 ✓) | -0.570 (0.0464 ✓) |
| 3 | -0.679 (0.0464 ✓) | -0.674 (0.0464 ✓) | -0.634 (0.0464 ✓) | -0.614 (0.0464 ✓) |
| 10 | -0.454 (0.0464 ✓) | -0.474 (0.0464 ✓) | -0.487 (0.0464 ✓) | -0.607 (0.0464 ✓) |
| 30 | -0.369 (0.0464 ✓) | -0.403 (0.0464 ✓) | -0.495 (0.0464 ✓) | -0.615 (0.0464 ✓) |
| 100 | -0.697 (0.0464 ✓) | -0.639 (0.0464 ✓) | -0.532 (0.0464 ✓) | -0.596 (0.0464 ✓) |

Arm A, head `mlp`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.527 (0.0464 ✓) | +0.536 (0.0464 ✓) | +0.549 (0.0464 ✓) | +0.484 (0.0464 ✓) |
| 0.3 | +0.499 (0.0464 ✓) | +0.460 (0.0464 ✓) | +0.490 (0.0464 ✓) | +0.461 (0.0464 ✓) |
| 1 | +0.382 (0.0464 ✓) | +0.369 (0.0464 ✓) | +0.276 (0.3213) | +0.421 (0.0464 ✓) |
| 3 | +0.269 (0.1221) | +0.264 (0.4395) | +0.275 (0.0754) | +0.335 (0.0542) |
| 10 | +0.437 (0.0464 ✓) | +0.453 (0.0464 ✓) | +0.505 (0.0464 ✓) | +0.431 (0.0464 ✓) |
| 30 | +0.476 (0.0464 ✓) | +0.482 (0.0464 ✓) | +0.503 (0.0464 ✓) | +0.491 (0.0464 ✓) |
| 100 | +0.313 (0.0464 ✓) | +0.315 (0.0464 ✓) | +0.414 (0.0542) | +0.471 (0.0464 ✓) |

Arm A, head `mlp`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.049 (0.6768) | -0.040 (1.0000) | -0.027 (1.0000) | -0.092 (0.0464 ✓) |
| 0.3 | -0.077 (1.0000) | -0.116 (0.1221) | -0.087 (0.5371) | -0.115 (0.0464 ✓) |
| 1 | -0.194 (0.0464 ✓) | -0.207 (0.0464 ✓) | -0.300 (0.0464 ✓) | -0.155 (0.0542) |
| 3 | -0.307 (0.0464 ✓) | -0.312 (0.0464 ✓) | -0.301 (0.0464 ✓) | -0.241 (0.0542) |
| 10 | -0.139 (0.5371) | -0.123 (0.6768) | -0.071 (1.0000) | -0.145 (0.1658) |
| 30 | -0.100 (1.0000) | -0.094 (1.0000) | -0.073 (1.0000) | -0.085 (0.5371) |
| 100 | -0.263 (0.0464 ✓) | -0.261 (0.0464 ✓) | -0.162 (0.0542) | -0.105 (0.3213) |

The other arms:

| arm | against C/linear | against C/mlp |
|---|---|---|
| B/linear γ/Ω=0 | -0.080 (1.0000) | -0.656 (0.0464 ✓) |
| B/linear γ/Ω=0.1 | -0.087 (1.0000) | -0.663 (0.0464 ✓) |
| B/linear γ/Ω=1 | +0.015 (1.0000) | -0.561 (0.0464 ✓) |
| B/linear γ/Ω=10 | +0.049 (1.0000) | -0.527 (0.0464 ✓) |
| B/mlp γ/Ω=0 | +0.530 (0.0464 ✓) | -0.046 (1.0000) |
| B/mlp γ/Ω=0.1 | +0.554 (0.0464 ✓) | -0.022 (1.0000) |
| B/mlp γ/Ω=1 | +0.541 (0.0464 ✓) | -0.035 (1.0000) |
| B/mlp γ/Ω=10 | +0.486 (0.0464 ✓) | -0.090 (0.0464 ✓) |
| D/linear | -0.154 (1.0000) | -0.730 (0.0464 ✓) |
| D/mlp | -0.210 (0.5371) | -0.786 (0.0464 ✓) |
| D-hybrid/hybrid | +0.163 (1.0000) | -0.413 (0.0464 ✓) |

### Iranian Churn

Folds per row: [13].  Holm family: 190 comparisons at α = 0.05, first threshold α/m = 2.63e-04.

#### Scores

| arm | head | V/Ω | γ/Ω | n | MCC mean ± sd | MCC 95 % CI | PR-AUC mean ± sd | PR-AUC 95 % CI | dim | params | feature s | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | linear | 0.1 | 0 | 13 | 0.563 ± 0.064 | [0.525, 0.602] | 0.561 ± 0.065 | [0.521, 0.600] | 4 | 5 | 0.14 | 0.34 |
| A | linear | 0.3 | 0 | 13 | 0.552 ± 0.070 | [0.509, 0.594] | 0.554 ± 0.061 | [0.518, 0.591] | 4 | 5 | 0.14 | 0.31 |
| A | linear | 1 | 0 | 13 | 0.377 ± 0.056 | [0.344, 0.411] | 0.472 ± 0.090 | [0.418, 0.526] | 4 | 5 | 0.14 | 0.27 |
| A | linear | 3 | 0 | 13 | 0.495 ± 0.039 | [0.471, 0.518] | 0.554 ± 0.075 | [0.509, 0.599] | 4 | 5 | 0.14 | 0.33 |
| A | linear | 10 | 0 | 13 | 0.508 ± 0.058 | [0.473, 0.542] | 0.581 ± 0.061 | [0.544, 0.618] | 4 | 5 | 0.14 | 0.33 |
| A | linear | 30 | 0 | 13 | 0.535 ± 0.066 | [0.495, 0.575] | 0.580 ± 0.059 | [0.544, 0.616] | 4 | 5 | 0.14 | 0.38 |
| A | linear | 100 | 0 | 13 | 0.473 ± 0.071 | [0.430, 0.515] | 0.452 ± 0.060 | [0.415, 0.488] | 4 | 5 | 0.13 | 0.41 |
| A | linear | 0.1 | 0.1 | 13 | 0.561 ± 0.061 | [0.524, 0.598] | 0.563 ± 0.063 | [0.525, 0.602] | 4 | 5 | 2.09 | 0.32 |
| A | linear | 0.3 | 0.1 | 13 | 0.546 ± 0.070 | [0.504, 0.589] | 0.557 ± 0.060 | [0.521, 0.593] | 4 | 5 | 2.08 | 0.40 |
| A | linear | 1 | 0.1 | 13 | 0.380 ± 0.062 | [0.343, 0.418] | 0.509 ± 0.043 | [0.483, 0.535] | 4 | 5 | 2.07 | 0.32 |
| A | linear | 3 | 0.1 | 13 | 0.502 ± 0.041 | [0.477, 0.527] | 0.555 ± 0.071 | [0.512, 0.597] | 4 | 5 | 2.08 | 0.30 |
| A | linear | 10 | 0.1 | 13 | 0.516 ± 0.060 | [0.480, 0.552] | 0.582 ± 0.063 | [0.544, 0.620] | 4 | 5 | 2.05 | 0.29 |
| A | linear | 30 | 0.1 | 13 | 0.544 ± 0.065 | [0.505, 0.583] | 0.581 ± 0.060 | [0.544, 0.617] | 4 | 5 | 2.05 | 0.32 |
| A | linear | 100 | 0.1 | 13 | 0.469 ± 0.061 | [0.433, 0.506] | 0.461 ± 0.062 | [0.423, 0.498] | 4 | 5 | 2.05 | 0.54 |
| A | linear | 0.1 | 1 | 13 | 0.553 ± 0.060 | [0.517, 0.589] | 0.567 ± 0.061 | [0.530, 0.604] | 4 | 5 | 2.09 | 0.40 |
| A | linear | 0.3 | 1 | 13 | 0.556 ± 0.063 | [0.518, 0.594] | 0.564 ± 0.060 | [0.528, 0.600] | 4 | 5 | 2.09 | 0.47 |
| A | linear | 1 | 1 | 13 | 0.486 ± 0.065 | [0.447, 0.525] | 0.542 ± 0.046 | [0.514, 0.570] | 4 | 5 | 2.07 | 0.41 |
| A | linear | 3 | 1 | 13 | 0.555 ± 0.057 | [0.521, 0.590] | 0.551 ± 0.063 | [0.513, 0.589] | 4 | 5 | 2.09 | 0.39 |
| A | linear | 10 | 1 | 13 | 0.561 ± 0.062 | [0.524, 0.599] | 0.584 ± 0.064 | [0.545, 0.622] | 4 | 5 | 2.06 | 0.37 |
| A | linear | 30 | 1 | 13 | 0.555 ± 0.062 | [0.518, 0.592] | 0.582 ± 0.060 | [0.546, 0.618] | 4 | 5 | 2.05 | 0.27 |
| A | linear | 100 | 1 | 13 | 0.423 ± 0.092 | [0.367, 0.478] | 0.536 ± 0.054 | [0.504, 0.568] | 4 | 5 | 2.08 | 0.48 |
| A | linear | 0.1 | 10 | 13 | 0.557 ± 0.063 | [0.519, 0.595] | 0.590 ± 0.059 | [0.554, 0.625] | 4 | 5 | 2.08 | 0.49 |
| A | linear | 0.3 | 10 | 13 | 0.556 ± 0.059 | [0.521, 0.591] | 0.590 ± 0.056 | [0.557, 0.624] | 4 | 5 | 2.10 | 0.46 |
| A | linear | 1 | 10 | 13 | 0.553 ± 0.067 | [0.513, 0.594] | 0.587 ± 0.052 | [0.556, 0.618] | 4 | 5 | 2.05 | 0.55 |
| A | linear | 3 | 10 | 13 | 0.521 ± 0.064 | [0.482, 0.560] | 0.561 ± 0.049 | [0.531, 0.590] | 4 | 5 | 2.06 | 0.45 |
| A | linear | 10 | 10 | 13 | 0.547 ± 0.070 | [0.504, 0.589] | 0.585 ± 0.052 | [0.554, 0.617] | 4 | 5 | 2.06 | 0.51 |
| A | linear | 30 | 10 | 13 | 0.558 ± 0.062 | [0.520, 0.595] | 0.592 ± 0.054 | [0.559, 0.624] | 4 | 5 | 2.04 | 0.45 |
| A | linear | 100 | 10 | 13 | 0.548 ± 0.065 | [0.509, 0.588] | 0.586 ± 0.052 | [0.555, 0.618] | 4 | 5 | 2.06 | 0.48 |
| A | mlp | 0.1 | 0 | 13 | 0.543 ± 0.058 | [0.508, 0.578] | 0.637 ± 0.060 | [0.601, 0.673] | 4 | 49 | 0.14 | 0.78 |
| A | mlp | 0.3 | 0 | 13 | 0.532 ± 0.056 | [0.498, 0.566] | 0.618 ± 0.071 | [0.575, 0.660] | 4 | 49 | 0.14 | 0.66 |
| A | mlp | 1 | 0 | 13 | 0.515 ± 0.035 | [0.494, 0.537] | 0.619 ± 0.061 | [0.582, 0.656] | 4 | 49 | 0.14 | 1.08 |
| A | mlp | 3 | 0 | 13 | 0.574 ± 0.049 | [0.544, 0.603] | 0.654 ± 0.066 | [0.614, 0.694] | 4 | 49 | 0.14 | 0.83 |
| A | mlp | 10 | 0 | 13 | 0.566 ± 0.045 | [0.539, 0.593] | 0.665 ± 0.063 | [0.627, 0.704] | 4 | 49 | 0.14 | 0.73 |
| A | mlp | 30 | 0 | 13 | 0.598 ± 0.080 | [0.549, 0.646] | 0.684 ± 0.083 | [0.634, 0.734] | 4 | 49 | 0.14 | 1.00 |
| A | mlp | 100 | 0 | 13 | 0.536 ± 0.060 | [0.500, 0.573] | 0.608 ± 0.059 | [0.572, 0.644] | 4 | 49 | 0.13 | 0.67 |
| A | mlp | 0.1 | 0.1 | 13 | 0.543 ± 0.053 | [0.511, 0.574] | 0.613 ± 0.068 | [0.572, 0.654] | 4 | 49 | 2.09 | 0.62 |
| A | mlp | 0.3 | 0.1 | 13 | 0.536 ± 0.055 | [0.502, 0.569] | 0.620 ± 0.074 | [0.575, 0.665] | 4 | 49 | 2.08 | 0.81 |
| A | mlp | 1 | 0.1 | 13 | 0.502 ± 0.053 | [0.470, 0.534] | 0.616 ± 0.063 | [0.577, 0.654] | 4 | 49 | 2.07 | 0.99 |
| A | mlp | 3 | 0.1 | 13 | 0.579 ± 0.058 | [0.544, 0.613] | 0.671 ± 0.065 | [0.631, 0.710] | 4 | 49 | 2.08 | 0.68 |
| A | mlp | 10 | 0.1 | 13 | 0.550 ± 0.051 | [0.519, 0.581] | 0.647 ± 0.057 | [0.613, 0.682] | 4 | 49 | 2.05 | 0.55 |
| A | mlp | 30 | 0.1 | 13 | 0.577 ± 0.064 | [0.539, 0.616] | 0.690 ± 0.066 | [0.650, 0.730] | 4 | 49 | 2.05 | 0.79 |
| A | mlp | 100 | 0.1 | 13 | 0.527 ± 0.065 | [0.487, 0.566] | 0.597 ± 0.076 | [0.551, 0.643] | 4 | 49 | 2.05 | 0.69 |
| A | mlp | 0.1 | 1 | 13 | 0.553 ± 0.053 | [0.522, 0.585] | 0.630 ± 0.056 | [0.596, 0.664] | 4 | 49 | 2.09 | 0.67 |
| A | mlp | 0.3 | 1 | 13 | 0.548 ± 0.069 | [0.506, 0.590] | 0.622 ± 0.063 | [0.584, 0.660] | 4 | 49 | 2.09 | 0.71 |
| A | mlp | 1 | 1 | 13 | 0.512 ± 0.066 | [0.472, 0.552] | 0.646 ± 0.042 | [0.621, 0.672] | 4 | 49 | 2.07 | 0.85 |
| A | mlp | 3 | 1 | 13 | 0.575 ± 0.039 | [0.551, 0.598] | 0.641 ± 0.029 | [0.623, 0.658] | 4 | 49 | 2.09 | 0.66 |
| A | mlp | 10 | 1 | 13 | 0.586 ± 0.053 | [0.554, 0.618] | 0.665 ± 0.069 | [0.623, 0.707] | 4 | 49 | 2.06 | 0.59 |
| A | mlp | 30 | 1 | 13 | 0.575 ± 0.051 | [0.544, 0.605] | 0.659 ± 0.072 | [0.616, 0.703] | 4 | 49 | 2.05 | 0.57 |
| A | mlp | 100 | 1 | 13 | 0.505 ± 0.054 | [0.472, 0.538] | 0.602 ± 0.062 | [0.564, 0.640] | 4 | 49 | 2.08 | 0.94 |
| A | mlp | 0.1 | 10 | 13 | 0.538 ± 0.058 | [0.503, 0.573] | 0.640 ± 0.072 | [0.596, 0.683] | 4 | 49 | 2.08 | 0.93 |
| A | mlp | 0.3 | 10 | 13 | 0.524 ± 0.066 | [0.484, 0.564] | 0.626 ± 0.066 | [0.586, 0.666] | 4 | 49 | 2.10 | 0.87 |
| A | mlp | 1 | 10 | 13 | 0.545 ± 0.063 | [0.507, 0.583] | 0.629 ± 0.056 | [0.595, 0.663] | 4 | 49 | 2.05 | 0.79 |
| A | mlp | 3 | 10 | 13 | 0.509 ± 0.065 | [0.470, 0.548] | 0.651 ± 0.059 | [0.615, 0.687] | 4 | 49 | 2.06 | 0.77 |
| A | mlp | 10 | 10 | 13 | 0.544 ± 0.064 | [0.505, 0.582] | 0.627 ± 0.054 | [0.594, 0.659] | 4 | 49 | 2.06 | 0.55 |
| A | mlp | 30 | 10 | 13 | 0.543 ± 0.046 | [0.515, 0.570] | 0.650 ± 0.065 | [0.610, 0.689] | 4 | 49 | 2.04 | 0.86 |
| A | mlp | 100 | 10 | 13 | 0.531 ± 0.043 | [0.505, 0.557] | 0.615 ± 0.055 | [0.581, 0.648] | 4 | 49 | 2.06 | 0.84 |
| B | linear | – | 0 | 13 | 0.558 ± 0.061 | [0.521, 0.595] | 0.566 ± 0.067 | [0.526, 0.606] | 4 | 5 | 0.14 | 0.41 |
| B | linear | – | 0.1 | 13 | 0.559 ± 0.064 | [0.520, 0.597] | 0.565 ± 0.062 | [0.528, 0.603] | 4 | 5 | 2.08 | 0.41 |
| B | linear | – | 1 | 13 | 0.561 ± 0.057 | [0.526, 0.595] | 0.572 ± 0.061 | [0.534, 0.609] | 4 | 5 | 2.10 | 0.40 |
| B | linear | – | 10 | 13 | 0.554 ± 0.064 | [0.516, 0.593] | 0.590 ± 0.059 | [0.555, 0.626] | 4 | 5 | 2.09 | 0.50 |
| B | mlp | – | 0 | 13 | 0.568 ± 0.051 | [0.538, 0.599] | 0.657 ± 0.092 | [0.601, 0.713] | 4 | 49 | 0.14 | 0.83 |
| B | mlp | – | 0.1 | 13 | 0.530 ± 0.036 | [0.508, 0.551] | 0.638 ± 0.074 | [0.594, 0.683] | 4 | 49 | 2.08 | 0.73 |
| B | mlp | – | 1 | 13 | 0.533 ± 0.056 | [0.499, 0.567] | 0.634 ± 0.074 | [0.590, 0.679] | 4 | 49 | 2.10 | 0.61 |
| B | mlp | – | 10 | 13 | 0.547 ± 0.063 | [0.509, 0.585] | 0.628 ± 0.073 | [0.583, 0.672] | 4 | 49 | 2.09 | 0.80 |
| C | linear | – | – | 13 | 0.574 ± 0.058 | [0.538, 0.609] | 0.631 ± 0.060 | [0.595, 0.667] | 4 | 5 | 0.00 | 0.35 |
| C | mlp | – | – | 13 | 0.632 ± 0.049 | [0.602, 0.661] | 0.762 ± 0.069 | [0.720, 0.803] | 4 | 49 | 0.00 | 0.87 |
| D | linear | – | – | 13 | 0.250 ± 0.105 | [0.186, 0.313] | 0.361 ± 0.057 | [0.326, 0.395] | 4 | 5 | 0.03 | 0.37 |
| D | mlp | – | – | 13 | 0.339 ± 0.079 | [0.291, 0.386] | 0.442 ± 0.073 | [0.397, 0.486] | 4 | 49 | 0.03 | 0.90 |
| D-hybrid | hybrid | – | – | 13 | 0.493 ± 0.065 | [0.454, 0.532] | 0.571 ± 0.052 | [0.540, 0.603] | 4 | 29 | 0.00 | 19.31 |

#### A against B

Mean ΔMCC = A − B over the folds (Holm-adjusted p; ✓: at most α).

Head `linear`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.006 (1.0000) | +0.002 (1.0000) | -0.008 (1.0000) | +0.002 (1.0000) |
| 0.3 | -0.006 (1.0000) | -0.012 (1.0000) | -0.005 (1.0000) | +0.001 (1.0000) |
| 1 | -0.181 (0.0464 ✓) | -0.178 (0.0781) | -0.074 (0.1453) | -0.001 (1.0000) |
| 3 | -0.063 (0.6226) | -0.057 (1.0000) | -0.005 (1.0000) | -0.034 (0.1897) |
| 10 | -0.050 (0.8137) | -0.042 (1.0000) | +0.001 (1.0000) | -0.008 (1.0000) |
| 30 | -0.023 (1.0000) | -0.015 (1.0000) | -0.006 (1.0000) | +0.003 (1.0000) |
| 100 | -0.085 (0.0967) | -0.089 (0.0781) | -0.138 (0.0781) | -0.006 (1.0000) |

Head `mlp`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.025 (1.0000) | +0.013 (1.0000) | +0.020 (1.0000) | -0.009 (1.0000) |
| 0.3 | -0.037 (1.0000) | +0.006 (1.0000) | +0.015 (1.0000) | -0.023 (1.0000) |
| 1 | -0.053 (0.2686) | -0.028 (1.0000) | -0.021 (1.0000) | -0.002 (1.0000) |
| 3 | +0.006 (1.0000) | +0.049 (0.8137) | +0.042 (1.0000) | -0.037 (1.0000) |
| 10 | -0.002 (1.0000) | +0.020 (1.0000) | +0.053 (0.0464 ✓) | -0.003 (1.0000) |
| 30 | +0.029 (1.0000) | +0.048 (1.0000) | +0.042 (1.0000) | -0.004 (1.0000) |
| 100 | -0.032 (1.0000) | -0.003 (1.0000) | -0.028 (1.0000) | -0.016 (1.0000) |

#### Every arm against the classical baselines

Mean ΔMCC = arm − baseline (Holm-adjusted p; ✓: at most α).

Arm A, head `linear`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.010 (1.0000) | -0.013 (1.0000) | -0.021 (1.0000) | -0.017 (1.0000) |
| 0.3 | -0.022 (1.0000) | -0.027 (0.8137) | -0.018 (1.0000) | -0.018 (1.0000) |
| 1 | -0.196 (0.0464 ✓) | -0.194 (0.0464 ✓) | -0.087 (0.0464 ✓) | -0.020 (1.0000) |
| 3 | -0.079 (0.0781) | -0.072 (0.1453) | -0.018 (1.0000) | -0.053 (0.0781) |
| 10 | -0.066 (0.1453) | -0.057 (0.0464 ✓) | -0.012 (1.0000) | -0.027 (1.0000) |
| 30 | -0.039 (0.3623) | -0.030 (1.0000) | -0.019 (1.0000) | -0.016 (1.0000) |
| 100 | -0.101 (0.0464 ✓) | -0.105 (0.0464 ✓) | -0.151 (0.0967) | -0.025 (1.0000) |

Arm A, head `linear`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.068 (0.0967) | -0.071 (0.0967) | -0.079 (0.0464 ✓) | -0.075 (0.1453) |
| 0.3 | -0.080 (0.0781) | -0.085 (0.0781) | -0.076 (0.0781) | -0.076 (0.0781) |
| 1 | -0.254 (0.0464 ✓) | -0.251 (0.0464 ✓) | -0.145 (0.0464 ✓) | -0.078 (0.0781) |
| 3 | -0.137 (0.0464 ✓) | -0.130 (0.0464 ✓) | -0.076 (0.1453) | -0.111 (0.0781) |
| 10 | -0.124 (0.0781) | -0.115 (0.0781) | -0.070 (0.0967) | -0.085 (0.0967) |
| 30 | -0.097 (0.0781) | -0.087 (0.0781) | -0.077 (0.1453) | -0.074 (0.0781) |
| 100 | -0.159 (0.0464 ✓) | -0.162 (0.0464 ✓) | -0.209 (0.0464 ✓) | -0.083 (0.0781) |

Arm A, head `mlp`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.031 (1.0000) | -0.031 (1.0000) | -0.020 (1.0000) | -0.036 (1.0000) |
| 0.3 | -0.042 (1.0000) | -0.038 (1.0000) | -0.025 (1.0000) | -0.050 (1.0000) |
| 1 | -0.058 (1.0000) | -0.072 (1.0000) | -0.062 (1.0000) | -0.029 (1.0000) |
| 3 | +0.000 (1.0000) | +0.005 (1.0000) | +0.001 (1.0000) | -0.064 (1.0000) |
| 10 | -0.008 (1.0000) | -0.024 (1.0000) | +0.012 (1.0000) | -0.030 (1.0000) |
| 30 | +0.024 (1.0000) | +0.004 (1.0000) | +0.001 (1.0000) | -0.031 (1.0000) |
| 100 | -0.037 (1.0000) | -0.047 (1.0000) | -0.069 (0.1453) | -0.043 (1.0000) |

Arm A, head `mlp`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.089 (0.0781) | -0.089 (0.0781) | -0.078 (0.1453) | -0.093 (0.0464 ✓) |
| 0.3 | -0.100 (0.0464 ✓) | -0.096 (0.0781) | -0.083 (0.0967) | -0.108 (0.0967) |
| 1 | -0.116 (0.0464 ✓) | -0.130 (0.0781) | -0.120 (0.0781) | -0.086 (0.0967) |
| 3 | -0.058 (0.2686) | -0.053 (0.2686) | -0.057 (0.3623) | -0.122 (0.0967) |
| 10 | -0.065 (0.3623) | -0.082 (0.2686) | -0.046 (1.0000) | -0.088 (0.0464 ✓) |
| 30 | -0.034 (1.0000) | -0.054 (1.0000) | -0.057 (1.0000) | -0.089 (0.0464 ✓) |
| 100 | -0.095 (0.0781) | -0.105 (0.0464 ✓) | -0.126 (0.0464 ✓) | -0.101 (0.0464 ✓) |

The other arms:

| arm | against C/linear | against C/mlp |
|---|---|---|
| B/linear γ/Ω=0 | -0.016 (1.0000) | -0.074 (0.0781) |
| B/linear γ/Ω=0.1 | -0.015 (1.0000) | -0.073 (0.0781) |
| B/linear γ/Ω=1 | -0.013 (1.0000) | -0.071 (0.0781) |
| B/linear γ/Ω=10 | -0.019 (1.0000) | -0.077 (0.0967) |
| B/mlp γ/Ω=0 | -0.005 (1.0000) | -0.063 (0.4778) |
| B/mlp γ/Ω=0.1 | -0.044 (1.0000) | -0.102 (0.0781) |
| B/mlp γ/Ω=1 | -0.041 (1.0000) | -0.099 (0.0967) |
| B/mlp γ/Ω=10 | -0.027 (1.0000) | -0.085 (0.0967) |
| D/linear | -0.324 (0.0464 ✓) | -0.382 (0.0464 ✓) |
| D/mlp | -0.235 (0.0464 ✓) | -0.293 (0.0464 ✓) |
| D-hybrid/hybrid | -0.081 (0.0781) | -0.138 (0.0464 ✓) |

### What the results show

Everything below holds for four atoms, one chain, one pulse, exact features, the stated tuning
budget and these two datasets only.

1. **On Iranian Churn the interactions add nothing a head could use.** Against arm B, the linear
   head's mean MCC difference lies between −0.181 and +0.006 over the 28 grid points, with one
   Holm-significant cell, and that one is a loss (`V/Ω = 1`, `γ/Ω = 0`: −0.181). The MLP head's
   lies between −0.053 and +0.053, with one significant cell, a gain of +0.053 at `V/Ω = 10`,
   `γ/Ω = 1`, beside 27 that are not. On this dataset, then, interaction-induced mixing of the
   inputs gave one significant gain and one significant loss among 56 comparisons, and no gain
   above 0.053.
2. **On Iranian Churn no Rydberg row beats a classical baseline.** The inputs themselves with the
   MLP head (C/mlp, MCC 0.632) are the best row of the table; no row of A or B is significantly
   above C/linear (0.574) or C/mlp, while 23 of the 112 comparisons of an A row with a baseline
   are significant losses (linear head: 6 against C/linear, 9 against C/mlp; MLP head: 8 against
   C/mlp). Arm B stays within 0.044 of C/linear and is never significantly different from
   either baseline.
3. **On the positive control the interactions help a linear head at some interaction strengths
   and hurt an MLP head at others** (next section).
4. **Dephasing removes what the interactions changed.** At `γ/Ω = 10` no A − B difference is
   significant on either dataset or head, the largest being −0.151 (positive control, MLP,
   `V/Ω = 3`); the A − B differences of the positive control's linear head, up to +0.376 at
   `γ/Ω ≤ 0.1`, are at most +0.074 at `γ/Ω = 1`. Because the heads see standardised features, this
   is a statement about the features' information, not their scale (see *Limits*).
5. **The frozen gate-based map (D) is the weakest fixed map**: below C/mlp on both datasets, and
   below C/linear on Iranian Churn, every one of those Holm-significant. The trained
   `HybridBinaryClassifier` (D-hybrid) is below C/mlp on both datasets as well.
6. **Cost.** Arm A's features took 0.02 s per fold of the positive control at `γ = 0` and about
   0.1 s with dephasing (200 solver steps), about 0.14 s and 2.1 s on Iranian Churn: the feature
   map is not what limits this benchmark at four atoms. D-hybrid's final fit (2.20 s and 19.31 s)
   is the slowest training of every table.

## Mechanism

The separation measures of #500 on the features as the map returns them (before standardisation),
on the real rows of each fold's training part (training and validation rows, no SMOTE rows),
averaged over the 13 folds. On Iranian Churn, whose training parts hold 2,906 to 2,908 such rows,
they are computed on a class-stratified subsample of 2,000 (the runner's `max_separation_rows`,
seed 0); on the positive control on all 553 to 555. For arm B one value per
`γ/Ω`, repeated beside every `V/Ω`. `docs/rydberg-model.md` (*Separation measures*) says what each
supports: they describe what this readout distinguishes, and none predicts a particular head's
score.

Whether they track the scores is measured, not read off the table: for each head and measure,
Spearman's rank correlation across the 28 grid points between the measure's A − B difference and
the A − B difference of mean MCC, with a two-sided permutation p-value from 10,000 draws (the
smallest reportable value is 1/10,001 ≈ 1.00e-04). These 16 p-values are not corrected for
multiplicity, and the grid points are not independent (they share folds, and B is shared across
`V/Ω`), so they rank the measures rather than test them.

### Positive control

| V/Ω | γ/Ω | distance_ratio A | distance_ratio B | fisher_ratio A | fisher_ratio B | effective_rank A | effective_rank B | kernel_alignment A | kernel_alignment B |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 1.028 | 1.028 | 0.013 | 0.012 | 3.994 | 3.993 | 0.002 | 0.002 |
| 0.3 | 0 | 1.027 | 1.028 | 0.029 | 0.012 | 3.973 | 3.993 | 0.004 | 0.002 |
| 1 | 0 | 1.048 | 1.028 | 1.905 | 0.012 | 3.113 | 3.993 | 0.056 | 0.002 |
| 3 | 0 | 1.011 | 1.028 | 0.066 | 0.012 | 3.308 | 3.993 | 0.008 | 0.002 |
| 10 | 0 | 1.042 | 1.028 | 0.743 | 0.012 | 3.319 | 3.993 | 0.038 | 0.002 |
| 30 | 0 | 1.048 | 1.028 | 0.741 | 0.012 | 3.449 | 3.993 | 0.041 | 0.002 |
| 100 | 0 | 1.018 | 1.028 | 0.023 | 0.012 | 2.967 | 3.993 | 0.001 | 0.002 |
| 0.1 | 0.1 | 1.028 | 1.028 | 0.013 | 0.012 | 3.994 | 3.993 | 0.002 | 0.002 |
| 0.3 | 0.1 | 1.027 | 1.028 | 0.029 | 0.012 | 3.974 | 3.993 | 0.003 | 0.002 |
| 1 | 0.1 | 1.044 | 1.028 | 1.724 | 0.012 | 3.109 | 3.993 | 0.051 | 0.002 |
| 3 | 0.1 | 1.010 | 1.028 | 0.057 | 0.012 | 3.167 | 3.993 | 0.007 | 0.002 |
| 10 | 0.1 | 1.040 | 1.028 | 0.662 | 0.012 | 3.324 | 3.993 | 0.033 | 0.002 |
| 30 | 0.1 | 1.045 | 1.028 | 0.676 | 0.012 | 3.431 | 3.993 | 0.036 | 0.002 |
| 100 | 0.1 | 1.020 | 1.028 | 0.022 | 0.012 | 2.900 | 3.993 | 0.001 | 0.002 |
| 0.1 | 1 | 1.027 | 1.027 | 0.013 | 0.012 | 3.993 | 3.993 | 0.002 | 0.001 |
| 0.3 | 1 | 1.026 | 1.027 | 0.023 | 0.012 | 3.979 | 3.993 | 0.003 | 0.001 |
| 1 | 1 | 1.023 | 1.027 | 0.515 | 0.012 | 3.306 | 3.993 | 0.023 | 0.001 |
| 3 | 1 | 1.011 | 1.027 | 0.087 | 0.012 | 2.419 | 3.993 | 0.004 | 0.001 |
| 10 | 1 | 1.031 | 1.027 | 0.248 | 0.012 | 3.431 | 3.993 | 0.012 | 0.001 |
| 30 | 1 | 1.034 | 1.027 | 0.288 | 0.012 | 3.408 | 3.993 | 0.013 | 0.001 |
| 100 | 1 | 1.030 | 1.027 | 0.114 | 0.012 | 2.867 | 3.993 | 0.004 | 0.001 |
| 0.1 | 10 | 1.024 | 1.024 | 0.014 | 0.014 | 3.992 | 3.992 | 0.002 | 0.002 |
| 0.3 | 10 | 1.023 | 1.024 | 0.015 | 0.014 | 3.989 | 3.992 | 0.002 | 0.002 |
| 1 | 10 | 1.022 | 1.024 | 0.020 | 0.014 | 3.939 | 3.992 | 0.002 | 0.002 |
| 3 | 10 | 1.014 | 1.024 | 0.049 | 0.014 | 3.455 | 3.992 | 0.003 | 0.002 |
| 10 | 10 | 1.020 | 1.024 | 0.023 | 0.014 | 3.760 | 3.992 | 0.002 | 0.002 |
| 30 | 10 | 1.023 | 1.024 | 0.019 | 0.014 | 3.838 | 3.992 | 0.002 | 0.002 |
| 100 | 10 | 1.022 | 1.024 | 0.021 | 0.014 | 3.782 | 3.992 | 0.002 | 0.002 |

| head | measure | Spearman ρ (Δmeasure, ΔMCC) | n grid points | permutation p |
|---|---|---|---|---|
| linear | distance_ratio | 0.74 | 28 | 1.00e-04 |
| linear | fisher_ratio | 0.64 | 28 | 3.00e-04 |
| linear | effective_rank | -0.21 | 28 | 0.2782 |
| linear | kernel_alignment | 0.71 | 28 | 1.00e-04 |
| mlp | distance_ratio | 0.36 | 28 | 0.0590 |
| mlp | fisher_ratio | -0.57 | 28 | 0.0016 |
| mlp | effective_rank | 0.83 | 28 | 1.00e-04 |
| mlp | kernel_alignment | -0.37 | 28 | 0.0514 |

### Iranian Churn

| V/Ω | γ/Ω | distance_ratio A | distance_ratio B | fisher_ratio A | fisher_ratio B | effective_rank A | effective_rank B | kernel_alignment A | kernel_alignment B |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 1.028 | 1.034 | 2.163 | 2.190 | 3.912 | 3.903 | 0.125 | 0.130 |
| 0.3 | 0 | 1.013 | 1.034 | 2.100 | 2.190 | 3.902 | 3.903 | 0.118 | 0.130 |
| 1 | 0 | 0.954 | 1.034 | 1.387 | 2.190 | 3.140 | 3.903 | 0.091 | 0.130 |
| 3 | 0 | 1.091 | 1.034 | 2.143 | 2.190 | 3.150 | 3.903 | 0.151 | 0.130 |
| 10 | 0 | 1.096 | 1.034 | 2.795 | 2.190 | 3.308 | 3.903 | 0.172 | 0.130 |
| 30 | 0 | 1.082 | 1.034 | 2.617 | 2.190 | 3.418 | 3.903 | 0.175 | 0.130 |
| 100 | 0 | 1.057 | 1.034 | 1.484 | 2.190 | 3.153 | 3.903 | 0.106 | 0.130 |
| 0.1 | 0.1 | 1.027 | 1.033 | 2.166 | 2.191 | 3.913 | 3.905 | 0.125 | 0.129 |
| 0.3 | 0.1 | 1.013 | 1.033 | 2.104 | 2.191 | 3.903 | 3.905 | 0.118 | 0.129 |
| 1 | 0.1 | 0.956 | 1.033 | 1.449 | 2.191 | 3.138 | 3.905 | 0.093 | 0.129 |
| 3 | 0.1 | 1.100 | 1.033 | 2.175 | 2.191 | 3.025 | 3.905 | 0.154 | 0.129 |
| 10 | 0.1 | 1.093 | 1.033 | 2.764 | 2.191 | 3.294 | 3.905 | 0.171 | 0.129 |
| 30 | 0.1 | 1.079 | 1.033 | 2.587 | 2.191 | 3.389 | 3.905 | 0.174 | 0.129 |
| 100 | 0.1 | 1.053 | 1.033 | 1.427 | 2.191 | 3.060 | 3.905 | 0.110 | 0.129 |
| 0.1 | 1 | 1.020 | 1.025 | 2.176 | 2.195 | 3.917 | 3.914 | 0.124 | 0.127 |
| 0.3 | 1 | 1.010 | 1.025 | 2.131 | 2.195 | 3.907 | 3.914 | 0.119 | 0.127 |
| 1 | 1 | 0.972 | 1.025 | 1.801 | 2.195 | 3.310 | 3.914 | 0.107 | 0.127 |
| 3 | 1 | 1.118 | 1.025 | 2.319 | 2.195 | 2.435 | 3.914 | 0.159 | 0.127 |
| 10 | 1 | 1.069 | 1.025 | 2.461 | 2.195 | 3.308 | 3.914 | 0.164 | 0.127 |
| 30 | 1 | 1.063 | 1.025 | 2.355 | 2.195 | 3.304 | 3.914 | 0.164 | 0.127 |
| 100 | 1 | 1.041 | 1.025 | 1.493 | 2.195 | 2.798 | 3.914 | 0.135 | 0.127 |
| 0.1 | 10 | 1.001 | 1.001 | 2.147 | 2.152 | 3.922 | 3.923 | 0.116 | 0.116 |
| 0.3 | 10 | 1.000 | 1.001 | 2.137 | 2.152 | 3.917 | 3.923 | 0.116 | 0.116 |
| 1 | 10 | 0.997 | 1.001 | 2.093 | 2.152 | 3.867 | 3.923 | 0.116 | 0.116 |
| 3 | 10 | 0.997 | 1.001 | 1.933 | 2.152 | 3.406 | 3.923 | 0.115 | 0.116 |
| 10 | 10 | 1.018 | 1.001 | 2.110 | 2.152 | 3.700 | 3.923 | 0.135 | 0.116 |
| 30 | 10 | 1.021 | 1.001 | 2.140 | 2.152 | 3.783 | 3.923 | 0.137 | 0.116 |
| 100 | 10 | 1.019 | 1.001 | 2.093 | 2.152 | 3.747 | 3.923 | 0.135 | 0.116 |

| head | measure | Spearman ρ (Δmeasure, ΔMCC) | n grid points | permutation p |
|---|---|---|---|---|
| linear | distance_ratio | -0.02 | 28 | 0.9382 |
| linear | fisher_ratio | 0.48 | 28 | 0.0123 |
| linear | effective_rank | 0.70 | 28 | 1.00e-04 |
| linear | kernel_alignment | 0.22 | 28 | 0.2557 |
| mlp | distance_ratio | 0.59 | 28 | 1.00e-03 |
| mlp | fisher_ratio | 0.69 | 28 | 3.00e-04 |
| mlp | effective_rank | -0.07 | 28 | 0.7373 |
| mlp | kernel_alignment | 0.66 | 28 | 3.00e-04 |

### What the measures show

- **Positive control, linear head:** the gains of A over B go with gains in the class-separation
  measures: ρ = 0.74 for the distance ratio, 0.64 for the Fisher ratio and 0.71 for the kernel
  alignment (n = 28, permutation p ≤ 3.00e-04 each), but not with the effective rank (ρ = −0.21,
  p = 0.2782). The Fisher ratio of arm B stays at 0.012 to 0.014 at every rate, as the task's
  construction requires (no single input carries the label), while arm A's reaches 1.905 at
  `V/Ω = 1`, `γ/Ω = 0`, the cell where the linear head gains most, and falls to 0.066 at
  `V/Ω = 3`, where it gains nothing: the score follows the measure across the non-monotone
  interaction axis.
- **Positive control, MLP head:** its losses against B go with the effective rank (ρ = 0.83,
  p = 1.00e-04): where the interactions collapse the four features onto fewer directions (2.419
  at `V/Ω = 3`, `γ/Ω = 1`; 2.867 to 2.967 at `V/Ω = 100`, `γ/Ω ≤ 1`, against 3.993 for B), the MLP,
  which can form the needed products of B's four independent features itself, loses most. The
  Fisher ratio correlates negatively here (ρ = −0.57, p = 0.0016).
- **Iranian Churn:** the linear head's differences track the effective rank (ρ = 0.70,
  p = 1.00e-04) and the Fisher ratio (ρ = 0.48, p = 0.0123); the MLP head's track the Fisher ratio
  (ρ = 0.69), the kernel alignment (ρ = 0.66) and the distance ratio (ρ = 0.59), all with
  p ≤ 1.00e-03. The cell where A loses most for the linear head, `V/Ω = 1`, `γ/Ω = 0`, has A's
  lowest Fisher ratio (1.387 against B's 2.190) and a distance ratio below 1 (0.954).
- **Strong dephasing** brings A's measures back towards B's: at `γ/Ω = 10` the Fisher ratio of A
  on the positive control is 0.014 to 0.049 against B's 0.014, and the effective rank 3.455 to
  3.992 against 3.992.

So the measures do move with the scores in this run, but which measure moves with them depends on
the head and the dataset; none of them alone would have predicted the sign of A − B in advance.

## Positive control

The positive control asks whether this pipeline can show an interaction effect where one exists
by construction. Its label depends only on products of neighbouring latents, so B's features and
the inputs carry no class-mean difference, and a linear head on them should stay near its floor.
That holds for the **linear** head; an MLP can form products of its inputs itself, so its A − B is
no check of the pipeline (#501).

- **The effect is detected.** At `V/Ω = 1`, `γ/Ω = 0`, arm A with the linear head scores MCC
  0.511 [0.452, 0.570] against arm B's 0.135 [0.032, 0.239]: a difference of +0.376 that A wins in
  all 13 folds (exact p = 2.44e-04, the design's floor; Holm p = 0.0464 over 190 comparisons, the
  only Holm-significant A − B gain of the linear head). Against the linear baseline C/linear the
  same cell gains +0.296 (Holm p = 0.0542), and its neighbour at `γ/Ω = 0.1` (+0.273, Holm
  p = 0.0464) is the only row of A with the linear head that beats C/linear significantly. (Rows
  of A with the MLP head beat C/linear at most grid points, but so does B with the MLP head: that
  is the head, not the interactions.)
- **It depends on the interaction strength, not monotonically.** The linear head gains +0.287 at
  `V/Ω = 30` and +0.202 at 10 (not significant after Holm: 0.0754 and 0.8503), and nothing at
  `V/Ω` = 3 (−0.023) or 100 (−0.041), nor at 0.1 (+0.052), where the atoms barely interact. The
  separation measures of the previous section follow the same pattern.
- **It survives weak dephasing and not strong.** At `γ/Ω = 0.1` the gain at `V/Ω = 1` is +0.360
  (Holm p = 0.0542); at `γ/Ω = 1` and 10 no grid point gains more than +0.074.
- **Detecting it did not make A the best model.** C/mlp scores 0.791, and every A row with the
  linear head is significantly below it except `V/Ω = 1`, `γ/Ω = 0` (−0.280, Holm p = 0.0542). An
  MLP head on B's features, which are each a function of one input, scores 0.702 to 0.769, and A
  with the MLP head is significantly *below* B at 6 of 28 grid points (−0.217 to −0.290 at
  `V/Ω` = 1, 3 and 100), never above.

So the pipeline can show an interaction effect where the task is built for one, with the head
for which it is a check, at the significance level the design can reach. The Iranian Churn result
of the previous sections is therefore a null result about that dataset, not a failure of the
pipeline to detect anything.

## Limits

- **System size.** Four atoms, one geometry (an open chain), one pulse (`ΩT = π`). Arm A is a
  classical simulation of a 16-dimensional state; nothing here bears on computational advantage,
  and a simulated array of at most six atoms could not support a claim of one. Matched
  parameter counts (A, B, C and D heads have the same count by construction) do not make the
  comparison fair on their own: the fixed maps differ in what they compute before the first
  trainable parameter.
- **Two-level approximation.** Intermediate-state scattering, other Rydberg levels and atomic
  motion are left out (`docs/rydberg-model.md`).
- **Dephasing model.** Markovian, local `√γ n_i`, no Rydberg decay, no quasi-static Doppler
  noise. `γ/Ω` = 1 and 10 are stress tests far above a current device's indicative 0.01–0.04
  at this Ω; the 200-step solver error grows about as `γ²` (`1.0e-4` at `γ = 10 Ω`).
- **Exact against sampled readout.** Every feature is an exact expectation value. Because the
  heads see standardised features, a collapse of the features under strong dephasing is a
  change of scale they undo; with a finite number of shots it would not be. No `--shots` run
  is in this note.
- **Datasets.** One synthetic positive control and one UCI dataset, `iranian-churn`, loaded
  with its 300 exact duplicate rows, which can put the same record in a training and a test
  fold (the same for every arm, so paired differences are less affected than absolute
  scores; an option to drop them is #704). `taiwanese-bankruptcy` and `cervical-cancer-risk`
  were not run.
- **Statistics.** The fold intervals understate the uncertainty (*Statistics*), and the rank
  correlations of *Mechanism* are 16 uncorrected tests over 28 dependent grid points. Holm over
  190 comparisons controls the family-wise error, so a true but small interaction effect can go
  undetected: in this run every Holm-significant comparison (79 on the positive control, 30 on
  Iranian Churn) is one side winning all 13 folds, and the next possible p-value, `4.88e-4`
  (12 of 13 folds, the lost one smallest), was not significant after Holm anywhere (adjusted
  0.0542 at best).
- **Not tried:** pair correlations as features, a second geometry, time-dependent pulses,
  differentiable pulse training (#420), device constraints, shots, more than four atoms, a
  tuned architecture, more than 4 tuning trials.

## References

- Bengio & Grandvalet (2004), "No Unbiased Estimator of the Variance of K-Fold
  Cross-Validation", *Journal of Machine Learning Research* 5, 1089–1105.
- Phipson & Smyth (2010), "Permutation P-values Should Never Be Zero", *Statistical
  Applications in Genetics and Molecular Biology* 9(1), Article 39.
