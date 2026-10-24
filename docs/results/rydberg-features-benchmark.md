# Do the interactions of a Rydberg feature map add anything a head can use?

A worked run of the Rydberg benchmark runner (#501), behind #502. Each sample's inputs set the
detunings of a small simulated atom chain, and the excitation probabilities after one pulse are
the features of a small classical head. This note shows how to read one run: scores over the
interaction and dephasing grid, the paired comparisons, the separation measures of #500 beside
them, and the positive control that says whether the pipeline could have shown an effect at
all. It also checks that the model (#495), the solver (#496–#498, #507), the feature map
(#499), the separation measures (#500) and the runner work together end to end.

!!! warning "Draft"
    The results of this note are pending a decision on how the raw results are stored (see the
    pull request). The Setup below is final; the Results, Mechanism and Positive control
    sections are filled in from the committed results file once it is there.

## Setup

- **Script:** `examples/benchmark_rydberg_features.py` at the commit of this note. Raw results,
  one JSON line per arm, head, grid point and fold: `docs/results/rydberg_features_benchmark.jsonl`.
  Every table below is printed by `examples/summarise_rydberg_benchmark.py` from that file.
- **Reproduce:**

    ```bash
    uv run python examples/benchmark_rydberg_features.py \
        --dataset positive-control iranian-churn --data-path data/raw --download \
        --folds 13 --out docs/results/rydberg_features_benchmark.jsonl
    uv run python examples/summarise_rydberg_benchmark.py \
        docs/results/rydberg_features_benchmark.jsonl
    ```

    Every other option is the runner's default and is recorded in each line as `settings`.
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
  folds, on the real training rows) and the A − B difference of mean MCC, with a two-sided
  permutation p-value (10,000 draws, seed 0). The 28 grid points are not independent either
  (they share folds, and B is shared across `V/Ω`).

## Results

*Pending; see the warning above.*

## Mechanism

*Pending.*

## Positive control

*Pending.*

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
  scores). `taiwanese-bankruptcy` and `cervical-cancer-risk` were not run.
- **Not tried:** pair correlations as features, a second geometry, time-dependent pulses,
  differentiable pulse training (#420), device constraints, shots, more than four atoms, a
  tuned architecture, more than 4 tuning trials.

## References

- Bengio & Grandvalet (2004), "No Unbiased Estimator of the Variance of K-Fold
  Cross-Validation", *Journal of Machine Learning Research* 5, 1089–1105.
- Phipson & Smyth (2010), "Permutation P-values Should Never Be Zero", *Statistical
  Applications in Genetics and Molecular Biology* 9(1), Article 39.
