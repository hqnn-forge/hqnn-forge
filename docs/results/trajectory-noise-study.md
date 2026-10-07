# Does trajectory noise train as well as the exact channel?

The study behind #311. `noise_method="trajectories"` (#229) samples depolarizing noise as
Pauli trajectories at pure-state cost; its gradient is unbiased but noisier than the exact
`"density"` one. This compares the two, and noiseless training, on held-out data.

## Setup

- **Script:** `examples/study_trajectory_noise.py` (since the run, only the class balance in its
  docstring was corrected). Raw results, one JSON line per run:
  `docs/results/trajectory_noise_study.jsonl`.
- **Data:** scikit-learn's breast-cancer set (569 samples, 63 % positive: benign is class 1). It
  is bundled with scikit-learn, so the study runs offline; the credit-card and UCI benchmarks
  named in the issue were not used (see *Limits*).
- **Splits:** for each of 5 seeds, a stratified 60/20/20 train/validation/test split, with
  standardisation and PCA fitted on the training rows. The seed also fixes the initial weights,
  so every method starts from the same point.
- **Model:** `HybridBinaryClassifier(n_qubits, n_qubits, 2)` with 4 and 6 qubits, on
  `default.qubit` with backprop, trained with focal loss, Adam (lr 0.05), batch 32, up to
  30 epochs, and MCC early stopping (patience 10).
- **Methods:** noiseless, `density`, `trajectories` with `k = 1` and `k = 4` draws per sample,
  at p = 0.01 and 0.05 after every gate (`"all"`), and p = 0.05 before measurement (`"end"`).
- **Scores:** test MCC clean (validation threshold), and test MCC under the training noise
  (threshold re-tuned on the validation rows under that noise, as a model deployed on a noisy
  device would be).
- **Environment:** PennyLane 0.45.1, torch 2.14, one laptop CPU; 120 runs.

## Results

Test MCC under the training noise, mean ± sd over 5 seeds, and seconds per run. In the `end`
rows this score equals the clean one by construction (see *Limits*), so those rows compare the
training methods only:

| qubits | p | position | noiseless | density | trajectories k=1 | trajectories k=4 |
|---|---|---|---|---|---|---|
| 4 | 0.01 | all | 0.850 ± 0.071 | 0.861 ± 0.094 (17 s) | 0.902 ± 0.024 (8 s) | 0.893 ± 0.064 (12 s) |
| 4 | 0.05 | all | 0.848 ± 0.079 | 0.872 ± 0.050 (22 s) | 0.825 ± 0.085 (10 s) | 0.713 ± 0.326 (10 s) |
| 4 | 0.05 | end | 0.846 ± 0.072 | 0.839 ± 0.039 (13 s) | 0.856 ± 0.059 (6 s) | 0.882 ± 0.037 (6 s) |
| 6 | 0.01 | all | 0.875 ± 0.068 | 0.876 ± 0.087 (137 s) | 0.927 ± 0.039 (13 s) | 0.905 ± 0.067 (16 s) |
| 6 | 0.05 | all | 0.855 ± 0.065 | 0.883 ± 0.053 (120 s) | 0.673 ± 0.351 (16 s) | 0.922 ± 0.040 (17 s) |
| 6 | 0.05 | end | 0.847 ± 0.072 | 0.890 ± 0.035 (60 s) | 0.884 ± 0.058 (6 s) | 0.879 ± 0.101 (7 s) |

The noiseless model is trained once per seed, taking about 5 s. Total training time: density
1843 s, trajectories 298 s (k=1) and 340 s (k=4).

## What this shows

Everything below was measured on the breast-cancer proxy only: one small dataset, 4 and 6
qubits, 5 seeds. It has not been checked on the credit-card or UCI data that #311 names; that
is #414.

1. **At p = 0.01, and with noise only before measurement, trajectories train as well as the
   exact channel.** Paired by seed, the mean difference from density is between −0.01 and
   +0.05, which is within the seed-to-seed spread.
2. **At p = 0.05 after every gate, trajectory training was slow to start within this budget,
   and at least once did not start.** Calling a run *collapsed* when its test MCC under the
   noise is below 0.5 (the cut-off of the follow-up study, #347), 3 of 20 trajectory runs
   collapsed, against 0 of 10 density runs:
   - k=1 at 6 qubits, seeds 0 and 4: test MCC 0.19 and 0.42;
   - k=4 at 4 qubits, seed 1: test MCC 0.13.

   These are not three failures to train. All three were still on the initial loss plateau
   (training loss about 0.07) when the 30-epoch, patience-10 budget ended them; the k=4 run
   was stopped at epoch 15 and restored to its epoch-5 weights. Re-run during review with no
   early stopping and 60 epochs (not in the committed data):
   - k=4, 4 qubits, seed 1 left the plateau near epoch 25 and reached test MCC 0.94;
   - k=1, 6 qubits, seed 4 reached test MCC 0.82;
   - k=1, 6 qubits, seed 0 stayed on the plateau (loss 0.071, test MCC 0.23).

   So one of the three is a stall and two are slow starts that the budget cut off. Every
   density run trained within the same budget. In the other runs trajectories matched density,
   and k=4 at 6 qubits beat it on 4 of 5 seeds. The extra gradient variance is the likely
   cause of the longer plateau; whether k=4 prevents stalls cannot be told from one stall.
   Separating the two properly is #480.
3. **Cost.** Per epoch, trajectories trained about 2× faster at 4 qubits (1.8–2.5× across the
   settings) and 7–11× faster at 6. Density becomes impractical past about 6 qubits (#229),
   where trajectories keep pure-state memory.
4. **Noise-aware training did not beat noiseless training here**, even under noise: the
   noiseless model scores within 0.05 of density in every setting. For the two `end` settings
   that is a statement about clean accuracy only. At these noise levels on this data the noise
   is too weak to hurt a noiselessly trained model much, so this study says nothing either way
   about noise-aware training's value at stronger noise.

## Recommendation

- **Keep `noise_method="density"` as the default** wherever it fits in memory (up to about 6
  qubits). It never failed here.
- **Beyond that, use `"trajectories"` with `noise_trajectories ≥ 4`**, give it a longer
  training budget than density needs (more epochs, and a patience well above 10), and check
  the runs, especially at noise strengths of a few percent per gate. With 5 seeds, a run that
  has not left the plateau is visible as an outlier in the seed spread. The follow-up in
  `trajectory-collapse-study.md` (#347) raises this to **≥ 8**: at k = 8 every run trained
  within the same budget as density.
- **Do not switch the default automatically by qubit count** on this evidence. A slower start
  and an occasional stall are behaviour changes that a default should not introduce silently.

## Limits

- One small, easy dataset (noiseless MCC ≈ 0.85), 4 and 6 qubits, 5 seeds. With 5 paired seeds,
  a Wilcoxon signed-rank test cannot go below p = 0.0625, so none of the differences above is
  statistically significant. The slower start of trajectory training at p = 0.05 after every
  gate is the robust observation.
- **Training budget.** Every run had at most 30 epochs, with early stopping (patience 10) and
  checkpoint selection on the clean validation rows: `train_model` validates in eval mode,
  which is the noiseless circuit. There is no longer-budget control in the committed data, so
  it cannot separate "starts later" from "does not train"; the three re-runs under finding 2
  are the only evidence, and #480 is the full control.
- **The `end` rows do not test evaluation under noise.** Depolarizing before measurement scales
  every ⟨Z⟩ by the same factor 1 − 4p/3, and the head is one linear layer, so the samples keep
  their order and the re-tuned threshold returns the clean labels: the noisy and the clean
  score agree in all 40 `end` rows.
- The issue asked for the credit-card and UCI benchmarks through `run_benchmark`. When the study
  ran, the loaders (#266) and the benchmark runner (#269–#297) were on other open stacks. They
  have since been merged into this branch's base, but the study has not been re-run on them,
  for two reasons. First, cost: a 6-qubit density step at batch 256 took 9.1 s, and a credit-card
  training fold is about 364k rows after SMOTE balancing (about 1,420 steps), so one epoch of
  one fold would take about 3.6 h. Second, `run_benchmark` does not return its
  trained models, so `noise_sweep` cannot score them. The re-run is #414.
- The learning rate and schedule were not tuned per method. #347 tried lower learning rates
  within the same budget (`trajectory-collapse-study.md`): they did not help there, which a
  longer plateau would also explain. Whether they shorten or lengthen it is #480.
