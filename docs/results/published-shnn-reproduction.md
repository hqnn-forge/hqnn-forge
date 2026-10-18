# Does the library reproduce the published SHNN's numbers?

The measurement behind #184. `HybridBinaryClassifier.published_shnn()` matches the published
model structurally (`tests/test_published_shnn_parity.py`). This run checks whether the library,
following the benchmark's training recipe, also lands on the published MCC 0.5758 ± 0.0371 and
MCC/kParam 4.720.

## Setup

- **Harness:** `tests/test_published_shnn_reproduction.py`, whose module docstring lists the
  recipe and every deliberate deviation from the benchmark code. Raw result and provenance:
  `docs/results/published_shnn_reproduction.json`.
- **Command:**

  ```bash
  HQNN_FORGE_REPRODUCE=1 HQNN_FORGE_DATA=data/raw \
      pytest tests/test_published_shnn_reproduction.py -m reproducibility -s
  ```

- **Data:** the full Kaggle credit-card fraud set, 284,807 rows with 492 frauds. An 80/20
  stratified split; the 80 % supplies five training/validation folds and every fold's restored
  model is scored on the same held-out 20 %.
- **Recipe:** SMOTE on each training fold, Adam (lr 0.01), batch 256, up to 100 epochs, early
  stopping on validation MCC at the benchmark's threshold grid (0.05..0.94) with patience 20,
  best epoch restored.
- **Differences from the benchmark:** the fold and SMOTE draws are the library's, and the loss
  is `BCEWithLogitsLoss` rather than `Sigmoid` + `BCELoss`. No recipe or tolerance was changed
  for this measurement.
- **Environment:** commit `ef61672915da05fb8b317f2ddbeb2bf20dbdba93`, analytic `default.qubit`
  with `backprop` (the preset's resolved `auto` backend), PennyLane 0.45.1, torch 2.14.1,
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set to 4, on an AMD Ryzen 7 250
  CPU. The run took about 2 hours (7,244 s for 106 epochs).

## Results

Measured on 2026-10-16 (UTC). Both unchanged opt-in assertions passed (exit 0):

- mean test MCC **0.54222358 ± 0.03191070** (sample standard deviation over five folds), inside
  [0.5387, 0.6129] and 0.03357642 below the published 0.5758;
- MCC/kParam **4.44445560** with all 122 parameters, inside 4.720 ± 0.30409836.

| Fold (zero based) | Test MCC | Best epoch | Validation threshold | Epochs run |
|---|---:|---:|---:|---:|
| 0 | 0.53880796 | 1 | 0.94 | 21 |
| 1 | 0.56472257 | 1 | 0.94 | 21 |
| 2 | 0.56049143 | 1 | 0.94 | 21 |
| 3 | 0.55907282 | 2 | 0.94 | 22 |
| 4 | 0.48802314 | 1 | 0.94 | 21 |

## Limits

- **The models barely trained.** Every fold's best validation MCC came at epoch 1 or 2, and
  early stopping ended each fold at epoch 21 or 22. These scores therefore reflect one or two
  epochs of training, not the 100-epoch budget. Every restored threshold also sits at 0.94,
  the top of the benchmark's grid. The retained stdout logs contain all 106 per-epoch
  observations, including validation MCC and training loss rounded to four decimal places.
  These logs are retained but are not published with this page. The published result JSON
  omits this history, and full-precision trajectories, per-epoch thresholds, and
  uncapped-threshold diagnostics were not captured. The rounded logs cannot always explain
  why a particular epoch was restored, because early stopping selects the best epoch using
  full-precision validation MCC, and different epochs can have the same value when rounded
  to four decimal places. Whether the grid cap contributed to early stopping is not
  established; #587 requests the full-precision per-epoch history for future runs, and #588
  investigates the cause.
- **The bands are loose.** A pass means the mean lies within one published fold standard
  deviation of the published mean. It does not establish exact numerical replication or
  statistical equivalence.
- **One run.** A later run, such as the rerun in #588, belongs on this page next to this one,
  so the two can be compared for drift.
