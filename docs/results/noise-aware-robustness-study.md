# Noise-aware robustness: frozen study and execution

**NO SCIENTIFIC ROBUSTNESS MEASUREMENTS YET. PR #593 remains draft.**
The [prospective Decision](https://github.com/hqnn-forge/hqnn-forge/pull/593#issuecomment-6245122029)
for [#348](https://github.com/hqnn-forge/hqnn-forge/issues/348) is frozen in
[protocol JSON](noise_aware_robustness_protocol.json), with its canonical JSON
[SHA-256](noise_aware_robustness_protocol.sha256). The implementation is awaiting
independent numerical preflight before the full run. Synthetic checks verify
software contracts; they are not study evidence. The final note must report
actual MCC, wall times, learning diagnostics and a qualified recommendation.

## Acceptance status

All eight scientific acceptance criteria remain **not done: measurements and
independent audits pending**. Source support is present for criteria 1–5 and
for the paired inference in criterion 7:

1. Inherit the training-only standardisation/PCA method of
   `examples/study_trajectory_noise.py`, with matched training arms.
2. Measure stronger noise through probability 0.2.
3. Include depolarizing, amplitude damping and phase damping.
4. Use real credit-card and all three UCI datasets through `run_benchmark`.
5. Measure training/evaluation mismatch.
6. Answer whether, when and by how much noise-aware training changes test MCC.
7. Pair initialisations within more than five outer fold pairs and report
   Wilcoxon floors, effects, multiplicity and limitations.
8. Recommend use under the measured conditions or document that benefit was
   not established. Non-rejection does not demonstrate equality.

## Frozen data and recipe

Strict source loaders run before feature-vector deduplication. Keep the first
source-order row per distinct feature vector, independently of labels. This
prevents identical vectors crossing folds, but changes the population being
studied. Conflicting labels remain a data-quality limitation. Credit card
alone is then sampled using RNG 348: draw 116,359 negatives, then 200 positives,
without replacement, sort the combined indices and retain source order. This
preserves the eligible prevalence to rounding. Twenty positives per outer test
fold give five percentage-point recall resolution, not an MCC precision or
power guarantee. Do not generalise this sample to all transactions.

| Dataset | Loader rows / positives | Unique rows / positives | Study rows / positives | Test positives/fold |
|---|---:|---:|---:|---:|
| Credit card, time excluded | 284807 / 492 | 275663 / 473 | 116559 / 200 | 20 |
| Taiwan bankruptcy | 6819 / 220 | 6819 / 220 | 6819 / 220 | 22 |
| Iranian churn | 3150 / 495 | 2836 / 441 | 2836 / 441 | 44–45 |
| Cervical complete cases | 668 / 45 | 646 / 43 | 646 / 43 | 4–5 |

Churn has 14 feature-identical conflicting-label groups (64 source rows);
cervical has one (four rows). Credit card and Taiwan have none. Cervical's
complete-case selection, sparse positives and first-representative policy
materially limit inference. Manifest metadata retains source SHA-256, loader
options, raw/eligible/unique/sample counts, prevalence, feature names, selected
source-index hash and final array fingerprint. Private paths and raw rows
never enter shared records. The earlier
[feasibility metadata](noise_aware_robustness_feasibility.json) contains only
pre-study data and timing checks, with no robustness findings.

Every dataset is a single-entry mapping into `run_benchmark`, root seed 348,
ten outer folds, validation fraction 1/5 and two paired initialisations.
Derive all ten folds and the full seed tree before selecting a shard. Fit
mean/std and four-component full SVD only on inner training rows:
`Z=(X-mean_train)/(std_train+1e-12)`, projected onto the first four right
singular vectors, without whitening or rescaling. Both learners and all noise
arms share the transform and split/init/batch seeds. No SMOTE or tuning.

Use `HybridBinaryClassifier(4, 4, 2)`, angle/RX embedding, ring entangler, all Z
readouts, tanh encoder, restricted initialisation, no dropout, analytic
`default.qubit`/backprop, CPU float32 inputs. Focal loss uses alpha .25,
gamma 2 and mean reduction; Adam uses learning rate .05 and locked-version
defaults; batch size is 32. Begin with 30 epochs/patience 10, selecting and
restoring the best **clean-validation MCC** checkpoint.

The optional `ValidationExtension(60, .10, 2.0)` applies identically to both
learners. After the first attempt, validation under its own training noise
must be finite, have MCC at least .10 and average precision at least twice
validation prevalence. Otherwise restart once from the original weights and
Torch/batch seeds with fresh Adam, 60 epochs and no patience. Always retain
the second attempt when triggered, even if worse. Record both attempts,
full-precision epoch loss and clean/noisy validation MCC, selected epoch,
threshold, reason, steps and time. Only then score outer test rows. This is a
common adaptive training policy, not equal realised compute or a convergence
guarantee. The library's default training policy remains single-fit.

## Scoring and inference

Train noiselessly and at .2 under each of the three exact density channels,
placement `all`; evaluate at 0, .01, .05, .1, .2 with chunks of 256. Record
fixed clean-validation and evaluation-noise-validation-retuned thresholds,
both test MCCs and confusion counts, average precision and Brier score.
Retuned MCC is primary. Test labels never select a threshold. The noiseless
hybrid is the noise-training comparator; the native classical control is
retained only for context. Repeated baseline fits across channels must match
in seeds, validation diagnostics, clean MCC and checkpoint hash.

The off-diagonal .2→0/.01/.05/.1 cells describe downward mismatch and the clean
performance tradeoff. No upward mismatch, milder-noise training or cross-channel
mismatch is claimed. End placement is an implementation control only:
`z'=(1-4p/3)z`, `z'=(1-p)z+p`, and `z'=z` for depolarizing, amplitude damping,
and phase damping respectively. With a linear head these give positive
affine logits over this range; retuning can conceal calibration displacement
without demonstrating changed discrimination.

There are exactly twelve primary hypotheses: four datasets × three channels,
training .2 versus noiseless hybrid, both evaluated at .2/all with retuned
thresholds. Mean the two paired seed MCC differences within each fold, then
use ten fold differences for two-sided exact Wilcoxon and rank-biserial effect.
Map absolute fold means ≤1e-12 to zero **only for this study's ranks/test**;
retain unrounded descriptive effects. Holm uses all twelve slots at alpha .05.
Undefined/non-finite/all-zero tests occupy an unrejectable internal p=1 slot
while reported p-values remain missing. Report effective pairs and attainable
floors. Ten, nine and eight nonzero pairs have raw floors .001953125,
.00390625 and .0078125; first Holm bounds are .0234375, .046875 and .09375.
All lower-noise levels, fixed-threshold scores, ranking and calibration
metrics are descriptive secondary evidence.

Before interpreting noisy tests, assess the noiseless baseline's clean
validation diagnostics: each fold passes when both seeds are finite, median
MCC ≥.10 and median average precision ≥twice prevalence. Fewer than 8/10
passing folds, or predominantly single-class/near-chance held-out clean
performance, means **learning-limited/inconclusive**. Retain every cell;
this interpretation guard never triggers another retry or changes weights.
The final interpretation must inspect retained clean test scores/confusion
counts as well as validation pass counts. A positive recommendation requires
positive primary effect, Holm p<.05 and a learned baseline. Non-rejection is
inconclusive about equality. Fold test rows are disjoint but training sets
overlap, so conventional paired-CV p-values are not exact population guarantees.

## Reproduce after independent preflight

Run from a clean tracked checkout with the frozen lockfile. Use a fresh local
output directory. Raw source locations below are placeholders, never published
record fields. A process-held lock prevents duplicate owners. Scientific
`prepare` requires a matching synthetic smoke proof from a separate directory.

```bash
uv run --frozen --all-extras python examples/study_noise_robustness.py smoke --output /tmp/robustness-smoke
uv run --frozen --all-extras python examples/study_noise_robustness.py prepare --output /tmp/robustness-study --credit-card /path/to/creditcard.csv --uci /path/to/uci --smoke-proof /tmp/robustness-smoke/smoke.json
uv run --frozen --all-extras python examples/study_noise_robustness.py run --output /tmp/robustness-study --credit-card /path/to/creditcard.csv --uci /path/to/uci
uv run --frozen --all-extras python examples/study_noise_robustness.py analyze --output /tmp/robustness-study
```

`run` can select `--dataset`, `--channel`, `--training` and `--fold` while
preserving the complete seed tree. Per-block records are atomic and immutable;
complete records are reused after protocol/data/code/recipe/seed/inventory/hash
checks. Incomplete blocks restart with the same settings. Selected model,
optimizer and preprocessor state and per-attempt diagnostics stay in local
sidecars. `progress.json`, epoch/fold logs, local process metadata and exit
status survive interruption. Analysis refuses missing/duplicate/mispaired
blocks and requires all 240 blocks, each containing two initialisations and
their classical controls. Its `analysis.json` is computed exclusively from
saved records and contains the primary and descriptive tables, fold/seed
scores, twelve-slot corrections, learning counts and summed block wall time.
The final human interpretation and actual results note remain pending.

One credit-card density epoch needs about 2,623 optimizer steps. Its sixty
noisy fits imply about 4.72 million steps if each first attempt reaches thirty
epochs, or 14.16 million if every fit also triggers sixty more. The prior
four-qubit depolarizing probe's .061–.063 seconds/step implies 3.3–3.4 serial
days for those first attempts or 10.0–10.3 days at maximum, excluding other
channels' cost variation, other datasets, baseline/control fits, validation
and scoring. This is an illustrative step extrapolation, not measured total
wall time. Actual timing must be reported separately. Process-isolated block
scheduling may use up to four workers with four Torch threads/one interop
thread each where resources permit, using distinct owned directories and
subsequently verified artifacts. Recipe settings cannot change for speed.

Refs [#414](https://github.com/hqnn-forge/hqnn-forge/issues/414) and
[#428](https://github.com/hqnn-forge/hqnn-forge/issues/428): only the necessary
additive scoring/record subset is implemented here; their broader trajectory,
width, kernel and recommendation questions remain open.
