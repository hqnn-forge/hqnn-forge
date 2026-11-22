# What is left of the Rydberg A − B differences under Aquila's device settings?

The run of [the Rydberg features benchmark](rydberg-features-benchmark.md) (#502) repeated
under the device settings of #775, behind #776. Every published Rydberg run uses the idealised
model: a square pulse, a free detuning per atom, fixed positions and exact expectation values
without detection error. `examples/benchmark_rydberg_features.py --device aquila` replaces each
of these by the model its *Hardware realism* table names (`docs/rydberg-model.md`). This note
is to report what is left of the published A − B differences under those settings, and which
setting removes a difference where one goes.

**Status: no results yet.** The full run is not started. The timing below puts it at about
320 hours of this machine, far above a budget of about 10 hours, and `levels` has no source to
be set from. Both wait for a decision on #776's pull request. This page records the timing,
the planned runs and their exact commands, and the limits any result of them will have.

## Planned setup

Everything not listed here is as in #502's *Setup*: model and units, geometry (the open chain
of four atoms), the inputs, the arms, the heads, the tuning budget, the data and the splits.

- **Script:** `examples/benchmark_rydberg_features.py` with the options of #775 (#781).
- **Folds and pairing.** 13 outer folds on both datasets, with #502's data and split seed
  (`--seed 0`, the default), so each fold holds the same rows as the fold of the same index in
  `rydberg_features_benchmark.jsonl.xz`. A device line and the ideal line of the same arm, head,
  grid point and fold are then scored on the same test rows, and their difference can be
  paired by fold. Iranian Churn needs no 15 folds here (#746): the family of one run has
  m = 144 comparisons (*Statistics*), whose first Holm threshold `0.05/144 = 3.47e-4` lies above
  the 13-fold floor `2/2^13 = 2.44e-4`.
- **The preset** (`--device aquila`, `docs/rydberg-model.md`, *Hardware realism*): detection
  error `(p01, p10) = (0.01, 0.08)`, position jitter `σ = 0.2` µm over `K` displaced registers,
  the global drive with `t_ramp = 0.0503` µs (the slew bound at the reference Ω) in 32 segments
  per edge, and 100 shots per sample for arms A, B and D. Arm D reads through the same detection
  error; C and `D-hybrid` stay exact (#781).
- **Grid:** `V/Ω` = 0.1, 0.3, 1, 3, 10, 30, 100, as published. `γ/Ω` = 0, 0.03 and 0.1: 0.03
  lies in the device's indicative range of 0.014 to 0.042 at the reference Ω, and 0 and 0.1 are
  in the published grid. The stress values 1 and 10 are left out.
- **`levels`: not decided.** The device's pattern resolution is not stated in the sources the
  model page cites. The Braket program schema takes each `h_k` as a decimal between 0 and 1,
  and the Aquila capabilities schema it links gives `siteCoefficientRange` and a site-to-site
  RMS error of `h_k`, but no resolution. The runner therefore requires the value. The proposal
  on #776's pull request is `continuous` for the run and `8` at a subset of the grid.
- **`K`: not decided.** The jitter seed is one per run (#781), so the Monte Carlo error of the
  `K` registers is common to every fold and grid point and is not in the fold spread. It has to
  be bounded separately, by a second `K` at a subset of the grid. `--seed` cannot do it: it
  also seeds the folds, and the lines would no longer pair.
- **Ablation:** at `V/Ω` = 1, 3, 10 and 100 (every `V/Ω` with a Holm-significant A − B cell in
  #502) and `γ/Ω` = 0 and 0.1, each device factor alone on the square pulse, with every other
  option at its default: shots only, detection error only, jitter only, and the global drive
  only (with exact readout).

## Timing

Measured on the machine of #502 (one AMD Ryzen 7 250 laptop CPU, 16 threads), at commit
`0acd271` of #781.

- **`--quick` under the preset**, `--quick --device aquila --levels 8 --jitter-samples 4
  --no-cache`, alone on the machine: 33 s wall time (163 s of CPU), 9 lines. Arm A's features
  took 5.5 s for the fold's 161 rows, arm B's 1.2 s. `--quick` keeps its own grid point
  (`V/Ω = 1`, `γ/Ω = 0.1`), 40 solver steps, 160 samples, one fold of 4 and two inner folds.
- **What `--quick` leaves out.** A full run takes 200 steps per segment instead of 40, which
  is about 5 times the solver cost per map (the audit on #781), and 600 or 3,150 samples instead
  of 160. A line's `feature_seconds` covers the fold's own matrix only. The matrices of its 3
  tuning folds are simulated too and are recorded in no field (#785), so a fold costs about 4
  times its `feature_seconds`.
- **One fold at full settings**, to replace the extrapolation by a measurement:
  `--folds 13 --max-folds 1 --arms A B --device aquila --levels 8 --jitter-samples 4
  --v-over-omega 1 --no-cache`, the two datasets started side by side:
    - positive control, `--gamma-over-omega 0 0.03`: 14 min 4 s of wall time for both grid
      points, all of it beside the Iranian Churn run. Arm A's own matrix took 14.8 s at `γ = 0`
      and 147.7 s at `γ/Ω = 0.03`, arm B's 4.2 s and 36.2 s. The lines' `feature_seconds` add
      up to 203 s of the 844 s;
    - Iranian Churn, `--gamma-over-omega 0.03`: arm A's own matrix took 1,554 s, and its two
      lines (that matrix, the three tuning matrices and the tuning of both heads) were written
      60 min after the start; the first 14 min of it beside the control's run. The run was
      stopped there, before arm B.
  The `--quick` extrapolation (×5 for the steps, ×3.75 for the rows) gives 103 s for the
  control's arm A matrix at `γ > 0`, against 147.7 s measured beside the other run.
- **Estimate of the full run**, in hours of one run at a time, with `K = 4` and `levels 8`.
  Per fold and grid point, from the fold above: on the control arm A takes 60 s at `γ = 0` and
  590 s at `γ > 0` (4 times its own matrix), arm B 17 s and 145 s; on Iranian Churn arm A takes
  3,600 s at `γ > 0`, and the rest is scaled from the control by 10.5, the ratio of arm A's own
  matrices at `γ/Ω = 0.03`. The global drive alone, without jitter, costs what arm B costs under
  the preset. Arms C, D and `D-hybrid` and the square-pulse ablations are taken at
  the cost of the published runs, which is small beside these.

    | run | grid | positive control | Iranian Churn |
    |---|---|---|---|
    | preset | 7 `V/Ω` × 3 `γ/Ω` | 33 h | 211 h |
    | second `levels`, arms A, B, C | 2 × 2 | 5 h | 37 h |
    | second `K` (8), arms A, B, C | 2 × 1 (`γ = 0`) | 1 h | 10 h |
    | ablation: global drive only | 4 × 2 | 3 h | 31 h |
    | ablation: shots, detection error, jitter only; the ideal model at `γ/Ω = 0.03` | 4 × 2, 7 × 1 | 1 h | 4 h |
    | total | | 43 h | 292 h |

  The control's hours were measured beside another run. `--quick` took 33 s alone and 84 s
  beside the Iranian Churn run, so two runs at a time are not faster than one after the other,
  and alone the control's share may be nearer 17 h. **The runs #776 asks for therefore take
  about 310 to 335 hours, 13 to 14 days, of this machine**, against a budget of about 10 hours.
  The preset run on Iranian Churn alone is about 210 hours. `continuous` levels would cost
  somewhat more: in `--quick`, beside the other run, 103 s against 84 s for `levels 8`.
- **Where the time goes.** Most of it is arm A at `γ > 0`, ten times its cost at `γ = 0`
  (147.7 s against 14.8 s above): with dephasing each of the 65 segments of the ramped pulse
  gets the 200 steps of the whole pulse, and jitter multiplies that by `K` (#782).

## Reproduce

Not run yet. The commands, with `<L>` and `<K>` the values to be decided:

```bash
# Preset, both datasets (13 folds, the published V/Ω list, γ/Ω 0, 0.03, 0.1)
uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
    --dataset positive-control --folds 13 --gamma-over-omega 0 0.03 0.1 \
    --device aquila --levels <L> --jitter-samples <K> --out aquila-pc.jsonl
uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
    --dataset iranian-churn --data-path data/raw --download --folds 13 \
    --gamma-over-omega 0 0.03 0.1 \
    --device aquila --levels <L> --jitter-samples <K> --out aquila-churn.jsonl
# The other levels value, and a second K, at a subset of the grid (arms A, B and C)
for ds in positive-control iranian-churn; do
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
        --dataset $ds --data-path data/raw --download --folds 13 --arms A B C \
        --v-over-omega 1 10 --gamma-over-omega 0 0.03 \
        --device aquila --levels 8 --jitter-samples <K> --out aquila-$ds-levels-8.jsonl
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
        --dataset $ds --data-path data/raw --download --folds 13 --arms A B C \
        --v-over-omega 10 100 --gamma-over-omega 0 \
        --device aquila --levels <L> --jitter-samples <2K> --out aquila-$ds-k-2k.jsonl
done
# Ablation: each factor alone, square pulse unless stated
for ds in positive-control iranian-churn; do
    common="--dataset $ds --data-path data/raw --download --folds 13 \
        --v-over-omega 1 3 10 100 --gamma-over-omega 0 0.1"
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py $common \
        --shots 100 --out ablation-$ds-shots.jsonl
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py $common \
        --readout-error 0.01 0.08 --out ablation-$ds-detection.jsonl
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py $common \
        --position-jitter 0.2 --jitter-samples <K> --out ablation-$ds-jitter.jsonl
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py $common \
        --encoding global-drive --levels <L> --t-ramp 0.0503 --n-ramp-segments 32 \
        --out ablation-$ds-global-drive.jsonl
done
# The ideal model at γ/Ω = 0.03, which #502's file does not have
for ds in positive-control iranian-churn; do
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
        --dataset $ds --data-path data/raw --download --folds 13 \
        --gamma-over-omega 0.03 --out ideal-$ds-0.03.jsonl
done
```

Each file is to be compressed with `xz -9e` into `docs/results/` and summarised on its own with
`uv run --frozen --all-extras python examples/summarise_rydberg_benchmark.py <file>`: the
summariser refuses lines of different device models in one row (#775).

### Statistics

As in #502: the exact two-sided Wilcoxon signed-rank test on test MCC over the 13 folds,
Holm's correction at α = 0.05 with all comparisons of one run and dataset in one family, and
95 % t-intervals over the folds, which understate the uncertainty. The preset run has, per
dataset, 2 × 7 × 3 = 42 A − B comparisons and 2 × 51 = 102 of an arm row against the two
classical baselines (A 42, B 6, D 2, D-hybrid 1), m = 144. An ablation run has 2 × 4 × 2 = 16
and 2 × 23 = 46, m = 62.

The summariser prints the comparisons within one file. The paired difference between a device
line and the ideal line of the same fold, which the matched folds make possible, it does not
print (#784).

## Results

None yet.

## Limits

Any result of these runs will carry #502's *Limits* (system size, two-level approximation,
dephasing model, datasets, fold intervals), and in addition:

- **The detection error alone changes no score.** With exact features it maps each column
  `F` to `p01 + (1 − p01 − p10) F`, which the standardisation of the heads' features removes
  (#518, the audit on #781): without shots the heads of A, B and D see the same input with and
  without it, to rounding. It can change a score only together with shots, through the shot
  spread `√(F̂(1 − F̂)/S)` against the contrast `1 − p01 − p10`. The detection-only ablation can
  therefore show only the `separation` distances, and a difference that survives it says
  nothing about the detection error.
- **`K` adds an error that the fold spread does not contain.** The `K` displaced registers
  come from one jitter seed per run, shared by every fold and grid point, so their Monte Carlo
  error is the same in every fold: the intervals and the paired tests do not see it. Only the
  second `K` bounds it.
- **`levels` is a choice, not a device value.** The pattern resolution of Aquila's local
  detuning was not verified (*Planned setup*), and the local detuning itself is an experimental
  capability on request (`docs/rydberg-model.md`, *Hardware realism*).
- **The register is not checked against the device.** The preset checks the pulse, not the 4
  µm minimum spacing or the 75 µm region (#783). At `V/Ω = 100` the chain's spacing is 4.03 µm.
- **The timing above** is of one fold and one grid point per dataset. `--quick` alone would
  have understated a full preset map by about 5 times per map, before the grid size.
