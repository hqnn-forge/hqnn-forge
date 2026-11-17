# Is the chain's null result a property of the chain?

The run of [the Rydberg features benchmark](rydberg-features-benchmark.md) (#502) repeated on
the ring and the `2 × 2` grid (#733), behind #737. #502 found that on Iranian Churn the
interactions of a four-atom **chain** added nothing a head could use. On a chain each atom has at
most two neighbours and the next pair is already `V/64`. This note asks whether the result holds
when the same four inputs sit on a ring or on a grid, where every atom has two neighbours and the
next pair is `V/8`. It shows the scores, the paired comparisons and the separation measures for
each geometry. It reads them as #502's note does, and the reader is assumed to know that note.

## Setup

Everything not listed here is as in #502's *Setup*: model and units, encoding, readout, noise,
arms, heads, tuning budget, data and splits. Every option of the runner is at its default except
`--folds` and `--geometry`, and each line records them as `settings`. The positive controls
have 13 folds, as in #502. Iranian Churn has 15, so that its Holm family can reach α
(*Statistics*, #746).

- **Script:** `examples/benchmark_rydberg_features.py` as of commit `29c8931` for the controls,
  i.e. with `--geometry` (#733) and a positive control labelled over the bonds of its geometry
  (#738), and as of `efa6ef3` for Iranian Churn with 15 folds. The later script also records
  each fold's PCA fit (`pca`, #525) and otherwise writes the same lines: run at `efa6ef3` with
  `--folds 13 --max-folds 1 --geometry chain --arms A B C`, it reproduced all 114 lines of fold
  0 of the 13-fold Iranian Churn file in `mcc`, `pr_auc`, `threshold`, `epochs`, `data_sha256`
  and the train, validation and test rows.
- **Raw results:** five files in `docs/results/`, the runner's exact JSONL, xz-compressed. The
  file of #502, `rydberg_features_benchmark.jsonl.xz`, is unchanged.
    - `rydberg_features_geometries_iranian_churn_15_folds.jsonl.xz`: Iranian Churn with
      `--geometry chain ring grid --folds 15` in one run (#746), the run whose tables this note
      shows. Arm A runs per geometry. Arm B is computed once and recorded per geometry with
      `reused_from_geometry`. C and D have no geometry.
    - `rydberg_features_geometries_iranian_churn.jsonl.xz`: the same run with 13 folds (#737),
      kept unchanged as the record of that run. Its Iranian Churn tables were in this note
      until #746 and are printed by the same command; it is not read with the 15-fold file
      (*Statistics*).
    - `rydberg_features_geometries_control_{chain,ring,grid}.jsonl.xz`: the positive control,
      once per geometry. When they were run (#738), a run with the control took one geometry,
      because its label follows the geometry's bonds. Since #741 one run with
      `--geometry chain ring grid` runs the control once per geometry and writes the lines of the
      three runs, apart from `settings` and the timings.
- **Reproduce:**

    ```bash
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
        --dataset iranian-churn --data-path data/raw --download --folds 15 \
        --geometry chain ring grid --out churn-15.jsonl
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
        --dataset iranian-churn --data-path data/raw --download --folds 13 \
        --geometry chain ring grid --out churn.jsonl
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
        --dataset positive-control --folds 13 --geometry chain --out pc-chain.jsonl
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
        --dataset positive-control --folds 13 --geometry ring --out pc-ring.jsonl
    uv run --frozen --all-extras python examples/benchmark_rydberg_features.py \
        --dataset positive-control --folds 13 --geometry grid --out pc-grid.jsonl
    xz -9e -k -c churn-15.jsonl \
        > docs/results/rydberg_features_geometries_iranian_churn_15_folds.jsonl.xz
    xz -9e -k -c churn.jsonl > docs/results/rydberg_features_geometries_iranian_churn.jsonl.xz
    xz -9e -k -c pc-chain.jsonl > docs/results/rydberg_features_geometries_control_chain.jsonl.xz
    xz -9e -k -c pc-ring.jsonl > docs/results/rydberg_features_geometries_control_ring.jsonl.xz
    xz -9e -k -c pc-grid.jsonl > docs/results/rydberg_features_geometries_control_grid.jsonl.xz
    for f in iranian_churn_15_folds iranian_churn control_chain control_ring control_grid; do
        uv run --frozen --all-extras python examples/summarise_rydberg_benchmark.py \
            docs/results/rydberg_features_geometries_$f.jsonl.xz
    done
    ```

    Summarise each file on its own. The three control files hold three different labels under
    one dataset name. The summariser refused them read together when this note was written
    (#742); since #741 it prints them as three sections, one per `label_bonds`, each its own
    family of 190 and identical to that file's summary apart from its heading. Never join the two
    Iranian Churn files into one input: they have the same data and no label bonds, so they
    fall into one group, whose lines would be paired by fold index although the 13 and the 15
    folds are different splits. Since #753 the summariser refuses such a group (more than one
    `n_splits`, or more than one `test_idx` for a fold) before printing anything. Every table
    below is that output, with its headings moved down a level and its separation measures
    moved to *Mechanism*.
- **The chain reproduces #502 on the control, not on Iranian Churn with 15 folds.** Every chain
  line of the control run and of the 13-fold Iranian Churn run equals the line of #502's file
  for the same arm, head, grid point and fold in `mcc`, `pr_auc` and `data_sha256`: 1,521 of
  1,521 on each. The chain control, summarised alone, prints #502's positive-control tables,
  apart from the measured seconds. They are not repeated here. The 15-fold run has the same
  data, the same split seed and the same seeds for each of its first 13 folds, but other folds:
  15 stratified test parts of 210 rows instead of 13 of 242 to 244. Its chain lines therefore
  do not reproduce #502's, and its chain cells below differ from those of #502's note.
- **Geometries** (`docs/rydberg-model.md`, *Geometry*). Four atoms each. `V/Ω` is the
  nearest-neighbour interaction `C6/a⁶` in every geometry, set through the spacing as in #502
  (`a` = 12.76 µm at `V/Ω = 0.1` down to 4.03 µm at 100). Input `i` (the `i`-th principal
  component) sets the detuning of atom `i`.

    | Geometry | Atom `i` at | Neighbouring inputs | Next-nearest pair |
    |---|---|---|---|
    | chain | `(i · a, 0)` | (0,1), (1,2), (2,3) | `2a`, `V/64` (and `3a`, `V/729`) |
    | ring | corner `i` of a square of side `a`, counterclockwise | (0,1), (1,2), (2,3), (0,3) | the diagonals (0,2), (1,3): `√2 a`, `V/8` |
    | grid `2 × 2` | `(c · a, r · a)` for `i = 2r + c` | (0,1), (0,2), (1,3), (2,3) | the diagonals (0,3), (1,2): `√2 a`, `V/8` |

- **Next-nearest couplings at the swept `V/Ω`**, as `V_ij/Ω` of the next-nearest pair (the
  table of `docs/rydberg-model.md`):

    | `V/Ω` | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 |
    |---|---|---|---|---|---|---|---|
    | chain, `2a` | 0.0016 | 0.0047 | 0.016 | 0.047 | 0.16 | 0.47 | 1.6 |
    | ring of 4 and `2 × 2` grid, `√2 a` | 0.0125 | 0.0375 | 0.125 | 0.375 | 1.25 | 3.75 | 12.5 |

    On the square the diagonal pairs also exceed `Ω` from `V/Ω = 10` on (they do from `V/Ω > 8`),
    so at 10, 30 and 100 every pair of the four atoms is in the blockade. On the chain only the
    next-nearest pairs at `V/Ω = 100` are.
- **At four atoms the ring and the `2 × 2` grid are the same square.** Both have four sides of
  length `a` and two diagonals of length `√2 a`. They differ only in which inputs sit on
  neighbouring atoms: the grid is the ring with inputs 2 and 3 exchanged. On the ring inputs 1
  and 2 are neighbours and 0 and 2 are not; on the grid it is the other way round. So the ring
  against the grid asks whether it matters which principal components interact, not which shape
  they sit on.
- **The positive control follows the geometry** (#738). Its label is the sign of
  `Σ z_j z_k` over the neighbouring pairs `(j, k)` in the table above, recorded in every line as
  `label_bonds`. The inputs are the same in all three geometries (same `data_seed`), the labels
  are not: they differ from the chain's on 122 (ring) and 186 (grid) of the 600 samples
  (#738). **The three controls are therefore three different tasks.** Their B, C and D rows
  differ, and a control score of one geometry is not comparable with another's.
- **Environment:** as #502 (Python 3.13.15, the same laptop), except torch 2.15.0 for 2.14.1, as
  recorded in every line as `environment`. The chain lines are identical nonetheless (above).
  The 15-fold Iranian Churn run has the same packages on a newer kernel and glibc
  (`Linux-7.3.3-300.fc45`, glibc 2.44), on which the one-fold check above was run too.

### Statistics

As in #502 (exact two-sided Wilcoxon signed-rank test on test MCC over the outer folds, 13 on
the controls and 15 on Iranian Churn, Holm's correction at α = 0.05, 95 % t-intervals over the
folds), with these families:

- **One Holm family per dataset**, as #502 and #737 have it. On **Iranian Churn**, which holds
  all three geometries in one file, that family has
    - A against B per head, grid point and geometry: 3 × 56 = 168;
    - every arm row against each of the two classical baselines: A 168, B 8, D 2, D-hybrid 1,
      i.e. 179 × 2 = 358;
    - m = 526.

  The family is the 15-fold file's alone. The 13-fold file is a different split of the same
  rows and is not tested beside it.
- **Each positive control is its own dataset**, with its own label and data fingerprint, so
  each is its own family of m = 190, as in #502.
- **Why 15 folds on Iranian Churn.** Holm's first threshold for m = 526 is
  `0.05/526 = 9.51e-5`. The smallest exact p-value over `n` untied folds is `2/2^n`: `2.44e-4`
  at 13 folds and `1.22e-4` at 14, both above it, and `6.10e-5` at 15, below it. With 13 folds
  (#737's run, `rydberg_features_geometries_iranian_churn.jsonl.xz`) the smallest possible
  Holm-adjusted p was `526 · 2/2^13 = 0.1284`, so no comparison could be significant, and this
  note read that run as descriptive. With 15 folds it is `526 · 2/2^15 = 0.0321`.
- **What 15 folds can and cannot detect here.** In this run 109 of the 526 comparisons are
  significant, every one of them won or lost in all 15 folds (Holm p 0.0321). The next
  possible p-value, `4/2^15 = 1.22e-4` (14 of 15 folds, the other fold the smallest
  difference), adjusts to `(526 − 109) · 1.22e-4 = 0.0509` and is not significant. So a cell
  lost in 14 of 15 folds is not called a loss, however large: e.g. the chain's linear A − B at
  `V/Ω = 1`, `γ/Ω = 0`, −0.183, Holm p 0.0509.
- **The chain cells of #502 are not re-tested.** #502's chain comparisons on Iranian Churn
  were made on its 13 folds in its family of 190, and its statements hold there. The 15-fold
  run has other folds (*Setup*), so its chain cells are a new sample, not a reproduction. Of
  #502's two significant Iranian Churn cells, the linear loss at `V/Ω = 1`, `γ/Ω = 0` (−0.181,
  Holm p 0.0464 in #502) is −0.183 here, lost in 14 of 15 folds (Holm p 0.0509). The MLP gain
  at `V/Ω = 10`, `γ/Ω = 1` (+0.053, Holm p 0.0464 in #502) is +0.013 here, won in 10 of 15
  folds (Holm p 1.0000).

## Results

As in #502, `n` is the number of folds, the interval the 95 % t-interval of the fold mean,
`dim` the feature dimension, `params` the trainable parameters, and `feature s` and `fit s` the
mean seconds of feature simulation and of the final fit per fold. A grid cell of a comparison is
the mean MCC difference over the folds with its Holm-adjusted p-value, and ✓ marks the ones at
most α = 0.05. Arm B rows do not depend on `V/Ω`. On Iranian Churn they do not depend on the
geometry either, and are shown once. `--long` prints every comparison with its pairs, exact
p-value, floor and rank-biserial effect. Under the Iranian Churn scores table, the summariser's
line of #747 counts the folds whose PCA fit, or one of whose tuning fits, was degenerate (#525):
none. The control files predate that record, and their tables were printed before #747.

### Positive control, ring

Folds per row: [13].  Holm family: 190 comparisons at α = 0.05, first threshold α/m = 2.63e-04.

#### Scores

| arm | head | V/Ω | γ/Ω | n | MCC mean ± sd | MCC 95 % CI | PR-AUC mean ± sd | PR-AUC 95 % CI | dim | params | feature s | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | linear | 0.1 | 0 | 13 | 0.204 ± 0.159 | [0.108, 0.300] | 0.571 ± 0.089 | [0.517, 0.625] | 4 | 5 | 0.03 | 0.14 |
| A | linear | 0.3 | 0 | 13 | 0.212 ± 0.159 | [0.116, 0.308] | 0.587 ± 0.102 | [0.526, 0.649] | 4 | 5 | 0.02 | 0.08 |
| A | linear | 1 | 0 | 13 | 0.302 ± 0.200 | [0.181, 0.423] | 0.776 ± 0.115 | [0.707, 0.846] | 4 | 5 | 0.03 | 0.10 |
| A | linear | 3 | 0 | 13 | 0.381 ± 0.158 | [0.285, 0.476] | 0.791 ± 0.072 | [0.747, 0.835] | 4 | 5 | 0.02 | 0.09 |
| A | linear | 10 | 0 | 13 | 0.394 ± 0.065 | [0.355, 0.433] | 0.702 ± 0.044 | [0.675, 0.728] | 4 | 5 | 0.03 | 0.13 |
| A | linear | 30 | 0 | 13 | 0.273 ± 0.097 | [0.215, 0.332] | 0.674 ± 0.044 | [0.647, 0.701] | 4 | 5 | 0.03 | 0.09 |
| A | linear | 100 | 0 | 13 | 0.315 ± 0.112 | [0.247, 0.382] | 0.706 ± 0.060 | [0.670, 0.742] | 4 | 5 | 0.03 | 0.08 |
| A | linear | 0.1 | 0.1 | 13 | 0.215 ± 0.142 | [0.129, 0.301] | 0.570 ± 0.092 | [0.515, 0.625] | 4 | 5 | 0.20 | 0.12 |
| A | linear | 0.3 | 0.1 | 13 | 0.246 ± 0.133 | [0.165, 0.326] | 0.609 ± 0.083 | [0.559, 0.659] | 4 | 5 | 0.15 | 0.10 |
| A | linear | 1 | 0.1 | 13 | 0.333 ± 0.214 | [0.203, 0.462] | 0.784 ± 0.113 | [0.716, 0.852] | 4 | 5 | 0.24 | 0.11 |
| A | linear | 3 | 0.1 | 13 | 0.395 ± 0.142 | [0.309, 0.480] | 0.798 ± 0.069 | [0.756, 0.840] | 4 | 5 | 0.21 | 0.09 |
| A | linear | 10 | 0.1 | 13 | 0.359 ± 0.096 | [0.301, 0.417] | 0.699 ± 0.042 | [0.673, 0.724] | 4 | 5 | 0.17 | 0.09 |
| A | linear | 30 | 0.1 | 13 | 0.214 ± 0.078 | [0.167, 0.261] | 0.657 ± 0.053 | [0.625, 0.689] | 4 | 5 | 0.13 | 0.09 |
| A | linear | 100 | 0.1 | 13 | 0.312 ± 0.125 | [0.237, 0.388] | 0.730 ± 0.045 | [0.703, 0.757] | 4 | 5 | 0.13 | 0.12 |
| A | linear | 0.1 | 1 | 13 | 0.229 ± 0.138 | [0.146, 0.313] | 0.588 ± 0.083 | [0.538, 0.638] | 4 | 5 | 0.15 | 0.07 |
| A | linear | 0.3 | 1 | 13 | 0.218 ± 0.156 | [0.124, 0.313] | 0.578 ± 0.081 | [0.529, 0.627] | 4 | 5 | 0.14 | 0.09 |
| A | linear | 1 | 1 | 13 | 0.244 ± 0.145 | [0.157, 0.332] | 0.713 ± 0.077 | [0.666, 0.759] | 4 | 5 | 0.15 | 0.08 |
| A | linear | 3 | 1 | 13 | 0.214 ± 0.202 | [0.091, 0.336] | 0.605 ± 0.116 | [0.535, 0.674] | 4 | 5 | 0.16 | 0.17 |
| A | linear | 10 | 1 | 13 | 0.299 ± 0.116 | [0.229, 0.369] | 0.654 ± 0.043 | [0.628, 0.681] | 4 | 5 | 0.13 | 0.07 |
| A | linear | 30 | 1 | 13 | 0.172 ± 0.130 | [0.094, 0.251] | 0.619 ± 0.072 | [0.575, 0.663] | 4 | 5 | 0.21 | 0.08 |
| A | linear | 100 | 1 | 13 | 0.260 ± 0.146 | [0.172, 0.348] | 0.717 ± 0.066 | [0.677, 0.757] | 4 | 5 | 0.19 | 0.09 |
| A | linear | 0.1 | 10 | 13 | 0.227 ± 0.161 | [0.129, 0.324] | 0.590 ± 0.072 | [0.546, 0.633] | 4 | 5 | 0.19 | 0.14 |
| A | linear | 0.3 | 10 | 13 | 0.200 ± 0.159 | [0.104, 0.296] | 0.591 ± 0.075 | [0.546, 0.637] | 4 | 5 | 0.18 | 0.07 |
| A | linear | 1 | 10 | 13 | 0.241 ± 0.137 | [0.158, 0.324] | 0.607 ± 0.085 | [0.556, 0.659] | 4 | 5 | 0.15 | 0.08 |
| A | linear | 3 | 10 | 13 | 0.200 ± 0.151 | [0.108, 0.291] | 0.608 ± 0.072 | [0.565, 0.652] | 4 | 5 | 0.13 | 0.10 |
| A | linear | 10 | 10 | 13 | 0.248 ± 0.120 | [0.176, 0.321] | 0.608 ± 0.075 | [0.563, 0.653] | 4 | 5 | 0.17 | 0.08 |
| A | linear | 30 | 10 | 13 | 0.253 ± 0.099 | [0.194, 0.313] | 0.581 ± 0.091 | [0.525, 0.636] | 4 | 5 | 0.20 | 0.25 |
| A | linear | 100 | 10 | 13 | 0.220 ± 0.124 | [0.145, 0.295] | 0.579 ± 0.085 | [0.527, 0.630] | 4 | 5 | 0.16 | 0.13 |
| A | mlp | 0.1 | 0 | 13 | 0.714 ± 0.091 | [0.659, 0.769] | 0.955 ± 0.022 | [0.942, 0.969] | 4 | 49 | 0.03 | 0.28 |
| A | mlp | 0.3 | 0 | 13 | 0.593 ± 0.099 | [0.534, 0.653] | 0.908 ± 0.030 | [0.890, 0.926] | 4 | 49 | 0.02 | 0.13 |
| A | mlp | 1 | 0 | 13 | 0.465 ± 0.160 | [0.368, 0.561] | 0.834 ± 0.120 | [0.761, 0.907] | 4 | 49 | 0.03 | 0.14 |
| A | mlp | 3 | 0 | 13 | 0.492 ± 0.123 | [0.417, 0.566] | 0.843 ± 0.048 | [0.813, 0.872] | 4 | 49 | 0.02 | 0.16 |
| A | mlp | 10 | 0 | 13 | 0.576 ± 0.100 | [0.515, 0.636] | 0.871 ± 0.037 | [0.849, 0.894] | 4 | 49 | 0.03 | 0.19 |
| A | mlp | 30 | 0 | 13 | 0.262 ± 0.087 | [0.210, 0.315] | 0.723 ± 0.048 | [0.694, 0.753] | 4 | 49 | 0.03 | 0.13 |
| A | mlp | 100 | 0 | 13 | 0.383 ± 0.080 | [0.334, 0.432] | 0.795 ± 0.039 | [0.771, 0.819] | 4 | 49 | 0.03 | 0.15 |
| A | mlp | 0.1 | 0.1 | 13 | 0.680 ± 0.106 | [0.616, 0.744] | 0.953 ± 0.019 | [0.941, 0.964] | 4 | 49 | 0.20 | 0.17 |
| A | mlp | 0.3 | 0.1 | 13 | 0.574 ± 0.099 | [0.515, 0.634] | 0.898 ± 0.040 | [0.874, 0.922] | 4 | 49 | 0.15 | 0.13 |
| A | mlp | 1 | 0.1 | 13 | 0.460 ± 0.193 | [0.343, 0.577] | 0.828 ± 0.109 | [0.762, 0.894] | 4 | 49 | 0.24 | 0.23 |
| A | mlp | 3 | 0.1 | 13 | 0.517 ± 0.106 | [0.453, 0.581] | 0.861 ± 0.061 | [0.824, 0.898] | 4 | 49 | 0.21 | 0.25 |
| A | mlp | 10 | 0.1 | 13 | 0.645 ± 0.091 | [0.590, 0.700] | 0.914 ± 0.040 | [0.890, 0.938] | 4 | 49 | 0.17 | 0.30 |
| A | mlp | 30 | 0.1 | 13 | 0.287 ± 0.103 | [0.224, 0.349] | 0.743 ± 0.064 | [0.705, 0.782] | 4 | 49 | 0.13 | 0.20 |
| A | mlp | 100 | 0.1 | 13 | 0.397 ± 0.115 | [0.328, 0.467] | 0.789 ± 0.051 | [0.758, 0.819] | 4 | 49 | 0.13 | 0.33 |
| A | mlp | 0.1 | 1 | 13 | 0.695 ± 0.096 | [0.637, 0.753] | 0.949 ± 0.026 | [0.933, 0.965] | 4 | 49 | 0.15 | 0.24 |
| A | mlp | 0.3 | 1 | 13 | 0.621 ± 0.084 | [0.570, 0.672] | 0.915 ± 0.030 | [0.897, 0.933] | 4 | 49 | 0.14 | 0.14 |
| A | mlp | 1 | 1 | 13 | 0.250 ± 0.140 | [0.165, 0.335] | 0.733 ± 0.086 | [0.681, 0.785] | 4 | 49 | 0.15 | 0.16 |
| A | mlp | 3 | 1 | 13 | 0.543 ± 0.105 | [0.480, 0.606] | 0.844 ± 0.065 | [0.805, 0.884] | 4 | 49 | 0.16 | 0.15 |
| A | mlp | 10 | 1 | 13 | 0.643 ± 0.104 | [0.580, 0.706] | 0.927 ± 0.036 | [0.906, 0.949] | 4 | 49 | 0.13 | 0.19 |
| A | mlp | 30 | 1 | 13 | 0.349 ± 0.140 | [0.265, 0.434] | 0.710 ± 0.087 | [0.658, 0.763] | 4 | 49 | 0.21 | 0.15 |
| A | mlp | 100 | 1 | 13 | 0.491 ± 0.106 | [0.427, 0.556] | 0.817 ± 0.065 | [0.777, 0.856] | 4 | 49 | 0.19 | 0.14 |
| A | mlp | 0.1 | 10 | 13 | 0.605 ± 0.127 | [0.529, 0.682] | 0.913 ± 0.032 | [0.894, 0.933] | 4 | 49 | 0.19 | 0.23 |
| A | mlp | 0.3 | 10 | 13 | 0.569 ± 0.124 | [0.494, 0.644] | 0.901 ± 0.034 | [0.881, 0.922] | 4 | 49 | 0.18 | 0.23 |
| A | mlp | 1 | 10 | 13 | 0.519 ± 0.141 | [0.433, 0.604] | 0.882 ± 0.042 | [0.856, 0.908] | 4 | 49 | 0.15 | 0.19 |
| A | mlp | 3 | 10 | 13 | 0.401 ± 0.106 | [0.337, 0.465] | 0.798 ± 0.064 | [0.759, 0.836] | 4 | 49 | 0.13 | 0.15 |
| A | mlp | 10 | 10 | 13 | 0.494 ± 0.128 | [0.417, 0.571] | 0.860 ± 0.037 | [0.838, 0.883] | 4 | 49 | 0.17 | 0.09 |
| A | mlp | 30 | 10 | 13 | 0.534 ± 0.135 | [0.453, 0.616] | 0.877 ± 0.035 | [0.856, 0.899] | 4 | 49 | 0.20 | 0.15 |
| A | mlp | 100 | 10 | 13 | 0.608 ± 0.097 | [0.549, 0.667] | 0.912 ± 0.035 | [0.892, 0.933] | 4 | 49 | 0.16 | 0.15 |
| B | linear | – | 0 | 13 | 0.172 ± 0.193 | [0.056, 0.289] | 0.572 ± 0.082 | [0.523, 0.622] | 4 | 5 | 0.02 | 0.06 |
| B | linear | – | 0.1 | 13 | 0.178 ± 0.185 | [0.067, 0.290] | 0.566 ± 0.089 | [0.512, 0.619] | 4 | 5 | 0.23 | 0.10 |
| B | linear | – | 1 | 13 | 0.245 ± 0.117 | [0.175, 0.316] | 0.592 ± 0.081 | [0.543, 0.642] | 4 | 5 | 0.15 | 0.07 |
| B | linear | – | 10 | 13 | 0.243 ± 0.176 | [0.136, 0.349] | 0.609 ± 0.081 | [0.559, 0.658] | 4 | 5 | 0.22 | 0.10 |
| B | mlp | – | 0 | 13 | 0.681 ± 0.054 | [0.648, 0.714] | 0.948 ± 0.020 | [0.936, 0.960] | 4 | 49 | 0.02 | 0.33 |
| B | mlp | – | 0.1 | 13 | 0.684 ± 0.091 | [0.629, 0.738] | 0.948 ± 0.024 | [0.934, 0.963] | 4 | 49 | 0.23 | 0.18 |
| B | mlp | – | 1 | 13 | 0.704 ± 0.092 | [0.648, 0.760] | 0.956 ± 0.028 | [0.939, 0.973] | 4 | 49 | 0.15 | 0.31 |
| B | mlp | – | 10 | 13 | 0.582 ± 0.148 | [0.492, 0.671] | 0.920 ± 0.033 | [0.900, 0.940] | 4 | 49 | 0.22 | 0.15 |
| C | linear | – | – | 13 | 0.155 ± 0.154 | [0.062, 0.248] | 0.554 ± 0.097 | [0.495, 0.613] | 4 | 5 | 0.00 | 0.07 |
| C | mlp | – | – | 13 | 0.864 ± 0.077 | [0.818, 0.911] | 0.982 ± 0.023 | [0.968, 0.996] | 4 | 49 | 0.00 | 0.15 |
| D | linear | – | – | 13 | -0.032 ± 0.159 | [-0.128, 0.065] | 0.556 ± 0.070 | [0.514, 0.598] | 4 | 5 | 0.01 | 0.06 |
| D | mlp | – | – | 13 | 0.034 ± 0.135 | [-0.048, 0.116] | 0.571 ± 0.081 | [0.522, 0.620] | 4 | 49 | 0.01 | 0.11 |
| D-hybrid | hybrid | – | – | 13 | 0.229 ± 0.128 | [0.152, 0.306] | 0.689 ± 0.066 | [0.649, 0.729] | 4 | 29 | 0.00 | 4.53 |

#### A against B (MCC, paired by fold)

Mean ΔMCC = A − B over the folds (Holm-adjusted p; ✓: at most α).

Head `linear`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.032 (1.0000) | +0.037 (1.0000) | -0.016 (1.0000) | -0.016 (1.0000) |
| 0.3 | +0.040 (1.0000) | +0.068 (1.0000) | -0.027 (1.0000) | -0.042 (1.0000) |
| 1 | +0.130 (1.0000) | +0.154 (1.0000) | -0.001 (1.0000) | -0.001 (1.0000) |
| 3 | +0.209 (0.4639) | +0.216 (0.1978) | -0.032 (1.0000) | -0.043 (1.0000) |
| 10 | +0.222 (0.2256) | +0.181 (0.9534) | +0.053 (1.0000) | +0.006 (1.0000) |
| 30 | +0.101 (1.0000) | +0.036 (1.0000) | -0.073 (1.0000) | +0.011 (1.0000) |
| 100 | +0.142 (1.0000) | +0.134 (1.0000) | +0.015 (1.0000) | -0.023 (1.0000) |

Head `mlp`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.033 (1.0000) | -0.004 (1.0000) | -0.009 (1.0000) | +0.024 (1.0000) |
| 0.3 | -0.088 (0.7769) | -0.109 (1.0000) | -0.083 (1.0000) | -0.012 (1.0000) |
| 1 | -0.216 (0.0464 ✓) | -0.224 (0.0474 ✓) | -0.454 (0.0464 ✓) | -0.063 (1.0000) |
| 3 | -0.189 (0.0474 ✓) | -0.167 (0.1050) | -0.161 (0.1978) | -0.180 (0.0464 ✓) |
| 10 | -0.105 (0.0474 ✓) | -0.039 (1.0000) | -0.061 (1.0000) | -0.088 (1.0000) |
| 30 | -0.419 (0.0464 ✓) | -0.397 (0.0464 ✓) | -0.355 (0.0474 ✓) | -0.047 (1.0000) |
| 100 | -0.298 (0.0464 ✓) | -0.286 (0.0464 ✓) | -0.213 (0.0474 ✓) | +0.027 (1.0000) |

#### Every arm against the classical baselines (MCC, paired by fold)

Mean ΔMCC = arm − baseline (Holm-adjusted p; ✓: at most α).

Arm A, head `linear`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.049 (1.0000) | +0.060 (1.0000) | +0.074 (1.0000) | +0.072 (1.0000) |
| 0.3 | +0.057 (1.0000) | +0.091 (1.0000) | +0.063 (1.0000) | +0.045 (1.0000) |
| 1 | +0.147 (1.0000) | +0.178 (1.0000) | +0.089 (1.0000) | +0.086 (1.0000) |
| 3 | +0.226 (0.6042) | +0.240 (0.1401) | +0.059 (1.0000) | +0.045 (1.0000) |
| 10 | +0.239 (0.1050) | +0.204 (0.1050) | +0.144 (0.7769) | +0.093 (1.0000) |
| 30 | +0.118 (1.0000) | +0.059 (1.0000) | +0.017 (1.0000) | +0.098 (1.0000) |
| 100 | +0.159 (0.9534) | +0.157 (1.0000) | +0.105 (1.0000) | +0.065 (1.0000) |

Arm A, head `linear`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.661 (0.0464 ✓) | -0.650 (0.0464 ✓) | -0.635 (0.0464 ✓) | -0.638 (0.0464 ✓) |
| 0.3 | -0.652 (0.0464 ✓) | -0.619 (0.0464 ✓) | -0.646 (0.0464 ✓) | -0.664 (0.0464 ✓) |
| 1 | -0.563 (0.0464 ✓) | -0.532 (0.0464 ✓) | -0.620 (0.0464 ✓) | -0.623 (0.0464 ✓) |
| 3 | -0.484 (0.0464 ✓) | -0.470 (0.0464 ✓) | -0.651 (0.0464 ✓) | -0.665 (0.0464 ✓) |
| 10 | -0.470 (0.0464 ✓) | -0.506 (0.0464 ✓) | -0.566 (0.0464 ✓) | -0.616 (0.0464 ✓) |
| 30 | -0.591 (0.0464 ✓) | -0.651 (0.0464 ✓) | -0.692 (0.0464 ✓) | -0.611 (0.0464 ✓) |
| 100 | -0.550 (0.0464 ✓) | -0.552 (0.0464 ✓) | -0.604 (0.0464 ✓) | -0.645 (0.0464 ✓) |

Arm A, head `mlp`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.559 (0.0464 ✓) | +0.525 (0.0464 ✓) | +0.540 (0.0464 ✓) | +0.450 (0.0464 ✓) |
| 0.3 | +0.438 (0.0464 ✓) | +0.419 (0.0464 ✓) | +0.466 (0.0464 ✓) | +0.414 (0.0464 ✓) |
| 1 | +0.310 (0.0645) | +0.305 (0.1978) | +0.095 (1.0000) | +0.364 (0.0464 ✓) |
| 3 | +0.337 (0.0464 ✓) | +0.362 (0.0464 ✓) | +0.388 (0.0464 ✓) | +0.246 (0.1050) |
| 10 | +0.421 (0.0464 ✓) | +0.490 (0.0464 ✓) | +0.488 (0.0464 ✓) | +0.339 (0.0474 ✓) |
| 30 | +0.107 (0.7769) | +0.131 (1.0000) | +0.194 (0.9534) | +0.379 (0.0464 ✓) |
| 100 | +0.228 (0.0464 ✓) | +0.242 (0.1978) | +0.336 (0.0464 ✓) | +0.453 (0.0464 ✓) |

Arm A, head `mlp`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.151 (0.0474 ✓) | -0.185 (0.0850) | -0.170 (0.0474 ✓) | -0.259 (0.0464 ✓) |
| 0.3 | -0.271 (0.0464 ✓) | -0.290 (0.0464 ✓) | -0.243 (0.0464 ✓) | -0.295 (0.0464 ✓) |
| 1 | -0.400 (0.0464 ✓) | -0.405 (0.0464 ✓) | -0.614 (0.0464 ✓) | -0.346 (0.0464 ✓) |
| 3 | -0.373 (0.0464 ✓) | -0.348 (0.0464 ✓) | -0.321 (0.0464 ✓) | -0.463 (0.0464 ✓) |
| 10 | -0.289 (0.0464 ✓) | -0.220 (0.0464 ✓) | -0.222 (0.0464 ✓) | -0.371 (0.0464 ✓) |
| 30 | -0.602 (0.0464 ✓) | -0.578 (0.0464 ✓) | -0.515 (0.0464 ✓) | -0.330 (0.0464 ✓) |
| 100 | -0.481 (0.0464 ✓) | -0.467 (0.0464 ✓) | -0.373 (0.0464 ✓) | -0.256 (0.0464 ✓) |

The other arms:

| arm | against C/linear | against C/mlp |
|---|---|---|
| B/linear γ/Ω=0 | +0.017 (1.0000) | -0.692 (0.0464 ✓) |
| B/linear γ/Ω=0.1 | +0.023 (1.0000) | -0.686 (0.0464 ✓) |
| B/linear γ/Ω=1 | +0.090 (1.0000) | -0.619 (0.0464 ✓) |
| B/linear γ/Ω=10 | +0.088 (1.0000) | -0.622 (0.0464 ✓) |
| B/mlp γ/Ω=0 | +0.526 (0.0464 ✓) | -0.183 (0.0464 ✓) |
| B/mlp γ/Ω=0.1 | +0.529 (0.0464 ✓) | -0.181 (0.0474 ✓) |
| B/mlp γ/Ω=1 | +0.549 (0.0464 ✓) | -0.160 (0.0464 ✓) |
| B/mlp γ/Ω=10 | +0.427 (0.0464 ✓) | -0.283 (0.0464 ✓) |
| D/linear | -0.187 (1.0000) | -0.896 (0.0464 ✓) |
| D/mlp | -0.121 (1.0000) | -0.831 (0.0464 ✓) |
| D-hybrid/hybrid | +0.074 (1.0000) | -0.636 (0.0464 ✓) |

### Positive control, grid

Folds per row: [13].  Holm family: 190 comparisons at α = 0.05, first threshold α/m = 2.63e-04.

#### Scores

| arm | head | V/Ω | γ/Ω | n | MCC mean ± sd | MCC 95 % CI | PR-AUC mean ± sd | PR-AUC 95 % CI | dim | params | feature s | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | linear | 0.1 | 0 | 13 | 0.170 ± 0.155 | [0.076, 0.264] | 0.570 ± 0.119 | [0.498, 0.641] | 4 | 5 | 0.04 | 0.08 |
| A | linear | 0.3 | 0 | 13 | 0.193 ± 0.115 | [0.124, 0.263] | 0.609 ± 0.095 | [0.551, 0.666] | 4 | 5 | 0.02 | 0.06 |
| A | linear | 1 | 0 | 13 | 0.273 ± 0.205 | [0.149, 0.397] | 0.756 ± 0.084 | [0.705, 0.807] | 4 | 5 | 0.03 | 0.11 |
| A | linear | 3 | 0 | 13 | 0.391 ± 0.128 | [0.314, 0.469] | 0.778 ± 0.091 | [0.723, 0.833] | 4 | 5 | 0.03 | 0.11 |
| A | linear | 10 | 0 | 13 | 0.379 ± 0.096 | [0.321, 0.437] | 0.712 ± 0.078 | [0.665, 0.760] | 4 | 5 | 0.02 | 0.10 |
| A | linear | 30 | 0 | 13 | 0.237 ± 0.102 | [0.175, 0.299] | 0.645 ± 0.081 | [0.596, 0.694] | 4 | 5 | 0.03 | 0.11 |
| A | linear | 100 | 0 | 13 | 0.308 ± 0.125 | [0.233, 0.384] | 0.704 ± 0.103 | [0.642, 0.767] | 4 | 5 | 0.03 | 0.15 |
| A | linear | 0.1 | 0.1 | 13 | 0.159 ± 0.154 | [0.066, 0.252] | 0.570 ± 0.119 | [0.498, 0.642] | 4 | 5 | 0.19 | 0.09 |
| A | linear | 0.3 | 0.1 | 13 | 0.209 ± 0.104 | [0.146, 0.272] | 0.619 ± 0.090 | [0.565, 0.674] | 4 | 5 | 0.18 | 0.08 |
| A | linear | 1 | 0.1 | 13 | 0.358 ± 0.170 | [0.255, 0.460] | 0.748 ± 0.083 | [0.698, 0.798] | 4 | 5 | 0.18 | 0.06 |
| A | linear | 3 | 0.1 | 13 | 0.335 ± 0.164 | [0.236, 0.435] | 0.719 ± 0.144 | [0.632, 0.806] | 4 | 5 | 0.23 | 0.14 |
| A | linear | 10 | 0.1 | 13 | 0.363 ± 0.125 | [0.288, 0.439] | 0.707 ± 0.075 | [0.661, 0.752] | 4 | 5 | 0.17 | 0.08 |
| A | linear | 30 | 0.1 | 13 | 0.231 ± 0.111 | [0.164, 0.298] | 0.640 ± 0.075 | [0.595, 0.685] | 4 | 5 | 0.22 | 0.09 |
| A | linear | 100 | 0.1 | 13 | 0.329 ± 0.123 | [0.254, 0.403] | 0.711 ± 0.106 | [0.647, 0.775] | 4 | 5 | 0.17 | 0.08 |
| A | linear | 0.1 | 1 | 13 | 0.199 ± 0.136 | [0.117, 0.281] | 0.577 ± 0.097 | [0.519, 0.636] | 4 | 5 | 0.15 | 0.08 |
| A | linear | 0.3 | 1 | 13 | 0.167 ± 0.121 | [0.094, 0.240] | 0.596 ± 0.102 | [0.534, 0.658] | 4 | 5 | 0.18 | 0.08 |
| A | linear | 1 | 1 | 13 | 0.211 ± 0.159 | [0.115, 0.307] | 0.712 ± 0.076 | [0.666, 0.757] | 4 | 5 | 0.17 | 0.06 |
| A | linear | 3 | 1 | 13 | 0.185 ± 0.190 | [0.071, 0.300] | 0.584 ± 0.147 | [0.495, 0.673] | 4 | 5 | 0.21 | 0.12 |
| A | linear | 10 | 1 | 13 | 0.316 ± 0.077 | [0.269, 0.362] | 0.676 ± 0.069 | [0.634, 0.718] | 4 | 5 | 0.16 | 0.07 |
| A | linear | 30 | 1 | 13 | 0.131 ± 0.189 | [0.016, 0.245] | 0.604 ± 0.113 | [0.535, 0.672] | 4 | 5 | 0.18 | 0.06 |
| A | linear | 100 | 1 | 13 | 0.317 ± 0.149 | [0.227, 0.407] | 0.729 ± 0.094 | [0.673, 0.786] | 4 | 5 | 0.22 | 0.12 |
| A | linear | 0.1 | 10 | 13 | 0.188 ± 0.139 | [0.104, 0.272] | 0.561 ± 0.068 | [0.520, 0.603] | 4 | 5 | 0.23 | 0.12 |
| A | linear | 0.3 | 10 | 13 | 0.184 ± 0.167 | [0.083, 0.285] | 0.545 ± 0.069 | [0.503, 0.587] | 4 | 5 | 0.24 | 0.08 |
| A | linear | 1 | 10 | 13 | 0.196 ± 0.161 | [0.099, 0.293] | 0.553 ± 0.067 | [0.512, 0.593] | 4 | 5 | 0.17 | 0.10 |
| A | linear | 3 | 10 | 13 | 0.129 ± 0.174 | [0.024, 0.234] | 0.581 ± 0.071 | [0.538, 0.624] | 4 | 5 | 0.16 | 0.07 |
| A | linear | 10 | 10 | 13 | 0.150 ± 0.124 | [0.075, 0.225] | 0.555 ± 0.078 | [0.508, 0.603] | 4 | 5 | 0.22 | 0.09 |
| A | linear | 30 | 10 | 13 | 0.149 ± 0.143 | [0.063, 0.235] | 0.548 ± 0.069 | [0.506, 0.590] | 4 | 5 | 0.17 | 0.09 |
| A | linear | 100 | 10 | 13 | 0.172 ± 0.120 | [0.099, 0.244] | 0.562 ± 0.092 | [0.506, 0.617] | 4 | 5 | 0.20 | 0.14 |
| A | mlp | 0.1 | 0 | 13 | 0.614 ± 0.076 | [0.567, 0.660] | 0.917 ± 0.041 | [0.893, 0.942] | 4 | 49 | 0.04 | 0.17 |
| A | mlp | 0.3 | 0 | 13 | 0.526 ± 0.116 | [0.456, 0.596] | 0.884 ± 0.040 | [0.859, 0.908] | 4 | 49 | 0.02 | 0.18 |
| A | mlp | 1 | 0 | 13 | 0.454 ± 0.134 | [0.373, 0.535] | 0.836 ± 0.044 | [0.809, 0.863] | 4 | 49 | 0.03 | 0.22 |
| A | mlp | 3 | 0 | 13 | 0.483 ± 0.113 | [0.415, 0.552] | 0.845 ± 0.052 | [0.814, 0.877] | 4 | 49 | 0.03 | 0.21 |
| A | mlp | 10 | 0 | 13 | 0.539 ± 0.129 | [0.461, 0.617] | 0.859 ± 0.055 | [0.826, 0.892] | 4 | 49 | 0.02 | 0.13 |
| A | mlp | 30 | 0 | 13 | 0.283 ± 0.104 | [0.220, 0.346] | 0.734 ± 0.071 | [0.691, 0.777] | 4 | 49 | 0.03 | 0.26 |
| A | mlp | 100 | 0 | 13 | 0.418 ± 0.182 | [0.308, 0.528] | 0.796 ± 0.095 | [0.738, 0.854] | 4 | 49 | 0.03 | 0.17 |
| A | mlp | 0.1 | 0.1 | 13 | 0.631 ± 0.122 | [0.557, 0.705] | 0.921 ± 0.042 | [0.896, 0.946] | 4 | 49 | 0.19 | 0.17 |
| A | mlp | 0.3 | 0.1 | 13 | 0.526 ± 0.095 | [0.469, 0.583] | 0.891 ± 0.045 | [0.863, 0.918] | 4 | 49 | 0.18 | 0.24 |
| A | mlp | 1 | 0.1 | 13 | 0.449 ± 0.174 | [0.344, 0.554] | 0.833 ± 0.067 | [0.793, 0.874] | 4 | 49 | 0.18 | 0.30 |
| A | mlp | 3 | 0.1 | 13 | 0.528 ± 0.105 | [0.465, 0.591] | 0.857 ± 0.044 | [0.830, 0.883] | 4 | 49 | 0.23 | 0.17 |
| A | mlp | 10 | 0.1 | 13 | 0.593 ± 0.141 | [0.507, 0.678] | 0.885 ± 0.036 | [0.863, 0.906] | 4 | 49 | 0.17 | 0.18 |
| A | mlp | 30 | 0.1 | 13 | 0.342 ± 0.129 | [0.264, 0.420] | 0.730 ± 0.071 | [0.687, 0.774] | 4 | 49 | 0.22 | 0.14 |
| A | mlp | 100 | 0.1 | 13 | 0.405 ± 0.160 | [0.308, 0.502] | 0.801 ± 0.078 | [0.754, 0.848] | 4 | 49 | 0.17 | 0.19 |
| A | mlp | 0.1 | 1 | 13 | 0.612 ± 0.098 | [0.553, 0.671] | 0.925 ± 0.035 | [0.904, 0.946] | 4 | 49 | 0.15 | 0.19 |
| A | mlp | 0.3 | 1 | 13 | 0.567 ± 0.126 | [0.491, 0.643] | 0.901 ± 0.044 | [0.874, 0.927] | 4 | 49 | 0.18 | 0.23 |
| A | mlp | 1 | 1 | 13 | 0.348 ± 0.147 | [0.259, 0.436] | 0.741 ± 0.075 | [0.696, 0.787] | 4 | 49 | 0.17 | 0.15 |
| A | mlp | 3 | 1 | 13 | 0.517 ± 0.153 | [0.425, 0.609] | 0.836 ± 0.071 | [0.794, 0.879] | 4 | 49 | 0.21 | 0.15 |
| A | mlp | 10 | 1 | 13 | 0.568 ± 0.162 | [0.470, 0.665] | 0.887 ± 0.069 | [0.846, 0.929] | 4 | 49 | 0.16 | 0.18 |
| A | mlp | 30 | 1 | 13 | 0.323 ± 0.143 | [0.237, 0.410] | 0.742 ± 0.079 | [0.694, 0.789] | 4 | 49 | 0.18 | 0.20 |
| A | mlp | 100 | 1 | 13 | 0.450 ± 0.081 | [0.401, 0.499] | 0.805 ± 0.059 | [0.769, 0.841] | 4 | 49 | 0.22 | 0.36 |
| A | mlp | 0.1 | 10 | 13 | 0.571 ± 0.118 | [0.500, 0.642] | 0.901 ± 0.045 | [0.874, 0.929] | 4 | 49 | 0.23 | 0.24 |
| A | mlp | 0.3 | 10 | 13 | 0.550 ± 0.110 | [0.484, 0.617] | 0.897 ± 0.042 | [0.871, 0.922] | 4 | 49 | 0.24 | 0.20 |
| A | mlp | 1 | 10 | 13 | 0.525 ± 0.113 | [0.457, 0.594] | 0.875 ± 0.047 | [0.847, 0.904] | 4 | 49 | 0.17 | 0.11 |
| A | mlp | 3 | 10 | 13 | 0.399 ± 0.130 | [0.320, 0.477] | 0.798 ± 0.068 | [0.757, 0.840] | 4 | 49 | 0.16 | 0.13 |
| A | mlp | 10 | 10 | 13 | 0.484 ± 0.090 | [0.430, 0.538] | 0.863 ± 0.050 | [0.833, 0.893] | 4 | 49 | 0.22 | 0.18 |
| A | mlp | 30 | 10 | 13 | 0.513 ± 0.097 | [0.455, 0.572] | 0.870 ± 0.051 | [0.839, 0.901] | 4 | 49 | 0.17 | 0.20 |
| A | mlp | 100 | 10 | 13 | 0.582 ± 0.100 | [0.522, 0.643] | 0.908 ± 0.041 | [0.883, 0.933] | 4 | 49 | 0.20 | 0.18 |
| B | linear | – | 0 | 13 | 0.147 ± 0.172 | [0.043, 0.251] | 0.591 ± 0.105 | [0.527, 0.654] | 4 | 5 | 0.03 | 0.07 |
| B | linear | – | 0.1 | 13 | 0.136 ± 0.178 | [0.029, 0.244] | 0.569 ± 0.117 | [0.499, 0.640] | 4 | 5 | 0.15 | 0.10 |
| B | linear | – | 1 | 13 | 0.150 ± 0.131 | [0.071, 0.229] | 0.554 ± 0.098 | [0.495, 0.613] | 4 | 5 | 0.16 | 0.07 |
| B | linear | – | 10 | 13 | 0.183 ± 0.166 | [0.083, 0.283] | 0.546 ± 0.066 | [0.507, 0.586] | 4 | 5 | 0.22 | 0.08 |
| B | mlp | – | 0 | 13 | 0.633 ± 0.118 | [0.562, 0.704] | 0.918 ± 0.037 | [0.896, 0.941] | 4 | 49 | 0.03 | 0.35 |
| B | mlp | – | 0.1 | 13 | 0.629 ± 0.139 | [0.545, 0.713] | 0.917 ± 0.040 | [0.893, 0.941] | 4 | 49 | 0.15 | 0.20 |
| B | mlp | – | 1 | 13 | 0.659 ± 0.122 | [0.586, 0.733] | 0.931 ± 0.038 | [0.908, 0.954] | 4 | 49 | 0.16 | 0.26 |
| B | mlp | – | 10 | 13 | 0.549 ± 0.104 | [0.486, 0.611] | 0.904 ± 0.037 | [0.882, 0.927] | 4 | 49 | 0.22 | 0.20 |
| C | linear | – | – | 13 | 0.203 ± 0.163 | [0.104, 0.301] | 0.620 ± 0.097 | [0.561, 0.678] | 4 | 5 | 0.00 | 0.06 |
| C | mlp | – | – | 13 | 0.825 ± 0.100 | [0.764, 0.885] | 0.977 ± 0.018 | [0.966, 0.988] | 4 | 49 | 0.00 | 0.16 |
| D | linear | – | – | 13 | 0.043 ± 0.168 | [-0.058, 0.145] | 0.547 ± 0.069 | [0.505, 0.588] | 4 | 5 | 0.01 | 0.08 |
| D | mlp | – | – | 13 | 0.064 ± 0.151 | [-0.028, 0.155] | 0.545 ± 0.082 | [0.495, 0.594] | 4 | 49 | 0.01 | 0.07 |
| D-hybrid | hybrid | – | – | 13 | 0.173 ± 0.143 | [0.087, 0.259] | 0.657 ± 0.070 | [0.615, 0.699] | 4 | 29 | 0.00 | 3.54 |

#### A against B (MCC, paired by fold)

Mean ΔMCC = A − B over the folds (Holm-adjusted p; ✓: at most α).

Head `linear`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.023 (1.0000) | +0.022 (1.0000) | +0.049 (1.0000) | +0.005 (1.0000) |
| 0.3 | +0.046 (1.0000) | +0.072 (1.0000) | +0.017 (1.0000) | +0.002 (1.0000) |
| 1 | +0.126 (1.0000) | +0.221 (0.6687) | +0.061 (1.0000) | +0.013 (1.0000) |
| 3 | +0.244 (0.1111) | +0.199 (0.8608) | +0.036 (1.0000) | -0.054 (1.0000) |
| 10 | +0.232 (0.0710) | +0.227 (0.2124) | +0.166 (0.0464 ✓) | -0.033 (1.0000) |
| 30 | +0.090 (1.0000) | +0.095 (1.0000) | -0.019 (1.0000) | -0.034 (1.0000) |
| 100 | +0.161 (1.0000) | +0.192 (1.0000) | +0.167 (1.0000) | -0.011 (1.0000) |

Head `mlp`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.019 (1.0000) | +0.003 (1.0000) | -0.047 (1.0000) | +0.022 (1.0000) |
| 0.3 | -0.108 (1.0000) | -0.103 (1.0000) | -0.092 (0.8608) | +0.001 (1.0000) |
| 1 | -0.179 (0.3896) | -0.180 (0.8608) | -0.312 (0.0503) | -0.023 (1.0000) |
| 3 | -0.150 (1.0000) | -0.101 (1.0000) | -0.143 (1.0000) | -0.150 (0.0464 ✓) |
| 10 | -0.094 (1.0000) | -0.036 (1.0000) | -0.092 (1.0000) | -0.065 (0.9644) |
| 30 | -0.350 (0.0464 ✓) | -0.287 (0.0464 ✓) | -0.336 (0.0503) | -0.035 (1.0000) |
| 100 | -0.215 (0.1538) | -0.224 (0.1538) | -0.209 (0.1538) | +0.033 (1.0000) |

#### Every arm against the classical baselines (MCC, paired by fold)

Mean ΔMCC = arm − baseline (Holm-adjusted p; ✓: at most α).

Arm A, head `linear`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.033 (1.0000) | -0.044 (1.0000) | -0.004 (1.0000) | -0.015 (1.0000) |
| 0.3 | -0.009 (1.0000) | +0.006 (1.0000) | -0.035 (1.0000) | -0.018 (1.0000) |
| 1 | +0.070 (1.0000) | +0.155 (1.0000) | +0.008 (1.0000) | -0.007 (1.0000) |
| 3 | +0.189 (0.0710) | +0.133 (1.0000) | -0.017 (1.0000) | -0.074 (1.0000) |
| 10 | +0.176 (1.0000) | +0.161 (1.0000) | +0.113 (1.0000) | -0.053 (1.0000) |
| 30 | +0.035 (1.0000) | +0.028 (1.0000) | -0.072 (1.0000) | -0.054 (1.0000) |
| 100 | +0.106 (1.0000) | +0.126 (1.0000) | +0.115 (1.0000) | -0.031 (1.0000) |

Arm A, head `linear`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.655 (0.0464 ✓) | -0.666 (0.0464 ✓) | -0.626 (0.0464 ✓) | -0.637 (0.0464 ✓) |
| 0.3 | -0.632 (0.0464 ✓) | -0.616 (0.0464 ✓) | -0.658 (0.0464 ✓) | -0.641 (0.0464 ✓) |
| 1 | -0.552 (0.0464 ✓) | -0.467 (0.0464 ✓) | -0.614 (0.0464 ✓) | -0.629 (0.0464 ✓) |
| 3 | -0.434 (0.0464 ✓) | -0.490 (0.0464 ✓) | -0.639 (0.0464 ✓) | -0.696 (0.0464 ✓) |
| 10 | -0.446 (0.0464 ✓) | -0.462 (0.0464 ✓) | -0.509 (0.0464 ✓) | -0.675 (0.0464 ✓) |
| 30 | -0.588 (0.0464 ✓) | -0.594 (0.0464 ✓) | -0.694 (0.0464 ✓) | -0.676 (0.0464 ✓) |
| 100 | -0.516 (0.0464 ✓) | -0.496 (0.0464 ✓) | -0.508 (0.0464 ✓) | -0.653 (0.0464 ✓) |

Arm A, head `mlp`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.411 (0.0464 ✓) | +0.429 (0.0464 ✓) | +0.410 (0.0464 ✓) | +0.368 (0.0503) |
| 0.3 | +0.323 (0.0464 ✓) | +0.324 (0.0464 ✓) | +0.365 (0.0464 ✓) | +0.348 (0.0464 ✓) |
| 1 | +0.251 (0.2124) | +0.246 (1.0000) | +0.145 (1.0000) | +0.323 (0.0710) |
| 3 | +0.281 (0.0464 ✓) | +0.326 (0.0464 ✓) | +0.314 (0.0464 ✓) | +0.196 (0.2124) |
| 10 | +0.337 (0.0710) | +0.390 (0.0503) | +0.365 (0.0464 ✓) | +0.281 (0.0464 ✓) |
| 30 | +0.080 (1.0000) | +0.139 (1.0000) | +0.121 (1.0000) | +0.311 (0.0464 ✓) |
| 100 | +0.216 (1.0000) | +0.203 (1.0000) | +0.248 (0.0710) | +0.380 (0.0464 ✓) |

Arm A, head `mlp`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.211 (0.0464 ✓) | -0.194 (0.0464 ✓) | -0.213 (0.0464 ✓) | -0.254 (0.0464 ✓) |
| 0.3 | -0.299 (0.0464 ✓) | -0.299 (0.0464 ✓) | -0.258 (0.0464 ✓) | -0.275 (0.0464 ✓) |
| 1 | -0.371 (0.0464 ✓) | -0.376 (0.0464 ✓) | -0.477 (0.0464 ✓) | -0.300 (0.0464 ✓) |
| 3 | -0.342 (0.0464 ✓) | -0.297 (0.0464 ✓) | -0.308 (0.0464 ✓) | -0.426 (0.0464 ✓) |
| 10 | -0.285 (0.0503) | -0.232 (0.0503) | -0.257 (0.0464 ✓) | -0.341 (0.0464 ✓) |
| 30 | -0.542 (0.0464 ✓) | -0.483 (0.0464 ✓) | -0.502 (0.0464 ✓) | -0.312 (0.0464 ✓) |
| 100 | -0.407 (0.0464 ✓) | -0.420 (0.0464 ✓) | -0.375 (0.0464 ✓) | -0.243 (0.0464 ✓) |

The other arms:

| arm | against C/linear | against C/mlp |
|---|---|---|
| B/linear γ/Ω=0 | -0.055 (1.0000) | -0.678 (0.0464 ✓) |
| B/linear γ/Ω=0.1 | -0.066 (1.0000) | -0.688 (0.0464 ✓) |
| B/linear γ/Ω=1 | -0.053 (1.0000) | -0.675 (0.0464 ✓) |
| B/linear γ/Ω=10 | -0.020 (1.0000) | -0.642 (0.0464 ✓) |
| B/mlp γ/Ω=0 | +0.431 (0.0464 ✓) | -0.192 (0.0464 ✓) |
| B/mlp γ/Ω=0.1 | +0.426 (0.0464 ✓) | -0.196 (0.0464 ✓) |
| B/mlp γ/Ω=1 | +0.457 (0.0464 ✓) | -0.165 (0.0464 ✓) |
| B/mlp γ/Ω=10 | +0.346 (0.0464 ✓) | -0.276 (0.0464 ✓) |
| D/linear | -0.159 (1.0000) | -0.782 (0.0464 ✓) |
| D/mlp | -0.139 (0.0898) | -0.761 (0.0464 ✓) |
| D-hybrid/hybrid | -0.030 (1.0000) | -0.652 (0.0464 ✓) |

### Iranian Churn, three geometries, 15 folds

Folds per row: [15].  Holm family: 526 comparisons at α = 0.05, first threshold α/m = 9.51e-05.

#### Scores

| arm | head | geometry | V/Ω | γ/Ω | n | MCC mean ± sd | MCC 95 % CI | PR-AUC mean ± sd | PR-AUC 95 % CI | dim | params | feature s | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | linear | chain | 0.1 | 0 | 15 | 0.559 ± 0.061 | [0.525, 0.592] | 0.562 ± 0.072 | [0.522, 0.602] | 4 | 5 | 0.15 | 0.46 |
| A | linear | chain | 0.3 | 0 | 15 | 0.553 ± 0.057 | [0.522, 0.585] | 0.557 ± 0.071 | [0.518, 0.596] | 4 | 5 | 0.14 | 0.39 |
| A | linear | chain | 1 | 0 | 15 | 0.382 ± 0.081 | [0.337, 0.426] | 0.483 ± 0.064 | [0.448, 0.519] | 4 | 5 | 0.14 | 0.24 |
| A | linear | chain | 3 | 0 | 15 | 0.487 ± 0.060 | [0.454, 0.520] | 0.562 ± 0.085 | [0.515, 0.609] | 4 | 5 | 0.14 | 0.36 |
| A | linear | chain | 10 | 0 | 15 | 0.514 ± 0.064 | [0.478, 0.549] | 0.584 ± 0.074 | [0.542, 0.625] | 4 | 5 | 0.14 | 0.27 |
| A | linear | chain | 30 | 0 | 15 | 0.532 ± 0.062 | [0.497, 0.567] | 0.582 ± 0.069 | [0.543, 0.620] | 4 | 5 | 0.14 | 0.30 |
| A | linear | chain | 100 | 0 | 15 | 0.456 ± 0.079 | [0.412, 0.499] | 0.448 ± 0.064 | [0.412, 0.483] | 4 | 5 | 0.13 | 0.50 |
| A | linear | chain | 0.1 | 0.1 | 15 | 0.560 ± 0.059 | [0.528, 0.593] | 0.564 ± 0.073 | [0.523, 0.604] | 4 | 5 | 1.94 | 0.39 |
| A | linear | chain | 0.3 | 0.1 | 15 | 0.557 ± 0.063 | [0.522, 0.591] | 0.559 ± 0.071 | [0.519, 0.598] | 4 | 5 | 1.94 | 0.38 |
| A | linear | chain | 1 | 0.1 | 15 | 0.412 ± 0.058 | [0.380, 0.444] | 0.490 ± 0.067 | [0.453, 0.527] | 4 | 5 | 1.95 | 0.25 |
| A | linear | chain | 3 | 0.1 | 15 | 0.503 ± 0.061 | [0.469, 0.537] | 0.556 ± 0.074 | [0.515, 0.597] | 4 | 5 | 1.92 | 0.32 |
| A | linear | chain | 10 | 0.1 | 15 | 0.523 ± 0.056 | [0.492, 0.554] | 0.586 ± 0.073 | [0.545, 0.626] | 4 | 5 | 1.95 | 0.27 |
| A | linear | chain | 30 | 0.1 | 15 | 0.527 ± 0.052 | [0.499, 0.556] | 0.581 ± 0.069 | [0.542, 0.619] | 4 | 5 | 1.94 | 0.26 |
| A | linear | chain | 100 | 0.1 | 15 | 0.444 ± 0.060 | [0.411, 0.478] | 0.462 ± 0.077 | [0.419, 0.504] | 4 | 5 | 1.90 | 0.52 |
| A | linear | chain | 0.1 | 1 | 15 | 0.556 ± 0.061 | [0.522, 0.589] | 0.566 ± 0.073 | [0.526, 0.607] | 4 | 5 | 1.91 | 0.46 |
| A | linear | chain | 0.3 | 1 | 15 | 0.556 ± 0.059 | [0.523, 0.588] | 0.568 ± 0.067 | [0.531, 0.605] | 4 | 5 | 1.94 | 0.45 |
| A | linear | chain | 1 | 1 | 15 | 0.481 ± 0.061 | [0.447, 0.514] | 0.537 ± 0.061 | [0.503, 0.571] | 4 | 5 | 1.95 | 0.41 |
| A | linear | chain | 3 | 1 | 15 | 0.548 ± 0.061 | [0.514, 0.582] | 0.551 ± 0.064 | [0.516, 0.586] | 4 | 5 | 1.94 | 0.45 |
| A | linear | chain | 10 | 1 | 15 | 0.554 ± 0.057 | [0.522, 0.585] | 0.583 ± 0.072 | [0.543, 0.623] | 4 | 5 | 1.94 | 0.34 |
| A | linear | chain | 30 | 1 | 15 | 0.542 ± 0.057 | [0.510, 0.574] | 0.583 ± 0.070 | [0.544, 0.621] | 4 | 5 | 1.94 | 0.37 |
| A | linear | chain | 100 | 1 | 15 | 0.431 ± 0.097 | [0.378, 0.485] | 0.538 ± 0.068 | [0.500, 0.575] | 4 | 5 | 1.90 | 0.53 |
| A | linear | chain | 0.1 | 10 | 15 | 0.552 ± 0.062 | [0.517, 0.586] | 0.588 ± 0.066 | [0.552, 0.625] | 4 | 5 | 1.92 | 0.46 |
| A | linear | chain | 0.3 | 10 | 15 | 0.549 ± 0.068 | [0.512, 0.587] | 0.587 ± 0.066 | [0.550, 0.623] | 4 | 5 | 1.93 | 0.42 |
| A | linear | chain | 1 | 10 | 15 | 0.544 ± 0.066 | [0.507, 0.581] | 0.584 ± 0.070 | [0.546, 0.623] | 4 | 5 | 1.94 | 0.46 |
| A | linear | chain | 3 | 10 | 15 | 0.516 ± 0.061 | [0.482, 0.550] | 0.564 ± 0.068 | [0.526, 0.601] | 4 | 5 | 1.97 | 0.41 |
| A | linear | chain | 10 | 10 | 15 | 0.543 ± 0.076 | [0.501, 0.585] | 0.583 ± 0.068 | [0.546, 0.621] | 4 | 5 | 1.96 | 0.54 |
| A | linear | chain | 30 | 10 | 15 | 0.548 ± 0.062 | [0.513, 0.582] | 0.587 ± 0.067 | [0.550, 0.624] | 4 | 5 | 1.93 | 0.46 |
| A | linear | chain | 100 | 10 | 15 | 0.548 ± 0.073 | [0.508, 0.588] | 0.585 ± 0.069 | [0.547, 0.623] | 4 | 5 | 1.94 | 0.55 |
| A | linear | ring | 0.1 | 0 | 15 | 0.556 ± 0.060 | [0.522, 0.589] | 0.564 ± 0.074 | [0.523, 0.605] | 4 | 5 | 0.14 | 0.40 |
| A | linear | ring | 0.3 | 0 | 15 | 0.531 ± 0.072 | [0.491, 0.571] | 0.557 ± 0.072 | [0.518, 0.597] | 4 | 5 | 0.14 | 0.39 |
| A | linear | ring | 1 | 0 | 15 | 0.128 ± 0.076 | [0.086, 0.171] | 0.197 ± 0.036 | [0.177, 0.217] | 4 | 5 | 0.15 | 0.45 |
| A | linear | ring | 3 | 0 | 15 | 0.549 ± 0.061 | [0.515, 0.583] | 0.569 ± 0.070 | [0.530, 0.608] | 4 | 5 | 0.16 | 0.42 |
| A | linear | ring | 10 | 0 | 15 | 0.295 ± 0.052 | [0.267, 0.324] | 0.445 ± 0.070 | [0.406, 0.483] | 4 | 5 | 0.14 | 0.44 |
| A | linear | ring | 30 | 0 | 15 | 0.223 ± 0.079 | [0.179, 0.266] | 0.282 ± 0.064 | [0.247, 0.318] | 4 | 5 | 0.14 | 0.41 |
| A | linear | ring | 100 | 0 | 15 | 0.264 ± 0.077 | [0.222, 0.307] | 0.344 ± 0.061 | [0.310, 0.378] | 4 | 5 | 0.15 | 0.39 |
| A | linear | ring | 0.1 | 0.1 | 15 | 0.552 ± 0.054 | [0.522, 0.582] | 0.565 ± 0.073 | [0.525, 0.606] | 4 | 5 | 1.94 | 0.43 |
| A | linear | ring | 0.3 | 0.1 | 15 | 0.532 ± 0.070 | [0.493, 0.570] | 0.560 ± 0.070 | [0.522, 0.599] | 4 | 5 | 1.95 | 0.34 |
| A | linear | ring | 1 | 0.1 | 15 | 0.133 ± 0.067 | [0.096, 0.171] | 0.218 ± 0.024 | [0.205, 0.231] | 4 | 5 | 1.99 | 0.56 |
| A | linear | ring | 3 | 0.1 | 15 | 0.534 ± 0.080 | [0.490, 0.579] | 0.563 ± 0.074 | [0.522, 0.604] | 4 | 5 | 2.05 | 0.33 |
| A | linear | ring | 10 | 0.1 | 15 | 0.332 ± 0.074 | [0.292, 0.373] | 0.473 ± 0.074 | [0.432, 0.514] | 4 | 5 | 2.07 | 0.37 |
| A | linear | ring | 30 | 0.1 | 15 | 0.235 ± 0.086 | [0.188, 0.283] | 0.296 ± 0.075 | [0.255, 0.338] | 4 | 5 | 2.05 | 0.39 |
| A | linear | ring | 100 | 0.1 | 15 | 0.268 ± 0.093 | [0.217, 0.320] | 0.359 ± 0.070 | [0.321, 0.398] | 4 | 5 | 2.04 | 0.44 |
| A | linear | ring | 0.1 | 1 | 15 | 0.553 ± 0.058 | [0.521, 0.585] | 0.567 ± 0.072 | [0.527, 0.607] | 4 | 5 | 1.95 | 0.39 |
| A | linear | ring | 0.3 | 1 | 15 | 0.541 ± 0.068 | [0.503, 0.579] | 0.569 ± 0.069 | [0.531, 0.608] | 4 | 5 | 2.00 | 0.40 |
| A | linear | ring | 1 | 1 | 15 | 0.284 ± 0.066 | [0.248, 0.321] | 0.437 ± 0.056 | [0.406, 0.468] | 4 | 5 | 2.05 | 0.38 |
| A | linear | ring | 3 | 1 | 15 | 0.499 ± 0.067 | [0.462, 0.536] | 0.569 ± 0.074 | [0.528, 0.609] | 4 | 5 | 2.06 | 0.36 |
| A | linear | ring | 10 | 1 | 15 | 0.433 ± 0.059 | [0.400, 0.466] | 0.571 ± 0.057 | [0.539, 0.602] | 4 | 5 | 2.05 | 0.36 |
| A | linear | ring | 30 | 1 | 15 | 0.415 ± 0.077 | [0.372, 0.457] | 0.523 ± 0.080 | [0.478, 0.567] | 4 | 5 | 2.05 | 0.38 |
| A | linear | ring | 100 | 1 | 15 | 0.389 ± 0.090 | [0.339, 0.439] | 0.544 ± 0.075 | [0.502, 0.585] | 4 | 5 | 2.04 | 0.42 |
| A | linear | ring | 0.1 | 10 | 15 | 0.551 ± 0.063 | [0.516, 0.586] | 0.586 ± 0.065 | [0.550, 0.622] | 4 | 5 | 1.93 | 0.45 |
| A | linear | ring | 0.3 | 10 | 15 | 0.549 ± 0.061 | [0.515, 0.583] | 0.586 ± 0.069 | [0.548, 0.624] | 4 | 5 | 1.97 | 0.53 |
| A | linear | ring | 1 | 10 | 15 | 0.523 ± 0.061 | [0.490, 0.557] | 0.581 ± 0.067 | [0.544, 0.618] | 4 | 5 | 2.03 | 0.49 |
| A | linear | ring | 3 | 10 | 15 | 0.373 ± 0.075 | [0.332, 0.414] | 0.507 ± 0.052 | [0.478, 0.535] | 4 | 5 | 2.06 | 0.30 |
| A | linear | ring | 10 | 10 | 15 | 0.506 ± 0.066 | [0.470, 0.543] | 0.571 ± 0.067 | [0.534, 0.609] | 4 | 5 | 2.05 | 0.69 |
| A | linear | ring | 30 | 10 | 15 | 0.512 ± 0.061 | [0.478, 0.545] | 0.574 ± 0.068 | [0.536, 0.611] | 4 | 5 | 2.07 | 0.53 |
| A | linear | ring | 100 | 10 | 15 | 0.556 ± 0.069 | [0.518, 0.594] | 0.589 ± 0.069 | [0.551, 0.627] | 4 | 5 | 2.00 | 0.52 |
| A | linear | grid-2x2 | 0.1 | 0 | 15 | 0.548 ± 0.054 | [0.518, 0.578] | 0.562 ± 0.074 | [0.520, 0.603] | 4 | 5 | 0.14 | 0.43 |
| A | linear | grid-2x2 | 0.3 | 0 | 15 | 0.523 ± 0.068 | [0.486, 0.561] | 0.568 ± 0.075 | [0.527, 0.609] | 4 | 5 | 0.14 | 0.43 |
| A | linear | grid-2x2 | 1 | 0 | 15 | 0.188 ± 0.089 | [0.139, 0.237] | 0.225 ± 0.060 | [0.192, 0.258] | 4 | 5 | 0.15 | 0.37 |
| A | linear | grid-2x2 | 3 | 0 | 15 | 0.168 ± 0.062 | [0.133, 0.202] | 0.249 ± 0.060 | [0.216, 0.282] | 4 | 5 | 0.15 | 0.22 |
| A | linear | grid-2x2 | 10 | 0 | 15 | 0.261 ± 0.054 | [0.231, 0.290] | 0.271 ± 0.050 | [0.243, 0.299] | 4 | 5 | 0.14 | 0.40 |
| A | linear | grid-2x2 | 30 | 0 | 15 | 0.291 ± 0.080 | [0.246, 0.335] | 0.299 ± 0.068 | [0.261, 0.336] | 4 | 5 | 0.13 | 0.33 |
| A | linear | grid-2x2 | 100 | 0 | 15 | 0.266 ± 0.076 | [0.224, 0.309] | 0.351 ± 0.063 | [0.316, 0.386] | 4 | 5 | 0.13 | 0.28 |
| A | linear | grid-2x2 | 0.1 | 0.1 | 15 | 0.551 ± 0.054 | [0.521, 0.581] | 0.563 ± 0.074 | [0.521, 0.604] | 4 | 5 | 1.98 | 0.43 |
| A | linear | grid-2x2 | 0.3 | 0.1 | 15 | 0.524 ± 0.062 | [0.489, 0.558] | 0.568 ± 0.074 | [0.527, 0.610] | 4 | 5 | 2.02 | 0.45 |
| A | linear | grid-2x2 | 1 | 0.1 | 15 | 0.176 ± 0.052 | [0.147, 0.205] | 0.210 ± 0.056 | [0.179, 0.241] | 4 | 5 | 1.98 | 0.48 |
| A | linear | grid-2x2 | 3 | 0.1 | 15 | 0.223 ± 0.065 | [0.187, 0.259] | 0.326 ± 0.069 | [0.288, 0.364] | 4 | 5 | 2.02 | 0.30 |
| A | linear | grid-2x2 | 10 | 0.1 | 15 | 0.272 ± 0.053 | [0.242, 0.301] | 0.285 ± 0.055 | [0.254, 0.315] | 4 | 5 | 1.94 | 0.47 |
| A | linear | grid-2x2 | 30 | 0.1 | 15 | 0.278 ± 0.093 | [0.227, 0.329] | 0.340 ± 0.071 | [0.301, 0.380] | 4 | 5 | 1.90 | 0.36 |
| A | linear | grid-2x2 | 100 | 0.1 | 15 | 0.281 ± 0.100 | [0.225, 0.336] | 0.361 ± 0.075 | [0.319, 0.402] | 4 | 5 | 1.90 | 0.36 |
| A | linear | grid-2x2 | 0.1 | 1 | 15 | 0.556 ± 0.065 | [0.520, 0.592] | 0.567 ± 0.071 | [0.527, 0.606] | 4 | 5 | 2.05 | 0.39 |
| A | linear | grid-2x2 | 0.3 | 1 | 15 | 0.544 ± 0.068 | [0.506, 0.581] | 0.572 ± 0.072 | [0.532, 0.612] | 4 | 5 | 1.97 | 0.63 |
| A | linear | grid-2x2 | 1 | 1 | 15 | 0.245 ± 0.065 | [0.209, 0.281] | 0.371 ± 0.073 | [0.330, 0.412] | 4 | 5 | 1.96 | 0.46 |
| A | linear | grid-2x2 | 3 | 1 | 15 | 0.513 ± 0.055 | [0.483, 0.544] | 0.599 ± 0.075 | [0.557, 0.641] | 4 | 5 | 1.92 | 0.34 |
| A | linear | grid-2x2 | 10 | 1 | 15 | 0.462 ± 0.063 | [0.427, 0.496] | 0.520 ± 0.063 | [0.485, 0.555] | 4 | 5 | 1.94 | 0.50 |
| A | linear | grid-2x2 | 30 | 1 | 15 | 0.384 ± 0.122 | [0.317, 0.452] | 0.526 ± 0.077 | [0.483, 0.569] | 4 | 5 | 1.90 | 0.28 |
| A | linear | grid-2x2 | 100 | 1 | 15 | 0.393 ± 0.094 | [0.342, 0.445] | 0.540 ± 0.070 | [0.501, 0.579] | 4 | 5 | 1.90 | 0.43 |
| A | linear | grid-2x2 | 0.1 | 10 | 15 | 0.549 ± 0.065 | [0.513, 0.585] | 0.587 ± 0.065 | [0.551, 0.623] | 4 | 5 | 1.97 | 0.45 |
| A | linear | grid-2x2 | 0.3 | 10 | 15 | 0.546 ± 0.065 | [0.510, 0.582] | 0.586 ± 0.069 | [0.547, 0.624] | 4 | 5 | 1.96 | 0.49 |
| A | linear | grid-2x2 | 1 | 10 | 15 | 0.523 ± 0.061 | [0.489, 0.557] | 0.579 ± 0.065 | [0.543, 0.615] | 4 | 5 | 1.98 | 0.51 |
| A | linear | grid-2x2 | 3 | 10 | 15 | 0.374 ± 0.062 | [0.339, 0.408] | 0.491 ± 0.045 | [0.466, 0.515] | 4 | 5 | 1.92 | 0.37 |
| A | linear | grid-2x2 | 10 | 10 | 15 | 0.507 ± 0.063 | [0.472, 0.542] | 0.570 ± 0.067 | [0.533, 0.608] | 4 | 5 | 1.91 | 0.57 |
| A | linear | grid-2x2 | 30 | 10 | 15 | 0.511 ± 0.060 | [0.477, 0.544] | 0.571 ± 0.070 | [0.532, 0.609] | 4 | 5 | 1.90 | 0.43 |
| A | linear | grid-2x2 | 100 | 10 | 15 | 0.558 ± 0.065 | [0.522, 0.594] | 0.590 ± 0.069 | [0.551, 0.628] | 4 | 5 | 1.89 | 0.47 |
| A | mlp | chain | 0.1 | 0 | 15 | 0.564 ± 0.057 | [0.532, 0.596] | 0.648 ± 0.085 | [0.601, 0.695] | 4 | 49 | 0.15 | 0.71 |
| A | mlp | chain | 0.3 | 0 | 15 | 0.532 ± 0.094 | [0.480, 0.584] | 0.623 ± 0.095 | [0.570, 0.675] | 4 | 49 | 0.14 | 0.62 |
| A | mlp | chain | 1 | 0 | 15 | 0.493 ± 0.085 | [0.447, 0.540] | 0.617 ± 0.069 | [0.579, 0.656] | 4 | 49 | 0.14 | 0.75 |
| A | mlp | chain | 3 | 0 | 15 | 0.564 ± 0.072 | [0.525, 0.604] | 0.686 ± 0.071 | [0.647, 0.726] | 4 | 49 | 0.14 | 0.81 |
| A | mlp | chain | 10 | 0 | 15 | 0.576 ± 0.064 | [0.540, 0.612] | 0.683 ± 0.070 | [0.645, 0.722] | 4 | 49 | 0.14 | 0.75 |
| A | mlp | chain | 30 | 0 | 15 | 0.561 ± 0.082 | [0.516, 0.607] | 0.676 ± 0.095 | [0.623, 0.729] | 4 | 49 | 0.14 | 1.17 |
| A | mlp | chain | 100 | 0 | 15 | 0.540 ± 0.060 | [0.506, 0.573] | 0.627 ± 0.085 | [0.581, 0.674] | 4 | 49 | 0.13 | 0.70 |
| A | mlp | chain | 0.1 | 0.1 | 15 | 0.552 ± 0.072 | [0.512, 0.592] | 0.646 ± 0.092 | [0.595, 0.698] | 4 | 49 | 1.94 | 0.72 |
| A | mlp | chain | 0.3 | 0.1 | 15 | 0.552 ± 0.060 | [0.519, 0.585] | 0.623 ± 0.084 | [0.577, 0.669] | 4 | 49 | 1.94 | 0.68 |
| A | mlp | chain | 1 | 0.1 | 15 | 0.491 ± 0.053 | [0.462, 0.521] | 0.613 ± 0.086 | [0.566, 0.661] | 4 | 49 | 1.95 | 0.94 |
| A | mlp | chain | 3 | 0.1 | 15 | 0.562 ± 0.054 | [0.532, 0.592] | 0.688 ± 0.084 | [0.641, 0.734] | 4 | 49 | 1.92 | 0.78 |
| A | mlp | chain | 10 | 0.1 | 15 | 0.559 ± 0.063 | [0.524, 0.594] | 0.700 ± 0.085 | [0.653, 0.747] | 4 | 49 | 1.95 | 0.89 |
| A | mlp | chain | 30 | 0.1 | 15 | 0.577 ± 0.066 | [0.541, 0.614] | 0.665 ± 0.114 | [0.602, 0.728] | 4 | 49 | 1.94 | 0.98 |
| A | mlp | chain | 100 | 0.1 | 15 | 0.543 ± 0.092 | [0.492, 0.594] | 0.634 ± 0.091 | [0.584, 0.684] | 4 | 49 | 1.90 | 0.97 |
| A | mlp | chain | 0.1 | 1 | 15 | 0.540 ± 0.060 | [0.507, 0.573] | 0.642 ± 0.078 | [0.599, 0.685] | 4 | 49 | 1.91 | 0.77 |
| A | mlp | chain | 0.3 | 1 | 15 | 0.546 ± 0.072 | [0.507, 0.586] | 0.641 ± 0.069 | [0.603, 0.680] | 4 | 49 | 1.94 | 0.71 |
| A | mlp | chain | 1 | 1 | 15 | 0.496 ± 0.087 | [0.448, 0.544] | 0.633 ± 0.050 | [0.605, 0.661] | 4 | 49 | 1.95 | 0.87 |
| A | mlp | chain | 3 | 1 | 15 | 0.561 ± 0.062 | [0.526, 0.595] | 0.624 ± 0.064 | [0.588, 0.660] | 4 | 49 | 1.94 | 0.61 |
| A | mlp | chain | 10 | 1 | 15 | 0.565 ± 0.069 | [0.527, 0.604] | 0.668 ± 0.105 | [0.609, 0.726] | 4 | 49 | 1.94 | 0.73 |
| A | mlp | chain | 30 | 1 | 15 | 0.557 ± 0.069 | [0.519, 0.595] | 0.669 ± 0.092 | [0.618, 0.720] | 4 | 49 | 1.94 | 0.77 |
| A | mlp | chain | 100 | 1 | 15 | 0.476 ± 0.078 | [0.433, 0.519] | 0.599 ± 0.083 | [0.553, 0.645] | 4 | 49 | 1.90 | 0.77 |
| A | mlp | chain | 0.1 | 10 | 15 | 0.542 ± 0.079 | [0.499, 0.586] | 0.644 ± 0.087 | [0.595, 0.692] | 4 | 49 | 1.92 | 0.71 |
| A | mlp | chain | 0.3 | 10 | 15 | 0.547 ± 0.090 | [0.497, 0.597] | 0.643 ± 0.091 | [0.592, 0.693] | 4 | 49 | 1.93 | 0.85 |
| A | mlp | chain | 1 | 10 | 15 | 0.527 ± 0.061 | [0.493, 0.560] | 0.632 ± 0.086 | [0.584, 0.679] | 4 | 49 | 1.94 | 0.70 |
| A | mlp | chain | 3 | 10 | 15 | 0.513 ± 0.068 | [0.475, 0.550] | 0.634 ± 0.057 | [0.603, 0.666] | 4 | 49 | 1.97 | 0.69 |
| A | mlp | chain | 10 | 10 | 15 | 0.548 ± 0.064 | [0.513, 0.583] | 0.650 ± 0.060 | [0.617, 0.683] | 4 | 49 | 1.96 | 0.85 |
| A | mlp | chain | 30 | 10 | 15 | 0.559 ± 0.052 | [0.530, 0.588] | 0.643 ± 0.076 | [0.601, 0.685] | 4 | 49 | 1.93 | 0.65 |
| A | mlp | chain | 100 | 10 | 15 | 0.532 ± 0.064 | [0.496, 0.567] | 0.649 ± 0.066 | [0.613, 0.686] | 4 | 49 | 1.94 | 0.92 |
| A | mlp | ring | 0.1 | 0 | 15 | 0.559 ± 0.061 | [0.524, 0.593] | 0.653 ± 0.094 | [0.601, 0.705] | 4 | 49 | 0.14 | 0.93 |
| A | mlp | ring | 0.3 | 0 | 15 | 0.523 ± 0.055 | [0.493, 0.553] | 0.616 ± 0.073 | [0.576, 0.656] | 4 | 49 | 0.14 | 0.52 |
| A | mlp | ring | 1 | 0 | 15 | 0.351 ± 0.102 | [0.295, 0.408] | 0.466 ± 0.096 | [0.413, 0.519] | 4 | 49 | 0.15 | 0.91 |
| A | mlp | ring | 3 | 0 | 15 | 0.548 ± 0.056 | [0.517, 0.579] | 0.696 ± 0.070 | [0.657, 0.735] | 4 | 49 | 0.16 | 0.73 |
| A | mlp | ring | 10 | 0 | 15 | 0.441 ± 0.080 | [0.397, 0.486] | 0.571 ± 0.060 | [0.538, 0.604] | 4 | 49 | 0.14 | 0.96 |
| A | mlp | ring | 30 | 0 | 15 | 0.463 ± 0.084 | [0.417, 0.510] | 0.565 ± 0.096 | [0.512, 0.619] | 4 | 49 | 0.14 | 1.04 |
| A | mlp | ring | 100 | 0 | 15 | 0.510 ± 0.102 | [0.453, 0.567] | 0.634 ± 0.090 | [0.584, 0.684] | 4 | 49 | 0.15 | 0.97 |
| A | mlp | ring | 0.1 | 0.1 | 15 | 0.549 ± 0.066 | [0.513, 0.586] | 0.646 ± 0.092 | [0.595, 0.697] | 4 | 49 | 1.94 | 0.72 |
| A | mlp | ring | 0.3 | 0.1 | 15 | 0.515 ± 0.073 | [0.475, 0.556] | 0.599 ± 0.066 | [0.563, 0.636] | 4 | 49 | 1.95 | 0.50 |
| A | mlp | ring | 1 | 0.1 | 15 | 0.377 ± 0.092 | [0.326, 0.428] | 0.477 ± 0.067 | [0.440, 0.514] | 4 | 49 | 1.99 | 1.52 |
| A | mlp | ring | 3 | 0.1 | 15 | 0.553 ± 0.083 | [0.507, 0.600] | 0.701 ± 0.084 | [0.654, 0.748] | 4 | 49 | 2.05 | 0.72 |
| A | mlp | ring | 10 | 0.1 | 15 | 0.470 ± 0.055 | [0.440, 0.501] | 0.572 ± 0.054 | [0.542, 0.601] | 4 | 49 | 2.07 | 0.70 |
| A | mlp | ring | 30 | 0.1 | 15 | 0.476 ± 0.075 | [0.435, 0.518] | 0.578 ± 0.093 | [0.526, 0.629] | 4 | 49 | 2.05 | 0.91 |
| A | mlp | ring | 100 | 0.1 | 15 | 0.498 ± 0.056 | [0.468, 0.529] | 0.618 ± 0.067 | [0.581, 0.655] | 4 | 49 | 2.04 | 0.75 |
| A | mlp | ring | 0.1 | 1 | 15 | 0.551 ± 0.069 | [0.512, 0.589] | 0.633 ± 0.086 | [0.585, 0.680] | 4 | 49 | 1.95 | 0.82 |
| A | mlp | ring | 0.3 | 1 | 15 | 0.542 ± 0.068 | [0.505, 0.580] | 0.615 ± 0.091 | [0.564, 0.665] | 4 | 49 | 2.00 | 0.66 |
| A | mlp | ring | 1 | 1 | 15 | 0.479 ± 0.079 | [0.435, 0.522] | 0.600 ± 0.063 | [0.565, 0.635] | 4 | 49 | 2.05 | 1.11 |
| A | mlp | ring | 3 | 1 | 15 | 0.568 ± 0.055 | [0.538, 0.598] | 0.677 ± 0.059 | [0.644, 0.710] | 4 | 49 | 2.06 | 0.79 |
| A | mlp | ring | 10 | 1 | 15 | 0.505 ± 0.091 | [0.455, 0.555] | 0.628 ± 0.082 | [0.583, 0.674] | 4 | 49 | 2.05 | 0.84 |
| A | mlp | ring | 30 | 1 | 15 | 0.536 ± 0.071 | [0.496, 0.575] | 0.625 ± 0.089 | [0.576, 0.674] | 4 | 49 | 2.05 | 0.85 |
| A | mlp | ring | 100 | 1 | 15 | 0.585 ± 0.076 | [0.543, 0.627] | 0.702 ± 0.087 | [0.654, 0.750] | 4 | 49 | 2.04 | 0.94 |
| A | mlp | ring | 0.1 | 10 | 15 | 0.552 ± 0.087 | [0.503, 0.600] | 0.640 ± 0.093 | [0.588, 0.692] | 4 | 49 | 1.93 | 0.71 |
| A | mlp | ring | 0.3 | 10 | 15 | 0.537 ± 0.083 | [0.491, 0.583] | 0.629 ± 0.089 | [0.580, 0.678] | 4 | 49 | 1.97 | 0.89 |
| A | mlp | ring | 1 | 10 | 15 | 0.521 ± 0.071 | [0.482, 0.561] | 0.639 ± 0.070 | [0.600, 0.678] | 4 | 49 | 2.03 | 0.74 |
| A | mlp | ring | 3 | 10 | 15 | 0.536 ± 0.049 | [0.509, 0.563] | 0.647 ± 0.052 | [0.619, 0.676] | 4 | 49 | 2.06 | 1.02 |
| A | mlp | ring | 10 | 10 | 15 | 0.523 ± 0.061 | [0.489, 0.557] | 0.629 ± 0.069 | [0.591, 0.667] | 4 | 49 | 2.05 | 0.71 |
| A | mlp | ring | 30 | 10 | 15 | 0.513 ± 0.055 | [0.482, 0.543] | 0.629 ± 0.063 | [0.594, 0.664] | 4 | 49 | 2.07 | 0.71 |
| A | mlp | ring | 100 | 10 | 15 | 0.552 ± 0.056 | [0.521, 0.583] | 0.652 ± 0.067 | [0.615, 0.689] | 4 | 49 | 2.00 | 0.64 |
| A | mlp | grid-2x2 | 0.1 | 0 | 15 | 0.554 ± 0.060 | [0.521, 0.588] | 0.634 ± 0.091 | [0.584, 0.685] | 4 | 49 | 0.14 | 0.78 |
| A | mlp | grid-2x2 | 0.3 | 0 | 15 | 0.517 ± 0.076 | [0.475, 0.559] | 0.614 ± 0.085 | [0.567, 0.661] | 4 | 49 | 0.14 | 0.58 |
| A | mlp | grid-2x2 | 1 | 0 | 15 | 0.417 ± 0.062 | [0.382, 0.451] | 0.505 ± 0.084 | [0.459, 0.552] | 4 | 49 | 0.15 | 0.64 |
| A | mlp | grid-2x2 | 3 | 0 | 15 | 0.557 ± 0.102 | [0.500, 0.614] | 0.686 ± 0.076 | [0.644, 0.728] | 4 | 49 | 0.15 | 0.72 |
| A | mlp | grid-2x2 | 10 | 0 | 15 | 0.421 ± 0.088 | [0.373, 0.470] | 0.559 ± 0.086 | [0.511, 0.606] | 4 | 49 | 0.14 | 0.79 |
| A | mlp | grid-2x2 | 30 | 0 | 15 | 0.484 ± 0.063 | [0.449, 0.519] | 0.611 ± 0.060 | [0.577, 0.644] | 4 | 49 | 0.13 | 1.06 |
| A | mlp | grid-2x2 | 100 | 0 | 15 | 0.529 ± 0.101 | [0.473, 0.585] | 0.646 ± 0.094 | [0.595, 0.698] | 4 | 49 | 0.13 | 0.92 |
| A | mlp | grid-2x2 | 0.1 | 0.1 | 15 | 0.549 ± 0.058 | [0.517, 0.581] | 0.650 ± 0.098 | [0.596, 0.705] | 4 | 49 | 1.98 | 0.90 |
| A | mlp | grid-2x2 | 0.3 | 0.1 | 15 | 0.506 ± 0.068 | [0.468, 0.543] | 0.603 ± 0.089 | [0.554, 0.652] | 4 | 49 | 2.02 | 0.42 |
| A | mlp | grid-2x2 | 1 | 0.1 | 15 | 0.412 ± 0.074 | [0.371, 0.453] | 0.525 ± 0.077 | [0.482, 0.567] | 4 | 49 | 1.98 | 0.76 |
| A | mlp | grid-2x2 | 3 | 0.1 | 15 | 0.552 ± 0.078 | [0.509, 0.596] | 0.690 ± 0.086 | [0.642, 0.737] | 4 | 49 | 2.02 | 0.57 |
| A | mlp | grid-2x2 | 10 | 0.1 | 15 | 0.409 ± 0.074 | [0.368, 0.449] | 0.544 ± 0.090 | [0.494, 0.593] | 4 | 49 | 1.94 | 0.85 |
| A | mlp | grid-2x2 | 30 | 0.1 | 15 | 0.488 ± 0.065 | [0.452, 0.524] | 0.618 ± 0.052 | [0.590, 0.647] | 4 | 49 | 1.90 | 0.71 |
| A | mlp | grid-2x2 | 100 | 0.1 | 15 | 0.503 ± 0.074 | [0.462, 0.544] | 0.613 ± 0.079 | [0.569, 0.657] | 4 | 49 | 1.90 | 0.71 |
| A | mlp | grid-2x2 | 0.1 | 1 | 15 | 0.545 ± 0.074 | [0.504, 0.587] | 0.641 ± 0.096 | [0.588, 0.695] | 4 | 49 | 2.05 | 0.78 |
| A | mlp | grid-2x2 | 0.3 | 1 | 15 | 0.550 ± 0.070 | [0.511, 0.588] | 0.612 ± 0.076 | [0.570, 0.654] | 4 | 49 | 1.97 | 0.71 |
| A | mlp | grid-2x2 | 1 | 1 | 15 | 0.468 ± 0.060 | [0.434, 0.501] | 0.611 ± 0.069 | [0.573, 0.649] | 4 | 49 | 1.96 | 0.83 |
| A | mlp | grid-2x2 | 3 | 1 | 15 | 0.515 ± 0.083 | [0.469, 0.561] | 0.636 ± 0.082 | [0.591, 0.682] | 4 | 49 | 1.92 | 0.66 |
| A | mlp | grid-2x2 | 10 | 1 | 15 | 0.480 ± 0.062 | [0.446, 0.514] | 0.616 ± 0.095 | [0.563, 0.668] | 4 | 49 | 1.94 | 0.76 |
| A | mlp | grid-2x2 | 30 | 1 | 15 | 0.527 ± 0.058 | [0.495, 0.559] | 0.655 ± 0.064 | [0.619, 0.690] | 4 | 49 | 1.90 | 1.10 |
| A | mlp | grid-2x2 | 100 | 1 | 15 | 0.589 ± 0.076 | [0.547, 0.631] | 0.696 ± 0.091 | [0.646, 0.747] | 4 | 49 | 1.90 | 0.82 |
| A | mlp | grid-2x2 | 0.1 | 10 | 15 | 0.532 ± 0.080 | [0.488, 0.576] | 0.636 ± 0.069 | [0.598, 0.675] | 4 | 49 | 1.97 | 0.83 |
| A | mlp | grid-2x2 | 0.3 | 10 | 15 | 0.538 ± 0.085 | [0.491, 0.585] | 0.635 ± 0.082 | [0.589, 0.680] | 4 | 49 | 1.96 | 0.90 |
| A | mlp | grid-2x2 | 1 | 10 | 15 | 0.535 ± 0.046 | [0.509, 0.560] | 0.635 ± 0.070 | [0.596, 0.674] | 4 | 49 | 1.98 | 0.78 |
| A | mlp | grid-2x2 | 3 | 10 | 15 | 0.504 ± 0.071 | [0.465, 0.544] | 0.658 ± 0.060 | [0.625, 0.691] | 4 | 49 | 1.92 | 0.93 |
| A | mlp | grid-2x2 | 10 | 10 | 15 | 0.508 ± 0.068 | [0.470, 0.545] | 0.623 ± 0.072 | [0.584, 0.663] | 4 | 49 | 1.91 | 0.60 |
| A | mlp | grid-2x2 | 30 | 10 | 15 | 0.499 ± 0.054 | [0.469, 0.529] | 0.616 ± 0.083 | [0.570, 0.662] | 4 | 49 | 1.90 | 0.71 |
| A | mlp | grid-2x2 | 100 | 10 | 15 | 0.550 ± 0.078 | [0.506, 0.593] | 0.646 ± 0.069 | [0.607, 0.684] | 4 | 49 | 1.89 | 0.60 |
| B | linear | – | – | 0 | 15 | 0.565 ± 0.063 | [0.529, 0.600] | 0.566 ± 0.071 | [0.527, 0.605] | 4 | 5 | 0.14 | 0.46 |
| B | linear | – | – | 0.1 | 15 | 0.563 ± 0.063 | [0.528, 0.597] | 0.566 ± 0.072 | [0.526, 0.606] | 4 | 5 | 1.91 | 0.45 |
| B | linear | – | – | 1 | 15 | 0.562 ± 0.059 | [0.529, 0.594] | 0.572 ± 0.073 | [0.531, 0.612] | 4 | 5 | 1.95 | 0.43 |
| B | linear | – | – | 10 | 15 | 0.551 ± 0.064 | [0.515, 0.587] | 0.586 ± 0.066 | [0.550, 0.623] | 4 | 5 | 1.95 | 0.49 |
| B | mlp | – | – | 0 | 15 | 0.568 ± 0.063 | [0.533, 0.602] | 0.669 ± 0.078 | [0.626, 0.712] | 4 | 49 | 0.14 | 0.64 |
| B | mlp | – | – | 0.1 | 15 | 0.574 ± 0.051 | [0.545, 0.602] | 0.653 ± 0.087 | [0.605, 0.701] | 4 | 49 | 1.91 | 0.73 |
| B | mlp | – | – | 1 | 15 | 0.552 ± 0.057 | [0.521, 0.584] | 0.638 ± 0.111 | [0.577, 0.700] | 4 | 49 | 1.95 | 0.89 |
| B | mlp | – | – | 10 | 15 | 0.545 ± 0.065 | [0.509, 0.581] | 0.645 ± 0.104 | [0.587, 0.702] | 4 | 49 | 1.95 | 0.75 |
| C | linear | – | – | – | 15 | 0.558 ± 0.054 | [0.528, 0.588] | 0.640 ± 0.072 | [0.600, 0.680] | 4 | 5 | 0.00 | 0.36 |
| C | mlp | – | – | – | 15 | 0.611 ± 0.077 | [0.568, 0.653] | 0.756 ± 0.058 | [0.724, 0.788] | 4 | 49 | 0.00 | 0.70 |
| D | linear | – | – | – | 15 | 0.222 ± 0.109 | [0.161, 0.282] | 0.359 ± 0.067 | [0.322, 0.397] | 4 | 5 | 0.03 | 0.36 |
| D | mlp | – | – | – | 15 | 0.352 ± 0.064 | [0.317, 0.388] | 0.440 ± 0.089 | [0.390, 0.489] | 4 | 49 | 0.03 | 0.65 |
| D-hybrid | hybrid | – | – | – | 15 | 0.469 ± 0.078 | [0.426, 0.512] | 0.578 ± 0.072 | [0.538, 0.618] | 4 | 29 | 0.00 | 15.07 |

Degenerate PCA fits (#525): fold fit in 0 of 15 folds, a tuning fit in 0 of 15 folds.

#### A against B (MCC, paired by fold)

Mean ΔMCC = A − B over the folds (Holm-adjusted p; ✓: at most α).

Head `linear`, geometry `chain`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.006 (1.0000) | -0.002 (1.0000) | -0.006 (1.0000) | +0.000 (1.0000) |
| 0.3 | -0.011 (1.0000) | -0.006 (1.0000) | -0.006 (1.0000) | -0.002 (1.0000) |
| 1 | -0.183 (0.0509) | -0.150 (0.0509) | -0.081 (0.0321 ✓) | -0.007 (1.0000) |
| 3 | -0.077 (0.3127) | -0.060 (0.7863) | -0.014 (1.0000) | -0.035 (1.0000) |
| 10 | -0.051 (0.1425) | -0.039 (0.3127) | -0.008 (1.0000) | -0.008 (1.0000) |
| 30 | -0.033 (1.0000) | -0.035 (1.0000) | -0.019 (1.0000) | -0.003 (1.0000) |
| 100 | -0.109 (0.0321 ✓) | -0.118 (0.0321 ✓) | -0.130 (0.4163) | -0.003 (1.0000) |

Head `linear`, geometry `ring`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.009 (1.0000) | -0.011 (1.0000) | -0.008 (1.0000) | -0.001 (1.0000) |
| 0.3 | -0.034 (1.0000) | -0.031 (1.0000) | -0.021 (1.0000) | -0.002 (1.0000) |
| 1 | -0.436 (0.0321 ✓) | -0.429 (0.0321 ✓) | -0.277 (0.0321 ✓) | -0.028 (1.0000) |
| 3 | -0.016 (1.0000) | -0.028 (1.0000) | -0.062 (1.0000) | -0.178 (0.0321 ✓) |
| 10 | -0.269 (0.0321 ✓) | -0.230 (0.0321 ✓) | -0.128 (0.0321 ✓) | -0.045 (0.5310) |
| 30 | -0.342 (0.0321 ✓) | -0.327 (0.0321 ✓) | -0.147 (0.0321 ✓) | -0.039 (0.3127) |
| 100 | -0.300 (0.0321 ✓) | -0.294 (0.0321 ✓) | -0.173 (0.0321 ✓) | +0.005 (1.0000) |

Head `linear`, geometry `grid-2x2`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.017 (1.0000) | -0.012 (1.0000) | -0.005 (1.0000) | -0.002 (1.0000) |
| 0.3 | -0.041 (1.0000) | -0.039 (1.0000) | -0.018 (1.0000) | -0.005 (1.0000) |
| 1 | -0.376 (0.0321 ✓) | -0.387 (0.0321 ✓) | -0.317 (0.0321 ✓) | -0.028 (1.0000) |
| 3 | -0.397 (0.0321 ✓) | -0.340 (0.0321 ✓) | -0.049 (0.2338) | -0.177 (0.0321 ✓) |
| 10 | -0.304 (0.0321 ✓) | -0.291 (0.0321 ✓) | -0.100 (0.0734) | -0.045 (0.2338) |
| 30 | -0.274 (0.0321 ✓) | -0.285 (0.0321 ✓) | -0.177 (0.0509) | -0.040 (0.6888) |
| 100 | -0.298 (0.0321 ✓) | -0.282 (0.0321 ✓) | -0.168 (0.0321 ✓) | +0.007 (1.0000) |

Head `mlp`, geometry `chain`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.004 (1.0000) | -0.022 (1.0000) | -0.013 (1.0000) | -0.003 (1.0000) |
| 0.3 | -0.036 (1.0000) | -0.022 (1.0000) | -0.006 (1.0000) | +0.002 (1.0000) |
| 1 | -0.074 (1.0000) | -0.082 (0.2338) | -0.056 (0.8871) | -0.018 (1.0000) |
| 3 | -0.003 (1.0000) | -0.011 (1.0000) | +0.008 (1.0000) | -0.032 (1.0000) |
| 10 | +0.008 (1.0000) | -0.014 (1.0000) | +0.013 (1.0000) | +0.003 (1.0000) |
| 30 | -0.006 (1.0000) | +0.004 (1.0000) | +0.005 (1.0000) | +0.014 (1.0000) |
| 100 | -0.028 (1.0000) | -0.031 (1.0000) | -0.076 (1.0000) | -0.013 (1.0000) |

Head `mlp`, geometry `ring`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.009 (1.0000) | -0.024 (1.0000) | -0.002 (1.0000) | +0.007 (1.0000) |
| 0.3 | -0.045 (0.5862) | -0.058 (0.8871) | -0.010 (1.0000) | -0.008 (1.0000) |
| 1 | -0.216 (0.0321 ✓) | -0.196 (0.0509) | -0.073 (0.8871) | -0.024 (1.0000) |
| 3 | -0.020 (1.0000) | -0.020 (1.0000) | +0.016 (1.0000) | -0.009 (1.0000) |
| 10 | -0.126 (0.1658) | -0.103 (0.1212) | -0.047 (1.0000) | -0.022 (1.0000) |
| 30 | -0.104 (0.4163) | -0.097 (0.0509) | -0.017 (1.0000) | -0.032 (1.0000) |
| 100 | -0.057 (1.0000) | -0.075 (0.2338) | +0.033 (1.0000) | +0.007 (1.0000) |

Head `mlp`, geometry `grid-2x2`:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.013 (1.0000) | -0.025 (1.0000) | -0.007 (1.0000) | -0.013 (1.0000) |
| 0.3 | -0.051 (1.0000) | -0.068 (0.1212) | -0.003 (1.0000) | -0.007 (1.0000) |
| 1 | -0.151 (0.0321 ✓) | -0.162 (0.0321 ✓) | -0.085 (0.2338) | -0.010 (1.0000) |
| 3 | -0.010 (1.0000) | -0.021 (1.0000) | -0.037 (1.0000) | -0.041 (1.0000) |
| 10 | -0.146 (0.0321 ✓) | -0.165 (0.0321 ✓) | -0.072 (0.3127) | -0.037 (1.0000) |
| 30 | -0.083 (0.6888) | -0.085 (0.3127) | -0.026 (1.0000) | -0.045 (1.0000) |
| 100 | -0.039 (1.0000) | -0.070 (1.0000) | +0.037 (1.0000) | +0.005 (1.0000) |

#### Every arm against the classical baselines (MCC, paired by fold)

Mean ΔMCC = arm − baseline (Holm-adjusted p; ✓: at most α).

Arm A, head `linear`, geometry `chain`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.001 (1.0000) | +0.002 (1.0000) | -0.002 (1.0000) | -0.006 (1.0000) |
| 0.3 | -0.004 (1.0000) | -0.001 (1.0000) | -0.002 (1.0000) | -0.008 (1.0000) |
| 1 | -0.176 (0.0321 ✓) | -0.146 (0.0321 ✓) | -0.077 (0.2338) | -0.014 (1.0000) |
| 3 | -0.071 (0.5310) | -0.055 (1.0000) | -0.010 (1.0000) | -0.042 (1.0000) |
| 10 | -0.044 (1.0000) | -0.034 (1.0000) | -0.004 (1.0000) | -0.015 (1.0000) |
| 30 | -0.026 (1.0000) | -0.030 (1.0000) | -0.016 (1.0000) | -0.010 (1.0000) |
| 100 | -0.102 (0.0509) | -0.113 (0.0321 ✓) | -0.127 (0.2338) | -0.010 (1.0000) |

Arm A, head `linear`, geometry `ring`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.002 (1.0000) | -0.006 (1.0000) | -0.004 (1.0000) | -0.007 (1.0000) |
| 0.3 | -0.027 (1.0000) | -0.026 (1.0000) | -0.017 (1.0000) | -0.009 (1.0000) |
| 1 | -0.430 (0.0321 ✓) | -0.424 (0.0321 ✓) | -0.273 (0.0321 ✓) | -0.034 (1.0000) |
| 3 | -0.009 (1.0000) | -0.024 (1.0000) | -0.059 (1.0000) | -0.185 (0.0321 ✓) |
| 10 | -0.263 (0.0321 ✓) | -0.225 (0.0321 ✓) | -0.125 (0.0321 ✓) | -0.051 (1.0000) |
| 30 | -0.335 (0.0321 ✓) | -0.322 (0.0321 ✓) | -0.143 (0.0509) | -0.046 (1.0000) |
| 100 | -0.293 (0.0321 ✓) | -0.289 (0.0321 ✓) | -0.169 (0.0321 ✓) | -0.001 (1.0000) |

Arm A, head `linear`, geometry `grid-2x2`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.010 (1.0000) | -0.007 (1.0000) | -0.002 (1.0000) | -0.009 (1.0000) |
| 0.3 | -0.035 (1.0000) | -0.034 (1.0000) | -0.014 (1.0000) | -0.012 (1.0000) |
| 1 | -0.370 (0.0321 ✓) | -0.382 (0.0321 ✓) | -0.313 (0.0321 ✓) | -0.035 (1.0000) |
| 3 | -0.390 (0.0321 ✓) | -0.335 (0.0321 ✓) | -0.045 (1.0000) | -0.184 (0.0321 ✓) |
| 10 | -0.297 (0.0321 ✓) | -0.286 (0.0321 ✓) | -0.096 (0.0509) | -0.051 (1.0000) |
| 30 | -0.267 (0.0321 ✓) | -0.280 (0.0321 ✓) | -0.174 (0.0321 ✓) | -0.047 (1.0000) |
| 100 | -0.291 (0.0321 ✓) | -0.277 (0.0321 ✓) | -0.164 (0.0321 ✓) | -0.000 (1.0000) |

Arm A, head `linear`, geometry `chain`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.052 (1.0000) | -0.050 (1.0000) | -0.055 (1.0000) | -0.059 (1.0000) |
| 0.3 | -0.057 (1.0000) | -0.054 (1.0000) | -0.055 (1.0000) | -0.061 (1.0000) |
| 1 | -0.229 (0.0321 ✓) | -0.198 (0.0509) | -0.130 (0.0321 ✓) | -0.067 (1.0000) |
| 3 | -0.123 (0.0734) | -0.108 (0.2338) | -0.063 (1.0000) | -0.094 (0.8871) |
| 10 | -0.097 (0.0509) | -0.087 (0.2338) | -0.057 (1.0000) | -0.068 (1.0000) |
| 30 | -0.079 (1.0000) | -0.083 (1.0000) | -0.068 (1.0000) | -0.063 (1.0000) |
| 100 | -0.155 (0.0321 ✓) | -0.166 (0.0321 ✓) | -0.179 (0.0321 ✓) | -0.062 (1.0000) |

Arm A, head `linear`, geometry `ring`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.055 (1.0000) | -0.059 (1.0000) | -0.057 (1.0000) | -0.060 (1.0000) |
| 0.3 | -0.080 (1.0000) | -0.079 (1.0000) | -0.070 (1.0000) | -0.061 (1.0000) |
| 1 | -0.482 (0.0321 ✓) | -0.477 (0.0321 ✓) | -0.326 (0.0321 ✓) | -0.087 (1.0000) |
| 3 | -0.062 (1.0000) | -0.076 (1.0000) | -0.111 (0.2338) | -0.238 (0.0321 ✓) |
| 10 | -0.315 (0.0321 ✓) | -0.278 (0.0321 ✓) | -0.177 (0.0321 ✓) | -0.104 (0.4163) |
| 30 | -0.388 (0.0321 ✓) | -0.375 (0.0321 ✓) | -0.196 (0.0321 ✓) | -0.099 (0.4163) |
| 100 | -0.346 (0.0321 ✓) | -0.342 (0.0321 ✓) | -0.222 (0.0321 ✓) | -0.054 (1.0000) |

Arm A, head `linear`, geometry `grid-2x2`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.063 (1.0000) | -0.060 (1.0000) | -0.054 (1.0000) | -0.061 (1.0000) |
| 0.3 | -0.087 (1.0000) | -0.087 (1.0000) | -0.067 (1.0000) | -0.064 (1.0000) |
| 1 | -0.422 (0.0321 ✓) | -0.435 (0.0321 ✓) | -0.366 (0.0321 ✓) | -0.087 (1.0000) |
| 3 | -0.443 (0.0321 ✓) | -0.388 (0.0321 ✓) | -0.097 (0.1658) | -0.237 (0.0321 ✓) |
| 10 | -0.350 (0.0321 ✓) | -0.339 (0.0321 ✓) | -0.149 (0.0321 ✓) | -0.104 (0.4163) |
| 30 | -0.320 (0.0321 ✓) | -0.333 (0.0321 ✓) | -0.226 (0.0321 ✓) | -0.100 (0.5310) |
| 100 | -0.344 (0.0321 ✓) | -0.330 (0.0321 ✓) | -0.217 (0.0321 ✓) | -0.053 (1.0000) |

Arm A, head `mlp`, geometry `chain`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.006 (1.0000) | -0.006 (1.0000) | -0.018 (1.0000) | -0.016 (1.0000) |
| 0.3 | -0.026 (1.0000) | -0.006 (1.0000) | -0.011 (1.0000) | -0.011 (1.0000) |
| 1 | -0.064 (1.0000) | -0.066 (0.5310) | -0.062 (1.0000) | -0.031 (1.0000) |
| 3 | +0.007 (1.0000) | +0.004 (1.0000) | +0.003 (1.0000) | -0.045 (1.0000) |
| 10 | +0.018 (1.0000) | +0.001 (1.0000) | +0.008 (1.0000) | -0.010 (1.0000) |
| 30 | +0.004 (1.0000) | +0.019 (1.0000) | -0.001 (1.0000) | +0.001 (1.0000) |
| 100 | -0.018 (1.0000) | -0.015 (1.0000) | -0.082 (0.1212) | -0.026 (1.0000) |

Arm A, head `mlp`, geometry `ring`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | +0.001 (1.0000) | -0.008 (1.0000) | -0.007 (1.0000) | -0.006 (1.0000) |
| 0.3 | -0.035 (1.0000) | -0.042 (1.0000) | -0.016 (1.0000) | -0.021 (1.0000) |
| 1 | -0.206 (0.0321 ✓) | -0.181 (0.0321 ✓) | -0.079 (1.0000) | -0.037 (1.0000) |
| 3 | -0.010 (1.0000) | -0.004 (1.0000) | +0.010 (1.0000) | -0.022 (1.0000) |
| 10 | -0.116 (0.2338) | -0.088 (0.4163) | -0.053 (1.0000) | -0.035 (1.0000) |
| 30 | -0.094 (0.4163) | -0.081 (1.0000) | -0.022 (1.0000) | -0.045 (1.0000) |
| 100 | -0.048 (1.0000) | -0.060 (1.0000) | +0.027 (1.0000) | -0.006 (1.0000) |

Arm A, head `mlp`, geometry `grid-2x2`, against C/linear:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.004 (1.0000) | -0.009 (1.0000) | -0.012 (1.0000) | -0.026 (1.0000) |
| 0.3 | -0.041 (1.0000) | -0.052 (1.0000) | -0.008 (1.0000) | -0.020 (1.0000) |
| 1 | -0.141 (0.0509) | -0.146 (0.0734) | -0.090 (0.0734) | -0.023 (1.0000) |
| 3 | -0.001 (1.0000) | -0.005 (1.0000) | -0.043 (1.0000) | -0.054 (1.0000) |
| 10 | -0.137 (0.1658) | -0.149 (0.0509) | -0.078 (0.8871) | -0.050 (1.0000) |
| 30 | -0.074 (1.0000) | -0.070 (1.0000) | -0.031 (1.0000) | -0.059 (1.0000) |
| 100 | -0.029 (1.0000) | -0.055 (1.0000) | +0.031 (1.0000) | -0.008 (1.0000) |

Arm A, head `mlp`, geometry `chain`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.046 (1.0000) | -0.059 (1.0000) | -0.071 (1.0000) | -0.068 (1.0000) |
| 0.3 | -0.079 (1.0000) | -0.059 (1.0000) | -0.064 (1.0000) | -0.064 (1.0000) |
| 1 | -0.117 (0.4163) | -0.119 (0.0509) | -0.115 (0.2338) | -0.084 (1.0000) |
| 3 | -0.046 (1.0000) | -0.048 (1.0000) | -0.050 (1.0000) | -0.098 (0.3127) |
| 10 | -0.035 (1.0000) | -0.051 (0.4163) | -0.045 (1.0000) | -0.062 (1.0000) |
| 30 | -0.049 (1.0000) | -0.033 (1.0000) | -0.053 (1.0000) | -0.052 (1.0000) |
| 100 | -0.071 (1.0000) | -0.068 (1.0000) | -0.135 (0.0321 ✓) | -0.079 (1.0000) |

Arm A, head `mlp`, geometry `ring`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.052 (1.0000) | -0.061 (1.0000) | -0.060 (1.0000) | -0.059 (1.0000) |
| 0.3 | -0.088 (0.6888) | -0.095 (0.5310) | -0.068 (1.0000) | -0.073 (1.0000) |
| 1 | -0.259 (0.0321 ✓) | -0.233 (0.0321 ✓) | -0.132 (0.0509) | -0.089 (1.0000) |
| 3 | -0.063 (1.0000) | -0.057 (1.0000) | -0.043 (1.0000) | -0.075 (1.0000) |
| 10 | -0.169 (0.1212) | -0.140 (0.2338) | -0.105 (0.8871) | -0.088 (1.0000) |
| 30 | -0.147 (0.3127) | -0.134 (0.1212) | -0.075 (1.0000) | -0.098 (1.0000) |
| 100 | -0.100 (0.2338) | -0.112 (0.1212) | -0.025 (1.0000) | -0.059 (1.0000) |

Arm A, head `mlp`, geometry `grid-2x2`, against C/mlp:

| V/Ω | γ/Ω = 0 | γ/Ω = 0.1 | γ/Ω = 1 | γ/Ω = 10 |
|---|---|---|---|---|
| 0.1 | -0.056 (1.0000) | -0.062 (1.0000) | -0.065 (1.0000) | -0.078 (1.0000) |
| 0.3 | -0.093 (1.0000) | -0.105 (0.2338) | -0.061 (1.0000) | -0.073 (1.0000) |
| 1 | -0.194 (0.0321 ✓) | -0.199 (0.0509) | -0.143 (0.0321 ✓) | -0.076 (1.0000) |
| 3 | -0.053 (1.0000) | -0.058 (1.0000) | -0.096 (0.2338) | -0.106 (0.4163) |
| 10 | -0.189 (0.0321 ✓) | -0.202 (0.0321 ✓) | -0.131 (0.0509) | -0.103 (0.8871) |
| 30 | -0.126 (0.1212) | -0.122 (0.1658) | -0.084 (0.8871) | -0.111 (0.2338) |
| 100 | -0.082 (1.0000) | -0.107 (0.1658) | -0.022 (1.0000) | -0.061 (1.0000) |

The other arms:

| arm | against C/linear | against C/mlp |
|---|---|---|
| B/linear γ/Ω=0 | +0.007 (1.0000) | -0.046 (1.0000) |
| B/linear γ/Ω=0.1 | +0.005 (1.0000) | -0.048 (1.0000) |
| B/linear γ/Ω=1 | +0.004 (1.0000) | -0.049 (1.0000) |
| B/linear γ/Ω=10 | -0.007 (1.0000) | -0.059 (1.0000) |
| B/mlp γ/Ω=0 | +0.010 (1.0000) | -0.043 (1.0000) |
| B/mlp γ/Ω=0.1 | +0.016 (1.0000) | -0.037 (1.0000) |
| B/mlp γ/Ω=1 | -0.006 (1.0000) | -0.058 (1.0000) |
| B/mlp γ/Ω=10 | -0.013 (1.0000) | -0.066 (1.0000) |
| D/linear | -0.336 (0.0321 ✓) | -0.389 (0.0321 ✓) |
| D/mlp | -0.205 (0.0321 ✓) | -0.258 (0.0321 ✓) |
| D-hybrid/hybrid | -0.089 (0.4163) | -0.141 (0.1212) |

### What the results show

Everything below holds for four atoms on the three shapes of *Setup*, with one pulse, exact
features, the stated tuning budget and these two datasets only.

1. **On Iranian Churn, the ring and the grid make the interactions significantly worse, not
   useful** (15 folds, one family of 526, *Statistics*).
    - The chain's null result was not a property of the chain alone. No A − B gain is
      significant in any geometry. The largest are +0.000 (chain), +0.005 (ring) and +0.007
      (grid) with the linear head, and +0.014 (chain), +0.033 (ring) and +0.037 (grid) with the
      MLP head. No A − B cell of any geometry is won in more than 12 of the 15 folds.
    - A − B losses that are significant: 14 on the ring (13 linear, 1 MLP) and 17 on the grid
      (13 linear, 4 MLP), against 3 on the chain (all linear: `V/Ω = 1`, `γ/Ω = 1`, −0.081, and
      `V/Ω = 100`, `γ/Ω` = 0 and 0.1, −0.109 and −0.118). Each is lost in all 15 folds.
    - The linear head loses −0.436 (ring) and −0.376 (grid) at `V/Ω = 1`, `γ/Ω = 0`, both
      significant, against −0.183 on the chain (Holm p 0.0509, *Statistics*). The grid's largest
      loss is −0.397 at `V/Ω = 3`, `γ/Ω = 0`, where the ring loses only −0.016.
    - That the ring and the grid lose *more than the chain* is read from these cells; the
      geometries are not compared with each other by a test.
    - The 13-fold run (#737) showed the same picture descriptively, with 14 (ring) and 16
      (grid) A − B cells lost in all 13 folds against one on the chain.
2. **On Iranian Churn no Rydberg row beats a classical baseline in any geometry.** No A or B
   row is significantly above C/linear (MCC 0.558) or C/mlp (0.611). The largest margins of an A
   row over C/linear are +0.019 (chain, MLP, `V/Ω = 30`, `γ/Ω = 0.1`), +0.027 (ring) and +0.031
   (grid, both MLP, `V/Ω = 100`, `γ/Ω = 1`), each with Holm p 1.0000. Every A row is below
   C/mlp, the best at 0.589 (grid, MLP, `V/Ω = 100`, `γ/Ω = 1`). Significant losses of an A row
   against a baseline: 9 on the chain, 29 on the ring and 33 on the grid, over both heads and
   both baselines. Arm B is as in #502: its rows are identical across geometries, within 0.016
   of C/linear and never significantly different from either baseline. D/linear and D/mlp are
   significantly below both baselines; D-hybrid is below both, not significantly.
3. **On the positive control the interactions help a linear head less clearly on the ring and the
   grid than on the chain.**
    - The chain detects the effect, as in #502 (+0.376 at `V/Ω = 1`, `γ/Ω = 0`, Holm p 0.0464).
    - The ring's linear head gains at most +0.222 (`V/Ω = 10`, `γ/Ω = 0`, Holm p 0.2256), with
      no A − B cell significant.
    - The grid's linear head gains at most +0.244 (`V/Ω = 3`, `γ/Ω = 0`, Holm p 0.1111), and
      one cell is significant: +0.166 at `V/Ω = 10`, `γ/Ω = 1` (Holm p 0.0464).
    - The labels differ, so this compares how well each geometry's features suit that
      geometry's own task, which has four product terms on the ring and the grid against three
      on the chain.
4. **The MLP head again loses to B on every control.** It loses significantly at 12 grid points
   on the ring and 3 on the grid (6 on the chain), and never gains significantly. It forms
   products of B's independent features itself, as in #502.
5. **Dephasing.** At `γ/Ω = 10` the Iranian Churn A − B differences mostly vanish, but not
   everywhere. The linear head still loses −0.178 (ring) and −0.177 (grid) at `V/Ω = 3`, both
   in all 15 folds and significant, the only significant A − B cells at that `γ/Ω`. On the
   chain the largest loss there is −0.035 (linear head, `V/Ω = 3`, Holm p 1.0000).

## Mechanism

The separation measures of #500, as in #502: on the features before standardisation, on the
real rows of each fold's training part, averaged over the folds (13 on the controls, 15 on
Iranian Churn, where a class-stratified subsample of 2,000 of the 2,940 rows is used). The
Spearman correlations are across the 28 grid points of one geometry, with a permutation p-value
from 10,000 draws. They are uncorrected and over dependent grid points, so they rank the
measures and do not test them. Arm B's values are one per `γ/Ω`, repeated beside every `V/Ω`.
On the controls they differ between geometries, because the labels do.

### Positive control, ring

| V/Ω | γ/Ω | distance_ratio A | distance_ratio B | fisher_ratio A | fisher_ratio B | effective_rank A | effective_rank B | kernel_alignment A | kernel_alignment B |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 1.018 | 1.018 | 0.009 | 0.007 | 3.995 | 3.993 | 0.001 | 0.001 |
| 0.3 | 0 | 1.018 | 1.018 | 0.026 | 0.007 | 3.983 | 3.993 | 0.003 | 0.001 |
| 1 | 0 | 1.080 | 1.018 | 1.138 | 0.007 | 3.895 | 3.993 | 0.142 | 0.001 |
| 3 | 0 | 1.102 | 1.018 | 1.347 | 0.007 | 3.353 | 3.993 | 0.103 | 0.001 |
| 10 | 0 | 1.037 | 1.018 | 0.267 | 0.007 | 2.530 | 3.993 | 0.022 | 0.001 |
| 30 | 0 | 1.019 | 1.018 | 0.239 | 0.007 | 3.995 | 3.993 | 0.028 | 0.001 |
| 100 | 0 | 1.033 | 1.018 | 0.489 | 0.007 | 3.892 | 3.993 | 0.036 | 0.001 |
| 0.1 | 0.1 | 1.018 | 1.018 | 0.009 | 0.007 | 3.995 | 3.993 | 0.001 | 0.001 |
| 0.3 | 0.1 | 1.018 | 1.018 | 0.025 | 0.007 | 3.985 | 3.993 | 0.003 | 0.001 |
| 1 | 0.1 | 1.074 | 1.018 | 1.107 | 0.007 | 3.920 | 3.993 | 0.135 | 0.001 |
| 3 | 0.1 | 1.104 | 1.018 | 1.387 | 0.007 | 3.023 | 3.993 | 0.080 | 0.001 |
| 10 | 0.1 | 1.039 | 1.018 | 0.261 | 0.007 | 2.474 | 3.993 | 0.020 | 0.001 |
| 30 | 0.1 | 1.017 | 1.018 | 0.208 | 0.007 | 3.989 | 3.993 | 0.025 | 0.001 |
| 100 | 0.1 | 1.034 | 1.018 | 0.500 | 0.007 | 3.869 | 3.993 | 0.035 | 0.001 |
| 0.1 | 1 | 1.018 | 1.018 | 0.012 | 0.010 | 3.994 | 3.993 | 0.001 | 0.001 |
| 0.3 | 1 | 1.018 | 1.018 | 0.022 | 0.010 | 3.991 | 3.993 | 0.003 | 0.001 |
| 1 | 1 | 1.022 | 1.018 | 0.426 | 0.010 | 3.979 | 3.993 | 0.050 | 0.001 |
| 3 | 1 | 1.090 | 1.018 | 0.630 | 0.010 | 1.781 | 3.993 | 0.004 | 0.001 |
| 10 | 1 | 1.046 | 1.018 | 0.213 | 0.010 | 2.624 | 3.993 | 0.010 | 0.001 |
| 30 | 1 | 1.015 | 1.018 | 0.109 | 0.010 | 3.601 | 3.993 | 0.004 | 0.001 |
| 100 | 1 | 1.028 | 1.018 | 0.470 | 0.010 | 3.465 | 3.993 | 0.010 | 0.001 |
| 0.1 | 10 | 1.018 | 1.018 | 0.019 | 0.019 | 3.993 | 3.992 | 0.002 | 0.002 |
| 0.3 | 10 | 1.018 | 1.018 | 0.021 | 0.019 | 3.993 | 3.992 | 0.002 | 0.002 |
| 1 | 10 | 1.018 | 1.018 | 0.026 | 0.019 | 3.992 | 3.992 | 0.003 | 0.002 |
| 3 | 10 | 1.014 | 1.018 | 0.058 | 0.019 | 3.954 | 3.992 | 0.006 | 0.002 |
| 10 | 10 | 1.022 | 1.018 | 0.033 | 0.019 | 3.778 | 3.992 | 0.003 | 0.002 |
| 30 | 10 | 1.023 | 1.018 | 0.032 | 0.019 | 3.617 | 3.992 | 0.003 | 0.002 |
| 100 | 10 | 1.020 | 1.018 | 0.023 | 0.019 | 3.714 | 3.992 | 0.002 | 0.002 |

| head | measure | Spearman ρ (Δmeasure, ΔMCC) | n grid points | permutation p |
|---|---|---|---|---|
| linear | distance_ratio | 0.64 | 28 | 6.00e-04 |
| linear | fisher_ratio | 0.60 | 28 | 6.00e-04 |
| linear | effective_rank | -0.29 | 28 | 0.1385 |
| linear | kernel_alignment | 0.65 | 28 | 4.00e-04 |
| mlp | distance_ratio | -0.23 | 28 | 0.2294 |
| mlp | fisher_ratio | -0.73 | 28 | 1.00e-04 |
| mlp | effective_rank | 0.18 | 28 | 0.3584 |
| mlp | kernel_alignment | -0.81 | 28 | 1.00e-04 |

### Positive control, grid

| V/Ω | γ/Ω | distance_ratio A | distance_ratio B | fisher_ratio A | fisher_ratio B | effective_rank A | effective_rank B | kernel_alignment A | kernel_alignment B |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 1.016 | 1.016 | 0.039 | 0.035 | 3.990 | 3.993 | 0.005 | 0.004 |
| 0.3 | 0 | 1.018 | 1.016 | 0.071 | 0.035 | 3.970 | 3.993 | 0.009 | 0.004 |
| 1 | 0 | 1.064 | 1.016 | 0.872 | 0.035 | 3.908 | 3.993 | 0.102 | 0.004 |
| 3 | 0 | 1.086 | 1.016 | 1.320 | 0.035 | 3.411 | 3.993 | 0.100 | 0.004 |
| 10 | 0 | 1.044 | 1.016 | 0.418 | 0.035 | 2.673 | 3.993 | 0.041 | 0.004 |
| 30 | 0 | 1.015 | 1.016 | 0.170 | 0.035 | 3.990 | 3.993 | 0.021 | 0.004 |
| 100 | 0 | 1.031 | 1.016 | 0.502 | 0.035 | 3.892 | 3.993 | 0.036 | 0.004 |
| 0.1 | 0.1 | 1.016 | 1.016 | 0.039 | 0.035 | 3.990 | 3.993 | 0.005 | 0.004 |
| 0.3 | 0.1 | 1.018 | 1.016 | 0.070 | 0.035 | 3.972 | 3.993 | 0.009 | 0.004 |
| 1 | 0.1 | 1.063 | 1.016 | 0.879 | 0.035 | 3.927 | 3.993 | 0.100 | 0.004 |
| 3 | 0.1 | 1.087 | 1.016 | 1.362 | 0.035 | 3.085 | 3.993 | 0.079 | 0.004 |
| 10 | 0.1 | 1.044 | 1.016 | 0.409 | 0.035 | 2.618 | 3.993 | 0.038 | 0.004 |
| 30 | 0.1 | 1.013 | 1.016 | 0.151 | 0.035 | 3.981 | 3.993 | 0.018 | 0.004 |
| 100 | 0.1 | 1.032 | 1.016 | 0.530 | 0.035 | 3.869 | 3.993 | 0.036 | 0.004 |
| 0.1 | 1 | 1.016 | 1.016 | 0.044 | 0.040 | 3.991 | 3.993 | 0.005 | 0.005 |
| 0.3 | 1 | 1.017 | 1.016 | 0.064 | 0.040 | 3.982 | 3.993 | 0.008 | 0.005 |
| 1 | 1 | 1.028 | 1.016 | 0.449 | 0.040 | 3.980 | 3.993 | 0.050 | 0.005 |
| 3 | 1 | 1.067 | 1.016 | 0.677 | 0.040 | 1.852 | 3.993 | 0.005 | 0.005 |
| 10 | 1 | 1.039 | 1.016 | 0.335 | 0.040 | 2.821 | 3.993 | 0.022 | 0.005 |
| 30 | 1 | 1.010 | 1.016 | 0.112 | 0.040 | 3.584 | 3.993 | 0.005 | 0.005 |
| 100 | 1 | 1.022 | 1.016 | 0.654 | 0.040 | 3.466 | 3.993 | 0.014 | 0.005 |
| 0.1 | 10 | 1.017 | 1.017 | 0.058 | 0.057 | 3.992 | 3.992 | 0.007 | 0.007 |
| 0.3 | 10 | 1.017 | 1.017 | 0.059 | 0.057 | 3.992 | 3.992 | 0.007 | 0.007 |
| 1 | 10 | 1.017 | 1.017 | 0.068 | 0.057 | 3.991 | 3.992 | 0.008 | 0.007 |
| 3 | 10 | 1.015 | 1.017 | 0.107 | 0.057 | 3.979 | 3.992 | 0.012 | 0.007 |
| 10 | 10 | 1.018 | 1.017 | 0.079 | 0.057 | 3.855 | 3.992 | 0.008 | 0.007 |
| 30 | 10 | 1.018 | 1.017 | 0.077 | 0.057 | 3.700 | 3.992 | 0.007 | 0.007 |
| 100 | 10 | 1.015 | 1.017 | 0.067 | 0.057 | 3.735 | 3.992 | 0.006 | 0.007 |

| head | measure | Spearman ρ (Δmeasure, ΔMCC) | n grid points | permutation p |
|---|---|---|---|---|
| linear | distance_ratio | 0.72 | 28 | 1.00e-04 |
| linear | fisher_ratio | 0.74 | 28 | 1.00e-04 |
| linear | effective_rank | -0.44 | 28 | 0.0203 |
| linear | kernel_alignment | 0.82 | 28 | 1.00e-04 |
| mlp | distance_ratio | -0.16 | 28 | 0.3992 |
| mlp | fisher_ratio | -0.69 | 28 | 2.00e-04 |
| mlp | effective_rank | 0.22 | 28 | 0.2652 |
| mlp | kernel_alignment | -0.53 | 28 | 0.0042 |

### Iranian Churn, 15 folds

Geometry `chain`:

| V/Ω | γ/Ω | distance_ratio A | distance_ratio B | fisher_ratio A | fisher_ratio B | effective_rank A | effective_rank B | kernel_alignment A | kernel_alignment B |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 1.031 | 1.037 | 2.206 | 2.237 | 3.915 | 3.907 | 0.127 | 0.132 |
| 0.3 | 0 | 1.015 | 1.037 | 2.133 | 2.237 | 3.904 | 3.907 | 0.120 | 0.132 |
| 1 | 0 | 0.953 | 1.037 | 1.368 | 2.237 | 3.145 | 3.907 | 0.090 | 0.132 |
| 3 | 0 | 1.094 | 1.037 | 2.186 | 2.237 | 3.151 | 3.907 | 0.154 | 0.132 |
| 10 | 0 | 1.100 | 1.037 | 2.859 | 2.237 | 3.312 | 3.907 | 0.175 | 0.132 |
| 30 | 0 | 1.086 | 1.037 | 2.664 | 2.237 | 3.423 | 3.907 | 0.178 | 0.132 |
| 100 | 0 | 1.058 | 1.037 | 1.527 | 2.237 | 3.159 | 3.907 | 0.108 | 0.132 |
| 0.1 | 0.1 | 1.030 | 1.036 | 2.208 | 2.237 | 3.916 | 3.908 | 0.127 | 0.131 |
| 0.3 | 0.1 | 1.015 | 1.036 | 2.137 | 2.237 | 3.905 | 3.908 | 0.120 | 0.131 |
| 1 | 0.1 | 0.954 | 1.036 | 1.431 | 2.237 | 3.143 | 3.908 | 0.092 | 0.131 |
| 3 | 0.1 | 1.103 | 1.036 | 2.215 | 2.237 | 3.026 | 3.908 | 0.157 | 0.131 |
| 10 | 0.1 | 1.096 | 1.036 | 2.827 | 2.237 | 3.299 | 3.908 | 0.174 | 0.131 |
| 30 | 0.1 | 1.083 | 1.036 | 2.634 | 2.237 | 3.394 | 3.908 | 0.176 | 0.131 |
| 100 | 0.1 | 1.053 | 1.036 | 1.466 | 2.237 | 3.066 | 3.908 | 0.111 | 0.131 |
| 0.1 | 1 | 1.023 | 1.027 | 2.213 | 2.235 | 3.919 | 3.916 | 0.126 | 0.129 |
| 0.3 | 1 | 1.012 | 1.027 | 2.162 | 2.235 | 3.908 | 3.916 | 0.121 | 0.129 |
| 1 | 1 | 0.972 | 1.027 | 1.800 | 2.235 | 3.313 | 3.916 | 0.107 | 0.129 |
| 3 | 1 | 1.122 | 1.027 | 2.353 | 2.235 | 2.436 | 3.916 | 0.161 | 0.129 |
| 10 | 1 | 1.072 | 1.027 | 2.510 | 2.235 | 3.313 | 3.916 | 0.166 | 0.129 |
| 30 | 1 | 1.066 | 1.027 | 2.395 | 2.235 | 3.310 | 3.916 | 0.166 | 0.129 |
| 100 | 1 | 1.042 | 1.027 | 1.508 | 2.235 | 2.806 | 3.916 | 0.136 | 0.129 |
| 0.1 | 10 | 1.002 | 1.003 | 2.171 | 2.176 | 3.922 | 3.923 | 0.117 | 0.117 |
| 0.3 | 10 | 1.001 | 1.003 | 2.160 | 2.176 | 3.917 | 3.923 | 0.117 | 0.117 |
| 1 | 10 | 0.998 | 1.003 | 2.113 | 2.176 | 3.867 | 3.923 | 0.117 | 0.117 |
| 3 | 10 | 0.997 | 1.003 | 1.941 | 2.176 | 3.408 | 3.923 | 0.115 | 0.117 |
| 10 | 10 | 1.019 | 1.003 | 2.131 | 2.176 | 3.702 | 3.923 | 0.135 | 0.117 |
| 30 | 10 | 1.023 | 1.003 | 2.163 | 2.176 | 3.785 | 3.923 | 0.138 | 0.117 |
| 100 | 10 | 1.020 | 1.003 | 2.112 | 2.176 | 3.748 | 3.923 | 0.136 | 0.117 |

| head | measure | Spearman ρ (Δmeasure, ΔMCC) | n grid points | permutation p |
|---|---|---|---|---|
| linear | distance_ratio | -0.11 | 28 | 0.6002 |
| linear | fisher_ratio | 0.38 | 28 | 0.0471 |
| linear | effective_rank | 0.82 | 28 | 1.00e-04 |
| linear | kernel_alignment | 0.12 | 28 | 0.5327 |
| mlp | distance_ratio | 0.62 | 28 | 5.00e-04 |
| mlp | fisher_ratio | 0.80 | 28 | 1.00e-04 |
| mlp | effective_rank | 0.17 | 28 | 0.3981 |
| mlp | kernel_alignment | 0.70 | 28 | 2.00e-04 |

Geometry `ring`:

| V/Ω | γ/Ω | distance_ratio A | distance_ratio B | fisher_ratio A | fisher_ratio B | effective_rank A | effective_rank B | kernel_alignment A | kernel_alignment B |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 1.026 | 1.037 | 2.201 | 2.237 | 3.911 | 3.907 | 0.124 | 0.132 |
| 0.3 | 0 | 0.996 | 1.037 | 2.106 | 2.237 | 3.889 | 3.907 | 0.107 | 0.132 |
| 1 | 0 | 0.948 | 1.037 | 0.096 | 2.237 | 3.719 | 3.907 | 0.007 | 0.132 |
| 3 | 0 | 1.237 | 1.037 | 2.664 | 2.237 | 3.226 | 3.907 | 0.213 | 0.132 |
| 10 | 0 | 1.035 | 1.037 | 1.068 | 2.237 | 2.515 | 3.907 | 0.103 | 0.132 |
| 30 | 0 | 0.955 | 1.037 | 0.441 | 2.237 | 3.789 | 3.907 | 0.027 | 0.132 |
| 100 | 0 | 0.972 | 1.037 | 0.615 | 2.237 | 3.802 | 3.907 | 0.039 | 0.132 |
| 0.1 | 0.1 | 1.025 | 1.036 | 2.202 | 2.237 | 3.912 | 3.908 | 0.124 | 0.131 |
| 0.3 | 0.1 | 0.996 | 1.036 | 2.111 | 2.237 | 3.891 | 3.908 | 0.108 | 0.131 |
| 1 | 0.1 | 0.937 | 1.036 | 0.149 | 2.237 | 3.725 | 3.908 | 0.010 | 0.131 |
| 3 | 0.1 | 1.269 | 1.036 | 2.607 | 2.237 | 2.847 | 3.908 | 0.215 | 0.131 |
| 10 | 0.1 | 1.038 | 1.036 | 1.156 | 2.237 | 2.468 | 3.908 | 0.110 | 0.131 |
| 30 | 0.1 | 0.943 | 1.036 | 0.437 | 2.237 | 3.800 | 3.908 | 0.027 | 0.131 |
| 100 | 0.1 | 0.969 | 1.036 | 0.720 | 2.237 | 3.750 | 3.908 | 0.046 | 0.131 |
| 0.1 | 1 | 1.019 | 1.027 | 2.207 | 2.235 | 3.916 | 3.916 | 0.123 | 0.129 |
| 0.3 | 1 | 0.999 | 1.027 | 2.137 | 2.235 | 3.903 | 3.916 | 0.111 | 0.129 |
| 1 | 1 | 0.913 | 1.027 | 1.167 | 2.235 | 3.816 | 3.916 | 0.054 | 0.129 |
| 3 | 1 | 1.294 | 1.027 | 2.340 | 2.235 | 1.682 | 3.916 | 0.196 | 0.129 |
| 10 | 1 | 1.073 | 1.027 | 1.827 | 2.235 | 2.627 | 3.916 | 0.154 | 0.129 |
| 30 | 1 | 0.968 | 1.027 | 1.566 | 2.235 | 3.584 | 3.916 | 0.092 | 0.129 |
| 100 | 1 | 0.991 | 1.027 | 2.171 | 2.235 | 3.253 | 3.916 | 0.109 | 0.129 |
| 0.1 | 10 | 1.001 | 1.003 | 2.169 | 2.176 | 3.922 | 3.923 | 0.116 | 0.117 |
| 0.3 | 10 | 0.999 | 1.003 | 2.152 | 2.176 | 3.920 | 3.923 | 0.114 | 0.117 |
| 1 | 10 | 0.988 | 1.003 | 2.072 | 2.176 | 3.911 | 3.923 | 0.106 | 0.117 |
| 3 | 10 | 0.962 | 1.003 | 1.630 | 2.176 | 3.901 | 3.923 | 0.084 | 0.117 |
| 10 | 10 | 1.021 | 1.003 | 2.025 | 2.176 | 3.840 | 3.923 | 0.138 | 0.117 |
| 30 | 10 | 1.034 | 1.003 | 2.043 | 2.176 | 3.695 | 3.923 | 0.146 | 0.117 |
| 100 | 10 | 1.021 | 1.003 | 2.170 | 2.176 | 3.729 | 3.923 | 0.134 | 0.117 |

| head | measure | Spearman ρ (Δmeasure, ΔMCC) | n grid points | permutation p |
|---|---|---|---|---|
| linear | distance_ratio | 0.61 | 28 | 0.0014 |
| linear | fisher_ratio | 0.89 | 28 | 1.00e-04 |
| linear | effective_rank | 0.44 | 28 | 0.0208 |
| linear | kernel_alignment | 0.75 | 28 | 1.00e-04 |
| mlp | distance_ratio | 0.49 | 28 | 0.0079 |
| mlp | fisher_ratio | 0.82 | 28 | 1.00e-04 |
| mlp | effective_rank | 0.22 | 28 | 0.2506 |
| mlp | kernel_alignment | 0.64 | 28 | 4.00e-04 |

Geometry `grid-2x2`:

| V/Ω | γ/Ω | distance_ratio A | distance_ratio B | fisher_ratio A | fisher_ratio B | effective_rank A | effective_rank B | kernel_alignment A | kernel_alignment B |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 1.029 | 1.037 | 2.197 | 2.237 | 3.912 | 3.907 | 0.128 | 0.132 |
| 0.3 | 0 | 1.004 | 1.037 | 2.057 | 2.237 | 3.902 | 3.907 | 0.114 | 0.132 |
| 1 | 0 | 0.973 | 1.037 | 0.183 | 2.237 | 3.894 | 3.907 | 0.012 | 0.132 |
| 3 | 0 | 0.981 | 1.037 | 0.287 | 2.237 | 3.282 | 3.907 | 0.013 | 0.132 |
| 10 | 0 | 0.893 | 1.037 | 0.457 | 2.237 | 2.135 | 3.907 | 0.019 | 0.132 |
| 30 | 0 | 0.976 | 1.037 | 0.535 | 2.237 | 3.794 | 3.907 | 0.036 | 0.132 |
| 100 | 0 | 0.980 | 1.037 | 0.645 | 2.237 | 3.797 | 3.907 | 0.040 | 0.132 |
| 0.1 | 0.1 | 1.028 | 1.036 | 2.198 | 2.237 | 3.913 | 3.908 | 0.128 | 0.131 |
| 0.3 | 0.1 | 1.005 | 1.036 | 2.064 | 2.237 | 3.903 | 3.908 | 0.115 | 0.131 |
| 1 | 0.1 | 0.959 | 1.036 | 0.127 | 2.237 | 3.903 | 3.908 | 0.008 | 0.131 |
| 3 | 0.1 | 0.984 | 1.036 | 0.429 | 2.237 | 2.890 | 3.908 | 0.013 | 0.131 |
| 10 | 0.1 | 0.888 | 1.036 | 0.527 | 2.237 | 2.114 | 3.908 | 0.018 | 0.131 |
| 30 | 0.1 | 0.981 | 1.036 | 0.689 | 2.237 | 3.759 | 3.908 | 0.046 | 0.131 |
| 100 | 0.1 | 0.977 | 1.036 | 0.758 | 2.237 | 3.744 | 3.908 | 0.048 | 0.131 |
| 0.1 | 1 | 1.021 | 1.027 | 2.201 | 2.235 | 3.918 | 3.916 | 0.125 | 0.129 |
| 0.3 | 1 | 1.005 | 1.027 | 2.106 | 2.235 | 3.914 | 3.916 | 0.116 | 0.129 |
| 1 | 1 | 0.924 | 1.027 | 0.840 | 2.235 | 3.890 | 3.916 | 0.039 | 0.129 |
| 3 | 1 | 0.957 | 1.027 | 2.017 | 2.235 | 1.736 | 3.916 | 0.019 | 0.129 |
| 10 | 1 | 0.890 | 1.027 | 1.666 | 2.235 | 2.361 | 3.916 | 0.028 | 0.129 |
| 30 | 1 | 1.015 | 1.027 | 1.854 | 2.235 | 3.519 | 3.916 | 0.116 | 0.129 |
| 100 | 1 | 0.990 | 1.027 | 2.195 | 2.235 | 3.258 | 3.916 | 0.106 | 0.129 |
| 0.1 | 10 | 1.002 | 1.003 | 2.168 | 2.176 | 3.922 | 3.923 | 0.116 | 0.117 |
| 0.3 | 10 | 0.999 | 1.003 | 2.151 | 2.176 | 3.921 | 3.923 | 0.114 | 0.117 |
| 1 | 10 | 0.989 | 1.003 | 2.066 | 2.176 | 3.913 | 3.923 | 0.106 | 0.117 |
| 3 | 10 | 0.950 | 1.003 | 1.565 | 2.176 | 3.847 | 3.923 | 0.069 | 0.117 |
| 10 | 10 | 0.972 | 1.003 | 2.003 | 2.176 | 3.707 | 3.923 | 0.087 | 0.117 |
| 30 | 10 | 0.975 | 1.003 | 2.046 | 2.176 | 3.554 | 3.923 | 0.088 | 0.117 |
| 100 | 10 | 1.004 | 1.003 | 2.168 | 2.176 | 3.702 | 3.923 | 0.117 | 0.117 |

| head | measure | Spearman ρ (Δmeasure, ΔMCC) | n grid points | permutation p |
|---|---|---|---|---|
| linear | distance_ratio | 0.81 | 28 | 1.00e-04 |
| linear | fisher_ratio | 0.95 | 28 | 1.00e-04 |
| linear | effective_rank | 0.49 | 28 | 0.0092 |
| linear | kernel_alignment | 0.92 | 28 | 1.00e-04 |
| mlp | distance_ratio | 0.78 | 28 | 1.00e-04 |
| mlp | fisher_ratio | 0.73 | 28 | 1.00e-04 |
| mlp | effective_rank | 0.24 | 28 | 0.2019 |
| mlp | kernel_alignment | 0.68 | 28 | 2.00e-04 |

### What the measures show

- **Iranian Churn, linear head** (15 folds). On the ring and the grid its A − B tracks the class
  separation more closely than on the chain:
    - Fisher ratio: ρ = 0.89 (ring) and 0.95 (grid), against 0.38 on the chain (p = 0.0471);
    - kernel alignment: 0.75 and 0.92, against 0.12 on the chain (p = 0.5327);
    - distance ratio: 0.61 (p = 0.0014) and 0.81, against −0.11 on the chain (p = 0.6002);
    - permutation p = 1.00e-04 where not given.

  On the chain it goes with the effective rank instead (ρ = 0.82), as #502 found on its folds
  (0.70). Where the linear head loses most on the square, at `V/Ω = 1`, `γ/Ω = 0`, arm A's
  Fisher ratio is 0.096 (ring) and 0.183 (grid), against B's 2.237: on those features the
  classes have nearly lost their mean separation. At `γ/Ω = 0` the one ring cell with A's
  Fisher ratio above B's (2.664 at `V/Ω = 3`; at `V/Ω = 3` it is also above B's for `γ/Ω` = 0.1
  and 1) has an A − B of only −0.016, against −0.269 to −0.436 at `V/Ω` = 1, 10, 30 and 100.
- **Positive control, linear head:** as on the chain, its gains go with the distance ratio, the
  Fisher ratio and the kernel alignment:
    - ring: ρ = 0.64, 0.60 and 0.65, p ≤ 6.00e-04;
    - grid: ρ = 0.72, 0.74 and 0.82, p = 1.00e-04.
- **Positive control, MLP head:** its A − B goes *against* the Fisher ratio (ρ = −0.73 ring,
  −0.69 grid) and the kernel alignment (−0.81, −0.53). The chain's MLP head tracked the
  effective rank instead (ρ = 0.83 in #502). Here that correlation is 0.18 (ring) and 0.22
  (grid), not distinguishable from none.

So the linear head's A − B moves with the class-separation measures on the control in all three
geometries, and on Iranian Churn on the ring and the grid (on the chain only weakly, through
the Fisher ratio, ρ = 0.38). There the ring and the grid move those measures down, and the score
with them. On the 13 folds of #737 the same ordering held (Fisher ratio ρ = 0.85 ring, 0.96 grid,
0.48 chain).

## Limits

#502's *Limits* apply unchanged (two-level approximation, dephasing model, exact readout,
datasets, fold intervals). In addition:

- **Three geometries of four atoms, two of them one shape.** The ring and the `2 × 2` grid are
  the same square with different inputs on neighbouring atoms (*Setup*). So this note compares
  two shapes, the chain and the square, and two assignments of the inputs on the square. Larger
  rings and grids, where the shapes do differ (`2a cos(π/N)` against `√2 a`), were not run.
- **Next-nearest couplings** are `V/64` on the chain and `V/8` on the square (*Setup*). At
  `V/Ω ≥ 10` every pair of the square is blockaded. So the geometries differ both in their
  neighbours and in how strong the second-order coupling is, and this run does not separate
  the two.
- **Iranian Churn: 15 folds detect only unanimous results** (*Statistics*). With three
  geometries in one family of 526, 15 folds are the fewest that can reach α, and in this run
  only comparisons won or lost in every fold do. A difference that goes the other way in one
  fold, such as the chain's −0.183 at `V/Ω = 1`, `γ/Ω = 0` (lost in 14 of 15 folds), is not
  significant, however large. The tests are of each geometry against B and the baselines; the
  geometries are not tested against each other.
- **Two Iranian Churn runs.** The 13-fold run of #737 stays in `docs/results/` as its record;
  read alone it is descriptive, since 13 folds cannot reach α in a family of 526. The two runs
  share data and seeds but not folds, and are not pooled or tested together. Its chain lines
  reproduce #502's; the 15-fold run's do not, because its folds differ.
- **The controls are three tasks.** Each geometry's control has its own label, and so its own B,
  C and D rows and its own family of 190. The control's detection on the chain and its
  weaker results on the ring and the grid are about each geometry on its own task, not one task
  across geometries. Since #741 one run can hold all three, and each label still gets its
  own B, C and D lines.
- **One input order.** Input `i` is the `i`-th principal component. Which components become
  neighbours is fixed by that order, and other assignments, beyond the ring/grid exchange of
  inputs 2 and 3, were not tried.
- **Not tried:** as in #502 (pair correlations as features, time-dependent pulses, shots, more
  than four atoms, a tuned architecture, more tuning trials), and the other datasets
  (`taiwanese-bankruptcy`, `cervical-cancer-risk`).
