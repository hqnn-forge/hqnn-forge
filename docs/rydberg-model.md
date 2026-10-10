# The Rydberg feature model

This page specifies a small simulated array of neutral atoms whose excitation probabilities
serve as precomputed features for a classical head. It fixes every convention before any code:
units, Hamiltonian, geometry, encoding, pulse, readout and noise. The register and Hamiltonian
(#496), the solver (#497) and the feature map (#499) implement exactly what is written here,
and each convention comes with a closed-form limit that their tests check (see
[Closed-form limits](#closed-form-limits)).

The defaults are chosen so that the interaction strength and the dephasing rate are the only
swept quantities, and so that the model without interactions stays a closed-form reference:

| | Default | Section |
|---|---|---|
| Atoms | `N = d`, one per input, `4 ≤ N ≤ 6`, open chain at spacing `a` | Geometry |
| Interaction | `C6 / r⁶` between every pair (full tail), `C6 = 2π × 862 690 MHz µm⁶` | Hamiltonian |
| Encoding | `Δ_i = Δ_max · σ(x_i)`, one-signed, `Δ_max = √3 Ω` | Encoding |
| Pulse | constant, `ΩT = π`, global Ω | Pulse |
| Readout | exact `⟨n_i⟩`, optionally `⟨n_i n_j⟩` and a stated number of shots | Readout |
| Noise | Markovian dephasing, `L_i = √γ n_i`, no decay | Noise |
| Swept | `V/Ω` through the spacing, and `γ/Ω` | Geometry, Noise |

Everything here is a simulation of an idealised model. Which assumptions a current device
supports is listed under [Hardware realism](#hardware-realism).

## Units

- **ħ = 1**, so energies are angular frequencies. **Angular frequencies are in rad/µs, lengths
  in µm, times in µs.** A frequency of `f` MHz is `2π f` rad/µs; `C6` is in rad/µs · µm⁶.
- **Results are reported in dimensionless groups:** the pulse area `ΩT`, the detuning `Δ/Ω`,
  the interaction `V/Ω` with `V` the nearest-neighbour interaction, and the dephasing rate
  `γ/Ω`.
- **Ω only sets the scale.** Dividing the master equation below by Ω and measuring time as
  `Ωt` leaves `Δ_i/Ω`, `V_ij/Ω`, `γ/Ω` and `ΩT` as its only parameters, so the features depend
  on nothing else. The same features result at any Ω when the spacing is rescaled to keep `V/Ω`.
  A value of Ω is needed only to convert to µm and µs: the **reference value is
  `Ω = 4π rad/µs` (2π × 2 MHz)**, the Rabi frequency of Bernien et al. (2017) and inside the
  range of the device compared against below.

## Hamiltonian

Each atom is a two-level system of its ground state `|g⟩ = |0⟩` and one Rydberg state
`|r⟩ = |1⟩`. In the frame rotating at the laser frequency, with the rotating-wave
approximation:

```text
H = Ω/2 · Σ_i X_i  −  Σ_i Δ_i n_i  +  Σ_{i<j} V_ij n_i n_j

n_i = |r⟩⟨r|_i = (1 − Z_i)/2        V_ij = C6 / |r_i − r_j|⁶
```

For one atom, in the basis `(|g⟩, |r⟩)`, this is the matrix `[[0, Ω/2], [Ω/2, −Δ]]`. The
evolution starts with every atom in `|g⟩`.

- **Ω is the Rabi frequency, not half of it.** The coefficient of `X` is `Ω/2`, so a resonant
  atom has `⟨n⟩(t) = sin²(Ωt/2)`: the population oscillates at angular frequency Ω, and
  `ΩT = π` is a π pulse. Off resonance, with `Ω_eff = √(Ω² + Δ²)`,

    ```text
    ⟨n⟩(t) = Ω²/Ω_eff² · sin²(Ω_eff t / 2)
    ```

    With the other common convention, `H = Ω X`, the same pulse would be a 2π pulse.

- **Δ is the laser frequency minus the atomic transition frequency.** In the laboratory frame
  the atom has energy `ω₀ n`; rotating at the laser frequency `ω_L` subtracts `ω_L n`, which
  leaves `−(ω_L − ω₀) n = −Δ n`. A positive Δ therefore lowers the energy of `|r⟩`. The
  interaction enters with the opposite sign: an atom whose neighbours are excited sees the
  effective detuning `Δ_i − Σ_j V_ij n_j`, and the doubly excited state of two atoms has the
  diagonal entry `−Δ_1 − Δ_2 + V_12`.
- **Basis ordering.** Atom `i` (counted from 0) is tensor factor `i` from the left and PennyLane
  wire `i`. The basis state `|b_0 b_1 … b_{N−1}⟩`, with `b_i = 1` for `|r⟩`, has the index
  `Σ_i b_i 2^(N−1−i)`: atom 0 is the most significant bit. For two atoms the order is
  `|gg⟩, |gr⟩, |rg⟩, |rr⟩`, and `n_0 = diag(0, 0, 1, 1)`. This is the matrix
  `qml.matrix(op, wire_order=range(N))` returns.
- **C6.** The default is `C6 = 2π × 862 690 MHz µm⁶ = 5 420 441 rad/µs · µm⁶`, positive
  (repulsive), for two ⁸⁷Rb atoms in `70S₁/₂`. It is the default of PennyLane's
  `qml.pulse.rydberg_interaction`, so a cross-check against PennyLane needs no coefficient
  passed. The Aquila device description (Wurtz et al. 2023) states 5 420 503 for the same
  state, 1.1 × 10⁻⁵ larger, which changes no number on this page. As an independent check,
  Bernien et al. (2017) measured `V = 2π × 24.4 MHz` between `70S` atoms about 5.7 µm apart;
  the default gives `2π × 24.1 MHz` at 5.74 µm. The interaction is taken as isotropic.
- **The same model in other software.** With a drive phase φ the first term reads
  `Ω/2 · Σ_i (cos φ X_i − sin φ Y_i) = Ω/2 · Σ_i (e^{iφ} |g⟩⟨r|_i + h.c.)`, the form of the
  trainable analog layer (#420), of `qml.pulse.rydberg_drive` and of Wurtz et al. (2023). This
  page fixes `φ = 0` and constant Ω and Δ, the special case of one pulse segment; nothing here
  conflicts with a time-dependent `Ω(t)`, `Δ_i(t)`, `φ(t)`. PennyLane takes its amplitudes in
  MHz and multiplies by 2π internally; with that conversion its matrix equals this one.

**What the two-level model leaves out:**

- **Intermediate-state scattering.** The transition is driven with two photons through
  `6P₃/₂`. In Bernien et al. (2017), scattering from that state occurs on a timescale of 40 µs
  and gives, together with the Rydberg lifetime, an effective lifetime of 50 µs.
- **Other Rydberg levels.** One interaction channel `C6/r⁶` stands for the full pair-state
  structure. It is a long-distance form; this page does not use it below 4 µm.
- **Atomic motion.** Atoms at 12 µK have random Doppler shifts of about 2π × 50 kHz (Bernien
  et al. 2017), and their positions vary from shot to shot (0.2 µm in Wurtz et al. 2023), which
  changes `V_ij`. See [Noise](#noise) for why the Doppler shift is not the dephasing modelled
  here.
- **Decay of the Rydberg state, preparation and detection errors, pulse ramps** and
  inhomogeneity of Ω and Δ across the array.

## Geometry

**One shape: an open chain.** `N` atoms on a line at spacing `a`, atom `i` at
`r_i = (i · a, 0)`, in the order of the inputs: input `x_i` goes to atom `i`, so neighbouring
inputs sit on neighbouring atoms. `N` equals the input dimension `d`, and the model is
specified for `4 ≤ N ≤ 6`.

- **The full `1/r⁶` tail is kept.** `V_ij = V / |i − j|⁶` with `V = C6/a⁶` the
  nearest-neighbour interaction: next-nearest neighbours interact with `V/64`, the next with
  `V/729`. At this size the tail costs nothing, and truncating to nearest neighbours would be
  one more approximation to justify.
- **Blockade radius.** `R_b = (C6/Ω)^(1/6)` is the distance at which two atoms interact with
  `V_ij = Ω`, so `V/Ω = (R_b/a)⁶` and `a = R_b · (V/Ω)^(−1/6)`. At the reference Ω,
  `R_b = 8.69 µm`. A unit-disk graph at radius `R_b` (the gate-based entangler of #418)
  connects exactly the pairs with `V_ij ≥ Ω`.
- **The regime is not fixed: `V/Ω` is swept through the spacing**, from weakly interacting
  through `V ≈ Ω` into the blockade. At the reference Ω:

    | `V/Ω` | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 |
    |---|---|---|---|---|---|---|---|
    | `a` in µm | 12.76 | 10.62 | 8.69 | 7.24 | 5.92 | 4.93 | 4.03 |

- **The non-interacting control is `V_ij = 0` exactly**, the Hamiltonian without its third
  term, not a large spacing. Everything else stays the same.

## Encoding

**One input per atom, in a one-signed local detuning:**

```text
Δ_i(x_i) = Δ_max · σ(x_i)        σ(x) = 1 / (1 + e^(−x))        Δ_max = √3 Ω
```

- **Data and controls.** The detunings `Δ_i` carry the data and nothing else does. Ω, `T`, the
  positions and `C6` are fixed controls, the same for every sample.
- **Bounds.** `Δ_i` lies in the open interval `(0, Δ_max)` for every finite input. The bound is
  enforced by the logistic function, not by clipping: clipping would map every input beyond
  the range to the same detuning. A non-finite input is an error, not a value to squash.
- **Why the logistic function.** Any strictly increasing map onto `(0, Δ_max)` keeps the
  properties below; the logistic function is fixed here because it uses the range for both
  input scales the library produces. Standardised inputs within ±2 cover 12 % to 88 % of the
  range, and `PCANormalizer` outputs in (−π, π) cover 4 % to 96 %. No scale parameter is added:
  scaling the inputs is preprocessing.
- **Why local detuning.** It keeps the control informative: without interactions each feature
  is the single-atom formula of that atom's own input (next section), and no other input
  enters it.
- **Why one-signed.** Without interactions the excitation probability depends on Δ only
  through Δ² (the formula above contains Δ only in `Ω_eff`). A symmetric range would discard
  the sign of the input in the control. Interactions shift the effective detuning by
  `−Σ_j V_ij n_j` and would recover the sign, which would credit them with information the
  encoding itself threw away.

## Pulse

**Constant Ω and `Δ_i` for a time `T`, with `ΩT = π`.** Without interactions and noise, feature
`i` is then, with `s_i = Ω_eff,i / Ω = √(1 + (Δ_i/Ω)²)`,

```text
⟨n_i⟩ = f(s_i),        f(s) = sin²(π s / 2) / s²,        s_i ∈ (1, 2)
```

- **`f` falls strictly from 1 at `Δ = 0` to 0 at `Δ_max`.** Its derivative is
  `sin(πs/2) · [π s cos(πs/2) − 2 sin(πs/2)] / s³`; on `1 < s < 2` the sine is positive and the
  cosine negative, so it is negative. The control therefore loses no information about its
  input: the composition with σ is strictly decreasing in `x_i`, from 1 as `x_i → −∞` to 0 as
  `x_i → +∞`, and 0.437 at `x_i = 0`.
- **Why `Δ_max = √3 Ω`.** It is the detuning with `s = 2`, the first zero of `f`. Beyond it `f`
  rises again (to 0.116 at `Δ = 2.68 Ω`), and two inputs would give the same feature.
- **Why `ΩT = π`.** It is the shortest pulse with full contrast: the feature reaches 1 at
  `Δ = 0`.
- **The ends are flat.** `f ≈ 1 − (Δ/Ω)²` near `Δ = 0` and
  `f ≈ (3π²/64) · (√3 − Δ/Ω)²` near `Δ_max`: injective, but with vanishing slope at both ends.
- **Time-dependent schedules are out of scope.**

## Readout

- **Features are excitation probabilities,** `F_i = ⟨n_i⟩ = Tr[ρ(T) n_i]`, one per atom, in
  the order of the atoms.
- **Pair correlations are optional:** `⟨n_i n_j⟩` for `i < j`, appended in the order
  `(0,1), (0,2), …, (N−2,N−1)`, which gives `N + N(N−1)/2` features.
- **Both come from the diagonal of ρ alone,** as `Σ_b b_i ρ_bb` and `Σ_b b_i b_j ρ_bb` over the
  basis states `b`. This is a measurement of every atom in the `(|g⟩, |r⟩)` basis.
- **Exact expectation values by default.** As an option, the features are estimated from `S`
  bitstrings sampled from the diagonal of ρ, as the mean of `b_i` (and of `b_i b_j`). The
  estimate is unbiased with standard error `√(p(1 − p)/S) ≤ 1/(2√S)` for a feature of value
  `p`. `S` has no default and is stated with every result that uses it, so the effect of
  finite statistics is shown separately from that of the model.

## Noise

**One mechanism: Markovian dephasing during the evolution,** given by a rate γ and the
evolution time `T`. The state follows the Lindblad equation

```text
dρ/dt = −i [H, ρ] + Σ_i ( L_i ρ L_i† − ½ { L_i† L_i , ρ } ),        L_i = √γ n_i
```

with one collapse operator per atom, independent of the others.

- **What γ means.** Between basis states `a` and `b`, `n_i ρ n_i` contributes `a_i b_i ρ_ab`
  and the anticommutator `−½ (a_i + b_i) ρ_ab`, together `−½ (a_i − b_i)² ρ_ab`. Summed over
  the atoms,

    ```text
    (dρ_ab/dt)_dephasing = −(γ/2) · d_H(a, b) · ρ_ab
    ```

    with `d_H` the Hamming distance. **The coherence of a single atom decays at rate `γ/2`**
    (`T₂ = 2/γ` without a drive), and populations are untouched. Over a time τ the dissipator
    alone multiplies ρ elementwise by `exp(−γ τ d_H(a, b) / 2)`, exactly.

- **Other ways to write it.** `√γ n_i` and `√(γ/4) Z_i` give the same equation. A collapse
  operator `√κ Z_i` corresponds to `γ = 4κ`.
- **One atom.** With `p = ρ_rr` and `c = ρ_gr`:

    ```text
    dp/dt = Ω · Im c
    dc/dt = −(γ/2 + iΔ) c − i (Ω/2)(2p − 1)
    ```

    At `Δ = 0` this is a damped oscillator, `q'' + (γ/2) q' + Ω² q = 0` for `q = p − ½`, with
    `q(0) = −½` and `q'(0) = 0`:

    ```text
    γ < 4Ω:   ⟨n⟩(t) = ½ − ½ e^(−γt/4) [ cos λt + γ/(4λ) · sin λt ],     λ = √(Ω² − γ²/16)
    γ = 4Ω:   ⟨n⟩(t) = ½ − ½ e^(−Ωt) (1 + Ωt)
    γ > 4Ω:   ⟨n⟩(t) = ½ − ½ e^(−γt/4) [ cosh κt + γ/(4κ) · sinh κt ],   κ = √(γ²/16 − Ω²)
    ```

    Rabi oscillations decay with the envelope `e^(−γt/4)`, and `γ = 4Ω` is critical damping.
    After the π pulse, `⟨n⟩ ≈ 1 − πγ/(8Ω)` for `γ ≪ Ω`.

- **Why local operators.** The elementwise action above is exact, and without interactions
  both `H` and the dissipator are sums of single-atom terms, so the state stays a product of
  single-atom states. The control stays analytic under noise: each of its features solves the
  two equations above for its own `Δ_i`. Its range shrinks with γ (from 0.72 to 0.21 at
  `γ = Ω`), and it stayed strictly decreasing in `Δ_i` at every rate checked numerically
  (`γ/Ω` from 0.03 to 10); that is an observation, not a proof.
- **Strong dephasing.** Every `L_i` is Hermitian, so the generator is unital and the maximally
  mixed state `1/2^N` is stationary for any `H`. For `Ω > 0` and `γ > 0` it is the only
  stationary state. The Hamiltonian part does not change the purity and the dissipator gives
  `d/dt Tr ρ² = −γ Σ_i ‖[n_i, ρ]‖²` (Hilbert-Schmidt norm), so a stationary state commutes
  with every `n_i`. An operator that commutes with every `n_i` is diagonal, and the dissipator
  vanishes on it, so a stationary one also commutes with `H`; a diagonal operator that
  commutes with `H` is a multiple of the identity, because the drive connects all basis
  states. So `ρ(t) → 1/2^N`, every `⟨n_i⟩ → ½` and every `⟨n_i n_j⟩ → ¼` as `t → ∞`,
  whatever the input and the interaction.

    How fast is set by `g`, the smallest nonzero decay rate of the Liouvillian (minus the
    largest real part among its nonzero eigenvalues): at long times the distance from `1/2^N`
    falls as `e^(−gt)`, times a power of `t` where that eigenvalue is defective (`t e^(−gt)`
    at critical damping, `γ = 4Ω` and `Δ = 0`), so the limit is reached when `gt ≫ 1`.
    `γt ≫ 1` does not ensure that. For one atom at `Δ = 0` the closed form above gives
    `g = γ/4` up to `γ = 4Ω` and `g = γ/4 − κ` above it, exactly. `g` is largest at `γ = 4Ω`,
    where it equals Ω, and falls beyond it, approaching `2Ω²/γ` only for `γ ≫ 4Ω` (the exact
    `γ/4 − κ` is 1.64 times that at `γ = 4.1 Ω`, 1.04 times at `10 Ω`): at `γ = 100 Ω` and
    `Ωt = 10`, `γt = 1000` and `⟨n⟩ = 0.0905`.

    The rest of this item is numerical, from the spectrum and the exponential of the
    Liouvillian of one to four atoms, and is not derived here. Without interactions `g` is the
    smallest of the single-atom values; over `Δ ∈ [0, Δ_max]` and `γ/Ω` from 0.01 to 1000 a
    single-atom `g` was never below 1/3.5 of its `Δ = 0` value (lowest ratio at `γ = 4Ω`,
    `Δ = Δ_max`, where `g = 0.289 Ω`), and a small detuning can raise it. Strong interactions
    lower `g` without bound: for `V ≫ Ω, γ` it is of order `γΩ²/V²` (at `Δ = 0`,
    `0.667 γΩ²/V²` for two atoms and `0.432 γΩ²/V²` for three, within 2.5 % for `V/Ω ≥ 10`
    and `γ/Ω` from 0.1 to 3), so two atoms at `γ = Ω` and `Ωt = 40` still have
    `⟨n_i⟩ = 0.372` at `V/Ω = 10` and 0.334 at 100. A point that does reach the limit:
    `γ = 4Ω`, `Ωt = 100`, `V/Ω` of 0 or 1 puts every element of ρ within `10⁻¹²` of `1/2^N`
    for one to four atoms, at every set of detunings in `[0, Δ_max]` that was sampled.

Two idealisations are part of this choice:

- **Doppler noise is a different model and is left out.** An atom's velocity is constant over
  one pulse, so its Doppler shift is a detuning offset that is fixed within a shot and random
  between shots. Its effect is an average of unitary evolutions over that offset, not a
  Lindblad term.
- **Decay of the Rydberg state is neglected, which requires `T ≪ τ_r`.** For `70S` of ⁸⁷Rb,
  `τ_r ≈ 150 µs` at 300 K including blackbody radiation (Bernien et al. 2017, from Beterov et
  al. 2009). At the reference Ω, `T = π/Ω = 0.25 µs`, so `T/τ_r = 1.7 × 10⁻³`: an excited atom
  decays during the pulse with a probability below 0.2 %. `T ≤ τ_r/100` holds for
  `Ω ≥ 2.1 rad/µs`.

## Closed-form limits

Each convention above is fixed by a limit that the tests of #496 and #497 check; the last row
is checked by the feature map (#499).

| Convention | Closed-form check |
|---|---|
| Factor of 2 in Ω | One atom, `Δ = γ = 0`: `⟨n⟩(t) = sin²(Ωt/2)`; the matrix is `[[0, Ω/2], [Ω/2, −Δ]]` |
| Sign of Δ, sign of `C6` | Two atoms at distance `r`: the diagonal entry of the doubly excited state is `−Δ_1 − Δ_2 + C6/r⁶` |
| Detuned Rabi formula | One atom, `γ = 0`: `⟨n⟩(t) = Ω²/Ω_eff² · sin²(Ω_eff t/2)` |
| Basis ordering | A detuning on atom `i` alone changes exactly the diagonal entries with `b_i = 1` |
| Full tail | Three atoms in a chain: the next-nearest coupling is 1/64 of the nearest |
| Interactions off | `H` is the Kronecker sum of single-atom Hamiltonians, and ρ their tensor product, with and without dephasing |
| Blockade | Two atoms, `V ≫ Ω`, `Δ = γ = 0`: `⟨n_1 + n_2⟩ → sin²(√2 Ωt/2)` as `Ω/V → 0`, and the doubly excited population stays below `(Ω/V)²` (its maximum over `Ωt ≤ 20` was 0.57, 0.52 and 0.51 times that at `V/Ω` = 10, 30 and 100) |
| What γ means | No drive: a single-atom coherence decays as `e^(−γt/2)` and `ρ_ab` as `e^(−γ t d_H(a,b)/2)` |
| Dephasing against the drive | One atom, `Δ = 0`: the damped oscillation above, in all three cases |
| Strong dephasing | `Ω > 0`, `γ > 0`, any input and any `V`: `ρ → 1/2^N` and every `⟨n_i⟩ → ½` once `gt ≫ 1`, with `g` the smallest nonzero decay rate of the Liouvillian (one atom, `Δ = 0`: `γ/4` up to `γ = 4Ω`, `γ/4 − κ` above). `γt ≫ 1` alone is not sufficient. Reached at `γ = 4Ω`, `Ωt = 100`, `V/Ω` of 0 or 1 |
| Pulse and bound | Interactions off, `γ = 0`: feature `i` equals `f(√(1 + 3σ(x_i)²))` and does not change with any other input |

## Separation measures

A classification score does not say why a fixed feature map helps or fails. The measures below
are properties of the map on a set of inputs, computed without any trained head: on the states
`ρ(x)` (`RydbergFeatureMap.states`), or on the feature matrix `F` the head actually sees
(`RydbergFeatureMap.transform`, one row per sample), with binary labels `y`. The feature
measures are in `hqnn_forge.diagnostics` (`separation_measures` computes all of them); their
definitions, derivations and edge cases are in that module's docstring.

| Measure | Computed from | Range | Statement it supports |
|---|---|---|---|
| State kernel `K(x, x′) = Tr[ρ(x) ρ(x′)]`, and its centred alignment with the labels | `kernel_from_density_matrices`, `kernel_target_alignment` | `K ∈ [0, 1]`, purity on the diagonal; alignment in `[0, 1]` | An upper bound on what any measurement of these states could distinguish (below) |
| Pairwise feature distances `‖F(x) − F(x′)‖`, within and between classes, and the ratio of the two means | `pairwise_distances` | Distances in `[0, √N]`; ratio in `(0, ∞]`, about 1 without class information | How far apart this readout puts two inputs, in units of excitation probability: the scale to hold against the shot error `≤ 1/(2√S)` of [Readout](#readout) |
| Fisher discriminant ratio along the best linear direction, `J = δᵀ S_w⁻¹ δ` with the pooled within-class covariance `S_w` (the squared Mahalanobis distance between the class means) | `fisher_discriminant_ratio` | `[0, ∞]`, 0 for equal class means | How far apart a linear function of the readout puts the class means, in units of the spread within the classes |
| Effective rank of the feature covariance | `effective_rank` | `[1, min(N, M − 1)]` for `M` samples | In how many directions the readout varies: features collapsing onto fewer directions lower it |
| Centred alignment of the linear feature kernel `F Fᵀ` with the labels | `linear_feature_kernel`, `kernel_target_alignment` | `[0, 1]`, 1 for features that are affine in the label | How much of the readout's variation is the label, to set beside the alignment of the state kernel |

- **The state kernel bounds what any measurement could distinguish.** Its entries give the
  Hilbert–Schmidt distance of two states,
  `‖ρ − ρ′‖² = Tr[(ρ − ρ′)²] = K(x, x) + K(x′, x′) − 2 K(x, x′)` (the diagonal is the purity,
  not 1). For any observable `A` and any number `c`, because `Tr[ρ − ρ′] = 0` and by the
  Cauchy–Schwarz inequality, with `‖·‖` the Hilbert–Schmidt norm throughout,

    ```text
    | Tr[A ρ] − Tr[A ρ′] | = | Tr[(A − c) (ρ − ρ′)] | ≤ ‖A − c‖ · ‖ρ − ρ′‖
    ```

    For the readout, `n_i − ½ = −Z_i / 2`, and the `Z_i / √(2^N)` are orthonormal in the
    Hilbert–Schmidt inner product, so the squared feature differences sum to at most the
    squared norm of `ρ − ρ′` (Bessel's inequality) times `2^N / 4`:

    ```text
    ‖F(x) − F(x′)‖ ≤ ½ √(2^N) · ‖ρ(x) − ρ(x′)‖
    ```

    with equality when `ρ − ρ′` is a combination of the `Z_i`, as for one atom and two diagonal
    states. This holds for the `N` excitation probabilities, not for the appended pair
    correlations. Two inputs with the same state give the same value of every measurement, and
    the bound is far from tight in general: the state has `4^N − 1` real parameters, the readout
    `N`.
- **The feature measures describe what this readout distinguishes,** and nothing about the
  state beyond it: the readout can leave a large state distance unused, which the bound
  allows and the distances show. The alignments of the two kernels are not ordered either: a
  readout that drops variation unrelated to the label can align better than the state kernel.
- **None of them predicts the test score of a particular head.** They are computed on one
  sample with its labels, without held-out data, a model or a threshold. The Fisher ratio and
  the feature alignment see linear structure only, which a nonlinear head is not limited to,
  and both grow on a small sample without any class difference behind them (`J = ∞` with fewer
  samples than features plus two). For `M` samples of `d` features in classes of `n0` and `n1`
  that do not differ, `J` is about `d · M/(n0 n1)` on average, roughly `d` over the size of
  the rarer class: values of `J` from sets of different size or class balance compare only
  where they are large against that level, and with unequal class covariances `J` weights the
  larger class more. The Fisher ratio gives an error rate, `Φ(−√J / 2)`, only
  for two Gaussian classes of equal covariance and prior, which excitation probabilities are
  not.
- **Under strong dephasing every distance falls to 0,** since every `⟨n_i⟩ → ½` (see
  [Noise](#noise)). The ratios, the rank and the alignments do not depend on the scale of the
  features, so a collapse alone does not change them; they change because inputs lose their
  trace in the state at different rates. On the way to the limit they moved towards their
  no-information values in the cases tried (`γ = 4Ω`, three atoms, `V/Ω` of 0 and 1: Fisher
  ratio from 19 and 17 after the π pulse to 1.6 and 1.1 at `Ωt = 30`); that is an observation.
- **No monotonicity is claimed.** The Hamiltonian depends on the input, so two inputs do not
  pass through the same channel and the data-processing inequality does not apply to their
  states directly. Whether dephasing contracts the state or feature distances with γ in this
  model is an open question, not an assumption of the code.

## Hardware realism

The comparison is with QuEra's Aquila, an analog device of ⁸⁷Rb atoms using the same Rydberg
state, as described by Wurtz et al. (2023) and the Amazon Braket documentation (AWS 2024). A
device run of this model is out of scope; this section only says which assumptions one would
meet.

| Assumption | On a current device | Status |
|---|---|---|
| Hamiltonian, units, `C6` | Aquila's native Hamiltonian has this form, with the phase convention above and `C6` of `70S` | Supported |
| Chain, spacings 4 to 15 µm | Sites at least 4 µm apart in a region 75 µm wide: a row of 6 atoms fits up to `a = 15 µm`, so `0.04 ≤ V/Ω ≤ 105` at the reference Ω | Supported |
| Fixed positions | Positions vary by 0.2 µm from shot to shot, which near `a = R_b` spreads `V` by about 20 % (`6 √2 · 0.2 µm / a`); Wurtz et al. advise placing atoms well inside or outside `R_b` | Idealisation, strongest at `V ≈ Ω` |
| Pulse bounds | `Ω ≤ 15.8 rad/µs`, `−125 ≤ Δ ≤ 125 rad/µs`, at most 4 µs: the reference `Ω = 12.57`, `Δ_max = 21.8 rad/µs` and `T = 0.25 µs` are inside | Supported |
| Constant pulse | Ω may change by at most 250 rad/µs², so each edge of the pulse takes 0.05 µs of the 0.25 µs | Idealisation |
| Local addressing: one detuning per atom | Ω and the phase are global. A local detuning `−Δ_local(t) · h_k n_k` with a static pattern `0 ≤ h_k ≤ 1` exists as an experimental capability, on request (AWS 2024); with the global detuning it reaches a one-signed range. Its magnitude limits and the resolution of `h_k` were not verified for this page | Idealisation |
| Dephasing rate | Aquila reports `T₂* = 5.8 µs`, an echo time of 11.4 µs and a Rabi decay time of 7.5 µs. Read as this model's `2/γ`, `2/γ` and `4/γ` and divided by the maximum Ω, they give `γ/Ω` between 0.01 and 0.04 | Indicative only: those decays include shot-to-shot noise and loss, which are not Markovian dephasing. Larger `γ/Ω` is a stress test, not a device regime |
| No decay | `T/τ_r = 1.7 × 10⁻³` for the Rydberg lifetime, `5 × 10⁻³` against the 50 µs effective lifetime of Bernien et al. | Good approximation |
| No detection error | Aquila misreads a ground-state atom as excited with probability 0.01 and an excited atom as ground with 0.08; Bernien et al. report detection fidelities of 98 to 99 % and 93 %. With independent errors `ε` and `ε′` a feature becomes `ε + (1 − ε − ε′) ⟨n_i⟩` | Idealisation |
| Exact expectation values | 1 to 1000 shots per task at fewer than 10 shots per second, and each input sample is its own task; about 100 shots are typical | Idealisation; the shots option models the statistics |

**The encoding is an idealisation.** A detuning that is set freely and independently for each
atom is more than current devices offer as a standard capability. A device-compatible variant
would have to give up:

- **the exact `Δ_i`:** each sample's detunings become one pattern `h_k` of finite resolution,
  scaled by a waveform shared by all atoms;
- **the square pulse and exact features:** ramped edges, and at most 1000 shots per sample
  with the detection errors above;
- **on a device with global detuning only, one input per atom itself.** The inputs would have
  to enter global pulse parameters or the positions, and the control without interactions
  would no longer have features that each depend on one input.

## References

- AWS (2024), "Submit an analog program using QuEra Aquila", *Amazon Braket Developer Guide*,
  <https://docs.aws.amazon.com/braket/latest/developerguide/braket-quera-submitting-analog-program-aquila.html>;
  and Komar, Bylinskii, Becker & Lin (2024), "Local detuning now available on QuEra's Aquila
  device with Braket Direct", *AWS Quantum Technologies Blog*, 17 April 2024.
- Bernien, Schwartz, Keesling, Levine, Omran, Pichler, Choi, Zibrov, Endres, Greiner, Vuletić
  & Lukin (2017), "Probing many-body dynamics on a 51-atom quantum simulator", *Nature* 551,
  579–584, doi:10.1038/nature24622, arXiv:1707.04344.
- Beterov, Ryabtsev, Tretyakov & Entin (2009), "Quasiclassical calculations of
  blackbody-radiation-induced depopulation rates and effective lifetimes of Rydberg nS, nP,
  and nD alkali-metal atoms with n ≤ 80", *Physical Review A* 79, 052504,
  doi:10.1103/PhysRevA.79.052504.
- Wurtz, Bylinskii, Braverman, Amato-Grill, Cantu, Huber, Lukin, Liu, Weinberg, Long, Wang,
  Gemelke & Keesling (2023), "Aquila: QuEra's 256-qubit neutral-atom quantum computer",
  arXiv:2306.11727.
