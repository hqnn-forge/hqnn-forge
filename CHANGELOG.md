# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Packaging (`pyproject.toml`, `uv_build`), a PEP 561 `py.typed` marker and the Apache-2.0
  `LICENSE` file (#2, #86, #87)
- `PCANormalizer`: pure-NumPy PCA plus standardisation, with optional scaling of the
  components to (−π, π) for angle encoding (#4). It warns when degenerate eigenvalues make
  `components_` non-reproducible (#107)
- `restricted_normal_init_` and `block_local_init_`: small-angle initialisation of the circuit
  weights (the σ formulas are this library's own heuristics, not a published prescription)
  (#6, #109)
- `QuantumEncodingLayer`: angle embedding followed by an entangling variational ansatz, as an
  `nn.Module` around a PennyLane `TorchLayer`, differentiated with adjoint on
  `lightning.qubit` by default (#8)
- `hqnn_forge.circuits`: strongly-entangling and hardware-efficient ansatz primitives (#10)
- `FocalLoss` and `weighted_bce_loss` with `compute_class_weights` for imbalanced
  classification (#12)
- `HybridBinaryClassifier`: classical encoder → quantum layer → linear head (#14), and a
  quick-start example (#18)
- `IQPEncodingLayer`: IQP embedding (Hadamards, `RZ(x_i)` and pairwise `x_i x_j` phases,
  Havlíček et al. 2019) ahead of the same ansatz (#24)
- `ParallelHybridClassifier`: a classical MLP branch beside the quantum branch, concatenated
  into one linear head (#56)
- `hqnn_forge.diagnostics.circuit_summary` and `draw_circuit`: depth, gate and trainable
  parameter counts of a layer or classifier, and a text drawing of its circuit (#116)
- `hqnn_forge.evaluation`: `find_optimal_threshold` (MCC-optimal decision threshold),
  `matthews_corrcoef`, `f1_score`, `balanced_accuracy` and `parameter_efficiency` (#117);
  `wilcoxon_signed_rank` and `rank_biserial_correlation` for paired model comparisons (#119);
  `plot_confusion_matrix`, `plot_fold_metric_boxplot` and `plot_efficiency_frontier` (#125)
- `hqnn_forge.training.train_model`: a training loop with validation monitoring, early
  stopping and threshold selection, returning a `TrainingHistory` (#118)
- `hqnn_forge.sklearn.HybridClassifierEstimator`: a scikit-learn classifier around the binary
  models, for `cross_val_score`, `GridSearchCV` and `Pipeline` (#130)
- `save_checkpoint` / `load_checkpoint`: class, constructor arguments and weights in one file
  (#120)
- `disable_quantum_layer`: a context manager that replaces the quantum layer's output with a
  constant, for ablation studies (#121)
- `hqnn_forge.diagnostics.gradient_variance` and `gradient_variance_sweep`: variance of the
  cost gradient over random weight draws, for barren-plateau checks (#123)
- `hqnn_forge.preprocessing`: `stratified_kfold` / `iter_folds` and fold-safe `smote` /
  `oversample_fold`, which oversample the training fold only (#124)
- `hqnn_forge.data.load_credit_card_fraud`: loader for the Kaggle Credit Card Fraud dataset
  with an opt-in download and schema checks (#126)
- `hqnn_forge.noise.apply_depolarizing_noise` and `noise_sweep`: post-hoc depolarizing noise
  on a trained model's circuit, for NISQ robustness testing (#127)
- `entangler` (`"ring"` / `"strongly_entangling"`) and `readout` (`"all"` / `"first"`) on the
  encoding layers; `embedding_rotation`, `entangler`, `readout`, `encoder_activation` and
  `init_strategy="normal"` with `init_std` on the classifiers, and `published_shnn()` on both
  `HybridBinaryClassifier` (the thesis's 122-parameter SHNN configuration) and
  `ParallelHybridClassifier` (that quantum branch beside the classical MLP branch) (#159)
- `AmplitudeEncodingLayer`: amplitude embedding of up to `2**n_qubits` features per sample,
  with zero-padding and L2 normalisation in `forward`, ahead of the same entangling ansatz;
  gradients with respect to the inputs are only supported under `backprop` (#148)
- `DataReuploadingLayer`: angle embedding repeated before every variational layer
  (Pérez-Salinas et al. 2020), with the same `entangler` and `readout` options and optional
  trainable per-upload input scaling; `rotation="Z"` requires `n_layers >= 2` and leaves its
  first upload, a phase on `|0⟩`, unscaled (#149)
- `apply_variational_layers` takes a `layer_offset`, so a circuit applying the blocks one at
  a time keeps the `"strongly_entangling"` ranges of the whole ansatz (#149)
- `hqnn_forge.kernels.quantum_kernel_matrix`: pairwise state-fidelity kernel
  `|⟨Φ(x_i)|Φ(x_j)⟩|²` from any encoding layer, for `SVC(kernel="precomputed")` (#151)
- `hqnn_forge.diagnostics.fisher_information_matrix` and `effective_dimension`: Fisher
  information spectrum of the quantum weights and the effective dimension of Abbas et al. (2021);
  the effective dimension requires `κ = γn / (2π log n) ≥ e` (`n_data ≥ 74` at `γ = 1`), below
  which the formula changes sign or explodes (#153)
- `lightning.gpu` and `lightning.kokkos` accepted as `device_name`, with a fallback chain
  through `lightning.qubit` to `default.qubit` and a `RuntimeWarning` per step (#155)
- `MulticlassHybridClassifier`: shared quantum layer with `n_classes` linear heads, softmax or
  one-vs-rest probabilities, argmax `predict`; supported by `save_checkpoint` /
  `load_checkpoint` (#156)
- `noise_level` / `noise_position` on `QuantumEncodingLayer`, `IQPEncodingLayer`,
  `HybridBinaryClassifier` and `ParallelHybridClassifier`: depolarizing noise applied in train
  mode so gradients flow through the noisy circuit, practical up to about 6 qubits;
  `examples/noise_aware_training.py` compares noiseless and noise-aware training under the
  post-hoc noise sweep. Both are weight-safe `load_checkpoint` overrides, and a checkpoint
  from before them loads as noiseless. `apply_depolarizing_noise` at `p = 0` suppresses the
  training channel, and `gradient_variance` measures the layer in eval mode, so neither sees
  the train-mode noise (#157)
- `CircuitSummary.n_inert_params` / `count_inert_parameters`: trainable gate parameters that
  can never reach a measurement, found structurally and counted as a lower bound; with
  `readout="all"` they are the `n_qubits` last-layer `Rot` ω angles, with `readout="first"`
  many more (12 of 24 for the ring ansatz at 4 qubits and 2 layers).  `n_effective_params`
  appears in `to_dict()` and the printed summary; templates are decomposed before counting
  and broadcast tapes are rejected (#165)
- `entangler="brickwork"`: nearest-neighbour CNOT pairs without wrap-around, so each ⟨Z_i⟩
  readout keeps a local light cone at shallow depth; at 2 layers its total gradient variance
  stays flat from 4 to 8 qubits where the ring's falls 4.6x (#237)
- `hqnn_forge.encoding` exports every encoder and its circuit helpers: `IQPEncodingLayer`,
  `build_iqp_qnode`, `apply_variational_layers`, `readout_wires`, `measure_z`,
  `input_scaling_shape` and the option literals (#242)
- `hqnn_forge.evaluation.pr_auc`: average precision (PR-AUC) in pure torch, with the step
  interpolation of scikit-learn's `average_precision_score` and tied probabilities as one
  operating point; threshold-free, so it stays out of `METRICS` and `find_optimal_threshold`
  refuses it (#244)
- `init_seed` on `HybridBinaryClassifier`, `ParallelHybridClassifier` and
  `MulticlassHybridClassifier`: seeds weight initialisation from a private RNG, so the same
  seed gives the same weights and the global torch RNG is left exactly as it was, also when
  the constructor raises. It is recorded in `get_config()`, so rebuilding from a seeded
  model's config repeats its initial weights (#248)
- `hqnn_forge.utils.train_mode`: the train-mode counterpart of `eval_mode`, which keeps a
  submodule the caller froze in eval mode and restores every submodule's mode on exit (#249)
- `ClassicalBaseline` (a plain MLP with the classifiers' interface) and
  `hqnn_forge.utils.classical_baseline(model)`, which builds the untrained classical control
  of a hybrid model with its trainable parameter count matched to the hybrid's, every rotation
  angle counted as one parameter. `ClassicalBaseline` takes `init_seed` like the other
  classifiers, and the builder carries the hybrid's `init_seed` over, so a seeded hybrid gets
  a seeded control (#250)
- `hqnn_forge.utils.permute_quantum_layer`: the permutation null for quantum ablation, which
  runs the circuit and shuffles its readouts across the batch with a seedable generator, so
  they keep their distribution and lose only their link to the input (#251)
- `load_credit_card_fraud(download=True)` passes the Kaggle CLI's progress through as it runs,
  stops a download that prints nothing for 120 s or runs past 3600 s with a
  `DatasetDownloadError` saying which (and repeating the command to run by hand), and removes
  the partial files a failed or interrupted download created. Both bounds are fixed; the
  loader's signature is unchanged (#254)
- `hqnn_forge.data.uci`: `load_taiwanese_bankruptcy`, `load_iranian_churn` and
  `load_cervical_cancer_risk`, further imbalanced binary tabular benchmarks (CC BY 4.0), with
  an opt-in download checked against the published SHA-256 and validated headers, values and
  labels; `strict=True` also checks the documented counts (#266)
- `classical_encoder` on `HybridBinaryClassifier` and `ParallelHybridClassifier`: a
  user-supplied `nn.Module` in place of the built-in `Linear`, followed by the same activation
  and π scaling; its output width is checked at construction, and `save_checkpoint` refuses a
  model that has one (#267)
- `hqnn_forge.benchmark.run_benchmark`: a hybrid model against its matched classical control
  on identical folds of several datasets, one row per dataset and model (MCC, parameters,
  MCC per 1,000 parameters, training time, paired Wilcoxon test), with `write_csv` and
  `examples/benchmark.py` (#269)
- `hqnn_forge.experiment`: an experiment record (JSON) of a benchmark run with its config,
  seeds, fold indices, dependency versions, devices used, dataset fingerprints and metrics,
  written by `run_benchmark(record_path=...)`; `load_record` reports version differences from
  the current environment and `rerun_benchmark` repeats the run (#270)
- `examples/does_the_quantum_layer_help.py`: a step-by-step hybrid-versus-control comparison on
  one's own data, with a plain-words verdict from the paired Wilcoxon test (#271)
- `CITATION.cff`, so GitHub offers "Cite this repository" with BibTeX and APA output; its
  `version` and `license` are kept equal to `pyproject.toml` by a test (#272)
- Comparing several models over several datasets (Demšar 2006) in `hqnn_forge.evaluation`:
  `friedman_test` with the Iman–Davenport F, `nemenyi_critical_difference`,
  `compare_to_control` and `holm_correction`, NumPy-only (#273)
- `bootstrap_ci` and `paired_bootstrap_ci` in `hqnn_forge.evaluation`: class-stratified
  bootstrap intervals (BCa or percentile) for MCC, F1, balanced accuracy or any callable
  metric, and for the difference between two models on the same samples; NumPy-only (#274)
- `hqnn_forge.kernels.overlap_kernel_matrix`: the kernel estimated entry by entry from the
  compute-uncompute circuit with optional shots and any device, and `nearest_psd` to
  project an estimate onto the positive semi-definite cone (#277)
- Trainable quantum kernels: `differentiable=True` on `encoded_states` and
  `quantum_kernel_matrix`, `kernel_target_alignment` (centred) and `train_kernel_alignment`
  to fit a `DataReuploadingLayer`'s weights to a task before the SVM (#278)
- `batch_size` on `encoded_states` and `quantum_kernel_matrix`: the kernel states are simulated
  that many rows at a time, bounding the simulator's working memory; results are unchanged (#282)
- Quantum kernels under depolarising noise: `noise_level`/`noise_position` on
  `quantum_kernel_matrix` give the Hilbert–Schmidt kernel `Tr[ρ(x)ρ(y)]` with the same channel
  insertion as `hqnn_forge.noise`; `encoded_density_matrices` and `kernel_from_density_matrices` (#283)
- The published SHNN's live parameter count: 102 of its 122 trainable parameters can move
  the output (autograd, pinned by a test); the 20 dead circuit weights are kept so published
  checkpoints load unchanged, and both counts are documented (#293)
- `run_benchmark(n_seeds=...)`: each fold trained with several recorded initialisation seeds,
  fold scores averaged over them for the paired test, and the across-seed spread reported as
  `mcc_seed_std` (#294)
- `run_benchmark(tuning=Tuning(...))`: random search over training settings with the same
  trial budget and inner folds for the hybrid model and its control, inside each outer training
  split; the chosen settings are recorded (#295)
- `run_benchmark(noise_levels=...)`: each trained hybrid model also scored under
  inference-time depolarising noise, compared per level with the noise-free control, with the
  noise level at which it stops being significantly better (#296)
- `hqnn_forge.sklearn.QuantumKernelClassifier`: an `SVC(kernel="precomputed")` on the fidelity
  kernel of any encoding, with optional kernel-target alignment, depolarising noise and Platt
  probabilities, for `cross_val_score`, `GridSearchCV` and `Pipeline`; not yet usable in
  `run_benchmark` (see #409) (#339)
- The encoding layer contract, exported from `hqnn_forge.encoding`: the `EncodingLayer`
  protocol (`qlayer`, `n_qubits`, `n_features`, `prepare_inputs`, with `forward(x)` equal to
  `qlayer(prepare_inputs(x))` outside training noise), the weaker `CircuitLayer` (`qlayer`,
  `n_qubits`) the diagnostics accept, and the `is_encoding_layer` / `is_circuit_layer` runtime
  checks. `QuantumEncodingLayer`, `IQPEncodingLayer` and `DataReuploadingLayer` gain an
  `n_features` attribute, their input width, equal to `n_qubits` (#298)
- `noise_method="trajectories"` (with `noise_trajectories`) on the encoding layers and
  classifiers: training noise by Pauli-trajectory sampling, whose average output and gradient
  equal the depolarising channel's, on the layer's own device and `diff_method` in O(2^n)
  memory instead of the density path's O(4^n). `"density"` stays the default, and both are
  recorded in checkpoints (#300)
- `AmplitudeEncodingLayer` and `DataReuploadingLayer` accept the training-noise options
  (`noise_level`, `noise_position`, `noise_method`, `noise_trajectories`), so every encoding
  layer and classifier now does, with the same meaning (#302)
- `noise_level`, `noise_position`, `noise_method` and `noise_trajectories` on
  `HybridClassifierEstimator`, tunable through `GridSearchCV`; `fit` trains through the noisy
  circuit and `predict` / `predict_proba` stay noiseless (#303)
- `hqnn_forge.diagnostics.expressibility` (Sim et al. 2019, the KL divergence of the circuit's
  fidelity histogram from Haar, with exact Haar bin probabilities) and `entangling_capability`
  (mean Meyer–Wallach Q with its standard error and the Haar reference), with the building
  block `meyer_wallach`; `expressibility_from_fidelities` and `haar_fidelity_bin_probabilities`
  are in the `hqnn_forge.diagnostics.expressibility` module (#325)
- `entangler="hardware_efficient"`: a nearest-neighbour `CZ` ladder then `RY` on every qubit
  (Kandala et al. 2017), backed by `hqnn_forge.circuits.hardware_efficient_layer`, in the
  angle, IQP and re-uploading encoders and in the classifiers (and, since #333, the amplitude
  encoder). Its weights have shape `(n_layers, n_qubits)`, not `(n_layers, n_qubits, 3)`: code
  that assumes a trailing Euler-angle axis should read the shape from
  `hqnn_forge.encoding.angle_embedding.variational_weight_shape`. The default `"ring"` block now runs through
  `hqnn_forge.circuits.strongly_entangling_layer`, gate for gate as before (#331)
- `encoding_type="reuploading"` and `encoding_type="amplitude"` in `HybridBinaryClassifier`,
  `ParallelHybridClassifier`, `MulticlassHybridClassifier` and the scikit-learn estimator, with
  `trainable_input_scaling` for re-uploading. Amplitude with a classical encoder requires
  `diff_method="backprop"` and is refused at construction otherwise; `AmplitudeEncodingLayer`
  gains `entangler` and `readout`. Options an encoding cannot use raise at construction (#333)
- Multiclass in `HybridClassifierEstimator`: three or more classes train
  `MulticlassHybridClassifier` with a new `strategy` (`"softmax"` with cross-entropy or the
  new `hqnn_forge.utils.SoftmaxFocalLoss`, `"one_vs_rest"` with BCE or focal loss on one-hot targets);
  `predict_proba` is `(n, n_classes)` in `classes_` order and `threshold_` is `None`.
  `train_model` accepts `(batch, n_classes)` logits and monitors them with the new
  `MULTICLASS_METRICS` (`multiclass_matthews_corrcoef`, `macro_f1_score`,
  `multiclass_balanced_accuracy`); their labels must be class indices in `[0, n_classes)`
  in both splits, checked once before the first optimiser step (#338)
- Any PennyLane device name as `device_name`: the four simulators keep their fallback chain;
  any other name (a plugin or hardware) is constructed as given, and PennyLane's own error
  surfaces if it cannot be, so a typo still raises instead of falling back (#334)
- `shots: int | None = None` on the four encoding layers, the three classifiers and
  checkpoints (weight-safe, `None` for older files): a finite count samples every readout,
  needs `diff_method="parameter-shift"` (adjoint, backprop and finite-diff are refused at construction)
  and trains with noise only under `noise_method="trajectories"`. `hqnn_forge.noise.apply_shots` evaluates a
  model at a shot count inside a block and `shot_sweep` repeats `predict_proba` across shot
  counts. `shots=None` is bit-identical to before. The scikit-learn estimator does not take
  `shots` yet (#411) (#334)
- `hqnn_forge.training.SPSA`: simultaneous perturbation stochastic approximation (Spall 1992)
  as a `torch.optim.Optimizer`, two loss evaluations per step whatever the parameter count,
  with Spall's gain schedules and common random numbers for the torch CPU and CUDA RNGs. Its
  default direction generator is seeded from `torch.initial_seed()`, so runs under different
  `torch.manual_seed` seeds draw different directions. `train_model` calls
  a gradient-free optimiser's `step(closure)` without a backward pass. `gradient_optimizer=`
  trains the parameters after the circuit (the classical head) with a gradient optimiser such
  as Adam from the same two evaluations, at no extra circuit executions (#335)
- Noise channels beside depolarizing: `channel=` on `apply_depolarizing_noise` and
  `noise_sweep`, and `noise_channel` on the four encoding layers, the three classifiers and
  checkpoints (weight-safe, `"depolarizing"` for older files), selecting
  `"amplitude_damping"` (T1), `"phase_damping"` (T2 dephasing), `"bit_flip"` or
  `"phase_flip"`, each with its own `p` range (`[0, 1]`; depolarizing stays `[0, 0.75]`). A
  readout error is `"bit_flip"` at `position="end"`, the symmetric case only: an asymmetric
  one (`p01 ≠ p10`) is not modelled. `noise_method="trajectories"` samples the Pauli channels
  (depolarizing, bit flip, phase flip), including inside `apply_shots`; the damping channels
  need `"density"` (#336)
- Calibration of binary probabilities in `hqnn_forge.evaluation`: `brier_score`,
  `expected_calibration_error` and `reliability_curve` (uniform or equal-count bins), the
  post-hoc `TemperatureScaler` and `PlattScaler` fitted on a validation split (raising when
  the NLL has no finite optimum, e.g. on separated classes), `plots.plot_reliability_diagram`,
  `TrainingHistory.temperature` for the returned weights, and per-fold `brier`/`ece` with
  `brier_mean`/`ece_mean` in `run_benchmark` (NaN for a fold whose probabilities are not
  finite) (#340)
- `examples/benchmark_batching.py`: inference and training-step time of the per-sample split,
  native broadcasting, `batch_obs` and `default.qubit`/backprop for every encoder, with a
  correctness check, plus a crossover by qubit count with peak memory; the README now says
  when to train with backprop instead of `lightning.qubit` (#341)
- A rendered API reference (mkdocs-material + mkdocstrings, one page per public module), built
  with `mkdocs build --strict` on every PR so a broken cross-reference, or a name in a documented
  module's `__all__` with no entry, fails CI, and deployed to GitHub Pages from `main` once
  Pages is enabled; a `docs` dependency group installs the tools (#342)
- `seed` on the encoding layers and classifiers seeds the device's shot sampling, which
  `torch.manual_seed` does not reach, so seeded shot-based runs repeat exactly; `seed` is a
  weight-safe checkpoint argument. `SPSA(..., model=model)` starts both evaluations of a step
  from the same device generator state, which cut the shot-noise variance of its gradient
  estimate about fourfold; `train_model` passes the model it trains when `model=` is not
  given (#354, #369)
- `examples/hardware_workflow.py` and a README section: shots, parameter-shift checked against
  backprop, SPSA, per-step circuit counts with `qml.Tracker`, and shot and noise sweeps, with
  how to point it at a real device (#356)
- `noise_method="trajectories"` for amplitude and phase damping: phase damping is sampled as the
  phase flip it equals, amplitude damping by weighted Kraus branches whose mean is the density
  channel exactly; the latter needs backprop, parameter-shift or finite-diff, `default.qubit` or
  `lightning.qubit`, and exact expectation values (#357)
- An asymmetric readout error `(p01, p10)`, applied exactly to each ⟨Z⟩ as
  `(1 − p01 − p10)⟨Z⟩ + (p10 − p01)`: `apply_readout_error` post hoc,
  `noise_sweep(..., channel="readout")`, and `readout_error=` on the layers and classifiers for
  training (weight-safe in checkpoints) (#358)

### Changed
- The package metadata links the repository, issue tracker and changelog, and the README's
  image and file links are absolute, so both work on PyPI (#327)
- The README links the API reference site near the top and from Quick Start, Architecture and
  Methodology, and the package metadata gains a `Documentation` URL (#453)
- The quantum layers run a whole batch in one QNode call instead of looping over samples
  (#104)
- `predict_proba` and `predict` run in eval mode whatever mode the model is in, restoring every
  submodule's `training` flag afterwards, so dropout no longer makes them random (#71); they
  and `count_parameters` live once in `BinaryClassifierBase` (#113)
- `load_checkpoint` fills constructor arguments a checkpoint predates from
  `checkpoint._LEGACY_DEFAULTS` — the behaviour from before each argument existed — with a
  `RuntimeWarning` naming them, instead of refusing the file. A checkpoint written before the
  options added above still rebuilds the model it holds; a config missing anything else is
  still refused (#159)
- The restricted-variance initialiser is no longer described as barren-plateau-safe or
  -aware. Measured with `gradient_variance` (2 layers, ⟨Z_0⟩ cost), it gives no gain in initial
  gradient variance for inputs spread over (-π, π), which both classifiers feed the circuit, and
  a gain growing from 1.1x to 1.75x between 4 and 8 qubits near zero input; over (-π, π) the
  ~3x decay per two qubits is the same under either init. README and docstrings now state that
  (#162)
- Raised the `pennylane` and `pennylane-lightning` floors from `>=0.38` to `>=0.45`, the lowest
  version CI runs; 0.38 is incompatible with `autoray>=0.7`, and 0.42 was only tested on
  Python 3.10 (#80)
- Raised the `torch` floor from `>=2.2` to `>=2.3` and the `numpy` floor from `>=1.26` to
  `>=2.0`. `pennylane>=0.45` requires NumPy 2, and the torch 2.2 wheels were compiled against
  NumPy 1.x and fail to initialise NumPy 2; CI now installs every declared floor of the runtime
  dependencies and the `lightning` and `sklearn` extras (`test-lowest` job) and fails if one
  is unreachable, so a floor that stops working fails a PR instead of a user install (#143)
- `PCANormalizer`'s fitted attributes (`mean_`, `components_`, `explained_variance_`,
  `std_`) are absent before `fit`, as in scikit-learn, instead of set to `None`; reading one
  on an unfitted instance raises `AttributeError`. Check `is_fitted_` instead of comparing an
  attribute to `None` (#164)
- `block_local_init_` (and `init_strategy="block_local"`) draws layer ℓ of an L-layer circuit
  with σ_ℓ = scale / sqrt(n_qubits · (L + ℓ)) instead of scale / sqrt(n_qubits · (ℓ + 1)), so
  every σ_ℓ² is O(1/L): layer 0 gets the `restricted_normal_init_` σ and later layers taper
  by up to √2. The old first layer did not shrink with depth, sqrt(L) wider than the global
  scheme. Every `block_local` model now initialises differently, and since the whole tensor is
  now drawn in one call it advances the global RNG as `restricted` does, so seeded draws made
  after building the model (DataLoader shuffles, dropout masks) change too (#239)
- `restricted_normal_init_` and `block_local_init_` emit a `UserWarning` naming both standard
  deviations when the σ they draw is not narrower than a uniform draw over [0, 2π)
  (2π/sqrt(12) ≈ 1.81 rad), which at the default `scale = π` means `n_qubits * n_layers <= 3`;
  such a toy circuit was silently initialised wider than the regime the init exists to avoid.
  The warning is attributed to the first frame outside `hqnn_forge`, so a classifier built at
  such a size reports the user's own line. `load_checkpoint` and `gradient_variance`, whose
  draws are discarded or deliberately small, do not emit it (#240)
- `circuit_summary` decomposes a `MultiRZ` on more than two wires into one- and two-qubit
  gates before counting, so `n_two_qubit_gates` is the circuit's two-qubit cost: a k-wire
  `MultiRZ` counts as its 2(k-1) CNOTs instead of once. No circuit in the library emits such a
  gate today, so no current summary changes (#243)
- `save_checkpoint` records the constructor arguments the writing version had
  (`known_args`), and `load_checkpoint` refuses a file whose config lacks one of them instead
  of filling it from the legacy defaults; files written before `known_args` load as before
  (#247)
- `HybridClassifierEstimator.fit` no longer reseeds the global torch RNG: with `random_state`
  set, the model draws its initial weights with `init_seed=random_state`, dropout masks and
  batch order come from seeds spawned from it, and the caller's stream is restored
  afterwards. A given `random_state` therefore yields different initial weights, dropout
  masks and batch order than before (#248)
- `gradient_variance` measures layers with several trainable tensors, such as
  `DataReuploadingLayer(trainable_input_scaling=True)`, instead of refusing them:
  `total_variance` sums over the whole gradient vector, the init draws only the `weights`
  angles, and the new `GradientVarianceResult.per_tensor` maps each tensor to its variance.
  `per_parameter` keeps the weight tensor's shape for a single-tensor layer and is the flat
  concatenation otherwise (#252)
- The diagnostics resolve the encoding layer behind a model in one place, which now refuses a
  `bool` `n_qubits` (#253)
- The `torch` and `numpy` floors and the `sklearn` extra's `scikit-learn` floor carry
  per-interpreter markers, so each supported Python has floors with wheels for it: on 3.13
  `torch>=2.5` and `numpy>=2.1`, on 3.14 `torch>=2.9`, `numpy>=2.3.2` and
  `scikit-learn>=1.7.2` (#263)
- `fisher_information_matrix` and `effective_dimension` measure layers with several trainable
  tensors: the matrix spans all of them in the TorchLayer's argument order,
  `FisherSpectrum.parameter_slices` and `block(name)` locate each tensor, and `parameters=`
  measures a subset, whose matrix is the matching block. `effective_dimension` counts every
  tensor in `d` but, like `gradient_variance`, draws only the `weights` angles (#337)
- **Behaviour change:** `device_name` and `diff_method` default to `"auto"` everywhere
  (encoding layers, QNode factories, classifiers including `published_shnn()`, the
  scikit-learn estimator): `default.qubit` with backprop up to 12 qubits, `lightning.qubit`
  with adjoint above, parameter-shift with `shots`, and backprop for amplitude encoding behind
  the classical encoder. The default was `lightning.qubit` with adjoint, about 12× slower for
  batched training at the default 8 qubits (#341, #349). Outputs and gradients agree with the
  old default to float32 rounding, but Adam can take different steps on inert weights (zero
  gradient up to rounding), so training runs are not bit-identical to earlier ones; pass
  `device_name="lightning.qubit", diff_method="adjoint"` to reproduce them. Checkpoints
  written before this change record their explicit names and load unchanged; new ones record
  `"auto"`, and the encoding layers' `repr` shows the device and method actually used (#368)
- `MulticlassHybridClassifier` builds the same encoder → circuit → dropout trunk as
  `HybridBinaryClassifier`, from shared code, so it now accepts every trunk option:
  `embedding_rotation`, `entangler`, `readout` (the head reads `Linear(n_outputs, n_classes)`),
  `encoder_activation`, `noise_level` / `noise_position` / `noise_method` /
  `noise_trajectories`, and a custom `classical_encoder`. Each defaults to the behaviour the
  model had before, and a multiclass checkpoint that predates them loads as before, with the
  "predates" warning. `save_checkpoint` / `load_checkpoint` are typed against the new
  head-agnostic `ClassifierBase`. The binary models' constructors do not change (#301)
- `classical_baseline(model)` matches the control to the hybrid's live parameter count
  (`count_parameters()` minus `circuit_summary(model).n_inert_params`) rather than its total,
  so the published SHNN's control has 101 parameters, not 121, and the parallel model's 523,
  not 571. Efficiency figures (MCC/kParam) still divide by the total (#293)
- Device fallback remembers, per process, a backend that failed to initialise, so later layers
  skip it without probing it again or warning twice (`reset_device_fallback()` forgets it), and
  the fallback warning points at the caller's line rather than inside the library (#285)
- `count_inert_parameters` recognises diagonal gates and measurements structurally rather than
  by name (`Z`-only Pauli sums and products, `probs` / `sample` / `counts`, `CRZ`,
  `ControlledPhaseShift`, adjoint or controlled diagonal gates), so it can count more inert
  parameters on custom circuits; it remains a lower bound (#291)
- `CircuitSummary.n_inert_params` counts weight entries, like `n_trainable_params`, rather than
  gate-parameter slots, so `n_effective_params` can no longer go negative; the new
  `count_inert_weights` gives the per-entry count for any tape. The built-in layers report the
  same numbers as before (#292)
- Forward passes under `torch.no_grad()` skip computing unused adjoint Jacobians in
  quantum encoding layers, cutting evaluation runtime by up to 2.68× (measured on
  lightning.qubit with 8 qubits and batch size 256) (#439)
- The README shows how a benchmark runs as a Mermaid diagram (matched control, shared per-fold
  rules, the reading of the Wilcoxon result), and its architecture diagrams are Mermaid
  instead of ASCII (#451)
- Switched the package build backend from setuptools to `uv_build` and updated the
  package metadata for PEP 639-compatible license handling (#458)

### Fixed
- Encoding layers raise a construction-time `ValueError` when a resolved device has finite
  shots but `shots=None` was requested, guiding users to pass explicit `shots` and
  `diff_method="parameter-shift"` (#431)
- The classifiers applied the `·π` angle scaling to input that bypasses the classical encoder,
  so `PCANormalizer(scale_to_pi=True)` output was scaled twice (#63)
- `PCANormalizer`: `transform` centred with the batch's mean instead of the training mean (#65);
  non-2-D input, too few samples, `n_components < 1` or non-integer, single-feature input and
  training data of rank below `n_components` now raise a clear `ValueError` (#67, #73, #78,
  #83, #106); eigenvector signs are canonicalised, so `components_` no longer depends on the
  LAPACK build (#81)
- The initialiser tests had too little power to reject a flat σ, and one failed at random
  (#22, #105)
- Device fallback raised `AttributeError` on PennyLane 0.45, where `qml.DeviceError` no longer
  exists; the chain now catches `pennylane.exceptions.DeviceError` and is exercised by a test
  (#155)
- `disable_quantum_layer` fills the quantum layer's readout width (`n_outputs`) rather than
  `n_qubits`, so an ablated model built with `readout="first"` matches its `Linear(1 → 1)` head
  instead of raising a shape error where the real model works (#159)
- `QuantumEncodingLayer` and `build_encoding_qnode` reject an unknown `rotation` axis at
  construction, where `entangler` and `readout` are already rejected; it used to construct
  cleanly and fail inside PennyLane on the first forward pass (#159)
- `circuit_summary` and `count_inert_parameters` raised `TypeError` with PennyLane's
  graph-based decomposition enabled (`qml.decomposition.enable_graph()`), which requires a
  `gate_set`; both now pass one. The library's own layers count the same in both modes, and
  `GlobalPhase` ops the graph emits are not counted; a gate outside `LOGICAL_GATE_SET`
  (`CRX`, `Toffoli`, ...) may be decomposed by a different rule, so its counts can differ
  (#243)
- The evaluation plots return the root `Figure` when drawing into an axes on a (nested)
  sub-figure, and raise `ValueError` for a detached axes instead of returning `None` (#246)
- `HybridClassifierEstimator.fit` raised on a NumPy integer `random_state` (as scikit-learn
  tooling passes) in `torch.Generator().manual_seed`; it is now taken as the int it is (#248)
- `train_model` no longer calls `model.train()` each epoch, which unfroze submodules the
  caller had put in eval mode (a frozen batch-norm layer's statistics kept moving); it trains
  under `train_mode` and returns the model in the modes it had on entry, so a model passed in
  eval mode now comes back in eval mode. With `batch_size > 1`, a trailing batch of one sample
  is merged into the batch before it, so a batch-norm model no longer fails when
  `n % batch_size == 1` (#249)
- `weighted_bce_loss` raises `ValueError` for a `reduction` other than `"mean"`, `"sum"` or
  `"none"`, as `FocalLoss` does, instead of silently returning the unreduced per-sample loss
  (#261)
- `circuit_summary`, `draw_circuit` and `gradient_variance` raised on an
  `AmplitudeEncodingLayer`, because they built their sample input with `n_qubits` features;
  they now use the layer's input width and `prepare_inputs`, and the default amplitude sample
  shows the full state preparation, an upper bound on the gate count for any input (#279)
- `circuit_summary` raised `AttributeError` on PennyLane 0.46, whose `specs` resources drop
  `num_gates`, `gate_types` and `gate_sizes`; it now counts depth and gates on the decomposed
  tape itself and gives the same summary on 0.45 and 0.46 (#364)

### Removed
- `PCANormalizer`'s `copy` option, which never had an effect (#77)
- The unused `apply_restricted_init` helper (#114)
- `black` from the `dev` extra; `ruff format` is the only formatter, sharing the
  `line-length = 99` ruff already enforces, so the two tools can no longer disagree (#141)
- `requirements.txt`; `pyproject.toml` is now the only place dependencies are declared (#80)
- Python 3.10 support; `requires-python` is now `>=3.11`. 3.10 reaches end of life in
  October 2026 and current PennyLane releases no longer install on it, so CI tests 3.11
  (the floor) and 3.14 (the newest) instead (#108)
- `rotation="Z"` on `QuantumEncodingLayer` and `build_encoding_qnode`, and
  `embedding_rotation="Z"` on `HybridBinaryClassifier` and `ParallelHybridClassifier`, now
  raise `ValueError`. The single `RZ` embedding acts on `|0…0⟩`, where it is a global phase, so
  the quantum layer returned the same outputs for every input and a `"Z"` kernel was all ones.
  A checkpoint saved with `"Z"` no longer loads; its quantum layer never depended on the input,
  so retrain with `"X"` or `"Y"`. `DataReuploadingLayer(rotation="Z", n_layers >= 2)` is
  unchanged (#275)
