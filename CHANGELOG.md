# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `noise_method="trajectories"` for amplitude and phase damping: phase damping is sampled as the
  phase flip it equals, amplitude damping by weighted Kraus branches whose mean is the density
  channel exactly; the latter needs backprop, parameter-shift or finite-diff, `default.qubit` or
  `lightning.qubit`, and exact expectation values (#357)
- Project scaffold and packaging setup
- PCANormalizer (pure-NumPy) for quantum angle encoding
- Small-angle restricted-variance initialiser (the σ formulas are this library's own
  heuristics, not a published prescription)
- Angle-embedding quantum encoding layer with adjoint diff
- Strongly-entangling and hardware-efficient VQC primitives
- Focal loss and weighted BCE for imbalanced classification
- HybridBinaryClassifier end-to-end model
- Smoke tests for quantum encoding layer
- Quick-start example for training
- `entangler` (`"ring"` / `"strongly_entangling"`) and `readout` (`"all"` / `"first"`) on the
  encoding layers; `embedding_rotation`, `entangler`, `readout`, `encoder_activation` and
  `init_strategy="normal"` with `init_std` on the classifiers, and `published_shnn()` on both
  `HybridBinaryClassifier` (the thesis's 122-parameter SHNN configuration) and
  `ParallelHybridClassifier` (that quantum branch beside the classical MLP branch)
- `entangler="brickwork"`: nearest-neighbour CNOT pairs without wrap-around, so each ⟨Z_i⟩
  readout keeps a local light cone at shallow depth; at 2 layers its total gradient variance
  stays flat from 4 to 8 qubits where the ring's falls 4.6x
- `entangler="hardware_efficient"`: a nearest-neighbour `CZ` ladder then `RY` on every qubit
  (Kandala et al. 2017), backed by `hqnn_forge.circuits.hardware_efficient_layer`, in every
  encoder but amplitude and in the classifiers. Its weights have shape `(n_layers, n_qubits)`,
  not `(n_layers, n_qubits, 3)`: code that assumes a trailing Euler-angle axis should read the
  shape from `variational_weight_shape`. The default `"ring"` block now runs through
  `hqnn_forge.circuits.strongly_entangling_layer`, gate for gate as before
- `AmplitudeEncodingLayer`: amplitude embedding of up to `2**n_qubits` features per sample,
  with zero-padding and L2 normalisation in `forward`, ahead of the same entangling ansatz;
  gradients with respect to the inputs are only supported under `backprop`
- `DataReuploadingLayer`: angle embedding repeated before every variational layer
  (Pérez-Salinas et al. 2020), with the same `entangler` and `readout` options and optional
  trainable per-upload input scaling; `rotation="Z"` requires `n_layers >= 2` and leaves its
  first upload, a phase on `|0⟩`, unscaled
- `apply_variational_layers` takes a `layer_offset`, so a circuit applying the blocks one at
  a time keeps the `"strongly_entangling"` ranges of the whole ansatz
- `hqnn_forge.kernels.quantum_kernel_matrix`: pairwise state-fidelity kernel
  `|⟨Φ(x_i)|Φ(x_j)⟩|²` from any encoding layer, for `SVC(kernel="precomputed")`
- `hqnn_forge.diagnostics.fisher_information_matrix` and `effective_dimension`: Fisher
  information spectrum of the quantum weights and the effective dimension of Abbas et al. (2021);
  the effective dimension requires `κ = γn / (2π log n) ≥ e` (`n_data ≥ 74` at `γ = 1`), below
  which the formula changes sign or explodes
- `lightning.gpu` and `lightning.kokkos` accepted as `device_name`, with a fallback chain
  through `lightning.qubit` to `default.qubit` and a `RuntimeWarning` per step
- `MulticlassHybridClassifier`: shared quantum layer with `n_classes` linear heads, softmax or
  one-vs-rest probabilities, argmax `predict`; supported by `save_checkpoint` /
  `load_checkpoint`
- `noise_level` / `noise_position` on `QuantumEncodingLayer`, `IQPEncodingLayer`,
  `HybridBinaryClassifier` and `ParallelHybridClassifier`: depolarizing noise applied in train
  mode so gradients flow through the noisy circuit, practical up to about 6 qubits;
  `examples/noise_aware_training.py` compares noiseless and noise-aware training under the
  post-hoc noise sweep. Both are weight-safe `load_checkpoint` overrides, and a checkpoint
  from before them loads as noiseless. `apply_depolarizing_noise` at `p = 0` suppresses the
  training channel, and `gradient_variance` measures the layer in eval mode, so neither sees
  the train-mode noise
- `CircuitSummary.n_inert_params` / `count_inert_parameters`: trainable gate parameters that
  can never reach a measurement, found structurally and counted as a lower bound; with
  `readout="all"` they are the `n_qubits` last-layer `Rot` ω angles, with `readout="first"`
  many more (12 of 24 for the ring ansatz at 4 qubits and 2 layers).  `n_effective_params`
  appears in `to_dict()` and the printed summary; templates are decomposed before counting
  and broadcast tapes are rejected
- `hqnn_forge.evaluation.pr_auc`: average precision (PR-AUC) in pure torch, with the step
  interpolation of scikit-learn's `average_precision_score` and tied probabilities as one
  operating point; threshold-free, so it stays out of `METRICS` and `find_optimal_threshold`
  refuses it
- `init_seed` on `HybridBinaryClassifier`, `ParallelHybridClassifier` and
  `MulticlassHybridClassifier`: seeds weight initialisation from a private RNG, so the same
  seed gives the same weights and the global torch RNG is left exactly as it was, also when
  the constructor raises. It is recorded in `get_config()`, so rebuilding from a seeded
  model's config repeats its initial weights
- `hqnn_forge.utils.train_mode`: the train-mode counterpart of `eval_mode`, which keeps a
  submodule the caller froze in eval mode and restores every submodule's mode on exit
- `load_credit_card_fraud(download=True)` passes the Kaggle CLI's progress through as it runs,
  stops a download that prints nothing for 120 s or runs past 3600 s with a
  `DatasetDownloadError` saying which (and repeating the command to run by hand), and removes
  the partial files a failed or interrupted download created. Both bounds are fixed; the
  loader's signature is unchanged

- `ClassicalBaseline` (a plain MLP with the classifiers' interface) and
  `hqnn_forge.utils.classical_baseline(model)`, which builds the untrained classical control
  of a hybrid model with its trainable parameter count matched to the hybrid's, every rotation
  angle counted as one parameter. `ClassicalBaseline` takes `init_seed` like the other
  classifiers, and the builder carries the hybrid's `init_seed` over, so a seeded hybrid gets
  a seeded control
- `hqnn_forge.benchmark.run_benchmark`: a hybrid model against its matched classical control
  on identical folds of several datasets, one row per dataset and model (MCC, parameters,
  MCC per 1,000 parameters, training time, paired Wilcoxon test), with `write_csv` and
  `examples/benchmark.py`
- `hqnn_forge.experiment`: an experiment record (JSON) of a benchmark run with its config,
  seeds, fold indices, dependency versions, devices used, dataset fingerprints and metrics,
  written by `run_benchmark(record_path=...)`; `load_record` reports version differences from
  the current environment and `rerun_benchmark` repeats the run
- `hqnn_forge.utils.permute_quantum_layer`: the permutation null for quantum ablation, which
  runs the circuit and shuffles its readouts across the batch with a seedable generator, so
  they keep their distribution and lose only their link to the input
- Comparing several models over several datasets (Demšar 2006) in `hqnn_forge.evaluation`:
  `friedman_test` with the Iman–Davenport F, `nemenyi_critical_difference`,
  `compare_to_control` and `holm_correction`, NumPy-only
- `examples/does_the_quantum_layer_help.py`: a step-by-step hybrid-versus-control comparison on
  one's own data, with a plain-words verdict from the paired Wilcoxon test
- Any PennyLane device name as `device_name`: the four simulators keep their fallback chain;
  any other name (a plugin or hardware) is constructed as given, and PennyLane's own error
  surfaces if it cannot be, so a typo still raises instead of falling back (#314)
- `shots: int | None = None` on the four encoding layers, the three classifiers and
  checkpoints (weight-safe, `None` for older files): a finite count samples every readout,
  needs `diff_method="parameter-shift"` (adjoint and backprop are refused at construction)
  and trains with noise only under `noise_method="trajectories"`. `hqnn_forge.noise.apply_shots` evaluates a
  model at a shot count inside a block and `shot_sweep` repeats `predict_proba` across shot
  counts. `shots=None` is bit-identical to before. The scikit-learn estimator does not take
  `shots` yet (#411) (#314)
- `hqnn_forge.training.SPSA`: simultaneous perturbation stochastic approximation (Spall 1992)
  as a `torch.optim.Optimizer`, two loss evaluations per step whatever the parameter count,
  with Spall's gain schedules and common random numbers for the torch RNG. `train_model` calls
  a gradient-free optimiser's `step(closure)` without a backward pass. `gradient_optimizer=`
  trains the parameters after the circuit (the classical head) with a gradient optimiser such
  as Adam from the same two evaluations, at no extra circuit executions (#316)

### Changed
- `MulticlassHybridClassifier` builds the same encoder → circuit → dropout trunk as
  `HybridBinaryClassifier`, from shared code, so it now accepts every trunk option:
  `embedding_rotation`, `entangler`, `readout` (the head reads `Linear(n_outputs, n_classes)`),
  `encoder_activation`, `noise_level` / `noise_position` / `noise_method` /
  `noise_trajectories`, and a custom `classical_encoder`. Each defaults to the behaviour the
  model had before, and a multiclass checkpoint that predates them loads as before, with the
  "predates" warning. `save_checkpoint` / `load_checkpoint` are typed against the new
  head-agnostic `ClassifierBase`. The binary models' constructors do not change
- `load_checkpoint` fills constructor arguments a checkpoint predates from
  `checkpoint._LEGACY_DEFAULTS` — the behaviour from before each argument existed — with a
  `RuntimeWarning` naming them, instead of refusing the file. A checkpoint written before the
  options added above still rebuilds the model it holds; a config missing anything else is
  still refused
- The restricted-variance initialiser is no longer described as barren-plateau-safe or
  -aware. Measured with `gradient_variance` (2 layers, ⟨Z_0⟩ cost), it gives no gain in initial
  gradient variance for inputs spread over (-π, π), which both classifiers feed the circuit, and
  a gain growing from 1.1x to 1.75x between 4 and 8 qubits near zero input; over (-π, π) the
  ~3x decay per two qubits is the same under either init. README and docstrings now state that
- Raised the `pennylane` and `pennylane-lightning` floors from `>=0.38` to `>=0.45`, the lowest
  version CI runs; 0.38 is incompatible with `autoray>=0.7`, and 0.42 was only tested on
  Python 3.10
- Raised the `torch` floor from `>=2.2` to `>=2.3` and the `numpy` floor from `>=1.26` to
  `>=2.0`. `pennylane>=0.45` requires NumPy 2, and the torch 2.2 wheels were compiled against
  NumPy 1.x and fail to initialise NumPy 2; CI now installs every declared floor of the runtime
  dependencies and the `lightning` and `sklearn` extras (`test-lowest` job) and fails if one
  is unreachable, so a floor that stops working fails a PR instead of a user install
- `PCANormalizer`'s fitted attributes (`mean_`, `components_`, `explained_variance_`,
  `std_`) are absent before `fit`, as in scikit-learn, instead of set to `None`; reading one
  on an unfitted instance raises `AttributeError`. Check `is_fitted_` instead of comparing an
  attribute to `None`
- `block_local_init_` (and `init_strategy="block_local"`) draws layer ℓ of an L-layer circuit
  with σ_ℓ = scale / sqrt(n_qubits · (L + ℓ)) instead of scale / sqrt(n_qubits · (ℓ + 1)), so
  every σ_ℓ² is O(1/L): layer 0 gets the `restricted_normal_init_` σ and later layers taper
  by up to √2. The old first layer did not shrink with depth, sqrt(L) wider than the global
  scheme. Every `block_local` model now initialises differently, and since the whole tensor is
  now drawn in one call it advances the global RNG as `restricted` does, so seeded draws made
  after building the model (DataLoader shuffles, dropout masks) change too
- `restricted_normal_init_` and `block_local_init_` emit a `UserWarning` naming both standard
  deviations when the σ they draw is not narrower than a uniform draw over [0, 2π)
  (2π/sqrt(12) ≈ 1.81 rad), which at the default `scale = π` means `n_qubits * n_layers <= 3`;
  such a toy circuit was silently initialised wider than the regime the init exists to avoid.
  The warning is attributed to the first frame outside `hqnn_forge`, so a classifier built at
  such a size reports the user's own line. `load_checkpoint` and `gradient_variance`, whose
  draws are discarded or deliberately small, do not emit it
- `circuit_summary` decomposes a `MultiRZ` on more than two wires into one- and two-qubit
  gates before counting, so `n_two_qubit_gates` is the circuit's two-qubit cost: a k-wire
  `MultiRZ` counts as its 2(k-1) CNOTs instead of once. No circuit in the library emits such a
  gate today, so no current summary changes
- `HybridClassifierEstimator.fit` no longer reseeds the global torch RNG: with `random_state`
  set, the model draws its initial weights with `init_seed=random_state`, dropout masks and
  batch order come from seeds spawned from it, and the caller's stream is restored
  afterwards. A given `random_state` therefore yields different initial weights, dropout
  masks and batch order than before

- `gradient_variance` measures layers with several trainable tensors, such as
  `DataReuploadingLayer(trainable_input_scaling=True)`, instead of refusing them:
  `total_variance` sums over the whole gradient vector, the init draws only the `weights`
  angles, and the new `GradientVarianceResult.per_tensor` maps each tensor to its variance.
  `per_parameter` keeps the weight tensor's shape for a single-tensor layer and is the flat
  concatenation otherwise
- `fisher_information_matrix` and `effective_dimension` measure layers with several trainable
  tensors: the matrix spans all of them in the TorchLayer's argument order,
  `FisherSpectrum.parameter_slices` and `block(name)` locate each tensor, and `parameters=`
  measures a subset, whose matrix is the matching block. `effective_dimension` counts every
  tensor in `d` but, like `gradient_variance`, draws only the `weights` angles

### Fixed
- `circuit_summary` and `count_inert_parameters` raised `TypeError` with PennyLane's
  graph-based decomposition enabled (`qml.decomposition.enable_graph()`), which requires a
  `gate_set`; both now pass one. The library's own layers count the same in both modes, and
  `GlobalPhase` ops the graph emits are not counted; a gate outside `LOGICAL_GATE_SET`
  (`CRX`, `Toffoli`, ...) may be decomposed by a different rule, so its counts can differ
- Device fallback raised `AttributeError` on PennyLane 0.45, where `qml.DeviceError` no longer
  exists; the chain now catches `pennylane.exceptions.DeviceError` and is exercised by a test
- `disable_quantum_layer` fills the quantum layer's readout width (`n_outputs`) rather than
  `n_qubits`, so an ablated model built with `readout="first"` matches its `Linear(1 → 1)` head
  instead of raising a shape error where the real model works
- `QuantumEncodingLayer` and `build_encoding_qnode` reject an unknown `rotation` axis at
  construction, where `entangler` and `readout` are already rejected; it used to construct
  cleanly and fail inside PennyLane on the first forward pass
- `HybridClassifierEstimator.fit` raised on a NumPy integer `random_state` (as scikit-learn
  tooling passes) in `torch.Generator().manual_seed`; it is now taken as the int it is
- `train_model` no longer calls `model.train()` each epoch, which unfroze submodules the
  caller had put in eval mode (a frozen batch-norm layer's statistics kept moving); it trains
  under `train_mode` and returns the model in the modes it had on entry, so a model passed in
  eval mode now comes back in eval mode. With `batch_size > 1`, a trailing batch of one sample
  is merged into the batch before it, so a batch-norm model no longer fails when
  `n % batch_size == 1`

### Removed
- `black` from the `dev` extra; `ruff format` is the only formatter, sharing the
  `line-length = 99` ruff already enforces, so the two tools can no longer disagree
- `requirements.txt`; `pyproject.toml` is now the only place dependencies are declared
- Python 3.10 support; `requires-python` is now `>=3.11`. 3.10 reaches end of life in
  October 2026 and current PennyLane releases no longer install on it, so CI tests 3.11
  (the floor) and 3.14 (the newest) instead
