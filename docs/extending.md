# Extending HQNN-Forge

How to add a dataset loader, an encoding layer or a variational block (ansatz) without breaking the
conventions the rest of the library relies on. Each section covers:

- the interface and its conventions,
- the reference implementation to copy,
- the tests a new implementation must have,
- what else has to change alongside it.

The workflow itself (issue first, branch naming, commit style, PR checklist) is in
[`CONTRIBUTING.md`](https://github.com/hqnn-forge/hqnn-forge/blob/main/CONTRIBUTING.md).

---

## A dataset loader

**Reference:** `hqnn_forge/data/uci.py` (`load_taiwanese_bankruptcy` and the two other UCI
loaders) and its tests in `tests/test_data_uci.py`, for a file fetched over HTTPS; this is the
pattern to copy for most new datasets. `hqnn_forge/data/credit_card.py`
(`load_credit_card_fraud`, tests in `tests/test_data_credit_card.py`) shows a download through
an external CLI (the Kaggle client) instead.

### Interface and conventions

- **One public function, `load_<name>(path=None, *, download=False, ...)`.** Every option after
  `path` is keyword-only. It returns a `BinaryDataset` (from `hqnn_forge.data`; do not define a
  new `NamedTuple`) with `X` (float64, shape `(n_samples, n_features)`, C-contiguous), `y`
  (int64, shape `(n_samples,)`, 1 = the positive, minority class) and `feature_names` (a tuple
  with one name per column of `X`).
- **Explicit path, then environment, then default.** `path` may name the file or the directory
  holding it. A path whose name is not the dataset's file name counts as a directory, because
  with `download=True` the directory often does not exist yet. With no `path`, the loader looks
  under `$HQNN_FORGE_DATA`, then under `data/raw/`; reuse `DATA_DIR_ENV` and `DEFAULT_DIR` from
  `credit_card.py` rather than spelling them again.
- **Nothing is fetched unless asked.** A missing file raises `DatasetNotFoundError` (from
  `hqnn_forge.data`). Its message gives the exact command or URL that fetches the file and
  mentions `download=True`. If `download=True` and fetching fails (tool missing, command
  failed, nothing produced), the loader raises `DatasetDownloadError`. It must not raise
  `DatasetNotFoundError` there, so that a caller who catches the first error and retries with
  `download=True` cannot loop.
- **Check what was downloaded.** Record the published file's SHA-256 and compare the
  downloaded bytes against it before writing anything; a mismatch is a `DatasetDownloadError`
  that says nothing was written. Bound the request with a timeout (`DOWNLOAD_TIMEOUT`), and
  write to a `.part` file that is renamed into place, so an interrupted download never leaves a
  truncated file for the next load to find. `_download` in `uci.py` does all three.
- **Validate the schema before trusting the numbers.** Check the header against the published
  column names and report the first differences. Reject a header-only file, a non-numeric value,
  and labels outside {0, 1}, each with a `ValueError` that names the file. `strict=True`
  additionally requires the published file (its SHA-256, where one is recorded) and checks the
  published row and positive counts, so a truncated or re-sampled copy is caught.
- **NumPy only.** The core package does not depend on pandas; parse with `np.loadtxt` or the
  standard library.
- **Leakage-prone columns are flags, not silent drops.** An example is `drop_time`. The default
  returns the published columns unchanged. The exception is a column that encodes the label
  itself (the other examinations in `load_cervical_cancer_risk`): drop it always, and name it in
  the docstring.
- **No preprocessing.** Scaling, PCA and resampling belong to `hqnn_forge.preprocessing`,
  applied inside each cross-validation fold. A loader that standardised the data would leak
  test-fold statistics into training.

### Tests (`tests/test_data_<name>.py`)

The dataset itself is never downloaded in CI. Tests write a small file with the real header to
`tmp_path` and monkeypatch the download (`urllib.request.urlopen`, or `shutil.which` and
`subprocess.run` for a CLI). A loader in `uci.py` joins the `LOADERS` list in
`tests/test_data_uci.py`, which runs the location, `strict` and download tests
(`TestEveryLoader`) against it. At minimum:

- arrays match the file: values, dtypes (float64 / int64), shapes, `feature_names`, and the
  positive rate of the fixture;
- file path, directory path, `$HQNN_FORGE_DATA` and the default location all resolve;
- a missing file raises `DatasetNotFoundError`, and the message contains the fetch command;
- a download that fails, returns something other than the expected file, or has the wrong
  checksum raises `DatasetDownloadError` and writes nothing; a download that succeeds is
  loaded;
- wrong column count, a renamed column, header only, a non-numeric value and a non-binary label
  each raise `ValueError`;
- `strict=True` accepts the published file and counts and rejects others (monkeypatch the
  expected checksum and counts so a small fixture can pass).

### Also update

- `hqnn_forge/data/__init__.py`: the import, `__all__`, and the table in the module docstring.
- The module docstring of the file the loader lives in: `uci.py` keeps a table of rows,
  features and positives, and states the licence and the DOI to cite (say so if the file may
  not be redistributed). Also the `data/` line in the README's folder structure.
- Default to a location under `data/raw/`, which `.gitignore` excludes. Never commit a dataset,
  even a small one; tests build their own fixture files.

---

## An encoding layer

**Reference:** `IQPEncodingLayer` in `hqnn_forge/encoding/iqp_embedding.py`, the shortest complete
encoder. `AmplitudeEncodingLayer` shows a `prepare_inputs` that transforms its input, not just
checks it.

### Interface and conventions

An encoding layer is an `nn.Module` that satisfies `hqnn_forge.encoding.EncodingLayer`. The
contract is spelled out in `hqnn_forge/_encoding_contract.py`:

- **`qlayer`**: a `qml.qnn.TorchLayer`. The QNode's first argument is named `inputs`; every other
  argument is a trainable weight declared in `weight_shapes`.
- **`n_qubits`**: an `int`, the circuit's width.
- **`n_features`**: an `int`, the input's width, meaning the number of features per sample that
  `prepare_inputs` accepts. It is `n_qubits` for the angle-type layers and up to `2**n_qubits`
  for the amplitude layer. The diagnostics (`circuit_summary`, `gradient_variance`) size their
  sample inputs by it, and `is_encoding_layer` (and with it the kernels, such as
  `quantum_kernel_matrix`) refuses a layer without it.
- **`prepare_inputs(x)`**: the *whole* classical step between the batch and the QNode, meaning
  validation (call `check_inputs(x, self.n_features, name="n_features")` from
  `hqnn_forge.encoding._common`, which rejects the wrong width and NaN/inf) and any transform. `forward(x)`
  must be exactly `self._run_circuit(self.prepare_inputs(x))`, which is
  `self.qlayer(self.prepare_inputs(x))` but for training-time noise (below). The kernels (`hqnn_forge.kernels`)
  replay the circuit on `prepare_inputs(X)`, so a step done inline in `forward` is silently
  skipped there, and the kernel describes a different feature map without raising.

The one allowed exception is training-time noise. A layer built with `noise_level > 0` runs a
noisy circuit in `train()` mode. Inherit `TrainingNoiseMixin` from `hqnn_forge.noise`, take
`noise_level`, `noise_position`, `noise_method` and `noise_trajectories` in the constructor, call
`self._init_training_noise(...)` once the QNode exists, run the circuit through
`self._run_circuit`, and append `self._noise_repr()` to `extra_repr`, as every encoding layer
does.

Beyond the protocol:

- **Build the QNode through the shared helpers** in `hqnn_forge/encoding/_common.py`:
  - `resolve_device(device_name, n_qubits)` provides the fallback chain to `default.qubit`.
  - `expand_batch_dimension(qnode, diff_method)` makes a batched `inputs` work under adjoint.
  - `apply_variational_layers` and `measure_z` provide the ansatz and the readout.
  - `validate_circuit_options` rejects a bad option at construction, not at the first forward.
- **Name the angle tensor `weights`**, with dim 0 indexing layers. `gradient_variance` and the
  Fisher diagnostics find the rotation angles by that name (or as the only tensor), and the
  initialisers draw per layer along dim 0. Expose `n_layers` as an attribute. Set `n_outputs` to
  the readout width; `disable_quantum_layer` fills that many outputs.
- **State the input range in the class docstring.** By default the models feed
  `tanh(·)·π ∈ (−π, π)` to the encoder. An encoding that expects something else, such as the
  IQP phases `x_i x_j`, must say so, and the model docstrings must say what they feed it.
- **Do not transform the QNode.** The kernel replay reads the untransformed tape. A compile pass
  or a noise transform on the QNode makes `quantum_kernel_matrix` raise rather than return the
  kernel of a different circuit.

### Tests

- **Register the layer in `ENCODERS` in `tests/test_encoding_contract.py`.** That runs the
  contract tests: mypy checks the class against the protocol, `is_encoding_layer` must accept
  it, `prepare_inputs` must accept `n_features` inputs and reject one more or one fewer as
  `forward` does, `forward(x)` must equal `qlayer(prepare_inputs(x))` bit for bit, and
  `prepare_inputs` must reject what `forward` rejects. Include a configuration that exercises
  any transform in `prepare_inputs`. Build it on `N_QUBITS` qubits; if its `n_features` is
  not `n_qubits`, extend the expected width in `test_satisfies_the_runtime_check`.
- Add it to `ALL_LAYERS` in `tests/test_kernels.py`. That runs the kernel tests (symmetry, PSD,
  unit diagonal, rejection of bad inputs) and checks that the replayed states reproduce the
  layer's own `forward`.
- A module of its own, `tests/test_<name>_embedding.py`, which must check:
  - the output shape and the ⟨Z⟩ range;
  - gradients reaching every weight;
  - batched output equal to per-sample output;
  - input validation;
  - the circuit against a hand-written or template version, compared numerically on fixed
    inputs, not only by shape.
- If the layer can sit in a model, extend the parametrisations in `tests/test_circuit_options.py`
  and `tests/test_circuit_summary.py` that list encoder classes.

### Also update

- `hqnn_forge/encoding/__init__.py`: the import, `__all__`, and the table in the docstring.
- The list of accepted layers in the `TypeError` message of `resolve_encoding_layer`
  (`hqnn_forge/_resolve.py`).
- If the models should offer it, the `encoding_type` branches in `hqnn_forge/models/`, their
  `get_config` round trip (checkpoints store the config) and the model docstrings.
- The README feature table and folder layout.

---

## A variational block (ansatz)

**Reference:** `apply_variational_layers` in `hqnn_forge/encoding/_common.py`. The
`"hardware_efficient"` branch, the most recent addition, is the example to copy: a primitive in
`hqnn_forge/circuits/` applied once per layer, with a weight shape of its own. `"brickwork"`
shows a block written inline, and `"strongly_entangling"` one that depends on the layer index.

The functions in `hqnn_forge/circuits/` are the per-layer blocks the encoders run:
`strongly_entangling_layer` is `entangler="ring"` (not `"strongly_entangling"`, which is
PennyLane's `qml.StronglyEntanglingLayers`) and `hardware_efficient_layer` is
`entangler="hardware_efficient"`. The block the encoders use is selected by their `entangler`
argument, so a new ansatz is a new primitive there plus a new `entangler` value; a primitive
alone has no effect on the models.

### Interface and conventions

- **Signature:** a block is applied as
  `apply_variational_layers(weights, n_qubits, n_layers, entangler, layer_offset)` inside the
  QNode, and records gates only.
- **Weight shape.** `variational_weight_shape(entangler, n_qubits, n_layers)`, next to
  `apply_variational_layers`, is the one definition of the `weights` shape: every encoder
  registers its `weights` from it, and `extra_repr` counts `n_params` from the layer's
  parameters. The `Rot` blocks take `(n_layers, n_qubits, 3)` and `"hardware_efficient"`
  takes `(n_layers, n_qubits)`; a block with another shape adds a branch there and nowhere
  else. Within the shape, dim 0 must stay
  the layer index, which is what `block_local_init_` and the diagnostics' `n_layers` fallback
  read.
- **`layer_offset`.** `DataReuploadingLayer` applies the blocks one at a time, with an embedding
  in between. A block whose gates depend on the layer index (as the strongly-entangling ranges
  do) must use `layer_offset + ℓ`, so that one-at-a-time and all-at-once give the same circuit.
- **Small circuits.** Define the block on `n_qubits = 1` or refuse it in
  `validate_circuit_options` (the strongly-entangling branch shows the one-qubit case).
- **Inert parameters.** A rotation that commutes with everything after it up to a Z readout
  never changes a prediction. `circuit_summary` reports these as `n_inert_params`. Check the
  count for the new block and, if it is not zero, say so in the docstring.

### Tests (`tests/test_circuit_options.py`, class `TestEntangler`)

- The block matches a hand-written gate sequence or a PennyLane template, compared on the state
  or the expectation values for fixed weights. `"brickwork"` has a test class of its own next to
  `TestEntangler`; do the same.
- Which features each readout sees after one and two layers: add rows for the new value to the
  parametrisation of `TestReadoutFeatures.test_features_each_readout_sees` in
  `tests/test_circuit_options.py`.
- Applying layers one by one with `layer_offset` gives the same circuit as all at once.
- One qubit, and the smallest `n_qubits` it accepts.
- Gradients reach every weight.
- The gate and parameter counts that `circuit_summary` reports.
- An unknown `entangler` value still raises.

### Also update

- The `Entangler` `Literal` and a branch in `apply_variational_layers` (with its docstring
  entry), both in `_common.py`, and a branch in `variational_weight_shape` if the
  block's shape differs. `validate_circuit_options` and its error message read
  the choices from the `Literal`, so they need no change.
- The `entangler` parameter docstrings: the angle, IQP and re-uploading builders and layers, and
  `hqnn_forge/models/` (`HybridBinaryClassifier`, `ParallelHybridClassifier`).
- The README feature table, if it is offered as a model option.
