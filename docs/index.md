# hqnn-forge

Hybrid quantum-classical neural networks for tabular binary and multiclass
classification: quantum encoding layers, hybrid classifiers, training with
early stopping and imbalance-aware losses, noise models, quantum kernels and
the diagnostics to judge them, built on PennyLane and PyTorch.

hqnn-forge is not on PyPI; install it from a clone of the repository:

```bash
git clone https://github.com/hqnn-forge/hqnn-forge.git
cd hqnn-forge
pip install -e ".[lightning]"    # the library, plus the fast C++ simulator
```

```python
import torch
from hqnn_forge.models import HybridBinaryClassifier

model = HybridBinaryClassifier(n_input_features=30, n_qubits=8, n_layers=2)
probabilities = model.predict_proba(torch.randn(4, 30))
```

The [README](https://github.com/hqnn-forge/hqnn-forge#readme) covers installation,
device backends and the architectures in detail; this site is the API
reference, generated from the docstrings.

## API reference

| Package | Contents |
|---|---|
| [`hqnn_forge.encoding`](api/encoding.md) | Quantum feature maps (angle, IQP, amplitude, data re-uploading) and the circuit pieces they share |
| [`hqnn_forge.models`](api/models.md) | The hybrid classifiers, their MLP control and a linear classifier |
| [`hqnn_forge.noise`](api/noise.md) | Noise channels (depolarizing, amplitude and phase damping, bit and phase flip), post hoc and during training |
| [`hqnn_forge.kernels`](api/kernels.md) | Quantum fidelity kernels |
| [`hqnn_forge.diagnostics`](api/diagnostics.md) | Circuit summaries, gradient variance, Fisher spectrum, effective dimension, expressibility, entangling capability, separation measures of a fixed feature matrix |
| [`hqnn_forge.evaluation`](api/evaluation.md) | Metrics, threshold search, statistical tests, [plots](api/evaluation.plots.md) |
| [`hqnn_forge.training`](api/training.md) | The training loop |
| [`hqnn_forge.data`](api/data.md) | Dataset loaders |
| [`hqnn_forge.preprocessing`](api/preprocessing.md) | PCA, cross-validation and fold-safe SMOTE |
| [`hqnn_forge.sklearn`](api/sklearn.md) | The scikit-learn estimator |
| [`hqnn_forge.utils`](api/utils.md) | Losses, checkpoints, ablation, evaluation-mode helper |
| [`hqnn_forge.circuits`](api/circuits.md) | Ansatz primitives |
| [`hqnn_forge.initializers`](api/initializers.md) | Small-angle weight initialisation |
| [`hqnn_forge.rydberg`](api/rydberg.md) | Atom register, dense Rydberg Hamiltonian, dephasing (Lindblad) solver and fixed feature map (inputs to excitation probabilities) of a small neutral-atom array ([model](rydberg-model.md)) |
| [`hqnn_forge.benchmark`](api/benchmark.md) | Hybrid model against its matched classical control, on identical folds |
| [`hqnn_forge.experiment`](api/experiment.md) | JSON experiment records: save, load and rerun a benchmark |
