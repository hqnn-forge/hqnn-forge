"""
tests/seeded_reference.py
=========================
The seeded-initialisation reference that ``tests/test_seeded_reference.py``
checks against (#310).

A seeded model's initial weights depend on the order in which its modules are
built and initialised: every ``nn.Linear`` and ``TorchLayer`` draws from the
global RNG when constructed, and ``_initialise_weights`` draws again.  A change
to that order changes every seeded result without raising and without failing
a shape test.  With ``init_seed`` a classifier reseeds right before
``_initialise_weights`` (#175), so there only the draws inside it count; the
``*-init-seed`` configurations pin that path, which the scikit-learn estimator
and the benchmark runner take.  This module builds a grid of seeded configurations and
summarises each one's initial state dict and its eval-mode output on a fixed
input; the test compares against the summary stored in
``tests/data/seeded_reference.json``.

Regenerate the reference only for an *intended* change to initialisation, and
say so in the PR::

    uv run python tests/seeded_reference.py

The diff then shows exactly which configurations moved.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    IQPEncodingLayer,
    QuantumEncodingLayer,
)
from hqnn_forge.models import (
    ClassicalBaseline,
    HybridBinaryClassifier,
    LinearClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)

REFERENCE = Path(__file__).parent / "data" / "seeded_reference.json"
SEED = 1234
#: Leading entries stored verbatim per tensor, besides its checksums.
N_VALUES = 4

CPU: dict[str, Any] = {"device_name": "default.qubit", "diff_method": "backprop"}
CLASSIFIER: dict[str, Any] = {"n_input_features": 5, "n_qubits": 3, "n_layers": 2, **CPU}
LAYER: dict[str, Any] = {"n_qubits": 3, "n_layers": 2, **CPU}

#: A seed other than ``SEED``, so an ``init_seed`` that fell back to the
#: global RNG (seeded with ``SEED`` in :func:`summarise`) would not match.
INIT_SEED = 7

#: name -> (builder, width of the fixed input)
CONFIGS: dict[str, tuple[Callable[[], nn.Module], int]] = {
    # The serial binary classifier: every trunk option that changes what is built.
    "serial": (lambda: HybridBinaryClassifier(**CLASSIFIER), 5),
    "serial-iqp": (lambda: HybridBinaryClassifier(**CLASSIFIER, encoding_type="iqp"), 5),
    "serial-block-local": (
        lambda: HybridBinaryClassifier(**CLASSIFIER, init_strategy="block_local"),
        5,
    ),
    "serial-normal-init": (
        lambda: HybridBinaryClassifier(**CLASSIFIER, init_strategy="normal", init_std=0.3),
        5,
    ),
    "serial-no-encoder": (
        lambda: HybridBinaryClassifier(
            **{**CLASSIFIER, "n_input_features": 3}, use_classical_encoder=False
        ),
        3,
    ),
    "serial-published-options": (
        lambda: HybridBinaryClassifier(
            **CLASSIFIER,
            embedding_rotation="Y",
            entangler="strongly_entangling",
            readout="first",
            encoder_activation="sigmoid",
            init_strategy="normal",
        ),
        5,
    ),
    "serial-dropout": (lambda: HybridBinaryClassifier(**CLASSIFIER, dropout_p=0.2), 5),
    "serial-init-seed": (
        lambda: HybridBinaryClassifier(**CLASSIFIER, init_seed=INIT_SEED),
        5,
    ),
    # The parallel classifier: its MLP branch is built first and initialised
    # with its own scheme.
    "parallel": (lambda: ParallelHybridClassifier(**CLASSIFIER), 5),
    "parallel-iqp-hidden": (
        lambda: ParallelHybridClassifier(
            **CLASSIFIER, encoding_type="iqp", classical_hidden_dim=7
        ),
        5,
    ),
    "parallel-first-readout": (
        lambda: ParallelHybridClassifier(
            **CLASSIFIER, readout="first", entangler="strongly_entangling"
        ),
        5,
    ),
    "parallel-init-seed": (
        lambda: ParallelHybridClassifier(**CLASSIFIER, init_seed=INIT_SEED),
        5,
    ),
    # The multiclass classifier.
    "multiclass": (lambda: MulticlassHybridClassifier(**CLASSIFIER, n_classes=3), 5),
    "multiclass-ovr-iqp": (
        lambda: MulticlassHybridClassifier(
            **CLASSIFIER, n_classes=4, strategy="one_vs_rest", encoding_type="iqp"
        ),
        5,
    ),
    "multiclass-normal-init": (
        lambda: MulticlassHybridClassifier(**CLASSIFIER, init_strategy="normal", init_std=0.2),
        5,
    ),
    "multiclass-init-seed": (
        lambda: MulticlassHybridClassifier(**CLASSIFIER, n_classes=3, init_seed=INIT_SEED),
        5,
    ),
    # The classical control (#178), initialised like the hybrid heads.
    "classical-baseline": (lambda: ClassicalBaseline(5, [6, 4]), 5),
    "classical-baseline-init-seed": (
        lambda: ClassicalBaseline(5, [6, 4], init_seed=INIT_SEED),
        5,
    ),
    # The linear head (#501), initialised like the other heads.
    "linear-classifier": (lambda: LinearClassifier(5), 5),
    "linear-classifier-init-seed": (lambda: LinearClassifier(5, init_seed=INIT_SEED), 5),
    # The encoding layers on their own: TorchLayer's own initialisation.
    "layer-angle": (lambda: QuantumEncodingLayer(**LAYER), 3),
    "layer-iqp": (lambda: IQPEncodingLayer(**LAYER, n_repeats=2), 3),
    "layer-amplitude": (lambda: AmplitudeEncodingLayer(**LAYER, n_features=5), 5),
    "layer-reuploading-scaled": (
        lambda: DataReuploadingLayer(**LAYER, trainable_input_scaling=True),
        3,
    ),
}


def _tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    flat = t.detach().to(torch.float64).flatten()
    return {
        "shape": list(t.shape),
        "sum": float(flat.sum()),
        "sum_abs": float(flat.abs().sum()),
        "values": [float(v) for v in flat[:N_VALUES]],
    }


def summarise(name: str) -> dict[str, Any]:
    """Build ``name`` from ``SEED`` and summarise its state dict and eval output."""
    build, width = CONFIGS[name]
    torch.manual_seed(SEED)
    model = build()
    x = torch.linspace(-1.0, 1.0, 4 * width).reshape(4, width)
    model.eval()
    with torch.no_grad():
        output = model(x)
    return {
        "state_dict": {key: _tensor_summary(t) for key, t in model.state_dict().items()},
        "output": _tensor_summary(output),
    }


def main() -> None:
    reference = {
        "seed": SEED,
        "torch": torch.__version__,
        "configs": {name: summarise(name) for name in CONFIGS},
    }
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(CONFIGS)} configurations to {REFERENCE}")


if __name__ == "__main__":
    main()
