"""
examples/hardware_workflow.py
=============================
Training for a quantum device, end to end, on a simulator standing in for it
(#356).

On hardware every expectation value is estimated from a finite number of
shots, gradients have to come from circuit evaluations rather than from the
simulator's state, and the budget that matters is the number of circuits
executed.  This script walks through the pieces the library has for that:

1. **Check the gradient method.**  Parameter-shift, the method hardware
   supports, is compared against backprop on the same weights, exactly, on
   ``default.qubit`` whatever ``DEVICE`` is: it checks the method, not the
   device, and a device that samples cannot match to 1e-5.
2. **Count the cost.**  One training step with parameter-shift (Adam) and one
   with SPSA, each under ``qml.Tracker``: circuits executed and shots used.
3. **Train with shots.**  SPSA on the circuit weights of a shot-based model
   and Adam on the classical head, whose exact gradients cost no circuits
   (``gradient_optimizer``), with the device seeded and SPSA's two
   evaluations sharing their shot noise.
4. **Evaluate.**  Test MCC under shot noise (``shot_sweep``) and under
   depolarizing noise (``noise_sweep``).

Data: scikit-learn's breast-cancer set (bundled, so this runs offline),
reduced to 4 principal components scaled to [-π, π], fed straight into a
4-qubit circuit (no classical encoder: its gradient would run through the
circuit's input gradient and multiply the parameter-shift cost).

Running on a real device
------------------------
Change ``DEVICE`` to the plugin's device name, e.g. ``"braket.aws.qubit"``
(``amazon-braket-pennylane-plugin``), and give the device its options and
credentials the way the plugin documents.  The library builds the device
from its name alone, so options go in PennyLane's configuration file, which
``qml.device`` reads for every device it builds, e.g. in ``config.toml``::

    [braket.aws.qubit]
    device_arn = "arn:aws:braket:::device/qpu/..."

An option that has to be a Python object rather than a string cannot be
given this way, so a device that needs one, such as ``pennylane-qiskit``'s
``"qiskit.remote"`` (its ``backend``), cannot be used here as it stands.
The library passes ``seed`` to any device it does not know as given, so
``build`` seeds only the simulators in ``KNOWN_DEVICES``.  The
execution counts printed in step 2 are what a run costs there, and they are
the reason to prefer SPSA: its cost per step does not grow with the number of
parameters.

Usage::

    python examples/hardware_workflow.py

Takes about a minute and a half on a laptop CPU.
"""

from __future__ import annotations

import time

import numpy as np
import pennylane as qml
import torch

from hqnn_forge.encoding import DiffMethod
from hqnn_forge.encoding.angle_embedding import KNOWN_DEVICES
from hqnn_forge.evaluation import find_optimal_threshold, matthews_corrcoef
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.noise import noise_sweep, shot_sweep
from hqnn_forge.preprocessing import PCANormalizer
from hqnn_forge.training import SPSA, train_model

#: The device name: the only line to change for hardware (see the docstring).
DEVICE = "default.qubit"
SHOTS = 1000
SEED = 0
N_QUBITS = 4
N_LAYERS = 2
BATCH = 32
EPOCHS = 30


def load() -> tuple[torch.Tensor, ...]:
    """Stratified 60/20/20 split, PCA fitted on the training rows."""
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(SEED)
    parts: list[list[np.ndarray]] = [[], [], []]
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        a, b = int(0.6 * idx.size), int(0.8 * idx.size)
        for part, chunk in zip(parts, (idx[:a], idx[a:b], idx[b:]), strict=True):
            part.append(chunk)
    train, val, test = (np.concatenate(p) for p in parts)
    pca = PCANormalizer(n_components=N_QUBITS).fit(X[train])
    out: list[torch.Tensor] = []
    for rows in (train, val, test):
        out.append(torch.as_tensor(pca.transform(X[rows]), dtype=torch.float32))
        out.append(torch.tensor(y[rows], dtype=torch.float32))
    return tuple(out)


def build(
    shots: int | None, diff_method: DiffMethod, device: str = DEVICE
) -> HybridBinaryClassifier:
    torch.manual_seed(SEED)
    return HybridBinaryClassifier(
        n_input_features=N_QUBITS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        use_classical_encoder=False,
        device_name=device,
        diff_method=diff_method,
        shots=shots,
        # A plugin device is passed seed as given, and may reject it.
        seed=SEED if device in KNOWN_DEVICES else None,
    )


def make_spsa(
    classifier: HybridBinaryClassifier,
    lr: float = 0.1,
    stability: float = 0.0,
    model: torch.nn.Module | None = None,
) -> SPSA:
    """SPSA on everything before the head, Adam on the head.

    The head acts after the circuit, so its exact gradient costs no circuit
    evaluations: SPSA still runs two circuits per sample per step.
    """
    head = list(classifier.head.parameters())
    rest = [p for n, p in classifier.named_parameters() if not n.startswith("head.")]
    adam = torch.optim.Adam(head, lr=0.05)
    return SPSA(
        rest, lr, perturbation=0.1, stability=stability, gradient_optimizer=adam, model=model
    )


def check_gradients(x: torch.Tensor, y: torch.Tensor) -> None:
    """Parameter-shift against backprop, both exact, on the same weights."""
    grads = []
    methods: tuple[DiffMethod, ...] = ("parameter-shift", "backprop")
    for method in methods:
        model = build(None, method, "default.qubit")
        torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y).backward()
        grads.append(model.quantum_layer.qlayer.weights.grad)
    error = (grads[0] - grads[1]).abs().max().item()
    print(f"1. parameter-shift vs backprop, max |Δ grad| = {error:.1e}")
    assert error < 1e-5, "parameter-shift disagrees with backprop"


def count_step(x: torch.Tensor, y: torch.Tensor) -> None:
    """Circuits and shots of one training step, per optimiser."""
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"2. cost of one training step on a batch of {len(x)}, {SHOTS} shots per circuit:")
    for name in ("parameter-shift + Adam", "SPSA"):
        model = build(SHOTS, "parameter-shift")
        device = model.quantum_layer.qlayer.qnode.device
        with qml.Tracker(device) as tracker:
            if name == "SPSA":
                # model= shares the two evaluations' shot noise; train_model
                # passes it itself, a manual step has to.
                spsa = make_spsa(model, model=model)

                def closure(m: torch.nn.Module = model) -> torch.Tensor:
                    return loss_fn(m(x).squeeze(-1), y)

                spsa.step(closure)
            else:
                adam = torch.optim.Adam(model.parameters())
                loss_fn(model(x).squeeze(-1), y).backward()
                adam.step()
        executions = tracker.totals["executions"]
        print(
            f"   {name:24s} {executions:6d} circuits, {tracker.totals['shots']:9d} shots"
            f"  ({executions / len(x):.0f} per sample)"
        )


def main() -> None:
    x_tr, y_tr, x_va, y_va, x_te, y_te = load()
    n_weights = build(None, "backprop", "default.qubit").quantum_layer.qlayer.weights.numel()
    print(f"{N_QUBITS} qubits, {N_LAYERS} layers ({n_weights} circuit weights), device {DEVICE}")

    check_gradients(x_tr[:BATCH], y_tr[:BATCH])
    count_step(x_tr[:BATCH], y_tr[:BATCH])

    # 3. Train with shots: SPSA on the circuit weights, Adam on the head.
    # SPSA's lr is not on Adam's scale: it multiplies a raw gradient estimate,
    # here of size ~0.05, so it needs to be large.  With SPSA on every
    # parameter, lr 0.1-0.4 barely moved this model in 30 epochs and lr 1-8
    # all trained; with the head on Adam, lr 1 and 2 did best of 1-8.
    # stability ~10 % of the ~330 steps, as Spall suggests.
    model = build(SHOTS, "parameter-shift")
    start = time.perf_counter()
    history = train_model(
        model,
        torch.nn.BCEWithLogitsLoss(),
        make_spsa(model, lr=1.0, stability=30),
        x_tr,
        y_tr,
        x_va,
        y_va,
        max_epochs=EPOCHS,
        batch_size=BATCH,
        monitor="mcc",
        patience=None,
        generator=torch.Generator().manual_seed(SEED),
    )
    print(
        f"3. SPSA with {SHOTS} shots: {history.n_epochs} epochs in "
        f"{time.perf_counter() - start:.0f} s, best validation MCC "
        f"{max(r.val_score or 0.0 for r in history.epochs):.3f}"
    )

    # 4. Evaluate: the same weights in an exact copy, swept over shot counts
    # and depolarizing noise.  The threshold is the validation one.
    exact = build(None, "backprop", "default.qubit")
    exact.load_state_dict(model.state_dict())
    threshold = find_optimal_threshold(y_va.long(), exact.predict_proba(x_va)).threshold

    def mcc(labels: torch.Tensor, prob: torch.Tensor) -> float:
        return matthews_corrcoef(labels.long(), (prob >= threshold).long())

    # For reference: the same model trained exactly (backprop, Adam), which a
    # device cannot do.
    reference = build(None, "backprop", "default.qubit")
    train_model(
        reference,
        torch.nn.BCEWithLogitsLoss(),
        torch.optim.Adam(reference.parameters(), lr=0.05),
        x_tr,
        y_tr,
        x_va,
        y_va,
        max_epochs=EPOCHS,
        batch_size=BATCH,
        monitor="mcc",
        patience=None,
        generator=torch.Generator().manual_seed(SEED),
    )
    t_ref = find_optimal_threshold(y_va.long(), reference.predict_proba(x_va)).threshold
    ref_mcc = matthews_corrcoef(y_te.long(), (reference.predict_proba(x_te) >= t_ref).long())

    print(f"4. test MCC (exact backprop + Adam reference: {ref_mcc:.3f})")
    for point in shot_sweep(exact, x_te, [100, SHOTS, None], y=y_te, score_fn=mcc):
        scores = np.array(point.scores)
        label = "exact" if point.shots is None else f"{point.shots} shots"
        print(f"   {label:12s} {scores.mean():.3f} ± {scores.std():.3f}")
    for noisy in noise_sweep(exact, x_te, [0.01, 0.05], y=y_te, score_fn=mcc):
        print(f"   depolarizing p = {noisy.p:.2f} after every gate: {noisy.score:.3f}")


if __name__ == "__main__":
    main()
