"""
examples/benchmark_batching.py
==============================
How a batch runs through an encoding layer, and what that costs (#312).

For every ``diff_method`` except ``backprop``, the encoders split a batch into
one tape per sample before the gradient transform sees it
(``hqnn_forge.encoding._common.expand_batch_dimension``).  This script
compares, for each encoder, inference time (a forward pass with the parameters
frozen) and training-step time (forward+backward) of

* ``split``           -- lightning.qubit / adjoint, one tape per sample (the library default);
* ``native``          -- lightning.qubit / adjoint on the broadcast tape, no library split
                         (the device splits it itself in its preprocessing);
* ``split+batch_obs`` -- the split, on a lightning device built with ``batch_obs=True``;
* ``backprop``        -- default.qubit / backprop, which vectorises the batch;

and checks that every variant's outputs and gradients agree with ``split`` (to a
float32 relative tolerance).  ``--crossover`` then times one training step at
batch 64 over a range of qubit counts for the two main paths and reports the
peak memory of each, in a fresh interpreter per point.

Run::

    python examples/benchmark_batching.py            # the comparison table, batches 1-1024
    python examples/benchmark_batching.py --batches 1 2 4 8  # other batch sizes
    python examples/benchmark_batching.py --crossover

Results on one laptop CPU (PennyLane 0.45.1, pennylane-lightning 0.45.0,
torch 2.14), angle layer, 2 layers, 8 qubits, inference / training step:

    batch 128:   split  235 /  603 ms, native  218 /  521 ms, backprop 16 /  39 ms
    batch 1024:  split 1949 / 4585 ms, native 1846 / 4990 ms, backprop 37 / 205 ms
    batch 64, crossover:  qubits   lightning/adjoint   default.qubit/backprop
                               8     0.36 s   +11 MB      0.03 s    +10 MB
                              10     0.44 s   +14 MB      0.08 s    +55 MB
                              12     0.64 s   +18 MB      0.25 s   +283 MB
                              14     1.47 s   +21 MB      1.66 s  +1125 MB
                              16     8.85 s   +31 MB     10.42 s  +3129 MB

Timings are single runs and vary by some tens of percent between runs, enough
to swap native and split at batch 128.  At batch 1024, over all encoders at 4
and 8 qubits, native took 1.1-1.3x the split's training step and 0.9-1.2x its
inference time, split+batch_obs 1.0-1.1x the split's step, and backprop was
20-210x faster than the split per step and 40-170x per inference.  On the
lightning paths inference is about 40-50 % of the step.  It is only that cheap
with the parameters frozen: PennyLane differentiates whenever a parameter
requires grad, so a forward under ``torch.no_grad()`` alone still computes the
adjoint Jacobian and costs about as much as the whole step (#426).
Native broadcasting is correct on lightning's adjoint path in this PennyLane
version but is not faster than the split: lightning.qubit's own preprocessing
applies ``broadcast_expand``, so ``native`` is the same per-sample split done
on the device, and the split stays.  backprop is the fast path for
batches of small circuits (for a single sample lightning is faster), and loses
on memory from about 14 qubits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import pennylane as qml
import torch

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer

ENCODERS: dict[str, tuple[Callable[..., Any], dict[str, Any], bool]] = {
    # name: (class, extra options, needs one feature per qubit)
    "angle": (QuantumEncodingLayer, {}, True),
    "iqp": (IQPEncodingLayer, {}, True),
    "reuploading": (DataReuploadingLayer, {"trainable_input_scaling": True}, True),
    "amplitude": (AmplitudeEncodingLayer, {}, False),
}


def _variants(cls: Callable[..., Any], n: int, extra: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(0)
    split = cls(
        n_qubits=n, n_layers=2, device_name="lightning.qubit", diff_method="adjoint", **extra
    )

    def copy(device: str, diff_method: str) -> Any:
        layer = cls(n_qubits=n, n_layers=2, device_name=device, diff_method=diff_method, **extra)
        layer.load_state_dict(split.state_dict())
        return layer

    native = copy("lightning.qubit", "adjoint")
    q = native.qlayer.qnode
    native.qlayer.qnode = qml.QNode(q.func, q.device, interface="torch", diff_method="adjoint")
    batch_obs = copy("lightning.qubit", "adjoint")
    q = batch_obs.qlayer.qnode
    batch_obs.qlayer.qnode = qml.transforms.broadcast_expand(
        qml.QNode(
            q.func,
            qml.device("lightning.qubit", wires=n, batch_obs=True),
            interface="torch",
            diff_method="adjoint",
        )
    )
    return {
        "split": split,
        "native": native,
        "split+batch_obs": batch_obs,
        "backprop": copy("default.qubit", "backprop"),
    }


def _run(
    layer: Any, x: torch.Tensor, input_grads: bool
) -> tuple[list[torch.Tensor], float, float]:
    """Outputs and every gradient, the inference time and the training-step time."""
    # Inference, with the parameters frozen: PennyLane decides whether to
    # differentiate from the parameters' requires_grad, not from grad mode, so
    # under torch.no_grad() alone lightning's adjoint path still computes the
    # Jacobian during the forward call.
    params = list(layer.parameters())
    for p in params:
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            start = time.perf_counter()
            layer(x)
            forward = time.perf_counter() - start
    finally:
        for p in params:
            p.requires_grad_(True)
    x = x.clone().requires_grad_(input_grads)
    layer.zero_grad()
    start = time.perf_counter()
    out = layer(x)
    out.sum().backward()
    step = time.perf_counter() - start
    tensors = [out.detach()]
    for p in layer.parameters():
        assert p.grad is not None
        tensors.append(p.grad.clone())
    if input_grads:
        assert x.grad is not None
        tensors.append(x.grad.clone())
    return tensors, forward, step


def _agree(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    # float32: relative to the largest entry, as summed gradients grow with the batch.
    return all(
        (u - v).abs().max() <= 1e-5 * max(1.0, float(u.abs().max()))
        for u, v in zip(a, b, strict=True)
    )


def compare(qubits: tuple[int, ...], batches: tuple[int, ...]) -> None:
    names = ("split", "native", "split+batch_obs", "backprop")
    print("ms, inference (parameters frozen) / training step (forward+backward)")
    print(f"{'encoder':12s} {'n':>2s} {'batch':>5s}  " + "  ".join(f"{v:>26s}" for v in names))
    for name, (cls, extra, per_qubit) in ENCODERS.items():
        # The amplitude layer refuses input gradients outside backprop.
        input_grads = name != "amplitude"
        for n in qubits:
            variants = _variants(cls, n, extra)
            width = n if per_qubit else 2**n
            # One untimed step per variant, so first-call costs stay out of the table.
            for layer in variants.values():
                _run(layer, torch.rand(2, width) + 0.05, input_grads)
            for batch in batches:
                x = torch.rand(batch, width, generator=torch.Generator().manual_seed(1)) + 0.05
                cells = []
                reference: list[torch.Tensor] = []
                for layer in variants.values():  # split first: it is the reference
                    tensors, fwd, step = _run(layer, x, input_grads)
                    reference = reference or tensors
                    flag = "" if _agree(reference, tensors) else " DIFFERS"
                    cells.append(f"{fwd * 1e3:8.1f} /{step * 1e3:8.1f}{flag:>8s}")
                print(f"{name:12s} {n:2d} {batch:5d}  " + "  ".join(f"{c:>26s}" for c in cells))


_POINT = """
import json, resource, sys, time, warnings, torch
warnings.simplefilter("ignore")
from hqnn_forge.encoding import QuantumEncodingLayer
device, method, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(0)
layer = QuantumEncodingLayer(n_qubits=n, n_layers=2, device_name=device, diff_method=method)
# Warnings are off, so a silent fallback to another device has to be caught here.
assert layer.qlayer.qnode.device.name == device, layer.qlayer.qnode.device.name
x = torch.rand(64, n) * 2 - 1
layer(x[:2]).sum().backward()
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start = time.perf_counter()
layer(x).sum().backward()
seconds = time.perf_counter() - start
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# ru_maxrss is in KiB on Linux, in bytes on macOS.
per_mb = 1024 * 1024 if sys.platform == "darwin" else 1024
print(json.dumps({"seconds": seconds, "mb": (peak - base) / per_mb}))
"""


def crossover(qubits: tuple[int, ...]) -> None:
    if sys.platform == "win32":
        sys.exit("--crossover reads peak memory with the resource module, which Windows lacks")
    print(f"{'qubits':>6s}  {'lightning/adjoint':>22s}  {'default.qubit/backprop':>24s}")
    for n in qubits:
        cells = []
        for device, method in (("lightning.qubit", "adjoint"), ("default.qubit", "backprop")):
            out = subprocess.run(
                [sys.executable, "-c", _POINT, device, method, str(n)],
                capture_output=True,
                text=True,
                check=False,
            )
            if out.returncode:
                sys.exit(f"{device}/{method} at {n} qubits failed:\n{out.stderr}")
            point = json.loads(out.stdout.strip().splitlines()[-1])
            cells.append(f"{point['seconds']:7.2f} s {point['mb']:+7.0f} MB")
        print(f"{n:6d}  {cells[0]:>22s}  {cells[1]:>24s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--crossover", action="store_true", help="time a training step by qubit count"
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 16, 128, 1024],
        help="batch sizes for the comparison table",
    )
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    if args.crossover:
        crossover((8, 10, 12, 14, 16))
    else:
        compare(qubits=(4, 8), batches=tuple(args.batches))


if __name__ == "__main__":
    main()
