"""
hqnn_forge.encoding.iqp_embedding
==================================
Quantum encoding module projecting classical tabular feature vectors
into an n-qubit Hilbert space via an Instantaneous Quantum Polynomial (IQP) embedding,
followed by a strongly-entangled variational ansatz.

Design Rationale
----------------
* **IQP Embedding** maps features into a highly entangled state using a diagonal
  Hamiltonian. It applies Hadamards, followed by RZ(x_i) and IsingZZ(x_i * x_j)
  entangling operations. This is known to be classically hard to simulate.
* **Strongly-Entangling Ansatz** — after embedding, L layers of a CNOT ring
  followed by per-qubit SU(2) Rot(φ, θ, ω) gates are applied.
* **Differentiation** — `diff_method="auto"` by default: backprop on `default.qubit` up to
  12 qubits, adjoint on `lightning.qubit` above (see `hqnn_forge.encoding.resolve_backend`).
* **Initialisation** — use `restricted_normal_init_` on the returned layer; see
  `hqnn_forge.initializers` for what the small-angle init does and does not guarantee.

References
----------
* Havlíček et al. (2019) "Supervised learning with quantum-enhanced feature
  spaces", Nature 567, 209–212.  Introduces the IQP-type feature map
  (Hadamards, diagonal phases in the features and their pairwise products).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import combinations

import pennylane as qml
import torch
import torch.nn as nn

from hqnn_forge.encoding._common import (
    DeviceName,
    DiffMethod,
    Entangler,
    Readout,
    apply_variational_layers,
    backend_repr,
    check_inputs,
    expand_batch_dimension,
    measure_z,
    readout_wires,
    resolve_backend,
    resolve_device,
    shots_repr,
    validate_circuit_options,
    validate_device_shots,
    validate_seed,
    validate_shots,
    variational_weight_shape,
)
from hqnn_forge.noise import Channel, NoiseMethod, Position, TrainingNoiseMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases and device factory: shared with angle_embedding
# ---------------------------------------------------------------------------


def _make_iqp_embedding_circuit(
    n_qubits: int,
    n_layers: int,
    n_repeats: int = 1,
    entangler: Entangler = "ring",
    readout: Readout = "all",
) -> Callable[[torch.Tensor, torch.Tensor], list[qml.measurements.ExpectationMP]]:
    """
    Factory returning the bare quantum function for the IQP embedding.

    The returned function has the signature::

        circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[ExpectationMP]

    where ``inputs`` has shape ``(n_qubits,)`` (or ``(batch, n_qubits)`` when
    broadcasted) and ``weights`` has shape ``(n_layers, n_qubits, 3)``: the
    ``qml.Rot`` angles per layer and qubit (``(n_layers, n_qubits)``, the
    ``RY`` angles, for ``entangler="hardware_efficient"``; see
    :func:`~hqnn_forge.encoding.angle_embedding.variational_weight_shape`).

    Called inside a QNode it records one ``qml.expval(PauliZ)`` measurement per
    readout wire; the QNode turns them into the expectation values.

    ``entangler`` and ``readout`` are as in
    :func:`hqnn_forge.encoding.angle_embedding.apply_variational_layers` and
    :func:`~hqnn_forge.encoding.angle_embedding.measure_z`.
    """
    validate_circuit_options(n_qubits, entangler, readout)
    # All-to-all entangling pattern, the same as qml.IQPEmbedding(pattern=None)
    pairs = list(combinations(range(n_qubits), 2))

    def circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[qml.measurements.ExpectationMP]:
        # ── 1. IQP embedding: H → RZ(x_i) → exp(-i x_i x_j Z_i Z_j / 2) ─────
        # This is qml.IQPEmbedding's decomposition written out gate by gate,
        # with the two-qubit MultiRZ replaced by its exact CNOT·RZ·CNOT form.
        # Noiselessly the two are the same unitary, but the gates written here
        # are the circuit's physical content, and two things read them
        # (#136, #230):
        #
        # * Noise.  qml.noise.insert (hqnn_forge.noise, training-time noise,
        #   noisy kernels) puts a channel after every gate on every wire it
        #   touches: 5 per ZZ term here (2 + 1 + 2), 2 for a template MultiRZ.
        #   At n_qubits=3, p=0.05 the outputs differ by up to 0.06.
        # * Resources.  circuit_summary counts 2·C(n, 2)·n_repeats CNOTs here,
        #   half that as MultiRZ.
        #
        # On hardware MultiRZ compiles to CNOT·RZ·CNOT, so this form gives the
        # realistic noise model and gate count; swapping in qml.IQPEmbedding
        # would silently change both (tests/test_iqp_embedding.py pins them).
        # It also broadcasts a batched ``inputs`` of shape (batch, n_qubits)
        # through single-parameter gates only, a safeguard for lightning's
        # adjoint path, which mis-shapes a broadcasted MultiRZ, should the
        # circuit ever run broadcasted without expand_batch_dimension.
        # ``inputs[..., i]`` selects feature i for one sample or a batch alike.
        for _ in range(n_repeats):
            for qubit in range(n_qubits):
                qml.Hadamard(wires=qubit)
                qml.RZ(inputs[..., qubit], wires=qubit)
            for i, j in pairs:
                qml.CNOT(wires=[i, j])
                qml.RZ(inputs[..., i] * inputs[..., j], wires=j)
                qml.CNOT(wires=[i, j])

        # ── 2 & 3. Variational layers, then ⟨Z⟩ on the readout wires ─────
        apply_variational_layers(weights, n_qubits, n_layers, entangler)
        return measure_z(n_qubits, readout)

    return circuit


def build_iqp_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    n_repeats: int = 1,
    device_name: DeviceName = "auto",
    diff_method: DiffMethod = "auto",
    entangler: Entangler = "ring",
    readout: Readout = "all",
    shots: int | None = None,
    seed: int | None = None,
) -> qml.QNode:
    """Build and return a PennyLane QNode for the IQP feature map."""
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2; got {n_qubits}.")

    device_name, diff_method = resolve_backend(device_name, diff_method, n_qubits, shots=shots)
    validate_shots(shots, diff_method)
    device = resolve_device(device_name, n_qubits, seed=seed)
    validate_device_shots(device, shots)
    circuit_fn = _make_iqp_embedding_circuit(n_qubits, n_layers, n_repeats, entangler, readout)

    qnode = qml.QNode(
        func=circuit_fn,
        device=device,
        diff_method=diff_method,
        interface="torch",
        shots=shots,
    )
    # Batched inputs: see hqnn_forge.encoding._common.expand_batch_dimension
    return expand_batch_dimension(qnode, diff_method)


class IQPEncodingLayer(TrainingNoiseMixin, nn.Module):
    """
    A PyTorch nn.Module wrapping the IQP-embedding QNode.

    ``entangler`` and ``readout`` are the options of
    :class:`~hqnn_forge.encoding.QuantumEncodingLayer`; the output width is
    ``n_outputs`` (``n_qubits``, or 1 with ``readout="first"``), and the
    input width ``n_features`` is ``n_qubits``, one feature per qubit.
    ``noise_level`` / ``noise_position`` / ``noise_method`` /
    ``noise_trajectories`` / ``noise_channel`` add training-time noise
    (depolarizing by default; ``noise_level``'s range depends on the
    channel), and
    ``shots`` and ``seed`` finite-shot sampling and its device seed, exactly as in
    :class:`~hqnn_forge.encoding.QuantumEncodingLayer`.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_repeats: int = 1,
        device_name: DeviceName = "auto",
        diff_method: DiffMethod = "auto",
        entangler: Entangler = "ring",
        readout: Readout = "all",
        noise_level: float = 0.0,
        noise_position: Position = "all",
        noise_method: NoiseMethod = "density",
        noise_trajectories: int = 1,
        shots: int | None = None,
        noise_channel: Channel = "depolarizing",
        seed: int | None = None,
        readout_error: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_features = n_qubits
        self.n_layers = n_layers
        self.n_repeats = n_repeats
        self.entangler = entangler
        self.readout = readout
        self.n_outputs = len(readout_wires(n_qubits, readout))

        qnode = build_iqp_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_repeats=n_repeats,
            device_name=device_name,
            diff_method=diff_method,
            entangler=entangler,
            readout=readout,
            shots=shots,
            seed=seed,
        )

        weight_shapes: dict[str, tuple[int, ...]] = {
            "weights": variational_weight_shape(entangler, n_qubits, n_layers),
        }

        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        # Training-time noise; see QuantumEncodingLayer.
        self._init_training_noise(
            qnode,
            n_qubits,
            noise_level,
            noise_position,
            noise_method,
            noise_trajectories,
            shots=shots,
            noise_channel=noise_channel,
            readout_error=readout_error,
        )
        self.seed = validate_seed(seed)

    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check that ``x`` has ``n_qubits`` finite features; the values are used as given.

        This is the classical step ``forward`` applies before the QNode.  Every
        encoding layer has one, so tools which replay the circuit
        (:mod:`hqnn_forge.kernels`) validate and transform inputs exactly as
        ``forward`` does.
        """
        check_inputs(x, self.n_qubits)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of feature vectors."""
        # Whole batch in one call; see QuantumEncodingLayer.forward.
        x = self.prepare_inputs(x)
        return self._run_circuit(x)

    def extra_repr(self) -> str:
        options = ""
        if self.entangler != "ring":
            options += f", entangler={self.entangler!r}"
        if self.readout != "all":
            options += f", readout={self.readout!r}"
        options += self._noise_repr()
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_repeats={self.n_repeats}, "
            f"n_params={sum(p.numel() for p in self.parameters())}{options}"
            f"{shots_repr(self.shots, self.seed)}{backend_repr(self.qlayer)}"
        )
