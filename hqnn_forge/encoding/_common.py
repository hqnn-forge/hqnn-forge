"""
hqnn_forge.encoding._common
===========================
QNode plumbing every encoding layer shares.

The type aliases of the circuit options, the variational block and its weight
shape, input checks and readout, the device factory with its fallback chain,
and the batch expansion that makes a QNode accept ``(batch, n_features)``
inputs under every differentiation method.  They lived in
``angle_embedding.py`` until #306, which is why that module still re-exports
them; import them from here in new code.
"""

from __future__ import annotations

import inspect
import logging
import os
import warnings
from typing import Literal, assert_never, get_args

import pennylane as qml
import torch
from pennylane.exceptions import AllocationError, DeviceError

from hqnn_forge.circuits import hardware_efficient_layer, strongly_entangling_layer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RotationAxis = Literal["X", "Y", "Z"]
DiffMethod = Literal["adjoint", "parameter-shift", "backprop", "finite-diff"]
#: The simulators the fallback chain knows.  Any other PennyLane device name
#: (a plugin such as ``"qiskit.aer"``, or hardware) is accepted too, and
#: constructed exactly as given: see :func:`resolve_device`.
KnownDevice = Literal["lightning.gpu", "lightning.kokkos", "lightning.qubit", "default.qubit"]
#: The simulators :func:`resolve_device` puts on the fallback chain when their
#: backend is unavailable (``default.qubit``, the chain's end, has nowhere to
#: fall back to).  Every other name is constructed as given.
KNOWN_DEVICES: tuple[str, ...] = get_args(KnownDevice)
#: A PennyLane device name: one of :data:`KNOWN_DEVICES`, or any other.
DeviceName = str
Entangler = Literal["ring", "strongly_entangling", "brickwork", "hardware_efficient"]
Readout = Literal["all", "first"]

#: Devices tried, in order, after the requested one fails.  Each is a strict
#: subset of the previous one's requirements: ``lightning.qubit`` needs only
#: the ``pennylane-lightning`` wheel, ``default.qubit`` ships with PennyLane.
FALLBACK_CHAIN: tuple[str, ...] = ("lightning.qubit", "default.qubit")

#: What creating a device raises when its plugin or hardware is missing:
#: ``DeviceError`` for a device name no installed plugin registers,
#: ``ImportError`` / ``OSError`` when a plugin's compiled extension or a CUDA
#: library cannot be loaded, ``RuntimeError`` when the plugin loads but finds
#: no usable GPU.  A ``RuntimeError`` about memory is not one of these: see
#: :func:`is_out_of_memory`.
DEVICE_FAILURES: tuple[type[BaseException], ...] = (
    DeviceError,
    ImportError,
    OSError,
    RuntimeError,
)


# ---------------------------------------------------------------------------
# Variational block and readout, shared by every encoding circuit
# ---------------------------------------------------------------------------


def apply_variational_layers(
    weights: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    entangler: Entangler = "ring",
    layer_offset: int = 0,
) -> None:
    """
    Apply the ``n_layers`` variational blocks to the current circuit.

    ``entangler`` selects the block:

    * ``"ring"`` (the library's default): CNOT ring ``CNOT(i → i+1 mod n)``,
      then ``Rot(φ, θ, ω)`` on every qubit.
    * ``"strongly_entangling"``: ``qml.StronglyEntanglingLayers``, i.e.
      ``Rot`` on every qubit **then** a CNOT ring whose range grows with the
      layer index, ``r = ℓ mod (n-1) + 1``.  This is the block the published
      SHNN uses (Schuld et al. 2020, PennyLane template).
    * ``"brickwork"``: nearest-neighbour CNOTs on the even pairs
      ``(0,1), (2,3), …``, then on the odd pairs ``(1,2), (3,4), …``, with
      no wrap-around, then ``Rot`` on every qubit.  Unlike the two cascades
      above, which carry a readout across the whole register at shallow
      depth (⟨Z_0⟩ ↦ Z_1⋯Z_{n-1} through one ring), each layer widens the
      backward light cone of a single-qubit readout by at most two qubits
      on each side, so the ⟨Z_i⟩ readouts stay local costs in the sense of
      Cerezo et al. (2021) while ``n_layers`` is small against ``n_qubits``;
      see :mod:`hqnn_forge.initializers.restricted_variance` for the
      measured gradient variance.
    * ``"hardware_efficient"``: a nearest-neighbour ``CZ(i, i+1)`` ladder,
      then ``RY(θ)`` on every qubit (Kandala et al. 2017):
      :func:`hqnn_forge.circuits.hardware_efficient_layer`.  One angle per
      qubit per layer and ``n − 1`` two-qubit gates per layer: a third of the
      parameters of the ``Rot`` blocks, with CZ native on many devices.

    ``"ring"``, ``"strongly_entangling"`` and ``"brickwork"`` take ``weights``
    of shape ``(n_layers, n_qubits, 3)`` and use ``n_layers · n_qubits``
    ``Rot`` gates.  The ring and ``"strongly_entangling"`` use ``n_qubits``
    CNOTs per layer and differ in gate order and, from the second layer on, in
    which qubits the CNOTs connect; ``"brickwork"`` uses ``n_qubits - 1``.
    ``"hardware_efficient"`` takes ``(n_layers, n_qubits)``.
    :func:`variational_weight_shape` gives the shape for each.  The ``"ring"``
    block is :func:`hqnn_forge.circuits.strongly_entangling_layer` applied per
    layer.

    ``layer_offset`` is the index of the first block within the whole ansatz,
    for circuits that interleave other gates between blocks and so apply them
    a few at a time: the ``"strongly_entangling"`` range of block ``ℓ`` is
    ``(layer_offset + ℓ) mod (n-1) + 1``, so applying the blocks one by one
    with offsets ``0 … L-1`` gives the same ranges as applying all ``L`` at
    once.  The ``"ring"``, ``"brickwork"`` and ``"hardware_efficient"``
    blocks do not depend on the layer index.
    """
    if entangler == "strongly_entangling":
        # A single wire has no CNOT partner: leave the ranges to the template,
        # which uses 0 there instead of dividing by n - 1 = 0.
        ranges = (
            [(layer_offset + layer) % (n_qubits - 1) + 1 for layer in range(n_layers)]
            if n_qubits > 1
            else None
        )
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits), ranges=ranges)
        return
    _check_entangler(entangler)
    for layer in range(n_layers):
        if entangler == "ring":
            # CNOT ring (last qubit → first), then Rot(φ, θ, ω) on every qubit
            strongly_entangling_layer(weights[layer], n_qubits)
        elif entangler == "hardware_efficient":
            # CZ ladder, then RY(θ) on every qubit
            hardware_efficient_layer(weights[layer], n_qubits)
        elif entangler == "brickwork":
            # Brickwork: even nearest-neighbour pairs, then odd ones
            for start in (0, 1):
                for qubit in range(start, n_qubits - 1, 2):
                    qml.CNOT(wires=[qubit, qubit + 1])
            # Per-qubit SU(2) rotation block
            for qubit in range(n_qubits):
                qml.Rot(
                    weights[layer, qubit, 0],  # φ
                    weights[layer, qubit, 1],  # θ
                    weights[layer, qubit, 2],  # ω
                    wires=qubit,
                )
        else:
            assert_never(entangler)


def variational_weight_shape(
    entangler: Entangler, n_qubits: int, n_layers: int
) -> tuple[int, ...]:
    """
    Shape of the ``weights`` tensor :func:`apply_variational_layers` reads for ``entangler``.

    The one place an encoder's variational weight shape is defined: every
    encoding layer registers its ``weights`` with this shape.  Dim 0 is always
    the layer index, which :func:`~hqnn_forge.initializers.block_local_init_`
    and the diagnostics' ``n_layers`` fallback rely on: ``(n_layers, n_qubits,
    3)`` for the ``Rot`` blocks (``"ring"``, ``"strongly_entangling"``,
    ``"brickwork"``), ``(n_layers, n_qubits)`` for the ``RY`` of
    ``"hardware_efficient"``.

    Raises
    ------
    ValueError
        For an unknown ``entangler``.
    """
    _check_entangler(entangler)
    if entangler == "hardware_efficient":
        return (n_layers, n_qubits)
    return (n_layers, n_qubits, 3)


def validate_circuit_options(
    n_qubits: int,
    entangler: Entangler,
    readout: Readout,
    rotation: RotationAxis | None = None,
) -> None:
    """
    Raise ``ValueError`` for an ``entangler``, ``readout`` or (if given)
    ``rotation`` outside the allowed values.

    The encoding builders call this eagerly: ``qml.AngleEmbedding`` only
    rejects the axis when the circuit first runs, which is a forward pass away
    from the constructor that was given it -- and past get_config and a
    checkpoint.
    """
    readout_wires(n_qubits, readout)
    _check_entangler(entangler)
    if rotation is not None and rotation not in ("X", "Y", "Z"):
        raise ValueError(f"rotation must be 'X', 'Y' or 'Z'; got {rotation!r}.")


def _check_entangler(entangler: str) -> None:
    if entangler not in get_args(Entangler):
        raise ValueError(
            f"entangler must be one of {', '.join(map(repr, get_args(Entangler)))}; "
            f"got {entangler!r}."
        )


def check_inputs(x: torch.Tensor, expected: int, name: str = "n_qubits", hint: str = "") -> None:
    """
    Raise ``ValueError`` unless ``x`` has ``expected`` features and is finite.

    The shared check of every encoding layer's ``prepare_inputs``.  A NaN or
    ±inf angle is simulated without error and gives NaN outputs, so it is
    refused here, where ``forward`` and :mod:`hqnn_forge.kernels` both see it.
    ``hint`` is appended to the width message.
    """
    if x.shape[-1] != expected:
        raise ValueError(
            f"Input feature dimension {x.shape[-1]} does not match {name}={expected}.{hint}"
        )
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Encoding layer inputs contain NaN or ±inf.")


def readout_wires(n_qubits: int, readout: Readout = "all") -> list[int]:
    """Wires measured in ⟨Z⟩: every qubit (``"all"``) or qubit 0 only (``"first"``)."""
    if readout == "all":
        return list(range(n_qubits))
    if readout == "first":
        return [0]
    raise ValueError(f"readout must be 'all' or 'first'; got {readout!r}.")


def measure_z(n_qubits: int, readout: Readout = "all") -> list[qml.measurements.ExpectationMP]:
    """``[⟨Z_i⟩ for i in readout_wires(...)]``: the circuit's return value."""
    return [qml.expval(qml.PauliZ(i)) for i in readout_wires(n_qubits, readout)]


# ---------------------------------------------------------------------------
# Device factory — graceful fallback down to default.qubit
# ---------------------------------------------------------------------------


def is_out_of_memory(exc: BaseException) -> bool:
    """
    Whether a device-creation failure is the state vector not fitting.

    Every backend in the chain allocates the same ``2**n_qubits`` amplitudes,
    so falling back cannot help and would only move the allocation from GPU
    memory to host memory, where it can get the process killed instead of
    raising.  The lightning plugins raise a bare ``RuntimeError`` naming
    memory.  :class:`~pennylane.exceptions.AllocationError` is accepted too,
    as a precaution: PennyLane raises it for dynamically allocated wires, not
    for device creation.
    """
    return isinstance(exc, AllocationError) or (
        isinstance(exc, RuntimeError) and "memory" in str(exc).lower()
    )


#: Backends that failed to initialise in this process, with the failure.  A
#: failed plugin import is not cached by Python, and a CUDA library load or a
#: GPU probe is slow, so every layer built with the same device_name would
#: otherwise repeat them -- and warn again.  Out-of-memory failures are never
#: recorded: they depend on n_qubits and are raised, not fallen back from.
_FAILED_BACKENDS: dict[str, BaseException] = {}

#: The hqnn_forge package directory, for attributing warnings to user code.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reset_device_fallback() -> None:
    """
    Forget the backends that failed to initialise, so the next layer tries
    them again -- e.g. after installing a plugin in a running session.
    """
    _FAILED_BACKENDS.clear()


def _stacklevel_outside_package() -> int:
    """
    ``stacklevel`` that attributes a warning issued by the caller of this
    function to the first frame outside ``hqnn_forge``: the user's own call,
    however deep the layer or classifier constructors that led here.
    (``warnings.warn(skip_file_prefixes=...)`` does this from Python 3.12 on;
    the floor is 3.11.)
    """
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None  # the function that warns
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        _PACKAGE_DIR + os.sep
    ):
        frame = frame.f_back
        level += 1
    return level


SHOT_FREE_METHODS = ("adjoint", "backprop", "finite-diff")


def validate_shots(shots: int | None, diff_method: str) -> None:
    """
    Raise ``ValueError`` unless ``shots`` is ``None`` or a positive ``int``
    usable with ``diff_method``.

    ``shots=None`` gives exact expectation values.  A finite shot count samples
    them, as hardware does, and rules out ``adjoint`` and ``backprop``: both
    differentiate the simulator's state vector, which sampling does not give
    (PennyLane refuses even the forward pass).  It rules out ``finite-diff``
    too: a difference quotient with a step ``h`` near 1e-7 divides the shot
    noise of each expectation value by ``h``, so its gradients are noise of
    order ``1 / (h·sqrt(shots))`` -- about 1e6 at 1000 shots against an exact
    value of order 1 -- and training silently diverges.  ``parameter-shift``
    shifts by π/2 and stays unbiased; it is also the method that runs on
    hardware.
    """
    if shots is None:
        return
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError(f"shots must be None or a positive int; got {shots!r}.")
    if diff_method in SHOT_FREE_METHODS:
        raise ValueError(
            f"shots={shots} samples the expectation values, and diff_method="
            f"{diff_method!r} needs exact ones ("
            + (
                "its tiny step divides the shot noise into the gradient"
                if diff_method == "finite-diff"
                else "it differentiates the exact state vector"
            )
            + "); use diff_method='parameter-shift', the method that also runs on hardware."
        )


def _raise_device_samples(device_name: str, *, finite_shots: int | None = None) -> None:
    reason = (
        f"has finite shots ({finite_shots})"
        if finite_shots is not None
        else "does not support analytic execution"
    )
    raise ValueError(
        f"Device {device_name!r} {reason}, but "
        "shots=None was requested. This device samples, so pass shots= and "
        "diff_method='parameter-shift'."
    )


def validate_device_shots(device: qml.devices.Device, shots: int | None) -> None:
    """
    Raise ``ValueError`` if *device* only samples or has finite shots but *shots* is ``None``.

    A real sampling device (hardware, ``"qiskit.remote"``, sampling plugins, ...)
    cannot return exact expectation values and fails at the first forward pass when
    given ``shots=None``, typically with the default ``diff_method="adjoint"``.
    Raising at construction with an informative message guides users to pass
    explicit ``shots`` and ``diff_method='parameter-shift'``.
    """
    if shots is not None:
        return

    # 1. Device-level shots (while supported in PennyLane)
    dev_shots = getattr(device, "shots", None)
    if dev_shots is not None:
        total_shots = getattr(dev_shots, "total_shots", dev_shots)
        if total_shots is not None:
            _raise_device_samples(device.name, finite_shots=total_shots)

    # 2. Probe device preprocessing pipeline
    cached = getattr(device, "_hqnn_supports_analytic", None)
    if cached is not None:
        if not cached:
            _raise_device_samples(device.name)
        return

    preprocess_fn = getattr(device, "preprocess", None)
    if not callable(preprocess_fn):
        return

    try:
        pipeline, _ = device.preprocess()
    except Exception:  # noqa: BLE001 - unexpected device errors must not block layer construction
        return

    tape_analytic = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=None)
    try:
        pipeline([tape_analytic])
        device._hqnn_supports_analytic = True
        return
    except DeviceError:
        pass
    except Exception:  # noqa: BLE001 - non-DeviceError means the probe failed for an unrelated reason
        return

    tape_finite = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=100)
    try:
        pipeline([tape_finite])
    except Exception:  # noqa: BLE001 - finite shots probe failed; device rejects the probe for other reasons
        return

    # Analytic fails with DeviceError and finite shots pass: device is sampling-only
    device._hqnn_supports_analytic = False
    _raise_device_samples(device.name)


def shots_repr(shots: int | None) -> str:
    """The ``extra_repr`` fragment for a finite shot count; empty for exact values."""
    return "" if shots is None else f", shots={shots}"


def resolve_device(device_name: DeviceName, n_qubits: int) -> qml.devices.Device:
    """
    Create *device_name*, falling back along :data:`FALLBACK_CHAIN` when a
    backend is not installed or has no usable hardware.

    A backend that fails is remembered for the rest of the process: later
    layers skip it without trying again, and its ``RuntimeWarning`` is issued
    once, not once per layer (:func:`reset_device_fallback` forgets them).
    The warning is attributed to the first frame outside ``hqnn_forge`` --
    the user's call -- whichever layer or classifier constructor led here.

    The chain is ``requested → lightning.qubit → default.qubit``; entries at
    or before the requested device are skipped, so ``lightning.qubit`` falls
    straight to ``default.qubit`` and ``default.qubit`` has no fallback.

    Only the four simulators in :data:`KNOWN_DEVICES` enter the chain.  Any
    other name -- a PennyLane plugin device or hardware -- is constructed exactly as
    given, and PennyLane's error surfaces if it cannot be: a typo such as
    ``"default.qbit"`` raises rather than quietly running on another
    simulator.

    Parameters
    ----------
    device_name:
        Preferred PennyLane device string.  ``"lightning.gpu"`` (cuQuantum,
        NVIDIA) and ``"lightning.kokkos"`` (Kokkos: OpenMP on the PyPI wheel,
        CUDA/HIP when built from source) are the accelerated backends; see
        the README for their prerequisites.
    n_qubits:
        Number of qubits to allocate.

    Returns
    -------
    qml.devices.Device
        An initialised PennyLane device ready for QNode attachment.

    Raises
    ------
    DeviceError
        If ``device_name`` is outside :data:`KNOWN_DEVICES` and no installed
        plugin registers it, e.g. a typo.  A plugin that registers the name
        but cannot construct the device raises its own exception.
    RuntimeError
        If the state vector does not fit in memory: the lightning plugins
        raise one naming memory (see :func:`is_out_of_memory`), which is
        raised rather than fallen back from.
    RuntimeWarning
        If a warnings-as-errors filter is active: the warning a fallback
        issues is raised instead of falling back.
    Exception
        Any other error constructing a device in the chain, which is raised
        at that step, and the last backend's own error if every step fails;
        ``default.qubit`` has no dependencies, so the latter means PennyLane
        itself is broken.
    """
    if device_name not in KNOWN_DEVICES:
        dev = qml.device(device_name, wires=n_qubits)
        logger.debug("Quantum device initialised: %s (%d qubits)", device_name, n_qubits)
        return dev
    start = FALLBACK_CHAIN.index(device_name) + 1 if device_name in FALLBACK_CHAIN else 0
    candidates = [device_name, *FALLBACK_CHAIN[start:]]
    last = len(candidates) - 1
    for attempt, name in enumerate(candidates):
        if attempt < last and name in _FAILED_BACKENDS:
            # Already failed and warned about in this process: go straight on.
            logger.debug("Skipping %s, which failed before: %r", name, _FAILED_BACKENDS[name])
            continue
        try:
            dev = qml.device(name, wires=n_qubits)
        except DEVICE_FAILURES as exc:
            if attempt == last or is_out_of_memory(exc):
                raise
            fallback = candidates[attempt + 1]
            hint = (
                "  Install pennylane-lightning for adjoint differentiation support and "
                "significantly faster simulation."
                if fallback == "default.qubit"
                else ""
            )
            warnings.warn(
                f"Could not initialise '{name}' ({type(exc).__name__}: {exc}).  "
                f"Falling back to '{fallback}'.{hint}",
                RuntimeWarning,
                stacklevel=_stacklevel_outside_package(),
            )
            # Recorded only once warned: if a warnings-as-errors filter turns
            # the warning into an exception, the next build must try (and
            # raise) again rather than fall back silently.
            _FAILED_BACKENDS[name] = exc
            continue
        if attempt:
            logger.info("Quantum device fell back from %s to %s", device_name, name)
        logger.debug("Quantum device initialised: %s (%d qubits)", name, n_qubits)
        return dev
    raise AssertionError("unreachable: the fallback chain always ends in a raise or a return")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def expand_batch_dimension(qnode: qml.QNode, diff_method: str) -> qml.QNode:
    """
    Make *qnode* accept a batched ``inputs`` tensor of shape ``(batch, n_qubits)``
    under every supported differentiation method.

    A 2-D ``inputs`` reaches the circuit as a *broadcasted* tape: one tape whose
    embedding gates carry a batch of angles.  How that is executed depends on
    ``diff_method``:

    * ``"backprop"`` differentiates through the simulator, which handles the
      batch natively as one vectorised state-vector evolution.  This is the fast
      path and the tape is left broadcasted.
    * Every other method (``"adjoint"``, ``"parameter-shift"``,
      ``"finite-diff"``) is a gradient *transform* on the tape, and the
      parameter-shift and finite-difference transforms refuse a broadcasted
      tape when the gradient with respect to the broadcasted parameters is
      requested -- which is exactly the case when a classical encoder upstream
      needs input gradients.  For these the tape is split into one tape per
      sample *before* the gradient transform sees it, so each tape is
      unbroadcasted and the whole batch is still handed to the device as a
      single list of tapes.  ``lightning.qubit``'s adjoint path once
      mis-shaped results for broadcast two-qubit rotations; with PennyLane
      0.45 it returns them correctly, but it has no vectorised path to gain:
      its own preprocessing applies ``broadcast_expand`` too, so a broadcast
      tape is split into one tape per sample on the device either way, and
      measured no faster than splitting here -- 1.1-1.3x the split's
      training step at batch 1024 (#312, ``examples/benchmark_batching.py``).
      So the split stays for every method.

    Either way the QNode's signature and results are unchanged: it returns
    ``n_qubits`` expectation values, each of shape ``(batch,)``.
    """
    if diff_method == "backprop":
        return qnode
    return qml.transforms.broadcast_expand(qnode)
