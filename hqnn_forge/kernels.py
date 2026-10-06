"""
hqnn_forge.kernels
==================
Quantum kernel estimation from the library's encoding layers.

A quantum kernel is the fidelity between the states two inputs are encoded
into::

    k(x, x') = |⟨Φ(x) | Φ(x')⟩|²

Feeding the Gram matrix ``K[i, j] = k(x_i, x_j)`` to a classical SVM
(``sklearn.svm.SVC(kernel="precomputed")``) gives the quantum-kernel (QSVM)
approach to classification: the quantum device only evaluates the feature
map, and the optimisation is the SVM's convex problem, with a unique optimum
and no barren plateaus.  This is the complementary method to the trainable
VQCs in :mod:`hqnn_forge.models`.

Which circuit defines the kernel
--------------------------------
:func:`quantum_kernel_matrix` replays the circuit of an encoding layer
(``QuantumEncodingLayer``, ``IQPEncodingLayer``, ``AmplitudeEncodingLayer``,
``DataReuploadingLayer``) up to but not including its measurements, and
reads the state vector.  The layer's variational block is included as it
stands, with the layer's current weights.  For the single-upload encoders
this makes no difference: the ansatz is a data-independent unitary ``V`` and
``|⟨Φ(x)|V†V|Φ(x')⟩|² = |⟨Φ(x)|Φ(x')⟩|²``, so the kernel is that of the
embedding alone whatever the weights are.

For :class:`~hqnn_forge.encoding.DataReuploadingLayer` the blocks ``weights[0]`` to
``weights[-2]`` sit between uploads and do shape the kernel; they are then
part of the kernel's definition (a "trainable kernel" in the sense of
Hubregtsen et al. 2022), and the matrix is that of the layer as currently
parametrised.  :func:`train_kernel_alignment` chooses them for the task, by
maximising :func:`kernel_target_alignment` through the differentiable path
(``differentiable=True``).  The last block, ``weights[-1]``, comes after the last upload
and cancels as the single-upload ansatz does, so with ``n_layers=1`` the
kernel does not depend on ``weights`` at all.  A trainable
``input_scaling`` multiplies the features inside every scaled upload and
always shapes the kernel.

Only the circuit as written is replayed.  A transform on the layer's QNode
(noise, compilation) would be dropped by the replay, so any transform other
than the ``broadcast_expand`` the encoders add for non-backprop
differentiation makes these functions raise rather than return the kernel
of a different circuit.  For noise, pass ``noise_level`` instead (below).

Under depolarising noise
------------------------
With ``noise_level = p > 0`` the circuit is replayed on ``default.mixed``
with ``DepolarizingChannel(p)`` inserted exactly as
:mod:`hqnn_forge.noise` inserts it for the models (the same construction,
at ``noise_position``), and each input is encoded into a density matrix
``ρ(x)``.  The fidelity ``|⟨Φ(x)|Φ(y)⟩|²`` then generalises to the
Hilbert–Schmidt kernel ``k(x, y) = Tr[ρ(x) ρ(y)]``: still symmetric and
positive semi-definite (a Gram matrix of the vectorised ``ρ``), and equal to
the fidelity kernel for pure states.  Its diagonal is the purity
``Tr[ρ(x)²] < 1``, which falls as ``p`` grows; that is the measurable effect
of the noise, and the matrix is deliberately not renormalised to hide it.

This is not what :func:`overlap_kernel_matrix` estimates on a noisy device:
there the noise acts on the whole compute-uncompute circuit ``U(x)† U(y)``,
the estimate is ``⟨0|N(U(x)†U(y)|0⟩⟨0|U(y)†U(x))|0⟩`` for the device's noise
``N``, which is neither ``Tr[ρ(x)ρ(y)]`` nor necessarily symmetric.
Mixed-state simulation needs ``4^n`` entries per sample instead of ``2^n``,
and ``default.mixed`` stops at 23 wires.

Scaling: O(M²) against the VQC
------------------------------
A kernel matrix over ``M`` training points has ``M(M+1)/2`` distinct entries,
of which the ``M`` diagonal ones are 1 by construction.  On hardware the
other ``M(M-1)/2``, that is ``O(M²)``, are circuit evaluations, each an
overlap estimate with shot noise, before the SVM even starts, and every
prediction costs ``M`` more overlaps against the training set.

A VQC with ``P`` trainable circuit parameters estimates its gradient on
hardware by the parameter-shift rule, about ``2P + 1`` circuit evaluations
per sample, so ``E`` epochs cost about ``E · M · (2P + 1)`` evaluations, and
a prediction costs one.  Training the kernel is therefore the cheaper of the
two until ``M(M-1)/2`` overtakes ``E · M · (2P + 1)``, around
``M ≈ 2E(2P + 1)``: for ``QuantumEncodingLayer(n_qubits=8, n_layers=2)``
(``P = 48``) trained for 50 epochs, that is ``M ≈ 9,700``.  Past that point,
and at every prediction whatever ``M`` is, the kernel costs more.

On a state-vector simulator the picture is different: ``M`` state vectors of
size ``2^n`` and one ``M × M`` Gram product, which is what
:func:`quantum_kernel_matrix` does.  There the limit is memory for the
``M × M`` matrix (see its Notes), not circuit evaluations.

:func:`overlap_kernel_matrix` estimates the kernel the way hardware has to,
one compute-uncompute circuit ``U(x)† U(y)`` per entry, reading the
probability of the all-zeros outcome, with ``shots`` samples per circuit and
on any device.  That makes the ``M(M-1)/2`` cost and the effect of shot noise
(and, on a noisy device, of noise) on the SVM measurable rather than
described.

References
----------
* Havlíček et al. (2019) "Supervised learning with quantum-enhanced feature
  spaces", Nature 567, 209.
* Schuld & Killoran (2019) "Quantum machine learning in feature Hilbert
  spaces", PRL 122, 040504.
* Hubregtsen et al. (2022) "Training quantum embedding kernels on near-term
  quantum computers", PRA 106, 042431.
* Cortes, Mohri & Rostamizadeh (2012) "Algorithms for learning kernels based
  on centered alignment", JMLR 13, 795–828.
* Higham (1988) "Computing a nearest symmetric positive semidefinite matrix",
  Linear Algebra and its Applications 103, 103–118.
"""

from __future__ import annotations

import numbers

import pennylane as qml
import torch
from torch import nn

from hqnn_forge._encoding_contract import EncodingLayer
from hqnn_forge._resolve import require_prepare_inputs, resolve_encoding_layer
from hqnn_forge.noise import Position, _noisy_qnode, validate_noise

__all__ = [
    "encoded_density_matrices",
    "encoded_states",
    "kernel_from_density_matrices",
    "kernel_from_states",
    "kernel_target_alignment",
    "nearest_psd",
    "overlap_kernel_matrix",
    "quantum_kernel_matrix",
    "train_kernel_alignment",
]


def _resolve_layer(layer: nn.Module, caller: str) -> EncodingLayer:
    """
    ``layer`` as an :class:`~hqnn_forge.encoding.EncodingLayer`, or raise.

    A hybrid classifier is refused, not unwrapped: the kernel is defined by the
    encoder alone, and the classifier's classical encoder would sit between
    ``X`` and the feature map (see ``resolve_encoding_layer``).
    """
    found, qlayer, _ = resolve_encoding_layer(layer, caller, allow_model=False)
    encoder = require_prepare_inputs(found, caller)
    # The level=0 tape drops every transform on the QNode and the replay runs
    # on default.qubit, so a transformed circuit (apply_depolarizing_noise's
    # qml.noise.insert, qml.add_noise, a compile pass) would silently give the
    # kernel of the untransformed one.  broadcast_expand only splits a batch
    # into per-sample tapes and leaves each circuit as it is.
    unknown = [
        t
        for t in qlayer.qnode.compile_pipeline
        if t.tape_transform is not qml.transforms.broadcast_expand.tape_transform
    ]
    if unknown:
        raise RuntimeError(
            f"quantum_kernel_matrix computes the untransformed state-vector kernel, but "
            f"the layer's QNode carries {unknown}, which the replay would drop.  Inside "
            f"apply_depolarizing_noise, call it outside the block, and pass noise_level= "
            f"for the kernel under the same depolarising noise."
        )
    return encoder


def _prepare(X: torch.Tensor, encoder: EncodingLayer, name: str) -> torch.Tensor:
    """Check ``X`` is a non-empty 2-D tensor and apply the layer's ``prepare_inputs``."""
    if not isinstance(X, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(X).__name__}.")
    if X.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features); got {tuple(X.shape)}.")
    if X.shape[0] == 0:
        raise ValueError(f"{name} has no samples.")
    # The same validation and transform forward applies (width and finiteness
    # checks, the amplitude encoder's padding and normalisation).
    return encoder.prepare_inputs(X.detach().to(torch.float64))


def _check_batch_size(batch_size: int | None) -> None:
    if batch_size is not None and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, numbers.Integral)
        or batch_size < 1
    ):
        raise ValueError(f"batch_size must be a positive integer or None; got {batch_size!r}.")


def _simulate(
    prepared: torch.Tensor,
    qlayer: qml.qnn.TorchLayer,
    n_qubits: int,
    differentiable: bool = False,
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    State vectors for inputs that have already been through ``_prepare``.

    With ``differentiable`` the layer's weights are not detached and the
    replay runs under backprop, so the states carry gradients to them.  With
    ``batch_size`` the rows are replayed that many at a time and written into
    one preallocated output, so the simulator's working memory scales with
    ``batch_size`` rather than with the number of rows.  (Joining a list of
    slices with ``torch.cat`` would hold every slice and the joined copy at
    once, twice the states.)
    """
    n_rows = prepared.shape[0]
    if batch_size is not None and batch_size < n_rows:
        states = torch.empty(n_rows, 2**n_qubits, dtype=torch.complex128)
        for start in range(0, n_rows, batch_size):
            stop = start + batch_size
            states[start:stop] = _simulate(prepared[start:stop], qlayer, n_qubits, differentiable)
        return states
    # One tape for the whole batch from the layer's own QNode (level=0: the
    # circuit as written, before any batching or gradient transform), with the
    # measurements swapped for the state and run on a state-vector device,
    # which executes the broadcast tape as one vectorised pass.  The layer's
    # weights are used as they are, detached.
    weights = {
        name: (p if differentiable else p.detach()).to(torch.float64)
        for name, p in qlayer.qnode_weights.items()
    }
    tape = qml.workflow.construct_tape(qlayer.qnode, level=0)(prepared, **weights)
    tape = tape.copy(measurements=[qml.state()])
    device = qml.device("default.qubit", wires=n_qubits)
    (result,) = qml.execute([tape], device, diff_method="backprop" if differentiable else None)
    states = torch.as_tensor(result).to(torch.complex128)
    return states.reshape(prepared.shape[0], 2**n_qubits)


def encoded_states(
    X: torch.Tensor,
    layer: nn.Module,
    *,
    differentiable: bool = False,
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    State vectors ``|Φ(x_i)⟩`` the layer prepares for each row of ``X``.

    ``X`` first goes through the layer's ``prepare_inputs``, the validation and
    classical transform ``forward`` applies before its QNode (a width check,
    and for the amplitude encoder padding and normalisation).  The layer's
    circuit is then replayed on ``default.qubit`` for the whole batch at once
    (or ``batch_size`` rows at a time), with its measurements replaced by
    ``qml.state()``.

    Compute the states once and pass them to :func:`kernel_from_states` to
    reuse them, for example the training states at every prediction.

    Parameters
    ----------
    X:
        Inputs, shape ``(n_samples, n_features)``.
    layer:
        An encoding layer.
    differentiable:
        Keep the graph to the layer's trainable parameters (its weights and
        any ``input_scaling``), simulating under backprop, so a loss on the
        states or the kernel can train them.  Default: ``False``, detached.
    batch_size:
        Replay the circuit on this many rows at a time and join the states.
        ``default.qubit`` copies the broadcast state at every gate, so one
        pass over ``M`` rows needs a multiple of ``M · 2**n_qubits · 16``
        bytes while it runs; batches bound that working memory by
        ``batch_size`` rows.  The returned states still take
        ``M · 2**n_qubits · 16`` bytes.  All of ``X`` is validated before the
        first batch runs.  ``None`` (default): one pass.  The result is the
        same either way.  With ``differentiable=True`` the bound does not
        hold: autograd keeps every batch's per-gate intermediates until
        ``backward``, so the graph takes as much memory as a single pass.

    Returns
    -------
    torch.Tensor
        Complex tensor of shape ``(n_samples, 2**n_qubits)``, one normalised
        state per row, ``complex128``.

    Raises
    ------
    TypeError
        If ``layer`` is not an encoding layer or ``X`` is not a tensor.
    ValueError
        If ``X`` is not a non-empty 2-D tensor, contains NaN or ±inf, or
        ``prepare_inputs`` rejects it (wrong number of features, an all-zero
        amplitude vector), or if ``batch_size`` is not a positive integer.
    RuntimeError
        If the layer's QNode carries a transform other than
        ``broadcast_expand``, for example inside
        :func:`hqnn_forge.noise.apply_depolarizing_noise`: the replay would
        drop it, so it refuses rather than return the untransformed states.
    """
    _check_batch_size(batch_size)
    encoder = _resolve_layer(layer, "encoded_states")
    qlayer, n_qubits = encoder.qlayer, encoder.n_qubits
    return _simulate(_prepare(X, encoder, "X"), qlayer, n_qubits, differentiable, batch_size)


def kernel_from_states(
    states_x: torch.Tensor,
    states_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fidelity kernel ``K[i, j] = |⟨ψ_i|φ_j⟩|²`` from precomputed state vectors.

    Parameters
    ----------
    states_x:
        States of shape ``(n_x, dim)``, as returned by :func:`encoded_states`.
        Any real or complex dtype; both sets are promoted to ``complex128``,
        so states stored as ``complex64`` can be reused as they are.
    states_y:
        Optional second set of states, shape ``(n_y, dim)``.  ``None``
        (default) gives the square Gram matrix of ``states_x`` with itself,
        made exactly symmetric.

    Returns
    -------
    torch.Tensor
        ``float64`` tensor of shape ``(n_x, n_y)`` (or ``(n_x, n_x)``).

    Examples
    --------
    Simulate each set once and build the train and test kernels from the
    same training states:

    >>> import torch
    >>> from sklearn.svm import SVC
    >>> from hqnn_forge.encoding import QuantumEncodingLayer
    >>> from hqnn_forge.kernels import encoded_states, kernel_from_states
    >>> layer = QuantumEncodingLayer(n_qubits=4, n_layers=1, device_name="default.qubit")
    >>> g = torch.Generator().manual_seed(0)
    >>> X_train, X_test = torch.rand(10, 4, generator=g), torch.rand(3, 4, generator=g)
    >>> y_train = [0, 1] * 5
    >>> S_train = encoded_states(X_train, layer)
    >>> K_train = kernel_from_states(S_train)
    >>> bool(torch.allclose(K_train.diagonal(), torch.ones(10, dtype=torch.float64)))
    True
    >>> svm = SVC(kernel="precomputed").fit(K_train.numpy(), y_train)
    >>> K_test = kernel_from_states(encoded_states(X_test, layer), S_train)
    >>> K_test.shape
    torch.Size([3, 10])
    >>> y_pred = svm.predict(K_test.numpy())
    """
    symmetric = states_y is None
    if states_y is None:
        states_y = states_x
    if states_x.ndim != 2 or states_y.ndim != 2 or states_x.shape[1] != states_y.shape[1]:
        raise ValueError(
            f"states_x and states_y must be 2-D with the same state dimension; got "
            f"{tuple(states_x.shape)} and {tuple(states_y.shape)}."
        )
    states_x = states_x.to(torch.complex128)
    states_y = states_x if symmetric else states_y.to(torch.complex128)
    # The n_x × n_y intermediates dominate memory for a large training set:
    # the complex Gram matrix (16 bytes per entry) and the float64 kernel
    # (8 bytes).  The Gram matrix is freed as soon as its modulus is taken.
    # The squaring is in place; the symmetric path's mirroring allocates one
    # more n × n float64 matrix, which still fits under the 24 n² byte peak.
    gram = states_x @ states_y.conj().T
    kernel = gram.abs()
    del gram
    kernel.square_()
    if symmetric:
        # Exact symmetry, not just up to rounding: mirror the upper triangle
        # onto the lower one.  Nothing is clamped: the diagonal is left as
        # computed, so a state that is not normalised shows up as K[i, i] != 1.
        kernel.triu_()
        kernel += kernel.triu(1).T
    return kernel


def quantum_kernel_matrix(
    X: torch.Tensor,
    layer: nn.Module,
    Y: torch.Tensor | None = None,
    *,
    differentiable: bool = False,
    batch_size: int | None = None,
    noise_level: float = 0.0,
    noise_position: Position = "all",
) -> torch.Tensor:
    """
    Pairwise state-fidelity kernel ``K[i, j] = |⟨Φ(x_i)|Φ(y_j)⟩|²``.

    Parameters
    ----------
    X:
        Inputs, shape ``(n_samples_x, n_features)``.
    layer:
        The encoding layer whose circuit defines ``Φ``.  See the module
        docstring for the role of its variational weights.
    Y:
        Optional second set of inputs, shape ``(n_samples_y, n_features)``.
        ``None`` (default) computes the square Gram matrix of ``X`` with
        itself.  Pass the training inputs here to build the rectangular
        matrix an SVM needs at prediction time; to avoid simulating the
        training set again on every call, keep its :func:`encoded_states`
        and use :func:`kernel_from_states` instead.
    differentiable:
        As for :func:`encoded_states`: the matrix carries gradients to the
        layer's trainable parameters.
    batch_size:
        As for :func:`encoded_states`: the states of ``X`` and ``Y`` are
        simulated this many rows at a time.  Both sets are validated before
        the first batch.
    noise_level, noise_position:
        With ``noise_level > 0``, the Hilbert–Schmidt kernel
        ``Tr[ρ(x_i) ρ(y_j)]`` of the states under depolarising noise (see the
        module docstring and :func:`encoded_density_matrices`); its diagonal
        is the purity, below 1.  ``0`` (default) is the noiseless kernel,
        computed from state vectors as before.

    Returns
    -------
    torch.Tensor
        ``float64`` tensor of shape ``(n_samples_x, n_samples_y)`` (or
        ``(n_samples_x, n_samples_x)``), entries in ``[0, 1]`` up to rounding
        of order 1e-15.  The square matrix is symmetric, positive
        semi-definite, and has ones on the diagonal.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As :func:`encoded_states`, for ``X`` and ``Y``.  Both are validated
        before anything is simulated.

    Examples
    --------
    >>> import torch
    >>> from sklearn.svm import SVC
    >>> from hqnn_forge.encoding import QuantumEncodingLayer
    >>> from hqnn_forge.kernels import quantum_kernel_matrix
    >>> layer = QuantumEncodingLayer(n_qubits=4, n_layers=1, device_name="default.qubit")
    >>> g = torch.Generator().manual_seed(0)
    >>> X_train, X_test = torch.rand(10, 4, generator=g), torch.rand(3, 4, generator=g)
    >>> y_train = [0, 1] * 5
    >>> K_train = quantum_kernel_matrix(X_train, layer)
    >>> K_train.shape, K_train.dtype
    (torch.Size([10, 10]), torch.float64)
    >>> svm = SVC(kernel="precomputed").fit(K_train.numpy(), y_train)
    >>> K_test = quantum_kernel_matrix(X_test, layer, Y=X_train)
    >>> y_pred = svm.predict(K_test.numpy())

    Notes
    -----
    ``X`` and ``Y`` are simulated together in one batched circuit replay,
    and the kernel is one ``(n, 2^q) × (2^q, n)`` product, so time is
    O(n² · 2^q).  Memory is O(n · 2^q) for the states plus O(n²) for the
    matrix: at peak the complex128 Gram product and the float64 kernel sit
    side by side, about ``24 n²`` bytes, which is 9.6 GB at ``n = 20,000``.
    The states take ``(n_x + n_y) · 2^q · 16`` bytes more, and simulating them
    in one pass a multiple of that; ``batch_size`` bounds the simulation
    part (not with ``differentiable=True``, see :func:`encoded_states`).  For larger training sets, compute :func:`encoded_states` once and
    build the matrix in row blocks with :func:`kernel_from_states`, so the
    full matrix is never held as a complex Gram product::

        S = encoded_states(X_train, layer, batch_size=256)
        K = torch.cat([kernel_from_states(S[i : i + 1000], S) for i in range(0, len(S), 1000)])

    ``|G|²`` with ``G = S S†`` is the Schur
    product of a positive semi-definite matrix with its conjugate, hence
    positive semi-definite itself; small negative eigenvalues of order 1e-15
    are rounding.
    """
    _check_batch_size(batch_size)
    validate_noise(
        noise_level, noise_position, p_name="noise_level", position_name="noise_position"
    )
    encoder = _resolve_layer(layer, "quantum_kernel_matrix")
    qlayer, n_qubits = encoder.qlayer, encoder.n_qubits
    # Validate both input sets before simulating either.
    prepared_x = _prepare(X, encoder, "X")
    if noise_level > 0.0:
        prepared_y = None if Y is None else _prepare(Y, encoder, "Y")
        both = prepared_x if prepared_y is None else torch.cat([prepared_x, prepared_y])
        rho = _simulate_density(
            both, qlayer, n_qubits, noise_level, noise_position, differentiable, batch_size
        )
        if prepared_y is None:
            return kernel_from_density_matrices(rho)
        n_x = prepared_x.shape[0]
        return kernel_from_density_matrices(rho[:n_x], rho[n_x:])
    if Y is None:
        return kernel_from_states(
            _simulate(prepared_x, qlayer, n_qubits, differentiable, batch_size)
        )
    prepared_y = _prepare(Y, encoder, "Y")
    # One replay for both sets: same circuit and weights, one tape and device.
    states = _simulate(
        torch.cat([prepared_x, prepared_y]), qlayer, n_qubits, differentiable, batch_size
    )
    n_x = prepared_x.shape[0]
    return kernel_from_states(states[:n_x], states[n_x:])


# ---------------------------------------------------------------------------
# Kernel-target alignment
# ---------------------------------------------------------------------------


def kernel_target_alignment(K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Centred kernel-target alignment of ``K`` with labels ``y`` (Cortes et al.
    2012): the cosine between the centred kernel ``HKH`` and the centred ideal
    kernel ``H yyᵀ H``, ``H = I - 11ᵀ/M``, in the Frobenius inner product.

    It lies in [-1, 1] and is 1 when the kernel separates the classes exactly
    as the labels do.  Centring removes what every entry shares, so the
    alignment is not inflated by class imbalance or by a kernel that is large
    everywhere.

    Parameters
    ----------
    K:
        Square kernel matrix; differentiable if it comes from the
        ``differentiable=True`` path.
    y:
        Binary labels, ``{0, 1}`` or ``{-1, +1}``, with both classes present.

    Returns
    -------
    torch.Tensor
        0-d ``float64`` tensor.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix; got shape {tuple(K.shape)}.")
    labels = torch.as_tensor(y).reshape(-1).to(torch.float64)
    if labels.numel() != K.shape[0]:
        raise ValueError(f"y must have {K.shape[0]} labels; got {labels.numel()}.")
    values = set(labels.unique().tolist())
    if values == {0.0, 1.0}:
        labels = 2.0 * labels - 1.0
    elif values != {-1.0, 1.0}:
        raise ValueError(f"y must hold both classes as 0/1 or -1/+1; got values {sorted(values)}.")
    m = K.shape[0]
    centring = torch.eye(m, dtype=torch.float64) - torch.full((m, m), 1.0 / m, dtype=torch.float64)
    Kc = centring @ K.to(torch.float64) @ centring
    Yc = centring @ torch.outer(labels, labels) @ centring
    norm = torch.linalg.norm(Kc) * torch.linalg.norm(Yc)
    if norm == 0:
        return torch.zeros((), dtype=torch.float64)
    return (Kc * Yc).sum() / norm


def train_kernel_alignment(
    layer: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int = 50,
    lr: float = 0.05,
    subset_size: int | None = None,
    generator: torch.Generator | None = None,
) -> list[float]:
    """
    Train ``layer``'s parameters to maximise the kernel-target alignment on
    ``(X, y)``, in place; then fit ``SVC(kernel="precomputed")`` on
    ``quantum_kernel_matrix(X, layer)``.

    Every step computes the differentiable kernel on ``X`` (or, with
    ``subset_size``, on a random stratified subset, as Hubregtsen et al. 2022
    do to keep a step at ``subset_size²`` entries instead of ``M²``) and takes
    one Adam step on ``-alignment``.

    Only the weights that sit *between* uploads and the ``input_scaling`` of a
    :class:`~hqnn_forge.encoding.DataReuploadingLayer` change the kernel.  For the single-upload
    encoders the ansatz cancels in the kernel, so their gradient is zero and
    this does nothing useful: see the module docstring.

    Parameters
    ----------
    layer:
        An encoding layer; all its trainable parameters are optimised.
    X, y:
        Training inputs and binary labels.
    steps, lr:
        Adam steps and learning rate.
    subset_size:
        Samples per step, drawn per class in proportion; ``None`` uses all.
    generator:
        Source of the subsets.

    Returns
    -------
    list of float
        The alignment at each step, before that step's update.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps}.")
    labels = torch.as_tensor(y).reshape(-1)
    if subset_size is not None and not 2 <= subset_size <= labels.numel():
        raise ValueError(f"subset_size must lie in [2, {labels.numel()}]; got {subset_size}.")
    params = [p for p in layer.parameters() if p.requires_grad]
    if not params:
        raise ValueError("layer has no trainable parameters.")
    optimiser = torch.optim.Adam(params, lr=lr)
    positives = torch.nonzero(labels == labels.max()).reshape(-1)
    negatives = torch.nonzero(labels != labels.max()).reshape(-1)
    history: list[float] = []
    for _ in range(steps):
        if subset_size is None:
            rows = torch.arange(labels.numel())
        else:
            n_pos = max(1, round(subset_size * positives.numel() / labels.numel()))
            n_pos = min(n_pos, subset_size - 1)
            rows = torch.cat(
                [
                    positives[torch.randperm(positives.numel(), generator=generator)[:n_pos]],
                    negatives[
                        torch.randperm(negatives.numel(), generator=generator)[
                            : subset_size - n_pos
                        ]
                    ],
                ]
            )
        K = quantum_kernel_matrix(X[rows], layer, differentiable=True)
        alignment = kernel_target_alignment(K, labels[rows])
        history.append(float(alignment.detach()))
        optimiser.zero_grad()
        (-alignment).backward()
        optimiser.step()
    return history


# ---------------------------------------------------------------------------
# Overlap-circuit estimate
# ---------------------------------------------------------------------------


def _operations(
    prepared: torch.Tensor, qlayer: qml.qnn.TorchLayer
) -> list[list[qml.operation.Operator]]:
    """The layer's circuit (operations only, level 0) for each prepared input row."""
    weights = {name: p.detach().to(torch.float64) for name, p in qlayer.qnode_weights.items()}
    return [
        list(qml.workflow.construct_tape(qlayer.qnode, level=0)(row, **weights).operations)
        for row in prepared
    ]


def _overlap_tape(
    ops_x: list[qml.operation.Operator],
    ops_y: list[qml.operation.Operator],
    n_qubits: int,
    shots: int | None,
) -> qml.tape.QuantumScript:
    """``U(x)† U(y)|0⟩`` with ``P(0…0)`` = ``|⟨Φ(x)|Φ(y)⟩|²`` as its first probability."""
    ops = [*ops_y, *(qml.adjoint(op) for op in reversed(ops_x))]
    return qml.tape.QuantumScript(ops, [qml.probs(wires=range(n_qubits))], shots=shots)


def overlap_kernel_matrix(
    X: torch.Tensor,
    layer: nn.Module,
    Y: torch.Tensor | None = None,
    *,
    shots: int | None = None,
    seed: int | None = None,
    device: qml.devices.Device | None = None,
    project_psd: bool = False,
) -> torch.Tensor:
    """
    Kernel estimated entry by entry from the compute-uncompute circuit.

    Each entry is the probability of measuring all zeros after
    ``U(x_i)† U(y_j)|0…0⟩``, which equals ``|⟨Φ(x_i)|Φ(y_j)⟩|²``; ``U`` is the
    layer's own circuit, replayed exactly as :func:`quantum_kernel_matrix`
    replays it.  This is how a device without state-vector access estimates
    the kernel.

    For the square matrix only the ``M(M-1)/2`` pairs above the diagonal are
    circuits; the diagonal is set to 1 and the lower triangle mirrored.  The
    rectangular matrix against ``Y`` takes one circuit per entry.

    Parameters
    ----------
    X, layer, Y:
        As for :func:`quantum_kernel_matrix`.
    shots:
        Samples per circuit.  ``None`` (default) gives the exact probability,
        which agrees with :func:`quantum_kernel_matrix` to rounding; with
        shots each entry is a binomial estimate with standard error
        ``sqrt(k(1-k)/shots)``.
    seed:
        Seed of the sampling on the default device; ignored with ``device``.
    device:
        A PennyLane device to run the circuits on, e.g. a noisy simulator.
        Default: ``default.qubit``.
    project_psd:
        With finite shots or a noisy device the estimate need not be positive
        semi-definite, which ``SVC(kernel="precomputed")`` assumes.  ``True``
        projects the square matrix onto the nearest PSD matrix
        (:func:`nearest_psd`) before returning it.

    Returns
    -------
    torch.Tensor
        ``float64``, shape ``(n_x, n_y)`` or ``(n_x, n_x)``.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As :func:`quantum_kernel_matrix`; also ``ValueError`` for
        ``shots < 1`` or ``project_psd`` with ``Y``.
    """
    if shots is not None and shots < 1:
        raise ValueError(f"shots must be a positive integer or None; got {shots}.")
    if project_psd and Y is not None:
        raise ValueError("project_psd applies to the square matrix only; pass Y=None.")
    encoder = _resolve_layer(layer, "overlap_kernel_matrix")
    qlayer, n_qubits = encoder.qlayer, encoder.n_qubits
    prepared_x = _prepare(X, encoder, "X")
    prepared_y = None if Y is None else _prepare(Y, encoder, "Y")
    ops_x = _operations(prepared_x, qlayer)
    ops_y = ops_x if prepared_y is None else _operations(prepared_y, qlayer)

    if prepared_y is None:
        pairs = [(i, j) for i in range(len(ops_x)) for j in range(i + 1, len(ops_x))]
    else:
        pairs = [(i, j) for i in range(len(ops_x)) for j in range(len(ops_y))]
    kernel = torch.zeros(len(ops_x), len(ops_y), dtype=torch.float64)
    if pairs:
        tapes = [_overlap_tape(ops_x[i], ops_y[j], n_qubits, shots) for i, j in pairs]
        run_on = (
            device
            if device is not None
            else qml.device("default.qubit", wires=n_qubits, seed=seed)
        )
        results = qml.execute(tapes, run_on, diff_method=None)
        for (i, j), probs in zip(pairs, results):
            kernel[i, j] = float(torch.as_tensor(probs).reshape(-1)[0])
    if prepared_y is None:
        kernel = kernel + kernel.T
        kernel.fill_diagonal_(1.0)
        if project_psd:
            kernel = nearest_psd(kernel)
    return kernel


def nearest_psd(K: torch.Tensor) -> torch.Tensor:
    """
    The positive semi-definite matrix nearest to symmetric ``K`` in the
    Frobenius norm: ``K``'s eigendecomposition with negative eigenvalues set to
    zero (Higham 1988).

    The diagonal is not restored to 1 afterwards, so entries can move
    slightly; the projection changes nothing when ``K`` is already PSD.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix; got shape {tuple(K.shape)}.")
    symmetric = 0.5 * (K + K.T).to(torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    projected = (eigenvectors * eigenvalues.clamp(min=0.0)) @ eigenvectors.T
    return 0.5 * (projected + projected.T)


# ---------------------------------------------------------------------------
# Density matrices under depolarising noise (#221)
# ---------------------------------------------------------------------------


def _simulate_density(
    prepared: torch.Tensor,
    qlayer: qml.qnn.TorchLayer,
    n_qubits: int,
    noise_level: float,
    noise_position: Position,
    differentiable: bool = False,
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    Density matrices of the layer's circuit with depolarising channels inserted.

    ``batch_size`` works as in :func:`_simulate`: batches are written into
    one preallocated output rather than joined with ``torch.cat``.
    """
    n_rows = prepared.shape[0]
    dim = 2**n_qubits
    if batch_size is not None and batch_size < n_rows:
        rho = torch.empty(n_rows, dim, dim, dtype=torch.complex128)
        for start in range(0, n_rows, batch_size):
            stop = start + batch_size
            rho[start:stop] = _simulate_density(
                prepared[start:stop], qlayer, n_qubits, noise_level, noise_position, differentiable
            )
        return rho
    weights = {
        name: (p if differentiable else p.detach()).to(torch.float64)
        for name, p in qlayer.qnode_weights.items()
    }
    # The noisy QNode the models use (hqnn_forge.noise), taken at the "user"
    # level so its inserted channels are on the tape; p = 0 inserts none, as
    # apply_depolarizing_noise leaves the circuit untouched at p = 0.
    if noise_level > 0.0:
        qnode = _noisy_qnode(qlayer.qnode, n_qubits, noise_level, noise_position, "depolarizing")
        tape = qml.workflow.construct_tape(qnode, level="user")(prepared, **weights)
    else:
        tape = qml.workflow.construct_tape(qlayer.qnode, level=0)(prepared, **weights)
    tape = tape.copy(measurements=[qml.density_matrix(wires=range(n_qubits))])
    device = qml.device("default.mixed", wires=n_qubits)
    (result,) = qml.execute([tape], device, diff_method="backprop" if differentiable else None)
    return torch.as_tensor(result).to(torch.complex128).reshape(n_rows, dim, dim)


def encoded_density_matrices(
    X: torch.Tensor,
    layer: nn.Module,
    *,
    noise_level: float = 0.0,
    noise_position: Position = "all",
    differentiable: bool = False,
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    Density matrices ``ρ(x_i)`` the layer prepares under depolarising noise.

    The circuit is replayed as by :func:`encoded_states`, on ``default.mixed``
    with ``DepolarizingChannel(noise_level)`` inserted at ``noise_position``
    as :mod:`hqnn_forge.noise` does for the models; see "Under depolarising
    noise" in the module docstring.

    Parameters
    ----------
    X, layer, differentiable, batch_size:
        As for :func:`encoded_states`.
    noise_level:
        Depolarising probability per channel, in ``[0, 0.75]``.  ``0``
        inserts no channel, so the result is ``|Φ⟩⟨Φ|`` for the noiseless
        states.
    noise_position:
        ``"all"`` (after every gate) or ``"end"`` (before measurement), as in
        :func:`hqnn_forge.noise.apply_depolarizing_noise`.

    Returns
    -------
    torch.Tensor
        ``complex128`` of shape ``(n_samples, 2**n_qubits, 2**n_qubits)``,
        each Hermitian with unit trace.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As :func:`encoded_states`; also ``ValueError`` for a ``noise_level``
        or ``noise_position`` out of range.
    """
    validate_noise(
        noise_level, noise_position, p_name="noise_level", position_name="noise_position"
    )
    _check_batch_size(batch_size)
    encoder = _resolve_layer(layer, "encoded_density_matrices")
    qlayer, n_qubits = encoder.qlayer, encoder.n_qubits
    return _simulate_density(
        _prepare(X, encoder, "X"),
        qlayer,
        n_qubits,
        noise_level,
        noise_position,
        differentiable,
        batch_size,
    )


def kernel_from_density_matrices(
    rho_x: torch.Tensor, rho_y: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Hilbert–Schmidt kernel ``K[i, j] = Tr[ρ_i σ_j]`` from density matrices.

    ``Tr[ρσ] = Σ_ab ρ_ab conj(σ_ab)`` for Hermitian matrices, so ``K`` is the
    real Gram matrix of the vectorised matrices: symmetric and positive
    semi-definite.  The square matrix is made exactly symmetric; its
    diagonal holds the purities, not ones.

    Parameters
    ----------
    rho_x:
        Shape ``(n_x, d, d)``, as returned by :func:`encoded_density_matrices`.
    rho_y:
        Optional second set, shape ``(n_y, d, d)``.  ``None``: the square
        matrix of ``rho_x`` with itself.
    """
    symmetric = rho_y is None
    rho_y = rho_x if rho_y is None else rho_y
    if (
        rho_x.ndim != 3
        or rho_y.ndim != 3
        or rho_x.shape[1] != rho_x.shape[2]
        or rho_x.shape[1:] != rho_y.shape[1:]
    ):
        raise ValueError(
            f"rho_x and rho_y must be stacks of equal square matrices; got "
            f"{tuple(rho_x.shape)} and {tuple(rho_y.shape)}."
        )
    vx = rho_x.to(torch.complex128).reshape(rho_x.shape[0], -1)
    vy = vx if symmetric else rho_y.to(torch.complex128).reshape(rho_y.shape[0], -1)
    kernel = (vx @ vy.conj().T).real
    if symmetric:
        kernel = torch.triu(kernel) + torch.triu(kernel, 1).T
    return kernel
