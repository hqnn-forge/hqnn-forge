"""
hqnn_forge.rydberg.lindblad
===========================
Time evolution of a small Rydberg array under its Hamiltonian and Markovian
dephasing, and the readout of the resulting state.  This is the solver of
``docs/rydberg-model.md``.

Units: ħ = 1, the Hamiltonian and γ in rad/µs, times in µs, so ``H t`` and
``γ t`` are dimensionless.  Basis ordering is that of
:mod:`hqnn_forge.rydberg.hamiltonian`: atom ``i`` is tensor factor ``i`` from
the left, and bit ``b_i(k) = (k >> (N−1−i)) & 1`` of the index ``k`` of a
basis state is 1 when atom ``i`` is in ``|r⟩`` (atom 0 the most significant
bit).  Index 0 is ``|g…g⟩``.

Master equation
---------------
::

    dρ/dt = 𝓛_H ρ + 𝓛_D ρ

    𝓛_H ρ = −i [H, ρ]
    𝓛_D ρ = Σ_i ( L_i ρ L_i† − ½ { L_i† L_i , ρ } ),        L_i = √γ n_i

with one collapse operator per atom, ``n_i = |r⟩⟨r|_i``, and
``ρ(0) = |g…g⟩⟨g…g|`` unless :func:`evolve` is given another ``rho0``.  There
is no global variant of the operator and no decay term.

The dissipator, elementwise
---------------------------
``n_i`` is diagonal in the computational basis with entry ``a_i`` on the
state ``a``, and ``n_i² = n_i``, so ``L_i† L_i = γ n_i``.  On the matrix
element ``ρ_ab`` between basis states ``a`` and ``b``::

    (n_i ρ n_i)_ab        =  a_i b_i ρ_ab
    −½ ({n_i, ρ})_ab      = −½ (a_i + b_i) ρ_ab
    sum                   = −½ (a_i + b_i − 2 a_i b_i) ρ_ab = −½ (a_i − b_i)² ρ_ab

(the last step uses ``a_i² = a_i`` for a bit).  ``(a_i − b_i)²`` is 1 where
the two states differ on atom ``i`` and 0 otherwise, so the sum over the
atoms counts the differing atoms, the Hamming distance ``d_H(a, b)``::

    (𝓛_D ρ)_ab = −(γ/2) · d_H(a, b) · ρ_ab

``𝓛_D`` is therefore diagonal in the elements of ρ, and its exponential is an
elementwise multiplication, exactly, for any τ::

    (e^(𝓛_D τ) ρ)_ab = exp(−γ τ d_H(a, b) / 2) · ρ_ab

* **What γ means.**  The coherence of a single atom (``d_H = 1``) decays at
  rate ``γ/2``, i.e. ``T₂ = 2/γ`` without a drive.  A coherence between
  states that differ on ``d`` atoms decays at ``d γ/2``.  Populations
  (``d_H = 0``) are untouched.  ``√γ n_i`` and ``√(γ/4) Z_i`` give the same
  equation; a collapse operator ``√κ Z_i`` corresponds to ``γ = 4κ``.
* **No ordering enters.**  ``d_H`` is symmetric in the atoms, so the factor
  does not depend on which bit belongs to which atom.

The unitary part
----------------
``H`` is constant in time.  With its eigendecomposition ``H = V E V†``
(``E`` real and diagonal, ``V`` unitary)::

    U(τ) = e^(−iHτ) = V e^(−iEτ) V†,        e^(𝓛_H τ) ρ = U(τ) ρ U(τ)†

``evolve`` computes ``U`` this way in every autograd mode, so its result does
not depend on whether, or how, it is being differentiated (#674).

The sign is that of the Schrödinger equation ``i dψ/dt = Hψ``.  For one
resonant atom it gives ``ψ(t) = (cos(Ωt/2), −i sin(Ωt/2))`` in the basis
``(|g⟩, |r⟩)``, so ``⟨n⟩(t) = sin²(Ωt/2)`` and ``ρ_gr = +(i/2) sin(Ωt)``.

Without dephasing the state stays pure, ``ρ(t) = ψ(t) ψ(t)†`` with::

    ψ(t) = U(t) e_0,        ψ_k(t) = Σ_m V_km e^(−i E_m t) conj(V_0m)

``e_0`` being the basis vector of ``|g…g⟩``.  This is exact at any ``t``.
From another initial state ``ρ(0)``, which may be mixed, it is
``ρ(t) = U(t) ρ(0) U(t)†``, exact as well.

Derivatives of the propagator
-----------------------------
Differentiating ``V`` and ``E`` (the backward pass of ``eigh``) needs
``1/(E_m − E_k)``, which is not defined where two energies coincide: autograd
then returns a finite, wrong gradient (#505).  ``U`` itself is a smooth
function of ``H``, and its derivative is taken in the eigenbasis of ``H``
without that term (the Daleckii–Krein formula; Higham, *Functions of
Matrices*, SIAM 2008, chapter 3).  For ``f(x) = e^(−ixτ)``, ``U = f(H)``, the derivative in a
direction ``Ḣ`` (any complex matrix, not only a Hermitian one) is::

    L(H, Ḣ) = d/dε f(H + εḢ) |_(ε=0) = V ( F ∘ (V† Ḣ V) ) V†

    F_mk = (f(E_m) − f(E_k)) / (E_m − E_k),        F_mm = f′(E_m) = −iτ f(E_m)

with ``∘`` the elementwise product.  Rewritten around the mean energy, with
``sinc(x) = sin(x)/x`` and ``sinc(0) = 1``::

    f(E_m) − f(E_k) = e^(−iτ(E_m + E_k)/2) · (e^(−iτδ/2) − e^(iτδ/2))
                    = e^(−iτ(E_m + E_k)/2) · (−2i) sin(τδ/2),        δ = E_m − E_k

    F_mk = −iτ · e^(−iτ(E_m + E_k)/2) · sinc(τδ/2)

which holds at ``δ = 0`` as well, where it is ``f′(E_m)``.  No difference is
divided, so ``F`` is accurate to rounding at any gap, ``0``, ``1e-9`` or
large: a degenerate or nearly degenerate spectrum needs no tolerance.  Units:
``E`` and ``δ`` in rad/µs, ``τ`` in µs, ``F`` in µs.  This gives forward mode
(``jvp``) directly.  For reverse mode, autograd's gradient of a real loss is
the adjoint of ``L(H, ·)`` under ``⟨A, B⟩ = Re Tr(A† B)``; with ``V`` unitary
it replaces ``F`` by ``conj(F)``, the ``F`` of ``e^(+ixτ)``::

    grad_H = V ( conj(F) ∘ (V† G V) ) V† = L(H, G) at −τ,        G = grad_U

Both reuse the ``V`` and ``E`` of the forward pass: four ``2^N × 2^N`` matrix
products, and no second eigendecomposition.

``L`` is itself differentiated, for second derivatives, through the
block-triangular form ``f([[M, X], [0, M]]) = [[f(M), L(M, X)], [0, f(M)]]``
(Higham 2008, chapter 3), which holds for any square ``M``.  With
``M = [[H, X], [0, H]]``, ``L(H, X)`` is the upper right ``2^N × 2^N`` block
of ``f(M)``.  Its derivative in ``H`` is the upper right block of
``L(M, [[Ḣ, 0], [0, Ḣ]])`` (forward mode).  In reverse mode the cotangent of
``f(M)`` is ``[[0, G], [0, 0]]``, the gradient of ``f`` at ``M``, which is not
Hermitian, is the adjoint ``L(M†, ·)`` at ``−τ``, and ``H`` sits in both
diagonal blocks of ``M``, so ``grad_H`` is the sum of those two blocks of
it.  Each ``L(M, ·)`` is the upper right block of a
``torch.linalg.matrix_exp`` of twice the size of ``M``.
Only second derivatives take this path, which costs what
``torch.linalg.matrix_exp`` costs; first derivatives never do.

Splitting
---------
With dephasing the two parts do not commute, and the evolution over ``t`` is
taken in ``n`` steps of ``dt = t/n`` by the symmetric (Strang) splitting::

    S(dt) = e^(𝓛_D dt/2) · e^(𝓛_H dt) · e^(𝓛_D dt/2),        ρ(t) ≈ S(dt)^n ρ(0)

read from the right: damp for ``dt/2``, rotate for ``dt``, damp for ``dt/2``.
Both factors are exact, so the only error is that of the splitting.

* **Order.**  ``S(dt)`` is symmetric, ``S(dt) S(−dt) = 1``, so its logarithm
  holds only odd powers of ``dt``: ``S(dt) = exp((𝓛_H + 𝓛_D) dt + O(dt³))``.
  The error of one step is ``O(dt³)`` and the **global error after
  ``n = t/dt`` steps is of second order, ``O(t · dt²)``**: halving ``dt``
  divides it by 4.
* **Its size.**  The ``dt³`` term is built from the double commutators
  ``[𝓛_D, [𝓛_D, 𝓛_H]]`` and ``[𝓛_H, [𝓛_H, 𝓛_D]]``, of order ``(Nγ)² W``
  and ``Nγ W²``, with ``W = E_max − E_min`` the spread of the energies of
  ``H`` (a constant shift of ``H`` leaves ``𝓛_H`` unchanged, so only the
  spread counts).  ``dt`` therefore has to resolve both the fastest
  dephasing rate ``Nγ/2`` and ``W``, which in the blockade is set by ``V``,
  not Ω; :func:`evolve` states the condition.  It vanishes when ``H`` is
  diagonal (no drive): ``𝓛_H``
  then also acts elementwise, the two parts commute and one step is exact.
* **Every step is a physical map** at any ``dt``.  ``U ρ U†`` is unitary
  conjugation.  The damping factor ``exp(−γ τ d_H(a, b)/2)`` is the matrix of
  a product of single-atom dephasing channels: it has a unit diagonal and is
  positive semidefinite, so the elementwise product keeps trace,
  Hermiticity and non-negative eigenvalues (Schur product theorem).  A step
  that is too large gives an inaccurate state, never an unphysical one.
* **The maximally mixed state is a fixed point of every step**, as it is of
  the equation: the factor is 1 on the diagonal and ``U 1 U† = 1``.
* **Adjacent half steps are merged.**  The damping factors of consecutive
  steps multiply, ``e^(𝓛_D dt/2) e^(𝓛_D dt/2) = e^(𝓛_D dt)``, exactly.  The
  loop applies one half step at the start, a full step between rotations and
  one half step at the end, which is ``S(dt)^n`` term by term.

Piecewise-constant Hamiltonians
-------------------------------
A Hamiltonian that is constant on each of several segments, ``H_k`` for a
time ``τ_k``, is evolved by one call per segment, each starting from the
state the previous one returned (``rho0``)::

    ρ_(k+1) = S_k(τ_k / n_k)^(n_k) ρ_k,        S_k(dt) = e^(𝓛_D dt/2) e^(𝓛_(H_k) dt) e^(𝓛_D dt/2)

The splitting of each segment is that of a constant ``H`` and keeps its
second order; the segment boundaries add no error of their own, since the
exact evolution is that same product of the exact segment maps.  Two calls
with the same ``H`` and the same ``dt`` are one call with the steps added
up: the half step that ends the first and the half step that starts the
second are one full step, as inside a call.

Readout
-------
``⟨n_i⟩ = Tr[ρ n_i] = Σ_k b_i(k) ρ_kk`` and
``⟨n_i n_j⟩ = Σ_k b_i(k) b_j(k) ρ_kk``: both are sums over the diagonal of ρ,
the outcome probabilities of measuring every atom in the ``(|g⟩, |r⟩)`` basis.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from hqnn_forge.rydberg.hamiltonian import MAX_ATOMS, _bits

# torch 2.3 imports torch._dynamo the first time an autograd.Function runs
# under a torch.func transform, and that import fails there (a circular
# import, then a random operation inside vmap).  Imported here, outside any
# transform, it does not; torch 2.4 needs no help, so this goes once the
# torch floor reaches 2.4.
if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 4):
    import torch._dynamo

__all__ = ["evolve", "readout", "step_count"]

#: Largest ``|H − H†|`` entry :func:`evolve` accepts in a sample, relative to
#: the largest ``|H|`` entry of that sample.  A matrix built in ``float64`` from
#: a Hermitian expression is Hermitian to a few units of 1e-16.
_HERMITIAN_RTOL = 1e-12
#: The same bound in units of the machine epsilon of the input's dtype, which
#: is the larger of the two below double precision: a matrix computed in
#: ``complex64`` is Hermitian to its own rounding only (``1.2e-5`` relative).
#: The epsilon is that of single precision at most: with a half-precision
#: one (``bfloat16``: 100 ε = 0.78) the bound would accept nearly any matrix.
_HERMITIAN_EPS = 100
#: Smallest ``tol`` :func:`step_count` accepts.  Below an error of about
#: ``1e-10`` the error of :func:`evolve` stops falling by 4 per halving of
#: ``dt``, as rounding takes over, and two runs that close say nothing about
#: it (``evolve``, "Choosing ``n_steps``"); ``1e-9`` keeps a factor 10 above
#: that floor.  Measured: at ``tol = 1e-11`` a four-atom case was accepted
#: with an error of ``4.1e-11``, four times ``tol``.
_SMALLEST_TOL = 1e-9
#: The factor by which each estimate of :func:`step_count` must fall below the
#: one before while it is above ``tol``.  In the second-order regime it falls
#: by 4 (3.91 to 4.12 over the sweep of #598); a smaller fall means rounding,
#: which grows with the step count, has taken over.
_SMALLEST_FALL = 2.0


def _non_negative_number(value: Any, name: str) -> float:
    """
    ``value`` as a float ``>= 0``, or a ``ValueError`` naming the argument.

    Numbers only (#530): a Python or NumPy scalar, never a tensor or an array.
    ``float()`` would read a one-element tensor and drop its graph without a
    word, so a tensor or array is refused whatever its shape, with a message
    naming the shape for more than one element.  For one element it names
    ``.item()`` only when the value ``.item()`` returns would be accepted, so
    a ``tensor(nan)`` or ``tensor(-1.)`` is not sent to a second refusal.
    """
    if isinstance(value, (torch.Tensor, np.ndarray)):
        kind = "tensor" if isinstance(value, torch.Tensor) else "NumPy array"
        if math.prod(value.shape) == 1:
            message = f"{name} must be a finite number >= 0, not a {kind}; got {value!r}."
            try:
                _non_negative_number(value.item(), name)
            except (ValueError, RuntimeError):  # RuntimeError: a meta tensor has no value
                raise ValueError(message) from None
            raise ValueError(
                f"{message} Pass {name}.item() to use its value: no derivative with "
                f"respect to {name} is taken."
            )
        # ValueError, as for any other value that is not a finite number >= 0
        raise ValueError(  # noqa: TRY004
            f"{name} must be a single finite number >= 0, not a {kind} of shape "
            f"{tuple(value.shape)}."
        )
    number = math.nan
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:  # an int beyond the float range
            number = math.inf
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite number >= 0; got {value!r}.")
    return number


def _step_count(n_steps: Any) -> int:
    """``n_steps`` as an ``int >= 1``, or a ``ValueError``; :func:`step_count` chooses one."""
    if isinstance(n_steps, bool) or not isinstance(n_steps, numbers.Integral) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1; got {n_steps!r}.")
    return int(n_steps)


def _atom_count(matrix: Any, name: str) -> int:
    """
    The number of atoms ``n`` of a ``(2^n, 2^n)`` or ``(batch, 2^n, 2^n)`` tensor.

    Raises ``TypeError`` for anything but a real or complex floating tensor and
    ``ValueError`` for another shape, more than :data:`MAX_ATOMS` atoms, or a
    NaN or an infinity.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(matrix).__name__}.")
    if not (matrix.is_complex() or matrix.is_floating_point()):
        raise TypeError(
            f"{name} must be a real or complex floating tensor; got dtype {matrix.dtype}."
        )
    dim = matrix.shape[-1] if matrix.ndim else 0
    n = dim.bit_length() - 1
    if (
        matrix.ndim not in (2, 3)
        or matrix.shape[-2] != dim
        or dim != 2**n
        or not 1 <= n <= MAX_ATOMS
    ):
        raise ValueError(
            f"{name} must have shape (2^n, 2^n) or (batch, 2^n, 2^n) for 1 to {MAX_ATOMS} "
            f"atoms; got shape {tuple(matrix.shape)}."
        )
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must be finite; got a NaN or an infinity.")
    return n


def _initial_state(rho0: Any, shape: torch.Size) -> torch.Tensor:
    """
    ``rho0`` as ``complex128``, after checking it against the Hamiltonian's ``shape``.

    Accepted: ``(2^N, 2^N)``, shared by the batch, or ``shape`` itself.
    ``TypeError`` for anything but a real or complex floating tensor,
    ``ValueError`` for another shape or a NaN or an infinity.
    """
    if not isinstance(rho0, torch.Tensor):
        raise TypeError(f"rho0 must be a torch.Tensor; got {type(rho0).__name__}.")
    if not (rho0.is_complex() or rho0.is_floating_point()):
        raise TypeError(f"rho0 must be a real or complex floating tensor; got dtype {rho0.dtype}.")
    single = tuple(shape[-2:])
    if rho0.shape not in (shape, single):
        allowed = (
            f"{single}, that of hamiltonian"
            if len(shape) == 2
            else f"{single} or {tuple(shape)}: one state for every sample, or one per sample"
        )
        raise ValueError(f"rho0 must have shape {allowed}; got shape {tuple(rho0.shape)}.")
    if not bool(torch.isfinite(rho0).all()):
        raise ValueError("rho0 must be finite; got a NaN or an infinity.")
    return rho0.to(torch.complex128)


def _hermitian(hamiltonian: torch.Tensor) -> torch.Tensor:
    """
    The Hermitian part of ``hamiltonian`` as ``complex128``, after checking it is Hermitian.

    The caller has checked its shape and finiteness (:func:`_atom_count`).
    ``ValueError`` if a sample's largest ``|H − H†|`` entry exceeds the bound
    of :func:`evolve` (Notes, *Hermiticity*).
    """
    h = hamiltonian.to(torch.complex128)
    # eigh reads one triangle only: a non-Hermitian matrix would be evolved as
    # another, Hermitian one without any error.
    # Each sample is measured against its own largest entry: one scale for the
    # batch would let a sample with a strong interaction hide another's
    # asymmetry.  An empty batch, which rydberg_hamiltonian returns for no
    # samples, has no sample to compare and comes back empty.
    eps = min(torch.finfo(hamiltonian.dtype).eps, torch.finfo(torch.float32).eps)
    rtol = max(_HERMITIAN_RTOL, _HERMITIAN_EPS * eps)
    scale = h.detach().abs().amax(dim=(-2, -1)).reshape(-1)
    asymmetry = (h - h.mH).detach().abs().amax(dim=(-2, -1)).reshape(-1)
    failed = (asymmetry > rtol * scale).nonzero()
    if failed.numel():
        sample = int(failed[0])
        where = f" of sample {sample}" if h.ndim == 3 else ""
        raise ValueError(
            f"hamiltonian must be Hermitian; the largest |H - H†| entry{where} is "
            f"{float(asymmetry[sample]):.3g} (largest |H| entry {float(scale[sample]):.3g})."
        )
    # The Hermitian part, so that the lower triangle, the only one eigh reads,
    # determines the whole matrix: the derivatives of _Propagator treat every
    # entry of its input as a variable, so it has to be the matrix that is
    # evolved, not just one triangle of it.
    return (h + h.mH) / 2


def _divided_differences(energies: torch.Tensor, tau: float) -> torch.Tensor:
    """
    ``F_mk = −iτ · e^(−iτ(E_m + E_k)/2) · sinc(τ(E_m − E_k)/2)`` for real ``energies``.

    The divided differences of ``f(x) = e^(−ixτ)`` at the energies, with
    ``f′(E_m)`` on the diagonal (module docstring, *Derivatives of the
    propagator*).  ``torch.sinc(x)`` is ``sin(πx)/(πx)``, hence the ``/ π``.
    Shape ``(..., 2^N)`` in, ``(..., 2^N, 2^N)`` out, ``m`` on the rows.
    """
    mean = (energies[..., :, None] + energies[..., None, :]) / 2
    half_gap = (energies[..., :, None] - energies[..., None, :]) * (tau / 2)
    return -1j * tau * torch.exp(-1j * tau * mean) * torch.sinc(half_gap / math.pi)


def _block_frechet(matrix: torch.Tensor, direction: torch.Tensor, tau: float) -> torch.Tensor:
    """
    ``L(M, X)``, the derivative of ``e^(−iMτ)`` at any square ``M`` in direction ``X``.

    The upper right block of ``e^(−iτ [[M, X], [0, M]])`` (module docstring).
    Used for second derivatives only: ``torch.linalg.matrix_exp`` of twice the
    size of ``M``.
    """
    matrix, direction = torch.broadcast_tensors(matrix, direction)
    dim = matrix.shape[-1]
    top = torch.cat([matrix, direction], dim=-1)
    bottom = torch.cat([torch.zeros_like(matrix), matrix], dim=-1)
    block = torch.cat([top, bottom], dim=-2)
    return torch.linalg.matrix_exp(-1j * tau * block)[..., :dim, dim:]


class _Frechet(torch.autograd.Function):
    """
    ``L(H, X) = V (F ∘ (V† X V)) V†``, the derivative of ``e^(−iHτ)`` in direction ``X``.

    ``H`` is Hermitian with eigendecomposition ``H = V E V†``; ``energies`` and
    ``vectors`` are that decomposition and carry no gradient.  The first
    derivative of ``U`` evaluates this function; its own derivatives (second
    derivatives of ``U``) are those of the module docstring: in ``X`` it is
    linear, in ``H`` it goes through the block-triangular form.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(  # type: ignore[override]
        h: torch.Tensor,
        direction: torch.Tensor,
        energies: torch.Tensor,
        vectors: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        rotated = vectors.mH @ direction @ vectors  # X in the eigenbasis of H
        return vectors @ (_divided_differences(energies, tau) * rotated) @ vectors.mH

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        h, direction, energies, vectors, tau = inputs
        ctx.save_for_backward(h, direction, energies, vectors)
        ctx.save_for_forward(h, direction, energies, vectors)
        ctx.tau = tau

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        h, direction, energies, vectors = ctx.saved_tensors
        tau = ctx.tau
        grad_h = grad_direction = None
        if ctx.needs_input_grad[1]:
            # adjoint of X ↦ L(H, X): L(H, ·) at −τ, H being Hermitian
            grad_direction = _Frechet.apply(h, grad, energies, vectors, -tau)
        if ctx.needs_input_grad[0]:
            # L(H, X) is the upper right block of f(M), M = [[H, X], [0, H]].  The
            # gradient of f at M for the cotangent [[0, G], [0, 0]] is L(M†, ·) at
            # −τ; H sits in both diagonal blocks of M, so their gradients add.
            dim = h.shape[-1]
            zero = torch.zeros_like(h)
            matrix = torch.cat(
                [torch.cat([h, direction], dim=-1), torch.cat([zero, h], dim=-1)], dim=-2
            )
            cotangent = torch.cat(
                [torch.cat([zero, grad], dim=-1), torch.cat([zero, zero], dim=-1)], dim=-2
            )
            grad_matrix = _block_frechet(matrix.mH, cotangent, -tau)
            grad_h = grad_matrix[..., :dim, :dim] + grad_matrix[..., dim:, dim:]
        return grad_h, grad_direction, None, None, None

    @staticmethod
    def jvp(  # type: ignore[override]
        ctx: Any,
        h_tangent: torch.Tensor | None,
        direction_tangent: torch.Tensor | None,
        *_: Any,
    ) -> torch.Tensor:
        h, direction, energies, vectors = ctx.saved_tensors
        tau = ctx.tau
        dim = h.shape[-1]
        tangent = torch.zeros_like(direction)
        if direction_tangent is not None:  # linear in X
            tangent = tangent + _Frechet.apply(h, direction_tangent, energies, vectors, tau)
        if h_tangent is not None:
            # upper right block of L(M, [[Ḣ, 0], [0, Ḣ]]), M = [[H, X], [0, H]]
            zero = torch.zeros_like(h)
            matrix = torch.cat(
                [torch.cat([h, direction], dim=-1), torch.cat([zero, h], dim=-1)], dim=-2
            )
            h_tangent = h_tangent.expand_as(h)
            diagonal = torch.cat(
                [
                    torch.cat([h_tangent, torch.zeros_like(h_tangent)], dim=-1),
                    torch.cat([torch.zeros_like(h_tangent), h_tangent], dim=-1),
                ],
                dim=-2,
            )
            tangent = tangent + _block_frechet(matrix, diagonal, tau)[..., :dim, dim:]
        return tangent


class _Propagator(torch.autograd.Function):
    """
    ``U(τ) = V e^(−iEτ) V†`` from ``eigh``, or its first column only.

    Differentiated by ``_Frechet`` in both modes, never through ``eigh``.
    Returns ``(U, E, V)``; ``E`` and ``V`` are not differentiable.  With
    ``first_column``, ``U e_0`` of shape ``(..., 2^N, 1)``: the state that
    ``|g…g⟩`` evolves to, ``ψ_k = Σ_m V_km e^(−i E_m τ) conj(V_0m)``.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(  # type: ignore[override]
        h: torch.Tensor, tau: float, first_column: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # H = V E V†: the columns of V are the energy eigenvectors.
        energies, vectors = torch.linalg.eigh(h)
        phases = torch.exp(-1j * energies * tau)
        if first_column:
            # ψ_k(τ) = Σ_m V_km e^(−i E_m τ) conj(V_0m); evolve |g…g⟩.
            propagator = vectors @ (phases * vectors[..., 0, :].conj())[..., :, None]
        else:
            # U(τ) = V e^(−i E τ) V†: the exact unitary step from the eigendecomposition.
            propagator = (vectors * phases[..., None, :]) @ vectors.mH
        return propagator, energies, vectors

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        h, tau, first_column = inputs
        _, energies, vectors = output
        ctx.mark_non_differentiable(energies, vectors)
        ctx.save_for_backward(h, energies, vectors)
        ctx.save_for_forward(h, energies, vectors)
        ctx.tau = tau
        ctx.first_column = first_column

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad: torch.Tensor, *_: Any
    ) -> tuple[torch.Tensor | None, None, None]:
        h, energies, vectors = ctx.saved_tensors
        if ctx.first_column:  # U e_0 = U[:, :1]: the other columns get no gradient
            grad = torch.nn.functional.pad(grad, (0, h.shape[-1] - 1))
        # grad_H = L(H, grad_U) at −τ, the adjoint of L(H, ·) (module docstring)
        return _Frechet.apply(h, grad, energies, vectors, -ctx.tau), None, None

    @staticmethod
    def jvp(  # type: ignore[override]
        ctx: Any, h_tangent: torch.Tensor, *_: Any
    ) -> tuple[torch.Tensor, None, None]:
        h, energies, vectors = ctx.saved_tensors
        tangent = _Frechet.apply(h, h_tangent, energies, vectors, ctx.tau)
        if ctx.first_column:
            tangent = tangent[..., :, :1]
        return tangent, None, None


def evolve(
    hamiltonian: torch.Tensor,
    t: float,
    *,
    gamma: float = 0.0,
    n_steps: int | None = None,
    rho0: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    The state ``ρ(t)`` of an array that starts in ``rho0``, by default every atom in ``|g⟩``.

    Solves, for one or a batch of time-independent Hamiltonians::

        dρ/dt = −i [H, ρ] + γ Σ_i ( n_i ρ n_i − ½ { n_i , ρ } ),    ρ(0) = |g…g⟩⟨g…g|

    the Lindblad equation of ``docs/rydberg-model.md`` with the local
    collapse operators ``L_i = √γ n_i``.  The module docstring derives every
    formula used here, with the sign and ordering conventions.

    * ``gamma = 0``: ``ρ(t) = ψψ†`` with ``ψ = e^(−iHt) |g…g⟩``, exact up to
      floating-point rounding, from one eigendecomposition per sample and
      without time stepping; from a given ``rho0``, ``U ρ(0) U†`` with
      ``U = e^(−iHt)``, exact as well.
    * ``gamma > 0``: ``n_steps`` steps of ``dt = t / n_steps`` of the
      symmetric (Strang) splitting
      ``e^(𝓛_D dt/2) · e^(𝓛_H dt) · e^(𝓛_D dt/2)``.  The unitary step is
      ``ρ → U ρ U†`` with ``U = e^(−iH dt)`` from the same
      eigendecomposition; the dissipative step multiplies ``ρ_ab`` by
      ``exp(−γ τ d_H(a, b) / 2)``, ``d_H`` the Hamming distance of the basis
      states ``a`` and ``b``.  Both are exact up to floating-point rounding;
      the splitting has a **global error of second order in ``dt``**.

    The result is the same in every autograd mode: with or without
    gradients, under ``torch.no_grad``, ``torch.func`` or forward-mode dual
    tensors, ``evolve`` runs the same eigendecomposition (#674).

    Parameters
    ----------
    hamiltonian:
        Shape ``(2^N, 2^N)``, or ``(batch, 2^N, 2^N)`` for a batch, in
        rad/µs, e.g. from :func:`~hqnn_forge.rydberg.rydberg_hamiltonian`;
        ``N`` may be 1 to :data:`~hqnn_forge.rydberg.MAX_ATOMS`.  Hermitian,
        real or complex.  Its basis ordering defines that of the result:
        index 0 is the initial state.
    t:
        Evolution time in µs, ``>= 0``, the same for every sample.  A Python
        or NumPy number; **a tensor is not accepted**, nor an array, whatever
        its shape (#530): ``t`` is not differentiable, so pass ``t.item()``
        for the value of a one-element tensor.  One time per sample is not
        supported (#506).
    gamma:
        Dephasing rate γ in rad/µs, ``>= 0``, the same for every atom and
        sample.  It is the rate of the collapse operators ``√γ n_i``: **a
        single-atom coherence decays at ``γ/2``** (``T₂ = 2/γ``), and
        ``ρ_ab`` at ``γ/2 · d_H(a, b)``.  A number, as for ``t``: **a tensor
        is not accepted**, nor an array, and ``gamma`` is not
        differentiable.
    n_steps:
        Number of splitting steps, an integer ``>= 1``.  Required when
        ``gamma > 0`` and without a default: the step that is accurate enough
        depends on γ and on the spread ``W = E_max − E_min`` of the energies
        of ``H`` (set by the interaction, in the blockade), which this
        function does not guess (see "Choosing ``n_steps``" below).  Not
        used when ``gamma = 0``, where the result is exact up to
        floating-point rounding.
    rho0:
        The state ``ρ(0)``, in the basis of ``hamiltonian``: shape
        ``(2^N, 2^N)``, shared by every sample, or the shape of a batched
        ``hamiltonian``, one state per sample.  Real or complex; converted to
        ``complex128``.  ``None`` (default) is ``|g…g⟩⟨g…g|``.  It is not
        checked to be a density matrix: the evolution is linear and evolves
        whatever matrix it is given.  Gradients reach it.  A
        piecewise-constant Hamiltonian is one call per segment with ``rho0``
        the previous segment's result (module docstring, *Piecewise-constant
        Hamiltonians*).

    Returns
    -------
    torch.Tensor
        ``ρ(t)``, ``complex128``, with the shape and device of
        ``hamiltonian``.  Entry ``[..., a, b]`` is ``⟨a|ρ|b⟩``.  Trace 1,
        Hermitian and positive semidefinite up to rounding, at any step size,
        when ``rho0`` is (as the default is);
        the rounding error of the trace grows by about ``1e-16`` to ``1e-15``
        per step (``1.4e-15`` per step for 6 atoms 0.6 µm apart,
        ``‖H‖₂ ≈ 6e8`` rad/µs).

    Raises
    ------
    TypeError
        If ``hamiltonian`` or ``rho0`` is not a real or complex floating
        tensor.
    ValueError
        If ``hamiltonian`` has another shape, is for more than
        :data:`~hqnn_forge.rydberg.MAX_ATOMS` atoms, holds a NaN or an
        infinity, or is not Hermitian; if ``rho0`` has a shape other than
        the two above or is not finite; if ``t`` or ``gamma`` is not a finite
        number ``>= 0``, or is a tensor or an array of any shape; if
        ``n_steps`` is not an integer ``>= 1``, or is missing with
        ``gamma > 0``.

    Notes
    -----
    * **Precision.**  The Hamiltonian is converted to ``complex128`` whatever
      its dtype.
    * **Hermiticity** is checked per sample: the largest ``|H − H†|`` entry
      may be at most ``1e-12`` of the sample's largest ``|H|`` entry, or 100
      machine epsilons of the input's dtype where that is more (``1.2e-5``
      for single precision, and no more than that below it).
    * **Choosing ``n_steps``.**  No count is assumed here, and none suits
      every input (#507).  Once ``dt`` is small, halving it divides the error
      by 4, so two runs with ``n`` and ``2n`` steps differ by about 3/4 of
      the coarser run's error, and **the finer run is off by about a third
      of their difference**.  ``dt`` is small when it resolves the fastest
      rate of the generator: the rates of ``𝓛_H`` are the differences
      ``E_k − E_l`` of the energies of ``H`` (``𝓛_H`` maps ``|k⟩⟨l|`` to
      ``−i (E_k − E_l) |k⟩⟨l|`` in the eigenbasis of ``H``), at most their
      spread ``W = E_max − E_min``, and those of ``𝓛_D`` are
      ``(γ/2) d_H(a, b)`` for basis states ``a``, ``b``, at most ``Nγ/2``.
      The two parts do not commute, so ``W + Nγ/2`` bounds the rates of the
      generator rather than adding up shared ones.  With
      ``dt · (W + Nγ/2) <= 1/2`` for the coarser run, the finer run's largest
      error over the features ``⟨n_i⟩``, ``⟨n_i n_j⟩`` was within 10 % of a
      third of their largest difference in 99 % of the cases of five seeded
      sweeps of 300 cases each (#627): 1, 2 and 4 atoms, ``V/Ω`` from 0 to
      100, ``γ/Ω`` from 0.1 to 100, ``ΩT`` from π to 16π, detunings within
      ``±2 Ω``.  It is an estimate, not a bound: single atoms reached 1.6
      times it in the sweeps, and 0.8 at a point outside them.  A single
      feature's error was 0.84 to 5.0 times a third of its own difference,
      and at ``dt · (W + Nγ/2) <= 1`` the largest error was up to 3.5 times
      the estimate.  Errors below about
      ``1e-10`` stop falling by 4 per halving, as rounding takes over, and
      there the estimate says nothing.  That is a condition for trusting the
      estimate, not the count the features need, which can be far smaller.
      **Before it holds, two runs can agree while both are wrong**: one
      resonant atom at ``ΩT = 16π``, ``γ = 0.1 Ω`` ends with ``⟨n⟩ = 0`` to
      rounding at 1, 2, 4, 8 and 16 steps, where the answer is 0.358.  Each
      of those steps is a whole number of π pulses, so between steps the
      atom is in ``|g⟩`` or ``|r⟩`` and leaves no coherence to dephase.
      :func:`step_count` doubles from that condition on until the estimate
      meets a tolerance, and returns the count.
    * **Cost.**  One eigendecomposition per sample, then two
      ``2^N × 2^N`` matrix products per step and sample.  A backward pass, or
      a forward-mode derivative, reuses the eigendecomposition and adds four
      matrix products per sample: at ``MAX_ATOMS``, forward and backward
      together take about twice as long as the forward pass alone.
    * **Gradients.**  Every operation is differentiable and none is in
      place on a tensor that carries a gradient, so gradients reach the
      tensors ``hamiltonian`` was built from, in reverse mode
      (``backward``, ``torch.func.grad``, ``jacrev``) and in forward mode
      (``torch.func.jvp``, ``jacfwd``, ``torch.autograd.forward_ad``).  The
      propagator is differentiated in the eigenbasis by the Daleckii–Krein
      formula of the module docstring, not through the backward pass of
      ``eigh``, so the derivative is right where eigenvalues coincide
      (#505, #674): two atoms with equal detunings and no interaction are
      enough for that.  It comes from the same eigendecomposition as ``ρ``,
      whose energies are accurate to about ``ε ‖H‖₂``, and is about as
      accurate as ``ρ``: that error only shows when ``‖H‖₂`` is many orders
      above the drive, as for 6 atoms 0.6 µm apart (``‖H‖₂ ≈ 6e8`` rad/µs,
      ``t = 4`` µs), where the populations were ``8e-9`` and the derivative
      ``1.5e-8`` off (#674).  Second derivatives (``gradgradcheck``,
      ``torch.func.hessian``) are correct too, but their extra pass goes
      through ``torch.linalg.matrix_exp`` of four times the size of ``H``.
      ``torch.func.vmap`` over ``evolve`` is not supported: its argument
      checks read values.
    * **Time-dependent pulses** are out of scope beyond piecewise-constant
      ones, which are one call per segment chained through ``rho0``.

    Examples
    --------
    >>> import math
    >>> from hqnn_forge.rydberg import AtomRegister, evolve, readout, rydberg_hamiltonian
    >>> one = AtomRegister([[0.0, 0.0]])
    >>> h = rydberg_hamiltonian(one, 2.0, [0.0], c6=1.0)  # Ω = 2 rad/µs, resonant
    >>> rho = evolve(h, math.pi / 2)  # Ωt = π: a π pulse
    >>> round(readout(rho).item(), 12)
    1.0
    >>> damped = evolve(h, math.pi / 2, gamma=0.2, n_steps=200)
    >>> round(readout(damped).item(), 3)  # 1 − πγ/(8Ω) = 0.961 to first order in γ
    0.962
    """
    n = _atom_count(hamiltonian, "hamiltonian")
    start = None if rho0 is None else _initial_state(rho0, hamiltonian.shape)
    t = _non_negative_number(t, "t")
    gamma = _non_negative_number(gamma, "gamma")
    steps = None if n_steps is None else _step_count(n_steps)
    h = _hermitian(hamiltonian)
    # Asked for before the eigendecomposition, the expensive part of a call.
    if gamma > 0 and steps is None:
        raise ValueError(
            "n_steps is required when gamma > 0: the splitting has an error of order "
            "(t / n_steps)^2, and the step that is small enough depends on gamma and on "
            "the spread E_max - E_min of the energies of the Hamiltonian."
        )

    if gamma == 0 or steps is None:  # no steps only without dephasing, see above
        if start is not None:
            # U ρ(0) U†, exact; broadcast when one ρ(0) is shared by a batch
            unitary = _propagator(h, t)
            return unitary @ start @ unitary.mH
        psi = _Propagator.apply(h, t, True)[0]
        return psi @ psi.mH

    dt = t / steps
    propagator = _propagator(h, dt)
    # d_H(a, b) = Σ_i (a_i − b_i)² = |a| + |b| − 2 a·b, the number of atoms on
    # which a and b differ; small integers, exact in float64.
    bits = _bits(n, h.device)
    weight = bits.sum(dim=-1)
    distance = weight[:, None] + weight[None, :] - 2 * (bits @ bits.T)
    # e^(𝓛_D τ) multiplies ρ_ab by exp(−γ τ d_H(a, b) / 2), for τ = dt/2 and τ = dt.
    half_damping = torch.exp(-gamma * (dt / 2) * distance / 2)
    full_damping = torch.exp(-gamma * dt * distance / 2)

    if start is None:
        rho = torch.zeros_like(h)
        rho[..., 0, 0] = 1.0  # |g…g⟩⟨g…g|
    else:
        rho = start  # the first rotation broadcasts a shared ρ(0) over the batch
    rho = rho * half_damping
    for step in range(steps):
        rho = propagator @ rho @ propagator.mH
        # two adjacent half steps are one full step; the last step ends on a half
        rho = rho * (full_damping if step < steps - 1 else half_damping)
    return rho


def _tolerance(value: Any) -> float:
    """``value`` as a float ``>= _SMALLEST_TOL``, or a ``ValueError`` naming ``tol``."""
    number = math.nan
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:  # an int beyond the float range
            number = math.inf
    if not math.isfinite(number) or number < _SMALLEST_TOL:
        raise ValueError(
            f"tol must be a finite number >= {_SMALLEST_TOL:g}; got {value!r}.  Below an "
            "error of about 1e-10 the solver's rounding, not its step, sets the difference "
            "of two runs, and the estimate says nothing (see evolve, 'Choosing n_steps')."
        )
    return number


def step_count(hamiltonian: torch.Tensor, t: float, *, gamma: float, tol: float) -> int:
    """
    A step count for :func:`evolve` whose estimated feature error is at most ``tol``.

    One integer for the whole batch: the smallest count tried at which the
    estimated error of the features ``⟨n_i⟩`` and ``⟨n_i n_j⟩`` (those of
    :func:`readout` with ``pairs=True``) is at most ``tol`` for every sample.
    The counts tried start at the resolution condition of :func:`evolve`
    ("Choosing ``n_steps``"), not at 1.  ``evolve`` itself is unchanged: the
    caller passes the result to it, or to
    :class:`~hqnn_forge.rydberg.RydbergFeatureMap`, as ``n_steps``.

    **The estimate.**  The splitting's global error is of second order
    (module docstring, *Splitting*): ``S(dt)`` has a logarithm with odd
    powers of ``dt`` only, so ``S(dt)^n = exp(𝓛 t + t dt² C + O(t dt⁴))``
    with ``C`` fixed by ``H`` and γ, and every feature ``F`` comes out of
    ``n`` steps as::

        F_n = F + a / n² + O(n⁻⁴),        a independent of n

    Two runs with ``n`` and ``2n`` steps then differ by::

        F_n − F_2n = a/n² − a/(4n²) = (3/4) a/n² = 3 (F_2n − F)

    to that order, so **the finer run is off by a third of their
    difference**, ``|F_2n − F| ≈ |F_n − F_2n| / 3``.  The estimate at
    ``2n`` steps is the largest of these over the features and the
    samples.  ``F`` is a probability (no unit); ``a`` has the sign of the
    error and drops out of the absolute values.

    **The starting count.**  The expansion holds once ``dt`` resolves the
    fastest rate of the generator, ``W + Nγ/2`` (in rad/µs: ``W = E_max −
    E_min ≥ 0`` the spread of the energies of a sample's ``H``, ``Nγ/2`` the
    largest rate of the dissipator for ``N`` atoms).  Before that it can
    fail outright: one resonant atom at ``ΩT = 16π``, ``γ = 0.1 Ω`` gives
    ``⟨n⟩ = 0`` to rounding at 1, 2, 4, 8 and 16 steps, where the answer is
    0.358, so a doubling from 1 stops at once on two runs that agree while
    both are wrong.  The estimate held with ``dt · (W + Nγ/2) <= 1/2`` for
    the coarser run (#507, #599), so the coarsest run has::

        n₀ = max over samples of ceil(2t · (W + Nγ/2)),    at least 1

    steps (``t`` in µs, so ``2t · (W + Nγ/2)`` is a number), and the counts
    tried are ``2n₀, 4n₀, 8n₀, …``, each with the estimate from it and the
    count before it.  The first whose estimate is at most ``tol`` comes
    back, so the result is ``2^k n₀`` with ``k >= 1``.  It never stops below
    ``2n₀``, although where ``W`` is large (``V`` in the blockade) the
    features can need far fewer steps: no estimate is backed there (#598).

    Parameters
    ----------
    hamiltonian:
        Shape ``(2^N, 2^N)`` or ``(batch, 2^N, 2^N)`` in rad/µs, as for
        :func:`evolve`, with at least one sample.  **Pass the Hamiltonians
        of all inputs at once** (see Notes).
    t:
        Evolution time in µs, ``>= 0``, as for :func:`evolve`.
    gamma:
        Dephasing rate γ in rad/µs, ``>= 0``, as for :func:`evolve`.  At
        ``gamma = 0`` ``evolve`` is exact and does not use ``n_steps``, so
        1 is returned without evolving anything.
    tol:
        The largest estimated error of ``⟨n_i⟩`` and ``⟨n_i n_j⟩`` over the
        batch, a finite number ``>= 1e-9``, without a default (as
        ``n_steps`` has none).  Below an error of about ``1e-10`` rounding
        takes over and two runs say nothing about it.

    Returns
    -------
    int
        The step count, ``2^k n₀`` with ``k >= 1``; 1 when ``gamma = 0``.

    Raises
    ------
    TypeError
        If ``hamiltonian`` is not a real or complex floating tensor.
    ValueError
        If ``hamiltonian`` is refused by :func:`evolve` or holds no sample;
        if ``t`` or ``gamma`` is not a finite number ``>= 0``; if ``tol`` is
        not a finite number ``>= 1e-9``.
    RuntimeError
        If an estimate above ``tol`` is more than half of the one before:
        in the second-order regime it is a quarter (3.91 to 4.12 times
        smaller over the sweep of #598), and a smaller fall means the
        rounding of ``evolve``, which grows with the step count, has taken
        over, so no count reaches ``tol`` for these Hamiltonians.

    Notes
    -----
    * **One value for the whole dataset.**  The count is a maximum over the
      samples, and a sample with a larger ``W`` raises it for all.  Called
      per chunk it would differ between chunkings and break the chunking
      invariance of :class:`~hqnn_forge.rydberg.RydbergFeatureMap` (#499),
      whose features depend on a chunk only through rounding because
      ``n_steps`` is fixed.  Call it once, on the Hamiltonians of all inputs,
      and pass the value to ``RydbergFeatureMap(n_steps=...)``, where
      ``get_config()`` records it.  For a map with a
      :class:`~hqnn_forge.rydberg.PulseEncoding` and no position jitter
      these are ``rydberg_hamiltonian(register, encoding.omega,
      encoding.detunings(X), c6=c6, interactions=interactions)`` at
      ``t = encoding.t`` (example below).  The segments of a
      :class:`~hqnn_forge.rydberg.GlobalDrivePulseEncoding` and the displaced
      registers of position jitter are other Hamiltonians, which a count
      from these does not cover:
      :meth:`hqnn_forge.rydberg.RydbergFeatureMap.step_count` takes the
      count from the map's own chain of segments and registers (#760), and
      gives this function's result where the map has neither.  The pulse
      segments of :class:`~hqnn_forge.encoding.AnalogRydbergLayer` are such
      a chain too: :meth:`~hqnn_forge.encoding.AnalogRydbergLayer.step_count`
      (#769).
    * **Accuracy.**  ``tol`` is met to the accuracy of the estimate, which
      is not a bound.  Over the 216 cases of #598 (1, 2 and 4 atoms, ``V/Ω``
      0 to 100, ``γ/Ω`` 0.1 to 100, ``ΩT`` π to 16π, detunings 0 or in
      ``[0, √3 Ω]``) every count met ``tol = 1e-6``, and the error was 0.987
      to 1.084 times the estimate wherever it was above ``1e-10``; the
      five seeded sweeps of #627 reached 1.64 for single atoms.  A caller
      who needs a margin passes a smaller ``tol``.
    * **Shots.**  The shot estimates of
      :meth:`~hqnn_forge.rydberg.RydbergFeatureMap.transform` are unbiased
      for the exact features at the same count, so their bias is the
      features' error, bounded by ``tol`` as above; their spread is that of
      the shots.
    * **The state.**  ``tol`` is a target for the features, not for ρ or
      :meth:`~hqnn_forge.rydberg.RydbergFeatureMap.states`: at ``2n₀``
      steps, in the cases of the sweep of #598 whose ``n₀`` is at most 3000,
      the largest entry of ρ was off by 0.13 to 20 times the features'
      largest error (#506, #759).
    * **Cost.**  One ``eigvalsh`` per sample for ``W``, then ``k + 1`` runs
      of :func:`evolve` on the batch, at ``n₀, 2n₀, …, 2^k n₀`` steps:
      ``2 · result − n₀`` steps in all, less than twice the cost of one run
      at the result, plus one eigendecomposition per sample and run.  The
      memory is that of one ``evolve`` on the batch.  The counts can be
      large: one atom at ``γ = 100 Ω``, ``ΩT = 16π`` needs 328 192 steps for
      ``tol = 1e-6``.

    Examples
    --------
    >>> import math
    >>> import torch
    >>> from hqnn_forge.rydberg import (
    ...     DEFAULT_C6, AtomRegister, PulseEncoding, RydbergFeatureMap, rydberg_hamiltonian,
    ...     step_count,
    ... )
    >>> omega = 4 * math.pi
    >>> one = rydberg_hamiltonian(AtomRegister([[0.0, 0.0]]), omega, [0.0], c6=1.0)
    >>> step_count(one, math.pi / omega, gamma=omega, tol=1e-6)  # n₀ = ceil(3π) = 10
    320

    One count for a feature map, from all of its inputs:

    >>> register = AtomRegister.chain(3, AtomRegister.blockade_radius(DEFAULT_C6, omega))
    >>> encoding = PulseEncoding(3, omega=omega)
    >>> X = torch.tensor([[-1.0, 0.0, 2.0], [0.5, 0.5, 0.5], [3.0, -2.0, 0.1]])
    >>> h = rydberg_hamiltonian(register, omega, encoding.detunings(X), c6=DEFAULT_C6)
    >>> n = step_count(h, encoding.t, gamma=omega, tol=1e-5)
    >>> feature_map = RydbergFeatureMap(
    ...     register, encoding, c6=DEFAULT_C6, gamma=omega, n_steps=n, correlations=True
    ... )
    >>> feature_map.get_config()["n_steps"] == n
    True
    """
    n_atoms = _atom_count(hamiltonian, "hamiltonian")
    t = _non_negative_number(t, "t")
    gamma = _non_negative_number(gamma, "gamma")
    tol = _tolerance(tol)
    if hamiltonian.ndim == 3 and hamiltonian.shape[0] == 0:
        raise ValueError(
            "hamiltonian must hold at least one sample: the count is fixed by all inputs "
            "at once; got an empty batch."
        )
    h = _hermitian(hamiltonian).detach()
    if gamma == 0:  # exact without steps: any count, so the smallest
        return 1
    with torch.no_grad():
        # W = E_max − E_min per sample; the batch's largest rate sets n₀
        start = _starting_count(_spread(h), t, n_atoms, gamma)
        return _doubling_count(
            start, lambda steps: readout(evolve(h, t, gamma=gamma, n_steps=steps), pairs=True), tol
        )


def _spread(h: torch.Tensor) -> float:
    """``W = E_max − E_min`` of a Hermitian ``h``, the largest over a batch; ``eigvalsh``."""
    energies = torch.linalg.eigvalsh(h)
    return float((energies.amax(dim=-1) - energies.amin(dim=-1)).max().item())


def _starting_count(spread: float, t: float, n_atoms: int, gamma: float) -> int:
    """
    ``n₀ = ceil(2t · (W + Nγ/2))``, at least 1: ``dt · (W + Nγ/2) <= 1/2`` at ``dt = t/n₀``.

    The resolution condition of :func:`step_count` for a constant ``H`` with
    spread ``W`` (rad/µs) evolved for ``t`` µs.
    """
    return max(1, math.ceil(2 * t * (spread + n_atoms * gamma / 2)))


def _doubling_count(start: int, features: Callable[[int], torch.Tensor], tol: float) -> int:
    """
    The doubling of :func:`step_count` from ``start`` steps, for any linear solver.

    ``features(n)`` returns the features at ``n`` steps (one row per sample;
    the maximum is taken over every entry).  Tries ``2·start, 4·start, …``
    and returns the first count at which a third of the largest
    ``|F_(n/2) − F_n|`` is at most ``tol``; ``RuntimeError`` when an estimate
    above ``tol`` is more than half of the one before (``step_count``,
    Raises).  :meth:`~hqnn_forge.rydberg.RydbergFeatureMap.step_count` passes
    its whole chain of segments and registers as ``features``, and
    :meth:`~hqnn_forge.encoding.AnalogRydbergLayer.step_count` the layer's
    chain of pulse segments (#769).
    """
    steps = start
    coarse = features(steps)
    previous = math.inf
    while True:
        fine = features(2 * steps)
        # |F_2n − F| ≈ |F_n − F_2n| / 3, largest over samples and features
        estimate = (coarse - fine).abs().max().item() / 3
        if estimate <= tol:
            return 2 * steps
        if estimate > previous / _SMALLEST_FALL:
            raise RuntimeError(
                f"the two-run estimate stopped falling at {2 * steps} steps "
                f"({previous:.3g}, then {estimate:.3g}) above tol={tol:g}: the solver's "
                "rounding, which grows with the step count, has taken over, and no "
                "count reaches this tol for these Hamiltonians."
            )
        steps, coarse, previous = 2 * steps, fine, estimate


def _propagator(h: torch.Tensor, tau: float) -> torch.Tensor:
    """
    ``U(τ) = e^(−iHτ)`` of a Hermitian ``complex128`` ``h``, batched over leading axes.

    The unitary step of :func:`evolve`, and the per-segment propagator of the
    analog encoding layer (:mod:`hqnn_forge.encoding.analog_rydberg`), so a
    piecewise-constant pulse is propagated exactly as one constant segment
    is.  ``h`` in rad/µs, ``τ`` in µs.  The caller has checked that ``h`` is
    Hermitian (``evolve`` symmetrises it; the analog layer builds it with
    :func:`~hqnn_forge.rydberg.rydberg_hamiltonian`, Hermitian by
    construction).

    ``U = V e^(−iEτ) V†`` from ``torch.linalg.eigh``, through
    :class:`_Propagator`, which differentiates it with the Fréchet derivative
    in the eigenbasis rather than through ``eigh``; reverse and forward mode
    (``torch.func.jvp``, ``jacfwd``) are both correct at a degenerate spectrum
    (#505, #674).
    """
    return _Propagator.apply(h, tau, False)[0]


def _populations_to_features(
    populations: torch.Tensor,
    *,
    pairs: bool = False,
) -> torch.Tensor:
    """
    Features (atoms and optionally pairs) from computational-basis populations.

    The caller is expected to have validated the inputs: a real floating or
    integer tensor of shape ``(2^N,)`` or ``(batch, 2^N)`` for 1 to
    ``MAX_ATOMS`` atoms.

    Parameters
    ----------
    populations:
        Tensor of shape ``(2^N,)`` or ``(batch, 2^N)``, with basis states ordered
        by atom 0 as the most significant bit.
    pairs:
        Whether to append the ``N(N-1)/2`` pair correlations.

    Returns
    -------
    torch.Tensor
        Features tensor of shape ``(N,)``, ``(batch, N)``, ``(N + N(N-1)/2,)``,
        or ``(batch, N + N(N-1)/2)`` with the same dtype and on the same device
        as ``populations``.
    """
    n = populations.shape[-1].bit_length() - 1
    bits = _bits(n, populations.device).to(populations.dtype)
    singles = populations @ bits
    if not pairs:
        return singles
    # Row-major upper triangle: (0,1), (0,2), …, (0,N−1), (1,2), …, (N−2,N−1).
    first, second = torch.triu_indices(n, n, offset=1, device=populations.device)
    both = bits[:, first] * bits[:, second]
    return torch.cat([singles, populations @ both], dim=-1)


def readout(rho: torch.Tensor, *, pairs: bool = False) -> torch.Tensor:
    """
    Excitation probabilities ``⟨n_i⟩`` of every atom, and optionally ``⟨n_i n_j⟩``.

    With ``b_i(k) = (k >> (N−1−i)) & 1`` the bit of atom ``i`` in the index
    ``k`` of a basis state (atom 0 the most significant bit)::

        ⟨n_i⟩     = Tr[ρ n_i]     = Σ_k b_i(k) ρ_kk
        ⟨n_i n_j⟩ = Tr[ρ n_i n_j] = Σ_k b_i(k) b_j(k) ρ_kk

    Only the diagonal of ρ is read, the outcome probabilities of measuring
    every atom in the ``(|g⟩, |r⟩)`` basis.  These are exact expectation
    values, not estimates from shots.

    Parameters
    ----------
    rho:
        Density matrix of shape ``(2^N, 2^N)``, or ``(batch, 2^N, 2^N)``,
        e.g. from :func:`evolve`.  Real or complex; the imaginary part of the
        diagonal, zero for a Hermitian matrix, is dropped.  It is not checked
        to be a density matrix.
    pairs:
        Also return the pair correlations ``⟨n_i n_j⟩`` for ``i < j``.

    Returns
    -------
    torch.Tensor
        ``float64``, on the device of ``rho``, with the atoms (and pairs) on
        the last axis: shape ``(N,)`` or ``(batch, N)`` holding
        ``⟨n_0⟩, …, ⟨n_{N−1}⟩``.  With ``pairs=True`` the ``N(N−1)/2`` pair
        correlations are appended in the order
        ``(0,1), (0,2), …, (0,N−1), (1,2), …, (N−2,N−1)``, which gives
        ``N + N(N−1)/2`` values; one atom has no pair.

    Raises
    ------
    TypeError
        If ``rho`` is not a real or complex floating tensor.
    ValueError
        If ``rho`` has another shape, is for more than
        :data:`~hqnn_forge.rydberg.MAX_ATOMS` atoms, or holds a NaN or an
        infinity.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.rydberg import readout
    >>> rho = torch.diag(torch.tensor([0.125, 0.125, 0.25, 0.5]))  # |gg>, |gr>, |rg>, |rr>
    >>> readout(rho).tolist()  # atom 0 is excited in |rg> and |rr>
    [0.75, 0.625]
    >>> readout(rho, pairs=True).tolist()
    [0.75, 0.625, 0.5]
    """
    _atom_count(rho, "rho")
    diagonal = rho.diagonal(dim1=-2, dim2=-1)
    populations = (diagonal.real if diagonal.is_complex() else diagonal).to(torch.float64)
    return _populations_to_features(populations, pairs=pairs)
