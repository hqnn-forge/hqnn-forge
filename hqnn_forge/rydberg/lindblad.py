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
``ρ(0) = |g…g⟩⟨g…g|``.  There is no global variant of the operator and no
decay term.

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

The sign is that of the Schrödinger equation ``i dψ/dt = Hψ``.  For one
resonant atom it gives ``ψ(t) = (cos(Ωt/2), −i sin(Ωt/2))`` in the basis
``(|g⟩, |r⟩)``, so ``⟨n⟩(t) = sin²(Ωt/2)`` and ``ρ_gr = +(i/2) sin(Ωt)``.

Without dephasing the state stays pure, ``ρ(t) = ψ(t) ψ(t)†`` with::

    ψ(t) = U(t) e_0,        ψ_k(t) = Σ_m V_km e^(−i E_m t) conj(V_0m)

``e_0`` being the basis vector of ``|g…g⟩``.  This is exact at any ``t``.

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
  ``[𝓛_D, [𝓛_D, 𝓛_H]]`` and ``[𝓛_H, [𝓛_H, 𝓛_D]]``, of order ``γ² ‖H‖`` and
  ``γ ‖H‖²``.  ``dt`` therefore has to resolve both ``1/γ`` and the largest
  level spacing of ``H``, which in the blockade is the interaction ``V``,
  not Ω.  It vanishes when ``H`` is diagonal (no drive): ``𝓛_H`` then also
  acts elementwise, the two parts commute and one step is exact.
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

Readout
-------
``⟨n_i⟩ = Tr[ρ n_i] = Σ_k b_i(k) ρ_kk`` and
``⟨n_i n_j⟩ = Σ_k b_i(k) b_j(k) ρ_kk``: both are sums over the diagonal of ρ,
the outcome probabilities of measuring every atom in the ``(|g⟩, |r⟩)`` basis.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import torch

from hqnn_forge.rydberg.hamiltonian import MAX_ATOMS

__all__ = ["evolve", "readout"]

#: Largest ``|H − H†|`` entry :func:`evolve` accepts, relative to the largest
#: ``|H|`` entry.  A matrix built in ``float64`` from a Hermitian expression is
#: Hermitian to a few units of 1e-16.
_HERMITIAN_RTOL = 1e-12


def _non_negative_number(value: Any, name: str) -> float:
    """``value`` as a float ``>= 0``, or a ``ValueError`` naming the argument."""
    number = (
        float(value)
        if isinstance(value, numbers.Real) and not isinstance(value, bool)
        else math.nan
    )
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite number >= 0; got {value!r}.")
    return number


def _step_count(n_steps: Any) -> int:
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


def _bits(n: int, device: torch.device) -> torch.Tensor:
    """``bits[k, i] = b_i(k) = (k >> (n−1−i)) & 1`` as ``float64``, shape ``(2^n, n)``."""
    index = torch.arange(2**n, device=device)
    shifts = torch.arange(n - 1, -1, -1, device=device)
    return ((index[:, None] >> shifts) & 1).to(torch.float64)


def evolve(
    hamiltonian: torch.Tensor,
    t: float,
    *,
    gamma: float = 0.0,
    n_steps: int | None = None,
) -> torch.Tensor:
    """
    The state ``ρ(t)`` of an array that starts with every atom in ``|g⟩``.

    Solves, for one or a batch of time-independent Hamiltonians::

        dρ/dt = −i [H, ρ] + γ Σ_i ( n_i ρ n_i − ½ { n_i , ρ } ),    ρ(0) = |g…g⟩⟨g…g|

    the Lindblad equation of ``docs/rydberg-model.md`` with the local
    collapse operators ``L_i = √γ n_i``.  The module docstring derives every
    formula used here, with the sign and ordering conventions.

    * ``gamma = 0``: ``ρ(t) = ψψ†`` with ``ψ = e^(−iHt) |g…g⟩``, exact, from
      one eigendecomposition per sample and without time stepping.
    * ``gamma > 0``: ``n_steps`` steps of ``dt = t / n_steps`` of the
      symmetric (Strang) splitting
      ``e^(𝓛_D dt/2) · e^(𝓛_H dt) · e^(𝓛_D dt/2)``.  The unitary step is
      ``ρ → U ρ U†`` with ``U = e^(−iH dt)`` from the same
      eigendecomposition; the dissipative step multiplies ``ρ_ab`` by
      ``exp(−γ τ d_H(a, b) / 2)``, ``d_H`` the Hamming distance of the basis
      states ``a`` and ``b``.  Both are exact; the splitting has a **global
      error of second order in ``dt``**.

    Parameters
    ----------
    hamiltonian:
        Shape ``(2^N, 2^N)``, or ``(batch, 2^N, 2^N)`` for a batch, in
        rad/µs, e.g. from :func:`~hqnn_forge.rydberg.rydberg_hamiltonian`;
        ``N`` may be 1 to :data:`~hqnn_forge.rydberg.MAX_ATOMS`.  Hermitian,
        real or complex.  Its basis ordering defines that of the result:
        index 0 is the initial state.
    t:
        Evolution time in µs, ``>= 0``, the same for every sample.
    gamma:
        Dephasing rate γ in rad/µs, ``>= 0``, the same for every atom and
        sample.  It is the rate of the collapse operators ``√γ n_i``: **a
        single-atom coherence decays at ``γ/2``** (``T₂ = 2/γ``), and
        ``ρ_ab`` at ``γ/2 · d_H(a, b)``.
    n_steps:
        Number of splitting steps, an integer ``>= 1``.  Required when
        ``gamma > 0`` and without a default: the step that is accurate enough
        depends on γ and on the level spacings of ``H`` (the interaction, in
        the blockade), which this function does not guess.  Not used when
        ``gamma = 0``, where the result is exact.

    Returns
    -------
    torch.Tensor
        ``ρ(t)``, ``complex128``, with the shape and device of
        ``hamiltonian``.  Entry ``[..., a, b]`` is ``⟨a|ρ|b⟩``.  Trace 1,
        Hermitian and positive semidefinite up to rounding, at any step size;
        the rounding error of the trace grows by a few ``1e-16`` per step.

    Raises
    ------
    TypeError
        If ``hamiltonian`` is not a real or complex floating tensor.
    ValueError
        If ``hamiltonian`` has another shape, is for more than
        :data:`~hqnn_forge.rydberg.MAX_ATOMS` atoms, holds a NaN or an
        infinity, or is not Hermitian; if ``t`` or ``gamma`` is not a finite
        number ``>= 0``; if ``n_steps`` is not an integer ``>= 1``, or is
        missing with ``gamma > 0``.

    Notes
    -----
    * **Precision.**  The Hamiltonian is converted to ``complex128`` whatever
      its dtype.
    * **Choosing ``n_steps``.**  Halving ``dt`` divides the error by 4, so
      two runs with ``n_steps`` and ``2 · n_steps`` differ by about 3/4 of
      the coarser run's error.  ``dt`` must be small against ``1/γ`` and
      against the inverse of the largest level spacing of ``H``.
    * **Cost.**  One eigendecomposition per sample, then two
      ``2^N × 2^N`` matrix products per step and sample.
    * **Gradients.**  Every operation is differentiable and none is in
      place, so gradients reach the tensors ``hamiltonian`` was built from,
      and they match finite differences where the spectrum of ``H`` is not
      degenerate.  They pass through ``torch.linalg.eigh``, whose backward
      pass holds ``1/(E_m − E_k)``: **where two eigenvalues coincide the
      gradient is wrong without any error**, finite but not the derivative
      (#505).  Two atoms with equal detunings and no interaction are enough.
      Nothing here needs a gradient; the features are precomputed.
    * **Time-dependent pulses** are out of scope.  A piecewise-constant one
      would repeat the loop below per segment, starting from the previous
      segment's state.

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
    t = _non_negative_number(t, "t")
    gamma = _non_negative_number(gamma, "gamma")
    steps = None if n_steps is None else _step_count(n_steps)
    h = hamiltonian.to(torch.complex128)
    # eigh reads one triangle only: a non-Hermitian matrix would be evolved as
    # another, Hermitian one without any error.
    scale = float(h.detach().abs().max())
    asymmetry = float((h - h.mH).detach().abs().max())
    if asymmetry > _HERMITIAN_RTOL * scale:
        raise ValueError(
            "hamiltonian must be Hermitian; its largest |H - H†| entry is "
            f"{asymmetry:.3g} (largest |H| entry {scale:.3g})."
        )

    # H = V E V†: energies[..., m] = E_m, vectors[..., k, m] = V_km.
    energies, vectors = torch.linalg.eigh(h)

    if gamma == 0:
        # ψ_k(t) = Σ_m V_km e^(−i E_m t) conj(V_0m); row 0 of V is ⟨g…g|m⟩.
        amplitudes = torch.exp(-1j * energies * t) * vectors[..., 0, :].conj()
        psi = vectors @ amplitudes[..., :, None]
        return psi @ psi.mH

    if steps is None:
        raise ValueError(
            "n_steps is required when gamma > 0: the splitting has an error of order "
            "(t / n_steps)^2, and the step that is small enough depends on gamma and on "
            "the level spacings of the Hamiltonian."
        )
    dt = t / steps
    # U(dt) = V e^(−i E dt) V†: column m of V scaled by its phase, times V†.
    propagator = (vectors * torch.exp(-1j * energies * dt)[..., None, :]) @ vectors.mH
    # d_H(a, b) = Σ_i (a_i − b_i)², the number of atoms on which a and b differ.
    bits = _bits(n, h.device)
    distance = ((bits[:, None, :] - bits[None, :, :]) ** 2).sum(dim=-1)
    # e^(𝓛_D τ) multiplies ρ_ab by exp(−γ τ d_H(a, b) / 2), for τ = dt/2 and τ = dt.
    half_damping = torch.exp(-gamma * (dt / 2) * distance / 2)
    full_damping = torch.exp(-gamma * dt * distance / 2)

    rho = torch.zeros_like(h)
    rho[..., 0, 0] = 1.0  # |g…g⟩⟨g…g|
    rho = rho * half_damping
    for step in range(steps):
        rho = propagator @ rho @ propagator.mH
        # two adjacent half steps are one full step; the last step ends on a half
        rho = rho * (full_damping if step < steps - 1 else half_damping)
    return rho


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
    n = _atom_count(rho, "rho")
    diagonal = rho.diagonal(dim1=-2, dim2=-1)
    populations = (diagonal.real if diagonal.is_complex() else diagonal).to(torch.float64)
    bits = _bits(n, rho.device)
    singles = populations @ bits  # Σ_k ρ_kk b_i(k)
    if not pairs:
        return singles
    # Row-major upper triangle: (0,1), (0,2), …, (0,N−1), (1,2), …, (N−2,N−1).
    first, second = torch.triu_indices(n, n, offset=1, device=rho.device)
    both = bits[:, first] * bits[:, second]  # b_i(k) b_j(k), shape (2^N, N(N−1)/2)
    return torch.cat([singles, populations @ both], dim=-1)
