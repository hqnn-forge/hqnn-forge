"""
hqnn_forge.rydberg.hamiltonian
==============================
The Hamiltonian of ``docs/rydberg-model.md`` as a dense matrix.

Each atom is a two-level system of its ground state ``|g⟩ = |0⟩`` and one
Rydberg state ``|r⟩ = |1⟩``.  In the frame rotating at the laser frequency,
with the rotating-wave approximation and ħ = 1::

    H = Σ_i Ω_i/2 · X_i  −  Σ_i Δ_i n_i  +  Σ_{i<j} V_ij n_i n_j

    n_i = |r⟩⟨r|_i = (1 − Z_i)/2        V_ij = C6 / |r_i − r_j|⁶

Units: angular frequencies (Ω, Δ, V and ``H`` itself) in rad/µs, lengths in
µm, ``C6`` in rad/µs · µm⁶.

Sign conventions
----------------
* **Ω is the Rabi frequency, not half of it.**  The coefficient of ``X`` is
  ``Ω/2``: a resonant atom has ``⟨n⟩(t) = sin²(Ωt/2)``, and its two levels are
  split by Ω (eigenvalues ``±Ω/2``).
* **Δ is the laser frequency minus the atomic transition frequency.**  The
  atom's energy ``ω₀ n`` minus the frame's ``ω_L n`` is ``−Δ n``: a positive Δ
  lowers ``|r⟩``.
* **A positive C6 is repulsive.**  It raises a doubly excited pair by
  ``V_ij``, so two atoms at distance ``r`` have
  ``⟨rr|H|rr⟩ = −Δ_0 − Δ_1 + C6/r⁶``.

For one atom, in the basis ``(|g⟩, |r⟩)``, ``H = [[0, Ω/2], [Ω/2, −Δ]]``.

Basis ordering
--------------
Atom ``i`` (counted from 0, the row of the register) is tensor factor ``i``
from the left and PennyLane wire ``i``.  The basis state
``|b_0 b_1 … b_{N−1}⟩``, with ``b_i = 1`` for ``|r⟩``, has the index::

    k = Σ_i b_i 2^(N−1−i)        ⇔        b_i(k) = (k >> (N−1−i)) & 1

so atom 0 is the most significant bit.  For two atoms the order is
``|gg⟩, |gr⟩, |rg⟩, |rr⟩`` and ``n_0 = diag(0, 0, 1, 1)``.  This is the matrix
``qml.matrix(op, wire_order=range(N))`` returns for the same operator.

Matrix elements
---------------
In that basis ``n_i`` is diagonal with entry ``b_i(k)``, and ``X_i`` flips bit
``N−1−i`` of the index.  Term by term:

* detuning: ``−Σ_i Δ_i b_i(k)`` on the diagonal;
* interaction: ``+Σ_{i<j} V_ij b_i(k) b_j(k)`` on the diagonal;
* drive: ``Ω_i/2`` at ``(k, k XOR 2^(N−1−i))`` for every ``k`` and ``i``.  No
  two of these coincide (different ``i`` flip different bits) and none is on
  the diagonal, so every row holds exactly ``N`` off-diagonal entries.

With the default drive phase ``φ = 0`` the matrix is real and symmetric.  It
is returned as ``complex128`` because what consumes it (``exp(−iHt)``, a
density matrix) is complex, and because a drive phase makes it complex: with
``φ_i`` the drive entry is ``Ω_i/2 · e^{iφ_i}`` where atom ``i`` is in ``|g⟩``
in the row state and ``Ω_i/2 · e^{−iφ_i}`` where it is in ``|r⟩`` (derived in
the Notes of :func:`rydberg_hamiltonian`), so the matrix stays Hermitian.

References
----------
* Bernien et al. (2017) "Probing many-body dynamics on a 51-atom quantum
  simulator", Nature 551, 579–584.
* Wurtz et al. (2023) "Aquila: QuEra's 256-qubit neutral-atom quantum
  computer", arXiv:2306.11727.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from hqnn_forge.rydberg.register import AtomRegister, _finite_number, _real_tensor

__all__ = ["DEFAULT_C6", "MAX_ATOMS", "rydberg_hamiltonian"]

#: ``C6`` of two ⁸⁷Rb atoms in ``70S₁/₂``, in rad/µs · µm⁶:
#: ``2π × 862 690 MHz µm⁶ = 5 420 441``, positive (repulsive).  It is the
#: value fixed in ``docs/rydberg-model.md`` and, up to PennyLane 0.45, the
#: default of ``qml.pulse.rydberg_interaction`` (which takes it in MHz µm⁶;
#: 0.46 removes ``qml.pulse``).  No function here defaults to it: ``c6`` is
#: always passed, so the atom is stated at the call.
DEFAULT_C6: float = 2 * math.pi * 862690.0

#: Largest register :func:`rydberg_hamiltonian` builds a matrix for.  ``N``
#: atoms have ``2^N`` basis states, so one Hamiltonian, like one density
#: matrix, holds ``4^N`` ``complex128`` entries, ``16 · 4^N`` bytes: 16 MiB at
#: 10 atoms, and 4 times more for each further atom (1 GiB at 13), per sample
#: of a batch.  The model of ``docs/rydberg-model.md`` uses 4 to 6 atoms.
MAX_ATOMS: int = 10


def _finite_tensor(value: Any, name: str) -> torch.Tensor:
    """``value`` as a finite ``float64`` tensor, keeping its device and graph."""
    tensor = _real_tensor(value, name)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite; got a NaN or an infinity.")
    return tensor


def _bits(n: int, device: torch.device) -> torch.Tensor:
    """
    ``bits[k, i] = b_i(k) = (k >> (n−1−i)) & 1`` as ``float64``, shape ``(2^n, n)``.

    Atom 0 is the most significant bit of the basis index.  The Hamiltonian,
    the solver and the readout all take the ordering from this one table.
    """
    index = torch.arange(2**n, device=device)
    shifts = torch.arange(n - 1, -1, -1, device=device)
    return ((index[:, None] >> shifts) & 1).to(torch.float64)


def rydberg_hamiltonian(
    register: AtomRegister,
    omega: Any,
    delta: Any,
    *,
    c6: float,
    interactions: bool = True,
    phase: Any = 0.0,
) -> torch.Tensor:
    """
    The dense Rydberg Hamiltonian of a register, for one or a batch of pulses.

    ::

        H = Σ_i Ω_i/2 · (cos φ_i X_i − sin φ_i Y_i)  −  Σ_i Δ_i n_i
            +  Σ_{i<j} C6/|r_i − r_j|⁶ · n_i n_j

    with ``n_i = |r⟩⟨r|_i``, in rad/µs, and the drive phase ``φ_i = 0`` by
    default, which leaves ``Ω_i/2 · X_i``.  The derivation of every matrix
    element, the sign conventions and the basis ordering (atom ``i`` is tensor
    factor and PennyLane wire ``i``, atom 0 the most significant bit of the
    index) are in the module docstring.

    Parameters
    ----------
    register:
        The atoms' positions.  ``N = register.n_atoms`` may be 1 to
        :data:`MAX_ATOMS`.
    omega:
        Rabi frequency Ω in rad/µs: a scalar (one global drive), or one value
        per atom with the atoms on the **last** axis.  Accepted shapes are
        ``()``, ``(N,)``, ``(batch, N)`` and ``(batch, 1)``, the last being
        one global Ω per sample.  A one-dimensional ``omega`` is always read
        along the atoms, never along the batch, so it has length ``N``: shape
        ``(batch,)`` raises unless ``batch == N``, also for a batch of 1.  The
        sign is not restricted; a negative Ω is a drive of phase π.
    delta:
        Detuning Δ_i in rad/µs, one per atom: shape ``(N,)``, or
        ``(batch, N)`` for a batch.
    c6:
        The ``C6`` coefficient in rad/µs · µm⁶, e.g. :data:`DEFAULT_C6`.
        Required, and checked, also with ``interactions=False``, so that the
        control is the same call with one flag changed.
    interactions:
        ``False`` drops the term ``Σ_{i<j} V_ij n_i n_j`` and nothing else.
        The result is then the Kronecker sum of the single-atom Hamiltonians
        ``[[0, Ω_i/2], [Ω_i/2, −Δ_i]]``: the non-interacting control
        (``V_ij = 0`` exactly, not a large spacing).
    phase:
        Drive phase φ in radians, default 0: a scalar or one value per atom,
        with the shapes accepted for ``omega`` (``()``, ``(N,)``,
        ``(batch, N)`` or ``(batch, 1)``).  The drive of atom ``i`` becomes
        ``Ω_i/2 · (cos φ_i X_i − sin φ_i Y_i) = Ω_i/2 · (e^{iφ_i} |g⟩⟨r|_i
        + e^{−iφ_i} |r⟩⟨g|_i)``, the form of ``docs/rydberg-model.md`` and of
        the trainable analog layer (#420); see "Drive phase" in the Notes.

    Returns
    -------
    torch.Tensor
        ``complex128``, on the device of ``delta``.  Shape ``(2^N, 2^N)`` if
        neither ``omega`` nor ``delta`` has a batch axis, else
        ``(batch, 2^N, 2^N)`` with sample ``b`` built from ``omega[b]`` and
        ``delta[b]``; an argument without a batch axis is shared by all
        samples.  The imaginary part is zero when ``phase`` is 0.

    Raises
    ------
    TypeError
        If ``register`` is not an :class:`~hqnn_forge.rydberg.AtomRegister`,
        or ``omega``, ``delta`` or ``phase`` is complex or boolean.
    ValueError
        If the register has more than :data:`MAX_ATOMS` atoms; if ``delta``,
        ``omega`` or ``phase`` has another shape than those above or their
        batch sizes differ (a batch of 1 is not broadcast to another batch
        size); if any of them holds a NaN or an infinity; or if ``c6`` is not
        a finite number.

    Notes
    -----
    * **Precision.**  ``omega`` and ``delta`` are converted to ``float64``
      whatever their dtype, and the result is ``complex128``.
    * **Gradients.**  The matrix is linear in ``omega`` and ``delta`` and
      built with differentiable operations: ``∂H/∂Δ_i = −n_i``,
      ``∂H/∂Ω_i = (cos φ_i X_i − sin φ_i Y_i)/2`` and
      ``∂H/∂φ_i = −Ω_i/2 · (sin φ_i X_i + cos φ_i Y_i)`` reach tensors that
      require them.  The positions and ``c6`` are constants.
    * **Batched and unbatched calls agree bit for bit.**  Each diagonal entry
      is accumulated atom by atom in the same order either way, so sample
      ``b`` of a batched call equals the unbatched call on ``omega[b]`` and
      ``delta[b]`` exactly.
    * **Drive phase.**  With ``X = |g⟩⟨r| + |r⟩⟨g|`` and
      ``Y = −i |g⟩⟨r| + i |r⟩⟨g|`` in the basis ``(|g⟩, |r⟩)``::

          cos φ X − sin φ Y = (cos φ + i sin φ) |g⟩⟨r| + (cos φ − i sin φ) |r⟩⟨g|
                            = e^{iφ} |g⟩⟨r| + e^{−iφ} |r⟩⟨g|

      so the drive entry at ``(k, k XOR 2^(N−1−i))`` is ``Ω_i/2 · e^{iφ_i}``
      where atom ``i`` is in ``|g⟩`` in the row state ``k`` (``b_i(k) = 0``)
      and its complex conjugate where it is in ``|r⟩``.  The two entries of a
      pair are built as one value and its conjugate, so the matrix is
      Hermitian exactly.  ``φ`` and ``φ + 2π`` give the same matrix; a
      negative Ω is the phase ``φ + π``.  At ``φ = 0`` the factor is exactly
      ``1 + 0i`` and the matrix is bit for bit that of a call without
      ``phase``.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.rydberg import AtomRegister, rydberg_hamiltonian
    >>> one = AtomRegister([[0.0, 0.0]])
    >>> rydberg_hamiltonian(one, 2.0, [0.5], c6=1.0).real.tolist()
    [[0.0, 1.0], [1.0, -0.5]]
    >>> pair = AtomRegister.chain(2, spacing=2.0)  # V = C6 / 2^6
    >>> h = rydberg_hamiltonian(pair, 2.0, [0.5, 0.25], c6=64.0)
    >>> h.real.diagonal().tolist()  # |gg>, |gr>, |rg>, |rr>
    [0.0, -0.25, -0.5, 0.25]
    >>> rydberg_hamiltonian(pair, 2.0, torch.zeros(8, 2), c6=64.0).shape
    torch.Size([8, 4, 4])
    """
    if not isinstance(register, AtomRegister):
        raise TypeError(f"register must be an AtomRegister; got {type(register).__name__}.")
    n = register.n_atoms
    if n > MAX_ATOMS:
        raise ValueError(
            f"rydberg_hamiltonian builds a dense 2^n x 2^n matrix and accepts at most "
            f"{MAX_ATOMS} atoms; got {n} (2^{n} states, 4^{n} matrix entries per sample)."
        )
    c6 = _finite_number(c6, "c6")
    delta = _finite_tensor(delta, "delta")
    omega = _finite_tensor(omega, "omega")
    phase = _finite_tensor(phase, "phase")
    if delta.ndim not in (1, 2) or delta.shape[-1] != n:
        raise ValueError(
            f"delta must have shape ({n},) or (batch, {n}), one detuning per atom; "
            f"got shape {tuple(delta.shape)}."
        )
    # The accepted shapes are checked one by one, not by broadcasting: that
    # would also take an omega of shape (1,) for a global one on N > 1 atoms,
    # and a batch of 1 for any other batch size.
    batch = {int(tensor.shape[0]) for tensor in (omega, delta, phase) if tensor.ndim == 2}
    for name, tensor in (("omega", omega), ("phase", phase)):
        atoms_last = (
            tensor.ndim == 0
            or (tensor.ndim == 1 and tensor.shape[0] == n)
            or (tensor.ndim == 2 and tensor.shape[1] in (1, n))
        )
        if not atoms_last or len(batch) > 1:
            raise ValueError(
                f"{name} must be a scalar or have its atoms on the last axis, with shape "
                f"({n},), (batch, {n}) or (batch, 1), and match the batch of delta "
                f"(shape {tuple(delta.shape)}); got shape {tuple(tensor.shape)}. "
                f"One global {name} per sample has shape (batch, 1)."
            )
    # All now (*batch, n) with the atoms last; *batch is () or (batch,).
    shape = (*batch, n)
    omega = omega.to(delta.device).expand(shape)
    phase = phase.to(delta.device).expand(shape)
    delta = delta.expand(shape)

    dim = 2**n
    index = torch.arange(dim, device=delta.device)
    bits = _bits(n, delta.device)

    # Detuning: -Σ_i Δ_i b_i(k), accumulated atom by atom.
    diagonal = torch.zeros(*shape[:-1], dim, dtype=torch.float64, device=delta.device)
    for i in range(n):
        diagonal = diagonal - delta[..., i, None] * bits[:, i]
    if interactions:
        # Interaction: +Σ_{i<j} V_ij b_i(k) b_j(k); the strict upper triangle is i < j.
        v = torch.triu(register.interaction_matrix(c6), diagonal=1).to(delta.device)
        diagonal = diagonal + ((bits @ v) * bits).sum(dim=-1)

    h = torch.diag_embed(diagonal.to(torch.complex128))
    # Drive: Ω_i/2 · e^{iφ_i} at (k, k XOR 2^(n-1-i)), the index with atom i's
    # bit flipped, where b_i(k) = 0 (|g⟩ in the row), and its conjugate where
    # b_i(k) = 1.  e^{iφ} = cos φ + i sin φ, exactly 1 + 0i at φ = 0.  The
    # conjugate of a zero imaginary part is −0.0; adding 0 turns it back into
    # +0.0, so at φ = 0 even the sign bits are those of the real matrix.
    up = (omega / 2) * torch.polar(torch.ones_like(phase), phase)
    down = up.conj() + 0.0
    for i in range(n):
        h[..., index, index ^ (1 << (n - 1 - i))] = torch.where(
            bits[:, i] == 0, up[..., i, None], down[..., i, None]
        )
    return h
