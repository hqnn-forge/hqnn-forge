"""
hqnn_forge.rydberg.feature_map
==============================
The fixed feature map of ``docs/rydberg-model.md``: a batch of classical
inputs goes in, a matrix of excitation probabilities comes out, and every
physical setting is recorded.

Units: ħ = 1, angular frequencies (Ω, Δ, γ, ``C6/r⁶``) in rad/µs, lengths in
µm, times in µs.  The inputs and the features are dimensionless.

The map, step by step
---------------------
For one sample ``x ∈ ℝ^N`` and a register of ``N`` atoms::

    x_i  →  Δ_i = Δ_max · σ(x_i)                           (PulseEncoding)
         →  H = Ω/2 Σ_i X_i − Σ_i Δ_i n_i + Σ_{i<j} V_ij n_i n_j   (rydberg_hamiltonian)
         →  ρ(T), from every atom in |g⟩, with dephasing γ   (evolve)
         →  F_i = ⟨n_i⟩ = Tr[ρ(T) n_i]                       (readout)

* **Which input feeds which atom.**  Column ``i`` of ``X`` sets the detuning
  of atom ``i``, the atom at row ``i`` of the register's positions, which is
  tensor factor (PennyLane wire) ``i`` of the Hamiltonian and feature column
  ``i`` of the result.  There is one input per atom, so the input width
  equals the number of atoms.
* **What carries data.**  Only the detunings.  Ω, ``T``, the positions,
  ``C6`` and γ are fixed controls, the same for every sample.
* **Sign of Δ.**  Δ is the laser frequency minus the transition frequency
  and enters ``H`` as ``−Δ n``; the encoding only produces ``Δ_i > 0``.

Encoding
--------
::

    Δ_i(x_i) = Δ_max · σ(x_i),        σ(x) = 1 / (1 + e^(−x))

σ is strictly increasing from 0 (``x → −∞``) to 1 (``x → +∞``), with
``σ(0) = ½``, so ``Δ_i`` lies in the open interval ``(0, Δ_max)`` for every
finite input.  The bound is enforced by this squashing function and not by
clipping, which would send every input beyond the range to the same
detuning.  There is no scale parameter: scaling the inputs is preprocessing.

The single-atom feature
-----------------------
Without interactions and dephasing each atom evolves alone under
``H_1 = [[0, Ω/2], [Ω/2, −Δ]]`` in the basis ``(|g⟩, |r⟩)``.  With
``n = (1 − Z)/2``::

    H_1 = −(Δ/2) · 1 + ½ (Ω X + Δ Z),        (Ω X + Δ Z)² = (Ω² + Δ²) · 1 = Ω_eff² · 1

so, with ``Ω_eff = √(Ω² + Δ²)``::

    e^(−i H_1 T) = e^(iΔT/2) [ cos(Ω_eff T/2) · 1 − i sin(Ω_eff T/2) · (Ω X + Δ Z)/Ω_eff ]

    ⟨r| e^(−i H_1 T) |g⟩ = −i e^(iΔT/2) · (Ω/Ω_eff) · sin(Ω_eff T/2)

    ⟨n⟩(T) = Ω²/Ω_eff² · sin²(Ω_eff T/2)

With the default pulse ``ΩT = π`` and ``s = Ω_eff/Ω = √(1 + (Δ/Ω)²)`` this is
``f(s) = sin²(π s/2) / s²``.  The default bound ``Δ_max = √3 Ω`` is the
detuning with ``s = 2``, the first zero of ``f``: ``f`` falls strictly from 1
at ``Δ = 0`` to 0 at ``Δ_max``, so the feature

::

    F_i = f( √(1 + 3 σ(x_i)²) )            (interactions off, γ = 0, default pulse)

is strictly decreasing in ``x_i`` and depends on no other input.  With
``γ > 0`` and no interactions the state is still a product of single-atom
states, and each feature solves, for its own ``Δ_i``, with ``p = ρ_rr`` and
``c = ρ_gr``::

    dp/dt = Ω · Im c,        dc/dt = −(γ/2 + iΔ) c − i (Ω/2)(2p − 1)

Readout and its order
---------------------
The features are ``⟨n_0⟩, …, ⟨n_{N−1}⟩``, in the order of the atoms.  With
``correlations=True`` the pair correlations ``⟨n_i n_j⟩``, ``i < j``, are
appended in the order ``(0,1), (0,2), …, (0,N−1), (1,2), …, (N−2,N−1)``,
which gives ``N + N(N−1)/2`` features.  Both are sums over the diagonal of ρ
(:func:`~hqnn_forge.rydberg.readout`).

Shots
-----
By default the features are exact expectation values.  With ``shots=S`` they
are estimated as a device would: ``S`` bitstrings ``b^(1), …, b^(S)`` are
drawn for each sample, each with probability ``ρ_kk`` for the basis state
``k`` (bit ``b_i = 1`` when atom ``i`` is found in ``|r⟩``), and::

    F̂_i = (1/S) Σ_s b_i^(s),        F̂_ij = (1/S) Σ_s b_i^(s) b_j^(s)

``b_i`` is 1 with probability ``Σ_k b_i(k) ρ_kk = ⟨n_i⟩`` and 0 otherwise, so
``F̂_i`` is the mean of ``S`` independent Bernoulli variables: unbiased, with
variance ``F_i (1 − F_i)/S``, i.e. standard error
``√(F_i (1 − F_i)/S) ≤ 1/(2√S)``.  The same holds for ``b_i b_j`` with
``⟨n_i n_j⟩``.  All features of a sample are computed from the same ``S``
bitstrings, so their errors are correlated, as on a device.

Not an encoding layer
---------------------
:class:`RydbergFeatureMap` is a preprocessing step, like
:class:`~hqnn_forge.preprocessing.PCANormalizer`, followed by any classifier.
It is not an ``nn.Module``, has no PennyLane circuit (``qlayer``) and no
trainable weights, so the encoding-layer contract
(``hqnn_forge.encoding.EncodingLayer``) does not apply to it, and neither do
the gate-based noise options (``noise_level``, ``apply_depolarizing_noise``):
its noise is the dephasing rate γ.  Nothing is differentiated; the results
carry no gradient.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

import torch

from hqnn_forge.rydberg.hamiltonian import MAX_ATOMS, _finite_tensor, rydberg_hamiltonian
from hqnn_forge.rydberg.lindblad import _non_negative_number, _step_count, evolve, readout
from hqnn_forge.rydberg.register import AtomRegister, _finite_number

__all__ = ["PulseEncoding", "RydbergFeatureMap"]

#: Matrix entries one chunk may hold per tensor when ``chunk_size`` is not
#: given: ``2^20`` ``complex128`` entries are 16 MiB.  A sample of ``N`` atoms
#: has ``4^N`` entries, so the default chunk is ``2^20 / 4^N`` samples (at
#: least 1): 4096 at 4 atoms, 256 at 6, 1 at 10.
_CHUNK_ENTRIES = 2**20


def _positive_integer(value: Any, name: str, *, or_none: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        suffix = " or None" if or_none else ""
        raise ValueError(f"{name} must be an integer >= 1{suffix}; got {value!r}.")
    return int(value)


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool; got {value!r}.")
    return value


class PulseEncoding:
    """
    The map from inputs to bounded pulse parameters: ``Δ_i = Δ_max · σ(x_i)``.

    One input per atom sets that atom's detuning through the logistic
    function ``σ(x) = 1/(1 + e^(−x))``; the Rabi frequency Ω and the pulse
    duration ``T`` are fixed, the same for every atom and sample.  Input
    column ``i`` gives the detuning of atom ``i``.  The module docstring
    states the conventions and derives what the defaults give.

    Parameters
    ----------
    n_features:
        Width of the inputs, an integer ``>= 1``: the number of atoms.
    omega:
        Rabi frequency Ω in rad/µs, finite and positive (the coefficient of
        ``X`` in the Hamiltonian is ``Ω/2``).  Required: Ω sets the scale of
        the model, and ``docs/rydberg-model.md`` names ``4π`` rad/µs as its
        reference value, not as a default.
    t:
        Pulse duration ``T`` in µs, finite and ``>= 0``.  Default: ``π/Ω``,
        a π pulse (``ΩT = π``).
    delta_max:
        Upper bound ``Δ_max`` of the detunings in rad/µs, finite and
        positive; the lower bound is 0.  Default: ``√3 Ω``, the first zero of
        the single-atom feature at ``ΩT = π``.

    Attributes
    ----------
    n_features : int
    omega : float
    t : float
        The resolved duration, ``π/Ω`` if none was given.
    delta_max : float
        The resolved bound, ``√3 Ω`` if none was given.

    Raises
    ------
    ValueError
        If ``n_features`` is not an integer ``>= 1``, ``omega`` or
        ``delta_max`` is not a finite number ``> 0``, or ``t`` is not a
        finite number ``>= 0``.

    Examples
    --------
    >>> import math
    >>> from hqnn_forge.rydberg import PulseEncoding
    >>> encoding = PulseEncoding(2, omega=4 * math.pi)
    >>> encoding.t  # ΩT = π
    0.25
    >>> round(encoding.delta_max / encoding.omega, 12)  # √3
    1.732050807569
    >>> (encoding.detunings([[0.0, math.log(3)]]) / encoding.delta_max).tolist()  # σ(x)
    [[0.5, 0.75]]
    """

    def __init__(
        self,
        n_features: int,
        *,
        omega: float,
        t: float | None = None,
        delta_max: float | None = None,
    ) -> None:
        self._n_features = _positive_integer(n_features, "n_features")
        self._omega = _finite_number(omega, "omega", positive=True)
        # ΩT = π and Δ_max = √3 Ω, the defaults of docs/rydberg-model.md.
        self._t = math.pi / self._omega if t is None else _non_negative_number(t, "t")
        self._delta_max = (
            math.sqrt(3.0) * self._omega
            if delta_max is None
            else _finite_number(delta_max, "delta_max", positive=True)
        )

    @property
    def n_features(self) -> int:
        """Width of the inputs: one per atom."""
        return self._n_features

    @property
    def omega(self) -> float:
        """Rabi frequency Ω in rad/µs."""
        return self._omega

    @property
    def t(self) -> float:
        """Pulse duration ``T`` in µs."""
        return self._t

    @property
    def delta_max(self) -> float:
        """Upper bound ``Δ_max`` of the detunings in rad/µs."""
        return self._delta_max

    def detunings(self, X: Any) -> torch.Tensor:
        """
        The detunings ``Δ_i = Δ_max · σ(x_i)`` of a batch of inputs.

        Parameters
        ----------
        X:
            Inputs of shape ``(n_samples, n_features)``, any real-valued
            array-like.  They are converted to ``float64`` and not modified.

        Returns
        -------
        torch.Tensor
            Shape ``(n_samples, n_features)``, ``float64``, in rad/µs, on the
            device of ``X`` and without a gradient: entry ``[b, i]`` is the
            detuning of atom ``i`` for sample ``b``.

        Raises
        ------
        TypeError
            If ``X`` is complex or boolean.
        ValueError
            If ``X`` does not have shape ``(n_samples, n_features)``, or
            holds a NaN or an infinity (a non-finite input is an error, not a
            value to squash).

        Notes
        -----
        ``0 < Δ_i < Δ_max`` holds for every finite input in exact arithmetic.
        In ``float64``, ``σ(x)`` rounds to 1 for ``x`` above about 37 and to
        0 for ``x`` below about −709.8 (where ``e^(−x)`` overflows), where
        ``Δ_i`` equals a bound.
        """
        n = self._n_features
        x = _finite_tensor(X, "X").detach()
        if x.ndim != 2 or x.shape[1] != n:
            raise ValueError(
                f"X must have shape (n_samples, {n}), one input per atom; "
                f"got shape {tuple(x.shape)}."
            )
        return self._delta_max * torch.sigmoid(x)

    def get_config(self) -> dict[str, Any]:
        """
        The constructor arguments, as a fresh JSON-serialisable dict.

        ``t`` and ``delta_max`` are the resolved values, so
        ``PulseEncoding(**encoding.get_config())`` is the same pulse whatever
        the defaults are.
        """
        return {
            "n_features": self._n_features,
            "omega": self._omega,
            "t": self._t,
            "delta_max": self._delta_max,
        }

    def __repr__(self) -> str:
        return (
            f"PulseEncoding(n_features={self._n_features}, omega={self._omega!r}, "
            f"t={self._t!r}, delta_max={self._delta_max!r})"
        )


class RydbergFeatureMap:
    """
    A fixed map from a batch of inputs to the excitation probabilities of a Rydberg array.

    Each sample sets the detunings of the atoms (``encoding``), the array
    evolves for the pulse duration under the Hamiltonian of
    ``docs/rydberg-model.md`` and dephasing at rate γ, and the features are
    ``⟨n_i⟩`` of every atom, optionally followed by ``⟨n_i n_j⟩`` of every
    pair.  The module docstring gives the pipeline, the conventions (input
    ``i`` feeds atom ``i``, the pair order) and the closed form without
    interactions.

    The map has no trainable parameters and nothing to fit: it is a
    preprocessing step, like :class:`~hqnn_forge.preprocessing.PCANormalizer`,
    and not an encoding layer.

    Parameters
    ----------
    register:
        The atoms' positions: an :class:`~hqnn_forge.rydberg.AtomRegister`,
        or its positions in µm, shape ``(n_atoms, 2)``, as :meth:`get_config`
        records them.  1 to :data:`~hqnn_forge.rydberg.MAX_ATOMS` atoms.
    encoding:
        The :class:`PulseEncoding` (detuning bound, Ω and ``T``), or its
        ``get_config()`` dict.  Its ``n_features`` must equal the number of
        atoms.
    c6:
        The ``C6`` coefficient in rad/µs · µm⁶, e.g.
        :data:`~hqnn_forge.rydberg.DEFAULT_C6`.  Required, also with
        ``interactions=False``.
    gamma:
        Dephasing rate γ in rad/µs, ``>= 0``: the rate of the local collapse
        operators ``√γ n_i``, so a single-atom coherence decays at ``γ/2``.
        Required; ``0.0`` is the noiseless map.
    n_steps:
        Number of splitting steps of the solver, an integer ``>= 1``.
        **Required when ``gamma > 0``, without a default** (see
        :func:`~hqnn_forge.rydberg.evolve` on how to choose it); not used
        when ``gamma = 0``, where the evolution is exact.  It is passed
        unchanged to every solver call and never derived from the data, so
        the features of a sample do not depend, beyond rounding, on which
        other samples it is processed with (see :meth:`transform`).
    interactions:
        ``False`` drops the term ``Σ_{i<j} V_ij n_i n_j`` and nothing else:
        the non-interacting control, in which feature ``i`` depends on input
        ``i`` alone.
    correlations:
        Append the pair correlations ``⟨n_i n_j⟩``, ``i < j``, in the order
        ``(0,1), (0,2), …, (N−2,N−1)``.

    Attributes
    ----------
    n_features_in : int
        Width of the inputs, the number of atoms ``N``.
    n_features_out : int
        Width of the features: ``N``, or ``N + N(N−1)/2`` with
        ``correlations=True``.
    evolution_time : float
        The pulse duration ``T`` in µs, ``encoding.t``.
    register, encoding, c6, gamma, n_steps, interactions, correlations
        The settings, read-only.

    Raises
    ------
    TypeError
        If ``encoding`` is neither a :class:`PulseEncoding` nor a mapping, is
        a mapping with a key :class:`PulseEncoding` does not take or without
        one it requires (``n_features``, ``omega``), or ``interactions`` or
        ``correlations`` is not a ``bool``.
    ValueError
        If the register has more than
        :data:`~hqnn_forge.rydberg.MAX_ATOMS` atoms or another number of
        atoms than ``encoding.n_features``; if ``c6`` is not a finite number
        or ``gamma`` not a finite number ``>= 0``; if ``n_steps`` is not an
        integer ``>= 1``, or is missing with ``gamma > 0``; or if positions
        or an encoding config are refused by
        :class:`~hqnn_forge.rydberg.AtomRegister` or :class:`PulseEncoding`.

    Examples
    --------
    >>> import math
    >>> import torch
    >>> from hqnn_forge.rydberg import DEFAULT_C6, AtomRegister, PulseEncoding, RydbergFeatureMap
    >>> omega = 4 * math.pi
    >>> spacing = AtomRegister.blockade_radius(DEFAULT_C6, omega)  # V = Ω
    >>> control = RydbergFeatureMap(
    ...     AtomRegister.chain(4, spacing),
    ...     PulseEncoding(4, omega=omega),
    ...     c6=DEFAULT_C6,
    ...     gamma=0.0,
    ...     interactions=False,
    ... )
    >>> X = torch.tensor([[-40.0, 0.0, 0.0, 40.0]])
    >>> [round(v, 3) for v in control.transform(X)[0].tolist()]  # f(1), f(√1.75), ·, f(2)
    [1.0, 0.437, 0.437, 0.0]
    >>> noisy = RydbergFeatureMap(
    ...     AtomRegister.chain(4, spacing),
    ...     PulseEncoding(4, omega=omega),
    ...     c6=DEFAULT_C6,
    ...     gamma=omega,
    ...     n_steps=200,
    ...     correlations=True,
    ... )
    >>> noisy.n_features_out
    10
    >>> rebuilt = RydbergFeatureMap(**noisy.get_config())
    >>> torch.equal(rebuilt.transform(X), noisy.transform(X))
    True
    """

    def __init__(
        self,
        register: Any,
        encoding: PulseEncoding | Mapping[str, Any],
        *,
        c6: float,
        gamma: float,
        n_steps: int | None = None,
        interactions: bool = True,
        correlations: bool = False,
    ) -> None:
        if not isinstance(register, AtomRegister):
            register = AtomRegister(register)
        if isinstance(encoding, Mapping):
            encoding = PulseEncoding(**encoding)
        elif not isinstance(encoding, PulseEncoding):
            raise TypeError(
                f"encoding must be a PulseEncoding or its get_config() dict; "
                f"got {type(encoding).__name__}."
            )
        n = register.n_atoms
        if n > MAX_ATOMS:
            raise ValueError(
                f"RydbergFeatureMap evolves dense 2^n x 2^n density matrices and accepts at "
                f"most {MAX_ATOMS} atoms; got {n} (4^{n} matrix entries per sample)."
            )
        if encoding.n_features != n:
            raise ValueError(
                f"the encoding takes one input per atom: encoding.n_features must equal the "
                f"register's {n} atoms; got {encoding.n_features}."
            )
        self._register = register
        self._encoding = encoding
        self._c6 = _finite_number(c6, "c6")
        self._gamma = _non_negative_number(gamma, "gamma")
        self._n_steps = None if n_steps is None else _step_count(n_steps)
        if self._gamma > 0 and self._n_steps is None:
            raise ValueError(
                "n_steps is required when gamma > 0: the solver's splitting has an error of "
                "order (t / n_steps)^2, and no step count is assumed (see evolve)."
            )
        self._interactions = _flag(interactions, "interactions")
        self._correlations = _flag(correlations, "correlations")

    @property
    def register(self) -> AtomRegister:
        """The atoms' positions."""
        return self._register

    @property
    def encoding(self) -> PulseEncoding:
        """The map from inputs to detunings, with Ω and ``T``."""
        return self._encoding

    @property
    def c6(self) -> float:
        """The ``C6`` coefficient in rad/µs · µm⁶."""
        return self._c6

    @property
    def gamma(self) -> float:
        """Dephasing rate γ in rad/µs."""
        return self._gamma

    @property
    def n_steps(self) -> int | None:
        """Splitting steps of the solver; ``None`` only when ``gamma = 0``."""
        return self._n_steps

    @property
    def interactions(self) -> bool:
        """Whether the Hamiltonian keeps its ``n_i n_j`` term."""
        return self._interactions

    @property
    def correlations(self) -> bool:
        """Whether the pair correlations are appended to the features."""
        return self._correlations

    @property
    def n_features_in(self) -> int:
        """Width of the inputs: the number of atoms."""
        return self._register.n_atoms

    @property
    def n_features_out(self) -> int:
        """Width of the features: ``N``, or ``N + N(N−1)/2`` with ``correlations``."""
        n = self._register.n_atoms
        return n + n * (n - 1) // 2 if self._correlations else n

    @property
    def evolution_time(self) -> float:
        """The pulse duration ``T`` in µs."""
        return self._encoding.t

    def _chunk_size(self, chunk_size: int | None) -> int:
        if chunk_size is None:
            return max(1, _CHUNK_ENTRIES // 4**self._register.n_atoms)
        return _positive_integer(chunk_size, "chunk_size", or_none=True)

    @torch.no_grad()
    def _evolve(self, delta: torch.Tensor) -> torch.Tensor:
        """``ρ(T)`` for the detunings of one chunk, shape ``(chunk, N)``."""
        hamiltonian = rydberg_hamiltonian(
            self._register,
            self._encoding.omega,
            delta,
            c6=self._c6,
            interactions=self._interactions,
        )
        return evolve(hamiltonian, self._encoding.t, gamma=self._gamma, n_steps=self._n_steps)

    @torch.no_grad()
    def _sample(self, rho: torch.Tensor, shots: int, generator: torch.Generator) -> torch.Tensor:
        """
        The features of one chunk estimated from ``shots`` bitstrings per sample.

        The outcome probabilities are the diagonal of ρ.  ``counts[k]`` is how
        often basis state ``k`` was drawn, and the estimate of a feature is the
        feature of the diagonal state with the frequencies ``counts / shots``:
        ``Σ_k b_i(k) counts[k] / shots`` is the mean of ``b_i`` over the
        bitstrings.  The samples are drawn one at a time, in order, so the
        generator's stream does not depend on how they are chunked.
        """
        # A population can come out of the splitting as −1e-17; a probability cannot.
        populations = rho.diagonal(dim1=-2, dim2=-1).real.clamp(min=0.0).to(generator.device)
        frequencies = torch.empty_like(populations)
        for sample in range(populations.shape[0]):
            outcomes = torch.multinomial(
                populations[sample], shots, replacement=True, generator=generator
            )
            counts = torch.bincount(outcomes, minlength=populations.shape[1])
            frequencies[sample] = counts.to(torch.float64) / shots
        estimates = readout(torch.diag_embed(frequencies), pairs=self._correlations)
        return estimates.to(rho.device)

    @torch.no_grad()
    def transform(
        self,
        X: Any,
        *,
        shots: int | None = None,
        generator: torch.Generator | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        The features of a batch of inputs.

        Parameters
        ----------
        X:
            Inputs of shape ``(n_samples, n_features_in)``, any real-valued
            array-like; column ``i`` sets the detuning of atom ``i``.
        shots:
            ``None`` (default): exact expectation values.  An integer
            ``S >= 1``: each feature is estimated from ``S`` bitstrings
            sampled from the diagonal of that sample's ρ, as the mean of
            ``b_i`` (and of ``b_i b_j``).  The estimate is unbiased with
            standard error ``√(F(1 − F)/S) ≤ 1/(2√S)`` for a feature of
            value ``F``.  There is no default number of shots.
        generator:
            The ``torch.Generator`` the bitstrings are drawn with.  Required
            when ``shots`` is given, so that every estimate comes from a
            stated source of randomness; it is advanced by the call, and a
            generator with the same seed repeats the estimates.  Not used
            with ``shots=None``.
        chunk_size:
            Samples evolved at a time, an integer ``>= 1``.  Default: as
            many as hold ``2^20`` matrix entries, ``max(1, 2^20 / 4^N)``.
            It changes the result by rounding only (see the Notes).

        Returns
        -------
        torch.Tensor
            Shape ``(n_samples, n_features_out)``, ``float64``, on the device
            of ``X``, without a gradient: ``⟨n_0⟩, …, ⟨n_{N−1}⟩`` and, with
            ``correlations=True``, the pairs in the order
            ``(0,1), (0,2), …, (N−2,N−1)``.

        Raises
        ------
        TypeError
            If ``X`` is complex or boolean, or ``generator`` is not a
            ``torch.Generator``.
        ValueError
            If ``X`` does not have shape ``(n_samples, n_features_in)`` or
            holds a NaN or an infinity; if ``shots`` or ``chunk_size`` is not
            an integer ``>= 1`` or ``None``; or if ``shots`` is given without
            a ``generator``.

        Notes
        -----
        * **Memory.**  The samples are evolved ``chunk_size`` at a time and
          the features written into one preallocated result, so the working
          memory is that of one chunk: the solver holds a handful of
          ``complex128`` tensors of ``chunk_size · 4^N`` entries
          (``16 · 4^N`` bytes per sample each; 16 MiB each at the default).
        * **Chunked and unchunked results are equal to rounding, not bit for
          bit.**  Every sample is evolved with the same ``T``, γ and
          ``n_steps`` whatever the chunk, and bitstrings are drawn sample by
          sample in order, so no setting and no random draw depends on
          ``chunk_size``.  What is left is the rounding of the solver's
          batched matrix products, which a linear-algebra backend may sum in
          another order for another batch size.  Measured on a CPU, features
          and states of two chunkings do not differ at all up to 4 atoms;
          from 5 atoms on they differ by up to ``1e-15`` at ``gamma = 0`` and
          by up to ``2e-16`` per solver step with dephasing (``2e-13`` at
          2000 steps), the size of the rounding error either result has
          anyway, and orders of magnitude below the splitting error.  The
          same holds for :meth:`states`.  With ``shots`` the drawn bitstrings
          were the same in every chunking tried, and the estimates differed
          by a few ``1e-16``.  For bit-for-bit repeatable features, keep
          ``chunk_size`` (and the machine) fixed.
        * **Precision.**  ``X`` is converted to ``float64`` whatever its
          dtype.  With ``shots``, a population that rounding left slightly
          below 0 is taken as 0.
        """
        if shots is not None:
            shots = _positive_integer(shots, "shots", or_none=True)
            if generator is None:
                raise ValueError(
                    "generator is required when shots is given: pass a torch.Generator, "
                    "e.g. torch.Generator().manual_seed(0)."
                )
            if not isinstance(generator, torch.Generator):
                raise TypeError(
                    f"generator must be a torch.Generator; got {type(generator).__name__}."
                )
        delta = self._encoding.detunings(X)
        size = self._chunk_size(chunk_size)
        n_samples = delta.shape[0]
        features = torch.empty(
            n_samples, self.n_features_out, dtype=torch.float64, device=delta.device
        )
        for start in range(0, n_samples, size):
            rho = self._evolve(delta[start : start + size])
            if shots is None or generator is None:
                chunk = readout(rho, pairs=self._correlations)
            else:
                chunk = self._sample(rho, shots, generator)
            features[start : start + size] = chunk
        return features

    @torch.no_grad()
    def states(self, X: Any, *, chunk_size: int | None = None) -> torch.Tensor:
        """
        The density matrices ``ρ(T)`` the features are read from.

        ``readout(feature_map.states(X), pairs=feature_map.correlations)``
        equals ``feature_map.transform(X)``.

        Parameters
        ----------
        X:
            Inputs of shape ``(n_samples, n_features_in)``, as for
            :meth:`transform`.
        chunk_size:
            Samples evolved at a time, as for :meth:`transform`: it changes
            the states by rounding only.  It bounds the solver's working
            memory, not the result, which holds ``n_samples · 4^N`` entries.

        Returns
        -------
        torch.Tensor
            Shape ``(n_samples, 2^N, 2^N)``, ``complex128``, on the device of
            ``X``, without a gradient.  Entry ``[b, k, l]`` is ``⟨k|ρ|l⟩`` of
            sample ``b`` in the basis of
            :func:`~hqnn_forge.rydberg.rydberg_hamiltonian` (atom 0 the most
            significant bit of the index).

        Raises
        ------
        TypeError
            If ``X`` is complex or boolean.
        ValueError
            If ``X`` does not have shape ``(n_samples, n_features_in)`` or
            holds a NaN or an infinity, or ``chunk_size`` is not an integer
            ``>= 1`` or ``None``.
        """
        delta = self._encoding.detunings(X)
        size = self._chunk_size(chunk_size)
        n_samples = delta.shape[0]
        dim = 2**self._register.n_atoms
        states = torch.empty(n_samples, dim, dim, dtype=torch.complex128, device=delta.device)
        for start in range(0, n_samples, size):
            states[start : start + size] = self._evolve(delta[start : start + size])
        return states

    def get_config(self) -> dict[str, Any]:
        """
        Every setting of the map, as a fresh JSON-serialisable dict.

        The keys are the constructor's arguments, so
        ``type(m)(**m.get_config())`` builds a map with the same features,
        also after a round trip through JSON:

        ``register``
            The positions in µm, a list of ``[x, y]`` per atom.
        ``encoding``
            ``PulseEncoding.get_config()``: ``n_features``, Ω (``omega``),
            ``T`` (``t``) and the bound ``delta_max``, in rad/µs and µs.
        ``c6``, ``gamma``, ``n_steps``, ``interactions``, ``correlations``
            As passed; ``n_steps`` is the solver's step count, ``None`` if
            none was given (``gamma = 0``).

        The number of shots and the chunk size are arguments of
        :meth:`transform`, not settings of the map, and are not recorded.
        """
        return {
            "register": self._register.positions.tolist(),
            "encoding": self._encoding.get_config(),
            "c6": self._c6,
            "gamma": self._gamma,
            "n_steps": self._n_steps,
            "interactions": self._interactions,
            "correlations": self._correlations,
        }

    def __repr__(self) -> str:
        return (
            f"RydbergFeatureMap(n_atoms={self._register.n_atoms}, encoding={self._encoding!r}, "
            f"c6={self._c6!r}, gamma={self._gamma!r}, n_steps={self._n_steps!r}, "
            f"interactions={self._interactions}, correlations={self._correlations})"
        )
