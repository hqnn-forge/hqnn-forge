"""
hqnn_forge.initializers.restricted_variance
=============================================
Small-angle weight initialisation for variational quantum circuits.

Theory
------
In variational circuits that are deep and expressive enough to approximate a
2-design -- for which uniformly drawn parameters are one ingredient, not the
whole condition -- the gradient variance decays exponentially in the number of
qubits n:

    Var[∂L/∂θ] ∝ 2^{-n}   (global cost, McClean et al. 2018)

Two published results bound that decay.  Cerezo et al. (2021) show that for
*local* cost functions and shallow circuits (depth O(log n)) the decay is
only polynomial.  A single-qubit ⟨Z_i⟩ readout is a local observable, but
whether it is a local *cost* in their sense depends on how far the circuit
spreads it; for this library's default circuit it is not, and for
``entangler="brickwork"`` at shallow depth it is (both measured below).
Zhang et al. (2022) show that drawing the parameters from N(0, σ²) with
σ² = O(1/L) instead of uniformly bounds the gradient norm below by a
polynomial in n and L, for deep circuits too.

The two initialisers here are **this library's own heuristics**; neither
formula is taken from a paper:

    restricted_normal_init_:  σ   = scale / sqrt(n_qubits * n_layers)
    block_local_init_:        σ_ℓ = scale / sqrt(n_qubits * (n_layers + ℓ))

Both shrink with the total depth L and are loosely in the spirit of Zhang et
al.: σ² = scale²/(n L) carries their 1/L factor, with an extra 1/n the paper
does not ask for.  ``block_local_init_`` starts at the same σ in layer 0 and
tapers from there, to σ/sqrt((2L - 1)/L) -- just over σ/sqrt(2) at depth --
in the last layer, so every σ_ℓ² lies in [scale²/(n(2L-1)), scale²/(nL)]
and is O(1/L) as well.  Until #166 its schedule was scale/sqrt(n (ℓ + 1)),
which depends on the layer index alone: layer 0 of a 64-layer circuit was as
wide as layer 0 of a 2-layer one, sqrt(L) = 8x wider than the global scheme.

What the σ are for is conditioning, not a plateau guarantee: small angles keep
the initial state near the encoded product state, where the single-qubit ⟨Z_i⟩
readouts are still informative.  Neither σ is derived to guarantee any
particular gradient variance, and two caveats are worth stating outright:

* "Small" only holds above a certain circuit size.  Angles drawn uniformly
  from [0, 2π) have standard deviation 2π/sqrt(12) ≈ 1.81 rad, so with the
  default ``scale = π`` this initialisation is narrower than that uniform
  draw only for ``n_qubits * n_layers ≥ 4``.  At the library defaults (8 qubits ×
  2 layers) σ = π/4 ≈ 0.79 rad, 43% of the uniform spread; at the smallest
  circuit the encoders accept (2 qubits, 1 layer) σ ≈ 2.22 rad, 22% *wider*
  than uniform.  Both initialisers emit a ``UserWarning`` naming the two
  standard deviations whenever the σ they draw (for ``block_local_init_``,
  that of layer 0, its widest) is not narrower than the uniform one.
* Leaving the uniform regime is not by itself what avoids a plateau.  The
  2-design argument needs depth and structure too.  Nor is the default
  circuit in the shallow local-cost regime of Cerezo et al.: its readouts
  behave as global costs, as the next section measures.

The initial circuit is therefore a *small-angle* one, not an identity one.
Grant et al. (2019) is a different strategy (identity blocks: parameters
chosen so that consecutive blocks compose to the identity) and is not
implemented here; it is cited for contrast.

What the initialiser does for this library's circuits (measured)
----------------------------------------------------------------
The local-cost argument above does not apply to the library's default
circuit.  Its CNOT ring is a cascade, CNOT(0,1), CNOT(1,2), …, CNOT(n-1,0),
so the backward light cone of ⟨Z_i⟩ through one ring covers qubits
{0, …, i+1} for 0 < i < n-1 and all n qubits for i = 0 and i = n-1; through
a second ring the closing CNOT(n-1,0) pulls in every qubit.  From 2 layers on
(the default) every ⟨Z_i⟩ therefore depends on every input, and the readouts
behave as global costs.  Measured with
:func:`hqnn_forge.diagnostics.gradient_variance` on ``QuantumEncodingLayer``
(2 layers, cost ⟨Z_0⟩, ``default.qubit``, mean
per-weight gradient variance over 5 seeds × 300 draws of weights and inputs),
the ratio of restricted-init to uniform-init variance is:

    inputs uniform in   n=4    n=6    n=8
    {0}                 1.09   1.52   1.75
    ±π/4                1.00   1.20   1.53
    ±π                  0.97   0.97   1.00     (5-seed range at n=8: 0.80–1.12)

and the uniform-init variance itself at ±π falls 0.0153 → 0.00428 → 0.00166
from 4 to 8 qubits, about 3x per two qubits, with the restricted init
following the same curve (0.0149 → 0.0042 → 0.0017).

So:

* With inputs spread over (-π, π), which is what both classifiers feed the
  circuit (``tanh(·)·π``) and what ``PCANormalizer(scale_to_pi=True)``
  produces, the initialiser makes **no measurable difference**: the angle
  embedding already randomises the state, and shrinking the weight angles
  cannot bring it back near the identity.
* With inputs near zero it keeps more gradient variance, by a factor that
  grows over the measured range: 1.09x at 4 qubits, 1.75x at 8 (5-seed range
  1.67–1.89).  Whether that growth continues past 8 qubits -- i.e. whether the
  initialiser slows the decay for near-zero inputs rather than shifting it --
  has not been measured.
* With inputs spread over (-π, π) it does **not** change the exponential
  decay with qubit count: both inits lose about 3x per two qubits.  That decay
  is set by the circuit, not the initialisation.

The initialisers are kept as the default because they are harmless and
cheap, and because the ``scale`` argument gives a one-parameter handle on
the initial angle spread.  They should not be relied on for trainability at
larger qubit counts; a locality-preserving entangler is the lever for that.

``entangler="brickwork"`` is that entangler: nearest-neighbour CNOT pairs,
even then odd, with no wrap-around, so each layer widens the backward light
cone of a readout by at most two qubits on each side.  The weights ⟨Z_0⟩
depends on sit on qubit {0} after one layer, {0, 1} after two and
{0, …, 3} after three; its inputs reach one layer further, since the
embedding sits before the first CNOT pairs: x_0, x_1 after one layer and
x_0 … x_3 after two, so at n=4 two layers already make ⟨Z_0⟩ depend on
every input.  Measured under the same protocol, now as the **total** gradient
variance (summed over all weights) at inputs ±π, because the weights outside
the light cone of ⟨Z_0⟩ have exactly zero gradient and would dilute a
per-weight mean by 1/n on their own:

    total variance       n=4     n=6     n=8     n=4 → n=8
    ring, uniform        0.367   0.154   0.0795  4.6x  (5-seed range 4.1–5.4x)
    brickwork, uniform   0.412   0.407   0.433   0.95x (5-seed range 0.90–1.02x)

The ring loses about 2x per two qubits in total (the 3x above is per weight,
over a weight count that grows with n).  Brickwork loses nothing over the
measured range: at 2 layers the light cone of ⟨Z_0⟩'s weights is 2 qubits
wide (of its inputs, 4) whatever ``n_qubits`` is, so adding qubits only adds
weights with zero gradient.  The
restricted init's ratio to uniform init for brickwork is:

    inputs uniform in   n=4    n=6    n=8
    {0}                 0.97   0.88   0.82     (5-seed range at n=8: 0.76–0.88)
    ±π                  1.00   0.93   0.80     (5-seed range at n=8: 0.77–0.83)

so on a circuit that keeps its readouts local the restricted init adds no
variance at 4 qubits and costs some at 6 and 8, near zero input as well.  The escape
from the decay is the entangler's, not the initialiser's.  It lasts while
the light cone is narrower than the register: ⟨Z_0⟩'s weight light cone
covers every qubit from about ``n_layers = n_qubits / 2 + 1`` (its input
light cone one layer earlier), a middle qubit's from about
``n_qubits / 4 + 1``, and past that brickwork's readouts are global too.

``tests/test_gradient_variance.py`` pins the statements above so a change
that alters them is noticed; of the brickwork tables, it pins the 4 → 8
decays and the ±π ratio at 8 qubits, while the {0} row and the 6-qubit
column are measurements only.

Functions
---------
restricted_normal_init_     In-place; fills a tensor with restricted-normal values.
block_local_init_           restricted_normal_init_ tapered by layer: sqrt(L/(L+ℓ)) on layer ℓ.

References
----------
* McClean et al. (2018) "Barren plateaus in quantum neural network training
  landscapes", Nature Communications 9, 4812.
* Cerezo et al. (2021) "Cost function dependent barren plateaus in shallow
  parametrized quantum circuits", Nature Communications 12, 1791.
* Grant et al. (2019) "An initialization strategy for addressing barren
  plateaus in parametrized quantum circuits", Quantum 3, 214.
* Zhang et al. (2022) "Escaping from the barren plateau via Gaussian
  initializations in deep variational quantum circuits", NeurIPS 35.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from hqnn_forge._warnings import external_stacklevel

#: Standard deviation of an angle drawn uniformly from [0, 2π): 2π/sqrt(12).
UNIFORM_STD = 2 * math.pi / math.sqrt(12)

# Set by _not_restricting_ignored() below.
_SUPPRESSED: ContextVar[bool] = ContextVar("_not_restricting_suppressed", default=False)


def _warn_if_not_restricting(
    std: float, name: str, n_qubits: int, n_layers: int, scale: float
) -> None:
    """
    Warn when ``std`` is not narrower than a uniform draw over [0, 2π).

    The relative slack of 1e-9 makes the boundary case, σ equal to the uniform
    std (``n_qubits * n_layers == 3`` at ``scale = π``), warn regardless of
    how the two expressions round.  The warning is attributed to the first
    frame outside ``hqnn_forge``, so a classifier built in a user's script
    reports that script's line.
    """
    if _SUPPRESSED.get():
        return
    if std >= UNIFORM_STD * (1 - 1e-9):
        rule = (
            "  At scale=π that happens for n_qubits * n_layers <= 3."
            if math.isclose(scale, math.pi)
            else ""
        )
        warnings.warn(
            f"{name}: σ = {std:.4f} rad at scale={scale:.4g}, n_qubits={n_qubits}, "
            f"n_layers={n_layers} is not narrower than a uniform draw over [0, 2π) "
            f"(std {UNIFORM_STD:.4f} rad), so this initialisation restricts nothing.{rule}  "
            f"For a classifier, pass init_strategy='normal' or build a larger "
            f"n_qubits * n_layers; see hqnn_forge.initializers.restricted_variance.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )


@contextmanager
def _not_restricting_ignored() -> Iterator[None]:
    """
    Silence the warning above for draws that are meant to be small or are
    discarded: a diagnostic comparing inits at toy sizes, or a model rebuilt
    only to receive a checkpoint's weights.

    A context variable rather than ``warnings.catch_warnings``: that mutates
    the process-global filter list, which is not thread-safe, and every exit
    resets the once-per-location registry, so any other warning raised inside
    (a PennyLane deprecation in a sampling loop, say) would print every time.
    """
    token = _SUPPRESSED.set(True)
    try:
        yield
    finally:
        _SUPPRESSED.reset(token)


# ---------------------------------------------------------------------------
# In-place initialiser: single call, shared σ across all parameters
# ---------------------------------------------------------------------------


def restricted_normal_init_(
    tensor: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    scale: float = math.pi,
) -> torch.Tensor:
    """
    Fill *tensor* **in-place** with values drawn from N(0, σ²) where

        σ = scale / sqrt(n_qubits * n_layers)

    A small-angle initialisation: σ shrinks with both width and depth, so for
    circuits above a minimum size the initial parameters are narrower than a
    uniform draw over [0, 2π) (standard deviation ≈ 1.81 rad) -- with the
    default ``scale``, that means ``n_qubits * n_layers ≥ 4``; below that size
    this σ is the wider of the two.  The formula is this library's heuristic (see
    the module docstring), not a published prescription, and it does not by
    itself guarantee O(1) gradient variance.

    Parameters
    ----------
    tensor:
        The weight tensor to initialise.  Typically the ``weights`` parameter
        of a ``QuantumEncodingLayer``, shape ``(n_layers, n_qubits, 3)``.
    n_qubits:
        Number of qubits in the circuit.
    n_layers:
        Number of variational layers.
    scale:
        Numerator of the standard deviation formula.  Default: π.
        Smaller values narrow the initial angle spread; per the module
        docstring, that does not counter the decay of gradient variance with
        qubit count for inputs spread over (-π, π).

    Returns
    -------
    torch.Tensor
        The initialised tensor (modified in-place and returned for chaining).

    Raises
    ------
    ValueError
        If ``n_qubits`` or ``n_layers`` is less than 1.

    Warns
    -----
    UserWarning
        If σ is not narrower than the uniform std 2π/sqrt(12) ≈ 1.81 rad,
        i.e. ``n_qubits * n_layers <= 3`` at the default ``scale``.

    Examples
    --------
    >>> import math
    >>> import torch
    >>> from hqnn_forge.initializers import restricted_normal_init_
    >>> w = torch.empty(2, 8, 3)  # (n_layers=2, n_qubits=8, 3 Euler angles)
    >>> restricted_normal_init_(w, n_qubits=8, n_layers=2) is w  # filled in place
    True
    >>> round(math.pi / math.sqrt(8 * 2), 3)  # the σ w was drawn with
    0.785
    """
    if n_qubits < 1 or n_layers < 1:
        raise ValueError(
            f"n_qubits and n_layers must be ≥ 1; got n_qubits={n_qubits}, n_layers={n_layers}."
        )

    std = scale / math.sqrt(n_qubits * n_layers)
    _warn_if_not_restricting(std, "restricted_normal_init_", n_qubits, n_layers, scale)
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std)
    return tensor


# ---------------------------------------------------------------------------
# Block-local variant: the restricted σ, tapered by layer
# ---------------------------------------------------------------------------


def block_local_init_(
    tensor: torch.Tensor,
    n_qubits: int,
    scale: float = math.pi,
) -> torch.Tensor:
    """
    Fill *tensor* **in-place** with restricted-normal values tapered by layer.

    For a circuit of ``L = tensor.shape[0]`` layers, layer ℓ is drawn with

        σ_ℓ = scale / sqrt(n_qubits * (L + ℓ))

    A tapered variant of :func:`restricted_normal_init_`: layer 0 gets its σ,
    scale / sqrt(n_qubits * L), and each later layer a little less, down to
    that σ divided by sqrt((2L - 1)/L) < sqrt(2) in the last layer.  Every
    σ_ℓ² is therefore O(1/L) in the total depth, the scaling of Zhang et al.
    (2022), and the layers are still ordered from widest to narrowest.  The
    schedule itself is this library's heuristic -- neither the identity-block
    scheme of Grant et al. (2019) nor a formula from Zhang et al.

    The whole tensor is drawn from N(0, 1) in one call and each layer then
    scaled by σ_ℓ, so for the same RNG state the result is exactly the
    output of :func:`restricted_normal_init_` times ``sqrt(L / (L + ℓ))`` on
    layer ℓ.

    Parameters
    ----------
    tensor:
        Weight tensor of shape ``(n_layers, n_qubits, 3)`` or any shape
        where ``dim 0`` indexes layers.
    n_qubits:
        Number of qubits.
    scale:
        Numerator for std computation.  Default: π.

    Returns
    -------
    torch.Tensor
        The initialised tensor (in-place).

    Warns
    -----
    UserWarning
        If layer 0's σ, the widest, is not narrower than the uniform std
        2π/sqrt(12) ≈ 1.81 rad, i.e. ``n_qubits * L <= 3`` at the default
        ``scale``.

    Examples
    --------
    >>> import math
    >>> import torch
    >>> from hqnn_forge.initializers import block_local_init_
    >>> w = torch.empty(4, 8, 3)  # 4-layer circuit
    >>> block_local_init_(w, n_qubits=8) is w  # filled in place
    True
    >>> [round(math.pi / math.sqrt(8 * (4 + layer)), 3) for layer in range(4)]  # σ_ℓ of w[ℓ]
    [0.555, 0.497, 0.453, 0.42]
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be ≥ 1; got {n_qubits}.")

    n_layers: int = tensor.shape[0]
    sigmas = [scale / math.sqrt(n_qubits * (n_layers + layer)) for layer in range(n_layers)]
    if sigmas:
        # Layer 0 is the widest.
        _warn_if_not_restricting(sigmas[0], "block_local_init_", n_qubits, n_layers, scale)
    stds = torch.tensor(sigmas, dtype=tensor.dtype, device=tensor.device)
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=1.0)
        tensor.mul_(stds.view(-1, *([1] * (tensor.dim() - 1))))
    return tensor
