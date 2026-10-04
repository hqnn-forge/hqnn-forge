"""
hqnn_forge.utils.rng
====================
Seeded torch draws that leave the caller's global RNG exactly as it was.

The library's initialisers and ``nn.Dropout`` draw from torch's global RNG and
take no generator.  :func:`seeded_rng` runs a block on that RNG seeded with a
given seed and restores the caller's state on the way out -- also when the
block raises -- so a seeded model or fit never reseeds, or advances, the
stream the caller set up (#175).  Every seeded draw in the library goes
through it, and SPSA's common random numbers go through the same
:func:`rng_state` / :func:`set_rng_state` pair, so what "restored" covers is
decided in one place.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch


def as_seed(seed: object, name: str = "init_seed") -> int | None:
    """
    ``seed`` as a plain ``int`` (or ``None``), for an ``init_seed`` argument.

    NumPy integers, which scikit-learn hands around as ``random_state``, are
    accepted and converted, so the config -- and a checkpoint of it, read with
    ``weights_only=True`` -- only ever holds a Python int.
    """
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"{name} must be an int or None; got {type(seed).__name__}.")
    return int(seed)


RngState = tuple[torch.Tensor, list[torch.Tensor] | None]


def rng_state() -> RngState:
    """
    The global RNG state :func:`seeded_rng` saves: the CPU RNG's, and every
    CUDA device's if CUDA is initialised.  Put back with :func:`set_rng_state`.
    """
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return torch.random.get_rng_state(), cuda


def set_rng_state(state: RngState) -> None:
    """Restore a state from :func:`rng_state`."""
    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _reseed(seed: int) -> None:
    # Not torch.manual_seed: with CUDA not yet initialised, that queues a
    # reseed which runs whenever CUDA is first used -- after the caller's
    # state has been "restored" -- and it reseeds MPS/XPU, whose state has no
    # accessor to save.  Only the RNGs saved on entry are seeded.
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def seeded_rng(seed: int | None) -> Iterator[Callable[[], None]]:
    """
    Run the block on torch's global RNG seeded with ``seed``, then restore it.

    On entry the CPU RNG -- and every CUDA device's, if CUDA is initialised --
    is saved and seeded with ``seed``; on exit, normal or not, the saved state
    is put back.  The draws inside are those ``torch.manual_seed(seed)`` would
    give on those devices.  Other accelerators (MPS, XPU) expose no state to
    save, so they are neither seeded nor touched.

    The context value is a ``reseed()`` callable that seeds the RNG afresh
    with ``seed``: a classifier builds its layers (whose default init draws are
    all overwritten) and then calls it right before ``_initialise_weights``,
    so its initial weights do not depend on how many draws construction took.

    With ``seed=None`` the block runs on the global RNG unchanged and
    ``reseed()`` does nothing.
    """
    if seed is None:
        yield lambda: None
        return
    fixed = int(seed)
    saved = rng_state()
    _reseed(fixed)
    try:
        yield lambda: _reseed(fixed)
    finally:
        set_rng_state(saved)
