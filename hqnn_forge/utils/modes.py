"""
hqnn_forge.utils.modes
======================
Temporarily switch a module to eval or train mode.

``torch.no_grad()`` only disables autograd.  Layers such as ``nn.Dropout`` read
``module.training`` instead, so inference code that can run mid-training has to
switch to eval mode itself and put every submodule's mode back afterwards.
Training code has the mirror problem: ``module.train()`` recurses, and would
unfreeze a submodule the caller put in eval mode on purpose, such as a
batch-norm layer whose statistics are frozen while a head is fine-tuned.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def eval_mode(module: nn.Module) -> Iterator[None]:
    """
    Put ``module`` and all its submodules in eval mode for the ``with`` block.

    On exit, including when the block raises, every submodule gets back the
    ``training`` flag it had on entry.  A single ``module.train(was_training)``
    would not do that: it overwrites submodules the caller had put in a
    different mode, such as a model in train mode with its dropout frozen.

    Parameters
    ----------
    module:
        Module to switch.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from hqnn_forge.utils import eval_mode
    >>> model = nn.Sequential(nn.Linear(4, 1), nn.Dropout(0.5))
    >>> with eval_mode(model):
    ...     logits = model(torch.ones(2, 4))  # dropout off
    ...     model.training, model[1].training
    (False, False)
    >>> model.training, model[1].training  # train mode is back
    (True, True)
    """
    modes = _modes(module)
    module.eval()
    try:
        yield
    finally:
        _restore(modes)


@contextmanager
def train_mode(module: nn.Module) -> Iterator[None]:
    """
    Train ``module`` for the ``with`` block without unfreezing what the caller froze.

    If any submodule is in train mode on entry, every submodule keeps the mode
    it has: a model in train mode with a batch-norm layer put in eval mode
    trains with that layer still frozen, and so does a model put in eval mode
    with only its head switched back to train mode.  Only if ``module`` and all
    its submodules are in eval mode -- a model fresh from ``load_checkpoint``,
    say -- is there no training configuration to respect, and every submodule
    is put in train mode.  On exit, including when the block raises, every
    submodule gets back the ``training`` flag it had on entry, as with
    :func:`eval_mode`.

    Parameters
    ----------
    module:
        Module to switch.

    Examples
    --------
    A model loaded in eval mode is put in train mode for the block:

    >>> import torch
    >>> from torch import nn
    >>> from hqnn_forge.utils import train_mode
    >>> model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 1)).eval()
    >>> with train_mode(model):
    ...     model.training, model[1].training
    (True, True)
    >>> model.training, model[1].training  # eval mode is back
    (False, False)

    A batch-norm layer the caller froze stays frozen:

    >>> _ = model.train()
    >>> _ = model[1].eval()
    >>> with train_mode(model):
    ...     loss = model(torch.ones(2, 4)).sum()
    ...     model.training, model[1].training
    (True, False)
    """
    modes = _modes(module)
    if not any(training for _, training in modes):
        module.train()
    try:
        yield
    finally:
        _restore(modes)


def _modes(module: nn.Module) -> list[tuple[nn.Module, bool]]:
    return [(submodule, submodule.training) for submodule in module.modules()]


def _restore(modes: list[tuple[nn.Module, bool]]) -> None:
    # Restore through train() so overrides on submodules still run.
    for submodule, training in modes:
        submodule.train(training)
    # train() recurses, so a submodule registered under two parents can be
    # overwritten by the parent restored after it.  Set the flags last.
    for submodule, training in modes:
        submodule.training = training
