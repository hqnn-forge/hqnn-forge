"""
hqnn_forge.utils.checkpoint
===========================
Save and reload a hybrid classifier together with its architecture.

``torch.save(model.state_dict())`` alone is not enough to get a model back:
the state dict holds weights, not ``n_qubits``, ``n_layers``,
``encoding_type`` or the other constructor arguments needed to build a module
those weights fit.  A checkpoint written here stores both, plus the class
name and the library version, and ``load_checkpoint`` rebuilds the model and
loads the weights in one call.

Format
------
A ``torch.save`` file containing a plain dict::

    {
        "format_version": 1,
        "hqnn_forge_version": "0.1.0",
        "class_name": "HybridBinaryClassifier",
        "config": {...constructor kwargs...},
        "known_args": [...every constructor argument the writer had...],
        "state_dict": {...},
    }

``known_args`` was added after the first checkpoints were written; a file
without it is read exactly as before (see the rules below).

Compatibility rules
-------------------
Two things decide whether a file still rebuilds its model, and each has its
own mechanism:

* **The dict layout** is what ``FORMAT_VERSION`` covers.  It is bumped when a
  key is removed, renamed, or changes meaning, i.e. when a reader of the old
  version would misread the new file or the reverse.  Adding a key that an
  older reader ignores and a newer one does not require (``known_args``) is
  not a bump.
* **The constructors** are covered by the config.  Every argument is stored
  with its value, so a *changed default* never touches a checkpoint that
  recorded the argument, and needs no entry anywhere.  An *added* argument is
  missing from every older file, and the PR adding it must record a decision:

  - an entry in ``_LEGACY_DEFAULTS`` giving the value that reproduces what
    the circuit did before the argument existed -- the old behaviour, never
    the new default -- so older files load with a ``RuntimeWarning``; or
  - an entry in ``_NO_LEGACY_DEFAULT``, when no value reproduces the old
    model, so older files missing it are refused.

  ``tests/test_checkpoint.py`` pins every constructor's arguments and fails
  when one appears without either entry.  A *removed* or *renamed* argument
  makes older files carry an unexpected key, which is refused; supporting
  them needs a migration in ``load_checkpoint``.  A *value* a constructor
  stops accepting is refused the same way, by the constructor's own
  ``ValueError`` when ``load_checkpoint`` rebuilds the model: a file saved
  with ``embedding_rotation="Z"`` held a constant quantum layer (#212) and
  no longer loads.

``known_args`` makes the added-argument case explicit.  A name missing from
the config but absent from ``known_args`` too was added after the file was
written, and is filled from ``_LEGACY_DEFAULTS``.  A name missing from the
config although ``known_args`` lists it means the config was edited or
truncated, and is refused rather than filled.  For a file without
``known_args`` the two cannot be told apart, and every missing name in
``_LEGACY_DEFAULTS`` is filled.

Only primitives and tensors are stored, so the file loads with
``torch.load(weights_only=True)``: loading a checkpoint never executes code
from it, and only classes in ``hqnn_forge.models`` can be rebuilt.
"""

from __future__ import annotations

import inspect
import os
import warnings
from typing import TYPE_CHECKING, Any

import torch

import hqnn_forge
from hqnn_forge.initializers.restricted_variance import _not_restricting_ignored

if TYPE_CHECKING:
    # Type-only: importing this at runtime would be circular, since
    # hqnn_forge.models imports hqnn_forge.utils.
    from hqnn_forge.models.base import ClassifierBase

#: Bumped whenever the dict layout above changes incompatibly.
FORMAT_VERSION: int = 1

#: Constructor arguments whose override cannot invalidate the stored weights:
#: the two simulator knobs, plus the train-mode-only regularisers --
#: ``dropout_p`` (``nn.Dropout`` has no parameters of its own) and the
#: training-noise options, which only replace the circuit in train mode, and
#: ``shots``, which samples the same circuit, and ``seed``, which only seeds that
#: sampling.  None of them touches a weight.
#: ``load_checkpoint`` takes these without an opt-in; every other argument
#: describes the circuit the weights were trained in, so overriding it needs
#: ``allow_architecture_override=True``.  That includes ``init_strategy`` and
#: ``init_seed``, although the loaded state dict overwrites what they draw:
#: they record where the initial weights came from, and an override would
#: falsify that.
WEIGHT_SAFE_ARGS: frozenset[str] = frozenset(
    {
        "device_name",
        "diff_method",
        "dropout_p",
        "noise_level",
        "noise_position",
        "noise_method",
        "noise_trajectories",
        "shots",
        "noise_channel",
        "seed",
    }
)

#: Constructor arguments the classifiers have gained since checkpoints were
#: first written, mapped to the behaviour that predates each one.  A config
#: missing one of these, in a file whose ``known_args`` is absent or does not
#: list it, is a checkpoint older than the argument, and ``load_checkpoint``
#: fills it from here -- with a warning, never silently.
#: The values are written out rather than read from the signature on purpose:
#: they must stay the *old* behaviour even if the constructor default changes,
#: and every addition to this table is then a deliberate line in a diff.  A
#: config missing anything else is a broken checkpoint and still raises.
_LEGACY_DEFAULTS: dict[str, Any] = {
    "embedding_rotation": "X",  # added with the published-SHNN options; before
    "entangler": "ring",  # them the circuit was RX + CNOT ring + Rot,
    "readout": "all",  # read out on every qubit, with a tanh encoder
    "encoder_activation": "tanh",
    "init_std": 0.1,  # inert unless init_strategy="normal"
    "noise_level": 0.0,  # training-time depolarizing noise: none
    "noise_position": "all",
    "noise_method": "density",  # the only method before Pauli trajectories
    "noise_trajectories": 1,
    "init_seed": None,  # weights drawn from the global RNG; inert once loaded
    "classical_encoder": None,  # the built-in Linear encoder
    "trainable_input_scaling": False,  # added with encoding_type="reuploading"
    "shots": None,  # exact expectation values
    "noise_channel": "depolarizing",  # the only channel before #313
    "seed": None,  # the device's generator unseeded, as before #354
}

#: Constructor arguments added deliberately without a legacy default: no value
#: reproduces the model an older checkpoint holds, so a checkpoint that predates
#: one is refused.  See "Compatibility rules" in the module docstring.
_NO_LEGACY_DEFAULT: frozenset[str] = frozenset()

#: Set by ``load_checkpoint`` on a model it rebuilt under a forced
#: architecture override, and refused by ``save_checkpoint``.  Without it, one
#: forced load followed by a save yields a checkpoint that needs no override to
#: read back and so can never be caught again.
_FORCED_OVERRIDES_ATTR = "_forced_overrides"

PathLike = str | os.PathLike[str]


def _registry() -> dict[str, type[ClassifierBase]]:
    # Imported lazily: hqnn_forge.models imports hqnn_forge.utils, so a
    # module-level import here would be circular.
    from hqnn_forge import models

    # Every concrete classifier; each records its constructor arguments and
    # exposes them through ClassifierBase.get_config().
    return {
        name: getattr(models, name)
        for name in models.__all__
        if name not in ("ClassifierBase", "BinaryClassifierBase")
    }


def save_checkpoint(model: ClassifierBase, path: PathLike) -> None:
    """
    Write ``model``'s class, constructor arguments and weights to ``path``.

    Parameters
    ----------
    model:
        A classifier from ``hqnn_forge.models``.
    path:
        Destination file; overwritten if it exists.

    Raises
    ------
    TypeError
        If ``model`` is not one of the library's classifiers.
    ValueError
        If ``model`` has a custom ``classical_encoder``: a module is code, not
        data, and the file must load with ``weights_only=True``.  Save
        ``model.state_dict()`` and rebuild the model with the same encoder.
    ValueError
        If ``model`` came from a ``load_checkpoint`` with forced architecture
        overrides.  Its ``get_config()`` reports the overridden architecture
        while its weights were trained in the original one, so the checkpoint
        would be internally consistent and wrong -- and would load back without
        any override, past the guard that caught it the first time.
    """
    class_name = type(model).__name__
    if _registry().get(class_name) is not type(model):
        raise TypeError(
            f"save_checkpoint supports the classifiers in hqnn_forge.models "
            f"({', '.join(sorted(_registry()))}); got {type(model).__module__}.{class_name}."
        )
    config = model.get_config()
    modules = sorted(name for name, value in config.items() if isinstance(value, torch.nn.Module))
    if modules:
        raise ValueError(
            f"this {class_name} has a custom {', '.join(modules)}, which a checkpoint "
            f"cannot store: checkpoints hold primitives and tensors only, so they load "
            f"with torch.load(weights_only=True) and never execute code.  Save "
            f"model.state_dict() instead, and load it into a model rebuilt with the "
            f"same encoder class."
        )
    forced = getattr(model, _FORCED_OVERRIDES_ATTR, ())
    if forced:
        raise ValueError(
            f"this {class_name} was rebuilt by load_checkpoint with "
            f"allow_architecture_override=True, forcing {list(forced)}, so its "
            f"weights were trained in a different circuit than the config now "
            f"describes.  Saving it would write a checkpoint that reloads with no "
            f"override at all and is indistinguishable from a genuine one.  "
            f"Rebuild the architecture you want and retrain instead."
        )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "hqnn_forge_version": hqnn_forge.__version__,
        "class_name": class_name,
        "config": config,
        "known_args": sorted(_init_parameter_names(type(model))),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: PathLike,
    *,
    map_location: str | torch.device = "cpu",
    allow_version_mismatch: bool = False,
    allow_architecture_override: bool = False,
    **overrides: Any,
) -> ClassifierBase:
    """
    Rebuild a classifier saved with :func:`save_checkpoint`.

    Parameters
    ----------
    path:
        Checkpoint file.
    map_location:
        Where the rebuilt model ends up: the stored tensors are read onto this
        device and the model is moved there before its weights are loaded.
        Default: ``"cpu"``.  Note that this places the *classical* layers and
        the variational parameters only -- which simulator executes the
        circuit is chosen by the ``device_name`` argument, so moving a model to
        a GPU generally means overriding ``device_name`` as well.
    allow_version_mismatch:
        Load a checkpoint written by a different ``hqnn_forge`` version.  Off
        by default because weight layouts are not guaranteed stable across
        versions before 1.0.
    allow_architecture_override:
        Permit ``**overrides`` outside :data:`WEIGHT_SAFE_ARGS`.  Off by
        default: an architecture override makes the rebuilt model something
        other than the one that was saved, and the mismatch is not always
        loud.  ``encoding_type`` is the dangerous case -- ``QuantumEncodingLayer``
        and ``IQPEncodingLayer`` both register their weights at
        ``(n_layers, n_qubits, 3)``, so an ``"angle"`` checkpoint loaded as
        ``"iqp"`` fits, raises nothing, and predicts differently.  A model
        rebuilt this way is marked and :func:`save_checkpoint` refuses it, so
        the mismatch cannot be laundered into a fresh checkpoint.
    **overrides:
        Constructor arguments that replace the stored ones.  Without
        ``allow_architecture_override``, only :data:`WEIGHT_SAFE_ARGS`
        (``device_name``, ``diff_method``, ``dropout_p``, the five
        ``noise_*`` training-noise options, ``shots`` and ``seed``) may be given --
        typically to run a saved model on a different simulator or with a
        finite shot count, or to fine-tune it at a different dropout rate or
        training-noise level.

    Returns
    -------
    ClassifierBase
        The rebuilt model with the saved weights, in eval mode.

    Raises
    ------
    ValueError
        If the file is not a checkpoint or is incomplete, on an unknown format
        version, a library version mismatch (unless allowed), an unknown class,
        an override outside :data:`WEIGHT_SAFE_ARGS` without
        ``allow_architecture_override``, a malformed ``known_args``, or a config
        with unexpected fields or with missing ones that cannot be filled: a
        name outside :data:`_LEGACY_DEFAULTS`, a name in
        :data:`_NO_LEGACY_DEFAULT`, or a name the file's ``known_args`` lists
        (the config was edited or truncated).

    Warns
    -----
    RuntimeWarning
        If the stored config is missing constructor arguments added after it
        was written.  They are filled from :data:`_LEGACY_DEFAULTS`, the
        behaviour from before each argument existed, and named in the warning,
        so a checkpoint from before an option was introduced still rebuilds the
        model it holds.
    RuntimeError
        If the stored weights do not fit the rebuilt architecture.
    """
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except OSError:
        # Missing file, unreadable path: the caller's problem, not a malformed
        # checkpoint.
        raise
    except Exception as exc:
        # torch.load fails its own way on anything that is not a torch archive
        # (a text file raises KeyError from the zip reader), so the structural
        # check below is never reached for a genuinely foreign file.
        raise ValueError(
            f"{os.fspath(path)!r} could not be read as a torch archive, so it is "
            f"not an hqnn_forge checkpoint."
        ) from exc

    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(f"{os.fspath(path)!r} is not an hqnn_forge checkpoint.")

    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {payload['format_version']} is not supported; "
            f"this hqnn_forge reads format version {FORMAT_VERSION}."
        )

    if "state_dict" not in payload:
        raise ValueError(f"{os.fspath(path)!r} has no 'state_dict'; the checkpoint is incomplete.")

    saved_version = payload.get("hqnn_forge_version")
    if saved_version != hqnn_forge.__version__ and not allow_version_mismatch:
        raise ValueError(
            f"checkpoint was written by hqnn_forge {saved_version}, this is "
            f"{hqnn_forge.__version__}.  Pass allow_version_mismatch=True to load it anyway."
        )

    class_name = payload.get("class_name")
    registry = _registry()
    if class_name not in registry:
        raise ValueError(
            f"checkpoint holds unknown class {class_name!r}; expected one of {sorted(registry)}."
        )
    cls = registry[class_name]

    expected = _init_parameter_names(cls)
    unknown_overrides = sorted(set(overrides) - expected)
    if unknown_overrides:
        raise ValueError(f"unknown constructor arguments for {class_name}: {unknown_overrides}.")

    architecture_overrides = sorted(set(overrides) - WEIGHT_SAFE_ARGS)
    if architecture_overrides and not allow_architecture_override:
        raise ValueError(
            f"{architecture_overrides} describe the circuit the saved weights were "
            f"trained in, so overriding them rebuilds a different model -- and the "
            f"mismatch is not always caught: an 'angle' checkpoint loads without "
            f"error as encoding_type='iqp' and predicts differently.  Only "
            f"{sorted(WEIGHT_SAFE_ARGS)} may be overridden; pass "
            f"allow_architecture_override=True if that is really what you want."
        )

    config = dict(payload.get("config") or {})
    config.update(overrides)

    missing = sorted(expected - set(config))
    unexpected = sorted(set(config) - expected)

    # Only an absent key means a file from before known_args; a key holding
    # None or anything else that is not a list of names is malformed, not old.
    if "known_args" in payload:
        known_args = payload["known_args"]
        if not isinstance(known_args, list) or not all(isinstance(a, str) for a in known_args):
            raise ValueError(
                f"checkpoint 'known_args' must be a list of names; got {known_args!r}."
            )
        # The writer had these arguments and still stored no value for them:
        # the config was edited or truncated, not written before they existed.
        dropped = [name for name in missing if name in known_args]
        if dropped:
            raise ValueError(
                f"checkpoint config lacks {dropped}, although the version that wrote it "
                f"had {'that argument' if len(dropped) == 1 else 'those arguments'}; the "
                f"file was edited or is corrupt, so it is not filled from defaults."
            )
    # Written before these arguments existed.  One added deliberately without
    # a legacy default cannot be rebuilt; refuse it by name.
    breaking = [name for name in missing if name in _NO_LEGACY_DEFAULT]
    if breaking:
        raise ValueError(
            f"checkpoint predates {breaking} on {class_name}, which have no legacy "
            f"default: no value rebuilds the model it holds, so it cannot be loaded."
        )

    # A checkpoint written before the constructor gained an argument does not
    # carry it.  Filling it from _LEGACY_DEFAULTS rebuilds the model that was
    # saved, since those values are what the circuit did before the argument
    # existed -- but loudly, so a reload of an old checkpoint is never a silent
    # change of architecture.  Anything missing that is not in the table is a
    # broken config and still raises below.
    back_filled = {name: _LEGACY_DEFAULTS[name] for name in missing if name in _LEGACY_DEFAULTS}
    if back_filled:
        config.update(back_filled)
        missing = [name for name in missing if name not in back_filled]
        warnings.warn(
            f"checkpoint predates {sorted(back_filled)} on {class_name}; "
            f"rebuilding it with {back_filled}, the behaviour from before those "
            f"arguments existed, so it is the model that was saved.  Re-save it "
            f"to pin them in the config.",
            RuntimeWarning,
            stacklevel=2,
        )

    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(
            f"checkpoint config does not match {class_name}'s constructor: {'; '.join(details)}."
        )

    # The constructor's weight draw is overwritten below, so a warning about
    # it (a toy size whose restricted init restricts nothing) is noise here.
    with _not_restricting_ignored():
        model = cls(**config)
    # Before load_state_dict, so the stored tensors -- already read onto
    # map_location -- are copied into parameters that live there too.  Building
    # the model alone always puts it on the CPU.
    model.to(map_location)
    model.load_state_dict(payload["state_dict"])
    if architecture_overrides:
        # Only reachable with allow_architecture_override=True.  The model now
        # reports an architecture its weights were not trained in, and nothing
        # in a state dict can show that, so carry the fact on the instance and
        # let save_checkpoint refuse it rather than let it become a checkpoint
        # that loads back clean.
        setattr(model, _FORCED_OVERRIDES_ATTR, tuple(architecture_overrides))
    model.eval()
    return model


def _init_parameter_names(cls: type) -> set[str]:
    """
    Keyword names of ``cls.__init__``.

    A checkpoint is expected to carry every one of them, not only those
    without a default: get_config records them all, and a default that changed
    between versions would otherwise silently change the rebuilt model.  A
    checkpoint older than an argument is the one exception: unless the file's
    ``known_args`` lists the name, :func:`load_checkpoint` fills it from
    :data:`_LEGACY_DEFAULTS` with a warning rather than in silence.
    """
    return {
        name
        for name, p in inspect.signature(cls).parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
