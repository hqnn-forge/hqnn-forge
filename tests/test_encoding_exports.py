"""
tests/test_encoding_exports.py
==============================
The public surface of ``hqnn_forge.encoding`` (#169): a flat package export.
Every encoder's layer and QNode factory is importable from the package, and a
new encoder submodule that forgets to register its layer fails here rather
than being reachable only through its submodule.  That the package docstring
lists exactly ``__all__`` is checked for every package in
``test_exported_symbols_docstring.py`` (#380).
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

from torch import nn

import hqnn_forge.encoding as encoding

SUBMODULES = [
    importlib.import_module(f"hqnn_forge.encoding.{m.name}")
    for m in pkgutil.iter_modules(encoding.__path__)
]


def test_iqp_is_importable_from_the_package() -> None:
    """The import the issue reported as an ImportError."""
    from hqnn_forge.encoding import IQPEncodingLayer, build_iqp_qnode
    from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer as FromSubmodule

    assert IQPEncodingLayer is FromSubmodule
    assert callable(build_iqp_qnode)


def test_every_encoder_layer_and_factory_is_exported() -> None:
    """The convention for future encoders: every layer and ``build_*_qnode`` factory."""
    missing = [
        f"{module.__name__}.{name}"
        for module in SUBMODULES
        for name, obj in vars(module).items()
        if getattr(obj, "__module__", None) == module.__name__
        and (
            (inspect.isclass(obj) and issubclass(obj, nn.Module))
            or (inspect.isfunction(obj) and re.fullmatch(r"build_\w+_qnode", name))
        )
        and name not in encoding.__all__
    ]
    assert not missing, missing
