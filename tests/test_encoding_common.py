"""
tests/test_encoding_common.py
=============================
The shared QNode plumbing lives in hqnn_forge.encoding._common (#306).
"""

from __future__ import annotations

import ast
from pathlib import Path

import hqnn_forge.encoding._common as common
import hqnn_forge.encoding.angle_embedding as angle

ENCODING = Path(common.__file__).parent


def _imports(path: Path) -> list[tuple[str, str]]:
    """``(module, name)`` for every ``from hqnn_forge... import name`` in ``path``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hqnn_forge"):
            found += [(node.module or "", alias.name) for alias in node.names]
    return found


def test_no_encoding_module_imports_a_private_name_from_another() -> None:
    offenders = [
        f"{path.name}: from {module} import {name}"
        for path in sorted(ENCODING.glob("*.py"))
        for module, name in _imports(path)
        if module.startswith("hqnn_forge.encoding.") and name.startswith("_")
    ]
    assert not offenders, offenders


def test_encoders_take_the_plumbing_from_common_not_from_the_angle_layer() -> None:
    for name in ("amplitude_embedding.py", "iqp_embedding.py", "data_reuploading.py"):
        modules = {module for module, _ in _imports(ENCODING / name)}
        assert "hqnn_forge.encoding.angle_embedding" not in modules, name
        assert "hqnn_forge.encoding._common" in modules, name


def test_angle_embedding_still_exports_what_it_used_to() -> None:
    # Code written against the old location keeps working for one release,
    # including the underscored spellings, and they are the same objects.
    for name in (
        "FALLBACK_CHAIN",
        "DeviceName",
        "DiffMethod",
        "Entangler",
        "Readout",
        "RotationAxis",
        "apply_variational_layers",
        "check_inputs",
        "measure_z",
        "readout_wires",
        "reset_device_fallback",
        "validate_circuit_options",
        "variational_weight_shape",
    ):
        assert getattr(angle, name) is getattr(common, name), name
        assert name in angle.__all__
    assert angle._resolve_device is common.resolve_device
    assert angle._expand_batch_dimension is common.expand_batch_dimension
    assert angle._is_out_of_memory is common.is_out_of_memory
    assert angle._DEVICE_FAILURES is common.DEVICE_FAILURES


def test_angle_embedding_is_about_the_angle_embedding() -> None:
    # What is left defines the angle circuit, its QNode factory and its layer.
    defined = {
        node.name
        for node in ast.parse(Path(angle.__file__).read_text()).body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }
    assert defined == {
        "_make_angle_embedding_circuit",
        "build_encoding_qnode",
        "QuantumEncodingLayer",
    }
