"""
docs/test_griffe_sphinx_roles.py
================================
The docs site's griffe extension (#322).  Run by the docs CI job
(``pytest docs/``), where griffe is installed; skipped elsewhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

griffe = pytest.importorskip("griffe")

sys.path.insert(0, str(Path(__file__).parent))
from griffe_sphinx_roles import DOCSTRING_STYLE, SphinxRoles, comment_docstring, convert, resolve


@pytest.fixture(scope="module")
def package() -> object:
    return griffe.load("hqnn_forge", search_paths=[str(Path(__file__).parents[1])])


def test_a_qualified_re_export_resolves_to_where_it_is_defined(package: object) -> None:
    obj = package["noise"]  # type: ignore[index]
    assert (
        resolve("hqnn_forge.models.HybridBinaryClassifier", obj, package)  # type: ignore[arg-type]
        == "hqnn_forge.models.hybrid_classifier.HybridBinaryClassifier"
    )


def test_an_unqualified_name_resolves_in_the_enclosing_scopes(package: object) -> None:
    fn = package["noise.noise_sweep"]  # type: ignore[index]
    # A sibling function of the same module.
    assert (
        resolve("apply_depolarizing_noise", fn, package)  # type: ignore[arg-type]
        == "hqnn_forge.noise.apply_depolarizing_noise"
    )
    # A method from inside its class.
    cls = package["models.hybrid_classifier.HybridBinaryClassifier"]  # type: ignore[index]
    assert resolve("published_shnn", cls, package) == (  # type: ignore[arg-type]
        "hqnn_forge.models.hybrid_classifier.HybridBinaryClassifier.published_shnn"
    )


def test_private_and_foreign_targets_become_plain_code(package: object) -> None:
    obj = package["utils.checkpoint"]  # type: ignore[index]
    assert resolve("_LEGACY_DEFAULTS", obj, package) is None  # type: ignore[arg-type]
    assert resolve("torch.nn.Module", obj, package) is None  # type: ignore[arg-type]


def test_an_unresolvable_name_is_still_linked_so_the_build_fails(package: object) -> None:
    obj = package["noise"]  # type: ignore[index]
    assert resolve("no_such_function", obj, package) == "hqnn_forge.noise.no_such_function"  # type: ignore[arg-type]


def test_convert_rewrites_roles_and_honours_the_tilde(package: object) -> None:
    obj = package["noise"]  # type: ignore[index]
    text = (
        "See :func:`~hqnn_forge.noise.noise_sweep`, :class:`torch.nn.Module` and :data:`_private`."
    )
    assert convert(text, obj, package) == (  # type: ignore[arg-type]
        "See [`noise_sweep`][hqnn_forge.noise.noise_sweep], `torch.nn.Module` and `_private`."
    )


def test_comment_docstrings() -> None:
    lines = ["x = 1", "#: First line.", "#: Second line.", "NAME = 2", "#:not spaced", "OTHER = 3"]
    assert comment_docstring(lines, 4) == "First line.\nSecond line."
    assert comment_docstring(lines, 6) == "not spaced"
    assert comment_docstring(lines, 1) is None


def test_comment_docstrings_are_parsed_as_numpy_style() -> None:
    # Rendered as a member of the evaluation page, METRICS never gets its parser
    # from mkdocstrings, so the extension has to set it.
    package = griffe.load(
        "hqnn_forge",
        search_paths=[str(Path(__file__).parents[1])],
        extensions=griffe.load_extensions(SphinxRoles()),
    )
    docstring = package["evaluation.METRICS"].docstring
    assert docstring is not None
    assert docstring.parser == griffe.Parser.numpy


def test_the_docstring_style_matches_mkdocs_yml() -> None:
    config = (Path(__file__).parents[1] / "mkdocs.yml").read_text()
    styles = re.findall(r"^\s*docstring_style:\s*(\w+)", config, re.MULTILINE)
    assert styles == [DOCSTRING_STYLE]
