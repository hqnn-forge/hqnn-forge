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


@pytest.mark.parametrize("role", ["func", "class", "meth", "data", "attr", "exc"])
def test_non_module_roles_resolve_a_shadowed_member(package: object, role: str) -> None:
    obj = package["diagnostics.fisher"]  # type: ignore[index]
    assert convert(f":{role}:`expressibility`", obj, package) == (  # type: ignore[arg-type]
        "[`expressibility`][hqnn_forge.diagnostics.expressibility.expressibility]"
    )


@pytest.mark.parametrize("role", ["mod", "obj"])
def test_module_and_object_roles_keep_the_module(package: object, role: str) -> None:
    obj = package["diagnostics.fisher"]  # type: ignore[index]
    assert convert(f":{role}:`expressibility`", obj, package) == (  # type: ignore[arg-type]
        "[`expressibility`][hqnn_forge.diagnostics.expressibility]"
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("func", "hqnn_forge.diagnostics.expressibility.expressibility"),
        ("mod", "hqnn_forge.diagnostics.expressibility"),
    ],
)
def test_a_qualified_target_passes_the_role_on(package: object, role: str, expected: str) -> None:
    obj = package["diagnostics.fisher"]  # type: ignore[index]
    target = "hqnn_forge.diagnostics.expressibility"
    assert convert(f":{role}:`{target}`", obj, package) == f"[`{target}`][{expected}]"  # type: ignore[arg-type]


def test_a_shadowed_alias_resolves_to_its_definition() -> None:
    package = griffe.Module("hqnn_forge")
    module = griffe.Module("work")
    function = griffe.Function("implementation")
    package.set_member("work", module)
    package.set_member("implementation", function)
    module.set_member("work", griffe.Alias("work", function))
    assert convert(":func:`work`", package, package) == ("[`work`][hqnn_forge.implementation]")


@pytest.mark.parametrize("member_kind", ["module", "unresolved_alias"])
def test_an_unusable_shadowed_member_continues_to_the_next_scope(member_kind: str) -> None:
    modules = griffe.ModulesCollection()
    package = griffe.Module("hqnn_forge", modules_collection=modules)
    modules.set_member("hqnn_forge", package)
    scope = griffe.Module("scope")
    module = griffe.Module("work")
    package.set_member("scope", scope)
    package.set_member("work", griffe.Function("work"))
    scope.set_member("work", module)
    member = (
        griffe.Module("work")
        if member_kind == "module"
        else griffe.Alias("work", "hqnn_forge.missing")
    )
    module.set_member("work", member)
    assert convert(":func:`work`", scope, package) == "[`work`][hqnn_forge.work]"


def test_a_module_without_a_same_named_member_keeps_existing_resolution(package: object) -> None:
    obj = package["diagnostics"]  # type: ignore[index]
    assert convert(":func:`fisher`", obj, package) == (  # type: ignore[arg-type]
        "[`fisher`][hqnn_forge.diagnostics.fisher]"
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
