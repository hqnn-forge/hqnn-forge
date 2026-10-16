"""
docs/test_griffe_sphinx_roles.py
================================
The docs site's griffe extension (#322).  Run by the docs CI job
(``pytest docs/``), where griffe is installed; skipped elsewhere.
"""

from __future__ import annotations

import re
import subprocess
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


@pytest.mark.parametrize(
    "scope", ["benchmark", "benchmark.run_benchmark", "benchmark.BenchmarkResult"]
)
def test_a_missing_member_of_a_local_class_is_still_linked(package: object, scope: str) -> None:
    obj = package[scope]  # type: ignore[index]
    assert convert(":attr:`BenchmarkResult.missing_member`", obj, package) == (  # type: ignore[arg-type]
        "[`BenchmarkResult.missing_member`][hqnn_forge.benchmark.BenchmarkResult.missing_member]"
    )


def test_an_existing_class_member_resolves_to_its_canonical_path(package: object) -> None:
    obj = package["models"]  # type: ignore[index]
    assert convert(":meth:`HybridBinaryClassifier.published_shnn`", obj, package) == (  # type: ignore[arg-type]
        "[`HybridBinaryClassifier.published_shnn`]"
        "[hqnn_forge.models.hybrid_classifier.HybridBinaryClassifier.published_shnn]"
    )


@pytest.mark.parametrize("target", ["torch.nn.Module", "np.ndarray"])
def test_foreign_dotted_targets_stay_plain_code(package: object, target: str) -> None:
    # benchmark imports numpy as np; an imported foreign root is still foreign.
    obj = package["benchmark.run_benchmark"]  # type: ignore[index]
    assert convert(f":class:`{target}`", obj, package) == f"`{target}`"  # type: ignore[arg-type]


def test_a_missing_member_is_linked_under_its_owner(package: object) -> None:
    # Under the module where noise is defined, not the module referencing it.
    obj = package["benchmark"]  # type: ignore[index]
    assert resolve("noise.missing", obj, package) == "hqnn_forge.noise.missing"  # type: ignore[arg-type]
    # Under the longest existing prefix, however deep the missing part.
    assert resolve("BenchmarkResult.missing.deeper", obj, package) == (  # type: ignore[arg-type]
        "hqnn_forge.benchmark.BenchmarkResult.missing.deeper"
    )


@pytest.mark.parametrize(
    ("scope", "target"),
    [
        # No scope holds pathlib; data.credit_card imports only Path from it.
        ("data.credit_card", "pathlib.Path.is_dir"),
        # Inherited from torch.nn.Module, which griffe does not load.
        ("models.hybrid_classifier", "HybridBinaryClassifier.parameters"),
        # records is a list; griffe models no members of an attribute's value.
        ("benchmark", "BenchmarkResult.records.append"),
        ("benchmark.BenchmarkResult", "records.append"),
        # Implicit dunders of a class with no bases.
        ("benchmark", "BenchmarkResult.__repr__"),
        ("benchmark", "BenchmarkResult.__eq__"),
    ],
)
def test_members_griffe_cannot_see_stay_plain_code(
    package: object, scope: str, target: str
) -> None:
    obj = package[scope]  # type: ignore[index]
    assert resolve(target, obj, package) is None  # type: ignore[arg-type]


def test_a_foreign_name_re_exported_in_the_package_stays_plain_code(tmp_path: Path) -> None:
    root = tmp_path / "hqnn_forge"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "types.py").write_text("import numpy as np\nfrom torch import Tensor\n")
    (root / "mod.py").write_text(
        "from hqnn_forge import types\nfrom hqnn_forge.types import np\n"
        "from .types import Tensor\nfrom other import Foreign\n\nclass Local:\n    pass\n"
    )
    other = tmp_path / "other"
    other.mkdir()
    (other / "__init__.py").write_text("class Foreign:\n    pass\n")
    # Paths, not names: the name finds the installed hqnn_forge first.  One
    # loader, so the import of the loaded package other resolves.
    loader = griffe.GriffeLoader(search_paths=[str(tmp_path)])
    synthetic = loader.load(root)
    loader.load(other)
    obj = synthetic["mod"]
    assert resolve("np.ndarray", obj, synthetic) is None
    assert resolve("Tensor.shape", obj, synthetic) is None
    # The whole target exists, as an import from another package.
    assert resolve("types.Tensor", obj, synthetic) is None
    assert resolve("types.np", obj, synthetic) is None
    # Resolved, but to another package's object, which is not rendered.
    assert resolve("Foreign", obj, synthetic) is None
    assert resolve("Foreign.missing", obj, synthetic) is None
    assert resolve("Local.missing", obj, synthetic) == "hqnn_forge.mod.Local.missing"


def test_a_missing_class_member_fails_a_strict_build(package: object, tmp_path: Path) -> None:
    # A minimal site isolates autorefs validation from the project's rendered API.
    docs = tmp_path / "docs"
    docs.mkdir()
    obj = package["benchmark"]  # type: ignore[index]
    (docs / "index.md").write_text(
        convert(":attr:`BenchmarkResult.missing_member`", obj, package)  # type: ignore[arg-type]
    )
    config = tmp_path / "mkdocs.yml"
    config.write_text("site_name: Missing member\ndocs_dir: docs\nplugins:\n  - autorefs\n")
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--config-file", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    # The target only: autorefs' wording around it is not ours to pin.
    assert "hqnn_forge.benchmark.BenchmarkResult.missing_member" in result.stderr


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
