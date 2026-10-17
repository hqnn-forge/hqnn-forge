"""Exact inventory-check regressions, run with the docs dependency group."""

from __future__ import annotations

import importlib.util
import zlib
from pathlib import Path

import pytest

griffe = pytest.importorskip("griffe")

SCRIPT = Path(__file__).parents[1] / ".github/scripts/check_docs_inventory.py"
spec = importlib.util.spec_from_file_location("check_docs_inventory", SCRIPT)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture
def fixture_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    package = tmp_path / "hqnn_forge"
    package.mkdir()
    (package / "__init__.py").write_text('__all__ = ["__version__"]\n__version__ = "test"\n')
    docs = tmp_path / "docs"
    (docs / "api").mkdir(parents=True)
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.chdir(tmp_path)
    return package, docs, site


def write_inventory(site: Path, *names: str) -> None:
    header = b"# Sphinx inventory version 2\n# Project: fixture\n# Version: 1\n# zlib\n"
    entries = "".join(f"{name} py:function 1 api/#anchor -\n" for name in names)
    (site / "objects.inv").write_bytes(header + zlib.compress(entries.encode()))


def test_a_rendered_module_without_all_fails(fixture_site: tuple[Path, Path, Path]) -> None:
    package, docs, site = fixture_site
    (package / "sample.py").write_text("def public(): pass\n")
    (docs / "api/sample.md").write_text("::: hqnn_forge.sample\n")
    write_inventory(site, "hqnn_forge.sample.public")
    assert checker.missing(docs, site) == ["sample.md: hqnn_forge.sample has no __all__"]


def test_missing_exports_are_reported_in_name_order(fixture_site: tuple[Path, Path, Path]) -> None:
    package, docs, site = fixture_site
    (package / "sample.py").write_text(
        '__all__ = ["zebra", "present", "alpha"]\ndef alpha(): pass\n'
        "def present(): pass\ndef zebra(): pass\n"
    )
    (docs / "api/sample.md").write_text("::: hqnn_forge.sample\n")
    write_inventory(site, "hqnn_forge.sample.present")
    assert checker.missing(docs, site) == [
        "sample.md: hqnn_forge.sample.alpha is exported but not documented",
        "sample.md: hqnn_forge.sample.zebra is exported but not documented",
    ]


def test_empty_all_is_explicit_and_imports_are_not_api(
    fixture_site: tuple[Path, Path, Path],
) -> None:
    package, docs, site = fixture_site
    (package / "sample.py").write_text("import os\n__all__ = []\n")
    (docs / "api/sample.md").write_text("::: hqnn_forge.sample\n")
    write_inventory(site)
    assert checker.missing(docs, site) == []


def test_a_public_definition_left_out_of_all_fails(
    fixture_site: tuple[Path, Path, Path],
) -> None:
    package, docs, site = fixture_site
    (package / "sample.py").write_text(
        "import os\nfrom math import pi\n__all__ = ['present']\n"
        "LIMIT = 3\n_private = 1\ndef present(): pass\ndef forgotten(): pass\n"
    )
    (docs / "api/sample.md").write_text("::: hqnn_forge.sample\n")
    write_inventory(site, "hqnn_forge.sample.present", "hqnn_forge.sample.forgotten")
    assert checker.missing(docs, site) == [
        "sample.md: hqnn_forge.sample.LIMIT is defined but not in __all__",
        "sample.md: hqnn_forge.sample.forgotten is defined but not in __all__",
    ]


def test_a_public_module_without_a_page_still_fails(fixture_site: tuple[Path, Path, Path]) -> None:
    package, docs, site = fixture_site
    (package / "sample.py").write_text("__all__ = []\n")
    write_inventory(site)
    assert checker.missing(docs, site) == [
        "hqnn_forge.sample is a public module with no page under docs/api/"
    ]


def test_a_submodule_version_is_not_exempt(fixture_site: tuple[Path, Path, Path]) -> None:
    package, docs, site = fixture_site
    (package / "sample.py").write_text('__all__ = ["__version__"]\n__version__ = "test"\n')
    (docs / "api/sample.md").write_text("::: hqnn_forge.sample\n")
    write_inventory(site)
    assert checker.missing(docs, site) == [
        "sample.md: hqnn_forge.sample.__version__ is exported but not documented"
    ]


def test_nested_rendered_module_is_checked(fixture_site: tuple[Path, Path, Path]) -> None:
    package, docs, site = fixture_site
    nested = package / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("__all__ = []\n")
    (nested / "sample.py").write_text("def public(): pass\n")
    (docs / "api/nested.md").write_text("::: hqnn_forge.nested\n::: hqnn_forge.nested.sample\n")
    write_inventory(site)
    assert checker.missing(docs, site) == ["nested.md: hqnn_forge.nested.sample has no __all__"]


@pytest.mark.parametrize("render_root", [False, True])
def test_only_root_version_is_exempt(
    fixture_site: tuple[Path, Path, Path], render_root: bool
) -> None:
    package, docs, site = fixture_site
    (package / "__init__.py").write_text(
        '__all__ = ["__version__", "public"]\n__version__ = "test"\ndef public(): pass\n'
    )
    if render_root:
        (docs / "api/root.md").write_text("::: hqnn_forge\n")
    write_inventory(site)
    expected = (
        "root.md: hqnn_forge.public is exported but not documented"
        if render_root
        else "hqnn_forge.public is exported but not documented"
    )
    assert checker.missing(docs, site) == [expected]
    write_inventory(site, "hqnn_forge.public")
    assert checker.missing(docs, site) == []


def test_cli_returns_failure_and_prints_each_problem(
    fixture_site: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    package, docs, site = fixture_site
    (package / "sample.py").write_text("def public(): pass\n")
    (docs / "api/sample.md").write_text("::: hqnn_forge.sample\n")
    write_inventory(site)
    assert checker.main() == 1
    assert capsys.readouterr().err == "::error::sample.md: hqnn_forge.sample has no __all__\n"
