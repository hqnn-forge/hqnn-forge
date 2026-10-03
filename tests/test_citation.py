"""
tests/test_citation.py
======================
``CITATION.cff`` stays in step with ``pyproject.toml`` (#203): a release that
bumps the package version without the citation file would make GitHub's
"Cite this repository" cite the wrong version.  Parsed line by line, so the
test needs no YAML library.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _cff_field(name: str) -> str:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    assert match, f"CITATION.cff has no top-level {name!r}"
    return match.group(1).strip("\"'")


def _project() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]


def test_version_matches_pyproject() -> None:
    assert _cff_field("version") == _project()["version"]


def test_licence_matches_pyproject() -> None:
    assert _cff_field("license") == _project()["license"]["text"]


def test_format_version() -> None:
    assert _cff_field("cff-version") == "1.2.0"
