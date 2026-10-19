"""
tests/test_exported_symbols_docstring.py
========================================
Every public subpackage, and the top-level package, documents its public API
in an "Exported symbols" section of its docstring, and that section lists
exactly the names in ``__all__`` (#380).  It generalises the check #242 added
for ``hqnn_forge.encoding`` alone.
"""

from __future__ import annotations

import importlib
import pkgutil
import re

import pytest

import hqnn_forge

HEADING = "Exported symbols"
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def public_packages() -> list[str]:
    """The top-level package and every subpackage whose name is public."""
    subpackages = sorted(
        f"hqnn_forge.{info.name}"
        for info in pkgutil.iter_modules(hqnn_forge.__path__)
        if info.ispkg and not info.name.startswith("_")
    )
    return ["hqnn_forge", *subpackages]


def listed_symbols(doc: str, owner: str) -> set[str]:
    """The names listed in the "Exported symbols" section of a docstring.

    An entry starts at column 0 with one name, or several separated by commas,
    optionally followed by two or more spaces and a description.  Lines that
    start with whitespace continue the previous entry, a line ending in a colon
    is a group heading, and blank lines and the underline are skipped.  Any
    other line raises, naming it, instead of being guessed at.
    """
    lines = doc.splitlines()
    start = next((i for i, line in enumerate(lines) if line.rstrip() == HEADING), None)
    if start is None:
        raise AssertionError(f"{owner}: the docstring has no {HEADING!r} section")
    names: set[str] = set()
    for number, line in enumerate(lines[start + 1 :], start=start + 2):
        if not line.strip() or set(line.strip()) == {"-"}:
            continue
        if line[0].isspace():
            continue
        if line.rstrip().endswith(":") and "  " not in line:
            continue
        head = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]
        for name in (part.strip() for part in head.split(",")):
            if not IDENTIFIER.fullmatch(name):
                raise AssertionError(
                    f"{owner}: cannot read an entry from line {number} of the docstring: {line!r}"
                )
            names.add(name)
    return names


@pytest.mark.parametrize("package", public_packages())
def test_docstring_lists_exactly_all(package: str) -> None:
    module = importlib.import_module(package)
    listed = listed_symbols(module.__doc__ or "", package)
    exported = set(module.__all__)
    assert listed == exported, (
        f"docstring only: {sorted(listed - exported)}; __all__ only: {sorted(exported - listed)}"
    )


SECTION = "Exported symbols\n----------------\n"


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ("Alpha  one line\n", {"Alpha"}),
        ("Alpha  wrapped\n    onto a second line\n", {"Alpha"}),
        ("Alpha\n    description below\n", {"Alpha"}),
        ("Alpha, Beta, Gamma\n    one description\n", {"Alpha", "Beta", "Gamma"}),
        ("Group:\n\nAlpha  one\n\nMore things:\n\nBeta  two\n", {"Alpha", "Beta"}),
    ],
)
def test_the_parser_reads_every_entry_layout(entries: str, expected: set[str]) -> None:
    assert listed_symbols("Intro.\n\n" + SECTION + entries, "pkg") == expected


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ("Intro.\n", "no 'Exported symbols' section"),
        (SECTION + "Alpha one space only\n", "cannot read an entry from line 3"),
    ],
)
def test_the_parser_fails_clearly_instead_of_guessing(doc: str, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        listed_symbols(doc, "pkg")
