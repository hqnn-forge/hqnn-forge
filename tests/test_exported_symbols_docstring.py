"""
tests/test_exported_symbols_docstring.py
========================================
Every public subpackage, and the top-level package, documents its public API
in an "Exported symbols" section of its docstring, and that section lists
exactly the names in ``__all__`` (#380). It generalises the check #242 added
for ``hqnn_forge.encoding`` alone.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections import Counter

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


def listed_symbols(doc: str, owner: str) -> dict[str, list[int]]:
    """Return each documented symbol and the line numbers where it occurs.

    An entry starts at column 0 with one name, or several separated by commas,
    optionally followed by two or more spaces and a description. Descriptions
    that wrap onto additional lines must use indented continuation lines.
    Indented lines before the first entry are rejected.

    A group heading ends in a colon and is followed by a blank line. Parsing
    stops at the next underlined section heading. Blank lines and the initial
    underline below ``Exported symbols`` are skipped. Any other malformed
    entry raises an assertion instead of being guessed at.
    """
    lines = doc.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.rstrip() == HEADING),
        None,
    )
    if start is None:
        raise AssertionError(f"{owner}: the docstring has no {HEADING!r} section")

    names: dict[str, list[int]] = {}

    for index in range(start + 1, len(lines)):
        line = lines[index]
        number = index + 1
        stripped = line.strip()

        if not stripped:
            continue

        # Skip the underline belonging to the Exported symbols heading.
        if index == start + 1 and set(stripped) in ({"-"}, {"="}):
            continue

        # Stop before a subsequent section heading and its underline.
        if (
            index > start + 1
            and index + 1 < len(lines)
            and set(lines[index + 1].strip()) in ({"-"}, {"="})
        ):
            break

        if set(stripped) in ({"-"}, {"="}):
            break

        if line[0].isspace():
            if not names:
                raise AssertionError(
                    f"{owner}: unexpected indented line {number} before the first entry: {line!r}"
                )
            continue

        # A colon denotes a group heading only when followed by a blank line.
        if stripped.endswith(":") and index + 1 < len(lines) and not lines[index + 1].strip():
            continue

        head = re.split(r"\s{2,}", stripped, maxsplit=1)[0]
        for name in (part.strip() for part in head.split(",")):
            if not IDENTIFIER.fullmatch(name):
                raise AssertionError(
                    f"{owner}: cannot read an entry from line {number} of the docstring: {line!r}"
                )
            names.setdefault(name, []).append(number)

    return names


def format_occurrences(occurrences: dict[str, list[int]]) -> str:
    """Format symbol names with their source line numbers."""
    return ", ".join(
        f"{name} (line{'s' if len(numbers) != 1 else ''} {', '.join(map(str, numbers))})"
        for name, numbers in sorted(occurrences.items())
    )


def assert_docstring_lists_exactly_all(
    doc: str,
    exported: list[str],
    package: str,
) -> None:
    """Assert that documented symbols match __all__, reporting source lines."""
    listed = listed_symbols(doc, package)
    exported_counts = Counter(exported)

    listed_names = set(listed)
    exported_names = set(exported_counts)

    docstring_only = {name: listed[name] for name in listed_names - exported_names}
    all_only = exported_names - listed_names
    duplicate_docstring_names = {
        name: numbers for name, numbers in listed.items() if len(numbers) > 1
    }
    duplicate_exported_names = {
        name: count for name, count in exported_counts.items() if count > 1
    }

    assert not (
        docstring_only or all_only or duplicate_docstring_names or duplicate_exported_names
    ), (
        f"{package}: "
        f"docstring only: {format_occurrences(docstring_only) or '[]'}; "
        f"__all__ only: {sorted(all_only)}; "
        f"duplicate docstring symbols: "
        f"{format_occurrences(duplicate_docstring_names) or '[]'}; "
        f"duplicate __all__ symbols: {duplicate_exported_names}"
    )


@pytest.mark.parametrize("package", public_packages())
def test_docstring_lists_exactly_all(package: str) -> None:
    module = importlib.import_module(package)
    assert_docstring_lists_exactly_all(
        module.__doc__ or "",
        list(module.__all__),
        package,
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
def test_the_parser_reads_every_entry_layout(
    entries: str,
    expected: set[str],
) -> None:
    assert set(listed_symbols("Intro.\n\n" + SECTION + entries, "pkg")) == expected


def test_the_parser_stops_at_the_next_section() -> None:
    doc = (
        SECTION
        + "Alpha  first symbol\n"
        + "Beta  second symbol\n"
        + "\n"
        + "Notes\n"
        + "-----\n"
        + "This is not an exported symbol.\n"
    )
    assert set(listed_symbols(doc, "pkg")) == {"Alpha", "Beta"}


def test_the_parser_reports_line_number_for_column_zero_wrapped_description() -> None:
    doc = SECTION + "Alpha  a symbol\nfrontier  wrapped description\n"

    with pytest.raises(AssertionError, match=re.escape("frontier (line 4)")):
        assert_docstring_lists_exactly_all(doc, ["Alpha"], "pkg")


def test_the_parser_records_duplicate_symbol_occurrences() -> None:
    doc = SECTION + "Alpha  first occurrence\nAlpha  second occurrence\n"
    assert listed_symbols(doc, "pkg") == {"Alpha": [3, 4]}


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ("Intro.\n", "no 'Exported symbols' section"),
        (SECTION + "Alpha one space only\n", "cannot read an entry from line 3"),
        (SECTION + "    Alpha\n", "indented line 3 before the first entry"),
        (
            SECTION + "Alpha the one-space entry, built from:\n",
            "cannot read an entry from line 3",
        ),
    ],
)
def test_the_parser_fails_clearly_instead_of_guessing(
    doc: str,
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=re.escape(message)):
        listed_symbols(doc, "pkg")
