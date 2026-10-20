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
from types import SimpleNamespace
from typing import Any

import pytest

import hqnn_forge

HEADING = "Exported symbols"
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
ENTRY_SEPARATOR = re.compile(r"\s{2,}")
UNDERLINE_CHARACTERS = frozenset("-=~^`:#*+_'\"")


def public_packages() -> list[str]:
    """The top-level package and every subpackage whose name is public."""
    subpackages = sorted(
        f"hqnn_forge.{info.name}"
        for info in pkgutil.iter_modules(hqnn_forge.__path__)
        if info.ispkg and not info.name.startswith("_")
    )
    return ["hqnn_forge", *subpackages]


def exported_names(module: Any, package: str) -> list[str]:
    """Return a package's exports, raising a clear error if __all__ is missing."""
    assert hasattr(module, "__all__"), (
        f"{package} has no __all__; every public package must define its exports"
    )
    return list(module.__all__)


def is_underline(line: str) -> bool:
    """Return whether a non-empty line consists of one repeated RST character."""
    stripped = line.strip()
    return (
        bool(stripped)
        and stripped[0] in UNDERLINE_CHARACTERS
        and all(character == stripped[0] for character in stripped)
    )


def is_entry(line: str) -> bool:
    """Return whether a line begins with a valid symbol entry."""
    stripped = line.strip()
    head = ENTRY_SEPARATOR.split(stripped, maxsplit=1)[0]
    return all(IDENTIFIER.fullmatch(part.strip()) for part in head.split(","))


def listed_symbols(doc: str, owner: str) -> dict[str, list[int]]:
    """Return each documented symbol and the line numbers where it occurs.

    Entries start at column 0 with one name, or several separated by commas,
    optionally followed by two or more spaces and a description. Descriptions
    may continue on indented lines immediately after an entry or another
    continuation line. Descriptions must be wrapped onto indented lines only.

    Group headings end in a colon and are followed by a blank line. Lines
    with a name/description separator are always entries. Parsing stops at
    the next underlined section heading, whose underline must be at least as
    long as its title. Other underline-like lines and malformed entries
    raise an assertion.
    """
    lines = doc.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.rstrip() == HEADING),
        None,
    )
    if start is None:
        raise AssertionError(f"{owner}: the docstring has no {HEADING!r} section")

    names: dict[str, list[int]] = {}
    can_continue = False

    for index in range(start + 1, len(lines)):
        line = lines[index]
        number = index + 1
        stripped = line.strip()

        if not stripped:
            can_continue = False
            continue

        # Skip the underline belonging to the Exported symbols heading.
        if index == start + 1 and is_underline(line):
            continue

        # Stop at a real section heading, but never mistake an entry for one.
        if index + 1 < len(lines) and is_underline(lines[index + 1]):
            underline = lines[index + 1].strip()
            if len(underline) >= len(stripped) and not ENTRY_SEPARATOR.search(stripped):
                break

        if is_underline(line):
            raise AssertionError(f"{owner}: unexpected underline on line {number}: {line!r}")

        if line[0].isspace():
            if not can_continue:
                raise AssertionError(
                    f"{owner}: unexpected indented line {number}; "
                    f"a continuation must immediately follow an entry "
                    f"or another continuation: {line!r}"
                )
            continue

        has_entry_separator = bool(ENTRY_SEPARATOR.search(stripped))

        # A colon line can be a group heading only when it has no entry
        # separator and is followed by a blank line and further content.
        if stripped.endswith(":") and not has_entry_separator:
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1

            has_blank_after = index + 1 < len(lines) and not lines[index + 1].strip()
            has_following_content = next_index < len(lines) and (
                lines[next_index][0].isspace() or is_entry(lines[next_index])
            )

            if has_blank_after and has_following_content:
                can_continue = False
                continue

        head = ENTRY_SEPARATOR.split(stripped, maxsplit=1)[0]
        for name in (part.strip() for part in head.split(",")):
            if not IDENTIFIER.fullmatch(name):
                raise AssertionError(
                    f"{owner}: cannot read an entry from line {number} of the docstring: {line!r}"
                )
            names.setdefault(name, []).append(number)

        can_continue = True

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
    exported_names_set = set(exported_counts)

    docstring_only = {name: listed[name] for name in listed_names - exported_names_set}
    all_only = exported_names_set - listed_names
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
    exported = exported_names(module, package)
    assert_docstring_lists_exactly_all(
        module.__doc__ or "",
        exported,
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
        ("Alpha  built from:\n\nBeta  two\n", {"Alpha", "Beta"}),
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


def test_an_indented_line_after_a_group_heading_raises() -> None:
    doc = SECTION + "Beta  b\n\nGroup:\n\n    Alpha  one\n"
    with pytest.raises(AssertionError, match="unexpected indented line"):
        listed_symbols(doc, "pkg")


@pytest.mark.parametrize(
    "doc",
    [
        SECTION + "Alpha  a\nBeta  b\n-----\nGamma  c\n",
        SECTION + "Alpha  a\n----\n",
        SECTION + "Alpha  a\n~~~~~\n",
        SECTION + "Alpha  a\nNotes\n---\nBeta  b\n",
    ],
)
def test_unexpected_underline_like_lines_raise(doc: str) -> None:
    with pytest.raises(AssertionError, match="unexpected underline"):
        listed_symbols(doc, "pkg")


def test_a_section_title_with_a_long_enough_underline_stops_parsing() -> None:
    doc = SECTION + "Alpha  a\nNotes\n-----\nBeta  b\n"
    assert listed_symbols(doc, "pkg") == {"Alpha": [3]}


def test_a_module_without_all_has_a_clear_error() -> None:
    module = SimpleNamespace(__doc__=SECTION)

    with pytest.raises(AssertionError, match=r"sample_package has no __all__"):
        exported_names(module, "sample_package")


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ("Intro.\n", "no 'Exported symbols' section"),
        (SECTION + "Alpha one space only\n", "cannot read an entry from line 3"),
        (SECTION + "    Alpha\n", "unexpected indented line 3"),
        (
            SECTION + "Alpha the one-space entry, built from:\n",
            "cannot read an entry from line 3",
        ),
        (
            SECTION + "Alpha the one-space entry, built from:\n\n",
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
