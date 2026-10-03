"""
tests/test_readme_references.py
===============================
Every source cited in a module docstring's ``References`` section is listed
in the README's ``References`` section, with the same year (#168).

Only module-level ``References`` blocks are checked; a work mentioned in
passing in prose elsewhere need not be listed.  Citations are compared by
surnames and year, so ``King, G. & Zeng, L. (2001)`` in a docstring matches
``King & Zeng (2001)`` in the README, and ``and`` matches ``&``.

The scan also fails when a block does not parse completely: a ``References``
heading that is not a numpy-style section, or a bullet that does not read
``Authors (YYYY)`` on its first line.  Otherwise such a citation would be
skipped and never compared against the README.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "hqnn_forge"
README = ROOT / "README.md"

# "* Authors (YYYY)" at the start of a bullet: the authors run up to the year.
_BULLET = re.compile(
    r"^\s*[*-]\s+(?P<authors>[^()\n]+?)\s*\((?P<year>\d{4}[a-z]?)\)", re.MULTILINE
)
# Any bullet line, parsed or not.
_BULLET_LINE = re.compile(r"^\s*[*-]\s", re.MULTILINE)
# A numpy-style section header: a title line underlined with dashes.
_SECTION = re.compile(r"^(?P<title>\S[^\n]*)\n-{3,}[ \t]*\n", re.MULTILINE)
# A line that looks like a References heading, in any form.
_REFERENCES_LINE = re.compile(r"^\s*references\s*:?\s*$", re.MULTILINE | re.IGNORECASE)
# The README's References heading.
_README_HEADING = re.compile(r"^##\s+References\b[^\n]*\n", re.MULTILINE)
# Initials such as "G.", "T.-Y." following a surname.
_INITIALS = re.compile(r",?\s*\b[A-Z]\.(?:-[A-Z]\.)*,?")


def normalise_authors(authors: str) -> str:
    """``"Lin, T.-Y., et al."`` → ``"Lin et al."``; ``"King, G. and Zeng, L."`` → ``"King & Zeng"``."""
    authors = re.sub(r"\s+and\s+", " & ", authors)
    return " ".join(_INITIALS.sub(" ", authors).replace(",", " ").split())


def citations(text: str) -> list[tuple[str, str]]:
    """``(normalised authors, year)`` for each bullet in ``text``."""
    return [(normalise_authors(m["authors"]), m["year"]) for m in _BULLET.finditer(text)]


def references_block(docstring: str) -> str | None:
    """The body of the ``References`` section, up to the next section header."""
    headers = list(_SECTION.finditer(docstring))
    for i, header in enumerate(headers):
        if header["title"].strip() == "References":
            end = headers[i + 1].start() if i + 1 < len(headers) else len(docstring)
            return docstring[header.end() : end]
    return None


def block_problems(docstring: str) -> list[str]:
    """Why the ``References`` section of ``docstring`` does not parse completely, if it doesn't."""
    block = references_block(docstring)
    if block is None:
        if _REFERENCES_LINE.search(docstring):
            return [
                "has a References heading that is not a 'References' line underlined with dashes"
            ]
        return []
    bullets = len(_BULLET_LINE.findall(block))
    parsed = len(citations(block))
    if bullets == 0:
        return ["has a References section with no bullets"]
    if parsed != bullets:
        return [
            (
                f"has {bullets} References bullets but only {parsed} parse as "
                "'Authors (YYYY)' on the bullet's first line"
            )
        ]
    return []


@functools.cache
def scan() -> tuple[dict[tuple[str, str], list[str]], tuple[str, ...]]:
    """Each cited ``(authors, year)`` with the modules citing it, and any unparsed blocks."""
    cited: dict[tuple[str, str], list[str]] = {}
    problems: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        module = str(path.relative_to(ROOT))
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=False)
        problems += [f"{module} {p}" for p in block_problems(docstring or "")]
        for citation in citations(references_block(docstring or "") or ""):
            cited.setdefault(citation, []).append(module)
    return cited, tuple(problems)


def readme_citations() -> set[tuple[str, str]]:
    text = README.read_text(encoding="utf-8")
    heading = _README_HEADING.search(text)
    assert heading, "README.md has no '## References' heading"
    end = text.find("\n## ", heading.end())
    return set(citations(text[heading.end() : end if end != -1 else len(text)]))


class TestParser:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Lin, T.-Y., et al.", "Lin et al."),
            ("King, G. & Zeng, L.", "King & Zeng"),
            ("King, G. and Zeng, L.", "King & Zeng"),
            ("Kingma and Ba", "Kingma & Ba"),
            ("Schuld, Sweke & Meyer", "Schuld Sweke & Meyer"),
            ("Pérez-Salinas et al.", "Pérez-Salinas et al."),
            ("Anderson", "Anderson"),
            ("Wilcoxon", "Wilcoxon"),
        ],
    )
    def test_normalise_authors(self, raw: str, expected: str) -> None:
        assert normalise_authors(raw) == expected

    def test_block_stops_at_the_next_section(self) -> None:
        doc = (
            "Intro\n\nReferences\n----------\n* A et al. (2001) x\n\nNotes\n-----\n* B (2002) y\n"
        )
        assert citations(references_block(doc) or "") == [("A et al.", "2001")]

    def test_readme_bullets_parse(self) -> None:
        text = "- Jones & Gacon (2020) — *Efficient calculation*\n- Wilcoxon (1945) — *x*\n"
        assert citations(text) == [("Jones & Gacon", "2020"), ("Wilcoxon", "1945")]

    def test_year_suffix_parses(self) -> None:
        assert citations("* Kingma & Ba (2015a) x\n") == [("Kingma & Ba", "2015a")]

    def test_trailing_whitespace_after_the_dashes(self) -> None:
        doc = "Intro\n\nReferences\n---------- \n* A (2001) x\n"
        assert citations(references_block(doc) or "") == [("A", "2001")]
        assert block_problems(doc) == []

    @pytest.mark.parametrize(
        "bullet",
        [
            "* Schuld, M., Sinayskiy, I. &\n  Petruccione, F. (2014) x\n",
            "* Bergholm et al. (PennyLane, 2022) x\n",
            "* Wilcoxon 1945 x\n",
        ],
    )
    def test_an_unparsed_bullet_is_a_problem(self, bullet: str) -> None:
        doc = f"Intro\n\nReferences\n----------\n* A (2001) x\n{bullet}"
        assert block_problems(doc) == [
            "has 2 References bullets but only 1 parse as 'Authors (YYYY)' on the bullet's first line"
        ]

    @pytest.mark.parametrize(
        "heading",
        ["References:\n", "references\n----------\n", "References\n~~~~~~~~~~\n", "References\n"],
    )
    def test_a_malformed_heading_is_a_problem(self, heading: str) -> None:
        doc = f"Intro\n\n{heading}* A (2001) x\n"
        assert references_block(doc) is None
        assert len(block_problems(doc)) == 1

    def test_no_references_is_not_a_problem(self) -> None:
        assert block_problems("Intro\n\nNotes\n-----\nSee the references in the paper.\n") == []


class TestReadmeListsDocstringSources:
    def test_every_references_block_parses(self) -> None:
        """Guard against citations the parser skips, which would pass unchecked."""
        cited, problems = scan()
        assert not problems, "\n".join(problems)
        assert ("King & Zeng", "2001") in cited
        assert ("Lin et al.", "2017") in cited

    def test_every_docstring_citation_is_in_the_readme(self) -> None:
        listed = readme_citations()
        years: dict[str, list[str]] = {}
        for authors, year in listed:
            years.setdefault(authors, []).append(year)
        problems = []
        for (authors, year), modules in sorted(scan()[0].items()):
            if (authors, year) in listed:
                continue
            where = ", ".join(modules)
            if authors in years:
                problems.append(
                    f"{authors} ({year}) in {where}: not in README References, which has "
                    f"{authors} ({', '.join(sorted(years[authors]))}); either the year is "
                    "wrong or this is a different work missing from the README"
                )
            else:
                problems.append(f"{authors} ({year}) in {where}: missing from README References")
        assert not problems, "\n".join(problems)
