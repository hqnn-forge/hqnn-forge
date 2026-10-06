"""Check a release tag against pyproject.toml and CHANGELOG.md, and print its notes.

Used by the release workflow before anything is built or published:

    python .github/scripts/release_notes.py check v0.1.0
    python .github/scripts/release_notes.py notes v0.1.0 > notes.md

``check`` fails unless the tag is ``v`` + ``project.version`` and CHANGELOG.md has
a non-empty ``## [<version>]`` section.  So a release needs a release PR first: it
sets the version and renames ``## [Unreleased]`` to ``## [<version>] - <date>``,
above a fresh empty ``[Unreleased]``.  ``notes`` prints that section's body, which
becomes the GitHub release's description.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\](?P<rest>.*)$", re.MULTILINE)


def changelog_section(text: str, version: str) -> str | None:
    """Body of ``## [version]`` (up to the next ``## [`` heading), stripped; None if absent."""
    headings = list(HEADING.finditer(text))
    for i, match in enumerate(headings):
        if match.group("version") == version:
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            # Link references at the bottom ("[0.1.0]: https://...") are not notes.
            body = re.split(r"^\[[^\]]+\]: ", text[match.end() : end], flags=re.MULTILINE)[0]
            return body.strip()
    return None


def check(tag: str, root: Path) -> list[str]:
    """Problems that must stop a release of ``tag``; empty when it may go ahead."""
    problems = []
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", tag):
        return [f"tag {tag!r} is not v<MAJOR>.<MINOR>.<PATCH> (optionally a/b/rc<N>)."]
    version = tag[1:]
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if declared != version:
        problems.append(
            f"tag {tag} does not match project.version {declared!r} in pyproject.toml."
        )
    section = changelog_section((root / "CHANGELOG.md").read_text(), version)
    if section is None:
        problems.append(
            f"CHANGELOG.md has no '## [{version}]' section; rename '## [Unreleased]' in the "
            f"release PR."
        )
    elif not section:
        problems.append(f"CHANGELOG.md's '## [{version}]' section is empty.")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("check", "notes"):
        print(__doc__, file=sys.stderr)
        return 2
    command, tag, root = argv[1], argv[2], Path.cwd()
    problems = check(tag, root)
    if problems:
        for problem in problems:
            print(f"::error::{problem}", file=sys.stderr)
        return 1
    if command == "notes":
        print(changelog_section((root / "CHANGELOG.md").read_text(), tag[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
