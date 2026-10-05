"""Fail unless every name in a documented module's ``__all__`` has an API entry.

Run by the docs CI job after ``mkdocs build --strict`` (#322).  For each page
under ``docs/api/`` that renders a package or module (``::: hqnn_forge.x``),
every name in that module's ``__all__`` must appear in the built site's
``objects.inv``: a name exported but missing from the reference is a hole in
the documentation that the strict build does not see.  ``__all__`` is read
statically with griffe, so the check needs neither torch nor PennyLane.
"""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import griffe

DIRECTIVE = re.compile(r"^::: (hqnn_forge(?:\.[A-Za-z_]\w*)*)\s*$", re.MULTILINE)


def inventory(site: Path) -> set[str]:
    raw = (site / "objects.inv").read_bytes()
    body = zlib.decompress(raw.split(b"\n", 4)[4]).decode()
    return {line.split(" ", 1)[0] for line in body.splitlines() if line}


def _shadowed(module: griffe.Object | griffe.Alias, name: str) -> set[str]:
    """Where an export griffe sees as a submodule of the same name is rendered.

    ``from hqnn_forge.diagnostics.expressibility import expressibility`` binds
    the function, but griffe keeps the submodule under that name, so the page
    renders the function at its own path instead (``docs/api/diagnostics.md``).
    """
    member = module.members.get(name)
    if member is None or member.is_alias or not member.is_module or name not in member.members:
        return set()
    return {str(member.members[name].path)}


def missing(docs: Path, site: Path) -> list[str]:
    package = griffe.load("hqnn_forge", search_paths=["."])
    documented = inventory(site)
    problems = []
    rendered = {
        path
        for page in (docs / "api").glob("*.md")
        for path in DIRECTIVE.findall(page.read_text())
    }
    # A public module of the package with no page at all is the larger hole:
    # none of its names are checked below.
    for name, member in sorted(package.members.items()):
        if member.is_alias or not member.is_module or name.startswith("_"):
            continue
        if f"hqnn_forge.{name}" not in rendered:
            problems.append(f"hqnn_forge.{name} is a public module with no page under docs/api/")
    for page in sorted((docs / "api").glob("*.md")):
        for path in DIRECTIVE.findall(page.read_text()):
            obj = package if path == "hqnn_forge" else package[path[len("hqnn_forge.") :]]
            if not obj.is_module:
                continue
            for name in sorted(obj.exports or ()):
                if not {f"{path}.{name}", *_shadowed(obj, name)} & documented:
                    problems.append(f"{page.name}: {path}.{name} is exported but not documented")
    return problems


def main() -> int:
    problems = missing(Path("docs"), Path("site"))
    for problem in problems:
        print(f"::error::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
