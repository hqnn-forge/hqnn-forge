"""Fail unless every rendered module declares and documents its public exports.

Run by the docs CI job after ``mkdocs build --strict`` (#322).  For each page
under ``docs/api/`` that renders a package or module (``::: hqnn_forge.x``),
the module must declare ``__all__``, every public name the module defines
itself (not imports or submodules) must be in it, and every name in it must
appear in the built site's ``objects.inv``: a name exported but missing from
the reference is a hole in the documentation that the strict build does not
see.  ``__all__`` is read statically with griffe, so the check needs neither
torch nor PennyLane.
Root-package exports are also checked, even without a root page; only
``hqnn_forge.__version__`` is explicitly exempt from an API entry.
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
    modules = [
        (f"{page.name}: ", path)
        for page in sorted((docs / "api").glob("*.md"))
        for path in DIRECTIVE.findall(page.read_text())
    ]
    rendered = {path for _, path in modules}
    # A public module of the package with no page at all is the larger hole:
    # none of its names are checked below.
    for name, member in sorted(package.members.items()):
        if member.is_alias or not member.is_module or name.startswith("_"):
            continue
        if f"hqnn_forge.{name}" not in rendered:
            problems.append(f"hqnn_forge.{name} is a public module with no page under docs/api/")
    if "hqnn_forge" not in rendered:
        modules.insert(0, ("", "hqnn_forge"))
    for prefix, path in modules:
        obj = package if path == "hqnn_forge" else package[path[len("hqnn_forge.") :]]
        if not obj.is_module:
            continue
        if obj.exports is None:
            problems.append(f"{prefix}{path} has no __all__")
            continue
        for name in sorted(obj.exports):
            if path == "hqnn_forge" and name == "__version__":
                continue
            if not {f"{path}.{name}", *_shadowed(obj, name)} & documented:
                problems.append(f"{prefix}{path}.{name} is exported but not documented")
        # A public name the module defines itself (not an import or a
        # submodule) but leaves out of ``__all__`` would otherwise go unchecked.
        for name, member in sorted(obj.members.items()):
            if name.startswith("_") or member.is_alias or member.is_module:
                continue
            if name not in obj.exports:
                problems.append(f"{prefix}{path}.{name} is defined but not in __all__")
    return problems


def main() -> int:
    problems = missing(Path("docs"), Path("site"))
    for problem in problems:
        print(f"::error::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
