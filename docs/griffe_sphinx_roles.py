"""
A griffe extension: Sphinx cross-reference roles to mkdocs autorefs (#322).

The docstrings use Sphinx roles -- ``:class:`~hqnn_forge.models.HybridBinaryClassifier```,
``:func:`gradient_variance``` -- which mkdocstrings renders as literal text.  This
extension rewrites them, once the whole package is loaded, into autorefs links
(``[`HybridBinaryClassifier`][hqnn_forge.models.HybridBinaryClassifier]``), which
``mkdocs build --strict`` then checks: a reference to an object that does not
exist, or is not rendered, fails the build.

Resolution, for a role in the docstring of object ``obj``:

* ``hqnn_forge.…`` -- qualified.
* anything else -- tried against ``obj``'s members, then each enclosing scope
  up to the package root; the first existing object wins.  If none exists,
  the reference is still emitted, at ``obj``'s module, so the build reports it.
* A re-export resolves to the object's canonical path, where it is rendered:
  ``hqnn_forge.models.HybridBinaryClassifier`` and the class in
  ``hqnn_forge.models.hybrid_classifier`` are one object.
* a target naming a private object (a component starting with ``_``), or a
  qualified name outside the package (``torch.nn.Module``), becomes plain code:
  private objects have no page, and other packages' docs are not linked.

``~`` shows only the last component, as in Sphinx.

The extension also reads Sphinx's ``#:`` attribute comments -- the consecutive
``#:`` lines right above an assignment -- as that attribute's docstring when it
has none, since the package documents its constants that way and griffe does
not.  Without this, a documented constant such as ``METRICS`` would not be
rendered at all.
"""

from __future__ import annotations

import re
from typing import Any

import griffe

ROLE = re.compile(r":(?:class|func|meth|mod|data|attr|exc|obj):`(~?)([^`<>]+)`")
PACKAGE = "hqnn_forge"
# As ``docstring_style`` in mkdocs.yml.
DOCSTRING_STYLE = "numpy"


def _canonical(package: griffe.Module, path: str) -> str | None:
    """The canonical path of the object at ``path`` (an import resolved to its
    definition, where the page renders it), or None if there is none."""
    if path == PACKAGE:
        return PACKAGE
    if not path.startswith(PACKAGE + "."):
        return None
    try:
        return str(package[path[len(PACKAGE) + 1 :]].canonical_path)
    except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError):
        return None


def _scopes(obj: griffe.Object) -> list[str]:
    """``obj`` itself, then every enclosing object's path, innermost first."""
    scopes, current = [], obj
    while current is not None:
        scopes.append(current.path)
        current = current.parent  # type: ignore[assignment]
    return scopes


def resolve(target: str, obj: griffe.Object, package: griffe.Module) -> str | None:
    """The object path ``target`` refers to from ``obj``, or None for plain code."""
    target = target.strip().rstrip("()")
    parts = target.split(".")
    if any(p.startswith("_") and not (p.startswith("__") and p.endswith("__")) for p in parts):
        return None
    if parts[0] == PACKAGE:
        return _canonical(package, target) or target
    for scope in _scopes(obj):
        found = _canonical(package, f"{scope}.{target}")
        if found is not None:
            return found
    if "." in target:
        # Qualified in another package (torch.nn.Module): not linked.
        return None
    # Unresolvable: emit it anyway, so the strict build reports it.
    return f"{obj.module.path}.{target}"


def convert(text: str, obj: griffe.Object, package: griffe.Module) -> str:
    def replace(match: re.Match[str]) -> str:
        short, target = match.group(1), match.group(2)
        path = resolve(target, obj, package)
        shown = target.strip().split(".")[-1] if short else target.strip()
        if path is None:
            return f"`{shown}`"
        return f"[`{shown}`][{path}]"

    return ROLE.sub(replace, text)


def comment_docstring(lines: list[str], lineno: int) -> str | None:
    """The ``#:`` comment block directly above 1-based line ``lineno``, if any."""
    block: list[str] = []
    i = lineno - 2
    while i >= 0 and lines[i].lstrip().startswith("#:"):
        block.insert(0, lines[i].lstrip()[2:].removeprefix(" "))
        i -= 1
    return "\n".join(block) if block else None


class SphinxRoles(griffe.Extension):
    """Rewrite every docstring of the package once it is fully loaded."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        if pkg.path != PACKAGE:
            return
        stack: list[griffe.Object] = [pkg]
        while stack:
            obj = stack.pop()
            if obj.is_attribute and obj.docstring is None and obj.lineno is not None:
                text = comment_docstring(obj.lines_collection[obj.filepath], obj.lineno)  # type: ignore[index]
                if text is not None:
                    # mkdocstrings sets the parser only on the object a ``:::``
                    # directive names, not on its members, so a constant rendered
                    # on its module's page would otherwise not be parsed at all.
                    obj.docstring = griffe.Docstring(
                        text, lineno=obj.lineno, parent=obj, parser=DOCSTRING_STYLE
                    )
            if obj.docstring is not None:
                obj.docstring.value = convert(obj.docstring.value, obj, pkg)
            stack.extend(m for m in obj.members.values() if not m.is_alias)
