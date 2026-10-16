"""
A griffe extension: Sphinx cross-reference roles to mkdocs autorefs (#322).

The docstrings use Sphinx roles -- ``:class:`~hqnn_forge.models.HybridBinaryClassifier```,
``:func:`gradient_variance``` -- which mkdocstrings renders as literal text.  This
extension rewrites them, once the whole package is loaded, into autorefs links
(``[`HybridBinaryClassifier`][hqnn_forge.models.HybridBinaryClassifier]``), which
``mkdocs build --strict`` then checks: a reference to an object that does not
exist, or is not rendered, fails the build.

Resolution, for a role in the docstring of object ``obj``:

* ``hqnn_forge.…`` -- qualified.  If it does not exist it is emitted as
  written, whatever its owner: the rules below for members griffe cannot see
  apply to unqualified targets only (#578).
* anything else -- tried against ``obj``'s members, then each enclosing scope
  up to the package root; the first existing object wins.  If none exists,
  the reference is still emitted, at ``obj``'s module, so the build reports it.
* A re-export resolves to the object's canonical path, where it is rendered:
  ``hqnn_forge.models.HybridBinaryClassifier`` and the class in
  ``hqnn_forge.models.hybrid_classifier`` are one object.
* a target naming a private object (a component starting with ``_``), or a
  dotted name whose first component is absent from every scope or imported
  from outside the package (``torch.nn.Module``, ``np.ndarray``), becomes plain
  code: private objects have no page, and other packages' docs are not linked.
  A missing member of an in-package module, or of a class with only in-package
  bases, is still emitted under that object's canonical path.  Members of
  anything else -- a class inheriting from ``torch.nn.Module``, an attribute's
  value -- are invisible to griffe, so such a reference is plain code too, as
  is a missing dunder, which every class has implicitly.

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


def _in_package(path: str) -> bool:
    return path == PACKAGE or path.startswith(PACKAGE + ".")


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _lookup(package: griffe.Module, path: str) -> griffe.Object | griffe.Alias:
    """The object at the in-package ``path``; raises KeyError if there is none."""
    return package if path == PACKAGE else package[path[len(PACKAGE) + 1 :]]


def _canonical(package: griffe.Module, path: str) -> str | None:
    """The canonical path of the object at ``path`` (an import resolved to its
    definition, where the page renders it), or None if there is none."""
    if not _in_package(path):
        return None
    try:
        return str(_lookup(package, path).canonical_path)
    except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError):
        return None


def _scopes(obj: griffe.Object) -> list[str]:
    """``obj`` itself, then every enclosing object's path, innermost first."""
    scopes, current = [], obj
    while current is not None:
        scopes.append(current.path)
        current = current.parent  # type: ignore[assignment]
    return scopes


def _members_known(obj: griffe.Object | griffe.Alias) -> bool:
    """Whether griffe sees every member of ``obj``: an in-package module, or an
    in-package class whose bases are all such classes.  griffe does not load
    other packages, so it knows neither the members a class inherits from
    ``torch.nn.Module`` nor those of an attribute's value (a ``list``).  Of
    instance attributes it sees only those assigned in ``__init__`` or
    declared in the class body; one first assigned elsewhere (``self.fitted_``
    in ``fit``) must be declared to be referenced."""
    try:
        obj = obj.final_target if obj.is_alias else obj  # type: ignore[union-attr]
    except (griffe.AliasResolutionError, griffe.CyclicAliasError):
        return False
    if not _in_package(obj.path):
        return False
    if obj.is_module:
        return True
    if not obj.is_class:
        return False
    bases = obj.resolved_bases  # type: ignore[union-attr]
    return len(bases) == len(obj.bases) and all(_members_known(b) for b in bases)  # type: ignore[union-attr]


def resolve(target: str, obj: griffe.Object, package: griffe.Module) -> str | None:
    """The object path ``target`` refers to from ``obj``, or None for plain code.

    Search scopes innermost first.  An unresolved dotted target is checked in
    the innermost scope holding its first component: from there, its longest
    existing prefix must be an object whose members griffe all sees
    (`_members_known`), and the reference is emitted under that object's
    canonical path for autorefs to report.  Otherwise -- no scope holds the
    first component, the whole target is a foreign import, the prefix is
    foreign, reached through a foreign import, inherits from another package,
    or is a function or attribute, or the first missing member is a dunder --
    the target is plain code.  An unresolved undotted target is emitted at
    ``obj``'s module.
    """
    target = target.strip().rstrip("()")
    parts = target.split(".")
    if any(p.startswith("_") and not _is_dunder(p) for p in parts):
        return None
    if parts[0] == PACKAGE:
        return _canonical(package, target) or target
    root_scope = None
    for scope in _scopes(obj):
        found = _canonical(package, f"{scope}.{target}")
        if found is not None:
            return found if _in_package(found) else None
        if root_scope is None and parts[0] in _lookup(package, scope).all_members:
            root_scope = scope
    if "." not in target:
        # Unresolvable: emit it anyway, so the strict build reports it.
        return f"{obj.module.path}.{target}"
    if root_scope is None:
        # No local root: qualified in another package (torch.nn.Module).
        return None
    for i in range(len(parts), 0, -1):
        try:
            prefix = _lookup(package, f"{root_scope}.{'.'.join(parts[:i])}")
        except KeyError:
            continue
        except (griffe.AliasResolutionError, griffe.CyclicAliasError):
            return None  # Reached through an import from another package.
        # The whole target exists but has no canonical path: a foreign import
        # (types.Tensor).  A missing dunder is implicit (object.__repr__, a
        # dataclass's __eq__) and has no page.
        if i == len(parts) or _is_dunder(parts[i]) or not _members_known(prefix):
            return None
        # A missing member of a local object: emit it, so the strict build reports it.
        return ".".join([prefix.canonical_path, *parts[i:]])
    return None


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
