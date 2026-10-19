"""Fail unless every direct runtime dependency is installed at exactly its declared floor.

The same holds for QuTiP, the test reference from the dev extra.

Run by the test-lowest CI job:
- before resolution with ``--print-dev-floors`` to output the dev dependencies to install
  at their declared floor alongside the runtime packages;
- after resolution to check that every direct runtime dependency, checked extra, and
  checked dev dependency is installed at exactly its declared floor.

The resolver does not report a floor it cannot reach: if another dependency requires
a newer version, it installs that one and the suite passes against it, so an unreachable
floor in pyproject.toml would otherwise go unnoticed.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

# The optional extras the test-lowest job resolves lowest alongside the runtime deps
CHECKED_EXTRAS = ["lightning", "sklearn"]
# The packages of the dev extra that job also installs at their floor; the other
# dev tools are installed at their normal resolution or not at all. Editing this list
# changes what CI installs (via --print-dev-floors), not just what it checks.
CHECKED_DEV = ["qutip"]


def get_checked_dev_specs(project: dict) -> list[str]:
    """Return the raw dependency specifications from the dev extra matching CHECKED_DEV."""
    checked = {canonicalize_name(name) for name in CHECKED_DEV}
    dev_deps = project["optional-dependencies"]["dev"]
    matched_names: set[str] = set()
    specs: list[str] = []
    for spec in dev_deps:
        canonical = canonicalize_name(Requirement(spec).name)
        if canonical in checked:
            matched_names.add(canonical)
            specs.append(spec)
    missing = checked - matched_names
    if missing:
        raise ValueError(
            f"No dev dependencies found in pyproject.toml matching CHECKED_DEV: {sorted(missing)}"
        )
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that dependencies are installed at their declared floor, or print dev floors."
    )
    parser.add_argument(
        "--print-dev-floors",
        action="store_true",
        help="Print the raw dependency specifications from the dev extra matching CHECKED_DEV.",
    )

    args = parser.parse_args(argv)

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    if args.print_dev_floors:
        for spec in get_checked_dev_specs(project):
            print(spec)
        return 0

    specs = list(project["dependencies"])
    for extra in CHECKED_EXTRAS:
        specs += project["optional-dependencies"][extra]
    specs += get_checked_dev_specs(project)

    failed = False
    for spec in specs:
        req = Requirement(spec)
        if req.marker is not None and not req.marker.evaluate():
            continue
        floors = [s.version for s in req.specifier if s.operator == ">="]
        if len(floors) != 1:
            print(f"{req.name}: expected exactly one '>=' floor in {spec!r}")
            failed = True
            continue
        floor = Version(floors[0])
        try:
            installed = Version(version(req.name))
        except PackageNotFoundError:
            print(f"{req.name}: not installed (floor {floor})")
            failed = True
            continue
        status = "ok" if installed == floor else "FLOOR NOT INSTALLED"
        print(f"{req.name}: floor {floor}, installed {installed}  {status}")
        failed |= installed != floor
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
