"""Fail unless every hand-written copy of the Python floor agrees with requires-python.

Run by the lint job. The floor (3.11 today) is repeated in pyproject.toml and in
.github/workflows/tests.yml, and each tool only ever sees its own copy, so raising
requires-python without the others fails nothing: vermin would keep rejecting APIs the
new floor allows, and CI would keep testing an interpreter the package no longer
supports. This checks, against the floor in requires-python:

- ruff's target-version in pyproject.toml
- the lowest "Programming Language :: Python :: 3.x" classifier
- vermin's -t=X.Y- target in tests.yml, and in the local copies of that command in
  README.md and CONTRIBUTING.md
- the lowest literal python-version of every job in tests.yml, except the jobs in
  EXEMPT_JOBS, which run above the floor on purpose

[tool.mypy] python_version is not checked: it is 3.12 on purpose, because the numpy
stubs use PEP 695 syntax (see the comment in pyproject.toml).
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Documents that repeat the lint job's vermin command for running it locally.
DOCS = ("README.md", "CONTRIBUTING.md")

#: Jobs whose python-version is deliberately not the floor, with the reason.
EXEMPT_JOBS = {
    "lint": "runs mypy, which must target 3.12 for the numpy stubs' PEP 695 syntax",
}


def _version(text: str) -> tuple[int, int]:
    major, minor = text.split(".")[:2]
    return int(major), int(minor)


def _show(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def job_python_versions(workflow: str) -> dict[str, list[tuple[int, int]]]:
    """Literal python-version values per job; ``${{ ... }}`` references are skipped."""
    jobs: dict[str, list[tuple[int, int]]] = {}
    in_jobs = False
    current: str | None = None
    for line in workflow.splitlines():
        if re.match(r"^jobs:\s*$", line):
            in_jobs = True
            continue
        if in_jobs and re.match(r"^\S", line):
            in_jobs = False
        if not in_jobs:
            continue
        header = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if header:
            current = header.group(1)
            jobs[current] = []
            continue
        match = re.match(r"^\s+python-version:\s*(.+?)\s*$", line)
        if current is None or not match or "${{" in match.group(1):
            continue
        jobs[current] += [_version(v) for v in re.findall(r"\d+\.\d+", match.group(1))]
    return jobs


def check(
    pyproject_text: str, workflow_text: str, docs: dict[str, str] | None = None
) -> list[str]:
    """Every disagreement with the floor in ``requires-python``, as messages.

    ``docs`` maps a file name to its text; each vermin ``-t=X.Y-`` in it is checked too.
    """
    project = tomllib.loads(pyproject_text)
    requires = project["project"]["requires-python"]
    match = re.search(r">=\s*(\d+\.\d+)", requires)
    if not match:
        return [f"requires-python {requires!r} has no '>=X.Y' floor to check against"]
    floor = _version(match.group(1))
    errors: list[str] = []

    target = project.get("tool", {}).get("ruff", {}).get("target-version")
    if target != f"py{floor[0]}{floor[1]}":
        errors.append(
            f"[tool.ruff] target-version is {target!r}, expected 'py{floor[0]}{floor[1]}'"
        )

    classifiers = [
        _version(m.group(1))
        for c in project["project"].get("classifiers", [])
        if (m := re.fullmatch(r"Programming Language :: Python :: (\d+\.\d+)", c))
    ]
    if not classifiers or min(classifiers) != floor:
        lowest = _show(min(classifiers)) if classifiers else "none"
        errors.append(f"the lowest Python classifier is {lowest}, expected {_show(floor)}")

    vermin = re.findall(r"vermin\b[^\n]*?-t=(\d+\.\d+)-", workflow_text)
    if not vermin:
        errors.append("tests.yml: no vermin -t=X.Y- target found")
    for v in vermin:
        if _version(v) != floor:
            errors.append(f"tests.yml: vermin targets -t={v}-, expected -t={_show(floor)}-")
    for name, text in (docs or {}).items():
        for v in re.findall(r"vermin\b[^\n]*?-t=(\d+\.\d+)-", text):
            if _version(v) != floor:
                errors.append(f"{name}: vermin targets -t={v}-, expected -t={_show(floor)}-")

    for job, versions in job_python_versions(workflow_text).items():
        if not versions:
            continue
        below = [v for v in versions if v < floor]
        if below:
            errors.append(
                f"tests.yml job {job!r} runs {', '.join(map(_show, below))}, "
                f"below the floor {_show(floor)}"
            )
        elif job not in EXEMPT_JOBS and min(versions) != floor:
            errors.append(
                f"tests.yml job {job!r}: lowest python-version is {_show(min(versions))}, "
                f"expected the floor {_show(floor)}"
            )
    return errors


def main() -> int:
    errors = check(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"),
        {name: (ROOT / name).read_text(encoding="utf-8") for name in DOCS},
    )
    for error in errors:
        print(f"::error::{error}")
    if not errors:
        print("every copy of the Python floor agrees with requires-python")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
