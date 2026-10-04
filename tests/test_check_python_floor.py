"""
tests/test_check_python_floor.py
================================
``.github/scripts/check_python_floor.py`` (#211): the repository's own files
agree, and each hand-written copy of the floor that drifts is reported.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script() -> ModuleType:
    path = ROOT / ".github" / "scripts" / "check_python_floor.py"
    spec = importlib.util.spec_from_file_location("check_python_floor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _script()
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
WORKFLOW = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
DOCS = {name: (ROOT / name).read_text(encoding="utf-8") for name in checker.DOCS}


def _replace(text: str, old: str, new: str) -> str:
    assert text.count(old) >= 1, old
    return text.replace(old, new)


def test_the_repository_agrees() -> None:
    assert checker.check(PYPROJECT, WORKFLOW, DOCS) == []
    assert checker.main() == 0


def test_raising_only_requires_python_reports_every_other_copy() -> None:
    errors = checker.check(
        _replace(PYPROJECT, 'requires-python = ">=3.11"', 'requires-python = ">=3.12"'),
        WORKFLOW,
        DOCS,
    )
    text = "\n".join(errors)
    assert "target-version is 'py311', expected 'py312'" in text
    assert "lowest Python classifier is 3.11, expected 3.12" in text
    assert "tests.yml: vermin targets -t=3.11-, expected -t=3.12-" in text
    for name in checker.DOCS:
        assert f"{name}: vermin targets -t=3.11-, expected -t=3.12-" in text
    for job in ("test", "test-locked", "test-lowest"):
        assert f"job {job!r} runs 3.11, below the floor 3.12" in text
    assert "'lint'" not in text  # 3.12 there is the floor now


@pytest.mark.parametrize(
    ("where", "old", "new", "message"),
    [
        ("pyproject", 'target-version = "py311"', 'target-version = "py312"', "target-version"),
        ("workflow", "-t=3.11-", "-t=3.12-", "vermin targets -t=3.12-"),
        (
            "workflow",
            'python-version: ["3.11", "3.13", "3.14"]',
            'python-version: ["3.12", "3.13", "3.14"]',
            "job 'test-lowest': lowest python-version is 3.12",
        ),
        ("README.md", "-t=3.11-", "-t=3.12-", "README.md: vermin targets -t=3.12-"),
        ("CONTRIBUTING.md", "-t=3.11-", "-t=3.12-", "CONTRIBUTING.md: vermin targets -t=3.12-"),
        (
            "pyproject",
            '    "Programming Language :: Python :: 3.11",\n',
            "",
            "lowest Python classifier is 3.12",
        ),
        (
            "workflow",
            'python-version: "3.12"\n        enable-cache: true\n',
            'python-version: "3.10"\n        enable-cache: true\n',
            "job 'lint' runs 3.10, below the floor 3.11",
        ),
    ],
    ids=[
        "ruff",
        "vermin",
        "test-lowest",
        "readme-vermin",
        "contributing-vermin",
        "classifier",
        "exempt-job-below-floor",
    ],
)
def test_one_drifted_copy_is_reported(where: str, old: str, new: str, message: str) -> None:
    pyproject = _replace(PYPROJECT, old, new) if where == "pyproject" else PYPROJECT
    workflow = _replace(WORKFLOW, old, new) if where == "workflow" else WORKFLOW
    docs = {n: _replace(d, old, new) if n == where else d for n, d in DOCS.items()}
    errors = checker.check(pyproject, workflow, docs)
    assert len(errors) == 1 and message in errors[0], errors


def test_matrix_lists_and_expressions() -> None:
    workflow = """
jobs:
  a:
    strategy:
      matrix:
        python-version: ["3.11", "3.13", "3.14"]
    steps:
      - with:
          python-version: ${{ matrix.python-version }}
  b:
    steps:
      - with:
          python-version: "3.13"
"""
    assert checker.job_python_versions(workflow) == {
        "a": [(3, 11), (3, 13), (3, 14)],
        "b": [(3, 13)],
    }


def test_requires_python_without_a_floor() -> None:
    pyproject = _replace(PYPROJECT, 'requires-python = ">=3.11"', 'requires-python = "~=3.11"')
    assert "has no '>=X.Y' floor" in checker.check(pyproject, WORKFLOW, DOCS)[0]
