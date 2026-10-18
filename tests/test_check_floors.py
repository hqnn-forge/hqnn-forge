"""
tests/test_check_floors.py
==========================
``.github/scripts/check_floors.py`` (#564): dev dependency floor extraction
and single-source resolution for test-lowest CI job.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script() -> ModuleType:
    path = ROOT / ".github" / "scripts" / "check_floors.py"
    spec = importlib.util.spec_from_file_location("check_floors", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _script()


def test_qutip_casing_matches() -> None:
    project = {
        "optional-dependencies": {
            "dev": ["QuTiP>=5.1.1"],
        }
    }
    assert checker.get_checked_dev_specs(project) == ["QuTiP>=5.1.1"]


def test_marker_passed_through_verbatim() -> None:
    spec = "qutip>=5.2.2; python_version >= '3.14'"
    project = {
        "optional-dependencies": {
            "dev": [spec],
        }
    }
    assert checker.get_checked_dev_specs(project) == [spec]


def test_checked_dev_missing_raises_error() -> None:
    project = {
        "optional-dependencies": {
            "dev": ["pytest>=8.0.0"],
        }
    }
    with pytest.raises(
        ValueError, match=r"No dev dependencies found in pyproject\.toml matching CHECKED_DEV"
    ):
        checker.get_checked_dev_specs(project)


def test_missing_dev_extra_raises_key_error() -> None:
    project: dict[str, object] = {"optional-dependencies": {}}
    with pytest.raises(KeyError):
        checker.get_checked_dev_specs(project)


def test_unknown_argument_returns_exit_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert checker.main(["check_floors.py", "--unknown"]) == 2
    assert checker.main(["--unknown"]) == 2


def test_print_dev_floors_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert checker.main(["--print-dev-floors"]) == 0
    captured = capsys.readouterr().out.splitlines()
    assert any("qutip>=5.1.1" in line for line in captured)
