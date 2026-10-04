"""
tests/test_check_large_files.py
===============================
``.github/scripts/check_large_files.py`` (#220): reads its limit from the
pre-commit hook, applies the hook's size rule, and passes on this repository.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script() -> ModuleType:
    path = ROOT / ".github" / "scripts" / "check_large_files.py"
    spec = importlib.util.spec_from_file_location("check_large_files", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _script()


def test_limit_comes_from_the_pre_commit_hook() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert checker.maxkb_from_config(config) == 1000


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ("repos: []\n", "no check-added-large-files hook"),
        ("hooks:\n  - id: check-added-large-files\n", "has no --maxkb"),
    ],
)
def test_config_without_the_limit(config: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        checker.maxkb_from_config(config)


def test_size_rule_matches_the_hook() -> None:
    # The hook rounds up to whole KiB and fails above the limit.
    sizes = {"at_limit": 10 * 1024, "one_byte_over": 10 * 1024 + 1, "small": 5}
    assert checker.oversized(sizes, 10) == [("one_byte_over", 11)]


def test_largest_first() -> None:
    sizes = {"a": 50 * 1024, "b": 90 * 1024, "c": 1}
    assert checker.oversized(sizes, 20) == [("b", 90), ("a", 50)]


def test_ls_tree_parsing_keeps_blobs_only() -> None:
    listing = (
        "100644 blob 1111111111111111111111111111111111111111    1234\tdata/raw/file.csv\n"
        "160000 commit 2222222222222222222222222222222222222222       -\tvendor/submodule\n"
        "100644 blob 3333333333333333333333333333333333333333      10\tname with spaces.txt\n"
    )
    assert checker.tracked_sizes(listing) == {
        "data/raw/file.csv": 1234,
        "name with spaces.txt": 10,
    }


def test_this_repository_passes() -> None:
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "-l", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    sizes = checker.tracked_sizes(listing)
    assert "uv.lock" in sizes
    assert checker.oversized(sizes, 1000) == []
