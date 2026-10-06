"""Fail if a tracked file is larger than the check-added-large-files limit.

Run by the lint job. The pre-commit hook refuses a commit that adds a large file, but
hooks are installed per clone and are optional, so a dataset, a checkpoint or a plot dump
can still reach a PR. This checks every file tracked at HEAD, which in a pull request is
the merge result, against the same limit, read from --maxkb in .pre-commit-config.yaml so
that the two cannot drift apart. The size rule matches the hook's: a file fails when its
size rounded up to whole KiB exceeds the limit.
"""

import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def maxkb_from_config(text: str) -> int:
    """The --maxkb of the check-added-large-files hook in a pre-commit config."""
    hook = text.find("id: check-added-large-files")
    if hook < 0:
        raise ValueError("no check-added-large-files hook in the pre-commit config")
    match = re.search(r"--maxkb=(\d+)", text[hook:])
    if not match:
        raise ValueError("the check-added-large-files hook has no --maxkb argument")
    return int(match.group(1))


def tracked_sizes(ls_tree: str) -> dict[str, int]:
    """``{path: bytes}`` from ``git ls-tree -r -l`` output (blobs only)."""
    sizes: dict[str, int] = {}
    for line in ls_tree.splitlines():
        meta, _, path = line.partition("\t")
        _mode, kind, _sha, size = meta.split()
        if kind == "blob":
            sizes[path] = int(size)
    return sizes


def oversized(sizes: dict[str, int], maxkb: int) -> list[tuple[str, int]]:
    """``(path, KiB)`` for every file over ``maxkb``, largest first."""
    too_big = [(path, math.ceil(size / 1024)) for path, size in sizes.items()]
    return sorted(((p, kb) for p, kb in too_big if kb > maxkb), key=lambda item: -item[1])


def main() -> int:
    maxkb = maxkb_from_config((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "-l", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    failures = oversized(tracked_sizes(listing), maxkb)
    for path, kb in failures:
        print(f"::error file={path}::{path} is {kb} KB, over the {maxkb} KB limit")
    if not failures:
        print(f"no tracked file is over {maxkb} KB")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
