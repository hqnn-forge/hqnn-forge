"""Generate README.pypi.md from README.md with Mermaid diagrams replaced by links.

PyPI does not render Mermaid fenced code blocks and displays them as raw source.
This script generates a PyPI-compatible README that replaces each Mermaid diagram
block with an absolute Markdown link to the corresponding heading in the GitHub
repository README.

Usage:
    python .github/scripts/pypi_readme.py
    python .github/scripts/pypi_readme.py --check
    python .github/scripts/pypi_readme.py --in-place
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/hqnn-forge/hqnn-forge"


def heading_to_slug(heading: str) -> str:
    """Convert a markdown heading into a GitHub anchor slug."""
    cleaned = re.sub(r"[^\w\s-]", "", heading.lower())
    return re.sub(r"[\s]+", "-", cleaned.strip())


def repository_url(root: Path) -> str:
    """Read the repository URL from pyproject.toml, falling back to the default."""
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.exists():
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            url = data.get("project", {}).get("urls", {}).get("Repository")
            if isinstance(url, str):
                return url
        except (tomllib.TOMLDecodeError, OSError):
            return DEFAULT_REPO_URL
    return DEFAULT_REPO_URL


def generate(readme_text: str, repo_url: str = DEFAULT_REPO_URL) -> str:
    """Generate a PyPI-compatible README by replacing Mermaid blocks with links."""
    lines = readme_text.splitlines(keepends=True)
    out: list[str] = []
    current_slug: str | None = None
    in_mermaid = False
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        if in_mermaid:
            if stripped.startswith(("```", "~~~")):
                in_mermaid = False
            continue

        if in_code_block:
            if stripped.startswith(("```", "~~~")):
                in_code_block = False
            out.append(line)
            continue

        if stripped.startswith("```mermaid"):
            in_mermaid = True
            target_url = f"{repo_url}#{current_slug}" if current_slug else repo_url
            nl = "\r\n" if line.endswith("\r\n") else "\n"
            out.append(f"*[View diagram on GitHub]({target_url})*{nl}")
            continue

        if stripped.startswith(("```", "~~~")):
            in_code_block = True
            out.append(line)
            continue

        heading_match = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if heading_match:
            current_slug = heading_to_slug(heading_match.group(1))

        out.append(line)

    return "".join(out)


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    """Generate PyPI-compatible README, check it, or rewrite README.md in place."""
    parser = argparse.ArgumentParser(
        description="Generate PyPI-compatible README by replacing Mermaid diagrams with links."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Check that README.pypi.md exists and is up to date with README.md.",
    )
    group.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite README.md in place in the repository root.",
    )

    if argv is None:
        args_to_parse = sys.argv[1:]
    elif argv and not argv[0].startswith("-"):
        args_to_parse = argv[1:]
    else:
        args_to_parse = argv

    try:
        args = parser.parse_args(args_to_parse)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    if root is None:
        root = Path(__file__).resolve().parents[2]
    readme_path = root / "README.md"
    pypi_readme_path = root / "README.pypi.md"

    if not readme_path.exists():
        print(f"Error: {readme_path} not found", file=sys.stderr)
        return 1

    repo_url = repository_url(root)
    original_text = readme_path.read_text(encoding="utf-8")
    generated_text = generate(original_text, repo_url=repo_url)

    if args.check:
        if not pypi_readme_path.exists():
            print(
                f"{pypi_readme_path.name} does not exist. "
                "Regenerate it with: python .github/scripts/pypi_readme.py",
                file=sys.stderr,
            )
            return 1
        committed_text = pypi_readme_path.read_text(encoding="utf-8")
        if committed_text != generated_text:
            print(
                f"{pypi_readme_path.name} is out of date with {readme_path.name}. "
                "Regenerate it with: python .github/scripts/pypi_readme.py",
                file=sys.stderr,
            )
            return 1
        return 0

    if args.in_place:
        readme_path.write_text(generated_text, encoding="utf-8", newline="\n")
        return 0

    pypi_readme_path.write_text(generated_text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
