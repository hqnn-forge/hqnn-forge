"""Generate a PyPI-compatible README by replacing Mermaid diagrams with links.

PyPI does not render Mermaid fenced code blocks and displays them as raw source.
This script generates a PyPI-compatible README that replaces each Mermaid diagram
block with an absolute Markdown link to the corresponding heading in the GitHub
repository README at the release tag read from pyproject.toml.

Usage:
    python .github/scripts/pypi_readme.py
    python .github/scripts/pypi_readme.py --in-place
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/hqnn-forge/hqnn-forge"


def heading_to_slug(heading: str, *, used_slugs: set[str] | None = None) -> str:
    """Convert a Markdown heading into a GitHub anchor slug.

    A closing ``#`` sequence is dropped, as GitHub does (``Foo ##`` becomes
    ``foo``). Inline links contribute their visible text. Lowercase that text, remove
    punctuation, then replace each whitespace character with a hyphen without
    collapsing runs (``What & why`` becomes ``what--why``). When ``used_slugs``
    is supplied, record anchors in document order and append the first unused
    ``-1``, ``-2``, ... suffix on collisions. Include all headings, even those
    without a diagram; headings inside code fences are excluded by ``generate``.
    """
    text = re.sub(r"(^|\s+)#+\s*$", "", heading.strip())
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    cleaned = re.sub(r"[^\w\s-]", "", text.strip().lower())
    slug = re.sub(r"\s", "-", cleaned)
    if used_slugs is not None:
        base_slug = slug
        suffix = 0
        while slug in used_slugs:
            suffix += 1
            slug = f"{base_slug}-{suffix}"
        used_slugs.add(slug)
    return slug


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


def project_version(root: Path) -> str:
    """Read the release version from pyproject.toml without a fallback tag."""
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def generate(readme_text: str, repo_url: str = DEFAULT_REPO_URL, *, version: str) -> str:
    """Replace Mermaid blocks with links to the README at ``v<version>``.

    The info string may follow the fence after whitespace (``~~~ mermaid``).
    Both backtick and tilde fences close only on the same character with at
    least the opening fence's length. Other code blocks remain unchanged.
    """
    readme_url = f"{repo_url.rstrip('/')}/blob/v{version}/README.md"
    lines = readme_text.splitlines(keepends=True)
    out: list[str] = []
    used_slugs: set[str] = set()
    current_slug: str | None = None
    # The open fence (character and length) and whether it is a Mermaid block,
    # whose lines are dropped instead of kept.
    fence_char: str | None = None
    fence_len = 0
    is_mermaid = False

    for line in lines:
        stripped = line.strip()

        if fence_char is not None:
            if re.match(rf"^{re.escape(fence_char)}{{{fence_len},}}\s*$", stripped):
                fence_char = None
                if is_mermaid:
                    is_mermaid = False
                    continue
            if not is_mermaid:
                out.append(line)
            continue

        fence_match = re.match(r"^(`{3,}|~{3,})(\s*mermaid\b)?", stripped)
        if fence_match:
            fence_char = fence_match.group(1)[0]
            fence_len = len(fence_match.group(1))
            is_mermaid = fence_match.group(2) is not None
            if is_mermaid:
                target_url = f"{readme_url}#{current_slug}" if current_slug else readme_url
                nl = "\r\n" if line.endswith("\r\n") else "\n"
                out.append(f"*[View diagram on GitHub]({target_url})*{nl}")
            else:
                out.append(line)
            continue

        heading_match = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if heading_match:
            current_slug = heading_to_slug(heading_match.group(1), used_slugs=used_slugs)

        out.append(line)

    return "".join(out)


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    """Generate a PyPI-compatible README or rewrite README.md in place."""
    parser = argparse.ArgumentParser(
        description="Generate a PyPI-compatible README by replacing Mermaid diagrams with links."
    )
    parser.add_argument(
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

    if not readme_path.exists():
        print(f"Error: {readme_path} not found", file=sys.stderr)
        return 1

    try:
        version = project_version(root)
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        print(f"Error: cannot read project.version from pyproject.toml: {exc!r}", file=sys.stderr)
        return 1

    repo_url = repository_url(root)
    original_text = readme_path.read_text(encoding="utf-8")
    generated_text = generate(original_text, repo_url=repo_url, version=version)

    if args.in_place:
        readme_path.write_text(generated_text, encoding="utf-8", newline="\n")
        return 0

    sys.stdout.write(generated_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
