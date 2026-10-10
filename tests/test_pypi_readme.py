"""
tests/test_pypi_readme.py
=========================
Tests for .github/scripts/pypi_readme.py: generating a PyPI-compatible README
with Mermaid diagrams replaced by links.
"""

from __future__ import annotations

import importlib.util
import re
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_pypi_readme() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pypi_readme", ROOT / ".github" / "scripts" / "pypi_readme.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pypi_readme = _load_pypi_readme()


class TestPyPIReadme:
    def test_heading_to_slug(self) -> None:
        assert pypi_readme.heading_to_slug("How a benchmark works") == "how-a-benchmark-works"
        assert (
            pypi_readme.heading_to_slug("`HybridBinaryClassifier` (serial)")
            == "hybridbinaryclassifier-serial"
        )
        assert (
            pypi_readme.heading_to_slug("`ParallelHybridClassifier` (parallel)")
            == "parallelhybridclassifier-parallel"
        )

    def test_generate_replaces_mermaid_and_preserves_other_blocks(self) -> None:
        sample = textwrap.dedent(
            """\
            # Test Title

            Text before diagram.

            ```mermaid
            flowchart TD
                A --> B
            ```

            Text between diagrams.

            ```python
            print("hello")
            ```

            ## Subheading

            ```mermaid
            graph LR
                C --> D
            ```
            """
        )
        output = pypi_readme.generate(sample, repo_url="https://github.com/org/repo")
        assert "```mermaid" not in output
        assert "*[View diagram on GitHub](https://github.com/org/repo#test-title)*" in output
        assert "*[View diagram on GitHub](https://github.com/org/repo#subheading)*" in output
        assert '```python\nprint("hello")\n```' in output
        assert "Text before diagram." in output
        assert "Text between diagrams." in output

    def test_generate_without_preceding_heading(self) -> None:
        sample = "```mermaid\ngraph TD\n```\n"
        output = pypi_readme.generate(sample, repo_url="https://github.com/org/repo")
        assert "*[View diagram on GitHub](https://github.com/org/repo)*" in output

    def test_heading_inside_fenced_code_block_is_ignored(self) -> None:
        sample = textwrap.dedent(
            """\
            # Title

            ```bash
            # install it
            ```

            ```python
            # python comment
            ```

            ```mermaid
            graph TD; A-->B
            ```
            """
        )
        output = pypi_readme.generate(sample, repo_url="https://github.com/org/repo")
        assert "*[View diagram on GitHub](https://github.com/org/repo#title)*" in output
        assert "install-it" not in output
        assert "python-comment" not in output

    def test_unknown_argument_exits_with_error(self) -> None:
        assert pypi_readme.main(["pypi_readme.py", "--chekc"]) == 2

    def test_mutually_exclusive_check_and_in_place(self) -> None:
        assert pypi_readme.main(["--check", "--in-place"]) == 2

    def test_normal_generation(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title\n\n```mermaid\ngraph TD\n```\n", encoding="utf-8")
        pypi_readme_file = tmp_path / "README.pypi.md"
        assert pypi_readme.main([], root=tmp_path) == 0
        assert pypi_readme_file.exists()
        content = pypi_readme_file.read_text(encoding="utf-8")
        assert "```mermaid" not in content
        assert "*[View diagram on GitHub]" in content

    def test_check_flag_passes_when_up_to_date(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title\n\n```mermaid\ngraph TD\n```\n", encoding="utf-8")
        assert pypi_readme.main([], root=tmp_path) == 0
        assert pypi_readme.main(["--check"], root=tmp_path) == 0

    def test_check_flag_fails_when_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title\n\n```mermaid\ngraph TD\n```\n", encoding="utf-8")
        assert pypi_readme.main(["--check"], root=tmp_path) == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_check_flag_fails_when_stale_without_overwriting(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title\n\n```mermaid\ngraph TD\n```\n", encoding="utf-8")
        pypi_readme_file = tmp_path / "README.pypi.md"
        pypi_readme_file.write_text("old stale content", encoding="utf-8")

        assert pypi_readme.main(["--check"], root=tmp_path) == 1
        # Confirm it did NOT overwrite the file
        assert pypi_readme_file.read_text(encoding="utf-8") == "old stale content"
        captured = capsys.readouterr()
        assert "is out of date" in captured.err

    def test_in_place_flag(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title\n\n```mermaid\ngraph TD\n```\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project.urls]\nRepository = "https://github.com/org/repo"\n',
            encoding="utf-8",
        )
        assert pypi_readme.main(["--in-place"], root=tmp_path) == 0
        content = readme.read_text(encoding="utf-8")
        assert "```mermaid" not in content
        assert "*[View diagram on GitHub](https://github.com/org/repo#title)*" in content

    def test_generate_on_real_readme(self) -> None:
        readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
        output = pypi_readme.generate(readme_text)
        assert "```mermaid" not in output

        anchors = re.findall(r"\[View diagram on GitHub\]\(.*?#([^)]+)\)", output)
        assert anchors

        headings = {
            pypi_readme.heading_to_slug(m.group(1))
            for line in readme_text.splitlines()
            if (m := re.match(r"^#{1,6}\s+(.*)$", line.strip()))
        }
        for anchor in anchors:
            assert anchor in headings
