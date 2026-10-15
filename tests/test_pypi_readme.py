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
        # Inline Markdown links preserve link text and strip URL
        assert (
            pypi_readme.heading_to_slug("See [our guide](https://example.com/guide) here")
            == "see-our-guide-here"
        )
        assert pypi_readme.heading_to_slug("[Link](url)") == "link"
        # Whitespace runs do not collapse into a single hyphen
        assert pypi_readme.heading_to_slug("What & why") == "what--why"
        assert pypi_readme.heading_to_slug("Multiple   spaces") == "multiple---spaces"
        # Duplicate headings tracking
        used: set[str] = set()
        assert pypi_readme.heading_to_slug("Section", used_slugs=used) == "section"
        assert pypi_readme.heading_to_slug("Section", used_slugs=used) == "section-1"
        assert pypi_readme.heading_to_slug("Section", used_slugs=used) == "section-2"
        pre_populated = {"overview", "overview-1"}
        assert pypi_readme.heading_to_slug("Overview", used_slugs=pre_populated) == "overview-2"

    def test_project_version(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "1.2.3"\n', encoding="utf-8"
        )
        assert pypi_readme.project_version(tmp_path) == "1.2.3"

    def test_project_version_missing_raises_key_error(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
        with pytest.raises(KeyError):
            pypi_readme.project_version(tmp_path)

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
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert "```mermaid" not in output
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#test-title)*"
            in output
        )
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#subheading)*"
            in output
        )
        assert '```python\nprint("hello")\n```' in output
        assert "Text before diagram." in output
        assert "Text between diagrams." in output

    def test_generate_without_preceding_heading(self) -> None:
        sample = "```mermaid\ngraph TD\n```\n"
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md)*"
            in output
        )

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
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#title)*"
            in output
        )
        assert "install-it" not in output
        assert "python-comment" not in output

    def test_mermaid_inside_outer_code_fence_is_not_replaced(self) -> None:
        sample = textwrap.dedent(
            """\
            # Document Title

            ````markdown
            Here is how you write a Mermaid block:
            ```mermaid
            graph TD; A-->B
            ```
            ````

            ```mermaid
            graph LR; X-->Y
            ```
            """
        )
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        # The inner markdown example must retain its mermaid code verbatim
        assert "```mermaid\ngraph TD; A-->B\n```" in output
        # The real mermaid diagram after the block must be replaced with link to Document Title
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#document-title)*"
            in output
        )

    def test_tilde_fence_handles_headings_and_backticks(self) -> None:
        sample = textwrap.dedent(
            """\
            # Tilde Heading

            ~~~python
            # code comment
            echo "```"
            ~~~

            ```mermaid
            graph TD; A-->B
            ```
            """
        )
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#tilde-heading)*"
            in output
        )
        assert "code-comment" not in output

    def test_closing_fence_does_not_cause_incorrect_state(self) -> None:
        sample = textwrap.dedent(
            """\
            # Section One

            ```bash
            run command
            ```

            ```mermaid
            graph TD; A-->B
            ```

            # Section Two

            ```python
            test()
            ```

            ```mermaid
            graph LR; C-->D
            ```
            """
        )
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#section-one)*"
            in output
        )
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#section-two)*"
            in output
        )
        assert "run command" in output
        assert "test()" in output

    def test_generate_tilde_mermaid(self) -> None:
        sample = textwrap.dedent(
            """\
            # Section

            ~~~mermaid
            flowchart TD
                A --> B
            ~~~

            ~~~~mermaid
            graph LR
                C --> D
            ~~~~
            """
        )
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert "~~~mermaid" not in output
        assert "~~~~mermaid" not in output
        assert (
            output.count(
                "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#section)*"
            )
            == 2
        )

    def test_mermaid_fence_must_match_opening_fence_length_and_char(self) -> None:
        sample = textwrap.dedent(
            """\
            # Diagram

            ````mermaid
            flowchart TD
                subgraph "nested"
                ```
                ~~~
                end
            ````
            """
        )
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert "```" not in output
        assert "~~~" not in output
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#diagram)*"
            in output
        )

    def test_all_headings_tracked_for_duplicate_anchors(self) -> None:
        sample = textwrap.dedent(
            """\
            # Overview
            Introductory text without diagram.

            # Overview
            ```mermaid
            flowchart TD
                A --> B
            ```

            # Overview
            ~~~mermaid
            graph LR
                C --> D
            ~~~
            """
        )
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#overview-1)*"
            in output
        )
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#overview-2)*"
            in output
        )

    def test_in_place_flag(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title\n\n```mermaid\ngraph TD\n```\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nversion = "0.2.0"\n\n[project.urls]\nRepository = "https://github.com/org/repo"\n',
            encoding="utf-8",
        )
        assert pypi_readme.main(["--in-place"], root=tmp_path) == 0
        content = readme.read_text(encoding="utf-8")
        assert "```mermaid" not in content
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.2.0/README.md#title)*"
            in content
        )

    def test_normal_invocation_writes_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title\n\n```mermaid\ngraph TD\n```\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nversion = "0.2.0"\n\n[project.urls]\nRepository = "https://github.com/org/repo"\n',
            encoding="utf-8",
        )
        assert pypi_readme.main([], root=tmp_path) == 0
        captured = capsys.readouterr()
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.2.0/README.md#title)*"
            in captured.out
        )
        assert not (tmp_path / "README.pypi.md").exists()

    def test_check_flag_not_supported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        readme = tmp_path / "README.md"
        original = "# Title\n\n```mermaid\ngraph TD\n```\n"
        readme.write_text(original, encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nversion = "0.2.0"\n',
            encoding="utf-8",
        )
        assert pypi_readme.main(["--check"], root=tmp_path) == 2
        assert readme.read_text(encoding="utf-8") == original
        assert not (tmp_path / "README.pypi.md").exists()

    def test_unknown_argument_fails_without_modifying_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        readme = tmp_path / "README.md"
        original = "# Title\n\n```mermaid\ngraph TD\n```\n"
        readme.write_text(original, encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nversion = "0.2.0"\n',
            encoding="utf-8",
        )
        assert pypi_readme.main(["--chekc"], root=tmp_path) == 2
        assert pypi_readme.main(["--unknown-option"], root=tmp_path) == 2
        assert readme.read_text(encoding="utf-8") == original
        assert not (tmp_path / "README.pypi.md").exists()

    def test_generate_on_real_readme(self) -> None:
        readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
        version = pypi_readme.project_version(ROOT)
        output = pypi_readme.generate(readme_text, version=version)
        assert "```mermaid" not in output
        assert "~~~mermaid" not in output

        expected_prefix = f"*[View diagram on GitHub](https://github.com/hqnn-forge/hqnn-forge/blob/v{version}/README.md#"
        assert f"{expected_prefix}how-a-benchmark-works)*" in output
        assert f"{expected_prefix}hybridbinaryclassifier-serial)*" in output
        assert f"{expected_prefix}parallelhybridclassifier-parallel)*" in output

        anchors = re.findall(
            rf"\[View diagram on GitHub\]\(https://github\.com/hqnn-forge/hqnn-forge/blob/v{re.escape(version)}/README\.md#([^)]+)\)",
            output,
        )
        assert anchors == [
            "how-a-benchmark-works",
            "hybridbinaryclassifier-serial",
            "parallelhybridclassifier-parallel",
        ]

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("```mermaid", True),
            ("```mermaid\n", True),
            ("~~~mermaid", True),
            ("````mermaid", True),
            ("~~~~~mermaid", True),
            ("```mermaid flowchart TD", True),
            ("```python", False),
            ("~~~markdown", False),
            ("This mentions mermaid in text", False),
            ("```not_mermaid", False),
        ],
    )
    def test_release_guard_regex(self, line: str, expected: bool) -> None:
        pattern = r"(`{3,}|~{3,})mermaid\b"
        assert bool(re.search(pattern, line)) is expected

    def test_release_workflow_contains_mermaid_guard(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        assert r"grep -Eq '(`{3,}|~{3,})mermaid\b'" in workflow
