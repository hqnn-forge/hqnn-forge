"""
tests/test_pypi_readme.py
=========================
Tests for .github/scripts/pypi_readme.py: generating a PyPI-compatible README
with Mermaid diagrams replaced by links.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import textwrap
import zipfile
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
    def test_heading_whitespace_keeps_each_space(self) -> None:
        assert pypi_readme.heading_to_slug("What & why") == "what--why"

    def test_heading_link_uses_visible_text(self) -> None:
        assert pypi_readme.heading_to_slug("[link](http://x.y) title") == "link-title"

    @pytest.mark.parametrize("heading", ["Foo ##", "Foo #", "Foo ##   "])
    def test_heading_closing_hashes_are_dropped(self, heading: str) -> None:
        assert pypi_readme.heading_to_slug(heading) == "foo"

    def test_repeated_headings_use_unique_anchors(self) -> None:
        sample = (
            "# Same\n# Same\n```mermaid\ngraph TD; A-->B\n```\n"
            "# Same\n~~~mermaid\ngraph TD; C-->D\n~~~\n"
        )
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert re.findall(r"README.md#([^)]+)", output) == ["same-1", "same-2"]

    @pytest.mark.parametrize("info", ["mermaid", " mermaid", "\tmermaid"])
    @pytest.mark.parametrize("fence", ["```", "~~~", "````", "~~~~"])
    def test_both_mermaid_fences_are_replaced(self, fence: str, info: str) -> None:
        sample = f"# Diagram\n{fence}{info}\ngraph TD; A-->B\n{fence}\nAfter\n"
        output = pypi_readme.generate(
            sample, repo_url="https://github.com/org/repo", version="0.1.0"
        )
        assert output == (
            "# Diagram\n"
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v0.1.0/README.md#diagram)*\n"
            "After\n"
        )

    @pytest.mark.parametrize("fence", ["```", "~~~", "````", "~~~~"])
    def test_mermaid_closing_fence_matches_opener(self, fence: str) -> None:
        other_fence = "~~~" if fence.startswith("`") else "```"
        # One character short of the opener: a real fence for 4-character
        # openers, which still must not close the block.
        short_fence = fence[0] * (len(fence) - 1)
        sample = (
            f"# Diagram\n{fence}mermaid\n{other_fence}\n{short_fence}\n"
            f"# Not a heading\n{fence}\nAfter\n"
        )
        output = pypi_readme.generate(sample, version="0.1.0")
        assert "Not a heading" not in output
        assert output.endswith("After\n")

    @pytest.mark.parametrize(
        "metadata",
        [
            "```mermaid\nA-->B\n```",
            "~~~mermaid\nA-->B\n~~~",
            "``` mermaid\nA-->B\n```",
            "~~~ mermaid\nA-->B\n~~~",
            "A readable description",
        ],
    )
    def test_release_metadata_guard(self, tmp_path: Path, metadata: str) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        guard = workflow.split("- name: Check the wheel contains no Mermaid fence\n", 1)[1]
        guard = textwrap.dedent(guard.split("      run: |\n", 1)[1].split("    - name:", 1)[0])
        (tmp_path / "dist").mkdir()
        with zipfile.ZipFile(tmp_path / "dist/test.whl", "w") as wheel:
            wheel.writestr("test.dist-info/METADATA", metadata)
        result = subprocess.run(
            ["bash", "-c", guard], cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert result.returncode == (1 if "mermaid" in metadata else 0)

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

    def test_in_place_flag(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title\n\n```mermaid\ngraph TD\n```\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nversion = "2.3.4"\n[project.urls]\nRepository = "https://github.com/org/repo"\n',
            encoding="utf-8",
        )
        assert pypi_readme.main(["--in-place"], root=tmp_path) == 0
        content = readme.read_text(encoding="utf-8")
        assert "```mermaid" not in content
        assert (
            "*[View diagram on GitHub](https://github.com/org/repo/blob/v2.3.4/README.md#title)*"
            in content
        )

    def test_normal_invocation_writes_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Title\n\n```mermaid\ngraph TD\n```\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nversion = "1.2.0rc1"\n', encoding="utf-8"
        )
        assert pypi_readme.main([], root=tmp_path) == 0
        captured = capsys.readouterr()
        assert "*[View diagram on GitHub]" in captured.out
        assert "/blob/v1.2.0rc1/README.md#title" in captured.out
        # Confirm no extra files like README.pypi.md were generated
        assert not (tmp_path / "README.pypi.md").exists()

    @pytest.mark.parametrize(
        "pyproject", [None, '[project]\nname = "x"\n', '[project\nversion = "1.0"\n']
    )
    def test_missing_version_fails_without_modifying_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], pyproject: str | None
    ) -> None:
        readme = tmp_path / "README.md"
        original = "# Title\n\n```mermaid\ngraph TD\n```\n"
        readme.write_text(original, encoding="utf-8")
        if pyproject is not None:
            (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        assert pypi_readme.main(["--in-place"], root=tmp_path) == 1
        assert capsys.readouterr().err.startswith(
            "Error: cannot read project.version from pyproject.toml"
        )
        assert readme.read_text(encoding="utf-8") == original

    def test_check_flag_not_supported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        readme = tmp_path / "README.md"
        original = "# Title\n\n```mermaid\ngraph TD\n```\n"
        readme.write_text(original, encoding="utf-8")
        assert pypi_readme.main(["--check"], root=tmp_path) == 2
        assert readme.read_text(encoding="utf-8") == original
        assert not (tmp_path / "README.pypi.md").exists()

    def test_unknown_argument_fails_without_modifying_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        readme = tmp_path / "README.md"
        original = "# Title\n\n```mermaid\ngraph TD\n```\n"
        readme.write_text(original, encoding="utf-8")
        assert pypi_readme.main(["--chekc"], root=tmp_path) == 2
        assert pypi_readme.main(["--unknown-option"], root=tmp_path) == 2
        assert readme.read_text(encoding="utf-8") == original
        assert not (tmp_path / "README.pypi.md").exists()

    def test_generate_on_real_readme(self) -> None:
        readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
        version = pypi_readme.project_version(ROOT)
        output = pypi_readme.generate(readme_text, version=version)
        assert re.search(r"(`{3,}|~{3,})\s*mermaid", output) is None
        links = re.findall(r"\[View diagram on GitHub\]\(([^)]+)\)", output)
        assert links == [
            f"https://github.com/hqnn-forge/hqnn-forge/blob/v{version}/README.md#how-a-benchmark-works",
            f"https://github.com/hqnn-forge/hqnn-forge/blob/v{version}/README.md#hybridbinaryclassifier-serial",
            f"https://github.com/hqnn-forge/hqnn-forge/blob/v{version}/README.md#parallelhybridclassifier-parallel",
        ]
