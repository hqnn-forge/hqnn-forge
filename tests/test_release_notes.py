"""
tests/test_release_notes.py
===========================
The release workflow's gate, .github/scripts/release_notes.py (#321), and the
repository's own CHANGELOG.md staying in the shape it parses.
"""

from __future__ import annotations

import importlib.util
import re
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "release_notes", ROOT / ".github" / "scripts" / "release_notes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_notes = _load()

PR_REFERENCE = re.compile(r"\(#\d+(?:, #\d+)*\)$")

CHANGELOG = textwrap.dedent(
    """\
    # Changelog

    ## [Unreleased]

    ## [0.2.0] - 2026-10-01

    ### Added
    - Second thing (#20)

    ## [0.1.0] - 2026-09-30

    ### Added
    - First thing (#10)

    [0.2.0]: https://github.com/g8rdier/hqnn-forge/compare/v0.1.0...v0.2.0
    [0.1.0]: https://github.com/g8rdier/hqnn-forge/releases/tag/v0.1.0
    """
)


def _repo(tmp_path: Path, version: str = "0.2.0", changelog: str = CHANGELOG) -> Path:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(changelog)
    return tmp_path


class TestChangelogSection:
    def test_a_middle_section_stops_at_the_next_heading(self) -> None:
        assert (
            release_notes.changelog_section(CHANGELOG, "0.2.0")
            == "### Added\n- Second thing (#20)"
        )

    def test_the_last_section_leaves_out_the_link_references(self) -> None:
        assert (
            release_notes.changelog_section(CHANGELOG, "0.1.0") == "### Added\n- First thing (#10)"
        )

    def test_an_empty_and_an_absent_section(self) -> None:
        assert release_notes.changelog_section(CHANGELOG, "Unreleased") == ""
        assert release_notes.changelog_section(CHANGELOG, "0.3.0") is None

    def test_a_version_is_not_matched_as_a_prefix(self) -> None:
        assert release_notes.changelog_section(CHANGELOG, "0.2") is None


class TestCheck:
    def test_a_matching_tag_passes(self, tmp_path: Path) -> None:
        assert release_notes.check("v0.2.0", _repo(tmp_path)) == []

    @pytest.mark.parametrize("tag", ["0.2.0", "v0.2", "v0.2.0-beta", "release-0.2.0"])
    def test_a_malformed_tag(self, tag: str, tmp_path: Path) -> None:
        (problem,) = release_notes.check(tag, _repo(tmp_path))
        assert "is not v<MAJOR>.<MINOR>.<PATCH>" in problem

    def test_a_pre_release_tag_is_well_formed(self, tmp_path: Path) -> None:
        changelog = CHANGELOG.replace("[0.2.0] - 2026", "[0.2.0rc1] - 2026")
        assert release_notes.check("v0.2.0rc1", _repo(tmp_path, "0.2.0rc1", changelog)) == []

    def test_a_tag_that_is_not_project_version(self, tmp_path: Path) -> None:
        problems = release_notes.check("v0.2.0", _repo(tmp_path, version="0.1.0"))
        assert problems == ["tag v0.2.0 does not match project.version '0.1.0' in pyproject.toml."]

    def test_a_version_without_a_changelog_section(self, tmp_path: Path) -> None:
        (problem,) = release_notes.check("v0.3.0", _repo(tmp_path, version="0.3.0"))
        assert "no '## [0.3.0]' section" in problem

    def test_an_empty_section(self, tmp_path: Path) -> None:
        changelog = CHANGELOG.replace("## [Unreleased]", "## [0.3.0] - 2026-10-02")
        (problem,) = release_notes.check("v0.3.0", _repo(tmp_path, "0.3.0", changelog))
        assert "section is empty" in problem


class TestCommandLine:
    def test_notes_prints_the_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(_repo(tmp_path))
        assert release_notes.main(["release_notes.py", "notes", "v0.2.0"]) == 0
        assert capsys.readouterr().out.strip() == "### Added\n- Second thing (#20)"

    def test_check_fails_with_an_annotated_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(_repo(tmp_path, version="0.1.0"))
        assert release_notes.main(["release_notes.py", "check", "v0.2.0"]) == 1
        assert capsys.readouterr().err.startswith("::error::tag v0.2.0 does not match")

    def test_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert release_notes.main(["release_notes.py", "publish", "v0.2.0"]) == 2


class TestThisRepository:
    def test_the_changelog_has_an_unreleased_section_the_script_can_read(self) -> None:
        text = (ROOT / "CHANGELOG.md").read_text()
        section = release_notes.changelog_section(text, "Unreleased")
        assert section is not None and "### Added" in section

    def test_every_changelog_entry_names_its_pull_request(self) -> None:
        # The convention CONTRIBUTING asks for: an entry ends with "(#<PR>)", or
        # "(#<PR>, #<PR>)" for an entry several PRs built, so a release's notes
        # link back. A "#" elsewhere in the entry does not count.
        text = (ROOT / "CHANGELOG.md").read_text()
        entries = []
        for line in text.splitlines():
            if line.startswith("- "):
                entries.append(line)
            elif line.startswith("  ") and entries:
                entries[-1] += " " + line.strip()
        assert entries
        unreferenced = [e for e in entries if not PR_REFERENCE.search(e.rstrip())]
        assert not unreferenced, unreferenced

    def test_the_readme_links_work_on_pypi(self) -> None:
        # PyPI renders README.md as the project description without the
        # repository beside it, so a relative link or image there is broken,
        # and so is a "#heading" anchor: PyPI gives headings no ids.
        readme = (ROOT / "README.md").read_text()
        text = re.sub(r"^```.*?^```", "", readme, flags=re.DOTALL | re.MULTILINE)
        targets = re.findall(r"\]\(\s*<?([^)\s>]+)", text)
        targets += re.findall(r"^ {0,3}\[[^\]]+\]:\s*<?([^\s>]+)", text, flags=re.MULTILINE)
        targets += re.findall(r"\b(?:src|href)\s*=\s*[\"']?([^\"'\s>]+)", text)
        assert targets
        relative = [t for t in targets if not re.match(r"https?://|mailto:", t)]
        assert not relative, relative
