"""Tests for .github/scripts/check_dist_contents.py."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _import_script() -> ModuleType:
    path = ROOT / ".github" / "scripts" / "check_dist_contents.py"
    spec = importlib.util.spec_from_file_location("check_dist_contents", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _import_script()


def _make_wheel(path: Path, files: Mapping[str, str | bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            if isinstance(data, str):
                z.writestr(name, data.encode("utf-8"))
            else:
                z.writestr(name, data)


def _make_sdist(path: Path, files: Mapping[str, str | bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            encoded = data.encode("utf-8") if isinstance(data, str) else data
            info = tarfile.TarInfo(name=name)
            info.size = len(encoded)
            t.addfile(info, io.BytesIO(encoded))


def _default_wheel_files() -> dict[str, str | bytes]:
    return {
        "hqnn_forge/__init__.py": "__version__ = '0.1.0'\n",
        "hqnn_forge-0.1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: hqnn-forge\nVersion: 0.1.0\n",
        "hqnn_forge-0.1.0.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: uv (0.13.1)\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }


def _default_sdist_files() -> dict[str, str | bytes]:
    return {
        "hqnn_forge-0.1.0/pyproject.toml": "[project]\nname = 'hqnn-forge'\nversion = '0.1.0'\n",
        "hqnn_forge-0.1.0/PKG-INFO": "Metadata-Version: 2.1\nName: hqnn-forge\nVersion: 0.1.0\n",
    }


def test_identical_distributions_pass(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    errors = checker.compare(dir_a, dir_b)
    assert errors == []


def test_wheel_generator_difference_ignored(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    wheel_files_a = _default_wheel_files()
    wheel_files_b = _default_wheel_files()
    wheel_files_a["hqnn_forge-0.1.0.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: uv 0.13.1\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    wheel_files_b["hqnn_forge-0.1.0.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: uv 0.12.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_a)
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_b)
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    errors = checker.compare(dir_a, dir_b)
    assert errors == []


def test_wheel_purelib_difference_reported(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    wheel_files_a = _default_wheel_files()
    wheel_files_b = _default_wheel_files()
    wheel_files_a["hqnn_forge-0.1.0.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: uv 0.13.1\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    wheel_files_b["hqnn_forge-0.1.0.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: uv 0.12.0\nRoot-Is-Purelib: false\nTag: py3-none-any\n"
    )

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_a)
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_b)
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    errors = checker.compare(dir_a, dir_b)
    assert len(errors) == 1
    assert any("Wheel .dist-info/WHEEL (excluding Generator line) differs" in e for e in errors)
    assert any("-Root-Is-Purelib: true" in e and "+Root-Is-Purelib: false" in e for e in errors)


def test_wheel_tag_difference_reported(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    wheel_files_a = _default_wheel_files()
    wheel_files_b = _default_wheel_files()
    wheel_files_a["hqnn_forge-0.1.0.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: uv 0.13.1\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    wheel_files_b["hqnn_forge-0.1.0.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: uv 0.12.0\nRoot-Is-Purelib: true\nTag: py3-none-manylinux\n"
    )

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_a)
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_b)
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    errors = checker.compare(dir_a, dir_b)
    assert len(errors) == 1
    assert any("Wheel .dist-info/WHEEL (excluding Generator line) differs" in e for e in errors)
    assert any("-Tag: py3-none-any" in e and "+Tag: py3-none-manylinux" in e for e in errors)


def test_wheel_version_difference_reported(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    wheel_files_a = _default_wheel_files()
    wheel_files_b = _default_wheel_files()
    wheel_files_a["hqnn_forge-0.1.0.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: uv 0.13.1\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    wheel_files_b["hqnn_forge-0.1.0.dist-info/WHEEL"] = (
        "Wheel-Version: 2.0\nGenerator: uv 0.12.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_a)
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_b)
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    errors = checker.compare(dir_a, dir_b)
    assert len(errors) == 1
    assert any("Wheel .dist-info/WHEEL (excluding Generator line) differs" in e for e in errors)
    assert any("-Wheel-Version: 1.0" in e and "+Wheel-Version: 2.0" in e for e in errors)


def test_generator_difference_in_metadata_is_not_ignored(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    wheel_files_a = _default_wheel_files()
    wheel_files_b = _default_wheel_files()
    wheel_files_a["hqnn_forge-0.1.0.dist-info/METADATA"] = (
        "Metadata-Version: 2.1\nName: hqnn-forge\nGenerator: uv_build 0.13.1\n"
    )
    wheel_files_b["hqnn_forge-0.1.0.dist-info/METADATA"] = (
        "Metadata-Version: 2.1\nName: hqnn-forge\nGenerator: uv_build 0.12.0\n"
    )

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_a)
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_b)
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    errors = checker.compare(dir_a, dir_b)
    assert len(errors) == 1
    assert any("Wheel .dist-info/METADATA differs" in e for e in errors)
    assert any(
        "-Generator: uv_build 0.13.1" in e and "+Generator: uv_build 0.12.0" in e for e in errors
    )


def test_generator_difference_in_pkg_info_is_not_ignored(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    sdist_files_a = _default_sdist_files()
    sdist_files_b = _default_sdist_files()
    sdist_files_a["hqnn_forge-0.1.0/PKG-INFO"] = (
        "Metadata-Version: 2.1\nName: hqnn-forge\nGenerator: uv_build 0.13.1\n"
    )
    sdist_files_b["hqnn_forge-0.1.0/PKG-INFO"] = (
        "Metadata-Version: 2.1\nName: hqnn-forge\nGenerator: uv_build 0.12.0\n"
    )

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", sdist_files_a)
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", sdist_files_b)

    errors = checker.compare(dir_a, dir_b)
    assert len(errors) == 1
    assert any("Sdist PKG-INFO differs" in e for e in errors)
    assert any(
        "-Generator: uv_build 0.13.1" in e and "+Generator: uv_build 0.12.0" in e for e in errors
    )


def test_metadata_difference_reported(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    wheel_files_a = _default_wheel_files()
    wheel_files_b = _default_wheel_files()
    wheel_files_a["hqnn_forge-0.1.0.dist-info/METADATA"] = (
        "Metadata-Version: 2.1\nName: hqnn-forge\nSummary: Original\n"
    )
    wheel_files_b["hqnn_forge-0.1.0.dist-info/METADATA"] = (
        "Metadata-Version: 2.1\nName: hqnn-forge\nSummary: Modified\n"
    )

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_a)
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_b)
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    errors = checker.compare(dir_a, dir_b)
    assert len(errors) == 1
    assert any("Wheel .dist-info/METADATA differs" in e for e in errors)
    assert any("-Summary: Original" in e and "+Summary: Modified" in e for e in errors)


def test_sdist_metadata_difference_reported(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    sdist_files_a = _default_sdist_files()
    sdist_files_b = _default_sdist_files()
    sdist_files_a["hqnn_forge-0.1.0/PKG-INFO"] = "Metadata-Version: 2.1\nSummary: Alpha\n"
    sdist_files_b["hqnn_forge-0.1.0/PKG-INFO"] = "Metadata-Version: 2.1\nSummary: Beta\n"

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", sdist_files_a)
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", sdist_files_b)

    errors = checker.compare(dir_a, dir_b)
    assert len(errors) == 1
    assert any("Sdist PKG-INFO differs" in e for e in errors)
    assert any("-Summary: Alpha" in e and "+Summary: Beta" in e for e in errors)


def test_wheel_file_manifest_difference(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    wheel_files_a = _default_wheel_files()
    wheel_files_a["hqnn_forge/extra.py"] = ""
    wheel_files_b = _default_wheel_files()

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_a)
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_b)
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    errors = checker.compare(dir_a, dir_b)
    assert len(errors) == 1
    assert any("Wheel file lists differ" in e for e in errors)
    assert any("hqnn_forge/extra.py" in e for e in errors)


def test_sdist_file_manifest_difference(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    sdist_files_a = _default_sdist_files()
    sdist_files_a["hqnn_forge-0.1.0/README.md"] = "readme\n"
    sdist_files_b = _default_sdist_files()

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", sdist_files_a)
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", sdist_files_b)

    errors = checker.compare(dir_a, dir_b)
    assert len(errors) == 1
    assert any("Sdist file lists differ" in e for e in errors)
    assert any("hqnn_forge-0.1.0/README.md" in e for e in errors)


def test_missing_and_ambiguous_archives(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    errors = checker.compare(dir_a, dir_b)
    assert any("Missing wheel archive in" in e for e in errors)
    assert any("Missing sdist archive in" in e for e in errors)

    _make_wheel(dir_a / "one-0.1.0-py3-none-any.whl", {"one/__init__.py": ""})
    _make_wheel(dir_a / "two-0.1.0-py3-none-any.whl", {"two/__init__.py": ""})
    _make_wheel(dir_b / "one-0.1.0-py3-none-any.whl", {"one/__init__.py": ""})

    errors = checker.compare(dir_a, dir_b)
    assert any("Ambiguous: found multiple wheels" in e for e in errors)


def test_missing_metadata_in_archive(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    wheel_files_a = {"hqnn_forge/__init__.py": ""}
    wheel_files_b = {"hqnn_forge/__init__.py": ""}
    sdist_files = {"hqnn_forge-0.1.0/README.md": ""}

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_a)
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_b)
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", sdist_files)
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", sdist_files)

    errors = checker.compare(dir_a, dir_b)
    assert any("Missing .dist-info/METADATA" in e for e in errors)
    assert any("Missing .dist-info/WHEEL" in e for e in errors)
    assert any("Missing PKG-INFO" in e for e in errors)


def test_corrupted_archives(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    (dir_a / "hqnn_forge-0.1.0-py3-none-any.whl").write_bytes(b"not a valid zip")
    (dir_b / "hqnn_forge-0.1.0-py3-none-any.whl").write_bytes(b"also not a zip")
    (dir_a / "hqnn_forge-0.1.0.tar.gz").write_bytes(b"\x1f\x8b\x08\x00truncated")
    (dir_b / "hqnn_forge-0.1.0.tar.gz").write_bytes(b"\x1f\x8b\x08\x00truncated")

    errors = checker.compare(dir_a, dir_b)
    assert any("Corrupted wheel archive" in e for e in errors)
    assert any("Corrupted sdist archive" in e for e in errors)


def test_unicode_decode_error_in_metadata(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    wheel_files_a = _default_wheel_files()
    wheel_files_b = _default_wheel_files()
    wheel_files_a["hqnn_forge-0.1.0.dist-info/METADATA"] = b"\xff\xfe\x00\x00invalid utf8"
    wheel_files_b["hqnn_forge-0.1.0.dist-info/METADATA"] = b"\xff\xfe\x00\x00invalid utf8"

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_a)
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", wheel_files_b)
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    errors = checker.compare(dir_a, dir_b)
    assert any("Failed to decode .dist-info/METADATA as UTF-8" in e for e in errors)


def test_main_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert checker.main([]) == 2

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    _make_wheel(dir_a / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_wheel(dir_b / "hqnn_forge-0.1.0-py3-none-any.whl", _default_wheel_files())
    _make_sdist(dir_a / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())
    _make_sdist(dir_b / "hqnn_forge-0.1.0.tar.gz", _default_sdist_files())

    assert checker.main([str(dir_a), str(dir_b)]) == 0
    captured = capsys.readouterr()
    assert "Package contents check passed" in captured.out


@pytest.mark.parametrize(
    ("spec", "expected_floor"),
    [
        ("uv_build>=0.12,<0.14", "0.12"),
        ("uv_build>=0.12.1", "0.12.1"),
        ("uv_build>=0.12.0", "0.12.0"),
        ("uv-build>=0.12", "0.12"),
        ("UV_Build>=0.12", "0.12"),
        ("uv_build >= 0.12.1rc1, < 0.14", "0.12.1rc1"),
        ("uv_build >= 0.12", "0.12"),
    ],
)
def test_extract_uv_build_floor_valid(tmp_path: Path, spec: str, expected_floor: str) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[build-system]\nrequires = ["{spec}"]\n', encoding="utf-8")
    assert checker.extract_uv_build_floor(pyproject) == expected_floor


@pytest.mark.parametrize(
    ("content", "error_match"),
    [
        ('[project]\nname = "foo"\n', "Missing \\[build-system\\] table"),
        ('[build-system]\nbuild-backend = "foo"\n', "Missing build-system.requires list"),
        ('[build-system]\nrequires = ["uv_build invalid >= 1"]\n', "Invalid requirement"),
        ('[build-system]\nrequires = ["other_pkg>=1.0"]\n', "No 'uv_build' requirement found"),
        (
            '[build-system]\nrequires = ["uv_build>=0.12", "uv-build>=0.12"]\n',
            "Multiple 'uv_build' requirements",
        ),
        ('[build-system]\nrequires = ["uv_build==0.12"]\n', "Expected exactly one '>=' floor"),
        ('[build-system]\nrequires = ["uv_build<0.14"]\n', "Expected exactly one '>=' floor"),
        (
            '[build-system]\nrequires = ["uv_build>=0.12,>=0.13"]\n',
            "Expected exactly one '>=' floor",
        ),
        ('[build-system]\nrequires = ["uv_build>=invalid.version"]\n', "Invalid requirement"),
    ],
)
def test_extract_uv_build_floor_invalid(tmp_path: Path, content: str, error_match: str) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(content, encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=error_match):
        checker.extract_uv_build_floor(pyproject)


def test_extract_uv_build_floor_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        checker.extract_uv_build_floor(tmp_path / "nonexistent.toml")


def test_main_cli_print_constraint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[build-system]\nrequires = ["uv_build>=0.12.2,<0.14"]\n', encoding="utf-8"
    )
    assert checker.main(["--print-uv-build-constraint", str(pyproject)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "uv_build==0.12.2"

    bad_pyproject = tmp_path / "bad.toml"
    bad_pyproject.write_text('[build-system]\nrequires = ["uv_build==0.12"]\n', encoding="utf-8")
    assert checker.main(["--print-uv-build-constraint", str(bad_pyproject)]) == 1
    captured = capsys.readouterr()
    assert "Expected exactly one '>=' floor" in captured.err
