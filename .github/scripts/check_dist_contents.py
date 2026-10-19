"""Compare package distributions built with different backend versions.

Verifies that file manifests and package metadata match between two build outputs.
Compares wheel file lists, METADATA, and WHEEL (ignoring only the Generator line in WHEEL),
and sdist file lists and PKG-INFO.
"""

from __future__ import annotations

import difflib
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Callable
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def _filter_wheel(lines: list[str]) -> list[str]:
    """Filter out lines that are expected to differ in the WHEEL file across backend versions."""
    return [line for line in lines if not line.startswith("Generator:")]


def extract_uv_build_floor(pyproject_path: Path | str = "pyproject.toml") -> str:
    """Extract the minimum declared uv_build version from pyproject.toml."""
    path = Path(pyproject_path)
    if not path.is_file():
        raise FileNotFoundError(f"pyproject.toml not found at {path}")

    try:
        content = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc

    build_system = content.get("build-system")
    if not isinstance(build_system, dict):
        raise TypeError(f"Missing [build-system] table in {path}")

    requires = build_system.get("requires")
    if not isinstance(requires, list):
        raise TypeError(f"Missing build-system.requires list in {path}")

    uv_build_reqs: list[Requirement] = []
    for spec in requires:
        try:
            req = Requirement(spec)
        except InvalidRequirement as exc:
            raise ValueError(
                f"Invalid requirement in build-system.requires: {spec!r}: {exc}"
            ) from exc
        if canonicalize_name(req.name) == "uv-build":
            uv_build_reqs.append(req)

    if not uv_build_reqs:
        raise ValueError(f"No 'uv_build' requirement found in build-system.requires of {path}")
    if len(uv_build_reqs) > 1:
        raise ValueError(
            f"Multiple 'uv_build' requirements found in build-system.requires: "
            f"{[str(r) for r in uv_build_reqs]}"
        )

    req = uv_build_reqs[0]
    floors = [s.version for s in req.specifier if s.operator == ">="]
    if len(floors) != 1:
        raise ValueError(
            f"Expected exactly one '>=' floor for {req.name} in build-system.requires, found: {floors!r}"
        )

    floor = floors[0]
    try:
        Version(floor)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid floor version for {req.name}: {floor!r}: {exc}") from exc

    return floor


def _single(directory: Path, pattern: str, kind: str) -> tuple[Path | None, list[str]]:
    """Resolve a single archive matching pattern in directory."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None, [f"Missing {kind} archive in {directory}"]
    if len(matches) > 1:
        return None, [
            f"Ambiguous: found multiple {kind}s in {directory}: {[p.name for p in matches]}"
        ]
    return matches[0], []


def _compare_archive_contents(
    kind: str,
    path_a: Path,
    path_b: Path,
    names_a: list[str],
    names_b: list[str],
    read_a: Callable[[str], bytes],
    read_b: Callable[[str], bytes],
    metadata_checks: list[
        tuple[str, Callable[[str], bool], Callable[[list[str]], list[str]] | None]
    ],
) -> list[str]:
    """Compare file lists and specified metadata files between two archives."""
    errors: list[str] = []
    if path_a.name != path_b.name:
        errors.append(f"{kind} filenames differ: {path_a.name!r} vs {path_b.name!r}")

    set_a = set(names_a)
    set_b = set(names_b)
    if set_a != set_b:
        only_a = sorted(set_a - set_b)
        only_b = sorted(set_b - set_a)
        msg = f"{kind} file lists differ:"
        if only_a:
            msg += f"\n  Only in {path_a.parent.name}: {only_a}"
        if only_b:
            msg += f"\n  Only in {path_b.parent.name}: {only_b}"
        errors.append(msg)

    for target_name, matcher, filter_fn in metadata_checks:
        candidates_a = [n for n in names_a if matcher(n)]
        candidates_b = [n for n in names_b if matcher(n)]

        if not candidates_a:
            errors.append(f"Missing {target_name} in {kind.lower()} {path_a.name}")
        elif len(candidates_a) > 1:
            errors.append(
                f"Ambiguous: multiple {target_name} files in {path_a.name}: {candidates_a}"
            )

        if not candidates_b:
            errors.append(f"Missing {target_name} in {kind.lower()} {path_b.name}")
        elif len(candidates_b) > 1:
            errors.append(
                f"Ambiguous: multiple {target_name} files in {path_b.name}: {candidates_b}"
            )

        if len(candidates_a) == 1 and len(candidates_b) == 1:
            try:
                data_a = read_a(candidates_a[0]).decode("utf-8")
                data_b = read_b(candidates_b[0]).decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"Failed to decode {target_name} as UTF-8: {exc}")
                continue

            lines_a = data_a.splitlines()
            lines_b = data_b.splitlines()
            if filter_fn is not None:
                lines_a = filter_fn(lines_a)
                lines_b = filter_fn(lines_b)

            if lines_a != lines_b:
                diff = "\n".join(
                    difflib.unified_diff(
                        lines_a,
                        lines_b,
                        fromfile=f"{path_a.parent.name}/{candidates_a[0]}",
                        tofile=f"{path_b.parent.name}/{candidates_b[0]}",
                        lineterm="",
                    )
                )
                label = f"{target_name} (excluding Generator line)" if filter_fn else target_name
                errors.append(f"{kind} {label} differs:\n{diff}")

    return errors


def compare_wheels(wheel_a: Path, wheel_b: Path) -> list[str]:
    """Compare file manifests and metadata (METADATA and WHEEL) of two wheel archives."""
    try:
        with zipfile.ZipFile(wheel_a) as za, zipfile.ZipFile(wheel_b) as zb:
            return _compare_archive_contents(
                "Wheel",
                wheel_a,
                wheel_b,
                za.namelist(),
                zb.namelist(),
                za.read,
                zb.read,
                [
                    (".dist-info/METADATA", lambda n: n.endswith(".dist-info/METADATA"), None),
                    (".dist-info/WHEEL", lambda n: n.endswith(".dist-info/WHEEL"), _filter_wheel),
                ],
            )
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        return [f"Corrupted wheel archive: {exc}"]


def compare_sdists(sdist_a: Path, sdist_b: Path) -> list[str]:
    """Compare file manifests and PKG-INFO of two source distribution archives."""
    try:
        with tarfile.open(sdist_a, "r:gz") as ta, tarfile.open(sdist_b, "r:gz") as tb:

            def _read_member(t: tarfile.TarFile, name: str) -> bytes:
                f = t.extractfile(name)
                if f is None:
                    raise OSError(f"Failed to extract {name} from archive")
                return f.read()

            return _compare_archive_contents(
                "Sdist",
                sdist_a,
                sdist_b,
                ta.getnames(),
                tb.getnames(),
                lambda n: _read_member(ta, n),
                lambda n: _read_member(tb, n),
                [
                    ("PKG-INFO", lambda n: n == "PKG-INFO" or n.endswith("/PKG-INFO"), None),
                ],
            )
    except (tarfile.TarError, EOFError, OSError) as exc:
        return [f"Corrupted sdist archive: {exc}"]


def compare(dir_a: Path, dir_b: Path) -> list[str]:
    """Compare distributions in two directories."""
    errors: list[str] = []

    if not dir_a.is_dir():
        return [f"Directory not found: {dir_a}"]
    if not dir_b.is_dir():
        return [f"Directory not found: {dir_b}"]

    wheel_a, errs = _single(dir_a, "*.whl", "wheel")
    errors.extend(errs)
    wheel_b, errs = _single(dir_b, "*.whl", "wheel")
    errors.extend(errs)

    sdist_a, errs = _single(dir_a, "*.tar.gz", "sdist")
    errors.extend(errs)
    sdist_b, errs = _single(dir_b, "*.tar.gz", "sdist")
    errors.extend(errs)

    if wheel_a is not None and wheel_b is not None:
        errors.extend(compare_wheels(wheel_a, wheel_b))

    if sdist_a is not None and sdist_b is not None:
        errors.extend(compare_sdists(sdist_a, sdist_b))

    return errors


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) >= 1 and args[0] == "--print-uv-build-constraint":
        pyproject = Path(args[1]) if len(args) > 1 else Path("pyproject.toml")
        try:
            floor = extract_uv_build_floor(pyproject)
            print(f"uv_build=={floor}")
            return 0
        except (ValueError, TypeError, OSError) as exc:
            print(f"::error::Failed to extract uv_build floor: {exc}", file=sys.stderr)
            return 1

    if len(args) != 2:
        print(
            "Usage:\n"
            "  check_dist_contents.py <dir_a> <dir_b>\n"
            "  check_dist_contents.py --print-uv-build-constraint [pyproject.toml]",
            file=sys.stderr,
        )
        return 2

    errors = compare(Path(args[0]), Path(args[1]))
    for err in errors:
        print(f"::error::{err}", file=sys.stderr)

    if not errors:
        print("Package contents check passed: wheel and sdist files and metadata match.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
