"""
tests/test_data_uci.py
======================
hqnn_forge.data.uci against small synthetic files in the formats of the real
UCI files (the Taiwan header's leading spaces and CRLF line ends, the churn
header's double spaces, the cervical file's ``?``), and a fake ``urlopen``
serving in-memory zips, so the suite never touches the network.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import urllib.error
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Self

import numpy as np
import pytest

from hqnn_forge.data import (
    BinaryDataset,
    DatasetDownloadError,
    DatasetNotFoundError,
    load_cervical_cancer_risk,
    load_iranian_churn,
    load_taiwanese_bankruptcy,
    uci,
)
from hqnn_forge.preprocessing import PCANormalizer, stratified_kfold

# ---------------------------------------------------------------------------
# Synthetic files
# ---------------------------------------------------------------------------

TAIWAN_NAMES = [f"Ratio {i} (¥)" if i == 7 else f"Ratio {i}" for i in range(95)]


def _taiwan_text(n: int = 30, n_pos: int = 3, seed: int = 0) -> tuple[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = np.column_stack([np.zeros(n), rng.random((n, 95))])
    data[:n_pos, 0] = 1
    data[0, 5] = 1e10  # the real file has ratios this large
    lines = [", ".join(["Bankrupt?", *TAIWAN_NAMES])]
    lines += [",".join([str(int(r[0])), *(repr(float(v)) for v in r[1:])]) for r in data]
    return "\r\n".join(lines) + "\r\n", data


RAW_CHURN_HEADER = [
    "Call  Failure",
    "Complains",
    "Subscription  Length",
    "Charge  Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
    "Status",
    "Age",
    "Customer Value",
    "Churn",
]


def _churn_text(n: int = 30, n_pos: int = 5, seed: int = 1) -> tuple[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = np.column_stack([rng.integers(0, 5000, (n, 12)).astype(float), rng.random(n) * 500])
    data = np.column_stack([data, np.zeros(n)])
    data[:n_pos, -1] = 1
    lines = [",".join(RAW_CHURN_HEADER)]
    lines += [
        ",".join(f"{v:g}" if j < 12 else repr(float(v)) for j, v in enumerate(r[:-1]))
        + f",{int(r[-1])}"
        for r in data
    ]
    return "\n".join(lines) + "\n", data


def _cervical_text(missing_rows: tuple[int, ...] = (2, 5)) -> tuple[str, np.ndarray]:
    """20 patients; Biopsy positive for rows 0-3; ``?`` in a feature for ``missing_rows``."""
    columns = uci.CERVICAL_CANCER_RISK_COLUMNS
    rng = np.random.default_rng(2)
    data = rng.integers(0, 4, (20, len(columns))).astype(float)
    data[:, -1] = 0
    data[:4, -1] = 1
    # The excluded examinations agree with the label, as a leak would.
    for name in ("Hinselmann", "Schiller", "Citology"):
        data[:, columns.index(name)] = data[:, -1]
    cells = [[f"{v:.1f}" for v in row] for row in data]
    for row in range(20):  # the time-since columns are mostly missing
        for name in ("STDs: Time since first diagnosis", "STDs: Time since last diagnosis"):
            cells[row][columns.index(name)] = "?"
    for row in missing_rows:
        cells[row][columns.index("IUD")] = "?"
    lines = [",".join(columns), *(",".join(r) for r in cells)]
    return "\n".join(lines) + "\n", data


def _write(directory: Path, archive: uci._Archive, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / archive.file_name
    path.write_bytes(text.encode("utf-8"))
    return path


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buffer.getvalue()


class _Response(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes | BaseException) -> list[str]:
    """Replace urlopen with one returning ``payload`` (or raising it); log the URLs."""
    requested: list[str] = []

    def urlopen(url: str, timeout: float) -> _Response:
        requested.append(url)
        assert timeout == uci.DOWNLOAD_TIMEOUT
        if isinstance(payload, BaseException):
            raise payload
        return _Response(payload)

    monkeypatch.setattr(uci.urllib.request, "urlopen", urlopen)
    return requested


def _with_sha(monkeypatch: pytest.MonkeyPatch, attr: str, content: bytes) -> uci._Archive:
    """Point the archive constant ``attr`` at ``content``'s SHA-256."""
    archive = dataclasses.replace(getattr(uci, attr), sha256=hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(uci, attr, archive)
    return archive


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestTaiwaneseBankruptcy:
    def test_arrays_match_the_file(self, tmp_path: Path) -> None:
        text, data = _taiwan_text()
        _write(tmp_path, uci.TAIWANESE_BANKRUPTCY, text)
        ds = load_taiwanese_bankruptcy(tmp_path)
        assert isinstance(ds, BinaryDataset)
        assert ds.X.shape == (30, 95) and ds.X.dtype == np.float64
        assert ds.y.dtype == np.int64 and ds.y.tolist() == [1, 1, 1] + [0] * 27
        np.testing.assert_array_equal(ds.X, data[:, 1:])
        # Leading spaces stripped, non-ASCII kept.
        assert ds.feature_names == tuple(TAIWAN_NAMES)

    def test_wrong_target_column(self, tmp_path: Path) -> None:
        text, _ = _taiwan_text()
        _write(tmp_path, uci.TAIWANESE_BANKRUPTCY, text.replace("Bankrupt?", "Label", 1))
        with pytest.raises(ValueError, match="starting with 'Bankrupt\\?'"):
            load_taiwanese_bankruptcy(tmp_path)

    def test_label_other_than_0_or_1(self, tmp_path: Path) -> None:
        text, _ = _taiwan_text()
        lines = text.split("\r\n")
        lines[1] = "2" + lines[1][1:]
        _write(tmp_path, uci.TAIWANESE_BANKRUPTCY, "\r\n".join(lines))
        with pytest.raises(ValueError, match="Bankrupt\\? must be 0 or 1; found \\[2.0\\]"):
            load_taiwanese_bankruptcy(tmp_path)


class TestIranianChurn:
    def test_arrays_match_the_file(self, tmp_path: Path) -> None:
        text, data = _churn_text()
        path = _write(tmp_path, uci.IRANIAN_CHURN, text)
        ds = load_iranian_churn(path)  # the file itself, not its directory
        assert ds.X.shape == (30, 13) and ds.X.dtype == np.float64
        assert ds.y.tolist() == [1] * 5 + [0] * 25
        np.testing.assert_array_equal(ds.X, data[:, :-1])
        assert ds.feature_names == uci.IRANIAN_CHURN_FEATURES
        assert ds.feature_names[0] == "Call Failure"  # double space collapsed

    def test_drop_duplicates_keeps_the_first_in_file_order(self, tmp_path: Path) -> None:
        text, data = _churn_text(n=6, n_pos=2)
        lines = text.splitlines()
        # Repeat row 3 (negative) after row 5, and row 0 (positive) at the end.
        lines += [lines[4], lines[1]]
        _write(tmp_path, uci.IRANIAN_CHURN, "\n".join(lines) + "\n")
        assert load_iranian_churn(tmp_path).X.shape == (8, 13)
        ds = load_iranian_churn(tmp_path, drop_duplicates=True)
        np.testing.assert_array_equal(ds.X, data[:, :-1])
        assert ds.y.tolist() == [1, 1, 0, 0, 0, 0]

    def test_renamed_column(self, tmp_path: Path) -> None:
        text, _ = _churn_text()
        _write(tmp_path, uci.IRANIAN_CHURN, text.replace("Tariff Plan", "Plan", 1))
        with pytest.raises(ValueError, match="differ at 9: 'Plan' != 'Tariff Plan'"):
            load_iranian_churn(tmp_path)

    def test_non_numeric_value(self, tmp_path: Path) -> None:
        text, _ = _churn_text()
        lines = text.splitlines()
        lines[3] = "x" + lines[3][lines[3].index(",") :]
        _write(tmp_path, uci.IRANIAN_CHURN, "\n".join(lines))
        with pytest.raises(ValueError, match="data row 3, column 1 is not a number: 'x'"):
            load_iranian_churn(tmp_path)

    def test_ragged_row(self, tmp_path: Path) -> None:
        text, _ = _churn_text()
        lines = text.splitlines()
        lines[2] += ",7"
        _write(tmp_path, uci.IRANIAN_CHURN, "\n".join(lines))
        with pytest.raises(ValueError, match="data row 2 has 15 values, expected 14"):
            load_iranian_churn(tmp_path)

    def test_header_only_and_blank_lines(self, tmp_path: Path) -> None:
        _write(tmp_path, uci.IRANIAN_CHURN, ",".join(RAW_CHURN_HEADER) + "\n\n\n")
        with pytest.raises(ValueError, match="no data rows"):
            load_iranian_churn(tmp_path)


class TestCervicalCancerRisk:
    def test_label_and_excluded_columns(self, tmp_path: Path) -> None:
        text, data = _cervical_text(missing_rows=())
        _write(tmp_path, uci.CERVICAL_CANCER_RISK, text)
        ds = load_cervical_cancer_risk(tmp_path)
        assert ds.X.shape == (20, 30) and ds.X.dtype == np.float64
        assert ds.y.tolist() == [1] * 4 + [0] * 16
        assert ds.feature_names == uci.CERVICAL_CANCER_RISK_FEATURES
        for leak in ("Hinselmann", "Schiller", "Citology", "Biopsy"):
            assert leak not in ds.feature_names
        columns = uci.CERVICAL_CANCER_RISK_COLUMNS
        kept = [columns.index(name) for name in ds.feature_names]
        np.testing.assert_array_equal(ds.X, data[:, kept])

    def test_missing_drop_removes_incomplete_patients(self, tmp_path: Path) -> None:
        text, data = _cervical_text(missing_rows=(2, 5))
        _write(tmp_path, uci.CERVICAL_CANCER_RISK, text)
        ds = load_cervical_cancer_risk(tmp_path)
        rows = [r for r in range(20) if r not in (2, 5)]
        assert ds.X.shape == (18, 30) and not np.isnan(ds.X).any()
        assert ds.y.tolist() == data[rows, -1].astype(int).tolist()
        assert int(ds.y.sum()) == 3  # row 2 was a positive

    def test_missing_keep_returns_nan(self, tmp_path: Path) -> None:
        text, _ = _cervical_text(missing_rows=(2, 5))
        _write(tmp_path, uci.CERVICAL_CANCER_RISK, text)
        ds = load_cervical_cancer_risk(tmp_path, missing="keep")
        assert ds.X.shape == (20, 30) and int(ds.y.sum()) == 4
        iud = ds.feature_names.index("IUD")
        assert np.flatnonzero(np.isnan(ds.X).any(axis=1)).tolist() == [2, 5]
        assert np.isnan(ds.X[[2, 5], iud]).all()

    def test_missing_label_is_refused(self, tmp_path: Path) -> None:
        text, _ = _cervical_text(missing_rows=())
        lines = text.splitlines()
        lines[1] = lines[1][: lines[1].rindex(",")] + ",?"
        _write(tmp_path, uci.CERVICAL_CANCER_RISK, "\n".join(lines))
        with pytest.raises(ValueError, match="Biopsy is missing"):
            load_cervical_cancer_risk(tmp_path)

    def test_unknown_missing_mode(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="missing must be 'drop' or 'keep'"):
            load_cervical_cancer_risk(tmp_path, missing="impute")  # type: ignore[arg-type]

    def test_question_mark_elsewhere_is_not_a_number_for_other_loaders(
        self, tmp_path: Path
    ) -> None:
        text, _ = _churn_text()
        lines = text.splitlines()
        lines[1] = "?" + lines[1][lines[1].index(",") :]
        _write(tmp_path, uci.IRANIAN_CHURN, "\n".join(lines))
        with pytest.raises(ValueError, match="not a number: '\\?'"):
            load_iranian_churn(tmp_path)


# ---------------------------------------------------------------------------
# Locating, strict mode and downloading
# ---------------------------------------------------------------------------

LOADERS: list[tuple[Callable[..., BinaryDataset], str, Callable[[], tuple[str, np.ndarray]]]] = [
    (load_taiwanese_bankruptcy, "TAIWANESE_BANKRUPTCY", _taiwan_text),
    (load_iranian_churn, "IRANIAN_CHURN", _churn_text),
    (load_cervical_cancer_risk, "CERVICAL_CANCER_RISK", _cervical_text),
]
IDS = ["taiwan", "churn", "cervical"]


@pytest.mark.parametrize(("load", "attr", "make"), LOADERS, ids=IDS)
class TestEveryLoader:
    def test_env_var_and_default_location(
        self,
        load: Callable[..., BinaryDataset],
        attr: str,
        make: Callable[[], tuple[str, np.ndarray]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        archive = getattr(uci, attr)
        _write(tmp_path / "env", archive, make()[0])
        monkeypatch.setenv(uci.DATA_DIR_ENV, str(tmp_path / "env"))
        n = load().X.shape[0]
        monkeypatch.delenv(uci.DATA_DIR_ENV)
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "data" / "raw", archive, make()[0])
        assert load().X.shape[0] == n

    def test_missing_file_names_the_download(
        self,
        load: Callable[..., BinaryDataset],
        attr: str,
        make: Callable[[], tuple[str, np.ndarray]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        requested = _serve(monkeypatch, AssertionError("must not download"))
        with pytest.raises(DatasetNotFoundError, match="download=True") as info:
            load(tmp_path)
        assert getattr(uci, attr).url in str(info.value)
        assert requested == []

    def test_strict_requires_the_published_file(
        self,
        load: Callable[..., BinaryDataset],
        attr: str,
        make: Callable[[], tuple[str, np.ndarray]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write(tmp_path, getattr(uci, attr), make()[0])
        load(tmp_path)  # fine without strict
        with pytest.raises(ValueError, match="is not the published file"):
            load(tmp_path, strict=True)

    def test_strict_checks_the_counts_of_a_matching_file(
        self,
        load: Callable[..., BinaryDataset],
        attr: str,
        make: Callable[[], tuple[str, np.ndarray]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A file with the right checksum but not the published counts can only
        # arise here, by patching the checksum; the count check still runs.
        path = _write(tmp_path, getattr(uci, attr), make()[0])
        _with_sha(monkeypatch, attr, path.read_bytes())
        with pytest.raises(ValueError, match="expected .* rows with .* positives"):
            load(tmp_path, strict=True)

    def test_download_writes_the_verified_file(
        self,
        load: Callable[..., BinaryDataset],
        attr: str,
        make: Callable[[], tuple[str, np.ndarray]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        content = make()[0].encode("utf-8")
        archive = _with_sha(monkeypatch, attr, content)
        requested = _serve(monkeypatch, _zip({archive.file_name: content, "other.txt": b"x"}))
        target = tmp_path / "not" / "there" / "yet"
        ds = load(target, download=True)
        assert requested == [archive.url]
        assert (target / archive.file_name).read_bytes() == content
        assert sorted(p.name for p in target.iterdir()) == [archive.file_name]  # no .part
        # A second load reads the file and does not download again.
        assert load(target, download=True).X.shape == ds.X.shape
        assert requested == [archive.url]

    def test_download_refuses_a_changed_file(
        self,
        load: Callable[..., BinaryDataset],
        attr: str,
        make: Callable[[], tuple[str, np.ndarray]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        archive = getattr(uci, attr)  # the real checksum; the synthetic file differs
        _serve(monkeypatch, _zip({archive.file_name: make()[0].encode("utf-8")}))
        with pytest.raises(DatasetDownloadError, match="published file has changed"):
            load(tmp_path, download=True)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (b"not a zip", "is not a zip archive"),
            (_zip({"unrelated.csv": b"a,b\n"}), "has no .*; the archive has changed"),
            (urllib.error.URLError("no route to host"), "failed: .*no route to host"),
            (TimeoutError("timed out"), "failed: timed out"),
        ],
        ids=["bad-zip", "member-missing", "url-error", "timeout"],
    )
    def test_download_failures(
        self,
        load: Callable[..., BinaryDataset],
        attr: str,
        make: Callable[[], tuple[str, np.ndarray]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        payload: bytes | BaseException,
        match: str,
    ) -> None:
        _serve(monkeypatch, payload)
        with pytest.raises(DatasetDownloadError, match=match) as info:
            load(tmp_path, download=True)
        assert "manually" in str(info.value)
        assert list(tmp_path.iterdir()) == []

    def test_output_runs_through_cv_and_pca(
        self,
        load: Callable[..., BinaryDataset],
        attr: str,
        make: Callable[[], tuple[str, np.ndarray]],
        tmp_path: Path,
    ) -> None:
        """
        The shape and dtype stratified_kfold and PCANormalizer expect.  Each
        fold is standardised on its training part first, as the module
        docstring advises: the 1e10 bankruptcy ratio would otherwise dominate
        the covariance.
        """
        _write(tmp_path, getattr(uci, attr), make()[0])
        ds = load(tmp_path)
        assert ds.X.ndim == 2 and ds.X.dtype == np.float64 and ds.X.flags.c_contiguous
        assert ds.y.shape == (ds.X.shape[0],) and set(ds.y.tolist()) == {0, 1}
        assert len(ds.feature_names) == ds.X.shape[1]
        for train, test in stratified_kfold(ds.y, n_splits=2, random_state=0):
            mean, std = ds.X[train].mean(axis=0), ds.X[train].std(axis=0)
            std[std == 0] = 1.0  # constant columns, as in the real files
            pca = PCANormalizer(n_components=2).fit((ds.X[train] - mean) / std)
            encoded = pca.transform((ds.X[test] - mean) / std)
            assert encoded.shape == (test.size, 2)
            assert bool(np.isfinite(encoded.numpy()).all())
