"""
tests/test_data_credit_card.py
===============================
hqnn_forge.data.load_credit_card_fraud against small synthetic CSVs with the
Kaggle schema, so the suite never needs the real (non-redistributable) file.
"""

from __future__ import annotations

import inspect
import re
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from hqnn_forge.data import (
    CreditCardFraud,
    DatasetDownloadError,
    DatasetNotFoundError,
    load_credit_card_fraud,
)
from hqnn_forge.data import credit_card as cc


def _write_csv(
    path: Path, rows: np.ndarray, header: list[str] | None = None, quote_class: bool = True
) -> Path:
    header = list(cc.COLUMNS) if header is None else header
    lines = [",".join(f'"{h}"' for h in header)]
    for row in rows:
        values = [repr(float(v)) for v in row[:-1]]
        label = str(int(row[-1]))
        values.append(f'"{label}"' if quote_class else label)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rows(n: int = 20, n_fraud: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.column_stack(
        [
            np.arange(n, dtype=float),
            rng.standard_normal((n, 28)),
            rng.uniform(0, 500, n),
            np.zeros(n),
        ]
    )
    rows[:n_fraud, -1] = 1.0
    return rows


@pytest.fixture
def csv_file(tmp_path: Path) -> tuple[Path, np.ndarray]:
    rows = _rows()
    return _write_csv(tmp_path / cc.FILE_NAME, rows), rows


class TestLoading:
    def test_arrays_match_the_file(self, csv_file: tuple) -> None:
        path, rows = csv_file
        data = load_credit_card_fraud(path)
        assert isinstance(data, CreditCardFraud)
        assert data.X.shape == (20, 30) and data.X.dtype == np.float64
        assert data.y.dtype == np.int64 and data.y.tolist() == [1, 1, 1] + [0] * 17
        np.testing.assert_allclose(data.X, rows[:, :-1], rtol=0, atol=0)
        assert data.feature_names == cc.FEATURE_NAMES
        assert data.feature_names[0] == "Time" and data.feature_names[-1] == "Amount"

    def test_directory_path(self, csv_file: tuple) -> None:
        path, _ = csv_file
        assert load_credit_card_fraud(path.parent).X.shape == (20, 30)

    def test_unquoted_labels(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path / "cc.csv", _rows(), quote_class=False)
        assert load_credit_card_fraud(path).y.sum() == 3

    def test_drop_time(self, csv_file: tuple) -> None:
        path, rows = csv_file
        data = load_credit_card_fraud(path, drop_time=True)
        assert data.X.shape == (20, 29) and data.feature_names[0] == "V1"
        np.testing.assert_allclose(data.X, rows[:, 1:-1], rtol=0, atol=0)
        assert data.X.flags["C_CONTIGUOUS"]

    def test_env_var_and_default_location(
        self, csv_file: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path, _ = csv_file
        monkeypatch.setenv(cc.DATA_DIR_ENV, str(path.parent))
        assert load_credit_card_fraud().X.shape == (20, 30)
        monkeypatch.delenv(cc.DATA_DIR_ENV)
        work = tmp_path / "work"
        (work / "data" / "raw").mkdir(parents=True)
        _write_csv(work / "data" / "raw" / cc.FILE_NAME, _rows(5, 1))
        monkeypatch.chdir(work)
        assert load_credit_card_fraud().X.shape == (5, 30)

    def test_strict_counts(self, csv_file: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
        path, _ = csv_file
        with pytest.raises(ValueError, match=r"expected 284807 rows, found 20"):
            load_credit_card_fraud(path, strict=True)
        monkeypatch.setattr(cc, "EXPECTED_ROWS", 20)
        with pytest.raises(ValueError, match=r"expected 492 frauds, found 3"):
            load_credit_card_fraud(path, strict=True)
        monkeypatch.setattr(cc, "EXPECTED_FRAUDS", 3)
        assert load_credit_card_fraud(path, strict=True).X.shape == (20, 30)


class TestSchemaValidation:
    def test_wrong_column_count(self, tmp_path: Path) -> None:
        rows = _rows()[:, 1:]
        path = _write_csv(tmp_path / "x.csv", rows, header=list(cc.COLUMNS[1:]))
        with pytest.raises(ValueError, match="expected 31 columns, found 30"):
            load_credit_card_fraud(path)

    def test_renamed_column(self, tmp_path: Path) -> None:
        header = list(cc.COLUMNS)
        header[30] = "Label"
        path = _write_csv(tmp_path / "x.csv", _rows(), header=header)
        with pytest.raises(ValueError, match=r"column names differ at 30: 'Label' != 'Class'"):
            load_credit_card_fraud(path)

    def test_header_only_file(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path / cc.FILE_NAME, _rows(0, 0))
        with warnings.catch_warnings():
            # np.loadtxt would warn about empty input and then report a width
            # mismatch; the guard must fire before that.
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="no data rows"):
                load_credit_card_fraud(path)

    def test_trailing_blank_lines_are_not_data_rows(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path / cc.FILE_NAME, _rows(0, 0))
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no data rows"):
            load_credit_card_fraud(path)

    def test_non_numeric_value(self, csv_file: tuple) -> None:
        path, _ = csv_file
        text = path.read_text().splitlines()
        text[2] = text[2].replace(text[2].split(",")[3], "abc", 1)
        path.write_text("\n".join(text) + "\n")
        with pytest.raises(ValueError, match="not a number"):
            load_credit_card_fraud(path)

    def test_non_binary_label(self, tmp_path: Path) -> None:
        rows = _rows()
        rows[5, -1] = 2
        path = _write_csv(tmp_path / "x.csv", rows)
        with pytest.raises(ValueError, match=r"Class must be 0 or 1; found \[2\.0\]"):
            load_credit_card_fraud(path)


class TestMissingFile:
    def test_message_contains_the_kaggle_command(self, tmp_path: Path) -> None:
        with pytest.raises(
            DatasetNotFoundError,
            match=r"kaggle datasets download -d mlg-ulb/creditcardfraud -p .* --unzip",
        ) as info:
            load_credit_card_fraud(tmp_path / "missing" / cc.FILE_NAME)
        assert isinstance(info.value, FileNotFoundError)
        assert not (tmp_path / "missing").exists()

    def test_download_without_cli(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cc.shutil, "which", lambda _: None)
        with pytest.raises(DatasetDownloadError, match="Kaggle CLI is not installed") as info:
            load_credit_card_fraud(tmp_path / cc.FILE_NAME, download=True)
        # Not a DatasetNotFoundError: catching that one and retrying with
        # download=True must not come back to this branch forever.
        assert not isinstance(info.value, FileNotFoundError)

    def test_download_runs_the_cli_and_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "dl" / cc.FILE_NAME
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> tuple[int, str]:
            calls.append(cmd)
            _write_csv(Path(cmd[cmd.index("-p") + 1]) / cc.FILE_NAME, _rows(7, 2))
            return 0, ""

        monkeypatch.setattr(cc.shutil, "which", lambda _: "/usr/bin/kaggle")
        monkeypatch.setattr(cc, "_run_cli", fake_run)
        data = load_credit_card_fraud(target, download=True)
        assert calls == [
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "mlg-ulb/creditcardfraud",
                "-p",
                str(target.parent),
                "--unzip",
            ]
        ]
        assert data.X.shape == (7, 30)

    def test_download_into_a_directory_that_does_not_exist_yet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The documented "data/raw" call on a fresh clone: the directory is only
        # created by the download, so it must still be read as a directory.
        directory = tmp_path / "data" / "raw"
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> tuple[int, str]:
            calls.append(cmd)
            _write_csv(Path(cmd[cmd.index("-p") + 1]) / cc.FILE_NAME, _rows(5, 1))
            return 0, ""

        monkeypatch.setattr(cc.shutil, "which", lambda _: "/usr/bin/kaggle")
        monkeypatch.setattr(cc, "_run_cli", fake_run)
        data = load_credit_card_fraud(directory, download=True)
        assert calls[0][calls[0].index("-p") + 1] == str(directory)
        assert (directory / cc.FILE_NAME).exists()
        assert data.X.shape == (5, 30)

    def test_missing_directory_resolves_to_the_file_inside_it(self, tmp_path: Path) -> None:
        directory = tmp_path / "data" / "raw"
        with pytest.raises(DatasetNotFoundError, match=re.escape(str(directory / cc.FILE_NAME))):
            load_credit_card_fraud(directory)
        assert not directory.exists()

    def test_download_rejects_another_file_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            raise AssertionError("the CLI must not run for a name it cannot produce")

        monkeypatch.setattr(cc.shutil, "which", lambda _: "/usr/bin/kaggle")
        monkeypatch.setattr(cc, "_run_cli", fail)
        with pytest.raises(DatasetDownloadError, match=f"path ending in {cc.FILE_NAME}"):
            load_credit_card_fraud(tmp_path / "fraud.csv", download=True)

    def test_download_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cc.shutil, "which", lambda _: "/usr/bin/kaggle")
        monkeypatch.setattr(cc, "_run_cli", lambda cmd, **kw: (1, "403 Forbidden"))
        with pytest.raises(DatasetDownloadError, match=r"exit 1\):\n403 Forbidden"):
            load_credit_card_fraud(tmp_path / cc.FILE_NAME, download=True)

    def test_download_that_creates_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cc.shutil, "which", lambda _: "/usr/bin/kaggle")
        monkeypatch.setattr(cc, "_run_cli", lambda cmd, **kw: (0, ""))
        with pytest.raises(DatasetDownloadError, match="was not created"):
            load_credit_card_fraud(tmp_path / cc.FILE_NAME, download=True)


def _script(tmp_path: Path, body: str) -> list[str]:
    """A local stand-in for the Kaggle CLI: a Python script, no network."""
    script = tmp_path / "fake_kaggle.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, "-u", str(script)]


class TestStreamingDownload:
    """
    #181: the real _run_cli against local scripts.  Output reaches the caller
    as it is produced, a stall and an overlong run are told apart, and a
    failed download leaves no partial file behind.
    """

    def test_progress_is_passed_through_as_it_arrives(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        cmd = _script(
            tmp_path,
            """
            import sys, time
            for pct in (10, 50, 100):
                sys.stderr.write(f"\\r{pct}%")  # a progress bar: no newlines
                sys.stderr.flush()
                time.sleep(0.05)
            print("Downloaded")
            """,
        )
        code, tail = cc._run_cli(cmd, timeout=30, stall_timeout=30)
        out, err = capfd.readouterr()
        assert code == 0
        assert "\r10%\r50%\r100%" in err
        assert "Downloaded" in out
        assert "100%" in tail and "Downloaded" in tail

    def test_a_stall_is_stopped_and_named(self, tmp_path: Path) -> None:
        cmd = _script(
            tmp_path,
            """
            import sys, time
            sys.stderr.write("\\r5%"); sys.stderr.flush()
            time.sleep(60)
            """,
        )
        start = time.monotonic()
        with pytest.raises(
            DatasetDownloadError,
            match=r"(?s)stalled: no output for 0.5 s.*run it by hand:\n    .*fake_kaggle\.py.*5%",
        ):
            cc._run_cli(cmd, timeout=30, stall_timeout=0.5)
        assert time.monotonic() - start < 10

    def test_output_is_drained_when_there_is_no_console(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Under pythonw sys.stdout is None.  160 KiB on stdout overfills a
        # 64 KiB pipe, so a pump that stopped reading would block the script
        # and the steady output would be misreported as a stall.
        cmd = _script(
            tmp_path,
            """
            import sys, time
            for _ in range(40):
                sys.stdout.write("x" * 4096); sys.stdout.flush()
                sys.stderr.write("\\r."); sys.stderr.flush()
                time.sleep(0.05)
            """,
        )
        monkeypatch.setattr(sys, "stdout", None)
        code, tail = cc._run_cli(cmd, timeout=30, stall_timeout=1.0)
        # Exit 0 means all 160 KiB were read; the two pumps interleave, so the
        # tail's last characters may come from either stream.
        assert code == 0
        assert tail.count("x") > 1900

    def test_steady_output_is_not_a_stall_but_the_total_bound_holds(self, tmp_path: Path) -> None:
        cmd = _script(
            tmp_path,
            """
            import sys, time
            while True:
                sys.stderr.write("\\r."); sys.stderr.flush()
                time.sleep(0.1)
            """,
        )
        with pytest.raises(DatasetDownloadError, match=r"did not finish within 1.5 s"):
            cc._run_cli(cmd, timeout=1.5, stall_timeout=0.5)

    def test_a_failed_download_leaves_no_partial_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "dl" / cc.FILE_NAME
        writer = _script(
            tmp_path,
            f"""
            import sys, time, pathlib
            d = pathlib.Path(sys.argv[sys.argv.index("-p") + 1])
            (d / "{cc._ZIP_NAME}").write_bytes(b"partial")
            (d / "{cc.FILE_NAME}").write_text("Time,V1\\n0,")
            time.sleep(60)
            """,
        )
        monkeypatch.setattr(cc.shutil, "which", lambda _: "/usr/bin/kaggle")
        monkeypatch.setattr(cc, "_download_command", lambda d: [*writer, "-p", str(d)])
        monkeypatch.setattr(cc, "_STALL_TIMEOUT", 0.5)
        with pytest.raises(DatasetDownloadError, match="stalled"):
            load_credit_card_fraud(target, download=True)
        assert not target.exists()
        assert not (target.parent / cc._ZIP_NAME).exists()

    def test_files_that_were_there_before_are_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "dl"
        directory.mkdir()
        (directory / cc._ZIP_NAME).write_bytes(b"the user's own archive")
        monkeypatch.setattr(cc.shutil, "which", lambda _: "/usr/bin/kaggle")
        monkeypatch.setattr(cc, "_run_cli", lambda cmd, **kw: (1, "boom"))
        with pytest.raises(DatasetDownloadError, match="exit 1"):
            load_credit_card_fraud(directory, download=True)
        assert (directory / cc._ZIP_NAME).read_bytes() == b"the user's own archive"

    def test_the_loader_bounds_the_cli_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both bounds are fixed, not caller-settable (#390 decides whether they become so):
        # the loader passes the module's private values, read at call time.
        seen: dict[str, object] = {}

        def fake_run(cmd: list[str], **kwargs: object) -> tuple[int, str]:
            seen.update(kwargs)
            return 1, ""

        monkeypatch.setattr(cc.shutil, "which", lambda _: "/usr/bin/kaggle")
        monkeypatch.setattr(cc, "_run_cli", fake_run)
        with pytest.raises(DatasetDownloadError):
            load_credit_card_fraud(tmp_path / cc.FILE_NAME, download=True)
        assert seen == {"timeout": 3600.0, "stall_timeout": 120.0}
        params = inspect.signature(load_credit_card_fraud).parameters
        assert "download_timeout" not in params and "stall_timeout" not in params
