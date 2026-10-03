"""
hqnn_forge.data.credit_card
===========================
Loader for the Kaggle "Credit Card Fraud Detection" dataset, the benchmark the
library's design and the accompanying thesis target.

The CSV (``creditcard.csv``, ~150 MB) is not redistributable and is not
shipped with the package.  Download it once with the Kaggle CLI::

    kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

or call ``load_credit_card_fraud(download=True)``, which runs that command.

Schema: 31 columns -- ``Time``, ``V1`` … ``V28`` (PCA-anonymised), ``Amount``
and the label ``Class`` (1 = fraud).  The published file has 284,807 rows and
492 frauds; ``strict=True`` checks both.

Loading uses NumPy only, like the rest of the library.
"""

from __future__ import annotations

import codecs
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, NamedTuple, TextIO

import numpy as np
import numpy.typing as npt

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
FILE_NAME = "creditcard.csv"
#: Environment variable naming the directory that holds ``creditcard.csv``.
DATA_DIR_ENV = "HQNN_FORGE_DATA"
DEFAULT_DIR = Path("data") / "raw"

FEATURE_NAMES: tuple[str, ...] = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
COLUMNS: tuple[str, ...] = (*FEATURE_NAMES, "Class")
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492
#: The archive the CLI writes next to the CSV before ``--unzip`` extracts it.
_ZIP_NAME = "creditcardfraud.zip"
#: Seconds without any output from the CLI after which a download is stalled.
#: Its progress bar redraws several times a second while bytes arrive; the
#: silent unzip of the ~150 MB CSV at the end takes seconds, not minutes.
_STALL_TIMEOUT = 120.0
#: Seconds a whole download may take: ~150 MB at well under 1 MB/s.
_DOWNLOAD_TIMEOUT = 3600.0
#: Characters of CLI output kept for an error message.
_TAIL_CHARS = 2000


class DatasetNotFoundError(FileNotFoundError):
    """The dataset file is missing and downloading was not requested."""


class DatasetDownloadError(RuntimeError):
    """Downloading was requested but could not be carried out or did not produce the file."""


class CreditCardFraud(NamedTuple):
    """
    Attributes
    ----------
    X:
        Features, shape ``(n_samples, n_features)``, float64.
    y:
        Labels, shape ``(n_samples,)``, int64, 1 = fraud.
    feature_names:
        Column name for each column of ``X``.
    """

    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.int64]
    feature_names: tuple[str, ...]


def _download_command(directory: Path) -> list[str]:
    return [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(directory),
        "--unzip",
    ]


def _resolve_path(path: str | os.PathLike[str] | None) -> Path:
    """
    Resolve ``path`` to the CSV file.

    A path that names an existing directory, or that does not end in ``.csv``, is
    treated as the directory holding ``creditcard.csv``.  Testing the suffix rather
    than only :meth:`~pathlib.Path.is_dir` matters for ``download=True``, where the
    directory usually does not exist yet.
    """
    if path is not None:
        p = Path(path)
        return p / FILE_NAME if p.is_dir() or p.suffix.lower() != ".csv" else p
    base = os.environ.get(DATA_DIR_ENV)
    return (Path(base) if base else DEFAULT_DIR) / FILE_NAME


def _run_cli(
    cmd: list[str], *, timeout: float | None, stall_timeout: float | None
) -> tuple[int, str]:
    """
    Run ``cmd``, passing its output through to this process's stdout and
    stderr as it arrives, and return ``(returncode, tail of its output)``.

    Output is read in raw chunks, not lines: the Kaggle CLI's progress bar
    redraws with carriage returns, and a line reader would see nothing until
    the download ended.  Every chunk counts as a sign of life for
    ``stall_timeout``.

    Raises
    ------
    DatasetDownloadError
        After killing the process, when it prints nothing for
        ``stall_timeout`` seconds (a stalled connection) or runs longer than
        ``timeout`` seconds in total (too large or too slow), with a message
        saying which.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lock = threading.Lock()
    last_output = [time.monotonic()]
    tail: list[str] = []

    def pump(source: IO[bytes], sink: TextIO | None) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        read = getattr(source, "read1", source.read)
        while chunk := read(4096):
            text = decoder.decode(chunk)
            if sink is not None:
                try:
                    sink.write(text)
                    sink.flush()
                except (OSError, ValueError, AttributeError):
                    # A closed or broken console, one that cannot encode the
                    # text (UnicodeEncodeError is a ValueError), or a stand-in
                    # without write().  Echoing is best effort; draining is not:
                    # a pump that died would let the pipe fill, block the
                    # CLI and have it misreported as a stall.
                    sink = None
            with lock:
                last_output[0] = time.monotonic()
                tail.append(text)
                joined = "".join(tail)[-_TAIL_CHARS:]
                tail[:] = [joined]

    assert proc.stdout is not None and proc.stderr is not None
    pumps = [
        threading.Thread(target=pump, args=(proc.stdout, sys.stdout), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, sys.stderr), daemon=True),
    ]
    for thread in pumps:
        thread.start()

    start = time.monotonic()
    reason = None
    try:
        while proc.poll() is None:
            now = time.monotonic()
            with lock:
                silent = now - last_output[0]
            if timeout is not None and now - start > timeout:
                reason = f"did not finish within {timeout:g} s (download too large or too slow)"
            elif stall_timeout is not None and silent > stall_timeout:
                reason = f"stalled: no output for {stall_timeout:g} s (connection stalled)"
            if reason is not None:
                break
            time.sleep(0.05)
    finally:
        # Also on Ctrl-C: the CLI must not keep writing into the directory.
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for thread in pumps:
            thread.join(timeout=5)
    output = "".join(tail).strip()
    if reason is not None:
        raise DatasetDownloadError(
            f"Kaggle download {reason}; the process was stopped.  "
            f"Retry, or run it by hand:\n    {' '.join(cmd)}"
            + (f"\nLast output:\n{output}" if output else "")
        )
    return proc.returncode, output


def _download(
    target: Path,
    *,
    timeout: float | None = _DOWNLOAD_TIMEOUT,
    stall_timeout: float | None = _STALL_TIMEOUT,
) -> None:
    # The archive unpacks under FILE_NAME, so any other file name can never be
    # produced.  Refuse up front rather than after a ~150 MB download.
    if target.name != FILE_NAME:
        raise DatasetDownloadError(
            f"download=True fetches {FILE_NAME}, but the resolved path is {target}.  "
            f"Pass a directory, or a path ending in {FILE_NAME}."
        )
    if shutil.which("kaggle") is None:
        raise DatasetDownloadError(
            f"{target} does not exist and the Kaggle CLI is not installed.  "
            f"Install it (pip install kaggle), configure ~/.kaggle/kaggle.json, "
            f"then retry, or download manually:\n    {' '.join(_download_command(target.parent))}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Whatever this call creates and does not finish is removed on failure, so a
    # retry does not find a truncated CSV (or a half-written archive) and take
    # it for the dataset.  Files that were there before are never touched.
    new_files = [
        target.parent / name
        for name in (FILE_NAME, _ZIP_NAME)
        if not (target.parent / name).exists()
    ]
    try:
        returncode, output = _run_cli(
            _download_command(target.parent), timeout=timeout, stall_timeout=stall_timeout
        )
        if returncode != 0:
            raise DatasetDownloadError(f"Kaggle download failed (exit {returncode}):\n{output}")
        if not target.exists():
            raise DatasetDownloadError(f"Kaggle download finished but {target} was not created.")
    except BaseException:
        # BaseException: a Ctrl-C mid-download must not leave a partial file either.
        for leftover in new_files:
            leftover.unlink(missing_ok=True)
        raise


def _read_header(path: Path) -> tuple[list[str], bool]:
    """Return the column names and whether at least one data row follows them."""
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline()
        rest = fh.readline()
        while rest != "" and rest.strip() == "":
            rest = fh.readline()
    names = [name.strip().strip('"') for name in first.rstrip("\r\n").split(",")]
    return names, rest != ""


def load_credit_card_fraud(
    path: str | os.PathLike[str] | None = None,
    *,
    download: bool = False,
    drop_time: bool = False,
    strict: bool = False,
) -> CreditCardFraud:
    """
    Load ``creditcard.csv`` as NumPy arrays.

    Parameters
    ----------
    path:
        The CSV file, or a directory containing ``creditcard.csv``; a path that
        does not end in ``.csv`` is taken as the directory.  Default:
        ``$HQNN_FORGE_DATA/creditcard.csv`` if that variable is set, else
        ``data/raw/creditcard.csv`` relative to the working directory (the
        benchmark repository's layout).
    download:
        If the file is missing, fetch it with the Kaggle CLI into the file's
        directory.  Requires ``kaggle`` on ``PATH`` and configured credentials.
        The CSV is ~150 MB once unpacked: seconds on a fast link, several
        minutes on a slow one.  The CLI's progress bar is passed
        through to this process's stderr as it runs.  The download is
        stopped if the CLI prints nothing for 120 s (a stalled connection)
        or runs past 3600 s in total; the error then repeats the command, to
        run by hand on a link too slow for that.  On any failure,
        including a timeout or Ctrl-C, the files this call created are
        removed, so a retry starts clean.
    drop_time:
        Drop the ``Time`` column (seconds since the first transaction), which
        the benchmark excludes as a leakage-prone ordering feature.
    strict:
        Also require the published row count (284,807) and fraud count (492).

    Returns
    -------
    CreditCardFraud

    Raises
    ------
    DatasetNotFoundError
        The file is missing and ``download`` is False.  The message contains
        the Kaggle command that fetches it.
    DatasetDownloadError
        ``download`` is True but the file could not be fetched: the Kaggle CLI
        is missing, the command failed, it produced no file, it stalled, or it
        ran past its total time -- the message says which.  Distinct from
        :class:`DatasetNotFoundError` so that retrying with ``download=True``
        after catching that one cannot loop.
    ValueError
        The file does not have the expected columns, has no data rows, has
        non-numeric values, labels other than 0/1, or (with ``strict``) the
        wrong row or fraud count.
    """
    csv = _resolve_path(path)
    if not csv.exists():
        if not download:
            raise DatasetNotFoundError(
                f"{csv} not found.  Download the Kaggle Credit Card Fraud dataset with:\n"
                f"    {' '.join(_download_command(csv.parent))}\n"
                f"or pass download=True, or point path at the file or its directory "
                f"(${DATA_DIR_ENV} names the directory)."
            )
        # Looked up at call time, so a test can monkeypatch them.  They are
        # deliberately not public yet; see #390.
        _download(csv, timeout=_DOWNLOAD_TIMEOUT, stall_timeout=_STALL_TIMEOUT)

    header, has_rows = _read_header(csv)
    if tuple(header) != COLUMNS:
        if len(header) != len(COLUMNS):
            detail = f"expected {len(COLUMNS)} columns, found {len(header)}"
        else:
            diffs = [
                f"{i}: {got!r} != {want!r}"
                for i, (got, want) in enumerate(zip(header, COLUMNS))
                if got != want
            ]
            detail = (
                "column names differ at " + ", ".join(diffs[:5]) + (" …" if len(diffs) > 5 else "")
            )
        raise ValueError(f"{csv} does not look like the Kaggle creditcard.csv: {detail}.")
    if not has_rows:
        raise ValueError(f"{csv} has the expected header but no data rows.")

    try:
        data = np.loadtxt(csv, delimiter=",", skiprows=1, quotechar='"', dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"{csv} contains a value that is not a number: {exc}") from exc
    if data.shape[1] != len(COLUMNS):
        raise ValueError(f"{csv}: rows have {data.shape[1]} values, expected {len(COLUMNS)}.")

    labels = data[:, -1]
    if not np.isin(labels, (0.0, 1.0)).all():
        bad = np.unique(labels[~np.isin(labels, (0.0, 1.0))])[:5]
        raise ValueError(f"{csv}: Class must be 0 or 1; found {bad.tolist()}.")
    y = labels.astype(np.int64)
    X = data[:, :-1]
    names = FEATURE_NAMES

    if strict:
        if X.shape[0] != EXPECTED_ROWS:
            raise ValueError(f"{csv}: expected {EXPECTED_ROWS} rows, found {X.shape[0]}.")
        if int(y.sum()) != EXPECTED_FRAUDS:
            raise ValueError(f"{csv}: expected {EXPECTED_FRAUDS} frauds, found {int(y.sum())}.")

    if drop_time:
        X = X[:, 1:]
        names = names[1:]
    return CreditCardFraud(np.ascontiguousarray(X), y, names)
