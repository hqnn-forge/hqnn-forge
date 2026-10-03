"""
hqnn_forge.data.uci
===================
Loaders for three imbalanced binary benchmarks from the UCI Machine Learning
Repository, so a result can be checked on more than the credit-card data.

=========================  ======  ========  ===========================  =========
Loader                     Rows    Features  Positive class               Positives
=========================  ======  ========  ===========================  =========
load_taiwanese_bankruptcy  6,819   95        bankrupt                     220 (3.2%)
load_iranian_churn         3,150   13        churned                      495 (15.7%)
load_cervical_cancer_risk  668     30        biopsy positive              45 (6.7%)
=========================  ======  ========  ===========================  =========

All three are published under CC BY 4.0; cite the dataset DOI given in each
loader's docstring.  None is shipped with the package: a loader reads the
original file from ``path`` (default ``$HQNN_FORGE_DATA``, else ``data/raw``,
as for :func:`~hqnn_forge.data.load_credit_card_fraud`), and with
``download=True`` fetches the UCI archive over HTTPS, checks each extracted
file against its published SHA-256 and writes it there.

Features are returned as stored, without scaling.  The PCA in
:class:`~hqnn_forge.preprocessing.PCANormalizer` works on the raw covariance,
so a column on a large scale dominates it: standardise first (bankruptcy
ratios run from 0 to 1e10, churn usage columns into the thousands).

Loading uses NumPy and the standard library only, like the rest of the library.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from hqnn_forge.data.credit_card import (
    DATA_DIR_ENV,
    DEFAULT_DIR,
    DatasetDownloadError,
    DatasetNotFoundError,
)

#: Seconds to wait for the UCI server before a download fails.
DOWNLOAD_TIMEOUT = 60.0


class BinaryDataset(NamedTuple):
    """
    Attributes
    ----------
    X:
        Features, shape ``(n_samples, n_features)``, float64.
    y:
        Labels, shape ``(n_samples,)``, int64, 1 = the positive (rare) class.
    feature_names:
        Column name for each column of ``X``.
    """

    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.int64]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class _Archive:
    """A UCI zip archive and the SHA-256 of the one file used from it."""

    url: str
    file_name: str
    sha256: str
    doi: str


TAIWANESE_BANKRUPTCY = _Archive(
    url="https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip",
    file_name="data.csv",
    sha256="67bf2e7c75490f7ad3f76bbce57d49cdc25967cdab607527b94f944863fa14d8",
    doi="10.24432/C5004D",
)
IRANIAN_CHURN = _Archive(
    url="https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip",
    file_name="Customer Churn.csv",
    sha256="90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1",
    doi="10.24432/C5JW3Z",
)
CERVICAL_CANCER_RISK = _Archive(
    url="https://archive.ics.uci.edu/static/public/383/cervical+cancer+risk+factors.zip",
    file_name="risk_factors_cervical_cancer.csv",
    sha256="8df193ad5c9ff4288fb4c401eef70dcd2cbda404ce7f82ac74c68cfc960ab063",
    doi="10.24432/C5Z310",
)

TAIWANESE_BANKRUPTCY_ROWS, TAIWANESE_BANKRUPTCY_POSITIVES = 6_819, 220
TAIWANESE_BANKRUPTCY_TARGET = "Bankrupt?"
TAIWANESE_BANKRUPTCY_N_FEATURES = 95

IRANIAN_CHURN_ROWS, IRANIAN_CHURN_POSITIVES = 3_150, 495
IRANIAN_CHURN_FEATURES: tuple[str, ...] = (
    "Call Failure",
    "Complains",
    "Subscription Length",
    "Charge Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
    "Status",
    "Age",
    "Customer Value",
)

CERVICAL_CANCER_RISK_ROWS, CERVICAL_CANCER_RISK_POSITIVES = 858, 55
CERVICAL_CANCER_RISK_COLUMNS: tuple[str, ...] = (
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
    "STDs:condylomatosis",
    "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV",
    "STDs: Number of diagnosis",
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
    "Hinselmann",
    "Schiller",
    "Citology",
    "Biopsy",
)
#: Left out of the features: the three other examinations of the same visit
#: (outcomes, not risk factors, so they would leak the target) and the two
#: columns missing for 787 of 858 patients.
CERVICAL_CANCER_RISK_EXCLUDED: frozenset[str] = frozenset(
    {
        "Hinselmann",
        "Schiller",
        "Citology",
        "STDs: Time since first diagnosis",
        "STDs: Time since last diagnosis",
    }
)
CERVICAL_CANCER_RISK_FEATURES: tuple[str, ...] = tuple(
    name for name in CERVICAL_CANCER_RISK_COLUMNS[:-1] if name not in CERVICAL_CANCER_RISK_EXCLUDED
)


# ---------------------------------------------------------------------------
# Locating and downloading the files
# ---------------------------------------------------------------------------


def _resolve_file(path: str | os.PathLike[str] | None, archive: _Archive) -> Path:
    """
    The dataset file for ``path``: the file itself when ``path`` names it,
    else ``archive.file_name`` inside the directory ``path``, or inside
    ``$HQNN_FORGE_DATA`` / ``data/raw`` when ``path`` is None.
    """
    if path is None:
        base = os.environ.get(DATA_DIR_ENV)
        return (Path(base) if base else DEFAULT_DIR) / archive.file_name
    p = Path(path)
    return p if p.name == archive.file_name else p / archive.file_name


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _download(target: Path, archive: _Archive) -> None:
    """Fetch ``archive``, verify its file and write it to ``target``."""
    manual = (
        f"Download {archive.url} manually and extract {archive.file_name!r} into {target.parent}."
    )
    try:
        with urllib.request.urlopen(archive.url, timeout=DOWNLOAD_TIMEOUT) as response:
            payload = response.read()
    except (urllib.error.URLError, OSError) as exc:  # URLError, timeouts, resets
        raise DatasetDownloadError(f"Downloading {archive.url} failed: {exc}.  {manual}") from exc
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            content = zf.read(archive.file_name)
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(f"{archive.url} is not a zip archive.  {manual}") from exc
    except KeyError as exc:
        raise DatasetDownloadError(
            f"{archive.url} has no {archive.file_name!r}; the archive has changed.  {manual}"
        ) from exc
    digest = _sha256(content)
    if digest != archive.sha256:
        raise DatasetDownloadError(
            f"{archive.file_name!r} from {archive.url} has SHA-256 {digest}, expected "
            f"{archive.sha256}: the published file has changed, so the documented row "
            f"and positive counts may no longer hold.  Nothing was written."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name first, so an interrupted write never
    # leaves a truncated file where the next load would find it.
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(content)
    partial.replace(target)


def _locate(
    path: str | os.PathLike[str] | None, archive: _Archive, *, download: bool, strict: bool
) -> Path:
    """Resolve, download if asked, and (with ``strict``) verify the dataset file."""
    file = _resolve_file(path, archive)
    if not file.exists():
        if not download:
            raise DatasetNotFoundError(
                f"{file} not found.  Pass download=True to fetch it from the UCI Machine "
                f"Learning Repository, or download {archive.url} and extract "
                f"{archive.file_name!r} there; path names the file or its directory "
                f"(${DATA_DIR_ENV} names the directory)."
            )
        _download(file, archive)
    if strict:
        digest = _sha256(file.read_bytes())
        if digest != archive.sha256:
            raise ValueError(
                f"{file} has SHA-256 {digest}, expected {archive.sha256}: it is not the "
                f"published file (doi:{archive.doi})."
            )
    return file


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _read_rows(file: Path) -> tuple[list[str], list[list[str]]]:
    """Header and data rows of a comma-separated file, blank lines skipped."""
    with file.open("r", encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.reader(fh) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"{file} is empty.")
    header = [name.strip() for name in rows[0]]
    if len(rows) == 1:
        raise ValueError(f"{file} has a header but no data rows.")
    return header, rows[1:]


def _to_float(
    file: Path, rows: list[list[str]], n_columns: int, missing: str | None = None
) -> npt.NDArray[np.float64]:
    """Rows as a float64 matrix; ``missing`` marks a missing value, read as NaN."""
    data = np.empty((len(rows), n_columns), dtype=np.float64)
    for i, row in enumerate(rows):
        if len(row) != n_columns:
            raise ValueError(
                f"{file}: data row {i + 1} has {len(row)} values, expected {n_columns}."
            )
        for j, cell in enumerate(row):
            cell = cell.strip()
            if missing is not None and cell == missing:
                data[i, j] = math.nan
                continue
            try:
                data[i, j] = float(cell)
            except ValueError:
                raise ValueError(
                    f"{file}: data row {i + 1}, column {j + 1} is not a number: {cell!r}."
                ) from None
    return data


def _labels(file: Path, column: npt.NDArray[np.float64], name: str) -> npt.NDArray[np.int64]:
    ok = np.isin(column, (0.0, 1.0))
    if not ok.all():
        bad = np.unique(column[~ok])[:5]
        raise ValueError(f"{file}: {name} must be 0 or 1; found {bad.tolist()}.")
    return column.astype(np.int64)


def _check_header(file: Path, got: list[str], want: tuple[str, ...], what: str) -> None:
    if tuple(got) == want:
        return
    if len(got) != len(want):
        detail = f"expected {len(want)} columns, found {len(got)}"
    else:
        diffs = [f"{i}: {g!r} != {w!r}" for i, (g, w) in enumerate(zip(got, want)) if g != w]
        detail = (
            "column names differ at " + ", ".join(diffs[:5]) + (" …" if len(diffs) > 5 else "")
        )
    raise ValueError(f"{file} does not look like the {what}: {detail}.")


def _check_counts(file: Path, y: npt.NDArray[np.int64], rows: int, positives: int) -> None:
    if y.size != rows or int(y.sum()) != positives:
        raise ValueError(
            f"{file}: expected {rows} rows with {positives} positives, found {y.size} "
            f"with {int(y.sum())}."
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_taiwanese_bankruptcy(
    path: str | os.PathLike[str] | None = None,
    *,
    download: bool = False,
    strict: bool = False,
) -> BinaryDataset:
    """
    Taiwanese Bankruptcy Prediction: 6,819 companies, 95 financial ratios,
    220 bankrupt (3.2%).

    Collected from the Taiwan Economic Journal for 1999-2009, with bankruptcy
    defined by the business regulations of the Taiwan Stock Exchange.
    UCI dataset 572, CC BY 4.0, doi:10.24432/C5004D.

    The file (``data.csv``) is used unchanged: no missing values, no duplicate
    rows.  Most ratios lie in [0, 1], but some reach 1e10, so scale before
    PCA; ``Net Income Flag`` is constant.

    Parameters
    ----------
    path:
        ``data.csv`` or the directory holding it.  Default:
        ``$HQNN_FORGE_DATA/data.csv`` if that variable is set, else
        ``data/raw/data.csv``.  The generic file name is UCI's; a directory of
        its own avoids a clash with other datasets.
    download:
        If the file is missing, fetch the UCI archive and extract it there.
    strict:
        Also require the file to be byte-identical to the published one
        (SHA-256), which fixes the row and positive counts above.

    Returns
    -------
    BinaryDataset
        ``y`` is ``Bankrupt?``; ``feature_names`` are the file's column names.

    Raises
    ------
    DatasetNotFoundError
        The file is missing and ``download`` is False.
    DatasetDownloadError
        ``download`` is True and the archive could not be fetched, has no
        ``data.csv``, or its ``data.csv`` is not the published file.
    ValueError
        The file does not have the expected columns, has no data rows, has a
        non-numeric value or a label other than 0/1, or (with ``strict``) is
        not the published file.
    """
    archive = TAIWANESE_BANKRUPTCY
    file = _locate(path, archive, download=download, strict=strict)
    header, rows = _read_rows(file)
    n_columns = TAIWANESE_BANKRUPTCY_N_FEATURES + 1
    if len(header) != n_columns or header[0] != TAIWANESE_BANKRUPTCY_TARGET:
        raise ValueError(
            f"{file} does not look like the Taiwanese Bankruptcy Prediction data.csv: "
            f"expected {n_columns} columns starting with {TAIWANESE_BANKRUPTCY_TARGET!r}, "
            f"found {len(header)} starting with {header[0]!r}."
        )
    data = _to_float(file, rows, n_columns)
    y = _labels(file, data[:, 0], TAIWANESE_BANKRUPTCY_TARGET)
    if strict:
        _check_counts(file, y, TAIWANESE_BANKRUPTCY_ROWS, TAIWANESE_BANKRUPTCY_POSITIVES)
    return BinaryDataset(np.ascontiguousarray(data[:, 1:]), y, tuple(header[1:]))


def load_iranian_churn(
    path: str | os.PathLike[str] | None = None,
    *,
    download: bool = False,
    drop_duplicates: bool = False,
    strict: bool = False,
) -> BinaryDataset:
    """
    Iranian Churn: 3,150 telecom customers, 13 features, 495 churned (15.7%).

    Customers drawn at random from an Iranian telecom company's database
    over 12 months; the label says whether the customer churned.
    UCI dataset 563, CC BY 4.0, doi:10.24432/C5JW3Z.

    The file holds 300 rows that repeat an earlier row exactly, so the same
    customer record can land in both the training and the test fold.
    ``drop_duplicates=True`` keeps the first of each, leaving 2,850 rows with
    446 churned (15.6%).  ``Age Group`` is a binned copy of ``Age``, and
    ``Complains``, ``Tariff Plan`` and ``Status`` are codes stored as numbers.

    Parameters
    ----------
    path:
        ``Customer Churn.csv`` or the directory holding it; the default is as
        for :func:`load_taiwanese_bankruptcy`.
    download:
        If the file is missing, fetch the UCI archive and extract it there.
    drop_duplicates:
        Drop rows identical to an earlier one, features and label alike.
    strict:
        Also require the published file (SHA-256).

    Returns
    -------
    BinaryDataset
        ``y`` is ``Churn``.  ``feature_names`` are the file's column names with
        runs of spaces collapsed (``"Call  Failure"`` → ``"Call Failure"``).

    Raises
    ------
    DatasetNotFoundError, DatasetDownloadError, ValueError
        As for :func:`load_taiwanese_bankruptcy`.
    """
    archive = IRANIAN_CHURN
    file = _locate(path, archive, download=download, strict=strict)
    header, rows = _read_rows(file)
    names = [" ".join(name.split()) for name in header]
    _check_header(file, names, (*IRANIAN_CHURN_FEATURES, "Churn"), "Iranian Churn file")
    data = _to_float(file, rows, len(names))
    y = _labels(file, data[:, -1], "Churn")
    if strict:
        _check_counts(file, y, IRANIAN_CHURN_ROWS, IRANIAN_CHURN_POSITIVES)
    if drop_duplicates:
        _, first = np.unique(data, axis=0, return_index=True)
        keep = np.sort(first)
        data, y = data[keep], y[keep]
    return BinaryDataset(np.ascontiguousarray(data[:, :-1]), y, IRANIAN_CHURN_FEATURES)


def load_cervical_cancer_risk(
    path: str | os.PathLike[str] | None = None,
    *,
    download: bool = False,
    missing: Literal["drop", "keep"] = "drop",
    strict: bool = False,
) -> BinaryDataset:
    """
    Cervical Cancer (Risk Factors): 30 risk factors, biopsy result as label;
    668 complete records with 45 positives (6.7%).

    Demographics, habits and medical history of 858 patients, collected at
    the Hospital Universitario de Caracas, Venezuela; patients could decline
    to answer questions, hence the missing values.  UCI dataset 383, CC BY
    4.0, doi:10.24432/C5Z310.

    Preprocessing, always applied: ``Biopsy`` is the label.  Left out of the
    features are ``Hinselmann``, ``Schiller`` and ``Citology``, the outcomes of
    the other examinations, which would leak the diagnosis, and the two
    ``STDs: Time since ... diagnosis`` columns, missing for 787 of 858
    patients.  That leaves 30 features.  ``STDs:cervical condylomatosis`` and
    ``STDs:AIDS`` are 0 for every complete record.  The file also repeats 23
    of its rows exactly.

    Parameters
    ----------
    path:
        ``risk_factors_cervical_cancer.csv`` or the directory holding it; the
        default is as for :func:`load_taiwanese_bankruptcy`.
    download:
        If the file is missing, fetch the UCI archive and extract it there.
    missing:
        The file marks a missing answer with ``?``.  ``"drop"`` (default)
        drops every patient with a missing value among the 30 features:
        668 records, 45 positives.  ``"keep"`` keeps all 858 patients (55
        positives) with NaN in ``X``, for a caller who imputes; the rest of
        the library does not accept NaN.
    strict:
        Also require the published file (SHA-256).

    Returns
    -------
    BinaryDataset

    Raises
    ------
    DatasetNotFoundError, DatasetDownloadError, ValueError
        As for :func:`load_taiwanese_bankruptcy`; also ``ValueError`` for a
        ``missing`` other than ``"drop"`` or ``"keep"``.
    """
    if missing not in ("drop", "keep"):
        raise ValueError(f"missing must be 'drop' or 'keep'; got {missing!r}.")
    archive = CERVICAL_CANCER_RISK
    file = _locate(path, archive, download=download, strict=strict)
    header, rows = _read_rows(file)
    _check_header(
        file, header, CERVICAL_CANCER_RISK_COLUMNS, "Cervical Cancer (Risk Factors) file"
    )
    data = _to_float(file, rows, len(header), missing="?")
    target = data[:, -1]
    if np.isnan(target).any():
        raise ValueError(f"{file}: Biopsy is missing for some patients.")
    y = _labels(file, target, "Biopsy")
    if strict:
        _check_counts(file, y, CERVICAL_CANCER_RISK_ROWS, CERVICAL_CANCER_RISK_POSITIVES)
    columns = [header.index(name) for name in CERVICAL_CANCER_RISK_FEATURES]
    X = data[:, columns]
    if missing == "drop":
        complete = ~np.isnan(X).any(axis=1)
        X, y = X[complete], y[complete]
    return BinaryDataset(np.ascontiguousarray(X), y, CERVICAL_CANCER_RISK_FEATURES)
