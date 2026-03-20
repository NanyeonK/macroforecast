"""Shared CSV download and parsing logic for FRED-MD and FRED-QD.

FRED-MD and FRED-QD CSV files share a specific layout:
- Row 0: variable name header (sasdate, RPI, W875RX1, ...)
- Row 1: transformation codes row (first cell is "Transform:", remaining are codes)
- Row 2 onward: data with the first column being dates in M/D/YYYY format

This module provides the low-level functions that the dataset-specific
loaders (fred_md.py, fred_qd.py) call internally.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from macrocast.utils.cache import download_file

_MD_BASE = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/monthly"
)
_QD_BASE = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/quarterly"
)


# Known vintage filename patterns, tried in order.
# If FRED changes the URL format again, just add a new entry here.
# The downloader tries each candidate and uses the first one that returns HTTP 200.
def _vintage_url_candidates(dataset: str, vintage: str) -> list[str]:
    """Return all plausible URLs for a vintage, from newest to oldest format.

    Parameters
    ----------
    dataset : str
        ``"fred_md"`` or ``"fred_qd"``.
    vintage : str
        Vintage identifier in ``"YYYY-MM"`` format.

    Returns
    -------
    list of str
        Candidate URLs ordered from most-recent naming convention to oldest.
    """
    suffix = "md" if dataset == "fred_md" else "qd"
    base = _MD_BASE if dataset == "fred_md" else _QD_BASE
    return [
        f"{base}/{vintage}-{suffix}.csv",  # 2025-04+: YYYY-MM-md.csv
        f"{base}/{vintage}.csv",  # pre-2025-04: YYYY-MM.csv
    ]


def _build_vintage_url(dataset: str, vintage: str, timeout: int = 30) -> str:
    """Resolve the actual download URL for a vintage by probing candidates.

    Tries each URL from ``_vintage_url_candidates`` with an HTTP HEAD request
    and returns the first one that responds with HTTP 200. This approach
    requires no hardcoded cutoff date and will continue to work if FRED
    changes the filename convention again.

    Parameters
    ----------
    dataset : str
        ``"fred_md"`` or ``"fred_qd"``.
    vintage : str
        Vintage identifier in ``"YYYY-MM"`` format.
    timeout : int
        Per-request timeout in seconds.

    Returns
    -------
    str
        The first URL that returns HTTP 200.

    Raises
    ------
    ValueError
        If no candidate URL is reachable.
    """
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; macrocast/0.1; "
            "+https://github.com/macrocast/macrocast)"
        )
    }
    candidates = _vintage_url_candidates(dataset, vintage)
    for url in candidates:
        try:
            resp = requests.head(
                url, headers=headers, timeout=timeout, allow_redirects=True
            )
            if resp.status_code == 200:
                return url
        except requests.RequestException:
            continue

    raise ValueError(
        f"Could not locate vintage '{vintage}' for {dataset}. Tried: {candidates}"
    )


def _build_url(base_template: str, vintage: str | None) -> str:
    """Construct the download URL for a given vintage or current release.

    Parameters
    ----------
    base_template : str
        URL template with an optional ``{vintage}`` placeholder, e.g.
        ``"https://example.com/{vintage}.csv"``.
    vintage : str or None
        Vintage identifier in ``"YYYY-MM"`` format. When None the
        template is treated as a literal URL to the current release.

    Returns
    -------
    str
        Fully resolved URL string.
    """
    if vintage is None:
        return re.sub(r"\{vintage\}", "current", base_template)
    return base_template.replace("{vintage}", vintage)


def _download_fred_csv(
    url: str,
    cache_path: Path,
    force_download: bool = False,
    timeout: int = 60,
) -> Path:
    """Download a FRED CSV if not already cached (or if forced).

    Parameters
    ----------
    url : str
        Remote URL.
    cache_path : Path
        Local destination path. The parent directory must already exist.
    force_download : bool
        If True, always download even if a cached copy exists.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    Path
        Path to the local file (either freshly downloaded or cached).
    """
    if not force_download and cache_path.exists():
        return cache_path
    return download_file(url, cache_path, timeout=timeout)


def _parse_fred_csv(filepath: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse a FRED-MD or FRED-QD CSV file.

    The expected layout is:
    - Row 0: transformation codes (numeric), first cell is empty or
      ``"sasdate"`` / ``"sasqdate"``.
    - Rows 1 onward: data with the first column as dates in M/D/YYYY
      (or MM/DD/YYYY) format.

    Missing value representations ``""`` and ``"."`` are converted to
    NaN.

    Parameters
    ----------
    filepath : Path
        Path to the local CSV file.

    Returns
    -------
    data : pd.DataFrame
        Panel with a monthly/quarterly DatetimeIndex and variable columns.
    tcodes : dict[str, int]
        Mapping from variable name to transformation code.

    Raises
    ------
    ValueError
        If the file does not appear to be a valid FRED-MD/QD CSV.
    """
    raw = pd.read_csv(
        filepath,
        header=None,
        na_values=["", ".", " "],
        dtype=str,
        low_memory=False,
    )

    if raw.empty:
        raise ValueError(f"CSV file appears to be empty: {filepath}")

    # Detect layout: actual FRED-MD format has variable names in row 0
    # and transformation codes in row 1 (first cell = "Transform:" or "sasdate").
    # Fixture/legacy format may have tcode row first.
    first_cell = str(raw.iloc[0, 0]).strip().lower()
    second_cell = str(raw.iloc[1, 0]).strip().lower()

    if first_cell in ("sasdate", "sasqdate") or (
        first_cell not in ("transform:", "") and second_cell == "transform:"
    ):
        # Official FRED format: row 0 = header, row 1 = "Transform:,5,5,..."
        header_row = raw.iloc[0]
        tcode_row = raw.iloc[1]
        data_start = 2
    else:
        # Legacy/fixture format: row 0 = tcodes, row 1 = header
        tcode_row = raw.iloc[0]
        header_row = raw.iloc[1]
        data_start = 2

    # Variable names (skip first column = date column)
    var_names = [str(v).strip() for v in header_row.iloc[1:]]

    # Build tcodes dict; coerce to int, default 1 for unparseable entries
    tcodes: dict[str, int] = {}
    for i, name in enumerate(var_names):
        raw_tc = tcode_row.iloc[i + 1]
        try:
            tcodes[name] = int(float(str(raw_tc)))
        except (ValueError, TypeError):
            tcodes[name] = 1

    # Data rows
    data_raw = raw.iloc[data_start:].copy()
    data_raw = data_raw.reset_index(drop=True)

    # Parse date column
    date_series = data_raw.iloc[:, 0].astype(str).str.strip()
    dates = _parse_fred_dates(date_series)

    # Build numeric DataFrame
    df = data_raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    df.index = pd.DatetimeIndex(dates)
    df.index.name = "date"
    df.columns = var_names

    # Drop rows where the date could not be parsed (NaT)
    valid_idx = df.index.notna()
    df = df[valid_idx]

    return df, tcodes


def _parse_fred_dates(date_series: pd.Series) -> list[pd.Timestamp | None]:
    """Parse FRED date strings into pandas Timestamps.

    Handles formats:
    - ``M/1/YYYY`` (FRED-MD monthly: day is always 1)
    - ``MM/DD/YYYY``
    - ``YYYY-MM-DD``
    - ``YYYY:QN`` quarterly notation (converted to first month of quarter)

    Parameters
    ----------
    date_series : pd.Series
        Raw string date column.

    Returns
    -------
    list of pd.Timestamp or None
        Parsed dates; None/NaT for unparseable entries.
    """
    results: list[pd.Timestamp | None] = []
    for raw in date_series:
        raw = str(raw).strip()
        if not raw or raw.lower() in ("nan", "none", ""):
            results.append(None)
            continue
        # Try pandas general parser first (handles most formats)
        try:
            results.append(pd.Timestamp(raw))
            continue
        except (ValueError, TypeError):
            pass
        # Quarterly notation YYYY:QN
        qm = re.match(r"(\d{4}):Q(\d)", raw)
        if qm:
            year = int(qm.group(1))
            quarter = int(qm.group(2))
            month = (quarter - 1) * 3 + 1
            results.append(pd.Timestamp(year=year, month=month, day=1))
            continue
        results.append(None)
    return results
