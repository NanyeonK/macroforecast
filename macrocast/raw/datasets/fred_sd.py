from __future__ import annotations

from html import unescape
from io import BytesIO
from pathlib import Path
import re
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import zipfile

import pandas as pd

from ..cache import atomic_copy_to_cache, atomic_write_bytes_to_cache, get_raw_file_path
from ..errors import RawDownloadError, RawParseError
from ..manager import build_raw_artifact_record, normalize_version_request
from ..manifest import append_raw_manifest_entry
from ..types import RawDatasetMetadata, RawLoadResult

_LANDING_PAGE_URL = "https://www.stlouisfed.org/research/economists/owyang/fred-sd"
_SOURCE_BASE_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd"
_SERIES_XLSX_URL_TEMPLATE = f"{_SOURCE_BASE_URL}/series/series-{{vintage}}.xlsx"
_SERIES_YEAR_ZIP_URL_TEMPLATE = f"{_SOURCE_BASE_URL}/series/fredsd_byseries_{{year}}.zip"
_SERIES_RANGE_ZIP_URL_TEMPLATE = f"{_SOURCE_BASE_URL}/series/fredsd_byseries_{{start_year}}_{{end_year}}.zip"
_SERIES_XLSX_LINK_RE = re.compile(r'href=["\']([^"\']*?/series/series-(\d{4}-\d{2})\.xlsx[^"\']*)["\']', re.I)
_FRED_SD_HEADERS = {"User-Agent": "macrocast FRED-SD loader (https://github.com/NanyeonK/macrocast)"}


def _local_source_format(local_source: str | Path | None) -> str:
    if local_source is None:
        return "xlsx"
    suffix = Path(local_source).suffix.lower().lstrip(".")
    return "csv" if suffix == "csv" else "xlsx"


def _request_url(url: str):
    return urlopen(Request(url, headers=_FRED_SD_HEADERS))


def _is_remote_xlsx(payload: bytes, content_type: str | None) -> bool:
    if not payload.startswith(b"PK"):
        return False
    if content_type is None:
        return True
    normalized = content_type.lower()
    return "spreadsheetml.sheet" in normalized or "application/octet-stream" in normalized


def _read_remote_xlsx(url: str) -> tuple[str, bytes] | None:
    with _request_url(url) as src:
        payload = src.read()
        content_type = src.headers.get("Content-Type")
        if _is_remote_xlsx(payload, content_type):
            return src.geturl(), payload
    return None


def _latest_series_url_from_html(html: str) -> str:
    candidates: list[tuple[str, str]] = []
    for href, vintage in _SERIES_XLSX_LINK_RE.findall(html):
        candidates.append((vintage, urljoin(_LANDING_PAGE_URL, unescape(href))))
    if not candidates:
        raise RawDownloadError("failed to find a FRED-SD by-series workbook link on the official landing page")
    return max(candidates, key=lambda item: item[0])[1]


def _latest_series_url() -> str:
    with _request_url(_LANDING_PAGE_URL) as src:
        html = src.read().decode("utf-8", errors="ignore")
    return _latest_series_url_from_html(html)


def _series_xlsx_url(vintage: str) -> str:
    return _SERIES_XLSX_URL_TEMPLATE.format(vintage=vintage)


def _series_zip_url(vintage: str) -> str:
    year = int(vintage[:4])
    if year >= 2023:
        return _SERIES_YEAR_ZIP_URL_TEMPLATE.format(year=year)
    start_year = year if year % 2 else year - 1
    end_year = start_year + 1
    return _SERIES_RANGE_ZIP_URL_TEMPLATE.format(start_year=start_year, end_year=end_year)


def _extract_vintage_xlsx_from_zip(payload: bytes, vintage: str) -> tuple[str, bytes]:
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        candidates = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".xlsx") and vintage in name and "__macosx" not in name.lower()
        ]
        if not candidates:
            raise RawDownloadError(f"FRED-SD by-series zip does not contain vintage={vintage!r}")
        entry = sorted(candidates, key=lambda name: ("series" not in name.lower(), len(name), name))[0]
        return entry, zf.read(entry)


def _read_vintage_series_xlsx(vintage: str) -> tuple[str, bytes]:
    direct_url = _series_xlsx_url(vintage)
    direct = _read_remote_xlsx(direct_url)
    if direct is not None:
        return direct

    zip_url = _series_zip_url(vintage)
    with _request_url(zip_url) as src:
        payload = src.read()
        content_type = (src.headers.get("Content-Type") or "").lower()
        if not payload.startswith(b"PK") or "html" in content_type:
            raise RawDownloadError(f"failed to obtain FRED-SD by-series zip for vintage={vintage!r}")
        entry, workbook = _extract_vintage_xlsx_from_zip(payload, vintage)
        return f"{src.geturl()}#{entry}", workbook


def _read_current_series_xlsx() -> tuple[str, bytes]:
    current_url = _latest_series_url()
    current = _read_remote_xlsx(current_url)
    if current is None:
        raise RawDownloadError(f"latest FRED-SD by-series link did not return an Excel workbook: {current_url}")
    return current


def _filter_wide_csv_columns(
    df: pd.DataFrame,
    *,
    states: list[str] | None,
    variables: list[str] | None,
) -> pd.DataFrame:
    if states is None and variables is None:
        return df
    state_set = set(states or [])
    variable_set = set(variables or [])
    keep: list[str] = []
    for column in df.columns:
        name = str(column)
        if "_" in name:
            variable, state = name.rsplit("_", 1)
        else:
            variable, state = name, ""
        if variable_set and variable not in variable_set:
            continue
        if state_set and state not in state_set:
            continue
        keep.append(column)
    return df[keep].copy()


def _read_local_fred_sd_csv(
    path: Path,
    *,
    states: list[str] | None,
    variables: list[str] | None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].set_index(date_col)
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce")
    df = _filter_wide_csv_columns(df, states=states, variables=variables)
    if df.empty:
        raise RawParseError("no matching columns found in local FRED-SD CSV")
    df.sort_index(inplace=True)
    return df


def load_fred_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_source: str | Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
) -> RawLoadResult:
    request = normalize_version_request("fred_sd", vintage=vintage)
    local_format = _local_source_format(local_source)
    target = get_raw_file_path(request, cache_root, suffix=local_format)

    cache_hit = target.exists() and not force and local_source is None
    source_url = _LANDING_PAGE_URL if request.mode == "current" else _series_xlsx_url(str(request.vintage))

    if not cache_hit:
        try:
            if local_source is not None:
                atomic_copy_to_cache(Path(local_source), target)
                source_url = str(local_source)
            else:
                if request.mode == "current":
                    source_url, payload = _read_current_series_xlsx()
                else:
                    source_url, payload = _read_vintage_series_xlsx(str(request.vintage))
                atomic_write_bytes_to_cache(payload, target)
        except Exception as exc:
            raise RawDownloadError(f"failed to obtain FRED-SD raw file for request={request}") from exc

    try:
        if local_format == "csv":
            df = _read_local_fred_sd_csv(target, states=states, variables=variables)
        else:
            sheet_name = None if variables is None else list(variables)
            loaded_sheets = pd.read_excel(target, sheet_name=sheet_name, index_col=0, engine="openpyxl")
            if isinstance(loaded_sheets, pd.DataFrame):
                sheets = {str(sheet_name[0]): loaded_sheets} if sheet_name else {}
            else:
                sheets = loaded_sheets
            if not sheets:
                raise RawParseError("no matching sheets found in FRED-SD workbook")

            wide_frames: list[pd.DataFrame] = []
            for var_name, sheet_df in sheets.items():
                if not isinstance(sheet_df.index, pd.DatetimeIndex):
                    sheet_df.index = pd.to_datetime(sheet_df.index, errors="coerce")
                    sheet_df = sheet_df[sheet_df.index.notna()]
                selected_states = states if states is not None else list(sheet_df.columns)
                available = [state for state in selected_states if state in sheet_df.columns]
                sub = sheet_df[available].copy().apply(pd.to_numeric, errors="coerce")
                sub.columns = [f"{var_name}_{state}" for state in available]
                wide_frames.append(sub)

            df = pd.concat(wide_frames, axis=1)
            df.index.name = "date"
            df.sort_index(inplace=True)
    except Exception as exc:
        source_kind = "CSV" if local_format == "csv" else "workbook"
        raise RawParseError(f"failed to parse FRED-SD {source_kind} at {target}") from exc

    artifact = build_raw_artifact_record(
        request=request,
        source_url=source_url,
        local_path=target,
        file_format=local_format,
        cache_hit=cache_hit,
    )
    metadata = RawDatasetMetadata(
        dataset="fred_sd",
        source_family="fred-sd",
        frequency="state_monthly",
        version_mode=request.mode,
        vintage=request.vintage,
        data_through=df.index[-1].strftime("%Y-%m") if len(df) else None,
        support_tier="provisional",
    )
    result = RawLoadResult(data=df, dataset_metadata=metadata, artifact=artifact)
    append_raw_manifest_entry(result, cache_root=cache_root)
    return result
