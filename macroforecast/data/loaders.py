from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Literal, cast
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import warnings
import zipfile

import pandas as pd

from .errors import RawDownloadError, RawManifestError, RawParseError, RawVersionFormatError
from .panel import DataBundle, as_panel, set_frequencies

DatasetId = Literal["fred_md", "fred_qd", "fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"]
VersionMode = Literal["current", "vintage"]

_CURRENT_MD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
_VINTAGE_MD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/{vintage}.csv"
_CURRENT_QD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv"
_VINTAGE_QD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/{vintage}.csv"

_FRED_SD_LANDING_PAGE_URL = "https://www.stlouisfed.org/research/economists/owyang/fred-sd"
_FRED_SD_SOURCE_BASE_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd"
_FRED_SD_SERIES_XLSX_URL_TEMPLATE = f"{_FRED_SD_SOURCE_BASE_URL}/series/series-{{vintage}}.xlsx"
_FRED_SD_SERIES_YEAR_ZIP_URL_TEMPLATE = f"{_FRED_SD_SOURCE_BASE_URL}/series/fredsd_byseries_{{year}}.zip"
_FRED_SD_SERIES_RANGE_ZIP_URL_TEMPLATE = f"{_FRED_SD_SOURCE_BASE_URL}/series/fredsd_byseries_{{start_year}}_{{end_year}}.zip"
_FRED_SD_SERIES_XLSX_LINK_RE = re.compile(r'href=["\']([^"\']*?/series/series-(\d{4}-\d{2})\.xlsx[^"\']*)["\']', re.I)
_FRED_SD_HEADERS = {"User-Agent": "macroforecast FRED-SD loader (https://github.com/NanyeonK/macroforecast)"}
_FRED_SD_SERIES_METADATA_CONTRACT_VERSION = "fred_sd_series_metadata_v1"

_FIRST_VINTAGE: dict[DatasetId, str] = {
    "fred_md": "1999-01",
    "fred_qd": "2005-01",
    "fred_sd": "2005-06",
    "fred_md+fred_sd": "2005-06",
    "fred_qd+fred_sd": "2005-06",
}


@dataclass(frozen=True)
class _VersionRequest:
    dataset: DatasetId
    mode: VersionMode
    vintage: str | None


FRED_SD_GROUP_SELECTION_CONTRACT_VERSION = "fred_sd_group_selection_v1"

FRED_SD_STATE_GROUPS: dict[str, tuple[str, ...]] = {
    "census_region_northeast": ("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"),
    "census_region_midwest": ("IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    "census_region_south": (
        "DE",
        "DC",
        "FL",
        "GA",
        "MD",
        "NC",
        "SC",
        "VA",
        "WV",
        "AL",
        "KY",
        "MS",
        "TN",
        "AR",
        "LA",
        "OK",
        "TX",
    ),
    "census_region_west": ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"),
    "census_division_new_england": ("CT", "ME", "MA", "NH", "RI", "VT"),
    "census_division_middle_atlantic": ("NJ", "NY", "PA"),
    "census_division_east_north_central": ("IL", "IN", "MI", "OH", "WI"),
    "census_division_west_north_central": ("IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    "census_division_south_atlantic": ("DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"),
    "census_division_east_south_central": ("AL", "KY", "MS", "TN"),
    "census_division_west_south_central": ("AR", "LA", "OK", "TX"),
    "census_division_mountain": ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"),
    "census_division_pacific": ("AK", "CA", "HI", "OR", "WA"),
    "contiguous_48_plus_dc": (
        "AL",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "DC",
        "FL",
        "GA",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    ),
}

FRED_SD_VARIABLE_GROUPS: dict[str, tuple[str, ...]] = {
    "labor_market_core": ("ICLAIMS", "LF", "NA", "PARTRATE", "UR"),
    "employment_sector": ("CONS", "FIRE", "GOVT", "INFO", "MFG", "MFGHRS", "MINNG", "PSERV"),
    "gsp_output": (
        "NQGSP",
        "CONSTNQGSP",
        "FIRENQGSP",
        "GOVNQGSP",
        "INFONQGSP",
        "MANNQGSP",
        "NATURNQGSP",
        "PSERVNQGSP",
        "UTILNQGSP",
    ),
    "housing": ("BPPRIVSA", "RENTS", "STHPI"),
    "trade": ("EXPORTS", "IMPORTS"),
    "income": ("OTOT",),
    "direct_analog_high_confidence": (
        "CONS",
        "FIRE",
        "GOVT",
        "ICLAIMS",
        "INFO",
        "LF",
        "MFG",
        "MFGHRS",
        "MINNG",
        "NA",
        "PARTRATE",
        "PSERV",
        "UR",
    ),
    "provisional_analog_medium": ("EXPORTS", "IMPORTS", "NQGSP", "OTOT"),
    "semantic_review_outputs": ("CONSTNQGSP", "GOVNQGSP", "MANNQGSP", "PSERVNQGSP"),
    "no_reliable_analog": ("FIRENQGSP", "INFONQGSP", "NATURNQGSP", "RENTS", "UTILNQGSP"),
}


def _normalize_member_list(values: Sequence[Any], *, uppercase: bool) -> list[str]:
    members = [str(value).strip() for value in values]
    if uppercase:
        members = [value.upper() for value in members]
    if not members or any(not value for value in members):
        raise ValueError("custom FRED-SD group members must be a non-empty list of non-empty strings")
    return members


def _resolve_custom_group(
    leaf_config: Mapping[str, Any] | None,
    *,
    members_key: str,
    mapping_key: str,
    name_key: str,
    label: str,
    uppercase: bool = False,
) -> tuple[list[str], str]:
    payload = leaf_config or {}
    direct = payload.get(members_key)
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return _normalize_member_list(direct, uppercase=uppercase), members_key

    groups = payload.get(mapping_key)
    name = payload.get(name_key)
    if isinstance(groups, Mapping) and name in groups:
        members = groups[name]
        if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
            return _normalize_member_list(members, uppercase=uppercase), f"{mapping_key}.{name}"

    raise ValueError(
        f"{label} requires leaf_config.{members_key} or "
        f"leaf_config.{mapping_key} plus leaf_config.{name_key}"
    )


def resolve_fred_sd_state_group(
    group: str | None,
    leaf_config: Mapping[str, Any] | None = None,
) -> tuple[list[str] | None, str]:
    """Resolve a FRED-SD state group to state abbreviations.

    Returns ``(None, "all_states")`` for the all-state default.
    """

    key = str(group or "all_states")
    if key == "all_states":
        return None, "all_states"
    if key == "custom_state_group":
        return _resolve_custom_group(
            leaf_config,
            members_key="sd_state_group_members",
            mapping_key="sd_state_groups",
            name_key="sd_state_group_name",
            label="fred_sd_state_group='custom_state_group'",
            uppercase=True,
        )
    if key not in FRED_SD_STATE_GROUPS:
        allowed = sorted(("all_states", "custom_state_group", *FRED_SD_STATE_GROUPS))
        raise ValueError(f"unsupported fred_sd_state_group={key!r}; allowed values: {allowed}")
    return list(FRED_SD_STATE_GROUPS[key]), key


def resolve_fred_sd_variable_group(
    group: str | None,
    leaf_config: Mapping[str, Any] | None = None,
) -> tuple[list[str] | None, str]:
    """Resolve a FRED-SD workbook-variable group to sheet names."""

    key = str(group or "all_sd_variables")
    if key == "all_sd_variables":
        return None, "all_sd_variables"
    if key == "custom_sd_variable_group":
        return _resolve_custom_group(
            leaf_config,
            members_key="sd_variable_group_members",
            mapping_key="sd_variable_groups",
            name_key="sd_variable_group_name",
            label="fred_sd_variable_group='custom_sd_variable_group'",
        )
    if key not in FRED_SD_VARIABLE_GROUPS:
        allowed = sorted(("all_sd_variables", "custom_sd_variable_group", *FRED_SD_VARIABLE_GROUPS))
        raise ValueError(f"unsupported fred_sd_variable_group={key!r}; allowed values: {allowed}")
    return list(FRED_SD_VARIABLE_GROUPS[key]), key


def list_vintages(dataset: str, start: str | None = None, end: str | None = None) -> list[str]:
    """Return monthly vintage labels between ``start`` and ``end`` inclusive."""

    dataset_id = _dataset_id(dataset)
    start_value = start or _FIRST_VINTAGE[dataset_id]
    _validate_vintage(start_value)
    if end is None:
        raise RawVersionFormatError("end must be supplied")
    _validate_vintage(end)
    start_year, start_month = map(int, start_value.split("-"))
    end_year, end_month = map(int, end.split("-"))
    vintages: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        vintages.append(f"{year:04d}-{month:02d}")
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return vintages


def load_fred_md(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_source: str | Path | None = None,
    local_zip_source: str | Path | None = None,
) -> DataBundle:
    request = _version_request("fred_md", vintage=vintage)
    target = _raw_file_path(request, cache_root, suffix="csv")
    cache_hit = target.exists() and not force and local_source is None and local_zip_source is None
    source_url = _CURRENT_MD_URL if request.mode == "current" else _VINTAGE_MD_URL.format(vintage=request.vintage)

    if not cache_hit:
        try:
            if local_source is not None:
                _atomic_copy(Path(local_source), target)
                source_url = str(local_source)
            elif local_zip_source is not None:
                if request.vintage is None:
                    raise RawDownloadError("local_zip_source requires an explicit vintage")
                _extract_md_vintage_from_zip(Path(local_zip_source), request.vintage, target)
                source_url = str(local_zip_source)
            else:
                with urlopen(source_url) as src:
                    _atomic_write(src.read(), target)
        except Exception as exc:
            raise RawDownloadError(f"failed to obtain FRED-MD raw file for request={request}") from exc

    try:
        panel, tcodes = _parse_fred_csv(target)
    except Exception as exc:
        raise RawParseError(f"failed to parse FRED-MD CSV at {target}") from exc
    return _official_bundle(
        panel,
        dataset="fred_md",
        frequency="monthly",
        request=request,
        source_url=source_url,
        local_path=target,
        file_format="csv",
        cache_hit=cache_hit,
        transform_codes=tcodes,
        cache_root=cache_root,
    )


def load_fred_qd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_source: str | Path | None = None,
) -> DataBundle:
    request = _version_request("fred_qd", vintage=vintage)
    target = _raw_file_path(request, cache_root, suffix="csv")
    cache_hit = target.exists() and not force and local_source is None
    source_url = _CURRENT_QD_URL if request.mode == "current" else _VINTAGE_QD_URL.format(vintage=request.vintage)

    if not cache_hit:
        try:
            if local_source is not None:
                _atomic_copy(Path(local_source), target)
                source_url = str(local_source)
            else:
                with urlopen(source_url) as src:
                    _atomic_write(src.read(), target)
        except Exception as exc:
            raise RawDownloadError(f"failed to obtain FRED-QD raw file for request={request}") from exc

    try:
        panel, tcodes = _parse_fred_csv(target)
    except Exception as exc:
        raise RawParseError(f"failed to parse FRED-QD CSV at {target}") from exc
    return _official_bundle(
        panel,
        dataset="fred_qd",
        frequency="quarterly",
        request=request,
        source_url=source_url,
        local_path=target,
        file_format="csv",
        cache_hit=cache_hit,
        transform_codes=tcodes,
        cache_root=cache_root,
    )


def load_fred_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_source: str | Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
) -> DataBundle:
    request = _version_request("fred_sd", vintage=vintage)
    source_format = _fred_sd_local_source_format(local_source)
    target = _raw_file_path(request, cache_root, suffix=source_format)
    cache_hit = target.exists() and not force and local_source is None
    source_url = _FRED_SD_LANDING_PAGE_URL if request.mode == "current" else _fred_sd_series_xlsx_url(str(request.vintage))

    if not cache_hit:
        try:
            if local_source is not None:
                _atomic_copy(Path(local_source), target)
                source_url = str(local_source)
            else:
                if request.mode == "current":
                    source_url, payload = _fred_sd_read_current_series_xlsx()
                else:
                    source_url, payload = _fred_sd_read_vintage_series_xlsx(str(request.vintage))
                _atomic_write(payload, target)
        except Exception as exc:
            raise RawDownloadError(f"failed to obtain FRED-SD raw file for request={request}") from exc

    try:
        if source_format == "csv":
            panel = _read_local_fred_sd_csv(target, states=states, variables=variables)
        else:
            panel = _read_fred_sd_workbook(target, states=states, variables=variables)
    except Exception as exc:
        source_kind = "CSV" if source_format == "csv" else "workbook"
        raise RawParseError(f"failed to parse FRED-SD {source_kind} at {target}") from exc

    report = _fred_sd_series_metadata(panel, states=states, variables=variables, source_format=source_format)
    reports = dict(panel.attrs.get("macrocast_reports", {}))
    reports["fred_sd_series_metadata"] = report
    panel.attrs["macrocast_reports"] = reports
    native_frequency_by_column = {
        str(row["column"]): str(row["native_frequency"])
        for row in report["series"]
        if isinstance(row, Mapping)
    }
    date_anchor_by_column = {
        str(row["column"]): str(row["date_anchor"])
        for row in report["series"]
        if isinstance(row, Mapping)
    }
    return _official_bundle(
        panel,
        dataset="fred_sd",
        frequency=_fred_sd_panel_frequency(report),
        request=request,
        source_url=source_url,
        local_path=target,
        file_format=source_format,
        cache_hit=cache_hit,
        extra_metadata={
            "native_frequency_by_column": native_frequency_by_column,
            "native_frequency_counts": report["native_frequency_counts"],
            "date_anchor_by_column": date_anchor_by_column,
            "date_anchor_counts": report["date_anchor_counts"],
        },
        cache_root=cache_root,
    )


def load_fred_md_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_fred_md_source: str | Path | None = None,
    local_fred_sd_source: str | Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
    frequency: str = "monthly",
    quarterly_to_monthly: str = "repeat_within_quarter",
    monthly_to_quarterly: str = "quarterly_average",
) -> DataBundle:
    """Load FRED-MD plus FRED-SD as one canonical data bundle."""

    national = load_fred_md(
        vintage=vintage,
        force=force,
        cache_root=cache_root,
        local_source=local_fred_md_source,
    )
    regional = load_fred_sd(
        vintage=vintage,
        force=force,
        cache_root=cache_root,
        local_source=local_fred_sd_source,
        states=states,
        variables=variables,
    )
    return combine(
        national,
        regional,
        dataset="fred_md+fred_sd",
        frequency=frequency,
        quarterly_to_monthly=quarterly_to_monthly,
        monthly_to_quarterly=monthly_to_quarterly,
    )


def load_fred_qd_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_fred_qd_source: str | Path | None = None,
    local_fred_sd_source: str | Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
    frequency: str = "quarterly",
    quarterly_to_monthly: str = "repeat_within_quarter",
    monthly_to_quarterly: str = "quarterly_average",
) -> DataBundle:
    """Load FRED-QD plus FRED-SD as one canonical data bundle."""

    national = load_fred_qd(
        vintage=vintage,
        force=force,
        cache_root=cache_root,
        local_source=local_fred_qd_source,
    )
    regional = load_fred_sd(
        vintage=vintage,
        force=force,
        cache_root=cache_root,
        local_source=local_fred_sd_source,
        states=states,
        variables=variables,
    )
    return combine(
        national,
        regional,
        dataset="fred_qd+fred_sd",
        frequency=frequency,
        quarterly_to_monthly=quarterly_to_monthly,
        monthly_to_quarterly=monthly_to_quarterly,
    )


def load_custom_csv(
    path: str | Path,
    *,
    date: str | None = None,
    date_col: str | int | None = None,
    columns: Iterable[str] | None = None,
    series_columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    dataset: str = "custom",
    frequency: str = "unknown",
    frequency_by_column: Mapping[str, str] | None = None,
    default_frequency: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    transform_codes: Mapping[str, int] | None = None,
    cache_root: str | Path | None = None,
    strict: bool = True,
) -> DataBundle:
    csv_path = Path(path)
    if not csv_path.exists():
        raise RawParseError(f"custom CSV source path does not exist: {csv_path}")
    try:
        raw = pd.read_csv(csv_path)
        resolved_date, resolved_columns = _normalize_custom_loader_columns(
            raw,
            date=date,
            date_col=date_col,
            columns=columns,
            series_columns=series_columns,
        )
        panel = as_panel(raw, date=resolved_date, columns=resolved_columns, rename=rename, strict=strict)
    except Exception as exc:
        raise RawParseError(f"failed to normalize custom CSV at {csv_path}") from exc
    return _custom_bundle(
        panel,
        dataset=dataset,
        source_family="custom-csv",
        frequency=frequency,
        local_path=csv_path,
        file_format="csv",
        metadata=metadata,
        transform_codes=transform_codes,
        frequency_by_column=frequency_by_column,
        default_frequency=default_frequency,
        cache_root=cache_root,
    )


def load_custom_parquet(
    path: str | Path,
    *,
    date: str | None = None,
    date_col: str | int | None = None,
    columns: Iterable[str] | None = None,
    series_columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    dataset: str = "custom",
    frequency: str = "unknown",
    frequency_by_column: Mapping[str, str] | None = None,
    default_frequency: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    transform_codes: Mapping[str, int] | None = None,
    cache_root: str | Path | None = None,
    strict: bool = True,
) -> DataBundle:
    pq_path = Path(path)
    if not pq_path.exists():
        raise RawParseError(f"custom Parquet source path does not exist: {pq_path}")
    try:
        raw = pd.read_parquet(pq_path)
        resolved_date, resolved_columns = _normalize_custom_loader_columns(
            raw,
            date=date,
            date_col=date_col,
            columns=columns,
            series_columns=series_columns,
        )
        panel = as_panel(raw, date=resolved_date, columns=resolved_columns, rename=rename, strict=strict)
    except Exception as exc:
        raise RawParseError(f"failed to normalize custom Parquet at {pq_path}") from exc
    return _custom_bundle(
        panel,
        dataset=dataset,
        source_family="custom-parquet",
        frequency=frequency,
        local_path=pq_path,
        file_format="parquet",
        metadata=metadata,
        transform_codes=transform_codes,
        frequency_by_column=frequency_by_column,
        default_frequency=default_frequency,
        cache_root=cache_root,
    )


def _normalize_custom_loader_columns(
    raw: pd.DataFrame,
    *,
    date: str | None,
    date_col: str | int | None,
    columns: Iterable[str] | None,
    series_columns: Iterable[str] | None,
) -> tuple[str | None, Iterable[str] | None]:
    if date is not None and date_col is not None and date != date_col:
        raise ValueError("provide either date or date_col, not both")
    if columns is not None and series_columns is not None:
        raise ValueError("provide either columns or series_columns, not both")
    resolved_date: str | None
    if isinstance(date_col, int):
        try:
            resolved_date = str(raw.columns[date_col])
        except IndexError as exc:
            raise ValueError(f"date_col index {date_col} is outside the input columns") from exc
    elif date_col is None:
        resolved_date = date
    else:
        resolved_date = str(date_col)
    resolved_columns = columns if columns is not None else series_columns
    return resolved_date, resolved_columns


def _dataset_id(dataset: str) -> DatasetId:
    if dataset not in _FIRST_VINTAGE:
        raise RawVersionFormatError(f"unknown dataset={dataset!r}")
    return cast(DatasetId, dataset)


def _validate_vintage(vintage: str) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}", vintage):
        raise RawVersionFormatError(f"invalid vintage format: {vintage!r}")


def _version_request(dataset: str, vintage: str | None = None) -> _VersionRequest:
    dataset_id = _dataset_id(dataset)
    if vintage is None:
        return _VersionRequest(dataset=dataset_id, mode="current", vintage=None)
    _validate_vintage(vintage)
    return _VersionRequest(dataset=dataset_id, mode="vintage", vintage=vintage)


def _raw_cache_root(cache_root: str | Path | None = None) -> Path:
    root = Path(cache_root).expanduser() if cache_root is not None else Path("~/.macroforecast/raw").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest_path(cache_root: str | Path | None = None) -> Path:
    path = _raw_cache_root(cache_root) / "manifest" / "raw_artifacts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _raw_file_path(request: _VersionRequest, cache_root: str | Path | None = None, *, suffix: str) -> Path:
    root = _raw_cache_root(cache_root)
    if request.mode == "current":
        path = root / request.dataset / "current" / f"raw.{suffix}"
    else:
        if request.vintage is None:
            raise ValueError("vintage mode requires vintage string")
        path = root / request.dataset / "vintages" / f"{request.vintage}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp_path = Path(tmp_name)
    os.close(fd)
    try:
        shutil.copyfile(source, tmp_path)
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_write(content: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


def _file_artifact(*, dataset: str, request: _VersionRequest | None, source_url: str, local_path: Path, file_format: str, cache_hit: bool) -> dict[str, Any]:
    content = local_path.read_bytes()
    return {
        "dataset": dataset,
        "version_mode": request.mode if request is not None else "current",
        "vintage": request.vintage if request is not None else None,
        "source_url": source_url,
        "local_path": str(local_path),
        "file_format": file_format,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "file_size_bytes": len(content),
        "cache_hit": cache_hit,
        "manifest_version": "v1",
    }


def _append_manifest_entry(metadata: Mapping[str, Any], *, cache_root: str | Path | None = None) -> None:
    try:
        entry = {
            **dict(metadata.get("artifact", {})),
            "data_through": metadata.get("data_through"),
            "support_tier": metadata.get("support_tier"),
            "parse_notes": list(metadata.get("parse_notes", ())),
        }
        with _manifest_path(cache_root).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise RawManifestError("failed to append raw manifest entry") from exc


def _official_bundle(
    panel: pd.DataFrame,
    *,
    dataset: str,
    frequency: str,
    request: _VersionRequest,
    source_url: str,
    local_path: Path,
    file_format: str,
    cache_hit: bool,
    source_family: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    transform_codes: Mapping[str, int] | None = None,
    cache_root: str | Path | None = None,
) -> DataBundle:
    metadata = {
        "dataset": dataset,
        "frequency": frequency,
        "version_mode": request.mode,
        "vintage": request.vintage,
        "data_through": _data_through(panel),
        "support_tier": "stable",
        "parse_notes": (),
        "artifact": _file_artifact(
            dataset=dataset,
            request=request,
            source_url=source_url,
            local_path=local_path,
            file_format=file_format,
            cache_hit=cache_hit,
        ),
        "transform_codes": dict(transform_codes or {}),
    }
    if source_family is not None:
        metadata["source_family"] = source_family
    if extra_metadata is not None:
        metadata.update(dict(extra_metadata))
    bundle = _bundle(panel, metadata)
    _append_manifest_entry(metadata, cache_root=cache_root)
    return bundle


def _custom_bundle(
    panel: pd.DataFrame,
    *,
    dataset: str,
    source_family: str,
    frequency: str,
    local_path: Path,
    file_format: str,
    metadata: Mapping[str, Any] | None,
    transform_codes: Mapping[str, int] | None,
    frequency_by_column: Mapping[str, str] | None,
    default_frequency: str | None,
    cache_root: str | Path | None,
) -> DataBundle:
    normalized_tcodes = _normalize_custom_transform_codes(transform_codes, columns=panel.columns)
    source_metadata = {
        "dataset": dataset,
        "source_family": source_family,
        "frequency": frequency,
        "version_mode": "current",
        "vintage": None,
        "data_through": _data_through(panel),
        "support_tier": "provisional",
        "parse_notes": (f"user-supplied {file_format}; no vintage tracking",),
        "artifact": _file_artifact(
            dataset=dataset,
            request=None,
            source_url=str(local_path),
            local_path=local_path,
            file_format=file_format,
            cache_hit=False,
        ),
        "transform_codes": normalized_tcodes,
    }
    merged = {**dict(metadata or {}), **source_metadata}
    bundle = _bundle(panel, merged)
    if frequency_by_column is not None:
        bundle = set_frequencies(
            bundle,
            frequency_by_column,
            default_frequency=default_frequency,
            frequency=frequency,
        )
    if cache_root is not None:
        _append_manifest_entry(bundle.metadata, cache_root=cache_root)
    return bundle


def _normalize_custom_transform_codes(
    transform_codes: Mapping[str, int] | None,
    *,
    columns: pd.Index,
) -> dict[str, int]:
    if transform_codes is None:
        return {}
    column_names = {str(column) for column in columns}
    normalized: dict[str, int] = {}
    for key, raw_code in transform_codes.items():
        name = str(key)
        if name not in column_names:
            raise ValueError(f"transform_codes includes unknown series {name!r}")
        code = int(raw_code)
        if code not in {1, 2, 3, 4, 5, 6, 7}:
            raise ValueError(f"transform code for {name!r} must be in 1..7")
        normalized[name] = code
    return normalized


def combine(
    *bundles: DataBundle,
    dataset: str | None = None,
    frequency: str = "native",
    quarterly_to_monthly: str = "repeat_within_quarter",
    monthly_to_quarterly: str = "quarterly_average",
) -> DataBundle:
    """Combine already-loaded data bundles into one canonical panel."""

    if len(bundles) < 2:
        raise ValueError("combine() requires at least two DataBundle objects")
    target_frequency = _normalize_combined_frequency(frequency)
    aligned_panels: list[pd.DataFrame] = []
    combined_reports: dict[str, Any] = {}
    combined_sources: list[dict[str, Any]] = []
    frequency_conversion_warnings: list[dict[str, Any]] = []
    transform_codes: dict[str, int] = {}
    source_by_column: dict[str, str] = {}
    native_frequency_by_column: dict[str, str] = {}
    date_anchor_by_column: dict[str, str] = {}
    output_frequency_by_column: dict[str, str] = {}
    seen_columns: set[str] = set()
    alignment_reports: list[dict[str, Any]] = []

    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, DataBundle):
            raise TypeError("combine() expects DataBundle arguments")
        source_dataset = str(bundle.metadata.get("dataset") or f"source_{index + 1}")
        panel, alignment = _align_bundle_for_combination(
            bundle,
            frequency=target_frequency,
            quarterly_to_monthly=quarterly_to_monthly,
            monthly_to_quarterly=monthly_to_quarterly,
        )
        for record in alignment.get("frequency_conversions", ()):
            frequency_conversion_warnings.append({"dataset": source_dataset, **dict(record)})
        duplicate_columns = sorted(seen_columns.intersection(str(column) for column in panel.columns))
        if duplicate_columns:
            raise ValueError(f"combined data has duplicate columns: {duplicate_columns[:5]}")
        seen_columns.update(str(column) for column in panel.columns)
        aligned_panels.append(panel)
        alignment_reports.append({"dataset": source_dataset, **alignment})
        combined_sources.append(dict(bundle.metadata))
        source_native_frequencies = dict(alignment.get("native_frequency_by_column", {}))
        source_date_anchors = dict(alignment.get("date_anchor_by_column", {}))
        for column in panel.columns:
            name = str(column)
            native_frequency_by_column[name] = str(source_native_frequencies.get(name, "unknown"))
            if name in source_date_anchors:
                date_anchor_by_column[name] = str(source_date_anchors[name])
            output_frequency_by_column[name] = (
                native_frequency_by_column[name] if target_frequency == "native" else target_frequency
            )
        source_codes = dict(bundle.metadata.get("transform_codes", {}))
        transform_codes.update({str(column): int(code) for column, code in source_codes.items() if str(column) in panel.columns})
        for column in panel.columns:
            source_by_column[str(column)] = source_dataset
        _merge_macrocast_reports(combined_reports, panel.attrs.get("macrocast_reports", {}), source_dataset=source_dataset)

    merged = pd.concat(aligned_panels, axis=1).sort_index()
    output_dataset = dataset or "+".join(str(bundle.metadata.get("dataset") or f"source_{idx + 1}") for idx, bundle in enumerate(bundles))
    parse_notes = _combined_parse_notes(output_dataset, target_frequency)
    metadata = {
        "dataset": output_dataset,
        "source_family": "combined",
        "frequency": target_frequency if target_frequency != "native" else "mixed",
        "version_mode": _combined_version_mode(combined_sources),
        "vintage": _combined_vintage(combined_sources),
        "data_through": _data_through(merged),
        "support_tier": "stable",
        "parse_notes": tuple(parse_notes),
        "artifact": None,
        "transform_codes": transform_codes,
        "combined_sources": combined_sources,
        "source_by_column": source_by_column,
        "native_frequency_by_column": dict(sorted(native_frequency_by_column.items())),
        "native_frequency_counts": dict(sorted(Counter(native_frequency_by_column.values()).items())),
        "output_frequency_by_column": dict(sorted(output_frequency_by_column.items())),
        "output_frequency_counts": dict(sorted(Counter(output_frequency_by_column.values()).items())),
        "frequency_conversion_warnings": frequency_conversion_warnings,
        "alignment": {
            "frequency": target_frequency,
            "quarterly_to_monthly": quarterly_to_monthly,
            "monthly_to_quarterly": monthly_to_quarterly,
            "sources": alignment_reports,
        },
    }
    if date_anchor_by_column:
        metadata["date_anchor_by_column"] = dict(sorted(date_anchor_by_column.items()))
        metadata["date_anchor_counts"] = dict(sorted(Counter(date_anchor_by_column.values()).items()))
    if combined_reports:
        merged.attrs["macrocast_reports"] = combined_reports
    _emit_frequency_conversion_warnings(frequency_conversion_warnings)
    return _bundle(merged, metadata)


def _bundle(panel: pd.DataFrame, metadata: Mapping[str, Any]) -> DataBundle:
    source_panel_report = getattr(panel, "attrs", {}).get("macroforecast_panel_report")
    frame = as_panel(panel, metadata=metadata)
    normalized_metadata = dict(frame.attrs.get("macroforecast_metadata", metadata))
    if isinstance(source_panel_report, Mapping):
        # Preserve the first normalization report from custom loaders. Re-running
        # as_panel() on an already canonical panel would otherwise replace a
        # lossy raw-file report, such as dropped invalid dates, with a clean
        # second-pass report.
        report = dict(source_panel_report)
        frame.attrs["macroforecast_panel_report"] = report
        normalized_metadata["panel"] = report
    if normalized_metadata.get("transform_codes"):
        frame.attrs["macroforecast_transform_codes"] = dict(normalized_metadata["transform_codes"])
    frame.attrs["macroforecast_metadata"] = normalized_metadata
    return DataBundle(panel=frame, metadata=normalized_metadata)


def _normalize_combined_frequency(frequency: str) -> str:
    aliases = {
        "native": "native",
        "mixed": "native",
        "keep": "native",
        "monthly": "monthly",
        "month": "monthly",
        "quarterly": "quarterly",
        "quarter": "quarterly",
    }
    key = str(frequency).lower()
    if key not in aliases:
        raise ValueError("frequency must be one of ['native', 'monthly', 'quarterly']")
    return aliases[key]


def _align_bundle_for_combination(
    bundle: DataBundle,
    *,
    frequency: str,
    quarterly_to_monthly: str,
    monthly_to_quarterly: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = bundle.panel.copy()
    native_frequencies = _bundle_native_frequencies(bundle)
    date_anchors = _bundle_date_anchors(bundle)
    native_frequency_summary = {
        "native_frequency_counts": dict(sorted(Counter(native_frequencies.values()).items())),
        "native_frequency_by_column": dict(sorted(native_frequencies.items())),
    }
    if date_anchors:
        native_frequency_summary["date_anchor_counts"] = dict(sorted(Counter(date_anchors.values()).items()))
        native_frequency_summary["date_anchor_by_column"] = dict(sorted(date_anchors.items()))
    if frequency == "native":
        return panel, {
            "method": "native",
            "source_frequency": bundle.metadata.get("frequency"),
            "frequency_conversions": [],
            **native_frequency_summary,
        }
    unsupported = {
        column: value
        for column, value in native_frequencies.items()
        if value not in {"monthly", "quarterly"}
    }
    if unsupported:
        sample = list(unsupported.items())[:5]
        raise ValueError(
            "combined monthly/quarterly output supports only monthly and quarterly native columns; "
            f"unsupported columns include {sample}. Use frequency='native' to inspect them first."
        )
    if frequency == "monthly":
        conversions = _frequency_conversion_records(
            native_frequencies,
            from_frequency="quarterly",
            to_frequency="monthly",
            rule=quarterly_to_monthly,
        )
        aligned = _align_panel_to_monthly(
            panel,
            native_frequencies=native_frequencies,
            quarterly_to_monthly=quarterly_to_monthly,
        )
    elif frequency == "quarterly":
        conversions = _frequency_conversion_records(
            native_frequencies,
            from_frequency="monthly",
            to_frequency="quarterly",
            rule=monthly_to_quarterly,
        )
        aligned = _align_panel_to_quarterly(
            panel,
            native_frequencies=native_frequencies,
            monthly_to_quarterly=monthly_to_quarterly,
        )
    else:  # pragma: no cover - guarded by _normalize_combined_frequency
        raise ValueError(f"unknown combined frequency {frequency!r}")
    aligned.attrs.update(dict(getattr(panel, "attrs", {}) or {}))
    return aligned, {
        "method": f"align_to_{frequency}",
        "source_frequency": bundle.metadata.get("frequency"),
        "frequency_conversions": conversions,
        **native_frequency_summary,
    }


def _bundle_native_frequencies(bundle: DataBundle) -> dict[str, str]:
    metadata_frequency = str(bundle.metadata.get("frequency", "")).lower()
    if metadata_frequency == "monthly":
        return {str(column): "monthly" for column in bundle.panel.columns}
    if metadata_frequency == "quarterly":
        return {str(column): "quarterly" for column in bundle.panel.columns}
    report_frequencies = _fred_sd_frequency_map(bundle.panel)
    if report_frequencies:
        return {
            str(column): report_frequencies.get(str(column), _infer_native_frequency(bundle.panel[column]))
            for column in bundle.panel.columns
        }
    return {str(column): _infer_native_frequency(bundle.panel[column]) for column in bundle.panel.columns}


def _fred_sd_frequency_map(panel: pd.DataFrame) -> dict[str, str]:
    reports = getattr(panel, "attrs", {}).get("macrocast_reports", {})
    if not isinstance(reports, Mapping):
        return {}
    report = reports.get("fred_sd_series_metadata", {})
    if not isinstance(report, Mapping):
        return {}
    rows = report.get("series", ())
    if not isinstance(rows, (list, tuple)):
        return {}
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        column = row.get("column")
        frequency = row.get("native_frequency")
        if column is not None and frequency:
            result[str(column)] = str(frequency)
    return result


def _bundle_date_anchors(bundle: DataBundle) -> dict[str, str]:
    metadata_anchors = bundle.metadata.get("date_anchor_by_column")
    if isinstance(metadata_anchors, Mapping):
        column_names = {str(column) for column in bundle.panel.columns}
        return {
            str(column): str(anchor)
            for column, anchor in metadata_anchors.items()
            if str(column) in column_names
        }
    return _fred_sd_date_anchor_map(bundle.panel)


def _fred_sd_date_anchor_map(panel: pd.DataFrame) -> dict[str, str]:
    reports = getattr(panel, "attrs", {}).get("macrocast_reports", {})
    if not isinstance(reports, Mapping):
        return {}
    report = reports.get("fred_sd_series_metadata", {})
    if not isinstance(report, Mapping):
        return {}
    rows = report.get("series", ())
    if not isinstance(rows, (list, tuple)):
        return {}
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        column = row.get("column")
        anchor = row.get("date_anchor")
        if column is not None and anchor:
            result[str(column)] = str(anchor)
    return result


def _align_panel_to_monthly(
    panel: pd.DataFrame,
    *,
    native_frequencies: Mapping[str, str],
    quarterly_to_monthly: str,
) -> pd.DataFrame:
    index = _monthly_index(panel)
    columns: dict[str, pd.Series] = {}
    for column in panel.columns:
        name = str(column)
        series = panel[column].dropna()
        native = native_frequencies.get(name, "unknown")
        if native == "quarterly":
            aligned = _quarterly_series_to_monthly(series, quarterly_to_monthly, index=index)
        elif native in {"monthly", "unknown", "irregular"}:
            aligned = _series_to_monthly(series)
        else:
            raise ValueError(f"cannot align {name!r} with native frequency {native!r} to monthly")
        columns[name] = aligned.reindex(index)
    return pd.DataFrame(columns, index=index)


def _align_panel_to_quarterly(
    panel: pd.DataFrame,
    *,
    native_frequencies: Mapping[str, str],
    monthly_to_quarterly: str,
) -> pd.DataFrame:
    index = _quarterly_index(panel)
    columns: dict[str, pd.Series] = {}
    for column in panel.columns:
        name = str(column)
        series = panel[column].dropna()
        native = native_frequencies.get(name, "unknown")
        if native == "monthly":
            aligned = _monthly_series_to_quarterly(series, monthly_to_quarterly)
        elif native in {"quarterly", "unknown", "irregular"}:
            aligned = _series_to_quarterly(series)
        else:
            raise ValueError(f"cannot align {name!r} with native frequency {native!r} to quarterly")
        columns[name] = aligned.reindex(index)
    return pd.DataFrame(columns, index=index)


def _monthly_index(panel: pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(panel.index)
    start = index.min().to_period("M").to_timestamp()
    end = index.max().to_period("M").to_timestamp()
    return pd.date_range(start, end, freq="MS", name="date")


def _quarterly_index(panel: pd.DataFrame) -> pd.DatetimeIndex:
    periods = pd.DatetimeIndex(panel.index).to_period("Q")
    start = periods.min().start_time
    end = periods.max().start_time
    return pd.date_range(start, end, freq="QS", name="date")


def _series_to_monthly(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.copy()
    result = series.copy()
    result.index = pd.DatetimeIndex(result.index).to_period("M").to_timestamp()
    return result.groupby(level=0).last().sort_index()


def _series_to_quarterly(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.copy()
    result = series.copy()
    result.index = pd.DatetimeIndex(result.index).to_period("Q").to_timestamp()
    return result.groupby(level=0).last().sort_index()


def _quarterly_series_to_monthly(series: pd.Series, rule: str, *, index: pd.DatetimeIndex) -> pd.Series:
    key = str(rule).lower()
    quarterly = _series_to_quarterly(series)
    if key in {"repeat_within_quarter", "repeat", "spread"}:
        by_quarter = dict(zip(pd.DatetimeIndex(quarterly.index).to_period("Q"), quarterly.to_numpy(), strict=False))
        values = [by_quarter.get(period) for period in pd.DatetimeIndex(index).to_period("Q")]
        return pd.Series(values, index=index, dtype="float64")
    if key in {"quarter_end_ffill", "step_forward", "ffill_from_quarter_end"}:
        observed = quarterly.copy()
        observed.index = pd.DatetimeIndex(observed.index).to_period("Q").asfreq("M", how="end").to_timestamp()
        return observed.reindex(index).ffill()
    if key in {"linear_interpolation", "linear"}:
        observed = quarterly.copy()
        observed.index = pd.DatetimeIndex(observed.index).to_period("Q").asfreq("M", how="end").to_timestamp()
        return observed.reindex(index.union(observed.index)).sort_index().interpolate(method="time").reindex(index)
    raise ValueError(
        "quarterly_to_monthly must be one of "
        "['repeat_within_quarter', 'quarter_end_ffill', 'linear_interpolation']"
    )


def _monthly_series_to_quarterly(series: pd.Series, rule: str) -> pd.Series:
    key = str(rule).lower()
    monthly = _series_to_monthly(series)
    if key in {"quarterly_average", "average", "mean"}:
        return monthly.groupby(pd.DatetimeIndex(monthly.index).to_period("Q").to_timestamp()).mean()
    if key in {"quarterly_endpoint", "endpoint", "last"}:
        return monthly.groupby(pd.DatetimeIndex(monthly.index).to_period("Q").to_timestamp()).last()
    if key in {"quarterly_sum", "sum"}:
        return monthly.groupby(pd.DatetimeIndex(monthly.index).to_period("Q").to_timestamp()).sum(min_count=1)
    raise ValueError("monthly_to_quarterly must be one of ['quarterly_average', 'quarterly_endpoint', 'quarterly_sum']")


def _frequency_conversion_records(
    native_frequencies: Mapping[str, str],
    *,
    from_frequency: str,
    to_frequency: str,
    rule: str,
) -> list[dict[str, Any]]:
    columns = [column for column, frequency in native_frequencies.items() if frequency == from_frequency]
    if not columns:
        return []
    variables = sorted({_series_variable_name(column) for column in columns})
    return [
        {
            "from_frequency": from_frequency,
            "to_frequency": to_frequency,
            "rule": str(rule),
            "variables": variables,
            "columns": sorted(columns),
            "n_columns": len(columns),
        }
    ]


def _series_variable_name(column: str) -> str:
    name = str(column)
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[0]


def _emit_frequency_conversion_warnings(records: list[dict[str, Any]]) -> None:
    for record in records:
        variables = ", ".join(record["variables"][:8])
        if len(record["variables"]) > 8:
            variables = f"{variables}, ..."
        message = (
            f"{record['dataset']} {record['from_frequency']} variables were aligned to "
            f"{record['to_frequency']} using {record['rule']}: {variables} "
            f"({record['n_columns']} columns)."
        )
        warnings.warn(message, UserWarning, stacklevel=3)


def _merge_macrocast_reports(target: dict[str, Any], source: Any, *, source_dataset: str) -> None:
    if not isinstance(source, Mapping):
        return
    for key, value in source.items():
        if key not in target:
            target[key] = value
        else:
            target[f"{source_dataset}.{key}"] = value


def _combined_version_mode(sources: list[dict[str, Any]]) -> str:
    modes = {str(source.get("version_mode")) for source in sources}
    return modes.pop() if len(modes) == 1 else "mixed"


def _combined_vintage(sources: list[dict[str, Any]]) -> str | None:
    vintages = {source.get("vintage") for source in sources}
    return vintages.pop() if len(vintages) == 1 else None


def _combined_parse_notes(dataset: str, frequency: str) -> list[str]:
    notes = ["combined national and state-level data bundle"]
    if dataset == "fred_md+fred_sd" and frequency == "quarterly":
        notes.append("quarterly alignment of FRED-MD is supported but not recommended; prefer fred_qd+fred_sd for quarterly targets")
    if dataset == "fred_qd+fred_sd" and frequency == "monthly":
        notes.append("monthly alignment of FRED-QD is supported but not recommended; prefer fred_md+fred_sd for monthly targets")
    return notes


def _data_through(panel: pd.DataFrame) -> str | None:
    index = pd.DatetimeIndex(panel.index)
    valid = index[index.notna()]
    return valid[-1].strftime("%Y-%m") if len(valid) else None


def _extract_md_vintage_from_zip(zip_path: Path, vintage: str, target: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [f"{vintage}.csv", f"{vintage}-md.csv"]
        match = next((name for name in zf.namelist() if any(name.endswith(candidate) for candidate in candidates)), None)
        if match is None:
            raise RawDownloadError(f"vintage {vintage!r} not found in historical zip {zip_path}")
        with zf.open(match) as src:
            _atomic_write(src.read(), target)


def _parse_fred_csv(filepath: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    path = Path(filepath)
    raw = pd.read_csv(path, header=None, dtype=str, na_values=["", ".", " "])
    if raw.shape[0] < 3 or raw.shape[1] < 2:
        raise ValueError(f"file does not look like a FRED CSV: {path}")

    header_idx: int | None = None
    tcodes_idx: int | None = None
    for idx, value in raw.iloc[:, 0].items():
        label = str(value).strip().lower()
        if label == "nan":
            label = ""
        if label in {"sasdate", "sasqdate"} and header_idx is None:
            header_idx = int(idx)
        elif label in {"transform", "transform:"} and tcodes_idx is None:
            tcodes_idx = int(idx)
        if header_idx is not None and tcodes_idx is not None:
            break

    if header_idx is None:
        fallback_header = str(raw.iloc[1, 0]).strip().lower()
        if fallback_header in {"sasdate", "sasqdate"}:
            header_idx = 1
            tcodes_idx = 0
        else:
            raise ValueError(f"missing sasdate/sasqdate header row in {path}")
    elif tcodes_idx is None:
        tcodes_idx = header_idx - 1 if header_idx > 0 else header_idx + 1

    header_row = raw.iloc[header_idx].tolist()
    tcodes_row = raw.iloc[tcodes_idx].tolist()
    data_start = max(header_idx, tcodes_idx) + 1

    columns = [str(x).strip() for x in header_row]
    if not columns or columns[0].lower() not in {"sasdate", "sasqdate"}:
        raise ValueError(f"missing sasdate/sasqdate header row in {path}")

    tcodes: dict[str, int] = {}
    for name, value in zip(columns[1:], tcodes_row[1:], strict=False):
        try:
            tcodes[name] = int(float(str(value)))
        except (TypeError, ValueError):
            tcodes[name] = 1

    data = raw.iloc[data_start:].copy()
    data.columns = columns
    date_col = columns[0]
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data[data[date_col].notna()].set_index(date_col)
    for col in columns[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data.index.name = "date"
    data.sort_index(inplace=True)
    return data, tcodes


def _fred_sd_local_source_format(local_source: str | Path | None) -> str:
    if local_source is None:
        return "xlsx"
    suffix = Path(local_source).suffix.lower().lstrip(".")
    return "csv" if suffix == "csv" else "xlsx"


def _fred_sd_request_url(url: str):
    return urlopen(Request(url, headers=_FRED_SD_HEADERS))


def _is_remote_xlsx(payload: bytes, content_type: str | None) -> bool:
    if not payload.startswith(b"PK"):
        return False
    if content_type is None:
        return True
    normalized = content_type.lower()
    return "spreadsheetml.sheet" in normalized or "application/octet-stream" in normalized


def _fred_sd_read_remote_xlsx(url: str) -> tuple[str, bytes] | None:
    with _fred_sd_request_url(url) as src:
        payload = src.read()
        content_type = src.headers.get("Content-Type")
        if _is_remote_xlsx(payload, content_type):
            return src.geturl(), payload
    return None


def _latest_series_url_from_html(html: str) -> str:
    candidates: list[tuple[str, str]] = []
    for href, vintage in _FRED_SD_SERIES_XLSX_LINK_RE.findall(html):
        candidates.append((vintage, urljoin(_FRED_SD_LANDING_PAGE_URL, unescape(href))))
    if not candidates:
        raise RawDownloadError("failed to find a FRED-SD by-series workbook link on the official landing page")
    return max(candidates, key=lambda item: item[0])[1]


def _fred_sd_latest_series_url() -> str:
    with _fred_sd_request_url(_FRED_SD_LANDING_PAGE_URL) as src:
        html = src.read().decode("utf-8", errors="ignore")
    return _latest_series_url_from_html(html)


def _fred_sd_series_xlsx_url(vintage: str) -> str:
    return _FRED_SD_SERIES_XLSX_URL_TEMPLATE.format(vintage=vintage)


def _fred_sd_series_zip_url(vintage: str) -> str:
    year = int(vintage[:4])
    if year >= 2023:
        return _FRED_SD_SERIES_YEAR_ZIP_URL_TEMPLATE.format(year=year)
    start_year = year if year % 2 else year - 1
    end_year = start_year + 1
    return _FRED_SD_SERIES_RANGE_ZIP_URL_TEMPLATE.format(start_year=start_year, end_year=end_year)


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


def _fred_sd_read_vintage_series_xlsx(vintage: str) -> tuple[str, bytes]:
    direct_url = _fred_sd_series_xlsx_url(vintage)
    direct = _fred_sd_read_remote_xlsx(direct_url)
    if direct is not None:
        return direct
    zip_url = _fred_sd_series_zip_url(vintage)
    with _fred_sd_request_url(zip_url) as src:
        payload = src.read()
        content_type = (src.headers.get("Content-Type") or "").lower()
        if not payload.startswith(b"PK") or "html" in content_type:
            raise RawDownloadError(f"failed to obtain FRED-SD by-series zip for vintage={vintage!r}")
        entry, workbook = _extract_vintage_xlsx_from_zip(payload, vintage)
        return f"{src.geturl()}#{entry}", workbook


def _fred_sd_read_current_series_xlsx() -> tuple[str, bytes]:
    current_url = _fred_sd_latest_series_url()
    current = _fred_sd_read_remote_xlsx(current_url)
    if current is None:
        raise RawDownloadError(f"latest FRED-SD by-series link did not return an Excel workbook: {current_url}")
    return current


def _filter_wide_csv_columns(df: pd.DataFrame, *, states: list[str] | None, variables: list[str] | None) -> pd.DataFrame:
    if states is None and variables is None:
        return df
    state_set = set(states or [])
    variable_set = set(variables or [])
    keep: list[str] = []
    for column in df.columns:
        name = str(column)
        variable, state = name.rsplit("_", 1) if "_" in name else (name, "")
        if variable_set and variable not in variable_set:
            continue
        if state_set and state not in state_set:
            continue
        keep.append(column)
    return df[keep].copy()


def _read_local_fred_sd_csv(path: Path, *, states: list[str] | None, variables: list[str] | None) -> pd.DataFrame:
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


def _read_fred_sd_workbook(path: Path, *, states: list[str] | None, variables: list[str] | None) -> pd.DataFrame:
    sheet_name = None if variables is None else list(variables)
    loaded_sheets = pd.read_excel(path, sheet_name=sheet_name, index_col=0, engine="openpyxl")
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
    df = pd.concat(wide_frames, axis=1, sort=False)
    df.index.name = "date"
    df.sort_index(inplace=True)
    return df


def _column_observed_window(series: pd.Series) -> tuple[str | None, str | None, int]:
    observed = series.dropna()
    if observed.empty:
        return None, None, 0
    return observed.index[0].strftime("%Y-%m-%d"), observed.index[-1].strftime("%Y-%m-%d"), int(observed.shape[0])


def _infer_native_frequency(series: pd.Series) -> str:
    observed = series.dropna()
    if observed.shape[0] < 2:
        return "unknown"
    index = pd.DatetimeIndex(observed.index).sort_values()
    day_deltas = [
        int((right - left).days)
        for left, right in zip(index[:-1], index[1:], strict=False)
        if right > left
    ]
    if not day_deltas:
        return "unknown"
    median_days = float(pd.Series(day_deltas).median())
    if 5 <= median_days <= 10:
        return "weekly"
    if 25 <= median_days <= 35:
        return "monthly"
    if 80 <= median_days <= 100:
        return "quarterly"
    if 350 <= median_days <= 380:
        return "annual"
    return "irregular"


def _infer_date_anchor(series: pd.Series, native_frequency: str) -> str:
    observed = series.dropna()
    if observed.empty:
        return "none"
    index = pd.DatetimeIndex(observed.index).sort_values()
    if native_frequency == "monthly":
        month_start = index.to_period("M").to_timestamp()
        month_end = index.to_period("M").asfreq("D", how="end").to_timestamp()
        if index.equals(month_start):
            return "month_start"
        if index.equals(month_end):
            return "month_end"
        if len(set(index.weekday)) == 1:
            return "monthly_weekday_anchor"
        return "monthly_other_anchor"
    if native_frequency == "quarterly":
        quarter_start = index.to_period("Q").to_timestamp()
        quarter_end = index.to_period("Q").asfreq("D", how="end").to_timestamp()
        if index.equals(quarter_start):
            return "quarter_start"
        if index.equals(quarter_end):
            return "quarter_end"
        return "quarterly_other_anchor"
    if native_frequency == "weekly":
        return "weekly"
    if native_frequency == "annual":
        return "annual"
    return native_frequency


def _fred_sd_panel_frequency(report: Mapping[str, object]) -> str:
    counts = report.get("native_frequency_counts", {})
    if not isinstance(counts, Mapping) or not counts:
        return "unknown"
    nonzero = [str(name) for name, count in counts.items() if int(count) > 0]
    if len(nonzero) == 1:
        return nonzero[0]
    return "mixed"


def _fred_sd_column_parts(column: object) -> tuple[str, str]:
    name = str(column)
    if "_" not in name:
        return name, ""
    variable, state = name.rsplit("_", 1)
    return variable, state


def _fred_sd_series_metadata(panel: pd.DataFrame, *, states: list[str] | None, variables: list[str] | None, source_format: str) -> dict[str, object]:
    series: list[dict[str, object]] = []
    for column in panel.columns:
        variable, state = _fred_sd_column_parts(column)
        observed_start, observed_end, non_missing_count = _column_observed_window(panel[column])
        native_frequency = _infer_native_frequency(panel[column])
        date_anchor = _infer_date_anchor(panel[column], native_frequency)
        series.append(
            {
                "column": str(column),
                "sd_variable": variable,
                "state": state,
                "source_sheet": variable,
                "native_frequency": native_frequency,
                "date_anchor": date_anchor,
                "observed_start": observed_start,
                "observed_end": observed_end,
                "non_missing_observation_count": non_missing_count,
            }
        )
    states_seen = sorted({str(row["state"]) for row in series if row["state"]})
    variables_seen = sorted({str(row["sd_variable"]) for row in series})
    frequency_counts = Counter(str(row["native_frequency"]) for row in series)
    date_anchor_counts = Counter(str(row["date_anchor"]) for row in series)
    return {
        "schema_version": _FRED_SD_SERIES_METADATA_CONTRACT_VERSION,
        "contract_version": _FRED_SD_SERIES_METADATA_CONTRACT_VERSION,
        "owner_stage": "data",
        "dataset": "fred_sd",
        "source_format": source_format,
        "selector": {
            "states": None if states is None else [str(state) for state in states],
            "variables": None if variables is None else [str(variable) for variable in variables],
        },
        "series_count": len(series),
        "state_count": len(states_seen),
        "sd_variable_count": len(variables_seen),
        "states": states_seen,
        "sd_variables": variables_seen,
        "native_frequency_counts": dict(sorted(frequency_counts.items())),
        "date_anchor_counts": dict(sorted(date_anchor_counts.items())),
        "series": series,
    }


__all__ = [
    "combine",
    "load_fred_md",
    "load_fred_qd",
    "load_fred_sd",
    "load_fred_md_sd",
    "load_fred_qd_sd",
    "load_custom_csv",
    "load_custom_parquet",
    "list_vintages",
]
