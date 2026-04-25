from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from ..cache import atomic_copy_to_cache, atomic_write_bytes_to_cache, get_raw_file_path
from ..errors import RawDownloadError, RawParseError
from ..manager import build_raw_artifact_record, normalize_version_request
from ..manifest import append_raw_manifest_entry
from ..types import RawDatasetMetadata, RawLoadResult

_CURRENT_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/FRED_SD.xlsx"
_VINTAGE_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/{vintage}.xlsx"


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
    target = get_raw_file_path(request, cache_root, suffix="xlsx")

    cache_hit = target.exists() and not force and local_source is None
    source_url = _CURRENT_URL if request.mode == "current" else _VINTAGE_URL.format(vintage=request.vintage)

    if not cache_hit:
        try:
            if local_source is not None:
                atomic_copy_to_cache(Path(local_source), target)
                source_url = str(local_source)
            else:
                with urlopen(source_url) as src:
                    atomic_write_bytes_to_cache(src.read(), target)
        except Exception as exc:
            raise RawDownloadError(f"failed to obtain FRED-SD raw file for request={request}") from exc

    try:
        sheets: dict[str, pd.DataFrame] = pd.read_excel(target, sheet_name=None, index_col=0, engine="openpyxl")
    except Exception as exc:
        raise RawParseError(f"failed to parse FRED-SD workbook at {target}") from exc

    if variables is not None:
        sheets = {k: v for k, v in sheets.items() if k in variables}
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

    artifact = build_raw_artifact_record(
        request=request,
        source_url=source_url,
        local_path=target,
        file_format="xlsx",
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
