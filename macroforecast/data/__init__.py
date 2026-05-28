from __future__ import annotations

from typing import Any

import pandas as pd

from .cache import get_manifest_path, get_raw_cache_root, get_raw_file_path
from .config import data
from .sources import (
    load_custom_csv as load_custom_csv_result,
    load_custom_parquet as load_custom_parquet_result,
    load_fred_md as load_fred_md_result,
    load_fred_qd as load_fred_qd_result,
    load_fred_sd as load_fred_sd_result,
    parse_fred_csv,
)
from .manifest import append_raw_manifest_entry, read_raw_manifest
from .manager import build_raw_artifact_record, list_vintages, normalize_version_request
from .sd_inferred_tcodes import (
    DEFAULT_RUNTIME_STATUSES,
    MAP_VERSION as SD_INFERRED_TCODE_MAP_VERSION,
    OFFICIAL as SD_INFERRED_TCODE_OFFICIAL,
    SD_INFERRED_TCODE_MAP,
    SOURCE as SD_INFERRED_TCODE_SOURCE,
    STATE_SERIES_STATIONARITY_OVERRIDE_VERSION,
    VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION,
    VARIABLE_GLOBAL_STATIONARITY_TCODE_MAP,
    build_sd_state_series_stationarity_transform_codes,
    build_sd_transform_codes_for_policy,
    build_sd_variable_global_stationarity_transform_codes,
    build_sd_inferred_transform_codes,
    resolve_sd_inferred_tcode,
)
from .types import RawArtifactRecord, RawDatasetMetadata, RawLoadResult, RawVersionRequest


def metadata(obj: pd.DataFrame | RawLoadResult) -> dict[str, Any]:
    """Return macroforecast data metadata for a loaded frame or raw result."""

    if isinstance(obj, RawLoadResult):
        return _metadata_from_result(obj)
    if isinstance(obj, pd.DataFrame):
        return dict(obj.attrs.get("macroforecast_metadata", {}))
    raise TypeError("metadata expects a pandas DataFrame or RawLoadResult")


def load_fred_md(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return _frame_from_result(load_fred_md_result(*args, **kwargs))


def load_fred_qd(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return _frame_from_result(load_fred_qd_result(*args, **kwargs))


def load_fred_sd(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return _frame_from_result(load_fred_sd_result(*args, **kwargs))


def load_custom_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return _frame_from_result(load_custom_csv_result(*args, **kwargs))


def load_custom_parquet(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return _frame_from_result(load_custom_parquet_result(*args, **kwargs))


def _frame_from_result(result: RawLoadResult) -> pd.DataFrame:
    frame = result.data.copy()
    frame.attrs["macroforecast_metadata"] = _metadata_from_result(result)
    if result.transform_codes:
        frame.attrs["macroforecast_transform_codes"] = dict(result.transform_codes)
    return frame


def _metadata_from_result(result: RawLoadResult) -> dict[str, Any]:
    return {
        "dataset": result.dataset_metadata.dataset,
        "source_family": result.dataset_metadata.source_family,
        "frequency": result.dataset_metadata.frequency,
        "version_mode": result.dataset_metadata.version_mode,
        "vintage": result.dataset_metadata.vintage,
        "data_through": result.dataset_metadata.data_through,
        "support_tier": result.dataset_metadata.support_tier,
        "parse_notes": result.dataset_metadata.parse_notes,
        "artifact": {
            "source_url": result.artifact.source_url,
            "local_path": result.artifact.local_path,
            "file_format": result.artifact.file_format,
            "downloaded_at": result.artifact.downloaded_at,
            "file_sha256": result.artifact.file_sha256,
            "file_size_bytes": result.artifact.file_size_bytes,
            "cache_hit": result.artifact.cache_hit,
            "manifest_version": result.artifact.manifest_version,
        },
        "transform_codes": dict(result.transform_codes),
    }

__all__ = [
    "data",
    "metadata",
    "normalize_version_request",
    "list_vintages",
    "get_raw_cache_root",
    "get_manifest_path",
    "get_raw_file_path",
    "build_raw_artifact_record",
    "append_raw_manifest_entry",
    "read_raw_manifest",
    "parse_fred_csv",
    "load_fred_md",
    "load_fred_qd",
    "load_fred_sd",
    "load_fred_md_result",
    "load_fred_qd_result",
    "load_fred_sd_result",
    "load_custom_csv",
    "load_custom_parquet",
    "load_custom_csv_result",
    "load_custom_parquet_result",
    "SD_INFERRED_TCODE_MAP",
    "SD_INFERRED_TCODE_MAP_VERSION",
    "SD_INFERRED_TCODE_OFFICIAL",
    "SD_INFERRED_TCODE_SOURCE",
    "VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION",
    "VARIABLE_GLOBAL_STATIONARITY_TCODE_MAP",
    "STATE_SERIES_STATIONARITY_OVERRIDE_VERSION",
    "DEFAULT_RUNTIME_STATUSES",
    "resolve_sd_inferred_tcode",
    "build_sd_inferred_transform_codes",
    "build_sd_variable_global_stationarity_transform_codes",
    "build_sd_state_series_stationarity_transform_codes",
    "build_sd_transform_codes_for_policy",
    "RawVersionRequest",
    "RawDatasetMetadata",
    "RawArtifactRecord",
    "RawLoadResult",
]
