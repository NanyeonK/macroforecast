from __future__ import annotations

from typing import Any

from .cache import get_manifest_path, get_raw_cache_root, get_raw_file_path
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
from .panel import (
    DataBundle,
    DataSpec,
    as_panel,
    attach_metadata,
    bundle_from_result,
    metadata,
    panel_info,
    spec,
    validate_panel,
)
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


def load_fred_md(*args: Any, **kwargs: Any) -> DataBundle:
    return bundle_from_result(load_fred_md_result(*args, **kwargs))


def load_fred_qd(*args: Any, **kwargs: Any) -> DataBundle:
    return bundle_from_result(load_fred_qd_result(*args, **kwargs))


def load_fred_sd(*args: Any, **kwargs: Any) -> DataBundle:
    return bundle_from_result(load_fred_sd_result(*args, **kwargs))


def load_custom_csv(*args: Any, **kwargs: Any) -> DataBundle:
    return bundle_from_result(load_custom_csv_result(*args, **kwargs))


def load_custom_parquet(*args: Any, **kwargs: Any) -> DataBundle:
    return bundle_from_result(load_custom_parquet_result(*args, **kwargs))

__all__ = [
    "DataBundle",
    "DataSpec",
    "as_panel",
    "attach_metadata",
    "metadata",
    "panel_info",
    "spec",
    "validate_panel",
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
