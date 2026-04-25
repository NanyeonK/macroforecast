"""Custom CSV loader — load a user-supplied CSV conforming to a FRED-family schema.

The caller provides the CSV path via ``leaf_config.custom_data_path``;
the compiler validates that the path is set when
``source_adapter == "custom_csv"``. The ``dataset`` axis still declares
the schema (``fred_md`` / ``fred_qd`` / ``fred_sd``) so downstream code
treats the panel identically to the canonical FRED loader output.

Schema requirements:

- First column is a date index (any parseable date format).
- Remaining columns are numeric; column names are series IDs.
- Optional first row may hold FRED-MD-style transformation codes — the
  loader does NOT detect or consume this automatically. Users who need
  T-code-aware preprocessing should pre-strip the T-code row from the
  file OR rely on Layer 2 ``tcode_policy: raw_only`` (the default).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..errors import RawParseError
from ..manager import build_raw_artifact_record, normalize_version_request
from ..manifest import append_raw_manifest_entry
from ..types import RawDatasetMetadata, RawLoadResult


_SUPPORTED_SCHEMAS = {"fred_md", "fred_qd", "fred_sd"}


def load_custom_csv(
    path: str | Path,
    *,
    dataset: str,
    cache_root: str | Path | None = None,
) -> RawLoadResult:
    """Load ``path`` as a CSV conforming to ``dataset``'s schema.

    Args:
        path: Filesystem path to the CSV.
        dataset: Schema label (``fred_md`` / ``fred_qd`` / ``fred_sd``).
            Determines how the loaded panel is labelled downstream.
        cache_root: Ignored for custom loaders (no cache — user supplies
            the file directly). Accepted to match the canonical FRED
            loader signature.
    """

    if dataset not in _SUPPORTED_SCHEMAS:
        raise RawParseError(
            f"dataset={dataset!r} is not a supported schema for custom_csv; "
            f"expected one of {_SUPPORTED_SCHEMAS}"
        )

    csv_path = Path(path)
    if not csv_path.exists():
        raise RawParseError(f"custom_csv source path does not exist: {csv_path}")

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as exc:
        raise RawParseError(f"failed to parse custom CSV at {csv_path}") from exc

    if not isinstance(df.index, pd.DatetimeIndex):
        raise RawParseError(
            f"custom CSV at {csv_path} must have a parseable date index as its first column"
        )

    # Coerce all remaining columns to numeric, dropping any that are
    # entirely non-numeric (matches FRED loader behaviour).
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df.index.name = "date"
    df.sort_index(inplace=True)

    request = normalize_version_request(dataset, vintage=None)
    artifact = build_raw_artifact_record(
        request=request,
        source_url=str(csv_path),
        local_path=csv_path,
        file_format="csv",
        cache_hit=False,
    )
    metadata = RawDatasetMetadata(
        dataset=dataset,
        source_family="custom-csv",
        frequency=_frequency_for_dataset(dataset),
        version_mode="current",
        vintage=None,
        data_through=df.index[-1].strftime("%Y-%m") if len(df) else None,
        support_tier="provisional",
        parse_notes=("user-supplied CSV; no vintage tracking",),
    )
    result = RawLoadResult(data=df, dataset_metadata=metadata, artifact=artifact)
    if cache_root is not None:
        append_raw_manifest_entry(result, cache_root=cache_root)
    return result


def _frequency_for_dataset(dataset: str) -> str:
    return {
        "fred_md": "monthly",
        "fred_qd": "quarterly",
        "fred_sd": "state_monthly",
    }.get(dataset, "monthly")


__all__ = ["load_custom_csv"]
