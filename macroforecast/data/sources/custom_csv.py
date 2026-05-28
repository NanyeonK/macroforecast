"""Custom CSV loader for user-supplied macro panels.

The loader reads a CSV file, normalizes it to macroforecast's canonical
``DatetimeIndex`` panel contract, and returns a raw load result that the
public ``macroforecast.data.load_custom_csv`` wrapper converts to
``DataBundle``.

Schema requirements:

- A parseable date column, or a first column that can be parsed as dates.
- Remaining retained columns are macro variables.
- Values are coerced to numeric values or ``NaN``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from ..errors import RawParseError
from ..manifest import append_raw_manifest_entry
from ..panel import as_panel
from ..types import RawDatasetMetadata, RawLoadResult
from ..types import RawArtifactRecord


def load_custom_csv(
    path: str | Path,
    *,
    date: str | None = None,
    columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    dataset: str = "custom",
    frequency: str = "unknown",
    metadata: Mapping[str, Any] | None = None,
    cache_root: str | Path | None = None,
) -> RawLoadResult:
    """Load ``path`` as a user panel and normalize it to the canonical format.

    Args:
        path: Filesystem path to the CSV.
        date: Optional date column name. If omitted, a DatetimeIndex is used
            when present, otherwise the first column is parsed as dates.
        columns: Optional columns to keep before renaming.
        rename: Optional mapping from source column names to canonical names.
        dataset: Metadata label for the loaded panel.
        frequency: Metadata frequency label.
        metadata: Optional user metadata to attach.
        cache_root: Ignored for custom loaders (no cache — user supplies
            the file directly). Accepted to match the canonical FRED
            loader signature.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise RawParseError(f"custom_csv source path does not exist: {csv_path}")

    try:
        raw = pd.read_csv(csv_path)
    except Exception as exc:
        raise RawParseError(f"failed to parse custom CSV at {csv_path}") from exc

    try:
        df = as_panel(raw, date=date, columns=columns, rename=rename, metadata=metadata)
    except Exception as exc:
        raise RawParseError(f"failed to normalize custom CSV at {csv_path}") from exc

    artifact = _custom_artifact(dataset=dataset, path=csv_path, file_format="csv")
    metadata = RawDatasetMetadata(
        dataset=dataset,
        source_family="custom-csv",
        frequency=frequency,
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


def _custom_artifact(*, dataset: str, path: Path, file_format: str) -> RawArtifactRecord:
    content = path.read_bytes()
    return RawArtifactRecord(
        dataset=dataset,
        version_mode="current",
        vintage=None,
        source_url=str(path),
        local_path=str(path),
        file_format=file_format,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        file_sha256=hashlib.sha256(content).hexdigest(),
        file_size_bytes=len(content),
        cache_hit=False,
        manifest_version="v1",
    )


__all__ = ["load_custom_csv"]
