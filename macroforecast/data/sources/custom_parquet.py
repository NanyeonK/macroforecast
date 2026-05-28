"""Custom Parquet loader for user-supplied macro panels.

The loader reads a Parquet file, normalizes it to macroforecast's canonical
``DatetimeIndex`` panel contract, and returns a raw load result that the
public ``macroforecast.data.load_custom_parquet`` wrapper converts to
``DataBundle``. Requires ``pyarrow`` or ``fastparquet`` through pandas.
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
from ..types import RawArtifactRecord
from ..types import RawDatasetMetadata, RawLoadResult


def load_custom_parquet(
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
    """Load ``path`` as a user Parquet panel and normalize it."""

    pq_path = Path(path)
    if not pq_path.exists():
        raise RawParseError(f"custom_parquet source path does not exist: {pq_path}")

    try:
        df = pd.read_parquet(pq_path)
    except Exception as exc:
        raise RawParseError(f"failed to parse custom Parquet at {pq_path}") from exc

    try:
        df = as_panel(df, date=date, columns=columns, rename=rename, metadata=metadata)
    except Exception as exc:
        raise RawParseError(f"failed to normalize custom Parquet at {pq_path}") from exc

    artifact = _custom_artifact(dataset=dataset, path=pq_path, file_format="parquet")
    metadata = RawDatasetMetadata(
        dataset=dataset,
        source_family="custom-parquet",
        frequency=frequency,
        version_mode="current",
        vintage=None,
        data_through=df.index[-1].strftime("%Y-%m") if len(df) else None,
        support_tier="provisional",
        parse_notes=("user-supplied Parquet; no vintage tracking",),
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


__all__ = ["load_custom_parquet"]
