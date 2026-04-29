"""Custom Parquet loader — load a user-supplied Parquet file conforming to a FRED-family schema.

Parquet-counterpart to :mod:`macrocast.raw.datasets.custom_csv`. Same
contract: recipes choose an official ``dataset``, select
``custom_source_policy``, and provide ``leaf_config.custom_source_path``.
The compiler infers the parser from ``.parquet``/``.pq`` and infers the
internal loader schema from the selected ``dataset``/``frequency`` route.

Requires ``pyarrow`` or ``fastparquet`` to be installed (pandas picks
whichever is available).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..errors import RawParseError
from ..manager import build_raw_artifact_record, normalize_version_request
from ..manifest import append_raw_manifest_entry
from ..types import RawDatasetMetadata, RawLoadResult


_SUPPORTED_SCHEMAS = {"fred_md", "fred_qd", "fred_sd"}


def load_custom_parquet(
    path: str | Path,
    *,
    dataset: str,
    cache_root: str | Path | None = None,
) -> RawLoadResult:
    """Load ``path`` as a Parquet file conforming to ``dataset``'s schema.

    See :func:`macrocast.raw.datasets.custom_csv.load_custom_csv` for the
    shared schema contract.
    """

    if dataset not in _SUPPORTED_SCHEMAS:
        raise RawParseError(
            f"dataset={dataset!r} is not a supported schema for custom_parquet; "
            f"expected one of {_SUPPORTED_SCHEMAS}"
        )

    pq_path = Path(path)
    if not pq_path.exists():
        raise RawParseError(f"custom_parquet source path does not exist: {pq_path}")

    try:
        df = pd.read_parquet(pq_path)
    except Exception as exc:
        raise RawParseError(f"failed to parse custom Parquet at {pq_path}") from exc

    # Normalise to date-indexed numeric panel.
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to coerce the first column as the date index.
        date_col = df.columns[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        except Exception:
            raise RawParseError(
                f"custom Parquet at {pq_path} must have a DatetimeIndex "
                "or a parseable date as its first column"
            )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df.index.name = "date"
    df.sort_index(inplace=True)

    request = normalize_version_request(dataset, vintage=None)
    artifact = build_raw_artifact_record(
        request=request,
        source_url=str(pq_path),
        local_path=pq_path,
        file_format="parquet",
        cache_hit=False,
    )
    metadata = RawDatasetMetadata(
        dataset=dataset,
        source_family="custom-parquet",
        frequency=_frequency_for_dataset(dataset),
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


def _frequency_for_dataset(dataset: str) -> str:
    return {
        "fred_md": "monthly",
        "fred_qd": "quarterly",
        "fred_sd": "state_monthly",
    }.get(dataset, "monthly")


__all__ = ["load_custom_parquet"]
