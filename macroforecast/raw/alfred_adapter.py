"""ALFRED (ArchivaL Federal Reserve Economic Data) vintage adapter.

C50 promotion: ``vintage_policy="real_time_alfred"`` is now operational.

Default mode is local snapshot (no network dependency). API mode is
optional, gated behind the ``[alfred]`` extra (``requests>=2.28``).

Schema
------
Local snapshot parquet file expected at::

    {alfred_snapshot_dir}/alfred_vintages.parquet

Columns required:

    series_id       : str  -- FRED series identifier (e.g. "CPIAUCSL")
    observation_date: datetime or str  -- date the data point refers to
    vintage_date    : str "YYYY-MM"   -- vintage release month
    value           : float           -- reported value at that vintage

A CSV fallback is tried when the parquet file is absent:

    {alfred_snapshot_dir}/alfred_vintages.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


_REQUIRED_COLS = {"series_id", "observation_date", "vintage_date", "value"}


def _read_snapshot_file(snapshot_path: Path) -> pd.DataFrame:
    """Read the local ALFRED snapshot file (parquet with CSV fallback).

    Validates that the expected columns are present before returning.
    Raises ``FileNotFoundError`` when the directory does not exist and
    ``ValueError`` when the file is present but the schema is wrong.
    """
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"ALFRED snapshot directory not found: {snapshot_path!s}. "
            "Set alfred_snapshot_dir in leaf_config to point to the directory "
            "containing alfred_vintages.parquet (or alfred_vintages.csv)."
        )
    parquet_file = snapshot_path / "alfred_vintages.parquet"
    csv_file = snapshot_path / "alfred_vintages.csv"
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        source = str(parquet_file)
    elif csv_file.exists():
        df = pd.read_csv(csv_file)
        source = str(csv_file)
    else:
        raise FileNotFoundError(
            f"Neither alfred_vintages.parquet nor alfred_vintages.csv found in "
            f"{snapshot_path!s}. Provide one of these files containing columns: "
            f"{sorted(_REQUIRED_COLS)}."
        )
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"ALFRED snapshot file {source} is missing expected columns: "
            f"{sorted(missing)}. Required schema: {sorted(_REQUIRED_COLS)}."
        )
    return df


def load_alfred_vintage_snapshot(
    snapshot_path: str | Path,
    series_ids: list[str] | None,
    vintage_date: str,
) -> pd.DataFrame:
    """Load a wide-format ALFRED snapshot for a given vintage month.

    Parameters
    ----------
    snapshot_path:
        Path to the directory containing ``alfred_vintages.parquet``
        (or ``alfred_vintages.csv``).
    series_ids:
        If provided, only rows for these FRED series are loaded.
        Pass ``None`` to load all series.
    vintage_date:
        Vintage month in "YYYY-MM" format. Rows with ``vintage_date``
        strictly greater than this value are excluded, so the returned
        data reflects how the world looked at the end of *vintage_date*.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with ``pd.DatetimeIndex`` as the index
        (``observation_date``) and ``series_id`` values as column names.
    """
    resolved_path = Path(snapshot_path)
    df = _read_snapshot_file(resolved_path)

    # Filter to requested series.
    if series_ids is not None:
        df = df[df["series_id"].isin(series_ids)]

    # Restrict to vintages available at or before vintage_date.
    # String comparison is correct for "YYYY-MM" format.
    df = df[df["vintage_date"] <= vintage_date]

    # For each (series_id, observation_date) pair pick the most recent
    # vintage_date available within the cutoff.
    df = (
        df.sort_values("vintage_date")
        .groupby(["series_id", "observation_date"])
        .last()
        .reset_index()
    )

    # Pivot to wide format: rows = observation_date, columns = series_id.
    wide = df.pivot(index="observation_date", columns="series_id", values="value")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "observation_date"
    wide.columns.name = None
    return wide


def load_alfred_vintage_api(
    series_ids: list[str],
    vintage_date: str,
    api_key: str,
) -> pd.DataFrame:
    """Fetch an ALFRED vintage from the FRED REST API.

    This function is only called when ``alfred_mode="api"`` is set in
    ``leaf_config``. It is gated behind the ``[alfred]`` extra
    (``requests>=2.28``).

    Parameters
    ----------
    series_ids:
        List of FRED series identifiers to retrieve.
    vintage_date:
        Vintage month in "YYYY-MM" format. The API is queried with
        ``realtime_start={vintage_date}-01`` and
        ``realtime_end={vintage_date}-28`` to capture the final revision
        published within that calendar month.
    api_key:
        FRED API key for the stlouisfed.org REST API.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with ``pd.DatetimeIndex`` and one column
        per ``series_id``.
    """
    # Lazy import: requests is only needed in API mode.
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise NotImplementedError(
            "install macroforecast[alfred] for ALFRED API mode"
        ) from exc

    realtime_start = f"{vintage_date}-01"
    realtime_end = f"{vintage_date}-28"
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    series_frames: dict[str, pd.Series] = {}
    for series_id in series_ids:
        params = {
            "series_id": series_id,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
            "api_key": api_key,
            "file_type": "json",
        }
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        observations = response.json().get("observations", [])
        rows: dict[str, float] = {}
        for obs in observations:
            raw_value = obs.get("value", ".")
            # FRED uses "." as the missing-value sentinel.
            if raw_value == ".":
                continue
            try:
                rows[obs["date"]] = float(raw_value)
            except (ValueError, KeyError):
                continue
        series_frames[series_id] = pd.Series(rows, name=series_id)

    if not series_frames:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    wide = pd.DataFrame(series_frames)
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "observation_date"
    return wide


def apply_alfred_vintage_to_panel(
    panel_frame: pd.DataFrame,
    resolved: dict[str, Any],
    leaf_config: dict[str, Any],
) -> pd.DataFrame:
    """Apply ALFRED vintage correction to the raw panel frame.

    Guard: if ``vintage_policy`` is not ``"real_time_alfred"``, returns
    ``panel_frame`` unchanged.

    Two operating modes:

    **Static mode** (``alfred_vintage_date`` is set in ``leaf_config``):
    The snapshot is loaded once for the specified vintage month and all
    panel columns that appear in the ALFRED snapshot are replaced with
    the vintage-correct values.

    **Rolling mode** (``alfred_vintage_date`` is ``None``):
    For each observation date ``d`` in ``panel_frame.index``, the most
    recent ALFRED vintage available at ``d`` (i.e. vintage ≤
    ``d.strftime("%Y-%m")``) is used to fill that row. The full snapshot
    is loaded upfront for efficiency; a merge on ``(observation_date,
    vintage_date_cutoff)`` drives the row-by-row vintage selection.

    Columns not found in the ALFRED snapshot retain their original values.

    Parameters
    ----------
    panel_frame:
        Raw panel DataFrame (output of the L1 data loading step).
    resolved:
        Resolved axis values for the recipe (contains ``vintage_policy``).
    leaf_config:
        Leaf-level configuration dict (may contain ``alfred_mode``,
        ``alfred_snapshot_dir``, ``alfred_vintage_date``,
        ``alfred_api_key``).

    Returns
    -------
    pd.DataFrame
        Panel frame with ALFRED vintage-correct values applied.
    """
    if resolved.get("vintage_policy") != "real_time_alfred":
        return panel_frame

    alfred_mode: str = leaf_config.get("alfred_mode", "local")
    alfred_vintage_date: str | None = leaf_config.get("alfred_vintage_date")
    series_ids: list[str] | None = list(panel_frame.columns) or None

    if alfred_mode == "local":
        snapshot_dir = leaf_config.get("alfred_snapshot_dir")
        if not snapshot_dir:
            # Soft validation already issued a warning in l1.py; skip silently.
            return panel_frame
        snapshot_path = Path(str(snapshot_dir))

        if alfred_vintage_date:
            # Static mode: one vintage for all rows.
            alfred_wide = load_alfred_vintage_snapshot(
                snapshot_path, series_ids, alfred_vintage_date
            )
            result = panel_frame.copy()
            # Only overwrite columns present in the ALFRED snapshot.
            common_cols = [c for c in panel_frame.columns if c in alfred_wide.columns]
            if common_cols:
                alfred_aligned = alfred_wide[common_cols].reindex(panel_frame.index)
                result[common_cols] = alfred_aligned
            return result
        else:
            # Rolling mode: each observation row uses its own vintage cutoff.
            # Load the full snapshot (no vintage_date cutoff) for efficiency.
            if not Path(snapshot_path).exists():
                return panel_frame
            full_snapshot = _read_snapshot_file(Path(snapshot_path))
            if series_ids:
                full_snapshot = full_snapshot[
                    full_snapshot["series_id"].isin(series_ids)
                ]
            result = panel_frame.copy()
            # Iterate over each observation date and apply the appropriate
            # vintage cutoff row by row.
            for obs_date in panel_frame.index:
                vintage_cutoff = obs_date.strftime("%Y-%m")
                row_df = full_snapshot[full_snapshot["vintage_date"] <= vintage_cutoff]
                if row_df.empty:
                    continue
                # Pick the most recent vintage per (series_id, observation_date).
                row_df = (
                    row_df.sort_values("vintage_date")
                    .groupby(["series_id", "observation_date"])
                    .last()
                    .reset_index()
                )
                # Find the value for this specific observation_date.
                obs_str = obs_date.strftime("%Y-%m-%d")
                row_match = row_df[
                    pd.to_datetime(row_df["observation_date"]).dt.strftime("%Y-%m-%d")
                    == obs_str
                ]
                if row_match.empty:
                    continue
                for _, record in row_match.iterrows():
                    col = record["series_id"]
                    if col in result.columns:
                        result.at[obs_date, col] = record["value"]
            return result

    elif alfred_mode == "api":
        api_key: str = leaf_config.get("alfred_api_key") or __import__("os").environ.get("FRED_API_KEY", "")
        if not api_key:
            return panel_frame
        series_list = list(panel_frame.columns) if series_ids is None else series_ids
        vintage = alfred_vintage_date or panel_frame.index[0].strftime("%Y-%m")
        alfred_wide = load_alfred_vintage_api(series_list, vintage, api_key)
        result = panel_frame.copy()
        common_cols = [c for c in panel_frame.columns if c in alfred_wide.columns]
        if common_cols:
            alfred_aligned = alfred_wide[common_cols].reindex(panel_frame.index)
            result[common_cols] = alfred_aligned
        return result

    # Unknown mode: return unchanged.
    return panel_frame
