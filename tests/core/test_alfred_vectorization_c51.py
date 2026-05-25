"""test(c51): ALFRED rolling vintage vectorization — Flag-C regression guard.

Verifies that the vectorized merge-based rolling vintage resolution in
apply_alfred_vintage_to_panel produces bit-exact results against a Python
loop reference defined inline in this test module.

The reference implementation is a copy of the original loop logic as it
existed before the vectorization refactor. The test must confirm that
np.allclose(vectorized, reference, atol=1e-10) on a deterministic seed-99
fixture (test-spec.md TC-FlagC-1).

Additional tests:
- TC-FlagC-2: empty snapshot returns panel_frame unchanged.
- TC-FlagC-3: non-real_time_alfred policy returns panel_frame unchanged.

Pipeline isolation: this module defines its own reference logic inline
and NEVER imports from implementation details or reads test-spec.md.

Scope: pytest -m "not slow and not heavy and not deep"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Reference implementation (inline copy of original loop logic).
# This replicates what the pre-vectorization code did, as described in
# test-spec.md TC-FlagC-1. It is NOT imported from production code.
# ---------------------------------------------------------------------------

def _rolling_loop_reference(
    panel_frame: pd.DataFrame,
    full_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Python loop reference for rolling ALFRED vintage resolution.

    Replicates the original O(N_obs x snapshot_rows) loop that was replaced
    by the vectorized merge. Used only as a test oracle.

    For each observation date in panel_frame.index:
      - Cut the snapshot to vintages <= that obs_date's "YYYY-MM".
      - Within the cut snapshot, for each (series_id, observation_date)
        keep only the most recently published vintage.
      - Write that value into result at the matching obs_date x series_id.
    """
    result = panel_frame.copy()
    for obs_date in panel_frame.index:
        vintage_cutoff = obs_date.strftime("%Y-%m")
        row_df = full_snapshot[full_snapshot["vintage_date"] <= vintage_cutoff]
        if row_df.empty:
            continue
        row_df = (
            row_df.sort_values("vintage_date")
            .groupby(["series_id", "observation_date"])
            .last()
            .reset_index()
        )
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


# ---------------------------------------------------------------------------
# Deterministic fixture (seed 99, from test-spec.md TC-FlagC-1)
# ---------------------------------------------------------------------------

def _make_fixture_seed99():
    """Return (snapshot_df, panel_df) for seed 99."""
    rng = np.random.default_rng(99)

    # Simulate a small ALFRED snapshot (long format)
    obs_dates = pd.date_range("2010-01-01", periods=12, freq="MS")  # 12 months
    series_ids = ["A", "B"]
    vintage_dates = ["2010-01", "2010-02", "2010-03"]  # 3 vintages

    rows = []
    for sid in series_ids:
        for obs in obs_dates:
            for vd in vintage_dates:
                if vd <= obs.strftime("%Y-%m"):
                    rows.append({
                        "series_id": sid,
                        "observation_date": obs.strftime("%Y-%m-%d"),
                        "vintage_date": vd,
                        "value": float(rng.standard_normal(1)[0]),
                    })
    snapshot_df = pd.DataFrame(rows)

    # Panel frame: 12 monthly rows, 2 columns
    panel_df = pd.DataFrame(
        rng.standard_normal((12, 2)),
        index=obs_dates,
        columns=["A", "B"],
    )

    return snapshot_df, panel_df


# ---------------------------------------------------------------------------
# Helper: call the vectorized rolling path with an in-memory snapshot.
# We patch _read_snapshot_file to bypass the disk I/O and inject our fixture.
# ---------------------------------------------------------------------------

def _call_vectorized_rolling(
    panel_frame: pd.DataFrame,
    full_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Invoke apply_alfred_vintage_to_panel rolling path with an in-memory snapshot.

    Patches _read_snapshot_file to return full_snapshot so no disk I/O occurs.
    Uses a sentinel snapshot_path so the path-existence guard passes.
    """
    from macroforecast.layers.l1_data.alfred_adapter import apply_alfred_vintage_to_panel

    resolved = {"vintage_policy": "real_time_alfred"}
    # Use a sentinel path; we will patch _read_snapshot_file to return our df.
    leaf_config = {
        "alfred_mode": "local",
        "alfred_snapshot_dir": "/tmp/alfred_test_sentinel",
        # No alfred_vintage_date -> rolling mode
    }

    with patch("macroforecast.layers.l1_data.alfred_adapter._read_snapshot_file",
               return_value=full_snapshot), \
         patch("macroforecast.layers.l1_data.alfred_adapter.Path") as mock_path_cls:
        # Make Path(...).exists() return True so the guard passes
        mock_path_instance = mock_path_cls.return_value
        mock_path_instance.exists.return_value = True
        # __str__ must return our sentinel so Path(str(snapshot_path)) works
        mock_path_instance.__str__ = lambda self: "/tmp/alfred_test_sentinel"
        result = apply_alfred_vintage_to_panel(panel_frame, resolved, leaf_config)

    return result


# ---------------------------------------------------------------------------
# TC-FlagC-1: bit-exact comparison (vectorized vs loop reference, seed 99)
# ---------------------------------------------------------------------------

def test_alfred_rolling_vectorized_matches_loop_reference() -> None:
    """Vectorized rolling ALFRED vintage must match loop reference bit-for-bit.

    Tolerance: atol=1e-10 (test-spec.md TC-FlagC-1).
    """
    snapshot_df, panel_df = _make_fixture_seed99()

    # Reference: inline loop
    reference_result = _rolling_loop_reference(panel_df, snapshot_df)

    # Vectorized: production code path
    vectorized_result = _call_vectorized_rolling(panel_df, snapshot_df)

    # Assertion 1: return type
    assert isinstance(vectorized_result, pd.DataFrame), (
        f"Expected pd.DataFrame, got {type(vectorized_result)}"
    )

    # Assertion 2: index matches
    pd.testing.assert_index_equal(vectorized_result.index, panel_df.index), None

    # Assertion 3: columns match
    assert list(vectorized_result.columns) == list(panel_df.columns), (
        f"Expected columns {list(panel_df.columns)}, got {list(vectorized_result.columns)}"
    )

    # Assertion 4: values are bit-exact (atol=1e-10)
    max_diff = float(np.max(np.abs(vectorized_result.values - reference_result.values)))
    assert np.allclose(vectorized_result.values, reference_result.values, atol=1e-10), (
        f"Vectorized result does not match loop reference.\n"
        f"Max abs diff: {max_diff:.6e}\n"
        f"Tolerance (from test-spec.md): atol=1e-10\n"
        f"Vectorized:\n{vectorized_result}\n"
        f"Reference:\n{reference_result}"
    )


# ---------------------------------------------------------------------------
# TC-FlagC-2: empty snapshot returns panel_frame unchanged
# ---------------------------------------------------------------------------

def test_alfred_rolling_empty_snapshot_unchanged() -> None:
    """Empty snapshot must return panel_frame copy unchanged (TC-FlagC-2)."""
    _, panel_df = _make_fixture_seed99()
    empty_snapshot = pd.DataFrame(
        columns=["series_id", "observation_date", "vintage_date", "value"]
    )

    result = _call_vectorized_rolling(panel_df, empty_snapshot)

    assert isinstance(result, pd.DataFrame)
    # Values should be unchanged from panel_df
    np.testing.assert_array_equal(result.values, panel_df.values)


# ---------------------------------------------------------------------------
# TC-FlagC-3: non-real_time_alfred vintage_policy returns panel_frame unchanged
# ---------------------------------------------------------------------------

def test_alfred_non_real_time_policy_guard() -> None:
    """Guard: resolved vintage_policy != 'real_time_alfred' returns unchanged (TC-FlagC-3)."""
    from macroforecast.layers.l1_data.alfred_adapter import apply_alfred_vintage_to_panel

    _, panel_df = _make_fixture_seed99()

    resolved = {"vintage_policy": "current_vintage"}
    leaf_config = {"alfred_mode": "local"}

    result = apply_alfred_vintage_to_panel(panel_df, resolved, leaf_config)

    # Must be the same object or equal values (guard fires at line 242)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, panel_df)
