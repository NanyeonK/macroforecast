"""Tests for PR6: linear_interpolation silent lookahead leak fix.

Three locations in runtime.py contained lookahead leaks via pandas interpolate:
- Location 1: _apply_fred_sd_frequency_alignment — bidirectional interpolation on
  the full panel before any per-origin split (must raise ValueError).
- Location 2: _apply_imputation_per_origin — interpolation ignores cutoff_ts,
  leaking future values into pre-cutoff NaNs.
- Location 3: _apply_imputation full-sample — interpolation uses default
  limit_direction which may be bidirectional; must emit UserWarning and use
  forward-only direction.

These tests are written TDD-style: they were authored to FAIL before the fix
and to PASS after. Do not modify these assertions.
"""
from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import (
    _apply_fred_sd_frequency_alignment,
    _apply_imputation,
    _apply_imputation_per_origin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_resolved(policy: str) -> dict[str, Any]:
    """Return a minimal resolved L2 axes dict for the given imputation policy."""
    return {"imputation_policy": policy}


def _make_l1_artifact(target_freq: str = "monthly") -> Any:
    """Return a mock L1DataDefinitionArtifact with required attributes."""
    artifact = MagicMock()
    artifact.frequency = target_freq
    # Provide a quarterly series labelled "q_col" and a monthly "m_col"
    artifact.raw_panel.metadata.values = {
        "series_frequency": {"q_col": "quarterly", "m_col": "monthly"}
    }
    return artifact


def _make_mixed_freq_df(n_months: int = 12) -> pd.DataFrame:
    """Return a mixed-frequency DataFrame: monthly index with quarterly values
    on 'q_col' (non-quarter-end months are NaN) and monthly values on 'm_col'."""
    idx = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    data: dict[str, list[float | None]] = {"m_col": list(range(n_months))}
    q_vals: list[float | None] = []
    for i in range(n_months):
        # Quarterly series: value only at month 3, 6, 9, 12 (quarter-end)
        month = idx[i].month
        if month in (3, 6, 9, 12):
            q_vals.append(float(i))
        else:
            q_vals.append(None)
    data["q_col"] = q_vals
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Test 1 — Location 2: _apply_imputation_per_origin cutoff-respect
# ---------------------------------------------------------------------------

class TestPerOriginCutoffRespect:
    """_apply_imputation_per_origin must not fill t > cutoff_ts."""

    def _make_frame(self) -> tuple[pd.DataFrame, Any]:
        """10-row integer-index DataFrame.

        Col 'a': [NaN, 1.0, NaN, 3.0, NaN, NaN, NaN, NaN, NaN, NaN]
        cutoff at index position 3 (inclusive).
        """
        idx = list(range(10))
        a = [np.nan, 1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df = pd.DataFrame({"a": a}, index=idx)
        cutoff_ts = 3
        return df, cutoff_ts

    def test_interpolates_within_window(self) -> None:
        """Row 2 (between known t=1 and t=3) must be filled: (1+3)/2 = 2.0."""
        df, cutoff_ts = self._make_frame()
        resolved = _make_resolved("linear_interpolation")
        result = _apply_imputation_per_origin(df, resolved, cutoff_ts)
        assert result.iloc[2]["a"] == pytest.approx(2.0), (
            f"Expected 2.0 at t=2 (interpolated between t=1=1.0 and t=3=3.0), "
            f"got {result.iloc[2]['a']}"
        )

    def test_does_not_fill_beyond_cutoff(self) -> None:
        """Rows 4..9 are all NaN and beyond cutoff — must remain NaN."""
        df, cutoff_ts = self._make_frame()
        resolved = _make_resolved("linear_interpolation")
        result = _apply_imputation_per_origin(df, resolved, cutoff_ts)
        for row_idx in [4, 5, 6, 7, 8, 9]:
            val = result.iloc[row_idx]["a"]
            assert pd.isna(val), (
                f"Expected NaN at t={row_idx} (beyond cutoff_ts=3), "
                f"got {val} — lookahead leak detected"
            )

    def test_not_influenced_by_future_values(self) -> None:
        """Critical leak test: changing t>cutoff values must NOT change result at t<=cutoff.

        Construct two frames identical up to cutoff=3 but with different values at t>3.
        After per-origin imputation, result at t<=3 must be identical in both frames.
        """
        resolved = _make_resolved("linear_interpolation")

        # Scenario A: t>3 values are large
        a_vals_A = [np.nan, 1.0, np.nan, 3.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        df_A = pd.DataFrame({"a": a_vals_A}, index=list(range(10)))

        # Scenario B: t>3 values are small
        a_vals_B = [np.nan, 1.0, np.nan, 3.0, -100.0, -200.0, -300.0, -400.0, -500.0, -600.0]
        df_B = pd.DataFrame({"a": a_vals_B}, index=list(range(10)))

        result_A = _apply_imputation_per_origin(df_A, resolved, cutoff_ts=3)
        result_B = _apply_imputation_per_origin(df_B, resolved, cutoff_ts=3)

        # t=0..3 must be identical regardless of t>3
        for t in range(4):
            val_A = result_A.iloc[t]["a"]
            val_B = result_B.iloc[t]["a"]
            if pd.isna(val_A) and pd.isna(val_B):
                continue
            assert val_A == pytest.approx(val_B), (
                f"Lookahead leak at t={t}: result differs when t>cutoff values differ. "
                f"Scenario A={val_A}, Scenario B={val_B}"
            )

    def test_cutoff_none_forward_only(self) -> None:
        """When cutoff_ts is None, interpolation must be forward-only (no leading NaN fill)."""
        # Leading NaN followed by known values; forward-only must NOT fill the leading NaN
        df = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 3.0]}, index=list(range(4)))
        resolved = _make_resolved("linear_interpolation")
        result = _apply_imputation_per_origin(df, resolved, cutoff_ts=None)
        # Row 0 has no prior value — must remain NaN (leading NaN preserved)
        assert pd.isna(result.iloc[0]["a"]), (
            "Leading NaN at t=0 must not be filled when cutoff_ts is None (forward-only)"
        )
        # Row 2 is between t=1 (1.0) and t=3 (3.0) — forward-only fills it
        assert result.iloc[2]["a"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 2 — Location 1: FRED-SD alignment raises ValueError
# ---------------------------------------------------------------------------

class TestFredSdLinearInterpolationReject:
    """_apply_fred_sd_frequency_alignment must raise ValueError for linear_interpolation."""

    def test_raises_value_error_for_linear_interpolation(self) -> None:
        """linear_interpolation with FRED-SD causes lookahead; must be rejected."""
        df = _make_mixed_freq_df(n_months=12)
        resolved = {
            "quarterly_to_monthly_policy": "linear_interpolation",
            "monthly_to_quarterly_policy": "quarterly_average",
            "sd_series_frequency_filter": "both",
            "fred_sd_frequency_policy": "report_only",
        }
        l1_artifact = _make_l1_artifact(target_freq="monthly")
        cleaning_log: dict[str, Any] = {}

        with pytest.raises(ValueError, match="lookahead|future data|bidirectional"):
            _apply_fred_sd_frequency_alignment(df, resolved, l1_artifact, cleaning_log)

    def test_step_forward_is_unaffected(self) -> None:
        """step_forward is causal; must NOT raise."""
        df = _make_mixed_freq_df(n_months=12)
        resolved = {
            "quarterly_to_monthly_policy": "step_forward",
            "monthly_to_quarterly_policy": "quarterly_average",
            "sd_series_frequency_filter": "both",
            "fred_sd_frequency_policy": "report_only",
        }
        l1_artifact = _make_l1_artifact(target_freq="monthly")
        cleaning_log: dict[str, Any] = {}

        # Must not raise
        result = _apply_fred_sd_frequency_alignment(df, resolved, l1_artifact, cleaning_log)
        assert isinstance(result, pd.DataFrame)

    def test_step_backward_is_unaffected(self) -> None:
        """step_backward (bfill+ffill) is not lookahead; must NOT raise."""
        df = _make_mixed_freq_df(n_months=12)
        resolved = {
            "quarterly_to_monthly_policy": "step_backward",
            "monthly_to_quarterly_policy": "quarterly_average",
            "sd_series_frequency_filter": "both",
            "fred_sd_frequency_policy": "report_only",
        }
        l1_artifact = _make_l1_artifact(target_freq="monthly")
        cleaning_log: dict[str, Any] = {}

        result = _apply_fred_sd_frequency_alignment(df, resolved, l1_artifact, cleaning_log)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test 3 — Location 3: _apply_imputation full-sample warning + forward-only
# ---------------------------------------------------------------------------

class TestFullSampleImputationWarning:
    """_apply_imputation must emit UserWarning and use forward-only interpolation."""

    def _make_frame_with_leading_nan(self) -> pd.DataFrame:
        """DataFrame where:
        - Row 0: NaN (leading — should remain NaN after forward-only interpolation)
        - Row 1: 1.0
        - Row 2: NaN (between 1.0 and 3.0 — forward-fill fills to 1.0; linear fills to 2.0)
        - Row 3: 3.0
        - Row 4: NaN (trailing — no future value; stays NaN)
        """
        return pd.DataFrame(
            {"a": [np.nan, 1.0, np.nan, 3.0, np.nan]},
            index=list(range(5)),
        )

    def test_user_warning_is_emitted(self) -> None:
        """Calling _apply_imputation with linear_interpolation must emit UserWarning."""
        df = self._make_frame_with_leading_nan()
        resolved = _make_resolved("linear_interpolation")
        cleaning_log: dict[str, Any] = {"steps": []}

        with pytest.warns(UserWarning, match="linear_interpolation|full.sample|forward"):
            _apply_imputation(df, resolved, cleaning_log)

    def test_forward_fill_only_no_leading_nan_fill(self) -> None:
        """Leading NaN at row 0 must NOT be filled (forward-only: no prior value)."""
        df = self._make_frame_with_leading_nan()
        resolved = _make_resolved("linear_interpolation")
        cleaning_log: dict[str, Any] = {"steps": []}

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result, filled = _apply_imputation(df, resolved, cleaning_log)

        # Row 0 has no prior non-NaN — must remain NaN
        assert pd.isna(result.iloc[0]["a"]), (
            f"Leading NaN at row 0 must remain NaN after forward-only interpolation; "
            f"got {result.iloc[0]['a']}"
        )

    def test_inter_value_nan_is_filled(self) -> None:
        """Row 2 (between known 1.0 and 3.0) must be filled by interpolation."""
        df = self._make_frame_with_leading_nan()
        resolved = _make_resolved("linear_interpolation")
        cleaning_log: dict[str, Any] = {"steps": []}

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result, filled = _apply_imputation(df, resolved, cleaning_log)

        # Row 2: linear interpolation between 1.0 (row1) and 3.0 (row3) = 2.0
        assert result.iloc[2]["a"] == pytest.approx(2.0), (
            f"Expected 2.0 at row 2, got {result.iloc[2]['a']}"
        )

    def test_no_backward_fill_from_future_to_leading_nan(self) -> None:
        """The leading NaN at row 0 must NOT be filled by backward extrapolation.

        With bidirectional interpolation (pre-fix default), pandas would fill
        the leading NaN at row 0 using the value at row 1 (backward fill).
        Forward-only must prevent this. Trailing NaN extrapolation (row 4 → 3.0)
        is forward-only and acceptable; it does not use future information.
        """
        df = self._make_frame_with_leading_nan()
        resolved = _make_resolved("linear_interpolation")
        cleaning_log: dict[str, Any] = {"steps": []}

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result, filled = _apply_imputation(df, resolved, cleaning_log)

        # Row 0 is leading NaN with no prior value — forward-only cannot fill it.
        # (Backward fill would produce 1.0; that is the lookahead we prevent.)
        assert pd.isna(result.iloc[0]["a"]), (
            f"Leading NaN at row 0 must not be backward-filled; got {result.iloc[0]['a']}"
        )

    def test_non_linear_policy_unaffected(self) -> None:
        """Other imputation policies (e.g. forward_fill) must NOT emit UserWarning."""
        df = self._make_frame_with_leading_nan()
        resolved = _make_resolved("forward_fill")
        cleaning_log: dict[str, Any] = {"steps": []}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, filled = _apply_imputation(df, resolved, cleaning_log)

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, (
            f"forward_fill must not emit UserWarning; got {user_warnings}"
        )
