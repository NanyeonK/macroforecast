"""Tests for macrocast.data.merge.merge_macro_frames."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from macrocast.data.merge import merge_macro_frames
from macrocast.data.schema import MacroFrame, MacroFrameMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_frame(
    n_periods: int,
    freq: str,
    columns: list[str],
    dataset: str,
    meta_freq: str,
) -> MacroFrame:
    """Create a minimal MacroFrame for testing."""
    dates = pd.date_range("2000-01", periods=n_periods, freq=freq)
    data = pd.DataFrame(
        np.arange(n_periods * len(columns), dtype=float).reshape(n_periods, -1),
        index=dates,
        columns=columns,
    )
    meta = MacroFrameMetadata(dataset=dataset, vintage=None, frequency=meta_freq)
    return MacroFrame(data, meta)


@pytest.fixture()
def mf_md() -> MacroFrame:
    return _make_frame(24, "ME", ["A", "B", "C"], "FRED-MD", "monthly")


@pytest.fixture()
def mf_qd() -> MacroFrame:
    return _make_frame(8, "QE", ["D", "E"], "FRED-QD", "quarterly")


@pytest.fixture()
def mf_sd() -> MacroFrame:
    return _make_frame(24, "ME", ["CA_UR", "TX_UR"], "FRED-SD", "state_monthly")


# ---------------------------------------------------------------------------
# Basic merges
# ---------------------------------------------------------------------------


class TestMergeBasic:
    def test_single_frame_monthly(self, mf_md: MacroFrame) -> None:
        result = merge_macro_frames(mf_md, target_freq="ME")
        assert list(result.columns) == ["A", "B", "C"]
        assert len(result) == 24

    def test_two_monthly_frames(self, mf_md: MacroFrame, mf_sd: MacroFrame) -> None:
        result = merge_macro_frames(mf_md, mf_sd, target_freq="ME")
        assert set(result.columns) == {"A", "B", "C", "CA_UR", "TX_UR"}
        assert len(result) == 24

    def test_md_qd_monthly_target(self, mf_md: MacroFrame, mf_qd: MacroFrame) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = merge_macro_frames(mf_md, mf_qd, target_freq="ME")
        assert set(result.columns) == {"A", "B", "C", "D", "E"}
        # QD upsampled warning expected
        assert any("upsampled" in str(wi.message) for wi in w)

    def test_md_qd_quarterly_target(self, mf_md: MacroFrame, mf_qd: MacroFrame) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = merge_macro_frames(mf_md, mf_qd, target_freq="QE")
        assert set(result.columns) == {"A", "B", "C", "D", "E"}
        # MD downsampled warning expected
        assert any("downsampled" in str(wi.message) for wi in w)
        # Should have 8 quarter-end rows (inner join)
        assert len(result) == 8


# ---------------------------------------------------------------------------
# Frequency aliases
# ---------------------------------------------------------------------------


class TestFreqAliases:
    @pytest.mark.parametrize("alias", ["ME", "monthly", "m", "me"])
    def test_monthly_aliases(self, mf_md: MacroFrame, alias: str) -> None:
        result = merge_macro_frames(mf_md, target_freq=alias)
        assert len(result) > 0

    @pytest.mark.parametrize("alias", ["QE", "quarterly", "q", "qe"])
    def test_quarterly_aliases(self, mf_qd: MacroFrame, alias: str) -> None:
        result = merge_macro_frames(mf_qd, target_freq=alias)
        assert len(result) > 0

    def test_unknown_freq_raises(self, mf_md: MacroFrame) -> None:
        with pytest.raises(ValueError, match="Unknown target_freq"):
            merge_macro_frames(mf_md, target_freq="daily")


# ---------------------------------------------------------------------------
# Column conflict resolution
# ---------------------------------------------------------------------------


class TestColumnConflicts:
    def test_conflict_warns(self, mf_md: MacroFrame) -> None:
        """When same column exists in both frames, warn and keep primary."""
        mf_overlap = _make_frame(24, "ME", ["A", "X"], "FRED-SD", "state_monthly")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = merge_macro_frames(mf_md, mf_overlap, target_freq="ME")
        assert any("conflict" in str(wi.message).lower() for wi in w)
        # A should be present once only, from mf_md (first, primary)
        assert "A" in result.columns
        assert result.columns.tolist().count("A") == 1

    def test_primary_freq_wins_conflict(
        self, mf_md: MacroFrame, mf_qd: MacroFrame
    ) -> None:
        """Primary-freq (MD, monthly) wins over resampled QD on column conflict."""
        mf_qd_overlap = _make_frame(8, "QE", ["A", "D"], "FRED-QD", "quarterly")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = merge_macro_frames(mf_md, mf_qd_overlap, target_freq="ME")
        # A from MD (monthly=primary) should survive, not from QD
        assert "A" in result.columns
        assert result.columns.tolist().count("A") == 1


# ---------------------------------------------------------------------------
# Date alignment (inner join)
# ---------------------------------------------------------------------------


class TestDateAlignment:
    def test_inner_join_range(self) -> None:
        """Merged date range is the intersection."""
        mf1 = _make_frame(24, "ME", ["A"], "MD", "monthly")  # 2000-01 to 2001-12
        mf2 = _make_frame(24, "ME", ["B"], "MD2", "monthly")  # same range
        result = merge_macro_frames(mf1, mf2, target_freq="ME")
        assert len(result) == 24

    def test_no_rows_raises_gracefully(self) -> None:
        """Non-overlapping date ranges produce empty DataFrame."""
        mf1 = _make_frame(6, "ME", ["A"], "MD", "monthly")
        # mf2 starts after mf1 ends
        dates = pd.date_range("2010-01", periods=6, freq="ME")
        data = pd.DataFrame(np.ones((6, 1)), index=dates, columns=["B"])
        meta = MacroFrameMetadata(dataset="MD2", vintage=None, frequency="monthly")
        mf2 = MacroFrame(data, meta)
        result = merge_macro_frames(mf1, mf2, target_freq="ME")
        assert result.empty


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_frames_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            merge_macro_frames(target_freq="ME")

    def test_state_monthly_treated_as_monthly(self, mf_sd: MacroFrame) -> None:
        result = merge_macro_frames(mf_sd, target_freq="ME")
        assert len(result) == 24
