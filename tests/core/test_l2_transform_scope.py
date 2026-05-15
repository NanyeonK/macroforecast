"""F-P1-14 -- L2 official_transform_scope is now honoured.

Tests verify that each scope value applies tcodes correctly:
- target_and_predictors: all columns transformed (previous default)
- target_only: only target column gets tcode applied
- predictors_only: all except target get tcode applied
- not_applicable / none: no tcode applied

Hand-computed expected values: tcode=2 means first-difference (diff).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _apply_transform
from macroforecast.core.layers import l2 as l2_layer


def _make_resolved(transform_policy: str = "apply_official_tcode",
                   transform_scope: str = "target_and_predictors") -> l2_layer.L2ResolvedAxes:
    """Build a minimal resolved-axes dict for the transform step."""
    return {
        "transform_policy": transform_policy,
        "transform_scope": transform_scope,
        "outlier_policy": "none",
        "imputation_policy": "none_propagate",
        "imputation_temporal_rule": "full_sample_once",
        "frame_edge_policy": "keep_unbalanced",
        "sd_tcode_policy": "none",
    }


def _make_frame() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    return pd.DataFrame(
        {"y": [1.0, 2.0, 4.0, 7.0, 11.0],
         "x1": [10.0, 12.0, 14.0, 17.0, 21.0],
         "x2": [100.0, 105.0, 110.0, 116.0, 122.0]},
        index=idx,
    )


# tcode=2 = first difference
_TCODE_MAP = {"y": 2, "x1": 2, "x2": 2}
_L1_LEAF = {"target": "y", "official_tcode_map": _TCODE_MAP}
_L2_LEAF = {}
_LOG: dict = {"steps": []}


def _fresh_log():
    return {"steps": []}


class TestTransformScopeTargetAndPredictors:
    def test_all_columns_transformed(self):
        frame = _make_frame()
        resolved = _make_resolved(transform_scope="target_and_predictors")
        result, applied = _apply_transform(frame, resolved, _L2_LEAF, _L1_LEAF, _fresh_log())
        # All three columns in applied
        assert set(applied.keys()) == {"y", "x1", "x2"}
        # First row becomes NaN (diff of first element)
        assert np.isnan(result["y"].iloc[0])
        assert np.isnan(result["x1"].iloc[0])


class TestTransformScopeTargetOnly:
    def test_only_target_transformed(self):
        frame = _make_frame()
        resolved = _make_resolved(transform_scope="target_only")
        result, applied = _apply_transform(frame, resolved, _L2_LEAF, _L1_LEAF, _fresh_log())
        # Only target in applied
        assert "y" in applied
        assert "x1" not in applied
        assert "x2" not in applied
        # Target is differenced: row 0 = NaN, row 1 = 2.0-1.0 = 1.0
        assert np.isnan(result["y"].iloc[0])
        assert result["y"].iloc[1] == pytest.approx(1.0)
        # Predictors unchanged
        assert result["x1"].iloc[0] == pytest.approx(10.0)
        assert result["x2"].iloc[0] == pytest.approx(100.0)

    def test_predictor_values_equal_original(self):
        frame = _make_frame()
        original = frame.copy()
        resolved = _make_resolved(transform_scope="target_only")
        result, _ = _apply_transform(frame, resolved, _L2_LEAF, _L1_LEAF, _fresh_log())
        pd.testing.assert_series_equal(result["x1"], original["x1"])
        pd.testing.assert_series_equal(result["x2"], original["x2"])


class TestTransformScopePredictorsOnly:
    def test_only_predictors_transformed(self):
        frame = _make_frame()
        resolved = _make_resolved(transform_scope="predictors_only")
        result, applied = _apply_transform(frame, resolved, _L2_LEAF, _L1_LEAF, _fresh_log())
        # Only predictors in applied
        assert "x1" in applied
        assert "x2" in applied
        assert "y" not in applied
        # Predictors differenced
        assert np.isnan(result["x1"].iloc[0])
        assert result["x1"].iloc[1] == pytest.approx(2.0)
        # Target unchanged
        assert result["y"].iloc[0] == pytest.approx(1.0)

    def test_target_value_equal_original(self):
        frame = _make_frame()
        original = frame.copy()
        resolved = _make_resolved(transform_scope="predictors_only")
        result, _ = _apply_transform(frame, resolved, _L2_LEAF, _L1_LEAF, _fresh_log())
        pd.testing.assert_series_equal(result["y"], original["y"])


class TestTransformScopeNotApplicable:
    def test_no_transform_applied(self):
        frame = _make_frame()
        original = frame.copy()
        resolved = _make_resolved(transform_scope="not_applicable")
        result, applied = _apply_transform(frame, resolved, _L2_LEAF, _L1_LEAF, _fresh_log())
        assert applied == {}
        pd.testing.assert_frame_equal(result, original)


class TestTransformScopeNone:
    def test_no_transform_when_scope_none(self):
        """scope='none' (alias) returns unchanged frame with empty applied."""
        frame = _make_frame()
        original = frame.copy()
        resolved = _make_resolved(transform_scope="none")
        result, applied = _apply_transform(frame, resolved, _L2_LEAF, _L1_LEAF, _fresh_log())
        assert applied == {}
        pd.testing.assert_frame_equal(result, original)
