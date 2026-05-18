"""Tests for Cycle 35 L4 tree/ensemble family standalone callables.

Each test class covers one of the 6 new callables in ``mf.functions``.
Bit-exact assertions compare against the ``_build_l4_model`` recipe path
(which IS the canonical computation, since both paths use the same
estimator with identical parameters).

Protocol conformance is verified via ``isinstance(r, FitResultBase)``
(requires ``@runtime_checkable`` on the Protocol).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.functions import FitResultBase
from macroforecast.core.runtime import _build_l4_model


# ---------------------------------------------------------------------------
# Shared deterministic fixtures
# ---------------------------------------------------------------------------

def _make_xy_rng42(n: int = 100, p: int = 5):
    """RNG-42 panel: X ~ N(0,1), y = X @ [1,2,3,4,5] + 0.5*noise."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    beta = np.arange(1, p + 1, dtype=float)
    y = X @ beta + 0.5 * rng.randn(n)
    return X, y


@pytest.fixture(scope="module")
def xy_rng42():
    return _make_xy_rng42()


# ---------------------------------------------------------------------------
# Helper: recipe-path prediction extraction
# ---------------------------------------------------------------------------

def _recipe_predict(family: str, params: dict, X_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Build + fit recipe model; return predictions on X_arr."""
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr.ravel(), name="y")
    model = _build_l4_model(family, params)
    model.fit(X, y)
    return np.asarray(model.predict(X), dtype=float).ravel()


# ---------------------------------------------------------------------------
# TestRandomForestFit
# ---------------------------------------------------------------------------

class TestRandomForestFit:
    """random_forest_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.random_forest_fit(X, y)
        assert hasattr(r, "feature_importances_")
        assert r.feature_importances_.shape == (5,)
        assert r.n_estimators_used == 200

    def test_feature_importances_sum_to_one(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.random_forest_fit(X, y)
        np.testing.assert_allclose(r.feature_importances_.sum(), 1.0, atol=1e-10)

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        # Use small n_estimators for speed
        r = mf.functions.random_forest_fit(X_arr, y_arr, n_estimators=10, random_state=7)
        preds_ref = _recipe_predict(
            "random_forest", {"n_estimators": 10, "random_state": 7}, X_arr, y_arr
        )
        np.testing.assert_allclose(r.predict(X_arr), preds_ref, rtol=1e-12)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.random_forest_fit(X, y, n_estimators=5)
        preds = r.predict(X)
        assert preds.shape == (100,)
        assert preds.dtype == float

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.random_forest_fit(X_arr, y_arr, n_estimators=5)
        X_df = pd.DataFrame(X_arr)
        preds = r.predict(X_df)
        assert preds.shape == (100,)

    def test_summary_contains_family_name(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.random_forest_fit(X, y, n_estimators=5)
        s = r.summary()
        assert "RandomForest" in s
        assert "n_estimators" in s

    def test_summary_top3_features(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.random_forest_fit(X, y, n_estimators=5)
        s = r.summary()
        # top-3 feature names x0..x4 should appear
        found = sum(f"x{i}" in s for i in range(5))
        assert found >= 3

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.random_forest_fit(X, y, n_estimators=5)
        assert isinstance(r, FitResultBase)

    def test_validation_n_estimators(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_estimators"):
            mf.functions.random_forest_fit(X, y, n_estimators=0)

    def test_validation_min_samples_leaf(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="min_samples_leaf"):
            mf.functions.random_forest_fit(X, y, min_samples_leaf=0)

    def test_namespace_wiring(self):
        assert "random_forest_fit" in mf.functions.__all__
        assert "RandomForestFitResult" in mf.functions.__all__


# ---------------------------------------------------------------------------
# TestExtraTreesFit
# ---------------------------------------------------------------------------

class TestExtraTreesFit:
    """extra_trees_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.extra_trees_fit(X, y)
        assert hasattr(r, "feature_importances_")
        assert r.feature_importances_.shape == (5,)
        assert r.n_estimators_used == 200

    def test_feature_importances_sum_to_one(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.extra_trees_fit(X, y)
        np.testing.assert_allclose(r.feature_importances_.sum(), 1.0, atol=1e-10)

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.extra_trees_fit(X_arr, y_arr, n_estimators=10, random_state=7)
        preds_ref = _recipe_predict(
            "extra_trees", {"n_estimators": 10, "random_state": 7}, X_arr, y_arr
        )
        np.testing.assert_allclose(r.predict(X_arr), preds_ref, rtol=1e-12)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.extra_trees_fit(X, y, n_estimators=5)
        preds = r.predict(X)
        assert preds.shape == (100,)

    def test_summary_contains_family_name(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.extra_trees_fit(X, y, n_estimators=5)
        s = r.summary()
        assert "ExtraTrees" in s
        assert "n_estimators" in s

    def test_summary_top3_features(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.extra_trees_fit(X, y, n_estimators=5)
        s = r.summary()
        found = sum(f"x{i}" in s for i in range(5))
        assert found >= 3

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.extra_trees_fit(X, y, n_estimators=5)
        assert isinstance(r, FitResultBase)

    def test_validation_n_estimators(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_estimators"):
            mf.functions.extra_trees_fit(X, y, n_estimators=0)

    def test_namespace_wiring(self):
        assert "extra_trees_fit" in mf.functions.__all__
        assert "ExtraTreesFitResult" in mf.functions.__all__


# ---------------------------------------------------------------------------
# TestGradientBoostingFit
# ---------------------------------------------------------------------------

class TestGradientBoostingFit:
    """gradient_boosting_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gradient_boosting_fit(X, y)
        assert hasattr(r, "feature_importances_")
        assert r.feature_importances_.shape == (5,)
        assert r.n_estimators_used == 200

    def test_feature_importances_sum_to_one(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gradient_boosting_fit(X, y, n_estimators=10)
        np.testing.assert_allclose(r.feature_importances_.sum(), 1.0, atol=1e-10)

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.gradient_boosting_fit(
            X_arr, y_arr, n_estimators=20, learning_rate=0.1, max_depth=3, random_state=0
        )
        preds_ref = _recipe_predict(
            "gradient_boosting",
            {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3, "random_state": 0},
            X_arr, y_arr,
        )
        np.testing.assert_allclose(r.predict(X_arr), preds_ref, rtol=1e-12)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gradient_boosting_fit(X, y, n_estimators=5)
        preds = r.predict(X)
        assert preds.shape == (100,)

    def test_summary_contains_family_name(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gradient_boosting_fit(X, y, n_estimators=5)
        s = r.summary()
        assert "GradientBoosting" in s
        assert "n_estimators" in s

    def test_summary_top3_features(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gradient_boosting_fit(X, y, n_estimators=20)
        s = r.summary()
        found = sum(f"x{i}" in s for i in range(5))
        assert found >= 3

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gradient_boosting_fit(X, y, n_estimators=5)
        assert isinstance(r, FitResultBase)

    def test_validation_n_estimators(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_estimators"):
            mf.functions.gradient_boosting_fit(X, y, n_estimators=0)

    def test_validation_learning_rate(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="learning_rate"):
            mf.functions.gradient_boosting_fit(X, y, learning_rate=0.0)

    def test_validation_max_depth(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="max_depth"):
            mf.functions.gradient_boosting_fit(X, y, max_depth=0)

    def test_namespace_wiring(self):
        assert "gradient_boosting_fit" in mf.functions.__all__
        assert "GradientBoostingFitResult" in mf.functions.__all__


# ---------------------------------------------------------------------------
# TestXGBoostFit
# ---------------------------------------------------------------------------

class TestXGBoostFit:
    """xgboost_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.xgboost_fit(X, y)
        assert hasattr(r, "feature_importances_")
        assert r.feature_importances_.shape == (5,)
        assert r.n_estimators_used == 300

    def test_feature_importances_nonneg(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.xgboost_fit(X, y, n_estimators=10)
        assert np.all(r.feature_importances_ >= 0)

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.xgboost_fit(
            X_arr, y_arr, n_estimators=10, learning_rate=0.1, max_depth=6, random_state=0
        )
        preds_ref = _recipe_predict(
            "xgboost",
            {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 6, "random_state": 0},
            X_arr, y_arr,
        )
        np.testing.assert_allclose(r.predict(X_arr), preds_ref, rtol=1e-6, atol=1e-6)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.xgboost_fit(X, y, n_estimators=5)
        preds = r.predict(X)
        assert preds.shape == (100,)

    def test_summary_contains_family_name(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.xgboost_fit(X, y, n_estimators=5)
        s = r.summary()
        assert "XGBoost" in s
        assert "n_estimators" in s

    def test_summary_top3_features(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.xgboost_fit(X, y, n_estimators=20)
        s = r.summary()
        found = sum(f"x{i}" in s for i in range(5))
        assert found >= 3

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.xgboost_fit(X, y, n_estimators=5)
        assert isinstance(r, FitResultBase)

    def test_validation_subsample(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="subsample"):
            mf.functions.xgboost_fit(X, y, subsample=0.0)
        with pytest.raises(ValueError, match="subsample"):
            mf.functions.xgboost_fit(X, y, subsample=1.1)

    def test_validation_n_estimators(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_estimators"):
            mf.functions.xgboost_fit(X, y, n_estimators=0)

    def test_namespace_wiring(self):
        assert "xgboost_fit" in mf.functions.__all__
        assert "XGBoostFitResult" in mf.functions.__all__


# ---------------------------------------------------------------------------
# TestLightGBMFit
# ---------------------------------------------------------------------------

class TestLightGBMFit:
    """lightgbm_fit: correctness, predict, summary, protocol, validation, max_depth edge cases."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lightgbm_fit(X, y)
        assert hasattr(r, "feature_importances_")
        assert r.feature_importances_.shape == (5,)
        assert r.n_estimators_used == 300

    def test_feature_importances_nonneg(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lightgbm_fit(X, y, n_estimators=10)
        assert np.all(r.feature_importances_ >= 0)

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.lightgbm_fit(
            X_arr, y_arr, n_estimators=10, learning_rate=0.1, max_depth=-1, random_state=0
        )
        preds_ref = _recipe_predict(
            "lightgbm",
            {"n_estimators": 10, "learning_rate": 0.1, "max_depth": -1, "random_state": 0},
            X_arr, y_arr,
        )
        np.testing.assert_allclose(r.predict(X_arr), preds_ref, rtol=1e-6, atol=1e-6)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lightgbm_fit(X, y, n_estimators=5)
        preds = r.predict(X)
        assert preds.shape == (100,)

    def test_summary_contains_family_name(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lightgbm_fit(X, y, n_estimators=5)
        s = r.summary()
        assert "LightGBM" in s
        assert "n_estimators" in s

    def test_summary_top3_features(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lightgbm_fit(X, y, n_estimators=50)
        s = r.summary()
        found = sum(f"x{i}" in s for i in range(5))
        assert found >= 3

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lightgbm_fit(X, y, n_estimators=5)
        assert isinstance(r, FitResultBase)

    def test_max_depth_minus_one_valid(self, xy_rng42):
        """max_depth=-1 is valid (unlimited depth per LightGBM convention)."""
        X, y = xy_rng42
        r = mf.functions.lightgbm_fit(X, y, n_estimators=5, max_depth=-1)
        assert r.n_estimators_used == 5

    def test_max_depth_zero_raises(self, xy_rng42):
        """max_depth=0 must raise ValueError (invalid in LightGBM)."""
        X, y = xy_rng42
        with pytest.raises(ValueError, match="max_depth"):
            mf.functions.lightgbm_fit(X, y, max_depth=0)

    def test_max_depth_minus_two_raises(self, xy_rng42):
        """max_depth=-2 must raise ValueError (only -1 is the valid negative)."""
        X, y = xy_rng42
        with pytest.raises(ValueError, match="max_depth"):
            mf.functions.lightgbm_fit(X, y, max_depth=-2)

    def test_positive_max_depth_valid(self, xy_rng42):
        """max_depth=3 should work fine."""
        X, y = xy_rng42
        r = mf.functions.lightgbm_fit(X, y, n_estimators=5, max_depth=3)
        assert r.n_estimators_used == 5

    def test_validation_n_estimators(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_estimators"):
            mf.functions.lightgbm_fit(X, y, n_estimators=0)

    def test_namespace_wiring(self):
        assert "lightgbm_fit" in mf.functions.__all__
        assert "LightGBMFitResult" in mf.functions.__all__


# ---------------------------------------------------------------------------
# TestCatBoostFit
# ---------------------------------------------------------------------------

class TestCatBoostFit:
    """catboost_fit: correctness, predict shape (1-D guarantee), summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.catboost_fit(X, y)
        assert hasattr(r, "feature_importances_")
        assert r.feature_importances_.shape == (5,)
        assert r.n_estimators_used == 300

    def test_feature_importances_nonneg(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.catboost_fit(X, y, n_estimators=10)
        assert np.all(r.feature_importances_ >= 0)

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.catboost_fit(
            X_arr, y_arr, n_estimators=10, learning_rate=0.1, max_depth=6, random_state=0
        )
        preds_ref = _recipe_predict(
            "catboost",
            {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 6, "random_state": 0},
            X_arr, y_arr,
        )
        np.testing.assert_allclose(r.predict(X_arr), preds_ref, rtol=1e-6, atol=1e-6)

    def test_predict_shape_1d(self, xy_rng42):
        """CatBoost predict must be 1-D after .ravel()."""
        X, y = xy_rng42
        r = mf.functions.catboost_fit(X, y, n_estimators=5)
        preds = r.predict(X)
        assert preds.ndim == 1, f"Expected 1-D, got shape {preds.shape}"
        assert preds.shape == (100,)

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.catboost_fit(X_arr, y_arr, n_estimators=5)
        X_df = pd.DataFrame(X_arr)
        preds = r.predict(X_df)
        assert preds.ndim == 1
        assert preds.shape == (100,)

    def test_summary_contains_family_name(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.catboost_fit(X, y, n_estimators=5)
        s = r.summary()
        assert "CatBoost" in s
        assert "n_estimators" in s

    def test_summary_top3_features(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.catboost_fit(X, y, n_estimators=20)
        s = r.summary()
        found = sum(f"x{i}" in s for i in range(5))
        assert found >= 3

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.catboost_fit(X, y, n_estimators=5)
        assert isinstance(r, FitResultBase)

    def test_validation_n_estimators(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_estimators"):
            mf.functions.catboost_fit(X, y, n_estimators=0)

    def test_validation_learning_rate(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="learning_rate"):
            mf.functions.catboost_fit(X, y, learning_rate=0.0)

    def test_validation_max_depth(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="max_depth"):
            mf.functions.catboost_fit(X, y, max_depth=0)

    def test_namespace_wiring(self):
        assert "catboost_fit" in mf.functions.__all__
        assert "CatBoostFitResult" in mf.functions.__all__


# ---------------------------------------------------------------------------
# C35 Fixup Gate-6: percent-format smoke tests
# ---------------------------------------------------------------------------

class TestSummaryPercentFormat:
    """Gate 6 fixup: LightGBM and CatBoost summary() must use percent format."""

    def test_lightgbm_summary_contains_percent_symbol(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + 0.5 * rng.randn(100)
        r = mf.functions.lightgbm_fit(X, y)
        s = r.summary()
        assert "%" in s, f"LightGBM summary must contain % symbol; got:\n{s}"

    def test_lightgbm_summary_no_raw_float(self):
        """Ensure LightGBM does not display raw split counts like 46.000000."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + 0.5 * rng.randn(100)
        r = mf.functions.lightgbm_fit(X, y)
        s = r.summary()
        # raw split counts produce 6-decimal floats without %; verify absent
        import re
        raw_floats = re.findall(r"\d+\.\d{6}", s)
        assert not raw_floats, (
            f"LightGBM summary contains raw floats (not %) {raw_floats}; summary:\n{s}"
        )

    def test_lightgbm_summary_percent_values_in_range(self):
        """Each percent value shown must be in (0, 100]."""
        import re
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + 0.5 * rng.randn(100)
        r = mf.functions.lightgbm_fit(X, y)
        s = r.summary()
        vals = [float(v) for v in re.findall(r"([\d.]+)%", s)]
        assert vals, "No percent values found in LightGBM summary"
        for v in vals:
            assert 0.0 < v <= 100.0, f"Percent value out of (0,100]: {v}"

    def test_catboost_summary_contains_percent_symbol(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + 0.5 * rng.randn(100)
        r = mf.functions.catboost_fit(X, y, n_estimators=50)
        s = r.summary()
        assert "%" in s, f"CatBoost summary must contain % symbol; got:\n{s}"

    def test_catboost_summary_percent_values_in_range(self):
        """Each percent value shown must be in (0, 100]."""
        import re
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + 0.5 * rng.randn(100)
        r = mf.functions.catboost_fit(X, y, n_estimators=50)
        s = r.summary()
        vals = [float(v) for v in re.findall(r"([\d.]+)%", s)]
        assert vals, "No percent values found in CatBoost summary"
        for v in vals:
            assert 0.0 < v <= 100.0, f"Percent value out of (0,100]: {v}"
