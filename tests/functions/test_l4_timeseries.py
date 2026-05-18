"""Tests for Cycle 37 L4 timeseries family standalone callables.

14 test classes cover the 14 new callables in ``mf.functions``:
``var_fit``, ``bvar_minnesota_fit``, ``bvar_niw_fit``, ``ar_fit``,
``far_fit``, ``pcr_fit``, ``favar_fit``, ``garch11_fit``, ``egarch_fit``,
``realized_garch_fit``, ``ets_fit``, ``theta_fit``, ``holt_winters_fit``,
``dfm_fit``.

Bit-exact assertions compare against ``_build_l4_model`` recipe path
where deterministic (VAR/AR/FAR/PCR/FAVAR/BVAR/DFM/ETS/Theta/HoltWinters).
GARCH / EGARCH / RealizedGARCH require optional ``arch`` dep: skipif guard.

Protocol conformance: ``isinstance(r, FitResultBase)``.

Uses small panels (50x3) and n_lags=1 for CI speed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.functions import FitResultBase
from macroforecast.core.runtime import _build_l4_model


# ---------------------------------------------------------------------------
# Optional dep availability checks
# ---------------------------------------------------------------------------

def _arch_available() -> bool:
    try:
        import arch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared deterministic fixtures
# ---------------------------------------------------------------------------

def _make_xy_rng42(n: int = 50, p: int = 3):
    """RNG-42 small panel: X ~ N(0,1), y = X @ [1,2,3] + 0.5*noise."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    beta = np.arange(1, p + 1, dtype=float)
    y = X @ beta + 0.5 * rng.randn(n)
    return X, y


@pytest.fixture(scope="module")
def xy_rng42():
    return _make_xy_rng42()


def _make_y_returns(n: int = 100):
    """Return-like series for GARCH tests (needs >= 30 obs)."""
    rng = np.random.RandomState(7)
    return rng.randn(n)


def _recipe_predict(family: str, params: dict, X_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr.ravel(), name="y")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build_l4_model(family, params)
        model.fit(X, y)
    return np.asarray(model.predict(X), dtype=float).ravel()


# ===========================================================================
# TestVARFit
# ===========================================================================

class TestVARFit:
    """var_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.var_fit(X, y, n_lags=1)
        assert r.n_lags == 1
        assert r.n_obs == 50

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.var_fit(X_arr, y_arr, n_lags=1)
        ref = _recipe_predict("var", {"n_lag": 1}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.var_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.var_fit(X_arr, y_arr)
        assert r.predict(pd.DataFrame(X_arr)).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.var_fit(X, y)
        s = r.summary()
        assert "VAR" in s
        assert "n_lags" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.var_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_n_lags(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_lags"):
            mf.functions.var_fit(X, y, n_lags=0)

    def test_namespace_wiring(self):
        assert "var_fit" in mf.functions.__all__
        assert "VARFitResult" in mf.functions.__all__


# ===========================================================================
# TestBVARMinnesotaFit
# ===========================================================================

class TestBVARMinnesotaFit:
    """bvar_minnesota_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.bvar_minnesota_fit(X, y, n_lags=1)
        assert r.n_lags == 1
        assert r.lambda1 == 0.2
        assert r.n_obs == 50

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.bvar_minnesota_fit(X_arr, y_arr, n_lags=1, lambda1=0.2)
        ref = _recipe_predict("bvar_minnesota", {"n_lag": 1, "lambda_1": 0.2}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.bvar_minnesota_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.bvar_minnesota_fit(X, y)
        s = r.summary()
        assert "BVAR" in s
        assert "lambda1" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.bvar_minnesota_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_n_lags(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_lags"):
            mf.functions.bvar_minnesota_fit(X, y, n_lags=0)

    def test_validation_lambda1(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="lambda1"):
            mf.functions.bvar_minnesota_fit(X, y, lambda1=0.0)

    def test_namespace_wiring(self):
        assert "bvar_minnesota_fit" in mf.functions.__all__
        assert "BVARMinnesotaFitResult" in mf.functions.__all__


# ===========================================================================
# TestBVARNIWFit
# ===========================================================================

class TestBVARNIWFit:
    """bvar_niw_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.bvar_niw_fit(X, y, n_lags=1)
        assert r.n_lags == 1
        assert r.n_obs == 50

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.bvar_niw_fit(X_arr, y_arr, n_lags=1)
        ref = _recipe_predict("bvar_normal_inverse_wishart", {"n_lag": 1}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.bvar_niw_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.bvar_niw_fit(X, y)
        s = r.summary()
        assert "BVAR" in s
        assert "NIW" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.bvar_niw_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_n_lags(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_lags"):
            mf.functions.bvar_niw_fit(X, y, n_lags=0)

    def test_namespace_wiring(self):
        assert "bvar_niw_fit" in mf.functions.__all__
        assert "BVARNIWFitResult" in mf.functions.__all__


# ===========================================================================
# TestARFit
# ===========================================================================

class TestARFit:
    """ar_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.ar_fit(X, y, n_lags=1)
        assert r.n_lags == 1
        assert r.coef_.shape == (1,)
        assert isinstance(r.intercept_, float)

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.ar_fit(X_arr, y_arr, n_lags=1)
        ref = _recipe_predict("ar_p", {"n_lag": 1}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-12)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.ar_fit(X, y)
        preds = r.predict(X)
        assert preds.shape == (50,)
        assert preds.dtype == float

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.ar_fit(X_arr, y_arr)
        assert r.predict(pd.DataFrame(X_arr)).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.ar_fit(X, y)
        s = r.summary()
        assert "AR" in s
        assert "n_lags" in s
        assert "intercept_" in s

    def test_coef_shape_n_lags_2(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.ar_fit(X, y, n_lags=2)
        assert r.coef_.shape == (2,)
        assert r.n_lags == 2

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.ar_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_n_lags(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_lags"):
            mf.functions.ar_fit(X, y, n_lags=0)

    def test_namespace_wiring(self):
        assert "ar_fit" in mf.functions.__all__
        assert "ARFitResult" in mf.functions.__all__


# ===========================================================================
# TestFARFit
# ===========================================================================

class TestFARFit:
    """far_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.far_fit(X, y, n_factors=2, n_lags=1)
        assert r.n_factors == 2
        assert r.n_lags == 1

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.far_fit(X_arr, y_arr, n_factors=2, n_lags=1)
        ref = _recipe_predict("factor_augmented_ar", {"n_factors": 2, "n_lag": 1}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.far_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.far_fit(X, y)
        s = r.summary()
        assert "FAR" in s
        assert "n_factors" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.far_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_n_factors(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_factors"):
            mf.functions.far_fit(X, y, n_factors=0)

    def test_validation_n_lags(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_lags"):
            mf.functions.far_fit(X, y, n_lags=0)

    def test_namespace_wiring(self):
        assert "far_fit" in mf.functions.__all__
        assert "FARFitResult" in mf.functions.__all__


# ===========================================================================
# TestPCRFit
# ===========================================================================

class TestPCRFit:
    """pcr_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.pcr_fit(X, y, n_components=2)
        assert r.n_components == 2

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.pcr_fit(X_arr, y_arr, n_components=2)
        ref = _recipe_predict("principal_component_regression", {"n_components": 2}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.pcr_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.pcr_fit(X, y)
        s = r.summary()
        assert "PCR" in s
        assert "n_components" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.pcr_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_n_components(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_components"):
            mf.functions.pcr_fit(X, y, n_components=0)

    def test_namespace_wiring(self):
        assert "pcr_fit" in mf.functions.__all__
        assert "PCRFitResult" in mf.functions.__all__


# ===========================================================================
# TestFAVARFit
# ===========================================================================

class TestFAVARFit:
    """favar_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.favar_fit(X, y, n_factors=2, n_lags=1)
        assert r.n_factors == 2
        assert r.n_lags == 1

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.favar_fit(X_arr, y_arr, n_factors=2, n_lags=1)
        ref = _recipe_predict("factor_augmented_var", {"n_factors": 2, "n_lag": 1}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.favar_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.favar_fit(X, y)
        s = r.summary()
        assert "FAVAR" in s
        assert "n_factors" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.favar_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_n_factors(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_factors"):
            mf.functions.favar_fit(X, y, n_factors=0)

    def test_validation_n_lags(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_lags"):
            mf.functions.favar_fit(X, y, n_lags=0)

    def test_namespace_wiring(self):
        assert "favar_fit" in mf.functions.__all__
        assert "FAVARFitResult" in mf.functions.__all__


# ===========================================================================
# TestGARCH11Fit
# ===========================================================================

@pytest.mark.skipif(not _arch_available(), reason="arch not installed")
class TestGARCH11Fit:
    """garch11_fit: correctness (arch available), predict, summary, protocol."""

    @pytest.fixture(scope="class")
    def garch_xy(self):
        rng = np.random.RandomState(7)
        X = rng.randn(100, 3)
        y = rng.randn(100)
        return X, y

    def test_returns_result(self, garch_xy):
        X, y = garch_xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.garch11_fit(X, y)
        assert isinstance(r.conditional_mu, float)
        assert r.n_obs == 100
        assert isinstance(r.params_, dict)

    def test_predict_shape(self, garch_xy):
        X, y = garch_xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.garch11_fit(X, y)
        preds = r.predict(X)
        assert preds.shape == (100,)

    def test_predict_variance_shape(self, garch_xy):
        X, y = garch_xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.garch11_fit(X, y)
        var_preds = r.predict_variance(h_steps=3)
        assert var_preds.shape == (3,)

    def test_summary_contains_required_fields(self, garch_xy):
        X, y = garch_xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.garch11_fit(X, y)
        s = r.summary()
        assert "GARCH" in s
        assert "conditional_mu" in s

    def test_protocol_conformance(self, garch_xy):
        X, y = garch_xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.garch11_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_namespace_wiring(self):
        assert "garch11_fit" in mf.functions.__all__
        assert "GARCH11FitResult" in mf.functions.__all__


# ===========================================================================
# TestEGARCHFit
# ===========================================================================

@pytest.mark.skipif(not _arch_available(), reason="arch not installed")
class TestEGARCHFit:
    """egarch_fit: correctness (arch available), predict, summary, protocol."""

    @pytest.fixture(scope="class")
    def garch_xy(self):
        rng = np.random.RandomState(7)
        X = rng.randn(100, 3)
        y = rng.randn(100)
        return X, y

    def test_returns_result(self, garch_xy):
        X, y = garch_xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.egarch_fit(X, y)
        assert isinstance(r.conditional_mu, float)
        assert r.n_obs == 100
        assert isinstance(r.params_, dict)

    def test_predict_shape(self, garch_xy):
        X, y = garch_xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.egarch_fit(X, y)
        preds = r.predict(X)
        assert preds.shape == (100,)

    def test_summary_contains_required_fields(self, garch_xy):
        X, y = garch_xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.egarch_fit(X, y)
        s = r.summary()
        assert "EGARCH" in s

    def test_protocol_conformance(self, garch_xy):
        X, y = garch_xy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.egarch_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_namespace_wiring(self):
        assert "egarch_fit" in mf.functions.__all__
        assert "EGARCHFitResult" in mf.functions.__all__


# ===========================================================================
# TestRealizedGARCHFit
# ===========================================================================

@pytest.mark.skipif(not _arch_available(), reason="arch not installed")
class TestRealizedGARCHFit:
    """realized_garch_fit: correctness (arch available), predict, validation."""

    @pytest.fixture(scope="class")
    def garch_xy_rv(self):
        rng = np.random.RandomState(7)
        X = rng.randn(100, 3)
        y = rng.randn(100)
        rv = np.abs(rng.randn(100))
        return X, y, rv

    def test_returns_result(self, garch_xy_rv):
        X, y, rv = garch_xy_rv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.realized_garch_fit(X, y, rv)
        assert isinstance(r.conditional_mu, float)
        assert r.n_obs == 100

    def test_predict_shape(self, garch_xy_rv):
        X, y, rv = garch_xy_rv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.realized_garch_fit(X, y, rv)
        assert r.predict(X).shape == (100,)

    def test_validation_rv_length_mismatch(self, garch_xy_rv):
        X, y, rv = garch_xy_rv
        with pytest.raises(ValueError, match="rv length"):
            mf.functions.realized_garch_fit(X, y, rv[:50])

    def test_summary_contains_required_fields(self, garch_xy_rv):
        X, y, rv = garch_xy_rv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.realized_garch_fit(X, y, rv)
        s = r.summary()
        assert "RealizedGARCH" in s

    def test_namespace_wiring(self):
        assert "realized_garch_fit" in mf.functions.__all__
        assert "RealizedGARCHFitResult" in mf.functions.__all__


# ===========================================================================
# TestETSFit
# ===========================================================================

class TestETSFit:
    """ets_fit: correctness, predict, summary, protocol."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.ets_fit(X, y)
        assert r.error_trend_seasonal == "AAN"
        assert r.n_obs == 50

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.ets_fit(X_arr, y_arr)
        ref = _recipe_predict("ets", {}, X_arr, y_arr)
        # ETS is deterministic; bit-exact within float tolerance
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-8)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.ets_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.ets_fit(X, y)
        s = r.summary()
        assert "ETS" in s
        assert "error_trend_seasonal" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.ets_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_namespace_wiring(self):
        assert "ets_fit" in mf.functions.__all__
        assert "ETSFitResult" in mf.functions.__all__


# ===========================================================================
# TestThetaFit
# ===========================================================================

class TestThetaFit:
    """theta_fit: correctness, predict, summary, protocol."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.theta_fit(X, y)
        assert r.theta == 2.0
        assert 0.0 < r.alpha_ <= 1.0
        assert r.n_obs == 50

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.theta_fit(X_arr, y_arr)
        ref = _recipe_predict("theta_method", {}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.theta_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.theta_fit(X, y)
        s = r.summary()
        assert "Theta" in s
        assert "alpha_" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.theta_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_namespace_wiring(self):
        assert "theta_fit" in mf.functions.__all__
        assert "ThetaFitResult" in mf.functions.__all__


# ===========================================================================
# TestHoltWintersFit
# ===========================================================================

class TestHoltWintersFit:
    """holt_winters_fit: correctness, predict, summary, protocol."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.holt_winters_fit(X, y)
        assert r.seasonal == "add"
        assert r.seasonal_periods == 12
        assert r.n_obs == 50

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.holt_winters_fit(X_arr, y_arr)
        ref = _recipe_predict("holt_winters", {}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-8)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.holt_winters_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.holt_winters_fit(X, y)
        s = r.summary()
        assert "Holt-Winters" in s
        assert "seasonal" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.holt_winters_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_namespace_wiring(self):
        assert "holt_winters_fit" in mf.functions.__all__
        assert "HoltWintersFitResult" in mf.functions.__all__


# ===========================================================================
# TestDFMFit
# ===========================================================================

class TestDFMFit:
    """dfm_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.dfm_fit(X, y, n_factors=2)
        assert r.n_factors == 2
        assert r.n_obs == 50

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.dfm_fit(X_arr, y_arr, n_factors=2)
        ref = _recipe_predict("dfm_mixed_mariano_murasawa", {"n_factors": 2}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-8)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.dfm_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.dfm_fit(X, y)
        s = r.summary()
        assert "DFM" in s
        assert "n_factors" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.dfm_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_n_factors(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_factors"):
            mf.functions.dfm_fit(X, y, n_factors=0)

    def test_namespace_wiring(self):
        assert "dfm_fit" in mf.functions.__all__
        assert "DFMFitResult" in mf.functions.__all__
