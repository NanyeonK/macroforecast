"""Tests for mf.functions.ridge_fit (Cycle 22 POC)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _make_xy(n: int = 40, p: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X.sum(axis=1) + rng.standard_normal(n) * 0.1
    return X, y


class TestRidgeFitReturnsResult:
    """ridge_fit returns a RidgeFitResult with coef_ of the right shape."""

    def test_returns_result_with_coef(self):
        X, y = _make_xy()
        result = mf.functions.ridge_fit(X, y, alpha=0.5)
        assert hasattr(result, "coef_"), "RidgeFitResult must have coef_"
        assert result.coef_.shape == (5,), f"expected (5,), got {result.coef_.shape}"
        assert result.alpha == 0.5

    def test_predict_shape_matches_input(self):
        X, y = _make_xy()
        result = mf.functions.ridge_fit(X, y, alpha=1.0)
        preds = result.predict(X)
        assert preds.shape == (40,), f"expected (40,), got {preds.shape}"

    def test_alpha_zero_near_ols(self):
        """alpha=0 (pure OLS limit) should fit tightly on low-noise data."""
        X, y = _make_xy(n=60, p=3, seed=7)
        result = mf.functions.ridge_fit(X, y, alpha=0.0)
        preds = result.predict(X)
        resid = y - preds
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        # With only 3 features and very low noise, OLS should fit well.
        assert rmse < 0.5, f"OLS-limit RMSE too high: {rmse:.4f}"


class TestRidgeFitInputValidation:
    """ridge_fit raises ValueError on invalid alpha."""

    def test_negative_alpha_raises(self):
        X, y = _make_xy()
        with pytest.raises(ValueError, match="alpha"):
            mf.functions.ridge_fit(X, y, alpha=-1.0)
