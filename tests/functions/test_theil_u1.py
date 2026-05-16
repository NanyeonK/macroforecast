"""Tests for mf.functions.theil_u1 (Cycle 22 POC)."""
from __future__ import annotations

import math

import numpy as np
import pytest

import macroforecast as mf


class TestTheilU1:
    def test_perfect_forecast_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        u = mf.functions.theil_u1(y, y)
        assert u == 0.0, f"perfect forecast should give U1=0, got {u}"

    def test_hand_computed(self):
        """Verify against the closed-form formula with a simple 3-point example."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        # rmse_forecast = sqrt(mean((true-pred)^2)) = sqrt(mean(0.25)) = 0.5
        # sqrt(mean(y_true^2)) = sqrt((1+4+9)/3) = sqrt(14/3)
        # sqrt(mean(y_pred^2)) = sqrt((2.25+6.25+12.25)/3) = sqrt(20.75/3)
        rmse = 0.5
        denom = math.sqrt(14 / 3) + math.sqrt(20.75 / 3)
        expected = rmse / denom
        u = mf.functions.theil_u1(y_true, y_pred)
        assert abs(u - expected) < 1e-12, f"expected {expected}, got {u}"

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.theil_u1(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.theil_u1(np.array([]), np.array([]))

    def test_all_zeros_returns_nan(self):
        """When both arrays are zero, denominator is 0 -> nan."""
        u = mf.functions.theil_u1(np.zeros(5), np.zeros(5))
        assert math.isnan(u), f"expected nan, got {u}"
