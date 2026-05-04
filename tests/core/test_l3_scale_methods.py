"""Tests for the three L3 scale methods (PR-E of the v0.1 honesty pass).

The design-time L3 step library (``plans/design/part2_l2_l3_l4.md``)
listed ``scale (zscore/robust/minmax)`` as a single operational op. v0.1
only implemented ``zscore`` and raised ``NotImplementedError`` for the
other two; PR-E delivers the missing methods. Each test cross-checks
against the canonical sklearn implementation at machine precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from macrocast.core.runtime import _scale_frame


def _toy_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(loc=2.0, scale=1.5, size=64),
            "x2": rng.standard_normal(64) * 5.0,
            "x3": rng.uniform(-10, 10, size=64),
        }
    )


@pytest.mark.parametrize("method", ["zscore", "standard", "standardize"])
def test_zscore_aliases_match_sklearn_standard_scaler(method):
    frame = _toy_frame()
    macro = _scale_frame(frame, method=method)
    expected = StandardScaler().fit_transform(frame.to_numpy())
    np.testing.assert_allclose(macro.to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_robust_method_matches_sklearn_robust_scaler():
    frame = _toy_frame(seed=7)
    macro = _scale_frame(frame, method="robust")
    expected = RobustScaler(with_centering=True, with_scaling=True).fit_transform(frame.to_numpy())
    np.testing.assert_allclose(macro.to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_minmax_method_matches_sklearn_minmax_scaler():
    frame = _toy_frame(seed=13)
    macro = _scale_frame(frame, method="minmax")
    expected = MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(frame.to_numpy())
    np.testing.assert_allclose(macro.to_numpy(), expected, rtol=1e-9, atol=1e-9)


def test_unknown_method_still_raises():
    frame = _toy_frame()
    with pytest.raises(NotImplementedError, match=r"scale method"):
        _scale_frame(frame, method="totally_made_up")


def test_zero_variance_column_does_not_crash_zscore():
    frame = pd.DataFrame({"const": [3.0] * 10, "varying": list(range(10))})
    out = _scale_frame(frame, method="zscore")
    assert out.shape == frame.shape
    # constant column produces NaN (divide by zero replaced); varying column
    # has finite values.
    assert out["varying"].notna().all()


def test_zero_iqr_column_does_not_crash_robust():
    frame = pd.DataFrame({"const": [3.0] * 10, "varying": list(range(10))})
    out = _scale_frame(frame, method="robust")
    assert out.shape == frame.shape
    assert out["varying"].notna().all()


def test_zero_range_column_does_not_crash_minmax():
    frame = pd.DataFrame({"const": [3.0] * 10, "varying": list(range(10))})
    out = _scale_frame(frame, method="minmax")
    assert out.shape == frame.shape
    # The varying column should now be in [0, 1] after rescaling.
    assert out["varying"].between(0.0, 1.0).all()
