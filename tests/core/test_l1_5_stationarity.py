"""Issue #210 -- L1.5 stationarity_test axis (adf / pp / kpss / multi)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.core.runtime import _diagnostic_stationarity_tests


def _trending_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    trend = np.linspace(0, 5, n)  # I(1)-like trending
    noise = rng.normal(scale=0.2, size=n)
    stationary = rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"trending": trend + noise, "stationary": stationary}, index=idx)


def test_adf_runs_per_series():
    frame = _trending_frame()
    results = _diagnostic_stationarity_tests(
        frame=frame,
        test="adf",
        scope="target_and_predictors",
        target="trending",
        targets=("trending",),
    )
    assert results["test"] == "adf"
    assert {"trending", "stationary"}.issubset(results["by_series"])
    s = results["by_series"]["stationary"]["adf"]
    assert "statistic" in s and "p_value" in s
    # white noise should reject unit root
    assert s["reject_unit_root"] is True


def test_kpss_runs_and_distinguishes_stationary_vs_trending():
    frame = _trending_frame()
    results = _diagnostic_stationarity_tests(
        frame=frame,
        test="kpss",
        scope="target_and_predictors",
        target="trending",
        targets=("trending",),
    )
    by = results["by_series"]
    # KPSS null = stationarity. Trending series should reject it; white noise
    # should not.
    assert by["trending"]["kpss"]["reject_stationarity"] is True
    assert by["stationary"]["kpss"]["reject_stationarity"] is False


def test_multi_runs_all_three_when_available():
    frame = _trending_frame()
    results = _diagnostic_stationarity_tests(
        frame=frame,
        test="multi",
        scope="target_and_predictors",
        target="trending",
        targets=("trending",),
    )
    s = results["by_series"]["trending"]
    assert "adf" in s and "kpss" in s
    assert "pp" in s
    # PP either runs (with arch installed) or reports unavailable.
    pp = s["pp"]
    assert "statistic" in pp or pp.get("status") == "unavailable"


def test_scope_target_only_filters_to_target():
    frame = _trending_frame()
    results = _diagnostic_stationarity_tests(
        frame=frame,
        test="adf",
        scope="target_only",
        target="trending",
        targets=("trending",),
    )
    assert set(results["by_series"]) == {"trending"}


def test_scope_predictors_only_excludes_target():
    frame = _trending_frame()
    results = _diagnostic_stationarity_tests(
        frame=frame,
        test="adf",
        scope="predictors_only",
        target="trending",
        targets=("trending",),
    )
    assert set(results["by_series"]) == {"stationary"}


def test_insufficient_data_does_not_crash():
    short = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    results = _diagnostic_stationarity_tests(
        frame=short, test="adf", scope="target_and_predictors", target=None, targets=()
    )
    assert results["by_series"]["x"]["status"] == "insufficient_data"


def test_phillips_perron_native_runs_without_arch():
    """Issue #252 -- PP must produce a statistic + p-value even when
    ``arch`` is not installed. Native implementation runs the OLS regression
    + Newey-West HAC adjustment.
    """

    from macrocast.core.runtime import _phillips_perron_native

    rng = np.random.default_rng(0)
    # White noise -> stationary -> reject unit root.
    y_stationary = rng.normal(size=200)
    res = _phillips_perron_native(y_stationary, alpha=0.05)
    assert "statistic" in res and "p_value" in res
    assert res["p_value"] < 0.05
    # Random walk -> unit root -> do not reject.
    y_rw = np.cumsum(rng.normal(size=200))
    res_rw = _phillips_perron_native(y_rw, alpha=0.05)
    assert res_rw["p_value"] > 0.05
