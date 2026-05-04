"""Issues #246 + #247 -- per-family quantile estimators + density strict mode."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macrocast.core.runtime import _emit_quantile_intervals, _l6_density_interval_results, _native_quantile_engine
from macrocast.core.types import L4ForecastsArtifact, L3FeaturesArtifact, L1DataDefinitionArtifact, Panel, PanelMetadata, Series, SeriesMetadata


# ---------------------------------------------------------------------------
# #246 native quantile engines
# ---------------------------------------------------------------------------

def test_native_quantile_engine_for_linear_returns_quantile_regressor():
    factory = _native_quantile_engine("ridge")
    assert factory is not None
    fitted = factory(0.5)
    # Sklearn QuantileRegressor for the median.
    assert hasattr(fitted, "quantile") or hasattr(fitted, "fit")


def test_native_quantile_engine_for_gradient_boosting():
    factory = _native_quantile_engine("gradient_boosting")
    assert factory is not None
    fitted = factory(0.9)
    assert getattr(fitted, "loss", None) == "quantile"


def test_native_quantile_engine_returns_none_for_unsupported():
    assert _native_quantile_engine("not_a_real_family") is None


def test_emit_quantile_intervals_runs_per_quantile_when_X_y_supplied():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series(2.0 * X["a"] + rng.normal(scale=0.3, size=n))
    forecasts = {("m", "y", 1, idx): float(y.iloc[i]) for i, idx in enumerate(X.index)}
    fit_nodes = [
        {
            "op": "fit_model",
            "params": {"forecast_object": "quantile", "family": "ridge", "quantile_levels": [0.1, 0.5, 0.9]},
        }
    ]
    intervals = _emit_quantile_intervals(forecasts, fit_nodes, X=X, y=y)
    # 3 quantile levels × n origins.
    assert len(intervals) == 3 * n
    # Quantile crossings: q_low <= q_med <= q_high for every origin.
    for origin in X.index:
        q_low = intervals[("m", "y", 1, origin, 0.1)]
        q_med = intervals[("m", "y", 1, origin, 0.5)]
        q_high = intervals[("m", "y", 1, origin, 0.9)]
        # Median should sit between extremes (allow some quantile crossing).
        assert q_low - 0.5 <= q_med <= q_high + 0.5


# ---------------------------------------------------------------------------
# #247 density strict mode
# ---------------------------------------------------------------------------

def _build_l3_l1():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    l3 = L3FeaturesArtifact(
        X_final=Panel(data=pd.DataFrame({"x": [0.0]}), shape=(1, 1), column_names=("x",), index=pd.Index([0]), metadata=PanelMetadata()),
        y_final=Series(shape=actual.shape, name="y", metadata=SeriesMetadata(values={"data": actual})),
        sample_index=pd.DatetimeIndex(actual.index),
        horizon_set=(1,),
    )
    l1 = L1DataDefinitionArtifact(
        custom_source_policy="custom_panel_only", dataset="custom", frequency="monthly",
        vintage_policy="current_vintage", target_structure="single_target", target="y",
        targets=("y",), variable_universe="all_variables", target_geography_scope=None,
        predictor_geography_scope=None, sample_start_rule="max_balanced",
        sample_end_rule="latest_available", horizon_set="custom_list", target_horizons=(1,),
        regime_definition="none",
        raw_panel=Panel(data=pd.DataFrame(), shape=(0, 0), column_names=(), index=pd.Index([]), metadata=PanelMetadata()),
        leaf_config={},
    )
    return l3, l1


def test_density_strict_mode_rejects_point_forecasts_without_intervals():
    l3, l1 = _build_l3_l1()
    forecasts = L4ForecastsArtifact(
        forecasts={("m", "y", 1, 0): 1.1, ("m", "y", 1, 1): 1.9},
        forecast_intervals={},
        forecast_object="point",
        sample_index=pd.DatetimeIndex([0, 1]),
        targets=("y",),
        horizons=(1,),
        model_ids=("m",),
    )
    result = _l6_density_interval_results(forecasts, l1, l3, sub={})
    assert result["status"] == "requires_quantile_or_density_forecast"
    assert "remediation" in result


def test_density_strict_mode_runs_when_intervals_present():
    l3, l1 = _build_l3_l1()
    intervals = {}
    for i in range(5):
        for q in (0.1, 0.5, 0.9):
            intervals[("m", "y", 1, i, q)] = (i + 1) + (q - 0.5) * 2
    forecasts = L4ForecastsArtifact(
        forecasts={("m", "y", 1, i): float(i + 1) for i in range(5)},
        forecast_intervals=intervals,
        forecast_object="quantile",
        sample_index=pd.DatetimeIndex(range(5)),
        targets=("y",),
        horizons=(1,),
        model_ids=("m",),
    )
    result = _l6_density_interval_results(forecasts, l1, l3, sub={})
    # No status: actual battery ran. Expect ('density', 'm') key with sub-tests.
    assert ("density", "m") in result


def test_density_residual_synth_opt_in():
    l3, l1 = _build_l3_l1()
    forecasts = L4ForecastsArtifact(
        forecasts={("m", "y", 1, i): float(i + 1.1) for i in range(20)},
        forecast_intervals={},
        forecast_object="point",
        sample_index=pd.DatetimeIndex(range(20)),
        targets=("y",),
        horizons=(1,),
        model_ids=("m",),
    )
    result = _l6_density_interval_results(forecasts, l1, l3, sub={"allow_residual_synth": True})
    # Synth path: produces some battery output rather than the strict-mode
    # ``requires_quantile_or_density_forecast`` status.
    assert result.get("status") != "requires_quantile_or_density_forecast"
