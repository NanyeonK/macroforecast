"""v0.91 interim: tests for stat-test planned-status promotions (batch 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.execution.stat_tests.dispatch import dispatch_stat_tests


def _predictions(n: int = 30, seed: int = 0, bias: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    df = pd.DataFrame({
        "horizon": [1] * n,
        "target_date": dates,
        "origin_date": dates - pd.DateOffset(months=1),
        "y_true": rng.standard_normal(n) + bias,
        "y_pred": rng.standard_normal(n),
        "benchmark_pred": rng.standard_normal(n),
    })
    df["squared_error"] = (df["y_true"] - df["y_pred"]) ** 2
    df["benchmark_squared_error"] = (df["y_true"] - df["benchmark_pred"]) ** 2
    return df


def test_paired_t_on_loss_diff_returns_expected_keys():
    df = _predictions()
    out = dispatch_stat_tests(
        predictions=df,
        stat_test_spec={"equal_predictive": "paired_t_on_loss_diff"},
        dependence_correction="none",
    )
    result = out["equal_predictive"]
    assert result["stat_test"] == "paired_t_on_loss_diff"
    for key in ("n", "mean_loss_diff", "variance", "t_statistic", "p_value", "significant_5pct"):
        assert key in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_wilcoxon_signed_rank_returns_expected_keys():
    df = _predictions()
    out = dispatch_stat_tests(
        predictions=df,
        stat_test_spec={"equal_predictive": "wilcoxon_signed_rank"},
        dependence_correction="none",
    )
    result = out["equal_predictive"]
    assert result["stat_test"] == "wilcoxon_signed_rank"
    for key in ("n", "mean_loss_diff", "statistic", "p_value", "significant_5pct"):
        assert key in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_autocorrelation_of_errors_returns_expected_keys():
    df = _predictions()
    out = dispatch_stat_tests(
        predictions=df,
        stat_test_spec={"residual_diagnostics": "autocorrelation_of_errors"},
        dependence_correction="none",
    )
    result = out["residual_diagnostics"]
    assert result["stat_test"] == "autocorrelation_of_errors"
    for key in ("n", "max_lag", "rho", "q_statistic", "p_value", "significant_5pct"):
        assert key in result
    assert isinstance(result["rho"], list)
    assert len(result["rho"]) == result["max_lag"]


def test_paired_t_detects_systematic_advantage():
    """Constructing loss_diff with a clear positive mean should yield low p."""
    n = 80
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "horizon": [1] * n,
        "target_date": pd.date_range("2020-01-01", periods=n, freq="MS"),
        "origin_date": pd.date_range("2019-12-01", periods=n, freq="MS"),
        "y_true": np.zeros(n),
        "y_pred": rng.standard_normal(n) * 0.1,
        "benchmark_pred": rng.standard_normal(n) * 1.0,
    })
    df["squared_error"] = (df["y_true"] - df["y_pred"]) ** 2
    df["benchmark_squared_error"] = (df["y_true"] - df["benchmark_pred"]) ** 2
    out = dispatch_stat_tests(
        predictions=df,
        stat_test_spec={"equal_predictive": "paired_t_on_loss_diff"},
        dependence_correction="none",
    )
    assert out["equal_predictive"]["p_value"] < 0.05
    assert out["equal_predictive"]["significant_5pct"] is True


def test_registry_promotions_are_operational():
    from macrocast.registry.build import _discover_axis_definitions

    defs = _discover_axis_definitions()

    def _status(axis: str, value: str) -> str:
        return next(e.status for e in defs[axis].entries if e.id == value)

    assert _status("equal_predictive", "paired_t_on_loss_diff") == "operational"
    assert _status("equal_predictive", "wilcoxon_signed_rank") == "operational"
    assert _status("residual_diagnostics", "autocorrelation_of_errors") == "operational"
