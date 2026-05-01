"""v0.9.2 batch: agg_time / decomposition_target / forecast_type metadata +
   3 new stat tests (mcnemar / forecast_encompassing_nested /
   serial_dependence_loss_diff).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.execution.stat_tests.dispatch import dispatch_stat_tests
from macrocast.registry.build import _discover_axis_definitions


METADATA_FLIPS = (
    ("agg_time", "mean"),
    ("agg_time", "median"),
    ("agg_time", "per_subperiod"),
    ("decomposition_target", "none"),
    ("decomposition_target", "by_horizon"),
    ("decomposition_target", "by_target"),
    ("decomposition_target", "by_state"),
    ("decomposition_target", "by_regime"),
    # training + meta
    ("embargo_gap", "fixed_gap"),
    ("embargo_gap", "horizon_gap"),
    ("data_richness_mode", "factors_plus_target_lags"),
    ("data_richness_mode", "selected_sparse_predictors"),
    ("y_lag_count", "cv_select"),
    # data_task rules
)


@pytest.mark.parametrize("axis,value", METADATA_FLIPS)
def test_metadata_flip_is_operational(axis, value):
    defs = _discover_axis_definitions()
    status = next(e.status for e in defs[axis].entries if e.id == value)
    assert status == "operational"


def test_test_scope_metadata_statuses_match_layer6_runtime_contract():
    defs = _discover_axis_definitions()
    statuses = {entry.id: entry.status for entry in defs["test_scope"].entries}
    assert statuses["per_target_horizon"] == "operational"
    assert statuses["per_target"] == "operational"
    assert statuses["per_horizon"] == "operational"
    assert statuses["pooled"] == "operational"


def test_layer7_detail_axis_metadata_statuses_match_runtime_contract():
    defs = _discover_axis_definitions()
    temporal = {entry.id: entry.status for entry in defs["importance_temporal"].entries}
    gradient = {entry.id: entry.status for entry in defs["importance_gradient_path"].entries}
    assert temporal["static_snapshot"] == "operational"
    assert temporal["time_average"] == "registry_only"
    assert temporal["rolling_path"] == "registry_only"
    assert gradient["none"] == "operational"
    assert gradient["coefficient_path"] == "registry_only"


def _preds(n=40, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "horizon": [1] * n,
        "target_date": pd.date_range("2020-01-01", periods=n, freq="MS"),
        "origin_date": pd.date_range("2019-12-01", periods=n, freq="MS"),
        "y_true": rng.standard_normal(n),
        "y_pred": rng.standard_normal(n),
        "benchmark_pred": rng.standard_normal(n),
    })
    df["squared_error"] = (df["y_true"] - df["y_pred"]) ** 2
    df["benchmark_squared_error"] = (df["y_true"] - df["benchmark_pred"]) ** 2
    return df


def test_mcnemar_returns_expected_keys():
    out = dispatch_stat_tests(
        predictions=_preds(), stat_test_spec={"direction": "mcnemar"}, dependence_correction="none",
    )
    r = out["direction"]
    assert r["stat_test"] == "mcnemar"
    for k in ("n", "model_hit_rate", "benchmark_hit_rate", "statistic", "p_value", "significant_5pct"):
        assert k in r
    assert 0.0 <= r["p_value"] <= 1.0


def test_forecast_encompassing_nested_returns_expected_keys():
    out = dispatch_stat_tests(
        predictions=_preds(),
        stat_test_spec={"nested": "forecast_encompassing_nested"},
        dependence_correction="none",
    )
    r = out["nested"]
    assert r["stat_test"] == "forecast_encompassing_nested"
    for k in ("n", "beta", "se_beta", "t_statistic", "p_value", "encompassed_5pct"):
        assert k in r


def test_serial_dependence_loss_diff_returns_expected_keys():
    out = dispatch_stat_tests(
        predictions=_preds(),
        stat_test_spec={"residual_diagnostics": "serial_dependence_loss_diff"},
        dependence_correction="none",
    )
    r = out["residual_diagnostics"]
    assert r["stat_test"] == "serial_dependence_loss_diff"
    for k in ("n", "durbin_watson", "lag1_autocorr_estimate", "flag_serial_dependence"):
        assert k in r


def test_stepwise_mcs_returns_expected_keys():
    out = dispatch_stat_tests(
        predictions=_preds(), stat_test_spec={"multiple_model": "stepwise_mcs"},
        dependence_correction="none",
    )
    r = out["multiple_model"]
    # Falls back to plain mcs when only single model vs benchmark
    assert r["stat_test"] in {"stepwise_mcs", "mcs"}
    assert "n" in r


def test_bootstrap_best_model_returns_expected_keys():
    out = dispatch_stat_tests(
        predictions=_preds(), stat_test_spec={"multiple_model": "bootstrap_best_model"},
        dependence_correction="none",
    )
    r = out["multiple_model"]
    assert r["stat_test"] == "bootstrap_best_model"
    assert 0.0 <= r["freq_model_beats_benchmark"] <= 1.0
    assert r["n_bootstrap"] > 0


def test_roc_comparison_returns_expected_keys():
    out = dispatch_stat_tests(
        predictions=_preds(), stat_test_spec={"direction": "roc_comparison"},
        dependence_correction="none",
    )
    r = out["direction"]
    assert r["stat_test"] == "roc_comparison"
    for k in ("n", "auc_model", "auc_benchmark", "auc_delta"):
        assert k in r


def test_cusum_on_loss_returns_expected_keys():
    out = dispatch_stat_tests(
        predictions=_preds(), stat_test_spec={"cpa_instability": "cusum_on_loss"},
        dependence_correction="none",
    )
    r = out["cpa_instability"]
    assert r["stat_test"] == "cusum_on_loss"
    for k in ("n", "max_abs_normalized_cusum", "critical_5pct", "flag_instability"):
        assert k in r


def test_new_stat_tests_registry_operational():
    defs = _discover_axis_definitions()
    assert next(e.status for e in defs["direction"].entries if e.id == "mcnemar") == "operational"
    assert next(e.status for e in defs["nested"].entries if e.id == "forecast_encompassing_nested") == "operational"
    assert next(e.status for e in defs["residual_diagnostics"].entries if e.id == "serial_dependence_loss_diff") == "operational"
    assert next(e.status for e in defs["multiple_model"].entries if e.id == "stepwise_mcs") == "operational"
    assert next(e.status for e in defs["multiple_model"].entries if e.id == "bootstrap_best_model") == "operational"
    assert next(e.status for e in defs["direction"].entries if e.id == "roc_comparison") == "operational"
    assert next(e.status for e in defs["cpa_instability"].entries if e.id == "cusum_on_loss") == "operational"
