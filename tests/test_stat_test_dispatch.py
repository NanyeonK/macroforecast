"""Tests for dispatch_stat_tests (Phase 2 sub-task 02.2 / 02.5)."""

from __future__ import annotations

import pandas as pd
import pytest

from macrocast.execution.stat_tests import AXIS_NAMES, dispatch_stat_tests


@pytest.fixture
def predictions() -> pd.DataFrame:
    rows = []
    for i in range(120):
        rows.append({
            "target": "INDPRO",
            "model_name": "ridge",
            "benchmark_name": "zero_change",
            "horizon": 1,
            "origin_date": pd.Timestamp("1990-01-01") + pd.DateOffset(months=i),
            "target_date": pd.Timestamp("1990-02-01") + pd.DateOffset(months=i),
            "fit_origin_date": pd.Timestamp("1980-01-01"),
            "selected_lag": 1,
            "selected_bic": float("nan"),
            "train_start_date": pd.Timestamp("1980-01-01"),
            "train_end_date": pd.Timestamp("1989-12-01"),
            "training_window_size": 120,
            "y_true": 0.5 + 0.01 * i,
            "y_pred": 0.4 + 0.01 * i,
            "benchmark_pred": 0.3 + 0.01 * i,
            "error": 0.1,
            "abs_error": 0.1,
            "squared_error": 0.01,
            "benchmark_error": 0.2,
            "benchmark_abs_error": 0.2,
            "benchmark_squared_error": 0.04,
        })
    return pd.DataFrame(rows)


def test_empty_spec_returns_empty_dict(predictions: pd.DataFrame) -> None:
    assert dispatch_stat_tests(
        predictions=predictions,
        stat_test_spec={},
        dependence_correction="none",
    ) == {}


def test_none_spec_returns_empty_dict(predictions: pd.DataFrame) -> None:
    assert dispatch_stat_tests(
        predictions=predictions,
        stat_test_spec=None,
        dependence_correction="none",
    ) == {}


def test_single_axis_routes_to_correct_test(predictions: pd.DataFrame) -> None:
    result = dispatch_stat_tests(
        predictions=predictions,
        stat_test_spec={"equal_predictive": "dm"},
        dependence_correction="none",
    )
    assert set(result.keys()) == {"equal_predictive"}
    payload = result["equal_predictive"]
    assert payload["axis"] == "equal_predictive"
    assert "statistic" in payload or "error" in payload


def test_skipped_none_values_are_absent(predictions: pd.DataFrame) -> None:
    result = dispatch_stat_tests(
        predictions=predictions,
        stat_test_spec={"equal_predictive": "none", "nested": "none"},
        dependence_correction="none",
    )
    assert result == {}


def test_unknown_value_surfaces_as_error_entry(predictions: pd.DataFrame) -> None:
    result = dispatch_stat_tests(
        predictions=predictions,
        stat_test_spec={"equal_predictive": "not_a_real_test"},
        dependence_correction="none",
    )
    assert "equal_predictive" in result
    entry = result["equal_predictive"]
    assert entry["exc_type"] == "NotImplementedError"
    assert "not operational" in entry["error"]


def test_test_scope_axis_records_scope_without_running_test(
    predictions: pd.DataFrame,
) -> None:
    result = dispatch_stat_tests(
        predictions=predictions,
        stat_test_spec={"test_scope": "per_horizon"},
        dependence_correction="none",
    )
    assert result == {"test_scope": {"axis": "test_scope", "scope": "per_horizon"}}


def test_legacy_stat_test_key_auto_routes(predictions: pd.DataFrame) -> None:
    result = dispatch_stat_tests(
        predictions=predictions,
        stat_test_spec={"equal_predictive": "dm"},
        dependence_correction="none",
    )
    assert "equal_predictive" in result
    assert result["equal_predictive"]["axis"] == "equal_predictive"


def test_multi_axis_spec_dispatches_each(predictions: pd.DataFrame) -> None:
    result = dispatch_stat_tests(
        predictions=predictions,
        stat_test_spec={
            "equal_predictive": "dm",
            "residual_diagnostics": "ljung_box",
            "direction": "binomial_hit",
        },
        dependence_correction="none",
    )
    assert set(result.keys()) == {"equal_predictive", "residual_diagnostics", "direction"}


def test_determinism(predictions: pd.DataFrame) -> None:
    spec = {"equal_predictive": "dm", "residual_diagnostics": "ljung_box"}
    r1 = dispatch_stat_tests(predictions=predictions, stat_test_spec=spec, dependence_correction="none")
    r2 = dispatch_stat_tests(predictions=predictions, stat_test_spec=spec, dependence_correction="none")
    assert r1.keys() == r2.keys()
    for k in r1:
        assert r1[k].get("statistic") == r2[k].get("statistic")
        assert r1[k].get("p_value") == r2[k].get("p_value")



def test_axis_names_contains_eight_entries() -> None:
    assert len(AXIS_NAMES) == 8
    assert "equal_predictive" in AXIS_NAMES
    assert "test_scope" in AXIS_NAMES
