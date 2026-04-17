"""Phase 2 sub-task 02.1 / 02.5 — registry coverage for 8 new stat-test axes."""

from __future__ import annotations

import pytest

from macrocast.registry.build import get_axis_registry_entry

NEW_AXES = (
    "equal_predictive",
    "nested",
    "cpa_instability",
    "multiple_model",
    "density_interval",
    "direction",
    "residual_diagnostics",
    "test_scope",
)

OPERATIONAL_VALUES = {
    "equal_predictive": {"none", "dm", "dm_hln", "dm_modified", "paired_t_on_loss_diff", "wilcoxon_signed_rank"},
    "nested": {"none", "cw", "enc_new", "mse_f", "mse_t"},
    "cpa_instability": {"none", "cpa", "rossi", "rolling_dm"},
    "multiple_model": {"none", "reality_check", "spa", "mcs"},
    "density_interval": {"none"},
    "direction": {"none", "pesaran_timmermann", "binomial_hit"},
    "residual_diagnostics": {
        "none", "mincer_zarnowitz", "ljung_box", "arch_lm", "bias_test", "diagnostics_full",
        "autocorrelation_of_errors",
    },
    "test_scope": {"per_target", "per_horizon", "per_model_pair"},
}

PLANNED_PRESENT = {
    "equal_predictive": set(),
    "nested": {"forecast_encompassing_nested"},
    "cpa_instability": {"fluctuation_test", "chow_break_forecast", "cusum_on_loss"},
    "multiple_model": {"stepwise_mcs", "bootstrap_best_model"},
    "density_interval": {
        "PIT_uniformity",
        "berkowitz",
        "kupiec",
        "christoffersen_unconditional",
        "christoffersen_independence",
        "christoffersen_conditional",
        "interval_coverage",
    },
    "direction": {"mcnemar", "roc_comparison"},
    "residual_diagnostics": {"serial_dependence_loss_diff"},
    "test_scope": {"full_grid_pairwise", "benchmark_vs_all", "regime_specific_tests", "subsample_tests"},
}


@pytest.mark.parametrize("axis_name", NEW_AXES)
def test_axis_is_registered_on_layer_6_stat_tests(axis_name: str) -> None:
    entry = get_axis_registry_entry(axis_name)
    assert entry.axis_name == axis_name
    assert entry.layer == "6_stat_tests"
    assert entry.axis_type == "enum"
    assert entry.default_policy == "fixed"


@pytest.mark.parametrize("axis_name", NEW_AXES)
def test_axis_has_expected_operational_values(axis_name: str) -> None:
    entry = get_axis_registry_entry(axis_name)
    operational = {
        value for value, status in entry.current_status.items() if status == "operational"
    }
    assert operational == OPERATIONAL_VALUES[axis_name], (
        f"axis {axis_name!r}: operational set mismatch; got {operational!r}"
    )


@pytest.mark.parametrize("axis_name", NEW_AXES)
def test_axis_has_expected_planned_values(axis_name: str) -> None:
    entry = get_axis_registry_entry(axis_name)
    planned = {
        value for value, status in entry.current_status.items() if status == "planned"
    }
    assert planned == PLANNED_PRESENT[axis_name]


def test_legacy_stat_test_axis_still_registered() -> None:
    entry = get_axis_registry_entry("stat_test")
    assert entry.layer == "6_stat_tests"
    assert "dm" in entry.allowed_values
    assert entry.current_status["dm"] == "operational"
