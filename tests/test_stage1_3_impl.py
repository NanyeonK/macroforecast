"""End-to-end tests for 1.3 implementation wiring.

Covers:
- training_start_rule=fixed_start + leaf_config.training_start_date (+ missing-date guard).
- min_train_size axis dispatch via raw/windowing.py (5 rules).
- oos_period=recession_only_oos / expansion_only_oos filters via NBER fixture.
- overlap_handling=evaluate_with_hac compatibility gate (non-HAC Layer 6 test blocked).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from macrocast.compiler.build import compile_recipe_dict, run_compiled_recipe
from macrocast.execution.nber import is_recession, is_expansion


def _recipe(
    *,
    training_start_rule: str | None = None,
    training_start_date: str | None = None,
    min_train_size: str | None = None,
    oos_period: str | None = None,
    overlap_handling: str | None = None,
    stat_test: str | None = None,
) -> dict:
    axes_1 = {
        "dataset": "fred_md",
        "information_set_type": "final_revised_data",
        "target_structure": "single_target",
    }
    if training_start_rule is not None:
        axes_1["training_start_rule"] = training_start_rule
    if min_train_size is not None:
        axes_1["min_train_size"] = min_train_size

    axes_4 = {"primary_metric": "msfe"}
    axes_6: dict[str, str] = {}
    if overlap_handling is not None:
        axes_6["overlap_handling"] = overlap_handling
    if stat_test and stat_test != "none":
        stat_axis = {"dm": "equal_predictive", "dm_hln": "equal_predictive", "mse_f": "nested"}[stat_test]
        axes_6[stat_axis] = stat_test
    if oos_period is not None:
        axes_4["oos_period"] = oos_period

    leaf = {"target": "INDPRO", "horizons": [1]}
    if training_start_date is not None:
        leaf["training_start_date"] = training_start_date

    return {
        "recipe_id": "s13-impl-test",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_forecast_run"}},
            "1_data_task": {"fixed_axes": axes_1, "leaf_config": leaf},
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level",
                "x_transform_policy": "raw_level",
                "tcode_policy": "raw_only",
                "target_missing_policy": "none",
                "x_missing_policy": "none",
                "target_outlier_policy": "none",
                "x_outlier_policy": "none",
                "scaling_policy": "none",
                "dimensionality_reduction_policy": "none",
                "feature_selection_policy": "none",
                "preprocess_order": "none",
                "preprocess_fit_scope": "not_applicable",
                "inverse_transform_policy": "none",
                "evaluation_scale": "raw_level",
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding",
                "benchmark_family": "historical_mean",
                "feature_builder": "target_lag_features",
                "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": axes_4},
            "5_output_provenance": {"leaf_config": {
                "manifest_mode": "full",
                "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5},
            }},
            "6_stat_tests": {"fixed_axes": axes_6},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


# ---------- training_start_rule ----------

def test_fixed_start_requires_training_start_date() -> None:
    r = compile_recipe_dict(_recipe(training_start_rule="fixed_start"))
    assert r.compiled.execution_status == "blocked_by_incompatibility"
    assert any("training_start_date" in msg for msg in r.manifest.get("blocked_reasons", []))


def test_fixed_start_with_date_executes(tmp_path: Path) -> None:
    recipe = _recipe(training_start_rule="fixed_start", training_start_date="1965-01-01")
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        r.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    predictions = pd.read_csv(Path(execution.artifact_dir) / "predictions.csv")
    assert not predictions.empty
    # Training start cannot precede the fixed date: train_start_date column on every
    # row must be >= 1965-01-01.
    train_starts = pd.to_datetime(predictions["train_start_date"])
    assert (train_starts >= pd.Timestamp("1965-01-01")).all()


def test_earliest_possible_is_default_when_rule_unset() -> None:
    r = compile_recipe_dict(_recipe())
    assert r.compiled.execution_status == "executable"
    assert r.manifest["training_spec"]["training_start_rule"] == "earliest_possible"
    assert "training_start_rule" not in r.manifest["data_task_spec"]


# ---------- min_train_size ----------

@pytest.mark.parametrize(
    "rule",
    ["fixed_n_obs", "fixed_years", "model_specific_min_train",
     "target_specific_min_train", "horizon_specific_min_train"],
)
def test_min_train_size_rules_compile(rule: str) -> None:
    r = compile_recipe_dict(_recipe(min_train_size=rule))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["training_spec"]["min_train_size"] == rule
    assert "min_train_size" not in r.manifest["data_task_spec"]


def test_min_train_size_axis_propagates_to_manifest() -> None:
    """Every min_train_size rule must land in training_spec for runtime dispatch."""
    for rule in ["fixed_n_obs", "fixed_years", "model_specific_min_train",
                 "target_specific_min_train", "horizon_specific_min_train"]:
        r = compile_recipe_dict(_recipe(min_train_size=rule))
        assert r.manifest["training_spec"]["min_train_size"] == rule
        assert "min_train_size" not in r.manifest["data_task_spec"]


# ---------- oos_period regime filter ----------

def test_nber_recession_fixture_known_dates() -> None:
    assert is_recession("2008-09-01")  # global financial crisis
    assert is_recession("2020-03-01")  # COVID
    assert not is_recession("2015-06-01")  # mid-expansion
    assert is_expansion("2015-06-01")


def test_recession_only_oos_compiles_and_filters_unit() -> None:
    """Compile is executable. Filter logic is unit-tested via filter_origins_by_regime
    since the small synthetic fixture has no recession dates in its index."""
    from macrocast.execution.nber import filter_origins_by_regime
    r = compile_recipe_dict(_recipe(oos_period="recession_only_oos"))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["evaluation_spec"]["oos_period"] == "recession_only_oos"
    # Unit: synthetic origins across recession and expansion dates
    idx = pd.to_datetime(["2000-01-01", "2008-09-01", "2015-06-01", "2020-03-01"])
    idx = pd.DatetimeIndex(idx)
    plan = [(i, 0, i) for i in range(len(idx))]
    kept = filter_origins_by_regime(plan, index=idx, regime="recession_only_oos")
    kept_dates = [idx[item[0]] for item in kept]
    assert all(is_recession(d) for d in kept_dates)
    assert pd.Timestamp("2008-09-01") in kept_dates
    assert pd.Timestamp("2020-03-01") in kept_dates
    assert pd.Timestamp("2015-06-01") not in kept_dates


def test_expansion_only_oos_compiles_and_filters_unit() -> None:
    from macrocast.execution.nber import filter_origins_by_regime
    r = compile_recipe_dict(_recipe(oos_period="expansion_only_oos"))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["evaluation_spec"]["oos_period"] == "expansion_only_oos"
    idx = pd.DatetimeIndex(pd.to_datetime(["2000-01-01", "2008-09-01", "2015-06-01", "2020-03-01"]))
    plan = [(i, 0, i) for i in range(len(idx))]
    kept = filter_origins_by_regime(plan, index=idx, regime="expansion_only_oos")
    kept_dates = [idx[item[0]] for item in kept]
    assert all(is_expansion(d) for d in kept_dates)
    assert pd.Timestamp("2015-06-01") in kept_dates
    assert pd.Timestamp("2008-09-01") not in kept_dates


def test_all_oos_data_is_default_no_filter(tmp_path: Path) -> None:
    r = compile_recipe_dict(_recipe())
    assert r.manifest["evaluation_spec"]["oos_period"] == "all_oos_data"
    assert r.manifest["data_task_spec"]["oos_period"] == "all_oos_data"
    execution = run_compiled_recipe(
        r.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    predictions = pd.read_csv(Path(execution.artifact_dir) / "predictions.csv")
    origins = pd.to_datetime(predictions["origin_date"])
    # With no filter, both recession and expansion origins can appear in a long sample.
    # The fixture is short so we just assert non-empty and that the default runs.
    assert len(origins) > 0


def test_legacy_layer1_oos_period_still_compiles_as_compatibility_alias() -> None:
    recipe = _recipe()
    recipe["path"]["1_data_task"]["fixed_axes"]["oos_period"] = "recession_only_oos"
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"
    assert r.manifest["evaluation_spec"]["oos_period"] == "recession_only_oos"
    assert r.manifest["data_task_spec"]["oos_period"] == "recession_only_oos"


# ---------- overlap_handling HAC gate ----------

def test_evaluate_with_hac_blocks_non_hac_stat_test() -> None:
    r = compile_recipe_dict(_recipe(
        overlap_handling="evaluate_with_hac",
        stat_test="mse_f",
    ))
    assert r.compiled.execution_status == "blocked_by_incompatibility"
    assert any("evaluate_with_hac" in msg and "mse_f" in msg
               for msg in r.manifest.get("blocked_reasons", []))


def test_evaluate_with_hac_allows_dm_hln() -> None:
    r = compile_recipe_dict(_recipe(
        overlap_handling="evaluate_with_hac",
        stat_test="dm_hln",
    ))
    assert r.compiled.execution_status == "executable"


def test_evaluate_with_hac_allows_no_stat_test() -> None:
    """No stat test -> no HAC needed; the gate should not block."""
    r = compile_recipe_dict(_recipe(
        overlap_handling="evaluate_with_hac",
        stat_test="none",
    ))
    assert r.compiled.execution_status == "executable"


def test_allow_overlap_default_with_any_layer6_test() -> None:
    r = compile_recipe_dict(_recipe(stat_test="dm"))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["overlap_handling"] == "allow_overlap"
    assert r.manifest["stat_test_spec"]["overlap_handling"] == "allow_overlap"
