"""End-to-end tests for horizon_target_construction (1.2.4).

Each operational value (future_level_y_t_plus_h, future_diff, future_logdiff,
cumulative_growth_to_h) must:
1. compile to execution_status == "executable",
2. produce rows carrying `horizon_target_construction` + level-scale provenance,
3. produce metrics (error / squared_error / benchmark_error) on the declared
   construction scale, not on the level scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from macrocast.compiler.build import compile_recipe_dict
from macrocast.compiler.build import run_compiled_recipe


OPERATIONAL_CONSTRUCTIONS = (
    "future_level_y_t_plus_h",
    "future_diff",
    "future_logdiff",
)


def _recipe(construction: str) -> dict:
    axes_1 = {
        "dataset": "fred_md",
        "info_set": "revised",
        "task": "single_target_point_forecast",
        "horizon_target_construction": construction,
    }
    return {
        "recipe_id": f"htc-{construction}",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": axes_1,
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
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
                "feature_builder": "autoreg_lagged_target",
                "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {
                "manifest_mode": "full",
                "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5},
            }},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


def _run(construction: str, tmp_path: Path):
    recipe = _recipe(construction)
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable", (
        f"{construction} did not compile as executable"
    )
    return run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )


def _predictions(execution) -> pd.DataFrame:
    import pandas as pd
    return pd.read_csv(Path(execution.artifact_dir) / "predictions.csv")



@pytest.mark.parametrize("construction", OPERATIONAL_CONSTRUCTIONS)
def test_horizon_target_construction_executes(construction: str, tmp_path: Path) -> None:
    execution = _run(construction, tmp_path)
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    # Compiler records the construction in the data_task spec
    assert manifest["data_task_spec"]["horizon_target_construction"] == construction


def test_future_level_matches_level_scale_metrics(tmp_path: Path) -> None:
    """Default construction keeps y_pred / y_true on level scale; level-preserved
    copies should equal the primary fields."""
    execution = _run("future_level_y_t_plus_h", tmp_path)
    predictions = _predictions(execution)
    assert not predictions.empty
    # Level-preserved columns should be present and identical to primary fields
    for col in ("y_true_level", "y_pred_level", "benchmark_pred_level"):
        assert col in predictions.columns, f"missing {col}"
    assert np.allclose(predictions["y_true"], predictions["y_true_level"], equal_nan=True)
    assert np.allclose(predictions["y_pred"], predictions["y_pred_level"], equal_nan=True)
    # Construction column records the value used
    assert (predictions["horizon_target_construction"] == "future_level_y_t_plus_h").all()


def test_future_diff_shifts_metrics_by_anchor(tmp_path: Path) -> None:
    """future_diff sets y_pred = y_pred_level - y_anchor and y_true similarly;
    numerically the error is unchanged but the reported y_true / y_pred move."""
    execution = _run("future_diff", tmp_path)
    predictions = _predictions(execution)
    assert not predictions.empty
    # y_true / y_pred on diff scale differ from the level copy by a common anchor
    # so error = y_true - y_pred equals level-error; verify that invariant.
    diff_err = predictions["y_true"] - predictions["y_pred"]
    level_err = predictions["y_true_level"] - predictions["y_pred_level"]
    assert np.allclose(diff_err, level_err, atol=1e-9, equal_nan=True)
    # But the reported y_true on diff scale must not equal level y_true (they
    # differ by the anchor).
    assert not np.allclose(
        predictions["y_true"].dropna(),
        predictions["y_true_level"].dropna(),
        atol=1e-9,
    )
    assert (predictions["horizon_target_construction"] == "future_diff").all()


def test_future_logdiff_changes_error_scale(tmp_path: Path) -> None:
    """log-growth construction genuinely changes the error scale
    (unlike diff, where anchor cancels)."""
    level = _predictions(_run("future_level_y_t_plus_h", tmp_path))
    logdiff = _predictions(_run("future_logdiff", tmp_path / "logdiff"))
    assert not logdiff.empty
    # Level-scale values are preserved
    assert np.allclose(level["y_pred"], logdiff["y_pred_level"], equal_nan=True)
    # log-diff error = log(y_true/y_anchor) - log(y_pred/y_anchor) = log(y_true/y_pred)
    #                = log(y_true_level) - log(y_pred_level)
    expected = np.log(logdiff["y_true_level"]) - np.log(logdiff["y_pred_level"])
    assert np.allclose(logdiff["error"], expected, atol=1e-9, equal_nan=True)
    assert (logdiff["horizon_target_construction"] == "future_logdiff").all()


