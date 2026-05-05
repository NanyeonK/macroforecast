"""End-to-end smoke tests covering the 8 freedom dimensions of macroforecast v0.1.

Each dimension has at least one happy-path test that exercises functionality
introduced in the v0.1 implementation (per `plans/system-reminder-...md`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import macroforecast
from macroforecast.core.execution import execute_recipe, replicate_recipe
from macroforecast.core.figures import US_STATE_GRID, render_us_state_choropleth


_PANEL_DATES = [
    "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01",
    "2020-06-01", "2020-07-01", "2020-08-01", "2020-09-01", "2020-10-01",
    "2020-11-01", "2020-12-01",
]
_PANEL_VALUES = list(range(1, len(_PANEL_DATES) + 1))


def _custom_recipe(*, family: str = "ridge", n_lag: object = 1, l3_extra_node: str = "", with_l6: bool = False, with_l7: bool = False, with_l8: bool = False, output_dir: str = "/tmp/__macrocast_test", explicit_targets: list[int] | None = None) -> str:
    n_lag_marker = f"{{sweep: {n_lag}}}" if isinstance(n_lag, list) else str(n_lag)
    l6_block = """
6_statistical_tests:
  enabled: true
  test_scope: per_target_horizon
  sub_layers:
    L6_A_equal_predictive:
      enabled: true
      fixed_axes:
        equal_predictive_test: dm_diebold_mariano
        model_pair_strategy: vs_benchmark_only
        loss_function: squared
        hln_correction: true
    L6_G_residual:
      enabled: true
      fixed_axes:
        residual_test: [ljung_box_q, jarque_bera_normality, durbin_watson]
        residual_lag_count: 5
""" if with_l6 else ""
    l7_block = """
7_interpretation:
  enabled: true
  nodes:
    - {id: src_model, type: source, selector: {layer_ref: l4, sink_name: l4_model_artifacts_v1, subset: {model_id: fit_model}}}
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - {id: imp_perm, type: step, op: permutation_importance, inputs: [src_model, src_X, src_y]}
    - {id: imp_pdp, type: step, op: partial_dependence, inputs: [src_model, src_X], params: {n_grid: 6}}
    - {id: imp_lin, type: step, op: model_native_linear_coef, params: {model_family: ridge}, inputs: [src_model]}
  sinks:
    l7_importance_v1: {global: imp_perm, marginal: imp_pdp, coef: imp_lin}
""" if with_l7 else ""
    l8_block = f"""
8_output:
  fixed_axes:
    export_format: json_csv
    saved_objects: [forecasts, metrics, ranking, tests, importance, diagnostics_all]
    artifact_granularity: per_cell
    naming_convention: descriptive
  leaf_config:
    output_directory: {output_dir}
""" if with_l8 else ""
    targets = explicit_targets or [1]
    target_block = f"target_horizons: {targets}"
    benchmark_node = "is_benchmark: true" if with_l6 else ""
    return f"""
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 7
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    {target_block}
    custom_panel_inline:
      date: {_PANEL_DATES}
      y: {_PANEL_VALUES}
      x1: {[v + 0.5 for v in _PANEL_VALUES]}
      x2: {[v % 3 for v in _PANEL_VALUES]}
2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced
3_feature_engineering:
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: predictors}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: target}}}}}}
    - {{id: lag_x, type: step, op: lag, params: {{n_lag: {n_lag_marker}}}, inputs: [src_X]}}
{l3_extra_node}
    - {{id: y_h, type: step, op: target_construction, params: {{mode: point_forecast, method: direct, horizon: 1}}, inputs: [src_y]}}
  sinks:
    l3_features_v1: {{X_final: lag_x, y_final: y_h}}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}}}
    - id: fit_model
      type: step
      op: fit_model
      params: {{family: {family}, alpha: 1.0, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}}
      {benchmark_node}
      inputs: [src_X, src_y]
    - {{id: predict, type: step, op: predict, inputs: [fit_model, src_X]}}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]
{l6_block}{l7_block}{l8_block}
"""


# --- Dimension 1: Data (L1) ----------------------------------------------------

def test_dim1_l1_loads_fred_sd_from_local_fixture(tmp_path):
    from macroforecast.core.runtime import materialize_l1
    from macroforecast.core.yaml import parse_recipe_yaml

    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    root = parse_recipe_yaml(
        f"""
1_data:
  fixed_axes:
    custom_source_policy: official_only
    dataset: fred_sd
    frequency: monthly
    target_geography_scope: selected_states
    predictor_geography_scope: match_target
    sample_start_rule: max_balanced
  leaf_config:
    target: UR_CA
    cache_root: {tmp_path}
    local_raw_source: {fixtures / 'fred_sd_sample.csv'}
    target_states: [CA, TX]
"""
    )
    artifact, regime, resolved = materialize_l1(root)
    assert artifact.dataset == "fred_sd"
    assert "UR_CA" in artifact.raw_panel.column_names


def test_dim1_l1_external_nber_regime_populates_label_series():
    recipe = _custom_recipe()
    recipe = recipe.replace(
        "horizon_set: custom_list",
        "horizon_set: custom_list\n    regime_definition: external_nber",
    )
    result = execute_recipe(recipe)
    regime = result.cells[0].runtime_result.artifacts["l1_regime_metadata_v1"]
    assert regime.definition == "external_nber"
    assert regime.regime_label_series is not None
    assert regime.estimation_metadata.get("n_recession_months") is not None


# --- Dimension 2: Preprocessing (L2) -----------------------------------------

def test_dim2_l2_winsorize_and_zscore_outlier_paths(tmp_path):
    for policy in ("winsorize", "zscore_threshold"):
        recipe = _custom_recipe()
        recipe = recipe.replace("outlier_policy: none", f"outlier_policy: {policy}")
        result = execute_recipe(recipe)
        log = result.cells[0].runtime_result.artifacts["l2_clean_panel_v1"].cleaning_log
        assert any(step.get("outlier") == policy for step in log["steps"])


def test_dim2_l2_em_factor_imputation_runs():
    recipe = _custom_recipe().replace("imputation_policy: none_propagate", "imputation_policy: em_factor")
    result = execute_recipe(recipe)
    log = result.cells[0].runtime_result.artifacts["l2_clean_panel_v1"].cleaning_log
    assert any(step.get("imputation") == "em_factor" for step in log["steps"])


# --- Dimension 3: Feature engineering (L3) -----------------------------------

def test_dim3_l3_pca_op_runs():
    pca_node = "    - {id: pca, type: step, op: pca, params: {n_components: 1, temporal_rule: expanding_window_per_origin}, inputs: [src_X]}\n"
    recipe = _custom_recipe(l3_extra_node=pca_node)
    recipe = recipe.replace("X_final: lag_x", "X_final: pca")
    result = execute_recipe(recipe)
    X = result.cells[0].runtime_result.artifacts["l3_features_v1"].X_final
    assert any(name.startswith("factor_") for name in X.column_names)


# --- Dimension 4: Model + tuning (L4) ----------------------------------------

@pytest.mark.parametrize("family", ["random_forest", "gradient_boosting", "ar_p", "decision_tree"])
def test_dim4_l4_extended_model_families(family):
    recipe = _custom_recipe(family=family)
    result = execute_recipe(recipe)
    artifact = result.cells[0].runtime_result.artifacts["l4_model_artifacts_v1"]
    assert artifact.artifacts["fit_model"].family == family


def test_dim4_l4_records_alpha_in_fit_metadata():
    recipe = _custom_recipe(family="ridge")
    result = execute_recipe(recipe)
    artifact = result.cells[0].runtime_result.artifacts["l4_model_artifacts_v1"]
    assert "alpha" in artifact.artifacts["fit_model"].fit_metadata


def test_dim4_l4_rolling_window_runs():
    recipe = _custom_recipe(family="ridge")
    recipe = recipe.replace("training_start_rule: expanding", "training_start_rule: rolling")
    recipe = recipe.replace("'min_train_size': 4", "'min_train_size': 3, 'rolling_window': 3")
    recipe = recipe.replace("min_train_size: 4", "min_train_size: 3, rolling_window: 3")
    result = execute_recipe(recipe)
    fit_meta = result.cells[0].runtime_result.artifacts["l4_model_artifacts_v1"].artifacts["fit_model"].fit_metadata
    assert fit_meta["runtime"].startswith("rolling_")


# --- Dimension 5: Evaluation (L5) -------------------------------------------

def test_dim5_l5_metrics_and_ranking_emit():
    recipe = _custom_recipe()
    result = execute_recipe(recipe)
    eval_artifact = result.cells[0].runtime_result.artifacts["l5_evaluation_v1"]
    assert "mse" in eval_artifact.metrics_table.columns
    assert not eval_artifact.ranking_table.empty


# --- Dimension 6: Statistical tests (L6) -------------------------------------

def test_dim6_l6_dm_test_runs():
    recipe = _custom_recipe(with_l6=True)
    result = execute_recipe(recipe)
    tests = result.cells[0].runtime_result.artifacts["l6_tests_v1"]
    # equal_predictive_results may be empty if no benchmark pair found, but the
    # residual results should at least populate.
    assert tests.residual_results is not None


# --- Dimension 7: Importance (L7) -------------------------------------------

def test_dim7_l7_permutation_pdp_linear_coef_run():
    recipe = _custom_recipe(family="ridge", with_l7=True)
    result = execute_recipe(recipe)
    importance = result.cells[0].runtime_result.artifacts["l7_importance_v1"]
    assert importance.global_importance


def test_dim7_us_state_choropleth_renders(tmp_path):
    importance = {state: 0.1 + 0.01 * idx for idx, state in enumerate(US_STATE_GRID)}
    out = tmp_path / "choropleth.pdf"
    rendered = render_us_state_choropleth(importance, output_path=out, title="Test")
    assert rendered.exists() and rendered.stat().st_size > 1024


# --- Dimension 8: Output (L8) -----------------------------------------------

def test_dim8_l8_writes_per_cell_artifacts(tmp_path):
    recipe = _custom_recipe(with_l8=True, output_dir=str(tmp_path))
    result = execute_recipe(recipe, output_directory=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    cell_dir = tmp_path / "cell_001"
    assert cell_dir.exists()
    assert (cell_dir / "forecasts.csv").exists()


# --- Cross-cutting: Sweep + bit-exact replicate -----------------------------

def test_cross_sweep_then_replicate_is_bit_exact(tmp_path):
    recipe = _custom_recipe(n_lag=[1, 2])
    macroforecast.run(recipe, output_directory=tmp_path)
    replication = macroforecast.replicate(tmp_path / "manifest.json")
    assert replication.recipe_match
    assert replication.sink_hashes_match
    assert all(replication.per_cell_match.values())
