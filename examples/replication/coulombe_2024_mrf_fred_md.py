"""Replication: Coulombe (2024) Macroeconomic Random Forest on FRED-MD.

Compares the GTVP MRF estimator (operational since v0.2 #187) against a
ridge baseline on the FRED-MD shipped sample. Writes a manifest with
per-cell sink hashes and per-model figures into ``./out/coulombe_2024/``.

Usage:

    python3 examples/replication/coulombe_2024_mrf_fred_md.py
    # then:
    python3 -c "import macrocast; r = macrocast.replicate('out/coulombe_2024/manifest.json'); print(r.sink_hashes_match)"
"""
from __future__ import annotations

from pathlib import Path

import macrocast


RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 1989  # Hamilton (1989) MS year, in honour of v0.2 #195
1_data:
  fixed_axes:
    custom_source_policy: official_only
    dataset: fred_md
    frequency: monthly
    horizon_set: standard_md
    target_structure: single_target
    variable_universe: all_variables
    sample_start_rule: max_balanced
    sample_end_rule: latest_available
  leaf_config:
    target: INDPRO
2_preprocessing:
  fixed_axes:
    transform_policy: apply_official_tcode
    outlier_policy: mccracken_ng_iqr
    outlier_action: flag_as_nan
    imputation_policy: em_factor
    frame_edge_policy: truncate_to_balanced
3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 6}, inputs: [src_X]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_ridge
      type: step
      op: fit_model
      params:
        family: ridge
        alpha: 1.0
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
      inputs: [src_X, src_y]
      is_benchmark: true
    - id: fit_mrf
      type: step
      op: fit_model
      params:
        family: macroeconomic_random_forest
        n_estimators: 100
        max_depth: 6
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
      inputs: [src_X, src_y]
    - {id: predict_ridge, type: step, op: predict, inputs: [fit_ridge, src_X]}
    - {id: predict_mrf, type: step, op: predict, inputs: [fit_mrf, src_X]}
  sinks:
    l4_forecasts_v1: predict_mrf
    l4_model_artifacts_v1: fit_mrf
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]
6_statistical_tests:
  enabled: true
  fixed_axes:
    L6_A_equal_predictive:
      enabled: true
      equal_predictive_test: dm_diebold_mariano
      hln_correction: true
8_output:
  fixed_axes:
    export_format: json_csv
    saved_objects: [forecasts, metrics, ranking, tests]
    artifact_granularity: per_cell
    naming_convention: descriptive
  leaf_config:
    output_directory: out/coulombe_2024
"""


def main() -> None:
    out = Path("out/coulombe_2024")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Running Coulombe (2024) MRF vs ridge replication on FRED-MD INDPRO...")
    result = macrocast.run(RECIPE, output_directory=out)
    print(f"  cells:    {len(result.cells)}")
    print(f"  duration: {result.duration_seconds:.1f}s")
    cell = result.cells[0]
    if not cell.succeeded:
        print(f"  FAILED:   {cell.error}")
        return
    metrics = cell.runtime_result.artifacts["l5_evaluation_v1"].metrics_table
    if not metrics.empty:
        print("\nPer-model MSE:")
        print(metrics[["model_id", "mse", "rmse", "mae"]].to_string(index=False))
    rep = macrocast.replicate(out / "manifest.json")
    print(f"\nReplication: sink_hashes_match = {rep.sink_hashes_match}")


if __name__ == "__main__":
    main()
