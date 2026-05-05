"""End-to-end smoke check for v0.1 of macroforecast.

Run from repo root:
    python3 scripts/v01_smoke_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import macroforecast
from macroforecast.core.figures import US_STATE_GRID, render_us_state_choropleth


def _custom_recipe(*, n_lag, family="ridge", with_l7=False, with_l8=True, output_dir):
    n_lag_marker = f"{{sweep: {n_lag}}}" if isinstance(n_lag, list) else str(n_lag)
    panel_dates = [
        "2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01",
        "2018-07-01", "2018-08-01", "2018-09-01", "2018-10-01", "2018-11-01", "2018-12-01",
        "2019-01-01", "2019-02-01", "2019-03-01", "2019-04-01", "2019-05-01", "2019-06-01",
        "2019-07-01", "2019-08-01", "2019-09-01", "2019-10-01", "2019-11-01", "2019-12-01",
    ]
    n = len(panel_dates)
    y_values = [10.0 + i * 0.5 for i in range(n)]
    x1 = [v + 0.3 for v in y_values]
    x2 = [(i % 4) * 1.5 for i in range(n)]
    x3 = [(i % 7) - 3 for i in range(n)]
    l7_block = """
7_interpretation:
  enabled: true
  nodes:
    - {id: src_model, type: source, selector: {layer_ref: l4, sink_name: l4_model_artifacts_v1, subset: {model_id: fit_model}}}
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - {id: imp_perm, type: step, op: permutation_importance, inputs: [src_model, src_X, src_y]}
    - {id: imp_lin, type: step, op: model_native_linear_coef, params: {model_family: ridge}, inputs: [src_model]}
  sinks:
    l7_importance_v1: {global: imp_perm, coef: imp_lin}
""" if with_l7 else ""
    l8_block = f"""
8_output:
  fixed_axes:
    export_format: json_csv
    saved_objects: [forecasts, metrics, ranking, importance, diagnostics_all]
    artifact_granularity: per_cell
    naming_convention: descriptive
  leaf_config:
    output_directory: {output_dir}
""" if with_l8 else ""
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
    target_horizons: [1]
    custom_panel_inline:
      date: {panel_dates}
      y: {y_values}
      x1: {x1}
      x2: {x2}
      x3: {x3}
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
      params: {{family: {family}, alpha: 1.0, min_train_size: 6, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}}
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
{l7_block}{l8_block}
"""


def main() -> int:
    out_root = Path("/tmp/v01_smoke_check_out")
    out_root.mkdir(exist_ok=True)
    print(f"== macroforecast {macroforecast.__version__} smoke check ==\n")

    print("[1/4] single-cell run + manifest.json")
    out1 = out_root / "single_cell"
    result = macroforecast.run(_custom_recipe(n_lag=2, output_dir=str(out1)), output_directory=out1)
    cell = result.cells[0]
    print(f"  cell_id={cell.cell_id} succeeded={cell.succeeded}")
    print(f"  sinks: {sorted(cell.sink_hashes)}")
    print(f"  manifest written: {(out1 / 'manifest.json').exists()}")
    metrics = cell.runtime_result.artifacts['l5_evaluation_v1'].metrics_table
    print(f"  L5 metrics: {metrics.to_dict('records')[0]}\n")

    print("[2/4] bit-exact replicate")
    rep = macroforecast.replicate(out1 / "manifest.json")
    print(f"  recipe_match={rep.recipe_match} sink_hashes_match={rep.sink_hashes_match}")
    print(f"  per_cell_match={rep.per_cell_match}\n")

    print("[3/4] multi-cell sweep ({sweep: [1,2,3]} on n_lag)")
    out2 = out_root / "sweep"
    sweep = macroforecast.run(_custom_recipe(n_lag=[1, 2, 3], output_dir=str(out2)), output_directory=out2)
    print(f"  cells: {[c.cell_id for c in sweep.cells]}")
    for c in sweep.cells:
        mse = c.runtime_result.artifacts['l5_evaluation_v1'].metrics_table['mse'].iloc[0]
        print(f"    {c.cell_id}: mse={mse:.6f}")
    print()

    print("[4/4] FRED-SD US state choropleth")
    sample = {state: 0.05 + 0.01 * idx for idx, state in enumerate(US_STATE_GRID)}
    map_path = out_root / "state_choropleth.pdf"
    render_us_state_choropleth(sample, output_path=map_path, title="v0.1 smoke choropleth")
    print(f"  rendered {map_path} size={map_path.stat().st_size} bytes\n")

    print("== smoke check complete ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
