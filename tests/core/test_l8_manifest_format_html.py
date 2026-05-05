"""Issue #209 -- yaml ``manifest_format`` and ``html_report`` export.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import macroforecast


_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
2_preprocessing:
  fixed_axes: {transform_policy: no_transform, outlier_policy: none, imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}
3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_model
      type: step
      op: fit_model
      params: {family: ridge, alpha: 0.1, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_model, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {primary_metric: mse, point_metrics: [mse]}
8_output:
  fixed_axes:
    export_format: __EXPORT_FORMAT__
    saved_objects: [forecasts, metrics]
    artifact_granularity: per_cell
    naming_convention: descriptive
    manifest_format: __MANIFEST_FORMAT__
  leaf_config:
    output_directory: __PLACEHOLDER__
"""


def _run(tmp_path: Path, manifest_format: str = "json", export_format: str = "json_csv") -> Path:
    recipe = (
        _RECIPE.replace("__PLACEHOLDER__", str(tmp_path))
        .replace("__MANIFEST_FORMAT__", manifest_format)
        .replace("__EXPORT_FORMAT__", export_format)
    )
    macroforecast.run(recipe, output_directory=tmp_path)
    return tmp_path


def test_manifest_format_yaml_writes_yaml_file(tmp_path):
    pytest.importorskip("yaml")
    out = _run(tmp_path, manifest_format="yaml")
    assert (out / "manifest.yaml").exists()
    assert not (out / "manifest.json").exists()
    text = (out / "manifest.yaml").read_text()
    assert "recipe_root" in text and "provenance" in text


def test_manifest_format_json_lines(tmp_path):
    out = _run(tmp_path, manifest_format="json_lines")
    target = out / "manifest.jsonl"
    assert target.exists()
    lines = target.read_text().strip().splitlines()
    # First line = study-level, subsequent lines = one per cell.
    assert len(lines) >= 2


def test_manifest_format_json_default(tmp_path):
    out = _run(tmp_path)
    assert (out / "manifest.json").exists()


def test_export_format_html_report_writes_report_html(tmp_path):
    out = _run(tmp_path, export_format="html_report")
    report = out / "report.html"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "macroforecast study report" in text
    assert "<html" in text and "</html>" in text
    # Recipe digest section pulls the L1 target.
    assert "<code>y</code>" in text
