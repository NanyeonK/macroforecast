"""Issue #206 -- L8 export ``compression`` axis (gzip / zip).

Pins:

* ``compression: gzip`` writes each export as ``<name>.<ext>.gz``.
* ``compression: zip`` bundles every export into ``<output_dir>.zip``.
* ``compression: none`` (default) is unchanged.
* ``compression_level`` leaf_config flows through.
"""
from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import macrocast


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
    export_format: json_csv
    saved_objects: [forecasts, metrics]
    artifact_granularity: per_cell
    naming_convention: descriptive
    compression: __COMPRESSION__
  leaf_config:
    output_directory: __PLACEHOLDER__
    compression_level: 6
"""


def _run(tmp_path: Path, compression: str) -> Path:
    recipe = _RECIPE.replace("__PLACEHOLDER__", str(tmp_path)).replace("__COMPRESSION__", compression)
    macrocast.run(recipe, output_directory=tmp_path)
    return tmp_path


def test_gzip_compression_writes_gz_files(tmp_path):
    out = _run(tmp_path, "gzip")
    gz_files = list(out.rglob("*.gz"))
    assert gz_files, "expected at least one .gz file under output dir"
    # And the original uncompressed forecasts.csv is gone.
    assert not (out / "cell_001" / "forecasts.csv").exists()
    # The gzip file is readable.
    sample = gz_files[0]
    with gzip.open(sample, "rt", encoding="utf-8") as fh:
        content = fh.read()
    assert content  # non-empty


def test_zip_compression_writes_single_bundle(tmp_path):
    out = _run(tmp_path, "zip")
    bundles = list(out.rglob("*.zip"))
    assert len(bundles) == 1, f"expected exactly one .zip bundle, found {bundles}"
    bundle = bundles[0]
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert names, "zip bundle is empty"
    # Originals should be removed (only the manifest.json + recipe.json
    # written *after* compression remain alongside).
    assert not (out / "cell_001" / "forecasts.csv").exists()


def test_no_compression_default_is_unchanged(tmp_path):
    out = _run(tmp_path, "none")
    assert (out / "cell_001" / "forecasts.csv").exists()
    assert not list(out.rglob("*.gz"))
    # No archive bundle appears in the no-op path.
    assert not list(out.rglob(f"{out.name}.zip"))
