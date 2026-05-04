"""Issue #208 -- L8 manifest must populate every design-listed provenance
field, not just the 4-5 that v0.1 carried.

Pins (in addition to the v0.1 baseline):

* git_commit_sha + git_branch_name (best-effort; tolerates non-git checkouts)
* r_version + julia_version (None when binary missing)
* dependency_lockfile_content (truncated)
* random_seed_used (from L0 leaf_config OR seeded_reproducible default)
* runtime_environment with cpu_info populated
* runtime_duration_per_layer (one entry per executed layer)
* recipe_yaml_full embedded in the manifest
* cell_resolved_axes embedded in the manifest
* data_revision_tag (from L1 leaf_config vintage_date_or_tag)
"""
from __future__ import annotations

import json
from pathlib import Path

import macrocast


_RECIPE = """
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
    vintage_policy: current_vintage
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
  fixed_axes: {primary_metric: mse, point_metrics: [mse, rmse]}
8_output:
  fixed_axes:
    export_format: json_csv
    saved_objects: [forecasts, metrics]
    artifact_granularity: per_cell
    naming_convention: descriptive
  leaf_config:
    output_directory: __PLACEHOLDER__
"""


def _run(tmp_path: Path):
    recipe = _RECIPE.replace("__PLACEHOLDER__", str(tmp_path))
    macrocast.run(recipe, output_directory=tmp_path)
    return json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_carries_random_seed_used(tmp_path):
    payload = _run(tmp_path)
    assert payload["provenance"]["random_seed_used"] == 7


def test_manifest_carries_runtime_duration_per_layer(tmp_path):
    payload = _run(tmp_path)
    cell = payload["cells"][0]
    durations = cell["runtime_duration_per_layer"]
    assert {"l1", "l2", "l3", "l4", "l5"}.issubset(durations.keys())
    for k, v in durations.items():
        assert isinstance(v, (int, float))
        assert v >= 0.0


def test_manifest_carries_recipe_root(tmp_path):
    payload = _run(tmp_path)
    recipe = payload["recipe_root"]
    assert isinstance(recipe, dict)
    assert "1_data" in recipe and "4_forecasting_model" in recipe


def test_manifest_carries_cell_resolved_axes(tmp_path):
    payload = _run(tmp_path)
    axes = payload["cells"][0]["cell_resolved_axes"]
    assert isinstance(axes, dict)
    assert "l1" in axes and "l2" in axes and "l5" in axes


def test_manifest_carries_dependency_lockfile_content(tmp_path):
    payload = _run(tmp_path)
    lockfile = payload["provenance"]["dependency_lockfile_content"]
    # When the project ships a pyproject.toml/uv.lock/requirements.txt the
    # content (truncated) should be embedded; otherwise the dict is empty.
    assert isinstance(lockfile, dict)


def test_manifest_carries_runtime_environment_cpu(tmp_path):
    payload = _run(tmp_path)
    env = payload["provenance"]["runtime_environment"]
    assert env["python_version"]
    assert env["cpu_info"]
    assert env["os_name"]


def test_manifest_carries_git_state_or_none(tmp_path):
    payload = _run(tmp_path)
    sha = payload["provenance"]["git_commit_sha"]
    branch = payload["provenance"]["git_branch_name"]
    # Either both populated (in a checkout) or both None (CI/non-git).
    assert sha is None or isinstance(sha, str) and len(sha) >= 7
    assert branch is None or isinstance(branch, str)


def test_manifest_carries_data_revision_tag(tmp_path):
    payload = _run(tmp_path)
    # Falls back to vintage_policy when no explicit vintage_date_or_tag.
    tag = payload["provenance"]["data_revision_tag"]
    assert tag == "current_vintage"


def test_manifest_carries_package_version(tmp_path):
    payload = _run(tmp_path)
    version = payload["provenance"]["package_version"]
    assert version
    assert isinstance(version, str)


def test_manifest_carries_r_and_julia_version_or_none(tmp_path):
    payload = _run(tmp_path)
    prov = payload["provenance"]
    assert "r_version" in prov and "julia_version" in prov
    # Either populated when the binary is installed, or None when not.
    for key in ("r_version", "julia_version"):
        assert prov[key] is None or isinstance(prov[key], str)


def test_manifest_provenance_block_has_all_14_keys(tmp_path):
    """Pin: every design-listed provenance field is present (even when None)."""

    payload = _run(tmp_path)
    prov = payload["provenance"]
    expected = {
        "package_version",
        "python_version",
        "r_version",
        "julia_version",
        "git_commit_sha",
        "git_branch_name",
        "data_revision_tag",
        "random_seed_used",
        "dependency_lockfile_paths",
        "dependency_lockfile_content",
        "runtime_environment",
    }
    assert expected <= prov.keys(), f"missing: {expected - prov.keys()}"
    # And the cell-level fields:
    cell = payload["cells"][0]
    assert "runtime_duration_per_layer" in cell
    assert "cell_resolved_axes" in cell
    # Finally, recipe_root carries the full recipe spec for replication.
    assert "recipe_root" in payload
