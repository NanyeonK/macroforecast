"""Realistic-shape integration tests for v0.1 (closes #168).

These tests run the full L1->L8 pipeline against a 48-month / 32-series
synthetic FRED-MD-shaped panel (``tests/fixtures/fred_md_2020_2023_subset.csv``)
to catch numerical / convergence / shape edge cases that the toy 12-month
inline panels in ``tests/core/test_v01_dimensions.py`` are too small to
surface.

Marked ``slow``: gated in CI via ``pytest -m slow``. The default fast
matrix continues to use the inline panels.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import macrocast


_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "fred_md_2020_2023_subset.csv"

pytestmark = pytest.mark.slow


def _custom_panel_recipe(target: str, n_lag: int = 4, family: str = "ridge") -> str:
    """Build a recipe that loads the realistic FRED-MD CSV via custom_panel,
    runs L2 cleaning + L3 lag + L4 fit + L5 metrics."""

    return f"""
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 13
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: {target}
    target_horizons: [1]
    custom_source_path: {_FIXTURE}
    date_column: sasdate
2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: mccracken_ng_iqr
    outlier_action: flag_as_nan
    imputation_policy: forward_fill
    frame_edge_policy: truncate_to_balanced
3_feature_engineering:
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: predictors}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: target}}}}}}
    - {{id: lag_x, type: step, op: lag, params: {{n_lag: {n_lag}}}, inputs: [src_X]}}
    - {{id: y_h, type: step, op: target_construction, params: {{mode: point_forecast, method: direct, horizon: 1}}, inputs: [src_y]}}
  sinks:
    l3_features_v1: {{X_final: lag_x, y_final: y_h}}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}}}
    - id: fit
      type: step
      op: fit_model
      params:
        family: {family}
        alpha: 1.0
        min_train_size: 20
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
      inputs: [src_X, src_y]
    - {{id: predict, type: step, op: predict, inputs: [fit, src_X]}}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {{primary_metric: mse}}
"""


def test_realistic_fixture_exists():
    assert _FIXTURE.exists(), f"missing fixture: {_FIXTURE}"
    df = pd.read_csv(_FIXTURE)
    assert df.shape[0] >= 48, f"expected >= 48 rows, got {df.shape}"
    assert df.shape[1] >= 30, f"expected >= 30 columns, got {df.shape}"


def test_realistic_recipe_runs_l1_through_l5(tmp_path):
    recipe = _custom_panel_recipe(target="INDPRO")
    result = macrocast.run(recipe, output_directory=tmp_path)
    assert len(result.cells) == 1
    cell = result.cells[0]
    assert cell.succeeded, f"failed: {cell.error}"
    metrics = cell.runtime_result.artifacts["l5_evaluation_v1"].metrics_table
    assert not metrics.empty
    assert metrics["mse"].iloc[0] >= 0


@pytest.mark.parametrize("family", ["ridge", "lasso", "random_forest"])
def test_realistic_fixture_supports_multiple_families(tmp_path, family):
    recipe = _custom_panel_recipe(target="UNRATE", n_lag=3, family=family)
    result = macrocast.run(recipe, output_directory=tmp_path / family)
    cell = result.cells[0]
    assert cell.succeeded, f"family {family} failed: {cell.error}"
    artifact = cell.runtime_result.artifacts["l4_model_artifacts_v1"]
    assert artifact.artifacts["fit"].family == family


def test_realistic_fixture_replicate_bit_exact(tmp_path):
    recipe = _custom_panel_recipe(target="INDPRO")
    macrocast.run(recipe, output_directory=tmp_path)
    rep = macrocast.replicate(tmp_path / "manifest.json")
    assert rep.recipe_match
    assert rep.sink_hashes_match
    assert all(rep.per_cell_match.values())


def test_realistic_fixture_sweep_produces_distinct_artifacts(tmp_path):
    """4-cell sweep over n_lag = [2, 4, 6, 8] on the realistic fixture."""

    recipe = _custom_panel_recipe(target="CPIAUCSL").replace(
        "{n_lag: 4}", "{n_lag: {sweep: [2, 4, 6, 8]}}"
    )
    result = macrocast.run(recipe, output_directory=tmp_path)
    assert len(result.cells) == 4
    assert all(c.succeeded for c in result.cells)
    # Each cell must produce a different L3 features hash (because n_lag differs).
    hashes = {c.sink_hashes["l3_features_v1"] for c in result.cells}
    assert len(hashes) == 4, f"expected 4 distinct L3 hashes, got {hashes}"
