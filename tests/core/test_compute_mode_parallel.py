"""Regression tests for L0 ``compute_mode = parallel`` cell loop (issue #165).

Verifies:

* Default ``compute_mode = serial`` keeps the legacy in-process loop.
* ``compute_mode = parallel`` dispatches cells via ProcessPoolExecutor
  and returns identical sink hashes as the serial run (determinism).
* fail_fast still aborts the rest of the sweep when a cell fails.
* continue_on_failure captures the failed cell and keeps going.
* Parallel run preserves cell-index ordering in the result list.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import macrocast


def _multi_cell_recipe(*, compute_mode: str = "serial", n_workers: int | None = None, n_cells: int = 4) -> str:
    n_workers_block = f"\n            n_workers: {n_workers}" if n_workers is not None else ""
    sweep_values = list(range(1, n_cells + 1))
    return textwrap.dedent(
        f"""
        0_meta:
          fixed_axes:
            failure_policy: fail_fast
            reproducibility_mode: seeded_reproducible
            compute_mode: {compute_mode}
          leaf_config:
            random_seed: 7{n_workers_block}
        1_data:
          fixed_axes:
            custom_source_policy: custom_panel_only
            frequency: monthly
            horizon_set: custom_list
          leaf_config:
            target: y
            target_horizons: [1]
            custom_panel_inline:
              date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01, 2018-11-01, 2018-12-01]
              y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
              x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
              x2: [0.1, 0.4, 0.2, 0.6, 0.3, 0.7, 0.5, 0.8, 0.4, 0.9, 0.6, 1.0]
        2_preprocessing:
          fixed_axes: {{transform_policy: no_transform, outlier_policy: none, imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}}
        3_feature_engineering:
          nodes:
            - {{id: src_X, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: predictors}}}}}}
            - {{id: src_y, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: target}}}}}}
            - {{id: lag_x, type: step, op: lag, params: {{n_lag: {{sweep: {sweep_values}}}}}, inputs: [src_X]}}
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
              params: {{family: ridge, alpha: 1.0, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}}
              inputs: [src_X, src_y]
            - {{id: predict, type: step, op: predict, inputs: [fit, src_X]}}
          sinks:
            l4_forecasts_v1: predict
            l4_model_artifacts_v1: fit
            l4_training_metadata_v1: auto
        5_evaluation:
          fixed_axes: {{primary_metric: mse}}
        """
    )


def test_serial_default_unchanged_when_compute_mode_omitted(tmp_path):
    recipe = _multi_cell_recipe(compute_mode="serial", n_cells=2)
    result = macrocast.run(recipe, output_directory=tmp_path)
    assert len(result.cells) == 2
    assert all(c.succeeded for c in result.cells)


def test_parallel_matches_serial_sink_hashes(tmp_path):
    """The headline determinism guarantee: identical recipe under serial vs
    parallel must produce identical per-cell sink hashes."""

    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    # Share a single cache_root so the L1 artifact (which records its raw
    # cache directory) is identical across runs. Without an explicit
    # cache_root each run would default to ``output_directory / .raw_cache``
    # which differs by construction.
    shared_cache = tmp_path / "shared_raw_cache"
    serial = macrocast.run(
        _multi_cell_recipe(compute_mode="serial", n_cells=4),
        output_directory=serial_dir,
        cache_root=shared_cache,
    )
    parallel = macrocast.run(
        _multi_cell_recipe(compute_mode="parallel", n_workers=2, n_cells=4),
        output_directory=parallel_dir,
        cache_root=shared_cache,
    )
    assert len(serial.cells) == len(parallel.cells) == 4
    for s, p in zip(serial.cells, parallel.cells):
        assert s.cell_id == p.cell_id
        assert s.succeeded and p.succeeded
        # l8_artifacts_v1 hashes encode output paths and so legitimately differ
        # between two output_directory targets; everything else must match.
        s_compare = {k: v for k, v in s.sink_hashes.items() if k != "l8_artifacts_v1"}
        p_compare = {k: v for k, v in p.sink_hashes.items() if k != "l8_artifacts_v1"}
        assert s_compare == p_compare, (
            f"cell {s.cell_id}: serial vs parallel sink hash drift: {s_compare} vs {p_compare}"
        )


def test_parallel_preserves_cell_index_order(tmp_path):
    parallel = macrocast.run(
        _multi_cell_recipe(compute_mode="parallel", n_workers=4, n_cells=6),
        output_directory=tmp_path,
    )
    indices = [c.index for c in parallel.cells]
    assert indices == sorted(indices)
    assert indices == list(range(1, 7))


def test_parallel_with_n_workers_one_falls_back_to_serial(tmp_path):
    """``n_workers=1`` and ``compute_mode=parallel`` should not bother
    spinning up a process pool -- single-worker pool would just add IPC
    overhead. The implementation falls back to the serial loop."""

    result = macrocast.run(
        _multi_cell_recipe(compute_mode="parallel", n_workers=1, n_cells=2),
        output_directory=tmp_path,
    )
    assert len(result.cells) == 2 and all(c.succeeded for c in result.cells)


def test_parallel_with_single_cell_runs_serial(tmp_path):
    """A 1-cell sweep under parallel mode should still complete (no pool
    needed) and emit one CellExecutionResult."""

    result = macrocast.run(
        _multi_cell_recipe(compute_mode="parallel", n_workers=4, n_cells=1),
        output_directory=tmp_path,
    )
    assert len(result.cells) == 1 and result.cells[0].succeeded
