"""Cycle 16 N-2: parallel_unit=cells validator acceptance and correctness.

Verifies:
* PARALLEL_UNIT_OPTIONS now contains "cells".
* ParallelUnit Literal includes "cells".
* mf.run with compute_mode=parallel + parallel_unit=cells + n_workers=2 completes.
* cell_ids match between serial and cells-parallel runs (structural equivalence).
"""
from __future__ import annotations

import textwrap

import pytest

import macroforecast


def _multi_cell_recipe(
    *,
    compute_mode: str = "serial",
    n_workers: int | None = None,
    parallel_unit: str | None = None,
    n_cells: int = 2,
) -> str:
    n_workers_block = f"\n            n_workers: {n_workers}" if n_workers is not None else ""
    parallel_unit_block = (
        f"\n            parallel_unit: {parallel_unit}" if parallel_unit is not None else ""
    )
    sweep_values = list(range(1, n_cells + 1))
    return textwrap.dedent(
        f"""
        0_meta:
          fixed_axes:
            failure_policy: fail_fast
            reproducibility_mode: seeded_reproducible
            compute_mode: {compute_mode}
          leaf_config:
            random_seed: 42{parallel_unit_block}{n_workers_block}
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


def test_parallel_unit_cells_in_options():
    """Cycle 16 N-2: PARALLEL_UNIT_OPTIONS contains cells."""
    from macroforecast.core.layers.l0 import PARALLEL_UNIT_OPTIONS
    assert "cells" in PARALLEL_UNIT_OPTIONS, (
        f"cells missing from PARALLEL_UNIT_OPTIONS: {PARALLEL_UNIT_OPTIONS}"
    )


def test_parallel_unit_cells_all_original_options_retained():
    """Cycle 16 N-2: adding cells does not drop original options."""
    from macroforecast.core.layers.l0 import PARALLEL_UNIT_OPTIONS
    for opt in ("models", "horizons", "targets", "oos_dates"):
        assert opt in PARALLEL_UNIT_OPTIONS, f"{opt} missing after adding cells"


def test_parallel_unit_cells_run(tmp_path):
    """Cycle 16 N-2: mf.run with compute_mode=parallel + parallel_unit=cells succeeds."""
    result = macroforecast.run(
        _multi_cell_recipe(compute_mode="parallel", n_workers=2, parallel_unit="cells", n_cells=2),
        output_directory=tmp_path,
    )
    assert len(result.cells) == 2
    assert all(c.succeeded for c in result.cells)


def test_parallel_unit_cells_bit_exact_serial(tmp_path):
    """Cycle 16 N-2: cells parallel produces same cell_ids and sink hashes as serial."""
    shared_cache = tmp_path / "cache"
    serial_dir = tmp_path / "serial"
    cells_dir = tmp_path / "cells"

    serial = macroforecast.run(
        _multi_cell_recipe(compute_mode="serial", n_cells=2),
        output_directory=serial_dir,
        cache_root=shared_cache,
    )
    cells = macroforecast.run(
        _multi_cell_recipe(compute_mode="parallel", n_workers=2, parallel_unit="cells", n_cells=2),
        output_directory=cells_dir,
        cache_root=shared_cache,
    )
    assert len(serial.cells) == len(cells.cells) == 2

    for s, p in zip(serial.cells, cells.cells):
        assert s.cell_id == p.cell_id
        assert s.succeeded and p.succeeded
        # l8_artifacts_v1 encodes output paths so exclude it
        s_hashes = {k: v for k, v in s.sink_hashes.items() if k != "l8_artifacts_v1"}
        c_hashes = {k: v for k, v in p.sink_hashes.items() if k != "l8_artifacts_v1"}
        assert s_hashes == c_hashes, (
            f"cell {s.cell_id}: serial vs cells hash drift: {s_hashes} vs {c_hashes}"
        )
