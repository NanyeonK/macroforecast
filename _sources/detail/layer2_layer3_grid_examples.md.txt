# Layer 2 And Layer 3 Full Grid Examples

Date: 2026-04-24

This page gives concrete full-route recipes for sweeping researcher feature
representations together with forecast generators. It is a usage companion to
the Layer 2/3 sweep contract.

## What Belongs In The Grid

Layer 2 grid axes change the representation `Z`:

- `target_lag_block`
- `x_lag_feature_block`
- `factor_feature_block`
- `level_feature_block`
- `temporal_feature_block`
- `rotation_feature_block`
- `feature_block_combination`
- preprocessing axes that change the model input frame, such as imputation,
  scaling, selection, and dimensionality reduction

Layer 3 grid axes change forecast generation after `Z` exists:

- `model_family`, the current compatibility spelling for candidate
  `forecast_generator_family`
- `benchmark_family`, the current compatibility spelling for baseline
  generator role assignment
- `forecast_type`
- `forecast_object`
- training-window and refit axes
- validation and tuning axes

Benchmarks are not a separate model species in the canonical design. They are
forecast generators assigned the benchmark/baseline role for the comparison.

Do not sweep Layer 2 representation axes under Layer 3. In the current runtime,
`feature_builder` is still required as a fixed compatibility bridge for some
compiled specs. Treat it as a fixed runtime bridge, not as the research
representation language. The research grid should use explicit Layer 2 block
axes.

## Minimal Runnable Grid

The example recipe lives at `examples/recipes/layer2-layer3-grid.yaml`.

It sweeps:

- Layer 2 `target_lag_block`: no target lags vs fixed target lags;
- Layer 2 `x_lag_feature_block`: no X lags vs fixed X lags;
- Layer 3 candidate forecast generator family, currently spelled
  `model_family`: ridge, lasso, and AR.

The recipe intentionally keeps `failure_policy=skip_failed_cell`. This is the
right policy for broad research grids because invalid cells are often expected.
Here, raw-panel feature representations crossed with `model_family=ar` are
compile-invalid cells. They are skipped and recorded, not run as failed model
executions.

```yaml
recipe_id: layer2-layer3-grid
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
      failure_policy: skip_failed_cell
      compute_mode: serial
  1_data_task:
    fixed_axes:
      dataset: fred_md
      information_set_type: revised
      target_structure: single_target_point_forecast
    leaf_config:
      target: INDPRO
      horizons: [1]
  2_preprocessing:
    fixed_axes:
      target_transform_policy: raw_level
      x_transform_policy: raw_level
      tcode_policy: extra_preprocess_without_tcode
      target_missing_policy: none
      x_missing_policy: mean_impute
      target_outlier_policy: none
      x_outlier_policy: none
      scaling_policy: standard
      dimensionality_reduction_policy: none
      feature_selection_policy: none
      preprocess_order: extra_only
      preprocess_fit_scope: train_only
      inverse_transform_policy: none
      evaluation_scale: raw_level
    leaf_config:
      training_config:
        target_lag_count: 2
    sweep_axes:
      target_lag_block: [none, fixed_target_lags]
      x_lag_feature_block: [none, fixed_x_lags]
  3_training:
    fixed_axes:
      framework: rolling
      benchmark_family: zero_change
      feature_builder: raw_feature_panel
    leaf_config:
      benchmark_config:
        minimum_train_size: 5
        rolling_window_size: 5
    sweep_axes:
      model_family: [ridge, lasso, ar]
```

Expected expansion:

| Axis group | Count |
|---|---:|
| Layer 2 target-lag block | 2 |
| Layer 2 X-lag block | 2 |
| Layer 3 model family | 3 |
| Total variants | 12 |

Expected runtime status with `skip_failed_cell`:

| Cell type | Status |
|---|---|
| raw-panel `Z` with ridge/lasso | `success` |
| raw-panel `Z` with AR | `skipped` at compile gate |

The skipped AR cells are not runtime failures. They are compatibility cells
where the Layer 3 generator cannot consume the selected Layer 2 handoff.

## Running The Grid

```python
from pathlib import Path

import yaml

from macrocast import compile_sweep_plan, execute_sweep

recipe = yaml.safe_load(Path("examples/recipes/layer2-layer3-grid.yaml").read_text())
plan = compile_sweep_plan(recipe)

result = execute_sweep(
    plan=plan,
    output_root="runs/layer2-layer3-grid",
    local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
)
```

After execution, read `runs/layer2-layer3-grid/study_manifest.json`.

Important fields:

| Field | Meaning |
|---|---|
| `summary.successful` | Count of executed variants with saved artifacts. |
| `summary.skipped` | Count of compile-invalid cells skipped under the failure policy. |
| `summary.invalid_cells` | Count of cells rejected before model execution. |
| `summary.runnable_variants` | Count of variants that passed the compiler gate. |
| `sweep_plan.variants[].axis_values` | Layer-qualified axis values for the cell. |
| `sweep_plan.variants[].compiler_status` | Per-cell compiler result. |
| `sweep_plan.variants[].compiler_blocked_reasons` | Exact incompatibility reasons. |
| `sweep_plan.variants[].layer3_capability_cell` | Model/runtime/forecast-object capability cell. |

Each skipped variant directory contains `compiler_manifest.json`, so the
research grid is auditable even when a model was never run.

## Extending The Grid

Add one representation axis at a time and keep `max_variants` explicit when the
grid grows:

```yaml
2_preprocessing:
  sweep_axes:
    target_lag_block: [none, fixed_target_lags]
    x_lag_feature_block: [none, fixed_x_lags]
    temporal_feature_block: [none, moving_average_features, rolling_moments]
    rotation_feature_block: [none, moving_average_rotation]
3_training:
  sweep_axes:
    model_family: [ridge, lasso, randomforest, ar]
```

For MARX-style rotation research, use `rotation_feature_block: marx_rotation`
and set `leaf_config.marx_max_lag`. Keep MARX in the Layer 2 grid. Model-family
choice remains Layer 3.

## Interpretation Rule

When a cell is skipped, decide which side owns the reason:

- if `Z` cannot be built, it is Layer 2 representation support debt;
- if `Z` is valid but the model cannot consume that runtime, it is Layer 2 x
  Layer 3 compatibility support debt;
- if the model consumes `Z` but fails during fitting, it is Layer 3 runtime
  behavior;
- if scoring or inference fails after forecasts exist, it belongs to later
  layers.

This split is what makes full research sweeps useful: invalid combinations are
first-class audit records rather than hidden crashes.
