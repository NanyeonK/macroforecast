# macroforecast.evaluation

[Back to reference](index.md)

`macroforecast.evaluation` owns forecast-evaluation configuration and
recipe-facing evaluation artifacts. In the current codebase this surface is
still implemented through `macroforecast.evaluation.schema` and
`macroforecast.evaluation.ops`; direct pandas/dataframe callables will be
documented here when evaluation is refactored into the same style as
`macroforecast.data`, `macroforecast.preprocessing`, and
`macroforecast.data_analysis`.

## Public Flow

```python
import macroforecast as mf

recipe = {
    "5_evaluation": {
        "fixed_axes": {
            "primary_metric": "mse",
            "point_metrics": ["mse", "mae"],
            "relative_metrics": ["relative_mse", "r2_oos"],
        }
    }
}

result = mf.run(recipe)
metrics = result.metrics()
ranking = result.ranking()
```

## Evaluation Choices

The evaluation schema currently exposes these choices through
`macroforecast.evaluation.schema.DEFAULT_AXES`.

| Choice | Default | Meaning |
| --- | --- | --- |
| `primary_metric` | `"mse"` | Main ranking metric. |
| `point_metrics` | `["mse", "mae"]` | Point forecast metrics to compute. |
| `density_metrics` | `["log_score", "crps"]` | Density or quantile forecast metrics when applicable. |
| `direction_metrics` | `[]` | Directional forecast metrics. |
| `relative_metrics` | `["relative_mse", "r2_oos"]` | Metrics relative to a benchmark model. |
| `benchmark_window` | `"full_oos"` | Sample window used for benchmark-relative comparisons. |
| `benchmark_scope` | `"all_targets_horizons"` | Scope for benchmark-relative comparisons. |
| `agg_time` | `"mean"` | Time aggregation rule. |
| `agg_horizon` | `"per_horizon_separate"` | Horizon aggregation rule. |
| `agg_target` | `"per_target_separate"` | Target aggregation rule. |
| `agg_state` | `"pool_states"` | State aggregation rule for state-level panels. |
| `oos_period` | `"full_oos"` | Out-of-sample period selection. |
| `regime_use` | `"pooled"` | Whether regime-specific evaluation is used. |
| `regime_metrics` | `[]` | Metrics used for regime-specific evaluation. |
| `decomposition_target` | `"none"` | Decomposition target for evaluation summaries. |
| `decomposition_order` | `"marginal"` | Decomposition ordering rule. |
| `ranking` | `"by_primary_metric"` | Ranking rule. |
| `report_style` | `"single_table"` | Evaluation report layout. |

## Metric Sets

| Set | Choices |
| --- | --- |
| Point metrics | `mse`, `rmse`, `mae`, `mape`, `medae`, `theil_u1`, `theil_u2` |
| Density metrics | `log_score`, `crps`, `interval_score`, `coverage_rate` |
| Direction metrics | `success_ratio`, `pesaran_timmermann_metric` |
| Relative metrics | `relative_mse`, `r2_oos`, `relative_mae`, `mse_reduction` |

## Schema Helpers

| Function | Input | Output | Purpose |
| --- | --- | --- | --- |
| `parse_layer_yaml(yaml_text, layer_id="l5")` | YAML text | `dict` | Parse the evaluation block. |
| `parse_recipe_yaml(yaml_text_or_root)` | YAML text or dict | `L5Recipe` | Parse a recipe root and expose evaluation layers. |
| `resolve_axes(dag_or_layer)` | parsed evaluation object | `L5ResolvedAxes` | Return resolved evaluation choices. |
| `resolve_axes_from_raw(fixed, context=None)` | fixed choices plus context | `L5ResolvedAxes` | Fill defaults and deactivate incompatible choices. |
| `validate_layer(layer, context=None)` | evaluation block | `ValidationReport` | Validate evaluation choices. |
| `validate_recipe(recipe)` | recipe text/dict/object | `ValidationReport` | Validate the evaluation block in a full recipe. |

## Execution Ops

These functions are registered as recipe execution operations and generally run
through `mf.run(...)`, not by direct user calls.

| Function | Purpose |
| --- | --- |
| `l5_collect_inputs(inputs, params)` | Collect forecast, model, data, and feature artifacts for evaluation. |
| `metric_compute(inputs, params)` | Evaluation metric step placeholder used by the runtime. |
| `benchmark_relative(inputs, params)` | Benchmark-relative metric step placeholder used by the runtime. |
| `aggregate(inputs, params)` | Evaluation aggregation step placeholder used by the runtime. |
| `slice_and_decompose(inputs, params)` | Slice/decomposition step placeholder used by the runtime. |
| `rank_and_report(inputs, params)` | Ranking/reporting step placeholder used by the runtime. |
| `blocked_oob_reality_check(inputs, params)` | Block-bootstrap OOB reality-check primitive for serially dependent losses. |

## Output

Evaluation runtime output is stored in the `l5_evaluation_v1` artifact. The
high-level result facade exposes the common tables through:

| Method | Output |
| --- | --- |
| `result.metrics()` | Concatenated metrics table across cells. |
| `result.ranking()` | Concatenated ranking table across cells. |

## Boundary

| Question | Use |
| --- | --- |
| What data entered evaluation? | `macroforecast.evaluation.schema.resolve_axes(...)` and runtime artifacts. |
| Which metrics/ranking choices are active? | `macroforecast.evaluation.schema.resolve_axes_from_raw(...)`. |
| What are the final metric tables? | `result.metrics()` after `mf.run(...)`. |
| What changed from raw data to processed data? | `mf.data_analysis.analyze_data(raw, processed.panel)`. |
