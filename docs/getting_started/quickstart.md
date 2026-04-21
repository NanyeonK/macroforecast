# Quickstart

Run your first macrocast forecast in under 5 minutes.

## Option A: From a YAML recipe

The fastest way to run a study is from a pre-written YAML recipe:

```python
from macrocast.compiler import compile_recipe_yaml, run_compiled_recipe

# Compile the recipe
result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")

# Check if it is executable
print(result.compiled.execution_status)  # "executable"

# Run
execution = run_compiled_recipe(
    result.compiled,
    output_root="runs/",
)

print(f"Artifacts saved to: {execution.artifact_dir}")
```

This runs an AR model against a zero-change benchmark on FRED-MD INDPRO at horizons 1 and 3, using an expanding window.

## Option B: From a Python dict

You can also define the recipe as a Python dictionary:

```python
from macrocast.compiler import compile_recipe_dict, run_compiled_recipe

recipe = {
    "recipe_id": "my-first-study",
    "path": {
        "0_meta": {
            "fixed_axes": {"research_design": "single_path_benchmark"}
        },
        "1_data_task": {
            "fixed_axes": {
                "dataset": "fred_md",
                "information_set_type": "revised",
                "task": "single_target_point_forecast",
                "benchmark_family": "ar_bic",
                "evaluation_scale": "raw_level",
            },
            "leaf_config": {
                "target": "INDPRO",
                "horizons": [1, 3, 6, 12],
            },
        },
        "2_preprocessing": {
            "fixed_axes": {
                "target_transform_policy": "raw_level",
                "x_transform_policy": "raw_level",
                "tcode_policy": "raw_only",
                "target_missing_policy": "none",
                "x_missing_policy": "none",
                "target_outlier_policy": "none",
                "x_outlier_policy": "none",
                "scaling_policy": "none",
                "dimensionality_reduction_policy": "none",
                "feature_selection_policy": "none",
                "preprocess_order": "none",
                "preprocess_fit_scope": "not_applicable",
                "inverse_transform_policy": "none",
                "representation_policy": "raw_only",
                "preprocessing_axis_role": "fixed_preprocessing",
                "tcode_application_scope": "apply_tcode_to_none",
                "target_transform": "level",
                "target_normalization": "none",
                "target_domain": "unconstrained",
                "scaling_scope": "columnwise",
                "additional_preprocessing": "none",
                "x_lag_creation": "no_x_lags",
                "feature_grouping": "none",
                "recipe_mode": "fixed_recipe",
            },
        },
        "3_training": {
            "fixed_axes": {
                "framework": "expanding",
                "feature_builder": "autoreg_lagged_target",
            },
            "sweep_axes": {
                "model_family": ["ridge"],
            },
        },
        "4_evaluation": {
            "fixed_axes": {"primary_metric": "msfe"},
        },
        "6_stat_tests": {
            "fixed_axes": {"stat_test": "dm"},
        },
        "7_importance": {
            "fixed_axes": {"importance_method": "none"},
        },
    },
}

result = compile_recipe_dict(recipe)
print(f"Status: {result.compiled.execution_status}")

execution = run_compiled_recipe(result.compiled, output_root="runs/")
print(f"Artifacts: {execution.artifact_dir}")
```

## What just happened?

1. **Compile**: macrocast validated all axis selections against the registry, built a design (Stage 0) grammar frame via `build_design_frame`, and confirmed the recipe is executable.
2. **Execute**: The runtime loaded FRED-MD data, built an expanding-window out-of-sample evaluation, fitted Ridge at each forecast origin, computed predictions and the AR-BIC benchmark, and ran a Diebold-Mariano test.
3. **Artifacts**: The run directory contains `predictions.csv`, `metrics.json`, `stat_test_dm.json`, `comparison_summary.json`, `manifest.json`, and `tuning_result.json`.

## Reading the results

```python
import json
import pandas as pd

# Load predictions
predictions = pd.read_csv(f"{execution.artifact_dir}/predictions.csv")
print(predictions[["origin_date", "target_date", "y_true", "y_pred", "error"]].head())

# Load metrics
with open(f"{execution.artifact_dir}/metrics.json") as f:
    metrics = json.load(f)
for horizon, m in metrics["metrics_by_horizon"].items():
    print(f"{horizon}: MSFE={m['msfe']:.4f}, OOS R2={m['oos_r2']:.4f}")

# Load DM test
with open(f"{execution.artifact_dir}/stat_test_dm.json") as f:
    dm = json.load(f)
print(f"DM statistic: {dm['statistic']:.3f}, p-value: {dm['p_value']:.4f}")
```

## Next steps

- [Your First Study](first_study.md) — design a complete model comparison with preprocessing and importance
- [Understanding Output](understanding_output.md) — detailed guide to every artifact

**See also:** [Installation](../install.md) | [API Reference](../api/index.md)
