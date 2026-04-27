# Your First Study: Ridge vs Lasso on FRED-MD

This walkthrough designs a complete forecasting comparison study from scratch. By the end, you will have:
- Compared Ridge and Lasso on INDPRO (Industrial Production)
- Used EM imputation and standard scaling
- Run a Diebold-Mariano test
- Computed permutation importance

## Step 1: Choose your fixed design

The **fixed design** defines the comparison environment. Everything here stays constant across models:

- **Dataset**: `fred_md` (FRED-MD monthly macro panel)
- **Information set**: `final_revised_data` (latest available data, not real-time vintages)
- **Framework**: `expanding` (expanding training window)
- **Benchmark**: `ar_bic` (AR model with BIC-selected lag order)
- **Horizons**: 1, 3, 6, 12 months ahead

These choices go in `0_meta`, `1_data_task`, and the `fixed_axes` of `3_training`.

## Step 2: Choose your varying design

The **varying design** is your research question — what you want to compare:

- **Model families**: Ridge and Lasso (these go in `sweep_axes`)

## Step 3: Choose preprocessing

For a data-rich panel like FRED-MD, we need to handle missing values and scale predictors:

- `x_missing_policy`: `em_impute` (EM algorithm, train-only fit)
- `scaling_policy`: `standard` (zero mean, unit variance, train-only fit)

## Step 4: Write the YAML recipe

```yaml
recipe_id: ridge-vs-lasso-indpro

path:
  0_meta:
    fixed_axes:
      research_design: single_forecast_run
      experiment_unit: single_target_generator_grid

  1_data_task:
    fixed_axes:
      dataset: fred_md
      information_set_type: final_revised_data
      target_structure: single_target
      benchmark_family: ar_bic
      evaluation_scale: raw_level
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6, 12]

  2_preprocessing:
    fixed_axes:
      target_transform_policy: raw_level
      x_transform_policy: raw_level
      tcode_policy: extra_preprocess_only
      target_missing_policy: none
      x_missing_policy: em_impute
      target_outlier_policy: none
      x_outlier_policy: none
      scaling_policy: standard
      dimensionality_reduction_policy: none
      feature_selection_policy: none
      preprocess_order: extra_only
      preprocess_fit_scope: train_only
      inverse_transform_policy: none
      representation_policy: raw_only
      tcode_application_scope: none
      target_transform: level
      target_normalization: none
      target_domain: unconstrained
      scaling_scope: columnwise
      additional_preprocessing: none
      x_lag_creation: no_x_lags
      feature_grouping: none

  3_training:
    fixed_axes:
      framework: expanding
      feature_builder: raw_feature_panel
    sweep_axes:
      model_family: [ridge, lasso]

  4_evaluation:
    fixed_axes:
      primary_metric: msfe

  6_stat_tests:
    fixed_axes:
      stat_test: dm

  7_importance:
    fixed_axes:
      importance_method: permutation_importance
```

Save this as `my_study.yaml`.

## Step 5: Compile and inspect

```python
from macrocast.compiler import compile_recipe_yaml

result = compile_recipe_yaml("my_study.yaml")
compiled = result.compiled

print(f"Execution status: {compiled.execution_status}")
print(f"Warnings: {list(compiled.warnings)}")
print(f"Tree context route: {compiled.tree_context['route_owner']}")
```

If the status is `"executable"`, your study is ready to run. If not, the warnings tell you exactly what is unsupported.

## Step 6: Execute

```python
from macrocast.compiler import run_compiled_recipe

execution = run_compiled_recipe(compiled, output_root="runs/")
print(f"Artifacts saved to: {execution.artifact_dir}")
```

## Step 7: Analyze results

```python
import json
import pandas as pd

art = execution.artifact_dir

# Predictions
pred = pd.read_csv(f"{art}/predictions.csv")
print(pred.groupby("horizon")[["squared_error", "benchmark_squared_error"]].mean())

# Metrics
metrics = json.load(open(f"{art}/metrics.json"))
for h, m in metrics["metrics_by_horizon"].items():
    print(f"{h}: MSFE={m['msfe']:.4f}, relative_MSFE={m['relative_msfe']:.4f}, OOS_R2={m['oos_r2']:.4f}")

# DM test
dm = json.load(open(f"{art}/stat_test_dm.json"))
print(f"\nDM test: stat={dm['statistic']:.3f}, p={dm['p_value']:.4f}")
if dm["p_value"] < 0.05:
    print("  => Model significantly differs from benchmark at 5% level")

# Importance
imp = json.load(open(f"{art}/importance_permutation_importance.json"))
print(f"\nTop 5 important features:")
for feat in sorted(imp["feature_importances"], key=lambda x: -abs(x["importance"]))[:5]:
    print(f"  {feat['feature']}: {feat['importance']:.4f}")
```

## What you learned

- **Fixed axes** define the fair comparison environment (dataset, framework, benchmark)
- **Sweep axes** define what varies (model_family)
- **Preprocessing governance** ensures both models see identical preprocessed data
- **Statistical tests** quantify whether performance differences are significant
- **Importance methods** explain which variables drive the forecast

## Next steps

- [Understanding Output](understanding_output.md) — every artifact explained
- [User Guide: Design (Stage 0)](../user_guide/design.md) — six axes that decide study shape.
- [User Guide: Data (Stage 1)](../user_guide/data/index.md) — twenty axes for data, target structure, evaluation window.
- [Stages Reference](stages_reference.md) — cheat sheet with every operational value.
