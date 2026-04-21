# Understanding Output

Every macrocast execution writes a set of artifacts to a run directory. This guide explains each one.

## Artifact directory structure

```
runs/
  {recipe_id}__{target}__h{horizons}/
    manifest.json               # Full provenance record
    predictions.csv             # OOS prediction table
    metrics.json                # Per-horizon evaluation metrics
    comparison_summary.json     # Model vs benchmark summary
    tuning_result.json          # HP tuning result
    summary.txt                 # Human-readable summary
    data_preview.csv            # Raw data sample
    stat_test_{name}.json       # Statistical test result (if requested)
    importance_{name}.json      # Importance result (if requested)
```

## predictions.csv

The core output table. Each row is one (forecast origin, target date, horizon) triple:

| Column | Meaning |
|--------|---------|
| `target` | Forecast target variable name |
| `model_name` | Model executor used |
| `benchmark_name` | Benchmark family |
| `horizon` | Forecast horizon (months ahead) |
| `origin_date` | Last date in training window |
| `target_date` | Date being forecast |
| `y_true` | Actual realized value |
| `y_pred` | Model forecast |
| `benchmark_pred` | Benchmark forecast |
| `error` | y_true - y_pred |
| `squared_error` | error^2 |
| `benchmark_error` | y_true - benchmark_pred |
| `training_window_size` | Number of observations in training window |

## metrics.json

Per-horizon evaluation metrics:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| `msfe` | Mean squared forecast error | Lower is better |
| `rmse` | Root MSFE | Same scale as target |
| `mae` | Mean absolute error | Robust to outliers |
| `mape` | Mean absolute percentage error | Scale-independent |
| `relative_msfe` | MSFE_model / MSFE_benchmark | < 1 means model beats benchmark |
| `oos_r2` | 1 - relative_MSFE | > 0 means model beats benchmark |
| `csfe` | Cumulative squared forecast error | Time-aggregated loss |
| `benchmark_win_rate` | Fraction of dates where model < benchmark | Higher is better |
| `directional_accuracy` | Fraction of correct direction forecasts | Higher is better |

## stat_test_{name}.json

Statistical test results. The structure depends on the test:

**Diebold-Mariano (dm, dm_hln, dm_modified):**
```json
{
  "test": "dm",
  "statistic": -2.145,
  "p_value": 0.032,
  "loss_differential_mean": -0.0023,
  "n_observations": 120
}
```
Interpretation: p < 0.05 means model and benchmark have significantly different predictive ability.

**Clark-West (cw):**
```json
{
  "test": "cw",
  "statistic": 1.987,
  "p_value": 0.047,
  "mspe_adjusted": 0.0015
}
```
Interpretation: Use for nested models (when the benchmark is nested within the model).

**Model Confidence Set (mcs):**
```json
{
  "test": "mcs",
  "confidence_set": ["ridge", "lasso"],
  "eliminated": ["ar"],
  "alpha": 0.1
}
```
Interpretation: The confidence set contains models that are not significantly worse than the best.

## importance_{name}.json

Feature importance results. Structure depends on the method:

```text
{
  "method": "permutation_importance",
  "model_family": "ridge",
  "feature_importances": [
    {"feature": "UNRATE", "importance": 0.0234},
    {"feature": "CPIAUCSL", "importance": 0.0189},
    ...
  ]
}
```

## manifest.json

The complete provenance record. Key fields:

| Field | Content |
|-------|---------|
| `recipe_id` | Study identifier |
| `tree_context` | Full axis selection record (fixed/sweep/conditional) |
| `preprocess_contract` | Exact preprocessing configuration |
| `model_spec` | Model family, feature builder, executor name |
| `benchmark_spec` | Benchmark family and configuration |
| `stat_test_spec` | Statistical test requested |
| `importance_spec` | Importance method requested |
| `tuning_result` | HP tuning result (best_hp, trials, score) |
| `reproducibility_spec` | Reproducibility mode and seed |

## tuning_result.json

Hyperparameter tuning result:

```json
{
  "tuning_enabled": false,
  "model_family": "ridge_autoreg_v0",
  "search_algorithm": "none",
  "best_hp": {},
  "best_score": null,
  "total_trials": 0,
  "total_time_seconds": 0.0
}
```

When tuning is enabled (via `search_algorithm` in the recipe), this contains the selected hyperparameters and search statistics.

**See also:**
- [Quickstart](quickstart.md) — run your first study
- [User Guide: Data (Stage 1)](../user_guide/data/index.md) — axes that shape metric inputs (horizon, oos_period, overlap_handling).
- [Stages Reference](stages_reference.md) — cheat sheet with every operational value.
