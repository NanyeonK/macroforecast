# Understanding Output

Every macrocast execution writes a set of artifacts to a run directory. This guide explains each one.

## Artifact directory structure

```
runs/
  {recipe_id}__{target}__h{horizons}/
    manifest.json               # Full provenance record
    layer1_official_frame.json  # Layer 1 official frame handoff contract
    fred_sd_series_metadata.json # FRED-SD selected panel metadata, if used
    predictions.csv             # OOS prediction table
    prediction_row_schema.json   # Versioned predictions.csv column contract
    metrics.json                # Per-horizon evaluation metrics
    comparison_summary.json     # Model vs benchmark summary
    evaluation_summary.json     # Layer 4 evaluation contract summary
    evaluation_report.md        # Optional report when report_style=markdown_table
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

`prediction_row_schema.json` records `prediction_row_schema_v1`: required
base columns, observed columns, optional payload column groups, dtypes, payload
families, and forecast objects. Treat it as the stable contract for consumers
that parse `predictions.csv`.

## layer1_official_frame.json

Layer 1 writes this file before Layer 2 representation construction. It records
`layer1_official_frame_v1`, the exact official frame contract used by the run:
source metadata, target and horizon identity, frame shape/index/columns,
information-set provenance, raw missing/outlier handling, missing-availability,
release-lag, variable-universe choices, official transform/T-code evidence,
data warnings, and data reports.

Use this file when comparing runs that differ in data vintage, release-lag,
raw missing/outlier handling, or official transform policy. The manifest keeps a
compact `layer1_official_frame_summary`, while this file owns the full contract.

## fred_sd_series_metadata.json

Runs that include FRED-SD write `fred_sd_series_metadata.json`. It records
`fred_sd_series_metadata_v1`: selected states, selected SD variables, source
sheets, canonical `VARIABLE_STATE` columns, per-column observed windows, and
native-frequency counts inferred from the non-missing calendar. For composite
FRED-MD/FRED-QD + FRED-SD runs, this file describes the FRED-SD component before
generic post-load `variable_universe` filtering.

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

## evaluation_summary.json

Layer 4 writes a canonical summary of the selected evaluation contract. This
file does not recompute forecasts. It records:

| Field | Meaning |
|-------|---------|
| `contract_version` | Evaluation summary schema version |
| `evaluation_spec` | Exact Layer 4 choices used at runtime |
| `summary.primary_metric` | Metric selected for headline ranking |
| `summary.by_horizon` | Per-horizon model, benchmark, and winner records |
| `summary.overall_equal_weight` | Equal-weight aggregate when available |
| `summary.selected_metric_availability` | Whether each selected metric family is materialized by the current payload |

When `evaluation_spec.report_style=markdown_table`, execution also writes
`evaluation_report.md`. When `report_style=latex_table`, it writes
`evaluation_report.tex`. `report_style=tidy_dataframe` keeps the structured
JSON summary only.

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
| `layer1_official_frame_contract` | Versioned Layer 1 official-frame handoff |
| `layer1_official_frame_file` | Full Layer 1 official-frame contract artifact |
| `fred_sd_series_metadata_file` | FRED-SD selected-panel metadata artifact, if present |
| `model_spec` | Model family, feature builder, executor name |
| `benchmark_spec` | Benchmark family and configuration |
| `evaluation_spec` | Layer 4 evaluation choices |
| `evaluation_summary_file` | Canonical Layer 4 summary artifact |
| `evaluation_report_file` | Optional human/table report artifact |
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
- [User Guide: Data (Stage 1)](../user_guide/data/index.md) — data axes that shape metric inputs. `oos_period` is now a Layer 4 evaluation axis, with old data-layer placement kept as a compatibility alias.
- [Stages Reference](stages_reference.md) — cheat sheet with every operational value.
