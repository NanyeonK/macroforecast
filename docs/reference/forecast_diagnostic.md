# macroforecast.forecast_diagnostic

[Back to reference](index.md)

`macroforecast.forecast_diagnostic` inspects outputs from
`macroforecast.forecasting.run(...)`. It does not refit models and does not
change forecasts. It reads two sources:

| Source | Used for |
| --- | --- |
| `ForecastResult.forecasts` | Forecast-vs-actual rows, residuals, rolling loss, selection metadata, combination rows. |
| Saved model sidecar JSON from `stored_model["metadata_path"]` | Coefficients, intercepts, and fit diagnostics recorded by `macroforecast.models`. |

This module corresponds to the old generator/model diagnostic role, but the new
API is callable-first. No YAML, recipe graph, or runtime materialization is
involved.

## Public Flow

```python
import macroforecast as mf

result = mf.forecasting.run(
    processed_panel,
    ["ridge", "random_forest"],
    window=window_spec,
    features=feature_spec,
    selection=mf.selection.grid({"alpha": [0.01, 0.1]}),
    combination=["mean", "inverse_mspe"],
)

diagnostic = mf.forecast_diagnostic.diagnose_forecasts(
    result,
    rolling_window=12,
)
```

## diagnose_forecasts

```python
macroforecast.forecast_diagnostic.diagnose_forecasts(
    forecasts,
    *,
    include_fitted: bool = True,
    include_residuals: bool = True,
    include_rolling_loss: bool = True,
    rolling_window: int = 12,
    rolling_metric: str = "rmse",
    include_coefficients: bool = True,
    include_tuning: bool = True,
    include_ensemble_weights: bool = True,
    include_stage_updates: bool = True,
    include_combined: bool = True,
) -> ForecastDiagnosticReport
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `forecasts` | `ForecastResult` or `pandas.DataFrame` | required | Runner result or forecast table. |
| `include_fitted` | `bool` | `True` | Include row-level fitted-vs-actual table. |
| `include_residuals` | `bool` | `True` | Include grouped residual summary. |
| `include_rolling_loss` | `bool` | `True` | Include rolling OOS loss table. |
| `rolling_window` | positive int | `12` | Rolling window length in forecast rows within each group. |
| `rolling_metric` | `str` | `"rmse"` | `"mse"`, `"rmse"`, `"mae"`, or `"bias"`. |
| `include_coefficients` | `bool` | `True` | Read saved model sidecar JSON and return coefficient paths when available. |
| `include_tuning` | `bool` | `True` | Include per-forecast selection metadata. |
| `include_ensemble_weights` | `bool` | `True` | Reconstruct weights for identifiable combination methods. |
| `include_stage_updates` | `bool` | `True` | Include preprocessing/feature stage update trace from result metadata. |
| `include_combined` | `bool` | `True` | Include combined forecast rows in fitted/residual/loss tables. |

### Output

Returns `ForecastDiagnosticReport`.

| Field | Type | Meaning |
| --- | --- | --- |
| `overview` | `dict` | Forecast count, model/horizon counts, date range, missing prediction/actual counts, stored-model count, selection count, uncertainty count. |
| `fitted` | `DataFrame` or `None` | Forecast rows plus residual, absolute error, squared error, and percent error. |
| `residuals` | `DataFrame` or `None` | Grouped residual diagnostics by model and horizon. |
| `rolling_loss` | `DataFrame` or `None` | Rolling OOS loss by model and horizon. |
| `coefficients` | `DataFrame` or `None` | Coefficient path from saved model sidecars when available. |
| `tuning` | `DataFrame` or `None` | Parameter-selection event trace from forecast table metadata. |
| `ensemble_weights` | `DataFrame` or `None` | Reconstructed combination weights for supported methods. |
| `stage_updates` | `DataFrame` or `None` | Runner stage update trace. |
| `metadata` | `dict` | Input metadata plus compact `forecast_diagnostic` stage. |

Returned tables carry `attrs["macroforecast_metadata"] == report.metadata`.

## Helper Functions

### forecast_overview

```python
macroforecast.forecast_diagnostic.forecast_overview(forecasts) -> dict
```

Returns counts and coverage for one forecast table:

| Key | Meaning |
| --- | --- |
| `n_forecasts`, `n_models`, `models`, `horizons` | Forecast-table shape. |
| `start`, `end` | Forecast date range. |
| `missing_prediction_count`, `missing_actual_count` | Missingness in forecast/actual columns. |
| `combined_count`, `base_model_count` | Combination versus base model rows. |
| `stored_model_count` | Rows with saved model metadata. |
| `selection_count`, `retuned_count` | Rows with selection metadata and rows that retuned. |
| `variance_prediction_count`, `quantile_prediction_count` | Uncertainty output coverage. |

### fitted_vs_actual

```python
macroforecast.forecast_diagnostic.fitted_vs_actual(
    forecasts,
    *,
    include_combined: bool = True,
    drop_missing_actual: bool = True,
) -> pandas.DataFrame
```

Returns row-level diagnostics:

| Column | Meaning |
| --- | --- |
| `prediction`, `actual` | Forecast and realized target from the runner. |
| `residual` | `actual - prediction`. |
| `abs_error` | Absolute residual. |
| `squared_error` | Squared residual. |
| `percent_error` | `residual / abs(actual)`; zero actuals become missing. |

### residual_report

```python
macroforecast.forecast_diagnostic.residual_report(
    forecasts,
    *,
    group_by: Sequence[str] = ("model", "horizon"),
    include_combined: bool = True,
) -> pandas.DataFrame
```

Default grouping is model by horizon. Output columns include `n`, `bias`,
`mae`, `mse`, `rmse`, `residual_sd`, `residual_autocorr1`, `mean_actual`,
`mean_prediction`, `first_date`, and `last_date`.

### rolling_loss

```python
macroforecast.forecast_diagnostic.rolling_loss(
    forecasts,
    *,
    metric: str = "rmse",
    window: int = 12,
    min_periods: int | None = None,
    group_by: Sequence[str] = ("model", "horizon"),
    include_combined: bool = True,
) -> pandas.DataFrame
```

Computes rolling OOS loss inside each group. `rmse` rolls the squared error and
takes the square root after averaging.

### coefficient_trace

```python
macroforecast.forecast_diagnostic.coefficient_trace(
    forecasts,
    *,
    include_intercept: bool = True,
    load_pickle: bool = False,
) -> pandas.DataFrame
```

Reads `stored_model["metadata_path"]` sidecar JSON for each forecast row and
extracts `fit.fit.diagnostics.coefficients`. It does not unpickle model objects
by default. Set `load_pickle=True` only for trusted local artifacts when a
sidecar is missing.

Returned rows include `date`, `origin`, `origin_pos`, `horizon`, `model`,
`feature`, `coefficient`, `component`, and stored artifact paths.

### tuning_trace

```python
macroforecast.forecast_diagnostic.tuning_trace(forecasts) -> pandas.DataFrame
```

Returns one row per forecast row with selection metadata. It records method,
metric, validation window, retune flag, best score, best params, trial counts,
and policy. The current runner stores selection-event summaries, not full
per-trial histories.

### ensemble_weights_over_time

```python
macroforecast.forecast_diagnostic.ensemble_weights_over_time(
    forecasts,
    *,
    unsupported: str = "skip",
) -> pandas.DataFrame
```

Reconstructs weights when the combination method has identifiable weights.

| Method | Weight support |
| --- | --- |
| `mean` | Equal weights. |
| `inverse_mspe`, `dmspe` | Recomputed from historical squared forecast errors using the same discount/min-weight parameters. |
| `best_n` | Equal weights over historically best models. |
| `median`, `trimmed_mean`, `winsorized_mean` | No unique model weights; skipped by default. |

`unsupported` controls non-identifiable methods: `"skip"`, `"nan"`, or
`"raise"`.

### stage_update_trace

```python
macroforecast.forecast_diagnostic.stage_update_trace(forecasts) -> pandas.DataFrame
```

Returns preprocessing and feature-engineering stage update records stored in
`ForecastResult.metadata["stages"]`. This is empty for direct `FeatureSet`
inputs because the runner does not refit preprocessing/features in that path.

## Boundary

| Question | Use |
| --- | --- |
| Score forecasts numerically | `mf.metrics` |
| Run forecast-comparison statistical tests | `mf.tests` |
| Inspect forecast rows, residuals, tuning, coefficients, weights, and stage updates | `mf.forecast_diagnostic` |
| Interpret a single fitted model's feature effects | `mf.interpretation` |
