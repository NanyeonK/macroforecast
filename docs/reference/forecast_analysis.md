# macroforecast.forecast_analysis

[Back to reference](index.md)

`macroforecast.forecast_analysis` inspects outputs from
`macroforecast.forecasting.run(...)`. It does not refit models and does not
change forecasts. It reads two sources:

| Source | Used for |
| --- | --- |
| `ForecastResult.forecasts` | Forecast-vs-actual rows, residuals, rolling loss, selection metadata, and combination rows. |
| Saved model sidecar JSON from `stored_model["metadata_path"]` | Coefficients, intercepts, hyperparameters, and fit diagnostics recorded by `macroforecast.models`. |

`macroforecast.forecast_diagnostic` remains available as a compatibility alias.

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

analysis = mf.forecast_analysis.diagnose_forecasts(
    result,
    rolling_window=12,
    include_residual_acf=True,
    include_residual_qq=True,
)
```

## diagnose_forecasts

```python
macroforecast.forecast_analysis.diagnose_forecasts(
    forecasts,
    *,
    include_fitted: bool = True,
    include_residuals: bool = True,
    include_residual_acf: bool = False,
    include_residual_qq: bool = False,
    include_rolling_loss: bool = True,
    rolling_window: int = 12,
    rolling_metric: str = "rmse",
    include_forecast_scale: bool = False,
    levels=None,
    scale_view: str = "both_overlay",
    back_transform=None,
    include_training_loss: bool = False,
    include_rolling_training_loss: bool = False,
    training_loss_metric: str = "rmse",
    include_first_vs_last: bool = False,
    include_dfm_idiosyncratic_acf: bool = False,
    include_dfm_factor_stability: bool = False,
    include_coefficients: bool = True,
    include_parameter_stability: bool = True,
    include_tuning: bool = True,
    include_tuning_objective: bool = True,
    include_hyperparameters: bool = True,
    include_tuning_scores: bool = True,
    include_ensemble_weights: bool = True,
    include_ensemble_concentration: bool = True,
    include_member_contribution: bool = False,
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
| `include_residual_acf` | `bool` | `False` | Include residual autocorrelation by model/horizon. |
| `include_residual_qq` | `bool` | `False` | Include normal QQ reference points for residuals. |
| `include_rolling_loss` | `bool` | `True` | Include rolling OOS loss table. |
| `rolling_window` | positive int | `12` | Rolling window length in forecast rows within each group. |
| `rolling_metric` | `str` | `"rmse"` | `"mse"`, `"rmse"`, `"mae"`, or `"bias"`. |
| `include_forecast_scale` | `bool` | `False` | Include transformed and/or back-transformed forecast rows. |
| `levels` | Series, DataFrame, or `None` | `None` | Original-level target series used by `forecast_scale_view` for change/growth back transforms. |
| `scale_view` | `str` | `"both_overlay"` | `"transformed_only"`, `"back_transformed_only"`, or `"both_overlay"`. |
| `back_transform` | callable or `None` | `None` | Optional custom row-level back-transform function. |
| `include_training_loss` | `bool` | `False` | Read saved model sidecar fit metrics. |
| `include_rolling_training_loss` | `bool` | `False` | Add rolling traces of saved fit metrics. |
| `training_loss_metric` | `str` | `"rmse"` | Metric name from model sidecar diagnostics, usually `"rmse"`, `"mse"`, `"mae"`, `"mean"`, `"std"`, or `"n"`. |
| `include_first_vs_last` | `bool` | `False` | Include first-versus-last forecast comparison by model/horizon. |
| `include_dfm_idiosyncratic_acf` | `bool` | `False` | Include ACF of DFM residual diagnostics when sidecars contain DFM residuals. |
| `include_dfm_factor_stability` | `bool` | `False` | Include filtered-factor stability summaries when sidecars contain DFM factors. |
| `include_coefficients` | `bool` | `True` | Read saved model sidecar JSON and return coefficient paths when available. |
| `include_parameter_stability` | `bool` | `True` | Summarize coefficient stability over origins/windows. |
| `include_tuning` | `bool` | `True` | Include per-forecast selection metadata. |
| `include_tuning_objective` | `bool` | `True` | Extract the selected objective and best score from tuning metadata. |
| `include_hyperparameters` | `bool` | `True` | Return selected hyperparameter values over time. |
| `include_tuning_scores` | `bool` | `True` | Summarize tuning best-score distributions. |
| `include_ensemble_weights` | `bool` | `True` | Reconstruct weights for identifiable combination methods. |
| `include_ensemble_concentration` | `bool` | `True` | Summarize ensemble-weight concentration by combined forecast row. |
| `include_member_contribution` | `bool` | `False` | Attach member prediction, weight, and contribution rows where weights are identifiable. |
| `include_stage_updates` | `bool` | `True` | Include preprocessing/feature stage update trace from result metadata. |
| `include_combined` | `bool` | `True` | Include combined forecast rows in fitted/residual/loss tables. |

### Output

Returns `ForecastDiagnosticReport`.

| Field | Type | Meaning |
| --- | --- | --- |
| `overview` | `dict` | Forecast count, model/horizon counts, date range, missing prediction/actual counts, stored-model count, selection count, uncertainty count. |
| `fitted` | `DataFrame` or `None` | Forecast rows plus residual, absolute error, squared error, and percent error. |
| `residuals` | `DataFrame` or `None` | Grouped residual diagnostics by model and horizon. |
| `residual_acf` | `DataFrame` or `None` | Residual autocorrelation table by model, horizon, and lag. |
| `residual_qq` | `DataFrame` or `None` | Residual quantiles paired with standard-normal theoretical quantiles. |
| `rolling_loss` | `DataFrame` or `None` | Rolling OOS loss by model and horizon. |
| `forecast_scale` | `DataFrame` or `None` | Transformed/back-transformed forecast rows. |
| `training_loss` | `DataFrame` or `None` | Sidecar fit metrics by origin/model/horizon. |
| `rolling_training_loss` | `DataFrame` or `None` | Rolling sidecar fit metrics. |
| `first_vs_last` | `DataFrame` or `None` | First and last forecast rows in each group plus changes. |
| `coefficients` | `DataFrame` or `None` | Coefficient path from saved model sidecars when available. |
| `parameter_stability` | `DataFrame` or `None` | Coefficient stability summary across origins/windows. |
| `tuning` | `DataFrame` or `None` | Parameter-selection event trace from forecast table metadata. |
| `tuning_objective` | `DataFrame` or `None` | Selected objective, best score, retune flag, and trial counts by forecast row. |
| `hyperparameters` | `DataFrame` or `None` | Long-form selected hyperparameter values over forecast rows. |
| `tuning_scores` | `DataFrame` or `None` | Tuning-score distribution summary by model/horizon/method. |
| `ensemble_weights` | `DataFrame` or `None` | Reconstructed combination weights for supported methods. |
| `ensemble_concentration` | `DataFrame` or `None` | Herfindahl/effective-number summary of ensemble weights. |
| `member_contribution` | `DataFrame` or `None` | Member prediction and weighted contribution rows. |
| `dfm_idiosyncratic_acf` | `DataFrame` or `None` | DFM idiosyncratic residual ACF from saved fit diagnostics. |
| `dfm_factor_stability` | `DataFrame` or `None` | Filtered-factor mean, variance, drift, and autocorrelation summaries. |
| `stage_updates` | `DataFrame` or `None` | Runner stage update trace. |
| `metadata` | `dict` | Input metadata plus compact `forecast_analysis` stage. |

Returned tables carry `attrs["macroforecast_metadata"] == report.metadata`.

### Metadata

`diagnose_forecasts(...)` attaches one compact stage:

```python
analysis.metadata["forecast_analysis"]
```

The stage records:

| Key | Meaning |
| --- | --- |
| `overview` | Compact counts: forecasts, models, combined rows, stored-model rows, and selection rows. |
| `options` | Residual, rolling-loss, coefficient, tuning, ensemble, and stage-update choices. |
| `tables` | Number of rows generated by each analysis table. |

## Helper Functions

### forecast_overview

```python
macroforecast.forecast_analysis.forecast_overview(forecasts) -> dict
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
macroforecast.forecast_analysis.fitted_vs_actual(
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
macroforecast.forecast_analysis.residual_report(
    forecasts,
    *,
    group_by: Sequence[str] = ("model", "horizon"),
    include_combined: bool = True,
) -> pandas.DataFrame
```

Default grouping is model by horizon. Output columns include `n`, `bias`,
`mae`, `mse`, `rmse`, `residual_sd`, `residual_autocorr1`, `mean_actual`,
`mean_prediction`, `first_date`, and `last_date`.

### residual_autocorrelation

```python
macroforecast.forecast_analysis.residual_autocorrelation(
    forecasts,
    *,
    max_lag: int = 12,
    group_by: Sequence[str] = ("model", "horizon"),
    include_combined: bool = True,
) -> pandas.DataFrame
```

Returns one row per model/horizon/lag with residual ACF values. This supports
forecast-residual correlogram checks without rerunning the model.

### residual_qq

```python
macroforecast.forecast_analysis.residual_qq(
    forecasts,
    *,
    n_quantiles: int = 21,
    group_by: Sequence[str] = ("model", "horizon"),
    include_combined: bool = True,
) -> pandas.DataFrame
```

Returns empirical residual quantiles and matching standard-normal theoretical
quantiles. Use it for QQ plots or tail-shape checks. It does not run a
normality test.

### rolling_loss

```python
macroforecast.forecast_analysis.rolling_loss(
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

### forecast_scale_view

```python
macroforecast.forecast_analysis.forecast_scale_view(
    forecasts,
    *,
    levels=None,
    target=None,
    transform=None,
    view="both_overlay",
    back_transform=None,
    include_combined=True,
) -> pandas.DataFrame
```

Returns forecast rows on the transformed scale, the original level scale, or
both. The runner records `target_transform` when available. Built-in
back-transform support covers `level`, one-step `change`, `growth`, and
`log_growth`. For path-average targets or custom transforms, pass
`back_transform`, a callable that returns either a mapping with `prediction`
and `actual` or a two-value sequence.

Output columns include `date`, `origin`, `horizon`, `model`, `scale`,
`target_transform`, `prediction`, `actual`, `residual`, and
`back_transform_available`.

### select_forecast_origins

```python
macroforecast.forecast_analysis.select_forecast_origins(
    forecasts,
    *,
    view="all_origins",
    every_n=12,
    include_last=True,
    include_combined=True,
) -> pandas.DataFrame
```

Filters the forecast table to one of three origin views:
`"all_origins"`, `"last_origin_only"`, or `"every_n_origins"`. The last
origin is retained by default in `"every_n_origins"` so the final test window
is visible even when it does not land exactly on the spacing.

### first_vs_last_forecast

```python
macroforecast.forecast_analysis.first_vs_last_forecast(
    forecasts,
    *,
    group_by=("model", "horizon"),
    include_combined=True,
) -> pandas.DataFrame
```

Compares the first and last forecast row inside each group. Output includes
first/last dates, origins, predictions, actuals, residuals, and changes. This
is the callable equivalent of the old first-window versus last-window view.

### training_loss_trace

```python
macroforecast.forecast_analysis.training_loss_trace(
    forecasts,
    *,
    load_pickle=False,
) -> pandas.DataFrame
```

Reads saved model sidecar JSON and returns fit metrics recorded by
`macroforecast.models`, usually `n`, `mean`, `std`, `mae`, `mse`, and `rmse`.
It does not unpickle models by default. Path-average forecasts can save one
fit per step; those rows use `fit_step`.

### rolling_training_loss

```python
macroforecast.forecast_analysis.rolling_training_loss(
    forecasts_or_trace,
    *,
    metric="rmse",
    window=12,
    min_periods=None,
    group_by=("model", "horizon"),
    load_pickle=False,
) -> pandas.DataFrame
```

Computes a rolling average of sidecar training metrics. It accepts either a
runner `ForecastResult`/forecast table or the output of
`training_loss_trace(...)`.

### dfm_idiosyncratic_acf

```python
macroforecast.forecast_analysis.dfm_idiosyncratic_acf(
    source,
    *,
    max_lag=12,
    load_pickle=False,
) -> pandas.DataFrame
```

Reads DFM residual diagnostics from a `ModelFit`, `ForecastResult`, or forecast
table with saved model sidecars. Output columns include model context,
residual name, lag, ACF, and observation count.

### dfm_factor_stability

```python
macroforecast.forecast_analysis.dfm_factor_stability(
    source,
    *,
    load_pickle=False,
) -> pandas.DataFrame
```

Reads filtered DFM factor diagnostics from a `ModelFit`, `ForecastResult`, or
forecast table with saved model sidecars. Output includes factor name, count,
mean, standard deviation, variance, first/last values, drift, and lag-1
autocorrelation.

### coefficient_trace

```python
macroforecast.forecast_analysis.coefficient_trace(
    forecasts,
    *,
    include_intercept: bool = True,
    load_pickle: bool = False,
    models: Iterable[str] | None = None,
) -> pandas.DataFrame
```

Reads `stored_model["metadata_path"]` sidecar JSON for each forecast row and
extracts `fit.fit.diagnostics.coefficients`. It does not unpickle model objects
by default. Set `load_pickle=True` only for trusted local artifacts when a
sidecar is missing.

Returned rows include `date`, `origin`, `origin_pos`, `horizon`, `model`,
`fit_step`, `feature`, `coefficient`, `component`, and stored artifact paths.

### parameter_stability

```python
macroforecast.forecast_analysis.parameter_stability(
    forecasts,
    *,
    include_intercept: bool = True,
    load_pickle: bool = False,
    group_by: Sequence[str] = ("model", "horizon", "feature"),
    models: Iterable[str] | None = None,
) -> pandas.DataFrame
```

Summarizes coefficient traces across forecast origins. Output includes count,
mean, standard deviation, min/max, first/last coefficient, and sign-change
count. It is available only when model sidecars contain coefficient metadata.

### tuning_trace

```python
macroforecast.forecast_analysis.tuning_trace(forecasts) -> pandas.DataFrame
```

Returns one row per forecast row with selection metadata. It records method,
metric, validation window, retune flag, best score, best params, trial counts,
and policy. The current runner stores selection-event summaries, not full
per-trial histories.

### tuning_objective_trace

```python
macroforecast.forecast_analysis.tuning_objective_trace(forecasts) -> pandas.DataFrame
```

Extracts the objective-facing part of `tuning_trace`: model, horizon, origin,
method, metric, validation policy, retune flag, best score, and trial count.
Use this when the question is whether selection itself behaved consistently.

### hyperparameter_path

```python
macroforecast.forecast_analysis.hyperparameter_path(forecasts) -> pandas.DataFrame
```

Returns selected hyperparameters in long form: one row per forecast row and
parameter. This is the callable table behind hyperparameter-over-time plots.

### tuning_score_distribution

```python
macroforecast.forecast_analysis.tuning_score_distribution(
    forecasts,
    *,
    group_by: Sequence[str] = ("model", "horizon", "method"),
) -> pandas.DataFrame
```

Summarizes selected tuning scores by group. Output includes count, mean,
standard deviation, min, max, and median best score.

### ensemble_weights_over_time

```python
macroforecast.forecast_analysis.ensemble_weights_over_time(
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

### ensemble_weight_concentration

```python
macroforecast.forecast_analysis.ensemble_weight_concentration(forecasts) -> pandas.DataFrame
```

Summarizes reconstructed weights for each combined forecast row. Output
includes member count, Herfindahl index, effective number of members, max
weight, and min weight.

### ensemble_member_contribution

```python
macroforecast.forecast_analysis.ensemble_member_contribution(forecasts) -> pandas.DataFrame
```

Returns long-form member contribution rows when weights are identifiable:
member model, member prediction, weight, contribution, and combined
prediction.

### stage_update_trace

```python
macroforecast.forecast_analysis.stage_update_trace(forecasts) -> pandas.DataFrame
```

Returns preprocessing and feature-engineering stage update records stored in
`ForecastResult.metadata["stages"]`. This is empty for direct `FeatureSet`
inputs because the runner does not refit preprocessing/features in that path.

### custom_forecast_diagnostic

```python
macroforecast.forecast_analysis.custom_forecast_diagnostic(
    forecasts,
    func,
    *,
    name=None,
    metadata=None,
    **params,
) -> pandas.DataFrame
```

Runs one user diagnostic on a runner `ForecastResult` or forecast table. This
is for inspection only; it does not refit models, recompute selection, or
change forecast rows.

Callable signature:

```python
func(forecasts, *, metadata=None, **params)
```

The `forecasts` argument passed to the callable is a copy of the forecast
`DataFrame`. Accepted callable outputs are `DataFrame`, `Series`, mapping, or a
sequence convertible to a `DataFrame`.

The returned table carries:

| Attr | Meaning |
| --- | --- |
| `macroforecast_metadata_schema.kind` | Always `custom_forecast_diagnostic`. |
| `macroforecast_metadata_schema.method` | `name` or callable name. |
| `macroforecast_metadata` | Input metadata plus a `custom_forecast_diagnostic` stage. |

Example:

```python
def tail_errors(forecasts, *, metadata=None, q=0.95):
    err = (forecasts["actual"] - forecasts["prediction"]).abs()
    return {"q": q, "abs_error": float(err.quantile(q))}

diag = mf.forecast_analysis.custom_forecast_diagnostic(
    result,
    tail_errors,
    name="tail_errors",
    q=0.9,
)
```

## Boundary

| Question | Use |
| --- | --- |
| Run windowed forecasts and combinations | `mf.forecasting` |
| Score and rank forecasts | `mf.metrics` |
| Run forecast-comparison tests | `mf.tests` |
| Inspect forecast rows, residuals, tuning, coefficients, weights, and stage updates | `mf.forecast_analysis` |
