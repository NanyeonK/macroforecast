# Metrics

[Back to reference](index.md)

`macroforecast.metrics` owns forecast scoring only. It does not choose windows,
fit models, run statistical comparison tests, or write artifacts.

Use the namespace form:

```python
import macroforecast as mf

mf.metrics.rmse(y_true, y_pred)
```

Top-level shortcuts such as `mf.rmse(...)` are intentionally not exported.

## Forecast Table Helpers

### evaluate_forecasts

```python
macroforecast.metrics.evaluate_forecasts(
    forecasts,
    *,
    by=("model", "horizon"),
    metrics=("mse", "rmse", "mae"),
    actual="actual",
    prediction="prediction",
    variance_prediction="variance_prediction",
    quantile_predictions="quantile_predictions",
    previous_actual="previous_actual",
    benchmark_model=None,
    model_column="model",
)
```

Input: a `ForecastResult`, forecast table, or pandas-like table with realized
values and forecast columns.

Output: a pandas `DataFrame`, one row per `by` group. The result carries
`attrs["macroforecast_metadata_schema"] = {"kind": "forecast_metrics",
"version": 1, ...}`.

Forecast-table behavior:

| Available input | Added scores |
| --- | --- |
| `actual`, `prediction` | Requested point metrics such as `mse`, `rmse`, `mae`. |
| `benchmark_model` plus benchmark rows | Relative metrics such as `relative_mse`, `relative_mae`, `mse_reduction`, `r2_oos`. |
| `previous_actual` | `theil_u2` and `success_ratio`. |
| `variance_prediction` | `gaussian_nll`, `crps`, and requested `qlike`. |
| `quantile_predictions` dictionaries | Pinball loss by quantile and interval coverage/width/score for matched lower-upper pairs. |

```python
scores = mf.metrics.evaluate_forecasts(
    result,
    metrics=("mse", "rmse", "relative_mse", "r2_oos"),
    benchmark_model="historical_mean",
)
```

### rank_forecasts

```python
macroforecast.metrics.rank_forecasts(
    evaluation,
    *,
    metric="mse",
    by=("horizon",),
    ascending=None,
    rank_column="rank",
)
```

Input: an evaluation table from `evaluate_forecasts(...)` or an equivalent
pandas table.

Output: the same rows with a rank column. If `ascending=None`, lower is better
for loss metrics and higher is better for `r2_oos` and `mse_reduction`.

### get_metric

```python
macroforecast.metrics.get_metric(metric)
```

Input: a metric name or callable.

Output: the resolved callable. Name aliases include `msfe -> mse`,
`validation_mse -> mse`, and `validation_rmse -> rmse`.

Custom metrics do not need registration. Pass a callable directly anywhere a
metric is accepted:

```python
def mean_bias(y_true, y_pred):
    return float(pd.Series(y_pred).sub(pd.Series(y_true)).mean())

scores = mf.metrics.evaluate_forecasts(
    forecasts,
    metrics=("mse", mean_bias),
)
```

The metric callable should accept `(y_true, y_pred)` and return one scalar
`float`. In evaluation tables, the output column name is the callable's
`__name__`, or `"callable_metric"` when no name is available. Metrics requiring
benchmark forecasts, variances, intervals, or previous actuals need one of the
specialized built-in metric names because `evaluate_forecasts()` must know
which forecast-table columns to pass.

## Point Metrics

All point metrics align inputs as pandas Series, drop missing paired
observations, and return a single `float`.

| Function | Signature | Output |
| --- | --- | --- |
| `mse` | `mse(y_true, y_pred)` | Mean squared error. |
| `rmse` | `rmse(y_true, y_pred)` | Root mean squared error. |
| `mae` | `mae(y_true, y_pred)` | Mean absolute error. |
| `medae` | `medae(y_true, y_pred)` | Median absolute error. |
| `mape` | `mape(y_true, y_pred, *, eps=1e-10)` | Mean absolute percentage error on the 0-100 scale. |
| `smape` | `smape(y_true, y_pred, *, eps=1e-10)` | Symmetric MAPE on the 0-100 scale. |
| `theil_u1` | `theil_u1(y_true, y_pred)` | Theil U1 inequality coefficient. |
| `theil_u2` | `theil_u2(y_true, y_pred, y_prev)` | Theil U2 relative to a no-change forecast. |

## Benchmark-Relative Metrics

These functions require realized values, candidate forecasts, and benchmark
forecasts aligned on the same index.

| Function | Signature | Interpretation |
| --- | --- | --- |
| `relative_mse` | `relative_mse(y_true, y_model, y_benchmark)` | Candidate MSE divided by benchmark MSE. Below 1 favors candidate. |
| `relative_mae` | `relative_mae(y_true, y_model, y_benchmark)` | Candidate MAE divided by benchmark MAE. Below 1 favors candidate. |
| `mse_reduction` | `mse_reduction(y_true, y_model, y_benchmark)` | Benchmark MSE minus candidate MSE. Positive favors candidate. |
| `r2_oos` | `r2_oos(y_true, y_model, y_benchmark)` | Out-of-sample `R^2 = 1 - relative_mse`. |

## Density, Interval, And Volatility Metrics

| Function | Signature | Output |
| --- | --- | --- |
| `pinball_loss` | `pinball_loss(y_true, y_quantile, *, quantile)` | Mean quantile pinball loss. |
| `gaussian_nll` | `gaussian_nll(y_true, y_pred, variance)` | Gaussian negative log likelihood. |
| `log_score` | `log_score(y_true, y_pred, variance)` | Alias for Gaussian negative log score. |
| `crps` | `crps(y_true, y_pred, variance)` | Gaussian continuous ranked probability score. |
| `qlike` | `qlike(y_true, variance, *, eps=1e-12)` | QLIKE volatility loss using realized variance or squared realization. |
| `coverage_rate` | `coverage_rate(y_true, lower, upper)` | Share of observations inside the interval. |
| `interval_width` | `interval_width(lower, upper)` | Mean interval width. |
| `interval_score` | `interval_score(y_true, lower, upper, *, alpha=0.05)` | Winkler interval score. |

`evaluate_forecasts(...)` uses `variance_prediction` for `gaussian_nll`,
`log_score`, `crps`, and `qlike`. It uses `quantile_predictions` dictionaries
for pinball and interval metrics.

## Direction Metrics

| Function | Signature | Output |
| --- | --- | --- |
| `success_ratio` | `success_ratio(y_true, y_pred, y_prev)` | Directional hit rate relative to the previous realized value. |
| `pesaran_timmermann_metric` | `pesaran_timmermann_metric(y_true, y_pred, *, threshold=0.0)` | Pesaran-Timmermann directional accuracy statistic. |
