# macroforecast.metrics

[Back to reference](index.md)

`macroforecast.metrics` owns forecast scoring only. It does not choose windows,
fit models, run statistical comparison tests, or write artifacts.

Use the namespace form:

```python
import macroforecast as mf

mf.metrics.rmse(y_true, y_pred)
```

Top-level shortcuts such as `mf.rmse(...)` are intentionally not exported.

`MetricLike` is the public input type used by metric resolvers: a metric can
be a registered metric name or a callable with the expected scoring signature.

## Risk-Return Forecast Evaluation

### Paper Citation And Scope

This section implements the evaluation framework from:

> Goulet Coulombe, Philippe. 2026. "Quantifying the Risk-Return Tradeoff in
> Forecasting." arXiv:2605.09712v1, submitted May 10, 2026.
> arXiv page: <https://arxiv.org/abs/2605.09712>.

This is not a portfolio-construction module. `macroforecast` does not treat
macroeconomic forecasts as traded assets here. The word "return" means a
date-level **loss differential**:

```text
forecast_return_t(model | benchmark)
    = loss_t(benchmark) - loss_t(model)
```

Positive values mean the candidate model reduced forecast loss relative to the
benchmark on that date. Negative values mean it underperformed. The financial
language is useful because it gives precise names for stability of gains:
volatility, downside risk, upside/downside balance, and drawdown. The object
being evaluated remains a macro forecast panel.

These functions live in `macroforecast.metrics` because they score forecasts.
They do not explain fitted models, so they do not belong in
`macroforecast.interpretation`. They also do not run hypothesis tests, so they
do not belong in `macroforecast.tests`. Higher-level report integration can
later call these functions from `macroforecast.evaluation`.

### Paper Motivation

Standard forecast evaluation usually asks whether a model has lower average
loss than a benchmark. The risk-return view asks whether those gains are stable
enough to trust. A model can have lower RMSE on average while generating large
negative episodes in recessions, inflation spikes, post-COVID periods, or other
macroeconomic regimes where forecast failures are costly.

The paper's primitive object is therefore not an aggregated RMSE table. It is a
date-level sequence of benchmark-relative loss improvements. This is why the
functions below operate on forecast panels and return paths rather than only on
already-aggregated metric tables.

### compute_point_loss

```python
macroforecast.metrics.compute_point_loss(
    y_true,
    y_pred,
    *,
    loss="squared_error",
    variance=None,
    quantile=None,
    eps=1e-12,
) -> pandas.Series
```

Input: aligned realized values and forecasts.

Output: one observation-level loss per aligned row, where lower is better.

Supported losses:

| `loss` | Required inputs | Formula or meaning |
| --- | --- | --- |
| `"squared_error"`, `"mse"`, `"msfe"` | `y_true`, `y_pred` | `(y_true - y_pred)^2` at each date. |
| `"absolute_error"`, `"mae"` | `y_true`, `y_pred` | `abs(y_true - y_pred)` at each date. |
| `"pinball_loss"` | `y_true`, `y_pred`, `quantile` | Quantile loss for one requested quantile. |
| `"negative_log_score"`, `"gaussian_nll"`, `"log_score"` | `y_true`, `y_pred`, `variance` | Gaussian negative log score. |
| `"qlike"` | realized variance in `y_true`, forecast variance in `y_pred` | QLIKE volatility loss. |

### forecast_returns

```python
macroforecast.metrics.forecast_returns(
    forecasts,
    *,
    benchmark,
    group_cols=("target", "horizon"),
    loss="squared_error",
    model_col="model",
    actual="actual",
    prediction="prediction",
    variance_prediction="variance_prediction",
    support_cols=None,
    include_benchmark=False,
    quantile=None,
) -> pandas.DataFrame
```

Input: a `ForecastResult`, forecast table, or pandas-like table with candidate
and benchmark rows. The benchmark must already exist in the same forecast
panel. The function does not create a benchmark forecast.

Required columns:

| Column | Meaning |
| --- | --- |
| `model_col` | Candidate and benchmark model identifiers. Default: `model`. |
| `actual` | Realized value. Default: `actual`. |
| `prediction` | Point forecast. Default: `prediction`. |
| `date`, `origin`, or `origin_pos` | Support identity. At least one is required unless supplied through `support_cols`. |
| `group_cols` | Alignment groups such as `target` and `horizon`; all requested columns must exist. |

Output columns:

| Column | Meaning |
| --- | --- |
| `model_loss` | Candidate date-level loss. |
| `benchmark_loss` | Benchmark date-level loss on the same support row. |
| `forecast_return` | `benchmark_loss - model_loss`; positive favors candidate. |
| `return_sign` | `"positive"`, `"negative"`, or `"zero"`. |
| `cumulative_return` | Cumulative sum of `forecast_return` within model/benchmark/group/loss path. |
| `drawdown` | Cumulative return minus running peak. |
| `loss_name` | Canonical loss label. |
| `model_id`, `benchmark_id` | Stable model labels for downstream grouping. |

Validation is intentionally strict. Candidate and benchmark support must match
exactly within every group, and realized values must match after alignment.
This prevents a benchmark with a different window, horizon, target, or missing
date pattern from being treated as a fair comparator.

```python
returns = mf.metrics.forecast_returns(
    forecast_result,
    benchmark="ar",
    group_cols=("target", "horizon"),
    loss="squared_error",
)
```

### sharpe_ratio

```python
macroforecast.metrics.sharpe_ratio(returns, *, hac_lags=None) -> float
```

Computes mean forecast return divided by return volatility. With
`hac_lags=None`, the denominator is the ordinary sample standard deviation of
the return sequence. With `hac_lags="auto"` or a nonnegative integer, the
denominator is a Newey-West/Bartlett long-run standard deviation. This is a
path-stability score, not a trading Sharpe ratio.

### sortino_ratio

```python
macroforecast.metrics.sortino_ratio(
    returns,
    *,
    target_return=0.0,
) -> float
```

Computes mean excess forecast return divided by downside semideviation:

```text
downside_t = min(return_t - target_return, 0)
```

If all nonzero returns are above the target, the denominator is zero and the
ratio is `inf`. If numerator and denominator are both zero, the ratio is `nan`.

### omega_ratio

```python
macroforecast.metrics.omega_ratio(
    returns,
    *,
    threshold=0.0,
) -> float
```

Computes total upside divided by total downside around a threshold:

```text
omega = sum(max(return_t - threshold, 0))
        / sum(max(threshold - return_t, 0))
```

`inf` means there is upside and no downside; `nan` means there is neither
upside nor downside.

### drawdown_series and max_drawdown

```python
macroforecast.metrics.drawdown_series(returns) -> pandas.Series
macroforecast.metrics.max_drawdown(returns) -> float
```

Drawdown is computed from cumulative forecast returns:

```text
cumulative_t = sum_{s <= t} return_s
drawdown_t = cumulative_t - max_{s <= t}(cumulative_s)
```

For example, returns `[1, 1, -3, 1]` have cumulative returns
`[1, 2, -1, 0]`, drawdowns `[0, 0, -3, -2]`, and maximum drawdown `-3`.

### risk_adjusted_forecast_metrics

```python
macroforecast.metrics.risk_adjusted_forecast_metrics(
    returns,
    *,
    group_cols=None,
    return_col="forecast_return",
    hac_lags="auto",
    target_return=0.0,
    omega_threshold=0.0,
) -> pandas.DataFrame
```

Input: the date-level output of `forecast_returns(...)`, or any DataFrame with
a return column.

Output: one row per group with:

| Column | Meaning |
| --- | --- |
| `n_obs` | Number of finite return observations. |
| `mean_return` | Average benchmark-relative loss reduction. |
| `return_sd` | Sample standard deviation of returns. |
| `hac_return_sd` | HAC long-run standard deviation when requested. |
| `sharpe`, `hac_sharpe` | Mean return divided by ordinary or HAC volatility. |
| `sortino` | Downside-risk-adjusted return. |
| `omega` | Upside/downside ratio. |
| `max_drawdown` | Worst cumulative-return drawdown. |
| `final_cumulative_return` | Sum of returns over the evaluated path. |
| `win_rate` | Share of dates with positive forecast return. |

Default grouping uses available columns such as `model_id`, `benchmark_id`,
`target`, `horizon`, `sample`, `regime`, and `loss_name`.

### edge_ratio

```python
macroforecast.metrics.edge_ratio(
    forecasts,
    *,
    group_cols=("target", "horizon"),
    loss="squared_error",
    model_col="model",
    actual="actual",
    prediction="prediction",
    variance_prediction="variance_prediction",
    support_cols=None,
    quantile=None,
) -> pandas.DataFrame
```

Edge Ratio asks whether a model delivers unique gains relative to the model
pool, not only relative to one benchmark. For each date and model:

```text
edge_t(model) = min_loss_t(all other models) - loss_t(model)
```

Therefore:

| Edge sign | Meaning |
| --- | --- |
| `edge > 0` | The model is strictly better than every alternative on that date. |
| `edge = 0` | The model ties the best alternative. |
| `edge < 0` | At least one alternative is better. |

Aggregated Edge Ratio is:

```text
edge_ratio
    = (sum(max(edge_t, 0)) / sum(max(-edge_t, 0)))
      * (number_of_models - 1)
```

If a model has positive edge wins and no edge regrets, the ratio is `inf`. If a
model never has edge wins, the ratio is `0`. The result also carries the
date-level edge path in `attrs["macroforecast_edge_path"]` for inspection.

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
    volatility_actual=None,
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
The metadata schema also records `by`, `requested_metrics`, `benchmark_model`,
`relative_support_columns`, input columns, and automatically added metric
groups.

Validation: every requested `by` column must exist in the forecast table.
`evaluate_forecasts()` fails loudly instead of dropping unavailable grouping
dimensions. Relative metrics such as `relative_mse`, `relative_mae`,
`mse_reduction`, and `r2_oos` require `benchmark_model`, and the benchmark must
have matching rows for every scored non-benchmark group. The grouping must
include `model_column` because relative metrics compare each candidate model
against a named benchmark model.

`benchmark_model` does not create benchmark forecasts. It selects existing rows
from the forecast table. For a fair comparison, generate the benchmark in the
same forecasting run with the same window/origin/horizon/target contract, or
append an external benchmark CSV only after validating that it has the same
forecast-table schema and the same evaluation support. Relative metrics fail
when candidate and benchmark supports differ. Forecast-table relative metrics
require at least one support identity column: `date`, `origin`, or
`origin_pos`. For matching support rows, candidate and benchmark `actual`
values must also match; otherwise the forecast table is treated as inconsistent.

Forecast-table behavior:

| Available input | Added scores |
| --- | --- |
| `actual`, `prediction` | Requested point metrics such as `mse`, `rmse`, `mae`, `bias`. |
| `benchmark_model` plus benchmark rows | Relative metrics such as `relative_mse`, `relative_mae`, `mse_reduction`, `r2_oos`. |
| `previous_actual` | `theil_u2` and `success_ratio`. |
| `variance_prediction` | `gaussian_nll`, `crps`, and requested `qlike`. |
| `volatility_actual` plus `variance_prediction` | `qlike` against an explicit realized-variance column. If omitted, `actual` is used. |
| `quantile_predictions` dictionaries | Pinball loss by quantile and interval coverage/width/score for matched lower-upper pairs. |

Malformed probabilistic inputs fail validation. Quantile forecasts must be
per-row dictionaries mapping levels strictly inside `(0, 1)` to finite numeric
predictions. Invalid variance, volatility, interval, or quantile values are not
silently clipped or skipped.

Requested specialized metrics fail loudly when their required support columns
are absent:

| Requested metric group | Required forecast-table column |
| --- | --- |
| `gaussian_nll`, `negative_log_score`, `log_score`, `crps` | `variance_prediction` |
| `qlike` | `variance_prediction`; use `volatility_actual` when realized variance is not in `actual` |
| `theil_u2`, `success_ratio` | `previous_actual` |
| `pinball_loss`, `coverage_rate`, `interval_width`, `interval_score` | `quantile_predictions` |

```python
scores = mf.metrics.evaluate_forecasts(
    result,
    metrics=("mse", "rmse", "relative_mse", "r2_oos"),
    benchmark_model="ols",
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
for recognized loss metrics and higher is better for recognized gain metrics
such as `r2_oos`, `mse_reduction`, `success_ratio`, and
`pesaran_timmermann_metric`. Every requested `by` column must exist in the
evaluation table. Signed `bias`, coverage metrics, and custom metrics require
an explicit `ascending=True` or `ascending=False`. Coverage is intentionally
not treated as automatically higher-is-better because interval coverage should
usually be assessed against a nominal level, not maximized.

### get_metric

```python
macroforecast.metrics.get_metric(metric)
```

Input: a metric name or callable.

Output: the resolved callable. Name aliases include `msfe -> mse`,
`validation_mse -> mse`, `validation_rmse -> rmse`,
`mean_error -> bias`, and `negative_log_score -> gaussian_nll`.

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

### metric_kind

```python
macroforecast.metrics.metric_kind(metric)
```

Input: a metric name or callable (`MetricLike`).

Output: one string classifying the metric by the forecast-table column(s) its
table-level evaluation requires. Classification is name-based registry
plumbing only (no new metric math): a custom callable carries no registry
name, so a callable is always `"point"`.

| Kind | Metrics | Required inputs |
| --- | --- | --- |
| `"variance"` | `crps`, `gaussian_nll`, `log_score`, `negative_log_score` | `(y_true, y_pred, variance_prediction)` |
| `"volatility"` | `qlike` | realized variance vs `variance_prediction` |
| `"quantile"` | `pinball_loss`, `coverage_rate`, `interval_width`, `interval_score` | `quantile_predictions` |
| `"relative"` | `relative_mse`, `relative_mae`, `mse_reduction`, `r2_oos` | a benchmark forecast |
| `"direction"` | `theil_u2`, `success_ratio` | a previous-actual reference |
| `"point"` | every other `(y_true, y_pred)` metric, and any callable | plain point forecasts |

### DENSITY_METRIC_NAMES

```python
macroforecast.metrics.DENSITY_METRIC_NAMES
```

A `frozenset[str]` of every metric name whose table-level evaluation needs a
distributional forecast column (`variance_prediction` or
`quantile_predictions`) rather than plain `(y_true, y_pred)` -- exactly the
names `metric_kind` classifies as `"variance"`, `"volatility"`, or
`"quantile"`. Note `mase`, `seasonal_naive_mae`, and `acf1` are NOT members:
they carry a different input shape but are point-adjacent, not distributional.
The managed pipeline routes `EvalSpec.metrics` entries of these density kinds
into `PipelineReport.density` (see the pipeline reference, "Density and
interval forecasting").

## Point Metrics

All point metrics align inputs as pandas Series, drop missing paired
observations, and return a single `float`.

| Function | Signature | Output |
| --- | --- | --- |
| `mse` | `mse(y_true, y_pred)` | Mean squared error. |
| `rmse` | `rmse(y_true, y_pred)` | Root mean squared error. |
| `mae` | `mae(y_true, y_pred)` | Mean absolute error. |
| `bias` | `bias(y_true, y_pred)` | Mean residual `actual - prediction`. |
| `medae` | `medae(y_true, y_pred)` | Median absolute error. |
| `mape` | `mape(y_true, y_pred, *, eps=1e-10)` | Mean absolute percentage error on the 0-100 scale. |
| `smape` | `smape(y_true, y_pred, *, eps=1e-10)` | Symmetric MAPE on the 0-100 scale. |
| `theil_u1` | `theil_u1(y_true, y_pred)` | Theil U1 inequality coefficient. |
| `theil_u2` | `theil_u2(y_true, y_pred, y_prev)` | Theil U2 relative to a no-change forecast. |

## Benchmark-Relative Metrics

These functions require realized values, candidate forecasts, and benchmark
forecasts aligned on the same index.

The direct functions and `evaluate_forecasts(...)` require candidate and
benchmark support to match exactly. They do not silently score only the
intersection of two forecast histories. Forecast-table evaluation also checks
that candidate and benchmark rows carry the same realized value for each support
point.

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
| `negative_log_score` | `negative_log_score(y_true, y_pred, variance)` | Gaussian negative log score. |
| `log_score` | `log_score(y_true, y_pred, variance)` | Backward-compatible alias for `negative_log_score`; lower is better. |
| `crps` | `crps(y_true, y_pred, variance)` | Gaussian continuous ranked probability score. |
| `qlike` | `qlike(y_true, variance, *, eps=1e-12)` | QLIKE volatility loss using realized variance or squared realization. |
| `coverage_rate` | `coverage_rate(y_true, lower, upper)` | Share of observations inside the interval. |
| `interval_width` | `interval_width(lower, upper)` | Mean interval width. |
| `interval_score` | `interval_score(y_true, lower, upper, *, alpha=0.05)` | Winkler interval score. |

`evaluate_forecasts(...)` uses `variance_prediction` for `gaussian_nll`,
`negative_log_score`, `log_score`, and `crps`. `qlike` should be evaluated
against realized variance or squared realization. Pass `volatility_actual` when
that column differs from `actual`. It uses `quantile_predictions` dictionaries
for pinball and interval metrics.

Variance inputs must be finite and strictly positive. QLIKE realized variance
must be finite and nonnegative, while the forecast variance must be strictly
positive. Interval metrics require `upper >= lower` for every evaluated row.
Quantile levels must be strictly inside `(0, 1)`, and quantile predictions must
be finite.

## Direction Metrics

| Function | Signature | Output |
| --- | --- | --- |
| `success_ratio` | `success_ratio(y_true, y_pred, y_prev)` | Directional hit rate relative to the previous realized value. |
| `pesaran_timmermann_metric` | `pesaran_timmermann_metric(y_true, y_pred, *, threshold=0.0)` | Pesaran-Timmermann directional accuracy statistic. |

- `mase` -- Mean Absolute Scaled Error (Hyndman-Koehler), out-of-sample MAE scaled by the in-sample (seasonal-)naive MAE.
- `seasonal_naive_mae` -- in-sample (seasonal-)naive MAE `mean(|y[t]-y[t-m]|)`, the MASE scaling denominator.
- `acf1` -- lag-1 autocorrelation (e.g. of forecast residuals), the ACF1 reported by `forecast::accuracy`.
