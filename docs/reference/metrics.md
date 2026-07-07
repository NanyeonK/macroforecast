# macroforecast.metrics

[Back to reference](index.md)

Point, density, directional, financial, and benchmark-relative scoring callables.

Guide context: [../guide/concepts/evaluation.md](../guide/concepts/evaluation.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `MetricLike` | data | Represent a PEP 604 union type |
| `DENSITY_METRIC_NAMES` | data | frozenset() -> empty frozenset object |
| `bias` | function | Mean forecast residual, computed as ``actual - prediction``. |
| `compute_point_loss` | function | Return observation-level forecast loss where lower is better. |
| `coverage_rate` | function | Share of observations covered by lower/upper forecasts. |
| `crps` | function | Continuous ranked probability score for Gaussian predictive densities. |
| `drawdown_series` | function | Return cumulative forecast return minus its running peak. |
| `edge_ratio` | function | Return frontier-based Edge Ratio by model and group. |
| `evaluate_forecasts` | function | Evaluate a forecasting runner output or forecast table. |
| `forecast_returns` | function | Construct date-level forecast-return rows relative to a benchmark model. |
| `gaussian_nll` | function | Gaussian negative log likelihood using supplied predictive variances. |
| `get_metric` | function | Return a metric callable by name or pass through a callable metric. |
| `interval_score` | function | Winkler interval score for a nominal ``1 - alpha`` interval. |
| `interval_width` | function | Mean forecast interval width. |
| `log_score` | function | Alias for Gaussian negative log score; lower is better. |
| `mae` | function | Mean absolute error. |
| `mape` | function | Mean absolute percentage error on the 0-100 scale. |
| `max_drawdown` | function | Return the most negative drawdown of a forecast-return path. |
| `medae` | function | Median absolute error. |
| `metric_kind` | function | Classify a metric by the forecast-table column(s) its table-level |
| `mse` | function | Mean squared error. |
| `mse_reduction` | function | Benchmark MSE minus candidate model MSE. |
| `negative_log_score` | function | Gaussian negative log score; lower is better. |
| `omega_ratio` | function | Return total upside divided by total downside around a threshold. |
| `pesaran_timmermann_metric` | function | Pesaran-Timmermann directional accuracy statistic. |
| `pinball_loss` | function | Mean quantile pinball loss. |
| `qlike` | function | QLIKE loss for volatility forecasts. |
| `r2_oos` | function | Out-of-sample R squared relative to a benchmark forecast. |
| `rank_forecasts` | function | Rank evaluated models within horizon/target groups. |
| `relative_mae` | function | Candidate model MAE divided by benchmark MAE. |
| `relative_mse` | function | Candidate model MSE divided by benchmark MSE. |
| `risk_adjusted_forecast_metrics` | function | Aggregate forecast-return paths into risk-adjusted performance metrics. |
| `rmse` | function | Root mean squared error. |
| `sharpe_ratio` | function | Return mean forecast return divided by naive or HAC return volatility. |
| `smape` | function | Symmetric mean absolute percentage error, M4/Mcomp convention. |
| `sortino_ratio` | function | Return mean excess forecast return divided by downside semideviation. |
| `success_ratio` | function | Directional hit rate relative to a previous actual value. |
| `theil_u1` | function | Theil U1 inequality coefficient. |
| `theil_u2` | function | Proportional-change Theil U relative to a no-change forecast. |

## Data And Module Values

### `MetricLike`

Kind: `data`

```python
MetricLike = str | collections.abc.Callable[..., float]
```
### `DENSITY_METRIC_NAMES`

Kind: `data`

```python
DENSITY_METRIC_NAMES = frozenset({'coverage_rate', 'crps', 'gaussian_nll', 'interval_score', 'interval_width', 'log_score', 'negative_log_score', 'pinball_loss', 'qlike'})
```

## Callable And Class Reference

### bias

Qualified name: `macroforecast.metrics.bias`

#### Signature

```python
macroforecast.metrics.bias(y_true: Any, y_pred: Any) -> float
```

#### Description

Mean forecast residual, computed as ``actual - prediction``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.bias(...)
```
### compute_point_loss

Qualified name: `macroforecast.metrics.compute_point_loss`

#### Signature

```python
macroforecast.metrics.compute_point_loss(y_true: Any, y_pred: Any, *, loss: str = "squared_error", variance: Any | None = None, quantile: float | None = None, eps: float = 1e-12) -> pd.Series
```

#### Description

Return observation-level forecast loss where lower is better.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `variance` | keyword only | `Any \| None` | `None` |
| `quantile` | keyword only | `float \| None` | `None` |
| `eps` | keyword only | `float` | `1e-12` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.compute_point_loss(...)
```
### coverage_rate

Qualified name: `macroforecast.metrics.coverage_rate`

#### Signature

```python
macroforecast.metrics.coverage_rate(y_true: Any, lower: Any, upper: Any) -> float
```

#### Description

Share of observations covered by lower/upper forecasts.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `lower` | positional or keyword | `Any` | `required` |
| `upper` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.coverage_rate(...)
```
### crps

Qualified name: `macroforecast.metrics.crps`

#### Signature

```python
macroforecast.metrics.crps(y_true: Any, y_pred: Any, variance: Any) -> float
```

#### Description

Continuous ranked probability score for Gaussian predictive densities.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `variance` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.crps(...)
```
### drawdown_series

Qualified name: `macroforecast.metrics.drawdown_series`

#### Signature

```python
macroforecast.metrics.drawdown_series(returns: Any) -> pd.Series
```

#### Description

Return cumulative forecast return minus its running peak.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `returns` | positional or keyword | `Any` | `required` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.drawdown_series(...)
```
### edge_ratio

Qualified name: `macroforecast.metrics.edge_ratio`

#### Signature

```python
macroforecast.metrics.edge_ratio(forecasts: Any, *, group_cols: Sequence[str] = ('target', 'horizon'), loss: str = "squared_error", model_col: str = "model", actual: str = "actual", prediction: str = "prediction", variance_prediction: str = "variance_prediction", support_cols: Sequence[str] | None = None, quantile: float | None = None) -> pd.DataFrame
```

#### Description

Return frontier-based Edge Ratio by model and group.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `group_cols` | keyword only | `Sequence[str]` | `("target", "horizon")` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `model_col` | keyword only | `str` | `"model"` |
| `actual` | keyword only | `str` | `"actual"` |
| `prediction` | keyword only | `str` | `"prediction"` |
| `variance_prediction` | keyword only | `str` | `"variance_prediction"` |
| `support_cols` | keyword only | `Sequence[str] \| None` | `None` |
| `quantile` | keyword only | `float \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.edge_ratio(...)
```
### evaluate_forecasts

Qualified name: `macroforecast.metrics.evaluate_forecasts`

#### Signature

```python
macroforecast.metrics.evaluate_forecasts(forecasts: Any, *, by: Sequence[str] = ('model', 'horizon'), metrics: Sequence[str | MetricLike] = ('mse', 'rmse', 'mae'), actual: str = "actual", prediction: str = "prediction", variance_prediction: str = "variance_prediction", volatility_actual: str | None = None, quantile_predictions: str = "quantile_predictions", previous_actual: str = "previous_actual", benchmark_model: str | None = None, model_column: str = "model") -> pd.DataFrame
```

#### Description

Evaluate a forecasting runner output or forecast table.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `metrics` | keyword only | `Sequence[str \| MetricLike]` | `("mse", "rmse", "mae")` |
| `actual` | keyword only | `str` | `"actual"` |
| `prediction` | keyword only | `str` | `"prediction"` |
| `variance_prediction` | keyword only | `str` | `"variance_prediction"` |
| `volatility_actual` | keyword only | `str \| None` | `None` |
| `quantile_predictions` | keyword only | `str` | `"quantile_predictions"` |
| `previous_actual` | keyword only | `str` | `"previous_actual"` |
| `benchmark_model` | keyword only | `str \| None` | `None` |
| `model_column` | keyword only | `str` | `"model"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.evaluate_forecasts(...)
```
### forecast_returns

Qualified name: `macroforecast.metrics.forecast_returns`

#### Signature

```python
macroforecast.metrics.forecast_returns(forecasts: Any, *, benchmark: str, group_cols: Sequence[str] = ('target', 'horizon'), loss: str = "squared_error", model_col: str = "model", actual: str = "actual", prediction: str = "prediction", variance_prediction: str = "variance_prediction", support_cols: Sequence[str] | None = None, include_benchmark: bool = False, quantile: float | None = None) -> pd.DataFrame
```

#### Description

Construct date-level forecast-return rows relative to a benchmark model.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `benchmark` | keyword only | `str` | `required` |
| `group_cols` | keyword only | `Sequence[str]` | `("target", "horizon")` |
| `loss` | keyword only | `str` | `"squared_error"` |
| `model_col` | keyword only | `str` | `"model"` |
| `actual` | keyword only | `str` | `"actual"` |
| `prediction` | keyword only | `str` | `"prediction"` |
| `variance_prediction` | keyword only | `str` | `"variance_prediction"` |
| `support_cols` | keyword only | `Sequence[str] \| None` | `None` |
| `include_benchmark` | keyword only | `bool` | `False` |
| `quantile` | keyword only | `float \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.forecast_returns(...)
```
### gaussian_nll

Qualified name: `macroforecast.metrics.gaussian_nll`

#### Signature

```python
macroforecast.metrics.gaussian_nll(y_true: Any, y_pred: Any, variance: Any) -> float
```

#### Description

Gaussian negative log likelihood using supplied predictive variances.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `variance` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.gaussian_nll(...)
```
### get_metric

Qualified name: `macroforecast.metrics.get_metric`

#### Signature

```python
macroforecast.metrics.get_metric(metric: MetricLike) -> Callable[..., float]
```

#### Description

Return a metric callable by name or pass through a callable metric.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `metric` | positional or keyword | `MetricLike` | `required` |

#### Returns

`Callable[..., float]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.get_metric(...)
```
### interval_score

Qualified name: `macroforecast.metrics.interval_score`

#### Signature

```python
macroforecast.metrics.interval_score(y_true: Any, lower: Any, upper: Any, *, alpha: float = 0.05) -> float
```

#### Description

Winkler interval score for a nominal ``1 - alpha`` interval.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `lower` | positional or keyword | `Any` | `required` |
| `upper` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.interval_score(...)
```
### interval_width

Qualified name: `macroforecast.metrics.interval_width`

#### Signature

```python
macroforecast.metrics.interval_width(lower: Any, upper: Any) -> float
```

#### Description

Mean forecast interval width.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `lower` | positional or keyword | `Any` | `required` |
| `upper` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.interval_width(...)
```
### log_score

Qualified name: `macroforecast.metrics.log_score`

#### Signature

```python
macroforecast.metrics.log_score(y_true: Any, y_pred: Any, variance: Any) -> float
```

#### Description

Alias for Gaussian negative log score; lower is better.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `variance` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.log_score(...)
```
### mae

Qualified name: `macroforecast.metrics.mae`

#### Signature

```python
macroforecast.metrics.mae(y_true: Any, y_pred: Any) -> float
```

#### Description

Mean absolute error.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.mae(...)
```
### mape

Qualified name: `macroforecast.metrics.mape`

#### Signature

```python
macroforecast.metrics.mape(y_true: Any, y_pred: Any, *, eps: float = 1e-10) -> float
```

#### Description

Mean absolute percentage error on the 0-100 scale.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `eps` | keyword only | `float` | `1e-10` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.mape(...)
```
### max_drawdown

Qualified name: `macroforecast.metrics.max_drawdown`

#### Signature

```python
macroforecast.metrics.max_drawdown(returns: Any) -> float
```

#### Description

Return the most negative drawdown of a forecast-return path.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `returns` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.max_drawdown(...)
```
### medae

Qualified name: `macroforecast.metrics.medae`

#### Signature

```python
macroforecast.metrics.medae(y_true: Any, y_pred: Any) -> float
```

#### Description

Median absolute error.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.medae(...)
```
### metric_kind

Qualified name: `macroforecast.metrics.metric_kind`

#### Signature

```python
macroforecast.metrics.metric_kind(metric: MetricLike) -> str
```

#### Description

Classify a metric by the forecast-table column(s) its table-level
evaluation requires (registry plumbing only -- no new metric math).

Returns one of:

- ``"variance"``: Gaussian predictive-density metrics needing
  ``(y_true, y_pred, variance_prediction)`` -- ``crps``, ``gaussian_nll``,
  ``log_score``, ``negative_log_score``.
- ``"volatility"``: realized-vs-forecast variance metrics needing
  ``(realized_variance, variance_prediction)`` -- ``qlike``.
- ``"quantile"``: interval/quantile metrics needing ``quantile_predictions``
  -- ``pinball_loss``, ``coverage_rate``, ``interval_width``,
  ``interval_score``.
- ``"relative"``: benchmark-relative metrics needing a benchmark forecast --
  ``relative_mse``, ``relative_mae``, ``mse_reduction``, ``r2_oos``.
- ``"direction"``: metrics needing a ``previous_actual`` reference --
  ``theil_u2``, ``success_ratio``.
- ``"point"`` (the default): every ordinary ``(y_true, y_pred)`` metric,
  including any custom callable (classification is name-based, so a
  callable -- which carries no registry name -- is always ``"point"``).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `metric` | positional or keyword | `MetricLike` | `required` |

#### Returns

`str`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.metric_kind(...)
```
### mse

Qualified name: `macroforecast.metrics.mse`

#### Signature

```python
macroforecast.metrics.mse(y_true: Any, y_pred: Any) -> float
```

#### Description

Mean squared error.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.mse(...)
```
### mse_reduction

Qualified name: `macroforecast.metrics.mse_reduction`

#### Signature

```python
macroforecast.metrics.mse_reduction(y_true: Any, y_model: Any, y_benchmark: Any) -> float
```

#### Description

Benchmark MSE minus candidate model MSE.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_model` | positional or keyword | `Any` | `required` |
| `y_benchmark` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.mse_reduction(...)
```
### negative_log_score

Qualified name: `macroforecast.metrics.negative_log_score`

#### Signature

```python
macroforecast.metrics.negative_log_score(y_true: Any, y_pred: Any, variance: Any) -> float
```

#### Description

Gaussian negative log score; lower is better.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `variance` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.negative_log_score(...)
```
### omega_ratio

Qualified name: `macroforecast.metrics.omega_ratio`

#### Signature

```python
macroforecast.metrics.omega_ratio(returns: Any, *, threshold: float = 0.0) -> float
```

#### Description

Return total upside divided by total downside around a threshold.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `returns` | positional or keyword | `Any` | `required` |
| `threshold` | keyword only | `float` | `0.0` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.omega_ratio(...)
```
### pesaran_timmermann_metric

Qualified name: `macroforecast.metrics.pesaran_timmermann_metric`

#### Signature

```python
macroforecast.metrics.pesaran_timmermann_metric(y_true: Any, y_pred: Any, *, threshold: float = 0.0) -> float
```

#### Description

Pesaran-Timmermann directional accuracy statistic.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `threshold` | keyword only | `float` | `0.0` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.pesaran_timmermann_metric(...)
```
### pinball_loss

Qualified name: `macroforecast.metrics.pinball_loss`

#### Signature

```python
macroforecast.metrics.pinball_loss(y_true: Any, y_quantile: Any, *, quantile: float) -> float
```

#### Description

Mean quantile pinball loss.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_quantile` | positional or keyword | `Any` | `required` |
| `quantile` | keyword only | `float` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.pinball_loss(...)
```
### qlike

Qualified name: `macroforecast.metrics.qlike`

#### Signature

```python
macroforecast.metrics.qlike(y_true: Any, variance: Any, *, eps: float = 1e-12) -> float
```

#### Description

QLIKE loss for volatility forecasts.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `variance` | positional or keyword | `Any` | `required` |
| `eps` | keyword only | `float` | `1e-12` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.qlike(...)
```
### r2_oos

Qualified name: `macroforecast.metrics.r2_oos`

#### Signature

```python
macroforecast.metrics.r2_oos(y_true: Any, y_model: Any, y_benchmark: Any) -> float
```

#### Description

Out-of-sample R squared relative to a benchmark forecast.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_model` | positional or keyword | `Any` | `required` |
| `y_benchmark` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.r2_oos(...)
```
### rank_forecasts

Qualified name: `macroforecast.metrics.rank_forecasts`

#### Signature

```python
macroforecast.metrics.rank_forecasts(evaluation: pd.DataFrame, *, metric: str = "mse", by: Sequence[str] = ('horizon',), ascending: bool | None = None, rank_column: str = "rank") -> pd.DataFrame
```

#### Description

Rank evaluated models within horizon/target groups.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `evaluation` | positional or keyword | `pd.DataFrame` | `required` |
| `metric` | keyword only | `str` | `"mse"` |
| `by` | keyword only | `Sequence[str]` | `("horizon",)` |
| `ascending` | keyword only | `bool \| None` | `None` |
| `rank_column` | keyword only | `str` | `"rank"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.rank_forecasts(...)
```
### relative_mae

Qualified name: `macroforecast.metrics.relative_mae`

#### Signature

```python
macroforecast.metrics.relative_mae(y_true: Any, y_model: Any, y_benchmark: Any) -> float
```

#### Description

Candidate model MAE divided by benchmark MAE.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_model` | positional or keyword | `Any` | `required` |
| `y_benchmark` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.relative_mae(...)
```
### relative_mse

Qualified name: `macroforecast.metrics.relative_mse`

#### Signature

```python
macroforecast.metrics.relative_mse(y_true: Any, y_model: Any, y_benchmark: Any) -> float
```

#### Description

Candidate model MSE divided by benchmark MSE.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_model` | positional or keyword | `Any` | `required` |
| `y_benchmark` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.relative_mse(...)
```
### risk_adjusted_forecast_metrics

Qualified name: `macroforecast.metrics.risk_adjusted_forecast_metrics`

#### Signature

```python
macroforecast.metrics.risk_adjusted_forecast_metrics(returns: Any, *, group_cols: Sequence[str] | None = None, return_col: str = "forecast_return", hac_lags: int | str | None = "auto", target_return: float = 0.0, omega_threshold: float = 0.0) -> pd.DataFrame
```

#### Description

Aggregate forecast-return paths into risk-adjusted performance metrics.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `returns` | positional or keyword | `Any` | `required` |
| `group_cols` | keyword only | `Sequence[str] \| None` | `None` |
| `return_col` | keyword only | `str` | `"forecast_return"` |
| `hac_lags` | keyword only | `int \| str \| None` | `"auto"` |
| `target_return` | keyword only | `float` | `0.0` |
| `omega_threshold` | keyword only | `float` | `0.0` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.risk_adjusted_forecast_metrics(...)
```
### rmse

Qualified name: `macroforecast.metrics.rmse`

#### Signature

```python
macroforecast.metrics.rmse(y_true: Any, y_pred: Any) -> float
```

#### Description

Root mean squared error.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.rmse(...)
```
### sharpe_ratio

Qualified name: `macroforecast.metrics.sharpe_ratio`

#### Signature

```python
macroforecast.metrics.sharpe_ratio(returns: Any, *, hac_lags: int | str | None = None) -> float
```

#### Description

Return mean forecast return divided by naive or HAC return volatility.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `returns` | positional or keyword | `Any` | `required` |
| `hac_lags` | keyword only | `int \| str \| None` | `None` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.sharpe_ratio(...)
```
### smape

Qualified name: `macroforecast.metrics.smape`

#### Signature

```python
macroforecast.metrics.smape(y_true: Any, y_pred: Any, *, eps: float = 1e-10) -> float
```

#### Description

Symmetric mean absolute percentage error, M4/Mcomp convention.

Uses the ``(|A|+|F|)/2`` denominator, so each term is bounded by 200 and the
statistic ranges on **0-200** (not 0-100); this matches published M4 sMAPE.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `eps` | keyword only | `float` | `1e-10` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.smape(...)
```
### sortino_ratio

Qualified name: `macroforecast.metrics.sortino_ratio`

#### Signature

```python
macroforecast.metrics.sortino_ratio(returns: Any, *, target_return: float = 0.0) -> float
```

#### Description

Return mean excess forecast return divided by downside semideviation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `returns` | positional or keyword | `Any` | `required` |
| `target_return` | keyword only | `float` | `0.0` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.sortino_ratio(...)
```
### success_ratio

Qualified name: `macroforecast.metrics.success_ratio`

#### Signature

```python
macroforecast.metrics.success_ratio(y_true: Any, y_pred: Any, y_prev: Any) -> float
```

#### Description

Directional hit rate relative to a previous actual value.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `y_prev` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.success_ratio(...)
```
### theil_u1

Qualified name: `macroforecast.metrics.theil_u1`

#### Signature

```python
macroforecast.metrics.theil_u1(y_true: Any, y_pred: Any) -> float
```

#### Description

Theil U1 inequality coefficient.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.theil_u1(...)
```
### theil_u2

Qualified name: `macroforecast.metrics.theil_u2`

#### Signature

```python
macroforecast.metrics.theil_u2(y_true: Any, y_pred: Any, y_prev: Any) -> float
```

#### Description

Proportional-change Theil U relative to a no-change forecast.

Each squared error is normalised by ``y_prev`` (the Theil/Bliemel
proportional-change form). This differs from ``forecast::accuracy``'s
"Theil's U", which is the unweighted RMSE ratio (model vs naive).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y_true` | positional or keyword | `Any` | `required` |
| `y_pred` | positional or keyword | `Any` | `required` |
| `y_prev` | positional or keyword | `Any` | `required` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.metrics.theil_u2(...)
```
