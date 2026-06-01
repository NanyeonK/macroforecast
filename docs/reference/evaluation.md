# macroforecast.evaluation

[Back to reference](index.md)

`macroforecast.evaluation` owns evaluation reports. Raw scoring functions still
live in `macroforecast.metrics`, and forecast-comparison statistical tests still
live in `macroforecast.tests`.

```python
import macroforecast as mf

mf.evaluation.metrics is mf.metrics
mf.evaluation.tests is mf.tests
```

The public API contract is:

| Namespace | Owns | Does not own |
| --- | --- | --- |
| `macroforecast.metrics` | Forecast scoring, ranking, metric resolution. | Statistical comparison tests. |
| `macroforecast.tests` | Forecast-comparison tests, density diagnostics, residual diagnostics. | General scoring tables. |
| `macroforecast.evaluation` | Multi-slice evaluation reports, OOS-period filtering, benchmark comparisons, regime scoring, and error decomposition. | Raw metric functions or statistical test functions. |

Public defaults:

| Symbol | Meaning |
| --- | --- |
| `DEFAULT_METRICS` | Default metric tuple used by `evaluate_report(...)`. |
| `DEFAULT_SCORE_BY` | Default grouping columns for score aggregation. |
| `BENCHMARK_METRICS` | Default benchmark-comparison metrics. |

## Public Flow

```python
report = mf.evaluation.evaluate_report(
    forecast_result,
    metrics=("mse", "rmse", "mae", "relative_mse", "r2_oos"),
    benchmark_model="historical_mean",
    time_frequency="Q",
)

scores = report.scores
ranking = report.ranking
by_regime = report.regime
```

## evaluate_report

```python
macroforecast.evaluation.evaluate_report(
    forecasts,
    *,
    metrics=("mse", "rmse", "mae"),
    score_by=("model", "horizon"),
    aggregations=None,
    rank_metric=None,
    rank_by=None,
    benchmark_model=None,
    benchmark_metrics=("mse", "mae", "relative_mse", "relative_mae", "mse_reduction", "r2_oos"),
    oos_start=None,
    oos_end=None,
    regimes=None,
    regime_column="regime",
    target_column="target",
    state_column="state",
    time_frequency=None,
    time_column="date",
    time_bucket_column="time_bucket",
    include_decomposition=False,
    decomposition_by=None,
    include_combined=True,
) -> EvaluationReport
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `forecasts` | `ForecastResult` or `DataFrame` | required | Forecast runner output or forecast-like table. |
| `metrics` | sequence | `("mse", "rmse", "mae")` | Metric names accepted by `mf.metrics.get_metric(...)` or callables. |
| `score_by` | sequence | `("model", "horizon")` | Main score grouping. Columns must exist. |
| `aggregations` | mapping, sequence, or `None` | auto | Extra groupings to evaluate. `None` creates model, horizon, model-horizon, and available target/state/regime/time slices. |
| `rank_metric` | str or `None` | auto | Metric used for `ranking`. Auto preference is `rmse`, `mse`, `mae`, `r2_oos`, `relative_mse`. |
| `rank_by` | sequence or `None` | `score_by` without `model` | Ranking groups. |
| `benchmark_model` | str or `None` | `None` | Model name used for relative metrics and benchmark table. |
| `benchmark_metrics` | sequence | default benchmark metrics | Metrics for `benchmark_comparison`. |
| `oos_start`, `oos_end` | date-like or `None` | `None` | Restrict forecast rows before scoring. Dates are inclusive. |
| `regimes` | mapping, Series, str, or `None` | `None` | Date-to-regime labels, existing regime column name, or no extra regime attachment. |
| `regime_column` | str | `"regime"` | Column used for regime scoring. |
| `target_column`, `state_column` | str | `"target"`, `"state"` | Optional slice columns when present. |
| `time_frequency` | str or `None` | `None` | Pandas period frequency such as `"M"`, `"Q"`, `"A"` for time-bucket aggregation. |
| `include_decomposition` | bool | `False` | Add MSE decomposition into squared bias and residual variance. |
| `decomposition_by` | sequence or `None` | `score_by` | Grouping used by `error_decomposition`. |
| `include_combined` | bool | `True` | Include forecast-combination rows. |

Custom scoring belongs in the `metrics` argument:

```python
def mean_bias(y_true, y_pred):
    return float(pd.Series(y_pred).sub(pd.Series(y_true)).mean())

report = mf.evaluation.evaluate_report(
    forecast_result,
    metrics=("mse", "rmse", mean_bias),
    aggregations={
        "model_target": ("model", "target"),
        "model_regime": ("model", "regime"),
    },
)
```

Custom aggregation slices belong in `aggregations`. The value is a grouping
tuple over existing forecast-table columns; evaluation still uses
`mf.metrics.evaluate_forecasts()` to compute the metric table.

### Output

Returns `EvaluationReport`.

| Field | Type | Meaning |
| --- | --- | --- |
| `scores` | `DataFrame` | Main metric table over `score_by`. |
| `ranking` | `DataFrame` | Ranked `scores` table. |
| `aggregations` | `dict[str, DataFrame]` | Extra metric tables by requested or auto-discovered slices. |
| `benchmark` | `DataFrame` or `None` | Candidate rows relative to `benchmark_model`. |
| `regime` | `DataFrame` or `None` | Regime-specific metric table when regime labels are available. |
| `decomposition` | `DataFrame` or `None` | Error decomposition table when requested. |
| `metadata` | `dict` | Input metadata plus compact `evaluation_report` stage. |

`EvaluationReport.to_dict()` serializes all tables into JSON-ready records.

The metadata stage records options, table row counts, and forecast-table input
shape:

```python
report.metadata["evaluation_report"]
```

## aggregate_scores

```python
macroforecast.evaluation.aggregate_scores(
    forecasts,
    *,
    groupings,
    metrics=("mse", "rmse", "mae"),
    benchmark_model=None,
) -> dict[str, pandas.DataFrame]
```

Evaluates one forecast table over multiple explicit groupings.

```python
tables = mf.evaluation.aggregate_scores(
    result,
    groupings={
        "model": ("model",),
        "model_horizon_target": ("model", "horizon", "target"),
    },
)
```

All requested columns must exist. This function fails loudly instead of silently
dropping unavailable dimensions.

## filter_oos_period

```python
macroforecast.evaluation.filter_oos_period(
    forecasts,
    *,
    start=None,
    end=None,
    date_column="date",
) -> pandas.DataFrame
```

Returns forecast rows inside an inclusive out-of-sample date interval. This is
the callable replacement for an `oos_period` setting. Use it directly when you
want to score only a subsample, or pass `oos_start`/`oos_end` to
`evaluate_report(...)`.

## error_decomposition

```python
macroforecast.evaluation.error_decomposition(
    forecasts,
    *,
    by=("model", "horizon"),
    actual="actual",
    prediction="prediction",
) -> pandas.DataFrame
```

Decomposes MSE within each group as:

```text
mse = bias_squared + residual_variance
```

where `bias` is the mean residual `actual - prediction`. Output columns
include `n`, `mse`, `bias`, `bias_squared`, `residual_variance`,
`bias_share`, and `variance_share`.

## benchmark_comparison

```python
macroforecast.evaluation.benchmark_comparison(
    forecasts,
    *,
    benchmark_model,
    by=("model", "horizon"),
    metrics=("mse", "mae", "relative_mse", "relative_mae", "mse_reduction", "r2_oos"),
) -> pandas.DataFrame
```

Returns candidate model rows with benchmark-relative scores. The benchmark row
itself is removed from the output. `benchmark_model` must be present in the
forecast table.

## regime_scores

```python
macroforecast.evaluation.regime_scores(
    forecasts,
    *,
    regimes=None,
    regime_column="regime",
    by=("model", "horizon", "regime"),
    metrics=("mse", "rmse", "mae"),
    benchmark_model=None,
) -> pandas.DataFrame
```

`regimes` can be:

| Form | Meaning |
| --- | --- |
| `None` | Use an existing `regime_column`. |
| `str` | Use that existing column as the source and copy to `regime_column` when names differ. |
| mapping or `Series` | Map forecast `date` values to regime labels. |

## Boundary

| Question | Use |
| --- | --- |
| One metric value or one metric table | `mf.metrics` |
| Multi-slice report with ranking, OOS filtering, benchmark, regime, target/state/time aggregation, decomposition | `mf.evaluation` |
| Diebold-Mariano, Clark-West, MCS, residual tests | `mf.tests` |
