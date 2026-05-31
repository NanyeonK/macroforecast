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
| `macroforecast.evaluation` | Multi-slice evaluation reports, benchmark comparisons, and regime scoring. | Raw metric functions or statistical test functions. |

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
    regimes=None,
    regime_column="regime",
    target_column="target",
    state_column="state",
    time_frequency=None,
    time_column="date",
    time_bucket_column="time_bucket",
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
| `regimes` | mapping, Series, str, or `None` | `None` | Date-to-regime labels, existing regime column name, or no extra regime attachment. |
| `regime_column` | str | `"regime"` | Column used for regime scoring. |
| `target_column`, `state_column` | str | `"target"`, `"state"` | Optional slice columns when present. |
| `time_frequency` | str or `None` | `None` | Pandas period frequency such as `"M"`, `"Q"`, `"A"` for time-bucket aggregation. |
| `include_combined` | bool | `True` | Include forecast-combination rows. |

### Output

Returns `EvaluationReport`.

| Field | Type | Meaning |
| --- | --- | --- |
| `scores` | `DataFrame` | Main metric table over `score_by`. |
| `ranking` | `DataFrame` | Ranked `scores` table. |
| `aggregations` | `dict[str, DataFrame]` | Extra metric tables by requested or auto-discovered slices. |
| `benchmark` | `DataFrame` or `None` | Candidate rows relative to `benchmark_model`. |
| `regime` | `DataFrame` or `None` | Regime-specific metric table when regime labels are available. |
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
| Multi-slice report with ranking, benchmark, regime, target/state/time aggregation | `mf.evaluation` |
| Diebold-Mariano, Clark-West, MCS, residual tests | `mf.tests` |
