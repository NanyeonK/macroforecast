# macroforecast.evaluation

[Back to reference](index.md)

Report-level aggregation, benchmark comparison, regime scores, and namespace access to metrics/tests.

Guide context: [../guide/concepts/evaluation.md](../guide/concepts/evaluation.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `BENCHMARK_METRICS` | data | Built-in immutable sequence. |
| `DEFAULT_METRICS` | data | Built-in immutable sequence. |
| `DEFAULT_SCORE_BY` | data | Built-in immutable sequence. |
| `EvaluationReport` | class | Container returned by :func:`evaluate_report`. |
| `aggregate_scores` | function | Evaluate the same forecasts over multiple explicit groupings. |
| `benchmark_comparison` | function | Evaluate candidate models relative to one benchmark model. |
| `error_decomposition` | function | Decompose forecast MSE into squared bias and residual variance. |
| `evaluate_report` | function | Build a multi-slice forecast evaluation report. |
| `filter_oos_period` | function | Return forecast rows restricted to an out-of-sample date interval. |
| `metrics` | module | No public docstring is available. |
| `regime_scores` | function | Evaluate forecasts by regime labels. |
| `tests` | module | No public docstring is available. |

## Data And Module Values

### `BENCHMARK_METRICS`

Kind: `data`

```python
BENCHMARK_METRICS = ("mse", "mae", "relative_mse", "relative_mae", "mse_reduction", "r2_oos")
```
### `DEFAULT_METRICS`

Kind: `data`

```python
DEFAULT_METRICS = ("mse", "rmse", "mae")
```
### `DEFAULT_SCORE_BY`

Kind: `data`

```python
DEFAULT_SCORE_BY = ("model", "horizon")
```
### `metrics`

Kind: `module`

```python
metrics = <module macroforecast.metrics>
```
### `tests`

Kind: `module`

```python
tests = <module macroforecast.tests>
```

## Callable And Class Reference

### EvaluationReport

Qualified name: `macroforecast.evaluation.report.EvaluationReport`

#### Signature

```python
macroforecast.evaluation.EvaluationReport(scores: pd.DataFrame, ranking: pd.DataFrame, aggregations: dict[str, pd.DataFrame] = <factory>, benchmark: pd.DataFrame | None = None, regime: pd.DataFrame | None = None, decomposition: pd.DataFrame | None = None, metadata: dict[str, Any] = <factory>, metadata_schema: dict[str, Any] = <factory>) -> None
```

#### Description

Container returned by :func:`evaluate_report`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `scores` | positional or keyword | `pd.DataFrame` | `required` |
| `ranking` | positional or keyword | `pd.DataFrame` | `required` |
| `aggregations` | positional or keyword | `dict[str, pd.DataFrame]` | `<factory>` |
| `benchmark` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `regime` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `decomposition` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata_schema` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.evaluation.EvaluationReport(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `scores` | `pd.DataFrame` | `required` |
| `ranking` | `pd.DataFrame` | `required` |
| `aggregations` | `dict[str, pd.DataFrame]` | `default_factory` |
| `benchmark` | `pd.DataFrame \| None` | `None` |
| `regime` | `pd.DataFrame \| None` | `None` |
| `decomposition` | `pd.DataFrame \| None` | `None` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `metadata_schema` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### aggregate_scores

Qualified name: `macroforecast.evaluation.report.aggregate_scores`

#### Signature

```python
macroforecast.evaluation.aggregate_scores(forecasts: Any, *, groupings: Mapping[str, Sequence[str]] | Sequence[Sequence[str]], metrics: Sequence[str | MetricLike] = ('mse', 'rmse', 'mae'), benchmark_model: str | None = None) -> dict[str, pd.DataFrame]
```

#### Description

Evaluate the same forecasts over multiple explicit groupings.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `groupings` | keyword only | `Mapping[str, Sequence[str]] \| Sequence[Sequence[str]]` | `required` |
| `metrics` | keyword only | `Sequence[str \| MetricLike]` | `("mse", "rmse", "mae")` |
| `benchmark_model` | keyword only | `str \| None` | `None` |

#### Returns

`dict[str, pd.DataFrame]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.evaluation.aggregate_scores(...)
```
### benchmark_comparison

Qualified name: `macroforecast.evaluation.report.benchmark_comparison`

#### Signature

```python
macroforecast.evaluation.benchmark_comparison(forecasts: Any, *, benchmark_model: str, by: Sequence[str] = ('model', 'horizon'), metrics: Sequence[str | MetricLike] = ('mse', 'mae', 'relative_mse', 'relative_mae', 'mse_reduction', 'r2_oos')) -> pd.DataFrame
```

#### Description

Evaluate candidate models relative to one benchmark model.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `benchmark_model` | keyword only | `str` | `required` |
| `by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `metrics` | keyword only | `Sequence[str \| MetricLike]` | `("mse", "mae", "relative_mse", "relative_mae", "mse_reduction...` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.evaluation.benchmark_comparison(...)
```
### error_decomposition

Qualified name: `macroforecast.evaluation.report.error_decomposition`

#### Signature

```python
macroforecast.evaluation.error_decomposition(forecasts: Any, *, by: Sequence[str] = ('model', 'horizon'), actual: str = "actual", prediction: str = "prediction") -> pd.DataFrame
```

#### Description

Decompose forecast MSE into squared bias and residual variance.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `actual` | keyword only | `str` | `"actual"` |
| `prediction` | keyword only | `str` | `"prediction"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.evaluation.error_decomposition(...)
```
### evaluate_report

Qualified name: `macroforecast.evaluation.report.evaluate_report`

#### Signature

```python
macroforecast.evaluation.evaluate_report(forecasts: Any, *, metrics: Sequence[str | MetricLike] = ('mse', 'rmse', 'mae'), score_by: Sequence[str] = ('model', 'horizon'), aggregations: Mapping[str, Sequence[str]] | Sequence[Sequence[str]] | None = None, rank_metric: str | None = None, rank_by: Sequence[str] | None = None, benchmark_model: str | None = None, benchmark_metrics: Sequence[str | MetricLike] = ('mse', 'mae', 'relative_mse', 'relative_mae', 'mse_reduction', 'r2_oos'), oos_start: Any | None = None, oos_end: Any | None = None, regimes: Mapping[Any, Any] | pd.Series | str | None = None, regime_column: str = "regime", target_column: str = "target", state_column: str = "state", time_frequency: str | None = None, time_column: str = "date", time_bucket_column: str = "time_bucket", include_decomposition: bool = False, decomposition_by: Sequence[str] | None = None, include_combined: bool = True) -> EvaluationReport
```

#### Description

Build a multi-slice forecast evaluation report.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `metrics` | keyword only | `Sequence[str \| MetricLike]` | `("mse", "rmse", "mae")` |
| `score_by` | keyword only | `Sequence[str]` | `("model", "horizon")` |
| `aggregations` | keyword only | `Mapping[str, Sequence[str]] \| Sequence[Sequence[str]] \| None` | `None` |
| `rank_metric` | keyword only | `str \| None` | `None` |
| `rank_by` | keyword only | `Sequence[str] \| None` | `None` |
| `benchmark_model` | keyword only | `str \| None` | `None` |
| `benchmark_metrics` | keyword only | `Sequence[str \| MetricLike]` | `("mse", "mae", "relative_mse", "relative_mae", "mse_reduction...` |
| `oos_start` | keyword only | `Any \| None` | `None` |
| `oos_end` | keyword only | `Any \| None` | `None` |
| `regimes` | keyword only | `Mapping[Any, Any] \| pd.Series \| str \| None` | `None` |
| `regime_column` | keyword only | `str` | `"regime"` |
| `target_column` | keyword only | `str` | `"target"` |
| `state_column` | keyword only | `str` | `"state"` |
| `time_frequency` | keyword only | `str \| None` | `None` |
| `time_column` | keyword only | `str` | `"date"` |
| `time_bucket_column` | keyword only | `str` | `"time_bucket"` |
| `include_decomposition` | keyword only | `bool` | `False` |
| `decomposition_by` | keyword only | `Sequence[str] \| None` | `None` |
| `include_combined` | keyword only | `bool` | `True` |

#### Returns

`EvaluationReport`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.evaluation.evaluate_report(...)
```
### filter_oos_period

Qualified name: `macroforecast.evaluation.report.filter_oos_period`

#### Signature

```python
macroforecast.evaluation.filter_oos_period(forecasts: Any, *, start: Any | None = None, end: Any | None = None, date_column: str = "date") -> pd.DataFrame
```

#### Description

Return forecast rows restricted to an out-of-sample date interval.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `start` | keyword only | `Any \| None` | `None` |
| `end` | keyword only | `Any \| None` | `None` |
| `date_column` | keyword only | `str` | `"date"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.evaluation.filter_oos_period(...)
```
### regime_scores

Qualified name: `macroforecast.evaluation.report.regime_scores`

#### Signature

```python
macroforecast.evaluation.regime_scores(forecasts: Any, *, regimes: Mapping[Any, Any] | pd.Series | str | None = None, regime_column: str = "regime", by: Sequence[str] = ('model', 'horizon', 'regime'), metrics: Sequence[str | MetricLike] = ('mse', 'rmse', 'mae'), benchmark_model: str | None = None) -> pd.DataFrame
```

#### Description

Evaluate forecasts by regime labels.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `regimes` | keyword only | `Mapping[Any, Any] \| pd.Series \| str \| None` | `None` |
| `regime_column` | keyword only | `str` | `"regime"` |
| `by` | keyword only | `Sequence[str]` | `("model", "horizon", "regime")` |
| `metrics` | keyword only | `Sequence[str \| MetricLike]` | `("mse", "rmse", "mae")` |
| `benchmark_model` | keyword only | `str \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.evaluation.regime_scores(...)
```
