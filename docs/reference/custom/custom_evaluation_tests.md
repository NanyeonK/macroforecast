# Custom Evaluation And Tests

[Back to custom extensions](index.md)

Use custom metrics when the output is a scalar forecast score. Use
`custom_test()` when the output is a statistical forecast-comparison result.
Use custom aggregation mappings when the evaluation report needs project-local
slices.

## Custom Metrics

Custom point metrics are plain callables:

```python
def mean_bias(y_true, y_pred):
    return float(pandas.Series(y_pred).sub(pandas.Series(y_true)).mean())

scores = mf.metrics.evaluate_forecasts(
    forecast_table,
    metrics=("mse", mean_bias),
)
```

### Metric Callable Contract

```python
metric(y_true, y_pred) -> float
```

The callable must return one scalar. The output column uses the callable name
unless the surrounding evaluation function renames it.

## custom_test

```python
mf.tests.custom_test(
    name,
    func,
    *args,
    alternative="two-sided",
    **params,
) -> mf.tests.TestResult
```

### Test Callable Contract

The callable receives `*args` plus `**params` and should return either a
mapping or a `TestResult`-like object containing:

| Field | Meaning |
| --- | --- |
| `statistic` | Test statistic. |
| `p_value` | P-value, or `None` if unavailable. |
| `decision` | Optional reject flag. |
| `n_obs` | Number of aligned observations. |
| `metadata` | Optional source, null hypothesis, reference distribution, or warning metadata. |

### Example

```python
def my_loss_test(loss_a, loss_b):
    diff = pandas.Series(loss_a).sub(pandas.Series(loss_b)).dropna()
    return {
        "statistic": float(diff.mean()),
        "p_value": 0.04,
        "n_obs": len(diff),
    }

test = mf.tests.custom_test("my_loss_test", my_loss_test, loss_a, loss_b)
```

## Custom Evaluation Slices

```python
report = mf.evaluation.evaluate_report(
    forecast_result,
    metrics=("mse", mean_bias),
    aggregations={
        "model_target": ("model", "target"),
        "model_regime": ("model", "regime"),
    },
)
```

Custom aggregations create additional `EvaluationReport.aggregations` tables.
They do not change raw metric definitions.

## Output Flow

```python
main_table = mf.reporting.test_report_table({"custom": test})
manifest = mf.output.write_artifacts(
    {"scores": scores, "custom_test": test.to_dict()},
    "results/custom_eval",
)
```
