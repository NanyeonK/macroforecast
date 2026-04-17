# CLSS Replication Pattern

This page shows the CLSS-style relative-RMSE horse-race **pattern** using the
FRED-MD INDPRO target. It is not a 1-for-1 replication of the original CLSS
paper - see `docs/dev/papers/` for those - but the wiring is identical, so the
pattern transfers directly.

## 1. The setup

CLSS-style horse races report relative RMSE (model / benchmark) for several
benchmarks across a horizon grid. macrocast emits these numbers as standard
metrics fields once you choose `multi_benchmark_suite` and supply a list of
benchmark specs.

## 2. Recipe

```yaml
study:
  name: indpro_clss_pattern
fixed_design:
  raw_dataset: fred_md
  target: INDPRO
  horizons: [1, 3, 6, 12]
  minimum_train_size: 60
  outer_window: expanding
varying_design:
  model_families: [ar, ridge, lasso]
  feature_recipes: [autoreg_lagged_target]
benchmark_config:
  benchmark_family: ar_bic
  benchmark_window: expanding
  benchmark_scope: same_for_all
  max_ar_lag: 6
preprocess_contract:
  target_transform_policy: raw_level
  x_transform_policy: raw_level
  scaling_policy: none
```

The reported relative_rmse field on each metrics-by-horizon entry gives the
single-benchmark CLSS number directly.

## 3. Multi-benchmark variant

To report several benchmarks at once, swap to a multi suite. The resolver
populates one row per (origin, benchmark_name); the metrics suite returns a
dict-of-dicts.

```python
from macrocast.execution.evaluation import (
    BenchmarkSpec,
    resolve_benchmark_suite,
    compute_relative_metrics_suite,
)

suite = [
    BenchmarkSpec(benchmark_model="historical_mean"),
    BenchmarkSpec(benchmark_model="random_walk"),
    BenchmarkSpec(benchmark_model="ar_bic", max_p=6),
    BenchmarkSpec(benchmark_model="rolling_mean",
                  estimation_window="rolling", window_len=60),
]
benchmark_forecasts = resolve_benchmark_suite(
    target_series=indpro,
    horizon=1,
    suite=suite,
    train_origins=indpro.index[60:],
)
metrics = compute_relative_metrics_suite(
    model_predictions=model_pred,
    benchmark_forecasts=benchmark_forecasts,
    actuals=indpro,
)
```

The resulting metrics["relative_rmse"] is a benchmark_name -> scalar mapping,
ready to dump into a horse-race table.

## 4. Emitting a CLSS-style table

```python
import pandas as pd

cells = dict(
    historical_mean=[metrics["relative_rmse"]["historical_mean"]],
    random_walk=[metrics["relative_rmse"]["random_walk"]],
    ar_bic=[metrics["relative_rmse"]["ar_bic"]],
    rolling_mean=[metrics["relative_rmse"]["rolling_mean"]],
)
table = pd.DataFrame(cells, index=["model_relative_rmse"])
print(table.round(3).to_markdown())
```

The output mirrors the rows of Table 2 in CLSS (2017): one column per
benchmark family, one cell per (model, horizon) pair.

## 5. Conventions to remember

- Lower is better for relative_rmse; values below 1 mean the model beats the
  benchmark.
- oos_r2 = 1 - msfe(model) / msfe(benchmark); positive means improvement
  over the benchmark in MSFE terms.
- Zero benchmark loss returns relative_* = 1.0 by macrocast convention.
- Benchmark fits use the same expanding / rolling / fixed window semantics
  as the model fits, so the horse race is apples-to-apples.

## 6. See also

- [User Guide: Benchmarks](../user_guide/benchmarks.md) for the full axis
  catalogue.
- tests/test_benchmark_resolver.py and tests/test_relative_metrics.py for
  executable examples.
