# Benchmarks

Phase 4 promoted three axes to operational status so that every variant in a
sweep can declaratively answer "relative to what?". The relative_msfe,
relative_rmse, relative_mae, and oos_r2 fields are emitted as standard outputs
on every metrics dict.

## 1. The three axes

macrocast describes a benchmark choice as a triple:

| Axis | Layer | Operational values | Stub / future values |
|---|---|---|---|
| benchmark_family | 1_data_task | historical_mean, zero_change, ar_bic, custom_benchmark, rolling_mean, random_walk, ar_fixed_p, ardi, factor_model, expert_benchmark, multi_benchmark_suite | survey_forecast, paper_specific_benchmark, var |
| benchmark_window | 4_evaluation | expanding, rolling, fixed | paper_exact_window |
| benchmark_scope | 4_evaluation | same_for_all, target_specific, horizon_specific | target_horizon_specific |

The `benchmark_family` axis lives on the data/task layer because the choice of
baseline is part of the forecasting task definition; the window and scope axes
control how the baseline is fit and applied at evaluation time.

## 2. When to use each baseline

| Family | Use when |
|---|---|
| historical_mean | series is roughly stationary (nominal money growth, labour-force participation); a no-skill mean is a credible floor |
| random_walk | I(1) levels (log GDP, log price level); the no-change forecast is the textbook benchmark |
| zero_change | growth-rate or first-differenced series; equivalent to "forecast no change in the differenced series" |
| rolling_mean | structural breaks suspected; bound the influence of the distant past with a fixed window |
| ar_bic | autoregressive baseline with data-driven lag length; the standard h-step AR baseline |
| ar_fixed_p | replication of a paper that fixes p (e.g. Stock & Watson 2002 fixes p=4 for monthly INDPRO) |
| ardi | diffusion-index AR augmentation (CLSS-style); use when you care about beating the canonical large-N benchmark |
| factor_model | pure factor regression; useful as a cross-check against ardi |
| expert_benchmark | user-provided callable returning a numeric forecast; covers consultants, NowCasting models, etc. |
| multi_benchmark_suite | report relative metrics against a list of baselines simultaneously (horse-race tables) |

## 3. YAML examples

### 3.1 Single benchmark

```yaml
benchmark_config:
  benchmark_family: ar_bic
  benchmark_window: expanding
  benchmark_scope: same_for_all
  max_ar_lag: 6
  minimum_train_size: 60
```

### 3.2 Rolling-window mean

```yaml
benchmark_config:
  benchmark_family: rolling_mean
  benchmark_window: rolling
  benchmark_window_len: 60
  benchmark_scope: same_for_all
  minimum_train_size: 60
```

### 3.3 Multi-benchmark suite

```yaml
benchmark_config:
  benchmark_family: multi_benchmark_suite
  benchmark_window: expanding
  benchmark_scope: same_for_all
  benchmark_suite:
    - benchmark_model: historical_mean
    - benchmark_model: random_walk
    - benchmark_model: ar_bic
      max_p: 6
  minimum_train_size: 60
```

The runtime currently degrades suite execution to historical_mean for the
inline metrics dict; relative metrics across the full suite are emitted by
calling `compute_relative_metrics_suite` on the `benchmark_resolver` output.

## 4. Estimation windows

| benchmark_window | Behaviour |
|---|---|
| expanding | Refit on all observations available at the forecast origin (the macrocast default) |
| rolling | Refit on the last `benchmark_window_len` observations only |
| fixed | Fit once on the first `benchmark_window_len` observations and never re-fit |
| paper_exact_window | Reserved for explicit (start, end) windows; raises NotImplementedError in v0.6 |

## 5. Scope

| benchmark_scope | Behaviour |
|---|---|
| same_for_all | One BenchmarkSpec applies across all (target, horizon) pairs |
| target_specific | Override per target via `BenchmarkSpec.target_overrides` |
| horizon_specific | Override per horizon via `BenchmarkSpec.horizon_overrides` |
| target_horizon_specific | Reserved for the joint key; v0.6 leaves this in registry_only |

## 6. Relative metrics

`macrocast.execution.evaluation.metrics.compute_relative_metrics` follows the
CLSS convention:

- `relative_msfe = msfe(model) / msfe(benchmark)`
- `relative_rmse = rmse(model) / rmse(benchmark)`
- `relative_mae  = mae(model)  / mae(benchmark)`
- `oos_r2        = 1 - sse(model) / sse(benchmark)`

When the benchmark loss is exactly zero (perfect benchmark on a degenerate
target), the relative_* values default to 1.0 to avoid division by zero. The
existing `_compute_metrics` in `execution/build.py` follows the same
convention, so downstream consumers see a stable schema.

For multi_benchmark_suite recipes, `compute_relative_metrics_suite` returns a
dict-of-dicts where outer keys are the four metric names and inner keys are
the benchmark names. Consumer code should branch on `isinstance(value, dict)`
to handle both single and suite cases.

## 7. Programmatic API

```python
from macrocast.execution.evaluation import (
    BenchmarkSpec,
    resolve_benchmark_forecasts,
    resolve_benchmark_suite,
    compute_relative_metrics,
    compute_relative_metrics_suite,
)
```

`BenchmarkSpec` is a frozen dataclass; build a list of specs and feed it to
`resolve_benchmark_suite` to obtain a stacked DataFrame with columns
`date, forecast_target_date, benchmark_name, benchmark_pred`. That output is
the input to `compute_relative_metrics_suite`.

See [examples/clss_replication_pattern](../examples/clss_replication_pattern.md)
for a complete CLSS-style horse race walk-through.
