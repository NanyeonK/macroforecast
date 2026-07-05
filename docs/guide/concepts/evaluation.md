# Evaluation

[Back to User Guide](../index.md)

`macroforecast.pipeline` performs automatic evaluation when `run_pipeline`
completes. It computes accuracy metrics, runs forecast-comparison tests, and
identifies the Model Confidence Set for every (target, horizon) combination. Raw
metric functions live in `macroforecast.metrics`, forecast-comparison statistical
tests live in `macroforecast.tests`, and `macroforecast.evaluation` provides
multi-slice evaluation reports combining both.

## Accuracy metrics

The pipeline reports several metrics per (contender, target, horizon) cell. The
two most important for model comparison are:

- **RMSE** (`"rmse"`): root mean squared forecast error over the test origins.
- **relative MSE** (`"relative_mse"`): the ratio of contender MSE to benchmark
  MSE. A value below 1.0 means the contender beats the benchmark. This is the
  standard metric in the macro forecasting literature (not relative RMSE, which
  would be the square root of this ratio).

`relative_mse` and relative RMSE are related by `relative_mse = (relative_RMSE)^2`,
but they are not the same quantity. The `EvalSpec` default uses `"relative_mse"`;
to report the square-root version, add a post-processing step or use
`mf.metrics.relative_rmse` directly.

## Evaluation sample

Per-contender metrics (RMSE, relative MSE, OOS-R2) are scored on each contender's
pairwise common sample with the benchmark, meaning the origins where both that
contender and the benchmark have a forecast and the realised target is observed.
A contender whose feature block starts late (for example a raw lag of a series
that only begins mid-sample) is therefore scored on its own shorter window
without truncating the other contenders. `n_common` in the accuracy table is
per-contender, and the pipeline emits a `RuntimeWarning` when coverage is ragged
so the heterogeneity is visible rather than silent.

The Model Confidence Set is different. It needs a single joint sample where every
candidate is observed, so it uses the listwise-common sample across all
contenders. The accuracy table and the MCS therefore rest on different samples by
design: a pairwise relative metric uses all the data each pair shares, while the
joint MCS needs origins common to every candidate.

## Forecast comparison tests

The pipeline runs statistical forecast comparison tests across all contenders:

- **Diebold-Mariano (DM)**: tests whether contender and benchmark have equal
  predictive accuracy. Valid for any pair of forecasts (nested or non-nested).
- **Clark-West (CW)**: adjusts the DM test for the finite-sample upward bias of
  a larger nested model. Valid only when the benchmark is nested within the
  contender (declare `nested_in_benchmark=True` on the arm). The pipeline emits
  CW only for arms that declare nesting; CW is silently invalid otherwise.
- **Model Confidence Set (MCS)**: identifies the set of models that cannot be
  statistically distinguished from the best model at a given significance level
  (`mcs_alpha`). Uses the iterative elimination algorithm by default.

## Choosing the benchmark

The relative metrics (`relative_mse`, `r2_oos`) and the comparison tests score
every contender against one benchmark, named by `EvalSpec(benchmark=...)`. The
benchmark is itself an arm, so it is fit and forecast like any other contender,
and you can point it at whichever arm you want. The relative metrics divide by
that arm's realised forecast error.

```python
evaluation = mf.pipeline.EvalSpec(benchmark="AR")   # any arm name
```

Because an arm is just a model with its preprocessing and features, the benchmark
can be the same model as the contenders under a different configuration. A common
design is a base model as the benchmark and enhanced variants as the contenders,
for example a plain random forest on the base features scored against random
forests that add feature blocks (MARX, factors). User-defined models built with
`mf.custom_model` are arms too, so a custom model works as a contender and as the
benchmark.

```python
arms = [
    mf.pipeline.Arm("RF_base", model="random_forest", features=base_features, is_benchmark=True),
    mf.pipeline.Arm("RF_MARX", model="random_forest", features=marx_features),
    mf.pipeline.Arm("RF_factors", model="random_forest", features=factor_features),
    mf.pipeline.Arm("my_model", model=mf.custom_model("my_model", my_fit_func)),
]
evaluation = mf.pipeline.EvalSpec(benchmark="RF_base")   # every arm scored vs base RF
```

The benchmark is matched by contender name within each `(target, horizon)` cell,
which is enough when the benchmark shares the forecast policy of the contenders.

### A benchmark from another policy (or any fixed benchmark)

Sometimes the benchmark you want is produced under a different forecast policy
than the contenders. The GCLS (2021) appendix, for instance, scores both its
direct and its path-average tables against a single FM benchmark, the direct FM.
`run_pipeline` accepts several policies for one target in a single spec, so run
them together and score with `evaluate_cross_policy`, which makes each
`(arm, forecast_policy)` its own contender and scores all of them against the one
benchmark policy you name:

```python
report = mf.pipeline.run_pipeline(mf.pipeline.pipeline_spec(
    data=bundle,
    targets=[
        mf.pipeline.TargetSpec("Y", transform="value", policy="direct_average"),
        mf.pipeline.TargetSpec("Y", transform="value", policy="path_average"),
    ],
    arms=[fm_arm, rf_arm, ar_arm],
    horizons=[1, 3, 6, 12, 24],
    window=window,
    evaluation=mf.pipeline.EvalSpec(benchmark="FM"),
))

# every contender, direct and path, scored against the DIRECT FM
acc = mf.pipeline.evaluate_cross_policy(
    report.forecasts, benchmark="FM", benchmark_policy="direct_average",
)
```

The returned table has one row per `(target, horizon, arm, forecast_policy)` with
`relative_mse` / `r2_oos` / `rmse` computed against the fixed benchmark, and keeps
`arm` and `forecast_policy` as their own columns.

This is also the safety note for multi-policy specs. `accuracy_table` keys the
relative metrics on contender name within a `(target, horizon)` cell and does not
split on policy. If you run more than one policy for a target in a single spec and
score with the plain accuracy table, the two policies' rows for the same arm are
pooled and the relative metrics mix them. `evaluate_cross_policy` qualifies the
contender by `forecast_policy` for you and is the recommended path.

## Key Callable

`EvalSpec` declares the benchmark arm, which metrics and tests to compute, and
MCS settings. Pass it to `pipeline_spec`.

```python
from macroforecast.pipeline import EvalSpec

evaluation = EvalSpec(
    benchmark="AR",
    metrics=("rmse", "relative_mse", "r2_oos"),
    tests=("dm", "cw", "mcs"),
    by=("target", "horizon"),
    cw_for_nested=True,    # compute CW only for arms with nested_in_benchmark=True
    mcs_alpha=0.10,
    mcs_method="iterative",
)
```

The accuracy table, significance tests, and Model Confidence Set are produced by
`run_pipeline`. See the runnable [Getting Started](../getting_started.md) snippets
and the [Replication Gallery](../gallery.md) for the full report objects in
context.

## Reference

- [Evaluation reference page](../../reference/evaluation.md) — `evaluate_report`, `EvalSpec`, `DEFAULT_METRICS`, `DEFAULT_SCORE_BY`.
- [Metrics reference page](../../reference/metrics.md) — `rmse`, `relative_mse`, `r2_oos`, `mae`, and the full scoring function list.
- [Tests reference page](../../reference/tests.md) — Diebold-Mariano, Clark-West, and MCS implementations.
