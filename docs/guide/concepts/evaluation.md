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
contenders. The accuracy table and the MCS can therefore rest on different
samples by design. This is correct, because a pairwise relative metric should use
all the data each pair shares while a joint comparison needs common origins.

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
