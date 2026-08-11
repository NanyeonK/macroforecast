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
  The default applies the Harvey-Leybourne-Newbold small-sample correction and
  uses a Student-t reference with `df=n_obs-1`, matching the package's
  `forecast::dm.test` parity contract. Use `test_options={"dm":
  {"small_sample": False}}` only when a replication design needs the plain
  Diebold-Mariano (1995) statistic and asymptotic standard-normal p-value, such
  as MATLAB oracles that report the uncorrected DM statistic.
- **Clark-West (CW)**: adjusts the DM test for the finite-sample upward bias of
  a larger nested model. Valid only when the benchmark is nested within the
  contender (declare `nested_in_benchmark=True` on the arm). The pipeline emits
  CW only for arms that declare nesting; CW is silently invalid otherwise.
- **Additional pairwise tests**: opt in with `"gw"` (Giacomini-White conditional
  predictive ability), `"gr"` (Giacomini-Rossi fluctuation), `"enc_new"` /
  `"enc_t"` (nested encompassing), `"mz"` (Mincer-Zarnowitz forecast-rationality
  regression), or `"pt"` / `"hm"` / `"ag"` (directional accuracy). Directional
  tests evaluate the contender's own sign skill on the same benchmark-aligned
  origins. Degenerate directional forecasts are reported with
  `status="degenerate"` rows rather than aborting evaluation; ENC-NEW/ENC-T rows
  without a p-value or configured critical value are marked
  `status="inconclusive"`.
- **Joint multi-horizon tests**: `"uspa"` and `"aspa"` run Quaedvlieg-style
  uniform and average SPA jointly across all horizons for each
  target/contender/benchmark triple. They require at least two horizons and land
  in `report.significance` with `horizon="joint"`.
- **Model Confidence Set (MCS)**: identifies the set of models that cannot be
  statistically distinguished from the best model at a given significance level
  (`mcs_alpha`). The elimination is the iterative Hansen-Lunde-Nason one, and it
  is the only one implemented: `mcs_method` is reserved, accepts `"iterative"`
  alone, and the resolved value is recorded in
  `report.provenance["spec_echo"]["evaluation"]["mcs_method"]`.
- **Full-set benchmark tests**: `"spa"`, `"rc"`, and `"stepm"` compare the full
  contender set against the benchmark and land in `report.mcs` alongside MCS.
  They require the `arch` extra (`pip install "macroforecast[arch]"`) and carry
  a dependent-loss size caveat; prefer `model_confidence_set` or `uspa`/`aspa`
  when serial dependence in losses is central to the inference.

Tests that estimate a HAC or lag-truncated long-run variance accept fixed lag
overrides through `test_options`. Use `hac_lags` when a replication design pins a
Newey-West bandwidth rather than deriving it from the forecast horizon:

```python
evaluation = mf.pipeline.EvalSpec(
    benchmark="AR",
    tests=("dm", "cw", "gw", "enc_t", "gr", "mz"),
    test_options={
        "dm": {"hac_lags": 4},
        "cw": {"hac_lags": 4},
        "gw": {"hac_lags": 4},
        "enc_t": {"hac_lags": 4},
        "gr": {"hac_lags": 4},
        "mz": {"hac_lags": 4},
    },
)
```

`hac_lags` must be an integer greater than or equal to zero and is validated when
`pipeline_spec` is built. For `"gr"`, `hac_lags` is the paper-facing alias for the
legacy `lag_truncate` option and takes precedence if both are supplied.

An option the evaluator supplies itself is refused at `pipeline_spec` time, with
a pointer to what really owns it, rather than being accepted and then overwritten
when the test runs:

| tests | refused | owner |
|---|---|---|
| `"mcs"`, `"spa"`, `"rc"`, `"stepm"` | `loss`, `model`, `origin`, `target`, `horizon`; plus `benchmark` except for `"mcs"` | `EvalSpec.loss` / `EvalSpec.benchmark`, or the loss panel's own schema — the pipeline builds that panel |
| `"dm"`, `"cw"`, `"gw"`, `"enc_t"` | `horizon` | the horizon of the cell being evaluated, from `pipeline_spec(horizons=...)` |
| `"dm"` | `input_type` | the pipeline, which passes losses it has already computed |
| `"pt"`, `"hm"`, `"ag"`, `"gr"` | `method` | the requested test name in `EvalSpec.tests` |
| `"uspa"`, `"aspa"` | `statistic` | the requested test name in `EvalSpec.tests` |

Everything you genuinely control is unaffected: `alpha`, `hac_lags`, `threshold`,
`kernel`, `correction`, `small_sample`, `cw_adjustment`, `critical_value`,
`instruments`, `"gr"`'s `lag_truncate`, and the bootstrap parameters (`n_boot`,
`block_length`, `bootstrap_method`, `studentize`, an explicit `random_state`).

## When a test cannot be run

A pairwise test needs at least **8** origins where the contender, the benchmark,
and the realised target are all observed; a joint multi-horizon test (`"uspa"` /
`"aspa"`) needs at least **4** such origins on each of at least **two** horizons,
and then at least 4 origins common to all of them once aligned. These minimums
are not adjustable: a Diebold-Mariano statistic on five origins is not a weaker
result, it is a number with no sampling distribution behind it.

A cell that misses a minimum is **reported, not dropped**. It appears in
`report.significance` with `status="degraded"`, `n_obs` set to the sample that was
actually available, `statistic`/`p_value` left `NaN`, and a `reason` naming the
requirement and the evidence:

```
insufficient joint sample: only 1 horizon(s) reached the 4 common origins a
horizon needs, and a joint test needs at least 2 such horizons (observed common
origins -- h=1: 9, h=2: 3); not computed.
```

`evaluate()` also emits a `RuntimeWarning` counting the degraded rows. Before
this, such cells vanished silently, so a missing row in a results table was
indistinguishable from a test that ran and found nothing.

## Multiple testing

Comparing N contenders against one benchmark is N tests. `EvalSpec(
multiple_testing=...)` controls the family-wise error rate or false discovery
rate across the contenders **within one cell**; cells are adjusted independently,
because a cell is what a reader scans at once looking for the winner.

- `"bonferroni"` / `"holm"` / `"bh"` are closed-form. Wide DM/CW columns gain
  `dm_p_adj` / `cw_p_adj`; long-form rows (`"gw"`, `"mz"`, the directional
  tests) gain `p_value_adj`, grouped by `(target, horizon, test)` so that one
  test's contenders form the family -- not a pool of statistics answering
  different questions.
- `"romano_wolf"` resamples the contenders' loss differentials jointly, so it
  inherits their cross-sectional correlation instead of assuming the worst case
  and is markedly less conservative. **Each wide family is resampled from its own
  series**: `dm_p_adj` from the raw loss differential `loss_b - loss_c`, and
  `cw_p_adj` from the Clark-West improvement
  `loss_b - loss_c + (f_b - f_c)^2` — the same series `clark_west_test` forms,
  and the unadjusted difference when you set
  `test_options={"cw": {"cw_adjustment": False}}`. So `dm_p_adj` and `cw_p_adj`
  answer their own tests and generally differ, as their raw p-values do.

  Stated narrowly, this aligns the *inputs*: each family resamples the
  observation-level series its own test is built on. The mean of that series is
  not the test statistic — `dm_test` and `clark_west_test` studentize it with
  their own HAC and reference rules, and Romano-Wolf re-studentizes it for the
  step-down bootstrap. Nothing here claims the bootstrap reproduces either test's
  reference distribution.

  Long-form rows are the exception. The retained panels are
  contender-vs-benchmark series and cannot reconstruct a long-form test's own
  statistic and null: Giacomini-White is itself a loss-differential test, yet its
  statistic is conditional on instruments the adjustment step never sees, and
  Mincer-Zarnowitz and the directional tests are not loss-differential statistics
  at all. Those rows therefore keep `p_value_adj` as `NaN` and a single
  `RuntimeWarning` names the tests left unadjusted, rather than reporting a
  step-down p-value for a statistic that was never resampled. Use a closed-form
  method when the long-form tests are the ones you need controlled.

A row whose p-value is `NaN` -- degraded, inconclusive, or a test like `"gr"` that
reports a critical value instead -- is excluded from its family rather than
counted in it.

## Calibration tests

`calibration_alpha` is the significance level of the `"berkowitz"` and
`"pit_autocorr"` tests. It does **not** govern `"coverage"`: that test checks an
interval against its own nominal level, derived from the widest symmetric
quantile pair in `quantile_predictions` (a 5%/95% pair is a 90% interval, so
`alpha=0.10`), because the nominal coverage of an interval is a property of the
interval rather than a reporting choice. `calibration_alpha` does not affect
`mcs_alpha` either.

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

`EvalSpec` declares the benchmark arm, which metrics and tests to compute,
per-test options, MCS alpha, and optional evaluation-window subsamples. Pass it
to `pipeline_spec`.

```python
from macroforecast.pipeline import EvalSpec, SubsampleWindow

evaluation = EvalSpec(
    benchmark="AR",
    metrics=("rmse", "relative_mse", "r2_oos"),
    tests=("dm", "cw", "mcs", "spa", "uspa", "mz"),
    test_options={"spa": {"n_boot": 999, "block_length": 5},
                  "uspa": {"n_boot": 999, "block_length": 3},
                  "dm": {"hac_lags": 4}},
    cw_for_nested=True,    # compute CW only for arms with nested_in_benchmark=True
    mcs_alpha=0.10,
    subsamples={
        "full": SubsampleWindow(),
        "ex_covid": SubsampleWindow(exclude=(("2020-03-01", "2021-12-31"),)),
        "post_gfc": SubsampleWindow(start="2010-01-01"),
        "nber_recession": SubsampleWindow(mask="nber_recession"),
        "nber_expansion": SubsampleWindow(mask="nber_expansion"),
    },
)
```

The accuracy table, significance tests, and Model Confidence Set are produced by
`run_pipeline`. Subsamples filter the already-produced forecast frame by target
date before scoring; they do not refit models. `SubsampleWindow(mask=...)`
intersects the date window with a boolean state series. Pass a date-indexed
boolean `Series`, a `{date: bool}` mapping, or the named masks
`"nber_recession"` / `"nber_expansion"`. The NBER masks fetch `USREC` for
month-start targets and `USRECQ` for quarter-start targets through the raw FRED
cache, then record the raw-file hash in report provenance.

Mask dates must exactly cover the forecast target dates being evaluated. A
month-end mask will not be silently shifted onto month-start forecasts, and
missing mask dates or `NaN` states raise with the first missing target dates.
Clark-West is emitted only for contenders that **declare** they nest the
benchmark. Arms do so with `Arm(nested_in_benchmark=True)`; a forecast
combination does so with `CombinationContender(nested_in_benchmark=True)`, which
is what the combination literature's headline test needs. A simple pool
(mean/median/trimmed) of arms that each nest the benchmark nests it too, since
zeroing every slope returns the benchmark forecast; an estimated-weight
combination need not, so the package does not infer the flag from the members.
Without it the combination's `cw_stat`/`cw_p` are NaN, and a `UserWarning` says so.

When subsamples are configured, evaluation tables include a `subsample` column,
and paper tables can select a window with
`mf.reporting.paper_accuracy_table(report, subsample="ex_covid")`.
See the runnable [Getting Started](../getting_started.md) snippets and the
[Replication Gallery](../gallery.md) for the full report objects in context.

## Reference

- [Evaluation reference page](../../reference/evaluation.md) — `evaluate_report`, `EvalSpec`, `DEFAULT_METRICS`, `DEFAULT_SCORE_BY`.
- [Metrics reference page](../../reference/metrics.md) — `rmse`, `relative_mse`, `r2_oos`, `mae`, and the full scoring function list.
- [Tests reference page](../../reference/tests.md) — Diebold-Mariano, Clark-West, and MCS implementations.
