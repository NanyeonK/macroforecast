# macroforecast.tests

[Back to reference](index.md)

`macroforecast.tests` owns forecast-comparison tests and residual diagnostics.
It does not compute general scoring tables, fit models, or choose windows.

Use the namespace form:

```python
import macroforecast as mf

mf.tests.dm_test(loss_a, loss_b, horizon=1)
```

Top-level shortcuts such as `mf.dm_test(...)` are intentionally not exported.

## TestResult

Most pairwise forecast-comparison tests return `TestResult`.

```python
macroforecast.tests.TestResult(
    statistic,
    p_value,
    decision,
    alternative,
    correction_policy=None,
    n_obs=None,
    metadata={},
)
```

| Field | Meaning |
| --- | --- |
| `statistic` | Test statistic, or `None` when the sample is too small or degenerate. |
| `p_value` | P-value, or `None` when unavailable. |
| `decision` | `True` when the null is rejected at the supplied `alpha`. |
| `alternative` | `two_sided` or `one_sided`. |
| `correction_policy` | HAC or small-sample correction label. |
| `n_obs` | Number of aligned observations used. |
| `metadata` | Test-specific details. |

Methods:

| Method | Output |
| --- | --- |
| `to_dict()` | JSON-ready dictionary with `metadata_schema.kind="forecast_test_result"`. |
| `to_json(path=None)` | JSON text and optional file write. |
| `summary()` | Compact string summary. |

## Custom Tests

### custom_test

```python
macroforecast.tests.custom_test(
    name,
    func,
    *args,
    alternative="two_sided",
    alpha=0.05,
    correction_policy=None,
    metadata=None,
    **params,
) -> TestResult
```

Runs a user-supplied forecast test and coerces the result to `TestResult`.

The callable receives `*args` and `**params`. It may return:

| Return type | Meaning |
| --- | --- |
| `TestResult` | Used directly, with custom metadata merged. |
| mapping | Must contain `statistic` or `stat`, and may contain `p_value`/`pvalue`, `decision`, `alternative`, `correction_policy`, `n_obs`, and `metadata`. |
| `(statistic, p_value)` | Decision is `p_value < alpha`. |
| `(statistic, p_value, n_obs)` | Same as above plus sample size. |

```python
def sign_test_stat(loss_a, loss_b):
    diff = pd.Series(loss_a).sub(pd.Series(loss_b)).dropna()
    return {
        "statistic": float((diff < 0).mean()),
        "p_value": 0.04,
        "n_obs": len(diff),
    }

result = mf.tests.custom_test(
    "sign_loss_test",
    sign_test_stat,
    loss_a,
    loss_b,
)
```

`custom_test()` records the callable name, parameters, `alpha`, and
`custom=True` in `result.metadata`.

## Equal Predictive Accuracy

### dm_test

```python
macroforecast.tests.dm_test(
    loss_a,
    loss_b,
    *,
    horizon=1,
    correction="hln",
    kernel="acf",
    input_type="loss",
    power=2.0,
    alternative="two_sided",
    alpha=0.05,
)
```

Input: two aligned loss series by default. Set `input_type="error"` to match
`forecast::dm.test(e1, e2, h, power, varestimator)` from the R `forecast`
package: the function then computes `abs(e1)^power - abs(e2)^power` internally.
Output: `TestResult` for the Diebold-Mariano equal predictive accuracy test.
`correction="hln"` applies the Harvey-Leybourne-Newbold small-sample
correction. P-values use a Student-t reference distribution with `df=n-1`,
matching `forecast/R/DM2.R::dm.test`.

`kernel="acf"` matches the R `varestimator="acf"` autocovariance estimator.
`kernel="bartlett"` or `"newey_west"` uses the Bartlett-weighted estimator,
matching the R `varestimator="bartlett"` option.

R/source alignment:

| Setting | Alignment |
| --- | --- |
| `input_type="error"`, `correction="hln"`, `kernel="acf"` | Same statistic and Student-t p-value as `forecast/R/DM2.R::dm.test(varestimator="acf")`. |
| `input_type="error"`, `correction="hln"`, `kernel="bartlett"` or `"newey_west"` | Same statistic and Student-t p-value as `forecast/R/DM2.R::dm.test(varestimator="bartlett")`. |
| `input_type="loss"` | Uses the same DM statistic after accepting precomputed losses. This is convenient for custom losses, but it is not a direct call-equivalent to R `forecast::dm.test(e1, e2)`. |
| `correction="none"` | Omits the Harvey-Leybourne-Newbold small-sample factor used by `forecast::dm.test`. |
| `kernel="parzen"` or `"andrews"` | Macroforecast extension. These HAC estimators are not options in R `forecast::dm.test`. |

Returned metadata includes `statistic_type="t"`,
`null_hypothesis="equal predictive accuracy"`, `p_value_status`,
`p_value_reference`, `source_reference`, `r_reference`, `r_alignment`, and
`r_argument_mapping`.

### gw_test

```python
macroforecast.tests.gw_test(
    loss_a,
    loss_b,
    *,
    horizon=1,
    correction="hln",
    kernel="acf",
    input_type="loss",
    power=2.0,
    alternative="two_sided",
)
```

Input: two aligned loss series. Output: `TestResult` using the package's
Giacomini-White-compatible loss differential surface. This callable uses the
same aligned DM-style loss-differential statistic; conditional predictive
ability with time-varying fluctuation paths is exposed separately through
`conditional_predictive_ability_test(...)`.

Source boundary: `gw_test()` does not claim exact R-package alignment. It
preserves the legacy callable surface by reusing the DM/HLN loss-differential
statistic on aligned inputs. For the package's time-varying conditional
predictive-ability path, use `conditional_predictive_ability_test(...)`.

### dmp_test

```python
macroforecast.tests.dmp_test(
    loss_differences,
    *,
    kernel="newey_west",
    alpha=0.05,
)
```

Input: one loss-difference series or a sequence of loss-difference series.
Output: `TestResult` for a stacked Diebold-Mariano-Pesaran-style joint test.

The test stacks finite loss-difference values, computes a HAC standard error
for the stacked mean, and reports a two-sided standard-normal p-value. No exact
R-package comparator is claimed in the checked R sources. Metadata records
`statistic_type="z"`, `null_hypothesis`, `p_value_status`,
`p_value_reference`, `source_reference`, and `r_alignment`.

### equal_predictive_tests

```python
macroforecast.tests.equal_predictive_tests(
    loss_a,
    loss_b,
    *,
    tests=("dm", "gw", "dmp"),
    error_a=None,
    error_b=None,
    horizon=1,
    correction="hln",
    kernel="acf",
    alpha=0.05,
) -> pandas.DataFrame
```

Runs multiple equal-predictive-ability tests and stacks one row per test.
Supported names are `dm`, `gw`, `dmp`, and `hn`. `hn` requires `error_a` and
`error_b` because Harvey-Newbold is an encompassing test on forecast errors.

Output: a `pandas.DataFrame` with one row per requested test. The table keeps
the full component metadata in the `metadata` column and also promotes the
paper-facing fields below to top-level columns.

| Column | Meaning |
| --- | --- |
| `test`, `name` | Requested key and display name. |
| `statistic_type`, `statistic` | Reference family (`t` or `z`) and test statistic. |
| `p_value`, `p_value_status`, `p_value_reference` | P-value, availability flag, and reference distribution. |
| `decision`, `alternative`, `null_hypothesis` | Rejection flag, alternative direction, and null statement. |
| `correction_policy`, `n_obs` | Small-sample/HAC policy and aligned observation count. |
| `source_reference`, `external_reference`, `r_reference`, `r_alignment` | Provenance and source-comparison fields. |
| `metadata` | Full `TestResult.metadata` dictionary for the component test. |

Current source alignment by row:

| Test | R/source status |
| --- | --- |
| `dm` | Exact `forecast::dm.test` alignment only under the settings listed in `dm_test`. |
| `gw` | Legacy GW-compatible DM-style surface; no exact R comparator claimed. |
| `dmp` | Macroforecast stacked HAC screen; no exact R comparator claimed. |
| `hn` | Legacy encompassing covariance approximation; not `forecast::dm.test`. |

For paper output, pass this table to
`macroforecast.reporting.test_report_table(...)`. For an appendix/audit table
that spells out source and R alignment, use
`macroforecast.reporting.test_provenance_table(...)`.

### harvey_newbold_test

```python
macroforecast.tests.harvey_newbold_test(
    error_a,
    error_b,
    *,
    horizon=1,
    kernel="newey_west",
    small_sample=True,
    alpha=0.05,
)
```

Input: two forecast-error series. Output: one-sided `TestResult` for the legacy
forecast-error covariance approximation.

Source note: this is not `forecast::dm.test`. The R `forecast` package function
implements Harvey-Leybourne-Newbold Diebold-Mariano equal-accuracy testing.
`harvey_newbold_test()` remains a callable encompassing-style covariance
approximation and records that distinction in `result.metadata`.

The callable forms `d_t = e_a,t * (e_a,t - e_b,t)`, computes a HAC standard
error, optionally applies an HLN-style small-sample factor, and reports a
one-sided Student-t upper-tail p-value. Metadata records
`statistic_type="t"`, `p_value_status`, `p_value_reference`,
`source_reference`, `r_reference=None`, and `r_alignment`.

Alias: `hn_test`.

## Nested And Encompassing Tests

### clark_west_test

```python
macroforecast.tests.clark_west_test(
    loss_small,
    loss_large,
    forecast_small,
    forecast_large,
    *,
    horizon=1,
    cw_adjustment=True,
    kernel="newey_west",
    alpha=0.05,
)
```

Input: small-model loss, large-model loss, and both forecast series. Output:
one-sided `TestResult` for the Clark-West nested forecast comparison.

Statistic:

```text
q_t = e_r,t^2 - e_u,t^2 + (f_r,t - f_u,t)^2
z = mean(q_t) / sqrt(LRV(q_t) / n)
```

Here `r` is the restricted/small model and `u` is the unrestricted/large model.
The implementation follows the standard adjusted MSPE differential used by
Clark-West references such as GAUSS `cwTest` and HypothesisTests.jl
`ClarkWestTest`. Archived R examples can differ by sign convention, so this
page treats the formula above as the package contract.

Alias: `cw_test`.

### enc_new_test

```python
macroforecast.tests.enc_new_test(
    error_small,
    error_large,
    *,
    critical_value=None,
    alpha=0.05,
)
```

Input: restricted/small-model forecast errors and unrestricted/large-model
forecast errors. Output: one-sided `TestResult`.

Statistic:

```text
c_t = e_r,t * (e_r,t - e_u,t)
ENC-NEW = n * mean(c_t) / mean(e_u,t^2)
```

Default `p_value` is `None` because Clark-McCracken nested forecast
encompassing tests have nonstandard distributions. Pass a design-appropriate
`critical_value` to get a boolean decision.

### enc_t_test

```python
macroforecast.tests.enc_t_test(
    error_small,
    error_large,
    *,
    horizon=1,
    kernel="newey_west",
    critical_value=None,
    normal_approximation=False,
    alpha=0.05,
)
```

Input: restricted/small-model forecast errors and unrestricted/large-model
forecast errors. Output: one-sided `TestResult`.

Statistic:

```text
c_t = e_r,t * (e_r,t - e_u,t)
ENC-T = mean(c_t) / sqrt(LRV(c_t) / n)
```

Default `p_value` is `None`. Set `normal_approximation=True` only for
diagnostic screening, or pass `critical_value` for a design-specific decision.

### nested_tests

```python
macroforecast.tests.nested_tests(
    loss_small,
    loss_large,
    *,
    forecast_small=None,
    forecast_large=None,
    error_small=None,
    error_large=None,
    tests=("clark_west", "enc_new", "enc_t"),
    horizon=1,
    kernel="newey_west",
    enc_critical_value=None,
    enc_normal_approximation=False,
    alpha=0.05,
) -> pandas.DataFrame
```

Runs multiple nested-model tests and stacks one row per test. Clark-West
requires `forecast_small` and `forecast_large`; `enc_new` and `enc_t` require
`error_small` and `error_large`. This separation is intentional because
Clark-West is an adjusted MSPE differential while ENC-NEW and ENC-T are
forecast-error encompassing covariance statistics.

## Directional Accuracy Tests

### directional_accuracy_test

```python
macroforecast.tests.directional_accuracy_test(
    y_true,
    y_pred,
    *,
    threshold=0.0,
    method="pesaran_timmermann",
    alpha=0.05,
)
```

Input: realized values and forecasts. Output: `TestResult`. Supported methods
are `pesaran_timmermann`, `anatolyev_gerko`, and `henriksson_merton`.

The `pesaran_timmermann` and `anatolyev_gerko` branches are aligned with
R `tstests/R/dac.R::dac_test` and `rugarch/R/rugarch-tests.R::DACTest`. The
p-value is a one-sided upper-tail normal p-value, `1 - Phi(statistic)`.
Forecasts that are constant after subtracting `threshold` are rejected because
the directional tests are undefined for a constant sign forecast.

Options:

| Option | Default | Choices | Meaning |
| --- | --- | --- | --- |
| `threshold` | `0.0` | numeric | Values above this threshold are positive-direction observations. |
| `method` | `"pesaran_timmermann"` | `"pesaran_timmermann"`, `"anatolyev_gerko"`, `"henriksson_merton"` | Directional statistic to compute. |
| `alpha` | `0.05` | probability in `(0, 1)` | Rejection level. |

Method notes:

| Method | Null | Statistic input |
| --- | --- | --- |
| `pesaran_timmermann` | No sign predictability. | Exact R alignment with `.pt_test` / `DACTest(test="PT")`: sign hit rate versus independence-implied sign hit rate. |
| `anatolyev_gerko` | No excess profitability. | Exact R alignment with `.ag_test` / `DACTest(test="AG")`: `sign(forecast) * actual` excess profitability, using raw actual and forecast values after threshold subtraction. |
| `henriksson_merton` | No market-timing skill. | Macroforecast extension. No exact comparator in `tstests::dac_test` or `rugarch::DACTest`; statistic is based on up/down conditional hit rates. |

R/source alignment:

| Branch | R comparator | Notes |
| --- | --- | --- |
| `pesaran_timmermann` | `tstests/R/dac.R::.pt_test`; `rugarch/R/rugarch-tests.R::DACTest(test="PT")` | Uses `x_t=1{actual>0}`, `y_t=1{forecast>0}`, `z_t=1{forecast*actual>0}`, and `p.value=1-pnorm(statistic)`. |
| `anatolyev_gerko` | `tstests/R/dac.R::.ag_test`; `rugarch/R/rugarch-tests.R::DACTest(test="AG")` | Uses `r_t=sign(forecast)*actual`, excess-profitability variance `V_EP`, and `p.value=1-pnorm(statistic)`. |
| `henriksson_merton` | None | Kept as a callable screening diagnostic, not claimed as an R-package-aligned DAC branch. |

Zero rule: R uses strict positivity, `actual > 0` and `forecast > 0`.
`macroforecast` applies the same strict rule after subtracting `threshold`, so
values equal to `threshold` are treated as non-positive.

Aliases:

| Alias | Equivalent call |
| --- | --- |
| `pesaran_timmermann_test(y_true, y_pred)` | `directional_accuracy_test(..., method="pesaran_timmermann")` |
| `anatolyev_gerko_test(y_true, y_pred)` | `directional_accuracy_test(..., method="anatolyev_gerko")` |
| `henriksson_merton_test(y_true, y_pred)` | `directional_accuracy_test(..., method="henriksson_merton")` |

## Density And Interval Diagnostics

### density_interval_tests

```python
macroforecast.tests.density_interval_tests(
    pit,
    *,
    alpha=0.05,
    n_bins=10,
    pit_lag=1,
)
```

Input: probability integral transform values in `[0, 1]`. Output: JSON-ready
dictionary with `metadata_schema.kind="density_interval_tests"` plus
Berkowitz, KS, Kupiec POF, Christoffersen independence, Engle-Manganelli DQ,
Du-Escanciano shortfall, PIT histogram, and PIT autocorrelation diagnostics.

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `alpha` | `0.05` | Tail probability for VaR/shortfall-style hit tests. |
| `n_bins` | `10` | Number of PIT histogram bins. |
| `pit_lag` | `1` | Lag used for PIT autocorrelation, Berkowitz AR lag, and Du-Escanciano conditional shortfall lag. |

Output keys:

| Key | Meaning |
| --- | --- |
| `berkowitz` | Berkowitz density LR test plus Jarque-Bera normality check after normal score transform. |
| `ks` | Kolmogorov-Smirnov test against uniform PIT. |
| `kupiec_pof` | Unconditional hit-rate test at `alpha`. |
| `christoffersen_independence` | Markov independence test for hits. |
| `engle_manganelli_dq` | PIT hit-only DQ proxy. Use `dynamic_quantile_test(...)` for the full Engle-Manganelli VaR DQ test. |
| `du_escanciano_shortfall` | Du-Escanciano unconditional and conditional shortfall tests. |
| `pit_histogram` | One record per histogram bin. |
| `pit_autocorrelation` | `TestResult` dictionary for serial PIT dependence. |
| `r_reference`, `r_alignment` | Composite provenance metadata. Component-level diagnostics also carry their own R/source metadata. |

R/source alignment:

| Diagnostic | Reference |
| --- | --- |
| Berkowitz | `tstests/R/berkowitz.R::berkowitz_test`: PIT to normal scores, ARIMA(`pit_lag`,0,0) unrestricted likelihood versus Normal(0,1); LR df is `2 + pit_lag`. |
| Du-Escanciano shortfall | `tstests/R/shortfall_de.R::shortfall_de_test`: cumulative tail shortfall mean test and portmanteau test on centered tail shortfall autocorrelations. |
| Kupiec/Christoffersen | `tstests/R/var_cp.R::var_cp_test` and `rugarch/R/rugarch-tests.R`: Bernoulli/transition likelihood-ratio construction. |
| PIT hit-only DQ proxy | No direct R comparator. It is a PIT-hit lag diagnostic inside this composite wrapper, not the full Engle-Manganelli VaR DQ test. |

Boundary handling: values outside `[0, 1]` raise. Boundary PIT values `0` and
`1` are accepted as PIT values but clipped internally for the normal-score
Berkowitz transform to avoid infinite ARIMA inputs.

### shortfall_de_test

```python
macroforecast.tests.shortfall_de_test(
    pit,
    *,
    alpha=0.05,
    lags=1,
    boot=False,
    n_boot=2000,
    random_state=0,
) -> dict
```

Input: PIT values in `[0, 1]`. Output: JSON-ready dictionary with
`metadata_schema.kind="shortfall_de_test"`.

The unconditional statistic is the sample mean of cumulative tail shortfall,
`mean((alpha - pit) * 1{pit <= alpha} / alpha)`. The conditional statistic is
a portmanteau statistic on autocorrelations of that series centered by
`alpha / 2`. With `boot=False`, the unconditional p-value uses the
Du-Escanciano normal approximation and the conditional p-value uses
`Chi-squared(lags)`. With `boot=True`, both p-values use simulated uniform PIT
draws with the same sample size.

### dynamic_quantile_test

```python
macroforecast.tests.dynamic_quantile_test(
    y_true,
    var,
    *,
    alpha=0.05,
    lag=1,
    lag_hit=1,
    lag_var=1,
) -> TestResult
```

Input: realized values and one-step-ahead lower-tail VaR forecasts. Output:
`TestResult` for the Engle-Manganelli dynamic quantile test.

This is the full VaR DQ callable. It is separate from
`density_interval_tests(...)` because the exact DQ statistic needs realized
values and VaR forecasts, not PIT values alone.

R/source alignment: `segMGarch/R/DQtest.R::DQtest`. The hit series is
`1 - alpha` when `y_true < var` and `-alpha` otherwise. The regressor matrix
contains a constant, lag-aligned VaR forecasts, `lag_hit` lagged hit columns,
and lagged squared realized values. The statistic is
`Hit' X (X'X)^(-1) X' Hit / (alpha * (1 - alpha))`, with a chi-squared
reference distribution using the number of columns of `X`.

R argument mapping: `segMGarch::DQtest` names the VaR probability
`VaR_level` and converts it internally to the lower-tail probability
`1 - VaR_level`. `macroforecast` accepts the lower-tail probability directly
as `alpha`; therefore a 5% lower-tail VaR is `alpha=0.05`, corresponding to
`VaR_level=0.95` in the R function.

Source: https://rdrr.io/cran/segMGarch/src/R/DQtest.R

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `alpha` | `0.05` | Lower-tail probability. A 5% VaR uses `alpha=0.05`. |
| `lag` | `1` | Lag used for squared realized values. |
| `lag_hit` | `1` | Number of lagged hit columns. |
| `lag_var` | `1` | Lag alignment for VaR forecasts. |

### pit_histogram

```python
macroforecast.tests.pit_histogram(pit, *, n_bins=10) -> pandas.DataFrame
```

Returns one row per PIT histogram bin with observed count, expected count under
uniformity, and deviation.

### pit_autocorrelation_test

```python
macroforecast.tests.pit_autocorrelation_test(
    pit,
    *,
    lag=1,
    alpha=0.05,
) -> TestResult
```

Runs a normal-approximation test for serial dependence in PIT values.

### interval_coverage_test

```python
macroforecast.tests.interval_coverage_test(
    y_true,
    lower,
    upper,
    *,
    alpha=0.05,
) -> dict
```

Runs Kupiec POF, Christoffersen independence, combined conditional coverage,
and Christoffersen-Pelletier duration diagnostics for forecast intervals.
`alpha` is the expected non-coverage rate, so a 90% interval uses
`alpha=0.10`.

Boundary cases follow the likelihood-ratio convention used by R
`tstests::var_cp_test` and `rugarch::VaRTest`: zero violations do not
automatically imply a passing Kupiec statistic; the restricted Bernoulli
likelihood is compared with the boundary unrestricted likelihood.

The `christoffersen_pelletier_duration` output follows the duration-test
logic in `tstests/R/var_cp.R::.duration_test`: durations between interval
misses are modeled with a Weibull likelihood, and the no-memory exponential
restriction is tested by setting the Weibull shape parameter to `1`. The
duration statistic is unavailable when there is one or fewer misses.

Coverage output:

| Key | Meaning |
| --- | --- |
| `kupiec_pof` | Unconditional coverage LR. Carries `tstests` and `rugarch` references. |
| `christoffersen_independence` | First-order Markov independence LR plus transition counts `n00`, `n01`, `n10`, `n11`. |
| `christoffersen_conditional_coverage` | Sum of Kupiec and independence LR statistics with chi-squared df 2. |
| `christoffersen_pelletier_duration` | Weibull duration LR for the exponential no-memory restriction. |
| `r_reference`, `rugarch_reference`, `r_alignment` | Package-level provenance. |

Duration likelihood note: the duration construction is the same in
`tstests` and `rugarch`. The implemented density/survival likelihood follows
`rugarch/R/rugarch-tests.R::VaRDurTest`, which is the internally consistent
Christoffersen-Pelletier Weibull likelihood form.

## Conditional Predictive Ability

### conditional_predictive_ability_test

```python
macroforecast.tests.conditional_predictive_ability_test(
    loss_a,
    loss_b,
    *,
    method="giacomini_rossi",
    window_ratio=0.5,
    dmv_fullsample=True,
    lag_truncate=0,
    alpha=0.05,
)
```

Input: two aligned loss series. Output: JSON-ready dictionary with
`metadata_schema.kind="conditional_predictive_ability"`, a fluctuation
statistic, critical value, decision, time path, window size, loss-difference
orientation, and source-alignment metadata.

Supported methods: `giacomini_rossi`, `recursive_fluctuation`.

The `giacomini_rossi` branch is aligned with
`murphydiagram/R/procs.R::fluctuation_test`, which implements Proposition 1 of
Giacomini and Rossi (2010). It computes rolling-window Diebold-Mariano-type
statistics for the loss difference `loss_a - loss_b`, uses Bartlett HAC
variance, and compares the supremum absolute statistic with the tabulated
critical values from Giacomini-Rossi Table 1. Positive path values mean
`loss_a` is larger than `loss_b` over that window, so the final statistic is
two-sided because it uses the supremum absolute path.

R alignment:

| R package / function | macroforecast branch | Alignment |
| --- | --- | --- |
| `murphydiagram/R/procs.R::fluctuation_test` | `method="giacomini_rossi"` | Same `ld <- loss1 - loss2`, same `mu` grid, same Table 1 critical values, same `lag_truncate in 0:5`, same Bartlett HAC convention, same `dmv_fullsample` and rolling-denominator branches. |
| None | `method="recursive_fluctuation"` | Package extension over expanding-prefix loss windows. It reuses the same Bartlett HAC helper but does not claim to implement a named R-package test. |

Options:

| Option | Default | Choices | Meaning |
| --- | --- | --- | --- |
| `method` | `"giacomini_rossi"` | `"giacomini_rossi"`, `"recursive_fluctuation"` | `giacomini_rossi` is R-aligned; `recursive_fluctuation` is a package extension over expanding loss windows. |
| `window_ratio` | `0.5` | `0.1`, `0.2`, ..., `0.9` for `giacomini_rossi` | Rolling window size as a fraction of the evaluation sample. |
| `dmv_fullsample` | `True` | boolean | If `True`, estimate HAC variance on the full loss-difference sample, matching the R default. If `False`, use each rolling window's HAC variance. |
| `lag_truncate` | `0` | `0`, `1`, ..., `5` | Bartlett HAC truncation lag, matching the R package's allowed range. |
| `alpha` | `0.05` | `0.05`, `0.10` for `giacomini_rossi` | Test size used to select the tabulated critical value. |

Output fields:

| Field | Meaning |
| --- | --- |
| `statistic` | Supremum absolute value of the fluctuation path. |
| `time_path` | Rolling or recursive fluctuation path before the supremum is taken. |
| `critical_value`, `critical_band`, `decision` | Tabulated Giacomini-Rossi comparison when available; `None` for the recursive extension. |
| `variance_scope` | `"full_sample"` when `dmv_fullsample=True`; `"rolling_window"` otherwise. |
| `loss_difference_orientation` | Always `loss_a - loss_b`; positive path values mean `loss_a` has larger loss. |
| `source_reference`, `external_reference`, `r_reference`, `r_alignment` | Source and R-package comparison metadata. |
| `requested_method`, `method`, `alias_warning` | The user-supplied method, normalized method, and any alias caveat. |

`method="rossi_sekhposyan"` remains accepted as a legacy alias for
`recursive_fluctuation`, but Rossi-Sekhposyan forecast rationality is a
different test family and is not represented by this loss-comparison callable.

## Multiple-Model Tests

### blocked_oob_reality_check

```python
macroforecast.tests.blocked_oob_reality_check(
    loss_panel,
    *,
    benchmark,
    loss="squared_error",
    alpha=0.05,
    n_boot=1000,
    block_length=4,
    bootstrap_method="fixed_block_bootstrap",
    random_state=0,
    target="target",
    horizon="horizon",
    origin="origin",
    model="model_id",
) -> pandas.DataFrame
```

Block-bootstrap one-sided benchmark-superiority screen against a named
benchmark model. This is the direct callable replacement for the legacy
`blocked_oob_reality_check` operation. It is intentionally documented as a
legacy screen, not as the exact White Reality Check.

Inputs:

| Form | Required columns |
| --- | --- |
| Long panel | `origin`, `model_id`, `squared_error`; optional `target` and `horizon`. Column names are configurable. |
| Wide matrix | One column per model, including the `benchmark` column. The index is treated as origin order. |

Long-panel input must have one row per target/horizon/origin/model key. If the
loss table contains duplicate rows for that key, aggregate them explicitly
before calling; the test helpers do not average duplicates silently.

Output: one row per candidate model and target/horizon group.

| Column | Meaning |
| --- | --- |
| `target`, `horizon` | Group labels. Wide input uses `"all"` for both. |
| `model` | Candidate model tested against the benchmark. |
| `benchmark` | Benchmark model name. |
| `mean_diff` | `benchmark_loss - candidate_loss`; positive means candidate has lower loss. |
| `statistic` | Mean loss differential scaled by bootstrap standard error. |
| `p_value` | Pairwise one-sided block-bootstrap p-value for no improvement over benchmark. |
| `decision` | `True` when `p_value < alpha`. |
| `familywise_p_value` | Max-bootstrap p-value adjusted across all candidate models in the same target/horizon group. |
| `familywise_decision` | `True` when `familywise_p_value < alpha`. |
| `familywise_n_obs` | Complete-case origins used for the family-wise adjustment. |
| `n_obs` | Number of aligned origins. |
| `block_length`, `n_boot`, `bootstrap_method` | Bootstrap settings used. |
| `source_reference`, `r_reference`, `r_alignment` | Provenance metadata. `r_reference` is `None` because this legacy screen has no exact R-package comparator. |

The returned table carries
`attrs["macroforecast_metadata_schema"]["kind"] = "blocked_oob_reality_check"`.

R/source comparison:

| Function | Status |
| --- | --- |
| `blocked_oob_reality_check(...)` | No exact R-package comparator. It computes pairwise and family-wise max-centered block bootstrap p-values from precomputed benchmark/candidate loss differences. |
| `ttrTests/R/dataSnoop.R::dataSnoop(test="RC" or "SPA")` | Strategy-specific data-snooping code. It rebuilds technical-trading parameter-grid performance on each bootstrapped price sample, so it is not the same API contract. |
| `reality_check_test(...)`, `superior_predictive_ability_test(...)`, `stepm_test(...)` | Exact multiple-comparison callable family for White RC, Hansen SPA, and Romano-Wolf StepM using the optional `arch.bootstrap` backend. |

Pipeline integration: request `"spa"`, `"rc"`, or `"stepm"` in
`macroforecast.pipeline.EvalSpec.tests` to run these full-set comparisons per
`(target, horizon)` against `EvalSpec.benchmark`. Per-test bootstrap controls
enter through `EvalSpec.test_options`, for example
`test_options={"spa": {"n_boot": 999, "block_length": 5, "random_state": 123}}`.
The results append rows to `PipelineReport.mcs` with `test`, `contender`,
`superior`, `reject`, `p_value`, and sample-size metadata.

### superior_predictive_ability_test

```python
macroforecast.tests.superior_predictive_ability_test(
    loss_panel,
    *,
    benchmark,
    loss="squared_error",
    alpha=0.05,
    n_boot=1000,
    block_length="auto",
    bootstrap_method="stationary_bootstrap",
    p_value_type="consistent",
    studentize=True,
    nested=False,
    random_state=0,
    target="target",
    horizon="horizon",
    origin="origin",
    model="model_id",
) -> dict
```

Input: long or wide loss panel with a named benchmark model. Output:
JSON-ready dictionary with one record per target/horizon group. The record
contains `p_values` for `lower`, `consistent`, and `upper` SPA p-value
variants, `critical_values`, selected `p_value`, `superior_models`, and
backend metadata.

Backend alignment: delegates to `arch.bootstrap.SPA`. The backend takes
benchmark losses and candidate losses, forms loss differentials internally as
`benchmark_loss - candidate_loss`, and reports `lower`, `consistent`, and
`upper` p-values from Hansen's recentering choices. Positive
`mean_loss_difference` in the output means the candidate has lower average loss
than the benchmark.

R/source comparison: archived R `ttrTests/R/dataSnoop.R::dataSnoop(test="SPA")`
implements Hansen SPA for technical-trading rule parameter grids. It recomputes
strategy performance on each bootstrapped price sample, so it is not a direct
general loss-matrix API. `macroforecast` keeps the general forecast-evaluation
contract and records this as conceptual R alignment in each output record.

Options:

| Option | Default | Choices | Meaning |
| --- | --- | --- | --- |
| `bootstrap_method` | `"stationary_bootstrap"` | `"stationary_bootstrap"`, `"fixed_block_bootstrap"` | Bootstrap family. Fixed-block inputs are mapped to `arch`'s moving-block backend. |
| `p_value_type` | `"consistent"` | `"lower"`, `"consistent"`, `"upper"` | Which SPA p-value variant to use for `p_value` and `decision`. |
| `studentize` | `True` | boolean | Passed to `arch.bootstrap.SPA`. |
| `nested` | `False` | boolean | Passed to `arch.bootstrap.SPA` for nested model sets. |

### reality_check_test

```python
macroforecast.tests.reality_check_test(
    loss_panel,
    *,
    benchmark,
    loss="squared_error",
    alpha=0.05,
    n_boot=1000,
    block_length="auto",
    bootstrap_method="stationary_bootstrap",
    p_value_type="consistent",
    studentize=True,
    nested=False,
    random_state=0,
    target="target",
    horizon="horizon",
    origin="origin",
    model="model_id",
) -> dict
```

Input and output follow `superior_predictive_ability_test(...)`. Backend:
`arch.bootstrap.RealityCheck`. In the current `arch` backend this class is a
Reality Check alias over the same SPA machinery, with the same p-value fields.
Use this when the research design calls for the White Reality Check against a
benchmark model.

R/source comparison: archived R `ttrTests/R/dataSnoop.R::dataSnoop(test="RC")`
implements White's Reality Check for technical-trading rule grids. As with SPA,
the R function is strategy-generator specific; `macroforecast` uses
precomputed benchmark and candidate forecast-loss series.

### stepm_test

```python
macroforecast.tests.stepm_test(
    loss_panel,
    *,
    benchmark,
    loss="squared_error",
    alpha=0.05,
    n_boot=1000,
    block_length="auto",
    bootstrap_method="stationary_bootstrap",
    studentize=True,
    nested=False,
    random_state=0,
    target="target",
    horizon="horizon",
    origin="origin",
    model="model_id",
) -> dict
```

Input: long or wide loss panel with a named benchmark model. Output:
JSON-ready dictionary with `superior_models` for each target/horizon group.
Backend: `arch.bootstrap.StepM`.

R/source comparison: `oosanalysis-R-library/R/stepm.R::stepm` implements a
generic Romano-Wolf stepdown loop from supplied test statistics and bootstrap
test-statistic draws. `macroforecast` delegates to `arch.bootstrap.StepM`, which
constructs the benchmark-vs-candidate loss-difference statistics using the SPA
backend and then applies the stepdown procedure. The objective is aligned, but
the inputs are higher level in `macroforecast`: forecast-loss panel in,
superior model names out.

### model_confidence_set

```python
macroforecast.tests.model_confidence_set(
    loss_panel,
    *,
    loss="squared_error",
    alpha=0.10,
    n_boot=1000,
    block_length="auto",
    bootstrap_method="mcs_fixed_block",
    statistic="max",
    random_state=0,
    target="target",
    horizon="horizon",
    origin="origin",
    model="model_id",
) -> dict
```

Exact Hansen-Lunde-Nason model confidence set callable aligned with the R
`MCS` package's `MCSprocedure`. It constructs pairwise loss-difference
statistics, bootstraps those loss-difference means, removes one model per step,
tracks cumulative MCS p-values, and records included and rejected model sets by
target/horizon group.

Inputs:

| Form | Required columns |
| --- | --- |
| Long panel | `origin`, `model_id`, and the selected loss column. `target` and `horizon` are optional grouping columns. |
| Wide matrix | Numeric model-loss columns. The target/horizon labels are set to `"all"`. |

Long-panel input must have one row per target/horizon/origin/model key. Duplicate
loss rows are rejected instead of being averaged inside the pivot step.

Options:

| Option | Default | Choices | Meaning |
| --- | --- | --- | --- |
| `statistic` | `"max"` | `"max"`, `"range"` | `"max"` maps to R `statistic="Tmax"` over `d_i.`; `"range"` maps to R `statistic="TR"` over pairwise `d_ij`. |
| `bootstrap_method` | `"mcs_fixed_block"` | `"mcs_fixed_block"`, `"stationary_bootstrap"`, `"fixed_block_bootstrap"` | `mcs_fixed_block` follows R `MCS/R/internalFunctions.R::GetIndices`; the other choices are package extensions. |
| `block_length` | `"auto"` | positive int or `"auto"` | Block length. `"auto"` follows the R rule conceptually: selected AR order across loss columns, with a minimum of 3. |

Output: JSON-ready dictionary with
`metadata_schema.kind="model_confidence_set"`.

| Key | Meaning |
| --- | --- |
| `mcs_inclusion` | Included model records by target, horizon, and alpha after the iterative procedure stops. |
| `mcs_rejections` | Eliminated model records by target, horizon, and alpha. |
| `p_values` | Final stopping-test p-value by target and horizon. |
| `iteration_path` | One record per removal step, including active models, statistic, p-value, cumulative MCS p-value, removed model, rejected model if any, and mean losses. |
| `block_lengths_used` | Block length used by target and horizon. |

R/source alignment:

| R source | Python contract |
| --- | --- |
| `MCS/R/MCSprocedure.R::MCSprocedure` | Sequential elimination until one model remains; included/excluded sets are determined by `p-Value for H_{0,M_k}` relative to `alpha`. |
| `MCS/R/internalFunctions.R::GetD` | Pairwise loss differences `d_ij` and model-average differences `d_i.`. |
| `MCS/R/internalFunctions.R::GetIndices` | Default `bootstrap_method="mcs_fixed_block"` samples consecutive fixed blocks and truncates to sample length. |

`block_length="auto"` follows the same rule conceptually as R `k=NULL`: choose
the maximum selected AR order across loss columns and enforce a minimum of 3.
For bit-level reproducibility across software stacks, pass an explicit integer
`block_length`.

**Coverage under a global null with many ties (WP-A1 Step 0).** Monte Carlo
size/coverage validation (`tests/mc/test_mcs_coverage.py`) confirms the MCS
inclusion guarantee `P(true best retained) >= 1-alpha` holds when one model
has a genuine (even small) loss advantage over the rest. But when **all**
candidate models are exactly tied (an exact global null with `K` symmetric
models, no true best), `P(all K models jointly retained)` is measurably
**below** the naive `1-alpha` reading of the guarantee -- e.g. ~0.82, not
~0.90, at `alpha=0.10`, `K=5`. This was cross-checked directly against R's
own `MCS::MCSprocedure` on the identical design (subprocess-Rscript bridge,
R=200 replications, `B=500`): R reproduces the same ~0.82 rate
(`rate=0.8200`, 99% CI `[0.7401,0.8841]`), matching this package's own
longer-run measurement (`rate=0.8180`, CI `[0.7846,0.8483]`, n_reps=1000).
Since the reference R implementation shows the identical behavior, this is a
property of the Hansen-Lunde-Nason sequential-elimination procedure itself
under many exact ties, not a `macroforecast` defect -- treat `alpha` as
calibrated for "is there a worse model to eliminate" one step at a time,
not as a literal `1-alpha` bound on "are all K models jointly retained"
when many of them are exactly tied.

### iterative_model_confidence_set

```python
macroforecast.tests.iterative_model_confidence_set(
    loss_panel,
    *,
    loss="squared_error",
    alpha=0.10,
    n_boot=1000,
    block_length="auto",
    bootstrap_method="mcs_fixed_block",
    statistic="max",
    random_state=0,
    target="target",
    horizon="horizon",
    origin="origin",
    model="model_id",
)
```

Descriptive alias for `model_confidence_set(...)`. It calls the same exact MCS
engine and returns the same fields, with
`metadata_schema.kind="iterative_model_confidence_set"` so older code can trace
which callable produced the result.

## Residual Diagnostics

### residual_diagnostics

```python
macroforecast.tests.residual_diagnostics(
    residuals,
    *,
    tests=(
        "ljung_box_q",
        "arch_lm",
        "jarque_bera_normality",
        "durbin_watson",
    ),
    lag=10,
    alpha=0.05,
    model_df=0,
    exog=None,
    demean_arch=False,
)
```

Input: residual series. Output: one-row-per-test pandas `DataFrame` with
`test`, `statistic`, `p_value`, `decision`, `lag_used`, `df`, `n_obs`,
`source_reference`, `r_reference`, `r_alignment`, and `status`. The result carries
`attrs["macroforecast_metadata_schema"] = {"kind": "residual_diagnostics",
"version": 1, ...}`.

Supported tests:

| Name | Meaning |
| --- | --- |
| `ljung_box_q` | Ljung-Box serial-correlation diagnostic, aligned with `stats::Box.test(type="Ljung-Box")`; `model_df` maps to R `fitdf`. |
| `breusch_godfrey_serial_correlation` | Breusch-Godfrey Chisq LM diagnostic under the residual-series contract; default is equivalent to testing `residuals ~ 1`, and `exog` supplies additional original-regression design columns. |
| `arch_lm` | Engle ARCH LM diagnostic, aligned with `FinTS::ArchTest`; `demean_arch=True` matches its `demean=TRUE` option. |
| `jarque_bera_normality` | Jarque-Bera normality diagnostic using the same population-moment formula as `tseries::jarque.bera.test`. |
| `durbin_watson` | Durbin-Watson statistic aligned with the statistic in `lmtest::dwtest`; p-value is not supplied because `lmtest`'s exact p-value uses a model-design distribution not available from residuals alone. |

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `lag` | `10` | Maximum lag for Ljung-Box, ARCH-LM, and Breusch-Godfrey. |
| `alpha` | `0.05` | Rejection level used for `decision`. |
| `model_df` | `0` | Degrees of freedom consumed by the fitted model. Used in Ljung-Box p-values and ARCH-LM degrees-of-freedom adjustment. |
| `exog` | `None` | Optional design matrix for the Breusch-Godfrey auxiliary regression. If omitted, an intercept-only design is used. |
| `demean_arch` | `False` | Demean residuals before ARCH-LM, matching `FinTS::ArchTest(demean=TRUE)` when enabled. |

Source-alignment notes:

| Diagnostic | Source logic |
| --- | --- |
| Ljung-Box | `stats::Box.test(type="Ljung-Box")`: `Q = n(n+2) sum rho_k^2/(n-k)`, chi-squared df `lag - model_df`; `model_df` is R `fitdf`. |
| ARCH-LM | `FinTS/R/ArchTest.R::ArchTest`: optionally demean residuals, embed `x^2`, regress current squared residuals on lagged squared residuals, statistic is effective sample size times auxiliary `R^2`. `model_df` is a statsmodels degrees-of-freedom adjustment beyond the R API. |
| Jarque-Bera | `tseries/R/test.R::jarque.bera.test`: population central moments, `n * skewness^2 / 6 + n * (kurtosis - 3)^2 / 24`, chi-squared df `2`. |
| Breusch-Godfrey | `lmtest/R/bgtest.R::bgtest`: R takes a fitted model or formula. `macroforecast` takes residuals and optional `exog`, then applies the same Chisq LM auxiliary formula with fill-zero lagged residual columns under that residual-series contract. |
| Durbin-Watson | `lmtest/R/dwtest.R::dwtest`: statistic `sum(diff(residuals)^2) / sum(residuals^2)`. P-values are omitted because R's exact/asymptotic p-value depends on the original regression design matrix. |

- `jarque_bera_test` -- Jarque-Bera normality test (single series, chi2 df=2; tseries::jarque.bera.test convention).

- `granger_causality` -- Granger causality test in a VAR (vars::causality; F or Wald).
- `instantaneous_causality` -- instantaneous (contemporaneous) causality test in a VAR.

- `giacomini_white_test` -- Giacomini-White (2006) CONDITIONAL predictive ability Wald test, instrument [1, dL_{t-h}]. **Default changed (WP-A1):** `small_sample=True` uses an untapered ("acf"-style) HAC over lags 0..horizon-1 (matching `dm_test`'s own kernel convention, with a bandwidth-shrink-on-non-PSD fallback) referenced against `F(q, ESS-q)` with `ESS = n/(1+2*bandwidth_used)` whenever horizon > 1 -- the original Bartlett-tapered-HAC + chi2(q) construction was Monte Carlo-confirmed oversized (2-2.5x nominal) for horizon > 1, non-vanishing out to n=100,000 (see `tests/mc/test_giacomini_white_size.py`). `small_sample=False` restores the exact pre-WP-A1 Bartlett-HAC + chi2(q) p-values. horizon=1 is unaffected either way.

- `var_serial_test` -- multivariate residual serial-correlation (Portmanteau/LM) test for a VAR (vars::serial.test).

- `var_normality_test` -- multivariate normality (Doornik-Hansen/Lutkepohl JB) test for VAR residuals (vars::normality.test).

- `var_arch_test` -- multivariate ARCH-LM test for VAR residuals (vars::arch.test, Lutkepohl).
