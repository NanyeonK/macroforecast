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
    kernel="newey_west",
    alpha=0.05,
)
```

Input: two aligned loss series. Output: `TestResult` for the Diebold-Mariano
equal predictive accuracy test. `correction="hln"` applies the
Harvey-Leybourne-Newbold small-sample correction.

### gw_test

```python
macroforecast.tests.gw_test(loss_a, loss_b, *, horizon=1, correction="hln")
```

Input: two aligned loss series. Output: `TestResult` using the package's
Giacomini-White-compatible loss differential surface.

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
    kernel="newey_west",
    alpha=0.05,
) -> pandas.DataFrame
```

Runs multiple equal-predictive-ability tests and stacks one row per test.
Supported names are `dm`, `gw`, `dmp`, and `hn`. `hn` requires `error_a` and
`error_b` because Harvey-Newbold is an encompassing test on forecast errors.

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

Input: two forecast-error series. Output: one-sided `TestResult` for forecast
encompassing.

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

Alias: `cw_test`.

### enc_new_test

```python
macroforecast.tests.enc_new_test(loss_small, loss_large, *, horizon=1)
```

Input: nested-model loss series. Output: one-sided `TestResult`.

### enc_t_test

```python
macroforecast.tests.enc_t_test(loss_small, loss_large, *, horizon=1)
```

Input: nested-model loss series. Output: one-sided `TestResult`.

### nested_tests

```python
macroforecast.tests.nested_tests(
    loss_small,
    loss_large,
    *,
    forecast_small=None,
    forecast_large=None,
    tests=("clark_west", "enc_new", "enc_t"),
    horizon=1,
    kernel="newey_west",
    alpha=0.05,
) -> pandas.DataFrame
```

Runs multiple nested-model tests and stacks one row per test. Clark-West
requires `forecast_small` and `forecast_large`; `enc_new` and `enc_t` need only
the nested-model loss series.

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
are `pesaran_timmermann` and `henriksson_merton`.

Aliases:

| Alias | Equivalent call |
| --- | --- |
| `pesaran_timmermann_test(y_true, y_pred)` | `directional_accuracy_test(..., method="pesaran_timmermann")` |
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

Input: probability integral transform values. Output: JSON-ready dictionary
with `metadata_schema.kind="density_interval_tests"` plus Berkowitz, KS,
Kupiec POF, Christoffersen independence, and Engle-Manganelli DQ diagnostics
when the sample is large enough. The output also includes a PIT histogram and
a PIT autocorrelation test.

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

Runs Kupiec POF, Christoffersen independence, and combined conditional
coverage diagnostics for forecast intervals. `alpha` is the expected
non-coverage rate, so a 90% interval uses `alpha=0.10`.

## Conditional Predictive Ability

### conditional_predictive_ability_test

```python
macroforecast.tests.conditional_predictive_ability_test(
    loss_a,
    loss_b,
    *,
    method="giacomini_rossi",
    window_ratio=0.25,
    alpha=0.05,
)
```

Input: two aligned loss series. Output: JSON-ready dictionary with
`metadata_schema.kind="conditional_predictive_ability"`, a fluctuation
statistic, critical value, decision, time path, and window size.

Supported methods: `giacomini_rossi`, `rossi_sekhposyan`.

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

Block-bootstrap one-sided reality check against a named benchmark model. This
is the direct callable replacement for the legacy `blocked_oob_reality_check`
operation.

Inputs:

| Form | Required columns |
| --- | --- |
| Long panel | `origin`, `model_id`, `squared_error`; optional `target` and `horizon`. Column names are configurable. |
| Wide matrix | One column per model, including the `benchmark` column. The index is treated as origin order. |

Output: one row per candidate model and target/horizon group.

| Column | Meaning |
| --- | --- |
| `target`, `horizon` | Group labels. Wide input uses `"all"` for both. |
| `model` | Candidate model tested against the benchmark. |
| `benchmark` | Benchmark model name. |
| `mean_diff` | `benchmark_loss - candidate_loss`; positive means candidate has lower loss. |
| `statistic` | Mean loss differential scaled by bootstrap standard error. |
| `p_value` | One-sided block-bootstrap p-value for no improvement over benchmark. |
| `decision` | `True` when `p_value < alpha`. |
| `n_obs` | Number of aligned origins. |
| `block_length`, `n_boot`, `bootstrap_method` | Bootstrap settings used. |

The returned table carries
`attrs["macroforecast_metadata_schema"]["kind"] = "blocked_oob_reality_check"`.

### iterative_model_confidence_set

```python
macroforecast.tests.iterative_model_confidence_set(
    loss_panel,
    *,
    loss="squared_error",
    alpha=0.10,
    n_boot=1000,
    block_length="auto",
    bootstrap_method="stationary_bootstrap",
    statistic="range",
    random_state=0,
    target="target",
    horizon="horizon",
    origin="origin",
    model="model_id",
) -> dict
```

Sequential Hansen-Lunde-Nason-style model confidence set callable. This is the
full iterative counterpart to `model_confidence_set()`, which remains a
single-step approximation for backward-stable behavior.

Inputs:

| Form | Required columns |
| --- | --- |
| Long panel | `origin`, `model_id`, and the selected loss column. `target` and `horizon` are optional grouping columns. |
| Wide matrix | Numeric model-loss columns. The target/horizon labels are set to `"all"`. |

Options:

| Option | Default | Choices | Meaning |
| --- | --- | --- | --- |
| `statistic` | `"range"` | `"range"`, `"max"` | Test statistic used for sequential elimination. `"range"` uses the max absolute pairwise loss-difference statistic; `"max"` uses the max centered mean-loss statistic. |
| `bootstrap_method` | `"stationary_bootstrap"` | `"stationary_bootstrap"`, `"fixed_block_bootstrap"` | Serial-dependence-respecting bootstrap over forecast origins. |
| `block_length` | `"auto"` | positive int or `"auto"` | Block length. `"auto"` uses the package's default rule. |

Output: JSON-ready dictionary with
`metadata_schema.kind="iterative_model_confidence_set"`.

| Key | Meaning |
| --- | --- |
| `mcs_inclusion` | Included model records by target, horizon, and alpha after the iterative procedure stops. |
| `mcs_rejections` | Eliminated model records by target, horizon, and alpha. |
| `p_values` | Final stopping-test p-value by target and horizon. |
| `iteration_path` | One record per elimination/stopping step, including active models, statistic, p-value, worst model, eliminated model, and mean losses. |
| `block_lengths_used` | Block length used by target and horizon. |

### model_confidence_set

```python
macroforecast.tests.model_confidence_set(
    loss_panel,
    *,
    loss="squared_error",
    alpha=0.10,
    n_boot=1000,
    block_length="auto",
    bootstrap_method="stationary_bootstrap",
    spa_benchmark_model=None,
    random_state=0,
    target="target",
    horizon="horizon",
    origin="origin",
    model="model_id",
)
```

Input: long loss panel with target, horizon, origin, model, and loss columns.

Output: JSON-ready dictionary with `metadata_schema.kind="model_confidence_set"`.
The output stores record lists, not tuple-keyed dictionaries:

| Key | Meaning |
| --- | --- |
| `mcs_inclusion` | Records with `target`, `horizon`, `alpha`, and included `models`. |
| `spa_p_values` | SPA p-value records by target and horizon. |
| `reality_check_p_values` | Reality-check p-value records by target and horizon. |
| `stepm_rejected` | StepM rejected-model records. |
| `block_lengths_used` | Block length used by target and horizon. |

`model_confidence_set()` is intentionally a single-step approximation with
SPA/StepM fields. Use `iterative_model_confidence_set()` when the research
design requires sequential MCS elimination.

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
)
```

Input: residual series. Output: one-row-per-test pandas `DataFrame` with
`test`, `statistic`, `p_value`, `lag_used`, and `n_obs`. The result carries
`attrs["macroforecast_metadata_schema"] = {"kind": "residual_diagnostics",
"version": 1, ...}`.

Supported tests:

| Name | Meaning |
| --- | --- |
| `ljung_box_q` | Serial correlation diagnostic. |
| `breusch_godfrey_serial_correlation` | Serial correlation diagnostic using a trend regression fallback. |
| `arch_lm` | Conditional heteroskedasticity diagnostic. |
| `jarque_bera_normality` | Normality diagnostic. |
| `durbin_watson` | Durbin-Watson statistic; p-value is not supplied. |
