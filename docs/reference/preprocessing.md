# macroforecast.preprocessing

[Back to reference](index.md)

## Purpose

`macroforecast.preprocessing` turns a canonical pandas panel from
[`macroforecast.data`](data.md) into a processed panel plus metadata. It accepts
a `DataSpec`, `DataBundle`, `(panel, metadata)` tuple, or `pandas.DataFrame`,
then returns a `PreprocessedData` object. The preferred input is a
`DataBundle` or `DataSpec` produced by `macroforecast.data`; if preprocessing
receives a plain panel without data-generated metadata, it emits a warning.

The default `reprocess()` path follows the public McCracken-Ng FRED-MD Matlab
workflow for FRED-MD/FRED-QD style panels. FRED-SD has no official t-code map,
so the user must explicitly choose `transform="none"` or pass custom codes.

Preprocessing fails closed on transformation metadata. If
`transform="official"` is selected but no t-code map is available from
`transform_codes`, `metadata["transform_codes"]`, or
`panel.attrs["macroforecast_transform_codes"]`, `reprocess()` raises
`ValueError`. If explicit transform-code keys do not match panel columns, it
also raises. This prevents accidental no-op preprocessing.

## Public Functions

| Function | Purpose | Output |
| --- | --- | --- |
| `reprocess` | Run the full-sample preprocessing sequence. | `PreprocessedData` |
| `preprocess_spec` | Store preprocessing choices for runner-fitted execution. | `PreprocessSpec` |
| `custom_preprocess` | Apply one user callable directly to data. | `PreprocessedData` |
| `custom_preprocess_step` | Build a custom step for `preprocess_spec(custom_steps=[...])`. | `dict` |
| `plan` | Validate and summarize configured preprocessing choices without changing data. | `dict` |
| `report` | Summarize a completed preprocessing result. | `dict` |
| `apply_transform_codes` | Apply McCracken-Ng t-code formulas to matching panel columns. | `pandas.DataFrame` |
| `fred_sd_transform_codes` | Expand FRED-SD variable/state t-code choices and suggestions. | `dict` or `(dict, DataFrame)` |
| `handle_tcode_lag` | Keep or remove transform-induced leading missing rows. | `pandas.DataFrame` |
| `handle_outliers` | Apply one outlier rule. | `pandas.DataFrame` |
| `impute_missing` | Fill missing panel values with one imputation rule. | `pandas.DataFrame` |
| `standardize_panel` | Fit and apply one full-panel scaling rule. | `pandas.DataFrame` |
| `handle_frame_edges` | Keep, truncate, drop, or fill remaining unbalanced edges. | `pandas.DataFrame` |

Low-level clean helpers are also public for exact single-operation use. They
are listed in [Low-Level Clean Helpers](#low-level-clean-helpers).

## Public Flow

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()
data_spec = mf.data.spec(bundle, target="INDPRO", horizons=[1, 3, 6, 12])

processed = mf.preprocessing.reprocess(data_spec)

panel = processed.panel
metadata = processed.metadata
```

## Public Classes And Values

| Symbol | Meaning |
| --- | --- |
| `PreprocessedData` | Output object returned by `reprocess(...)` and `custom_preprocess(...)`. |
| `PreprocessInput` | Accepted direct preprocessing input type: `DataSpec`, `DataBundle`, `(panel, metadata)`, or `DataFrame`. |
| `PreprocessSpec` | Runner-compatible fit/transform preprocessing contract. |
| `FittedPreprocessor` | Fitted preprocessing state used by the runner for fit-window or fixed-reference policies. |
| `FRED_SD_NATIONAL_ANALOG_TRANSFORM_CODES` | High-confidence package t-code suggestions for FRED-SD variables with national analogs. |
| `FRED_SD_MEDIUM_CONFIDENCE_TRANSFORM_CODES` | Broader provisional FRED-SD t-code suggestions. |

## PreprocessedData

```python
macroforecast.preprocessing.PreprocessedData(
    panel: pandas.DataFrame,
    metadata: dict,
    target: str | None = None,
    targets: tuple[str, ...] = (),
    horizons: tuple[int, ...] = (),
    start: str | None = None,
    end: str | None = None,
    predictors = "all",
    steps: tuple[dict, ...] = (),
)
```

### Output Schema

| Field | Type | Meaning |
| --- | --- | --- |
| `panel` | `pandas.DataFrame` | Processed canonical date-indexed panel. |
| `metadata` | `dict` | Input metadata plus preprocessing stages and transform/standardization state. |
| `target`, `targets`, `horizons`, `start`, `end`, `predictors` | copied from `DataSpec` when supplied | Run-level data choices preserved for downstream stages. |
| `steps` | `tuple[dict, ...]` | Ordered preprocessing step log. |

### Methods

| Method | Input | Output | Meaning |
| --- | --- | --- | --- |
| `attach(stage, values)` | `stage: str`, `values: Mapping` | `PreprocessedData` | Return a new object with one metadata stage added. |

`PreprocessedData` also supports tuple unpacking:

```python
panel, metadata = processed
```

## Default Order

| Step | Default | Meaning |
| --- | --- | --- |
| 1. Frequency | `frequency="keep"` | Keep the input frequency unless the user asks for monthly/quarterly alignment. |
| 2. Transform | `transform="official"` | Apply official t-code transforms from FRED-MD/FRED-QD metadata. |
| 3. T-code lag | `tcode_lag="drop"` | Remove leading rows implied by the largest t-code lag. This is two rows for full FRED-MD. |
| 4. Outliers | `outliers="iqr"`, `outlier_action="flag_as_nan"`, `iqr_threshold=10.0` | Flag observations with `abs(x - median) > 10 * IQR` and set them to missing. |
| 5. Imputation | `impute="em_factor"` | Run FRED-MD style PCA-EM with Bai-Ng `PC_p2`, `kmax=8`, `DEMEAN=2`, `max_iter=50`, `tol=1e-6`. |
| 6. Standardize | `standardize="none"` | Optional column-wise scaling after imputation. Choices are `"zscore"`, `"robust"`, and `"minmax"`. |
| 7. Frame | `frame="keep"` | Keep the post-EM frame. No final balanced-panel truncation is applied by default. |

Set `transform_order="before_frequency"` when a mixed-frequency panel should be
transformed in each native frequency before monthly or quarterly alignment. The
default is `transform_order="after_frequency"`, which first aligns frequency and
then applies t-codes.

## T-Code Formulas

The official FRED-MD/FRED-QD t-code map uses these formulas for a raw series
`x_t`.

| T-code | Formula | Leading missing values | Log-domain rule |
| --- | --- | --- | --- |
| `1` | `x_t` | `0` | none |
| `2` | `x_t - x_{t-1}` | `1` | none |
| `3` | `(x_t - x_{t-1}) - (x_{t-1} - x_{t-2})` | `2` | none |
| `4` | `log(x_t)` | `0` | if `min(x) < 1e-6`, the transformed series is all missing |
| `5` | `log(x_t) - log(x_{t-1})` | `1` | requires `min(x) > 1e-6`; otherwise all missing |
| `6` | `(log(x_t) - log(x_{t-1})) - (log(x_{t-1}) - log(x_{t-2}))` | `2` | requires `min(x) > 1e-6`; otherwise all missing |
| `7` | `(x_t / x_{t-1} - 1) - (x_{t-1} / x_{t-2} - 1)` | `2` | none |

There is no `preprocess(...)` compatibility alias in the clean public API. Use
`reprocess(...)` for full-sample preprocessing and `preprocess_spec(...)` for a
runner-fitted preprocessing contract.

Most empirical macro papers preprocess the full panel once before fitting
models. That is supported by `reprocess(...)`. For a real-time forecast design,
where each origin should only use information available at that origin, use
`preprocess_spec(...)` inside `macroforecast.forecasting.run(...)`.
`preprocess_spec(...)` only stores what preprocessing should do; the runner
receives `preprocessing_policy=mf.window.stage_policy(...)` and decides where
the spec may fit.

Common runner policies:

| Policy scope | Meaning |
| --- | --- |
| `"full_panel"` | Fit preprocessing once on the full panel. This is useful for retrospective replication designs. |
| `"origin_available"` | Re-run preprocessing on observations available at each origin plus requested test rows. This supports EM imputation on variables observed by that origin. |
| `"fit_window"` | Fit outlier, imputation, and standardization state on the model fit window, then apply that state to validation/test rows. It currently supports `impute="none"`, `"mean"`, and `"forward_fill"`; use `"origin_available"` for EM or linear imputation. |
| `"fixed_reference"` | Fit supported preprocessing state on a fixed reference period, then apply that state to later windows. |

```python
pre = macroforecast.preprocessing.preprocess_spec(
    transform="official",
    outliers="iqr",
    impute="em_factor",
    frame="keep",
)

result = macroforecast.forecasting.run(
    panel,
    "ridge",
    preprocessing=pre,
    preprocessing_policy=macroforecast.window.stage_policy("origin_available"),
    features=features,
    window=window,
)
```

## reprocess

```python
macroforecast.preprocessing.reprocess(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    frequency: str = "keep",
    quarterly_to_monthly: str = "step_backward",
    weekly_to_monthly: str = "mean",
    monthly_to_quarterly: str = "quarterly_average",
    weekly_to_quarterly: str = "mean",
    transform_order: str = "after_frequency",
    transform: str = "official",
    transform_codes: Mapping[str, int] | None = None,
    transform_code_overrides: Mapping[str, int] | None = None,
    tcode_lag: str = "drop",
    outliers: str = "iqr",
    outlier_action: str = "flag_as_nan",
    iqr_threshold: float = 10.0,
    zscore_threshold: float = 3.0,
    winsorize_quantiles: tuple[float, float] = (0.01, 0.99),
    impute: str = "em_factor",
    em_n_factors: int = 8,
    em_factor_selection: str = "baing_p2",
    em_demean: int = 2,
    em_max_iter: int = 50,
    em_tolerance: float = 1e-6,
    standardize: str = "none",
    standardize_columns: str | Sequence[str] = "all",
    standardize_ddof: int = 0,
    frame: str = "keep",
    warn_metadata: bool = True,
) -> PreprocessedData
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `data` | `DataSpec`, `DataBundle`, `(panel, metadata)`, or `DataFrame` | required | Canonical data input. |
| `metadata` | mapping or `None` | `None` | Extra metadata to merge before preprocessing. |
| `frequency` | `str` | `"keep"` | `"keep"`, `"monthly"`, `"quarterly"`, `"drop_non_monthly"`, `"drop_non_quarterly"`. |
| `quarterly_to_monthly` | `str` | `"step_backward"` | `"step_backward"`, `"repeat_within_quarter"`, `"step_forward"`, `"quarter_end_ffill"`, `"linear_interpolation"`. |
| `weekly_to_monthly` | `str` | `"mean"` | `"mean"`, `"last"`, `"sum"`. |
| `monthly_to_quarterly` | `str` | `"quarterly_average"` | `"quarterly_average"`, `"quarterly_endpoint"`, `"quarterly_sum"`. |
| `weekly_to_quarterly` | `str` | `"mean"` | `"mean"`, `"last"`, `"sum"`. |
| `transform_order` | `str` | `"after_frequency"` | `"after_frequency"`/`"frequency_then_transform"` or `"before_frequency"`/`"transform_then_frequency"`. |
| `transform` | `str` | `"official"` | `"official"`, `"custom"`, `"none"`; accepts aliases `apply_official_tcode`, `custom_tcode`, `no_transform`. |
| `transform_codes` | mapping or `None` | from metadata | Full t-code map. Required for `transform="custom"` and required for `transform="official"` when metadata does not provide codes. Explicit keys must match panel columns. |
| `transform_code_overrides` | mapping or `None` | `None` | Per-series override applied on top of official or custom codes. Override keys must match panel columns. |
| `tcode_lag` | `str` | `"drop"` | `"drop"`, `"keep"`, `"drop_all_missing_rows"`, `"drop_any_missing_rows"`. |
| `outliers` | `str` | `"iqr"` | `"iqr"`, `"zscore"`, `"winsorize"`, `"none"`. |
| `outlier_action` | `str` | `"flag_as_nan"` | `"flag_as_nan"`, `"replace_with_median"`, `"replace_with_cap_value"` for IQR/z-score methods. |
| `impute` | `str` | `"em_factor"` | `"em_factor"`, `"em_multivariate"`, `"mean"`, `"forward_fill"`, `"linear"`, `"none"`. |
| `em_factor_selection` | `str` | `"baing_p2"` | `"baing_p1"`, `"baing_p2"`, `"baing_p3"`, `"fixed"`. |
| `em_demean` | `int` | `2` | `0`, `1`, `2`, `3`, matching `factors_em.m`. |
| `standardize` | `str` | `"none"` | `"none"`, `"zscore"`, `"robust"`, `"minmax"`. Aliases include `"standard"` and `"standardize"` for z-score. |
| `standardize_columns` | `str` or sequence | `"all"` | `"all"`, `"predictors"`, `"targets"`, or explicit column names. `"predictors"` and `"targets"` use `DataSpec` choices when available. |
| `standardize_ddof` | `int` | `0` | Degrees of freedom used by z-score scaling. |
| `frame` | `str` | `"keep"` | `"keep"`, `"truncate"`, `"drop_unbalanced_series"`, `"zero_fill"`. |
| `warn_metadata` | `bool` | `True` | Warn when plain panels lack metadata from `macroforecast.data`. `preprocess_spec(...)` defaults this to `False` unless explicitly overridden. |

### Output

Returns `PreprocessedData`.

| Field | Type | Meaning |
| --- | --- | --- |
| `panel` | `pandas.DataFrame` | Processed canonical date-indexed panel. |
| `metadata` | `dict` | Original data metadata plus a `preprocessing` stage. |
| `target`, `targets`, `horizons`, `start`, `end`, `predictors` | copied from `DataSpec` when supplied | Run-level data choices preserved for downstream stages. |
| `steps` | `tuple[dict, ...]` | Ordered preprocessing log. |

`metadata["preprocessing"]["transform_state"]` stores inverse-transform support
metadata for every transformed series: t-code, log-domain requirement, lag
count, and the last observed raw values/dates available before transformation.
`metadata["preprocessing"]["standardization_state"]` stores the fitted center
and scale values when `standardize != "none"`.

When transforms are applied, the final post-override t-code map is also stored
in `metadata["transform_codes_applied"]` and
`processed.panel.attrs["macroforecast_transform_codes"]`. This is the map that
actually ran, not just the raw loader metadata.

### Error Conditions

| Condition | Result |
| --- | --- |
| Plain `DataFrame` without data metadata | `UserWarning`; preprocessing still runs if the panel is canonical. |
| `transform="official"` with no t-code map | `ValueError`. |
| `transform="custom"` with no t-code map | `ValueError`. |
| Explicit transform-code or override key not in the panel | `ValueError`. |
| FRED-SD with default `transform="official"` | `ValueError`; choose `transform="none"` or custom FRED-SD codes. |
| Frequency inference finds sparse unknown columns during alignment | `UserWarning`; supply data metadata when the source frequency is known. |
| EM imputation sees an all-missing row or column | `ValueError`. |
| Standardization sees a zero-variance numeric column | `ValueError`. |

`PreprocessedData` supports tuple unpacking:

```python
panel, metadata = processed
```

## preprocess_spec

`preprocess_spec(...)` stores the same preprocessing options accepted by
`reprocess(...)`, excluding input-only arguments such as `data` and `metadata`.
It rejects unknown options immediately, so stage timing options must be passed
to `forecasting.run(..., preprocessing_policy=...)`, not hidden inside the
preprocessing spec.

```python
macroforecast.preprocessing.preprocess_spec(
    **options,
) -> PreprocessSpec
```

### Input

`**options` may include any `reprocess(...)` option except `data` and
`metadata`. It also accepts:

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `custom_steps` | sequence or omitted | omitted | Custom preprocessing steps created by `custom_preprocess_step(...)`. |
| `warn_metadata` | `bool` | `False` inside runner specs unless supplied | Whether to warn when input lacks `macroforecast.data` metadata. |

Do not pass window timing, stage scope, or split choices here. Those belong to
`forecasting.run(..., preprocessing_policy=...)`.

### Output

Returns `PreprocessSpec`.

| Method | Input | Output | Meaning |
| --- | --- | --- | --- |
| `fit(data, metadata=None, policy="origin_available")` | preprocessing input | `FittedPreprocessor` | Fit preprocessing choices on a training/history panel. |
| `fit_transform(data, metadata=None, policy="origin_available")` | preprocessing input | `PreprocessedData` | Fit and return the processed training panel. |
| `to_dict()` | none | `dict` | JSON-ready preprocessing options. |
| `to_metadata()` | none | `dict` | Compact runner metadata. |

`FittedPreprocessor.transform(data, metadata=None, history=None, policy=None)`
returns `PreprocessedData` for new rows. `policy="origin_available"` replays
preprocessing on `history + data`; `policy="fit_window"` applies state fitted
on the training window where supported.

```python
pre = mf.preprocessing.preprocess_spec(
    transform="official",
    outliers="iqr",
    impute="em_factor",
    standardize="zscore",
    frame="keep",
)
```

For direct advanced use:

```python
fitted = pre.fit(train_panel, policy="origin_available")
processed_test = fitted.transform(test_panel, history=train_panel)
```

The fitted and transformed metadata records `fit_period`, `history_period`,
`transform_period`, and `output_period`. `policy="fit_window"` applies
fit-window outlier, imputation, and standardization state; it currently supports
`impute="none"`, `"mean"`, and `"forward_fill"`.

`preprocess_spec(...)` also accepts `custom_steps=[...]`. These steps run after
the built-in preprocessing options. Inside `forecasting.run(...)`, the custom
steps are fitted or applied inside the same stage policy as the rest of the
preprocessing spec.

```python
def add_spread(panel, *, metadata=None, scale=1.0):
    out = panel.copy()
    out["spread"] = (out["long_rate"] - out["short_rate"]) * scale
    return out

pre = mf.preprocessing.preprocess_spec(
    transform="none",
    impute="mean",
    custom_steps=[
        mf.preprocessing.custom_preprocess_step("spread", add_spread, scale=100.0),
    ],
)
```

## custom_preprocess

Apply one user-supplied preprocessing callable directly to a panel or bundle.

```python
macroforecast.preprocessing.custom_preprocess(
    data,
    func,
    *,
    metadata: Mapping[str, object] | None = None,
    name: str | None = None,
    **params,
) -> PreprocessedData
```

### Callable Contract

The callable receives:

```python
func(panel: pandas.DataFrame, *, metadata: dict, **params)
```

It must return one of:

| Return type | Meaning |
| --- | --- |
| `pandas.DataFrame` | New canonical or normalizable panel. Existing `attrs["macroforecast_metadata"]` is merged with input metadata. |
| `DataBundle` | Panel plus metadata to continue with. |
| `PreprocessedData` | Full preprocessing object to continue with. |
| `(DataFrame, metadata)` | Explicit panel and metadata pair. |

### Output

Returns `PreprocessedData`. Metadata gains `metadata["custom_preprocess"]`,
including callable name, parameters, input panel summary, and output panel
summary. The output panel also carries
`panel.attrs["macroforecast_metadata"]`.

## custom_preprocess_step

Create a runner-compatible preprocessing step for
`preprocess_spec(custom_steps=[...])`.

```python
macroforecast.preprocessing.custom_preprocess_step(
    name: str,
    func,
    **params,
) -> dict
```

| Input | Meaning |
| --- | --- |
| `name` | Stable step name stored in metadata. |
| `func` | Callable following the `custom_preprocess()` callable contract. |
| `**params` | JSON-ready parameters passed to `func`. |

The returned dictionary keeps the callable for Python execution, but
`PreprocessSpec.to_dict()` records only the callable name so runner metadata is
JSON-ready.

## Step Helpers

These helpers return `pandas.DataFrame` unless noted.

| Function | Input | Output | Meaning |
| --- | --- | --- | --- |
| `plan(data, ...)` | DataFrame/bundle/spec | `dict` | Dry-run summary of configured choices, transform codes, metadata warning, and detected native frequencies. |
| `report(processed)` | `PreprocessedData` | `dict` | Compact report from a completed preprocessing result. |
| `custom_preprocess(data, func, ...)` | DataFrame/bundle/spec and callable | `PreprocessedData` | Apply one custom preprocessing function directly. |
| `custom_preprocess_step(name, func, **params)` | name and callable | `dict` | Build a custom step for `preprocess_spec(custom_steps=[...])`. |
| `apply_transform_codes(panel, codes)` | DataFrame, t-code map | DataFrame | Apply McCracken-Ng t-code formulas. |
| `fred_sd_transform_codes(data, ...)` | FRED-SD panel/bundle/spec | `dict[str, int]`, or `(dict, DataFrame)` with `return_table=True` | Build FRED-SD state-series t-codes from user choices and optional national-analog suggestions. |
| `handle_tcode_lag(panel, method=..., codes=...)` | DataFrame | DataFrame | Handle missing rows introduced by t-code transforms. |
| `handle_outliers(panel, method=...)` | DataFrame | DataFrame | Apply one outlier policy. |
| `impute_missing(panel, method=...)` | DataFrame | DataFrame | Fill missing values. |
| `standardize_panel(panel, method=...)` | DataFrame | DataFrame | Apply one full-panel standardization policy. |
| `handle_frame_edges(panel, method=...)` | DataFrame | DataFrame | Keep/drop/truncate/fill remaining unbalanced edges. |

Low-level callable variants are public for users who want one exact operation
without the full `reprocess(...)` sequence.

## Low-Level Clean Helpers

These helpers accept a `pandas.DataFrame` and return a new `pandas.DataFrame`
unless the output column says otherwise.

| Function | Key options | Output | Meaning |
| --- | --- | --- | --- |
| `iqr_outlier_clean(panel, threshold=10.0, action="flag_as_nan")` | `threshold`, `action` | DataFrame | IQR outlier rule used by `handle_outliers(method="iqr")`. |
| `zscore_outlier_clean(panel, threshold=3.0, action="flag_as_nan")` | `threshold`, `action` | DataFrame | Z-score outlier rule used by `handle_outliers(method="zscore")`. |
| `winsorize_clean(panel, lower_quantile=0.01, upper_quantile=0.99)` | quantile bounds | DataFrame | Winsorization rule used by `handle_outliers(method="winsorize")`. |
| `em_factor_impute_clean(panel, n_factors=8, max_iter=50, tol=1e-6, factor_selection="baing_p2", demean=2)` | EM factor controls | DataFrame | PCA-EM imputation used by `impute_missing(method="em_factor")`. |
| `em_multivariate_impute_clean(panel, max_iter=20, tol=1e-4)` | EM controls | DataFrame | Multivariate EM imputation used by `impute_missing(method="em_multivariate")`. |
| `mean_impute_clean(panel)` | none | DataFrame | Column-mean imputation. |
| `forward_fill_clean(panel)` | none | DataFrame | Forward-fill imputation. |
| `linear_interpolate_clean(panel)` | none | DataFrame | Time interpolation imputation. |
| `truncate_to_balanced_clean(panel)` | none | DataFrame | Keep the largest balanced sample. |
| `drop_unbalanced_series_clean(panel)` | none | DataFrame | Drop series that keep unbalanced sample edges. |
| `zero_fill_leading_clean(panel)` | none | DataFrame | Fill leading missing values with zero. |
| `fit_standardization_state(panel, method="zscore", ddof=0)` | scaling method | `dict` | Fit reusable scaling state. |
| `apply_standardization_state(panel, state)` | fitted state | DataFrame | Apply previously fitted scaling state. |
| `standardize_clean(panel, method="zscore", ddof=0)` | scaling method | DataFrame | One-shot panel standardization. |
| `apply_tcode_transform(panel, tcode_map)` | t-code map | DataFrame | Apply McCracken-Ng t-code formulas to matching panel columns. |
| `freq_align_quarterly_to_monthly_clean(panel, quarterly_columns, rule="step_backward")` | column list, rule | DataFrame | Low-level quarterly-to-monthly alignment helper. |
| `freq_align_monthly_to_quarterly_clean(panel, monthly_columns, rule="quarterly_average")` | column list, rule | DataFrame | Low-level monthly-to-quarterly alignment helper. |

## plan

```python
macroforecast.preprocessing.plan(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    frequency: str = "keep",
    transform_order: str = "after_frequency",
    transform: str = "official",
    transform_codes: Mapping[str, int] | None = None,
    transform_code_overrides: Mapping[str, int] | None = None,
    tcode_lag: str = "drop",
    outliers: str = "iqr",
    impute: str = "em_factor",
    standardize: str = "none",
    standardize_columns: str | Sequence[str] = "all",
    standardize_ddof: int = 0,
    frame: str = "keep",
) -> dict
```

### Input

Same data input contract as `reprocess()`. `plan()` validates the panel and
normalizes choices, but it does not transform, impute, or mutate the panel.

### Output

| Key | Meaning |
| --- | --- |
| `input_panel` | Shape, date range, columns, missing count, and inferred index frequency. |
| `metadata_warning` | Warning text that would matter for a panel without data-generated metadata, or `None`. |
| `steps` | Ordered step names implied by `transform_order`. |
| `frequency` | Requested frequency policy plus native-frequency map and metadata source. |
| `frequency["issues"]` | Native-frequency inference concerns such as sparse `unknown`, `irregular`, or `annual` columns. |
| `transform` | Transform method, applied t-code map, ignored metadata-only codes, and any no-code/no-match error note. |
| `tcode_lag`, `outliers`, `impute`, `standardize`, `frame` | Normalized choice values. |

## report

```python
macroforecast.preprocessing.report(processed: PreprocessedData) -> dict
```

### Input

`processed` must be the object returned by `reprocess()`.

### Output

| Key | Meaning |
| --- | --- |
| `input_panel` | Panel summary before preprocessing. |
| `output_panel` | Panel summary after preprocessing. |
| `steps` | Ordered execution log with input/output shapes where relevant. |
| `choices` | Final normalized preprocessing choices. |
| `transform_state` | Inverse-transform support metadata saved during the transform step. |
| `standardization_state` | Fitted scaling metadata saved during the standardization step. |

## apply_transform_codes

```python
macroforecast.preprocessing.apply_transform_codes(
    panel: pandas.DataFrame,
    codes: Mapping[str, int],
) -> pandas.DataFrame
```

### Input

| Name | Type | Required | Choices |
| --- | --- | --- | --- |
| `panel` | `pandas.DataFrame` | yes | Canonical date-indexed numeric panel. |
| `codes` | mapping from column name to integer | yes | T-codes `1` through `7`. Columns absent from the panel are ignored. |

### Output

Returns a new `pandas.DataFrame` with matching columns transformed by the
McCracken-Ng formulas above. Columns without a matching t-code are copied
unchanged. Leading missing values are not removed here; call
`handle_tcode_lag()` or use `reprocess(tcode_lag=...)`.

Note the distinction between this low-level helper and `reprocess()`.
`apply_transform_codes()` ignores absent code keys for convenience when used
interactively. `reprocess()` is stricter: explicit transform-code keys must
match panel columns so a production run cannot silently miss a requested
series.

## handle_tcode_lag

```python
macroforecast.preprocessing.handle_tcode_lag(
    panel: pandas.DataFrame,
    *,
    method: str = "drop",
    codes: Mapping[str, int] | None = None,
) -> pandas.DataFrame
```

### Input

| `method` | Meaning |
| --- | --- |
| `"drop"` | Drop the first `max(t-code lag)` rows. This is the FRED-MD default path after applying official t-codes. |
| `"keep"` | Keep all rows, including transform-induced leading missing values. |
| `"drop_all_missing_rows"` | Drop only rows where every column is missing. |
| `"drop_any_missing_rows"` | Drop every row with at least one missing value. This is strict and often removes too much data. |

### Output

Returns a new `pandas.DataFrame`. The function does not impute; it only handles
missing rows introduced by transformations.

## handle_outliers

```python
macroforecast.preprocessing.handle_outliers(
    panel: pandas.DataFrame,
    *,
    method: str = "iqr",
    action: str = "flag_as_nan",
    iqr_threshold: float = 10.0,
    zscore_threshold: float = 3.0,
    winsorize_quantiles: tuple[float, float] = (0.01, 0.99),
) -> pandas.DataFrame
```

### Input

| Name | Default | Choices |
| --- | --- | --- |
| `method` | `"iqr"` | `"iqr"`, `"zscore"`, `"winsorize"`, `"none"` |
| `action` | `"flag_as_nan"` | `"flag_as_nan"`, `"replace_with_median"`, `"replace_with_cap_value"` for IQR/z-score methods |
| `iqr_threshold` | `10.0` | Positive float. McCracken-Ng default is `10.0`. |
| `zscore_threshold` | `3.0` | Positive float. |
| `winsorize_quantiles` | `(0.01, 0.99)` | Lower and upper quantiles for winsorization. |

### Output

Returns a new `pandas.DataFrame`. The default marks IQR outliers as `NaN`, so
the next imputation step can fill them.

## impute_missing

```python
macroforecast.preprocessing.impute_missing(
    panel: pandas.DataFrame,
    *,
    method: str = "em_factor",
    em_n_factors: int = 8,
    em_factor_selection: str = "baing_p2",
    em_demean: int = 2,
    em_max_iter: int = 50,
    em_tolerance: float = 1e-6,
) -> pandas.DataFrame
```

### Input

| Name | Default | Choices |
| --- | --- | --- |
| `method` | `"em_factor"` | `"em_factor"`, `"em_multivariate"`, `"mean"`, `"forward_fill"`, `"linear"`, `"none"` |
| `em_n_factors` | `8` | Maximum factor count for `em_factor`; fixed rank when `em_factor_selection="fixed"`. |
| `em_factor_selection` | `"baing_p2"` | `"baing_p1"`, `"baing_p2"`, `"baing_p3"`, `"fixed"` |
| `em_demean` | `2` | `0`, `1`, `2`, `3`, matching `factors_em.m` standardization modes. |
| `em_max_iter` | `50` | Positive integer. |
| `em_tolerance` | `1e-6` | Positive float. |

### Output

Returns a new `pandas.DataFrame`. The default `em_factor` path uses the
FRED-MD-style PCA-EM algorithm. It raises if the panel contains an all-missing
row or all-missing column; use `handle_tcode_lag()` before this step for the
usual FRED-MD transform-induced leading missing rows.

`method="linear"` fills only interior missing values bracketed by observed
data. It does not extrapolate leading or trailing missing values, because those
edges usually encode unavailable source observations.

`method="em_multivariate"` uses the same all-missing row/column guard as
`em_factor`.

## standardize_panel

```python
macroforecast.preprocessing.standardize_panel(
    panel: pandas.DataFrame,
    *,
    method: str = "zscore",
    ddof: int = 0,
) -> pandas.DataFrame
```

### Input

| Name | Default | Choices |
| --- | --- | --- |
| `method` | `"zscore"` | `"zscore"`, `"robust"`, `"minmax"` |
| `ddof` | `0` | Non-negative integer used only for z-score standardization. |

### Output

Returns a new `pandas.DataFrame` with numeric columns scaled. `zscore` uses
column means and standard deviations, `robust` uses median and IQR, and
`minmax` uses minimum and range. The helper fits scaling parameters on the full
panel supplied to it.

For forecasting experiments that require origin-by-origin information sets,
prefer `preprocess_spec(standardize=...)` through the forecasting runner. In
that path, scaling parameters are fitted on the train window and reused for the
test rows.

Inside `reprocess(...)`, use `standardize_columns="predictors"` when a
`DataSpec` should scale predictor columns while leaving the target in its
post-transform units.

## handle_frame_edges

```python
macroforecast.preprocessing.handle_frame_edges(
    panel: pandas.DataFrame,
    *,
    method: str = "keep",
) -> pandas.DataFrame
```

### Input

| `method` | Meaning |
| --- | --- |
| `"keep"` | Keep the panel as-is. This is the default after EM imputation. |
| `"truncate"` | Truncate to the largest balanced sample. |
| `"drop_unbalanced_series"` | Drop columns that keep unbalanced edges. |
| `"zero_fill"` | Fill leading missing values with zero. |

### Output

Returns a new `pandas.DataFrame`.

## FRED-SD

FRED-SD does not provide official t-codes. `reprocess(fred_sd_bundle)` with
the default `transform="official"` raises an error. The user must choose one
of these paths.

Package suggestion tables are exposed as constants for inspection:

| Symbol | Meaning |
| --- | --- |
| `FRED_SD_NATIONAL_ANALOG_TRANSFORM_CODES` | High-confidence t-code suggestions based on national FRED-MD/FRED-QD analogs. |
| `FRED_SD_MEDIUM_CONFIDENCE_TRANSFORM_CODES` | Broader provisional t-code suggestions; opt in with `include_medium_confidence=True`. |

## fred_sd_transform_codes

```python
macroforecast.preprocessing.fred_sd_transform_codes(
    data,
    *,
    variable_codes: Mapping[str, int] | None = None,
    state_series_codes: Mapping[str, int] | None = None,
    use_national_analog_suggestions: bool = True,
    include_medium_confidence: bool = False,
    return_table: bool = False,
) -> dict[str, int] | tuple[dict[str, int], pandas.DataFrame]
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `data` | `DataBundle`, `DataSpec`, `(panel, metadata)`, or `DataFrame` | required | FRED-SD wide state-series panel. |
| `variable_codes` | mapping or `None` | `None` | User t-code choices by FRED-SD variable, such as `{"UR": 2}`. Expanded to every matching state series. |
| `state_series_codes` | mapping or `None` | `None` | User t-code choices by exact column, such as `{"UR_CA": 2}`. Overrides variable-level choices. |
| `use_national_analog_suggestions` | `bool` | `True` | Include high-confidence package suggestions based on national FRED-MD/FRED-QD analogs. |
| `include_medium_confidence` | `bool` | `False` | Include broader provisional suggestions. |
| `return_table` | `bool` | `False` | Return a provenance table with the expanded code map. |

### Output

By default, returns `dict[str, int]` mapping FRED-SD state-series columns to
t-codes. With `return_table=True`, returns `(codes, table)`. The table columns
are `column`, `sd_variable`, `state`, `tcode`, `source`, and
`suggestion_confidence`.

`suggestion_confidence` is not a statistical confidence interval. It records
whether the t-code came from a user state-series override, user variable-level
choice, high-confidence package suggestion, medium-confidence package
suggestion, or no assignment.

No transform:

```python
processed = mf.preprocessing.reprocess(fred_sd_bundle, transform="none")
```

Variable-level t-codes expanded to all state series:

```python
codes = mf.preprocessing.fred_sd_transform_codes(
    fred_sd_bundle,
    variable_codes={"UR": 2, "ICLAIMS": 5},
)

processed = mf.preprocessing.reprocess(
    fred_sd_bundle,
    frequency="monthly",
    transform="custom",
    transform_codes=codes,
)
```

Built-in national-analog suggestions are offered for high-confidence FRED-SD
variables such as `UR`, `PARTRATE`, `ICLAIMS`, `LF`, `NA`, and major employment
sector variables. These are suggestions, not official FRED-SD metadata. Pass
`include_medium_confidence=True` to also include broader output, housing, trade,
and income analogs.

To inspect provenance:

```python
codes, table = mf.preprocessing.fred_sd_transform_codes(
    fred_sd_bundle,
    variable_codes={"UR": 2},
    return_table=True,
)
```

`table` has columns `column`, `sd_variable`, `state`, `tcode`, `source`, and
`suggestion_confidence`. Sources distinguish user state-series overrides, user
variable-level choices, high- or medium-confidence national-analog suggestions,
and unassigned columns. `suggestion_confidence` is not a statistical confidence
interval; it is a provenance label for non-official package suggestions.

For FRED-SD frequency alignment, preprocessing reads the data-generated
`fred_sd_series_metadata` report first. Observed-date inference is only a
fallback. FRED-SD is mixed monthly/quarterly data; combined dataset frequency
alignment belongs in `macroforecast.data`, not in preprocessing.

## FRED-QD and Dataset Combination

`mf.data.load_fred_qd()` returns a quarterly panel with
`metadata["frequency"] == "quarterly"` and official FRED-QD t-codes. FRED-QD is
not mixed-frequency in the same sense as FRED-SD.

Combinations such as FRED-MD + FRED-SD or FRED-QD + FRED-SD should be built in
`macroforecast.data`, not in preprocessing. Dataset composition decides which
sources to load, how to align indices before a run, how to merge metadata, and
how to record frequency-conversion provenance. Preprocessing then operates on
the combined canonical panel it receives.

Use:

```python
monthly_bundle = mf.data.load_fred_md_sd(states=["CA"], variables=["UR"])
quarterly_bundle = mf.data.load_fred_qd_sd(states=["CA"], variables=["UR"])
```

## Source

The FRED-MD/FRED-QD defaults are based on the public FRED-Databases Matlab code
linked from the St. Louis Fed FRED-MD/FRED-QD page, specifically
`fredfactors.m`, `prepare_missing.m`, `remove_outliers.m`, and `factors_em.m`.
