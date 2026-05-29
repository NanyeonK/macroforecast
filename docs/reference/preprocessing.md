# macroforecast.preprocessing

[Back to reference](index.md)

`macroforecast.preprocessing` turns a canonical pandas panel from
[`macroforecast.data`](data.md) into a processed panel plus metadata. It accepts
a `DataSpec`, `DataBundle`, `(panel, metadata)` tuple, or `pandas.DataFrame`,
then returns a `PreprocessedData` object. The preferred input is a
`DataBundle` or `DataSpec` produced by `macroforecast.data`; if preprocessing
receives a plain panel without data-generated metadata, it emits a warning.

The default `reprocess()` path follows the public McCracken-Ng FRED-MD Matlab
workflow for FRED-MD/FRED-QD style panels. FRED-SD has no official t-code map,
so the user must explicitly choose `transform="none"` or pass custom codes.

## Public Flow

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()
data_spec = mf.data.spec(bundle, target="INDPRO", horizons=[1, 3, 6, 12])

processed = mf.preprocessing.reprocess(data_spec)

panel = processed.panel
metadata = processed.metadata
```

YAML wrappers are intentionally not the primary surface here. A future wrapper
can map a file into the same direct-call functions.

## Default Order

| Step | Default | Meaning |
| --- | --- | --- |
| 1. Frequency | `frequency="keep"` | Keep the input frequency unless the user asks for monthly/quarterly alignment. |
| 2. Transform | `transform="official"` | Apply official t-code transforms from FRED-MD/FRED-QD metadata. |
| 3. T-code lag | `tcode_lag="drop"` | Remove leading rows implied by the largest t-code lag. This is two rows for full FRED-MD. |
| 4. Outliers | `outliers="iqr"`, `outlier_action="flag_as_nan"`, `iqr_threshold=10.0` | Flag observations with `abs(x - median) > 10 * IQR` and set them to missing. |
| 5. Imputation | `impute="em_factor"` | Run FRED-MD style PCA-EM with Bai-Ng `PC_p2`, `kmax=8`, `DEMEAN=2`, `max_iter=50`, `tol=1e-6`. |
| 6. Frame | `frame="keep"` | Keep the post-EM frame. No final balanced-panel truncation is applied by default. |

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

`mf.preprocessing.preprocess(...)` is kept as a backward-compatible alias for
`reprocess(...)`.

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
    frame: str = "keep",
) -> PreprocessedData
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `data` | `DataSpec`, `DataBundle`, `(panel, metadata)`, or `DataFrame` | required | Canonical data input. |
| `metadata` | mapping or `None` | `None` | Extra metadata to merge before preprocessing. |
| `frequency` | `str` | `"keep"` | `"keep"`, `"monthly"`, `"quarterly"`, `"drop_non_monthly"`, `"drop_non_quarterly"`. |
| `quarterly_to_monthly` | `str` | `"step_backward"` | `"step_backward"`, `"step_forward"`, `"linear_interpolation"`. |
| `weekly_to_monthly` | `str` | `"mean"` | `"mean"`, `"last"`, `"sum"`. |
| `monthly_to_quarterly` | `str` | `"quarterly_average"` | `"quarterly_average"`, `"quarterly_endpoint"`, `"quarterly_sum"`. |
| `weekly_to_quarterly` | `str` | `"mean"` | `"mean"`, `"last"`, `"sum"`. |
| `transform_order` | `str` | `"after_frequency"` | `"after_frequency"`/`"frequency_then_transform"` or `"before_frequency"`/`"transform_then_frequency"`. |
| `transform` | `str` | `"official"` | `"official"`, `"custom"`, `"none"`; accepts aliases `apply_official_tcode`, `custom_tcode`, `no_transform`. |
| `transform_codes` | mapping or `None` | from metadata | Full t-code map. Required for `transform="custom"` unless overrides provide every intended code. |
| `transform_code_overrides` | mapping or `None` | `None` | Per-series override applied on top of official or custom codes. |
| `tcode_lag` | `str` | `"drop"` | `"drop"`, `"keep"`, `"drop_all_missing_rows"`, `"drop_any_missing_rows"`. |
| `outliers` | `str` | `"iqr"` | `"iqr"`, `"zscore"`, `"winsorize"`, `"none"`. |
| `outlier_action` | `str` | `"flag_as_nan"` | `"flag_as_nan"`, `"replace_with_median"`, `"replace_with_cap_value"` for IQR/z-score methods. |
| `impute` | `str` | `"em_factor"` | `"em_factor"`, `"em_multivariate"`, `"mean"`, `"forward_fill"`, `"linear"`, `"none"`. |
| `em_factor_selection` | `str` | `"baing_p2"` | `"baing_p1"`, `"baing_p2"`, `"baing_p3"`, `"fixed"`. |
| `em_demean` | `int` | `2` | `0`, `1`, `2`, `3`, matching `factors_em.m`. |
| `frame` | `str` | `"keep"` | `"keep"`, `"truncate"`, `"drop_unbalanced_series"`, `"zero_fill"`. |

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

`PreprocessedData` supports tuple unpacking:

```python
panel, metadata = processed
```

## Step Helpers

These helpers return `pandas.DataFrame` unless noted.

| Function | Input | Output | Meaning |
| --- | --- | --- | --- |
| `plan(data, ...)` | DataFrame/bundle/spec | `dict` | Dry-run summary of configured choices, transform codes, metadata warning, and detected native frequencies. |
| `report(processed)` | `PreprocessedData` | `dict` | Compact report from a completed preprocessing result. |
| `handle_mixed_frequency(panel, method=...)` | DataFrame | DataFrame | Keep, filter, or align mixed monthly/quarterly/weekly panels. |
| `apply_transform_codes(panel, codes)` | DataFrame, t-code map | DataFrame | Apply McCracken-Ng t-code formulas. |
| `fred_sd_transform_codes(data, ...)` | FRED-SD panel/bundle/spec | `dict[str, int]`, or `(dict, DataFrame)` with `return_table=True` | Build FRED-SD state-series t-codes from user choices and optional national-analog suggestions. |
| `expand_fred_sd_transform_codes(data, ...)` | FRED-SD panel/bundle/spec | same as `fred_sd_transform_codes` | Backward-compatible alias. Prefer `fred_sd_transform_codes`. |
| `handle_tcode_lag(panel, method=..., codes=...)` | DataFrame | DataFrame | Handle missing rows introduced by t-code transforms. |
| `handle_outliers(panel, method=...)` | DataFrame | DataFrame | Apply one outlier policy. |
| `impute_missing(panel, method=...)` | DataFrame | DataFrame | Fill missing values. |
| `handle_frame_edges(panel, method=...)` | DataFrame | DataFrame | Keep/drop/truncate/fill remaining unbalanced edges. |

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
| `transform` | Transform method, applied t-code map, and FRED-SD official-code error note when relevant. |
| `tcode_lag`, `outliers`, `impute`, `frame` | Normalized choice values. |

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

## FRED-SD

FRED-SD does not provide official t-codes. `reprocess(fred_sd_bundle)` with
the default `transform="official"` raises an error. The user must choose one
of these paths.

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
