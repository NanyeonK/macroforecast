# macroforecast.feature_engineering

[Back to reference](index.md)

`macroforecast.feature_engineering` is the direct pandas surface for building
forecast targets and model-ready feature matrices. It accepts the same direct Python inputs used by
previous stages: `PreprocessedData`, `DataSpec`, `DataBundle`,
`(panel, metadata)`, or a canonical `pandas.DataFrame`.

For strict windowed forecasting, use `feature_spec(...)`. The spec is fitted by
`macroforecast.forecasting.run(...)` inside each train window and then
transformed for the matching test rows. Individual functions such as `lag()`,
`rolling_mean()`, `pca_features()`, and `feature_matrix()` remain callable
one-shot helpers; runner composition belongs in `macroforecast.forecasting`.

The preferred flow is:

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()
data_spec = mf.data.spec(bundle, target="INDPRO", horizons=[1, 3, 6], predictors="all")
processed = mf.preprocessing.reprocess(data_spec)

features = mf.feature_engineering.build_features(
    processed,
    lags=(0, 1, 2, 3),
    rolling_windows=(3, 6),
    add_time=True,
)

X = features.X
y = features.y
metadata = features.metadata
```

`build_features()` emits a warning when it receives a canonical panel that does
not carry `metadata["preprocessing"]`. This is allowed, but the default package
workflow is data -> preprocessing -> feature engineering.

Common callable examples:

```python
# Direct horizon targets: y[t+h].
y_direct = mf.feature_engineering.direct_target(processed, target="INDPRO", horizons=[1, 3, 6])

# Direct average target: one y column per requested horizon.
y_avg = mf.feature_engineering.average_target(processed, target="INDPRO", horizon=12, transform="growth")

# Path target: one y column per future step. Model fits/forecasts each step;
# evaluation averages the step forecasts.
y_path = mf.feature_engineering.path_targets(processed, target="INDPRO", horizon=12, transform="growth")

# Simple lagged predictors.
X_lag = mf.feature_engineering.lag(processed, columns=["PAYEMS", "INDPRO"], lags=range(0, 13))
```

## Code Structure

The public namespace stays `macroforecast.feature_engineering`, while the
implementation is split by responsibility:

| File | Responsibility |
| --- | --- |
| `targets.py` | `direct_target()`, `average_target()`, and `path_targets()`. |
| `transforms.py` | Direct pandas feature transforms: lags, rolling means, scaling, PCA, PLS, DFM-style factors, Chen-Rohe sparse component analysis, varimax rotation, grouped PCA, MAF, Hamilton filtering, AlbaMA, wavelet-style decomposition, and time features. |
| `selection.py` | Shared fitted feature-selection algorithms used by direct selection callables and runner-safe `feature_spec()` method names. |
| `compose.py` | Reusable step builders and sequential feature composition. |
| `matrix.py` | Paper-style `X`, `F`, `MARX`, `MAF`, and `LEVEL` feature-matrix combinations. |
| `builder.py` | End-to-end `build_features()` alignment of `X`, `y`, and metadata. |
| `shared.py` | Internal normalization, metadata, fitting, and validation helpers. |
| `core.py` | Compatibility re-export only. |

## Feature Boundary

This stage is direct and pandas-first. It constructs target columns and
ML-oriented feature transforms. Multiple transformations can be composed in
sequence through plain Python callables, and higher-level orchestration can call
the same functions later.

| Function | Owns | Does not own yet |
| --- | --- | --- |
| `direct_target()` | Direct-forecast target columns, including direct average targets. | Train/test split, recursive forecasting, inverse transforms. |
| `average_target()` | Explicit wrapper for direct average change/growth targets. | Model fitting. |
| `path_targets()` | Step-level targets for path-average forecasting. | Model-stage step fit/forecast and evaluation-stage forecast averaging. |
| `lag()` | Current and lagged predictor columns. | Model-specific lag search. |
| `mixed_frequency_lags()` | Exact-date lag blocks for native mixed-frequency panels. | Frequency conversion or model estimation. |
| `rolling_mean()` | Rolling-window means. | Fit-based filters or learned smoothers. |
| `moving_average_ladder()` | Multi-scale trailing moving-average block used before optional factor/PCA steps. | PCA/factor extraction itself. |
| `maf_features()` | Moving Average Factors from variable-specific lag panels. | Model fitting or choosing final feature combinations. |
| `hamilton_filter_features()` | Hamilton-filter trend/cycle columns with explicit expanding or full-sample policy. | Model fitting, test windows, or choosing filter horizons. |
| `feature_matrix()` | Named `X`, `F`, `MARX`, `MAF`, and `LEVEL` feature-matrix combinations. | Loading or preprocessing the raw/level panel. |
| `scale_features()` | Fit-policy-aware z-score, min-max, or robust scaling. | Model fitting. |
| `pca_features()` | Fit-policy-aware PCA factors. | Forecast model fitting. |
| `sparse_pca_chen_rohe_features()` | Chen-Rohe sparse component analysis factors using an L1 loading budget. | Model fitting; runner-safe fitting should use `sparse_pca_chen_rohe_step()` inside `feature_spec()`. |
| `varimax_features()` | Orthogonal varimax rotation of already-created factor-score columns. | Factor extraction itself; usually call after `pca_features()` or a factor step. |
| `sliced_inverse_regression_features()` | Target-aware Sliced Inverse Regression factors. | Model fitting; runner-safe fitting should use `sliced_inverse_regression_step()` inside `feature_spec()`. |
| `partial_least_squares_features()` | Target-aware PLS factor scores. | Model fitting; runner-safe fitting should use `partial_least_squares_step()` inside `feature_spec()`. |
| `dfm_features()` | Static DFM approximation by standardized PCA. | State-space DFM estimation; use model callables for that. |
| `variance_selection()`, `correlation_selection()`, `lasso_selection()`, `lasso_path_selection()`, `rfe_selection()`, `boruta_selection()`, `stability_selection()`, `genetic_selection()` | Direct column selection by one explicit algorithm. | Model fitting; runner-safe fitting uses the same method names inside `feature_spec(..., steps=[...])`. |
| `asymmetric_trim_features()` | Per-period rank-space columns for asymmetric trimming weights. | Estimating the nonnegative rank weights. |
| `wavelet_features()` | Causal rolling multi-resolution approximation/detail columns. | True DWT family-specific filtering. |
| `adaptive_ma_rf_features()` | Random-forest adaptive moving-average smoothing over time. | Forecast model fitting. |
| `group_pca()` | PCA factors within named column groups. | FAVAR-specific slow/fast construction, model estimation, or structural identification. |
| `compose_features()` | Sequential combinations such as `pca -> lag`, `lag -> pca`, `maf`, or `moving_average_ladder -> pca -> lag`. | Model fitting or evaluation. |
| `time_features()` | Trend, month, quarter, and year columns. | Public-holiday or trading-day calendars; the package targets monthly and quarterly macro panels. |
| `build_features()` | Aligned `X`, `y`, feature metadata, and feature-engineering metadata. | Model evaluation. |

Fit-based transformations require a declared `fit_policy`. The default is
`fit_policy="expanding"`, which estimates transform parameters using only data
available through each date. `fit_policy="full_sample"` is available for
exploration or already-split training data. Public fitted transforms warn by
default when `full_sample` is used; pass `warn_full_sample=False` only when the
input panel is already training-only or the call is intentionally diagnostic.

## direct_target

```python
macroforecast.feature_engineering.direct_target(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    transform: str = "level",
) -> pandas.DataFrame
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `data` | `PreprocessedData`, `DataSpec`, `DataBundle`, `(panel, metadata)`, or `DataFrame` | required | Canonical macroforecast input. |
| `metadata` | mapping or `None` | `None` | Extra metadata to merge into the input metadata. |
| `target` | `str` or `None` | from `data` | One target column. |
| `targets` | iterable or `None` | from `data` | Multiple target columns. Mutually exclusive with `target`. |
| `horizon` | positive `int` or `None` | from `data`, then `1` | One forecast horizon. |
| `horizons` | positive int/iterable or `None` | from `data`, then `(1,)` | Multiple forecast horizons. Mutually exclusive with `horizon`. |
| `transform` | `str` | `"level"` | `"level"`, `"change"`, `"growth"`, `"log_growth"`, `"average_change"`, `"average_growth"`, `"average_log_growth"`; common aliases include `"future_level"`, `"diff"`, `"pct_change"`, `"simple_growth"`, `"log_change"`, `"log_diff"`, `"avg_change"`, `"avg_growth"`, and `"direct_average_growth"`. |

### Output

Returns a `pandas.DataFrame` indexed by `date`. Column names are
`{target}_{transform}_h{horizon}`.

| Transform | Formula aligned on row `t` |
| --- | --- |
| `"level"` | `x[t + h]` |
| `"change"` | `x[t + h] - x[t]` |
| `"growth"` | `x[t + h] / x[t] - 1` |
| `"log_growth"` | `log(x[t + h]) - log(x[t])`; non-positive pairs become missing. |
| `"average_change"` | Average of one-period changes from `t+1` through `t+h`. |
| `"average_growth"` | Average of one-period simple growth rates from `t+1` through `t+h`. |
| `"average_log_growth"` | Average of one-period log growth rates from `t+1` through `t+h`. |

The final `h` rows are missing by construction because the future target is not
observed.

The returned frame also carries `attrs["macroforecast_target_metadata"]`.
Core columns are `target_column`, `source`, `horizon`, `step`, `mode`,
`transform`, `operation`, `formula`, `aggregation`, and `used_for_horizons`.

## average_target

```python
macroforecast.feature_engineering.average_target(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    transform: str = "change",
) -> pandas.DataFrame
```

`average_target()` is a readability wrapper for direct average targets.
It returns the same output as:

```python
mf.feature_engineering.direct_target(..., transform="average_change")
mf.feature_engineering.direct_target(..., transform="average_growth")
mf.feature_engineering.direct_target(..., transform="average_log_growth")
```

This is the direct average approach: one final target column is created per
requested horizon, and a later model can fit that column directly.

| `transform` | Meaning |
| --- | --- |
| `"change"` | Average one-period differences over the future path. |
| `"growth"` | Average one-period simple growth rates over the future path. |
| `"log_growth"` | Average one-period log growth rates over the future path. |

## path_targets

```python
macroforecast.feature_engineering.path_targets(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    transform: str = "change",
) -> pandas.DataFrame
```

`path_targets()` creates step-level future targets for path-average
forecasting. For `horizon=3`, it returns step columns for `t+1`, `t+2`, and
`t+3`. The model stage should fit and forecast each step target separately.
The evaluation stage can then average the step forecasts for the final horizon.

```python
path_y = mf.feature_engineering.path_targets(
    processed,
    target="INDPRO",
    horizon=3,
    transform="growth",
)
```

Output columns are named `{target}_{transform}_step{step}`. Metadata includes
`metadata["path_target"]["columns_by_horizon"]`, which records which step
columns should be averaged for each requested horizon.

`macroforecast_target_metadata` marks these rows with `mode="path"`,
`operation="path_step_target"`, a non-null `step`, and
`aggregation="average_step_forecasts_in_evaluation"`. This records the intended
later use without moving model fitting or forecast averaging into this stage.

## lag

```python
macroforecast.feature_engineering.lag(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    lags: Iterable[int] | int = (1,),
    drop_missing: bool = False,
) -> pandas.DataFrame
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `data` | feature input | required | Canonical macroforecast input. |
| `columns` | iterable or `None` | all columns | Source columns to lag. |
| `lags` | int or iterable of ints | `(1,)` | Non-negative lags. `lags=3` expands to `1, 2, 3`; `lags=0` means current values only; pass `(0, 1, 3)` for exact lags including current values. |
| `drop_missing` | `bool` | `False` | Drop rows with any lag-induced missing values. |

### Output

Returns a `pandas.DataFrame` with columns named `{column}_lag{lag}`.

## mixed_frequency_lags

```python
macroforecast.feature_engineering.mixed_frequency_lags(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    target: str | None = None,
    anchor_dates: Iterable[object] | None = None,
    columns: Iterable[str] | None = None,
    lags: Iterable[int] | int = (0, 1, 2),
    frequency_by_column: Mapping[str, str] | None = None,
    target_frequency: str | None = None,
    anchor_position: str = "date",
    drop_missing: bool = False,
) -> pandas.DataFrame
```

Builds a lag matrix for MIDAS-style and other mixed-frequency regressions.
Unlike `lag()`, lags are measured in each source column's native frequency,
using `metadata["native_frequency_by_column"]` from `mf.data.set_frequencies()`
or `mf.data.combine(..., frequency="native")`.

Lookup is period based, not timestamp-string based. A monthly source dated
`2020-03-01` and the same source dated `2020-03-31` both map to the March 2020
source period. This prevents month-start/month-end conventions from silently
breaking MIDAS lag construction.

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `data` | feature input | required | Panel or bundle with a mixed-frequency contract. |
| `target` | `str` or `None` | input target if available | Column whose non-missing dates define anchors when `anchor_dates` is not supplied. |
| `anchor_dates` | iterable or `None` | target non-missing dates | Explicit rows to build features for. |
| `columns` | iterable or `None` | all non-target columns | Source columns to lag. |
| `lags` | int or iterable | `(0, 1, 2)` | Native-frequency lags. Pass an iterable for exact lags. |
| `frequency_by_column` | mapping or `None` | metadata map | Override native frequency by source column. |
| `target_frequency` | `str` or `None` | target metadata/inference | Frequency used when positioning anchor dates. |
| `anchor_position` | `str` | `"date"` | `"date"`, `"period_start"`, or `"period_end"`. |
| `drop_missing` | `bool` | `False` | Drop rows with missing lag values. |

For FRED-QD-style quarterly targets dated at the first month of the quarter,
use `target_frequency="quarterly", anchor_position="period_end"` to construct
monthly lag blocks at the quarter-end month:

```python
X_midas = mf.feature_engineering.mixed_frequency_lags(
    bundle,
    target="GDPC1",
    columns=["PAYEMS", "INDPRO"],
    lags=range(0, 12),
    target_frequency="quarterly",
    anchor_position="period_end",
)
```

The output columns are named `{column}_lag{lag}`, which is the grouping format
expected by `mf.models.midas_almon`, `mf.models.midas_beta`, and
`mf.models.midas_step`.

The returned DataFrame records metadata in two places:

| Location | Meaning |
| --- | --- |
| `attrs["macroforecast_metadata"]["feature_engineering_mixed_frequency_lags"]` | Target, anchor dates, selected columns, exact lags, frequency map, anchor positioning, lookup calendar, and row counts before/after `drop_missing`. |
| `attrs["macroforecast_feature_metadata"]` | One row per generated lag feature, including source column, lag, native source frequency, anchor position, and lookup start/end dates. |

## rolling_mean

```python
macroforecast.feature_engineering.rolling_mean(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    windows: Iterable[int] | int = (3,),
    min_periods: int | None = None,
    shift: int = 0,
    drop_missing: bool = False,
) -> pandas.DataFrame
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `columns` | iterable or `None` | all columns | Source columns. |
| `windows` | positive int or iterable | `(3,)` | Rolling-window lengths. |
| `min_periods` | positive int or `None` | window length | Minimum observations required for a value. |
| `shift` | non-negative int | `0` | Shift source series before rolling. Use `1` for strictly lagged rolling means. |
| `drop_missing` | `bool` | `False` | Drop rows with window-induced missing values. |

### Output

Returns a `pandas.DataFrame` with columns named `{column}_roll{window}_mean`.
When `shift > 0`, names end in `_lag{shift}`.

## moving_average_ladder

```python
macroforecast.feature_engineering.moving_average_ladder(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    windows: Iterable[int] | None = None,
    max_window: int = 12,
    min_periods: int | None = None,
    shift: int = 0,
    drop_missing: bool = False,
) -> pandas.DataFrame
```

### Meaning

`moving_average_ladder()` builds a stacked block of trailing moving averages at
multiple horizons. With the default `max_window=12`, the implicit windows are
`1, 2, 4, 8`. Pass `windows=(1, 2, 4, 8, 12)` or any other explicit sequence
when the endpoint should be included.

### MARX in macroforecast

Some papers describe this step as `marx_features(P)` or Moving Average Rotation
of `X` (MARX). In `macroforecast`, the direct pandas form is the following
explicit moving-average-ladder call:

```python
MARX = mf.feature_engineering.moving_average_ladder(
    X,
    windows=range(1, P + 1),
    shift=1,
)
```

This means that, for each source series, the feature block contains increasing
moving averages of lagged `X`: one-period lag, two-period average ending at
`t-1`, three-period average ending at `t-1`, and so on through `P`. The
`shift=1` part is important because the MARX block uses lagged predictors, not
the contemporaneous realization at the forecast date.

The runner-safe shorthand is `marx_step(max_lag=P)`, used inside
`feature_spec(..., steps=[...])`. It emits the same columns as the direct call,
but lets `forecasting.run()` decide which rows are available for any fitted
state through `feature_policy`.

The original author R snippet builds a VAR lag matrix ordered as lag 1 for all
variables, lag 2 for all variables, and so on. Then each lag-`l` slot for a
variable is replaced by the row average of that variable's lag 1 through lag
`l` columns. The direct call and `marx_step(scale_lags=False)` match that
unscaled calculation. Through `feature_matrix(..., specification="MARX",
scale_marx=True)` or `marx_step(scale_lags=True)`, macroforecast also supports
the optional R-code scaling step: z-score the lag matrix first using sample
standard deviations, then apply the same increasing-lag averages. In
`feature_spec()` mode, `scale_lags=True` fits those lag-matrix center/scale
values only on the feature-fit panel and reuses them for validation/test rows.

This function is not PCA. It is the moving-average block used before optional
factor extraction. Moving-average PCA should be represented as:

```python
ma_block = mf.feature_engineering.moving_average_ladder(panel, windows=(1, 2, 4, 8, 12))
factors = mf.feature_engineering.pca_features(ma_block, fit_policy="expanding")
```

Keeping the moving-average block and PCA step separate matters because PCA is a
fit-based transformation. Running PCA on the full sample before train/test or
walk-forward boundaries would leak future information.

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `columns` | iterable or `None` | all columns | Source columns. |
| `windows` | iterable of positive ints or `None` | powers of two up to `max_window` | Exact moving-average windows. |
| `max_window` | positive int | `12` | Used only when `windows=None`; default creates `1, 2, 4, 8`. |
| `min_periods` | positive int or `None` | window length | Minimum observations required for a value. |
| `shift` | non-negative int | `0` | Shift source series before rolling. Use `1` for strictly lagged moving averages. |
| `drop_missing` | bool | `False` | Drop rows with window-induced missing values. |

### Output

Returns a `pandas.DataFrame` with columns named `{column}_ma{window}`. When
`shift > 0`, names end in `_lag{shift}`.

## scale_features

```python
macroforecast.feature_engineering.scale_features(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    method: str = "zscore",
    fit_policy: str = "expanding",
    min_train_size: int | None = None,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pandas.DataFrame
```

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `method` | str | `"zscore"` | `"zscore"`, `"minmax"`, `"robust"`; aliases: `"standard"`, `"standardize"`, `"min_max"`. |
| `fit_policy` | str | `"expanding"` | `"expanding"` or `"full_sample"`. |
| `min_train_size` | positive int or `None` | `5` | Minimum complete rows before emitting scaled values. |
| `drop_missing` | bool | `False` | Drop rows where scaling is unavailable. |
| `warn_full_sample` | bool | `True` | Warn when `fit_policy="full_sample"` is used. |

## pca_features

```python
macroforecast.feature_engineering.pca_features(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 1,
    fit_policy: str = "expanding",
    min_train_size: int | None = None,
    scale: bool = True,
    prefix: str = "pc",
    drop_missing: bool = False,
    random_state: int | None = None,
    warn_full_sample: bool = True,
) -> pandas.DataFrame
```

`pca_features()` returns columns named `{prefix}1`, `{prefix}2`, and so on.
The default `fit_policy="expanding"` avoids full-sample leakage. Use
`fit_policy="full_sample"` only after the input sample has already been split
or for exploratory diagnostics. `warn_full_sample=True` emits a warning for
that choice.

## sparse_pca_chen_rohe_features

```python
macroforecast.feature_engineering.sparse_pca_chen_rohe_features(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 4,
    zeta: float = 0.0,
    max_iter: int = 200,
    var_innovations: bool = False,
    prefix: str | None = None,
    min_train_size: int | None = None,
    drop_missing: bool = False,
    random_state: int | None = 0,
    warn_full_sample: bool = True,
) -> pandas.DataFrame
```

`sparse_pca_chen_rohe_features()` implements the legacy package's
Chen-Rohe-style Sparse Component Analysis (SCA) routine directly with NumPy. It
is not `sklearn.decomposition.SparsePCA`. The transform centers the selected
predictor panel, alternates over the score and loading matrices, and constrains
the loading matrix with an L1 budget `zeta`.

The direct callable fits on all complete rows of the supplied input. It warns by
default because that is a full-input fitted transform. For strict walk-forward
forecasting, use `sparse_pca_chen_rohe_step()` inside `feature_spec(...)`; the
runner will fit the sparse loading matrix on the feature-fit panel and reuse the
fixed loading matrix on validation/test rows.

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | iterable or `None` | all columns | Predictor columns used to fit sparse components. |
| `n_components` | positive int | `4` | Requested number of sparse components. The resolved number is `min(n_components, complete_rows, selected_columns)`. |
| `zeta` | non-negative float | `0.0` | L1 loading-budget parameter. `zeta <= 0` uses the resolved component count, matching the legacy default. Smaller values create sparser loadings. |
| `max_iter` | positive int | `200` | Maximum alternating updates. |
| `var_innovations` | bool | `False` | If `True`, fit a VAR(1) on the sparse factor scores and return residual sparse macro-finance factors. |
| `prefix` | string or `None` | `"sca"` or `"scaf"` | Output prefix. Default is `"sca"`; with `var_innovations=True`, default is `"scaf"`. |
| `min_train_size` | positive int or `None` | `1`, or `3` when `var_innovations=True` | Minimum complete rows before emitting factors. |
| `drop_missing` | bool | `False` | Drop rows where sparse factors are unavailable. |
| `random_state` | int or `None` | `0` | Initialization seed for the alternating algorithm. |
| `warn_full_sample` | bool | `True` | Warn because the direct callable fits on all complete input rows. |

### Output

Returns a `pandas.DataFrame` indexed by `date`, with columns such as `sca1`,
`sca2`, or `scaf1`. Metadata is stored under
`attrs["macroforecast_metadata"]["feature_engineering_sparse_pca_chen_rohe"]`.
The stage records selected columns, requested/resolved components, `zeta`,
resolved `zeta`, iteration count, final objective, VAR-innovation use, fit rows,
and `fit_policy="full_input_complete_rows"`.

`macroforecast_feature_metadata` records one row per factor with
`operation="sparse_pca_chen_rohe"`, the source columns, component index, and fit
policy.

## varimax_features

```python
macroforecast.feature_engineering.varimax_features(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    max_iter: int = 50,
    tol: float = 1e-7,
    prefix: str = "varimax",
    min_train_size: int | None = None,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pandas.DataFrame
```

`varimax_features()` rotates a factor-score panel with an orthogonal varimax
rotation. It should be applied to factor columns, not raw macro variables. A
typical direct use is:

```python
factors = mf.feature_engineering.pca_features(
    processed,
    columns=["INDPRO", "PAYEMS", "UNRATE"],
    n_components=3,
    fit_policy="full_sample",
    warn_full_sample=False,
)
rotated = mf.feature_engineering.varimax_features(factors, warn_full_sample=False)
```

The direct callable fits the rotation on all complete rows and warns by default.
For strict walk-forward forecasting, use:

```python
spec = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors=["PAYEMS", "UNRATE", "HOUST"],
    steps=[
        mf.feature_engineering.pca_step(name="pc", n_components=3, include=False),
        mf.feature_engineering.varimax_step(name="rot", input="pc"),
    ],
)
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `columns` | iterable or `None` | all columns | Factor-score columns to rotate. |
| `max_iter` | positive int | `50` | Maximum varimax iterations. |
| `tol` | non-negative float | `1e-7` | Convergence tolerance for the rotation objective. |
| `prefix` | string | `"varimax"` | Output prefix. |
| `min_train_size` | positive int or `None` | `1` | Minimum complete rows before emitting rotated factors. |
| `drop_missing` | bool | `False` | Drop rows where rotated factors are unavailable. |
| `warn_full_sample` | bool | `True` | Warn because the direct callable fits on all complete input rows. |

### Output

Returns a `pandas.DataFrame` with columns such as `varimax1`, `varimax2`, and so
on. Metadata is stored under `metadata["feature_engineering_varimax"]`, and
`macroforecast_feature_metadata` records `operation="varimax"`, component index,
source factor columns, and fit policy.

## sliced_inverse_regression_features

```python
macroforecast.feature_engineering.sliced_inverse_regression_features(
    data,
    target: str | pandas.Series | None = None,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 3,
    n_slices: int = 10,
    scaling_policy: str = "scaled_pca",
    prefix: str = "sir",
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pandas.DataFrame
```

`sliced_inverse_regression_features()` implements a target-aware SIR factor
transform. It aligns the predictor panel with a target series, standardizes
predictors, optionally applies predictive column scaling, slices observations by
the target distribution, and projects the full panel onto the leading
between-slice directions.

The direct callable fits on all target-aligned complete rows in the supplied
input. For strict walk-forward forecasting, use
`sliced_inverse_regression_step()` inside `feature_spec(...)`; the runner then
fits SIR directions only on the feature-fit panel and applies the fixed
directions to validation/test rows.

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `target` | string, `Series`, or `None` | input target metadata | Target signal used for SIR slicing and optional predictive scaling. |
| `columns` | iterable or `None` | all non-target columns | Predictor columns. |
| `n_components` | positive int | `3` | Number of SIR factors to return. If the effective rank is smaller, remaining columns are zero-padded for stable shape. |
| `n_slices` | int | `10` | Target-distribution slices. Must be at least `2`; internally capped by aligned row count. |
| `scaling_policy` | string | `"scaled_pca"` | `"scaled_pca"`, `"marginal_R2"`, or `"none"`. |
| `prefix` | string | `"sir"` | Output prefix. Use `prefix="factor_"` for legacy-style names `factor_1`, `factor_2`, ... |
| `drop_missing` | bool | `False` | Drop rows with missing predictor values after projection. |
| `warn_full_sample` | bool | `True` | Warn because the direct callable fits on all target-aligned complete rows. |

### Output

Returns columns such as `sir1`, `sir2`, and so on. Metadata is stored under
`metadata["feature_engineering_sliced_inverse_regression"]` and records the
target, predictor columns, requested/resolved components, slices, scaling policy,
fit row count, and `fit_policy="full_input_target_aligned_rows"`.

## Target-Aware Feature Steps

`feature_spec(..., steps=[...])` also supports target-aware fitted transforms.
These steps use the resolved `feature_spec()` target during `.fit(...)`, store a
fixed fit state, and do not look at target values during `.transform(...)`.

```python
features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors=["PAYEMS", "UNRATE", "HOUST"],
    steps=[
        mf.feature_engineering.scale_step(name="scaled", include=False),
        mf.feature_engineering.partial_least_squares_step(
            name="pls",
            input="scaled",
            n_components=2,
            min_train_size=60,
        ),
    ],
)
```

Target-aware steps require exactly one resolved target column. In practice that
means one `target` and one `horizon` for the step pipeline. If multiple targets
or horizons are requested, fit raises before any model is run.

| Step builder or method | Direct callable | Main options | Output |
| --- | --- | --- | --- |
| `partial_least_squares_step()` | `partial_least_squares_features()` | `n_components`, `columns`, `min_train_size`, `prefix` | `pls1`, `pls2`, ... |
| `sliced_inverse_regression_step()` | `sliced_inverse_regression_features()` | `n_components`, `n_slices`, `scaling_policy`, `min_train_size`, `prefix` | `sir1`, `sir2`, ... |
| `"variance_selection"` | `variance_selection()` | `n_features`; `columns`; `min_train_size` | Selected input columns. |
| `"correlation_selection"` | `correlation_selection()` | `n_features`; `columns`; `min_train_size` | Selected input columns. |
| `"lasso_selection"` | `lasso_selection()` | `n_features`; `alpha`; `min_train_size` | Selected input columns. |
| `"lasso_path_selection"` | `lasso_path_selection()` | `n_features`; `eps`; `n_alphas`; `normalize_features`; `positive` | Selected input columns. |
| `"rfe_selection"` | `rfe_selection()` | `n_features`; `estimator`; `step`; `use_cv`; `cv_folds` | Selected input columns. |
| `"boruta_selection"` | `boruta_selection()` | `n_features`; `n_estimators`; `max_iter`; `alpha`; `include_tentative` | Selected input columns. |
| `"stability_selection"` | `stability_selection()` | `n_features`; `n_subsamples`; `subsample_fraction`; `pi_threshold`; `base_estimator` | Selected input columns. |
| `"genetic_selection"` | `genetic_selection()` | `n_features`; `population_size`; `n_generations`; `crossover_prob`; `fitness_estimator` | Selected input columns. |

Fit-state metadata records the resolved target column, selected source columns,
requested/resolved component or feature count, fit row count, and
`fit_policy="fixed_fit_panel_target_aligned_rows"` for target-dependent methods.
For `method="variance_selection"`, no target is used and the fit policy is
`fixed_fit_panel_columns`.

Feature selection deliberately has no generic wrapper step.
Use each algorithm name directly inside `feature_spec()`:

```python
features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors=["PAYEMS", "UNRATE", "HOUST"],
    steps=[
        {"name": "boruta", "method": "boruta_selection", "n_features": 2},
    ],
)
```

## group_pca

```python
macroforecast.feature_engineering.group_pca(
    data,
    *,
    groups: Mapping[str, Iterable[str]],
    metadata: Mapping[str, object] | None = None,
    n_components: int | Mapping[str, int] = 1,
    fit_policy: str = "expanding",
    min_train_size: int | None = None,
    scale: bool = True,
    prefix: str | None = None,
    drop_missing: bool = False,
    random_state: int | None = None,
    warn_full_sample: bool = True,
) -> pandas.DataFrame
```

`group_pca()` extracts PCA factors separately within named groups. It is a
generic grouped factor transform. FAVAR-specific slow/fast grouping,
observed-policy variables, VAR dynamics, identification, and IRFs belong to
later model and evaluation stages.

```python
factors = mf.feature_engineering.group_pca(
    processed,
    groups={
        "real_activity": ["INDPRO", "PAYEMS", "UNRATE"],
        "prices": ["CPIAUCSL", "PPIACO"],
    },
    n_components={"real_activity": 3, "prices": 2},
    fit_policy="expanding",
)
```

Output columns use the group name as the prefix by default:

```text
real_activity1, real_activity2, real_activity3, prices1, prices2
```

`group_pca_step()` provides the same operation inside `compose_features()`.

### Supervised And Sparse Component Boundary

Unsupervised group PCA belongs in `feature_engineering` because it only uses
the predictor panel. PLS and SIR are target-aware and are available as
runner-safe feature steps when the resolved `feature_spec()` target is single.
Supervised PCA variants that fit a full predictive model still belong in
`macroforecast.models`. Chen-Rohe sparse component analysis is unsupervised and
is available as `sparse_pca_chen_rohe_features()` /
`sparse_pca_chen_rohe_step()`.

## maf_features

```python
macroforecast.feature_engineering.maf_features(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    max_lag: int = 12,
    lags: Iterable[int] | None = None,
    n_components: int = 2,
    fit_policy: str = "expanding",
    min_train_size: int | None = None,
    scale: bool = False,
    prefix: str = "maf",
    drop_missing: bool = False,
    random_state: int | None = None,
    warn_full_sample: bool = True,
) -> pandas.DataFrame
```

`maf_features()` implements Moving Average Factors. For each selected variable
`x_k`, it builds a variable-specific lag panel:

```text
[x_k(t), x_k(t-1), ..., x_k(t-P)]
```

Then it extracts PCA components from that lag panel only. This is different
from `pca_features()`, which runs PCA across all selected variables, and
different from `moving_average_pca_lags()`, which runs PCA on a moving-average
block.

The MAF implementation is intentionally limited to the construction described
in the paper: variable-specific lag panels followed by PCA. The package does
not assume undocumented author-code details beyond that description.

Validation status: MARX is tested against the author-supplied R-loop pattern.
MAF is tested for the documented variable-specific lag-panel PCA contract, but
there is no author-code benchmark in the package yet. If author MAF code becomes
available, it should be added as a separate equivalence test before tightening
the claim.

```python
MAF = mf.feature_engineering.maf_features(
    X,
    max_lag=12,
    n_components=2,
    fit_policy="expanding",
)
```

With two input series, this returns columns like:

```text
INDPRO_maf1, INDPRO_maf2, PAYEMS_maf1, PAYEMS_maf2
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `columns` | iterable or `None` | all columns | Source series for variable-specific lag panels. |
| `max_lag` | non-negative int | `12` | Used when `lags=None`; builds lags `0` through `max_lag`. |
| `lags` | iterable of non-negative ints or `None` | `None` | Exact lag set. Overrides `max_lag`. |
| `n_components` | positive int | `2` | Number of MAF components per source series. |
| `fit_policy` | str | `"expanding"` | `"expanding"` or `"full_sample"`. |
| `min_train_size` | positive int or `None` | `max(5, n_components + 1)` | Minimum complete rows before emitting PCA values. |
| `scale` | bool | `False` | Whether to z-score the lag columns before PCA. Default is `False` because lags of the same variable are already in the same unit. |
| `prefix` | str | `"maf"` | Component label used in output names. |
| `drop_missing` | bool | `False` | Drop rows where MAF values are unavailable. |
| `warn_full_sample` | bool | `True` | Warn when `fit_policy="full_sample"` is used. |

### Output

Returns a `pandas.DataFrame` with one block per source series. Metadata is
stored in `metadata["feature_engineering_maf"]`, and
`macroforecast_feature_metadata` records the source series for each component.

## feature_matrix

```python
macroforecast.feature_engineering.feature_matrix(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    specification: str | Iterable[str] = "X",
    columns: Iterable[str] | None = None,
    level_data: feature input | None = None,
    level_columns: Iterable[str] | None = None,
    lags: Iterable[int] | int = (0,),
    max_lag: int = 12,
    n_factors: int = 8,
    n_maf_components: int = 2,
    fit_policy: str = "expanding",
    min_train_size: int | None = None,
    include_current_factor: bool = True,
    scale_factors: bool = True,
    scale_marx: bool = False,
    scale_maf: bool = False,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pandas.DataFrame
```

`feature_matrix()` builds named combinations used in macro-ML forecasting
papers without requiring the user to hand-write `compose_features()` steps.

| Block | Package implementation |
| --- | --- |
| `X` | `lag(data, lags=...)` on the supplied, usually preprocessed, panel. |
| `F` | PCA factors from the supplied panel, then lags of those factors. |
| `MARX` | `moving_average_ladder(data, windows=range(1, max_lag + 1), shift=1)`; use `marx_step()` for runner-safe windowed construction. With `scale_marx=True`, first z-score the full lag matrix with sample standard deviations and then average lag 1 through lag `l`. |
| `MAF` | `maf_features(data, max_lag=max_lag, n_components=n_maf_components)`. |
| `LEVEL` / `H` | `lag(level_data, lags=...)`; requires a separate `level_data` input. |

`specification` can be a string such as `"F-X-MARX"` or an iterable such as
`("F", "X", "MAF")`. Output columns are prefixed by block, for example
`F__F1_lag0`, `X__INDPRO_lag1`, `MARX__INDPRO_ma3_lag1`, or
`MAF__INDPRO_maf1`.

`include_current_factor=True` ensures the `F` block includes current factors
even when `lags` contains only positive values such as `range(1, 13)`. Set it
to `False` when the factor block should exactly follow the supplied lag set.

### Paper-Style Specifications

The paper-style feature families are handled directly by `feature_matrix()`.
The parser accepts `-`, `+`, or `_` separators.

| Specification | Meaning |
| --- | --- |
| `"X"` | Lagged predictor panel. |
| `"F"` | PCA factors from the predictor panel, then factor lags. |
| `"F-X"` | Factor lags plus lagged predictors. |
| `"H"` or `"LEVEL"` | Lagged level variables from `level_data`. |
| `"X-H"` | Lagged predictors plus lagged level variables from `level_data`. |
| `"F-X-H"` or `"F-X-LEVEL"` | `F-X` plus lagged level variables from `level_data`. |
| `"F-X-MARX"` | `F-X` plus MARX increasing averages of lagged predictors. |
| `"F-X-MAF"` | `F-X` plus Moving Average Factors. |
| `"F-X-H-MARX"` | `F-X-H` plus MARX. |
| `"F-X-H-MAF"` | `F-X-H` plus MAF. |

| Specification | Requires `level_data` | Fitted transform | Main output blocks |
| --- | --- | --- | --- |
| `"X"` | No | No | `X__...` |
| `"F"` | No | PCA | `F__...` |
| `"F-X"` | No | PCA | `F__...`, `X__...` |
| `"H"` / `"LEVEL"` | Yes | No | `LEVEL__...` |
| `"X-H"` | Yes | No | `X__...`, `LEVEL__...` |
| `"F-X-H"` | Yes | PCA | `F__...`, `X__...`, `LEVEL__...` |
| `"F-X-MARX"` | No | PCA; optional MARX scaling | `F__...`, `X__...`, `MARX__...` |
| `"F-X-MAF"` | No | PCA and MAF PCA | `F__...`, `X__...`, `MAF__...` |
| `"F-X-H-MARX"` | Yes | PCA; optional MARX scaling | `F__...`, `X__...`, `LEVEL__...`, `MARX__...` |
| `"F-X-H-MAF"` | Yes | PCA and MAF PCA | `F__...`, `X__...`, `LEVEL__...`, `MAF__...` |

Examples:

```python
FX = mf.feature_engineering.feature_matrix(
    processed,
    specification="F-X",
    lags=range(0, 13),
    n_factors=8,
    fit_policy="expanding",
)

FXH = mf.feature_engineering.feature_matrix(
    processed,
    specification="F-X-H",
    level_data=raw_bundle,
    lags=range(0, 13),
    n_factors=8,
)

FXMARX = mf.feature_engineering.feature_matrix(
    processed,
    specification="F-X-MARX",
    lags=range(0, 13),
    max_lag=12,
    n_factors=8,
    scale_marx=False,
)

FXMAF = mf.feature_engineering.feature_matrix(
    processed,
    specification="F-X-MAF",
    lags=range(0, 13),
    max_lag=12,
    n_factors=8,
    n_maf_components=2,
)
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `specification` | string or iterable | `"X"` | Blocks `X`, `F`, `MARX`, `MAF`, `LEVEL`/`H`; strings can use `-`, `+`, or `_` separators. |
| `columns` | iterable or `None` | all columns | Source columns from the preprocessed panel. |
| `level_data` | feature input or `None` | `None` | Required when `specification` includes `LEVEL` or `H`. |
| `level_columns` | iterable or `None` | `columns` | Level-data columns to include. |
| `lags` | int or iterable | `(0,)` | Lag set for `X`, `F`, and `LEVEL`. |
| `max_lag` | positive int | `12` | Maximum lag for `MARX` and `MAF`. |
| `n_factors` | positive int | `8` | Number of PCA factors for `F`. |
| `n_maf_components` | positive int | `2` | MAF components per source variable. |
| `fit_policy` | str | `"expanding"` | `"expanding"` or `"full_sample"` for fitted transforms. |
| `min_train_size` | positive int or `None` | transform-specific | Minimum complete rows for fitted transforms. |
| `include_current_factor` | bool | `True` | Force lag 0 in the `F` block even when `lags` excludes it. |
| `scale_factors` | bool | `True` | Z-score variables before PCA in the `F` block. |
| `scale_marx` | bool | `False` | Match the optional author R-code scaling step for `MARX`. |
| `scale_maf` | bool | `False` | Z-score variable-specific MAF lag panels before PCA. |
| `drop_missing` | bool | `False` | Drop rows with missing feature values. |
| `warn_full_sample` | bool | `True` | Warn when any fitted block uses `fit_policy="full_sample"`. |

```python
Z = mf.feature_engineering.feature_matrix(
    processed,
    specification="F-X-MARX",
    lags=range(0, 13),
    max_lag=12,
    n_factors=8,
    fit_policy="expanding",
    drop_missing=True,
)
```

Use `level_data=` when the combination includes level variables:

```python
Z = mf.feature_engineering.feature_matrix(
    processed,
    specification="F-LEVEL",
    level_data=raw_bundle,
    lags=range(0, 13),
)
```

## compose_features

```python
macroforecast.feature_engineering.compose_features(
    data,
    steps,
    *,
    metadata: Mapping[str, object] | None = None,
    columns: Iterable[str] | None = None,
    include_original: bool = False,
    drop_missing: bool = False,
) -> pandas.DataFrame
```

`steps` is an ordered list of mappings. Each step has:

| Key | Meaning |
| --- | --- |
| `name` | Step name. Later steps can reference this name. |
| `method` | One of `"lag"`, `"rolling_mean"`, `"moving_average_ladder"`, `"marx"`, `"transform"`, `"seasonal_lag"`, `"season_dummy"`, `"fourier"`, `"polynomial"`, `"interaction"`, `"maf"`, `"scale"`, `"pca"`, `"sparse_pca_chen_rohe"`, `"varimax"`, `"group_pca"`, `"time"`. |
| `input` | `"panel"` by default, or a previous step name. |
| `include` | Whether this step's output is included in final `X`; default `True`. |
| other keys | Method-specific parameters such as `lags`, `windows`, `n_components`, `fit_policy`, `warn_full_sample`, or `columns`. |

Examples:

```python
# PCA, then lags of the PCA factors.
X = mf.feature_engineering.compose_features(
    processed,
    [
        {"name": "pc", "method": "pca", "columns": ["PAYEMS", "INDPRO"], "n_components": 2, "include": False},
        {"name": "pc_lags", "method": "lag", "input": "pc", "lags": [1, 2, 3]},
    ],
)

# Lags first, then PCA on the lag block.
X = mf.feature_engineering.compose_features(
    processed,
    [
        {"name": "lag_block", "method": "lag", "lags": [0, 1, 2, 3], "include": False},
        {"name": "lag_pc", "method": "pca", "input": "lag_block", "n_components": 4},
    ],
)

# Moving-average ladder, PCA, then lags of the factor.
X = mf.feature_engineering.compose_features(
    processed,
    [
        {"name": "ma", "method": "moving_average_ladder", "windows": [1, 2, 4, 8, 12], "include": False},
        {"name": "ma_pc", "method": "pca", "input": "ma", "n_components": 4, "include": False},
        {"name": "ma_pc_lags", "method": "lag", "input": "ma_pc", "lags": [1, 2]},
    ],
)

# MAF as a direct block inside a composed feature matrix.
X = mf.feature_engineering.compose_features(
    processed,
    [
        {"name": "maf", "method": "maf", "max_lag": 12, "n_components": 2},
    ],
)

# MARX shorthand: increasing averages of lagged predictors.
X = mf.feature_engineering.compose_features(
    processed,
    [
        mf.feature_engineering.marx_step(max_lag=12, scale_lags=False),
    ],
)

# Extra deterministic transforms can be composed the same way.
X = mf.feature_engineering.compose_features(
    processed,
    [
        mf.feature_engineering.transform_step(name="log_ip", transform="log", columns=["INDPRO"], include=False),
        mf.feature_engineering.lag_step(name="log_ip_lag", input="log_ip", lags=[1, 2, 3]),
        mf.feature_engineering.interaction_step(name="cross", columns=["PAYEMS", "HOUST"]),
    ],
)
```

## time_features

```python
macroforecast.feature_engineering.time_features(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    trend: bool = True,
    month: bool = False,
    quarter: bool = False,
    year: bool = False,
) -> pandas.DataFrame
```

### Input And Output

| Option | Output columns |
| --- | --- |
| `trend=True` | `trend`, starting at `1.0`. |
| `month=True` | `month_01` through `month_12`. |
| `quarter=True` | `quarter_1` through `quarter_4`. |
| `year=True` | `year`. |

## Additional Transform Helpers

These helpers are feature-engineering transforms, not preprocessing t-codes.
Use them when the model feature set needs extra ML-oriented columns after the
canonical panel has already been cleaned.

| Function | Main options | Output |
| --- | --- | --- |
| `transform_features(data, transform=...)` | `transform`: `"log"`, `"diff"`, `"log_diff"`, `"pct_change"`, `"cumsum"`; `periods`; `columns`; `drop_missing` | `{column}_{transform}` columns. |
| `log_features`, `diff_features`, `log_diff_features`, `pct_change_features`, `cumsum_features` | Thin named wrappers around `transform_features`. | Same as above. |
| `seasonal_lag(data, season_length=12, lags=...)` | Seasonal step length and seasonal lag count. | `{column}_seasonlag{actual_lag}`. |
| `season_dummy(data, frequency="auto")` | `"month"` or `"quarter"`, optional `drop_first`. | Month or quarter dummies. |
| `fourier_features(data, period=12, order=2)` | Seasonal period and harmonic order. | Sine/cosine seasonal terms. |
| `polynomial_features(data, degree=2)` | `degree`, `include_bias`, `interaction_only`. | Named polynomial expansion columns. |
| `interaction_features(data, order=2)` | Exact-order pure interaction expansion without lower-order terms or powers. | `interaction_{col1}__{col2}` style columns. |
| `hp_filter_features(data, lamb=129600.0)` | HP `lambda`, `component`: `"cycle"`, `"trend"`, or `"both"`; `warn_full_sample=True`. | HP cycle/trend columns. |
| `hamilton_filter_features(data, h=8, p=4)` | Hamilton horizon `h`, regressor count `p`, `component`, `fit_policy`: `"expanding"` or `"full_sample"`, and missing policy. | `{column}_hamilton_cycle` and/or `{column}_hamilton_trend`. |
| `savitzky_golay_features(data, window_length=5, polyorder=2)` | Centered filter window, polynomial order, derivative; `warn_full_sample=True`. | Smoothed columns. |
| `wavelet_features(data, n_levels=3)` | Causal rolling approximation/detail levels; `wavelet` name is recorded for compatibility. | `{column}_wA{level}`, `{column}_wD{level}`. |
| `adaptive_ma_rf_features(data, sided="two")` | Random forest smoother over time; `sided="two"` warns, `sided="one"` uses expanding one-sided fits. | `{column}_albama`. |
| `asymmetric_trim_features(data)` | Sorts each row's selected columns in ascending order. | `rank_1`, `rank_2`, ... |
| `partial_least_squares_features(data, target=..., n_components=...)` | Target-aware PLSRegression scores; warns by default. | `pls1`, `pls2`, ... |
| `dfm_features(data, n_factors=...)` | Static DFM approximation by standardized PCA; warns by default. | `dfm1`, `dfm2`, ... |
| `variance_selection(data, n_features=...)` | Select by sample variance; no target required. | Subset of original columns. |
| `correlation_selection(data, target=..., n_features=...)` | Select by absolute target correlation. | Subset of original columns. |
| `lasso_selection(data, target=..., alpha=...)` | Select by absolute lasso coefficient. | Subset of original columns. |
| `lasso_path_selection(data, target=..., eps=..., n_alphas=...)` | Select by lasso-path inclusion frequency. | Subset of original columns. |
| `rfe_selection(data, target=..., estimator=...)` | Select by recursive feature elimination. | Subset of original columns. |
| `boruta_selection(data, target=..., n_estimators=..., max_iter=...)` | Select by Boruta-style shadow-feature tests. | Subset of original columns. |
| `stability_selection(data, target=..., n_subsamples=..., pi_threshold=...)` | Select by repeated sparse-model subsampling frequency. | Subset of original columns. |
| `genetic_selection(data, target=..., population_size=..., n_generations=...)` | Select by genetic subset search. | Subset of original columns. |
| `random_projection_features(data, n_components=...)` | Gaussian random projection; `warn_full_sample=True` by default. | `rp1`, `rp2`, ... |
| `nystroem_features(data, kernel="rbf", n_components=...)` | Kernel approximation settings; `warn_full_sample=True` by default. | `nys1`, `nys2`, ... |

`random_projection_features()` and `nystroem_features()` fit on complete rows
of the provided input and warn by default because the direct helpers are
full-input fitted helpers. For strict origin-by-origin forecasting, use
`random_projection_step()` and `nystroem_step()` inside `feature_spec()`; the
runner fits the projection/kernel state on the feature-fit panel and reuses the
fixed state for validation/test rows.

`hamilton_filter_features()` follows Hamilton's regression form:
`y[t+h] = a + b_0 y[t] + ... + b_{p-1} y[t-p+1] + e[t+h]`.
The fitted value is stored as the trend and the residual as the cycle, both
labeled at `t+h`. Defaults `h=8, p=4` match the common quarterly setting; for
monthly panels, `h=24, p=12` is the usual analogue. The default
`fit_policy="expanding"` estimates each row with only earlier completed
Hamilton-regression rows. `fit_policy="full_sample"` reproduces the ordinary
in-sample filter style and warns by default because it can use future
information relative to a forecasting origin.

## feature_spec

```python
macroforecast.feature_engineering.feature_spec(
    *,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    predictors: Literal["all"] | Iterable[str] | None = None,
    lags: Iterable[int] | int | None = (0, 1),
    target_lags: Iterable[int] | int | None = None,
    rolling_windows: Iterable[int] | int | None = None,
    rolling_min_periods: int | None = None,
    add_time: bool = False,
    time_trend: bool = True,
    time_month: bool = False,
    time_quarter: bool = False,
    time_year: bool = False,
    pca_components: int | None = None,
    pca_columns: Iterable[str] | None = None,
    pca_scale: bool = True,
    pca_prefix: str = "pc",
    steps: Iterable[Mapping[str, object]] | None = None,
    feature_steps: Iterable[Mapping[str, object]] | None = None,
    include_original: bool = False,
    target_transform: str = "level",
    target_mode: str = "direct",
    drop_missing: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> FeatureSpec
```

`feature_spec()` is the runner-safe feature contract. It is fitted by
`forecasting.run()` according to `feature_policy`, so stateful choices such as
scaling, PCA, grouped PCA, and MAF are estimated on the allowed
training/reference panel and reused when transforming validation/test rows.

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `target`, `targets` | string/iterable or `None` | from input metadata | Target column or target columns. |
| `horizon`, `horizons` | positive int/iterable or `None` | from input, then `(1,)` | Forecast horizon choices. |
| `predictors` | `"all"`, iterable, or `None` | from input, then all non-target columns | Predictor columns. |
| `lags` | int, iterable, or `None` | `(0, 1)` | Predictor lags. `0` includes the current predictor value at the forecast origin. `None` disables ordinary predictor lags. |
| `target_lags` | int, iterable, or `None` | `None` | Explicit autoregressive target lags added to `X` while keeping target columns out of `predictors`. Use this for AR-X and recursive runner designs. |
| `rolling_windows` | positive int/iterable or `None` | `None` | Optional rolling means. |
| `rolling_min_periods` | positive int or `None` | window length | Minimum observations for rolling means. |
| `add_time` | bool | `False` | Add deterministic date features. |
| `time_trend`, `time_month`, `time_quarter`, `time_year` | bool | see signature | Which deterministic date features to add. |
| `pca_components` | positive int or `None` | `None` | Fit PCA on the allowed feature-fit panel and append fixed-loadings components. |
| `pca_columns` | iterable or `None` | predictors | Columns used for PCA. |
| `pca_scale` | bool | `True` | Standardize PCA inputs using the feature-fit panel. |
| `pca_prefix` | string | `"pc"` | PCA output prefix. |
| `steps`, `feature_steps` | iterable of step mappings or `None` | `None` | Fit-aware feature-step pipeline. Use public step builders for deterministic/fitted transforms: `lag_step()`, `rolling_step()`, `moving_average_step()`, `marx_step()`, `transform_step()`, `seasonal_lag_step()`, `season_dummy_step()`, `fourier_step()`, `time_step()`, `polynomial_step()`, `interaction_step()`, `scale_step()`, `pca_step()`, `sparse_pca_chen_rohe_step()`, `varimax_step()`, `group_pca_step()`, `maf_step()`, `hamilton_step()`, `random_projection_step()`, `nystroem_step()`, `partial_least_squares_step()`, and `sliced_inverse_regression_step()`. For selection, pass step mappings with `method` equal to one of `variance_selection`, `correlation_selection`, `lasso_selection`, `lasso_path_selection`, `rfe_selection`, `boruta_selection`, `stability_selection`, or `genetic_selection`. `steps` and `feature_steps` are aliases; provide only one. |
| `include_original` | bool | `False` | Include the original predictor panel as part of `X` when using `steps`. |
| `target_transform` | string | `"level"` | Same target choices as `direct_target()`. |
| `target_mode` | string | `"direct"` | `"direct"` for horizon-level targets; `"path"` for step-level path targets. |
| `drop_missing` | bool | `True` | Drop rows with missing selected `X` or `y` during fit rows. Test rows are transformed without dropping by the runner. |
| `metadata` | mapping or `None` | `None` | User metadata stored inside the feature spec record. |

When `steps` are supplied, they replace the shortcut predictor options
`rolling_windows`, `add_time`, and `pca_components`; use the corresponding step
builders instead. The default `lags=(0, 1)` shortcut is also not used in step
mode unless you explicitly add a `lag_step()`.

### Output

Returns `FeatureSpec`. Important methods:

| Method | Output | Meaning |
| --- | --- | --- |
| `.fit(data)` | `FittedFeatureBuilder` | Fits reusable feature state for PCA/scaling/grouped PCA/MAF steps on the supplied panel. |
| `.fit_transform(data)` | `FeatureSet` | Fits and transforms the same panel. |
| `.to_dict()` | `dict` | JSON-ready feature choices for result metadata. |
| `.to_metadata()` | `dict` | Compact runner metadata. |

### Fit-Aware Step Pipeline

Step pipelines let the runner refit feature transformations inside each
forecasting window:

```python
features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors=["PAYEMS", "HOUST", "S&P 500"],
    steps=[
        mf.feature_engineering.scale_step(name="scaled", include=False),
        mf.feature_engineering.pca_step(
            name="pc",
            input="scaled",
            n_components=3,
            min_train_size=60,
            include=False,
        ),
        mf.feature_engineering.lag_step(name="pc_lag", input="pc", lags=range(0, 13)),
    ],
)
```

Each step has a `name`, `method`, `input`, and `include` flag. `input="panel"`
reads the original predictor panel; `input="<step name>"` reads a prior step.
If `include=False`, the step is an intermediate fitted transformation and its
output is not included in the final `X`, but its metadata is still recorded.

Stateful step builders are interpreted as fixed-fit transformations inside
`FeatureSpec`: the runner's `feature_policy` determines which rows are used to
fit the step. Any `fit_policy` value inherited from reusable step builders is
ignored in `feature_spec()` mode because the runner owns the temporal fit
policy.

| Step builder | Runner-safe behavior |
| --- | --- |
| `lag_step()` | Deterministic lag transform. |
| `rolling_step()` | Deterministic rolling mean transform. |
| `moving_average_step()` | Deterministic moving-average ladder. |
| `marx_step()` | MARX increasing lag averages; with `scale_lags=True`, fits lag-matrix center/scale on the feature-fit panel and reuses fixed parameters. |
| `transform_step()` | Deterministic column transform: `log`, `diff`, `log_diff`, `pct_change`, or `cumsum`. |
| `seasonal_lag_step()` | Deterministic seasonal lag such as 12-month or 4-quarter lag blocks. |
| `season_dummy_step()` | Deterministic month or quarter date dummies from the index. |
| `fourier_step()` | Deterministic Fourier seasonal terms from the index. |
| `time_step()` | Deterministic trend, month, quarter, and year columns from the index. |
| `polynomial_step()` | Deterministic polynomial expansion. |
| `interaction_step()` | Deterministic pure interaction terms. |
| `scale_step()` | Fits center/scale on the feature-fit panel, then applies fixed parameters. |
| `pca_step()` | Fits PCA loadings on the feature-fit panel, then applies fixed loadings. |
| `sparse_pca_chen_rohe_step()` | Fits Chen-Rohe sparse loadings on the feature-fit panel, then applies fixed loadings; optional `var_innovations=True` fits the VAR(1) residual mapping on the same feature-fit panel. |
| `varimax_step()` | Fits an orthogonal rotation on factor-score columns from the feature-fit panel, then applies the fixed rotation. |
| `group_pca_step()` | Fits separate PCA states inside named groups. |
| `maf_step()` | Fits variable-specific lag-panel PCA states for Moving Average Factors. |
| `hamilton_step()` | Fits Hamilton-regression beta on the feature-fit panel, then applies fixed beta to train/validation/test rows. |
| `random_projection_step()` | Fits a Gaussian random-projection transformer on the feature-fit panel and applies fixed components. |
| `nystroem_step()` | Fits Nystroem kernel-approximation landmarks on the feature-fit panel and applies fixed components. |
| `partial_least_squares_step()` | Fits PLS components against the single resolved target on the feature-fit panel and applies fixed weights. |
| `sliced_inverse_regression_step()` | Fits target-sliced directions on the feature-fit panel and applies fixed directions. |
| `variance_selection`, `correlation_selection`, `lasso_selection`, `lasso_path_selection`, `rfe_selection`, `boruta_selection`, `stability_selection`, `genetic_selection` | Select columns on the feature-fit panel and reuse the selected columns. Use these as `method` strings in step mappings, not as step-builder functions. |

In `feature_spec()` mode, `hamilton_step()` ignores the reusable step's
`fit_policy` argument because the runner's `feature_policy` owns the allowed
fit rows. The fitted state records `fit_policy="fixed_fit_panel"`. Runner-safe
Hamilton currently requires `missing="drop"`; impute missing values in
preprocessing before using it. The direct helper and `compose_features()` still
support `missing="interpolate"` for one-shot exploratory construction.

Direct pandas functions and runner-safe step builders are intentionally paired:

| Direct function | Runner-safe step | Fit state? | Typical use |
| --- | --- | --- | --- |
| `lag()` | `lag_step()` | No | Add current/lagged predictors. |
| `rolling_mean()` | `rolling_step()` | No | Add trailing rolling means. |
| `moving_average_ladder()` | `moving_average_step()` | No | Add multi-scale moving-average blocks. |
| `moving_average_ladder(..., shift=1)` / `feature_matrix(..., "MARX")` | `marx_step()` | Only when `scale_lags=True` | Add MARX increasing lag averages. |
| `transform_features()` and wrappers such as `log_features()` / `diff_features()` | `transform_step()` | No | Add ML-side transforms after preprocessing. |
| `seasonal_lag()` | `seasonal_lag_step()` | No | Add seasonal lag blocks. |
| `season_dummy()` | `season_dummy_step()` | No | Add calendar dummies. |
| `fourier_features()` | `fourier_step()` | No | Add deterministic seasonal Fourier terms. |
| `time_features()` | `time_step()` or `feature_spec(add_time=True, ...)` | No | Add deterministic trend/month/quarter/year terms. |
| `polynomial_features()` | `polynomial_step()` | No | Add nonlinear expansions. |
| `interaction_features()` | `interaction_step()` | No | Add cross-products. |
| `scale_features()` | `scale_step()` | Yes | Fit center/scale on allowed rows. |
| `pca_features()` | `pca_step()` | Yes | Fit PCA loadings on allowed rows. |
| `sparse_pca_chen_rohe_features()` | `sparse_pca_chen_rohe_step()` | Yes | Fit Chen-Rohe sparse component loadings on allowed rows. |
| `varimax_features()` | `varimax_step()` | Yes | Fit orthogonal factor rotation on allowed rows. |
| `group_pca()` | `group_pca_step()` | Yes | Fit separate PCA states by group. |
| `maf_features()` | `maf_step()` | Yes | Fit variable-specific lag-panel PCA states. |
| `hamilton_filter_features()` | `hamilton_step()` | Yes | Fit Hamilton-regression beta on allowed rows, then apply fixed beta. |
| `random_projection_features()` | `random_projection_step()` | Yes | Fit Gaussian random-projection state on allowed rows. |
| `nystroem_features()` | `nystroem_step()` | Yes | Fit Nystroem kernel landmarks on allowed rows. |
| `partial_least_squares_features()` | `partial_least_squares_step()` | Yes | Fit PLS scores against the resolved target on allowed rows. |
| `sliced_inverse_regression_features()` | `sliced_inverse_regression_step()` | Yes | Fit SIR directions against the resolved target on allowed rows. |
| `variance_selection()` | `{"method": "variance_selection", ...}` | Yes | Select columns by sample variance on allowed rows; no target required. |
| `correlation_selection()` | `{"method": "correlation_selection", ...}` | Yes | Select columns by target correlation on allowed rows. |
| `lasso_selection()` | `{"method": "lasso_selection", ...}` | Yes | Select columns by lasso coefficient magnitude on allowed rows. |
| `lasso_path_selection()` | `{"method": "lasso_path_selection", ...}` | Yes | Select columns by lasso-path inclusion frequency on allowed rows. |
| `rfe_selection()` | `{"method": "rfe_selection", ...}` | Yes | Select columns by recursive feature elimination on allowed rows. |
| `boruta_selection()` | `{"method": "boruta_selection", ...}` | Yes | Select columns by Boruta-style shadow-feature tests on allowed rows. |
| `stability_selection()` | `{"method": "stability_selection", ...}` | Yes | Select columns by sparse-model subsampling frequency on allowed rows. |
| `genetic_selection()` | `{"method": "genetic_selection", ...}` | Yes | Select columns by genetic subset search on allowed rows. |

The remaining helpers remain callable but are intentionally not accepted as
`FeatureSpec` step methods yet:

| Helper | Why not a runner-safe step yet |
| --- | --- |
| `mixed_frequency_lags()` | It changes the date anchor and native-frequency lookup calendar. This belongs with mixed-frequency data/model design, not ordinary same-index feature steps. |
| `hp_filter_features()` | HP filtering is two-sided on the supplied sample. It remains direct-only and warns by default; use `hamilton_step()` for a runner-safe trend/cycle filter. |
| `savitzky_golay_features()` | The smoother uses a centered local window over the supplied sample. It remains direct-only and warns by default; use trailing `rolling_step()` when a past-only smoother is needed. |

`build_features()` remains broader for one-shot construction, including
`feature_specification="F-X-MARX"` and `feature_specification="F-X-MAF"`.
Use it when you want to materialize a complete `FeatureSet` first. Use
`feature_spec(..., steps=...)` when the feature transformations themselves must
be refit inside `forecasting.run()` according to the window design.

## build_features

```python
macroforecast.feature_engineering.build_features(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    predictors: Literal["all"] | Iterable[str] | None = None,
    lags: Iterable[int] | int = (0, 1),
    rolling_windows: Iterable[int] | int | None = None,
    rolling_min_periods: int | None = None,
    add_time: bool = False,
    time_trend: bool = True,
    time_month: bool = False,
    time_quarter: bool = False,
    time_year: bool = False,
    feature_steps: Iterable[Mapping[str, object]] | None = None,
    feature_specification: str | Iterable[str] | None = None,
    include_original: bool = False,
    level_data: feature input | None = None,
    max_lag: int = 12,
    n_factors: int = 8,
    n_maf_components: int = 2,
    feature_fit_policy: str = "expanding",
    feature_min_train_size: int | None = None,
    feature_warn_full_sample: bool = True,
    include_current_factor: bool = True,
    scale_factors: bool = True,
    scale_marx: bool = False,
    scale_maf: bool = False,
    target_transform: str = "level",
    target_mode: str = "direct",
    drop_missing: bool = True,
) -> FeatureSet
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `target`, `targets` | string/iterable or `None` | from `DataSpec`/`PreprocessedData` | Target column choices. One of them is required if the input does not already define targets. |
| `horizon`, `horizons` | positive int/iterable or `None` | from input, then `(1,)` | Forecast horizons. |
| `predictors` | `"all"`, iterable, or `None` | from input, then all non-target columns | Predictor columns. Target columns are rejected as predictors. |
| `lags` | int or iterable | `(0, 1)` | Current value plus lag one by default. `lags=3` means `1, 2, 3`; `lags=0` means current values only; pass exact iterables when needed. |
| `rolling_windows` | positive int/iterable or `None` | `None` | Add rolling-mean features for each window. |
| `rolling_min_periods` | positive int or `None` | window length | Passed to `rolling_mean()`. |
| `add_time` | bool | `False` | Add deterministic date features. |
| `time_trend`, `time_month`, `time_quarter`, `time_year` | bool | `True`, `False`, `False`, `False` | Which date features to include when `add_time=True`. |
| `feature_steps` | iterable of mappings or `None` | `None` | If supplied, use `compose_features()` instead of the simple lag/rolling/time defaults. Mutually exclusive with `feature_specification`. |
| `feature_specification` | string/iterable or `None` | `None` | If supplied, use `feature_matrix()` blocks such as `"F-X-MARX"` or `"F-X-MAF"` instead of the simple lag/rolling/time defaults. |
| `include_original` | bool | `False` | Include original predictors when `feature_steps` is supplied. |
| `level_data` | feature input or `None` | `None` | Passed to `feature_matrix()` when `feature_specification` includes `LEVEL`/`H`. |
| `max_lag` | positive int | `12` | Passed to `feature_matrix()` for `MARX` and `MAF`. |
| `n_factors` | positive int | `8` | Number of `F` factors when `feature_specification` uses `F`. |
| `n_maf_components` | positive int | `2` | MAF components per source variable when `feature_specification` uses `MAF`. |
| `feature_fit_policy` | str | `"expanding"` | Fit policy passed to `feature_matrix()` fitted transforms. |
| `feature_min_train_size` | positive int or `None` | `None` | Minimum complete rows passed to `feature_matrix()` fitted transforms. |
| `feature_warn_full_sample` | bool | `True` | Warn when block-based fitted transforms use `feature_fit_policy="full_sample"`. |
| `include_current_factor` | bool | `True` | Force lag 0 for the `F` block. |
| `scale_factors` | bool | `True` | Scale variables before `F` PCA. |
| `scale_marx` | bool | `False` | Apply optional author R-code lag-matrix scaling for `MARX`. |
| `scale_maf` | bool | `False` | Scale MAF lag panels before PCA. |
| `target_transform` | str | `"level"` | Same choices as `direct_target(transform=...)`. |
| `target_mode` | str | `"direct"` | `"direct"` returns horizon-level target columns. `"path"` returns step-level target columns from `path_targets()`. |
| `drop_missing` | bool | `True` | Drop rows where any selected `X` or `y` column is missing. |

`target_mode="path"` is a target-construction shortcut only. It does not fit
or forecast one model per step; that belongs in the model stage. It also does
not average forecasts; horizon-level forecast averaging belongs in evaluation.
The returned `FeatureSet.y` contains step columns, and metadata records which
step columns belong to each requested horizon.

### Output

Returns `FeatureSet`.

| Field | Type | Meaning |
| --- | --- | --- |
| `X` | `pandas.DataFrame` | Predictor matrix aligned on forecast origin dates. |
| `y` | `pandas.DataFrame` | Direct horizon targets or path step targets aligned to `X`. |
| `metadata` | `dict` | Input metadata plus a `feature_engineering` stage. |
| `feature_metadata` | `pandas.DataFrame` | Generated-feature provenance. Core columns are `feature`, `step`, `block`, `operation`, `source`, `parameter`, `lag`, `window`, `component`, `fit_policy`, `inputs`, and `included`. |
| `target_metadata` | `pandas.DataFrame` | Target-column provenance. Core columns are `target_column`, `source`, `horizon`, `step`, `mode`, `transform`, `operation`, `formula`, `aggregation`, and `used_for_horizons`. |
| `target`, `targets`, `horizons`, `predictors` | scalar/tuple fields | Resolved study choices. |

`FeatureSet` supports tuple unpacking:

```python
X, y, metadata = features
```

### Metadata

`metadata["feature_engineering"]` records:

| Key | Meaning |
| --- | --- |
| `input_panel` | Shape, date range, columns, missing count, and inferred index frequency. |
| `predictors`, `targets`, `horizons` | Resolved study choices. |
| `target_transform` | Target formula choice. |
| `target_mode` | `"direct"` or `"path"`. |
| `path_target_columns_by_horizon` | Step columns to average later when `target_mode="path"`. |
| `lags`, `target_lags`, `rolling_windows`, `rolling_min_periods` | Predictor and autoregressive target-lag construction choices. |
| `feature_specification` | `feature_matrix()` block specification when used. |
| `feature_matrix` | `feature_matrix()` options when block-based features are used. |
| `feature_steps` | Ordered composition steps when `compose_features()` is used through `build_features()`. |
| `time` | Deterministic date-feature choices. |
| `drop_missing` | Whether rows with missing `X` or `y` values were removed. |
| `output` | Final row count, feature count, target count, and sample dates. |

### Feature Metadata

Each feature-producing function attaches `macroforecast_feature_metadata` to
the returned `DataFrame`. `build_features()` exposes the same table as
`FeatureSet.feature_metadata`.

The table is normalized through a single schema helper. The first columns are
always:

| Column | Meaning |
| --- | --- |
| `feature` | Generated feature column name. |
| `step` | Producing step name when created through `compose_features()` or `feature_spec(..., steps=...)`; otherwise empty. |
| `block` | Paper-style block such as `X`, `F`, `MARX`, `MAF`, or `LEVEL` when created by `feature_matrix()`. |
| `operation` | Operation family, for example `lag`, `rolling_mean`, `marx`, `pca`, `pct_change`, `season_dummy`, or `interaction`. |
| `source` | Main source column, source group, or `date` for calendar features. |
| `parameter` | Compact parameter string such as `lag=1`, `window=3`, `component=1`, or `periods=1`. |
| `lag`, `window`, `component` | Parsed numeric fields when the feature name/operation carries them. |
| `fit_policy` | Fitting policy for stateful transforms. In `feature_spec()` this is `fixed_fit_panel` for fit-aware steps. |
| `inputs` | Comma-separated source columns used by the feature. |
| `included` | `True` when the feature is included in final `X`; `False` for intermediate pipeline steps. |

Extra columns are preserved after the standard columns. For example,
`mixed_frequency_lags()` adds source-frequency and lookup-calendar fields.
The metadata frame also carries
`attrs["macroforecast_metadata_schema"] = {"kind": "feature_metadata", "version": 1, ...}`.

```python
features = mf.feature_engineering.build_features(
    processed,
    feature_specification="F-X-MARX",
    lags=range(0, 13),
    max_lag=12,
    n_factors=8,
)

features.feature_metadata.loc[
    features.feature_metadata["feature"] == "MARX__INDPRO_ma3_lag1",
    ["block", "operation", "source", "window", "lag"],
]
```

This records that `MARX__INDPRO_ma3_lag1` came from the `MARX` block, source
series `INDPRO`, window 3, and lag 1. Intermediate `compose_features()` steps
are also recorded; the `included` column marks whether a step output is part of
the final `X` matrix.

### Target Metadata

Target-producing functions attach `macroforecast_target_metadata` to the
returned target frame. `build_features()` exposes the same table as
`FeatureSet.target_metadata`.

```python
features = mf.feature_engineering.build_features(
    processed,
    target="INDPRO",
    horizons=[1, 3, 6],
    target_transform="growth",
)

features.target_metadata.loc[
    features.target_metadata["target_column"] == "INDPRO_growth_h3",
    ["source", "horizon", "mode", "transform", "formula"],
]
```

For direct targets, `horizon` is the forecast horizon and `step` is empty. For
path targets, `step` identifies the future step and `used_for_horizons` records
which requested horizons later consume that step forecast. This keeps the
target construction, model-stage step fitting, and evaluation-stage averaging
separate while preserving the contract in metadata.

### Error Conditions

| Condition | Result |
| --- | --- |
| Input is not a canonical panel-like object | `TypeError`. |
| Target is missing and input has no target metadata | `ValueError`. |
| Target/predictor names are not in the panel | `ValueError`. |
| Predictors include target columns | `ValueError`. |
| Horizons, windows, or min periods are non-positive | `ValueError`. |
| Feature construction leaves no aligned rows | `ValueError`. |
