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
| `transforms.py` | Direct pandas feature transforms: lags, rolling means, scaling, PCA, grouped PCA, MAF, and time features. |
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
| `rolling_mean()` | Rolling-window means. | Fit-based filters or learned smoothers. |
| `moving_average_ladder()` | Multi-scale trailing moving-average block used before optional factor/PCA steps. | PCA/factor extraction itself. |
| `maf_features()` | Moving Average Factors from variable-specific lag panels. | Model fitting or choosing final feature combinations. |
| `feature_matrix()` | Named `X`, `F`, `MARX`, `MAF`, and `LEVEL` feature-matrix combinations. | Loading or preprocessing the raw/level panel. |
| `scale_features()` | Fit-policy-aware z-score, min-max, or robust scaling. | Model fitting. |
| `pca_features()` | Fit-policy-aware PCA factors. | Forecast model fitting. |
| `group_pca()` | PCA factors within named column groups. | FAVAR-specific slow/fast construction, model estimation, or structural identification. |
| `compose_features()` | Sequential combinations such as `pca -> lag`, `lag -> pca`, `maf`, or `moving_average_ladder -> pca -> lag`. | Model fitting or evaluation. |
| `time_features()` | Trend, month, quarter, and year columns. | Holiday calendars. |
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
of `X` (MARX). In `macroforecast`, MARX is not a separate wrapper function. It
is the following explicit call:

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

The original author R snippet builds a VAR lag matrix ordered as lag 1 for all
variables, lag 2 for all variables, and so on. Then each lag-`l` slot for a
variable is replaced by the row average of that variable's lag 1 through lag
`l` columns. The call above matches that unscaled calculation. Through
`feature_matrix(..., specification="MARX", scale_marx=True)`, macroforecast also
supports the optional R-code scaling step: z-score the full lag matrix first
using sample standard deviations, then apply the same increasing-lag averages.

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

### Supervised And Sparse PCA Boundary

Unsupervised group PCA belongs in `feature_engineering` because it only uses
the predictor panel. Supervised PCA uses the target `y`, so it should be fit
inside the model pipeline or a target-aware feature-selection step to avoid
leakage. Sparse PCA is also kept out of the current feature-engineering surface:
it is model-specification-dependent in practice and should be added with model
selection, cross-validation, and reporting rules rather than as a default
pre-model transform.

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
| `MARX` | `moving_average_ladder(data, windows=range(1, max_lag + 1), shift=1)`; with `scale_marx=True`, first z-score the full lag matrix with sample standard deviations and then average lag 1 through lag `l`. |
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
| `method` | One of `"lag"`, `"rolling_mean"`, `"moving_average_ladder"`, `"maf"`, `"scale"`, `"pca"`, `"group_pca"`, `"time"`. |
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
| `lags`, `rolling_windows`, `rolling_min_periods` | Predictor construction choices. |
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
