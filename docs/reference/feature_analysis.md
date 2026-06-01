# macroforecast.feature_analysis

[Back to reference](index.md)

`macroforecast.feature_analysis` inspects feature matrices after
`macroforecast.feature_engineering`. It does not create new predictors and does
not fit a forecasting model. Its job is to make the constructed `X` auditable:
missingness, high correlations, PCA/factor columns, lag/MARX structure, feature
stage changes, and feature-selection stability.

`macroforecast.feature_diagnostic` remains available as a compatibility alias.

Accepted feature inputs are:

| Input | Meaning |
| --- | --- |
| `FeatureSet` | Uses `FeatureSet.X`, `FeatureSet.feature_metadata`, and `FeatureSet.metadata`. |
| `pandas.DataFrame` | Uses the frame as `X`; reads `attrs["macroforecast_feature_metadata"]` when present. |
| `DataBundle`, `DataSpec`, `(DataFrame, metadata)` | Uses the panel as the inspected matrix and carries metadata forward. |

The DataFrame input must satisfy the canonical macroforecast panel contract:
`DatetimeIndex` named `"date"`, sorted, no duplicate dates, numeric columns,
finite values or `NaN`, and non-empty shape.

## Public Flow

```python
import macroforecast as mf

processed = mf.preprocessing.reprocess(data_spec)
features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizons=(1, 3, 6),
    predictors="all",
    lags=(0, 1, 2, 3),
    pca_components=8,
).fit_transform(processed)

diagnostic = mf.feature_analysis.diagnose_features(
    features,
    include_correlation=True,
    include_correlation_matrix=True,
    include_lag_autocorrelation=True,
    selections={"origin_1": ["pc1", "PAYEMS_lag0"]},
    selection_similarity_metric="jaccard",
)
```

## diagnose_features

```python
macroforecast.feature_analysis.diagnose_features(
    data,
    *,
    feature_metadata: pandas.DataFrame | None = None,
    stages: Mapping[str, object] | None = None,
    include_correlation: bool = False,
    include_correlation_matrix: bool = False,
    correlation_method: str = "pearson",
    correlation_threshold: float | None = 0.9,
    correlation_min_periods: int = 3,
    correlation_order: str = "original",
    correlation_scope: str = "all",
    target=None,
    include_target_correlation: bool = False,
    high_missing_threshold: float = 0.5,
    include_factors: bool = True,
    include_factor_variance: bool = True,
    include_factor_loadings: bool = False,
    include_factor_timeseries: bool = False,
    factor_source_data=None,
    include_lags: bool = True,
    include_lag_autocorrelation: bool = False,
    include_lag_correlation_decay: bool = False,
    include_marx: bool = True,
    include_marx_weight_decay: bool = True,
    include_stage_distribution_shift: bool = True,
    selections: Mapping | Sequence | pandas.DataFrame | None = None,
    selection_similarity_metric: str | None = None,
) -> FeatureDiagnosticReport
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `data` | feature input | required | `FeatureSet`, `DataFrame`, `DataBundle`, `DataSpec`, or `(DataFrame, metadata)`. |
| `feature_metadata` | `DataFrame` or `None` | auto | Overrides metadata stored on the input. |
| `stages` | mapping or `None` | `None` | Named feature-like panels to compare in construction order. |
| `include_correlation` | `bool` | `False` | Whether to compute high-correlation feature pairs. |
| `include_correlation_matrix` | `bool` | `False` | Include a full correlation matrix. |
| `correlation_method` | `str` | `"pearson"` | `"pearson"`, `"spearman"`, or `"kendall"`. |
| `correlation_threshold` | float or `None` | `0.9` | Pair filter. Uses absolute correlation when `feature_correlation(..., absolute=True)`. `None` returns all non-missing pairs. |
| `correlation_min_periods` | positive int | `3` | Minimum overlapping observations for correlation. |
| `correlation_order` | `str` | `"original"` | `"original"` or `"clustered"` for the full correlation matrix. |
| `correlation_scope` | `str` | `"all"` | `"all"`, `"within_block"`, or `"cross_block"`. Block comes from feature metadata `block`, then operation/source fallback. |
| `target` | Series, DataFrame, array-like, string, or `None` | `None` | Target used by `include_target_correlation`. A string refers to a column in `data`. |
| `include_target_correlation` | `bool` | `False` | Include feature-to-target correlation rows. |
| `high_missing_threshold` | float | `0.5` | Features with missing-rate above this value are flagged in `overview`. |
| `include_factors` | `bool` | `True` | Include PCA/factor/component diagnostics. |
| `include_factor_variance` | `bool` | `True` | Include scree/cumulative-variance table for detected factor columns. |
| `include_factor_loadings` | `bool` | `False` | Include source-factor correlation loadings. Use `factor_source_data` for original source variables. |
| `include_factor_timeseries` | `bool` | `False` | Include long-form factor-score time series. |
| `include_lags` | `bool` | `True` | Include lag/window diagnostics. |
| `include_lag_autocorrelation` | `bool` | `False` | Include ACF table for detected lag/window columns. |
| `include_lag_correlation_decay` | `bool` | `False` | Include lag-correlation decay against target or lag-0/current source columns. |
| `include_marx` | `bool` | `True` | Include MARX-style moving-average lag diagnostics. |
| `include_marx_weight_decay` | `bool` | `True` | Include equal lag weights implied by MARX moving-average windows. |
| `include_stage_distribution_shift` | `bool` | `True` | When `stages` is supplied, include adjacent-stage distribution-shift diagnostics. |
| `selections` | mapping, sequence, DataFrame, or `None` | `None` | Feature selections by origin/fold/window for stability counts. |
| `selection_similarity_metric` | `str` or `None` | `None` | `"jaccard"` or `"kuncheva"` for pairwise selection similarity. |

### Output

Returns `FeatureDiagnosticReport`.

| Field | Type | Meaning |
| --- | --- | --- |
| `overview` | `dict` | Shape, date range, missingness, zero-variance features, operation/source counts, and feature-metadata coverage. |
| `correlation` | `DataFrame` or `None` | Long-form feature pairs above the requested correlation threshold. |
| `correlation_matrix` | `DataFrame` or `None` | Full correlation matrix, optionally cluster-ordered. |
| `target_correlation` | `DataFrame` or `None` | Feature-to-target correlation rows. |
| `factors` | `DataFrame` or `None` | PCA/factor/component feature diagnostics. |
| `factor_variance` | `DataFrame` or `None` | Scree-style variance and cumulative variance share. |
| `factor_loadings` | `DataFrame` or `None` | Source-factor correlations for loading heatmaps. |
| `factor_timeseries` | `DataFrame` or `None` | Long-form factor/component values by date. |
| `lags` | `DataFrame` or `None` | Lag/window feature diagnostics. |
| `lag_autocorrelation` | `DataFrame` or `None` | ACF/PACF style lag-feature autocorrelation table. |
| `lag_correlation_decay` | `DataFrame` or `None` | Correlation decay by lag/window. |
| `marx` | `DataFrame` or `None` | MARX-style moving-average lag diagnostics. |
| `marx_weight_decay` | `DataFrame` or `None` | Equal lag weights implied by MARX windows. |
| `selection_stability` | `DataFrame` or `None` | Per-feature selection frequency across origins/folds/windows. |
| `selection_similarity` | `DataFrame` or `None` | Pairwise Jaccard or Kuncheva stability across origins/folds/windows. |
| `stage_comparison` | `DataFrame` or `None` | Shape/missingness/column-delta comparison across named feature stages. |
| `stage_distribution_shift` | `DataFrame` or `None` | Adjacent-stage mean, standard-deviation, missingness, and KS-statistic shifts. |
| `metadata` | `dict` | Input metadata plus a compact `feature_analysis` stage. |

`FeatureDiagnosticReport.to_dict()` converts tables to JSON-ready nested
dictionaries/lists.

### Metadata

`diagnose_features(...)` attaches one compact stage:

```python
diagnostic.metadata["feature_analysis"]
```

The stage records:

| Key | Meaning |
| --- | --- |
| `overview` | Compact counts: observations, features, missing cells, high-missing feature count, zero-variance feature count. |
| `options` | Correlation, factor, lag, MARX, selection, and stage-comparison choices. |
| `tables` | Number of rows generated by each diagnostic table. |

Returned diagnostic DataFrames also carry
`attrs["macroforecast_metadata"] == diagnostic.metadata`.

## Helper Functions

### feature_overview

```python
macroforecast.feature_analysis.feature_overview(
    data,
    *,
    feature_metadata: pandas.DataFrame | None = None,
    high_missing_threshold: float = 0.5,
) -> dict
```

Returns one compact dictionary. It is the quickest check for whether the feature
matrix is sparse, constant, or missing feature metadata.

### compare_feature_stages

```python
macroforecast.feature_analysis.compare_feature_stages(
    stages: Mapping[str, object] | None = None,
    **named_stages,
) -> pandas.DataFrame
```

Compares named feature-like panels in order. The table reports observations,
feature counts, missingness, zero-variance counts, and column additions/removals
relative to the previous stage.

Example:

```python
comparison = mf.feature_analysis.compare_feature_stages(
    {
        "base": processed.panel[["PAYEMS", "INDPRO"]],
        "lagged": mf.feature_engineering.lag(processed, columns=["PAYEMS"], lags=(0, 1, 2)),
    }
)
```

### stage_distribution_shift

```python
macroforecast.feature_analysis.stage_distribution_shift(
    stages: Mapping[str, object] | None = None,
    *,
    columns=None,
    min_obs: int = 3,
    **named_stages,
) -> pandas.DataFrame
```

Compares adjacent named stages column by column. Output columns include
`stage_a`, `stage_b`, `feature`, observation counts, means, standard
deviations, `mean_shift`, `sd_ratio`, missing-rate shift, and a two-sample
KS-statistic. Use it to check whether scaling, lag construction, factor
construction, or selection changed feature distributions unexpectedly.

### feature_correlation

```python
macroforecast.feature_analysis.feature_correlation(
    data,
    *,
    feature_metadata: pandas.DataFrame | None = None,
    method: str = "pearson",
    min_periods: int = 3,
    threshold: float | None = 0.9,
    absolute: bool = True,
    max_pairs: int | None = None,
    scope: str = "all",
    block_column: str = "block",
) -> pandas.DataFrame
```

Returns long-form pairs:

| Column | Meaning |
| --- | --- |
| `feature_a`, `feature_b` | Pair names. |
| `correlation`, `abs_correlation` | Signed and absolute correlation. |
| `block_a`, `block_b` | Block labels from feature metadata when available. |
| `operation_a`, `operation_b` | Feature operations from metadata when available. |
| `source_a`, `source_b` | Source columns from metadata when available. |

Use `threshold=None` for a full long-form correlation table.
Use `scope="within_block"` or `scope="cross_block"` to restrict pairs using
metadata blocks.

### feature_target_correlation

```python
macroforecast.feature_analysis.feature_target_correlation(
    data,
    target,
    *,
    feature_metadata=None,
    method: str = "pearson",
    min_periods: int = 3,
    absolute: bool = True,
    max_features: int | None = None,
) -> pandas.DataFrame
```

Returns one row per feature with correlation against the supplied target.
Output columns include `feature`, `target`, `correlation`,
`abs_correlation`, `operation`, `source`, `block`, and `n_obs`.

### feature_correlation_matrix

```python
macroforecast.feature_analysis.feature_correlation_matrix(
    data,
    *,
    method: str = "pearson",
    min_periods: int = 3,
    order: str = "original",
    absolute_distance: bool = True,
) -> pandas.DataFrame
```

Returns a square correlation matrix. `order="clustered"` reorders rows and
columns so highly correlated features are adjacent; this is the callable table
behind a clustered heatmap.

### factor_diagnostics

```python
macroforecast.feature_analysis.factor_diagnostics(
    data,
    *,
    feature_metadata: pandas.DataFrame | None = None,
    operations: Sequence[str] = (...),
    prefixes: Sequence[str] = ("pc", "factor", "maf"),
) -> pandas.DataFrame
```

Detects factor/component features using either feature metadata
(`operation in {"pca", "group_pca", "maf", ...}` or a non-null `component`) or
name prefixes such as `pc1`, `factor1`, and `maf1`.

Returned columns include `feature`, `group`, `operation`, `block`, `source`,
`component`, `n_obs`, `missing_rate`, `mean`, `sd`, `variance`, and
`variance_share`. `variance_share` is a diagnostic share of variance within the
detected factor group. It is not the PCA model's explained-variance ratio unless
the upstream transform recorded that exact quantity.

### factor_variance

```python
macroforecast.feature_analysis.factor_variance(data, *, feature_metadata=None)
```

Returns scree-style rows with `variance_share` and
`cumulative_variance_share`. This is the callable table behind scree and
cumulative-variance views.

### factor_loadings

```python
macroforecast.feature_analysis.factor_loadings(
    data,
    *,
    source_data=None,
    feature_metadata=None,
    method="pearson",
    max_sources=None,
)
```

Approximates factor loadings as correlations between source variables and
factor columns. Supply `source_data` when `data` contains only factor scores.
Returned rows are long-form: `factor`, `source`, `loading`, `abs_loading`.

### factor_timeseries

```python
macroforecast.feature_analysis.factor_timeseries(
    data,
    *,
    feature_metadata=None,
    operations=(...),
    prefixes=("pc", "factor", "maf"),
    max_factors=None,
) -> pandas.DataFrame
```

Returns detected factor/component columns in long time-series form. Output
columns are `date`, `factor`, `value`, `group`, `operation`, `component`, and
`source`. Use this for factor-score line plots or factor stability checks
without reconstructing the feature metadata manually.

### lag_diagnostics

```python
macroforecast.feature_analysis.lag_diagnostics(
    data,
    *,
    feature_metadata: pandas.DataFrame | None = None,
    operations: Sequence[str] = (...),
) -> pandas.DataFrame
```

Detects lag/window features using metadata fields `lag`, `window`,
`operation`, or feature names such as `x_lag3`, `x_roll6_mean`, and
`x_ma4_lag1`.

Returned columns include `feature`, `operation`, `source`, `lag`, `window`,
`n_obs`, `missing_rate`, `first_valid`, and `last_valid`.

### lag_autocorrelation

```python
macroforecast.feature_analysis.lag_autocorrelation(
    data,
    *,
    max_lag: int = 12,
    kind: str = "acf",
) -> pandas.DataFrame
```

Returns ACF or PACF values for detected lag/window feature columns. This is the
callable table behind autocorrelation-per-lag and partial-autocorrelation views.

### lag_correlation_decay

```python
macroforecast.feature_analysis.lag_correlation_decay(
    data,
    *,
    target=None,
    method="pearson",
) -> pandas.DataFrame
```

Returns correlation decay by lag/window. If `target` is supplied, each lag
feature is correlated with that target. Otherwise, each lag feature is compared
with its same-source lag-0/current column when available.

### marx_diagnostics

```python
macroforecast.feature_analysis.marx_diagnostics(
    data,
    *,
    feature_metadata: pandas.DataFrame | None = None,
) -> pandas.DataFrame
```

Detects MARX-style columns named like `x_ma4_lag1`. These are moving-average
lag features, not PCA. The returned table adds `marx_formula`, for example:

```text
mean(x[t-1]...x[t-4])
```

### marx_weight_decay

```python
macroforecast.feature_analysis.marx_weight_decay(
    data,
    *,
    feature_metadata=None,
) -> pandas.DataFrame
```

Returns the equal lag weights implied by each MARX moving-average feature.
For `x_ma4_lag1`, the table has four rows with weight `0.25` for lags 1
through 4 and cumulative weights from `0.25` to `1.0`.

### selection_stability

```python
macroforecast.feature_analysis.selection_stability(
    selections,
    *,
    all_features: Iterable[str] | None = None,
) -> pandas.DataFrame
```

Accepts any of these inputs:

| Input form | Example |
| --- | --- |
| Mapping of origin to selected names | `{"2020-01": ["x1", "x2"], "2020-02": ["x2"]}` |
| Sequence of selected-name iterables | `[["x1"], ["x1", "x3"]]` |
| Indicator DataFrame | rows are origins, columns are features, truthy values mean selected |
| Long DataFrame | columns `feature`, `selected`, and optionally `origin`, `window`, `fold`, or `split` |

The result is indexed by `feature` and includes `selected_count`,
`selection_rate`, `n_origins`, `first_selected_origin`, and
`last_selected_origin`.

### selection_similarity

```python
macroforecast.feature_analysis.selection_similarity(
    selections,
    *,
    metric: str = "jaccard",
    all_features=None,
    n_features=None,
) -> pandas.DataFrame
```

Returns pairwise stability across origins/folds/windows. `metric="jaccard"`
uses overlap divided by union. `metric="kuncheva"` adjusts overlap for expected
random overlap using the declared or inferred feature universe size.

### custom_feature_diagnostic

```python
macroforecast.feature_analysis.custom_feature_diagnostic(
    data,
    func,
    *,
    name=None,
    feature_metadata=None,
    metadata=None,
    **params,
) -> pandas.DataFrame
```

Runs one user diagnostic on a feature matrix or `FeatureSet`. This is for
inspection only; it does not create new predictors.

Callable signature:

```python
func(X, *, feature_metadata=None, metadata=None, **params)
```

Accepted callable outputs are `DataFrame`, `Series`, mapping, or a sequence
convertible to a `DataFrame`. The returned table carries:

| Attr | Meaning |
| --- | --- |
| `macroforecast_metadata_schema.kind` | Always `custom_feature_diagnostic`. |
| `macroforecast_metadata_schema.method` | `name` or callable name. |
| `macroforecast_metadata` | Input metadata plus a `custom_feature_diagnostic` stage. |

Example:

```python
def block_missingness(X, *, feature_metadata=None, metadata=None, block="all"):
    return pd.DataFrame(
        [{"block": block, "missing_rate": float(X.isna().mean().mean())}]
    )

diag = mf.feature_analysis.custom_feature_diagnostic(
    features,
    block_missingness,
    name="block_missingness",
    block="rates",
)
```

## Boundary

| Question | Use |
| --- | --- |
| Create predictors and target matrices | `mf.feature_engineering` |
| Inspect feature matrix quality and metadata | `mf.feature_analysis` |
| Compare raw and preprocessed panels | `mf.data_analysis` |
| Inspect fitted model residuals or tuning trace | `mf.forecast_analysis` |
