# macroforecast.feature_analysis

[Back to reference](index.md)

Feature-stage diagnostics for factors, lags, MARX transforms, selections, and distribution shift.

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `FeatureDiagnosticReport` | class | Container returned by :func:`diagnose_features`. |
| `compare_feature_stages` | function | Compare feature-like panels across named construction stages. |
| `custom_feature_diagnostic` | function | Run a user-supplied feature diagnostic and attach macroforecast metadata. |
| `diagnose_features` | function | Run the standard feature-diagnostic suite on a feature matrix. |
| `effective_window` | function | Return the nonzero source-observation count for each weighted feature date. |
| `factor_diagnostics` | function | Summarize factor/component feature columns. |
| `factor_loadings` | function | Approximate factor loadings as source-column correlations with factors. |
| `factor_timeseries` | function | Return factor/component columns in long time-series form. |
| `factor_variance` | function | Return scree-style variance and cumulative variance share by factor group. |
| `feature_correlation` | function | Return long-form high-correlation feature pairs. |
| `feature_correlation_matrix` | function | Return a full feature-correlation matrix, optionally cluster-ordered. |
| `feature_overview` | function | Return shape, missingness, variance, and metadata coverage for features. |
| `feature_target_correlation` | function | Return feature-to-target correlations. |
| `lag_autocorrelation` | function | Return ACF or PACF values for lag/window feature columns. |
| `lag_correlation_decay` | function | Return correlation decay across lag/window features. |
| `lag_diagnostics` | function | Summarize feature columns that carry lag or window information. |
| `marx_diagnostics` | function | Summarize MARX-style moving-average lag features. |
| `marx_weight_decay` | function | Return implied equal lag weights for MARX moving-average features. |
| `recent_weight_share` | function | Summarize adaptive feature weights into recent lag/lead buckets. |
| `selection_similarity` | function | Return pairwise feature-selection stability across origins/folds/windows. |
| `selection_stability` | function | Return selection frequency by feature across folds, windows, or origins. |
| `stage_distribution_shift` | function | Return distribution-shift diagnostics between adjacent feature stages. |

## Callable And Class Reference

### FeatureDiagnosticReport

Qualified name: `macroforecast.feature_analysis.core.FeatureDiagnosticReport`

#### Signature

```python
macroforecast.feature_analysis.FeatureDiagnosticReport(overview: dict[str, Any], correlation: pd.DataFrame | None = None, correlation_matrix: pd.DataFrame | None = None, target_correlation: pd.DataFrame | None = None, factors: pd.DataFrame | None = None, factor_variance: pd.DataFrame | None = None, factor_loadings: pd.DataFrame | None = None, factor_timeseries: pd.DataFrame | None = None, lags: pd.DataFrame | None = None, lag_autocorrelation: pd.DataFrame | None = None, lag_correlation_decay: pd.DataFrame | None = None, marx: pd.DataFrame | None = None, marx_weight_decay: pd.DataFrame | None = None, selection_stability: pd.DataFrame | None = None, selection_similarity: pd.DataFrame | None = None, stage_comparison: pd.DataFrame | None = None, stage_distribution_shift: pd.DataFrame | None = None, metadata: dict[str, Any] = <factory>) -> None
```

#### Description

Container returned by :func:`diagnose_features`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `overview` | positional or keyword | `dict[str, Any]` | `required` |
| `correlation` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `correlation_matrix` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `target_correlation` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `factors` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `factor_variance` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `factor_loadings` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `factor_timeseries` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `lags` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `lag_autocorrelation` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `lag_correlation_decay` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `marx` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `marx_weight_decay` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `selection_stability` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `selection_similarity` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `stage_comparison` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `stage_distribution_shift` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.feature_analysis.FeatureDiagnosticReport(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `overview` | `dict[str, Any]` | `required` |
| `correlation` | `pd.DataFrame \| None` | `None` |
| `correlation_matrix` | `pd.DataFrame \| None` | `None` |
| `target_correlation` | `pd.DataFrame \| None` | `None` |
| `factors` | `pd.DataFrame \| None` | `None` |
| `factor_variance` | `pd.DataFrame \| None` | `None` |
| `factor_loadings` | `pd.DataFrame \| None` | `None` |
| `factor_timeseries` | `pd.DataFrame \| None` | `None` |
| `lags` | `pd.DataFrame \| None` | `None` |
| `lag_autocorrelation` | `pd.DataFrame \| None` | `None` |
| `lag_correlation_decay` | `pd.DataFrame \| None` | `None` |
| `marx` | `pd.DataFrame \| None` | `None` |
| `marx_weight_decay` | `pd.DataFrame \| None` | `None` |
| `selection_stability` | `pd.DataFrame \| None` | `None` |
| `selection_similarity` | `pd.DataFrame \| None` | `None` |
| `stage_comparison` | `pd.DataFrame \| None` | `None` |
| `stage_distribution_shift` | `pd.DataFrame \| None` | `None` |
| `metadata` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### compare_feature_stages

Qualified name: `macroforecast.feature_analysis.core.compare_feature_stages`

#### Signature

```python
macroforecast.feature_analysis.compare_feature_stages(stages: Mapping[str, Any] | None = None, **named_stages: Any) -> pd.DataFrame
```

#### Description

Compare feature-like panels across named construction stages.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `stages` | positional or keyword | `Mapping[str, Any] \| None` | `None` |
| `named_stages` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.compare_feature_stages(...)
```
### custom_feature_diagnostic

Qualified name: `macroforecast.feature_analysis.core.custom_feature_diagnostic`

#### Signature

```python
macroforecast.feature_analysis.custom_feature_diagnostic(data: Any, func: Callable[..., Any], *, name: str | None = None, feature_metadata: pd.DataFrame | None = None, metadata: Mapping[str, Any] | None = None, **params: Any) -> pd.DataFrame
```

#### Description

Run a user-supplied feature diagnostic and attach macroforecast metadata.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `name` | keyword only | `str \| None` | `None` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.custom_feature_diagnostic(...)
```
### diagnose_features

Qualified name: `macroforecast.feature_analysis.core.diagnose_features`

#### Signature

```python
macroforecast.feature_analysis.diagnose_features(data: Any, *, feature_metadata: pd.DataFrame | None = None, stages: Mapping[str, Any] | None = None, include_correlation: bool = False, include_correlation_matrix: bool = False, correlation_method: CorrelationMethod = "pearson", correlation_threshold: float | None = 0.9, correlation_min_periods: int = 3, correlation_order: CorrelationOrder = "original", correlation_scope: CorrelationScope = "all", target: Any | None = None, include_target_correlation: bool = False, high_missing_threshold: float = 0.5, include_factors: bool = True, include_factor_variance: bool = True, include_factor_loadings: bool = False, include_factor_timeseries: bool = False, factor_source_data: Any | None = None, include_lags: bool = True, include_lag_autocorrelation: bool = False, include_lag_correlation_decay: bool = False, include_marx: bool = True, include_marx_weight_decay: bool = True, include_stage_distribution_shift: bool = True, selections: Mapping[Any, Iterable[str]] | Sequence[Iterable[str]] | pd.DataFrame | None = None, selection_similarity_metric: SelectionSimilarityMetric | None = None) -> FeatureDiagnosticReport
```

#### Description

Run the standard feature-diagnostic suite on a feature matrix.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `stages` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `include_correlation` | keyword only | `bool` | `False` |
| `include_correlation_matrix` | keyword only | `bool` | `False` |
| `correlation_method` | keyword only | `CorrelationMethod` | `"pearson"` |
| `correlation_threshold` | keyword only | `float \| None` | `0.9` |
| `correlation_min_periods` | keyword only | `int` | `3` |
| `correlation_order` | keyword only | `CorrelationOrder` | `"original"` |
| `correlation_scope` | keyword only | `CorrelationScope` | `"all"` |
| `target` | keyword only | `Any \| None` | `None` |
| `include_target_correlation` | keyword only | `bool` | `False` |
| `high_missing_threshold` | keyword only | `float` | `0.5` |
| `include_factors` | keyword only | `bool` | `True` |
| `include_factor_variance` | keyword only | `bool` | `True` |
| `include_factor_loadings` | keyword only | `bool` | `False` |
| `include_factor_timeseries` | keyword only | `bool` | `False` |
| `factor_source_data` | keyword only | `Any \| None` | `None` |
| `include_lags` | keyword only | `bool` | `True` |
| `include_lag_autocorrelation` | keyword only | `bool` | `False` |
| `include_lag_correlation_decay` | keyword only | `bool` | `False` |
| `include_marx` | keyword only | `bool` | `True` |
| `include_marx_weight_decay` | keyword only | `bool` | `True` |
| `include_stage_distribution_shift` | keyword only | `bool` | `True` |
| `selections` | keyword only | `Mapping[Any, Iterable[str]] \| Sequence[Iterable[str]] \| pd.DataFrame \| None` | `None` |
| `selection_similarity_metric` | keyword only | `SelectionSimilarityMetric \| None` | `None` |

#### Returns

`FeatureDiagnosticReport`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.diagnose_features(...)
```
### effective_window

Qualified name: `macroforecast.feature_analysis.core.effective_window`

#### Signature

```python
macroforecast.feature_analysis.effective_window(weights: Any, *, threshold: float = 1e-12) -> pd.Series
```

#### Description

Return the nonzero source-observation count for each weighted feature date.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `Any` | `required` |
| `threshold` | keyword only | `float` | `1e-12` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.effective_window(...)
```
### factor_diagnostics

Qualified name: `macroforecast.feature_analysis.core.factor_diagnostics`

#### Signature

```python
macroforecast.feature_analysis.factor_diagnostics(data: Any, *, feature_metadata: pd.DataFrame | None = None, operations: Sequence[str] = ('pca', 'group_pca', 'maf', 'factor', 'factor_lag', 'scaled_pca', 'supervised_pca', 'supervised_scaled_pca'), prefixes: Sequence[str] = ('pc', 'factor', 'maf')) -> pd.DataFrame
```

#### Description

Summarize factor/component feature columns.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `operations` | keyword only | `Sequence[str]` | `("pca", "group_pca", "maf", "factor", "factor_lag", "scaled_p...` |
| `prefixes` | keyword only | `Sequence[str]` | `("pc", "factor", "maf")` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.factor_diagnostics(...)
```
### factor_loadings

Qualified name: `macroforecast.feature_analysis.core.factor_loadings`

#### Signature

```python
macroforecast.feature_analysis.factor_loadings(data: Any, *, source_data: Any | None = None, feature_metadata: pd.DataFrame | None = None, operations: Sequence[str] = ('pca', 'group_pca', 'maf', 'factor', 'factor_lag', 'scaled_pca', 'supervised_pca', 'supervised_scaled_pca'), prefixes: Sequence[str] = ('pc', 'factor', 'maf'), method: CorrelationMethod = "pearson", max_sources: int | None = None) -> pd.DataFrame
```

#### Description

Approximate factor loadings as source-column correlations with factors.

If `source_data` is supplied, its numeric columns are treated as original
source variables. Otherwise, non-factor columns in `data` are used. This
keeps the callable pandas-native without requiring a fitted PCA object.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `source_data` | keyword only | `Any \| None` | `None` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `operations` | keyword only | `Sequence[str]` | `("pca", "group_pca", "maf", "factor", "factor_lag", "scaled_p...` |
| `prefixes` | keyword only | `Sequence[str]` | `("pc", "factor", "maf")` |
| `method` | keyword only | `CorrelationMethod` | `"pearson"` |
| `max_sources` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.factor_loadings(...)
```
### factor_timeseries

Qualified name: `macroforecast.feature_analysis.core.factor_timeseries`

#### Signature

```python
macroforecast.feature_analysis.factor_timeseries(data: Any, *, feature_metadata: pd.DataFrame | None = None, operations: Sequence[str] = ('pca', 'group_pca', 'maf', 'factor', 'factor_lag', 'scaled_pca', 'supervised_pca', 'supervised_scaled_pca'), prefixes: Sequence[str] = ('pc', 'factor', 'maf'), max_factors: int | None = None) -> pd.DataFrame
```

#### Description

Return factor/component columns in long time-series form.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `operations` | keyword only | `Sequence[str]` | `("pca", "group_pca", "maf", "factor", "factor_lag", "scaled_p...` |
| `prefixes` | keyword only | `Sequence[str]` | `("pc", "factor", "maf")` |
| `max_factors` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.factor_timeseries(...)
```
### factor_variance

Qualified name: `macroforecast.feature_analysis.core.factor_variance`

#### Signature

```python
macroforecast.feature_analysis.factor_variance(data: Any, *, feature_metadata: pd.DataFrame | None = None, operations: Sequence[str] = ('pca', 'group_pca', 'maf', 'factor', 'factor_lag', 'scaled_pca', 'supervised_pca', 'supervised_scaled_pca'), prefixes: Sequence[str] = ('pc', 'factor', 'maf')) -> pd.DataFrame
```

#### Description

Return scree-style variance and cumulative variance share by factor group.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `operations` | keyword only | `Sequence[str]` | `("pca", "group_pca", "maf", "factor", "factor_lag", "scaled_p...` |
| `prefixes` | keyword only | `Sequence[str]` | `("pc", "factor", "maf")` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.factor_variance(...)
```
### feature_correlation

Qualified name: `macroforecast.feature_analysis.core.feature_correlation`

#### Signature

```python
macroforecast.feature_analysis.feature_correlation(data: Any, *, feature_metadata: pd.DataFrame | None = None, method: CorrelationMethod = "pearson", min_periods: int = 3, threshold: float | None = 0.9, absolute: bool = True, max_pairs: int | None = None, scope: CorrelationScope = "all", block_column: str = "block") -> pd.DataFrame
```

#### Description

Return long-form high-correlation feature pairs.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `method` | keyword only | `CorrelationMethod` | `"pearson"` |
| `min_periods` | keyword only | `int` | `3` |
| `threshold` | keyword only | `float \| None` | `0.9` |
| `absolute` | keyword only | `bool` | `True` |
| `max_pairs` | keyword only | `int \| None` | `None` |
| `scope` | keyword only | `CorrelationScope` | `"all"` |
| `block_column` | keyword only | `str` | `"block"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.feature_correlation(...)
```
### feature_correlation_matrix

Qualified name: `macroforecast.feature_analysis.core.feature_correlation_matrix`

#### Signature

```python
macroforecast.feature_analysis.feature_correlation_matrix(data: Any, *, method: CorrelationMethod = "pearson", min_periods: int = 3, order: CorrelationOrder = "original", absolute_distance: bool = True) -> pd.DataFrame
```

#### Description

Return a full feature-correlation matrix, optionally cluster-ordered.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `method` | keyword only | `CorrelationMethod` | `"pearson"` |
| `min_periods` | keyword only | `int` | `3` |
| `order` | keyword only | `CorrelationOrder` | `"original"` |
| `absolute_distance` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.feature_correlation_matrix(...)
```
### feature_overview

Qualified name: `macroforecast.feature_analysis.core.feature_overview`

#### Signature

```python
macroforecast.feature_analysis.feature_overview(data: Any, *, feature_metadata: pd.DataFrame | None = None, high_missing_threshold: float = 0.5) -> dict[str, Any]
```

#### Description

Return shape, missingness, variance, and metadata coverage for features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `high_missing_threshold` | keyword only | `float` | `0.5` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.feature_overview(...)
```
### feature_target_correlation

Qualified name: `macroforecast.feature_analysis.core.feature_target_correlation`

#### Signature

```python
macroforecast.feature_analysis.feature_target_correlation(data: Any, target: Any, *, feature_metadata: pd.DataFrame | None = None, method: CorrelationMethod = "pearson", min_periods: int = 3, absolute: bool = True, max_features: int | None = None) -> pd.DataFrame
```

#### Description

Return feature-to-target correlations.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `target` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `method` | keyword only | `CorrelationMethod` | `"pearson"` |
| `min_periods` | keyword only | `int` | `3` |
| `absolute` | keyword only | `bool` | `True` |
| `max_features` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.feature_target_correlation(...)
```
### lag_autocorrelation

Qualified name: `macroforecast.feature_analysis.core.lag_autocorrelation`

#### Signature

```python
macroforecast.feature_analysis.lag_autocorrelation(data: Any, *, feature_metadata: pd.DataFrame | None = None, columns: Iterable[str] | None = None, max_lag: int = 12, kind: AutocorrelationKind = "acf", operations: Sequence[str] = ('lag', 'mixed_frequency_lag', 'seasonal_lag', 'factor_lag', 'rolling_mean', 'moving_average', 'marx')) -> pd.DataFrame
```

#### Description

Return ACF or PACF values for lag/window feature columns.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `max_lag` | keyword only | `int` | `12` |
| `kind` | keyword only | `AutocorrelationKind` | `"acf"` |
| `operations` | keyword only | `Sequence[str]` | `("lag", "mixed_frequency_lag", "seasonal_lag", "factor_lag", ...` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.lag_autocorrelation(...)
```
### lag_correlation_decay

Qualified name: `macroforecast.feature_analysis.core.lag_correlation_decay`

#### Signature

```python
macroforecast.feature_analysis.lag_correlation_decay(data: Any, *, target: str | pd.Series | None = None, feature_metadata: pd.DataFrame | None = None, operations: Sequence[str] = ('lag', 'mixed_frequency_lag', 'seasonal_lag', 'factor_lag', 'rolling_mean', 'moving_average', 'marx'), method: CorrelationMethod = "pearson") -> pd.DataFrame
```

#### Description

Return correlation decay across lag/window features.

If `target` is supplied, lag features are correlated with that target. If
not, each lag feature is correlated with the same source's lag-0/current
column when one is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `target` | keyword only | `str \| pd.Series \| None` | `None` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `operations` | keyword only | `Sequence[str]` | `("lag", "mixed_frequency_lag", "seasonal_lag", "factor_lag", ...` |
| `method` | keyword only | `CorrelationMethod` | `"pearson"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.lag_correlation_decay(...)
```
### lag_diagnostics

Qualified name: `macroforecast.feature_analysis.core.lag_diagnostics`

#### Signature

```python
macroforecast.feature_analysis.lag_diagnostics(data: Any, *, feature_metadata: pd.DataFrame | None = None, operations: Sequence[str] = ('lag', 'mixed_frequency_lag', 'seasonal_lag', 'factor_lag', 'rolling_mean', 'moving_average', 'marx')) -> pd.DataFrame
```

#### Description

Summarize feature columns that carry lag or window information.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |
| `operations` | keyword only | `Sequence[str]` | `("lag", "mixed_frequency_lag", "seasonal_lag", "factor_lag", ...` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.lag_diagnostics(...)
```
### marx_diagnostics

Qualified name: `macroforecast.feature_analysis.core.marx_diagnostics`

#### Signature

```python
macroforecast.feature_analysis.marx_diagnostics(data: Any, *, feature_metadata: pd.DataFrame | None = None) -> pd.DataFrame
```

#### Description

Summarize MARX-style moving-average lag features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.marx_diagnostics(...)
```
### marx_weight_decay

Qualified name: `macroforecast.feature_analysis.core.marx_weight_decay`

#### Signature

```python
macroforecast.feature_analysis.marx_weight_decay(data: Any, *, feature_metadata: pd.DataFrame | None = None) -> pd.DataFrame
```

#### Description

Return implied equal lag weights for MARX moving-average features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `feature_metadata` | keyword only | `pd.DataFrame \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.marx_weight_decay(...)
```
### recent_weight_share

Qualified name: `macroforecast.feature_analysis.core.recent_weight_share`

#### Signature

```python
macroforecast.feature_analysis.recent_weight_share(weights: Any, *, mode: str = "one_sided") -> pd.DataFrame
```

#### Description

Summarize adaptive feature weights into recent lag/lead buckets.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `Any` | `required` |
| `mode` | keyword only | `str` | `"one_sided"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.recent_weight_share(...)
```
### selection_similarity

Qualified name: `macroforecast.feature_analysis.core.selection_similarity`

#### Signature

```python
macroforecast.feature_analysis.selection_similarity(selections: Mapping[Any, Iterable[str]] | Sequence[Iterable[str]] | pd.DataFrame, *, metric: SelectionSimilarityMetric = "jaccard", all_features: Iterable[str] | None = None, n_features: int | None = None) -> pd.DataFrame
```

#### Description

Return pairwise feature-selection stability across origins/folds/windows.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `selections` | positional or keyword | `Mapping[Any, Iterable[str]] \| Sequence[Iterable[str]] \| pd.DataFrame` | `required` |
| `metric` | keyword only | `SelectionSimilarityMetric` | `"jaccard"` |
| `all_features` | keyword only | `Iterable[str] \| None` | `None` |
| `n_features` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.selection_similarity(...)
```
### selection_stability

Qualified name: `macroforecast.feature_analysis.core.selection_stability`

#### Signature

```python
macroforecast.feature_analysis.selection_stability(selections: Mapping[Any, Iterable[str]] | Sequence[Iterable[str]] | pd.DataFrame, *, all_features: Iterable[str] | None = None) -> pd.DataFrame
```

#### Description

Return selection frequency by feature across folds, windows, or origins.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `selections` | positional or keyword | `Mapping[Any, Iterable[str]] \| Sequence[Iterable[str]] \| pd.DataFrame` | `required` |
| `all_features` | keyword only | `Iterable[str] \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.selection_stability(...)
```
### stage_distribution_shift

Qualified name: `macroforecast.feature_analysis.core.stage_distribution_shift`

#### Signature

```python
macroforecast.feature_analysis.stage_distribution_shift(stages: Mapping[str, Any] | None = None, *, columns: Iterable[str] | None = None, min_obs: int = 3, **named_stages: Any) -> pd.DataFrame
```

#### Description

Return distribution-shift diagnostics between adjacent feature stages.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `stages` | positional or keyword | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `min_obs` | keyword only | `int` | `3` |
| `named_stages` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_analysis.stage_distribution_shift(...)
```
