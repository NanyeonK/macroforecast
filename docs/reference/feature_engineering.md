# macroforecast.feature_engineering

[Back to reference](index.md)

Target construction, lag/MARX/factor transforms, feature-step composition, and feature selection.

Guide context: [../guide/concepts/features.md](../guide/concepts/features.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `FeatureInput` | data | Represent a PEP 604 union type |
| `FeatureSpec` | class | Reusable feature-building callable for forecasting runners. |
| `FeatureSet` | class | Predictor matrix, target matrix, and feature-engineering metadata. |
| `FeatureSelectionResult` | class | Fitted column-selection result shared by direct and runner-safe APIs. |
| `FittedFeatureBuilder` | class | Feature builder fitted on a training window. |
| `adaptive_ma_rf_features` | function | Create adaptive moving-average smoothers using random forests over time. |
| `align_reference_weights` | function | Align component reference weights to a model column order. |
| `asymmetric_trim_features` | function | Sort each row across columns into rank-space features. |
| `average_target` | function | Construct direct average targets over each forecast horizon. |
| `boruta_selection` | function | Select columns using a Boruta-style shadow-feature test. |
| `build_features` | function | Build an aligned predictor matrix and target matrix. |
| `compose_features` | function | Run named feature-engineering steps sequentially. |
| `correlation_selection` | function | Select columns with the largest absolute target correlation. |
| `cumsum_features` | function | Create cumulative-sum features. |
| `custom_features` | function | Apply a user supplied feature-engineering callable to a panel. |
| `custom_step` | function | Return a user-supplied feature step for ``feature_spec``. |
| `dfm_features` | function | Create static dynamic-factor approximation features by standardized PCA. |
| `diff_features` | function | Create difference features. |
| `direct_target` | function | Construct direct-forecast target columns from a canonical panel. |
| `feature_matrix` | function | Build named macro-ML feature-matrix combinations. |
| `feature_spec` | function | Create a reusable feature-building specification. |
| `feature_selection_requires_target` | function | No public docstring is available. |
| `forward_average_target` | function | Construct the forward average target used by assemblage-style models. |
| `fourier_step` | function | Return a reusable Fourier seasonal-term step. |
| `fourier_features` | function | Create deterministic Fourier seasonal terms. |
| `genetic_selection` | function | Select columns using a small genetic subset search. |
| `group_pca` | function | Create PCA factors separately within named column groups. |
| `group_pca_step` | function | Return a reusable grouped-PCA step for ``compose_features``. |
| `hamilton_filter_features` | function | Create Hamilton-filter cycle or trend features. |
| `hamilton_step` | function | Return a reusable Hamilton-filter step for ``compose_features``. |
| `hp_filter_features` | function | Create Hodrick-Prescott filter cycle or trend features. |
| `interaction_features` | function | Create pure interaction terms without powers. |
| `interaction_step` | function | Return a reusable pure-interaction step. |
| `lag` | function | Create lagged predictor columns from a canonical panel. |
| `lag_step` | function | Return a reusable lag step for ``compose_features``. |
| `lags_then_pca` | function | Create lag block first, then PCA on that lag block. |
| `lasso_path_selection` | function | Select columns by lasso-path inclusion frequency. |
| `lasso_selection` | function | Select columns by absolute lasso coefficient magnitude. |
| `marx_step` | function | Return a reusable MARX step for ``compose_features`` or ``feature_spec``. |
| `log_diff_features` | function | Create log-difference features; non-positive values become missing. |
| `log_features` | function | Create log features; non-positive values become missing. |
| `maf_features` | function | Create Moving Average Factors from variable-specific lag panels. |
| `maf_step` | function | Return a reusable MAF step for ``compose_features``. |
| `mixed_frequency_lags` | function | Build exact-date lag blocks for mixed-frequency regressions. |
| `moving_average_changes` | function | Convert one-period component changes to a trailing moving-average unit. |
| `moving_average_ladder` | function | Create a multi-scale trailing moving-average feature block. |
| `moving_average_pca_lags` | function | Create moving-average block, PCA factors, then lags of those factors. |
| `moving_average_step` | function | Return a reusable moving-average-ladder step for ``compose_features``. |
| `normalize_feature_selection_method` | function | No public docstring is available. |
| `nystroem_features` | function | Create Nystroem kernel-approximation features. |
| `nystroem_step` | function | Return a reusable Nystroem kernel-approximation step. |
| `path_targets` | function | Construct one-period target columns for path-average forecasting. |
| `pca_features` | function | Create principal-component features with a declared fit policy. |
| `pca_step` | function | Return a reusable PCA step for ``compose_features``. |
| `pca_then_lags` | function | Create PCA factors and lagged PCA factors in one direct call. |
| `partial_least_squares_features` | function | Create target-aware PLS latent-component scores. |
| `partial_least_squares_step` | function | Return a target-aware PLS step for ``feature_spec``. |
| `pct_change_features` | function | Create simple-growth features. |
| `polynomial_features` | function | Create polynomial expansion features with readable column names. |
| `polynomial_step` | function | Return a reusable polynomial-expansion step. |
| `random_projection_features` | function | Create Gaussian random-projection features. |
| `random_projection_step` | function | Return a reusable Gaussian random-projection step. |
| `rank_space_features` | function | Sort each row into rank-space features for supervised aggregation. |
| `rfe_selection` | function | Select columns by recursive feature elimination. |
| `rolling_mean` | function | Create rolling-mean feature columns from a canonical panel. |
| `rolling_step` | function | Return a reusable rolling-mean step for ``compose_features``. |
| `savitzky_golay_features` | function | Smooth columns with a centered Savitzky-Golay filter. |
| `scale_features` | function | Scale features with either expanding or explicit full-sample fitting. |
| `scale_step` | function | Return a reusable scaling step for ``compose_features``. |
| `season_dummy` | function | Create month or quarter seasonal dummies from the date index. |
| `season_dummy_step` | function | Return a reusable date-index seasonal-dummy step. |
| `seasonal_lag` | function | Create seasonal lag features such as 12-month or 4-quarter lags. |
| `seasonal_lag_step` | function | Return a reusable seasonal-lag step. |
| `select_features` | function | Fit one feature-selection rule on a training matrix. |
| `sliced_inverse_regression_features` | function | Create Sliced Inverse Regression factors from a target signal. |
| `sliced_inverse_regression_step` | function | Return a target-aware SIR step for ``feature_spec``. |
| `sparse_pca_chen_rohe_features` | function | Create Chen-Rohe sparse component analysis factors. |
| `sparse_pca_chen_rohe_step` | function | Return a reusable Chen-Rohe sparse component step. |
| `stability_selection` | function | Select columns by repeated sparse-model subsampling frequency. |
| `time_features` | function | Create deterministic date-index features. |
| `time_step` | function | Return a reusable deterministic trend/month/quarter/year step. |
| `transform_step` | function | Return a reusable deterministic column transform step. |
| `transform_features` | function | Apply a simple column-wise transformation as feature engineering. |
| `varimax_features` | function | Rotate factor-score columns with an orthogonal varimax rotation. |
| `varimax_step` | function | Return a reusable orthogonal varimax-rotation step. |
| `variance_selection` | function | Select columns with the largest sample variance. |
| `wavelet_features` | function | Create rolling multi-resolution approximation/detail features. |
| `weighted_aggregate` | function | Apply aligned component weights to a panel and return one aggregate. |

## Data And Module Values

### `FeatureInput`

Kind: `data`

```python
FeatureInput = macroforecast.preprocessing.types.PreprocessedData | macroforecast.data.panel.DataSpec | macroforecast.data.panel.DataBundle | tuple[pandas.DataFrame, collections.abc.Mapping[str, typing.Any]] | pandas.DataFrame
```

## Callable And Class Reference

### FeatureSpec

Qualified name: `macroforecast.feature_engineering.specs.FeatureSpec`

#### Signature

```python
macroforecast.feature_engineering.FeatureSpec(target: str | None = None, targets: tuple[str, ...] = (), horizon: int | None = None, horizons: tuple[int, ...] = (), predictors: "Literal['all'] | tuple[str, ...] | None" = None, lags: tuple[int, ...] = (0, 1), target_lags: tuple[int, ...] = (), rolling_windows: tuple[int, ...] = (), rolling_min_periods: int | None = None, add_time: bool = False, time_trend: bool = True, time_month: bool = False, time_quarter: bool = False, time_year: bool = False, pca_components: int | None = None, pca_columns: tuple[str, ...] | None = None, pca_scale: bool = True, pca_prefix: str = "pc", feature_steps: tuple[dict[str, Any], ...] = (), include_original: bool = False, target_transform: TargetTransform = "level", target_mode: TargetMode = "direct", drop_missing: bool = True, metadata: dict[str, Any] = <factory>) -> None
```

#### Description

Reusable feature-building callable for forecasting runners.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `target` | positional or keyword | `str \| None` | `None` |
| `targets` | positional or keyword | `tuple[str, ...]` | `()` |
| `horizon` | positional or keyword | `int \| None` | `None` |
| `horizons` | positional or keyword | `tuple[int, ...]` | `()` |
| `predictors` | positional or keyword | `Literal['all'] \| tuple[str, ...] \| None` | `None` |
| `lags` | positional or keyword | `tuple[int, ...]` | `(0, 1)` |
| `target_lags` | positional or keyword | `tuple[int, ...]` | `()` |
| `rolling_windows` | positional or keyword | `tuple[int, ...]` | `()` |
| `rolling_min_periods` | positional or keyword | `int \| None` | `None` |
| `add_time` | positional or keyword | `bool` | `False` |
| `time_trend` | positional or keyword | `bool` | `True` |
| `time_month` | positional or keyword | `bool` | `False` |
| `time_quarter` | positional or keyword | `bool` | `False` |
| `time_year` | positional or keyword | `bool` | `False` |
| `pca_components` | positional or keyword | `int \| None` | `None` |
| `pca_columns` | positional or keyword | `tuple[str, ...] \| None` | `None` |
| `pca_scale` | positional or keyword | `bool` | `True` |
| `pca_prefix` | positional or keyword | `str` | `"pc"` |
| `feature_steps` | positional or keyword | `tuple[dict[str, Any], ...]` | `()` |
| `include_original` | positional or keyword | `bool` | `False` |
| `target_transform` | positional or keyword | `TargetTransform` | `"level"` |
| `target_mode` | positional or keyword | `TargetMode` | `"direct"` |
| `drop_missing` | positional or keyword | `bool` | `True` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.feature_engineering.FeatureSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `target` | `str \| None` | `None` |
| `targets` | `tuple[str, ...]` | `()` |
| `horizon` | `int \| None` | `None` |
| `horizons` | `tuple[int, ...]` | `()` |
| `predictors` | `Literal['all'] \| tuple[str, ...] \| None` | `None` |
| `lags` | `tuple[int, ...]` | `(0, 1)` |
| `target_lags` | `tuple[int, ...]` | `()` |
| `rolling_windows` | `tuple[int, ...]` | `()` |
| `rolling_min_periods` | `int \| None` | `None` |
| `add_time` | `bool` | `False` |
| `time_trend` | `bool` | `True` |
| `time_month` | `bool` | `False` |
| `time_quarter` | `bool` | `False` |
| `time_year` | `bool` | `False` |
| `pca_components` | `int \| None` | `None` |
| `pca_columns` | `tuple[str, ...] \| None` | `None` |
| `pca_scale` | `bool` | `True` |
| `pca_prefix` | `str` | `"pc"` |
| `feature_steps` | `tuple[dict[str, Any], ...]` | `()` |
| `include_original` | `bool` | `False` |
| `target_transform` | `TargetTransform` | `"level"` |
| `target_mode` | `TargetMode` | `"direct"` |
| `drop_missing` | `bool` | `True` |
| `metadata` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, data: FeatureInput, *, metadata: Mapping[str, Any] \| None = None) -> FittedFeatureBuilder` | Fit feature-building state on a training panel. |
| `fit_transform` | `fit_transform(self, data: FeatureInput, *, metadata: Mapping[str, Any] \| None = None) -> FeatureSet` | Fit on ``data`` and return a feature set for the same panel. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return JSON-ready feature choices. |
| `to_metadata` | `to_metadata(self) -> dict[str, Any]` | Return compact metadata for runners. |
### FeatureSet

Qualified name: `macroforecast.feature_engineering.types.FeatureSet`

#### Signature

```python
macroforecast.feature_engineering.FeatureSet(X: pd.DataFrame, y: pd.DataFrame, metadata: dict[str, Any] = <factory>, feature_metadata: pd.DataFrame = <factory>, target_metadata: pd.DataFrame = <factory>, target: str | None = None, targets: tuple[str, ...] = (), horizons: tuple[int, ...] = (), predictors: tuple[str, ...] = ()) -> None
```

#### Description

Predictor matrix, target matrix, and feature-engineering metadata.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `y` | positional or keyword | `pd.DataFrame` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `feature_metadata` | positional or keyword | `pd.DataFrame` | `<factory>` |
| `target_metadata` | positional or keyword | `pd.DataFrame` | `<factory>` |
| `target` | positional or keyword | `str \| None` | `None` |
| `targets` | positional or keyword | `tuple[str, ...]` | `()` |
| `horizons` | positional or keyword | `tuple[int, ...]` | `()` |
| `predictors` | positional or keyword | `tuple[str, ...]` | `()` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.feature_engineering.FeatureSet(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `X` | `pd.DataFrame` | `required` |
| `y` | `pd.DataFrame` | `required` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `feature_metadata` | `pd.DataFrame` | `default_factory` |
| `target_metadata` | `pd.DataFrame` | `default_factory` |
| `target` | `str \| None` | `None` |
| `targets` | `tuple[str, ...]` | `()` |
| `horizons` | `tuple[int, ...]` | `()` |
| `predictors` | `tuple[str, ...]` | `()` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `attach` | `attach(self, stage: str, values: Mapping[str, Any]) -> FeatureSet` | No public docstring is available. |
### FeatureSelectionResult

Qualified name: `macroforecast.feature_engineering.feature_selection.FeatureSelectionResult`

#### Signature

```python
macroforecast.feature_engineering.FeatureSelectionResult(selected_columns: tuple[str, ...], scores: dict[str, float], method: str, n_features: int | float | None, resolved_n_features: int, n_fit_rows: int, fit_policy: str, target_required: bool, metadata: dict[str, Any]) -> None
```

#### Description

Fitted column-selection result shared by direct and runner-safe APIs.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `selected_columns` | positional or keyword | `tuple[str, ...]` | `required` |
| `scores` | positional or keyword | `dict[str, float]` | `required` |
| `method` | positional or keyword | `str` | `required` |
| `n_features` | positional or keyword | `int \| float \| None` | `required` |
| `resolved_n_features` | positional or keyword | `int` | `required` |
| `n_fit_rows` | positional or keyword | `int` | `required` |
| `fit_policy` | positional or keyword | `str` | `required` |
| `target_required` | positional or keyword | `bool` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `required` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.feature_engineering.FeatureSelectionResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `selected_columns` | `tuple[str, ...]` | `required` |
| `scores` | `dict[str, float]` | `required` |
| `method` | `str` | `required` |
| `n_features` | `int \| float \| None` | `required` |
| `resolved_n_features` | `int` | `required` |
| `n_fit_rows` | `int` | `required` |
| `fit_policy` | `str` | `required` |
| `target_required` | `bool` | `required` |
| `metadata` | `dict[str, Any]` | `required` |
### FittedFeatureBuilder

Qualified name: `macroforecast.feature_engineering.specs.FittedFeatureBuilder`

#### Signature

```python
macroforecast.feature_engineering.FittedFeatureBuilder(spec: FeatureSpec, fit_panel: pd.DataFrame, fit_metadata: dict[str, Any], targets: tuple[str, ...], horizons: tuple[int, ...], predictors: tuple[str, ...], pca_state: _PCAState | None = None, step_states: tuple[_FittedFeatureStep, ...] = ()) -> None
```

#### Description

Feature builder fitted on a training window.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `spec` | positional or keyword | `FeatureSpec` | `required` |
| `fit_panel` | positional or keyword | `pd.DataFrame` | `required` |
| `fit_metadata` | positional or keyword | `dict[str, Any]` | `required` |
| `targets` | positional or keyword | `tuple[str, ...]` | `required` |
| `horizons` | positional or keyword | `tuple[int, ...]` | `required` |
| `predictors` | positional or keyword | `tuple[str, ...]` | `required` |
| `pca_state` | positional or keyword | `_PCAState \| None` | `None` |
| `step_states` | positional or keyword | `tuple[_FittedFeatureStep, ...]` | `()` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.feature_engineering.FittedFeatureBuilder(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `spec` | `FeatureSpec` | `required` |
| `fit_panel` | `pd.DataFrame` | `required` |
| `fit_metadata` | `dict[str, Any]` | `required` |
| `targets` | `tuple[str, ...]` | `required` |
| `horizons` | `tuple[int, ...]` | `required` |
| `predictors` | `tuple[str, ...]` | `required` |
| `pca_state` | `_PCAState \| None` | `None` |
| `step_states` | `tuple[_FittedFeatureStep, ...]` | `()` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_metadata` | `to_metadata(self) -> dict[str, Any]` | Return fit metadata for forecasting records. |
| `transform` | `transform(self, data: FeatureInput, *, metadata: Mapping[str, Any] \| None = None, index: Iterable[Any] \| pd.Index \| None = None) -> FeatureSet` | Create model-ready ``X`` and ``y`` from a panel. |
### adaptive_ma_rf_features

Qualified name: `macroforecast.feature_engineering.transforms.adaptive_ma_rf_features`

#### Signature

```python
macroforecast.feature_engineering.adaptive_ma_rf_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_estimators: int = 500, min_samples_leaf: int = 6, sample_fraction: float = 0.6, sided: str = "one", random_state: int | None = 0, drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create adaptive moving-average smoothers using random forests over time.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_estimators` | keyword only | `int` | `500` |
| `min_samples_leaf` | keyword only | `int` | `6` |
| `sample_fraction` | keyword only | `float` | `0.6` |
| `sided` | keyword only | `str` | `"one"` |
| `random_state` | keyword only | `int \| None` | `0` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.adaptive_ma_rf_features(...)
```
### align_reference_weights

Qualified name: `macroforecast.feature_engineering.aggregation.align_reference_weights`

#### Signature

```python
macroforecast.feature_engineering.align_reference_weights(weights: Mapping[str, float] | pd.Series | pd.DataFrame | Sequence[float], columns: Iterable[str], *, normalize: bool = True, fill_value: float = 0.0, name: str = "reference_weight") -> pd.Series
```

#### Description

Align component reference weights to a model column order.

This is the generic version of ``weight.transformation`` in the R
``assemblage`` package. It accepts official basket weights for Albacore,
but can also align sector, state, or survey-item reference weights.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `Mapping[str, float] \| pd.Series \| pd.DataFrame \| Sequence[float]` | `required` |
| `columns` | positional or keyword | `Iterable[str]` | `required` |
| `normalize` | keyword only | `bool` | `True` |
| `fill_value` | keyword only | `float` | `0.0` |
| `name` | keyword only | `str` | `"reference_weight"` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.align_reference_weights(...)
```
### asymmetric_trim_features

Qualified name: `macroforecast.feature_engineering.transforms.asymmetric_trim_features`

#### Signature

```python
macroforecast.feature_engineering.asymmetric_trim_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, prefix: str = "rank_", drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Sort each row across columns into rank-space features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `prefix` | keyword only | `str` | `"rank_"` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.asymmetric_trim_features(...)
```
### average_target

Qualified name: `macroforecast.feature_engineering.targets.average_target`

#### Signature

```python
macroforecast.feature_engineering.average_target(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, target: str | None = None, targets: Iterable[str] | None = None, horizon: int | None = None, horizons: Iterable[int] | int | None = None, transform: PathTransform = "change") -> pd.DataFrame
```

#### Description

Construct direct average targets over each forecast horizon.

For horizon ``h``, this returns the average of one-period target
transformations over steps ``1, ..., h``. It is the target used when one
model is fit directly to an average growth or average difference object.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `targets` | keyword only | `Iterable[str] \| None` | `None` |
| `horizon` | keyword only | `int \| None` | `None` |
| `horizons` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `transform` | keyword only | `PathTransform` | `"change"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.average_target(...)
```
### boruta_selection

Qualified name: `macroforecast.feature_engineering.feature_selection.boruta_selection`

#### Signature

```python
macroforecast.feature_engineering.boruta_selection(data: FeatureInput, target: str | pd.Series | None = None, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_features: int | float = 0.5, n_estimators: int = 100, max_iter: int = 100, alpha: float = 0.05, include_tentative: bool = False, max_depth: int | None = None, min_train_size: int | None = None, random_state: int | None = 0, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Select columns using a Boruta-style shadow-feature test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `target` | positional or keyword | `str \| pd.Series \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_features` | keyword only | `int \| float` | `0.5` |
| `n_estimators` | keyword only | `int` | `100` |
| `max_iter` | keyword only | `int` | `100` |
| `alpha` | keyword only | `float` | `0.05` |
| `include_tentative` | keyword only | `bool` | `False` |
| `max_depth` | keyword only | `int \| None` | `None` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `0` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.boruta_selection(...)
```
### build_features

Qualified name: `macroforecast.feature_engineering.builder.build_features`

#### Signature

```python
macroforecast.feature_engineering.build_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, target: str | None = None, targets: Iterable[str] | None = None, horizon: int | None = None, horizons: Iterable[int] | int | None = None, predictors: "Literal['all'] | Iterable[str] | None" = None, lags: Iterable[int] | int = (0, 1), target_lags: Iterable[int] | int | None = None, rolling_windows: Iterable[int] | int | None = None, rolling_min_periods: int | None = None, add_time: bool = False, time_trend: bool = True, time_month: bool = False, time_quarter: bool = False, time_year: bool = False, feature_steps: Iterable[Mapping[str, Any]] | None = None, feature_specification: str | Iterable[str] | None = None, include_original: bool = False, level_data: FeatureInput | None = None, max_lag: int = 12, n_factors: int = 8, n_maf_components: int = 2, feature_fit_policy: FitPolicy = "expanding", feature_min_train_size: int | None = None, feature_warn_full_sample: bool = True, include_current_factor: bool = True, scale_factors: bool = True, scale_marx: bool = False, scale_maf: bool = False, target_transform: TargetTransform = "level", target_mode: TargetMode = "direct", drop_missing: bool = True) -> FeatureSet
```

#### Description

Build an aligned predictor matrix and target matrix.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `targets` | keyword only | `Iterable[str] \| None` | `None` |
| `horizon` | keyword only | `int \| None` | `None` |
| `horizons` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `predictors` | keyword only | `Literal['all'] \| Iterable[str] \| None` | `None` |
| `lags` | keyword only | `Iterable[int] \| int` | `(0, 1)` |
| `target_lags` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `rolling_windows` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `rolling_min_periods` | keyword only | `int \| None` | `None` |
| `add_time` | keyword only | `bool` | `False` |
| `time_trend` | keyword only | `bool` | `True` |
| `time_month` | keyword only | `bool` | `False` |
| `time_quarter` | keyword only | `bool` | `False` |
| `time_year` | keyword only | `bool` | `False` |
| `feature_steps` | keyword only | `Iterable[Mapping[str, Any]] \| None` | `None` |
| `feature_specification` | keyword only | `str \| Iterable[str] \| None` | `None` |
| `include_original` | keyword only | `bool` | `False` |
| `level_data` | keyword only | `FeatureInput \| None` | `None` |
| `max_lag` | keyword only | `int` | `12` |
| `n_factors` | keyword only | `int` | `8` |
| `n_maf_components` | keyword only | `int` | `2` |
| `feature_fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `feature_min_train_size` | keyword only | `int \| None` | `None` |
| `feature_warn_full_sample` | keyword only | `bool` | `True` |
| `include_current_factor` | keyword only | `bool` | `True` |
| `scale_factors` | keyword only | `bool` | `True` |
| `scale_marx` | keyword only | `bool` | `False` |
| `scale_maf` | keyword only | `bool` | `False` |
| `target_transform` | keyword only | `TargetTransform` | `"level"` |
| `target_mode` | keyword only | `TargetMode` | `"direct"` |
| `drop_missing` | keyword only | `bool` | `True` |

#### Returns

`FeatureSet`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.build_features(...)
```
### compose_features

Qualified name: `macroforecast.feature_engineering.compose.compose_features`

#### Signature

```python
macroforecast.feature_engineering.compose_features(data: FeatureInput, steps: Iterable[Mapping[str, Any]], *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, include_original: bool = False, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Run named feature-engineering steps sequentially.

Each step reads either the original ``panel`` or a prior step via
``input='step_name'``. This keeps the callable API composable while any
future recipe wrapper can call the same functions.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `steps` | positional or keyword | `Iterable[Mapping[str, Any]]` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `include_original` | keyword only | `bool` | `False` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.compose_features(...)
```
### correlation_selection

Qualified name: `macroforecast.feature_engineering.feature_selection.correlation_selection`

#### Signature

```python
macroforecast.feature_engineering.correlation_selection(data: FeatureInput, target: str | pd.Series | None = None, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_features: int | float = 0.5, min_train_size: int | None = None, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Select columns with the largest absolute target correlation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `target` | positional or keyword | `str \| pd.Series \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_features` | keyword only | `int \| float` | `0.5` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.correlation_selection(...)
```
### cumsum_features

Qualified name: `macroforecast.feature_engineering.transforms.cumsum_features`

#### Signature

```python
macroforecast.feature_engineering.cumsum_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create cumulative-sum features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.cumsum_features(...)
```
### custom_features

Qualified name: `macroforecast.feature_engineering.transforms.custom_features`

#### Signature

```python
macroforecast.feature_engineering.custom_features(data: FeatureInput, func: Callable[..., Any], *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, name: str | None = None, **params: Any) -> pd.DataFrame
```

#### Description

Apply a user supplied feature-engineering callable to a panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `name` | keyword only | `str \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.custom_features(...)
```
### custom_step

Qualified name: `macroforecast.feature_engineering.compose.custom_step`

#### Signature

```python
macroforecast.feature_engineering.custom_step(name: str, func: Callable[..., Any] | None = None, *, input: str = "panel", include: bool = True, columns: Iterable[str] | None = None, fit_func: Callable[..., Any] | None = None, transform_func: Callable[..., Any] | None = None, requires_target: bool = False, min_train_size: int | None = None, prefix: str | None = None, drop_missing: bool = False, **params: Any) -> dict[str, Any]
```

#### Description

Return a user-supplied feature step for ``feature_spec``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any] \| None` | `None` |
| `input` | keyword only | `str` | `"panel"` |
| `include` | keyword only | `bool` | `True` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `fit_func` | keyword only | `Callable[..., Any] \| None` | `None` |
| `transform_func` | keyword only | `Callable[..., Any] \| None` | `None` |
| `requires_target` | keyword only | `bool` | `False` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `prefix` | keyword only | `str \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.custom_step(...)
```
### dfm_features

Qualified name: `macroforecast.feature_engineering.transforms.dfm_features`

#### Signature

```python
macroforecast.feature_engineering.dfm_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_factors: int = 3, prefix: str = "dfm", drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create static dynamic-factor approximation features by standardized PCA.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_factors` | keyword only | `int` | `3` |
| `prefix` | keyword only | `str` | `"dfm"` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.dfm_features(...)
```
### diff_features

Qualified name: `macroforecast.feature_engineering.transforms.diff_features`

#### Signature

```python
macroforecast.feature_engineering.diff_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, periods: int = 1, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create difference features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `periods` | keyword only | `int` | `1` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.diff_features(...)
```
### direct_target

Qualified name: `macroforecast.feature_engineering.targets.direct_target`

#### Signature

```python
macroforecast.feature_engineering.direct_target(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, target: str | None = None, targets: Iterable[str] | None = None, horizon: int | None = None, horizons: Iterable[int] | int | None = None, transform: TargetTransform = "level") -> pd.DataFrame
```

#### Description

Construct direct-forecast target columns from a canonical panel.

For date ``t`` and horizon ``h``, ``transform="level"`` or
``transform="value"`` returns
``target[t + h]`` aligned on row ``t``. Other transforms compare
``target[t + h]`` with ``target[t]``. ``average_*`` transforms build direct
average targets from steps ``t + 1`` through ``t + h``; use
``average_value`` when the input series is already a one-period transformed
forecasting object such as monthly growth or monthly difference.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `targets` | keyword only | `Iterable[str] \| None` | `None` |
| `horizon` | keyword only | `int \| None` | `None` |
| `horizons` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `transform` | keyword only | `TargetTransform` | `"level"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.direct_target(...)
```
### feature_matrix

Qualified name: `macroforecast.feature_engineering.matrix.feature_matrix`

#### Signature

```python
macroforecast.feature_engineering.feature_matrix(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, specification: str | Iterable[str] = "X", columns: Iterable[str] | None = None, level_data: FeatureInput | None = None, level_columns: Iterable[str] | None = None, lags: Iterable[int] | int = (0,), max_lag: int = 12, n_factors: int = 8, n_maf_components: int = 2, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, include_current_factor: bool = True, scale_factors: bool = True, scale_marx: bool = False, scale_maf: bool = False, drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Build named macro-ML feature-matrix combinations.

``specification`` accepts combinations such as ``"X"``, ``"F"``,
``"MARX"``, ``"MAF"``, ``"F-X-MARX"``, or ``("F", "X", "MAF")``.
``X`` is the supplied panel, usually after preprocessing. ``LEVEL`` needs
an explicit ``level_data`` panel because levels and stationarized ``X`` are
different data objects in a clean pandas workflow.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `specification` | keyword only | `str \| Iterable[str]` | `"X"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `level_data` | keyword only | `FeatureInput \| None` | `None` |
| `level_columns` | keyword only | `Iterable[str] \| None` | `None` |
| `lags` | keyword only | `Iterable[int] \| int` | `(0,)` |
| `max_lag` | keyword only | `int` | `12` |
| `n_factors` | keyword only | `int` | `8` |
| `n_maf_components` | keyword only | `int` | `2` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `include_current_factor` | keyword only | `bool` | `True` |
| `scale_factors` | keyword only | `bool` | `True` |
| `scale_marx` | keyword only | `bool` | `False` |
| `scale_maf` | keyword only | `bool` | `False` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.feature_matrix(...)
```
### feature_spec

Qualified name: `macroforecast.feature_engineering.specs.feature_spec`

#### Signature

```python
macroforecast.feature_engineering.feature_spec(*, target: str | None = None, targets: Iterable[str] | None = None, horizon: int | None = None, horizons: Iterable[int] | int | None = None, predictors: "Literal['all'] | Iterable[str] | None" = None, lags: Iterable[int] | int | None = (0, 1), target_lags: Iterable[int] | int | None = None, rolling_windows: Iterable[int] | int | None = None, rolling_min_periods: int | None = None, add_time: bool = False, time_trend: bool = True, time_month: bool = False, time_quarter: bool = False, time_year: bool = False, pca_components: int | None = None, pca_columns: Iterable[str] | None = None, pca_scale: bool = True, pca_prefix: str = "pc", steps: Iterable[Mapping[str, Any]] | None = None, feature_steps: Iterable[Mapping[str, Any]] | None = None, include_original: bool = False, target_transform: TargetTransform = "level", target_mode: TargetMode = "direct", drop_missing: bool = True, metadata: Mapping[str, Any] | None = None) -> FeatureSpec
```

#### Description

Create a reusable feature-building specification.

Parameters define the target columns, horizons, predictor columns, simple
lag/rolling/PCA shortcuts, or an explicit ``feature_steps`` pipeline. The
returned spec is inert until a runner calls ``fit(...)`` or
``fit_transform(...)`` on a training panel, so stateful steps such as PCA,
sparse PCA, scaling, and feature selection are fitted inside the training
window rather than on the full sample.

``target``/``targets`` select the source series to forecast.
``horizon``/``horizons`` select direct forecast horizons. ``predictors`` may
be ``"all"``, an iterable of column names, ``None`` for metadata/default
resolution, or an empty iterable for target-only designs. ``lags`` and
``target_lags`` build simple lag matrices when no explicit step pipeline is
supplied. ``steps`` is an alias for ``feature_steps``.

Returns
FeatureSpec
    Frozen feature-builder configuration with ``fit``, ``fit_transform``,
    ``to_dict``, and ``to_metadata`` methods.

Example
>>> import macroforecast as mf
>>> features = mf.feature_engineering.feature_spec(
...     target="INDPRO",
...     predictors=["UNRATE", "CPIAUCSL"],
...     horizons=[1, 3],
...     lags=(0, 1, 2),
... )

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `target` | keyword only | `str \| None` | `None` |
| `targets` | keyword only | `Iterable[str] \| None` | `None` |
| `horizon` | keyword only | `int \| None` | `None` |
| `horizons` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `predictors` | keyword only | `Literal['all'] \| Iterable[str] \| None` | `None` |
| `lags` | keyword only | `Iterable[int] \| int \| None` | `(0, 1)` |
| `target_lags` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `rolling_windows` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `rolling_min_periods` | keyword only | `int \| None` | `None` |
| `add_time` | keyword only | `bool` | `False` |
| `time_trend` | keyword only | `bool` | `True` |
| `time_month` | keyword only | `bool` | `False` |
| `time_quarter` | keyword only | `bool` | `False` |
| `time_year` | keyword only | `bool` | `False` |
| `pca_components` | keyword only | `int \| None` | `None` |
| `pca_columns` | keyword only | `Iterable[str] \| None` | `None` |
| `pca_scale` | keyword only | `bool` | `True` |
| `pca_prefix` | keyword only | `str` | `"pc"` |
| `steps` | keyword only | `Iterable[Mapping[str, Any]] \| None` | `None` |
| `feature_steps` | keyword only | `Iterable[Mapping[str, Any]] \| None` | `None` |
| `include_original` | keyword only | `bool` | `False` |
| `target_transform` | keyword only | `TargetTransform` | `"level"` |
| `target_mode` | keyword only | `TargetMode` | `"direct"` |
| `drop_missing` | keyword only | `bool` | `True` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`FeatureSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.feature_spec(...)
```
### feature_selection_requires_target

Qualified name: `macroforecast.feature_engineering.feature_selection.feature_selection_requires_target`

#### Signature

```python
macroforecast.feature_engineering.feature_selection_requires_target(method: str) -> bool
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `method` | positional or keyword | `str` | `required` |

#### Returns

`bool`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.feature_selection_requires_target(...)
```
### forward_average_target

Qualified name: `macroforecast.feature_engineering.aggregation.forward_average_target`

#### Signature

```python
macroforecast.feature_engineering.forward_average_target(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, target: str | None = None, targets: Iterable[str] | None = None, horizon: int | None = None, horizons: Iterable[int] | int | None = None, transform: PathTransform = "change") -> pd.DataFrame
```

#### Description

Construct the forward average target used by assemblage-style models.

This is a named, reusable wrapper around :func:`average_target`. Its source
cue is Albacore/assemblage, where the target is future average aggregate
inflation. The function itself is generic: it can build a future average
target for any aggregate macro series.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `targets` | keyword only | `Iterable[str] \| None` | `None` |
| `horizon` | keyword only | `int \| None` | `None` |
| `horizons` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `transform` | keyword only | `PathTransform` | `"change"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.forward_average_target(...)
```
### fourier_step

Qualified name: `macroforecast.feature_engineering.compose.fourier_step`

#### Signature

```python
macroforecast.feature_engineering.fourier_step(*, name: str = "fourier", input: str = "panel", period: int = 12, order: int = 2, prefix: str = "fourier", include: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable Fourier seasonal-term step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"fourier"` |
| `input` | keyword only | `str` | `"panel"` |
| `period` | keyword only | `int` | `12` |
| `order` | keyword only | `int` | `2` |
| `prefix` | keyword only | `str` | `"fourier"` |
| `include` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.fourier_step(...)
```
### fourier_features

Qualified name: `macroforecast.feature_engineering.transforms.fourier_features`

#### Signature

```python
macroforecast.feature_engineering.fourier_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, period: int = 12, order: int = 2, prefix: str = "fourier") -> pd.DataFrame
```

#### Description

Create deterministic Fourier seasonal terms.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `period` | keyword only | `int` | `12` |
| `order` | keyword only | `int` | `2` |
| `prefix` | keyword only | `str` | `"fourier"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.fourier_features(...)
```
### genetic_selection

Qualified name: `macroforecast.feature_engineering.feature_selection.genetic_selection`

#### Signature

```python
macroforecast.feature_engineering.genetic_selection(data: FeatureInput, target: str | pd.Series | None = None, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_features: int | float = 0.5, population_size: int = 30, n_generations: int = 50, crossover_prob: float = 0.8, mutation_prob: float | None = None, fitness_estimator: str = "ridge", cv_folds: int = 3, min_train_size: int | None = None, random_state: int | None = 0, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Select columns using a small genetic subset search.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `target` | positional or keyword | `str \| pd.Series \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_features` | keyword only | `int \| float` | `0.5` |
| `population_size` | keyword only | `int` | `30` |
| `n_generations` | keyword only | `int` | `50` |
| `crossover_prob` | keyword only | `float` | `0.8` |
| `mutation_prob` | keyword only | `float \| None` | `None` |
| `fitness_estimator` | keyword only | `str` | `"ridge"` |
| `cv_folds` | keyword only | `int` | `3` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `0` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.genetic_selection(...)
```
### group_pca

Qualified name: `macroforecast.feature_engineering.transforms.group_pca`

#### Signature

```python
macroforecast.feature_engineering.group_pca(data: FeatureInput, *, groups: Mapping[str, Iterable[str]], metadata: Mapping[str, Any] | None = None, n_components: int | Mapping[str, int] = 1, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, scale: bool = True, prefix: str | None = None, drop_missing: bool = False, random_state: int | None = None, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create PCA factors separately within named column groups.

This is useful when factors should be extracted from economically
meaningful blocks rather than from the whole predictor panel at once.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `groups` | keyword only | `Mapping[str, Iterable[str]]` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `n_components` | keyword only | `int \| Mapping[str, int]` | `1` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `scale` | keyword only | `bool` | `True` |
| `prefix` | keyword only | `str \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int \| None` | `None` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.group_pca(...)
```
### group_pca_step

Qualified name: `macroforecast.feature_engineering.compose.group_pca_step`

#### Signature

```python
macroforecast.feature_engineering.group_pca_step(*, groups: Mapping[str, Iterable[str]], name: str = "group_pca", input: str = "panel", n_components: int | Mapping[str, int] = 1, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, scale: bool = True, prefix: str | None = None, include: bool = True, drop_missing: bool = False, random_state: int | None = None, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable grouped-PCA step for ``compose_features``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `groups` | keyword only | `Mapping[str, Iterable[str]]` | `required` |
| `name` | keyword only | `str` | `"group_pca"` |
| `input` | keyword only | `str` | `"panel"` |
| `n_components` | keyword only | `int \| Mapping[str, int]` | `1` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `scale` | keyword only | `bool` | `True` |
| `prefix` | keyword only | `str \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int \| None` | `None` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.group_pca_step(...)
```
### hamilton_filter_features

Qualified name: `macroforecast.feature_engineering.transforms.hamilton_filter_features`

#### Signature

```python
macroforecast.feature_engineering.hamilton_filter_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, h: int = 8, p: int = 4, component: str = "cycle", fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, missing: str = "drop", drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create Hamilton-filter cycle or trend features.

The Hamilton filter regresses ``y[t+h]`` on a constant and
``y[t], y[t-1], ..., y[t-p+1]``. The trend is the fitted value and the
cycle is the residual, both labeled at ``t+h``. Defaults ``h=8, p=4``
match the common quarterly specification. For monthly data, callers often
use ``h=24, p=12``.

``fit_policy='expanding'`` estimates each row with only earlier completed
target rows, which avoids full-sample leakage when the output is used as a
forecasting feature. ``fit_policy='full_sample'`` reproduces the usual
in-sample filter style and emits a warning by default.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `h` | keyword only | `int` | `8` |
| `p` | keyword only | `int` | `4` |
| `component` | keyword only | `str` | `"cycle"` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `missing` | keyword only | `str` | `"drop"` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.hamilton_filter_features(...)
```
### hamilton_step

Qualified name: `macroforecast.feature_engineering.compose.hamilton_step`

#### Signature

```python
macroforecast.feature_engineering.hamilton_step(*, name: str = "hamilton", input: str = "panel", columns: Iterable[str] | None = None, h: int = 8, p: int = 4, component: str = "cycle", fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, missing: str = "drop", include: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable Hamilton-filter step for ``compose_features``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"hamilton"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `h` | keyword only | `int` | `8` |
| `p` | keyword only | `int` | `4` |
| `component` | keyword only | `str` | `"cycle"` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `missing` | keyword only | `str` | `"drop"` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.hamilton_step(...)
```
### hp_filter_features

Qualified name: `macroforecast.feature_engineering.transforms.hp_filter_features`

#### Signature

```python
macroforecast.feature_engineering.hp_filter_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, lamb: float = 129600.0, component: str = "cycle", drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create Hodrick-Prescott filter cycle or trend features.

HP filtering is two-sided on the supplied sample. It is direct-only and
warns by default because using it before a forecasting split can leak future
information into trend/cycle features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `lamb` | keyword only | `float` | `129600.0` |
| `component` | keyword only | `str` | `"cycle"` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.hp_filter_features(...)
```
### interaction_features

Qualified name: `macroforecast.feature_engineering.transforms.interaction_features`

#### Signature

```python
macroforecast.feature_engineering.interaction_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, order: int = 2, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create pure interaction terms without powers.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `order` | keyword only | `int` | `2` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.interaction_features(...)
```
### interaction_step

Qualified name: `macroforecast.feature_engineering.compose.interaction_step`

#### Signature

```python
macroforecast.feature_engineering.interaction_step(*, name: str = "interaction", input: str = "panel", columns: Iterable[str] | None = None, order: int = 2, include: bool = True, drop_missing: bool = False) -> dict[str, Any]
```

#### Description

Return a reusable pure-interaction step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"interaction"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `order` | keyword only | `int` | `2` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.interaction_step(...)
```
### lag

Qualified name: `macroforecast.feature_engineering.transforms.lag`

#### Signature

```python
macroforecast.feature_engineering.lag(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, lags: Iterable[int] | int = (1,), drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create lagged predictor columns from a canonical panel.

``lags=3`` means lags ``1, 2, 3``. Pass an iterable such as ``(0, 1, 3)``
when the current value and specific lag lengths should be included.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `lags` | keyword only | `Iterable[int] \| int` | `(1,)` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.lag(...)
```
### lag_step

Qualified name: `macroforecast.feature_engineering.compose.lag_step`

#### Signature

```python
macroforecast.feature_engineering.lag_step(*, name: str = "lag", input: str = "panel", columns: Iterable[str] | None = None, lags: Iterable[int] | int = (1,), include: bool = True, drop_missing: bool = False) -> dict[str, Any]
```

#### Description

Return a reusable lag step for ``compose_features``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"lag"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `lags` | keyword only | `Iterable[int] \| int` | `(1,)` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.lag_step(...)
```
### lags_then_pca

Qualified name: `macroforecast.feature_engineering.compose.lags_then_pca`

#### Signature

```python
macroforecast.feature_engineering.lags_then_pca(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, lags: Iterable[int] | int = (0, 1), n_components: int = 1, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, scale: bool = True, prefix: str = "lag_pc", include_lags: bool = False, drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create lag block first, then PCA on that lag block.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `lags` | keyword only | `Iterable[int] \| int` | `(0, 1)` |
| `n_components` | keyword only | `int` | `1` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `scale` | keyword only | `bool` | `True` |
| `prefix` | keyword only | `str` | `"lag_pc"` |
| `include_lags` | keyword only | `bool` | `False` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.lags_then_pca(...)
```
### lasso_path_selection

Qualified name: `macroforecast.feature_engineering.feature_selection.lasso_path_selection`

#### Signature

```python
macroforecast.feature_engineering.lasso_path_selection(data: FeatureInput, target: str | pd.Series | None = None, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_features: int | float = 0.5, eps: float = 0.001, n_alphas: int = 100, normalize_features: bool = True, positive: bool = False, min_train_size: int | None = None, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Select columns by lasso-path inclusion frequency.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `target` | positional or keyword | `str \| pd.Series \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_features` | keyword only | `int \| float` | `0.5` |
| `eps` | keyword only | `float` | `0.001` |
| `n_alphas` | keyword only | `int` | `100` |
| `normalize_features` | keyword only | `bool` | `True` |
| `positive` | keyword only | `bool` | `False` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.lasso_path_selection(...)
```
### lasso_selection

Qualified name: `macroforecast.feature_engineering.feature_selection.lasso_selection`

#### Signature

```python
macroforecast.feature_engineering.lasso_selection(data: FeatureInput, target: str | pd.Series | None = None, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_features: int | float = 0.5, alpha: float = 0.001, min_train_size: int | None = None, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Select columns by absolute lasso coefficient magnitude.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `target` | positional or keyword | `str \| pd.Series \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_features` | keyword only | `int \| float` | `0.5` |
| `alpha` | keyword only | `float` | `0.001` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.lasso_selection(...)
```
### marx_step

Qualified name: `macroforecast.feature_engineering.compose.marx_step`

#### Signature

```python
macroforecast.feature_engineering.marx_step(*, name: str = "marx", input: str = "panel", columns: Iterable[str] | None = None, max_lag: int = 12, scale_lags: bool = False, min_train_size: int | None = None, include: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable MARX step for ``compose_features`` or ``feature_spec``.

MARX uses increasing averages of lagged predictors. With
``scale_lags=True``, lag-matrix columns are z-scored before averaging,
matching the author R-code variant.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"marx"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `max_lag` | keyword only | `int` | `12` |
| `scale_lags` | keyword only | `bool` | `False` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.marx_step(...)
```
### log_diff_features

Qualified name: `macroforecast.feature_engineering.transforms.log_diff_features`

#### Signature

```python
macroforecast.feature_engineering.log_diff_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, periods: int = 1, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create log-difference features; non-positive values become missing.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `periods` | keyword only | `int` | `1` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.log_diff_features(...)
```
### log_features

Qualified name: `macroforecast.feature_engineering.transforms.log_features`

#### Signature

```python
macroforecast.feature_engineering.log_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create log features; non-positive values become missing.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.log_features(...)
```
### maf_features

Qualified name: `macroforecast.feature_engineering.transforms.maf_features`

#### Signature

```python
macroforecast.feature_engineering.maf_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, max_lag: int = 12, lags: Iterable[int] | None = None, n_components: int = 2, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, scale: bool = False, prefix: str = "maf", drop_missing: bool = False, random_state: int | None = None, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create Moving Average Factors from variable-specific lag panels.

For each selected variable ``x_k``, this builds
``[x_k, L x_k, ..., L^P x_k]`` and extracts PCA components from that
variable-specific lag panel. This is the MAF construction from
macro-ML forecasting papers; it is not global PCA over all variables.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `max_lag` | keyword only | `int` | `12` |
| `lags` | keyword only | `Iterable[int] \| None` | `None` |
| `n_components` | keyword only | `int` | `2` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `scale` | keyword only | `bool` | `False` |
| `prefix` | keyword only | `str` | `"maf"` |
| `drop_missing` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int \| None` | `None` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.maf_features(...)
```
### maf_step

Qualified name: `macroforecast.feature_engineering.compose.maf_step`

#### Signature

```python
macroforecast.feature_engineering.maf_step(*, name: str = "maf", input: str = "panel", columns: Iterable[str] | None = None, max_lag: int = 12, lags: Iterable[int] | None = None, n_components: int = 2, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, scale: bool = False, prefix: str | None = None, include: bool = True, drop_missing: bool = False, random_state: int | None = None, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable MAF step for ``compose_features``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"maf"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `max_lag` | keyword only | `int` | `12` |
| `lags` | keyword only | `Iterable[int] \| None` | `None` |
| `n_components` | keyword only | `int` | `2` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `scale` | keyword only | `bool` | `False` |
| `prefix` | keyword only | `str \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int \| None` | `None` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.maf_step(...)
```
### mixed_frequency_lags

Qualified name: `macroforecast.feature_engineering.transforms.mixed_frequency_lags`

#### Signature

```python
macroforecast.feature_engineering.mixed_frequency_lags(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, target: str | None = None, anchor_dates: Iterable[Any] | None = None, columns: Iterable[str] | None = None, lags: Iterable[int] | int = (0, 1, 2), frequency_by_column: Mapping[str, str] | None = None, target_frequency: str | None = None, anchor_position: str = "date", drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Build exact-date lag blocks for mixed-frequency regressions.

Lags are measured in each source column's native frequency. For example,
monthly predictors with ``lags=(0, 1, 2)`` produce the current, previous,
and two-month-lag values at each target anchor date. Quarterly predictors
with the same lags use quarter steps.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `anchor_dates` | keyword only | `Iterable[Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `lags` | keyword only | `Iterable[int] \| int` | `(0, 1, 2)` |
| `frequency_by_column` | keyword only | `Mapping[str, str] \| None` | `None` |
| `target_frequency` | keyword only | `str \| None` | `None` |
| `anchor_position` | keyword only | `str` | `"date"` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.mixed_frequency_lags(...)
```
### moving_average_changes

Qualified name: `macroforecast.feature_engineering.aggregation.moving_average_changes`

#### Signature

```python
macroforecast.feature_engineering.moving_average_changes(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, window: int = 3, method: MovingAverageMethod = "compound_percent", suffix: str | None = None, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Convert one-period component changes to a trailing moving-average unit.

``method="compound_percent"`` follows the R ``assemblage``
``x.transformation`` convention for month-over-month percentage changes:
``prod(1 + x / 100) - 1``, returned in percent units. Other methods are
provided so the same helper can be reused outside inflation applications.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `window` | keyword only | `int` | `3` |
| `method` | keyword only | `MovingAverageMethod` | `"compound_percent"` |
| `suffix` | keyword only | `str \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.moving_average_changes(...)
```
### moving_average_ladder

Qualified name: `macroforecast.feature_engineering.transforms.moving_average_ladder`

#### Signature

```python
macroforecast.feature_engineering.moving_average_ladder(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, windows: Iterable[int] | None = None, max_window: int = 12, min_periods: int | None = None, shift: int = 0, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create a multi-scale trailing moving-average feature block.

This function is the moving-average ladder used by MARX-style macro-ML
feature pipelines. In this package, the paper notation ``marx_features(P)``
is expressed as ``moving_average_ladder(..., windows=range(1, P + 1),
shift=1)``: increasing-order moving averages of lagged ``X``. With the
default ``max_window=12`` it builds windows ``1, 2, 4, 8``. It does not run
PCA. Moving-average PCA is a separate composition: first call
``moving_average_ladder(...)`` to create the stacked moving-average block,
then apply a fit-aware PCA/factor step.

The default ladder uses powers of two because those windows give a compact
short/medium/long persistence basis without manually choosing every lag.
Pass ``windows=...`` for an exact window set such as ``(1, 2, 4, 8, 12)``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `windows` | keyword only | `Iterable[int] \| None` | `None` |
| `max_window` | keyword only | `int` | `12` |
| `min_periods` | keyword only | `int \| None` | `None` |
| `shift` | keyword only | `int` | `0` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.moving_average_ladder(...)
```
### moving_average_pca_lags

Qualified name: `macroforecast.feature_engineering.compose.moving_average_pca_lags`

#### Signature

```python
macroforecast.feature_engineering.moving_average_pca_lags(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, windows: Iterable[int] | None = None, max_window: int = 12, n_components: int = 1, lags: Iterable[int] | int = (1,), fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, scale: bool = True, prefix: str = "ma_pc", include_pca: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create moving-average block, PCA factors, then lags of those factors.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `windows` | keyword only | `Iterable[int] \| None` | `None` |
| `max_window` | keyword only | `int` | `12` |
| `n_components` | keyword only | `int` | `1` |
| `lags` | keyword only | `Iterable[int] \| int` | `(1,)` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `scale` | keyword only | `bool` | `True` |
| `prefix` | keyword only | `str` | `"ma_pc"` |
| `include_pca` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.moving_average_pca_lags(...)
```
### moving_average_step

Qualified name: `macroforecast.feature_engineering.compose.moving_average_step`

#### Signature

```python
macroforecast.feature_engineering.moving_average_step(*, name: str = "ma", input: str = "panel", columns: Iterable[str] | None = None, windows: Iterable[int] | None = None, max_window: int = 12, min_periods: int | None = None, shift: int = 0, include: bool = True, drop_missing: bool = False) -> dict[str, Any]
```

#### Description

Return a reusable moving-average-ladder step for ``compose_features``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"ma"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `windows` | keyword only | `Iterable[int] \| None` | `None` |
| `max_window` | keyword only | `int` | `12` |
| `min_periods` | keyword only | `int \| None` | `None` |
| `shift` | keyword only | `int` | `0` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.moving_average_step(...)
```
### normalize_feature_selection_method

Qualified name: `macroforecast.feature_engineering.feature_selection.normalize_feature_selection_method`

#### Signature

```python
macroforecast.feature_engineering.normalize_feature_selection_method(value: str) -> str
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `str` | `required` |

#### Returns

`str`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.normalize_feature_selection_method(...)
```
### nystroem_features

Qualified name: `macroforecast.feature_engineering.transforms.nystroem_features`

#### Signature

```python
macroforecast.feature_engineering.nystroem_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_components: int = 10, kernel: str = "rbf", gamma: float | None = None, random_state: int | None = None, prefix: str = "nys", drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create Nystroem kernel-approximation features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `10` |
| `kernel` | keyword only | `str` | `"rbf"` |
| `gamma` | keyword only | `float \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `None` |
| `prefix` | keyword only | `str` | `"nys"` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.nystroem_features(...)
```
### nystroem_step

Qualified name: `macroforecast.feature_engineering.compose.nystroem_step`

#### Signature

```python
macroforecast.feature_engineering.nystroem_step(*, name: str = "nys", input: str = "panel", columns: Iterable[str] | None = None, n_components: int = 10, kernel: str = "rbf", gamma: float | None = None, random_state: int | None = None, prefix: str | None = None, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, include: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable Nystroem kernel-approximation step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"nys"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `10` |
| `kernel` | keyword only | `str` | `"rbf"` |
| `gamma` | keyword only | `float \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `None` |
| `prefix` | keyword only | `str \| None` | `None` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.nystroem_step(...)
```
### path_targets

Qualified name: `macroforecast.feature_engineering.targets.path_targets`

#### Signature

```python
macroforecast.feature_engineering.path_targets(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, target: str | None = None, targets: Iterable[str] | None = None, horizon: int | None = None, horizons: Iterable[int] | int | None = None, transform: PathTransform = "change") -> pd.DataFrame
```

#### Description

Construct one-period target columns for path-average forecasting.

A path-average workflow fits and forecasts one model per future step; a
later evaluation stage can average the step forecasts. This function
creates the ``step1`` through ``stepH`` target columns required by the
model stage. Use ``transform="value"`` when the input series is already a
one-period transformed forecasting object.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `targets` | keyword only | `Iterable[str] \| None` | `None` |
| `horizon` | keyword only | `int \| None` | `None` |
| `horizons` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `transform` | keyword only | `PathTransform` | `"change"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.path_targets(...)
```
### pca_features

Qualified name: `macroforecast.feature_engineering.transforms.pca_features`

#### Signature

```python
macroforecast.feature_engineering.pca_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_components: int = 1, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, scale: bool = True, prefix: str = "pc", drop_missing: bool = False, random_state: int | None = None, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create principal-component features with a declared fit policy.

PCA is a fitted transformation, so the default is expanding-window fitting.
Use ``fit_policy='full_sample'`` only for exploratory analysis or after an
external split has already made the input sample training-only.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `1` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `scale` | keyword only | `bool` | `True` |
| `prefix` | keyword only | `str` | `"pc"` |
| `drop_missing` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int \| None` | `None` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.pca_features(...)
```
### pca_step

Qualified name: `macroforecast.feature_engineering.compose.pca_step`

#### Signature

```python
macroforecast.feature_engineering.pca_step(*, name: str = "pc", input: str = "panel", columns: Iterable[str] | None = None, n_components: int = 1, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, scale: bool = True, prefix: str | None = None, include: bool = True, drop_missing: bool = False, random_state: int | None = None, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable PCA step for ``compose_features``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"pc"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `1` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `scale` | keyword only | `bool` | `True` |
| `prefix` | keyword only | `str \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int \| None` | `None` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.pca_step(...)
```
### pca_then_lags

Qualified name: `macroforecast.feature_engineering.compose.pca_then_lags`

#### Signature

```python
macroforecast.feature_engineering.pca_then_lags(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_components: int = 1, lags: Iterable[int] | int = (1,), fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, scale: bool = True, prefix: str = "pc", include_pca: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create PCA factors and lagged PCA factors in one direct call.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `1` |
| `lags` | keyword only | `Iterable[int] \| int` | `(1,)` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `scale` | keyword only | `bool` | `True` |
| `prefix` | keyword only | `str` | `"pc"` |
| `include_pca` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.pca_then_lags(...)
```
### partial_least_squares_features

Qualified name: `macroforecast.feature_engineering.transforms.partial_least_squares_features`

#### Signature

```python
macroforecast.feature_engineering.partial_least_squares_features(data: FeatureInput, target: str | pd.Series | None = None, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_components: int = 3, prefix: str = "pls", drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create target-aware PLS latent-component scores.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `target` | positional or keyword | `str \| pd.Series \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `3` |
| `prefix` | keyword only | `str` | `"pls"` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.partial_least_squares_features(...)
```
### partial_least_squares_step

Qualified name: `macroforecast.feature_engineering.compose.partial_least_squares_step`

#### Signature

```python
macroforecast.feature_engineering.partial_least_squares_step(*, name: str = "pls", input: str = "panel", columns: Iterable[str] | None = None, n_components: int = 3, min_train_size: int | None = None, prefix: str | None = None, include: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a target-aware PLS step for ``feature_spec``.

``compose_features`` has no target contract; use the direct
``partial_least_squares_features`` callable for full-sample manual use.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"pls"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `3` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `prefix` | keyword only | `str \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.partial_least_squares_step(...)
```
### pct_change_features

Qualified name: `macroforecast.feature_engineering.transforms.pct_change_features`

#### Signature

```python
macroforecast.feature_engineering.pct_change_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, periods: int = 1, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create simple-growth features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `periods` | keyword only | `int` | `1` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.pct_change_features(...)
```
### polynomial_features

Qualified name: `macroforecast.feature_engineering.transforms.polynomial_features`

#### Signature

```python
macroforecast.feature_engineering.polynomial_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, degree: int = 2, include_bias: bool = False, interaction_only: bool = False, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create polynomial expansion features with readable column names.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `degree` | keyword only | `int` | `2` |
| `include_bias` | keyword only | `bool` | `False` |
| `interaction_only` | keyword only | `bool` | `False` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.polynomial_features(...)
```
### polynomial_step

Qualified name: `macroforecast.feature_engineering.compose.polynomial_step`

#### Signature

```python
macroforecast.feature_engineering.polynomial_step(*, name: str = "polynomial", input: str = "panel", columns: Iterable[str] | None = None, degree: int = 2, include_bias: bool = False, interaction_only: bool = False, include: bool = True, drop_missing: bool = False) -> dict[str, Any]
```

#### Description

Return a reusable polynomial-expansion step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"polynomial"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `degree` | keyword only | `int` | `2` |
| `include_bias` | keyword only | `bool` | `False` |
| `interaction_only` | keyword only | `bool` | `False` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.polynomial_step(...)
```
### random_projection_features

Qualified name: `macroforecast.feature_engineering.transforms.random_projection_features`

#### Signature

```python
macroforecast.feature_engineering.random_projection_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_components: int = 2, random_state: int | None = None, prefix: str = "rp", drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create Gaussian random-projection features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `2` |
| `random_state` | keyword only | `int \| None` | `None` |
| `prefix` | keyword only | `str` | `"rp"` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.random_projection_features(...)
```
### random_projection_step

Qualified name: `macroforecast.feature_engineering.compose.random_projection_step`

#### Signature

```python
macroforecast.feature_engineering.random_projection_step(*, name: str = "rp", input: str = "panel", columns: Iterable[str] | None = None, n_components: int = 2, random_state: int | None = None, prefix: str | None = None, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, include: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable Gaussian random-projection step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"rp"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `2` |
| `random_state` | keyword only | `int \| None` | `None` |
| `prefix` | keyword only | `str \| None` | `None` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.random_projection_step(...)
```
### rank_space_features

Qualified name: `macroforecast.feature_engineering.aggregation.rank_space_features`

#### Signature

```python
macroforecast.feature_engineering.rank_space_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, prefix: str = "rank_", drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Sort each row into rank-space features for supervised aggregation.

The primitive is generic order-statistic feature construction. It is the
reusable form of ``x.transformation`` in the R ``assemblage`` package used
for Albacoreranks.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `prefix` | keyword only | `str` | `"rank_"` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.rank_space_features(...)
```
### rfe_selection

Qualified name: `macroforecast.feature_engineering.feature_selection.rfe_selection`

#### Signature

```python
macroforecast.feature_engineering.rfe_selection(data: FeatureInput, target: str | pd.Series | None = None, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_features: int | float = 0.5, estimator: str = "ridge", step: int | float = 1, use_cv: bool = False, cv_folds: int = 5, min_train_size: int | None = None, random_state: int | None = 0, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Select columns by recursive feature elimination.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `target` | positional or keyword | `str \| pd.Series \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_features` | keyword only | `int \| float` | `0.5` |
| `estimator` | keyword only | `str` | `"ridge"` |
| `step` | keyword only | `int \| float` | `1` |
| `use_cv` | keyword only | `bool` | `False` |
| `cv_folds` | keyword only | `int` | `5` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `0` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.rfe_selection(...)
```
### rolling_mean

Qualified name: `macroforecast.feature_engineering.transforms.rolling_mean`

#### Signature

```python
macroforecast.feature_engineering.rolling_mean(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, windows: Iterable[int] | int = (3,), min_periods: int | None = None, shift: int = 0, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create rolling-mean feature columns from a canonical panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `windows` | keyword only | `Iterable[int] \| int` | `(3,)` |
| `min_periods` | keyword only | `int \| None` | `None` |
| `shift` | keyword only | `int` | `0` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.rolling_mean(...)
```
### rolling_step

Qualified name: `macroforecast.feature_engineering.compose.rolling_step`

#### Signature

```python
macroforecast.feature_engineering.rolling_step(*, name: str = "rolling_mean", input: str = "panel", columns: Iterable[str] | None = None, windows: Iterable[int] | int = (3,), min_periods: int | None = None, shift: int = 0, include: bool = True, drop_missing: bool = False) -> dict[str, Any]
```

#### Description

Return a reusable rolling-mean step for ``compose_features``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"rolling_mean"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `windows` | keyword only | `Iterable[int] \| int` | `(3,)` |
| `min_periods` | keyword only | `int \| None` | `None` |
| `shift` | keyword only | `int` | `0` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.rolling_step(...)
```
### savitzky_golay_features

Qualified name: `macroforecast.feature_engineering.transforms.savitzky_golay_features`

#### Signature

```python
macroforecast.feature_engineering.savitzky_golay_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, window_length: int = 5, polyorder: int = 2, derivative: int = 0, drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Smooth columns with a centered Savitzky-Golay filter.

The scipy filter uses a centered local window by default. It is direct-only
and warns by default because centered smoothing can leak future values
relative to a forecasting origin.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `window_length` | keyword only | `int` | `5` |
| `polyorder` | keyword only | `int` | `2` |
| `derivative` | keyword only | `int` | `0` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.savitzky_golay_features(...)
```
### scale_features

Qualified name: `macroforecast.feature_engineering.transforms.scale_features`

#### Signature

```python
macroforecast.feature_engineering.scale_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, method: str = "zscore", fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Scale features with either expanding or explicit full-sample fitting.

``fit_policy='expanding'`` estimates scaling parameters using observations
available through each row's date. ``fit_policy='full_sample'`` is useful
for exploratory transforms but should not be used before a forecasting
split unless the caller deliberately accepts full-sample leakage.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `method` | keyword only | `str` | `"zscore"` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.scale_features(...)
```
### scale_step

Qualified name: `macroforecast.feature_engineering.compose.scale_step`

#### Signature

```python
macroforecast.feature_engineering.scale_step(*, name: str = "scale", input: str = "panel", columns: Iterable[str] | None = None, method: str = "zscore", fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, include: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable scaling step for ``compose_features``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"scale"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `method` | keyword only | `str` | `"zscore"` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.scale_step(...)
```
### season_dummy

Qualified name: `macroforecast.feature_engineering.transforms.season_dummy`

#### Signature

```python
macroforecast.feature_engineering.season_dummy(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, frequency: str = "auto", drop_first: bool = False) -> pd.DataFrame
```

#### Description

Create month or quarter seasonal dummies from the date index.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `frequency` | keyword only | `str` | `"auto"` |
| `drop_first` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.season_dummy(...)
```
### season_dummy_step

Qualified name: `macroforecast.feature_engineering.compose.season_dummy_step`

#### Signature

```python
macroforecast.feature_engineering.season_dummy_step(*, name: str = "season_dummy", input: str = "panel", frequency: str = "auto", drop_first: bool = False, include: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable date-index seasonal-dummy step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"season_dummy"` |
| `input` | keyword only | `str` | `"panel"` |
| `frequency` | keyword only | `str` | `"auto"` |
| `drop_first` | keyword only | `bool` | `False` |
| `include` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.season_dummy_step(...)
```
### seasonal_lag

Qualified name: `macroforecast.feature_engineering.transforms.seasonal_lag`

#### Signature

```python
macroforecast.feature_engineering.seasonal_lag(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, season_length: int = 12, lags: Iterable[int] | int = (1,), drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create seasonal lag features such as 12-month or 4-quarter lags.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `season_length` | keyword only | `int` | `12` |
| `lags` | keyword only | `Iterable[int] \| int` | `(1,)` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.seasonal_lag(...)
```
### seasonal_lag_step

Qualified name: `macroforecast.feature_engineering.compose.seasonal_lag_step`

#### Signature

```python
macroforecast.feature_engineering.seasonal_lag_step(*, name: str = "seasonal_lag", input: str = "panel", columns: Iterable[str] | None = None, season_length: int = 12, lags: Iterable[int] | int = (1,), include: bool = True, drop_missing: bool = False) -> dict[str, Any]
```

#### Description

Return a reusable seasonal-lag step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"seasonal_lag"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `season_length` | keyword only | `int` | `12` |
| `lags` | keyword only | `Iterable[int] \| int` | `(1,)` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.seasonal_lag_step(...)
```
### select_features

Qualified name: `macroforecast.feature_engineering.feature_selection.select_features`

#### Signature

```python
macroforecast.feature_engineering.select_features(source: pd.DataFrame, target: pd.Series | None = None, *, n_features: int | float | None = 0.5, method: str = "variance_selection", min_train_size: int | None = None, random_state: int | None = 0, **params: Any) -> FeatureSelectionResult
```

#### Description

Fit one feature-selection rule on a training matrix.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `source` | positional or keyword | `pd.DataFrame` | `required` |
| `target` | positional or keyword | `pd.Series \| None` | `None` |
| `n_features` | keyword only | `int \| float \| None` | `0.5` |
| `method` | keyword only | `str` | `"variance_selection"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `0` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`FeatureSelectionResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.select_features(...)
```
### sliced_inverse_regression_features

Qualified name: `macroforecast.feature_engineering.transforms.sliced_inverse_regression_features`

#### Signature

```python
macroforecast.feature_engineering.sliced_inverse_regression_features(data: FeatureInput, target: str | pd.Series | None = None, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_components: int = 3, n_slices: int = 10, scaling_policy: str = "scaled_pca", prefix: str = "sir", drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create Sliced Inverse Regression factors from a target signal.

SIR is target-aware, so this direct callable fits on all target-aligned rows.
For runner-safe use, call ``sliced_inverse_regression_step()`` inside
``feature_spec()`` so the directions are fitted only on each fit window.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `target` | positional or keyword | `str \| pd.Series \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `3` |
| `n_slices` | keyword only | `int` | `10` |
| `scaling_policy` | keyword only | `str` | `"scaled_pca"` |
| `prefix` | keyword only | `str` | `"sir"` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.sliced_inverse_regression_features(...)
```
### sliced_inverse_regression_step

Qualified name: `macroforecast.feature_engineering.compose.sliced_inverse_regression_step`

#### Signature

```python
macroforecast.feature_engineering.sliced_inverse_regression_step(*, name: str = "sir", input: str = "panel", columns: Iterable[str] | None = None, n_components: int = 3, n_slices: int = 10, scaling_policy: str = "scaled_pca", min_train_size: int | None = None, prefix: str | None = None, include: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a target-aware SIR step for ``feature_spec``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"sir"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `3` |
| `n_slices` | keyword only | `int` | `10` |
| `scaling_policy` | keyword only | `str` | `"scaled_pca"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `prefix` | keyword only | `str \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.sliced_inverse_regression_step(...)
```
### sparse_pca_chen_rohe_features

Qualified name: `macroforecast.feature_engineering.transforms.sparse_pca_chen_rohe_features`

#### Signature

```python
macroforecast.feature_engineering.sparse_pca_chen_rohe_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_components: int = 4, zeta: float = 0.0, max_iter: int = 200, var_innovations: bool = False, prefix: str | None = None, min_train_size: int | None = None, drop_missing: bool = False, random_state: int | None = 0, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Create Chen-Rohe sparse component analysis factors.

The loading matrix is learned from complete input rows and then applied to
the same panel, so this direct callable is a full-sample transform. Use
``sparse_pca_chen_rohe_step()`` inside ``feature_spec()`` when the runner
must fit loadings only on each training window.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `4` |
| `zeta` | keyword only | `float` | `0.0` |
| `max_iter` | keyword only | `int` | `200` |
| `var_innovations` | keyword only | `bool` | `False` |
| `prefix` | keyword only | `str \| None` | `None` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int \| None` | `0` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.sparse_pca_chen_rohe_features(...)
```
### sparse_pca_chen_rohe_step

Qualified name: `macroforecast.feature_engineering.compose.sparse_pca_chen_rohe_step`

#### Signature

```python
macroforecast.feature_engineering.sparse_pca_chen_rohe_step(*, name: str = "sca", input: str = "panel", columns: Iterable[str] | None = None, n_components: int = 4, zeta: float = 0.0, max_iter: int = 200, var_innovations: bool = False, prefix: str | None = None, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, include: bool = True, drop_missing: bool = False, random_state: int | None = 0, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable Chen-Rohe sparse component step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"sca"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_components` | keyword only | `int` | `4` |
| `zeta` | keyword only | `float` | `0.0` |
| `max_iter` | keyword only | `int` | `200` |
| `var_innovations` | keyword only | `bool` | `False` |
| `prefix` | keyword only | `str \| None` | `None` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int \| None` | `0` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.sparse_pca_chen_rohe_step(...)
```
### stability_selection

Qualified name: `macroforecast.feature_engineering.feature_selection.stability_selection`

#### Signature

```python
macroforecast.feature_engineering.stability_selection(data: FeatureInput, target: str | pd.Series | None = None, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_features: int | float | None = 0.5, n_subsamples: int = 100, subsample_fraction: float = 0.5, pi_threshold: float = 0.6, base_estimator: str = "lasso", alpha: float = 0.01, l1_ratio: float = 0.5, min_train_size: int | None = None, random_state: int | None = 0, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Select columns by repeated sparse-model subsampling frequency.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `target` | positional or keyword | `str \| pd.Series \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_features` | keyword only | `int \| float \| None` | `0.5` |
| `n_subsamples` | keyword only | `int` | `100` |
| `subsample_fraction` | keyword only | `float` | `0.5` |
| `pi_threshold` | keyword only | `float` | `0.6` |
| `base_estimator` | keyword only | `str` | `"lasso"` |
| `alpha` | keyword only | `float` | `0.01` |
| `l1_ratio` | keyword only | `float` | `0.5` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `0` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.stability_selection(...)
```
### time_features

Qualified name: `macroforecast.feature_engineering.transforms.time_features`

#### Signature

```python
macroforecast.feature_engineering.time_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, trend: bool = True, month: bool = False, quarter: bool = False, year: bool = False) -> pd.DataFrame
```

#### Description

Create deterministic date-index features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `trend` | keyword only | `bool` | `True` |
| `month` | keyword only | `bool` | `False` |
| `quarter` | keyword only | `bool` | `False` |
| `year` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.time_features(...)
```
### time_step

Qualified name: `macroforecast.feature_engineering.compose.time_step`

#### Signature

```python
macroforecast.feature_engineering.time_step(*, name: str = "time", input: str = "panel", trend: bool = True, month: bool = False, quarter: bool = False, year: bool = False, include: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable deterministic trend/month/quarter/year step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"time"` |
| `input` | keyword only | `str` | `"panel"` |
| `trend` | keyword only | `bool` | `True` |
| `month` | keyword only | `bool` | `False` |
| `quarter` | keyword only | `bool` | `False` |
| `year` | keyword only | `bool` | `False` |
| `include` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.time_step(...)
```
### transform_step

Qualified name: `macroforecast.feature_engineering.compose.transform_step`

#### Signature

```python
macroforecast.feature_engineering.transform_step(*, transform: str, name: str | None = None, input: str = "panel", columns: Iterable[str] | None = None, periods: int = 1, include: bool = True, drop_missing: bool = False) -> dict[str, Any]
```

#### Description

Return a reusable deterministic column transform step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `transform` | keyword only | `str` | `required` |
| `name` | keyword only | `str \| None` | `None` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `periods` | keyword only | `int` | `1` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.transform_step(...)
```
### transform_features

Qualified name: `macroforecast.feature_engineering.transforms.transform_features`

#### Signature

```python
macroforecast.feature_engineering.transform_features(data: FeatureInput, *, transform: str, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, periods: int = 1, drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Apply a simple column-wise transformation as feature engineering.

These helpers are separate from preprocessing t-codes. Use them when the
model feature set needs extra ML transforms after the canonical panel has
already been cleaned.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `transform` | keyword only | `str` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `periods` | keyword only | `int` | `1` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.transform_features(...)
```
### varimax_features

Qualified name: `macroforecast.feature_engineering.transforms.varimax_features`

#### Signature

```python
macroforecast.feature_engineering.varimax_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, max_iter: int = 50, tol: float = 1e-07, prefix: str = "varimax", min_train_size: int | None = None, drop_missing: bool = False, warn_full_sample: bool = True) -> pd.DataFrame
```

#### Description

Rotate factor-score columns with an orthogonal varimax rotation.

This direct callable fits the rotation on all complete rows. Use
``varimax_step()`` inside ``feature_spec()`` when the rotation must be fit
only on a forecasting window's feature-fit panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `max_iter` | keyword only | `int` | `50` |
| `tol` | keyword only | `float` | `1e-07` |
| `prefix` | keyword only | `str` | `"varimax"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.varimax_features(...)
```
### varimax_step

Qualified name: `macroforecast.feature_engineering.compose.varimax_step`

#### Signature

```python
macroforecast.feature_engineering.varimax_step(*, name: str = "varimax", input: str = "panel", columns: Iterable[str] | None = None, max_iter: int = 50, tol: float = 1e-07, prefix: str | None = None, fit_policy: FitPolicy = "expanding", min_train_size: int | None = None, include: bool = True, drop_missing: bool = False, warn_full_sample: bool = True) -> dict[str, Any]
```

#### Description

Return a reusable orthogonal varimax-rotation step.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | keyword only | `str` | `"varimax"` |
| `input` | keyword only | `str` | `"panel"` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `max_iter` | keyword only | `int` | `50` |
| `tol` | keyword only | `float` | `1e-07` |
| `prefix` | keyword only | `str \| None` | `None` |
| `fit_policy` | keyword only | `FitPolicy` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `include` | keyword only | `bool` | `True` |
| `drop_missing` | keyword only | `bool` | `False` |
| `warn_full_sample` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.varimax_step(...)
```
### variance_selection

Qualified name: `macroforecast.feature_engineering.feature_selection.variance_selection`

#### Signature

```python
macroforecast.feature_engineering.variance_selection(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_features: int | float = 0.5, min_train_size: int | None = None) -> pd.DataFrame
```

#### Description

Select columns with the largest sample variance.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_features` | keyword only | `int \| float` | `0.5` |
| `min_train_size` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.variance_selection(...)
```
### wavelet_features

Qualified name: `macroforecast.feature_engineering.transforms.wavelet_features`

#### Signature

```python
macroforecast.feature_engineering.wavelet_features(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, n_levels: int = 3, wavelet: str = "db4", drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Create rolling multi-resolution approximation/detail features.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `n_levels` | keyword only | `int` | `3` |
| `wavelet` | keyword only | `str` | `"db4"` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.wavelet_features(...)
```
### weighted_aggregate

Qualified name: `macroforecast.feature_engineering.aggregation.weighted_aggregate`

#### Signature

```python
macroforecast.feature_engineering.weighted_aggregate(data: FeatureInput, weights: Mapping[str, float] | pd.Series | pd.DataFrame | Sequence[float], *, metadata: Mapping[str, Any] | None = None, columns: Iterable[str] | None = None, normalize: bool = True, name: str = "weighted_aggregate", drop_missing: bool = False) -> pd.DataFrame
```

#### Description

Apply aligned component weights to a panel and return one aggregate.

Albacore uses this object as a learned core-inflation index. This callable
keeps the operation generic for any component-to-aggregate macro panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `FeatureInput` | `required` |
| `weights` | positional or keyword | `Mapping[str, float] \| pd.Series \| pd.DataFrame \| Sequence[float]` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `normalize` | keyword only | `bool` | `True` |
| `name` | keyword only | `str` | `"weighted_aggregate"` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.feature_engineering.weighted_aggregate(...)
```
