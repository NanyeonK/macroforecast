# macroforecast.preprocessing

[Back to reference](index.md)

Transform-code application, outlier handling, imputation, frame-edge handling, and reusable preprocessing specs.

Guide context: [../guide/concepts/preprocessing.md](../guide/concepts/preprocessing.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `reprocess` | function | Preprocess a canonical macroforecast panel. |
| `plan` | function | Return a dry-run summary of preprocessing choices and metadata provenance. |
| `report` | function | Return a compact preprocessing report from a processed object. |
| `PreprocessedData` | class | Cleaned macroforecast panel plus metadata and data-spec choices. |
| `PreprocessInput` | data | Represent a PEP 604 union type |
| `PreprocessSpec` | class | Reusable preprocessing callable for window-local forecasting runners. |
| `FittedPreprocessor` | class | Preprocessing spec fitted on a training window. |
| `preprocess_spec` | function | Create a reusable preprocessing specification. |
| `custom_preprocess` | function | Apply a user supplied preprocessing callable to a canonical panel. |
| `custom_preprocess_step` | function | Return a custom preprocessing step for ``preprocess_spec(custom_steps=...)``. |
| `apply_transform_codes` | function | Apply McCracken-Ng transform codes to matching panel columns. |
| `fred_sd_transform_codes` | function | Build FRED-SD t-code choices for state-series columns. |
| `handle_tcode_lag` | function | Handle missing values introduced by stationarity transforms. |
| `handle_outliers` | function | Apply one outlier policy to a panel. |
| `impute_missing` | function | Fill missing panel values with the selected imputation method. |
| `standardize_panel` | function | Standardize numeric columns with full-panel fitted parameters. |
| `handle_frame_edges` | function | Handle remaining unbalanced panel edges. |
| `FRED_SD_NATIONAL_ANALOG_TRANSFORM_CODES` | data | dict() -> new empty dictionary |
| `FRED_SD_MEDIUM_CONFIDENCE_TRANSFORM_CODES` | data | dict() -> new empty dictionary |
| `iqr_outlier_clean` | function | Flag or replace outliers with a per-column IQR rule. |
| `zscore_outlier_clean` | function | Flag or replace outliers with a per-column z-score rule. |
| `winsorize_clean` | function | Clip numeric columns to quantile bounds. |
| `em_factor_impute_clean` | function | Impute missing numeric cells with PCA-EM factor reconstruction. |
| `em_multivariate_impute_clean` | function | Impute missing numeric cells with an uncapped PCA-EM rank rule. |
| `mean_impute_clean` | function | Replace missing numeric cells with full-column means. |
| `forward_fill_clean` | function | Carry each series' most recent observed value forward. |
| `linear_interpolate_clean` | function | Fill interior missing values by linear interpolation. |
| `truncate_to_balanced_clean` | function | Keep only rows with no missing values. |
| `drop_unbalanced_series_clean` | function | Keep only columns with no missing values. |
| `zero_fill_leading_clean` | function | Replace remaining missing cells with zero. |
| `fit_standardization_state` | function | Fit column-wise scaling parameters on a numeric panel. |
| `apply_standardization_state` | function | Apply fitted column-wise scaling parameters to a panel. |
| `box_cox_clean` | function | Apply a Box-Cox variance-stabilising transform to each numeric column. |
| `box_cox_lambda` | function | Select a Box-Cox transformation parameter ``lambda`` for one series. |
| `inverse_box_cox` | function | Invert a Box-Cox transform with parameter ``lmbda``. |
| `standardize_clean` | function | Standardize numeric columns with full-sample fitted parameters. |
| `apply_tcode_transform` | function | Apply McCracken-Ng transformation codes to matching columns. |
| `freq_align_quarterly_to_monthly_clean` | function | Align selected quarterly columns on the panel's monthly grid. |
| `freq_align_monthly_to_quarterly_clean` | function | Aggregate selected monthly columns to a quarterly grid. |

## Data And Module Values

### `PreprocessInput`

Kind: `data`

```python
PreprocessInput = macroforecast.preprocessing.types.PreprocessedData | macroforecast.data.panel.DataSpec | macroforecast.data.panel.DataBundle | tuple[pandas.core.frame.DataFrame, collections.abc.Mapping[str, typing.Any]] | pandas.core.frame.DataFrame
```
### `FRED_SD_NATIONAL_ANALOG_TRANSFORM_CODES`

Kind: `data`

```python
FRED_SD_NATIONAL_ANALOG_TRANSFORM_CODES = dict(13 entries: CONS, FIRE, GOVT, ICLAIMS, INFO, LF, MFG, MFGHRS, MINNG, NA, PARTRATE, PSERV, ...)
```
### `FRED_SD_MEDIUM_CONFIDENCE_TRANSFORM_CODES`

Kind: `data`

```python
FRED_SD_MEDIUM_CONFIDENCE_TRANSFORM_CODES = dict(15 entries: BPPRIVSA, CONSTNQGSP, EXPORTS, FIRENQGSP, GOVNQGSP, IMPORTS, INFONQGSP, MANNQGSP, NATURNQGSP, NQGSP, OTOT, PSERVNQGSP, ...)
```

## Callable And Class Reference

### reprocess

Qualified name: `macroforecast.preprocessing.preprocess.reprocess`

#### Signature

```python
macroforecast.preprocessing.reprocess(data: PreprocessInput, *, metadata: Mapping[str, Any] | None = None, frequency: str = "keep", quarterly_to_monthly: str = "step_forward", weekly_to_monthly: str = "mean", monthly_to_quarterly: str = "quarterly_average", weekly_to_quarterly: str = "mean", transform_order: str = "after_frequency", transform: str = "official", transform_codes: Mapping[str, int] | None = None, transform_code_overrides: Mapping[str, int] | None = None, tcode_lag: str = "drop", outliers: str = "iqr", outlier_action: str = "flag_as_nan", iqr_threshold: float = 10.0, zscore_threshold: float = 3.0, winsorize_quantiles: tuple[float, float] = (0.01, 0.99), impute: str = "em_factor", em_n_factors: int = 8, em_factor_selection: str = "baing_p2", em_demean: int = 2, em_max_iter: int = 50, em_tolerance: float = 1e-06, standardize: str = "none", standardize_columns: str | Sequence[str] = "all", standardize_ddof: int = 0, frame: str = "keep", warn_metadata: bool = True) -> PreprocessedData
```

#### Description

Preprocess a canonical macroforecast panel.

Parameters use user-facing names. Common legacy aliases such as
``apply_official_tcode`` and ``truncate_to_balanced`` are accepted, but
returned metadata records the canonical direct-call names.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PreprocessInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `frequency` | keyword only | `str` | `"keep"` |
| `quarterly_to_monthly` | keyword only | `str` | `"step_forward"` |
| `weekly_to_monthly` | keyword only | `str` | `"mean"` |
| `monthly_to_quarterly` | keyword only | `str` | `"quarterly_average"` |
| `weekly_to_quarterly` | keyword only | `str` | `"mean"` |
| `transform_order` | keyword only | `str` | `"after_frequency"` |
| `transform` | keyword only | `str` | `"official"` |
| `transform_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `transform_code_overrides` | keyword only | `Mapping[str, int] \| None` | `None` |
| `tcode_lag` | keyword only | `str` | `"drop"` |
| `outliers` | keyword only | `str` | `"iqr"` |
| `outlier_action` | keyword only | `str` | `"flag_as_nan"` |
| `iqr_threshold` | keyword only | `float` | `10.0` |
| `zscore_threshold` | keyword only | `float` | `3.0` |
| `winsorize_quantiles` | keyword only | `tuple[float, float]` | `(0.01, 0.99)` |
| `impute` | keyword only | `str` | `"em_factor"` |
| `em_n_factors` | keyword only | `int` | `8` |
| `em_factor_selection` | keyword only | `str` | `"baing_p2"` |
| `em_demean` | keyword only | `int` | `2` |
| `em_max_iter` | keyword only | `int` | `50` |
| `em_tolerance` | keyword only | `float` | `1e-06` |
| `standardize` | keyword only | `str` | `"none"` |
| `standardize_columns` | keyword only | `str \| Sequence[str]` | `"all"` |
| `standardize_ddof` | keyword only | `int` | `0` |
| `frame` | keyword only | `str` | `"keep"` |
| `warn_metadata` | keyword only | `bool` | `True` |

#### Returns

`PreprocessedData`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.reprocess(...)
```
### plan

Qualified name: `macroforecast.preprocessing.preprocess.plan`

#### Signature

```python
macroforecast.preprocessing.plan(data: PreprocessInput, *, metadata: Mapping[str, Any] | None = None, frequency: str = "keep", transform_order: str = "after_frequency", transform: str = "official", transform_codes: Mapping[str, int] | None = None, transform_code_overrides: Mapping[str, int] | None = None, tcode_lag: str = "drop", outliers: str = "iqr", impute: str = "em_factor", standardize: str = "none", standardize_columns: str | Sequence[str] = "all", standardize_ddof: int = 0, frame: str = "keep") -> dict[str, Any]
```

#### Description

Return a dry-run summary of preprocessing choices and metadata provenance.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PreprocessInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `frequency` | keyword only | `str` | `"keep"` |
| `transform_order` | keyword only | `str` | `"after_frequency"` |
| `transform` | keyword only | `str` | `"official"` |
| `transform_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `transform_code_overrides` | keyword only | `Mapping[str, int] \| None` | `None` |
| `tcode_lag` | keyword only | `str` | `"drop"` |
| `outliers` | keyword only | `str` | `"iqr"` |
| `impute` | keyword only | `str` | `"em_factor"` |
| `standardize` | keyword only | `str` | `"none"` |
| `standardize_columns` | keyword only | `str \| Sequence[str]` | `"all"` |
| `standardize_ddof` | keyword only | `int` | `0` |
| `frame` | keyword only | `str` | `"keep"` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.plan(...)
```
### report

Qualified name: `macroforecast.preprocessing.preprocess.report`

#### Signature

```python
macroforecast.preprocessing.report(processed: PreprocessedData) -> dict[str, Any]
```

#### Description

Return a compact preprocessing report from a processed object.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `processed` | positional or keyword | `PreprocessedData` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.report(...)
```
### PreprocessedData

Qualified name: `macroforecast.preprocessing.types.PreprocessedData`

#### Signature

```python
macroforecast.preprocessing.PreprocessedData(panel: pd.DataFrame, metadata: dict[str, Any] = <factory>, target: str | None = None, targets: tuple[str, ...] = (), horizons: tuple[int, ...] = (), start: str | None = None, end: str | None = None, predictors: Any = "all", steps: tuple[dict[str, Any], ...] = ()) -> None
```

#### Description

Cleaned macroforecast panel plus metadata and data-spec choices.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `target` | positional or keyword | `str \| None` | `None` |
| `targets` | positional or keyword | `tuple[str, ...]` | `()` |
| `horizons` | positional or keyword | `tuple[int, ...]` | `()` |
| `start` | positional or keyword | `str \| None` | `None` |
| `end` | positional or keyword | `str \| None` | `None` |
| `predictors` | positional or keyword | `Any` | `"all"` |
| `steps` | positional or keyword | `tuple[dict[str, Any], ...]` | `()` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.preprocessing.PreprocessedData(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `panel` | `pd.DataFrame` | `required` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `target` | `str \| None` | `None` |
| `targets` | `tuple[str, ...]` | `()` |
| `horizons` | `tuple[int, ...]` | `()` |
| `start` | `str \| None` | `None` |
| `end` | `str \| None` | `None` |
| `predictors` | `Any` | `"all"` |
| `steps` | `tuple[dict[str, Any], ...]` | `()` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `attach` | `attach(self, stage: str, values: Mapping[str, Any]) -> PreprocessedData` | No public docstring is available. |
### PreprocessSpec

Qualified name: `macroforecast.preprocessing.specs.PreprocessSpec`

#### Signature

```python
macroforecast.preprocessing.PreprocessSpec(options: dict[str, Any] = <factory>) -> None
```

#### Description

Reusable preprocessing callable for window-local forecasting runners.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `options` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.preprocessing.PreprocessSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `options` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, data: PreprocessInput, *, metadata: dict[str, Any] \| None = None, policy: str = "origin_available") -> FittedPreprocessor` | Fit preprocessing choices on a training panel. |
| `fit_transform` | `fit_transform(self, data: PreprocessInput, *, metadata: dict[str, Any] \| None = None, policy: str = "origin_available") -> PreprocessedData` | Fit on ``data`` and return the processed training panel. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return JSON-ready preprocessing choices. |
| `to_metadata` | `to_metadata(self) -> dict[str, Any]` | Return compact metadata for runners. |
### FittedPreprocessor

Qualified name: `macroforecast.preprocessing.specs.FittedPreprocessor`

#### Signature

```python
macroforecast.preprocessing.FittedPreprocessor(spec: PreprocessSpec, fit_panel: pd.DataFrame, fit_metadata: dict[str, Any], processed_train: PreprocessedData, preprocessing_scope: str = "origin_available", standardization_state: dict[str, Any] | None = None, state_panel: pd.DataFrame | None = None, outlier_state: dict[str, Any] | None = None, impute_state: dict[str, Any] | None = None, train_after_outlier: pd.DataFrame | None = None) -> None
```

#### Description

Preprocessing spec fitted on a training window.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `spec` | positional or keyword | `PreprocessSpec` | `required` |
| `fit_panel` | positional or keyword | `pd.DataFrame` | `required` |
| `fit_metadata` | positional or keyword | `dict[str, Any]` | `required` |
| `processed_train` | positional or keyword | `PreprocessedData` | `required` |
| `preprocessing_scope` | positional or keyword | `str` | `"origin_available"` |
| `standardization_state` | positional or keyword | `dict[str, Any] \| None` | `None` |
| `state_panel` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `outlier_state` | positional or keyword | `dict[str, Any] \| None` | `None` |
| `impute_state` | positional or keyword | `dict[str, Any] \| None` | `None` |
| `train_after_outlier` | positional or keyword | `pd.DataFrame \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.preprocessing.FittedPreprocessor(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `spec` | `PreprocessSpec` | `required` |
| `fit_panel` | `pd.DataFrame` | `required` |
| `fit_metadata` | `dict[str, Any]` | `required` |
| `processed_train` | `PreprocessedData` | `required` |
| `preprocessing_scope` | `str` | `"origin_available"` |
| `standardization_state` | `dict[str, Any] \| None` | `None` |
| `state_panel` | `pd.DataFrame \| None` | `None` |
| `outlier_state` | `dict[str, Any] \| None` | `None` |
| `impute_state` | `dict[str, Any] \| None` | `None` |
| `train_after_outlier` | `pd.DataFrame \| None` | `None` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_metadata` | `to_metadata(self) -> dict[str, Any]` | Return fit metadata for forecasting records. |
| `transform` | `transform(self, data: PreprocessInput, *, metadata: dict[str, Any] \| None = None, history: pd.DataFrame \| None = None, policy: str \| None = None, available: pd.Index \| None = None) -> PreprocessedData` | Transform new rows using the fitted training history. |
### preprocess_spec

Qualified name: `macroforecast.preprocessing.specs.preprocess_spec`

#### Signature

```python
macroforecast.preprocessing.preprocess_spec(**options: Any) -> PreprocessSpec
```

#### Description

Create a reusable preprocessing specification.

Keyword options are the same data-cleaning choices accepted by
``reprocess(...)``: frequency alignment, transform-code handling,
outlier policy, imputation policy, standardization, frame-edge handling,
and optional custom preprocessing steps. Stage timing and metadata are not
accepted here; they are supplied later through ``PreprocessSpec.fit(...)``
or by the forecasting/pipeline runner.

Custom preprocessing callables are safe for in-memory use. Disk-backed
preprocessing caches require a stable callable identity: use named
functions and set ``func.__mf_digest__`` whenever cached reuse should span
processes or runs. Anonymous lambdas without ``__mf_digest__`` are rejected
because they cannot be distinguished by a stable content identity.

With ``policy="fit_window"``, custom steps are re-executed on the apply
window at each origin. Those steps must be row-local/stateless; a custom
step that computes statistics from its whole input can read post-origin
rows and leak future information.

Returns
PreprocessSpec
    Frozen preprocessing configuration. Call ``fit(data)`` to get a
    ``FittedPreprocessor`` or ``fit_transform(data)`` to obtain a
    ``PreprocessedData`` object for the training panel.

Example
>>> import macroforecast as mf
>>> prep = mf.preprocessing.preprocess_spec(
...     transform="official",
...     outliers="iqr",
...     impute="em_factor",
...     standardize="zscore",
... )

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `options` | var keyword | `Any` | `required` |

#### Returns

`PreprocessSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.preprocess_spec(...)
```
### custom_preprocess

Qualified name: `macroforecast.preprocessing.preprocess.custom_preprocess`

#### Signature

```python
macroforecast.preprocessing.custom_preprocess(data: PreprocessInput, func: Callable[..., Any], *, metadata: Mapping[str, Any] | None = None, name: str | None = None, **params: Any) -> PreprocessedData
```

#### Description

Apply a user supplied preprocessing callable to a canonical panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PreprocessInput` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `name` | keyword only | `str \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`PreprocessedData`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.custom_preprocess(...)
```
### custom_preprocess_step

Qualified name: `macroforecast.preprocessing.specs.custom_preprocess_step`

#### Signature

```python
macroforecast.preprocessing.custom_preprocess_step(name: str, func: Callable[..., Any], **params: Any) -> dict[str, Any]
```

#### Description

Return a custom preprocessing step for ``preprocess_spec(custom_steps=...)``.

For disk-backed preprocessing caches, set ``func.__mf_digest__`` to a stable
string and update it when the callable's behavior changes. Without that
opt-in digest, the runner skips disk get/put for specs containing the
callable and recomputes instead of risking stale reuse.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.custom_preprocess_step(...)
```
### apply_transform_codes

Qualified name: `macroforecast.preprocessing.preprocess.apply_transform_codes`

#### Signature

```python
macroforecast.preprocessing.apply_transform_codes(panel: pd.DataFrame, codes: Mapping[str, int]) -> pd.DataFrame
```

#### Description

Apply McCracken-Ng transform codes to matching panel columns.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `codes` | positional or keyword | `Mapping[str, int]` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.apply_transform_codes(...)
```
### fred_sd_transform_codes

Qualified name: `macroforecast.preprocessing.preprocess.fred_sd_transform_codes`

#### Signature

```python
macroforecast.preprocessing.fred_sd_transform_codes(data: PreprocessInput, *, variable_codes: Mapping[str, int] | None = None, state_series_codes: Mapping[str, int] | None = None, use_national_analog_suggestions: bool = True, include_medium_confidence: bool = False, return_table: bool = False) -> dict[str, int] | tuple[dict[str, int], pd.DataFrame]
```

#### Description

Build FRED-SD t-code choices for state-series columns.

FRED-SD does not publish official t-codes. The built-in suggestions are
national-analog defaults, not official transformations.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PreprocessInput` | `required` |
| `variable_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `state_series_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `use_national_analog_suggestions` | keyword only | `bool` | `True` |
| `include_medium_confidence` | keyword only | `bool` | `False` |
| `return_table` | keyword only | `bool` | `False` |

#### Returns

`dict[str, int] | tuple[dict[str, int], pd.DataFrame]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.fred_sd_transform_codes(...)
```
### handle_tcode_lag

Qualified name: `macroforecast.preprocessing.preprocess.handle_tcode_lag`

#### Signature

```python
macroforecast.preprocessing.handle_tcode_lag(panel: pd.DataFrame, *, method: str = "drop", codes: Mapping[str, int] | None = None) -> pd.DataFrame
```

#### Description

Handle missing values introduced by stationarity transforms.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | keyword only | `str` | `"drop"` |
| `codes` | keyword only | `Mapping[str, int] \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.handle_tcode_lag(...)
```
### handle_outliers

Qualified name: `macroforecast.preprocessing.preprocess.handle_outliers`

#### Signature

```python
macroforecast.preprocessing.handle_outliers(panel: pd.DataFrame, *, method: str = "iqr", action: str = "flag_as_nan", iqr_threshold: float = 10.0, zscore_threshold: float = 3.0, winsorize_quantiles: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame
```

#### Description

Apply one outlier policy to a panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | keyword only | `str` | `"iqr"` |
| `action` | keyword only | `str` | `"flag_as_nan"` |
| `iqr_threshold` | keyword only | `float` | `10.0` |
| `zscore_threshold` | keyword only | `float` | `3.0` |
| `winsorize_quantiles` | keyword only | `tuple[float, float]` | `(0.01, 0.99)` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.handle_outliers(...)
```
### impute_missing

Qualified name: `macroforecast.preprocessing.preprocess.impute_missing`

#### Signature

```python
macroforecast.preprocessing.impute_missing(panel: pd.DataFrame, *, method: str = "em_factor", em_n_factors: int = 8, em_factor_selection: str = "baing_p2", em_demean: int = 2, em_max_iter: int = 50, em_tolerance: float = 1e-06) -> pd.DataFrame
```

#### Description

Fill missing panel values with the selected imputation method.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | keyword only | `str` | `"em_factor"` |
| `em_n_factors` | keyword only | `int` | `8` |
| `em_factor_selection` | keyword only | `str` | `"baing_p2"` |
| `em_demean` | keyword only | `int` | `2` |
| `em_max_iter` | keyword only | `int` | `50` |
| `em_tolerance` | keyword only | `float` | `1e-06` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.impute_missing(...)
```
### standardize_panel

Qualified name: `macroforecast.preprocessing.preprocess.standardize_panel`

#### Signature

```python
macroforecast.preprocessing.standardize_panel(panel: pd.DataFrame, *, method: str = "zscore", ddof: int = 0) -> pd.DataFrame
```

#### Description

Standardize numeric columns with full-panel fitted parameters.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | keyword only | `str` | `"zscore"` |
| `ddof` | keyword only | `int` | `0` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.standardize_panel(...)
```
### handle_frame_edges

Qualified name: `macroforecast.preprocessing.preprocess.handle_frame_edges`

#### Signature

```python
macroforecast.preprocessing.handle_frame_edges(panel: pd.DataFrame, *, method: str = "keep") -> pd.DataFrame
```

#### Description

Handle remaining unbalanced panel edges.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | keyword only | `str` | `"keep"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.handle_frame_edges(...)
```
### iqr_outlier_clean

Qualified name: `macroforecast.preprocessing.clean.iqr_outlier_clean`

#### Signature

```python
macroforecast.preprocessing.iqr_outlier_clean(panel: pd.DataFrame, *, threshold: float = 10.0, action: str = "flag_as_nan") -> pd.DataFrame
```

#### Description

Flag or replace outliers with a per-column IQR rule.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `threshold` | keyword only | `float` | `10.0` |
| `action` | keyword only | `str` | `"flag_as_nan"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.iqr_outlier_clean(...)
```
### zscore_outlier_clean

Qualified name: `macroforecast.preprocessing.clean.zscore_outlier_clean`

#### Signature

```python
macroforecast.preprocessing.zscore_outlier_clean(panel: pd.DataFrame, *, threshold: float = 3.0, action: str = "flag_as_nan") -> pd.DataFrame
```

#### Description

Flag or replace outliers with a per-column z-score rule.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `threshold` | keyword only | `float` | `3.0` |
| `action` | keyword only | `str` | `"flag_as_nan"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.zscore_outlier_clean(...)
```
### winsorize_clean

Qualified name: `macroforecast.preprocessing.clean.winsorize_clean`

#### Signature

```python
macroforecast.preprocessing.winsorize_clean(panel: pd.DataFrame, *, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame
```

#### Description

Clip numeric columns to quantile bounds.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `lower_quantile` | keyword only | `float` | `0.01` |
| `upper_quantile` | keyword only | `float` | `0.99` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.winsorize_clean(...)
```
### em_factor_impute_clean

Qualified name: `macroforecast.preprocessing.clean.em_factor_impute_clean`

#### Signature

```python
macroforecast.preprocessing.em_factor_impute_clean(panel: pd.DataFrame, *, n_factors: int = 8, max_iter: int = 50, tol: float = 1e-06, factor_selection: str = "baing_p2", demean: int = 2) -> pd.DataFrame
```

#### Description

Impute missing numeric cells with PCA-EM factor reconstruction.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `n_factors` | keyword only | `int` | `8` |
| `max_iter` | keyword only | `int` | `50` |
| `tol` | keyword only | `float` | `1e-06` |
| `factor_selection` | keyword only | `str` | `"baing_p2"` |
| `demean` | keyword only | `int` | `2` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.em_factor_impute_clean(...)
```
### em_multivariate_impute_clean

Qualified name: `macroforecast.preprocessing.clean.em_multivariate_impute_clean`

#### Signature

```python
macroforecast.preprocessing.em_multivariate_impute_clean(panel: pd.DataFrame, *, max_iter: int = 20, tol: float = 0.0001) -> pd.DataFrame
```

#### Description

Impute missing numeric cells with an uncapped PCA-EM rank rule.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `max_iter` | keyword only | `int` | `20` |
| `tol` | keyword only | `float` | `0.0001` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.em_multivariate_impute_clean(...)
```
### mean_impute_clean

Qualified name: `macroforecast.preprocessing.clean.mean_impute_clean`

#### Signature

```python
macroforecast.preprocessing.mean_impute_clean(panel: pd.DataFrame) -> pd.DataFrame
```

#### Description

Replace missing numeric cells with full-column means.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.mean_impute_clean(...)
```
### forward_fill_clean

Qualified name: `macroforecast.preprocessing.clean.forward_fill_clean`

#### Signature

```python
macroforecast.preprocessing.forward_fill_clean(panel: pd.DataFrame) -> pd.DataFrame
```

#### Description

Carry each series' most recent observed value forward.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.forward_fill_clean(...)
```
### linear_interpolate_clean

Qualified name: `macroforecast.preprocessing.clean.linear_interpolate_clean`

#### Signature

```python
macroforecast.preprocessing.linear_interpolate_clean(panel: pd.DataFrame) -> pd.DataFrame
```

#### Description

Fill interior missing values by linear interpolation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.linear_interpolate_clean(...)
```
### truncate_to_balanced_clean

Qualified name: `macroforecast.preprocessing.clean.truncate_to_balanced_clean`

#### Signature

```python
macroforecast.preprocessing.truncate_to_balanced_clean(panel: pd.DataFrame) -> pd.DataFrame
```

#### Description

Keep only rows with no missing values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.truncate_to_balanced_clean(...)
```
### drop_unbalanced_series_clean

Qualified name: `macroforecast.preprocessing.clean.drop_unbalanced_series_clean`

#### Signature

```python
macroforecast.preprocessing.drop_unbalanced_series_clean(panel: pd.DataFrame) -> pd.DataFrame
```

#### Description

Keep only columns with no missing values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.drop_unbalanced_series_clean(...)
```
### zero_fill_leading_clean

Qualified name: `macroforecast.preprocessing.clean.zero_fill_leading_clean`

#### Signature

```python
macroforecast.preprocessing.zero_fill_leading_clean(panel: pd.DataFrame) -> pd.DataFrame
```

#### Description

Replace remaining missing cells with zero.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.zero_fill_leading_clean(...)
```
### fit_standardization_state

Qualified name: `macroforecast.preprocessing.clean.fit_standardization_state`

#### Signature

```python
macroforecast.preprocessing.fit_standardization_state(panel: pd.DataFrame, *, method: str = "zscore", ddof: int = 0) -> dict[str, object]
```

#### Description

Fit column-wise scaling parameters on a numeric panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | keyword only | `str` | `"zscore"` |
| `ddof` | keyword only | `int` | `0` |

#### Returns

`dict[str, object]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.fit_standardization_state(...)
```
### apply_standardization_state

Qualified name: `macroforecast.preprocessing.clean.apply_standardization_state`

#### Signature

```python
macroforecast.preprocessing.apply_standardization_state(panel: pd.DataFrame, state: Mapping[str, object]) -> pd.DataFrame
```

#### Description

Apply fitted column-wise scaling parameters to a panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `state` | positional or keyword | `Mapping[str, object]` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.apply_standardization_state(...)
```
### box_cox_clean

Qualified name: `macroforecast.preprocessing.clean.box_cox_clean`

#### Signature

```python
macroforecast.preprocessing.box_cox_clean(panel: pd.DataFrame, *, lmbda: float | None = None, method: str = "loglik", period: int | None = None) -> pd.DataFrame
```

#### Description

Apply a Box-Cox variance-stabilising transform to each numeric column.

When ``lmbda`` is None the parameter is selected per column by ``method``
('loglik' or 'guerrero'); the chosen lambdas are recorded in
``panel.attrs['macroforecast_box_cox_lambda']``. Columns must be strictly
positive.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `lmbda` | keyword only | `float \| None` | `None` |
| `method` | keyword only | `str` | `"loglik"` |
| `period` | keyword only | `int \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.box_cox_clean(...)
```
### box_cox_lambda

Qualified name: `macroforecast.preprocessing.clean.box_cox_lambda`

#### Signature

```python
macroforecast.preprocessing.box_cox_lambda(series: Any, *, method: str = "loglik", period: int | None = None, bounds: tuple[float, float] = (-1.0, 2.0)) -> float
```

#### Description

Select a Box-Cox transformation parameter ``lambda`` for one series.

``method='loglik'`` uses the profile-likelihood (Box-Cox MLE, via scipy);
``method='guerrero'`` minimises the coefficient of variation of the
subseries dispersion (Guerrero 1993), the ``forecast::BoxCox.lambda`` default.
Requires strictly positive values (use a signed/Yeo-Johnson transform for
series with non-positive values).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `method` | keyword only | `str` | `"loglik"` |
| `period` | keyword only | `int \| None` | `None` |
| `bounds` | keyword only | `tuple[float, float]` | `(-1.0, 2.0)` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.box_cox_lambda(...)
```
### inverse_box_cox

Qualified name: `macroforecast.preprocessing.clean.inverse_box_cox`

#### Signature

```python
macroforecast.preprocessing.inverse_box_cox(values: Any, lmbda: float) -> np.ndarray
```

#### Description

Invert a Box-Cox transform with parameter ``lmbda``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `values` | positional or keyword | `Any` | `required` |
| `lmbda` | positional or keyword | `float` | `required` |

#### Returns

`np.ndarray`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.inverse_box_cox(...)
```
### standardize_clean

Qualified name: `macroforecast.preprocessing.clean.standardize_clean`

#### Signature

```python
macroforecast.preprocessing.standardize_clean(panel: pd.DataFrame, *, method: str = "zscore", ddof: int = 0) -> pd.DataFrame
```

#### Description

Standardize numeric columns with full-sample fitted parameters.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | keyword only | `str` | `"zscore"` |
| `ddof` | keyword only | `int` | `0` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.standardize_clean(...)
```
### apply_tcode_transform

Qualified name: `macroforecast.preprocessing.clean.apply_tcode_transform`

#### Signature

```python
macroforecast.preprocessing.apply_tcode_transform(panel: pd.DataFrame, tcode_map: Mapping[str, int]) -> pd.DataFrame
```

#### Description

Apply McCracken-Ng transformation codes to matching columns.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `tcode_map` | positional or keyword | `Mapping[str, int]` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.apply_tcode_transform(...)
```
### freq_align_quarterly_to_monthly_clean

Qualified name: `macroforecast.preprocessing.clean.freq_align_quarterly_to_monthly_clean`

#### Signature

```python
macroforecast.preprocessing.freq_align_quarterly_to_monthly_clean(panel: pd.DataFrame, quarterly_columns: Sequence[str], *, rule: str = "step_forward") -> pd.DataFrame
```

#### Description

Align selected quarterly columns on the panel's monthly grid.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `quarterly_columns` | positional or keyword | `Sequence[str]` | `required` |
| `rule` | keyword only | `str` | `"step_forward"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.freq_align_quarterly_to_monthly_clean(...)
```
### freq_align_monthly_to_quarterly_clean

Qualified name: `macroforecast.preprocessing.clean.freq_align_monthly_to_quarterly_clean`

#### Signature

```python
macroforecast.preprocessing.freq_align_monthly_to_quarterly_clean(panel: pd.DataFrame, monthly_columns: Sequence[str], *, rule: str = "quarterly_average") -> pd.DataFrame
```

#### Description

Aggregate selected monthly columns to a quarterly grid.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `monthly_columns` | positional or keyword | `Sequence[str]` | `required` |
| `rule` | keyword only | `str` | `"quarterly_average"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.preprocessing.freq_align_monthly_to_quarterly_clean(...)
```
