# macroforecast.data

[Back to reference](index.md)

Canonical date-indexed panels, metadata, FRED loaders, custom loaders, and real-time vintage sources.

Guide context: [../guide/concepts/data.md](../guide/concepts/data.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `DataBundle` | class | Canonical data payload: a pandas panel plus explicit metadata. |
| `DataSpec` | class | Panel plus target, horizon, sample, and predictor choices for a run. |
| `RegimeDirection` | callable | No public docstring is available. |
| `SamePeriodPolicy` | callable | No public docstring is available. |
| `VintagePanelSpec` | class | Run-level wrapper for a point-in-time vintage source. |
| `VintageSource` | class | A lazily-resolved source of point-in-time data, one bundle per real-time origin. |
| `VintageUnavailableError` | class | Raised when no point-in-time vintage is available for an origin. |
| `as_panel` | function | Return ``frame`` as macroforecast's canonical date-indexed panel. |
| `attach_metadata` | function | No public docstring is available. |
| `custom_dataset` | function | Build a canonical custom ``DataBundle`` from an in-memory DataFrame. |
| `custom_vintages` | function | Return a custom point-in-time source. |
| `metadata` | function | Return metadata from a bundle, spec, tuple, or DataFrame. |
| `panel_info` | function | Return a compact diagnostic summary for a canonical panel. |
| `set_frequencies` | function | Attach a column-level frequency contract to a panel or bundle. |
| `spec` | function | Build a run-level data specification from a canonical panel. |
| `validate_panel` | function | Validate macroforecast's canonical panel contract. |
| `align_frequency` | function | Keep, filter, or align a panel to a common data frequency. |
| `availability_lag` | function | Delay selected columns to match an information-availability policy. |
| `chow_lin_disaggregate` | function | Disaggregate a low-frequency series with a high-frequency indicator. |
| `combine` | function | Combine already-loaded data bundles into one canonical panel. |
| `define_regime` | function | Attach a binary regime series to panel metadata. |
| `frequency_hardening_issues` | function | Return frequency-classification issues that should be surfaced. |
| `infer_frequencies` | function | Infer or read native frequency by panel column. |
| `load_fred_md` | function | Load FRED-MD as a canonical monthly ``DataBundle``. |
| `load_fred_qd` | function | No public docstring is available. |
| `load_fred_sd` | function | No public docstring is available. |
| `load_fred_md_sd` | function | Load FRED-MD plus FRED-SD as one canonical data bundle. |
| `load_fred_qd_sd` | function | Load FRED-QD plus FRED-SD as one canonical data bundle. |
| `load_custom_csv` | function | Load a user CSV into a canonical ``DataBundle``. |
| `load_custom_parquet` | function | No public docstring is available. |
| `list_vintages` | function | Return monthly vintage labels between ``start`` and ``end`` inclusive. |
| `fred_md_vintages` | function | Return a FRED-MD point-in-time source resolved by forecast origin. |
| `fred_qd_vintages` | function | Return a FRED-QD point-in-time source resolved by origin date. |
| `with_static_extras` | function | Join non-revised extra columns observable before each origin. |
| `same_period_predictors` | function | Apply a same-period predictor policy to a run-level data spec. |

## Callable And Class Reference

### DataBundle

Qualified name: `macroforecast.data.panel.DataBundle`

#### Signature

```python
macroforecast.data.DataBundle(panel: pd.DataFrame, metadata: dict[str, Any] = <factory>) -> None
```

#### Description

Canonical data payload: a pandas panel plus explicit metadata.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.data.DataBundle(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `panel` | `pd.DataFrame` | `required` |
| `metadata` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `attach` | `attach(self, stage: str, values: Mapping[str, Any]) -> DataBundle` | No public docstring is available. |
### DataSpec

Qualified name: `macroforecast.data.panel.DataSpec`

#### Signature

```python
macroforecast.data.DataSpec(panel: pd.DataFrame, metadata: dict[str, Any], target: str | None, targets: tuple[str, ...], horizons: tuple[int, ...], start: str | None = None, end: str | None = None, predictors: PredictorSelection = "all") -> None
```

#### Description

Panel plus target, horizon, sample, and predictor choices for a run.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `required` |
| `target` | positional or keyword | `str \| None` | `required` |
| `targets` | positional or keyword | `tuple[str, ...]` | `required` |
| `horizons` | positional or keyword | `tuple[int, ...]` | `required` |
| `start` | positional or keyword | `str \| None` | `None` |
| `end` | positional or keyword | `str \| None` | `None` |
| `predictors` | positional or keyword | `PredictorSelection` | `"all"` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.data.DataSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `panel` | `pd.DataFrame` | `required` |
| `metadata` | `dict[str, Any]` | `required` |
| `target` | `str \| None` | `required` |
| `targets` | `tuple[str, ...]` | `required` |
| `horizons` | `tuple[int, ...]` | `required` |
| `start` | `str \| None` | `None` |
| `end` | `str \| None` | `None` |
| `predictors` | `PredictorSelection` | `"all"` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `attach` | `attach(self, stage: str, values: Mapping[str, Any]) -> DataSpec` | No public docstring is available. |
### RegimeDirection

Qualified name: `typing.Literal`

#### Signature

```python
macroforecast.data.RegimeDirection(*args, **kwargs)
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `unspecified` | `required` |
| `kwargs` | var keyword | `unspecified` | `required` |

#### Returns

See the description and object-specific contract.

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.RegimeDirection(...)
```
### SamePeriodPolicy

Qualified name: `typing.Literal`

#### Signature

```python
macroforecast.data.SamePeriodPolicy(*args, **kwargs)
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `unspecified` | `required` |
| `kwargs` | var keyword | `unspecified` | `required` |

#### Returns

See the description and object-specific contract.

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.SamePeriodPolicy(...)
```
### VintagePanelSpec

Qualified name: `macroforecast.data.vintage.VintagePanelSpec`

#### Signature

```python
macroforecast.data.VintagePanelSpec(source: VintageSource, reference_calendar: pd.DatetimeIndex, actuals_vintage: "Literal['latest', 'first_release']" = "latest", first_release_max_vintages: int = 12) -> None
```

#### Description

Run-level wrapper for a point-in-time vintage source.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `source` | positional or keyword | `VintageSource` | `required` |
| `reference_calendar` | positional or keyword | `pd.DatetimeIndex` | `required` |
| `actuals_vintage` | positional or keyword | `Literal['latest', 'first_release']` | `"latest"` |
| `first_release_max_vintages` | positional or keyword | `int` | `12` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.data.VintagePanelSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `source` | `VintageSource` | `required` |
| `reference_calendar` | `pd.DatetimeIndex` | `required` |
| `actuals_vintage` | `Literal['latest', 'first_release']` | `"latest"` |
| `first_release_max_vintages` | `int` | `12` |
### VintageSource

Qualified name: `macroforecast.data.vintage.VintageSource`

#### Signature

```python
macroforecast.data.VintageSource(*args, **kwargs)
```

#### Description

A lazily-resolved source of point-in-time data, one bundle per real-time origin.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `unspecified` | `required` |
| `kwargs` | var keyword | `unspecified` | `required` |

#### Returns

See the description and object-specific contract.

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.data.VintageSource(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `available_vintages` | `available_vintages(self) -> Sequence[Any]` | Return sorted canonical vintage identifiers this source can resolve. |
| `resolve` | `resolve(self, origin_date: pd.Timestamp) -> DataBundle` | Return the DataBundle observable as of ``origin_date``. |
### VintageUnavailableError

Qualified name: `macroforecast.data.errors.VintageUnavailableError`

#### Signature

```python
macroforecast.data.VintageUnavailableError(...)
```

#### Description

Raised when no point-in-time vintage is available for an origin.

#### Returns

See the description and object-specific contract.

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.data.VintageUnavailableError(...)
```
### as_panel

Qualified name: `macroforecast.data.panel.as_panel`

#### Signature

```python
macroforecast.data.as_panel(frame: pd.DataFrame, *, date: str | None = None, columns: Iterable[str] | None = None, rename: Mapping[str, str] | None = None, metadata: Mapping[str, Any] | None = None, strict: bool = True) -> pd.DataFrame
```

#### Description

Return ``frame`` as macroforecast's canonical date-indexed panel.

``strict=True`` is intentional. A forecasting panel should not silently
lose rows because date parsing failed, nor should string cells such as
``"missing"`` become ``NaN`` without the caller noticing. Official FRED
files use real missing-value markers that are already parsed upstream; this
guard is mainly for custom CSV/Parquet inputs and ad hoc DataFrames.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `frame` | positional or keyword | `pd.DataFrame` | `required` |
| `date` | keyword only | `str \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `strict` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.as_panel(...)
```
### attach_metadata

Qualified name: `macroforecast.data.panel.attach_metadata`

#### Signature

```python
macroforecast.data.attach_metadata(metadata: Mapping[str, Any], stage: str, values: Mapping[str, Any]) -> dict[str, Any]
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `metadata` | positional or keyword | `Mapping[str, Any]` | `required` |
| `stage` | positional or keyword | `str` | `required` |
| `values` | positional or keyword | `Mapping[str, Any]` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.attach_metadata(...)
```
### custom_dataset

Qualified name: `macroforecast.data.panel.custom_dataset`

#### Signature

```python
macroforecast.data.custom_dataset(frame: pd.DataFrame, *, date: str | None = None, columns: Iterable[str] | None = None, rename: Mapping[str, str] | None = None, dataset: str = "custom", source_family: str = "custom", frequency: str = "unknown", frequency_by_column: Mapping[str, str] | None = None, transform_codes: Mapping[str, int] | None = None, metadata: Mapping[str, Any] | None = None, strict: bool = True) -> DataBundle
```

#### Description

Build a canonical custom ``DataBundle`` from an in-memory DataFrame.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `frame` | positional or keyword | `pd.DataFrame` | `required` |
| `date` | keyword only | `str \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `dataset` | keyword only | `str` | `"custom"` |
| `source_family` | keyword only | `str` | `"custom"` |
| `frequency` | keyword only | `str` | `"unknown"` |
| `frequency_by_column` | keyword only | `Mapping[str, str] \| None` | `None` |
| `transform_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `strict` | keyword only | `bool` | `True` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.custom_dataset(...)
```
### custom_vintages

Qualified name: `macroforecast.data.vintage.custom_vintages`

#### Signature

```python
macroforecast.data.custom_vintages(source: Callable[[pd.Timestamp], DataBundle | pd.DataFrame] | Mapping[Any, DataBundle | pd.DataFrame] | pd.DataFrame, *, vintage_column: str | None = None, date_column: str | None = None, vintage_id: Callable[[Any], Any] | None = None, dataset: str = "custom_vintages", frequency: str = "unknown", strict: bool = True) -> VintageSource
```

#### Description

Return a custom point-in-time source.

``source`` may be a callable ``origin_date -> DataBundle | DataFrame``, a
mapping from timestamp-parsable vintage keys to snapshots, or a grouped-wide
DataFrame. The grouped-wide form has one ``vintage_column``, one
``date_column``, and one numeric column per series; each vintage group is a
complete wide snapshot. Every snapshot is normalized through
:func:`as_panel` / :func:`custom_dataset` and then validated. Resolved
snapshots are memoized by the stable identifier produced by ``vintage_id``
(default: ``str(resolved_key)``). If a callable reads from a
non-deterministic source whose content can change for the same identifier,
run the forecast with runner/pipeline preprocessing caching disabled.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `source` | positional or keyword | `Callable[[pd.Timestamp], DataBundle \| pd.DataFrame] \| Mapping[Any, DataBundle \| pd.DataFrame] \| pd.DataFrame` | `required` |
| `vintage_column` | keyword only | `str \| None` | `None` |
| `date_column` | keyword only | `str \| None` | `None` |
| `vintage_id` | keyword only | `Callable[[Any], Any] \| None` | `None` |
| `dataset` | keyword only | `str` | `"custom_vintages"` |
| `frequency` | keyword only | `str` | `"unknown"` |
| `strict` | keyword only | `bool` | `True` |

#### Returns

`VintageSource`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.custom_vintages(...)
```
### metadata

Qualified name: `macroforecast.data.panel.metadata`

#### Signature

```python
macroforecast.data.metadata(obj: PanelInput) -> dict[str, Any]
```

#### Description

Return metadata from a bundle, spec, tuple, or DataFrame.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `obj` | positional or keyword | `PanelInput` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.metadata(...)
```
### panel_info

Qualified name: `macroforecast.data.panel.panel_info`

#### Signature

```python
macroforecast.data.panel_info(panel: PanelInput) -> dict[str, Any]
```

#### Description

Return a compact diagnostic summary for a canonical panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `PanelInput` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.panel_info(...)
```
### set_frequencies

Qualified name: `macroforecast.data.panel.set_frequencies`

#### Signature

```python
macroforecast.data.set_frequencies(data: PanelInput, frequency_by_column: Mapping[str, str], *, default_frequency: str | None = None, output_frequency_by_column: Mapping[str, str] | None = None, frequency: str | None = None, metadata: Mapping[str, Any] | None = None) -> DataBundle
```

#### Description

Attach a column-level frequency contract to a panel or bundle.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PanelInput` | `required` |
| `frequency_by_column` | positional or keyword | `Mapping[str, str]` | `required` |
| `default_frequency` | keyword only | `str \| None` | `None` |
| `output_frequency_by_column` | keyword only | `Mapping[str, str] \| None` | `None` |
| `frequency` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.set_frequencies(...)
```
### spec

Qualified name: `macroforecast.data.panel.spec`

#### Signature

```python
macroforecast.data.spec(data: PanelInput, *, metadata: Mapping[str, Any] | None = None, target: str | None = None, targets: Iterable[str] | None = None, horizons: Iterable[int] | int | None = None, start: str | None = None, end: str | None = None, predictors: "Literal['all'] | Iterable[str]" = "all") -> DataSpec
```

#### Description

Build a run-level data specification from a canonical panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PanelInput` | `required` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `targets` | keyword only | `Iterable[str] \| None` | `None` |
| `horizons` | keyword only | `Iterable[int] \| int \| None` | `None` |
| `start` | keyword only | `str \| None` | `None` |
| `end` | keyword only | `str \| None` | `None` |
| `predictors` | keyword only | `Literal['all'] \| Iterable[str]` | `"all"` |

#### Returns

`DataSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.spec(...)
```
### validate_panel

Qualified name: `macroforecast.data.panel.validate_panel`

#### Signature

```python
macroforecast.data.validate_panel(panel: pd.DataFrame) -> None
```

#### Description

Validate macroforecast's canonical panel contract.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.validate_panel(...)
```
### align_frequency

Qualified name: `macroforecast.data.policies.align_frequency`

#### Signature

```python
macroforecast.data.align_frequency(data: PanelInput, *, method: str = "keep", quarterly_to_monthly: str = "step_forward", weekly_to_monthly: str = "mean", monthly_to_quarterly: str = "quarterly_average", weekly_to_quarterly: str = "mean", chow_lin_indicator: str | Mapping[str, str] | None = None, chow_lin_aggregation: str = "mean", chow_lin_rho: float | None = None, chow_lin_rho_method: str = "fixed") -> DataBundle
```

#### Description

Keep, filter, or align a panel to a common data frequency.

This is a data-level callable because it changes the panel's calendar and
column frequency contract. Statistical cleaning such as t-code transforms,
outlier handling, imputation, and standardization stays in
``macroforecast.preprocessing``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PanelInput` | `required` |
| `method` | keyword only | `str` | `"keep"` |
| `quarterly_to_monthly` | keyword only | `str` | `"step_forward"` |
| `weekly_to_monthly` | keyword only | `str` | `"mean"` |
| `monthly_to_quarterly` | keyword only | `str` | `"quarterly_average"` |
| `weekly_to_quarterly` | keyword only | `str` | `"mean"` |
| `chow_lin_indicator` | keyword only | `str \| Mapping[str, str] \| None` | `None` |
| `chow_lin_aggregation` | keyword only | `str` | `"mean"` |
| `chow_lin_rho` | keyword only | `float \| None` | `None` |
| `chow_lin_rho_method` | keyword only | `str` | `"fixed"` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.align_frequency(...)
```
### availability_lag

Qualified name: `macroforecast.data.policies.availability_lag`

#### Signature

```python
macroforecast.data.availability_lag(data: PanelInput, *, lags: int | Mapping[str, int] = 1, columns: Iterable[str] | None = None, drop_missing: bool = False) -> DataBundle
```

#### Description

Delay selected columns to match an information-availability policy.

A positive lag means the value dated ``t`` is treated as usable only from
later forecast origins. For example, ``lags=1`` shifts ``x[t-1]`` onto row
``t``. This is the direct callable replacement for the old release-lag
data policy module; release calendars can be expressed by passing a per-column
lag mapping.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PanelInput` | `required` |
| `lags` | keyword only | `int \| Mapping[str, int]` | `1` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.availability_lag(...)
```
### chow_lin_disaggregate

Qualified name: `macroforecast.data.policies.chow_lin_disaggregate`

#### Signature

```python
macroforecast.data.chow_lin_disaggregate(low_frequency: pd.Series, indicator: pd.Series | pd.DataFrame, *, aggregation: str = "mean", rho: float | None = None, rho_method: str = "fixed") -> pd.Series
```

#### Description

Disaggregate a low-frequency series with a high-frequency indicator.

This implements the standard Chow-Lin regression-distribution identity with
an AR(1) high-frequency residual covariance. The returned high-frequency
series conserves the supplied low-frequency observations under
``aggregation='mean'`` or ``aggregation='sum'``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `low_frequency` | positional or keyword | `pd.Series` | `required` |
| `indicator` | positional or keyword | `pd.Series \| pd.DataFrame` | `required` |
| `aggregation` | keyword only | `str` | `"mean"` |
| `rho` | keyword only | `float \| None` | `None` |
| `rho_method` | keyword only | `str` | `"fixed"` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.chow_lin_disaggregate(...)
```
### combine

Qualified name: `macroforecast.data.loaders.combine`

#### Signature

```python
macroforecast.data.combine(*bundles: DataBundle, dataset: str | None = None, frequency: str = "native", quarterly_to_monthly: str = "repeat_within_quarter", monthly_to_quarterly: str = "quarterly_average") -> DataBundle
```

#### Description

Combine already-loaded data bundles into one canonical panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `bundles` | var positional | `DataBundle` | `required` |
| `dataset` | keyword only | `str \| None` | `None` |
| `frequency` | keyword only | `str` | `"native"` |
| `quarterly_to_monthly` | keyword only | `str` | `"repeat_within_quarter"` |
| `monthly_to_quarterly` | keyword only | `str` | `"quarterly_average"` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.combine(...)
```
### define_regime

Qualified name: `macroforecast.data.policies.define_regime`

#### Signature

```python
macroforecast.data.define_regime(data: PanelInput, *, name: str = "regime", column: str | None = None, threshold: float | None = None, direction: RegimeDirection = "above", dates: Iterable[str | pd.Timestamp] | None = None, values: Sequence[bool | int | float] | pd.Series | None = None, append: bool = False, output_column: str | None = None) -> DataBundle
```

#### Description

Attach a binary regime series to panel metadata.

Regimes can be built from a threshold rule, explicit regime dates, or an
aligned vector/Series of values. The panel is unchanged unless
``append=True``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PanelInput` | `required` |
| `name` | keyword only | `str` | `"regime"` |
| `column` | keyword only | `str \| None` | `None` |
| `threshold` | keyword only | `float \| None` | `None` |
| `direction` | keyword only | `RegimeDirection` | `"above"` |
| `dates` | keyword only | `Iterable[str \| pd.Timestamp] \| None` | `None` |
| `values` | keyword only | `Sequence[bool \| int \| float] \| pd.Series \| None` | `None` |
| `append` | keyword only | `bool` | `False` |
| `output_column` | keyword only | `str \| None` | `None` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.define_regime(...)
```
### frequency_hardening_issues

Qualified name: `macroforecast.data.policies.frequency_hardening_issues`

#### Signature

```python
macroforecast.data.frequency_hardening_issues(frequencies: Mapping[str, str]) -> list[dict[str, Any]]
```

#### Description

Return frequency-classification issues that should be surfaced.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `frequencies` | positional or keyword | `Mapping[str, str]` | `required` |

#### Returns

`list[dict[str, Any]]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.frequency_hardening_issues(...)
```
### infer_frequencies

Qualified name: `macroforecast.data.policies.infer_frequencies`

#### Signature

```python
macroforecast.data.infer_frequencies(data: PanelInput | pd.DataFrame) -> tuple[dict[str, str], str]
```

#### Description

Infer or read native frequency by panel column.

Metadata from ``set_frequencies`` / ``combine(..., frequency="native")`` is
preferred, then FRED-SD series reports, then observed-date spacing.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `PanelInput \| pd.DataFrame` | `required` |

#### Returns

`tuple[dict[str, str], str]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.infer_frequencies(...)
```
### load_fred_md

Qualified name: `macroforecast.data.loaders.load_fred_md`

#### Signature

```python
macroforecast.data.load_fred_md(vintage: str | None = None, *, force: bool = False, cache_root: str | Path | None = None, local_source: str | Path | None = None, local_zip_source: str | Path | None = None) -> DataBundle
```

#### Description

Load FRED-MD as a canonical monthly ``DataBundle``.

Parameters
vintage
    Vintage label in ``YYYY-MM`` form. ``None`` loads the current official
    CSV. A vintage request is resolved from the official historical-vintage
    archive unless ``local_zip_source`` supplies that archive locally.
force
    Redownload or recopy the raw CSV even when the cache target already
    exists.
cache_root
    Optional cache directory for raw FRED files. ``None`` uses the package
    cache policy.
local_source
    Local CSV file to copy into the cache instead of downloading. This is
    for deterministic tests and offline workflows.
local_zip_source
    Local official historical-vintage ZIP file used with an explicit
    ``vintage``.

Returns
DataBundle
    Canonical date-indexed panel plus metadata containing dataset,
    frequency, source, and vintage information.

Example
>>> import macroforecast as mf
>>> bundle = mf.data.load_fred_md(vintage="2020-01")
>>> bundle.panel.index.name
'date'

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `vintage` | positional or keyword | `str \| None` | `None` |
| `force` | keyword only | `bool` | `False` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `local_source` | keyword only | `str \| Path \| None` | `None` |
| `local_zip_source` | keyword only | `str \| Path \| None` | `None` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.load_fred_md(...)
```
### load_fred_qd

Qualified name: `macroforecast.data.loaders.load_fred_qd`

#### Signature

```python
macroforecast.data.load_fred_qd(vintage: str | None = None, *, force: bool = False, cache_root: str | Path | None = None, local_source: str | Path | None = None, local_zip_source: str | Path | None = None) -> DataBundle
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `vintage` | positional or keyword | `str \| None` | `None` |
| `force` | keyword only | `bool` | `False` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `local_source` | keyword only | `str \| Path \| None` | `None` |
| `local_zip_source` | keyword only | `str \| Path \| None` | `None` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.load_fred_qd(...)
```
### load_fred_sd

Qualified name: `macroforecast.data.loaders.load_fred_sd`

#### Signature

```python
macroforecast.data.load_fred_sd(vintage: str | None = None, *, force: bool = False, cache_root: str | Path | None = None, local_source: str | Path | None = None, states: list[str] | None = None, variables: list[str] | None = None) -> DataBundle
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `vintage` | positional or keyword | `str \| None` | `None` |
| `force` | keyword only | `bool` | `False` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `local_source` | keyword only | `str \| Path \| None` | `None` |
| `states` | keyword only | `list[str] \| None` | `None` |
| `variables` | keyword only | `list[str] \| None` | `None` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.load_fred_sd(...)
```
### load_fred_md_sd

Qualified name: `macroforecast.data.loaders.load_fred_md_sd`

#### Signature

```python
macroforecast.data.load_fred_md_sd(vintage: str | None = None, *, force: bool = False, cache_root: str | Path | None = None, local_fred_md_source: str | Path | None = None, local_fred_sd_source: str | Path | None = None, states: list[str] | None = None, variables: list[str] | None = None, frequency: str = "monthly", quarterly_to_monthly: str = "repeat_within_quarter", monthly_to_quarterly: str = "quarterly_average") -> DataBundle
```

#### Description

Load FRED-MD plus FRED-SD as one canonical data bundle.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `vintage` | positional or keyword | `str \| None` | `None` |
| `force` | keyword only | `bool` | `False` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `local_fred_md_source` | keyword only | `str \| Path \| None` | `None` |
| `local_fred_sd_source` | keyword only | `str \| Path \| None` | `None` |
| `states` | keyword only | `list[str] \| None` | `None` |
| `variables` | keyword only | `list[str] \| None` | `None` |
| `frequency` | keyword only | `str` | `"monthly"` |
| `quarterly_to_monthly` | keyword only | `str` | `"repeat_within_quarter"` |
| `monthly_to_quarterly` | keyword only | `str` | `"quarterly_average"` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.load_fred_md_sd(...)
```
### load_fred_qd_sd

Qualified name: `macroforecast.data.loaders.load_fred_qd_sd`

#### Signature

```python
macroforecast.data.load_fred_qd_sd(vintage: str | None = None, *, force: bool = False, cache_root: str | Path | None = None, local_fred_qd_source: str | Path | None = None, local_fred_sd_source: str | Path | None = None, states: list[str] | None = None, variables: list[str] | None = None, frequency: str = "quarterly", quarterly_to_monthly: str = "repeat_within_quarter", monthly_to_quarterly: str = "quarterly_average") -> DataBundle
```

#### Description

Load FRED-QD plus FRED-SD as one canonical data bundle.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `vintage` | positional or keyword | `str \| None` | `None` |
| `force` | keyword only | `bool` | `False` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `local_fred_qd_source` | keyword only | `str \| Path \| None` | `None` |
| `local_fred_sd_source` | keyword only | `str \| Path \| None` | `None` |
| `states` | keyword only | `list[str] \| None` | `None` |
| `variables` | keyword only | `list[str] \| None` | `None` |
| `frequency` | keyword only | `str` | `"quarterly"` |
| `quarterly_to_monthly` | keyword only | `str` | `"repeat_within_quarter"` |
| `monthly_to_quarterly` | keyword only | `str` | `"quarterly_average"` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.load_fred_qd_sd(...)
```
### load_custom_csv

Qualified name: `macroforecast.data.loaders.load_custom_csv`

#### Signature

```python
macroforecast.data.load_custom_csv(path: str | Path, *, date: str | None = None, date_col: str | int | None = None, columns: Iterable[str] | None = None, series_columns: Iterable[str] | None = None, rename: Mapping[str, str] | None = None, dataset: str = "custom", frequency: str = "unknown", frequency_by_column: Mapping[str, str] | None = None, default_frequency: str | None = None, metadata: Mapping[str, Any] | None = None, transform_codes: Mapping[str, int] | None = None, cache_root: str | Path | None = None, strict: bool = True, na_values: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None, date_format: str | None = None, dayfirst: bool = False) -> DataBundle
```

#### Description

Load a user CSV into a canonical ``DataBundle``.

By default the CSV is read with pandas' normal parsing behavior and then
normalized through :func:`as_panel`. ``na_values`` is passed to
:func:`pandas.read_csv` when supplied. ``date_format`` and ``dayfirst`` parse
the resolved date column before panel normalization, which is useful for
strict loads with non-ISO or ambiguous dates.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `path` | positional or keyword | `str \| Path` | `required` |
| `date` | keyword only | `str \| None` | `None` |
| `date_col` | keyword only | `str \| int \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `series_columns` | keyword only | `Iterable[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `dataset` | keyword only | `str` | `"custom"` |
| `frequency` | keyword only | `str` | `"unknown"` |
| `frequency_by_column` | keyword only | `Mapping[str, str] \| None` | `None` |
| `default_frequency` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `transform_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `strict` | keyword only | `bool` | `True` |
| `na_values` | keyword only | `str \| Sequence[str] \| Mapping[str, str \| Sequence[str]] \| None` | `None` |
| `date_format` | keyword only | `str \| None` | `None` |
| `dayfirst` | keyword only | `bool` | `False` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.load_custom_csv(...)
```
### load_custom_parquet

Qualified name: `macroforecast.data.loaders.load_custom_parquet`

#### Signature

```python
macroforecast.data.load_custom_parquet(path: str | Path, *, date: str | None = None, date_col: str | int | None = None, columns: Iterable[str] | None = None, series_columns: Iterable[str] | None = None, rename: Mapping[str, str] | None = None, dataset: str = "custom", frequency: str = "unknown", frequency_by_column: Mapping[str, str] | None = None, default_frequency: str | None = None, metadata: Mapping[str, Any] | None = None, transform_codes: Mapping[str, int] | None = None, cache_root: str | Path | None = None, strict: bool = True) -> DataBundle
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `path` | positional or keyword | `str \| Path` | `required` |
| `date` | keyword only | `str \| None` | `None` |
| `date_col` | keyword only | `str \| int \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `series_columns` | keyword only | `Iterable[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `dataset` | keyword only | `str` | `"custom"` |
| `frequency` | keyword only | `str` | `"unknown"` |
| `frequency_by_column` | keyword only | `Mapping[str, str] \| None` | `None` |
| `default_frequency` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `transform_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `strict` | keyword only | `bool` | `True` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.load_custom_parquet(...)
```
### list_vintages

Qualified name: `macroforecast.data.loaders.list_vintages`

#### Signature

```python
macroforecast.data.list_vintages(dataset: str, start: str | None = None, end: str | None = None) -> list[str]
```

#### Description

Return monthly vintage labels between ``start`` and ``end`` inclusive.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `dataset` | positional or keyword | `str` | `required` |
| `start` | positional or keyword | `str \| None` | `None` |
| `end` | positional or keyword | `str \| None` | `None` |

#### Returns

`list[str]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.list_vintages(...)
```
### fred_md_vintages

Qualified name: `macroforecast.data.vintage.fred_md_vintages`

#### Signature

```python
macroforecast.data.fred_md_vintages(*, start: str | None = None, end: str | None = None, cache_root: str | Path | None = None, local_zip_source: str | Path | None = None, force: bool = False) -> VintageSource
```

#### Description

Return a FRED-MD point-in-time source resolved by forecast origin.

Parameters bound the available monthly vintage labels and cache/download
behavior. ``start`` and ``end`` use ``YYYY-MM`` labels. ``cache_root``
controls where raw vintage CSVs are stored. ``local_zip_source`` points to
an official historical-vintage ZIP for offline or deterministic runs.
``force=True`` refreshes cached vintage files.

Returns
VintageSource
    Source object with ``resolve(origin_date)`` and
    ``available_vintages()``. Resolving an origin returns the latest
    FRED-MD vintage available at or before that origin and raises
    ``VintageUnavailableError`` when no eligible vintage exists.

Example
>>> import pandas as pd
>>> import macroforecast as mf
>>> source = mf.data.fred_md_vintages(start="2020-01", end="2020-03")
>>> labels = source.available_vintages()
>>> isinstance(labels, tuple)
True

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `start` | keyword only | `str \| None` | `None` |
| `end` | keyword only | `str \| None` | `None` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `local_zip_source` | keyword only | `str \| Path \| None` | `None` |
| `force` | keyword only | `bool` | `False` |

#### Returns

`VintageSource`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.fred_md_vintages(...)
```
### fred_qd_vintages

Qualified name: `macroforecast.data.vintage.fred_qd_vintages`

#### Signature

```python
macroforecast.data.fred_qd_vintages(*, start: str | None = None, end: str | None = None, cache_root: str | Path | None = None, local_zip_source: str | Path | None = None, force: bool = False) -> VintageSource
```

#### Description

Return a FRED-QD point-in-time source resolved by origin date.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `start` | keyword only | `str \| None` | `None` |
| `end` | keyword only | `str \| None` | `None` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `local_zip_source` | keyword only | `str \| Path \| None` | `None` |
| `force` | keyword only | `bool` | `False` |

#### Returns

`VintageSource`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.fred_qd_vintages(...)
```
### with_static_extras

Qualified name: `macroforecast.data.vintage.with_static_extras`

#### Signature

```python
macroforecast.data.with_static_extras(source: VintageSource, extra: DataBundle | pd.DataFrame, *, join: "Literal['outer', 'inner', 'left']" = "outer") -> VintageSource
```

#### Description

Join non-revised extra columns observable before each origin.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `source` | positional or keyword | `VintageSource` | `required` |
| `extra` | positional or keyword | `DataBundle \| pd.DataFrame` | `required` |
| `join` | keyword only | `Literal['outer', 'inner', 'left']` | `"outer"` |

#### Returns

`VintageSource`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.with_static_extras(...)
```
### same_period_predictors

Qualified name: `macroforecast.data.policies.same_period_predictors`

#### Signature

```python
macroforecast.data.same_period_predictors(data: DataSpec, *, policy: SamePeriodPolicy = "allow", lag: int = 1, columns: Iterable[str] | None = None, drop_missing: bool = False) -> DataSpec
```

#### Description

Apply a same-period predictor policy to a run-level data spec.

``allow`` records that same-period predictors are intentionally allowed.
``lag`` shifts selected predictors by ``lag`` periods. ``drop`` removes
selected predictors from the spec. ``forbid`` raises when selected
same-period predictors are present. Targets are never shifted by this
helper.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `DataSpec` | `required` |
| `policy` | keyword only | `SamePeriodPolicy` | `"allow"` |
| `lag` | keyword only | `int` | `1` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `drop_missing` | keyword only | `bool` | `False` |

#### Returns

`DataSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.same_period_predictors(...)
```
