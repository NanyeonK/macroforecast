# macroforecast.filters

[Back to reference](index.md)

One-series filters, smoothers, decompositions, and adaptive moving averages.

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `AlbaMA` | class | Alias class for the Goulet Coulombe-Klieber AlbaMA smoother. |
| `AlbaMAResult` | class | Result returned by the AlbaMA adaptive moving-average feature builder. |
| `AdaptiveMovingAverage` | class | Reusable AlbaMA adaptive moving-average feature builder. |
| `FilterResult` | class | Result returned by direct one-series filter callables. |
| `albama` | function | Create Goulet Coulombe-Klieber AlbaMA features for one time series. |
| `hamilton_filter` | function | Apply Hamilton's trend-cycle regression filter to one series. |
| `hp_filter` | function | Apply the two-sided Hodrick-Prescott filter to one series. |
| `savitzky_golay` | function | Apply the centered Savitzky-Golay filter to one series. |
| `wavelet_filter` | function | Create causal rolling approximation/detail components for one series. |
| `stl_decompose` | function | Seasonal-Trend decomposition using Loess (STL). |

## Callable And Class Reference

### AlbaMA

Qualified name: `macroforecast.filters.albama.AlbaMA`

#### Signature

```python
macroforecast.filters.AlbaMA(*, mode: AlbaMAMode | str = "one_sided", n_estimators: int = 500, min_samples_leaf: int = 6, sample_fraction: float = 0.6, random_state: int | None = 42, replace: bool = True, inbag_rule: InbagRule | str = "single", min_train_size: int = 2) -> None
```

#### Description

Alias class for the Goulet Coulombe-Klieber AlbaMA smoother.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `mode` | keyword only | `AlbaMAMode \| str` | `"one_sided"` |
| `n_estimators` | keyword only | `int` | `500` |
| `min_samples_leaf` | keyword only | `int` | `6` |
| `sample_fraction` | keyword only | `float` | `0.6` |
| `random_state` | keyword only | `int \| None` | `42` |
| `replace` | keyword only | `bool` | `True` |
| `inbag_rule` | keyword only | `InbagRule \| str` | `"single"` |
| `min_train_size` | keyword only | `int` | `2` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.filters.AlbaMA(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, y: Any, *, dates: Any \| None = None, name: str \| None = None) -> "'AdaptiveMovingAverage'"` | No public docstring is available. |
| `fit_transform` | `fit_transform(self, y: Any, *, dates: Any \| None = None, name: str \| None = None) -> AlbaMAResult` | No public docstring is available. |
| `result` | `result(self) -> AlbaMAResult` | No public docstring is available. |
### AlbaMAResult

Qualified name: `macroforecast.filters.albama.AlbaMAResult`

#### Signature

```python
macroforecast.filters.AlbaMAResult(smoothed: pd.Series, weights: pd.DataFrame, mode: str, backend: str, params: dict[str, Any], metadata: dict[str, Any]) -> None
```

#### Description

Result returned by the AlbaMA adaptive moving-average feature builder.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `smoothed` | positional or keyword | `pd.Series` | `required` |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `mode` | positional or keyword | `str` | `required` |
| `backend` | positional or keyword | `str` | `required` |
| `params` | positional or keyword | `dict[str, Any]` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `required` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.filters.AlbaMAResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `smoothed` | `pd.Series` | `required` |
| `weights` | `pd.DataFrame` | `required` |
| `mode` | `str` | `required` |
| `backend` | `str` | `required` |
| `params` | `dict[str, Any]` | `required` |
| `metadata` | `dict[str, Any]` | `required` |
### AdaptiveMovingAverage

Qualified name: `macroforecast.filters.albama.AdaptiveMovingAverage`

#### Signature

```python
macroforecast.filters.AdaptiveMovingAverage(*, mode: AlbaMAMode | str = "one_sided", n_estimators: int = 500, min_samples_leaf: int = 6, sample_fraction: float = 0.6, random_state: int | None = 42, replace: bool = True, inbag_rule: InbagRule | str = "single", min_train_size: int = 2) -> None
```

#### Description

Reusable AlbaMA adaptive moving-average feature builder.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `mode` | keyword only | `AlbaMAMode \| str` | `"one_sided"` |
| `n_estimators` | keyword only | `int` | `500` |
| `min_samples_leaf` | keyword only | `int` | `6` |
| `sample_fraction` | keyword only | `float` | `0.6` |
| `random_state` | keyword only | `int \| None` | `42` |
| `replace` | keyword only | `bool` | `True` |
| `inbag_rule` | keyword only | `InbagRule \| str` | `"single"` |
| `min_train_size` | keyword only | `int` | `2` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.filters.AdaptiveMovingAverage(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, y: Any, *, dates: Any \| None = None, name: str \| None = None) -> "'AdaptiveMovingAverage'"` | No public docstring is available. |
| `fit_transform` | `fit_transform(self, y: Any, *, dates: Any \| None = None, name: str \| None = None) -> AlbaMAResult` | No public docstring is available. |
| `result` | `result(self) -> AlbaMAResult` | No public docstring is available. |
### FilterResult

Qualified name: `macroforecast.filters.core.FilterResult`

#### Signature

```python
macroforecast.filters.FilterResult(values: pd.DataFrame, method: str, params: dict[str, Any], metadata: dict[str, Any], source: str | None = None) -> None
```

#### Description

Result returned by direct one-series filter callables.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `values` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | positional or keyword | `str` | `required` |
| `params` | positional or keyword | `dict[str, Any]` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `required` |
| `source` | positional or keyword | `str \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.filters.FilterResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `values` | `pd.DataFrame` | `required` |
| `method` | `str` | `required` |
| `params` | `dict[str, Any]` | `required` |
| `metadata` | `dict[str, Any]` | `required` |
| `source` | `str \| None` | `None` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `component` | `component(self, name: str) -> pd.Series` | Return one named component from ``values``. |
### albama

Qualified name: `macroforecast.filters.albama.albama`

#### Signature

```python
macroforecast.filters.albama(y: Any, *, dates: Any | None = None, mode: AlbaMAMode | str = "one_sided", n_estimators: int = 500, min_samples_leaf: int = 6, sample_fraction: float = 0.6, random_state: int | None = 42, replace: bool = True, inbag_rule: InbagRule | str = "single", min_train_size: int = 2, name: str | None = None) -> AlbaMAResult
```

#### Description

Create Goulet Coulombe-Klieber AlbaMA features for one time series.

AlbaMA is a learned feature transform, not a multivariate forecast model.
It fits bagged CART trees with a deterministic time trend as the only
regressor. The output is the tree-averaged adaptive moving average plus a
date-by-date observation-weight matrix.

R alignment:
- Goulet Coulombe and Klieber's `AlbaMA/AMA_main.R` uses
  `ranger(MAIN ~ Time_Trend, keep.inbag = TRUE, terminalNodes)`.
- This port uses explicit `DecisionTreeRegressor` bagging so the in-bag
  counts needed for terminal-node co-membership weights are stored directly
  instead of relying on private sklearn RandomForest attributes.
- `mode="two_sided"` mirrors `Albama_center`; `mode="one_sided"` mirrors
  the recursive `Albama_right` loop.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y` | positional or keyword | `Any` | `required` |
| `dates` | keyword only | `Any \| None` | `None` |
| `mode` | keyword only | `AlbaMAMode \| str` | `"one_sided"` |
| `n_estimators` | keyword only | `int` | `500` |
| `min_samples_leaf` | keyword only | `int` | `6` |
| `sample_fraction` | keyword only | `float` | `0.6` |
| `random_state` | keyword only | `int \| None` | `42` |
| `replace` | keyword only | `bool` | `True` |
| `inbag_rule` | keyword only | `InbagRule \| str` | `"single"` |
| `min_train_size` | keyword only | `int` | `2` |
| `name` | keyword only | `str \| None` | `None` |

#### Returns

`AlbaMAResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.filters.albama(...)
```
### hamilton_filter

Qualified name: `macroforecast.filters.core.hamilton_filter`

#### Signature

```python
macroforecast.filters.hamilton_filter(y: Any, *, dates: Any | None = None, h: int = 8, p: int = 4, component: str = "both", fit_policy: str = "expanding", min_train_size: int | None = None, missing: str = "drop", name: str | None = None) -> FilterResult
```

#### Description

Apply Hamilton's trend-cycle regression filter to one series.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y` | positional or keyword | `Any` | `required` |
| `dates` | keyword only | `Any \| None` | `None` |
| `h` | keyword only | `int` | `8` |
| `p` | keyword only | `int` | `4` |
| `component` | keyword only | `str` | `"both"` |
| `fit_policy` | keyword only | `str` | `"expanding"` |
| `min_train_size` | keyword only | `int \| None` | `None` |
| `missing` | keyword only | `str` | `"drop"` |
| `name` | keyword only | `str \| None` | `None` |

#### Returns

`FilterResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.filters.hamilton_filter(...)
```
### hp_filter

Qualified name: `macroforecast.filters.core.hp_filter`

#### Signature

```python
macroforecast.filters.hp_filter(y: Any, *, dates: Any | None = None, lamb: float = 129600.0, component: str = "both", interpolate_missing: bool = True, name: str | None = None) -> FilterResult
```

#### Description

Apply the two-sided Hodrick-Prescott filter to one series.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y` | positional or keyword | `Any` | `required` |
| `dates` | keyword only | `Any \| None` | `None` |
| `lamb` | keyword only | `float` | `129600.0` |
| `component` | keyword only | `str` | `"both"` |
| `interpolate_missing` | keyword only | `bool` | `True` |
| `name` | keyword only | `str \| None` | `None` |

#### Returns

`FilterResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.filters.hp_filter(...)
```
### savitzky_golay

Qualified name: `macroforecast.filters.core.savitzky_golay`

#### Signature

```python
macroforecast.filters.savitzky_golay(y: Any, *, dates: Any | None = None, window_length: int = 5, polyorder: int = 2, derivative: int = 0, interpolate_missing: bool = True, name: str | None = None) -> FilterResult
```

#### Description

Apply the centered Savitzky-Golay filter to one series.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y` | positional or keyword | `Any` | `required` |
| `dates` | keyword only | `Any \| None` | `None` |
| `window_length` | keyword only | `int` | `5` |
| `polyorder` | keyword only | `int` | `2` |
| `derivative` | keyword only | `int` | `0` |
| `interpolate_missing` | keyword only | `bool` | `True` |
| `name` | keyword only | `str \| None` | `None` |

#### Returns

`FilterResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.filters.savitzky_golay(...)
```
### wavelet_filter

Qualified name: `macroforecast.filters.core.wavelet_filter`

#### Signature

```python
macroforecast.filters.wavelet_filter(y: Any, *, dates: Any | None = None, n_levels: int = 3, wavelet: str = "db4", name: str | None = None) -> FilterResult
```

#### Description

Create causal rolling approximation/detail components for one series.

This is the package's existing wavelet-style filter helper. It records the
requested wavelet name for provenance, but the current implementation is a
causal rolling multi-resolution approximation rather than a true DWT.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y` | positional or keyword | `Any` | `required` |
| `dates` | keyword only | `Any \| None` | `None` |
| `n_levels` | keyword only | `int` | `3` |
| `wavelet` | keyword only | `str` | `"db4"` |
| `name` | keyword only | `str \| None` | `None` |

#### Returns

`FilterResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.filters.wavelet_filter(...)
```
### stl_decompose

Qualified name: `macroforecast.filters.core.stl_decompose`

#### Signature

```python
macroforecast.filters.stl_decompose(y: Any, *, period: int | None = None, seasonal: int = 7, trend: int | None = None, robust: bool = False, dates: Any | None = None, name: str | None = None) -> FilterResult
```

#### Description

Seasonal-Trend decomposition using Loess (STL).

Decomposes a single series into trend, seasonal and remainder components
(Cleveland et al. 1990; R ``stats::stl`` / statsmodels ``STL``). ``period`` is
the seasonal period (inferred from a monthly/quarterly/weekly DatetimeIndex
when omitted). This is a two-sided full-sample decomposition, so the
components are not real-time safe as forecasting features without a per-origin
refit (``fit_policy='full_input_two_sided'``).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y` | positional or keyword | `Any` | `required` |
| `period` | keyword only | `int \| None` | `None` |
| `seasonal` | keyword only | `int` | `7` |
| `trend` | keyword only | `int \| None` | `None` |
| `robust` | keyword only | `bool` | `False` |
| `dates` | keyword only | `Any \| None` | `None` |
| `name` | keyword only | `str \| None` | `None` |

#### Returns

`FilterResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.filters.stl_decompose(...)
```
