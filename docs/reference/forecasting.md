# macroforecast.forecasting

[Back to reference](index.md)

Single-model forecasting runner, forecast result objects, checkpoint helpers, and forecast-combination specs.

Guide context: [../guide/concepts/running.md](../guide/concepts/running.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `ForecastResult` | class | Forecast runner output. |
| `LEAN_FORECAST_COLUMNS` | data | Built-in immutable sequence. |
| `SELECTION_HISTORY_COLUMNS` | data | Built-in immutable sequence. |
| `load_checkpoint_frame` | function | Load all persisted lean records as a single frame (empty if none/missing). |
| `load_selection_history_frame` | function | Load optional selection-history JSONL sidecars from one checkpoint dir. |
| `CombinationSpec` | class | Forecast-combination request consumed by ``forecasting.run``. |
| `combine_best_n` | function | Average the historically best ``n`` models by MSPE. |
| `combine_bates_granger` | function | Bates-Granger (1969) minimum error-variance combination (full covariance). |
| `combine_granger_ramanathan` | function | Granger-Ramanathan (1984) regression combination. |
| `combine_constrained_ls` | function | Non-negative weights summing to one minimising squared combination error. |
| `combine_eigenvector` | function | Eigenvector (principal-component) combination (Hsiao-Wan). |
| `combine_regularized` | function | Ridge/Lasso-penalised regression combination (high-dimensional weights). |
| `combine_linear_pool` | function | Linear opinion pool of (Gaussian) density forecasts. |
| `combine_log_pool` | function | Logarithmic opinion pool of Gaussian density forecasts. |
| `combine_dmspe` | function | Combine forecasts with inverse discounted MSPE weights. |
| `combine_inverse_mspe` | function | Combine forecasts with inverse discounted MSPE weights. |
| `combine_mean` | function | Equal-weight average forecast. |
| `combine_median` | function | Cross-model median forecast. |
| `combine_trimmed_mean` | function | Trim extreme model forecasts before averaging. |
| `combine_winsorized_mean` | function | Winsorize cross-model forecasts before averaging. |
| `combination_spec` | function | Build a runner-compatible forecast-combination spec. |
| `custom_combination` | function | Build a custom forecast-combination spec for ``forecasting.run``. |
| `run` | function | Run a windowed macro forecasting experiment. |
| `run_forecast` | function | Run a windowed macro forecasting experiment. |

## Data And Module Values

### `LEAN_FORECAST_COLUMNS`

Kind: `data`

```python
LEAN_FORECAST_COLUMNS = ("target", "horizon", "origin", "origin_pos", "date", "model", "prediction", "actual", "forecast_policy", "target_transform", "variance_prediction", "vintage_id", "actuals_vintage_id")
```
### `SELECTION_HISTORY_COLUMNS`

Kind: `data`

```python
SELECTION_HISTORY_COLUMNS = ("target", "arm", "horizon", "origin", "origin_pos", "date", "kind", "name", "value", "model", "step", "method", "score", "source")
```

## Callable And Class Reference

### ForecastResult

Qualified name: `macroforecast.forecasting.types.ForecastResult`

#### Signature

```python
macroforecast.forecasting.ForecastResult(forecasts: pd.DataFrame, metadata: dict[str, Any] = <factory>, sidecars: dict[str, Any] = <factory>) -> None
```

#### Description

Forecast runner output.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `pd.DataFrame` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `sidecars` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.forecasting.ForecastResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `forecasts` | `pd.DataFrame` | `required` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `sidecars` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `anatomy_explain` | `anatomy_explain(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame` | Explain a precomputed ``anatomy.Anatomy`` object for this run. |
| `anatomy_oshapley_vi` | `anatomy_oshapley_vi(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame` | Compute backend oShapley-VI rows for a precomputed anatomy object. |
| `anatomy_pbsv` | `anatomy_pbsv(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame` | Compute backend PBSV rows for a precomputed ``anatomy.Anatomy`` object. |
| `evaluate` | `evaluate(self, **kwargs: Any) -> pd.DataFrame` | Evaluate this forecast result with ``macroforecast.metrics``. |
| `get_sidecar` | `get_sidecar(self, name: str, default: Any = None) -> Any` | Return a named sidecar, or ``default`` when it is absent. |
| `oshapley_vi` | `oshapley_vi(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame` | Compute oShapley-VI rows for a precomputed forecast-Shapley object. |
| `pbsv` | `pbsv(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame` | Compute PBSV rows for a precomputed forecast-Shapley backend object. |
| `sidecar_names` | `sidecar_names(self) -> tuple[str, ...]` | Return attached sidecar names. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return a JSON-ready forecast result. |
| `to_frame` | `to_frame(self) -> pd.DataFrame` | Return a copy of the forecast table. |
| `to_json` | `to_json(self, path: str \| Path \| None = None, *, indent: int \| None = 2) -> str` | Return JSON text, and optionally write it to ``path``. |
| `with_anatomy` | `with_anatomy(self, X: Any, y: Any, models: Any, *, window: Any, sidecar_name: str = "anatomy", **kwargs: Any) -> "'ForecastResult'"` | Build and attach a forecast-accuracy anatomy sidecar. |
| `with_dual` | `with_dual(self, model: Any \| None, X_train: Any, y_train: Any, X_test: Any \| None = None, *, sidecar_name: str = "dual", **kwargs: Any) -> "'ForecastResult'"` | Build and attach a dual interpretation sidecar. |
| `with_oshapley` | `with_oshapley(self, X: Any, y: Any, models: Any, *, window: Any, sidecar_name: str = "oshapley", **kwargs: Any) -> "'ForecastResult'"` | Build and attach an oShapley/PBSV forecast-accuracy sidecar. |
| `with_sidecar` | `with_sidecar(self, name: str, value: Any) -> "'ForecastResult'"` | Return a copy with a named runtime sidecar attached. |
### load_checkpoint_frame

Qualified name: `macroforecast.forecasting.checkpoint.load_checkpoint_frame`

#### Signature

```python
macroforecast.forecasting.load_checkpoint_frame(checkpoint_path: str | Path) -> pd.DataFrame
```

#### Description

Load all persisted lean records as a single frame (empty if none/missing).

Origin files may carry different wide ``q_<pct>`` quantile columns (a
point-only or pre-density-pipeline origin has none); ``pd.concat`` unions
them, filling gaps with NaN, so this needs no cross-file coordination. When
any ``q_<pct>`` column is present, a ``quantile_predictions`` mapping column
is additionally synthesized (the wide columns are kept alongside it, not
dropped) so this frame's quantile representation matches the rich
(non-checkpointed) forecast table's -- one ``{level_str: value}`` dict per
row -- and every downstream consumer (``evaluate()``'s density stage,
``rescore()``) can use the SAME dict-based dispatch regardless of whether
the forecasts came from a live run or a checkpoint.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `checkpoint_path` | positional or keyword | `str \| Path` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.load_checkpoint_frame(...)
```
### load_selection_history_frame

Qualified name: `macroforecast.forecasting.checkpoint.load_selection_history_frame`

#### Signature

```python
macroforecast.forecasting.load_selection_history_frame(checkpoint_path: str | Path) -> pd.DataFrame
```

#### Description

Load optional selection-history JSONL sidecars from one checkpoint dir.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `checkpoint_path` | positional or keyword | `str \| Path` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.load_selection_history_frame(...)
```
### CombinationSpec

Qualified name: `macroforecast.forecasting.combination.CombinationSpec`

#### Signature

```python
macroforecast.forecasting.CombinationSpec(method: str, name: str, models: tuple[str, ...] | None = None, params: dict[str, Any] = <factory>, func: Callable[..., Any] | None = None) -> None
```

#### Description

Forecast-combination request consumed by ``forecasting.run``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `method` | positional or keyword | `str` | `required` |
| `name` | positional or keyword | `str` | `required` |
| `models` | positional or keyword | `tuple[str, ...] \| None` | `None` |
| `params` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `func` | positional or keyword | `Callable[..., Any] \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.forecasting.CombinationSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `method` | `str` | `required` |
| `name` | `str` | `required` |
| `models` | `tuple[str, ...] \| None` | `None` |
| `params` | `dict[str, Any]` | `default_factory` |
| `func` | `Callable[..., Any] \| None` | `None` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### combine_best_n

Qualified name: `macroforecast.forecasting.combination.combine_best_n`

#### Signature

```python
macroforecast.forecasting.combine_best_n(forecasts: Any, y_true: Any, *, n: int = 3, horizon: int = 1) -> pd.Series
```

#### Description

Average the historically best ``n`` models by MSPE.

The ranking at each target date uses only errors observable at the forecast
origin, so the expanding MSPE is lagged by ``horizon`` target-date rows
(a 1-row lag for one-step forecasts).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `y_true` | positional or keyword | `Any` | `required` |
| `n` | keyword only | `int` | `3` |
| `horizon` | keyword only | `int` | `1` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_best_n(...)
```
### combine_bates_granger

Qualified name: `macroforecast.forecasting.combination.combine_bates_granger`

#### Signature

```python
macroforecast.forecasting.combine_bates_granger(forecasts: Any, y_true: Any, *, horizon: int = 1, min_periods: int = 10, window: int | None = None, shrink_to_equal: float | None = None) -> pd.Series
```

#### Description

Bates-Granger (1969) minimum error-variance combination (full covariance).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `y_true` | positional or keyword | `Any` | `required` |
| `horizon` | keyword only | `int` | `1` |
| `min_periods` | keyword only | `int` | `10` |
| `window` | keyword only | `int \| None` | `None` |
| `shrink_to_equal` | keyword only | `float \| None` | `None` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_bates_granger(...)
```
### combine_granger_ramanathan

Qualified name: `macroforecast.forecasting.combination.combine_granger_ramanathan`

#### Signature

```python
macroforecast.forecasting.combine_granger_ramanathan(forecasts: Any, y_true: Any, *, variant: str = "constrained", horizon: int = 1, min_periods: int = 10, window: int | None = None, shrink_to_equal: float | None = None) -> pd.Series
```

#### Description

Granger-Ramanathan (1984) regression combination.

``variant``: ``"ols"`` (with intercept), ``"no_intercept"``, or
``"constrained"`` (no intercept, weights sum to one).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `y_true` | positional or keyword | `Any` | `required` |
| `variant` | keyword only | `str` | `"constrained"` |
| `horizon` | keyword only | `int` | `1` |
| `min_periods` | keyword only | `int` | `10` |
| `window` | keyword only | `int \| None` | `None` |
| `shrink_to_equal` | keyword only | `float \| None` | `None` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_granger_ramanathan(...)
```
### combine_constrained_ls

Qualified name: `macroforecast.forecasting.combination.combine_constrained_ls`

#### Signature

```python
macroforecast.forecasting.combine_constrained_ls(forecasts: Any, y_true: Any, *, horizon: int = 1, min_periods: int = 10, window: int | None = None, shrink_to_equal: float | None = None) -> pd.Series
```

#### Description

Non-negative weights summing to one minimising squared combination error.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `y_true` | positional or keyword | `Any` | `required` |
| `horizon` | keyword only | `int` | `1` |
| `min_periods` | keyword only | `int` | `10` |
| `window` | keyword only | `int \| None` | `None` |
| `shrink_to_equal` | keyword only | `float \| None` | `None` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_constrained_ls(...)
```
### combine_eigenvector

Qualified name: `macroforecast.forecasting.combination.combine_eigenvector`

#### Signature

```python
macroforecast.forecasting.combine_eigenvector(forecasts: Any, y_true: Any, *, horizon: int = 1, min_periods: int = 10, window: int | None = None, shrink_to_equal: float | None = None) -> pd.Series
```

#### Description

Eigenvector (principal-component) combination (Hsiao-Wan).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `y_true` | positional or keyword | `Any` | `required` |
| `horizon` | keyword only | `int` | `1` |
| `min_periods` | keyword only | `int` | `10` |
| `window` | keyword only | `int \| None` | `None` |
| `shrink_to_equal` | keyword only | `float \| None` | `None` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_eigenvector(...)
```
### combine_regularized

Qualified name: `macroforecast.forecasting.combination.combine_regularized`

#### Signature

```python
macroforecast.forecasting.combine_regularized(forecasts: Any, y_true: Any, *, penalty: str = "ridge", alpha: float = 1.0, intercept: bool = True, horizon: int = 1, min_periods: int = 10, window: int | None = None, shrink_to_equal: float | None = None) -> pd.Series
```

#### Description

Ridge/Lasso-penalised regression combination (high-dimensional weights).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `y_true` | positional or keyword | `Any` | `required` |
| `penalty` | keyword only | `str` | `"ridge"` |
| `alpha` | keyword only | `float` | `1.0` |
| `intercept` | keyword only | `bool` | `True` |
| `horizon` | keyword only | `int` | `1` |
| `min_periods` | keyword only | `int` | `10` |
| `window` | keyword only | `int \| None` | `None` |
| `shrink_to_equal` | keyword only | `float \| None` | `None` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_regularized(...)
```
### combine_linear_pool

Qualified name: `macroforecast.forecasting.combination.combine_linear_pool`

#### Signature

```python
macroforecast.forecasting.combine_linear_pool(means: Any, sds: Any | None = None, *, weights: Any | None = None) -> pd.DataFrame
```

#### Description

Linear opinion pool of (Gaussian) density forecasts.

The combined density is the mixture ``sum_i w_i N(mu_i, sigma_i^2)``. ``means``
and ``sds`` are frames (rows = dates, cols = models); ``weights`` default to
equal. Returns a frame with the pooled ``mean``, ``variance`` and ``sd``. The
mixture variance exceeds the weighted component variance, capturing model
disagreement. If ``sds`` is omitted the pool reduces to the weighted mean.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `means` | positional or keyword | `Any` | `required` |
| `sds` | positional or keyword | `Any \| None` | `None` |
| `weights` | keyword only | `Any \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_linear_pool(...)
```
### combine_log_pool

Qualified name: `macroforecast.forecasting.combination.combine_log_pool`

#### Signature

```python
macroforecast.forecasting.combine_log_pool(means: Any, sds: Any, *, weights: Any | None = None) -> pd.DataFrame
```

#### Description

Logarithmic opinion pool of Gaussian density forecasts.

The combined density is proportional to ``prod_i f_i^{w_i}``. For Gaussians the
pool is itself Gaussian with precision ``tau = sum_i w_i / sigma_i^2`` and mean
``(sum_i w_i mu_i / sigma_i^2) / tau``. The log pool is sharper (smaller
variance) than the linear pool. Returns a frame with pooled ``mean``,
``variance`` and ``sd``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `means` | positional or keyword | `Any` | `required` |
| `sds` | positional or keyword | `Any` | `required` |
| `weights` | keyword only | `Any \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_log_pool(...)
```
### combine_dmspe

Qualified name: `macroforecast.forecasting.combination.combine_inverse_mspe`

#### Signature

```python
macroforecast.forecasting.combine_dmspe(forecasts: Any, y_true: Any, *, discount: float = 1.0, min_weight: float = 1e-12, horizon: int = 1) -> pd.Series
```

#### Description

Combine forecasts with inverse discounted MSPE weights.

For an h-step forecast the weights at a target date are decided at the
origin (h target-date steps earlier), so only forecast errors realised on or
before that origin are observable. The error history is therefore lagged by
``horizon`` target-date rows (a 1-row lag for one-step forecasts), preventing
the use of not-yet-realised errors for multi-step combinations.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `y_true` | positional or keyword | `Any` | `required` |
| `discount` | keyword only | `float` | `1.0` |
| `min_weight` | keyword only | `float` | `1e-12` |
| `horizon` | keyword only | `int` | `1` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_dmspe(...)
```
### combine_inverse_mspe

Qualified name: `macroforecast.forecasting.combination.combine_inverse_mspe`

#### Signature

```python
macroforecast.forecasting.combine_inverse_mspe(forecasts: Any, y_true: Any, *, discount: float = 1.0, min_weight: float = 1e-12, horizon: int = 1) -> pd.Series
```

#### Description

Combine forecasts with inverse discounted MSPE weights.

For an h-step forecast the weights at a target date are decided at the
origin (h target-date steps earlier), so only forecast errors realised on or
before that origin are observable. The error history is therefore lagged by
``horizon`` target-date rows (a 1-row lag for one-step forecasts), preventing
the use of not-yet-realised errors for multi-step combinations.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `y_true` | positional or keyword | `Any` | `required` |
| `discount` | keyword only | `float` | `1.0` |
| `min_weight` | keyword only | `float` | `1e-12` |
| `horizon` | keyword only | `int` | `1` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_inverse_mspe(...)
```
### combine_mean

Qualified name: `macroforecast.forecasting.combination.combine_mean`

#### Signature

```python
macroforecast.forecasting.combine_mean(forecasts: Any) -> pd.Series
```

#### Description

Equal-weight average forecast.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_mean(...)
```
### combine_median

Qualified name: `macroforecast.forecasting.combination.combine_median`

#### Signature

```python
macroforecast.forecasting.combine_median(forecasts: Any) -> pd.Series
```

#### Description

Cross-model median forecast.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_median(...)
```
### combine_trimmed_mean

Qualified name: `macroforecast.forecasting.combination.combine_trimmed_mean`

#### Signature

```python
macroforecast.forecasting.combine_trimmed_mean(forecasts: Any, *, trim: float = 0.1) -> pd.Series
```

#### Description

Trim extreme model forecasts before averaging.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `trim` | keyword only | `float` | `0.1` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_trimmed_mean(...)
```
### combine_winsorized_mean

Qualified name: `macroforecast.forecasting.combination.combine_winsorized_mean`

#### Signature

```python
macroforecast.forecasting.combine_winsorized_mean(forecasts: Any, *, limits: tuple[float, float] = (0.1, 0.1)) -> pd.Series
```

#### Description

Winsorize cross-model forecasts before averaging.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `limits` | keyword only | `tuple[float, float]` | `(0.1, 0.1)` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combine_winsorized_mean(...)
```
### combination_spec

Qualified name: `macroforecast.forecasting.combination.combination_spec`

#### Signature

```python
macroforecast.forecasting.combination_spec(method: str, *, name: str | None = None, models: Sequence[str] | None = None, **params: Any) -> CombinationSpec
```

#### Description

Build a runner-compatible forecast-combination spec.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `method` | positional or keyword | `str` | `required` |
| `name` | keyword only | `str \| None` | `None` |
| `models` | keyword only | `Sequence[str] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`CombinationSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.combination_spec(...)
```
### custom_combination

Qualified name: `macroforecast.forecasting.combination.custom_combination`

#### Signature

```python
macroforecast.forecasting.custom_combination(name: str, func: Callable[..., Any], *, models: Sequence[str] | None = None, **params: Any) -> CombinationSpec
```

#### Description

Build a custom forecast-combination spec for ``forecasting.run``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `models` | keyword only | `Sequence[str] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`CombinationSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.custom_combination(...)
```
### run

Qualified name: `macroforecast.forecasting.runner.run`

#### Signature

```python
macroforecast.forecasting.run(data: Any, model: str | Callable[..., Any] | ModelSpec, *, window: WindowSpec | str | None = None, preprocessing: PreprocessSpec | None = None, preprocessing_policy: StagePolicy | str | None = None, features: FeatureSpec | None = None, feature_policy: StagePolicy | str | None = None, model_selection: SearchSpec | Mapping[str, SearchSpec | None] | None = None, model_selection_policy: StagePolicy | str | None = None, model_selection_metric: str | Callable[..., float] = "mse", maximize_model_selection: bool = False, preset: str | Mapping[str, str | None] | None = None, params: Mapping[str, Any] | None = None, target: str | None = None, horizon: int = 1, horizons: Sequence[int] | int | None = None, forecast_policy: str = "direct", future_feature_policy: str | None = None, target_transform: str | None = None, combination: str | CombinationSpec | Sequence[str | CombinationSpec | Mapping[str, Any]] | Mapping[str, Any] | None = None, save_models: bool = True, model_store: str | Path = "trained_model", preprocessing_cache: dict[Any, FittedPreprocessor | _PreparedStage | FittedFeatureBuilder] | None = None, preprocessing_store: PreprocessorStore | None = None, checkpoint_path: str | Path | None = None, selection_history: bool = False) -> ForecastResult
```

#### Description

Run a windowed macro forecasting experiment.

The runner composes small stage callables. ``window`` owns the temporal
design, stage policies decide where preprocessing, features, and model
selection are fitted, model specs fit predictors to targets, and the result
records a run-level metadata ledger.

A ``run`` is ATOMIC: it fits exactly ONE model. ``model`` must be a single
``str`` model name, a ``Callable`` model factory, or a ``ModelSpec`` (a
fit-time model-ensemble spec still counts as one model). Passing a sequence
or a mapping of models raises ``TypeError`` -- run one model per call, or use
the pipeline with one ``Arm`` per model when comparing models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `model` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |
| `window` | keyword only | `WindowSpec \| str \| None` | `None` |
| `preprocessing` | keyword only | `PreprocessSpec \| None` | `None` |
| `preprocessing_policy` | keyword only | `StagePolicy \| str \| None` | `None` |
| `features` | keyword only | `FeatureSpec \| None` | `None` |
| `feature_policy` | keyword only | `StagePolicy \| str \| None` | `None` |
| `model_selection` | keyword only | `SearchSpec \| Mapping[str, SearchSpec \| None] \| None` | `None` |
| `model_selection_policy` | keyword only | `StagePolicy \| str \| None` | `None` |
| `model_selection_metric` | keyword only | `str \| Callable[..., float]` | `"mse"` |
| `maximize_model_selection` | keyword only | `bool` | `False` |
| `preset` | keyword only | `str \| Mapping[str, str \| None] \| None` | `None` |
| `params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `horizon` | keyword only | `int` | `1` |
| `horizons` | keyword only | `Sequence[int] \| int \| None` | `None` |
| `forecast_policy` | keyword only | `str` | `"direct"` |
| `future_feature_policy` | keyword only | `str \| None` | `None` |
| `target_transform` | keyword only | `str \| None` | `None` |
| `combination` | keyword only | `str \| CombinationSpec \| Sequence[str \| CombinationSpec \| Mapping[str, Any]] \| Mapping[str, Any] \| None` | `None` |
| `save_models` | keyword only | `bool` | `True` |
| `model_store` | keyword only | `str \| Path` | `"trained_model"` |
| `preprocessing_cache` | keyword only | `dict[Any, FittedPreprocessor \| _PreparedStage \| FittedFeatureBuilder] \| None` | `None` |
| `preprocessing_store` | keyword only | `PreprocessorStore \| None` | `None` |
| `checkpoint_path` | keyword only | `str \| Path \| None` | `None` |
| `selection_history` | keyword only | `bool` | `False` |

#### Returns

`ForecastResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.run(...)
```
### run_forecast

Qualified name: `macroforecast.forecasting.runner.run`

#### Signature

```python
macroforecast.forecasting.run_forecast(data: Any, model: str | Callable[..., Any] | ModelSpec, *, window: WindowSpec | str | None = None, preprocessing: PreprocessSpec | None = None, preprocessing_policy: StagePolicy | str | None = None, features: FeatureSpec | None = None, feature_policy: StagePolicy | str | None = None, model_selection: SearchSpec | Mapping[str, SearchSpec | None] | None = None, model_selection_policy: StagePolicy | str | None = None, model_selection_metric: str | Callable[..., float] = "mse", maximize_model_selection: bool = False, preset: str | Mapping[str, str | None] | None = None, params: Mapping[str, Any] | None = None, target: str | None = None, horizon: int = 1, horizons: Sequence[int] | int | None = None, forecast_policy: str = "direct", future_feature_policy: str | None = None, target_transform: str | None = None, combination: str | CombinationSpec | Sequence[str | CombinationSpec | Mapping[str, Any]] | Mapping[str, Any] | None = None, save_models: bool = True, model_store: str | Path = "trained_model", preprocessing_cache: dict[Any, FittedPreprocessor | _PreparedStage | FittedFeatureBuilder] | None = None, preprocessing_store: PreprocessorStore | None = None, checkpoint_path: str | Path | None = None, selection_history: bool = False) -> ForecastResult
```

#### Description

Run a windowed macro forecasting experiment.

The runner composes small stage callables. ``window`` owns the temporal
design, stage policies decide where preprocessing, features, and model
selection are fitted, model specs fit predictors to targets, and the result
records a run-level metadata ledger.

A ``run`` is ATOMIC: it fits exactly ONE model. ``model`` must be a single
``str`` model name, a ``Callable`` model factory, or a ``ModelSpec`` (a
fit-time model-ensemble spec still counts as one model). Passing a sequence
or a mapping of models raises ``TypeError`` -- run one model per call, or use
the pipeline with one ``Arm`` per model when comparing models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `model` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |
| `window` | keyword only | `WindowSpec \| str \| None` | `None` |
| `preprocessing` | keyword only | `PreprocessSpec \| None` | `None` |
| `preprocessing_policy` | keyword only | `StagePolicy \| str \| None` | `None` |
| `features` | keyword only | `FeatureSpec \| None` | `None` |
| `feature_policy` | keyword only | `StagePolicy \| str \| None` | `None` |
| `model_selection` | keyword only | `SearchSpec \| Mapping[str, SearchSpec \| None] \| None` | `None` |
| `model_selection_policy` | keyword only | `StagePolicy \| str \| None` | `None` |
| `model_selection_metric` | keyword only | `str \| Callable[..., float]` | `"mse"` |
| `maximize_model_selection` | keyword only | `bool` | `False` |
| `preset` | keyword only | `str \| Mapping[str, str \| None] \| None` | `None` |
| `params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `target` | keyword only | `str \| None` | `None` |
| `horizon` | keyword only | `int` | `1` |
| `horizons` | keyword only | `Sequence[int] \| int \| None` | `None` |
| `forecast_policy` | keyword only | `str` | `"direct"` |
| `future_feature_policy` | keyword only | `str \| None` | `None` |
| `target_transform` | keyword only | `str \| None` | `None` |
| `combination` | keyword only | `str \| CombinationSpec \| Sequence[str \| CombinationSpec \| Mapping[str, Any]] \| Mapping[str, Any] \| None` | `None` |
| `save_models` | keyword only | `bool` | `True` |
| `model_store` | keyword only | `str \| Path` | `"trained_model"` |
| `preprocessing_cache` | keyword only | `dict[Any, FittedPreprocessor \| _PreparedStage \| FittedFeatureBuilder] \| None` | `None` |
| `preprocessing_store` | keyword only | `PreprocessorStore \| None` | `None` |
| `checkpoint_path` | keyword only | `str \| Path \| None` | `None` |
| `selection_history` | keyword only | `bool` | `False` |

#### Returns

`ForecastResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecasting.run_forecast(...)
```
