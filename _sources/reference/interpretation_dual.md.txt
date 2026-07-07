# macroforecast.interpretation_dual

[Back to reference](index.md)

Observation-based dual interpretation for forecast results.

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `DualInterpretationResult` | class | Paper-aligned dual interpretation output bundle. |
| `data_portfolio_diagnostics` | function | Return concentration, short-position, leverage, and turnover diagnostics. |
| `dual_from_forecast_result` | function | Build a dual interpretation sidecar for a completed forecast result. |
| `dual_interpretation` | function | Run the ridge/KRR/RF DualML interpretation path in one callable. |
| `episode_group_weights` | function | Aggregate observation weights over named historical groups. |
| `forecast_diagnostics` | function | Return concentration, short-position, leverage, and turnover diagnostics. |
| `group_observation_weights` | function | Aggregate observation weights over named historical groups. |
| `observation_contributions` | function | Convert observation weights into observation-level forecast contributions. |
| `observation_weights` | function | Return DualML observation/data-portfolio weights. |
| `outcome_contributions` | function | Convert observation weights into observation-level forecast contributions. |
| `top_episodes` | function | Return the largest historical observations per forecast row. |
| `top_observations` | function | Return the largest historical observations per forecast row. |

## Callable And Class Reference

### DualInterpretationResult

Qualified name: `macroforecast.interpretation.dual.DualInterpretationResult`

#### Signature

```python
macroforecast.interpretation_dual.DualInterpretationResult(weights: pd.DataFrame, contributions: pd.DataFrame | None = None, diagnostics: pd.DataFrame | None = None, top_observations: pd.DataFrame | None = None, group_weights: pd.DataFrame | None = None, metadata: dict[str, Any] = <factory>, metadata_schema: dict[str, Any] = <factory>) -> None
```

#### Description

Paper-aligned dual interpretation output bundle.

Goulet Coulombe, Goebel, and Klieber's DualML code reports observation
weights, observation contributions, and data-portfolio diagnostics as
connected objects. The result container keeps that relation explicit while
still exposing output-ready tables through ``to_tables``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `contributions` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `diagnostics` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `top_observations` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `group_weights` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata_schema` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.interpretation_dual.DualInterpretationResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `weights` | `pd.DataFrame` | `required` |
| `contributions` | `pd.DataFrame \| None` | `None` |
| `diagnostics` | `pd.DataFrame \| None` | `None` |
| `top_observations` | `pd.DataFrame \| None` | `None` |
| `group_weights` | `pd.DataFrame \| None` | `None` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `metadata_schema` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
| `to_tables` | `to_tables(self, *, prefix: str = "dual") -> dict[str, pd.DataFrame]` | Return output-ready tables with paper-aligned names. |
### data_portfolio_diagnostics

Qualified name: `macroforecast.interpretation.dual.forecast_diagnostics`

#### Signature

```python
macroforecast.interpretation_dual.data_portfolio_diagnostics(weights: pd.DataFrame, *, top_q: float = 0.05) -> pd.DataFrame
```

#### Description

Return concentration, short-position, leverage, and turnover diagnostics.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `top_q` | keyword only | `float` | `0.05` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.data_portfolio_diagnostics(...)
```
### dual_from_forecast_result

Qualified name: `macroforecast.interpretation.dual.dual_from_forecast_result`

#### Signature

```python
macroforecast.interpretation_dual.dual_from_forecast_result(result: Any, model: Any | None, X_train: pd.DataFrame, y_train: pd.Series | Sequence[float], X_test: pd.DataFrame | None = None, *, attach: bool = True, sidecar_name: str = "dual", **kwargs: Any) -> Any
```

#### Description

Build a dual interpretation sidecar for a completed forecast result.

A forecast table cannot reconstruct the exact train/test feature matrices
used by the fitted model. The caller therefore passes the fitted model,
training features, training target, and forecast-row features explicitly.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `result` | positional or keyword | `Any` | `required` |
| `model` | positional or keyword | `Any \| None` | `required` |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| Sequence[float]` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `attach` | keyword only | `bool` | `True` |
| `sidecar_name` | keyword only | `str` | `"dual"` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.dual_from_forecast_result(...)
```
### dual_interpretation

Qualified name: `macroforecast.interpretation.dual.dual_interpretation`

#### Signature

```python
macroforecast.interpretation_dual.dual_interpretation(model: Any | None, X_train: pd.DataFrame, y_train: pd.Series | Sequence[float], X_test: pd.DataFrame | None = None, *, method: str = "auto", lambda_: float = 1e-08, kernel: str = "linear", sigma: float = 1.0, add_intercept: bool = False, ridge_penalty_scale: str = "n_train", normalize: bool = False, center: bool = False, include_base: bool = False, top_n: int = 10, top_sort_by: str = "abs_weight", top_q: float = 0.05, groups: Mapping[str, Sequence[Any]] | None = None, include_contributions: bool = True, include_diagnostics: bool = True, include_top_observations: bool = True, include_group_weights: bool | None = None) -> DualInterpretationResult
```

#### Description

Run the ridge/KRR/RF DualML interpretation path in one callable.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any \| None` | `required` |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| Sequence[float]` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `method` | keyword only | `str` | `"auto"` |
| `lambda_` | keyword only | `float` | `1e-08` |
| `kernel` | keyword only | `str` | `"linear"` |
| `sigma` | keyword only | `float` | `1.0` |
| `add_intercept` | keyword only | `bool` | `False` |
| `ridge_penalty_scale` | keyword only | `str` | `"n_train"` |
| `normalize` | keyword only | `bool` | `False` |
| `center` | keyword only | `bool` | `False` |
| `include_base` | keyword only | `bool` | `False` |
| `top_n` | keyword only | `int` | `10` |
| `top_sort_by` | keyword only | `str` | `"abs_weight"` |
| `top_q` | keyword only | `float` | `0.05` |
| `groups` | keyword only | `Mapping[str, Sequence[Any]] \| None` | `None` |
| `include_contributions` | keyword only | `bool` | `True` |
| `include_diagnostics` | keyword only | `bool` | `True` |
| `include_top_observations` | keyword only | `bool` | `True` |
| `include_group_weights` | keyword only | `bool \| None` | `None` |

#### Returns

`DualInterpretationResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.dual_interpretation(...)
```
### episode_group_weights

Qualified name: `macroforecast.interpretation.dual.group_observation_weights`

#### Signature

```python
macroforecast.interpretation_dual.episode_group_weights(weights: pd.DataFrame, groups: Mapping[str, Sequence[Any]], *, y_train: pd.Series | Sequence[float] | None = None) -> pd.DataFrame
```

#### Description

Aggregate observation weights over named historical groups.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `groups` | positional or keyword | `Mapping[str, Sequence[Any]]` | `required` |
| `y_train` | keyword only | `pd.Series \| Sequence[float] \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.episode_group_weights(...)
```
### forecast_diagnostics

Qualified name: `macroforecast.interpretation.dual.forecast_diagnostics`

#### Signature

```python
macroforecast.interpretation_dual.forecast_diagnostics(weights: pd.DataFrame, *, top_q: float = 0.05) -> pd.DataFrame
```

#### Description

Return concentration, short-position, leverage, and turnover diagnostics.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `top_q` | keyword only | `float` | `0.05` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.forecast_diagnostics(...)
```
### group_observation_weights

Qualified name: `macroforecast.interpretation.dual.group_observation_weights`

#### Signature

```python
macroforecast.interpretation_dual.group_observation_weights(weights: pd.DataFrame, groups: Mapping[str, Sequence[Any]], *, y_train: pd.Series | Sequence[float] | None = None) -> pd.DataFrame
```

#### Description

Aggregate observation weights over named historical groups.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `groups` | positional or keyword | `Mapping[str, Sequence[Any]]` | `required` |
| `y_train` | keyword only | `pd.Series \| Sequence[float] \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.group_observation_weights(...)
```
### observation_contributions

Qualified name: `macroforecast.interpretation.dual.observation_contributions`

#### Signature

```python
macroforecast.interpretation_dual.observation_contributions(weights: pd.DataFrame, y_train: pd.Series | Sequence[float], *, center: bool = False, include_base: bool = False) -> pd.DataFrame
```

#### Description

Convert observation weights into observation-level forecast contributions.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| Sequence[float]` | `required` |
| `center` | keyword only | `bool` | `False` |
| `include_base` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.observation_contributions(...)
```
### observation_weights

Qualified name: `macroforecast.interpretation.dual.observation_weights`

#### Signature

```python
macroforecast.interpretation_dual.observation_weights(model: Any | None, X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, *, method: str = "auto", lambda_: float = 1e-08, kernel: str = "linear", sigma: float = 1.0, add_intercept: bool = False, ridge_penalty_scale: str = "n_train", normalize: bool = False) -> pd.DataFrame
```

#### Description

Return DualML observation/data-portfolio weights.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any \| None` | `required` |
| `X_train` | positional or keyword | `pd.DataFrame` | `required` |
| `X_test` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `method` | keyword only | `str` | `"auto"` |
| `lambda_` | keyword only | `float` | `1e-08` |
| `kernel` | keyword only | `str` | `"linear"` |
| `sigma` | keyword only | `float` | `1.0` |
| `add_intercept` | keyword only | `bool` | `False` |
| `ridge_penalty_scale` | keyword only | `str` | `"n_train"` |
| `normalize` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.observation_weights(...)
```
### outcome_contributions

Qualified name: `macroforecast.interpretation.dual.observation_contributions`

#### Signature

```python
macroforecast.interpretation_dual.outcome_contributions(weights: pd.DataFrame, y_train: pd.Series | Sequence[float], *, center: bool = False, include_base: bool = False) -> pd.DataFrame
```

#### Description

Convert observation weights into observation-level forecast contributions.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | positional or keyword | `pd.Series \| Sequence[float]` | `required` |
| `center` | keyword only | `bool` | `False` |
| `include_base` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.outcome_contributions(...)
```
### top_episodes

Qualified name: `macroforecast.interpretation.dual.top_observations`

#### Signature

```python
macroforecast.interpretation_dual.top_episodes(weights: pd.DataFrame, *, y_train: pd.Series | Sequence[float] | None = None, n: int = 10, sort_by: str = "abs_weight") -> pd.DataFrame
```

#### Description

Return the largest historical observations per forecast row.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | keyword only | `pd.Series \| Sequence[float] \| None` | `None` |
| `n` | keyword only | `int` | `10` |
| `sort_by` | keyword only | `str` | `"abs_weight"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.top_episodes(...)
```
### top_observations

Qualified name: `macroforecast.interpretation.dual.top_observations`

#### Signature

```python
macroforecast.interpretation_dual.top_observations(weights: pd.DataFrame, *, y_train: pd.Series | Sequence[float] | None = None, n: int = 10, sort_by: str = "abs_weight") -> pd.DataFrame
```

#### Description

Return the largest historical observations per forecast row.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `weights` | positional or keyword | `pd.DataFrame` | `required` |
| `y_train` | keyword only | `pd.Series \| Sequence[float] \| None` | `None` |
| `n` | keyword only | `int` | `10` |
| `sort_by` | keyword only | `str` | `"abs_weight"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation_dual.top_observations(...)
```
