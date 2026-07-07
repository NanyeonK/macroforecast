# Custom Interpretation And Analysis

[Back to custom extensions](index.md)

This page is generated from the live callable signatures.

## Callable Reference

### custom_interpretation

Qualified name: `macroforecast.interpretation.core.custom_interpretation`

#### Signature

```python
macroforecast.interpretation.custom_interpretation(model: Any, X: pd.DataFrame, func: Callable[..., Any], *, y: pd.Series | np.ndarray | None = None, name: str | None = None, metadata: Mapping[str, Any] | None = None, **params: Any) -> pd.DataFrame
```

#### Description

Run a user-supplied interpretation callable and attach metadata.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `Any` | `required` |
| `X` | positional or keyword | `pd.DataFrame` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `y` | keyword only | `pd.Series \| np.ndarray \| None` | `None` |
| `name` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.interpretation.custom_interpretation(...)
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

### custom_forecast_diagnostic

Qualified name: `macroforecast.forecast_analysis.core.custom_forecast_diagnostic`

#### Signature

```python
macroforecast.forecast_analysis.custom_forecast_diagnostic(forecasts: Any, func: Callable[..., Any], *, name: str | None = None, metadata: Mapping[str, Any] | None = None, **params: Any) -> pd.DataFrame
```

#### Description

Run a user-supplied forecast diagnostic and attach macroforecast metadata.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | positional or keyword | `Any` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `name` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.forecast_analysis.custom_forecast_diagnostic(...)
```
