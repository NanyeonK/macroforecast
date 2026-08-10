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

Run a user-supplied interpretation callable and attach schema metadata.

This is the escape hatch for an attribution method the package does not
ship. What it does is small and worth stating exactly, because everything
else about the result depends on it.

**How your callable is invoked.** Always with this signature::

    func(model, X, y=y, metadata={...}, **params)

- ``model`` is passed through unchanged -- a ``ModelFit`` if that is what you
  gave, not an unwrapped estimator. Use ``fit.estimator`` if you need the
  underlying object, and ``fit.feature_names`` for the column order it was
  trained on.
- ``X`` is coerced to a DataFrame first, so you can rely on ``.columns``.
- ``y`` and ``metadata`` are always passed as keywords, even when ``None``
  and ``{}``. Accept them explicitly or absorb them with ``**kwargs``; a
  callable taking only ``(model, X)`` will raise ``TypeError``.

**What you must return.** A table: a DataFrame, or something that coerces to
one (a mapping of column to values, or a sequence of row mappings). A scalar
or a bare array is not accepted, because the result has to be joinable with
every other interpretation output.

**What this function does NOT do**, and each absence is deliberate:

- It does not validate that your numbers mean anything. No axiom is checked
  -- not efficiency, not that attributions sum to a prediction, not that an
  unused feature gets zero. If your method is wrong, the output is wrong and
  carries the same schema as a correct one.
- It does not restrict what your callable sees. You get the full ``X`` you
  passed in. Inside a runner that is the stage's own frame, but calling this
  directly on a full-sample panel explains a full-sample fit, which is not
  the same thing as explaining a forecast.
- It does not make your method reproducible. If it samples, seed it yourself.

What it *does* add is the schema every other interpretation output carries:
``kind="custom_interpretation"``, the resolved ``name``, the callable's
qualified name, the parameters, ``n_obs``, ``n_features``, whether a target
was supplied, and your own ``metadata`` under ``user_metadata`` -- so a
custom result is as traceable in a report as a built-in one.

Example::

    def range_importance(model, X, *, y=None, metadata=None, **params):
        spread = X.max() - X.min()
        return {"feature": list(X.columns), "importance": spread.to_numpy()}

    table = mf.interpretation.custom_interpretation(fit, X, range_importance)

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
