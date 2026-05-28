# `ar_p` -- Autoregressive AR(p) on the target.

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.ar_fit`.

## Function signature

```python
mf.functions.ar_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> ARFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`ARFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_lags` | `int` | AR lag order p. |
| `.coef_` | `np.ndarray` | Fitted AR coefficients, shape (n_lags,). |
| `.intercept_` | `float` | Fitted intercept. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: AR order, intercept, per-lag coefficients. |

## Behavior

Pure autoregression -- predictor matrix is the lagged target (no exogenous regressors). ``params.n_lag`` sets p. Useful as a non-trivial benchmark in macro forecasting where the lagged target captures most of the predictability.

**When to use**

Default benchmark in any forecasting horse race; replication of papers reporting AR baselines.

## In recipe context

Set ``params.model = "ar_p"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: ar_p
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Stock & Watson (2007) 'Why Has US Inflation Become Harder to Forecast?', JMCB 39.

## Related ops

See also: `var`, `factor_augmented_ar` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
