# `var` -- Vector autoregression VAR(p).

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.var_fit`.

## Function signature

```python
mf.functions.var_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> VARFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`VARFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_lags` | `int` | VAR lag order p. |
| `.n_obs` | `int` | Number of observations. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: lag order and observation count. |

## Behavior

Joint AR(p) over the target plus its predictors. Uses statsmodels' ``VAR`` and forecasts the target component of the joint system. Captures cross-series dynamics that single-equation AR misses.

**When to use**

Multi-series joint forecasting; impulse-response decomposition (paired with L7 ``orthogonalised_irf`` for Cholesky-identified shocks; ``generalized_irf`` reserved for the future Pesaran-Shin 1998 order-invariant variant).

**When NOT to use**

High-dimensional panels (VAR scales O(p²)); use BVAR shrinkage instead.

## In recipe context

Set ``params.model = "var"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: var
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Sims (1980) 'Macroeconomics and Reality', Econometrica 48(1).

## Related ops

See also: `bvar_minnesota`, `factor_augmented_var`, `ar_p` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
