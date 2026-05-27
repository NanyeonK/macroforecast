# `lasso` -- Lasso regression (L1-regularised OLS).

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.lasso_fit`.

## Function signature

```python
mf.functions.lasso_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
    max_iter: int = 20000,
) -> LassoFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |
| `alpha` | `float` | `1.0` | >=0 | L1 regularisation strength. Larger values force more coefficients to exactly zero. |
| `max_iter` | `int` | `20000` | >=1 | Maximum number of coordinate descent iterations. |

## Returns

`LassoFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coef_` | `np.ndarray` | Fitted coefficient vector, shape (n_features,). |
| `.intercept_` | `float` | Fitted intercept scalar. |
| `.alpha` | `float` | Regularisation strength used. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable text table of fit results. |

## Behavior

Iterative coordinate descent: minimises ``||y - Xβ||² + α||β||₁``. Forces a subset of coefficients to exactly zero, yielding a sparse solution. Uses sklearn's ``Lasso`` with ``max_iter=20000`` for stability.

**When to use**

Variable selection; sparse forecasts on high-dimensional panels.

## In recipe context

Set ``params.model = "lasso"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: lasso
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Tibshirani (1996) 'Regression Shrinkage and Selection via the Lasso', JRSS-B 58(1).

## Related ops

See also: `ridge`, `elastic_net`, `lasso_path` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
