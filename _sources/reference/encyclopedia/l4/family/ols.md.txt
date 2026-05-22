# `ols` -- Ordinary least squares -- baseline linear regression.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.ols_fit`.

## Function signature

```python
mf.functions.ols_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> OLSFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`OLSFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coef_` | `np.ndarray` | Fitted coefficient vector, shape (n_features,). |
| `.intercept_` | `float` | Fitted intercept scalar. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable text table of fit results. |

## Behavior

Closed-form linear regression with no regularisation. Cheapest linear estimator; appropriate when p << n and predictors are well-conditioned. Returns NaN coefficients when the design matrix is rank-deficient (sklearn raises an error in that case).

**When to use**

Low-dimensional baselines; sanity-check sweeps.

**When NOT to use**

High-dimensional panels (p ≈ n) -- use ridge / lasso instead.

## In recipe context

Set ``params.family = "ols"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: ols
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Greene (2018) 'Econometric Analysis', 8th ed., Pearson.

## Related ops

See also: `ridge`, `lasso`, `elastic_net`, `ar_p` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
