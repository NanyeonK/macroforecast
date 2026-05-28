# `bvar_minnesota` -- Bayesian VAR with Minnesota prior shrinkage.

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.bvar_minnesota_fit`.

## Function signature

```python
mf.functions.bvar_minnesota_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> BVARMinnesotaFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`BVARMinnesotaFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_lags` | `int` | VAR lag order p. |
| `.lambda1` | `float` | Minnesota prior tightness. |
| `.n_obs` | `int` | Number of observations. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: lag order, tightness, observation count. |

## Behavior

Litterman (1986) Minnesota prior: shrinks each equation toward a univariate random walk. ``params.minnesota_lambda1`` controls overall tightness; ``params.minnesota_lambda_decay`` controls lag decay; ``params.minnesota_lambda_cross`` controls cross-equation shrinkage.

Returns a closed-form posterior mean -- no MCMC. Cheap and deterministic.

**When to use**

Multi-series forecasting where standard VAR overfits; macro panels with strong unit-root behaviour.

## In recipe context

Set ``params.model = "bvar_minnesota"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: bvar_minnesota
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Litterman (1986) 'Forecasting With Bayesian Vector Autoregressions -- Five Years of Experience', JBES 4(1).

## Related ops

See also: `bvar_normal_inverse_wishart`, `var`, `factor_augmented_var` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
