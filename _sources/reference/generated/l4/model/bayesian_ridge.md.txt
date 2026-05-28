# `bayesian_ridge` -- Bayesian ridge with empirical-Bayes prior.

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.bayesian_ridge_fit`.

## Function signature

```python
mf.functions.bayesian_ridge_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> BayesianRidgeFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`BayesianRidgeFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coef_` | `np.ndarray` | Posterior mean coefficient vector, shape (n_features,). |
| `.intercept_` | `float` | Posterior mean intercept scalar. |
| `.alpha_` | `float` | Posterior noise precision (empirical Bayes). |
| `.lambda_` | `float` | Posterior weight precision (empirical Bayes). |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable text table of fit results. |

## Behavior

sklearn ``BayesianRidge``: gamma priors on noise + coefficient precision; type-II ML estimates of both. Returns posterior mean coefficients + posterior variance. Useful when the user wants a coefficient credible interval without bootstrapping.

**When to use**

Studies that need coefficient credible intervals; default-Bayesian baselines.

## In recipe context

Set ``params.model = "bayesian_ridge"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: bayesian_ridge
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

## Related ops

See also: `ridge`, `bvar_minnesota` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
