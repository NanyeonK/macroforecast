# `mars` -- Multivariate Adaptive Regression Splines (Friedman 1991).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.mars_fit`.

## Function signature

```python
mf.functions.mars_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> MARSFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`MARSFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_terms` | `int` | Number of MARS basis terms. |
| `.n_features_in_` | `int` | Number of input features. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: term count and feature count. |

## Behavior

Greedy forward / backward selection of piecewise-linear hinge basis functions ``max(0, x - c)`` and their products. Atomic primitive -- sklearn does not provide a MARS implementation. Runtime wraps ``pyearth`` as an optional dep; install via ``pip install macroforecast[mars]``. Required as the base learner for the Coulombe (2024) 'MARSquake' recipe (``bagging(base_family=mars, ...)``).

Operational from v0.9.0; raises ``NotImplementedError`` with an install hint when ``pyearth`` is not present (mirrors the xgboost / lightgbm / catboost optional-dep error pattern).

**When to use**

Non-linear regression with interpretable basis functions; MARSquake recipe base learner.

**When NOT to use**

Without ``[mars]`` extra installed -- raises a clear NotImplementedError.

## In recipe context

Set ``params.family = "mars"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: mars
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Friedman (1991) 'Multivariate Adaptive Regression Splines', Annals of Statistics 19(1).

## Related ops

See also: `gradient_boosting`, `decision_tree`, `bagging` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
