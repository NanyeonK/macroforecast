# `random_forest` -- Random forest (sklearn).

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.random_forest_fit`.

## Function signature

```python
mf.functions.random_forest_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> RandomForestFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`RandomForestFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.feature_importances_` | `np.ndarray` | Mean decrease in impurity per feature, shape (n_features,). Sums to 1.0. |
| `.n_estimators_used` | `int` | Number of trees grown (= n_estimators parameter). |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable table of fit results including top-3 feature importances. |

## Behavior

Bagged collection of decorrelated trees. ``params.n_estimators`` (default 200) controls the ensemble size; ``params.max_depth`` controls tree complexity. Standard non-linear baseline.

**When to use**

Default non-linear benchmark; non-stationary series where linear models fail.

## In recipe context

Set ``params.model = "random_forest"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: random_forest
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Breiman (2001) 'Random Forests', Machine Learning 45(1).

## Related ops

See also: `extra_trees`, `gradient_boosting`, `xgboost`, `macroeconomic_random_forest`, `quantile_regression_forest` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
