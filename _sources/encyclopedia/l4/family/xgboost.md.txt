# `xgboost` -- XGBoost gradient-boosted trees (optional dependency).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.xgboost_fit`.

## Function signature

```python
mf.functions.xgboost_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> XGBoostFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`XGBoostFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.feature_importances_` | `np.ndarray` | Feature importances (gain-based) from XGBoost, shape (n_features,). |
| `.n_estimators_used` | `int` | Number of boosting rounds (= n_estimators parameter). |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable table of fit results including top-3 feature importances. |

## Behavior

Requires ``pip install macroforecast[xgboost]``. Histogram-based tree construction; native quantile loss; GPU support. Standard production-grade boosting library.

**When to use**

Production sweeps where xgboost's speed matters; quantile forecasting (xgb 2.0+).

**When NOT to use**

Lightweight installs (no extra installed) -- raises ImportError.

## In recipe context

Set ``params.family = "xgboost"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: xgboost
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Chen & Guestrin (2016) 'XGBoost: A Scalable Tree Boosting System', KDD.

## Related ops

See also: `gradient_boosting`, `lightgbm`, `catboost` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
