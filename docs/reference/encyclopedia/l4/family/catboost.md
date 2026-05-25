<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `catboost` -- CatBoost gradient-boosted trees (optional dependency).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.catboost_fit`.

## Function signature

```python
mf.functions.catboost_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> CatBoostFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`CatBoostFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.feature_importances_` | `np.ndarray` | Feature importances from CatBoost (percentage-based), shape (n_features,). |
| `.n_estimators_used` | `int` | Number of boosting iterations (= n_estimators parameter). |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, guaranteed 1-D via .ravel(). |
| `.summary()` | `str` | Human-readable table of fit results including top-3 feature importances. |

## Behavior

Requires ``pip install macroforecast[catboost]``. Ordered boosting + native categorical handling.

**When to use**

Categorical-heavy panels; ordered-boosting research.

## In recipe context

Set ``params.family = "catboost"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: catboost
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Prokhorenkova et al. (2018) 'CatBoost: unbiased boosting with categorical features', NeurIPS.

## Related ops

See also: `xgboost`, `lightgbm` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
