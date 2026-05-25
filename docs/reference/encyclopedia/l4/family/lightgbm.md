<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `lightgbm` -- LightGBM gradient-boosted trees (optional dependency).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.lightgbm_fit`.

## Function signature

```python
mf.functions.lightgbm_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> LightGBMFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`LightGBMFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.feature_importances_` | `np.ndarray` | Feature importances (split count) from LightGBM, shape (n_features,). |
| `.n_estimators_used` | `int` | Number of boosting rounds (= n_estimators parameter). |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable table of fit results including top-3 feature importances. |

## Behavior

Requires ``pip install macroforecast[lightgbm]``. Leaf-wise tree growth; fast on wide / categorical-heavy panels.

**When to use**

Wide categorical panels; production sweeps where lightgbm's speed matters.

**When NOT to use**

Lightweight installs (no extra installed) -- raises ImportError.

## In recipe context

Set ``params.family = "lightgbm"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: lightgbm
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Ke et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', NeurIPS.

## Related ops

See also: `xgboost`, `gradient_boosting` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
