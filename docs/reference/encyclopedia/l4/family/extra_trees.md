<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `extra_trees` -- Extremely randomized trees (sklearn).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.extra_trees_fit`.

## Function signature

```python
mf.functions.extra_trees_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> ExtraTreesFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`ExtraTreesFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.feature_importances_` | `np.ndarray` | Mean decrease in impurity per feature, shape (n_features,). Sums to 1.0. |
| `.n_estimators_used` | `int` | Number of trees grown (= n_estimators parameter). |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable table of fit results including top-3 feature importances. |

## Behavior

Like RF but splits at random thresholds (no greedy search). Faster than RF; sometimes lower variance.

**v0.9 sub-axis**:
* ``params.max_features`` -- number of predictors considered at each split. ``"sqrt"`` (default) matches sklearn; ``1`` (operational, v0.9) implements Coulombe (2024) 'To Bag is to Prune' Perfectly Random Forest baseline (one random feature per split, fully random structure).

**When to use**

Quick non-linear baseline; large ensemble experiments; PRF baseline (max_features=1).

## In recipe context

Set ``params.family = "extra_trees"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: extra_trees
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Geurts, Ernst & Wehenkel (2006) 'Extremely randomized trees', Machine Learning 63(1).

## Related ops

See also: `random_forest`, `gradient_boosting` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
