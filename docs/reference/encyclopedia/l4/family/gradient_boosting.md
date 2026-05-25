<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `gradient_boosting` -- Gradient-boosted regression trees (sklearn).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.gradient_boosting_fit`.

## Function signature

```python
mf.functions.gradient_boosting_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> GradientBoostingFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`GradientBoostingFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.feature_importances_` | `np.ndarray` | Feature importances from the GBM, shape (n_features,). Sums to 1.0. |
| `.n_estimators_used` | `int` | Number of boosting iterations (= n_estimators parameter). |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable table of fit results including top-3 feature importances. |

## Behavior

Sklearn ``GradientBoostingRegressor``. Sequential boosting with shallow trees. ``params.n_estimators`` (default 200) and ``params.learning_rate`` (default 0.05) trade variance for bias.

**When to use**

Default boosted baseline when xgboost / lightgbm are unavailable.

## In recipe context

Set ``params.family = "gradient_boosting"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: gradient_boosting
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Friedman (2001) 'Greedy function approximation: A gradient boosting machine', Annals of Statistics 29(5).

## Related ops

See also: `xgboost`, `lightgbm`, `catboost` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
