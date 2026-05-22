# `model_native_tree_importance` -- Mean-decrease-impurity importance from a fitted tree ensemble.

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.model_native_tree_importance`.

## Function signature

```python
mf.functions.model_native_tree_importance(
    result: FitResultBase,
    X: np.ndarray | pd.DataFrame,
) -> NativeImportanceResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `result` | `FitResultBase` | — | — | Fitted result object exposing ._model (the raw sklearn estimator). Returned by any L4 standalone callable such as mf.functions.ridge_fit, mf.functions.random_forest_fit, etc. |
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix used for importance computation. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |

## Returns

`NativeImportanceResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.importances_` | `np.ndarray` | MDI feature_importances_ values, shape (n_features,). |
| `.feature_names_` | `list[str]` | Feature names matching importances_. |
| `.method` | `str` | 'tree_native' -- method descriptor. |
| `.summary(top_n=10)` | `str` | Human-readable text table sorted by descending importance. |

## Behavior

Returns sklearn's ``feature_importances_`` for the fitted estimator -- the average reduction in node impurity attributable to each feature, weighted by node sample count. Available for every tree-family L4 model (``decision_tree`` / ``random_forest`` / ``extra_trees`` / ``gradient_boosting`` / ``xgboost`` / ``lightgbm`` / ``catboost``).

Cheap and built-in; biases toward high-cardinality features. For unbiased tree importance, prefer ``permutation_importance`` or ``permutation_importance_strobl``.

**When to use**

Quick first-pass tree importance; pair with permutation importance for bias-correction.

**When NOT to use**

High-cardinality continuous features dominate -- known MDI bias (Strobl et al. 2007).

## In recipe context

Set ``params.op = "model_native_tree_importance"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: model_native_tree_importance
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Breiman (2001) 'Random Forests', Machine Learning 45(1): 5-32.
* Strobl, Boulesteix, Zeileis & Hothorn (2007) 'Bias in random forest variable importance measures', BMC Bioinformatics 8: 25.

## Related ops

See also: `permutation_importance`, `permutation_importance_strobl` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
