# `shap_tree` -- Tree SHAP -- exact polynomial-time Shapley values for tree ensembles.

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.shap_tree_importance`.

## Function signature

```python
mf.functions.shap_tree_importance(
    result: FitResultBase,
    X: np.ndarray | pd.DataFrame,
) -> SHAPImportanceResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `result` | `FitResultBase` | — | — | Fitted result object exposing ._model (the raw sklearn estimator). Returned by any L4 standalone callable such as mf.functions.ridge_fit, mf.functions.random_forest_fit, etc. |
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix used for importance computation. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |

## Returns

`SHAPImportanceResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.shap_values_` | `np.ndarray` | SHAP values, shape (n_samples, n_features). |
| `.expected_value_` | `float` | SHAP base value (expected model output). |
| `.feature_names_` | `list[str]` | Feature names. |
| `.explainer_type` | `str` | Explainer used (TreeExplainer / KernelExplainer). |
| `.summary(top_n=10)` | `str` | Table of mean absolute SHAP values. |

## Behavior

Lundberg-Erion-Lee (2020) algorithm computing exact Shapley values in ``O(T·L·D²)`` time (T trees, L leaves, D depth) instead of ``O(2^M)`` brute-force. Available for ``random_forest`` / ``extra_trees`` / ``gradient_boosting`` / ``xgboost`` / ``lightgbm`` / ``catboost``.

Returns per-prediction SHAP values; the ``output_table_format`` L7.B axis controls whether the result is the global mean-``|SHAP|`` ranking or the per-row decomposition.

**When to use**

Default importance op for tree ensembles; exact and fast.

**When NOT to use**

Non-tree models -- use ``shap_kernel`` or ``shap_linear`` instead.

## In recipe context

Set ``params.op = "shap_tree"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: shap_tree
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Lundberg & Lee (2017) 'A Unified Approach to Interpreting Model Predictions', NeurIPS 30: 4765-4774.
* Lundberg, Erion & Lee (2020) 'From local explanations to global understanding with explainable AI for trees', Nature Machine Intelligence 2: 56-67.

## Related ops

See also: `shap_kernel`, `shap_linear`, `shap_interaction`, `shap_deep` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
