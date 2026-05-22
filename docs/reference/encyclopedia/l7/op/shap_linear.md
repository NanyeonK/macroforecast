# `shap_linear` -- Linear SHAP -- closed-form Shapley values for linear models.

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.shap_linear_importance`.

## Function signature

```python
mf.functions.shap_linear_importance(
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
| `.explainer_type` | `str` | Explainer used (LinearExplainer / KernelExplainer). |
| `.summary(top_n=10)` | `str` | Table of mean absolute SHAP values. |

## Behavior

For a fitted linear model ``f(x) = β'x + b``, the SHAP value for feature ``j`` reduces to ``β_j (x_j - E[x_j])``. Uses the training-sample mean as the reference. Available for every linear L4 family.

**When to use**

Linear models when the SHAP per-row decomposition is needed (otherwise ``model_native_linear_coef`` suffices).

## In recipe context

Set ``params.op = "shap_linear"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: shap_linear
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Lundberg & Lee (2017) 'A Unified Approach to Interpreting Model Predictions', NeurIPS 30: 4765-4774.

## Related ops

See also: `model_native_linear_coef`, `shap_tree`, `shap_kernel` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
