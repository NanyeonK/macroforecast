# `model_native_linear_coef` -- Standardised regression coefficients from a fitted linear model.

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.model_native_linear_coef_importance`.

## Function signature

```python
mf.functions.model_native_linear_coef_importance(
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
| `.importances_` | `np.ndarray` | Absolute coefficient values |coef_j|, shape (n_features,). |
| `.feature_names_` | `list[str]` | Feature names matching importances_. |
| `.method` | `str` | 'linear_coef' -- method descriptor. |
| `.summary(top_n=10)` | `str` | Human-readable text table sorted by descending importance. |

## Behavior

Returns ``β̂_j`` for each predictor as the importance score; with ``standardize=True`` (default) the predictors are pre-scaled so coefficients are directly comparable. Compatible with every linear-family L4 model (``ols / ridge / lasso / elastic_net / lasso_path / bayesian_ridge / huber / glmboost``).

Cheapest meaningful importance score; the natural sanity-check to run before the more expensive permutation / SHAP families.

**When to use**

Linear-model baselines; quick interpretation when a tree / NN model is overkill.

**When NOT to use**

Non-linear models -- coefficients no longer summarise marginal effects.

## In recipe context

Set ``params.op = "model_native_linear_coef"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: model_native_linear_coef
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Greene (2018) 'Econometric Analysis', 8th ed., Pearson, Chapter 4.

## Related ops

See also: `model_native_tree_importance`, `lasso_inclusion_frequency` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
