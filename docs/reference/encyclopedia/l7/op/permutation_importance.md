# `permutation_importance` -- Breiman-Fisher-Rudin (2019) model-agnostic permutation importance.

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.permutation_importance`.

## Function signature

```python
mf.functions.permutation_importance(
    result: FitResultBase,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> PermutationImportanceResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `result` | `FitResultBase` | — | — | Fitted result object exposing ._model (the raw sklearn estimator). Returned by any L4 standalone callable. |
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix used for importance computation. Shape (n_samples, n_features). |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Used to compute baseline and permuted MSE losses. |

## Returns

`PermutationImportanceResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.importances_mean_` | `np.ndarray` | Mean importance over n_repeats, shape (n_features,). |
| `.importances_std_` | `np.ndarray` | Std deviation over n_repeats, shape (n_features,). |
| `.feature_names_` | `list[str]` | Feature names. |
| `.n_repeats` | `int` | Number of permutation repeats used. |
| `.summary(top_n=10)` | `str` | Human-readable text table with mean and std. |

## Behavior

For each predictor ``j``, computes the increase in OOS loss when ``x_j`` is randomly permuted. The score is ``L(y, f(X_perm_j)) - L(y, f(X))`` averaged over ``n_repeats`` (default 10). Model-agnostic: works for every L4 model.

Bias-free alternative to ``model_native_tree_importance``; the gold-standard fallback for any model that does not expose a native importance attribute.

**When to use**

Default importance score for non-linear models; comparing across model families.

**When NOT to use**

Highly correlated predictors -- permutation breaks the dependence and inflates importance. Use ``permutation_importance_strobl`` instead.

## In recipe context

Set ``params.op = "permutation_importance"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: permutation_importance
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Fisher, Rudin & Dominici (2019) 'All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously', JMLR 20(177): 1-81.
* Breiman (2001) 'Random Forests', Machine Learning 45(1): 5-32.

## Related ops

See also: `permutation_importance_strobl`, `lofo`, `model_native_tree_importance` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
