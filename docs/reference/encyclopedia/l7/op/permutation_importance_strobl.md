# `permutation_importance_strobl` -- Strobl (2008) conditional permutation importance.

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.cond_permutation_importance`.

## Function signature

```python
mf.functions.cond_permutation_importance(
    result: FitResultBase,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> CondPermutationImportanceResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `result` | `FitResultBase` | — | — | Fitted result object exposing ._model (the raw sklearn estimator). Returned by any L4 standalone callable. |
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix used for importance computation. Shape (n_samples, n_features). |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Used to compute baseline and permuted MSE losses. |

## Returns

`CondPermutationImportanceResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.importances_mean_` | `np.ndarray` | Mean conditional permutation importance, shape (n_features,). |
| `.importances_std_` | `np.ndarray` | Std deviation over n_repeats, shape (n_features,). |
| `.feature_names_` | `list[str]` | Feature names. |
| `.method` | `str` | 'strobl' -- Strobl (2008) conditional permutation. |
| `.summary(top_n=10)` | `str` | Human-readable text table. |

## Behavior

Permutes ``x_j`` only within bins defined by the joint distribution of correlated predictors, eliminating the extrapolation bias of plain permutation importance for correlated features. v0.3 implementation uses tree-partition bins (Strobl et al. 2008 §4).

**When to use**

Highly correlated macro panels (FRED-MD / -QD with redundant aggregates).

**When NOT to use**

When predictor correlations are negligible -- the cheaper plain permutation importance suffices.

## In recipe context

Set ``params.op = "permutation_importance_strobl"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: permutation_importance_strobl
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Strobl, Boulesteix, Kneib, Augustin & Zeileis (2008) 'Conditional variable importance for random forests', BMC Bioinformatics 9: 307.

## Related ops

See also: `permutation_importance` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
