# `partial_dependence` -- Friedman (2001) partial dependence plot.

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.partial_dependence_importance`.

## Function signature

```python
mf.functions.partial_dependence_importance(
    result: FitResultBase,
    X: np.ndarray | pd.DataFrame,
) -> PDPImportanceResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `result` | `FitResultBase` | — | — | Fitted result object exposing ._model (the raw sklearn estimator). Returned by any L4 standalone callable such as mf.functions.ridge_fit, mf.functions.random_forest_fit, etc. |
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix used for importance computation. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |

## Returns

`PDPImportanceResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.importances_` | `np.ndarray` | PDP range (max - min) per feature, shape (n_features,). |
| `.feature_names_` | `list[str]` | Feature names. |
| `.pdp_values_` | `dict[str, np.ndarray]` | Mean predictions at grid points per feature. |
| `.grid_values_` | `dict[str, np.ndarray]` | Grid evaluation points per feature. |
| `.summary(top_n=10)` | `str` | Human-readable text table. |

## Behavior

For feature ``j``, computes ``E_{X_{-j}}[f(x_j, X_{-j})]`` over a grid of ``x_j`` values. Visualises the marginal effect of ``x_j`` on the prediction averaged over the joint distribution of remaining features. sklearn ``partial_dependence`` backend.

**When to use**

Visualising marginal feature effects; first-pass non-linearity audit.

**When NOT to use**

Highly correlated features -- PDP averages over impossible regions of feature space. Use ``accumulated_local_effect`` instead.

## In recipe context

Set ``params.op = "partial_dependence"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: partial_dependence
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Friedman (2001) 'Greedy Function Approximation: A Gradient Boosting Machine', Annals of Statistics 29(5): 1189-1232.

## Related ops

See also: `accumulated_local_effect`, `friedman_h_interaction` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
