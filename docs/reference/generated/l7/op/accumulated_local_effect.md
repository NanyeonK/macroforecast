# `accumulated_local_effect` -- Apley & Zhu (2020) accumulated local effects -- PDP alternative robust to correlation.

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.ale_importance`.

## Function signature

```python
mf.functions.ale_importance(
    result: FitResultBase,
    X: np.ndarray | pd.DataFrame,
) -> ALEImportanceResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `result` | `FitResultBase` | — | — | Fitted result object exposing ._model (the raw sklearn estimator). Returned by any L4 standalone callable such as mf.functions.ridge_fit, mf.functions.random_forest_fit, etc. |
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix used for importance computation. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |

## Returns

`ALEImportanceResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.importances_` | `np.ndarray` | Mean absolute centred ALE (L1 norm) per feature. |
| `.feature_names_` | `list[str]` | Feature names. |
| `.ale_values_` | `dict[str, np.ndarray]` | Centred cumulative ALE values per feature. |
| `.summary(top_n=10)` | `str` | Human-readable text table. |

## Behavior

For feature ``j``, computes the cumulative local change ``Σ_{k≤K} E_{X_{-j} | x_j ∈ bin_k}[∂f/∂x_j]·Δx_j``. The binning + conditioning eliminates the 'extrapolation into low-density regions' bias of plain PDPs.

**When to use**

Correlated feature panels (FRED-MD / -QD) where PDPs are misleading.

## In recipe context

Set ``params.op = "accumulated_local_effect"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: accumulated_local_effect
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Apley & Zhu (2020) 'Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models', JRSS Series B 82(4): 1059-1086.

## Related ops

See also: `partial_dependence` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
