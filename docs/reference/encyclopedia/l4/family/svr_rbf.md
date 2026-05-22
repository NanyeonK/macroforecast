# `svr_rbf` -- Support vector regression with RBF kernel.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.svr_rbf_fit`.

## Function signature

```python
mf.functions.svr_rbf_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> SVRRBFFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`SVRRBFFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.C` | `float` | Regularisation parameter used. |
| `.gamma` | `str|float` | RBF bandwidth parameter. |
| `.n_support_vectors` | `int` | Number of support vectors. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: C, gamma, support vector count. |

## Behavior

Non-linear regression via kernel trick. Slow on large panels (O(n³)).

Configures the ``family`` axis on ``L4_A_model_selection`` (layer ``l4``); the ``svr_rbf`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Small / medium-dim non-linear regression; kernel-method ablations.

## In recipe context

Set ``params.family = "svr_rbf"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: svr_rbf
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

## Related ops

See also: `svr_linear`, `svr_poly`, `random_forest` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
