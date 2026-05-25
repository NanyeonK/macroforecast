<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `svr_linear` -- Support vector regression with linear kernel.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.svr_linear_fit`.

## Function signature

```python
mf.functions.svr_linear_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> SVRLinearFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`SVRLinearFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.C` | `float` | Regularisation parameter used. |
| `.n_support_vectors` | `int` | Number of support vectors. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: C and support vector count. |

## Behavior

ε-insensitive loss + L2 regularisation. Sparse in support vectors.

Configures the ``family`` axis on ``L4_A_model_selection`` (layer ``l4``); the ``svr_linear`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Robust linear baselines; comparison against ridge.

## In recipe context

Set ``params.family = "svr_linear"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: svr_linear
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Drucker, Burges, Kaufman, Smola & Vapnik (1997) 'Support Vector Regression Machines', NeurIPS.

## Related ops

See also: `svr_rbf`, `svr_poly`, `ridge` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
