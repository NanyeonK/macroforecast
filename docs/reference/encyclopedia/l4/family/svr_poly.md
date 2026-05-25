<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `svr_poly` -- Support vector regression with polynomial kernel.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.svr_poly_fit`.

## Function signature

```python
mf.functions.svr_poly_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> SVRPolyFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`SVRPolyFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.C` | `float` | Regularisation parameter used. |
| `.degree` | `int` | Polynomial degree. |
| `.n_support_vectors` | `int` | Number of support vectors. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: C, degree, support vector count. |

## Behavior

Polynomial-kernel SVR. Useful for studies that want explicit polynomial features without manual expansion.

**When to use**

Polynomial-kernel ablations. Selecting ``svr_poly`` on ``l4.family`` activates this branch of the layer's runtime.

## In recipe context

Set ``params.family = "svr_poly"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: svr_poly
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

## Related ops

See also: `svr_rbf`, `svr_linear` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
