<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `lasso_path` -- Lasso with CV-selected alpha (LassoCV).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.lasso_path_fit`.

## Function signature

```python
mf.functions.lasso_path_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    cv: int = 5,
    max_iter: int = 20000,
    random_state: int | None = None,
) -> LassoPathFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |
| `cv` | `int` | `5` | >=2 | Number of cross-validation folds for alpha selection. |
| `max_iter` | `int` | `20000` | >=1 | Maximum coordinate descent iterations per alpha. |
| `random_state` | `int | None` | `None` | — | Random seed for CV fold generation. None uses system entropy. |

## Returns

`LassoPathFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coef_` | `np.ndarray` | Fitted coefficient vector, shape (n_features,). |
| `.intercept_` | `float` | Fitted intercept scalar. |
| `.alpha_selected` | `float` | CV-selected regularisation strength. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable text table of fit results. |

## Behavior

Wraps sklearn's ``LassoCV``. Picks α automatically from a regularisation path via k-fold CV (``params.cv``). Equivalent to setting ``model: lasso, search_algorithm: cv_path``.

**When to use**

When the recipe wants automatic α selection without an explicit search_algorithm.

## In recipe context

Set ``params.family = "lasso_path"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: lasso_path
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

## Related ops

See also: `lasso`, `ridge` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
