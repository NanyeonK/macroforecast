<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `principal_component_regression` -- Principal component regression (PCA → OLS).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.pcr_fit`.

## Function signature

```python
mf.functions.pcr_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> PCRFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`PCRFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_components` | `int` | Number of principal components used in regression. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: component count. |

## Behavior

Identical to ``factor_augmented_ar`` without the AR lags. Useful when the target's own lags add noise (rare but happens for highly seasonal series).

**When to use**

Diffusion-index forecasts where AR augmentation hurts performance.

## In recipe context

Set ``params.family = "principal_component_regression"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: principal_component_regression
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

## Related ops

See also: `factor_augmented_ar` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
