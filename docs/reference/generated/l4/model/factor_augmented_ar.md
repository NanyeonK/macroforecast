# `factor_augmented_ar` -- Factor-augmented AR (PCA factors + AR lags on target).

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.far_fit`.

## Function signature

```python
mf.functions.far_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> FARFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`FARFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_factors` | `int` | Number of PCA factors extracted from X. |
| `.n_lags` | `int` | AR lag order p. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: factor count and lag order. |

## Behavior

Stock-Watson (2002) FAR: extract the first ``params.n_factors`` principal components from the predictor panel, augment with AR(``params.n_lag``) lags of the target, run OLS. Standard high-dimensional macro forecasting baseline.

**When to use**

High-dimensional macro panels (FRED-MD/QD); diffusion-index baselines.

## In recipe context

Set ``params.model = "factor_augmented_ar"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: factor_augmented_ar
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460).

## Related ops

See also: `factor_augmented_var`, `principal_component_regression`, `ar_p` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
