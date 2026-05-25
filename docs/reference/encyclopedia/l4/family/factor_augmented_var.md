<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `factor_augmented_var` -- Factor-augmented VAR (Bernanke-Boivin-Eliasz 2005).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.favar_fit`.

## Function signature

```python
mf.functions.favar_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> FAVARFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`FAVARFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_factors` | `int` | Number of PCA factors extracted from X. |
| `.n_lags` | `int` | VAR lag order p. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: factor count and lag order. |

## Behavior

Two-stage estimator: PCA factors from the predictor panel + VAR(``params.n_lag``) on (factors, target). Captures dynamic interactions between latent factors and the target series.

Useful for monetary-policy studies where the factors stand in for unobserved economic state.

**When to use**

Monetary-policy / macro-state studies; diffusion-index VAR baselines.

## In recipe context

Set ``params.family = "factor_augmented_var"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: factor_augmented_var
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Bernanke, Boivin & Eliasz (2005) 'Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive Approach', QJE 120(1).

## Related ops

See also: `var`, `factor_augmented_ar`, `bvar_minnesota` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
