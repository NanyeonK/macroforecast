# `dfm_mixed_mariano_murasawa` -- Mariano-Murasawa-style mixed-frequency dynamic factor model.

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.dfm_fit`.

## Function signature

```python
mf.functions.dfm_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> DFMFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`DFMFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_factors` | `int` | Number of dynamic factors. |
| `.n_obs` | `int` | Number of observations. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: factor count and observation count. |

## Behavior

Linear-Gaussian state-space model with monthly-aggregator observation equation. Routes to ``statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`` when ``params.mixed_frequency = True`` and per-column frequency tags are supplied; otherwise falls back to the single-frequency ``DynamicFactor`` estimator (Kalman MLE).

**When to use**

Mixed-frequency nowcasting (e.g., quarterly GDP from monthly indicators).

## In recipe context

Set ``params.model = "dfm_mixed_mariano_murasawa"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: dfm_mixed_mariano_murasawa
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Mariano & Murasawa (2010) 'A coincident index, common factors, and monthly real GDP', Oxford Bulletin of Economics and Statistics 72(1).

## Related ops

See also: `factor_augmented_ar`, `factor_augmented_var` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
