# `huber` -- Huber regression (robust to outliers).

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.huber_fit`.

## Function signature

```python
mf.functions.huber_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    epsilon: float = 1.35,
    max_iter: int = 1000,
) -> HuberFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |
| `epsilon` | `float` | `1.35` | >1.0 | Huber loss transition point. Residuals with |r| <= epsilon * scale_ are treated as inliers (quadratic loss); larger residuals are outliers (linear loss). Must be > 1.0 (sklearn requirement). |
| `max_iter` | `int` | `1000` | >=1 | Maximum number of LBFGS iterations. |

## Returns

`HuberFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coef_` | `np.ndarray` | Fitted coefficient vector, shape (n_features,). |
| `.intercept_` | `float` | Fitted intercept scalar. |
| `.epsilon` | `float` | Huber loss transition point used. |
| `.scale_` | `float` | Robust scale estimate from the fitted model. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable text table of fit results. |

## Behavior

Replaces squared loss with the Huber loss: quadratic for small residuals, linear for large ones. Down-weights outliers without removing them. ``params.epsilon`` (default 1.35) sets the transition point.

**When to use**

Series with sporadic outliers that aren't worth flagging in L2.

## In recipe context

Set ``params.model = "huber"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: huber
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Huber (1964) 'Robust Estimation of a Location Parameter', Annals of Mathematical Statistics 35(1).

## Related ops

See also: `ols`, `ridge` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
