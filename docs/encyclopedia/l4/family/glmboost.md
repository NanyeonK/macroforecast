# `glmboost` -- Componentwise L2-boosting with linear base learners.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.glmboost_fit`.

## Function signature

```python
mf.functions.glmboost_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_iter: int = 100,
    learning_rate: float = 0.1,
) -> GLMBoostFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |
| `n_iter` | `int` | `100` | >=1 | Number of boosting iterations. More iterations = finer coefficient path. |
| `learning_rate` | `float` | `0.1` | >0 | Shrinkage factor applied to each coefficient update. Smaller = slower convergence, more regularisation. |

## Returns

`GLMBoostFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coef_` | `np.ndarray` | Fitted coefficient vector, shape (n_features,). |
| `.intercept_` | `float` | Fitted intercept scalar (initialised to mean(y)). |
| `.n_iter` | `int` | Number of boosting iterations used. |
| `.learning_rate` | `float` | Shrinkage factor used. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable text table of fit results. |

## Behavior

Bühlmann-Hothorn (2007) componentwise boosting: at each iteration picks the predictor most correlated with the residual and updates only its coefficient. Approximates lasso with a boosting interpretation.

**When to use**

Transparent feature-selection pathways; alternative to lasso.

## In recipe context

Set ``params.family = "glmboost"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: glmboost
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Bühlmann & Hothorn (2007) 'Boosting algorithms: Regularization, prediction and model fitting', Statistical Science 22(4).

## Related ops

See also: `lasso`, `elastic_net` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
