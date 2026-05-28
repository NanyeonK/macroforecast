# `elastic_net` -- Elastic net (L1 + L2 hybrid).

[Back to `model` axis](../axes/model.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `model`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.elastic_net_fit`.

## Function signature

```python
mf.functions.elastic_net_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 20000,
) -> ElasticNetFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |
| `alpha` | `float` | `1.0` | >=0 | Overall regularisation strength. |
| `l1_ratio` | `float` | `0.5` | in [0.0, 1.0] | L1/L2 mixing parameter. 0 = pure ridge, 1 = pure lasso. |
| `max_iter` | `int` | `20000` | >=1 | Maximum number of coordinate descent iterations. |

## Returns

`ElasticNetFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.coef_` | `np.ndarray` | Fitted coefficient vector, shape (n_features,). |
| `.intercept_` | `float` | Fitted intercept scalar. |
| `.alpha` | `float` | Regularisation strength used. |
| `.l1_ratio` | `float` | L1/L2 mixing parameter used. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Human-readable text table of fit results. |

## Behavior

Combines ridge and lasso penalties via ``params.l1_ratio`` (0 = ridge, 1 = lasso). Useful when predictors are correlated and pure lasso struggles with the selection.

**When to use**

Correlated predictor blocks where lasso alone gives unstable selection.

## In recipe context

Set ``params.model = "elastic_net"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: elastic_net
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Zou & Hastie (2005) 'Regularization and variable selection via the elastic net', JRSS-B 67(2).

## Related ops

See also: `ridge`, `lasso` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
