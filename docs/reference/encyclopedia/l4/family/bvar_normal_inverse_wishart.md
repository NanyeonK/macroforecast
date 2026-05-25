<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `bvar_normal_inverse_wishart` -- Bayesian VAR with Normal-Inverse-Wishart prior.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.bvar_niw_fit`.

## Function signature

```python
mf.functions.bvar_niw_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> BVARNIWFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`BVARNIWFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_lags` | `int` | VAR lag order p. |
| `.n_obs` | `int` | Number of observations. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: lag order and observation count. |

## Behavior

Conjugate Normal-IW prior on (β, Σ); the posterior mean of β has the same closed form as Minnesota but with the prior tightness scaled to reflect parameter-uncertainty inflation. Slightly less aggressive than the bare Minnesota prior.

**When to use**

Studies preferring a fully-conjugate prior over Litterman's hand-tuned shrinkage.

## In recipe context

Set ``params.family = "bvar_normal_inverse_wishart"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: bvar_normal_inverse_wishart
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Kadiyala & Karlsson (1997) 'Numerical Methods for Estimation and Inference in Bayesian VAR-models', Journal of Applied Econometrics 12(2).

## Related ops

See also: `bvar_minnesota`, `var` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
