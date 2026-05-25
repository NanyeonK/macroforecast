<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `theta_method` -- Theta method (Assimakopoulos-Nikolopoulos 2000) -- M3-competition winning baseline.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.theta_fit`.

## Function signature

```python
mf.functions.theta_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> ThetaFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`ThetaFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.theta` | `float` | Theta parameter (default 2.0 = M3 winner). |
| `.alpha_` | `float` | Fitted SES smoothing parameter. |
| `.n_obs` | `int` | Number of observations. |
| `.predict(X)` | `np.ndarray` | Forecast, len(X) steps ahead. |
| `.summary()` | `str` | Table: theta, alpha, observation count. |

## Behavior

Hand-coded Theta(2) closed-form forecast: blends a long-run linear-trend regression with a short-run simple-exponential-smoothing (SES) level. For ``θ = 2`` (M3 winner), the h-step-ahead forecast is ``ŷ_{T+h} = 0.5 · (a + b · (T+h)) + 0.5 · ℓ_T``, where ``(a, b)`` are the OLS trend slope/intercept on time index and ``ℓ_T`` is the SES level at time T (smoothing parameter ``α`` selected via ``scipy.optimize.minimize_scalar`` minimising the in-sample 1-step MSE).

**Defaults**: ``theta = 2.0`` (M3 winner), ``seasonal = False``, ``seasonal_periods = 12``. The constructor exposes ``theta`` for forward compatibility; only the θ=2 closed form is exercised in v0.9.0 -- general θ requires a θ-line decomposition out of scope for this run.

**When to use**

M3 / M4-style univariate baselines; quick reference forecast against more elaborate models.

**When NOT to use**

Strongly seasonal series (use ``holt_winters`` or seasonally-adjusted target); covariate-driven forecasting.

## In recipe context

Set ``params.family = "theta_method"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: theta_method
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Assimakopoulos & Nikolopoulos (2000) 'The theta model: a decomposition approach to forecasting', International Journal of Forecasting 16(4): 521-530.
* Hyndman & Billah (2003) 'Unmasking the Theta method', International Journal of Forecasting 19(2): 287-290.
* Petropoulos et al. (2022) 'Forecasting: theory and practice', International Journal of Forecasting 38(3): 705-871.

## Related ops

See also: `ets`, `holt_winters`, `ar_p` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
