<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `ets` -- Exponential Smoothing State-Space (Hyndman-Koehler-Ord-Snyder 2008) -- ETS family.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.ets_fit`.

## Function signature

```python
mf.functions.ets_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> ETSFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`ETSFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.error_trend_seasonal` | `str` | 3-character ETS code, e.g. AAN. |
| `.n_obs` | `int` | Number of observations. |
| `.predict(X)` | `np.ndarray` | Forecast, len(X) steps ahead. |
| `.summary()` | `str` | Table: ETS code and observation count. |

## Behavior

Exponential-smoothing state-space framework: ``error_trend_seasonal`` is a 3-character code ``ETS`` where ``E ∈ {A, M}`` (additive / multiplicative error), ``T ∈ {A, M, N}`` (additive / multiplicative / no trend), ``S ∈ {A, M, N}`` (additive / multiplicative / no seasonality). Wraps ``statsmodels.tsa.exponential_smoothing.ets.ETSModel`` (MLE fitting; auto-selects the closed-form initialisation per Hyndman 2008 §5.4).

**Defaults**: ``error_trend_seasonal = 'AAN'`` (additive error, additive trend, no seasonal -- the workhorse non-seasonal spec), ``seasonal_periods = 12`` (monthly), ``initialization_method = 'estimated'``. Auto-disables seasonal fitting when ``len(y) < 2 · seasonal_periods``.

**When to use**

M-competition baseline; non-seasonal / seasonal univariate forecasting where a state-space exponential-smoothing model is the natural reference.

**When NOT to use**

Multivariate or covariate-driven forecasting (ETS ignores ``X``); short series where seasonal estimation is unstable.

## In recipe context

Set ``params.family = "ets"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: ets
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Hyndman, Koehler, Ord & Snyder (2008) 'Forecasting with Exponential Smoothing: The State Space Approach', Springer.
* Hyndman & Athanasopoulos (2018) 'Forecasting: Principles and Practice', 2nd ed., OTexts §7.

## Related ops

See also: `theta_method`, `holt_winters`, `ar_p` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
