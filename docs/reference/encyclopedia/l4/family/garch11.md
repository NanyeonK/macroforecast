<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `garch11` -- GARCH(1,1) univariate conditional-variance model (Bollerslev 1986).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.garch11_fit`.

## Function signature

```python
mf.functions.garch11_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> GARCH11FitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`GARCH11FitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.conditional_mu` | `float` | Fitted conditional mean mu. |
| `.n_obs` | `int` | Number of non-missing observations. |
| `.params_` | `dict` | Fitted GARCH parameters dict. |
| `.predict(X)` | `np.ndarray` | Conditional mean broadcast over len(X) rows. |
| `.predict_variance(h)` | `np.ndarray` | h-step-ahead variance forecast. |
| `.summary()` | `str` | Table: conditional mean and fitted parameters. |

## Behavior

Standard GARCH(1,1) volatility model: ``σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}``. The L4 wrapper treats ``y`` as the return-like series and ignores ``X``; ``predict(X)`` returns the conditional mean (μ broadcast over ``len(X)``) and the variance forecast is exposed via ``predict_variance(h_steps)`` for L7 inspection.

**Defaults** (paper-faithful, Bollerslev 1986 §3): ``p = q = 1``, ``mean_model = 'constant'``, ``dist = 'normal'``. Wraps ``arch.arch_model`` -- requires the optional ``[arch]`` extra (``pip install macroforecast[arch]``); raises ``NotImplementedError`` with an install hint when missing.

**When to use**

Macro / financial volatility forecasting; baseline GARCH benchmark; volatility-targeting risk applications.

**When NOT to use**

Without ``[arch]`` extra installed -- raises a clear NotImplementedError.

## In recipe context

Set ``params.family = "garch11"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: garch11
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Bollerslev (1986) 'Generalized Autoregressive Conditional Heteroskedasticity', Journal of Econometrics 31(3): 307-327.
* Engle (1982) 'Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation', Econometrica 50(4): 987-1007.

## Related ops

See also: `egarch`, `realized_garch_with_rv_exog` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
