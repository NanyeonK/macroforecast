# `egarch` -- Exponential GARCH with leverage asymmetry (Nelson 1991).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.egarch_fit`.

## Function signature

```python
mf.functions.egarch_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> EGARCHFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`EGARCHFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.conditional_mu` | `float` | Fitted conditional mean mu. |
| `.n_obs` | `int` | Number of non-missing observations. |
| `.params_` | `dict` | Fitted EGARCH parameters dict. |
| `.predict(X)` | `np.ndarray` | Conditional mean broadcast over len(X) rows. |
| `.predict_variance(h)` | `np.ndarray` | h-step-ahead variance forecast. |
| `.summary()` | `str` | Table: conditional mean and fitted parameters. |

## Behavior

EGARCH(p, o, q) on log-variance: ``ln σ²_t = ω + Σ α_i (|z_{t-i}| − E|z|) + Σ γ_i z_{t-i} + Σ β_j ln σ²_{t-j}``. The asymmetry term ``γ`` captures the leverage effect (negative shocks raise volatility more than positive ones), and the log specification removes any need for non-negativity constraints on the parameters.

**Defaults** (Nelson 1991 §3): ``p = o = q = 1``, ``mean_model = 'constant'``, ``dist = 'normal'``. Wraps ``arch.arch_model(vol='EGARCH')`` -- requires ``[arch]`` extra.

**When to use**

Asymmetric / leverage volatility; equity returns where bad news amplifies vol; macro variables with sign-asymmetric volatility responses.

**When NOT to use**

Without ``[arch]`` extra installed; symmetric volatility series where GARCH(1,1) is sufficient (parsimony).

## In recipe context

Set ``params.family = "egarch"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: egarch
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Nelson (1991) 'Conditional Heteroskedasticity in Asset Returns: A New Approach', Econometrica 59(2): 347-370.

## Related ops

See also: `garch11`, `realized_garch_with_rv_exog` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
