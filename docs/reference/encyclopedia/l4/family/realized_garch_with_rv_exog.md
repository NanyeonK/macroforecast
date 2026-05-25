<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `realized_garch_with_rv_exog` -- GARCH(1,1) with realised-variance series fed as the exogenous regressor (NOT Hansen-Huang-Shek 2012 joint MLE).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.realized_garch_fit`.

## Function signature

```python
mf.functions.realized_garch_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> RealizedGARCHFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`RealizedGARCHFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.conditional_mu` | `float` | Fitted conditional mean mu. |
| `.n_obs` | `int` | Number of non-missing observations. |
| `.params_` | `dict` | Fitted model parameters dict. |
| `.predict(X)` | `np.ndarray` | Conditional mean broadcast over len(X) rows. |
| `.predict_variance(h)` | `np.ndarray` | h-step-ahead variance forecast. |
| `.summary()` | `str` | Table: conditional mean and fitted parameters. |

## Behavior

Phase C-3 audit-fix (M9) honest rename. The L4 wrapper consumes ``params['realized_variance']`` (a column name in ``X``) as the RV series and feeds it as the **exogenous regressor** ``x=`` into a vanilla GARCH(1,1) spec. This is useful in practice (RV improves volatility forecasts), but it is **NOT** the Hansen-Huang-Shek (2012) joint return + measurement-equation MLE: there is no ``ξ``, ``φ``, ``δ_1``, ``δ_2`` measurement-equation parameters in the fitted output. The proper RealizedGARCH spec is reserved as FUTURE under the name ``realized_garch`` (awaiting native ``arch.RealizedGARCH`` API or manual joint-MLE implementation).

Returns the conditional mean as the point forecast; ``predict_variance(h_steps)`` exposes the variance path.

**Defaults**: ``mean_model = 'constant'``, ``dist = 'normal'``. Falls back to a squared-returns proxy when the RV column is unavailable.

**When to use**

Volatility forecasting when intraday realised variance is observable as a leading indicator (RV-as-exogenous improves vol forecast); honest baseline labelling for studies that need to distinguish from the proper Hansen-Huang-Shek MLE.

**When NOT to use**

When the proper joint-MLE Realized GARCH is required (the family name ``realized_garch`` is FUTURE / unrunnable until upstream supports it); without ``[arch]`` extra installed; without an RV measurement available.

## In recipe context

Set ``params.family = "realized_garch_with_rv_exog"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: realized_garch_with_rv_exog
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Hansen, Huang & Shek (2012) 'Realized GARCH: A Joint Model for Returns and Realized Measures of Volatility', Journal of Applied Econometrics 27(6): 877-906 — the *target* spec, not implemented here.

## Related ops

See also: `garch11`, `egarch` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
