# `dfm_unrestricted_midas` -- Unrestricted MIDAS (U-MIDAS) -- OLS on all HF lags (Foroni-Marcellino-Schumacher 2015).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.dfm_unrestricted_midas`.

## Function signature

```python
mf.functions.dfm_unrestricted_midas(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    freq_ratio: int = 1,
    n_lags_high: int | str enum {"bic", "aic"} = '"bic"',
    include_y_lag: bool = False,
    random_state: int = 0,
) -> _UnrestrictedMidasModel
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `pd.DataFrame` | — | — | Feature matrix as a DataFrame. When ``freq_ratio = 1``: LF-aligned lag columns. When ``freq_ratio > 1``: raw HF DataFrame internally lag-stacked to K lags. |
| `y` | `pd.Series` | — | — | Low-frequency target series aligned to the LF index. |
| `freq_ratio` | `int` | `1` | >=1 | High-frequency periods per low-frequency period. 1 = X already LF-aligned. |
| `n_lags_high` | `int | str enum {"bic", "aic"}` | `'"bic"'` | — | Lag order K. Integer fixes K; ``'bic'`` or ``'aic'`` selects K via information criterion (``_bic_select_k``). When ``freq_ratio = 1`` and IC selected, defaults to ``X.shape[1]``. |
| `include_y_lag` | `bool` | `False` | — | Include lagged dependent variable y_{t-1} as an additional predictor (eq. 20 in Foroni et al. 2015). Yields a mixed-frequency ADL specification. |
| `random_state` | `int` | `0` | — | Accepted for API symmetry with NLS families; unused (OLS is deterministic). |

## Returns

`_UnrestrictedMidasModel` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `._coef` | `np.ndarray` | OLS coefficient vector, shape (K_eff + 1,) with intercept at index 0. |
| `._intercept` | `float` | OLS intercept (= _coef[0]). |
| `._K_fit` | `int` | Resolved number of HF lags K used in fitting. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_rows,). |

## Behavior

Unrestricted mixed-data sampling regression: every HF lag enters linearly with its own OLS coefficient ``ψ_k`` (Foroni, Marcellino & Schumacher 2015, §3 eq. 7). The lag polynomial is fully flexible with no shape restriction, making U-MIDAS equivalent to OLS on the lag-stacked design matrix.

``params.n_lags_high`` accepts an integer (fixed K), ``'bic'``, or ``'aic'`` for information-criterion lag selection (Marcellino & Schumacher 2010; the pre-existing ``_bic_select_k`` helper is reused). When ``freq_ratio = 1``, BIC/AIC selection defaults to ``K = X.shape[1]`` (all available columns).

``params.include_y_lag = True`` augments the design matrix with the lagged dependent variable ``y_{t-1}`` (eq. 20), yielding a mixed-frequency ADL specification. In predict, the last observed ``y`` value is used as the y-lag.

OLS via ``numpy.linalg.lstsq(rcond=None)``. No stochastic step; ``random_state`` is accepted for API symmetry but unused.

**When to use**

Flexible mixed-frequency benchmark when T is large relative to K. BIC/AIC lag selection avoids manual K tuning. Pairs well with upstream L3 ``u_midas`` for feature preprocessing.

**When NOT to use**

When T is small relative to K (use ``midas_almon`` or ``midas_beta`` for parsimonious alternatives). The 'dfm_' prefix is a historical naming artifact -- this is not a dynamic factor model.

## In recipe context

Set ``params.family = "dfm_unrestricted_midas"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: dfm_unrestricted_midas
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Foroni, Marcellino & Schumacher (2015) 'Unrestricted Mixed Data Sampling (MIDAS): MIDAS Regressions with Unrestricted Lag Polynomials', Journal of the Royal Statistical Society: Series A 178(1).
* Marcellino & Schumacher (2010) 'Factor MIDAS for Nowcasting and Forecasting with Ragged-Edge Data', Journal of Applied Econometrics 25(7).

## Related ops

See also: `midas_almon`, `midas_beta`, `midas_step`, `dfm_mixed_mariano_murasawa` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
