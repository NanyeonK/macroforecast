# `midas_step` -- MIDAS with piecewise-constant step-function weights, OLS (Foroni-Marcellino-Schumacher 2015).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.midas_step`.

## Function signature

```python
mf.functions.midas_step(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    freq_ratio: int = 1,
    n_lags_high: int = 12,
    n_steps: int = 'freq_ratio',
) -> _MidasStepModel
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `pd.DataFrame` | — | — | Feature matrix as a DataFrame. When ``freq_ratio = 1``: LF-aligned lag columns grouped by position. When ``freq_ratio > 1``: raw HF DataFrame internally lag-stacked. |
| `y` | `pd.Series` | — | — | Low-frequency target series aligned to the LF index. |
| `freq_ratio` | `int` | `1` | >=1 | High-frequency periods per low-frequency period. 1 = X already LF-aligned. |
| `n_lags_high` | `int` | `12` | >=1 | Number of high-frequency lags K to include. |
| `n_steps` | `int` | `'freq_ratio'` | >=1 | Number of piecewise-constant groups S. Defaults to ``freq_ratio`` (one group per HF sub-period). Determines the coarseness of the lag weight profile. |

## Returns

`_MidasStepModel` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `._step_coef` | `np.ndarray` | OLS step-group coefficients, shape (n_steps,). |
| `._intercept` | `float` | OLS intercept. |
| `._group_boundaries` | `list[tuple[int,int]]` | Lag index boundaries per group, e.g. [(0,4),(4,8),(8,12)]. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_rows,). |

## Behavior

Restricted MIDAS with a step-function lag weight profile. The K HF lags are partitioned into S equal-size groups; within each group all lags share one coefficient estimated by OLS on the group-mean aggregate (Foroni, Marcellino & Schumacher 2015, §2.2). Group boundaries: group s covers lags in ``[s*K//S, (s+1)*K//S)``.

No stochastic initialization (OLS is deterministic). Closed-form via ``numpy.linalg.lstsq(rcond=None)``.

``params.n_steps`` defaults to ``freq_ratio`` (one group per HF sub-period). ``params.freq_ratio = 1`` treats X columns as a flat lag sequence grouped by position index.

**When to use**

Fast, interpretable mixed-frequency baseline. Useful when K is large and only the coarse lag structure matters. No NLS overhead.

**When NOT to use**

When the lag weight profile is smooth (use ``midas_almon`` or ``midas_beta``) or when the fully-flexible U-MIDAS parameterization is affordable (use ``dfm_unrestricted_midas``).

## In recipe context

Set ``params.family = "midas_step"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: midas_step
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Foroni, Marcellino & Schumacher (2015) 'Unrestricted Mixed Data Sampling (MIDAS): MIDAS Regressions with Unrestricted Lag Polynomials', Journal of the Royal Statistical Society: Series A 178(1).

## Related ops

See also: `midas_almon`, `midas_beta`, `dfm_unrestricted_midas` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
