# `midas_almon` -- MIDAS with Almon polynomial lag weights (Ghysels-Santa-Clara-Valkanov 2004).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.midas_almon`.

## Function signature

```python
mf.functions.midas_almon(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    freq_ratio: int = 1,
    n_lags_high: int = 12,
    polynomial_order: int = 2,
    sum_to_one: bool = True,
    max_iter: int = 200,
    n_starts: int = 5,
    random_state: int = 0,
) -> _MidasAlmonModel
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `pd.DataFrame` | — | — | Feature matrix as a DataFrame. When ``freq_ratio = 1``: LF-aligned lag columns (one column per lag feature). When ``freq_ratio > 1``: raw HF DataFrame that is internally lag-stacked. Shape (n_samples, n_features). |
| `y` | `pd.Series` | — | — | Low-frequency target series aligned to the LF index. |
| `freq_ratio` | `int` | `1` | >=1 | High-frequency periods per low-frequency period (m). 1 = X is already LF-aligned (primary path). >1 = model internally lag-stacks X via ``_midas_lag_stack``. |
| `n_lags_high` | `int` | `12` | >=1 | Number of high-frequency lags K to include in the weight vector. |
| `polynomial_order` | `int` | `2` | >=0 | Almon polynomial degree Q. Number of θ hyperparameters is Q+1. |
| `sum_to_one` | `bool` | `True` | — | Normalize lag weights to sum to 1 after non-negativity clamp. |
| `max_iter` | `int` | `200` | — | Maximum Nelder-Mead iterations per start. |
| `n_starts` | `int` | `5` | >=1 | Number of NLS restarts. Start 0 is canonical; remaining starts perturb θ. |
| `random_state` | `int` | `0` | — | RNG seed for perturbed NLS starts. Propagated from L0 via per-origin RNG contract (#279). |

## Returns

`_MidasAlmonModel` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `._w_hat` | `np.ndarray` | Estimated Almon lag weights, shape (n_lags_high,). Zero-padded when K_eff < n_lags_high. |
| `._theta_hat` | `np.ndarray` | Estimated Almon polynomial coefficients, shape (Q+1,). |
| `._intercept` | `float` | Fitted intercept mu. |
| `._slope` | `float` | Fitted overall slope beta. |
| `._converged` | `bool` | NLS convergence flag (True if any start converged). |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_rows,). |

## Behavior

Estimates mixed-frequency regressions via non-linear least squares (Nelder-Mead) on Almon polynomial lag weights. The weight function ``b(k; θ) = Σ_{q=0}^{Q} θ_q · k^q`` (Ghysels, Santa-Clara & Valkanov 2004, §2 eq. 3) maps lag index ``k`` to a scalar weight; weights are clamped to non-negative values and optionally normalized to sum to one. The full parameter vector is ``(θ_0, ..., θ_Q, μ, β)`` where ``μ`` is the intercept and ``β`` is the overall slope multiplier.

``params.freq_ratio`` (default 1) governs the mixed-frequency contract. When ``freq_ratio > 1`` the model internally calls ``_midas_lag_stack`` to build the high-frequency lag design matrix from a raw HF DataFrame. When ``freq_ratio = 1`` (default) X is treated as already low-frequency aligned -- this is the primary usage path following an upstream L3 ``midas`` or ``u_midas`` op.

Multi-start NLS uses ``params.n_starts`` restarts (default 5): start 0 is canonical (θ_0 = 1, rest zero → flat weights); remaining starts perturb θ from ``N(0, 0.1)``. Seed propagated from L0 via the per-origin RNG contract (#279).

Two weight attributes are maintained: ``_w_hat`` (length ``n_lags_high``, zero-padded when ``K_eff < n_lags_high``) for external inspection; ``_w_hat_effective`` (length ``K_eff``) for internal predict matmul. This resolves the ``n_lags_high != X.shape[1]`` edge case at ``freq_ratio = 1``.

**When to use**

Mixed-frequency macro forecasting where monthly or weekly predictors inform a quarterly target. Parsimonious alternative to U-MIDAS when K is large relative to T.

**When NOT to use**

Very short samples (T < K + Q + 3) -- the model falls back to uniform weights and mean intercept. When frequency alignment has already been handled by an upstream L3 op and ``freq_ratio = 1`` is sufficient, prefer ``midas_step`` (OLS, cheaper) or ``dfm_unrestricted_midas`` (U-MIDAS, more flexible).

## In recipe context

Set ``params.family = "midas_almon"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: midas_almon
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Ghysels, Santa-Clara & Valkanov (2004) 'The MIDAS Touch: Mixed Data Sampling Regression Models', CIRANO Working Paper.

## Related ops

See also: `midas_beta`, `midas_step`, `dfm_unrestricted_midas` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
