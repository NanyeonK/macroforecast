# `realized_garch` -- Hansen-Huang-Shek (2012) Realized GARCH -- joint return + measurement MLE.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.realized_garch`.

## Function signature

```python
mf.functions.realized_garch(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    realized_variance: str | None = None,
    mean_model: str enum {"constant"} = '"constant"',
    dist: str enum {"normal"} = '"normal"',
    max_iter: int = 2000,
    n_starts: int = 3,
    random_state: int = 0,
) -> RealizedGARCHFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |
| `realized_variance` | `str | None` | `None` | — | Column name in X containing the realized-variance series (e.g. 5-minute RV or bipower variation). If None or if the column is missing / all-NaN, falls back to r^2 as a realized-variance proxy (degrades estimation quality relative to a true RV measure). |
| `mean_model` | `str enum {"constant"}` | `'"constant"'` | — | Mean model for the return equation. Only 'constant' is supported in C49 (AR-mean deferred to a future cycle). Raises ValueError for any other value. |
| `dist` | `str enum {"normal"}` | `'"normal"'` | — | Error distribution. Only 'normal' is supported in C49 (fat-tail extensions deferred). Raises ValueError for any other value. |
| `max_iter` | `int` | `2000` | >=1 | Maximum L-BFGS-B iterations per optimization start. |
| `n_starts` | `int` | `3` | >=1 | Number of multi-start optimization restarts. Start 0 uses canonical initialization; subsequent starts perturb the parameter vector by N(0, 0.05). Best objective is selected. |
| `random_state` | `int` | `0` | — | RNG seed for multi-start perturbations. Propagated from L0 via the per-origin RNG contract (#279): random_state = base_seed + origin_position. |

## Returns

`RealizedGARCHFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.conditional_mu` | `float` | Fitted conditional mean mu from the return equation. |
| `.n_obs` | `int` | Number of aligned non-NaN return/RV observations used in fitting. |
| `.params_` | `dict` | Fitted parameter dict with keys: mu, omega, beta, tau_1, tau_2, gamma, xi, phi, delta_1, delta_2, sigma_u (all float). |
| `.conditional_volatility_` | `np.ndarray | None` | In-sample conditional volatility sqrt(h_t) series, shape (T_fit,). None before fit(). |
| `.predict(X)` | `np.ndarray` | Conditional mean broadcast over len(X) rows. |
| `.predict_variance(h)` | `np.ndarray` | h-step-ahead variance forecast, shape (h,). |
| `.summary()` | `str` | Table: conditional mean, observation count, fitted parameters. |

## Behavior

True Hansen, Huang & Shek (2012) Realized GARCH: a three-equation joint system for returns and a realized-variance measurement, estimated by maximum likelihood. All 11 parameters are recovered simultaneously via ``scipy.optimize.minimize(method='L-BFGS-B')``. No ``arch`` package dependency -- depends only on NumPy and SciPy.

**Model equations** (Hansen et al. 2012, JAE 27(6): 877-906):
* **Return**: ``r_t = mu + sqrt(h_t) * z_t``, ``z_t ~ N(0, 1)``
* **Log-variance**: ``log(h_t) = omega + beta * log(h_{t-1}) + tau_1 * z_{t-1} + tau_2 * (z_{t-1}^2 - 1) + gamma * u_{t-1}``
* **Measurement**: ``log(x_t) = xi + phi * log(h_t) + delta_1 * z_t + delta_2 * (z_t^2 - 1) + u_t``, ``u_t ~ N(0, sigma_u^2)``

**Parameter vector** (length 11): ``(mu, omega, beta, tau_1, tau_2, gamma, xi, phi, delta_1, delta_2, log_sigma_u)``.

**Multi-start NLS**: ``params.n_starts`` restarts (default 3). Start 0 uses canonical initialization (``h_0 = max(var(r), 1e-6)``, ``u_0 = z_0 = 0``). Subsequent starts perturb theta by ``N(0, 0.05)`` via ``numpy.random.default_rng(random_state + start_index)``. Best objective selected.

**RV column**: ``params.realized_variance`` specifies the column name in ``X`` containing the realized-variance series. If missing or all-NaN, falls back to ``r^2`` as a proxy (WARNING: proxy degrades estimation quality).

**Numerical stability**: ``log_h_t`` clipped to ``[-30, 30]`` before ``exp()``; ``h_t`` clipped to ``min 1e-8``; ``log_sigma_u`` clipped to ``[-10, 10]``; ``x_t`` clipped to ``[1e-8, inf)`` before taking log.

**``random_state`` propagation**: seed injected via the #215/#279 contract (``params['random_state'] = base_seed + origin_position`` at materialize time).

**When to use**

Volatility forecasting studies where high-frequency realized-variance measures (e.g., 5-minute RV, BPV, QV) are available as inputs. Produces superior volatility forecasts relative to GARCH(1,1) and EGARCH in the presence of a reliable RV series. Appropriate for equity, FX, and commodity return panels paired with corresponding RV data.

**When NOT to use**

When no realized-variance series is available (model falls back to r^2 proxy, which is equivalent to a vanilla GARCH without the measurement equation benefit). When only the approximate RV-as-exogenous approach is needed, ``realized_garch_with_rv_exog`` is simpler and requires fewer parameters. When ``len(y) < 30`` -- raises ``NotImplementedError`` with an informative message.

## In recipe context

Set ``params.family = "realized_garch"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: realized_garch
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Hansen, Huang & Shek (2012) 'Realized GARCH: A Joint Model for Returns and Realized Measures of Volatility', Journal of Applied Econometrics 27(6): 877-906.
* Bollerslev (1986) 'Generalized Autoregressive Conditional Heteroskedasticity', Journal of Econometrics 31(3): 307-327.

## Related ops

See also: `realized_garch_with_rv_exog`, `garch11`, `egarch` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
