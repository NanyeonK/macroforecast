# Mathematical Definitions: Statistical Tests

This reference groups the tests implemented in `macrocast.execution.build` by the Phase 2 axis they belong to. For each test we give the null, the statistic, and the distribution used for the p-value.

All tests take a long-form `predictions` table whose columns include `y_true`, `y_pred`, `benchmark_pred`, `squared_error`, `benchmark_squared_error`, `origin_date`, `horizon`, `target`, `model_name`.

---

## Axis: `equal_predictive`

Loss-differential tests for two non-nested model forecasts $\hat{y}^A$ and $\hat{y}^B$. Let $d_t = L(y_t, \hat{y}^A_t) - L(y_t, \hat{y}^B_t)$ be the loss differential.

### `dm` — Diebold-Mariano (1995)
- $H_0$: $E[d_t] = 0$.
- Statistic: $\mathrm{DM} = \bar d / \sqrt{\hat V(\bar d)/n}$ with $\hat V$ a simple sample variance.
- Distribution: standard normal.

### `dm_hln` — DM with Harvey-Leybourne-Newbold (1997) small-sample correction
- Same $H_0$ and statistic as `dm`, multiplied by $\sqrt{(n + 1 - 2h + h(h-1)/n)/n}$.
- Distribution: $t$-distribution with $n-1$ degrees of freedom.

### `dm_modified`
- `dm_hln` variant with a HAC variance estimate (Newey-West) for $\hat V(\bar d)$. Suited for long-horizon forecasts where $d_t$ has persistent autocorrelation.

---

## Axis: `nested`

Nested-model tests. Let model B nest model A ($A \subset B$). Clark-West and ENC-NEW correct for the downward bias in MSE differentials when comparing nested models.

### `cw` — Clark-West (2007)
- $H_0$: parent model A has the same out-of-sample MSFE as the nested B.
- Statistic: adjusted loss differential $\tilde d_t = (\hat{y}^A_t - \hat{y}^B_t)^2 - 2(\hat{y}^A_t - \hat{y}^B_t)(y_t - \hat{y}^B_t)$; test is one-sided $t$ on $\bar{\tilde d}$.
- Distribution: standard normal.

### `enc_new` — Clark-McCracken (2001) forecast encompassing
- $H_0$: B encompasses A.
- Statistic: $\mathrm{ENC\text{-}NEW} = n \cdot \sum_t (\hat{y}^A_t - \hat{y}^B_t)(y_t - \hat{y}^A_t) / \sum_t (y_t - \hat{y}^B_t)^2$.
- Distribution: non-standard; macrocast uses the Clark-McCracken asymptotic critical values.

### `mse_f` / `mse_t`
- F and t statistics on $\mathrm{MSFE}_A - \mathrm{MSFE}_B$. See Clark & McCracken (2001) for critical values.

---

## Axis: `cpa_instability`

Conditional predictive ability and forecast stability.

### `cpa` — Giacomini-White (2006)
- $H_0$: constant-only conditional predictive ability — $E[d_t | \mathcal{F}_{t-1}] = 0$.
- Statistic: Wald on regression of $d_t$ on an intercept.
- Distribution: $\chi^2_1$ (k instruments = 1).

### `rossi` — Rossi-Sekhposyan forecast stability
- $H_0$: no change in predictive ability over the out-of-sample period.
- Statistic: maximum of a rolling DM-type statistic over the sample.
- Distribution: simulated critical values.

### `rolling_dm`
- Summary: rolling-window DM applied with a fixed window width; returns min/max/mean statistic across windows.

---

## Axis: `multiple_model`

Tests comparing a set of candidate models against a benchmark (or each other).

### `reality_check` — White (2000)
- $H_0$: no model strictly beats the benchmark.
- Statistic: $\max_k \sqrt{n}(\overline{L_{bench}} - \overline{L_k})$ across $k$ candidate models.
- Distribution: stationary-bootstrap p-value.

### `spa` — Hansen (2005) Superior Predictive Ability
- Variant of Reality Check that is not conservative when some candidate models are dominated. Same bootstrap, different statistic.

### `mcs` — Model Confidence Set, Hansen-Lunde-Nason (2011)
- Produces a set $\widehat{\mathcal{M}}$ of models not significantly beaten by any other model at a given confidence level.
- Iterative elimination using $T_R$ or $T_{\max}$ statistic; bootstrap p-value.

---

## Axis: `direction`

### `pesaran_timmermann` — Pesaran-Timmermann (1992)
- $H_0$: forecast sign and realized sign are independent.
- Statistic: $(\hat P - \hat P^*) / \sqrt{\widehat{\mathrm{Var}}(\hat P - \hat P^*)}$ where $\hat P$ is the hit rate and $\hat P^*$ the hit rate expected under independence.
- Distribution: standard normal.

### `binomial_hit`
- $H_0$: hit rate = 0.5 (naive).
- Statistic: binomial test on the count of correctly-signed forecasts.

---

## Axis: `residual_diagnostics`

Diagnostics run on forecast errors $e_t = y_t - \hat{y}_t$.

### `mincer_zarnowitz`
- Regression $y_t = \alpha + \beta \hat{y}_t + u_t$; joint test of $\alpha = 0, \beta = 1$ ($F$-distributed).

### `ljung_box`
- Portmanteau test of serial correlation in $\{e_t\}$ at lag $m$; asymptotically $\chi^2_m$.

### `arch_lm`
- Engle (1982) ARCH-LM: regress $e_t^2$ on $m$ lags; test joint significance. Asymptotically $\chi^2_m$.

### `bias_test`
- One-sample $t$-test on $\bar e = 0$.

### `diagnostics_full`
- Bundle: runs `mincer_zarnowitz`, `ljung_box`, `arch_lm`, `bias_test` and packs all statistics under one JSON payload.

---

## Axis: `density_interval` (Phase 10 §10.8)

No operational entries in v0.4. Planned entries: `PIT_uniformity`, `berkowitz`, `kupiec`, `christoffersen_{unconditional,independence,conditional}`, `interval_coverage`. Mathematical definitions will land with Phase 10 §10.8 (v1.1).

---

## Axis: `test_scope`

Meta-control, not a test. Values choose the slicing at which other axes' tests are applied:

- `per_target` — run once per target
- `per_horizon` — run once per (target, horizon)
- `per_model_pair` — pairwise across all model pairs

Other values (`full_grid_pairwise`, `benchmark_vs_all`, `regime_specific_tests`, `subsample_tests`) are `planned` — see Phase 10 §10.8.

---

## References

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
- Harvey, D. I., Leybourne, S. J., & Newbold, P. (1997). Testing the equality of prediction mean squared errors.
- Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models.
- Clark, T. E., & McCracken, M. W. (2001). Tests of equal forecast accuracy and encompassing for nested models.
- Giacomini, R., & White, H. (2006). Tests of conditional predictive ability.
- Rossi, B., & Sekhposyan, T. (2016). Forecast rationality tests in the presence of instabilities.
- White, H. (2000). A reality check for data snooping.
- Hansen, P. R. (2005). A test for superior predictive ability.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
- Pesaran, M. H., & Timmermann, A. (1992). A simple nonparametric test of predictive performance.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation.
