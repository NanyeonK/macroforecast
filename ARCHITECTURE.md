# macroforecast — Architecture

Version: **v0.9.2b1**. Package root: `macroforecast/`.

---

## Top-level module layout

| Module | Role |
|--------|------|
| `macroforecast.api` | Public entry points: `mf.run`, `mf.replicate`, `mf.forecast` |
| `macroforecast.api_high` | High-level `Experiment` class and `ForecastResult` |
| `macroforecast.core` | DAG runtime, layer schemas, op registries, sweep, cache, manifest |
| `macroforecast.scaffold` | Option-doc registry and encyclopedia generator |
| `macroforecast.functions` | 118 standalone callables organized by layer (L2–L7) |
| `macroforecast.raw` | FRED-MD/QD/SD adapters, vintage manager, raw manifest |
| `macroforecast.preprocessing` | Preprocessing contract helpers |
| `macroforecast.custom` | User-defined model, preprocessor, and target transformer registration |
| `macroforecast.defaults` | Default profile dict template |
| `macroforecast.tuning` | Hyperparameter search engines (optional, integrated via L4) |

---

## 12-layer canonical design (L0–L8 + diagnostic half-layers)

| Layer | Purpose | Primary module paths |
|-------|---------|----------------------|
| L0 | Study setup: failure policy, seed, compute mode, study scope | `core/layers/l0.py` |
| L1 | Data definition: FRED-MD/QD/SD source, target y, predictor x, regime, availability | `core/layers/l1.py` |
| L1.5 | Diagnostic hook (default-off): stationarity tests, raw data summaries | `core/layers/l1_5.py` |
| L2 | Preprocessing: transform, outlier, imputation, frame edge, mixed-frequency alignment | `core/layers/l2.py` |
| L2.5 | Diagnostic hook (default-off): preprocessed panel summaries | `core/layers/l2_5.py` |
| L3 | Feature engineering DAG: 41 ops (lags, factors, filters, selection, targets) | `core/layers/l3.py`, `core/ops/l3_ops.py` |
| L3.5 | Diagnostic hook (default-off): feature distribution summaries | `core/layers/l3_5.py` |
| L4 | Forecasting model + tuning: 47 operational families, 5 combine ops | `core/layers/l4.py`, `core/ops/l4_ops.py` |
| L4.5 | Diagnostic hook (default-off): residual diagnostics, fitted-vs-actual | `core/layers/l4_5.py` |
| L5 | Evaluation: metrics, benchmarks, decomposition, aggregation, ranking | `core/layers/l5.py`, `core/ops/l5_ops.py` |
| L6 | Statistical tests: DM/HLN, CW, MCS/SPA/RC/StepM, PT/HM, residual battery, density tests | `core/layers/l6.py`, `core/ops/l6_ops.py` |
| L7 | Interpretation: 35 importance ops, group aggregate, lineage, transformation attribution | `core/layers/l7.py`, `core/ops/l7_ops.py` |
| L8 | Output and provenance: json/csv/parquet/latex/markdown, manifest, saved objects | `core/layers/l8.py`, `core/ops/l8_ops.py` |

Layers L6 and L7 are default-off and require explicit `enabled: true`. Layer L8
is always on. The diagnostic half-layers L1.5, L2.5, L3.5, L4.5 are
default-off and non-blocking.

---

## 2-paradigm model

macroforecast exposes two complementary access patterns.

**Recipe DSL** (`mf.run`). A YAML recipe fully specifies a study as an
end-to-end DAG from L0 through L8. Sweep markers expand the recipe into
independent cells. Every run writes a manifest with per-cell sink hashes and
supports bit-exact replication via `mf.replicate(manifest_path)`. This
paradigm is appropriate for reproducible comparative studies.

**Standalone callables** (`mf.functions.*`). Individual operations from L2
through L7 are also available as direct Python callables requiring no YAML.
They accept NumPy arrays or DataFrames and return frozen dataclasses with typed
result attributes. This paradigm is appropriate for exploratory analysis,
notebook workflows, and integration into custom pipelines.

See [docs/two_entry_points.md](docs/two_entry_points.md) for a decision guide
on when to choose each paradigm.

---

## Core runtime module map

| Module | Role |
|--------|------|
| `core/execution.py` | Cell loop (`execute_recipe`), seed propagation, `replicate_recipe` |
| `core/runtime.py` | Per-layer `materialize_l*` helpers; layer artifact construction |
| `core/layers/l0.py`–`l8.py` | Layer schema definitions (axes, gates, defaults, status) |
| `core/layers/l1_5.py`, `l2_5.py`, `l3_5.py`, `l4_5.py` | Diagnostic half-layer schemas |
| `core/ops/l3_ops.py`–`l8_ops.py` | Op registries per layer |
| `core/dag.py` | Universal DAG schema: 5 node types (source, axis, step, combine, sink) |
| `core/sweep.py` | Sweep expansion: param-level, recipe-level (external axis), node-level (sweep_groups) |
| `core/cache.py` | Content-addressed artifact cache; SHA-256 sink hash generation |
| `core/manifest.py` | Manifest read/write; provenance record schema |

Foundation contracts (typed artifacts, validator, YAML normalizer, recipe
schema, selector types) live in `core/types.py`, `core/validator.py`,
`core/yaml.py`, `core/recipe.py`, and `core/selectors.py`.

---

## Standalone functions summary

`macroforecast.functions` contains **118** standalone callables organized by
layer. Per-layer counts (verified via `tools/gen_standalone_docs.py` against
live `macroforecast.functions.__all__` at HEAD afc28282):

| Layer | Count | Module |
|-------|-------|--------|
| L2 | 14 | `functions/clean.py` |
| L3 | 36 | `functions/transforms.py` |
| L4 | 38 | `functions/fit.py` |
| L5 | 15 | `functions/metrics.py` |
| L6 | 7 | `functions/tests.py` |
| L7 | 8 | `functions/importance.py` |

For the full per-callable reference (signatures, result attributes,
examples), see [docs/standalone_functions/](docs/standalone_functions/index.md).

---

## Cycle 49 — realized_garch + Pesaran-Shin GIRF (2026-05-21)

Cycle 49 promotes two items from `future` to `operational`: the Hansen-Huang-Shek
(2012) Realized GARCH joint MLE (L4 family `realized_garch`) and the Pesaran-Shin
(1998) generalized impulse-response function (L7 op `generalized_irf`). After this
cycle, `FUTURE_MODEL_FAMILIES` is empty `()` and `FUTURE_OPS` contains only
`lstm_hidden_state`.

| Item | Layer | Paper | Implementation | Changed in C49 |
|------|-------|-------|----------------|----------------|
| `realized_garch` | L4 | Hansen, Huang & Shek (2012, JAE 27(6)) | `_RealizedGARCHModel` in `runtime.py`; scipy L-BFGS-B joint MLE, 11-parameter vector, multi-start with `random_state` | Yes |
| `generalized_irf` | L7 | Pesaran & Shin (1998, Economics Letters 58) | `_var_girf_frame` in `runtime.py`; `irf(var_decomp=eye(K))` to obtain raw reduced-form MA coefficients; `GIRF_h(j) = sigma_jj^{-1/2} * A_h * Sigma * e_j` | Yes |

### Count changes

| Counter | Pre-C49 | Post-C49 |
|---------|---------|---------|
| L4 operational families | 46 | 47 |
| L4 FUTURE_MODEL_FAMILIES | 1 (`realized_garch`) | 0 (empty) |
| L7 FUTURE_OPS | 2 (`lstm_hidden_state`, `generalized_irf`) | 1 (`lstm_hidden_state`) |

### Hansen-Huang-Shek (2012) `_RealizedGARCHModel`

Three-equation joint system estimated by `scipy.optimize.minimize(method="L-BFGS-B")`:

- **Return**: `r_t = mu + sqrt(h_t) * z_t`, `z_t ~ N(0,1)`
- **Log-variance**: `log(h_t) = omega + beta*log(h_{t-1}) + tau_1*z_{t-1} + tau_2*(z_{t-1}^2-1) + gamma*u_{t-1}`
- **Measurement**: `log(x_t) = xi + phi*log(h_t) + delta_1*z_t + delta_2*(z_t^2-1) + u_t`

Parameter vector (length 11): `(mu, omega, beta, tau_1, tau_2, gamma, xi, phi, delta_1, delta_2, log_sigma_u)`.
Multi-start `n_starts=3`; seeds derived from `random_state + start_index` following the #279 contract.
No `arch` package dependency — depends only on NumPy and SciPy.

Distinct from `realized_garch_with_rv_exog` (which feeds RV as an exogenous regressor into
a vanilla GARCH(1,1) via the `arch` package — a useful practical approximation but not the
Hansen-Huang-Shek joint MLE).

### Pesaran-Shin (1998) `_var_girf_frame`

Order-invariant IRF formula: `GIRF_h(j) = sigma_jj^{-1/2} * A_h * Sigma * e_j`

where `A_h = irf_obj.irfs[h]` (raw reduced-form MA coefficients, obtained via
`fitted_results.irf(n_periods, var_decomp=np.eye(K))` — the identity `var_decomp`
skips the Cholesky factorisation and returns raw MA matrices directly).

Importance metric: `importance[j] = sum_{h=0}^{H} |GIRF_h(j)[target_index]|` (L1 norm
of target-variable response across all horizons). Order-invariance verified to 1e-19 tolerance
(test: `test_generalized_irf_order_invariance_k3`).

Distinct from `orthogonalised_irf` (Cholesky-identified; order-dependent; operational since v0.2).

### References

- Hansen, Huang & Shek (2012) 'Realized GARCH: A Joint Model for Returns and Realized
  Measures of Volatility', Journal of Applied Econometrics 27(6): 877-906.
- Pesaran & Shin (1998) 'Generalized impulse response analysis in linear multivariate
  models', Economics Letters 58(1): 17-29.

## Cycle 48 — MIDAS Family Honesty Pass (2026-05-21)

Cycle 48 promotes four L4 families from `future` to `operational`, bringing
the L4 operational count from 42 to 46. Each family implements the published
mixed-data sampling (MIDAS) estimator using native SciPy/NumPy/statsmodels
primitives. No new external dependencies were added.

| Family | Paper | Estimation | Key params |
|--------|-------|-----------|------------|
| `midas_almon` | Ghysels, Santa-Clara & Valkanov (2004) "The MIDAS Touch", §2 eq. (3) | Multi-start NLS (Nelder-Mead) on Almon polynomial lag weights; `Q+1` hyperparameters | `freq_ratio`, `n_lags_high`, `polynomial_order`, `sum_to_one`, `n_starts`, `random_state` |
| `midas_beta` | Ghysels, Sinko & Valkanov (2007) "MIDAS Regressions", §2 Beta kernel | Multi-start NLS; 2 Beta shape parameters `a`, `b`; initial point `[1,1]` then Gamma-perturbed restarts | `freq_ratio`, `n_lags_high`, `sum_to_one`, `n_starts`, `random_state` |
| `midas_step` | Foroni, Marcellino & Schumacher (2015) "Unrestricted Mixed Data Sampling", §2.2 | OLS on step-aggregated design matrix; `S` piecewise-constant lag groups | `freq_ratio`, `n_lags_high`, `n_steps` |
| `dfm_unrestricted_midas` | Foroni, Marcellino & Schumacher (2015) §3 eq. (7) and eq. (20); Marcellino & Schumacher (2010) U-MIDAS | OLS with optional AR(1) y-lag; BIC/AIC or fixed-K lag selection | `freq_ratio`, `n_lags_high`, `include_y_lag`, `random_state` |

All four classes (`_MidasAlmonModel`, `_MidasBetaModel`, `_MidasStepModel`,
`_UnrestrictedMidasModel`) are inlined in `core/runtime.py` following the
existing pattern. They reuse the pre-existing `_midas_lag_stack` helper for
high-frequency lag construction and share the per-origin seed contract
established by issue #279 (`random_state = base_seed + origin_position`).

The L3 MIDAS feature-engineering ops (`midas`, `u_midas`) are separate and
unchanged: they produce LF-aggregated feature columns. The L4 MIDAS families
receive those (or raw HF data when `freq_ratio > 1`) and produce forecasts.

## Cycle 47 — L3 feature-selection honesty pass (2026-05-21)

Cycle 47 promoted five L3 feature-selection ops from `status="future"` to
`status="operational"`. This is the fourth honesty pass in the package's
history (following v0.1, v0.25, and v0.3 passes) and the first to target
the L3 selection sub-family exclusively.

### Promoted ops

Each op implements the published algorithm exactly. No approximations are
accepted under the `operational` label.

| Op | Reference | Procedure |
|-----|-----------|-----------|
| `boruta_selection` | Kursa & Rudnicki (2010), JSS 36(11) | Shadow-feature permutation test with RF mean-decrease-impurity importance; Bonferroni-corrected per-feature binomial acceptance/rejection iterated until all features decided or `max_iter` reached. |
| `recursive_feature_elimination` | Guyon, Weston, Barnhill & Vapnik (2002), Machine Learning 46 | Recursive backward elimination by squared coefficient magnitude; optional CV step for automatic `n_features_to_select` selection (RFECV extension). |
| `lasso_path_selection` | Efron, Hastie, Johnstone & Tibshirani (2004), AoS 32(2) | LARS path entry order: features selected in the order they first enter the active set as the regularization parameter decreases from infinity. Distinct from `feature_selection(method="lasso")`, which uses LassoCV coefficient magnitude ranking. |
| `stability_selection` | Meinshausen & Bühlmann (2010), JRSS-B 72(4) | Subsample-based selection probability: lasso/elastic-net fitted on `n_subsamples` random half-samples; features retained where selection probability exceeds threshold `pi_thr`. |
| `genetic_algorithm_selection` | Goldberg (1989), Addison-Wesley | Binary-chromosome GA with tournament selection, single-point crossover, bit-flip mutation at rate 1/N, and elitism; fitness evaluated via CV neg-MSE. Pure NumPy implementation; no `deap` dependency. |

### L3 op count change

| Metric | Pre-C47 | Post-C47 |
|--------|---------|---------|
| L3 operational ops | >= 32 | >= 37 |
| L3 future ops | >= 6 | >= 3 (remaining: `chow_lin_disaggregation` L2-family, `lstm_hidden_state` L7-only, `generalized_irf` L7-only; `generalized_irf` promoted in C49) |
| L7 future ops count | >= 6 | >= 1 (`lstm_hidden_state` only after C49 promoted `generalized_irf`) |

### Implementation notes

All five helpers live in `macroforecast/core/runtime.py` immediately after
`_feature_selection`. Each is private (`_boruta_selection` etc.) and is
dispatched via `_execute_l3_op`. The validator no longer hard-rejects any of
the five op names in L3 recipes; all five return a `Panel` whose column set
is the selected subset of the input.

Random-state propagation follows the established #215 / #279 contracts: L0
`random_seed` is injected into `params["random_state"]` at materialize time,
and per-iteration seeds within stochastic ops (Boruta, stability, GA) are
derived as `(seed + iteration_index) % (2**31 - 1)`, guaranteeing bit-exact
replication across identical calls.

The `boruta` and `deap` packages are optional accelerators and are NOT
required. All default code paths use pure NumPy and scikit-learn primitives.
`scipy.stats.binom` (already a transitive dependency) is used in
`_boruta_selection` for the Bonferroni-corrected binomial test.

Four of the five ops (`boruta_selection`, `recursive_feature_elimination`,
`lasso_path_selection`, `stability_selection`) were previously listed in L7's
`FUTURE_OPS` tuple, which would have extended their `layer_scope` to include
`"l7"` via the tail registration loop. This extension was unintended. Cycle 47
removed these four names from `FUTURE_OPS` so all five ops remain
`layer_scope=("l3",)` exclusively.
