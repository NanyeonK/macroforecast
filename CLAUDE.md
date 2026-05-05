# macroforecast

> Fair, reproducible macro forecasting benchmarking package.
> Current version is in ``pyproject.toml`` and ``macroforecast/__init__.py``;
> 12-layer canonical design lives in ``plans/design/part1-4``.

## Quick start

```bash
python3 -m pytest tests/ -x -q                     # ~953 tests, <30s on a laptop
python3 -c "import macroforecast; print(macroforecast.__version__)"
python3 -c "import macroforecast; macroforecast.run('examples/recipes/l4_minimal_ridge.yaml')"
```

## Public API

```python
import macroforecast

result = macroforecast.run("path/to/recipe.yaml", output_directory="out/")
# result.cells -> per-sweep-cell RuntimeResult + sink_hashes
# manifest.json + per-cell artifacts written to out/

replication = macroforecast.replicate("out/manifest.json")
# bit-exact: replication.sink_hashes_match  -> True when artifacts identical
```

## Architecture (12-layer design, canonical order)

| Layer | Purpose | Module |
|-------|---------|--------|
| L0 | Study setup (failure_policy, seed, compute_mode) | `core/layers/l0.py` |
| L1 | Data definition (FRED-MD/QD/SD, target, geography, regime) | `core/layers/l1.py` |
| L2 | Preprocessing (transform / outlier / imputation / frame edge) | `core/layers/l2.py` |
| L3 | Feature engineering DAG (37 ops + cascade β) | `core/layers/l3.py`, `core/ops/l3_ops.py` |
| L4 | Forecasting model + tuning (30+ families, 5 combine ops) | `core/layers/l4.py`, `core/ops/l4_ops.py` |
| L5 | Evaluation (metrics × benchmark × aggregation × decomposition × ranking) | `core/layers/l5.py`, `core/ops/l5_ops.py` |
| L6 | Statistical tests (DM/HLN, CW, MCS bootstrap, PT/HM, residual battery) | `core/layers/l6.py`, `core/ops/l6_ops.py` |
| L7 | Interpretation (29 importance ops, group_aggregate, lineage, US choropleth) | `core/layers/l7.py`, `core/ops/l7_ops.py` |
| L8 | Output / provenance (json/csv/parquet/latex/markdown, manifest) | `core/layers/l8.py`, `core/ops/l8_ops.py` |
| L1.5 / L2.5 / L3.5 / L4.5 | Diagnostic hooks (default-off, non-blocking) | `core/layers/l{1,2,3,4}_5.py` |

Foundation contracts (5 node types, sweep machinery, cache/hash, manifest schema)
live in `core/{dag, sweep, cache, manifest, types, validator, yaml}.py`. The
end-to-end runtime (cell loop, seed propagation, bit-exact replicate) is
`core/execution.py`; per-layer materialization helpers are in `core/runtime.py`.

## Operational coverage (v0.1)

- **Models (30+):** ar_p, ols, ridge, lasso, elastic_net, lasso_path,
  bayesian_ridge, huber, glmboost, var, factor_augmented_ar,
  principal_component_regression, decision_tree, random_forest, extra_trees,
  gradient_boosting, xgboost / lightgbm / catboost (try-import),
  svr_linear/rbf/poly, mlp, lstm/gru/transformer (torch try-import + sklearn
  fallback), knn, macroeconomic_random_forest, bvar_minnesota,
  bvar_normal_inverse_wishart, dfm_mixed_mariano_murasawa.
- **L3 ops (37):** lag/seasonal_lag, ma_window/ma_increasing_order (MARX),
  pca/sparse_pca/scaled_pca/dfm/varimax/partial_least_squares/random_projection,
  wavelet/fourier, hp_filter/hamilton_filter, polynomial/interaction/kernel/
  nystroem, regime_indicator/season_dummy/time_trend/holiday,
  feature_selection (variance/correlation/lasso), cumulative_average target,
  hierarchical_pca, weighted_concat, simple_average.
- **L6 tests:** Diebold-Mariano with Newey-West HAC + HLN correction,
  Clark-West with MSE adjustment, MCS / SPA / Reality Check / StepM via
  bootstrap, Pesaran-Timmermann + Henriksson-Merton, statsmodels Ljung-Box /
  ARCH-LM / Jarque-Bera / Breusch-Godfrey / Durbin-Watson.
- **L7 importance:** SHAP family (shap try-import → linear/permutation
  proxy), partial_dependence, accumulated_local_effect, friedman_h_interaction,
  lasso_inclusion_frequency, cumulative_r2_contribution, bootstrap_jackknife,
  rolling_recompute, forecast_decomposition, fevd / historical_decomposition /
  generalized_irf, group_aggregate, lineage_attribution,
  transformation_attribution, mrf_gtvp.
- **L7 figures:** matplotlib bar_global, heatmap, pdp_line, plus a stylized
  US state choropleth (`render_us_state_choropleth`) for FRED-SD geographic
  visualization. Default mappings auto-render per importance op.
- **L8 export:** json / csv / parquet / latex / markdown / json_csv /
  json_parquet / all + per-cell granularity + descriptive naming.
- **Sweep:** param-level + external-axis sweeps; grid (default) and zip
  combination modes; cell loop iterates `{sweep: [...]}` markers in the
  recipe root.
- **Replication:** `macroforecast.replicate(manifest_path)` re-runs the stored
  recipe and verifies per-cell sink hashes match bit-for-bit.

## Module layout

```
macroforecast/
  __init__.py             # lazy-export top-level surface
  api.py                  # macroforecast.run / macroforecast.replicate
  core/
    execution.py          # execute_recipe (cell loop) + replicate_recipe
    runtime.py            # per-layer materialize_l{1..8}_minimal helpers
    figures.py            # matplotlib backend + US state choropleth
    cache.py, dag.py, sweep.py, manifest.py, validator.py, yaml.py, types.py
    layer_specs.py, recipe.py, selectors.py
    layers/               # l0..l8 + l1_5/l2_5/l3_5/l4_5 schema definitions
    ops/                  # universal/l3/l4/l5/l6/l7/l8/diagnostic op registry
  raw/                    # FRED-MD/QD/SD adapters, vintage manager, manifest
  preprocessing/          # preprocessing contract helpers (legacy support)
  custom.py               # user-defined model / preprocessor registration
  defaults.py             # default profile dict template
  tuning/                 # HP search engines (optional, integrated via L4)
plans/design/             # 4-part design document (canonical source of truth)
tests/                    # 474 tests (core, layers, integration)
examples/recipes/         # YAML recipe examples per layer
```

## Key design principles

- **Schema before runtime:** every layer declares an axis × option × gate
  contract in `LayerImplementationSpec`; runtime helpers in `core/runtime.py`
  materialize that contract.
- **One recipe = one study:** a YAML recipe fully specifies the DAG; sweep
  markers expand into independent cells.
- **Bit-exact replication:** seed propagation + canonical key ordering +
  per-cell sink hashes guarantee that `replicate(manifest_path)` produces
  identical artifacts.
- **Default-off diagnostics:** L1.5/L2.5/L3.5/L4.5 + L6/L7 require explicit
  `enabled: true` so a minimal recipe stays fast.
- **Cross-layer references:** `is_benchmark` (L4 → L5/L6), `mcs_inclusion`
  (L6.D → L7), `lineage` (L3.metadata → L7), `regime` (L1.G → L3/L4/L5/L6.C).

## Status levels

The package uses a **two-value vocabulary** (defined in
`macroforecast.core.status`):

- **`operational`** -- runtime executes the full design-spec procedure.
  The output matches the published method named in the design.
- **`future`** -- schema-only. The validator hard-rejects use at recipe
  time; the runtime raises `NotImplementedError`. Tracked for v0.2+
  implementation in the GitHub issue tracker.

Earlier releases experimented with intermediate values (`planned`,
`approximation`, `simplified`, `registry_only`); these are kept as
deprecated aliases (`normalize_status` collapses every legacy alias to
`future`) but new code should write `operational` or `future` only. The
honesty pass that introduced this 2-value vocabulary is documented in
PR-A (#177) through PR-G of the v0.1.1 series.

Helpers live on `macroforecast.core` for typed comparison:

```python
from macroforecast.core import OPERATIONAL, FUTURE, ItemStatus, is_runnable, is_future, normalize_status
from macroforecast.core.ops.l4_ops import get_family_status

assert get_family_status("ridge") == OPERATIONAL
assert get_family_status("macroeconomic_random_forest") == FUTURE
assert is_runnable("planned")  # legacy alias collapses to operational? no:
assert not is_runnable("planned")  # legacy `planned` -> future, not runnable
```

### v0.1 honesty-pass demotions (all closed in v0.2)

The codex review on PR #163 flagged 19 families / ops whose v0.1
runtime did not match the published procedure named in the design. They
were demoted from `operational` to `future`; **every demotion was
re-promoted in v0.2** with a real implementation. The 35 issues tracked
in milestones `v0.2 honesty-pass` and `v0.2 design coverage` are now
all closed.

| Layer | Item | v0.2 implementation | Issue |
|------|------|---------------------|-------|
| L1.G | `estimated_markov_switching` | statsmodels `MarkovRegression` (Hamilton 1989) | #195 |
| L1.G | `estimated_threshold` | Tong (1990) SETAR quantile-split estimator | #196 |
| L1.G | `estimated_structural_break` | Bai-Perron (1998) global LSE greedy break detection | #197 |
| L4 | `factor_augmented_var` | Bernanke-Boivin-Eliasz (2005) FAVAR (PCA factors + VAR) | #184 |
| L4 | `bvar_minnesota` / `bvar_normal_inverse_wishart` | closed-form Litterman (1986) Minnesota / NIW posterior mean | #185, #186 |
| L4 | `macroeconomic_random_forest` | Coulombe (2024) GTVP -- per-leaf local linear regressions | #187 |
| L4 | `dfm_mixed_mariano_murasawa` | statsmodels `DynamicFactor` (Kalman state-space MLE) | #188 |
| L7 | `fevd` / `historical_decomposition` / `generalized_irf` | statsmodels VAR `fevd` / `irf` builders | #189 |
| L7 | `mrf_gtvp` | per-leaf coefficient series from `_MRFWrapper` | #190 |
| L7 | `lasso_inclusion_frequency` | bootstrap inclusion frequency in `[0, 1]` | #191 |
| L7 | `accumulated_local_effect` | Apley & Zhu (2020) centred-cumulative-effect | #192 |
| L7 | `friedman_h_interaction` | Friedman & Popescu (2008) H² statistic | #193 |
| L7 | `gradient_shap` / `integrated_gradients` / `saliency_map` / `deep_lift` | captum-backed gradient attributions (operational with `[deep]` extra) | #194 |
| L4 (deep NN) | `lstm` / `gru` / `transformer` without torch | `NotImplementedError("install macroforecast[deep]")` -- explicit, no silent MLP fallback | #198 |

### v0.2 design-coverage additions

Beyond the honesty-pass promotions, v0.2 added the following capabilities
on top of v0.1:

| Capability | Issue |
|------------|-------|
| L0 `random_seed` -> L4 estimator `random_state` automatic propagation | #215 |
| L8 manifest carries every design-listed provenance field (14 fields) | #208 |
| L8 export `compression: gzip / zip` | #206 |
| L8 `artifact_granularity = per_target / per_horizon / per_target_horizon / flat` | #207 |
| L8 `manifest_format = yaml`, `export_format = html_report` | #209 |
| L1.5 ADF / Phillips-Perron / KPSS stationarity tests | #210 |
| `macroforecast.custom.register_model` callables dispatched in L4 runtime | #216 |
| L4 `search_algorithm = grid / random / bayesian / genetic / cv_path` dispatch | #217 |
| L6.C Giacomini-Rossi (2010) rolling-window fluctuation test | #199 |
| L6.E density / interval test battery (PIT-Berkowitz / KS / Kupiec / Christoffersen) | #200 |
| L4 `forecast_object = quantile / density` path | #201 |
| L2.A FRED-SD frequency alignment rules | #202 |
| `sweep_groups` (NodeGroupSweep) into `execute_recipe` | #203 |
| `parallel_unit = models` sub-cell parallelism | #204 |
| 14 additional L7 figure types (SHAP / ALE / lasso / decomp / lineage families) | #205 |
| L1.5 / L2.5 / L3.5 / L4.5 diagnostic visualisations + multi-format export | #211, #212, #213, #214 |
| Real Shapley-over-pipelines `transformation_attribution` | #218 |

### v0.25 second honesty pass (all closed)

After v0.2 the audit flagged 19 items that were operational but shipped
as minimum-viable proxies relative to the published design. v0.25
promotes every one of them to the procedure named in the literature.

| Item | v0.25 implementation | Issue |
|------|---------------------|-------|
| Phillips-Perron native | OLS + Newey-West HAC, no `arch` dependency | #252 |
| Lasso inclusion rolling-window mode | `sampling = bootstrap | rolling | both` | #253 |
| Sampling Shapley for n > 8 pipelines | Castro-Gomez-Tejada (2009) permutation Shapley | #254 |
| McCracken-Ng / FRED-SD canonical block memberships | 8-group MD + 14-group QD + 50-state grid | #260 |
| L8 derived `saved_objects` defaults | per-active-layer auto-include | #261 |
| Tong (1990) SETAR full grid-search | joint-SSR objective with AR(p) per regime | #243 |
| Bai (1997) DP exact break search | global LSE recursion with BIC | #244 |
| L3 cascade β + pipeline_id propagation | `cascade_max_depth` enforced; pipeline_id inherited | #257 |
| L6 HAC kernels (Newey-West / Andrews / Parzen) | `_long_run_variance(kernel)` helper | #259 |
| L5 decomposition / oos_period / aggregation matrix | per_subperiod / by_predictor_block / per_horizon_then_mean / top_k_worst | #258 |
| L2 preprocessor + L3 feature_block / combiner dispatch | end-to-end runtime hook | #251 |
| Mariano-Murasawa mixed-frequency Kalman | `DynamicFactorMQ` route when `mixed_frequency=True` | #245 |
| Per-family quantile estimators | QuantileRegressor / GBR-quantile / xgb / lgbm | #246 |
| Density tests strict mode | requires real `forecast_intervals` | #247 |
| Giacomini-Rossi simulated CVs | per (m/T, alpha) Monte Carlo | #248 |
| Real Chow-Lin (1971) disaggregation | regression-based with monthly indicator | #255 |
| L4.5 residual ACF + QQ views | `fit_view = residual_acf | residual_qq | multi` | #256 |
| 9 L7 figure renderers replaced (force / dependence / IRF / decomp / etc.) | distinct layouts matching the design table | #249 |
| Sub-cell parallelism (`oos_dates` / `horizons` / `targets`) | walk-forward origin loop fan-out | #250 |

### v0.3 third honesty pass + new features (all closed)

| Item | v0.3 implementation | Issue |
|------|---------------------|-------|
| PP MacKinnon (2010) p-value | finite-sample table interpolation | #273 |
| DFM-MQ idiosyncratic_ar1=True | Mariano-Murasawa Eq. (4) | #274 |
| L5 by_predictor_block refit-per-subset | true Shapley share via per-coalition OLS | #275 |
| L6.E Engle-Manganelli DQ test | hit regression + chi-square | #276 |
| `register_target_transformer` runtime dispatch | fit -> transform -> inverse at L5 boundary | #277 |
| L4.5 fitted_vs_actual + residual_time | sub-views per model | #278 |
| Sub-cell parallel deterministic seed | `base + position` per-origin RNG | #279 |
| **Quantile Regression Forest** (Meinshausen 2006) | leaf-conditional empirical CDF | #280 |
| **Strobl (2008) conditional PFI** | bin-restricted permutation | #281 |
| **Bagging meta-estimator** | bootstrap-aggregated wrapper + quantile bands | #282 |
| **Diebold-Mariano-Pesaran joint multi-horizon** | HAC-adjusted stacked-DM | #283 |
| README rewrite + recipe gallery + replication script + sphinx skeleton | release polish | #284-#287 |

### Status snapshot

* L4 operational families: **35+** (linear, tree, boosting, SVM, kNN,
  MLP, deep NN, AR_p, BVAR Minnesota / NIW, FAVAR, MRF GTVP,
  Mariano-Murasawa DFM, QRF, bagging).
* L7 operational ops: **~30** (no honesty-pass demotions remain).
* L1.G regimes: 6 (none / NBER / user / Hamilton MS / Tong SETAR / Bai-Perron).
* L6 tests: DM (HLN + HAC kernels), CW, GR (simulated CVs), MCS / SPA / RC / StepM,
  PT/HM, residual battery, density tests with DQ, DMP joint multi-horizon.
* Sub-cell parallelism: cells / models / oos_dates / horizons / targets.
