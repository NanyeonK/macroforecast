# Macrocast Canonical Status — 2026-04-16

Status: canonical current-state reference
Supersedes: plan_04_14_1958.md phase status, implementation-issues.md issue status

---

## Package scale

| Metric | Value |
|--------|-------|
| Source code | ~12,000 LOC |
| Test code | ~4,300 LOC |
| Tests | 291 pass (0 fail) |
| Registry axes | 125 |
| Registry values | 717 |
| Operational values | 310 (43.2%) |

## Per-layer status

| Layer | Axes | Values | Operational | Status |
|-------|:----:|:------:|:-----------:|--------|
| 0_meta | 7 | 49 | 19 | **Complete** — all planned axes registered |
| 1_data_task | 29 | 191 | 43 | **Complete** — all planned axes registered |
| 2_preprocessing | 24 | 124 | 44 | **Complete** — governance fields operational |
| 3_training | 28 | 152 | 88 | **Complete** — 24 models, 4 search algorithms, full tuning engine |
| 4_evaluation | 18 | 111 | 34 | **Complete** — 7 primary metrics, 6 relative metrics, regime support |
| 5_output_provenance | 4 | 19 | 14 | **Complete** — 5 export formats, full provenance |
| 6_stat_tests | 2 | 25 | 25 | **Complete** — 20 stat tests + 4 dependence corrections, all operational |
| 7_importance | 13 | 46 | 43 | **Complete** — 12 methods including SHAP/LIME/PDP/ICE/ALE |

## Operational components (verified by 96 parametrized sweep tests)

### Models (24/24 operational)
ar, ols, ridge, lasso, elasticnet, bayesianridge, huber, adaptivelasso, svr_linear, svr_rbf, componentwise_boosting, boosting_ridge, boosting_lasso, pcr, pls, factor_augmented_linear, quantile_linear, randomforest, extratrees, gbm, xgboost, lightgbm, catboost, mlp

### Statistical tests (20 operational + diagnostics_full)
dm, dm_hln, dm_modified, cw, mcs, enc_new, mse_f, mse_t, cpa, rossi, rolling_dm, reality_check, spa, mincer_zarnowitz, ljung_box, arch_lm, bias_test, pesaran_timmermann, binomial_hit, diagnostics_full

### Importance methods (12 operational)
minimal_importance, tree_shap, kernel_shap, linear_shap, permutation_importance, lime, feature_ablation, pdp, ice, ale, grouped_permutation, importance_stability

### Tuning algorithms (4 operational)
grid_search, random_search, bayesian_optimization, genetic_algorithm

### Export formats (5 operational)
json, csv, parquet, json+csv, all

## Completed phases (from original roadmap)

| Phase | Original scope | Status |
|-------|---------------|--------|
| A. Feature-builder widening | autoreg + raw_panel + factor paths | **Done** — 5 feature builders |
| B. Preprocessing widening | train-only, scaling, governance | **Done** — 24 preprocessing axes |
| C. Framework widening | expanding + rolling | **Done** — + anchored_rolling |
| D. Benchmark widening | custom benchmark bridge | **Done** — 4 benchmarks |
| E. Evaluation + stat tests | DM, CW, metrics | **Done** — 20 tests, 12+ metrics |
| F. Importance layer | minimal + SHAP | **Done** — 12 methods |
| G. Data/task widening | real-time, multi-target | **Done** — slices operational |

## Completed epic/issues (from implementation-issues.md)

| Epic | Description | Status |
|------|-------------|--------|
| 0 | Registry architecture refactor | **Done** |
| 1 | Stage 0 grammar lock (7 axes) | **Done** |
| 2 | Stage 1 data/task (29 axes) | **Done** |
| 3 | Stage 2 preprocessing (24 axes) | **Done** |
| 4 | Stage 3 training expansion (28 axes, 24 models, tuning) | **Done** |
| 5 | Stage 4 evaluation (18 axes) | **Done** |
| 6 | Stage 5 output/provenance (4 axes) | **Done** |
| 7 | Stage 6 stat tests (2 axes, 21 tests) | **Done** |
| 8 | Stage 7 importance (13 axes, 12 methods) | **Done** |

## Audit fixes applied (2026-04-16)

1. **Tuning payload capture** — all 48 model executors now propagate tuning_payload to tuning_result.json artifact and manifest
2. **Documentation overhaul** — 9 docs files updated/created to match actual 24 models / 20 tests / 12 importance
3. **CLAUDE.md created** — project guide with architecture, module map, operational slice
4. **Full operational sweep test** — 96 parametrized tests verifying every operational registry value executes end-to-end

## Known remaining gaps

### Not yet operational (registry_only / planned / future)
- Density/interval evaluation metrics (quantile infrastructure needed)
- Economic decision metrics
- Regime-specialized training
- Some bootstrap variants (stationary, circular, wild, cluster)
- Some preprocessing variants (broader tcode paths, rolling/expanding fit scope)
- Sequence/neural framework (LSTM, Transformer, etc.)
- Panel/hierarchical/state-space models
- GPU/distributed compute
- Advanced gradient interpretability (IG, DeepLift, LRP)
- Some data adapters (OECD, IMF, ECB, BIS, news_text, satellite)

### Documentation gaps remaining
- docs/raw.md, docs/recipes.md, docs/stage0.md — still at earlier state (functional but less detailed)
- Tuning result artifact has empty best_hp when tuning not explicitly enabled in recipe
