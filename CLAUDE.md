# macrocast

> Fair, reproducible macro forecasting benchmarking package.

## Quick start

```bash
python3 -m pytest tests/ -x -q        # run tests (~195 tests, ~3 min)
python3 -c "import macrocast"          # verify package import
```

## Architecture (8 layers, canonical order)

| Layer | Module | Purpose |
|-------|--------|---------|
| Stage 0 | `macrocast.stage0` | Study grammar: fixed/varying design, comparison contract, execution posture |
| Stage 1 | `macrocast.raw` | FRED-MD/QD/SD raw data loading, vintage management, provenance |
| Stage 2 | `macrocast.recipes` | RecipeSpec / RunSpec declarative study definition |
| Stage 3 | `macrocast.preprocessing` | Preprocessing contract: tcode/missing/outlier/scaling governance |
| Stage 4 | `macrocast.registry` | Per-axis choice-space registry (operational vs registry_only) |
| Stage 5 | `macrocast.compiler` | Recipe YAML -> axis validation -> execution eligibility |
| Stage 6 | `macrocast.execution` | Runtime: model/benchmark executors, predictions, metrics, artifacts |
| Stage 7 | `macrocast.tuning` | HP tuning engine: grid/random/bayesian/genetic search |

## Current operational slice

- **Models (24):** ar, ols, ridge, lasso, elasticnet, bayesian_ridge, huber, adaptive_lasso, svr_linear, svr_rbf, componentwise_boosting, boosting_ridge, boosting_lasso, pcr, pls, factor_augmented_linear, quantile_linear, random_forest, extra_trees, gradient_boosting, xgboost, lightgbm, catboost, mlp
- **Stat tests (20):** dm, dm_hln, dm_modified, cw, mcs, enc_new, mse_f, mse_t, cpa, rossi, rolling_dm, reality_check, spa, mincer_zarnowitz, ljung_box, arch_lm, bias_test, pesaran_timmermann, binomial_hit, full_residual_diagnostics
- **Importance (12):** minimal_importance, tree_shap, kernel_shap, linear_shap, permutation_importance, lime, feature_ablation, pdp, ice, ale, grouped_permutation, importance_stability
- **Benchmarks (4):** historical_mean, zero_change, autoregressive_bic, custom_benchmark
- **Frameworks:** expanding, rolling, anchored_rolling
- **Feature builders (5):** target_lag_features, raw_feature_panel, raw_predictors_only, factors_plus_target_lags, pca_factor_features
- **Tuning (4):** grid_search, random_search, bayesian_optimization, genetic_algorithm
- **Export:** json, csv, parquet, json_csv, all

## Registry (125 axes, 717 values, 310 operational)

Status levels: `operational` > `planned` > `registry_only` > `future`

## Key design principles

- **Grammar first:** Stage 0 fixes study language before registries are filled
- **One path = one study:** each recipe defines a complete, fully specified forecasting study
- **Represent before execute:** registry can represent more choices than runtime can execute
- **Fair comparison:** preprocessing is governed to prevent hidden design drift

## Module layout

```
macrocast/
  stage0/          # types, build, derive, normalize, validate, serialize, errors
  raw/             # datasets/ (fred_md, fred_qd, fred_sd), manager, cache, manifest
  recipes/         # types, build, construct, validate
  preprocessing/   # types, build (governance checks)
  registry/        # base, build (auto-loader), per-stage dirs (stage0/, data/, preprocessing/, training/, evaluation/, output/, tests/, importance/)
  compiler/        # types, build (recipe->execution bridge)
  execution/       # types, build (main runtime), deep_training (models+tuning integration)
  tuning/          # types, engine, budget, hp_spaces, search/ (grid, random, bayesian, genetic), validation/ (splitter, scorer)
  start.py         # wizard/preview entry point
tests/             # ~195 tests
docs/              # public documentation
plans/             # internal planning (not public)
```
