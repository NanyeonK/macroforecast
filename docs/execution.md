# Execution pipeline

## Purpose

The execution layer consumes compiled recipe contracts and emits deterministic run artifacts.

## Current operational models (24)

### Benchmark / naive
`ar` (AR with BIC lag selection)

### Linear ML / regularized
`ols`, `ridge`, `lasso`, `elasticnet`, `bayesianridge`, `huber`, `adaptivelasso`, `quantile_linear`

### Kernel / margin
`svr_linear`, `svr_rbf`

### Linear boosting
`componentwise_boosting` (L2Boost), `boosting_ridge`, `boosting_lasso`

### Factor-based linear
`pcr` (Principal Component Regression), `pls` (Partial Least Squares), `factor_augmented_linear` (Diffusion Index)

### Tree / ensemble
`randomforest`, `extratrees`, `gbm`, `xgboost`, `lightgbm`, `catboost`

### Neural
`mlp` (sklearn MLPRegressor)

## Current operational frameworks
- `expanding`
- `rolling`
- `anchored_rolling`

## Current operational statistical tests (20)

| Category | Tests |
|----------|-------|
| Equal predictive ability | `dm`, `dm_hln`, `dm_modified` |
| Nested model tests | `cw`, `enc_new`, `mse_f`, `mse_t` |
| Conditional / instability | `cpa`, `rossi`, `rolling_dm` |
| Multiple model comparison | `mcs`, `reality_check`, `spa` |
| Residual / calibration | `mincer_zarnowitz`, `ljung_box`, `arch_lm`, `bias_test`, `diagnostics_full` |
| Direction / classification | `pesaran_timmermann`, `binomial_hit` |

### Dependence corrections
`none`, `nw_hac`, `nw_hac_auto`, `block_bootstrap`

## Current operational importance methods (12)

| Category | Methods |
|----------|---------|
| SHAP family | `tree_shap`, `kernel_shap`, `linear_shap` |
| Model-agnostic | `permutation_importance`, `feature_ablation` |
| Local surrogate | `lime` |
| Partial dependence | `pdp`, `ice`, `ale` |
| Grouped | `grouped_permutation` |
| Stability | `importance_stability` |
| Legacy | `minimal_importance` |

## Run artifacts

Every execution produces:
- `predictions.csv` — full OOS prediction table
- `metrics.json` — per-horizon metrics (MSFE, RMSE, MAE, MAPE, relative metrics, OOS R2, etc.)
- `comparison_summary.json` — model vs benchmark summary
- `manifest.json` — full provenance including tree_context, preprocessing_contract, all specs
- `tuning_result.json` — HP tuning result (best_hp, trials, score)
- `summary.txt` — human-readable run summary
- `data_preview.csv` — raw data sample
- `stat_test_{name}.json` — statistical test artifact (when stat_test != none)
- `importance_{name}.json` — importance artifact (when importance_method != none)

## Provenance

The manifest preserves: preprocessing_contract, tree_context (fixed/sweep/conditional axes), model_spec, benchmark_spec, stat_test_spec, importance_spec, reproducibility_spec, failure_policy_spec, compute_mode_spec, tuning_result.
