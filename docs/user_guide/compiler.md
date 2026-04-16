# Compiler guide

## Purpose

The compiler bridges recipe YAML, canonical grammar, and runtime execution eligibility.
It validates axis selections, builds preprocessing contracts, and determines whether a recipe is executable.

## Current executable slice

### Models (24)
ar, ols, ridge, lasso, elasticnet, bayesianridge, huber, adaptivelasso, svr_linear, svr_rbf, componentwise_boosting, boosting_ridge, boosting_lasso, pcr, pls, factor_augmented_linear, quantile_linear, randomforest, extratrees, gbm, xgboost, lightgbm, catboost, mlp

### Feature builders (5)
autoreg_lagged_target, raw_feature_panel, raw_X_only, factors_plus_AR, factor_pca

### Benchmarks (4)
historical_mean, zero_change, ar_bic, custom_benchmark

### Frameworks
expanding, rolling, anchored_rolling

### Statistical tests (20)
dm, dm_hln, dm_modified, cw, mcs, enc_new, mse_f, mse_t, cpa, rossi, rolling_dm, reality_check, spa, mincer_zarnowitz, ljung_box, arch_lm, bias_test, pesaran_timmermann, binomial_hit, diagnostics_full

### Importance methods (12)
minimal_importance, tree_shap, kernel_shap, linear_shap, permutation_importance, lime, feature_ablation, pdp, ice, ale, grouped_permutation, importance_stability

### Tuning algorithms (4)
grid_search, random_search, bayesian_optimization, genetic_algorithm

### Export formats (5)
json, csv, parquet, json+csv, all

## Incompatibility rules

- `model_family='ar'` with `feature_builder='raw_feature_panel'` is blocked
- `predictor_family='all_macro_vars'` requires `feature_builder='raw_feature_panel'`
- `quantile_linear` requires `forecast_object='point_median'`
- Preprocessing governance cross-validation enforced (representation_policy vs tcode_application_scope consistency)
