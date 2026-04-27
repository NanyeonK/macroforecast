# API Reference

Auto-generated reference for every public package surface. Links point to the
corresponding user-guide chapter for conceptual background.

## Package surfaces

| Module | User-guide companion |
|--------|---------------------|
| [`macrocast.design`](design.md) | [Design (Stage 0): Study Grammar](../user_guide/design.md) |
| [`macrocast.raw`](raw.md) | [Raw Data](../user_guide/raw.md) |
| [`macrocast.recipes`](recipes.md) | [Recipes](../user_guide/recipes.md) |
| [`macrocast.preprocessing`](preprocessing.md) | [Preprocessing](../user_guide/preprocessing.md) |
| [`macrocast.registry`](registry.md) | [Registry](../user_guide/registry.md) |
| [`macrocast.compiler`](compiler.md) | [Compiler](../user_guide/compiler.md) |
| [`macrocast.execution`](execution.md) | [Execution](../user_guide/execution.md) |
| [`macrocast.tuning`](../user_guide/tuning.md) | [Tuning](../user_guide/tuning.md) |
| [`macrocast.start`](start.md) | [Getting Started: Quickstart](../getting_started/quickstart.md) |

## Current operational subset summary

- **Models (27)**: ar, ols, ridge, lasso, elasticnet, bayesian_ridge, huber, adaptive_lasso, svr_linear, svr_rbf, componentwise_boosting, boosting_ridge, boosting_lasso, pcr, pls, factor_augmented_linear, quantile_linear, random_forest, extra_trees, gradient_boosting, xgboost, lightgbm, catboost, mlp, lstm, gru, tcn
- **Feature builders (5)**: target_lag_features, raw_feature_panel, raw_predictors_only, factors_plus_target_lags, pca_factor_features
- **Benchmarks (4)**: historical_mean, zero_change, autoregressive_bic, custom_benchmark
- **Statistical tests (20)**: dm, dm_hln, dm_modified, cw, mcs, enc_new, mse_f, mse_t, cpa, rossi, rolling_dm, reality_check, spa, mincer_zarnowitz, ljung_box, arch_lm, bias_test, pesaran_timmermann, binomial_hit, full_residual_diagnostics
- **Importance methods (12)**: minimal_importance, tree_shap, kernel_shap, linear_shap, permutation_importance, lime, feature_ablation, pdp, ice, ale, grouped_permutation, importance_stability
- **Tuning algorithms (4)**: grid_search, random_search, bayesian_optimization, genetic_algorithm
- **Export formats (5)**: json, csv, parquet, json_csv, all

```{toctree}
:hidden:
:maxdepth: 1

design
raw
recipes
preprocessing
registry
compiler
execution
start
sweep_runner
models/deep
decomposition
```
