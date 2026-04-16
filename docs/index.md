# macrocast

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

## Package surfaces

The package has eight documented layers:

- [`macrocast.stage0`](stage0.md) — study grammar and comparison contract
- [`macrocast.raw`](raw.md) — raw dataset loading and provenance
- [`macrocast.recipes`](recipes.md) — recipe and run specification
- [`macrocast.preprocessing`](preprocessing.md) — preprocessing contract and governance
- [`macrocast.registry`](registry.md) — per-axis choice-space registry
- [`macrocast.compiler`](compiler.md) — recipe compilation and execution eligibility
- [`macrocast.execution`](execution.md) — runtime execution pipeline
- [`macrocast.tuning`](tuning.md) — hyperparameter tuning engine

## Current operational subset

### Models (24)
`ar`, `ols`, `ridge`, `lasso`, `elasticnet`, `bayesianridge`, `huber`, `adaptivelasso`, `svr_linear`, `svr_rbf`, `componentwise_boosting`, `boosting_ridge`, `boosting_lasso`, `pcr`, `pls`, `factor_augmented_linear`, `quantile_linear`, `randomforest`, `extratrees`, `gbm`, `xgboost`, `lightgbm`, `catboost`, `mlp`

### Feature builders (5)
`autoreg_lagged_target`, `raw_feature_panel`, `raw_X_only`, `factors_plus_AR`, `factor_pca`

### Benchmarks (4)
`historical_mean`, `zero_change`, `ar_bic`, `custom_benchmark`

### Frameworks
`expanding`, `rolling`, `anchored_rolling`

### Preprocessing
- Governance fields: `representation_policy`, `tcode_application_scope`, `preprocessing_axis_role`
- Operational paths: `raw_only`, train-only EM impute + standard/robust/minmax scaling, PCA dimensionality reduction

### Evaluation metrics
`msfe`, `rmse`, `mae`, `mape`, `relative_msfe`, `relative_rmse`, `relative_mae`, `oos_r2`, `csfe`, `benchmark_win_rate`, `directional_accuracy`, `sign_accuracy`

### Statistical tests (20)
`dm`, `dm_hln`, `dm_modified`, `cw`, `mcs`, `enc_new`, `mse_f`, `mse_t`, `cpa`, `rossi`, `rolling_dm`, `reality_check`, `spa`, `mincer_zarnowitz`, `ljung_box`, `arch_lm`, `bias_test`, `pesaran_timmermann`, `binomial_hit`, `diagnostics_full`

### Dependence corrections (4)
`none`, `nw_hac`, `nw_hac_auto`, `block_bootstrap`

### Importance methods (12)
`minimal_importance`, `tree_shap`, `kernel_shap`, `linear_shap`, `permutation_importance`, `lime`, `feature_ablation`, `pdp`, `ice`, `ale`, `grouped_permutation`, `importance_stability`

### Tuning (4 search algorithms)
`grid_search`, `random_search`, `bayesian_optimization`, `genetic_algorithm`

### Output
- Export formats: `json`, `csv`, `parquet`, `json+csv`, `all`
- Provenance: `none`, `minimal`, `standard`, `full`
- Artifacts: predictions, metrics, comparison_summary, stat_test, importance, tuning_result, manifest

## Registry

125 axes, 717 values, 310 operational across 8 layers.
