# Public Python API

[Back to reference](index.md)

The importable surface is module-based and pandas-first. The table below
mirrors `macroforecast.__all__`: every symbol listed there should be importable
from the top-level package. Module-level helpers that are not top-level
convenience exports are documented on their module reference pages.

## Top-Level Exports

| Symbol | Source | Description |
| --- | --- | --- |
| `meta`, `data`, `preprocessing`, `feature_engineering`, `filters`, `data_analysis`, `feature_analysis`, `feature_diagnostic`, `forecast_analysis`, `forecast_diagnostic`, `models`, `model_ensemble`, `model_selection`, `forecasting`, `metrics`, `tests`, `evaluation`, `window`, `interpretation`, `output`, `reporting` | package namespaces | Top-level module namespaces. `feature_diagnostic` and `forecast_diagnostic` are compatibility aliases for the corresponding analysis modules. |
| `configure`, `get_config`, `get_option`, `reset_config`, `use_config`, `DEFAULT_RANDOM_SEED`, `StageDefaultScope`, `MetadataLevel` | `macroforecast.meta` | Global package defaults and config types. |
| `DataBundle`, `DataSpec`, `RegimeDirection`, `SamePeriodPolicy`, `as_panel`, `attach_metadata`, `custom_dataset`, `metadata`, `panel_info`, `set_frequencies`, `spec`, `validate_panel` | `macroforecast.data` | Canonical panel and metadata helpers. |
| `align_frequency`, `chow_lin_disaggregate`, `infer_frequencies`, `frequency_hardening_issues`, `availability_lag`, `same_period_predictors`, `define_regime` | `macroforecast.data` | Frequency alignment/inference, Chow-Lin disaggregation, data availability, same-period predictor, and regime metadata policies. |
| `load_fred_md`, `load_fred_qd`, `load_fred_sd`, `load_fred_md_sd`, `load_fred_qd_sd` | `macroforecast.data` | Dataset loaders. |
| `load_custom_csv`, `load_custom_parquet`, `list_vintages`, `combine` | `macroforecast.data` | Custom loading, vintage discovery, and panel combination. |
| `reprocess`, `custom_preprocess`, `standardize_panel`, `PreprocessedData` | `macroforecast.preprocessing` | Direct pandas preprocessing. |
| `PreprocessSpec`, `FittedPreprocessor`, `preprocess_spec`, `custom_preprocess_step` | `macroforecast.preprocessing` | Runner-compatible preprocessing fit/transform specs. |
| `FeatureSet`, `FeatureSpec`, `FittedFeatureBuilder`, `build_features`, `feature_spec` | `macroforecast.feature_engineering` | Aligned forecast matrices, metadata, and runner-compatible feature specs. |
| `direct_target`, `average_target`, `forward_average_target`, `path_targets` | `macroforecast.feature_engineering` | Direct, forward-average, and path target construction. |
| `feature_matrix`, `compose_features` | `macroforecast.feature_engineering` | Paper-style feature blocks and sequential feature composition. |
| `lag`, `mixed_frequency_lags`, `rolling_mean`, `moving_average_ladder`, `scale_features`, `pca_features`, `sparse_pca_chen_rohe_features`, `varimax_features`, `group_pca`, `maf_features`, `time_features`, `custom_features` | `macroforecast.feature_engineering` | Core direct pandas feature transforms. |
| `transform_features`, `log_features`, `diff_features`, `log_diff_features`, `pct_change_features`, `cumsum_features`, `seasonal_lag`, `season_dummy`, `fourier_features`, `polynomial_features`, `interaction_features`, `hp_filter_features`, `hamilton_filter_features`, `savitzky_golay_features`, `wavelet_features`, `adaptive_ma_rf_features`, `asymmetric_trim_features`, `rank_space_features`, `moving_average_changes`, `align_reference_weights`, `weighted_aggregate`, `partial_least_squares_features`, `sliced_inverse_regression_features`, `dfm_features`, `variance_selection`, `correlation_selection`, `lasso_selection`, `lasso_path_selection`, `rfe_selection`, `boruta_selection`, `stability_selection`, `genetic_selection`, `select_features`, `feature_selection_requires_target`, `normalize_feature_selection_method`, `FeatureSelectionResult`, `random_projection_features`, `nystroem_features` | `macroforecast.feature_engineering` | Additional transform, seasonal, expansion, filter, supervised aggregation, supervised factor, feature-selection, and kernel-approximation feature functions. |
| `lag_step`, `rolling_step`, `moving_average_step`, `marx_step`, `transform_step`, `seasonal_lag_step`, `season_dummy_step`, `fourier_step`, `time_step`, `polynomial_step`, `interaction_step`, `scale_step`, `pca_step`, `sparse_pca_chen_rohe_step`, `varimax_step`, `group_pca_step`, `maf_step`, `hamilton_step`, `random_projection_step`, `nystroem_step`, `partial_least_squares_step`, `sliced_inverse_regression_step`, `custom_step` | `macroforecast.feature_engineering` | Reusable step dictionaries for `compose_features` and runner-safe `feature_spec` pipelines. Feature selection uses individual `method` names in step mappings instead of a generic step builder. |
| `pca_then_lags`, `lags_then_pca`, `moving_average_pca_lags` | `macroforecast.feature_engineering` | Convenience composed feature callables. |
| `ModelFit`, `VolatilityFit`, `SavedModel`, `save_fit`, `load_fit` | `macroforecast.models` | Fitted model result wrappers and low-level fit persistence. |
| `ols`, `ridge`, `nonneg_ridge`, `shrink_to_target_ridge`, `fused_difference_ridge`, `random_walk_ridge`, `tvp_ridge`, `lasso`, `elastic_net`, `adaptive_lasso`, `adaptive_elastic_net`, `group_lasso`, `sparse_group_lasso`, `bayesian_ridge`, `huber`, `kernel_ridge`, `knn`, `glmboost`, `pls`, `scaled_pca`, `supervised_pca`, `supervised_scaled_pca` | `macroforecast.models` | Linear, penalized, grouped, time-varying ridge, kernel, nearest-neighbor, and supervised dimension-reduction models. |
| `supervised_aggregation`, `component_aggregation`, `rank_aggregation`, `assemblage_regression`, `albacore_components`, `albacore_ranks` | `macroforecast.models` | Generic supervised aggregation and Albacore/assemblage wrappers. |
| `solve_nonnegative_ridge`, `solve_simplex_ridge`, `solve_target_shrinkage_ridge`, `solve_mean_aligned_ridge`, `solve_fused_difference_ridge` | `macroforecast.models` | Low-level constrained aggregation solver helpers returning weight vectors. |
| `svr`, `linear_svr`, `nu_svr` | `macroforecast.models` | Support-vector regression models. |
| `nn`, `lstm`, `gru`, `transformer`, `hemisphere_nn`, `density_hnn` | `macroforecast.models` | Torch-backed neural-network and density-forecast regressors; require `macroforecast[deep]`. |
| `ar`, `arima`, `auto_arima`, `var`, `bvar_minnesota`, `bvar_normal_inverse_wishart`, `ets`, `holt_winters`, `theta_method`, `far`, `favar` | `macroforecast.models` | Time-series, Bayesian VAR, exponential-smoothing, and factor-augmented forecasting models. |
| `dfm_mixed_mariano_murasawa`, `dfm_unrestricted_midas`, `midas_almon`, `midas_beta`, `midas_step`, `restricted_midas`, `unrestricted_midas` | `macroforecast.models` | Mixed-frequency dynamic-factor and MIDAS models. |
| `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `mars`, `xgboost`, `lightgbm`, `lgb_plus`, `lgba_plus`, `catboost` | `macroforecast.models` | Tree, spline, ML, and LGB+ hybrid regressors. |
| `quantile_regression_forest`, `macro_random_forest` | `macroforecast.models` | Macro-specific tree models. |
| `LGBPlusRegressor`, `LGBAPlusRegressor` | `macroforecast.models` | LGB+ competition and LGB^A+ alternating estimator classes. |
| `garch11`, `egarch`, `gjr_garch`, `tgarch`, `realized_garch` | `macroforecast.models` | Volatility models. |
| `ModelSpec`, `ModelParameter`, `custom_model`, `get_model`, `list_model_specs`, `describe_model`, `model_search_space` | `macroforecast.models` | Model-owned defaults and hyperparameter spaces. |
| `BaggingRegressor`, `BoogingRegressor`, `RandomSubspaceRegressor`, `StackingRegressor`, `SuperLearnerRegressor`, `MODEL_ENSEMBLE_BASE_ESTIMATORS`, `MODEL_ENSEMBLE_SPECS`, `bagging`, `subagging`, `random_subspace`, `stacking`, `super_learner`, `booging`, `custom_model_ensemble`, `get_model_ensemble`, `list_model_ensemble_bases`, `list_model_ensemble_specs`, `describe_model_ensemble`, `model_ensemble_search_space` | `macroforecast.model_ensemble` | Fit-time model-composition callables, estimator classes, and specs. |
| `WindowSpec`, `EstimationWindow`, `ValWindow`, `TestWindow`, `AlignmentWindow`, `StagePolicy`, `Split` | `macroforecast.window` | Forecast experiment and stage timing objects. |
| `from_cutoffs`, `estimation_expanding`, `estimation_rolling`, `estimation_fixed`, `val_last_block`, `val_poos`, `val_expanding`, `val_rolling_blocks`, `val_blocked_kfold`, `val_random_kfold`, `test_origins`, `alignment_drop_incomplete`, `alignment_keep_missing` | `macroforecast.window` | Component window builders. |
| `last_block`, `poos`, `expanding`, `rolling_blocks`, `blocked_kfold`, `random_kfold` | `macroforecast.window` | Shortcut window specs. |
| `stage_policy`, `custom_stage_policy`, `stage_index`, `stage_panel`, `last_block_split`, `poos_split`, `expanding_split`, `rolling_blocks_split`, `blocked_kfold_split`, `random_kfold_split`, `make_splitter`, `resolve_window`, `resolve_stage_policy`, `split_table`, `normalize_window_name` | `macroforecast.window` | Stage timing, resolver helpers, and train/val split inspection. |
| `metrics` | `macroforecast.metrics` | Forecast scoring namespace, including point scores and risk-adjusted forecast-return metrics. Use `mf.metrics.rmse`, not `mf.rmse`. |
| `tests` | `macroforecast.tests` | Forecast-comparison test namespace, including `mf.tests.custom_test`, `mf.tests.equal_predictive_tests`, `mf.tests.model_confidence_set`, `mf.tests.blocked_oob_reality_check`, `mf.tests.superior_predictive_ability_test`, `mf.tests.reality_check_test`, `mf.tests.stepm_test`, interval coverage, and PIT diagnostics. Use `mf.tests.dm_test`, not `mf.dm_test`. |
| `EvaluationReport`, `evaluate_report`, `aggregate_scores`, `filter_oos_period`, `error_decomposition`, `benchmark_comparison`, `regime_scores` | `macroforecast.evaluation` | Multi-slice evaluation reports, OOS filtering, error decomposition, benchmark comparisons, and regime scoring. |
| `evaluation` | `macroforecast.evaluation` | Evaluation namespace exposing report functions plus `metrics` and `tests`; raw metric/test functions are not re-exported directly from it. |
| `SearchSpec`, `SearchResult`, `SearchError`, `ParamDistribution`, `search_spec`, `select_params` | `macroforecast.model_selection` | Model-parameter selection over a supplied window and metric. |
| `fixed`, `grid`, `random_search`, `cv_path`, `bayesian_search`, `genetic_search`, `custom_search`, `choice`, `uniform`, `log_uniform`, `randint` | `macroforecast.model_selection` | Search specification and distribution builders. |
| `ForecastResult`, `run`, `run_forecast` | `macroforecast.forecasting` | Windowed forecast runner. |
| `CombinationSpec`, `combination_spec`, `custom_combination`, `combine_mean`, `combine_median`, `combine_trimmed_mean`, `combine_winsorized_mean`, `combine_inverse_mspe`, `combine_dmspe`, `combine_best_n` | `macroforecast.forecasting` | Runner-integrated and direct forecast combination methods. |

## Submodules

| Module | Purpose |
| --- | --- |
| `macroforecast.meta` | Global defaults. |
| `macroforecast.data` | Data loading and study data specs. |
| `macroforecast.preprocessing` | Pandas preprocessing functions. |
| `macroforecast.filters` | Direct one-series filters and smoothers such as HP, Hamilton, Savitzky-Golay, wavelet-style components, and AlbaMA. |
| `macroforecast.feature_engineering` | Direct-forecast target construction and composable ML feature transforms. |
| `macroforecast.feature_analysis` | Feature-matrix analysis after feature engineering. |
| `macroforecast.feature_diagnostic` | Compatibility alias for `macroforecast.feature_analysis`. |
| `macroforecast.models` | Callable model fits. |
| `macroforecast.model_ensemble` | Fit-time model composition over multiple member models. |
| `macroforecast.window` | Macro forecasting time-frame specs. |
| `macroforecast.model_selection` | Hyperparameter search and parameter selection. |
| `macroforecast.forecasting` | Windowed runner and forecast combination. |
| `macroforecast.forecast_analysis` | Forecast-result analysis after runner execution. |
| `macroforecast.forecast_diagnostic` | Compatibility alias for `macroforecast.forecast_analysis`. |
| `macroforecast.metrics` | Scoring metrics, forecast ranking, and metric resolution. |
| `macroforecast.tests` | Forecast-comparison tests and residual diagnostics. |
| `macroforecast.evaluation` | Evaluation reports, OOS filtering, error decomposition, benchmark comparisons, regime scoring, and namespace links to `metrics` and `tests`. |
| `macroforecast.interpretation` | Model-native importance, model-agnostic effects, SHAP/anatomy attribution, OLS-as-attention, VAR interpretation, and deep optional helpers. |
| `macroforecast.output` | Output generation, artifact writing, provenance collection, hashing, and compression. |
| `macroforecast.reporting` | Presentation/report formatting, paper-table presets, and rendering without artifact writing. |
| `macroforecast.data_analysis` | Single-panel diagnostics, summaries, and raw-versus-processed comparison. |
