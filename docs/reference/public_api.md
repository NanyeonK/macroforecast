# Public Python API

[Back to reference](index.md)

The importable surface is intentionally narrow and pandas-first.

## Top-Level Exports

| Symbol | Source | Description |
| --- | --- | --- |
| `configure`, `get_config`, `get_option`, `reset_config`, `use_config`, `StageDefaultScope`, `MetadataLevel` | `macroforecast.meta` | Global package defaults and config types. |
| `DataBundle`, `DataSpec`, `as_panel`, `custom_dataset`, `metadata`, `panel_info`, `set_frequencies`, `spec`, `validate_panel` | `macroforecast.data` | Canonical panel and metadata helpers. |
| `align_frequency`, `chow_lin_disaggregate`, `infer_frequencies`, `frequency_hardening_issues`, `availability_lag`, `same_period_predictors`, `define_regime` | `macroforecast.data` | Frequency alignment/inference, Chow-Lin disaggregation, data availability, same-period predictor, and regime metadata policies. |
| `load_fred_md`, `load_fred_qd`, `load_fred_sd`, `load_fred_md_sd`, `load_fred_qd_sd` | `macroforecast.data` | Dataset loaders. |
| `load_custom_csv`, `load_custom_parquet`, `list_vintages`, `combine` | `macroforecast.data` | Custom loading, vintage discovery, and panel combination. |
| `reprocess`, `custom_preprocess`, `standardize_panel`, `PreprocessedData` | `macroforecast.preprocessing` | Direct pandas preprocessing. |
| `PreprocessSpec`, `FittedPreprocessor`, `preprocess_spec`, `custom_preprocess_step` | `macroforecast.preprocessing` | Runner-compatible preprocessing fit/transform specs. |
| `FeatureSet`, `FeatureSpec`, `FittedFeatureBuilder`, `build_features`, `feature_spec` | `macroforecast.feature_engineering` | Aligned forecast matrices, metadata, and runner-compatible feature specs. |
| `direct_target`, `average_target`, `path_targets` | `macroforecast.feature_engineering` | Direct and path target construction. |
| `feature_matrix`, `compose_features` | `macroforecast.feature_engineering` | Paper-style feature blocks and sequential feature composition. |
| `lag`, `mixed_frequency_lags`, `rolling_mean`, `moving_average_ladder`, `scale_features`, `pca_features`, `sparse_pca_chen_rohe_features`, `varimax_features`, `group_pca`, `maf_features`, `time_features`, `custom_features` | `macroforecast.feature_engineering` | Core direct pandas feature transforms. |
| `transform_features`, `log_features`, `diff_features`, `log_diff_features`, `pct_change_features`, `cumsum_features`, `seasonal_lag`, `season_dummy`, `fourier_features`, `polynomial_features`, `interaction_features`, `hp_filter_features`, `hamilton_filter_features`, `savitzky_golay_features`, `wavelet_features`, `adaptive_ma_rf_features`, `asymmetric_trim_features`, `partial_least_squares_features`, `sliced_inverse_regression_features`, `dfm_features`, `variance_selection`, `correlation_selection`, `lasso_selection`, `lasso_path_selection`, `rfe_selection`, `boruta_selection`, `stability_selection`, `genetic_selection`, `random_projection_features`, `nystroem_features` | `macroforecast.feature_engineering` | Additional transform, seasonal, expansion, filter, supervised factor, selection, and kernel-approximation feature functions. |
| `lag_step`, `rolling_step`, `moving_average_step`, `marx_step`, `transform_step`, `seasonal_lag_step`, `season_dummy_step`, `fourier_step`, `time_step`, `polynomial_step`, `interaction_step`, `scale_step`, `pca_step`, `sparse_pca_chen_rohe_step`, `varimax_step`, `group_pca_step`, `maf_step`, `hamilton_step`, `random_projection_step`, `nystroem_step`, `partial_least_squares_step`, `sliced_inverse_regression_step`, `custom_step` | `macroforecast.feature_engineering` | Reusable step dictionaries for `compose_features` and runner-safe `feature_spec` pipelines. Feature selection uses individual `method` names in step mappings instead of a generic step builder. |
| `pca_then_lags`, `lags_then_pca`, `moving_average_pca_lags` | `macroforecast.feature_engineering` | Convenience composed feature callables. |
| `FeatureDiagnosticReport`, `diagnose_features`, `custom_feature_diagnostic`, `feature_overview`, `compare_feature_stages`, `stage_distribution_shift`, `feature_correlation`, `feature_correlation_matrix`, `feature_target_correlation`, `factor_diagnostics`, `factor_variance`, `factor_loadings`, `factor_timeseries`, `lag_diagnostics`, `lag_autocorrelation`, `lag_correlation_decay`, `marx_diagnostics`, `marx_weight_decay`, `selection_stability`, `selection_similarity` | `macroforecast.feature_analysis` | Feature-matrix quality, structure, metadata, custom diagnostics, and selection-stability analysis. |
| `ModelFit`, `VolatilityFit`, `SavedModel`, `save_fit`, `load_fit` | `macroforecast.models` | Fitted model result wrappers and low-level fit persistence. |
| `ols`, `ridge`, `nonneg_ridge`, `shrink_to_target_ridge`, `fused_difference_ridge`, `random_walk_ridge`, `lasso`, `elastic_net`, `adaptive_lasso`, `adaptive_elastic_net`, `group_lasso`, `sparse_group_lasso`, `bayesian_ridge`, `huber`, `kernel_ridge`, `knn`, `glmboost`, `pls`, `scaled_pca`, `supervised_pca`, `supervised_scaled_pca` | `macroforecast.models` | Linear, penalized, grouped, kernel, nearest-neighbor, and supervised dimension-reduction models. |
| `svr`, `linear_svr`, `nu_svr` | `macroforecast.models` | Support-vector regression models. |
| `nn`, `lstm`, `gru`, `transformer`, `hemisphere_nn` | `macroforecast.models` | Torch-backed neural-network regressors; require `macroforecast[deep]`. |
| `ar`, `var`, `bvar_minnesota`, `bvar_normal_inverse_wishart`, `ets`, `holt_winters`, `theta_method`, `far`, `favar` | `macroforecast.models` | Time-series, Bayesian VAR, exponential-smoothing, and factor-augmented forecasting models. |
| `dfm_mixed_mariano_murasawa`, `dfm_unrestricted_midas`, `midas_almon`, `midas_beta`, `midas_step`, `unrestricted_midas` | `macroforecast.models` | Mixed-frequency dynamic-factor and MIDAS models. |
| `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `mars`, `xgboost`, `lightgbm`, `catboost` | `macroforecast.models` | Tree, spline, and ML regressors. |
| `slow_growing_tree`, `quantile_regression_forest`, `bagging`, `booging`, `macro_random_forest` | `macroforecast.models` | Macro-specific tree and ensemble models. |
| `garch11`, `egarch`, `realized_garch` | `macroforecast.models` | Volatility models. |
| `ModelSpec`, `ModelParameter`, `custom_model`, `get_model`, `list_model_specs`, `describe_model`, `model_search_space` | `macroforecast.models` | Model-owned defaults and hyperparameter spaces. |
| `WindowSpec`, `EstimationWindow`, `ValWindow`, `TestWindow`, `AlignmentWindow`, `StagePolicy` | `macroforecast.window` | Forecast experiment and stage timing objects. |
| `from_cutoffs`, `estimation_expanding`, `estimation_rolling`, `estimation_fixed`, `val_last_block`, `val_poos`, `val_expanding`, `val_rolling_blocks`, `val_blocked_kfold`, `test_origins`, `alignment_drop_incomplete`, `alignment_keep_missing` | `macroforecast.window` | Component window builders. |
| `last_block`, `poos`, `expanding`, `rolling_blocks`, `blocked_kfold` | `macroforecast.window` | Shortcut temporal window specs. |
| `stage_policy`, `custom_stage_policy`, `stage_index`, `stage_panel`, `last_block_split`, `poos_split`, `expanding_split`, `rolling_blocks_split`, `blocked_kfold_split`, `split_table`, `normalize_window_name` | `macroforecast.window` | Stage timing and train/val split inspection. |
| `metrics` | `macroforecast.metrics` | Forecast scoring namespace, including `mf.metrics.bias`. Use `mf.metrics.rmse`, not `mf.rmse`. |
| `tests` | `macroforecast.tests` | Forecast-comparison test namespace, including `mf.tests.custom_test`, `mf.tests.equal_predictive_tests`, interval coverage, and PIT diagnostics. Use `mf.tests.dm_test`, not `mf.dm_test`. |
| `EvaluationReport`, `evaluate_report`, `aggregate_scores`, `filter_oos_period`, `error_decomposition`, `benchmark_comparison`, `regime_scores` | `macroforecast.evaluation` | Multi-slice evaluation reports, OOS filtering, error decomposition, benchmark comparisons, and regime scoring. |
| `evaluation` | `macroforecast.evaluation` | Evaluation namespace exposing report functions plus `metrics` and `tests`; raw metric/test functions are not re-exported directly from it. |
| `SearchSpec`, `SearchResult`, `SearchError`, `search_spec`, `select_params` | `macroforecast.selection` | Parameter selection over a supplied window and metric. |
| `fixed`, `grid`, `random_search`, `cv_path`, `bayesian_search`, `genetic_search`, `custom_search` | `macroforecast.selection` | Search specification builders. |
| `ForecastResult`, `run`, `run_forecast` | `macroforecast.forecasting` | Windowed forecast runner. |
| `CombinationSpec`, `combination_spec`, `custom_combination`, `combine_mean`, `combine_median`, `combine_trimmed_mean`, `combine_winsorized_mean`, `combine_inverse_mspe`, `combine_dmspe`, `combine_best_n` | `macroforecast.forecasting` | Runner-integrated and direct forecast combination methods. |
| `ForecastDiagnosticReport`, `diagnose_forecasts`, `custom_forecast_diagnostic`, `forecast_overview`, `fitted_vs_actual`, `residual_report`, `residual_autocorrelation`, `residual_qq`, `rolling_loss`, `forecast_scale_view`, `select_forecast_origins`, `first_vs_last_forecast`, `training_loss_trace`, `rolling_training_loss`, `dfm_idiosyncratic_acf`, `dfm_factor_stability`, `coefficient_trace`, `parameter_stability`, `tuning_trace`, `tuning_objective_trace`, `hyperparameter_path`, `tuning_score_distribution`, `ensemble_weights_over_time`, `ensemble_weight_concentration`, `ensemble_member_contribution`, `stage_update_trace` | `macroforecast.forecast_analysis` | Forecast-result analysis for residuals, tuning, coefficients, combinations, custom diagnostics, and runner stage updates. |
| `linear_coefficients`, `tree_importance`, `permutation_importance`, `permutation_importance_strobl`, `lofo_importance`, `partial_dependence`, `accumulated_local_effect`, `friedman_h_interaction`, `shap_values`, `shap_importance`, `forecast_decomposition`, `cumulative_r2_contribution`, `rolling_recompute`, `bootstrap_jackknife`, `group_aggregate`, `lineage_attribution`, `transformation_attribution`, `attention_weights`, `dual_decomposition`, `mrf_gtvp`, `generalized_irf`, `orthogonalised_irf`, `fevd`, `historical_decomposition`, `lasso_inclusion_frequency`, `lstm_hidden_state`, `saliency_map`, `integrated_gradients`, `gradient_shap`, `deep_lift`, `custom_interpretation` | `macroforecast.interpretation` | Model-native, model-agnostic, effect-curve, SHAP, group/pipeline attribution, OLS attention, MRF GTVP, VAR interpretation, neural attribution, and custom interpretation helpers. |
| `forecast_table`, `metric_table`, `ranking_table`, `test_table`, `model_table`, `selection_table`, `interpretation_table`, `metadata_table`, `run_summary`, `bundle_outputs`, `select_outputs`, `name_outputs`, `artifact_index`, `OutputBundle` | `macroforecast.output` | Output-generating helpers. These create named tables and JSON summaries without writing files. |
| `write_artifacts`, `collect_provenance`, `ArtifactManifest`, `ArtifactRecord`, `CompressionFormat` | `macroforecast.output` | Artifact export, custom artifact storage, file hashing, compression, and provenance manifest helpers. |
| `summarize_data`, `DataSummaryReport` | `macroforecast.data_summary` | Single-panel summaries. |
| `analyze_data`, `DataAnalysisReport` | `macroforecast.data_analysis` | Before/after panel analysis. |

## Submodules

| Module | Purpose |
| --- | --- |
| `macroforecast.meta` | Global defaults. |
| `macroforecast.data` | Data loading and study data specs. |
| `macroforecast.preprocessing` | Pandas preprocessing functions. |
| `macroforecast.feature_engineering` | Direct-forecast target construction and composable ML feature transforms. |
| `macroforecast.feature_analysis` | Feature-matrix analysis after feature engineering. |
| `macroforecast.feature_diagnostic` | Compatibility alias for `macroforecast.feature_analysis`. |
| `macroforecast.models` | Callable model fits. |
| `macroforecast.window` | Macro forecasting time-frame specs. |
| `macroforecast.selection` | Hyperparameter search and parameter selection. |
| `macroforecast.forecasting` | Windowed runner and forecast combination. |
| `macroforecast.forecast_analysis` | Forecast-result analysis after runner execution. |
| `macroforecast.forecast_diagnostic` | Compatibility alias for `macroforecast.forecast_analysis`. |
| `macroforecast.metrics` | Scoring metrics, forecast ranking, and metric resolution. |
| `macroforecast.tests` | Forecast-comparison tests and residual diagnostics. |
| `macroforecast.evaluation` | Evaluation reports, OOS filtering, error decomposition, benchmark comparisons, regime scoring, and namespace links to `metrics` and `tests`. |
| `macroforecast.interpretation` | Model-native importance, model-agnostic effects, attribution, VAR interpretation, and SHAP/deep optional helpers. |
| `macroforecast.output` | Output generation, artifact writing, provenance collection, hashing, and compression. |
| `macroforecast.data_summary` | Single-panel diagnostics and summaries. |
| `macroforecast.data_analysis` | Raw-versus-processed comparison. |
