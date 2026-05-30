# Public Python API

[Back to reference](index.md)

The importable surface is intentionally narrow and pandas-first.

## Top-Level Exports

| Symbol | Source | Description |
| --- | --- | --- |
| `configure`, `get_config`, `get_option`, `reset_config`, `use_config`, `StageDefaultScope`, `MetadataLevel` | `macroforecast.meta` | Global package defaults and config types. |
| `DataBundle`, `DataSpec`, `as_panel`, `metadata`, `panel_info`, `spec`, `validate_panel` | `macroforecast.data` | Canonical panel and metadata helpers. |
| `load_fred_md`, `load_fred_qd`, `load_fred_sd`, `load_fred_md_sd`, `load_fred_qd_sd` | `macroforecast.data` | Dataset loaders. |
| `load_custom_csv`, `load_custom_parquet`, `list_vintages`, `combine` | `macroforecast.data` | Custom loading, vintage discovery, and panel combination. |
| `reprocess`, `standardize_panel`, `PreprocessedData` | `macroforecast.preprocessing` | Direct pandas preprocessing. |
| `PreprocessSpec`, `FittedPreprocessor`, `preprocess_spec` | `macroforecast.preprocessing` | Runner-compatible preprocessing fit/transform specs. |
| `FeatureSet`, `FeatureSpec`, `FittedFeatureBuilder`, `build_features`, `feature_spec` | `macroforecast.feature_engineering` | Aligned forecast matrices, metadata, and runner-compatible feature specs. |
| `direct_target`, `average_target`, `path_targets` | `macroforecast.feature_engineering` | Direct and path target construction. |
| `feature_matrix`, `compose_features` | `macroforecast.feature_engineering` | Paper-style feature blocks and sequential feature composition. |
| `lag`, `rolling_mean`, `moving_average_ladder`, `scale_features`, `pca_features`, `group_pca`, `maf_features`, `time_features` | `macroforecast.feature_engineering` | Direct pandas feature transforms. |
| `lag_step`, `rolling_step`, `moving_average_step`, `scale_step`, `pca_step`, `group_pca_step`, `maf_step` | `macroforecast.feature_engineering` | Reusable step dictionaries for `compose_features`. |
| `pca_then_lags`, `lags_then_pca`, `moving_average_pca_lags` | `macroforecast.feature_engineering` | Convenience composed feature callables. |
| `ModelFit`, `VolatilityFit`, `SavedModel`, `save_fit`, `load_fit` | `macroforecast.models` | Fitted model result wrappers and low-level fit persistence. |
| `ols`, `ridge`, `lasso`, `elastic_net`, `adaptive_lasso`, `adaptive_elastic_net`, `group_lasso`, `sparse_group_lasso`, `bayesian_ridge`, `huber`, `glmboost`, `pls`, `scaled_pca`, `supervised_pca`, `supervised_scaled_pca` | `macroforecast.models` | Linear, penalized, grouped, and supervised dimension-reduction models. |
| `svr`, `linear_svr`, `nu_svr` | `macroforecast.models` | Support-vector regression models. |
| `nn`, `lstm`, `gru` | `macroforecast.models` | Torch-backed neural-network regressors; require `macroforecast[deep]`. |
| `ar`, `var`, `far`, `favar` | `macroforecast.models` | Time-series and factor-augmented forecasting models. |
| `decision_tree`, `random_forest`, `extra_trees`, `gradient_boosting`, `xgboost`, `lightgbm`, `catboost` | `macroforecast.models` | Tree and ML regressors. |
| `slow_growing_tree`, `quantile_regression_forest`, `bagging`, `booging`, `macro_random_forest` | `macroforecast.models` | Macro-specific tree and ensemble models. |
| `garch11`, `egarch`, `realized_garch` | `macroforecast.models` | Volatility models. |
| `ModelSpec`, `ModelParameter`, `get_model`, `list_model_specs`, `describe_model`, `model_search_space` | `macroforecast.models` | Model-owned defaults and hyperparameter spaces. |
| `WindowSpec`, `EstimationWindow`, `ValWindow`, `TestWindow`, `AlignmentWindow`, `StagePolicy` | `macroforecast.window` | Forecast experiment and stage timing objects. |
| `from_cutoffs`, `estimation_expanding`, `estimation_rolling`, `estimation_fixed`, `val_last_block`, `val_poos`, `val_expanding`, `val_rolling_blocks`, `val_blocked_kfold`, `test_origins`, `alignment_drop_incomplete`, `alignment_keep_missing` | `macroforecast.window` | Component window builders. |
| `last_block`, `poos`, `expanding`, `rolling_blocks`, `blocked_kfold` | `macroforecast.window` | Shortcut temporal window specs. |
| `stage_policy`, `stage_index`, `stage_panel`, `last_block_split`, `poos_split`, `expanding_split`, `rolling_blocks_split`, `blocked_kfold_split`, `split_table`, `normalize_window_name` | `macroforecast.window` | Stage timing and train/val split inspection. |
| `mse`, `rmse`, `mae`, `pinball_loss`, `gaussian_nll`, `coverage_rate`, `interval_width`, `evaluate_forecasts`, `get_metric` | `macroforecast.evaluation` | Point, variance, quantile, interval, and forecast-table scoring. |
| `SearchSpec`, `SearchResult`, `SearchError`, `search_spec`, `select_params` | `macroforecast.selection` | Parameter selection over a supplied window and metric. |
| `fixed`, `grid`, `random_search`, `cv_path`, `bayesian_search`, `genetic_search` | `macroforecast.selection` | Search specification builders. |
| `ForecastResult`, `run`, `run_forecast` | `macroforecast.forecasting` | Windowed forecast runner. |
| `combine_mean`, `combine_median`, `combine_trimmed_mean`, `combine_winsorized_mean`, `combine_inverse_mspe`, `combine_dmspe`, `combine_best_n` | `macroforecast.forecasting` | Forecast combination methods. |
| `summarize_data`, `DataSummaryReport` | `macroforecast.data_summary` | Single-panel summaries. |
| `analyze_data`, `DataAnalysisReport` | `macroforecast.data_analysis` | Before/after panel analysis. |

## Submodules

| Module | Purpose |
| --- | --- |
| `macroforecast.meta` | Global defaults. |
| `macroforecast.data` | Data loading and study data specs. |
| `macroforecast.preprocessing` | Pandas preprocessing functions. |
| `macroforecast.feature_engineering` | Direct-forecast target construction and composable ML feature transforms. |
| `macroforecast.models` | Callable model fits. |
| `macroforecast.window` | Macro forecasting time-frame specs. |
| `macroforecast.selection` | Hyperparameter search and parameter selection. |
| `macroforecast.forecasting` | Windowed runner and forecast combination. |
| `macroforecast.evaluation` | Scoring metrics. |
| `macroforecast.data_summary` | Single-panel diagnostics and summaries. |
| `macroforecast.data_analysis` | Raw-versus-processed comparison. |
