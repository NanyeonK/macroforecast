# Layer Axis Census

Date: 2026-04-27

This census is generated from the live registry and Navigator tree after PR #92. It is the baseline for the next layer-by-layer audit: each layer should be checked for registry vocabulary, runtime support, compiler pruning, Navigator visibility, docs coverage, and example coverage.

## How To Read This

- `Axis count` counts registry axes owned by the layer.
- `Value statuses` counts registry values, not axes.
- `Navigator tree axes` are the axes exposed in the primary decision tree. Some lowered/internal axes are intentionally not exposed as first-click UI choices.
- `Registry axes outside Navigator tree` are not automatically bugs; they are the first review queue for the layer audit.
- Virtual axes such as `exogenous_x_path_policy` are Navigator/compiler route axes that do not have a normal registry module yet.

## Summary

| Layer | Axis count | Value statuses | Navigator tree axes | Registry axes outside Navigator tree | Virtual/non-registry tree axes |
|---|---:|---|---|---|---|
| `0_meta` | 6 | operational=28, registry_only=7, future=3 | 5 | `axis_type` | - |
| `1_data_task` | 19 | operational=83 | 17 | `sd_variable_selection`, `state_selection` | - |
| `2_preprocessing` | 45 | operational=156, operational_narrow=8, external_plugin=12, registry_only=52, future=3 | 19 | `additional_preprocessing`, `custom_preprocessor`, `data_richness_mode`, `deterministic_components`, `dimensionality_reduction_policy`, `factor_count`, `factor_rotation_order`, `feature_block_set`, `feature_grouping`, `inverse_transform_policy`, `predictor_family`, `preprocess_fit_scope`, `preprocess_order`, `representation_policy`, `scaling_scope`, `separation_rule`, `structural_break_segmentation`, `target_domain`, `target_lag_selection`, `target_missing_policy`, `target_outlier_policy`, `target_transform_policy`, `target_transformer`, `tcode_application_scope`, `x_lag_creation`, `x_transform_policy` | - |
| `3_training` | 31 | operational=123, operational_narrow=8, registry_only=28, future=10 | 18 | `alignment_fairness`, `cache_policy`, `checkpointing`, `convergence_handling`, `early_stopping`, `embargo_gap`, `execution_backend`, `horizon_modelization`, `hp_space_style`, `logging_level`, `lookback`, `seed_policy`, `sequence_framework`, `shuffle_rule`, `split_family` | `exogenous_x_path_policy`, `recursive_x_model_family` |
| `4_evaluation` | 19 | operational=57, registry_only=40, future=17 | 19 | - | - |
| `5_output_provenance` | 4 | operational=13, registry_only=2, future=4 | 4 | - | - |
| `6_stat_tests` | 11 | operational=74, registry_only=3, future=3 | 11 | - | - |
| `7_importance` | 13 | operational=42, registry_only=12 | 13 | - | - |

## Operational-Narrow Contract Queue

| Axis | Owner layer | Values | Contract | Required companions | Enforcement surface |
|---|---|---|---|---|---|
| `feature_block_set` | `2_preprocessing` | `transformed_predictor_lags`, `selected_sparse_predictors`, `level_augmented_predictors`, `rotation_augmented_predictors`, `mixed_feature_blocks`, `custom_feature_blocks` | `feature_block_set_public_axis_v1` | x_lag_feature_block=fixed_predictor_lags for transformed_predictor_lags<br>non-none feature_selection_policy for selected_sparse_predictors<br>non-none level_feature_block for level_augmented_predictors<br>non-none rotation_feature_block for rotation_augmented_predictors<br>at least two active block sources for mixed_feature_blocks<br>registered custom block or custom combiner for custom_feature_blocks | compiler blocked_reasons and skip_failed_cell manifests |
| `fred_sd_mixed_frequency_representation` | `2_preprocessing` | `native_frequency_block_payload`, `mixed_frequency_model_adapter` | `fred_sd_native_frequency_block_payload_v1` | dataset includes fred_sd<br>feature_builder=raw_feature_panel<br>registered custom model_family or built-in MIDAS model_family<br>forecast_type=direct<br>mixed_frequency_model_adapter additionally records fred_sd_mixed_frequency_model_adapter_v1 | Navigator compatibility, compiler blocked_reasons, and runtime route guard |
| `exogenous_x_path_policy` | `3_training` | `hold_last_observed`, `observed_future_x`, `scheduled_known_future_x`, `recursive_x_model` | `exogenous_x_path_contract_v1` | forecast_type=iterated<br>raw-panel feature runtime<br>target_lag_block=fixed_target_lags<br>scheduled_known_future_x_columns for scheduled_known_future_x<br>recursive_x_model_family=ar1 for recursive_x_model | Layer 3 capability matrix and compiler blocked_reasons |
| `recursive_x_model_family` | `3_training` | `ar1` | `exogenous_x_path_contract_v1` | exogenous_x_path_policy=recursive_x_model<br>raw-panel iterated point forecast slice | Navigator compatibility and compiler blocked_reasons |

## Layer Detail

### Layer 0: study design and execution grammar

| Axis | Component | Type | Default policy | Value status counts | Values by status |
|---|---|---|---|---|---|
| `axis_type` | - | `enum` | `fixed` | operational=5 | `operational`: `fixed`, `sweep`, `nested_sweep`, `conditional`, `derived` |
| `compute_mode` | - | `enum` | `fixed` | operational=5 | `operational`: `serial` (default), `parallel_by_model`, `parallel_by_horizon`, `parallel_by_target`, `parallel_by_oos_date` |
| `study_scope` | - | `enum` | `fixed` | operational=4 | `operational`: `one_target_one_method`, `one_target_compare_methods`, `multiple_targets_one_method`, `multiple_targets_compare_methods` |
| `failure_policy` | - | `enum` | `fixed` | operational=5 | `operational`: `fail_fast` (default), `skip_failed_cell`, `skip_failed_model`, `save_partial_results`, `warn_only` |
| `reproducibility_mode` | - | `enum` | `fixed` | operational=4 | `operational`: `strict_reproducible`, `seeded_reproducible` (default), `best_effort`, `exploratory` |

### Layer 1: source data, target, and official frame

| Axis | Component | Type | Default policy | Value status counts | Values by status |
|---|---|---|---|---|---|
| `contemporaneous_x_rule` | - | `enum` | `fixed` | operational=2 | `operational`: `allow_same_period_predictors`, `forbid_same_period_predictors` |
| `custom_source_format` | - | `enum` | `fixed` | operational=3 | `operational`: `none`, `csv`, `parquet` |
| `custom_source_policy` | - | `enum` | `fixed` | operational=3 | `operational`: `official_only`, `custom_panel_only`, `official_plus_custom` |
| `custom_source_schema` | - | `enum` | `fixed` | operational=4 | `operational`: `none`, `fred_md`, `fred_qd`, `fred_sd` |
| `dataset` | - | `enum` | `fixed` | operational=5 | `operational`: `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd` |
| `fred_sd_frequency_policy` | - | `enum` | `fixed` | operational=4 | `operational`: `report_only`, `allow_mixed_frequency`, `reject_mixed_known_frequency`, `require_single_known_frequency` |
| `fred_sd_state_group` | - | `enum` | `fixed` | operational=16 | `operational`: `all_states`, `census_region_northeast`, `census_region_midwest`, `census_region_south`, `census_region_west`, `census_division_new_england`, `census_division_middle_atlantic`, `census_division_east_north_central`, `census_division_west_north_central`, `census_division_south_atlantic`, `census_division_east_south_central`, `census_division_west_south_central`, `census_division_mountain`, `census_division_pacific`, `contiguous_48_plus_dc`, `custom_state_group` |
| `fred_sd_variable_group` | - | `enum` | `fixed` | operational=12 | `operational`: `all_sd_variables`, `labor_market_core`, `employment_sector`, `gsp_output`, `housing`, `trade`, `income`, `direct_analog_high_confidence`, `provisional_analog_medium`, `semantic_review_outputs`, `no_reliable_analog`, `custom_sd_variable_group` |
| `frequency` | - | `enum` | `fixed` | operational=2 | `operational`: `monthly`, `quarterly` |
| `information_set_type` | - | `enum` | `fixed` | operational=2 | `operational`: `final_revised_data`, `pseudo_oos_on_revised_data` |
| `missing_availability` | - | `enum` | `fixed` | operational=4 | `operational`: `require_complete_rows`, `keep_available_rows`, `impute_predictors_only`, `zero_fill_leading_predictor_gaps` |
| `official_transform_policy` | - | `enum` | `fixed` | operational=2 | `operational`: `apply_official_tcode`, `keep_official_raw_scale` |
| `official_transform_scope` | - | `enum` | `fixed` | operational=4 | `operational`: `target_only`, `predictors_only`, `target_and_predictors`, `none` |
| `raw_missing_policy` | - | `enum` | `fixed` | operational=4 | `operational`: `preserve_raw_missing`, `zero_fill_leading_predictor_missing_before_tcode`, `impute_raw_predictors`, `drop_raw_missing_rows` |
| `raw_outlier_policy` | - | `enum` | `fixed` | operational=6 | `operational`: `preserve_raw_outliers`, `winsorize_raw`, `iqr_clip_raw`, `mad_clip_raw`, `zscore_clip_raw`, `set_raw_outliers_to_missing` |
| `release_lag_rule` | - | `enum` | `fixed` | operational=3 | `operational`: `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag` |
| `sd_variable_selection` | - | `enum` | `fixed` | operational=2 | `operational`: `all_sd_variables`, `selected_sd_variables` |
| `state_selection` | - | `enum` | `fixed` | operational=2 | `operational`: `all_states`, `selected_states` |
| `target_structure` | - | `enum` | `fixed` | operational=2 | `operational`: `single_target`, `multi_target` |
| `variable_universe` | - | `enum` | `fixed` | operational=5 | `operational`: `all_variables`, `core_variables`, `category_variables`, `target_specific_variables`, `explicit_variable_list` |

### Layer 2: preprocessing and representation construction

| Axis | Component | Type | Default policy | Value status counts | Values by status |
|---|---|---|---|---|---|
| `additional_preprocessing` | - | `enum` | `fixed` | operational=2, registry_only=3 | `operational`: `none`, `hp_filter`<br>`registry_only`: `smoothing_ma`, `ema`, `bandpass_filter` |
| `custom_preprocessor` | preprocessing | `plugin` | `fixed` | operational=1 | `operational`: `none` |
| `data_richness_mode` | feature_representation | `enum` | `fixed` | operational=4, registry_only=1 | `operational`: `target_lags_only`, `factors_plus_target_lags`, `high_dimensional_predictors`, `selected_sparse_predictors`<br>`registry_only`: `mixed_feature_blocks` |
| `deterministic_components` | - | `enum` | `fixed` | operational=6 | `operational`: `none`, `constant_only`, `linear_trend`, `monthly_seasonal`, `quarterly_seasonal`, `break_dummies` |
| `dimensionality_reduction_policy` | preprocessing | `enum` | `fixed` | operational=3, external_plugin=1 | `operational`: `none`, `pca`, `static_factor`<br>`external_plugin`: `custom` |
| `evaluation_scale` | - | `enum` | `fixed` | operational=4 | `operational`: `original_scale`, `raw_level`, `transformed_scale`, `both` |
| `factor_count` | feature_representation | `enum` | `fixed` | operational=3, registry_only=2 | `operational`: `fixed`, `cv_select`, `BaiNg_rule`<br>`registry_only`: `variance_explained_rule`, `model_specific` |
| `factor_feature_block` | feature_representation | `enum` | `fixed` | operational=4, registry_only=1 | `operational`: `none`, `pca_static_factors`, `pca_factor_lags`, `supervised_factors`<br>`registry_only`: `custom_factors` |
| `factor_rotation_order` | feature_representation | `enum` | `fixed` | operational=2 | `operational`: `rotation_then_factor`, `factor_then_rotation` |
| `feature_block_combination` | feature_representation | `enum` | `fixed` | operational=4, registry_only=1 | `operational`: `replace_with_selected_blocks`, `append_to_base_predictors`, `append_to_target_lags`, `concatenate_named_blocks`<br>`registry_only`: `custom_feature_combiner` |
| `feature_block_set` | feature_representation | `enum` | `fixed` | operational=5, operational_narrow=6, registry_only=1 | `operational`: `target_lags_only`, `transformed_predictors`, `factors_plus_target_lags`, `factor_blocks_only`, `high_dimensional_predictors`<br>`operational_narrow`: `transformed_predictor_lags`, `selected_sparse_predictors`, `level_augmented_predictors`, `rotation_augmented_predictors`, `mixed_feature_blocks`, `custom_feature_blocks`<br>`registry_only`: `feature_builder_compatibility_bridge` |
| `feature_builder` | feature_representation | `enum` | `fixed` | operational=5, future=1 | `operational`: `target_lag_features`, `factors_plus_target_lags`, `raw_feature_panel`, `raw_predictors_only`, `pca_factor_features`<br>`future`: `sequence_tensor` |
| `feature_grouping` | feature_representation | `enum` | `fixed` | operational=1, registry_only=4 | `operational`: `none`<br>`registry_only`: `fred_category_group`, `economic_theme_group`, `lag_group`, `factor_group` |
| `feature_selection_policy` | preprocessing | `enum` | `fixed` | operational=3, external_plugin=1, registry_only=1 | `operational`: `none`, `correlation_filter`, `lasso_selection`<br>`external_plugin`: `custom`<br>`registry_only`: `mutual_information_screen` |
| `feature_selection_semantics` | feature_representation | `enum` | `fixed` | operational=3 | `operational`: `select_before_factor`, `select_after_factor`, `select_after_custom_feature_blocks` |
| `fred_sd_mixed_frequency_representation` | feature_representation | `enum` | `fixed` | operational=3, operational_narrow=2 | `operational`: `calendar_aligned_frame`, `drop_unknown_native_frequency`, `drop_non_target_native_frequency`<br>`operational_narrow`: `native_frequency_block_payload`, `mixed_frequency_model_adapter` |
| `horizon_target_construction` | - | `enum` | `fixed` | operational=10 | `operational`: `future_target_level_t_plus_h`, `future_target_level_t_plus_h`, `future_diff`, `future_logdiff`, `average_growth_1_to_h`, `path_average_growth_1_to_h`, `average_difference_1_to_h`, `path_average_difference_1_to_h`, `average_log_growth_1_to_h`, `path_average_log_growth_1_to_h` |
| `inverse_transform_policy` | - | `enum` | `fixed` | operational=3, external_plugin=1 | `operational`: `none`, `target_only`, `forecast_scale_only`<br>`external_plugin`: `custom` |
| `level_feature_block` | feature_representation | `enum` | `fixed` | operational=5 | `operational`: `none`, `target_level_addback`, `x_level_addback`, `selected_level_addbacks`, `level_growth_pairs` |
| `predictor_family` | feature_representation | `enum` | `fixed` | operational=5 | `operational`: `target_lags_only`, `all_macro_vars`, `category_based`, `factor_only`, `explicit_variable_list` |
| `preprocess_fit_scope` | - | `enum` | `fixed` | operational=2, registry_only=2 | `operational`: `not_applicable`, `train_only`<br>`registry_only`: `expanding_train_only`, `rolling_train_only` |
| `preprocess_order` | - | `enum` | `fixed` | operational=4, external_plugin=1, registry_only=1 | `operational`: `none`, `official_tcode_only`, `extra_only`, `official_tcode_then_extra`<br>`external_plugin`: `custom`<br>`registry_only`: `extra_preprocess_then_official_tcode` |
| `representation_policy` | - | `enum` | `fixed` | operational=2, registry_only=1 | `operational`: `raw_only`, `official_tcode_only`<br>`registry_only`: `custom_transform_only` |
| `rotation_feature_block` | feature_representation | `enum` | `fixed` | operational=4, registry_only=1 | `operational`: `none`, `marx_rotation`, `maf_rotation`, `moving_average_rotation`<br>`registry_only`: `custom_rotation` |
| `scaling_policy` | preprocessing | `enum` | `fixed` | operational=6, external_plugin=1, registry_only=1 | `operational`: `none`, `standard`, `robust`, `minmax`, `demean_only`, `unit_variance_only`<br>`external_plugin`: `custom`<br>`registry_only`: `rank_scale` |
| `scaling_scope` | - | `enum` | `fixed` | operational=2, registry_only=3 | `operational`: `columnwise`, `global_train_only`<br>`registry_only`: `datewise_cross_sectional`, `groupwise`, `categorywise` |
| `separation_rule` | - | `enum` | `fixed` | operational=1, registry_only=4 | `operational`: `strict_separation`<br>`registry_only`: `shared_transform_then_split`, `X_only_transform`, `target_only_transform`, `joint_preprocessor` |
| `structural_break_segmentation` | - | `enum` | `fixed` | operational=3 | `operational`: `none`, `pre_post_crisis`, `pre_post_covid` |
| `target_domain` | - | `enum` | `fixed` | operational=1, registry_only=2, future=2 | `operational`: `unconstrained`<br>`registry_only`: `nonnegative`, `bounded_0_1`<br>`future`: `integer_count`, `probability_target` |
| `target_lag_block` | feature_representation | `enum` | `fixed` | operational=2, registry_only=3 | `operational`: `none`, `fixed_target_lags`<br>`registry_only`: `ic_selected_target_lags`, `horizon_specific_target_lags`, `custom_target_lags` |
| `target_lag_selection` | feature_representation | `enum` | `fixed` | operational=2, registry_only=4 | `operational`: `none`, `fixed`<br>`registry_only`: `ic_select`, `cv_select`, `horizon_specific`, `custom` |
| `target_missing_policy` | - | `enum` | `fixed` | operational=1, external_plugin=1, registry_only=2 | `operational`: `none`<br>`external_plugin`: `custom`<br>`registry_only`: `drop`, `em_impute` |
| `target_normalization` | - | `enum` | `fixed` | operational=5 | `operational`: `none`, `zscore_train_only`, `robust_zscore`, `minmax`, `unit_variance` |
| `target_outlier_policy` | - | `enum` | `fixed` | operational=1, external_plugin=1, registry_only=2 | `operational`: `none`<br>`external_plugin`: `custom`<br>`registry_only`: `clip`, `outlier_to_nan` |
| `target_transform` | - | `enum` | `fixed` | operational=5 | `operational`: `level`, `difference`, `log`, `log_difference`, `growth_rate` |
| `target_transform_policy` | preprocessing | `enum` | `fixed` | operational=2, external_plugin=1 | `operational`: `raw_level`, `official_tcode_transformed`<br>`external_plugin`: `custom_target_transform` |
| `target_transformer` | preprocessing | `plugin` | `fixed` | operational=1 | `operational`: `none` |
| `tcode_application_scope` | - | `enum` | `fixed` | operational=4 | `operational`: `target_only`, `predictors_only`, `target_and_predictors`, `none` |
| `tcode_policy` | preprocessing | `enum` | `fixed` | operational=4, external_plugin=1, registry_only=1 | `operational`: `raw_only`, `official_tcode_only`, `official_tcode_then_extra_preprocess`, `extra_preprocess_only`<br>`external_plugin`: `custom_transform_sequence`<br>`registry_only`: `extra_preprocess_then_official_tcode` |
| `temporal_feature_block` | feature_representation | `enum` | `fixed` | operational=5, registry_only=1 | `operational`: `none`, `moving_average_features`, `rolling_moments`, `local_temporal_factors`, `volatility_features`<br>`registry_only`: `custom_temporal_features` |
| `x_lag_creation` | - | `enum` | `fixed` | operational=2, registry_only=3 | `operational`: `no_predictor_lags`, `fixed_predictor_lags`<br>`registry_only`: `cv_selected_predictor_lags`, `variable_specific_predictor_lags`, `category_specific_predictor_lags` |
| `x_lag_feature_block` | feature_representation | `enum` | `fixed` | operational=2, registry_only=4 | `operational`: `none`, `fixed_predictor_lags`<br>`registry_only`: `variable_specific_predictor_lags`, `category_specific_predictor_lags`, `cv_selected_predictor_lags`, `custom_predictor_lags` |
| `x_missing_policy` | - | `enum` | `fixed` | operational=10, external_plugin=1, registry_only=1 | `operational`: `none`, `em_impute`, `mean_impute`, `median_impute`, `ffill`, `interpolate_linear`, `drop_rows`, `drop_columns`, `drop_if_above_threshold`, `missing_indicator`<br>`external_plugin`: `custom`<br>`registry_only`: `drop` |
| `x_outlier_policy` | - | `enum` | `fixed` | operational=7, external_plugin=1, registry_only=2 | `operational`: `none`, `winsorize`, `trim`, `iqr_clip`, `mad_clip`, `zscore_clip`, `outlier_to_missing`<br>`external_plugin`: `custom`<br>`registry_only`: `clip`, `outlier_to_nan` |
| `x_transform_policy` | preprocessing | `enum` | `fixed` | operational=2, external_plugin=1 | `operational`: `raw_level`, `official_tcode_transformed`<br>`external_plugin`: `custom_x_transform` |

### Layer 3: forecast generation and runtime discipline

| Axis | Component | Type | Default policy | Value status counts | Values by status |
|---|---|---|---|---|---|
| `alignment_fairness` | - | `enum` | `fixed` | operational=3, registry_only=2 | `operational`: `same_split_across_models`, `same_split_across_targets`, `same_split_across_horizons`<br>`registry_only`: `model_specific_split_allowed`, `target_specific_split_allowed` |
| `benchmark_family` | benchmark | `enum` | `fixed` | operational=12 | `operational`: `historical_mean`, `autoregressive_bic`, `zero_change`, `custom_benchmark`, `rolling_mean`, `autoregressive_fixed_lag`, `autoregressive_diffusion_index`, `factor_model_benchmark`, `expert_benchmark`, `paper_specific_benchmark`, `survey_forecast`, `benchmark_suite` |
| `cache_policy` | - | `enum` | `fixed` | operational=3, registry_only=2 | `operational`: `no_cache`, `data_cache`, `feature_cache`<br>`registry_only`: `fold_cache`, `prediction_cache` |
| `checkpointing` | - | `enum` | `fixed` | operational=3, registry_only=2 | `operational`: `none`, `per_model`, `per_horizon`<br>`registry_only`: `per_date`, `per_trial` |
| `convergence_handling` | - | `enum` | `fixed` | operational=2, registry_only=1, future=1 | `operational`: `fallback_to_safe_hp`, `mark_fail`<br>`registry_only`: `retry_new_seed`<br>`future`: `clip_grad_and_retry` |
| `early_stopping` | - | `enum` | `fixed` | operational=3, registry_only=2 | `operational`: `none`, `validation_patience`, `loss_plateau`<br>`registry_only`: `time_budget_stop`, `trial_pruning` |
| `embargo_gap` | - | `enum` | `fixed` | operational=3, registry_only=1, future=1 | `operational`: `none`, `fixed_gap`, `horizon_gap`<br>`registry_only`: `custom_gap`<br>`future`: `publication_gap` |
| `execution_backend` | - | `enum` | `fixed` | operational=2, future=3 | `operational`: `local_cpu`, `joblib`<br>`future`: `local_gpu`, `ray`, `dask` |
| `forecast_object` | - | `enum` | `fixed` | operational=6 | `operational`: `point_mean`, `point_median`, `quantile`, `direction`, `interval`, `density` |
| `forecast_type` | - | `enum` | `fixed` | operational=2 | `operational`: `direct`, `iterated` |
| `framework` | - | `enum` | `fixed` | operational=2 | `operational`: `expanding`, `rolling` |
| `horizon_modelization` | - | `enum` | `fixed` | operational=2, registry_only=2, future=1 | `operational`: `separate_model_per_h`, `recursive_one_step_model`<br>`registry_only`: `shared_model_multi_h`, `hybrid_h_specific`<br>`future`: `shared_backbone_multi_head` |
| `hp_space_style` | - | `enum` | `fixed` | operational=4, registry_only=1 | `operational`: `discrete_grid`, `continuous_box`, `log_uniform`, `categorical`<br>`registry_only`: `conditional_space` |
| `logging_level` | - | `enum` | `fixed` | operational=3, registry_only=1 | `operational`: `silent`, `info`, `debug`<br>`registry_only`: `trace` |
| `lookback` | - | `enum` | `fixed` | operational=2, registry_only=2 | `operational`: `fixed_lookback`, `horizon_specific_lookback`<br>`registry_only`: `target_specific_lookback`, `cv_select_lookback` |
| `midasr_weight_family` | nonlinearity | `enum` | `fixed` | operational_narrow=5 | `operational_narrow`: `nealmon`, `almonp`, `nbeta`, `genexp`, `harstep` |
| `min_train_size` | - | `enum` | `fixed` | operational=5 | `operational`: `fixed_n_obs`, `fixed_years`, `model_specific_min_train`, `target_specific_min_train`, `horizon_specific_min_train` |
| `model_family` | nonlinearity | `enum` | `sweep` | operational=27, operational_narrow=3 | `operational`: `ar`, `ols`, `ridge`, `lasso`, `elasticnet`, `bayesian_ridge`, `huber`, `adaptive_lasso`, `svr_linear`, `svr_rbf`, `componentwise_boosting`, `boosting_ridge`, `boosting_lasso`, `pcr`, `pls`, `factor_augmented_linear`, `quantile_linear`, `random_forest`, `extra_trees`, `gradient_boosting`, `xgboost`, `lightgbm`, `catboost`, `mlp`, `lstm`, `gru`, `tcn`<br>`operational_narrow`: `midas_almon`, `midasr`, `midasr_nealmon` |
| `outer_window` | - | `enum` | `fixed` | operational=3, registry_only=2 | `operational`: `expanding`, `rolling`, `anchored_rolling`<br>`registry_only`: `hybrid_expanding_rolling`, `recursive_reestimation` |
| `refit_policy` | - | `enum` | `fixed` | operational=3, registry_only=1 | `operational`: `refit_every_step`, `refit_every_k_steps`, `fit_once_predict_many`<br>`registry_only`: `warm_start_refit` |
| `search_algorithm` | - | `enum` | `fixed` | operational=4 | `operational`: `grid_search`, `random_search`, `bayesian_optimization`, `genetic_algorithm` |
| `seed_policy` | - | `enum` | `fixed` | operational=3, registry_only=1 | `operational`: `fixed_seed`, `multi_seed_average`, `deterministic_only`<br>`registry_only`: `seed_sweep` |
| `sequence_framework` | - | `enum` | `fixed` | operational=1, future=2 | `operational`: `not_sequence`<br>`future`: `fixed_lookback_sequence`, `variable_lookback_sequence` |
| `shuffle_rule` | - | `enum` | `fixed` | operational=2, registry_only=2 | `operational`: `no_shuffle`, `forbidden_for_time_series`<br>`registry_only`: `restricted_shuffle_for_iid_only`, `groupwise_shuffle` |
| `split_family` | - | `enum` | `fixed` | operational=5 | `operational`: `simple_holdout`, `time_split`, `blocked_kfold`, `expanding_cv`, `rolling_cv` |
| `training_start_rule` | - | `enum` | `fixed` | operational=2 | `operational`: `earliest_possible`, `fixed_start` |
| `tuning_budget` | - | `enum` | `fixed` | operational=3, registry_only=1, future=1 | `operational`: `max_trials`, `max_time`, `early_stop_trials`<br>`registry_only`: `max_models`<br>`future`: `max_epochs` |
| `tuning_objective` | - | `enum` | `fixed` | operational=3, registry_only=1, future=1 | `operational`: `validation_mse`, `validation_rmse`, `validation_mae`<br>`registry_only`: `validation_mape`<br>`future`: `validation_quantile_loss` |
| `validation_location` | - | `enum` | `fixed` | operational=4, registry_only=1 | `operational`: `last_block`, `rolling_blocks`, `expanding_validation`, `blocked_cv`<br>`registry_only`: `nested_time_cv` |
| `validation_size_rule` | - | `enum` | `fixed` | operational=3, registry_only=2 | `operational`: `ratio`, `fixed_n`, `fixed_years`<br>`registry_only`: `fixed_dates`, `horizon_specific_n` |
| `y_lag_count` | - | `enum` | `fixed` | operational=3, registry_only=1 | `operational`: `fixed`, `cv_select`, `IC_select`<br>`registry_only`: `model_specific` |

### Layer 4: evaluation

| Axis | Component | Type | Default policy | Value status counts | Values by status |
|---|---|---|---|---|---|
| `agg_horizon` | - | `enum` | `fixed` | operational=2, registry_only=2 | `operational`: `equal_weight`, `report_separately_only`<br>`registry_only`: `short_horizon_weighted`, `long_horizon_weighted` |
| `agg_target` | - | `enum` | `fixed` | operational=2, registry_only=1, future=1 | `operational`: `equal_weight`, `report_separately_only`<br>`registry_only`: `scale_adjusted_weighting`<br>`future`: `economic_priority_weighting` |
| `agg_time` | - | `enum` | `fixed` | operational=4, registry_only=1 | `operational`: `full_out_of_sample_average`, `rolling_average`, `regime_subsample_average`, `pre_post_break_average`<br>`registry_only`: `event_window_average` |
| `benchmark_scope` | - | `enum` | `fixed` | operational=3, registry_only=1 | `operational`: `same_for_all`, `target_specific`, `horizon_specific`<br>`registry_only`: `target_horizon_specific` |
| `benchmark_window` | - | `enum` | `fixed` | operational=3, registry_only=1 | `operational`: `expanding`, `rolling`, `fixed`<br>`registry_only`: `paper_exact_window` |
| `decomposition_order` | - | `enum` | `fixed` | operational=1, registry_only=1, future=3 | `operational`: `marginal_effect_only`<br>`registry_only`: `two_way_interaction`<br>`future`: `three_way_interaction`, `full_factorial`, `shapley_style_effect_decomp` |
| `decomposition_target` | - | `enum` | `fixed` | operational=4, registry_only=5 | `operational`: `preprocessing_effect`, `feature_representation_effect`, `feature_builder_effect`, `benchmark_effect`<br>`registry_only`: `nonlinearity_effect`, `regularization_effect`, `validation_scheme_effect`, `loss_function_effect`, `importance_method_effect` |
| `density_metrics` | - | `enum` | `fixed` | registry_only=5, future=3 | `registry_only`: `pinball_loss`, `crps`, `interval_score`, `coverage_rate`, `winkler_score`<br>`future`: `log_score`, `nll`, `pit_based_metric` |
| `direction_metrics` | - | `enum` | `fixed` | operational=2, registry_only=7 | `operational`: `directional_accuracy`, `sign_accuracy`<br>`registry_only`: `turning_point_accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`, `auc`, `brier_score` |
| `economic_metrics` | - | `enum` | `fixed` | future=5 | `future`: `utility_gain`, `certainty_equivalent`, `cost_sensitive_loss`, `policy_loss`, `turning_point_value` |
| `oos_period` | - | `enum` | `fixed` | operational=3 | `operational`: `all_oos_data`, `recession_only_oos`, `expansion_only_oos` |
| `point_metrics` | - | `enum` | `fixed` | operational=5, registry_only=7 | `operational`: `mse`, `msfe`, `rmse`, `mae`, `mape`<br>`registry_only`: `smape`, `mase`, `rmsse`, `median_absolute_error`, `huber_loss`, `qlike`, `theil_u` |
| `primary_metric` | - | `enum` | `fixed` | operational=7 | `operational`: `msfe`, `relative_msfe`, `oos_r2`, `csfe`, `rmse`, `mae`, `mape` |
| `ranking` | - | `enum` | `fixed` | operational=5, registry_only=1, future=1 | `operational`: `mean_metric_rank`, `median_metric_rank`, `win_count`, `benchmark_beat_frequency`, `mcs_inclusion_priority`<br>`registry_only`: `stability_weighted_rank`<br>`future`: `ensemble_selection_rank` |
| `regime_definition` | - | `enum` | `fixed` | operational=3, registry_only=3, future=2 | `operational`: `none`, `nber_recession`, `user_defined_regime`<br>`registry_only`: `quantile_uncertainty`, `financial_stress`, `volatility_regime`<br>`future`: `markov_switching_regime`, `clustering_regime` |
| `regime_metrics` | - | `enum` | `fixed` | operational=3, registry_only=1 | `operational`: `all_main_metrics_by_regime`, `crisis_period_gain`, `state_dependent_oos_r2`<br>`registry_only`: `regime_transition_performance` |
| `regime_use` | - | `enum` | `fixed` | operational=1, registry_only=2, future=2 | `operational`: `evaluation_only`<br>`registry_only`: `train_only`, `train_and_eval`<br>`future`: `regime_specific_model`, `regime_interaction_features` |
| `relative_metrics` | - | `enum` | `fixed` | operational=6 | `operational`: `relative_msfe`, `relative_rmse`, `relative_mae`, `oos_r2`, `benchmark_win_rate`, `csfe_difference` |
| `report_style` | - | `enum` | `fixed` | operational=3, registry_only=2 | `operational`: `tidy_dataframe`, `latex_table`, `markdown_table`<br>`registry_only`: `plot_dashboard`, `paper_ready_bundle` |

### Layer 5: outputs and provenance

| Axis | Component | Type | Default policy | Value status counts | Values by status |
|---|---|---|---|---|---|
| `artifact_granularity` | - | `enum` | `fixed` | operational=1, registry_only=1, future=2 | `operational`: `aggregated`<br>`registry_only`: `per_target`<br>`future`: `per_target_horizon`, `hierarchical` |
| `export_format` | - | `enum` | `fixed` | operational=5 | `operational`: `json`, `csv`, `parquet`, `json_csv`, `all` |
| `provenance_fields` | - | `enum` | `fixed` | operational=4 | `operational`: `none`, `minimal`, `standard`, `full` |
| `saved_objects` | - | `enum` | `fixed` | operational=3, registry_only=1, future=2 | `operational`: `predictions_only`, `predictions_and_metrics`, `full_bundle`<br>`registry_only`: `none`<br>`future`: `models_only`, `data_only` |

### Layer 6: statistical tests

| Axis | Component | Type | Default policy | Value status counts | Values by status |
|---|---|---|---|---|---|
| `cpa_instability` | - | `enum` | `fixed` | operational=7 | `operational`: `none`, `cpa`, `rossi`, `rolling_dm`, `fluctuation_test`, `chow_break_forecast`, `cusum_on_loss` |
| `density_interval` | - | `enum` | `fixed` | operational=8 | `operational`: `none`, `pit_uniformity`, `berkowitz`, `kupiec`, `christoffersen_unconditional`, `christoffersen_independence`, `christoffersen_conditional`, `interval_coverage` |
| `dependence_correction` | - | `enum` | `fixed` | operational=4 | `operational`: `none`, `nw_hac`, `nw_hac_auto`, `block_bootstrap` |
| `direction` | - | `enum` | `fixed` | operational=5 | `operational`: `none`, `pesaran_timmermann`, `binomial_hit`, `mcnemar`, `roc_comparison` |
| `equal_predictive` | - | `enum` | `fixed` | operational=6 | `operational`: `none`, `dm`, `dm_hln`, `dm_modified`, `paired_t_on_loss_diff`, `wilcoxon_signed_rank` |
| `multiple_model` | - | `enum` | `fixed` | operational=6 | `operational`: `none`, `reality_check`, `spa`, `mcs`, `stepwise_mcs`, `bootstrap_best_model` |
| `nested` | - | `enum` | `fixed` | operational=6 | `operational`: `none`, `cw`, `enc_new`, `mse_f`, `mse_t`, `forecast_encompassing_nested` |
| `overlap_handling` | - | `enum` | `fixed` | operational=2 | `operational`: `allow_overlap`, `evaluate_with_hac` |
| `residual_diagnostics` | - | `enum` | `fixed` | operational=8 | `operational`: `none`, `mincer_zarnowitz`, `ljung_box`, `arch_lm`, `bias_test`, `full_residual_diagnostics`, `autocorrelation_of_errors`, `serial_dependence_loss_diff` |
| `stat_test` | - | `enum` | `fixed` | operational=21 | `operational`: `none`, `dm`, `dm_hln`, `dm_modified`, `cw`, `mcs`, `enc_new`, `mse_f`, `mse_t`, `cpa`, `rossi`, `rolling_dm`, `reality_check`, `spa`, `mincer_zarnowitz`, `ljung_box`, `arch_lm`, `bias_test`, `pesaran_timmermann`, `binomial_hit`, `full_residual_diagnostics` |
| `test_scope` | - | `enum` | `fixed` | operational=1, registry_only=3, future=3 | `operational`: `per_target`<br>`registry_only`: `per_horizon`, `per_model_pair`, `benchmark_vs_all`<br>`future`: `full_grid_pairwise`, `regime_specific_tests`, `subsample_tests` |

### Layer 7: interpretation and importance

| Axis | Component | Type | Default policy | Value status counts | Values by status |
|---|---|---|---|---|---|
| `importance_aggregation` | - | `enum` | `fixed` | operational=1, registry_only=2 | `operational`: `mean_abs`<br>`registry_only`: `mean_signed`, `top_k` |
| `importance_gradient_path` | - | `enum` | `fixed` | operational=1, registry_only=1 | `operational`: `none`<br>`registry_only`: `coefficient_path` |
| `importance_grouped` | - | `enum` | `fixed` | operational=2, registry_only=1 | `operational`: `none`, `grouped_permutation`<br>`registry_only`: `variable_root_groups` |
| `importance_local_surrogate` | - | `enum` | `fixed` | operational=3 | `operational`: `none`, `lime`, `feature_ablation` |
| `importance_method` | importance | `enum` | `fixed` | operational=13 | `operational`: `none`, `minimal_importance`, `tree_shap`, `kernel_shap`, `linear_shap`, `permutation_importance`, `lime`, `feature_ablation`, `pdp`, `ice`, `ale`, `grouped_permutation`, `importance_stability` |
| `importance_model_agnostic` | - | `enum` | `fixed` | operational=4 | `operational`: `none`, `kernel_shap`, `permutation_importance`, `feature_ablation` |
| `importance_model_native` | - | `enum` | `fixed` | operational=4, registry_only=1 | `operational`: `none`, `minimal_importance`, `tree_shap`, `linear_shap`<br>`registry_only`: `feature_gain` |
| `importance_output_style` | - | `enum` | `fixed` | operational=1, registry_only=2 | `operational`: `ranked_table`<br>`registry_only`: `curve_bundle`, `nested_report` |
| `importance_partial_dependence` | - | `enum` | `fixed` | operational=4 | `operational`: `none`, `pdp`, `ice`, `ale` |
| `importance_scope` | - | `enum` | `fixed` | operational=2, registry_only=1 | `operational`: `global`, `local`<br>`registry_only`: `both` |
| `importance_shap` | - | `enum` | `fixed` | operational=4 | `operational`: `none`, `tree_shap`, `kernel_shap`, `linear_shap` |
| `importance_stability` | - | `enum` | `fixed` | operational=2, registry_only=2 | `operational`: `none`, `importance_stability`<br>`registry_only`: `bootstrap_rank_stability`, `seed_stability` |
| `importance_temporal` | - | `enum` | `fixed` | operational=1, registry_only=2 | `operational`: `static_snapshot`<br>`registry_only`: `time_average`, `rolling_path` |

## First Review Queue After This Census

1. Layer 0: verify whether `axis_type` should stay internal or become visible only in authoring docs.
2. Layer 1: verify whether lowered FRED-SD selector axes `state_selection` and `sd_variable_selection` need a Navigator expert toggle or should remain hidden behind group/API helpers.
3. Layer 2: audit the large set of registry axes outside the primary Navigator tree, especially custom feature-block templates, target transformer examples, and bridge-only compatibility axes.
4. Layer 3: audit training axes outside the primary Navigator tree, plus virtual route axes for raw-panel iterated future-X paths.
5. Layers 4-7: verify that public runtime artifacts cover every visible decision axis and that registry-only/future values stay disabled with clear reasons.
