
# Macrocast Full Option-Universe Registry Plan

Status: canonical expanded taxonomy + implementation registry
Date: 2026-04-15
Depends on: plan_04_14_1958.md, preprocessing-axis-governance.md, stage0-grammar-contract.md
Purpose: translate the complete choice space into per-stage registry files, schema fields, and status tracking

---

## Document structure

This document defines:
1. Every enumerated choice in the macrocast option universe
2. The registry file that will own each choice axis
3. The schema fields each registry entry must carry
4. The implementation status of each value (operational / registry_only / planned)
5. The A/B priority split (A = implement now, B = registry-only placeholder)

---

## Conventions

Status values:
- `operational` — executable in current runtime
- `registry_only` — representable in grammar, not yet executable
- `planned` — will be added to registry in near-term
- `future` — long-run, not scheduled
- `external_plugin` — requires adapter/plugin bridge

Priority:
- `A` — v1 operational target
- `B` — registry-only in v1

---

# Stage 0. Meta / Package Grammar

> This stage fixes package-wide execution language before any content layer is populated.
> Core invariant: **one path = one fully specified forecasting study**.

## 0.1 experiment_unit

Registry file: `macrocast/registry/stage0/experiment_unit.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `route_owner: Literal["single_run", "wrapper", "orchestrator", "replication"]`
- `requires_multi_target: bool`
- `requires_wrapper: bool`
- `status: SupportStatus`
- `priority: Literal["A", "B"]`

| id | route_owner | requires_multi_target | requires_wrapper | status | priority |
|----|-------------|----------------------|------------------|--------|----------|
| `single_target_single_model` | single_run | no | no | operational | A |
| `single_target_model_grid` | single_run | no | no | operational | A |
| `single_target_full_sweep` | wrapper | no | yes | operational | A |
| `multi_target_separate_runs` | wrapper | yes | yes | registry_only | A |
| `multi_target_shared_design` | wrapper | yes | yes | planned | A |
| `multi_output_joint_model` | single_run | yes | no | registry_only | B |
| `hierarchical_forecasting_run` | orchestrator | yes | yes | future | B |
| `panel_forecasting_run` | orchestrator | yes | yes | future | B |
| `state_space_run` | single_run | no | no | future | B |
| `replication_recipe` | replication | no | no | registry_only | A |
| `benchmark_suite` | orchestrator | no | yes | planned | A |
| `ablation_study` | wrapper | no | yes | planned | A |

## 0.2 axis_type

Registry file: `macrocast/registry/stage0/axis_type.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `affects_path_identity: bool`
- `affects_fairness: bool`
- `status: SupportStatus`

| id | affects_path_identity | affects_fairness | status | priority |
|----|----------------------|------------------|--------|----------|
| `fixed` | yes | yes | operational | A |
| `sweep` | yes | no | operational | A |
| `nested_sweep` | yes | no | planned | A |
| `conditional` | yes | depends | operational | A |
| `derived` | no | no | operational | A |
| `eval_only` | no | no | registry_only | A |
| `report_only` | no | no | registry_only | B |

## 0.3 registry_type

Registry file: `macrocast/registry/stage0/registry_type.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `validation_contract: str`
- `status: SupportStatus`

| id | validation_contract | status | priority |
|----|-------------------|--------|----------|
| `enum_registry` | value in allowed set | operational | A |
| `numeric_registry` | value in range/grid | operational | A |
| `callable_registry` | callable signature check | planned | A |
| `custom_plugin` | adapter interface check | planned | A |
| `user_defined_yaml` | schema validation | registry_only | B |
| `external_adapter` | bridge contract check | future | B |

## 0.4 reproducibility_mode

Registry file: `macrocast/registry/stage0/reproducibility_mode.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `seed_required: bool`
- `deterministic_guarantee: bool`
- `status: SupportStatus`

| id | seed_required | deterministic_guarantee | status | priority |
|----|--------------|------------------------|--------|----------|
| `strict_reproducible` | yes | yes | planned | A |
| `seeded_reproducible` | yes | best_effort | operational | A |
| `best_effort` | optional | no | operational | A |
| `exploratory` | no | no | registry_only | B |

## 0.5 failure_policy

Registry file: `macrocast/registry/stage0/failure_policy.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `halts_execution: bool`
- `saves_partial: bool`
- `status: SupportStatus`

| id | halts_execution | saves_partial | status | priority |
|----|----------------|---------------|--------|----------|
| `fail_fast` | yes | no | operational | A |
| `skip_failed_cell` | no | yes | planned | A |
| `skip_failed_model` | no | yes | planned | A |
| `retry_then_skip` | no | yes | registry_only | B |
| `fallback_to_default_hp` | no | yes | registry_only | B |
| `save_partial_results` | no | yes | planned | A |
| `warn_only` | no | yes | registry_only | B |
| `hard_error` | yes | no | operational | A |

## 0.6 compute_mode

Registry file: `macrocast/registry/stage0/compute_mode.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `backend: str`
- `requires_gpu: bool`
- `requires_cluster: bool`
- `status: SupportStatus`

| id | backend | requires_gpu | requires_cluster | status | priority |
|----|---------|-------------|-----------------|--------|----------|
| `serial` | local | no | no | operational | A |
| `parallel_by_model` | joblib/ray | no | no | planned | A |
| `parallel_by_horizon` | joblib/ray | no | no | planned | A |
| `parallel_by_oos_date` | joblib/ray | no | no | registry_only | B |
| `parallel_by_trial` | joblib/ray | no | no | registry_only | B |
| `gpu_single` | pytorch/jax | yes | no | future | B |
| `gpu_multi` | pytorch/jax | yes | no | future | B |
| `distributed_cluster` | ray/dask/slurm | no | yes | future | B |

---

# Stage 1. Data / Task Definition

> This stage fixes data semantics. Must not be mixed with preprocessing.

## 1.1 Data source / information set

### 1.1.1 data_domain

Registry file: `macrocast/registry/data/data_domain.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `primary_frequency: str`
- `typical_datasets: tuple[str, ...]`
- `status: SupportStatus`

| id | primary_frequency | status | priority |
|----|------------------|--------|----------|
| `macro` | monthly/quarterly | operational | A |
| `macro_finance` | monthly/daily | planned | A |
| `housing` | monthly | registry_only | B |
| `energy` | monthly/daily | registry_only | B |
| `labor` | monthly | registry_only | B |
| `regional` | monthly/quarterly | registry_only | B |
| `panel_macro` | mixed | future | B |
| `text_macro` | mixed | future | B |
| `mixed_domain` | mixed | future | B |

### 1.1.2 dataset_source

Registry file: `macrocast/registry/data/dataset_source.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `frequency: str`
- `format: str`
- `loader_module: str | None`
- `support_tier: Literal["stable", "provisional", "planned", "external"]`
- `status: SupportStatus`

| id | frequency | format | support_tier | status | priority |
|----|-----------|--------|-------------|--------|----------|
| `fred_md` | monthly | csv | stable | operational | A |
| `fred_qd` | quarterly | csv | stable | operational | A |
| `fred_sd` | monthly | xlsx | provisional | operational | A |
| `fred_api_custom` | mixed | api | planned | planned | A |
| `bea` | quarterly | api/csv | planned | registry_only | B |
| `bls` | monthly | api/csv | planned | registry_only | B |
| `census` | monthly | api/csv | planned | registry_only | B |
| `oecd` | mixed | api | planned | registry_only | B |
| `imf_ifs` | mixed | api | planned | registry_only | B |
| `ecb_sdw` | mixed | api | planned | registry_only | B |
| `bis` | quarterly | csv | planned | registry_only | B |
| `world_bank` | yearly | api | planned | registry_only | B |
| `wrds_macro_finance` | daily/monthly | sql | external | future | B |
| `survey_spf` | quarterly | csv | planned | registry_only | B |
| `blue_chip` | monthly | manual | external | future | B |
| `market_prices` | daily | api | external | future | B |
| `high_frequency_surprises` | event | csv | external | future | B |
| `google_trends` | daily/weekly | api | external | future | B |
| `news_text` | daily | custom | external | future | B |
| `climate_series` | monthly | csv | external | future | B |
| `satellite_proxy` | daily | custom | external | future | B |
| `custom_csv` | any | csv | external | planned | A |
| `custom_parquet` | any | parquet | external | planned | A |
| `custom_duckdb` | any | duckdb | external | future | B |
| `custom_sql` | any | sql | external | future | B |

### 1.1.3 frequency

Registry file: `macrocast/registry/data/frequency.py`

| id | status | priority |
|----|--------|----------|
| `daily` | registry_only | B |
| `weekly` | registry_only | B |
| `monthly` | operational | A |
| `quarterly` | operational | A |
| `yearly` | registry_only | B |
| `mixed_frequency` | future | B |

### 1.1.4 information_set_type

Registry file: `macrocast/registry/data/information_set.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `requires_vintage: bool`
- `requires_release_calendar: bool`
- `status: SupportStatus`

| id | requires_vintage | requires_release_calendar | status | priority |
|----|-----------------|--------------------------|--------|----------|
| `revised` | no | no | operational | A |
| `real_time_vintage` | yes | no | planned | A |
| `pseudo_oos_revised` | no | no | planned | A |
| `pseudo_oos_vintage_aware` | yes | no | registry_only | B |
| `release_calendar_aware` | yes | yes | future | B |
| `publication_lag_aware` | yes | yes | future | B |

### 1.1.5 vintage_policy

Registry file: `macrocast/registry/data/vintage_policy.py`

| id | status | priority |
|----|--------|----------|
| `latest_only` | operational | A |
| `single_vintage` | operational | A |
| `rolling_vintage` | planned | A |
| `all_vintages_available` | registry_only | B |
| `event_vintage_subset` | future | B |
| `vintage_range` | registry_only | B |

### 1.1.6 alignment_rule

Registry file: `macrocast/registry/data/alignment_rule.py`

| id | status | priority |
|----|--------|----------|
| `end_of_period` | operational | A |
| `average_within_period` | registry_only | B |
| `last_available` | planned | A |
| `first_available` | registry_only | B |
| `quarter_to_month_repeat` | registry_only | B |
| `month_to_quarter_average` | planned | A |
| `month_to_quarter_last` | planned | A |
| `ragged_edge_keep` | registry_only | B |
| `ragged_edge_fill` | registry_only | B |
| `calendar_strict` | registry_only | B |

### 1.1.7 release_lag_rule

Registry file: `macrocast/registry/data/release_lag.py`

| id | status | priority |
|----|--------|----------|
| `ignore_release_lag` | operational | A |
| `fixed_lag_all_series` | registry_only | B |
| `series_specific_lag` | future | B |
| `calendar_exact_lag` | future | B |
| `lag_conservative` | registry_only | B |
| `lag_aggressive` | registry_only | B |

### 1.1.8 missing_availability

Registry file: `macrocast/registry/data/missing_availability.py`

| id | status | priority |
|----|--------|----------|
| `complete_case_only` | planned | A |
| `available_case` | operational | A |
| `target_date_drop_if_missing` | planned | A |
| `x_impute_only` | operational | A |
| `real_time_missing_as_missing` | registry_only | B |
| `state_space_fill` | future | B |
| `factor_fill` | future | B |
| `em_fill` | operational | A |

### 1.1.9 variable_universe

Registry file: `macrocast/registry/data/variable_universe.py`

| id | status | priority |
|----|--------|----------|
| `all_variables` | operational | A |
| `preselected_core` | planned | A |
| `category_subset` | planned | A |
| `paper_replication_subset` | registry_only | A |
| `target_specific_subset` | registry_only | B |
| `expert_curated_subset` | registry_only | B |
| `stability_filtered_subset` | future | B |
| `correlation_screened_subset` | future | B |
| `feature_selection_dynamic_subset` | future | B |

## 1.2 Sample / split / horizon setup

### 1.2.1 training_start_rule

Registry file: `macrocast/registry/data/training_start.py`

| id | status | priority |
|----|--------|----------|
| `earliest_possible` | operational | A |
| `fixed_start` | planned | A |
| `post_warmup_start` | planned | A |
| `post_break_start` | registry_only | B |
| `rolling_train_start` | operational | A |

### 1.2.2 oos_period

Registry file: `macrocast/registry/data/oos_period.py`

| id | status | priority |
|----|--------|----------|
| `single_oos_block` | operational | A |
| `multiple_oos_blocks` | registry_only | B |
| `rolling_origin` | operational | A |
| `recession_only_oos` | planned | A |
| `expansion_only_oos` | planned | A |
| `event_window_oos` | registry_only | B |

### 1.2.3 min_train_size

Registry file: `macrocast/registry/data/min_train_size.py`

| id | status | priority |
|----|--------|----------|
| `fixed_n_obs` | operational | A |
| `fixed_years` | planned | A |
| `model_specific_min_train` | registry_only | B |
| `target_specific_min_train` | registry_only | B |
| `horizon_specific_min_train` | registry_only | B |

### 1.2.4 warmup_rule

Registry file: `macrocast/registry/data/warmup_rule.py`

| id | status | priority |
|----|--------|----------|
| `lags_only_warmup` | operational | A |
| `lags_and_factors_warmup` | planned | A |
| `sequence_warmup` | future | B |
| `transform_warmup` | planned | A |
| `indicator_warmup` | registry_only | B |

### 1.2.5 structural_break_segmentation

Registry file: `macrocast/registry/data/structural_break.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `pre_post_crisis` | planned | A |
| `pre_post_covid` | planned | A |
| `user_break_dates` | planned | A |
| `break_test_detected` | future | B |
| `rolling_break_adaptive` | future | B |

## 1.3 Forecast target definition

### 1.3.1 forecast_type

Registry file: `macrocast/registry/data/forecast_type.py`

| id | status | priority |
|----|--------|----------|
| `direct` | operational | A |
| `iterated` | planned | A |
| `dirrec` | registry_only | B |
| `mimo` | future | B |
| `multi_horizon_joint` | future | B |
| `recursive_state_space` | future | B |
| `seq2seq` | future | B |

### 1.3.2 forecast_object

Registry file: `macrocast/registry/data/forecast_object.py`

| id | status | priority |
|----|--------|----------|
| `point_mean` | operational | A |
| `point_median` | planned | A |
| `quantile` | planned | A |
| `interval` | registry_only | B |
| `density` | registry_only | B |
| `direction` | planned | A |
| `turning_point` | registry_only | B |
| `regime_probability` | future | B |
| `event_probability` | future | B |

### 1.3.3 horizon_target_construction

Registry file: `macrocast/registry/data/horizon_target.py`

| id | status | priority |
|----|--------|----------|
| `future_level_y_t_plus_h` | operational | A |
| `future_diff` | planned | A |
| `future_logdiff` | planned | A |
| `cumulative_growth_to_h` | planned | A |
| `average_growth_1_to_h` | registry_only | B |
| `annualized_growth_to_h` | planned | A |
| `realized_future_average` | registry_only | B |
| `future_sum` | registry_only | B |
| `future_volatility` | future | B |
| `future_drawdown` | future | B |
| `future_indicator` | registry_only | B |

### 1.3.4 overlap_handling

Registry file: `macrocast/registry/data/overlap_handling.py`

| id | status | priority |
|----|--------|----------|
| `allow_overlap` | operational | A |
| `evaluate_with_hac` | planned | A |
| `evaluate_with_block_bootstrap` | registry_only | B |
| `non_overlapping_subsample` | registry_only | B |
| `horizon_specific_subsample` | registry_only | B |

## 1.4 Target / predictor design

### 1.4.1 target_family

Registry file: `macrocast/registry/data/target_family.py`

| id | status | priority |
|----|--------|----------|
| `single_macro_series` | operational | A |
| `multiple_macro_series` | planned | A |
| `panel_target` | future | B |
| `state_target` | registry_only | B |
| `factor_target` | future | B |
| `latent_target` | future | B |
| `constructed_target` | registry_only | B |
| `classification_target` | registry_only | B |

### 1.4.2 predictor_family

Registry file: `macrocast/registry/data/predictor_family.py`

| id | status | priority |
|----|--------|----------|
| `target_lags_only` | operational | A |
| `all_macro_vars` | operational | A |
| `all_except_target` | planned | A |
| `category_based` | planned | A |
| `financial_only` | registry_only | B |
| `macro_plus_finance` | registry_only | B |
| `survey_plus_macro` | future | B |
| `text_plus_macro` | future | B |
| `factor_only` | planned | A |
| `latent_state_plus_lags` | future | B |
| `selected_sparse_set` | registry_only | B |
| `handpicked_set` | registry_only | B |

### 1.4.3 contemporaneous_x_rule

Registry file: `macrocast/registry/data/contemporaneous_x.py`

| id | status | priority |
|----|--------|----------|
| `allow_contemporaneous` | registry_only | B |
| `forbid_contemporaneous` | operational | A |
| `allow_if_available_in_real_time` | future | B |
| `lag_all_predictors_by_one` | planned | A |
| `series_specific_contemporaneous` | future | B |

### 1.4.4 own_target_lags

Registry file: `macrocast/registry/data/own_target_lags.py`

| id | status | priority |
|----|--------|----------|
| `include` | operational | A |
| `exclude` | planned | A |
| `cv_select_lags` | planned | A |
| `fixed_lag_count` | operational | A |
| `target_specific_lag_count` | registry_only | B |

### 1.4.5 deterministic_components

Registry file: `macrocast/registry/data/deterministic_components.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `constant_only` | operational | A |
| `linear_trend` | planned | A |
| `seasonal_dummies` | planned | A |
| `month_dummies` | planned | A |
| `quarter_dummies` | planned | A |
| `holiday_calendar` | future | B |
| `event_dummies` | registry_only | B |
| `break_dummies` | registry_only | B |

### 1.4.6 exogenous_block

Registry file: `macrocast/registry/data/exogenous_block.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `strict_exogenous_only` | registry_only | B |
| `endogenous_allowed` | operational | A |
| `instrumented_exogenous` | future | B |
| `policy_shock_block` | future | B |

## 1.5 Multi-target design

### 1.5.1 x_map_policy

Registry file: `macrocast/registry/data/x_map_policy.py`

| id | status | priority |
|----|--------|----------|
| `shared_X` | planned | A |
| `target_specific_X` | registry_only | B |
| `category_specific_X` | registry_only | B |
| `domain_specific_X` | future | B |
| `availability_specific_X` | future | B |
| `handcrafted_target_X` | future | B |
| `learned_target_X` | future | B |

### 1.5.2 target_to_target_inclusion

Registry file: `macrocast/registry/data/target_to_target.py`

| id | status | priority |
|----|--------|----------|
| `allow_other_targets_as_X` | registry_only | B |
| `forbid_other_targets_as_X` | planned | A |
| `allow_selected_targets_as_X` | registry_only | B |
| `Granger_style_lagged_targets_only` | future | B |
| `system_wide_joint_model` | future | B |

### 1.5.3 multi_target_architecture

Registry file: `macrocast/registry/data/multi_target_architecture.py`

| id | status | priority |
|----|--------|----------|
| `separate_univariate_runs` | planned | A |
| `same_design_different_targets` | planned | A |
| `joint_multivariate_model` | future | B |
| `shared_encoder_multi_head` | future | B |
| `hierarchical_bottom_up` | future | B |
| `hierarchical_top_down` | future | B |
| `reconciliation_after_forecast` | future | B |

## 1.6 Task-level mandatory add-ons

### 1.6.1 evaluation_scale

Registry file: `macrocast/registry/data/evaluation_scale.py`

| id | status | priority |
|----|--------|----------|
| `transformed_scale` | planned | A |
| `original_scale` | operational | A |
| `both` | planned | A |

### 1.6.2 benchmark_family

Registry file: `macrocast/registry/data/benchmark_family.py`

| id | status | priority |
|----|--------|----------|
| `historical_mean` | operational | A |
| `rolling_mean` | planned | A |
| `random_walk` | planned | A |
| `ar_bic` | operational | A |
| `ar_fixed_p` | planned | A |
| `ardi` | registry_only | B |
| `factor_model` | registry_only | B |
| `var` | future | B |
| `expert_benchmark` | future | B |
| `paper_specific_benchmark` | registry_only | B |

### 1.6.3 regime_conditional_task

Registry file: `macrocast/registry/data/regime_task.py`

| id | status | priority |
|----|--------|----------|
| `unconditional` | operational | A |
| `recession_conditioned` | planned | A |
| `expansion_conditioned` | planned | A |
| `high_uncertainty_conditioned` | registry_only | B |
| `state_dependent_train` | future | B |
| `state_dependent_eval` | future | B |

---

# Stage 2. Preprocessing

> Critical governance stage. t-code representation vs extra preprocessing must be explicitly separated.

## 2.0 Preprocessing global governance

### 2.0.1 separation_rule

Registry file: `macrocast/registry/preprocessing/separation_rule.py`

| id | status | priority |
|----|--------|----------|
| `strict_separation` | operational | A |
| `shared_transform_then_split` | registry_only | B |
| `joint_preprocessor` | future | B |
| `target_only_transform` | planned | A |
| `X_only_transform` | planned | A |

### 2.0.2 preprocessing_fit_scope

Registry file: `macrocast/registry/preprocessing/fit_scope.py`

| id | status | priority |
|----|--------|----------|
| `fit_on_train_only` | operational | A |
| `fit_on_train_expand` | planned | A |
| `fit_on_train_roll` | planned | A |
| `fit_on_full_sample_forbidden` | operational | A |
| `leakage_checked` | planned | A |

### 2.0.3 representation_policy (GOVERNANCE — mandatory)

Registry file: `macrocast/registry/preprocessing/representation_policy.py`

| id | description | status | priority |
|----|-------------|--------|----------|
| `raw_only` | no representation transform | operational | A |
| `tcode_only` | t-code transform only, no extra preprocess | planned | A |
| `custom_transform_only` | user-defined representation | registry_only | B |

### 2.0.4 tcode_policy (GOVERNANCE — mandatory)

Registry file: `macrocast/registry/preprocessing/tcode_policy.py`

| id | description | status | priority |
|----|-------------|--------|----------|
| `apply_tcode_to_target` | t-code on target only | planned | A |
| `apply_tcode_to_X` | t-code on predictors only | planned | A |
| `apply_tcode_to_both` | t-code on target and X | planned | A |
| `apply_tcode_to_none` | no t-code (raw representation) | operational | A |

### 2.0.5 tcode_extra_preprocessing_order (GOVERNANCE — mandatory)

Registry file: `macrocast/registry/preprocessing/tcode_order.py`

| id | description | status | priority |
|----|-------------|--------|----------|
| `tcode_then_extra` | representation transform first, then extra | planned | A |
| `extra_without_tcode` | extra preprocessing only, no representation transform | operational | A |
| `extra_then_tcode` | extra preprocessing first, then representation transform | planned | A |
| `task_specific_pipeline` | user-defined ordering | registry_only | B |

### 2.0.6 preprocessing_axis_role (GOVERNANCE — mandatory)

Registry file: `macrocast/registry/preprocessing/axis_role.py`

| id | description | status | priority |
|----|-------------|--------|----------|
| `fixed_preprocessing` | preprocessing is fixed for fair comparison | operational | A |
| `swept_preprocessing` | preprocessing is intentionally varied | planned | A |
| `ablation_preprocessing` | preprocessing is part of ablation study | planned | A |

## 2.1 Target preprocessing

### 2.1.1 target_missing

Registry file: `macrocast/registry/preprocessing/target_missing.py`

| id | status | priority |
|----|--------|----------|
| `drop_missing_target_rows` | operational | A |
| `drop_forecast_dates_if_target_missing` | planned | A |
| `linear_interpolate` | planned | A |
| `ffill` | planned | A |
| `bfill` | registry_only | B |
| `seasonal_interpolate` | registry_only | B |
| `kalman_smooth` | future | B |
| `model_based_impute` | future | B |
| `multiple_imputation` | future | B |
| `do_not_impute_target` | operational | A |

### 2.1.2 target_outlier

Registry file: `macrocast/registry/preprocessing/target_outlier.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `winsorize` | planned | A |
| `trim` | planned | A |
| `clip` | planned | A |
| `iqr_flag_to_missing` | planned | A |
| `mad_flag_to_missing` | planned | A |
| `robust_smoother` | registry_only | B |
| `manual_event_override` | registry_only | B |
| `tail_transform` | registry_only | B |

### 2.1.3 target_transform

Registry file: `macrocast/registry/preprocessing/target_transform.py`

| id | status | priority |
|----|--------|----------|
| `level` | operational | A |
| `difference` | planned | A |
| `log` | planned | A |
| `log_difference` | planned | A |
| `growth_rate` | planned | A |
| `annualized_growth` | planned | A |
| `yoy_growth` | planned | A |
| `qoq_saar` | registry_only | B |
| `standardized_target` | planned | A |
| `BoxCox` | registry_only | B |
| `YeoJohnson` | registry_only | B |
| `rank_normal` | registry_only | B |
| `sign_preserving_log` | registry_only | B |
| `binary_direction` | planned | A |
| `threshold_event_target` | registry_only | B |

### 2.1.4 transform_timing

Registry file: `macrocast/registry/preprocessing/transform_timing.py`

| id | status | priority |
|----|--------|----------|
| `transform_then_horizon_build` | operational | A |
| `horizon_build_then_transform` | registry_only | B |
| `task_specific` | registry_only | B |

### 2.1.5 inverse_transform

Registry file: `macrocast/registry/preprocessing/inverse_transform.py`

| id | status | priority |
|----|--------|----------|
| `required` | planned | A |
| `optional` | planned | A |
| `not_needed` | operational | A |
| `evaluate_both_scales` | planned | A |

### 2.1.6 target_normalization

Registry file: `macrocast/registry/preprocessing/target_normalization.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `zscore_train_only` | planned | A |
| `robust_zscore` | planned | A |
| `minmax` | registry_only | B |
| `unit_variance` | registry_only | B |
| `de_mean_only` | registry_only | B |
| `rolling_standardize` | future | B |

### 2.1.7 target_domain_restriction

Registry file: `macrocast/registry/preprocessing/target_domain.py`

| id | status | priority |
|----|--------|----------|
| `unconstrained` | operational | A |
| `nonnegative` | registry_only | B |
| `bounded_0_1` | registry_only | B |
| `integer_count` | future | B |
| `probability_target` | future | B |

### 2.1.8 target_class_handling

Registry file: `macrocast/registry/preprocessing/target_class.py`

| id | status | priority |
|----|--------|----------|
| `not_applicable` | operational | A |
| `class_weighting` | registry_only | B |
| `oversample` | future | B |
| `undersample` | future | B |
| `smote_time_aware` | future | B |
| `threshold_tuning` | registry_only | B |

## 2.2 X preprocessing

### 2.2.1 x_missing

Registry file: `macrocast/registry/preprocessing/x_missing.py`

| id | status | priority |
|----|--------|----------|
| `drop_rows` | planned | A |
| `drop_columns` | planned | A |
| `drop_if_above_missing_threshold` | planned | A |
| `mean_impute` | planned | A |
| `median_impute` | planned | A |
| `group_mean_impute` | registry_only | B |
| `timewise_mean_impute` | registry_only | B |
| `ffill` | planned | A |
| `bfill` | registry_only | B |
| `interpolate_linear` | planned | A |
| `interpolate_spline` | registry_only | B |
| `em_impute` | operational | A |
| `kalman_impute` | future | B |
| `factor_impute` | future | B |
| `knn_impute` | registry_only | B |
| `iterative_imputer` | registry_only | B |
| `multiple_imputation` | future | B |
| `missing_indicator_addition` | planned | A |
| `leave_as_missing_for_model` | registry_only | B |

### 2.2.2 x_outlier

Registry file: `macrocast/registry/preprocessing/x_outlier.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `winsorize` | planned | A |
| `trim` | planned | A |
| `iqr_clip` | planned | A |
| `mad_clip` | planned | A |
| `zscore_clip` | planned | A |
| `outlier_to_missing` | planned | A |
| `robust_scaler_only` | operational | A |
| `Huberize` | registry_only | B |
| `quantile_cap` | registry_only | B |

### 2.2.3 x_standardize_scale

Registry file: `macrocast/registry/preprocessing/x_scale.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `zscore` | operational | A |
| `robust_zscore` | operational | A |
| `demean_only` | planned | A |
| `unit_variance_only` | planned | A |
| `minmax` | planned | A |
| `rank_scale` | registry_only | B |
| `quantile_transform` | registry_only | B |
| `whitening` | future | B |
| `groupwise_standardize` | registry_only | B |
| `expanding_standardize` | registry_only | B |
| `rolling_standardize` | registry_only | B |

### 2.2.4 scaling_scope

Registry file: `macrocast/registry/preprocessing/scaling_scope.py`

| id | status | priority |
|----|--------|----------|
| `columnwise` | operational | A |
| `datewise_cross_sectional` | registry_only | B |
| `groupwise` | registry_only | B |
| `categorywise` | registry_only | B |
| `global_train_only` | operational | A |
| `train_window_only` | operational | A |

### 2.2.5 additional_preprocessing

Registry file: `macrocast/registry/preprocessing/additional.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `smoothing_ma` | registry_only | B |
| `ema` | registry_only | B |
| `hp_filter` | planned | A |
| `bandpass_filter` | registry_only | B |
| `wavelet_denoise` | future | B |
| `seasonal_adjustment` | planned | A |
| `detrending` | planned | A |
| `deseasonalizing` | planned | A |
| `nonlinear_transform_bank` | future | B |
| `threshold_transform` | registry_only | B |
| `interaction_generation` | planned | A |
| `polynomial_expansion` | registry_only | B |
| `spline_basis` | registry_only | B |
| `kernel_features` | future | B |
| `text_embedding` | future | B |
| `autoencoder_embedding` | future | B |

### 2.2.6 x_lag_creation

Registry file: `macrocast/registry/preprocessing/x_lag.py`

| id | status | priority |
|----|--------|----------|
| `no_x_lags` | operational | A |
| `fixed_x_lags` | planned | A |
| `cv_selected_x_lags` | planned | A |
| `variable_specific_lags` | registry_only | B |
| `category_specific_lags` | registry_only | B |
| `distributed_lags` | future | B |
| `MIDAS_lags` | future | B |
| `Almon_lags` | future | B |

### 2.2.7 dimensionality_reduction

Registry file: `macrocast/registry/preprocessing/dim_reduction.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `PCA` | planned | A |
| `static_factor` | planned | A |
| `dynamic_factor` | future | B |
| `targeted_PCA` | registry_only | B |
| `sparse_PCA` | registry_only | B |
| `PLS` | registry_only | B |
| `PCR` | planned | A |
| `ICA` | registry_only | B |
| `autoencoder` | future | B |
| `supervised_encoder` | future | B |
| `random_projection` | registry_only | B |
| `feature_clustering` | registry_only | B |

### 2.2.8 feature_selection

Registry file: `macrocast/registry/preprocessing/feature_selection.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `correlation_screen` | planned | A |
| `mutual_information_screen` | registry_only | B |
| `univariate_F_test` | registry_only | B |
| `lasso_selection` | planned | A |
| `stability_selection` | registry_only | B |
| `recursive_feature_elimination` | registry_only | B |
| `tree_based_screen` | planned | A |
| `Boruta` | registry_only | B |
| `group_selection` | future | B |
| `economic_prior_selection` | future | B |

### 2.2.9 feature_grouping

Registry file: `macrocast/registry/preprocessing/feature_grouping.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `fred_category_group` | planned | A |
| `economic_theme_group` | registry_only | B |
| `lag_group` | planned | A |
| `factor_group` | registry_only | B |
| `text_group` | future | B |
| `spatial_group` | future | B |

## 2.3 Preprocessing order / recipe

### 2.3.1 recipe_mode

Registry file: `macrocast/registry/preprocessing/recipe_mode.py`

| id | status | priority |
|----|--------|----------|
| `fixed_recipe` | operational | A |
| `recipe_grid` | planned | A |
| `recipe_ablation` | planned | A |
| `paper_exact_recipe` | registry_only | A |
| `model_specific_recipe` | registry_only | B |
| `target_specific_recipe` | registry_only | B |

---

# Stage 3. Forecasting / Training

## 3.1 Framework

### 3.1.1 outer_window

Registry file: `macrocast/registry/training/outer_window.py`

| id | status | priority |
|----|--------|----------|
| `expanding` | operational | A |
| `rolling` | operational | A |
| `anchored_rolling` | planned | A |
| `hybrid_expanding_rolling` | registry_only | B |
| `recursive_reestimation` | registry_only | B |
| `event_retrain` | future | B |

### 3.1.2 refit_policy

Registry file: `macrocast/registry/training/refit_policy.py`

| id | status | priority |
|----|--------|----------|
| `refit_every_step` | operational | A |
| `refit_every_k_steps` | planned | A |
| `fit_once_predict_many` | planned | A |
| `warm_start_refit` | registry_only | B |
| `online_update` | future | B |
| `partial_fit` | future | B |

### 3.1.3 data_richness_mode

Registry file: `macrocast/registry/training/data_richness.py`

| id | status | priority |
|----|--------|----------|
| `target_lags_only` | operational | A |
| `factor_plus_lags` | planned | A |
| `full_high_dimensional_X` | operational | A |
| `selected_sparse_X` | planned | A |
| `mixed_mode` | registry_only | B |

### 3.1.4 sequence_framework

Registry file: `macrocast/registry/training/sequence_framework.py`

| id | status | priority |
|----|--------|----------|
| `not_sequence` | operational | A |
| `fixed_lookback_sequence` | future | B |
| `variable_lookback_sequence` | future | B |
| `multi_resolution_sequence` | future | B |
| `encoder_decoder_sequence` | future | B |

### 3.1.5 horizon_modelization

Registry file: `macrocast/registry/training/horizon_model.py`

| id | status | priority |
|----|--------|----------|
| `separate_model_per_h` | operational | A |
| `shared_model_multi_h` | registry_only | B |
| `shared_backbone_multi_head` | future | B |
| `recursive_one_step_model` | planned | A |
| `hybrid_h_specific` | registry_only | B |

## 3.2 Validation design

### 3.2.1 validation_size_rule

Registry file: `macrocast/registry/training/validation_size.py`

| id | status | priority |
|----|--------|----------|
| `ratio` | planned | A |
| `fixed_n` | planned | A |
| `fixed_years` | planned | A |
| `fixed_dates` | registry_only | B |
| `horizon_specific_n` | registry_only | B |
| `model_specific_n` | registry_only | B |

### 3.2.2 validation_location

Registry file: `macrocast/registry/training/validation_location.py`

| id | status | priority |
|----|--------|----------|
| `last_block` | planned | A |
| `rolling_blocks` | planned | A |
| `expanding_validation` | planned | A |
| `blocked_cv` | planned | A |
| `nested_time_cv` | registry_only | B |
| `walk_forward_validation` | planned | A |

### 3.2.3 embargo_gap

Registry file: `macrocast/registry/training/embargo.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `fixed_gap` | planned | A |
| `horizon_gap` | planned | A |
| `publication_gap` | future | B |
| `custom_gap` | registry_only | B |

## 3.3 Split method

### 3.3.1 split_family

Registry file: `macrocast/registry/training/split_family.py`

| id | status | priority |
|----|--------|----------|
| `simple_holdout` | operational | A |
| `time_split` | operational | A |
| `blocked_kfold` | planned | A |
| `expanding_cv` | planned | A |
| `rolling_cv` | planned | A |
| `nested_cv` | registry_only | B |
| `poos_cv` | registry_only | B |
| `BIC_selection` | operational | A |
| `AIC_selection` | planned | A |
| `IC_grid` | registry_only | B |
| `purged_time_cv` | registry_only | B |
| `combinatorial_purged_cv` | future | B |

### 3.3.2 shuffle_rule

Registry file: `macrocast/registry/training/shuffle_rule.py`

| id | status | priority |
|----|--------|----------|
| `no_shuffle` | operational | A |
| `restricted_shuffle_for_iid_only` | registry_only | B |
| `groupwise_shuffle` | registry_only | B |
| `forbidden_for_time_series` | operational | A |

### 3.3.3 alignment_fairness

Registry file: `macrocast/registry/training/alignment_fairness.py`

| id | status | priority |
|----|--------|----------|
| `same_split_across_models` | operational | A |
| `same_split_across_targets` | planned | A |
| `same_split_across_horizons` | planned | A |
| `model_specific_split_allowed` | registry_only | B |
| `target_specific_split_allowed` | registry_only | B |

## 3.4 Model layer

### 3.4.1 model_family

Registry file: `macrocast/registry/training/model_family.py`

Schema fields per entry:
- `id: str`
- `description: str`
- `category: Literal["naive_benchmark", "linear_ml", "kernel", "tree_ensemble", "neural", "panel_spatial", "probabilistic", "custom"]`
- `requires_gpu: bool`
- `supports_importance: tuple[str, ...]`
- `status: SupportStatus`

**Naive / benchmark models:**

| id | category | status | priority |
|----|----------|--------|----------|
| `historical_mean` | naive_benchmark | operational | A |
| `rolling_mean` | naive_benchmark | planned | A |
| `random_walk` | naive_benchmark | planned | A |
| `drift` | naive_benchmark | registry_only | B |
| `seasonal_naive` | naive_benchmark | registry_only | B |
| `AR` | naive_benchmark | operational | A |
| `ARDI` | naive_benchmark | registry_only | B |
| `factor_model` | naive_benchmark | planned | A |
| `VAR` | naive_benchmark | future | B |
| `BVAR` | naive_benchmark | future | B |
| `FAVAR` | naive_benchmark | future | B |
| `DFM` | naive_benchmark | future | B |
| `TVP_AR` | naive_benchmark | future | B |
| `MIDAS` | naive_benchmark | future | B |
| `U_MIDAS` | naive_benchmark | future | B |
| `ETS` | naive_benchmark | registry_only | B |
| `ARIMA` | naive_benchmark | registry_only | B |
| `SARIMA` | naive_benchmark | registry_only | B |
| `UnobservedComponents` | naive_benchmark | future | B |
| `state_space` | naive_benchmark | future | B |

**Linear ML / regularized:**

| id | category | status | priority |
|----|----------|--------|----------|
| `OLS` | linear_ml | planned | A |
| `Ridge` | linear_ml | operational | A |
| `Lasso` | linear_ml | operational | A |
| `AdaptiveLasso` | linear_ml | planned | A |
| `GroupLasso` | linear_ml | registry_only | B |
| `ElasticNet` | linear_ml | operational | A |
| `PCR` | linear_ml | planned | A |
| `PLS` | linear_ml | registry_only | B |
| `BayesianRidge` | linear_ml | planned | A |
| `HuberReg` | linear_ml | registry_only | B |
| `QuantileLinear` | linear_ml | registry_only | B |
| `SparseGroupLasso` | linear_ml | future | B |
| `TVP_Ridge` | linear_ml | future | B |
| `boosting` | linear_ml | registry_only | B |
| `factor_augmented_linear` | linear_ml | planned | A |

**Kernel / margin:**

| id | category | status | priority |
|----|----------|--------|----------|
| `KRR` | kernel | registry_only | B |
| `SVR_linear` | kernel | planned | A |
| `SVR_rbf` | kernel | planned | A |
| `SVR_poly` | kernel | registry_only | B |
| `GaussianProcess` | kernel | registry_only | B |
| `kernel_quantile_regression` | kernel | future | B |

**Tree / ensemble:**

| id | category | status | priority |
|----|----------|--------|----------|
| `RandomForest` | tree_ensemble | operational | A |
| `ExtraTrees` | tree_ensemble | planned | A |
| `GradientBoosting` | tree_ensemble | planned | A |
| `XGBoost` | tree_ensemble | planned | A |
| `LightGBM` | tree_ensemble | planned | A |
| `CatBoost` | tree_ensemble | registry_only | B |
| `AdaBoost` | tree_ensemble | registry_only | B |
| `bagging_regressor` | tree_ensemble | registry_only | B |
| `quantile_forest` | tree_ensemble | registry_only | B |
| `distributional_boosting` | tree_ensemble | future | B |

**Neural:**

| id | category | status | priority |
|----|----------|--------|----------|
| `MLP` | neural | planned | A |
| `DeepMLP` | neural | registry_only | B |
| `ResNet_tabular` | neural | registry_only | B |
| `LSTM` | neural | future | B |
| `GRU` | neural | future | B |
| `TCN` | neural | future | B |
| `Transformer_encoder` | neural | future | B |
| `Informer` | neural | future | B |
| `NBEATS` | neural | future | B |
| `NHITS` | neural | future | B |
| `TFT` | neural | future | B |
| `seq2seq_rnn` | neural | future | B |
| `mixture_of_experts` | neural | future | B |
| `foundation_model_adapter` | neural | future | B |

**Panel / spatial / hierarchical:**

| id | category | status | priority |
|----|----------|--------|----------|
| `panel_FE_forecast` | panel_spatial | future | B |
| `panel_RE_forecast` | panel_spatial | future | B |
| `dynamic_panel` | panel_spatial | future | B |
| `spatial_AR` | panel_spatial | future | B |
| `spatial_Durbin` | panel_spatial | future | B |
| `graph_neural_forecast` | panel_spatial | future | B |
| `hierarchical_reconciliation_model` | panel_spatial | future | B |
| `cross_state_factor_model` | panel_spatial | future | B |

**Probabilistic / quantile:**

| id | category | status | priority |
|----|----------|--------|----------|
| `quantile_RF` | probabilistic | registry_only | B |
| `quantile_GBM` | probabilistic | registry_only | B |
| `quantile_XGB` | probabilistic | registry_only | B |
| `quantile_LSTM` | probabilistic | future | B |
| `mixture_density_network` | probabilistic | future | B |
| `BayesianNN` | probabilistic | future | B |
| `distributional_regression` | probabilistic | future | B |
| `conformal_wrapper` | probabilistic | registry_only | B |

**Custom / plugin:**

| id | category | status | priority |
|----|----------|--------|----------|
| `sklearn_adapter` | custom | planned | A |
| `statsmodels_adapter` | custom | planned | A |
| `pytorch_adapter` | custom | future | B |
| `jax_adapter` | custom | future | B |
| `R_adapter` | custom | registry_only | B |
| `external_binary_adapter` | custom | future | B |

## 3.5 Tuning

### 3.5.1 search_algorithm

Registry file: `macrocast/registry/training/search_algorithm.py`

| id | status | priority |
|----|--------|----------|
| `grid_search` | planned | A |
| `random_search` | planned | A |
| `bayesian_optimization` | registry_only | B |
| `genetic_algorithm` | future | B |
| `evolutionary_search` | future | B |
| `hyperband` | registry_only | B |
| `asha` | registry_only | B |
| `successive_halving` | registry_only | B |
| `coordinate_search` | registry_only | B |
| `manual_fixed_hp` | operational | A |
| `paper_exact_hp` | registry_only | A |

### 3.5.2 tuning_objective

Registry file: `macrocast/registry/training/tuning_objective.py`

| id | status | priority |
|----|--------|----------|
| `validation_mse` | planned | A |
| `validation_rmse` | planned | A |
| `validation_mae` | planned | A |
| `validation_mape` | registry_only | B |
| `validation_quantile_loss` | future | B |
| `relative_msfe_to_benchmark` | registry_only | B |
| `oos_r2_proxy` | registry_only | B |
| `economic_utility` | future | B |
| `custom_loss` | registry_only | B |

### 3.5.3 tuning_budget

Registry file: `macrocast/registry/training/tuning_budget.py`

| id | status | priority |
|----|--------|----------|
| `max_trials` | planned | A |
| `max_time` | planned | A |
| `max_epochs` | future | B |
| `max_models` | registry_only | B |
| `early_stop_trials` | planned | A |

### 3.5.4 hp_space_style

Registry file: `macrocast/registry/training/hp_space.py`

| id | status | priority |
|----|--------|----------|
| `discrete_grid` | operational | A |
| `continuous_box` | planned | A |
| `log_uniform` | planned | A |
| `categorical` | operational | A |
| `conditional_space` | registry_only | B |
| `hierarchical_space` | future | B |

### 3.5.5 seed_policy

Registry file: `macrocast/registry/training/seed_policy.py`

| id | status | priority |
|----|--------|----------|
| `fixed_seed` | operational | A |
| `multi_seed_average` | planned | A |
| `seed_sweep` | registry_only | B |
| `deterministic_only` | operational | A |

### 3.5.6 early_stopping

Registry file: `macrocast/registry/training/early_stopping.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `validation_patience` | planned | A |
| `loss_plateau` | planned | A |
| `time_budget_stop` | registry_only | B |
| `trial_pruning` | registry_only | B |

### 3.5.7 convergence_handling

Registry file: `macrocast/registry/training/convergence.py`

| id | status | priority |
|----|--------|----------|
| `retry_same_hp` | registry_only | B |
| `retry_new_seed` | registry_only | B |
| `clip_grad_and_retry` | future | B |
| `fallback_to_safe_hp` | planned | A |
| `mark_fail` | operational | A |

## 3.6 Feature construction

### 3.6.1 feature_builder_type

Registry file: `macrocast/registry/training/feature_builder.py`

| id | status | priority |
|----|--------|----------|
| `AR_only` | operational | A |
| `factors_plus_AR` | planned | A |
| `raw_X_plus_AR` | operational | A |
| `raw_X_only` | operational | A |
| `sequence_tensor` | future | B |
| `grouped_features` | registry_only | B |
| `mixed_frequency_features` | future | B |
| `interaction_features` | planned | A |
| `calendar_augmented_features` | registry_only | B |

### 3.6.2 y_lag_count

Registry file: `macrocast/registry/training/y_lag_count.py`

| id | status | priority |
|----|--------|----------|
| `fixed` | operational | A |
| `cv_select` | planned | A |
| `IC_select` | operational | A |
| `model_specific` | registry_only | B |

### 3.6.3 factor_count

Registry file: `macrocast/registry/training/factor_count.py`

| id | status | priority |
|----|--------|----------|
| `fixed` | planned | A |
| `cv_select` | planned | A |
| `variance_explained_rule` | registry_only | B |
| `BaiNg_rule` | planned | A |
| `model_specific` | registry_only | B |

### 3.6.4 lookback

Registry file: `macrocast/registry/training/lookback.py`

| id | status | priority |
|----|--------|----------|
| `fixed_lookback` | operational | A |
| `horizon_specific_lookback` | planned | A |
| `target_specific_lookback` | registry_only | B |
| `cv_select_lookback` | registry_only | B |

## 3.7 Execution runtime

### 3.7.1 logging_level

Registry file: `macrocast/registry/training/logging_level.py`

| id | status | priority |
|----|--------|----------|
| `silent` | operational | A |
| `info` | operational | A |
| `debug` | planned | A |
| `trace` | registry_only | B |

### 3.7.2 checkpointing

Registry file: `macrocast/registry/training/checkpointing.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `per_model` | planned | A |
| `per_horizon` | planned | A |
| `per_date` | registry_only | B |
| `per_trial` | registry_only | B |

### 3.7.3 cache_policy

Registry file: `macrocast/registry/training/cache_policy.py`

| id | status | priority |
|----|--------|----------|
| `no_cache` | operational | A |
| `data_cache` | operational | A |
| `feature_cache` | planned | A |
| `fold_cache` | registry_only | B |
| `prediction_cache` | registry_only | B |

### 3.7.4 execution_backend

Registry file: `macrocast/registry/training/execution_backend.py`

| id | status | priority |
|----|--------|----------|
| `local_cpu` | operational | A |
| `local_gpu` | future | B |
| `ray` | future | B |
| `dask` | future | B |
| `joblib` | planned | A |
| `slurm` | future | B |

---

# Stage 4. Evaluation

## 4.1 Metrics

### 4.1.1 point_forecast_metrics

Registry file: `macrocast/registry/evaluation/point_metrics.py`

| id | status | priority |
|----|--------|----------|
| `MSE` | operational | A |
| `MSFE` | operational | A |
| `RMSE` | planned | A |
| `MAE` | planned | A |
| `MAPE` | planned | A |
| `sMAPE` | registry_only | B |
| `MASE` | registry_only | B |
| `RMSSE` | registry_only | B |
| `MedAE` | registry_only | B |
| `Huber_loss` | registry_only | B |
| `QLIKE` | registry_only | B |
| `TheilU` | registry_only | B |

### 4.1.2 relative_metrics

Registry file: `macrocast/registry/evaluation/relative_metrics.py`

| id | status | priority |
|----|--------|----------|
| `relative_MSFE` | operational | A |
| `relative_RMSE` | planned | A |
| `relative_MAE` | planned | A |
| `oos_R2` | operational | A |
| `benchmark_win_rate` | planned | A |
| `CSFE_difference` | operational | A |

### 4.1.3 direction_event_metrics

Registry file: `macrocast/registry/evaluation/direction_metrics.py`

| id | status | priority |
|----|--------|----------|
| `directional_accuracy` | planned | A |
| `sign_accuracy` | planned | A |
| `turning_point_accuracy` | registry_only | B |
| `precision` | registry_only | B |
| `recall` | registry_only | B |
| `F1` | registry_only | B |
| `balanced_accuracy` | registry_only | B |
| `AUC` | registry_only | B |
| `Brier_score` | registry_only | B |

### 4.1.4 quantile_interval_density_metrics

Registry file: `macrocast/registry/evaluation/density_metrics.py`

| id | status | priority |
|----|--------|----------|
| `pinball_loss` | registry_only | B |
| `CRPS` | registry_only | B |
| `interval_score` | registry_only | B |
| `coverage_rate` | registry_only | B |
| `winkler_score` | registry_only | B |
| `log_score` | future | B |
| `NLL` | future | B |
| `PIT_based_metric` | future | B |

### 4.1.5 economic_decision_metrics

Registry file: `macrocast/registry/evaluation/economic_metrics.py`

| id | status | priority |
|----|--------|----------|
| `utility_gain` | future | B |
| `certainty_equivalent` | future | B |
| `portfolio_SR_if_finance` | future | B |
| `cost_sensitive_loss` | future | B |
| `policy_loss` | future | B |
| `turning_point_value` | future | B |

## 4.2 Benchmark evaluation

### 4.2.1 benchmark_estimation_window

Registry file: `macrocast/registry/evaluation/benchmark_window.py`

| id | status | priority |
|----|--------|----------|
| `expanding` | operational | A |
| `rolling` | operational | A |
| `fixed` | planned | A |
| `paper_exact_window` | registry_only | A |

### 4.2.2 benchmark_by_target_horizon

Registry file: `macrocast/registry/evaluation/benchmark_scope.py`

| id | status | priority |
|----|--------|----------|
| `same_for_all` | operational | A |
| `target_specific` | planned | A |
| `horizon_specific` | planned | A |
| `target_horizon_specific` | registry_only | B |

## 4.3 Aggregation / reporting

### 4.3.1 aggregation_over_time

Registry file: `macrocast/registry/evaluation/agg_time.py`

| id | status | priority |
|----|--------|----------|
| `full_oos_average` | operational | A |
| `rolling_average` | planned | A |
| `regime_subsample_average` | planned | A |
| `event_window_average` | registry_only | B |
| `pre_post_break_average` | planned | A |

### 4.3.2 aggregation_over_horizons

Registry file: `macrocast/registry/evaluation/agg_horizon.py`

| id | status | priority |
|----|--------|----------|
| `equal_weight` | operational | A |
| `short_horizon_weighted` | registry_only | B |
| `long_horizon_weighted` | registry_only | B |
| `report_separately_only` | planned | A |

### 4.3.3 aggregation_over_targets

Registry file: `macrocast/registry/evaluation/agg_target.py`

| id | status | priority |
|----|--------|----------|
| `equal_weight` | planned | A |
| `scale_adjusted_weight` | registry_only | B |
| `economic_priority_weight` | future | B |
| `report_separately_only` | operational | A |

### 4.3.4 ranking_rule

Registry file: `macrocast/registry/evaluation/ranking.py`

| id | status | priority |
|----|--------|----------|
| `mean_metric_rank` | planned | A |
| `median_metric_rank` | planned | A |
| `win_count` | planned | A |
| `benchmark_beat_freq` | planned | A |
| `MCS_inclusion_priority` | planned | A |
| `stability_weighted_rank` | registry_only | B |
| `ensemble_selection_rank` | future | B |

### 4.3.5 report_style

Registry file: `macrocast/registry/evaluation/report_style.py`

| id | status | priority |
|----|--------|----------|
| `tidy_dataframe` | operational | A |
| `latex_table` | planned | A |
| `markdown_table` | planned | A |
| `plot_dashboard` | registry_only | B |
| `paper_ready_bundle` | registry_only | B |

## 4.4 Regime / conditional evaluation

### 4.4.1 regime_definition

Registry file: `macrocast/registry/evaluation/regime_definition.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `NBER_recession` | planned | A |
| `quantile_uncertainty` | registry_only | B |
| `financial_stress` | registry_only | B |
| `volatility_regime` | registry_only | B |
| `Markov_switching_regime` | future | B |
| `clustering_regime` | future | B |
| `user_defined_regime` | planned | A |

### 4.4.2 regime_use

Registry file: `macrocast/registry/evaluation/regime_use.py`

| id | status | priority |
|----|--------|----------|
| `eval_only` | planned | A |
| `train_only` | registry_only | B |
| `train_and_eval` | registry_only | B |
| `regime_specific_model` | future | B |
| `regime_interaction_features` | future | B |

### 4.4.3 regime_metrics

Registry file: `macrocast/registry/evaluation/regime_metrics.py`

| id | status | priority |
|----|--------|----------|
| `all_main_metrics_by_regime` | planned | A |
| `regime_transition_performance` | registry_only | B |
| `crisis_period_gain` | planned | A |
| `state_dependent_oos_r2` | planned | A |

## 4.5 Decomposition

### 4.5.1 decomposition_target

Registry file: `macrocast/registry/evaluation/decomposition_target.py`

| id | status | priority |
|----|--------|----------|
| `nonlinearity_effect` | registry_only | B |
| `regularization_effect` | registry_only | B |
| `cv_scheme_effect` | registry_only | B |
| `loss_function_effect` | registry_only | B |
| `preprocessing_effect` | planned | A |
| `feature_builder_effect` | planned | A |
| `benchmark_effect` | planned | A |
| `importance_method_effect` | registry_only | B |

### 4.5.2 decomposition_order

Registry file: `macrocast/registry/evaluation/decomposition_order.py`

| id | status | priority |
|----|--------|----------|
| `marginal_effect_only` | planned | A |
| `two_way_interaction` | registry_only | B |
| `three_way_interaction` | future | B |
| `full_factorial` | future | B |
| `Shapley_style_effect_decomp` | future | B |

---

# Stage 5. Output / Provenance

## 5.1 saved_objects

Registry file: `macrocast/registry/output/saved_objects.py`

| id | status | priority |
|----|--------|----------|
| `raw_predictions` | operational | A |
| `fold_predictions` | planned | A |
| `residuals` | planned | A |
| `loss_series` | operational | A |
| `selected_hp` | planned | A |
| `fitted_metadata` | planned | A |
| `feature_metadata` | planned | A |
| `importance_outputs` | operational | A |
| `test_outputs` | operational | A |
| `plots` | registry_only | B |
| `paper_tables` | registry_only | B |

## 5.2 provenance_fields

Registry file: `macrocast/registry/output/provenance_fields.py`

All provenance fields below must be written to manifest:

| field | status | priority |
|-------|--------|----------|
| `config_hash` | planned | A |
| `recipe_name` | operational | A |
| `dataset_version` | operational | A |
| `vintage_id` | operational | A |
| `sample_period` | operational | A |
| `seed` | planned | A |
| `git_commit` | planned | A |
| `package_version` | planned | A |
| `runtime_env` | planned | A |
| `failure_log` | planned | A |
| `tree_context` | operational | A |
| `fixed_axes` | operational | A |
| `sweep_axes` | operational | A |
| `preprocessing_contract` | operational | A |
| `evaluation_scale` | operational | A |
| `benchmark_family` | operational | A |

## 5.3 export_format

Registry file: `macrocast/registry/output/export_format.py`

| id | status | priority |
|----|--------|----------|
| `parquet` | planned | A |
| `csv` | planned | A |
| `json` | operational | A |
| `yaml` | operational | A |
| `pickle` | planned | A |
| `feather` | registry_only | B |
| `latex` | planned | A |
| `html_report` | registry_only | B |
| `pdf_report` | future | B |

## 5.4 artifact_granularity

Registry file: `macrocast/registry/output/artifact_granularity.py`

| id | status | priority |
|----|--------|----------|
| `one_file_per_run` | operational | A |
| `one_file_per_target` | planned | A |
| `one_file_per_model` | planned | A |
| `one_file_per_horizon` | planned | A |
| `one_file_per_layer_output` | registry_only | B |

---

# Stage 6. Statistical Tests

## 6.1 equal_predictive_ability

Registry file: `macrocast/registry/tests/equal_ability.py`

| id | status | priority |
|----|--------|----------|
| `DM` | operational | A |
| `DM_HLN_small_sample` | planned | A |
| `modified_DM` | planned | A |
| `paired_t_on_loss_diff` | planned | A |
| `Wilcoxon_signed_rank` | registry_only | B |

## 6.2 nested_model_tests

Registry file: `macrocast/registry/tests/nested_model.py`

| id | status | priority |
|----|--------|----------|
| `Clark_West` | operational | A |
| `ENC_NEW` | planned | A |
| `MSE_F` | planned | A |
| `MSE_t` | planned | A |
| `forecast_encompassing_nested` | registry_only | B |

## 6.3 conditional_predictive_ability

Registry file: `macrocast/registry/tests/conditional_ability.py`

| id | status | priority |
|----|--------|----------|
| `Giacomini_White_CPA` | planned | A |
| `Rossi_Sekhposyan_stability` | planned | A |
| `rolling_DM` | planned | A |
| `fluctuation_test` | registry_only | B |
| `Chow_break_forecast` | registry_only | B |
| `CUSUM_on_loss` | registry_only | B |

## 6.4 multiple_model_tests

Registry file: `macrocast/registry/tests/multiple_model.py`

| id | status | priority |
|----|--------|----------|
| `White_Reality_Check` | planned | A |
| `Hansen_SPA` | planned | A |
| `MCS` | planned | A |
| `stepwise_MCS` | registry_only | B |
| `bootstrap_best_model_test` | registry_only | B |

## 6.5 density_interval_tests

Registry file: `macrocast/registry/tests/density_interval.py`

| id | status | priority |
|----|--------|----------|
| `PIT_uniformity` | registry_only | B |
| `Berkowitz_test` | registry_only | B |
| `Kupiec_test` | registry_only | B |
| `Christoffersen_unconditional` | registry_only | B |
| `Christoffersen_independence` | registry_only | B |
| `Christoffersen_conditional` | registry_only | B |
| `interval_coverage_test` | registry_only | B |

## 6.6 direction_classification_tests

Registry file: `macrocast/registry/tests/direction_class.py`

| id | status | priority |
|----|--------|----------|
| `Pesaran_Timmermann` | planned | A |
| `McNemar` | registry_only | B |
| `binomial_hit_test` | planned | A |
| `ROC_comparison` | registry_only | B |

## 6.7 residual_calibration_diagnostics

Registry file: `macrocast/registry/tests/residual_calibration.py`

| id | status | priority |
|----|--------|----------|
| `Mincer_Zarnowitz` | planned | A |
| `autocorrelation_of_errors` | planned | A |
| `Ljung_Box_on_errors` | planned | A |
| `ARCH_LM_on_errors` | planned | A |
| `bias_test` | planned | A |
| `serial_dependence_loss_diff` | planned | A |

## 6.8 dependence_correction

Registry file: `macrocast/registry/tests/dependence_correction.py`

| id | status | priority |
|----|--------|----------|
| `iid` | operational | A |
| `Newey_West` | planned | A |
| `HAC_auto_bandwidth` | planned | A |
| `block_bootstrap` | planned | A |
| `stationary_bootstrap` | registry_only | B |
| `circular_bootstrap` | registry_only | B |
| `wild_bootstrap` | registry_only | B |
| `cluster_robust` | future | B |

## 6.9 test_scope

Registry file: `macrocast/registry/tests/test_scope.py`

| id | status | priority |
|----|--------|----------|
| `per_target` | operational | A |
| `per_horizon` | operational | A |
| `per_model_pair` | operational | A |
| `full_grid_pairwise` | planned | A |
| `benchmark_vs_all` | operational | A |
| `regime_specific_tests` | planned | A |
| `subsample_tests` | planned | A |

---

# Stage 7. Variable Importance / Interpretability

## 7.1 scope

Registry file: `macrocast/registry/importance/scope.py`

| id | status | priority |
|----|--------|----------|
| `none` | operational | A |
| `global` | operational | A |
| `local` | planned | A |
| `global_and_local` | planned | A |
| `time_varying` | registry_only | B |
| `regime_specific` | registry_only | B |
| `horizon_specific` | planned | A |
| `target_specific` | planned | A |
| `cross_model_consensus` | registry_only | B |

## 7.2 model_native_importance

Registry file: `macrocast/registry/importance/model_native.py`

| id | status | priority |
|----|--------|----------|
| `linear_coefficients` | operational | A |
| `standardized_coefficients` | planned | A |
| `t_stats_if_linear` | planned | A |
| `RF_Gini_importance` | operational | A |
| `RF_permutation_importance` | planned | A |
| `XGB_gain` | planned | A |
| `XGB_cover` | registry_only | B |
| `XGB_weight` | registry_only | B |
| `LGB_split_importance` | planned | A |
| `attention_weight_proxy` | future | B |
| `feature_dropout_score` | future | B |

## 7.3 model_agnostic_importance

Registry file: `macrocast/registry/importance/model_agnostic.py`

| id | status | priority |
|----|--------|----------|
| `PMI_or_PFI_permutation_importance` | planned | A |
| `leave_one_covariate_out` | planned | A |
| `group_permutation_importance` | planned | A |
| `conditional_permutation_importance` | registry_only | B |
| `Sobol_sensitivity` | future | B |
| `variance_based_sensitivity` | future | B |
| `Shapley_value_global` | registry_only | B |

## 7.4 SHAP family

Registry file: `macrocast/registry/importance/shap.py`

| id | status | priority |
|----|--------|----------|
| `TreeSHAP` | planned | A |
| `KernelSHAP` | planned | A |
| `DeepSHAP` | future | B |
| `LinearSHAP` | planned | A |
| `GroupedSHAP` | registry_only | B |
| `InteractionSHAP` | registry_only | B |
| `SHAP_time_average` | planned | A |
| `SHAP_regime_split` | registry_only | B |
| `SHAP_horizon_split` | planned | A |
| `SHAP_target_split` | registry_only | B |

## 7.5 gradient_path_methods

Registry file: `macrocast/registry/importance/gradient_path.py`

| id | status | priority |
|----|--------|----------|
| `IntegratedGradients` | future | B |
| `PathIntegratedGradients` | future | B |
| `GradientXInput` | future | B |
| `SmoothGrad` | future | B |
| `ExpectedGradients` | future | B |
| `DeepLift` | future | B |
| `LRP` | future | B |
| `saliency_map` | future | B |

## 7.6 local_surrogate_perturbation

Registry file: `macrocast/registry/importance/local_surrogate.py`

| id | status | priority |
|----|--------|----------|
| `LIME` | planned | A |
| `local_linear_surrogate` | registry_only | B |
| `counterfactual_explanation` | future | B |
| `occlusion_importance` | registry_only | B |
| `feature_ablation` | planned | A |
| `masking_importance` | future | B |

## 7.7 partial_dependence_style

Registry file: `macrocast/registry/importance/partial_dependence.py`

| id | status | priority |
|----|--------|----------|
| `PDP` | planned | A |
| `ICE` | planned | A |
| `ALE` | planned | A |
| `2D_PDP` | registry_only | B |
| `2D_ALE` | registry_only | B |
| `accumulated_local_effect_by_group` | future | B |

## 7.8 grouped_importance

Registry file: `macrocast/registry/importance/grouped.py`

| id | status | priority |
|----|--------|----------|
| `by_FRED_category` | planned | A |
| `by_economic_theme` | planned | A |
| `by_variable_family` | planned | A |
| `by_lag_block` | planned | A |
| `by_factor_block` | planned | A |
| `by_time_window` | planned | A |
| `by_regime` | registry_only | B |
| `by_target` | planned | A |
| `by_horizon` | planned | A |
| `by_state_or_region` | registry_only | B |
| `custom_group_map` | registry_only | B |

## 7.9 sequence_temporal_importance

Registry file: `macrocast/registry/importance/temporal.py`

| id | status | priority |
|----|--------|----------|
| `time_step_importance` | future | B |
| `feature_time_heatmap` | future | B |
| `attention_rollout` | future | B |
| `temporal_occlusion` | future | B |
| `temporal_IG` | future | B |
| `window_importance` | future | B |
| `lag_saliency_profile` | future | B |

## 7.10 stability_of_importance

Registry file: `macrocast/registry/importance/stability.py`

| id | status | priority |
|----|--------|----------|
| `bootstrap_rank_stability` | planned | A |
| `seed_stability` | planned | A |
| `window_stability` | planned | A |
| `vintage_stability` | registry_only | B |
| `model_consensus_importance` | planned | A |
| `rank_correlation_across_runs` | planned | A |
| `sign_consistency` | planned | A |

## 7.11 importance_aggregation

Registry file: `macrocast/registry/importance/aggregation.py`

| id | status | priority |
|----|--------|----------|
| `mean_abs_importance` | operational | A |
| `median_abs_importance` | planned | A |
| `signed_mean_importance` | planned | A |
| `rank_average` | planned | A |
| `top_k_frequency` | planned | A |
| `stability_weighted_rank` | registry_only | B |
| `group_share_of_total_importance` | planned | A |

## 7.12 output_style

Registry file: `macrocast/registry/importance/output_style.py`

| id | status | priority |
|----|--------|----------|
| `bar_plot` | planned | A |
| `heatmap` | planned | A |
| `waterfall` | registry_only | B |
| `beeswarm` | registry_only | B |
| `time_series_plot` | planned | A |
| `regime_comparison_plot` | registry_only | B |
| `category_stack_plot` | planned | A |
| `dashboard` | future | B |
| `paper_table` | planned | A |

---

# Implementation priority summary

## A-tier: v1 operational target

### Stage 0 — meta grammar
- experiment_unit: single_target_single_model, single_target_model_grid, single_target_full_sweep, replication_recipe, benchmark_suite, ablation_study
- axis_type: fixed, sweep, nested_sweep, conditional, derived, eval_only
- registry_type: enum_registry, numeric_registry, callable_registry, custom_plugin
- reproducibility_mode: seeded_reproducible, best_effort
- failure_policy: fail_fast, skip_failed_cell, skip_failed_model, save_partial_results, hard_error
- compute_mode: serial, parallel_by_model, parallel_by_horizon

### Stage 1 — data/task
- datasets: fred_md, fred_qd, fred_sd, custom_csv, custom_parquet
- info set: revised, real_time_vintage, pseudo_oos_revised
- forecast: direct, iterated, point_mean, point_median, quantile, direction
- benchmarks: historical_mean, rolling_mean, random_walk, ar_bic, ar_fixed_p

### Stage 2 — preprocessing
- governance: strict_separation, fit_on_train_only, raw_only/tcode_only, all tcode policies, all tcode orders
- x_missing: em_impute + mean/median/ffill/interpolate/drop variants
- x_scale: zscore, robust_zscore, minmax + scope variants
- dim reduction: PCA, static_factor, PCR
- feature selection: correlation_screen, lasso_selection, tree_based_screen

### Stage 3 — training
- frameworks: expanding, rolling, anchored_rolling
- models: AR, Ridge, Lasso, ElasticNet, RF + OLS, AdaptiveLasso, BayesianRidge, PCR, factor_augmented_linear, SVR, ExtraTrees, GBM, XGBoost, LightGBM, MLP
- tuning: grid_search, random_search, manual_fixed_hp
- feature builders: AR_only, factors_plus_AR, raw_X_plus_AR, interaction_features

### Stage 4 — evaluation
- metrics: MSFE, RMSE, MAE, MAPE, relative_MSFE, oos_R2, CSFE, directional_accuracy
- aggregation: time/horizon/target averaging, ranking rules
- regime: NBER recession, user-defined

### Stage 5 — output
- provenance: full manifest with tree_context, preprocessing_contract, all governance fields
- formats: json, yaml, parquet, csv, latex

### Stage 6 — tests
- equal ability: DM, DM_HLN, modified_DM, paired_t
- nested: Clark_West, ENC_NEW, MSE_F, MSE_t
- conditional: Giacomini_White, Rossi_Sekhposyan, rolling_DM
- multiple: White_RC, Hansen_SPA, MCS
- dependence: Newey_West, HAC, block_bootstrap

### Stage 7 — importance
- native: coefficients, t-stats, RF Gini/permutation, XGB_gain, LGB_split
- agnostic: PFI, leave-one-out, group permutation
- SHAP: TreeSHAP, KernelSHAP, LinearSHAP
- local: LIME, feature_ablation
- PD: PDP, ICE, ALE
- grouped: by FRED category, economic theme, lag block, factor block
- stability: bootstrap/seed/window stability, model consensus

## B-tier: registry-only in v1

All remaining values marked `registry_only` or `future` above.
Key B-tier domains:
- panel/hierarchical/state-space/foundation model
- density/conformal/multi-output
- OECD, IMF, ECB, BIS, news_text, satellite_proxy adapters
- full regime-specialized training architecture
- distributed compute (ray, dask, slurm, GPU)
- advanced gradient/path interpretability (IG, DeepLift, LRP)
- economic decision metrics

---

# Registry file inventory

Total registry files required: ~95

By stage:
- Stage 0: 6 files
- Stage 1: ~18 files
- Stage 2: ~22 files
- Stage 3: ~22 files
- Stage 4: ~17 files
- Stage 5: 4 files
- Stage 6: 9 files
- Stage 7: 12 files

Each file follows the same pattern:
```python
# macrocast/registry/{stage}/{axis}.py

from macrocast.registry.types import AxisRegistryEntry, SupportStatus

AXIS_ID = "{axis_name}"
AXIS_LAYER = "{layer_number}_{layer_name}"
AXIS_TYPE = "fixed"  # or "sweep", "conditional", etc.

ENTRIES: list[AxisRegistryEntry] = [
    AxisRegistryEntry(
        id="value_id",
        description="...",
        status=SupportStatus.OPERATIONAL,  # or REGISTRY_ONLY, PLANNED, FUTURE
        # ... axis-specific fields
    ),
    ...
]
```

---

# Recommended implementation order

1. **Grammar lock** — Stage 0 registry files + validation
2. **Data-task semantics lock** — Stage 1 registry files + existing raw layer integration
3. **Preprocessing governance lock** — Stage 2 governance fields (representation_policy, tcode_policy, tcode_order, axis_role) as mandatory contract fields
4. **Training/evaluation minimal slice** — Stage 3 + 4 A-tier operational entries
5. **Provenance/test/importance expansion** — Stage 5 + 6 + 7 A-tier entries
6. **B-tier population** — registry_only entries for future extensibility

