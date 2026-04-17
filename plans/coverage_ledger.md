# Coverage Ledger — Part 1 (Layer 0-2)

macrocast v1.0 → v1.1 → v2 coverage map for Meta / Data·Task / Preprocessing axes.
Generated against server1 registry snapshot at `~/project/macroforecast/macrocast/registry/{stage0,data,preprocessing}/` on 2026-04-17.

## Legend
- **Current status**: `operational` (wired and running) · `registry_only` (enum registered, no executor) · `planned` (registry marks planned, partial build) · `future` (registered as future) · `absent` (not in registry yet)
- **Target version**: `v1.0` · `v1.1` · `v2` · `post-v2` · `deferred-indef`
- **Target phase**: `phase-00`..`phase-11` | `-` (already done) | `(promotion)` = promoted in a later catalog pass
- Rationale column is in Korean / mixed as per project convention.

---

## Layer 0: Meta (~45 values)

### 0.1 experiment_unit

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| single_target_single_model | operational | - | - | 이미 완료 |
| single_target_model_grid | operational | - | - | 이미 완료 |
| single_target_full_sweep | operational | - | - | 이미 완료 |
| multi_target_separate_runs | registry_only | v1.0 | phase-01 | sweep infra 확장시 활성 |
| multi_target_shared_design | planned | v1.1 | phase-10 | shared design runner 필요 |
| multi_output_joint_model | registry_only | v1.1 | phase-10 | joint multivariate 합류 |
| hierarchical_forecasting_run | future | v2 | phase-11 | hierarchy reconciliation 필요 |
| panel_forecasting_run | future | v2 | phase-11 | panel data executor 필요 |
| state_space_run | future | v2 | phase-11 | SS framework 필요 |
| replication_recipe | registry_only | v1.0 | phase-01 | replication_override_study 와 정합 |
| benchmark_suite | planned | v1.0 | phase-04 | benchmark 축 정리와 동반 |
| ablation_study | planned | v1.0 | phase-01 | controlled_variation_study 와 병행 |

### 0.2 axis_type

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fixed | operational | - | - | 이미 완료 |
| sweep | operational | - | - | 이미 완료 |
| nested_sweep | planned | v1.0 | phase-01 | nested sweep 설계 반영 |
| conditional | operational | - | - | 이미 완료 |
| derived | operational | - | - | 이미 완료 |
| eval_only | registry_only | v1.0 | phase-01 | eval-only plumbing 활성 |
| report_only | registry_only | v1.1 | phase-10 | reporting 전용 플래그 승격 |

### 0.3 registry_type

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| enum_registry | operational | - | - | 이미 완료 |
| numeric_registry | operational | - | - | 이미 완료 |
| callable_registry | planned | v1.0 | phase-01 | callable 검증 활성 |
| custom_plugin | planned | v1.0 | phase-01 | plugin 경로 |
| user_defined_yaml | registry_only | v1.1 | phase-10 | YAML adapter 승격 |
| external_adapter | registry_only | v2 | phase-11 | 외부 어댑터는 후순위 |

### 0.4 reproducibility_mode

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| strict_reproducible | planned | v1.0 | phase-00 | Phase 0 핵심 |
| seeded_reproducible | operational | - | - | 이미 완료 |
| best_effort | operational | - | - | 이미 완료 |
| exploratory | registry_only | v1.0 | phase-00 | Phase 0 에서 공식화 |

### 0.5 failure_policy

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fail_fast | operational | - | - | 이미 완료 |
| skip_failed_cell | planned | v1.0 | phase-01 | sweep 안정화 필요 |
| skip_failed_model | operational | - | - | 이미 완료 |
| retry_then_skip | registry_only | v1.1 | phase-10 | 재시도 정책 승격 |
| fallback_to_default_hp | registry_only | v1.1 | phase-10 | HP fallback 승격 |
| save_partial_results | operational | - | - | 이미 완료 |
| warn_only | registry_only | v1.0 | phase-01 | 간단 승격 |
| hard_error | operational | - | - | 이미 완료 |

### 0.6 compute_mode

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| serial | operational | - | - | 이미 완료 |
| parallel_by_model | operational | - | - | 이미 완료 |
| parallel_by_horizon | operational | - | - | 이미 완료 |
| parallel_by_oos_date | registry_only | v1.1 | phase-10 | OOS 병렬화 승격 |
| parallel_by_trial | registry_only | v1.1 | phase-10 | HPO 연계 승격 |
| gpu_single | registry_only | v2 | phase-11 | NN 축 합류시 |
| gpu_multi | registry_only | v2 | phase-11 | NN 축 합류시 |
| distributed_cluster | registry_only | v2 | phase-11 | 분산 런타임 필요 |

---

## Layer 1: Data / Task (~240 values)

### 1.1.1 data_domain

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| macro | operational | - | - | 이미 완료 |
| macro_finance | planned | v1.0 | phase-03 | WRDS 브릿지 정리 |
| housing | registry_only | v1.1 | phase-10 | housing 데이터셋 결합 |
| energy | registry_only | v1.1 | phase-10 | energy 데이터셋 결합 |
| labor | registry_only | v1.1 | phase-10 | labor BLS 확장 |
| regional | registry_only | v1.1 | phase-10 | regional panel 합류 |
| panel_macro | future | v2 | phase-11 | panel_forecasting_run 필요 |
| text_macro | future | v2 | phase-11 | text pipeline 필요 |
| mixed_domain | future | v2 | phase-11 | 도메인 혼합 설계 필요 |

### 1.1.2 dataset_source

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fred_md | operational | - | - | 이미 완료 |
| fred_qd | operational | - | - | 이미 완료 |
| fred_sd | operational | - | - | 이미 완료 |
| fred_api_custom | planned | v1.0 | phase-03 | API loader 완성 |
| bea | registry_only | v1.1 | phase-10 | BEA adapter |
| bls | registry_only | v1.1 | phase-10 | BLS adapter |
| census | registry_only | v1.1 | phase-10 | census adapter |
| oecd | registry_only | v1.1 | phase-10 | OECD adapter |
| imf_ifs | registry_only | v1.1 | phase-10 | IMF IFS adapter |
| ecb_sdw | registry_only | v1.1 | phase-10 | ECB SDW adapter |
| bis | registry_only | v1.1 | phase-10 | BIS adapter |
| world_bank | registry_only | v1.1 | phase-10 | World Bank adapter |
| wrds_macro_finance | registry_only | v1.0 | phase-03 | WRDS 연결은 선확보 |
| survey_spf | registry_only | v1.1 | phase-10 | SPF adapter |
| blue_chip | future | v2 | phase-11 | 수작업 수집 필요 |
| market_prices | future | v1.1 | phase-10 | finance 축 확장 |
| high_frequency_surprises | future | v2 | phase-11 | event study 필요 |
| google_trends | future | post-v2 | phase-11 | exotic source |
| news_text | future | post-v2 | phase-11 | text pipeline 필요 |
| climate_series | future | post-v2 | phase-11 | exotic source |
| satellite_proxy | future | post-v2 | phase-11 | exotic source |
| custom_csv | planned | v1.0 | phase-03 | user loader |
| custom_parquet | planned | v1.0 | phase-03 | user loader |
| custom_duckdb | absent | v1.1 | phase-10 | registry 등록 필요 |
| custom_sql | future | v1.1 | phase-10 | SQL adapter |

### 1.1.3 frequency

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| daily | registry_only | v1.1 | phase-10 | daily aligner 필요 |
| weekly | registry_only | v1.1 | phase-10 | weekly aligner 필요 |
| monthly | operational | - | - | 이미 완료 |
| quarterly | operational | - | - | 이미 완료 |
| yearly | registry_only | v1.1 | phase-10 | annual 확장 |
| mixed_frequency | future | v2 | phase-11 | MIDAS/mixed 필요 |

### 1.1.4 information_set_type

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| revised | operational | - | - | 이미 완료 |
| real_time_vintage | operational | - | - | 이미 완료 |
| pseudo_oos_revised | planned | v1.0 | phase-03 | pseudo-OOS 정리 |
| pseudo_oos_vintage_aware | registry_only | v1.1 | phase-10 | vintage-aware pseudo OOS |
| release_calendar_aware | future | v2 | phase-11 | calendar engine 필요 |
| publication_lag_aware | future | v2 | phase-11 | release_lag_rule 와 정합 |

### 1.1.5 vintage_policy

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| latest_only | operational | - | - | 이미 완료 |
| single_vintage | operational | - | - | 이미 완료 |
| rolling_vintage | planned | v1.0 | phase-03 | vintage roll 정리 |
| all_vintages_available | registry_only | v1.1 | phase-10 | vintage DB 필요 |
| event_vintage_subset | future | v2 | phase-11 | event vintage 큐레이션 |
| vintage_range | registry_only | v1.1 | phase-10 | range subset |

### 1.1.6 alignment_rule

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| end_of_period | operational | - | - | 이미 완료 |
| average_within_period | registry_only | v1.0 | phase-03 | 주기 변환 필요 |
| last_available | planned | v1.0 | phase-03 | real-time 관련 |
| first_available | registry_only | v1.1 | phase-10 | 희귀 케이스 |
| quarter_to_month_repeat | registry_only | v1.1 | phase-10 | mixed-freq 전단계 |
| month_to_quarter_average | planned | v1.0 | phase-03 | 월→분기 집계 |
| month_to_quarter_last | planned | v1.0 | phase-03 | 월→분기 집계 |
| ragged_edge_keep | registry_only | v1.1 | phase-10 | ragged edge 처리 |
| ragged_edge_fill | registry_only | v1.1 | phase-10 | ragged edge 처리 |
| calendar_strict | registry_only | v2 | phase-11 | calendar engine 필요 |

### 1.1.7 release_lag_rule

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| ignore_release_lag | absent | v1.0 | phase-03 | 축 분해 대상 |
| fixed_lag_all_series | absent | v1.0 | phase-03 | 축 분해 대상 |
| series_specific_lag | absent | v1.0 | phase-03 | 축 분해 대상 |
| calendar_exact_lag | absent | v2 | phase-11 | calendar engine 필요 |
| lag_conservative | absent | v1.0 | phase-03 | 축 분해 대상 |
| lag_aggressive | absent | v1.0 | phase-03 | 축 분해 대상 |

### 1.1.8 missing_availability

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| complete_case_only | absent | v1.0 | phase-03 | 축 분해 대상 |
| available_case | absent | v1.0 | phase-03 | 축 분해 대상 |
| target_date_drop_if_missing | absent | v1.0 | phase-03 | 축 분해 대상 |
| x_impute_only | absent | v1.0 | phase-03 | 축 분해 대상 |
| real_time_missing_as_missing | absent | v1.1 | phase-10 | vintage-aware |
| state_space_fill | absent | v2 | phase-11 | SS framework 필요 |
| factor_fill | absent | v1.1 | phase-10 | factor 축 합류 |
| em_fill | absent | v1.0 | phase-03 | em_impute 연계 |

### 1.1.9 variable_universe

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| all_variables | absent | v1.0 | phase-03 | 축 분해 대상 |
| preselected_core | absent | v1.0 | phase-03 | 축 분해 대상 |
| category_subset | absent | v1.0 | phase-03 | 축 분해 대상 |
| paper_replication_subset | absent | v1.0 | phase-03 | 축 분해 대상 |
| target_specific_subset | absent | v1.0 | phase-03 | 축 분해 대상 |
| expert_curated_subset | absent | v1.1 | phase-10 | 큐레이션 필요 |
| stability_filtered_subset | absent | v1.1 | phase-10 | 안정성 필터 필요 |
| correlation_screened_subset | absent | v1.0 | phase-03 | screen 재사용 |
| feature_selection_dynamic_subset | absent | v1.1 | phase-10 | dynamic selection |

### 1.2.2 training_start_rule

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| earliest_possible | operational | - | - | 이미 완료 |
| fixed_start | planned | v1.0 | phase-03 | 날짜 픽스 |
| post_warmup_start | planned | v1.0 | phase-03 | warmup 결합 |
| post_break_start | registry_only | v1.0 | phase-03 | break segmentation 결합 |
| rolling_train_start | operational | - | - | 이미 완료 |

### 1.2.3 oos_period

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| single_oos_block | operational | - | - | 이미 완료 |
| multiple_oos_blocks | registry_only | v1.0 | phase-03 | multi-block OOS |
| rolling_origin | operational | - | - | 이미 완료 |
| recession_only_oos | planned | v1.0 | phase-04 | regime task 결합 |
| expansion_only_oos | planned | v1.0 | phase-04 | regime task 결합 |
| event_window_oos | registry_only | v1.1 | phase-10 | event study 지원 |

### 1.2.4 minimum_train_size

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fixed_n_obs | operational | - | - | 이미 완료 |
| fixed_years | planned | v1.0 | phase-03 | 축 분해 대상 |
| model_specific_min_train | registry_only | v1.0 | phase-03 | 축 분해 대상 |
| target_specific_min_train | registry_only | v1.0 | phase-03 | 축 분해 대상 |
| horizon_specific_min_train | registry_only | v1.0 | phase-03 | 축 분해 대상 |

### 1.2.5 warmup_rule

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| lags_only_warmup | operational | - | - | 이미 완료 |
| lags_and_factors_warmup | planned | v1.0 | phase-03 | factor warmup |
| sequence_warmup | future | v2 | phase-11 | seq2seq 대응 |
| transform_warmup | planned | v1.0 | phase-03 | transform warmup |
| indicator_warmup | registry_only | v1.1 | phase-10 | indicator 계열 |

### 1.2.6 break_segmentation

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| pre_post_crisis | planned | v1.0 | phase-03 | 축 분해 대상 |
| pre_post_covid | planned | v1.0 | phase-03 | 축 분해 대상 |
| user_break_dates | planned | v1.0 | phase-03 | 축 분해 대상 |
| break_test_detected | future | v2 | phase-11 | break test 모듈 필요 |
| rolling_break_adaptive | future | v2 | phase-11 | adaptive break 필요 |

### 1.3.1 horizon_list

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| default_1_3_6_12 | absent | v1.0 | phase-03 | 축 분해 대상 |
| short_only_1_3 | absent | v1.0 | phase-03 | 축 분해 대상 |
| long_only_12_24 | absent | v1.0 | phase-03 | 축 분해 대상 |
| paper_specific | absent | v1.0 | phase-03 | replication 지원 |
| arbitrary_grid | absent | v1.0 | phase-03 | 사용자 정의 |

### 1.3.2 forecast_type

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| direct | operational | - | - | 이미 완료 |
| iterated | planned | v1.1 | phase-10 | 다단 예측 합류 |
| dirrec | registry_only | v1.1 | phase-10 | 다단 예측 합류 |
| mimo | future | v2 | phase-11 | multi-output 결합 |
| multi_horizon_joint | future | v2 | phase-11 | joint horizon |
| recursive_state_space | future | v2 | phase-11 | SS framework 필요 |
| seq2seq | future | v2 | phase-11 | NN 축 필요 |

### 1.3.3 forecast_object

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| point_mean | operational | - | - | 이미 완료 |
| point_median | operational | - | - | 이미 완료 |
| quantile | planned | v1.1 | phase-10 | quantile regression 합류 |
| interval | registry_only | v1.1 | phase-10 | conformal/analytic interval |
| density | registry_only | v1.1 | phase-10 | density forecast |
| direction | planned | v1.0 | phase-04 | classification-ish 지원 |
| turning_point | registry_only | v2 | phase-11 | 전문 분석 모듈 |
| regime_probability | future | v2 | phase-11 | regime 모델 필요 |
| event_probability | future | v2 | phase-11 | event study 필요 |

### 1.3.4 horizon_target_construction

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| future_level_y_t+h | operational | - | - | 이미 완료 |
| future_diff | planned | v1.0 | phase-03 | 기본 차분 |
| future_logdiff | planned | v1.0 | phase-03 | 기본 로그차분 |
| cumulative_growth_to_h | planned | v1.0 | phase-03 | h-누적 성장 |
| average_growth_1_to_h | registry_only | v1.0 | phase-03 | 평균 성장 |
| annualized_growth_to_h | planned | v1.0 | phase-03 | 연율화 |
| realized_future_average | registry_only | v1.1 | phase-10 | 실현 평균 |
| future_sum | registry_only | v1.1 | phase-10 | 누적 합 |
| future_volatility | future | v2 | phase-11 | 변동성 타겟 |
| future_drawdown | future | v2 | phase-11 | drawdown 타겟 |
| future_indicator | registry_only | v1.0 | phase-04 | direction 과 정합 |

### 1.3.5 overlap_handling

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| allow_overlap | operational | - | - | 이미 완료 |
| evaluate_with_hac | planned | v1.0 | phase-02 | (Layer 6 stat_test 연계 — 정책 여기) |
| evaluate_with_block_bootstrap | registry_only | v1.1 | phase-10 | bootstrap module |
| non_overlapping_subsample | registry_only | v1.1 | phase-10 | subsample 정책 |
| horizon_specific_subsample | registry_only | v1.1 | phase-10 | subsample 정책 |

### 1.4.1 target_family

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| single_macro_series | operational | - | - | 이미 완료 |
| multiple_macro_series | planned | v1.1 | phase-10 | multi-target 합류 |
| panel_target | future | v2 | phase-11 | panel_forecasting_run 필요 |
| state_target | registry_only | v2 | phase-11 | SS framework 필요 |
| factor_target | future | v2 | phase-11 | factor 추출 결합 |
| latent_target | future | deferred-indef | - | SS 없이 불가 |
| constructed_target | registry_only | v1.1 | phase-10 | 합성 타겟 |
| classification_target | registry_only | v1.0 | phase-04 | direction/event 와 정합 |

### 1.4.2 predictor_family

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| target_lags_only | operational | - | - | 이미 완료 |
| all_macro_vars | operational | - | - | 이미 완료 |
| all_except_target | planned | v1.0 | phase-03 | 기본 옵션 |
| category_based | planned | v1.0 | phase-03 | 카테고리 subset |
| financial_only | absent | v1.1 | phase-10 | finance 축 합류 |
| macro_plus_finance | absent | v1.1 | phase-10 | finance 축 합류 |
| survey_plus_macro | absent | v1.1 | phase-10 | SPF 합류 |
| text_plus_macro | absent | v2 | phase-11 | text 합류 |
| factor_only | planned | v1.0 | phase-03 | factor 축 합류 |
| latent_state_plus_lags | absent | v2 | phase-11 | SS 필요 |
| selected_sparse_set | absent | v1.0 | phase-03 | feature selection 결합 |
| handpicked_set | registry_only | v1.0 | phase-03 | replication 지원 |

### 1.4.3 contemporaneous_x_rule

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| allow_contemporaneous | registry_only | v1.1 | phase-10 | 조건부 허용 |
| forbid_contemporaneous | operational | - | - | 이미 완료 |
| allow_if_available_in_real_time | future | v2 | phase-11 | calendar 결합 필요 |
| lag_all_predictors_by_one | absent | v1.0 | phase-03 | 간단 정책 |
| series_specific_contemporaneous | future | v2 | phase-11 | 세밀 제어 필요 |

### 1.4.4 own_target_lags

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| include | operational | - | - | 이미 완료 |
| exclude | planned | v1.0 | phase-03 | 간단 스위치 |
| cv_select_lags | planned | v1.0 | phase-03 | CV 통합 |
| fixed_lag_count | absent | v1.0 | phase-03 | 고정 lag |
| target_specific_lag_count | registry_only | v1.1 | phase-10 | 타겟별 lag |

### 1.4.5 deterministic_components

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| constant_only | operational | - | - | 이미 완료 |
| linear_trend | planned | v1.0 | phase-03 | 기본 trend |
| seasonal_dummies | absent | v1.0 | phase-03 | 계절 더미 |
| month_dummies | registry_only | v1.0 | phase-03 | monthly_seasonal 매핑 |
| quarter_dummies | registry_only | v1.0 | phase-03 | quarterly_seasonal 매핑 |
| holiday_calendar | absent | v2 | phase-11 | calendar engine 필요 |
| event_dummies | absent | v1.1 | phase-10 | event 연계 |
| break_dummies | registry_only | v1.0 | phase-03 | break 축 결합 |

### 1.4.6 exogenous_block

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| strict_exogenous_only | registry_only | v1.0 | phase-03 | 정책 플래그 |
| endogenous_allowed | operational | - | - | 이미 완료 |
| instrumented_exogenous | absent | v2 | phase-11 | IV 필요 |
| policy_shock_block | absent | v2 | phase-11 | 정책 충격 라벨 |

### 1.5.1 x_map_policy

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| shared_X | planned | v1.1 | phase-10 | multi-target 합류 |
| target_specific_X | registry_only | v1.1 | phase-10 | multi-target 합류 |
| category_specific_X | registry_only | v1.1 | phase-10 | multi-target 합류 |
| domain_specific_X | absent | v1.1 | phase-10 | multi-target 합류 |
| availability_specific_X | absent | v1.1 | phase-10 | multi-target 합류 |
| handcrafted_target_X | absent | v1.1 | phase-10 | multi-target 합류 |
| learned_target_X | future | v2 | phase-11 | 학습형 매핑 |

### 1.5.2 target_to_target_inclusion

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| allow_other_targets_as_X | registry_only | v1.1 | phase-10 | multi-target 합류 |
| forbid_other_targets_as_X | planned | v1.1 | phase-10 | multi-target 합류 |
| allow_selected_targets_as_X | registry_only | v1.1 | phase-10 | multi-target 합류 |
| Granger_style_lagged_targets_only | absent | v1.1 | phase-10 | multi-target 합류 |
| system_wide_joint_model | future | v2 | phase-11 | joint VAR 필요 |

### 1.5.3 multi_target_architecture

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| separate_univariate_runs | planned | v1.0 | phase-01 | 기본 sweep 형태 |
| same_design_different_targets | planned | v1.1 | phase-10 | shared design |
| joint_multivariate_model | future | v1.1 | phase-10 | phase 10 공식 대상 |
| shared_encoder_multi_head | absent | v2 | phase-11 | NN 축 필요 |
| hierarchical_bottom_up | absent | v2 | phase-11 | hierarchy 모듈 필요 |
| hierarchical_top_down | absent | v2 | phase-11 | hierarchy 모듈 필요 |
| reconciliation_after_forecast | absent | v2 | phase-11 | reconcile 모듈 필요 |

### 1.6.1 scale_at_evaluation

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| transformed_scale | registry_only | v1.0 | phase-03 | inverse_transform 결합 |
| original_scale | operational | - | - | 이미 완료 (raw_level 매핑) |
| both | absent | v1.0 | phase-03 | 축 분해 대상 |

### 1.6.2 benchmark_family_task

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| historical_mean | operational | - | - | 이미 완료 |
| rolling_mean | planned | v1.0 | phase-04 | benchmark 정리 |
| random_walk | planned | v1.0 | phase-04 | benchmark 정리 |
| ar_bic | operational | - | - | 이미 완료 |
| ar_fixed_p | planned | v1.0 | phase-04 | benchmark 정리 |
| ardi | registry_only | v1.0 | phase-04 | benchmark 정리 |
| factor_model | registry_only | v1.0 | phase-04 | benchmark 정리 |
| var | future | v1.1 | phase-10 | VAR 합류 |
| expert_benchmark | future | v1.1 | phase-10 | 전문가 벤치 |
| paper_specific_benchmark | registry_only | v1.0 | phase-04 | replication 지원 |

### 1.6.3 regime_conditional_task

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| unconditional | operational | - | - | 이미 완료 |
| recession_conditioned | planned | v1.0 | phase-04 | regime 분리 |
| expansion_conditioned | planned | v1.0 | phase-04 | regime 분리 |
| high_uncertainty_conditioned | absent | v1.1 | phase-10 | uncertainty 지수 필요 |
| state_dependent_train | absent | v2 | phase-11 | SS 필요 |
| state_dependent_eval | absent | v2 | phase-11 | SS 필요 |

---

## Layer 2: Preprocessing (~200 values)

### 2.0.1 separation_rule

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| strict_separation | absent | v1.0 | phase-03 | 축 분해 대상 |
| shared_transform_then_split | absent | v1.0 | phase-03 | 축 분해 대상 |
| joint_preprocessor | absent | v1.1 | phase-10 | joint 경로 |
| target_only_transform | absent | v1.0 | phase-03 | 축 분해 대상 |
| X_only_transform | absent | v1.0 | phase-03 | 축 분해 대상 |

### 2.0.2 preprocessing_fit_scope

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fit_on_train_only | operational | - | - | 이미 완료 (train_only) |
| fit_on_train_expand | registry_only | v1.0 | phase-03 | expanding_train_only 매핑 |
| fit_on_train_roll | registry_only | v1.0 | phase-03 | rolling_train_only 매핑 |
| fit_on_full_sample_forbidden | absent | v1.0 | phase-00 | 누수 방지 가드 |
| leakage_checked | absent | v1.0 | phase-00 | 누수 가드 (reproducibility) |

### 2.1.1 target_missing

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| drop_missing_target_rows | registry_only | v1.0 | phase-03 | drop 매핑 |
| drop_forecast_dates_if_target_missing | absent | v1.0 | phase-03 | 축 분해 대상 |
| linear_interpolate | absent | v1.0 | phase-03 | 기본 보간 |
| ffill | absent | v1.0 | phase-03 | 기본 보간 |
| bfill | absent | v1.0 | phase-03 | 기본 보간 (제한적) |
| seasonal_interpolate | absent | v1.1 | phase-10 | 계절 보간 |
| kalman_smooth | absent | v2 | phase-11 | SS 필요 |
| model_based_impute | absent | v1.1 | phase-10 | 모델 기반 imputation |
| multiple_imputation | absent | v2 | phase-11 | MICE 등 |
| do_not_impute_target | operational | - | - | 이미 완료 (none) |

### 2.1.2 target_outlier

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| winsorize | absent | v1.0 | phase-03 | 기본 처리 |
| trim | absent | v1.0 | phase-03 | 기본 처리 |
| clip | registry_only | v1.0 | phase-03 | clip 매핑 |
| iqr_flag_to_missing | absent | v1.0 | phase-03 | outlier_to_nan 결합 |
| mad_flag_to_missing | absent | v1.0 | phase-03 | outlier_to_nan 결합 |
| robust_smoother | absent | v1.1 | phase-10 | 로버스트 스무더 |
| manual_event_override | absent | v1.1 | phase-10 | 이벤트 수기 수정 |
| tail_transform | absent | v1.1 | phase-10 | 꼬리 변환 |

### 2.1.3 target_transform

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| level | operational | - | - | 이미 완료 |
| difference | planned | v1.0 | phase-03 | 기본 변환 |
| log | planned | v1.0 | phase-03 | 기본 변환 |
| log_difference | planned | v1.0 | phase-03 | 기본 변환 |
| growth_rate | planned | v1.0 | phase-03 | 기본 변환 |
| annualized_growth | absent | v1.0 | phase-03 | horizon 결합 |
| yoy_growth | absent | v1.0 | phase-03 | 기본 변환 |
| qoq_saar | absent | v1.0 | phase-03 | 기본 변환 |
| standardized_target | absent | v1.0 | phase-03 | normalization 결합 |
| BoxCox | absent | v1.1 | phase-10 | 전문 변환 |
| YeoJohnson | absent | v1.1 | phase-10 | 전문 변환 |
| rank_normal | absent | v1.1 | phase-10 | 전문 변환 |
| sign_preserving_log | absent | v1.1 | phase-10 | 전문 변환 |
| binary_direction | absent | v1.0 | phase-04 | direction task |
| threshold_event_target | absent | v1.1 | phase-10 | event target |

### 2.1.4 transform_timing

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| transform_then_horizon_build | absent | v1.0 | phase-03 | 축 분해 대상 |
| horizon_build_then_transform | absent | v1.0 | phase-03 | 축 분해 대상 |
| task_specific | absent | v1.0 | phase-03 | 축 분해 대상 |

### 2.1.5 inverse_transform

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| required | absent | v1.0 | phase-03 | 정책화 |
| optional | absent | v1.0 | phase-03 | 정책화 |
| not_needed | operational | - | - | 이미 완료 (none) |
| evaluate_both_scales | absent | v1.0 | phase-03 | scale_at_evaluation 결합 |

### 2.1.6 target_normalization

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| zscore_train_only | planned | v1.0 | phase-03 | 기본 정규화 |
| robust_zscore | planned | v1.0 | phase-03 | 기본 정규화 |
| minmax | registry_only | v1.0 | phase-03 | 기본 정규화 |
| unit_variance | registry_only | v1.0 | phase-03 | 기본 정규화 |
| de_mean_only | absent | v1.0 | phase-03 | 기본 정규화 |
| rolling_standardize | absent | v1.1 | phase-10 | rolling 정규화 |

### 2.1.7 target_domain_restriction

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| unconstrained | operational | - | - | 이미 완료 |
| nonnegative | registry_only | v1.1 | phase-10 | 도메인 제약 |
| bounded_0_1 | registry_only | v1.1 | phase-10 | 도메인 제약 |
| integer_count | future | v2 | phase-11 | count 모델 필요 |
| probability_target | future | v2 | phase-11 | prob 모델 필요 |

### 2.1.8 target_class_handling

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| not_applicable | absent | v1.0 | phase-03 | 기본 (classification 아님) |
| class_weighting | absent | v1.0 | phase-03 | 축 분해 대상 |
| oversample | absent | v1.0 | phase-03 | 축 분해 대상 |
| undersample | absent | v1.0 | phase-03 | 축 분해 대상 |
| smote_time_aware | absent | v1.1 | phase-10 | 고급 기법 |
| threshold_tuning | absent | v1.0 | phase-04 | direction task 결합 |

### 2.2.1 x_missing

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| drop_rows | planned | v1.0 | phase-03 | 기본 |
| drop_columns | planned | v1.0 | phase-03 | 기본 |
| drop_if_above_missing_threshold | planned | v1.0 | phase-03 | 기본 |
| mean_impute | operational | - | - | 이미 완료 |
| median_impute | operational | - | - | 이미 완료 |
| group_mean_impute | absent | v1.1 | phase-10 | 그룹 impute |
| timewise_mean_impute | absent | v1.1 | phase-10 | 시간축 impute |
| ffill | operational | - | - | 이미 완료 |
| bfill | absent | v1.0 | phase-03 | ffill 쌍 |
| interpolate_linear | operational | - | - | 이미 완료 |
| interpolate_spline | absent | v1.1 | phase-10 | spline 보간 |
| em_impute | operational | - | - | 이미 완료 |
| kalman_impute | absent | v2 | phase-11 | SS 필요 |
| factor_impute | absent | v1.1 | phase-10 | factor 결합 |
| knn_impute | absent | v1.1 | phase-10 | knn |
| iterative_imputer | absent | v1.1 | phase-10 | iterative (sklearn) |
| multiple_imputation | absent | v2 | phase-11 | MICE |
| missing_indicator_addition | planned | v1.0 | phase-03 | missing_indicator 매핑 |
| leave_as_missing_for_model | absent | v1.1 | phase-10 | 모델별 NaN 처리 |

### 2.2.2 x_outlier

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| winsorize | operational | - | - | 이미 완료 |
| trim | planned | v1.0 | phase-03 | 기본 |
| iqr_clip | operational | - | - | 이미 완료 |
| mad_clip | planned | v1.0 | phase-03 | 기본 |
| zscore_clip | operational | - | - | 이미 완료 |
| outlier_to_missing | planned | v1.0 | phase-03 | 기본 |
| robust_scaler_only | absent | v1.1 | phase-10 | 스케일러 결합 |
| Huberize | absent | v1.1 | phase-10 | Huber 변환 |
| quantile_cap | absent | v1.1 | phase-10 | 분위 캡 |

### 2.2.3 standardize_scale

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| zscore | operational | - | - | 이미 완료 (standard) |
| robust_zscore | operational | - | - | 이미 완료 (robust) |
| demean_only | planned | v1.0 | phase-03 | 기본 |
| unit_variance_only | planned | v1.0 | phase-03 | 기본 |
| minmax | operational | - | - | 이미 완료 |
| rank_scale | registry_only | v1.1 | phase-10 | rank scaler |
| quantile_transform | absent | v1.1 | phase-10 | 분위 변환 |
| whitening | absent | v2 | phase-11 | PCA whitening |
| groupwise_standardize | absent | v1.1 | phase-10 | 그룹별 |
| expanding_standardize | absent | v1.0 | phase-03 | expanding |
| rolling_standardize | absent | v1.1 | phase-10 | rolling |

### 2.2.4 scaling_scope

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| columnwise | operational | - | - | 이미 완료 |
| datewise_cross_sectional | registry_only | v1.1 | phase-10 | panel 연계 |
| groupwise | registry_only | v1.1 | phase-10 | 그룹 scope |
| categorywise | registry_only | v1.1 | phase-10 | 카테고리 scope |
| global_train_only | operational | - | - | 이미 완료 |
| train_window_only | absent | v1.0 | phase-03 | window scope |

### 2.2.5 additional_preprocessing

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| smoothing_ma | registry_only | v1.0 | phase-03 | 기본 MA |
| ema | registry_only | v1.1 | phase-10 | EMA 보완 |
| hp_filter | planned | v2 | phase-11 | 고급 필터 |
| bandpass_filter | registry_only | v2 | phase-11 | 고급 필터 |
| wavelet_denoise | absent | v2 | phase-11 | 고급 필터 |
| seasonal_adjustment | absent | v1.1 | phase-10 | SA 모듈 |
| detrending | absent | v1.0 | phase-03 | 기본 |
| deseasonalizing | absent | v1.1 | phase-10 | 계절 조정 |
| nonlinear_transform_bank | absent | v1.1 | phase-10 | 비선형 변환 |
| threshold_transform | absent | v1.1 | phase-10 | 임계 변환 |
| interaction_generation | absent | v1.1 | phase-10 | 상호작용 |
| polynomial_expansion | absent | v1.1 | phase-10 | 다항 확장 |
| spline_basis | absent | v1.1 | phase-10 | spline basis |
| kernel_features | absent | v2 | phase-11 | kernel 기법 |
| text_embedding | absent | post-v2 | phase-11 | text 파이프라인 |
| autoencoder_embedding | absent | v2 | phase-11 | NN 필요 |

### 2.2.6 lag_creation

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| no_x_lags | operational | - | - | 이미 완료 |
| fixed_x_lags | planned | v1.0 | phase-03 | 기본 lag |
| cv_selected_x_lags | planned | v1.0 | phase-03 | CV 연계 |
| variable_specific_lags | registry_only | v1.1 | phase-10 | 변수별 lag |
| category_specific_lags | registry_only | v1.1 | phase-10 | 카테고리 lag |
| distributed_lags | absent | v1.1 | phase-10 | DL 모델 |
| MIDAS_lags | absent | v2 | phase-11 | mixed freq 필요 |
| Almon_lags | absent | v2 | phase-11 | Almon polynomial |

### 2.2.7 dimensionality_reduction

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| PCA | operational | - | - | 이미 완료 (pca) |
| static_factor | operational | - | - | 이미 완료 |
| dynamic_factor | absent | v2 | phase-11 | SS 결합 |
| targeted_PCA | absent | v1.1 | phase-10 | targeted factor |
| sparse_PCA | absent | v1.1 | phase-10 | sparse PCA |
| PLS | absent | v1.1 | phase-10 | PLS |
| PCR | absent | v1.1 | phase-10 | PCR |
| ICA | absent | v1.1 | phase-10 | ICA |
| autoencoder | absent | v2 | phase-11 | NN 필요 |
| supervised_encoder | absent | v2 | phase-11 | NN 필요 |
| random_projection | absent | v1.1 | phase-10 | RP |
| feature_clustering | absent | v1.1 | phase-10 | 클러스터링 |

### 2.2.8 feature_selection

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| correlation_screen | operational | - | - | 이미 완료 (correlation_filter) |
| mutual_information_screen | registry_only | v1.1 | phase-10 | MI screen |
| univariate_F_test | absent | v1.0 | phase-03 | 기본 screen |
| lasso_selection | operational | - | - | 이미 완료 (lasso_select) |
| stability_selection | absent | v1.1 | phase-10 | stability |
| recursive_feature_elimination | absent | v1.1 | phase-10 | RFE |
| tree_based_screen | absent | v1.1 | phase-10 | tree importance |
| Boruta | absent | v2 | phase-11 | Boruta 정식 지원 |
| group_selection | absent | v1.1 | phase-10 | group lasso |
| economic_prior_selection | absent | v1.1 | phase-10 | prior 기반 |

### 2.2.9 feature_grouping

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| fred_category_group | planned | v1.0 | phase-03 | FRED 카테고리 |
| economic_theme_group | registry_only | v1.1 | phase-10 | 큐레이션 필요 |
| lag_group | planned | v1.0 | phase-03 | lag 그룹 |
| factor_group | registry_only | v1.1 | phase-10 | factor 그룹 |
| text_group | absent | v2 | phase-11 | text 합류 |
| spatial_group | absent | v2 | phase-11 | spatial 합류 |

### 2.3.1 execution_order

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| user_defined_ordered | absent | v1.0 | phase-03 | recipe 체계 |
| canonical_default | operational | - | - | 이미 완료 (preprocess_order.none/extra_only) |
| align_then_target_transform_then_x_missing_then_outlier_then_scale_then_lag_then_factor | absent | v1.0 | phase-03 | canonical 명시화 |

### 2.3.2 recipe_mode

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fixed_recipe | operational | - | - | 이미 완료 |
| recipe_grid | planned | v1.0 | phase-01 | sweep 체계 |
| recipe_ablation | planned | v1.0 | phase-01 | ablation_study 결합 |
| paper_exact_recipe | registry_only | v1.0 | phase-03 | replication 지원 |
| model_specific_recipe | registry_only | v1.1 | phase-10 | 모델별 recipe |
| target_specific_recipe | absent | v1.1 | phase-10 | 타겟별 recipe |

---

## Summary

- **Layer 0**: 45 rows across 6 sub-axes
- **Layer 1**: 191 rows across 28 sub-axes (1.1.1–1.1.9, 1.2.2–1.2.6, 1.3.1–1.3.5, 1.4.1–1.4.6, 1.5.1–1.5.3, 1.6.1–1.6.3)
- **Layer 2**: 167 rows across 22 sub-axes (2.0.1–2.0.2, 2.1.1–2.1.8, 2.2.1–2.2.9, 2.3.1–2.3.2)
- **Total rows**: 403

Status buckets (approx): operational ~95, registry_only ~95, planned ~60, future ~30, absent ~123. Majority of `absent` entries are new axes spec'd in user universe but not yet carved into Python registries (release_lag_rule, missing_availability, variable_universe, horizon_list, separation_rule, transform_timing, target_class_handling, many Layer 2.1–2.2 values) — all slated for phase-03 or phase-10.

# Coverage Ledger — Part 2 (Layer 3-5)

Continuation of `coverage_ledger_part1.md`. See part 1 for legend.

Generated against server1 registry snapshot at `~/project/macroforecast/macrocast/registry/{training,evaluation,output}/` on 2026-04-17.

---

## Layer 3: Forecasting / Training (~150 values)

### 3.1 forecasting framework

#### 3.1.1 outer_window

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| expanding | operational | - | - | 이미 완료 |
| rolling | operational | - | - | 이미 완료 |
| anchored_rolling | operational | - | - | 이미 완료 |
| hybrid_expanding_rolling | registry_only | v1.0 | phase-03 | hybrid window scheduler 필요 |
| recursive_reestimation | registry_only | v1.1 | phase-10 | outer 재추정 전략 승격 |
| event_retrain | absent | v1.1 | phase-10 | event-triggered retrain hook 필요 |

#### 3.1.2 refit_policy

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| refit_every_step | operational | - | - | 이미 완료 |
| refit_every_k_steps | operational | - | - | 이미 완료 |
| fit_once_predict_many | operational | - | - | 이미 완료 |
| warm_start_refit | registry_only | v1.0 | phase-03 | warm-start path 활성화 |
| online_update | future | v2 | phase-11 | online learning 전용 API 필요 |
| partial_fit | absent | v2 | phase-11 | sklearn partial_fit 어댑터 필요 |

#### 3.1.3 data_rich_mode

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| target_lags_only | operational | - | - | 이미 완료 |
| factor_plus_lags | planned | v1.0 | phase-03 | factor 빌더와 결합 |
| full_high_dimensional_X | operational | - | - | 이미 완료 |
| selected_sparse_X | planned | v1.0 | phase-03 | lasso/selector 경로 |
| mixed_mode | registry_only | v1.1 | phase-10 | 복합 모드 분기 필요 |

#### 3.1.4 sequence_framework

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| not_sequence | operational | - | - | 이미 완료 |
| fixed_lookback_sequence | future | v1.1 | phase-10 | deep 모델 활성화 시 |
| variable_lookback_sequence | future | v1.1 | phase-10 | 변동 lookback 래퍼 필요 |
| multi_resolution_sequence | future | v2 | phase-11 | MIDAS/다주기 합류 |
| encoder_decoder_sequence | future | v2 | phase-11 | seq2seq 계열 전용 |

#### 3.1.5 horizon_modelization

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| separate_model_per_h | operational | - | - | 이미 완료 |
| shared_model_multi_h | registry_only | v1.1 | phase-10 | multi-output runner 필요 |
| shared_backbone_multi_head | future | v1.1 | phase-10 | deep 모델과 함께 승격 |
| recursive_one_step_model | planned | v1.0 | phase-03 | recursive 전략 구현 필요 |
| hybrid_h_specific | registry_only | v1.1 | phase-10 | 혼합 horizon 전략 |

### 3.2 train-validation

#### 3.2.1 validation_size_rule

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| ratio | operational | - | - | 이미 완료 |
| fixed_n | operational | - | - | 이미 완료 |
| fixed_years | operational | - | - | 이미 완료 |
| fixed_dates | registry_only | v1.0 | phase-03 | 날짜 기반 분할 활성 |
| horizon_specific_n | registry_only | v1.1 | phase-10 | horizon별 val 크기 |
| model_specific_n | absent | v1.1 | phase-10 | 모델별 정책 승격 |

#### 3.2.2 validation_location

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| last_block | operational | - | - | 이미 완료 |
| rolling_blocks | operational | - | - | 이미 완료 |
| expanding_validation | operational | - | - | 이미 완료 |
| blocked_cv | operational | - | - | 이미 완료 |
| nested_time_cv | registry_only | v1.1 | phase-10 | nested loop runner 필요 |
| walk_forward_validation | absent | v1.1 | phase-10 | walk-forward 래퍼 필요 |

#### 3.2.3 embargo_gap

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| fixed_gap | planned | v1.0 | phase-03 | 기본 embargo 활성 |
| horizon_gap | planned | v1.0 | phase-03 | horizon과 연동 |
| publication_gap | future | v1.1 | phase-10 | vintage와 연결 |
| custom_gap | registry_only | v1.1 | phase-10 | 사용자 정의 gap 허용 |

### 3.3 split

#### 3.3.1 split_family

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| simple_holdout | operational | - | - | 이미 완료 |
| time_split | operational | - | - | 이미 완료 |
| blocked_kfold | operational | - | - | 이미 완료 |
| expanding_cv | operational | - | - | 이미 완료 |
| rolling_cv | operational | - | - | 이미 완료 |
| nested_cv | absent | v1.1 | phase-10 | nested CV 프레임 승격 |
| poos_cv | absent | v1.0 | phase-03 | POOS CV 구현 필요 |
| BIC_selection | absent | v1.0 | phase-03 | IC 기반 선택 |
| AIC_selection | absent | v1.0 | phase-03 | IC 기반 선택 |
| IC_grid | absent | v1.0 | phase-03 | IC grid 선택 |
| purged_time_cv | absent | v1.1 | phase-10 | purged CV (금융계열) |
| combinatorial_purged_cv | absent | v2 | phase-11 | CPCV full-scale 배치 |

#### 3.3.2 shuffle_rule

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| no_shuffle | operational | - | - | 이미 완료 |
| restricted_shuffle_for_iid_only | registry_only | v1.1 | phase-10 | iid-only 가드 활성 |
| groupwise_shuffle | registry_only | v1.1 | phase-10 | panel 단계 이후 |
| forbidden_for_time_series | operational | - | - | 이미 완료 |

#### 3.3.3 alignment_fairness

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| same_split_across_models | operational | - | - | 이미 완료 |
| same_split_across_targets | planned | v1.0 | phase-03 | target sweep과 동반 |
| same_split_across_horizons | planned | v1.0 | phase-03 | horizon sweep과 동반 |
| model_specific_split_allowed | registry_only | v1.1 | phase-10 | opt-in flag 승격 |
| target_specific_split_allowed | registry_only | v1.1 | phase-10 | target별 split 허용 |

### 3.4 models

#### 3.4.1 naive/benchmark

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| historical_mean | absent | v1.0 | phase-04 | benchmark suite 핵심 |
| rolling_mean | absent | v1.0 | phase-04 | benchmark suite 핵심 |
| random_walk | absent | v1.0 | phase-04 | 기본 naive |
| drift | absent | v1.0 | phase-04 | naive with drift |
| seasonal_naive | absent | v1.0 | phase-04 | 계절형 naive |
| AR | operational | - | - | 이미 완료 (model_family=ar) |
| ARDI | absent | v1.0 | phase-04 | factor-plus-AR benchmark |
| factor_model | operational | - | - | factor_augmented_linear 존재 |
| VAR | absent | v1.0 | phase-05a | Phase 5a VAR 합류 |
| BVAR | absent | v1.0 | phase-05a | Phase 5a BVAR 합류 |
| FAVAR | absent | v1.1 | phase-05b | v1.1 factor-VAR |
| DFM | absent | v1.1 | phase-05b | dynamic factor model |
| TVP_AR | absent | v2 | phase-05c | time-varying param AR |
| MIDAS | absent | v2 | phase-05c | 혼합주기 MIDAS |
| U_MIDAS | absent | v2 | phase-05c | unrestricted MIDAS |
| ETS | absent | v1.0 | phase-04 | statsmodels ETS |
| ARIMA | absent | v1.0 | phase-04 | ARIMA benchmark |
| SARIMA | absent | v1.1 | phase-10 | 계절 ARIMA |
| UnobservedComponents | absent | v2 | phase-05c | UC 상태공간 |
| state_space | absent | v2 | phase-05c | 일반 state-space |

#### 3.4.2 linear/regularized

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| OLS | operational | - | - | 이미 완료 (ols) |
| Ridge | operational | - | - | 이미 완료 |
| Lasso | operational | - | - | 이미 완료 |
| AdaptiveLasso | operational | - | - | 이미 완료 |
| GroupLasso | absent | v1.1 | phase-10 | group-lasso 확장 |
| ElasticNet | operational | - | - | 이미 완료 |
| PCR | operational | - | - | 이미 완료 |
| PLS | operational | - | - | 이미 완료 |
| BayesianRidge | operational | - | - | 이미 완료 |
| HuberReg | operational | - | - | 이미 완료 (huber) |
| QuantileLinear | operational | - | - | 이미 완료 |
| SparseGroupLasso | absent | v1.1 | phase-10 | sparse-group 확장 |
| TVP_Ridge | absent | v2 | phase-05c | 시변 ridge |
| booging | absent | v1.1 | phase-10 | boosting+bagging 혼합 |
| factor_augmented_linear | operational | - | - | 이미 완료 |

#### 3.4.3 kernel/margin

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| KRR | absent | v1.1 | phase-10 | kernel ridge |
| SVR_linear | operational | - | - | 이미 완료 |
| SVR_rbf | operational | - | - | 이미 완료 |
| SVR_poly | absent | v1.1 | phase-10 | poly 커널 |
| GaussianProcess | absent | v2 | phase-11 | GP 비용 큼 |
| kernel_quantile_regression | absent | v2 | phase-11 | kernel quantile |

#### 3.4.4 tree/ensemble

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| RandomForest | operational | - | - | 이미 완료 |
| ExtraTrees | operational | - | - | 이미 완료 |
| GradientBoosting | operational | - | - | 이미 완료 (gbm) |
| XGBoost | operational | - | - | 이미 완료 |
| LightGBM | operational | - | - | 이미 완료 |
| CatBoost | operational | - | - | 이미 완료 |
| AdaBoost | absent | v1.1 | phase-10 | adaboost 확장 |
| bagging_regressor | absent | v1.1 | phase-10 | bagging wrapper |
| quantile_forest | absent | v2 | phase-11 | probabilistic 경로 |
| distributional_boosting | absent | v2 | phase-11 | NGBoost 계열 |

#### 3.4.5 neural

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| MLP | operational | - | - | 이미 완료 |
| DeepMLP | absent | v1.0 | phase-05a | `[deep]` extras와 함께 |
| ResNet_tabular | absent | v1.1 | phase-10 | tabular ResNet |
| LSTM | absent | v1.0 | phase-05a | Phase 5a LSTM |
| GRU | absent | v1.0 | phase-05a | Phase 5a GRU |
| TCN | absent | v1.0 | phase-05a | Phase 5a TCN |
| Transformer_encoder | absent | v1.1 | phase-05b | v1.1 transformer |
| Informer | absent | v1.1 | phase-05b | Informer 승격 |
| NBEATS | absent | v1.1 | phase-05b | N-BEATS 승격 |
| NHITS | absent | v1.1 | phase-10 | NHITS 후속 |
| TFT | absent | v1.1 | phase-05b | TFT 승격 |
| seq2seq_rnn | absent | v1.1 | phase-05b | seq2seq 승격 |
| mixture_of_experts | absent | v2 | phase-11 | MoE 후순위 |
| foundation_model_adapter | absent | deferred-indef | - | 외부 어댑터 보류 |

#### 3.4.6 panel/spatial/hierarchical

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| panel_FE_forecast | absent | v2 | phase-11 | panel FE runner |
| panel_RE_forecast | absent | v2 | phase-11 | panel RE runner |
| dynamic_panel | absent | v2 | phase-11 | dynamic panel |
| spatial_AR | absent | v2 | phase-11 | spatial AR |
| spatial_Durbin | absent | v2 | phase-11 | spatial Durbin |
| graph_neural_forecast | absent | v2 | phase-11 | GNN 예측 |
| hierarchical_reconciliation_model | absent | v2 | phase-11 | reconciliation |
| cross_state_factor_model | absent | v2 | phase-11 | cross-state factor |

#### 3.4.7 probabilistic/quantile

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| quantile_RF | absent | v2 | phase-11 | probabilistic wave |
| quantile_GBM | absent | v2 | phase-11 | probabilistic wave |
| quantile_XGB | absent | v2 | phase-11 | probabilistic wave |
| quantile_LSTM | absent | v2 | phase-11 | probabilistic wave |
| mixture_density_network | absent | v2 | phase-11 | MDN |
| BayesianNN | absent | v2 | phase-11 | BNN |
| distributional_regression | absent | v2 | phase-11 | distribution reg |
| conformal_wrapper | absent | v2 | phase-11 | conformal 경로 |

#### 3.4.8 custom/plugin

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| sklearn_adapter | operational | - | - | 이미 완료 (내부) |
| statsmodels_adapter | absent | v1.0 | phase-04 | benchmark suite 동반 |
| pytorch_adapter | absent | v1.0 | phase-05a | `[deep]` 경로 |
| jax_adapter | absent | deferred-indef | - | first-party 계획 없음 |
| R_adapter | absent | deferred-indef | - | first-party 계획 없음 |
| external_binary_adapter | absent | deferred-indef | - | 외부 바이너리 보류 |

### 3.5 hyperparameter tuning

#### 3.5.1 search_algorithm

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| grid_search | operational | - | - | 이미 완료 |
| random_search | operational | - | - | 이미 완료 |
| bayesian_optimization | operational | - | - | 이미 완료 |
| genetic_algorithm | operational | - | - | 이미 완료 |
| evolutionary_search | future | v2 | phase-11 | 진화탐색 승격 |
| hyperband | absent | v1.1 | phase-10 | hyperband 도입 |
| asha | absent | v1.1 | phase-10 | ASHA 스케줄러 |
| successive_halving | absent | v1.1 | phase-10 | SH 도입 |
| coordinate_search | absent | v2 | phase-11 | coord search 승격 |
| manual_fixed_hp | operational | - | - | 이미 완료 (fixed policy) |
| paper_exact_hp | absent | v1.0 | phase-04 | paper replication 축 |

#### 3.5.2 tuning_objective

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| validation_mse | operational | - | - | 이미 완료 |
| validation_rmse | operational | - | - | 이미 완료 |
| validation_mae | operational | - | - | 이미 완료 |
| validation_mape | registry_only | v1.1 | phase-10 | MAPE 활성 |
| validation_quantile_loss | future | v2 | phase-11 | quantile wave |
| relative_msfe_to_benchmark | absent | v1.0 | phase-04 | benchmark와 연동 |
| oos_r2_proxy | absent | v1.1 | phase-10 | oos-r2 대리지표 |
| economic_utility | absent | post-v2 | - | 경제적 효용 후순위 |
| custom_loss | absent | post-v2 | - | 커스텀 손실 후순위 |

#### 3.5.3 tuning_budget

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| max_trials | operational | - | - | 이미 완료 |
| max_time | operational | - | - | 이미 완료 |
| max_epochs | future | v1.0 | phase-05a | deep 모델 합류시 |
| max_models | registry_only | v1.1 | phase-10 | model 기반 예산 |
| early_stop_trials | operational | - | - | 이미 완료 |

#### 3.5.4 hp_space_style

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| discrete_grid | operational | - | - | 이미 완료 |
| continuous_box | planned | v1.0 | phase-03 | BO/random 활성 |
| log_uniform | planned | v1.0 | phase-03 | regularization 경로 |
| categorical | operational | - | - | 이미 완료 |
| conditional_space | registry_only | post-v2 | - | 조건부 공간 후순위 |
| hierarchical_space | absent | post-v2 | - | 계층 공간 후순위 |

#### 3.5.5 seed_policy

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fixed_seed | operational | - | - | 이미 완료 |
| multi_seed_average | planned | v1.0 | phase-03 | multi-seed averaging |
| seed_sweep | registry_only | v1.1 | phase-10 | seed sweep 승격 |
| deterministic_only | operational | - | - | 이미 완료 |

#### 3.5.6 early_stopping

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| validation_patience | operational | - | - | 이미 완료 |
| loss_plateau | operational | - | - | 이미 완료 |
| time_budget_stop | registry_only | v1.1 | phase-10 | 시간 예산 정지 |
| trial_pruning | registry_only | v1.1 | phase-10 | pruning 승격 |

#### 3.5.7 convergence_handling

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| retry_same_hp | registry_only | v1.1 | phase-10 | 재시도 정책 |
| retry_new_seed | registry_only | v1.1 | phase-10 | seed 재시도 |
| clip_grad_and_retry | future | v1.1 | phase-10 | deep 전용 gradient clip |
| fallback_to_safe_hp | operational | - | - | 이미 완료 |
| mark_fail | operational | - | - | 이미 완료 |

### 3.6 feature construction

#### 3.6.1 feature_builder_type

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| AR_only | operational | - | - | autoreg_lagged_target |
| factors_plus_AR | operational | - | - | 이미 완료 |
| raw_X_plus_AR | operational | - | - | raw_feature_panel |
| raw_X_only | operational | - | - | 이미 완료 |
| sequence_tensor | future | v1.0 | phase-05a | deep 모델 진입 |
| grouped_features | absent | v1.1 | phase-10 | grouped feature builder |
| mixed_frequency_features | absent | v2 | phase-05c | MIDAS 계열 |
| interaction_features | absent | v1.1 | phase-10 | interaction builder |
| calendar_augmented_features | absent | v1.1 | phase-10 | calendar 확장 |

#### 3.6.2 y_lag_count

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fixed | operational | - | - | 이미 완료 |
| cv_select | planned | v1.0 | phase-03 | CV 기반 선택 |
| IC_select | operational | - | - | 이미 완료 |
| model_specific | registry_only | v1.1 | phase-10 | 모델별 lag 정책 |

#### 3.6.3 factor_count

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fixed | operational | - | - | 이미 완료 |
| cv_select | operational | - | - | 이미 완료 |
| variance_explained_rule | registry_only | v1.1 | phase-10 | ve rule 승격 |
| BaiNg_rule | operational | - | - | 이미 완료 |
| model_specific | registry_only | v1.1 | phase-10 | 모델별 factor |

#### 3.6.4 lookback

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| fixed_lookback | operational | - | - | 이미 완료 |
| horizon_specific_lookback | planned | v1.0 | phase-03 | horizon별 lookback |
| target_specific_lookback | registry_only | v1.1 | phase-10 | target별 lookback |
| cv_select_lookback | registry_only | v1.1 | phase-10 | CV 선택 lookback |

### 3.7 compute/provenance/execution

#### 3.7.1 logging_level

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| silent | operational | - | - | 이미 완료 |
| info | operational | - | - | 이미 완료 |
| debug | planned | v1.0 | phase-03 | debug 로깅 활성 |
| trace | registry_only | v1.1 | phase-10 | trace 승격 |

#### 3.7.2 checkpointing

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| per_model | planned | v1.0 | phase-03 | 모델별 checkpoint |
| per_horizon | planned | v1.0 | phase-03 | horizon별 checkpoint |
| per_date | registry_only | v1.1 | phase-10 | 날짜별 checkpoint |
| per_trial | registry_only | v1.1 | phase-10 | trial별 checkpoint |

#### 3.7.3 cache

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| no_cache | operational | - | - | 이미 완료 |
| data_cache | operational | - | - | 이미 완료 |
| feature_cache | planned | v1.0 | phase-03 | feature cache 활성 |
| fold_cache | registry_only | v1.1 | phase-10 | fold cache 승격 |
| prediction_cache | registry_only | v1.1 | phase-10 | prediction cache |

#### 3.7.4 execution_backend

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| local_cpu | operational | - | - | 이미 완료 |
| local_gpu | future | v2 | phase-11 | GPU 백엔드 승격 |
| ray | future | v2 | phase-11 | Ray 백엔드 |
| dask | future | v2 | phase-11 | Dask 백엔드 |
| joblib | planned | v1.0 | phase-03 | joblib 병렬 |
| slurm | absent | v2 | phase-11 | SLURM 백엔드 |

---

## Layer 4: Evaluation (~110 values)

### 4.1 metrics

#### 4.1.1 point_forecast_metrics

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| MSE | operational | - | - | 이미 완료 |
| MSFE | operational | - | - | 이미 완료 |
| RMSE | operational | - | - | 이미 완료 |
| MAE | operational | - | - | 이미 완료 |
| MAPE | operational | - | - | 이미 완료 |
| sMAPE | registry_only | v1.0 | phase-04 | 기본 지표 활성 |
| MASE | registry_only | v1.0 | phase-04 | naive 상대 지표 |
| RMSSE | registry_only | v1.1 | phase-10 | RMSSE 승격 |
| MedAE | registry_only | v1.1 | phase-10 | 중앙값 지표 |
| Huber_loss | registry_only | v1.1 | phase-10 | robust 지표 |
| QLIKE | registry_only | v2 | phase-11 | 변동성 지표 |
| TheilU | registry_only | v1.1 | phase-10 | Theil U 승격 |

#### 4.1.2 relative_metrics

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| relative_MSFE | operational | - | - | 이미 완료 |
| relative_RMSE | operational | - | - | 이미 완료 |
| relative_MAE | operational | - | - | 이미 완료 |
| oos_R2 | operational | - | - | 이미 완료 |
| benchmark_win_rate | operational | - | - | 이미 완료 |
| CSFE_difference | operational | - | - | 이미 완료 |

#### 4.1.3 direction_event_metrics

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| directional_accuracy | operational | - | - | 이미 완료 |
| sign_accuracy | operational | - | - | 이미 완료 |
| turning_point_accuracy | registry_only | v1.1 | phase-10 | turning-point 승격 |
| precision | registry_only | v1.1 | phase-10 | classification 보조 |
| recall | registry_only | v1.1 | phase-10 | classification 보조 |
| F1 | registry_only | v1.1 | phase-10 | classification 보조 |
| balanced_accuracy | registry_only | v1.1 | phase-10 | imbalance 대응 |
| AUC | registry_only | v1.1 | phase-10 | AUC 승격 |
| Brier_score | registry_only | v1.1 | phase-10 | probability calibration |

#### 4.1.4 quantile_interval_density_metrics

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| pinball_loss | registry_only | v1.1 | phase-10 | quantile wave |
| CRPS | registry_only | v1.1 | phase-10 | density wave |
| interval_score | registry_only | v1.1 | phase-10 | interval wave |
| coverage_rate | registry_only | v1.1 | phase-10 | coverage 지표 |
| winkler_score | registry_only | v1.1 | phase-10 | winkler 지표 |
| log_score | future | v2 | phase-11 | 확률모델 필요 |
| NLL | future | v2 | phase-11 | BNN/MDN 필요 |
| PIT_based_metric | future | v2 | phase-11 | PIT 필요 |

#### 4.1.5 economic_decision_metrics

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| utility_gain | future | v1.1 | phase-10 | 경제 지표 승격 |
| certainty_equivalent | future | v1.1 | phase-10 | CE 계산 |
| portfolio_SR_if_finance | future | v2 | phase-11 | 금융 전용 |
| cost_sensitive_loss | future | v1.1 | phase-10 | cost-sensitive |
| policy_loss | future | v2 | phase-11 | policy eval |
| turning_point_value | future | v2 | phase-11 | turning-point 경제지 |

### 4.2 benchmark

#### 4.2.1 benchmark_model

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| historical_mean | absent | v1.0 | phase-04 | Phase 4 benchmark 핵심 |
| rolling_mean | absent | v1.0 | phase-04 | Phase 4 benchmark |
| random_walk | absent | v1.0 | phase-04 | Phase 4 benchmark |
| AR_BIC | absent | v1.0 | phase-04 | BIC 기반 AR |
| AR_fixed_p | absent | v1.0 | phase-04 | fixed-p AR |
| ARDI | absent | v1.0 | phase-04 | ARDI benchmark |
| factor_model | absent | v1.0 | phase-04 | factor benchmark |
| survey_forecast | absent | v1.1 | phase-10 | SPF/consensus |
| paper_benchmark | absent | v1.0 | phase-04 | 논문 벤치마크 |
| multi_benchmark_suite | absent | v1.0 | phase-04 | suite 구성 |

#### 4.2.2 benchmark_estimation_window

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| expanding | operational | - | - | benchmark_window=expanding |
| rolling | operational | - | - | benchmark_window=rolling |
| fixed | planned | v1.0 | phase-04 | fixed window 활성 |
| paper_exact_window | registry_only | v1.0 | phase-04 | replication 매칭 |

#### 4.2.3 benchmark_by_target_horizon

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| same_for_all | operational | - | - | 이미 완료 (benchmark_scope) |
| target_specific | planned | v1.0 | phase-04 | target별 benchmark |
| horizon_specific | planned | v1.0 | phase-04 | horizon별 benchmark |
| target_horizon_specific | registry_only | v1.1 | phase-10 | 쌍별 benchmark |

### 4.3 aggregation/reporting

#### 4.3.1 aggregation_over_time

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| full_oos_average | operational | - | - | 이미 완료 |
| rolling_average | planned | v1.0 | phase-04 | rolling 집계 |
| regime_subsample_average | planned | v1.0 | phase-08 | regime 집계 |
| event_window_average | registry_only | v1.1 | phase-10 | event window 승격 |
| pre_post_break_average | planned | v1.0 | phase-08 | break 기반 집계 |

#### 4.3.2 aggregation_over_horizons

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| equal_weight | operational | - | - | 이미 완료 |
| short_horizon_weighted | registry_only | v1.1 | phase-10 | horizon 가중 |
| long_horizon_weighted | registry_only | v1.1 | phase-10 | horizon 가중 |
| report_separately_only | planned | v1.0 | phase-04 | 분리 리포트 |

#### 4.3.3 aggregation_over_targets

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| equal_weight | planned | v1.0 | phase-04 | target 평균 |
| scale_adjusted_weight | registry_only | v1.1 | phase-10 | scale-adjusted |
| economic_priority_weight | future | post-v2 | - | 경제 우선순위 |
| report_separately_only | operational | - | - | 이미 완료 |

#### 4.3.4 ranking_rule

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| mean_metric_rank | planned | v1.0 | phase-08 | Phase 8 ranking |
| median_metric_rank | planned | v1.0 | phase-08 | Phase 8 ranking |
| win_count | planned | v1.0 | phase-08 | Phase 8 ranking |
| benchmark_beat_freq | planned | v1.0 | phase-08 | Phase 8 ranking |
| MCS_inclusion_priority | planned | v1.0 | phase-08 | MCS 연동 |
| stability_weighted_rank | registry_only | v1.1 | phase-10 | stability 승격 |
| ensemble_selection_rank | future | v2 | phase-11 | ensemble 선택 |

#### 4.3.5 report_style

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| tidy_dataframe | operational | - | - | 이미 완료 |
| latex_table | planned | v1.0 | phase-08 | Phase 8 latex |
| markdown_table | planned | v1.0 | phase-08 | Phase 8 markdown |
| plot_dashboard | registry_only | v1.1 | phase-10 | dashboard 승격 |
| paper_ready_bundle | registry_only | v1.0 | phase-08 | Phase 8 bundle |

### 4.4 regime/conditional

#### 4.4.1 regime_definition

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | 이미 완료 |
| NBER_recession | operational | - | - | 이미 완료 |
| quantile_uncertainty | registry_only | v1.1 | phase-10 | Phase 10 regime |
| financial_stress | registry_only | v1.1 | phase-10 | Phase 10 regime |
| volatility_regime | registry_only | v1.1 | phase-10 | Phase 10 regime |
| Markov_switching_regime | future | v2 | phase-11 | MS regime |
| clustering_regime | future | v2 | phase-11 | clustering regime |
| user_defined_regime | operational | - | - | 이미 완료 |

#### 4.4.2 regime_use

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| eval_only | operational | - | - | 이미 완료 |
| train_only | registry_only | v1.1 | phase-10 | train only path |
| train_and_eval | registry_only | v1.1 | phase-10 | 양방향 regime |
| regime_specific_model | future | v2 | phase-11 | regime 모델 |
| regime_interaction_features | future | v2 | phase-11 | regime 교호작용 |

#### 4.4.3 regime_metrics

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| all_main_metrics_by_regime | operational | - | - | 이미 완료 |
| regime_transition_performance | registry_only | v1.0 | phase-08 | Phase 8 aggregation |
| crisis_period_gain | operational | - | - | 이미 완료 |
| state_dependent_oos_r2 | operational | - | - | 이미 완료 |

### 4.5 decomposition layer (§4.5, macrocast identity)

#### 4.5.1 decomposition_target

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| nonlinearity_effect | registry_only | v1.0 | phase-07 | Phase 7 v0.9 |
| regularization_effect | registry_only | v1.0 | phase-07 | Phase 7 v0.9 |
| cv_scheme_effect | registry_only | v1.0 | phase-07 | Phase 7 v0.9 |
| loss_function_effect | registry_only | v1.0 | phase-07 | Phase 7 v0.9 |
| preprocessing_effect | planned | v1.0 | phase-07 | Phase 7 v0.9 |
| feature_builder_effect | planned | v1.0 | phase-07 | Phase 7 v0.9 |
| benchmark_effect | planned | v1.0 | phase-07 | Phase 7 v0.9 |
| importance_method_effect | registry_only | v1.1 | phase-10 | importance decomp |

#### 4.5.2 decomposition_order

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| marginal_effect_only | planned | v1.0 | phase-07 | Phase 7 v0.9 |
| two_way_interaction | registry_only | v1.1 | phase-10 | 2-way 승격 |
| three_way_interaction | future | v2 | phase-11 | 3-way 비용 |
| full_factorial | future | v2 | phase-11 | full-factorial 확장 |
| Shapley_style_effect_decomp | future | v2 | phase-11 | Shapley decomp |

---

## Layer 5: Output / Provenance (~40 values)

### 5.1 saved_objects

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| raw_predictions | operational | - | - | predictions_only 포함 |
| fold_predictions | operational | - | - | full_bundle 포함 |
| residuals | operational | - | - | predictions_and_metrics 포함 |
| loss_series | operational | - | - | predictions_and_metrics 포함 |
| selected_hp | operational | - | - | full_bundle 포함 |
| fitted_metadata | operational | - | - | full_bundle 포함 |
| feature_metadata | planned | v1.0 | phase-03 | feature metadata 승격 |
| importance_outputs | planned | v1.0 | phase-07 | Phase 7 동반 |
| test_outputs | operational | - | - | full_bundle 포함 |
| plots | registry_only | v1.1 | phase-10 | 플롯 아티팩트 |
| paper_tables | planned | v1.0 | phase-08 | Phase 8 paper bundle |

### 5.2 provenance_fields

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| config_hash | operational | - | - | standard/full 포함 |
| recipe_name | operational | - | - | minimal 포함 |
| dataset_version | operational | - | - | standard 포함 |
| vintage_id | planned | v1.0 | phase-02 | vintage 경로와 연동 |
| sample_period | operational | - | - | standard 포함 |
| seed | operational | - | - | minimal 포함 |
| git_commit | operational | - | - | standard 포함 |
| package_version | operational | - | - | full 포함 |
| runtime_env | operational | - | - | full 포함 |
| failure_log | planned | v1.0 | phase-03 | 실패 로그 활성 |

### 5.3 export_format

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| parquet | operational | - | - | 이미 완료 |
| csv | operational | - | - | 이미 완료 |
| json | operational | - | - | 이미 완료 |
| yaml | absent | v1.0 | phase-03 | yaml export 필요 |
| pickle | absent | v1.1 | phase-10 | 보안 검토 후 |
| feather | absent | v1.1 | phase-10 | feather 선택지 |
| latex | absent | v1.0 | phase-08 | Phase 8 bundle |
| html_report | absent | v1.1 | phase-10 | html report |
| pdf_report | absent | v2 | phase-11 | pdf report |

### 5.4 artifact_granularity

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| one_file_per_run | operational | - | - | aggregated 매핑 |
| one_file_per_target | planned | v1.0 | phase-03 | per_target 활성 |
| one_file_per_model | absent | v1.1 | phase-10 | 모델별 파일 |
| one_file_per_horizon | planned | v1.0 | phase-03 | per_target_horizon 승격 |
| one_file_per_layer_output | planned | v1.1 | phase-10 | hierarchical 경로 |

---

## Summary — Part 2 row counts

| Layer | Axis group | Rows |
|-------|------------|-----:|
| 3.1 | forecasting framework (5 sub-axes) | 27 |
| 3.2 | train-validation (3 sub-axes) | 17 |
| 3.3 | split (3 sub-axes) | 21 |
| 3.4 | models (8 sub-axes) | 79 |
| 3.5 | hyperparameter tuning (7 sub-axes) | 41 |
| 3.6 | feature construction (4 sub-axes) | 22 |
| 3.7 | compute/provenance/execution (4 sub-axes) | 20 |
| **Layer 3 subtotal** | | **227** |
| 4.1 | metrics (5 sub-axes) | 41 |
| 4.2 | benchmark (3 sub-axes) | 18 |
| 4.3 | aggregation/reporting (5 sub-axes) | 26 |
| 4.4 | regime/conditional (3 sub-axes) | 17 |
| 4.5 | decomposition (2 sub-axes) | 13 |
| **Layer 4 subtotal** | | **115** |
| 5.1 | saved_objects | 11 |
| 5.2 | provenance_fields | 10 |
| 5.3 | export_format | 9 |
| 5.4 | artifact_granularity | 5 |
| **Layer 5 subtotal** | | **35** |
| **Part 2 total** | | **377** |

### Status distribution (Part 2)

- `operational` rows: roughly Layer 3 = 66, Layer 4 = 27, Layer 5 = 19 → ~112 / 377 (~30%)
- `planned` rows: Layer 3 = 24, Layer 4 = 20, Layer 5 = 9 → ~53 / 377 (~14%)
- `registry_only` rows: Layer 3 = 47, Layer 4 = 38, Layer 5 = 2 → ~87 / 377 (~23%)
- `future` rows: Layer 3 = 11, Layer 4 = 19, Layer 5 = 0 → ~30 / 377 (~8%)
- `absent` rows: Layer 3 = 79, Layer 4 = 11, Layer 5 = 5 → ~95 / 377 (~25%)

### Phase hotspots (Part 2)

- **Phase 4 (benchmark suite)**: Layer 4.2 benchmark_model (10 values absent), AR_BIC/AR_fixed_p/paper_benchmark, benchmark_estimation_window fixed/paper_exact, benchmark_by_target_horizon target/horizon_specific; Layer 3.4.1 naive/benchmark models (historical_mean/rolling_mean/random_walk/drift/seasonal_naive/ARDI/ETS/ARIMA), statsmodels_adapter. Heaviest phase in Part 2.
- **Phase 5a (deep extras)**: Layer 3.4.1 VAR/BVAR, Layer 3.4.5 DeepMLP/LSTM/GRU/TCN, Layer 3.4.8 pytorch_adapter, Layer 3.6.1 sequence_tensor, Layer 3.5.3 max_epochs (companion).
- **Phase 5b (v1.1 deep)**: FAVAR, DFM, Transformer_encoder, Informer, NBEATS, TFT, seq2seq_rnn.
- **Phase 5c (v2 state-space/mixed-freq)**: state_space, TVP_AR, MIDAS, U_MIDAS, UnobservedComponents, TVP_Ridge, mixed_frequency_features.
- **Phase 7 (§4.5 decomposition v0.9)**: Layer 4.5.1 all 7 operational-track decomposition_target values + decomposition_order marginal_effect_only; Layer 5.1 importance_outputs companion.
- **Phase 8 (ranking/reporting)**: Layer 4.3.4 ranking (5 planned), Layer 4.3.5 latex_table/markdown_table/paper_ready_bundle, Layer 4.4.3 regime_transition_performance, Layer 4.3.1 regime_subsample_average/pre_post_break_average, Layer 5.1 paper_tables, Layer 5.3 latex.
- **Phase 10 v1.1**: Layer 4.1.4 full quantile/density family (pinball/CRPS/interval_score/coverage_rate/winkler), Layer 4.1.5 utility_gain/certainty_equivalent/cost_sensitive_loss, Layer 4.4.1 quantile_uncertainty/financial_stress/volatility_regime.
- **Phase 11 v2**: Layer 3.4.6 panel/spatial/hierarchical/GNN (8 absent), Layer 3.4.7 probabilistic/quantile (8 absent), Layer 3.7.4 ray/dask/local_gpu/slurm, regime_specific_model, Markov_switching_regime, Shapley decomp.
- **deferred-indef**: foundation_model_adapter, jax_adapter, R_adapter, external_binary_adapter.
- **post-v2**: economic_utility, custom_loss tuning_objective; conditional_space/hierarchical_space hp_space_style; economic_priority_weight agg_target.

### Registry status caveats

- Layer 5.1 `saved_objects` registry currently uses a *profile-style* enum (`none/predictions_only/predictions_and_metrics/full_bundle/models_only/data_only`) rather than the per-artifact flags in the spec (`raw_predictions/fold_predictions/...`). Ledger rows are mapped conceptually (which profile already contains each artifact). A refactor to flag-style is expected around Phase 3 wiring; not tracked as a separate row here.
- Layer 5.2 `provenance_fields` registry uses bundle-levels (`none/minimal/standard/full`). Spec-level fields mapped to the bundle that currently surfaces them.
- Layer 5.3 `export_format` registry exposes `json+csv` and `all` composite enums in addition to singletons; not enumerated separately in the ledger.
- Layer 4 `primary_metric` axis (7 operational values: msfe/relative_msfe/oos_r2/csfe/rmse/mae/mape) is a *selector* axis drawn from 4.1.1/4.1.2 and is not re-listed as a separate row group.
- Part 2 values universe counts the per-spec values as authored; registry coverage (47 files across training/evaluation/output) is narrower than spec since several spec values (poos_cv, BIC_selection, walk_forward_validation, event_retrain, benchmark_model entries, historical_mean benchmark, statsmodels_adapter, etc.) are marked `absent` until their phase lands.

# Coverage Ledger — Part 3 (Layer 6-7) + Summary

Continuation of `coverage_ledger_part1.md` and `coverage_ledger_part2.md`. See part 1 for legend.

Generated against server1 registry snapshot at `~/project/macroforecast/macrocast/registry/{tests,importance}/` on 2026-04-17.

Registry verification:
- `registry/tests/`: `stat_test.py` (21 entries, all operational), `dependence_correction.py` (4 entries, all operational)
- `registry/importance/`: 12 modules; `importance_method.py` (13 entries), plus decomposed axes (`shap`, `model_native`, `model_agnostic`, `scope`, `stability`, `aggregation`, `output_style`, `partial_dependence`, `local_surrogate`, `grouped`, `temporal`, `gradient_path`)

---

## Layer 6: Statistical Test Layer (~50 values)

Currently crammed into a single `stat_test` axis (21 operational entries) + `dependence_correction` axis (4 entries). Phase 2 decomposes into 8 sub-axes; Phase 2 adds `test_scope` (new axis).

### 6.1 equal_predictive_tests

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| DM | operational | - | - | `stat_test.dm` 등록 완료 |
| DM_HLN_small_sample | operational | - | - | `stat_test.dm_hln` 등록 완료 |
| modified_DM | operational | - | - | `stat_test.dm_modified` 등록 완료 (long-horizon) |
| paired_t_on_loss_diff | absent | v1.0 | phase-02 | 손실차 paired t 헬퍼 신규 |
| Wilcoxon_signed_rank | absent | v1.0 | phase-02 | 비모수 대안 신규 |

### 6.2 nested_tests

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| Clark_West | operational | - | - | `stat_test.cw` 등록 완료 |
| ENC_NEW | operational | - | - | `stat_test.enc_new` 등록 완료 |
| MSE_F | operational | - | - | `stat_test.mse_f` 등록 완료 |
| MSE_t | operational | - | - | `stat_test.mse_t` 등록 완료 |
| forecast_encompassing_nested | absent | v1.0 | phase-02 | nested encompassing 변형 신규 |

### 6.3 cpa_instability

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| Giacomini_White_CPA | operational | - | - | `stat_test.cpa` (constant-only minimal) |
| Rossi_Sekhposyan_stability | operational | - | - | `stat_test.rossi` 등록 완료 |
| rolling_DM | operational | - | - | `stat_test.rolling_dm` 등록 완료 |
| fluctuation_test | absent | v1.0 | phase-02 | Giacomini-Rossi fluctuation test 신규 |
| Chow_break_forecast | absent | v1.1 | phase-10 | 구조변화 기반 예측 break 검정 |
| CUSUM_on_loss | absent | v1.1 | phase-10 | loss-CUSUM 순차 안정성 검정 |

### 6.4 multiple_model_tests

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| White_Reality_Check | operational | - | - | `stat_test.reality_check` 등록 완료 |
| Hansen_SPA | operational | - | - | `stat_test.spa` 등록 완료 |
| MCS | operational | - | - | `stat_test.mcs` 등록 완료 |
| stepwise_MCS | absent | v1.1 | phase-10 | stepwise-M 변형 추가 |
| bootstrap_best_model_test | absent | v1.1 | phase-10 | 최적모델 bootstrap 검정 |

### 6.5 density_interval_tests

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| PIT_uniformity | planned | v1.0 | phase-02 | PIT 균일성 검정; density output 의존 |
| Berkowitz_test | absent | v1.0 | phase-02 | Berkowitz likelihood test 신규 |
| Kupiec_test | absent | v1.0 | phase-02 | Kupiec POF 검정 신규 |
| Christoffersen_unconditional | absent | v1.0 | phase-02 | Christoffersen UC 검정 신규 |
| Christoffersen_independence | absent | v1.0 | phase-02 | Christoffersen IND 검정 신규 |
| Christoffersen_conditional | absent | v1.0 | phase-02 | Christoffersen CC 검정 신규 |
| interval_coverage_test | absent | v1.0 | phase-02 | 구간 커버리지 통합 검정 |

### 6.6 direction_tests

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| Pesaran_Timmermann | operational | - | - | `stat_test.pesaran_timmermann` 등록 완료 |
| McNemar | absent | v1.0 | phase-02 | 방향 예측 쌍별 McNemar 신규 |
| binomial_hit_test | operational | - | - | `stat_test.binomial_hit` 등록 완료 |
| ROC_comparison | absent | v1.1 | phase-10 | 방향 AUC DeLong 검정 |

### 6.7 residual_diagnostics

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| Mincer_Zarnowitz | operational | - | - | `stat_test.mincer_zarnowitz` 등록 완료 |
| autocorrelation_of_errors | absent | v1.0 | phase-02 | 오차 자기상관 요약 신규 (LB와 분리) |
| Ljung_Box_on_errors | operational | - | - | `stat_test.ljung_box` 등록 완료 |
| ARCH_LM_on_errors | operational | - | - | `stat_test.arch_lm` 등록 완료 |
| bias_test | operational | - | - | `stat_test.bias_test` 등록 완료 |
| serial_dependence_loss_diff | absent | v1.0 | phase-02 | 손실차 자기상관 검정 신규 |
| diagnostics_full (bundle) | operational | - | - | `stat_test.diagnostics_full` 등록 완료 (MZ+LB+ARCH+bias) |

### 6.8 dependence_correction

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| iid | operational | - | - | `dependence_correction.none` |
| Newey_West | operational | - | - | `dependence_correction.nw_hac` (horizon bandwidth) |
| HAC_auto_bandwidth | operational | - | - | `dependence_correction.nw_hac_auto` |
| block_bootstrap | operational | - | - | `dependence_correction.block_bootstrap` |
| stationary_bootstrap | absent | v1.0 | phase-02 | Politis-Romano 고정평균 블록 길이 |
| circular_bootstrap | absent | v1.1 | phase-10 | 순환 블록 부트스트랩 |
| wild_bootstrap | absent | v1.1 | phase-10 | heteroskedasticity-robust 와일드 부트스트랩 |
| cluster_robust | absent | v2 | phase-11 | panel 클러스터 로버스트 변형 |

### 6.9 test_scope (new axis in Phase 2)

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| per_target | absent | v1.0 | phase-02 | 타깃별 실행 스코프 |
| per_horizon | absent | v1.0 | phase-02 | horizon별 실행 스코프 |
| per_model_pair | absent | v1.0 | phase-02 | 모델쌍별 스코프 (DM/CW 기본) |
| full_grid_pairwise | absent | v1.1 | phase-10 | 모든 쌍 전수 스코프 |
| benchmark_vs_all | absent | v1.0 | phase-02 | 벤치마크 대 전체 (RC/SPA/MCS 기본) |
| regime_specific_tests | absent | v2 | phase-11 | regime-조건부 스코프 |
| subsample_tests | absent | v1.1 | phase-10 | 하위표본 스코프 |

---

## Layer 7: Variable Importance / Interpretability (~100 values)

Current `importance_method` axis has 13 operational entries (none, minimal_importance, tree_shap, kernel_shap, linear_shap, permutation_importance, lime, feature_ablation, pdp, ice, ale, grouped_permutation, importance_stability). Phase 2 (current) decomposes into 12 sub-axes (scope, model_native, model_agnostic, shap, gradient_path, local_surrogate, partial_dependence, grouped, temporal, stability, aggregation, output_style). Phase 10 (v1.1) consolidates SHAP family. Phase 11 (v2) adds deep-model gradient attribution.

### 7.1 importance_scope

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| none | operational | - | - | `importance_method.none` |
| global | operational | - | - | `importance_scope.global` 등록 완료 |
| local | operational | - | - | `importance_scope.local` 등록 완료 |
| global_and_local | operational | - | - | `importance_scope.both` 등록 완료 |
| time_varying | absent | v1.1 | phase-10 | rolling/path scope 활성화 |
| regime_specific | absent | v2 | phase-11 | regime hook 의존 |
| horizon_specific | absent | v1.0 | phase-02 | horizon split scope 신규 |
| target_specific | absent | v1.0 | phase-02 | target split scope 신규 |
| cross_model_consensus | absent | v1.1 | phase-10 | 다모델 합의 scope 신규 |

### 7.2 model_native_importance

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| linear_coefficients | operational | - | - | `model_native.feature_gain` / linear path |
| standardized_coefficients | absent | v1.0 | phase-02 | 표준화 계수 추출 헬퍼 신규 |
| t_stats_if_linear | absent | v1.0 | phase-02 | OLS/GLS t-통계 추출 |
| RF_Gini_importance | operational | - | - | `model_native.feature_gain` (impurity) |
| RF_permutation_importance | operational | - | - | `model_agnostic.permutation_importance` |
| XGB_gain | operational | - | - | XGB native `feature_gain` |
| XGB_cover | absent | v1.0 | phase-02 | XGB cover metric 추출 |
| XGB_weight | absent | v1.0 | phase-02 | XGB weight metric 추출 |
| LGB_split_importance | absent | v1.1 | phase-10 | LightGBM 어댑터 의존 |
| attention_weight_proxy | absent | deferred-indef | - | 모델-의존, 표준화 어려움 |
| feature_dropout_score | absent | v2 | phase-11 | dropout 기반 feature score |

### 7.3 model_agnostic_importance

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| PMI_or_PFI_permutation_importance | operational | - | - | `model_agnostic.permutation_importance` |
| leave_one_covariate_out | absent | v1.0 | phase-02 | LOCO 신규 |
| group_permutation_importance | operational | - | - | `grouped.grouped_permutation` |
| conditional_permutation_importance | absent | v1.1 | phase-10 | 조건부 PFI 신규 |
| Sobol_sensitivity | absent | v2 | phase-11 | variance decomposition, 계산 비용 큼 |
| variance_based_sensitivity | absent | v2 | phase-11 | Sobol 일반화 |
| Shapley_value_global | absent | v1.1 | phase-10 | global Shapley 집계 |

### 7.4 SHAP_family

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| TreeSHAP | operational | - | - | `shap.tree_shap` 등록 완료 |
| KernelSHAP | operational | - | - | `shap.kernel_shap` 등록 완료 |
| DeepSHAP | absent | v2 | phase-11 | deep 모델 의존 |
| LinearSHAP | operational | - | - | `shap.linear_shap` 등록 완료 |
| GroupedSHAP | absent | v1.1 | phase-10 | SHAP unification 단계 |
| InteractionSHAP | absent | v1.1 | phase-10 | SHAP unification 단계 |
| SHAP_time_average | absent | v1.1 | phase-10 | SHAP unification 단계 |
| SHAP_regime_split | absent | v2 | phase-11 | regime hook 의존 |
| SHAP_horizon_split | absent | v1.1 | phase-10 | horizon scope 기반 |
| SHAP_target_split | absent | v1.1 | phase-10 | target scope 기반 |

### 7.5 gradient_path_methods

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| IntegratedGradients | planned | v2 | phase-11 | `gradient_path.coefficient_path` 확장; deep 모델 필요 |
| PathIntegratedGradients | absent | v2 | phase-11 | deep 모델 필요 |
| GradientXInput | absent | v2 | phase-11 | deep 모델 grad hook 필요 |
| SmoothGrad | absent | v2 | phase-11 | deep 모델 grad hook 필요 |
| ExpectedGradients | absent | v2 | phase-11 | deep 모델 grad hook 필요 |
| DeepLift | absent | v2 | phase-11 | deep 모델 grad hook 필요 |
| LRP | absent | post-v2 | - | layer-wise relevance; 구조 의존 |
| saliency_map | absent | v2 | phase-11 | deep 모델 grad hook 필요 |

### 7.6 local_surrogate_perturbation

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| LIME | operational | - | - | `local_surrogate.lime` 등록 완료 |
| local_linear_surrogate | absent | v1.1 | phase-10 | 단순 선형 서로게이트 변형 |
| counterfactual_explanation | absent | v2 | phase-11 | CF 검색 엔진 필요 |
| occlusion_importance | absent | v1.1 | phase-10 | 입력 마스킹 변형 |
| feature_ablation | operational | - | - | `local_surrogate.feature_ablation` 등록 완료 |
| masking_importance | absent | v1.1 | phase-10 | sequence 마스킹 변형 |

### 7.7 partial_dependence

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| PDP | operational | - | - | `partial_dependence.pdp` 등록 완료 |
| ICE | operational | - | - | `partial_dependence.ice` 등록 완료 |
| ALE | operational | - | - | `partial_dependence.ale` 등록 완료 |
| 2D_PDP | absent | v1.1 | phase-10 | 2변수 PDP 확장 |
| 2D_ALE | absent | v1.1 | phase-10 | 2변수 ALE 확장 |
| accumulated_local_effect_by_group | absent | v1.1 | phase-10 | 그룹별 ALE 변형 |

### 7.8 grouped_importance

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| by_FRED_category | absent | v1.1 | phase-10 | FRED 카테고리 매핑 기반 |
| by_economic_theme | absent | v1.1 | phase-10 | 경제 테마 그룹 (infl/activity 등) |
| by_variable_family | operational | - | - | `grouped.variable_root_groups` |
| by_lag_block | operational | - | - | `grouped.variable_root_groups` (lag prefix) |
| by_factor_block | absent | v1.0 | phase-02 | factor 빌더와 연계 |
| by_text_group (추정) | absent | v2 | phase-11 | text pipeline 의존 |
| by_spatial_group (추정) | absent | v2 | phase-11 | 공간 메타 의존 |
| by_time_window | absent | v1.1 | phase-10 | 시간창 그룹 |
| by_regime | absent | v2 | phase-11 | regime hook 의존 |
| by_target | absent | v1.0 | phase-02 | target split 기반 |
| by_horizon | absent | v1.0 | phase-02 | horizon split 기반 |
| by_state_or_region | absent | v2 | phase-11 | panel/regional 메타 의존 |
| custom_group_map | absent | v1.1 | phase-10 | 사용자 정의 매핑 API |

### 7.9 sequence_temporal_importance

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| time_step_importance | absent | v1.1 | phase-10 | sequence 모델 (Phase 5) 의존 |
| feature_time_heatmap | absent | v1.1 | phase-10 | sequence 모델 의존 |
| attention_rollout | absent | v2 | phase-11 | attention 기반 모델 필요 |
| temporal_occlusion | absent | v1.1 | phase-10 | sequence 마스킹 |
| temporal_IG | absent | v2 | phase-11 | IG + sequence 조합 |
| window_importance | absent | v1.1 | phase-10 | 윈도 중요도 |
| lag_saliency_profile | absent | v1.1 | phase-10 | lag별 saliency |

### 7.10 stability_of_importance

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| bootstrap_rank_stability | operational | - | - | `stability.bootstrap_rank_stability` |
| seed_stability | operational | - | - | `stability.seed_stability` |
| window_stability | absent | v1.1 | phase-10 | 윈도 일관성 |
| vintage_stability | absent | v1.1 | phase-10 | vintage 교차 안정성 |
| model_consensus_importance | absent | v1.1 | phase-10 | 다모델 합의 |
| rank_correlation_across_runs | absent | v1.0 | phase-02 | Spearman/Kendall 변형 |
| sign_consistency | absent | v1.0 | phase-02 | 부호 일관성 |
| importance_stability (bundle) | operational | - | - | `stability.importance_stability` |

### 7.11 importance_aggregation

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| mean_abs_importance | operational | - | - | `aggregation.mean_abs` |
| median_abs_importance | absent | v1.1 | phase-10 | 중앙 절댓값 집계 |
| signed_mean_importance | operational | - | - | `aggregation.mean_signed` |
| rank_average | absent | v1.1 | phase-10 | 순위 평균 |
| top_k_frequency | operational | - | - | `aggregation.top_k` |
| stability_weighted_rank | absent | v1.1 | phase-10 | stability-가중 순위 |
| group_share_of_total_importance | absent | v1.0 | phase-02 | 그룹 점유율 집계 |

### 7.12 output_style

| Value | Current | Target version | Target phase | Rationale |
|-------|---------|:---:|:---:|-----------|
| bar_plot | operational | - | - | `output_style.ranked_table` (기본 플롯 포함) |
| heatmap | absent | v1.1 | phase-10 | heatmap 렌더러 |
| waterfall | absent | v1.1 | phase-10 | SHAP waterfall |
| beeswarm | absent | v1.1 | phase-10 | SHAP beeswarm |
| time_series_plot | absent | v1.1 | phase-10 | 시간축 플롯 |
| regime_comparison_plot | absent | v2 | phase-11 | regime 분할 플롯 |
| category_stack_plot | absent | v1.1 | phase-10 | 카테고리 스택 |
| dashboard | absent | v2 | phase-11 | 통합 대시보드 |
| paper_table | absent | v1.0 | phase-02 | 논문용 테이블 포맷 |
| ranked_table | operational | - | - | `output_style.ranked_table` |
| curve_bundle | operational | - | - | `output_style.curve_bundle` |
| nested_report | operational | - | - | `output_style.nested_report` |

---

## Grand Summary

Counts aggregated across Parts 1-3 (Layer 0-7). Each row in each Part's tables counts once. Bundle rows (e.g. `diagnostics_full`, `importance_stability`) are counted separately from their component entries. Approximate totals:

| Bucket | Count | % of universe |
|--------|:-----:|:---:|
| operational (pre-plan) | ~225 | ~24% |
| v1.0 (phase 0-9) adds | ~140 | ~15% |
| v1.1 (phase 10) adds | ~215 | ~23% |
| v2 (phase 11) adds | ~195 | ~21% |
| post-v2 | ~85 | ~9% |
| deferred-indef | ~50 | ~5% |
| remaining (registry_only / planned / future not yet phase-tagged) | ~30 | ~3% |
| **Total in user spec** | **~940** | **100%** |

### Per-Part row tally

| Part | Layers | Rows | Markdown lines |
|------|--------|:---:|:---:|
| Part 1 | Layer 0-2 | 403 | 780 |
| Part 2 | Layer 3-5 | 377 | 765 |
| Part 3 | Layer 6-7 + Summary | 163 | see tail |
| **Grand total (approx)** | **0-7** | **~943** | **~2000+** |

### Per-Layer row tally (Part 3 detail)

| Layer | Sub-axes | Rows |
|-------|:---:|:---:|
| 6.1 equal_predictive_tests | 1 | 5 |
| 6.2 nested_tests | 1 | 5 |
| 6.3 cpa_instability | 1 | 6 |
| 6.4 multiple_model_tests | 1 | 5 |
| 6.5 density_interval_tests | 1 | 7 |
| 6.6 direction_tests | 1 | 4 |
| 6.7 residual_diagnostics | 1 | 7 |
| 6.8 dependence_correction | 1 | 8 |
| 6.9 test_scope | 1 | 7 |
| **Layer 6 subtotal** | **9** | **54** |
| 7.1 importance_scope | 1 | 9 |
| 7.2 model_native_importance | 1 | 11 |
| 7.3 model_agnostic_importance | 1 | 7 |
| 7.4 SHAP_family | 1 | 10 |
| 7.5 gradient_path_methods | 1 | 8 |
| 7.6 local_surrogate_perturbation | 1 | 6 |
| 7.7 partial_dependence | 1 | 6 |
| 7.8 grouped_importance | 1 | 13 |
| 7.9 sequence_temporal_importance | 1 | 7 |
| 7.10 stability_of_importance | 1 | 8 |
| 7.11 importance_aggregation | 1 | 7 |
| 7.12 output_style | 1 | 13 |
| **Layer 7 subtotal** | **12** | **105** |
| **Part 3 total** | **21 axes** | **159** |

### Bucket distribution by Layer (approx row counts)

| Layer | operational | v1.0 | v1.1 | v2 | post-v2 | deferred-indef | other |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 (meta) | ~15 | ~5 | ~3 | ~1 | 0 | 0 | ~2 |
| 1 (data) | ~65 | ~35 | ~45 | ~30 | ~15 | ~10 | ~5 |
| 2 (features) | ~40 | ~25 | ~40 | ~45 | ~25 | ~15 | ~5 |
| 3 (training) | ~40 | ~30 | ~40 | ~35 | ~15 | ~10 | ~5 |
| 4 (evaluation) | ~25 | ~15 | ~30 | ~20 | ~10 | ~5 | ~3 |
| 5 (output) | ~15 | ~10 | ~20 | ~20 | ~15 | ~5 | ~3 |
| 6 (tests) | ~21 | ~17 | ~9 | ~2 | 0 | 0 | ~5 |
| 7 (importance) | ~22 | ~13 | ~28 | ~28 | ~5 | ~5 | ~4 |
| **Total** | **~243** | **~150** | **~215** | **~181** | **~85** | **~50** | **~32** |

Note: per-layer totals are approximate; some Layer-1 through Layer-3 rows hit two buckets (registry_only current + v1.x target). Grand summary uses target-bucket assignment (i.e. each row counted under its Target column).

---

End of Coverage Ledger (Parts 1-3).
