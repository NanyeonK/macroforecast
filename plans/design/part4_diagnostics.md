# macrocast 설계도 — Part 4

L1.5 + L2.5 + L3.5 + L4.5 (diagnostic layer)

---

## Diagnostic 공통 원칙

4개 layer 모두 공유.

| 항목 | 값 |
|---|---|
| Category | diagnostic |
| Default state | enabled: false (명시적 opt-in 필요) |
| Non-blocking | construction layer DAG에 영향 없음 |
| Sweep | 모든 axis sweepable=false |
| UI mode | list |
| Sink type | DiagnosticArtifact |

### 공통 Z sub-layer

4개 layer 모두 마지막 slot으로 자동 포함.

| Axis | Default |
|---|---|
| diagnostic_format | pdf (옵션: png/pdf/html/json/latex_table/csv/multi) |
| attach_to_manifest | true |
| figure_dpi | 300 |
| latex_export | true |

`attach_to_manifest=true` → L8 manifest의 diagnostics/ 섹션에 등록.

### L8 통합

L8.B saved_objects에서 개별 또는 shortcut 선택.

| 옵션 | 의미 |
|---|---|
| diagnostics_l1_5 | L1.5만 |
| diagnostics_l2_5 | L2.5만 |
| diagnostics_l3_5 | L3.5만 |
| diagnostics_l4_5 | L4.5만 |
| diagnostics_all | 4개 모두 (active 한정) |

### Default 사용 패턴

| 진단 | 의도 |
|---|---|
| L1.5 | raw data 품질 검사 |
| L2.5 | preprocessing 효과 확인 |
| L3.5 | feature engineering 효과 확인 |
| L4.5 | model fit 품질 검사 |

전체 진단을 다 켜면 비싸다. 단계별로 의심 가는 부분만 활성화 권장.

---

## L1.5: Data Summary

### 답하는 질문

raw panel의 *데이터 품질*은 어떤가.

L1 sink를 hook하여 inspection. Cleaning 전 단계.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l1_data_definition_v1 |
| 만드는 sink | l1_5_diagnostic_v1 |
| 다음 사용처 | L8 (diagnostics_l1_5) |

### Sub-layer

| Slot | 이름 | 답하는 질문 |
|---|---|---|
| L1.5.A | Sample coverage | 각 series가 언제부터 언제까지인가 |
| L1.5.B | Univariate summary | 각 series의 기술 통계 |
| L1.5.C | Stationarity tests | (optional) 정상성 검정 |
| L1.5.D | Missing & outlier audit | NaN, outlier 분포 |
| L1.5.E | Correlation pre-cleaning | (optional) raw correlation matrix |
| L1.5.Z | Diagnostic export | 공통 |

### Axis 선택지 (요약)

**coverage_view (L1.5.A)**

| 선택지 | 의미 |
|---|---|
| multi (default) | 모두 포함 |
| per_series_start_end | series별 시작/끝 |
| panel_balance_matrix | T × N boolean 행렬 |
| observation_count | series별 obs 수 |

**summary_metrics (L1.5.B)**

multi-select. Default `[mean, sd, min, max, n_missing]`. 옵션: mean, sd, min, max, skew, kurtosis, n_obs, n_missing.

**summary_split (L1.5.B)**

| 선택지 | 의미 |
|---|---|
| full_sample (default) | 전체 |
| pre_oos_only | OOS 이전만 |
| per_decade | decade별 |
| per_regime | regime별 (regime != none 시) |

**stationarity_test (L1.5.C)**

| 선택지 | 의미 |
|---|---|
| none (default) | 검정 안 함 |
| adf / pp / kpss / multi | 각 검정 |

**stationarity_test_scope (L1.5.C)**

`target_only` / `predictors_only` / `target_and_predictors` (default).

**missing_view, outlier_view (L1.5.D)**

| missing_view 선택지 | outlier_view 선택지 |
|---|---|
| multi (default) | iqr_flag (default) |
| heatmap | none |
| per_series_count | zscore_flag |
| longest_gap | multi |

leaf_config: outlier_threshold_iqr (default 10.0, McCracken-Ng), outlier_zscore_threshold (default 3.0).

**correlation_view (L1.5.E)**

| 선택지 | 의미 |
|---|---|
| none (default) | correlation 안 함 |
| full_matrix / clustered_heatmap / top_k_per_target | 표시 방식 |

raw panel은 NaN이 많아 correlation default가 none.

### Gate 흐름

| 조건 | 결과 |
|---|---|
| L1.5.enabled = false | 모든 axis 비활성, sink 비어 있음 |
| stationarity_test = none | stationarity_test_scope 비활성 |
| summary_split = per_regime + L1.G regime = none | hard error |
| correlation_view = top_k_per_target | leaf_config.correlation_top_k 필요 |

### Layer 간 관계

L1만 hook. 단순.

### 함정

- Default off. 명시 필요.
- correlation_view default가 none (다른 *_view들과 다름). Raw panel correlation은 NaN dominant라 의미 적음.
- outlier_view는 *진단용*. L2.C와 다름. L2.C는 cleaning, L1.5.D는 inspection.
- per_regime split은 regime 활성 필요.

### Sample

```yaml
1_5_data_summary:
  enabled: true
  fixed_axes: {}
```

```yaml
1_5_data_summary:
  enabled: true
  fixed_axes:
    summary_metrics: [mean, sd, skew, kurtosis, n_missing]
    summary_split: per_decade
    stationarity_test: adf
    correlation_view: clustered_heatmap
```

---

## L2.5: Pre vs Post Preprocessing

### 답하는 질문

L1 raw와 L2 cleaned가 *어떻게 다른가*. Preprocessing이 무엇을 했는가.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l1_data_definition_v1, l2_clean_panel_v1 (multi_stage 시 L2 intermediate) |
| 만드는 sink | l2_5_diagnostic_v1 |
| 다음 사용처 | L8 |

### Sub-layer

| Slot | 이름 | 답하는 질문 |
|---|---|---|
| L2.5.A | Comparison axis | 어느 단계를 비교할 것인가 |
| L2.5.B | Distribution shift | 분포가 어떻게 바뀌었나 |
| L2.5.C | Correlation shift | (optional) correlation 변화 |
| L2.5.D | Cleaning effect summary | imputation/outlier/truncation 통계 |
| L2.5.Z | Diagnostic export | 공통 |

### Axis 선택지 (요약)

**comparison_pair (L2.5.A)**

| 선택지 | 의미 |
|---|---|
| raw_vs_final_clean (default) | L1 raw vs L2 final |
| raw_vs_tcoded | L1 vs L2.B 후 |
| raw_vs_outlier_handled | L1 vs L2.C 후 |
| raw_vs_imputed | L1 vs L2.D 후 |
| multi_stage | 4단계 모두 |

`multi_stage`는 L2 intermediate sink 필요 (runtime hook).

**comparison_output_form (L2.5.A)**

`side_by_side_summary` / `overlay_timeseries` / `difference_table` / `distribution_shift` / `multi` (default).

**distribution_metric (L2.5.B)**

multi-select list. Default `[mean_change, sd_change, ks_statistic]`. 옵션: mean_change, sd_change, skew_change, kurtosis_change, ks_statistic.

**distribution_view (L2.5.B)**

`summary_table` / `qq_plot` / `histogram_overlay` / `multi` (default).

**correlation_shift (L2.5.C)**

| 선택지 | 의미 |
|---|---|
| none (default) | correlation 비교 안 함 |
| delta_matrix | post − pre matrix |
| pre_post_overlay | side-by-side |

**cleaning_summary_view (L2.5.D)**

`n_imputed_per_series` / `n_outliers_flagged` / `n_truncated_obs` / `multi` (default).

**t_code_application_log (L2.5.D)**

| 선택지 | 의미 |
|---|---|
| none | log 없음 |
| summary (default) | 요약 통계 |
| per_series_detail | series별 적용된 tcode + 효과 |

### Gate 흐름

| 조건 | 결과 |
|---|---|
| L2.5.enabled = false | 모든 axis 비활성 |
| correlation_shift = none | correlation_method 비활성 |
| distribution_metric에 잘못된 값 | hard error |

### Layer 간 관계

L1 + L2를 hook. multi_stage 시 L2 intermediate sinks 필요.

### 함정

- multi_stage는 schema 지원, runtime은 L2 pipeline hook 필요 (execution PR).
- correlation_shift default none. raw panel correlation 자체가 의미 작음.
- t_code_application_log는 *L2가 무엇을 했는지* 추적용. L2.B 결정과 별개.

### Sample

```yaml
2_5_pre_post_preprocessing:
  enabled: true
  fixed_axes: {}
```

```yaml
2_5_pre_post_preprocessing:
  enabled: true
  fixed_axes:
    comparison_pair: multi_stage
    distribution_metric: [mean_change, sd_change, ks_statistic]
    correlation_shift: delta_matrix
    t_code_application_log: per_series_detail
```

---

## L3.5: Feature Diagnostics

### 답하는 질문

L3 feature engineering이 *raw/cleaned 대비 어떻게 다른가*. Factor/lag/selection이 적절한가.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l1_data_definition_v1, l2_clean_panel_v1, l3_features_v1, l3_metadata_v1 |
| 만드는 sink | l3_5_diagnostic_v1 |
| 다음 사용처 | L8 |

L3.5는 *3-stage 비교* 가능. 매크로 forecasting paper 표준.

### Sub-layer

| Slot | 이름 | Gate |
|---|---|---|
| L3.5.A | Comparison axis | always |
| L3.5.B | Factor block inspection | L3에 factor reduction step 있을 때 |
| L3.5.C | Feature correlation | optional |
| L3.5.D | Lag block inspection | L3에 lag step 있을 때 |
| L3.5.E | Selected features post-selection | L3.D feature_selection 사용 시 |
| L3.5.Z | Diagnostic export | 공통 |

각 sub-layer가 *L3 DAG의 step 종류*에 따라 gated.

### Axis 선택지 (요약)

**comparison_stages (L3.5.A)**

| 선택지 | 의미 |
|---|---|
| cleaned_vs_features (default) | L2 vs L3 |
| raw_vs_cleaned_vs_features | 3-stage |
| features_only | L3만 |

**comparison_output_form (L3.5.A)**

`side_by_side` / `dimension_summary` / `distribution_shift` / `multi` (default).

**factor_view (L3.5.B)**

L3에 pca/dfm/scaled_pca/sparse_pca/varimax/pls/random_projection 있을 때만 active.

| 선택지 | 의미 |
|---|---|
| multi (default) | 모두 |
| scree_plot | eigenvalue scree |
| cumulative_variance | 변동 설명도 |
| loadings_heatmap | factor × var loading |
| factor_timeseries | factor 시계열 |

leaf_config: n_factors_to_show (default 8), loading_top_k_per_factor (default 10).

**dfm_diagnostics (L3.5.B)**

dfm step 있을 때만.

| 선택지 | 의미 |
|---|---|
| multi (default) | 모두 |
| none | 안 함 |
| idiosyncratic_acf | idiosyncratic 잔차 ACF |
| factor_var_stability | factor VAR 계수 안정성 |

**feature_correlation (L3.5.C)**

| 선택지 | 의미 |
|---|---|
| cross_block (default) | L3 pipeline 블록 간 correlation |
| within_block | 블록 안 |
| with_target | target과의 correlation |
| multi | 모두 |
| none | 안 함 |

**lag_view (L3.5.D)**

L3에 lag/seasonal_lag/ma_increasing_order 있을 때만.

`autocorrelation_per_lag` / `partial_autocorrelation` / `lag_correlation_decay` / `multi` (default).

**marx_view (L3.5.D)**

ma_increasing_order step 있을 때만. MARX 고유 가중치 시각화.

| 선택지 | 의미 |
|---|---|
| weight_decay_visualization (default) | Coulombe Figure 형태 |
| none | 안 함 |

**selection_view (L3.5.E)**

feature_selection step 있을 때만.

`selected_list` / `selection_count_per_origin` / `selection_stability` / `multi` (default).

**stability_metric (L3.5.E)**

`jaccard` (default) / `kuncheva`.

### Gate 흐름

| 조건 | 결과 |
|---|---|
| L3 DAG에 factor reduction step 없음 + factor_view 설정 | hard error |
| L3 DAG에 lag step 없음 + lag_view 설정 | hard error |
| L3 DAG에 ma_increasing_order 없음 + marx_view 설정 | hard error |
| L3 DAG에 feature_selection 없음 + selection_view 설정 | hard error |
| L3 DAG에 dfm 없음 + dfm_diagnostics 설정 | hard error |
| feature_correlation = none + correlation_method 설정 | inactive |
| selection_view ∉ {selection_stability, multi} + stability_metric 설정 | inactive |

### Layer 간 관계

L1, L2, L3 모두 hook. L3 metadata로 pipeline ID 추적.

### 함정

- 모든 sub-layer가 *L3 DAG step 존재*로 gated. Validator가 L3 DAG inspection.
- 3-stage 비교는 raw_vs_cleaned_vs_features 명시 필요. Default는 cleaned_vs_features.
- MARX는 별도 axis (marx_view). 일반 lag와 weight 패턴이 다르기 때문.
- Pipeline 블록은 L3 metadata의 pipeline_id로 결정.

### Sample

```yaml
3_5_feature_diagnostics:
  enabled: true
  fixed_axes: {}
```

```yaml
3_5_feature_diagnostics:
  enabled: true
  fixed_axes:
    comparison_stages: raw_vs_cleaned_vs_features
    factor_view: multi
    dfm_diagnostics: multi
    feature_correlation: multi
    lag_view: multi
    marx_view: weight_decay_visualization
    selection_view: multi
```

---

## L4.5: Generator Diagnostics

### 답하는 질문

L4 model fit이 *건전한가*. Window stability, tuning 결과, ensemble 동작 어떤가.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l4_forecasts_v1, l4_model_artifacts_v1, l4_training_metadata_v1, l3_features_v1 |
| 만드는 sink | l4_5_diagnostic_v1 |
| 다음 사용처 | L8 |

### Sub-layer

| Slot | 이름 | Gate |
|---|---|---|
| L4.5.A | In-sample fit | always |
| L4.5.B | Forecast scale view | always |
| L4.5.C | Window stability | always |
| L4.5.D | Tuning history | L4.D search_algorithm != none |
| L4.5.E | Ensemble diagnostics | L4 ensemble combine 있을 때 |
| L4.5.Z | Diagnostic export | 공통 |

### Axis 선택지 (요약)

**fit_view (L4.5.A)**

| 선택지 | 의미 |
|---|---|
| multi (default) | 모두 |
| fitted_vs_actual | scatter / 시계열 |
| residual_time | residual 시계열 |
| residual_acf | residual ACF |
| residual_qq | residual QQ |

**fit_per_origin (L4.5.A)**

| 선택지 | 의미 |
|---|---|
| last_origin_only (default) | 가장 최근 origin (가장 cheap) |
| every_n_origins | N origin마다 |
| all_origins | 모든 origin (가장 비싸다) |

**forecast_scale_view (L4.5.B)**

| 선택지 | 의미 |
|---|---|
| both_overlay (default) | transformed + back-transformed |
| transformed_only | transformed scale만 |
| back_transformed_only | level forecast만 |

**back_transform_method (L4.5.B)**

| 선택지 | 의미 |
|---|---|
| auto (default) | L1.target_construction.mode에서 자동 감지 |
| manual_function | leaf_config 콜러블 |

**window_view (L4.5.C)**

| 선택지 | 의미 |
|---|---|
| multi (default) | 모두 |
| rolling_train_loss | window별 training loss |
| rolling_coef | rolling 계수 (linear models) |
| first_vs_last_window_forecast | 첫 vs 마지막 window forecast |
| parameter_stability | parameter SE across windows |

**coef_view_models (L4.5.C)**

L4에 linear model 있을 때만.

`all_linear_models` (default) / `user_list`.

leaf_config: coef_top_k (default 10).

**tuning_view (L4.5.D)**

L4.D search_algorithm != none일 때만.

| 선택지 | 의미 |
|---|---|
| multi (default) | 모두 |
| objective_trace | objective value 추이 |
| hyperparameter_path | hyperparameter 추이 |
| cv_score_distribution | CV score 분포 |

**ensemble_view (L4.5.E)**

L4에 ensemble combine 있을 때만.

| 선택지 | 의미 |
|---|---|
| multi (default) | 모두 |
| weights_over_time | 시간별 weight |
| weight_concentration | Gini-like concentration |
| member_contribution | 각 member 기여 |

**weights_over_time_method (L4.5.E)**

| 선택지 | 의미 |
|---|---|
| stacked_area (default) | stacked area chart |
| line_plot | 각 모델 line |
| heatmap | (model × time) heatmap |

### Gate 흐름

| 조건 | 결과 |
|---|---|
| L4.5.enabled = false | 모든 비활성 |
| L4 search_algorithm = none for all models + tuning_view 설정 | hard error |
| L4에 ensemble combine 없음 + ensemble_view 설정 | hard error |
| L4에 linear model 없음 + coef_view_models 설정 | hard error |
| ensemble_view ∉ {weights_over_time, multi} + weights_over_time_method 설정 | inactive |
| back_transform_method = manual_function + 콜러블 없음 | hard error |
| fit_per_origin = every_n_origins + leaf_config.fit_n_origins_step 없음 | default 12 적용 |

### Layer 간 관계

L4 + L3 hook. training_metadata로 origin schedule, runtime 추적.

### 함정

- fit_per_origin default가 last_origin_only (가장 cheap). 모든 origin 보려면 명시.
- forecast_scale_view default가 both_overlay. transformed와 level 모두 시각화 필요.
- coef_view는 *linear model 한정*. Tree/NN 모델은 다른 importance 사용 (L7).
- tuning_view는 L4.D search 활성 시만 의미. None이면 표시할 trace 없음.
- ensemble_view는 L4에 combine 노드 있을 때만. Single model recipe면 비활성.

### Sample

```yaml
4_5_generator_diagnostics:
  enabled: true
  fixed_axes: {}
```

```yaml
4_5_generator_diagnostics:
  enabled: true
  fixed_axes:
    fit_view: multi
    fit_per_origin: every_n_origins
    forecast_scale_view: both_overlay
    window_view: multi
    tuning_view: multi
    ensemble_view: multi
  leaf_config:
    fit_n_origins_step: 24
    coef_top_k: 15
```

---

## 4 diagnostic 비교

| 항목 | L1.5 | L2.5 | L3.5 | L4.5 |
|---|---|---|---|---|
| Hook | L1 | L1+L2 | L1+L2+L3 | L4+L3 |
| 목적 | data 품질 | preprocessing 효과 | feature engineering 효과 | model fit 품질 |
| Sub-layer | 5 + Z | 4 + Z | 5 + Z | 5 + Z |
| Cross-layer gate | regime (B) | 없음 | 5종 (B factor / B dfm / D lag / D marx / E selection) | 3종 (D tuning / E ensemble / C linear coef) |
| Multi-stage 비교 | 자체 | comparison_pair | comparison_stages | window |

### 사용 패턴

| 시나리오 | 권장 활성 |
|---|---|
| 신규 데이터셋 첫 분석 | L1.5만 |
| Cleaning policy 의심 | L1.5 + L2.5 |
| Feature engineering 검증 | L2.5 + L3.5 |
| Model 결과 의심 | L4.5 |
| Paper replication | 전부 (diagnostic_format = multi, attach_to_manifest = true) |
