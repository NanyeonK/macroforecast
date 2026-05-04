# macrocast 설계도 — Part 3

L5 + L6 + L7 + L8 (consumption layer)

---

## L5: Evaluation

### 답하는 질문

forecast의 *정확도를 어떻게 측정*할 것인가.

Metric 종류, benchmark 비교, aggregation, 기간 분할, decomposition, ranking. *기술 통계*. 추론 통계는 L6.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l4_forecasts_v1, l4_model_artifacts_v1, l1_data_definition_v1, l1_regime_metadata_v1 (gated), l3_metadata_v1 |
| 만드는 sink | l5_evaluation_v1 |
| 다음 사용처 | L6, L7, L8 |

### Sub-layer

순서대로 적용.

| Slot | 이름 | 역할 |
|---|---|---|
| L5.A | Metric specification | 어떤 metric을 계산할 것인가 |
| L5.B | Benchmark comparison | benchmark 대비 어떻게 비교할 것인가 |
| L5.C | Aggregation | 시간/horizon/target/state 어떻게 집계 |
| L5.D | Sample slicing & decomposition | OOS 어떻게 자르고 분해 |
| L5.E | Ranking & reporting | 어떻게 순위 매기고 리포트 |

L5.B는 L4에 is_benchmark 모델 있을 때만 활성.

### Axis 선택지 (요약)

**primary_metric (L5.A)**

| 선택지 | 언제 |
|---|---|
| mse (default) | 기본 |
| rmse | 표시 선호 |
| mae | outlier robust |
| relative_mse | benchmark 있을 때 권장 |
| r2_oos | Campbell-Thompson, benchmark 있을 때 |
| log_score / crps | density forecast |

**point_metrics, density_metrics, direction_metrics, relative_metrics**

multi-select list. 추가 컬럼으로 표에 등장.

**benchmark_window (L5.B)**

| 선택지 | 의미 |
|---|---|
| full_oos (default) | 전체 OOS 기간 |
| rolling / expanding | window 기반 |

**benchmark_scope (L5.B)**

| 선택지 | 의미 |
|---|---|
| all_targets_horizons (default) | 단일 benchmark 모든 (target, h) |
| per_target / per_horizon / per_target_horizon | 세분 |

**Aggregation (L5.C)**

| axis | 선택지 |
|---|---|
| agg_time | mean (default) / median / weighted_recent / per_subperiod |
| agg_horizon | per_horizon_separate (default) / mean / per_horizon_then_mean |
| agg_target | (multi-target 시) per_target_separate (default) / mean / weighted |
| agg_state | (FRED-SD 시) pool_states (default) / per_state_separate / weighted_average / top_k_worst |

**oos_period (L5.D)**

| 선택지 | 의미 |
|---|---|
| full_oos (default) | L1.E sample window의 OOS 부분 |
| fixed_dates | 사용자 지정 |
| rolling_window | rolling |
| multiple_subperiods | 여러 sub-period (pre/post-Volcker 등) |

**regime_use (L5.D)**

| 선택지 | 의미 |
|---|---|
| pooled (default) | regime 무시 |
| per_regime | regime별 |
| both | pooled + per-regime |

**decomposition_target, decomposition_order (L5.D)**

| decomposition_target | 의미 |
|---|---|
| none (default) | decomposition 안 함 |
| by_target / by_horizon / by_predictor_block / by_oos_period | metric을 axis별로 분해 |
| by_state | (FRED-SD 시) |
| by_regime | (regime != none 시) |

| decomposition_order | 의미 |
|---|---|
| marginal (default) | 단변량 |
| sequential | 순차 추가 |
| shapley | Shapley value (비싸다) |
| interaction_first_order | 1차 interaction 포함 |

**ranking (L5.E)**

| 선택지 | 의미 |
|---|---|
| by_primary_metric (default) | primary metric 기준 |
| by_relative_metric | benchmark-relative |
| by_average_rank | 모든 metric 평균 rank |
| borda_count | Borda voting |
| mcs_inclusion | MCS 통과 여부 (L6.D 의존) |

**report_style (L5.E)**

| 선택지 | 의미 |
|---|---|
| single_table (default) | (model × metric) 단일 표 |
| per_target_horizon_panel | (target, h)별 panel |
| heatmap / forest_plot / latex_table / markdown_table | export 형식 |

### Gate 흐름

| 조건 | 결과 |
|---|---|
| L4에 is_benchmark 모델 없음 | L5.B 비활성, relative_metrics 비활성 |
| forecast_object = point | density_metrics 비활성 |
| L1.B = single_target | agg_target 비활성 |
| L1.A.dataset에 fred_sd 미포함 | agg_state 비활성, decomposition by_state hard error |
| L1.G regime = none | regime_use 비활성, regime_metrics 비활성, decomposition by_regime hard error |
| ranking = mcs_inclusion + L6.D MCS 비활성 | hard error |
| ranking = by_relative_metric + benchmark 없음 | hard error |
| report_style = latex_table | leaf_config.latex_caption, latex_label 필요 |

### Layer 간 관계

| 흐름 | 무엇을 |
|---|---|
| L4 → L5 | forecasts, model_artifacts (is_benchmark 감지) |
| L1 → L5 | actual y, regime metadata |
| L3 → L5 | metadata (decomposition by_predictor_block에서) |
| L5 → L6 | metric values (DM, MCS 등에 사용) |
| L5 → L7 | ranking (top_k_by_metric에서) |
| L5 → L8 | export |

### 함정

- L5 axis는 sweepable=false. "compare metric choices"는 meta-decision.
- 모든 metric은 `mse`. `msfe`는 거부 (L4의 dmsfe combine method만 paper-standard 이름 유지).
- Benchmark는 L5 axis로 지정 안 함. L4의 is_benchmark flag.
- PT/HM은 L5에서 *metric value*만 (statistic value), L6.F에서 *test* (p-value).
- economic_metrics 없음. Portfolio metric은 out of scope.
- L5.D oos_period는 L1.E sample window의 *OOS 부분*만. L1.E 자체와 다름.

### Sample

```yaml
5_evaluation:
  fixed_axes: {}
```

```yaml
5_evaluation:
  fixed_axes:
    primary_metric: relative_mse
    point_metrics: [mse, mae]
    relative_metrics: [relative_mse, r2_oos]
    benchmark_scope: per_target_horizon
    agg_horizon: per_horizon_separate
    decomposition_target: by_predictor_block
    ranking: by_relative_metric
    report_style: per_target_horizon_panel
```

---

## L6: Statistical Tests

### 답하는 질문

forecast 차이가 *통계적으로 유의*한가.

DM, GW, CW, ENC-NEW, MCS, SPA, RC, StepM, PIT, Christoffersen, PT, HM, Ljung-Box 등. *추론 통계*.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l4_forecasts_v1, l4_model_artifacts_v1, l5_evaluation_v1, l1_data_definition_v1, l1_regime_metadata_v1 (gated) |
| 만드는 sink | l6_tests_v1 |
| 다음 사용처 | L7 (mcs_inclusion), L8 |

**Default off.** `enabled: true` 필요.

### Sub-layer

| Slot | 이름 | 검정 family |
|---|---|---|
| L6 globals | (advanced) | test_scope, dependence_correction, overlap_handling |
| L6.A | Equal predictive | DM (1995), GW (2006) |
| L6.B | Nested | CW (2007), ENC-NEW, ENC-T |
| L6.C | CPA & instability | Giacomini-Rossi (2010), Rossi-Sekhposyan |
| L6.D | Multiple model | MCS, SPA, Reality Check, StepM |
| L6.E | Density / interval | PIT-based, Kupiec, Christoffersen, DQ |
| L6.F | Direction | PT (1992), HM (1981) |
| L6.G | Residual | Ljung-Box, ARCH-LM, JB, BG, DW |

각 sub-layer도 자기 `enabled` flag. 두 단계 enabled (L6 + sub-layer).

### L6 globals

전역 정책. 모든 sub-layer에 균일 적용.

| axis | 선택지 |
|---|---|
| test_scope | per_target_horizon (default) / per_target / per_horizon / pooled |
| dependence_correction | newey_west (default) / andrews / parzen_kernel / none |
| overlap_handling | nw_with_h_minus_1_lag (default) / west_1996_adjustment / none |

### Sub-layer별 핵심 axis (요약)

**L6.A Equal predictive**

| axis | default | 의미 |
|---|---|---|
| equal_predictive_test | dm_diebold_mariano | DM / GW / multi |
| loss_function | squared | absolute / lin_lin / custom |
| model_pair_strategy | vs_benchmark_only | all_pairs / user_list |
| hln_correction | true | Harvey-Leybourne-Newbold 작은 sample 보정 |

**L6.B Nested**

| axis | default | 의미 |
|---|---|---|
| nested_test | clark_west | enc_new / enc_t / multi |
| nested_pair_strategy | vs_benchmark_auto | auto_detect / user_list |
| cw_adjustment | true | CW MSE adjustment 적용 |

**L6.C CPA**

| axis | default | 의미 |
|---|---|---|
| cpa_test | giacomini_rossi_2010 | rossi_sekhposyan / multi |
| cpa_window_type | derived | rolling_window (GR) / recursive (RS) |
| cpa_conditioning_info | none | lagged_loss_difference / regime / external_indicator |
| cpa_critical_value_method | simulated | bootstrap / asymptotic |

**L6.D Multiple model**

| axis | default | 의미 |
|---|---|---|
| multiple_model_test | mcs_hansen | spa_hansen / reality_check_white / step_m_romano_wolf / multi |
| mcs_alpha | 0.10 | confidence level |
| mmt_loss_function | squared | absolute |
| bootstrap_method | stationary_bootstrap | block / circular |
| bootstrap_n_replications | 1000 | int |
| bootstrap_block_length | auto | int (Politis-White auto) |
| mcs_t_statistic | t_max | t_range |
| spa_studentization | consistent | lower / upper |
| stepm_alpha | 0.10 | StepM FWER level |

`mcs_t_statistic`은 mcs_hansen일 때만, `spa_studentization`은 spa_hansen일 때만, `stepm_alpha`는 step_m_romano_wolf일 때만 활성 (test별 gating).

**L6.E Density / interval**

forecast_object ∈ {quantile, density}일 때만.

| axis | default | 의미 |
|---|---|---|
| density_test | pit_berkowitz | pit_kolmogorov_smirnov / pit_anderson_darling / pit_ljung_box / multi |
| interval_test | christoffersen_conditional_coverage | kupiec / christoffersen_independence / dynamic_quantile_test / multi |
| coverage_levels | [0.5, 0.9, 0.95] | nominal level list |
| pit_n_bins | 10 | int |
| pit_test_horizon_dependence | nw_correction | none |

**L6.F Direction**

| axis | default | 의미 |
|---|---|---|
| direction_test | pesaran_timmermann_1992 | henriksson_merton / multi |
| direction_threshold | zero | median / user_defined |
| direction_alpha | 0.05 | significance |

**L6.G Residual**

| axis | default | 의미 |
|---|---|---|
| residual_test | [ljung_box_q, arch_lm, jarque_bera_normality] | list |
| residual_lag_count | derived (10 monthly / 4 quarterly) | int |
| residual_test_scope | per_model_target_horizon | per_model |
| residual_alpha | 0.05 | significance |

### Gate 흐름

| 조건 | 결과 |
|---|---|
| L6.enabled = false | 모든 sub-layer 비활성 |
| forecast_object = point | L6.E 비활성 |
| L4에 is_benchmark 없음 + model_pair_strategy = vs_benchmark_only | hard error |
| nested_test = clark_west + cw_adjustment = false | hard error |
| cpa_conditioning_info = regime + L1.G regime = none | hard error |
| overlap_handling = none + h > 1 | hard error |
| bootstrap_n_replications < 100 | hard error |
| bootstrap_n_replications < 500 | soft warning |
| mcs_t_statistic 사용 + test != mcs_hansen | gated inactive |
| stepm_alpha 사용 + test != step_m_romano_wolf | gated inactive |

### Layer 간 관계

| 흐름 | 무엇을 |
|---|---|
| L4 → L6 | forecasts, is_benchmark |
| L5 → L6 | metric values |
| L1.regime → L6.C | conditioning |
| L6.D mcs_inclusion → L7 | importance target_models filter |
| L6 → L8 | export |

### 함정

- Default off + sub-layer별 enabled까지 두 단계. `L6.enabled: true`만으론 부족, sub-layer마다 `enabled: true` 필요.
- mmt_loss_function은 L6.A의 loss_function과 별도 axis. Inferential test와 selection test가 다른 loss 사용 가능.
- StepM은 operational. Future 아님.
- L6 axis는 sweepable=false.
- PT/HM은 여기서 *test* (p-value 포함). L5에서는 *metric value*만.

### Sample

```yaml
6_statistical_tests:
  enabled: true
  test_scope: per_target_horizon
  dependence_correction: newey_west
  sub_layers:
    L6_A_equal_predictive:
      enabled: true
      fixed_axes:
        equal_predictive_test: dm_diebold_mariano
        model_pair_strategy: vs_benchmark_only
    L6_D_multiple_model:
      enabled: true
      fixed_axes:
        multiple_model_test: mcs_hansen
        mcs_alpha: 0.10
        bootstrap_n_replications: 1000
```

---

## L7: Interpretation / Importance

### 답하는 질문

어느 feature가 forecast에 *어떻게 기여*하는가.

SHAP, permutation, PDP, ALE, gradient-based, lasso inclusion, BVAR PIP, FEVD, IRF, transformation attribution. 매크로 forecasting paper의 표준 해석 layer.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l4_model_artifacts_v1, l4_forecasts_v1, l3_features_v1, l3_metadata_v1, l5_evaluation_v1, l6_tests_v1 (gated), l1_data_definition_v1, l1_regime_metadata_v1 (gated) |
| 만드는 sink | l7_importance_v1, l7_transformation_attribution_v1 |
| 다음 사용처 | L8 |

**Default off.**

### Sub-layer

| Slot | 이름 | 역할 |
|---|---|---|
| L7.A | Importance DAG body | Step 합성으로 importance 계산 |
| L7.B | Output shape & export | figure type, table format |

L3 패턴과 동일 (DAG body + sinks).

### DAG mode

Graph mode. 다양한 importance 패턴.

```
Single method:        [model, X] → shap_tree → sink
Multi method:         [model, X] → shap, perm, pdp, ale → sinks
Group analysis:       shap → group_aggregate → sink
Lineage analysis:     shap → lineage_attribution(L3 metadata) → sink
Temporal:             [model, X, y] → rolling_recompute → sink
Transformation attr:  [all_l4_cells, l5_metrics] → transformation_attribution → sink
```

### Step library (~18 operational + 17 future)

> v0.1 honesty pass (PR-A..G) demoted 11 design-time-operational ops to
> `future` because their runtime did not match the named procedure:
> `fevd`, `historical_decomposition`, `generalized_irf`, `mrf_gtvp`,
> `lasso_inclusion_frequency`, `accumulated_local_effect`,
> `friedman_h_interaction`, `gradient_shap`, `integrated_gradients`,
> `saliency_map`, `deep_lift`. Each is tracked for v0.2 implementation
> in the GitHub issue tracker. `HONESTY_DEMOTED_L7_OPS` in
> `core/ops/l7_ops.py` enumerates them. The 6 always-future ops
> (`attention_weights`, `lstm_hidden_state`, `boruta_selection`,
> `recursive_feature_elimination`, `lasso_path_selection`,
> `stability_selection`) remain.

| 카테고리 | 개수 | 대표 op |
|---|---|---|
| Model-agnostic | 2 | permutation_importance, lofo |
| Model-native | 3 | linear_coef, tree_importance, mrf_gtvp |
| SHAP family | 5 | shap_tree, shap_kernel, shap_linear, shap_deep, shap_interaction |
| Marginal effect | 3 | partial_dependence, accumulated_local_effect, friedman_h_interaction |
| Gradient-based (NN) | 4 | integrated_gradients, saliency_map, deep_lift, gradient_shap |
| Variable inclusion | 3 | lasso_inclusion_frequency, bvar_pip, cumulative_r2_contribution |
| VAR-specific | 3 | fevd, historical_decomposition, generalized_irf |
| Linear forecast decomp | 1 | forecast_decomposition |
| Aggregation/temporal | 4 | group_aggregate, lineage_attribution, rolling_recompute, bootstrap_jackknife |
| Transformation attr | 1 | transformation_attribution |

Future 6개: attention_weights, lstm_hidden_state, boruta_selection, recursive_feature_elimination, lasso_path_selection, stability_selection.

### 호환성 규칙 (요약)

| Op family | 호환 model_family |
|---|---|
| shap_tree, shap_interaction | tree_set (RF, XGB, LGBM, GBM, DT, ET, CatBoost) |
| shap_linear | linear_set (OLS, ridge, lasso, EN, AR, VAR, BVAR, FA-AR) |
| shap_deep | NN_set (MLP, LSTM, GRU, Transformer) |
| shap_kernel | any (느림) |
| Gradient-based 4종 | NN_set only |
| VAR-specific 3종 | VAR, BVAR_* only |
| linear_coef, cumulative_r2 | linear_set |
| tree_importance | tree_set |
| mrf_gtvp | macroeconomic_random_forest only |
| lasso_inclusion_frequency | lasso, elastic_net |
| bvar_pip | bvar_minnesota, bvar_normal_inverse_wishart |
| forecast_decomposition | linear_set |

호환 안 맞으면 모두 hard error.

### Pre-defined block (8개)

`group_aggregate` op이 사용.

**FRED-derived (auto-mapped)**

| Block | 조건 |
|---|---|
| mccracken_ng_md_groups | dataset에 fred_md |
| mccracken_ng_qd_groups | dataset에 fred_qd |
| fred_sd_states | dataset에 fred_sd |

**Theme blocks (fixed)**

| Block | 정의 |
|---|---|
| nber_real_activity | INDPRO, PAYEMS, RPI, CMRMTSPL |
| taylor_rule_block | CPIAUCSL, GDPC1, FEDFUNDS |
| term_structure_block | TB3MS, GS1, GS5, GS10, T10Y3M |
| credit_spread_block | BAA, AAA, BAAFFM, AAAFFM |
| financial_conditions_block | NFCI |

User-defined block은 leaf_config.user_groups로.

### Figure type (18개)

각 step마다 default mapping. 사용자 override 가능.

| Step | Default figure |
|---|---|
| permutation, lofo, model_native_*, cumulative_r2 | bar_global |
| shap_tree, shap_kernel, shap_linear, shap_deep | beeswarm + force_plot |
| shap_interaction, friedman_h | heatmap |
| partial_dependence | pdp_line |
| accumulated_local_effect | ale_line |
| Gradient-based 4종 | attribution_heatmap |
| lasso_inclusion_frequency | inclusion_heatmap |
| bvar_pip | pip_bar |
| fevd, historical_decomposition, forecast_decomposition | historical_decomp_stacked_bar |
| generalized_irf | irf_with_confidence_band |
| group_aggregate | bar_grouped |
| lineage_attribution | bar_grouped_by_pipeline |
| rolling_recompute, mrf_gtvp | feature_heatmap_over_time |
| bootstrap_jackknife | bar_global (with error bars) |
| transformation_attribution | shapley_waterfall |

### L7.B Output axes

| axis | default |
|---|---|
| output_table_format | long |
| figure_type | auto (step default 사용) |
| top_k_features_to_show | 20 |
| precision_digits | 4 |
| figure_dpi | 300 |
| figure_format | pdf |
| latex_table_export | true |
| markdown_table_export | false |

### Gate 흐름

| 조건 | 결과 |
|---|---|
| Op family와 model_family 불호환 | hard error |
| FRED block + dataset 미포함 | hard error |
| theme block + 구성 series 부재 | hard error |
| mcs_inclusion source + L6.D MCS 비활성 | hard error |
| Future op 사용 | hard error |
| L7.B axes는 sweepable=false |  |

### Layer 간 관계

| 흐름 | 무엇을 |
|---|---|
| L4 → L7 | model_artifacts (importance), all_cells (transformation_attribution) |
| L3 → L7 | features, metadata (lineage_attribution) |
| L5 → L7 | ranking (top_k_by_metric) |
| L6.D → L7 | mcs_inclusion (target_models filter) |
| L7 → L8 | export |

### 함정

- 두 sink. `l7_transformation_attribution_v1`은 transformation_attribution step 사용 시만 채워짐 (안 쓰면 비어 있음).
- Op family hard rule이 강함. 모델 family 안 맞으면 schema 단계에서 hard error.
- Pre-defined block과 user-defined block 둘 다 가능. 같은 grouping axis로 표현.
- L7.B output axes는 *output decision*. Sweep 안 됨.
- Forecast combination ops는 L7에 없음. L4 전용.
- mrf_gtvp는 v0.1 honesty pass 후 **future** status (L4 MRF 와 동시 구현 예정 — v0.2 GitHub issue tracker).

### Sample

```yaml
7_interpretation:
  enabled: true
  nodes:
    - {id: src_xgb, type: source, selector: {layer_ref: l4, sink_name: l4_model_artifacts_v1, subset: {model_id: xgb_full}}}
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_l3_meta, type: source, selector: {layer_ref: l3, sink_name: l3_metadata_v1}}

    - {id: shap, type: step, op: shap_tree, inputs: [src_xgb, src_X]}
    - {id: shap_groups, type: step, op: group_aggregate, params: {grouping: mccracken_ng_md_groups, aggregation: sum}, inputs: [shap]}
    - {id: shap_lineage, type: step, op: lineage_attribution, params: {level: pipeline_name}, inputs: [shap, src_l3_meta]}

  sinks:
    l7_importance_v1: {global: shap, group: shap_groups, lineage: shap_lineage}

  fixed_axes:
    figure_type: auto
    top_k_features_to_show: 20
```

---

## L8: Output / Provenance

### 답하는 질문

모든 sink를 *어떤 디렉토리 구조와 형식*으로 export할 것인가.

L8은 macrocast의 *외부 인터페이스*. paper, replication, audit이 읽는 단위.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | L0~L7 + 활성 diagnostic 모두 (최대 13 sink) |
| 만드는 sink | l8_artifacts_v1 (manifest of exported files) |
| 다음 사용처 | 외부 (file system, paper) |

가장 많은 upstream input 가진 layer.

### Sub-layer

| Slot | 이름 | 역할 |
|---|---|---|
| L8.A | Export format | 파일 포맷 |
| L8.B | Saved objects | 어느 sink를 저장 |
| L8.C | Provenance | manifest 내용 |
| L8.D | Artifact granularity | 디렉토리 구조 |

### Axis 선택지 (요약)

**export_format (L8.A)**

| 선택지 | 의미 |
|---|---|
| json_csv (default) | JSON metadata + CSV tabular (paper standard) |
| json | metadata only |
| csv | tabular only |
| parquet | 큰 데이터 |
| json_parquet | metadata + parquet |
| latex_tables | paper-ready LaTeX |
| markdown_report / html_report | summary |
| all | 위 모두 |

**compression (L8.A)**

`none` (default) / `gzip` / `zip`.

**saved_objects (L8.B, multi-select list)**

Default는 active layer에서 derive. 다음을 자동 포함:

- forecasts (항상)
- forecast_intervals (forecast_object가 quantile/density일 때)
- metrics, ranking (L5 active)
- decomposition (L5.D 활성 시)
- regime_metrics (regime_use != pooled 시)
- state_metrics (Geography 활성 시)
- combination_weights (ensemble 시)
- diagnostics_l1_5 ~ diagnostics_l4_5 (각 진단 활성 시)
- diagnostics_all (shortcut)
- tests (L6 enabled)
- importance (L7 enabled)
- transformation_attribution (transformation_attribution step 사용 시)

Default 미포함 (크기 큼, 사용자 명시 시만):

- model_artifacts
- feature_metadata
- clean_panel
- raw_panel

**model_artifacts_format (L8.B)**

| 선택지 | 상태 |
|---|---|
| pickle (default) | operational |
| joblib | operational |
| onnx | future |
| pmml | future |

**provenance_fields (L8.C, multi-select)**

Default = ALL 14 fields.

| Field | 내용 |
|---|---|
| recipe_yaml_full | 전체 YAML 임베드 |
| recipe_hash | canonical hash |
| package_version | macrocast 버전 |
| python_version, r_version, julia_version | runtime 버전 |
| dependency_lockfile | uv.lock + renv.lock 내용 |
| git_commit_sha, git_branch_name | git 정보 |
| data_revision_tag | FRED vintage 또는 download timestamp |
| random_seed_used | L0의 seed |
| runtime_environment | OS, CPU, GPU |
| runtime_duration | per-layer 실행 시간 |
| cell_resolved_axes | 각 cell의 default vs explicit 구분 |

**manifest_format (L8.C)**

| 선택지 | 의미 |
|---|---|
| json (default) | 단일 manifest.json |
| yaml | manifest.yaml |
| json_lines | 큰 sweep용 line-delimited |

**artifact_granularity (L8.D)**

| 선택지 | 의미 |
|---|---|
| per_cell (default) | 각 sweep cell별 디렉토리 |
| per_target / per_horizon / per_target_horizon | 차원별 |
| flat | 단일 디렉토리 |

**naming_convention (L8.D)**

| 선택지 | 의미 |
|---|---|
| descriptive (default) | `xgb_direct_h6/` 형태 |
| cell_id | `cell_001/`, `cell_002/` |
| recipe_hash | hash 기반 |
| custom | leaf_config 콜러블 |

`descriptive_naming_template` default: `"{model_family}_{forecast_strategy}_h{horizon}"`.

### 디렉토리 구조 (default)

```
output_directory/
├── manifest.json
├── recipe.yaml
├── lockfiles/
│   ├── uv.lock
│   └── renv.lock
├── summary/
│   ├── metrics_all_cells.csv
│   ├── ranking.csv
│   ├── tests_summary.csv
│   └── importance_summary.csv
├── ridge_direct_h1/
│   ├── forecasts.csv
│   ├── metrics.json
│   ├── tests.json
│   ├── importance.json
│   ├── cell_manifest.json
│   └── figures/
├── xgb_direct_h6/
│   └── ...
├── diagnostics/
│   ├── l1_5_data_summary/
│   ├── l2_5_pre_post/
│   ├── l3_5_features/
│   └── l4_5_generator/
└── tests/
    ├── dm_test.csv
    └── mcs_inclusion.csv
```

### Gate 흐름

| 조건 | 결과 |
|---|---|
| L8 axes는 sweepable=false |  |
| latex_tables / markdown_report / html_report | L5 active 필요 |
| state_metrics in saved_objects + Geography 미활성 | hard error |
| regime_metrics in saved_objects + regime = none | hard error |
| combination_weights in saved_objects + ensemble 없음 | hard error |
| onnx, pmml format | hard error (future) |
| custom naming + custom_naming_function 없음 | hard error |
| descriptive_naming_template 잘못된 placeholder | hard error |

### Layer 간 관계

| 흐름 | 무엇을 |
|---|---|
| L0~L7 → L8 | 모든 sink |
| 활성 diagnostic → L8 | DiagnosticArtifact |
| L8 → 외부 | output directory |

### 함정

- saved_objects는 *derived default*. Recipe에 무엇이 활성인지에 따라 자동 결정.
- model_artifacts는 default에 없음. 용량 커서 explicit opt-in.
- recipe_yaml_full은 *임베드*. 별도 파일 참조 아님. Self-contained replication.
- diagnostics_all은 shortcut. l1_5 ~ l4_5 4개 모두 포함.
- L8 자체가 sweep 안 되지만 sweep cell 결과는 L8이 organize.
- inverse_msfe 등은 L4의 weights_method. L8에서 export될 때도 그 이름 그대로.

### Sample

```yaml
8_output:
  fixed_axes: {}
```

```yaml
8_output:
  fixed_axes:
    export_format: all
    saved_objects: [forecasts, metrics, ranking, decomposition, model_artifacts, feature_metadata, diagnostics_all, tests, importance, transformation_attribution]
    artifact_granularity: per_cell
    naming_convention: descriptive
  leaf_config:
    output_directory: ./paper_replication/coulombe_2021/
    descriptive_naming_template: "{model_family}_{forecast_strategy}_h{horizon}_{combine_method}"
```
