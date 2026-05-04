# macrocast 설계도 — Part 2

L2 + L3 + L4 (construction layer)

---

## L2: Preprocessing

### 답하는 질문

raw panel을 어떻게 *cleaned panel*로 만들 것인가.

McCracken-Ng (2016) cleaned FRED-MD format 호환. T-coded stationary, outlier-handled, EM-imputed, edge-cleaned, **unscaled**.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l1_data_definition_v1 |
| 만드는 sink | l2_clean_panel_v1 |
| 다음 사용처 | L3, L1.5, L2.5, L3.5 |

L1의 regime_metadata는 안 받음 (regime은 L3+의 관심사).

### Sub-layer

순서대로 적용되는 pipeline.

| Slot | 이름 | 답하는 질문 |
|---|---|---|
| L2.A | FRED-SD frequency alignment | (FRED-SD only) 다른 빈도 series를 어떻게 맞출 것인가 |
| L2.B | Transform | t-code 적용 (level/log/diff/log_diff/pct_change) |
| L2.C | Outlier handling | outlier를 어떻게 처리할 것인가 |
| L2.D | Imputation | missing을 어떻게 채울 것인가 |
| L2.E | Frame edge | leading NaN gap을 어떻게 처리할 것인가 |

L2.F (scaling) 없음. Scaling은 L3 model-adjacent stage.

### Cleaning scope 메커니즘

각 sub-layer가 자기 `*_scope` axis를 가짐. 단일 layer-global cleaning_scope는 없음.

| sub-layer | scope axis | default |
|---|---|---|
| L2.B | transform_scope | target_and_predictors |
| L2.C | outlier_scope | predictors_only (McCracken-Ng) |
| L2.D | imputation_scope | predictors_only |
| L2.E | frame_edge_scope | predictors_only |

target y는 보통 cleaning 대상에서 제외 (missingness/outlier가 평가 정보를 담음).

### Axis 선택지 (요약)

**transform_policy (L2.B)**

| 선택지 | 의미 |
|---|---|
| apply_official_tcode (default) | FRED-MD/QD 공식 t-code |
| no_transform | 그대로 (raw level) |
| custom_tcode | 사용자 지정 map |

t-code는 L3 step library를 내부 호출 (log/diff/log_diff/pct_change). 별도 구현 아님.

**outlier_policy (L2.C)**

| 선택지 | 의미 |
|---|---|
| mccracken_ng_iqr (default) | \|x - median\| > 10·IQR flagged |
| winsorize | quantile cap |
| zscore_threshold | \|z\| > k flagged |
| none | 처리 안 함 |

**outlier_action (L2.C)**

| 선택지 | 의미 |
|---|---|
| flag_as_nan (default) | NaN으로 바꾸고 imputation에 위임 |
| replace_with_median | rolling median으로 대체 |
| replace_with_cap_value | winsorize cap 값으로 대체 |
| keep_with_indicator | future |

**imputation_policy (L2.D)**

| 선택지 | 의미 |
|---|---|
| em_factor (default) | McCracken-Ng PCA-EM |
| em_multivariate | covariance-based EM |
| mean | in-sample mean |
| forward_fill | LOCF |
| linear_interpolation | linear |
| none_propagate | NaN 그대로 |

**imputation_temporal_rule (L2.D)**

| 선택지 | 의미 |
|---|---|
| expanding_window_per_origin (default) | 각 origin마다 history 전부로 EM 재추정 |
| rolling_window_per_origin | rolling window |
| block_recompute | N origin마다 |

`full_sample_once`는 거부 (leakage).

**frame_edge_policy (L2.E)**

| 선택지 | 의미 |
|---|---|
| truncate_to_balanced (default) | 가장 늦게 시작하는 series 시작점부터 |
| drop_unbalanced_series | 늦게 시작하는 series 제거 |
| keep_unbalanced | NaN 유지, L3로 위임 |
| zero_fill_leading | 0으로 채움 (비추천) |

### Gate 흐름

| 조건 | 결과 |
|---|---|
| dataset에 fred_sd 미포함 | L2.A 비활성 |
| L1.frequency = monthly + sd_series_frequency_filter ∈ {quarterly_only, both} | quarterly_to_monthly_rule 활성 |
| L1.frequency = quarterly + sd_series_frequency_filter ∈ {monthly_only, both} | monthly_to_quarterly_rule 활성 |
| transform_policy = no_transform | transform_scope = not_applicable |
| outlier_policy = none | outlier_scope = not_applicable, outlier_action 비활성 |
| imputation_policy = none_propagate | imputation_scope = not_applicable |
| frame_edge_policy = keep_unbalanced | frame_edge_scope = not_applicable |

### Layer 간 관계

| 흐름 | 무엇을 |
|---|---|
| L2 → L3 | clean panel을 feature engineering input으로 |
| L2 → L2.5 | 진단 (L1과 비교) |
| L2 → L3.5 | 3-stage 비교 (L1, L2, L3) |
| L1.regime → L2 | 안 사용 |

### 함정

- L2 sink는 *unscaled*. Scaling은 L3에 있음. McCracken-Ng publish format 일치.
- t-code는 별도 구현 아님. L3 step library (log, diff, log_diff)를 내부 호출.
- target y는 default로 outlier/imputation/frame_edge 처리 안 됨. McCracken-Ng 표준.
- `full_sample_once`는 모든 temporal_rule에서 schema-rejected. Leakage 방지.
- Pipeline 순서 강제 (A → B → C → D → E). 다른 순서는 hard error.

### Sample

```yaml
2_preprocessing:
  fixed_axes: {}
```

```yaml
2_preprocessing:
  fixed_axes:
    transform_policy: apply_official_tcode
    outlier_policy: mccracken_ng_iqr
    outlier_action: flag_as_nan
    imputation_policy: em_factor
    frame_edge_policy: truncate_to_balanced
  leaf_config:
    outlier_threshold_iqr: 10.0
    em_n_factors: 8
```

---

## L3: Feature Engineering

### 답하는 질문

clean panel을 *어떤 X_final, y_final*로 만들 것인가.

매크로 forecasting paper가 가장 분기되는 layer. Coulombe et al. (2021)이 "data transformation matters"를 보여준 핵심 영역.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l2_clean_panel_v1, l1_data_definition_v1 (raw access), l1_regime_metadata_v1 (gated) |
| 만드는 sink | l3_features_v1 (X, y), l3_metadata_v1 (lineage) |
| 다음 사용처 | L4, L3.5, L7 lineage_attribution |

L3은 **2개 sink** 만드는 첫 layer. metadata는 L7에서 lineage 추적.

### Sub-layer

기능적 그룹. Linear pipeline 아님 (DAG body).

| Slot | 이름 | 역할 |
|---|---|---|
| L3.A | Target construction | y_{t+h} 만들기 (point vs cumulative) |
| L3.B | Feature pipelines | DAG body (parallel pipelines) |
| L3.C | Pipeline combine | parallel output을 X로 합침 |
| L3.D | Feature selection | (optional) X에서 부분집합 선택 |

### DAG mode

**Graph mode UI**. Sugar form 없음. L3은 multi-parent DAG가 자연스럽기 때문.

```
[L1 raw] [L2 cleaned] [L1 regime]
        ↓ ↓ ↓
   [L3.A target]  [L3.B parallel pipelines]
                          ↓
                   [L3.C combine]
                          ↓
                   [L3.D selection]
                          ↓
                       sinks
```

### Source 종류

L3는 source selector를 풍부하게 사용.

| Source | 의미 |
|---|---|
| L2 cleaned panel (predictors) | 표준 X input |
| L2 cleaned panel (target) | 표준 y input |
| L1 raw panel (`raw: true`) | Coulombe "L" pipeline (level features) |
| L1 regime metadata | regime feature engineering |
| L3 pipeline output (cascade) | β extension. 다른 pipeline 출력을 input으로 |

### Step library (37 operational + 6 future)

카테고리별.

| 카테고리 | 개수 | 대표 op |
|---|---|---|
| stationary_transform | 5 | log, diff, log_diff, pct_change |
| lag | 2 | lag, seasonal_lag |
| aggregation | 3 | ma_window, ma_increasing_order (MARX), cumsum |
| scale | 1 | scale (zscore/robust/minmax) |
| reduction | 7 | pca, sparse_pca, scaled_pca, dfm, varimax, partial_least_squares, random_projection |
| spectral | 2 | wavelet, fourier |
| detrending | 2 | hp_filter, hamilton_filter |
| feature_expansion | 4 | polynomial, interaction, kernel, nystroem |
| auxiliary | 4 | regime_indicator, season_dummy, time_trend, holiday |
| target | 1 | target_construction (L3.A only) |
| feature_selection | 1 + 5 future | feature_selection (variance/correlation/lasso/...) |
| combine | 5 | concat, interact, hierarchical_pca, weighted_concat, simple_average |

### Cascade β extension

한 pipeline의 output을 다른 pipeline의 input으로 사용 가능.

```
[clean] → [pipeline_A: MARX] → cascade_source → [pipeline_B: PCA on MARX] → [combine] → X
```

표현: step에 `pipeline_id: marx` 라벨. 다른 step에서 `selector: {layer_ref: l3, sink_name: pipeline_output, subset: {pipeline_id: marx}}`.

| 규칙 | 의미 |
|---|---|
| max_cascade_depth | default 3 (leaf_config로 변경 가능) |
| cycle 검출 | cascade graph 포함해서 cycle 있으면 hard error |
| 같은 pipeline_id 모호성 | 여러 step에 같은 라벨 + 모두 sink 아니면 hard error |

### 3-tier validation

| Tier | 의미 |
|---|---|
| Hard | DAG cycle, type mismatch, target_construction이 L3.A 외에 등장, future op 사용 등 → 실행 불가 |
| Soft | ordering 비표준 (lag before log), n_lag/T 비율 높음, dfm n_lags 큼 등 → 경고만 |
| Pass-through | order of pca and lag, wavelet vs fourier 등 → 침묵 |

### Gate 흐름

| 조건 | 결과 |
|---|---|
| target_construction op이 L3.B에 등장 | hard error (L3.A 전용) |
| target_construction.horizon이 L1.F.horizon_set에 없음 | hard error |
| factor op (pca/dfm/...)에 temporal_rule = full_sample_once | hard error |
| scaled_pca, partial_least_squares에 target_signal input 없음 | hard error |
| n_components > min(T, N) | hard error (rank constraint) |
| regime_indicator step + L1.G regime = none | hard error |
| feature_selection 사용 + Series input 없음 | hard error |

### Layer 간 관계

| 흐름 | 무엇을 |
|---|---|
| L2 → L3 | clean panel input |
| L1 → L3 | raw panel access (Coulombe L), regime metadata |
| L3 → L4 | X_final, y_final |
| L3 metadata → L7 | lineage_attribution |
| L3 → L3.5 | 3-stage 진단 |

### 함정

- target_construction은 L3.A에만 등장 가능. L3.B에 두면 hard error.
- L3 sink는 (X_final, y_final) tuple. y_final은 Series, X_final은 Panel. 혼동하면 type mismatch.
- Forecast combination은 L3에 없음. L4의 일이다 (forecast→forecast는 모델 단계).
- macroeconomic_random_forest는 L4 model_family. L3 step 아님.
- L3 step은 L2 step과 일부 공유 (log, diff 등). 같은 op 등록, 다른 layer_scope.
- L3은 sugar form 안 됨. Multi-parent DAG라 항상 explicit nodes 필요.

### Sample (간단)

```yaml
3_feature_engineering:
  nodes:
    - {id: src_x, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: pca, type: step, op: pca, params: {n_components: 8, temporal_rule: expanding_window_per_origin}, inputs: [src_x]}
    - {id: lag_pca, type: step, op: lag, params: {n_lag: 4}, inputs: [pca]}
    - {id: y_h, type: step, op: target_construction, params: {mode: cumulative_average, method: direct, horizon: 6}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_pca, y_final: y_h}
    l3_metadata_v1: auto
```

---

## L4: Forecasting Model

### 답하는 질문

X_final, y_final로 *어떻게 forecast*를 만들 것인가.

Model family, ensemble, forecast strategy, training window, tuning. Forecast가 *primary scientific output*.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | l3_features_v1, l3_metadata_v1, l1_regime_metadata_v1 (gated) |
| 만드는 sink | l4_forecasts_v1, l4_model_artifacts_v1, l4_training_metadata_v1 |
| 다음 사용처 | L5, L6, L7, L4.5 |

L4는 **3개 sink**. training_metadata는 L4.5 window stability 진단에 사용.

### Sub-layer

| Slot | 이름 | 역할 |
|---|---|---|
| L4.A | Model selection | model_family + ensemble combine 노드 |
| L4.B | Forecast strategy | direct / iterated / path_average |
| L4.C | Training window | expanding / rolling / fixed |
| L4.D | Tuning | hyperparameter search |

L4.E (ensemble) 별도 sub-layer 안 만듦. Combine 노드는 L4.A 안.

### DAG mode

Graph mode. Single model부터 ensemble까지.

```
Single:
  [src_X, src_y] → fit_model_ridge → predict → sink

Ensemble:
  [src_X, src_y] → fit_model_ridge → predict_ridge ─┐
                 → fit_model_xgb   → predict_xgb  ─┤
                 → fit_model_ar1   → predict_ar1  ─┤  (is_benchmark)
                                                    ↓
                                       weighted_average_forecast → sink
```

### Benchmark 메커니즘

특정 fit_model node에 `is_benchmark: true` flag. L5/L6가 자동 감지.

| 조건 | L5/L6 동작 |
|---|---|
| is_benchmark 모델 1개 | 해당 모델이 benchmark, relative metric 활성 |
| is_benchmark 모델 0개 | benchmark 없음, relative metric 비활성 |
| is_benchmark 모델 2개 이상 | hard error (모호함) |

### Model family library (30 operational + 4 future)

| 카테고리 | 개수 | 대표 |
|---|---|---|
| Linear | 8 | ar_p, ols, ridge, lasso, elastic_net, lasso_path, glmboost, var |
| Factor-augmented | 3 | factor_augmented_ar, factor_augmented_var, principal_component_regression |
| Tree | 3 | decision_tree, random_forest, extra_trees |
| Boosting | 4 | gradient_boosting, xgboost, lightgbm, catboost |
| SVM | 3 | svr_linear, svr_rbf, svr_poly |
| Neural net | 4 | mlp, lstm, gru, transformer |
| Neighbors | 1 | knn |
| Macro-specific | 1 (planned, v0.1) | macroeconomic_random_forest (Coulombe 2024) |
| Bayesian | 2 | bvar_minnesota, bvar_normal_inverse_wishart |
| Mixed-frequency | 1 planned + 4 future | dfm_mixed_mariano_murasawa (planned, v0.1) / midas_almon, midas_beta, midas_step, dfm_unrestricted_midas (future) |

`macroeconomic_random_forest`와 `dfm_mixed_mariano_murasawa`는 v0.1에서 **planned** status로 분류 (`PLANNED_MODEL_FAMILIES`). Schema는 operational families와 동일하게 통과, runtime wrapper도 동작하나, 본 wrapper는 *acknowledged approximation* — Coulombe 2024 MRF의 GTVP local-linear refinement + asymmetric loss, 그리고 Mariano-Murasawa Kalman state-space EM은 별도 PR (v1.x)에서 진짜 구현 예정. `get_family_status(family) == "planned"` 로 런타임 detect.

### Forecast combine ops (5)

L4.A 안의 combine 노드. ForecastArtifact → ForecastArtifact.

| Op | 용도 |
|---|---|
| weighted_average_forecast | 가장 일반. weights_method axis로 다양 |
| median_forecast | 중앙값 |
| trimmed_mean_forecast | trim된 평균 |
| bma_forecast | Bayesian model averaging |
| bivariate_ardl_combination | Rapach-Strauss 정확히 2개 |

**weighted_average_forecast의 weights_method**

| 선택지 | 의미 |
|---|---|
| equal | 1/N |
| dmsfe | Discounted MSFE (Bates-Granger with discount) |
| inverse_msfe | dmsfe with theta=1.0 (alias, paper-standard 이름) |
| mallows_cp | Hansen (2007) |
| sic_weights | Stock-Watson (2008) |
| granger_ramanathan | OLS-optimal |
| cv_optimized | CV로 최적화 |

### Forecast strategy (L4.B)

| 선택지 | 의미 |
|---|---|
| direct (default) | h마다 별도 모델 |
| iterated | h=1 모델 재귀 |
| path_average | cumulative_average target (Coulombe path-averaged) |

`oracle_strategy`는 거부 (leakage).

### Training window (L4.C)

| training_start_rule | 의미 |
|---|---|
| expanding (default) | history 전부 |
| rolling | rolling window |
| fixed | 한 번 학습, 재학습 없음 |

| refit_policy | 의미 |
|---|---|
| every_origin (default) | 매 origin 재학습 |
| every_n_origins | N origin마다 |
| single_fit | 한 번만 |

### Tuning (L4.D)

**search_algorithm**

| 선택지 | 의미 |
|---|---|
| none (default) | 튜닝 안 함 |
| grid_search | 격자 |
| random_search | 무작위 |
| bayesian_optimization | GP-BO |
| genetic_algorithm | GA |
| cv_path | 정규화 path (lasso/ridge/elastic_net 전용) |

**tuning_objective, validation_method**

walk-forward 우선. kfold는 시계열 데이터에서 hard error (n_splits > 2일 때).

### Gate 흐름

| 조건 | 결과 |
|---|---|
| forecast_strategy = path_average + target.mode != cumulative_average | hard error |
| search_algorithm = cv_path + family이 lasso/ridge/elastic_net 외 | hard error |
| validation_method = kfold + datetime index + n_splits > 2 | hard error |
| training_start_rule = fixed + validation_method != kfold/validation_method | hard error |
| weighted_average_forecast.temporal_rule = full_sample_once | hard error |
| midas_* (future) 사용 | hard error |
| regime_wrapper + L1.G regime = none | hard error |

### Layer 간 관계

| 흐름 | 무엇을 |
|---|---|
| L3 → L4 | X, y, metadata |
| L1.regime → L4 | regime_wrapper |
| L4 → L5 | forecasts, model_artifacts |
| L4 → L6 | forecasts, is_benchmark |
| L4 → L7 | model_artifacts (importance), all_cells (transformation_attribution) |
| L4 → L4.5 | training_metadata |

### 함정

- Benchmark는 axis가 아니다. fit_model node의 `is_benchmark: true` flag.
- Forecast combination은 L4 안에 있음. L3 combine과 다른 type (forecast vs panel).
- `inverse_msfe`는 `dmsfe`의 alias (theta=1.0). 이름은 paper-standard 유지.
- MRF, dfm_mixed_mariano_murasawa는 v0.1 기준 **planned** status. Schema 통과 + runtime wrapper 동작 (RandomForest + time_trend / PCA + AR(1) approximation). 진짜 Coulombe / Mariano-Murasawa 구현은 v1.x로 deferred. `get_family_status` 로 detect.
- search_algorithm = none이면 tuning_objective, validation_method 비활성.

### Sample

```yaml
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}

    - id: fit_ar1
      type: step
      op: fit_model
      params: {family: ar_p, n_lag: 1, forecast_strategy: direct}
      is_benchmark: true
      inputs: [src_y]
    - {id: predict_ar1, type: step, op: predict, inputs: [fit_ar1]}

    - id: fit_xgb
      type: step
      op: fit_model
      params: {family: xgboost, search_algorithm: bayesian_optimization, tuning_objective: cv_mse}
      inputs: [src_X, src_y]
    - {id: predict_xgb, type: step, op: predict, inputs: [fit_xgb, src_X]}

    - id: ensemble
      type: combine
      op: weighted_average_forecast
      params: {weights_method: dmsfe, dmsfe_theta: 0.95}
      inputs: [predict_ar1, predict_xgb]

  sinks:
    l4_forecasts_v1: ensemble
    l4_model_artifacts_v1: [fit_ar1, fit_xgb]
    l4_training_metadata_v1: auto
```
