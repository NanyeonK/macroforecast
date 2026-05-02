# macrocast 설계도 — Part 1

Foundation + L0 + L1

이해를 위한 문서. Implementation spec과 별개. 각 layer의 의도, 선택지의 목적, gate의 의미, layer 간 관계를 한 눈에 본다.

---

## 시스템 개요

### 12 layer 분류

| 카테고리 | Layer | 역할 |
|---|---|---|
| Setup | L0 | study 단위 runtime 정책 |
| Construction | L1 | 데이터 contract |
| Construction | L2 | preprocessing |
| Construction | L3 | feature engineering |
| Construction | L4 | forecasting model |
| Diagnostic | L1.5 | raw data 진단 |
| Diagnostic | L2.5 | 전후 비교 진단 |
| Diagnostic | L3.5 | feature 진단 |
| Diagnostic | L4.5 | model fit 진단 |
| Consumption | L5 | metric 평가 |
| Consumption | L6 | 통계 검정 |
| Consumption | L7 | importance 해석 |
| Consumption | L8 | export, manifest |

### 데이터 흐름

```
L1 → L2 → L3 → L4 → L5 → L6 → L7 → L8
                          (L6/L7는 평행)

진단은 non-blocking hook:
L1.5 ← L1
L2.5 ← L1, L2
L3.5 ← L1, L2, L3
L4.5 ← L4, L3
```

### Cross-layer 참조 5개

| 출처 | 사용처 | 무엇을 |
|---|---|---|
| L1.G regime | L3, L4, L5, L6.C | regime indicator, regime_wrapper, per_regime metric |
| L4 is_benchmark flag | L5, L6 | relative_metric, vs_benchmark_only test |
| L6.D mcs_inclusion | L7 | MCS 통과 model만 importance |
| L3 metadata (lineage) | L7 | lineage_attribution |
| L4 multi-cell sweep | L7 | transformation_attribution |

### Sweep 위치

L0~L4 axis만 sweep 가능. L5~L8은 cell의 결과를 평가/탐색/내보내기만 함.

### Default-off layer

L6, L7, L1.5, L2.5, L3.5, L4.5는 명시적 `enabled: true` 필요. L0~L5, L8은 항상 active.

---

## Foundation

Layer가 아니다. 모든 layer가 의존하는 공통 구조.

### 5 노드 종류

| Type | 의미 |
|---|---|
| source | 외부 sink에서 데이터 끌어옴 |
| axis | enum 옵션 중 하나 선택 |
| step | data를 변환 |
| combine | 여러 input 결합 |
| sink | 다음 layer로 contract artifact 전달 |

### Adaptive UI

| DAG 모양 | UI |
|---|---|
| Tree-shaped (단일 부모) | axis list (L0/L1/L2/L5/L6/L8/diagnostic) |
| Multi-parent | graph editor (L3/L4/L7) |

같은 schema, 다른 표현.

### Sweep 종류

| 종류 | 표현 |
|---|---|
| Param-level | `n_lag: {sweep: [4, 8, 12]}` |
| Node-level | `sweep_groups: [pipeline_marx, pipeline_factors]` |
| External axis | `model_family: {sweep: [ridge, xgboost]}` |

여러 sweep은 grid (default) 또는 zip으로 결합.

### Cache

`(op, params, input_hashes)` 단위 hash. 같은 sub-graph는 재사용.

### Manifest

L8이 산출. 다음을 포함.

- recipe YAML 임베드
- recipe canonical hash
- macrocast/Python/R version
- uv.lock + renv.lock 내용
- git commit, OS/CPU/GPU info
- 각 cell의 resolved axes (default vs explicit 구분)

---

## L0: Study Setup

### 답하는 질문

이 study의 runtime 정책 (forecast 외의 기술적 결정).

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | 없음 |
| 만드는 sink | l0_meta_v1 |
| 다음 사용처 | manifest로 모든 layer가 참조 |

### Sub-layer

L0.A 1개. 모든 axis가 layer 전역.

### Axis 선택지

**failure_policy**

| 선택지 | 언제 |
|---|---|
| fail_fast (default) | 빠른 피드백, 디버깅 |
| continue_on_failure | 큰 sweep, 일부 실패 허용 |

**reproducibility_mode**

| 선택지 | 언제 |
|---|---|
| seeded_reproducible (default) | paper replication |
| exploratory | 시드 고정 전 탐색 |

`strict`는 거부. GPU 결정성은 leaf_config.gpu_deterministic으로 별도.

**compute_mode**

| 선택지 | leaf_config |
|---|---|
| serial (default) | 없음 |
| parallel | parallel_unit (models/horizons/targets/oos_dates), n_workers |

`parallel_models` 같은 sub-type은 거부. 두 결정 (병렬화 + 단위)을 분리.

### Gate 흐름

| 조건 | 결과 |
|---|---|
| reproducibility_mode = seeded_reproducible | leaf_config.random_seed 필요 |
| reproducibility_mode = exploratory | leaf_config.random_seed 거부 |
| compute_mode = parallel | leaf_config.parallel_unit, n_workers 필요 |

### Layer 간 관계

L0이 다른 layer에 직접 영향 없음. Manifest를 통한 간접 참조만.

### 함정

- L0 axis는 sweepable=false. "fail_fast vs continue_on_failure 비교"는 의미 없음.
- study_scope는 axis 아님. Recipe shape에서 자동 derive하여 manifest에만 기록.
- L0은 navigator 메인에 안 보임. Advanced 토글 안에.

### Sample

```yaml
0_meta:
  fixed_axes: {}
```

---

## L1: Data

### 답하는 질문

어떤 데이터를 분석 단위로 삼는가.

### Pipeline 위치

| 항목 | 값 |
|---|---|
| 받는 sink | 없음 |
| 만드는 sink | l1_data_definition_v1, l1_regime_metadata_v1 |
| 다음 사용처 | L2, L3, L1.5 |

### Sub-layer

| Slot | 이름 | 답하는 질문 |
|---|---|---|
| L1.A | Source selection | 어떤 데이터셋을 가져올 것인가 |
| L1.B | Target | 예측 대상은 무엇인가 |
| L1.C | Predictor universe | 예측에 사용할 변수 집합 |
| L1.D | Geography | (FRED-SD only) 지역적 범위 |
| L1.E | Sample window | 분석 기간 |
| L1.F | Horizon set | 어떤 h를 예측할 것인가 |
| L1.G | Regime definition | regime 정의 (default none) |

분리 이유: 각 결정이 서로 독립. 같은 dataset이라도 target과 predictor universe는 다를 수 있다.

### Axis 선택지 (요약)

**custom_source_policy (L1.A)**

| 선택지 | 의미 |
|---|---|
| official_only (default) | FRED 데이터만 |
| custom_panel_only | 사용자 데이터만 |
| official_plus_custom | FRED + 사용자 보조 |

**dataset (L1.A)**

`fred_md` / `fred_qd` / `fred_sd` / `fred_md+fred_sd` / `fred_qd+fred_sd`. frequency는 dataset에서 derive.

**vintage_policy (L1.A)**

| 선택지 | 상태 |
|---|---|
| current_vintage (default) | operational |
| real_time_alfred | future |

**target_structure (L1.B)**

`single_target` (default) / `multi_series_target`.

**variable_universe (L1.C)**

`all_variables` (default) / `core_variables` / `category_variables` / `target_specific_variables` / `explicit_variable_list`.

**target_geography_scope, predictor_geography_scope (L1.D)**

FRED-SD에서만 활성. 50 states + DC에서 부분집합 결정.

**sample_start_rule, sample_end_rule (L1.E)**

| 선택지 | 의미 |
|---|---|
| max_balanced (default start) | 모든 변수 관측되는 가장 이른 날짜 |
| latest_available (default end) | 가장 최근 |
| fixed_date | 사용자 지정 |

**horizon_set (L1.F)**

| 선택지 | 의미 |
|---|---|
| standard_md (derived monthly) | [1, 3, 6, 12] |
| standard_qd (derived quarterly) | [1, 2, 4, 8] |
| single / custom_list / range_up_to_h | 사용자 지정 |

**regime_definition (L1.G)**

| 선택지 | 의미 |
|---|---|
| none (default) | 사용 안 함 |
| external_nber | NBER recession dates auto-load |
| external_user_provided | 사용자 path 또는 list |
| estimated_markov_switching | Hamilton (1989) MS estimate |
| estimated_threshold | Tong (1990) threshold |
| estimated_structural_break | Bai-Perron break detection |

estimated_* 옵션은 추가 axis로 언제 다시 추정할지 결정. `full_sample_once`는 거부 (미래 정보 누설).

### Gate 흐름

| 조건 | 결과 |
|---|---|
| custom_source_policy = custom_panel_only | dataset, vintage_policy, variable_universe 모두 비활성 |
| dataset에 fred_sd 포함 | L1.D 활성 |
| dataset 단독 fred_sd | variable_universe 비활성, frequency 사용자 명시 |
| regime_definition = none | L1.G 다른 axis 비활성, sink는 빈 상태로 산출 |
| regime_definition = estimated_* | regime_estimation_temporal_rule 활성 |
| regime_definition = external_user_provided | leaf_config.regime_indicator_path 또는 regime_dates_list 필요 |

### Layer 간 관계

| 흐름 | 무엇을 |
|---|---|
| L1 → L2 | raw panel을 cleaning input으로 |
| L1 → L3 | raw panel access (Coulombe "L" pipeline) |
| L1 → L1.5 | 진단 input |
| L1.G regime → L3 | regime_indicator step |
| L1.G regime → L4 | model regime_wrapper |
| L1.G regime → L5 | per_regime metric |
| L1.G regime → L6.C | cpa_conditioning_info |

### 함정

- L1.B (target)와 L1.D (geography target_scope)는 다른 차원. L1.B는 어떤 series, L1.D는 어떤 지역.
- L1.E (sample window)와 L5.D (oos_period)는 다르다. L1.E는 전체 기간, L5.D는 OOS sub-window.
- vintage_policy (L1)와 information_set_type (L3)는 orthogonal. L1은 source decision, L3은 model decision.
- L1.G regime이 none이어도 sink는 빈 상태로 만들어짐. Downstream detect용.

### Sample

```yaml
1_data:
  fixed_axes:
    dataset: fred_md
    target_structure: single_target
  leaf_config:
    target: CPIAUCSL
```

```yaml
1_data:
  fixed_axes:
    dataset: fred_qd
    regime_definition: external_nber
  leaf_config:
    target: GDPC1
```
