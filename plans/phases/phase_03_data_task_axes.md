# Phase 03 — Data/Task Axes + Preprocessing Separation

| Field | Value |
|-------|-------|
| Phase ID | phase-03 |
| Priority (inter-phase) | **P1** |
| Depends on | phase-01 |
| Unlocks | phase-04 |
| Version tag target | v0.5 |
| Status | pending |

## 1. Goal

7개 data/task 축 (`release_lag_rule`, `missing_availability`, `variable_universe`, `minimum_train_size`, `break_segmentation`, `horizon_list`, `scale_at_evaluation`)과 preprocessing의 `separation_rule` 축 1개를 추가하여 recipe의 methodological defensibility를 완성합니다. 현재 이 축들은 registry에 부재하거나 leaf로만 박혀 있어 sweep/비교가 불가능하며, 이는 vintage-aware evaluation 및 regime 분석의 전제조건입니다.

## 1a. Plan Revision — 2026-04-17 (코드 일치)

Phase 0/1/2 구현 후 registry 실측 결과, plan §4의 8개 축 중 5개는 신규 추가, 3개는 기존 axis 재사용/확장으로 정리. §3~§9의 plan 이름은 아래 매핑으로 읽음.

| Plan §4 name | Final action | Registry name |
|---|---|---|
| `release_lag_rule` | 신규 | `release_lag_rule` (layer=1_data_task) |
| `missing_availability` | 신규 | `missing_availability` (layer=1_data_task) |
| `variable_universe` | 신규 | `variable_universe` (layer=1_data_task) |
| `horizon_list` | 신규 | `horizon_list` (layer=1_data_task) |
| `separation_rule` | 신규 | `separation_rule` (layer=2_preprocessing) |
| `minimum_train_size` | 기존 재사용 | `min_train_size` (값 5개 일치) |
| `break_segmentation` | 기존 재사용 | `structural_break_segmentation` (값 6개 일치) |
| `scale_at_evaluation` | 기존 확장 | `evaluation_scale` (raw_level→original_scale, +both) |

## 2. Scope

**In scope:**
- 7개 data/task axis 파일 신규 등록
- 1개 preprocessing axis (`separation_rule`) 신규 등록
- 런타임 wiring — 8개 축 모두 `execute_recipe()`에서 실제 동작에 영향
- Phase 1 sweep runner와 통합 (`horizon_list` as axis → sweep_axes 가능)
- `macrocast/preprocessing/separation.py` 신규 (leak check 런타임)
- `macrocast/raw/windowing.py` 신규 (break/minimum_train 확장)
- 3개 신규 test 파일

**Out of scope:**
- 신규 data source (FRED 외 확장) — v1.1 Phase 10
- Advanced break detection 알고리즘 (structural break, Bai-Perron 등 고급) — v2
- Release calendar 자동 fetch — v1.1
- Real-time vintage DB 통합 — v2

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 03.1 | 7개 data_task axis 등록 | **P0** | ~500 | `macrocast/registry/data/{release_lag_rule,missing_availability,variable_universe,minimum_train_size,break_segmentation,horizon_list,scale_at_evaluation}.py` (신규 7) | `test_data_task_axes.py` enum 검증 green |
| 03.2 | `separation_rule` axis 등록 | **P0** | ~80 | `macrocast/registry/preprocessing/separation_rule.py` (신규) | enum 등록 + layer=2 |
| 03.3 | 런타임 wiring — `execute_recipe()` | **P0** | ~600 | `macrocast/execution/build.py` (8개 축 consume), `macrocast/raw/windowing.py` (신규), `macrocast/preprocessing/separation.py` (신규) | `test_data_task_axes_runtime.py` + `test_separation_rule.py` green |
| 03.4 | Phase 3 tests | **P0** | ~450 | `tests/test_data_task_axes.py`, `tests/test_data_task_axes_runtime.py`, `tests/test_separation_rule.py` (신규 3) | 3개 test 전부 green |
| 03.5 | Phase 3 docs | P1 | ~350 | `docs/user_guide/data_task_axes.md` (신규), `docs/user_guide/preprocessing_separation.md` (신규), `docs/math/vintage_and_release_lag.md` (신규) | RTD build green |

## 4. API / Schema Specifications

### 4.1 7개 Data/Task Axes

| Axis | Layer | Values |
|------|:-----:|--------|
| `release_lag_rule` | 1 | `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag`, `calendar_exact_lag`, `lag_conservative`, `lag_aggressive` |
| `missing_availability` | 1 | `complete_case_only`, `available_case`, `target_date_drop_if_missing`, `x_impute_only`, `real_time_missing_as_missing`, `state_space_fill`, `factor_fill`, `em_fill` |
| `variable_universe` | 1 | `all_variables`, `preselected_core`, `category_subset`, `paper_replication_subset`, `target_specific_subset`, `expert_curated_subset`, `stability_filtered_subset`, `correlation_screened_subset`, `feature_selection_dynamic_subset` |
| `minimum_train_size` | 1 | `fixed_n_obs`, `fixed_years`, `model_specific_min_train`, `target_specific_min_train`, `horizon_specific_min_train` |
| `break_segmentation` | 1 | `none`, `pre_post_crisis`, `pre_post_covid`, `user_break_dates`, `break_test_detected`, `rolling_break_adaptive` |
| `horizon_list` | 1 | `arbitrary_grid`, `default_1_3_6_12`, `short_only_1_3`, `long_only_12_24`, `paper_specific` |
| `scale_at_evaluation` | 1 | `transformed_scale`, `original_scale`, `both` |

```python
# macrocast/registry/data/release_lag_rule.py
from macrocast.registry.base import register_axis, AxisSpec

register_axis(AxisSpec(
    name="release_lag_rule",
    layer=1,
    status="operational",
    values=(
        "ignore_release_lag",
        "fixed_lag_all_series",
        "series_specific_lag",
        "calendar_exact_lag",
        "lag_conservative",
        "lag_aggressive",
    ),
    description=(
        "Rule that governs how publication lag is applied to predictor availability "
        "at forecast origin t. Controls information set realism (pseudo-real-time vs revised)."
    ),
))
```

### 4.2 `separation_rule` Axis

```python
# macrocast/registry/preprocessing/separation_rule.py
register_axis(AxisSpec(
    name="separation_rule",
    layer=2,
    status="operational",
    values=(
        "strict_separation",         # fit on train only, transform on test
        "shared_transform_then_split",  # fit on full, then split (leak)
        "joint_preprocessor",         # paper-faithful shared prep
        "target_only_transform",      # transform y only
        "X_only_transform",           # transform X only
    ),
    description=(
        "Governs train/test leakage discipline during preprocessing. "
        "strict_separation is default; other modes are explicit opt-ins for replication."
    ),
))
```

### 4.3 Runtime Wiring Map

| Axis | 소비 위치 | 동작 |
|------|-----------|------|
| `release_lag_rule` | `_load_raw_for_recipe` | vintage publish 시점 적용 (series별 lag table 로드) |
| `missing_availability` | target / X construction | drop/impute 정책 |
| `variable_universe` | feature subset | Layer 1/3 사이에서 X columns 필터 |
| `minimum_train_size` | `_minimum_train_size()` 확장 | 기존 함수를 축 값에 따라 분기 |
| `break_segmentation` | OOS block 생성 | train/test split 분할 규칙 변경 |
| `horizon_list` | sweep plan (Phase 1) + eval | axis로 등록되어 sweep_axes에 넣을 수 있음 |
| `scale_at_evaluation` | evaluation 전 역변환 | metrics 계산 직전 |
| `separation_rule` | preprocess pipeline | fit/transform 경계 enforce + leak check |

### 4.4 `macrocast/preprocessing/separation.py`

```python
# macrocast/preprocessing/separation.py
from __future__ import annotations
import pandas as pd

class LeakError(RuntimeError):
    """Raised when separation_rule invariants are violated."""

def apply_separation_rule(
    *,
    rule: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor,  # sklearn-like
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Apply train/test preprocessing per separation_rule.

    - strict_separation: preprocessor.fit(X_train); transform both
    - shared_transform_then_split: preprocessor.fit(pd.concat([X_train, X_test]))  # leak
    - joint_preprocessor: user-supplied joint pipeline (paper replication)
    - target_only_transform: transform y_train/y_test, X passthrough
    - X_only_transform: transform X_train/X_test, y passthrough

    Post-condition invariants (enforced):
    - strict_separation: fitted state must NOT depend on X_test rows (hash check)
    - all modes: shapes preserved, index alignment preserved
    """
```

### 4.5 `macrocast/raw/windowing.py`

```python
# macrocast/raw/windowing.py
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class WindowSpec:
    minimum_train_rule: str
    break_rule: str
    break_dates: tuple[pd.Timestamp, ...] = ()

def compute_train_test_blocks(
    *,
    index: pd.DatetimeIndex,
    spec: WindowSpec,
    horizon: int,
    model_family: str,
    target: str,
) -> list[tuple[slice, slice]]:
    """Produce (train_slice, test_slice) pairs given break/min_train policy.

    - break_rule=none → single block
    - pre_post_crisis → 2 blocks split at 2008-09
    - pre_post_covid → 2 blocks split at 2020-03
    - user_break_dates → N+1 blocks split at given dates
    - break_test_detected → run Chow/Bai-Perron, split at detected dates
    - rolling_break_adaptive → sliding adaptive windows

    minimum_train_size values (fixed_n_obs, fixed_years, model/target/horizon_specific)
    gate block eligibility (too-short blocks dropped).
    """
```

## 5. File Layout

**신규:**
- `macrocast/registry/data/__init__.py`
- `macrocast/registry/data/release_lag_rule.py`
- `macrocast/registry/data/missing_availability.py`
- `macrocast/registry/data/variable_universe.py`
- `macrocast/registry/data/minimum_train_size.py`
- `macrocast/registry/data/break_segmentation.py`
- `macrocast/registry/data/horizon_list.py`
- `macrocast/registry/data/scale_at_evaluation.py`
- `macrocast/registry/preprocessing/__init__.py`
- `macrocast/registry/preprocessing/separation_rule.py`
- `macrocast/preprocessing/__init__.py`
- `macrocast/preprocessing/separation.py`
- `macrocast/raw/windowing.py`
- `tests/test_data_task_axes.py`
- `tests/test_data_task_axes_runtime.py`
- `tests/test_separation_rule.py`
- `docs/user_guide/data_task_axes.md`
- `docs/user_guide/preprocessing_separation.md`
- `docs/math/vintage_and_release_lag.md`

**수정:**
- `macrocast/execution/build.py` — 8개 축 consume (release_lag_rule, missing_availability, variable_universe, minimum_train_size, break_segmentation, horizon_list, scale_at_evaluation, separation_rule)
- `macrocast/raw/loader.py` — `release_lag_rule` 적용 훅
- `macrocast/__init__.py` — `apply_separation_rule`, `compute_train_test_blocks` 공개 API
- `macrocast/compiler/sweep_plan.py` — `horizon_list` as sweep axis 허용 (Phase 1 확장)

## 6. Test Strategy

### `tests/test_data_task_axes.py`
- 8개 신규 axis 모두 registry에 등록 + status = operational
- 각 axis의 value set이 §4.1/§4.2 표와 일치
- Layer 필드 정확 (`release_lag_rule`~`scale_at_evaluation`: layer 1, `separation_rule`: layer 2)
- `horizon_list`가 `sweep_axes` 허용 목록에 포함

### `tests/test_data_task_axes_runtime.py`
- **각 축이 실제로 영향을 주는지** smoke test:
  - `release_lag_rule=ignore_release_lag` vs `fixed_lag_all_series` → 같은 recipe에서 train X matrix가 다름
  - `missing_availability=complete_case_only` vs `x_impute_only` → row count 다름
  - `variable_universe=all_variables` vs `preselected_core` → X column count 다름
  - `minimum_train_size=fixed_n_obs` (n=60) vs (n=120) → 첫 test point 날짜 다름
  - `break_segmentation=pre_post_crisis` → train/test block 2개 생성
  - `horizon_list=short_only_1_3` vs `default_1_3_6_12` → horizons 다름
  - `scale_at_evaluation=transformed_scale` vs `original_scale` → MSFE 수치 다름 (단, `both`는 2 dict)

### `tests/test_separation_rule.py`
- 5개 mode 각각 leak check:
  - `strict_separation`: preprocessor.fit state가 X_test hash와 독립 (monkeypatch로 검증)
  - `shared_transform_then_split`: fit state가 X_test에 의존 (의도된 leak, 경고 로그)
  - `joint_preprocessor`: user pipeline call trace 확인
  - `target_only_transform`: X 변경 없음
  - `X_only_transform`: y 변경 없음
- Invalid rule → `ValueError`
- `strict_separation` + deterministic input → output reproducible

## 7. Acceptance Gate

- [ ] Phase 1 gate 선통과
- [ ] 8개 신규 axis (7 data + 1 preprocessing) registry에 등록 + status = operational
- [ ] 기존 291 test + Phase 0/1/2 test + Phase 3 신규 3개 test 전부 green
- [ ] `execute_recipe()` 내부에서 8개 축 모두 consume (grep으로 각 축 명시 확인)
- [ ] `horizon_list` axis로 `sweep_axes: {horizon_list: [...]}` recipe sweep 실행 성공
- [ ] `strict_separation` 모드에서 leak 시도하면 `LeakError` raise
- [ ] Phase 3 docs 3종 RTD build green
- [ ] `docs/math/vintage_and_release_lag.md` — release lag 적용 수식 재현 가능

## 8. Docs Deliverables

**신규:**
- `docs/user_guide/data_task_axes.md` — 7축 각각의 use case + YAML 예시 + 기본값 가이드
- `docs/user_guide/preprocessing_separation.md` — 5 mode leak/no-leak 설명 + 논문 replication 시나리오
- `docs/math/vintage_and_release_lag.md` — publication lag 적용 수식, series-specific vs calendar-exact 구분, pseudo-real-time 정의

## 9. Migration Notes

- Breaking change 없음 (순수 추가 축)
- 기존 recipe YAML: 8개 축 미지정 시 기본값 적용
  - `release_lag_rule` default: `ignore_release_lag` (기존 동작 유지)
  - `missing_availability` default: `complete_case_only`
  - `variable_universe` default: `all_variables`
  - `minimum_train_size` default: `fixed_n_obs` (n from 기존 `_minimum_train_size()`)
  - `break_segmentation` default: `none`
  - `horizon_list` default: `arbitrary_grid` (기존 leaf_config 방식)
  - `scale_at_evaluation` default: `transformed_scale`
  - `separation_rule` default: `strict_separation`
- `horizon_list` leaf_config 호환: `leaf_config.horizons: [1,3,6,12]` 방식은 `horizon_list=arbitrary_grid` 로 해석 (기본 default)
- 기존 291 test는 모든 축이 default라 영향 없음 (회귀 test로 확인)

## 10. Cross-references

- Infra files used: `plans/infra/vintage_policy.md` (신규, Phase 3 kickoff 시 작성), `plans/infra/windowing_spec.md` (신규)
- ADRs referenced: ADR-002 (axis layering — Layer 1 data/task vs Layer 2 preprocessing), ADR-005 (leak discipline default)
- Coverage Ledger rows resolved:
  - Layer 1 `release_lag_rule` (6 values) → operational
  - Layer 1 `missing_availability` (8 values) → operational
  - Layer 1 `variable_universe` (9 values) → operational
  - Layer 1 `minimum_train_size` (5 values) → operational
  - Layer 1 `break_segmentation` (6 values) → operational
  - Layer 1 `horizon_list` (5 values) → operational (as axis, not leaf)
  - Layer 1 `scale_at_evaluation` (3 values) → operational
  - Layer 2 `separation_rule` (5 values) → operational
- Upstream: Phase 1 sweep runner (horizon_list as sweep axis 활성화)
- Downstream: Phase 4 (feature builder가 variable_universe 결과를 소비)

## 11. GitHub Issue Map

- Epic: (TBD at kickoff — [PHASE-03] Data/task axes + preprocessing separation)
- Sub-task issues: 5개 (03.1~03.5)

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 §Phase 3에서 추출)
- 2026-04-17 (Phase 3 kickoff): §1a 추가 — registry 실측 일치 axis 매핑 (5신규 + 3재사용/확장)

## 13. References

- `plans/ultraplan_v2.2.md` §Phase 3 — 원본 사양
- ADR-002 — axis layer 정의
- ADR-005 — leak discipline 기본값
- `plans/phase0_audit_2026_04_17.md` — 현재 `_minimum_train_size()` 구현 상태 (확장 대상)
