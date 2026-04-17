# Phase 04 — Benchmark Evaluation 축

| Field | Value |
|-------|-------|
| Phase ID | phase-04 |
| Priority (inter-phase) | **P1** |
| Depends on | phase-01 |
| Unlocks | phase-07, phase-08 |
| Version tag target | v0.6 |
| Status | pending |

## 1. Goal

relative-RMSE horse race의 "relative to what?" 질문을 declarative 하게 표현합니다. CLSS-style relative RMSE가 sweep 기본 출력으로 자동 계산되어, 모든 variant가 baseline benchmark 대비 상대 성능을 명시적으로 공표합니다. 이 phase가 완료되면 Phase 1 sweep runner의 per-variant metrics에 relative_msfe, relative_rmse, relative_mae, oos_r2가 표준 필드로 등장합니다.


## 1a. Plan Revision — 2026-04-17 (코드 일치)

Phase 0~3 구현 후 registry 실측 결과, plan §4의 3축 모두 기존 axis 확장으로 정리. §3~§9의 plan 이름은 아래 매핑으로 읽음.

| Plan §4 name | Final action | Registry name |
|---|---|---|
| `benchmark_model` | 기존 확장 | `benchmark_family` (layer=1_data_task) — 9개 신규 값 status 승격 (operational/stub) |
| `benchmark_estimation_window` | 기존 확장 | `benchmark_window` (layer=4_evaluation) — fixed→operational, paper_exact_window→stub |
| `benchmark_by_target_horizon` | 기존 확장 | `benchmark_scope` (layer=4_evaluation) — target_specific/horizon_specific→operational |

Layer 차이: plan은 3축 모두 `4_evaluation` 가정이지만, `benchmark_family`는 기존 `1_data_task`라 유지 (다른 phase 코드 영향). benchmark_resolver는 4-layer 모듈에 생성.

기존 metrics 인프라: `_compute_metrics`가 이미 relative_msfe/rmse/mae/oos_r2 출력 중. Phase 4는 이를 `execution/evaluation/metrics.py`로 추출 + benchmark_resolver dispatch 추가.

## 2. Scope

**In scope:**
- 3개 신규 evaluation 축 (benchmark_model, benchmark_estimation_window, benchmark_by_target_horizon) registry entry
- `benchmark_resolver.py` — per (target, horizon, date) benchmark forecast 계산
- `metrics.py` 확장 — benchmark_resolver 결과 기반 relative metrics 자동 계산
- Test: 축 registry 테스트, resolver 4-variant 차이 테스트, relative metrics 테스트
- Docs: `user_guide/benchmarks.md` 대폭 확장, `examples/clss_replication_pattern.md` (CLSS replication **pattern** — not actual CLSS data)

**Out of scope:**
- Custom benchmark callable 확장 (이미 존재; 기존 API 유지)
- Multi-benchmark aggregation / panel-level benchmark → Phase 8
- Benchmark 간 상호 비교 (Phase 2 stat tests 담당)
- Survey consensus data ingestion (paper_specific 축에 stub만)

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 04.1 | 3 benchmark axes registry entries | **P0** | ~300 | `macrocast/registry/evaluation/benchmark_model.py` (신규), `benchmark_estimation_window.py` (신규), `benchmark_by_target_horizon.py` (신규) | `test_benchmark_axes.py` green |
| 04.2 | `benchmark_resolver.py` 구현 | **P0** | ~400 | `macrocast/execution/evaluation/benchmark_resolver.py` (신규) | `test_benchmark_resolver.py` 4 variant 차이 확인 |
| 04.3 | `metrics.py` relative metrics 자동 계산 | **P0** | ~200 | `macrocast/execution/evaluation/metrics.py` (수정) | relative_{msfe,rmse,mae}, oos_r2 출력 |
| 04.4 | Phase 4 tests | **P0** | ~500 | `tests/test_benchmark_axes.py`, `tests/test_benchmark_resolver.py`, `tests/test_relative_metrics.py` | 3개 test 전부 green |
| 04.5 | Phase 4 docs | P1 | ~300 | `docs/user_guide/benchmarks.md` (확장), `docs/examples/clss_replication_pattern.md` (신규) | RTD build green |

## 4. API / Schema Specifications

### 4.1 `benchmark_model` axis values

```python
# macrocast/registry/evaluation/benchmark_model.py
BENCHMARK_MODEL_VALUES = {
    "historical_mean":          "expanding-window mean of target",
    "rolling_mean":             "rolling-window mean (window_len required)",
    "random_walk":              "y_{t+h} = y_t (no-change forecast)",
    "ar_bic":                   "AR(p) with p selected by BIC on training window",
    "ar_fixed_p":               "AR(p) with fixed p (p required)",
    "ardi":                     "ARDI — AR with diffusion index factors (CLSS baseline)",
    "factor_model":             "static factor regression (n_factors required)",
    "var":                      "VAR(p) on target + auxiliary series",
    "expert_benchmark":         "user-provided callable (existing API)",
    "paper_specific_benchmark": "named paper baseline (registry lookup)",
    "survey_forecast":          "SPF / Greenbook survey consensus (data required)",
    "multi_benchmark_suite":    "list of benchmarks; relative metrics per each",
}
```

Status: `operational` for {historical_mean, rolling_mean, random_walk, ar_bic, ar_fixed_p, ardi, factor_model, var, expert_benchmark, multi_benchmark_suite}. `stub` for {paper_specific_benchmark, survey_forecast} (data ingestion이 v0.6 범위 밖).

### 4.2 `benchmark_estimation_window` axis values

```python
# macrocast/registry/evaluation/benchmark_estimation_window.py
BENCHMARK_ESTIMATION_WINDOW_VALUES = {
    "expanding":           "benchmark uses all data up to t",
    "rolling":             "benchmark uses rolling window (window_len required)",
    "fixed":               "benchmark trained once on initial window, never re-fit",
    "paper_exact_window":  "explicit (start, end) per paper replication",
}
```

### 4.3 `benchmark_by_target_horizon` axis values

```python
# macrocast/registry/evaluation/benchmark_by_target_horizon.py
BENCHMARK_BY_TARGET_HORIZON_VALUES = {
    "same_for_all":              "one benchmark for all (target, horizon) pairs",
    "target_specific":           "benchmark varies by target (dict of target→spec)",
    "horizon_specific":          "benchmark varies by horizon (dict of horizon→spec)",
    "target_horizon_specific":   "benchmark varies by (target, horizon) tuple",
}
```

### 4.4 `benchmark_resolver.py` API

```python
# macrocast/execution/evaluation/benchmark_resolver.py
from __future__ import annotations
import pandas as pd
from macrocast.execution.types import RecipeSpec, EvaluationSpec

def resolve_benchmark_forecasts(
    *,
    predictions: pd.DataFrame,         # (date, target, horizon, prediction) index
    recipe: RecipeSpec,
    evaluation_spec: EvaluationSpec,
    raw_target_panel: pd.DataFrame,    # y for benchmark fitting
) -> pd.DataFrame:
    """Compute benchmark forecasts per (target, horizon, date).

    Returns a DataFrame with columns:
    - date, target, horizon, benchmark_name, benchmark_prediction

    Dispatch rules:
    - evaluation_spec.benchmark_by_target_horizon determines per-cell benchmark
    - evaluation_spec.benchmark_model + benchmark_estimation_window selects the
      actual estimator
    - multi_benchmark_suite returns **stacked** rows (one row per benchmark_name)
    """
```

**Invariants:**
- 출력 DataFrame의 (date, target, horizon) 집합은 predictions와 정확히 일치 (multi_benchmark_suite의 경우 benchmark_name 축으로 cross join)
- estimation_window=rolling은 반드시 window_len>0
- expert_benchmark는 기존 callable API 위임

### 4.5 Relative metrics 자동 계산

```python
# macrocast/execution/evaluation/metrics.py (확장)
def compute_relative_metrics(
    *,
    predictions: pd.DataFrame,
    benchmark_forecasts: pd.DataFrame,
    actuals: pd.Series,
) -> dict:
    """Compute relative_msfe, relative_rmse, relative_mae, oos_r2.

    Conventions:
    - relative_msfe = msfe(model) / msfe(benchmark)
    - relative_rmse = rmse(model) / rmse(benchmark)  # CLSS 기준
    - relative_mae  = mae(model)  / mae(benchmark)
    - oos_r2        = 1 - sse(model) / sse(benchmark)
    - benchmark_forecasts가 multi-row (multi_benchmark_suite) → relative_*의 **dict**
    """
```

Metrics dict에 추가되는 키 (기존 msfe/rmse/mae 유지, 신규 추가):
```json
{
  "msfe": 0.023, "rmse": 0.152, "mae": 0.118,
  "relative_msfe": {"ar_bic": 0.78, "historical_mean": 0.62},
  "relative_rmse": {"ar_bic": 0.88, "historical_mean": 0.79},
  "relative_mae":  {"ar_bic": 0.85, "historical_mean": 0.75},
  "oos_r2":        {"ar_bic": 0.22, "historical_mean": 0.38}
}
```

Benchmark 축이 single (multi_benchmark_suite 아님)인 경우 relative_* 는 scalar.

## 5. File Layout

**신규:**
- `macrocast/registry/evaluation/benchmark_model.py`
- `macrocast/registry/evaluation/benchmark_estimation_window.py`
- `macrocast/registry/evaluation/benchmark_by_target_horizon.py`
- `macrocast/execution/evaluation/benchmark_resolver.py`
- `tests/test_benchmark_axes.py`
- `tests/test_benchmark_resolver.py`
- `tests/test_relative_metrics.py`
- `docs/examples/clss_replication_pattern.md`

**수정:**
- `macrocast/execution/evaluation/metrics.py` — relative metrics auto-compute
- `macrocast/registry/evaluation/__init__.py` — 3개 신규 축 export
- `docs/user_guide/benchmarks.md` — 대폭 확장

## 6. Test Strategy

### `tests/test_benchmark_axes.py`
- 3개 축 registry에 기대 값 전부 등록됨
- operational vs stub status 플래그 정상
- axis_value 미등록 → validation error (기존 compiler pipeline 통과)

### `tests/test_benchmark_resolver.py`
- 4개 variant 비교 (historical_mean, random_walk, ar_bic, ardi)로 동일 input → **서로 다른** benchmark 시계열 반환
- (date, target, horizon) 일치성 검증
- multi_benchmark_suite → stacked rows (benchmark_name 축 추가)
- rolling estimation_window: window_len=60 vs expanding → 명확히 다른 결과

### `tests/test_relative_metrics.py`
- 인위적 perfect model (prediction=actual) → relative_rmse=0, oos_r2=1
- 인위적 worst-than-benchmark (prediction=2*actual) → relative_rmse>1, oos_r2<0
- multi_benchmark_suite → dict 반환, 각 benchmark별 key

## 7. Acceptance Gate

- [ ] Phase 1 gate 선통과
- [ ] 3개 benchmark 축 registry에 operational status로 등록됨
- [ ] `resolve_benchmark_forecasts` 공개 API 내보냄
- [ ] `metrics.json`에 relative_msfe/relative_rmse/relative_mae/oos_r2 키 기본 출력
- [ ] 4 variant (historical_mean, random_walk, ar_bic, ardi) smoke test green
- [ ] 기존 test + Phase 0/1 test + Phase 4 신규 test 전부 green
- [ ] `docs/user_guide/benchmarks.md` 확장판 + `docs/examples/clss_replication_pattern.md` RTD build green

## 8. Docs Deliverables

**신규:**
- `docs/examples/clss_replication_pattern.md` — CLSS 스타일 relative-RMSE horse race **pattern** 설명 (실제 CLSS 데이터 아님, FRED-MD INDPRO 위에서 재현)

**확장:**
- `docs/user_guide/benchmarks.md`
  - 3개 축 조합으로 표현 가능한 benchmark matrix
  - `multi_benchmark_suite` → 여러 baseline 동시 보고 cookbook
  - Expert callable API (기존) 및 paper_specific stub 현황

## 9. Migration Notes

- 기존 recipe가 `benchmark_family`만 쓰고 있었다면 Phase 4 이후에도 작동 (benchmark_model 축 default = historical_mean)
- metrics.json 스키마에 relative_* 키 **추가** (기존 키 유지) → downstream 소비자 영향 없음
- `multi_benchmark_suite` 선택 시 relative_* 값이 dict로 바뀜 → consumer가 `isinstance(v, dict)` 분기 필요 (docs 고지)

## 10. Cross-references

- Infra files used: `plans/infra/metrics_schema.md`, `plans/infra/axis_registry_pattern.md`
- Phase dependencies:
  - Phase 1 sweep runner — `benchmark_model`이 sweep_axes에 올 수 있음 (model × benchmark 2D sweep)
  - Phase 7 paper replication — paper_specific_benchmark stub 이 Phase 7 에서 구체화
  - Phase 8 multi-target — benchmark_by_target_horizon=target_specific 가 multi-target 시나리오에서 유효
- ADRs referenced: ADR-005 (benchmark registry design), ADR-007 (relative metric conventions)
- Coverage Ledger rows resolved:
  - Layer 4 `benchmark_model` → operational (9/12)
  - Layer 4 `benchmark_estimation_window` → operational (4/4)
  - Layer 4 `benchmark_by_target_horizon` → operational (4/4)

## 11. GitHub Issue Map

- Epic: (TBD at kickoff — [PHASE-04] Benchmark evaluation axes + relative metrics)
- Sub-task issues: 5개 (04.1 ~ 04.5)

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 §Phase 4에서 추출)
- 2026-04-17 (Phase 4 kickoff): §1a 추가 — registry 실측 일치 axis 매핑 (3축 모두 기존 확장)
