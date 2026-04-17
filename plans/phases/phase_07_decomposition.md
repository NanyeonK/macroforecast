# Phase 07 — §4.5 Decomposition Engine

| Field | Value |
|-------|-------|
| Phase ID | phase-07 |
| Priority (inter-phase) | **P0** |
| Depends on | phase-01, phase-06 |
| Unlocks | phase-08 |
| Version tag target | v0.9 (with phase-08) |
| Status | in_progress |

## 1. Goal

macrocast를 cite-worthy로 만드는 레이어입니다. §4.5에서 user가 "macrocast의 identity"로 명시했습니다. Horse race 결과를 단순히 "누가 이겼는가"에서 멈추지 않고, **forecast-error variance component**으로 분해하여 "왜 이겼는가 / 어떤 axis가 variance의 몇 %를 설명하는가"를 reviewer-defensible하게 attribute합니다. ANOVA baseline이 v1.0, Shapley attribution은 v1.1 enhancement로 예약.

## 2. Scope

**In scope:**
- `DecompositionPlan` / `DecompositionResult` dataclass + Engine API
- ANOVA baseline attribution (one-way, SS_between_axis / SS_total)
- 8 component plug-in (nonlinearity, regularization, cv_scheme, loss, preprocessing, feature_builder, benchmark, importance)
- `AxisDefinition`에 `component` metadata 필드 추가
- `decomposition_result.parquet` schema freeze
- Synthetic known-effect 회귀 테스트

**Out of scope:**
- Shapley attribution — v1.1 enhancement (ADR-002: ANOVA before Shapley)
- Interaction decomposition beyond one-way ANOVA — v1.1
- Bootstrap CI for component shares — v1.1

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 07.1 | `component` metadata 추가 to AxisDefinition | **P0** | ~80 | `macrocast/registry/base.py`, 기존 axis 파일 전부 (component 속성) | 모든 axis 파일에 component 명시 |
| 07.2 | ANOVA attribution 구현 | **P0** | ~250 | `macrocast/decomposition/attribution.py` (신규) | `test_decomposition_engine.py` synthetic green |
| 07.3 | 8 component plug-in | P1 | ~400 | `macrocast/decomposition/components/{8 files}.py` (신규) | 각 component가 axis → 기여 여부 판정 |
| 07.4 | `decomposition_result.parquet` schema freeze | P1 | ~80 | `plans/infra/decomposition_result_schema.md`, `macrocast/decomposition/schema.py` (신규) | parquet JSONSchema-equivalent validation |
| 07.5 | Engine API (`DecompositionPlan`, `run_decomposition`) | **P0** | ~300 | `macrocast/decomposition/engine.py` (신규) | `test_decomposition_engine.py` green |
| 07.6 | Phase 7 tests | **P0** | ~350 | `tests/test_decomposition_engine.py`, `tests/test_decomposition_component_mapping.py`, `tests/test_decomposition_stability.py` | 3개 test 전부 green |
| 07.7 | Phase 7 docs | P1 | ~350 | `docs/user_guide/decomposition_tutorial.md`, `docs/math/decomposition_attribution.md`, `docs/api/decomposition.md` | RTD build green |

## 4. API / Schema Specifications

### 4.1 `AxisDefinition.component` 필드 확장 (revised 2026-04-17)

The real dataclass is wider than the original sketch; this phase adds **one optional field**, defaulted to `None`, so every pre-existing axis stays valid without touching its file:

```python
# macrocast/registry/base.py (edit)
@dataclass(frozen=True)
class AxisDefinition:
    axis_name: str                                                    # existing
    layer: str                                                        # existing
    axis_type: Literal["enum", "numeric", "callable", "plugin"]       # existing
    entries: tuple[BaseRegistryEntry, ...]                            # existing
    compatible_with: dict[str, tuple[str, ...]]                       # existing
    incompatible_with: dict[str, tuple[str, ...]]                     # existing
    registry_type: Literal[...] = "enum_registry"                     # existing
    default_policy: Literal["fixed", "sweep", "conditional"] = "fixed"# existing
    component: str | None = None                                      # NEW
```

Concrete axis-to-component mappings landed in Phase 7 (v0.9):

| axis (existing in registry) | component |
|---|---|
| `scaling_policy` | `preprocessing` |
| `dimensionality_reduction_policy` | `preprocessing` |
| `feature_selection_policy` | `preprocessing` |
| `target_transform_policy` | `preprocessing` |
| `x_transform_policy` | `preprocessing` |
| `tcode_policy` | `preprocessing` |
| `model_family` | `nonlinearity` |
| `feature_builder` | `feature_builder` |
| `benchmark_family` | `benchmark` |
| `importance_method` | `importance` |

The remaining three components (`regularization`, `cv_scheme`, `loss`) are defined in the enum but do **not** yet map to an existing axis — the original plan named `regularization_penalty` / `cv_strategy` / `loss_function`, none of which exist in the current registry. Future axes that surface those dimensions can be tagged as-they-land; Phase 7 ships the slot.

All other 120+ axis files stay unchanged because the new field defaults to `None`.
### 4.2 `DecompositionPlan` / Engine API

```python
# macrocast/decomposition/engine.py
from dataclasses import dataclass

@dataclass(frozen=True)
class DecompositionPlan:
    components_to_decompose: list[str]   # ["preprocessing", "nonlinearity"]
    attribution_method: str              # "anova" (v1.0) | "shapley" (v1.1)
    study_manifest_path: str

@dataclass(frozen=True)
class DecompositionResult:
    study_id: str
    plan: DecompositionPlan
    result_parquet_path: str
    report_json_path: str
    per_component_shares: dict[str, float]  # {"preprocessing": 0.72, ...}

def run_decomposition(plan: DecompositionPlan) -> DecompositionResult:
    """Given a completed sweep study, attribute primary-metric variance to axes.

    Algorithm (ANOVA baseline, aggregate level — revised 2026-04-17):
    1. Load study_manifest.json → variants[].axis_values, variants[].metrics_summary
    2. Build long-format DataFrame: one row per (variant, horizon) carrying the
       variant's *primary metric* (default: msfe) lifted from
       metrics_summary['metrics_by_horizon'][<h>][<primary_metric>].
    3. For each component c in plan.components_to_decompose:
       - Identify which axes belong to component c (AxisDefinition.component == c)
       - Pool each axis's one-way ANOVA: ss_between_axis / ss_total
       - Emit component_share_c = Σ ss_between over axes in c / ss_total
    4. Emit decomposition_result.parquet + decomposition_report.json

    Per-(oos_date) decomposition (reading each variant's predictions.csv for
    per-date squared error) is a v1.1 enhancement — aggregate-level is enough
    for the Phase 7 acceptance gate and for §4.5 identity.
    """
```

### 4.3 ANOVA attribution detail

```python
# macrocast/decomposition/attribution.py
def one_way_anova_ss(df: pd.DataFrame, axis_col: str, metric_col: str) -> tuple[float, float, int]:
    """Return (ss_between, ss_total, n_groups).

    ss_between = sum_g n_g * (mean_g - mean_all)^2
    ss_total   = sum_i (y_i - mean_all)^2
    """
```

Significance p-value는 scipy.stats.f_oneway로 부가 계산. H0: 모든 axis-group 평균 동일.

### 4.4 `decomposition_result.parquet` schema

| column | type | 설명 |
|--------|------|------|
| component | string | "preprocessing" / "nonlinearity" / ... |
| ss_between | float64 | axis-level sum-of-squares |
| ss_total | float64 | total variance (모든 variant × horizon × date) |
| share | float64 | ss_between / ss_total (0..1) |
| n_variants | int32 | component에 기여하는 variant 수 |
| significance_p | float64 | F-test p-value (NaN 허용 if n_groups < 2) |

`decomposition_report.json`은 plan + per_component_shares + 실행 metadata 요약.

### 4.5 Autonomous-execution decisions (pinned 2026-04-17)

- **Observation granularity**: aggregate — one row per (variant, horizon). Per-date loss attribution deferred to v1.1.
- **Primary metric default**: `msfe`. `DecompositionPlan.primary_metric: str = "msfe"` is the new knob.
- **Component enum**: `{"nonlinearity", "regularization", "cv_scheme", "loss", "preprocessing", "feature_builder", "benchmark", "importance"}` plus implicit `None`. Frozen as `COMPONENT_NAMES` in `macrocast/decomposition/components/__init__.py`.
- **Missing axes** (`regularization_penalty`, `cv_strategy`, `loss_function` from the plan's original sample list): not mapped in v0.9 — no existing axis has those semantics. The enum slots are reserved for v1.1 axes.
- **Parquet engine**: `pyarrow` (already in `[parquet]` optional extra). If missing, `ExecutionError` with install hint surfaces before any decomposition work runs.
- **Determinism**: axis iteration order is sorted lexicographically; variant rows within each ANOVA group are sorted by `variant_id`; all float arithmetic uses `numpy.float64`.
- **Empty `components_to_decompose`**: returns a valid `DecompositionResult` with `per_component_shares={}` and a parquet with zero rows — clean no-op.
- **Zero-variance cases**: if `ss_total == 0` (all variants identical metric) → all component shares set to 0.0, `significance_p = NaN`.
- **Single-group cases**: if a component has ≤1 distinct axis value across the sweep → that component's share is 0.0, `significance_p = NaN`.

## 5. File Layout

**신규:**
- `macrocast/decomposition/__init__.py`
- `macrocast/decomposition/engine.py`
- `macrocast/decomposition/attribution.py`
- `macrocast/decomposition/schema.py`
- `macrocast/decomposition/components/__init__.py`
- `macrocast/decomposition/components/nonlinearity.py`
- `macrocast/decomposition/components/regularization.py`
- `macrocast/decomposition/components/cv_scheme.py`
- `macrocast/decomposition/components/loss.py`
- `macrocast/decomposition/components/preprocessing.py`
- `macrocast/decomposition/components/feature_builder.py`
- `macrocast/decomposition/components/benchmark.py`
- `macrocast/decomposition/components/importance.py`
- `tests/test_decomposition_engine.py`
- `tests/test_decomposition_component_mapping.py`
- `tests/test_decomposition_stability.py`
- `docs/user_guide/decomposition_tutorial.md`
- `docs/math/decomposition_attribution.md`
- `docs/api/decomposition.md`
- `plans/infra/decomposition_result_schema.md`

**수정:**
- `macrocast/registry/base.py` — `AxisDefinition.component` 필드 추가
- `macrocast/registry/stage*/`의 axis 정의 파일 전부 (component 속성 부여)
- `macrocast/__init__.py` — 공개 API (`DecompositionPlan`, `run_decomposition`, `DecompositionResult`)

## 6. Test Strategy

### `tests/test_decomposition_engine.py`
- known-effect synthetic sweep: preprocessing axis effect가 design에 의해 dominant (noise ≪ preprocessing_effect)
  → decomposition 결과에서 `preprocessing` component share > 0.7
- trivial 2-axis sweep: ANOVA 수치가 손으로 계산한 값과 일치
- `components_to_decompose=[]` → empty result (clean no-op)

### `tests/test_decomposition_component_mapping.py`
- 레지스트리 모든 axis가 component 속성을 가짐 (None도 명시적)
- component 이름이 8개 허용 값 + None 집합에 속함

### `tests/test_decomposition_stability.py`
- 같은 study_manifest.json을 두 번 `run_decomposition` → 동일 `decomposition_result.parquet` byte-identical
- share 값이 float determinism 오차 범위 (1e-12) 내 일치

## 7. Acceptance Gate

- [ ] Phase 6 gate 선통과
- [ ] 8 component 모두 axis 매핑 완료 (None 포함 명시)
- [ ] Synthetic preprocessing-dominant sweep → `preprocessing` share > 0.7
- [ ] `decomposition_result.parquet`이 schema v1 준수
- [ ] `run_decomposition()` 결정성 (두 번 실행 byte-identical)
- [ ] 공개 API export: `DecompositionPlan`, `run_decomposition`, `DecompositionResult`
- [ ] Phase 7 docs 3종 RTD latest build green (특히 `decomposition_tutorial.md` = v1.0 landing의 identity page)
- [ ] 기존 test + Phase 0-6 test + Phase 7 신규 test 전부 green

## 8. Docs Deliverables

**신규:**
- `docs/user_guide/decomposition_tutorial.md` — **identity page, v1.0 landing에서 직접 링크**
- `docs/math/decomposition_attribution.md` — ANOVA / Shapley 수학 정의, component 분해 공식
- `docs/api/decomposition.md` — autodoc (`DecompositionPlan`, `run_decomposition`)

## 9. Migration Notes

- `AxisDefinition.component`은 optional 필드로 추가 (기존 axis도 None으로 호환)
- Breaking 없음 — decomposition은 opt-in (사용자가 `run_decomposition` 명시적으로 호출)
- study_manifest.json schema 불변 (Phase 1 v1 그대로)
- Future: Shapley attribution 도입 시 `attribution_method="shapley"`로 확장만 하면 됨

## 10. Cross-references

- Infra files used: `plans/infra/decomposition_result_schema.md`, `plans/infra/adr/ADR-005-component-metadata-field.md`, `plans/infra/adr/ADR-002-anova-before-shapley.md`
- ADRs referenced: ADR-005 (component metadata field), ADR-002 (ANOVA baseline before Shapley)
- Coverage Ledger rows resolved:
  - §4.5 "decomposition as identity" → operational
  - 8-component attribution → operational (ANOVA baseline)
  - Shapley attribution → deferred (v1.1)

## 11. GitHub Issue Map

- Epic: (TBD at kickoff — [PHASE-07] §4.5 Decomposition engine)
- Sub-task issues: 7개 (07.1~07.7)

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 §Phase 7에서 추출)
- 2026-04-17 (kickoff revision): §4.1 AxisDefinition sketch rewritten to match the actual dataclass (adds `component: str | None = None` as a single optional field; 120+ existing axes stay intact); concrete axis-to-component mapping table added in place of the sample list so the missing `regularization_penalty` / `cv_strategy` / `loss_function` axes don't block implementation. §4.2 algorithm simplified to aggregate-level ANOVA (per-(variant, horizon)) — per-date loss attribution deferred to v1.1. §10 ADR paths corrected from `plans/adr/` to `plans/infra/adr/`. §4.5 added with the autonomous-execution decisions pinned (observation granularity, primary-metric default, component enum, parquet engine, determinism policy, zero-variance handling).
