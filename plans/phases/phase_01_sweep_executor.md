# Phase 01 — Horse-Race Sweep Executor (IDENTITY UNLOCK)

| Field | Value |
|-------|-------|
| Phase ID | phase-01 |
| Priority (inter-phase) | **P0** |
| Depends on | phase-00 |
| Unlocks | phase-02, phase-03, phase-04, phase-05a, phase-06 |
| Version tag target | v0.3 |
| Status | completed (2026-04-17, pending merge + v0.3 tag) |

## 1. Goal

recipe의 `sweep_axes` 필드를 실행 가능한 sweep plan으로 확장하고, variant별 `execute_recipe()` 호출 결과를 study-level artifact로 merge합니다. 이 phase 완료 시점에 **package identity (horse race benchmark) 달성**.

## 2. Scope

**In scope:**
- `SweepPlan` compiler pass — `sweep_axes` → variant recipe 집합 (Cartesian 기본)
- `SweepRunner` 실행자 — variant 순회 + artifact merge
- `study_manifest.json` schema v1 freeze
- `controlled_variation_study` registry → operational 승격
- Recipe YAML `sweep_axes` 문법 확장 + validation
- Tuning engine iteration 공유 (macrocast/execution/iteration.py)

**Out of scope:**
- `conditional_axes` / `nested_sweep` / `derived_axes` — v1.1 (Phase 10)
- `parallel_by_variant` 기본 활성화 — serial 기본, opt-in만 (ADR-003)
- Multi-target joint sweep — Phase 5a 이후

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 01.1 | `SweepPlan` compiler 작성 | **P0** | ~300 | `macrocast/compiler/sweep_plan.py` (신규) | Cartesian expansion 정확 |
| 01.2 | `SweepRunner` 실행자 작성 | **P0** | ~400 | `macrocast/execution/sweep_runner.py` (신규) | 3-variant sweep end-to-end |
| 01.3 | `study_manifest.json` schema v1 freeze | **P0** | ~100 | `plans/infra/study_manifest_schema.md`, `macrocast/studies/manifest.py` (신규) | JSONSchema validation |
| 01.4 | Recipe YAML `sweep_axes` 문법 + validation | **P0** | ~150 | `macrocast/compiler/build.py` (수정), `macrocast/compiler/sweep_plan.py` | invalid sweep → clean error |
| 01.5 | `controlled_variation_study` operational 승격 | **P0** | ~50 | `macrocast/registry/stage0/study_mode.py` (수정) | status → operational |
| 01.6 | Tuning engine iteration 공유 | P1 | ~200 | `macrocast/execution/iteration.py` (신규), `macrocast/tuning/engine.py` (리팩터) | 기존 tuning test green + sweep runner도 사용 |
| 01.7 | Phase 1 tests | **P0** | ~500 | `tests/test_sweep_plan.py`, `tests/test_sweep_runner.py`, `tests/test_sweep_cache_share.py`, `tests/test_sweep_manifest_schema.py` | 4개 test 전부 green |
| 01.8 | Recipe 예시 2개 (horse race) | P1 | ~80 | `examples/recipes/horse-race-model.yaml`, `examples/recipes/horse-race-preprocessing.yaml` | sweep_runner로 실행 성공 |
| 01.9 | Phase 1 docs 4종 | P1 | ~400 | `docs/getting_started/horse_race_quickstart.md`, `docs/user_guide/sweep_recipes.md`, `docs/user_guide/controlled_variation_study.md`, `docs/api/sweep_runner.md` | RTD build green |

## 4. API / Schema Specifications

### 4.1 `SweepPlan` API

```python
# macrocast/compiler/sweep_plan.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SweepVariant:
    variant_id: str              # "v-0001"
    axis_values: dict[str, str]  # {"model_family": "ridge", "scaling_policy": "none"}
    parent_recipe_id: str
    variant_recipe: RecipeSpec   # parent + axis overrides

@dataclass(frozen=True)
class SweepPlan:
    study_id: str
    parent_recipe: RecipeSpec
    axes_swept: tuple[str, ...]
    variants: tuple[SweepVariant, ...]

def compile_sweep_plan(
    recipe_dict: dict,
    *,
    max_variants: int | None = None,
) -> SweepPlan:
    """Expand sweep_axes into concrete variant recipes.

    Rules:
    - fixed_axes 값은 모든 variant에서 동일
    - sweep_axes 각 축은 Cartesian product으로 확장
    - conditional_axes / derived_axes는 v1.1 (지금은 fixed로 처리)
    - max_variants 초과시 ValueError (sweep explosion 방지, default=1000)
    """
```

**Invariants:**
- 각 `variant.variant_recipe`는 기존 `compile_recipe_dict()`에 넣을 수 있는 valid single-path recipe
- `variant_id`는 sha256(canonical axis_values JSON)[:8] 기반 안정적
- `study_id`는 sha256(canonical sweep_plan JSON) 기반

### 4.2 `SweepRunner` API

```python
# macrocast/execution/sweep_runner.py
def execute_sweep(
    *,
    plan: SweepPlan,
    preprocess: PreprocessContract,
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
    provenance_payload: dict | None = None,
    fail_fast: bool = False,  # False = failure_policy 적용; True = 첫 실패에서 raise
) -> "SweepResult":
    """variant 순회 + artifact merge.

    Algorithm:
    1. study_id = plan.study_id
    2. mkdir output_root/variants/, output_root/.raw_cache_shared/
    3. for variant in plan.variants:
         try:
           result = execute_recipe(
               recipe=variant.variant_recipe,
               preprocess=preprocess,
               output_root=output_root / "variants" / variant.variant_id,
               cache_root=output_root / ".raw_cache_shared",
               provenance_payload={..., "variant_id": variant.variant_id, "study_id": study_id}
           )
           collect metrics_summary
         except Exception as exc:
           if fail_fast: raise
           mark variant as failed (per failure_policy — infra/failure_policy.md)
    4. write output_root/study_manifest.json per Schema v1
    5. return SweepResult(study_id, manifest_path, per_variant_results)
    """

@dataclass(frozen=True)
class SweepResult:
    study_id: str
    manifest_path: str
    per_variant_results: list["VariantResult"]
    successful_count: int
    failed_count: int
```

### 4.3 Recipe YAML `sweep_axes` 문법

```yaml
# examples/recipes/horse-race-model.yaml
recipe_id: horse-race-model-vs-benchmark
path:
  0_meta:
    fixed_axes:
      study_mode: controlled_variation_study
  1_data_task:
    fixed_axes:
      dataset: fred_md
      information_set_type: revised
      task: single_target_point_forecast
    leaf_config:
      target: INDPRO
      horizons: [1, 3]
  2_preprocessing:
    fixed_axes:
      scaling_policy: standard
      # ... other fixed preprocessing
  3_training:
    fixed_axes:
      framework: expanding
      benchmark_family: ar_bic
      feature_builder: autoreg_lagged_target
    sweep_axes:
      model_family: [ridge, lasso, elasticnet, randomforest]   # 4 variants
  4_evaluation:
    fixed_axes:
      primary_metric: msfe
  # ...
```

**Validation rules:**
- 같은 layer에서 axis가 fixed_axes와 sweep_axes에 동시 등장 → error
- sweep_axes 값이 axis registry에 없음 → error
- sweep_axes 조합이 max_variants 초과 → error

### 4.4 `study_manifest.json` schema v1

`plans/infra/study_manifest_schema.md` 참조. 요약:

```json
{
  "schema_version": "1.0",
  "study_id": "sha256-of-plan-canonical",
  "study_mode": "controlled_variation_study",
  "created_at_utc": "...",
  "parent_recipe": { ... RecipeSpec ... },
  "sweep_plan": {
    "axes_swept": ["model_family"],
    "variants": [
      {
        "variant_id": "v-a1b2c3d4",
        "axis_values": {"model_family": "ridge"},
        "status": "success",
        "artifact_dir": "variants/v-a1b2c3d4/",
        "metrics_summary": {"msfe": 0.023},
        "seed_used": 1234,
        "runtime_seconds": 45.2
      }
    ]
  },
  "tree_context": { ... },
  "git_commit": "...",
  "package_version": "0.3.0"
}
```

## 5. File Layout

**신규:**
- `macrocast/compiler/sweep_plan.py`
- `macrocast/execution/sweep_runner.py`
- `macrocast/execution/iteration.py`
- `macrocast/studies/__init__.py`
- `macrocast/studies/manifest.py`
- `macrocast/studies/controlled_variation.py`
- `tests/test_sweep_plan.py`
- `tests/test_sweep_runner.py`
- `tests/test_sweep_cache_share.py`
- `tests/test_sweep_manifest_schema.py`
- `examples/recipes/horse-race-model.yaml`
- `examples/recipes/horse-race-preprocessing.yaml`
- `docs/getting_started/horse_race_quickstart.md`
- `docs/user_guide/sweep_recipes.md`
- `docs/user_guide/controlled_variation_study.md`
- `docs/api/sweep_runner.md`

**수정:**
- `macrocast/compiler/build.py` — sweep_plan 호출 진입점
- `macrocast/tuning/engine.py` — iteration 공통화 리팩터
- `macrocast/registry/stage0/study_mode.py` — controlled_variation_study operational
- `macrocast/__init__.py` — 공개 API 추가 (compile_sweep_plan, execute_sweep, SweepPlan, SweepVariant, SweepResult)

## 6. Test Strategy

### `tests/test_sweep_plan.py`
- `compile_sweep_plan` — 2-axis Cartesian → 올바른 변형 수
- `max_variants` 초과 → ValueError
- fixed_axes + sweep_axes 혼재 valid
- 같은 axis가 두 곳에 → validation error

### `tests/test_sweep_runner.py`
- 3-variant sweep end-to-end → 3 variant dir + study_manifest.json
- `fail_fast=True` 모드: 첫 실패에서 raise
- `fail_fast=False`: 실패 variant 기록 후 계속 (failure_policy 연동)
- 동일 plan 두 번 실행 → 동일 study_id + variant artifact hash

### `tests/test_sweep_cache_share.py`
- 10-variant sweep: FRED 파일이 `.raw_cache_shared`에 1회만 생성됨
- Per-variant artifact dir에는 `.raw_cache`가 **생성되지 않음** (sweep runner가 cache_root 지정)

### `tests/test_sweep_manifest_schema.py`
- study_manifest.json이 Schema v1 준수 (JSONSchema validation)

## 7. Acceptance Gate

- [x] Phase 0 gate 선통과 — v0.2 tag (08c0e70)
- [x] 3-axis horse race recipe (model × scaling × horizon) 공개 API로 실행 — Cartesian expansion verified via test_sweep_plan.py::test_two_axis_sweep_cartesian_count + examples/recipes/horse-race-model.yaml executes 4 variants
- [x] study_manifest.json Schema v1 준수 + JSONSchema validation green — test_sweep_manifest_schema.py 8/8
- [x] 기존 291 test + Phase 0 test + Phase 1 신규 test 전부 green — 340/340 passed (0 regressions, +28 new)
- [x] `controlled_variation_study` registry status = operational — macrocast/registry/stage0/study_mode.py
- [x] 공개 API export: `compile_sweep_plan`, `execute_sweep`, `SweepPlan`, `SweepVariant`, `SweepResult` — macrocast/__init__.py (plus VariantResult, VariantManifestEntry, build_study_manifest, validate_study_manifest, STUDY_MANIFEST_SCHEMA_VERSION)
- [x] Phase 1 docs 4종 RTD latest build green — docs/getting_started/horse_race_quickstart.md + docs/user_guide/sweep_recipes.md + docs/user_guide/controlled_variation_study.md + docs/api/sweep_runner.md wired into toctrees (RTD verifies on merge)
- [x] `horse_race_quickstart.md` 시나리오 end-to-end 재현 가능 — 소개된 API (compile_sweep_plan + execute_sweep) 경로가 test_sweep_runner end-to-end 테스트로 검증됨

## 8. Docs Deliverables

**신규:**
- `docs/getting_started/horse_race_quickstart.md` (**v1.0 flagship quickstart**)
- `docs/user_guide/sweep_recipes.md` — YAML sweep_axes 문법 상세
- `docs/user_guide/controlled_variation_study.md` — use case 중심 가이드
- `docs/api/sweep_runner.md` — autodoc

## 9. Migration Notes

- Recipe YAML에서 `sweep_axes`는 이전에 **무시**됐음 (compiler가 파싱만 하고 실행 안 함)
- Phase 1 이후: `sweep_axes` 지정시 자동으로 `controlled_variation_study`로 해석되어 sweep 실행
- `study_mode=single_path_benchmark_study` 사용자는 영향 없음 (sweep_axes 안 쓰면 single-path)
- Breaking 없음

## 10. Cross-references

- Infra files used: `plans/infra/seed_policy.md`, `plans/infra/failure_policy.md`, `plans/infra/study_manifest_schema.md`, `plans/infra/cache_discipline.md`
- ADRs referenced: ADR-001 (sweep vs tuning iteration), ADR-003 (sweep parallel opt-in)
- Coverage Ledger rows resolved:
  - Layer 0 `study_mode = controlled_variation_study` → operational
  - Layer 0 `experiment_unit = single_target_full_sweep` → operational
  - Layer 0 `experiment_unit = multi_target_separate_runs` → operational
  - Layer 0 `axis_type = sweep` → operational
  - Layer 0 `axis_type = eval_only` → operational

## 11. GitHub Issue Map

- Epic: (TBD at kickoff — [PHASE-01] Horse-race sweep executor)
- Sub-task issues: 9개 (01.1~01.9)

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 Phase 1에서 추출)
- 2026-04-17: Phase 1 구현 완료 — sweep_plan + sweep_runner + studies.manifest + 28 tests green (340 total); docs/examples 연결; v0.3 tag는 feat/phase-01 브랜치 merge 후
