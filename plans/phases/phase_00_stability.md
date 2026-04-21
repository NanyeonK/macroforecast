# Phase 00 — Single-Path Stability & Sweep-Safety

| Field | Value |
|-------|-------|
| Phase ID | phase-00 |
| Priority (inter-phase) | **P0** |
| Depends on | (none — first phase) |
| Unlocks | phase-01 (horse race executor) |
| Version tag target | v0.2 |
| Status | completed (2026-04-17) |

## 1. Goal

`execute_recipe()`를 N회 반복 호출해도 오염 없이 결정적 실행이 보장되도록 single-path runtime을 안정화합니다. Phase 1 sweep runner가 이 runtime을 N회 호출할 때 variant 간 오염이 없어야 horse race 통계가 유효합니다.

## 2. Scope

**In scope:**
- `random_state=42` 하드코딩 20+ 사이트 교체 → `resolve_seed()` API
- `manifest.json` / `tuning_result.json` 이중 쓰기 제거
- `execute_recipe()`에 `cache_root` 파라미터 추가 (sweep variant 간 FRED 캐시 공유)
- `deep_training.py` 동일 seed-hardcoding audit
- 결정성 회귀 테스트 작성
- `docs/dev/reproducibility_policy.md` 신규 작성

**Out of scope:**
- Seed policy의 runtime 활용 범위 확장 → Phase 1 (sweep variant별 seed 부여)
- Deep model seed 통일 → Phase 5a (torch.manual_seed + CUDA determinism)
- Parallelism nesting 정책 → Phase 5 이후

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 00.1 | `seed_policy.py` 작성 + 20+ 사이트 교체 | **P0** | ~200 | `macrocast/execution/seed_policy.py` (신규), `macrocast/execution/build.py` (20+ 사이트) | `test_seed_policy.py` green |
| 00.2 | `execute_recipe()` 이중 쓰기 제거 | P1 | ~-20 | `macrocast/execution/build.py` lines 2199-2212 | 기존 test 회귀 없음 |
| 00.3 | `cache_root` 파라미터 추가 | **P0** | ~30 | `macrocast/execution/build.py`, `macrocast/execution/types.py` | `test_execution_cache.py` green |
| 00.4 | `deep_training.py` seed audit | P1 | ~50 | `macrocast/execution/deep_training.py` | grep으로 `random_state=42`/`random_seed=42` zero |
| 00.5 | 결정성 회귀 테스트 작성 | **P0** | ~150 | `tests/test_deterministic_replay.py` (신규), `tests/test_execution_cache.py` (신규), `tests/test_seed_policy.py` (신규) | 3개 test 모두 green |
| 00.6 | Phase 0 docs | P1 | ~100 | `docs/dev/reproducibility_policy.md` (신규) | RTD build green |

## 4. API / Schema Specifications

### 4.1 `resolve_seed()` API

```python
# macrocast/execution/seed_policy.py
from __future__ import annotations
import hashlib
import numpy as np

def resolve_seed(
    *,
    recipe_id: str,
    variant_id: str | None = None,
    reproducibility_spec: dict,
    model_family: str | None = None,
) -> int:
    """Deterministic per-variant seed resolution.

    Modes (reproducibility_spec['reproducibility_mode']):
    - strict_reproducible: hash(recipe_id|variant_id|model_family) & 0x7FFFFFFF
    - seeded_reproducible: reproducibility_spec.get('seed', 42)
    - best_effort: reproducibility_spec.get('seed', 42)
    - exploratory: np.random.randint(0, 2**31 - 1)
    """
    mode = reproducibility_spec.get("reproducibility_mode", "seeded_reproducible")
    base_seed = int(reproducibility_spec.get("seed", 42))

    if mode == "strict_reproducible":
        key = f"{recipe_id}|{variant_id or 'main'}|{model_family or ''}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
    if mode == "seeded_reproducible":
        return base_seed
    if mode == "best_effort":
        return base_seed
    if mode == "exploratory":
        return int(np.random.randint(0, 2**31 - 1))
    raise ValueError(f"unknown reproducibility_mode: {mode}")
```

### 4.2 `cache_root` 파라미터 확장

```python
# macrocast/execution/build.py
def execute_recipe(
    *,
    recipe: RecipeSpec,
    preprocess: PreprocessContract,
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
    provenance_payload: dict | None = None,
    cache_root: str | Path | None = None,   # 신규
) -> ExecutionResult:
    ...
    # 기존:
    #   raw_result = _load_raw_for_recipe(recipe, local_raw_source, output_root / ".raw_cache")
    # 변경:
    effective_cache = Path(cache_root) if cache_root is not None else (Path(output_root) / ".raw_cache")
    raw_result = _load_raw_for_recipe(recipe, local_raw_source, effective_cache)
```

Single-path 호출자는 기존 동작 유지 (cache_root=None → output_root/.raw_cache). Sweep runner는 study-level 공유 캐시 지정.

### 4.3 Seed 교체 대상 사이트

`phase0_audit_2026_04_17.md` 1 참조. `macrocast/execution/build.py`의 다음 라인:
- 557, 577, 587, 592, 597, 602, 607, 612 — autoreg model variants (RF/SVR/ET/GBM/XGB/LGB/CB/MLP)
- 648, 672, 684, 690, 696, 702, 708, 714 — raw-panel model variants
- 1429, 1576, 1592, 1640, 1722, 1727 — importance / permutation_importance / default_rng

각 사이트 패턴:
```python
# Before
RandomForestRegressor(n_estimators=200, random_state=42)

# After
RandomForestRegressor(
    n_estimators=200,
    random_state=resolve_seed(
        recipe_id=recipe.recipe_id,
        variant_id=provenance_payload.get("variant_id") if provenance_payload else None,
        reproducibility_spec=_reproducibility_spec(provenance_payload),
        model_family="randomforest",
    ),
)
```

## 5. File Layout

**신규:**
- `macrocast/execution/seed_policy.py`
- `tests/test_seed_policy.py`
- `tests/test_deterministic_replay.py`
- `tests/test_execution_cache.py`
- `docs/dev/reproducibility_policy.md`

**수정:**
- `macrocast/execution/build.py` (~30 edits)
- `macrocast/execution/deep_training.py` (seed audit/fix)
- `macrocast/execution/types.py` (cache_root hint 추가 if needed)

## 6. Test Strategy

### `tests/test_seed_policy.py`
- 4개 mode (strict/seeded/best_effort/exploratory) 각각 반환 값 검증
- strict_reproducible: 같은 (recipe_id, variant_id)은 항상 동일 seed
- 다른 variant_id은 다른 seed
- model_family가 포함되어 모델별 seed가 독립

### `tests/test_deterministic_replay.py`
```python
def test_identical_recipe_yields_identical_artifacts(tmp_path):
    recipe_dict = load_example_recipe("model-benchmark.yaml")
    r1 = execute_recipe(..., output_root=tmp_path/"r1")
    r2 = execute_recipe(..., output_root=tmp_path/"r2")
    assert sha256(r1.artifact_dir/"predictions.csv") == sha256(r2.artifact_dir/"predictions.csv")
    assert sha256(r1.artifact_dir/"metrics.json") == sha256(r2.artifact_dir/"metrics.json")

def test_distinct_variant_id_yields_distinct_artifacts(tmp_path):
    # reproducibility_mode=strict_reproducible, variant_id='A' vs 'B'
    ...
```

### `tests/test_execution_cache.py`
- 동일 `cache_root`로 두 번 호출 → FRED download 1회만 (file mtime 불변)
- 다른 `cache_root` → 각자 다운로드

## 7. Acceptance Gate

- [x] 기존 291 test green — 2026-04-17 full suite 312 passed (0 regressions)
- [x] Phase 0 신규 test 3개 (seed_policy / deterministic_replay / execution_cache) green — 2026-04-17 20 passed in 4.43s
- [x] `grep -E 'random_state=42|random_seed=42' macrocast/execution/` → 0 hits (seed_policy로 전환) — verified
- [x] `execute_recipe()`의 `manifest.json` write가 함수당 정확히 1회 — verified build.py:2364
- [x] `execute_recipe()` 시그니처에 `cache_root` 존재 — verified build.py:2075 (`cache_root: str | Path | None = None`)
- [x] `docs/dev/reproducibility_policy.md` 존재 + RTD build green — file present, wired in docs/dev/index.md toctree

## 8. Docs Deliverables

- **신규:** `docs/dev/reproducibility_policy.md`
  - seed policy API + 4개 mode 설명
  - 사용 예시 (single-path vs sweep variant)
  - 결정성 보장 범위 (single-path 내 / variant 간 / across repo clones)

## 9. Migration Notes

- Breaking change 없음
- `execute_recipe()` 새 파라미터 `cache_root`는 optional (default=None → 기존 동작)
- Registry / recipe YAML schema 불변

## 10. Cross-references

- Infra files used: `plans/infra/seed_policy.md`
- ADRs referenced: (none 직접; ADR-004 deep-learning-optional은 Phase 5a)
- Coverage Ledger rows resolved:
  - Layer 0 `reproducibility_mode` 4개 값 → runtime wiring 완료

## 11. GitHub Issue Map

- Epic: (TBD, Phase kickoff 시 생성 — [PHASE-00] Single-path stability & sweep-safety)
- Sub-task issues: (TBD, kickoff 시 6개 생성)

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 Phase 0에서 추출)
- 2026-04-17: Phase 0 완료 — PR #8 (commit 08c0e70) 머지, Acceptance Gate 전부 통과, v0.2 tag 준비

## 13. References

- `plans/phase0_audit_2026_04_17.md` — 현재 상태 audit 보고서
- `plans/infra/seed_policy.md` — 공통 seed 인프라 상세
