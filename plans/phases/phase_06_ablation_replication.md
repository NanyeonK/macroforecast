# Phase 06 — Ablation & Replication Runners

| Field | Value |
|-------|-------|
| Phase ID | phase-06 |
| Priority (inter-phase) | **P1** |
| Depends on | phase-01 |
| Unlocks | phase-07 |
| Version tag target | v0.8 |
| Status | in_progress |

## 1. Goal

Phase 1 sweep executor 위에 2개 대표 use case 패턴을 공식화합니다 — ablation (baseline + N drop-one variants) 과 replication (frozen recipe + overrides + synthetic round-trip diff). 이 phase 완료 시점에 `experiment_unit = {ablation_study, replication_recipe}` registry가 operational status로 승격되고, `macrocast/studies/` 모듈이 두 runner를 export 합니다.

## 2. Scope

**In scope:**
- `AblationRunner` — baseline + N drop-one variants 자동 생성 + sweep 실행
- `ReplicationRunner` — 과거 study artifact 로딩 + overrides 적용 + diff report
- `override_diff.py` compiler tool — base recipe + overrides → (new recipe, diff entries)
- `experiment_unit` axis values ablation_study, replication_recipe operational 승격
- Synthetic replication round-trip test (execute_recipe → replay → byte-identical 검증)
- Docs: ablation_cookbook, replication_cookbook, synthetic_replication_roundtrip example
- Example recipes: ablation-preprocessing.yaml, replication-synthetic.yaml

**Out of scope:**
- 구체 논문 replication (Phase 7 이후, post-v1.0)
- Cross-repo replication (다른 repo clone 간 결정성) → v2
- Multi-baseline ablation (하나의 baseline 기준) → v1.1
- Interactive ablation visualization → 별도 dashboard

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 06.1 | `AblationRunner` 구현 | P1 | ~350 | `macrocast/studies/ablation.py` (신규) | `test_ablation_runner.py` green |
| 06.2 | `ReplicationRunner` 구현 + diff report | P1 | ~450 | `macrocast/studies/replication.py` (신규) | `test_replication_runner.py` green |
| 06.3 | `override_diff.py` compiler tool | **P0** | ~200 | `macrocast/compiler/override_diff.py` (신규) | `test_override_diff.py` green |
| 06.4 | `experiment_unit` operational 승격 | P1 | ~50 | `macrocast/registry/stage0/experiment_unit.py` (수정) | status → operational (ablation_study, replication_recipe) |
| 06.5 | Phase 6 tests | **P0** | ~700 | `tests/test_ablation_runner.py`, `tests/test_replication_runner.py`, `tests/test_replication_end_to_end.py`, `tests/test_override_diff.py` | 4개 test 전부 green |
| 06.6 | Phase 6 docs | P1 | ~400 | `docs/user_guide/ablation_cookbook.md`, `docs/user_guide/replication_cookbook.md`, `docs/examples/synthetic_replication_roundtrip.md` | RTD build green |
| 06.7 | Example recipes | P1 | ~100 | `examples/recipes/ablation-preprocessing.yaml`, `examples/recipes/replication-synthetic.yaml` | CLI로 실행 성공 |

## 4. API / Schema Specifications

### 4.1 `AblationRunner` API

```python
# macrocast/studies/ablation.py
from dataclasses import dataclass
from macrocast.execution.types import RecipeSpec
from macrocast.execution.sweep_runner import SweepResult, execute_sweep
from macrocast.compiler.sweep_plan import SweepPlan, SweepVariant

@dataclass(frozen=True)
class AblationSpec:
    baseline_recipe: RecipeSpec
    components_to_ablate: list[tuple[str, str]]  # [(axis_name, neutral_value), ...]
    ablation_study_id: str | None = None         # 자동 생성시 hash(baseline+components)

def execute_ablation(
    *,
    spec: AblationSpec,
    preprocess: "PreprocessContract",
    output_root: str,
) -> SweepResult:
    """baseline + N variants (each one component neutralized) → sweep_runner.

    Algorithm:
    1. baseline variant (axis_values = all original)
    2. for each (axis_name, neutral_value) in components_to_ablate:
         variant: baseline + {axis_name: neutral_value}
    3. compile into SweepPlan
    4. execute_sweep(plan, ...) — Phase 1 runner 재활용
    5. emit ablation_report.json in output_root (delta vs baseline per component)
    """
```

**Semantics:**
- "Neutralize" = axis를 trivial / no-op 값으로 치환 (e.g., scaling_policy='none', feature_engineering='identity', benchmark_model='historical_mean')
- neutral_value는 사용자 지정 (axis 마다 "off" 개념이 다름)
- Output: standard SweepResult + 추가 `ablation_report.json` with per-component delta

### 4.2 `ablation_report.json` schema

```json
{
  "schema_version": "1.0",
  "ablation_study_id": "abl-sha256...",
  "baseline_variant_id": "v-baseline",
  "baseline_metrics": {"msfe": 0.023, "rmse": 0.152},
  "components": [
    {
      "axis_name": "scaling_policy",
      "original_value": "standard",
      "neutral_value": "none",
      "variant_id": "v-ablate-scaling",
      "metrics": {"msfe": 0.031, "rmse": 0.176},
      "delta_vs_baseline": {"msfe": 0.008, "msfe_pct": 34.8}
    }
  ],
  "package_version": "0.8.0",
  "created_at_utc": "..."
}
```

### 4.3 `ReplicationRunner` API

```python
# macrocast/studies/replication.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ReplicationResult:
    source_study_id: str
    replayed_study_id: str
    diff_report_path: str            # path to replication_diff.json
    byte_identical_predictions: bool
    sweep_result: "SweepResult"      # actual re-execution

def execute_replication(
    *,
    source_artifact_dir: str,
    overrides: dict,                  # e.g. {"2_preprocessing.scaling_policy": "robust"}
    preprocess: "PreprocessContract",
    output_root: str,
) -> ReplicationResult:
    """Load frozen recipe from source, apply overrides, execute, emit diff report.

    Steps:
    1. Load recipe from source_artifact_dir/manifest.json (or study_manifest.json)
    2. apply_overrides(base_recipe, overrides) -> (new_recipe, diff_entries)
    3. execute_recipe(new_recipe, ...) -> new_result
    4. Compare predictions.csv, metrics.json against source
    5. Write replication_diff.json with metrics delta + cause narrative
    """
```

### 4.4 `replication_diff.json` schema

```json
{
  "schema_version": "1.0",
  "source_study_id": "orig-study-xyz",
  "source_package_version": "0.1.0",
  "replayed_package_version": "0.2.0",
  "source_artifact_dir": "/path/to/source/",
  "replayed_artifact_dir": "/path/to/replay/",
  "overrides_applied": {"2_preprocessing.scaling_policy": "robust"},
  "metrics_delta": {
    "msfe": {"source": 0.021, "replayed": 0.020, "delta_abs": -0.001, "delta_pct": -4.8}
  },
  "byte_identical_predictions": false,
  "cause_of_difference": "scaling_policy override",
  "created_at_utc": "..."
}
```

`byte_identical_predictions=true` 조건: overrides={} 이고 source/replayed package_version 동일하고 seed policy가 strict_reproducible.

### 4.5 `override_diff.py` API (revised 2026-04-17)

Operates on the **YAML recipe-dict** form, not `RecipeSpec` — overrides address layer/field paths that map to the dict structure (`2_preprocessing.scaling_policy`), not to RecipeSpec dataclass fields. Callers who need a compiled spec run `compile_recipe_dict(new_dict)` afterwards.

```python
# macrocast/compiler/override_diff.py

def apply_overrides(
    base_recipe_dict: dict,
    overrides: dict,
) -> tuple[dict, list[dict]]:
    """Apply dotted-path overrides, returning (new_dict, diff_entries).

    overrides 예: {"2_preprocessing.scaling_policy": "robust",
                   "3_training.model_family": "lasso",
                   "3_training.hyperparams.alpha": 0.1}

    Returns:
      new_dict: `copy.deepcopy(base_recipe_dict)` + 경로별 치환
      diff_entries: [
        {"path": "2_preprocessing.scaling_policy",
         "old": "standard", "new": "robust"},
        ...
      ]

    Nested paths (`a.b.c.d`) are supported via recursive descent.
    base_recipe_dict is NOT mutated.

    Raises:
      KeyError — dotted path does not resolve to an existing key
      ValueError — attempt to override into a leaf (non-dict) intermediate
    """
```

### 4.6 `experiment_unit` 승격 (revised 2026-04-17)

Current registry is a tuple of `ExperimentUnitEntry` (an `EnumRegistryEntry` subclass with `route_owner`, `requires_multi_target`, `requires_wrapper` fields). Phase 6 adds an optional `runner: str | None = None` field and flips two entries to `operational`:

```python
# macrocast/registry/stage0/experiment_unit.py (edit)

@dataclass(frozen=True)
class ExperimentUnitEntry(EnumRegistryEntry):
    route_owner: RouteOwner
    requires_multi_target: bool
    requires_wrapper: bool
    runner: str | None = None   # NEW in Phase 6

# Two existing rows flipped:
ExperimentUnitEntry(
    id="ablation_study",
    ...
    status="operational",            # was "planned"
    runner="macrocast.studies.ablation:execute_ablation",
    ...
),
ExperimentUnitEntry(
    id="replication_recipe",
    ...
    status="operational",            # was "registry_only"
    runner="macrocast.studies.replication:execute_replication",
    ...
),
```

Studies/__init__.py exports the two callables so the `runner` string lookup can be resolved via `importlib` if and when a future phase needs dynamic dispatch.

### 4.7 Autonomous-execution decisions (pinned 2026-04-17)

- `ablation_report.json` is written to `<output_root>/ablation_report.json` alongside the sweep"s `study_manifest.json`.
- `replication_diff.json` is written to `<output_root>/replication_diff.json`.
- `ablation_study_id` = `"abl-" + sha256(json.dumps({"baseline": <recipe_dict>, "components": [<sorted components>]}, sort_keys=True))[:12]` when the caller doesn"t supply one.
- Byte-identical round-trip test forces `reproducibility_mode=strict_reproducible` on the source recipe so seeded-reproducible defaults don"t mask non-determinism.
- `execute_ablation` and `execute_replication` take the same `preprocess: PreprocessContract` parameter as `execute_recipe` — consistency with the Phase 1 runner surface.
- Nested override paths (`a.b.c`) descend through dicts; any intermediate resolving to a non-dict raises `ValueError`.
- Metric delta percentage formula: `delta_pct = 100 * (replayed - source) / source` when `abs(source) > 1e-12`, else `None`.

## 5. File Layout

**신규:**
- `macrocast/studies/ablation.py`
- `macrocast/studies/replication.py`
- `macrocast/compiler/override_diff.py`
- `tests/test_ablation_runner.py`
- `tests/test_replication_runner.py`
- `tests/test_replication_end_to_end.py`
- `tests/test_override_diff.py`
- `docs/user_guide/ablation_cookbook.md`
- `docs/user_guide/replication_cookbook.md`
- `docs/examples/synthetic_replication_roundtrip.md`
- `examples/recipes/ablation-preprocessing.yaml`
- `examples/recipes/replication-synthetic.yaml`

**수정:**
- `macrocast/registry/stage0/experiment_unit.py` — 2개 operational 승격
- `macrocast/studies/__init__.py` — `execute_ablation`, `execute_replication` export
- `macrocast/__init__.py` — 공개 API (AblationSpec, execute_ablation, ReplicationResult, execute_replication, apply_overrides)

## 6. Test Strategy

### `tests/test_ablation_runner.py`
- 3-component ablation (scaling_policy, feature_builder, benchmark_model 각각 neutralize)
- baseline + 3 variants = 4 variants total in sweep plan
- `ablation_report.json`에 per-component delta 기록
- baseline metrics vs each ablation의 delta 부호 합리적 (ablate 시 성능 저하)
- baseline variant_id는 "v-baseline" 고정

### `tests/test_replication_runner.py`
- Stub source artifact dir 생성 (manifest.json + predictions.csv)
- overrides={"3_training.model_family": "lasso"} 으로 replicate
- `replication_diff.json`에 overrides_applied 기록
- `byte_identical_predictions=False` (overrides 있으므로)
- metrics_delta 정상 계산

### `tests/test_replication_end_to_end.py` (synthetic round-trip)
```python
def test_synthetic_replication_roundtrip(tmp_path):
    # 1. execute_recipe → source artifact
    src = execute_recipe(recipe=R, output_root=tmp_path/"src")
    # 2. execute_replication with overrides={}
    rep = execute_replication(
        source_artifact_dir=src.artifact_dir,
        overrides={},
        output_root=tmp_path/"replay",
    )
    # 3. byte-identical 검증
    assert rep.byte_identical_predictions is True
    assert sha256(src/"predictions.csv") == sha256(rep.sweep_result/"predictions.csv")
```
- Phase 0 결정성 정책에 의존 (strict_reproducible + cache_root 공유)
- Package version 동일 조건 (tmp venv)

### `tests/test_override_diff.py`
- 단일 경로 override → 1개 diff entry
- 다중 경로 override → 다중 diff entries
- 존재하지 않는 경로 → ValueError
- nested path (e.g. "3_training.hyperparams.alpha": 0.1) 지원
- base_recipe는 원본 유지 (immutability 확인)

## 7. Acceptance Gate

- [ ] Phase 1 gate 선통과
- [ ] `execute_ablation` / `execute_replication` 공개 API export
- [ ] `experiment_unit ∈ {ablation_study, replication_recipe}` registry status = operational
- [ ] Synthetic round-trip test (byte-identical predictions) green
- [ ] `ablation_report.json` / `replication_diff.json` schema v1.0 준수
- [ ] 기존 + Phase 0/1 + Phase 6 신규 test 전부 green
- [ ] Phase 6 docs 3종 + example recipes 2종 RTD build green
- [ ] CLI 로 `examples/recipes/ablation-preprocessing.yaml`, `replication-synthetic.yaml` 실행 성공

## 8. Docs Deliverables

**신규:**
- `docs/user_guide/ablation_cookbook.md` — 대표 ablation 시나리오 (preprocessing 구성요소 제거, feature family 제거, benchmark 제거)
- `docs/user_guide/replication_cookbook.md` — 과거 study 재실행 + override 패턴 + diff report 해석
- `docs/examples/synthetic_replication_roundtrip.md` — 자체 replay 데모 (byte-identical 보장 시나리오)

**수정:**
- `docs/user_guide/controlled_variation_study.md` — ablation/replication 이 이 study_mode 의 특수 패턴임을 명시

## 9. Migration Notes

- Phase 6 이전에 `experiment_unit = ablation_study / replication_recipe` 는 **stub** 이었으므로 실제 사용자가 없음 → Breaking 없음
- `macrocast/studies/` 하위 모듈 확장 (Phase 1의 `manifest.py` 기존, Phase 6에서 `ablation.py`, `replication.py` 추가)
- 공개 API 추가만 (기존 symbol 유지)

## 10. Cross-references

- Infra files used:
  - `plans/infra/study_manifest_schema.md` (Phase 1 에서 freeze)
  - dotted-path override 규약은 `macrocast/compiler/override_diff.py` 모듈 docstring을 SoT로 둠 (별도 infra 문서 없음; Phase 7 이후 필요 시 승격)
- Phase dependencies:
  - Phase 1 sweep runner — AblationRunner / ReplicationRunner 가 내부적으로 `execute_sweep` / `execute_recipe` 호출
  - Phase 2 stat tests — ablation 결과 해석시 (component 삭제가 유의미한가?) stat test 활용
  - Phase 0 결정성 정책 — synthetic round-trip byte-identical의 전제
- ADRs referenced:
  - Phase 6 doesn"t ship new ADRs. The override syntax and the "ablation = drop-one sweep" pattern are documented in-module (`compiler/override_diff.py`, `studies/ablation.py`). If either becomes load-bearing for v1.1+, promote to ADR-008/009 at that time.
- Coverage Ledger rows resolved:
  - Layer 0 `experiment_unit = ablation_study` → operational
  - Layer 0 `experiment_unit = replication_recipe` → operational

## 11. GitHub Issue Map

- Epic: (TBD at kickoff — [PHASE-06] Ablation & replication runners)
- Sub-task issues: 7개 (06.1 ~ 06.7)

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 §Phase 6에서 추출)
- 2026-04-17 (kickoff revision): §4.5 `apply_overrides` signature changed RecipeSpec→dict (dotted paths target YAML recipe-dict layer/field structure). §4.6 registry schema reconciled with actual `ExperimentUnitEntry` dataclass (add optional `runner` field, flip two statuses). §9 factual correction (no prior `controlled_variation.py`). §10 ADR-008/009 + `override_syntax.md` references removed — in-module docstrings carry the contract until v1.1+. §4.7 added with autonomous-execution decisions pinned (report-file locations, id hashing, byte-identical mode, nested paths, delta_pct formula).
