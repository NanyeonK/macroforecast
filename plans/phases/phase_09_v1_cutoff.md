# Phase 09 — Docs Rewrite + v1.0 Cutoff

| Field | Value |
|-------|-------|
| Phase ID | phase-09 |
| Priority (inter-phase) | **P0** |
| Depends on | phase-08 |
| Unlocks | (v1.0 release) |
| Version tag target | v1.0.0 |
| Status | pending |

## 1. Goal

Phase 1-8까지 per-phase docs는 이미 작성되어 있습니다. Phase 9는 **랜딩 페이지 + 통합 + 검증**입니다. Horse race identity를 landing의 핵심 메시지로 끌어올리고, v0 → v1 migration guide를 명문화하며, identity verification hard gate를 통과시켜 **v1.0 release candidate**를 만듭니다. 이 Phase는 "코드를 더 쓰는" Phase가 아니라 "이미 쓴 것을 cite-ready로 consolidate하는" Phase입니다.

## 2. Scope

**In scope:**
- Landing page 재작성 (`docs/index.md`, `docs/user_guide/index.md`)
- Architecture doc 재작성 (11-phase outcome 반영)
- Migration guide v0 → v1 (breaking changes 명문화)
- README 재작성 (과장 claim 제거, v1 feature list / quickstart / citation)
- CHANGELOG (v0.1.0 → v1.0.0 전체 요약)
- Version bump (pyproject.toml, `__version__`)
- Identity verification hard gate
- GitHub Release + PyPI publish (Trusted Publishing)
- RTD stable branch tag

**Out of scope:**
- conda-forge release — Resolved Decision #7에 따라 PyPI only
- Interactive dashboard — Phase 11
- 신규 기능 — Phase 9는 consolidation only

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 09.1 | Landing & architecture docs 재작성 | **P0** | ~500 | `docs/index.md`, `docs/user_guide/index.md`, `docs/dev/architecture.md` | RTD latest build green |
| 09.2 | Migration guide v0 → v1 | **P0** | ~300 | `docs/migration/v0_to_v1.md` (신규) | breaking change 3종 명문화 |
| 09.3 | README 재작성 | **P0** | ~200 | `README.md` | 과장 claim 0, citation 포함 |
| 09.4 | CHANGELOG | **P0** | ~400 | `CHANGELOG.md` (신규) | 11 phase별 PR 링크 포함 |
| 09.5 | Version bump | **P0** | ~10 | `pyproject.toml`, `macrocast/__init__.py` | `__version__ == "1.0.0"` |
| 09.6 | Identity verification hard gate | **P0** | ~0 (실행) | (CI run) | 3 verification 모두 green |
| 09.7 | GitHub Release + PyPI publish | **P0** | ~50 | `.github/workflows/release.yml` (검증), git tag | PyPI `macrocast==1.0.0` install 가능 |
| 09.8 | Release announcement | P1 | ~100 | GitHub Release page body | RTD stable + PyPI install sanity |

## 4. API / Schema Specifications

Phase 9는 코드 API 변경 없음. 단, **version bump**와 **공개 API surface freeze**를 문서화합니다.

### 4.1 Version bump targets

```toml
# pyproject.toml
[project]
name = "macrocast"
version = "1.0.0"   # was "0.9.x"
```

```python
# macrocast/__init__.py
__version__ = "1.0.0"
```

### 4.2 Public API surface (v1.0 freeze)

Phase 0-8에서 export된 공개 API를 `__all__`로 명시 고정:

```python
# macrocast/__init__.py
__all__ = [
    # Phase 0
    "resolve_seed",
    # Phase 1
    "compile_sweep_plan", "execute_sweep", "SweepPlan", "SweepVariant", "SweepResult",
    # Phase 2-5 (statistical tests / regime / deep / evaluation)
    # ... (phase별 export)
    # Phase 7
    "DecompositionPlan", "run_decomposition", "DecompositionResult",
    # Phase 8
    "emit_paper_ready_bundle", "BundleSpec", "PaperReadyBundle",
    "rank_variants", "aggregate_regime_metrics",
]
```

### 4.3 Migration guide breaking changes

`docs/migration/v0_to_v1.md`에 명문화할 3개 breaking change:

1. **`stat_test` 1축 → 8축** (Phase 2): 기존 `stat_test: dm_test` 단일 값 → layer별 8 axis 분리
2. **Recipe YAML `sweep_axes` 의미 변경** (Phase 1): 이전에는 parsing만 되고 무시되었음. 이제는 자동으로 `controlled_variation`로 해석되어 sweep 실행
3. **`execute_recipe()` 시그니처 확장** (Phase 0): `cache_root` optional 파라미터 추가 (backward-compat이지만 시그니처 변화)

## 5. File Layout

**신규:**
- `CHANGELOG.md`
- `docs/migration/v0_to_v1.md`

**수정:**
- `README.md`
- `docs/index.md`
- `docs/user_guide/index.md`
- `docs/dev/architecture.md`
- `pyproject.toml`
- `macrocast/__init__.py` — `__version__`, `__all__` 최종화

## 6. Test Strategy

Phase 9는 신규 unit test를 추가하지 않지만, **3가지 identity verification**을 hard gate로 실행합니다:

### Identity verification 1: Synthetic replication round-trip
- `examples/recipes/synthetic-replication.yaml` 실행
- `execute_recipe` → `execute_sweep` → `run_decomposition` → `emit_paper_ready_bundle`
- 두 번 실행 → bundle_hash 일치

### Identity verification 2: LSTM vs baselines horse race
- `[deep]` extras 설치 후 `examples/recipes/horse-race-model.yaml` + deep variant
- 4-variant 대체 (deep 미설치 환경): ridge / lasso / rf / svr
- 두 경로 모두 green

### Identity verification 3: Preprocessing attribution demo
- `examples/recipes/horse-race-preprocessing.yaml` 실행 → decomposition에서 `preprocessing` share 유의 수준
- `paper_ready_bundle` tables / figures 생성 완료

### CI pipeline
- `.github/workflows/release.yml`의 pre-release job에서 위 3 verification 실행
- 전부 green이어야 `git tag v1.0.0` push가 PyPI publish trigger

## 7. Acceptance Gate

- [ ] Phase 8 gate 선통과
- [ ] `docs/index.md` landing에서 horse race identity가 첫 화면 메시지
- [ ] `docs/user_guide/index.md` reading order = quickstart(P1) → decomposition(P7) → bundle(P8)
- [ ] `docs/migration/v0_to_v1.md`에 3 breaking change 명문화
- [ ] `README.md`에 과장 claim 0 + citation info 포함
- [ ] `CHANGELOG.md`에 11 phase별 주요 PR 링크
- [ ] `pyproject.toml` version = "1.0.0", `__version__ == "1.0.0"`
- [ ] Identity verification 3종 모두 green
- [ ] `git tag -a v1.0.0` 생성
- [ ] PyPI `pip install macrocast==1.0.0` 가능 (Trusted Publishing)
- [ ] RTD stable branch가 v1.0.0 태그 빌드 green
- [ ] 기존 test + Phase 0-8 test 전부 green

## 8. Docs Deliverables

**신규:**
- `docs/migration/v0_to_v1.md` — breaking change 3종 상세 + migration snippet

**수정 (재작성):**
- `docs/index.md` — horse race identity landing
- `docs/user_guide/index.md` — reading order 갱신
- `docs/dev/architecture.md` — 11-phase outcome 반영, 신규 subpackage (`macrocast/decomposition`, `macrocast/output`, `macrocast/studies`) 포함
- `README.md` — v1 feature list + quickstart + citation
- `CHANGELOG.md` — v0.1.0 → v1.0.0 전체 요약

## 9. Migration Notes

Phase 9 자체가 migration guide 작성 Phase입니다. 사용자 관점 요약:

- **v0 사용자**: `docs/migration/v0_to_v1.md` 참조, 주요 3 breaking change 확인
- **신규 사용자**: `docs/index.md` → `horse_race_quickstart.md` → `decomposition_tutorial.md` → `paper_ready_bundle.md` 읽기 순서
- **Pinning**: `pip install macrocast==1.0.*` 권장 (1.x SemVer 약속)
- **Deprecation policy**: v1.x 내에서 breaking change 없음 (SemVer 엄수); v2.0까지 최소 1년

## 10. Cross-references

- Infra files used: `plans/infra/license_and_release.md` (PyPI Trusted Publishing), `plans/infra/semver_policy.md`
- ADRs referenced: Resolved Decision #7 (PyPI only, no conda-forge for v1.0)
- Coverage Ledger rows resolved:
  - v1.0 release candidate → operational
  - Public API surface freeze → operational
  - Migration documentation → operational
- Upstream consolidation: Phase 1-8 per-phase docs를 landing에서 통합 참조
- Downstream: Phase 10 (v1.1 enhancements) / Phase 11 (interactive dashboard)가 이 Phase의 v1.0 surface 위에서 출발

## 11. GitHub Issue Map

- Epic: (TBD at kickoff — [PHASE-09] v1.0 docs rewrite + release cutoff)
- Sub-task issues: 8개 (09.1~09.8)
- Release issue: `[RELEASE] v1.0.0` — identity verification checklist + PyPI publish checklist

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 Phase 9에서 추출)
