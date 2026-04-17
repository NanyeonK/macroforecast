# macrocast Roadmap — Horse Race Research Benchmark Package

**Master plan overview.** Detailed phase plans live under `plans/phases/`. Cross-cutting infrastructure lives under `plans/infra/`.

Last updated: 2026-04-17
Roadmap version: 3.0 (master, post-restructure)

---

## 1. Context

macrocast은 FRED-MD/QD/SD 기반 공정·재현가능·확장가능한 macro forecasting benchmarking package입니다. 현 상태 (2026-04-17):

- 소스 ~12,000 LOC, 테스트 291 green
- Registry 125 axes / 836 values / 224 operational
- End-to-end 실행 가능: `study_mode = single_path_benchmark_study` **하나뿐**
- Sphinx + Read the Docs 배포 파이프라인 가동 중

Package의 stated identity — **horse race research benchmark** — 가 아직 실행 불가능합니다. 이 roadmap은 그 gap을 11 phase로 해소합니다.

## 2. Target Identity

> **"연구자가 한 recipe를 기준으로 임의의 축을 sweep하면 — 나머지가 공정하게 고정된 상태에서 — 결과가 paper-ready bundle로 출력된다. Forecast 성능 차이의 효과 분해 (§4.5)가 포함되어 연구자가 자신의 기여를 cite-worthy 수준으로 입증할 수 있다."**

v1.0은 universe 전체 커버리지가 아닌 **horse race + decomposition + bundle 축의 최소 완결**을 목표로 합니다.

## 3. Design Principles

**기존 (plan_04_14_1734에서 계승):**

1. One recipe = one fully specified study
2. Grammar first, content later
3. Represent before execute
4. Fair comparison by construction

**추가 (이 roadmap을 통해 명문화):**

5. Identity-first sequencing — breadth 확장보다 horse-race 실행 완결이 우선
6. Reuse over rebuild — tuning engine의 budget/trial 루프를 sweep 루프와 공유
7. Docs ride each phase — phase별 산출과 함께 docs 페이지 동시 발행
8. Cite-worthy output — decomposition + significance + regime breakdown 자동 생성
9. Deterministic artifact — 동일 recipe + 동일 seed → 동일 artifact hash
10. Breaking changes → migration path — legacy recipe는 최소 1 version DeprecationWarning 유지

## 4. Phase Dependency Graph

```
Phase 0 (single-path stability)
   │
   ▼
Phase 1 (sweep executor) ◀── IDENTITY UNLOCK
   │
   ├─▶ Phase 2 (stat test 8축 분해)
   │
   ├─▶ Phase 3 (7 data_task 축 + separation_rule)
   │
   ├─▶ Phase 4 (3 benchmark eval 축)
   │
   ├─▶ Phase 5a (LSTM/GRU/TCN/VAR/BVAR)
   │
   └─▶ Phase 6 (ablation + replication runner)
          │
          └─▶ Phase 7 (§4.5 decomposition engine)
                 │
                 └─▶ Phase 8 (paper_ready_bundle + regime eval)
                        │
                        └─▶ Phase 9 (docs consolidation + v1.0 cutoff)

   [Post-v1.0]
   Phase 10 (v1.1: Transformer/NBEATS/TFT/DFM/FAVAR, benchmark_suite,
             multi-target joint, BEA/BLS/OECD/ECB/IMF/SPF operational,
             importance uplift)
   Phase 11 (v2: state_space/TVP_AR/MIDAS, distributed compute,
             exotic data, R-parity)
```

## 5. Phase Summary

| Phase | Name | Priority | Version | Plan | Status |
|:-----:|------|:--------:|:-------:|------|:------:|
| 0 | Single-path stability & sweep-safety | **P0** | v0.2 | [phase_00_stability.md](phases/phase_00_stability.md) | **completed** |
| 1 | Horse-race sweep executor — **IDENTITY** | **P0** | v0.3 | [phase_01_sweep_executor.md](phases/phase_01_sweep_executor.md) | pending |
| 2 | Statistical test axis 8-way split | P1 | v0.4 | [phase_02_stat_test_split.md](phases/phase_02_stat_test_split.md) | pending |
| 3 | Data/task axes + preprocessing separation | P1 | v0.5 | [phase_03_data_task_axes.md](phases/phase_03_data_task_axes.md) | pending |
| 4 | Benchmark evaluation axes | P1 | v0.6 | [phase_04_benchmark_eval.md](phases/phase_04_benchmark_eval.md) | pending |
| 5a | Deep & time-series models (core + [deep]) | P1 | v0.7 | [phase_05a_deep_tsm.md](phases/phase_05a_deep_tsm.md) | pending |
| 5b | More models (Transformer/NBEATS/TFT/DFM/FAVAR) | P2 | v1.1 | [phase_05b_more_models.md](phases/phase_05b_more_models.md) | pending |
| 5c | State-space, TVP_AR, MIDAS | P2 | v2 | [phase_05c_state_space.md](phases/phase_05c_state_space.md) | pending |
| 6 | Ablation + replication runners | P1 | v0.8 | [phase_06_ablation_replication.md](phases/phase_06_ablation_replication.md) | pending |
| 7 | §4.5 Decomposition engine | **P0** | v0.9 | [phase_07_decomposition.md](phases/phase_07_decomposition.md) | pending |
| 8 | paper_ready_bundle + aggregation | **P0** | v0.9 | [phase_08_paper_bundle.md](phases/phase_08_paper_bundle.md) | pending |
| 9 | Docs rewrite + v1.0 cutoff | **P0** | **v1.0** | [phase_09_v1_cutoff.md](phases/phase_09_v1_cutoff.md) | pending |
| 10 | v1.1 scope catalog | P2 | v1.1 | [phase_10_v1_1_scope.md](phases/phase_10_v1_1_scope.md) | pending |
| 11 | v2 scope catalog | P2 | v2 | [phase_11_v2_scope.md](phases/phase_11_v2_scope.md) | pending |

## 6. Phase Priority Matrix (inter-phase)

- **P0 critical path:** 0 → 1 → 7 → 8 → 9
- **P1 (v1.0 gate):** 2, 3, 4, 5a, 6
- **P2 (post-v1.0):** 5b, 5c, 10, 11

Phase 내 sub-task 우선순위는 각 phase plan의 §3 Sub-Tasks 테이블 참조.

## 7. Version Ladder

| Version | 포함 Phase | 마일스톤 |
|---------|-----------|----------|
| v0.2 | Phase 0 완료 | single-path sweep-safe |
| v0.3 | + Phase 1 | **horse race executor operational (identity unlock)** |
| v0.4 | + Phase 2 | stat test 8축 분해 |
| v0.5 | + Phase 3 | 7 data_task 축 + separation_rule |
| v0.6 | + Phase 4 | benchmark eval 축 |
| v0.7 | + Phase 5a | LSTM/GRU/TCN/VAR/BVAR (optional [deep] extra) |
| v0.8 | + Phase 6 | ablation + replication runner |
| v0.9 | + Phase 7 + 8 | decomposition + paper_ready_bundle |
| **v1.0.0** | + Phase 9 | docs consolidation + PyPI release |
| v1.1 | + Phase 5b + 10 | v1.1 카탈로그 |
| v2.0 | + Phase 5c + 11 | v2 카탈로그 |

**Release 원칙 (Resolved Decision #1):** 시간 기한 없음, gate 통과 기반. 각 phase의 Acceptance Gate 전부 통과 시점에 해당 minor 버전 tag.

## 8. v1.0 Release Criterion (Demo Scenarios)

릴리즈 전 3개 end-to-end 시나리오가 공개 study API로 통과해야 함:

1. **Synthetic replication round-trip** — `execute_recipe()` 결과를 `replication_recipe`로 재실행 → byte-identical predictions + metrics + manifest
2. **LSTM vs baselines horse race** — `[deep]` 설치 환경: LSTM 포함 5-variant / 미설치 환경: LSTM 제외 4-variant 둘 다 green + decomposition + paper-ready bundle
3. **Preprocessing attribution demo** — scaling/tcode/outlier 3-axis sweep → decomposition이 preprocessing share 정량화

구체적 논문 replication (CLSS 2021 등)은 **v1.0 이후 별도 examples로 추가** (Resolved Decision #3).

## 9. Status Tracker

| Phase | Status | Version | Epic issue | Updated |
|:-----:|:------:|:-------:|:----------:|:-------:|
| 0 | completed | v0.2 | PR #8 | 2026-04-17 |
| 1 | pending | v0.3 | - | - |
| 2-11 | pending | - | - | - |

현재 live phase: **Phase 1 kickoff 대기** (Phase 0 완료, v0.2 tag 준비)

## 10. Pointer Table (separate plans)

**Phase plans:**
- [`phases/phase_00_stability.md`](phases/phase_00_stability.md)
- [`phases/phase_01_sweep_executor.md`](phases/phase_01_sweep_executor.md)
- [`phases/phase_02_stat_test_split.md`](phases/phase_02_stat_test_split.md)
- [`phases/phase_03_data_task_axes.md`](phases/phase_03_data_task_axes.md)
- [`phases/phase_04_benchmark_eval.md`](phases/phase_04_benchmark_eval.md)
- [`phases/phase_05a_deep_tsm.md`](phases/phase_05a_deep_tsm.md)
- [`phases/phase_05b_more_models.md`](phases/phase_05b_more_models.md)
- [`phases/phase_05c_state_space.md`](phases/phase_05c_state_space.md)
- [`phases/phase_06_ablation_replication.md`](phases/phase_06_ablation_replication.md)
- [`phases/phase_07_decomposition.md`](phases/phase_07_decomposition.md)
- [`phases/phase_08_paper_bundle.md`](phases/phase_08_paper_bundle.md)
- [`phases/phase_09_v1_cutoff.md`](phases/phase_09_v1_cutoff.md)
- [`phases/phase_10_v1_1_scope.md`](phases/phase_10_v1_1_scope.md)
- [`phases/phase_11_v2_scope.md`](phases/phase_11_v2_scope.md)

**Cross-cutting infra:**
- [`infra/seed_policy.md`](infra/seed_policy.md)
- [`infra/failure_policy.md`](infra/failure_policy.md)
- [`infra/study_manifest_schema.md`](infra/study_manifest_schema.md)
- [`infra/cache_discipline.md`](infra/cache_discipline.md)
- [`infra/ci_workflow.md`](infra/ci_workflow.md)
- [`infra/license_and_release.md`](infra/license_and_release.md)

**ADRs:**
- [`infra/adr/ADR-001-sweep-vs-tuning-iteration.md`](infra/adr/ADR-001-sweep-vs-tuning-iteration.md)
- [`infra/adr/ADR-002-anova-before-shapley.md`](infra/adr/ADR-002-anova-before-shapley.md)
- [`infra/adr/ADR-003-sweep-parallel-opt-in.md`](infra/adr/ADR-003-sweep-parallel-opt-in.md)
- [`infra/adr/ADR-004-deep-learning-optional.md`](infra/adr/ADR-004-deep-learning-optional.md)
- [`infra/adr/ADR-005-component-metadata-field.md`](infra/adr/ADR-005-component-metadata-field.md)
- [`infra/adr/ADR-006-breaking-change-window.md`](infra/adr/ADR-006-breaking-change-window.md)

**Coverage tracking:**
- [`coverage_ledger.md`](coverage_ledger.md) — user option universe ~900 values → phase 배정

**Preserved live documents:**
- `phase0_audit_2026_04_17.md` — Phase 0 detailed audit
- `implementation-issues.md` — legacy issue tracker
- `status_04_16_canonical.md` — pre-restructure snapshot

## 11. Resolved Decisions (2026-04-17)

1. **개발 기간**: 완성도 최우선, 시간 제약 없음. 각 phase는 gate 통과로 완료 판정.
2. **Deep learning 설치 정책**: 사용자가 install 시점에 선택. `pip install macrocast` (core only) vs `pip install macrocast[deep]`.
3. **Replication 대상**: v1.0에서는 결정 안 함. synthetic round-trip으로 v1.0 gate 판정. 구체 논문 replication은 post-v1.0.
4. **Semantic versioning**: v0.2 → v0.3 → … → v0.9 → v1.0.0.
5. **License**: MIT 유지. data adapter 상단에 source terms 주석.
6. **CI/CD 플랫폼**: GitHub Actions (workflows 4종 — `infra/ci_workflow.md`).
7. **릴리즈 채널**: PyPI only. Trusted Publishing (OIDC).

## 12. Revision History

- **v3.1 (2026-04-17):** Phase 0 완료 반영 (status completed, v0.2 tag 준비)
- **v3.0 (2026-04-17):** 단일 파일 → master + per-phase + infra/ADR + coverage_ledger + issue templates 재구조화
- v2.2 (archived): priority matrix, failure_policy P0 발견, 시간 추정치 제거
- v2.1 (archived): Resolved Decisions 확정, version ladder, license/release, CI workflow
- v2.0 (archived): Phase 0 audit, per-phase API/schema/test/gate 상세화
- v1.0 (archived): 11-phase roadmap 초안

---

Canonical location: `server1:~/project/macroforecast/plans/ROADMAP.md`

Local pointer: `/Users/nanyeon/.claude/plans/ultraplan-cannot-launch-remote-peaceful-rainbow.md`
