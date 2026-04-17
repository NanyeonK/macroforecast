# ADR-003: Sweep Variant Parallelism is Opt-In for v1.0

Status: **Accepted** (2026-04-17)
Date: 2026-04-17
Context-phase: phase-01

## Context

Phase 1 sweep runner는 N variants를 순회. **각 variant가 execute_recipe 내부에서 `parallel_by_model` 사용 가능** → sweep 레벨 병렬화와 중첩 가능성.

**선택지:**
- (A) 기본 serial, `compute_mode=parallel_by_variant` opt-in
- (B) 기본 parallel, `compute_mode=serial` 명시적 지정
- (C) nested 금지 — inner `parallel_by_model` 자동 비활성화 when sweep parallel

## Decision

**(A) v1.0 = serial 기본. parallel_by_variant는 opt-in.**

## Consequences

**+** 결정성 보장이 최우선 (Phase 0 seed policy가 serial 전제)
**+** 디버그 용이 — variant 실패시 재현 쉬움
**+** Race condition 위험 최소
**−** 실행 시간 긺 (N variants × single-path runtime)
**−** CI 시간 길 수 있음 (완화: CI에서 min-variant recipe fixture 사용)

**Post-v1.0 계획:**
- v1.1에서 `compute_mode=parallel_by_variant` opt-in 형태로 smoke test
- v2에서 기본값 승격 고려 (stable 확인 후)

## References

- Used by: phase-01 (sweep runner default), phase-05 (deep models의 torch determinism과 조합)
- Related: phase-11 distributed compute — cluster-level 병렬은 v2 스코프
- Resolved Decision #1 (완성도 최우선) — parallelism > correctness 트레이드오프 없이 correctness 우선
