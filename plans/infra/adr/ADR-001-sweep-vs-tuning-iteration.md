# ADR-001: Sweep Executor Shares Iteration Layer with Tuning Engine

Status: **Accepted** (2026-04-17)
Date: 2026-04-17
Context-phase: phase-01

## Context

Phase 1 sweep runner는 `N variants × execute_recipe()` 구조. Phase 3+ tuning engine은 `N trials × train+validate` 구조. 두 루프는 **의미적으로 다르지만 기계적으로 동일** (budget, fail policy, parallel spread, artifact merge).

**선택지:**
- (A) Sweep runner를 tuning engine subclass로
- (B) 공통 iteration 레이어 추출, 둘 다 호출
- (C) 완전 분리 (중복 구현 용인)

## Decision

**(B) 공통 iteration 레이어 추출.** `macrocast/execution/iteration.py`에 `iterate_with_budget()` 추상화. SweepRunner와 `run_tuning()` 각자 호출.

## Consequences

**+** DRY — budget/failure/parallel 로직 한 곳
**+** Sweep runner와 tuning이 각자의 API surface 유지 (사용자 관점에서 독립)
**+** Tuning 내부 리팩터가 sweep에 영향 없음 (iteration 인터페이스만 유지)
**−** Strict subclass 관계 강요보다 coupling 낮지만, 공통 변경 시 두 호출자 모두 회귀 테스트 필요

## References

- Used by: phase-01 (sweep runner), phase-07 (decomposition도 iteration 활용 가능)
- Rejected alternative: subclass approach. 이유 — tuning의 inner loop는 validation score로 HP pick, sweep의 outer loop는 OOS metric aggregate. 의미 다른 두 루프를 하나의 class hierarchy로 강제하면 abstraction leak 위험.
