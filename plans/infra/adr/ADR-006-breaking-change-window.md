# ADR-006: Breaking Changes Maintain One-Version Deprecation Window

Status: **Accepted** (2026-04-17)
Date: 2026-04-17
Context-phase: cross-cutting

## Context

Phase 2 (stat_test 1축 → 8축 분해) 등 inevitable breaking change 존재. **선택지:**
- (A) Breaking change 즉시 적용, legacy 즉시 제거
- (B) DeprecationWarning + migration shim 1 version 유지 후 제거
- (C) 영구 backward compat (shim 영원히 유지)

## Decision

**(B) One-version deprecation window.**

**Pattern:**
1. Release N — new API 추가, legacy는 migration shim + `DeprecationWarning` emit
2. Release N+1 — legacy shim 유지, warning 강화 (`PendingDeprecationWarning → DeprecationWarning`)
3. Release N+2 — legacy 제거, CHANGELOG에 breaking 명기

**Concrete examples:**
- **Phase 2 stat_test 축 분해:**
  * v1.0 — `equal_predictive/nested/...` 신규 축 + `stat_test` legacy migrate + warning
  * v1.2 — `stat_test` legacy 완전 제거

- **Phase 0 `execute_recipe()` cache_root 추가:**
  * Not breaking (optional param, default=None preserves old behavior)

## Consequences

**+** Users가 점진적 migration 가능
**+** CHANGELOG / migration docs가 clear
**+** Legacy code 복잡성 누적 방지 (3 version 유지 아님)
**−** Release N에서 2 API 공존 — 내부 테스트 코드량 증가 (legacy + new 둘 다 test)
**−** 문서가 "legacy way + new way" 두 방식 설명해야 함 (one version only)

**When to break WITHOUT warning:**
- Pre-v1.0 (v0.X 단계)은 전부 pre-release로 간주. Breaking change 허용, CHANGELOG에만 기록.
- v1.0 이후부터는 이 ADR 규칙 엄격 적용.

## References

- Used by: phase-02 (stat_test split), any future axis rename/split
- Cross: phase-09 migration guide (`docs/migration/v0_to_v1.md`)
- Pattern source: Python PEP 387 backward compatibility policy 참조
- Resolved Decision #4 (version ladder v0.2→v1.0): pre-v1.0은 자유도 높음
