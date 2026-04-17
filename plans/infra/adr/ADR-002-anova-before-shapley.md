# ADR-002: ANOVA Baseline Before Shapley Attribution

Status: **Accepted** (2026-04-17)
Date: 2026-04-17
Context-phase: phase-07

## Context

§4.5 Decomposition engine은 forecast-error variance를 component (nonlinearity/regularization/cv_scheme/loss/preprocessing/feature_builder/benchmark/importance)로 분해. **선택지:**
- (A) ANOVA 기반 — 일원분산분석, 수학적 투명
- (B) Shapley attribution — 축 간 interaction까지 정확히, 계산 비싸
- (C) 둘 다 동시에

## Decision

**Phase 7 v1.0 = ANOVA baseline만.** Shapley는 v1.1 enhancement (같은 engine, `attribution_method` 파라미터).

## Consequences

**+** v1.0 ship 빠름 — ANOVA는 standard stats
**+** 수학적 투명성 — 연구자가 결과 재현 가능 (`statsmodels.stats.anova_lm` 수준)
**+** v1.1에서 Shapley 추가시 breaking change 없음 (새 attribution_method 옵션)
**−** One-way ANOVA는 축 간 interaction 정확히 못 잡음
**−** 일부 variance가 "residual"로 빠질 수 있음 (축에 매핑 안 된 효과)

## References

- Used by: phase-07 (v1.0 implementation), phase-10.5 (v1.1 Shapley enhancement)
- Cross: Phase 7 plan §4 engine algorithm
- Rationale: v1.0 cite-worthy 판정에 ANOVA 분해만으로 충분. Shapley는 독립 논문 형태로 v1.1에서 publish 가능.
