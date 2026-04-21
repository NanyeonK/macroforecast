# ADR-005: Registry Axis Declares `component` Metadata for 4.5 Decomposition

Status: **Accepted** (2026-04-17)
Date: 2026-04-17
Context-phase: phase-07

## Context

Phase 7 decomposition engine은 "이 axis가 어느 4.5 component에 기여하는가?"를 알아야 함 (nonlinearity/regularization/cv_scheme/loss/preprocessing/feature_builder/benchmark/importance).

**선택지:**
- (A) Engine이 hard-coded axis→component mapping 유지
- (B) `AxisDefinition`에 `component: str | None` 필드 추가, axis 파일이 self-declare
- (C) 별도 mapping YAML

## Decision

**(B) AxisDefinition 필드 추가.** 각 axis 파일이 소속 component를 self-declare.

```python
# macrocast/registry/preprocessing/scaling_policy.py
AXIS_DEFINITION = AxisDefinition(
    axis_name='scaling_policy',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    component='preprocessing',  # <-- NEW
    entries=(...),
)
```

**Default:** `component=None` (decomposition 대상 아님, 예: logging_level)

**Component 값 (fixed enum):** one of `["nonlinearity", "regularization", "cv_scheme", "loss", "preprocessing", "feature_builder", "benchmark", "importance", None]`

## Consequences

**+** Single source of truth — axis 파일이 자기 소속 선언
**+** 신규 axis 추가시 decomposition mapping 자동 포함
**+** Engine이 axis 메타만 읽어 처리 (hard-coded mapping 유지 불필요)
**−** 일부 axis는 component 2개 이상에 기여 (예: ridge vs randomforest = nonlinearity + regularization 혼합). **v1.0 ANOVA baseline은 primary component 1개만.** v1.1 Shapley에서 multi-component 지원.
**−** Legacy axis 파일 일괄 업데이트 필요 (Phase 7 sub-task)

## References

- Used by: phase-07 (engine + components)
- Cross: ADR-002 (ANOVA baseline 선택과 연결 — single-component mapping만 지원)
- Registry base: `macrocast/registry/base.py`의 `AxisDefinition` dataclass
- Migration: Phase 7 작업 중 기존 125 axis 파일 일괄 update 필요 (대부분 None, 핵심 축 ~30개가 non-None)
