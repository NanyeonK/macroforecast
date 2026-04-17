# Phase 02 — Statistical Test Axis 8-Way Split

| Field | Value |
|-------|-------|
| Phase ID | phase-02 |
| Priority (inter-phase) | **P1** |
| Depends on | phase-01 |
| Unlocks | phase-06, phase-07 |
| Version tag target | v0.4 |
| Status | pending |

## 1. Goal

Monolithic `stat_test` axis 1개를 의미론적 8축 (equal_predictive / nested / cpa_instability / multiple_model / density_interval / direction / residual_diagnostics / test_scope)으로 분해합니다. Horse race 결과를 해석할 때 어떤 가설을 어떤 검정으로 수행했는지 축별로 명시되어야 하며, 이는 Phase 1 sweep이 산출한 variant 비교의 통계적 defensibility를 직결로 결정합니다.

## 2. Scope

**In scope:**
- 8개 test axis 등록 (`equal_predictive`, `nested`, `cpa_instability`, `multiple_model`, `density_interval`, `direction`, `residual_diagnostics`, `test_scope`)
- Test dispatcher 재구성 — 8축 spec을 받아 per-axis result dict 반환
- `execute_recipe()` 내부 `stat_dispatch` dict 교체 (build.py line 2285 영역)
- Legacy `stat_test` → 8축 migration shim + `DeprecationWarning`
- 3개 신규 test 파일 작성
- User guide / math docs rewrite

**Out of scope:**
- 신규 statistical test 추가 (v1.1) — 기존 21 test만 재분류
- Legacy `stat_test` axis 완전 제거 (v1.2, breaking change window)
- Multiple-target joint test (Phase 5a 이후)

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 02.1 | 8개 신규 axis 파일 등록 | **P0** | ~400 | `macrocast/registry/tests/{equal_predictive,nested,cpa_instability,multiple_model,density_interval,direction,residual_diagnostics,test_scope}.py` (신규 8) | `test_stat_test_split.py` axis enum 검증 green |
| 02.2 | `dispatch_stat_tests()` 작성 | **P0** | ~250 | `macrocast/execution/stat_tests/dispatch.py` (신규), `macrocast/execution/stat_tests/__init__.py` (신규) | `test_stat_test_dispatch.py` green |
| 02.3 | `execute_recipe()` 내부 `stat_dispatch` 교체 | **P0** | ~80 | `macrocast/execution/build.py` (lines 2285 영역) | 기존 291 test 회귀 없음 |
| 02.4 | Legacy `stat_test` migration shim | P1 | ~120 | `macrocast/compiler/migrations/stat_test_split.py` (신규), `macrocast/compiler/migrations/__init__.py` (수정) | `test_stat_test_migration.py` green + `DeprecationWarning` 발생 |
| 02.5 | Phase 2 tests | **P0** | ~350 | `tests/test_stat_test_split.py`, `tests/test_stat_test_dispatch.py`, `tests/test_stat_test_migration.py` (신규 3) | 3개 test 전부 green |
| 02.6 | Phase 2 docs | P1 | ~300 | `docs/user_guide/stat_test_selection.md` (신규), `docs/user_guide/stat_tests.md` (rewrite), `docs/math/stat_tests.md` (rewrite) | RTD build green |

## 4. API / Schema Specifications

### 4.1 8-Axis Registry Structure

각 axis 파일은 Phase 1의 axis registry 패턴을 따릅니다.

```python
# macrocast/registry/tests/equal_predictive.py
from __future__ import annotations
from macrocast.registry.base import register_axis, AxisSpec

register_axis(AxisSpec(
    name="equal_predictive",
    layer=6,
    status="operational",
    values=(
        "dm",
        "dm_hln",
        "dm_modified",
        "paired_t_on_loss_diff",
        "wilcoxon_signed_rank",
    ),
    description=(
        "Equal predictive accuracy tests for non-nested model pairs. "
        "H0: E[loss_A - loss_B] = 0."
    ),
))
```

8축 values (total 43 leaf):

| Axis | Values |
|------|--------|
| `equal_predictive` | `dm`, `dm_hln`, `dm_modified`, `paired_t_on_loss_diff`, `wilcoxon_signed_rank` |
| `nested` | `cw`, `enc_new`, `mse_f`, `mse_t`, `forecast_encompassing_nested` |
| `cpa_instability` | `cpa`, `rossi`, `rolling_dm`, `fluctuation_test`, `chow_break_forecast`, `cusum_on_loss` |
| `multiple_model` | `reality_check`, `spa`, `mcs`, `stepwise_mcs`, `bootstrap_best_model` |
| `density_interval` | `PIT_uniformity`, `berkowitz`, `kupiec`, `christoffersen_unconditional`, `christoffersen_independence`, `christoffersen_conditional`, `interval_coverage` |
| `direction` | `pesaran_timmermann`, `mcnemar`, `binomial_hit`, `roc_comparison` |
| `residual_diagnostics` | `mincer_zarnowitz`, `ljung_box`, `arch_lm`, `bias_test`, `autocorrelation_of_errors`, `serial_dependence_loss_diff`, `diagnostics_full` |
| `test_scope` | `per_target`, `per_horizon`, `per_model_pair`, `full_grid_pairwise`, `benchmark_vs_all`, `regime_specific_tests`, `subsample_tests` |

### 4.2 `dispatch_stat_tests()` API

```python
# macrocast/execution/stat_tests/dispatch.py
from __future__ import annotations
import pandas as pd

def dispatch_stat_tests(
    *,
    predictions: pd.DataFrame,
    stat_test_spec: dict[str, str | None],  # {"equal_predictive": "dm", "nested": "cw", ...}
    dependence_correction: str,
) -> dict[str, dict]:
    """Run all requested tests across 8 axes, return per-axis result dict.

    Args:
        predictions: long-form (model, target, horizon, date, y_hat, y)
        stat_test_spec: dict keyed by axis name (8 axes total); None/missing axis = skip
        dependence_correction: HAC / Newey-West / bootstrap window rule identifier

    Returns:
        {"equal_predictive": {"test": "dm", "statistic": ..., "pvalue": ..., "n": ...},
         "nested":           {"test": "cw", ...},
         ...}
        Skipped axes are absent from output.
    """
```

**Invariants:**
- spec에 없는 axis는 결과에서 생략 (None 반환하지 않음)
- 각 axis 결과에는 항상 `test`, `statistic`, `pvalue`, `n` 키 존재
- `test_scope` axis는 다른 축의 적용 범위를 제어 (per-horizon → horizon별 반복 실행)
- 내부 실패는 per-axis error dict로 surface (`{"error": str, "exc_type": str}`)

### 4.3 Migration Shim

```python
# macrocast/compiler/migrations/stat_test_split.py
from __future__ import annotations
import warnings

_LEGACY_TO_NEW: dict[str, tuple[str, str]] = {
    # equal_predictive
    "dm": ("equal_predictive", "dm"),
    "dm_hln": ("equal_predictive", "dm_hln"),
    "dm_modified": ("equal_predictive", "dm_modified"),
    "paired_t_on_loss_diff": ("equal_predictive", "paired_t_on_loss_diff"),
    "wilcoxon_signed_rank": ("equal_predictive", "wilcoxon_signed_rank"),
    # nested
    "cw": ("nested", "cw"),
    "enc_new": ("nested", "enc_new"),
    "mse_f": ("nested", "mse_f"),
    "mse_t": ("nested", "mse_t"),
    # cpa_instability
    "cpa": ("cpa_instability", "cpa"),
    "rossi": ("cpa_instability", "rossi"),
    # multiple_model
    "reality_check": ("multiple_model", "reality_check"),
    "spa": ("multiple_model", "spa"),
    "mcs": ("multiple_model", "mcs"),
    # density_interval
    "PIT_uniformity": ("density_interval", "PIT_uniformity"),
    "berkowitz": ("density_interval", "berkowitz"),
    "kupiec": ("density_interval", "kupiec"),
    "christoffersen_conditional": ("density_interval", "christoffersen_conditional"),
    # direction
    "pesaran_timmermann": ("direction", "pesaran_timmermann"),
    # residual_diagnostics
    "mincer_zarnowitz": ("residual_diagnostics", "mincer_zarnowitz"),
    "ljung_box": ("residual_diagnostics", "ljung_box"),
}

def migrate_legacy_stat_test(path_layer_6: dict) -> dict:
    """Rewrite {stat_test: X} -> {<axis>: X} with DeprecationWarning.

    Only runs when legacy key present; idempotent on already-migrated dicts.
    Raises ValueError on unknown legacy values.
    """
    if "stat_test" not in path_layer_6:
        return path_layer_6
    legacy_val = path_layer_6["stat_test"]
    if legacy_val not in _LEGACY_TO_NEW:
        raise ValueError(f"unknown legacy stat_test value: {legacy_val!r}")
    new_axis, new_value = _LEGACY_TO_NEW[legacy_val]
    warnings.warn(
        f"`stat_test: {legacy_val}` is deprecated; use `{new_axis}: {new_value}` "
        "(legacy field will be removed in v1.2).",
        DeprecationWarning,
        stacklevel=3,
    )
    out = {k: v for k, v in path_layer_6.items() if k != "stat_test"}
    out.setdefault(new_axis, new_value)
    return out
```

## 5. File Layout

**신규:**
- `macrocast/registry/tests/__init__.py`
- `macrocast/registry/tests/equal_predictive.py`
- `macrocast/registry/tests/nested.py`
- `macrocast/registry/tests/cpa_instability.py`
- `macrocast/registry/tests/multiple_model.py`
- `macrocast/registry/tests/density_interval.py`
- `macrocast/registry/tests/direction.py`
- `macrocast/registry/tests/residual_diagnostics.py`
- `macrocast/registry/tests/test_scope.py`
- `macrocast/execution/stat_tests/__init__.py`
- `macrocast/execution/stat_tests/dispatch.py`
- `macrocast/compiler/migrations/stat_test_split.py`
- `tests/test_stat_test_split.py`
- `tests/test_stat_test_dispatch.py`
- `tests/test_stat_test_migration.py`
- `docs/user_guide/stat_test_selection.md`

**수정:**
- `macrocast/execution/build.py` — `stat_dispatch` dict 제거, `dispatch_stat_tests()` 호출로 교체 (line 2285 영역)
- `macrocast/compiler/build.py` — migration shim 호출 훅 (compile_recipe_dict 진입부)
- `macrocast/compiler/migrations/__init__.py` — migration 등록
- `macrocast/__init__.py` — `dispatch_stat_tests` 공개 API
- `docs/user_guide/stat_tests.md` (rewrite)
- `docs/math/stat_tests.md` (rewrite)

## 6. Test Strategy

### `tests/test_stat_test_split.py`
- 8 axis 파일이 모두 axis registry에 등록됨 (`get_axis("equal_predictive")` 등)
- 각 axis의 value set이 §4.1 표와 일치
- `layer=6` 필드 일관성
- status = `operational`

### `tests/test_stat_test_dispatch.py`
- Full 8-axis spec → 8 key result dict
- Partial spec (3축만 지정) → 3 key만 반환, 나머지 축 생략
- 잘못된 test value → per-axis `{"error": ...}` surface (전체 raise 아님)
- `test_scope=per_horizon` → equal_predictive를 horizon 수만큼 반복
- 동일 predictions + 동일 spec → 동일 result dict (결정성)

### `tests/test_stat_test_migration.py`
- `{stat_test: "dm"}` → `{equal_predictive: "dm"}` + `DeprecationWarning` 발생
- 21 legacy value 모두 rewrite 성공
- 알 수 없는 legacy value → `ValueError`
- 이미 `{equal_predictive: "dm"}`만 있는 dict → idempotent (변경 없음)
- Recipe YAML 로드 → compile 경로에서 shim이 자동 호출됨

## 7. Acceptance Gate

- [ ] Phase 1 gate 선통과
- [ ] 8개 axis 파일 모두 registry에 등록 + status = operational
- [ ] `dispatch_stat_tests()` 공개 API export
- [ ] 기존 291 test + Phase 0/1 test + Phase 2 신규 3개 test 전부 green
- [ ] Legacy `{stat_test: X}` recipe 로드 시 `DeprecationWarning` 정확히 1회 발생
- [ ] `grep -rE "stat_dispatch\s*=\s*\{" macrocast/execution/` → 0 hits (legacy dict 삭제 완료)
- [ ] Phase 2 docs 3종 RTD build green
- [ ] `docs/user_guide/stat_test_selection.md`의 axis 선택 결정 트리 재현 가능

## 8. Docs Deliverables

**신규:**
- `docs/user_guide/stat_test_selection.md` — "내 실험에 어떤 축을 써야 하나" 결정 트리 (nested vs non-nested, single-pair vs multi-model, density vs point, etc.)

**Rewrite:**
- `docs/user_guide/stat_tests.md` — 8축 각각의 use case + YAML 예시
- `docs/math/stat_tests.md` — 43 leaf 각각의 수식 + 가정 + 대표 reference

## 9. Migration Notes

- **Breaking change window:** v1.0에 migration shim + `DeprecationWarning` 도입, v1.2에서 legacy `stat_test` 필드 완전 제거 (ADR-006 참조).
- 사용자 영향:
  - v1.0~v1.1: 기존 recipe YAML 그대로 동작 (shim이 자동 rewrite) + warning
  - v1.2: legacy `stat_test` 필드 → `ValueError` (shim 제거)
- 마이그레이션 스크립트: 사용자는 `python -m macrocast.compiler.migrations.stat_test_split <recipe.yaml>` 로 자동 변환 가능 (v1.1에 CLI 추가, out of scope for Phase 2)
- Sweep plan과 호환: `sweep_axes` 에서 8 axis 중 임의 개수 sweep 가능 (Phase 1 기능 위에 얹힘)

## 10. Cross-references

- Infra files used: `plans/infra/test_dispatch.md` (신규, Phase 2 kickoff 시 작성)
- ADRs referenced: ADR-006 (breaking change window for v1.x axis split)
- Coverage Ledger rows resolved:
  - Layer 6 `stat_test` (monolithic, 21 values) → split into 8 axes
  - Layer 6 `equal_predictive` / `nested` / `cpa_instability` / `multiple_model` / `density_interval` / `direction` / `residual_diagnostics` / `test_scope` → operational
- Downstream: Phase 6 (per-axis result reporting), Phase 7 (horse race aggregation across test axes)

## 11. GitHub Issue Map

- Epic: (TBD at kickoff — [PHASE-02] Statistical test axis 8-way split)
- Sub-task issues: 6개 (02.1~02.6)

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 §Phase 2에서 추출)

## 13. References

- `plans/ultraplan_v2.2.md` §Phase 2 — 원본 사양
- ADR-006 — v1.x breaking change window policy
- `docs/math/stat_tests.md` (current) — 기존 21 test 수식 (재분류 기준)
