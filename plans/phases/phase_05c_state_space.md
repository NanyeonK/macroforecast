# Phase 05c — State-Space, TVP_AR, MIDAS (v2)

| Field | Value |
|-------|-------|
| Phase ID | phase-05c |
| Priority (inter-phase) | **P2** |
| Depends on | phase-05b (model catalog baseline), phase-10 (v1.1 complete) |
| Unlocks | phase-11 (v2 full scope) |
| Version tag target | v2.0 |
| Status | deferred (post-v1.1) |

## 1. Goal

v2 classical time-series expansion — 고전 계량경제 시계열 가족 3종을 catalog에 추가합니다. Phase 05a/05b에서 확립한 adapter 패턴을 재사용해 state-space / time-varying / mixed-frequency 영역을 커버합니다.

## 2. Scope

**In scope (3 families):**
- `state_space` — Kalman filter 기반 (general state-space via `statsmodels.tsa.statespace`)
- `TVP_AR` — Time-Varying Parameter AR (custom impl 또는 `dlm`-style library)
- `MIDAS` / `U_MIDAS` — Mixed Data Sampling (mixed-frequency data shaping 포함)

**Out of scope:**
- Panel / spatial / GNN models → Phase 11.6
- Bayesian hierarchical models → post-v2 (hierarchical_reconciliation은 Phase 11.6에서 일부 다룸)
- Regime-switching (Markov-switching, TAR, STAR) → post-v2 후보
- Local-level / structural decomposition 그 자체 → Phase 7과 중복, adapter는 본 phase에서만

## 3. Sub-Tasks

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 05c.1 | state_space adapter (statsmodels) | P2 | ~200 | `macrocast/execution/models/classical/state_space.py` (신규) | `test_state_space_adapter.py` green |
| 05c.2 | TVP_AR adapter (custom 또는 dlm) | P2 | ~350 | `macrocast/execution/models/classical/tvp_ar.py` (신규) | `test_tvp_ar_adapter.py` green |
| 05c.3 | MIDAS / U_MIDAS adapter + mixed-freq shaping | P2 | ~400 | `macrocast/execution/models/classical/midas.py` (신규), `macrocast/preprocess/mixed_frequency.py` (신규) | `test_midas_adapter.py` green |
| 05c.4 | Registry entries | P2 | ~30 | `macrocast/execution/models/registry.py` | schema 검증 green |
| 05c.5 | Tests (per-family + sweep-safety) | P2 | ~300 | `tests/test_*_adapter.py`, `tests/test_phase05c_sweep_safety.py` | 4 test files green |
| 05c.6 | Docs | P2 | ~120 | `docs/user/model_catalog.md` 추가 섹션 | RTD build green |

## 4. API / Schema Specifications

Phase 5a `ClassicalModelAdapter` 인터페이스 재사용.

MIDAS는 mixed-frequency input 때문에 preprocess 확장이 필요:

```python
# macrocast/preprocess/mixed_frequency.py
from __future__ import annotations

def align_mixed_frequency(
    *,
    low_freq: "pd.DataFrame",
    high_freq: "pd.DataFrame",
    aggregation: str = "u_midas",  # "u_midas" | "almon" | "beta"
    lags: int,
) -> "pd.DataFrame":
    """저주파 타겟에 고주파 regressor를 lag-polynomial로 맞춘 design matrix 반환."""
    ...
```

Recipe schema 확장:
```yaml
model:
  family: midas
  spec:
    aggregation: u_midas     # or almon / beta
    high_freq_lags: 12
    high_freq_sources: [FRED:DFF, FRED:T10Y2Y]
```

## 5. File Layout

**신규:**
- `macrocast/execution/models/classical/state_space.py`
- `macrocast/execution/models/classical/tvp_ar.py`
- `macrocast/execution/models/classical/midas.py`
- `macrocast/preprocess/mixed_frequency.py`
- `tests/test_state_space_adapter.py`
- `tests/test_tvp_ar_adapter.py`
- `tests/test_midas_adapter.py`
- `tests/test_phase05c_sweep_safety.py`

**수정:**
- `macrocast/execution/models/registry.py` (3 entry 추가)
- `macrocast/preprocess/__init__.py` (mixed_frequency export)
- `docs/user/model_catalog.md`

## 6. Test Strategy

- state_space: AR(1) 재현 → closed-form OLS 결과와 일치 (within numeric tolerance)
- TVP_AR: 합성 데이터로 drift 주입 → 계수 estimation이 true path를 따라가는지 확인
- MIDAS: 월간 inflation + 일간 yield spread toy recipe — predict → metric 산출 green
- `mixed_frequency.align_mixed_frequency()` unit test: shape + NaN 경계 검증
- Sweep-safety: 3 모델을 variant로 섞은 sweep 2회 → 동일 artifact

## 7. Acceptance Gate

- [ ] 3개 신규 모델 registry 등록
- [ ] 모든 adapter 단위 test green
- [ ] MIDAS recipe가 mixed-frequency FRED 데이터로 end-to-end 실행
- [ ] Phase 1 sweep runner에서 3개 모델 variant 오염 없음
- [ ] `docs/user/model_catalog.md`에 3개 모델 사용 예시

## 8. Docs Deliverables

- model_catalog.md에 "Classical — Advanced" 섹션 신설
- MIDAS 예시 recipe: `examples/recipes/midas_inflation.yaml` (신규)

## 9. Migration Notes

- Breaking change 없음 — 추가만
- Recipe schema `model.family` whitelist에 3개 값 추가
- preprocess 확장은 기존 recipe에 영향 없음 (opt-in)

## 10. Cross-references

- Phase 05a — ClassicalModelAdapter 인터페이스 baseline
- Phase 05b — DFM/FAVAR state-space 친척 모델 (statsmodels 기반 의존성 공유)
- Phase 11.6 — panel / spatial 은 별도 trace

## 11. GitHub Issue Map

- Epic: (TBD, v2 kickoff 시 생성 — [PHASE-05c] v2 classical time-series expansion)
- Sub-task issues: (TBD, 6개 생성 예정)

## 12. Revision Log

- 2026-04-17: 초안 (v2 scope catalog에서 분리)

## 13. References

- Durbin & Koopman, *Time Series Analysis by State Space Methods*
- Primiceri (2005) TVP-VAR
- Ghysels, Santa-Clara, Valkanov (2004) MIDAS; Foroni, Marcellino, Schumacher (2015) U-MIDAS
