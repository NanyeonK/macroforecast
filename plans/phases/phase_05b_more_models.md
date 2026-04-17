# Phase 05b — Transformer / NBEATS / TFT / DFM / FAVAR (v1.1)

| Field | Value |
|-------|-------|
| Phase ID | phase-05b |
| Priority (inter-phase) | **P2** |
| Depends on | phase-05a (deep learning integration baseline) |
| Unlocks | phase-10 (v1.1 scope), partial overlap with phase-11 parity items |
| Version tag target | v1.1 |
| Status | deferred (post-v1.0) |

## 1. Goal

v1.1 deep model expansion — attention-based와 non-recurrent 아키텍처 5종을 Phase 5a에서 확립한 adapter 패턴으로 추가합니다. 사용자가 NN/LSTM/GRU/TCN 이외에 구조적으로 상이한 선택지를 가질 수 있도록 모델 catalog를 넓히는 것이 목적입니다.

## 2. Scope

**In scope (5 models):**
- `transformer` — vanilla encoder-only transformer (torch)
- `nbeats` — N-BEATS pure-Python (torch만 필요, 추가 deps 없음)
- `tft` — Temporal Fusion Transformer (`pytorch-forecasting` 활용)
- `dfm` — Dynamic Factor Model (`statsmodels.tsa.statespace.dynamic_factor`, core 의존성)
- `favar` — Factor-Augmented VAR (statsmodels-based, core 의존성)

**Out of scope:**
- `state_space` / `TVP_AR` / `MIDAS` → Phase 05c (v2 classical expansion)
- `foundation_model_adapter` (e.g. TimesFM/Chronos) → post-v2 (Phase 11.12)
- `graph_neural_forecast` (GNN) → Phase 11.6
- Probabilistic/quantile 계열 (`mixture_density_network`, `BayesianNN`) → Phase 11.7

## 3. Sub-Tasks

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 05b.1 | Transformer (vanilla encoder) adapter | P2 | ~250 | `macrocast/execution/models/deep/transformer.py` (신규) | `test_transformer_adapter.py` green |
| 05b.2 | NBEATS adapter (torch-only) | P2 | ~300 | `macrocast/execution/models/deep/nbeats.py` (신규) | `test_nbeats_adapter.py` green |
| 05b.3 | TFT adapter via pytorch-forecasting | P2 | ~200 | `macrocast/execution/models/deep/tft.py` (신규) | `test_tft_adapter.py` green |
| 05b.4 | DFM adapter (statsmodels) | P2 | ~150 | `macrocast/execution/models/classical/dfm.py` (신규) | `test_dfm_adapter.py` green |
| 05b.5 | FAVAR adapter (statsmodels) | P2 | ~150 | `macrocast/execution/models/classical/favar.py` (신규) | `test_favar_adapter.py` green |
| 05b.6 | Registry entries + `requires_extra` metadata | P2 | ~50 | `macrocast/execution/models/registry.py` | model_registry schema 검증 green |
| 05b.7 | Per-model tests + sweep-safety | P2 | ~400 | `tests/test_*_adapter.py`, `tests/test_phase05b_sweep_safety.py` | 6 test files green |
| 05b.8 | Docs — model catalog expansion | P2 | ~150 | `docs/user/model_catalog.md` | RTD build green, 5 새 모델 항목 존재 |

## 4. API / Schema Specifications

Phase 5a에서 정립된 `DeepModelAdapter` / `ClassicalModelAdapter` 인터페이스를 그대로 따릅니다. 신규 API 없음.

Registry entry 형식 (발췌):
```python
# macrocast/execution/models/registry.py
"transformer": ModelSpec(
    family="deep",
    adapter="macrocast.execution.models.deep.transformer:TransformerAdapter",
    requires_extra="deep",            # [deep] optional dep 필요
    supports_multi_output=True,
),
"dfm": ModelSpec(
    family="classical",
    adapter="macrocast.execution.models.classical.dfm:DFMAdapter",
    requires_extra=None,              # core (statsmodels는 이미 core)
    supports_multi_output=True,
),
```

DFM / FAVAR는 `[deep]` extra 없이도 동작 — `requires_extra=None`.

## 5. File Layout

**신규:**
- `macrocast/execution/models/deep/transformer.py`
- `macrocast/execution/models/deep/nbeats.py`
- `macrocast/execution/models/deep/tft.py`
- `macrocast/execution/models/classical/dfm.py`
- `macrocast/execution/models/classical/favar.py`
- `tests/test_transformer_adapter.py`
- `tests/test_nbeats_adapter.py`
- `tests/test_tft_adapter.py`
- `tests/test_dfm_adapter.py`
- `tests/test_favar_adapter.py`
- `tests/test_phase05b_sweep_safety.py`

**수정:**
- `macrocast/execution/models/registry.py` (5 entry 추가)
- `pyproject.toml` — `[deep]` extra에 `pytorch-forecasting` 추가
- `docs/user/model_catalog.md`

## 6. Test Strategy

- Per-adapter smoke test: 소규모 recipe → predict → metrics 산출
- Sweep-safety: 동일 recipe 2회 실행 → 동일 metric (Phase 0 결정성 정책 준수)
- `requires_extra` 미설치 환경에서 명시적 error message 검증 (`RuntimeError("install macrocast[deep]")`)
- DFM/FAVAR는 `[deep]` 없는 환경에서도 green
- GPU 미사용 환경 (CPU-only CI) 에서 모든 test green

## 7. Acceptance Gate

- [ ] 5개 신규 모델 모두 registry에 등록됨
- [ ] 모든 adapter 단위 test green
- [ ] Phase 1 sweep runner에서 5개 모델을 variant로 섞어 실행 — 오염 없음
- [ ] DFM / FAVAR는 `pip install macrocast` (no extras) 환경에서 동작
- [ ] Transformer / NBEATS / TFT는 `[deep]` extra 없이 ImportError + 안내 메시지
- [ ] `docs/user/model_catalog.md`에 5개 모델 사용 예시

## 8. Docs Deliverables

- `docs/user/model_catalog.md` 확장: 각 모델별 (i) 추천 usage, (ii) 필수 hyperparam, (iii) 참고 논문
- ADR update 불필요 (ADR-004 deep-learning-optional 정책 내 포섭)

## 9. Migration Notes

- Breaking change 없음 — 추가만
- 기존 recipe는 그대로 동작
- 신규 `model_family` 값 5개가 schema validator에서 accept되도록 whitelist 확장

## 10. Cross-references

- ADR-004 (deep learning optional) — `requires_extra="deep"` 정책
- Phase 05a — DeepModelAdapter 인터페이스 baseline
- Phase 10.1 benchmark_suite — 신규 모델을 기본 suite에 편입

## 11. GitHub Issue Map

- Epic: (TBD, v1.1 kickoff 시 생성 — [PHASE-05b] v1.1 model expansion)
- Sub-task issues: (TBD, 8개 생성 예정)

## 12. Revision Log

- 2026-04-17: 초안 (v1.1 scope catalog에서 분리)

## 13. References

- Phase 05a 완료 보고서 (예정)
- `pytorch-forecasting` TFT 문서
- Stock & Watson (2002, 2016) DFM; Bernanke-Boivin-Eliasz (2005) FAVAR
