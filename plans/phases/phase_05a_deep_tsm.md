# Phase 05a — Deep & Time-Series Models (core + [deep] opt-in)

| Field | Value |
|-------|-------|
| Phase ID | phase-05a |
| Priority (inter-phase) | **P1** |
| Depends on | phase-01 |
| Unlocks | phase-07 (more model variance for replication) |
| Version tag target | v0.7 |
| Status | pending |

## 1. Goal

2026년 기준 forecasting package identity 구멍을 해소합니다. Core install에는 VAR/BVAR (statsmodels)를 추가해 multivariate baseline을 제공하고, `[deep]` optional install로 LSTM/GRU/TCN (PyTorch) 을 제공합니다. Phase 1 sweep runner가 이들 모델을 horse race에 포함시킬 수 있어야 하며, `[deep]` 미설치 환경에서는 명확한 `ExecutionError` 메시지로 실패해야 합니다.

## 2. Scope

**In scope:**
- 5개 모델 family: VAR, BVAR (core), LSTM, GRU, TCN ([deep] opt-in)
- Sequence adapter layer (`reshape_for_sequence`)
- `pyproject.toml` `[deep]` extra 정의
- `docs/conf.py` autodoc_mock_imports 확장 (torch, pytorch_lightning)
- `ci-deep.yml` GitHub Actions workflow 실제 구현
- `docs/install.md` 확장 ([deep] 섹션)
- 모델별 test + deep-missing-extra test + sweep-safety test

**Out of scope:**
- Transformer / NBEATS / TFT → Phase 5b
- state_space / TVP_AR / MIDAS → Phase 5c
- GPU multi-node / distributed training → Phase 11
- Hyperparameter tuning for deep models (기본 config만; Phase 5a는 모델 등록이 목표)
- AutoML / architecture search → post v1.0

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 05a.1 | VAR/BVAR (statsmodels, core) | P1 | ~400 | `macrocast/execution/models/tsm/{__init__,_base,var,bvar}.py` (신규), `macrocast/registry/training/model_family.py` (수정) | `test_model_var.py`, `test_model_bvar.py` green |
| 05a.2 | LSTM/GRU/TCN (PyTorch, [deep]) | P1 | ~800 | `macrocast/execution/models/deep/{__init__,_base,lstm,gru,tcn,_import_guard}.py` (신규) | deep extra installed 환경에서 end-to-end fit + predict |
| 05a.3 | Sequence adapter | P1 | ~150 | `macrocast/execution/adapters/sequence.py` (신규) | `test_sequence_adapter.py` green |
| 05a.4 | `pyproject.toml` `[deep]` extra | P1 | ~10 | `pyproject.toml` (수정) | `pip install .[deep]` 실제 작동 |
| 05a.5 | `docs/conf.py` autodoc mock 확장 | P1 | ~5 | `docs/conf.py` (수정) | core-only 환경에서 RTD build green |
| 05a.6 | `ci-deep.yml` GitHub Actions | P1 | ~80 | `.github/workflows/ci-deep.yml` (신규) | deep extra 설치 + `pytest -m deep` green |
| 05a.7 | Phase 5a tests | **P0** | ~700 | `tests/test_model_{var,bvar,lstm,gru,tcn}.py`, `tests/test_deep_models_sweep_safety.py`, `tests/test_deep_models_vs_baseline.py`, `tests/test_deep_missing_extra.py` | 전부 green (deep tests는 `pytest.mark.deep`) |
| 05a.8 | Phase 5a docs | P1 | ~400 | `docs/user_guide/model_catalog.md` (확장), `docs/install.md` ([deep] 섹션), `docs/api/models/deep.md` (신규), `docs/api/models/tsm.md` (신규) | RTD build green (deep autodoc은 mock import) |

## 4. API / Schema Specifications

### 4.1 Install-time 선택 정책 (Resolved Decision #2)

```bash
pip install macrocast          # core + VAR/BVAR + traditional models
pip install macrocast[deep]    # + torch + pytorch-lightning + LSTM/GRU/TCN
```

[deep] 미설치 상태에서 deep model 실행 시:
```python
raise ExecutionError(
    "model_family 'lstm' requires the [deep] extra. "
    "Install with: pip install macrocast[deep]"
)
```

### 4.2 `pyproject.toml` `[deep]` extra

```toml
[project.optional-dependencies]
deep = [
    "torch>=2.0",
    "pytorch-lightning>=2.0",
]
```

### 4.3 Deep model import guard

```python
# macrocast/execution/models/deep/_import_guard.py
from macrocast.errors import ExecutionError

def require_torch(model_family: str) -> "types.ModuleType":
    try:
        import torch
    except ImportError as exc:
        raise ExecutionError(
            f"model_family {model_family!r} requires the [deep] extra. "
            "Install with: pip install macrocast[deep]"
        ) from exc
    return torch

def require_lightning(model_family: str) -> "types.ModuleType":
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise ExecutionError(
            f"model_family {model_family!r} requires the [deep] extra. "
            "Install with: pip install macrocast[deep]"
        ) from exc
    return pl
```

### 4.4 Sequence adapter

```python
# macrocast/execution/adapters/sequence.py
import numpy as np

def reshape_for_sequence(
    *,
    X_train: np.ndarray,     # shape (T, n_features) flat time series
    y_train: np.ndarray,     # shape (T,) or (T, n_targets)
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat time series → (n_windows, lookback, n_features) + aligned y.

    Returns:
      X_seq: (n_windows, lookback, n_features)
      y_seq: (n_windows,) — y_train[t + lookback + horizon - 1]

    Windows with insufficient history or out-of-range horizon are dropped.
    """
```

### 4.5 VAR / BVAR registry entries

```python
# macrocast/registry/training/model_family.py (수정)
MODEL_FAMILY_ENTRIES = {
    # ... existing ...
    "var":  {"status": "operational", "requires_extra": None,   "impl": "tsm.var:VARModel"},
    "bvar": {"status": "operational", "requires_extra": None,   "impl": "tsm.bvar:BVARModel"},
    "lstm": {"status": "operational", "requires_extra": "deep", "impl": "deep.lstm:LSTMModel"},
    "gru":  {"status": "operational", "requires_extra": "deep", "impl": "deep.gru:GRUModel"},
    "tcn":  {"status": "operational", "requires_extra": "deep", "impl": "deep.tcn:TCNModel"},
}
```

BVAR default prior: Minnesota (statsmodels 사용; hyperparams 기본값 λ=0.2, own-lag tightness=1.0, cross-lag tightness=0.5).

### 4.6 Deep model base contract

```python
# macrocast/execution/models/deep/_base.py
from dataclasses import dataclass

@dataclass
class DeepModelConfig:
    lookback: int = 12
    hidden_size: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    max_epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 10
    seed: int = 42   # resolved by Phase 0 seed_policy
```

Phase 1 sweep runner는 `resolve_seed(..., model_family='lstm')`로 seed를 주입; 모델 내부에서 `torch.manual_seed(seed)` + `torch.use_deterministic_algorithms(True)` 호출.

## 5. File Layout

**신규:**
- `macrocast/execution/models/tsm/__init__.py`
- `macrocast/execution/models/tsm/_base.py`
- `macrocast/execution/models/tsm/var.py`
- `macrocast/execution/models/tsm/bvar.py`
- `macrocast/execution/models/deep/__init__.py`
- `macrocast/execution/models/deep/_base.py`
- `macrocast/execution/models/deep/_import_guard.py`
- `macrocast/execution/models/deep/lstm.py`
- `macrocast/execution/models/deep/gru.py`
- `macrocast/execution/models/deep/tcn.py`
- `macrocast/execution/adapters/sequence.py`
- `.github/workflows/ci-deep.yml`
- `tests/test_model_var.py`
- `tests/test_model_bvar.py`
- `tests/test_model_lstm.py`
- `tests/test_model_gru.py`
- `tests/test_model_tcn.py`
- `tests/test_sequence_adapter.py`
- `tests/test_deep_models_sweep_safety.py`
- `tests/test_deep_models_vs_baseline.py`
- `tests/test_deep_missing_extra.py`
- `docs/api/models/deep.md`
- `docs/api/models/tsm.md`

**수정:**
- `pyproject.toml` — `[deep]` extra 추가
- `docs/conf.py` — `autodoc_mock_imports += ["torch", "pytorch_lightning"]`
- `docs/install.md` — [deep] 섹션
- `docs/user_guide/model_catalog.md` — 5개 family 문서화
- `macrocast/registry/training/model_family.py` — 5개 entry 추가

## 6. Test Strategy

### `tests/test_model_var.py`, `tests/test_model_bvar.py`
- FRED-MD subset (INDPRO + CPIAUCSL + UNRATE) → 2-lag VAR fit → finite predictions
- BVAR Minnesota prior 기본 hyperparams → convergence
- Sweep runner 내 core extra 만으로 실행 가능 (torch import 없음 확인)

### `tests/test_model_{lstm,gru,tcn}.py` (pytest.mark.deep)
- 합성 AR(1) 데이터 → 학습 → RMSE < np.std(y) (sanity)
- 모델별 forward pass shape 검증
- [deep] 환경에서만 실행 (CI의 ci-deep.yml 담당)

### `tests/test_sequence_adapter.py`
- lookback=12, horizon=1 → 윈도우 개수 정확
- edge case: len(X) < lookback → 빈 배열
- y 정렬: X_seq[i]의 마지막 t = y_train[t + lookback + horizon - 1]

### `tests/test_deep_models_sweep_safety.py` (pytest.mark.deep)
- Phase 1 sweep runner로 3 variant (lstm, gru, tcn) 동시 실행
- 동일 seed에서 두 번 실행 → predictions.csv byte-identical
- `torch.manual_seed` + `torch.use_deterministic_algorithms(True)` 호출 확인

### `tests/test_deep_models_vs_baseline.py` (pytest.mark.deep)
- FRED-MD INDPRO 1-step forecast: LSTM의 RMSE < historical_mean의 RMSE
- Slow test (marker: `slow`) — CI의 deep workflow에서만 실행

### `tests/test_deep_missing_extra.py`
- `importlib` patch로 `torch` ImportError 모의
- `execute_recipe(model_family='lstm')` → ExecutionError with "pip install macrocast[deep]" substring

### `.github/workflows/ci-deep.yml`
- Trigger: push to main, PR to main, weekly schedule
- Matrix: python-3.11 × {ubuntu-latest}
- Steps: `pip install .[deep]` → `pytest -m deep --timeout=600`

## 7. Acceptance Gate

- [ ] Phase 1 gate 선통과
- [ ] `pip install macrocast` (core only) → VAR/BVAR 사용 가능, deep model 시도시 명확한 ExecutionError
- [ ] `pip install macrocast[deep]` → LSTM/GRU/TCN end-to-end fit/predict 가능
- [ ] Phase 1 sweep runner로 5 variant (ridge + var + bvar + lstm + gru) 실행 성공 ([deep] 환경)
- [ ] 기존 + Phase 0/1 + Phase 5a core test 전부 green
- [ ] `ci-deep.yml` workflow green
- [ ] `docs/api/models/deep.md` RTD build green (core-only 환경에서 autodoc_mock_imports 로 렌더)
- [ ] `docs/install.md`에 `[deep]` 섹션 존재

## 8. Docs Deliverables

**신규:**
- `docs/api/models/deep.md` — LSTM/GRU/TCN autodoc (mock import 사용)
- `docs/api/models/tsm.md` — VAR/BVAR autodoc

**확장:**
- `docs/install.md` — `[deep]` extra 설치 방법 + torch 버전 호환 매트릭스
- `docs/user_guide/model_catalog.md` — 5개 신규 family 설명 + 권장 hyperparams + sweep 예시

## 9. Migration Notes

- 기존 recipe 중 `model_family ∈ {var, bvar, lstm, gru, tcn}` 쓰던 사용자는 없음 (신규 도입)
- `pyproject.toml` extras 변경: **추가만** (기존 extras 유지)
- 기존 user가 `pip install macrocast[deep]` 로 재설치해야 deep models 활성화 (docs/install.md에 마이그레이션 가이드)
- Core-only 환경 RTD 빌드 무결성 확보 위해 `autodoc_mock_imports` 필수

## 10. Cross-references

- Infra files used: `plans/infra/seed_policy.md` (deep seed 주입), `plans/infra/optional_deps_pattern.md`
- Phase dependencies:
  - Phase 0 seed_policy → deep model seed 결정성
  - Phase 1 sweep runner → deep model을 sweep variant로 사용
  - Phase 7 paper replication → VAR / ARDI 등이 paper baseline으로 활용
- ADRs referenced:
  - ADR-004 (deep learning is optional extra, not core dep)
  - ADR-006 (PyTorch Lightning over raw PyTorch for training loop)
- Coverage Ledger rows resolved:
  - Layer 3 `model_family = var` → operational
  - Layer 3 `model_family = bvar` → operational
  - Layer 3 `model_family = lstm` → operational (requires [deep])
  - Layer 3 `model_family = gru` → operational (requires [deep])
  - Layer 3 `model_family = tcn` → operational (requires [deep])

## 11. GitHub Issue Map

- Epic: (TBD at kickoff — [PHASE-05a] Deep & time-series models)
- Sub-task issues: 8개 (05a.1 ~ 05a.8)

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 §Phase 5a에서 추출)
