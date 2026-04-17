# ADR-004: Deep Learning Dependencies are Optional (Install-Time Choice)

Status: **Accepted** (2026-04-17)
Date: 2026-04-17
Context-phase: phase-05a

## Context

Phase 5a 신규 모델 중 LSTM/GRU/TCN은 PyTorch 필요 (~800MB). macrocast 사용자 상당수는 traditional ML 모델(ridge/randomforest 등)만 씀. **선택지:**
- (A) torch를 core dependency로
- (B) torch를 optional `[deep]` extra로 (install-time 선택)
- (C) torch 완전 제외 (deep model registry 미포함)

## Decision

**(B) `[deep]` optional extra.**

```bash
pip install macrocast          # core only (VAR/BVAR included, no torch)
pip install macrocast[deep]    # + torch + LSTM/GRU/TCN
pip install macrocast[all]     # all extras
```

**Runtime behavior:**
- `[deep]` 미설치 상태에서 deep model recipe execute → `ExecutionError("model_family 'lstm' requires pip install macrocast[deep]")`
- Registry entry는 항상 parseable (recipe YAML validate됨)
- Inspection API (`macrocast_single_run`)은 `[deep]` 없이도 작동 (planned/registry_only로 보고)

## Consequences

**+** Core 설치 ~100MB 유지 (vs torch 포함시 ~1GB)
**+** Macro researchers가 굳이 deep 안 써도 cold start 빠름
**+** RTD 문서 빌드는 `autodoc_mock_imports`로 torch 없이도 성공
**+** User 결정 권한 명시적 (Resolved Decision #2)
**−** CI matrix 복잡 (ci-core.yml + ci-deep.yml 분리)
**−** `_import_guard.py` boilerplate 필요
**−** Error message UX 신경써야 함 (명확한 install 안내)

## References

- Used by: phase-05a (implementation), phase-05b (Transformer/NBEATS/TFT도 동일 패턴), all CI (ci-deep.yml)
- Resolved Decision #2: user 결정 "사용자가 install 시점에 선택"
- Pattern: scikit-learn optional deps (xgboost, lightgbm)와 동일 패턴 (이미 pyproject.toml에 존재)
- Phase 10.1 benchmark_suite는 `[deep]` 없이 실행되는 suite와 포함 suite 둘 다 제공
