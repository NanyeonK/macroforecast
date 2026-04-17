# Phase 11 — v2 Scope Catalog (Distributed, Exotic Data, R-Parity, Dashboard, and Everything Else)

| Field | Value |
|-------|-------|
| Phase ID | phase-11 |
| Priority (inter-phase) | **P2** |
| Depends on | v1.1 complete (phase-10) |
| Unlocks | (terminal — v2 is the current horizon) |
| Version tag target | v2.0 |
| Status | deferred (post-v1.1 release) |

## 1. Goal

v2 scope catalog — identity 확장 (R-parity, dashboard)과 scale 확장 (distributed, exotic data, probabilistic), 그리고 user-specified 지만 v1.0/v1.1에서 defer된 "everything else"의 수용처. 각 deliverable은 phase-sized 이며, 본 문서는 **catalog 수준의 경계와 순서만** 정의.

## 2. Scope

**In scope (13 deliverables):**
- 11.1 Distributed compute (Ray / Dask / Slurm)
- 11.2 Exotic data sources (trends / news / satellite / custom SQL)
- 11.3 Deep-integration subscriptions (WRDS / BIS / Census)
- 11.4 R companion parity (macrocastR)
- 11.5 Interactive dashboard (Streamlit / Dash)
- 11.6 Panel / spatial / hierarchical models (GNN 포함)
- 11.7 Probabilistic / quantile / density models
- 11.8 Gradient / path importance methods
- 11.9 Economic metrics runtime
- 11.10 Density / interval tests runtime
- 11.11 GPU multi / distributed_cluster (11.1과 연계되는 backend 레이어)
- 11.12 Foundation model / LLM adapter
- 11.13 Docs full-scale update

**Out of scope (v2 이후 — v3+):**
- Regime-switching family 완전체
- Real-time streaming ingestion
- Mobile/embedded deployment
- 본 catalog에 없는 user requests는 v2.1+ backlog로

## 3. Sub-Tasks (catalog form — each is a phase-sized item)

| ID | Deliverable | Priority | Rough scope |
|:---:|---------|:--------:|-------------|
| 11.1 | Distributed compute | P2 | Ray / Dask / Slurm adapter; `compute_mode = distributed_cluster` operational |
| 11.2 | Exotic data sources | P2 | Blue_chip, market_prices, high_freq_surprises, Google_trends, news_text, Climate_series, satellite_proxy, Custom_duckdb, Custom_sql |
| 11.3 | Deep integration | P2 | WRDS (institutional sub), BIS (varying terms), Census (public domain, business dynamics) |
| 11.4 | R companion parity | P2 | macrocastR이 Phase 5 신규 모델 + Phase 7 decomposition mirror |
| 11.5 | Interactive dashboard | P2 | `macrocast dashboard` CLI → local Streamlit/Dash app — sweep study 탐색 + decomposition 시각화 |
| 11.6 | Panel / Spatial / Hierarchical | P2 | panel_FE/RE, dynamic_panel, spatial_AR, spatial_Durbin, graph_neural_forecast (GNN via PyG), hierarchical_reconciliation_model, cross_state_factor_model |
| 11.7 | Probabilistic / quantile / density | P2 | quantile_RF/GBM/XGB/LSTM, mixture_density_network, BayesianNN, distributional_regression, conformal_wrapper |
| 11.8 | Gradient / path importance | P2 | IntegratedGradients, PathIntegratedGradients, GradientXInput, SmoothGrad, ExpectedGradients, DeepLift, LRP, saliency_map |
| 11.9 | Economic metrics runtime | P2 | utility_gain, certainty_equivalent, portfolio_SR, cost_sensitive_loss, policy_loss |
| 11.10 | Density / interval tests runtime | P2 | PIT_uniformity (Phase 2 등록만 됨 → runtime), Berkowitz, Kupiec, Christoffersen 3종, interval_coverage_test |
| 11.11 | GPU multi / distributed_cluster backend | P2 | 11.1과 연계, CUDA multi-device orchestration |
| 11.12 | Foundation model / LLM adapter | P2 | TimesFM / Chronos / Moirai 등 foundation model wrapper |
| 11.13 | Docs full-scale update | P2 | v2 전체 surface의 user guide + API ref |

각 deliverable은 post-v1.1 kickoff 시 별도 phase 문서로 split out. 대략 phase-11.1 ~ phase-11.13 형식.

## 4. API / Schema Specifications

Catalog 수준 — 구체 API는 각 sub-phase에서 확정. 본 문서에서는 3개 landmark만:

### 4.1 Distributed compute mode (11.1)

```yaml
sweep:
  compute:
    compute_mode: distributed_cluster
    backend: ray          # or dask / slurm
    n_workers: 32
    scheduler_uri: "ray://head:10001"
```

### 4.2 Dashboard CLI (11.5)

```bash
macrocast dashboard --study-root ./sweep_out --port 8501
# Streamlit app 기동, 브라우저에서 study 탐색 + decomposition plot
```

### 4.3 Foundation model adapter (11.12)

```yaml
model:
  family: foundation_model
  spec:
    backbone: chronos-t5-large   # or timesfm / moirai
    mode: zero_shot              # zero_shot | finetune | few_shot
    context_length: 512
```

## 5. File Layout

Catalog 수준. 주요 신규 directory:
- `macrocast/distributed/` (11.1)
- `macrocast/raw/datasets/` 대폭 확장 (11.2, 11.3)
- `macrocastR/` — 별도 R 패키지 repo 혹은 subdirectory (11.4)
- `macrocast/dashboard/` (11.5)
- `macrocast/execution/models/panel/`, `.../spatial/`, `.../gnn/`, `.../hierarchical/` (11.6)
- `macrocast/execution/models/probabilistic/` (11.7)
- `macrocast/explain/gradient/` (11.8)
- `macrocast/metrics/economic_runtime.py` (11.9)
- `macrocast/metrics/density_tests_runtime.py` (11.10)
- `macrocast/execution/models/foundation/` (11.12)

## 6. Test Strategy

- Per-deliverable test suite — 각 sub-phase에서 정의
- 11.1 distributed: CI는 local Ray/Dask mock; 실 cluster test는 integration bench (nightly)
- 11.2/11.3 data adapters: 실 API 키 없는 CI에서는 fixture로; live test는 manual gate
- 11.4 R-parity: R CMD check + R vs Python numeric parity tolerance test
- 11.5 dashboard: headless Streamlit component test
- 11.6-11.12 model/importance adapter: Phase 05 와 동일 sweep-safety 패턴

## 7. Acceptance Gate

- [ ] 11.1-11.13 각 deliverable에 독립 phase 문서 존재
- [ ] v1.1 기존 test suite 회귀 zero
- [ ] 11.9 economic metrics + 11.10 density/interval tests가 Phase 2에 등록된 항목을 runtime으로 전환 완료 (Coverage Ledger 0 → runtime green)
- [ ] 11.4 R-parity numeric tolerance 1e-6 (RMSE 등 주요 metric)
- [ ] v2 릴리스 태그 cut 가능 상태
- [ ] docs 전체 surface coverage

## 8. Docs Deliverables

- 13개 deliverable 각자 user guide + API ref
- Migration guide v1.1 → v2 — breaking change 있을 경우 명시
- 튜토리얼 재구성 — v2 풀 surface 기준

## 9. Migration Notes

- Major version bump (v1 → v2) — breaking change 허용되나 최소화 목표
- 기대되는 break 지점:
  - `compute_mode` enum 확장 (기존 값은 그대로 동작)
  - Registry schema 확장 (하위 호환)
  - CLI 그룹 재정리 (alias로 호환 유지)
- Deprecation path: v1.x 에서 warning → v2.0 에서 제거

## 10. Cross-references

- v1.0/v1.1 foundation 전체 (Phase 0-10)
- 본 phase는 ultraplan v2.2 에서 "post-v2" 꼬리표가 붙지 않은 모든 user-specified 항목의 수용처
- 개별 항목들은 post-v1.1 kickoff 시 phase-11.N 문서로 split out

## 11. GitHub Issue Map

- Epic: (TBD, v1.1 릴리스 직후 생성 — [PHASE-11] v2 scope catalog)
- Sub-epics: 11.1 ~ 11.13 각자 독립 epic
- Sub-task issues: 각 sub-epic 내부에서 분해

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2에서 v2 scope 추출)

## 13. References

- ultraplan v2.2 — v2 scope 섹션
- v1.1 post-release user feedback (TBD)
- Coverage Ledger — 11.9/11.10 항목의 registry_only 상태 tracking
