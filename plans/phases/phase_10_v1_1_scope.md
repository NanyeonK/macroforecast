# Phase 10 — v1.1 Scope Catalog (Benchmark Suite, Multi-Target Joint, Data Expansion, Importance Uplift)

| Field | Value |
|-------|-------|
| Phase ID | phase-10 |
| Priority (inter-phase) | **P2** |
| Depends on | v1.0 complete (phase-09) |
| Unlocks | phase-11 (v2 scope) |
| Version tag target | v1.1 |
| Status | deferred (post-v1.0 release) |

## 1. Goal

v1.0 출시 후 user demand 기반으로 우선순위가 붙는 v1.1 scope catalog. 각 deliverable은 독립적인 phase-sized 작업이며, 본 문서는 **scope boundary + 순서 + cross-phase 연결만** 정의합니다. 세부 계획은 v1.0 kickoff 이후 각 deliverable별 sub-phase 문서로 분리됩니다.

## 2. Scope

**In scope (8 deliverables):**
- 10.1 Benchmark Suite Runner
- 10.2 Multi-target joint runner
- 10.3 Extended data sources (BEA/BLS/OECD/ECB/IMF/SPF)
- 10.4 Importance uplift (SHAP unification + DL temporal importance)
- 10.5 Shapley attribution (Phase 7 enhancement)
- 10.6 Conditional / nested / derived sweep axes (Phase 1 extension)
- 10.7 Docs update
- 10.8 Statistical test catalog expansion (Phase 2 후속 — 19 new tests + test_scope 확장)

**Out of scope:**
- Distributed compute → Phase 11.1
- Exotic data (trends/news/satellite/WRDS/BIS) → Phase 11.2/11.3
- R-parity → Phase 11.4
- Dashboard → Phase 11.5
- Panel/spatial/GNN → Phase 11.6
- Probabilistic/quantile/density models → Phase 11.7
- Gradient/path importance (IG, SmoothGrad, LRP) → Phase 11.8
- Foundation model adapter → Phase 11.12

## 3. Sub-Tasks (catalog form — each is a phase-sized item)

| ID | Deliverable | Priority | Rough scope | Primary files |
|:---:|---------|:--------:|-------------|---------------|
| 10.1 | Benchmark Suite Runner | P2 | curated recipe suite + one-command execution + comparative bundle | `macrocast/studies/benchmark_suite.py` (신규) |
| 10.2 | Multi-target joint runner | P2 | `multi_target_shared_design`, `multi_output_joint_model`, `hierarchical_forecasting_run` operational | `macrocast/execution/multi_target.py` (신규) |
| 10.3 | Extended data sources | P2 | BEA, BLS, OECD, ECB, IMF, SPF adapters 각 신규 | `macrocast/raw/datasets/{bea,bls,oecd,ecb,imf,spf}.py` (6개 신규) |
| 10.4 | Importance uplift | P2 | SHAP family unification (tree/kernel/linear/deep) + DL sequence/temporal importance + cross-model consensus | `macrocast/explain/shap_unified.py` (신규), `macrocast/explain/temporal_importance.py` (신규) |
| 10.5 | Shapley attribution | P2 | Phase 7 ANOVA baseline에 Shapley 추가 | `macrocast/decompose/shapley_attribution.py` (신규) |
| 10.6 | Conditional / nested / derived sweep axes | P2 | `axis_type` 3종 operational | `macrocast/sweep/axes.py` (확장) |
| 10.7 | Docs | P2 | user guide + API ref for 10.1-10.6, 10.8 | `docs/user/**`, `docs/api/**` |
| 10.8 | Statistical test catalog expansion | P2 | Phase 2 §4.1 표의 미구현 19개 검정 + test_scope 확장 구현 — equal_predictive(+2: paired_t_on_loss_diff, wilcoxon_signed_rank), nested(+1: forecast_encompassing_nested), cpa_instability(+3: fluctuation_test, chow_break_forecast, cusum_on_loss), multiple_model(+2: stepwise_mcs, bootstrap_best_model), density_interval(+7: PIT_uniformity, berkowitz, kupiec, christoffersen_{unconditional,independence,conditional}, interval_coverage), direction(+2: mcnemar, roc_comparison), residual_diagnostics(+2: autocorrelation_of_errors, serial_dependence_loss_diff), test_scope(+4: full_grid_pairwise, benchmark_vs_all, regime_specific_tests, subsample_tests) | `macrocast/execution/stat_tests/{density,direction,residual,cpa_ext,multiple_ext,equal_ext,scope_ext}.py` (신규) |

각 deliverable의 세부 sub-tasks는 post-v1.0 kickoff에서 별도 phase 문서로 split out.

## 4. API / Schema Specifications

본 phase는 catalog 수준 — 구체 API spec은 deliverable별 sub-phase 문서에서 확정. 다음 3개만 본 문서에서 선확정:

### 4.1 Benchmark Suite Runner CLI

```bash
macrocast benchmark-suite run --suite clss_v1_1 --output-root ./bench_out
macrocast benchmark-suite list  # 사용 가능한 suite 목록
```

### 4.2 Sweep axis_type 확장 (10.6)

```yaml
sweep:
  axes:
    - axis_type: nested_sweep      # 부모 axis 값에 따라 자식 axis 범위 달라짐
      parent: model_family
      children:
        randomforest: {n_estimators: [100, 200, 500]}
        xgboost: {n_estimators: [100, 500, 1000]}
    - axis_type: conditional        # predicate 만족 시만 해당 variant 실행
      condition: "target == 'inflation'"
      values: [...]
    - axis_type: derived            # 다른 axis에서 computed
      formula: "learning_rate = 0.1 / sqrt(n_estimators)"
```

### 4.3 Multi-target joint API

```python
from macrocast.execution.multi_target import execute_multi_target
result = execute_multi_target(
    recipe=recipe,
    targets=["inflation", "unemployment"],
    mode="joint",  # "joint" | "shared_design" | "hierarchical"
    ...
)
```

## 5. File Layout

Catalog 수준이라 file list는 deliverable별 요약 (§3 참조). 상세 layout은 sub-phase 문서에서 확정.

## 6. Test Strategy

- 각 deliverable별 독립 test suite
- v1.0 기존 291+ test 회귀 없음이 최우선 invariant
- 10.3 data source adapter: mocked HTTP + goldfile test (실 API 키 없이도 CI green)
- 10.1 benchmark suite: small-N smoke (suite의 축소판)

## 7. Acceptance Gate

- [ ] 10.1-10.6, 10.8 각 deliverable에 독립 phase 문서 존재 + 각자 acceptance gate 통과
- [ ] v1.0 기존 test suite 회귀 zero
- [ ] 10.3 6개 데이터 소스가 `registry_only` → `operational` 상태 전환
- [ ] 10.7 docs가 위 기능 전체를 cover
- [ ] v1.1 릴리스 태그 cut 가능 상태

## 8. Docs Deliverables

- `docs/user/benchmark_suite.md` (신규)
- `docs/user/multi_target.md` (신규)
- `docs/user/data_sources.md` 확장 (6개 소스 섹션 추가)
- `docs/user/importance_advanced.md` (신규)
- `docs/user/sweep_axes_advanced.md` (신규)
- `docs/api/**` 자동 생성 확장

## 9. Migration Notes

- Breaking change 없음 목표 — 추가 기능만
- 10.6의 sweep axis 확장은 기존 axis 정의와 호환 (new `axis_type` opt-in)
- 10.2 multi-target은 기존 single-target recipe와 독립

## 10. Cross-references

- Phase 0-9 전체가 foundation (v1.0 complete)
- Phase 05b (모델 catalog 확장) — 10.1 benchmark suite가 활용
- Phase 11 (v2) — 본 phase에서 명시적으로 defer한 항목들의 수용처

## 11. GitHub Issue Map

- Epic: (TBD, v1.0 릴리스 직후 생성 — [PHASE-10] v1.1 scope catalog)
- Sub-epics: 10.1 ~ 10.7 각자 epic 분리
- Sub-task issues: 각 sub-epic 내부에서 분해

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2에서 v1.1 scope 추출)
- 2026-04-17: 10.8 추가 — Phase 2 §4.1에서 의도됐지만 Phase 2 scope(재분류 only) 밖이던 19개 신규 검정 + test_scope 확장 항목을 v1.1 카탈로그에 명시적으로 편입

## 13. References

- ultraplan v2.2 — v1.1 scope 섹션
- v1.0 post-release user feedback 창구 (TBD)
