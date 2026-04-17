# Phase 08 — paper_ready_bundle + Multi-Run Aggregation

| Field | Value |
|-------|-------|
| Phase ID | phase-08 |
| Priority (inter-phase) | **P0** |
| Depends on | phase-07 |
| Unlocks | phase-09 |
| Version tag target | v0.9 (with phase-07) |
| Status | pending |

## 1. Goal

"study 돌렸다" → "reviewer-defensible artifact bundle" closure. `PaperReadyBundle`은 horse race study와 decomposition 결과를 단일 디렉터리(table/figure/data/manifest/README)로 emit하여 논문 supplement에 그대로 첨부 가능하게 합니다. `ranking_rule` executor + regime_evaluation runner 포함. Phase 6에서 정의한 `wrapper_handoff`의 실제 consumer가 이 Phase에 구현됩니다.

## 2. Scope

**In scope:**
- `PaperReadyBundle` core API (one-shot emit, deterministic hash)
- `ranking_rule` executor (7 rule 모두 deterministic)
- LaTeX table generator (booktabs 스타일)
- Regime evaluation aggregation runner (study-level)
- `wrapper_handoff` payload consumer
- README 자동 생성 (how-to-cite)

**Out of scope:**
- Interactive dashboard — Phase 11
- Main 3 외 figure (forest / waterfall / heatmap 이외) — v1.1
- `paper_style` 옵션 중 ICML / NeurIPS / JAE style — v1.1 (v1.0은 generic booktabs만)

## 3. Sub-Tasks (GitHub issue source-of-truth)

| ID | Sub-task | Priority | Est LOC | Files | Gate |
|:---:|---------|:--------:|:-------:|-------|------|
| 08.1 | `PaperReadyBundle` core API | **P0** | ~400 | `macrocast/output/bundle.py` (신규) | `test_bundle_deterministic.py` green |
| 08.2 | Deterministic bundle hash | **P0** | ~60 | `macrocast/output/bundle.py` (hash util) | 두 번 emit → byte-identical |
| 08.3 | Ranking rule executor | P1 | ~250 | `macrocast/output/ranking.py` (신규) | `test_ranking_stability.py` green |
| 08.4 | LaTeX table generator | P1 | ~300 | `macrocast/output/latex_tables.py` (신규) | `test_latex_compiles.py` green (pdflatex 있으면) |
| 08.5 | Regime evaluation aggregation | P1 | ~250 | `macrocast/execution/evaluation/regime.py` (신규) | `test_regime_aggregation.py` green |
| 08.6 | `wrapper_handoff` consumer wire | P2 | ~100 | `macrocast/output/bundle.py` (consumer), `macrocast/execution/sweep_runner.py` (produce) | orchestrated_bundle_study E2E |
| 08.7 | Phase 8 tests | **P0** | ~500 | `tests/test_bundle_deterministic.py`, `tests/test_ranking_stability.py`, `tests/test_latex_compiles.py`, `tests/test_regime_aggregation.py` | 4개 test 전부 green |
| 08.8 | Phase 8 docs | P1 | ~350 | `docs/user_guide/paper_ready_bundle.md`, `docs/examples/v1_flagship_transformer_horse_race.md`, `docs/api/bundle.md` | RTD build green |

## 4. API / Schema Specifications

### 4.1 `PaperReadyBundle` core API

```python
# macrocast/output/bundle.py
from dataclasses import dataclass

@dataclass(frozen=True)
class BundleSpec:
    include_tables: tuple[str, ...] = ("main_comparison", "decomposition_breakdown", "regime_analysis")
    include_figures: tuple[str, ...] = ("variant_forest_plot", "decomposition_waterfall", "regime_heatmap")
    ranking_rule: str = "mean_metric_rank"
    paper_style: str = "generic_booktabs"    # v1.0에서 유일한 허용 값
    citation_key: str | None = None

@dataclass(frozen=True)
class PaperReadyBundle:
    bundle_dir: str
    bundle_hash: str
    manifest_path: str

def emit_paper_ready_bundle(
    *,
    study_manifest_paths: list[str],
    decomposition_result_paths: list[str],
    output_dir: str | Path,
    bundle_spec: BundleSpec,
) -> PaperReadyBundle:
    """One-shot bundle.

    Output layout:
    output_dir/bundle/
      tables/main_comparison.tex
      tables/decomposition_breakdown.tex
      tables/regime_analysis.tex
      figures/variant_forest_plot.png
      figures/decomposition_waterfall.png
      figures/regime_heatmap.png
      data/predictions_all_variants.parquet
      data/metrics_long.parquet
      bundle_manifest.json
      README.md   (auto-generated, how-to-cite)
    """
```

### 4.2 Deterministic bundle hash

- `bundle_hash = sha256(sorted(file_path, file_sha256) canonical JSON)`
- 동일 input (study_manifest + decomposition_result + spec) → byte-identical 모든 파일
- PNG 결정성: matplotlib `savefig(..., metadata={"Software": None, "Creation Time": None})`

### 4.3 Ranking rule executor

```python
# macrocast/output/ranking.py
RANKING_RULES = (
    "mean_metric_rank",
    "win_count",
    "mcs_inclusion_priority",
    "dm_test_vs_benchmark",
    "cw_test_vs_benchmark",
    "regime_weighted_rank",
    "decomposition_share_weighted",
)

def rank_variants(
    *,
    study_manifest: dict,
    ranking_rule: str,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Return DataFrame[variant_id, rank, score, tiebreaker]. Stable ordering.

    Invariant: 같은 input → 항상 동일 rank 순서 (tie-break까지 deterministic).
    """
```

### 4.4 LaTeX table generator

```python
# macrocast/output/latex_tables.py
def emit_main_comparison_table(
    *, study_manifest: dict, output_path: str | Path,
    significance_markers: bool = True,
) -> str:
    """booktabs 스타일. \\label{tab:main_comparison} 자동 부착.

    significance markers: *** p<0.01, ** p<0.05, * p<0.10 (DM test vs benchmark).
    """

def emit_decomposition_breakdown_table(*, decomposition_result: dict, output_path) -> str: ...
def emit_regime_analysis_table(*, study_regime_summary: dict, output_path) -> str: ...
```

### 4.5 Regime evaluation aggregation

```python
# macrocast/execution/evaluation/regime.py
def aggregate_regime_metrics(
    *, study_manifest_path: str, output_path: str | Path
) -> "StudyRegimeSummary":
    """Phase 3의 regime_definition 축 consumer.

    Regime sources:
    - NBER recession dates
    - Volatility regime (rolling vol quantile)
    - User-specified break dates (from recipe)

    Output (study_regime_summary.json):
      { "regimes": ["nber_recession", "nber_expansion", ...],
        "per_variant_per_regime": {
          "v-a1b2c3d4": {"nber_recession": {"msfe": 0.031, "n_obs": 42}, ...}
        } }
    """
```

기존 per-variant `regime_summary.json`은 Phase 3에서 생성됨. Phase 8은 study-level aggregation만 추가.

## 5. File Layout

**신규:**
- `macrocast/output/__init__.py`
- `macrocast/output/bundle.py`
- `macrocast/output/ranking.py`
- `macrocast/output/latex_tables.py`
- `macrocast/execution/evaluation/regime.py`
- `tests/test_bundle_deterministic.py`
- `tests/test_ranking_stability.py`
- `tests/test_latex_compiles.py`
- `tests/test_regime_aggregation.py`
- `docs/user_guide/paper_ready_bundle.md`
- `docs/examples/v1_flagship_transformer_horse_race.md`
- `docs/api/bundle.md`

**수정:**
- `macrocast/execution/sweep_runner.py` — `wrapper_handoff` payload produce
- `macrocast/__init__.py` — 공개 API (`emit_paper_ready_bundle`, `BundleSpec`, `PaperReadyBundle`, `rank_variants`, `aggregate_regime_metrics`)

## 6. Test Strategy

### `tests/test_bundle_deterministic.py`
- 같은 study + decomposition → `emit_paper_ready_bundle` 두 번 → bundle_hash 동일 + 모든 파일 sha256 일치
- 다른 `bundle_spec` → 다른 bundle_hash

### `tests/test_ranking_stability.py`
- 7개 `RANKING_RULES` 각각 호출 → 같은 input에서 항상 동일 rank 순서
- 완전 tie 상황에서도 deterministic (lexicographic tie-break)

### `tests/test_latex_compiles.py`
- CI에 `pdflatex` 있으면 `pdflatex -interaction=batchmode main_comparison.tex` 실행 → exit 0
- 없으면 `pytest.skip` (tex syntax validation만 수행)

### `tests/test_regime_aggregation.py`
- 3-variant × 2-regime synthetic study → `study_regime_summary.json`의 per_variant_per_regime 집계 정확
- 비어있는 regime (0 obs) → 해당 regime 항목 NaN 처리

## 7. Acceptance Gate

- [ ] Phase 7 gate 선통과
- [ ] `emit_paper_ready_bundle()` 동일 input → byte-identical bundle
- [ ] 7 ranking_rule 전부 deterministic
- [ ] 3 main figure (forest / waterfall / heatmap) + 3 main table 생성
- [ ] `study_regime_summary.json`이 schema 준수 (NBER / volatility / user-break 3 source)
- [ ] `wrapper_handoff` payload가 orchestrated_bundle_study에서 `emit_paper_ready_bundle`로 흐름
- [ ] 공개 API export: `emit_paper_ready_bundle`, `BundleSpec`, `PaperReadyBundle`, `rank_variants`, `aggregate_regime_metrics`
- [ ] Phase 8 docs 3종 RTD latest build green
- [ ] 기존 test + Phase 0-7 test + Phase 8 신규 test 전부 green

## 8. Docs Deliverables

**신규:**
- `docs/user_guide/paper_ready_bundle.md` — end-to-end tutorial (sweep → decomposition → bundle)
- `docs/examples/v1_flagship_transformer_horse_race.md` — v1.1 LSTM vs baselines 시연 (문서는 v1.0 포함)
- `docs/api/bundle.md` — autodoc

## 9. Migration Notes

- 이전 `emit_publication_kit()` (있었다면) 은 `emit_paper_ready_bundle()`로 대체
- Breaking: 기존 스크립트가 `publication_kit/` 디렉터리를 참조했다면 `bundle/`로 수정 필요
- `regime_summary.json` (per-variant) 기존 유지, 신규 `study_regime_summary.json` (aggregated) 추가
- wrapper_handoff payload는 Phase 6에서 정의, Phase 8에서 consume — backward-compat

## 10. Cross-references

- Infra files used: `plans/infra/wrapper_handoff_schema.md`, `plans/infra/bundle_manifest_schema.md`
- ADRs referenced: ADR-006 (ranking rule determinism), ADR-007 (PNG reproducibility)
- Coverage Ledger rows resolved:
  - "paper_ready_bundle closure" → operational
  - `ranking_rule` 7종 → operational
  - `regime_definition` (study-level aggregation) → operational
  - `wrapper_handoff` consumer → operational
- Downstream: Phase 9 v1.0 release docs에서 이 bundle을 flagship artifact로 참조

## 11. GitHub Issue Map

- Epic: (TBD at kickoff — [PHASE-08] paper_ready_bundle + multi-run aggregation)
- Sub-task issues: 8개 (08.1~08.8)

## 12. Revision Log

- 2026-04-17: 초안 (ultraplan v2.2 §Phase 8에서 추출)
