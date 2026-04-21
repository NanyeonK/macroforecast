# 1.4 Benchmark & Predictor Universe — cleanup + implementation plan

**Goal:** make Layer 1 1.4 honest. After an audit through "can this value be implemented via a recipe/leaf_config input channel?" many values that looked drop-worthy turn out to be implementable. This plan splits the work into (a) a cleanup PR that drops 4 values needing new infrastructure (NN/text embeddings, dynamic CV feature selection) and honest-demotes 9 metadata-only operational labels, and (b) an implementation PR that wires 19 values back to operational via simple recipe inputs + deterministic feature additions.

**Starting state:** 130 axes, 717 tests. 1.4 has 4 axes / 37 values (23 op + 12 registry_only + 2 future).
**After cleanup PR:** 1.4 has 4 axes / 33 values (14 op + 19 registry_only). Registry_only entries are the v1.0 impl roadmap.
**After implementation PR:** 1.4 fully operational — 4 axes, 33 values, every one either operational or dropped.

---

## 1. Audit findings

| axis | op (before) | truly wired | metadata-only "op" | notes |
|---|---|---|---|---|
| `benchmark_family` | 11 | 9 | 2 | factor_model + multi_benchmark_suite currently fall back to `historical_mean` |
| `predictor_family` | 5 | 2 | 3 | target_lags_only + all_macro_vars have compiler guards; all_except_target + category_based + factor_only have zero value-level dispatch |
| `variable_universe` | 4 | 2 | 2 | all_variables (default) + preselected_core (real column filter) are wired; category_subset + paper_replication_subset flow through as no-op |
| `deterministic_components` | 3 | 1 | 2 | `none` default; constant_only + linear_trend never modify X |

Total: 14 wired, 9 metadata-only "operational" label-lies.

---

## 2. Cleanup PR (this one)

### 2.1 Drops (4 values)

Truly unrealistic for v1.0 — require new infrastructure not in scope:

- `predictor_family.text_only` — text embeddings / NN domain (v2 Transformer stack).
- `predictor_family.mixed_blocks` — multi-block NN architecture.
- `variable_universe.feature_selection_dynamic_subset` — CV-in-training feature selection loop; needs a tuning-engine extension.
- `deterministic_components.trend_and_quadratic` — overlaps `linear_trend` if we add a `leaf_config.trend_order: int` channel later; the explicit "trend+quadratic" value is redundant.

### 2.2 Demotions (9 values → registry_only)

Label-lie values whose real wiring is the v1.0 implementation PR below:

- `benchmark_family`: `factor_model`, `multi_benchmark_suite`.
- `predictor_family`: `all_except_target`, `category_based`, `factor_only`.
- `variable_universe`: `category_subset`, `paper_replication_subset`.
- `deterministic_components`: `constant_only`, `linear_trend`.

`predictor_family.handpicked_set` and the 4 `variable_universe` pre-existing registry_only values (`target_specific_subset`, `expert_curated_subset`, `stability_filtered_subset`, `correlation_screened_subset`) and `deterministic_components` pre-existing registry_only values (`monthly_seasonal`, `quarterly_seasonal`, `break_dummies`) stay registry_only — the impl PR will implement them via leaf_config input channels.

### 2.3 Test updates

- `tests/test_benchmark_axes.py::test_benchmark_family_operational_set` — remove `factor_model`, `multi_benchmark_suite` from expected op.
- `tests/test_data_task_axes.py` — variable_universe expected set loses `feature_selection_dynamic_subset`.
- `tests/test_v0_9_2_group_h.py` — drop `("deterministic_components", "linear_trend")` from PROMOTED (demoted here; flips back in impl PR).

**Expected:** 717 → ~716 passed (−1 parametrized case from the removed PROMOTED entry).

---

## 3. Implementation PR (next) — v1.0 runtime wiring for 19 registry_only values

All wiring is "user provides a recipe input" OR "simple deterministic feature addition" — no new infrastructure layers.

### 3.1 `benchmark_family` (4 values)

- `factor_model`: use the auxiliary panel already threaded by `_build_predictions`; fit a plain factor model on the non-target columns. ~30 LOC.
- `multi_benchmark_suite`: `leaf_config.benchmark_suite: list[str]` declares the member benchmarks; the suite executor runs each and returns the mean (or a named aggregation). ~40 LOC.
- `paper_specific_benchmark`: `leaf_config.paper_forecast_series: dict[target → Series]` pre-supplies the forecast at every OOS origin; the executor reads from the dict. ~20 LOC.
- `survey_forecast`: same pattern as `paper_specific_benchmark`, keyed `leaf_config.survey_forecast_series`. ~20 LOC.

### 3.2 `predictor_family` (4 values)

- `all_except_target`: column filter `[c for c in panel.columns if c != target]`. ~10 LOC.
- `category_based`: use the tcode / category mapping already loaded with the dataset (FRED-MD groups). ~20 LOC.
- `factor_only`: force `feature_builder=factor_pca` compatibility, selecting the factor columns only. ~10 LOC.
- `handpicked_set`: `leaf_config.handpicked_columns: list[str]` — user-provided column subset. ~20 LOC.

### 3.3 `variable_universe` (6 values)

All take a user-provided or pre-computed subset via leaf_config. None require runtime discovery in v1.0.

- `category_subset`: use the category mapping + `leaf_config.variable_universe_category: str`. ~20 LOC.
- `paper_replication_subset`: `leaf_config.paper_replication_columns: list[str]`. ~15 LOC.
- `target_specific_subset`: `leaf_config.target_specific_columns: dict[target → list[str]]`. ~20 LOC.
- `expert_curated_subset`: `leaf_config.expert_columns: list[str]`. ~15 LOC.
- `stability_filtered_subset`: `leaf_config.stability_filtered_columns: list[str]` (user pre-computed). ~15 LOC.
- `correlation_screened_subset`: `leaf_config.correlation_screened_columns: list[str]`. ~15 LOC.

### 3.4 `deterministic_components` (5 values)

Deterministic feature additions to the X matrix, applied at `_build_raw_panel_training_data` (raw_panel path) and composed into autoreg lag vectors at fit time.

- `constant_only`: explicit `fit_intercept=True` record (most sklearn models default to this already). ~10 LOC.
- `linear_trend`: add a `t` column (time index). ~25 LOC.
- `monthly_seasonal`: add 11 monthly dummy columns. ~30 LOC.
- `quarterly_seasonal`: 3 quarterly dummies. ~20 LOC.
- `break_dummies`: `leaf_config.break_dates: list[str]` → 0/1 dummies. ~30 LOC.

### 3.5 Docs — `docs/user_guide/data/benchmark.md`

New 1.4 page mirroring `task.md` / `horizon.md` style: 4 axes, per-axis Value catalog / Functions & features / Dropped values / Recipe usage. Takeaways.

### 3.6 Total scope

~380 LOC + ~200 LOC tests across 19 values. Doable in a single follow-up PR.

---

## 4. Acceptance gates

### Cleanup PR (this)
- [ ] Registry 130 axes unchanged; 1.4 values 37 → 33; 1.4 op 23 → 14.
- [ ] `pytest tests/` green (~716 passed).
- [ ] coverage_ledger 1.4 / 1.5 / 1.6 sections annotated for drops + demotions.

### Implementation PR (next)
- [ ] 19 registry_only values flip to operational with real runtime wiring.
- [ ] Per-value positive tests + compile guards where the value requires a leaf_config field.
- [ ] `docs/user_guide/data/benchmark.md` reflects operational status.
- [ ] `plans/coverage_ledger.md` rows flipped to OPERATIONAL 2026-04-20 markers.

---

## 5. Breaking changes (pre-v1.0, ADR-006 window)

- Recipes using any of the 4 dropped values now fail compile. None had observable runtime behaviour beyond manifest metadata.
- Recipes using any of the 9 demoted values compile to `representable_but_not_executable` until the impl PR lands. Again, no observable runtime behaviour change — they were metadata-only "operational" before.

---

## 6. Out of scope

- `predictor_family.text_only` / `mixed_blocks` — deferred to v2 (NN / text embeddings stack).
- `variable_universe.feature_selection_dynamic_subset` — deferred to v1.1 (tuning-engine extension).
- 1.5 per-axis walk — separate PR.
- Phase 8 paper_ready_bundle — independent critical path.

---

## 7. v1.0 implementation status (2026-04-20 follow-up)

**All 19 demoted values flipped operational.** 1.4 registry has zero registry_only entries across all 4 axes.

Implementation highlights:

- benchmark_family: 4 new branches in _run_benchmark_executor — factor_model (z-scored leading-factor OLS), multi_benchmark_suite (inline dispatch over member families from leaf_config.benchmark_suite), paper_specific_benchmark / survey_forecast (pre-computed series lookup from leaf_config.paper_forecast_series / survey_forecast_series, keyed by target).
- predictor_family: _raw_panel_columns gained predictor_family + spec parameters; 4 new branches (all_except_target alias, category_based via leaf_config.predictor_category_columns, factor_only via F_ prefix, handpicked_set via leaf_config.handpicked_columns).
- variable_universe: _apply_variable_universe rewritten with full dispatch + 6 new branches; each reads a user-provided column list (or category/target mapping) from data_task_spec (propagated from leaf_config at compile time). Target and date columns are always preserved.
- deterministic_components: new macrocast/execution/deterministic.py module (augment_frame / augment_array) wired into _build_raw_panel_training_data after preprocessing. 5 augmentation rules: constant_only, linear_trend, monthly_seasonal (11 dummies), quarterly_seasonal (3 dummies), break_dummies (leaf_config.break_dates).

Compiler propagates 12 new leaf_config fields into data_task_spec for the impl.

Tests: 20 new positive / guard tests in tests/test_stage1_4_impl.py covering every new wiring. Full suite 716 -> 735 passed.

Docs:
- docs/user_guide/data/benchmark.md written (1.4 page, 4-axis catalog + per-axis Functions & features / Recipe usage / Takeaways).
- docs/user_guide/data/index.md 1.4 row now links to benchmark.md; Honest operational status paragraph updated; hidden toctree adds benchmark.
- plans/coverage_ledger.md 1.1.9 / 1.4.2 / 1.4.5 / 1.6.2 rows flipped to OPERATIONAL 2026-04-20 markers.
