# 1.2 Task & Target — cleanup plan (resolution locked 2026-04-20)

**Goal:** make Layer 1 1.2 honest. Of the 13 currently-operational values, 7 are label-lies (compiler records them, no execution branch). Plus the axis registry holds many registry_only / future values that will either become v1.1 roadmap items or get cut. This plan locks the resolution, demotes the label-lies, drops the cut values, drops one redundant axis (`target_to_target_inclusion`), writes the 1.2 user-guide page, and closes the ledger.

**Starting state (post PR #32):** 135 axes, 684 tests green. 1.2 has 5 axes, 29 values (13 op).
**Ending state (this PR):** 134 axes, ~690 tests green. 1.2 has 4 axes, 11 values (6 op, 5 registry_only marked for v1.1).

---

## 1. Resolution matrix (locked)

Rule: only values that have a real future runtime path survive as `registry_only` (= v1.1 roadmap). Values without a concrete implementation path are removed from the registry. "Future" as a holding bay is abandoned — if we do not have a v1.1 commitment, we drop.

| axis | value | current | resolution | rationale |
|---|---|---|---|---|
| `task` | `single_target_point_forecast` | op | **keep op** | wired (compiler branches at 3 sites) |
| `task` | `multi_target_point_forecast` | op | **keep op** | wired (aggregator dispatch) |
| `forecast_type` | `direct` | op | **keep op** | default, every executor is direct |
| `forecast_type` | `iterated` | op | **demote → registry_only (v1.1)** | Marcellino-Stock-Watson (2006) classic comparator; v1.1 adds recursive-1-step wrapper |
| `forecast_type` | `dirrec` | reg_only | **drop** | niche hybrid (Taieb-Bontempi 2011), no v1.x demand |
| `forecast_type` | `mimo` | future | **drop** | deep-only, v2 Transformer scope will re-introduce as model capability |
| `forecast_type` | `seq2seq` | future | **drop** | v2 Transformer scope, re-introduce as model capability |
| `forecast_object` | `point_mean` | op | **keep op** | wired (default metric is MSE on mean point) |
| `forecast_object` | `point_median` | op | **keep op** | wired via `quantile_linear` guard |
| `forecast_object` | `quantile` | op | **demote → registry_only (v1.1)** | v1.1 adds conformal/quantile-loss path; currently label-only |
| `forecast_object` | `direction` | op | **drop** | sign is a metric view on point forecast, not an independent forecast object |
| `forecast_object` | `interval` | reg_only | **drop** | subsumed by future conformal wrapper on point (no separate axis needed) |
| `forecast_object` | `density` | reg_only | **drop** | v2 distributional work, re-introduce when ready |
| `forecast_object` | `turning_point` | reg_only | **drop** | niche, can be reconstructed from point forecasts post-hoc |
| `forecast_object` | `regime_probability` | future | **drop** | bound to state_space (v2), will re-introduce with that stack |
| `forecast_object` | `event_probability` | future | **drop** | niche, no v1.1 commitment |
| `horizon_target_construction` | `future_level_y_t_plus_h` | op | **keep op** | default, actually wired via `y.shift(-h)` |
| `horizon_target_construction` | `future_diff` | op | **demote → registry_only (v1.1)** | 1st-difference target; v1.1 target-transform inverse pipeline |
| `horizon_target_construction` | `future_logdiff` | op | **demote → registry_only (v1.1)** | log-growth; same pipeline |
| `horizon_target_construction` | `cumulative_growth_to_h` | op | **demote → registry_only (v1.1)** | CLSS-style cumulative; same pipeline |
| `horizon_target_construction` | `annualized_growth_to_h` | op | **drop** | linear (×12/h) transform of cumulative_growth_to_h — not a distinct target shape, belongs in metric-time reporting |
| `horizon_target_construction` | `average_growth_1_to_h` | reg_only | **drop** | scaled variant of cumulative, redundant |
| `horizon_target_construction` | `realized_future_average` | reg_only | **drop** | niche, no v1.1 demand |
| `horizon_target_construction` | `future_sum` | reg_only | **drop** | niche |
| `horizon_target_construction` | `future_indicator` | reg_only | **drop** | overlaps dropped `forecast_object=direction` and `event_probability` |
| `target_to_target_inclusion` | all 4 values | mixed | **DROP AXIS** | single op value = current hardcoded behaviour; no dispatch anywhere. Re-enter as clean v1.1 axis (e.g. `cross_target_predictor_policy`) if/when needed |

**Totals:** 2 keep-op axes (`task`, `forecast_type` default, `forecast_object` point_*, `horizon_target_construction` default) + 5 values demoted to registry_only (v1.1 roadmap) + 14 values dropped + 1 axis dropped.

---

## 2. Per-axis sub-tasks

### 2.1 `forecast_type` — demote 1, drop 3
- File: `macrocast/registry/data/forecast_type.py`
- `iterated`: `operational` → `registry_only`
- Delete entries: `dirrec`, `mimo`, `seq2seq`
- Remaining: 2 values (`direct` op, `iterated` registry_only)

### 2.2 `forecast_object` — demote 1, drop 6
- File: `macrocast/registry/data/forecast_object.py`
- `quantile`: `operational` → `registry_only`
- Delete entries: `direction`, `interval`, `density`, `turning_point`, `regime_probability`, `event_probability`
- Compiler `quantile_linear` guard at `compiler/build.py:641` **unchanged** — still requires `forecast_object=point_median`.
- Remaining: 3 values (`point_mean` op, `point_median` op, `quantile` registry_only)

### 2.3 `horizon_target_construction` — demote 3, drop 5
- File: `macrocast/registry/data/horizon_target.py`
- Demote: `future_diff`, `future_logdiff`, `cumulative_growth_to_h` → `registry_only`
- Delete entries: `annualized_growth_to_h`, `average_growth_1_to_h`, `realized_future_average`, `future_sum`, `future_indicator`
- Remaining: 4 values (`future_level_y_t_plus_h` op + 3 registry_only)

### 2.4 `target_to_target_inclusion` — drop axis
- Delete file `macrocast/registry/data/target_to_target_inclusion.py`
- Remove `compiler/build.py:408` default line
- Tests:
  - `tests/test_registry_loader.py`: remove from expected Stage-1 set, decrement count at 5 sites (135→134)
  - `tests/test_v0_9_2_group_h.py`: remove `("target_to_target_inclusion", "forbid_other_targets_as_X")` from PROMOTED tuple

### 2.5 Negative tests (compile rejection for dropped / demoted values)
New test file `tests/test_stage1_12_cleanup.py`:
- 3 tests: `forecast_type in {dirrec, mimo, seq2seq}` → compile fails (unknown value)
- 1 test: `forecast_type=iterated` → compile fails (registry_only status)
- 6 tests: `forecast_object in {direction, interval, density, turning_point, regime_probability, event_probability}` → compile fails (unknown value)
- 1 test: `forecast_object=quantile` → compile fails (registry_only status)
- 5 tests: `horizon_target_construction in {annualized_growth_to_h, average_growth_1_to_h, realized_future_average, future_sum, future_indicator}` → compile fails
- 3 tests: `horizon_target_construction in {future_diff, future_logdiff, cumulative_growth_to_h}` → compile fails (registry_only status)
- 1 test: `target_to_target_inclusion=<any>` → compile fails (axis unknown)

Total: **20 new negative tests**.

### 2.6 `coverage_ledger.md` update
- Annotate dropped values with `**DROPPED 2026-04-20**` + one-line rationale.
- Annotate demoted values with `**DEMOTED 2026-04-20 → registry_only (v1.1)**`.
- Annotate `target_to_target_inclusion` section with `**AXIS DROPPED 2026-04-20**`.
- Mark 1.4.1 `target_family` + 1.5.3 `multi_target_architecture` with `**AXIS DROPPED 2026-04-20 (PR #32)**` (carryover from previous PR).
- Historical plan files (`v0_91_plan.md`, `v0_9_2_planned_completion_plan.md`, `implementation-issues.md`, `archive/*`) left untouched.

### 2.7 Docs — `docs/user_guide/data/task.md`
- New file mirroring `data/source.md` structure: title `# Task & Target (1.2)`, intro, value catalog table linking to 4 per-axis subsections.
- Each subsection (Purpose / Value catalog w/ honest-status column / Functions & features / Recipe usage / Known gaps):
  - 1.2.1 `task`
  - 1.2.2 `forecast_type`
  - 1.2.3 `forecast_object`
  - 1.2.4 `horizon_target_construction`
- End with `## Task & Target (1.2) takeaways` summarising kept / demoted / dropped.
- Update `docs/user_guide/data/index.md` table: row 1.2 becomes `| 1.2 | [Task & Target (1.2)](task.md) | 4 | ... |` (4 axes after target_to_target_inclusion drop).
- Update `docs/user_guide/index.md` Stages toctree: add `data/task` after `data/source`.

---

## 3. Acceptance gate

- [ ] Registry: **135 → 134 axes**. 1.2 value count **29 → 11**. 1.2 op values **13 → 6**.
- [ ] `pytest tests/` green. 684 → ~703 (+20 negative − 1 PROMOTED entry). Actual count TBD.
- [ ] `data/task.md` Sphinx builds with no new warnings (pre-existing 4 SweepVariant baseline holds).
- [ ] `data/index.md` table 1.2 links to task.md, lists 4 axes.
- [ ] `user_guide/index.md` Stages toctree includes `data/task`.
- [ ] coverage_ledger entries annotated.

---

## 4. Commit sequence (single stacked PR)

Branch: `feat/stage-1-12-cleanup` off main (after PR #32 merges).

1. `refactor(forecast_type)`: demote iterated, drop {dirrec, mimo, seq2seq}
2. `refactor(forecast_object)`: demote quantile, drop {direction, interval, density, turning_point, regime_probability, event_probability}
3. `refactor(horizon_target_construction)`: demote 3, drop 5
4. `refactor(stage-1): drop target_to_target_inclusion axis` + registry_loader / PROMOTED tuple updates
5. `test(stage-1): 1.2 negative compile tests` — the 20 new tests
6. `docs(coverage_ledger)`: annotate dropped + demoted entries
7. `docs(stage-1): 1.2 Task & Target page` + data/index table + user_guide toctree

PR title: `refactor(stage-1): 1.2 Task & Target honest cleanup`

---

## 5. Breaking changes (pre-v1.0, ADR-006 window)

Any recipe that currently compiles with one of these values will now fail:
- `forecast_type`: `iterated`, `dirrec`, `mimo`, `seq2seq`
- `forecast_object`: `quantile`, `direction`, `interval`, `density`, `turning_point`, `regime_probability`, `event_probability`
- `horizon_target_construction`: `future_diff`, `future_logdiff`, `cumulative_growth_to_h`, `annualized_growth_to_h`, `average_growth_1_to_h`, `realized_future_average`, `future_sum`, `future_indicator`
- `target_to_target_inclusion`: any value

None of these had observable runtime behaviour beyond manifest metadata, so the breakage is cosmetic / shape-only (no predictions change). v1.1 will re-enable the 5 demoted values with real runtime.

---

## 6. v1.1 roadmap items unlocked by this cleanup

5 registry_only values commit v1.1 to deliver these runtime paths:
1. `forecast_type=iterated` — recursive 1-step wrapper over any direct 1-step model.
2. `forecast_object=quantile` — conformal / quantile-loss forecast path with interval output as a by-product.
3. `horizon_target_construction` target-transform pipeline (forward + inverse) for `future_diff`, `future_logdiff`, `cumulative_growth_to_h`.

These belong in the Phase 10 (v1.1) catalog section "Layer 1 deferred dispatch"; after this PR, update `plans/phases/phase_10_v1_1_scope.md` to cite the registry_only entries as acceptance criteria.

---

## 7. Out of scope

- Actual v1.1 runtime for demoted values.
- 1.3 / 1.4 / 1.5 per-axis walks (separate PRs).
- Phase 8 paper_ready_bundle (independent critical path).

---

## 8. v1.0 implementation status (2026-04-20 follow-up)

**3 horizon_target_construction values flipped operational** (separate commit on top of this plan):

- future_diff, future_logdiff, cumulative_growth_to_h now all compile and execute end-to-end. v1.0 semantics is a **metric-scale transform** at _compute_origin — the executor emits a level-scale forecast, and the row builder forward-transforms y_true / y_pred / benchmark_pred using y_anchor before error / abs_error / squared_error are computed. Level-scale values preserved as y_{true,pred}_level / benchmark_pred_level columns.
- New module macrocast/execution/horizon_target.py exports forward_scalar (wired) plus build_horizon_target / inverse_horizon_target (reserved for a future training-time wiring).
- Tests: tests/test_horizon_target_construction.py (8 cases). Full suite 705 passed.

**All 5 deferred registry_only values are now operational** (second follow-up commit):

- **forecast_type=iterated** — flipped operational. The autoreg_lagged_target executor path is already iterated by construction (1-step fit + recursive prediction); the compiler now picks forecast_type dynamically by feature_builder (autoreg → iterated, raw_panel → direct). Cross combinations blocked_by_incompatibility: raw_feature_panel+iterated (requires exogenous X forecasting) and autoreg_lagged_target+direct (true direct autoreg executor deferred). No executor refactor needed.
- **forecast_object=quantile** — flipped operational. Compiler guard loosened from quantile_linear => forecast_object=point_median to quantile_linear => forecast_object IN {point_median, quantile}. The QuantileRegressor already reads quantile from training_spec.hp; users set leaf_config.training_spec.hp.quantile to pick the level (default 0.5 = numerically median).

**1.2 registry final state** — task/forecast_type/forecast_object/horizon_target_construction: every value in every axis is either operational or dropped. No registry_only entries remain in 1.2.

Tests: 9 new positive / guard tests in tests/test_forecast_type_quantile.py. Full suite 712 passed.
