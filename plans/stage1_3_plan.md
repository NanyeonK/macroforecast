# §1.3 Horizon & Evaluation Window — cleanup + implementation plan

**Goal:** make Layer 1 §1.3 honest. All 7 §1.3 candidate axes (as grouped in `docs/user_guide/data/index.md`) have "operational" labels but the live execution path branches on exactly zero of them — 22 of 22 op labels are metadata-only. This plan drops 3 redundant / abstract axes entirely, prunes the remaining 4 axes to a clean value set, and schedules the real v1.0 runtime wiring as a second follow-up PR.

**Starting state (post-#35):** 133 axes, 712 tests green. §1.3 has 7 axes / 35 values (22 op + 12 registry_only + 1 future).
**After cleanup PR:** 130 axes, §1.3 has 4 axes / 12 values (4 op + 8 registry_only). Registry_only entries are acceptance criteria for the implementation PR.
**After implementation PR:** §1.3 fully operational — 4 axes, 12 values, every value either operational or dropped.

---

## 1. Audit findings

| axis | registry claim | actual runtime | verdict |
|---|---|---|---|
| `horizon_list` | 4 op + 1 reg_only | axis read in `_rows_for_horizon` but never consumed; `leaf_config.horizons` is the real source | **DROP axis — redundant** |
| `oos_period` | 4 op + 2 reg_only | compiler default only; `framework` axis drives rolling-vs-expanding | drop 4 duplicates/niche + implement regime filter |
| `training_start_rule` | 4 op + 1 reg_only | compiler default only | drop 3 + implement `fixed_start` |
| `min_train_size` | 2 op + 3 reg_only | `raw/windowing.py` has the dispatch but is **dead code** (never imported) | wire in impl PR |
| `warmup_rule` | 3 op + 1 reg_only + 1 future | compiler default only; no concrete semantic | **DROP axis — abstract** |
| `overlap_handling` | 2 op + 3 reg_only | compiler default only; HAC implemented at stat test layer but not gated by this axis | drop 3 + implement HAC gate |
| `own_target_lags` | 3 op + 1 reg_only | `feature_builder` already determines y-lag inclusion | **DROP axis — redundant** |

---

## 2. Cleanup PR (this one) — pure removals + demotions

### 2.1 Axes dropped entirely (3)

- `horizon_list` — redundant with `leaf_config.horizons`.
- `warmup_rule` — abstract, no v1.0 dispatch semantic.
- `own_target_lags` — redundant with `feature_builder`.

Files deleted: `macrocast/registry/data/{horizon_list,warmup_rule,own_target_lags}.py`. Compiler references removed from `_data_task_spec`. Execution reference (`_PHASE3_DEFAULTS["horizon_list"]` + `_horizon_list_axis = _data_task_axis(..., "horizon_list")`) removed.

### 2.2 Value drops within kept axes (10)

- `oos_period`: drop `single_oos_block`, `rolling_origin` (duplicate `framework`), `multiple_oos_blocks`, `event_window_oos` (niche). Add `all_oos_data` as the new no-filter default.
- `training_start_rule`: drop `rolling_train_start` (duplicate `framework`), `post_warmup_start` (warmup axis dropped), `post_break_start` (structural_break not v1.0-operational).
- `overlap_handling`: drop `evaluate_with_block_bootstrap`, `non_overlapping_subsample`, `horizon_specific_subsample` (registry_only with no v1.0/v1.1 path).

### 2.3 Demotions to registry_only (until impl PR)

Values that stay in the registry but await runtime wiring:

- `oos_period.recession_only_oos`, `oos_period.expansion_only_oos` → implementation via NBER date filter.
- `training_start_rule.fixed_start` → implementation via `leaf_config.training_start_date`.
- `overlap_handling.evaluate_with_hac` → implementation via HAC gate in stat test layer.
- `min_train_size` — no status changes in this PR; current 2 op + 3 registry_only preserved. Impl PR will wire the `raw/windowing.py` dispatch into the main execution path and promote accordingly.

### 2.4 Compiler defaults updated

- `oos_period` default: was dynamic (`rolling_origin` vs. `single_oos_block`); now `"all_oos_data"` (static).
- `training_start_rule` default: was dynamic (`rolling_train_start` vs. `earliest_possible`); now `"earliest_possible"` (static).

### 2.5 Test updates

- `tests/test_registry_loader.py`: total axis count 133 → 130 (5 sites); drop `horizon_list` / `warmup_rule` / `own_target_lags` from the Stage-1 expected set.
- `tests/test_v0_9_2_group_h.py`: remove PROMOTED entries referencing dropped values (`min_train_size.fixed_years` still in test as this value is kept; `overlap_handling.evaluate_with_hac` removed since it demoted; `own_target_lags.exclude` removed — axis dropped).
- `tests/test_v0_9_2_group_d_f.py`: remove PROMOTED entries for dropped `warmup_rule.*` and dropped `training_start_rule.*`.
- `tests/test_data_task_axes.py`: drop `horizon_list` block from `_NEW_AXES` dict.
- `tests/test_data_task_axes_runtime.py`: remove `test_horizon_list_round_trip`.

**Expected:** 712 → 700 passing (~12 parametrized cases disappear with dropped axes).

---

## 3. Implementation PR (next) — v1.0 runtime wiring

Covers the demoted / pending values.

### 3.1 `oos_period` regime filter

NBER business-cycle peak/trough dates hardcoded as an internal fixture (~20 historical segments). At `_compute_origin` (or prior to row construction), filter rows by recession/expansion membership when `oos_period ∈ {recession_only_oos, expansion_only_oos}`. `all_oos_data` is the no-op default.

Estimated ~80 LOC + tests.

### 3.2 `training_start_rule.fixed_start`

Accept `leaf_config.training_start_date` (ISO date string). When `training_start_rule = fixed_start`, compiler validates the date exists in the index and propagates a `fixed_start_idx` that `_rows_for_horizon` uses as `base_start_idx` ceiling. `earliest_possible` is the no-op default.

Estimated ~30 LOC + tests.

### 3.3 `min_train_size` wiring

Connect `raw/windowing.py::_resolve_min_train_obs` into the main execution path. Today `_minimum_train_size(recipe)` reads from `_benchmark_spec["minimum_train_size"]` (the static leaf_config number) — ignore both the axis value and the windowing module. The impl PR wires the axis value → windowing dispatch → actual `minimum_train_size` used in `_build_predictions`.

All 5 values (`fixed_n_obs`, `fixed_years`, `model_specific_min_train`, `target_specific_min_train`, `horizon_specific_min_train`) get promoted to operational once wiring is in place, since the dispatch implementations already exist in `windowing.py`.

Estimated ~40 LOC + tests.

### 3.4 `overlap_handling.evaluate_with_hac`

HAC covariance is already used inside `dm_hln` / `cpa`-family stat tests. The axis value should act as a compile-time gate: when `overlap_handling = evaluate_with_hac`, assert the downstream stat test is one that respects HAC and propagate the choice to the stat test spec. When `allow_overlap` (default), keep current behaviour.

Estimated ~50 LOC + tests.

### 3.5 Docs — `docs/user_guide/data/horizon.md` (§1.3 page)

Mirror `task.md` structure: intro + per-axis subsections with Value catalog / Functions & features / Recipe usage. Note dropped axes at the top.

---

## 4. Acceptance gates

### Cleanup PR (this)
- [ ] Registry: 133 → 130 axes; §1.3 axes 7 → 4; values 35 → 12.
- [ ] `pytest tests/` green. Expected ~700 passed.
- [ ] coverage_ledger entries annotated for dropped axes + dropped values + demotions.

### Implementation PR (next)
- [ ] 8 registry_only values flip to operational as wiring lands: `oos_period.recession_only_oos`/`expansion_only_oos`, `training_start_rule.fixed_start`, `overlap_handling.evaluate_with_hac`, plus the 3 `min_train_size` registry_only entries once windowing is wired (total 8 — 2 + 1 + 1 + 3 + plus `min_train_size.fixed_years` re-verification).
- [ ] Per-value positive tests for each new runtime path.
- [ ] `docs/user_guide/data/horizon.md` reflects operational status.
- [ ] `plans/coverage_ledger.md` §1.2.2 / §1.2.3 / §1.2.4 / §1.3.5 flipped to OPERATIONAL 2026-04-20 markers.

---

## 5. Breaking changes (pre-v1.0, ADR-006 window)

- Recipes using any dropped `oos_period` / `training_start_rule` / `overlap_handling` value, or any value of the 3 dropped axes, now fail compile. None of these values had observable runtime behaviour beyond manifest metadata, so no forecasts change.
- Default `forecast_type`-style dynamic defaults on `oos_period` and `training_start_rule` removed — recipes that relied on the framework-based dynamic default now get the static `all_oos_data` / `earliest_possible` instead. Behaviour is identical (both were no-ops at runtime).

---

## 6. Out of scope

- Training-time horizon_target_construction (the v1.0 metric-scale approach is the current reality; training-time transforms are future work).
- Full structural-break-aware training (the `post_break_start` drop defers this until the break axis is itself operational).
- §1.4 / §1.5 per-axis walks — separate PRs.
- Phase 8 paper_ready_bundle — independent critical path.
