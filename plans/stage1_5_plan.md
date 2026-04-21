# 1.5 Data handling policies — cleanup + implementation plan

**Goal:** make Layer 1 1.5 honest. Audit across the 10 candidate 1.5 axes found that of 35 "operational" values, only ~5 actually dispatch at runtime — the rest are metadata-only. Five axes are flat-out redundant with what 1.1 / 1.2 / 1.3 / 1.4 already express, and one axis (`vintage_policy`) depends on real-time vintage data infrastructure that is not in v1.0 scope. This plan splits the work into (a) a cleanup PR that drops 6 axes entirely and prunes + demotes the 4 kept axes, and (b) an implementation PR that wires the 9 demoted values back to operational.

**Starting state (post-#39):** 130 axes, 735 tests green. 1.5 has 10 axes / 54 values (35 op + 17 registry_only + 2 future).
**After cleanup PR:** 124 axes, 1.5 has 4 axes / 12 values (4 op + 8 registry_only). `evaluation_scale` re-homed to Layer 2 (where it always belonged as a PreprocessContract field).
**After implementation PR:** 1.5 fully operational — 4 axes, 12 values, every one either operational or dropped.

---

## 1. Audit findings

| axis | op (before) | truly wired | verdict |
|---|---|---|---|
| `alignment_rule` | 4 | 0 | DROP — mixed-frequency axis, v1.1 scope with FRED-SD |
| `contemporaneous_x_rule` | 1 | 0 | keep (impl: forbid vs allow X shift policy) |
| `evaluation_scale` | 3 | 3 (as Layer 2 field) | MOVE to Layer 2 — it's the PreprocessContract field, not a 1.5 axis |
| `exogenous_block` | 2 | 0 | DROP — duplicates `feature_builder` default logic |
| `missing_availability` | 8 | 1 | keep (impl: available_case, x_impute_only) |
| `regime_task` | 3 | 0 | DROP — duplicates `oos_period.recession_only_oos` / `expansion_only_oos` (1.3) |
| `release_lag_rule` | 6 | 3 distinct | keep (collapse duplicates) |
| `structural_break_segmentation` | 4 | 0 | keep (impl: NBER-date anchors + leaf_config break_dates) |
| `vintage_policy` | 3 | 1 (latest_only) | DROP — non-default values need vintage-data infra (v1.1 FRED-SD) |
| `x_map_policy` | 1 | 0 | DROP — single-op non-axis; multi-target X mapping already lives in `experiment_unit` |

Total: ~5 truly wired values, 30 metadata-only "operational" label-lies.

---

## 2. Cleanup PR (this one)

### 2.1 Axis drops (6)

- `alignment_rule` — calendar alignment for mixed-frequency datasets; meaningful only for FRED-SD which is v1.1 scope.
- `exogenous_block` — redundant with `feature_builder` default (autoreg → none, raw_panel → endogenous_allowed).
- `regime_task` — redundant with `oos_period.recession_only_oos` / `expansion_only_oos` (1.3 already implements NBER regime filtering).
- `x_map_policy` — single operational value (`shared_X`); multi-target X mapping is owned by `experiment_unit` (0.2).
- `vintage_policy` — non-default values require real-time vintage data infrastructure which v1.0 does not have. Re-enter as part of the v1.1 FRED-SD / real-time-vintage stack.
- `evaluation_scale` re-homed to Layer 2 — the Layer 1 axis was a duplicate of the `PreprocessContract.evaluation_scale` field that the preprocessing contract already carries. Removed from Layer 1 registry; a Layer 2 axis file keeps recipe fixed_axes compile paths intact.

### 2.2 Value drops within kept axes (12)

- `missing_availability`: drop `target_date_drop_if_missing`, `real_time_missing_as_missing`, `state_space_fill`, `factor_fill`, `em_fill` (5). Complex / niche imputation strategies; v1.1+ if demand emerges.
- `release_lag_rule`: drop `calendar_exact_lag`, `lag_conservative`, `lag_aggressive` (3). All three shared observable behaviour with the other op values (duplicate labels for shift 0 or shift 2).
- `structural_break_segmentation`: drop `break_test_detected`, `rolling_break_adaptive` (2). Break detection / adaptive algorithms out of v1.0 scope.
- `contemporaneous_x_rule`: drop `allow_if_available_in_real_time`, `series_specific_contemporaneous` (2). Vintage / per-series metadata-dependent.

### 2.3 Demotions to registry_only (9 values)

Pending runtime wiring in the impl PR:

- `missing_availability`: `available_case`, `x_impute_only`.
- `release_lag_rule`: `series_specific_lag`.
- `structural_break_segmentation`: `pre_post_crisis`, `pre_post_covid`, `user_break_dates`.
- `contemporaneous_x_rule`: `allow_contemporaneous`.
- (2 already registry_only: `available_case`, `x_impute_only` were op; `allow_contemporaneous` was reg_only; the 3 break values were op-but-metadata.)

### 2.4 Compiler / execution cleanup

- `_data_task_spec` loses spec lines for `alignment_rule`, `evaluation_scale`, `exogenous_block`, `regime_task`, `x_map_policy`, `vintage_policy`.
- `_eval_scale` extraction in `_build_predictions` head removed.
- `_PHASE3_DEFAULTS["evaluation_scale"]` removed.
- `_apply_release_lag` kept (it's the one genuinely wired `_apply_*` in 1.5) — logic unchanged; 3 dropped duplicate values silently dead without the registry path.

### 2.5 Test updates

- `tests/test_registry_loader.py`: registry count 130 → 124; drop the 6 axis names from the Stage-1 expected set.
- `tests/test_v0_9_2_group_d_f.py`: drop PROMOTED entries referencing dropped values / axes (`alignment_rule.*`, `x_map_policy.shared_X` — see below).
- `tests/test_v0_9_2_group_h.py`: drop `("x_map_policy", "shared_X")`, any remaining dropped-value entries.
- `tests/test_data_task_axes.py`: update expected sets for kept axes (removed values).
- `tests/test_data_task_axes_runtime.py`: drop round-trip tests for dropped axes.
- `tests/test_sweep_runner.py`: recipe fixtures lose references to dropped axes where present.

**Expected:** 735 → ~720 passed (15-20 parametrised cases disappear with dropped axes / values).

---

## 3. Implementation PR (next) — 9 registry_only values → operational

### 3.1 `missing_availability.available_case`
Drop rows where the TARGET series (or any selected predictor if raw_panel) has NaN inside the training window; do not fill. ~25 LOC.

### 3.2 `missing_availability.x_impute_only`
Apply a user-specified imputation strategy via `leaf_config.x_imputation: str` to predictor columns only; target column stays NaN-sensitive. Reuses existing `SimpleImputer` or forward-fill infrastructure in `_apply_missing_policy`. ~30 LOC.

### 3.3 `release_lag_rule.series_specific_lag`
`leaf_config.release_lag_per_series: dict[str, int]` declares per-column release lag (months). Applied in `_apply_release_lag` as `shift(per_series[col])` instead of the uniform shift. ~35 LOC.

### 3.4 `structural_break_segmentation.pre_post_crisis`
Pre-2008-09 vs post-2008-09 segmentation. Implementation options: either (a) add a single break dummy at 2008-09-01, reusing 1.4 `deterministic_components.break_dummies` infra, or (b) filter OOS origins to one segment. Simplest v1.0: add the break dummy to X. ~20 LOC.

### 3.5 `structural_break_segmentation.pre_post_covid`
Same pattern at 2020-03-01. ~15 LOC.

### 3.6 `structural_break_segmentation.user_break_dates`
Reuses `leaf_config.break_dates` (already propagated in 1.4 impl). Break dummies are added to X. ~20 LOC.

### 3.7 `contemporaneous_x_rule.allow_contemporaneous`
Drop the implicit `X.shift(1)` when building the predictor matrix for raw_feature_panel. Contemporaneous (`X_t` at time `t`) is then allowed. ~25 LOC.

### 3.8 Docs — `docs/user_guide/data/policies.md`
New 1.5 page mirroring `task.md` / `horizon.md` / `benchmark.md` style.

### 3.9 Total scope

~170 LOC + ~120 LOC tests. Smaller than 1.3 / 1.4 impls.

---

## 4. Acceptance gates

### Cleanup PR (this)
- [ ] Registry 130 → 124 axes.
- [ ] 1.5 candidate set: 10 → 4 axes; 54 → 12 values (4 op + 8 registry_only).
- [ ] `pytest tests/` green (~720 passed).
- [ ] coverage_ledger + data/index.md honest-status updated.

### Implementation PR (next)
- [ ] 9 registry_only values flip to operational with real wiring.
- [ ] Per-value positive tests.
- [ ] `docs/user_guide/data/policies.md` written.
- [ ] coverage_ledger rows flipped to OPERATIONAL 2026-04-20 markers.

---

## 5. Breaking changes (pre-v1.0, ADR-006 window)

- Recipes using any of the 6 dropped axes now fail compile. None had observable runtime behaviour beyond manifest metadata (the exception is `vintage_policy.latest_only` which was a no-op default).
- Recipes using any of the 12 dropped values within kept axes fail compile. None had observable runtime behaviour.
- Recipes using the 9 demoted values compile to `representable_but_not_executable` until the impl PR lands.

---

## 6. Out of scope

- `vintage_policy` / `alignment_rule` — v1.1 FRED-SD real-time-vintage stack.
- `missing_availability` complex imputations — v1.1+.
- `structural_break_segmentation.break_test_detected` / `rolling_break_adaptive` — change-point detection library.

---

## 7. v1.0 implementation status (2026-04-21 follow-up)

**All 9 demoted values flipped operational.** 1.5 registry has zero registry_only entries across all 4 axes.

Implementations:

- contemporaneous_x_rule.allow_contemporaneous — _build_raw_panel_training_data branches on the axis value: default forbid pairs (X_t, y_{t+h}); allow pairs (X_{t+h}, y_{t+h}) and uses X at origin_idx + horizon for prediction (oracle / data-leak benchmark).

- release_lag_rule.series_specific_lag — _apply_release_lag rewritten with per-rule dispatch. series_specific_lag reads leaf_config.release_lag_per_series (dict[col -> int months]) and applies shift per column; missing dict raises ExecutionError. calendar_exact_lag / lag_conservative / lag_aggressive duplicates were dropped in the cleanup PR.

- missing_availability.available_case / x_impute_only — _apply_missing_availability rewritten. available_case drops rows with any NaN across non-date columns. x_impute_only reads leaf_config.x_imputation in {mean, median, ffill, bfill} and fills predictor columns only (target retains NaNs).

- structural_break_segmentation.pre_post_crisis / pre_post_covid / user_break_dates — new _resolve_structural_break_dates helper maps the axis value to a break-date list (2008-09-01 / 2020-03-01 presets, or leaf_config.break_dates). The list is passed to the existing 1.4 augment_array(component='break_dummies') path inside _build_raw_panel_training_data, so X_train and X_pred gain one 0/1 dummy per break date.

Compiler propagates 3 additional leaf_config fields (release_lag_per_series, x_imputation, break_dates — already in place from 1.4) into data_task_spec. missing_availability + release_lag_rule now also appear explicitly in data_task_spec for manifest visibility.

Tests: 12 new positive / guard tests in tests/test_stage1_5_impl.py. Stale available_case / x_impute_only pass-through check in test_data_task_axes_runtime.py reduced to complete_case_only only. Full suite 727 -> 737 passed.

Docs:
- docs/user_guide/data/policies.md written — 1.5 page mirroring task.md / horizon.md / benchmark.md style.
- docs/user_guide/data/index.md 1.5 row now links to policies.md; Honest operational status paragraph refreshed; hidden toctree adds policies.

Layer 1 per-axis walk complete — 1.1 through 1.5 all fully operational & honest.
