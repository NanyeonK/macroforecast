# Horizon & Evaluation Window (1.3)

Declares **which observations count as training data, which count as OOS, and how OOS rows are filtered or aggregated**. This page is the historical 1.3 user-guide entry; canonical ownership is split across later layers. `oos_period` is now a Layer 4 evaluation axis, and `overlap_handling` is now a Layer 6 inference axis. Old Layer 1 placement is accepted only as a compatibility alias.

| Section | axis | Role |
|---|---|---|
| 1.3.1 | [`min_train_size`](#131-min_train_size) | How the minimum training window is computed (fixed obs, years, model/target/horizon-specific) |
| 1.3.2 | [`training_start_rule`](#132-training_start_rule) | Where the training window starts — earliest observation or a fixed calendar date |
| 1.3.3 | [`oos_period`](#133-oos_period) | Regime filter on OOS origins (NBER recession / expansion) |
| 1.3.4 | [`overlap_handling`](#134-overlap_handling) | How overlapping forecast errors (h>1) are handled in stat tests |

**Note on dropped axes:**

- `horizon_list` was dropped (1.3 cleanup) — redundant with `leaf_config.horizons` which already specifies the horizons list directly.
- `warmup_rule` was dropped (1.3 cleanup) — abstract axis with no concrete v1.0 dispatch semantic.
- `own_target_lags` was dropped (1.3 cleanup) — redundant with `feature_builder` (target_lag_features includes target lags, raw_feature_panel does not).
**At a glance (defaults):**
- `min_train_size = fixed_n_obs` — your recipe's `benchmark_config.minimum_train_size` is the observation count.
- `training_start_rule = earliest_possible` — training starts at the first feasible row. Override only for paper-replication (calendar-exact) start dates.
- `oos_period = all_oos_data` — no regime filter. Switch to `recession_only_oos` / `expansion_only_oos` for NBER-conditioned evaluation.
- `overlap_handling = allow_overlap` — stat tests run with their default covariance. Switch to `evaluate_with_hac` only with an HAC-capable Layer 6 test and h > 1.

**Most research runs leave all four at the default.**


---

## 1.3.1 `min_train_size`

**Selects the rule for computing the minimum training window size.** Every value is operational in v1.0, wired via `macroforecast.raw.windowing._resolve_min_train_obs`.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `fixed_n_obs` | operational | Default. Use `leaf_config.benchmark_config.minimum_train_size` verbatim as the observation count. |
| `fixed_years` | operational | Multiply the base by 12 — the scalar is interpreted as years (monthly data). |
| `model_specific_min_train` | operational | `max(base, floor_for_model)` — per-family floors: ridge=60, lasso=80, linear_regression=30. |
| `target_specific_min_train` | operational | `max(base, floor_for_target)` — per-target floors: INDPRO=60, PAYEMS=80, CPIAUCSL=100. |
| `horizon_specific_min_train` | operational | `base + 6 * max(0, horizon - 1)` — longer horizons demand more history. |

### Functions & features

- `macroforecast.raw.windowing._resolve_min_train_obs(spec, model_family, target, horizon)` — the dispatch implementation, re-used from the windowing module (previously dead code; live in v1.0).
- `macroforecast.execution.build._minimum_train_size(recipe, *, horizon=None)` — the execution-layer entry point. Falls back to the largest recipe horizon when an explicit horizon is not supplied (conservative).

### Recipe usage

```yaml
# Use a 5-year minimum training window (monthly data, base 5 -> 60 obs)
path:
  1_data_task:
    fixed_axes:
      min_train_size: fixed_years
  8_output:
    leaf_config:
      benchmark_config:
        minimum_train_size: 5        # interpreted as 5 years
```

---

## 1.3.2 `training_start_rule`

**Selects where the training window begins.** Two operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `earliest_possible` | operational | Default. Training starts at the first observation that is at least `minimum_train_size` rows before the first OOS origin. |
| `fixed_start` | operational | Training starts no earlier than `leaf_config.training_start_date`. Required leaf_config field — compiler blocks if missing. |

### Compatibility guard (v1.0)

- `training_start_rule = fixed_start` without `leaf_config.training_start_date` → `blocked_by_incompatibility`.

### Functions & features

- Compiler-side validation in `macroforecast.compiler.build`'s `_execution_status` emits the guard.
- `macroforecast.execution.build._build_predictions` resolves the date to an index floor via `target_series.index.searchsorted`, then applies it as a `max(base_start_idx, fixed_start_idx)` in `_rows_for_horizon`.

### Dropped values

- `rolling_train_start`: duplicated `framework=rolling` which already sets the rolling start via `rolling_window_size`.
- `post_warmup_start`: depended on the now-dropped `warmup_rule` axis.
- `post_break_start`: depended on `structural_break_segmentation` which is not v1.0-operational.

### Recipe usage

```yaml
# Replication of a paper using a 1965 sample start
path:
  1_data_task:
    fixed_axes:
      training_start_rule: fixed_start
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
      training_start_date: "1965-01-01"
```

---

## 1.3.3 `oos_period`

**Selects an NBER business-cycle regime filter on OOS origins.** Three operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `all_oos_data` | operational | Default. No filter — every OOS origin from `_rows_for_horizon` is kept. |
| `recession_only_oos` | operational | Keep only origins whose date falls within an NBER recession interval. |
| `expansion_only_oos` | operational | Keep only origins whose date falls OUTSIDE every NBER recession interval. |

### Functions & features

- `macroforecast.execution.nber.NBER_RECESSIONS` — frozen fixture of 12 recessions from 1948 to 2020 (NBER Business Cycle Dating Committee).
- `macroforecast.execution.nber.is_recession(date)` / `is_expansion(date)` — scalar membership helpers.
- `macroforecast.execution.nber.filter_origins_by_regime(origin_plan, index, regime)` — filters an `_rows_for_horizon` origin plan in place.
- The filter is applied after `origin_plan` is finalised and before per-origin computation, so refit_policy state is unaffected.

### Dropped values

- `single_oos_block`, `rolling_origin`: duplicated `framework=expanding` / `framework=rolling` — those framework values already drive the base OOS block shape.
- `multiple_oos_blocks`: multi-window OOS evaluation; niche, no v1.0/v1.1 demand.
- `event_window_oos`: event-based OOS; niche, no v1.0/v1.1 demand.

### Recipe usage

```yaml
# Giacomini-White-style recession-only evaluation
path:
  5_evaluation:
    fixed_axes:
      oos_period: recession_only_oos
```

---

## 1.3.4 `overlap_handling`

**Selects how the correlation in overlapping forecast errors (h>1) is handled at the stat test layer.** Two operational values. Canonical placement is `path.6_stat_tests.fixed_axes.overlap_handling`.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `allow_overlap` | operational | Default, no-op. Stat tests receive loss differentials with no HAC adjustment (they may still apply it internally). |
| `evaluate_with_hac` | operational | Compile-time gate that requires HAC-capable split Layer 6 tests such as `equal_predictive=dm_hln`, `equal_predictive=dm_modified`, `nested=cw`, `nested=enc_new`, `nested=mse_t`, `cpa_instability=cpa`, `multiple_model=spa`, or `multiple_model=mcs`. |

### Compatibility guard (v1.0)

- `overlap_handling = evaluate_with_hac` + any active test not in the HAC-compatible set → `blocked_by_incompatibility`.
- Legacy `stat_test` values are still accepted and routed into the split axis before this guard runs.

### Functions & features

- Compiler-side guard in `macroforecast.compiler.build._execution_status`.
- Stat test HAC path already implemented in `macroforecast.execution.build._compute_dm_hln_test` / `_compute_dm_modified_test` with `dependence_correction="nw_hac"`.

### Dropped values

- `evaluate_with_block_bootstrap`: block-bootstrap SEs; requires bootstrap infrastructure not in v1.0 scope.
- `non_overlapping_subsample`: subsample every h-th row to avoid overlap; niche.
- `horizon_specific_subsample`: per-horizon subsample variant; niche.

### Recipe usage

```yaml
# h>1 forecast + HAC-adjusted DM test
path:
  1_data_task:
    leaf_config:
      target: INDPRO
      horizons: [3, 6, 12]
  6_stat_tests:
    fixed_axes:
      overlap_handling: evaluate_with_hac
      equal_predictive: dm_hln
      dependence_correction: nw_hac
```

---

## Horizon & Evaluation Window (1.3) takeaways

- Every value in every 1.3 axis listed here is operational in v1.0. `oos_period` is operational through the Layer 4 `evaluation_spec`.
- `min_train_size` exposes the five rules already implemented in `raw.windowing`, now live in the main execution path.
- `training_start_rule = fixed_start` unlocks calendar-exact replication of published paper samples via `leaf_config.training_start_date`.
- `oos_period` delivers NBER regime conditioning without recipe-side date bookkeeping. Use `path.5_evaluation.fixed_axes.oos_period`; the compiler still accepts the old Layer 1 placement as a compatibility alias.
- `overlap_handling = evaluate_with_hac` wires the HAC requirement into compile-time validation rather than leaving it implicit in stat-test choice. Use `path.6_stat_tests.fixed_axes.overlap_handling`; old Layer 1 placement remains a compatibility alias.

Next group: 1.4 Benchmark & predictor universe (coming).
