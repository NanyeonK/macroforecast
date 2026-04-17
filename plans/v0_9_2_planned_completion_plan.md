# v0.9.2 — Planned-status completion (all 130 remaining)

**Target tag:** `v0.9.2` — interim between v0.9.1 and v1.0 that closes out every `planned` status value, so Phase 8 / 9 can ship v1.0 without dragging the catalog debt.

**Scope:** all 130 `planned`-status registry values still outstanding after v0.9.1. Grouped by common infrastructure blocker; each group is one or more PRs with its own acceptance gate. After v0.9.2 the registry distribution should read **≈ 526 operational / 193 registry_only / 0 planned / 78 future / 12 external_plugin**.

**Non-goals:**

- No `registry_only` → `operational` promotions — those are separate declarative stubs, out of v0.9.2 scope.
- No `future` tier work — that is Phase 11 / v2 territory.
- No Phase 8 / 9 deliverables — paper_ready_bundle + v1.0 cutoff stay post-v0.9.2.

## Execution order (by dependency + risk, lowest to highest)

### Group H — small independent extensions (~28 values, ~1000 LOC, 2–3 PRs)

Warm-up. Low-risk scattered values with 20–50 LOC per axis.

Representative values:

- `scaling_policy`: rank_scale
- `feature_selection_policy`: mutual_information_screen, lasso_select (if not already)
- `dimensionality_reduction_policy`: static_factor (if planned) etc.
- `forecast_object`: quantile, direction
- `deterministic_components`: linear_trend
- `overlap_handling`: evaluate_with_hac
- `min_train_size`: fixed_years
- `information_set_type`: pseudo_oos_revised
- `target_domain`: nonnegative, bounded_0_1 (if planned)
- `x_map_policy`: shared_X
- `data_domain`: macro_finance
- `target_family`: multiple_macro_series
- `warmup_rule`: lags_and_factors_warmup, transform_warmup
- `target_to_target_inclusion`: forbid_other_targets_as_X

**PRs:** `v0.9.2/group-h-small-independents` (≤3 commits, all H values).

**Acceptance:** every value listed exercises a runtime code path that produces a distinct observable output vs the default; status flipped; one smoke test per axis.

### Group B — operational-contract gate relaxations (~15 values, ~500 LOC, 1 PR)

`_is_operational_preprocess_contract` currently rejects anything other than default for several preprocessing meta axes. Relax each gate one field at a time and commit.

Values:

- `tcode_application_scope`: apply_tcode_to_target, apply_tcode_to_X, apply_tcode_to_both
- `additional_preprocessing`: hp_filter
- `representation_policy`: tcode_only
- `preprocessing_axis_role`: swept_preprocessing, ablation_preprocessing
- `recipe_mode`: recipe_grid, recipe_ablation
- `target_outlier_policy`: winsorize, iqr_clip, zscore_clip, mad_clip
- `x_lag_creation`: fixed_x_lags

**PR:** `v0.9.2/group-b-contract-relaxation`.

**Acceptance:** each non-default value passes through contract validation, runtime executes without raising the old `is_operational_preprocess_contract` error, and the new behavior is distinguishable from default in at least one test.

### Group A — y-domain transform pipeline (~12 values, ~1200 LOC, 1 PR)

Biggest single infra unlock. Implement y-side transform with inverse, thread it through metrics.

Values:

- `target_transform`: difference, log, log_difference, growth_rate
- `target_normalization`: zscore_train_only, robust_zscore
- `target_domain`: nonnegative, bounded_0_1, integer_count, probability_target (check which are planned vs registry_only)
- `inverse_transform_policy`: target_only, forecast_scale_only (if planned)

**Architecture:**

- New module `macrocast/preprocessing/y_pipeline.py` with `YTransformPipeline.forward(y) → y_transformed` and `.inverse(y_hat) → y_hat_original_scale`.
- Wire forward into training series construction; wire inverse before metric computation.
- `evaluation_scale` axis controls whether metrics are reported on transformed or original scale (already operational, but now semantically real).
- Determinism: transforms are stateless for log/difference/log_difference/growth_rate; stateful for zscore/robust (compute on train only, apply to test).

**PR:** `v0.9.2/group-a-y-domain-pipeline`.

**Acceptance:** for each y-transform, (a) round-trip y → transform → inverse matches original within 1e-9; (b) metrics computed on original scale match a plain-forecast baseline when transform is `"level"`; (c) running with `target_transform="log_difference"` on positive-valued synthetic data produces finite metrics.

### Group F — spec-consumer gaps (~25 values, ~2500 LOC, 3–4 PRs)

The axes that the compiler stores but nothing downstream branches on. Implement the feature for each so that choosing the value actually changes behavior.

Sub-batches:

**F1 — ranking + report_style (~7 values, ~800 LOC)**

- `ranking`: mean_metric_rank, median_metric_rank, win_count, benchmark_beat_freq, MCS_inclusion_priority
- `report_style`: latex_table, markdown_table (already partly feeds into Phase 8; finalise now)

Implementation site: `macrocast/output/ranking.py` (new) + `macrocast/output/report_formatter.py` (new). Consumed by execute_sweep / execute_ablation when the caller asks for a ranked table.

**F2 — artifact_granularity + saved_objects (~5 values, ~600 LOC)**

- `artifact_granularity`: per_target, per_target_horizon, hierarchical
- `saved_objects`: models_only, data_only

Requires: (a) multi-target artifact separation (per-target subdirs) — read-only pass-through for single-target; (b) model pickling support (`joblib.dump` of the fitted sklearn estimator when `saved_objects=models_only`). Sklearn models are pickle-safe; deep models need a `torch.save(state_dict)` branch.

**F3 — execution meta (~7 values, ~600 LOC)**

- `checkpointing`: per_model, per_horizon
- `cache_policy`: feature_cache
- `execution_backend`: joblib
- `logging_level`: debug
- `seed_policy`: multi_seed_average

Checkpointing writes intermediate state files; cache_policy feeds a feature-level LRU; execution_backend:joblib wraps sweep variants with `joblib.Parallel`.

**F4 — remaining scattered (~6 values, ~500 LOC)**

- `decomposition_order`: marginal_effect_only
- `decomposition_target`: preprocessing_effect, feature_builder_effect, benchmark_effect (tie to Phase 7 decomposition)
- `agg_horizon`: report_separately_only
- `agg_target`: equal_weight

**PRs:** one per sub-batch, `v0.9.2/group-f1-ranking`, `v0.9.2/group-f2-artifacts`, `v0.9.2/group-f3-execution-meta`, `v0.9.2/group-f4-decomp-extensions`.

### Group G — 1_data_task raw pipeline extensions (~15 values, ~1200 LOC, 2 PRs)

Target construction + warmup + training-start-rule diversity.

**G1 — target construction & horizons (~8 values)**

- `forecast_type`: iterated
- `own_target_lags`: exclude, cv_select_lags
- `horizon_target_construction`: future_diff, future_logdiff, cumulative_growth_to_h, annualized_growth_to_h
- `horizon_modelization`: recursive_one_step_model

**G2 — warmup + training start (~7 values)**

- `warmup_rule`: lags_and_factors_warmup, transform_warmup
- `training_start_rule`: fixed_start, post_warmup_start
- `alignment_rule`: last_available, month_to_quarter_average, month_to_quarter_last

**PRs:** `v0.9.2/group-g1-target-construction`, `v0.9.2/group-g2-warmup-start`.

### Group D — regime-conditional evaluation (~10 values, ~800 LOC, 1 PR)

Requires regime-indexing + per-regime metric aggregation.

Values:

- `structural_break_segmentation`: pre_post_crisis, pre_post_covid, user_break_dates
- `oos_period`: recession_only_oos, expansion_only_oos
- `regime_task`: recession_conditioned, expansion_conditioned
- `agg_time`: rolling_average, regime_subsample_average, pre_post_break_average

Architecture: `macrocast/execution/evaluation/regime.py` (stub already planned in Phase 8 sub-task 08.5 — scope it for v0.9.2 and import from Phase 8 when 8.5 lands to avoid double implementation). Per-regime metrics are just per-subset reductions of the existing metrics path.

**PR:** `v0.9.2/group-d-regime-eval`.

### Group C — probabilistic output + density/interval tests (~14 values, ~1500 LOC, 2 PRs)

The heaviest infra item. Needs models to emit intervals / density in addition to point forecasts.

**C1 — probabilistic forecast output (~0 values directly, ~700 LOC infra)**

- Extend `predictions.csv` schema with optional `y_lo`, `y_hi` (interval) and `y_density_params` (distribution summary) columns.
- Conformal interval wrapper over point forecasts as the v1.0 minimum (matches the confo_vol_port pattern in this org). Future density methods (quantile regression, MDN) land in v1.1.

**C2 — density/interval test implementations (~14 values)**

- `density_interval`: PIT_uniformity, berkowitz, kupiec, christoffersen_unconditional, christoffersen_independence, christoffersen_conditional, interval_coverage
- `cpa_instability`: fluctuation_test, chow_break_forecast, cusum_on_loss
- `test_scope`: full_grid_pairwise, benchmark_vs_all, regime_specific_tests, subsample_tests

Each test consumes the new interval columns where applicable.

**PRs:** `v0.9.2/group-c1-probabilistic-output`, `v0.9.2/group-c2-density-interval-tests`.

### Group E — real-time vintage + multi-target (~11 values, ~1800 LOC, 2–3 PRs)

Most architectural. Real-time data adapters + multi-target orchestration.

**E1 — vintage_policy:rolling_vintage + dataset_source trio (~4 values, ~1000 LOC)**

- Add FRED ALFRED adapter (already stubbed in `raw/datasets/`?) — confirm + finish.
- `dataset_source`: fred_api_custom, custom_csv, custom_parquet (generic adapter shells).
- `vintage_policy`: rolling_vintage (vintage-timestamp-indexed pull).

**E2 — multi_target_architecture + predictor_family + alignment_rule (~7 values, ~800 LOC)**

- `multi_target_architecture`: separate_univariate_runs, same_design_different_targets
- `predictor_family`: all_except_target, category_based, factor_only
- `alignment_rule`: (already placed in G — cross-reference)

**PRs:** `v0.9.2/group-e1-vintage`, `v0.9.2/group-e2-multi-target`.

## Summary table

| Group | Values | Est LOC | PRs |
|:---:|:---:|:---:|:---:|
| H | 28 | 1000 | 2–3 |
| B | 15 | 500 | 1 |
| A | 12 | 1200 | 1 |
| F | 25 | 2500 | 3–4 |
| G | 15 | 1200 | 2 |
| D | 10 | 800 | 1 |
| C | 14 | 1500 | 2 |
| E | 11 | 1800 | 2–3 |
| **TOTAL** | **130** | **≈10500** | **14–17** |

## Acceptance gate (v0.9.2 release)

- [ ] Every planned-status value across 8 layers is now `operational`, `registry_only`, `future`, or `external_plugin` — zero `planned` remaining.
- [ ] Each group's per-PR acceptance criteria met (see per-group sections).
- [ ] `tests/test_v0_9_2_planned_completion.py` is added: iterates every axis, asserts no entry carries `status="planned"`.
- [ ] Full pytest suite green.
- [ ] ci-core green on PR merges (sphinx pre-existing failure tolerated, same as v0.9 / v0.9.1 convention).
- [ ] `plans/v0_9_2_planned_completion_plan.md` revision log updated per PR.

## Autonomous-execution decisions (pinned 2026-04-18)

- **Branch strategy:** one long-lived branch `feat/v0-9-2-planned-completion` with per-group sub-branches merged into it. Final PR merges the accumulated branch into main and emits `v0.9.2` tag.
- **Test convention:** one `tests/test_v0_9_2_<group>.py` per group; aggregated `test_v0_9_2_no_planned_remaining.py` gates the release.
- **Demotion protocol:** if a Group turns out to need >2× the estimated LOC (e.g., Group C's interval output balloons to 3000 LOC), the group is re-partitioned: the straightforward subset ships in v0.9.2, the ambitious subset moves to Phase 10.
- **Sphinx warnings:** no new warnings may be introduced; pre-existing 5 are tolerated.
- **Registry integrity:** every flipped value must have at least one test that exercises its code path and asserts a distinguishable effect vs default. No blind status flips.

## Revision log

- 2026-04-18: initial plan drafted after option-1 green light; scope = 130 values, 14–17 PRs, ~10 500 LOC estimate.
