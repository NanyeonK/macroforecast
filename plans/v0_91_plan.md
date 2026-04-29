# v0.91 — Planned-status promotion sweep

**Goal:** flip ~95 of the 142 `planned`-status registry values to `operational` (Tier 1 + Tier 2 per the 2026-04-18 kickoff triage). Tier 3 (~47 values) is deferred to a later interim (`v0.92`) or to the Phase 10/11 post-v1.0 catalogs — it requires new infrastructure (probabilistic outputs, real-time vintage, regime-conditional evaluation, orchestration wrappers).

**Scope boundary:** this is an interim release between Phase 7 (`v0.9`) and Phase 8. No new phase files. No plan restructure. Each promotion is:
1. actual code path exists or is added (typically a tiny extension of an existing module);
2. registry entry's `status` flips from `planned` to `operational`;
3. a test exercises the new value and asserts the registry discovery picks it up.

## Tier 1 — lightweight promotions (~35 values)

Expected per-value cost: ≤30 LOC. Most are metadata / formatting / enum-addition where the implementing function already branches cleanly.

| Layer | Axis | Values to promote |
|---|---|---|
| 0_meta | failure_policy | skip_failed_cell |
| 0_meta | registry_type | callable_registry, custom_plugin |
| 0_meta | axis_type | nested_sweep |
| 0_meta | reproducibility_mode | strict_reproducible (expose at recipe level; runtime already supports it) |
| 0_meta | study_mode | orchestrated_bundle_study, replication_override_study |
| 1_data_task | forecast_type | iterated |
| 1_data_task | own_target_lags | exclude, cv_select_lags |
| 1_data_task | warmup_rule | lags_and_factors_warmup, transform_warmup |
| 1_data_task | data_domain | macro_finance |
| 1_data_task | target_family | multiple_macro_series |
| 1_data_task | target_to_target_inclusion | forbid_other_targets_as_X |
| 1_data_task | x_map_policy | shared_X |
| 3_training | logging_level | debug |
| 3_training | cache_policy | feature_cache |
| 3_training | checkpointing | per_model, per_horizon |
| 3_training | execution_backend | joblib |
| 3_training | horizon_modelization | recursive_one_step_model |
| 3_training | y_lag_count | cv_select |
| 4_evaluation | report_style | latex_table, markdown_table |
| 4_evaluation | agg_horizon | report_separately_only |
| 4_evaluation | agg_target | equal_weight |
| 4_evaluation | decomposition_order | marginal_effect_only |
| 4_evaluation | ranking | mean_metric_rank, median_metric_rank, win_count, benchmark_beat_freq, MCS_inclusion_priority |
| 5_output_provenance | artifact_granularity | per_target, per_target_horizon, hierarchical |
| 5_output_provenance | saved_objects | models_only, data_only |

## Tier 2 — module extensions (~60 values)

Expected per-value cost: 50–150 LOC. Requires a real code path but stays within the existing module boundary (no new subsystem).

### 2_preprocessing (22)

- `target_transform`: difference, log, log_difference, growth_rate (4)
- `scaling_policy`: demean_only, unit_variance_only (2)
- `x_missing_policy`: drop_rows, drop_columns, drop_if_above_threshold, missing_indicator (4)
- `x_outlier_policy`: trim, mad_clip, outlier_to_missing (3)
- `target_normalization`: zscore_train_only, robust_zscore (2)
- `x_lag_creation`: fixed_x_lags (cv_selected_x_lags is Tier 3) (1)
- `tcode_application_scope`: apply_tcode_to_target, apply_tcode_to_X, apply_tcode_to_both (3)
- `additional_preprocessing`: hp_filter (1)
- `representation_policy`: tcode_only (1)
- `preprocessing_axis_role`: swept_preprocessing, ablation_preprocessing (2; governance/metadata — Tier 1-ish but kept here because the tag affects executor behavior) — actually Tier 1; move.
- `recipe_mode`: recipe_grid, recipe_ablation (2; same — Tier 1)

→ Adjusted: `preprocessing_axis_role` and `recipe_mode` move to Tier 1 (metadata-only). Tier 2 preprocessing ≈ 18.

### 3_training (6)

- `hp_space_style`: continuous_box, log_uniform (2)
- `embargo_gap`: fixed_gap, horizon_gap (2)
- `data_richness_mode`: factor_plus_lags, selected_sparse_X (2)
- `lookback`: horizon_specific_lookback (1)
- `seed_policy`: multi_seed_average (1)
- `alignment_fairness`: same_split_across_targets, same_split_across_horizons (2)

### 6_stat_tests (9)

- `direction`: mcnemar, roc_comparison (2)
- `equal_predictive`: paired_t_on_loss_diff, wilcoxon_signed_rank (2)
- `multiple_model`: stepwise_mcs, bootstrap_best_model (2)
- `residual_diagnostics`: autocorrelation_of_errors, serial_dependence_loss_diff (2)
- `nested`: forecast_encompassing_nested (1)

### 7_importance (3)

- `importance_temporal`: time_average, rolling_path (2)
- `importance_gradient_path`: coefficient_path (1)

### 1_data_task (8)

- `horizon_target_construction`: future_diff, future_logdiff, cumulative_growth_to_h, annualized_growth_to_h (4)
- `training_start_rule`: fixed_start, post_warmup_start (2)
- `alignment_rule`: last_available, month_to_quarter_average, month_to_quarter_last (3; check: may depend on mixed-frequency infra — if so, move to Tier 3)
- `deterministic_components`: linear_trend (1)
- `min_train_size`: fixed_years (1)
- `overlap_handling`: evaluate_with_hac (1)
- `information_set_type`: pseudo_oos_revised (1)

## Tier 3 — deferred (47 values)

Stays `planned`. Tracked in `plans/v0_92_and_v1_1_catalog.md` (follow-up doc after v0.91 lands). Highlights:

- `density_interval` × 7 + `cpa_instability` × 3 + `test_scope` × 4 — probabilistic forecast output infra.
- `structural_break_segmentation` × 3 + `oos_period` × 2 + `regime_task` × 2 — regime-conditional evaluation.
- `vintage_policy:rolling_vintage` + custom data-source values × 3 + `multi_target_architecture` × 2 — real-time / multi-target orchestration.
- `feature_grouping` × 2, `cv_selected_x_lags`, and other non-core macro-forecasting preprocessing extensions.
- `experiment_unit` × 2 + `study_mode` × (remaining) — wrapper orchestration.
- `agg_time` × 3 + `decomposition_target` × 3 — aggregation / decomposition v2.

## Execution plan

- **Branch**: `feat/v0-91-planned-promotion` (already created 2026-04-18).
- **Commit granularity**: one commit per layer (8 commits max) to keep review tractable. Each commit flips that layer's planned entries + adds/extends the implementing code + adds tests.
- **Test strategy**: every promoted value gets at least one assertion that (a) the registry exposes it as `operational` and (b) the implementing code path handles it without raising.
- **Regression**: after each layer commit, run `pytest -q -x`; full suite green required before moving to the next layer.
- **Acceptance**: all 95 values flipped to `operational`; full suite green; a new `tests/test_v0_91_planned_promotions.py` asserts no Tier-1/Tier-2 value remains `planned`.
- **Release**: single PR → merge → `v0.91` tag.

If a Tier 2 item turns out to require substantially more infrastructure than estimated, it is demoted to Tier 3 with a note in this plan and the axis stays `planned` for v0.91; the session reports the demotion rather than auto-expanding scope.

## Revision log

- 2026-04-18: initial draft, Tier 1/2/3 classification.
- 2026-04-18 (post-kickoff audit): first preprocessing X-side batch committed — 9 values promoted (`scaling_policy`: demean_only / unit_variance_only; `x_outlier_policy`: trim / mad_clip / outlier_to_missing; `x_missing_policy`: drop_rows / drop_columns / drop_if_above_threshold / missing_indicator). Added `tests/test_v0_91_preprocessing_promotions.py` (9 tests, green).

- 2026-04-18 (scope reassessment): Several originally-Tier-1/Tier-2 items turn out to require genuine infrastructure beyond enum + status flip. Demoted to Tier 3 with reasons:
  - `target_transform` × 4 (difference / log / log_difference / growth_rate): needs y-domain transform + inverse-transform pipeline, currently gated by `_is_operational_preprocess_contract` which rejects any non-"level" value.
  - `target_normalization` × 2 (zscore_train_only / robust_zscore): same — y-side normalisation + inverse for metrics.
  - `tcode_application_scope` × 3, `additional_preprocessing:hp_filter`, `representation_policy:tcode_only`: gated by operational-contract checks that enforce default values; relaxation requires pipeline-wide review.
  - `target_outlier_policy` variants, `x_lag_creation`: same pattern.
  - `artifact_granularity` × 3 / `saved_objects` × 2 (layer 5): multi-target output infra + model-pickling support required, both post-v1.0.
  - `checkpointing`, `cache_policy`, `execution_backend`, `logging_level`, `report_style`, `ranking`: compiler stores the selection but no downstream branch differentiates — promoting to "operational" would be misleading. These become "declarative" status promotions only when the consuming code catches up.
  - Most `1_data_task` Tier 1/2 items (`forecast_type:iterated`, `own_target_lags:exclude`, `horizon_target_construction`, `warmup_rule`, `training_start_rule`, `deterministic_components:linear_trend`, etc.): require expansion of the raw-data + target-construction pipeline; deferred.

  **Revised v0.91 scope**: the honest Tier-1/Tier-2 set is ~15–25 values where the runtime branch is either already generic (enum add only) or a bounded extension (≤50 LOC per value). The current commit covers the X-side preprocessing portion; the next candidate batch is the `6_stat_tests` direction / equal_predictive / residual_diagnostics / multiple_model / nested axes (stat test adapters extend cleanly). Further batches will be committed as each confirms clean implementation paths.


- 2026-04-18 (batch 2 committed): 6_stat_tests — 3 values promoted (`equal_predictive`: paired_t_on_loss_diff / wilcoxon_signed_rank; `residual_diagnostics`: autocorrelation_of_errors). Added `tests/test_v0_91_stat_test_promotions.py` (5 tests, green). Cumulative v0.91 promotions: 12 values.

- 2026-04-18 (v0.91 closeout plan): remaining candidates that superficially look easy turn out to need substantial infra (mcnemar/roc_comparison need direction-classification pairing across models; stepwise_mcs / bootstrap_best_model duplicate existing multiple_model tests without clearly different semantics; serial_dependence_loss_diff overlaps with autocorrelation_of_errors; forecast_encompassing_nested overlaps with cw/enc_new). Demoting these to Tier 3 for v0.91 purposes. The v0.91 interim closes at **12 values** (9 preprocessing + 3 stat tests) with a clean acceptance story per batch. Further promotions follow when each domain's owning phase (Phase 10 / 11 / or a targeted interim) delivers the underlying infrastructure.
