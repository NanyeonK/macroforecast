# Stages Reference (0 and 1)

A one-page map of every operational value on every axis in the first two
completed user-facing stages. Use this as a quick cheat sheet: pick a recipe
shape from Stage 0, pick data decisions from Stage 1, then cross over to the
in-depth user-guide pages for per-axis semantics and recipe YAML examples.

**Scope.** Layer 0 (Design: four user-facing axes plus the internal `axis_type` grammar) + Layer 1 (Data, 1.1–1.5).
Layer 2 full fixed recipes are documented in
`docs/detail/preprocessing_layer_audit.md`; public preprocessing sweeps remain
blocked. Layers 3+ are outside this quick-reference page.

Simple exposes only Layer 0 Study Scope. Full exposes all four user-facing Layer 0 axes; `failure_policy`, `reproducibility_mode`, and `compute_mode` can be omitted to use their defaults.

**At a glance.**

| Stage | Axes | Allowed values | What it governs |
|---|---|---|---|
| 0 — Design  | 4 user-facing + internal `axis_type` | see below | Study scope, failure handling, reproducibility, compute layout, and YAML sweep grammar |
| 1 — Data    | primary Navigator axes plus hidden FRED-SD helper selectors | all current values operational | Source frame, forecast-time information, Target (y) Definition, Predictor (x) Definition, raw cleaning, official transforms, availability |

---

## Stage 0 — Design

### 0.1 `study_scope`

**What it picks:** Target cardinality and whether the method path is fixed or compared.

| Value | Check / observe |
|---|---|
| `one_target_one_method` | one target + one fixed method path -> one `comparison_sweep` cell |
| `one_target_compare_methods` | one target + one or more method sweeps -> `compile_sweep_plan()` / `execute_sweep()` |
| `multiple_targets_one_method` | multiple targets + one fixed method path |
| `multiple_targets_compare_methods` | multiple targets + one or more method sweeps |

**Deep dive:** [user_guide/design.md 0.1](../user_guide/design.md#01-study_scope).

### 0.2 `axis_type`

**What it picks:** How a given axis is consumed at compile time (applies per axis, not per recipe).

| Value | Check / observe |
|---|---|
| `fixed` (default) | single value, manifest records it verbatim |
| `sweep` | list of values → each variant in the sweep plan |
| `nested_sweep` | nested list structure — one axis sweeps, dependent axis follows per variant |
| `conditional` | value chosen via rule (`apply_rule_value`) |
| `derived` | value inferred from other axes at compile time |

### 0.3 `failure_policy`

**What it picks:** What happens when a variant / cell fails.

**Default:** `fail_fast`. This is already selected when the recipe omits the axis.

| Value | Check / observe |
|---|---|
| `fail_fast` (default) | first failure aborts the run |
| `skip_failed_cell` | variant failure logged → `manifest.failed_components` → study continues |
| `skip_failed_model` | same pattern at model scope |
| `save_partial_results` | flush artifacts before aborting |
| `warn_only` | RuntimeWarning emitted, run continues |

### 0.4 `reproducibility_mode`

**What it picks:** How aggressive the seed/deterministic controls are.

**Default:** `seeded_reproducible` with seed `42`. This is already selected when the recipe omits the axis.

| Value | Check / observe |
|---|---|
| `seeded_reproducible` (default) | Python/numpy/torch seeded; no strict deterministic-library flags |
| `best_effort` | Same seed application, labeled non-strict for CI/regression interpretation |
| `strict_reproducible` | + `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| `exploratory` | no seed discipline (research drafting) |

`manifest.reproducibility_applied` records the resolved config; `PYTHONHASHSEED` not set → `RuntimeWarning`.

### 0.5 `compute_mode`

**What it picks:** How execution work is laid out.

**Default:** `serial`. This is already selected when the recipe omits the axis.

| Value | Check / observe |
|---|---|
| `serial` (default) | straight sequential loop |
| `parallel_by_model` | variant-level `ThreadPoolExecutor` (model_family sweep) |
| `parallel_by_horizon` | horizon-level thread pool inside `_rows_for_horizon` |
| `parallel_by_target` | target-level pool (multi_target only) |
| `parallel_by_oos_date` | origin-level pool (`_rows_for_horizon` stage 2) |

---

## Stage 1 — Data

### 1.1 Source, Frequency, And Forecast-Time Information — [source.md](../user_guide/data/source.md)

| Axis | Op values | Check / observe |
|---|---|---|
| `custom_source_policy` | `official_only`, `custom_panel_only`, `official_plus_custom` | Data Source Mode; first source decision; custom files require `leaf_config.custom_source_path`; parser/schema are inferred |
| `dataset` | `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd` | FRED Source Panel; active only when source mode uses FRED data; standalone FRED-SD requires explicit `frequency` |
| `custom_source_path` | local `.csv`, `.parquet`, or `.pq` path | leaf_config payload, not a Navigator axis; file rows must match selected `frequency` |
| `frequency` | `monthly`, `quarterly` | Analysis Frequency; inferred for FRED-MD/QD/composites, required for FRED-SD and custom-only |
| `information_set_type` | `final_revised_data`, `pseudo_oos_on_revised_data` | Data Revision / Vintage Regime; not the publication-lag rule |
| `release_lag_rule` | `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag` | Publication Lag Rule; `_apply_release_lag`; `series_specific_lag` requires `leaf_config.release_lag_per_series: dict[str, int]` |
| `contemporaneous_x_rule` | `allow_same_period_predictors`, `forbid_same_period_predictors` | Same-Period Predictor Rule; affects `_build_raw_panel_training_data` X alignment |
| `target_structure` | `single_target`, `multi_target` | Target (y) Definition; constrained by Layer 0 Study Scope |
| `variable_universe` | `all_variables`, `core_variables`, `category_variables`, `target_specific_variables`, `explicit_variable_list` | Predictor (x) Universe; `_apply_variable_universe`; explicit lists read `leaf_config.variable_universe_columns` |
| `fred_sd_frequency_policy` | `report_only`, `allow_mixed_frequency`, `reject_mixed_known_frequency`, `require_single_known_frequency` | FRED-SD selected-panel native-frequency gate; strict modes consume `fred_sd_frequency_report_v1` before Layer 2 |
| `fred_sd_state_group` | `all_states`, Census regions/divisions, `contiguous_48_plus_dc`, `custom_state_group` | FRED-SD recipe-level state bundle; non-default values resolve to `state_selection=selected_states` before loading |
| `fred_sd_variable_group` | `all_sd_variables`, economic/t-code-review groups, `custom_sd_variable_group` | FRED-SD recipe-level workbook-variable bundle; non-default values resolve to `sd_variable_selection=selected_sd_variables` before loading |
| `state_selection` | `all_states`, `selected_states` | FRED-SD source-load state selector; `selected_states` reads `leaf_config.sd_states` |
| `sd_variable_selection` | `all_sd_variables`, `selected_sd_variables` | FRED-SD source-load workbook-sheet selector; `selected_sd_variables` reads `leaf_config.sd_variables` |
| `raw_missing_policy` | `preserve_raw_missing`, `zero_fill_leading_predictor_missing_before_tcode`, `impute_raw_predictors`, `drop_raw_missing_rows` | raw-source missing treatment before FRED transforms/T-codes; `impute_raw_predictors` reads `leaf_config.raw_x_imputation` |
| `raw_outlier_policy` | `preserve_raw_outliers`, `winsorize_raw`, `iqr_clip_raw`, `mad_clip_raw`, `zscore_clip_raw`, `set_raw_outliers_to_missing` | raw-source outlier treatment before FRED transforms/T-codes; optional `leaf_config.raw_outlier_columns` limits affected columns |
| `official_transform_policy` | `apply_official_tcode`, `keep_official_raw_scale` | official FRED transform-code application policy |
| `official_transform_scope` | `target_only`, `predictors_only`, `target_and_predictors`, `none` | columns that receive official transforms when policy applies |
| `missing_availability` | `zero_fill_leading_predictor_gaps`, `require_complete_rows`, `keep_available_rows`, `impute_predictors_only` | Frame Availability Policy; default reports/fills predictor leading gaps; `impute_predictors_only` requires `leaf_config.x_imputation` |

Layer 2 FRED-SD follow-up:

| Axis | Values | What it governs |
|---|---|---|
| `fred_sd_mixed_frequency_representation` | `calendar_aligned_frame`, `drop_unknown_native_frequency`, `drop_non_target_native_frequency`, operational-narrow `native_frequency_block_payload`, operational-narrow `mixed_frequency_model_adapter` | Post-Layer-1 FRED-SD panel shaping before representation construction; runtime writes `fred_sd_mixed_frequency_representation.json`, and advanced routes write native-frequency block / adapter artifacts for registered custom models |

Moved out of Layer 1:

- Layer 2: `horizon_target_construction`, `deterministic_components`, `structural_break_segmentation`
- Layer 2: `predictor_family`, `feature_builder`, `data_richness_mode`, `factor_count`
- Layer 3: `benchmark_family`, `forecast_type`, `forecast_object`, `min_train_size`, `training_start_rule`
- Layer 4: `oos_period`
- Layer 6: `overlap_handling`

---

## How to verify what you picked

Every value above lands somewhere you can inspect:

1. **`manifest.json`** in the run's artifact directory — FRED data choices live under `data_task_spec` (Layer 1), while migrated model/preprocessing/evaluation choices live under their canonical specs.
2. **`predictions.csv`** — per-row columns show realised targets, forecasts, benchmark forecasts, horizon labels, and any evaluation-time transformed columns.
3. **Compile status** — `compile_result.compiled.execution_status` is one of `executable`, `ready_for_sweep_runner`, `ready_for_wrapper_runner`, `ready_for_replication_runner`, `not_supported`, or `blocked_by_incompatibility`. `blocked_reasons` lists hard compile guards; `warnings` lists runner handoff or unsupported-route context.

Start with `manifest.json`; every axis value you set (or let default) is recorded there verbatim.
