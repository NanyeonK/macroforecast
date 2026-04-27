# Stages Reference (0 and 1)

A one-page map of every operational value on every axis in the first two
completed user-facing stages. Use this as a quick cheat sheet: pick a recipe
shape from Stage 0, pick data decisions from Stage 1, then cross over to the
in-depth user-guide pages for per-axis semantics and recipe YAML examples.

**Scope.** Layer 0 (Design, 0.1–0.6) + Layer 1 (Data, 1.1–1.5).
Layer 2 full fixed recipes are documented in
`docs/detail/preprocessing_layer_audit.md`; public preprocessing sweeps remain
blocked. Layers 3+ are outside this quick-reference page.

**At a glance.**

| Stage | Axes | Allowed values | What it governs |
|---|---|---|---|
| 0 — Design  | 6 | 38 | Recipe grammar: runner, sweep shape, reproducibility, compute |
| 1 — Data    | 12 | 38 | Official data frame: dataset, source, frequency, information set, target structure, FRED-SD panel policy, availability |

---

## Stage 0 — Design

### 0.1 `research_design`

**What it picks:** The overall shape of the study — how axes sweep, which runner fires, which artifacts land.

| Value | Check / observe |
|---|---|
| `single_forecast_run` (default) | Single recipe → `execute_recipe()` → one predictions file + one metrics file |
| `controlled_variation` | Sweep plan → one run per variant, shared baseline contract → `study_manifest.json` |
| `study_bundle` | Compile-only in v1.0 (Phase 8 `PaperReadyBundle` will consume) — manifest carries `wrapper_handoff` |
| `replication_recipe` | `execute_replication()` runner → byte-identical re-run vs. source manifest |

**Deep dive:** [user_guide/design.md 0.1](../user_guide/design.md#01-research_design).

### 0.2 `experiment_unit`

**What it picks:** Which runner owns the recipe.

| Value | Check / observe |
|---|---|
| `single_target_single_generator` | one target + one model → default derivation |
| `single_target_generator_grid` | one target + `model_family` sweep |
| `single_target_full_sweep` | registry-only; dropped until a wrapper runner exists |
| `multi_target_separate_runs` | N targets → N independent `execute_recipe` calls (dedicated runner) |
| `multi_target_shared_design` | N targets → one run with shared preprocessing + benchmarks |
| `ablation_study` | registry-only compiled-wrapper route; standalone `execute_ablation()` uses `AblationSpec` |
| `replication_recipe` | `execute_replication()` — source-derived recipe |
| `benchmark_suite` | registry-only; dropped until a wrapper runner exists |

**Deep dive:** [user_guide/design.md 0.2](../user_guide/design.md#02-experiment_unit).

### 0.3 `axis_type`

**What it picks:** How a given axis is consumed at compile time (applies per axis, not per recipe).

| Value | Check / observe |
|---|---|
| `fixed` (default) | single value, manifest records it verbatim |
| `sweep` | list of values → each variant in the sweep plan |
| `nested_sweep` | nested list structure — one axis sweeps, dependent axis follows per variant |
| `conditional` | value chosen via rule (`apply_rule_value`) |
| `derived` | value inferred from other axes at compile time |

### 0.4 `failure_policy`

**What it picks:** What happens when a variant / cell fails.

| Value | Check / observe |
|---|---|
| `fail_fast` (default) | first failure aborts the run |
| `skip_failed_cell` | variant failure logged → `manifest.failed_components` → study continues |
| `skip_failed_model` | same pattern at model scope |
| `save_partial_results` | flush artifacts before aborting |
| `warn_only` | RuntimeWarning emitted, run continues |

### 0.5 `reproducibility_mode`

**What it picks:** How aggressive the seed/deterministic controls are.

| Value | Check / observe |
|---|---|
| `seeded_reproducible` (default) | Python/numpy/torch seeded; no strict deterministic-library flags |
| `best_effort` | Same seed application, labeled non-strict for CI/regression interpretation |
| `strict_reproducible` | + `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| `exploratory` | no seed discipline (research drafting) |

`manifest.reproducibility_applied` records the resolved config; `PYTHONHASHSEED` not set → `RuntimeWarning`.

### 0.6 `compute_mode`

**What it picks:** Which level of the sweep is parallelised.

| Value | Check / observe |
|---|---|
| `serial` (default) | straight sequential loop |
| `parallel_by_model` | variant-level `ThreadPoolExecutor` (model_family sweep) |
| `parallel_by_horizon` | horizon-level thread pool inside `_rows_for_horizon` |
| `parallel_by_target` | target-level pool (multi_target only) |
| `parallel_by_oos_date` | origin-level pool (`_rows_for_horizon` stage 2) |

---

## Stage 1 — Data

### 1.1 Official Data Frame — [source.md](../user_guide/data/source.md)

| Axis | Op values | Check / observe |
|---|---|---|
| `dataset` | `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd` | schema loaded/merged → `raw_result.data.columns`; standalone FRED-SD requires explicit `frequency` |
| `source_adapter` | `fred_md`, `fred_qd`, `fred_sd`, `custom_csv`, `custom_parquet` | which loader fires (`_load_raw_for_recipe`), `manifest.raw_artifact`; legacy `dataset_source` is accepted as an alias |
| `frequency` | `monthly`, `quarterly` | conversion target; MD+SD must be monthly, QD+SD must be quarterly |
| `information_set_type` | `final_revised_data`, `pseudo_oos_on_revised_data` | revised = post-revision truth; pseudo-oos masks to simulate real-time |
| `target_structure` | `single_target`, `multi_target` | target cautoregressive_diffusion_indexnality; legacy `task` is accepted as an alias; Layer 0 derives `experiment_unit` from this plus sweep shape |
| `fred_sd_frequency_policy` | `report_only`, `allow_mixed_frequency`, `reject_mixed_known_frequency`, `require_single_known_frequency` | FRED-SD selected-panel native-frequency gate; strict modes consume `fred_sd_frequency_report_v1` before Layer 2 |
| `fred_sd_state_group` | `all_states`, Census regions/divisions, `contiguous_48_plus_dc`, `custom_state_group` | FRED-SD recipe-level state bundle; non-default values resolve to `state_selection=selected_states` before loading |
| `fred_sd_variable_group` | `all_sd_variables`, economic/t-code-review groups, `custom_sd_variable_group` | FRED-SD recipe-level workbook-variable bundle; non-default values resolve to `sd_variable_selection=selected_sd_variables` before loading |
| `state_selection` | `all_states`, `selected_states` | FRED-SD source-load state selector; `selected_states` reads `leaf_config.sd_states` |
| `sd_variable_selection` | `all_sd_variables`, `selected_sd_variables` | FRED-SD source-load workbook-sheet selector; `selected_sd_variables` reads `leaf_config.sd_variables` |
| `variable_universe` | `all_variables`, `core_variables`, `category_variables`, `target_specific_variables`, `explicit_variable_list` | `_apply_variable_universe`; `explicit_variable_list` reads `leaf_config.variable_universe_columns`; target + date columns always preserved |
| `missing_availability` | `zero_fill_leading_predictor_gaps`, `require_complete_rows`, `keep_available_rows`, `impute_predictors_only` | default `zero_fill_leading_predictor_gaps` reports/fills predictor leading gaps; `impute_predictors_only` requires `leaf_config.x_imputation` ∈ {mean, median, ffill, bfill} |
| `release_lag_rule` | `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag` | `_apply_release_lag`; `series_specific_lag` requires `leaf_config.release_lag_per_series: dict[str, int]` |
| `contemporaneous_x_rule` | `allow_same_period_predictors`, `forbid_same_period_predictors` | affects `_build_raw_panel_training_data` X alignment (forbid: X_t paired with y_{t+h}; allow: X_{t+h} oracle benchmark) |

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

1. **`manifest.json`** in the run's artifact directory — official data choices live under `data_task_spec` (Layer 1), while migrated model/preprocessing/evaluation choices live under their canonical specs.
2. **`predictions.csv`** — per-row columns show realised targets, forecasts, benchmark forecasts, horizon labels, and any evaluation-time transformed columns.
3. **Compile status** — `compile_result.compiled.execution_status` is one of `executable`, `ready_for_sweep_runner`, `ready_for_wrapper_runner`, `ready_for_replication_runner`, `not_supported`, or `blocked_by_incompatibility`. `blocked_reasons` lists hard compile guards; `warnings` lists runner handoff or unsupported-route context.

Start with `manifest.json`; every axis value you set (or let default) is recorded there verbatim.
