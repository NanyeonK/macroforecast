# Stages Reference (0 and 1)

A one-page map of every operational value on every axis in the two completed stages. Use this as a quick cheat sheet: pick a recipe shape from Stage 0, pick data decisions from Stage 1, then cross over to the in-depth user-guide pages for per-axis semantics and recipe YAML examples.

**Scope.** Layer 0 (Design, 0.1–0.6) + Layer 1 (Data, 1.1–1.5). Layer 2+ (preprocessing, training, evaluation, provenance, stat tests, importance) are still in active development and are intentionally out of scope for the user-facing docs.

**At a glance.**

| Stage | Axes | Operational values | What it governs |
|---|---|---|---|
| 0 — Design  | 6 | 31 | Recipe grammar: runner, sweep shape, reproducibility, compute |
| 1 — Data    | 20 | 73 | Dataset + task + forecast object + time windows + benchmark + predictors + break policy |

---

## Stage 0 — Design

### 0.1 `research_design`

**What it picks:** The overall shape of the study — how axes sweep, which runner fires, which artifacts land.

| Value | Check / observe |
|---|---|
| `single_path_benchmark` (default) | Single recipe → `execute_recipe()` → one predictions file + one metrics file |
| `controlled_variation` | Sweep plan → one run per variant, shared baseline contract → `study_manifest.json` |
| `orchestrated_bundle` | Compile-only in v1.0 (Phase 8 `PaperReadyBundle` will consume) — manifest carries `wrapper_handoff` |
| `replication_override` | `execute_replication()` runner → byte-identical re-run vs. source manifest |

**Deep dive:** [user_guide/design.md 0.1](../user_guide/design.md#01-research_design).

### 0.2 `experiment_unit`

**What it picks:** Which runner owns the recipe.

| Value | Check / observe |
|---|---|
| `single_target_single_model` | one target + one model → default derivation |
| `single_target_model_grid` | one target + `model_family` sweep |
| `single_target_full_sweep` | one target + model + feature sweeps |
| `multi_target_separate_runs` | N targets → N independent `execute_recipe` calls (dedicated runner) |
| `multi_target_shared_design` | N targets → one run with shared preprocessing + benchmarks |
| `ablation_study` | `execute_ablation()` — baseline + per-axis reverts |
| `replication_recipe` | `execute_replication()` — source-derived recipe |
| `benchmark_suite` | compile-only (Phase 8) |

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
| `best_effort` (default) | Python/numpy seeded, torch optional |
| `seeded_reproducible` | + torch seed + cudnn.deterministic |
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

### 1.1 Source & Frame — [source.md](../user_guide/data/source.md)

| Axis | Op values | Check / observe |
|---|---|---|
| `dataset` | `fred_md`, `fred_qd`, `fred_sd` | schema loaded → `raw_result.data.columns` |
| `dataset_source` | `fred_md`, `fred_qd`, `fred_sd`, `custom_csv`, `custom_parquet` | which loader fires (`_get_dataset_loader`), `manifest.raw_artifact` |
| `frequency` | `monthly`, `quarterly` | declarative, derived from `dataset` default |
| `information_set_type` | `revised`, `pseudo_oos_revised` | revised = post-revision truth; pseudo-oos masks to simulate real-time |

### 1.2 Task & Target — [task.md](../user_guide/data/task.md)

| Axis | Op values | Check / observe |
|---|---|---|
| `task` | `single_target_point_forecast`, `multi_target_point_forecast` | triggers multi-target aggregator at line 516 in `compiler.build`; drives `experiment_unit` default |
| `forecast_type` | `direct`, `iterated` | **dynamic default** per `feature_builder` (autoreg→iterated, raw_panel→direct) — cross combos blocked |
| `forecast_object` | `point_mean`, `point_median`, `quantile` | `quantile_linear` model compat guard; quantile level via `leaf_config.training_spec.hp.quantile` |
| `horizon_target_construction` | `future_level_y_t_plus_h`, `future_diff`, `future_logdiff` | metric-scale transform at central row site; level-scale values preserved as `y_true_level` / `y_pred_level` |

### 1.3 Horizon & Evaluation Window — [horizon.md](../user_guide/data/horizon.md)

| Axis | Op values | Check / observe |
|---|---|---|
| `min_train_size` | `fixed_n_obs`, `fixed_years`, `model_specific_min_train`, `target_specific_min_train`, `horizon_specific_min_train` | `raw.windowing._resolve_min_train_obs` dispatch; `_minimum_train_size(recipe)` returns the resolved value |
| `training_start_rule` | `earliest_possible`, `fixed_start` | `fixed_start` requires `leaf_config.training_start_date` (compiler guard) — applied as `base_start_idx` floor |
| `oos_period` | `all_oos_data`, `recession_only_oos`, `expansion_only_oos` | NBER fixture filter on `origin_plan`; every `origin_date` in the predictions CSV must match the regime |
| `overlap_handling` | `allow_overlap`, `evaluate_with_hac` | compiler guard requires HAC-capable `stat_test` ∈ {dm_hln, dm_modified, spa, mcs, cw, cpa, none} when `evaluate_with_hac` is set |

### 1.4 Benchmark & Predictor Universe — [benchmark.md](../user_guide/data/benchmark.md)

| Axis | Op values | Check / observe |
|---|---|---|
| `benchmark_family` | `historical_mean`, `ar_bic`, `zero_change`, `custom_benchmark`, `rolling_mean`, `ar_fixed_p`, `ardi`, `factor_model`, `expert_benchmark`, `paper_specific_benchmark`, `survey_forecast`, `multi_benchmark_suite` | `benchmark_pred` column per row; `factor_model` uses leading-factor OLS; `multi_benchmark_suite` reads `leaf_config.benchmark_suite`; paper/survey pull from `leaf_config.paper_forecast_series` / `survey_forecast_series` |
| `predictor_family` | `target_lags_only`, `all_macro_vars`, `category_based`, `factor_only`, `handpicked_set` | `_raw_panel_columns` dispatch; `handpicked_set` requires `leaf_config.handpicked_columns`; `category_based` uses `leaf_config.predictor_category_columns` + `predictor_category` |
| `variable_universe` | `all_variables`, `preselected_core`, `category_subset`, `target_specific_subset`, `handpicked_set` | `_apply_variable_universe`; `handpicked_set` reads `leaf_config.variable_universe_columns`; target + date columns always preserved |
| `deterministic_components` | `none`, `constant_only`, `linear_trend`, `monthly_seasonal`, `quarterly_seasonal`, `break_dummies` | X augmentation via `macrocast.execution.deterministic.augment_array`; `break_dummies` uses `leaf_config.break_dates` |

### 1.5 Data Handling Policies — [policies.md](../user_guide/data/policies.md)

| Axis | Op values | Check / observe |
|---|---|---|
| `missing_availability` | `complete_case_only`, `available_case`, `x_impute_only` | `_apply_missing_availability`; `x_impute_only` requires `leaf_config.x_imputation` ∈ {mean, median, ffill, bfill} |
| `release_lag_rule` | `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag` | `_apply_release_lag`; `series_specific_lag` requires `leaf_config.release_lag_per_series: dict[str, int]` |
| `structural_break_segmentation` | `none`, `pre_post_crisis`, `pre_post_covid` | `_resolve_structural_break_dates` maps to break dates (2008-09-01 / 2020-03-01); feeds `augment_array(component='break_dummies')` (same path as 1.4 deterministic) |
| `contemporaneous_x_rule` | `allow_contemporaneous`, `forbid_contemporaneous` | affects `_build_raw_panel_training_data` X alignment (forbid: X_t paired with y_{t+h}; allow: X_{t+h} oracle benchmark) |

---

## How to verify what you picked

Every value above lands somewhere you can inspect:

1. **`manifest.json`** in the run's artifact directory — all resolved axis values live under `data_task_spec` (Layer 1), `stage0` (Layer 0), plus `research_design`, `experiment_unit`, `reproducibility_applied`, `blocked_reasons`, `warnings`.
2. **`predictions.csv`** — per-row columns honour 1.2.4 (`y_true_level`, `y_pred_level`, `horizon_target_construction`) + 1.3 (filtered rows when `oos_period` regime filter fires).
3. **Compile status** — `compile_result.compiled.execution_status` is one of `executable` / `representable_but_not_executable` / `blocked_by_incompatibility`. `blocked_reasons` lists the specific compile guards that fired.

Start with `manifest.json`; every axis value you set (or let default) is recorded there verbatim.
