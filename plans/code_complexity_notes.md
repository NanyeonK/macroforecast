# Code complexity — items found during docs-code audit (2026-04-21)

Flagged during the docs restructure that collapsed `docs/user_guide/*` to Stage 0 + Stage 1 only. None of these are blocking; they accumulate technical debt and should be picked up before the Layer 2+ per-axis walks turn them into bigger problems.

## 1. `macrocast.execution.build` is too large

The module is ~3000 lines and mixes four concerns that should live separately:

- Per-model executor closures (sklearn / deep / factor / quantile_linear, ~20 families).
- Row-building / prediction dispatch (`_build_predictions`, `_rows_for_horizon`, `_compute_origin`).
- Runtime data-task dispatch helpers (`_apply_release_lag`, `_apply_variable_universe`, `_apply_missing_availability`).
- Benchmark executor branches (inside one giant `_run_benchmark_executor`).

**Suggested refactor:** split into `execution/executors/` (per-family), `execution/dispatch/` (data-task helpers), `execution/predictions.py` (row builder), `execution/benchmark.py` (benchmark branches). Keeps `build.py` as the top-level `execute_recipe` orchestrator.

## 2. `_data_task_spec` is a long literal dict with two dynamic defaults

`macrocast.compiler.build._data_task_spec` is a ~30-entry literal dict where most entries are plain `_selection_value(..., default=<string>)` but two are dynamic defaults tied to `feature_builder` / `information_set_type`. The dynamic cases are buried mid-dict and hard to spot.

**Suggested refactor:** pull the two dynamic defaults into named helper constants / functions (`_default_forecast_type(feature_builder)`, `_default_oos_period(framework)`) so the spec dict is uniform.

## 3. `_build_raw_panel_training_data` does four things

After the Stage 1 walk, the function carries:

1. Compute predictor columns (`_raw_panel_columns`, respects `predictor_family`).
2. Time slicing (`X_train` / `y_train` / `X_pred` with contemporaneous_x_rule branches).
3. Preprocess application.
4. Deterministic-component augmentation + structural-break augmentation.

**Suggested refactor:** split into four composable helpers so each axis's dispatch is isolated. Today a change to `contemporaneous_x_rule` risks touching code that also handles augmentation; the two concerns should not share a frame.

## 4. Ad-hoc defaults scattered between compiler and execution

`_PHASE3_DEFAULTS` in `macrocast.execution.build` duplicates some defaults already recorded in `data_task_spec`. When the compiler default and the execution fallback disagree, the latter silently wins.

**Suggested refactor:** make execution-side reads require `recipe.data_task_spec[axis]` and drop `_PHASE3_DEFAULTS` entirely; any missing value should be a bug, not a silent default.

## 5. `_run_benchmark_executor` is a 60-line if-elif ladder

After the Stage 1.4 wiring, the benchmark executor has explicit branches for 12 operational values, including an inline "multi_benchmark_suite" loop that re-dispatches on the same set of labels.

**Suggested refactor:** per-family executor functions (`_bench_historical_mean`, `_bench_factor_model`, ...) keyed in a dict. Mirrors how model executors are organised. Makes the `multi_benchmark_suite` implementation trivially a `for family in members: dispatch[family](train, ...)`.

## 6. `_resolve_structural_break_dates` lives in `execution/build.py`

It's a pure utility that belongs in `execution/deterministic.py` (the module that consumes its output). Same pattern as `nber.filter_origins_by_regime` and `deterministic.augment_array` — each axis's helpers should live in a focused module, not in the 3000-line runner.

**Suggested refactor:** move to `macrocast.execution.deterministic` (or a new `execution.structural_break` module) and re-export from `build.py` if needed.

## Priority ordering

None of these are blockers. Suggested order when Layer 2+ work lands:

1. #4 `_PHASE3_DEFAULTS` drop — tiny change, prevents silent disagreements from growing.
2. #6 `_resolve_structural_break_dates` relocation — 5-minute move.
3. #3 `_build_raw_panel_training_data` split — prerequisite for clean Layer 2 (`target_transform`) wiring.
4. #5 `_run_benchmark_executor` dict-dispatch — needed before Phase 8 paper bundle.
5. #2 `_data_task_spec` named defaults — cosmetic clarity, low urgency.
6. #1 full `execution/build.py` split — biggest refactor, deferred until Layer 2+ doubles the file size.
