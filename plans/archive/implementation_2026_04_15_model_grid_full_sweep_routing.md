# Implementation Plan — Model Grid / Full Sweep Routing UX

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Surface `single_target_model_grid` and `single_target_full_sweep` directly inside the staged selector with explicit route explanations.

Smallest honest implementation
1. Add staged key `model_path_mode` before `model_family` / `feature_builder`.
2. Supported staged values:
   - `single_model`
   - `model_grid`
   - `full_sweep`
3. Map these values into live recipe grammar by mutating `3_training.fixed_axes` / `3_training.sweep_axes`:
   - `single_model` → fixed `model_family` and fixed `feature_builder`
   - `model_grid` → sweep `model_family`, fixed `feature_builder`
   - `full_sweep` → sweep `model_family` and sweep `feature_builder`
4. Extend route-preview explanation so internal sweep paths distinguish:
   - model grid
   - full sweep
   - generic planned branch
5. Stop the staged flow early once `model_grid` or `full_sweep` is selected, with explicit planned single-run extension messaging.

Acceptance criteria
- staged selector exposes `model_path_mode`
- selecting `model_grid` rewrites YAML with `model_family` under `sweep_axes`
- selecting `full_sweep` rewrites YAML with both `model_family` and `feature_builder` under `sweep_axes`
- route preview message explicitly names `model grid` vs `full sweep`
- focused and broad tests pass

Verification
- focused: `python3 -m pytest tests/test_start.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py -q`
