# Macrocast 2026-04-15 Phase G Multi-Target Slice Plan

Goal
- Continue Phase G by adding the smallest honest multi-target point-forecast execution slice while keeping the new explicit-vintage real-time support fixed.

Architecture
- Reuse the current single-run execution path and widen it to accept an explicit tuple of targets under one shared study environment.
- Keep the slice narrow: one dataset, one info set, one benchmark family, one model family, one feature-builder family, one preprocessing contract, multiple targets executed sequentially inside one run directory with shared provenance.

Tech stack
- Python 3.12
- pytest
- existing raw loader / execution pipeline
- pandas / numpy
- git

Work scope
- In scope:
  - TDD for `task='multi_target_point_forecast'`
  - recipe/compiler/runtime support for explicit `leaf_config.targets`
  - multi-target prediction/metric/comparison artifacts in one run directory
  - docs and roadmap updates
  - focused and broader regression plus smoke verification
- Out of scope:
  - target-specific model configs
  - heterogeneous benchmark/model mixes across targets
  - multi-target importance artifacts
  - wrapper/orchestrator bundles

Verification strategy
- Level: test-suite
- Focused command: `python3 -m pytest tests/test_recipe_execution.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- Broader command: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- What this validates:
  - multi-target recipes compile and execute in the narrow first slice
  - one run writes shared predictions/metrics/comparison artifacts across multiple targets
  - current single-target revised/real-time routes do not regress

## Task 1: Add failing multi-target tests

Dependencies: None
Files:
- Modify: `tests/test_recipe_execution.py`
- Modify: `tests/test_execution_pipeline.py`
- Modify: `tests/test_compiler.py`

- [ ] Add a recipe/run-spec test asserting multi-target recipes preserve `targets` and generate a deterministic multi-target run id.
- [ ] Add a runtime execution test asserting a multi-target recipe writes predictions for at least two targets in one `predictions.csv`, plus `metrics.json` and `comparison_summary.json` with per-target sections.
- [ ] Add a compile validation test asserting `task='multi_target_point_forecast'` requires `leaf_config.targets` and fails if only `target` is supplied.
- [ ] Add a compile-and-run test asserting a narrow multi-target recipe is executable.
- [ ] Run the focused command and confirm RED.

## Task 2: Implement the first multi-target execution slice

Dependencies: Task 1
Files:
- Modify: `macrocast/recipes/types.py`
- Modify: `macrocast/recipes/construct.py`
- Modify: `macrocast/recipes/build.py`
- Modify: `macrocast/compiler/build.py`
- Modify: `macrocast/execution/build.py`
- Modify: `macrocast/registry/build.py`

- [ ] Promote `task='multi_target_point_forecast'` to operational for the first narrow slice.
- [ ] Extend `RecipeSpec` to preserve `targets: tuple[str, ...]` in addition to the current single-target field.
- [ ] In compiler validation, require `leaf_config.targets` with length >= 2 when the task is multi-target and fail closed otherwise.
- [ ] Build deterministic multi-target run ids and summaries without breaking single-target behavior.
- [ ] In runtime, loop over targets using the existing per-target prediction logic, concatenate predictions, and write per-target metric/comparison sections into `metrics.json` and `comparison_summary.json`.
- [ ] Keep single-target behavior unchanged.
- [ ] Re-run the focused command and confirm GREEN.

## Task 3: Update docs and roadmap

Dependencies: Task 2
Files:
- Modify: `docs/index.md`
- Modify: `docs/api/index.md`
- Modify: `docs/compiler.md`
- Modify: `docs/execution.md`
- Modify: `docs/recipes.md`
- Modify: `plans/plan_04_14_1958.md`

- [ ] Document the first multi-target slice honestly as a shared-environment, shared-model-surface route with explicit `targets`, not a full orchestration system.
- [ ] Update roadmap so Phase G records both explicit-vintage real-time support and the first multi-target execution slice.

## Task 4: Final verification and checkpoint

Dependencies: Task 3
Files:
- Read-only verification

- [ ] Run the focused command. Expected: pass.
- [ ] Run the broader command. Expected: pass.
- [ ] Run a smoke execution with two targets and verify multi-target predictions plus per-target metric/comparison summaries.
- [ ] Commit with message: `feat: add multi-target point forecast slice`
