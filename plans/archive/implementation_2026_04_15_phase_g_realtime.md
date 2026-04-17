# Macrocast 2026-04-15 Phase G Real-Time Vintage Slice Plan

Goal
- Start Phase G with the smallest honest execution-surface widening slice: make `info_set = real_time` executable for single-target studies through explicit vintage requests.

Architecture
- Reuse the existing raw-layer vintage semantics instead of inventing a new multi-target orchestration layer.
- Add a narrow real-time route where recipe compilation/execution accepts one explicit `data_vintage` and passes it through to the raw loader, while keeping all current model/benchmark/preprocessing/comparison/importance behavior fixed.

Tech stack
- Python 3.12
- pytest
- existing raw loader/version request machinery
- git

Work scope
- In scope:
  - TDD for executable `info_set='real_time'`
  - compiler validation for explicit vintage requirement
  - runtime pass-through of explicit vintage into raw loaders
  - docs and roadmap updates
  - focused and broader regression plus smoke verification
- Out of scope:
  - multi-target execution
  - real-time rolling vintage sequences
  - live historical vintage discovery automation
  - new benchmark/preprocessing/importance semantics

Verification strategy
- Level: test-suite
- Focused command: `python3 -m pytest tests/test_raw.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- Broader command: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- What this validates:
  - real-time recipes compile as executable when an explicit vintage is supplied
  - runtime forwards the vintage to raw loading and preserves provenance
  - current revised-data routes do not regress

## Task 1: Add failing real-time vintage tests

Dependencies: None
Files:
- Modify: `tests/test_execution_pipeline.py`
- Modify: `tests/test_compiler.py`

- [ ] Add a runtime test that builds a recipe with `info_set='real_time'` and explicit `data_vintage='2020-01'`, executes it against a fixture, and asserts manifest/raw provenance preserve vintage mode.
- [ ] Add a compile-and-run test asserting a real-time recipe with explicit vintage compiles as executable and runs successfully.
- [ ] Add a compile validation test asserting `info_set='real_time'` without explicit vintage fails closed.
- [ ] Run the focused command and confirm RED.

## Task 2: Implement explicit-vintage real-time support

Dependencies: Task 1
Files:
- Modify: `macrocast/recipes/types.py`
- Modify: `macrocast/recipes/construct.py`
- Modify: `macrocast/compiler/build.py`
- Modify: `macrocast/execution/build.py`
- Modify: `macrocast/registry/build.py`

- [ ] Promote `info_set.real_time` from planned to operational for the first explicit-vintage slice.
- [ ] Extend `RecipeSpec` to preserve one explicit `data_vintage` value for runtime use.
- [ ] In compiler validation, require `leaf_config.data_vintage` when `info_set='real_time'`.
- [ ] In runtime raw loading, pass `data_vintage` through to `load_fred_md` / `load_fred_qd` / `load_fred_sd` so metadata/artifact provenance switch to `version_mode='vintage'`.
- [ ] Keep fail-closed behavior if real-time is requested without an explicit vintage.
- [ ] Re-run the focused command and confirm GREEN.

## Task 3: Update docs and roadmap

Dependencies: Task 2
Files:
- Modify: `docs/index.md`
- Modify: `docs/api/index.md`
- Modify: `docs/raw.md`
- Modify: `docs/compiler.md`
- Modify: `docs/execution.md`
- Modify: `plans/plan_04_14_1958.md`

- [ ] Document the first real-time slice honestly as explicit-vintage support, not a full historical real-time panel engine.
- [ ] Update roadmap so Phase G records the first executable real-time slice as complete while multi-target remains future work.

## Task 4: Final verification and checkpoint

Dependencies: Task 3
Files:
- Read-only verification

- [ ] Run the focused command. Expected: pass.
- [ ] Run the broader command. Expected: pass.
- [ ] Run a smoke execution with `info_set='real_time'` and `data_vintage='2020-01'` and verify manifest/raw metadata preserve the vintage request.
- [ ] Commit with message: `feat: add explicit-vintage real-time slice`
