# Macrocast 2026-04-15 Phase D Custom Benchmark Bridge Plan

Goal
- Continue Phase D by making `benchmark_family = custom_benchmark` executable through a narrow plugin-ready runtime bridge.

Architecture
- Keep the package benchmark grammar unchanged: path still carries `benchmark_family`, while `benchmark_config` carries free-form benchmark details.
- Add a first executable custom-benchmark bridge that loads a user-specified Python callable from a local module path and invokes it through a strict benchmark contract, without changing model execution semantics.

Tech stack
- Python 3.12
- pytest
- importlib
- pandas / numpy / statsmodels / scikit-learn
- git

Work scope
- In scope:
  - TDD for compiler/runtime support of `custom_benchmark`
  - benchmark plugin contract validation
  - runtime import-and-call bridge from `benchmark_config`
  - docs/roadmap/example updates
  - focused + broader regression + smoke run
- Out of scope:
  - remote package/plugin installation flows
  - benchmark plugin registries beyond direct file-path loading
  - multi-benchmark orchestration
  - model/plugin co-sweep logic

Verification strategy
- Level: test-suite
- Focused command: `python3 -m pytest tests/test_execution_pipeline.py tests/test_compiler.py -q`
- Broader command: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- What this validates:
  - custom benchmark routes compile as executable when the plugin contract is present
  - runtime writes normal prediction/metric artifacts while using the custom benchmark bridge
  - existing benchmark families and evaluation slices still pass regression

## Task 1: Add failing custom-benchmark tests

Dependencies: None
Files:
- Modify: `tests/test_execution_pipeline.py`
- Modify: `tests/test_compiler.py`

- [ ] Add a runtime test that executes a recipe with `benchmark_family="custom_benchmark"` and a plugin config pointing at a local Python file.
- [ ] Assert the run completes and the manifest preserves `benchmark_name == "custom_benchmark"` plus the plugin details in `benchmark_spec`.
- [ ] Add a compiler test asserting a custom-benchmark recipe is executable when `benchmark_config` contains a valid plugin contract.
- [ ] Add a runtime/compile failure test asserting missing plugin contract fields fail explicitly.
- [ ] Run the focused command and confirm RED.

## Task 2: Implement the plugin-ready benchmark bridge

Dependencies: Task 1
Files:
- Modify: `macrocast/registry/build.py`
- Modify: `macrocast/compiler/build.py`
- Modify: `macrocast/execution/build.py`

- [ ] Promote `benchmark_family.custom_benchmark` from external-plugin-only blocking status to the first operational plugin bridge status in registry truth.
- [ ] In compiler validation, require a minimal custom benchmark contract in `benchmark_config`:
  - `plugin_path`
  - `callable_name`
- [ ] In runtime, add a loader that imports a Python module from `plugin_path`, resolves `callable_name`, and calls it with a strict signature based on the current benchmark context.
- [ ] Preserve fail-closed behavior with explicit errors for:
  - missing plugin file
  - missing callable
  - non-numeric return value
- [ ] Re-run the focused command and confirm GREEN.

## Task 3: Update examples/docs/roadmap

Dependencies: Task 2
Files:
- Modify: `examples/recipes/custom-benchmark.yaml`
- Modify: `docs/index.md`
- Modify: `docs/api/index.md`
- Modify: `docs/compiler.md`
- Modify: `docs/execution.md`
- Modify: `plans/plan_04_14_1958.md`

- [ ] Replace the old representable-only custom benchmark example with the new plugin bridge contract.
- [ ] Update docs so benchmark executors now include `custom_benchmark` through the plugin-ready bridge.
- [ ] Document the minimum plugin callable contract and its current limitations honestly.
- [ ] Update the roadmap so Phase D reflects the first custom benchmark bridge as complete for the current slice.

## Task 4: Final verification and checkpoint

Dependencies: Task 3
Files:
- Read-only verification

- [ ] Run the focused command. Expected: pass.
- [ ] Run the broader command. Expected: pass.
- [ ] Run a direct smoke execution using a temporary local plugin file and confirm predictions/metrics plus manifest benchmark provenance are written.
- [ ] Commit with message: `feat: add plugin-ready custom benchmark bridge`
