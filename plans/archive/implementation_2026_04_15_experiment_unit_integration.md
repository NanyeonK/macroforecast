# Implementation Plan — Experiment Unit Stage 0 Integration

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Add `experiment_unit` as the first Stage 0 meta axis on top of the new per-axis registry architecture, then wire it through compiler, stage0 derivation, and the interactive start flow without breaking legacy recipes.

Smallest honest implementation
1. Add `macrocast/registry/stage0/experiment_unit.py` with:
   - `ExperimentUnitEntry`
   - 12 registry values from `plans/implementation-issues.md`
   - helper metadata maps / accessors for route owner and wrapper requirements
2. Extend Stage 0 build/derive so `build_stage0_frame(..., experiment_unit=...)` is supported.
   - explicit experiment_unit should control route posture
   - legacy callers without experiment_unit should still derive a default from study/task/sweep shape
3. Extend compiler integration.
   - accept optional `0_meta.fixed_axes.experiment_unit`
   - validate explicit axis against the recipe’s implied shape
   - preserve the selected experiment_unit in stage0 and tree_context
4. Extend `macrocast/start.py` wizard.
   - insert `experiment_unit` after `task`
   - offer route-family choices appropriate to current study/task state
   - sync experiment_unit choice back into recipe fields (`study_mode`, `task`, wrapper leafs, internal sweep shape)
5. Add/keep focused tests proving:
   - registry axis exists
   - stage0 frame accepts explicit experiment_unit
   - compiler preserves explicit experiment_unit
   - wizard surfaces experiment_unit after task

Compatibility rule
- Existing recipes that omit `experiment_unit` must still compile.
- When missing, compiler/stage0 derive the default from the current route shape.
- If explicit `experiment_unit` conflicts with current route shape, fail explicitly.

Verification
- red test first: `python3 -m pytest tests/test_stage0.py tests/test_compiler.py tests/test_start.py -q`
- focused green: same command after implementation
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py tests/test_registry_loader.py -q`
