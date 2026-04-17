# Implementation Plan — Fixed-vs-Sweep Provenance / Tree Context

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Enrich compile/run artifacts with an explicit tree-context payload that makes fixed-vs-sweep provenance visible without changing current execution behavior.

Why now
- Wrapper-owned routes now have an explicit handoff contract.
- The next architecture gain is not another execution widening slice; it is clearer provenance about what was fixed, what was allowed to vary, and how the current recipe selected each axis.

Scope
- compiler payload
- execution manifest payload
- summary text
- docs / roadmap sync
- focused + broad regression

Non-goals
- no new executable model / benchmark / preprocessing family
- no internal sweep runtime
- no wrapper runtime
- no change to route ownership or execution eligibility

## Target contract

Add deterministic `tree_context` payloads to compiled artifacts and execution artifacts.

`tree_context` should preserve at least:
- Stage 0 route semantics:
  - `study_mode`
  - `design_shape`
  - `execution_posture`
  - `experiment_unit`
  - `route_owner`
- Stage 0 fixed-vs-varying design:
  - `fixed_design`
  - `varying_design`
  - `comparison_contract`
- recipe path selection semantics:
  - `fixed_axes`
  - `sweep_axes`
  - `conditional_axes`
- leaf parameters relevant to the realized path:
  - full `leaf_config`

## Smallest honest implementation

1. Extend `CompiledRecipeSpec` with optional `tree_context`
2. Build `tree_context` in compiler from:
   - `Stage0Frame`
   - `RunSpec`
   - axis selections
   - leaf config
3. Serialize `tree_context` in `compiled_spec_to_dict()`
4. Pass `tree_context` into runtime manifest as top-level provenance
5. Add a compact `tree_context=` line to `summary.txt`
6. Add tests for:
   - compiler manifest preserves fixed/sweep grouping
   - wrapper route still gets `tree_context`
   - execution manifest preserves `tree_context`
7. Update docs + roadmap to make post-wrapper next step current and mark this provenance slice complete

## Acceptance criteria

- compiler emits deterministic `tree_context`
- wrapper-owned compile-only routes preserve `tree_context`
- execution manifest exposes top-level `tree_context`
- summary file includes compact tree-context provenance line
- existing execution behavior unchanged
- focused and broad suites pass

## Verification

Focused
- `python3 -m pytest tests/test_compiler.py tests/test_execution_pipeline.py -q`

Broad
- `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
