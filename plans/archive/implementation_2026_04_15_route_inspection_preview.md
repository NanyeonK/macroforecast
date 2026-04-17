# Implementation Plan — Route Inspection Preview Slice

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Add a minimal public `macrocast_single_run()` surface for route inspection on top of the live compiler/runtime contracts.
- Keep this slice honest: route/tree inspection first, not a fake full wizard restore.

Why this slice
- Live repo now has explicit `wrapper_handoff` and `tree_context` provenance.
- What is still missing is a public single-run entry point that lets a user inspect:
  - route owner
  - execution status
  - fixed environment
  - varying axes / sweep presence
  - whether downstream run/manifest previews are allowed
- The archived reboot snapshot had a much larger wizard surface, but the current live repo no longer includes the required support modules (`choice_stack`, `meta`, `output`, legacy compiler path). Restoring that whole stack would be dishonest scope creep.

Smallest honest implementation
1. Add `macrocast/start.py` with a narrow `macrocast_single_run()` built on the live compiler.
2. Accept existing YAML recipe paths and expose stage-based inspection only.
3. Stage set:
   - `route_preview`
   - `compile_preview`
   - `tree_context`
   - `runs_preview`
   - `manifest_preview`
4. Default behavior when `yaml_path` is supplied:
   - run route preview
   - run compile preview
   - run tree-context preview
   - only expose run/manifest previews if compiled status is `executable`
5. For wrapper-required or representable-but-not-executable routes:
   - do not pretend runs/manifest previews are runnable
   - return explicit blocked stage list + reason
6. Export `macrocast_single_run` from package root.
7. Add focused tests for:
   - executable single-run recipe preview
   - controlled-variation single-run extension preview
   - wrapper-required preview
8. Update public docs/API index to include `macrocast.start` and the new route-inspection semantics.

Acceptance criteria
- `macrocast_single_run(yaml_path=...)` works against live recipe YAMLs
- executable route returns route/compile/tree context plus run+manifest previews
- planned single-run extension returns route/compile/tree context but blocks run+manifest previews
- wrapper-required route returns route/compile/tree context plus explicit blocked-stage reason
- no hidden execution occurs

Verification
- focused: `python3 -m pytest tests/test_start.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py -q`
