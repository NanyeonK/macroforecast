# Implementation Plan — Minimal Staged Selector on Top of Route Inspection

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Extend `macrocast_single_run()` from pure YAML-path inspection into a minimal step-by-step YAML-building helper.
- Keep this slice narrow and honest: route-defining choices only, immediate route-preview refresh, no fake downstream execution.

Why this slice
- Public route inspection now exists.
- The next useful step is letting a user build a small valid path incrementally and see how route ownership changes immediately.
- Full archived wizard restoration is still out of scope because the live repo does not contain the old support stack.

Smallest honest implementation
1. Keep existing YAML inspection path unchanged.
2. Add a minimal interactive path when `yaml_path` is omitted.
3. Use one live base recipe template (`examples/recipes/model-benchmark.yaml`) and mutate it in place.
4. First staged selector only covers route-defining choices:
   - `study_mode`
   - `task`
   - `target` or `targets`
   - `model_family`
   - `feature_builder`
   - conditional wrapper metadata when `study_mode='orchestrated_bundle_study'`
5. After every step:
   - rewrite YAML to the chosen file path
   - recompile current recipe dict
   - refresh `route_preview`
6. If route becomes wrapper-required or planned single-run extension:
   - stop the staged flow cleanly
   - return current recipe YAML + explicit stop reason + current route preview
7. Preserve `max_steps` for focused tests.

Acceptance criteria
- `macrocast_single_run()` with no `yaml_path` starts minimal staged selector
- user can choose a YAML output file path
- each completed step rewrites the YAML file
- route preview refreshes immediately after each step
- wrapper-required branch stops with explicit message
- existing YAML inspection path remains unchanged

Verification
- focused: `python3 -m pytest tests/test_start.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py -q`
