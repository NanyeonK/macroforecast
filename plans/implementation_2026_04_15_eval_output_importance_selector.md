# Implementation Plan — Later Evaluation / Output / Importance Staged Choices

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Extend `macrocast_single_run()` staged selector into the later executable single-run block:
  - evaluation
  - output/provenance
  - stat test
  - importance

Smallest honest implementation
1. Keep existing YAML inspection path unchanged.
2. Append staged choices after current training/preprocessing/model-path block:
   - `primary_metric`
   - `manifest_mode`
   - `stat_test`
   - `importance_method`
3. Keep options grounded in current operational subset:
   - `primary_metric`: `msfe`, `relative_msfe`, `oos_r2`, `csfe`
   - `manifest_mode`: `full`
   - `stat_test`: `none`, `dm`, `cw`
   - `importance_method`: `none`, `minimal_importance`
4. Preserve honest compile feedback after each step.
5. Do not hide runtime incompatibilities:
   - e.g. `minimal_importance` on incompatible model/feature routes should remain visible through preview rather than silently coerced.

Acceptance criteria
- staged selector reaches evaluation/output/importance choices
- YAML rewrites preserve selected values in the correct layers
- route/compile/tree preview still refreshes after every step
- focused and broad tests pass

Verification
- focused: `python3 -m pytest tests/test_start.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py -q`
