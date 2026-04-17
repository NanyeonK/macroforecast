# Implementation Plan — Failure Policy Provenance-First Slice

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Add Stage 0 meta-axis `failure_policy` and establish the smallest honest compiler/runtime contract now: preserve policy provenance and keep current execution semantics equivalent to `fail_fast`/`hard_error` until partial-result execution support is intentionally built.

Smallest honest implementation
1. Add `macrocast/registry/stage0/failure_policy.py` with the 8 issue-tracker values.
2. Keep axis optional in recipe YAML.
   - default missing `failure_policy` to `fail_fast`
3. Compiler/provenance work only for now:
   - `tree_context["failure_policy"]`
   - `manifest["failure_policy_spec"] = {"failure_policy": ...}`
4. Minimal explicit policy validation:
   - `fail_fast` and `hard_error` remain executable/current behavior
   - `skip_failed_cell`, `skip_failed_model`, `save_partial_results`, `retry_then_skip`, `fallback_to_default_hp`, `warn_only` remain representable but not executable via existing status/warning flow
5. Do not modify execution loop control flow yet.

Verification
- focused: `python3 -m pytest tests/test_registry_loader.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py tests/test_registry_loader.py -q`
