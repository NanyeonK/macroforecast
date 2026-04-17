# Implementation Plan — Failure Policy Runtime Half

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Finish Issue 1-5 by adding the smallest real runtime support for `skip_failed_model` and `save_partial_results`, with honest partial-manifest marking and no fake support for finer-grained policies like `skip_failed_cell`.

Smallest honest implementation
1. Promote only these `failure_policy` values to operational:
   - `fail_fast`
   - `hard_error`
   - `skip_failed_model`
   - `save_partial_results`
2. Keep these values non-operational:
   - `skip_failed_cell`
   - `retry_then_skip`
   - `fallback_to_default_hp`
   - `warn_only`
3. Runtime support in `macrocast/execution/build.py`:
   - read `failure_policy` from compiler provenance
   - wrap per-target prediction build inside try/except
   - for `skip_failed_model` / `save_partial_results`, record target-level failure and continue
   - if all targets fail, still raise `ExecutionError`
4. `save_partial_results` additionally tolerates post-prediction artifact failures for optional layers:
   - stat-test artifact generation
   - importance artifact generation
   - record those failures in manifest and continue if core predictions/metrics exist
5. Partial-manifest semantics:
   - `partial_run: bool`
   - `successful_targets: list[str]`
   - `failed_components: list[dict]`
   - optional `failure_log_file`
6. Do not claim support for per-cell/date continuation yet.

Verification
- focused: `python3 -m pytest tests/test_execution_pipeline.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py tests/test_registry_loader.py -q`
