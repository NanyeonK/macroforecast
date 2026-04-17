# Implementation Plan — Compute Mode Runtime Half

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Finish Issue 1-6 by operationalizing `parallel_by_model` and `parallel_by_horizon` for the current executable surface, while keeping all other compute modes representable-only.

Smallest honest implementation
1. Promote only these compute modes to operational:
   - `serial`
   - `parallel_by_model`
   - `parallel_by_horizon`
2. Keep these non-operational:
   - `parallel_by_oos_date`
   - `parallel_by_trial`
   - `gpu_single`
   - `gpu_multi`
   - `distributed_cluster`
3. Runtime implementation in `macrocast/execution/build.py`:
   - read `compute_mode` from compiler provenance
   - `parallel_by_horizon`: parallelize horizon blocks inside `_build_predictions`
   - `parallel_by_model`: parallelize independent target jobs in `execute_recipe` for the current slice; single-target runs degenerate honestly to one job
   - use `ThreadPoolExecutor` and preserve deterministic output ordering
4. Preserve serial semantics as baseline.
5. Continue to keep GPU/distributed modes provenance-only.

Verification
- focused: `python3 -m pytest tests/test_execution_pipeline.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py tests/test_registry_loader.py -q`
