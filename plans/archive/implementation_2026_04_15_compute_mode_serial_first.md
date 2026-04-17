# Implementation Plan — Compute Mode Serial-First Slice

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Add Stage 0 meta-axis `compute_mode` while keeping runtime honestly serial-only for now.

Smallest honest implementation
1. Add `macrocast/registry/stage0/compute_mode.py` with the 8 issue-tracker values.
2. Keep axis optional in recipe YAML.
   - default missing `compute_mode` to `serial`
3. Preserve provenance:
   - `tree_context["compute_mode"]`
   - `manifest["compute_mode_spec"] = {"compute_mode": ...}`
4. Compiler semantics:
   - `serial` remains executable/current behavior
   - all parallel/GPU/distributed modes remain representable-but-not-executable through warnings/status metadata
5. Do not modify runtime execution backend yet.

Verification
- focused: `python3 -m pytest tests/test_registry_loader.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py tests/test_registry_loader.py -q`
