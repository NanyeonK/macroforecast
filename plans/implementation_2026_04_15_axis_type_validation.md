# Implementation Plan — Axis Type Meta-Axis Validation

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Add Stage 0 meta-axis `axis_type`, then use the existing per-axis `default_policy` metadata to make compiler selection-mode validation explicit without breaking legacy compile behavior.

Smallest honest implementation
1. Add `macrocast/registry/stage0/axis_type.py` with the 7 planned values from `plans/implementation-issues.md`.
2. Add lightweight compiler validation based on each registry entry's `default_policy`:
   - if a `fixed`-policy axis appears in `sweep_axes`, emit explicit compile warning
   - keep recipe representable unless another rule already blocks it
3. Preserve backward compatibility:
   - no existing executable recipes should flip to blocked only because of this governance check
   - warning should surface in `compile_result.manifest["warnings"]`
4. Add focused tests:
   - registry loader sees 27 axes including `axis_type`
   - governance table exposes `axis_type`
   - compiler emits warning when a fixed-policy axis is incorrectly swept

Verification
- focused: `python3 -m pytest tests/test_registry_loader.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py tests/test_registry_loader.py -q`
