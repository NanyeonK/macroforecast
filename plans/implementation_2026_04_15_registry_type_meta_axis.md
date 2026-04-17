# Implementation Plan — Registry Type Meta-Axis

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Add Stage 0 meta-axis `registry_type` and extend `AxisDefinition` with the smallest extra metadata field needed now, without widening the public legacy loader contract more than necessary.

Decision
- Minimal extension now: add `registry_type` to `AxisDefinition` with default `"enum_registry"`.
- Do not widen `AxisRegistryEntry` yet.
- `axis_definition_to_legacy_entry()` remains backward compatible and ignores `registry_type` for now.
- This keeps current callers stable while preparing for later `validation_contract` / `reproducibility_mode` work.

Smallest honest implementation
1. Add `macrocast/registry/stage0/registry_type.py` with the 6 planned values.
2. Extend `AxisDefinition` in `macrocast/registry/base.py`:
   - new field `registry_type`
   - default `"enum_registry"`
3. Keep every existing axis file working by relying on the new default.
4. Add focused tests:
   - registry loader sees 28 axes including `registry_type`
   - governance table exposes `registry_type`
   - `AxisDefinition` carries the new field and defaults to `enum_registry`

Verification
- focused: `python3 -m pytest tests/test_registry_loader.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py tests/test_registry_loader.py -q`
