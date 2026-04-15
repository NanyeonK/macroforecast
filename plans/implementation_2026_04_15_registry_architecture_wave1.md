# Implementation Plan — Registry Architecture Refactor Wave 1

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Start the full option-universe build-out by completing the first registry-architecture slice from `plans/implementation-issues.md`: typed registry base classes, per-stage registry package skeleton, auto-discovery loader, and migration of the current live 25 axes out of the monolithic `_AXIS_REGISTRY` dict.

Smallest honest implementation
1. Add `macrocast/registry/base.py` with typed registry base classes:
   - `BaseRegistryEntry`
   - `EnumRegistryEntry`
   - `AxisDefinition`
   - helper to convert an `AxisDefinition` back into the legacy `AxisRegistryEntry` shape
2. Expand `macrocast/registry/types.py` only as needed for the new registry architecture while preserving the existing public `AxisRegistryEntry` contract.
3. Create per-stage registry package directories:
   - `stage0`
   - `data`
   - `preprocessing`
   - `training`
   - `evaluation`
   - `output`
   - `tests`
   - `importance`
4. Replace the monolithic registry loader in `macrocast/registry/build.py` with module auto-discovery that imports `AXIS_DEFINITION` from per-axis files and converts each definition into the legacy `AxisRegistryEntry` return type.
5. Migrate the current live 25 axes into one file per axis under the new stage directories without changing statuses, values, or compiler-facing names.
6. Add a focused registry-loader test file so the refactor is pinned beyond the current behavior-only suite.

Acceptance criteria
- `AxisRegistryEntry` callers still receive the same fields and value sets as before
- `get_axis_registry()`, `get_axis_registry_entry()`, and `axis_governance_table()` remain backward compatible
- monolithic `_AXIS_REGISTRY` dict removed from `macrocast/registry/build.py`
- all existing axes load from per-axis files
- focused registry/compiler tests pass
- broad regression passes

Verification
- focused: `python3 -m pytest tests/test_registry_loader.py tests/test_compiler.py tests/test_start.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py tests/test_registry_loader.py -q`
