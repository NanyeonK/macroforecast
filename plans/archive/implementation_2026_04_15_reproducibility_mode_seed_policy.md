# Implementation Plan — Reproducibility Mode Seed Policy

Date
- 2026-04-15

Repo
- `~/project/macroforecast`

Goal
- Add Stage 0 meta-axis `reproducibility_mode`, enforce compile-time seed-policy validation, and preserve the chosen mode in compiler/manifest provenance before changing execution behavior.

Smallest honest implementation
1. Add `macrocast/registry/stage0/reproducibility_mode.py` with:
   - `strict_reproducible` -> planned
   - `seeded_reproducible` -> operational
   - `best_effort` -> operational
   - `exploratory` -> registry_only
2. Keep the axis optional in recipe YAML.
   - If omitted, compiler defaults to `best_effort`.
3. Add compiler validation:
   - `seeded_reproducible` requires `leaf_config.random_seed`
   - `strict_reproducible` also requires `leaf_config.random_seed`
   - missing seed should fail compile explicitly
4. Preserve provenance:
   - `tree_context["reproducibility_mode"]`
   - `manifest["reproducibility_spec"] = {"reproducibility_mode": ..., "random_seed": ...}`
5. Do not change runtime behavior yet beyond carrying provenance.

Verification
- focused: `python3 -m pytest tests/test_registry_loader.py tests/test_compiler.py -q`
- broad: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py tests/test_start.py tests/test_registry_loader.py -q`
