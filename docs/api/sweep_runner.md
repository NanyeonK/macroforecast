# Sweep Runner API

Public API surface introduced in v0.3 (Phase 1).

## compile_sweep_plan

```python
from macrocast import compile_sweep_plan
```

```python
compile_sweep_plan(
    recipe_dict: dict,
    *,
    max_variants: int | None = 1000,
) -> SweepPlan
```

Expand a recipe dict's `sweep_axes` across layers into a Cartesian
product of single-path variant recipe dicts. Raises `SweepPlanError` on
invalid plans (duplicate axis names, empty sweep lists, oversized
Cartesian).

## execute_sweep

```python
from macrocast import execute_sweep
```

```python
execute_sweep(
    *,
    plan: SweepPlan,
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
    execution_route: str = "comparison_sweep",
    extra_provenance: dict | None = None,
) -> SweepResult
```

Executes every variant under `output_root`, sharing a FRED cache and
writing a Schema v1 `study_manifest.json` at the root.

Failure behavior is controlled by the parent recipe's Layer 0
`failure_policy`. Current public values are `fail_fast`, `continue_on_error`,
and `collect_errors`.

## Dataclasses

### SweepPlan

- `study_id: str` — stable hash of the plan
- `parent_recipe_id: str`
- `parent_recipe_dict: dict`
- `axes_swept: tuple[str, ...]` — layer-qualified names, e.g.
  `("4_forecasting_model.fit_model.params.family",)`
- `variants: tuple[SweepVariant, ...]`
- `size: int` — number of variants

### SweepVariant

- `variant_id: str` — `v-<8-hex>`
- `axis_values: dict[str, str]`
- `parent_recipe_id: str`
- `variant_recipe_dict: dict` — standalone single-path recipe

### SweepResult

- `study_id: str`
- `output_root: str`
- `manifest_path: str`
- `per_variant_results: tuple[VariantResult, ...]`
- `successful_count: int`
- `failed_count: int`
- `skipped_count: int`
- `size: int`

### VariantResult

- `variant_id: str`
- `axis_values: dict[str, str]`
- `status: str` — `success | failed | skipped`
- `artifact_dir: str | None`
- `runtime_seconds: float`
- `compiler_status: str | None`
- `compiler_warnings: tuple[str, ...]`
- `compiler_blocked_reasons: tuple[str, ...]`
- `layer3_capability_cell: dict[str, Any]`
- `error: str | None`
- `metrics_summary: dict`

## Study manifest helpers

```python
from macrocast import (
    STUDY_MANIFEST_SCHEMA_VERSION,
    VariantManifestEntry,
    build_study_manifest,
    validate_study_manifest,
)
```

- `STUDY_MANIFEST_SCHEMA_VERSION` — currently `"1.0"`
- `build_study_manifest(...)` — produce a Schema-v1 dict
- `validate_study_manifest(manifest)` — raises
  `StudyManifestSchemaError` on mismatch
- `VariantManifestEntry` — per-variant entry dataclass

## Errors

- `SweepPlanError` — plan compilation failures
- `StudyManifestSchemaError` — manifest validation failures
- `ExecutionError` — bubbled from per-variant `execute_recipe`

## Migration from v0.2

Recipes that previously specified `sweep_axes` were accepted by
`compile_recipe_dict` but ignored at execution time. In v0.3 the
presence of `sweep_axes` routes the recipe through `compile_sweep_plan`
+ `execute_sweep`. Single-path recipes (no `sweep_axes`) keep working
exactly as before.
