# Macrocast Stage 0 — V1 Python API Design

Status: reboot-stage implementation design
Date: 2026-04-14
Purpose: translate the Stage 0 grammar contract into a pythonic v1 code surface

## 0. Why this document exists

The full Stage 0 grammar contract is the right architectural language, but it is too broad to expose directly as a public runtime API.

If every Stage 0 concept becomes its own large public object, the package will become unpythonic:
- too many top-level structures
- too much duplicated meaning
- too much ceremony for ordinary v1 usage

So v1 should implement a smaller, pythonic Stage 0 API that preserves the grammar contract without forcing every conceptual distinction into a separate public surface.

## 1. Design rule

In code, Stage 0 should look like:
- a small number of dataclasses
- a few normalize / validate / derive helpers
- one main constructor
- one or two routing/completeness helpers

The public API should be smaller than the conceptual contract.

## 2. What should be public in v1

Public Stage 0 surface should be limited to:
- `Stage0Frame`
- `FixedDesign`
- `ComparisonContract`
- `build_stage0_frame()`
- `resolve_route_owner()`
- `check_stage0_completeness()`

Optional but still acceptable in public surface:
- `VaryingDesign`
- `ExecutionPosture`
- `StudyMode`

Everything else should remain internal helper logic.

## 3. What should stay internal in v1

These should not become large public runtime systems in v1:
- a rich standalone `registry_scope_contract` runtime object
- a rich standalone `compatibility_mirrors` runtime object
- a registry-first Stage 0 builder
- a giant all-fields YAML-first Stage 0 editing surface

These ideas can remain in the architecture documents and be implemented later only if runtime pressure requires them.

## 4. Recommended file layout

```text
macrocast/
  stage0/
    __init__.py
    types.py
    normalize.py
    validate.py
    derive.py
    build.py
    serialize.py
```

## 5. Recommended dataclasses

### 5.1 `FixedDesign`

Purpose:
- hold the fairness-defining common environment of the study

```python
@dataclass(frozen=True)
class FixedDesign:
    dataset_adapter: str
    information_set: str
    sample_split: str
    benchmark: str
    evaluation_protocol: str
    forecast_task: str
```

Why this shape is pythonic:
- flat
- explicit
- fairness-relevant only
- no giant nested schema for v1

### 5.2 `VaryingDesign`

Purpose:
- hold explicitly allowed study variation

```python
@dataclass(frozen=True)
class VaryingDesign:
    model_families: tuple[str, ...] = ()
    feature_recipes: tuple[str, ...] = ()
    preprocess_variants: tuple[str, ...] = ()
    tuning_variants: tuple[str, ...] = ()
    horizons: tuple[str, ...] = ()
```

Why this shape is pythonic:
- empty means no variation beyond baseline single-path comparison
- bounded known fields
- no free-form arbitrary tree in v1

### 5.3 `ComparisonContract`

Purpose:
- make fairness conditions explicit and easy to inspect

```python
@dataclass(frozen=True)
class ComparisonContract:
    information_set_policy: str
    sample_split_policy: str
    benchmark_policy: str
    evaluation_policy: str
```

### 5.4 `ReplicationInput`

Purpose:
- keep replication explicit without making it the default front door

```python
@dataclass(frozen=True)
class ReplicationInput:
    source_type: str
    source_id: str
    locked_constraints: tuple[str, ...] = ()
    override_reason: str | None = None
```

Note:
- this can be public or semi-internal
- keep it small if public

### 5.5 `Stage0Frame`

Purpose:
- canonical Stage 0 output used by later layers

```python
@dataclass(frozen=True)
class Stage0Frame:
    study_mode: str
    fixed_design: FixedDesign
    comparison_contract: ComparisonContract
    varying_design: VaryingDesign
    execution_posture: str
    design_shape: str
    replication_input: ReplicationInput | None = None
    experiment_unit: str | None = None
```

Important note:
- `experiment_unit` here is a compatibility mirror only if needed
- it should never be the primary truth in v1

## 6. Recommended public functions

### 6.1 `build_stage0_frame()`

This is the main public constructor.

```python
def build_stage0_frame(
    *,
    study_mode: str,
    fixed_design: FixedDesign | dict,
    comparison_contract: ComparisonContract | dict,
    varying_design: VaryingDesign | dict | None = None,
    replication_input: ReplicationInput | dict | None = None,
) -> Stage0Frame:
    ...
```

Responsibilities:
- normalize incoming values
- validate required structure
- derive design shape
- derive execution posture
- derive optional compatibility mirrors
- return one canonical immutable frame

This should be the main entry point.

### 6.2 `resolve_route_owner()`

```python
def resolve_route_owner(stage0: Stage0Frame) -> str:
    ...
```

Returns:
- `single_run`
- `wrapper`
- `replication`

Purpose:
- make downstream routing extremely simple

### 6.3 `check_stage0_completeness()`

```python
def check_stage0_completeness(stage0: Stage0Frame) -> None:
    ...
```

Purpose:
- fail closed before execution if fairness or structure is incomplete

This function should raise explicit Stage 0 errors rather than return booleans.

### 6.4 `stage0_summary()`

```python
def stage0_summary(stage0: Stage0Frame) -> str:
    ...
```

Purpose:
- human-readable summary for logs, manifests, CLI, and docs examples

### 6.5 `stage0_to_dict()` / `stage0_from_dict()`

```python
def stage0_to_dict(stage0: Stage0Frame) -> dict:
    ...


def stage0_from_dict(payload: dict) -> Stage0Frame:
    ...
```

Purpose:
- recipe serialization and config I/O

## 7. Internal helper functions

These should support the public API but not dominate it.

### normalize helpers

- `normalize_study_mode(value)`
- `normalize_fixed_design(value)`
- `normalize_varying_design(value)`
- `normalize_comparison_contract(value)`
- `normalize_replication_input(value)`

### validate helpers

- `validate_fixed_design(fixed_design)`
- `validate_varying_design(varying_design)`
- `validate_comparison_contract(contract)`
- `validate_replication_input(replication_input)`
- `validate_stage0_frame(stage0)`

### derive helpers

- `derive_design_shape(study_mode, fixed_design, varying_design)`
- `derive_execution_posture(study_mode, design_shape, replication_input)`
- `derive_experiment_unit(stage0)`

These are implementation helpers, not the user-facing conceptual contract.

## 8. Why this is more pythonic than the raw conceptual contract

### Reason 1. Fewer public objects
The architecture note can talk about many conceptual categories.
The runtime API should expose only what users and downstream code actually need.

### Reason 2. One main constructor
Users should not manually assemble seven independent structural objects unless necessary.
A single `build_stage0_frame()` is cleaner.

### Reason 3. Flat dataclasses for v1
Deep nested schema trees are hard to read and easy to misuse.
Flat dataclasses are easier to inspect, test, and document.

### Reason 4. Internal helpers hide ceremony
Normalization, validation, and derivation still happen, but users do not need to manage each step manually.

## 9. Suggested v1 semantics

### `study_mode`
Use string literals or enum-like strings.

v1 values:
- `single_path_benchmark_study`
- `controlled_variation_study`
- `orchestrated_bundle_study`
- `replication_override_study`

### `design_shape`
Derived, not user-authored in normal use.

v1 values:
- `one_fixed_env_one_tool_surface`
- `one_fixed_env_multi_tool_surface`
- `one_fixed_env_controlled_axis_variation`
- `wrapper_managed_multi_run_bundle`

### `execution_posture`
Derived, not manually filled in ordinary use.

v1 values:
- `single_run_recipe`
- `single_run_with_internal_sweep`
- `wrapper_bundle_plan`
- `replication_locked_plan`

## 10. Example usage

### Example A. Ordinary single-path benchmark study

```python
stage0 = build_stage0_frame(
    study_mode="single_path_benchmark_study",
    fixed_design={
        "dataset_adapter": "fred_md",
        "information_set": "revised_monthly",
        "sample_split": "expanding_window_oos",
        "benchmark": "ar_bic",
        "evaluation_protocol": "point_forecast_core",
        "forecast_task": "single_target_point_forecast",
    },
    comparison_contract={
        "information_set_policy": "identical",
        "sample_split_policy": "identical",
        "benchmark_policy": "identical",
        "evaluation_policy": "identical",
    },
    varying_design={
        "model_families": ("ar", "ridge", "lasso", "rf"),
        "horizons": ("h1", "h3", "h6", "h12"),
    },
)
```

### Example B. Replication override

```python
stage0 = build_stage0_frame(
    study_mode="replication_override_study",
    fixed_design={
        "dataset_adapter": "fred_md",
        "information_set": "revised_monthly",
        "sample_split": "paper_locked_split",
        "benchmark": "paper_locked_benchmark",
        "evaluation_protocol": "paper_locked_eval",
        "forecast_task": "single_target_point_forecast",
    },
    comparison_contract={
        "information_set_policy": "paper_locked",
        "sample_split_policy": "paper_locked",
        "benchmark_policy": "paper_locked",
        "evaluation_policy": "paper_locked",
    },
    replication_input={
        "source_type": "paper_recipe",
        "source_id": "clss2021",
        "locked_constraints": ("split", "benchmark", "target_def"),
    },
)
```

## 11. Errors to define

Stage 0 code should use explicit package errors.

Suggested errors:
- `Stage0Error`
- `Stage0NormalizationError`
- `Stage0ValidationError`
- `Stage0CompletenessError`
- `Stage0RoutingError`

Do not use generic `ValueError` everywhere.

## 12. What to implement first

Implementation order for Stage 0 code:

1. `types.py`
- dataclasses and literal aliases

2. `normalize.py`
- normalize helper functions

3. `validate.py`
- validation helpers

4. `derive.py`
- design-shape and execution-posture derivation

5. `build.py`
- `build_stage0_frame()`
- `resolve_route_owner()`
- `check_stage0_completeness()`
- `stage0_summary()`

6. `serialize.py`
- dict conversion helpers

## 13. What not to implement in v1 Stage 0

Do not build yet:
- interactive Stage 0 wizard
- giant generic YAML grammar editor
- rich registry-scope runtime object system
- complicated mirror-management layer
- overgeneralized plugin architecture for Stage 0

## 14. Bottom line

The conceptual Stage 0 contract should remain broad.
The runtime Stage 0 API should remain small.

For v1, the right pythonic shape is:
- a few small dataclasses
- a main `build_stage0_frame()` constructor
- explicit validate / derive helpers
- a tiny public surface

That gives macrocast a strong execution language without making Stage 0 itself feel like a bureaucracy framework.