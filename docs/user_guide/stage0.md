# Stage 0

## Purpose

Stage 0 fixes the execution language of a macrocast study before later registries or recipe content are expanded.

In practical terms, Stage 0 answers these questions first:
- what kind of study is this?
- what is fixed for fairness?
- what is intentionally allowed to vary?
- what comparison conditions must remain identical?
- what execution posture should downstream code build?

Stage 0 does not decide registry inventories such as which model ids or dataset ids exist. It fixes the grammar those later content layers must obey.

## Why Stage 0 exists

macrocast is designed to compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

That comparison becomes unreliable if package structure allows:
- hidden route changes
- accidental drift in fairness conditions
- registry content that changes execution semantics
- model comparisons that no longer share the same study spine

Stage 0 prevents that by making study structure explicit before forecasting execution begins.

## Public code surface

The v1 Stage 0 code surface is intentionally small:

```python
from macrocast.stage0 import (
    FixedDesign,
    VaryingDesign,
    ComparisonContract,
    ReplicationInput,
    Stage0Frame,
    build_stage0_frame,
    resolve_route_owner,
    check_stage0_completeness,
    stage0_summary,
    stage0_to_dict,
    stage0_from_dict,
)
```

The package does not expose a giant Stage 0 configuration bureaucracy. Instead, it exposes a small number of dataclasses and one canonical builder.

## Main dataclasses

### `FixedDesign`

`FixedDesign` holds the fairness-defining common environment of the study.

```python
FixedDesign(
    dataset_adapter="fred_md",
    information_set="revised_monthly",
    sample_split="expanding_window_oos",
    benchmark="ar_bic",
    evaluation_protocol="point_forecast_core",
    forecast_task="single_target_point_forecast",
)
```

Use `FixedDesign` for choices that must remain the same across compared tools.

### `VaryingDesign`

`VaryingDesign` holds explicitly allowed study variation.

```python
VaryingDesign(
    model_families=("ar", "ridge", "lasso", "rf"),
    horizons=("h1", "h3", "h6", "h12"),
)
```

Use `VaryingDesign` for dimensions that are intentionally varied inside the study rather than drifting accidentally.

### `ComparisonContract`

`ComparisonContract` makes fairness conditions explicit.

```python
ComparisonContract(
    information_set_policy="identical",
    sample_split_policy="identical",
    benchmark_policy="identical",
    evaluation_policy="identical",
)
```

This object states the alignment conditions required for a valid comparison.

### `ReplicationInput`

`ReplicationInput` is optional and keeps replication explicit rather than making it the default front door.

```python
ReplicationInput(
    source_type="paper_recipe",
    source_id="clss2021",
    locked_constraints=("split", "benchmark"),
)
```

### `Stage0Frame`

`Stage0Frame` is the canonical Stage 0 output consumed by later layers.

It stores:
- `study_mode`
- `fixed_design`
- `comparison_contract`
- `varying_design`
- `execution_posture`
- `design_shape`
- optional `replication_input`
- optional compatibility mirror `experiment_unit`

In ordinary use, users should not assemble `Stage0Frame` manually. They should call `build_stage0_frame()`.

## Main functions

### `build_stage0_frame()`

This is the primary constructor.

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
        "model_families": ("ar", "ridge"),
        "horizons": ("h1", "h3"),
    },
)
```

`build_stage0_frame()` performs three jobs:
- normalize incoming values
- validate the study structure
- derive the downstream design shape and execution posture

### `resolve_route_owner()`

`resolve_route_owner(stage0)` tells downstream code who owns execution routing.

Possible return values in v1:
- `single_run`
- `wrapper`
- `replication`

Use this when later code must decide whether the study should remain a single-run object or move into wrapper-managed execution.
For wrapper-owned studies in the current package, compiler now emits a minimal `wrapper_handoff` payload instead of pretending the study is directly executable.

### `check_stage0_completeness()`

`check_stage0_completeness(stage0)` fails closed when Stage 0 is not ready for execution.

In the current v1 skeleton, it rejects single-run execution if no model family has been declared in `varying_design`.

### `stage0_summary()`

`stage0_summary(stage0)` returns a human-readable one-line summary suitable for logs, manifests, or CLI previews.

### `stage0_to_dict()` / `stage0_from_dict()`

These helpers support recipe serialization and config I/O.

## Derived semantics

Two important fields are derived rather than hand-authored in ordinary use.

### `design_shape`

`design_shape` is inferred from `study_mode` and `varying_design`.

Example values:
- `one_fixed_env_one_tool_surface`
- `one_fixed_env_multi_tool_surface`
- `one_fixed_env_controlled_axis_variation`
- `wrapper_managed_multi_run_bundle`

### `execution_posture`

`execution_posture` is inferred from `study_mode`, `design_shape`, and optional replication input.

Example values:
- `single_run_recipe`
- `single_run_with_internal_sweep`
- `wrapper_bundle_plan`
- `replication_locked_plan`

These derived values keep route semantics out of arbitrary registry payloads.

## Completeness rule

A Stage 0 frame is only usable if it answers all of the following:
- what kind of study is this?
- what is fixed?
- what is allowed to vary?
- what fairness conditions define the comparison?
- what execution posture should downstream code build?

If those questions are unanswered, later layers should not proceed.

## Relationship to later layers

Stage 0 sits above later content layers.

It feeds:
- raw/data layer
- design layer
- execution layer
- evaluation/test layer
- output/provenance layer

Later registries may supply admissible content, but they should not redefine the Stage 0 language.

## v1 implementation philosophy

The Stage 0 runtime API is deliberately smaller than the full architectural doctrine.

That is intentional.

The architecture documents describe the full language of the package. The code surface stays compact so that ordinary package use remains readable and pythonic.


## V1 completion status

The current Stage 0 layer should now be treated as complete for the v1 package foundation.

What that means:
- canonical Stage 0 dataclasses exist
- a main builder exists
- route ownership resolution exists
- completeness checks exist
- dict serialization round-trip exists
- replication override path exists
- explicit Stage 0 error classes exist

This does not mean Stage 0 will never grow again.
It means the current package can safely treat Stage 0 as a stable foundational layer while later package surfaces are built above it.
