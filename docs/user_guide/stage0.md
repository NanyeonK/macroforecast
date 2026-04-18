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

## Value catalogs for Stage 0 axes

Stage 0 pins seven enum catalogs whose values recur in every recipe. The dataclasses on this page accept only the `operational` values from these catalogs — `registry_only` and `future` values pass registry validation but are not wired to runtime behaviour in v1.

### `study_mode`

The top-level research identity of the study. `study_mode` selects which execution route the compiler produces.

| Value | Status | When to use |
|---|---|---|
| `single_path_benchmark_study` | operational | Default. One recipe produces one evaluation path — the core macrocast horse-race against a baseline. Executes via `execute_recipe()`. |
| `controlled_variation_study` | operational | One or more axes are explicitly swept; the rest stays identical. Executes via `execute_sweep()`. |
| `orchestrated_bundle_study` | operational (wrapper-route) | Multiple recipes bundled by an external orchestrator. Compiler emits a `wrapper_handoff` payload instead of a directly executable plan. |
| `replication_override_study` | operational (wrapper-route) | Replication of a prior recipe with locked overrides. Same wrapper-handoff semantics; runs via `execute_replication()`. |

The two wrapper-route modes compile to `representable_but_not_executable` for a direct `execute_recipe()` call. They are consumed through the sweep runner, `execute_replication()`, or Phase 8's `PaperReadyBundle`.

### `experiment_unit`

The per-recipe execution shape. Complements `study_mode` by stating how many targets and models the recipe is about.

| Value | Status | When to use |
|---|---|---|
| `single_target_single_model` | operational | Default. One target, one model — the smallest executable unit. |
| `single_target_model_grid` | operational | One target, multiple candidate models compared via the model-family axis. |
| `single_target_full_sweep` | operational (wrapper-route) | One target, full Cartesian sweep across user-chosen axes. Wrapper-managed. |
| `multi_target_shared_design` | operational (wrapper-route) | Multiple targets share the identical design; wrapper fan-out executes them. |
| `replication_recipe` | operational | Replication unit paired with `replication_override_study`. |
| `benchmark_suite` | operational (wrapper-route) | Collection of benchmarks evaluated against the user's proposed method. Wrapper-managed. |
| `ablation_study` | operational | Ablation unit paired with `controlled_variation_study`; runs via `execute_ablation()`. |
| `multi_target_separate_runs` | registry_only (v1.1) | Multi-target as independent runs; wrapper implementation pending. |
| `multi_output_joint_model` | registry_only (v1.1) | Joint multi-output model; requires joint predictor adapters. |
| `hierarchical_forecasting_run` | future (v2) | Hierarchical forecasting reconciliation. |
| `panel_forecasting_run` | future (v2) | Panel-data forecasting. |
| `state_space_run` | future (v2) | Single-run state-space forecasting. |

### `failure_policy`

Sweep-cell failure semantics.

| Value | Status | Behaviour |
|---|---|---|
| `fail_fast` | operational | Abort entire sweep on first failed cell. |
| `hard_error` | operational | Equivalent strict fail-fast with explicit `HardError`. |
| `skip_failed_cell` | operational | Skip the failed cell, continue remaining cells, emit warning log. |
| `skip_failed_model` | operational | Skip the failed model inside a model-family variant. |
| `save_partial_results` | operational | Persist partial state of the failed cell before skipping. |
| `retry_then_skip` | registry_only (v1.1) | Not wired in v1 runtime. |
| `fallback_to_default_hp` | registry_only (v1.1) | HP fallback not wired. |
| `warn_only` | registry_only (v1.1) | Warn-only path not wired. |

### `reproducibility_mode`

Deterministic replay contract. See `docs/dev/reproducibility_policy.md` and `macrocast.execution.seed_policy`.

| Value | Status | Contract |
|---|---|---|
| `strict_reproducible` | operational | Byte-identical reruns required; pins seeds, cache keys, library versions. |
| `seeded_reproducible` | operational | Default. Seeds fixed; small numerical drift across library versions accepted. |
| `best_effort` | operational | No seed pinning; suited to ad-hoc exploration. |
| `exploratory` | registry_only (v1.1) | Exploratory path without reproducibility guarantee; not wired. |

### `compute_mode`

Parallelism unit.

| Value | Status | Semantics |
|---|---|---|
| `serial` | operational | Single-threaded. Default. |
| `parallel_by_model` | operational | Parallel across model-family variants. |
| `parallel_by_horizon` | operational | Parallel across forecast horizons. |
| `parallel_by_oos_date` | registry_only (v1.1) | Not wired. |
| `parallel_by_trial` | registry_only (v1.1) | Not wired; awaits integration with `execution_backend.joblib`. |
| `distributed_cluster` | registry_only (v1.1) | Distributed execution not wired. |

### `axis_type`

How an axis participates in the study expansion.

| Value | Status | Role |
|---|---|---|
| `fixed` | operational | Held constant across the study. |
| `sweep` | operational | Expanded into multiple variants. |
| `nested_sweep` | operational | Participates in nested-sweep plans. |
| `conditional` | operational | Activated conditionally on other axes. |
| `derived` | operational | Computed from other recipe state. |

### `registry_type`

Catalog kind for a given axis.

| Value | Status | Kind |
|---|---|---|
| `enum_registry` | operational | Finite enumerated catalog — most axes. |
| `numeric_registry` | operational | Numeric range/grid. |
| `callable_registry` | operational | Callable-signature validated catalog. |
| `custom_plugin` | operational | Plugin-backed catalog. |
| `user_defined_yaml` | registry_only (v1.1) | User-supplied YAML schema adapter; not yet wired. |

Most users set only `study_mode`, `experiment_unit`, and optionally `compute_mode` / `failure_policy` / `reproducibility_mode` in their recipe. `axis_type` and `registry_type` are consumed by the registry infrastructure and rarely set by users directly.

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
