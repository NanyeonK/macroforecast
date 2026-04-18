# Design (Stage 0)

> **Naming:** The code module was `macrocast.stage0` before 2026-04-18; it is now `macrocast.design`. "Stage 0" remains the architectural term for this pre-execution grammar layer; `macrocast.design` is the code surface. The registry layer at ``macrocast.registry.stage0`` kept its path to distinguish framework (design) from registry (Layer 0 meta).

!!! note "Design framework vs. Layer 0 meta axes registry"
    - **Design framework** — pre-execution grammar dataclasses. Lives at ``macrocast.design``. Small set of pre-execution dataclasses (`FixedDesign`, `VaryingDesign`, `ComparisonContract`, `DesignFrame`) plus one builder (`build_design_frame`).
    - **Layer 0 meta axes registry** — 7 enum catalogs consumed by the framework. Lives at ``macrocast.registry.stage0``. This page walks through all 7 in order (§0.1 through §0.7).

## Purpose

The design frame fixes the execution language of a macrocast study before later registries or recipe content are expanded. It answers these questions first:

- what kind of study is this? (`study_mode`)
- what recipe shape is this? (`experiment_unit`)
- how does each axis participate? (`axis_type`)
- what happens on failure? (`failure_policy`)
- how reproducible must it be? (`reproducibility_mode`)
- how is parallelism applied? (`compute_mode`)
- what kind of catalog is each axis? (`registry_type`)

Design does not decide registry inventories such as which model ids or dataset ids exist. It fixes the grammar those later content layers must obey.

## Why design exists

macrocast compares forecasting tools under identical information set, sample split, benchmark, and evaluation protocol. Comparison becomes unreliable if the package allows hidden route changes, accidental drift in fairness conditions, or model comparisons that no longer share the same study spine. The design frame prevents that by making study structure explicit before forecasting execution begins.

---

## 0.1 `axis_type`

**Declares how an axis participates in a recipe.** Every axis carries a `default_policy`; the recipe overrides by placing the axis in one of five sections.

### Value catalog

| Value | Status | Recipe section | Role |
|---|---|---|---|
| `fixed` | operational | `fixed_axes` | Held constant across the study. One value per axis. |
| `sweep` | operational | `sweep_axes` | Cartesian-expanded across the listed values. |
| `conditional` | operational | `conditional_axes` | Value depends on another axis's choice; activated only when the trigger axis takes a matching value. |
| `nested_sweep` | operational | `nested_sweep_axes` | Hierarchical sweep: each parent value carries its own (possibly different) child axis and child values. |
| `derived` | operational | `derived_axes` | Computed by a registered derivation rule at compile time (e.g. `experiment_unit_default`). |

### Functions & features

- `AxisSelectionMode` Literal in `macrocast.registry.types` — the 5 modes are the exhaustive type space for axis selection.
- `AxisSelection` dataclass — each selected axis carries `selection_mode` set to one of the 5 above.
- `compile_sweep_plan()` in `macrocast.compiler.sweep_plan` — expands `sweep_axes` + `nested_sweep_axes` into the Cartesian variant grid.
- `DERIVATION_RULES` dict in `macrocast.compiler.build` — registered rules for `derived_axes`. Currently: `experiment_unit_default`.
- `_resolve_derived_axes()` in `macrocast.compiler.build` — compile-time resolver that adds AxisSelections with `selection_mode="derived"` to the selection tuple.

### Recipe usage

```yaml
path:
  3_training:
    fixed_axes:
      framework: rolling
    sweep_axes:
      model_family: [ridge, lasso]
    nested_sweep_axes:
      model_family:
        ridge: {hp_space_style: [paper_fixed_hp, grid_linear]}
        lasso: {hp_space_style: [paper_fixed_hp, grid_linear, grid_log]}
  0_meta:
    derived_axes:
      experiment_unit: experiment_unit_default
```

An axis can appear in at most one section; putting the same axis in multiple sections raises `CompileValidationError`. The YAML above expands to `sweep(2) × nested(5) = 10` variants, with `experiment_unit` derived per-variant.

---

## 0.2 `compute_mode`

**Declares the parallelism unit** for sweep / multi-target / multi-horizon execution. Default is `serial`. All parallel modes use `concurrent.futures.ThreadPoolExecutor` with `max_workers` capped at 4; speedup on CPU-heavy work depends on whether the underlying numpy/pandas/model code releases the GIL.

### Value catalog

| Value | Status | Trigger condition | Effect |
|---|---|---|---|
| `serial` | operational | — | Single-threaded execution throughout. Default. |
| `parallel_by_model` | operational | sweep plan contains a `model_family` sweep axis AND `len(plan.variants) > 1` | Variant-level threading in `execute_sweep`: different model_family variants run concurrently. Silent no-op (serial fallback) if the trigger condition fails. |
| `parallel_by_horizon` | operational | `len(recipe.horizons) > 1` | Horizon-level threading inside `execute_recipe`: each horizon computed in its own worker. Silent no-op for single-horizon recipes. |
| `parallel_by_target` | operational | `len(recipe.targets) > 1` | Target-level threading inside `execute_recipe`: each target's slice computed concurrently. Silent no-op for single-target recipes. |
| `parallel_by_oos_date` | registry_only (v1.1) | — | Compiler rejects as "representable but not executable". Needs phase-10 executor to chunk by OOS date. |
| `parallel_by_trial` | registry_only (v1.1) | — | Compiler rejects. Awaits tuning backend integration (`execution_backend.joblib`). |
| `distributed_cluster` | registry_only (v2) | — | Compiler rejects. Needs a distributed runtime (phase-11). |

The three operational parallel modes are mutually independent — they operate at different pipeline layers (sweep, horizon, target) — but a single study picks one. Stacked parallelism across layers is not supported in v1.

### Functions & features

- Compiler spec: `compute_mode_spec` dict embedded in `CompiledRecipeSpec` (`macrocast/compiler/build.py`).
- Compiler guard: `compute_mode` restricted to the 4 operational values at compile time; registry_only and future values raise "representable but not executable".
- Sweep runner (`macrocast.execution.sweep_runner`):
  - `_extract_parent_compute_mode(plan)` reads the compute_mode from the parent recipe's `0_meta` block; defaults to `serial`.
  - `execute_sweep()` dispatches variants via `ThreadPoolExecutor` when `compute_mode == "parallel_by_model"` and a `model_family` sweep axis is present.
- Execution build (`macrocast.execution.build`):
  - Horizon loop wraps its row-builder in `ThreadPoolExecutor` when `compute_mode == "parallel_by_horizon"` and `len(horizons) > 1`.
  - Target loop wraps its per-target job in `ThreadPoolExecutor` when `compute_mode == "parallel_by_target"` and `len(targets) > 1`.

### Recipe usage

```yaml
# Variant-level parallelism — run a model horse-race with different families in parallel.
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_model
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, random_forest]
```

```yaml
# Target-level parallelism — multi-target recipe with concurrent per-target execution.
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_target
  1_data_task:
    fixed_axes:
      task: multi_target_point_forecast
    leaf_config:
      targets: [INDPRO, RPI, CPIAUCSL]
```

```yaml
# Horizon-level parallelism — single recipe computing multiple horizons concurrently.
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_horizon
  1_data_task:
    leaf_config:
      horizons: [1, 3, 6, 12]
```

---

## 0.3 `experiment_unit`

**Declares the per-recipe execution shape** — how many targets and models the recipe produces. Complements `study_mode`. This is the richest Layer 0 axis, with a dedicated entry dataclass and helper functions.

### Value catalog

| Value | Status | When to use |
|---|---|---|
| `single_target_single_model` | operational | Default. One target, one model — the smallest executable unit. |
| `single_target_model_grid` | operational | One target, multiple candidate models compared via the model-family axis. |
| `single_target_full_sweep` | operational (wrapper-route) | One target, full Cartesian sweep. Wrapper-managed. |
| `multi_target_shared_design` | operational (wrapper-route) | Multiple targets share identical design; wrapper fan-out. |
| `replication_recipe` | operational | Replication unit paired with `replication_override_study`. |
| `benchmark_suite` | operational (wrapper-route) | Collection of benchmarks against the user's proposed method. |
| `ablation_study` | operational | Ablation unit; runs via `execute_ablation()`. |
| `multi_target_separate_runs` | registry_only (v1.1) | Multi-target as independent runs. |
| `multi_output_joint_model` | registry_only (v1.1) | Joint multi-output model; needs joint predictor adapters. |
| `hierarchical_forecasting_run` | future (v2) | Hierarchical reconciliation. |
| `panel_forecasting_run` | future (v2) | Panel-data forecasting. |
| `state_space_run` | future (v2) | Single-run state-space forecasting. |

### Functions & features

- `ExperimentUnitEntry` dataclass in `macrocast.registry.stage0.experiment_unit` — extends `EnumRegistryEntry` with `route_owner`, `requires_multi_target`, `requires_wrapper`, optional `runner` callable path.
- `get_experiment_unit_entry(id)` — lookup a single unit entry by id.
- `experiment_unit_options_for_wizard(study_mode, task)` — filter allowed unit options for a given study_mode + task pair. Useful for CLI wizards / UIs.
- `derive_experiment_unit_default(study_mode, task, model_axis_mode, feature_axis_mode, wrapper_family)` — infers the default unit from recipe shape. Also exposed as the `experiment_unit_default` derivation rule in `DERIVATION_RULES` (see §0.1).
- Compiler integration: `compile_recipe_dict()` auto-derives `experiment_unit` when it is not explicitly declared; conflict-checks explicit declarations against the derived default.

### Recipe usage

Explicit declaration:

```yaml
path:
  0_meta:
    fixed_axes:
      experiment_unit: ablation_study
```

Implicit (recommended for most recipes) — the compiler derives the right unit from the rest of the recipe. Equivalent to declaring:

```yaml
path:
  0_meta:
    derived_axes:
      experiment_unit: experiment_unit_default
```

---

## 0.4 `failure_policy`

**Declares sweep-cell failure semantics.** Default is `fail_fast`.

### Value catalog

| Value | Status | Behaviour |
|---|---|---|
| `fail_fast` | operational | Abort entire sweep on first failed cell. |
| `hard_error` | operational | Strict fail-fast with explicit `HardError`. |
| `skip_failed_cell` | operational | Skip the failed cell, continue remaining cells, emit warning log. |
| `skip_failed_model` | operational | Skip the failed model inside a model-family variant. |
| `save_partial_results` | operational | Persist partial state of the failed cell before skipping. |
| `retry_then_skip` | registry_only (v1.1) | Not wired. |
| `fallback_to_default_hp` | registry_only (v1.1) | HP fallback not wired. |
| `warn_only` | registry_only (v1.1) | Warn-only path not wired. |

### Functions & features

- Compiler spec: `failure_policy_spec` on the compiled payload (`compiler/build.py`).
- Runtime dispatch: `macrocast.execution.build` branches on the policy string — 4 of 5 operational values have explicit runtime paths; `hard_error` is equivalent to `fail_fast` with a different exception.
- Sweep runner integration: `execute_sweep()` honors `skip_failed_cell` / `skip_failed_model` / `save_partial_results` per-variant.

### Recipe usage

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: skip_failed_cell
```

---

## 0.5 `registry_type`

**Declares the catalog kind** of each axis in the registry. Set by the axis's own registry module, not by users.

### Value catalog

| Value | Status | Kind |
|---|---|---|
| `enum_registry` | operational | Finite enumerated catalog — most axes. |
| `numeric_registry` | operational | Numeric range/grid. |
| `callable_registry` | operational | Callable-signature validated catalog. |
| `custom_plugin` | operational | Plugin-backed catalog. |
| `user_defined_yaml` | registry_only (v1.1) | User-supplied YAML schema adapter; not yet wired. |

### Functions & features

- `AxisDefinition.registry_type` field in `macrocast.registry.base`. Each axis module sets this explicitly (usually to `enum_registry`).
- Informs the validation path in registry loaders (e.g. callable catalogs get signature checks; numeric catalogs get range checks).
- Users rarely consume this directly; it is infrastructure metadata.

### Recipe usage

Not set in a recipe. It is part of axis registration source, not recipe authoring.

---

## 0.6 `reproducibility_mode`

**Declares the deterministic replay contract.** Default is `seeded_reproducible`.

### Value catalog

| Value | Status | Contract |
|---|---|---|
| `strict_reproducible` | operational | Byte-identical reruns required; hash-derived per-variant seeds. |
| `seeded_reproducible` | operational | Default. Fixed `base_seed`; small numerical drift across library versions accepted. |
| `best_effort` | operational | No seed pinning; identical behavior to `seeded_reproducible` but marked non-strict for CI reporting. |
| `exploratory` | registry_only (v1.1) | Exploratory path; not wired. |

### Functions & features

- `macrocast.execution.seed_policy` — implements the 3 operational modes. The Literal `Mode = Literal["strict_reproducible", "seeded_reproducible", "best_effort"]` pins the set.
- Compiler spec: `reproducibility_spec` on the compiled payload.
- Runtime dispatch: every randomised operation pulls its RNG via `derive_seed(base_seed, scope, mode)` so the mode deterministically picks the seed policy.

### Recipe usage

```yaml
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: strict_reproducible
```

See also: `docs/dev/reproducibility_policy.md`.

---

## 0.7 `study_mode`

**Top-level research identity** of the study. Selects which execution route the compiler produces.

### Value catalog

| Value | Status | Route |
|---|---|---|
| `single_path_benchmark_study` | operational | Default. One recipe → one evaluation path. Runs via `execute_recipe()`. |
| `controlled_variation_study` | operational | One or more axes swept; the rest stays identical. Runs via `execute_sweep()`. |
| `orchestrated_bundle_study` | operational (wrapper-route) | Multiple recipes bundled by an external orchestrator. Compiler emits a `wrapper_handoff` payload. |
| `replication_override_study` | operational (wrapper-route) | Replication with locked overrides. Runs via `execute_replication()`. |

The two wrapper-route modes compile to `representable_but_not_executable` for a direct `execute_recipe()` call. They are consumed through the sweep runner, `execute_replication()`, or Phase 8's `PaperReadyBundle`.

### Functions & features

- `DEFAULT_STUDY_MODE` constant in `macrocast.execution.sweep_runner` (`"controlled_variation_study"`).
- `execute_sweep(recipe, study_mode=..., ...)` — sweep runner accepts an override; default is `controlled_variation_study`.
- Compiler routing: `compile_recipe_dict()` uses `study_mode` to pick executable vs wrapper-route, set execution_status, and bind the right runner.
- Study manifest: the chosen mode is recorded on `study_manifest.json` (schema v1).

### Recipe usage

```yaml
path:
  0_meta:
    fixed_axes:
      study_mode: controlled_variation_study
```

---

## Framework surface

The 7 axes above are consumed by a compact framework that the user typically touches only via `build_design_frame()`:

```python
from macrocast.design import (
    FixedDesign,
    VaryingDesign,
    ComparisonContract,
    ReplicationInput,
    DesignFrame,
    build_design_frame,
    resolve_route_owner,
    check_design_completeness,
    design_summary,
    design_to_dict,
    design_from_dict,
)
```

### Main dataclasses

- **`FixedDesign`** — fairness-defining common environment: `dataset_adapter`, `information_set`, `sample_split`, `benchmark`, `evaluation_protocol`, `forecast_task`.
- **`VaryingDesign`** — explicitly allowed variation: `model_families`, `horizons`, etc.
- **`ComparisonContract`** — fairness conditions: four `*_policy` flags (information_set / sample_split / benchmark / evaluation).
- **`ReplicationInput`** — optional; locks replication constraints (`source_type`, `source_id`, `locked_constraints`).
- **`DesignFrame`** — canonical output: `study_mode`, `fixed_design`, `comparison_contract`, `varying_design`, `execution_posture`, `design_shape`, optional `replication_input`, optional compat mirror `experiment_unit`.

### Main functions

- `build_design_frame(...)` — normalize, validate, derive `design_shape` and `execution_posture`.
- `resolve_route_owner(design)` — returns `single_run` / `wrapper` / `replication`.
- `check_design_completeness(design)` — fails closed if the frame cannot execute.
- `design_summary(design)` — one-line human summary for logs/manifests.
- `design_to_dict` / `design_from_dict` — round-trip serialization for config I/O.

### DesignFrame derived fields

Two fields are derived rather than hand-authored (distinct from `axis_type.derived` in §0.1 — these live on `DesignFrame`):

- **`design_shape`** — inferred from `study_mode` + `varying_design`. Examples: `one_fixed_env_one_tool_surface`, `one_fixed_env_controlled_axis_variation`, `wrapper_managed_multi_run_bundle`.
- **`execution_posture`** — inferred from `study_mode` + `design_shape` + optional replication input. Examples: `single_run_recipe`, `single_run_with_internal_sweep`, `wrapper_bundle_plan`, `replication_locked_plan`.

These keep route semantics out of arbitrary registry payloads.

---

## Completeness rule

A design frame is only usable if it answers all of:

- what kind of study is this?
- what is fixed?
- what is allowed to vary?
- what fairness conditions define the comparison?
- what execution posture should downstream code build?

`check_design_completeness()` fails closed when any answer is missing.

## Relationship to later layers

The design frame sits above later content layers. It feeds:

- [raw/data layer](raw.md)
- [data/task axes](data_task_axes.md)
- [recipes](recipes.md)
- [preprocessing](preprocessing.md)
- [compiler](compiler.md)
- [execution](execution.md)
- [evaluation/test layer](stat_tests.md)

Later registries may supply admissible content, but they do not redefine the design language defined on this page.

## V1 completion status

The design layer (Stage 0) is treated as complete for the v1 package foundation.

Framework:

- canonical dataclasses (5), main builder, route ownership resolution, completeness check, dict serialization round-trip, replication override path, explicit error classes.

Layer 0 meta axes registry (7 axes, curated 2026-04-18):

- `axis_type`: 5 values, all operational, all with real runtime wiring (fixed/sweep/conditional/nested_sweep/derived).
- `compute_mode`: 3 operational (serial/parallel_by_model/parallel_by_horizon), 3 registry_only (v1.1+).
- `experiment_unit`: 7 operational, 2 registry_only (v1.1), 3 future (v2).
- `failure_policy`: 5 operational, 3 registry_only (v1.1).
- `registry_type`: 4 operational, 1 registry_only (v1.1).
- `reproducibility_mode`: 3 operational, 1 registry_only (v1.1).
- `study_mode`: 4 operational (2 executable + 2 wrapper-route).

The design layer is a stable foundation. Later package surfaces are built above it.
