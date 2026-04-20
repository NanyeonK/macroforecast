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
| `parallel_by_oos_date` | operational | `len(origin_plan) > 1` within each horizon loop | Origin-level threading inside `_rows_for_horizon`: refit_policy state is computed serially in a pre-pass (ensures determinism with `fit_once_predict_many` / `refit_every_k_steps`), then model/benchmark fits run concurrently across OOS origins. Silent no-op for short OOS ranges. |
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
  - OOS origin loop (inside `_rows_for_horizon`) builds a deterministic plan, then dispatches the per-origin compute via `ThreadPoolExecutor` when `compute_mode == "parallel_by_oos_date"` and `len(origin_plan) > 1`.

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

```yaml
# OOS-origin-level parallelism — best with long OOS ranges and fast models.
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_oos_date
  1_data_task:
    leaf_config:
      target: INDPRO
      horizons: [1, 3]
```

---

## 0.3 `experiment_unit`

**Declares the per-recipe execution shape** — how many targets / models / variants the recipe produces, and which runner handles it. This is the richest Layer 0 axis, with a dedicated entry dataclass and helper functions. Auto-derived from the rest of the recipe when not set explicitly.

### Value catalog

`operational` is subdivided by **how the unit is executed**:

- **direct** — handled by `execute_recipe()` itself (single-path or aggregated multi-target)
- **dedicated runner** — has its own top-level runner function (`runner` field on the entry)
- **wrapper handoff** — compile produces a `wrapper_handoff` payload; the actual runner is either an external wrapper or a Phase 8 consumer (`PaperReadyBundle`)

| Value | Status | Execution form | Route owner | Runner / dispatcher |
|---|---|---|---|---|
| `single_target_single_model` | operational · direct | `execute_recipe()` | single_run | — |
| `single_target_model_grid` | operational · direct | `execute_recipe()` with `model_family` sweep via `compile_sweep_plan` | single_run | — |
| `single_target_full_sweep` | operational · wrapper handoff | `wrapper_handoff` payload; executed by the wrapper runtime | wrapper | — (wrapper pending) |
| `multi_target_shared_design` | operational · direct | `execute_recipe()` multi-target path — single aggregated predictions table | single_run | — |
| `ablation_study` | operational · dedicated runner | `execute_ablation()` | wrapper | `macrocast.studies.ablation:execute_ablation` |
| `replication_recipe` | operational · dedicated runner | `execute_replication()` | replication | `macrocast.studies.replication:execute_replication` |
| `benchmark_suite` | operational · wrapper handoff | `wrapper_handoff` payload; Phase 8 `PaperReadyBundle` consumer | wrapper | — (Phase 8 pending) |
| `multi_target_separate_runs` | registry_only (v1.1) | would emit per-target artifact directories via a fan-out wrapper | wrapper | pending — Phase 10 |
| `multi_output_joint_model` | registry_only (v1.1) | would call a joint multi-output predictor adapter | single_run | pending — Phase 10 (needs joint adapters) |
| `hierarchical_forecasting_run` | future (v2) | hierarchical reconciliation orchestrator | orchestrator | Phase 11 |
| `panel_forecasting_run` | future (v2) | panel-data forecasting orchestrator | orchestrator | Phase 11 |
| `state_space_run` | future (v2) | single-run state-space predictor | single_run | Phase 11 |

### Why three operational forms?

`execute_recipe()` in v1 handles a finite set of shapes directly: one target with one model, one target with a model grid, or many targets with one shared design. Anything beyond that — a full sweep, an ablation, a replication, a benchmark suite — delegates to a purpose-built runner or to a wrapper handoff payload. The `route_owner` field on each `ExperimentUnitEntry` makes the expected dispatcher explicit.

### Functions & features

- **`ExperimentUnitEntry`** dataclass in `macrocast.registry.stage0.experiment_unit` — extends `EnumRegistryEntry` with:
  - `route_owner: "single_run" | "wrapper" | "orchestrator" | "replication"`
  - `requires_multi_target: bool`
  - `requires_wrapper: bool`
  - optional `runner: str | None` — dotted path to the dedicated runner function.
- **`get_experiment_unit_entry(experiment_unit: str) -> ExperimentUnitEntry`** — lookup by id.
- **`experiment_unit_options_for_wizard(study_mode, task)`** — returns operational options only (registry_only / future values are filtered). Use from CLI wizards / UI to offer users the currently usable shapes.
- **`derive_experiment_unit_default(study_mode, task, model_axis_mode, feature_axis_mode, wrapper_family)`** — chooses the default unit from recipe shape. Also exposed as the `experiment_unit_default` derivation rule in `DERIVATION_RULES` (§0.1 `derived_axes`).
- **`derive_experiment_unit(study_mode, execution_posture, forecast_task)`** in `macrocast.design.derive` — the framework-level hook used by `build_design_frame`. Honours `execution_posture` (wrapper / replication) first, then falls back to multi_target_shared_design for multi-target, model_grid for controlled_variation_study, and single_target_single_model otherwise.
- **Compiler integration**: `compile_recipe_dict()` auto-derives `experiment_unit` when not explicit; conflict-checks explicit declarations against the derived default; emits a `wrapper_handoff` payload when `route_owner == "wrapper"`. Values `benchmark_suite` convert the compile status to `representable_but_not_executable` (wrapper runner pending).

### Recipe usage

Implicit (recommended for most recipes):

```yaml
# single-target, single model — default derivation → single_target_single_model
path:
  1_data_task:
    fixed_axes:
      task: single_target_point_forecast
  3_training:
    fixed_axes:
      model_family: ridge

# single-target, model sweep → derivation auto-picks single_target_model_grid
path:
  1_data_task:
    fixed_axes:
      task: single_target_point_forecast
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, random_forest]

# multi-target → derivation picks multi_target_shared_design (operational, direct)
path:
  1_data_task:
    fixed_axes:
      task: multi_target_point_forecast
    leaf_config:
      targets: [INDPRO, RPI, CPIAUCSL]
```

Explicit (ablation / replication paths):

```yaml
path:
  0_meta:
    fixed_axes:
      study_mode: controlled_variation_study
      experiment_unit: ablation_study
  # …
# Run via execute_ablation(), not execute_recipe().
```

Explicit (declarative derivation via `derived_axes` from §0.1):

```yaml
path:
  0_meta:
    derived_axes:
      experiment_unit: experiment_unit_default
```

### Not implemented in v1.0

| Value | Status | Gap | Target |
|---|---|---|---|
| `multi_target_separate_runs` | registry_only | No wrapper runner that fans out per-target artifact directories. Current multi-target path aggregates into one output. | v1.1 / Phase 10 |
| `multi_output_joint_model` | registry_only | No joint multi-output predictor adapter. | v1.1 / Phase 10 |
| `hierarchical_forecasting_run` | future | No hierarchical reconciliation orchestrator. | v2 / Phase 11 |
| `panel_forecasting_run` | future | No panel-data executor. | v2 / Phase 11 |
| `state_space_run` | future | No state-space predictor family. | v2 / Phase 11 |
| `single_target_full_sweep` wrapper runner | operational · handoff | Compile produces wrapper_handoff payload; external wrapper consumes. No built-in wrapper runtime ships with v1.0. | Phase 8+ |
| `benchmark_suite` wrapper runner | operational · handoff | Same as above; Phase 8 `PaperReadyBundle` will be the first consumer. | Phase 8 |

In short: 5 of 12 values execute end-to-end inside the package (3 direct + 2 dedicated runners). 2 more compile successfully with a wrapper_handoff payload but wait for an external or Phase 8 consumer. The remaining 5 are registry or future entries with no runtime in v1.0.

---

## 0.4 `failure_policy`

**Declares how execution handles in-recipe and in-sweep failures.** Drives both `execute_recipe` (per-recipe failure branches) and `execute_sweep` (per-variant failure branches). Default is `fail_fast`.

### Value catalog

| Value | Status | Recipe-level behaviour (`execute_recipe`) | Sweep-level behaviour (`execute_sweep`) |
|---|---|---|---|
| `fail_fast` | operational | First failure re-raises (abort). | First variant failure re-raises (abort the sweep). |
| `skip_failed_cell` | operational | Same as `fail_fast` at the recipe level (a "cell" is a sweep variant, so within one recipe the cell is the whole run). | Record the failure in the study manifest and continue with the remaining variants. |
| `skip_failed_model` | operational | Record the per-target/per-model failure and continue with the remaining targets/models. | Same as `skip_failed_cell` at the sweep level (the failed variant is skipped). |
| `save_partial_results` | operational | Record the failure, set `manifest.partial_run = True`, and still persist whatever partial output exists (predictions, stat tests, importance). | Same as `skip_failed_cell` at the sweep level. |
| `warn_only` | operational | Record the failure in `failed_components`, emit a `RuntimeWarning` at the failing stage, and continue. Matches `save_partial_results` semantics with the extra warning side-effect. | Same as `save_partial_results` at the sweep level plus a `RuntimeWarning` per failed variant. |
| `retry_then_skip` | registry_only (v1.1) | Compile rejects as "representable but not executable". Needs retry loop with backoff. | — |
| `fallback_to_default_hp` | registry_only (v1.1) | Compile rejects. Needs HP-fallback wiring into the tuning backend. | — |

Note: `hard_error` was dropped in this release — it had no runtime branch distinct from `fail_fast`.

### How the sweep runner reads the policy

`execute_sweep` no longer takes a `fail_fast: bool` parameter. Instead it reads `failure_policy` from the parent recipe's `0_meta.fixed_axes.failure_policy` (or `sweep_axes`). The mapping is:

- `fail_fast` → abort on the first failed variant (`raise`).
- `skip_failed_cell`, `skip_failed_model`, `save_partial_results`, `warn_only` → continue past the failed variant, record it in the study manifest.
- `warn_only` additionally emits `RuntimeWarning(f"variant {id} failed: ...")` per failed variant.

Recipes that do not specify `failure_policy` default to `fail_fast`.

### Functions & features

- **Compiler spec**: `failure_policy_spec` emitted on `CompiledRecipeSpec` (`macrocast/compiler/build.py`).
- **Compiler guard**: `failure_policy` restricted to the 5 operational values at compile time; `retry_then_skip` / `fallback_to_default_hp` raise "representable but not executable".
- **Recipe runtime** (`macrocast/execution/build.py`): four failure sites — `prediction_build` (per-target threaded + serial branches), `stat_test_artifact`, `importance_artifact` — all share the same policy-dispatch shape. Failures are collected into `manifest.failed_components`; the manifest also carries `partial_run: bool`.
- **Sweep runtime** (`macrocast/execution/sweep_runner.py`):
  - `_extract_parent_failure_policy(plan)` reads the parent recipe's failure_policy (defaults to `fail_fast`).
  - `_CONTINUE_ON_VARIANT_FAILURE` frozenset enumerates policies that allow the sweep to continue past a failed variant.
  - `execute_sweep` emits `RuntimeWarning` per failed variant when policy is `warn_only`.
- **Studies integration** (`macrocast/studies/ablation.py`): `execute_ablation` injects `failure_policy=skip_failed_cell` into the baseline recipe by default (ablations are tolerant of individual cell failures by design). User override via the baseline recipe's `0_meta.fixed_axes.failure_policy` is honoured.

### Recipe usage

```yaml
# Fail fast (default) - abort on first failure.
path:
  0_meta:
    fixed_axes:
      failure_policy: fail_fast
```

```yaml
# Skip failed variants in a horse-race sweep; continue remaining.
path:
  0_meta:
    fixed_axes:
      failure_policy: skip_failed_cell
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, random_forest]
```

```yaml
# Multi-target recipe that tolerates per-model failures and persists partial output.
path:
  0_meta:
    fixed_axes:
      failure_policy: save_partial_results
  1_data_task:
    fixed_axes:
      task: multi_target_point_forecast
    leaf_config:
      targets: [INDPRO, RPI, CPIAUCSL]
```

```yaml
# Research exploration: log warnings instead of aborting; continue.
path:
  0_meta:
    fixed_axes:
      failure_policy: warn_only
```

### Not implemented in v1.0

| Value | Status | Gap | Target |
|---|---|---|---|
| `retry_then_skip` | registry_only | Needs a retry loop with backoff inside both `execute_recipe` failure branches and `execute_sweep` variant dispatch. | v1.1 / Phase 10 |
| `fallback_to_default_hp` | registry_only | Needs integration with the tuning backend to swap the failing model's HP for a registered default when fit fails. | v1.1 / Phase 10 |

`hard_error` was dropped: it had no runtime branch distinct from `fail_fast`, so carrying it as a separate value only created confusion.

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
