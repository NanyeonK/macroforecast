# Design (Stage 0)

> **Naming:** The code module was `macrocast.stage0` before 2026-04-18; it is now `macrocast.design`. "Stage 0" remains the architectural term for this pre-execution grammar layer; `macrocast.design` is the code surface. The registry layer at ``macrocast.registry.stage0`` kept its path to distinguish framework (design) from registry (Layer 0 meta).

!!! note "Design framework vs. Layer 0 meta axes registry"
    - **Design framework** — pre-execution grammar dataclasses. Lives at ``macrocast.design``. Small set of pre-execution dataclasses (`FixedDesign`, `VaryingDesign`, `ComparisonContract`, `DesignFrame`) plus one builder (`build_design_frame`).
    - **Layer 0 meta axes registry** — 6 enum catalogs consumed by the framework. Lives at ``macrocast.registry.stage0``. This page walks through all 6 in order (§0.1 through §0.6).

## Purpose

The design frame fixes the execution language of a macrocast study before later registries or recipe content are expanded. It answers these questions first:

- what kind of study is this? (`study_mode`)
- what recipe shape is this? (`experiment_unit`)
- how does each axis participate? (`axis_type`)
- what happens on failure? (`failure_policy`)
- how reproducible must it be? (`reproducibility_mode`)
- how is parallelism applied? (`compute_mode`)

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

## 0.5 `reproducibility_mode`

**Declares the deterministic-replay contract** for a recipe. Controls both (1) the seed returned by `current_seed()` for explicit random-consuming call sites and (2) the global RNG state + determinism flags that downstream libraries (NumPy, Python `random`, PyTorch, cuDNN, CUBLAS) read directly. Installed once at the start of `execute_recipe()` via `apply_reproducibility_mode()`.

Default is `seeded_reproducible`.

### Value catalog

| Value | Status | What it installs | Variant seed strategy |
|---|---|---|---|
| `strict_reproducible` | operational | Python `random.seed` + `numpy.random.seed` + `torch.manual_seed` (+ CUDA) + `cudnn.deterministic=True` + `cudnn.benchmark=False` + `torch.use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG=':4096:8'` (if unset). Emits `RuntimeWarning` if `PYTHONHASHSEED` is not set in the shell. | `hashlib.sha256(f"{recipe_id}|{variant_id}|{model_family}")` — variants get distinct but reproducible seeds. |
| `seeded_reproducible` | operational | Python `random.seed` + `numpy.random.seed` + `torch.manual_seed` (+ CUDA) with the `base_seed` from the recipe. **No cuDNN / deterministic-algorithm flags** — small numerical drift across library versions accepted. Default mode. | Same `base_seed` for every variant. |
| `best_effort` | operational | Identical install-time behaviour to `seeded_reproducible`. The label exists so CI can exclude such runs from strict-regression checks. | Same as `seeded_reproducible`. |
| `exploratory` | operational | **No-op** at install time. Whatever the caller's pre-existing RNG state is, stays untouched. | `np.random.randint(0, 2**31 - 1)` per call — fresh non-deterministic seed each time. |

All four modes are now wired to `apply_reproducibility_mode()`; there is no registry_only value on this axis.

### What strict actually guarantees (and does not)

**Controlled by the package** (runtime-installed per mode):

- Python `random` module global state.
- NumPy global RNG state.
- PyTorch CPU + CUDA seeds (when torch is importable).
- cuDNN deterministic / benchmark flags (torch present, strict only).
- `torch.use_deterministic_algorithms` (torch present, strict only).
- `CUBLAS_WORKSPACE_CONFIG` environment variable (strict only, if currently unset).

**NOT controllable at runtime** — must be set in the shell before launching Python:

- `PYTHONHASHSEED` — Python hashes dict/set keys using a per-interpreter random salt. Strict mode emits a `RuntimeWarning` when this is unset; the recommendation is `export PYTHONHASHSEED=0` before invoking macrocast.
- `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `OMP_NUM_THREADS` — BLAS libraries read these at import time. Thread-count non-determinism is generally small but observable at the 1e-15 level; pin to `=1` in the shell for bit-exact matrix operations.

**Out of scope for v1.0**:

- GPU atomics non-determinism in operations that torch does not mark as deterministic (some scatter/gather variants, some cuDNN algorithms). `warn_only=True` is passed to `use_deterministic_algorithms` so these emit a warning rather than raising.
- Driver-level floating-point variation across CUDA versions.
- Python versions (CPython <3.11 ordering of small integers, etc.).

### Manifest recording

Every `execute_recipe()` call writes a `reproducibility_applied` dict into its manifest alongside the existing `reproducibility_spec`. Keys:

- `mode` — the mode that was installed
- `python_hash_seed` — value read from `os.environ` at install time (may be `None`)
- `numpy_seed_set` — whether NumPy global state was set
- `torch_seed_set` — whether torch seed was set (requires torch installed)
- `cudnn_deterministic` / `cudnn_benchmark` — strict-mode only; `None` otherwise
- `torch_deterministic_algorithms` — strict-mode only; `None` otherwise
- `cublas_workspace_config` — resulting env var value

This is sufficient to audit whether a strict-claimed manifest actually ran under strict conditions (e.g., was `PYTHONHASHSEED` set? did torch actually apply deterministic flags?).

### Functions & features

- `apply_reproducibility_mode(*, mode, seed, configure_torch=True)` in `macrocast.execution.seed_policy` — installs the global state described above; returns the audit summary. Called from `execute_recipe` after the `ReproducibilityContext` is pinned.
- `current_seed(model_family=None)` — returns a variant-aware seed under the installed mode; used by explicit random-consuming call sites.
- `resolve_seed(*, recipe_id, variant_id, reproducibility_spec, model_family=None)` — the underlying function that `current_seed` wraps.
- `VALID_MODES` frozenset in `seed_policy` — enumerates the 4 accepted mode strings.
- Compiler guard: `strict_reproducible` and `seeded_reproducible` require `leaf_config.random_seed`. `best_effort` and `exploratory` do not.

### Recipe usage

```yaml
# Strict bit-identical reproducibility. Also run with `PYTHONHASHSEED=0` in shell.
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: strict_reproducible
  5_output_provenance:
    leaf_config:
      random_seed: 42
```

```yaml
# Default: fixed seed, no strict deterministic flags.
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: seeded_reproducible
  5_output_provenance:
    leaf_config:
      random_seed: 42
```

```yaml
# Explicit best-effort marker (for CI filters).
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: best_effort
```

```yaml
# Ad-hoc exploration: fresh seed per call, no reproducibility claim.
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: exploratory
```

### Shell recipe for reviewer-grade strict runs

```bash
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
macrocast run --recipe strict.yaml
```

See also: `docs/dev/reproducibility_policy.md`, `macrocast.execution.seed_policy`.

---

## 0.6 `study_mode`

**Top-level research identity** of the study. Determines which execution route the compiler produces and which runner the user invokes. Default is `single_path_benchmark_study`.

Default recorded on study manifests (when swept via `execute_sweep`) is `controlled_variation_study`.

### Value catalog

Operational values are subdivided by **how they are executed**:

- **direct** — `execute_recipe()` handles the recipe path as-is.
- **sweep runner** — `execute_sweep()` expands and executes variants.
- **dedicated runner** — a purpose-built top-level function drives execution.
- **wrapper handoff (pending consumer)** — compile emits a `wrapper_handoff` payload but no in-package runner ships yet.

| Value | Status | Execution form | Compile status | Runner |
|---|---|---|---|---|
| `single_path_benchmark_study` | operational | direct | executable | `execute_recipe` |
| `controlled_variation_study` | operational | sweep runner | executable | `execute_sweep` (internally compiles each variant to single-path) |
| `replication_override_study` | operational | dedicated runner | executable (since 2026-04-20) | `execute_replication` (applies overrides, compiles, runs via `execute_recipe`) |
| `orchestrated_bundle_study` | operational | wrapper handoff (pending) | representable_but_not_executable | Phase 8 `PaperReadyBundle` consumer — not shipped yet |

### Which one to pick

- **You are running one recipe against a baseline** → `single_path_benchmark_study`. Default, no need to set explicitly.
- **You are sweeping an axis (horse race)** → `controlled_variation_study`. Required by `execute_sweep`.
- **You are reproducing a prior study with locked overrides** → `replication_override_study`. Then call `execute_replication(source_recipe_dict=..., overrides=...)`.
- **You are bundling multiple recipes for paper artifact (Phase 8)** → `orchestrated_bundle_study`. Compile-only today; the consumer ships in Phase 8 (`PaperReadyBundle`).

### Functions & features

- `DEFAULT_STUDY_MODE = "controlled_variation_study"` in `macrocast.execution.sweep_runner` — the value recorded on study manifests by default when `execute_sweep` is invoked without an override.
- `macrocast.design.normalize.normalize_study_mode(value)` — rejects unknown values with `DesignValidationError`. The 4 accepted values live in `_ALLOWED_STUDY_MODES`.
- `macrocast.design.derive.derive_execution_posture(study_mode, design_shape, replication_input, experiment_unit)`:
  - `replication_override_study` → `replication_locked_plan`
  - `orchestrated_bundle_study` → `wrapper_bundle_plan`
  - otherwise driven by design_shape
- `macrocast.compiler.build.compile_recipe_dict()`:
  - Rejects `study_mode` that is not one of the 4 registered values.
  - Emits `representable_but_not_executable` + wrapper-route warning for `orchestrated_bundle_study` (until Phase 8 consumer lands).
  - `replication_override_study` now passes as `executable` (was rejected until 2026-04-20 cleanup).
- `macrocast.execution.sweep_runner.execute_sweep(..., study_mode=...)` — parameter; study manifest records the choice.
- `macrocast.studies.replication.execute_replication(...)` — dedicated runner for `replication_override_study`. Accepts source recipe with any study_mode; applies overrides; compiles and runs.

### Recipe usage

```yaml
# Single-path benchmark study (default; omitting study_mode gets this).
path:
  0_meta:
    fixed_axes:
      study_mode: single_path_benchmark_study
```

```yaml
# Horse-race sweep.
path:
  0_meta:
    fixed_axes:
      study_mode: controlled_variation_study
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, random_forest]
```

```yaml
# Replication of a prior recipe — marks the source; execute_replication reruns.
path:
  0_meta:
    fixed_axes:
      study_mode: replication_override_study
```

```yaml
# Orchestrated bundle — compile-only in v1.0; Phase 8 PaperReadyBundle will consume.
path:
  0_meta:
    fixed_axes:
      study_mode: orchestrated_bundle_study
  5_output_provenance:
    leaf_config:
      wrapper_family: benchmark_suite     # required by wrapper_handoff contract
      bundle_label: my-bundle             # required
```

### Relation to other Layer 0 axes

- `experiment_unit` (§0.3) interacts with `study_mode`. E.g., `orchestrated_bundle_study` combined with `experiment_unit=benchmark_suite` produces a wrapper_handoff payload; `replication_override_study` defaults `experiment_unit` to `replication_recipe`.
- `reproducibility_mode` (§0.5) applies independently; a `replication_override_study` with `strict_reproducible` produces a byte-identical replay.
- `failure_policy` (§0.4) drives per-variant and per-recipe failure handling regardless of study_mode.

### Not implemented in v1.0

| Value | Gap | Target |
|---|---|---|
| `orchestrated_bundle_study` consumer | Phase 8 `PaperReadyBundle` is the intended runner — not shipped. Compile handoff exists; runtime consumer pending. | Phase 8 |

All other study_mode values execute end-to-end in v1.0.

---

## Framework surface

The 6 axes above are consumed by a compact framework that the user typically touches only via `build_design_frame()`:

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

Layer 0 meta axes registry (6 axes, curated 2026-04-20):

- `axis_type`: 5 values, all operational, all with real runtime wiring (fixed/sweep/conditional/nested_sweep/derived).
- `compute_mode`: 3 operational (serial/parallel_by_model/parallel_by_horizon), 3 registry_only (v1.1+).
- `experiment_unit`: 7 operational, 2 registry_only (v1.1), 3 future (v2).
- `failure_policy`: 5 operational, 3 registry_only (v1.1).
- `reproducibility_mode`: 3 operational, 1 registry_only (v1.1).
- `study_mode`: 4 operational (2 executable + 2 wrapper-route).

The design layer is a stable foundation. Later package surfaces are built above it.
