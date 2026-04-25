# Layer 0 Meta Audit

Layer 0 decides what kind of research design a recipe represents before data, preprocessing, models, metrics, or artifacts are interpreted.

For the public `Experiment` API, this layer answers one question:

```text
Is this a single default run, a one-axis controlled comparison, a multi-axis grid, a wrapper bundle, or a replication?
```

## Layer 0 Surfaces

Layer 0 is split across two code surfaces.

| Surface | Files | Role |
|---------|-------|------|
| registry axes | `macrocast/registry/stage0/` | allowed values and support status |
| design frame | `macrocast/design/` | derived research shape, execution posture, route owner |
| compiler lowering | `macrocast/compiler/build.py` | maps recipe axes into a `DesignFrame` and execution status |

## Registry Axes

Current source axes:

| Axis | Public role | Current default |
|------|-------------|-----------------|
| `research_design` | public concept, mostly derived by `Experiment` | `single_path_benchmark` |
| `experiment_unit` | internal/advanced route selector | derived |
| `compute_mode` | advanced execution control | `serial` |
| `failure_policy` | advanced execution control | `fail_fast` |
| `reproducibility_mode` | public advanced option | `seeded_reproducible` |
| `axis_type` | internal registry grammar | not a user recipe choice |

Stale note: old `study_mode` / `registry_type` pycache files may exist, but current source files do not define active axes for them.

## Full Route Contract

Layer 0 separates grammar from the runner that is allowed to execute it.

| Contract | Meaning | Direct `run_compiled_recipe` |
|----------|---------|------------------------------|
| `single_run_executable` | one fully specified single-run recipe | allowed |
| `sweep_runner_executable` | parent recipe with sweep axes; variants run through `compile_sweep_plan` / `execute_sweep` | parent blocked |
| `wrapper_handoff` | wrapper-owned bundle such as full sweep, benchmark suite, ablation, or separate multi-target runs | blocked |
| `replication_handoff` | replication-owned run; must go through `execute_replication` with source/override contract | blocked |
| `orchestrator_handoff` | future orchestrator-owned route | blocked |
| `not_supported_route` | grammar exists, but no runner contract is assigned | blocked |

The compiler records this in `manifest["tree_context"]["route_contract"]`.
Only `route_owner="single_run"` can be executed by `run_compiled_recipe`.

## Research Design

| Value | Meaning | MVP public API |
|-------|---------|----------------|
| `single_path_benchmark` | one fixed recipe path | `forecast(...)` / `Experiment(...).run()` |
| `controlled_variation` | one controlled comparison surface | `.compare_models(...)`, `.sweep({"models": ...})` |
| `orchestrated_bundle` | wrapper-managed bundle | advanced/detail only |
| `replication_override` | replication-locked route | future public replication API |

`Experiment` should set `research_design` automatically. Users should not need to pass this in the simple path.

## Experiment Unit

`experiment_unit` is derived from research shape and task.

| Value | Route owner | Runtime status | Public role |
|-------|-------------|----------------|-------------|
| `single_target_single_model` | `single_run` | executable | simple default |
| `single_target_model_grid` | `single_run` | executable through sweep runner | one controlled single-run axis, usually model comparison |
| `single_target_full_sweep` | `wrapper` | registry_only / not_supported | dropped until a wrapper runner exists |
| `multi_target_shared_design` | `single_run` | executable | advanced after Layer 1 audit |
| `multi_target_separate_runs` | `wrapper` | executable wrapper path | advanced after Layer 1 audit |
| `replication_recipe` | `replication` | ready_for_replication_runner | replication API |
| `benchmark_suite` | `wrapper` | registry_only / not_supported | dropped until a wrapper runner exists |
| `ablation_study` | `wrapper` | registry_only / not_supported | standalone runner only; no compiled wrapper contract |
| `hierarchical_forecasting_run` | `orchestrator` | future | closed |
| `panel_forecasting_run` | `orchestrator` | future | closed |
| `state_space_run` | `single_run` | future | closed |

The simple API should not expose `experiment_unit` directly. It should expose clearer methods like `compare_models`, future `grid`, future `replicate`, and future `ablate`.

Full contract notes:

- `single_target_model_grid` is historical naming. In the current full contract it means one controlled single-run axis, usually `model_family`.
- `multi_target_separate_runs` is a wrapper handoff with a concrete `execute_separate_runs` runner; it is not a direct executable compiled recipe.
- `single_target_full_sweep`, `benchmark_suite`, and `ablation_study` are wrapper handoffs without compiled-recipe runner contracts and compile as `not_supported`.
- `replication_recipe` is a replication handoff. It is consumed by `execute_replication`, not by `run_compiled_recipe`.
- `multi_target_shared_design` remains a single-run executable route because `execute_recipe` owns its shared-design multi-target path.

## DesignFrame Derived Fields

`macrocast.design` derives:

- `design_shape`
- `execution_posture`
- `experiment_unit`
- route owner via `resolve_route_owner`

Current shapes:

| Shape | Meaning |
|-------|---------|
| `one_fixed_env_one_tool_surface` | one fixed run |
| `one_fixed_env_multi_tool_surface` | multiple model/tools under one fixed design |
| `one_fixed_env_controlled_axis_variation` | one or more controlled axes varied |
| `wrapper_managed_multi_run_bundle` | wrapper/orchestrator owns execution |

Current postures:

| Posture | Meaning |
|---------|---------|
| `single_run_recipe` | direct recipe execution |
| `single_run_with_internal_sweep` | sweep runner can expand variants |
| `wrapper_bundle_plan` | wrapper runtime required |
| `replication_locked_plan` | replication runtime/contract required |

## Finding: Fixed Feature Recipe Was Misclassified

Before this audit, `derive_design_shape` treated any non-empty `feature_recipes` tuple as a controlled axis. That made a single fixed feature recipe look like controlled variation.

Correct behavior:

```text
one model + one feature recipe -> one_fixed_env_one_tool_surface
one model + multiple feature recipes -> one_fixed_env_controlled_axis_variation
multiple models + single_path_benchmark -> one_fixed_env_multi_tool_surface
controlled_variation -> one_fixed_env_controlled_axis_variation
```

The derivation now counts feature/preprocess/tuning axes only when they contain more than one value.
When that produces a single-run controlled axis, Layer 0 keeps the route in
`single_target_model_grid` until a finer-grained experiment unit is introduced.
The name is historical; the current contract is "one controlled single-run
axis", not strictly "model axis only".

## Current Simple API Mapping

| User action | Layer 0 mapping |
|-------------|-----------------|
| `mc.forecast(...)` | `single_path_benchmark`, `single_target_single_model`, `single_run_recipe` |
| `Experiment(...).run()` | same as `forecast` when no sweep axes exist |
| `.compare_models([...])` | `controlled_variation`, `single_target_model_grid`, `single_run_with_internal_sweep` |
| `.sweep({"models": [...]})` | same as `compare_models` |
| fixed `.use_preprocessor(...)` | still single path unless models are compared |
| fixed `.use_target_transformer(...)` | still single path unless models are compared |
| `.use_sd_inferred_tcodes()` | data/preprocessing policy, not a Layer 0 design change |

## Full Scenario Matrix

| Scenario | Route contract | Compile status | Runner |
|----------|----------------|----------------|--------|
| single default | `single_run_executable` | `executable` | `run_compiled_recipe` / `Experiment.run` |
| model comparison parent | `sweep_runner_executable` | `ready_for_sweep_runner` | `execute_sweep` |
| model comparison variant | `single_run_executable` | `executable` | `run_compiled_recipe` |
| feature-only comparison parent | `sweep_runner_executable` | `ready_for_sweep_runner` | `execute_sweep` |
| preprocessing sweep parent | `sweep_runner_executable` | `ready_for_sweep_runner` | blocked in simple; Layer 2 governs variants |
| full sweep explicit | `wrapper_handoff` | `not_supported` | dropped until a wrapper runner exists |
| benchmark suite | `wrapper_handoff` | `not_supported` | dropped until a wrapper runner exists |
| ablation study | `wrapper_handoff` | `not_supported` | standalone `AblationSpec` runner only; no compiled-recipe wrapper contract |
| replication override | `replication_handoff` | `ready_for_replication_runner` | `execute_replication` |
| multi-target shared design | `single_run_executable` | `executable` | `execute_recipe` |
| multi-target separate runs | `wrapper_handoff` | `ready_for_wrapper_runner` | `execute_separate_runs` |

`run_compiled_recipe` rejects every non-`single_run` route. A runner-ready
status means a dedicated runner can consume the recipe; `not_supported` means
the route remains in the registry but must not be exposed as runnable.

## Run Policy Axes

`compute_mode`, `failure_policy`, and `reproducibility_mode` are policy axes. They do not change the research design by themselves.

| Axis | Executable values | Not-supported registry values |
|------|-------------------|---------------------------|
| `compute_mode` | `serial`, `parallel_by_model`, `parallel_by_horizon`, `parallel_by_target`, `parallel_by_oos_date` | `parallel_by_trial`, `distributed_cluster` |
| `failure_policy` | `fail_fast`, `skip_failed_cell`, `skip_failed_model`, `save_partial_results`, `warn_only` | `retry_then_skip`, `fallback_to_default_hp` |
| `reproducibility_mode` | `strict_reproducible`, `seeded_reproducible`, `best_effort`, `exploratory` | none |

`strict_reproducible` and `seeded_reproducible` require `leaf_config.random_seed`.

## Axis Grammar

`axis_type` is registry grammar metadata, not a user-facing research-design choice.

| Type | Current meaning |
|------|-----------------|
| `fixed` | value is fixed in one recipe path |
| `sweep` | Cartesian sweep value list |
| `nested_sweep` | non-uniform dependent sweep groups expanded by `compile_sweep_plan` |
| `conditional` | conditionally active axis, represented by compiler selections |
| `derived` | value resolved from recipe state, e.g. `experiment_unit_default` |

## Closed Until Later Layers

Layer 0 can represent more designs than simple API should expose today.

| Desired public feature | Layer 0 need | Blocker |
|------------------------|--------------|---------|
| preprocessing-only sweep | controlled variation with preprocessing axis recorded in `VaryingDesign` | fixed Layer 2 contracts execute, but public sweep governance and reporting are not finalized |
| model x preprocessing grid | explicit advanced grid, likely wrapper or advanced internal sweep | governance and result interpretation not fixed |
| custom benchmark suite | wrapper bundle or multi-benchmark design | Layer 4 benchmark/result contract not public |
| custom metric ranking | controlled variation with metric direction metadata | Layer 4 custom metric contract not public |
| replication | replication-locked plan | public API and artifact diff contract not finalized |
| ablation | wrapper bundle | ablation public API not finalized |
| multi-target experiment | shared design or separate runs | Layer 1 target contract needs audit |

## Recommendation

Keep simple Layer 0 closed to three public shapes for now:

1. single default run
2. model comparison
3. fixed custom extension inside either of the above

After Layer 1 and Layer 2 audits, add explicit public methods instead of overloading `.sweep()`:

```python
exp.sweep_preprocessing(...)
exp.grid(...)
exp.replicate(...)
exp.ablate(...)
```

This keeps the simple API honest: each method names the research design it creates.
