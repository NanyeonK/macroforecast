# Layer 0 Meta Audit

Layer 0 decides the execution grammar a recipe represents before data, preprocessing, models, metrics, or artifacts are interpreted.

For the public `Experiment` API, this layer answers one question:

```text
What study scope is being run, compared, repeated, or handed off?
```

## Layer 0 Surfaces

Layer 0 is split across two code surfaces.

| Surface | Files | Role |
|---------|-------|------|
| registry axes | `macrocast/registry/stage0/` | allowed values and support status |
| design frame | `macrocast/design/` | derived execution shape, execution posture, route owner |
| compiler lowering | `macrocast/compiler/build.py` | maps recipe axes into a `DesignFrame` and execution status |

## Registry Axes

Current source axes:

| Axis | Public role | Current default |
|------|-------------|-----------------|
| `study_scope` | public execution-unit selector | `one_target_one_method` |
| `compute_mode` | advanced execution control | `serial` |
| `failure_policy` | advanced execution control | `fail_fast` |
| `reproducibility_mode` | public advanced option | `seeded_reproducible` |
| `axis_type` | internal registry grammar | not a user recipe choice |

Layer 0 has five active source axes. `axis_type` is intentionally outside the
Navigator decision tree because it describes recipe grammar (`fixed`,
`sweep`, `nested_sweep`, `conditional`, `derived`) rather than a research
choice.

Stale note: old `study_mode` / `registry_type` pycache files may exist, but
current source files do not define active axes for them.

## 2026-04-27 Census Finding

The live registry / Navigator census shows:

| Item | Finding | Decision |
|---|---|---|
| Active Layer 0 axes | 5 registry axes, 34 total values. | Keep. The old route axis was removed; `study_scope` is the first user-facing axis. |
| Primary Navigator axes | 4 axes: `study_scope`, `failure_policy`, `reproducibility_mode`, `compute_mode`. | Keep. These affect route, reproducibility, failure handling, or compute posture. |
| Internal axis | `axis_type`. | Keep hidden from the primary Navigator tree; document it as registry grammar. |
| Study Scope handoffs | Replication, benchmark suites, ablations, hierarchical, panel, and state-space routes are not Layer 0 Study Scope values. | Keep Study Scope to the four target/method cardinality branches. |
| Main open Layer 0 question | Whether benchmark/ablation wrapper families should become real public simple API methods. | Defer until wrapper result/artifact contracts are audited after Layers 4-6. |

## Full Route Contract

Layer 0 separates grammar from the runner that is allowed to execute it.

| Contract | Meaning | Direct `run_compiled_recipe` |
|----------|---------|------------------------------|
| `single_cell_executable` | one fully specified comparison cell | allowed |
| `sweep_runner_executable` | parent recipe with sweep axes; variants run through `compile_sweep_plan` / `execute_sweep` | parent blocked |
| `wrapper_handoff` | wrapper-owned bundle such as full sweep, benchmark suite, ablation, or separate multi-target runs | blocked |
| `replication_handoff` | replication-owned run; must go through `execute_replication` with source/override contract | blocked |
| `orchestrator_handoff` | future orchestrator-owned route | blocked |
| `not_supported_route` | grammar exists, but no runner contract is assigned | blocked |

The compiler records this in `manifest["tree_context"]["route_contract"]`.
Only `route_owner="comparison_sweep"` can be executed by `run_compiled_recipe`.

## Study Scope

`study_scope` is the first user-facing Layer 0 choice. It is explicit in Full YAML or derived from target structure and sweep shape.

| Value | Route owner | Runtime status | Public role |
|-------|-------------|----------------|-------------|
| `one_target_one_method` | `comparison_sweep` | executable | one target and one fixed method path |
| `one_target_compare_methods` | `comparison_sweep` | executable through sweep runner when sweeps are present | one target and method alternatives |
| `multiple_targets_one_method` | `comparison_sweep` | executable | multiple targets and one fixed method path |
| `multiple_targets_compare_methods` | `comparison_sweep` | executable through sweep runner when sweeps are present | multiple targets and method alternatives |

Full contract notes:

- Replication recipes are YAML entries in the Replication Library, not `study_scope` values.
- Benchmark suites live under Layer 3 `benchmark_family=benchmark_suite`, not Layer 0.
- Ablation studies are comparison sweeps or standalone helpers, not Layer 0 scopes.
- Hierarchical, panel, and state-space forecasting belong to future data/model-layer contracts.

## DesignFrame Derived Fields

`macrocast.design` derives:

- `design_shape`
- `execution_posture`
- `study_scope`
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
| `comparison_cell` | direct recipe execution |
| `comparison_sweep_plan` | sweep runner can expand variants |
| `wrapper_bundle_plan` | wrapper runtime required |
| `replication_locked_plan` | replication runtime/contract required |

## Finding: Fixed Feature Recipe Was Misclassified

Before this audit, `derive_design_shape` treated any non-empty `feature_recipes` tuple as a controlled axis. That made a single fixed feature recipe look like controlled variation.

Correct behavior:

```text
one model + one feature recipe -> one_fixed_env_one_tool_surface
one model + multiple feature recipes -> one_fixed_env_controlled_axis_variation
multiple models or swept axes -> one_fixed_env_controlled_axis_variation
```

The derivation now counts feature/preprocess/tuning axes only when they contain more than one value.
When that produces a controlled axis, Layer 0 keeps the route in
`one_target_compare_methods` until a finer-grained experiment unit is introduced.
The current contract is "one controlled comparison surface", not strictly "model axis only".

## Current Simple API Mapping

| User action | Layer 0 mapping |
|-------------|-----------------|
| `mc.forecast(...)` | `one_target_one_method`, `single_cell_executable` |
| `Experiment(...).run()` | same as `forecast` when no sweep axes exist |
| `.compare_models([...])` | `one_target_compare_methods`, `sweep_runner_executable` |
| `.sweep({"models": [...]})` | same as `compare_models` |
| fixed `.use_preprocessor(...)` | still single path unless models are compared |
| fixed `.use_target_transformer(...)` | still single path unless models are compared |
| `.use_sd_inferred_tcodes()` | data/preprocessing policy, not a Layer 0 design change |

## Full Scenario Matrix

| Scenario | Route contract | Compile status | Runner |
|----------|----------------|----------------|--------|
| single default | `single_cell_executable` | `executable` | `run_compiled_recipe` / `Experiment.run` |
| model comparison parent | `sweep_runner_executable` | `ready_for_sweep_runner` | `execute_sweep` |
| model comparison variant | `single_cell_executable` | `executable` | `run_compiled_recipe` |
| feature-only comparison parent | `sweep_runner_executable` | `ready_for_sweep_runner` | `execute_sweep` |
| preprocessing sweep parent | `sweep_runner_executable` | `ready_for_sweep_runner` | blocked in simple; Layer 2 governs variants |
| full sweep explicit | `wrapper_handoff` | `not_supported` | dropped until a wrapper runner exists |
| benchmark suite | `wrapper_handoff` | `not_supported` | dropped until a wrapper runner exists |
| ablation study | `wrapper_handoff` | `not_supported` | standalone `AblationSpec` runner only; no compiled-recipe wrapper contract |
| replication override | `replication_handoff` | `ready_for_replication_runner` | `execute_replication` |
| multi-target shared design | `single_cell_executable` | `executable` | `execute_recipe` |
| multi-target separate runs | `wrapper_handoff` | `ready_for_wrapper_runner` | `execute_separate_runs` |

`run_compiled_recipe` rejects every non-`comparison_sweep` route. A runner-ready
status means a dedicated runner can consume the recipe; `not_supported` means
the route remains in the registry but must not be exposed as runnable.

## Run Policy Axes

`compute_mode`, `failure_policy`, and `reproducibility_mode` are policy axes. They do not change the study scope by themselves.

| Axis | Executable values | Not-supported registry values |
|------|-------------------|---------------------------|
| `compute_mode` | `serial` (default), `parallel_by_model`, `parallel_by_horizon`, `parallel_by_target`, `parallel_by_oos_date` | none |
| `failure_policy` | `fail_fast` (default), `skip_failed_cell`, `skip_failed_model`, `save_partial_results`, `warn_only` | none |
| `reproducibility_mode` | `strict_reproducible`, `seeded_reproducible` (default), `best_effort`, `exploratory` | none |

`strict_reproducible` and `seeded_reproducible` require `leaf_config.random_seed`.

## Axis Grammar

`axis_type` is registry grammar metadata, not a user-facing research-design choice.

| Type | Current meaning |
|------|-----------------|
| `fixed` | value is fixed in one recipe path |
| `sweep` | Cartesian sweep value list |
| `nested_sweep` | non-uniform dependent sweep groups expanded by `compile_sweep_plan` |
| `conditional` | conditionally active axis, represented by compiler selections |
| `derived` | value resolved from recipe state, e.g. `study_scope_default` |

## Closed Until Later Layers

Layer 0 can represent more designs than simple API should expose today.

| Desired public feature | Layer 0 need | Blocker |
|------------------------|--------------|---------|
| preprocessing-only sweep | controlled variation with preprocessing axis recorded in `VaryingDesign` | fixed Layer 2 contracts execute, but public sweep governance and reporting are not finalized |
| model x preprocessing grid | explicit advanced grid, likely wrapper or advanced internal sweep | governance and result interpretation not fixed |
| custom benchmark suite | wrapper bundle or multi-benchmark design | Layer 4 benchmark/result contract not public |
| custom metric ranking | controlled variation with metric direction metadata | Layer 4 custom metric contract not public |
| replication | replication-locked plan | CLI/Navigator route exists; simple high-level replication API and artifact diff UX can still improve |
| ablation | wrapper bundle | ablation public API and wrapper artifact contract not finalized |
| multi-target experiment | shared design or separate runs | shared-design and separate-run paths exist; per-target/hierarchical reporting still needs later artifact audit |

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
