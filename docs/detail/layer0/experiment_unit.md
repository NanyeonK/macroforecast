# 4.1.1 experiment_unit

- Parent: [4.1 Layer 0: Study Setup](index.md)
- Current: `experiment_unit`
- Next: [4.1.2 failure_policy](failure_policy.md)

`experiment_unit` is the first Layer 0 decision. It names the unit of work that the study will run, compare, repeat, or hand off to a dedicated runner.

A one-path forecast is not a separate "single run" design route. It is the one-cell case of the same `comparison_sweep` grammar: if every downstream choice has one value, one cell runs; if one or more downstream axes are swept, the sweep runner expands the same path into multiple cells.

## Where It Lives In Code

| Purpose | Function or object |
|---|---|
| Registry entries and metadata | `macrocast.registry.stage0.experiment_unit.EXPERIMENT_UNIT_ENTRIES` |
| Lookup one unit | `macrocast.registry.stage0.experiment_unit.get_experiment_unit_entry` |
| Wizard-visible options | `macrocast.registry.stage0.experiment_unit.experiment_unit_options_for_wizard` |
| Default derivation | `macrocast.registry.stage0.experiment_unit.derive_experiment_unit_default` |
| Design-frame derivation | `macrocast.design.derive.derive_experiment_unit` |
| Design-frame build | `macrocast.design.build.build_design_frame` |
| Compiler default rule | `macrocast.compiler.build._rule_experiment_unit_default` |
| Compiler validation | `macrocast.compiler.build._build_stage0_and_recipe` |
| Runtime status routing | `macrocast.compiler.build._execution_status` |
| Wrapper metadata | `macrocast.compiler.build._build_wrapper_handoff` |
| Wizard read/write | `macrocast.start._read_wizard_value`, `macrocast.start._apply_wizard_value` |

## Choice Map

| Choice | Owner | Current State | Meaning |
|---|---|---|---|
| `single_target_single_generator` | `comparison_sweep` | runnable | One target, one forecasting path, one comparison cell. |
| `single_target_generator_grid` | `comparison_sweep` | runnable / sweep-aware | One target with one or more controlled downstream sweeps. |
| `multi_target_shared_design` | `comparison_sweep` | runnable | Multiple targets share one data, preprocessing, model, and evaluation design. |
| `multi_target_separate_runs` | `wrapper` | wrapper runner | Multiple targets are split into independent target-level runs. |
| `replication_recipe` | `replication` | replication runner | Re-run or adapt a source recipe with replication provenance. |
| `single_target_full_sweep` | `wrapper` | reserved | Wider full-grid wrapper grammar; not a direct compiled recipe. |
| `benchmark_suite` | `wrapper` | reserved | Bundle of benchmark recipes; requires wrapper contract. |
| `ablation_study` | `wrapper` | standalone runner only | Ablation grammar exists, but compiled-recipe wrapper handoff is not direct-run. |
| `hierarchical_forecasting_run` | `orchestrator` | future | Hierarchical forecasting placeholder. |
| `panel_forecasting_run` | `orchestrator` | future | Panel forecasting placeholder. |
| `state_space_run` | `comparison_sweep` | future | State-space forecasting placeholder. |

## Direct Comparison Units

### `single_target_single_generator`

Use when every selected downstream axis is fixed to one value.

```yaml
path:
  0_meta:
    fixed_axes:
      experiment_unit: single_target_single_generator
```

Runtime contract:

```text
route_owner     = comparison_sweep
route_contract  = single_cell_executable
runner          = execute_recipe
```

### `single_target_generator_grid`

Use when one target is evaluated across a controlled set of alternatives: model families, feature builders, Layer 2 representations, horizons, metrics, or other supported sweep axes.

```yaml
path:
  0_meta:
    fixed_axes:
      experiment_unit: single_target_generator_grid
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, random_forest]
```

Runtime contract:

```text
route_owner     = comparison_sweep
route_contract  = sweep_runner_executable when sweep_axes are present
runner          = compile_sweep_plan -> execute_sweep
```

If all grid axes collapse to one value, this becomes the same one-cell `comparison_sweep` execution as `single_target_single_generator`.

### `multi_target_shared_design`

Use when several targets share one design and should be evaluated in one compiled recipe.

```yaml
path:
  0_meta:
    fixed_axes:
      experiment_unit: multi_target_shared_design
  1_data_task:
    fixed_axes:
      target_structure: multi_target
    leaf_config:
      targets: [INDPRO, RPI]
```

Runtime contract:

```text
route_owner     = comparison_sweep
route_contract  = single_cell_executable
runner          = execute_recipe
target mode     = shared multi-target design
```

## Handoff Units

### `multi_target_separate_runs`

Use when each target should run as a separate job with its own output directory.

```text
route_owner     = wrapper
route_contract  = wrapper_handoff
runner          = execute_separate_runs
```

### `replication_recipe`

Use when a source recipe or paper recipe owns the path and the package should preserve replication provenance.

```text
route_owner     = replication
route_contract  = replication_handoff
runner          = execute_replication
```

### Reserved Wrapper And Orchestrator Units

`single_target_full_sweep`, `benchmark_suite`, `ablation_study`, `hierarchical_forecasting_run`, and `panel_forecasting_run` are valid registry concepts, but they are not ordinary direct-run cells. They compile only when a concrete wrapper/orchestrator contract exists; otherwise the compiler returns `not_supported` or keeps them out of the simple wizard.

## Derivation Rules

Full YAML may set `experiment_unit` explicitly. If omitted, the compiler derives it from target structure and sweep shape:

| Recipe Shape | Derived Unit |
|---|---|
| `replication_input` is present | `replication_recipe` |
| `leaf_config.wrapper_family` names a known unit | that wrapper family |
| `target_structure=multi_target` | `multi_target_shared_design` |
| both `model_family` and `feature_builder` are swept | `single_target_full_sweep` |
| either `model_family` or `feature_builder` is swept | `single_target_generator_grid` |
| none of the above | `single_target_single_generator` |

Simple API helpers set or derive this for you:

- `Experiment(...).to_recipe_dict()` emits `single_target_single_generator` for one fixed path.
- `Experiment(...).compare_models([...])` emits `single_target_generator_grid`.
- `Experiment(...).sweep({...})` emits `single_target_generator_grid` when any supported sweep axis has multiple values.
- The interactive wizard asks for `experiment_unit` first, then target structure and downstream axes.

## Compatibility Guards

- `multi_target_shared_design` and `multi_target_separate_runs` require `target_structure=multi_target`.
- Single-target units are incompatible with `target_structure=multi_target`.
- Direct `comparison_sweep` units can execute only if every selected downstream axis has an implemented runtime cell.
- Wrapper and replication units are not accepted by `run_compiled_recipe`; they must use their dedicated runner.

## YAML

One-cell comparison:

```yaml
path:
  0_meta:
    fixed_axes:
      experiment_unit: single_target_single_generator
      failure_policy: fail_fast
      reproducibility_mode: seeded_reproducible
      compute_mode: serial
```

Controlled comparison:

```yaml
path:
  0_meta:
    fixed_axes:
      experiment_unit: single_target_generator_grid
  3_training:
    fixed_axes:
      feature_builder: target_lag_features
    sweep_axes:
      model_family: [ar, ridge, lasso]
```
