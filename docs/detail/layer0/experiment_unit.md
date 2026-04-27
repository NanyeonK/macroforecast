# 4.1.2 experiment_unit

- Parent: [4.1 Layer 0: Study Setup](index.md)
- Previous: [4.1.1 research_design](research_design.md)
- Current: `experiment_unit`
- Next: [4.1.3 failure_policy](failure_policy.md)

`experiment_unit` is the runner unit. It says which execution unit owns the recipe after `research_design`, target structure, and sweep shape are known.

In most recipes, users should let the compiler derive it. Pin it only when writing a Full recipe and you want the compiler to reject any shape mismatch.

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

## Choices

Read this axis as the execution owner. In most recipes, do not choose it directly; let the compiler derive it from `research_design`, target structure, and swept axes. Pin it only when you want Full mode to reject mismatches.

### Quick Map

| Choice | Owner | Current State |
|---|---|---|
| `single_target_single_model` | `single_run` | runnable |
| `single_target_model_grid` | `single_run` | runnable / sweep-aware |
| `multi_target_shared_design` | `single_run` | runnable |
| `multi_target_separate_runs` | `wrapper` | runnable through wrapper |
| `replication_recipe` | `replication` | runnable through replication runner |
| `single_target_full_sweep` | `wrapper` | reserved grammar |
| `benchmark_suite` | `wrapper` | reserved grammar |
| `ablation_study` | `wrapper` | standalone runner only |
| `hierarchical_forecasting_run` | `orchestrator` | future |
| `panel_forecasting_run` | `orchestrator` | future |
| `state_space_run` | `single_run` | future |

### Runnable Direct Units

#### `single_target_single_model`

This is the smallest ordinary execution unit.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_path_benchmark
      experiment_unit: single_target_single_model
```

The direct runner owns it:

```text
route_owner = single_run
runner      = execute_recipe
```

#### `single_target_model_grid`

Use this when one target is evaluated across a controlled model or feature comparison.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
      experiment_unit: single_target_model_grid
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, random_forest]
```

Runtime ownership depends on how the variation is compiled:

```text
small/internal grid = execute_recipe
parent sweep        = compile_sweep_plan / execute_sweep
```

#### `multi_target_shared_design`

Use this when several targets share one design and should be evaluated in one compiled recipe.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_path_benchmark
      experiment_unit: multi_target_shared_design
  1_data_task:
    fixed_axes:
      target_structure: multi_target_point_forecast
    leaf_config:
      targets: [INDPRO, RPI]
```

The direct runner handles the shared design and writes aggregated outputs:

```text
route_owner = single_run
runner      = execute_recipe
target mode = shared design
```

### Runnable Handoff Units

#### `multi_target_separate_runs`

Use this when each target should run as a separate single-target job with its own output directory.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: orchestrated_bundle
      experiment_unit: multi_target_separate_runs
  1_data_task:
    fixed_axes:
      target_structure: multi_target_point_forecast
    leaf_config:
      targets: [INDPRO, RPI]
```

The wrapper runner owns the fan-out:

```text
route_owner = wrapper
runner      = macrocast.studies.multi_target:execute_separate_runs
```

#### `replication_recipe`

Use this when the route is replication-locked and should preserve paper-style provenance.

```yaml
recipe_id: goulet-coulombe-2021-fred-md-ridge
path:
  0_meta:
    fixed_axes:
      research_design: replication_override
      experiment_unit: replication_recipe
```

The replication runner owns it:

```text
route_owner = replication
runner      = macrocast.studies.replication:execute_replication
```

### Reserved Or Future Units

#### `single_target_full_sweep`

This is reserved grammar for a wider single-target sweep managed by a wrapper.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
      experiment_unit: single_target_full_sweep
```

Current status:

```text
status = registry_only
runtime = no general executable wrapper contract yet
```

#### `benchmark_suite`

This is reserved grammar for a wrapper-managed benchmark suite.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: orchestrated_bundle
      experiment_unit: benchmark_suite
```

Current status:

```text
status = registry_only
runtime = no compiled-recipe benchmark-suite wrapper yet
```

#### `ablation_study`

This is reserved grammar for ablation routes.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: orchestrated_bundle
      experiment_unit: ablation_study
```

Current status:

```text
status = registry_only
standalone runner = macrocast.studies.ablation:execute_ablation
compiled wrapper = not opened yet
```

#### `hierarchical_forecasting_run`

Reserved for hierarchy-aware forecasting.

```text
status = future
runtime = no current executable contract
```

#### `panel_forecasting_run`

Reserved for panel-oriented forecasting.

```text
status = future
runtime = no current executable contract
```

#### `state_space_run`

Reserved for state-space forecasting.

```text
status = future
runtime = registry placeholder; not opened as runnable route
```

## Derivation Rules

`derive_experiment_unit_default()` is the core rule.

| Recipe Shape | Derived Unit |
|---|---|
| `research_design=replication_override` | `replication_recipe` |
| `leaf_config.wrapper_family` names a known unit | that wrapper family |
| `target_structure=multi_target_point_forecast` | `multi_target_shared_design` |
| `research_design=orchestrated_bundle` | `benchmark_suite` unless a wrapper family overrides it |
| both `model_family` and `feature_builder` are swept | `single_target_full_sweep` |
| either `model_family` or `feature_builder` is swept | `single_target_model_grid` |
| none of the above | `single_target_single_model` |

If `experiment_unit` is explicit in YAML, `_build_stage0_and_recipe()` checks that it matches the unit implied by the recipe shape. A mismatch is a compile error.

## YAML

Usually derived:

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_path_benchmark
    derived_axes:
      experiment_unit: experiment_unit_default
```

Explicit Full recipe:

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_path_benchmark
      experiment_unit: single_target_single_model
```

Multi-target shared design:

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_path_benchmark
      experiment_unit: multi_target_shared_design
  1_data_task:
    fixed_axes:
      target_structure: multi_target_point_forecast
    leaf_config:
      targets: [INDPRO, RPI]
```

## Runtime Notes

Do not use `experiment_unit` as a cosmetic label. It is a contract between the compiler and the runner.

The compiler records route ownership in the tree context and wrapper handoff metadata. Direct `execute_recipe` owns `single_run` units. Wrapper and replication units require their own runner surface.
