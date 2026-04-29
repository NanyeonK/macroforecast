# 4.0.1 Study Scope

- Parent: [4.0 Layer 0: Study Scope](index.md)
- Current: `study_scope`
- Next: [4.0.2 Failure Handling](failure_policy.md)

`study_scope` is the first Layer 0 decision. It answers two questions at once:

1. Is the study about one target or multiple targets?
2. Is the downstream method path fixed, or will the study compare method alternatives?

Here, "method" means the researcher's downstream design path, not only the estimator.
It can include model family, feature construction, extra preprocessing, horizon handling,
tuning, or other sweepable choices owned by later layers.

Replication is not a `study_scope` value. Replication Library entries are normal YAML recipes that already contain one of the four scopes below.

## Where It Lives In Code

| Purpose | Function or object |
|---|---|
| Registry entries and metadata | `macrocast.registry.stage0.study_scope.STUDY_SCOPE_ENTRIES` |
| Lookup one scope | `macrocast.registry.stage0.study_scope.get_study_scope_entry` |
| Wizard-visible options | `macrocast.registry.stage0.study_scope.study_scope_options_for_wizard` |
| Default derivation | `macrocast.registry.stage0.study_scope.derive_study_scope_default` |
| Design-frame derivation | `macrocast.design.derive.derive_study_scope` |
| Design-frame build | `macrocast.design.build.build_design_frame` |
| Compiler default rule | `macrocast.compiler.build._rule_study_scope_default` |
| Compiler validation | `macrocast.compiler.build._build_stage0_and_recipe` |
| Wizard read/write | `macrocast.start._read_wizard_value`, `macrocast.start._apply_wizard_value` |

## Choice Map

| Choice | Target count | Method count | Meaning |
|---|---:|---:|---|
| `one_target_one_method` | one | one | One target, one fixed downstream method path. |
| `one_target_compare_methods` | one | many | One target with one or more controlled downstream method sweeps. |
| `multiple_targets_one_method` | many | one | Multiple targets share one fixed downstream method path. |
| `multiple_targets_compare_methods` | many | many | Multiple targets evaluated across controlled downstream method alternatives. |

All four scopes use `route_owner=comparison_sweep`. If a downstream axis is swept, the parent recipe is executed through `compile_sweep_plan()` / `execute_sweep()`.

## Runtime Contracts

### `one_target_one_method`

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_one_method
  1_data_task:
    fixed_axes:
      target_structure: single_target
    leaf_config:
      target: INDPRO
```

Runtime contract:

```text
route_owner     = comparison_sweep
route_contract  = single_cell_executable
runner          = execute_recipe
```

### `one_target_compare_methods`

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_compare_methods
  1_data_task:
    fixed_axes:
      target_structure: single_target
    leaf_config:
      target: INDPRO
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

### `multiple_targets_one_method`

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: multiple_targets_one_method
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

### `multiple_targets_compare_methods`

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: multiple_targets_compare_methods
  1_data_task:
    fixed_axes:
      target_structure: multi_target
    leaf_config:
      targets: [INDPRO, RPI]
  3_training:
    sweep_axes:
      model_family: [ridge, lasso]
```

Runtime contract:

```text
route_owner     = comparison_sweep
route_contract  = sweep_runner_executable when sweep_axes are present
runner          = compile_sweep_plan -> execute_sweep
```

## Derivation Rules

Full YAML may set `study_scope` explicitly. If omitted, the compiler derives it from target structure and sweep shape:

| Recipe shape | Derived scope |
|---|---|
| `target_structure=single_target`, no method sweep | `one_target_one_method` |
| `target_structure=single_target`, method sweep present | `one_target_compare_methods` |
| `target_structure=multi_target`, no method sweep | `multiple_targets_one_method` |
| `target_structure=multi_target`, method sweep present | `multiple_targets_compare_methods` |

## Compatibility Guards

- `one_target_*` scopes require `target_structure=single_target`.
- `multiple_targets_*` scopes require `target_structure=multi_target`.
- Replication recipes should be loaded from the Replication Library as YAML; they should not set `study_scope=replication_recipe`.
- Benchmark suites live under Layer 3 `benchmark_family=benchmark_suite`, not Layer 0.
- Ablations are comparison sweeps or standalone helper APIs, not Layer 0 scopes.
- Hierarchical, panel, and state-space forecasting belong to data/model layers when implemented, not to Layer 0 Study Scope.
