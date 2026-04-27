# 4.1.1 research_design

- Parent: [4.1 Layer 0: Study Setup](index.md)
- Previous: none inside Layer 0
- Current: `research_design`
- Next: [4.1.2 experiment_unit](experiment_unit.md)

`research_design` is the first user-facing Layer 0 choice. It names the study route before data, representation, and models are selected.

It answers: is this one resolved forecasting path, a controlled comparison, a wrapper bundle, or a replication route?

## Where It Lives In Code

| Purpose | Function or object |
|---|---|
| Registry entries | `macrocast.registry.stage0.research_design.AXIS_DEFINITION` |
| String validation | `macrocast.design.normalize.normalize_research_design` |
| Build full Stage 0 frame | `macrocast.design.build.build_design_frame` |
| Derive design shape | `macrocast.design.derive.derive_design_shape` |
| Derive execution posture | `macrocast.design.derive.derive_execution_posture` |
| Derive default runner unit | `macrocast.registry.stage0.experiment_unit.derive_experiment_unit_default` |
| Compiler default rule | `macrocast.compiler.build._rule_experiment_unit_default` |
| Compiler recipe build | `macrocast.compiler.build._build_stage0_and_recipe` |
| Wizard read/write | `macrocast.start._read_wizard_value`, `macrocast.start._apply_wizard_value` |

The axis is not an executor function by itself. It is a route selector. The compiler uses it to derive `experiment_unit`, `design_shape`, `execution_posture`, `route_owner`, and later execution status.

## Choices

Read this axis as a route decision. Pick the block that matches the study you want, copy the YAML shape, then let `experiment_unit` derive unless you need to pin it.

### Quick Map

| Choice | Route Type | Default Owner |
|---|---|---|
| `single_path_benchmark` | one resolved path | direct recipe |
| `controlled_variation` | controlled comparison | direct recipe or sweep runner |
| `orchestrated_bundle` | wrapper bundle grammar | wrapper runner |
| `replication_override` | replication route | replication runner |

### `single_path_benchmark`

Use this for the ordinary one-path case: one target design, one representation, one model family, one evaluation route.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_path_benchmark
```

Typical compiler result:

```text
research_design = single_path_benchmark
experiment_unit = single_target_single_model
route_owner     = single_run
```

If downstream model or feature axes are swept, the compiler may derive `single_target_model_grid` or `single_target_full_sweep` instead.

### `controlled_variation`

Use this when the point of the study is a controlled comparison. One axis, or a small set of axes, varies while the rest of the recipe stays fixed.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, random_forest]
```

Typical compiler result:

```text
research_design = controlled_variation
design_shape    = one_fixed_env_controlled_axis_variation
experiment_unit = single_target_model_grid
```

If the variation is represented as a parent sweep, the compiler reports `ready_for_sweep_runner`. In that case, run through `compile_sweep_plan` / `execute_sweep`, not as one ordinary leaf recipe.

### `orchestrated_bundle`

Use this when a higher-level wrapper owns the run. Examples are benchmark suites, ablation studies, or multi-target fan-out recipes.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: orchestrated_bundle
    leaf_config:
      wrapper_family: benchmark_suite
```

Typical compiler result:

```text
research_design   = orchestrated_bundle
execution_posture = wrapper_bundle_plan
route_owner       = wrapper
```

Direct `execute_recipe` is not the owner. The compiler records wrapper handoff metadata when a concrete wrapper family is supplied.

### `replication_override`

Use this when the recipe is anchored to a paper or known replication path and deviations must be explicit.

```yaml
recipe_id: goulet-coulombe-2021-fred-md-ridge
path:
  0_meta:
    fixed_axes:
      research_design: replication_override
```

Typical compiler result:

```text
research_design = replication_override
experiment_unit = replication_recipe
route_owner     = replication
```

Direct single-run execution is not the owner. The replication runner preserves replication provenance and records deviations from the original recipe.

## Selection Order

Pick `research_design` first. It constrains the next Layer 0 axis, `experiment_unit`.

Typical order:

1. Choose `research_design`.
2. Let the compiler derive `experiment_unit` unless you are writing a Full recipe that intentionally pins it.
3. Set `failure_policy`, `reproducibility_mode`, and `compute_mode`.

## Derived Contracts

`research_design` flows through two derivation levels:

1. `derive_design_shape()` maps the study route and varying axes to a design shape such as `one_fixed_env_one_tool_surface`, `one_fixed_env_controlled_axis_variation`, or `wrapper_managed_multi_run_bundle`.
2. `derive_execution_posture()` maps that design shape to a posture such as `single_run_recipe`, `single_run_with_internal_sweep`, `wrapper_bundle_plan`, or `replication_locked_plan`.

Then `derive_experiment_unit_default()` maps the route and recipe shape to the default `experiment_unit`.

## YAML

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_path_benchmark
```

For a controlled variation:

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
```

For replication:

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: replication_override
```

## Runtime Notes

`single_path_benchmark` is the default route for ordinary runnable recipes.

`controlled_variation` can produce a parent recipe that must be expanded by the sweep compiler. The important runtime distinction is whether the compiler says `executable` or `ready_for_sweep_runner`.

`orchestrated_bundle` and `replication_override` are not ordinary single-run calls. They route through wrapper or replication ownership. The compiler will surface that through execution status, warnings, and wrapper handoff metadata.
