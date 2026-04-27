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

| Choice | Status | What It Means | Main Effect |
|---|---|---|---|
| `single_path_benchmark` | operational | One resolved study path. This is the normal default for a single target, fixed representation, fixed model route. | Usually derives `experiment_unit=single_target_single_model`; if model or feature axes are swept, the compiler may derive `single_target_model_grid` or `single_target_full_sweep`. |
| `controlled_variation` | operational | A comparison where one or more axes vary while the rest of the path is fixed. | `derive_design_shape()` returns a controlled-axis variation shape; `derive_experiment_unit_default()` usually derives `single_target_model_grid`. Runner-managed sweep routes should be executed with `compile_sweep_plan` / `execute_sweep` when the compiler reports `ready_for_sweep_runner`. |
| `orchestrated_bundle` | operational route grammar | A higher-level wrapper route, such as a benchmark suite, ablation study, or multi-target fan-out. | `derive_execution_posture()` routes to `wrapper_bundle_plan`; direct `execute_recipe` is not the owner. The compiler records wrapper handoff metadata when a concrete wrapper family is supplied. |
| `replication_override` | operational route grammar | A replication-locked route. Use when a known recipe path should preserve paper-style provenance and deviations. | Derives `experiment_unit=replication_recipe`; route owner becomes `replication`; direct single-run execution is not the owner. |

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
