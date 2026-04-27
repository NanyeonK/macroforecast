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

| Choice | Status | Route Owner | What It Means | Runtime Contract |
|---|---:|---|---|---|
| `single_target_single_model` | operational | `single_run` | One target, one model path, one representation path. | Normal direct recipe execution through `execute_recipe`. |
| `single_target_model_grid` | operational | `single_run` | One target with a controlled model or feature comparison. | Can be a single-run internal comparison or a sweep-runner route depending on axis placement. |
| `single_target_full_sweep` | registry_only | `wrapper` | Reserved grammar for a wider single-target sweep managed by a wrapper. | Compiler can name the unit, but current runtime does not expose a general executable wrapper contract for it. |
| `multi_target_separate_runs` | operational | `wrapper` | Multi-target fan-out: each target runs as a separate single-target run with its own output directory. | Runner is `macrocast.studies.multi_target:execute_separate_runs`; direct `execute_recipe` is not the owner. |
| `multi_target_shared_design` | operational | `single_run` | Multi-target shared-design run: one compiled recipe evaluates all targets under the same design. | Handled by the multi-target path inside `execute_recipe`; outputs are aggregated. |
| `hierarchical_forecasting_run` | future | `orchestrator` | Reserved for hierarchy-aware forecasting. | No current executable runtime contract. |
| `panel_forecasting_run` | future | `orchestrator` | Reserved for panel-oriented forecasting. | No current executable runtime contract. |
| `state_space_run` | future | `single_run` | Reserved for state-space forecasting. | Registry placeholder; not opened as a runnable route. |
| `replication_recipe` | operational | `replication` | Replication-locked unit. | Runner is `macrocast.studies.replication:execute_replication`; compiler reports replication ownership rather than ordinary direct run ownership. |
| `benchmark_suite` | registry_only | `wrapper` | Reserved wrapper-managed benchmark suite grammar. | No executable compiled-recipe wrapper contract in current runtime. |
| `ablation_study` | registry_only | `wrapper` | Reserved ablation route. | Standalone ablation runner exists as `macrocast.studies.ablation:execute_ablation`, but this is not yet a compiled-recipe wrapper contract. |

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
