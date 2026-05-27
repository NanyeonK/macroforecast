# Derived Study Scope

- Parent: [L0 - Meta / Study Setup](layer0.md)
- Current: derived `study_scope`

`study_scope` is runtime provenance, not a user-set L0 axis. It describes the broad study shape after the recipe has been inspected: how many targets are active and whether the recipe is a fixed path or a comparison over methods.

## Possible Values

| Value | Meaning |
|---|---|
| `one_target_one_method` | one target, one fixed method path |
| `one_target_compare_methods` | one target, controlled method comparison |
| `multiple_targets_one_method` | multiple targets, one fixed method path |
| `multiple_targets_compare_methods` | multiple targets, controlled method comparison |

## Derivation Inputs

The runtime derives the value from downstream recipe shape:

| Input | Owner |
|---|---|
| target cardinality | L1 data / target structure |
| model or method sweep shape | L3/L4/L5 recipe graph and sweep expansion |
| output cell count | execution planner |

Do not set `study_scope` under `0_meta.fixed_axes`. If a recipe carries an old explicit `study_scope`, update the recipe by removing it and let the runtime record the derived value in the manifest.

## Manifest Role

The derived value is recorded for audit and provenance. It helps readers understand whether a result came from a single path, a model comparison, a multi-target study, or a multi-target comparison.

## Non-Roles

- It is not a target selector; target names belong to L1.
- It is not a model selector; model families belong to L4.
- It is not sweepable.
- It is not part of the callable L0 authoring surface.
