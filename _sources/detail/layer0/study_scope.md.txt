# Layer 0 Axis: `study_scope`

- Parent: [Layer 0](index.md)
- Current: `study_scope`

`study_scope` describes the broad shape of the study: how many targets are
being forecast and whether the recipe compares multiple method alternatives.
It is a Layer 0 setup axis because it affects routing, sweep interpretation,
Navigator flow, and manifest metadata before any data or model details are
chosen.

The compiler can derive `study_scope` when it is omitted. Navigator and
replication recipes may set it explicitly.

## Values

| Value | Status | Target Shape | Method Shape |
|---|---|---|---|
| `one_target_one_method` | operational | one target | one fixed method path |
| `one_target_compare_methods` | operational | one target | controlled method comparison |
| `multiple_targets_one_method` | operational | multiple targets | one fixed method path |
| `multiple_targets_compare_methods` | operational | multiple targets | controlled method comparison |

All four values currently route through the comparison-sweep owner. A run with
one target and one method is still a 1x1 sweep case for provenance consistency.

## Derivation

When `study_scope` is omitted, the compiler derives it from:

| Input Shape | Derived Value |
|---|---|
| single target, fixed method axes | `one_target_one_method` |
| single target, swept model or feature axes | `one_target_compare_methods` |
| multi target, fixed method axes | `multiple_targets_one_method` |
| multi target, swept model or feature axes | `multiple_targets_compare_methods` |

This derivation should be treated as a defaulting rule. If the recipe sets
`study_scope` explicitly, the compiler checks it against target structure.

## Compatibility Rules

| Rule | Outcome |
|---|---|
| `multiple_targets_*` with single-target task | invalid |
| `one_target_*` with `target_structure=multi_target` | invalid |
| `multiple_targets_*` with `target_structure=multi_target` | valid |
| method-comparison scope with no swept method axis | valid but semantically broad |

Layer 1 owns the actual target list and target structure. Layer 0 only records
the intended study shape.

## YAML

Single target, one method:

```yaml
0_meta:
  fixed_axes:
    study_scope: one_target_one_method
```

Single target, method comparison:

```yaml
0_meta:
  fixed_axes:
    study_scope: one_target_compare_methods
```

Multi-target method comparison:

```yaml
0_meta:
  fixed_axes:
    study_scope: multiple_targets_compare_methods

1_data_task:
  fixed_axes:
    target_structure: multi_target
  leaf_config:
    targets: [INDPRO, CPIAUCSL, UNRATE, PAYEMS, RPI]
```

## When To Choose Each Value

| Research Intent | Use |
|---|---|
| Debug one baseline forecast | `one_target_one_method` |
| Compare models/features for one macro target | `one_target_compare_methods` |
| Run the same specification over several targets | `multiple_targets_one_method` |
| Compare methods over several targets | `multiple_targets_compare_methods` |

## Notes

- `study_scope` is not a model choice.
- `study_scope` is not a target selector; Layer 1 stores target names.
- `study_scope` is not sweepable.
- The public values above are the registry values. Do not use aliases such as
  `multi_output_joint_model` in new recipes.
