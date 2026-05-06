# Layer 0: Study Setup

- Parent: [Architecture](../index.md)
- Current: Layer 0
- Next: [Layer 1: Data Source, Target y, Predictor x](../layer1/index.md)

Layer 0 defines the study-level contract before data, transformations,
features, models, evaluation, or export are chosen. It answers four setup
questions:

1. What study shape is this?
2. Should failed cells stop the run or be recorded while the run continues?
3. Should stochastic code use a fixed seed?
4. Should work run serially or in parallel?

Layer 0 is not a scientific modeling layer. It does not create a data panel,
target series, feature matrix, forecast, metric, or artifact selection. It
produces setup metadata that every downstream layer can read from the compiled
manifest.

## Layer Role

Layer 0 receives no upstream sinks and produces `l0_meta_v1`. The artifact is
metadata, not a dataset. It is recorded in the compiled manifest and copied into
runtime provenance so that every result cell can be traced to the same setup
policy.

Downstream layers use Layer 0 through manifest/runtime context:

| Consumer | Reads Layer 0 For |
|---|---|
| Compiler | default derivation, compatibility checks, manifest metadata |
| Sweep planner | parent study shape and failure/compute behavior |
| Execution runner | seed application, continuation policy, parallel routing |
| Navigator | setup defaults and target-cardinality guidance |
| Output layer | provenance fields saved with result artifacts |

Layer 0 should stay small. Dataset choice belongs in Layer 1, representation
choice belongs in Layer 2, feature graph construction belongs in Layer 3, model
choice belongs in Layer 4, and evaluation/export choices belong later.

## Sub-Layer

Layer 0 has one active sub-layer:

| Slot | Name | Gate |
|---|---|---|
| L0.A | Execution and study setup policy | always |

There is no L0.B. If a choice changes data, features, models, metrics, tests,
interpretation, or output files, it belongs in another layer.

## Axes

Layer 0 has four public registry axes plus one registry-internal catalog axis:

| Axis | Default | Sweepable | User-Facing | Purpose |
|---|---|---:|---:|---|
| `study_scope` | derived | no | yes | target cardinality and method-comparison shape |
| `failure_policy` | `fail_fast` | no | yes | behavior when a cell fails |
| `reproducibility_mode` | `seeded_reproducible` | no | yes | seed policy |
| `compute_mode` | `serial` | no | yes | serial or parallel execution |
| `axis_type` | registry catalog | no | no | internal registry taxonomy |

Layer 0 axes are fixed setup choices. Do not sweep them as research treatments.
If a study needs to compare models, horizons, feature blocks, metrics, or output
sets, express those comparisons in the owning downstream layer.

## Axis Values

### `study_scope`

`study_scope` is explicit in Navigator flows and may also be derived by the
compiler from target structure and sweep shape.

| Value | Meaning |
|---|---|
| `one_target_one_method` | one target, one fixed method path |
| `one_target_compare_methods` | one target, controlled method comparison |
| `multiple_targets_one_method` | multiple targets, one fixed method path |
| `multiple_targets_compare_methods` | multiple targets, controlled method comparison |

Rules:

- `multiple_targets_*` requires `target_structure=multi_target`.
- `one_target_*` is incompatible with `target_structure=multi_target`.
- Single-cell runs are still treated as a 1x1 comparison-sweep route.
- The route owner is currently `comparison_sweep` for every public value.

### `failure_policy`

| Value | Meaning |
|---|---|
| `fail_fast` | stop when the first execution cell fails |
| `continue_on_failure` | record failed cells and continue remaining cells |

Use `fail_fast` while developing a recipe. Use `continue_on_failure` for large
sweeps where partial completion is useful and failed cells must remain visible
in manifests.

### `reproducibility_mode`

| Value | Meaning |
|---|---|
| `seeded_reproducible` | set Python/NumPy/torch base seeds; default seed is `42` |
| `exploratory` | do not reset global RNG state; reproducibility is waived |

Seeded reproducibility is a best-effort runtime policy, not a promise that every
GPU/library/version combination is bit-identical. Hardware deterministic flags
are controlled separately by leaf config.

### `compute_mode`

| Value | Meaning |
|---|---|
| `serial` | run one work unit at a time |
| `parallel` | parallelize over a leaf-config work unit |

Parallel work unit is not encoded as a separate public axis value. Use
`compute_mode=parallel` and specify the work unit in `leaf_config.parallel_unit`.

## Leaf Config

Layer 0 accepts leaf config for runtime details that are too specific to be
axis values.

| Key | Applies When | Rule |
|---|---|---|
| `random_seed` | `seeded_reproducible` | optional int; default `42` |
| `random_seed` | `exploratory` | forbidden |
| `gpu_deterministic` | any reproducibility mode | optional bool; default `false` |
| `parallel_unit` | `compute_mode=parallel` | required; one of `models`, `horizons`, `targets`, `oos_dates` |
| `n_workers` | `compute_mode=parallel` | required; positive int or `auto` |

`leaf_config` should not be used to smuggle in scientific choices. If a value
changes what data or model is studied, it belongs in a downstream layer axis or
DAG node.

## Derived Manifest Fields

Layer 0 records setup fields in compiler/runtime provenance.

| Field | Source |
|---|---|
| `study_scope` | explicit L0 axis or compiler derivation |
| `execution_route` | currently `comparison_sweep` |
| `failure_policy_spec` | L0 `failure_policy` |
| `reproducibility_spec` | L0 `reproducibility_mode` plus seed details |
| `compute_mode_spec` | L0 `compute_mode` plus parallel details |

The manifest is the place to inspect resolved defaults. Recipes can omit many
Layer 0 fields, but compiled artifacts should record what was actually used.

## Minimal YAML

```yaml
0_meta:
  fixed_axes: {}
```

This resolves to:

| Field | Resolved Value |
|---|---|
| `failure_policy` | `fail_fast` |
| `reproducibility_mode` | `seeded_reproducible` |
| `random_seed` | `42` |
| `compute_mode` | `serial` |
| `study_scope` | derived from target and sweep shape |

## Explicit Single-Target YAML

```yaml
0_meta:
  fixed_axes:
    study_scope: one_target_compare_methods
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
    compute_mode: serial
  leaf_config:
    random_seed: 20260101
```

## Explicit Multi-Target Parallel YAML

```yaml
0_meta:
  fixed_axes:
    study_scope: multiple_targets_compare_methods
    failure_policy: continue_on_failure
    reproducibility_mode: seeded_reproducible
    compute_mode: parallel
  leaf_config:
    random_seed: 20260101
    parallel_unit: targets
    n_workers: 4
    gpu_deterministic: true
```

## Invalid Patterns

Do not use retired or compatibility-only values in new recipes:

| Invalid Pattern | Use Instead |
|---|---|
| `reproducibility_mode: strict` | `seeded_reproducible` plus `leaf_config.gpu_deterministic` |
| `compute_mode: parallel_by_model` | `compute_mode: parallel` plus `parallel_unit: models` |
| `compute_mode: parallel_by_horizon` | `compute_mode: parallel` plus `parallel_unit: horizons` |
| `compute_mode: parallel_by_target` | `compute_mode: parallel` plus `parallel_unit: targets` |
| `compute_mode: parallel_by_oos_date` | `compute_mode: parallel` plus `parallel_unit: oos_dates` |
| sweeping `failure_policy` | keep it fixed; compare scientific choices in later layers |
| setting `axis_type` in a recipe | omit it; it is registry metadata |

## Related Pages

```{toctree}
:maxdepth: 1

study_scope
failure_policy
reproducibility_mode
compute_mode
axis_type
```

## See encyclopedia

For the full per-axis × per-option catalogue (every value with its OptionDoc summary, when-to-use / when-NOT, references), see [`encyclopedia/l0/`](../../encyclopedia/l0/index.md).
