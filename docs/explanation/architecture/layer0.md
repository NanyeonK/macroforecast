# Layer L0: Meta / Study Setup

- Parent: [Architecture](index.md)
- Current: L0 / `macroforecast.meta`
- Next: [L1 - Data Source, Target y, Predictor x](layer1.md)

L0 defines study-wide execution policy before data, transformations, features, models, evaluation, tests, interpretation, or export are chosen. It is the meta layer: it does not create a data panel, feature matrix, forecast, metric, or artifact. It produces setup metadata that the runtime records in the manifest and applies to every execution cell.

The same contract is available through YAML and through Python callables:

```python
import macroforecast as mf

l0_block = mf.l0(
    failure_policy="fail_fast",
    reproducibility_policy="seeded_reproducible",
    compute_policy="serial",
    random_seed=42,
)
```

```yaml
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_policy: seeded_reproducible
    compute_policy: serial
  leaf_config:
    random_seed: 42
```

Both forms compile to the same `0_meta` block. A later YAML wrapper can therefore load all settings at once without defining a separate configuration semantics.

## Layer Role

L0 receives no upstream sinks and produces `l0_meta_v1`. The artifact is metadata, not data. It is recorded in the compiled manifest and copied into runtime provenance so every result cell can be traced to the same setup policy.

Downstream layers use L0 through manifest/runtime context:

| Consumer | Reads L0 For |
|---|---|
| Runtime | default derivation, compatibility checks, manifest metadata |
| Sweep planner | failure and compute behavior |
| Execution runner | seed application, continuation policy, parallel routing |
| Output layer | provenance fields saved with result artifacts |

Scientific choices belong downstream. Dataset choice belongs in L1, representation choice in L2, feature construction in L3, model choice in L4, and evaluation/export choices later.

## Public Axes

L0 has three public fixed axes:

| Axis | Default | Sweepable | Purpose |
|---|---|---:|---|
| `failure_policy` | `fail_fast` | no | behavior when an execution cell fails |
| `reproducibility_policy` | `seeded_reproducible` | no | seed policy |
| `compute_policy` | `serial` | no | serial or parallel execution |

`study_scope` is no longer a user-set L0 axis. The runtime derives it from target shape and comparison shape and records it as provenance. `axis_type` is registry metadata and does not appear in recipes.

## Axis Values

### `failure_policy`

| Value | Meaning |
|---|---|
| `fail_fast` | stop when the first execution cell fails |
| `continue_on_failure` | record failed cells and continue remaining cells |

Use `fail_fast` while developing a recipe. Use `continue_on_failure` for large sweeps where partial completion is useful and failed cells must remain visible in manifests.

### `reproducibility_policy`

| Value | Meaning |
|---|---|
| `seeded_reproducible` | set the base seed; default seed is `42` |
| `exploratory` | do not reset global RNG state; reproducibility is waived |

Seeded reproducibility is a best-effort runtime policy, not a promise that every hardware/library/version combination is bit-identical. Hardware deterministic flags are controlled by leaf config.

### `compute_policy`

| Value | Meaning |
|---|---|
| `serial` | run one work unit at a time |
| `parallel` | parallelize over `leaf_config.parallel_unit` |

Parallel work unit is not a fixed axis. Use `compute_policy: parallel` and specify the unit in `leaf_config.parallel_unit`.

## Leaf Config

L0 accepts leaf config for runtime details that are too specific to be axis values.

| Key | Applies When | Rule |
|---|---|---|
| `random_seed` | `seeded_reproducible` | optional int; default `42` |
| `random_seed` | `exploratory` | forbidden |
| `gpu_deterministic` | any reproducibility mode | optional bool; default `false` |
| `parallel_unit` | `compute_policy=parallel` | required; one of `cells`, `models`, `horizons`, `targets`, `oos_dates` |
| `n_workers` | `compute_policy=parallel` | required; positive int or `auto` |

`leaf_config` should not smuggle in scientific choices. If a value changes what data or model is studied, it belongs in a downstream layer axis or recipe node.

## Derived Manifest Fields

L0 records setup fields in runtime provenance.

| Field | Source |
|---|---|
| `study_scope` | derived from target and comparison shape |
| `execution_route` | currently `comparison_sweep` |
| `failure_policy_spec` | L0 `failure_policy` |
| `reproducibility_spec` | L0 `reproducibility_policy` plus seed details |
| `compute_mode_spec` | L0 `compute_policy` plus parallel details |

The manifest is the place to inspect resolved defaults. Recipes can omit many L0 fields, but compiled artifacts record what was actually used.

## Minimal YAML

```yaml
0_meta:
  fixed_axes: {}
```

This resolves to:

| Field | Resolved Value |
|---|---|
| `failure_policy` | `fail_fast` |
| `reproducibility_policy` | `seeded_reproducible` |
| `random_seed` | `42` |
| `compute_policy` | `serial` |
| `study_scope` | derived from target and comparison shape |

## Explicit Single-Target YAML

```yaml
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_policy: seeded_reproducible
    compute_policy: serial
  leaf_config:
    random_seed: 20260101
```

## Explicit Parallel YAML

```yaml
0_meta:
  fixed_axes:
    failure_policy: continue_on_failure
    reproducibility_policy: seeded_reproducible
    compute_policy: parallel
  leaf_config:
    random_seed: 20260101
    parallel_unit: cells
    n_workers: 4
    gpu_deterministic: true
```

## Invalid Patterns

Do not use retired or compatibility-only values in new recipes:

| Invalid Pattern | Use Instead |
|---|---|
| `study_scope` under `0_meta.fixed_axes` | omit it; runtime derives and records it |
| `reproducibility_policy: strict` | `seeded_reproducible` plus `leaf_config.gpu_deterministic` |
| `compute_policy: parallel_by_model` | `compute_policy: parallel` plus `leaf_config.parallel_unit: models` |
| `compute_policy: parallel_by_horizon` | `compute_policy: parallel` plus `leaf_config.parallel_unit: horizons` |
| `compute_policy: parallel_by_target` | `compute_policy: parallel` plus `leaf_config.parallel_unit: targets` |
| `compute_policy: parallel_by_oos_date` | `compute_policy: parallel` plus `leaf_config.parallel_unit: oos_dates` |
| `parallel_unit` under `fixed_axes` | move it to `leaf_config.parallel_unit` |
| sweeping any L0 axis | keep L0 fixed; compare scientific choices in later layers |
| setting `axis_type` in a recipe | omit it; it is registry metadata |

## Related Pages

```{toctree}
:maxdepth: 1

layer0_derived_study_scope
layer0_failure_policy
layer0_reproducibility_policy
layer0_compute_policy
```

## Reference

For the full per-axis and per-option catalogue, see [`generated/l0`](../../reference/generated/l0/index.md).
