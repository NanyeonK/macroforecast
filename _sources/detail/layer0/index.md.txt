# 4.0 Layer 0: Study Setup

- Parent: [4. Detail (code): Full](../index.md)
- Current: Layer 0
- Next: [4.1 Layer 1: Data Source, Target y, Predictor x](../layer1/index.md)

Layer 0 defines recipe-level execution policy before data, features, or
models are chosen. It is hidden from the navigator main view by default
and is available through an advanced toggle.

## Layer Role

Layer 0 is a setup layer. It receives no upstream sinks and produces
`l0_meta_v1`, a metadata artifact recorded in every cell manifest. Other
layers read L0 decisions from the manifest/runtime context, not through a
scientific data sink.

## Sub-Layer

Layer 0 has one sub-layer:

| Slot | Name | Gate |
|---|---|---|
| L0.A | Execution policy | always |

L0.B and later slots are not used.

## Axes

Layer 0 has exactly three user-facing axes:

| Axis | Default | Sweepable | Notes |
|---|---|---:|---|
| `failure_policy` | `fail_fast` | no | `continue_on_failure` records failed cells and continues remaining cells. |
| `reproducibility_mode` | `seeded_reproducible` | no | `random_seed` defaults to `42`; `exploratory` must not set a seed. |
| `compute_mode` | `serial` | no | `parallel` requires `leaf_config.parallel_unit` and `leaf_config.n_workers`. |

Rejected values remain invalid: there is no `strict` reproducibility mode
and no `parallel_models` or similar top-level compute mode. Use
`compute_mode=parallel` plus `leaf_config.parallel_unit`.

## Leaf Config

| Key | Rule |
|---|---|
| `random_seed` | Optional int; default `42` when `reproducibility_mode=seeded_reproducible`; forbidden in `exploratory`. |
| `parallel_unit` | Required when `compute_mode=parallel`; one of `models`, `horizons`, `targets`, `oos_dates`. |
| `n_workers` | Required when `compute_mode=parallel`; positive int or `auto`. |
| `gpu_deterministic` | Optional bool; default `false`. |

## Derived Manifest Fields

L0 derives and records:

| Field | Value |
|---|---|
| `study_scope` | Derived from target count and method/sweep shape. |
| `execution_route` | Always `comparison_sweep`; single-cell runs are 1x1 sweep cases. |

## Minimal YAML

```yaml
0_meta:
  fixed_axes: {}
```

This resolves to `fail_fast`, `seeded_reproducible` with seed `42`, and
`serial`.

## Fully Explicit YAML

```yaml
0_meta:
  fixed_axes:
    failure_policy: continue_on_failure
    reproducibility_mode: seeded_reproducible
    compute_mode: parallel
  leaf_config:
    random_seed: 20260101
    parallel_unit: oos_dates
    n_workers: 8
    gpu_deterministic: true
```

```{toctree}
:maxdepth: 1

failure_policy
reproducibility_mode
compute_mode
```
