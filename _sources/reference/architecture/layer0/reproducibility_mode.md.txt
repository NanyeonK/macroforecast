# Layer 0 Axis: `reproducibility_mode`

- Parent: [Layer 0](index.md)
- Current: `reproducibility_mode`

`reproducibility_mode` controls seed policy for the run. It determines whether
the execution layer resets stochastic state before fitting and forecasting.

## Values

| Value | Status | Meaning |
|---|---|---|
| `seeded_reproducible` | operational, default | apply a fixed base seed |
| `exploratory` | operational | do not reset global RNG state |

This axis is not sweepable. Reproducibility policy is part of study setup, not a
model alternative.

## Leaf Config

| Key | Applies When | Rule |
|---|---|---|
| `random_seed` | `seeded_reproducible` | optional int; default `42` |
| `random_seed` | `exploratory` | forbidden |
| `gpu_deterministic` | any mode | optional bool; default `false` |

`gpu_deterministic` is separate because hardware/library deterministic behavior
is not the same as seed policy.

## Runtime Semantics

| Mode | Runtime Behavior |
|---|---|
| `seeded_reproducible` | sets Python, NumPy, and torch seeds where available |
| `exploratory` | leaves existing global RNG state alone and records that reproducibility was waived |

`seeded_reproducible` is intended for reproducible empirical work. It does not
guarantee bit-identical results across different hardware, BLAS libraries,
torch/cuda versions, or nondeterministic algorithms unless the relevant runtime
stack also supports deterministic execution.

## YAML

Default seeded run:

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: seeded_reproducible
```

Seeded run with explicit seed:

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 20260101
```

Exploratory run:

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: exploratory
```

GPU deterministic request:

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 42
    gpu_deterministic: true
```

## Invalid Patterns

| Invalid Pattern | Use Instead |
|---|---|
| `reproducibility_mode: strict` | `seeded_reproducible` plus deterministic leaf config where supported |
| `reproducibility_mode: strict_reproducible` in new recipes | `seeded_reproducible` |
| `exploratory` with `random_seed` | remove `random_seed` or use `seeded_reproducible` |
| sweeping seed policy | keep fixed; sweep scientific axes only |

## Notes

- Default seed is `42`.
- Set a project-specific seed when results must be replayed exactly within the
  same environment.
- Use `exploratory` only when reproducibility is intentionally waived.
