# Layer 0 Axis: `reproducibility_mode`

- Parent: [Layer 0](index.md)
- Current: `reproducibility_mode`

`reproducibility_mode` controls seed policy for the run.

## Values

| Value | Status | Meaning |
|---|---|---|
| `seeded_reproducible` | operational, default | use a fixed seed |
| `exploratory` | operational | do not fix stochastic seeds |

This axis is not sweepable.

## Leaf Config

| Key | Rule |
|---|---|
| `random_seed` | required by explicit `seeded_reproducible` recipes when the caller wants a non-default seed; default is `42` |
| `gpu_deterministic` | optional bool; separate from this axis |

`exploratory` rejects `leaf_config.random_seed`.

## YAML

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 42
```

```yaml
0_meta:
  fixed_axes:
    reproducibility_mode: exploratory
```

## Notes

- `strict` is not a current value. GPU determinism is a leaf-config decision.
- The current contract keeps seed policy on the axis and hardware determinism in leaf config.
