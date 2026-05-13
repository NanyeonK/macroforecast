# Registry Catalog: `axis_type`

- Parent: [Layer 0](index.md)
- Current: `axis_type`

`axis_type` is registry metadata. It is not a user-facing Layer 0 recipe axis
and should not be set in new recipes.

The catalog exists so registry entries can describe how an axis behaves in the
larger design system.

## Values

| Value | Meaning |
|---|---|
| `fixed` | axis is normally fixed within one study path |
| `sweep` | axis is normally swept across multiple values |
| `nested_sweep` | axis participates in nested sweep designs |
| `conditional` | axis is activated only after another choice |
| `derived` | axis is derived from other recipe state |

## How This Differs From Layer 0 User Axes

| Concept | User Sets It In Recipe? | Purpose |
|---|---:|---|
| `study_scope` | yes or runtime-derived | study shape |
| `failure_policy` | yes | failure behavior |
| `reproducibility_mode` | yes | seed policy |
| `compute_mode` | yes | execution scheduling |
| `axis_type` | no | registry taxonomy |

## Invalid YAML

Do not write:

```yaml
0_meta:
  fixed_axes:
    axis_type: fixed
```

`axis_type` belongs to registry definitions, not recipe instances.

## Notes

- List layers use `fixed_axes` and, where supported, sweep structures.
- Graph layers use explicit `nodes` and `sinks`.
- L0 public axes are fixed setup choices and are not sweepable.
