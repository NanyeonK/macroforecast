# For recipe authors

You want to author recipes by hand, sweep multiple axes, register custom
models / preprocessors / target transformers, or use the YAML grammar deeply.

## If you want to ...

| Goal | Page |
|---|---|
| Read the layer-contract design that recipes encode | [Layer contract design](design.md) |
| Look up the data-task axes (source / target / predictor / horizon / ...) | [Data axes](data/index.md) |
| Read the consolidated Layer 1 data-handling policies (axis-by-axis) | [Data policies](data_policies.md) |
| Register a custom model / preprocessor / feature block / combiner / target transformer | [Custom hooks](custom_hooks.md) |
| Author a custom target transformer (fit-window, inverse, leakage rules) | [Target transformer](target_transformer.md) |
| See default profile shape and override patterns | [Default profiles](default_profiles.md) |

## Working examples

Bundled YAML recipes covering each layer live under `examples/recipes/` in the
repo. The `archive_v0/` subdirectory inside that tree holds older recipes that
are kept for replication only — start from the unprefixed files (e.g.
`l4_minimal_ridge.yaml`, `l3_mccracken_ng_baseline.yaml`).

For end-to-end replication walkthroughs see [Replications](../replications/index.md).

## Bridging audiences

- For the **why** of each layer (boundary contracts, sink names, manifest
  fields) read the canonical [architecture pages](../architecture/index.md).
- For the **public API** the recipe ultimately drives, see
  [`reference/public_api.md`](../reference/public_api.md).

```{toctree}
:hidden:
:maxdepth: 1

design
data/index
data_policies
custom_hooks
target_transformer
default_profiles
```
