# For recipe authors

You want to author recipes by hand, sweep multiple axes, register custom
models / preprocessors / target transformers, or use the YAML grammar deeply.

## If you want to ...

| Goal | Page |
|---|---|
| Read the layer-contract design that recipes encode | [Recipe layer contract](../recipe_api/layer_contract.md) |
| Look up the data-task axes (source / target / predictor / horizon / ...) | [Data layer](../recipe_api/data.md) |
| Read the consolidated Layer 1 data-handling policies (axis-by-axis) | [Data policies](../recipe_api/data_policies.md) |
| Register a custom model / preprocessor / feature block / combiner / target transformer | [Custom hooks](custom_hooks.md) |
| Quick reference for all three register_* APIs in one page | [Custom function quickstart](custom_function_quickstart.md) |
| Author a custom target transformer (fit-window, inverse, leakage rules) | [Target transformer](target_transformer.md) |
| Run L1 / L2 / L3 in isolation to debug a recipe or iterate on a layer | [Partial layer execution](partial_layer_execution.md) |
| See package-level recipe defaults | [Recipe defaults](../recipe_api/defaults.md) |

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
  [`encyclopedia/public_api.md`](../encyclopedia/public_api.md).
- To browse every recipe axis × option (with full OptionDoc prose)
  see the [encyclopedia](../encyclopedia/index.md).

```{toctree}
:hidden:
:maxdepth: 1

custom_hooks
custom_function_quickstart
target_transformer
partial_layer_execution
```
