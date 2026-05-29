# Reference

Lookup pages for the public Python surface, recipe YAML contract, and generated layer option dictionary. These pages are for exact lookup; tutorials and design narrative live elsewhere.

> **Looking for design rationale?** Use [Architecture](../explanation/architecture/index.md). Reference pages define names, keys, options, and import surfaces; architecture pages explain why the layers are separated and how they interact.

## Recipe API

- [Recipe gallery](gallery.md): runnable examples.
- [Layer contract](layer_contract.md): layer keys, graph node shape, and complete recipe form.
- [Data](data.md): source, target, horizon, and geography choices.
- [Data policies](data_policies.md): missingness, outliers, release lags, and same-period predictors.
- [Defaults](defaults.md): package-level default profiles.
- [Runtime support](runtime_support.md): what executes today.
- [Output](output.md): artifact directory and manifest layout.
- [FRED datasets](fred_datasets.md): FRED-MD, FRED-QD, and FRED-SD reference status.

## Python Surface

- [Public Python API](public_api.md): top-level imports and semantic package map.
- [Standalone functions](standalone_functions/index.md): direct `mf.functions.*` callables.
- [Navigator](navigator/tree_navigator.md): layer-topology navigator details.

## Option Encyclopedia

Generated from the live `LayerImplementationSpec` registry plus the `OPTION_DOCS` registry under `tools/docgen/option_docs/`.

- Layers: 8
- Sub-layers: 38
- Axes (operational + future): 103
- Option values: 366
- OptionDoc entries: 366

Browse indexes:

- [Browse by layer](browse_by_layer.md)
- [Browse by axis](browse_by_axis.md)
- [Browse by option](browse_by_option.md)

## Layer Pages

```{toctree}
:maxdepth: 1

gallery
layer_contract
data
data_policies
defaults
runtime_support
output
fred_datasets
public_api
standalone_functions/index
navigator/tree_navigator
browse_by_layer
browse_by_axis
browse_by_option
l3/index
l3_5/index
l4/index
l4_5/index
l5/index
l6/index
l7/index
l8/index
```
