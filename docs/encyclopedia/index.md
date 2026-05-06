# Encyclopedia

Encyclopedia-style browse for every macroforecast schema choice. Each layer, sub-layer, axis, and option value has its own page or anchor; the tree below is generated from the live `LayerImplementationSpec` registry plus the `OPTION_DOCS` documentation registry under `macroforecast/scaffold/option_docs/`.

## Counts

- Layers: 13
- Sub-layers: 62
- Axes (operational + future): 171
- Option values: 589
- OptionDoc entries (full prose): 586

## Browse

- [Browse by layer](browse_by_layer.md)  --  L0 to L8 + diagnostics, table form.
- [Browse by axis](browse_by_axis.md)  --  every axis A-Z.
- [Browse by option](browse_by_option.md)  --  every option *value* A-Z (e.g. `ridge`, `pca`, `ar_p`).
- [Public Python API](public_api.md)  --  curated `macroforecast.run` / `macroforecast.replicate` surface.

## Layer pages

```{toctree}
:maxdepth: 1

l0/index
l1/index
l1_5/index
l2/index
l2_5/index
l3/index
l3_5/index
l4/index
l4_5/index
l5/index
l6/index
l7/index
l8/index
```

```{toctree}
:hidden:
:maxdepth: 1

browse_by_layer
browse_by_axis
browse_by_option
public_api
```
