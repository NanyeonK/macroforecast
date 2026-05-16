# Encyclopedia

Encyclopedia-style browse for every macroforecast schema choice. Each layer, sub-layer, axis, and option value has its own page or anchor; the tree below is generated from the live `LayerImplementationSpec` registry plus the `OPTION_DOCS` documentation registry under `macroforecast/scaffold/option_docs/`.

> **Looking for the design narrative instead?** Use [Architecture](../architecture/index.md) -- that's where the prose "why is L2 separated from L3" / "how does L7 read L4 sinks" / cross-layer reference explanations live. Encyclopedia pages here are **auto-generated lookup** for individual option values (description, when to use, when NOT, references, related options); Architecture pages there are **hand-written narrative** for the design contracts. Both are sourced from the same `LayerImplementationSpec` registry -- encyclopedia is the machine-locked option dictionary, architecture is the human-edited design guide.

## Counts

- Layers: 13
- Sub-layers: 62
- Axes (operational + future): 172
- Option values: 631
- OptionDoc entries (full prose): 612

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
