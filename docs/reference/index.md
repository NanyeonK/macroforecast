# Reference

Lookup pages for the public Python surface and generated option dictionary.
These pages are for exact lookup; tutorials and design narrative live elsewhere.

> **Looking for design rationale?** Use [Architecture](../explanation/architecture/index.md). Reference pages define names, keys, options, and import surfaces; architecture pages explain why the packages are separated and how they interact.

## Recipe API
- [Meta](meta.md): package-wide execution settings.
- [Data](data.md): canonical panels, metadata, loaders, and run-level data specs.
- [Preprocessing](preprocessing.md): frequency alignment, transforms, outlier handling, imputation, and frame-edge rules.
- [Data policies](data_policies.md): missingness, outliers, release lags, and same-period predictors.
- [Defaults](defaults.md): package-level default profiles.
- [Runtime support](runtime_support.md): what executes today.
- [Output](output.md): artifact directory and manifest layout.
- [FRED datasets](fred_datasets.md): FRED-MD, FRED-QD, and FRED-SD reference status.

## Python Surface

- [Public Python API](public_api.md): top-level imports and semantic package map.
- [Standalone functions](standalone_functions/index.md): direct `mf.functions.*` callables.
- [Navigator](navigator/tree_navigator.md): generated topology navigator details.

## Generated Encyclopedia

Generated from the live implementation registry plus the `OPTION_DOCS` registry
under `tools/docgen/option_docs/`. These pages are useful for exact option
lookup, but they are not the primary user narrative.

- Numbered groups: 13
- Subgroups: 62
- Axes (operational + future): 164
- Option values: 600
- OptionDoc entries: 600

- [Generated encyclopedia](generated/index.md)
- [Browse by group](generated/browse_by_layer.md)
- [Browse by axis](generated/browse_by_axis.md)
- [Browse by option](generated/browse_by_option.md)

## Pages

```{toctree}
:maxdepth: 1

meta
data
preprocessing
data_policies
defaults
runtime_support
output
fred_datasets
public_api
standalone_functions/index
navigator/tree_navigator
generated/index
```
