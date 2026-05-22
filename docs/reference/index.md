# Reference

Complete, accurate descriptions of everything macroforecast exposes.
These pages are for look-up, not for learning.

## Architecture

The 12-layer design narrative — why each layer exists, its boundary
contracts, and cross-layer references.

[Architecture](architecture/index.md)

## Encyclopedia

Auto-generated look-up for every recipe axis and option value. One page
per option: definition, when to use, when not to use, references.

[Encyclopedia](encyclopedia/index.md)

## API: standalone functions

Call any layer operation directly as `mf.functions.<name>(...)` without
a YAML recipe.

[Standalone functions](api/standalone_functions/index.md)

## API: navigator

Layer and pipeline topology navigator.

[Navigator](api/navigator/index.md)

## Recipe schema

Full recipe grammar: data sources, defaults, layer contracts, runtime
support matrix, and the recipe gallery.

[Recipe schema](recipe_schema/index.md)

```{toctree}
:maxdepth: 2

architecture/index
api/standalone_functions/index
api/navigator/index
recipe_schema/index
```

```{toctree}
:hidden:
:maxdepth: 1

encyclopedia/index
```
