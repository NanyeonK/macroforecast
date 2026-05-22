# API Reference

Complete API documentation for macroforecast's public Python surface.

## Standalone Functions

Call any layer operation directly as `mf.functions.<name>(...)` without
a YAML recipe. Useful for integrating individual operations into existing
pipelines or for interactive exploration.

[Standalone functions](standalone_functions/index.md)

## Navigator

Layer and pipeline topology navigator. Use this to explore which axes,
options, and ops are available at each layer before writing a recipe.

[Navigator](navigator/index.md)

```{toctree}
:maxdepth: 2

standalone_functions/index
navigator/index
```
