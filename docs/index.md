# macrocast

> Choose a forecasting path, inspect the constraints, generate a runnable recipe, and run the same path from CLI, notebook, or code.

macrocast is a research-oriented forecasting package for economists and macro-finance researchers. The documentation is organized around decisions first and API signatures last.

## 0. Getting Started

Start here for installation and the three supported ways to use the package:

- choose a path in the Navigator, export YAML, then run it;
- write simple Python code with `forecast()` or `Experiment`;
- write full layer-path code or YAML when you need exact control.

[Open Getting Started](getting_started/index.md)

## 1. Philosophy

The package is built around explicit defaults, layer contracts, reproducible artifacts, and custom researcher methods that run beside built-ins.

[Read Philosophy](philosophy.md)

## 2. Navigator

Use the Navigator when you need a constraint-aware decision tree: selectable options, disabled options, disabled reasons, canonical path effects, replication recipes, and YAML generation.

- [Open Navigator App](navigator_app/index.html)
- [Navigator Docs](navigator/index.md)

## 3. Simple (code)

Use Simple docs when you want code-first forecasting without reading every layer contract.

[Open Simple Docs](simple/index.md)

## 4. Detail (code)

Use Detail docs when you need the full layer grammar, exact contracts, YAML shape, runtime artifacts, or custom method hooks.

[Open Detail Docs](detail/index.md)

## Reference

API Reference is for signatures after you know which path you want.

[Open API Reference](api/index.md)

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
philosophy
navigator/index
simple/index
detail/index
user_guide/index
api/index
CONVENTIONS
```
