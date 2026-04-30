# macrocast

macrocast is a research-oriented forecasting package for economists and macro-finance researchers. The docs are organized around decisions first and API signatures last.

## Choose Your Entry Point

### 0. Getting Started

Install macrocast and choose one of three usage paths: Navigator/YAML/CLI, Simple Python code, or full Detail code/YAML.

[Open Getting Started](getting_started/index.md)

### 1. Philosophy

Read this when you want the package rules: explicit defaults, layer contracts, reproducible artifacts, and custom researcher methods beside built-ins.

[Read Philosophy](philosophy.md)

### 2. Navigator

Use the Navigator when you need a constraint-aware decision tree. It shows selectable options, disabled options, disabled reasons, canonical path effects, and YAML generation.

- [Open Navigator App](navigator_app/index.html)
- [Navigator Docs](navigator/index.md)

### 3. Simple (code)

Use Simple docs when you want forecasting code without reading every layer contract first.

[Open Simple Docs](simple/index.md)

### 4. Detail (code)

Use Detail docs when you need exact layer control, YAML shape, runtime artifacts, or custom method hooks.

[Open Detail Docs](detail/index.md)

### 5. FRED-Dataset

Use FRED-Dataset docs when you need the current FRED-MD, FRED-QD, or FRED-SD
column dictionary before writing target y, predictor x, or custom-source
recipes.

[Open FRED-Dataset Docs](fred_dataset/index.md)

### Foundation Core

Use Foundation Core docs when you need the Phase 0 DAG schema, registry
migration contract, or next-generation layer implementation path.

[Open Foundation Core](foundation_core.md)

## Reference

Use API Reference after you know which path you want.

[Open API Reference](api/index.md)

```{toctree}
:hidden:
:maxdepth: 1

getting_started/index
philosophy
navigator/index
simple/index
detail/index
fred_dataset/index
foundation_core
api/index
CONVENTIONS
```
