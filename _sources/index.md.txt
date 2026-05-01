# macrocast

macrocast is a research-oriented macro forecasting package organized around explicit layer contracts and reproducible artifacts.

## Canonical Architecture

```text
L0 -> L1 -> L2 -> L3(DAG) -> L4(DAG) -> L5 -> L6 -> L7(DAG) -> L8
        |      |      |       |
       L1.5   L2.5   L3.5    L4.5 diagnostics
```

- `list` layers expose ordered sub-layers and axes.
- `graph` layers expose DAG nodes, inputs, steps, and sinks.
- diagnostic `.5` layers are default-off side branches.
- L8 exports the reproducible output directory and manifest.

## Entry Points

### Getting Started

Install macrocast and choose between the Navigator, YAML runtime, or simple Python facade.

[Open Getting Started](getting_started/index.md)

### Navigator Docs

Use the Navigator docs when you need to see the canonical layer/DAG map, diagnostic hooks, runtime support status, and compatibility rules.

- [Open Navigator App](navigator_app/index.html)
- [Navigator Docs](navigator/index.md)

### Runtime Support

Read this before relying on advanced layers in a paper workflow. It separates current runtime execution from schema-only surfaces.

[Open Runtime Support Matrix](getting_started/runtime_support.md)

### Simple Python API

Use Simple docs when you want forecasting code without reading every layer contract first.

[Open Simple Docs](simple/index.md)

### Detail Docs

Use Detail docs when you need exact layer control, YAML shape, runtime artifacts, or custom method hooks.

[Open Detail Docs](detail/index.md)

### FRED Dataset Docs

Use FRED docs when you need FRED-MD, FRED-QD, or FRED-SD column references before writing target or predictor recipes.

[Open FRED-Dataset Docs](fred_dataset/index.md)

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
