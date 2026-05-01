# Navigator Docs

The Navigator is the visual front door for the canonical macrocast layer system:

```text
L0 -> L1 -> L2 -> L3(DAG) -> L4(DAG) -> L5 -> L6 -> L7(DAG) -> L8
        |      |      |       |
       L1.5   L2.5   L3.5    L4.5 diagnostics
```

Open the interactive app:

```{raw} html
<p><a class="reference external" href="../navigator_app/index.html">Open Navigator App</a></p>
```

## What The App Shows First

The first screen is no longer an axis list. It starts with the registered layer topology:

- main layer flow from setup through output;
- diagnostic hooks L1.5-L4.5 as default-off side branches;
- `list` versus `graph` UI mode per layer;
- upstream sink inputs and produced sinks;
- sub-layer and axis counts from the runtime registry;
- runtime support labels for selected options.

The older compatibility axis explorer remains below the canonical layer map. Use it for branch-level compatibility checks and YAML preview, but treat the layer map as the source of truth for package architecture.

## Canonical Layer Roles

| Layer | UI mode | Role |
|---|---:|---|
| L0 | list | Study setup, reproducibility, failure and compute policies. |
| L1 | list | Data definition, target/predictor frame, regime metadata. |
| L2 | list | Cleaning and preprocessing into the clean panel. |
| L3 | graph | Feature engineering DAG, final features, feature metadata. |
| L4 | graph | Forecast/model DAG, forecasts, model artifacts, training metadata. |
| L5 | list | Evaluation metrics, aggregation, slicing, ranking. |
| L6 | list | Statistical tests over L4/L5 artifacts. |
| L7 | graph | Interpretation and importance DAG. |
| L8 | list | Output, provenance, manifest, file artifacts. |
| L1.5-L4.5 | list | Diagnostic hooks attached to construction sinks. |

## Pages

| Page | Purpose |
|---|---|
| [Tree Navigator](tree_navigator.md) | Explain the canonical layer map and the remaining compatibility axis explorer. |
| [Path Resolver](path_resolver.md) | Compile YAML and show execution status, warnings, blocked reasons, and capability matrix. |
| [Compatibility Engine](compatibility_engine.md) | Explain constraint rules such as model/feature compatibility, forecast-object metrics, and importance-method restrictions. |
| [Replication Library](replication_library.md) | Start from known paper-style routes, inspect exact paths, generate YAML, and understand deviations. |
| [YAML and Execution](yaml_execution.md) | Save a selected path as YAML, run it from CLI, and reproduce the same route in a notebook. |

## Recommended Flow

1. Start with the layer map and select the layer you are designing.
2. For L3/L4/L7, think in DAG nodes: sources, steps, combines, and sinks.
3. For list layers, inspect sub-layers and axes in the layer detail panel.
4. Use the compatibility axis explorer only when you need option-level disabled reasons.
5. Download or write YAML and run `macrocast-navigate resolve` before execution.
6. Execute with `macrocast-navigate run` or the core runtime API.

```{toctree}
:maxdepth: 1

tree_navigator
path_resolver
compatibility_engine
replication_library
yaml_execution
```
