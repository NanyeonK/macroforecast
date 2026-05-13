# Navigator Docs

The Navigator is the visual front door for the canonical macroforecast layer system:

```text
L0 -> L1 -> L2 -> L3(DAG) -> L4(DAG) -> L5 -> L6 -> L7(DAG) -> L8
        |      |      |       |
       L1.5   L2.5   L3.5    L4.5 diagnostics
```

The layer system defines how a recipe flows from study setup through output.
Each layer has a well-defined role, a schema contract, and a stage label
(`STAGE_BY_LAYER` in `macroforecast.core.stages`) used for color coding and
future Kedro adapter tagging.

## Canonical Layer Roles

| Layer | Role |
|---|---|
| L0 | Study setup, reproducibility, failure and compute policies. |
| L1 | Data definition, target/predictor frame, regime metadata. |
| L2 | Cleaning and preprocessing into the clean panel. |
| L3 | Feature engineering DAG, final features, feature metadata. |
| L4 | Forecast/model DAG, forecasts, model artifacts, training metadata. |
| L5 | Evaluation metrics, aggregation, slicing, ranking. |
| L6 | Statistical tests over L4/L5 artifacts. |
| L7 | Interpretation and importance DAG. |
| L8 | Output, provenance, manifest, file artifacts. |
| L1.5–L4.5 | Diagnostic hooks attached to construction sinks (default-off). |

## Layer Flow

```text
[L0 meta] → [L1 data] → [L2 clean] → [L3 features DAG] → [L4 forecasts DAG]
                                            ↓                      ↓
                                         L3.5 diag             L4.5 diag
[L4] → [L5 evaluation] → [L6 tests] → [L7 interpretation DAG] → [L8 artifacts]
```

Compatibility checks and YAML previews treat the layer map as the source of
truth for package architecture.

## Authoring recipes with `macroforecast wizard`

Recipe authoring and layer visualization are provided by the Solara-based
web wizard (added in v0.9.1, post-v0.9.0):

```bash
pip install 'macroforecast[wizard]'
macroforecast wizard --port 8765
```

Open `http://localhost:8765` to see a 3-pane editor with:

- left rail: layer navigation (color coded by `STAGE_BY_LAYER` stage)
- center workspace: layer form (L0 currently; L1/L2/L5/L6 in P2b/c)
- right pane: live YAML preview

See [the wizard module README](../../macroforecast/wizard/) for current
phase status (P2a MVP) and upcoming features (P3 DAG editor for L3/L4/L7).

## Visualization with Kedro-viz (future)

`macroforecast.adapters.kedro` (Phase P1, planned post-P2c) will expose
recipe → `kedro.Pipeline` conversion. Run `kedro viz` to get layer-color
band visualization. `STAGE_BY_LAYER` serves as the Kedro `layer` tag
source.

## Recommended Flow

1. Start with the layer map and select the layer you are designing.
2. For L3/L4/L7, think in DAG nodes: sources, steps, combines, and sinks.
3. For list layers, inspect sub-layers and axes in the layer detail panel.
4. Use the compatibility axis explorer only when you need option-level disabled reasons.
5. Download or write YAML and run `macroforecast-navigate resolve` before execution.
6. Execute with `macroforecast-navigate run` or the core runtime API.

## Pages

| Page | Purpose |
|---|---|
| [Tree Navigator](tree_navigator.md) | Explain the canonical layer map and the remaining compatibility axis explorer. |
| [Path Resolver](path_resolver.md) | Compile YAML and show execution status, warnings, blocked reasons, and capability matrix. |
| [Compatibility Engine](compatibility_engine.md) | Explain constraint rules such as model/feature compatibility, forecast-object metrics, and importance-method restrictions. |
| [Replication Library](replication_library.md) | Start from known paper-style routes, inspect exact paths, generate YAML, and understand deviations. |
| [YAML and Execution](yaml_execution.md) | Save a selected path as YAML, run it from CLI, and reproduce the same route in a notebook. |

```{toctree}
:maxdepth: 1

tree_navigator
path_resolver
compatibility_engine
replication_library
yaml_execution
```
