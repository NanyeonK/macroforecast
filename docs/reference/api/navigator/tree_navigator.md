# Tree Navigator

The Navigator has two surfaces.

## 1. Canonical Layer/DAG Map

This is the primary UI. It mirrors the registered runtime architecture:

- L0-L8 main flow;
- L1.5-L4.5 diagnostic side branches;
- graph mode for L3, L4, and L7;
- list mode for setup, construction list layers, diagnostics, evaluation, tests, and output;
- sink handoffs between registered layers.

Click a layer card to inspect:

- role and category;
- expected upstream sinks;
- produced sinks;
- sub-layers;
- layer globals;
- axes or output controls.

Graph/DAG layers do not have one flat axis list. Their decisions live in YAML `nodes`, `params`, `inputs`, and `sinks`. The UI now exposes clickable DAG items and writes a runnable YAML template from the selected layer axes plus selected DAG items.

## Canonical Sub-Layers

The layer map should expose every main and diagnostic sub-layer below. If a sub-layer is absent in the UI, treat that as a UI bug rather than a design change.

| Layer | UI mode | Sub-layers |
|---|---|---|
| L0 | list | L0.A Execution policy |
| L1 | list | L1.A Source selection; L1.B Target definition; L1.C Predictor universe; L1.D Geography scope; L1.E Sample window; L1.F Horizon set; L1.G Regime definition |
| L2 | list | L2.A FRED-SD frequency alignment; L2.B Transform; L2.C Outlier handling; L2.D Imputation; L2.E Frame edge |
| L3 | graph | L3.A Target construction; L3.B Feature pipelines; L3.C Pipeline combine; L3.D Feature selection |
| L4 | graph | L4.A Model selection; L4.B Forecast strategy; L4.C Training window; L4.D Tuning |
| L5 | list | L5.A Metrics; L5.B Benchmark; L5.C Aggregation; L5.D Slicing and decomposition; L5.E Ranking |
| L6 | list | L6 globals; L6_A_equal_predictive; L6_B_nested; L6_C_cpa; L6_D_multiple_model; L6_E_density_interval; L6_F_direction; L6_G_residual |
| L7 | graph | L7.A Importance DAG; L7.B Output shape |
| L8 | list | L8_A_export_format; L8_B_saved_objects; L8_C_provenance; L8_D_artifact_granularity |
| L1.5 | list | L1.5.A Sample coverage; L1.5.B Univariate summary; L1.5.C Stationarity; L1.5.D Missing and outlier; L1.5.E Correlation; L1.5.Z Export |
| L2.5 | list | L2.5.A Comparison; L2.5.B Distribution shift; L2.5.C Correlation shift; L2.5.D Cleaning summary; L2.5.Z Export |
| L3.5 | list | L3.5.A Comparison; L3.5.B Factor inspection; L3.5.C Feature correlation; L3.5.D Lag inspection; L3.5.E Selection; L3.5.Z Export |
| L4.5 | list | L4.5.A Fit; L4.5.B Scale; L4.5.C Window stability; L4.5.D Tuning; L4.5.E Ensemble; L4.5.Z Export |

## YAML Generation Contract

The YAML preview is generated from the canonical workbench, not the legacy compatibility explorer.

- List layers write `fixed_axes` from selected sub-layer axes.
- Layer-global controls, including diagnostic `enabled`, are selectable and written into YAML.
- Graph layers write template DAGs with `nodes`, `inputs`, `params`, and `sinks`.
- Clicked DAG items are stored in `leaf_config.navigator_selected_dag_items` so the YAML records the graph design choices that shaped the template.
- Multi-select axes write YAML lists.

The generated YAML is intended as a valid starting recipe. Advanced graph editing can still refine node IDs, custom leaf config, and model-specific parameters after download.

## 2. Compatibility Axis Explorer

The lower explorer is retained for option-level compatibility checks. It shows enabled/disabled choices, disabled reasons, path effects, and YAML preview. It still contains some historical axis-group labels because it is a compatibility surface, not the canonical architecture view.

Use it after choosing a canonical layer when you need to know why a specific option is blocked.

## Reading The UI

| UI element | Meaning |
|---|---|
| Main flow cards | Canonical L0-L8 execution order. |
| Diagnostic side branches | Default-off hooks that consume construction sinks and produce diagnostics. |
| `DAG` badge | Layer is configured as a graph of source/step/sink nodes. |
| `List` badge | Layer is configured by ordered sub-layers and axes. |
| Sink handoffs | Registry-derived edges from `produces` to `expected_inputs`. |
| Runtime support | Current runtime support class for selected options. |

## Boundary Rule

The layer map is authoritative for architecture. The compatibility explorer is useful for individual option constraints, but old labels in that explorer should not override the canonical layer IDs.
