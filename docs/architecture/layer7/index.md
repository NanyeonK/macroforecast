# Layer 7: Interpretation / Importance

- Parent: [Detail: Layer Contracts](../index.md)
- Previous: [Layer 6](../layer6/index.md)
- Current: Layer 7
- Next: [Layer 8](../layer8/index.md)

Layer 7 explains model forecasts through importance, attribution, marginal effects, lineage aggregation, and transformation attribution. It is default off and uses graph-form YAML.

## Contract

Inputs:

- `l4_model_artifacts_v1`;
- `l4_forecasts_v1`;
- `l3_features_v1`;
- `l3_metadata_v1`;
- `l5_evaluation_v1`;
- optional `l6_tests_v1`;
- optional L1 data/regime metadata.

Outputs:

- `l7_importance_v1`;
- `l7_transformation_attribution_v1` when transformation attribution is used.

## Sub-Layers

| Slot | Purpose |
|---|---|
| L7.A | importance DAG body |
| L7.B | output shape and export axes |

## Compatibility Rules

- Tree SHAP and tree-native importance require tree model families.
- Linear SHAP, coefficient importance, and forecast decomposition require linear model families.
- Deep attribution ops require neural-network model families.
- VAR-specific ops require VAR or BVAR families.
- `mrf_gtvp` requires `macroeconomic_random_forest`.
- MCS-filtered sources require active L6 MCS.
- L7 output axes are not sweepable.

## Example

```yaml
7_interpretation:
  enabled: true
  nodes:
    - {id: src_model, type: source, selector: {layer_ref: l4, sink_name: l4_model_artifacts_v1, subset: {model_id: xgb_full}}}
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_l3_meta, type: source, selector: {layer_ref: l3, sink_name: l3_metadata_v1}}
    - {id: shap, type: step, op: shap_tree, params: {model_family: xgboost}, inputs: [src_model, src_X]}
    - {id: lineage, type: step, op: lineage_attribution, params: {level: pipeline_name}, inputs: [shap, src_l3_meta]}
  sinks:
    l7_importance_v1: {global: shap, lineage: lineage}
  fixed_axes:
    figure_type: auto
```

## See encyclopedia

For the full per-axis × per-option catalogue (every value with its OptionDoc summary, when-to-use / when-NOT, references), see [`encyclopedia/l7/`](../../encyclopedia/l7/index.md).
