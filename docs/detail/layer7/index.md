# 4.8 Layer 7: Interpretation / Importance

Layer 7 owns interpretation and importance outputs. It consumes trained generators, prediction artifacts, and feature representation metadata.

## Decision order

| Group | Axes |
|---|---|
| Router and scope | `importance_method`, `importance_scope` |
| Method families | `importance_model_native`, `importance_model_agnostic`, `importance_shap`, `importance_local_surrogate`, `importance_partial_dependence`, `importance_grouped`, `importance_stability` |
| Output shape | `importance_aggregation`, `importance_output_style`, `importance_temporal`, `importance_gradient_path` |

## Layer contract

Input:
- trained model or generator artifacts;
- Layer 2 feature names and representation metadata;
- predictions and evaluation context.

Output:
- importance artifacts, local explanations, curves, grouped/stability reports, and manifest entries.

## Related reference

- [Layer Contract Ledger](../layer_contract_ledger.md)
- [Custom Extensions](../custom_extensions.md)
