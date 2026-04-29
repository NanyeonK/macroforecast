# 4.7 Layer 7: Interpretation / Importance

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.6 Layer 6: Statistical Tests](../layer6/index.md)
- Current: Layer 7

Layer 7 owns interpretation and importance outputs. It consumes trained generators, prediction artifacts, and feature representation metadata.

## Decision order

| Group | Axes |
|---|---|
| Router and scope | `importance_method`, `importance_scope` |
| Method families | `importance_model_native`, `importance_model_agnostic`, `importance_shap`, `importance_local_surrogate`, `importance_partial_dependence`, `importance_grouped`, `importance_stability` |
| Output shape | `importance_aggregation`, `importance_output_style`, `importance_temporal`, `importance_gradient_path` |

## Naming migration

Layer 7 already uses lower-snake canonical IDs for the current importance
sub-axes. This pass does not rename Layer 7 values; the important cleanup is
that Layer 7 remains separated from Layer 6 statistical tests and Layer 5
artifact export.

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
