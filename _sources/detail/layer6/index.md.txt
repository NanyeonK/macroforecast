# 4.6 Layer 6: Statistical Tests

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.5 Layer 5: Output / Provenance](../layer5/index.md)
- Current: Layer 6
- Next: [4.7 Layer 7: Interpretation / Importance](../layer7/index.md)

Layer 6 owns statistical testing over forecast errors, loss differences, density or interval outputs, directional outcomes, and residual diagnostics.

## Decision order

| Group | Axes |
|---|---|
| Test families | `equal_predictive`, `nested`, `cpa_instability`, `multiple_model`, `density_interval`, `direction`, `residual_diagnostics` |
| Scope and dependence | `test_scope`, `dependence_correction`, `overlap_handling` |

## Canonical names

Layer 6 uses split test-family axes only. Select `equal_predictive`, `nested`, `cpa_instability`, `multiple_model`, `density_interval`, `direction`, or `residual_diagnostics` under `fixed_axes`; leave a family at `none` when inactive. There is no single catch-all test axis in generated recipes.

## Layer contract

Input:
- predictions and evaluation outputs;
- forecast-object type from Layer 3;
- overlap and dependence assumptions.

Output:
- statistical-test artifacts and manifest entries.

## Related reference

- [Layer Contract Ledger](../layer_contract_ledger.md)
- [Artifacts and Manifest](../artifacts_and_manifest.md)
