# 4.7 Layer 6: Statistical Tests

Layer 6 owns statistical testing over forecast errors, loss differences, density or interval outputs, directional outcomes, and residual diagnostics.

## Decision order

| Group | Axes |
|---|---|
| Legacy router | `stat_test` |
| Test families | `equal_predictive`, `nested`, `cpa_instability`, `multiple_model`, `density_interval`, `direction`, `residual_diagnostics` |
| Scope and dependence | `test_scope`, `dependence_correction`, `overlap_handling` |

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
