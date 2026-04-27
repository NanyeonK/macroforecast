# 4.7 Layer 6: Statistical Tests

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.6 Layer 5: Output / Provenance](../layer5/index.md)
- Current: Layer 6
- Next: [4.8 Layer 7: Interpretation / Importance](../layer7/index.md)

Layer 6 owns statistical testing over forecast errors, loss differences, density or interval outputs, directional outcomes, and residual diagnostics.

## Decision order

| Group | Axes |
|---|---|
| Legacy router | `stat_test` |
| Test families | `equal_predictive`, `nested`, `cpa_instability`, `multiple_model`, `density_interval`, `direction`, `residual_diagnostics` |
| Scope and dependence | `test_scope`, `dependence_correction`, `overlap_handling` |

## Naming migration

Layer 6 split test-family axes are canonical. The legacy router still exists
for older recipes, but mixed-case legacy values compile through
`registry_naming_v1`.

| Axis | Legacy value | Canonical value |
|---|---:|---:|
| `density_interval` | `PIT_uniformity` | `pit_uniformity` |
| `residual_diagnostics` | `diagnostics_full` | `full_residual_diagnostics` |
| `stat_test` | `diagnostics_full` | `full_residual_diagnostics` |

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
