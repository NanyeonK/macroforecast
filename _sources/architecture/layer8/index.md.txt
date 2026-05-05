# Layer 8: Output / Provenance

- Parent: [Detail: Layer Contracts](../index.md)
- Previous: [Layer 7](../layer7/index.md)
- Current: Layer 8

Layer 8 is the external boundary. It collects active layer sinks, writes artifacts, and emits a reproducible manifest.

## Contract

Inputs:

- L0-L7 active sinks;
- active diagnostics from L1.5-L4.5.

Output:

- `l8_artifacts_v1`.

## Sub-Layers

| Slot | Axes |
|---|---|
| L8.A export format | `export_format`, `compression` |
| L8.B saved objects | `saved_objects`, `model_artifacts_format` |
| L8.C provenance | `provenance_fields`, `manifest_format` |
| L8.D artifact layout | `artifact_granularity`, `naming_convention` |

## Defaults

- `export_format: json_csv`;
- `compression: none`;
- `manifest_format: json`;
- `artifact_granularity: per_cell`;
- `naming_convention: descriptive`;
- `descriptive_naming_template: "{model_family}_{forecast_strategy}_h{horizon}"`.

When `saved_objects` is omitted, L8 derives it from the active recipe:

- always: `forecasts`, `metrics`, `ranking`;
- density/quantile forecasts: `forecast_intervals`;
- L5 decomposition: `decomposition`;
- active regimes: `regime_metrics`;
- FRED-SD geography: `state_metrics`;
- L4 ensemble: `combination_weights`;
- active diagnostics: `diagnostics_l1_5`, `diagnostics_l2_5`, `diagnostics_l3_5`, `diagnostics_l4_5`;
- L6: `tests`;
- L7: `importance`;
- L7 transformation attribution: `transformation_attribution`.

`diagnostics_all` expands to all four diagnostic saved-object names.

## Gates

- L8 axes are not sweepable.
- `onnx` and `pmml` model-artifact formats are future and rejected.
- `state_metrics` requires FRED-SD.
- `regime_metrics` requires active L1 regime.
- `combination_weights` requires an L4 ensemble combine node.
- `custom` naming requires `leaf_config.custom_naming_function`.
- `latex_tables`, `markdown_report`, and `html_report` require L5 to be active.

## Example

```yaml
8_output:
  fixed_axes:
    export_format: all
    saved_objects: [forecasts, metrics, ranking, diagnostics_all, tests, importance]
    artifact_granularity: per_cell
    naming_convention: descriptive
  leaf_config:
    output_directory: ./paper_replication/main/
    descriptive_naming_template: "{model_family}_{forecast_strategy}_h{horizon}_{combine_method}"
```
