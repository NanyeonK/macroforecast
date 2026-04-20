# Registry and grammar guide

## Purpose

macrocast separates two ideas explicitly:
- full choice-space representation (what *could* exist)
- current executable runtime support (what *does* work)

The registry layer is the canonical home for that distinction.

## Architecture

The registry uses per-axis files organized by stage:

```
macrocast/registry/
  base.py           # BaseRegistryEntry, EnumRegistryEntry, AxisDefinition
  build.py          # auto-discovery loader, get_axis_registry()
  types.py          # AxisRegistryEntry, AxisSelection, SupportStatus
  stage0/           # 6 axes (study_mode, experiment_unit, axis_type, ...)
  data/             # 29 axes (dataset, frequency, forecast_type, ...)
  preprocessing/    # 24 axes (tcode_policy, scaling_policy, ...)
  training/         # 28 axes (model_family, search_algorithm, ...)
  evaluation/       # 18 axes (point_metrics, regime_definition, ...)
  output/           # 4 axes (export_format, provenance_fields, ...)
  tests/            # 2 axes (stat_test, dependence_correction)
  importance/       # 13 axes (importance_method, shap, stability, ...)
```

## Current scale

- **125 axes** across 8 layers
- **717 total values**
- **310 operational** (43.2%)

## Support status

Each enumerated value has one explicit support state:
- `operational` — executable in current runtime
- `planned` — will be added in near-term
- `registry_only` — representable in grammar, not yet executable
- `future` — long-run, not scheduled

## Canonical layer order

```python
("0_meta", "1_data_task", "2_preprocessing", "3_training",
 "4_evaluation", "5_output_provenance", "6_stat_tests", "7_importance")
```

## Fixed / sweep / conditional semantics

Each axis carries a default policy: `fixed`, `sweep`, or `conditional`.
- `fixed` axes define the common comparison environment
- `sweep` axes are intentionally varied within the study
- `conditional` axes activate based on other choices
