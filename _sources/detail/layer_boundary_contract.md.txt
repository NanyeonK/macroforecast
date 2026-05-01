# Layer Boundary Contract

This page records the current public layer boundaries.

## Canonical Flow

```text
L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> L7 -> L8
```

Diagnostics attach as side branches:

```text
L1.5 <- L1
L2.5 <- L1 + L2
L3.5 <- L1 + L2 + L3
L4.5 <- L4 + L3
```

## Boundary Table

| Layer | Owns | Consumes | Produces |
|---|---|---|---|
| L0 | runtime policy | none | `l0_meta_v1` |
| L1 | data definition | none | `l1_data_definition_v1`, `l1_regime_metadata_v1` |
| L2 | preprocessing | `l1_data_definition_v1` | `l2_clean_panel_v1` |
| L3 | feature engineering and target construction | L1 raw/regime access, `l2_clean_panel_v1` | `l3_features_v1`, `l3_metadata_v1` |
| L4 | model fitting, forecasts, benchmarks, ensembles, tuning | `l3_features_v1`, `l3_metadata_v1`, optional regimes | `l4_forecasts_v1`, `l4_model_artifacts_v1`, `l4_training_metadata_v1` |
| L5 | evaluation metrics and ranking | L4 forecasts/artifacts, L1 data/regimes, L3 metadata | `l5_evaluation_v1` |
| L6 | statistical tests | L4, L5, L1 metadata | `l6_tests_v1` |
| L7 | interpretation and importance | L4, L3, L5, optional L6 | `l7_importance_v1`, optional `l7_transformation_attribution_v1` |
| L8 | export and provenance | all active upstream sinks | `l8_artifacts_v1` |

## Non-Negotiable Rules

- L3 owns panel and target construction. It does not own forecast combination.
- L4 owns forecast generation and forecast combination.
- L5 owns descriptive evaluation. It does not own inferential p-values.
- L6 owns inferential tests and is default off.
- L7 owns interpretation and is default off.
- L8 owns file export, saved object selection, provenance, and artifact layout.
- Diagnostics do not mutate construction sinks.

## Diagnostic Default-Off Contract

For L1.5-L4.5:

- absent layer key: no diagnostic layer;
- `enabled: false`: no DAG nodes, no sink, all axes inactive;
- `enabled: true`: diagnostic DAG and sink are available;
- L8 can include active diagnostics through explicit saved objects or `diagnostics_all`.

## Naming

| Concept | Current name |
|---|---|
| L3 YAML key | `3_feature_engineering` |
| L4 YAML key | `4_forecasting_model` |
| L5 YAML key | `5_evaluation` |
| L6 YAML key | `6_statistical_tests` |
| L7 YAML key | `7_interpretation` |
| L8 YAML key | `8_output` |
