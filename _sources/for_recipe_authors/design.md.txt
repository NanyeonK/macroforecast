# Layer Contract Design

macroforecast is organized as explicit layer contracts. Each layer either exposes a list of axes or a DAG of nodes. The contract is the public interface: recipe YAML, Navigator choices, validators, runtime artifacts, and L8 manifests must agree on the same layer IDs, sink names, and option names.

## Layer Map

```text
L0 -> L1 -> L2 -> L3(DAG) -> L4(DAG) -> L5 -> L6 -> L7(DAG) -> L8
        |      |      |       |
       L1.5   L2.5   L3.5    L4.5 diagnostics
```

| Layer | Category | Mode | Purpose |
|---|---|---|---|
| L0 | setup | list | runtime policy: failure handling, reproducibility, compute layout |
| L1 | construction | list | data source, target, predictor universe, geography, sample, horizons, regimes |
| L2 | construction | list | raw-to-clean preprocessing |
| L3 | construction | graph | feature engineering and target construction |
| L4 | construction | graph | model fitting, forecasting, benchmarks, ensembles, tuning |
| L5 | consumption | list | metrics, benchmark-relative evaluation, aggregation, ranking |
| L6 | consumption | list | statistical tests; default off |
| L7 | consumption | graph | interpretation, importance, transformation attribution; default off |
| L8 | consumption | list | export, saved objects, provenance, artifact layout |
| L1.5 | diagnostic | list | raw data summary; default off |
| L2.5 | diagnostic | list | pre/post preprocessing comparison; default off |
| L3.5 | diagnostic | list | feature diagnostics; default off |
| L4.5 | diagnostic | list | model-fit and generator diagnostics; default off |

## Rules That Matter

- L0-L4 are the sweepable construction surface. L5-L8 and diagnostics describe, test, interpret, or export existing cells.
- L3, L4, and L7 are graph layers. Use `nodes` and `sinks`; fixed-axis sugar is not accepted for L3/L4.
- L6, L7, and all `.5` diagnostic layers are default off. When a diagnostic layer has `enabled: false`, it produces no DAG nodes and no sink.
- L8 derives default `saved_objects` from active upstream layers. Active diagnostics are exported as `diagnostics_l1_5`, `diagnostics_l2_5`, `diagnostics_l3_5`, and `diagnostics_l4_5`.
- Forecast combination belongs in L4. L3 rejects L4 forecast-combine ops.
- `study_scope` is not a Layer 0 axis in the current layer-contract system. It is derived into manifest metadata when needed.

## Core Data Flow

L1 defines raw data and regime metadata. L2 consumes L1 and emits the cleaned panel. L3 consumes cleaned data plus optional raw/regime access, then emits `l3_features_v1` and `l3_metadata_v1`. L4 consumes L3 features and emits forecasts, model artifacts, and training metadata. L5 consumes forecasts and produces evaluation artifacts. L6 and L7 are optional consumption layers. L8 collects all active sinks into an export manifest.

Diagnostics are side branches. They inspect upstream artifacts but do not modify construction-layer sinks.

## YAML Shape

Minimal construction path:

```yaml
1_data:
  fixed_axes:
    dataset: fred_md
  leaf_config:
    target: CPIAUCSL

2_preprocessing:
  fixed_axes: {}

3_feature_engineering:
  nodes:
    - {id: src_x, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: x_lag, type: step, op: lag, params: {n_lag: 4}, inputs: [src_x]}
    - {id: y_h, type: step, op: target_construction, params: {horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: x_lag, y_final: y_h}
    l3_metadata_v1: auto

4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - {id: fit_ridge, type: step, op: fit_model, params: {family: ridge}, inputs: [src_X, src_y]}
    - {id: predict_ridge, type: step, op: predict, inputs: [fit_ridge, src_X]}
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: fit_ridge
    l4_training_metadata_v1: auto

5_evaluation:
  fixed_axes: {}

8_output:
  fixed_axes: {}
```

Optional diagnostics:

```yaml
1_5_data_summary:
  enabled: true
  fixed_axes: {}

3_5_feature_diagnostics:
  enabled: true
  fixed_axes:
    comparison_stages: raw_vs_cleaned_vs_features

8_output:
  fixed_axes:
    saved_objects: [forecasts, metrics, ranking, diagnostics_all]
```

## Naming Notes

- Layer IDs in code are lower-snake: `l1`, `l3_5`, `l8`.
- YAML layer keys are numeric names: `1_data`, `3_feature_engineering`, `4_5_generator_diagnostics`.
- Sink names are versioned: `l3_features_v1`, `l4_forecasts_v1`, `l8_artifacts_v1`.
- L3 supports canonical design names such as `varimax`, `polynomial`, `kernel`, and `nystroem`, while retaining compatibility aliases such as `varimax_rotation`, `polynomial_expansion`, `kernel_features`, and `nystroem_features`.

Next: [Data](data/index.md).
