# Layer 4: Forecasting Model

- Parent: [Detail: Layer Contracts](../index.md)
- Previous: [Layer 3](../layer3/index.md)
- Current: Layer 4
- Next: [Layer 5](../layer5/index.md)

Layer 4 consumes `X_final` and `y_final`, fits forecasting models, emits forecasts, and records model/training artifacts. It is a graph layer.

## Contract

Inputs:

- `l3_features_v1`;
- `l3_metadata_v1`;
- optional `l1_regime_metadata_v1`.

Outputs:

- `l4_forecasts_v1`;
- `l4_model_artifacts_v1`;
- `l4_training_metadata_v1`.

## Sub-Layers

| Slot | Purpose |
|---|---|
| L4.A | model selection and forecast-combine nodes |
| L4.B | forecast strategy |
| L4.C | training window and refit policy |
| L4.D | tuning |

## Benchmark Contract

A benchmark is not a separate axis. Mark exactly one `fit_model` node with `is_benchmark: true`. L5 and L6 detect the benchmark from L4 model artifacts. Zero benchmark nodes is valid; two or more benchmark nodes is a hard error.

## Forecast Combination

Forecast combination belongs in L4 and consumes forecast artifacts:

- `weighted_average_forecast`;
- `median_forecast`;
- `trimmed_mean_forecast`;
- `bma_forecast`;
- `bivariate_ardl_combination`.

`weighted_average_forecast.weights_method` supports `equal`, `dmsfe`, `inverse_msfe`, `mallows_cp`, `sic_weights`, `granger_ramanathan`, and `cv_optimized`. `full_sample_once` is rejected for forecast-combination temporal rules.

## Example

```yaml
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - {id: fit_ar, type: step, op: fit_model, params: {family: ar_p}, is_benchmark: true, inputs: [src_y]}
    - {id: predict_ar, type: step, op: predict, inputs: [fit_ar]}
    - {id: fit_ridge, type: step, op: fit_model, params: {family: ridge}, inputs: [src_X, src_y]}
    - {id: predict_ridge, type: step, op: predict, inputs: [fit_ridge, src_X]}
    - {id: ensemble, type: combine, op: weighted_average_forecast, params: {weights_method: dmsfe}, inputs: [predict_ar, predict_ridge]}
  sinks:
    l4_forecasts_v1: ensemble
    l4_model_artifacts_v1: [fit_ar, fit_ridge]
    l4_training_metadata_v1: auto
```

## Related Reference

- [Artifacts and Manifest](../artifacts_and_manifest.md)

## See encyclopedia

For the full per-axis × per-option catalogue (every value with its OptionDoc summary, when-to-use / when-NOT, references), see [`encyclopedia/l4/`](../../encyclopedia/l4/index.md).
