# Layer 3: Feature Engineering

- Parent: [Detail: Layer Contracts](../index.md)
- Previous: [Layer 2](../layer2/index.md)
- Current: Layer 3
- Next: [Layer 4](../layer4/index.md)

Layer 3 turns cleaned data into the feature and target artifacts consumed by forecasting models. It is a graph layer: recipes must use explicit `nodes` and `sinks`.

## Contract

Inputs:

- `l2_clean_panel_v1`;
- optional raw L1 access for level/raw-feature pipelines;
- optional `l1_regime_metadata_v1`;
- optional L3 pipeline outputs for cascade features.

Outputs:

- `l3_features_v1` with `X_final` and `y_final`;
- `l3_metadata_v1` for lineage and downstream interpretation.

## Sub-Layers

| Slot | Purpose |
|---|---|
| L3.A | target construction |
| L3.B | feature pipelines |
| L3.C | pipeline combine |
| L3.D | feature selection |

## Important Rules

- L3 is DAG-only. `fixed_axes` sugar is rejected.
- `target_construction` must appear exactly once and must be the `y_final` sink.
- `X_final` must be panel-like; `y_final` must be series-like.
- Cascade sources can reference previous L3 pipeline outputs, but cycles and ambiguous pipeline endpoints are hard errors.
- Forecast combination belongs in L4. L3 rejects `weighted_average_forecast`, `median_forecast`, `trimmed_mean_forecast`, `bma_forecast`, `bivariate_ardl_combination`, and retired combine aliases.

## Operational Step Families

Representative operational ops:

- stationary transforms: `log`, `diff`, `log_diff`, `pct_change`;
- lag and aggregation: `lag`, `seasonal_lag`, `ma_window`, `ma_increasing_order`, `cumsum`;
- scaling and reductions: `scale`, `pca`, `sparse_pca`, `scaled_pca`, `dfm`, `varimax`, `partial_least_squares`, `random_projection`;
- feature expansion: `polynomial`, `interaction`, `kernel`, `nystroem`;
- auxiliary: `regime_indicator`, `season_dummy`, `time_trend`, `holiday`;
- combine: `concat`, `interact`, `hierarchical_pca`, `weighted_concat`, `simple_average`;
- selection: `feature_selection`.

Compatibility aliases remain available where older recipes used them: `varimax_rotation`, `polynomial_expansion`, `kernel_features`, and `nystroem_features`.

## Example

```yaml
3_feature_engineering:
  nodes:
    - {id: src_x, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: pca, type: step, op: pca, params: {n_components: 8, temporal_rule: expanding_window_per_origin}, inputs: [src_x]}
    - {id: lag_pca, type: step, op: lag, params: {n_lag: 4}, inputs: [pca]}
    - {id: y_h, type: step, op: target_construction, params: {mode: cumulative_average, horizon: 6}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_pca, y_final: y_h}
    l3_metadata_v1: auto
```

## Related Reference

- [Layer Boundary Contract](../layer_boundary_contract.md)

## See encyclopedia

For the full per-axis × per-option catalogue (every value with its OptionDoc summary, when-to-use / when-NOT, references), see [`encyclopedia/l3/`](../../encyclopedia/l3/index.md).
