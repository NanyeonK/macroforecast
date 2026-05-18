# Standalone functions — L3 feature engineering

L3 is a DAG of typed transforms applied to the cleaned panel. In the recipe
DSL each node carries an `op` and optional `params`. In the standalone
paradigm each op is planned as a callable: `mf.functions.<op>(panel, **params)`.

> **Cycle 22 note** — L3 standalone callables are planned for a future cycle.
> This page documents the 47 operational L3 ops so you can understand the
> feature-engineering surface. The encyclopedia link at the bottom covers
> every option in detail.

## Basic transforms (10 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `lag` | Lagged copy of predictors (n_lag periods) | [op](../encyclopedia/l3/axes/op.md#lag) |
| `seasonal_lag` | Seasonal lag at a fixed seasonal period | [op](../encyclopedia/l3/axes/op.md#seasonal-lag) |
| `diff` | First difference | [op](../encyclopedia/l3/axes/op.md#diff) |
| `cumsum` | Running cumulative sum | [op](../encyclopedia/l3/axes/op.md#cumsum) |
| `level` | Identity pass-through (no transformation) | [op](../encyclopedia/l3/axes/op.md#level) |
| `log` | Natural logarithm | [op](../encyclopedia/l3/axes/op.md#log) |
| `log_diff` | Log first-difference | [op](../encyclopedia/l3/axes/op.md#log-diff) |
| `pct_change` | Percentage change | [op](../encyclopedia/l3/axes/op.md#pct-change) |
| `scale` | Standardise (zero mean, unit variance) | [op](../encyclopedia/l3/axes/op.md#scale) |
| `time_trend` | Linear or polynomial time trend variable | [op](../encyclopedia/l3/axes/op.md#time-trend) |

## Advanced transforms (12 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `ma_window` | Fixed-length moving average | [op](../encyclopedia/l3/axes/op.md#ma-window) |
| `ma_increasing_order` | MARX expanding-order MA (Goulet Coulombe) | [op](../encyclopedia/l3/axes/op.md#ma-increasing-order) |
| `adaptive_ma_rf` | AlbaMA — RF-driven adaptive moving average | [op](../encyclopedia/l3/axes/op.md#adaptive-ma-rf) |
| `asymmetric_trim` | Rank-space transform for Albacore-family (Goulet Coulombe et al. 2024) | [op](../encyclopedia/l3/axes/op.md#asymmetric-trim) |
| `hp_filter` | Hodrick-Prescott filter (trend + cycle decomposition) | [op](../encyclopedia/l3/axes/op.md#hp-filter) |
| `hamilton_filter` | Hamilton (2018) regression-based filter | [op](../encyclopedia/l3/axes/op.md#hamilton-filter) |
| `savitzky_golay_filter` | Savitzky-Golay polynomial smoother | [op](../encyclopedia/l3/axes/op.md#savitzky-golay-filter) |
| `wavelet` | Discrete wavelet decomposition | [op](../encyclopedia/l3/axes/op.md#wavelet) |
| `fourier` | Fourier-basis seasonal features | [op](../encyclopedia/l3/axes/op.md#fourier) |
| `polynomial` | Polynomial feature expansion | [op](../encyclopedia/l3/axes/op.md#polynomial) |
| `polynomial_expansion` | Full polynomial basis up to degree d | [op](../encyclopedia/l3/axes/op.md#polynomial-expansion) |
| `interaction` | Pairwise interaction terms | [op](../encyclopedia/l3/axes/op.md#interaction) |

## Dimension reduction (11 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `pca` | Principal components (standard) | [op](../encyclopedia/l3/axes/op.md#pca) |
| `scaled_pca` | PCA after standardisation | [op](../encyclopedia/l3/axes/op.md#scaled-pca) |
| `sparse_pca` | Sparse PCA (L1-penalised loadings) | [op](../encyclopedia/l3/axes/op.md#sparse-pca) |
| `sparse_pca_chen_rohe` | Fantope-projection sparse PCA (Chen & Rohe 2021) | [op](../encyclopedia/l3/axes/op.md#sparse-pca-chen-rohe) |
| `varimax` | Varimax-rotated PCA | [op](../encyclopedia/l3/axes/op.md#varimax) |
| `varimax_rotation` | Varimax rotation applied to an existing factor matrix | [op](../encyclopedia/l3/axes/op.md#varimax-rotation) |
| `dfm` | Dynamic Factor Model factors | [op](../encyclopedia/l3/axes/op.md#dfm) |
| `hierarchical_pca` | Block-hierarchical PCA (McCracken-Ng groups) | [op](../encyclopedia/l3/axes/op.md#hierarchical-pca) |
| `maf_per_variable_pca` | Maximum Autocorrelation Factor per-variable PCA | [op](../encyclopedia/l3/axes/op.md#maf-per-variable-pca) |
| `partial_least_squares` | Partial least squares (PLS) | [op](../encyclopedia/l3/axes/op.md#partial-least-squares) |
| `random_projection` | Johnson-Lindenstrauss random projection | [op](../encyclopedia/l3/axes/op.md#random-projection) |

## Supervised / selection (6 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `supervised_pca` | Target-supervised principal components | [op](../encyclopedia/l3/axes/op.md#supervised-pca) |
| `sliced_inverse_regression` | SIR dimension reduction | [op](../encyclopedia/l3/axes/op.md#sliced-inverse-regression) |
| `feature_selection` | Variance / correlation / Lasso filter | [op](../encyclopedia/l3/axes/op.md#feature-selection) |
| `kernel` | Kernel feature map | [op](../encyclopedia/l3/axes/op.md#kernel) |
| `kernel_features` | Explicit kernel feature matrix | [op](../encyclopedia/l3/axes/op.md#kernel-features) |
| `nystroem` | Nystroem kernel approximation | [op](../encyclopedia/l3/axes/op.md#nystroem) |

## Final / B1 combiners (8 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `target_construction` | Build forecast target (point / direct / h-step) | [op](../encyclopedia/l3/axes/op.md#target-construction) |
| `midas` | MIDAS distributed-lag features | [op](../encyclopedia/l3/axes/op.md#midas) |
| `u_midas` | Unrestricted MIDAS (U-MIDAS) polynomial features | [op](../encyclopedia/l3/axes/op.md#u-midas) |
| `season_dummy` | Month / quarter dummy variables | [op](../encyclopedia/l3/axes/op.md#season-dummy) |
| `holiday` | Holiday indicator features | [op](../encyclopedia/l3/axes/op.md#holiday) |
| `regime_indicator` | Regime membership indicator from L1.G | [op](../encyclopedia/l3/axes/op.md#regime-indicator) |
| `nystroem_features` | Explicit Nystroem feature set (combine path) | [op](../encyclopedia/l3/axes/op.md#nystroem-features) |
| `l3_feature_bundle` | Pack multiple streams into a single sink | [op](../encyclopedia/l3/axes/op.md#l3-feature-bundle) |

## Quick example (recipe DSL)

```yaml
3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag1, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: pca4, type: step, op: pca, params: {n_components: 4}, inputs: [lag1]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks: {l3_features_v1: {X_final: pca4, y_final: y_h}, l3_metadata_v1: auto}
```

## Related

- [L4 fit](l4_fit.md) — the L3 feature matrix is the input to L4 models.
- [Encyclopedia L3 op axis](../encyclopedia/l3/axes/op.md) — full per-op reference.
