# Standalone functions: L3 transforms (36 ops)

L3 transform callables apply a single feature engineering operation to a panel DataFrame and return a new DataFrame. They correspond to `op` values in the L3 recipe DAG.

Note: `apply_tcode_transform` is documented in [l2_clean.md](l2_clean.md) because its source module is `macroforecast.functions.clean`.

## Basic transforms (10 ops)

#### `asymmetric_trim_transform(panel: pd.DataFrame) -> pd.DataFrame`

Asymmetric trimming of extreme values (rank-space transform).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.asymmetric_trim_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/asymmetric_trim.md)

#### `cumsum_transform(panel: pd.DataFrame) -> pd.DataFrame`

Cumulative sum over time axis.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.cumsum_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/cumsum.md)

#### `diff_transform(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame`

k-th order difference y_t - y_{t-periods}.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.diff_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/diff.md)

#### `holiday_transform(panel: pd.DataFrame) -> pd.DataFrame`

US federal holiday indicator column (requires DatetimeIndex).

Returns `pd.DataFrame`.

```python
idx = pd.date_range('2023-01-01', periods=30, freq='D')
panel = pd.DataFrame(np.ones((30, 2)), index=idx)
out = mf.functions.holiday_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/holiday.md)

#### `log_diff_transform(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame`

Log-difference (log return): log(y_t) - log(y_{t-periods}).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.abs(np.random.randn(60, 5)) + 0.1)
out = mf.functions.log_diff_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/log_diff.md)

#### `log_transform(panel: pd.DataFrame) -> pd.DataFrame`

Natural logarithm of each column.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.abs(np.random.randn(60, 5)) + 0.1)
out = mf.functions.log_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/log.md)

#### `pct_change_transform(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame`

Percentage change: (y_t - y_{t-1}) / |y_{t-1}|.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.pct_change_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/pct_change.md)

#### `scale_transform(panel: pd.DataFrame, *, method: str = zscore) -> pd.DataFrame`

Column-wise normalisation (zscore or minmax).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.scale_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/scale.md)

#### `season_dummy_transform(panel: pd.DataFrame, *, season: str = quarter) -> pd.DataFrame`

Seasonal dummy variables (quarter or month).

Returns `pd.DataFrame`.

```python
idx = pd.date_range('2023-01-01', periods=60, freq='MS')
panel = pd.DataFrame(np.random.randn(60, 3), index=idx)
out = mf.functions.season_dummy_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/season_dummy.md)

#### `time_trend_transform(panel: pd.DataFrame) -> pd.DataFrame`

Linear time-trend column.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.time_trend_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/time_trend.md)

## Filter and window transforms (9 ops)

#### `fourier_transform(panel: pd.DataFrame, *, n_terms: int = 4, period: int = 12) -> pd.DataFrame`

Fourier sin/cos calendar features.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.fourier_transform(panel, n_terms=3, period=12)
```

[Encyclopedia](../encyclopedia/l3/op/fourier.md)

#### `hamilton_filter_transform(panel: pd.DataFrame, *, h: int = 8, p: int = 4) -> pd.DataFrame`

Hamilton (2018) regression-based detrending.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.hamilton_filter_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/hamilton_filter.md)

#### `hp_filter_transform(panel: pd.DataFrame, *, lambda_: float = 1600) -> pd.DataFrame`

Hodrick-Prescott cycle component (kwarg is `lambda_`).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.hp_filter_transform(panel, lambda_=1600)
```

[Encyclopedia](../encyclopedia/l3/op/hp_filter.md)

#### `lag_matrix(panel: pd.DataFrame, *, n_lag: int = 4, include_contemporaneous: bool = False) -> pd.DataFrame`

Lag matrix: panel shifted by 1..n_lag periods.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.lag_matrix(panel, n_lag=3)
```

[Encyclopedia](../encyclopedia/l3/op/lag.md)

#### `ma_increasing_order_transform(panel: pd.DataFrame, *, max_order: int = 12) -> pd.DataFrame`

Expanding moving-average sequence (MARX, Goulet Coulombe).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.ma_increasing_order_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/ma_increasing_order.md)

#### `ma_window_transform(panel: pd.DataFrame, *, window: int = 3) -> pd.DataFrame`

Fixed-window moving average.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.ma_window_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/ma_window.md)

#### `savitzky_golay_transform(panel: pd.DataFrame, *, window: int = 7, polyorder: int = 3) -> pd.DataFrame`

Savitzky-Golay smoothing filter (kwarg is `window`).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.savitzky_golay_transform(panel, window=7, polyorder=3)
```

[Encyclopedia](../encyclopedia/l3/op/savitzky_golay_filter.md)

#### `seasonal_lag_matrix(panel: pd.DataFrame, *, seasonal_period: int = 12, n_seasonal_lags: int = 1) -> pd.DataFrame`

Seasonal lag matrix at fixed seasonal period.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.seasonal_lag_matrix(panel, seasonal_period=12, n_seasonal_lags=1)
```

[Encyclopedia](../encyclopedia/l3/op/seasonal_lag.md)

#### `wavelet_transform(panel: pd.DataFrame, *, wavelet: str = db4, n_levels: int = 3) -> pd.DataFrame`

Wavelet decomposition detail and approximation features.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.wavelet_transform(panel, wavelet='db4', n_levels=2)
```

[Encyclopedia](../encyclopedia/l3/op/wavelet.md)

## Dimensionality reduction transforms (13 ops)

#### `kernel_features_transform(panel: pd.DataFrame, *, kind: str = rbf, gamma: float = 1.0) -> pd.DataFrame`

Kernel feature map (kwarg is `kind`).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.kernel_features_transform(panel, kind='rbf', gamma=1.0)
```

[Encyclopedia](../encyclopedia/l3/op/kernel_features.md)

#### `maf_per_variable_pca_transform(panel: pd.DataFrame, *, n_lags: int = 12, n_components_per_var: int = 2) -> pd.DataFrame`

Maximum autocorrelation factor per-variable PCA.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.maf_per_variable_pca_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/maf_per_variable_pca.md)

#### `nystroem_transform(panel: pd.DataFrame, *, n_components: int = 32) -> pd.DataFrame`

Nystroem approximation of kernel feature map.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.nystroem_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/nystroem.md)

#### `partial_least_squares_transform(panel: pd.DataFrame, target: pd.Series, *, n_components: int = 3) -> pd.DataFrame`

PLS regression factors (requires positional `target` arg).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
target = pd.Series(np.random.randn(60))
out = mf.functions.partial_least_squares_transform(panel, target, n_components=2)
```

[Encyclopedia](../encyclopedia/l3/op/partial_least_squares.md)

#### `pca_transform(panel: pd.DataFrame, *, n_components: "int | str" = 3) -> pd.DataFrame`

Principal component analysis.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.pca_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/pca.md)

#### `polynomial_expansion_transform(panel: pd.DataFrame, *, degree: int = 2) -> pd.DataFrame`

Polynomial feature expansion up to given degree.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 3))
out = mf.functions.polynomial_expansion_transform(panel, degree=2)
```

[Encyclopedia](../encyclopedia/l3/op/polynomial_expansion.md)

#### `random_projection_transform(panel: pd.DataFrame, *, n_components: int = 8) -> pd.DataFrame`

Johnson-Lindenstrauss random projection.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.random_projection_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/random_projection.md)

#### `scaled_pca_transform(panel: pd.DataFrame, target: pd.Series, *, n_components: int = 3) -> pd.DataFrame`

PCA with prior z-score scaling (requires positional `target` arg).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
target = pd.Series(np.random.randn(60))
out = mf.functions.scaled_pca_transform(panel, target, n_components=2)
```

[Encyclopedia](../encyclopedia/l3/op/scaled_pca.md)

#### `sliced_inverse_regression_transform(panel: pd.DataFrame, target: pd.Series, *, n_components: int = 3, n_slices: int = 10, scaling_method: str = scaled_pca) -> pd.DataFrame`

Sliced inverse regression (Li 1991) dimension reduction.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
target = pd.Series(np.random.randn(60))
out = mf.functions.sliced_inverse_regression_transform(panel, target, n_components=2)
```

[Encyclopedia](../encyclopedia/l3/op/sliced_inverse_regression.md)

#### `sparse_pca_chen_rohe_transform(panel: pd.DataFrame, *, n_components: int = 4, zeta: float = 0.0, max_iter: int = 200, var_innovations: bool = False, random_state: int = 0) -> pd.DataFrame`

Sparse PCA via Chen-Rohe SCA.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.sparse_pca_chen_rohe_transform(panel, n_components=2)
```

[Encyclopedia](../encyclopedia/l3/op/sparse_pca_chen_rohe.md)

#### `sparse_pca_transform(panel: pd.DataFrame, *, n_components: int = 8) -> pd.DataFrame`

Sparse PCA with L1-penalised loadings.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.sparse_pca_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/sparse_pca.md)

#### `supervised_pca_transform(panel: pd.DataFrame, target: pd.Series, *, n_components: int = 3, q: float = 0.5) -> pd.DataFrame`

Supervised PCA (Bair et al. 2006) - positional `target` required.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
target = pd.Series(np.random.randn(60))
out = mf.functions.supervised_pca_transform(panel, target, n_components=2)
```

[Encyclopedia](../encyclopedia/l3/op/supervised_pca.md)

#### `varimax_transform(panel: pd.DataFrame) -> pd.DataFrame`

Varimax-rotated factor loadings.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.varimax_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/varimax.md)

## Data / feature management transforms (4 ops)

#### `adaptive_ma_rf_transform(panel: pd.DataFrame, *, n_estimators: int = 100, min_samples_leaf: int = 40, sided: str = two, random_state: "int | None" = 0) -> pd.DataFrame`

Adaptive MA order selection via random-forest importance (AlbaMA).

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.adaptive_ma_rf_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/adaptive_ma_rf.md)

#### `dfm_transform(panel: pd.DataFrame, *, n_factors: int = 3) -> pd.DataFrame`

Dynamic factor model factors via EM.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.dfm_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/dfm.md)

#### `feature_selection_transform(panel: pd.DataFrame, target: "pd.Series | None" = None, *, n_features: "int | float" = 0.5, method: str = variance) -> pd.DataFrame`

Univariate or variance-based feature selection.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.feature_selection_transform(panel, n_features=3)
```

[Encyclopedia](../encyclopedia/l3/op/feature_selection.md)

#### `interaction_terms_transform(panel: pd.DataFrame) -> pd.DataFrame`

Pairwise interaction (product) terms.

Returns `pd.DataFrame`.

```python
panel = pd.DataFrame(np.random.randn(60, 5), columns=[f'x{i}' for i in range(5)])
out = mf.functions.interaction_terms_transform(panel)
```

[Encyclopedia](../encyclopedia/l3/op/interaction.md)

## Quick example

```python
import macroforecast as mf
import pandas as pd
import numpy as np

panel = pd.DataFrame(np.random.randn(100, 20), columns=[f'x{i}' for i in range(20)])
factors = mf.functions.pca_transform(panel, n_components=5)
lags    = mf.functions.lag_matrix(panel, n_lag=3)
cycle   = mf.functions.hp_filter_transform(panel, lambda_=1600)
```
