# Standalone functions: L5 metrics (15 ops)

L5 metric callables take arrays of true values and predictions and return a scalar float. They have no result dataclass - they return `float` directly.

## Point metrics (7 ops)

#### `mae(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Mean absolute error.

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
value = mf.functions.mae(y_true, y_pred)
print(value)
```

[Encyclopedia](../encyclopedia/l5/point_metrics/mae.md)

#### `mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-10) -> float`

Mean absolute percentage error.

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
value = mf.functions.mape(y_true, y_pred)
print(value)
```

[Encyclopedia](../encyclopedia/l5/point_metrics/mape.md)

#### `medae(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Median absolute error.

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
value = mf.functions.medae(y_true, y_pred)
print(value)
```

[Encyclopedia](../encyclopedia/l5/point_metrics/medae.md)

#### `mse(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Mean squared error.

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
value = mf.functions.mse(y_true, y_pred)
print(value)
```

[Encyclopedia](../encyclopedia/l5/point_metrics/mse.md)

#### `rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Root mean squared error.

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
value = mf.functions.rmse(y_true, y_pred)
print(value)
```

[Encyclopedia](../encyclopedia/l5/point_metrics/rmse.md)

#### `theil_u1(y_true: np.ndarray, y_pred: np.ndarray) -> float`

Theil U1 inequality coefficient in [0, 1].

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
value = mf.functions.theil_u1(y_true, y_pred)
print(value)
```

[Encyclopedia](../encyclopedia/l5/point_metrics/theil_u1.md)

#### `theil_u2(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float`

Theil U2 ratio vs naive forecast (requires `y_prev`).

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
y_prev = np.array([0.9, 1.9, 2.8, 3.7, 5.0])
value = mf.functions.theil_u2(y_true, y_pred, y_prev)
print(value)
```

[Encyclopedia](../encyclopedia/l5/point_metrics/theil_u2.md)

## Density metrics (2 ops)

#### `coverage_rate(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float`

Empirical coverage: fraction of y_true inside [y_lower, y_upper].

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
y_lower = y_pred - 0.5
y_upper = y_pred + 0.5
value = mf.functions.coverage_rate(y_true, y_lower, y_upper)
print(value)
```

[Encyclopedia](../encyclopedia/l5/density_metrics/coverage_rate.md)

#### `interval_score(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, *, alpha: float = 0.05) -> float`

Winkler interval score (lower = better).

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
y_lower = y_pred - 0.5
y_upper = y_pred + 0.5
value = mf.functions.interval_score(y_true, y_lower, y_upper)
print(value)
```

[Encyclopedia](../encyclopedia/l5/density_metrics/interval_score.md)

## Directional metrics (2 ops)

#### `pesaran_timmermann_metric(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.0) -> float`

Pesaran-Timmermann (1992) directional test statistic.

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
value = mf.functions.pesaran_timmermann_metric(y_true, y_pred)
print(value)
```

[Encyclopedia](../encyclopedia/l5/direction_metrics/pesaran_timmermann_metric.md)

#### `success_ratio(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float`

Directional accuracy: fraction of correct sign predictions (requires `y_prev`).

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
y_prev = np.array([0.9, 1.9, 2.8, 3.7, 5.0])
value = mf.functions.success_ratio(y_true, y_pred, y_prev)
print(value)
```

[Encyclopedia](../encyclopedia/l5/direction_metrics/success_ratio.md)

## Relative metrics (4 ops)

#### `mse_reduction(y_true: np.ndarray, y_model: np.ndarray, y_benchmark: np.ndarray) -> float`

Percentage MSE reduction vs benchmark.

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
y_bench = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
value = mf.functions.mse_reduction(y_true, y_pred, y_bench)
print(value)
```

[Encyclopedia](../encyclopedia/l5/relative_metrics/mse_reduction.md)

#### `r2_oos(y_true: np.ndarray, y_model: np.ndarray, y_benchmark: np.ndarray) -> float`

OOS R-squared (Campbell-Thompson 2008).

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
y_bench = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
value = mf.functions.r2_oos(y_true, y_pred, y_bench)
print(value)
```

[Encyclopedia](../encyclopedia/l5/relative_metrics/r2_oos.md)

#### `relative_mae(y_true: np.ndarray, y_model: np.ndarray, y_benchmark: np.ndarray) -> float`

MAE ratio: MAE(model) / MAE(benchmark).

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
y_bench = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
value = mf.functions.relative_mae(y_true, y_pred, y_bench)
print(value)
```

[Encyclopedia](../encyclopedia/l5/relative_metrics/relative_mae.md)

#### `relative_mse(y_true: np.ndarray, y_model: np.ndarray, y_benchmark: np.ndarray) -> float`

MSE ratio: MSE(model) / MSE(benchmark).

Returns `float`.

```python
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
y_bench = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
value = mf.functions.relative_mse(y_true, y_pred, y_bench)
print(value)
```

[Encyclopedia](../encyclopedia/l5/relative_metrics/relative_mse.md)

## Quick example

```python
import macroforecast as mf
import numpy as np

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])

print('MSE :', mf.functions.mse(y_true, y_pred))
print('RMSE:', mf.functions.rmse(y_true, y_pred))
print('MAE :', mf.functions.mae(y_true, y_pred))
```
