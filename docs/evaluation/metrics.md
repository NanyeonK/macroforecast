# Metrics

`macrocast.evaluation.metrics` provides five scalar and time-series accuracy functions. All functions accept NumPy arrays or pandas Series of shape `(T,)`.

---

## msfe

Mean squared forecast error.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | array-like (T,) | Realized values |
| `y_hat` | array-like (T,) | Point forecasts |

Returns `float`.

---

## mae

Mean absolute error.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | array-like (T,) | Realized values |
| `y_hat` | array-like (T,) | Point forecasts |

Returns `float`.

---

## relative_msfe

Ratio of candidate model MSFE to benchmark MSFE. Values below 1 indicate improvement over the benchmark.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | array-like (T,) | Realized values |
| `y_hat_model` | array-like (T,) | Candidate model forecasts |
| `y_hat_benchmark` | array-like (T,) | Benchmark (AR) forecasts |

Returns `float`.

---

## csfe

Cumulative squared forecast error path over the OOS period. Useful for diagnosing when gains or losses accumulate.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | array-like (T,) | Realized values |
| `y_hat` | array-like (T,) | Point forecasts |

Returns `ndarray` of shape `(T,)`.

---

## oos_r2

OOS R² following Campbell and Thompson (2008). Defined as `1 - relative_msfe`. Positive values indicate the model beats the benchmark in MSFE terms.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | array-like (T,) | Realized values |
| `y_hat_model` | array-like (T,) | Candidate model forecasts |
| `y_hat_benchmark` | array-like (T,) | Benchmark (AR) forecasts |

Returns `float`.

---

## Example

```python
from macrocast.evaluation import relative_msfe, oos_r2, csfe
import numpy as np

# y_true, y_hat_model, y_hat_ar are arrays of shape (T,)
rel = relative_msfe(y_true, y_hat_model, y_hat_ar)
print(f"Relative MSFE: {rel:.3f}")   # < 1 means improvement

r2 = oos_r2(y_true, y_hat_model, y_hat_ar)
print(f"OOS R²: {r2:.3f}")           # positive means model beats AR

# Cumulative squared forecast error over time
path = csfe(y_true, y_hat_model)
```
