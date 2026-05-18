# Standalone functions — L5 metrics

L5 evaluation metrics are planned as standalone callables:

```python
mf.functions.<metric>(y_true, y_pred, **kwargs) -> float
```

> **Cycle 22 note** — `mf.functions.theil_u1` and `mf.functions.theil_u2`
> are the only L5 standalone callables currently shipped (POC). Other metrics
> below are planned for subsequent cycles. Note that `theil_u1` / `theil_u2`
> are also accessible via `mf.functions` in addition to their metrics.py
> location.

## Point metrics (6 options)

| Option | One-liner | Encyclopedia |
|---|---|---|
| `mse` | Mean squared error | [point_metrics axis](../encyclopedia/l5/axes/point_metrics.md#mse) |
| `rmse` | Root mean squared error | [point_metrics axis](../encyclopedia/l5/axes/point_metrics.md#rmse) |
| `mae` | Mean absolute error | [point_metrics axis](../encyclopedia/l5/axes/point_metrics.md#mae) |
| `medae` | Median absolute error | [point_metrics axis](../encyclopedia/l5/axes/point_metrics.md#medae) |
| `mape` | Mean absolute percentage error | [point_metrics axis](../encyclopedia/l5/axes/point_metrics.md#mape) |
| `theil_u1` | Theil U1 inequality coefficient — bounded in [0, 1] | [theil_u1 op page](../encyclopedia/l5/point_metrics/theil_u1.md) |

## Relative metrics (4 options)

| Option | One-liner | Encyclopedia |
|---|---|---|
| `mse_reduction` | MSE reduction vs benchmark | [relative_metrics axis](../encyclopedia/l5/axes/relative_metrics.md#mse-reduction) |
| `r2_oos` | Out-of-sample R² (Clark-West style) | [relative_metrics axis](../encyclopedia/l5/axes/relative_metrics.md#r2-oos) |
| `relative_mae` | MAE ratio vs benchmark | [relative_metrics axis](../encyclopedia/l5/axes/relative_metrics.md#relative-mae) |
| `relative_mse` | MSE ratio vs benchmark | [relative_metrics axis](../encyclopedia/l5/axes/relative_metrics.md#relative-mse) |

## Density metrics (2 options)

| Option | One-liner | Encyclopedia |
|---|---|---|
| `crps` | Continuous ranked probability score | [density_metrics axis](../encyclopedia/l5/axes/density_metrics.md#crps) |
| `log_score` | Log-score for density forecasts | [density_metrics axis](../encyclopedia/l5/axes/density_metrics.md#log-score) |

## Direction metrics (2 options)

| Option | One-liner | Encyclopedia |
|---|---|---|
| `success_ratio` | Hit rate for directional forecasts | [direction_metrics axis](../encyclopedia/l5/axes/direction_metrics.md#success-ratio) |
| `pesaran_timmermann_metric` | Pesaran-Timmermann (1992) direction metric | [direction_metrics axis](../encyclopedia/l5/axes/direction_metrics.md#pesaran-timmermann-metric) |

## Theil family (2 options — POC shipped)

| Option | One-liner | Encyclopedia |
|---|---|---|
| `theil_u1` | Theil U1 — absolute inequality, bounded [0, 1] (**shipped**) | [theil_u1 op page](../encyclopedia/l5/point_metrics/theil_u1.md) |
| `theil_u2` | Theil U2 — ratio vs random-walk benchmark (**shipped**) | [point_metrics axis](../encyclopedia/l5/axes/point_metrics.md#theil-u2) |

## Quick example (theil_u1 — currently shipped)

```python
import macroforecast as mf
import numpy as np

rng = np.random.RandomState(0)
y_true = rng.randn(80)
y_pred = y_true + 0.3 * rng.randn(80)

u1 = mf.functions.theil_u1(y_true, y_pred)
print(f"Theil U1 = {u1:.4f}")   # smaller is better; 0 = perfect

# Theil U2 requires a naive benchmark (e.g. random walk)
naive = np.concatenate([[y_true[0]], y_true[:-1]])
u2 = mf.functions.theil_u2(y_true, naive, y_pred)
print(f"Theil U2 = {u2:.4f}")   # <1 = beats naive; 1 = same as naive
```

## Related

- [L4 fit](l4_fit.md) — generate the `y_pred` from a fit result.
- [L6 tests](l6_tests.md) — statistical significance of metric differences.
- [Encyclopedia L5 index](../encyclopedia/l5/index.md) — full axis × option
  reference including aggregation, decomposition, and ranking.
