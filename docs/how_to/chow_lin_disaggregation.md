# How to disaggregate a quarterly series with Chow-Lin

The goal is to disaggregate a quarterly series to monthly frequency
using the Chow and Lin (1971) canonical GLS method with an AR(1) error
structure and a related high-frequency indicator. We use the public
function `chow_lin_disaggregate` from `macroforecast.layers.l3_features.transforms`. The
The v0.9.5 promotion implements the full canonical algorithm directly (BLUE
correction with AR(1) covariance), in contrast to the simpler OLS
fallback used in older internal helpers.

## Setup

```python
import numpy as np
import pandas as pd
from macroforecast.layers.l3_features.transforms import chow_lin_disaggregate

rng = np.random.RandomState(0)

# 36 months of a monthly indicator series, for example retail sales.
idx_m = pd.date_range("2018-01-31", periods=36, freq="ME")
indicator = pd.Series(0.1 * np.arange(36) + rng.randn(36),
                      index=idx_m, name="retail_sales")

# 12 quarters of an observed quarterly target series, for example GDP.
idx_q = pd.date_range("2018-03-31", periods=12, freq="QE")
indicator_q = indicator.resample("QE").mean()
gdp_q = pd.Series(2.0 + 1.5 * indicator_q.values + 0.2 * rng.randn(12),
                  index=idx_q, name="gdp_q")
```

## Disaggregate, mean mode

```python
gdp_m_mean = chow_lin_disaggregate(
    low_freq=gdp_q,
    indicator_high_freq=indicator,
    aggregation="mean",
    rho_method="min_chi_squared",
)
```

The default `rho_method='min_chi_squared'` estimates the AR(1)
autocorrelation parameter over a 50-point grid in `[0, 0.98]`. Pass
`rho=0.7` explicitly to fix the parameter. The default
`aggregation='mean'` is appropriate when the low-frequency observation
is the quarterly average of the monthly flow.

## Disaggregate, sum mode

For series where the quarterly figure is the sum of the three monthly
values, use `aggregation='sum'`.

```python
gdp_m_sum = chow_lin_disaggregate(
    low_freq=gdp_q,
    indicator_high_freq=indicator,
    aggregation="sum",
)

# Round-trip conservation check.
recovered_q = gdp_m_sum.resample("QE").sum()
np.testing.assert_allclose(recovered_q.values, gdp_q.values, atol=1e-8)
```

## Output

The function returns a `pd.Series` aligned to the indicator index,
trimmed to `n_l * m` observations. The conservation property holds to
numerical tolerance. With `aggregation='sum'`, `gdp_m.resample('QE').sum()`
matches `gdp_q` to atol `1e-8`, and the corresponding mean-aggregation
identity holds for `aggregation='mean'`.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: aggregation must be 'sum' or 'mean'` | Invalid mode | Use `'sum'` for flows aggregated by sum and `'mean'` otherwise |
| `ValueError: rho must be in the open interval (-1, 1)` | Passed rho outside the valid range | Pass `rho=None` to let the function estimate it, or supply rho in (-1, 1) |
| Returned series is the bfill/ffill fallback | Fewer than three aligned low-frequency observations after resampling | Provide more quarters of data, or check that the indicator index covers the low-frequency span |
| Conservation property does not hold | Wrong aggregation mode for the underlying data type | Match `aggregation` to the stock-vs-flow nature of the series |

## See also

- {doc}`user_data_workflow`
- Paper: Chow and Lin (1971), Review of Economics and Statistics 53(4).
