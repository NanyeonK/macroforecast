# Diebold-Mariano Test

`macrocast.evaluation.dm` implements the Diebold-Mariano (1995) test for equal predictive accuracy, with the Harvey-Leybourne-Newbold (1997) small-sample correction applied by default.

**References:**

- Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., and Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.

---

## Test Statistic

Let `d_t = L(e_{1,t}) - L(e_{2,t})` be the per-period loss differential between model 1 and model 2. The DM statistic is:

```
DM = d_bar / sqrt(V_hat / T)
```

where `d_bar` is the sample mean of `d_t` and `V_hat` is the Newey-West HAC variance estimate with bandwidth selected to account for the forecast horizon `h`. Under the null hypothesis of equal predictive accuracy, `DM` is asymptotically `N(0,1)`. With the HLN correction, inference uses a `t(T-1)` distribution instead, which provides better size control in short samples.

---

## dm_test

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | array-like (T,) | required | Realized values |
| `y_hat_1` | array-like (T,) | required | Forecasts from model 1 |
| `y_hat_2` | array-like (T,) | required | Forecasts from model 2 (typically the benchmark) |
| `h` | int | `1` | Forecast horizon; determines the Newey-West bandwidth |
| `loss` | str | `"mse"` | Loss function: `"mse"` or `"mae"` |
| `loss_fn` | callable or None | None | Custom loss function `f(y_true, y_hat) -> array(T,)`; overrides `loss` if provided |
| `nw_bw` | int or None | None | Newey-West bandwidth override; defaults to `h - 1` if None |
| `hln_adjust` | bool | `True` | Whether to apply the Harvey-Leybourne-Newbold small-sample correction |

Returns a `DMResult`.

---

## DMResult

| Field | Type | Description |
|-------|------|-------------|
| `dm_stat` | float | DM test statistic |
| `p_value` | float | Two-sided p-value |
| `loss_diff_mean` | float | Mean loss differential `d_bar`; positive means model 1 is worse than model 2 |
| `hln_adjusted` | bool | Whether the HLN correction was applied |

---

## Example

```python
from macrocast.evaluation import dm_test

result = dm_test(
    y_true=y_true,
    y_hat_1=y_hat_model,
    y_hat_2=y_hat_ar,
    h=1,
    loss="mse",
    hln_adjust=True,
)
print(f"DM stat: {result.dm_stat:.3f}, p-value: {result.p_value:.3f}")
# Negative dm_stat means model 1 is more accurate than model 2
```
