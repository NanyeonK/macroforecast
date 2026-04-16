# Evaluation Metrics — Mathematical Background

## Point forecast metrics

Let $y_t$ denote the true value and $\hat{y}_t$ the forecast at time $t$, with $T$ out-of-sample observations.

### Mean Squared Forecast Error (MSFE)

$$\text{MSFE} = \frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2$$

### Root Mean Squared Forecast Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSFE}}$$

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{T} \sum_{t=1}^{T} |y_t - \hat{y}_t|$$

### Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{1}{T} \sum_{t=1}^{T} \frac{|y_t - \hat{y}_t|}{|y_t|}$$

## Relative metrics

Let $e_t^m = y_t - \hat{y}_t^m$ (model error) and $e_t^b = y_t - \hat{y}_t^b$ (benchmark error).

### Relative MSFE

$$\text{rMSFE} = \frac{\text{MSFE}_{model}}{\text{MSFE}_{benchmark}} = \frac{\sum (e_t^m)^2}{\sum (e_t^b)^2}$$

Interpretation: rMSFE < 1 means the model beats the benchmark.

### Out-of-Sample R-squared

$$R^2_{OOS} = 1 - \text{rMSFE} = 1 - \frac{\sum (e_t^m)^2}{\sum (e_t^b)^2}$$

Interpretation: $R^2_{OOS} > 0$ means the model outperforms the benchmark. Analogous to in-sample $R^2$ but computed on out-of-sample data.

Reference: Campbell & Thompson (2008)

### Cumulative Squared Forecast Error (CSFE)

$$\text{CSFE}_T = \sum_{t=1}^{T} (e_t^m)^2$$

The CSFE difference $\text{CSFE}^b_T - \text{CSFE}^m_T$ plotted over time reveals periods where the model gains or loses relative to the benchmark.

Reference: Goyal & Welch (2008)

### Benchmark Win Rate

$$\text{WR} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}\{(e_t^m)^2 < (e_t^b)^2\}$$

### Directional Accuracy

$$\text{DA} = \frac{1}{T-1} \sum_{t=2}^{T} \mathbf{1}\{\text{sign}(\Delta y_t) = \text{sign}(\Delta \hat{y}_t)\}$$

where $\Delta y_t = y_t - y_{t-1}$.

**See also:** [User Guide: Execution](../user_guide/execution.md) | [User Guide: Statistical Tests](../user_guide/stat_tests.md)
