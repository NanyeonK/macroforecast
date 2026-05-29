# Standalone functions: preprocessing clean helpers

These callables apply one data-cleaning operation to a panel DataFrame and
return a cleaned DataFrame. For the full callable preprocessing workflow, see
[macroforecast.preprocessing](../preprocessing.md).

## Callables

#### `apply_tcode_transform(panel: pd.DataFrame, tcode_map: dict[str, int]) -> pd.DataFrame`

Apply McCracken-Ng tcode transformations to each column via a tcode_map dict.

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
out = mf.functions.apply_tcode_transform(panel, {"a": 2})
```

[Preprocessing reference](../preprocessing.md)

#### `drop_unbalanced_series_clean(panel: pd.DataFrame) -> pd.DataFrame`

Drop columns that are entirely NaN or below minimum coverage.

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"a": [None, 1.0, 2.0, 3.0]})
out = mf.functions.drop_unbalanced_series_clean(panel)
```

[Preprocessing reference](../preprocessing.md)

#### `em_factor_impute_clean(panel: pd.DataFrame, *, n_factors: int = 8, max_iter: int = 20, tol: float = 0.0001) -> pd.DataFrame`

EM-algorithm factor model imputation for missing values.

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame(np.random.randn(60, 10))
panel.iloc[5:8, 2] = np.nan
out = mf.functions.em_factor_impute_clean(panel, n_factors=2)
```

[Preprocessing reference](../preprocessing.md)

#### `em_multivariate_impute_clean(panel: pd.DataFrame, *, max_iter: int = 20, tol: float = 0.0001) -> pd.DataFrame`

EM-algorithm multivariate (full-covariance) imputation.

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame(np.random.randn(60, 5))
panel.iloc[5:8, 2] = np.nan
out = mf.functions.em_multivariate_impute_clean(panel)
```

[Preprocessing reference](../preprocessing.md)

#### `forward_fill_clean(panel: pd.DataFrame) -> pd.DataFrame`

Forward-fill missing values (last observation carried forward).

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"a": [None, 1.0, 2.0, 3.0]})
out = mf.functions.forward_fill_clean(panel)
```

[Preprocessing reference](../preprocessing.md)

#### `freq_align_monthly_to_quarterly_clean(panel: pd.DataFrame, monthly_columns: list[str], *, rule: str = quarterly_average) -> pd.DataFrame`

Aggregate specified monthly columns to quarterly frequency.

Returns `pd.DataFrame` (same index as input unless noted).

```python
monthly_idx = pd.date_range('2020-01-01', periods=24, freq='MS')
panel = pd.DataFrame({'gdp': range(24)}, index=monthly_idx)
out = mf.functions.freq_align_monthly_to_quarterly_clean(panel, ['gdp'])
```

[Preprocessing reference](../preprocessing.md)

#### `freq_align_quarterly_to_monthly_clean(panel: pd.DataFrame, quarterly_columns: list[str], *, rule: str = step_backward) -> pd.DataFrame`

Expand specified quarterly columns to monthly frequency.

Returns `pd.DataFrame` (same index as input unless noted).

```python
q_idx = pd.date_range('2020-01-01', periods=8, freq='QS')
panel = pd.DataFrame({'cpi': range(8)}, index=q_idx)
out = mf.functions.freq_align_quarterly_to_monthly_clean(panel, ['cpi'])
```

[Preprocessing reference](../preprocessing.md)

#### `iqr_outlier_clean(panel: pd.DataFrame, *, threshold: float = 10.0, action: str = flag_as_nan) -> pd.DataFrame`

IQR-based outlier detection (McCracken-Ng convention), flags or winsorises extremes.

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"x": np.random.randn(100)})
out = mf.functions.iqr_outlier_clean(panel, threshold=3.0)
```

[Preprocessing reference](../preprocessing.md)

#### `linear_interpolate_clean(panel: pd.DataFrame) -> pd.DataFrame`

Linear interpolation for missing interior values.

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"a": [None, 1.0, 2.0, 3.0]})
out = mf.functions.linear_interpolate_clean(panel)
```

[Preprocessing reference](../preprocessing.md)

#### `mean_impute_clean(panel: pd.DataFrame) -> pd.DataFrame`

Column-mean imputation for missing values.

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"a": [None, 1.0, 2.0, 3.0]})
out = mf.functions.mean_impute_clean(panel)
```

[Preprocessing reference](../preprocessing.md)

#### `truncate_to_balanced_clean(panel: pd.DataFrame) -> pd.DataFrame`

Truncate rows until the panel is balanced (no leading NaN).

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"a": [None, 1.0, 2.0, 3.0]})
out = mf.functions.truncate_to_balanced_clean(panel)
```

[Preprocessing reference](../preprocessing.md)

#### `winsorize_clean(panel: pd.DataFrame, *, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame`

Winsorise at specified quantile bounds (lower_quantile / upper_quantile).

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"x": [-10.0, 0.0, 1.0, 2.0, 100.0]})
out = mf.functions.winsorize_clean(panel, lower_quantile=0.01, upper_quantile=0.99)
```

[Preprocessing reference](../preprocessing.md)

#### `zero_fill_leading_clean(panel: pd.DataFrame) -> pd.DataFrame`

Fill leading NaN with zeros.

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"a": [None, 1.0, 2.0, 3.0]})
out = mf.functions.zero_fill_leading_clean(panel)
```

[Preprocessing reference](../preprocessing.md)

#### `zscore_outlier_clean(panel: pd.DataFrame, *, threshold: float = 3.0, action: str = flag_as_nan) -> pd.DataFrame`

Z-score threshold outlier detection, flags or caps at ±threshold sigma.

Returns `pd.DataFrame` (same index as input unless noted).

```python
panel = pd.DataFrame({"x": np.random.randn(100)})
out = mf.functions.zscore_outlier_clean(panel, threshold=3.0)
```

[Preprocessing reference](../preprocessing.md)

## Quick example

```python
import macroforecast as mf
import pandas as pd
import numpy as np

panel = pd.DataFrame({"x1": [None, 1.0, 2.0, 3.0], "x2": [1.0, 2.0, None, 4.0]})
clean = mf.functions.forward_fill_clean(panel)
no_outliers = mf.functions.winsorize_clean(clean, lower_quantile=0.01, upper_quantile=0.99)
```
