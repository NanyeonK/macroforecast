# Results

The `macrocast.pipeline.results` module provides two data structures: `ForecastRecord`, which stores the output of a single (model, horizon, date) evaluation step, and `ResultSet`, which accumulates records and provides serialisation and summary methods.

---

## ForecastRecord

Each pseudo-OOS evaluation step produces one `ForecastRecord`. The dataclass has 16 fields.

| Field | Type | Description |
|-------|------|-------------|
| `experiment_id` | `str` | UUID of the parent `ForecastExperiment` run |
| `model_id` | `str` | Human-readable model name, e.g. `"krr__factors__KFoldCV(k=5)__l2"` |
| `nonlinearity` | `Nonlinearity` | Nonlinearity component |
| `regularization` | `Regularization` | Regularization component |
| `cv_scheme` | `CVSchemeType` | CV scheme used for HP selection |
| `loss_function` | `LossFunction` | Loss function optimised during training |
| `window` | `Window` | Outer evaluation window strategy |
| `horizon` | `int` | Forecast horizon h |
| `train_end` | `pd.Timestamp` | Last date in the training window |
| `forecast_date` | `pd.Timestamp` | Date being forecast (t*) |
| `y_hat` | `float` | Point forecast |
| `y_true` | `float` | Realised value |
| `n_train` | `int` | Number of training observations |
| `n_factors` | `int or None` | Number of PCA factors used (None if AR-only mode) |
| `n_lags` | `int` | Number of AR lags used |
| `hp_selected` | `dict` | Best hyperparameter values selected by CV |

**Derived properties:**

- `.error` — forecast error: y_true - y_hat
- `.squared_error` — squared forecast error: (y_true - y_hat)²

---

## ResultSet

`ResultSet` accumulates `ForecastRecord` objects and exposes methods for inspection, serialisation, and summary statistics.

### Accumulation

```python
results.add(record)          # add a single ForecastRecord
results.extend(records)      # add an iterable of ForecastRecords
```

### Conversion

**`to_dataframe()`** converts all records to a tidy pandas DataFrame. Enum values are serialised as strings (e.g. `"KRR"`, `"FACTORS"`) for compatibility with R and arrow/parquet interchange.

**`to_parquet(path)`** writes the DataFrame to a parquet file. Parent directories are created automatically. Returns the resolved path.

**`from_parquet(path)`** reads a parquet file and returns a new `ResultSet` with the DataFrame loaded into `_df_cache`. This is the standard way to reload results produced by macrocastR.

**`to_dataframe_cached()`** returns the cached DataFrame if available, otherwise calls `to_dataframe()`. Useful after `from_parquet`.

### Summary

**`msfe_by_model(horizon=None)`** computes mean squared forecast error per model. Returns a DataFrame with columns `[model_id, horizon, msfe, n_obs]`. If `horizon` is provided, filters to that horizon only.

---

## Code Examples

```python
# Inspect columns
df = results.to_dataframe()
print(df.columns.tolist())
# ['experiment_id', 'model_id', 'nonlinearity', 'regularization',
#  'cv_scheme', 'loss_function', 'window', 'horizon', 'train_end',
#  'forecast_date', 'y_hat', 'y_true', 'n_train', 'n_factors',
#  'n_lags', 'hp_selected']

# Save to disk
path = results.to_parquet("~/.macrocast/results/exp1.parquet")

# Load back
rs2 = ResultSet.from_parquet(path)
df2 = rs2.to_dataframe_cached()

# Quick MSFE summary at h=1
print(results.msfe_by_model(horizon=1))
```

---

## Downstream Usage

Pass `results.to_dataframe()` to any function in the [Evaluation Layer](../evaluation/index.md). The evaluation layer expects a tidy DataFrame with the column layout produced by `to_dataframe()`. Relative MSFE, MCS, and the CLSS 2022 decomposition regression are all computed from this format.
