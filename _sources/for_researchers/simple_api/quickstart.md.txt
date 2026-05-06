# Quickstart

Run one default macroeconomic forecasting experiment with explicit data, target, sample period, and horizons.

```python
import macroforecast as mf

result = mf.forecast(
    "fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)
```

The return value is an `ForecastResult` facade over saved artifacts:

```python
forecasts = result.forecasts
metrics = result.metrics
manifest = result.manifest
```

The default profile is `macroforecast-default-v1`. It uses a conservative baseline path:

- Layer 0 `study_scope = one_target_one_method`
- Layer 0 `failure_policy = fail_fast`
- Layer 0 `reproducibility_mode = seeded_reproducible` with seed `42`
- Layer 0 `compute_mode = serial`
- revised information set
- expanding-window point forecast
- `ar` model
- `zero_change` benchmark
- `mse` primary metric
- official FRED-MD/FRED-QD transformation codes when available
- no extra scaling, imputation, outlier handling, feature selection, or dimensionality reduction

Simple users normally choose only the first Layer 0 item, Study Scope. A single default call resolves to `one_target_one_method`; `Experiment.compare_models([...])` resolves to `one_target_compare_methods`. The failure, reproducibility, and compute policies above are written to the manifest but are not first-screen Simple decisions.

The sample period is required. `start` and `end` are part of the experiment definition, not optional runtime filters.

## Data Frequency

`fred_md` fixes frequency to monthly:

```python
mf.forecast("fred_md", target="INDPRO", start="1980-01", end="2019-12")
```

`fred_qd` fixes frequency to quarterly:

```python
mf.forecast("fred_qd", target="GDPC1", start="1980-01", end="2019-12")
```

`fred_sd` can be used alone only when `frequency` is supplied:

```python
mf.forecast(
    "fred_sd",
    target="UR_CA",
    start="1980-01",
    end="2019-12",
    frequency="monthly",
)
```

When FRED-SD is combined with FRED-MD or FRED-QD, the MD/QD dataset fixes the experiment frequency:

```python
mf.forecast("fred_md+fred_sd", target="INDPRO", start="1980-01", end="2019-12")
mf.forecast("fred_qd+fred_sd", target="GDPC1", start="1980-01", end="2019-12")
```

FRED-SD inferred/empirical transformation codes are off by default because FRED-SD does not publish official t-codes. Use `Experiment.use_sd_inferred_tcodes()` for the reviewed national-analog layer or `Experiment.use_sd_empirical_tcodes()` for empirical stationarity policies.
