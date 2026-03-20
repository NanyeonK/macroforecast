# Getting Started

This page covers installation and walkthroughs of the data, pipeline, and evaluation layers.

---

## Installation

**From PyPI:**

```bash
pip install macrocast
```

**From source (recommended for development):**

```bash
git clone https://github.com/macrocast/macrocast.git
cd macrocast
uv sync --all-extras
```

**Python version:** 3.10 or later.

---

---

## Data Layer Walkthrough

### Loading FRED-MD

```python
import macrocast as mc

# Load the latest vintage (cached under ~/.macrocast/cache/fred_md/)
md = mc.load_fred_md()
print(md)
# MacroFrame(dataset='FRED-MD', vintage='current', T=790, N=128, ...)

# Load a specific vintage
md_2020 = mc.load_fred_md(vintage="2020-01")

# Trim sample and transform in one call
md_t = mc.load_fred_md(
    start="1970-01",
    end="2023-12",
    transform=True,
)
```

### Inspecting the data

```python
# Access the underlying DataFrame
df = md.data
print(df.shape)          # (T, N)
print(df.index[:3])      # DatetimeIndex

# Transformation codes (1-7, McCracken & Ng 2016)
print(md.tcodes["INDPRO"])   # 5 = log-difference

# Variable metadata
vmeta = md.metadata.variables["INDPRO"]
print(vmeta.group)            # 'output_income'
print(vmeta.description)      # 'Industrial Production Index'
```

### Applying transformations

Transformations convert level series to approximately stationary series following the McCracken-Ng (2016) codes.

```python
# Apply default tcodes from the spec
md_t = md.transform()

# Override a specific variable's tcode
md_t = md.transform(override={"INDPRO": 5, "UNRATE": 2})
```

### Subsetting by variable group

```python
# Available groups: output_income, labor, housing, prices,
#                   money_credit, interest_rates, stock_market, other
output = md.group("output_income")
labor  = md.group("labor")
```

### Missing value handling

```python
# View the missing value report
report = md.missing_report()
print(report[["n_leading", "n_trailing", "n_intermittent", "pct_missing"]])

# Advance start date to eliminate leading NaNs (standard FRED-MD approach)
md_clean = md.handle_missing("trim_start")

# Interpolate interior gaps
md_interp = md.handle_missing("interpolate")

# Drop variables with more than 30% missing
md_drop = md.handle_missing("drop_vars", max_missing_pct=0.3)
```

### Trimming the sample

```python
md_sub = md.trim(start="1970-01", end="2019-12")

# Drop sparse variables during trimming
md_sub = md.trim(start="1970-01", min_obs_pct=0.9)
```

### Method chaining

MacroFrame methods return new objects, so they can be chained:

```python
md_ready = (
    mc.load_fred_md()
    .trim(start="1970-01", end="2023-12")
    .handle_missing("trim_start")
    .transform()
)
```

### FRED-QD (quarterly)

```python
qd = mc.load_fred_qd()
qd_t = qd.transform()
```

### FRED-SD (state-level)

```python
# Load unemployment rates for California and Texas
sd = mc.load_fred_sd(states=["CA", "TX"], variables=["UR"])
print(sd.data.shape)
```

### Vintage management

```python
# List available vintage identifiers (no network call)
vintages = mc.list_available_vintages("fred_md", start="2010-01", end="2020-12")

# Load multiple vintages
panel = mc.load_vintage_panel("fred_md", vintages=["2019-01", "2020-01"])
rt = mc.RealTimePanel(panel)
print(rt)
# RealTimePanel(n_vintages=2, range=2019-01 to 2020-01)
```

---

---

## Running a Forecast Experiment

After preparing a `MacroFrame`, pass the transformed panel to `ForecastExperiment`.

```python
from macrocast.pipeline import (
    ForecastExperiment, ModelSpec, FeatureSpec,
    Regularization, CVScheme, LossFunction, Window, KRRModel,
)

# ModelSpec binds a model class to its four component labels
spec = ModelSpec(
    model_cls=KRRModel,
    regularization=Regularization.FACTORS,
    cv_scheme=CVScheme.KFOLD(5),
    loss_function=LossFunction.L2,
)

# ForecastExperiment orchestrates the outer pseudo-OOS loop
exp = ForecastExperiment(
    panel=md_t.data,
    target=md_t.data["INDPRO"],
    horizons=[1, 3, 12],
    model_specs=[spec],
    feature_spec=FeatureSpec(n_factors=8, n_lags=4),
    window=Window.EXPANDING,
    n_jobs=-1,
)

results = exp.run()
print(results)
# ResultSet(experiment_id='...', n_records=...)
```

The experiment uses direct h-step-ahead forecasting: a separate model is trained for each horizon. The outer loop expands the training window by one period at each evaluation date, consistent with CLSS 2022.

Results can be exported:

```python
# Export to parquet for downstream R analysis
results.to_parquet("~/.macrocast/results/run1.parquet")

# Or work with the tidy DataFrame directly
df = results.to_dataframe()
print(df.columns.tolist())
# ['experiment_id', 'model_id', 'nonlinearity', 'regularization',
#  'cv_scheme', 'loss_function', 'window', 'horizon', 'train_end',
#  'forecast_date', 'y_hat', 'y_true', 'n_train', 'n_factors', 'n_lags']
```

See [ForecastExperiment](pipeline/experiment.md) for the full parameter reference.

---

## Evaluating Results

```python
from macrocast.evaluation import (
    relative_msfe, decompose_treatment_effects, mcs,
)
import numpy as np

df = results.to_dataframe()

# Separate AR benchmark from KRR
ar = df[df["model_id"].str.startswith("linear__none__bic")]
krr = df[df["model_id"].str.startswith("krr")]

# Relative MSFE (< 1 means improvement over AR)
rel = relative_msfe(
    y_true=krr["y_true"].values,
    y_hat_model=krr["y_hat"].values,
    y_hat_benchmark=ar["y_hat"].values,
)
print(f"Relative MSFE: {rel:.3f}")

# Four-component treatment effect decomposition
decomp = decompose_treatment_effects(df)
print(decomp.summary_df)
#      component      coef  se_hc3  t_stat
# 0    intercept   -0.02     ...     ...
# 1  d_nonlinear    0.05     ...     ...
# 2  d_data_rich    0.08     ...     ...
# 3      d_kfold    0.02     ...     ...
# 4        d_l2    0.01     ...     ...

# Model Confidence Set at 10% level
df["squared_error"] = (df["y_true"] - df["y_hat"]) ** 2
mcs_result = mcs(df[["model_id", "forecast_date", "squared_error"]])
print("MCS members:", mcs_result.included)
```

See [Evaluation Layer](evaluation/index.md) for the full reference.

---

## Next Steps

- [Data Layer overview](data/index.md) — FRED-MD vs FRED-QD vs FRED-SD comparison
- [MacroFrame](data/macroframe.md) — core container API
- [Transformations](data/transforms.md) — McCracken-Ng tcode reference
- [Pipeline Layer](pipeline/index.md) — components, model zoo, experiment orchestration
- [Evaluation Layer](evaluation/index.md) — metrics, decomposition, MCS, regime analysis
