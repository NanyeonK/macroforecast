# macrocast

**Decomposing ML Forecast Gains in Macroeconomic Forecasting**

macrocast is an open-source Python framework for systematic evaluation of machine learning methods in macroeconomic forecasting. It provides unified access to the FRED-MD, FRED-QD, and FRED-SD database ecosystem and implements the four-component decomposition framework of Coulombe et al. (2020, JBES).

---

## Current Status

| Layer | Status | Description |
|-------|--------|-------------|
| **Data (v0.1)** | Complete | FRED-MD, FRED-QD, FRED-SD loaders, MacroFrame, transformations, missing value handling, vintage management |
| **Pipeline (v0.2)** | Complete | ForecastExperiment, four-component decomposition, model zoo (Python + R) |
| **Evaluation (v0.3)** | Complete | MSFE, MCS, regime-conditional evaluation, decomposition tables |

---

## Installation

```bash
pip install macrocast
```

With optional extras:

```bash
pip install macrocast[ml]    # LightGBM + PyTorch
pip install macrocast[viz]   # matplotlib + seaborn
pip install macrocast[all]   # all extras
```

For development:

```bash
git clone https://github.com/macrocast/macrocast.git
cd macrocast
uv sync --all-extras
```

---

## Quick Example

### Data Layer

```python
import macrocast as mc

# Load and transform FRED-MD (latest vintage)
md = mc.load_fred_md()
md_t = md.transform()

# Inspect
print(md_t)
# MacroFrame(dataset='FRED-MD', vintage='current', T=790, N=128,
#            period=1959-01-01 to 2024-10-01, status=transformed)

# Subset to output and income variables
output = md_t.group("output_income")

# Check missing values
report = md_t.missing_report()
print(report[["n_leading", "n_trailing", "n_intermittent"]].head())
```

### Full Pipeline Example

```python
import macrocast as mc
from macrocast.pipeline import (
    ForecastExperiment, ModelSpec, FeatureSpec,
    Nonlinearity, Regularization, CVScheme, LossFunction, Window,
    KRRModel,
)
from macrocast.evaluation import relative_msfe, decompose_treatment_effects

# 1. Load and prepare data
md = mc.load_fred_md().trim(start="1970-01", end="2023-12").transform()
panel = md.data
target = panel["INDPRO"]

# 2. Define model specs
specs = [
    ModelSpec(
        model_cls=KRRModel,
        regularization=Regularization.FACTORS,
        cv_scheme=CVScheme.KFOLD(5),
        loss_function=LossFunction.L2,
    ),
]

# 3. Run the experiment
exp = ForecastExperiment(
    panel=panel, target=target,
    horizons=[1, 3, 12],
    model_specs=specs,
    feature_spec=FeatureSpec(n_factors=8, n_lags=4),
    window=Window.EXPANDING,
    n_jobs=-1,
)
results = exp.run()

# 4. Evaluate
df = results.to_dataframe()
ar_hat = df.loc[df["model_id"].str.startswith("linear"), "y_hat"].values
krr_hat = df.loc[df["model_id"].str.startswith("krr"), "y_hat"].values
y_true = df["y_true"].values[:len(krr_hat)]
print("Relative MSFE:", relative_msfe(y_true, krr_hat, ar_hat))

# 5. Decompose gains
decomp = decompose_treatment_effects(df)
print(decomp.summary_df)
```

See the [Pipeline Layer](pipeline/index.md) and [Evaluation Layer](evaluation/index.md) sections for full documentation.

---

## Design Principles

macrocast is built around three principles:

**Decomposability first.** Every design decision serves the goal of isolating individual sources of forecast improvement. The pipeline is not optimized for raw predictive accuracy but for transparent attribution of performance gains.

**FRED-native.** The data layer treats FRED-MD, FRED-QD, and FRED-SD as first-class citizens. Transformation codes, vintage structure, and variable groupings are built into the schema.

**Minimal core, extensible surface.** The package ships with a small set of well-tested models. External models enter through a standard scikit-learn-compatible interface.

