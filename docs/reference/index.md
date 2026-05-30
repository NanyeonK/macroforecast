# Reference

Function-level reference for the clean public Python API.

| Stage | Page | Owns |
| --- | --- | --- |
| Meta | [Meta](meta.md) | Global package defaults. |
| Data | [Data](data.md) | Panels, metadata, loaders, targets, horizons, and sample windows. |
| Workflow | [Workflow Contract](workflow.md) | Which layer owns callable functions, windows, and runner composition. |
| Preprocessing | [Preprocessing](preprocessing.md) | Frequency alignment, t-codes, outliers, imputation, and frame rules. |
| Feature Engineering | [Feature Engineering](feature_engineering.md) | Direct/path-average targets, MAF/MARX-style transforms, lag/window predictors, and aligned `X`, `y`. |
| Models | [Models](models.md) | Callable model fits for linear, tree, factor, and volatility models. |
| Window | [Window](window.md) | Estimation/validation/test windows shared across stages. |
| Selection | [Selection](selection.md) | Hyperparameter search specs and parameter selection. |
| Forecasting | [Forecasting](forecasting.md) | Windowed runner and forecast combination. |
| Evaluation | [Evaluation](evaluation.md) | Scoring metrics. |
| Data Summary | [Data Summary](data_summary.md) | One-panel summary tables. |
| Data Analysis | [Data Analysis](data_analysis.md) | Before/after preprocessing comparison. |

```{toctree}
:maxdepth: 1

meta
data
workflow
fred_md
fred_qd
fred_sd
fred_datasets
preprocessing
feature_engineering
models
window
selection
forecasting
evaluation
data_summary
data_analysis
public_api
```
