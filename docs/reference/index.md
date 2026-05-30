# Reference

Function-level reference for the clean public Python API.

| Stage | Page | Owns |
| --- | --- | --- |
| Meta | [Meta](meta.md) | Global package defaults. |
| Data | [Data](data.md) | Panels, metadata, loaders, targets, horizons, and sample windows. |
| Preprocessing | [Preprocessing](preprocessing.md) | Frequency alignment, t-codes, outliers, imputation, and frame rules. |
| Feature Engineering | [Feature Engineering](feature_engineering.md) | Direct/path-average targets, MAF/MARX-style transforms, lag/window predictors, and aligned `X`, `y`. |
| Models | [Models](models.md) | Callable model fits for linear, tree, factor, and volatility models. |
| Window | [Window](window.md) | Temporal train/validation windows shared across stages. |
| Selection | [Selection](selection.md) | Hyperparameter search specs and parameter selection. |
| Evaluation | [Evaluation](evaluation.md) | Scoring metrics. |
| Data Summary | [Data Summary](data_summary.md) | One-panel summary tables. |
| Data Analysis | [Data Analysis](data_analysis.md) | Before/after preprocessing comparison. |

```{toctree}
:maxdepth: 1

meta
data
fred_md
fred_qd
fred_sd
fred_datasets
preprocessing
feature_engineering
models
window
selection
evaluation
data_summary
data_analysis
public_api
```
