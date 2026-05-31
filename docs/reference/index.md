# Reference

Function-level reference for the clean public Python API.

| Stage | Page | Owns |
| --- | --- | --- |
| Meta | [Meta](meta.md) | Global package defaults. |
| Data | [Data](data.md) | Panels, metadata, loaders, targets, horizons, and sample windows. |
| Custom Extensions | [Custom Extensions](custom.md) | User-supplied datasets, preprocessing, features, models, stage policies, forecast combinations, diagnostics, interpretation, and artifacts. |
| Workflow | [Workflow Contract](workflow.md) | Which module owns callable functions, windows, and runner composition. |
| Preprocessing | [Preprocessing](preprocessing.md) | Frequency alignment, t-codes, outliers, imputation, and frame rules. |
| Feature Engineering | [Feature Engineering](feature_engineering.md) | Direct/path-average targets, MAF/MARX-style transforms, sparse components, lag/window predictors, and aligned `X`, `y`. |
| Feature Diagnostic | [Feature Diagnostic](feature_diagnostic.md) | Feature-matrix missingness, correlation, factor, lag, MARX, stage, and selection diagnostics. |
| Models | [Models](models.md) | Callable model fits for linear, tree, factor, and volatility models. |
| Window | [Window](window.md) | Estimation/validation/test windows shared across stages. |
| Selection | [Selection](selection.md) | Hyperparameter search specs and parameter selection. |
| Forecasting | [Forecasting](forecasting.md) | Windowed runner and forecast combination. |
| Forecast Diagnostic | [Forecast Diagnostic](forecast_diagnostic.md) | Forecast-vs-actual, residual, rolling-loss, tuning, coefficient, ensemble-weight, and stage-update diagnostics. |
| Metrics | [Metrics](metrics.md) | Scoring metrics, ranking, and metric resolution. |
| Tests | [Tests](tests.md) | Forecast-comparison tests and residual diagnostics. |
| Evaluation | [Evaluation Namespace](evaluation.md) | Thin wrapper exposing `metrics` and `tests`. |
| Interpretation | [Interpretation](interpretation.md) | Model-native importance and model-agnostic effect summaries. |
| Output | [Output](output.md) | Artifact writing and schema-aware provenance manifest. |
| Data Summary | [Data Summary](data_summary.md) | One-panel summary tables. |
| Data Analysis | [Data Analysis](data_analysis.md) | Before/after preprocessing comparison. |

```{toctree}
:maxdepth: 1

meta
data
custom
workflow
fred_md
fred_qd
fred_sd
fred_datasets
preprocessing
feature_engineering
feature_diagnostic
models
window
selection
forecasting
forecast_diagnostic
metrics
tests
evaluation
interpretation
output
data_summary
data_analysis
public_api
```
