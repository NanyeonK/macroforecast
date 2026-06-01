# Reference

Function-level reference for the clean public Python API.

| Stage | Page | Owns |
| --- | --- | --- |
| Meta | [Meta](meta.md) | Global package defaults. |
| Data | [Data](data.md) | Panels, metadata, loaders, targets, horizons, and sample windows. |
| Custom Extensions | [Custom Extensions](custom.md) | User-supplied datasets, preprocessing, features, models, stage policies, forecast combinations, diagnostics, interpretation, and artifacts. |
| Workflow | [Workflow Contract](workflow.md) | Which module owns callable functions, windows, and runner composition. |
| Legacy Coverage | [Legacy Callable Coverage](legacy_callable_coverage.md) | Migration audit from old runtime/reference surfaces to current callable modules. |
| Reference Verification | [Reference Verification](reference_verification.md) | Formula/reference anchors and verification-suite expansion rules. |
| Preprocessing | [Preprocessing](preprocessing.md) | Frequency alignment, t-codes, outliers, imputation, and frame rules. |
| Feature Engineering | [Feature Engineering](feature_engineering.md) | Direct/path-average targets, MAF/MARX-style transforms, sparse components, lag/window predictors, and aligned `X`, `y`. |
| Feature Analysis | [Feature Analysis](feature_analysis.md) | Feature-matrix missingness, correlation, target-correlation, factor, lag, MARX, stage, and selection analysis. |
| Models | [Models](models.md) | Callable model fits for linear, tree, factor, and volatility models. |
| Window | [Window](window.md) | Estimation/validation/test windows shared across stages. |
| Selection | [Selection](selection.md) | Hyperparameter search specs and parameter selection. |
| Forecasting | [Forecasting](forecasting.md) | Windowed runner and forecast combination. |
| Forecast Analysis | [Forecast Analysis](forecast_analysis.md) | Forecast-vs-actual, scale, residual, rolling-loss, training-loss, tuning, coefficient, DFM, ensemble-weight, and stage-update analysis. |
| Metrics | [Metrics](metrics.md) | Scoring metrics, bias, ranking, and metric resolution. |
| Tests | [Tests](tests.md) | Forecast-comparison tests, interval/PIT diagnostics, and residual diagnostics. |
| Evaluation | [Evaluation Namespace](evaluation.md) | Reports, OOS filtering, error decomposition, and links to `metrics`/`tests`. |
| Interpretation | [Interpretation](interpretation.md) | Model-native importance, model-agnostic effects, attribution, and VAR interpretation. |
| Output | [Output](output.md) | Output table generation, artifact writing, and schema-aware provenance manifest. |
| Reporting | [Reporting](reporting.md) | Paper/report table formatting, LaTeX/HTML/Markdown rendering, and figure-ready data. |
| Data Summary | [Data Summary](data_summary.md) | One-panel summary tables. |
| Data Analysis | [Data Analysis](data_analysis.md) | Before/after preprocessing comparison. |

```{toctree}
:maxdepth: 1

meta
data
custom
workflow
legacy_callable_coverage
reference_verification
fred_md
fred_qd
fred_sd
fred_datasets
preprocessing
feature_engineering
feature_analysis
models
window
selection
forecasting
forecast_analysis
metrics
tests
evaluation
interpretation
output
reporting
data_summary
data_analysis
public_api
```
