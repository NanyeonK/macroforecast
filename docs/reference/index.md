# Reference

[Back to documentation home](../index.md)

Current workflow reference for the clean public Python API and callable module
boundaries. Dataset descriptions live in [FRED Datasets](../datasets/index.md).

Start with [Documentation Map](documentation_map.md) if you are deciding what
to inspect. Use this page when you already know the part of the workflow you
want to read.

## First Review Path

| Order | Page | Why it comes first |
| --- | --- | --- |
| 1 | [Documentation Map](documentation_map.md) | Shows which page answers which question. |
| 2 | [Workflow Contract](workflow.md) | Defines module ownership and runner composition. |
| 3 | [Legacy Callable Coverage](legacy_callable_coverage.md) | Confirms what old runtime functionality is covered, removed, or deferred. |
| 4 | [Public Python API](public_api.md) | Lists importable public symbols. |
| 5 | [Reference Verification](reference_verification.md) | Shows formula/reference anchors and future verification expansion. |

## Workflow Groups

| Group | Pages | Owns |
| --- | --- | --- |
| Orientation | [Documentation Map](documentation_map.md), [Workflow](workflow.md), [Legacy Coverage](legacy_callable_coverage.md), [Reference Verification](reference_verification.md), [Public API](public_api.md) | How the package is organized and what is currently covered. |
| Package Configuration | [Meta](meta.md) | Package-wide defaults used only when a direct function or runner policy does not pass a more specific value. |
| Data Pipeline | [Data](data.md), [Preprocessing](preprocessing.md), [Data Analysis](data_analysis.md) | Data loading, metadata, cleaning, summaries, and before/after checks. See [FRED Datasets](../datasets/index.md) for dataset pages. |
| Feature Pipeline | [Feature Engineering](feature_engineering.md), [Feature Analysis](feature_analysis.md) | Targets, predictors, transforms, learned adaptive smoothers, factor/selection tools, and feature diagnostics. |
| Forecast Pipeline | [Window](window.md), [Models](models.md), [Model Ensemble](model_ensemble.md), [Model Selection](model_selection.md), [Forecasting](forecasting.md), [Forecast Analysis](forecast_analysis.md) | Timing, model fits, fit-time model composition, parameter search, runner execution, and forecast diagnostics. |
| Evaluation and Testing | [Metrics](metrics.md), [Tests](tests.md), [Evaluation](evaluation.md) | Scores, statistical tests, benchmark comparisons, regimes, aggregation, and reports. |
| Explanation and Delivery | [Interpretation](interpretation.md), [Output](output.md), [Reporting](reporting.md) | Attribution, output generation, artifact writing, paper-table presets, and report/table rendering. |
| Custom Hooks | [Custom Extensions](custom/index.md) | User-provided datasets, preprocessing, features, models, policies, tests, interpretation, and artifacts. |

## Orientation

```{toctree}
:maxdepth: 1
:caption: Orientation

documentation_map
workflow
legacy_callable_coverage
reference_verification
public_api
```

## Custom Hooks

```{toctree}
:maxdepth: 1
:caption: Custom Hooks

custom/index
```

## Package Configuration

```{toctree}
:maxdepth: 1
:caption: Package Configuration

meta
```

## Data Pipeline

```{toctree}
:maxdepth: 1
:caption: Data Pipeline

data
preprocessing
data_analysis
```

## Feature Pipeline

```{toctree}
:maxdepth: 1
:caption: Feature Pipeline

feature_engineering
feature_analysis
```

## Forecast Pipeline

```{toctree}
:maxdepth: 1
:caption: Forecast Pipeline

window
models
model_ensemble
model_selection
forecasting
forecast_analysis
```

## Evaluation and Testing

```{toctree}
:maxdepth: 1
:caption: Evaluation and Testing

metrics
tests
evaluation
```

## Explanation and Delivery

```{toctree}
:maxdepth: 1
:caption: Explanation and Delivery

interpretation
output
reporting
```
