# Reference

[Back to documentation home](../index.md)

Current workflow reference for the live public Python API. These pages are generated from importable package surfaces, model registry metadata, signatures, and docstrings.

Start with [Documentation Map](documentation_map.md) when deciding what to inspect, or [Public Python API](public_api.md) when checking top-level imports.

## First Review Path

| Order | Page | Why it comes first |
| --- | --- | --- |
| 1 | [Documentation Map](documentation_map.md) | Shows which page answers which question. |
| 2 | [Workflow Contract](workflow.md) | Defines current module ownership and runner composition. |
| 3 | [Public Python API](public_api.md) | Lists importable public symbols from the live package. |
| 4 | [Reference Verification](reference_verification.md) | Records the generation source and coverage counts. |
| 5 | [Custom Extensions](custom/index.md) | Shows where user-owned data, models, tests, and outputs plug in. |

## Workflow Groups

| Group | Pages |
| --- | --- |
| Orientation | [Documentation Map](documentation_map.md), [Workflow](workflow.md), [Legacy Callable Coverage](legacy_callable_coverage.md), [Reference Verification](reference_verification.md), [Public Api](public_api.md) |
| Package Configuration | [Package Configuration](meta.md) |
| Data Pipeline | [Data](data.md), [Preprocessing](preprocessing.md), [Data Analysis](data_analysis.md) |
| Feature Pipeline | [Filters](filters.md), [Feature Engineering](feature_engineering.md), [Feature Analysis](feature_analysis.md) |
| Forecast Pipeline | [Window](window.md), [Models](models.md), [Model Ensemble](model_ensemble.md), [Model Selection](model_selection.md), [Forecasting](forecasting.md), [Forecast Analysis](forecast_analysis.md) |
| Evaluation And Testing | [Metrics](metrics.md), [Tests](tests.md), [Evaluation](evaluation.md) |
| Explanation And Delivery | [Interpretation](interpretation.md), [Dual Interpretation](interpretation_dual.md), [Output](output.md), [Reporting](reporting.md), [Pipeline](pipeline.md) |
| Custom Hooks | [Custom Extensions](custom/index.md) |

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

filters
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

## Evaluation And Testing

```{toctree}
:maxdepth: 1
:caption: Evaluation And Testing

metrics
tests
evaluation
```

## Explanation And Delivery

```{toctree}
:maxdepth: 1
:caption: Explanation And Delivery

interpretation
interpretation_dual
output
reporting
pipeline
```

## Custom Hooks

```{toctree}
:maxdepth: 1
:caption: Custom Hooks

custom/index
```
