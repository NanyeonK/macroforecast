# macroforecast

Pandas-first macro forecasting workflow tools.

The current documentation is organized around the callable-first package
structure. Start with the map below before opening individual API pages.

## What To Check First

| If you want to check... | Open first | Then open |
| --- | --- | --- |
| Current package structure | [Documentation Map](reference/documentation_map.md) | [Workflow Contract](reference/workflow.md), [Public Python API](reference/public_api.md) |
| Package-wide defaults | [Meta](reference/meta.md) | [Forecasting](reference/forecasting.md) for runner policy usage. |
| Whether old runtime features are covered | [Legacy Callable Coverage](reference/legacy_callable_coverage.md) | [Reference Verification](reference/reference_verification.md) |
| Data and preprocessing behavior | [Data](reference/data.md) | [FRED Datasets](datasets/index.md), [Preprocessing](reference/preprocessing.md) |
| Paper replication settings | [Replication](replication/index.md) | [Reference Verification](reference/reference_verification.md), [Forecasting](reference/forecasting.md) |
| Forecast runner behavior | [Workflow Contract](reference/workflow.md) | [Window](reference/window.md), [Forecasting](reference/forecasting.md) |
| Models and parameter search | [Models](reference/models.md) | [Model Selection](reference/model_selection.md) |
| Evaluation and tests | [Evaluation](reference/evaluation.md) | [Metrics](reference/metrics.md), [Tests](reference/tests.md) |
| Interpretation, output, and reporting | [Interpretation](reference/interpretation.md) | [Output](reference/output.md), [Reporting](reference/reporting.md) |

## Main Sections

| Section | Purpose |
| --- | --- |
| [Documentation Map](reference/documentation_map.md) | One-page routing guide for what to inspect. |
| [FRED Datasets](datasets/index.md) | Dataset guide for FRED-MD, FRED-QD, FRED-SD, and combined loaders. |
| [Replication](replication/index.md) | Paper-level reconstructed settings and notebook-style replication guides. |
| [Reference](reference/index.md) | Current workflow reference pages grouped by package responsibility. |
| [Legacy Callable Coverage](reference/legacy_callable_coverage.md) | Migration audit from old runtime surfaces to current callable modules. |
| [Reference Verification](reference/reference_verification.md) | Formula/reference anchors and verification-suite expansion rules. |

```{toctree}
:maxdepth: 2

datasets/index
replication/index
reference/index
```
