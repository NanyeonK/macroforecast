# Documentation Map

[Back to reference](index.md)

Use this page as the inspection order for the current callable-first
`macroforecast` documentation.

## Quick Routing

| Question | Open this page | What to verify there |
| --- | --- | --- |
| What is the current package shape? | [Workflow Contract](workflow.md) | Module ownership, runner loop, and stage policies. |
| Did we preserve old statistical functionality? | [Legacy Callable Coverage](legacy_callable_coverage.md) | Covered callables, intentional removals, and remaining future work. |
| Which imports are public? | [Public Python API](public_api.md) | Top-level exports and module namespaces. |
| How do I load and shape data? | [Data](data.md) | Panel contract, metadata, custom loaders, FRED loaders, frequency policies. |
| Which FRED-family dataset should I use? | [FRED Datasets](../datasets/index.md) | FRED-MD, FRED-QD, FRED-SD, combined loaders, and frequency conversion warnings. |
| How do I clean a panel? | [Preprocessing](preprocessing.md) | Transform codes, outliers, imputation, standardization, frame rules. |
| How do I create targets and predictors? | [Feature Engineering](feature_engineering.md) | Target construction, lags, rolling features, factors, selection, runner-safe specs. |
| How do I define time windows? | [Window](window.md) | Train/validation/test windows, expanding/rolling/fixed policies, stage policies. |
| Which models are available? | [Models](models.md) | Model groups, parameter defaults, optional dependencies, model-owned search spaces. |
| How does the runner combine everything? | [Forecasting](forecasting.md) | Runner inputs, direct/recursive/path-average forecasts, combinations. |
| How do I score and test forecasts? | [Evaluation](evaluation.md) | Evaluation reports, metrics/tests split, benchmark/regime/decomposition tables. |
| How do I inspect model behavior? | [Interpretation](interpretation.md) | Importance, SHAP, attribution, VAR interpretation, neural attribution. |
| How do I save outputs? | [Output](output.md) | Output bundles, artifact writing, manifests, hashes, compression. |
| How do I format paper/report tables? | [Reporting](reporting.md) | Table formatting, LaTeX/HTML/Markdown rendering, figure-ready data. |
| How do I know the formulas are not drifting? | [Reference Verification](reference_verification.md) | Reference anchors and expansion policy. |

## Recommended Review Sequence

Use this sequence when auditing the whole docs site.

1. [Workflow Contract](workflow.md)
2. [Legacy Callable Coverage](legacy_callable_coverage.md)
3. [Public Python API](public_api.md)
4. [Data](data.md)
5. [FRED Datasets](../datasets/index.md)
6. [Preprocessing](preprocessing.md)
7. [Feature Engineering](feature_engineering.md)
8. [Window](window.md)
9. [Models](models.md)
10. [Selection](selection.md)
11. [Forecasting](forecasting.md)
12. [Metrics](metrics.md)
13. [Tests](tests.md)
14. [Evaluation](evaluation.md)
15. [Feature Analysis](feature_analysis.md)
16. [Forecast Analysis](forecast_analysis.md)
17. [Interpretation](interpretation.md)
18. [Output](output.md)
19. [Reporting](reporting.md)
20. [Reference Verification](reference_verification.md)

## Module Boundaries

| Boundary | Rule |
| --- | --- |
| `data` vs `preprocessing` | `data` creates canonical panels and metadata; `preprocessing` transforms values. |
| `preprocessing` vs `feature_engineering` | `preprocessing` cleans variables; `feature_engineering` creates targets and predictors. |
| `feature_engineering` vs `window` | Feature functions build matrices; `window` decides which dates belong to train/validation/test. |
| `models` vs `selection` | Models own fit functions and search spaces; `selection` runs parameter search on supplied windows. |
| `selection` vs `forecasting` | `selection` picks parameters; `forecasting` orchestrates repeated fits and predictions. |
| `metrics` vs `tests` vs `evaluation` | `metrics` score errors; `tests` run statistical tests; `evaluation` assembles reports and slices. |
| `output` vs `reporting` | `output` creates/writes artifacts; `reporting` formats presentation tables and figure data. |
| callable API vs future recipes | Current docs describe direct Python callables. YAML/recipe wrappers are intentionally deferred. |

## Current Review Focus

| Page | Why it matters now |
| --- | --- |
| [Legacy Callable Coverage](legacy_callable_coverage.md) | Confirms that intentional removals are not mistaken for missing work. |
| [Reference Verification](reference_verification.md) | Tracks the verification suite that should grow as paper-code checks are added. |
| [Output](output.md) and [Reporting](reporting.md) | These were recently split; check whether their responsibility boundary is clear. |
| [Tests](tests.md) | Contains `blocked_oob_reality_check()` and `iterative_model_confidence_set()`, which close recent legacy gaps. |
