# Documentation Map

[Back to reference](index.md)

Use this page as the inspection order for the current callable-first
`macroforecast` documentation.

## Quick Routing

| Question | Open this page | What to verify there |
| --- | --- | --- |
| What is the current package shape? | [Workflow Contract](workflow.md) | Module ownership, runner loop, and stage policies. |
| How do I set package-wide defaults? | [Meta](meta.md) | Seed, worker count, error handling, default runner stage scopes, and metadata level. |
| Did we preserve old statistical functionality? | [Legacy Callable Coverage](legacy_callable_coverage.md) | Covered callables, intentional removals, and remaining future work. |
| Which imports are public? | [Public Python API](public_api.md) | Top-level exports and module namespaces. |
| How do I load and shape data? | [Data](data.md) | Panel contract, metadata, custom loaders, FRED loaders, frequency policies. |
| Which FRED-family dataset should I use? | [FRED Datasets](../datasets/index.md) | FRED-MD, FRED-QD, FRED-SD, combined loaders, and frequency conversion warnings. |
| How do I clean a panel? | [Preprocessing](preprocessing.md) | Transform codes, outliers, imputation, standardization, frame rules. |
| How do I smooth a noisy macro series for monitoring? | [Feature Engineering](feature_engineering.md) | AlbaMA, one-sided/two-sided adaptive moving-average features, and learned weights. |
| How do I inspect learned feature weights? | [Feature Analysis](feature_analysis.md) | Effective windows and recent weight shares from adaptive feature weight matrices. |
| How do I create targets and predictors? | [Feature Engineering](feature_engineering.md) | Target construction, lags, rolling features, factors, selection, runner-safe specs. |
| How do I define time windows? | [Window](window.md) | Train/validation/test windows, expanding/rolling/fixed policies, stage policies. |
| Which models are available? | [Models](models.md) | Model groups, parameter defaults, optional dependencies, model-owned search spaces. |
| How do I fit several member models as one model? | [Model Ensemble](model_ensemble.md) | Bagging, subagging, random subspace, stacking, Super Learner, and Booging. |
| How does the runner combine everything? | [Forecasting](forecasting.md) | Runner inputs, direct/recursive/path-average forecasts, forecast-output combinations. |
| How do I score and test forecasts? | [Evaluation](evaluation.md) | Evaluation reports, metrics/tests split, benchmark/regime/decomposition tables. |
| How do I inspect model behavior? | [Interpretation](interpretation.md) | Importance, SHAP, attribution, VAR interpretation, neural attribution. |
| How do I save outputs? | [Output](output.md) | Output bundles, artifact writing, manifests, hashes, compression. |
| How do I format paper/report tables? | [Reporting](reporting.md) | Accuracy/model-comparison/test presets, table formatting, LaTeX/HTML/Markdown rendering, figure-ready data. |
| How do I plug in my own loader, transform, model, test, diagnostic, or artifact? | [Custom Extensions](custom/index.md) | Stage-local custom callable hooks and input/output contracts. |
| How do I know the formulas are not drifting? | [Reference Verification](reference_verification.md) | Reference anchors and expansion policy. |

## Recommended Review Sequence

Use this sequence when auditing the whole docs site.

1. [Workflow Contract](workflow.md)
2. [Meta](meta.md)
3. [Legacy Callable Coverage](legacy_callable_coverage.md)
4. [Public Python API](public_api.md)
5. [Data](data.md)
6. [FRED Datasets](../datasets/index.md)
7. [Preprocessing](preprocessing.md)
8. [Feature Engineering](feature_engineering.md)
9. [Window](window.md)
10. [Models](models.md)
11. [Model Ensemble](model_ensemble.md)
12. [Model Selection](model_selection.md)
13. [Forecasting](forecasting.md)
14. [Metrics](metrics.md)
15. [Tests](tests.md)
16. [Evaluation](evaluation.md)
17. [Feature Analysis](feature_analysis.md)
18. [Forecast Analysis](forecast_analysis.md)
19. [Interpretation](interpretation.md)
20. [Output](output.md)
21. [Reporting](reporting.md)
22. [Custom Extensions](custom/index.md)
23. [Reference Verification](reference_verification.md)

## Reference Page Format

Reference pages should use the same contract-first structure when a page
documents callable functions. Exact sections can vary by module, but each
public callable should make these items easy to find:

| Section | Required content |
| --- | --- |
| `Purpose` | What the module owns and what it explicitly does not own. |
| `Public Functions` | Function list grouped by role, with one-line outputs and purpose. |
| `Public Flow` | Minimal executable call sequence when the module has a normal workflow. |
| Function `Signature` | Fully qualified callable name, arguments, defaults, and return type. |
| Function `Input` | Parameter name, type, default, allowed values, and meaning. |
| Function `Defaults` | Defaults that matter for reproducibility, especially hidden constants or metadata behavior. |
| Function `Output` | Return object, fields, table columns, and serialization helpers. |
| `Metadata` | Stage key and stored provenance when the function writes metadata. |
| `Validation` or notes | Error conditions and boundary cases where they are non-obvious. |

Use display labels for user-facing choices when possible, and put stored enum
values in code formatting. Avoid bare lists such as `"raise" | "continue"`
without explaining the meaning of each choice.

## Module Boundaries

| Boundary | Rule |
| --- | --- |
| `meta` vs data pipeline | `meta` stores package defaults; it does not load, clean, summarize, or compare data. |
| `data` vs `preprocessing` | `data` creates canonical panels and metadata; `preprocessing` transforms values. |
| `preprocessing` vs `feature_engineering` | `preprocessing` cleans variables; `feature_engineering` creates targets and predictors. |
| feature generation vs feature analysis | `feature_engineering` creates AlbaMA smoothed features and stores learned weights; `feature_analysis` summarizes those weights through effective windows and recent weight shares. |
| `feature_engineering` vs `window` | Feature functions build matrices; `window` decides which dates belong to train/validation/test. |
| `models` vs `model_selection` | Models own fit functions and search spaces; `model_selection` runs parameter search on supplied windows. |
| `model_selection` vs `forecasting` | `model_selection` picks model parameters; `forecasting` orchestrates repeated fits and predictions. |
| `metrics` vs `tests` vs `evaluation` | `metrics` score errors; `tests` run statistical tests; `evaluation` assembles reports and slices. |
| `output` vs `reporting` | `output` creates/writes artifacts; `reporting` formats presentation tables and figure data. |
| built-in stages vs custom extensions | Custom callables stay inside the owning stage and must return the same object shape that the next stage expects. |
| callable API vs future recipes | Current docs describe direct Python callables. YAML/recipe wrappers are intentionally deferred. |

## Current Review Focus

| Page | Why it matters now |
| --- | --- |
| [Legacy Callable Coverage](legacy_callable_coverage.md) | Confirms that intentional removals are not mistaken for missing work. |
| [Reference Verification](reference_verification.md) | Tracks the verification suite that should grow as paper-code checks are added. |
| [Output](output.md) and [Reporting](reporting.md) | These were recently split; check whether their responsibility boundary is clear. |
| [Custom Extensions](custom/index.md) | Confirms that custom hooks enter the normal callable flow instead of creating a parallel registry. |
| [Tests](tests.md) | Contains `blocked_oob_reality_check()` and exact `model_confidence_set()`, which close recent legacy gaps. |
