# User Guide

In-depth documentation for every macrocast layer. Read in order for the full picture, or jump to a specific topic.

## Recommended reading order

| # | Guide | What you will learn |
|---|-------|-------------------|
| 1 | [Stage 0: Study Grammar](stage0.md) | Fixed vs varying design, comparison contract |
| 2 | [Raw Data](raw.md) | FRED-MD/QD/SD loading, vintage management |
| 3 | [Recipes](recipes.md) | YAML recipe anatomy, RecipeSpec, RunSpec |
| 4 | [Preprocessing](preprocessing.md) | Governance fields, scaling, imputation, t-code |
| 5 | [Models](models.md) | All 24 model families — when to use each |
| 6 | [Registry](registry.md) | Choice-space management, support status |
| 7 | [Compiler](compiler.md) | Recipe validation, execution eligibility |
| 8 | [Execution](execution.md) | Runtime pipeline, frameworks, artifacts |
| 9 | [Tuning](tuning.md) | HP optimization: grid, random, Bayesian, genetic |
| 10 | [Statistical Tests](stat_tests.md) | 20 forecast comparison tests — when to use each |
| 11 | [Importance](importance.md) | 12 feature importance methods — when to use each |
| 12 | [Data/Task Axes](data_task_axes.md) | 7 data/task axes (Phase 3) — release lag, missing, variable universe, horizons, breaks, scale |
| 13 | [Preprocessing Separation](preprocessing_separation.md) | 5 leak-discipline modes (Phase 3) |
| 14 | [Benchmarks](benchmarks.md) | Phase 4 benchmark axes - benchmark_family/window/scope, relative metrics |

**See also:** [Getting Started](../getting_started/index.md) | [Examples](../examples/index.md) | [API Reference](../api/index.md)

```{toctree}
:hidden:
:maxdepth: 1

stage0
raw
recipes
sweep_recipes
controlled_variation_study
preprocessing
models
registry
compiler
execution
tuning
stat_tests
stat_test_selection
importance
data_task_axes
preprocessing_separation
benchmarks
ablation_cookbook
replication_cookbook
decomposition_tutorial
```
