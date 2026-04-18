# Getting Started

Welcome to macrocast. This guide will take you from zero to running your first forecasting study.

## Learning path

| Step | Time | What you will learn |
|------|------|-------------------|
| 1. [Quickstart](quickstart.md) | 5 min | Run your first forecast in 10 lines of Python |
| 2. [Your First Study](first_study.md) | 20 min | Design a complete Ridge vs Lasso comparison study |
| 3. [Understanding Output](understanding_output.md) | 10 min | Read predictions, metrics, and statistical test results |

## Prerequisites

- Python 3.10+ with macrocast installed ([Installation](../install.md))
- Basic familiarity with pandas DataFrames
- Basic knowledge of macroeconomic forecasting (what FRED-MD is, what out-of-sample evaluation means)

## Key concepts (30-second version)

- **Recipe**: A YAML file that fully specifies one forecasting study — data, preprocessing, model, benchmark, metrics
- **Design (Stage 0)**: The study grammar that defines what is fixed (for fair comparison) and what varies (the research question). See ``macrocast.design``.
- **Compiler**: Validates your recipe and determines if it can execute with the current runtime
- **Execution**: Runs the forecast, writes predictions, metrics, and provenance artifacts

**See also:** [User Guide](../user_guide/index.md) for in-depth documentation of every layer

```{toctree}
:hidden:
:maxdepth: 1

horse_race_quickstart
quickstart
first_study
understanding_output
```
