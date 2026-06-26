# Concepts

[Back to User Guide](../index.md)

This section explains the core abstractions in `macroforecast`, one page per
workflow stage. Each page describes the abstraction in plain language, shows the
key public callables, and links to the corresponding reference documentation.

## Workflow Stages

::::{grid} 2

:::{grid-item-card} Data
:link: data
:link-type: doc

Loading FRED-MD, FRED-QD, FRED-SD, and custom panels into a `DataBundle`.
:::

:::{grid-item-card} Preprocessing
:link: preprocessing
:link-type: doc

Stationarity transforms, outlier rules, EM imputation, and standardization.
:::

:::{grid-item-card} Features
:link: features
:link-type: doc

F / X / MARX / MAF / Level feature families and the `FeatureSpec` abstraction.
:::

:::{grid-item-card} Models and Arms
:link: models_and_arms
:link-type: doc

`ModelSpec`, `Arm`, and how one arm becomes one contender in the evaluation.
:::

:::{grid-item-card} Windows
:link: windows
:link-type: doc

Expanding, rolling, and no-validation windows; retrain and retune cadence.
:::

:::{grid-item-card} Running
:link: running
:link-type: doc

`run` and `run_pipeline`: direct vs path-average forecast policy.
:::

:::{grid-item-card} Evaluation
:link: evaluation
:link-type: doc

RMSE, relative MSE, relative RMSE, DM/CW tests, and the Model Confidence Set.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

data
preprocessing
features
models_and_arms
windows
running
evaluation
```
