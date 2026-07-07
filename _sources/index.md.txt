# Welcome to macroforecast

A pandas-first framework for reproducible macroeconomic forecasting workflows.

macroforecast turns a forecasting study into one declarative specification and runs it as a leak-aware pseudo-out-of-sample (POOS) experiment.

- **A broad set of data transformations.** Many ways to turn raw macro series into model inputs.
- **A wide range of forecasting models.** From simple benchmarks to modern machine learning.
- **Many evaluation tests and interpretation tools.** Compare forecasts and understand what drives them.
- **Reproducible paper replications.** Rebuild published studies from start to finish.

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: guide/getting_started
:link-type: doc

Minimal end-to-end pipeline in a few lines.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` User Guide
:link: guide/index
:link-type: doc

Concept pages for every workflow stage.
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Models & Features
:link: guide/model_overview
:link-type: doc

Feature steps and every registered model.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` FRED Datasets
:link: datasets/index
:link-type: doc

FRED-MD, FRED-QD, FRED-SD, and combined loaders.
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Replication Gallery
:link: guide/gallery
:link-type: doc

Published paper replication studies.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Paper Figures
:link: guide/figures
:link-type: doc

CSSED, fluctuation, PIT, and forecast-path exhibits.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Custom Extensions
:link: reference/custom/index
:link-type: doc

Plug in your own datasets, features, models, tests, and outputs.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Reference
:link: reference/index
:link-type: doc

Workflow reference grouped by responsibility.
:::

:::{grid-item-card} {octicon}`bookmark;1.5em;sd-mr-1` Glossary
:link: guide/glossary
:link-type: doc

Definitions of every core abstraction.
:::

:::{grid-item-card} {octicon}`quote;1.5em;sd-mr-1` Citing
:link: guide/citing
:link-type: doc

Citation metadata for papers, reports, and replication packages.
:::

::::

```{toctree}
:hidden:

guide/index
guide/model_overview
datasets/index
guide/gallery
guide/figures
reference/index
guide/glossary
guide/citing
```
