# User Guide

[Back to documentation home](../index.md)

This guide explains how `macroforecast` turns a forecasting study into one
specification you can read from top to bottom. If you have ever hand-written a
backtest loop and then struggled to keep preprocessing, feature construction,
and evaluation in step, this is the part of the documentation that shows the
alternative.

The package is built around a single idea. You declare each part of the
experiment once, and the runner executes every combination and scores it. The
map below shows those parts and the order in which a forecast flows through
them, from raw data on the left to a scored report at the end.

## The pipeline at a glance

```{mermaid}
flowchart TD
    A["load_fred_md / load_custom_csv"] --> B["DataBundle"]
    B --> C["data.spec → DataSpec"]
    C --> D["preprocess_spec<br/>t-codes · outliers · EM imputation"]
    D --> E["window.from_cutoffs<br/>estimation · validation · test · cadence"]
    E --> F["feature_spec<br/>lags · MARX · factors · Level"]
    F --> G["Arm<br/>model + preprocessing + features"]
    G --> H["pipeline_spec → PipelineSpec"]
    H --> I["run_pipeline → PipelineReport"]
    I --> R1[".accuracy<br/>relative RMSE"]
    I --> R2[".significance<br/>DM · Clark-West"]
    I --> R3[".mcs<br/>Model Confidence Set"]
    I --> R4[".forecasts<br/>full forecast frame"]

    classDef input fill:#e8f1fb,stroke:#3b7dd8,color:#13243b;
    classDef stage fill:#f4f6f8,stroke:#9aa7b4,color:#1d2733;
    classDef output fill:#eaf6ec,stroke:#43a05a,color:#10331b;
    class A,B,C input;
    class D,E,F,G,H,I stage;
    class R1,R2,R3,R4 output;
```

## How the pieces fit

A study starts with [data](concepts/data.md). A loader returns a `DataBundle`,
and `data.spec` records which series you forecast and over what sample. From
there the workflow is a short chain. [Preprocessing](concepts/preprocessing.md)
makes the series stationary and fills gaps, a [window](concepts/windows.md) fixes
the estimation, validation, and test split together with how often models refit,
and [feature engineering](concepts/features.md) turns the cleaned panel into the
lag, factor, and moving-average inputs a model consumes.

Those three choices plus a single model form an
[arm](concepts/models_and_arms.md), which is one complete contender. You collect
the arms, the targets, and an evaluation rule into a `pipeline_spec`, and
[running](concepts/running.md) it executes every contender across every target
and horizon. The result is a `PipelineReport` whose
[evaluation](concepts/evaluation.md) tables score each arm against a benchmark
and report where the differences are statistically real.

Every stage is leak-aware. No observation dated after a forecast origin can
enter the training data for that origin, and stateful steps such as imputation,
factor extraction, and standardization are refit on the available rows at each
origin rather than once on the full sample.

## How to use this guide

If you have not run the package yet, start with
[Getting Started](getting_started.md) for the shortest path from install to a
first result. Then read the [Concepts](concepts/index.md) pages in the order of
the map above. Each one explains one stage and most include a short executed
walkthrough you can paste and run.

When you need exact signatures, follow the reference link at the foot of each
concept page. The [Glossary](glossary.md), [Models and Features](model_overview.md),
and [Replication Gallery](gallery.md) are reached from the documentation home and
are useful once the workflow is familiar.

## Concepts

::::{grid} 1

:::{grid-item-card} Start the walkthrough
:link: concepts/index
:link-type: doc

Data, preprocessing, features, windows, models, running, and evaluation, in
order.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

concepts/index
```
