# Getting Started

macrocast lets you run a complete forecasting study from a YAML recipe. Most recipes are a dozen lines because the defaults match what empirical-macro papers actually do.

## Learning path

| Step | Time | What you will learn |
|------|------|---------------------|
| 1. [Quickstart](quickstart.md) | 5 min | Run your first forecast with defaults. |
| 2. [Your First Study](first_study.md) | 20 min | Add one axis deviation — your first research choice. |
| 3. [Understanding Output](understanding_output.md) | 10 min | Read `predictions.csv`, `metrics.json`, and `manifest.json`. |
| 4. [Stages Reference](stages_reference.md) | reference | Every operational value on every axis for Stages 0 + 1. |

## Prerequisites

- Python 3.10+ with macrocast installed ([Installation](../install.md)).
- Basic familiarity with pandas DataFrames.
- Basic knowledge of macroeconomic forecasting (what FRED-MD is, what out-of-sample evaluation means).

## The 30-second mental model

- **Recipe** — a YAML file that fully specifies one forecasting study. Data + preprocessing + model + benchmark + metrics, in one place.
- **Stage 0 (Design)** — the study grammar. Decides runner, sweep shape, reproducibility. Six axes. Mostly auto-derived. [Deep dive](../user_guide/design.md).
- **Stage 1 (Data)** — everything about the dataset, target structure, evaluation window, benchmark, predictors, and data-handling policies. Twenty axes. Most have sensible defaults. [Deep dive](../user_guide/data/index.md).
- **Compiler** — validates your recipe and decides if it can execute in the current runtime.
- **Execution** — runs the forecast, writes predictions + metrics + a complete `manifest.json` that records every resolved choice.

## Key promise: defaults match research practice

If you only set `dataset`, `target`, and `horizons` in your leaf_config, you get a reasonable benchmark run: AR(BIC) benchmark, monthly FRED-MD, pseudo-OOS rolling origin, one model per horizon. No hidden knobs.

Want to deviate? Each axis has exactly one documentation page that tells you:

- **Selection question** — what research question does this axis answer?
- **Default** — which research convention sits at the default?
- **Value catalog** — every operational value with *when to use* and *verify* columns.
- **Recipe usage** — minimal YAML showing the non-default.

```{toctree}
:hidden:
:maxdepth: 1

quickstart
first_study
understanding_output
stages_reference
```
