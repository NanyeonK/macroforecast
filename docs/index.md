# macrocast

> Build reproducible macroeconomic forecasting experiments with defaults you can trust and sweeps only where you care.

macrocast is a research-oriented forecasting package for economists and macro-finance researchers. The user-facing unit is an `Experiment`: one workspace that can run a default forecast, compare models, sweep selected choices, and include custom methods under the same evaluation framework.

Most experiments should not require users to understand every internal axis. Defaults handle the standard empirical-macro path. When a researcher cares about a choice, that choice becomes explicit, sweepable, and recorded in the output manifest.

## Documentation Tracks

| Track | Use it when |
|-------|-------------|
| [Simple Docs](simple/index.md) | You want to run forecasts, compare models, sweep a few choices, or add a custom method without learning the internal recipe system. |
| [Detailed Docs](detail/index.md) | You want to understand defaults, recipe layers, registry axes, execution internals, artifacts, and reproducibility. |
| [API Reference](api/index.md) | You need function signatures and class documentation. |

## MVP API

The public MVP API is designed around this shape:

```python
import macrocast as mc

result = mc.forecast(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)
```

```python
exp = (
    mc.Experiment(
        dataset="fred_md",
        target="INDPRO",
        horizons=[1, 3, 6],
        start="1980-01",
        end="2019-12",
    )
    .compare_models(["ar", "ridge", "lasso"])
    .sweep({"scaling": ["none", "standard"]})
)

result = exp.run()
```

## Core design principles

1. **One experiment can start simple.** A default forecast should run with only dataset, target, horizons, and an explicit sample period.
2. **Defaults are explicit.** The default profile must be named, versioned, and written to the manifest.
3. **Only cared-about choices become sweeps.** Users should not configure every axis to compare one decision.
4. **Custom methods are first-class.** User-defined preprocessing, models, benchmarks, and metrics must run beside built-ins.
5. **Advanced detail remains auditable.** Every simple experiment can lower to recipes, runs, artifacts, and manifests.

```{toctree}
:hidden:
:maxdepth: 1

install
simple/index
detail/index
getting_started/index
user_guide/index
api/index
CONVENTIONS
```
