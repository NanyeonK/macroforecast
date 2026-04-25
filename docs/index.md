# macrocast

> Choose a forecasting path, see what is disabled, generate a runnable recipe, and then run the same path from CLI or notebook.

macrocast is a research-oriented forecasting package for economists and macro-finance researchers. The package now has two user surfaces:

- a decision/navigation surface for choosing a valid study path;
- an execution/API surface for running the selected path.

The docs should be read in that order. The API reference is not the entry point for a package with many compatible and incompatible choices.

## Start Here

| You want to... | Start with |
|---|---|
| use the interactive tree UI | [Open Navigator App](navigator_app/index.html) |
| choose a path and see disabled branches | [Navigator Docs](navigator/index.md) |
| inspect why a recipe can or cannot run | [Path Resolver](navigator/path_resolver.md) |
| map choices such as model, feature representation, forecast object, and importance method | [Compatibility Engine](navigator/compatibility_engine.md) |
| reproduce or adapt a paper-style route | [Replication Library](navigator/replication_library.md) |
| generate YAML and execute it from CLI or notebook | [YAML and Execution](navigator/yaml_execution.md) |

## Documentation Tracks

| Track | Use it when |
|-------|-------------|
| [Navigator Docs](navigator/index.md) | You want the package to guide a decision tree: selectable options, disabled branches, canonical path effects, replication recipes, and runnable YAML. |
| [Simple Docs](simple/index.md) | You want to run forecasts, compare models, sweep a few choices, or add a custom method without learning the internal recipe system. |
| [Detailed Docs](detail/index.md) | You want to understand defaults, recipe layers, registry axes, execution internals, artifacts, and reproducibility. |
| [API Reference](api/index.md) | You need function signatures and class documentation. |

## Minimal Execution API

After a path is chosen, simple execution can still use the high-level API:

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

1. **Navigation before execution.** Users should see valid and invalid branches before writing a recipe.
2. **Defaults are explicit.** The default profile must be named, versioned, and written to the manifest.
3. **Only cared-about choices become sweeps.** Users should not configure every axis to compare one decision.
4. **Custom methods are first-class.** User-defined preprocessing, models, benchmarks, and metrics must run beside built-ins.
5. **Advanced detail remains auditable.** Every simple experiment can lower to recipes, runs, artifacts, manifests, and canonical paths.

```{toctree}
:hidden:
:maxdepth: 1

install
navigator/index
simple/index
detail/index
getting_started/index
user_guide/index
api/index
CONVENTIONS
```
