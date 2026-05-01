# Sweep Only What You Care About

Use `.sweep()` when one part of the design should vary and everything else should stay on the default profile.

The executable MVP sweep is model comparison:

```python
import macrocast as mc

result = (
    mc.Experiment(
        dataset="fred_md",
        target="INDPRO",
        start="1980-01",
        end="2019-12",
        horizons=[1, 3, 6],
    )
    .sweep({"models": ["ar", "ridge", "lasso"]})
    .run()
)
```

Then rank the variants:

```python
ranking = result.compare("msfe")
```

MVP aliases:

| Alias | Internal axis | Runtime status |
|-------|---------------|----------------|
| `model`, `models`, `model_family` | `4_forecasting_model.fit_model.params.family` | executable |
| `scaling`, `scaling_policy` | `2_preprocessing.scaling_policy` | full-route executable; simple exposure pending result-contract docs |
| `missing`, `x_missing`, `x_missing_policy` | `2_preprocessing.x_missing_policy` | full-route executable; simple exposure pending result-contract docs |
| `preprocessor`, `custom_preprocessor` | `2_preprocessing.custom_preprocessor` | fixed custom preprocessor executable; sweep blocked |
| `target_transformer` | `3_feature_engineering.target_construction` | fixed transformer executable for autoreg path; sweep blocked |

A single-value choice becomes a fixed axis. A multi-value choice becomes a sweep axis.

```python
exp = mc.Experiment(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
).sweep({"models": ["ridge"]})

recipe = exp.to_recipe_dict()
```

Representation and preprocessing sweeps are a full-route feature first. The
full layer contract can route Layer 2 x Layer 3 x Layer 4 grids to the sweep runner.
The simple API will expose this only after the public naming and result-summary
contract is stable.

Full grids, ablations, replications, benchmark suites, and multi-target wrappers follow the same rule: they are internal/full routes first, simple routes later.
