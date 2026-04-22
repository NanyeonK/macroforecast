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
| `model`, `models`, `model_family` | `3_training.model_family` | executable |
| `scaling`, `scaling_policy` | `2_preprocessing.scaling_policy` | recipe-only until preprocessing audit |
| `missing`, `x_missing`, `x_missing_policy` | `2_preprocessing.x_missing_policy` | recipe-only until preprocessing audit |
| `preprocessor`, `custom_preprocessor` | `2_preprocessing.custom_preprocessor` | fixed custom preprocessor executable; sweep blocked |
| `target_transformer` | `2_preprocessing.target_transformer` | fixed transformer executable for autoreg path; sweep blocked |

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

Preprocessing sweeps are intentionally blocked in the simple runtime for now. The full Layer 0 grammar can route them to the sweep runner, but the simple API does not expose them until Layer 2 fixes the preprocessing order, leakage, and result interpretation contract.

Full grids, ablations, replications, benchmark suites, and multi-target wrappers follow the same rule: they are internal/full routes first, simple routes later.
