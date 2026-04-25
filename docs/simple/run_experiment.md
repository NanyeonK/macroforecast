# Run An Experiment

`Experiment` is the public workspace for one forecasting question. It collects the few choices a researcher should specify, fills the rest from the default profile, then lowers to the internal recipe and execution engine.

```python
import macrocast as mc

exp = mc.Experiment(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)

result = exp.run()
```

Required arguments:

- `dataset`: `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, or `fred_qd+fred_sd`
- `target`: target series name
- `start`: first sample date
- `end`: last sample date

Common optional arguments:

- `horizons`: forecast horizons, default `[1]`
- `frequency`: required only for `fred_sd` alone
- `vintage`: data vintage; if omitted, current/latest data is used
- `model_family`: default model, usually left as `ar`
- `primary_metric`: default `msfe`
- `random_seed`: default `42`

Inspect the lowered recipe before running:

```python
recipe = exp.to_recipe_dict()
```

Run with a custom output root:

```python
result = exp.run(output_root="outputs/indpro_default")
```

The simple API is a facade. It does not bypass recipes, manifests, or provenance. Every default it fills is written to the manifest.

## Single Run Or Sweep

If no sweep axes are present, `run()` returns `ExperimentRunResult`.

```python
result = exp.run()
result.forecasts
result.metrics
result.manifest
```

If `.compare_models()` or `.sweep()` creates multiple variants, `run()` returns `ExperimentSweepResult`.

```python
sweep = exp.compare_models(["ridge", "lasso"]).run()
sweep.variants
sweep.compare("msfe")
```
