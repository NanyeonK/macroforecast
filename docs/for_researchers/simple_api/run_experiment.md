# Run an experiment

Use `Experiment` when you want to build a study step by step before running it.

```python
import macroforecast as mf

exp = mf.Experiment(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)

result = exp.run()
```

Constructor arguments:

- `dataset`: `"fred_md"`, `"fred_qd"`, `"fred_sd"`, `"fred_md+fred_sd"`, or `"fred_qd+fred_sd"`
- `target`: target series name
- `horizons`: forecast horizons; default `(1,)`
- `frequency`: required when `dataset="fred_sd"` alone
- `start`, `end`: optional sample-window endpoints
- `model_family`: default `"ar_p"`
- `random_seed`: default `42`

## Default forecast settings

`Experiment` constructor only exposes the most common knobs. The rest are
inherited from the L4 schema defaults (kept in sync with
`macroforecast.defaults`):

| Setting | Default | Where to change |
|--------|--------|-----------------|
| Forecast strategy | `direct` | `.compare("4_forecasting_model.nodes.fit_main.params.forecast_strategy", [...])` |
| Training window | `expanding` | `.compare("4_forecasting_model.nodes.fit_main.params.training_start_rule", [...])` |
| Refit policy | `every_origin` | `.compare("4_forecasting_model.nodes.fit_main.params.refit_policy", [...])` |
| Benchmark min train size | 5 (zero_change benchmark) | Recipe API |

`training_start_rule` accepts `"expanding"`, `"rolling"` (requires
`params.rolling_window` size), or `"fixed"` (requires
`params.fixed_training_end_date`). For full control over L0–L8 axes, write a
YAML recipe and run it with `mf.run("recipe.yaml")`.

Inspect the generated recipe before running:

```python
recipe = exp.to_recipe_dict()
```

Write the generated recipe as YAML:

```python
yaml_text = exp.to_yaml()
```

Run and write artifacts to a chosen directory:

```python
result = exp.run(output_directory="outputs/indpro_default")
```

Validate the generated recipe before execution:

```python
exp.validate()
```

If the run wrote a manifest, replicate it later:

```python
replication = exp.replicate("outputs/indpro_default/manifest.json")
```
