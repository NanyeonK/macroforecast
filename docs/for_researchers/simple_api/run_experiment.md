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
- `random_seed`: default `0`

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
