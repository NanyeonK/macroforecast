# Simple API

The Simple API is the high-level Python interface for researchers who want to run a forecasting study without writing a YAML recipe first.

Use `mf.forecast(...)` for one default run:

```python
import macroforecast as mf

result = mf.forecast(
    "fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)
```

Use `mf.Experiment(...)` when you want to inspect the generated recipe, compare models, select FRED-SD variables, or write outputs to a chosen directory:

```python
import macroforecast as mf

exp = mf.Experiment(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1, 3, 6],
)

result = exp.compare_models(["ar_p", "ridge"]).run(
    output_directory="outputs/indpro_models"
)
```

The Simple API still lowers to the same recipe runtime. Use it when your study is one target, one dataset choice, and a small number of model or parameter choices. Use the Recipe API when you need full control over layer-by-layer YAML.

```{toctree}
:maxdepth: 1

quickstart
run_experiment
compare_models
read_results
fred_sd
```
