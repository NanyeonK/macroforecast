# Compare models and parameters

Use `compare_models(...)` to evaluate several model families under the same data, target, horizons, and sample window.

```python
import macroforecast as mf

result = (
    mf.Experiment(
        dataset="fred_md",
        target="INDPRO",
        start="1980-01",
        end="2019-12",
        horizons=[1, 3, 6],
    )
    .compare_models(["ar_p", "ridge", "lasso"])
    .run(output_directory="outputs/indpro_model_compare")
)

metrics = result.metrics
ranking = result.ranking
summary = result.mean(metric="mse")
```

For a parameter sweep, use `compare(axis_path, values)`. The default Simple API fit node is normalized to `fit_main`, so the L4 model-parameter path is stable for follow-up sweeps:

```python
import macroforecast as mf

result = (
    mf.Experiment(
        dataset="fred_md",
        target="INDPRO",
        start="1980-01",
        end="2019-12",
        horizons=[1],
        model_family="ridge",
    )
    .compare("4_forecasting_model.nodes.fit_main.params.alpha", [0.1, 1.0, 10.0])
    .run(output_directory="outputs/indpro_ridge_alpha")
)

summary = result.mean(metric="mse")
```

`sweep(axis_path, values)` is an alias for `compare(axis_path, values)`:

```python
exp = mf.Experiment(
    dataset="fred_md",
    target="INDPRO",
    start="1980-01",
    end="2019-12",
    horizons=[1],
    model_family="ridge",
)

exp.sweep("4_forecasting_model.nodes.fit_main.params.alpha", [0.1, 1.0])
```


You can also sweep the training-window rule itself:

```python
import macroforecast as mf

result = (
    mf.Experiment(
        dataset="fred_md",
        target="INDPRO",
        start="1980-01",
        end="2019-12",
        horizons=[1],
    )
    .compare(
        "4_forecasting_model.nodes.fit_main.params.training_start_rule",
        ["expanding", "rolling"],
    )
    .run(output_directory="outputs/indpro_window_compare")
)
```

Note: switching to `"rolling"` also requires setting `params.rolling_window` to
the desired window size — use a full YAML recipe (Recipe API) for that.
