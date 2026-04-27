# Compare Models

Model comparison is a first-class operation. Changing the model does not change the information set, sample split, benchmark, preprocessing defaults, or evaluation metric.

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
    .compare_models(["ar", "ridge", "lasso"])
    .run()
)
```

The return value is an `ExperimentSweepResult`:

```python
result.variants
result.metrics
result.compare("msfe")
result.forecasts
result.manifest
```

`compare_models()` accepts built-in model names and registered custom model names.

```python
@mc.custom_model("my_model")
def my_model(X_train, y_train, X_test, context):
    ...

result = (
    mc.Experiment(
        dataset="fred_md",
        target="INDPRO",
        start="1980-01",
        end="2019-12",
    )
    .compare_models(["ridge", "my_model"])
    .run()
)
```

Each variant writes its own artifacts. The sweep root writes one manifest that records the axes that changed and the status of every variant.
