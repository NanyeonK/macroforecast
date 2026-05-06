# Compare Models

Model comparison is a first-class operation. Changing the model does not change the information set, sample split, benchmark, preprocessing defaults, or evaluation metric.

In Layer 0 Simple, this call shape selects `study_scope = one_target_compare_methods`. Failure handling, reproducibility, and compute layout keep their Simple defaults unless you move to the Full YAML path.

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
    .compare_models(["ar", "ridge", "lasso"])
    .run()
)
```

The return value is an `ExperimentSweepResult`:

```python
result.variants
result.metrics
result.compare("mse")
result.forecasts
result.manifest
```

For follow-up parameter sweeps the L4 fit node has the stable id
``fit_main``, so chaining is predictable independent of the original
``model_family=``:

```python
result = (
    mf.Experiment(dataset="fred_md", target="INDPRO", horizons=[1])
    .compare_models(["ridge"])
    .compare("4_forecasting_model.nodes.fit_main.params.alpha", [0.1, 1.0])
    .run()
)
```

`compare_models()` accepts built-in model names and registered custom model names.

```python
@mf.custom_model("my_model")
def my_model(X_train, y_train, X_test, context):
    ...

result = (
    mf.Experiment(
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
