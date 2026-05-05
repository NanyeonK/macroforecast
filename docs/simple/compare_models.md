# Compare Models

> **API status note (v0.5.x)**: this page uses the planned mf.forecast / mf.Experiment Python facade
> shape. Those are not yet exported from macroforecast.__all__. For working v0.5.x code, use
> macroforecast.run("recipe.yaml"), macroforecast.replicate("manifest.json"),
> the RecipeBuilder (macroforecast.scaffold.builder.RecipeBuilder), or
> python -m macroforecast scaffold. See [Simple Docs index](index.md) for the full status note.


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
result.compare("msfe")
result.forecasts
result.manifest
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
