# custom_model

[Back to custom extensions](index.md)

Use custom models when the estimator is not built into `macroforecast`. A
custom model returns a `ModelSpec`, so it can be passed to the runner exactly
like a built-in model name.

## custom_model

```python
mf.models.custom_model(
    name,
    fit_func,
    *,
    default_params=None,
    default_preset=None,
    search_spaces=None,
    family="custom",
    requires_extra=None,
    metadata=None,
) -> mf.models.ModelSpec
```

### Fit Callable Contract

```python
fit_func(X: pandas.DataFrame, y: pandas.Series, **params) -> fitted_object
fitted_object.predict(X_test: pandas.DataFrame) -> predictions
```

Prediction output must have length `len(X_test)`. If it is a pandas object, it
must use `X_test.index` or `RangeIndex(len(X_test))`.

### Example

```python
class MeanFit:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return numpy.full(len(X), self.value)

def mean_model(X, y, *, offset=0.0):
    return MeanFit(pandas.Series(y).mean() + offset)

model = mf.models.custom_model(
    "mean_model",
    mean_model,
    default_params={"offset": 0.0},
    default_preset="small",
    search_spaces={"small": {"offset": (-0.1, 0.0, 0.1)}},
)
```

### Runner Use

```python
result = mf.forecasting.run(
    panel,
    model,
    window=window,
    features=features,
)
```

`mf.forecasting.run` is atomic: one model per call. To compare a custom model
against a benchmark, give each one its own `Arm` and run them through
`pipeline_spec`/`run_pipeline` instead -- for the full worked example (your
own CSV, this model, a scored horse race, and a paper-ready table), continue
to the full horse-race tutorial:
[Your Data, Your Model, One Table](../../guide/custom_data_tutorial.md).

## custom_model_ensemble

Use `mf.model_ensemble.custom_model_ensemble(...)` when the extension is a
fit-time composition of member models, not a post-forecast combination.

```python
spec = mf.model_ensemble.custom_model_ensemble(
    "my_stacker",
    fit_func=my_ensemble_fit,
    default_params={"base_models": ("ridge", "lasso")},
)
```

| Custom type | Stage | Output |
| --- | --- | --- |
| `custom_model` | one estimator fit | `ModelSpec` |
| `custom_model_ensemble` | fit-time member-model composition | `ModelSpec` with `family="model_ensemble"` |
| `custom_combination` | post-forecast combination | `CombinationSpec` |

## Metadata

The model spec stores the model name, callable name, default parameters,
search-space metadata, optional dependency labels, and user metadata. It does
not store the source code of `fit_func`.
