# How to add a custom model

Register a Python callable as a model family and use it in a recipe.

---

## Register a custom model

Decorator form:

```python
import macroforecast as mf

@mf.register_model("my_model")
def my_model(X_train, y_train, X_test, context):
    # fit only on X_train / y_train
    # return a scalar or one-element sequence
    return float(y_train.mean())
```

Direct-call form (useful when the function is defined elsewhere):

```python
mf.register_model("my_model", my_model, description="mean baseline")
```

Confirm registration:

```python
print(mf.list_custom_models())  # ('my_model',)
```

---

## Use in a recipe

Reference the registered name via `family:` in any `fit_model` node:

```yaml
4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: src_y
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
    - id: fit_custom
      type: step
      op: fit_model
      params:
        family: my_model
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
        min_train_size: 6
      inputs: [src_X, src_y]
    - id: predict_custom
      type: step
      op: predict
      inputs: [fit_custom, src_X]
  sinks:
    l4_forecasts_v1: predict_custom
    l4_model_artifacts_v1: fit_custom
    l4_training_metadata_v1: auto
```

The Python file that registers `my_model` must be imported before `mf.run()`.
YAML does not import Python modules.

---

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: unknown model family 'my_model'` | Registration module not imported before `mf.run()` | Import the module at the top of your script; YAML cannot register callables |
| Predictions are wrong shape | Return value has more than one element | Return a scalar: `return float(pred[0])` |
| Predictions are all NaN | Callable returned `np.nan` or raised silently | Guard inside the callable; raise `ValueError` on degenerate windows |
| Registry persists across test runs | Module-level dict survives in the same process | Call `mf.clear_custom_models()` in test teardown or notebook re-runs |
| `name must not start with '_'` | Name begins with underscore | Use a name that starts with a letter: `"my_model"` not `"_my_model"` |

---

## See also

- {doc}`use_custom_hooks` for all five extension points (models, preprocessors,
  target transformers, feature blocks, combiners)
- {doc}`../tutorial/03_custom_model` for the narrative tutorial with step-by-step
  explanation
