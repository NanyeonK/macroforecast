# Custom Function Quickstart

macroforecast provides three first-class Python extension points for custom
functions. Register once in Python, reference by name in a recipe or
experiment.

For full contract details see [Custom Extensions](custom_hooks.md). For
target-transformer runtime rules see [Target transformer](target_transformer.md).

## When to use which hook

| You want to... | Use |
|---|---|
| Replace the forecasting estimator (custom loss, recurrent rule, ML model) | `custom_model` |
| Post-process the feature matrix after Layer 2 construction | `custom_preprocessor` |
| Transform the training target (then inverse-transform predictions) | `target_transformer` |

All three hooks are decorator APIs from the `macroforecast.custom` module,
re-exported at the top-level `mf` namespace.

## Register a custom model

```python
import macroforecast as mf

@mf.custom_model("my_ridge")
def my_ridge(X_train, y_train, X_test, context):
    """Minimal custom ridge-like estimator."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return float(model.predict(X_test)[0])
```

Contract:

```python
fn(X_train, y_train, X_test, context) -> scalar or one-element sequence
```

- `X_train`: `(n_train, n_features)` array -- training features
- `y_train`: `(n_train,)` array -- training target (already split by the runtime)
- `X_test`: `(1, n_features)` array -- one test row per forecast origin
- `context`: dict with `model_name`, `target`, `horizon`, `feature_names`, and other runtime metadata

Rules:
- Fit only on `X_train` / `y_train`. Never read future rows or full-sample statistics.
- Return a scalar or a one-element array/sequence.

Use in a recipe (YAML):

```yaml
4_forecasting_model:
  nodes:
    - id: fit_custom
      type: step
      op: fit_model
      params: {family: my_ridge, min_train_size: 24, forecast_strategy: direct,
               training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
```

> **YAML + Python**: YAML selects the registered name via `family: my_ridge`.
> The Python file that registers `my_ridge` must be imported **before**
> `mf.run()` is called. YAML does not import Python modules automatically.

Use with `Experiment`:

```python
result = (
    mf.Experiment(
        dataset="fred_md",
        target="CPIAUCSL",
        start="1990-01",
        end="2019-12",
        horizons=[1, 3, 6],
        model_family="my_ridge",
    )
    .run()
)
```

## Register a custom preprocessor

```python
import macroforecast as mf
import numpy as np

@mf.custom_preprocessor("demean")
def demean(X_train, y_train, X_test, context):
    """Remove column means fit on training data only."""
    col_means = X_train.mean(axis=0)
    return X_train - col_means, X_test - col_means
```

Contract:

```python
fn(X_train, y_train, X_test, context) -> (X_train_new, X_test_new)
```

- Fit preprocessing decisions (e.g., column means, PCA components) on
  `X_train` only -- never on `X_test`.
- `y_train` is read-only context; do not transform it here.
- Return arrays with the same row count as the inputs.

Use with `Experiment`:

```python
result = (
    mf.Experiment(
        dataset="fred_md",
        target="CPIAUCSL",
        start="1990-01",
        end="2019-12",
        horizons=[1, 3, 6],
        model_family="ridge",
    )
    .use_preprocessor("demean")
    .run()
)
```

Use in a recipe (YAML):

```yaml
4_forecasting_model:
  nodes:
    - id: preprocess
      type: step
      op: apply_preprocessor
      params: {name: demean}
      inputs: [src_X, src_y]
```

## Register a target transformer

```python
import macroforecast as mf
import numpy as np

@mf.target_transformer("standardize_target")
class StandardizeTarget:
    """Standardize the training target; inverse-transform predictions."""

    def fit(self, target_train, context):
        self._mean = float(np.mean(target_train))
        self._std = float(np.std(target_train, ddof=1)) or 1.0
        return self

    def transform(self, target, context):
        return (target - self._mean) / self._std

    def inverse_transform_prediction(self, target_pred, context):
        return target_pred * self._std + self._mean
```

Contract: the transformer class must implement three methods:

| Method | Signature | Purpose |
|---|---|---|
| `fit` | `(target_train, context) -> self` | Fit on training window only |
| `transform` | `(target, context) -> target_transformed` | Applied to training target before model fitting |
| `inverse_transform_prediction` | `(target_pred, context) -> target_raw` | Restores predictions to raw scale |

Scale rules (enforced by the runtime):
- Model is trained on the **transformed** target.
- All reported forecasts and metrics are on the **raw** target scale.
- Benchmarks always remain on the raw scale for comparability.

**Current runtime gate**: `target_transformer` is executable only for
target-lag and raw-panel feature runtimes with `model_family` in
`ols`, `ridge`, `lasso`, `elasticnet`, or a registered `custom_model`.
Other feature runtimes reject non-`none` transformer values until their
scale contracts are designed.

Use in a recipe (YAML):

```yaml
1_data:
  leaf_config:
    target_transformer: standardize_target
```

> **Note**: `Experiment.use_target_transformer()` is not available in the
> current Python API. Use the YAML recipe path or mutate `to_recipe_dict()`
> before calling `mf.run()`:
>
> ```python
> recipe = exp.to_recipe_dict()
> recipe["1_data"]["leaf_config"]["target_transformer"] = "standardize_target"
> # then run via mf.run() with the dict or write it to YAML
> ```

## Using a custom function with YAML recipes

YAML recipes reference registered names as strings. Python registration must
happen in the same process that calls `mf.run()`.

Recommended pattern: keep your custom functions in a dedicated module, import
it at the top of your script.

```python
# my_study.py
import custom_functions          # registers @mf.custom_model, @mf.custom_preprocessor, etc.
import macroforecast as mf

result = mf.run("my_study.yaml", output_directory="output/")
```

```python
# custom_functions.py
import macroforecast as mf

@mf.custom_model("my_model")
def my_model(X_train, y_train, X_test, context):
    return float(y_train[-1])   # naive last-value baseline
```

Check what is registered:

```python
print(mf.list_custom_models())
print(mf.list_custom_preprocessors())
print(mf.list_custom_target_transformers())
```

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: custom model 'my_model' is not registered` | The Python file that calls `@mf.custom_model(...)` was not imported before `mf.run()`. | Import the registration module before `mf.run()`. |
| Custom model returns wrong shape | Return value has more than one element. | Return a scalar or a one-element array: `return float(pred[0])`. |
| Preprocessor causes row count mismatch | `X_train` and `y_train` row counts diverge after transformation. | Never add or remove rows in a preprocessor -- only transform feature values. |
| `target_transformer` rejected at runtime | Feature runtime is not in the supported allowlist (`ols`/`ridge`/`lasso`/`elasticnet`/custom model). | Switch to a supported model family, or set `target_transformer: none`. |
| `name must not start with '_'` | Custom name begins with underscore. | Use a name that starts with a letter: `"my_model"` not `"_my_model"`. |
| Registry survives across test runs | Module-level dicts persist in the same process. | Call `mf.clear_custom_extensions()` in test teardown or notebook re-runs. |

## See also

- [Custom Extensions](custom_hooks.md) -- full five-hook reference with all contract details
- [Target transformer](target_transformer.md) -- scale rules, runtime gate, inverse-transform contract
- [Bring your own data](../for_researchers/user_data_workflow.md) -- CSV/Parquet data format and loading
- [Simple API](../for_researchers/simple_api/index.md) -- `Experiment` and `mf.forecast(...)` walkthrough
