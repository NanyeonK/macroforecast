# Custom Extensions

Custom extensions are central to macrocast. Researchers should be able to add methods without editing package internals.

Target extension points:

```python
mc.custom_model(name)       # MVP implemented
mc.custom_preprocessor(name) # MVP implemented for matrix preprocessing
mc.target_transformer(name) # executable for autoreg target-lag models
mc.custom_benchmark(name)
mc.custom_metric(name)
```

Minimum custom model contract:

```python
@mc.custom_model("my_model")
def my_model(X_train, y_train, X_test, context):
    ...
    return y_pred
```

Minimum custom preprocessor contract:

```python
@mc.custom_preprocessor("my_preprocess")
def my_preprocess(X_train, y_train, X_test, context):
    ...
    return X_train_new, X_test_new
```

MVP custom model contract:

```python
fn(X_train, y_train, X_test, context) -> scalar
```

`X_test` is one row. The function may return a scalar or one-element sequence/array. `context` includes `model_name`, `feature_runtime_builder`, `legacy_feature_builder`, `feature_dispatch_source`, `target`, `horizon`, `feature_names`, `mode`, and `contract_version`.

Registered custom model names are accepted as `model_family` values in the current Python process and can be used in `Experiment.compare_models`.

MVP custom preprocessor contract:

```python
fn(X_train, y_train, X_test, context) -> (X_train, X_test)
```

`y_train` is provided as read-only context for target-aware feature preprocessing. The MVP API does not transform the training target. Target-scale transforms should become a separate extension contract later, because they require inverse-transform and evaluation-scale rules.

Target transformer skeleton:

```python
@mc.target_transformer("my_target_transform")
class MyTargetTransform:
    def fit(self, target_train, context):
        ...
        return self

    def transform(self, target, context):
        ...
        return target_transformed

    def inverse_transform_prediction(self, target_pred, context):
        ...
        return target_pred_raw
```

Registered target transformers are executable for the target-lag feature runtime and for raw-panel feature runtimes with supported linear model families or registered custom models. Raw-scale evaluation is required. See `target_transformer` for the full scale contract.

The final guide should specify:

- exact input types
- exact return types
- shape requirements
- context fields
- split and leakage discipline
- error behavior
- manifest recording
- how extension names interact with sweeps

Design requirements:

- no registry file edits for normal custom usage
- no subclass required for simple usage
- advanced class protocols can exist later
- custom and built-in methods share the same evaluation path
