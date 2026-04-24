# Custom Extensions

Custom extensions are first-class method research APIs. A researcher should be
able to add a Layer 2 representation method, a Layer 3 prediction method, or
both, then compare the custom method against built-in baselines under the same
Layer 0 task, Layer 1 data treatment, split, benchmark, and evaluation path.

## Extension Map

Implemented runtime extension points:

```python
mc.custom_feature_block(name, block_kind="temporal" | "rotation" | "factor")
mc.custom_preprocessor(name)
mc.target_transformer(name)
mc.custom_model(name)
```

The public contract metadata is available from:

```python
mc.custom_method_extension_contracts()
mc.custom_model_contract_metadata()
mc.custom_preprocessor_contract_metadata()
mc.target_transformer_contract_metadata()
```

Custom benchmarks and custom metrics are not decorator APIs yet. Benchmark
plugins are supported through benchmark configuration; metric plugins should be
added as a separate evaluation-layer contract.

## Layer 2: Custom Feature Blocks

Layer 2 owns the research representation `Z`. Custom feature blocks are the
right hook when the method changes what predictors exist before the model is
fit: temporal filters, lag-polynomial summaries, nonlinear transforms, factor
construction, rotation, supervised feature construction, or other
research-specific blocks.

Minimum contract:

```python
@mc.custom_feature_block("my_block", block_kind="temporal")
def my_block(context):
    ...
    return FeatureBlockCallableResult(
        train_features=train_frame,
        pred_features=pred_frame,
        feature_names=("public_name",),
        runtime_feature_names=("runtime_column",),
        fit_state={...},
        leakage_metadata={"lookahead": "forbidden"},
        provenance={...},
    )
```

The callable receives a `FeatureBlockCallableContext` with the aligned
training matrix, prediction row, target history, forecast origin, horizon,
train/pred indices, selected source frame, predictor names, fit scope, feature
namespace, and metadata.

Return rules:

- `train_features` and `pred_features` must have matching columns and stable
  row alignment.
- `feature_names` are public names recorded in manifests and model context.
- `runtime_feature_names` are the concrete matrix column names.
- `leakage_metadata["lookahead"]` is required and should normally be
  `"forbidden"`.
- `fit_state` and `provenance` should record fitted parameters, selected
  source columns, and any sweep-relevant options.

Custom feature blocks can be selected through Layer 2 feature-block axes, for
example `temporal_feature_block=custom_temporal_features` plus
`custom_temporal_feature_block=<registered name>`.

## Layer 2: Matrix Preprocessors

Custom preprocessors are post-representation matrix hooks. Use them when the
research method keeps the same conceptual Layer 2 representation but applies a
final matrix operation before Layer 3, such as a user-defined scaler,
winsorizer, shrinkage transform, residualization, or sparse screen.

Minimum contract:

```python
@mc.custom_preprocessor("my_preprocess")
def my_preprocess(X_train, y_train, X_test, context):
    ...
    return X_train_new, X_test_new
```

`y_train` is read-only context. This hook does not transform the target. If the
target scale changes, use `target_transformer`.

## Layer 2: Target Transformers

Target transformers support research designs where `y` itself is transformed
and forecasts must be inverted to the raw target scale.

Minimum contract:

```python
@mc.target_transformer("my_target_transform")
class MyTargetTransform:
    def fit(self, target_train, context):
        return self

    def transform(self, target, context):
        return target_transformed

    def inverse_transform_prediction(self, target_pred, context):
        return target_pred_raw
```

Final forecasts and evaluation metrics must remain on the raw target scale.
The transformer is executable for target-lag feature runtimes and supported
raw-panel runtimes.

## Layer 3: Custom Models

Layer 3 owns model fitting and prediction given the finalized Layer 2
representation. Use `custom_model` when the method changes the estimator,
loss, likelihood, optimization, recursive rule, or prediction function but
should consume the same `Z_train`, `y_train`, and `Z_pred` interface as built-in
models.

Minimum contract:

```python
@mc.custom_model("my_model")
def my_model(X_train, y_train, X_test, context):
    ...
    return y_pred
```

Runtime contract:

- `X_train`: `n_train x n_features` matrix from Layer 2.
- `y_train`: aligned training target.
- `X_test`: one-row prediction matrix.
- return value: scalar or one-element sequence/array.
- `context["contract_version"] == "custom_model_v1"`.

The model context includes `model_name`, `target`, `horizon`, `feature_names`,
`feature_builder`, `feature_runtime_builder`, `legacy_feature_builder`,
`feature_dispatch_source`, `block_order`, `block_roles`, `alignment`,
`leakage_contract`, and `mode`.

Registered custom model names are accepted as `model_family` values in the
current Python process and can be compared with built-in models through the
normal execution and experiment APIs.

## Fair Comparison Checklist

For a custom method comparison to be valid:

- Keep Layer 0 task axes fixed unless task design is the object of comparison.
- Keep Layer 1 raw data treatment fixed unless raw missing/outlier policy is
  the object of comparison.
- Put representation construction in Layer 2, not inside a model closure.
- Put estimator changes in Layer 3, not inside a feature block.
- Record all fitted choices in `fit_state`, `provenance`, or model tuning
  payload.
- Use identical split, horizon, benchmark, and evaluation settings across
  built-in and custom variants.

This separation matters because most method papers compare combinations: a new
representation with existing models, existing representations with a new
model, or a custom representation and custom model together. Macrocast should
make each of those comparisons executable without package-internal edits.
