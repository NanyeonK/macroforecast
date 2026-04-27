# Custom Extensions

Custom extensions are first-class method research APIs. A researcher should be
able to add a Layer 2 representation method, a Layer 3 prediction method, or
both, then compare the custom method against built-in baselines under the same
Layer 0 task, Layer 1 data treatment, split, benchmark, and evaluation path.

## Extension Map

Implemented runtime extension points:

```python
mc.custom_feature_block(name, block_kind="temporal" | "rotation" | "factor")
mc.custom_feature_combiner(name)
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

## Layer 2: Custom Feature Combiners

Feature combiners are broader than feature blocks. Use a combiner when the
research method changes how already-built blocks become final `Z`: nonlinear
interactions across blocks, supervised block weighting, residualized block
composition, or a custom low-dimensional representation built from multiple
blocks.

Minimum contract:

```python
@mc.custom_feature_combiner("my_combiner")
def my_combiner(context):
    train = context.blocks_train["candidate_z"]
    pred = context.blocks_pred["candidate_z"]
    ...
    return FeatureCombinerCallableResult(
        Z_train=Z_train,
        Z_pred=Z_pred,
        feature_names=("custom_feature",),
        block_roles={"custom_feature": "custom"},
        fit_state={...},
        leakage_metadata={"lookahead": "forbidden"},
        provenance={...},
    )
```

Select it with `feature_block_combination=custom_feature_combiner` plus
`custom_feature_combiner=<registered name>` in the Layer 1 leaf config or
`custom_feature_blocks.combiner=<registered name>`.

## Layer 2: Final-Z Selection After Custom Blocks

`feature_selection_semantics=select_after_custom_feature_blocks` applies an operational
feature-selection policy after custom blocks or a custom combiner have produced
final `Z` candidates. The runtime records `custom_final_z_selection_v1` with
candidate names, selected names, dropped names, policy, fit state, and leakage
metadata.

Use this when custom representation columns should participate in the same
selection sweep as built-in columns. The selector still fits only inside the
current training window.

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

The required context fields are:

| Field | Meaning |
|---|---|
| `contract_version` | Custom model contract identifier; currently `custom_model_v1`. |
| `model_name` | Registered model name selected through `model_family`. |
| `target` | Current target series name. |
| `horizon` | Current forecast horizon. |
| `feature_names` | Public Layer 2 feature names. |
| `feature_runtime_builder` | Runtime feature route used to build the current matrices. |
| `block_order` | Ordered Layer 2 block names that produced the final matrix. |
| `block_roles` | Feature/block role metadata used by manifests and diagnostics. |
| `alignment` | Train/pred alignment contract for the current forecast origin. |
| `leakage_contract` | Runtime leakage policy and audit metadata. |
| `mode` | Execution mode, such as direct or recursive runtime. |

Optional context fields include compatibility aliases
`feature_builder`, `legacy_feature_builder`, and `feature_dispatch_source`;
runtime-specific fields such as `forecast_type`, `forecast_object`,
`train_index`, and iterated raw-panel payload metadata; and
`auxiliary_payloads` when a Layer 2 route emits an extra model-facing payload.

Registered custom model names are accepted through the current `model_family`
compatibility axis. Canonically, they are custom forecast generator families
and can be compared with built-in generators through the normal execution and
experiment APIs.

### FRED-SD Mixed-Frequency Payloads

FRED-SD can emit extra Layer 2 payloads for custom Layer 3 models:

| Payload contract | Context key | Present when |
|---|---|---|
| `fred_sd_native_frequency_block_payload_v1` | `context["auxiliary_payloads"]["fred_sd_native_frequency_block_payload"]` | `fred_sd_mixed_frequency_representation` is `native_frequency_block_payload` or `mixed_frequency_model_adapter`. |
| `fred_sd_mixed_frequency_model_adapter_v1` | `context["auxiliary_payloads"]["fred_sd_mixed_frequency_model_adapter"]` | `fred_sd_mixed_frequency_representation` is `mixed_frequency_model_adapter`. |

The native-frequency payload includes `blocks`, `block_order`, and
`column_to_native_frequency`. The adapter payload records the adapter route and
the block-payload contract consumed by the model. This is the supported place
to implement research-specific mixed-frequency likelihoods, weighting schemes,
state updates, or direct forecast rules while leaving Layer 1 source/t-code
policy and Layer 2 representation choices auditable.

Use `examples/custom_fred_sd_mixed_frequency_model.py` as an executable Python
template and
`examples/recipes/templates/fred-sd-custom-mixed-frequency-model.yaml` as the
matching recipe skeleton. YAML can select the registered `model_family`, but it
does not import Python code. Import the module that calls `@mc.custom_model(...)`
before compiling or running the recipe.

## Method Comparison Sweeps

Method researchers often need to compare all four combinations:

- built-in Layer 2 representation with built-in Layer 3 generator;
- custom Layer 2 representation with built-in Layer 3 generator;
- built-in Layer 2 representation with custom Layer 3 generator;
- custom Layer 2 representation with custom Layer 3 generator.

Use `sweep_axes` for registry axes and `leaf_sweep_axes` for variant-specific
configuration names. `leaf_sweep_axes` materializes into `leaf_config` before
each variant is compiled, so registered custom names can vary across variants
without editing package internals.

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
      failure_policy: skip_failed_cell
  2_preprocessing:
    sweep_axes:
      temporal_feature_block: [moving_average_features, custom_temporal_features]
  3_training:
    sweep_axes:
      model_family: [ridge, my_custom_generator]
```

If the custom name should apply only when the parent custom axis is selected,
use `nested_sweep_axes` with a `leaf_config.<key>` child. This avoids duplicate
built-in variants that differ only by an unused custom name.

```yaml
path:
  2_preprocessing:
    nested_sweep_axes:
      temporal_feature_block:
        moving_average_features: {}
        custom_temporal_features:
          leaf_config.custom_temporal_feature_block:
            - my_temporal_block
            - my_second_temporal_block
  3_training:
    sweep_axes:
      model_family: [ridge, my_custom_generator]
```

For custom combiners, bind the combiner name the same way:

```yaml
path:
  2_preprocessing:
    nested_sweep_axes:
      feature_block_combination:
        concatenate_named_blocks: {}
        custom_feature_combiner:
          leaf_config.custom_feature_combiner:
            - my_combiner
```

The compiler expands the grid first, then validates each variant. With
`failure_policy=skip_failed_cell`, unsupported cells are recorded as skipped
with `compiler_manifest.json`; supported built-in/custom cells run and appear
in the same study manifest.

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
