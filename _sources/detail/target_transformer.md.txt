# Target Transformer Contract

`target_transformer` is the target-side extension point. It is intentionally separate from `custom_preprocessor`, which is predictor-only.

Status: executable for the target-lag feature runtime and for raw-panel feature
runtimes with `model_family` in `ols`, `ridge`, `lasso`, or `elasticnet`, plus
registered custom models. Raw-scale evaluation is required.

## Why This Is Separate

Transforming the training target changes the scale of the fitted model and the scale of predictions. That affects model recursion, benchmark comparison, metrics, artifacts, and reproducibility. It should not be hidden inside a predictor preprocessor.

## MVP Contract

The intended user protocol is:

```python
@mc.target_transformer("standardize_target")
class StandardizeTarget:
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

Required scale contract:

- model training scale: transformed
- model prediction scale: transformed
- final forecast scale: raw
- evaluation scale: raw
- benchmark scale: raw

This means the transformer helps model fitting, but every reported forecast and metric stays in the original target units.

## Runtime Rules

Execution must fit the transformer inside each training window only.

The model sees transformed training targets.

One-step predictions are inverse-transformed before being written to artifacts.

Recursive autoreg models must keep history on the raw scale and re-transform the training target inside each fit window. Transformed predictions must not be appended directly to raw history.

Benchmarks stay on raw scale. Comparisons happen only after model predictions are inverse-transformed back to raw scale.

Metrics are raw-scale only in the first executable version. `evaluation_scale="transformed"` and `evaluation_scale="both"` remain out of scope for this extension until explicitly designed.

The current runtime supports autoregressive target-lag models and selected
raw-panel supervised models. Raw-panel support is limited to `ols`, `ridge`,
`lasso`, `elasticnet`, or registered custom models because those paths can keep
the target transformer fit/inverse-transform contract aligned with each
horizon-specific supervised target vector. Compatibility builders such as
`pca_factor_features` are treated by this gate according to the feature runtime they
compile to; static-factor recipes therefore use the same raw-panel model-family
allowlist. Non-raw-panel feature runtimes remain gated until their
target-transformer scale contract is designed and tested.

## Relationship To Existing Axes

Existing built-in axes still describe dataset or built-in target handling:

- `target_transform`: built-in transforms such as level, difference, log, growth rate
- `target_normalization`: built-in normalization options
- `target_transform_policy`: high-level policy for raw/tcode/custom target handling
- `inverse_transform_policy`: built-in inverse-transform policy axis
- `evaluation_scale`: metric scale policy

`target_transformer` is the runtime plugin name for a custom target-side protocol. It should eventually coordinate with those axes, but it should not replace them silently.

## Current Behavior

`mc.target_transformer(name)` registers the protocol object.

`Experiment.use_target_transformer(name)` lowers to the `target_transformer` registry axis.

Compilation accepts registered names. Non-`none` values are executable only for
supported target-lag/raw-panel feature runtimes and raw-scale evaluation.

Execution records the transformer in the manifest and adds prediction columns for `target_transformer`, `model_target_scale`, `forecast_scale`, and the legacy `y_pred_model_scale` column.
