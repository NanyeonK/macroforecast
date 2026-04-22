# Target Transformer Contract

`target_transformer` is the y-side extension point. It is intentionally separate from `custom_preprocessor`, which is X-only.

Status: executable for `feature_builder="autoreg_lagged_target"`. Raw-panel and exogenous feature builders remain blocked until their horizon-aligned supervised target contract is implemented.

## Why This Is Separate

Transforming `y_train` changes the scale of the fitted model and the scale of predictions. That affects model recursion, benchmark comparison, metrics, artifacts, and reproducibility. It should not be hidden inside an X preprocessor.

## MVP Contract

The intended user protocol is:

```python
@mc.target_transformer("standardize_y")
class StandardizeY:
    def fit(self, y_train, context):
        ...
        return self

    def transform(self, y, context):
        ...
        return y_transformed

    def inverse_transform_prediction(self, y_pred, context):
        ...
        return y_pred_raw
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

The current runtime supports autoregressive target-lag models. It blocks `raw_feature_panel`, `factor_pca`, and other exogenous feature builders because those paths build horizon-specific supervised target vectors inside the raw-panel executor. That needs a separate implementation pass.

## Relationship To Existing Axes

Existing built-in axes still describe dataset or built-in target handling:

- `target_transform`: built-in transforms such as level, difference, log, growth rate
- `target_normalization`: built-in normalization options
- `target_transform_policy`: high-level policy for raw/tcode/custom target handling
- `inverse_transform_policy`: built-in inverse-transform policy axis
- `evaluation_scale`: metric scale policy

`target_transformer` is the runtime plugin name for a custom y-side protocol. It should eventually coordinate with those axes, but it should not replace them silently.

## Current Behavior

`mc.target_transformer(name)` registers the protocol object.

`Experiment.use_target_transformer(name)` lowers to the `target_transformer` registry axis.

Compilation accepts registered names. Non-`none` values are executable only for `feature_builder="autoreg_lagged_target"` and raw-scale evaluation.

Execution records the transformer in the manifest and adds prediction columns for `target_transformer`, `model_target_scale`, `forecast_scale`, and `y_pred_model_scale`.
