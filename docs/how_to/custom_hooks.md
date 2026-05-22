# Custom Extensions

Custom extensions are first-class method-research APIs. A researcher should be
able to add a Layer 2 representation method, a Layer 3 model, a target-side
transform, or a final-matrix preprocessor, then compare the custom method
against built-in baselines under the same Layer 0 task, Layer 1 data
treatment, split, benchmark, and evaluation path.

> See also: [Partial layer execution](partial_layer_execution.md) -- when
> developing a custom hook the partial-layer helpers are the fastest way to
> iterate without re-running upstream every time.

## Extension map

Five runtime extension points; all are decorator APIs from
``macroforecast.custom`` (re-exported under the top-level ``mf`` namespace):

```python
import macroforecast as mf

mf.custom_feature_block(name, block_kind="temporal" | "rotation" | "factor")
mf.custom_feature_combiner(name)
mf.custom_preprocessor(name)
mf.target_transformer(name)
mf.custom_model(name)
```

Each section below documents that hook with: decorator usage, required
signature, input contract, output contract, a runnable worked example, and
the most common errors.

The matching contract metadata (returned as plain dicts) is available from:

```python
mf.custom_method_extension_contracts()       # all five at once
mf.custom_model_contract_metadata()
mf.custom_preprocessor_contract_metadata()
mf.target_transformer_contract_metadata()
```

Custom benchmarks and custom metrics are not decorator APIs yet -- benchmark
plugins go through benchmark configuration, metric plugins should be added as
a separate evaluation-layer contract.

---

## 1. Layer 3: ``custom_model``

Use ``custom_model`` when the method changes the estimator, loss, likelihood,
optimization, recursive rule, or prediction function but should consume the
same ``X_train``, ``y_train``, ``X_test`` interface as built-in models.

### Decorator usage

```python
import macroforecast as mf

@mf.custom_model("my_constant_model")
def my_constant_model(X_train, y_train, X_test, context):
    return float(y_train.mean())
```

### Required signature

```python
def fn(
    X_train: pd.DataFrame,   # n_train x n_features
    y_train: pd.Series,      # n_train, aligned to X_train.index
    X_test: pd.DataFrame,    # 1 x n_features (one prediction row)
    context: dict[str, Any],
) -> float | Sequence[float]:
    ...
```

The wrapper iterates row-by-row when L4 calls ``predict`` on multi-row
matrices, so the callable always sees a one-row ``X_test``.

### Input contract

| Argument | Type | Shape / contents |
|---|---|---|
| ``X_train`` | ``pd.DataFrame`` | ``n_train x n_features``; columns = Layer 2 feature names; floats with no NaNs (L3 dropna ran upstream). |
| ``y_train`` | ``pd.Series`` | ``n_train``; index aligned with ``X_train.index``; target column from the L3 ``y_final`` artifact. |
| ``X_test`` | ``pd.DataFrame`` | ``1 x n_features``; one prediction row. |
| ``context`` | ``dict[str, Any]`` | Required keys: ``contract_version`` (``"custom_model_v1"``), ``model_name``, ``feature_names``, ``params`` (the recipe-supplied ``params:`` dict for the fit_model node). |

Optional context fields appear in
``mf.custom_model_contract_metadata()["optional_context_fields"]`` -- e.g.
``train_index``, ``forecast_type``, ``forecast_object``, ``recursive_step``,
``auxiliary_payloads`` (FRED-SD mixed-frequency payloads only).

### Output contract

Return one of:

- a Python ``float`` -- single forecast value, or
- a one-element sequence/array (``[value]``, ``np.array([value])``, etc.).

The wrapper coerces both forms via ``float(...)``.

### Worked example

```python
import macroforecast as mf
from macroforecast import custom

custom.clear_custom_models()

@mf.custom_model("constant_train_mean")
def constant_train_mean(X_train, y_train, X_test, context):
    return float(y_train.mean())

recipe = """
0_meta:
  fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible}
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01,
             2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01]
      y:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
2_preprocessing:
  fixed_axes: {transform_policy: no_transform, outlier_policy: none,
               imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}
3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: y_h,   type: step, op: target_construction,
       params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_main
      type: step
      op: fit_model
      params: {family: constant_train_mean, min_train_size: 4,
               forecast_strategy: direct, training_start_rule: expanding,
               refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_main, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_main
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {primary_metric: mse, point_metrics: [mse]}
"""

result = mf.run(recipe)
forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
print("first three forecasts:", list(forecasts.values())[:3])

art = result.cells[0].runtime_result.artifacts["l4_model_artifacts_v1"]
fitted = next(iter(art.artifacts.values()))
assert fitted.family == "constant_train_mean"
assert fitted.framework == "custom"
```

The fitted ``L4ModelArtifactsArtifact`` records ``family =
"constant_train_mean"`` and ``framework = "custom"`` -- those two fields
distinguish a custom model from a built-in family in the manifest.

### Common errors

| Symptom | Cause | Fix |
|---|---|---|
| ``ValueError: unknown model family 'foo'`` at validate time | Recipe references ``family: foo`` but the registered name is different, or the module that calls ``@mf.custom_model("foo")`` was never imported in this Python process. | Import the module before ``Experiment(...).run()``; YAML alone does not register Python callables. |
| ``RuntimeError: custom model 'foo' predict() called before fit()`` | The runtime called ``predict`` on the adapter without ``fit`` -- usually a custom L4 op short-circuiting the fit path. | Use the canonical ``fit_model`` -> ``predict`` node pair (the ``examples/recipes/l4_minimal_ridge.yaml`` shape). |
| Predictions are all NaN | The custom callable returned ``np.nan`` or a non-numeric value; the wrapper's ``float(value)`` keeps NaN as-is. | Guard inside the callable; raise ``ValueError`` on degenerate fit windows so ``failure_policy`` decides. |

---

## 2. Layer 2: ``custom_preprocessor``

Custom preprocessors are matrix hooks that fit between the L1 raw panel and
the L3 feature DAG. v0.8.6 routes the same registered name to two distinct
points inside L2 depending on ``applied_at``.

### Decorator usage

```python
import macroforecast as mf

@mf.custom_preprocessor("clip_x1_at_2")
def clip_x1_at_2(X_train, y_train, X_test, context):
    cleaned_train = X_train.copy()
    cleaned_test = X_test.copy()
    if "x1" in cleaned_train.columns:
        cleaned_train["x1"] = cleaned_train["x1"].clip(upper=2.0)
        cleaned_test["x1"] = cleaned_test["x1"].clip(upper=2.0)
    return cleaned_train, cleaned_test
```

### Required signature

```python
def fn(
    X_train: pd.DataFrame,           # full L2 panel (or post-pipeline panel; see table below)
    y_train: pd.Series | None,       # None at the L2 boundary -- target is not yet split out
    X_test: pd.DataFrame,            # same frame as X_train at the L2 boundary
    context: dict[str, Any],         # the recipe's L2.leaf_config dict (read-only)
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    ...
```

At the L2 boundary the runtime substitutes ``X_train = X_test = frame`` so the
callable can do a single-pass clean. Returning a single ``pd.DataFrame`` is
also accepted -- the runtime treats it as both the train and pred output.

### Input contract

| Field | Contents |
|---|---|
| ``X_train`` index | ``pd.DatetimeIndex`` of the panel rows. |
| ``X_train`` columns | Predictor names + the target column (the target stays in the panel until L3 splits it out). |
| ``X_train`` dtypes | ``float64`` numeric columns; ``pd.NA`` for missing cells. |
| ``y_train`` | ``None`` at the L2 boundary -- target lives inside ``X_train``. |
| ``context`` | Copy of ``2_preprocessing.leaf_config`` from the recipe; useful for reading user-supplied options. |

### Output contract

- Same ``pd.DatetimeIndex`` as ``X_train`` (drop / re-index only when
  intentional and the target column drop is acceptable).
- May add or remove columns; **must preserve the target column** when
  ``applied_at='l2'`` because the canonical pipeline (transform / outlier /
  impute / frame_edge) still needs to run on it.
- For ``applied_at='l3'`` the returned frame becomes the L2 clean panel
  directly -- L3 reads it via the ``l2_clean_panel_v1`` selector.

### ``applied_at='l2'`` vs ``applied_at='l3'``

The same registered callable can be wired in at two distinct timing points
within L2. The choice is made by the recipe author through
``Experiment.use_preprocessor(name, applied_at=...)`` (or by setting the
right L2 ``leaf_config`` key directly).

| Aspect | ``applied_at='l2'`` (pre-pipeline) | ``applied_at='l3'`` (post-pipeline) |
|---|---|---|
| Runs at | Top of ``materialize_l2``, before ``transform`` / ``outlier`` / ``impute`` / ``frame_edge`` | Bottom of ``materialize_l2``, output becomes the L2 clean panel that L3 reads |
| Input | Raw L1 panel (untransformed; contains official t-codes if a code map was applied at L1) | L2 clean panel (post-canonical pipeline) |
| Output replaces | The raw input feeding the canonical pipeline | The clean panel that L3 reads |
| Use when | You need to clean / deflate / normalise *before* the McCracken-Ng pipeline | You want to override or replace the canonical pipeline output entirely |
| ``leaf_config`` key | ``custom_preprocessor`` | ``custom_postprocessor`` |
| Cleaning-log entry | ``{"custom_preprocessor": "<name>", "applied": True}`` (first step) | ``{"custom_postprocessor": "<name>", "applied": True}`` (last step) |
| Available since | v0.8.6 (#PR-A) | v0.2.5 (#251) |

### Worked example

```python
import macroforecast as mf
from macroforecast import custom

custom.clear_custom_preprocessors()

@mf.custom_preprocessor("clip_x1_at_2")
def clip_x1_at_2(X_train, y_train, X_test, context):
    out_train = X_train.copy()
    out_test = X_test.copy()
    if "x1" in out_train.columns:
        out_train["x1"] = out_train["x1"].clip(upper=2.0)
        out_test["x1"] = out_test["x1"].clip(upper=2.0)
    return out_train, out_test

# Set ``custom_postprocessor`` for ``applied_at='l3'`` (post-pipeline);
# set ``custom_preprocessor`` for ``applied_at='l2'`` (pre-pipeline).
recipe = """
0_meta:
  fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible}
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01,
             2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01]
      y:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
2_preprocessing:
  fixed_axes: {transform_policy: no_transform, outlier_policy: none,
               imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}
  leaf_config:
    custom_postprocessor: clip_x1_at_2
3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: y_h,   type: step, op: target_construction,
       params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_main
      type: step
      op: fit_model
      params: {family: ridge, alpha: 0.1, min_train_size: 4,
               forecast_strategy: direct, training_start_rule: expanding,
               refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_main, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_main
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {primary_metric: mse}
"""

result = mf.run(recipe)
panel = result.cells[0].runtime_result.artifacts["l2_clean_panel_v1"].panel.data
print("max(x1) after clip:", panel["x1"].max())
assert panel["x1"].max() == 2.0
```

The same callable can be wired in as ``leaf_config.custom_preprocessor``
(``applied_at='l2'`` -- pre-pipeline) instead of ``custom_postprocessor``
(``applied_at='l3'`` -- post-pipeline). The cleaning-log entry that lands
in ``l2_clean_panel_v1.cleaning_log['steps']`` records which timing point
ran.

If you prefer the high-level builder, the equivalent two lines are:

```python
exp = mf.Experiment(dataset="fred_md", target="INDPRO", horizons=[1])
exp.use_preprocessor("clip_x1_at_2", applied_at="l3")  # or "l2"
```

Note: ``Experiment(...)`` always wires the official FRED path; reach for
the recipe-dict / YAML form whenever you need to drive a custom inline
panel like the example above.

Switch to ``applied_at='l2'`` to run the same callable *before* the
McCracken-Ng pipeline -- useful when you want the official t-code map to
operate on already-deflated columns, for example.

### Common errors

| Symptom | Cause | Fix |
|---|---|---|
| ``ValueError: minimal L3 runtime requires target column in L2 clean panel`` | Returned a frame that dropped the target column under ``applied_at='l2'``. | Preserve the target column; the L2 pipeline downstream still needs it. |
| Output ignored, no error | Runtime saw an exception and silently fell back to identity (the dispatcher wraps the call in ``try / except`` for safety). | Manually call ``mf.custom.get_custom_preprocessor(name).function(...)`` once during dev to surface the traceback. |
| Index drift (rows missing in ``l2_clean_panel_v1``) | Returned a smaller / re-indexed DataFrame and the canonical pipeline subsequently dropped more rows. | Preserve the input ``DatetimeIndex`` unless drop is the explicit goal of the hook. |

---

## 3. Layer 2: ``target_transformer``

``target_transformer`` is the target-side extension. Use it when ``y`` itself
must be transformed for fitting and forecasts must be inverted to the raw
target scale before they reach L5 evaluation.

### Decorator usage

```python
import macroforecast as mf
import numpy as np

@mf.target_transformer("scale_by_std")
class ScaleByStd:
    def fit(self, target_train, context):
        self.scale_ = float(target_train.std() or 1.0)

    def transform(self, target, context):
        return target / self.scale_

    def inverse_transform_prediction(self, target_pred, context):
        return np.asarray(target_pred) * self.scale_
```

The decorator accepts either a class (instantiated once per L3 run) or a
factory ``() -> instance``.

### Required signature

The registered object (or factory's product) must implement three methods:

```python
class TargetTransformer:
    def fit(self, target_train: pd.Series, context: dict[str, Any]) -> None | "TargetTransformer":
        ...

    def transform(self, target: pd.Series, context: dict[str, Any]) -> pd.Series:
        ...

    def inverse_transform_prediction(
        self,
        target_pred: Sequence[float] | np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        ...
```

### Input / output contract

| Method | Input | Output |
|---|---|---|
| ``fit`` | ``target_train``: ``pd.Series`` of training-window y values; ``context``: dict (currently empty). | Side-effects only; return value is ignored. |
| ``transform`` | ``target``: ``pd.Series`` (training y or live y); ``context``: dict. | ``pd.Series`` on the transformed scale. |
| ``inverse_transform_prediction`` | ``target_pred``: 1-D iterable of model predictions (transformed scale); ``context``: dict. | ``np.ndarray`` on the raw target scale. |

Scale rule: model fits on the transformed scale, but every reported forecast
and metric is on the raw target scale. The runtime applies
``inverse_transform_prediction`` between L4 and L5.

### Worked example

```python
import macroforecast as mf
from macroforecast import custom
import numpy as np

custom.clear_custom_target_transformers()

@mf.target_transformer("scale_by_std")
class ScaleByStd:
    def fit(self, target_train, context):
        self.scale_ = float(target_train.std() or 1.0)
    def transform(self, target, context):
        return target / self.scale_
    def inverse_transform_prediction(self, target_pred, context):
        return np.asarray(target_pred) * self.scale_

experiment = mf.Experiment(dataset="custom_panel_only", target="y", horizons=[1])
recipe = experiment.to_recipe_dict()
recipe["1_data"]["leaf_config"]["target_transformer"] = "scale_by_std"
result = mf.run(recipe)

forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
print("raw-scale forecasts:", list(forecasts.values())[:3])
```

Final forecasts are on the raw target scale -- the inverse step ran between
L4 and L5.

### Common errors

| Symptom | Cause | Fix |
|---|---|---|
| ``TypeError: target transformer 'foo' must provide callable methods: ['fit', ...]`` | Registered object missing one of the three required methods. | Implement all three; ``fit`` may be a no-op but must exist. |
| Forecast values look 1000x off | Forgot to wire ``inverse_transform_prediction`` (returned the same value). | Ensure ``inverse_transform_prediction`` undoes ``transform`` round-trip. |
| ``feature_runtime`` rejection | The feature runtime is not in the supported allowlist. See [Target transformer](target_transformer.md) for the gate. | Use ``ols`` / ``ridge`` / ``lasso`` / ``elasticnet`` / a registered ``custom_model`` with target-lag or raw-panel runtimes. |

For deeper rules see [Target transformer](target_transformer.md).

---

## 4. Layer 3: ``custom_feature_block``

Layer 3 builds the research representation. Use ``custom_feature_block`` when
the method changes what predictors exist before the model is fit -- e.g.
temporal filters, lag-polynomial summaries, factor construction, rotation,
research-specific transforms.

A registered feature block is dispatched whenever an L3 step node's ``op``
matches the registered name (the runtime checks the registry before falling
through to built-in ops).

### Decorator usage

```python
import macroforecast as mf

@mf.custom_feature_block("double_it", block_kind="temporal")
def double_it(frame, params):
    return frame * 2.0
```

``block_kind`` is one of ``"temporal"``, ``"rotation"``, ``"factor"`` and
governs which Layer 2 axis exposes the registered name.

### Required signature

```python
def fn(
    frame: pd.DataFrame,         # the upstream L3 node's output (predictors)
    params: dict[str, Any],      # the recipe's params dict for this step node
) -> pd.DataFrame:
    ...
```

### Input / output contract

| Aspect | Contract |
|---|---|
| ``frame`` | ``pd.DataFrame`` from the upstream node; index is the panel ``DatetimeIndex``. May contain ``pd.NA`` (drop / fill is the block's responsibility). |
| ``params`` | The L3 step node's ``params:`` dict; ``block_kind`` is injected by the dispatcher when present. |
| Return | ``pd.DataFrame`` with the same index (or a deterministic subset). New columns are stable feature names. |

### Worked example

```python
import macroforecast as mf
from macroforecast import custom

custom.clear_custom_feature_blocks()

@mf.custom_feature_block("double_it", block_kind="temporal")
def double_it(frame, params):
    return frame * float(params.get("scale", 2.0))

recipe = {
    "0_meta": {"fixed_axes": {"failure_policy": "fail_fast", "reproducibility_mode": "seeded_reproducible"}},
    "1_data": {
        "fixed_axes": {"custom_source_policy": "custom_panel_only", "frequency": "monthly", "horizon_set": "custom_list"},
        "leaf_config": {
            "target": "y", "target_horizons": [1],
            "custom_panel_inline": {
                "date": ["2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01",
                         "2018-06-01", "2018-07-01", "2018-08-01", "2018-09-01", "2018-10-01"],
                "y":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "x1": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            },
        },
    },
    "2_preprocessing": {"fixed_axes": {"transform_policy": "no_transform", "outlier_policy": "none",
                                        "imputation_policy": "none_propagate", "frame_edge_policy": "keep_unbalanced"}},
    "3_feature_engineering": {
        "nodes": [
            {"id": "src_X", "type": "source",
             "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
            {"id": "src_y", "type": "source",
             "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
            {"id": "scaled_x", "type": "step", "op": "double_it",
             "params": {"block_kind": "temporal", "scale": 3.0}, "inputs": ["src_X"]},
            {"id": "y_h", "type": "step", "op": "target_construction",
             "params": {"mode": "point_forecast", "method": "direct", "horizon": 1}, "inputs": ["src_y"]},
        ],
        "sinks": {"l3_features_v1": {"X_final": "scaled_x", "y_final": "y_h"}, "l3_metadata_v1": "auto"},
    },
    "4_forecasting_model": {
        "nodes": [
            {"id": "src_X", "type": "source",
             "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "X_final"}}},
            {"id": "src_y", "type": "source",
             "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "y_final"}}},
            {"id": "fit_main", "type": "step", "op": "fit_model",
             "params": {"family": "ridge", "alpha": 1.0, "min_train_size": 4,
                        "forecast_strategy": "direct", "training_start_rule": "expanding",
                        "refit_policy": "every_origin", "search_algorithm": "none"},
             "inputs": ["src_X", "src_y"]},
            {"id": "predict", "type": "step", "op": "predict", "inputs": ["fit_main", "src_X"]},
        ],
        "sinks": {"l4_forecasts_v1": "predict", "l4_model_artifacts_v1": "fit_main", "l4_training_metadata_v1": "auto"},
    },
}

result = mf.run(recipe)
print("custom block applied:", "scaled_x" in result.cells[0].runtime_result.artifacts["l3_features_v1"].X_final.column_names or
      list(result.cells[0].runtime_result.artifacts["l3_features_v1"].X_final.data.columns))
```

The L3 step node's ``op: double_it`` triggers the registered callable; the
output ``DataFrame`` becomes ``scaled_x`` and feeds the L4 fit node.

### Common errors

| Symptom | Cause | Fix |
|---|---|---|
| ``NotImplementedError: L3 runtime does not support op 'foo'`` | Step node references ``op: foo`` but no callable named ``foo`` is registered. | Import the module that calls ``@mf.custom_feature_block("foo", ...)`` before ``mf.run(...)``. |
| Custom block silently ignored, fallback ran instead | Registered callable raised inside the dispatcher (the dispatcher swallows exceptions for safety). | Call ``mf.custom.get_custom_feature_block(name, block_kind=...).function(frame, params)`` directly during dev to surface the traceback. |
| Index drift in the L3 sink | Block returned a smaller / re-indexed frame and the L3 ``concat`` later mis-aligned. | Keep the input ``DatetimeIndex`` (or document the drop precisely). |

---

## 5. Layer 3: ``custom_feature_combiner``

Feature combiners are broader than feature blocks. Use a combiner when the
method changes how already-built blocks become final ``Z`` -- nonlinear
interactions across blocks, supervised block weighting, residualized
composition, custom low-dim representations.

### Decorator usage

```python
import macroforecast as mf
import pandas as pd

@mf.custom_feature_combiner("merge_concat")
def merge_concat(inputs, params):
    if isinstance(inputs, list):
        return pd.concat(inputs, axis=1)
    return inputs
```

### Required signature

```python
def fn(
    inputs: list[pd.DataFrame],   # list of upstream block frames
    params: dict[str, Any],       # the recipe's params dict for this step node
) -> pd.DataFrame:
    ...
```

The combiner receives the **list** of upstream frames (multi-frame merge
contract). Always handle both the list form and -- defensively -- the single
frame form, since the dispatcher only routes the list form when the
upstream has multiple inputs.

### Input / output contract

| Aspect | Contract |
|---|---|
| ``inputs`` | ``list[pd.DataFrame]``; each frame shares the panel ``DatetimeIndex``. |
| ``params`` | The L3 step node's ``params:`` dict (e.g. weights, components). |
| Return | ``pd.DataFrame`` with the same index. Columns are the public feature names of the combined representation. |

### Worked example

```python
import macroforecast as mf
from macroforecast import custom
import pandas as pd

custom.clear_custom_feature_combiners()

@mf.custom_feature_combiner("merge_concat")
def merge_concat(inputs, params):
    return pd.concat(inputs, axis=1)

# Inside a recipe, an L3 step node like
#   {id: combine, type: step, op: merge_concat, inputs: [block_a, block_b]}
# triggers the combiner when both upstream blocks are pd.DataFrames.

a = pd.DataFrame({"x": [1.0, 2.0]})
b = pd.DataFrame({"y": [3.0, 4.0]})

# Direct invocation for unit tests / debugging:
spec = mf.custom.get_custom_feature_combiner("merge_concat")
print(spec.function([a, b], {}))
```

To wire it into a real recipe, define an L3 step node whose ``op`` matches
the registered name and whose ``inputs`` list the upstream block ids -- the
dispatcher routes the list of upstream frames to the combiner.

### Common errors

| Symptom | Cause | Fix |
|---|---|---|
| ``KeyError`` inside combiner | Indexed into ``inputs`` as if it were always a list, but the dispatcher passed a single frame. | Branch on ``isinstance(inputs, list)`` (the worked example does). |
| Column collisions in the combined frame | Two upstream blocks emit a column with the same name. | Rename inside the combiner -- ``pd.concat([a, b.add_prefix("b_")], axis=1)``. |

---

## Method comparison sweeps

Method researchers often compare combinations:

- built-in Layer 2 representation with built-in Layer 4 generator;
- custom Layer 2 representation with built-in Layer 4 generator;
- built-in Layer 2 representation with custom Layer 4 generator;
- custom Layer 2 representation with custom Layer 4 generator.

Use ``sweep_axes`` for registry axes and ``leaf_sweep_axes`` for
variant-specific configuration names. ``leaf_sweep_axes`` materializes into
``leaf_config`` before each variant is compiled, so registered custom names
can vary across variants without editing package internals.

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_compare_methods
      failure_policy: continue_on_failure
  2_preprocessing:
    sweep_axes:
      temporal_feature_block: [moving_average_features, custom_temporal_features]
  4_forecasting_model:
    nodes:
      - id: src_X
        type: source
        selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
      - id: src_y
        type: source
        selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
      - id: fit_candidate
        type: step
        op: fit_model
        params: {family: my_custom_generator}
        inputs: [src_X, src_y]
      - id: predict_candidate
        type: step
        op: predict
        inputs: [fit_candidate, src_X]
    sinks:
      l4_forecasts_v1: predict_candidate
      l4_model_artifacts_v1: fit_candidate
      l4_training_metadata_v1: auto
```

If the custom name should apply only when the parent custom axis is
selected, use ``nested_sweep_axes`` with a ``leaf_config.<key>`` child. This
avoids duplicate built-in variants that differ only by an unused custom name.

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
``failure_policy=continue_on_failure``, unsupported cells remain visible in
the manifest; supported built-in/custom cells run and appear in the same
study manifest.

## FRED-SD mixed-frequency payloads

FRED-SD can emit extra Layer 2 payloads for custom Layer 3 models:

| Payload contract | Context key | Present when |
|---|---|---|
| ``fred_sd_native_frequency_block_payload_v1`` | ``context["auxiliary_payloads"]["fred_sd_native_frequency_block_payload"]`` | ``fred_sd_mixed_frequency_representation`` is ``native_frequency_block_payload`` or ``mixed_frequency_model_adapter``. |
| ``fred_sd_mixed_frequency_model_adapter_v1`` | ``context["auxiliary_payloads"]["fred_sd_mixed_frequency_model_adapter"]`` | ``fred_sd_mixed_frequency_representation`` is ``mixed_frequency_model_adapter``. |

The native-frequency payload includes ``blocks``, ``block_order``, and
``column_to_native_frequency``. The adapter payload records the adapter
route and the block-payload contract consumed by the model. This is the
supported place to implement research-specific mixed-frequency likelihoods,
weighting schemes, state updates, or direct forecast rules while leaving
Layer 1 source / t-code policy and Layer 2 representation choices auditable.

Use ``examples/custom_fred_sd_mixed_frequency_model.py`` as an executable
Python template and
``examples/recipes/templates/fred-sd-custom-mixed-frequency-model.yaml`` as
the matching recipe skeleton. YAML can select the registered
``model_family``, but it does not import Python code -- import the module
that calls ``@mf.custom_model(...)`` before compiling or running the recipe.

## Fair-comparison checklist

For a custom-method comparison to be valid:

- Keep Layer 0 task axes fixed unless task design is the object of comparison.
- Keep Layer 1 raw-data treatment fixed unless raw missing/outlier policy is
  the object of comparison.
- Put representation construction in Layer 2/3, not inside a model closure.
- Put estimator changes in Layer 3, not inside a feature block.
- Record all fitted choices in ``fit_state``, ``provenance``, or model
  tuning payload.
- Use identical split, horizon, benchmark, and evaluation settings across
  built-in and custom variants.

This separation matters because most method papers compare combinations: a
new representation with existing models, existing representations with a new
model, or a custom representation and custom model together. macroforecast
should make each of those comparisons executable without package-internal
edits.
