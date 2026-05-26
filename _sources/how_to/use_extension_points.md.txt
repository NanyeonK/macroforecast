# How to use extension points

macroforecast provides five runtime extension points. All are decorator APIs
re-exported from `macroforecast.custom` at the top-level `mf` namespace.

---

## Extension map

| Extension point | What it changes | Decorator | Signature summary |
|---|---|---|---|
| `custom_model` | The forecasting estimator | `@mf.register_model(name)` | `fn(X_train, y_train, X_test, context) -> float` |
| `custom_preprocessor` | The feature matrix after L2 construction | `@mf.custom_preprocessor(name)` | `fn(X_train, y_train, X_test, context) -> (X_train, X_test)` |
| `target_transformer` | The training target (fit + inverse) | `@mf.target_transformer(name)` | class with `fit`, `transform`, `inverse_transform_prediction` |
| `custom_feature_block` | An L3 pipeline step | `@mf.custom_feature_block(name, block_kind=...)` | `fn(frame, params) -> pd.DataFrame` |
| `custom_feature_combiner` | How L3 blocks are merged | `@mf.custom_feature_combiner(name)` | `fn(inputs, params) -> pd.DataFrame` |

---

## One snippet per extension point

### custom_model

```python
import macroforecast as mf

@mf.register_model("mean_baseline")
def mean_baseline(X_train, y_train, X_test, context):
    # Required: fit on X_train / y_train only; return scalar
    return float(y_train.mean())
```

Reference in recipe: `params: {model: mean_baseline, ...}`

### custom_preprocessor

```python
@mf.custom_preprocessor("demean_x")
def demean_x(X_train, y_train, X_test, context):
    # Fit column means on X_train; apply to both
    col_means = X_train.mean(axis=0)
    return X_train - col_means, X_test - col_means
```

Wire in recipe L2 leaf_config: `custom_postprocessor: demean_x`

### target_transformer

```python
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

Wire in recipe L1 leaf_config: `target_transformer: scale_by_std`

### custom_feature_block

```python
@mf.custom_feature_block("double_it", block_kind="temporal")
def double_it(frame, params):
    # frame: pd.DataFrame from upstream L3 node
    return frame * float(params.get("scale", 2.0))
```

Reference in L3 node: `op: double_it`

### custom_feature_combiner

```python
import pandas as pd

@mf.custom_feature_combiner("concat_blocks")
def concat_blocks(inputs, params):
    # inputs: list of pd.DataFrame from upstream nodes
    if isinstance(inputs, list):
        return pd.concat(inputs, axis=1)
    return inputs
```

Reference in L3 node: `op: concat_blocks`

---

## Fair-comparison checklist

Before comparing a custom extension point against built-in baselines:

- Keep L0 task axes (failure_policy, seed) identical across variants.
- Keep L1 raw-data treatment (same panel, same horizons) identical.
- Put representation changes in L2/L3, not inside a model closure.
- Put estimator changes in L4 (via `custom_model`), not inside a feature block.
- Use identical split, horizon, benchmark, and evaluation settings across all cells.
- Record all fitted choices in `fit_state` or `context` so the manifest is auditable.

---

## See also

- {doc}`add_custom_model` for the custom_model extension point in detail
- {doc}`target_transformer` for the target transformer contract and runtime gate
- {doc}`../tutorial/03_custom_model` for the narrative tutorial
