# Interpretation

[Back to reference](index.md)

`macroforecast.interpretation` owns post-fit interpretation helpers. These
functions consume fitted models and feature matrices. They do not fit models,
construct features, choose forecast windows, or run forecast-comparison tests.

Use the namespace form:

```python
import macroforecast as mf

fit = mf.models.ridge(X_train, y_train)
mf.interpretation.linear_coefficients(fit)
```

Every table returned by this module carries
`attrs["macroforecast_metadata_schema"]` with `kind`, `version`, `method`,
`model`, `n_features`, output `columns`, and function-specific metadata.

## Function Summary

| Function | Input | Output | Meaning |
| --- | --- | --- | --- |
| `linear_coefficients(model, sort=True)` | `ModelFit` or estimator with `coef_` | `DataFrame` | Native linear coefficient table. |
| `tree_importance(model, sort=True)` | `ModelFit` or estimator with `feature_importances_` | `DataFrame` | Native tree feature importance table. |
| `permutation_importance(model, X, y, metric="mse", n_repeats=5, random_state=None)` | fitted predictor, feature frame, target vector | `DataFrame` | Loss degradation after permuting each feature. |
| `partial_dependence(model, X, features, grid_size=20)` | fitted predictor and feature frame | `DataFrame` | One-way manual partial-dependence curves. |
| `accumulated_local_effect(model, X, feature, bins=10)` | fitted predictor and feature frame | `DataFrame` | First-order accumulated local effect curve. |
| `shap_values(model, X, background=None, explainer="auto", check_additivity=True, **kwargs)` | fitted predictor and feature frame | long `DataFrame` | SHAP attribution values using optional `shap` backend. |
| `custom_interpretation(model, X, func, y=None, name=None, **params)` | fitted predictor, feature frame, user callable | `DataFrame` | User-defined interpretation table with metadata attrs. |

## Native Model Interpretation

### linear_coefficients

```python
macroforecast.interpretation.linear_coefficients(model, *, sort=True)
```

Input: a `ModelFit` or estimator exposing `coef_`.

Output columns:

| Column | Meaning |
| --- | --- |
| `feature` | Feature name. |
| `coefficient` | Signed coefficient. |
| `abs_coefficient` | Absolute coefficient used for sorting. |

### tree_importance

```python
macroforecast.interpretation.tree_importance(model, *, sort=True)
```

Input: a `ModelFit` or estimator exposing `feature_importances_`.

Output columns:

| Column | Meaning |
| --- | --- |
| `feature` | Feature name. |
| `importance` | Native estimator importance. |

## Model-Agnostic Importance

### permutation_importance

```python
macroforecast.interpretation.permutation_importance(
    model,
    X,
    y,
    *,
    metric="mse",
    n_repeats=5,
    random_state=None,
)
```

Input: fitted predictor, feature `DataFrame`, and aligned target vector.

Output columns:

| Column | Meaning |
| --- | --- |
| `feature` | Permuted feature. |
| `importance` | Mean loss increase after permutation. |
| `std` | Standard deviation across repeats. |
| `baseline_loss` | Loss before permutation. |
| `n_repeats` | Repeat count. |

Supported metric names: `mse`, `mae`. A custom callable can also be supplied.

## Effect Curves

### partial_dependence

```python
macroforecast.interpretation.partial_dependence(
    model,
    X,
    *,
    features,
    grid_size=20,
)
```

Input: fitted predictor, feature `DataFrame`, and one feature or list of
features.

Output columns: `feature`, `value`, `prediction`.

### accumulated_local_effect

```python
macroforecast.interpretation.accumulated_local_effect(
    model,
    X,
    *,
    feature,
    bins=10,
)
```

Input: fitted predictor, feature `DataFrame`, and one feature name.

Output columns: `feature`, `bin`, `center`, `ale`, `local_effect`.

## SHAP Values

### shap_values

```python
macroforecast.interpretation.shap_values(
    model,
    X,
    *,
    background=None,
    explainer="auto",
    check_additivity=True,
    **kwargs,
)
```

Input: a fitted `ModelFit` or estimator plus a feature `DataFrame`. `background`
defaults to `X`. For stable out-of-sample interpretation, pass a training or
reference sample as `background`.

Output: long `DataFrame`, one row per `(observation, feature)`.

| Column | Meaning |
| --- | --- |
| `row` | Integer row position in `X`. |
| `index` | Original pandas index value. |
| `feature` | Feature name. |
| `feature_value` | Observed feature value. |
| `shap_value` | SHAP contribution. |
| `base_value` | SHAP base value when provided by the backend. |

Supported `explainer` values:

| Value | Backend behavior |
| --- | --- |
| `auto` | Uses `shap.Explainer` with the model prediction function and background frame. |
| `permutation` | Uses `shap.PermutationExplainer`. |
| `tree` | Uses `shap.TreeExplainer` on the native estimator. |

SHAP is optional:

```bash
pip install "macroforecast[interpretation]"
```

## Custom Interpretation

```python
macroforecast.interpretation.custom_interpretation(
    model,
    X,
    func,
    *,
    y=None,
    name=None,
    metadata=None,
    **params,
)
```

Input: a fitted `ModelFit` or predictor, feature `DataFrame`, optional target,
and a user callable.

Callable signature:

```python
func(model, X, *, y=None, metadata=None, **params)
```

Accepted callable outputs are `DataFrame`, `Series`, mapping, or a sequence
convertible to a `DataFrame`. The wrapper attaches:

| Attr | Meaning |
| --- | --- |
| `macroforecast_metadata_schema.kind` | Always `custom_interpretation`. |
| `macroforecast_metadata_schema.method` | `name` or callable name. |
| `macroforecast_metadata_schema.metadata.params` | User parameters passed to the callable. |
| `macroforecast_metadata_schema.metadata.user_metadata` | User-supplied metadata mapping. |

Example:

```python
def signed_mean_effect(model, X, *, y=None, metadata=None, scale=1.0):
    pred = model.predict(X)
    return {"signed_mean_prediction": float(pred.mean() * scale)}

custom = mf.interpretation.custom_interpretation(
    fit,
    X_test,
    signed_mean_effect,
    name="signed_mean_effect",
    scale=100.0,
)
```

## Examples

```python
fit = mf.models.ridge(X_train, y_train)

coef = mf.interpretation.linear_coefficients(fit)
perm = mf.interpretation.permutation_importance(
    fit,
    X_test,
    y_test,
    n_repeats=20,
    random_state=123,
)
pdp = mf.interpretation.partial_dependence(fit, X_test, features=["PAYEMS"])
ale = mf.interpretation.accumulated_local_effect(fit, X_test, feature="PAYEMS")
```

```python
tree = mf.models.random_forest(X_train, y_train, random_state=123)
shap_table = mf.interpretation.shap_values(
    tree,
    X_test,
    background=X_train,
    explainer="tree",
)
```
