# Models

[Back to reference](index.md)

`macroforecast.models` contains direct callable model fits. Each function
accepts pandas data, fits immediately, and returns a fitted result object with
`predict()`.

`lasso_path` is intentionally not a public model family. Use `lasso()` with a
chosen `alpha`, or use `get_model("lasso")` with `selection.select_params()`
to choose `alpha` from the lasso-owned search space.

## Return Objects

Model functions return `ModelFit`.

```python
macroforecast.models.ModelFit(
    estimator,
    model,
    feature_names=(),
    target_name=None,
    metadata={},
    diagnostics={},
)
```

| Attribute | Type | Description |
| --- | --- | --- |
| `estimator` | object | Fitted underlying estimator. |
| `model` | str | Canonical model name. |
| `feature_names` | tuple[str, ...] | Feature columns used at fit time. |
| `target_name` | str or `None` | Target name when available. |
| `metadata` | dict | Fit metadata such as `n_obs`, `alpha`, or tree budget. |
| `diagnostics` | dict | Model-specific fitted diagnostics that can be collected safely. |

`ModelFit.predict(X)` returns a `pandas.Series` named `"prediction"` and keeps
the index of the provided `X` when `X` is a DataFrame.

For custom fitted objects used in `forecasting.run(...)`, `predict(X_test)`
may return an array-like object or a pandas `Series`/single-column `DataFrame`.
Array-like output is treated positionally. Pandas output must either be indexed
exactly like `X_test.index` or use the default positional index
`RangeIndex(len(X_test))`. Any other index raises an error rather than silently
creating missing forecasts. The same DataFrame index rule applies to
`predict_quantiles(X_test)` when a fitted object exposes quantile forecasts.

`ModelFit.to_dict()` returns JSON-ready fit metadata. It records the canonical
model name, the underlying estimator class name, the fitted feature names, the
target name, `n_features`, the fit `metadata`, and JSON-ready `diagnostics`.
It does not serialize the fitted estimator itself.

```python
fit = macroforecast.models.ridge(X, y, alpha=0.5)
fit.to_dict()
```

Output shape:

```python
{
    "model": "ridge",
    "estimator": "sklearn.linear_model._ridge.Ridge",
    "feature_names": ["x1", "x2"],
    "target_name": "y",
    "n_features": 2,
    "metadata": {"n_obs": 120, "alpha": 0.5},
    "diagnostics": {
        "fitted_values": {"name": "fitted", "index": [...], "data": [...]},
        "residuals": {"name": "residual", "index": [...], "data": [...]},
        "metrics": {"n": 120, "mae": 0.04, "mse": 0.003, "rmse": 0.055},
        "coefficients": {"name": "coefficient", "index": ["x1", "x2"], "data": [...]},
        "selected_features": ["x1", "x2"],
    },
}
```

`ModelFit.to_metadata()` wraps the same block under `{"model": ..., "fit": ...}`
for downstream forecasting and result records.

Volatility functions return `VolatilityFit`, which extends `ModelFit` with
`predict_variance(horizon=1)` and `conditional_volatility`.

### Fit Persistence

`models` owns the low-level persistence format for fitted model objects.
Forecasting runners decide which fitted object should be saved and which
window, selection, and parameter metadata should be attached.

```python
saved = macroforecast.models.save_fit(
    fit,
    "trained_model/ridge/origin_0_h1_20000131.pkl",
    metadata={
        "alias": "ridge",
        "params": {"alpha": 0.1},
        "selection": selection_metadata,
    },
)
loaded = macroforecast.models.load_fit(saved.model_path)
```

| Function | Input | Output | Meaning |
| --- | --- | --- | --- |
| `save_fit(fit, model_path, metadata_path=None, metadata=None)` | Fitted model object and output paths. | `SavedModel` | Writes pickle plus JSON sidecar. |
| `load_fit(model_path)` | Pickle path. | fitted object | Loads a saved fit. |

`SavedModel.to_dict()` returns `model_path`, `metadata_path`, and
`save_error`. If a custom/local model cannot be pickled, `model_path` is
`None`, `save_error` records the failure, and the JSON sidecar is still
written. The sidecar always includes the available `fit.to_metadata()` block
when the fit exposes it.

### Fit Diagnostics

Diagnostics are collected on a best-effort basis. A model only records values
that the fitted backend exposes and that can be computed without changing the
fit. Missing keys mean the model does not expose that diagnostic, not that the
fit failed.

Common keys:

| Key | Recorded when | Meaning |
| --- | --- | --- |
| `fitted_values` | Estimator exposes `predict()` on the training matrix. | In-sample fitted values indexed like the aligned target. |
| `residuals` | `fitted_values` is available. | Training residuals, `y - fitted`. |
| `metrics` | `residuals` is available. | Residual count, mean, standard deviation, MAE, MSE, and RMSE. |
| `coefficients` | Estimator exposes `coef_`. | Coefficients indexed by feature name when possible. |
| `intercept` | Estimator exposes `intercept_`. | Scalar or list intercept. |
| `selected_features` | Nonzero coefficients or estimator selection metadata is available. | Selected feature names. |
| `feature_importance` | Estimator exposes `feature_importances_`. | Tree-style importances sorted descending. |
| `factor_loadings` | Estimator exposes factor loadings, loadings, or components. | Factor/PCA loading matrix when available. |
| `component_selected_features` | Estimator exposes component-level selection metadata. | Selected source features for supervised PCA-style components. |
| `training_history` | Estimator records iterative training history. | Epoch loss or backend-specific training trace. |
| `conditional_volatility` | Volatility estimator exposes fitted conditional volatility. | In-sample conditional volatility path. |
| `params` | Volatility estimator exposes fitted parameter estimates. | Fitted volatility-model parameters. |

Example:

```python
fit = macroforecast.models.random_forest(X, y, n_estimators=200)
fit.diagnostics["feature_importance"].head()
```

## Model Specs And Hyperparameter Spaces

Model functions fit immediately. Model specs are the selection objects:
they keep the fit callable together with model-owned defaults, tunable
parameters, and preset search spaces.

```python
model = macroforecast.models.get_model("lasso", preset="standard")
result = macroforecast.selection.select_params(
    model,
    X,
    y,
    window=macroforecast.window.expanding(min_train_size=120),
    metric=macroforecast.metrics.rmse,
)
fit = model(X, y, **result.best_params)
```

### ModelSpec

```python
macroforecast.models.ModelSpec(
    name,
    family,
    fit_func,
    default_params={},
    parameters=(),
    search_spaces={},
    default_search_method="grid",
    default_preset="standard",
    input_kind="supervised",
    preset="standard",
    params={},
    backend="internal",
    requires_extra=None,
    requires_scaling=False,
    recommended_preprocessing=(),
)
```

| Attribute | Type | Meaning |
| --- | --- | --- |
| `name` | str | Canonical model name. |
| `family` | str | Model family such as `linear`, `tree`, `factor`, or `volatility`. |
| `fit_func` | callable | Underlying fit function. |
| `default_params` | dict | Model-owned default keyword arguments. |
| `parameters` | tuple | `ModelParameter` descriptions. |
| `search_spaces` | dict | Preset-specific hyperparameter candidates. |
| `default_search_method` | str | Search method normally used for the model. |
| `default_preset` | str | Default hyperparameter preset. |
| `input_kind` | str | Input convention: `supervised`, `target`, `panel`, or `volatility`. |
| `preset` | str | Active search-space preset. |
| `params` | dict | User-fixed model parameters. |
| `backend` | str | Implementation backend, for example `sklearn.svm.SVR` or `torch.nn.LSTM`. |
| `requires_extra` | str or `None` | Optional dependency extra required to fit the model. |
| `requires_scaling` | bool | Whether the model is scale-sensitive and expects explicit preprocessing. |
| `recommended_preprocessing` | tuple[str, ...] | Short preprocessing notes attached to metadata. |

`ModelSpec` is callable:

```python
model = macroforecast.models.get_model("ridge", params={"alpha": 0.5})
fit = model(X, y)
```

`ModelSpec.to_dict()` returns a detailed JSON-ready specification including
defaults, fixed params, parameter descriptions, and all preset search spaces.
`ModelSpec.to_metadata()` returns the compact runner-facing block:

```python
{
    "model": "ridge",
    "model_family": "linear",
    "model_preset": "small",
    "input_kind": "supervised",
    "backend": "internal",
    "requires_extra": None,
    "requires_scaling": False,
    "recommended_preprocessing": [],
    "default_search_method": "cv_path",
    "default_params": {"alpha": 1.0},
    "params": {"alpha": 0.5},
    "search_space": {"alpha": [0.01, 0.1, 1.0]},
}
```

### get_model

```python
macroforecast.models.get_model(model, *, preset=None, params=None)
```

| Input | Type | Meaning |
| --- | --- | --- |
| `model` | str, callable, or `ModelSpec` | Model name, registered model callable, or existing spec. |
| `preset` | str or `None` | Search-space preset to attach. |
| `params` | dict or `None` | Fixed model parameters to attach. |

| Output | Type | Meaning |
| --- | --- | --- |
| return | `ModelSpec` | Callable model spec with model-owned defaults and spaces. |

### list_model_specs

```python
macroforecast.models.list_model_specs(family=None)
```

Returns a DataFrame with one row per registered model: `name`, `family`,
`input_kind`, `backend`, `requires_extra`, `requires_scaling`,
`recommended_preprocessing`, `default_search_method`, `default_preset`,
available `presets`, and `n_tunable`.

### describe_model

```python
macroforecast.models.describe_model(model)
```

Returns a DataFrame with parameter-level documentation and preset search
spaces.

Example:

| parameter | default | tunable | small_space | standard_space |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `True` | `(0.01, 0.1, 1.0)` | `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `max_iter` | `20000` | `False` | `None` | `None` |

### model_search_space

```python
macroforecast.models.model_search_space(model, *, preset=None)
```

Returns the model-owned candidate dictionary for the selected preset.

```python
macroforecast.models.model_search_space("random_forest", preset="small")
```

Output:

```python
{
    "n_estimators": (50, 100),
    "max_depth": (3, 5, None),
    "min_samples_leaf": (1, 3),
}
```

### Presets

| Preset | Purpose |
| --- | --- |
| `small` | Fast smoke tests and short interactive checks. |
| `standard` | Default analysis-scale search space. |
| `wide` | Larger search space for more expensive runs. |

## Input And Output Conventions

| Input kind | Callable shape | Use case |
| --- | --- | --- |
| `supervised` | `model(X, y, **params)` | Most regression, factor, tree, and ensemble models. |
| `target` | `model(y, **params)` | Univariate target-only models such as `ar`. |
| `panel` | `model(panel, **params)` | Multivariate time-series models such as `var`. |
| `volatility` | `model(y, X=None, **params)` | Return and volatility models. |

For supervised models, `X` may be a pandas DataFrame, a 2-D array, or a
`FeatureSet`. `y` may be a Series, 1-D array, or one-column DataFrame. If `X`
is a `FeatureSet`, `y` can be omitted.

All non-volatility model functions return `ModelFit`. Volatility functions
return `VolatilityFit`.

## Scaling Policy

The clean model API does not silently standardize predictors for models that
are traditionally scale-sensitive. Instead, those models advertise
`requires_scaling=True` through `ModelSpec`, `list_model_specs()`,
`search_spec()`, and `select_params()` metadata.

Current scale-sensitive callable models:

| Model | Backend | Scaling policy |
| --- | --- | --- |
| `svr` | `sklearn.svm.SVR` | Standardize predictors with `preprocessing.standardize_panel()` or a runner preprocessing spec before fitting. |
| `linear_svr` | `sklearn.svm.LinearSVR` | Standardize predictors before fitting. |
| `nu_svr` | `sklearn.svm.NuSVR` | Standardize predictors before fitting. |
| `nn` | `torch.nn.Sequential` | Standardizes `X` and `y` inside each fit window and maps predictions back to target units. |
| `transformer` | `torch.nn.TransformerEncoder` | Standardizes `X` and `y` inside each fit window and maps predictions back to target units. |
| `hemisphere_nn` | torch dual-head dense network | Standardizes `X` inside each fit window, fits mean and variance heads, and returns point, variance, and normal-approximation quantile forecasts. |

`nn`, `lstm`, `gru`, and `transformer` standardize `X` and `y` inside each fit
window and map predictions back to the target scale. `hemisphere_nn`
standardizes `X` and keeps the target in original units because its variance
head is a density-forecast object. Their metadata records
`requires_extra="deep"` and `requires_scaling=False`.

## Registered Model Catalog

| Model | Family | Input kind | Default search | Presets |
| --- | --- | --- | --- | --- |
| `ols` | linear | supervised | `grid` | none |
| `ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `nonneg_ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `shrink_to_target_ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `fused_difference_ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `random_walk_ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `lasso` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `elastic_net` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `adaptive_lasso` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `adaptive_elastic_net` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `group_lasso` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `sparse_group_lasso` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `bayesian_ridge` | linear | supervised | `grid` | none |
| `huber` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `kernel_ridge` | kernel | supervised | `random` | `small`, `standard`, `wide` |
| `knn` | nonparametric | supervised | `random` | `small`, `standard`, `wide` |
| `glmboost` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `svr` | support_vector | supervised | `random` | `small`, `standard`, `wide` |
| `linear_svr` | support_vector | supervised | `random` | `small`, `standard`, `wide` |
| `nu_svr` | support_vector | supervised | `random` | `small`, `standard`, `wide` |
| `nn` | neural | supervised | `random` | `small`, `standard`, `wide` |
| `lstm` | neural | supervised | `random` | `small`, `standard`, `wide` |
| `gru` | neural | supervised | `random` | `small`, `standard`, `wide` |
| `transformer` | neural | supervised | `random` | `small`, `standard`, `wide` |
| `hemisphere_nn` | neural | supervised | `random` | `small`, `standard`, `wide` |
| `pls` | composite | supervised | `grid` | `small`, `standard`, `wide` |
| `scaled_pca` | composite | supervised | `grid` | `small`, `standard`, `wide` |
| `supervised_pca` | composite | supervised | `grid` | `small`, `standard`, `wide` |
| `supervised_scaled_pca` | composite | supervised | `grid` | `small`, `standard`, `wide` |
| `ar` | timeseries | target | `grid` | `small`, `standard`, `wide` |
| `var` | timeseries | panel | `grid` | `small`, `standard`, `wide` |
| `bvar_minnesota` | timeseries | panel | `grid` | `small`, `standard`, `wide` |
| `bvar_normal_inverse_wishart` | timeseries | panel | `grid` | `small`, `standard`, `wide` |
| `ets` | timeseries | target | `grid` | none |
| `holt_winters` | timeseries | target | `grid` | none |
| `theta_method` | timeseries | target | `grid` | none |
| `dfm_mixed_mariano_murasawa` | mixed_frequency | panel | `grid` | `small`, `standard`, `wide` |
| `midas_almon` | mixed_frequency | supervised | `grid` | `small`, `standard`, `wide` |
| `midas_beta` | mixed_frequency | supervised | `grid` | `small`, `standard`, `wide` |
| `midas_step` | mixed_frequency | supervised | `grid` | `small`, `standard`, `wide` |
| `unrestricted_midas` | mixed_frequency | supervised | `grid` | `small`, `standard`, `wide` |
| `dfm_unrestricted_midas` | mixed_frequency | panel | `grid` | `small`, `standard`, `wide` |
| `far` | factor | supervised | `grid` | `small`, `standard`, `wide` |
| `favar` | factor | supervised | `grid` | `small`, `standard`, `wide` |
| `decision_tree` | tree | supervised | `grid` | `small`, `standard`, `wide` |
| `random_forest` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `extra_trees` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `gradient_boosting` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `mars` | spline | supervised | `random` | `small`, `standard`, `wide` |
| `xgboost` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `lightgbm` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `catboost` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `slow_growing_tree` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `quantile_regression_forest` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `bagging` | ensemble | supervised | `random` | `small`, `standard`, `wide` |
| `booging` | ensemble | supervised | `random` | `small`, `standard`, `wide` |
| `macro_random_forest` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `garch11` | volatility | volatility | `grid` | `small`, `standard`, `wide` |
| `egarch` | volatility | volatility | `grid` | `small`, `standard`, `wide` |
| `realized_garch` | volatility | volatility | `grid` | `small`, `standard`, `wide` |

## Linear Models

### ols

```python
macroforecast.models.ols(X, y)
```

Fits ordinary least squares.

| Item | Value |
| --- | --- |
| Input | `X`, `y` |
| Output | `ModelFit` |
| Default params | none |
| Tunable params | none |
| Preset search spaces | none |

### ridge

```python
macroforecast.models.ridge(X, y, *, alpha=1.0)
```

Fits ridge regression with an L2 penalty.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | L2 penalty strength. |

| Preset | `alpha` |
| --- | --- |
| `small` | `(0.01, 0.1, 1.0)` |
| `standard` | `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

Default selection method: `cv_path`.

### nonneg_ridge

```python
macroforecast.models.nonneg_ridge(X, y, *, alpha=1.0, fit_intercept=True)
```

Fits ridge regression with coefficients constrained to be non-negative. This
uses SciPy NNLS on an augmented ridge design, so it does not require `cvxpy`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | L2 penalty strength. |
| `fit_intercept` | `True` | fixed by preset | Fit an intercept outside the constrained coefficients. |

Default selection method: `cv_path`.

### shrink_to_target_ridge

```python
macroforecast.models.shrink_to_target_ridge(
    X,
    y,
    *,
    alpha=1.0,
    prior_target=None,
    simplex=False,
    nonneg=False,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-9,
)
```

Fits a ridge-type model where coefficients are shrunk toward a user-specified
target vector. `prior_target` can be a scalar, a sequence ordered like `X`
columns, or a mapping from column name to target coefficient. If
`prior_target=None`, the target is zero, except under `simplex=True`, where the
target is a uniform coefficient vector. `simplex=True` constrains coefficients
to sum to one and uses no intercept; `nonneg=True` also enforces non-negative
coefficients. The solver is SciPy SLSQP.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Strength of shrinkage toward `prior_target`. |
| `prior_target` | `None` | fixed by preset | Scalar, sequence, mapping, or `None`. |
| `simplex` | `False` | fixed by preset | Constrain coefficients to sum to one. |
| `nonneg` | `False` | fixed by preset | Constrain coefficients to be non-negative. |
| `fit_intercept` | `True` | fixed by preset | Fit an intercept unless `simplex=True`. |
| `max_iter` | `1000` | fixed by preset | SLSQP iteration cap. |
| `tol` | `1e-9` | fixed by preset | SLSQP tolerance. |

Default selection method: `cv_path`.

### fused_difference_ridge

```python
macroforecast.models.fused_difference_ridge(
    X,
    y,
    *,
    alpha=1.0,
    difference_order=1,
    mean_equality=False,
    nonneg=False,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-9,
)
```

Fits ridge regression with a finite-difference penalty on adjacent
coefficients. This is useful when columns have an ordered meaning such as lag
age, maturity, or horizon and neighboring coefficients should vary smoothly.
`difference_order=1` penalizes first differences; larger orders penalize higher
order coefficient curvature. `mean_equality=True` adds a conservation-style
constraint that the fitted and observed sums match and uses no intercept.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Strength of the smoothness penalty. |
| `difference_order` | `1` | fixed by preset | Finite-difference order applied to coefficients. |
| `mean_equality` | `False` | fixed by preset | Constrain fitted and observed sums to match. |
| `nonneg` | `False` | fixed by preset | Constrain coefficients to be non-negative. |
| `fit_intercept` | `True` | fixed by preset | Fit an intercept unless `mean_equality=True`. |
| `max_iter` | `1000` | fixed by preset | SLSQP iteration cap. |
| `tol` | `1e-9` | fixed by preset | SLSQP tolerance. |

Default selection method: `cv_path`.

### random_walk_ridge

```python
macroforecast.models.random_walk_ridge(
    X,
    y,
    *,
    alpha=1.0,
    initial_alpha=1.0,
    fit_intercept=True,
)
```

Fits a time-varying coefficient path with a random-walk penalty:

```text
sum_t (y_t - x_t beta_t)^2
+ initial_alpha * ||beta_1||^2
+ alpha * sum_t ||beta_t - beta_{t-1}||^2
```

Predictions use the final estimated coefficient vector. The full fitted path is
stored on the estimator as `coef_path_`, and standard `ModelFit` diagnostics
record the final coefficients, fitted values, and residuals.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Penalty on adjacent coefficient changes. |
| `initial_alpha` | `1.0` | fixed by preset | Penalty on the first coefficient vector. |
| `fit_intercept` | `True` | fixed by preset | Fit an intercept outside the time-varying coefficient path. |

Default selection method: `cv_path`.

### lasso

```python
macroforecast.models.lasso(X, y, *, alpha=1.0, max_iter=20000)
```

Fits lasso regression with an L1 penalty. There is no `lasso_path()` model
callable; use `get_model("lasso")` and `selection.select_params()`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | L1 penalty strength. |
| `max_iter` | `20000` | fixed by preset | Optimization iteration cap. |

| Preset | `alpha` |
| --- | --- |
| `small` | `(0.01, 0.1, 1.0)` |
| `standard` | `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

Default selection method: `cv_path`.

### elastic_net

```python
macroforecast.models.elastic_net(
    X,
    y,
    *,
    alpha=1.0,
    l1_ratio=0.5,
    max_iter=20000,
)
```

Fits elastic net regression.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Overall penalty strength. |
| `l1_ratio` | `0.5` | yes | L1 share of the elastic-net penalty. |
| `max_iter` | `20000` | fixed by preset | Optimization iteration cap. |

| Preset | `alpha` | `l1_ratio` |
| --- | --- | --- |
| `small` | `(0.01, 0.1, 1.0)` | `(0.25, 0.5, 0.75)` |
| `standard` | `(0.001, 0.01, 0.1, 1.0, 10.0)` | `(0.1, 0.25, 0.5, 0.75, 0.9)` |
| `wide` | `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` | `(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)` |

### adaptive_lasso

```python
macroforecast.models.adaptive_lasso(
    X,
    y,
    *,
    alpha=1.0,
    gamma=1.0,
    initial="ridge",
    initial_alpha=1.0,
    eps=1e-4,
    max_iter=20000,
    tol=1e-4,
    random_state=None,
)
```

Fits adaptive lasso. The model first estimates initial coefficients with
`initial="ridge"` or `initial="ols"`, builds feature weights
`1 / (abs(beta_init) + eps) ** gamma`, and fits lasso on weighted standardized
predictors. Predictions are mapped back to the original target scale.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Final adaptive lasso penalty strength. |
| `gamma` | `1.0` | yes | Exponent applied to initial coefficient weights. |
| `initial` | `"ridge"` | manual | Initial model: `"ridge"` or `"ols"`. |
| `initial_alpha` | `1.0` | fixed by preset | Initial ridge penalty. |
| `eps` | `1e-4` | fixed by preset | Small denominator floor for adaptive weights. |
| `max_iter` | `20000` | fixed by preset | Final solver iteration cap. |
| `tol` | `1e-4` | fixed by preset | Final solver convergence tolerance. |
| `random_state` | `None` | fixed by preset | Final solver random seed. |

| Preset | `alpha` | `gamma` |
| --- | --- | --- |
| `small` | `(0.01, 0.1, 1.0)` | `(1.0,)` |
| `standard` | `(0.001, 0.01, 0.1, 1.0, 10.0)` | `(0.5, 1.0, 2.0)` |
| `wide` | `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)` | `(0.5, 1.0, 1.5, 2.0)` |

### adaptive_elastic_net

```python
macroforecast.models.adaptive_elastic_net(
    X,
    y,
    *,
    alpha=1.0,
    l1_ratio=0.5,
    gamma=1.0,
    initial="ridge",
    initial_alpha=1.0,
    eps=1e-4,
    max_iter=20000,
    tol=1e-4,
    random_state=None,
)
```

Fits an adaptive elastic-net variant with the same initial coefficient weights
as `adaptive_lasso`, followed by an elastic-net fit on weighted standardized
predictors.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Final adaptive elastic-net penalty strength. |
| `l1_ratio` | `0.5` | yes | L1 share of the final elastic-net penalty. |
| `gamma` | `1.0` | yes | Exponent applied to initial coefficient weights. |
| `initial` | `"ridge"` | manual | Initial model: `"ridge"` or `"ols"`. |
| `initial_alpha` | `1.0` | fixed by preset | Initial ridge penalty. |
| `eps` | `1e-4` | fixed by preset | Small denominator floor for adaptive weights. |
| `max_iter` | `20000` | fixed by preset | Final solver iteration cap. |
| `tol` | `1e-4` | fixed by preset | Final solver convergence tolerance. |
| `random_state` | `None` | fixed by preset | Final solver random seed. |

| Preset | `alpha` | `l1_ratio` | `gamma` |
| --- | --- | --- | --- |
| `small` | `(0.01, 0.1, 1.0)` | `(0.25, 0.5, 0.75)` | `(1.0,)` |
| `standard` | `(0.001, 0.01, 0.1, 1.0, 10.0)` | `(0.1, 0.25, 0.5, 0.75, 0.9)` | `(0.5, 1.0, 2.0)` |
| `wide` | `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)` | `(0.05, 0.1, 0.25, 0.5, 0.75, 0.9)` | `(0.5, 1.0, 1.5, 2.0)` |

### group_lasso

```python
macroforecast.models.group_lasso(
    X,
    y,
    *,
    groups=None,
    alpha=1.0,
    group_weights=None,
    max_iter=5000,
    tol=1e-5,
    scale=True,
)
```

Fits group lasso with a package-native proximal-gradient solver. `groups`
must contain one label per predictor column. If `groups=None`, each predictor
is treated as its own group.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `groups` | `None` | manual | One group label per predictor. |
| `alpha` | `1.0` | yes | Group penalty strength. |
| `group_weights` | `None` | manual | Optional group penalty weights. |
| `max_iter` | `5000` | fixed by preset | Proximal-gradient iteration cap. |
| `tol` | `1e-5` | fixed by preset | Proximal-gradient convergence tolerance. |
| `scale` | `True` | fixed by preset | Whether to standardize predictors inside the model. |

| Preset | `alpha` |
| --- | --- |
| `small` | `(0.01, 0.1, 1.0)` |
| `standard` | `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)` |

### sparse_group_lasso

```python
macroforecast.models.sparse_group_lasso(
    X,
    y,
    *,
    groups=None,
    alpha=1.0,
    l1_ratio=0.5,
    group_weights=None,
    max_iter=5000,
    tol=1e-5,
    scale=True,
)
```

Fits sparse group lasso. `l1_ratio` controls the feature-level L1 share; the
remaining penalty share is applied at the group level.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `groups` | `None` | manual | One group label per predictor. |
| `alpha` | `1.0` | yes | Total sparse-group penalty strength. |
| `l1_ratio` | `0.5` | yes | Feature-level L1 share. |
| `group_weights` | `None` | manual | Optional group penalty weights. |
| `max_iter` | `5000` | fixed by preset | Proximal-gradient iteration cap. |
| `tol` | `1e-5` | fixed by preset | Proximal-gradient convergence tolerance. |
| `scale` | `True` | fixed by preset | Whether to standardize predictors inside the model. |

| Preset | `alpha` | `l1_ratio` |
| --- | --- | --- |
| `small` | `(0.01, 0.1, 1.0)` | `(0.25, 0.5, 0.75)` |
| `standard` | `(0.001, 0.01, 0.1, 1.0, 10.0)` | `(0.1, 0.25, 0.5, 0.75, 0.9)` |
| `wide` | `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)` | `(0.05, 0.1, 0.25, 0.5, 0.75, 0.9)` |

### bayesian_ridge

```python
macroforecast.models.bayesian_ridge(X, y)
```

Fits sklearn empirical-Bayes Bayesian ridge.

| Item | Value |
| --- | --- |
| Input | `X`, `y` |
| Output | `ModelFit` |
| Default params | sklearn defaults |
| Tunable params | none in the clean preset catalog |
| Preset search spaces | none |

### huber

```python
macroforecast.models.huber(X, y, *, epsilon=1.35, max_iter=1000)
```

Fits robust Huber regression.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `epsilon` | `1.35` | yes | Huber loss transition threshold. |
| `max_iter` | `1000` | fixed by preset | Optimization iteration cap. |

| Preset | `epsilon` |
| --- | --- |
| `small` | `(1.1, 1.35, 1.75)` |
| `standard` | `(1.1, 1.35, 1.5, 1.75, 2.0)` |
| `wide` | `(1.01, 1.1, 1.35, 1.5, 1.75, 2.0, 2.5)` |

## Kernel And Nonparametric Models

### kernel_ridge

```python
macroforecast.models.kernel_ridge(
    X,
    y,
    *,
    alpha=1.0,
    kernel="linear",
    gamma=None,
    degree=3,
    coef0=1.0,
)
```

Fits sklearn kernel ridge regression. This model is scale-sensitive for
nonlinear kernels, so standardize predictors before `rbf`, `poly`, or
`sigmoid` kernels.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Ridge penalty strength. |
| `kernel` | `"linear"` | search option | Kernel name. |
| `gamma` | `None` | search option | Kernel coefficient. |
| `degree` | `3` | search option | Polynomial kernel degree. |
| `coef0` | `1.0` | fixed by preset | Independent term for polynomial/sigmoid kernels. |

### knn

```python
macroforecast.models.knn(
    X,
    y,
    *,
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
    p=2,
)
```

Fits sklearn k-nearest-neighbor regression. This is distance-based and should
usually receive standardized predictors.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_neighbors` | `5` | yes | Number of nearest neighbors. |
| `weights` | `"uniform"` | yes | `"uniform"` or `"distance"`. |
| `metric` | `"minkowski"` | fixed by preset | Distance metric. |
| `p` | `2` | search option | Minkowski distance order. |

## Linear Boosting

### glmboost

```python
macroforecast.models.glmboost(X, y, *, n_iter=100, learning_rate=0.1)
```

Fits componentwise L2 boosting with linear base learners.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_iter` | `100` | yes | Number of boosting iterations. |
| `learning_rate` | `0.1` | yes | Shrinkage applied to each update. |

| Preset | `n_iter` | `learning_rate` |
| --- | --- | --- |
| `small` | `(50, 100)` | `(0.05, 0.1)` |
| `standard` | `(50, 100, 200, 500)` | `(0.01, 0.05, 0.1)` |
| `wide` | `(50, 100, 200, 500, 1000)` | `(0.005, 0.01, 0.05, 0.1, 0.2)` |

## Support-Vector Models

Support-vector models are sklearn-backed and live in the base dependency set.
They are useful when nonlinear margins or robust epsilon-insensitive losses
are preferred over a pure least-squares fit. The forecasting runner treats
them as ordinary supervised models: call `model(X, y, **params)`, tune only
model-owned hyperparameters through `selection`, and let `window` decide the
train/validation/test dates.

Forecasting-runner example:

```python
pre = macroforecast.preprocessing.preprocess_spec(
    transform="none",
    outliers="none",
    impute="mean",
    standardize="zscore",
    standardize_columns="predictors",
)
features = macroforecast.feature_engineering.feature_spec(
    target="y",
    horizon=1,
    predictors=["x1", "x2"],
    lags=(0, 1),
)
result = macroforecast.forecasting.run(
    panel,
    "svr",
    preprocessing=pre,
    features=features,
    window=macroforecast.window.last_block(validation_size=24),
    selection=macroforecast.selection.grid({"C": [0.1, 1.0], "epsilon": [0.01, 0.1]}),
)
```

### svr

```python
macroforecast.models.svr(
    X,
    y,
    *,
    kernel="rbf",
    C=1.0,
    epsilon=0.1,
    gamma="scale",
    degree=3,
    coef0=0.0,
    shrinking=True,
    tol=1e-3,
    cache_size=200.0,
    max_iter=-1,
)
```

Fits sklearn `SVR`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `kernel` | `"rbf"` | fixed by preset | Kernel: `"linear"`, `"poly"`, `"rbf"`, `"sigmoid"`, or `"precomputed"`. |
| `C` | `1.0` | yes | Inverse regularization strength. |
| `epsilon` | `0.1` | yes | Epsilon-insensitive tube width. |
| `gamma` | `"scale"` | yes | Kernel coefficient for RBF/poly/sigmoid kernels. |
| `degree` | `3` | fixed by preset | Polynomial kernel degree. |
| `coef0` | `0.0` | fixed by preset | Independent term for poly/sigmoid kernels. |
| `shrinking` | `True` | fixed by preset | Whether to use the shrinking heuristic. |
| `tol` | `1e-3` | fixed by preset | Optimization tolerance. |
| `cache_size` | `200.0` | fixed by preset | Kernel cache size in MB. |
| `max_iter` | `-1` | fixed by preset | Solver iteration cap; `-1` means no cap. |

| Preset | `C` | `epsilon` | `gamma` |
| --- | --- | --- | --- |
| `small` | `(0.1, 1.0)` | `(0.01, 0.1)` | `("scale",)` |
| `standard` | `(0.1, 1.0, 10.0)` | `(0.01, 0.1, 0.2)` | `("scale", "auto")` |
| `wide` | `(0.01, 0.1, 1.0, 10.0, 100.0)` | `(0.001, 0.01, 0.1, 0.2)` | `("scale", "auto")` |

### linear_svr

```python
macroforecast.models.linear_svr(
    X,
    y,
    *,
    C=1.0,
    epsilon=0.0,
    loss="epsilon_insensitive",
    tol=1e-4,
    max_iter=10000,
    random_state=0,
)
```

Fits sklearn `LinearSVR`. Use this when a linear support-vector loss is wanted
without kernel overhead.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `C` | `1.0` | yes | Inverse regularization strength. |
| `epsilon` | `0.0` | yes | Epsilon-insensitive tube width. |
| `loss` | `"epsilon_insensitive"` | fixed by preset | LinearSVR loss function. |
| `tol` | `1e-4` | fixed by preset | Optimization tolerance. |
| `max_iter` | `10000` | fixed by preset | Solver iteration cap. |
| `random_state` | `0` | fixed by preset | Random seed. |

| Preset | `C` | `epsilon` |
| --- | --- | --- |
| `small` | `(0.1, 1.0)` | `(0.0, 0.1)` |
| `standard` | `(0.01, 0.1, 1.0, 10.0)` | `(0.0, 0.01, 0.1)` |
| `wide` | `(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` | `(0.0, 0.001, 0.01, 0.1, 0.2)` |

### nu_svr

```python
macroforecast.models.nu_svr(
    X,
    y,
    *,
    kernel="rbf",
    C=1.0,
    nu=0.5,
    gamma="scale",
    degree=3,
    coef0=0.0,
    shrinking=True,
    tol=1e-3,
    cache_size=200.0,
    max_iter=-1,
)
```

Fits sklearn `NuSVR`, where `nu` controls the admissible training-error and
support-vector fractions.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `kernel` | `"rbf"` | fixed by preset | Kernel: `"linear"`, `"poly"`, `"rbf"`, `"sigmoid"`, or `"precomputed"`. |
| `C` | `1.0` | yes | Inverse regularization strength. |
| `nu` | `0.5` | yes | Upper/lower control for training-error and support-vector fractions. |
| `gamma` | `"scale"` | yes | Kernel coefficient for RBF/poly/sigmoid kernels. |
| `degree` | `3` | fixed by preset | Polynomial kernel degree. |
| `coef0` | `0.0` | fixed by preset | Independent term for poly/sigmoid kernels. |
| `shrinking` | `True` | fixed by preset | Whether to use the shrinking heuristic. |
| `tol` | `1e-3` | fixed by preset | Optimization tolerance. |
| `cache_size` | `200.0` | fixed by preset | Kernel cache size in MB. |
| `max_iter` | `-1` | fixed by preset | Solver iteration cap; `-1` means no cap. |

| Preset | `C` | `nu` | `gamma` |
| --- | --- | --- | --- |
| `small` | `(0.1, 1.0)` | `(0.25, 0.5)` | `("scale",)` |
| `standard` | `(0.1, 1.0, 10.0)` | `(0.25, 0.5, 0.75)` | `("scale", "auto")` |
| `wide` | `(0.01, 0.1, 1.0, 10.0, 100.0)` | `(0.1, 0.25, 0.5, 0.75, 0.9)` | `("scale", "auto")` |

## Neural Models

`nn`, `lstm`, `gru`, `transformer`, and `hemisphere_nn` are all torch-backed neural-network models and require
`macroforecast[deep]`. `nn` is the feed-forward neural network for tabular
feature matrices; `lstm` and `gru` are recurrent neural networks that consume
trailing row sequences; `transformer` is a compact Transformer encoder using
the same trailing-row sequence contract. `hemisphere_nn` is a bagged dual-head
network for mean and variance forecasts. The `deep` extra is intentionally separate from
`macroforecast[all]` because torch is large and platform-sensitive.

Torch recurrent example:

```python
result = macroforecast.forecasting.run(
    panel,
    "lstm",
    features=features,
    window=macroforecast.window.last_block(validation_size=24),
    params={"lstm": {"sequence_length": 4, "hidden_size": 32, "device": "auto"}},
    selection={"lstm": None},
)
```

### nn

```python
macroforecast.models.nn(
    X,
    y,
    *,
    hidden_layer_sizes=(100,),
    activation="relu",
    dropout=0.0,
    learning_rate=0.001,
    max_epochs=100,
    batch_size=32,
    weight_decay=0.0,
    optimizer="adam",
    loss="mse",
    random_state=0,
    device="auto",
)
```

Fits a torch-backed feed-forward neural-network regressor. The estimator
standardizes `X` and `y` inside each fit window and maps predictions back to
target units. Use feature engineering for lagged, rolling, PCA, or MARX-style
inputs before fitting this model.

Forecasting-runner example:

```python
result = macroforecast.forecasting.run(
    panel,
    "nn",
    features=features,
    window=macroforecast.window.last_block(validation_size=24),
    params={"nn": {"max_epochs": 100, "device": "auto"}},
    selection=macroforecast.selection.grid({
        "hidden_layer_sizes": [(32,), (64,)],
        "weight_decay": [0.0, 0.0001],
    }),
)
```

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `hidden_layer_sizes` | `(100,)` | yes | Feed-forward hidden layer widths. |
| `activation` | `"relu"` | fixed by preset | Activation: `"identity"`, `"logistic"`, `"tanh"`, or `"relu"`. |
| `dropout` | `0.0` | yes | Dropout rate between hidden layers. |
| `learning_rate` | `0.001` | yes | Optimizer learning rate. |
| `max_epochs` | `100` | fixed by preset | Training epoch cap. |
| `batch_size` | `32` | fixed by preset | Mini-batch size. |
| `weight_decay` | `0.0` | yes | L2 weight decay. |
| `optimizer` | `"adam"` | fixed by preset | Torch optimizer: `"adam"`, `"sgd"`, or `"rmsprop"`. |
| `loss` | `"mse"` | fixed by preset | Torch loss: `"mse"` or `"huber"`. |
| `random_state` | `0` | fixed by preset | Random seed. |
| `device` | `"auto"` | fixed by preset | Torch device: `"auto"`, `"cpu"`, or `"cuda"`. |

| Preset | `hidden_layer_sizes` | `dropout` | `learning_rate` | `weight_decay` |
| --- | --- | --- | --- | --- |
| `small` | `((32,), (64,))` | `(0.0,)` | `(0.001,)` | `(0.0, 0.0001)` |
| `standard` | `((64,), (100,), (64, 32))` | `(0.0, 0.1)` | `(0.0005, 0.001)` | `(0.0, 0.0001, 0.001)` |
| `wide` | `((32,), (64,), (100,), (128,), (100, 50), (128, 64))` | `(0.0, 0.1, 0.25)` | `(0.0001, 0.0005, 0.001, 0.005)` | `(0.0, 0.00001, 0.0001, 0.001, 0.01)` |

### lstm

```python
macroforecast.models.lstm(
    X,
    y,
    *,
    sequence_length=4,
    hidden_size=32,
    num_layers=1,
    dropout=0.0,
    learning_rate=0.001,
    max_epochs=100,
    batch_size=32,
    random_state=0,
    device="auto",
)
```

Fits a compact torch-backed LSTM regressor. `sequence_length` controls how
many trailing rows are passed to the recurrent network for each target date.
The fitted estimator stores the trailing training rows, so `predict(X_test)`
can create the first test sequences without the caller manually prepending
training history. The backend is a regular `torch.nn.Module`, switches to
`train()` during fitting and `eval()` during prediction, and uses `device` to
choose CPU or CUDA. The fit diagnostics include `sequence_context`, recording
`sequence_length`, `fit_sample_size`, `train_tail_rows`, and the
`test_sequence_prefix` policy. The prefix is always the last fitted rows only,
so the forecasting runner can pass the test feature block directly without
leaking future rows.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `sequence_length` | `4` | yes | Trailing rows per recurrent sequence. |
| `hidden_size` | `32` | yes | Recurrent hidden-state width. |
| `num_layers` | `1` | fixed by preset | Number of recurrent layers. |
| `dropout` | `0.0` | fixed by preset | Dropout between recurrent layers. |
| `learning_rate` | `0.001` | yes | Adam learning rate. |
| `max_epochs` | `100` | fixed by preset | Training epoch cap. |
| `batch_size` | `32` | fixed by preset | Mini-batch size. |
| `random_state` | `0` | fixed by preset | Random seed. |
| `device` | `"auto"` | fixed by preset | Torch device: `"auto"`, `"cpu"`, or `"cuda"`. |

| Preset | `sequence_length` | `hidden_size` | `learning_rate` |
| --- | --- | --- | --- |
| `small` | `(2, 4)` | `(16, 32)` | `(0.001,)` |
| `standard` | `(2, 4, 8)` | `(16, 32, 64)` | `(0.0005, 0.001)` |
| `wide` | `(2, 4, 8, 12)` | `(16, 32, 64, 128)` | `(0.0001, 0.0005, 0.001, 0.005)` |

### gru

```python
macroforecast.models.gru(
    X,
    y,
    *,
    sequence_length=4,
    hidden_size=32,
    num_layers=1,
    dropout=0.0,
    learning_rate=0.001,
    max_epochs=100,
    batch_size=32,
    random_state=0,
    device="auto",
)
```

Fits a compact torch-backed GRU regressor with the same input/output contract
as `lstm`.

### transformer

```python
macroforecast.models.transformer(
    X,
    y,
    *,
    sequence_length=4,
    hidden_size=32,
    num_layers=1,
    dropout=0.0,
    learning_rate=0.001,
    max_epochs=100,
    batch_size=32,
    random_state=0,
    device="auto",
)
```

Fits a compact torch-backed Transformer encoder regressor. The input/output
contract matches `lstm` and `gru`: rows are standardized inside each fit
window, trailing sequences are built from the fitted sample, and predictions
for new rows are mapped back to target units. `hidden_size` is the
Transformer feed-forward width, not the input dimension; the encoder uses
`d_model = n_features` and `nhead=1` to keep the public callable small and
stable for macro panels.

### hemisphere_nn

```python
macroforecast.models.hemisphere_nn(
    X,
    y,
    *,
    lc=2,
    lm=2,
    lv=2,
    neurons=64,
    dropout=0.2,
    learning_rate=0.001,
    max_epochs=100,
    n_estimators=100,
    subsample=0.8,
    nu=None,
    variance_penalty=1.0,
    patience=15,
    validation_fraction=0.2,
    random_state=0,
    device="auto",
    quantile_levels=(0.05, 0.5, 0.95),
)
```

Fits a compact Hemisphere neural network inspired by Goulet Coulombe,
Frenette, and Klieber's dual-head density-forecast architecture. The network
has a shared common core, a mean head, and a positive variance head. The loss
is Gaussian negative log likelihood plus a soft variance-emphasis penalty:

```text
mean((y - h_m(X))^2 / h_v(X) + log h_v(X))
+ variance_penalty * (mean(h_v(X)) - nu * var(y))^2 / var(y)^2
```

`predict()` returns the ensemble mean forecast. The fitted estimator also
exposes `predict_variance(X)`, `predict_distribution(X)`, and
`predict_quantiles(X, levels=None)`. The forecasting runner stores the variance
and quantile outputs in `variance_prediction` and `quantile_predictions`. The public
callable accepts legacy aliases `lr`, `n_epochs`, `B`, `sub_rate`,
`lambda_emphasis`, and `val_frac`; normalized metadata records them as
`learning_rate`, `max_epochs`, `n_estimators`, `subsample`,
`variance_penalty`, and `validation_fraction`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `lc` | `2` | fixed by preset | Shared common-core depth. |
| `lm` | `2` | fixed by preset | Mean-head depth after the common core. |
| `lv` | `2` | fixed by preset | Variance-head depth after the common core. |
| `neurons` | `64` | yes | Hidden width. |
| `dropout` | `0.2` | fixed by preset | Dropout rate. |
| `learning_rate` | `0.001` | yes | Adam learning rate. |
| `max_epochs` | `100` | fixed by preset | Training epoch cap. |
| `n_estimators` | `100` | yes | Number of blocked-subsample bags. |
| `subsample` | `0.8` | fixed by preset | Blocked-subsample fraction. |
| `nu` | `None` | fixed by preset | Variance-emphasis target ratio; `None` uses `0.5`. |
| `variance_penalty` | `1.0` | fixed by preset | Soft penalty on the variance-emphasis target. |
| `patience` | `15` | fixed by preset | Early-stopping patience. |
| `validation_fraction` | `0.2` | fixed by preset | Chronological validation fraction. |
| `device` | `"auto"` | fixed by preset | Torch device. |
| `quantile_levels` | `(0.05, 0.5, 0.95)` | fixed by preset | Default normal-approximation quantile levels returned by `predict_quantiles()`. |

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `sequence_length` | `4` | yes | Trailing rows per recurrent sequence. |
| `hidden_size` | `32` | yes | Recurrent hidden-state width. |
| `num_layers` | `1` | fixed by preset | Number of recurrent layers. |
| `dropout` | `0.0` | fixed by preset | Dropout between recurrent layers. |
| `learning_rate` | `0.001` | yes | Adam learning rate. |
| `max_epochs` | `100` | fixed by preset | Training epoch cap. |
| `batch_size` | `32` | fixed by preset | Mini-batch size. |
| `random_state` | `0` | fixed by preset | Random seed. |
| `device` | `"auto"` | fixed by preset | Torch device: `"auto"`, `"cpu"`, or `"cuda"`. |

The GRU presets match the LSTM presets.

## Factor And Time-Series Models

### pls

```python
macroforecast.models.pls(
    X,
    y,
    *,
    n_components=3,
    scale=True,
    max_iter=500,
    tol=1e-6,
)
```

Fits partial least squares regression. Unlike unsupervised PCA, PLS uses the
target while constructing latent components, so it belongs in `models` rather
than `preprocessing` or `feature_engineering`.

`n_components` is treated as a requested upper bound. At fit time, the model
resolves it to `min(requested, n_predictors, n_observations)` so the default is
safe for small feature sets. Metadata records both `requested_n_components` and
`resolved_n_components`; `n_components` stores the resolved value used by
`sklearn.cross_decomposition.PLSRegression`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_components` | `3` | yes | Requested maximum number of latent PLS components. |
| `scale` | `True` | fixed by preset | Whether to scale predictors inside PLS. |
| `max_iter` | `500` | fixed by preset | NIPALS iteration cap. |
| `tol` | `1e-6` | fixed by preset | NIPALS convergence tolerance. |

| Preset | `n_components` |
| --- | --- |
| `small` | `(1, 2, 3)` |
| `standard` | `(1, 2, 3, 5, 8)` |
| `wide` | `(1, 2, 3, 5, 8, 10, 12, 20)` |

### scaled_pca

```python
macroforecast.models.scaled_pca(
    X,
    y,
    *,
    n_components=3,
    scale=True,
    control_columns=None,
    include_constant=True,
    drop_control_columns=True,
    winsorize_slopes=None,
)
```

Fits Huang, Jiang, Li, Tong, and Zhou scaled PCA (sPCA) with a linear
forecast head. The factor extraction step follows the original `spcaest.m`
contract: standardize predictors, estimate one marginal predictive slope for
each predictor, scale each standardized predictor by that slope, then run PCA
on the scaled panel.

Mathematical contract:

Let $X \in \mathbb{R}^{T \times N}$ be the model-window predictor matrix and
$y \in \mathbb{R}^{T}$ be the target. With `scale=True`, each predictor is
standardized inside the active model window using MATLAB's default sample
standard deviation convention:

$$
X^s_{tj} = \frac{X_{tj}-\bar X_j}{s_j}.
$$

For each predictor, estimate the marginal predictive slope from an intercept
regression:

$$
\hat\beta_j =
\frac{\sum_{t=1}^{T}(X^s_{tj}-\bar X^s_j)(y_t-\bar y)}
{\sum_{t=1}^{T}(X^s_{tj}-\bar X^s_j)^2}.
$$

Build the scaled panel:

$$
X^{\mathrm{sPCA}}_{tj} = X^s_{tj}\hat\beta_j.
$$

Then compute principal components with Huang's normalization
$\hat F'\hat F/T = I$:

$$
X^{\mathrm{sPCA}}X^{\mathrm{sPCA}\prime}
= UDU',
\qquad
\hat F = \sqrt{T}\,U_{[:,1:K]}.
$$

For forecasting, `macroforecast` regresses the target residual after optional
controls on these factors, then projects new scaled observations into the
same factor space. This forecast head is the package wrapper; the factor
extraction itself matches Huang's `spcaest.m` design.

Original-code match:

| Huang `spcaest.m` step | `macroforecast` implementation |
| --- | --- |
| `Xs = standard(X)` | model-window predictor standardization with `ddof=1` |
| `xvar = [ones(T,1) Xs(:,j)]` | `_marginal_slopes(factor_values, y_values)` |
| `parm = xvar\target` and `beta(j)=parm(2)` | closed-form marginal slope stored in `scaling_slopes_` |
| `scaleXs(:,j)=Xs(:,j)*beta(j)` | `scaled_values = factor_values * slopes` |
| `pc_T(scaleXs,nfac)` | `_huang_scaled_pca_state(...)` |
| `fhat=Fhat0(:,1:nfac)*sqrt(T)` | `factor_scores_` with `F'F/T=I` |

Scaling note: Huang's `spcaest.m` standardizes only `X` before estimating the
marginal slopes, while the target stays in its raw units. Therefore
`scaling_slopes_` in `scaled_pca` is in target-scale units. This differs from
the Hounyo-Li macro SsPCA code below, where both `X` and the target are
standardized before the slope step. The two slope vectors can differ by the
target scale, but that is a global scalar difference for the factor/forecast
structure; the practical scaling logic is the same once predictions are
mapped back to the target units.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_components` | `3` | yes | Number of Huang scaled-PCA factors. |
| `scale` | `True` | fixed by preset | Whether to standardize predictors inside the model. |
| `control_columns` | `None` | fixed by preset | Optional X columns used as forecasting controls. |
| `include_constant` | `True` | fixed by preset | Whether to include a constant in the control block. |
| `drop_control_columns` | `True` | fixed by preset | Whether controls are excluded from the PCA block. |
| `winsorize_slopes` | `None` | fixed by preset | Optional percentile winsorization for scaling slopes. |

The presets tune `n_components`. Inspect exact candidate lists with
`describe_model("scaled_pca")`.

### supervised_pca

```python
macroforecast.models.supervised_pca(
    X,
    y,
    *,
    n_components=3,
    n_selected=50,
    min_abs_corr=0.0,
    scale=True,
    control_columns=None,
    include_constant=True,
    drop_control_columns=True,
    preselect="none",
    t_threshold=1.28,
    elastic_net_alpha=0.0002,
    elastic_net_l1_ratio=0.5,
    random_state=0,
)
```

Fits original-style supervised PCA (SPCA). The implementation follows the
MATLAB reproducibility code structure from Hounyo and Li's IJF weak-factor
package: residualize the target on optional controls, rank the current
predictors by absolute correlation with the current target residual, select
`n_selected` predictors, extract one SVD factor, project both target and
predictor residuals, then repeat for `n_components`.

This is different from `feature_engineering.pca_features()`, which is
unsupervised and belongs in `macroforecast.feature_engineering`. Use this model when
the target is intentionally allowed to guide component construction inside
each model fit window.

Mathematical contract:

Let $X \in \mathbb{R}^{T \times N}$ be the model-window predictor matrix,
$y \in \mathbb{R}^{T}$ be the target, and $W \in \mathbb{R}^{T \times c}$ be
the optional control block. With `scale=True`, each predictor and the target
are standardized inside the active model window using the same sample standard
deviation convention as MATLAB `std(...,0,dim)`.

First residualize the target on controls. The paper code writes this with an
ordinary inverse; `macroforecast` uses the Moore-Penrose inverse for numerical
stability when the control block is singular or nearly singular.

$$
\hat a_W = W^{+}y, \qquad r_y^{(0)} = y - W\hat a_W,
\qquad R_X^{(0)} = X.
$$

For component $k = 1,\ldots,K$, compute residual correlations, select a
subset $I_k$, extract one SVD loading, and project:

$$
c_j^{(k)} =
\left|\operatorname{corr}\left(R_{X,j}^{(k-1)}, r_y^{(k-1)}\right)\right|,
\qquad
I_k = \operatorname{top}_{q}\{c_j^{(k)}\}_{j=1}^{N}.
$$

$$
u_k =
\operatorname{first\ left\ singular\ vector}
\left(R_{X,I_k}^{(k-1)\prime}\right), \qquad
\ell_{k,j}=0\ \text{for}\ j \notin I_k.
$$

$$
f_k = R_X^{(k-1)}\ell_k,\qquad
\hat\alpha_k = \frac{r_y^{(k-1)\prime}f_k}{f_k'f_k},\qquad
\hat\lambda_k = \frac{R_X^{(k-1)\prime}f_k}{f_k'f_k}.
$$

$$
r_y^{(k)} = r_y^{(k-1)}-\hat\alpha_k f_k,\qquad
R_X^{(k)} = R_X^{(k-1)}-f_k\hat\lambda_k'.
$$

Prediction for a new row $x_*$ and controls $w_*$ is:

$$
\hat y_* =
w_*\hat a_W
+ x_*\left(\sum_{k=1}^{K}\hat\alpha_k\ell_k'\right).
$$

Original-code match:

| MATLAB variable / step | `macroforecast` implementation |
| --- | --- |
| `alphawhat = yt_insample*wt_insample'*inv(wt_insample*wt_insample')` | `_least_squares_coef(control_values, y_values)` |
| `COR = abs(corr(xt0', ytplush0'))` | `_absolute_correlations(work_x, work_y)` |
| `idx_sorted(1:N1)` | `_selected_indices(..., n_selected=q)` |
| `[U,~,~] = svds(xt0(II,:),1)` | `np.linalg.svd(work_x[:, selected], ...)` and first right singular vector |
| `Fhat = leftvector * xt0` | `factor = work_x @ loading` |
| `alphahat = ytplush0*Fhat' / (Fhat*Fhat')` | `alpha = work_y @ factor / (factor @ factor)` |
| `lambdahat = xt0*Fhat' / (Fhat*Fhat')` | `lambdas = work_x.T @ factor / (factor @ factor)` |
| `ytplush0 = ytplush0 - alphahat*Fhat` | `work_y = work_y - alpha * factor` |
| `xt0 = xt0 - lambdahat * Fhat` | `work_x = work_x - np.outer(factor, lambdas)` |

Verification status: unit tests include a compact MATLAB-style reference
recursion for both SPCA and SsPCA and compare generated predictions against
`models.supervised_pca()` and `models.supervised_scaled_pca()`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_components` | `3` | yes | Number of sequential supervised components. |
| `n_selected` | `50` | yes | Predictors selected at each SPCA step. |
| `min_abs_corr` | `0.0` | yes | Minimum absolute residual correlation retained before PCA. |
| `scale` | `True` | fixed by preset | Whether to standardize predictors and target inside the model. |
| `control_columns` | `None` | fixed by preset | Optional X columns used as forecasting controls. |
| `include_constant` | `True` | fixed by preset | Whether to include a constant in the control block. |
| `drop_control_columns` | `True` | fixed by preset | Whether controls are excluded from the PCA block. |
| `preselect` | `"none"` | fixed by preset | Optional pre-selection: `"none"`, `"hard_tstat"`, or `"elastic_net"`. |
| `t_threshold` | `1.28` | fixed by preset | Hard t-stat pre-selection threshold. |
| `elastic_net_alpha` | `0.0002` | fixed by preset | Elastic-net pre-selection penalty. |
| `elastic_net_l1_ratio` | `0.5` | fixed by preset | Elastic-net pre-selection L1 ratio. |
| `random_state` | `0` | fixed by preset | Elastic-net pre-selection random seed. |

The presets tune `n_components`, `n_selected`, and `min_abs_corr`.
Inspect exact candidate lists with `describe_model("supervised_pca")`.

### supervised_scaled_pca

```python
macroforecast.models.supervised_scaled_pca(
    X,
    y,
    *,
    n_components=3,
    n_selected=50,
    min_abs_corr=0.0,
    scale=True,
    control_columns=None,
    include_constant=True,
    drop_control_columns=True,
    preselect="none",
    t_threshold=1.28,
    elastic_net_alpha=0.0002,
    elastic_net_l1_ratio=0.5,
    random_state=0,
)
```

Fits Hounyo-Li supervised scaled PCA (SsPCA). This adds the paper's
predictive-slope scaling step before the SPCA loop: each standardized
predictor is first multiplied by its marginal predictive slope for the target.
The scaled panel is then passed through the same iterative supervised
selection, SVD factor extraction, and projection loop as `supervised_pca`.

Mathematical contract:

After the within-window standardization used above, estimate one marginal
predictive slope per predictor:

$$
\hat\gamma_j =
\frac{\sum_{t=1}^{T}(x_{tj}-\bar x_j)(y_t-\bar y)}
{\sum_{t=1}^{T}(x_{tj}-\bar x_j)^2}.
$$

Build the supervised-scaled panel

$$
X^{\mathrm{scaled}}_j = \hat\gamma_j X_j,
\qquad j=1,\ldots,N.
$$

Then run the same SPCA recursion as `supervised_pca` with
$R_X^{(0)} = X^{\mathrm{scaled}}$. The forecast is therefore

$$
\hat y_* =
w_*\hat a_W
+ x_*^{\mathrm{scaled}}
\left(\sum_{k=1}^{K}\hat\alpha_k\ell_k'\right).
$$

This corresponds to Hounyo-Li SsPCA as implemented in the local MATLAB
package: `scaledPCA_emp002.m` supplies the predictive-slope scaling idea,
`SPCA_emp002.m` supplies the supervised selection/projection recursion, and
`SsPCA_emp002.m` combines the two by applying SPCA to `scaleXs`.

Source checked: the local MATLAB reproducibility package for Hounyo and Li,
`SsPCA_emp002.m`, `SsPCA_tune.m`, `SPCA_emp002.m`, `scaledPCA_emp002.m`, and
`inflation_linear_tune.m`. The Python implementation is a clean port of the
algorithmic contract, not copied MATLAB code.

Original-code match for the scaling step:

| MATLAB variable / step | `macroforecast` implementation |
| --- | --- |
| `xvar = [ones(1,T); xt_standardized(j,:)]` | `_marginal_slopes(factor_values, y_values)` uses an intercept-equivalent centered OLS slope |
| `parm = ytplush*xvar'*inv(xvar*xvar')` | closed-form marginal slope |
| `beta_scaled(j) = parm(2)` | `scaling_slopes_` |
| `scaleXs(j,:) = xt_standardized(j,:) * beta_scaled(j)` | `factor_values = factor_values * slopes` |
| `SsPCA_emp002(scaleXs, ytplush, wt, Khat, number)` | `SupervisedScaledPCARegressor` then calls the same SPCA extraction path |

Target scaling note: the Hounyo-Li macro code standardizes the target and
predictors before computing `beta_scaled`. Huang's `spcaest.m` standardizes
only predictors and keeps the target raw. Consequently, `supervised_scaled_pca`
stores standardized-target slopes when `scale=True`, while `scaled_pca` stores
raw-target slopes. These stored slope magnitudes are not directly comparable
without the target standard deviation. For factor construction and forecast
generation, however, the difference is a global target-scale multiplier rather
than a different screening or projection rule.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_components` | `3` | yes | Number of sequential SsPCA components. |
| `n_selected` | `50` | yes | Predictors selected at each SPCA step after slope scaling. |
| `min_abs_corr` | `0.0` | yes | Minimum absolute residual correlation retained before PCA. |
| `scale` | `True` | fixed by preset | Whether to standardize predictors and target inside the model. |
| `control_columns` | `None` | fixed by preset | Optional X columns used as forecasting controls. |
| `include_constant` | `True` | fixed by preset | Whether to include a constant in the control block. |
| `drop_control_columns` | `True` | fixed by preset | Whether controls are excluded from the PCA block. |
| `preselect` | `"none"` | fixed by preset | Optional pre-selection: `"none"`, `"hard_tstat"`, or `"elastic_net"`. |
| `t_threshold` | `1.28` | fixed by preset | Hard t-stat pre-selection threshold. |
| `elastic_net_alpha` | `0.0002` | fixed by preset | Elastic-net pre-selection penalty. |
| `elastic_net_l1_ratio` | `0.5` | fixed by preset | Elastic-net pre-selection L1 ratio. |
| `random_state` | `0` | fixed by preset | Elastic-net pre-selection random seed. |

The original empirical MATLAB code uses lagged target plus constant controls.
In `macroforecast`, pass the lagged target as an X column and list it in
`control_columns` when that exact control block is needed.

### ar

```python
macroforecast.models.ar(y, *, n_lag=1)
```

Fits a univariate autoregression on the target series.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_lag` | `1` | yes | Autoregressive lag order. |

| Preset | `n_lag` |
| --- | --- |
| `small` | `(1, 2, 4)` |
| `standard` | `(1, 2, 4, 6, 12)` |
| `wide` | `(1, 2, 3, 4, 6, 9, 12, 18, 24)` |

### var

```python
macroforecast.models.var(panel, *, target=None, n_lag=1)
```

Fits a VAR on a multivariate panel. `target` chooses the forecast output
column. If omitted, the first column is used.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `target` | `None` | fixed by preset | Target column in the panel. |
| `n_lag` | `1` | yes | VAR lag order. |

| Preset | `n_lag` |
| --- | --- |
| `small` | `(1, 2, 4)` |
| `standard` | `(1, 2, 4, 6, 12)` |
| `wide` | `(1, 2, 3, 4, 6, 9, 12, 18, 24)` |

### bvar_minnesota

```python
macroforecast.models.bvar_minnesota(
    panel,
    *,
    target=None,
    n_lag=1,
    shrinkage=0.2,
    intercept=True,
    random_walk_prior=True,
)
```

Fits a compact Minnesota-prior Bayesian VAR for point forecasts. The estimator
uses the conjugate-prior posterior mean for the VAR coefficients. It is a
callable forecasting model, not a full posterior simulation engine.

### bvar_normal_inverse_wishart

```python
macroforecast.models.bvar_normal_inverse_wishart(
    panel,
    *,
    target=None,
    n_lag=1,
    shrinkage=1.0,
    intercept=True,
)
```

Fits a compact normal-inverse-Wishart-style Bayesian VAR point-forecast model.
The inverse-Wishart variance prior matters for posterior uncertainty; the
current callable records the prior family and uses the posterior coefficient
mean for point forecasts.

### ets, holt_winters, theta_method

```python
macroforecast.models.ets(y, *, error="add", trend=None, seasonal=None, seasonal_periods=None)
macroforecast.models.holt_winters(y, *, trend="add", seasonal=None, seasonal_periods=None)
macroforecast.models.theta_method(y, *, period=None, deseasonalize=True, use_test=True)
```

These are target-only statsmodels forecasting wrappers. In the forecasting
runner they ignore `X` and fit on the stage target vector.

### dfm_mixed_mariano_murasawa

```python
macroforecast.models.dfm_mixed_mariano_murasawa(
    panel,
    *,
    target=None,
    metadata=None,
    monthly_columns=None,
    quarterly_columns=None,
    unsupported="raise",
    n_factors=1,
    factor_order=1,
    idiosyncratic_ar1=True,
    standardize=True,
    maxiter=500,
    tolerance=1e-6,
)
```

Fits a monthly/quarterly dynamic factor model through
`statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`. The callable
uses the Mariano-Murasawa state-space aggregation for quarterly variables by
ordering monthly columns first, quarterly columns second, and passing
`k_endog_monthly` to statsmodels.

The preferred input is a native mixed-frequency bundle:

```python
mixed = mf.data.combine(monthly_bundle, quarterly_bundle, frequency="native")
fit = mf.models.dfm_mixed_mariano_murasawa(mixed, target="GDPC1")
```

`metadata["native_frequency_by_column"]` is used to split monthly and quarterly
columns. If metadata are absent, the function infers frequencies from observed
date spacing. Explicit `monthly_columns` and `quarterly_columns` override
metadata. Unsupported frequencies raise by default; set `unsupported="drop"` to
drop those columns before fitting.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `target` | `None` | fixed by preset | Forecasted panel column. Defaults to first quarterly column, otherwise first column. |
| `metadata` | `None` | fixed by preset | Metadata with `native_frequency_by_column`; normally supplied by `DataBundle`. |
| `monthly_columns` | `None` | fixed by preset | Explicit monthly columns. |
| `quarterly_columns` | `None` | fixed by preset | Explicit quarterly columns. |
| `unsupported` | `"raise"` | fixed by preset | Unsupported frequency policy: `"raise"` or `"drop"`. |
| `n_factors` | `1` | yes | Number of dynamic factors. |
| `factor_order` | `1` | yes | VAR order for factor dynamics. |
| `idiosyncratic_ar1` | `True` | fixed by preset | Model idiosyncratic components as AR(1). |
| `standardize` | `True` | fixed by preset | Let statsmodels standardize observed variables before fitting. |
| `maxiter` | `500` | fixed by preset | EM iteration cap. |
| `tolerance` | `1e-6` | fixed by preset | EM convergence tolerance. |

| Preset | `n_factors` | `factor_order` |
| --- | --- | --- |
| `small` | `(1,)` | `(1,)` |
| `standard` | `(1, 2)` | `(1, 2)` |
| `wide` | `(1, 2, 3)` | `(1, 2, 3)` |

Diagnostics include filtered factors, fitted target values when available,
target residuals, likelihood, and fitted parameter estimates.

### dfm_unrestricted_midas

```python
macroforecast.models.dfm_unrestricted_midas(
    panel,
    *,
    target,
    lag_columns=None,
    lags=(0, 1, 2),
    factor_lags=(0,),
    target_frequency="quarterly",
    anchor_position="period_end",
    n_factors=1,
    factor_order=1,
    alpha=0.0,
)
```

Fits a composite mixed-frequency model:

1. Fit `dfm_mixed_mariano_murasawa(...)` on the native mixed-frequency panel.
2. Extract filtered DFM factors at the target anchor dates.
3. Add optional observed lag blocks from `mixed_frequency_lags(...)`.
4. Fit `unrestricted_midas(...)` as the forecast head.

This is a convenience composite, not a new state-space likelihood. The returned
fit's `predict()` method expects a prepared feature matrix with the same
columns as `fit.estimator.design_`. Use it directly after building the needed
mixed-frequency lag design, or use `dfm_mixed_mariano_murasawa` in
`forecasting.run(...)`. The runner does not yet rebuild the future composite
MIDAS design automatically.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `target` | required | fixed by preset | Forecasted target column. |
| `lag_columns` | `None` | fixed by preset | Observed columns added as unrestricted MIDAS lags. |
| `lags` | `(0, 1, 2)` | yes | Native-frequency lags for observed columns. |
| `factor_lags` | `(0,)` | yes | Monthly lags of filtered DFM factors. |
| `target_frequency` | `"quarterly"` | fixed by preset | Frequency used to position target anchor dates. |
| `anchor_position` | `"period_end"` | fixed by preset | Anchor positioning; useful for FRED-QD quarter-start dates. |
| `n_factors` | `1` | yes | Number of DynamicFactorMQ factors. |
| `factor_order` | `1` | yes | VAR order for factor dynamics. |
| `alpha` | `0.0` | yes | Ridge penalty on the unrestricted MIDAS head. |

### MIDAS variants

```python
macroforecast.models.midas_almon(X, y, *, polynomial_order=2, theta=None, alpha=0.0)
macroforecast.models.midas_beta(X, y, *, beta_params=(1.0, 1.0), alpha=0.0)
macroforecast.models.midas_step(X, y, *, n_steps=3, alpha=0.0)
macroforecast.models.unrestricted_midas(X, y, *, alpha=0.0)
```

The MIDAS callables expect lag-grouped predictor columns, typically names like
`PAYEMS_lag0`, `PAYEMS_lag1`, and `PAYEMS_lag2`. Columns sharing the same
prefix before `_lag#` are collapsed into one weighted aggregate before a
linear or ridge regression is fit. This keeps mixed-frequency weighting as a
model choice while leaving calendar alignment and lag construction in
`data`, `preprocessing`, and `feature_engineering`.

These callables are small model functions, not workflow recipes. They do not
infer target anchors, release calendars, or future design matrices. Build the
lag matrix explicitly with `mixed_frequency_lags(...)`, align `X` and `y`, then
call the model.

The constrained MIDAS variants use supplied weight-shape parameters:

| Function | Weight shape | Notes |
| --- | --- | --- |
| `midas_almon` | `w_j = exp(theta_0 + theta_1 x_j + ... + theta_p x_j^p) / sum_j exp(...)` | `theta=None` gives equal weights. If `theta` is supplied, it must have `polynomial_order + 1` values. |
| `midas_beta` | `w_j = z_j^(a-1) (1-z_j)^(b-1) / sum_j ...` | `beta_params=(a, b)` must be strictly positive. |
| `midas_step` | Equal total weight per lag bucket, equal weight within each bucket. | `n_steps` must be positive. |

The package currently treats these shape parameters as fixed or selected
hyperparameters. It does not yet implement nonlinear joint estimation of the
weight shape and regression coefficients inside one likelihood/objective.

`unrestricted_midas()` is the exception: it does not collapse lag groups.
Every supplied lag column receives its own coefficient, with optional ridge
shrinkage through `alpha`. Build its input matrix with
`mf.feature_engineering.mixed_frequency_lags(...)` when the source data are
native mixed-frequency panels.

All MIDAS callables preserve lag metadata. Weighted variants record lag groups,
resolved weights, weighted aggregate column names, aggregate coefficients, and
effective lag coefficients. `unrestricted_midas()` records the original lag
groups and per-lag coefficients.

```python
X_midas = mf.feature_engineering.mixed_frequency_lags(
    mixed,
    target="GDPC1",
    columns=["PAYEMS", "INDPRO"],
    lags=range(0, 12),
    target_frequency="quarterly",
    anchor_position="period_end",
    drop_missing=True,
)

y = mixed.panel["GDPC1"].dropna()
y.index = y.index.to_period("Q").asfreq("M", how="end").to_timestamp()
aligned = X_midas.join(y.rename("GDPC1")).dropna()

fit = mf.models.midas_beta(
    aligned.drop(columns="GDPC1"),
    aligned["GDPC1"],
    beta_params=(1.0, 2.0),
    alpha=0.1,
)

fit.metadata["weights"]
fit.diagnostics["effective_lag_coefficients"]
```

### far

```python
macroforecast.models.far(
    X,
    y,
    *,
    n_factors=3,
    n_lag=1,
    random_state=0,
)
```

Fits factor-augmented autoregression: PCA factors from `X` plus AR lags of
`y`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_factors` | `3` | yes | Number of PCA factors. |
| `n_lag` | `1` | yes | Autoregressive lag order. |
| `random_state` | `0` | fixed by preset | PCA random seed. |

| Preset | `n_factors` | `n_lag` |
| --- | --- | --- |
| `small` | `(1, 2, 3)` | `(1, 2, 4)` |
| `standard` | `(1, 2, 3, 5, 8)` | `(1, 2, 4, 6, 12)` |
| `wide` | `(1, 2, 3, 5, 8, 10, 12)` | `(1, 2, 3, 4, 6, 9, 12, 18, 24)` |

### favar

```python
macroforecast.models.favar(
    X,
    y,
    *,
    n_factors=3,
    n_lag=1,
    random_state=0,
)
```

Fits PCA factors and a VAR on the target plus factors.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_factors` | `3` | yes | Number of PCA factors. |
| `n_lag` | `1` | yes | VAR lag order on the target plus factors. |
| `random_state` | `0` | fixed by preset | PCA random seed. |

| Preset | `n_factors` | `n_lag` |
| --- | --- | --- |
| `small` | `(1, 2, 3)` | `(1, 2, 4)` |
| `standard` | `(1, 2, 3, 5, 8)` | `(1, 2, 4, 6, 12)` |
| `wide` | `(1, 2, 3, 5, 8, 10, 12)` | `(1, 2, 3, 4, 6, 9, 12, 18, 24)` |

## Tree And Machine-Learning Models

### decision_tree

```python
macroforecast.models.decision_tree(
    X,
    y,
    *,
    max_depth=None,
    min_samples_leaf=1,
    random_state=0,
)
```

Fits sklearn CART regression.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `max_depth` | `None` | yes | Maximum tree depth. |
| `min_samples_leaf` | `1` | yes | Minimum samples per terminal leaf. |
| `random_state` | `0` | fixed by preset | Tree random seed. |

| Preset | `max_depth` | `min_samples_leaf` |
| --- | --- | --- |
| `small` | `(3, 5, None)` | `(1, 3)` |
| `standard` | `(3, 5, 10, None)` | `(1, 3, 5)` |
| `wide` | `(2, 3, 5, 10, 20, None)` | `(1, 2, 3, 5, 10)` |

### random_forest

```python
macroforecast.models.random_forest(
    X,
    y,
    *,
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    random_state=0,
    n_jobs=1,
)
```

Fits sklearn random forest regression.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `200` | yes | Number of trees. |
| `max_depth` | `None` | yes | Maximum depth per tree. |
| `min_samples_leaf` | `1` | yes | Minimum samples per terminal leaf. |
| `random_state` | `0` | fixed by preset | Forest random seed. |
| `n_jobs` | `1` | fixed by preset | Parallel worker count. |

| Preset | `n_estimators` | `max_depth` | `min_samples_leaf` |
| --- | --- | --- | --- |
| `small` | `(50, 100)` | `(3, 5, None)` | `(1, 3)` |
| `standard` | `(100, 200, 500)` | `(3, 5, 10, None)` | `(1, 3, 5)` |
| `wide` | `(100, 200, 500, 1000)` | `(3, 5, 10, 20, None)` | `(1, 2, 3, 5, 10)` |

Default selection method: `random`.

### extra_trees

```python
macroforecast.models.extra_trees(
    X,
    y,
    *,
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    random_state=0,
    n_jobs=1,
)
```

Fits sklearn extremely randomized trees. Parameters and presets match
`random_forest`.

Default selection method: `random`.

### gradient_boosting

```python
macroforecast.models.gradient_boosting(
    X,
    y,
    *,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=0,
)
```

Fits sklearn gradient-boosted regression trees.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `200` | yes | Number of boosting stages. |
| `learning_rate` | `0.1` | yes | Shrinkage per stage. |
| `max_depth` | `3` | yes | Maximum tree depth. |
| `random_state` | `0` | fixed by preset | Boosting random seed. |

| Preset | `n_estimators` | `learning_rate` | `max_depth` |
| --- | --- | --- | --- |
| `small` | `(50, 100)` | `(0.05, 0.1)` | `(2, 3)` |
| `standard` | `(100, 200, 500)` | `(0.03, 0.05, 0.1)` | `(2, 3, 5)` |
| `wide` | `(100, 200, 500, 1000)` | `(0.01, 0.03, 0.05, 0.1)` | `(2, 3, 5, 8)` |

Default selection method: `random`.

### mars

```python
macroforecast.models.mars(
    X,
    y,
    *,
    max_terms=20,
    max_degree=1,
    n_knots=10,
    min_improvement=1e-6,
    penalty=2.0,
    prune=True,
)
```

Fits a package-native MARS-style hinge-basis regression. It uses forward
insertion of hinge basis pairs and optional backward pruning by generalized
cross-validation. This avoids the unmaintained `pyearth` dependency; it is a
clean internal implementation and does not claim bit-level equivalence to
other MARS backends.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `max_terms` | `20` | yes | Maximum number of basis terms including intercept. |
| `max_degree` | `1` | yes | Maximum interaction degree. |
| `n_knots` | `10` | yes | Candidate quantile knots per predictor. |
| `min_improvement` | `1e-6` | fixed by preset | Forward-step relative RSS improvement floor. |
| `penalty` | `2.0` | fixed by preset | GCV pruning complexity penalty. |
| `prune` | `True` | fixed by preset | Whether to prune terms by GCV. |

Default selection method: `random`.

### xgboost

```python
macroforecast.models.xgboost(
    X,
    y,
    *,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=1.0,
    random_state=0,
    **kwargs,
)
```

Fits `xgboost.XGBRegressor`. Requires `macroforecast[xgboost]`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `300` | yes | Number of boosting stages. |
| `learning_rate` | `0.1` | yes | Shrinkage per stage. |
| `max_depth` | `6` | yes | Maximum tree depth. |
| `subsample` | `1.0` | yes | Row subsample share. |
| `random_state` | `0` | fixed by preset | Boosting random seed. |

Preset spaces match `gradient_boosting` plus `subsample=(0.6, 0.8, 1.0)`.
Default selection method: `random`.

### lightgbm

```python
macroforecast.models.lightgbm(
    X,
    y,
    *,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=-1,
    num_leaves=31,
    random_state=0,
    **kwargs,
)
```

Fits `lightgbm.LGBMRegressor`. Requires `macroforecast[lightgbm]`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `300` | yes | Number of boosting stages. |
| `learning_rate` | `0.1` | yes | Shrinkage per stage. |
| `max_depth` | `-1` | yes | Maximum tree depth; `-1` means no limit. |
| `num_leaves` | `31` | yes | Maximum leaves per tree. |
| `random_state` | `0` | fixed by preset | Boosting random seed. |

| Preset | `n_estimators` | `learning_rate` | `max_depth` | `num_leaves` |
| --- | --- | --- | --- | --- |
| `small` | `(50, 100)` | `(0.05, 0.1)` | `(-1, 3, 5)` | `(15, 31)` |
| `standard` | `(100, 200, 500)` | `(0.03, 0.05, 0.1)` | `(-1, 3, 5, 10)` | `(15, 31, 63)` |
| `wide` | `(100, 200, 500, 1000)` | `(0.01, 0.03, 0.05, 0.1)` | `(-1, 3, 5, 10, 20)` | `(15, 31, 63, 127)` |

Default selection method: `random`.

### catboost

```python
macroforecast.models.catboost(
    X,
    y,
    *,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    random_state=0,
    verbose=False,
    **kwargs,
)
```

Fits `catboost.CatBoostRegressor`. Requires `macroforecast[catboost]`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `300` | yes | Number of boosting stages. |
| `learning_rate` | `0.1` | yes | Shrinkage per stage. |
| `max_depth` | `6` | yes | Tree depth. |
| `random_state` | `0` | fixed by preset | Boosting random seed. |
| `verbose` | `False` | fixed by preset | CatBoost console output flag. |

Preset spaces match `gradient_boosting`. Default selection method:
`random`.

## Macro-Specific Tree And Ensemble Models

### slow_growing_tree

```python
macroforecast.models.slow_growing_tree(
    X,
    y,
    *,
    eta=0.1,
    herfindahl_threshold=0.25,
    eta_depth_step=0.01,
    eta_max_plateau=0.5,
    mtry_frac=1.0,
    max_depth=10,
    random_state=0,
    min_leaf_size=5,
)
```

Fits a soft-split Slow-Growing Tree. At each split, rows on the non-followed
side keep fractional weight `1 - eta`, so prediction uses soft path
membership rather than a hard single leaf.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `eta` | `0.1` | yes | Soft split leakage parameter. |
| `herfindahl_threshold` | `0.25` | yes | Node concentration threshold for stopping. |
| `eta_depth_step` | `0.01` | fixed by preset | Per-depth increase in soft split leakage. |
| `eta_max_plateau` | `0.5` | fixed by preset | Upper plateau for depth-adjusted leakage. |
| `mtry_frac` | `1.0` | yes | Fraction of candidate features considered at each split. |
| `max_depth` | `10` | yes | Maximum tree depth. |
| `min_leaf_size` | `5` | yes | Minimum effective leaf size. |
| `random_state` | `0` | fixed by preset | Tree random seed. |

| Preset | `eta` | `herfindahl_threshold` | `mtry_frac` | `max_depth` | `min_leaf_size` |
| --- | --- | --- | --- | --- | --- |
| `small` | `(0.05, 0.1)` | `(0.2, 0.3)` | `(0.75, 1.0)` | `(5, 10)` | `(3, 5)` |
| `standard` | `(0.03, 0.05, 0.1)` | `(0.15, 0.25, 0.35)` | `(0.5, 0.75, 1.0)` | `(5, 10, None)` | `(3, 5, 10)` |
| `wide` | `(0.01, 0.03, 0.05, 0.1, 0.2)` | `(0.1, 0.15, 0.25, 0.35, 0.5)` | `(0.33, 0.5, 0.75, 1.0)` | `(3, 5, 10, 20, None)` | `(2, 3, 5, 10)` |

Default selection method: `random`.

### quantile_regression_forest

```python
macroforecast.models.quantile_regression_forest(
    X,
    y,
    *,
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    random_state=0,
    quantile_levels=(0.05, 0.5, 0.95),
)
```

Fits a random forest and stores per-leaf training-target distributions. The
underlying estimator exposes `predict_quantiles(X, levels=None)`. The
forecasting runner stores those outputs in the `quantile_predictions` column
as per-row dictionaries keyed by quantile level.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `200` | yes | Number of trees. |
| `max_depth` | `None` | yes | Maximum depth per tree. |
| `min_samples_leaf` | `1` | yes | Minimum samples per terminal leaf. |
| `random_state` | `0` | fixed by preset | Forest random seed. |
| `quantile_levels` | `(0.05, 0.5, 0.95)` | fixed by preset | Default levels returned by `predict_quantiles()`. |

Preset spaces match `random_forest`. Default selection method: `random`.

### bagging

```python
macroforecast.models.bagging(
    X,
    y,
    *,
    base="ridge",
    n_estimators=50,
    max_samples=0.8,
    random_state=0,
    base_params=None,
    strategy="standard",
    block_length=4,
)
```

Fits bootstrap aggregation over a supported base estimator. `strategy="block"`
uses moving-block bootstrap indices.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `base` | `"ridge"` | yes | Base estimator name. |
| `n_estimators` | `50` | yes | Number of bootstrap models. |
| `max_samples` | `0.8` | yes | Bootstrap sample fraction. |
| `base_params` | `{}` | fixed by preset | Parameters passed to each base estimator. |
| `strategy` | `"standard"` | yes | Bootstrap strategy: `standard` or `block`. |
| `block_length` | `4` | yes | Block length when `strategy="block"`. |
| `random_state` | `0` | fixed by preset | Ensemble random seed. |

| Preset | `base` | `n_estimators` | `max_samples` | `strategy` | `block_length` |
| --- | --- | --- | --- | --- | --- |
| `small` | `("ridge", "lasso")` | `(10, 25)` | `(0.6, 0.8)` | `("standard",)` | `(4,)` |
| `standard` | `("ridge", "lasso", "decision_tree")` | `(25, 50, 100)` | `(0.5, 0.7, 0.9)` | `("standard", "block")` | `(4, 8)` |
| `wide` | `("ridge", "lasso", "elastic_net", "decision_tree", "random_forest")` | `(25, 50, 100, 200)` | `(0.4, 0.6, 0.8, 1.0)` | `("standard", "block")` | `(2, 4, 8, 12)` |

Default selection method: `random`.

### booging

```python
macroforecast.models.booging(
    X,
    y,
    *,
    B=100,
    sample_frac=0.75,
    inner_n_estimators=1500,
    inner_learning_rate=0.1,
    inner_max_depth=3,
    inner_subsample=0.5,
    da_noise_frac=1/3,
    da_drop_rate=0.2,
    random_state=0,
)
```

Fits bagged overfit stochastic gradient boosting with Gaussian data
augmentation.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `B` | `100` | yes | Number of overfit boosting models. |
| `sample_frac` | `0.75` | yes | Row sample fraction per model. |
| `inner_n_estimators` | `1500` | yes | Boosting stages inside each model. |
| `inner_learning_rate` | `0.1` | yes | Inner boosting learning rate. |
| `inner_max_depth` | `3` | yes | Inner boosting tree depth. |
| `inner_subsample` | `0.5` | yes | Inner boosting subsample share. |
| `da_noise_frac` | `1/3` | yes | Scale of feature-noise augmentation. |
| `da_drop_rate` | `0.2` | yes | Share of augmented columns dropped per model. |
| `random_state` | `0` | fixed by preset | Ensemble random seed. |

The presets tune all Booging parameters marked `yes`; use
`describe_model("booging")` to inspect the exact candidate lists.

Default selection method: `random`.

### macro_random_forest

```python
macroforecast.models.macro_random_forest(
    X,
    y,
    *,
    x_columns=None,
    S_columns=None,
    x_pos=None,
    S_pos=None,
    y_pos=0,
    B=50,
    minsize=10,
    mtry_frac=1/3,
    min_leaf_frac_of_x=1.0,
    VI=False,
    ERT=False,
    quantile_rate=None,
    S_priority_vec=None,
    random_x=False,
    trend_push=1,
    howmany_random_x=1,
    howmany_keep_best_VI=20,
    cheap_look_at_GTVPs=True,
    prior_var=None,
    prior_mean=None,
    subsampling_rate=0.75,
    rw_regul=0.75,
    keep_forest=False,
    block_size=12,
    fast_rw=True,
    ridge_lambda=0.1,
    HRW=0,
    resampling_opt=2,
    print_b=False,
    parallelise=False,
    n_cores=1,
    **kwargs,
)
```

Adapter for Ryan Lucas's `MacroRandomForest` reference backend. The reference
implementation is vendored from `MacroRandomForest` 1.0.6 under the MIT
license, with source attribution in
`macroforecast.models._mrf_reference`. Install the optional runtime
dependencies with `macroforecast[macro_random_forest]`. The adapter fits on
the in-sample `X/y` and calls the reference `_ensemble_loop()` during
`predict(X_test)`. Repeated calls to `predict()` with the same test matrix
reuse the previous reference-backend output, so repeated result materialization
does not rerun the expensive forest loop.

By default all columns in `X` are used both as the time-varying linear equation
variables (`x_columns`) and the forest state variables (`S_columns`). Pass
`x_columns` and `S_columns` when those sets should differ.

The reference backend distinguishes two predictor sets:

| Argument | Role |
| --- | --- |
| `x_columns` | Columns in the local linear forecasting equation. These are the variables whose coefficients are allowed to vary over time. |
| `S_columns` | State variables used by the forest to split the sample and estimate those local coefficients. |

For example, a compact MRF can use a small local-linear equation but a wider
state vector for the tree:

```python
fit = macroforecast.models.macro_random_forest(
    X_train,
    y_train,
    x_columns=["INDPRO_lag0", "UNRATE_lag0"],
    S_columns=[
        "INDPRO_lag0",
        "UNRATE_lag0",
        "CPIAUCSL_lag0",
        "FEDFUNDS_lag0",
        "S&P500_lag0",
    ],
    B=50,
    minsize=10,
    mtry_frac=1.0,
    ridge_lambda=0.1,
    rw_regul=0.75,
    parallelise=False,
    print_b=False,
)

pred = fit.predict(X_test)
```

With the forecasting runner, pass model parameters through the model-keyed
`params` mapping. If you want fixed parameters rather than model-owned tuning,
also disable selection for this model:

```python
features = macroforecast.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors=["UNRATE", "CPIAUCSL", "FEDFUNDS", "S&P500"],
    lags=(0, 1),
)

window = macroforecast.window.spec(
    estimation=macroforecast.window.estimation_expanding(min_size=120),
    val=macroforecast.window.val_last_block(size=24),
    test=macroforecast.window.test_origins(horizon=1, step=1),
)

result = macroforecast.forecasting.run(
    panel,
    "macro_random_forest",
    window=window,
    features=features,
    params={
        "macro_random_forest": {
            "x_columns": ["UNRATE_lag0", "FEDFUNDS_lag0"],
            "S_columns": [
                "UNRATE_lag0",
                "UNRATE_lag1",
                "CPIAUCSL_lag0",
                "FEDFUNDS_lag0",
                "S&P500_lag0",
            ],
            "B": 100,
            "minsize": 10,
            "mtry_frac": 1.0,
            "parallelise": False,
            "print_b": False,
        }
    },
    selection={"macro_random_forest": None},
)
```

The reference implementation is sensitive to panel shape. Use numeric,
non-missing features after preprocessing and feature engineering. Keep at
least one `x_columns` variable, and prefer at least five `S_columns` variables;
with very small state sets, set `mtry_frac=1.0` so at least one state variable
is considered at each split. Small training samples can also fail when
`minsize` is too large relative to the number of local-linear variables.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `x_columns` | `None` | fixed by preset | Feature columns in the time-varying linear equation. |
| `S_columns` | `None` | fixed by preset | Feature columns used as forest state variables. |
| `x_pos` | `None` | fixed by preset | Reference-package predictor positions after the target column. |
| `S_pos` | `None` | fixed by preset | Reference-package state positions after the target column. |
| `y_pos` | `0` | fixed by preset | Reference-package target column position. |
| `B` | `50` | yes | Number of MRF trees. |
| `minsize` | `10` | yes | Minimum node size before split attempts. |
| `mtry_frac` | `1/3` | yes | Fraction of state variables considered at each split. |
| `min_leaf_frac_of_x` | `1.0` | yes | Minimum leaf-size multiplier relative to local x dimension. |
| `VI` | `False` | fixed by preset | Enable variable-importance split search mode. |
| `ERT` | `False` | fixed by preset | Enable extremely randomized tree split mode. |
| `quantile_rate` | `None` | fixed by preset | Optional quantile rate for quantile-oriented output. |
| `S_priority_vec` | `None` | fixed by preset | Optional priority weights over state variables. |
| `random_x` | `False` | fixed by preset | Use random subsets of local-linear predictors. |
| `trend_push` | `1` | fixed by preset | Reference-package trend-push option. |
| `howmany_random_x` | `1` | fixed by preset | Number of random local-linear predictor draws. |
| `howmany_keep_best_VI` | `20` | fixed by preset | Number of best VI candidates retained. |
| `cheap_look_at_GTVPs` | `True` | fixed by preset | Use the reference package's cheaper GTVP inspection. |
| `prior_var` | `None` | fixed by preset | Optional prior variances for local coefficients. |
| `prior_mean` | `None` | fixed by preset | Optional prior means for local coefficients. |
| `subsampling_rate` | `0.75` | yes | Subsample share used by each tree. |
| `rw_regul` | `0.75` | yes | Random-walk shrinkage strength. |
| `keep_forest` | `False` | fixed by preset | Keep full reference forest object in memory. |
| `block_size` | `12` | fixed by preset | Reference-package block size for time-series resampling. |
| `fast_rw` | `True` | fixed by preset | Use fast random-walk regularization path. |
| `ridge_lambda` | `0.1` | yes | Ridge penalty for local linear fits. |
| `HRW` | `0` | fixed by preset | Reference-package hierarchical random-walk option. |
| `resampling_opt` | `2` | yes | Reference MRF resampling option. |
| `parallelise` | `False` | fixed by preset | Whether to use reference-package parallel execution. |
| `n_cores` | `1` | fixed by preset | Worker count for the reference package. |
| `print_b` | `False` | fixed by preset | Reference-package progress printing. |

The MRF presets tune `B`, `minsize`, `mtry_frac`,
`min_leaf_frac_of_x`, `subsampling_rate`, `rw_regul`, `ridge_lambda`, and
`resampling_opt`; inspect the exact candidate lists with
`describe_model("macro_random_forest")`.

## Volatility Models

Volatility model fits return `VolatilityFit`. In addition to
`predict_variance(horizon=...)`, their diagnostics include fitted parameter
estimates under `params` and the in-sample `conditional_volatility` path when
the backend exposes it.

### garch11

```python
macroforecast.models.garch11(
    y,
    *,
    X=None,
    p=1,
    q=1,
    mean_model="constant",
    dist="normal",
    rescale=False,
)
```

Fits GARCH using the optional `arch` package. Requires `macroforecast[arch]`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `p` | `1` | yes | GARCH innovation lag order. |
| `q` | `1` | yes | GARCH variance lag order. |
| `mean_model` | `"constant"` | manual | Conditional mean model. |
| `dist` | `"normal"` | yes | Innovation distribution. |
| `rescale` | `False` | fixed by preset | `arch` package rescale option. |

| Preset | `p` | `q` | `dist` |
| --- | --- | --- | --- |
| `small` | `(1,)` | `(1,)` | `("normal", "t")` |
| `standard` | `(1, 2)` | `(1, 2)` | `("normal", "t")` |
| `wide` | `(1, 2, 3)` | `(1, 2, 3)` | `("normal", "t", "skewt")` |

### egarch

```python
macroforecast.models.egarch(
    y,
    *,
    X=None,
    p=1,
    o=0,
    q=1,
    mean_model="constant",
    dist="normal",
    rescale=False,
)
```

Fits EGARCH using the optional `arch` package. Requires `macroforecast[arch]`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `p` | `1` | yes | EGARCH innovation lag order. |
| `o` | `0` | yes | Asymmetric innovation lag order. |
| `q` | `1` | yes | EGARCH variance lag order. |
| `mean_model` | `"constant"` | manual | Conditional mean model. |
| `dist` | `"normal"` | yes | Innovation distribution. |
| `rescale` | `False` | fixed by preset | `arch` package rescale option. |

| Preset | `p` | `o` | `q` | `dist` |
| --- | --- | --- | --- | --- |
| `small` | `(1,)` | `(0, 1)` | `(1,)` | `("normal", "t")` |
| `standard` | `(1, 2)` | `(0, 1)` | `(1, 2)` | `("normal", "t")` |
| `wide` | `(1, 2, 3)` | `(0, 1, 2)` | `(1, 2, 3)` | `("normal", "t", "skewt")` |

### realized_garch

```python
macroforecast.models.realized_garch(
    y,
    *,
    X=None,
    rv=None,
    realized_variance=None,
    max_iter=2000,
    n_starts=5,
    random_state=0,
)
```

Fits a compact realized-GARCH joint likelihood. Provide `rv` directly or set
`realized_variance` to the column in `X` containing the realized measure.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `realized_variance` | `None` | manual | Column name for realized variance. |
| `max_iter` | `2000` | fixed by preset | Optimizer iteration cap. |
| `n_starts` | `5` | yes | Number of optimizer starting points. |
| `random_state` | `0` | fixed by preset | Optimizer random seed. |

| Preset | `n_starts` |
| --- | --- |
| `small` | `(3, 5)` |
| `standard` | `(3, 5, 10)` |
| `wide` | `(3, 5, 10, 20)` |

## Omitted From The Clean Model API

| Legacy name | Decision |
| --- | --- |
| `lasso_path` | Removed. Use `get_model("lasso")` and `selection.select_params()`. |
| `pcr` | Removed. Use `feature_engineering.feature_spec(pca_components=...)` with a regression model. |
