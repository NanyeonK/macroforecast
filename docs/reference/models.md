# macroforecast.models

[Back to reference](index.md)

`macroforecast.models` contains direct callable model fits. Each function
accepts pandas data, fits immediately, and returns a fitted result object with
`predict()`.

`lasso_path` is intentionally not a public model family. Use `lasso()` with a
chosen `alpha`, or use `get_model("lasso")` with `model_selection.select_params()`
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

Some model families expose backend estimator classes as public symbols. These
are useful when users need estimator-native attributes, custom wrappers, or
type checks; the normal user entry point remains the lowercase fit function.

| Estimator class | Fit function | Meaning |
| --- | --- | --- |
| `MARSRegressor` | `mars(...)` | Internal MARS-style spline regressor. |
| `LGBPlusRegressor` | `lgb_plus(...)` | Competition-based LGB+ hybrid tree/linear boosting backend. |
| `LGBAPlusRegressor` | `lgba_plus(...)` | Alternating LGB^A+ hybrid tree/linear boosting backend. |
| `QuantileRegressionForestRegressor` | `quantile_regression_forest(...)` | Quantile forest backend. |
| `MacroRandomForestRegressor` | `macro_random_forest(...)` | Macro Random Forest backend. |
| `SupervisedPCARegressor` | `supervised_pca(...)` | Supervised PCA backend. |
| `SupervisedScaledPCARegressor` | `supervised_scaled_pca(...)` | Supervised scaled-PCA backend. |
| `GARCHEstimator` | `garch11(...)` and `egarch(...)` | ARCH/GARCH volatility backend. |
| `RealizedGARCHEstimator` | `realized_garch(...)` | Realized-GARCH backend. |
| `SupervisedAggregationRegressor` | `supervised_aggregation(...)` and wrappers | Generic assemblage/Albacore-style constrained aggregation backend. |

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
        "model_selection": selection_metadata,
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

Model functions fit immediately. Model specs are the model-selection objects:
they keep the fit callable together with model-owned defaults, tunable
parameters, and preset search spaces.

```python
model = macroforecast.models.get_model("lasso", preset="standard")
result = macroforecast.model_selection.select_params(
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
    "backend": "sklearn.linear_model.Ridge",
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

### custom_model

Build a user-owned `ModelSpec` without registering a package model.

```python
macroforecast.models.custom_model(
    name: str,
    fit_func,
    *,
    family: str = "custom",
    default_params: Mapping[str, object] | None = None,
    parameters: tuple[ModelParameter, ...] = (),
    search_spaces: dict[str, dict[str, tuple[object, ...]]] | None = None,
    default_search_method: str = "grid",
    default_preset: str = "standard",
    input_kind: str = "supervised",
    backend: str = "custom",
    requires_extra: str | None = None,
    requires_scaling: bool = False,
    recommended_preprocessing: tuple[str, ...] = (),
    description: str | None = None,
) -> ModelSpec
```

### Callable Contract

The default supervised contract is:

```python
fit_func(X: pandas.DataFrame, y: pandas.Series, **params) -> fitted_object
```

The fitted object must expose:

```python
fitted_object.predict(X_test)
```

`predict(X_test)` may return a pandas `Series`, a single-column `DataFrame`, or
an array-like object with length `len(X_test)`. Pandas output must either use
`X_test.index` or `RangeIndex(len(X_test))`; any other index is rejected by
`forecasting.run(...)`.

Set `input_kind` when the custom model follows another convention:

| `input_kind` | Fit callable receives | Use case |
| --- | --- | --- |
| `"supervised"` | `fit_func(X, y, **params)` | Regression-style models. |
| `"target"` | `fit_func(y, **params)` | Target-only time-series models. |
| `"panel"` | `fit_func(panel, **params)` | Panel-input models. |
| `"volatility"` | `fit_func(y, X=None, **params)` | Volatility or density models. |

`search_spaces` uses the same model-owned preset contract as registered models:

```python
model = mf.models.custom_model(
    "mean_model",
    mean_model,
    default_params={"offset": 0.0},
    search_spaces={
        "small": {"offset": (-0.1, 0.0, 0.1)},
        "standard": {"offset": (-0.5, 0.0, 0.5)},
    },
)

result = mf.forecasting.run(
    panel,
    {"mean": model},
    window=window,
    features=features,
    preset={"mean": "small"},
)
```

`custom_model()` does not mutate the global registry. Pass the returned
`ModelSpec` directly to `forecasting.run(...)`, `model_selection.select_params(...)`,
or `model_search_space(...)`.

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
`MODEL_SPECS` is the public registry backing `get_model(...)`,
`list_model_specs()`, `describe_model(...)`, and `model_search_space(...)`.

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
| `supervised` | `model(X, y, **params)` | Most regression, factor, and tree models. Fit-time ensembles use the same shape in `macroforecast.model_ensemble`. |
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
`model_search_space()`, and `select_params()` metadata.
`lasso(..., standardize=True)` and `elastic_net(..., standardize=True)` are
explicit opt-in replication helpers; the default remains `False`.

There are two different scaling locations:

- `macroforecast.preprocessing.standardize_panel()` standardizes a panel before
  model fitting. If it is run on the full sample outside the forecasting runner,
  it uses full-sample moments. In a leakage-safe run, use runner preprocessing
  specs and window policies so the scaling state is fitted only on allowed rows.
- `model(..., standardize=True)` standardizes inside that model's own fit call.
  It is useful when only selected models need model-local scaling, or when a
  lasso/elastic-net replication requires the penalty grid to be defined on
  window-local standardized predictors. For broader model-specific
  transformations beyond scaling, run separate model pipelines or use a
  model-pipeline runner layer rather than hiding those transformations inside a
  single estimator.

Current scale-sensitive callable models:

| Model | Backend | Scaling policy |
| --- | --- | --- |
| `svr` | `sklearn.svm.SVR` | Standardize predictors with `preprocessing.standardize_panel()` or a runner preprocessing spec before fitting. |
| `linear_svr` | `sklearn.svm.LinearSVR` | Standardize predictors before fitting. |
| `nu_svr` | `sklearn.svm.NuSVR` | Standardize predictors before fitting. |
| `nn` | `torch.nn.Sequential` | Standardizes `X` and `y` inside each fit window and maps predictions back to target units. |
| `transformer` | `torch.nn.TransformerEncoder` | Standardizes `X` and `y` inside each fit window and maps predictions back to target units. |
| `hemisphere_nn` | torch dual-head dense network | Standardizes `X` inside each fit window, fits mean and variance heads, and returns point, variance, and normal-approximation quantile forecasts. |
| `density_hnn` | torch-native Aionx DensityHNN port | Standardizes `X` and `y`, estimates prior-DNN OOB volatility emphasis, fits a density HNN ensemble, and returns point, variance, volatility, and quantile forecasts. |

`nn`, `lstm`, `gru`, `transformer`, and `density_hnn` standardize `X` and `y`
inside each fit window and map predictions back to the target scale.
`hemisphere_nn` standardizes `X` and keeps the target in original units because
its variance head is a compact density-forecast object. Their metadata records
`requires_extra="deep"` and `requires_scaling=False`.

## Registered Model Catalog

| Model | Family | Input kind | Default search | Presets |
| --- | --- | --- | --- | --- |
| `ols` | linear | supervised | `grid` | none |
| `ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `nonneg_ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `shrink_to_target_ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `fused_difference_ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `supervised_aggregation` | assemblage | supervised | `cv_path` | `small`, `standard`, `wide` |
| `component_aggregation` | assemblage | supervised | `cv_path` | `small`, `standard`, `wide` |
| `rank_aggregation` | assemblage | supervised | `cv_path` | `small`, `standard`, `wide` |
| `assemblage_regression` | assemblage | supervised | `cv_path` | `small`, `standard`, `wide` |
| `albacore_components` | assemblage | supervised | `cv_path` | `small`, `standard`, `wide` |
| `albacore_ranks` | assemblage | supervised | `cv_path` | `small`, `standard`, `wide` |
| `random_walk_ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `tvp_ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `lasso` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `elastic_net` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `adaptive_lasso` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `adaptive_elastic_net` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `group_lasso` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `sparse_group_lasso` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `bayesian_ridge` | linear | supervised | `grid` | none |
| `huber` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `kernel_ridge` | nonparametric | supervised | `random` | `small`, `standard`, `wide` |
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
| `density_hnn` | neural | supervised | `random` | `small`, `standard`, `wide` |
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
| `restricted_midas` | mixed_frequency | supervised | `grid` | `small`, `standard`, `wide` |
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
| `lgb_plus` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `lgba_plus` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `catboost` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `quantile_regression_forest` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `macro_random_forest` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `garch11` | volatility | volatility | `grid` | `small`, `standard`, `wide` |
| `egarch` | volatility | volatility | `grid` | `small`, `standard`, `wide` |
| `realized_garch` | volatility | volatility | `grid` | `small`, `standard`, `wide` |

## Linear Models

### Linear implementation map

The `linear` family mixes thin sklearn wrappers, hybrid macroforecast code, and
package-native solvers. `backend` in `ModelSpec` records that distinction so
metadata exported by `describe_model()`, `list_model_specs()`, and saved
`ModelFit` objects is inspectable.

| Model | Implementation class | Runtime backend |
| --- | --- | --- |
| `ols` | external wrapper | `sklearn.linear_model.LinearRegression` |
| `ridge` | external wrapper | `sklearn.linear_model.Ridge` |
| `lasso` | external wrapper | `sklearn.linear_model.Lasso` |
| `elastic_net` | external wrapper | `sklearn.linear_model.ElasticNet` |
| `bayesian_ridge` | external wrapper | `sklearn.linear_model.BayesianRidge` |
| `huber` | external wrapper | `sklearn.linear_model.HuberRegressor` |
| `adaptive_lasso` | hybrid | macroforecast adaptive weights, final `sklearn.linear_model.Lasso` |
| `adaptive_elastic_net` | hybrid | macroforecast adaptive weights, final `sklearn.linear_model.ElasticNet` |
| `nonneg_ridge` | package-native | augmented ridge design solved by `scipy.optimize.nnls` |
| `shrink_to_target_ridge` | package-native | custom objective solved by `scipy.optimize.minimize(method="SLSQP")` |
| `fused_difference_ridge` | package-native | custom difference-penalty objective solved by SLSQP |
| `supervised_aggregation`, `component_aggregation`, `rank_aggregation`, `assemblage_regression`, `albacore_components`, `albacore_ranks` | package-native | Albacore/assemblage-derived constrained aggregation objectives solved by SLSQP |
| `random_walk_ridge` | package-native | expanded time-varying design solved by `numpy.linalg.lstsq` |
| `tvp_ridge` | package-native | Python port of `TVPRidge` R `tvp.ridge`, `Zfun`, `dualGRR`, and CV helpers |
| `group_lasso` | package-native | proximal-gradient group-lasso solver |
| `sparse_group_lasso` | package-native | proximal-gradient sparse-group-lasso solver |
| `glmboost` | package-native | componentwise L2 boosting loop |

`external wrapper` means the statistical estimator is delegated to an external
package and macroforecast only standardizes the callable contract, metadata,
diagnostics, and persistence. `hybrid` means macroforecast owns the macro-level
algorithmic transformation and delegates the final convex solver. `package-native`
means the objective, iteration, or coefficient path logic is implemented inside
macroforecast, using NumPy/SciPy only for basic numerical linear algebra or
generic optimization.

### R source comparison map

The following R sources are the comparison surface for the linear models where
macroforecast owns nontrivial logic. These sources are not vendored into the
package. They are used as independent algorithm references; the Python code
keeps short source cues in comments and implements the corresponding
mathematical objective in macroforecast's callable API.

| macroforecast model | R package/source to inspect | Comparison target | Current equivalence status |
| --- | --- | --- | --- |
| `adaptive_lasso` | `glmnet`, `R/glmnet.R`: <https://rdrr.io/cran/glmnet/src/R/glmnet.R> | Adaptive lasso is expressible in R by computing initial weights and passing them as `penalty.factor` to lasso. | Same fixed-weight objective after macroforecast standardizes `X`, rescales columns, and normalizes weights to mean one; macroforecast fits one chosen `alpha`, while `glmnet` usually builds a lambda path. |
| `adaptive_elastic_net` | `glmnet`, `R/glmnet.R`: <https://rdrr.io/cran/glmnet/src/R/glmnet.R> | Adaptive elastic net uses the same adaptive weights with elastic-net mixing. | Same fixed-weight idea with mean-one penalty weights; macroforecast delegates the final fit to sklearn `ElasticNet` rather than glmnet's path solver. |
| `nonneg_ridge` | `nnls`, `R/nnls.R`: <https://rdrr.io/cran/nnls/src/R/nnls.R>; also `glmnet` lower bounds | NNLS solves least squares under coefficient non-negativity. | Equivalent to NNLS on the augmented design `[X; sqrt(alpha) I]` and response `[y; 0]`, after optional centering. |
| `shrink_to_target_ridge` | `penalized`: <https://search.r-project.org/CRAN/refmans/penalized/html/penalized.html>; target ridge family in `rags2ridges`: <https://rdrr.io/cran/rags2ridges/src/R/rags2ridges.R> | Compare target-shrinkage/tikhonov logic, not an identical regression API. | No exact same R regression callable found. macroforecast solves `||y-Xb||^2 + alpha ||b-b0||^2` with optional simplex/nonnegative constraints. |
| `fused_difference_ridge` | fused L2 ridge family in `rags2ridgesFused.R`: <https://rdrr.io/cran/rags2ridges/src/R/rags2ridgesFused.R> | Compare L2 fusion/smoothness penalty structure. | Not identical domain: R source is primarily fused ridge for precision matrices; macroforecast applies an L2 finite-difference penalty directly to regression coefficients. |
| `component_aggregation`, `albacore_components` | `assemblage`, `R/assemblage_v240228.R`: `nonneg.ridge.sum1` | Nonnegative component weights, optional target-weight shrinkage, sum-to-one basket constraint. | Same fixed-alpha objective family as the R CVXR fit: SSE plus feature-std-scaled target shrinkage, `w >= 0`, and `sum(w)=1`. R owns block CV for lambda; macroforecast delegates alpha selection to model selection/forecasting. |
| `rank_aggregation`, `albacore_ranks` | `assemblage`, `R/assemblage_v240228.R`: `x.transformation`, `nonneg.ridge.meanD` | Sort components into rank space, estimate nonnegative smooth rank weights with a mean-matching constraint. | Same fixed-alpha rank objective family: row sorting, fused difference penalty on scaled rank weights, `w >= 0`, and `mean(Xw)=mean(y)`. |
| `supervised_aggregation`, `assemblage_regression` | `assemblage`, `R/assemblage_v240228.R`: `assemblage`, `nonneg.ridge`, `nonneg.ridge.mean`, `nonneg.ridge.sum1`, `nonneg.ridge.meanD` | Generic component/rank supervised aggregation. | Exposes the reusable primitives without requiring inflation data. Paper-specific inflation semantics live in the `albacore_*` wrappers. |
| `random_walk_ridge` | `walker`, `R/walker.R`: <https://rdrr.io/cran/walker/man/walker.html> | Random-walk coefficients in a time-varying regression. | Same modeling prior idea, different inference: `walker` is Bayesian/state-space via Stan; macroforecast computes the penalized least-squares MAP-style coefficient path and predicts with the final vector. |
| `tvp_ridge` | `TVPRidge`, local source `wiki/raw/paper_code/coulombe_site_github_20260530/tvpridge/R/MV2SRR_v210407.R`; upstream <https://github.com/philgoucou/tvpridge> | Goulet Coulombe TVP ridge / two-step ridge regression. | Direct Python port of the R `tvp.ridge` pipeline: `Zfun` expansion, `dualGRR` dual/primal generalized ridge, R-style random-fold CV helpers, 2SRR coefficient-innovation reweighting, residual-volatility reweighting, and R return fields. |
| `group_lasso` | `grpreg`, `R/grpreg.R`: <https://rdrr.io/cran/grpreg/src/R/grpreg.R> | Group penalty over coefficient blocks. | Same group-lasso penalty family for Gaussian loss; macroforecast uses a single-alpha proximal-gradient solver rather than a full regularization path. |
| `sparse_group_lasso` | `sparsegl`, `R/sparsegl.R`: <https://rdrr.io/cran/sparsegl/src/R/sparsegl.R> | Sparse group lasso objective with group and feature-level penalties. | Same penalty decomposition; macroforecast uses one selected `alpha` and `l1_ratio`, while `sparsegl` is a full path solver with additional bounds/families. |
| `glmboost` | `mboost`, `R/mboost.R`: <https://rdrr.io/cran/mboost/src/R/mboost.R>; component learner in `R/bolscw.R`; Goulet Coulombe et al. (2021) Appendix A.6 for random candidate sampling | Componentwise gradient/L2 boosting. | Same Gaussian componentwise L2 update: center predictors by default, select the base learner by normalized correlation, and apply shrinkage. The paper's per-step random candidate rule is expressed as `candidate_sampling="random"`, `candidate_fraction=1/3`, `candidate_cap=200`, `candidate_rounding="floor"`, not as a hidden preset. macroforecast omits formula handling, weights, families, hat values, and stopping machinery. |

The direct implementations should be reviewed against the objective, scaling,
intercept handling, penalty normalization, and solver stopping rule separately.
Matching an R package name is not enough: several R implementations solve a
path problem, a Bayesian state-space problem, or a matrix-estimation problem,
whereas macroforecast exposes a single callable forecasting estimator.

### ols

```python
macroforecast.models.ols(X, y)
```

Fits ordinary least squares.

| Item | Value |
| --- | --- |
| Input | `X`, `y` |
| Output | `ModelFit` |
| Backend | `sklearn.linear_model.LinearRegression` |
| Default params | none |
| Tunable params | none |
| Preset search spaces | none |

### ridge

```python
macroforecast.models.ridge(X, y, *, alpha=1.0)
```

Fits ridge regression with an L2 penalty.

Backend: `sklearn.linear_model.Ridge`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | L2 penalty strength. |

| Preset | `alpha` |
| --- | --- |
| `small` | `(0.01, 0.1, 1.0)` |
| `standard` | `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

Default model-selection method: `cv_path`.

### nonneg_ridge

```python
macroforecast.models.nonneg_ridge(X, y, *, alpha=1.0, fit_intercept=True)
```

Fits ridge regression with coefficients constrained to be non-negative. This
uses SciPy NNLS on an augmented ridge design, so it does not require `cvxpy`.

Backend: package-native augmented ridge design plus `scipy.optimize.nnls`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | L2 penalty strength. |
| `fit_intercept` | `True` | fixed by preset | Fit an intercept outside the constrained coefficients. |

Default model-selection method: `cv_path`.

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

Backend: package-native objective plus `scipy.optimize.minimize(method="SLSQP")`.

R comparison: this is a regression analogue of target-ridge/Tikhonov
shrinkage. `rags2ridges` uses target ridge for covariance and precision
matrices, not a direct `X, y` regression callable, but the same target idea is
present: shrink an estimated parameter object toward a target rather than
toward zero. In the unconstrained regression case, macroforecast solves

```text
min_beta ||y - X beta||^2 + alpha ||beta - beta0||^2
```

with closed-form normal equation

```text
(X'X + alpha I) beta = X'y + alpha beta0
```

after optional centering for the intercept. `simplex=True` changes the problem
into a forecast-combination form: coefficients must sum to one, no intercept is
fit, and `prior_target=None` means a uniform target vector.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Strength of shrinkage toward `prior_target`. |
| `prior_target` | `None` | fixed by preset | Scalar, sequence, mapping, or `None`. |
| `simplex` | `False` | fixed by preset | Constrain coefficients to sum to one. |
| `nonneg` | `False` | fixed by preset | Constrain coefficients to be non-negative. |
| `fit_intercept` | `True` | fixed by preset | Fit an intercept unless `simplex=True`. |
| `max_iter` | `1000` | fixed by preset | SLSQP iteration cap. |
| `tol` | `1e-9` | fixed by preset | SLSQP tolerance. |

Default model-selection method: `cv_path`.

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

Backend: package-native finite-difference objective plus SLSQP.

R comparison: `rags2ridges::ridgeP.fused` uses a fused L2 penalty for related
precision matrices. `fused_difference_ridge()` uses the same penalty idea on a
single ordered regression-coefficient vector. With no sign or equality
constraints, macroforecast solves

```text
min_beta ||y - X beta||^2 + alpha ||D beta||^2
```

where `D` is the finite-difference matrix over adjacent coefficients. The
closed-form normal equation is

```text
(X'X + alpha D'D) beta = X'y
```

after optional centering for the intercept. `mean_equality=True` is a
macro-forecasting conservation variant; it constrains fitted and observed sums
to match and is intentionally outside the rags2ridges precision-matrix API.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Strength of the smoothness penalty. |
| `difference_order` | `1` | fixed by preset | Finite-difference order applied to coefficients. |
| `mean_equality` | `False` | fixed by preset | Constrain fitted and observed sums to match. |
| `nonneg` | `False` | fixed by preset | Constrain coefficients to be non-negative. |
| `fit_intercept` | `True` | fixed by preset | Fit an intercept unless `mean_equality=True`. |
| `max_iter` | `1000` | fixed by preset | SLSQP iteration cap. |
| `tol` | `1e-9` | fixed by preset | SLSQP tolerance. |

Default model-selection method: `cv_path`.

## Assemblage / Supervised Aggregation

This family is derived from Goulet Coulombe, Klieber, Barrette, and Goebel,
*Maximally Forward-Looking Core Inflation*, and the R package `assemblage`.
The package splits the paper model into generic reusable primitives plus thin
inflation-specific wrappers.

The generic problem is:

```text
given components X_t and future aggregate target y_t,h,
learn weights w so X_t w predicts y_t,h
```

This is not ordinary ridge in disguise. The weights can be constrained to be
nonnegative, sum to one, match the target mean, shrink toward reference basket
weights, or vary smoothly across ranks. For inflation, those weights form an
Albacore core-inflation measure. Outside inflation, the same functions can
aggregate sectors, states, industries, survey items, or regional indicators.

### supervised_aggregation

```python
macroforecast.models.supervised_aggregation(
    X,
    y,
    *,
    space="component",
    penalty="ridge",
    alpha=1.0,
    reference_weights=None,
    nonneg=True,
    simplex=False,
    mean_match=False,
    difference_order=1,
    fit_intercept=False,
    penalty_scale="feature_std",
    max_iter=1000,
    tol=1e-9,
)
```

| Parameter | Default | Choices | Meaning |
| --- | --- | --- | --- |
| `space` | `"component"` | `"component"`, `"rank"` | Use named components or row-wise sorted order statistics. |
| `penalty` | `"ridge"` | `"ridge"`, `"target_shrinkage"`, `"fused_difference"` | Coefficient penalty family. |
| `alpha` | `1.0` | nonnegative float | Penalty strength; tune with `model_selection` or `forecasting`. |
| `reference_weights` | `None` | mapping, sequence, `Series`, or `None` | Target weights for `target_shrinkage`. |
| `nonneg` | `True` | bool | Enforce `w_j >= 0`. |
| `simplex` | `False` | bool | Enforce `sum(w)=1`. |
| `mean_match` | `False` | bool | Enforce `mean(Xw)=mean(y)`. |
| `difference_order` | `1` | positive int | Difference order for fused rank weights. |
| `fit_intercept` | `False` | bool | Fit an intercept outside the aggregation weights when no equality constraint is active. |
| `penalty_scale` | `"feature_std"` | `"feature_std"`, `"none"` | Match the R assemblage convention by scaling penalties with feature standard deviations. |

Output: `ModelFit`. The fitted estimator exposes `coef_`, `weights_`, and,
for rank space, `rank_weight_curve_`. Diagnostics include fitted values,
residuals, metrics, and coefficient weights.

### component_aggregation

```python
macroforecast.models.component_aggregation(
    X,
    y,
    *,
    alpha=1.0,
    reference_weights=None,
    penalty=None,
    simplex=True,
    nonneg=True,
    penalty_scale="feature_std",
    max_iter=1000,
    tol=1e-9,
)
```

Component-space aggregation estimates weights on named columns. With
`reference_weights` supplied, `penalty=None` selects `target_shrinkage`, making
this the generic version of Albacorecomps. Without reference weights, it is a
nonnegative simplex ridge basket.

R source cue: `nonneg.ridge.sum1` in `assemblage_v240228.R`.

### rank_aggregation

```python
macroforecast.models.rank_aggregation(
    X,
    y,
    *,
    alpha=1.0,
    penalty="fused_difference",
    mean_match=True,
    nonneg=True,
    difference_order=1,
    penalty_scale="feature_std",
    max_iter=1000,
    tol=1e-9,
)
```

Rank-space aggregation sorts each row of `X` before fitting, then learns
weights on `rank_1`, `rank_2`, ... rather than on named components. This is a
generic supervised trimmed-mean model. The fitted object stores
`estimator.rank_weight_curve_`, a table with rank, percentile, and weight.

R source cue: `x.transformation` plus `nonneg.ridge.meanD`.

### assemblage_regression

```python
macroforecast.models.assemblage_regression(
    X,
    y,
    *,
    space="component",
    alpha=1.0,
    reference_weights=None,
    penalty=None,
    max_iter=1000,
    tol=1e-9,
)
```

Convenience wrapper over `component_aggregation()` and `rank_aggregation()`.
Use it when the model family is known to be assemblage-style but the final
choice between component and rank space is part of the experiment design.

### albacore_components

```python
macroforecast.models.albacore_components(
    X,
    y,
    *,
    reference_weights=None,
    alpha=1.0,
    max_iter=1000,
    tol=1e-9,
)
```

Inflation-specific wrapper for component-space Albacore. `X` should be a panel
of price-component changes, `y` should be the forward average headline
inflation target, and `reference_weights` should be official basket or
expenditure weights when available. The wrapper sets `nonneg=True`,
`simplex=True`, `penalty="target_shrinkage"`, and `fit_intercept=False`.

### albacore_ranks

```python
macroforecast.models.albacore_ranks(
    X,
    y,
    *,
    alpha=1.0,
    difference_order=1,
    max_iter=1000,
    tol=1e-9,
)
```

Inflation-specific wrapper for rank-space Albacore. `X` should be price
component changes and `y` should be the forward average headline inflation
target. The wrapper sorts components row by row, estimates nonnegative fused
rank weights, and enforces the Albacoreranks mean-matching constraint.

### Low-Level Solver Helpers

These return a weight `Series` rather than a full `ModelFit`:

| Function | Meaning |
| --- | --- |
| `solve_nonnegative_ridge(X, y, alpha=...)` | R `nonneg.ridge`-style nonnegative ridge weights. |
| `solve_simplex_ridge(X, y, alpha=...)` | Nonnegative weights constrained to sum to one. |
| `solve_target_shrinkage_ridge(X, y, reference_weights=..., alpha=...)` | R `nonneg.ridge.sum1`-style component basket weights. |
| `solve_mean_aligned_ridge(X, y, alpha=...)` | R `nonneg.ridge.mean`-style nonnegative mean-aligned weights. |
| `solve_fused_difference_ridge(X, y, alpha=...)` | R `nonneg.ridge.meanD`-style fused rank-weight primitive. |

These helpers are deliberately not inflation-specific. They exist so users can
compose custom supervised aggregation models without taking the Albacore
wrappers.

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

Backend: package-native expanded design solved by `numpy.linalg.lstsq`.

R comparison: `walker::walker_rw1` is the closest R source. It treats
coefficients as random-walk state variables and estimates a Bayesian posterior
with Stan / state-space smoothing. `random_walk_ridge()` keeps the same RW1
prior idea but solves the penalized least-squares MAP-style objective as one
augmented linear system over the full coefficient path:

```text
min_{beta_1,...,beta_T}
sum_t (y_t - x_t beta_t)^2
+ initial_alpha ||beta_1||^2
+ alpha sum_t ||beta_t - beta_{t-1}||^2
```

The fitted `coef_path_` is the estimated path. `predict()` uses only the final
coefficient vector, because this callable is a deterministic forecasting model,
not a posterior simulation or Kalman-smoothing interface. `fit_intercept=True`
centers the fit and recovers a static intercept from the final coefficient
vector.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Penalty on adjacent coefficient changes. |
| `initial_alpha` | `1.0` | fixed by preset | Penalty on the first coefficient vector. |
| `fit_intercept` | `True` | fixed by preset | Fit an intercept outside the time-varying coefficient path. |

Default model-selection method: `cv_path`.

### tvp_ridge

```python
macroforecast.models.tvp_ridge(
    X,
    y,
    *,
    lambda_candidates=None,
    oosX=None,
    lambda2=0.1,
    kfold=5,
    cv_plot=False,
    cv_2srr=True,
    sig_u_param=0.75,
    sig_eps_param=0.75,
    ols_prior=False,
    random_state=1071,
    use_garch=True,
)
```

Fits Philippe Goulet Coulombe's TVP ridge / two-step ridge regression
estimator from **Time-varying parameters as ridge regressions**
(International Journal of Forecasting, DOI
<https://doi.org/10.1016/j.ijforecast.2024.08.006>). The implementation is a
Python port of the R package `TVPRidge`, source file
`R/MV2SRR_v210407.R`, local snapshot:

```text
wiki/raw/paper_code/coulombe_site_github_20260530/tvpridge/R/MV2SRR_v210407.R
```

This is not a thin wrapper around `random_walk_ridge()`. Both models use a
random-walk coefficient idea, but `tvp_ridge()` ports the paper's full
estimator:

| Stage | R function | Python implementation |
| --- | --- | --- |
| Basis expansion | `Zfun` | `_tvp_z_basis` |
| Generalized ridge solve | `dualGRR` | `_dual_generalized_ridge` |
| Initial lambda CV | `CV.KF.MV` | `_cv_kfold_multivariate` |
| Second-step lambda CV | `cv.univariate` | `_cv_univariate` |
| Dropout correction | `cumul_zeros` | `_cumul_zeros` |
| Public callable | `tvp.ridge` | `tvp_ridge` / `TVPRidgeRegressor` |

The estimated path solves the paper's time-varying parameter ridge problem:

```text
min_{beta_1,...,beta_T}
sum_t (y_t - x_t beta_t)^2
+ lambda * sum_t ||beta_t - beta_{t-1}||^2
+ lambda2 * ||beta_0||^2
```

`lambda` controls the amount of time variation. Large values force smoother
coefficient paths; small values allow more movement. `lambda2` is the soft
penalty on the starting coefficient values. The R code standardizes by sample
standard deviation without centering; macroforecast follows that convention and
rescales coefficient paths and fitted values back to the original data scale.

The 2SRR step follows the R package logic. First, the homogeneous ridge TVP is
estimated. Then coefficient innovations are used to build coefficient-specific
variance weights, and residual volatility weights are optionally estimated by a
GARCH(1,1) backend. The model is refit with those weights. If Python package
`arch` is unavailable or the GARCH fit fails, residual-volatility weights fall
back to ones and the reason is recorded in
`fit.estimator.diagnostics_["garch_status"]`; the ridge/2SRR fit still runs.

Input:

| Argument | Required | Expected object | Meaning |
| --- | --- | --- | --- |
| `X` | yes | pandas DataFrame, NumPy array, or `FeatureSet` | Predictor matrix with shape `T x K`. |
| `y` | yes unless `X` is a `FeatureSet` | pandas Series or one-column DataFrame for the public `ModelFit` wrapper | Target series aligned to `X`. |
| `lambda_candidates` | no | sequence of positive floats or `None` | Candidate values for the time-variation penalty. `None` uses the R default grid. |
| `oosX` | no | one predictor vector of length `K` | Optional one-step forecast using the final coefficient vector. |

Output:

| Object | Type | Contents |
| --- | --- | --- |
| return value | `ModelFit` | Standard macroforecast fitted model wrapper. |
| `fit.estimator.betas_rr_` | NumPy array, shape `M x (K+1) x T` | First-step ridge TVP coefficient paths, original scale. |
| `fit.estimator.betas_2srr_` | NumPy array, shape `M x (K+1) x T` | 2SRR coefficient paths, original scale. |
| `fit.estimator.lambdas_` | NumPy array | Initial CV lambda for each target. |
| `fit.estimator.lambda_step2_` | NumPy array | Second-step lambda used after reweighting. |
| `fit.estimator.yhat_rr_` | DataFrame | In-sample first-step fitted values. |
| `fit.estimator.yhat_2srr_` | DataFrame | In-sample 2SRR fitted values. |
| `fit.estimator.sig_eps_` | DataFrame | Normalized residual-volatility weights. |
| `fit.estimator.forecast_` | NumPy array | Optional forecast when `oosX` is supplied. |
| `fit.estimator.coef_path_` | DataFrame | Final 2SRR path for the first target, excluding intercept. |
| `fit.estimator.coef_path_full_` | DataFrame | MultiIndex coefficient path including intercept and target names. |

Prediction rule:

| Call | Behavior |
| --- | --- |
| `fit.predict(X_train)` with the original training index | Returns the time-varying in-sample `yhat_2srr_` path. |
| `fit.predict(X_new)` with new rows | Uses the final estimated coefficient vector `beta_T`. |

Default parameters:

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `lambda_candidates` | `exp(linspace(-6, 20, 15))` | yes | R default candidate grid for the time-variation penalty. |
| `lambda2` | `0.1` | fixed by preset | Soft penalty on starting coefficient values. |
| `kfold` | `5` | fixed by preset | Number of random CV folds. |
| `cv_2srr` | `True` | fixed by preset | Re-run lambda CV after variance reweighting. |
| `sig_u_param` | `0.75` | fixed by preset | Shrinkage exponent for coefficient-innovation variance weights. |
| `sig_eps_param` | `0.75` | fixed by preset | Shrinkage exponent for residual-volatility weights. |
| `ols_prior` | `False` | fixed by preset | Shrink starting coefficients toward OLS rather than zero. |
| `random_state` | `1071` | fixed by preset | Fold seed matching the R source's `set.seed(1071)` convention. |
| `use_garch` | `True` | fixed by preset | Use optional Python `arch` GARCH(1,1) for residual-volatility weights. |

R parity notes:

| Topic | Status |
| --- | --- |
| Standardization | Matches R: divide `X` and `Y` by sample standard deviation, no centering. |
| Basis columns | Matches R `Zfun`: innovation blocks by coefficient, then static intercept/predictor block. |
| Dual/primal solve | Matches R `dualGRR` algebra, with `numpy.linalg.solve` and pseudo-inverse fallback for singular systems. |
| CV folds | Same random-fold design and default seed, but NumPy's RNG is not bit-identical to R's `sample()`. |
| GARCH volatility | Uses optional Python `arch` backend; if unavailable, records fallback and continues with homogeneous residual weights. |
| Multivariate `Y` | Estimator internals preserve `M x (K+1) x T` arrays; the public `ModelFit` wrapper is optimized for one target, consistent with macroforecast's standard supervised callable. |

Default model-selection method: `cv_path`.

### lasso

```python
macroforecast.models.lasso(
    X,
    y,
    *,
    alpha=1.0,
    max_iter=20000,
    standardize=False,
)
```

Fits lasso regression with an L1 penalty. There is no `lasso_path()` model
callable; use `get_model("lasso")` and `model_selection.select_params()`.

Backend: `sklearn.linear_model.Lasso`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | L1 penalty strength. |
| `max_iter` | `20000` | fixed by preset | Optimization iteration cap. |
| `standardize` | `False` | fixed by preset | Standardize predictors inside the fitted estimator. Defaults to `False` because preprocessing/window policy usually owns scaling. Set `True` for lasso-style replications where scaling must be fit inside each model window. |

| Preset | `alpha` |
| --- | --- |
| `small` | `(0.01, 0.1, 1.0)` |
| `standard` | `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

Default model-selection method: `cv_path`.

### elastic_net

```python
macroforecast.models.elastic_net(
    X,
    y,
    *,
    alpha=1.0,
    l1_ratio=0.5,
    max_iter=20000,
    standardize=False,
)
```

Fits elastic net regression.

Backend: `sklearn.linear_model.ElasticNet`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Overall penalty strength. |
| `l1_ratio` | `0.5` | yes | L1 share of the elastic-net penalty. |
| `max_iter` | `20000` | fixed by preset | Optimization iteration cap. |
| `standardize` | `False` | fixed by preset | Standardize predictors inside the fitted estimator. Defaults to `False`; set `True` for glmnet/MATLAB-style elastic-net replications whose penalty grid assumes window-local standardized predictors. |

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
    normalize_weights=True,
    max_iter=20000,
    tol=1e-4,
    random_state=None,
)
```

Fits adaptive lasso. The model first estimates initial coefficients with
`initial="ridge"` or `initial="ols"`, builds feature weights
`1 / (abs(beta_init) + eps) ** gamma`, and fits lasso on weighted standardized
predictors. Predictions are mapped back to the original target scale.

Backend: macroforecast adaptive-weight construction plus final
`sklearn.linear_model.Lasso`.

R/glmnet comparison: `glmnet` accepts the same idea through `penalty.factor`.
It internally rescales penalty factors to sum to the number of predictors, so
macroforecast defaults to `normalize_weights=True`, which rescales adaptive
weights to mean one before fitting the final lasso. Set
`normalize_weights=False` only when the absolute weight scale should change the
effective penalty strength.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Final adaptive lasso penalty strength. |
| `gamma` | `1.0` | yes | Exponent applied to initial coefficient weights. |
| `initial` | `"ridge"` | manual | Initial model: `"ridge"` or `"ols"`. |
| `initial_alpha` | `1.0` | fixed by preset | Initial ridge penalty. |
| `eps` | `1e-4` | fixed by preset | Small denominator floor for adaptive weights. |
| `normalize_weights` | `True` | fixed by preset | Rescale adaptive weights to mean one, matching `glmnet` penalty-factor scaling. |
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
    normalize_weights=True,
    max_iter=20000,
    tol=1e-4,
    random_state=None,
)
```

Fits an adaptive elastic-net variant with the same initial coefficient weights
as `adaptive_lasso`, followed by an elastic-net fit on weighted standardized
predictors.

Backend: macroforecast adaptive-weight construction plus final
`sklearn.linear_model.ElasticNet`.

R/glmnet comparison: this is the elastic-net analogue of adaptive lasso.
`normalize_weights=True` gives the same mean-one penalty-factor convention as
`glmnet`; the remaining difference is solver style, because macroforecast fits
one selected `alpha` while `glmnet` usually estimates a regularization path.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Final adaptive elastic-net penalty strength. |
| `l1_ratio` | `0.5` | yes | L1 share of the final elastic-net penalty. |
| `gamma` | `1.0` | yes | Exponent applied to initial coefficient weights. |
| `initial` | `"ridge"` | manual | Initial model: `"ridge"` or `"ols"`. |
| `initial_alpha` | `1.0` | fixed by preset | Initial ridge penalty. |
| `eps` | `1e-4` | fixed by preset | Small denominator floor for adaptive weights. |
| `normalize_weights` | `True` | fixed by preset | Rescale adaptive weights to mean one, matching `glmnet` penalty-factor scaling. |
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

Backend: package-native proximal-gradient solver.

R comparison: this follows the Gaussian group-lasso objective used by
`grpreg::grpreg(..., penalty = "grLasso")`: standardized predictors,
group-level L2 shrinkage, and default group weights proportional to
`sqrt(group_size)`. macroforecast fits one selected `alpha` and does not
reproduce `grpreg`'s full path solver, GLM families, C backend, or within-group
orthogonalization step.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `groups` | `None` | manual | One group label per predictor. |
| `alpha` | `1.0` | yes | Group penalty strength. |
| `group_weights` | `None` | manual | Optional group penalty weights; default is `sqrt(group_size)`. |
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

Backend: package-native proximal-gradient solver.

R comparison: this follows the sparse-group penalty decomposition used by
`sparsegl::sparsegl`: a feature-level L1 part plus a group L2 part with default
`sqrt(group_size)` group weights. macroforecast fits one selected `alpha` and
`l1_ratio`; it does not reproduce `sparsegl`'s full lambda path, bounds, GLM
families, or C++ backend.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `groups` | `None` | manual | One group label per predictor. |
| `alpha` | `1.0` | yes | Total sparse-group penalty strength. |
| `l1_ratio` | `0.5` | yes | Feature-level L1 share. |
| `group_weights` | `None` | manual | Optional group penalty weights; default is `sqrt(group_size)`. |
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
| Backend | `sklearn.linear_model.BayesianRidge` |
| Default params | sklearn defaults |
| Tunable params | none in the clean preset catalog |
| Preset search spaces | none |

### huber

```python
macroforecast.models.huber(X, y, *, epsilon=1.35, max_iter=1000)
```

Fits robust Huber regression.

Backend: `sklearn.linear_model.HuberRegressor`.

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

These models are external sklearn wrappers, not package-native numerical
solvers. They live in `macroforecast.models.nonparametric` and are re-exported
from `macroforecast.models` and top-level `macroforecast`.

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

Backend: `sklearn.kernel_ridge.KernelRidge`.

R parity is intentionally not claimed for this callable. It is a thin sklearn
backend wrapper; macroforecast owns only the pandas `X, y` contract, `ModelFit`
metadata/diagnostics, and search-space registration.

| Item | Value |
| --- | --- |
| Input | `X`, `y` |
| Output | `ModelFit` |
| Internal scaling | none |
| `ModelSpec.requires_scaling` | `True` |
| Default model-selection method | `random` |

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | yes | Ridge penalty strength. |
| `kernel` | `"linear"` | search option | Kernel name. |
| `gamma` | `None` | search option | Kernel coefficient. |
| `degree` | `3` | search option | Polynomial kernel degree. |
| `coef0` | `1.0` | fixed by preset | Independent term for polynomial/sigmoid kernels. |

| Preset | `alpha` | `kernel` | extra searched params |
| --- | --- | --- | --- |
| `small` | `(0.1, 1.0, 10.0)` | `("linear", "rbf")` | none |
| `standard` | `(0.01, 0.1, 1.0, 10.0)` | `("linear", "rbf", "poly")` | `gamma=(None, 0.01, 0.1)` |
| `wide` | `(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` | `("linear", "rbf", "poly", "sigmoid")` | `gamma=(None, 0.001, 0.01, 0.1, 1.0)`, `degree=(2, 3, 4)` |

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

Backend: `sklearn.neighbors.KNeighborsRegressor`.

R parity is intentionally not claimed for this callable. It is a thin sklearn
backend wrapper; macroforecast owns only the pandas `X, y` contract, small-window
`n_neighbors` resolution, `ModelFit` metadata/diagnostics, and search-space
registration.

If the requested `n_neighbors` is larger than the fitted sample size,
macroforecast resolves it down to `n_obs` before constructing the sklearn
estimator. The fit metadata records the effective `n_neighbors` and, when
different, `requested_n_neighbors`. This avoids small-window forecasting runs
failing at prediction time.

| Item | Value |
| --- | --- |
| Input | `X`, `y` |
| Output | `ModelFit` |
| Internal scaling | none |
| `ModelSpec.requires_scaling` | `True` |
| Default model-selection method | `random` |

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_neighbors` | `5` | yes | Number of nearest neighbors. |
| `weights` | `"uniform"` | yes | `"uniform"` or `"distance"`. |
| `metric` | `"minkowski"` | fixed by preset | Distance metric. |
| `p` | `2` | search option | Minkowski distance order. |

| Preset | `n_neighbors` | `weights` | extra searched params |
| --- | --- | --- | --- |
| `small` | `(3, 5, 10)` | `("uniform", "distance")` | none |
| `standard` | `(3, 5, 10, 20)` | `("uniform", "distance")` | `p=(1, 2)` |
| `wide` | `(1, 3, 5, 10, 20, 40)` | `("uniform", "distance")` | `p=(1, 2)` |

## Linear Boosting

### glmboost

```python
macroforecast.models.glmboost(
    X,
    y,
    *,
    n_iter=100,
    learning_rate=0.1,
    center=True,
    candidate_sampling="all",
    candidate_count=None,
    candidate_fraction=None,
    candidate_cap=None,
    candidate_min=1,
    candidate_rounding="floor",
    random_state=None,
)
```

Fits componentwise L2 boosting with linear base learners.

Backend: package-native componentwise L2 boosting loop. The R comparison target
is `mboost::glmboost`. macroforecast implements the matrix-input Gaussian path:
predictors are centered by default, each iteration selects the column with the
largest normalized correlation with the current residual, and the selected
least-squares coefficient is shrunk by `learning_rate`.
Candidate sampling is deliberately decomposed into separate arguments. For
Goulet Coulombe, Leroux, Stevanovic, and Surprenant (2021), Appendix A.6, use
`candidate_sampling="random"`, `candidate_fraction=1/3`,
`candidate_cap=200`, and `candidate_rounding="floor"`, which gives
`m=min(200, floor(n_features / 3))` sampled predictors at each boosting step.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_iter` | `100` | yes | Number of boosting iterations. |
| `learning_rate` | `0.1` | yes | Shrinkage applied to each update. |
| `center` | `True` | no | Center predictors before componentwise updates, matching `mboost::glmboost(..., center = TRUE)`. |
| `candidate_sampling` | `"all"` | fixed by preset | `"all"` searches every usable predictor each step; `"random"` samples a candidate subset before selecting the best base learner. |
| `candidate_count` | `None` | fixed by preset | Fixed sampled candidate count when `candidate_sampling="random"`. Mutually exclusive with `candidate_fraction`. |
| `candidate_fraction` | `None` | fixed by preset | Fraction of predictors sampled each step when `candidate_sampling="random"`. |
| `candidate_cap` | `None` | fixed by preset | Maximum sampled candidate count after resolving `candidate_count` or `candidate_fraction`. |
| `candidate_min` | `1` | fixed by preset | Minimum sampled candidate count. |
| `candidate_rounding` | `"floor"` | fixed by preset | Rounding rule for `candidate_fraction`: `"floor"`, `"ceil"`, or `"round"`. |
| `random_state` | `None` | fixed by preset | Seed for per-step candidate feature sampling when `candidate_sampling="random"`. |

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
model-owned hyperparameters through `model_selection`, and let `window` decide the
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
    model_selection=macroforecast.model_selection.grid({"C": [0.1, 1.0], "epsilon": [0.01, 0.1]}),
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

Backend: `sklearn.svm.SVR`.

`kernel="precomputed"` is intentionally not supported because macroforecast
`ModelFit` expects `X` to be a feature matrix with stable column names. Use
`"linear"`, `"poly"`, `"rbf"`, or `"sigmoid"`.

| Item | Value |
| --- | --- |
| Input | `X`, `y` |
| Output | `ModelFit` |
| Internal scaling | none |
| `ModelSpec.requires_scaling` | `True` |
| Default model-selection method | `random` |

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `kernel` | `"rbf"` | fixed by preset | Kernel: `"linear"`, `"poly"`, `"rbf"`, or `"sigmoid"`. |
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

Backend: `sklearn.svm.LinearSVR`.

| Item | Value |
| --- | --- |
| Input | `X`, `y` |
| Output | `ModelFit` |
| Internal scaling | none |
| `ModelSpec.requires_scaling` | `True` |
| Default model-selection method | `random` |

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `C` | `1.0` | yes | Inverse regularization strength. |
| `epsilon` | `0.0` | yes | Epsilon-insensitive tube width. |
| `loss` | `"epsilon_insensitive"` | fixed by preset | LinearSVR loss function. |
| `tol` | `1e-4` | fixed by preset | Optimization tolerance. |
| `max_iter` | `10000` | fixed by preset | Solver iteration cap. |
| `random_state` | `0` | fixed by preset | Random seed; can be `None`. |

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

Backend: `sklearn.svm.NuSVR`.

`kernel="precomputed"` is intentionally not supported for the same feature-matrix
contract reason as `svr()`.

| Item | Value |
| --- | --- |
| Input | `X`, `y` |
| Output | `ModelFit` |
| Internal scaling | none |
| `ModelSpec.requires_scaling` | `True` |
| Default model-selection method | `random` |

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `kernel` | `"rbf"` | fixed by preset | Kernel: `"linear"`, `"poly"`, `"rbf"`, or `"sigmoid"`. |
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

`nn`, `lstm`, `gru`, `transformer`, `hemisphere_nn`, and `density_hnn` are all torch-backed neural-network models and require
`macroforecast[deep]`. `nn` is the feed-forward neural network for tabular
feature matrices; `lstm` and `gru` are recurrent neural networks that consume
trailing row sequences; `transformer` is a compact Transformer encoder using
the same trailing-row sequence contract. `hemisphere_nn` is a compact bagged
dual-head network for mean and variance forecasts, while `density_hnn` follows
the Aionx/Paper DensityHNN procedure with prior-DNN OOB volatility emphasis and
OOB volatility recalibration. The `deep` extra is intentionally separate from
`macroforecast[all]` because torch is large and platform-sensitive.

Torch recurrent example:

```python
result = macroforecast.forecasting.run(
    panel,
    "lstm",
    features=features,
    window=macroforecast.window.last_block(validation_size=24),
    params={"lstm": {"sequence_length": 4, "hidden_size": 32, "device": "auto"}},
    model_selection={"lstm": None},
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
    model_selection=macroforecast.model_selection.grid({
        "hidden_layer_sizes": [(32,), (64,)],
        "weight_decay": [0.0, 0.0001],
    }),
)
```

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `hidden_layer_sizes` | `(100,)` | yes | Feed-forward hidden layer widths. |
| `activation` | `"relu"` | fixed by preset | Activation: `"identity"`, `"logistic"`, `"sigmoid"`, `"tanh"`, `"relu"`, or `"gelu"`. |
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

### density_hnn

```python
macroforecast.models.density_hnn(
    X,
    y,
    *,
    common_layers=2,
    mean_layers=2,
    volatility_layers=2,
    prior_layers=3,
    neurons=400,
    dropout=0.2,
    learning_rate=0.001,
    max_epochs=100,
    n_estimators=100,
    prior_estimators=50,
    subsample=0.8,
    block_size=8,
    volatility_emphasis=None,
    rescale_volatility=True,
    patience=15,
    random_state=0,
    device="auto",
    quantile_levels=(0.05, 0.5, 0.95),
    volatility_clip=0.05,
)
```

Fits the Density Hemisphere Neural Network from Goulet Coulombe, Frenette, and
Klieber, "From Reactive to Proactive Volatility Modeling with Hemisphere Neural
Networks" (Journal of Applied Econometrics, 2025). The implementation is a
torch-native port of the public Aionx `DensityHNN` logic, not a TensorFlow
dependency wrapper. It is included because the method is a macro density
forecast model: `macroforecast` uses it to produce conditional means,
conditional variances, volatility forecasts, and normal-approximation
quantiles. It does not create portfolio weights.

The Aionx source-code correspondence is:

| Aionx source item | `macroforecast` implementation |
| --- | --- |
| `aionx.models.DensityHNN.prior_dnn_architecture` | `prior_estimators` plain DNN ensemble fitted before the HNN. |
| `aionx.bootstrap.TimeSeriesBlockBootstrap` | `block_size` and `subsample` create time-series block bootstrap samples. |
| `aionx.kerasnn.ensemble.OutOfBagPredictor` | OOB forecasts use the Aionx denominator formula `sum(oob forecast) / ((1 - subsample) * n_estimators)`. |
| `aionx.models.DensityHNN.base_architecture` | shared common core plus mean and volatility hemispheres; the volatility head is positive and normalized to the volatility-emphasis value. |
| `aionx.models.DensityHNN.volatility_rescaling_algorithm` | OOB log squared residuals are regressed on log predicted volatility squared, then all volatility forecasts are rescaled. |

The callable consumes the standard supervised model contract `density_hnn(X,
y, **params)`. In Aionx, lags and trend terms are created inside
`DensityHNN.run(...)`; in `macroforecast`, lags, MARX/MAF features, PCA,
trends, and seasonal/time features should be built explicitly with
`macroforecast.feature_engineering` before calling the model or through the
forecasting runner. This keeps the model callable small and lets the same
feature construction be reused by other models.

Fit sequence:

1. Standardize `X` and `y` inside the fit window.
2. Fit a prior DNN ensemble on blocked bootstrap samples.
3. Compute prior-DNN OOB mean squared error; this becomes the Aionx
   `volatility_emphasis` unless the user supplies an override.
4. Fit a DensityHNN ensemble with shared core, conditional-mean head, and
   conditional-volatility head.
5. Compute HNN OOB mean and volatility forecasts.
6. Recalibrate volatility using the Aionx log residual-square regression.
7. Return forecasts on the original target scale.

Output:

| Method | Output |
| --- | --- |
| `predict(X)` | pandas Series of conditional mean forecasts through `ModelFit`. |
| `predict_variance(X)` | numpy array of conditional variance forecasts in target units squared. |
| `predict_volatility(X)` | numpy array of conditional standard-deviation forecasts in target units. |
| `predict_distribution(X)` | `(mean, variance)` arrays in target units. |
| `predict_quantiles(X, levels=None)` | dictionary from quantile level to normal-approximation quantile forecast. |

Diagnostics:

| Field | Meaning |
| --- | --- |
| `fit.diagnostics["density"]["volatility_emphasis"]` | Aionx volatility-emphasis value used by the HNN volatility head. |
| `fit.diagnostics["density"]["prior_oob_mse"]` | Prior-DNN OOB mean squared error used when `volatility_emphasis=None`. |
| `fit.diagnostics["density"]["oob_rescaling"]` | Intercept, slope, scaler, and OOB count for the log residual-square volatility recalibration. |
| `fit.estimator.oob_prediction_` | Fit-window OOB conditional mean, volatility, and variance table. |

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `common_layers` | `2` | fixed by preset | Shared common-core depth. |
| `mean_layers` | `2` | fixed by preset | Conditional-mean hemisphere depth. |
| `volatility_layers` | `2` | fixed by preset | Conditional-volatility hemisphere depth. |
| `prior_layers` | `3` | fixed by preset | Prior-DNN hidden depth. |
| `neurons` | `400` | yes | Hidden width. The paper/Aionx default is 400; smaller values are useful for smoke tests. |
| `dropout` | `0.2` | fixed by preset | Dropout rate. |
| `learning_rate` | `0.001` | yes | Adam learning rate. |
| `max_epochs` | `100` | fixed by preset | Training epoch cap. |
| `n_estimators` | `100` | yes | DensityHNN bootstrap ensemble size. |
| `prior_estimators` | `50` | yes | Prior-DNN bootstrap ensemble size used to estimate volatility emphasis. |
| `subsample` | `0.8` | fixed by preset | Blocked bootstrap sampling rate. |
| `block_size` | `8` | fixed by preset | Time-series block size. The paper uses blocked subsampling to preserve temporal dependence. |
| `volatility_emphasis` | `None` | fixed by preset | `None` estimates the value from prior-DNN OOB MSE. Passing a float overrides it. Values outside Aionx's `[0.01, 1.0]` range are mapped to `0.99`, following the source code. |
| `rescale_volatility` | `True` | fixed by preset | Apply the blocked-OOB volatility reality-check recalibration. |
| `patience` | `15` | fixed by preset | Early-stopping patience. |
| `device` | `"auto"` | fixed by preset | Torch device. |
| `quantile_levels` | `(0.05, 0.5, 0.95)` | fixed by preset | Default normal-approximation quantile levels. |
| `volatility_clip` | `0.05` | fixed by preset | Minimum volatility used in Gaussian negative log likelihood, matching Aionx's numerical-stability clip. |

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
    control_columns=None,
    include_constant=True,
    drop_control_columns=True,
    quadratic_factors=False,
)
```

Fits partial least squares regression. Unlike unsupervised PCA, PLS uses the
target while constructing latent components, so it belongs in `models` rather
than `preprocessing` or `feature_engineering`.

`n_components` is treated as a requested upper bound. At fit time, the model
resolves it to `min(requested, n_factor_predictors, n_observations)` so the
default is safe for small feature sets. Metadata records both
`requested_n_components` and `resolved_n_components`; `n_components` stores the
resolved value used by `sklearn.cross_decomposition.PLSRegression`.

Implementation map:

| Item | Value |
| --- | --- |
| Backend | `sklearn.cross_decomposition.PLSRegression` |
| Paper-code comparison | Hounyo-Li `PLS_emp002.m` and `PLS_tune.m` |
| Control handling | Optional `control_columns` block is fitted first, target residuals are passed to PLS, and the control forecast is added back. |
| Difference from MATLAB code | MATLAB `plsregress` exposes `stats.W`; macroforecast uses sklearn PLS latent scores and records factor loadings/metadata. The forecasting contract is the same control-residualized PLS factor regression. |
| PC2 support | Set `quadratic_factors=True` to add the `PLS_PC2.m` squared-factor forecast head. |

Hounyo-Li PLS baseline:

```text
alphawhat = y_insample * wt_insample' * inv(wt_insample * wt_insample')
y_resid   = y_insample - alphawhat * wt_insample
[~,~,~,~,~,~,~,stats] = plsregress(X_insample', y_resid', K)
B         = stats.W
Fhat      = B' * X_insample
alphahat  = y_resid * Fhat' * inv(Fhat * Fhat')
yhat      = (alphahat * B') * X_out + alphawhat * wt_out
```

`macroforecast.models.pls()` mirrors this as:

| MATLAB step | `macroforecast` implementation |
| --- | --- |
| `wt` | `control_columns` plus optional constant |
| residualize `ytplush` on `wt` | `_PLSCompositeRegressor.control_coef_` and target residual |
| `plsregress(..., K)` | `sklearn.cross_decomposition.PLSRegression` |
| `Fhat` PLS scores | `transform(...)` / `x_scores_` |
| `alphahat` on PLS scores | `factor_coefs_` |
| add `alphawhat * wt_out` | `predict()` control block addition |

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_components` | `3` | yes | Requested maximum number of latent PLS components. |
| `scale` | `True` | fixed by preset | Whether to standardize predictors before PLS. |
| `max_iter` | `500` | fixed by preset | NIPALS iteration cap. |
| `tol` | `1e-6` | fixed by preset | NIPALS convergence tolerance. |
| `control_columns` | `None` | fixed by preset | Optional X columns used as forecasting controls. |
| `include_constant` | `True` | fixed by preset | Whether to include a constant in the control block. |
| `drop_control_columns` | `True` | fixed by preset | Whether controls are excluded from the PLS block. |
| `quadratic_factors` | `False` | fixed by preset | Whether to add the Hounyo-Li PC2 squared-factor forecast head. |

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
    quadratic_factors=False,
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

Set `quadratic_factors=True` to reproduce the Hounyo-Li `scaledPCA_PC2.m`
forecast head:

$$
\hat y_* =
w_*\hat a_W + \hat\alpha_1 \hat f_*
+ \hat\alpha_2 \hat f_*^2.
$$

For multiple factors, the squared term is applied componentwise, matching the
MATLAB code's `alphahat2 * ((leftvector * scaleXs_outofsample).^2)` contract.

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
| `quadratic_factors` | `False` | fixed by preset | Whether to add the Hounyo-Li PC2 squared-factor forecast head. |

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
    quadratic_factors=False,
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

Set `quadratic_factors=True` for the Hounyo-Li `SPCA_PC2.m` variant. In that
case each extraction step also estimates

$$
\hat\alpha_{2,k}
=
\frac{r_y^{(k-1)\prime}(f_k^2)}{(f_k^2)'(f_k^2)}
$$

and updates the target residual as

$$
r_y^{(k)}
=
r_y^{(k-1)}
-\hat\alpha_{1,k}f_k
-\hat\alpha_{2,k}f_k^2.
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
| `quadratic_factors` | `False` | fixed by preset | Whether to add the Hounyo-Li PC2 squared-factor forecast head. |
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
    quadratic_factors=False,
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

Set `quadratic_factors=True` for the `SsPCA_PC2.m` variant. This keeps the same
predictive-slope scaling and supervised selection recursion, then adds the
componentwise squared-factor forecast head used in the paper's PC2 scripts.

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
| `quadratic_factors` | `False` | fixed by preset | Whether to add the Hounyo-Li PC2 squared-factor forecast head. |
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

### stlf

```python
macroforecast.models.stlf(y, *, period=None, sa_method="ets")
```

STL decomposition forecaster (R `forecast::stlf`). Seasonally adjusts the target
with STL, forecasts the seasonally-adjusted series (additive-trend exponential
smoothing, random-walk-drift fallback), and adds back the last seasonal cycle.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `period` | `None` | no | Seasonal period; inferred from the index frequency if omitted. |
| `sa_method` | `"ets"` | no | Forecaster for the seasonally-adjusted series. |

### naive

```python
macroforecast.models.naive(y)
```

Random-walk baseline (R `forecast::naive`). Carries the last observed target
value forward, so the h-step path is constant at `y_T`. Target-only.

### seasonal_naive

```python
macroforecast.models.seasonal_naive(y, *, period=None)
```

Seasonal-naive baseline (R `forecast::snaive`). Repeats the last full seasonal
cycle of length `period`, so step `k` returns the value from one season earlier.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `period` | `None` | no | Seasonal period `m`; defaults to 1 (plain naive). |

### random_walk_drift

```python
macroforecast.models.random_walk_drift(y)
```

Random-walk-with-drift baseline (R `forecast::rwf(drift=TRUE)`). Extrapolates the
last value by the average historical change: `y_T + h * (y_T - y_1) / (T - 1)`.

### var

```python
macroforecast.models.var(panel, *, target=None, n_lag=1, type="const", season=None)
```

Fits a VAR on a multivariate panel. `target` chooses the forecast output
column. If omitted, the first column is used. The callable now uses an internal
OLS implementation aligned with R `vars::VAR` and `predict.varest`: lagged
endogenous variables are stacked in lag order, deterministic terms are controlled
by R-style `type`, and `predict()` recursively rolls the VAR state forward for
point forecasts.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `target` | `None` | fixed by preset | Target column in the panel. |
| `n_lag` | `1` | yes | VAR lag order. |
| `type` | `"const"` | fixed by preset | R `vars::VAR` deterministic terms: `"const"`, `"trend"`, `"both"`, or `"none"`. Short aliases `"c"`, `"t"`, `"ct"`, and `"n"` are accepted. |
| `season` | `None` | fixed by preset | Optional centered seasonal dummies, matching `vars::VAR(season=...)`. |

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
    kappa0=2.0,
    kappa1=0.5,
    nu0=0.0,
    s0=0.0,
    iter=10000,
    burnin=5000,
    random_state=0,
)
```

Fits a Bayesian VAR posterior sampler with the Minnesota prior variance logic
used by R `FAVAR::BVAR` and `bvartools::minnesota_prior`. Saved posterior
coefficient and covariance draws are available in diagnostics; `predict()` uses
posterior-mean VAR coefficients for recursive point forecasts. BVAR forecasting
is not a macroforecast-only extension: CRAN `BVAR` exposes `predict.bvar`, while
this callable is macroforecast's R-aligned ModelFit surface for the same class
of BVAR forecast object.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `target` | `None` | fixed by preset | Target column in the panel. |
| `n_lag` | `1` | yes | VAR lag order. |
| `kappa0` | `2.0` | yes | Minnesota own-lag prior scale. |
| `kappa1` | `0.5` | yes | Minnesota lag-decay exponent. |
| `nu0` | `0.0` | fixed by preset | Inverse-Wishart degrees-of-freedom prior parameter. |
| `s0` | `0.0` | fixed by preset | Inverse-Wishart scale prior parameter. |
| `iter` | `10000` | fixed by preset | Total Gibbs iterations. |
| `burnin` | `5000` | fixed by preset | Burn-in iterations removed before summaries. |
| `random_state` | `0` | fixed by preset | Random seed for posterior draws. |

### bvar_normal_inverse_wishart

```python
macroforecast.models.bvar_normal_inverse_wishart(
    panel,
    *,
    target=None,
    n_lag=1,
    b0=0.0,
    vb0=0.0,
    nu0=0.0,
    s0=0.0,
    iter=10000,
    burnin=5000,
    random_state=0,
)
```

Fits the same FAVAR-style Bayesian VAR posterior sampler with direct controls
for coefficient prior mean/variance and inverse-Wishart covariance prior terms.
Saved diagnostics include coefficient posterior mean, standard deviation,
credible interval bounds, and posterior mean covariance.

### ets

```python
macroforecast.models.ets(
    y,
    *,
    error="add",
    trend=None,
    seasonal=None,
    seasonal_periods=None,
    damped_trend=False,
)
```

Fits a target-only statsmodels exponential-smoothing model through ETS-style
arguments. In `forecasting.run(...)`, this model ignores `X` and fits on the
stage target vector.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `error` | `"add"` | fixed by preset | Error component; currently additive by default. |
| `trend` | `None` | fixed by preset | Optional trend component such as `"add"`. |
| `seasonal` | `None` | fixed by preset | Optional seasonal component such as `"add"`. |
| `seasonal_periods` | `None` | fixed by preset | Seasonal period length. |
| `damped_trend` | `False` | fixed by preset | Whether to damp the trend component. |

Output is `ModelFit`; `predict(X_future)` uses only the number of requested
future rows and preserves the provided index.

### holt_winters

```python
macroforecast.models.holt_winters(
    y,
    *,
    trend="add",
    seasonal=None,
    seasonal_periods=None,
    damped_trend=False,
)
```

Fits a target-only Holt-Winters exponential-smoothing model. In the forecasting
runner it is a target-input model: feature matrices are used only to provide
the forecast index and horizon length.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `trend` | `"add"` | fixed by preset | Trend component. |
| `seasonal` | `None` | fixed by preset | Optional seasonal component. |
| `seasonal_periods` | `None` | fixed by preset | Seasonal period length. |
| `damped_trend` | `False` | fixed by preset | Whether to damp the trend component. |

Output is `ModelFit`; predictions are indexed like the supplied future frame.

### theta_method

```python
macroforecast.models.theta_method(
    y,
    *,
    period=None,
    deseasonalize=True,
    use_test=True,
)
```

Fits statsmodels' target-only Theta method wrapper. Use it as a benchmark
univariate model; it does not consume predictor columns.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `period` | `None` | fixed by preset | Seasonal period passed to statsmodels. |
| `deseasonalize` | `True` | fixed by preset | Whether statsmodels deseasonalizes before fitting. |
| `use_test` | `True` | fixed by preset | Whether statsmodels uses its internal seasonality test. |

Output is `ModelFit`; `predict(X_future)` returns a point forecast series.

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

R comparison: this is a backend-wrapper analogue of the mixed-frequency DFM
contract used by `dfms::DFM(X, quarterly.vars=...)` and archived
`nowcasting::nowcast(method="EM")`. Those R implementations require quarterly
series to be positioned after monthly series and impose the
Mariano-Murasawa `[1, 2, 3, 2, 1]` temporal aggregation restriction for
quarterly growth/flow variables. macroforecast delegates the Kalman/EM
likelihood to statsmodels rather than reimplementing the R/C++ filter code.

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
    metadata=None,
    lag_columns=None,
    lags=(0, 1, 2),
    factor_lags=(0,),
    target_frequency="quarterly",
    anchor_position="period_end",
    n_factors=1,
    factor_order=1,
    idiosyncratic_ar1=True,
    standardize=True,
    maxiter=500,
    tolerance=1e-6,
    alpha=0.0,
    fit_intercept=True,
    drop_missing=True,
)
```

Fits a composite mixed-frequency model:

1. Fit `dfm_mixed_mariano_murasawa(...)` on the native mixed-frequency panel.
2. Extract filtered DFM factors at the target anchor dates.
3. Add optional observed lag blocks from `mixed_frequency_lags(...)`.
4. Fit `unrestricted_midas(...)` as the forecast head.

This is a convenience composite, not a new state-space likelihood. The returned
fit's `predict()` method accepts a prepared feature matrix with the same
columns as `fit.estimator.design_`. The lower-level
`fit.estimator.predict_from_panel(...)` method rebuilds the composite design
from a native mixed-frequency panel. `forecasting.run(...)` uses that method:
it fits the MIDAS head on the training panel, masks the test target values, then
refits the DFM on the available native panel so current monthly information can
enter the test-origin factor design without using the held-out target.

R comparison: this is the explicit callable version of a two-stage workflow,
not a single R estimator. The first stage follows the DFM contract above. The
forecast head is aligned with `midasr::midas_u` when `alpha=0`; `alpha>0` is a
macroforecast ridge extension.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `target` | required | fixed by preset | Forecasted target column. |
| `metadata` | `None` | fixed by preset | Metadata with native frequencies; normally supplied by `DataBundle`. |
| `lag_columns` | `None` | fixed by preset | Observed columns added as unrestricted MIDAS lags. |
| `lags` | `(0, 1, 2)` | yes | Native-frequency lags for observed columns. |
| `factor_lags` | `(0,)` | yes | Monthly lags of filtered DFM factors. |
| `target_frequency` | `"quarterly"` | fixed by preset | Frequency used to position target anchor dates. |
| `anchor_position` | `"period_end"` | fixed by preset | Anchor positioning; useful for FRED-QD quarter-start dates. |
| `n_factors` | `1` | yes | Number of DynamicFactorMQ factors. |
| `factor_order` | `1` | yes | VAR order for factor dynamics. |
| `idiosyncratic_ar1` | `True` | fixed by preset | Model DFM idiosyncratic components as AR(1). |
| `standardize` | `True` | fixed by preset | Let DynamicFactorMQ standardize observed variables. |
| `maxiter` | `500` | fixed by preset | DFM EM iteration cap. |
| `tolerance` | `1e-6` | fixed by preset | DFM EM convergence tolerance. |
| `alpha` | `0.0` | yes | Ridge penalty on the unrestricted MIDAS head. |
| `fit_intercept` | `True` | fixed by preset | Whether the unrestricted MIDAS head includes an intercept. |
| `drop_missing` | `True` | fixed by preset | Drop incomplete composite design rows before fitting the head. |

### midas_almon

```python
macroforecast.models.midas_almon(
    X,
    y,
    *,
    polynomial_order=2,
    theta=None,
    alpha=0.0,
    fit_intercept=True,
)
```

Fits a MIDAS regression where each lag group is compressed with normalized
exponential Almon weights before a linear or ridge head is fit.

R comparison: `midasr::midas_r(..., nealmon)` jointly estimates the aggregate
scale and Almon shape by nonlinear least squares. macroforecast keeps the shape
fixed as a hyperparameter and estimates only the aggregate regression
coefficient in a linear/ridge head. The weight shape matches the scale-free part
of `midasr::nealmon`.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `polynomial_order` | `2` | yes | Degree of the Almon polynomial. |
| `theta` | `None` | fixed by preset | Shape coefficients for the scale-free part of `midasr::nealmon`. `None` gives equal weights. |
| `alpha` | `0.0` | yes | Ridge penalty on the regression head. |
| `fit_intercept` | `True` | fixed by preset | Whether the regression head includes an intercept. |

Weight formula:

```text
h_j = j, j = 1, ..., d
w_j = exp(theta_1 h_j + ... + theta_p h_j^p) / sum_j exp(...)
```

If `theta` is supplied, it must contain `polynomial_order` values. The aggregate
coefficient scale is estimated by the regression head, corresponding to the
first scale parameter in `midasr::nealmon`.

### midas_beta

```python
macroforecast.models.midas_beta(
    X,
    y,
    *,
    beta_params=(1.0, 1.0),
    alpha=0.0,
    fit_intercept=True,
)
```

Fits a MIDAS regression where each lag group is compressed with normalized beta
weights before a linear or ridge head is fit.

R comparison: this uses the scale-free form of `midasr::nbetaMT` with
`p=(1, a, b, 0)`: endpoints are shifted by machine epsilon, the beta density is
normalized, and the aggregate scale is estimated by the regression head.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `beta_params` | `(1.0, 1.0)` | yes | Positive beta-shape parameters `(a, b)`. |
| `alpha` | `0.0` | yes | Ridge penalty on the regression head. |
| `fit_intercept` | `True` | fixed by preset | Whether the regression head includes an intercept. |

Weight formula:

```text
z_j = (j - 1) / (d - 1), with endpoint epsilon adjustment
w_j = z_j^(a-1) (1-z_j)^(b-1) / sum_j z_j^(a-1) (1-z_j)^(b-1)
```

Both beta parameters must be strictly positive.

### midas_step

```python
macroforecast.models.midas_step(
    X,
    y,
    *,
    n_steps=3,
    step_bounds=None,
    step_weights=None,
    alpha=0.0,
    fit_intercept=True,
)
```

Fits a MIDAS regression where lags are grouped into piecewise-constant step
blocks. If `step_bounds` and `step_weights` are omitted, the lag range is split
into `n_steps` blocks with equal raw step heights, then normalized to a
scale-free weight vector.

R comparison: `midasr::polystep(p, d, m, a)` repeats raw step coefficients
between interior cut points. macroforecast exposes the same idea through
`step_bounds=a` and `step_weights=p`, then normalizes the resulting shape
because the aggregate scale is estimated by the regression head.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_steps` | `3` | yes | Number of lag buckets when `step_bounds` is omitted. |
| `step_bounds` | `None` | fixed by preset | Optional interior cut points, matching `midasr::polystep(..., a=...)`. |
| `step_weights` | `None` | fixed by preset | Optional raw step heights, one per bucket. |
| `alpha` | `0.0` | yes | Ridge penalty on the regression head. |
| `fit_intercept` | `True` | fixed by preset | Whether the regression head includes an intercept. |

`n_steps` must be positive. If supplied, `step_bounds` must be strictly
increasing and smaller than the number of lag columns; `step_weights` must
contain one value per resulting bucket.

### restricted_midas

```python
macroforecast.models.restricted_midas(
    X,
    y,
    *,
    weighting="almon",
    polynomial_order=2,
    start_params=None,
    n_steps=3,
    step_bounds=None,
    fit_intercept=True,
    maxiter=1000,
    tolerance=1e-8,
)
```

Fits a nonlinear restricted MIDAS regression over an explicit lag matrix. This
is the direct callable counterpart to `midasr::midas_r` when the formula has
already been expanded into columns such as `PAYEMS_lag0`, `PAYEMS_lag1`, and
`PAYEMS_lag2`.

R comparison: `midasr::midas_r` maps each low-dimensional restriction parameter
vector into full lag coefficients and minimizes the nonlinear least-squares
objective. `restricted_midas()` uses the same objective and the same `nealmon`,
`nbetaMT`, or `polystep` coefficient maps. It uses SciPy `least_squares`
instead of R's default `optim(method="BFGS")`, so optimizer traces are not
bit-identical, but the restricted regression equation is the same. Formula
parsing, AR* common-factor terms, HAC covariance, model tables, and S3 forecast
utilities are not reproduced here.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `weighting` | `"almon"` | yes | Restriction map: `"almon"`/`"nealmon"`, `"beta"`/`"nbetaMT"`, or `"step"`/`"polystep"`. |
| `polynomial_order` | `2` | yes | Number of Almon shape terms after the aggregate scale parameter. |
| `start_params` | `None` | fixed by preset | Starting values. Pass one sequence for all lag groups, or a mapping from group name to sequence. |
| `n_steps` | `3` | yes | Number of step buckets when `weighting="step"` and `step_bounds` is omitted. |
| `step_bounds` | `None` | fixed by preset | Interior cut points for `polystep`-style step coefficients. |
| `fit_intercept` | `True` | fixed by preset | Whether to estimate an intercept outside the restricted lag coefficients. |
| `maxiter` | `1000` | fixed by preset | Maximum SciPy least-squares function evaluations. |
| `tolerance` | `1e-8` | fixed by preset | Shared `xtol`, `ftol`, and `gtol` stopping tolerance. |

Outputs include fitted values, residuals, unrestricted effective lag
coefficients, the optimized restricted parameter vector, convergence metadata,
and the lag-group metadata used to expand coefficients.

### unrestricted_midas

```python
macroforecast.models.unrestricted_midas(
    X,
    y,
    *,
    alpha=0.0,
    fit_intercept=True,
)
```

Fits an unrestricted MIDAS regression. Unlike the weighted variants, it does
not collapse lag groups; each supplied lag column receives its own coefficient.

R comparison: this matches `midasr::midas_u` when `alpha=0`, because every lag
coefficient is free and the regression is ordinary least squares. `alpha>0` is
a macroforecast ridge extension for high-dimensional lag matrices.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `alpha` | `0.0` | yes | Ridge penalty. `0.0` gives an ordinary linear head. |
| `fit_intercept` | `True` | fixed by preset | Whether the regression head includes an intercept. |

### MIDAS Input Contract

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

The weighted MIDAS callables `midas_almon()`, `midas_beta()`, and
`midas_step()` treat shape parameters as fixed or selected hyperparameters.
Use `restricted_midas()` when the shape and scale parameters should be
estimated jointly by nonlinear least squares, matching the `midasr::midas_r`
estimation target.

For `unrestricted_midas()`, build its input matrix with
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
    n_factors=2,
    n_lag=2,
    fctmethod="BBE",
    slowcode=None,
    factorprior=None,
    varprior=None,
    nburn=5000,
    nrep=15000,
    standardize=True,
    random_state=0,
)
```

Fits a Bayesian FAVAR aligned with CRAN `FAVAR::FAVAR`: optional R-style
standardization, `ExtrPC()` factor extraction, BBE `facrot()` or BGM factor
identification, conjugate loading-equation draws, and the internal
`FAVAR::BVAR` posterior sampler for the `[factors, y]` VAR block.

Important boundary: BVAR forecasting is standard, and CRAN `BVAR` has
`predict.bvar`. The macroforecast-specific extension here is narrower:
CRAN `FAVAR` exposes summaries, coefficients, and impulse responses for `favar`
objects, but not `predict.favar`. Therefore `macroforecast.models.favar(...).predict(...)`
is a ModelFit forecast wrapper over the fitted FAVAR posterior VAR state using
posterior-mean coefficients.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_factors` | `2` | yes | Number of latent factors. |
| `n_lag` | `2` | yes | VAR lag order on the target plus factors. |
| `fctmethod` | `"BBE"` | fixed by preset | Factor identification method: `"BBE"` or `"BGM"`. |
| `slowcode` | `None` | fixed by preset | Boolean slow-variable mask required by BBE. |
| `factorprior` | `None` | fixed by preset | Factor loading prior controls. |
| `varprior` | `None` | fixed by preset | BVAR prior controls for the factor VAR block. |
| `nburn` | `5000` | fixed by preset | Burn-in iterations for posterior draws. |
| `nrep` | `15000` | fixed by preset | Saved posterior draw count. |
| `standardize` | `True` | fixed by preset | Use R `scale()` semantics for X and y before factor extraction. |
| `random_state` | `0` | fixed by preset | Random seed for posterior draws. |

| Preset | `n_factors` | `n_lag` |
| --- | --- | --- |
| `small` | `(1, 2, 3)` | `(1, 2, 4)` |
| `standard` | `(1, 2, 3, 5, 8)` | `(1, 2, 4, 6, 12)` |
| `wide` | `(1, 2, 3, 5, 8, 10, 12)` | `(1, 2, 3, 4, 6, 9, 12, 18, 24)` |

## Tree And Machine-Learning Models

### Tree Implementation Map

Tree callables use backend wrappers, hybrid wrappers, or package-native code.
Fit-time model ensembles such as bagging, subagging, stacking, Super Learner,
and Booging live in [Model Ensemble](model_ensemble.md).

| Model | Implementation class | Runtime backend |
| --- | --- | --- |
| `decision_tree` | backend wrapper | `sklearn.tree.DecisionTreeRegressor` |
| `random_forest` | backend wrapper | `sklearn.ensemble.RandomForestRegressor` |
| `extra_trees` | backend wrapper | `sklearn.ensemble.ExtraTreesRegressor` |
| `gradient_boosting` | backend wrapper | `sklearn.ensemble.GradientBoostingRegressor` |
| `xgboost` | optional backend wrapper | `xgboost.XGBRegressor` |
| `lightgbm` | optional backend wrapper | `lightgbm.LGBMRegressor` |
| `lgb_plus` | package-native hybrid | LGB+ competition algorithm aligned to `philgoucou/lgbplus`, using `lightgbm.train` for residual tree candidates |
| `lgba_plus` | package-native hybrid | LGB^A+ alternating algorithm aligned to `philgoucou/lgbplus`, using `lightgbm.train` for residual tree blocks |
| `catboost` | optional backend wrapper | `catboost.CatBoostRegressor` |
| `quantile_regression_forest` | hybrid | sklearn forest plus macroforecast leaf-target quantile store |
| `macro_random_forest` | hybrid adapter | vendored `macroforecast.models._mrf_reference.MacroRandomForest` |

`backend wrapper` means the statistical estimator is delegated to the named
package and macroforecast standardizes callable input, metadata, diagnostics,
and persistence. `hybrid` means macroforecast owns part of the algorithmic
contract, such as resampling, leaf-distribution storage, feature augmentation,
or pandas-to-reference-package adaptation. `package-native` means the estimator
logic itself is implemented inside macroforecast.

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

R parity is intentionally not claimed for the sklearn tree wrappers. The named
backend owns the estimator; macroforecast owns the pandas `X, y` contract,
metadata, diagnostics, and search-space registration.

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

The fitted wrapper exposes sklearn feature importances through
`fit.diagnostics["feature_importance"]`.

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

Default model-selection method: `random`.

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

Default model-selection method: `random`.

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

This is one boosting estimator. Fit-time boosting ensembles such as Booging live
in [Model Ensemble](model_ensemble.md).

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

Default model-selection method: `random`.

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

Default model-selection method: `random`.

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
Default model-selection method: `random`.

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

Default model-selection method: `random`.

### lgb_plus

#### Paper Citation And Source

Primary paper:

> Goulet Coulombe, Philippe. 2026. "LGB+: A Macroeconomic Forecasting Road
> Test." Draft dated March 18, 2026. SSRN abstract 6439178.
> DOI: [`10.2139/ssrn.6439178`](https://doi.org/10.2139/ssrn.6439178).
> Paper page: <https://ssrn.com/abstract=6439178>.

Reference implementation:

| Source | Role in `macroforecast` |
| --- | --- |
| [`philgoucou/lgbplus`](https://github.com/philgoucou/lgbplus) | Original R/Python implementation repository. |
| `R/lgb_plus.R` | Competition algorithm, `linear_candidate_fraction`, component prediction helpers, and ensemble helper. |
| `python/lgb_plus.py` | Competition estimator class with in-class ensemble storage and step histories. |
| `R/lgb_plus_A.R` | Alternating algorithm and `lgb_plus_A_ensemble` helper. |
| `python/lgb_plus_A.py` | Alternating estimator class. |

#### Paper Motivation

The paper targets a macro forecasting problem that appears repeatedly in small
and medium macro samples. Tree boosting is strong for nonlinearities and
interactions, but a large share of macro predictive content can be simple:
autoregressive persistence, slowly moving accounting-like relationships, or
near-mechanical links such as claims before unemployment and permits before
housing starts. A standard tree booster can approximate those linear slopes,
but it spends splits and boosting capacity to do work that a one-variable
linear update could do cheaply.

LGB+ expands the boosting basis from "trees only" to "trees plus greedy linear
updates." The forecast remains additive:

```text
yhat(x) = intercept + tree_component(x) + linear_component(x)
```

That additive form is not a cosmetic detail. It lets the user inspect the
forecast through two channels:

| Channel | Intended role | Caveat |
| --- | --- | --- |
| Linear | Persistence, autoregressive slopes, near-accounting links, and other simple one-variable residual corrections. | The split is operational, not metaphysical; a tree can still learn linear-looking structure. |
| Tree | Nonlinear states, interactions, thresholds, and regime-dependent effects. | Tree gains can include simple structures if the linear candidate loses the competition. |

The paper emphasizes that the linear/nonlinear split should be read as an
algorithmic decomposition generated by the boosting path, not as proof that the
data-generating process is literally separated into pure linear and pure
nonlinear blocks.

#### Paper Empirical Design

The empirical road test uses transformed quarterly U.S. macro data from
FRED-QD. The review file summarized the design as six targets: headline CPI
inflation, GDP growth, unemployment, housing starts growth, industrial
production, and the term spread. Predictors include FRED-QD transformations,
four lags, MARX moving-average features, and principal components from the
transformed panel. The out-of-sample design is expanding-window forecasting,
with a pre-COVID evaluation period and a post-COVID stress period.

This matters for using `macroforecast`:

| Paper object | `macroforecast` stage |
| --- | --- |
| FRED-QD transformed panel | `macroforecast.data.load_fred_qd(...)` then `macroforecast.preprocessing.reprocess(...)`. |
| Four lags | `macroforecast.feature_engineering.lag(...)` or runner `feature_spec(...)`. |
| MARX moving-average features | `macroforecast.feature_engineering.moving_average_ladder(...)` or `marx_step(...)`. |
| Principal components | `macroforecast.feature_engineering.pca_features(...)` or preprocessing/factor features when fit-window-safe execution is needed. |
| Expanding OOS design | `macroforecast.window.estimation_expanding(...)` plus `test_origins(...)`. |
| Re-estimation schedule | `macroforecast.window.estimation_expanding(..., retrain_every=...)`. |
| LGB+ model | `macroforecast.models.lgb_plus(...)`. |
| LGB^A+ model | `macroforecast.models.lgba_plus(...)`. |
| Linear/tree forecast decomposition | `fit.estimator.predict_components(...)` and diagnostics in `ModelFit`. |

#### Method In The Paper

The paper has two closely related estimators.

`LGB+` is the competition version. At each boosting step:

| Step | Operation |
| --- | --- |
| 1 | Start from the current fitted value. |
| 2 | Compute residuals on the training sample. |
| 3 | Draw a row subsample. |
| 4 | Fit one small LightGBM residual tree on the subsample. |
| 5 | Fit one greedy univariate linear residual update on the same subsample. |
| 6 | Evaluate both candidate updates using `oob`, fixed `validation`, or `training` loss. |
| 7 | Accept only the lower-loss candidate. |
| 8 | Record which channel won, the selected linear feature when relevant, and the candidate losses. |

`LGB^A+` is the alternating version. It does not run a per-step competition.
Instead, each cycle applies a block of residual trees and then a greedy
univariate linear correction. This is computationally simpler and can be more
stable when the OOB judge is noisy in macro-sized samples.

#### Main Paper Findings To Keep In Mind

The paper's simulations and empirical road test support the following working
interpretation:

| Finding | Practical implication |
| --- | --- |
| In mostly linear designs, the linear channel can absorb much of the signal and avoid forcing trees to approximate simple slopes. | Include autoregressive and near-accounting predictors explicitly; then inspect the linear channel. |
| In nonlinear designs, the tree channel remains active and the hybrid does not have to behave like a linear model. | LGB+ is a flexible hybrid, not a linear model with tree residuals fixed in advance. |
| The linear channel is often useful for short-horizon unemployment and industrial production before COVID. | Channel diagnostics can reveal whether gains come from persistence-like relations or nonlinear state recognition. |
| In post-COVID stress periods, the linear channel can become harmful for some real-activity targets. | Report channel-specific diagnostics; do not rely only on total RMSE. |
| Forecasts can be decomposed natively into tree and linear pieces. | Use `predict_components()` and store `ModelFit.diagnostics` when writing replication outputs. |

#### What `macroforecast` Implements

`macroforecast.models.lgb_plus` implements the competition estimator as a
package-native hybrid model. LightGBM supplies the residual tree candidate, but
the step loop, linear candidate, OOB/validation/training selection, ensemble
aggregation, channel accounting, and pandas metadata are implemented inside
`macroforecast`.

The implementation is deliberately not just a thin `LGBMRegressor` wrapper:

| Feature | Implemented? | Notes |
| --- | --- | --- |
| Tree candidate per step | yes | Uses `lightgbm.train(..., num_boost_round=1)`. |
| Greedy univariate linear candidate | yes | Uses the same no-intercept residual slope as the competition reference code. |
| `linear_candidate_fraction` | yes | Kept from the R implementation. |
| `selection_method="oob"` | yes | Default; requires `subsample < 1`. |
| `selection_method="validation"` | yes | Uses a fixed random validation split inside the current fit window. |
| `selection_method="training"` | yes | Available for reference parity, but not recommended for macro evaluation. |
| Ensemble members | yes | Controlled by `n_ensemble`; predictions aggregate by mean or median. |
| Linear component prediction | yes | `predict_components(...).prediction_linear`. |
| Tree component prediction | yes | `predict_components(...).prediction_tree`. |
| Channel diagnostics | yes | `channel_summary`, `channel_importance`, and `training_history`. |
| AXIL historical weights | no | This belongs in interpretation/forecast analysis later, not inside the estimator. |
| Full paper table replication | no | The callable model is implemented; full empirical replication should be a separate replication package. |

```python
macroforecast.models.lgb_plus(
    X,
    y,
    *,
    n_ensemble=10,
    n_steps=200,
    learning_rate=0.05,
    subsample=0.7,
    num_leaves=5,
    min_data_in_leaf=20,
    lambda_l2=0.1,
    linear_candidate_fraction=0.5,
    selection_method="oob",
    val_fraction=0.2,
    early_stop_patience=50,
    aggregation="mean",
    random_state=0,
    verbose=False,
    **kwargs,
)
```

Fits LGB+ competition boosting from Goulet Coulombe's
[`philgoucou/lgbplus`](https://github.com/philgoucou/lgbplus) reference code.
This is not ordinary `lightgbm` with extra linear features. At every boosting
step the estimator builds two residual updates:

| Candidate | Reference-code operation | Accepted when |
| --- | --- | --- |
| Tree | Fit one small `lightgbm.train(...)` residual tree with manual shrinkage. | Candidate loss is no larger than the linear candidate. |
| Linear | Sample `linear_candidate_fraction` of features, choose the largest absolute residual correlation, and fit one no-intercept residual slope. | Candidate loss is lower than the tree candidate. |

The R reference file `R/lgb_plus.R` includes `linear_candidate_fraction`; the
Python reference file `python/lgb_plus.py` embeds the ensemble in the estimator
but does not expose that candidate-fraction argument. `macroforecast` combines
those two reference surfaces: `n_ensemble` controls independent runs and
`linear_candidate_fraction` controls greedy linear candidate subsampling.

Input is the standard supervised model contract:

| Input | Required shape | Meaning |
| --- | --- | --- |
| `X` | `pandas.DataFrame`, `FeatureSet`, or array-like with shape `(n_obs, n_features)` | Predictor matrix. DataFrame column names are preserved in diagnostics. |
| `y` | `pandas.Series` or array-like with length `n_obs` | Forecast target for the current fit window. |

Output is a `ModelFit` with `model="lgb_plus"`. Use
`fit.predict(X_new)` for the total prediction. The estimator also exposes:

| Method or diagnostic | Output | Meaning |
| --- | --- | --- |
| `fit.estimator.predict_components(X_new)` | DataFrame | `prediction_total`, `prediction_init`, `prediction_tree`, `prediction_linear`. |
| `fit.estimator.predict_individual(X_new)` | ndarray | One total prediction path per ensemble member. |
| `fit.estimator.channel_importance()` | DataFrame | Tree gain, linear selection count, and absolute linear update by feature. |
| `fit.diagnostics["channel_summary"]` | dict | Total tree and linear steps plus per-member counts. |
| `fit.diagnostics["training_history"]` | dict | Per-step candidate losses and selected channel metadata. |

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_ensemble` | `10` | yes | Independent boosting runs. Predictions are aggregated across runs. |
| `n_steps` | `200` | yes | Maximum tree/linear competition steps per run. |
| `learning_rate` | `0.05` | yes | Shared shrinkage for accepted tree or linear updates. |
| `subsample` | `0.7` | yes | Row subsample share per step. `selection_method="oob"` requires `< 1`. |
| `num_leaves` | `5` | yes | Maximum leaves for the one-step LightGBM tree candidate. |
| `min_data_in_leaf` | `20` | yes | Minimum rows per LightGBM leaf. |
| `lambda_l2` | `0.1` | fixed by preset | L2 penalty for LightGBM tree candidates. |
| `linear_candidate_fraction` | `0.5` | yes | Fraction of predictors sampled before greedy linear selection. |
| `selection_method` | `"oob"` | fixed by preset | Candidate judge: `"oob"`, `"validation"`, or `"training"`. |
| `val_fraction` | `0.2` | fixed by preset | Fixed validation share when `selection_method="validation"`. |
| `early_stop_patience` | `50` | fixed by preset | Stop after this many non-improving selected losses; `None` disables. |
| `aggregation` | `"mean"` | fixed by preset | Ensemble aggregation: `"mean"` or `"median"`. |
| `random_state` | `0` | fixed by preset | Base random seed. Each ensemble member increments it. |
| `**kwargs` | none | fixed by caller | Additional `lightgbm.train` parameters for residual tree candidates. |

| Preset | Main search dimensions |
| --- | --- |
| `small` | `n_ensemble`, `n_steps`, `learning_rate`, `subsample`, `num_leaves`, `min_data_in_leaf`, `linear_candidate_fraction` over narrow ranges. |
| `standard` | Same dimensions with 5-10 members, 100-400 steps, and candidate fractions `(0.33, 0.5, 1.0)`. |
| `wide` | Same dimensions with up to 20 members, 600 steps, lower learning rates, and broader subsampling. |

Default model-selection method: `random`.

### lgba_plus

#### Paper Link

`lgba_plus` is the `macroforecast` callable for the paper's alternating
variant, LGB^A+. It uses the same paper and source references as `lgb_plus`:
Goulet Coulombe (2026), "LGB+: A Macroeconomic Forecasting Road Test,"
SSRN 6439178, DOI
[`10.2139/ssrn.6439178`](https://doi.org/10.2139/ssrn.6439178), and the
[`philgoucou/lgbplus`](https://github.com/philgoucou/lgbplus) source
repository.

#### What Changes Relative To `lgb_plus`

The alternating version is easier to read and usually cheaper to fit:

| Dimension | `lgb_plus` | `lgba_plus` |
| --- | --- | --- |
| Update schedule | Tree and linear candidates compete; one winner advances. | Every cycle applies both a tree block and one linear correction. |
| Main count parameter | `n_steps` per ensemble member. | `n_cycles` and `trees_per_cycle`. |
| Tree learning rate | Shared `learning_rate`. | `lr_tree`. |
| Linear learning rate | Shared `learning_rate`. | `lr_linear`. |
| Linear update | No-intercept residual slope in the competition reference code. | Intercept plus slope, matching the alternating reference code. |
| Ensemble control | `n_ensemble`. | `n_runs`. |
| Main diagnostic | Winner path and candidate losses. | Cycle path and selected linear feature after each tree block. |

Use `lgba_plus` when the goal is a stable hybrid path rather than estimating the
best tree/linear mix at every individual step. Use `lgb_plus` when the winner
sequence itself is part of the analysis.

#### What `macroforecast` Implements

| Feature | Implemented? | Notes |
| --- | --- | --- |
| Residual tree blocks | yes | Uses `lightgbm.train(..., num_boost_round=trees_per_cycle)`. |
| Greedy linear correction after every tree block | yes | Selects the largest absolute residual correlation. |
| Separate tree and linear learning rates | yes | `lr_tree`, `lr_linear`. |
| `n_runs` alternating ensemble | yes | Folds the R `lgb_plus_A_ensemble` helper into the estimator API. |
| Component prediction | yes | Same `predict_components()` output columns as `lgb_plus`. |
| Channel importance | yes | Tree gain, linear selection count, and absolute linear update. |
| Full AXIL dual interpretation | no | Planned for interpretation/forecast analysis rather than model fitting. |

```python
macroforecast.models.lgba_plus(
    X,
    y,
    *,
    n_runs=1,
    n_cycles=25,
    trees_per_cycle=10,
    lr_tree=0.02,
    lr_linear=0.1,
    num_leaves=15,
    min_data_in_leaf=20,
    subsample=1.0,
    random_state=0,
    verbose=False,
    **kwargs,
)
```

Fits LGB^A+, the alternating variant from
[`philgoucou/lgbplus`](https://github.com/philgoucou/lgbplus). Each cycle first
fits a block of LightGBM residual trees, then fits one greedy univariate linear
residual update with an intercept. Unlike `lgb_plus`, there is no per-step
winner selection: both channels are updated every cycle.

The R reference file `R/lgb_plus_A.R` also provides an ensemble helper
`lgb_plus_A_ensemble`. `macroforecast` folds that helper into this estimator via
`n_runs`, so the same callable can represent both a single alternating model and
an averaged alternating ensemble.

Input and output follow the same supervised contract as `lgb_plus`.
`fit.estimator.predict_components(X_new)` returns the total, intercept, tree,
and linear channels; `fit.estimator.channel_importance()` reports tree gain and
linear update frequency by feature.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_runs` | `1` | yes | Independent alternating runs. |
| `n_cycles` | `25` | yes | Tree-block plus linear-update cycles per run. |
| `trees_per_cycle` | `10` | yes | Residual LightGBM trees per cycle. |
| `lr_tree` | `0.02` | yes | Shrinkage for tree-block predictions. |
| `lr_linear` | `0.1` | yes | Shrinkage for linear residual updates. |
| `num_leaves` | `15` | yes | Maximum leaves for each residual tree. |
| `min_data_in_leaf` | `20` | yes | Minimum rows per LightGBM leaf. |
| `subsample` | `1.0` | yes | LightGBM bagging fraction for tree blocks. |
| `random_state` | `0` | fixed by preset | Base random seed. Each run increments it. |
| `**kwargs` | none | fixed by caller | Additional `lightgbm.train` parameters. |

The linear slope is computed by the centered OLS identity
`sum((x - mean(x)) * (r - mean(r))) / sum((x - mean(x))^2)`. This is
statistically equivalent to the R code's `cov(x, residual) / var(x)` and avoids
the denominator mismatch in the reference Python expression that combines
`np.cov(...)` with `x.var()`.

| Preset | Main search dimensions |
| --- | --- |
| `small` | Short alternating runs for smoke or narrow-window use. |
| `standard` | 1 or 5 runs, 10 or 25 cycles, and tree/linear learning-rate grids. |
| `wide` | Up to 10 runs, 50 cycles, broader tree-block size and learning-rate ranges. |

Default model-selection method: `random`.

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

Preset spaces match `gradient_boosting`. Default model-selection method:
`random`.

## Macro-Specific Tree Models

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

The fitted wrapper also exposes sklearn forest feature importances through
`fit.diagnostics["feature_importance"]`.

Quantiles use tree-equal leaf weighting: within each tree, all training
observations that share the test row's terminal leaf receive equal weight, and
each tree contributes the same total weight. This avoids letting large leaves
dominate the empirical quantile solely because they contain more observations.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `200` | yes | Number of trees. |
| `max_depth` | `None` | yes | Maximum depth per tree. |
| `min_samples_leaf` | `1` | yes | Minimum samples per terminal leaf. |
| `random_state` | `0` | fixed by preset | Forest random seed. |
| `quantile_levels` | `(0.05, 0.5, 0.95)` | fixed by preset | Default levels returned by `predict_quantiles()`. |

Preset spaces match `random_forest`. Default model-selection method: `random`.

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

If the reference backend returns multiple prediction columns for the requested
test rows, the adapter averages them row by row. If the backend returns no
recognized prediction field or fewer than the requested number of predictions,
the adapter raises a runtime error instead of silently returning a misaligned
forecast vector.

By default all columns in `X` are used both as the time-varying linear equation
variables (`x_columns`) and the forest state variables (`S_columns`). Pass
`x_columns` and `S_columns` when those sets should differ.

The reference backend distinguishes two predictor sets:

| Argument | Role |
| --- | --- |
| `x_columns` | Columns in the local linear forecasting equation. These are the variables whose coefficients are allowed to vary over time. |
| `S_columns` | State variables used by the forest to split the sample and estimate those local coefficients. |

Use either column names or reference-style positions for each role. Passing both
`x_columns` and `x_pos`, or both `S_columns` and `S_pos`, raises an error rather
than silently prioritizing one selector.

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
also disable model selection for this model:

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
    model_selection={"macro_random_forest": None},
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
| `y_pos` | `0` | fixed by preset | Fixed target position for the separated `X/y` adapter; must remain `0`. |
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

These models are for volatility/variance forecasting, not ordinary conditional
mean macro forecasting. They accept a univariate return-like series `y` and
return both a point-mean prediction interface and a variance forecast interface.

| Function | Implementation | R/source comparison | Boundary |
| --- | --- | --- | --- |
| `garch11` | Optional Python `arch.arch_model` backend. | Same surface as `rugarch::ugarchspec(variance.model=list(model="sGARCH"))` plus `ugarchfit()`, but the likelihood is delegated to `arch`, not reimplemented. | Backend controls solver details, distribution aliases, convergence behavior, and forecast internals. |
| `egarch` | Optional Python `arch.arch_model` backend. | Same surface as `rugarch::ugarchspec(variance.model=list(model="eGARCH"))` plus `ugarchfit()`, but the likelihood is delegated to `arch`, not reimplemented. | Backend controls solver details, distribution aliases, convergence behavior, and forecast internals. |
| `realized_garch` | Internal compact Gaussian log-linear MLE. | Aligned with the p=q=1 Hansen-Huang-Shek / `rugarch` `realGARCH` state and measurement equations. | Not a full `rugarch` clone: no ARMA/ARFIMA mean, variance regressors, non-Gaussian distributions, fixed-parameter SE machinery, simulation/path/roll helpers, or xts-specific checks. |

### Common Output

```python
fit = macroforecast.models.garch11(y)
variance = fit.predict_variance(horizon=12)
sigma = fit.conditional_volatility
metadata = fit.to_metadata()
```

| Output | Type | Meaning |
| --- | --- | --- |
| `fit` | `VolatilityFit` | Fitted wrapper with `predict()`, `predict_variance()`, `summary()`, `to_dict()`, and `to_metadata()`. |
| `fit.predict(X)` | `pandas.Series` | Conditional mean prediction. For these models this is usually a constant mean from the volatility backend. |
| `fit.predict_variance(horizon)` | `pandas.Series` | Variance forecast indexed from `0` to `horizon - 1`. `horizon` must be positive. |
| `fit.conditional_volatility` | `pandas.Series` or `None` | In-sample conditional volatility path if available. |
| `fit.diagnostics["params"]` | `dict` | Fitted parameters. Names depend on the backend/model. |
| `fit.diagnostics["conditional_volatility"]` | `pandas.Series` | Same path as `fit.conditional_volatility`, stored for metadata export. |

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
The default is GARCH(1,1):

$$
r_t = \mu + \epsilon_t,\qquad \epsilon_t = \sigma_t z_t
$$

$$
\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2.
$$

For higher `p`/`q`, the lag orders are passed directly to
`arch.arch_model(vol="GARCH", p=p, q=q)`.

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

Implementation notes:

| Item | Value |
| --- | --- |
| Backend | `arch.arch_model` |
| Required extra | `macroforecast[arch]` |
| R comparison | `rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(p, q)))` |
| Internal likelihood? | No. macroforecast validates orders, passes inputs to `arch`, and records metadata/diagnostics. |
| Minimum data | 30 non-missing observations. |

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
The backend receives:

```python
arch.arch_model(y, vol="EGARCH", p=p, o=o, q=q, mean=mean_model, dist=dist)
```

For EGARCH(1,1), the log-variance structure is backend-defined by `arch`;
conceptually it is the exponential GARCH family where log variance reacts to
standardized shock magnitude and asymmetry terms rather than modeling variance
directly in levels.

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

Implementation notes:

| Item | Value |
| --- | --- |
| Backend | `arch.arch_model` |
| Required extra | `macroforecast[arch]` |
| R comparison | `rugarch::ugarchspec(variance.model=list(model="eGARCH", garchOrder=c(p, q)))` |
| Internal likelihood? | No. macroforecast validates orders, passes inputs to `arch`, and records metadata/diagnostics. |
| Minimum data | 30 non-missing observations. |

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

Fits a compact p=q=1 Gaussian log-linear realized-GARCH joint likelihood.
Provide `rv` directly or set `realized_variance` to the column in `X`
containing the realized measure. If neither is supplied, macroforecast uses
`y ** 2` as an explicit `rv_proxy` so the caller can still inspect the model
contract. For empirical realized-GARCH work, pass a true realized variance or
realized volatility measure.

The implemented state and measurement equations are:

$$
\log h_t = \omega + \alpha \log x_{t-1} + \beta \log h_{t-1},
$$

$$
z_t = \frac{r_t - \mu}{\sqrt{h_t}},
\qquad
\tau(z_t) = \eta_1 z_t + \eta_2(z_t^2 - 1),
$$

$$
\log x_t = \xi + \delta \log h_t + \tau(z_t) + u_t,
\qquad u_t \sim N(0, \sigma_u^2).
$$

This matches the compact `rugarch` realGARCH recursion in
`rugarch/src/filters.c::realgarchfilter()` for the p=q=1 case:
lagged log realized volatility enters through `alpha`, lagged log latent
variance enters through `beta`, and the measurement equation uses
`xi`, `delta`, `eta1`, and `eta2`. The stationarity-style persistence diagnostic
is:

$$
\text{persistence} = \beta + \delta \alpha.
$$

The multi-step variance forecast uses the conditional expectation recursion:
the first step uses the latest observed realized measure, then future
`tau(z_t)` and measurement shocks are set to zero so
`E[\log x_t \mid h_t] = \xi + \delta \log h_t`.

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

Implementation notes:

| Item | Value |
| --- | --- |
| Backend | Internal SciPy `L-BFGS-B` optimizer. |
| Required extra | None beyond base SciPy stack. |
| R comparison | Compact p=q=1 version of `rugarch::ugarchspec(variance.model=list(model="realGARCH"))` with `realizedVol`. |
| Parameter names | `mu`, `omega`, `alpha`, `beta`, `xi`, `delta`, `eta_1`, `eta_2`, `log_sigma_u`, `persistence`. |
| Restrictions | `alpha > 0`, `beta > 0`, `delta > 0`, and `beta + delta * alpha < 1` during optimization. |
| Minimum data | 30 aligned observations of `y` and realized measure. |

Example:

```python
fit = macroforecast.models.realized_garch(
    returns,
    rv=realized_variance,
    max_iter=2000,
    n_starts=5,
    random_state=0,
)

fit.diagnostics["params"]
fit.predict_variance(horizon=12)
```

## Omitted From The Clean Model API

| Legacy name | Decision |
| --- | --- |
| `lasso_path` | Removed. Use `get_model("lasso")` and `model_selection.select_params()`. |
| `pcr` | Removed. Use `feature_engineering.feature_spec(pca_components=...)` with a regression model. |

- `var_select_order` -- VAR lag-order selection by AIC/BIC/HQ/FPE (vars::VARselect), via statsmodels `VAR.select_order`.

- `gjr_garch` -- GJR-GARCH (Glosten-Jagannathan-Runkle) asymmetric/leverage volatility (arch GARCH, o>0; rugarch gjrGARCH).
- `tgarch` -- Threshold GARCH (TGARCH/Zakoian), absolute-value (power=1) asymmetric volatility.

- `risk_forecast` -- Value-at-Risk and Expected Shortfall forecast from a fitted volatility model (Normal / standardized-t).
- `value_at_risk` -- lower-tail VaR return quantile(s) from a fitted volatility model.
- `expected_shortfall` -- Expected Shortfall (mean return below VaR) from a fitted volatility model.
- `news_impact_curve` -- Engle-Ng (1993) news impact curve: conditional variance vs lagged shock for a fitted GARCH-family model.
- `garch_roll` -- rolling one-step volatility / VaR backtest with periodic refit and coverage summary (rugarch::ugarchroll).

- `var_roots` -- VAR stability check: moduli of the companion-matrix eigenvalues, spectral radius, and is_stable (vars::roots).

- `arima` -- (seasonal) ARIMA model via statsmodels, order (p,d,q) and seasonal_order (P,D,Q,m).
- `auto_arima` -- automatic (seasonal) ARIMA order selection (forecast::auto.arima): KPSS-based d, AICc grid over (p,q[,P,Q]).

### arima

```python
macroforecast.models.arima(y, *, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None)
```

### auto_arima

```python
macroforecast.models.auto_arima(y, *, max_p=5, max_q=5, max_d=2, seasonal=False, m=1, ic="aicc", trend=None)
```

### gjr_garch

```python
macroforecast.models.gjr_garch(y, *, X=None, p=1, o=1, q=1, mean_model="constant", dist="normal", rescale=False)
```

### tgarch

```python
macroforecast.models.tgarch(y, *, X=None, p=1, o=1, q=1, mean_model="constant", dist="normal", rescale=False)
```
