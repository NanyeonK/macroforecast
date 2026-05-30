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
)
```

| Attribute | Type | Description |
| --- | --- | --- |
| `estimator` | object | Fitted underlying estimator. |
| `model` | str | Canonical model name. |
| `feature_names` | tuple[str, ...] | Feature columns used at fit time. |
| `target_name` | str or `None` | Target name when available. |
| `metadata` | dict | Fit metadata such as `n_obs`, `alpha`, or tree budget. |

`ModelFit.predict(X)` returns a `pandas.Series` named `"prediction"` and keeps
the index of the provided `X` when `X` is a DataFrame.

Volatility functions return `VolatilityFit`, which extends `ModelFit` with
`predict_variance(horizon=1)` and `conditional_volatility`.

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
    metric=macroforecast.evaluation.rmse,
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

`ModelSpec` is callable:

```python
model = macroforecast.models.get_model("ridge", params={"alpha": 0.5})
fit = model(X, y)
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
`input_kind`, `default_search_method`, `default_preset`, available `presets`,
and `n_tunable`.

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

## Registered Model Catalog

| Model | Family | Input kind | Default search | Presets |
| --- | --- | --- | --- | --- |
| `ols` | linear | supervised | `grid` | none |
| `ridge` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `lasso` | linear | supervised | `cv_path` | `small`, `standard`, `wide` |
| `elastic_net` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `bayesian_ridge` | linear | supervised | `grid` | none |
| `huber` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `glmboost` | linear | supervised | `grid` | `small`, `standard`, `wide` |
| `pcr` | factor | supervised | `grid` | `small`, `standard`, `wide` |
| `ar` | timeseries | target | `grid` | `small`, `standard`, `wide` |
| `var` | timeseries | panel | `grid` | `small`, `standard`, `wide` |
| `far` | factor | supervised | `grid` | `small`, `standard`, `wide` |
| `favar` | factor | supervised | `grid` | `small`, `standard`, `wide` |
| `decision_tree` | tree | supervised | `grid` | `small`, `standard`, `wide` |
| `random_forest` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `extra_trees` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `gradient_boosting` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `xgboost` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `lightgbm` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `catboost` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `mars` | spline | supervised | `grid` | none |
| `slow_growing_tree` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `quantile_regression_forest` | tree | supervised | `random` | `small`, `standard`, `wide` |
| `bagging` | ensemble | supervised | `random` | `small`, `standard`, `wide` |
| `booging` | ensemble | supervised | `random` | `small`, `standard`, `wide` |
| `macro_random_forest` | tree | supervised | `grid` | none |
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

## Factor And Time-Series Models

### pcr

```python
macroforecast.models.pcr(X, y, *, n_components=3, random_state=0)
```

Fits principal component regression: PCA on `X`, then OLS on the component
scores.

| Parameter | Default | Tunable | Meaning |
| --- | --- | --- | --- |
| `n_components` | `3` | yes | Number of principal components. |
| `random_state` | `0` | fixed by preset | PCA random seed. |

| Preset | `n_components` |
| --- | --- |
| `small` | `(1, 2, 3)` |
| `standard` | `(1, 2, 3, 5, 8)` |
| `wide` | `(1, 2, 3, 5, 8, 10, 12, 20)` |

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

### mars

```python
macroforecast.models.mars(X, y, **kwargs)
```

Fits `pyearth.Earth`. Requires `macroforecast[mars]`.

| Item | Value |
| --- | --- |
| Input | `X`, `y` |
| Output | `ModelFit` |
| Default params | passed through to `pyearth.Earth` |
| Tunable params | none in the clean preset catalog |
| Preset search spaces | none |

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
| `max_depth` | `10` | yes | Maximum tree depth. |
| `min_leaf_size` | `5` | yes | Minimum effective leaf size. |
| `random_state` | `0` | fixed by preset | Tree random seed. |

| Preset | `eta` | `herfindahl_threshold` | `max_depth` | `min_leaf_size` |
| --- | --- | --- | --- | --- |
| `small` | `(0.05, 0.1)` | `(0.2, 0.3)` | `(5, 10)` | `(3, 5)` |
| `standard` | `(0.03, 0.05, 0.1)` | `(0.15, 0.25, 0.35)` | `(5, 10, None)` | `(3, 5, 10)` |
| `wide` | `(0.01, 0.03, 0.05, 0.1, 0.2)` | `(0.1, 0.15, 0.25, 0.35, 0.5)` | `(3, 5, 10, 20, None)` | `(2, 3, 5, 10)` |

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
underlying estimator exposes `predict_quantiles(X, levels=None)`.

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
| `strategy` | `"standard"` | manual | Bootstrap strategy: `standard` or `block`. |
| `block_length` | `4` | manual | Block length when `strategy="block"`. |
| `random_state` | `0` | fixed by preset | Ensemble random seed. |

| Preset | `base` | `n_estimators` | `max_samples` |
| --- | --- | --- | --- |
| `small` | `("ridge", "lasso")` | `(10, 25)` | `(0.6, 0.8)` |
| `standard` | `("ridge", "lasso", "decision_tree")` | `(25, 50, 100)` | `(0.5, 0.7, 0.9)` |
| `wide` | `("ridge", "lasso", "elastic_net", "decision_tree", "random_forest")` | `(25, 50, 100, 200)` | `(0.4, 0.6, 0.8, 1.0)` |

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
| `inner_learning_rate` | `0.1` | manual | Inner boosting learning rate. |
| `inner_max_depth` | `3` | yes | Inner boosting tree depth. |
| `inner_subsample` | `0.5` | manual | Inner boosting subsample share. |
| `random_state` | `0` | fixed by preset | Ensemble random seed. |

| Preset | `B` | `sample_frac` | `inner_n_estimators` | `inner_max_depth` |
| --- | --- | --- | --- | --- |
| `small` | `(5, 10)` | `(0.6, 0.8)` | `(100, 300)` | `(2, 3)` |
| `standard` | `(10, 25, 50)` | `(0.5, 0.75, 0.9)` | `(300, 750, 1500)` | `(2, 3, 5)` |
| `wide` | `(10, 25, 50, 100)` | `(0.4, 0.6, 0.75, 0.9)` | `(300, 750, 1500, 2500)` | `(2, 3, 5, 8)` |

Default selection method: `random`.

### macro_random_forest

```python
macroforecast.models.macro_random_forest(X, y, **kwargs)
```

Reserved for Goulet Coulombe's Macroeconomic Random Forest reference backend.
The callable exists in the clean API, but it raises a clear error until that
backend is added to the clean package.

## Volatility Models

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
| `mlp`, `lstm`, `gru`, `transformer`, `hemisphere_nn` | Deferred because torch/deep is intentionally out of scope. |
| `midas_almon`, `midas_beta`, `midas_step`, `unrestricted_midas` | Deferred as a specialized mixed-frequency block. |
