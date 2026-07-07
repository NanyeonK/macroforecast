# macroforecast.model_ensemble

[Back to reference](index.md)

Fit-time ensemble estimators and ensemble model specs.

Guide context: [../guide/concepts/models_and_arms.md](../guide/concepts/models_and_arms.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `BaggingRegressor` | class | Bootstrap or block-bootstrap ensemble over supported base estimators. |
| `BoogingRegressor` | class | Bagging of intentionally overfit stochastic gradient boosting models. |
| `MODEL_ENSEMBLE_BASE_ESTIMATORS` | data | dict() -> new empty dictionary |
| `MODEL_ENSEMBLE_SPECS` | data | dict() -> new empty dictionary |
| `RandomSubspaceRegressor` | class | Fit base models on randomly selected feature subsets and average them. |
| `StackingRegressor` | class | Out-of-fold stacking over supported base estimators. |
| `SuperLearnerRegressor` | class | Convex-weight Super Learner over supported base estimators. |
| `bagging` | function | Fit a bootstrap-aggregated fit-time model ensemble. |
| `booging` | function | Fit Booging: bagged overfit stochastic gradient boosting with augmentation. |
| `custom_model_ensemble` | function | Build a user-owned fit-time model-ensemble spec. |
| `describe_model_ensemble` | function | Describe fit-time model-ensemble parameters and preset spaces. |
| `get_model_ensemble` | function | Return a fit-time model-ensemble spec by name, callable, or spec. |
| `list_model_ensemble_bases` | function | Return supported inner estimators for fit-time model ensembles. |
| `list_model_ensemble_specs` | function | List registered fit-time model-ensemble specs. |
| `model_ensemble_search_space` | function | Return a model-ensemble-owned hyperparameter space. |
| `random_subspace` | function | Fit a random-subspace model ensemble. |
| `stacking` | function | Fit an out-of-fold stacked model ensemble. |
| `subagging` | function | Fit subagging: sampling without replacement before member fits. |
| `super_learner` | function | Fit a SuperLearner-style convex-weight model ensemble. |

## Data And Module Values

### `MODEL_ENSEMBLE_BASE_ESTIMATORS`

Kind: `data`

```python
MODEL_ENSEMBLE_BASE_ESTIMATORS = dict(10 entries: decision_tree, elastic_net, extra_trees, gradient_boosting, knn, lasso, ols, random_forest, ridge, svr)
```
### `MODEL_ENSEMBLE_SPECS`

Kind: `data`

```python
MODEL_ENSEMBLE_SPECS = dict(6 entries: bagging, booging, random_subspace, stacking, subagging, super_learner)
```

## Callable And Class Reference

### BaggingRegressor

Qualified name: `macroforecast.model_ensemble.core.BaggingRegressor`

#### Signature

```python
macroforecast.model_ensemble.BaggingRegressor(*, base: str = "ridge", n_estimators: int = 50, max_samples: float = 0.8, random_state: int = 0, base_params: dict[str, Any] | None = None, strategy: str = "standard", block_length: int = 4, replace: bool = True, max_features: float | int | str | None = None) -> None
```

#### Description

Bootstrap or block-bootstrap ensemble over supported base estimators.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `base` | keyword only | `str` | `"ridge"` |
| `n_estimators` | keyword only | `int` | `50` |
| `max_samples` | keyword only | `float` | `0.8` |
| `random_state` | keyword only | `int` | `0` |
| `base_params` | keyword only | `dict[str, Any] \| None` | `None` |
| `strategy` | keyword only | `str` | `"standard"` |
| `block_length` | keyword only | `int` | `4` |
| `replace` | keyword only | `bool` | `True` |
| `max_features` | keyword only | `float \| int \| str \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_ensemble.BaggingRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'BaggingRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
| `predict_quantiles` | `predict_quantiles(self, X: pd.DataFrame, levels: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[float, np.ndarray]` | No public docstring is available. |
### BoogingRegressor

Qualified name: `macroforecast.model_ensemble.core.BoogingRegressor`

#### Signature

```python
macroforecast.model_ensemble.BoogingRegressor(*, B: int = 100, sample_frac: float = 0.75, inner_n_estimators: int = 1000, inner_learning_rate: float = 0.3, inner_max_depth: int = 3, inner_subsample: float = 0.5, mtry: float | int | str | None = None, data_aug: bool = False, noise_level: float = 0.3, shuffle_rate: float = 0.2, n_augmented_copies: int = 2, scale_continuous: bool = True, fix_seeds: bool = True, random_state: int = 0, sampling_rate: float | None = None, n_trees: int | None = None, nu: float | None = None, tree_depth: int | None = None, bf: float | None = None, max_features: float | int | str | None = None, da_noise_frac: float | None = None, da_drop_rate: float | None = 0.2) -> None
```

#### Description

Bagging of intentionally overfit stochastic gradient boosting models.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `B` | keyword only | `int` | `100` |
| `sample_frac` | keyword only | `float` | `0.75` |
| `inner_n_estimators` | keyword only | `int` | `1000` |
| `inner_learning_rate` | keyword only | `float` | `0.3` |
| `inner_max_depth` | keyword only | `int` | `3` |
| `inner_subsample` | keyword only | `float` | `0.5` |
| `mtry` | keyword only | `float \| int \| str \| None` | `None` |
| `data_aug` | keyword only | `bool` | `False` |
| `noise_level` | keyword only | `float` | `0.3` |
| `shuffle_rate` | keyword only | `float` | `0.2` |
| `n_augmented_copies` | keyword only | `int` | `2` |
| `scale_continuous` | keyword only | `bool` | `True` |
| `fix_seeds` | keyword only | `bool` | `True` |
| `random_state` | keyword only | `int` | `0` |
| `sampling_rate` | keyword only | `float \| None` | `None` |
| `n_trees` | keyword only | `int \| None` | `None` |
| `nu` | keyword only | `float \| None` | `None` |
| `tree_depth` | keyword only | `int \| None` | `None` |
| `bf` | keyword only | `float \| None` | `None` |
| `max_features` | keyword only | `float \| int \| str \| None` | `None` |
| `da_noise_frac` | keyword only | `float \| None` | `None` |
| `da_drop_rate` | keyword only | `float \| None` | `0.2` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_ensemble.BoogingRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'BoogingRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
| `predict_quantiles` | `predict_quantiles(self, X: pd.DataFrame, levels: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[float, np.ndarray]` | No public docstring is available. |
### RandomSubspaceRegressor

Qualified name: `macroforecast.model_ensemble.core.RandomSubspaceRegressor`

#### Signature

```python
macroforecast.model_ensemble.RandomSubspaceRegressor(*, base: str = "ridge", n_estimators: int = 100, max_features: float | int | str = 0.5, max_samples: float = 1.0, random_state: int = 0, base_params: dict[str, Any] | None = None) -> None
```

#### Description

Fit base models on randomly selected feature subsets and average them.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `base` | keyword only | `str` | `"ridge"` |
| `n_estimators` | keyword only | `int` | `100` |
| `max_features` | keyword only | `float \| int \| str` | `0.5` |
| `max_samples` | keyword only | `float` | `1.0` |
| `random_state` | keyword only | `int` | `0` |
| `base_params` | keyword only | `dict[str, Any] \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_ensemble.RandomSubspaceRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'RandomSubspaceRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
| `predict_quantiles` | `predict_quantiles(self, X: pd.DataFrame, levels: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[float, np.ndarray]` | No public docstring is available. |
### StackingRegressor

Qualified name: `macroforecast.model_ensemble.core.StackingRegressor`

#### Signature

```python
macroforecast.model_ensemble.StackingRegressor(*, models: Sequence[str] = ('ridge', 'lasso', 'random_forest'), meta_model: str = "ridge", n_splits: int = 5, splitter: str = "forward", random_state: int = 0, model_params: dict[str, dict[str, Any]] | None = None, meta_params: dict[str, Any] | None = None, passthrough: bool = False) -> None
```

#### Description

Out-of-fold stacking over supported base estimators.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `models` | keyword only | `Sequence[str]` | `("ridge", "lasso", "random_forest")` |
| `meta_model` | keyword only | `str` | `"ridge"` |
| `n_splits` | keyword only | `int` | `5` |
| `splitter` | keyword only | `str` | `"forward"` |
| `random_state` | keyword only | `int` | `0` |
| `model_params` | keyword only | `dict[str, dict[str, Any]] \| None` | `None` |
| `meta_params` | keyword only | `dict[str, Any] \| None` | `None` |
| `passthrough` | keyword only | `bool` | `False` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_ensemble.StackingRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'StackingRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
### SuperLearnerRegressor

Qualified name: `macroforecast.model_ensemble.core.SuperLearnerRegressor`

#### Signature

```python
macroforecast.model_ensemble.SuperLearnerRegressor(*, models: Sequence[str] = ('ridge', 'lasso', 'random_forest'), n_splits: int = 5, splitter: str = "forward", weight_method: str = "nnls", random_state: int = 0, model_params: dict[str, dict[str, Any]] | None = None) -> None
```

#### Description

Convex-weight Super Learner over supported base estimators.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `models` | keyword only | `Sequence[str]` | `("ridge", "lasso", "random_forest")` |
| `n_splits` | keyword only | `int` | `5` |
| `splitter` | keyword only | `str` | `"forward"` |
| `weight_method` | keyword only | `str` | `"nnls"` |
| `random_state` | keyword only | `int` | `0` |
| `model_params` | keyword only | `dict[str, dict[str, Any]] \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_ensemble.SuperLearnerRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'SuperLearnerRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
### bagging

Qualified name: `macroforecast.model_ensemble.core.bagging`

#### Signature

```python
macroforecast.model_ensemble.bagging(X: Any, y: Any | None = None, *, base: str = "ridge", n_estimators: int = 50, max_samples: float = 0.8, random_state: int = 0, base_params: dict[str, Any] | None = None, strategy: str = "standard", block_length: int = 4, replace: bool = True, max_features: float | int | str | None = None) -> ModelFit
```

#### Description

Fit a bootstrap-aggregated fit-time model ensemble.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `base` | keyword only | `str` | `"ridge"` |
| `n_estimators` | keyword only | `int` | `50` |
| `max_samples` | keyword only | `float` | `0.8` |
| `random_state` | keyword only | `int` | `0` |
| `base_params` | keyword only | `dict[str, Any] \| None` | `None` |
| `strategy` | keyword only | `str` | `"standard"` |
| `block_length` | keyword only | `int` | `4` |
| `replace` | keyword only | `bool` | `True` |
| `max_features` | keyword only | `float \| int \| str \| None` | `None` |

#### Returns

`ModelFit`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.bagging(...)
```
### booging

Qualified name: `macroforecast.model_ensemble.core.booging`

#### Signature

```python
macroforecast.model_ensemble.booging(X: Any, y: Any | None = None, *, B: int = 100, sample_frac: float = 0.75, inner_n_estimators: int = 1000, inner_learning_rate: float = 0.3, inner_max_depth: int = 3, inner_subsample: float = 0.5, mtry: float | int | str | None = None, data_aug: bool = False, noise_level: float = 0.3, shuffle_rate: float = 0.2, n_augmented_copies: int = 2, scale_continuous: bool = True, fix_seeds: bool = True, random_state: int = 0, sampling_rate: float | None = None, n_trees: int | None = None, nu: float | None = None, tree_depth: int | None = None, bf: float | None = None, max_features: float | int | str | None = None, da_noise_frac: float | None = None, da_drop_rate: float | None = 0.2) -> ModelFit
```

#### Description

Fit Booging: bagged overfit stochastic gradient boosting with augmentation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `B` | keyword only | `int` | `100` |
| `sample_frac` | keyword only | `float` | `0.75` |
| `inner_n_estimators` | keyword only | `int` | `1000` |
| `inner_learning_rate` | keyword only | `float` | `0.3` |
| `inner_max_depth` | keyword only | `int` | `3` |
| `inner_subsample` | keyword only | `float` | `0.5` |
| `mtry` | keyword only | `float \| int \| str \| None` | `None` |
| `data_aug` | keyword only | `bool` | `False` |
| `noise_level` | keyword only | `float` | `0.3` |
| `shuffle_rate` | keyword only | `float` | `0.2` |
| `n_augmented_copies` | keyword only | `int` | `2` |
| `scale_continuous` | keyword only | `bool` | `True` |
| `fix_seeds` | keyword only | `bool` | `True` |
| `random_state` | keyword only | `int` | `0` |
| `sampling_rate` | keyword only | `float \| None` | `None` |
| `n_trees` | keyword only | `int \| None` | `None` |
| `nu` | keyword only | `float \| None` | `None` |
| `tree_depth` | keyword only | `int \| None` | `None` |
| `bf` | keyword only | `float \| None` | `None` |
| `max_features` | keyword only | `float \| int \| str \| None` | `None` |
| `da_noise_frac` | keyword only | `float \| None` | `None` |
| `da_drop_rate` | keyword only | `float \| None` | `0.2` |

#### Returns

`ModelFit`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.booging(...)
```
### custom_model_ensemble

Qualified name: `macroforecast.model_ensemble.specs.custom_model_ensemble`

#### Signature

```python
macroforecast.model_ensemble.custom_model_ensemble(name: str, fit_func: Callable[..., Any], *, default_params: Mapping[str, Any] | None = None, parameters: tuple[ModelParameter, ...] = (), search_spaces: SearchSpaces | None = None, default_search_method: str = "grid", default_preset: str = "standard", backend: str = "custom", description: str | None = None) -> ModelSpec
```

#### Description

Build a user-owned fit-time model-ensemble spec.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `fit_func` | positional or keyword | `Callable[..., Any]` | `required` |
| `default_params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `parameters` | keyword only | `tuple[ModelParameter, ...]` | `()` |
| `search_spaces` | keyword only | `SearchSpaces \| None` | `None` |
| `default_search_method` | keyword only | `str` | `"grid"` |
| `default_preset` | keyword only | `str` | `"standard"` |
| `backend` | keyword only | `str` | `"custom"` |
| `description` | keyword only | `str \| None` | `None` |

#### Returns

`ModelSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.custom_model_ensemble(...)
```
### describe_model_ensemble

Qualified name: `macroforecast.model_ensemble.specs.describe_model_ensemble`

#### Signature

```python
macroforecast.model_ensemble.describe_model_ensemble(ensemble: str | Callable[..., Any] | ModelSpec) -> pd.DataFrame
```

#### Description

Describe fit-time model-ensemble parameters and preset spaces.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `ensemble` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.describe_model_ensemble(...)
```
### get_model_ensemble

Qualified name: `macroforecast.model_ensemble.specs.get_model_ensemble`

#### Signature

```python
macroforecast.model_ensemble.get_model_ensemble(ensemble: str | Callable[..., Any] | ModelSpec, *, preset: str | None = None, params: Mapping[str, Any] | None = None) -> ModelSpec
```

#### Description

Return a fit-time model-ensemble spec by name, callable, or spec.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `ensemble` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |
| `preset` | keyword only | `str \| None` | `None` |
| `params` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ModelSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.get_model_ensemble(...)
```
### list_model_ensemble_bases

Qualified name: `macroforecast.model_ensemble.core.list_model_ensemble_bases`

#### Signature

```python
macroforecast.model_ensemble.list_model_ensemble_bases() -> pd.DataFrame
```

#### Description

Return supported inner estimators for fit-time model ensembles.

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.list_model_ensemble_bases(...)
```
### list_model_ensemble_specs

Qualified name: `macroforecast.model_ensemble.specs.list_model_ensemble_specs`

#### Signature

```python
macroforecast.model_ensemble.list_model_ensemble_specs(*, family: str | None = None) -> pd.DataFrame
```

#### Description

List registered fit-time model-ensemble specs.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `family` | keyword only | `str \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.list_model_ensemble_specs(...)
```
### model_ensemble_search_space

Qualified name: `macroforecast.model_ensemble.specs.model_ensemble_search_space`

#### Signature

```python
macroforecast.model_ensemble.model_ensemble_search_space(ensemble: str | Callable[..., Any] | ModelSpec, *, preset: str | None = None) -> dict[str, tuple[Any, ...]]
```

#### Description

Return a model-ensemble-owned hyperparameter space.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `ensemble` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |
| `preset` | keyword only | `str \| None` | `None` |

#### Returns

`dict[str, tuple[Any, ...]]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.model_ensemble_search_space(...)
```
### random_subspace

Qualified name: `macroforecast.model_ensemble.core.random_subspace`

#### Signature

```python
macroforecast.model_ensemble.random_subspace(X: Any, y: Any | None = None, *, base: str = "ridge", n_estimators: int = 100, max_features: float | int | str = 0.5, max_samples: float = 1.0, random_state: int = 0, base_params: dict[str, Any] | None = None) -> ModelFit
```

#### Description

Fit a random-subspace model ensemble.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `base` | keyword only | `str` | `"ridge"` |
| `n_estimators` | keyword only | `int` | `100` |
| `max_features` | keyword only | `float \| int \| str` | `0.5` |
| `max_samples` | keyword only | `float` | `1.0` |
| `random_state` | keyword only | `int` | `0` |
| `base_params` | keyword only | `dict[str, Any] \| None` | `None` |

#### Returns

`ModelFit`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.random_subspace(...)
```
### stacking

Qualified name: `macroforecast.model_ensemble.core.stacking`

#### Signature

```python
macroforecast.model_ensemble.stacking(X: Any, y: Any | None = None, *, models: Sequence[str] = ('ridge', 'lasso', 'random_forest'), meta_model: str = "ridge", n_splits: int = 5, splitter: str = "forward", random_state: int = 0, model_params: dict[str, dict[str, Any]] | None = None, meta_params: dict[str, Any] | None = None, passthrough: bool = False) -> ModelFit
```

#### Description

Fit an out-of-fold stacked model ensemble.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `models` | keyword only | `Sequence[str]` | `("ridge", "lasso", "random_forest")` |
| `meta_model` | keyword only | `str` | `"ridge"` |
| `n_splits` | keyword only | `int` | `5` |
| `splitter` | keyword only | `str` | `"forward"` |
| `random_state` | keyword only | `int` | `0` |
| `model_params` | keyword only | `dict[str, dict[str, Any]] \| None` | `None` |
| `meta_params` | keyword only | `dict[str, Any] \| None` | `None` |
| `passthrough` | keyword only | `bool` | `False` |

#### Returns

`ModelFit`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.stacking(...)
```
### subagging

Qualified name: `macroforecast.model_ensemble.core.subagging`

#### Signature

```python
macroforecast.model_ensemble.subagging(X: Any, y: Any | None = None, *, base: str = "ridge", n_estimators: int = 50, max_samples: float = 0.632, random_state: int = 0, base_params: dict[str, Any] | None = None, max_features: float | int | str | None = None) -> ModelFit
```

#### Description

Fit subagging: sampling without replacement before member fits.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `base` | keyword only | `str` | `"ridge"` |
| `n_estimators` | keyword only | `int` | `50` |
| `max_samples` | keyword only | `float` | `0.632` |
| `random_state` | keyword only | `int` | `0` |
| `base_params` | keyword only | `dict[str, Any] \| None` | `None` |
| `max_features` | keyword only | `float \| int \| str \| None` | `None` |

#### Returns

`ModelFit`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.subagging(...)
```
### super_learner

Qualified name: `macroforecast.model_ensemble.core.super_learner`

#### Signature

```python
macroforecast.model_ensemble.super_learner(X: Any, y: Any | None = None, *, models: Sequence[str] = ('ridge', 'lasso', 'random_forest'), n_splits: int = 5, splitter: str = "forward", weight_method: str = "nnls", random_state: int = 0, model_params: dict[str, dict[str, Any]] | None = None) -> ModelFit
```

#### Description

Fit a SuperLearner-style convex-weight model ensemble.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `models` | keyword only | `Sequence[str]` | `("ridge", "lasso", "random_forest")` |
| `n_splits` | keyword only | `int` | `5` |
| `splitter` | keyword only | `str` | `"forward"` |
| `weight_method` | keyword only | `str` | `"nnls"` |
| `random_state` | keyword only | `int` | `0` |
| `model_params` | keyword only | `dict[str, dict[str, Any]] \| None` | `None` |

#### Returns

`ModelFit`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_ensemble.super_learner(...)
```
