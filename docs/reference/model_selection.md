# macroforecast.model_selection

[Back to reference](index.md)

Hyperparameter distributions, search specs, search runners, and selection results.

Guide context: [../guide/concepts/models_and_arms.md](../guide/concepts/models_and_arms.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `select_by_information_criterion` | function | Select model hyperparameters by an in-sample information criterion. |
| `ParamDistribution` | class | Sampling rule for one hyperparameter. |
| `SearchError` | class | Raised when parameter search cannot select any successful candidate. |
| `SearchResult` | class | Result returned by ``select_params``. |
| `SearchSpec` | class | Parameter-search specification consumed by ``select_params``. |
| `ValidationSplitterSpec` | class | Validation-split rule carried by a ``SearchSpec``. |
| `bayesian_search` | function | Sequential Gaussian-process Bayesian search request. |
| `choice` | function | Categorical distribution over explicit values. |
| `custom_search` | function | Build a user-supplied parameter-search request. |
| `cv_path` | function | Evaluate an ordered one-parameter path, commonly lasso/ridge alpha values. |
| `explicit_folds` | function | Build fixed-boundary validation folds for a ``SearchSpec``. |
| `fixed` | function | Evaluate one fixed parameter set without tuning. |
| `genetic_search` | function | Lightweight genetic-style stochastic search over parameter distributions. |
| `grid` | function | Grid-search over explicit parameter values. |
| `log_uniform` | function | Continuous log-uniform distribution for positive parameters. |
| `random_search` | function | Seeded random search over parameter distributions. |
| `randint` | function | Inclusive integer distribution. |
| `recursive_threefold` | function | Build recursive three-fold validation with expanding train blocks. |
| `select_params` | function | Select model parameters by temporal validation. |
| `search_spec` | function | Build a SearchSpec from a registered model's owned search space. |
| `uniform` | function | Continuous uniform distribution. |
| `validation_splitter` | function | Build a named validation-splitter override for a ``SearchSpec``. |

## Callable And Class Reference

### select_by_information_criterion

Qualified name: `macroforecast.model_selection.search.select_by_information_criterion`

#### Signature

```python
macroforecast.model_selection.select_by_information_criterion(model: "'str | Callable[..., Any] | ModelSpec'", X: Any, y: Any | None = None, search: SearchSpec | None = None, *, criterion: str = "bic", fixed_params: dict[str, Any] | None = None, preset: str | None = None) -> SearchResult
```

#### Description

Select model hyperparameters by an in-sample information criterion.

Unlike ``select_params``, each candidate is fitted on the whole supplied
sample and scored by an information criterion (BIC by default) computed from
the in-sample residual sum of squares and the parameter count, so no
validation split is used. This matches the order selection the paper applies
to the autoregression and the factor model. The fitted estimator must expose
``ssr_``, ``nobs_`` and ``n_params_``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `'str \| Callable[..., Any] \| ModelSpec'` | `required` |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `search` | positional or keyword | `SearchSpec \| None` | `None` |
| `criterion` | keyword only | `str` | `"bic"` |
| `fixed_params` | keyword only | `dict[str, Any] \| None` | `None` |
| `preset` | keyword only | `str \| None` | `None` |

#### Returns

`SearchResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.select_by_information_criterion(...)
```
### ParamDistribution

Qualified name: `macroforecast.model_selection.types.ParamDistribution`

#### Signature

```python
macroforecast.model_selection.ParamDistribution(kind: DistributionKind, low: float | int | None = None, high: float | int | None = None, choices: tuple[Any, ...] = ()) -> None
```

#### Description

Sampling rule for one hyperparameter.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `kind` | positional or keyword | `DistributionKind` | `required` |
| `low` | positional or keyword | `float \| int \| None` | `None` |
| `high` | positional or keyword | `float \| int \| None` | `None` |
| `choices` | positional or keyword | `tuple[Any, ...]` | `()` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_selection.ParamDistribution(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `kind` | `DistributionKind` | `required` |
| `low` | `float \| int \| None` | `None` |
| `high` | `float \| int \| None` | `None` |
| `choices` | `tuple[Any, ...]` | `()` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `sample` | `sample(self, rng: np.random.Generator) -> Any` | No public docstring is available. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return a JSON-ready distribution description. |
| `validate` | `validate(self) -> None` | Validate distribution bounds before sampling or search execution. |
### SearchError

Qualified name: `macroforecast.model_selection.types.SearchError`

#### Signature

```python
macroforecast.model_selection.SearchError(message: str, *, trials: pd.DataFrame | None = None) -> None
```

#### Description

Raised when parameter search cannot select any successful candidate.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `message` | positional or keyword | `str` | `required` |
| `trials` | keyword only | `pd.DataFrame \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_selection.SearchError(...)
```
### SearchResult

Qualified name: `macroforecast.model_selection.types.SearchResult`

#### Signature

```python
macroforecast.model_selection.SearchResult(best_params: dict[str, Any], best_score: float, trials: pd.DataFrame, metric: MetricLike, method: str, window: str, metadata: dict[str, Any] = <factory>) -> None
```

#### Description

Result returned by ``select_params``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `best_params` | positional or keyword | `dict[str, Any]` | `required` |
| `best_score` | positional or keyword | `float` | `required` |
| `trials` | positional or keyword | `pd.DataFrame` | `required` |
| `metric` | positional or keyword | `MetricLike` | `required` |
| `method` | positional or keyword | `str` | `required` |
| `window` | positional or keyword | `str` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_selection.SearchResult(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `best_params` | `dict[str, Any]` | `required` |
| `best_score` | `float` | `required` |
| `trials` | `pd.DataFrame` | `required` |
| `metric` | `MetricLike` | `required` |
| `method` | `str` | `required` |
| `window` | `str` | `required` |
| `metadata` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self, *, include_trials: bool = True) -> dict[str, Any]` | Return a JSON-ready result dictionary. |
| `to_frame` | `to_frame(self) -> pd.DataFrame` | Return a copy of the trial table. |
| `to_json` | `to_json(self, path: str \| Path \| None = None, *, include_trials: bool = True, indent: int \| None = 2) -> str` | Return JSON text, and optionally write it to ``path``. |
| `to_metadata` | `to_metadata(self) -> dict[str, Any]` | Return JSON-ready result metadata without the trial table. |
### SearchSpec

Qualified name: `macroforecast.model_selection.types.SearchSpec`

#### Signature

```python
macroforecast.model_selection.SearchSpec(method: str, param_grid: dict[str, tuple[Any, ...]] = <factory>, param_distributions: dict[str, ParamDistribution] = <factory>, n_iter: int = 20, random_state: int | None = None, population_size: int = 12, generations: int = 4, mutation_rate: float = 0.2, custom_func: Callable[..., Any] | None = None, custom_params: dict[str, Any] = <factory>, metadata: dict[str, Any] = <factory>, criterion: str | None = None, validation_splitter: ValidationSplitterSpec | Callable[..., Any] | str | None = None) -> None
```

#### Description

Parameter-search specification consumed by ``select_params``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `method` | positional or keyword | `str` | `required` |
| `param_grid` | positional or keyword | `dict[str, tuple[Any, ...]]` | `<factory>` |
| `param_distributions` | positional or keyword | `dict[str, ParamDistribution]` | `<factory>` |
| `n_iter` | positional or keyword | `int` | `20` |
| `random_state` | positional or keyword | `int \| None` | `None` |
| `population_size` | positional or keyword | `int` | `12` |
| `generations` | positional or keyword | `int` | `4` |
| `mutation_rate` | positional or keyword | `float` | `0.2` |
| `custom_func` | positional or keyword | `Callable[..., Any] \| None` | `None` |
| `custom_params` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `criterion` | positional or keyword | `str \| None` | `None` |
| `validation_splitter` | positional or keyword | `ValidationSplitterSpec \| Callable[..., Any] \| str \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_selection.SearchSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `method` | `str` | `required` |
| `param_grid` | `dict[str, tuple[Any, ...]]` | `default_factory` |
| `param_distributions` | `dict[str, ParamDistribution]` | `default_factory` |
| `n_iter` | `int` | `20` |
| `random_state` | `int \| None` | `None` |
| `population_size` | `int` | `12` |
| `generations` | `int` | `4` |
| `mutation_rate` | `float` | `0.2` |
| `custom_func` | `Callable[..., Any] \| None` | `None` |
| `custom_params` | `dict[str, Any]` | `default_factory` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `criterion` | `str \| None` | `None` |
| `validation_splitter` | `ValidationSplitterSpec \| Callable[..., Any] \| str \| None` | `None` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return a JSON-ready full search specification. |
| `to_json` | `to_json(self, path: str \| Path \| None = None, *, indent: int \| None = 2) -> str` | Return JSON text, and optionally write it to ``path``. |
| `to_metadata` | `to_metadata(self) -> dict[str, Any]` | Return search-level metadata without trial results. |
### ValidationSplitterSpec

Qualified name: `macroforecast.model_selection.types.ValidationSplitterSpec`

#### Signature

```python
macroforecast.model_selection.ValidationSplitterSpec(method: str, explicit_folds: tuple[Any, ...] = (), within_fold: WithinFoldMode = "fixed", params: dict[str, Any] = <factory>) -> None
```

#### Description

Validation-split rule carried by a ``SearchSpec``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `method` | positional or keyword | `str` | `required` |
| `explicit_folds` | positional or keyword | `tuple[Any, ...]` | `()` |
| `within_fold` | positional or keyword | `WithinFoldMode` | `"fixed"` |
| `params` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.model_selection.ValidationSplitterSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `method` | `str` | `required` |
| `explicit_folds` | `tuple[Any, ...]` | `()` |
| `within_fold` | `WithinFoldMode` | `"fixed"` |
| `params` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return a JSON-ready splitter specification. |
### bayesian_search

Qualified name: `macroforecast.model_selection.builders.bayesian_search`

#### Signature

```python
macroforecast.model_selection.bayesian_search(param_distributions: dict[str, ParamDistribution | Iterable[Any] | Any], *, n_iter: int = 20, random_state: int | None = None) -> SearchSpec
```

#### Description

Sequential Gaussian-process Bayesian search request.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `param_distributions` | positional or keyword | `dict[str, ParamDistribution \| Iterable[Any] \| Any]` | `required` |
| `n_iter` | keyword only | `int` | `20` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`SearchSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.bayesian_search(...)
```
### choice

Qualified name: `macroforecast.model_selection.builders.choice`

#### Signature

```python
macroforecast.model_selection.choice(values: Iterable[Any]) -> ParamDistribution
```

#### Description

Categorical distribution over explicit values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `values` | positional or keyword | `Iterable[Any]` | `required` |

#### Returns

`ParamDistribution`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.choice(...)
```
### custom_search

Qualified name: `macroforecast.model_selection.builders.custom_search`

#### Signature

```python
macroforecast.model_selection.custom_search(name: str, func: Callable[..., Any], *, param_grid: dict[str, Iterable[Any] | Any] | None = None, param_distributions: dict[str, ParamDistribution | Iterable[Any] | Any] | None = None, n_iter: int = 20, random_state: int | None = None, metadata: dict[str, Any] | None = None, **params: Any) -> SearchSpec
```

#### Description

Build a user-supplied parameter-search request.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `func` | positional or keyword | `Callable[..., Any]` | `required` |
| `param_grid` | keyword only | `dict[str, Iterable[Any] \| Any] \| None` | `None` |
| `param_distributions` | keyword only | `dict[str, ParamDistribution \| Iterable[Any] \| Any] \| None` | `None` |
| `n_iter` | keyword only | `int` | `20` |
| `random_state` | keyword only | `int \| None` | `None` |
| `metadata` | keyword only | `dict[str, Any] \| None` | `None` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`SearchSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.custom_search(...)
```
### cv_path

Qualified name: `macroforecast.model_selection.builders.cv_path`

#### Signature

```python
macroforecast.model_selection.cv_path(*, param: str = "alpha", values: Iterable[Any] | None = None) -> SearchSpec
```

#### Description

Evaluate an ordered one-parameter path, commonly lasso/ridge alpha values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `param` | keyword only | `str` | `"alpha"` |
| `values` | keyword only | `Iterable[Any] \| None` | `None` |

#### Returns

`SearchSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.cv_path(...)
```
### explicit_folds

Qualified name: `macroforecast.model_selection.splitters.explicit_folds`

#### Signature

```python
macroforecast.model_selection.explicit_folds(boundaries: Sequence[Any], *, within_fold: str = "fixed") -> ValidationSplitterSpec
```

#### Description

Build fixed-boundary validation folds for a ``SearchSpec``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `boundaries` | positional or keyword | `Sequence[Any]` | `required` |
| `within_fold` | keyword only | `str` | `"fixed"` |

#### Returns

`ValidationSplitterSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.explicit_folds(...)
```
### fixed

Qualified name: `macroforecast.model_selection.builders.fixed`

#### Signature

```python
macroforecast.model_selection.fixed(params: dict[str, Any] | None = None, *, random_state: int | None = None) -> SearchSpec
```

#### Description

Evaluate one fixed parameter set without tuning.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `params` | positional or keyword | `dict[str, Any] \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`SearchSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.fixed(...)
```
### genetic_search

Qualified name: `macroforecast.model_selection.builders.genetic_search`

#### Signature

```python
macroforecast.model_selection.genetic_search(param_distributions: dict[str, ParamDistribution | Iterable[Any] | Any], *, population_size: int = 12, generations: int = 4, mutation_rate: float = 0.2, random_state: int | None = None) -> SearchSpec
```

#### Description

Lightweight genetic-style stochastic search over parameter distributions.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `param_distributions` | positional or keyword | `dict[str, ParamDistribution \| Iterable[Any] \| Any]` | `required` |
| `population_size` | keyword only | `int` | `12` |
| `generations` | keyword only | `int` | `4` |
| `mutation_rate` | keyword only | `float` | `0.2` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`SearchSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.genetic_search(...)
```
### grid

Qualified name: `macroforecast.model_selection.builders.grid`

#### Signature

```python
macroforecast.model_selection.grid(param_grid: dict[str, Iterable[Any] | Any], *, validation_splitter: ValidationSplitterSpec | Callable[..., Any] | str | None = None) -> SearchSpec
```

#### Description

Grid-search over explicit parameter values.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `param_grid` | positional or keyword | `dict[str, Iterable[Any] \| Any]` | `required` |
| `validation_splitter` | keyword only | `ValidationSplitterSpec \| Callable[..., Any] \| str \| None` | `None` |

#### Returns

`SearchSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.grid(...)
```
### log_uniform

Qualified name: `macroforecast.model_selection.builders.log_uniform`

#### Signature

```python
macroforecast.model_selection.log_uniform(low: float, high: float) -> ParamDistribution
```

#### Description

Continuous log-uniform distribution for positive parameters.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `low` | positional or keyword | `float` | `required` |
| `high` | positional or keyword | `float` | `required` |

#### Returns

`ParamDistribution`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.log_uniform(...)
```
### random_search

Qualified name: `macroforecast.model_selection.builders.random_search`

#### Signature

```python
macroforecast.model_selection.random_search(param_distributions: dict[str, ParamDistribution | Iterable[Any] | Any], *, n_iter: int = 20, random_state: int | None = None) -> SearchSpec
```

#### Description

Seeded random search over parameter distributions.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `param_distributions` | positional or keyword | `dict[str, ParamDistribution \| Iterable[Any] \| Any]` | `required` |
| `n_iter` | keyword only | `int` | `20` |
| `random_state` | keyword only | `int \| None` | `None` |

#### Returns

`SearchSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.random_search(...)
```
### randint

Qualified name: `macroforecast.model_selection.builders.randint`

#### Signature

```python
macroforecast.model_selection.randint(low: int, high: int) -> ParamDistribution
```

#### Description

Inclusive integer distribution.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `low` | positional or keyword | `int` | `required` |
| `high` | positional or keyword | `int` | `required` |

#### Returns

`ParamDistribution`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.randint(...)
```
### recursive_threefold

Qualified name: `macroforecast.model_selection.splitters.recursive_threefold`

#### Signature

```python
macroforecast.model_selection.recursive_threefold() -> ValidationSplitterSpec
```

#### Description

Build recursive three-fold validation with expanding train blocks.

#### Returns

`ValidationSplitterSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.recursive_threefold(...)
```
### select_params

Qualified name: `macroforecast.model_selection.search.select_params`

#### Signature

```python
macroforecast.model_selection.select_params(model: str | Callable[..., Any] | ModelSpec, X: Any, y: Any | None = None, search: SearchSpec | None = None, *, window: WindowSpec | str | None = None, splits: Sequence[tuple[Any, Any]] | None = None, metric: MetricLike = "mse", maximize: bool = False, fixed_params: dict[str, Any] | None = None, preset: str | None = None, method: str | None = None, random_state: int | None = None, n_iter: int | None = None, population_size: int | None = None, generations: int | None = None, mutation_rate: float | None = None, allow_non_temporal_splits: bool = False) -> SearchResult
```

#### Description

Select model parameters by temporal validation.

``model`` can be a model name, a ``ModelSpec``, or a callable such as
``macroforecast.models.ridge`` that returns an object with ``predict(X)``.
Registered models own their default parameters and hyperparameter spaces.
This function evaluates parameter candidates. Validation timing can be
supplied either as a window spec or as explicit integer-position splits
produced by ``macroforecast.window``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `search` | positional or keyword | `SearchSpec \| None` | `None` |
| `window` | keyword only | `WindowSpec \| str \| None` | `None` |
| `splits` | keyword only | `Sequence[tuple[Any, Any]] \| None` | `None` |
| `metric` | keyword only | `MetricLike` | `"mse"` |
| `maximize` | keyword only | `bool` | `False` |
| `fixed_params` | keyword only | `dict[str, Any] \| None` | `None` |
| `preset` | keyword only | `str \| None` | `None` |
| `method` | keyword only | `str \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `None` |
| `n_iter` | keyword only | `int \| None` | `None` |
| `population_size` | keyword only | `int \| None` | `None` |
| `generations` | keyword only | `int \| None` | `None` |
| `mutation_rate` | keyword only | `float \| None` | `None` |
| `allow_non_temporal_splits` | keyword only | `bool` | `False` |

#### Returns

`SearchResult`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.select_params(...)
```
### search_spec

Qualified name: `macroforecast.model_selection.builders.search_spec`

#### Signature

```python
macroforecast.model_selection.search_spec(model: str | Callable[..., Any] | ModelSpec, *, preset: str | None = None, method: str | None = None, random_state: int | None = None, n_iter: int | None = None, population_size: int | None = None, generations: int | None = None, mutation_rate: float | None = None) -> SearchSpec
```

#### Description

Build a SearchSpec from a registered model's owned search space.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |
| `preset` | keyword only | `str \| None` | `None` |
| `method` | keyword only | `str \| None` | `None` |
| `random_state` | keyword only | `int \| None` | `None` |
| `n_iter` | keyword only | `int \| None` | `None` |
| `population_size` | keyword only | `int \| None` | `None` |
| `generations` | keyword only | `int \| None` | `None` |
| `mutation_rate` | keyword only | `float \| None` | `None` |

#### Returns

`SearchSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.search_spec(...)
```
### uniform

Qualified name: `macroforecast.model_selection.builders.uniform`

#### Signature

```python
macroforecast.model_selection.uniform(low: float, high: float) -> ParamDistribution
```

#### Description

Continuous uniform distribution.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `low` | positional or keyword | `float` | `required` |
| `high` | positional or keyword | `float` | `required` |

#### Returns

`ParamDistribution`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.uniform(...)
```
### validation_splitter

Qualified name: `macroforecast.model_selection.splitters.validation_splitter`

#### Signature

```python
macroforecast.model_selection.validation_splitter(method: str, **params: Any) -> ValidationSplitterSpec
```

#### Description

Build a named validation-splitter override for a ``SearchSpec``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `method` | positional or keyword | `str` | `required` |
| `params` | var keyword | `Any` | `required` |

#### Returns

`ValidationSplitterSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.model_selection.validation_splitter(...)
```
