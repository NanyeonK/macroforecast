# macroforecast.model_selection

[Back to reference](index.md)

`macroforecast.model_selection` chooses model hyperparameters. It does not
select variables or create features; feature selection belongs to
`macroforecast.feature_engineering`. It can resolve specs from both
`macroforecast.models` and `macroforecast.model_ensemble`.

Use:

- `macroforecast.window` to define train/val splits.
- `macroforecast.metrics` to define the score.
- `macroforecast.model_selection` to evaluate parameter candidates and return the best
  parameter set.

```python
window = mf.window.last_block(validation_size=24)
search = mf.model_selection.grid({"alpha": [0.01, 0.1, 1.0]})

result = mf.model_selection.select_params(
    "ridge",
    X,
    y,
    search=search,
    window=window,
    metric=mf.metrics.rmse,
)
```

## Public Functions

| Task | Functions |
| --- | --- |
| Build a search spec | `fixed()`, `grid()`, `random_search()`, `cv_path()`, `bayesian_search()`, `genetic_search()`, `custom_search()`, `search_spec()` |
| Define stochastic distributions | `uniform()`, `log_uniform()`, `randint()`, `choice()` |
| Run model selection | `select_params()` |
| Store results | `SearchSpec`, `SearchResult`, `SearchError`, `ParamDistribution` |

## SearchSpec

```python
macroforecast.model_selection.SearchSpec(
    method,
    param_grid={},
    param_distributions={},
    n_iter=20,
    random_state=None,
    population_size=12,
    generations=4,
    mutation_rate=0.2,
    custom_func=None,
    custom_params={},
    metadata={},
)
```

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `method` | str | required | `fixed`, `grid`, `cv_path`, `random`, `bayesian`, `genetic`, or `custom`. |
| `param_grid` | dict | `{}` | Explicit finite candidates for `fixed`, `grid`, and `cv_path`. |
| `param_distributions` | dict | `{}` | Sampling rules for `random`, `bayesian`, and `genetic`. |
| `n_iter` | int | `20` | Candidate count for `random`; total sequential evaluations for `bayesian`. |
| `random_state` | int or `None` | `None` | Seed for stochastic searches. |
| `population_size` | int | `12` | Population size for `genetic`. |
| `generations` | int | `4` | Number of generations for `genetic`. |
| `mutation_rate` | float | `0.2` | Per-parameter mutation probability for `genetic`. |
| `custom_func` | callable or `None` | `None` | User search callable used only when `method="custom"`. |
| `custom_params` | dict | `{}` | User parameters passed to `custom_func`. |
| `metadata` | dict | `{}` | Search metadata, including model-owned search-space provenance. |

Output:

`SearchSpec` is consumed by `select_params()`. It also supports
`to_metadata()`, `to_dict()`, and `to_json(path=None)`.

Window and metric are intentionally absent from `SearchSpec`; they are supplied
to `select_params()`.

## SearchResult

```python
macroforecast.model_selection.SearchResult(
    best_params,
    best_score,
    trials,
    metric,
    method,
    window,
    metadata={},
)
```

Output fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `best_params` | dict | Selected parameter values. |
| `best_score` | float | Score for the selected trial. |
| `trials` | pandas DataFrame | One row per evaluated candidate. |
| `metric` | str or callable | Metric used during model selection. |
| `method` | str | Search method used. |
| `window` | str | Canonical window method used. |
| `metadata` | dict | Model, fixed model params, window, search, and runtime metadata. |

`SearchResult.to_frame()` returns a copy of the trial table.
`to_metadata()`, `to_dict()`, and `to_json()` provide JSON-ready exports.

## SearchError

```python
macroforecast.model_selection.SearchError(message, *, trials=None)
```

Raised when every candidate fit fails. The exception carries attempted trial
rows on `.trials`.

```python
try:
    result = mf.model_selection.select_params(model, X, y, search, window=window)
except mf.model_selection.SearchError as err:
    failed_trials = err.trials
```

## fixed

```python
macroforecast.model_selection.fixed(params=None, *, random_state=None)
```

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `params` | dict or `None` | `None` | One parameter combination. |
| `random_state` | int or `None` | `None` | Stored for reproducibility metadata. |

Output: `SearchSpec(method="fixed")`.

## grid

```python
macroforecast.model_selection.grid(param_grid)
```

Input:

| Argument | Type | Meaning |
| --- | --- | --- |
| `param_grid` | dict | Parameter name to iterable values. Scalars are treated as one-value grids. |

Output: `SearchSpec(method="grid")`.

## random_search

```python
macroforecast.model_selection.random_search(param_distributions, *, n_iter=20, random_state=None)
```

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `param_distributions` | dict | required | Distribution builders, lists, tuples, or scalar values. |
| `n_iter` | int | `20` | Number of random candidates. |
| `random_state` | int or `None` | `None` | Seed. |

Output: `SearchSpec(method="random")`.

## cv_path

```python
macroforecast.model_selection.cv_path(param="alpha", values=None)
```

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `param` | str | `"alpha"` | One parameter to sweep. |
| `values` | iterable or `None` | default alpha path | Ordered candidate values. |

Output: `SearchSpec(method="cv_path")`.

## bayesian_search

```python
macroforecast.model_selection.bayesian_search(param_distributions, *, n_iter=20, random_state=None)
```

Creates a sampled-pool Bayesian optimization request. Runtime behavior:

- seeded initial random trials
- Gaussian-process surrogate
- expected improvement over a sampled candidate pool
- random fallback when the surrogate cannot be fit or the candidate pool is
  exhausted

Output: `SearchSpec(method="bayesian")`.

## genetic_search

```python
macroforecast.model_selection.genetic_search(
    param_distributions,
    *,
    population_size=12,
    generations=4,
    mutation_rate=0.2,
    random_state=None,
)
```

Output: `SearchSpec(method="genetic")`.

## custom_search

```python
macroforecast.model_selection.custom_search(
    name,
    func,
    *,
    param_grid=None,
    param_distributions=None,
    n_iter=20,
    random_state=None,
    metadata=None,
    **params,
) -> SearchSpec
```

Builds a user-supplied search request. This is for custom parameter-search
algorithms, not custom metrics. Custom metrics already belong in
`select_params(..., metric=...)`.

The callable receives keyword arguments:

```python
func(
    *,
    model,
    X,
    y,
    splits,
    metric,
    fixed_params,
    search,
    rng,
    maximize,
    evaluate_candidate,
    **params,
)
```

| Argument | Meaning |
| --- | --- |
| `model` | Fit callable resolved from the model name, callable, or `ModelSpec`. |
| `X`, `y` | Aligned model-selection sample. |
| `splits` | List of train/validation position splits. The default contract is temporal unless non-temporal folds are explicitly allowed. |
| `metric` | Resolved metric callable. |
| `fixed_params` | Parameters applied to every candidate. |
| `search` | The prepared `SearchSpec`. |
| `rng` | NumPy random generator seeded by the spec. |
| `maximize` | Whether larger scores are better. |
| `evaluate_candidate` | Package helper for evaluating one parameter dictionary across all splits. |
| `**params` | User parameters supplied to `custom_search(...)`. |

The custom callable must return one of:

| Return type | Meaning |
| --- | --- |
| `list[SearchTrial]` | Already evaluated trial records. |
| `pandas.DataFrame` | Trial table with `trial`, candidate parameter columns, `score`, `n_splits`, `status`, and `error`. |
| `SearchResult` | Existing search result; its trial table is reused. |
| `(records, metadata)` | Any accepted records plus runtime metadata merged into `SearchResult.metadata`. |

The most common pattern is to use `evaluate_candidate` and return the resulting
trial rows:

```python
def ordered_search(
    *,
    model,
    X,
    y,
    splits,
    metric,
    fixed_params,
    evaluate_candidate,
    values,
    **_,
):
    return [
        evaluate_candidate(
            model,
            X,
            y,
            splits,
            metric,
            fixed_params,
            {"alpha": value},
            trial,
        )
        for trial, value in enumerate(values)
    ]

search = mf.model_selection.custom_search(
    "ordered_alpha",
    ordered_search,
    values=(0.01, 0.1, 1.0),
)

result = mf.model_selection.select_params(
    "ridge",
    X,
    y,
    search=search,
    window=window,
)
```

`SearchSpec.to_dict()` and `SearchResult.to_metadata()` store the callable
name and user parameters. The callable source code is not serialized.

## search_spec

```python
macroforecast.model_selection.search_spec(
    model,
    *,
    preset=None,
    method=None,
    random_state=None,
    n_iter=None,
    population_size=None,
    generations=None,
    mutation_rate=None,
)
```

Builds a `SearchSpec` from a model-owned search space.

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | str, callable, or `ModelSpec` | required | Registered model, fit-time model ensemble, or model spec. |
| `preset` | str or `None` | `None` | Model search-space preset. |
| `method` | str or `None` | model default | Override search method. |
| stochastic options | int/float or `None` | `None` | Passed to stochastic search builders. |

Output: `SearchSpec` with model metadata. The same resolver is used for
`macroforecast.models` and `macroforecast.model_ensemble`, so
`search_spec("bagging", preset="small")` returns a fit-time ensemble search
space with `metadata["model_family"] == "model_ensemble"`.

## Distributions

```python
macroforecast.model_selection.uniform(low, high)
macroforecast.model_selection.log_uniform(low, high)
macroforecast.model_selection.randint(low, high)
macroforecast.model_selection.choice(values)
```

Output: `ParamDistribution`.

Rules:

| Function | Meaning |
| --- | --- |
| `uniform()` | Continuous uniform sample on `[low, high)`. |
| `log_uniform()` | Continuous log-uniform sample; bounds must be positive. |
| `randint()` | Inclusive integer sample from `low` to `high`. |
| `choice()` | Categorical sample from explicit values. |

## select_params

```python
macroforecast.model_selection.select_params(
    model,
    X,
    y=None,
    search=None,
    *,
    window=None,
    splits=None,
    metric="mse",
    maximize=False,
    fixed_params=None,
    preset=None,
    method=None,
    random_state=None,
    n_iter=None,
    population_size=None,
    generations=None,
    mutation_rate=None,
    allow_non_temporal_splits=False,
)
```

Input:

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | str, callable, or `ModelSpec` | required | Model or fit-time model ensemble to fit for each candidate. |
| `X` | pandas object | required | Predictors, panel, or target series depending on model input kind. |
| `y` | pandas Series or `None` | `None` | Supervised target when separate from `X`. |
| `search` | `SearchSpec` or `None` | `None` | Explicit search spec. If absent, model-owned search space is used. |
| `window` | `WindowSpec`, str, or `None` | `None` | Window used to create validation splits. Do not pass with `splits`. |
| `splits` | sequence of `(train_pos, validation_pos)` or `None` | `None` | Explicit integer-position validation splits, usually produced by `macroforecast.window`. Do not pass with `window`. |
| `metric` | str or callable | `"mse"` | Metric from `macroforecast.metrics` or custom callable. |
| `maximize` | bool | `False` | Whether larger metric values are better. |
| `fixed_params` | dict or `None` | `None` | Parameters passed to every candidate fit. |
| `preset` | str or `None` | `None` | Model preset when resolving a registered model. |
| `method` | str or `None` | model default | Search method when `search=None`. |
| stochastic options | int/float or `None` | `None` | Used when building a model-owned search spec. |
| `allow_non_temporal_splits` | bool | `False` | Allow explicit splits whose training positions do not all precede validation positions. Use only for replications that intentionally use random folds. |

Output: `SearchResult`.

Example:

```python
window = mf.window.poos(min_train_size=120, validation_size=24, horizon=1)
search = mf.model_selection.search_spec("lasso", preset="small", method="cv_path")

result = mf.model_selection.select_params(
    "lasso",
    X,
    y,
    search=search,
    window=window,
    metric=mf.metrics.mae,
)
```

For loss metrics such as `mse`, `rmse`, and `mae`, keep `maximize=False`.
For custom reward metrics, set `maximize=True`.

When a forecasting runner already has a complete window plan, pass explicit
splits instead of another `window`:

```python
splits = [
    (range(0, 120), range(120, 132)),
    (range(0, 132), range(132, 144)),
]

result = mf.model_selection.select_params(
    "ridge",
    X,
    y,
    search=mf.model_selection.grid({"alpha": [0.01, 0.1, 1.0]}),
    splits=splits,
)
```

`select_params()` validates explicit splits before fitting:

- each split must contain non-empty train and validation integer positions
- positions must be inside `X`/`y`
- train and validation positions cannot overlap
- by default, train positions must precede validation positions
- boolean masks are allowed only when mask length equals the aligned sample

Non-temporal folds are opt-in:

```python
random_window = mf.window.random_kfold(n_splits=5, random_state=123)

result = mf.model_selection.select_params(
    "elastic_net",
    X,
    y,
    search=search,
    window=random_window,
)
```

`mf.window.random_kfold(...)` records that the fold assignment is intentionally
random and the selection metadata stores `temporal_order=False`. If you pass
explicit random folds yourself, set
`allow_non_temporal_splits=True`; otherwise `select_params()` raises. This
keeps ordinary macro-forecast validation time-aware while still allowing
paper replications whose appendix used random iid folds.

`SearchResult.window` is `"explicit_splits"` when `splits` is used. Metadata
stores `split_source`, `n_splits`, and a compact `split_summary` with counts and
position bounds for each split.

When a `ModelSpec` already carries fixed parameters, `select_params()` keeps
those fixed during every candidate fit and stores them in
`SearchResult.metadata["fixed_model_params"]`. The selected `best_params` remain
the searched candidate parameters.
