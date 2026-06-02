# Custom Window, Selection, And Forecasting

[Back to custom extensions](index.md)

Use these hooks when the timing, parameter search, or forecast-combination
logic is project-specific.

## custom_stage_policy

```python
mf.window.custom_stage_policy(selector, *, name=None, metadata=None) -> mf.window.StagePolicy
```

### Selector Contract

```python
selector(index: pandas.Index, *, item: dict, policy: StagePolicy)
```

The selector may return a boolean mask, a slice, integer positions, or index
labels. The result must select at least one label and must not select dates
outside the supplied index.

### Example

```python
def last_fit_half(index, *, item, policy):
    fit_idx = item["fit_idx"]
    return index[fit_idx[len(fit_idx) // 2 :]]

policy = mf.window.custom_stage_policy(last_fit_half)
```

Use this for non-standard model-selection or preprocessing samples. Standard
expanding, rolling, fixed-reference, and origin-available behavior should use
`mf.window.stage_policy(...)`.

## custom_search

```python
mf.model_selection.custom_search(
    name,
    custom_func,
    **params,
) -> mf.model_selection.SearchSpec
```

### Search Callable Contract

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

The callable may return trial records, a trial `DataFrame`, a `SearchResult`,
or `(records, metadata)`. Prefer `evaluate_candidate(...)` so the package owns
fit/predict/scoring consistency.

## custom_combination

```python
mf.forecasting.custom_combination(
    name,
    func,
    *,
    models=None,
    **params,
) -> mf.forecasting.CombinationSpec
```

### Combination Callable Contract

```python
func(forecasts: pandas.DataFrame, *, actual: pandas.Series, **params)
```

`forecasts` is a wide matrix of base-model predictions indexed by
`(date, origin, origin_pos, horizon)`. The output must be a `Series` or
one-dimensional array-like object aligned to those rows.

### Example

```python
def blend(forecasts, *, actual, weight=0.5):
    return weight * forecasts.iloc[:, 0] + (1.0 - weight) * forecasts.iloc[:, -1]

combination = mf.forecasting.custom_combination(
    "ridge_lasso_blend",
    blend,
    models=["ridge", "lasso"],
    weight=0.25,
)
```

## Boundary

| Hook | Owns | Does not own |
| --- | --- | --- |
| `custom_stage_policy` | sample labels for one stage | fitting or scoring |
| `custom_search` | hyperparameter search algorithm | model default search spaces |
| `custom_combination` | combining already produced forecasts | fit-time model ensembles |
