# custom_features

[Back to custom extensions](index.md)

Use custom feature functions when a transformation belongs after
preprocessing and before model fitting. The output must be a numeric feature
matrix aligned to the input date index.

## Direct Function

```python
mf.feature_engineering.custom_features(
    data,
    func,
    *,
    columns=None,
    name=None,
    prefix=None,
    **params,
) -> pandas.DataFrame
```

### Callable Signature

```python
func(source: pandas.DataFrame, *, metadata: dict, **params)
```

### Accepted Return Types

| Return type | Requirement |
| --- | --- |
| `DataFrame` | Same date index or same row count as `source`. |
| `Series` | Same date index or same row count as `source`. |
| 1-D array-like | Length equals `len(source)`. |
| 2-D array-like | Row count equals `len(source)`. |

The returned feature table receives `attrs["macroforecast_metadata_schema"]`
and `attrs["macroforecast_metadata"]`.

## Runner-Safe Step

```python
mf.feature_engineering.custom_step(
    name,
    func=None,
    *,
    fit_func=None,
    transform_func=None,
    columns=None,
    requires_target=False,
    prefix=None,
    **params,
) -> dict
```

### Stateless Step

```python
def square_feature(source, *, metadata=None, suffix="sq"):
    column = source.columns[0]
    return pandas.DataFrame(
        {f"{column}_{suffix}": source[column] ** 2},
        index=source.index,
    )

features = mf.feature_engineering.feature_spec(
    target="target",
    horizon=1,
    predictors=["x", "z"],
    steps=[
        mf.feature_engineering.custom_step(
            "x_square",
            square_feature,
            columns=["x"],
        ),
    ],
)
```

### Fitted Step

```python
features = mf.feature_engineering.feature_spec(
    target="target",
    horizon=1,
    predictors="all",
    steps=[
        mf.feature_engineering.custom_step(
            "my_factor",
            fit_func=my_factor_fit,
            transform_func=my_factor_transform,
            columns=["PAYEMS", "UNRATE", "HOUST"],
            requires_target=True,
            prefix="myf",
            n_components=2,
        ),
    ],
)
```

| Callable | Contract |
| --- | --- |
| `fit_func` | `fit_func(source, target=None, metadata=None, **params) -> state` |
| `transform_func` | `transform_func(source, state=state, metadata=None, **params) -> feature output` |
| fitted state object | `state.transform(source) -> feature output` |
| state-aware `func` | `func(source, state=state, metadata=None, **params) -> feature output` |

Set `requires_target=True` only when the feature fit step needs the resolved
target. Transform-time code should not use future target values.

## Flow

```python
feature_set = features.fit_transform(processed.panel)
fit = mf.models.ridge(feature_set.X.dropna(), feature_set.y.iloc[:, 0].dropna())
```
