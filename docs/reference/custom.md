# Custom Extensions

[Back to reference](index.md)

`macroforecast` keeps custom extension points as normal callable functions, not
as a second registry layer. The user can add a dataset, preprocessing function,
feature transform, model, stage policy, or forecast-combination rule by passing
a Python callable at the stage that owns that behavior.

The package still enforces the same contracts around each custom hook:

| Stage | Extension function | Input contract | Output contract |
| --- | --- | --- | --- |
| Data | `mf.data.custom_dataset(...)` | In-memory `pandas.DataFrame` plus date/column/frequency metadata. | `DataBundle` with canonical panel and metadata. |
| Preprocessing | `mf.preprocessing.custom_preprocess(...)` | Canonical panel plus metadata. | `PreprocessedData`. |
| Preprocessing spec | `mf.preprocessing.custom_preprocess_step(...)` | Callable step for `preprocess_spec(custom_steps=[...])`. | Step dictionary consumed by `PreprocessSpec`. |
| Feature engineering | `mf.feature_engineering.custom_features(...)` | Selected feature panel. | Numeric feature `DataFrame`. |
| Feature spec | `mf.feature_engineering.custom_step(...)` | Stateless or fitted feature callable for `feature_spec(steps=[...])`. | Step dictionary consumed by `FeatureSpec`. |
| Models | `mf.models.custom_model(...)` | Fit callable with model metadata and optional search spaces. | `ModelSpec`. |
| Window | `mf.window.custom_stage_policy(...)` | Selector callable for origin-specific sample labels. | `StagePolicy(scope="custom")`. |
| Forecasting | `mf.forecasting.custom_combination(...)` | Callable over base forecast matrix. | `CombinationSpec`. |

## Data

Use `custom_dataset()` when the input data are already in memory.

```python
bundle = mf.data.custom_dataset(
    frame,
    date="date",
    dataset="my_monthly_panel",
    frequency="monthly",
    transform_codes={"INDPRO": 5, "spread": 2},
)
```

The output is a `DataBundle`.

| Output field | Meaning |
| --- | --- |
| `bundle.panel` | Canonical `DataFrame`: `DatetimeIndex` named `date`, sorted, numeric columns, no duplicate dates. |
| `bundle.metadata` | Source labels, frequency labels, transform-code metadata, panel-normalization report, and `custom_dataset` stage. |

Use `load_custom_csv()` or `load_custom_parquet()` when the source is a file.
Use `custom_dataset()` when the data have already been loaded by user code.

## Preprocessing

Use `custom_preprocess()` for one direct panel operation:

```python
def add_spread(panel, *, metadata=None, scale=1.0):
    out = panel.copy()
    out["spread"] = (out["long_rate"] - out["short_rate"]) * scale
    return out

processed = mf.preprocessing.custom_preprocess(
    bundle,
    add_spread,
    name="add_spread",
    scale=100.0,
)
```

Callable signature:

```python
func(panel: pandas.DataFrame, *, metadata: dict, **params)
```

Accepted return types:

| Return type | Meaning |
| --- | --- |
| `DataFrame` | Re-normalized to the canonical panel contract. |
| `DataBundle` | Panel and metadata continue from the bundle. |
| `PreprocessedData` | Full preprocessing object continues. |
| `(DataFrame, metadata)` | Explicit panel and metadata pair. |

For runner use, put the same callable in `preprocess_spec(custom_steps=[...])`:

```python
pre = mf.preprocessing.preprocess_spec(
    transform="none",
    impute="mean",
    custom_steps=[
        mf.preprocessing.custom_preprocess_step("add_spread", add_spread, scale=100.0),
    ],
)
```

The runner applies this under `preprocessing_policy`, so the same callable can
run full-sample, origin-available, fit-window, or fixed-reference depending on
the chosen window policy.

## Feature Engineering

Use `custom_features()` for a direct feature transform:

```python
def square_feature(source, *, metadata=None, suffix="sq"):
    column = source.columns[0]
    return pd.DataFrame({f"{column}_{suffix}": source[column] ** 2}, index=source.index)

X_sq = mf.feature_engineering.custom_features(
    processed,
    square_feature,
    columns=["spread"],
    name="spread_square",
)
```

Callable signature:

```python
func(source: pandas.DataFrame, *, metadata: dict, **params)
```

Accepted return types are `DataFrame`, `Series`, or 1-D/2-D array-like output.
The output must have the same row count or keep a compatible `DatetimeIndex`.

For runner-safe use, put the callable inside `feature_spec(...)`:

```python
features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors=["spread", "UNRATE"],
    steps=[
        mf.feature_engineering.custom_step(
            "spread_square",
            square_feature,
            columns=["spread"],
        ),
    ],
)
```

Fitted custom steps are also supported:

```python
features = mf.feature_engineering.feature_spec(
    target="INDPRO",
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

Fit/transform callable contracts:

| Callable | Signature |
| --- | --- |
| `fit_func` | `fit_func(source, target=None, metadata=None, **params) -> state` |
| `transform_func` | `transform_func(source, state=state, metadata=None, **params) -> feature output` |
| state object | `state.transform(source) -> feature output` |
| state-aware `func` | `func(source, state=state, metadata=None, **params) -> feature output` |

Set `requires_target=True` only when the fit step needs the resolved target.
That requires one target and one horizon. The runner never passes target values
to transform-time code.

## Models

Use `custom_model()` to create a `ModelSpec` from a user fit function.

```python
class MeanFit:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)

def mean_model(X, y, *, offset=0.0):
    return MeanFit(pd.Series(y).mean() + offset)

model = mf.models.custom_model(
    "mean_model",
    mean_model,
    default_params={"offset": 0.0},
    default_preset="small",
    search_spaces={"small": {"offset": (-0.1, 0.0, 0.1)}},
)
```

Default callable contract:

```python
fit_func(X, y, **params) -> fitted_object
fitted_object.predict(X_test) -> predictions
```

`predict()` output must have length `len(X_test)`. A pandas output must use
`X_test.index` or `RangeIndex(len(X_test))`; otherwise the runner raises.

Pass the returned `ModelSpec` directly:

```python
result = mf.forecasting.run(
    panel,
    {"mean": model, "ridge": "ridge"},
    window=window,
    features=features,
)
```

## Window Policies

Use `custom_stage_policy()` when a runner stage should use a non-standard
sample definition.

```python
def last_half_of_fit(index, *, item, policy):
    fit_idx = item["fit_idx"]
    return fit_idx[len(fit_idx) // 2 :]

selection_policy = mf.window.custom_stage_policy(last_half_of_fit)
```

The selector receives:

```python
selector(index: pandas.Index, *, item: dict, policy: StagePolicy)
```

It may return a boolean mask, a slice, integer positions, or index labels.
The output must select at least one label. Use this for unusual validation or
selection-sample definitions; standard expanding, rolling, fixed-reference, and
origin-available designs should use `stage_policy(...)`.

## Forecast Combination

Use `custom_combination()` to append custom combined forecast rows after base
models have produced forecasts.

```python
def blend(forecasts, *, actual, weight=0.5):
    return weight * forecasts.iloc[:, 0] + (1.0 - weight) * forecasts.iloc[:, -1]

result = mf.forecasting.run(
    panel,
    {"ridge": "ridge", "lasso": "lasso"},
    window=window,
    features=features,
    combination=mf.forecasting.custom_combination(
        "ridge_lasso_blend",
        blend,
        models=["ridge", "lasso"],
        weight=0.25,
    ),
)
```

Callable signature:

```python
func(forecasts: pandas.DataFrame, *, actual: pandas.Series, **params)
```

`forecasts` is wide by base model and indexed by
`(date, origin, origin_pos, horizon)`. The output must be a `Series` or
one-dimensional array-like object aligned to those rows.

## Metadata Rules

Custom callables are stored by name, not serialized as code. Metadata records
the callable name, user parameters, selected columns, fit rows, and output
columns where available. For reproducible research, keep custom callable source
code in the project repository and version it with the run artifacts.

Custom extensions should remain stage-local:

| If you want to add... | Put it in... |
| --- | --- |
| New dataset loader or in-memory panel source | `macroforecast.data` via `custom_dataset()` or a project loader returning `DataBundle`. |
| New cleaning or preprocessing operation | `custom_preprocess()` or `custom_preprocess_step()`. |
| New feature transform | `custom_features()` or `custom_step()`. |
| New estimator | `custom_model()`. |
| New sample policy | `custom_stage_policy()`. |
| New forecast averaging or ensemble rule | `custom_combination()`. |
