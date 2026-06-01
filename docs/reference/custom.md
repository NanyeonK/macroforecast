# Custom Extensions

[Back to reference](index.md)

`macroforecast` keeps custom extension points as normal callable functions, not
as a second registry layer. The user can add a dataset, preprocessing function,
feature transform, model, fit-time model ensemble, stage policy, or
forecast-combination rule by passing a Python callable at the stage that owns
that behavior.

The package still enforces the same contracts around each custom hook:

| Stage | Extension function | Input contract | Output contract |
| --- | --- | --- | --- |
| Data | `mf.data.custom_dataset(...)` | In-memory `pandas.DataFrame` plus date/column/frequency metadata. | `DataBundle` with canonical panel and metadata. |
| Preprocessing | `mf.preprocessing.custom_preprocess(...)` | Canonical panel plus metadata. | `PreprocessedData`. |
| Preprocessing spec | `mf.preprocessing.custom_preprocess_step(...)` | Callable step for `preprocess_spec(custom_steps=[...])`. | Step dictionary consumed by `PreprocessSpec`. |
| Feature engineering | `mf.feature_engineering.custom_features(...)` | Selected feature panel. | Numeric feature `DataFrame`. |
| Feature spec | `mf.feature_engineering.custom_step(...)` | Stateless or fitted feature callable for `feature_spec(steps=[...])`. | Step dictionary consumed by `FeatureSpec`. |
| Models | `mf.models.custom_model(...)` | Fit callable with model metadata and optional search spaces. | `ModelSpec`. |
| Model ensemble | `mf.model_ensemble.custom_model_ensemble(...)` | Fit-time composition callable returning a model-like fit. | `ModelSpec` with `family="model_ensemble"`. |
| Window | `mf.window.custom_stage_policy(...)` | Selector callable for origin-specific sample labels. | `StagePolicy(scope="custom")`. |
| Selection | `mf.model_selection.custom_search(...)` | User search callable over model, data, splits, metric, and candidate evaluation helper. | `SearchSpec(method="custom")`. |
| Forecasting | `mf.forecasting.custom_combination(...)` | Callable over base forecast matrix. | `CombinationSpec`. |
| Metrics | metric callable | Custom `(y_true, y_pred) -> float` scorer. | Metric column in score tables. |
| Tests | `mf.tests.custom_test(...)` | User forecast-test callable. | `TestResult`. |
| Evaluation | callable metrics and custom groupings | User metric functions and grouping tuples. | `EvaluationReport` tables. |
| Interpretation | `mf.interpretation.custom_interpretation(...)` | Fitted model, feature frame, optional target, and user callable. | Interpretation `DataFrame` with macroforecast schema attrs. |
| Feature diagnostic | `mf.feature_analysis.custom_feature_diagnostic(...)` | Feature matrix or `FeatureSet` plus user diagnostic callable. | Diagnostic `DataFrame` with stage metadata. |
| Forecast diagnostic | `mf.forecast_analysis.custom_forecast_diagnostic(...)` | Forecast table or `ForecastResult` plus user diagnostic callable. | Diagnostic `DataFrame` with stage metadata. |
| Output | `mf.output.write_artifacts({...})` | Mapping of named custom artifacts. | Files plus `ArtifactManifest`. |

## End-To-End Flow

The full custom path is still one ordinary callable workflow. Each custom
piece stays at the module that owns the behavior:

```python
bundle = mf.data.custom_dataset(
    frame,
    date="date",
    dataset="local_panel",
    frequency="monthly",
    transform_codes={"target": 1, "x": 1, "z": 1},
)

pre = mf.preprocessing.preprocess_spec(
    transform="none",
    outliers="none",
    impute="none",
    standardize="none",
    custom_steps=[
        mf.preprocessing.custom_preprocess_step("spread", add_spread, scale=1.0),
    ],
)

features = mf.feature_engineering.feature_spec(
    target="target",
    horizon=1,
    predictors=["x", "z", "spread"],
    lags=(0,),
    steps=[
        mf.feature_engineering.custom_step(
            "x_square",
            square_feature,
            columns=["x"],
        ),
    ],
)

model = mf.models.custom_model(
    "mean_tuned",
    mean_model,
    default_params={"offset": 0.0},
)

search = mf.model_selection.custom_search(
    "ordered_offset",
    ordered_offset_search,
    values=(-0.1, 0.0, 0.1),
)

result = mf.forecasting.run(
    bundle,
    {"ols": "ols", "mean_tuned": model},
    window=window,
    preprocessing=pre,
    features=features,
    model_selection={"ols": None, "mean_tuned": search},
    model_selection_policy=mf.window.custom_stage_policy(last_fit_half),
    combination=mf.forecasting.custom_combination("blend", blend),
)
```

After the runner returns, custom scoring, testing, interpretation, diagnostics,
and output are separate callable steps:

```python
scores = result.evaluate(metrics=("mse", mean_bias))
test = mf.tests.custom_test("bias_test", bias_test, loss_a, loss_b)

interpretation = mf.interpretation.custom_interpretation(
    fit,
    X_test,
    local_interpretation,
    name="local_interpretation",
)

feature_diag = mf.feature_analysis.custom_feature_diagnostic(
    feature_set,
    feature_check,
    name="feature_check",
)

forecast_diag = mf.forecast_analysis.custom_forecast_diagnostic(
    result,
    forecast_check,
    name="forecast_check",
)

manifest = mf.output.write_artifacts(
    {
        "forecast_result": result,
        "scores": scores,
        "custom_test": test.to_dict(),
        "custom_interpretation": interpretation,
        "custom_feature_diagnostic": feature_diag,
        "custom_forecast_diagnostic": forecast_diag,
    },
    "results/custom_flow",
)
```

This flow is executed by the package test
`tests/test_custom_extensions.py::test_custom_extension_flow_runs_from_data_to_output`.

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

model_selection_policy = mf.window.custom_stage_policy(last_half_of_fit)
```

The selector receives:

```python
selector(index: pandas.Index, *, item: dict, policy: StagePolicy)
```

It may return a boolean mask, a slice, integer positions, or index labels.
The output must select at least one label. Use this for unusual validation or
model-selection-sample definitions; standard expanding, rolling, fixed-reference, and
origin-available designs should use `stage_policy(...)`.

## Model Selection

Use `custom_search()` when the hyperparameter search algorithm itself is
project-specific.

```python
def ordered_alpha_search(
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
    ordered_alpha_search,
    values=(0.01, 0.1, 1.0),
)
```

Callable signature:

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

The callable returns trial records, a trial `DataFrame`, a `SearchResult`, or
`(records, metadata)`. Use `evaluate_candidate` unless the custom algorithm has
to fit models itself. Custom metrics do not need `custom_search()`; pass them
directly as `select_params(..., metric=my_metric)`.

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

## Metrics, Tests, And Evaluation

Custom point metrics are plain callables:

```python
def mean_bias(y_true, y_pred):
    return float(pd.Series(y_pred).sub(pd.Series(y_true)).mean())

scores = mf.metrics.evaluate_forecasts(
    forecast_table,
    metrics=("mse", mean_bias),
)
```

The callable must return one scalar `float`. Its output column uses the
callable name.

Custom statistical tests use `mf.tests.custom_test(...)`:

```python
def my_loss_test(loss_a, loss_b):
    diff = pd.Series(loss_a).sub(pd.Series(loss_b)).dropna()
    return {
        "statistic": float(diff.mean()),
        "p_value": 0.04,
        "n_obs": len(diff),
    }

test = mf.tests.custom_test("my_loss_test", my_loss_test, loss_a, loss_b)
```

For multi-slice evaluation, pass custom metrics and custom grouping maps:

```python
report = mf.evaluation.evaluate_report(
    forecast_result,
    metrics=("mse", mean_bias),
    aggregations={
        "model_target": ("model", "target"),
        "model_regime": ("model", "regime"),
    },
)
```

## Interpretation And Diagnostics

Custom interpretation is for one fitted model and one feature matrix:

```python
def coefficient_ratio(model, X, *, y=None, metadata=None, denominator="x1"):
    coef = mf.interpretation.linear_coefficients(model)
    base = float(coef.loc[coef["feature"] == denominator, "coefficient"].iloc[0])
    out = coef.copy()
    out["ratio_to_base"] = out["coefficient"] / base
    return out

table = mf.interpretation.custom_interpretation(
    fit,
    X_test,
    coefficient_ratio,
    name="coefficient_ratio",
    denominator="PAYEMS_lag0",
)
```

Callable signature:

```python
func(model, X, *, y=None, metadata=None, **params)
```

Accepted return types are `DataFrame`, `Series`, mapping, or a sequence that
can be converted to a `DataFrame`. The returned table receives
`attrs["macroforecast_metadata_schema"]["kind"] == "custom_interpretation"`.

Custom feature diagnostics inspect constructed `X` without creating new
features:

```python
def block_missingness(X, *, feature_metadata=None, metadata=None, block="all"):
    return pd.DataFrame(
        [{"block": block, "missing_rate": float(X.isna().mean().mean())}]
    )

diag = mf.feature_analysis.custom_feature_diagnostic(
    features,
    block_missingness,
    name="block_missingness",
    block="rates",
)
```

Callable signature:

```python
func(X, *, feature_metadata=None, metadata=None, **params)
```

Custom forecast diagnostics inspect runner output:

```python
def horizon_bias(forecasts, *, metadata=None):
    out = forecasts.copy()
    out["residual"] = out["actual"] - out["prediction"]
    return out.groupby("horizon", as_index=False)["residual"].mean()

diag = mf.forecast_analysis.custom_forecast_diagnostic(
    result,
    horizon_bias,
    name="horizon_bias",
)
```

Callable signature:

```python
func(forecasts, *, metadata=None, **params)
```

The custom diagnostic wrappers do not refit models or rebuild features. They
only coerce inputs, run the user callable, and attach metadata so the output can
be exported and audited with the rest of the study.

## Output Artifacts

Use `write_artifacts()` for custom outputs that do not need a stage-specific
wrapper:

```python
mf.output.write_artifacts(
    {
        "forecast_result": result,
        "custom_interpretation": table,
        "custom_notes": {"spec": "local robustness check", "accepted": True},
    },
    "results/my_run",
)
```

`DataFrame` artifacts keep their `attrs` in JSON payloads and in the manifest
record. Mapping/list/scalar artifacts are written as JSON and receive object
metadata in the manifest.

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
| New fit-time model ensemble | `custom_model_ensemble()`. |
| New sample policy | `custom_stage_policy()`. |
| New hyperparameter-search algorithm | `custom_search()`. |
| New forecast-output averaging or combination rule | `custom_combination()`. |
| New scalar forecast metric | Pass a callable to `metrics=...`. |
| New forecast-comparison test | `mf.tests.custom_test()`. |
| New evaluation slice | Pass a grouping map to `evaluate_report(..., aggregations=...)`. |
| New model-interpretation table | `mf.interpretation.custom_interpretation()`. |
| New feature diagnostic table | `mf.feature_analysis.custom_feature_diagnostic()`. |
| New forecast diagnostic table | `mf.forecast_analysis.custom_forecast_diagnostic()`. |
| New saved artifact | Add it as a named mapping entry in `mf.output.write_artifacts(...)`. |
