# Custom Flow

[Back to custom extensions](index.md)

Custom extensions are stage-local. Put the callable where the package already
expects that kind of object, then return the object shape that the next stage
needs. Do not build a parallel registry.

## Minimal End-To-End Example

```python
import macroforecast as mf

bundle = mf.data.custom_dataset(
    frame,
    date="date",
    dataset="local_panel",
    frequency="monthly",
    transform_codes={"target": 1, "x": 1, "z": 1},
)

preprocessing = mf.preprocessing.preprocess_spec(
    transform="none",
    outliers="none",
    impute="none",
    standardize="none",
    custom_steps=[
        mf.preprocessing.custom_preprocess_step(
            "spread",
            add_spread,
            scale=100.0,
        ),
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
    search_spaces={"small": {"offset": (-0.1, 0.0, 0.1)}},
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
    preprocessing=preprocessing,
    features=features,
    model_selection={"ols": None, "mean_tuned": search},
    model_selection_policy=mf.window.custom_stage_policy(last_fit_half),
    combination=mf.forecasting.custom_combination("blend", blend, weight=0.5),
)
```

## Stage Input And Output

| Stage | Input from previous stage | Custom hook output | Next consumer |
| --- | --- | --- | --- |
| Data | User `DataFrame` or file | `DataBundle(panel, metadata)` | preprocessing, feature engineering, or runner |
| Preprocessing | `DataBundle`, `DataSpec`, `PreprocessedData`, or panel | `PreprocessedData` or step dictionary | feature engineering or runner |
| Feature engineering | processed panel | `FeatureSet`, feature `DataFrame`, or step dictionary | models or runner |
| Model | train `X`, train `y` | fitted object with `predict(X_test)` | runner prediction loop |
| Selection | model, train data, splits, metric | `SearchSpec` and selected params | runner fit loop |
| Forecasting | base forecast matrix | combined prediction rows | `ForecastResult` |
| Evaluation/tests | forecast table, losses, or arrays | score table or `TestResult` | reporting/output |
| Interpretation/analysis | fitted model, feature matrix, forecast table | schema-tagged `DataFrame` | output/reporting |
| Output | named objects | files and manifest records | replication package or paper appendix |

## Metadata Rule

Custom callables are stored by name, not serialized as source code. Package
metadata records callable names, parameters, selected columns, stage names, and
output schemas where available. Keep the callable source in the project
repository when a result must be reproducible.

## Namespace Rule

Use namespace calls:

```python
mf.data.custom_dataset(...)
mf.models.custom_model(...)
mf.output.write_artifacts(...)
```

Do not rely on root shortcuts such as `mf.custom_model` or
`mf.write_artifacts`.
