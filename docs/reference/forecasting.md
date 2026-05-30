# Forecasting

[Back to reference](index.md)

`macroforecast.forecasting` is the workflow composition layer. It connects
`window`, `preprocessing`, `feature_engineering`, `selection`, `models`, and
`evaluation`.

## run

```python
macroforecast.forecasting.run(
    data,
    model,
    *,
    window=None,
    preprocessing=None,
    preprocessing_policy=None,
    features=None,
    feature_policy=None,
    selection=None,
    selection_policy=None,
    selection_metric="mse",
    maximize_selection=False,
    preset=None,
    params=None,
    target=None,
    horizon=1,
    save_models=True,
    model_store="trained_model",
)
```

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `data` | `FeatureSet` or pandas-like panel | required | Prebuilt model matrix or raw canonical panel. |
| `model` | str, callable, `ModelSpec`, list, or mapping | required | One or more models to fit at each origin. |
| `window` | `WindowSpec`, str, or `None` | `None` | Forecast experiment design: estimation mode, validation, test origins, retrain and retune cadence. |
| `preprocessing` | `PreprocessSpec` or `None` | `None` | Callable preprocessing operations. |
| `preprocessing_policy` | `StagePolicy`, str, or `None` | `origin_available` when preprocessing is supplied | Where preprocessing may fit and apply: `full_panel`, `origin_available`, `fit_window`, or `fixed_reference`. |
| `features` | `FeatureSpec` or `None` | `None` | Feature and target construction operations. |
| `feature_policy` | `StagePolicy`, str, or `None` | `fit_window` | Where stateful feature engineering such as PCA may fit. |
| `selection` | `SearchSpec`, model-keyed mapping, or `None` | `None` | Hyperparameter candidate generation and search method. `None` uses each model's owned default search space; a model-keyed value of `None` disables selection for that model. |
| `selection_policy` | `StagePolicy`, str, or `None` | `fit_window` | Which feature rows are supplied to model selection. |
| `selection_metric` | str or callable | `"mse"` | Objective used during model selection. |
| `maximize_selection` | bool | `False` | Whether larger selection scores are better. |
| `preset` | str, mapping, or `None` | `None` | Model-owned search-space preset. |
| `params` | mapping or `None` | `None` | Fixed model parameters. |
| `target` | str or `None` | `None` | Required for raw panel input when `features` is omitted. |
| `horizon` | positive int | `1` | Target horizon when `features` is omitted. |
| `save_models` | bool | `True` | Save each fitted origin/model object and its metadata. |
| `model_store` | str or path-like | `"trained_model"` | Root directory for saved fitted models. |

Output: `ForecastResult`.

```python
pre = mf.preprocessing.preprocess_spec(
    transform="none",
    outliers="none",
    impute="mean",
    standardize="zscore",
    frame="keep",
)

features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors="all",
    lags=(0, 1, 2),
    pca_components=3,
)

window = mf.window.spec(
    estimation=mf.window.estimation_expanding(min_size=120),
    val=mf.window.val_last_block(size=24),
    test=mf.window.test_origins(horizon=1, step=1),
)

result = mf.forecasting.run(
    panel,
    ["ridge", "lasso"],
    window=window,
    preprocessing=pre,
    preprocessing_policy=mf.window.stage_policy("origin_available"),
    features=features,
    feature_policy=mf.window.stage_policy("fit_window"),
    selection=mf.selection.search_spec("ridge", preset="small"),
    selection_policy=mf.window.stage_policy("fit_window"),
    preset="small",
)
```

`selection=None` means “use registered model defaults” when a model has a
search space. To run fixed parameters with no tuning, pass a model-keyed
mapping such as `selection={"ridge": None}`. To evaluate a single explicit
candidate through the selection ledger, pass `selection={"ridge":
mf.selection.fixed({"alpha": 0.1})}`.

If the analysis intentionally follows the common full-sample empirical
workflow, preprocess first and pass the processed panel to `run(...,
preprocessing=None)`. If the analysis is a real-time forecasting exercise, pass
`preprocess_spec(...)` plus an explicit `preprocessing_policy`.

Stage policies are intentionally shared across preprocessing, feature
engineering, and selection:

| Scope | Meaning |
| --- | --- |
| `full_panel` | Fit the stage once on the full panel. Useful for retrospective replication designs. |
| `origin_available` | Fit using observations available up to each origin. This supports common macro cleaning designs, including EM imputation on variables observed by that origin. |
| `fit_window` | Fit only on the model fit window and apply that fitted state to validation/test rows. |
| `fixed_reference` | Fit on a named reference period, then keep that state fixed. Useful for fixed PCA loadings or fixed standardization windows. |

Each `StagePolicy` also has an `update` cadence. For preprocessing and feature
engineering, `run()` refits or reuses the fitted state according to
`"every_origin"`, `"on_retrain"`, `"never"`, a positive integer cadence, or a
pandas date offset such as `"12ME"`. This lets the same runner express both
full re-estimation designs and fixed-reference designs such as “fit PCA
loadings once, then keep the loadings fixed while origins roll forward.”

The runner metadata records the window, each stage policy, preprocessing
options, feature-engineering options, selection spec, model specs, and origin
stage records. Each origin stage record includes an `updated` flag showing
whether that stage fitted new state at that origin.

## Execution Order

For raw panel input, `run()` executes one test origin at a time:

| Step | Owner | What happens |
| --- | --- | --- |
| 1 | `meta` | Read package defaults such as random seed and default stage scopes. |
| 2 | `window` | Resolve estimation, validation, test, retrain, and retune rows. |
| 3 | `preprocessing` | Fit or reuse the preprocessing state according to `preprocessing_policy`; transform the rows needed by the origin. |
| 4 | `feature_engineering` | Fit or reuse the feature builder according to `feature_policy`; create `X_fit`, `y_fit`, `X_selection`, `y_selection`, `X_test`, and `y_test`. |
| 5 | `selection` | If enabled, evaluate model parameter candidates on validation splits supplied by the window. |
| 6 | `models` | Fit the model on the origin fit window with selected or fixed parameters. |
| 7 | `models` | Save the fitted object and JSON sidecar when `save_models=True`. |
| 8 | `forecasting` | Collect point, variance, and quantile forecasts plus the run metadata ledger. |

If `data` is already a `FeatureSet`, preprocessing and feature construction are
skipped. The runner slices the supplied `X` and `y` by the window plan and then
runs selection, model fitting, prediction, and optional model storage.

## Forecast Table

`ForecastResult.to_frame()` returns one row per `(origin, test date, model)`.

| Column | Meaning |
| --- | --- |
| `date` | Test target date. |
| `origin` | Forecast origin from the window row. |
| `origin_pos` | Integer position of the origin in the input index. |
| `horizon` | Forecast horizon for the row. |
| `model` | Runner alias, such as `ridge` or a key from a model mapping. |
| `model_spec` | Canonical registered model name. |
| `prediction` | Point forecast. |
| `variance_prediction` | Forecast variance when the fitted model exposes `predict_variance(...)`; otherwise `None`. |
| `quantile_predictions` | Per-row quantile dictionary when the fitted model exposes `predict_quantiles(X)`; otherwise `None`. |
| `actual` | Realized target value when available. |
| `params` | Parameters used for the origin fit after selection or fixed overrides. |
| `selection` | Selection ledger for the origin/model, including `retuned`. |
| `stored_model` | Model and metadata sidecar paths when `save_models=True`; otherwise `None`. |
| `window` | Full window row used for the origin. |
| `preprocessed` | Whether runner-level preprocessing was active. |

## Metadata Structure

`ForecastResult.metadata` is JSON-ready and contains:

| Key | Meaning |
| --- | --- |
| `run` | Forecast count, model count, meta config, model storage settings. |
| `data` | Input panel summary from `macroforecast.data.panel_info(...)`. |
| `window` | Complete `WindowSpec.to_dict()` output. |
| `stage_policies` | Resolved preprocessing, feature-engineering, and selection policies. |
| `preprocessing` | `PreprocessSpec.to_dict()` output, or `None`. |
| `features` | `FeatureSpec.to_dict()` output or supplied `FeatureSet` metadata. |
| `selection` | Search spec metadata, model-keyed search metadata, or `None`. |
| `models` | Runner aliases plus model spec metadata. |
| `stages` | Origin-level preprocessing and feature-engineering fit records unless `metadata_level="minimal"`. |

## Runner Examples

### Full-Sample Empirical Preprocessing

This pattern matches many retrospective empirical designs: clean the panel once,
then run the forecasting experiment on the processed panel.

```python
processed = mf.preprocessing.reprocess(
    panel,
    transform="official",
    outliers="iqr",
    outlier_action="flag_as_nan",
    impute="em_factor",
    standardize="zscore",
    frame="keep",
).panel

features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors="all",
    lags=(0, 1, 2),
)

result = mf.forecasting.run(
    processed,
    ["ridge", "lasso"],
    window=window,
    features=features,
    preprocessing=None,
    preset="small",
    save_models=False,
)
```

### Window-Local Preprocessing

This pattern is stricter for real-time forecasting. Preprocessing state is fit
inside each origin's fit window and then applied to validation/test rows.

```python
pre = mf.preprocessing.preprocess_spec(
    transform="official",
    outliers="iqr",
    impute="mean",
    standardize="zscore",
    standardize_columns="predictors",
    frame="keep",
)

result = mf.forecasting.run(
    panel,
    "ridge",
    window=window,
    preprocessing=pre,
    preprocessing_policy=mf.window.stage_policy("fit_window", update="every_origin"),
    features=features,
    feature_policy=mf.window.stage_policy("fit_window", update="on_retrain"),
)
```

### Fixed Standardization Reference

Use a fixed preprocessing reference when scaling or imputation state should come
from a named historical period rather than from every rolling origin.

```python
result = mf.forecasting.run(
    panel,
    "ridge",
    window=window,
    preprocessing=pre,
    preprocessing_policy=mf.window.stage_policy(
        "fixed_reference",
        reference_start="1985-01-31",
        reference_end="2004-12-31",
        update="never",
    ),
    features=features,
)
```

### Fixed PCA Loadings

Use a fixed feature reference when factor loadings should be estimated once and
held fixed while forecast origins move forward.

```python
pca_features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors="all",
    lags=(0, 1),
    pca_components=8,
)

result = mf.forecasting.run(
    panel,
    "ols",
    window=window,
    preprocessing=None,
    features=pca_features,
    feature_policy=mf.window.stage_policy(
        "fixed_reference",
        reference_start="1985-01-31",
        reference_end="2004-12-31",
        update="never",
    ),
)
```

When `feature_policy` is `full_panel` or `fixed_reference`, runner-level
preprocessing must be absent or `preprocessing_policy` must be `full_panel`.
This keeps fixed feature state fitted on one well-defined processed panel.

### Multiple Models with Model-Keyed Selection

`selection` can be one shared `SearchSpec` or a model-keyed mapping. A
model-keyed `None` disables selection for that model.

```python
result = mf.forecasting.run(
    panel,
    {"linear": "ridge", "sparse": "lasso", "tree": "random_forest"},
    window=window,
    features=features,
    selection={
        "linear": mf.selection.grid({"alpha": [0.01, 0.1, 1.0]}),
        "sparse": mf.selection.cv_path(
            param="alpha",
            values=[0.001, 0.01, 0.1, 1.0],
        ),
        "tree": mf.selection.random_search(
            {
                "n_estimators": mf.selection.randint(100, 500),
                "max_depth": mf.selection.choice([2, 3, 4, None]),
            },
            n_iter=12,
            random_state=123,
        ),
    },
    selection_policy=mf.window.stage_policy("fit_window"),
    selection_metric="mse",
)
```

### Quantile and Variance Outputs

Models that expose variance or quantile prediction methods fill extra forecast
columns automatically.

```python
quantile_result = mf.forecasting.run(
    panel,
    "quantile_regression_forest",
    window=window,
    features=features,
    params={
        "quantile_regression_forest": {
            "n_estimators": 200,
            "quantile_levels": (0.1, 0.5, 0.9),
            "random_state": 123,
        }
    },
    selection={"quantile_regression_forest": None},
)

quantile_result.to_frame()[["prediction", "quantile_predictions"]]
```

```python
variance_result = mf.forecasting.run(
    panel,
    "garch11",
    window=window,
    features=mf.feature_engineering.feature_spec(target="y", horizon=1),
    selection={"garch11": None},
)

variance_result.to_frame()[["prediction", "variance_prediction"]]
```

### Model Storage

Model storage is on by default. Use a custom root when the run should keep its
fitted objects separate from other experiments.

```python
stored = mf.forecasting.run(
    panel,
    ["ridge", "random_forest"],
    window=window,
    features=features,
    model_store="trained_model/monthly_baseline",
)

stored.to_frame()[["model", "stored_model"]]
```

Disable storage for fast exploratory runs:

```python
preview = mf.forecasting.run(
    panel,
    "ridge",
    window=window,
    features=features,
    save_models=False,
)
```

## Trained Model Storage

By default, `run()` saves the fitted model object produced at each forecast
origin. The runner decides which object to save: after selection, it refits the
model on the origin fit window with the selected best parameters, then delegates
the actual pickle and JSON write to `macroforecast.models.save_fit()`.

The default root is relative to the current working directory:

```text
trained_model/{model_name}/origin_{origin_pos}_h{horizon}_{origin}.pkl
trained_model/{model_name}/origin_{origin_pos}_h{horizon}_{origin}.json
```

Selection remains a runner responsibility because it depends on the window,
validation split, selection policy, and model-owned search space. Model
persistence remains a model-layer utility because it only knows how to save a
fitted object and a metadata sidecar.

The forecast table includes a `stored_model` dictionary for each row:

| Key | Meaning |
| --- | --- |
| `model_path` | Pickle path for the fitted model, or `None` when the object cannot be pickled. |
| `metadata_path` | JSON metadata path written for the fitted model. |
| `save_error` | `None` on success, otherwise the pickle error string. |

The sidecar JSON records the model alias, canonical model spec, fit metadata,
fit diagnostics, selected parameters, selection ledger, and window row used for
the fit. For custom/local callables that cannot be pickled, the runner still
writes the JSON sidecar and records `save_error`; forecasting continues.

To disable storage:

```python
result = mf.forecasting.run(
    panel,
    "ridge",
    window=window,
    features=features,
    save_models=False,
)
```

## ForecastResult

```python
macroforecast.forecasting.ForecastResult(forecasts, metadata={})
```

| Attribute | Type | Meaning |
| --- | --- | --- |
| `forecasts` | pandas DataFrame | One row per emitted forecast. |
| `metadata` | dict | Window, preprocessing, feature, model, and selection metadata. |

The forecast table always includes `prediction`. If the fitted model exposes
`predict_variance(horizon=...)`, the runner also fills
`variance_prediction`; otherwise that column is `None`. If the fitted model
exposes `predict_quantiles(X)`, the runner also fills
`quantile_predictions` with a per-row dictionary such as
`{"0.1": value, "0.5": value, "0.9": value}`; otherwise that column is
`None`.

Methods:

| Method | Output |
| --- | --- |
| `to_frame()` | Forecast table copy. |
| `evaluate(**kwargs)` | Calls `macroforecast.evaluation.evaluate_forecasts()` on this result. |
| `to_dict()` | JSON-ready dictionary. |
| `to_json(path=None)` | JSON text and optional file write. |

## Forecast Combination

Forecast combination lives in `macroforecast.forecasting` because it combines
forecast outputs, not model fits.

| Function | Meaning |
| --- | --- |
| `combine_mean(forecasts)` | Equal-weight average. |
| `combine_median(forecasts)` | Cross-model median. |
| `combine_trimmed_mean(forecasts, trim=0.1)` | Trim extremes before averaging. |
| `combine_winsorized_mean(forecasts, limits=(0.1, 0.1))` | Winsorize extremes before averaging. |
| `combine_inverse_mspe(forecasts, y_true, discount=1.0)` | Inverse discounted MSPE weights. |
| `combine_dmspe(forecasts, y_true, discount=1.0)` | Alias for inverse discounted MSPE. |
| `combine_best_n(forecasts, y_true, n=3)` | Average historically best `n` models. |
