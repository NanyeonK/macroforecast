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

The runner metadata records the window, each stage policy, preprocessing
options, feature-engineering options, selection spec, model specs, and origin
stage records.

## ForecastResult

```python
macroforecast.forecasting.ForecastResult(forecasts, metadata={})
```

| Attribute | Type | Meaning |
| --- | --- | --- |
| `forecasts` | pandas DataFrame | One row per emitted forecast. |
| `metadata` | dict | Window, preprocessing, feature, model, and selection metadata. |

Methods:

| Method | Output |
| --- | --- |
| `to_frame()` | Forecast table copy. |
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
