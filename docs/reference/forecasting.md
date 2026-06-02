# macroforecast.forecasting

[Back to reference](index.md)

`macroforecast.forecasting` is the workflow composition module. It connects
`window`, `preprocessing`, `feature_engineering`, `model_selection`, `models`,
`model_ensemble`, and `metrics`/`tests`.

## run

`run_forecast` is an alias for `run`. New code can call `run(...)`; use
`run_forecast(...)` only when the longer name makes a script clearer.

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
    model_selection=None,
    model_selection_policy=None,
    model_selection_metric="mse",
    maximize_model_selection=False,
    preset=None,
    params=None,
    target=None,
    horizon=1,
    horizons=None,
    forecast_policy="direct",
    future_feature_policy=None,
    target_transform=None,
    combination=None,
    save_models=True,
    model_store="trained_model",
)
```

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `data` | `FeatureSet`, `DataBundle`, `DataSpec`, `(panel, metadata)`, or pandas-like panel | required | Prebuilt model matrix or canonical panel. `DataBundle` metadata, including native frequencies, is preserved. |
| `model` | str, callable, `ModelSpec`, list, or mapping | required | One or more model or fit-time model-ensemble specs to fit at each origin. |
| `window` | `WindowSpec`, str, or `None` | `None` | Forecast experiment design: estimation mode, validation, test origins, retrain and retune cadence. |
| `preprocessing` | `PreprocessSpec` or `None` | `None` | Callable preprocessing operations. |
| `preprocessing_policy` | `StagePolicy`, str, or `None` | `origin_available` when preprocessing is supplied | Where preprocessing may fit and apply: `full_panel`, `origin_available`, `fit_window`, or `fixed_reference`. |
| `features` | `FeatureSpec` or `None` | `None` | Feature and target construction operations. For panel-input models such as `dfm_mixed_mariano_murasawa`, leave this as `None`. |
| `feature_policy` | `StagePolicy`, str, or `None` | `fit_window` | Where stateful feature engineering such as PCA may fit. |
| `model_selection` | `SearchSpec`, model-keyed mapping, or `None` | `None` | Hyperparameter candidate generation and search method. `None` uses each model's owned default search space; a model-keyed value of `None` disables model selection for that model. |
| `model_selection_policy` | `StagePolicy`, str, or `None` | `fit_window` | Which feature rows are supplied to model selection. |
| `model_selection_metric` | str or callable | `"mse"` | Objective used during model selection. |
| `maximize_model_selection` | bool | `False` | Whether larger selection scores are better. |
| `preset` | str, mapping, or `None` | `None` | Model-owned search-space preset. |
| `params` | mapping or `None` | `None` | Fixed model parameters. |
| `target` | str or `None` | `None` | Required for raw panel input when `features` is omitted, and required for panel-input models unless every model spec sets the same `target` parameter. |
| `horizon` | positive int | `1` | Target horizon when `features` is omitted. |
| `horizons` | positive int, sequence, or `None` | `None` | Multiple target horizons. Provide either `horizon` or `horizons`, not both. |
| `forecast_policy` | str | `"direct"` | Target/forecast construction policy: `"direct"`, `"direct_average"`, `"path_average"`, `"recursive"`, or alias `"iterated"`. |
| `future_feature_policy` | str or `None` | `None` | Used only for recursive forecasting. `None` becomes `"target_lags"`. Use `"observed_future"` only for explicit oracle/scenario paths where future predictors are known or supplied. |
| `target_transform` | str or `None` | `None` | Optional target transform override. For `direct_average`, `"growth"` becomes `"average_growth"`; for `path_average`, `"growth"` means average step growth forecasts; for `recursive`, allowed values are `level`, `change`, `growth`, and `log_growth`. |
| `combination` | str, `CombinationSpec`, sequence, mapping, or `None` | `None` | Optional forecast-combination requests. Combined forecasts are appended as additional model rows. |
| `save_models` | bool | `True` | Save each fitted origin/model object and its metadata. |
| `model_store` | str or path-like | `"trained_model"` | Root directory for saved fitted models. |

Output: `ForecastResult`.

`ForecastResult` methods:

| Method | Input | Output | Meaning |
| --- | --- | --- | --- |
| `to_frame()` | none | `DataFrame` | Copy of forecast table. |
| `evaluate(**kwargs)` | arguments forwarded to `mf.metrics.evaluate_forecasts` | `DataFrame` | Forecast-score table. |
| `with_sidecar(name, value)` | sidecar name and runtime object | `ForecastResult` | Copy with a named sidecar recorded in metadata. |
| `get_sidecar(name, default=None)` | sidecar name | object | Return an attached sidecar. |
| `sidecar_names()` | none | tuple | Names of attached sidecars. |
| `with_oshapley(X, y, models, window=..., ...)` | explicit aligned oShapley/PBSV inputs | `ForecastResult` | Build and attach an oShapley/PBSV sidecar through `mf.interpretation.oshapley_from_forecast_result(...)`. |
| `with_anatomy(X, y, models, window=..., ...)` | explicit aligned backend inputs | `ForecastResult` | Backend alias for `with_oshapley(...)`. |
| `with_dual(model, X_train, y_train, X_test=None, ...)` | fitted model and explicit train/test design | `ForecastResult` | Build and attach a DualML observation-weight sidecar through `mf.interpretation.dual_from_forecast_result(...)`. |
| `anatomy_explain(anatomy, **kwargs)` | precomputed `anatomy.Anatomy` object or saved path | `DataFrame` | Convenience call to `mf.interpretation.anatomy_explain(...)`, with forecast-result metadata attached. |
| `pbsv(anatomy, **kwargs)` | precomputed backend object or saved path | `DataFrame` | Convenience call to `mf.interpretation.pbsv(...)`. |
| `oshapley_vi(anatomy, **kwargs)` | precomputed backend object or saved path | `DataFrame` | Convenience call to `mf.interpretation.oshapley_vi(...)`. |
| `anatomy_pbsv(anatomy, **kwargs)` | precomputed backend object or saved path | `DataFrame` | Backend alias for `pbsv(...)`. |
| `anatomy_oshapley_vi(anatomy, **kwargs)` | precomputed backend object or saved path | `DataFrame` | Backend alias for `oshapley_vi(...)`. |

`pbsv(...)` and `oshapley_vi(...)` do not create the backend object. Use
`with_oshapley(...)` or
`mf.interpretation.oshapley_from_forecast_result(...)` when the sidecar should
be built from explicit `X/y`, model specs, and `WindowSpec`.

Dual interpretation is also attached after the runner. `forecasting.run()`
does not infer observation weights automatically because the forecast table
does not contain the exact fitted estimator, training feature matrix, training
target, and forecast-row feature matrix. Use `with_dual(...)` or
`mf.interpretation.dual_from_forecast_result(...)` with those objects
explicitly.

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
    model_selection=mf.model_selection.search_spec("ridge", preset="small"),
    model_selection_policy=mf.window.stage_policy("fit_window"),
    preset="small",
)
```

`model_selection=None` means “use registered model defaults” when a model has a
search space. To run fixed parameters with no tuning, pass a model-keyed
mapping such as `model_selection={"ridge": None}`. To evaluate a single explicit
candidate through the model-selection ledger, pass `model_selection={"ridge":
mf.model_selection.fixed({"alpha": 0.1})}`.

### Forecast Policies

`forecast_policy` decides how the target is constructed and how forecasts are
written to the forecast table.

| Policy | Target construction | Model fit | Forecast row |
| --- | --- | --- | --- |
| `"direct"` | `y[t+h]` or a direct transform such as growth/change. | One model per requested horizon. | `date` is the target date `t+h`; `horizon` is `h`. |
| `"direct_average"` | Direct average target, such as average change or average growth over steps `1..h`. | One model per requested horizon. | `prediction` and `actual` are horizon-average objects. |
| `"path_average"` | Step-level targets for steps `1..h`. | One model per future step, then average the step forecasts. | `prediction` and `actual` are averages of the step forecasts/outcomes. |
| `"recursive"` / `"iterated"` | One-step target `y[t+1]`. | Fit one one-step model at each origin, then feed predicted target values back into the feature panel for steps `2..h`. | `prediction` and `actual` are expressed at the requested horizon `h` under the selected `target_transform`. |

Examples:

```python
# Fit separate direct models for h=1, 3, and 12.
direct = mf.forecasting.run(
    panel,
    "ridge",
    target="INDPRO",
    horizons=[1, 3, 12],
    model_selection={"ridge": None},
)

# Direct average growth target over the next 12 months.
direct_avg = mf.forecasting.run(
    panel,
    "ridge",
    target="INDPRO",
    horizon=12,
    forecast_policy="direct_average",
    target_transform="growth",
    model_selection={"ridge": None},
)

# Fit step 1..12 models and average the step growth forecasts.
path_avg = mf.forecasting.run(
    panel,
    "ridge",
    target="INDPRO",
    horizon=12,
    forecast_policy="path_average",
    target_transform="growth",
    model_selection={"ridge": None},
)

# Recursive / iterated forecast: fit one-step AR-style model and iterate to h=12.
recursive = mf.forecasting.run(
    panel,
    "ridge",
    target="INDPRO",
    horizon=12,
    forecast_policy="recursive",
    target_transform="level",
    model_selection={"ridge": None},
)
```

Recursive forecasting has an explicit future-feature contract:

| `future_feature_policy` | Meaning | Use when |
| --- | --- | --- |
| `"target_lags"` | Default. The runner builds or requires target-lag features only, then updates the target path with its own predictions. It does not invent future exogenous predictors. | Real-time recursive or iterated forecasting. |
| `"observed_future"` | The runner uses future non-target predictor values already present in the panel while recursively updating only the target. This is an oracle/scenario path. | Scenario analysis, controlled simulations, or cases where future predictor paths are genuinely known. |

When `features=None` and `forecast_policy="recursive"`, the runner creates
`feature_spec(target=target, predictors=[], lags=None, target_lags=(0, 1),
horizon=1)`. In row-date convention, `target_lags=(0, 1)` means the current
target value at the one-step information date and its previous value. For
supplied `FeatureSpec` with `future_feature_policy="target_lags"`, regular
`predictors` must be empty and `target_lags` must be declared with lag `0`
included, because predicted target values are written into the next row before
the next one-step forecast is formed. For supplied `FeatureSpec` with
`future_feature_policy="observed_future"`, regular predictors are allowed, but
the user is responsible for the future predictor path.

For feature-matrix models, the runner uses the requested horizon to restrict
test origins to dates where `t+h` is observable, but it fits/predicts one
origin row per forecast origin. This keeps `origin` as the information date and
`date` as the realized target date.

Panel-input models consume the canonical panel directly instead of an
engineered `X, y` matrix. They cannot be mixed with feature-matrix models in one
runner call. Currently `forecasting.run(..., features=None)` supports
`dfm_mixed_mariano_murasawa` and `dfm_unrestricted_midas` as native
mixed-frequency panel models. It keeps `DataBundle` metadata such as
`native_frequency_by_column` so the model can separate monthly and quarterly
columns inside each fit window. Panel-input runner calls currently fit fixed
model parameters; pass `model_selection={model_name: None}` for these models.

For standard MIDAS regressions, build the mixed-frequency lag matrix explicitly
and pass it as a `FeatureSet`. This keeps calendar anchoring in
`mixed_frequency_lags()` and model weighting in `midas_almon()`,
`midas_beta()`, `midas_step()`, or `unrestricted_midas()`:

```python
X_midas = mf.feature_engineering.mixed_frequency_lags(
    mixed,
    target="GDPC1",
    columns=["PAYEMS", "INDPRO"],
    lags=range(0, 12),
    target_frequency="quarterly",
    anchor_position="period_end",
    drop_missing=True,
)
y = mixed.panel["GDPC1"].reindex(X_midas.index).rename("GDPC1").to_frame()
features = mf.feature_engineering.FeatureSet(
    X=X_midas,
    y=y,
    metadata={"feature_engineering": {"method": "mixed_frequency_lags"}},
    target="GDPC1",
    targets=("GDPC1",),
    horizons=(1,),
    predictors=tuple(X_midas.columns),
)
result = mf.forecasting.run(
    features,
    "midas_beta",
    window=mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=40),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=1),
    ),
    params={"midas_beta": {"beta_params": (1.0, 2.0), "alpha": 0.1}},
    model_selection={"midas_beta": None},
)
```

If the analysis intentionally follows the common full-sample empirical
workflow, preprocess first and pass the processed panel to `run(...,
preprocessing=None)`. If the analysis is a real-time forecasting exercise, pass
`preprocess_spec(...)` plus an explicit `preprocessing_policy`.

Stage policies are intentionally shared across preprocessing, feature
engineering, and model selection:

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
options, feature-engineering options, model-selection spec, model specs, and origin
stage records. Each origin stage record includes an `updated` flag showing
whether that stage fitted new state at that origin.

Before any origin is fitted, `run()` validates the resolved `WindowSpec` against
the input index. Window validation errors, such as invalid inner validation
splits or `reuse_params=False` with skipped retune origins, stop the run with
`window validation failed: ...`.

## Execution Order

For raw panel input, `run()` executes one test origin at a time:

| Step | Owner | What happens |
| --- | --- | --- |
| 1 | `meta` | Read package defaults such as random seed and default stage scopes. |
| 2 | `window` | Resolve estimation, validation, test, retrain, and retune rows. |
| 3 | `preprocessing` | Fit or reuse the preprocessing state according to `preprocessing_policy`; transform the rows needed by the origin. |
| 4 | `feature_engineering` | Fit or reuse the feature builder according to `feature_policy`; create `X_fit`, `y_fit`, `X_selection`, `y_selection`, `X_test`, and `y_test`. |
| 5 | `model_selection` | If enabled, evaluate model parameter candidates on validation splits supplied by the window. |
| 6 | `models` or `model_ensemble` | Fit the model or fit-time ensemble on the origin fit window with selected or fixed parameters. |
| 7 | `models` | Save the fitted object and JSON sidecar when `save_models=True`. |
| 8 | `forecasting` | Collect point, variance, and quantile forecasts. |
| 9 | `forecasting` | Append requested forecast-combination rows and write the run metadata ledger. |

If `data` is already a `FeatureSet`, preprocessing and feature construction are
skipped. The runner slices the supplied `X` and `y` by the window plan and then
runs model selection, model fitting, prediction, and optional model storage.

If every selected model has `input_kind="panel"` and `features=None`, the runner
uses the window plan to slice the panel directly into fit and test panels. This
is the path used by mixed-frequency DFM models. Runner-level preprocessing can
run on this path with `preprocess_spec(...)` and `preprocessing_policy`; feature
engineering is skipped because panel-input models consume native panel columns.

The result metadata records which execution path was used:

| `run.input_path` | Input | Stages run inside `run()` |
| --- | --- | --- |
| `panel_to_features` | Raw canonical panel, `DataBundle`, `DataSpec`, or `(panel, metadata)` with feature-matrix models | Optional preprocessing, feature engineering, model selection, model fitting, optional combination. |
| `feature_set` | Prebuilt `FeatureSet` | Model selection, model fitting, optional combination. Preprocessing and feature construction are assumed to have already happened. |
| `panel_model` | Canonical panel with panel-input models such as mixed-frequency DFM | Optional preprocessing, panel slicing, panel-model fitting, optional combination. Feature engineering is skipped. |

## Forecast Table

`ForecastResult.to_frame()` returns one row per `(origin, test date, model)`.

| Column | Meaning |
| --- | --- |
| `date` | Test target date. |
| `origin` | Forecast origin from the window row. |
| `origin_pos` | Integer position of the origin in the input index. |
| `horizon` | Forecast horizon for the row. |
| `forecast_policy` | Policy used to construct the row: `direct`, `direct_average`, `path_average`, or `recursive`. |
| `target` | Source target variable. |
| `model` | Runner alias, such as `ridge` or a key from a model mapping. |
| `model_spec` | Canonical registered model name. |
| `prediction` | Point forecast. |
| `variance_prediction` | Forecast variance when the fitted model exposes `predict_variance(...)`; otherwise `None`. |
| `quantile_predictions` | Per-row quantile dictionary when the fitted model exposes `predict_quantiles(X)`; otherwise `None`. |
| `actual` | Realized target value when available. |
| `params` | Actual fixed-plus-selected parameters used for the origin fit. |
| `model_selection` | Model-selection ledger for the origin/model, including `retuned`. |
| `stored_model` | Model and metadata sidecar paths when `save_models=True`; otherwise `None`. |
| `window` | Full window row used for the origin. |
| `preprocessed` | `True` when runner-level preprocessing was active for the row; otherwise `False`. |
| `combined` | `True` for runner-created combination rows; `False` for base model rows. |
| `combination` | Combination spec dictionary for combined rows; otherwise `None`. |

## Metadata Structure

`ForecastResult.metadata` is JSON-ready and contains:

| Key | Meaning |
| --- | --- |
| `metadata_schema` | Stable metadata contract: `kind="forecast_result"`, schema version, execution path, forecast-table columns, and stage-record columns. |
| `run` | Forecast count, model count, execution path, forecast policy, horizons, meta config, metadata level, and model storage settings. |
| `data` | Input panel summary from `macroforecast.data.panel_info(...)`. |
| `window` | Complete `WindowSpec.to_dict()` output. |
| `stage_policies` | Resolved preprocessing, feature-engineering, and model-selection policies. |
| `preprocessing` | `PreprocessSpec.to_dict()` output, or `None`. |
| `forecast_policy` | Policy metadata including method, horizons, `future_feature_policy`, and whether observed future predictors were used. |
| `features` | `FeatureSpec.to_dict()` output or supplied `FeatureSet` metadata. |
| `model_selection` | Search spec metadata, model-keyed search metadata, or `None`. |
| `combination` | List of resolved forecast-combination specs. |
| `models` | Runner aliases plus model spec metadata. |
| `stages` | Origin-level preprocessing and feature-engineering fit records unless `metadata_level="minimal"`. |

`metadata_schema.version` is currently `1`. Code that consumes runner outputs
should check this value before assuming column or metadata shape. The
`forecast_table_columns` entry lists the stable row fields emitted by the
runner. `stage_record_columns` lists the ledger fields used when a stateful
stage is fitted or reused at an origin.

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

### Multiple Models with Model-Keyed Model Selection

`model_selection` can be one shared `SearchSpec` or a model-keyed mapping. A
model-keyed `None` disables model selection for that model. Mapping keys can be
the output alias, such as `linear`, or the registered spec name, such as `ridge`.

```python
result = mf.forecasting.run(
    panel,
    {"linear": "ridge", "sparse": "lasso", "tree": "random_forest"},
    window=window,
    features=features,
    model_selection={
        "linear": mf.model_selection.grid({"alpha": [0.01, 0.1, 1.0]}),
        "sparse": mf.model_selection.cv_path(
            param="alpha",
            values=[0.001, 0.01, 0.1, 1.0],
        ),
        "tree": mf.model_selection.random_search(
            {
                "n_estimators": mf.model_selection.randint(100, 500),
                "max_depth": mf.model_selection.choice([2, 3, 4, None]),
            },
            n_iter=12,
            random_state=123,
        ),
    },
    model_selection_policy=mf.window.stage_policy("fit_window"),
    model_selection_metric="mse",
)
```

`params` and `preset` follow the same alias-or-spec-key rule. Unknown keys raise
an error instead of being silently ignored. For a single model, direct parameter
names are also accepted, including dict-valued parameters such as
`params={"base_params": {"alpha": 0.1}}` for a fit-time ensemble spec.

Forecast rows record the actual fixed-plus-selected parameter set in the
`params` column. For example, if `params={"ridge": {"fit_intercept": False}}`
and model selection picks `{"alpha": 0.1}`, the row records both values.

### Forecast Combination In The Runner

`combination` asks the runner to append combined forecasts after all base model
forecasts have been collected. Combination rows use the same `date`, `origin`,
`origin_pos`, `horizon`, `actual`, and `window` fields as the base rows, with
`model` set to the combination name.

The `models=` filter inside a combination spec refers to the output `model`
column, not the registry `model_spec`. If `model={"bagged": "bagging"}`, use
`models=["bagged"]` when selecting that fit-time ensemble for a forecast
combination.

```python
result = mf.forecasting.run(
    panel,
    ["ridge", "lasso", "random_forest"],
    window=window,
    features=features,
    preset="small",
    combination="mean",
)

result.to_frame().query("model == 'combined_mean'")
```

Multiple combinations can be requested together:

```python
result = mf.forecasting.run(
    panel,
    {"linear": "ridge", "sparse": "lasso", "tree": "random_forest"},
    window=window,
    features=features,
    combination={
        "avg": "mean",
        "dmspe": {
            "method": "dmspe",
            "models": ["linear", "sparse", "tree"],
            "discount": 0.95,
        },
        "best_two": {
            "method": "best_n",
            "n": 2,
        },
    },
)
```

`inverse_mspe`, `dmspe`, and `best_n` use only historical forecast errors when
forming the current combined forecast. The current row's realized value is used
only after that row's weight or best-model decision has already been made.

Custom forecast combinations use the same runner hook:

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

The callable receives a wide forecast matrix indexed by
`(date, origin, origin_pos, horizon)` and an `actual` series aligned to the same
index:

```python
func(forecasts: pandas.DataFrame, *, actual: pandas.Series, **params)
```

It must return a `Series` or one-dimensional array-like object with the same
length. The runner appends the output as rows with `combined=True` and records
the callable name in `metadata["combination"]`.

### Mixed-Frequency DFM In The Runner

Use the panel-input path for native monthly/quarterly state-space models. The
input should be a `DataBundle` whose metadata records column-level native
frequencies.

```python
mixed = mf.data.combine(monthly_bundle, quarterly_bundle, frequency="native")

window = mf.window.spec(
    estimation=mf.window.estimation_expanding(min_size=120),
    val=mf.window.val_last_block(size=24),
    test=mf.window.test_origins(horizon=1, step=3),
)

result = mf.forecasting.run(
    mixed,
    "dfm_mixed_mariano_murasawa",
    window=window,
    target="GDPC1",
    params={
        "dfm_mixed_mariano_murasawa": {
            "n_factors": 1,
            "factor_order": 1,
        }
    },
    model_selection={"dfm_mixed_mariano_murasawa": None},
    features=None,
)
```

The result metadata records `run.panel_model_runner=True` and keeps
`data.native_frequency_counts`, `data.output_frequency_counts`, and
`data.frequency="mixed"` when those fields are present on the input bundle.
The requested `target` must be present in the panel before fitting begins; a
missing target raises a direct error before any model backend is called.

For all runner paths, fitted model `predict(X_test)` output is validated before
records are appended. Array-like predictions are positional. A pandas `Series`
or single-column `DataFrame` must either use `X_test.index` or the default
`RangeIndex(len(X_test))`; any other index is rejected to avoid silently
writing NaN forecasts. `predict_quantiles(X_test)` follows the same DataFrame
index rule when it returns a DataFrame.

### Quantile and Variance Outputs

Models that expose variance or quantile prediction methods fill extra forecast
columns automatically. The runner first tries `predict_variance(X_test)` for
models such as `hemisphere_nn`, then falls back to
`predict_variance(horizon=len(X_test))` for volatility models.

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
    model_selection={"quantile_regression_forest": None},
)

quantile_result.to_frame()[["prediction", "quantile_predictions"]]
```

```python
variance_result = mf.forecasting.run(
    panel,
    "garch11",
    window=window,
    features=mf.feature_engineering.feature_spec(target="y", horizon=1),
    model_selection={"garch11": None},
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
origin. The runner decides which object to save: after model selection, it refits the
model on the origin fit window with the selected best parameters, then delegates
the actual pickle and JSON write to `macroforecast.models.save_fit()`.

The default root is relative to the current working directory:

```text
trained_model/{model_name}/origin_{origin_pos}_h{horizon}_{origin}.pkl
trained_model/{model_name}/origin_{origin_pos}_h{horizon}_{origin}.json
```

Model selection remains a runner responsibility because it depends on the window,
validation split, model-selection policy, and model-owned search space. Model
persistence remains a model-module utility because it only knows how to save a
fitted object and a metadata sidecar.

The forecast table includes a `stored_model` dictionary for each row:

| Key | Meaning |
| --- | --- |
| `model_path` | Pickle path for the fitted model, or `None` when the object cannot be pickled. |
| `metadata_path` | JSON metadata path written for the fitted model. |
| `save_error` | `None` on success, otherwise the pickle error string. |

The sidecar JSON records the model alias, canonical model spec, fit metadata,
fit diagnostics, selected parameters, model-selection ledger, and window row used for
the fit. For custom/local callables that cannot be pickled, the runner still
writes the JSON sidecar and records `save_error`; forecasting continues.

When a `ForecastResult` is passed to `mf.output.write_artifacts(...)`, the
artifact manifest also records the `stored_model` pickle and sidecar paths. The
output writer does not copy those model files; it links the already-written
runner artifacts into the manifest with `stored_model_pickle` and
`stored_model_metadata` records.

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

## Benchmark Forecasts

Benchmark-relative metrics are evaluated after `run()`, but the benchmark
forecast should normally be generated at the forecasting stage. Include the
benchmark model in the same runner call so it uses the same preprocessing,
feature policy, window, validation split, forecast origin, horizon, and target
support as the candidate models.

```python
result = mf.forecasting.run(
    panel,
    ["ridge", "ols"],
    window=window,
    features=features,
)

scores = result.evaluate(
    metrics=("mse", "relative_mse", "r2_oos"),
    benchmark_model="ols",
)
```

External benchmark forecasts are also allowed when they come from a published
system or an existing CSV. Append those rows to the forecast table first, using
the same `model`, `date` or origin, `horizon`, `target`, `actual`, and
`prediction` contract. `macroforecast.metrics.evaluate_forecasts()` then
checks that candidate and benchmark supports match before computing relative
metrics.

## ForecastResult

```python
macroforecast.forecasting.ForecastResult(forecasts, metadata={})
```

| Attribute | Type | Meaning |
| --- | --- | --- |
| `forecasts` | pandas DataFrame | One row per emitted forecast. |
| `metadata` | dict | Window, preprocessing, feature, model, and model-selection metadata. |
| `sidecars` | dict | Runtime objects attached after forecasting, such as a `ForecastShapleyResult`. |

The forecast table always includes `prediction`. If the fitted model exposes
`predict_variance(X_test)` or `predict_variance(horizon=...)`, the runner also
fills `variance_prediction`; otherwise that column is `None`. If the fitted model
exposes `predict_quantiles(X)`, the runner also fills
`quantile_predictions` with a per-row dictionary such as
`{"0.1": value, "0.5": value, "0.9": value}`; otherwise that column is
`None`.

Methods:

| Method | Output |
| --- | --- |
| `to_frame()` | Forecast table copy. |
| `evaluate(**kwargs)` | Calls `macroforecast.metrics.evaluate_forecasts()` on this result. |
| `with_sidecar(name, value)` | Returns a copy with a named runtime sidecar. |
| `with_oshapley(X, y, models, window=..., **kwargs)` | Builds and attaches an oShapley/PBSV sidecar. |
| `with_anatomy(X, y, models, window=..., **kwargs)` | Backend alias for `with_oshapley(...)`. |
| `with_dual(model, X_train, y_train, X_test=None, **kwargs)` | Builds and attaches a dual interpretation sidecar. |
| `get_sidecar(name, default=None)` | Retrieves one sidecar. |
| `sidecar_names()` | Lists attached sidecars. |
| `to_dict()` | JSON-ready dictionary. |
| `to_json(path=None)` | JSON text and optional file write. |

## Direct Forecast Combination Functions

Forecast combination lives in `macroforecast.forecasting` because it combines
forecast outputs, not model fits. Fit-time member-model composition lives in
`macroforecast.model_ensemble`.

| Function | Meaning |
| --- | --- |
| `combine_mean(forecasts)` | Equal-weight average. |
| `combine_median(forecasts)` | Cross-model median. |
| `combine_trimmed_mean(forecasts, trim=0.1)` | Trim extremes before averaging. |
| `combine_winsorized_mean(forecasts, limits=(0.1, 0.1))` | Winsorize extremes before averaging. |
| `combine_inverse_mspe(forecasts, y_true, discount=1.0)` | Inverse discounted MSPE weights. |
| `combine_dmspe(forecasts, y_true, discount=1.0)` | Alias for inverse discounted MSPE. |
| `combine_best_n(forecasts, y_true, n=3)` | Average historically best `n` models. |
| `combination_spec(method, name=None, models=None, **params)` | Build a reusable runner combination spec. |
| `custom_combination(name, func, models=None, **params)` | Build a runner combination spec from a user callable. |
