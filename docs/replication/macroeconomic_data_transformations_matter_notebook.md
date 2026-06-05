# Macroeconomic Data Transformations Matter: Package-Only Notebook

This notebook-style report shows how to run the Goulet Coulombe, Leroux,
Stevanovic, and Surprenant (2021) design with public `macroforecast` callables.
It is written as code block, output block, code block, output block, so the same
file can serve as an execution log.

Scope:

- Data, preprocessing, feature construction, windowing, model fitting,
  forecasting, and evaluation are all handled by `macroforecast`.
- The display code uses only objects returned by `macroforecast`.
- The paper does not expose a full machine-readable replication package or an
  exact FRED-MD vintage in the checked materials, so this is a reconstructed
  package notebook, not a table-identical replication claim.
- The quick outputs below use a short sanity window and a 20-tree random forest.
  The full-paper run cell keeps the paper-style 200-tree setting and full
  `1980-01` to `2017-12` test calendar.
- Tuned learners use the paper's randomly assigned 5-fold CV through
  `mf.window.val_random_kfold(...)` / `val_method="random_kfold"`. Random folds
  are used here for replication, not as the default time-safe validation design.

## Cell 0: Import the package

```python
import macroforecast as mf

print("macroforecast_version", mf.__version__)
```

Output:

```text
macroforecast_version 0.9.5a1
```

## Cell 1: Load the frozen FRED-MD vintage

```python
cache_root = "/private/tmp/macroforecast_fred_cache"
bundle = mf.data.load_fred_md(vintage="2018-01", cache_root=cache_root)

print("raw_type", type(bundle.panel).__name__)
print("raw_shape", bundle.panel.shape)
print("raw_index_start", bundle.panel.index.min())
print("raw_index_end", bundle.panel.index.max())
print("metadata_keys", sorted(list(bundle.metadata.keys()))[:20])
```

Output:

```text
raw_type DataFrame
raw_shape (708, 127)
raw_index_start 1959-01-01 00:00:00
raw_index_end 2017-12-01 00:00:00
metadata_keys ['artifact', 'data_through', 'dataset', 'frequency', 'panel', 'parse_notes', 'support_tier', 'transform_codes', 'version_mode', 'vintage']
```

## Cell 2: Apply the official FRED-MD preprocessing

```python
processed = mf.preprocessing.reprocess(bundle)

print("processed_type", type(processed.panel).__name__)
print("processed_shape", processed.panel.shape)
print("processed_index_start", processed.panel.index.min())
print("processed_index_end", processed.panel.index.max())
print("processed_missing_total", int(processed.panel.isna().sum().sum()))
print("sample_columns", list(processed.panel.columns[:10]))
```

Output:

```text
processed_type DataFrame
processed_shape (706, 127)
processed_index_start 1959-03-01 00:00:00
processed_index_end 2017-12-01 00:00:00
processed_missing_total 0
sample_columns ['RPI', 'W875RX1', 'DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD']
```

Interpretation:

- `load_fred_md(vintage="2018-01")` returns the raw monthly vintage panel.
- `reprocess(bundle)` applies the package default official FRED-MD t-code
  pipeline and returns a new bundle with `processed.panel`.
- The first two months drop because official transformations and t-code lagging
  need earlier observations.

## Cell 3: Verify the target map and grid size

```python
TARGET_MAP = {
    "INDPRO": "INDPRO",
    "EMP": "PAYEMS",
    "UNRATE": "UNRATE",
    "INCOME": "W875RX1",
    "CONS": "DPCERA3M086SBEA",
    "RETAIL": "RETAILx",
    "HOUST": "HOUST",
    "M2": "M2SL",
    "CPI": "CPIAUCSL",
    "PPI": "PPICMM",
}

horizons = [1, 3, 6, 9, 12, 24]
feature_cases = [
    "F", "F-X", "F-MARX", "F-MAF", "F-Level",
    "F-X-MARX", "F-X-MAF", "F-X-Level", "F-X-MARX-Level",
    "X", "MARX", "MAF", "X-MARX", "X-MAF", "X-Level", "X-MARX-Level",
]
ml_models = [
    "adaptive_lasso", "elastic_net", "glmboost",
    "random_forest", "gradient_boosting",
]

print("target_alias column available")
for alias, column in TARGET_MAP.items():
    print(f"{alias:7s} {column:16s} {column in processed.panel.columns}")

print("")
print("grid_axes")
print("targets", len(TARGET_MAP))
print("horizons", len(horizons))
print("feature_cases", len(feature_cases))
print("ml_models", len(ml_models))
print("target_policies", 2)
print("ml_forecast_runs", len(TARGET_MAP) * len(horizons) * len(feature_cases) * len(ml_models) * 2)
print("benchmarks", ["ar", "far"])
```

Output:

```text
target_alias column available
INDPRO  INDPRO           True
EMP     PAYEMS           True
UNRATE  UNRATE           True
INCOME  W875RX1          True
CONS    DPCERA3M086SBEA  True
RETAIL  RETAILx          True
HOUST   HOUST            True
M2      M2SL             True
CPI     CPIAUCSL         True
PPI     PPICMM           True

grid_axes
targets 10
horizons 6
feature_cases 16
ml_models 5
target_policies 2
ml_forecast_runs 9600
benchmarks ['ar', 'far']
```

## Cell 4: Build the panel used by level specifications

```python
replication_panel = processed.panel.join(
    bundle.panel.reindex(processed.panel.index).add_prefix("LEVEL__")
)

TARGET = "INDPRO"
transformed_predictors = [column for column in processed.panel.columns if column != TARGET]
level_predictors = [f"LEVEL__{column}" for column in bundle.panel.columns]

print("replication_panel_shape", replication_panel.shape)
print("transformed_predictor_count", len(transformed_predictors))
print("level_predictor_count", len(level_predictors))
print("first_level_columns", level_predictors[:5])
```

Output:

```text
replication_panel_shape (706, 254)
transformed_predictor_count 126
level_predictor_count 127
first_level_columns ['LEVEL__RPI', 'LEVEL__W875RX1', 'LEVEL__DPCERA3M086SBEA', 'LEVEL__CMRMTSPLx', 'LEVEL__RETAILx']
```

Interpretation:

- The target remains the transformed FRED-MD target column.
- `LEVEL__*` columns preserve raw level variables for `*-Level` feature cases,
  including `LEVEL__INDPRO` when the paper calls for `Y_t`.
- The feature step builder can select transformed and level predictors
  separately.

## Cell 5: Define paper feature steps

```python
def paper_feature_steps(case, target, transformed_predictors, level_predictors):
    steps = []
    parts = case.split("-")

    if "F" in parts:
        steps.append(
            mf.feature_engineering.pca_step(
                name="F_raw",
                columns=transformed_predictors,
                n_components=8,
                scale=True,
                include=False,
            )
        )
        steps.append(
            mf.feature_engineering.lag_step(
                name="F",
                input="F_raw",
                lags=range(0, 13),
                include=True,
            )
        )

    if "X" in parts:
        steps.append(
            mf.feature_engineering.lag_step(
                name="X",
                columns=transformed_predictors,
                lags=range(0, 13),
                include=True,
            )
        )

    if "MARX" in parts:
        steps.append(
            mf.feature_engineering.marx_step(
                name="MARX_X",
                columns=transformed_predictors,
                max_lag=12,
                scale_lags=False,
                include=True,
            )
        )
        steps.append(
            mf.feature_engineering.marx_step(
                name="MARX_y",
                input="target_panel",
                columns=[target],
                max_lag=12,
                scale_lags=False,
                include=True,
            )
        )

    if "MAF" in parts:
        steps.append(
            mf.feature_engineering.maf_step(
                name="MAF_X",
                columns=transformed_predictors,
                max_lag=12,
                n_components=2,
                scale=False,
                include=True,
            )
        )
        steps.append(
            mf.feature_engineering.maf_step(
                name="MAF_y",
                input="target_panel",
                columns=[target],
                max_lag=12,
                n_components=2,
                scale=False,
                include=True,
            )
        )

    if "Level" in parts:
        steps.append(
            mf.feature_engineering.lag_step(
                name="Level",
                columns=level_predictors,
                lags=range(0, 1),
                include=True,
            )
        )

    return steps


for step in paper_feature_steps("F-X-MARX-Level", TARGET, transformed_predictors, level_predictors):
    print(step["name"], step["method"], step["include"])
```

Output:

```text
F_raw pca False
F lag True
X lag True
MARX_X marx True
MARX_y marx True
Level lag True
```

Interpretation:

- `F_raw` is not included directly; it is a fitted PCA state used by the lagged
  factor block.
- `MARX_y` and `MAF_y` use `input="target_panel"` so the target is transformed
  as a target-derived block without becoming a normal predictor.
- `target_lags` are not mixed into `paper_feature_steps`; they are appended by
  `feature_spec(..., target_lags=range(0, 13))` so every feature case keeps the
  paper's target autoregressive block.

## Cell 6: Verify one feature matrix with target lags

```python
features = mf.feature_engineering.feature_spec(
    target=TARGET,
    horizon=3,
    predictors=transformed_predictors,
    steps=paper_feature_steps("F-MARX", TARGET, transformed_predictors, level_predictors),
    target_lags=range(0, 13),
    target_transform="average_value",
    target_mode="direct",
)

feature_set = features.fit_transform(processed.panel.iloc[:360])

print("X_shape", feature_set.X.shape)
print("y_shape", feature_set.y.shape)
print("last_X_columns", list(feature_set.X.columns[-13:]))
print(
    feature_set.feature_metadata
    .groupby(["step", "operation", "included"])
    .size()
    .rename("n_columns")
    .reset_index()
    .to_string(index=False)
)
print(
    feature_set.target_metadata[
        ["target_column", "source", "horizon", "mode", "transform", "operation", "aggregation"]
    ].to_string(index=False)
)
```

Output:

```text
X_shape (345, 1641)
y_shape (345, 1)
last_X_columns ['INDPRO_lag0', 'INDPRO_lag1', 'INDPRO_lag2', 'INDPRO_lag3', 'INDPRO_lag4', 'INDPRO_lag5', 'INDPRO_lag6', 'INDPRO_lag7', 'INDPRO_lag8', 'INDPRO_lag9', 'INDPRO_lag10', 'INDPRO_lag11', 'INDPRO_lag12']
       step  operation  included  n_columns
          F        lag      True        104
      F_raw        pca     False          8
     MARX_X       marx      True       1512
     MARX_y       marx      True         12
target_lags target_lag      True         13
          target_column source  horizon   mode     transform             operation     aggregation
INDPRO_average_value_h3 INDPRO        3 direct average_value direct_average_target mean_step_value
```

Interpretation:

- `target_lags=range(0, 13)` creates `INDPRO_lag0` through `INDPRO_lag12`.
- `MARX_y` creates the separate target MARX block required by the paper's
  feature table.
- Under the row-date convention, `lag0` is the transformed target available at
  forecast origin `t`; the runner masks unavailable future target values before
  prediction.
- `average_value` is required because official FRED-MD preprocessing has already
  transformed `INDPRO` into a one-period monthly forecasting object.

## Cell 7: Define the paper window

```python
paper_window_h3 = mf.window.from_cutoffs(
    estimation_start="1960-01",
    test_start="1980-01",
    test_end="2017-12",
    val_method="random_kfold",
    val_n_splits=5,
    val_random_state=123,
    horizon=3,
    step=1,
)

print("estimation_mode", paper_window_h3.estimation.mode)
print("estimation_start", paper_window_h3.estimation.start)
print("validation_method", paper_window_h3.val.method)
print("validation_n_splits", paper_window_h3.val.n_splits)
print("validation_random_state", paper_window_h3.val.random_state)
print("test_first_origin", paper_window_h3.test.first_origin)
print("test_last_origin", paper_window_h3.test.last_origin)
print("test_step", paper_window_h3.test.step)
print("test_horizon", paper_window_h3.test.horizon)
print("window_method", paper_window_h3.method)
```

Output:

```text
estimation_mode expanding
estimation_start 1960-01
validation_method random_kfold
validation_n_splits 5
validation_random_state 123
test_first_origin 1980-01
test_last_origin 2017-12
test_step 1
test_horizon 3
window_method random_kfold
```

## Cell 8: Run a direct-average smoke forecast

This smoke run uses the paper target construction and a paper feature case, but
only two origins and 20 trees. It is a pipeline check, not Table 2 evidence.

```python
smoke_window = mf.window.from_cutoffs(
    estimation_start="1960-01",
    test_start="2005-01",
    test_end="2005-04",
    val_size=60,
    horizon=3,
    step=3,
)

direct_features = mf.feature_engineering.feature_spec(
    target=TARGET,
    horizon=3,
    predictors=transformed_predictors,
    steps=paper_feature_steps("F", TARGET, transformed_predictors, level_predictors),
    target_lags=range(0, 13),
    target_transform="average_value",
    target_mode="direct",
)

direct_result = mf.forecasting.run(
    processed.panel,
    "random_forest",
    window=smoke_window,
    features=direct_features,
    target=TARGET,
    horizon=3,
    forecast_policy="direct_average",
    target_transform="value",
    model_selection={"random_forest": None},
    params={
        "n_estimators": 20,
        "max_features": 1 / 3,
        "min_samples_leaf": 5,
        "bootstrap": True,
        "random_state": 123,
        "n_jobs": 1,
    },
    save_models=False,
)

forecast_table = direct_result.to_frame()
print("forecast_rows", len(forecast_table))
print(
    forecast_table[
        ["origin", "date", "horizon", "model", "forecast_policy", "target_transform", "prediction", "actual"]
    ].to_string(index=False)
)
print("stage_policy_feature", direct_result.metadata["stage_policies"]["feature_engineering"]["scope"])
print("target_transform_meta", direct_result.metadata["features"]["target_transform"])
print("target_lags_meta", direct_result.metadata["features"]["target_lags"])
```

Output:

```text
forecast_rows 2
    origin       date  horizon         model forecast_policy target_transform  prediction   actual
2005-01-01 2005-04-01        3 random_forest  direct_average    average_value    0.002231 0.002181
2005-04-01 2005-07-01        3 random_forest  direct_average    average_value    0.002930 0.000828
stage_policy_feature fit_window
target_transform_meta average_value
target_lags_meta [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

Notes:

- `model_selection={"random_forest": None}` is deliberate. The paper's random
  forest setting is fixed rather than CV-tuned.
- The package default feature policy is `fit_window`, so PCA is refit only on
  data available in the estimation window for each origin.

## Cell 9: Run a path-average smoke forecast

```python
path_features = mf.feature_engineering.feature_spec(
    target=TARGET,
    horizon=3,
    predictors=transformed_predictors,
    steps=paper_feature_steps("F", TARGET, transformed_predictors, level_predictors),
    target_lags=range(0, 13),
    target_transform="value",
    target_mode="path",
)

path_result = mf.forecasting.run(
    processed.panel,
    "random_forest",
    window=smoke_window,
    features=path_features,
    target=TARGET,
    horizon=3,
    forecast_policy="path_average",
    target_transform="value",
    model_selection={"random_forest": None},
    params={
        "n_estimators": 20,
        "max_features": 1 / 3,
        "min_samples_leaf": 5,
        "bootstrap": True,
        "random_state": 123,
        "n_jobs": 1,
    },
    save_models=False,
)

forecast_table = path_result.to_frame()
print("forecast_rows", len(forecast_table))
print(
    forecast_table[
        ["origin", "date", "horizon", "model", "forecast_policy", "target_transform", "prediction", "actual"]
    ].to_string(index=False)
)
print("stage_policy_feature", path_result.metadata["stage_policies"]["feature_engineering"]["scope"])
print("target_transform_meta", path_result.metadata["features"]["target_transform"])
print("target_mode_meta", path_result.metadata["features"]["target_mode"])
```

Output:

```text
forecast_rows 2
    origin       date  horizon         model forecast_policy target_transform  prediction   actual
2005-01-01 2005-04-01        3 random_forest    path_average            value    0.003263 0.002181
2005-04-01 2005-07-01        3 random_forest    path_average            value    0.003667 0.000828
stage_policy_feature fit_window
target_transform_meta value
target_mode_meta path
```

## Cell 10: Evaluate the smoke forecasts

```python
print("direct_average")
print(
    direct_result
    .evaluate(by=("forecast_policy", "horizon"), metrics=("rmse", "mae"))
    .to_string(index=False)
)
print("")
print("path_average")
print(
    path_result
    .evaluate(by=("forecast_policy", "horizon"), metrics=("rmse", "mae"))
    .to_string(index=False)
)
```

Output:

```text
direct_average
forecast_policy  horizon  n    rmse      mae
 direct_average        3  2 0.00162 0.001226

path_average
forecast_policy  horizon  n     rmse      mae
   path_average        3  2 0.001935 0.001679
```

## Cell 11: Full-paper run template

This cell is the package-only long-run template. It should be run in a server
batch session, not as an interactive smoke check.
Random Forest is fixed in the paper, so `model_selection={"random_forest":
None}` is correct. For Adaptive Lasso, Elastic Net, Boosted Trees, and Linear
Boosting, use the same `paper_window_h3` random-fold validation and pass a
paper-style `SearchSpec` to `model_selection`.

```python
PAPER_RF_PARAMS = {
    "n_estimators": 200,
    "max_features": 1 / 3,
    "min_samples_leaf": 5,
    "bootstrap": True,
    "random_state": 123,
    "n_jobs": 1,
}


def run_one(target_alias, horizon, feature_case, target_policy):
    target = TARGET_MAP[target_alias]
    window = mf.window.from_cutoffs(
        estimation_start="1960-01",
        test_start="1980-01",
        test_end="2017-12",
        val_size=60,
        horizon=horizon,
        step=1,
    )
    panel = replication_panel if "Level" in feature_case else processed.panel
    transformed_cols = [column for column in processed.panel.columns if column != target]
    level_cols = [f"LEVEL__{column}" for column in bundle.panel.columns]
    feature_predictors = transformed_cols + (level_cols if "Level" in feature_case else [])
    features = mf.feature_engineering.feature_spec(
        target=target,
        horizon=horizon,
        predictors=feature_predictors,
        steps=paper_feature_steps(feature_case, target, transformed_cols, level_cols),
        target_lags=range(0, 13),
        target_transform="average_value" if target_policy == "direct_average" else "value",
        target_mode="direct" if target_policy == "direct_average" else "path",
    )
    return mf.forecasting.run(
        panel,
        "random_forest",
        window=window,
        features=features,
        target=target,
        horizon=horizon,
        forecast_policy=target_policy,
        target_transform="value",
        model_selection={"random_forest": None},
        params=PAPER_RF_PARAMS,
        save_models=True,
        model_store="trained_model/random_forest",
    )


full_candidate = run_one(
    target_alias="INDPRO",
    horizon=3,
    feature_case="F-MARX",
    target_policy="direct_average",
)
print(full_candidate.to_frame().head().to_string(index=False))
print(full_candidate.evaluate(by=("model", "horizon"), metrics=("rmse",)).to_string(index=False))
```

Expected role of this cell:

```text
Runs the full monthly OOS calendar for one target/horizon/feature/policy/model
combination. Repeat it over TARGET_MAP, horizons, feature_cases, and target
policies to build Table 2-style comparison files.
```
