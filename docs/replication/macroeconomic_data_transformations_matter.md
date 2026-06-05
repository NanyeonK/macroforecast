# Macroeconomic Data Transformations Matter

This page records the current replication setting for Goulet Coulombe, Leroux,
Stevanovic, and Surprenant (2021), "Macroeconomic data transformations matter,"
*International Journal of Forecasting*, 37(4), 1338-1354.

Sources checked for this page:

- IJF article DOI: <https://doi.org/10.1016/j.ijforecast.2021.05.005>
- arXiv working-paper page: <https://arxiv.org/abs/2008.01714>
- local main PDF:
  `/Users/nanyeon/Library/CloudStorage/SynologyDrive-second_brain/wiki/raw/papers/10.1016j.ijforecast.2021.05.005.pdf`
- local online appendix PDF:
  `/Users/nanyeon/Library/CloudStorage/SynologyDrive-second_brain/wiki/raw/papers/10.1016j.ijforecast.2021.05.005_appendix.pdf`
- author Code & Data page:
  <https://philippegouletcoulombe.com/code>
- author MARX note:
  <https://philippegouletcoulombe.com/blog/ml-based-time-series-modelling-with-marx-2>
- local MARX snippet:
  `/Users/nanyeon/Library/CloudStorage/SynologyDrive-second_brain/wiki/raw/paper_code/coulombe_site_github_20260530/marx/MARX_cheap_code.R`

## Replication Status

| Item | Status | Consequence |
| --- | --- | --- |
| Main-text design | available | The sample, target list, horizon grid, transformation grid, learner list, and evaluation objects can be fixed. |
| Appendix model algorithms | available | Elastic Net, Adaptive Lasso, Random Forest, Boosted Trees, Linear Boosting, GA search, and Bayesian search can be mapped to package settings. |
| Full official replication code | not found in the checked public/local materials | Table-identical replication is not claimed. |
| Exact FRED-MD vintage | not stated in the checked public/local materials | Use a historical FRED-MD vintage just after the `2017M12` sample end; verify `2018-01`, then compare `2018-02`/`2018-03` if Dec. 2017 coverage differs. |
| MATLAB default optimizer behavior | partially stated | We can match the high-level optimizer design, but MATLAB default internal randomness/surrogate settings are not exactly portable. |
| MARX reference code | available as a small R snippet | `macroforecast.feature_engineering.feature_matrix(..., specification="MARX")` implements the same moving-average lag rotation logic. |
| `macroforecast==0.9.5a1` diagnostic run | invalid as replication evidence | The runner and diagnostic harness used `row < origin` for h-step targets, which leaks labels whose realization date is after the forecast origin. The diagnostic also used `average_change` on an already transformed FRED-MD target, creating a double-difference target. Keep the run as an investigation log only. |

Replication level:
: `reconstructed_design`, not `exact_table_replication`.

Package feasibility:
: all Table 2 design axes can be expressed with public `macroforecast`
  callables: official FRED-MD vintage loading, McCracken-Ng t-code
  preprocessing, direct-average and path-average targets, all 16 feature
  matrices, AR/FM/AL/EN/LB/RF/BT learner families, RMSE/relative-RMSE scoring,
  and DM/MCS forecast-comparison tests.

Exact-replication blocker:
: the paper and appendix do not provide a full machine-readable replication
  package or exact FRED-MD vintage in the checked materials.

Vintage policy:
: do not use `current.csv` for this paper. The official FRED-MD page provides
  historical vintage zip files. Because the pseudo-out-of-sample period ends in
  `2017M12`, the first replication candidate is `2018-01.csv` extracted from
  the historical FRED-MD vintage zip; use `2018-02.csv` or `2018-03.csv` only if
  the January vintage does not contain the needed December 2017 realized values.

## Paper Design

| Axis | Paper setting | `macroforecast` setting |
| --- | --- | --- |
| Data | monthly FRED-MD after McCracken-Ng transformations | `mf.data.load_fred_md(...)` followed by `mf.preprocessing.reprocess(transform="official")` |
| Initial estimation sample | starts at `1960M01` | `mf.window.from_cutoffs(estimation_start="1960-01", ...)` |
| Pseudo-out-of-sample period | `1980M01` to `2017M12` | test start `1980-01`, test end `2017-12` |
| Window type | expanding | expanding train/validation/test policy |
| Targets | `INDPRO`, `EMP`, `UNRATE`, `INCOME`, `CONS`, `RETAIL`, `HOUST`, `M2`, `CPI`, `PPI` | pass one target at a time to `mf.forecasting.run(...)`; keep the list fixed for batch replication |
| Horizons | 1, 3, 6, 9, 12, 24 months | `horizons=[1, 3, 6, 9, 12, 24]` |
| Target autoregressive lags | include lags of the target transformation; appendix setting `P_y=12` | `target_lags=range(0, 13)` in row-date convention, i.e. current transformed target plus 12 past lags |
| Target construction | direct average growth/difference and path-average forecasts from one-period transformed targets | after official t-code preprocessing, use `target_transform="average_value"` for direct average and `target_transform="value"` for path average |
| Loss | RMSE | `mf.metrics.rmse(...)` or RMSE from forecast output |
| Reference benchmark | factor model, FM | package mapping: fixed or BIC-selected `mf.models.far` / OLS factor model |
| Accuracy tests | DM test and MCS | `mf.tests.dm_test(...)`, `mf.tests.model_confidence_set(...)` |

## Target Variables

The paper uses ten monthly targets:

| Paper label | Meaning in the article | FRED-MD column in `2018-01` vintage | Package alias |
| --- | --- | --- |
| `INDPRO` | Industrial production index | `INDPRO` | `INDPRO` |
| `EMP` | Total nonfarm employment | `PAYEMS` | `EMP` |
| `UNRATE` | Unemployment rate | `UNRATE` | `UNRATE` |
| `INCOME` | Real personal income excluding current transfers | `W875RX1` | `INCOME` |
| `CONS` | Real personal consumption expenditures | `DPCERA3M086SBEA` | `CONS` |
| `RETAIL` | Retail and food services sales | `RETAILx` | `RETAIL` |
| `HOUST` | Housing starts | `HOUST` | `HOUST` |
| `M2` | M2 money stock | `M2SL` | `M2` |
| `CPI` | Consumer price index | `CPIAUCSL` | `CPI` |
| `PPI` | Producer price index | `PPICMM` | `PPI` |

If the loaded FRED-MD vintage uses different column labels, create a target
alias map before running the grid. Do not silently substitute a nearby series.

## Feature Matrices

The main text fixes sixteen feature-matrix cases. Lags of the
month-to-month change or log-change of the target are always included.

| Paper case | Content | Package expression |
| --- | --- | --- |
| `F` | factors and factor lags | `feature_matrix(X, specification="F", n_factors=8, lags=range(1, 13))` |
| `F-X` | factors plus lagged transformed observables | `specification="F-X"` |
| `F-MARX` | factors plus MARX rotations | `specification="F-MARX"` |
| `F-MAF` | factors plus moving-average factors | `specification="F-MAF"` |
| `F-Level` | factors plus levels | `specification="F-Level"`, with `level_data=levels` |
| `F-X-MARX` | factors, lagged observables, MARX | `specification="F-X-MARX"` |
| `F-X-MAF` | factors, lagged observables, MAF | `specification="F-X-MAF"` |
| `F-X-Level` | factors, lagged observables, levels | `specification="F-X-Level"`, with `level_data=levels` |
| `F-X-MARX-Level` | factors, lagged observables, MARX, levels | `specification="F-X-MARX-Level"`, with `level_data=levels` |
| `X` | lagged transformed observables | `specification="X"` |
| `MARX` | MARX rotations | `specification="MARX"` |
| `MAF` | moving-average factors | `specification="MAF"` |
| `X-MARX` | lagged observables plus MARX | `specification="X-MARX"` |
| `X-MAF` | lagged observables plus MAF | `specification="X-MAF"` |
| `X-Level` | lagged observables plus levels | `specification="X-Level"`, with `level_data=levels` |
| `X-MARX-Level` | lagged observables, MARX, levels | `specification="X-MARX-Level"`, with `level_data=levels` |

Default reconstruction settings:

```python
feature_settings = {
    "lags": range(1, 13),
    "max_lag": 12,
    "n_factors": 8,
    "n_maf_components": 2,
    "fit_policy": "expanding",
    "include_current_factor": True,
    "scale_factors": True,
    "scale_marx": False,
    "scale_maf": False,
}
```

The MAF setting follows the article's stated `P_MAF = 12` and two MAFs per
series. The MARX setting follows the author R snippet: for each variable and lag
order `l`, replace the lag-`l` column by the mean of lags `1, ..., l`.

## Learner Settings

| Learner | Paper setting | Closest package setting |
| --- | --- | --- |
| AR | autoregressive benchmark; lag order selected by BIC | `mf.models.ar`; add BIC lag search in the runner grid when matching the benchmark |
| FM | Stock-Watson-style factor model; BIC hyperparameter selection mentioned in main text | `mf.models.far` or factor features with `mf.models.ols`; for ML comparisons keep `(P_y, P_f, k)=(12, 12, 8)` where relevant |
| Adaptive Lasso | gamma `1`; first-step ridge; ridge lambda by GA; 100 log-spaced lasso lambdas; 5-fold squared-loss CV | `mf.models.adaptive_lasso(gamma=1, initial="ridge")`; use `mf.model_selection` to search `initial_alpha` and `alpha` |
| Elastic Net | 100 lambda values; 100 alpha values in `[0.01, 1]`; 5-fold squared-loss CV | `mf.models.elastic_net`; use model-owned search space over `alpha` and `l1_ratio` |
| Linear Boosting | component-wise L2 boosting; `m=min(200,#Z/3)` sampled features; `N in 1..500`, `eta in [0,1]` via GA | `mf.models.glmboost`; search `n_iter` and `learning_rate`; candidate subsampling is an approximation unless a custom GLMBoost candidate sampler is supplied |
| Random Forest | 200 trees; bootstrap sample; terminal node size greater than 5; `#Z/3` split candidates; no CV | `mf.models.random_forest(n_estimators=200, min_samples_leaf=5, max_features=max(1, p//3), bootstrap=True)` |
| Boosted Trees | initial mean; depth 5 splits; `#Z/3` split candidates; `N in 1..500`, `eta in (0,1)` by Bayesian optimization, 5-fold CV | `mf.models.gradient_boosting`; search `n_estimators` and `learning_rate`; use `max_depth=5` or a leaf-count equivalent depending on the backend |

Important approximation:
: MATLAB tree defaults and Python/sklearn tree defaults are not identical. The
  package setting above matches the published algorithmic intent, not every
  backend implementation detail.

## Paper Result Anchors

These are not package test assertions yet. They are qualitative/structural
anchors that a successful reconstructed run should make visible.

| Anchor | Expected pattern from the article |
| --- | --- |
| MARX | material gains for real activity variables at short and medium horizons, especially with tree learners |
| MAF | weaker on average than MARX, but useful in selected longer-horizon cases |
| Factors | included in most best specifications |
| Level | useful for money and selected real/activity cases |
| Direct average vs path average | path average often helps cyclical real variables; direct average is stronger for selected nominal variables and horizons |
| RF vs BT | nonlinear tree methods dominate many best cases; RF appears more often than BT among best specifications |

For exact comparison against Table 2 or Appendix B relative RMSE tables, freeze a
FRED-MD vintage and record the backend versions before treating deviations as
package bugs.

## Notebook-Style Skeleton

The current page fixes the setting. The full replication notebook should execute
the same cells and then attach paper-table captures plus package outputs.

For the current package-only executed notebook report, see
[Macroeconomic Data Transformations Matter: Package-Only Notebook](macroeconomic_data_transformations_matter_notebook.md).

### Cell 1: Load a Frozen FRED-MD Panel

```python
import macroforecast as mf

frozen_vintage = "2018-01"  # first candidate after the paper's 2017M12 sample end
bundle = mf.data.load_fred_md(vintage=frozen_vintage)

raw_panel = bundle.panel
raw_metadata = bundle.metadata
```

Expected output:

```text
raw_panel: pandas.DataFrame indexed by monthly date
raw_metadata: metadata dictionary attached to the data bundle
raw_metadata["artifact"]["source_url"]: official historical archive plus member CSV
```

### Cell 2: Build the Stationary Panel and Preserve Levels

```python
levels = raw_panel.copy()

X = mf.preprocessing.reprocess(
    raw_panel,
    transform="official",
    outliers="iqr",
    outlier_action="flag_as_nan",
    iqr_threshold=10.0,
    impute="em_factor",
    em_n_factors=8,
    frame="keep",
)
```

Expected output:

```text
X: FRED-MD-style stationary pandas.DataFrame
levels: original level panel for Level feature specifications
```

### Cell 3: Define the Paper Grid

```python
target_map = {
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
targets = list(target_map.values())
horizons = [1, 3, 6, 9, 12, 24]
feature_cases = [
    "F", "F-X", "F-MARX", "F-MAF", "F-Level",
    "F-X-MARX", "F-X-MAF", "F-X-Level", "F-X-MARX-Level",
    "X", "MARX", "MAF", "X-MARX", "X-MAF", "X-Level", "X-MARX-Level",
]
models = ["ar", "far", "adaptive_lasso", "elastic_net", "glmboost", "random_forest", "gradient_boosting"]
target_policies = ["direct_average", "path_average"]
```

Expected output:

```text
10 targets x 6 horizons x 16 feature cases x 5 ML learners x 2 target policies,
plus AR and FM benchmarks.
```

### Cell 3B: Encode Table 2 Best-Specification Targets

The published Table 2 can be encoded as a target comparison dictionary. Each
entry is `(target_policy, learner, feature_case)`.

```python
PAPER_TABLE2 = {
    ("INDPRO", 1): ("direct_average", "random_forest", "F-X-MARX-Level"),
    ("INDPRO", 3): ("direct_average", "random_forest", "MARX"),
    ("INDPRO", 6): ("path_average", "random_forest", "MARX"),
    ("INDPRO", 9): ("path_average", "random_forest", "MARX"),
    ("INDPRO", 12): ("path_average", "random_forest", "MARX"),
    ("INDPRO", 24): ("direct_average", "random_forest", "F-Level"),

    ("EMP", 1): ("direct_average", "random_forest", "F-X-MARX-Level"),
    ("EMP", 3): ("path_average", "random_forest", "F-MARX"),
    ("EMP", 6): ("path_average", "gradient_boosting", "F-MARX"),
    ("EMP", 9): ("path_average", "gradient_boosting", "F-MARX"),
    ("EMP", 12): ("path_average", "gradient_boosting", "F-MARX"),
    ("EMP", 24): ("path_average", "gradient_boosting", "MAF"),

    ("UNRATE", 1): ("direct_average", "gradient_boosting", "F-MARX"),
    ("UNRATE", 3): ("direct_average", "random_forest", "F-X-MARX-Level"),
    ("UNRATE", 6): ("path_average", "random_forest", "F-MARX"),
    ("UNRATE", 9): ("path_average", "glmboost", "F-X-MARX-Level"),
    ("UNRATE", 12): ("path_average", "glmboost", "F-X-MARX-Level"),
    ("UNRATE", 24): ("direct_average", "gradient_boosting", "F-MAF"),

    ("INCOME", 1): ("direct_average", "random_forest", "MARX"),
    ("INCOME", 3): ("direct_average", "random_forest", "F-MARX"),
    ("INCOME", 6): ("path_average", "random_forest", "F-X-MARX"),
    ("INCOME", 9): ("path_average", "random_forest", "F-MARX"),
    ("INCOME", 12): ("path_average", "random_forest", "F-MARX"),
    ("INCOME", 24): ("path_average", "random_forest", "F-X-MARX"),

    ("CONS", 1): ("direct_average", "far", "F"),
    ("CONS", 3): ("direct_average", "random_forest", "F-Level"),
    ("CONS", 6): ("path_average", "random_forest", "F-Level"),
    ("CONS", 9): ("direct_average", "random_forest", "MAF"),
    ("CONS", 12): ("path_average", "random_forest", "F-MAF"),
    ("CONS", 24): ("path_average", "random_forest", "F-MAF"),

    ("RETAIL", 1): ("direct_average", "far", "F"),
    ("RETAIL", 3): ("path_average", "gradient_boosting", "F-X-MARX"),
    ("RETAIL", 6): ("path_average", "adaptive_lasso", "F-MARX"),
    ("RETAIL", 9): ("direct_average", "gradient_boosting", "F-X-MARX-Level"),
    ("RETAIL", 12): ("direct_average", "gradient_boosting", "F-X-Level"),
    ("RETAIL", 24): ("direct_average", "gradient_boosting", "F-X-MAF"),

    ("HOUST", 1): ("direct_average", "elastic_net", "F-Level"),
    ("HOUST", 3): ("path_average", "elastic_net", "F-Level"),
    ("HOUST", 6): ("path_average", "random_forest", "F-X-MARX"),
    ("HOUST", 9): ("direct_average", "random_forest", "F-MAF"),
    ("HOUST", 12): ("direct_average", "random_forest", "F"),
    ("HOUST", 24): ("direct_average", "random_forest", "F"),

    ("M2", 1): ("direct_average", "random_forest", "X-Level"),
    ("M2", 3): ("path_average", "adaptive_lasso", "X-Level"),
    ("M2", 6): ("path_average", "random_forest", "F-Level"),
    ("M2", 9): ("direct_average", "random_forest", "F-Level"),
    ("M2", 12): ("direct_average", "gradient_boosting", "F-Level"),
    ("M2", 24): ("path_average", "random_forest", "F-Level"),

    ("CPI", 1): ("direct_average", "adaptive_lasso", "MARX"),
    ("CPI", 3): ("direct_average", "random_forest", "F"),
    ("CPI", 6): ("direct_average", "random_forest", "F"),
    ("CPI", 9): ("direct_average", "random_forest", "F"),
    ("CPI", 12): ("direct_average", "random_forest", "F"),
    ("CPI", 24): ("path_average", "random_forest", "X"),

    ("PPI", 1): ("direct_average", "elastic_net", "F-MARX"),
    ("PPI", 3): ("direct_average", "elastic_net", "MARX"),
    ("PPI", 6): ("direct_average", "random_forest", "F"),
    ("PPI", 9): ("direct_average", "random_forest", "F"),
    ("PPI", 12): ("direct_average", "random_forest", "F"),
    ("PPI", 24): ("direct_average", "gradient_boosting", "F-Level"),
}

assert set(PAPER_TABLE2) == {(target, horizon) for target in target_map for horizon in horizons}
```

The learner labels are package callables:

```python
MODEL_CALLABLES = {
    "ar": mf.models.ar,
    "far": mf.models.far,
    "adaptive_lasso": mf.models.adaptive_lasso,
    "elastic_net": mf.models.elastic_net,
    "glmboost": mf.models.glmboost,
    "random_forest": mf.models.random_forest,
    "gradient_boosting": mf.models.gradient_boosting,
}
```

### Cell 4A: Build the Table-1 Feature Matrix Directly

```python
Z = mf.feature_engineering.feature_matrix(
    X,
    specification="F-X-MARX",
    level_data=levels,
    lags=range(1, 13),
    max_lag=12,
    n_factors=8,
    n_maf_components=2,
    fit_policy="expanding",
)
```

Expected output:

```text
Z: pandas.DataFrame matching the paper's F-X-MARX feature family.
Z.attrs["macroforecast_feature_metadata"]: feature lineage table.
```

This direct call is the clearest way to inspect Table 1. For strict
walk-forward forecasting, let the runner fit feature state inside each
expanding window with a `FeatureSpec` step pipeline.

### Cell 4B: Runner-Safe Equivalent for One Feature Case

```python
features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=3,
    predictors="all",
    lags=None,
    target_lags=range(0, 13),
    target_transform="average_value",
    target_mode="direct",
    steps=[
        mf.feature_engineering.pca_step(
            name="F_raw",
            n_components=8,
            scale=True,
            include=False,
        ),
        mf.feature_engineering.lag_step(
            name="F",
            input="F_raw",
            lags=range(0, 13),
            include=True,
        ),
        mf.feature_engineering.lag_step(
            name="X",
            input="panel",
            lags=range(1, 13),
            include=True,
        ),
        mf.feature_engineering.marx_step(
            name="MARX",
            input="panel",
            max_lag=12,
            include=True,
        ),
    ],
)
```

Expected output:

```text
features: FeatureSpec fitted by forecasting.run() according to feature_policy.
```

`FeatureSpec` does not call `feature_matrix()` internally. It uses smaller
step-level callables so PCA, MAF, scaling, and custom fitted transforms can be
estimated only on the rows available to each forecast origin.

### Cell 4C: Run One Reconstructed Design Point

```python
window = mf.window.from_cutoffs(
    estimation_start="1960-01",
    test_start="1980-01",
    test_end="2017-12",
    mode="expanding",
    val_method="blocked_kfold",
    val_n_splits=5,
    horizon=3,
    step=1,
)

result = mf.forecasting.run(
    X,
    model="random_forest",
    target="INDPRO",
    horizon=3,
    forecast_policy="direct_average",
    features=features,
    window=window,
    params={
        "n_estimators": 200,
        "min_samples_leaf": 5,
        "max_features": 0.3333333333333333,
        "bootstrap": True,
    },
)
```

Expected output:

```text
result.forecasts: one row per forecast origin/date with prediction, actual, model metadata, and window metadata
result.metadata: run-level record of data, feature, model, selection, and window choices
```

The `max_features` value uses a fraction because sklearn accepts fractional
feature subsampling. If an exact floor rule is required, resolve
`max(1, p // 3)` after the feature matrix has been materialized.

### Cell 5: Run Table 2 Cells With Package Callables

The following helper code runs the package-only Table 2 comparison path. It is
written as plain Python so the same block can be copied into a notebook or run
as a batch script.

```python
import math
import numpy as np
import pandas as pd

def package_model_fit_predict(model_name, X_train, y_train, X_test):
    """Fit one Table 2 learner with macroforecast public callables only."""
    if model_name == "far":
        # FM benchmark as factor autoregression. For Table 2 cells whose feature
        # matrix is already F-only, mf.models.ols on the factor matrix is also a
        # valid fixed-k FM reconstruction.
        fit = mf.models.ols(X_train, y_train)
    elif model_name == "adaptive_lasso":
        fit = mf.models.adaptive_lasso(
            X_train,
            y_train,
            gamma=1.0,
            initial="ridge",
            initial_alpha=1.0,
            alpha=0.01,
            max_iter=20_000,
            random_state=123,
        )
    elif model_name == "elastic_net":
        fit = mf.models.elastic_net(
            X_train,
            y_train,
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=20_000,
        )
    elif model_name == "glmboost":
        fit = mf.models.glmboost(
            X_train,
            y_train,
            n_iter=100,
            learning_rate=0.1,
        )
    elif model_name == "random_forest":
        p = X_train.shape[1]
        fit = mf.models.random_forest(
            X_train,
            y_train,
            n_estimators=200,
            min_samples_leaf=5,
            max_features=max(1, p // 3),
            bootstrap=True,
            random_state=123,
            n_jobs=1,
        )
    elif model_name == "gradient_boosting":
        p = X_train.shape[1]
        fit = mf.models.gradient_boosting(
            X_train,
            y_train,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            max_features=max(1, p // 3),
            random_state=123,
        )
    else:
        raise ValueError(f"unsupported Table 2 learner: {model_name}")

    prediction = fit.predict(X_test)
    return float(prediction.iloc[0] if hasattr(prediction, "iloc") else np.asarray(prediction).reshape(-1)[0])


def direct_target_column(target_column, horizon, columns):
    name = f"{target_column}_average_value_h{horizon}"
    if name in columns:
        return name
    matches = [
        column for column in columns
        if column.startswith(target_column) and "average_value" in column and f"h{horizon}" in column
    ]
    if not matches:
        raise KeyError((target_column, horizon, "direct_average"))
    return matches[0]


def path_target_columns(target_column, horizon, columns):
    out = []
    for step in range(1, horizon + 1):
        name = f"{target_column}_value_step{step}"
        if name in columns:
            out.append(name)
        else:
            matches = [
                column for column in columns
                if column.startswith(target_column) and "value" in column and f"step{step}" in column
            ]
            if not matches:
                raise KeyError((target_column, horizon, step, "path_average"))
            out.append(matches[0])
    return out


def annual_december_origins(index, horizon):
    """Small Table 2 smoke grid. Use every month for the full paper run."""
    index = set(pd.DatetimeIndex(index))
    origins = []
    for year in range(1980, 2018):
        realized = pd.Timestamp(year=year, month=12, day=1)
        origin = realized - pd.DateOffset(months=horizon)
        if origin in index:
            origins.append(origin)
    return origins


def available_target_training_frame(frame, base_index, origin, target_step):
    """Keep only rows whose h-step or step-specific label is observable at origin."""
    base_index = pd.DatetimeIndex(base_index)
    origin_pos = int(base_index.get_indexer([origin])[0])
    if origin_pos < 0:
        raise KeyError(origin)
    row_pos = pd.Series(base_index.get_indexer(frame.index), index=frame.index)
    mask = (row_pos >= 0) & ((row_pos + int(target_step)) <= origin_pos)
    return frame.loc[mask]


def run_one_table2_cell(processed, levels, target_label, horizon, policy, model_name, feature_case):
    target_column = target_map[target_label]
    feature_set = mf.feature_engineering.build_features(
        processed,
        targets=[target_column],
        horizons=[horizon],
        predictors="all",
        target_lags=range(0, 13),
        feature_specification=feature_case,
        max_lag=12,
        n_factors=8,
        n_maf_components=2,
        feature_fit_policy="expanding",
        feature_min_train_size=240,
        level_data=levels,
        target_mode="path" if policy == "path_average" else "direct",
        target_transform="value" if policy == "path_average" else "average_value",
        drop_missing=False,
    )

    realized_direct = mf.feature_engineering.average_target(
        processed,
        targets=[target_column],
        horizons=[horizon],
        transform="value",
    )
    actual_col = direct_target_column(target_column, horizon, realized_direct.columns)

    rows = []
    for origin in annual_december_origins(feature_set.X.index, horizon):
        if origin not in realized_direct.index:
            continue
        actual = float(realized_direct.loc[origin, actual_col])
        if math.isnan(actual):
            continue

        if policy == "direct_average":
            y_col = direct_target_column(target_column, horizon, feature_set.y.columns)
            frame = pd.concat([feature_set.X, feature_set.y[[y_col]]], axis=1).dropna()
            if origin not in frame.index:
                continue
            train = available_target_training_frame(frame, feature_set.X.index, origin, horizon)
            if len(train) < 240:
                continue
            pred = package_model_fit_predict(
                model_name,
                train[feature_set.X.columns],
                train[y_col],
                frame.loc[[origin], feature_set.X.columns],
            )
        else:
            step_preds = []
            for step, y_col in enumerate(path_target_columns(target_column, horizon, feature_set.y.columns), start=1):
                frame = pd.concat([feature_set.X, feature_set.y[[y_col]]], axis=1).dropna()
                if origin not in frame.index:
                    continue
                train = available_target_training_frame(frame, feature_set.X.index, origin, step)
                if len(train) < 240:
                    continue
                step_preds.append(
                    package_model_fit_predict(
                        model_name,
                        train[feature_set.X.columns],
                        train[y_col],
                        frame.loc[[origin], feature_set.X.columns],
                    )
                )
            if len(step_preds) != horizon:
                continue
            pred = float(np.mean(step_preds))

        rows.append(
            {
                "target_label": target_label,
                "target_column": target_column,
                "horizon": horizon,
                "policy": policy,
                "model": model_name,
                "feature_case": feature_case,
                "origin": origin,
                "realized_date": origin + pd.DateOffset(months=horizon),
                "prediction": pred,
                "actual": actual,
                "error": actual - pred,
            }
        )
    return pd.DataFrame(rows)


table2_forecasts = []
for (target_label, horizon), (policy, model_name, feature_case) in PAPER_TABLE2.items():
    table2_forecasts.append(
        run_one_table2_cell(
            processed=X,
            levels=levels,
            target_label=target_label,
            horizon=horizon,
            policy=policy,
            model_name=model_name,
            feature_case=feature_case,
        )
    )

table2_forecasts = pd.concat(table2_forecasts, ignore_index=True)
table2_rmse = (
    table2_forecasts.assign(squared_error=lambda d: d["error"] ** 2)
    .groupby(["target_label", "horizon", "policy", "model", "feature_case"], as_index=False)
    .agg(n=("error", "size"), rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))))
)
```

The annual-December origin loop is intentionally light. It verifies that every
Table 2 cell is callable with the package and creates a first comparison table.
For the full paper run, replace `annual_december_origins()` with every monthly
origin from `1980-01` through `2017-12`, and keep the same package calls.

Long-horizon boundary:
: the package window uses origin-date cutoffs. With monthly `step=1`, h=24
  forecasts overlap month by month, but a scored forecast at origin `t` still
  needs the realized target at `t + 24`. For a FRED-MD vintage ending in
  `2017-12`, the last h=24 origin that can be scored is `2015-12`. Tail blocks
  such as calendar year `2016` or `2017` should be skipped in a scored
  replication run unless a later vintage supplies the required realized target
  dates.

### Cell 6: Evaluate Against the FM Benchmark

```python
rf_forecasts = result.forecasts
fm_forecasts = ...  # run the matching FM benchmark with the same window and target policy

relative = mf.metrics.relative_mse(
    y_true=rf_forecasts["actual"],
    y_model=rf_forecasts["prediction"],
    y_benchmark=fm_forecasts["prediction"],
)

dm = mf.tests.dm_test(
    (rf_forecasts["actual"] - rf_forecasts["prediction"]) ** 2,
    (fm_forecasts["actual"] - fm_forecasts["prediction"]) ** 2,
    horizon=3,
    input_type="loss",
)
```

Expected output:

```text
relative_mse < 1 means the reconstructed model improves on FM under squared loss.
dm returns a statistic and p-value for equal predictive accuracy against the benchmark.
```

## Observed Package Diagnostics

This section records package runs already executed from an isolated PyPI
environment. These runs are not exact Table-identical replications. They are
diagnostic checks that the public package can execute the Table 2 cells and that
the resulting patterns can be compared with the paper's main anchors.

Corrected source smoke check:
: after fixing target-label availability and pre-transformed target semantics,
  the local source checkout was run on `mf.data.load_fred_md(vintage="2018-01")`
  with official preprocessing, target `INDPRO`, horizon `3`, and origin
  `2005-12-01`. `forecast_policy="direct_average", target_transform="value"`
  normalized to `average_value`; `forecast_policy="path_average",
  target_transform="value"` kept value step targets. Both policies produced the
  same realized target, `0.0011626872172302665`. The direct row recorded
  `target_availability_end_pos=558` and `target_availability_lag=3`; the path row
  recorded step cutoffs `{1: 560, 2: 559, 3: 558}`. This is a smoke check of the
  corrected callable contract, not a Table 2 result.

Execution environment:

- machine: server1
- package: `macroforecast==0.9.5a1` installed from PyPI in a fresh virtual
  environment
- data: `mf.data.load_fred_md(vintage="2018-01")`
- raw panel: 708 monthly rows x 127 columns, `1959-01` through `2017-12`
- preprocessing: official McCracken-Ng transformation codes with the package
  default FRED-MD preprocessing path
- benchmark: matching FM cell with the same target, horizon, forecast policy,
  diagnostic origins, and realized target construction

Important caveat:
: the main diagnostic below uses every monthly realized date from `1980-01`
  through `2017-12`, but still uses capped lightweight model settings and
  full-sample feature fitting for factors, MARX, and MAF. It is closer to the
  paper's OOS calendar than the sparse diagnostics. However, it was run with
  `macroforecast==0.9.5a1`, whose direct/path training cutoff did not enforce
  target realization availability for `h > 1`, and the diagnostic script built
  targets with `average_change` on an already McCracken-Ng transformed panel.
  It also used capped lightweight RF/BT settings and fixed hyperparameters
  instead of the appendix optimizer settings. Treat the numbers below as an
  investigation log, not replication evidence.

| Run | Origins | Table 2 cells | Forecast rows | Share beats FM | Mean relative RMSE vs FM | Runtime | Role |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ultrafast sparse OOS-lite | 1980, 1990, 2000, 2010, 2017 | 60 | 600 | 0.467 | 1.188 | 179.8s | smoke diagnostic |
| Capped sparse OOS-lite | 1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016, 2017 | 60 | 1680 | 0.367 | 1.148 | 333.3s | broader diagnostic |
| Monthly capped OOS-lite | every month from 1980-01 to 2017-12 | 60 | 54720 | 0.200 | 1.149 | 7461.1s | main diagnostic |

Verdict from the diagnostic runs:
: invalid as replication evidence. The runs are useful only because they exposed
  two package/harness errors to fix before any serious Table 2 comparison:
  label-availability leakage for h-step targets and a double-transformed target
  construction. Re-run the grid after the corrected runner and
  `average_value`/`value` target semantics are released.

Target-level summary for the invalid monthly capped diagnostic:

| Target | Cells beating FM | Mean relative RMSE vs FM |
| --- | ---: | ---: |
| CONS | 0/6 | 1.324 |
| CPI | 0/6 | 1.317 |
| EMP | 2/6 | 1.029 |
| HOUST | 4/6 | 0.928 |
| INCOME | 0/6 | 1.284 |
| INDPRO | 0/6 | 1.331 |
| M2 | 2/6 | 1.205 |
| PPI | 0/6 | 1.022 |
| RETAIL | 0/6 | 1.034 |
| UNRATE | 4/6 | 1.012 |

Horizon-level summary for the invalid monthly capped diagnostic:

| Horizon | Cells beating FM | Mean relative RMSE vs FM |
| ---: | ---: | ---: |
| 1 | 3/10 | 1.046 |
| 3 | 2/10 | 1.051 |
| 6 | 2/10 | 1.133 |
| 9 | 2/10 | 1.098 |
| 12 | 2/10 | 1.186 |
| 24 | 1/10 | 1.378 |

Full monthly capped diagnostic table:

| Target | h | Target policy | Learner | Feature case | n | Relative RMSE vs FM | Beats FM? |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| CONS | 1 | direct | FM | F | 456 | 1.000 | no |
| CONS | 3 | direct | RF | F-Level | 456 | 1.052 | no |
| CONS | 6 | path | RF | F-Level | 456 | 1.295 | no |
| CONS | 9 | direct | RF | MAF | 456 | 1.060 | no |
| CONS | 12 | path | RF | F-MAF | 456 | 1.493 | no |
| CONS | 24 | path | RF | F-MAF | 456 | 2.041 | no |
| CPI | 1 | direct | AL | MARX | 456 | 1.309 | no |
| CPI | 3 | direct | RF | F | 456 | 1.121 | no |
| CPI | 6 | direct | RF | F | 456 | 1.120 | no |
| CPI | 9 | direct | RF | F | 456 | 1.123 | no |
| CPI | 12 | direct | RF | F | 456 | 1.141 | no |
| CPI | 24 | path | RF | X | 456 | 2.086 | no |
| EMP | 1 | direct | RF | F-X-MARX-Level | 456 | 0.934 | yes |
| EMP | 3 | path | RF | F-MARX | 456 | 1.009 | no |
| EMP | 6 | path | BT | F-MARX | 456 | 0.933 | yes |
| EMP | 9 | path | BT | F-MARX | 456 | 1.027 | no |
| EMP | 12 | path | BT | F-MARX | 456 | 1.103 | no |
| EMP | 24 | path | BT | MAF | 456 | 1.171 | no |
| HOUST | 1 | direct | EN | F-Level | 456 | 1.064 | no |
| HOUST | 3 | path | EN | F-Level | 456 | 1.032 | no |
| HOUST | 6 | path | RF | F-X-MARX | 456 | 0.929 | yes |
| HOUST | 9 | direct | RF | F-MAF | 456 | 0.749 | yes |
| HOUST | 12 | direct | RF | F | 456 | 0.914 | yes |
| HOUST | 24 | direct | RF | F | 456 | 0.882 | yes |
| INCOME | 1 | direct | RF | MARX | 456 | 1.051 | no |
| INCOME | 3 | direct | RF | F-MARX | 456 | 1.047 | no |
| INCOME | 6 | path | RF | F-X-MARX | 456 | 1.163 | no |
| INCOME | 9 | path | RF | F-MARX | 456 | 1.295 | no |
| INCOME | 12 | path | RF | F-MARX | 456 | 1.445 | no |
| INCOME | 24 | path | RF | F-X-MARX | 456 | 1.703 | no |
| INDPRO | 1 | direct | RF | F-X-MARX-Level | 456 | 1.190 | no |
| INDPRO | 3 | direct | RF | MARX | 456 | 1.219 | no |
| INDPRO | 6 | path | RF | MARX | 456 | 1.337 | no |
| INDPRO | 9 | path | RF | MARX | 456 | 1.545 | no |
| INDPRO | 12 | path | RF | MARX | 456 | 1.658 | no |
| INDPRO | 24 | direct | RF | F-Level | 456 | 1.034 | no |
| M2 | 1 | direct | RF | X-Level | 456 | 0.961 | yes |
| M2 | 3 | path | AL | X-Level | 456 | 0.993 | yes |
| M2 | 6 | path | RF | F-Level | 456 | 1.320 | no |
| M2 | 9 | direct | RF | F-Level | 456 | 1.169 | no |
| M2 | 12 | direct | BT | F-Level | 456 | 1.103 | no |
| M2 | 24 | path | RF | F-Level | 456 | 1.686 | no |
| PPI | 1 | direct | EN | F-MARX | 456 | 1.008 | no |
| PPI | 3 | direct | EN | MARX | 456 | 1.000 | no |
| PPI | 6 | direct | RF | F | 456 | 1.002 | no |
| PPI | 9 | direct | RF | F | 456 | 1.012 | no |
| PPI | 12 | direct | RF | F | 456 | 1.021 | no |
| PPI | 24 | direct | BT | F-Level | 456 | 1.088 | no |
| RETAIL | 1 | direct | FM | F | 456 | 1.000 | no |
| RETAIL | 3 | path | BT | F-X-MARX | 456 | 1.044 | no |
| RETAIL | 6 | path | AL | F-MARX | 456 | 1.077 | no |
| RETAIL | 9 | direct | BT | F-X-MARX-Level | 456 | 1.020 | no |
| RETAIL | 12 | direct | BT | F-X-Level | 456 | 1.016 | no |
| RETAIL | 24 | direct | BT | F-X-MAF | 456 | 1.046 | no |
| UNRATE | 1 | direct | BT | F-MARX | 456 | 0.947 | yes |
| UNRATE | 3 | direct | RF | F-X-MARX-Level | 456 | 0.991 | yes |
| UNRATE | 6 | path | RF | F-MARX | 456 | 1.150 | no |
| UNRATE | 9 | path | LB | F-X-MARX-Level | 456 | 0.977 | yes |
| UNRATE | 12 | path | LB | F-X-MARX-Level | 456 | 0.965 | yes |
| UNRATE | 24 | direct | BT | F-MAF | 456 | 1.042 | no |

Why the diagnostics can differ from the paper:

- Model caps: RF and BT use ten lightweight trees in the diagnostics, not the
  paper's full tree counts and optimizer searches.
- Feature fitting: diagnostic feature matrices use full-sample fitting to keep
  all early origins; the final replication should use fit-aware expanding
  feature state.
- Hyperparameter search: the diagnostic does not run the appendix GA and
  Bayesian optimization loops.
- Backend: the paper is MATLAB-based, while this run uses Python package
  backends. Tree defaults and optimizer randomness are not table-identical.

## Gap Ledger

| Gap | Current handling |
| --- | --- |
| Exact FRED-MD vintage | start from `2018-01.csv` extracted from the official historical vintage zip; compare nearby 2018 vintages if Dec. 2017 coverage differs |
| MATLAB GA/Bayesian optimizer defaults | use the article's ranges, folds, and iteration counts; record backend difference |
| BIC lag selection for AR/FM | add a small benchmark helper or run fixed `(12, 12, 8)` for the first reconstructed pass |
| Boosted-tree depth interpretation | document whether Python backend uses `max_depth=5` or an exact 5-split/6-leaf equivalent |
| Linear Boosting candidate sampling | current `glmboost` is component-wise boosting; add a candidate-sampling option for exact appendix alignment if needed |
| Full monthly OOS calendar | monthly capped diagnostic now runs all realized months from `1980-01` through `2017-12`; exact replication still needs uncapped learner settings, optimizer search, and fit-aware feature state |
| Fit-aware feature replication | current diagnostics use full-sample feature fitting; next replication pass should use expanding feature state inside each origin |
| Paper captures | add static captures of Table 1, Table 2, and selected Appendix B tables only after the settings page is accepted |
