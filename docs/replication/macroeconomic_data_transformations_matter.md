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

Replication level:
: `reconstructed_design`, not `exact_table_replication`.

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
| Target construction | direct average growth and path-average forecasts | `forecast_policy="direct_average"` and `forecast_policy="path_average"` |
| Loss | RMSE | `mf.metrics.rmse(...)` or RMSE from forecast output |
| Reference benchmark | factor model, FM | package mapping: fixed or BIC-selected `mf.models.far` / OLS factor model |
| Accuracy tests | DM test and MCS | `mf.tests.dm_test(...)`, `mf.tests.model_confidence_set(...)` |

## Target Variables

The paper uses ten monthly targets:

| Paper label | Meaning in the article | Package target key |
| --- | --- | --- |
| Industrial production | Industrial production index | `INDPRO` |
| Employment | Total nonfarm employment | `EMP` |
| Unemployment | Unemployment rate | `UNRATE` |
| Income | Real personal income excluding current transfers | `INCOME` |
| Consumption | Real personal consumption expenditures | `CONS` |
| Retail | Retail and food services sales | `RETAIL` |
| Housing starts | Housing starts | `HOUST` |
| Money | M2 money stock | `M2` |
| Consumer prices | Consumer price index | `CPI` |
| Producer prices | Producer price index | `PPI` |

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
| AR | autoregressive benchmark; lag order selected by BIC | use `mf.models.ar`; add BIC lag search in the runner grid when matching the benchmark |
| FM | Stock-Watson-style factor model; BIC hyperparameter selection mentioned in main text | use factor features with `mf.models.ols` or `mf.models.far`; for ML comparisons keep `(P_y, P_f, k)=(12, 12, 8)` where relevant |
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

### Cell 1: Load a Frozen FRED-MD Panel

```python
import macroforecast as mf

frozen_vintage = "2018-01"  # first candidate after the paper's 2017M12 sample end
bundle = mf.data.load_fred_md(
    vintage=frozen_vintage,
    local_zip_source="data/fred_md_2015_2024_historical_vintages.zip",
)

raw_panel = bundle.panel
raw_metadata = bundle.metadata
```

Expected output:

```text
raw_panel: pandas.DataFrame indexed by monthly date
raw_metadata: metadata dictionary attached to the data bundle
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
targets = ["INDPRO", "EMP", "UNRATE", "INCOME", "CONS", "RETAIL", "HOUST", "M2", "CPI", "PPI"]
horizons = [1, 3, 6, 9, 12, 24]
feature_cases = [
    "F", "F-X", "F-MARX", "F-MAF", "F-Level",
    "F-X-MARX", "F-X-MAF", "F-X-Level", "F-X-MARX-Level",
    "X", "MARX", "MAF", "X-MARX", "X-MAF", "X-Level", "X-MARX-Level",
]
models = ["adaptive_lasso", "elastic_net", "glmboost", "random_forest", "gradient_boosting"]
target_policies = ["direct_average", "path_average"]
```

Expected output:

```text
10 targets x 6 horizons x 16 feature cases x 5 ML learners x 2 target policies,
plus AR and FM benchmarks.
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
    target_lags=range(1, 13),
    target_transform="average_change",
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

### Cell 5: Evaluate Against the FM Benchmark

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

## Gap Ledger

| Gap | Current handling |
| --- | --- |
| Exact FRED-MD vintage | start from `2018-01.csv` extracted from the official historical vintage zip; compare nearby 2018 vintages if Dec. 2017 coverage differs |
| MATLAB GA/Bayesian optimizer defaults | use the article's ranges, folds, and iteration counts; record backend difference |
| BIC lag selection for AR/FM | add a small benchmark helper or run fixed `(12, 12, 8)` for the first reconstructed pass |
| Boosted-tree depth interpretation | document whether Python backend uses `max_depth=5` or an exact 5-split/6-leaf equivalent |
| Linear Boosting candidate sampling | current `glmboost` is component-wise boosting; add a candidate-sampling option for exact appendix alignment if needed |
| Paper captures | add static captures of Table 1, Table 2, and selected Appendix B tables only after the settings page is accepted |
