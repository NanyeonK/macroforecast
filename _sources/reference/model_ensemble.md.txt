# macroforecast.model_ensemble

[Back to reference](index.md)

`macroforecast.model_ensemble` owns **fit-time model composition**. These
callables receive one aligned `X, y` training sample, fit multiple member models
inside that training window, and return one `ModelFit` whose `predict(X_new)`
returns one forecast series. This is different from
`macroforecast.forecasting.combination`, which combines already-produced
forecast rows after models have been fitted.

## Public Surface

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `bagging` | fit function | Bootstrap or block-bootstrap member fits. |
| `subagging` | fit function | Sampling-without-replacement member fits. |
| `random_subspace` | fit function | Member fits on random feature subsets. |
| `stacking` | fit function | Out-of-fold base predictions plus a meta learner. |
| `super_learner` | fit function | SuperLearner-style OOF convex weighted average. |
| `booging` | fit function | Bagged overfit stochastic boosting with feature perturbation. |
| `BaggingRegressor` | estimator class | Backend for `bagging` and `subagging`. |
| `RandomSubspaceRegressor` | estimator class | Backend for `random_subspace`. |
| `StackingRegressor` | estimator class | Backend for `stacking`. |
| `SuperLearnerRegressor` | estimator class | Backend for `super_learner`. |
| `BoogingRegressor` | estimator class | Backend for `booging`. |
| `MODEL_ENSEMBLE_BASE_ESTIMATORS` | registry | Supported inner estimators for `base`, `models`, and `meta_model`. |
| `MODEL_ENSEMBLE_SPECS` | registry | Registered fit-time ensemble specs. |
| `list_model_ensemble_bases` | helper | Return supported inner estimators and backend names. |
| `get_model_ensemble` | spec helper | Resolve a name, callable, or spec. |
| `list_model_ensemble_specs` | spec helper | Return a registry table. |
| `describe_model_ensemble` | spec helper | Return parameter/default/search-space documentation. |
| `model_ensemble_search_space` | spec helper | Return a preset search space. |
| `custom_model_ensemble` | extension helper | Build a user-owned `ModelSpec` with `family="model_ensemble"`. |

## Boundary

| Question | Use |
| --- | --- |
| Fit several member models inside one training window and expose one predictor? | `macroforecast.model_ensemble` |
| Fit independent models and combine their OOS forecast rows? | `macroforecast.forecasting.combination` |
| Use tree boosting as one estimator, such as XGBoost or LightGBM? | `macroforecast.models` |

## Common Contract

All public fit functions in this namespace use the same callable shape:

| Item | Contract |
| --- | --- |
| Input `X` | pandas-like predictor matrix. Index is preserved in fitted diagnostics. Missing values are aligned with `y` and filled with `0.0` inside member estimators after row alignment. |
| Input `y` | pandas-like target series. Required unless `X` is a `FeatureSet`. |
| Output | `macroforecast.models.ModelFit`. `fit.model` is the callable name, such as `subagging`, and `fit.predict(X_new)` returns one prediction series. |
| Metadata | `fit.metadata` stores the resolved training count, ensemble parameters, base-model choices, seeds, and implementation note. |
| Diagnostics | `fit.diagnostics["model_ensemble"]` stores member/fold diagnostics when available. Point fitted values and residual metrics are also stored by the shared model wrapper. |

Common diagnostics:

| Key | Produced by | Meaning |
| --- | --- | --- |
| `n_members` | all estimator-backed ensembles | Number of fitted member models stored in the ensemble. |
| `member_samples` | `bagging`, `subagging`, `booging` | Compact row-sampling ledger for each member. |
| `member_features` | `random_subspace`, `booging` | Feature subset ledger for each member. |
| `oob_predictions`, `oob_residuals`, `oob_metrics` | `bagging`, `subagging` when OOB rows exist | Out-of-bag diagnostics from member fits that did not use a row. |
| `folds` | `stacking`, `super_learner` | Fold ledger used to create out-of-fold predictions. |
| `oof_predictions` | `stacking`, `super_learner` | Out-of-fold library matrix. |
| `weights`, `oof_risk` | `super_learner` | Convex library weights and OOF MSE by learner. |

Example:

```python
fit = macroforecast.model_ensemble.subagging(
    X_train,
    y_train,
    base="ridge",
    n_estimators=50,
    max_samples=0.632,
)
pred = fit.predict(X_test)
```

Runner usage:

```python
result = macroforecast.forecasting.run(
    panel,
    model="super_learner",
    target="INDPRO",
    horizon=1,
    features=macroforecast.feature_engineering.feature_spec(
        target="INDPRO",
        horizon=1,
        lags=12,
    ),
    params={
        "super_learner": {
            "models": ("ridge", "lasso", "random_forest"),
            "n_splits": 5,
            "weight_method": "nnls",
        }
    },
)
```

When a model ensemble is passed through a runner alias, downstream forecast
combination selects the alias, not the registry name:

```python
result = macroforecast.forecasting.run(
    panel,
    {"bagged": "bagging", "linear": "ridge"},
    combination={
        "linear_plus_bagged": {
            "method": "mean",
            "models": ["linear", "bagged"],
        }
    },
)
```

## R And Paper Alignment

| macroforecast callable | R / paper reference | Alignment | Difference |
| --- | --- | --- | --- |
| `bagging` | [`ipred::bagging` / `ipredbagg`](https://search.r-project.org/CRAN/refmans/ipred/html/bagging.html) | Multiple resampled training sets, one member model per draw, average predictions. | R default is tree-focused; macroforecast allows several sklearn-compatible base regressors. |
| `subagging` | [`ipredbagg(ns < n)`](https://search.r-project.org/CRAN/refmans/ipred/html/bagging.html) | Samples fewer than `n` observations without replacement. | Exposed as a separate callable for clarity. |
| `random_subspace` | [`regRSM::regRSM`](https://search.r-project.org/CRAN/refmans/regRSM/html/regRSM.html) and random-forest `mtry` logic | Repeated random predictor-subspace fits. | General member-level feature bagging, not tree split-level `mtry` and not regRSM's variable-importance final-model selection. |
| `stacking` | [`caretEnsemble::caretStack`](https://search.r-project.org/CRAN/refmans/caretEnsemble/html/caretStack.html) | Fits a meta model on out-of-fold base predictions. | Adds `splitter="forward"` for macro time ordering; R caret examples often use generic CV. |
| `super_learner` | [`SuperLearner::SuperLearner`](https://search.r-project.org/CRAN/refmans/SuperLearner/html/SuperLearner.html) | Uses OOF library predictions and nonnegative weights that sum to one. | Regression-only callable; supports `nnls`, `best`, and `equal` weights, not every R family/loss/plugin. |
| `booging` | [Goulet Coulombe, *To Bag is to Prune*](https://arxiv.org/abs/2008.07063) and the `bagofprunes` R source `Booging(y, X, X.new, ...)` | Samples rows with `sampling_rate`, samples features with `mtry`, optionally appends two perturbed fake feature copies, fits intentionally overfit stochastic boosting members, and averages member predictions. | Uses sklearn `GradientBoostingRegressor` as the boosting backend. Continuous scaling is train-only for leakage-safe estimator semantics, while the R script scales train and `X.new` jointly inside one prediction call. |

The added functions beyond the old `bagging`/`booging` pair are `subagging`,
`random_subspace`, `stacking`, and `super_learner`. They cover the main
fit-time ensemble families that are useful before producing forecast rows:
row-resampling ensembles, feature-subspace ensembles, OOF meta-learner
ensembles, and convex-weight OOF library ensembles.

Not every R ensemble feature is copied. `ipred` double-bagging/bundling is a
classification-oriented extension that passes extra learner predictions into
trees; in macroforecast, cross-model forecast combinations belong in
`macroforecast.forecasting.combination`. `regRSM` variable-importance final
model selection belongs closer to feature selection and is not bundled into this
fit-time ensemble callable.

## Paper Citation And Scope

The Booging implementation is based on:

> Philippe Goulet Coulombe. 2024. *To Bag is to Prune*. arXiv:2008.07063v5.

The method code is cross-checked against the author's public `bagofprunes` R
source, especially `Booging(y, X, X.new, ...)` in
`PGC_Bag_of_Prunes_v200829.R`. The package treats the paper as a method port,
not as a full empirical replication. The goal is to make the randomized
greedy-ensemble mechanism callable inside the macroforecast workflow.

The review-mapped mechanism is:

| Paper idea | Meaning for macroforecast |
| --- | --- |
| Random Forest has a large in-sample versus out-of-sample R-squared wedge | The model can overfit member trees without necessarily damaging the ensemble forecast. |
| Bagging plus perturbation performs implicit pruning | Averaging many randomized greedy paths cancels unstable noise-fitting steps while preserving stable signal-fitting steps. |
| Greedy separability | Early greedy steps are not re-optimized after later overfit steps, so useful structure can survive overfitting. |
| Perfectly random forest null argument | Under pure noise, averaging many random predictions tends toward a sample-mean-like forecast, matching the optimal pruning intuition. |
| LASSO contrast | Global re-optimization along a regularization path weakens the same mechanism; setting the penalty to zero collapses to OLS rather than randomized implicit pruning. |
| Booging | Applies bagging and perturbation to boosted trees, so the stopping point is regularized by averaging randomized overfit boosting paths. |
| MARSquake | Applies the same outer idea to MARS through randomized predictor admission in the forward pass. |

This package currently implements Booging. MARSquake is documented as a related
paper method but is not exposed as a first-class callable yet, because exact
alignment requires either a stable Python analogue of R `earth::earth` with the
custom `allowed` callback or a package-native MARS forward-pass implementation.
The plain MARS model remains available under `macroforecast.models`; that is not
the same as MARSquake.

The paper's empirical evidence spans simulated DGPs, many non-macro benchmark
datasets, and six US macro forecasting tasks covering GDP, unemployment, and
inflation horizons. For macroforecast, the relevant takeaway is operational:
Booging is useful when small noisy macro samples make the exact boosting stopping
point unstable, and the user wants randomized overfitting plus averaging as an
alternative regularization route.

## Bagging, Random Forest, And Booging

These functions expose a common decomposition:

| Method | Row sampling | Feature perturbation | Base learner | macroforecast location |
| --- | --- | --- | --- | --- |
| Bagging | bootstrap or block bootstrap | optional member-level `max_features` | any registered inner base learner | `macroforecast.model_ensemble.bagging` |
| Random-subspace bagging | bootstrap/subsampling | member-level `max_features` | any registered inner base learner | `bagging(..., max_features=...)` or `random_subspace(...)` |
| Random forest | bootstrap plus split-level feature search | split-level tree `mtry` | CART trees | `macroforecast.models.random_forest` |
| Booging | row subsampling | member-level `mtry`, optional fake feature copies | overfit boosted trees | `macroforecast.model_ensemble.booging` |

The distinction matters. `bagging(base="decision_tree", max_features="sqrt")`
is a random-subspace tree ensemble, not an exact random forest, because the
feature subset is drawn once per member rather than at each tree split. Use
`macroforecast.models.random_forest` when the exact random-forest backend is the
target. Use `booging` when the target is Goulet Coulombe's randomized
greedy-boosting ensemble.

## Registered Specs

`MODEL_ENSEMBLE_SPECS` mirrors the model registry but is intentionally separate
from `macroforecast.models.MODEL_SPECS`.

```python
macroforecast.model_ensemble.list_model_ensemble_specs()
```

| Name | Default search | Presets | Backend |
| --- | --- | --- | --- |
| `bagging` | `random` | `small`, `standard`, `wide` | internal member resampling + sklearn-compatible base estimators |
| `subagging` | `random` | `small`, `standard`, `wide` | internal subagging + sklearn-compatible base estimators |
| `random_subspace` | `random` | `small`, `standard`, `wide` | internal random subspace + sklearn-compatible base estimators |
| `stacking` | `random` | `small`, `standard`, `wide` | internal OOF stacking + sklearn-compatible base/meta estimators |
| `super_learner` | `random` | `small`, `standard`, `wide` | internal SuperLearner-style OOF NNLS/equal/best weighting |
| `booging` | `random` | `small`, `standard`, `wide` | internal augmentation/bagging + sklearn.ensemble.GradientBoostingRegressor |

The forecasting runner and `macroforecast.model_selection.search_spec()` both
resolve these names, so `model="bagging"` now means the fit-time ensemble spec,
not a base model spec.

## Inner Base Estimators

`bagging`, `subagging`, and `random_subspace` use `base=...`.
`stacking` and `super_learner` use `models=(...)`, and `stacking` also uses
`meta_model=...`. These names are intentionally narrower than
`macroforecast.models.MODEL_SPECS`: they are inner sklearn-compatible
estimators used inside a fit-time ensemble.

```python
macroforecast.model_ensemble.list_model_ensemble_bases()
```

| Name | Backend |
| --- | --- |
| `ols` | `sklearn.linear_model.LinearRegression` |
| `ridge` | `sklearn.linear_model.Ridge` |
| `lasso` | `sklearn.linear_model.Lasso` |
| `elastic_net` | `sklearn.linear_model.ElasticNet` |
| `decision_tree` | `sklearn.tree.DecisionTreeRegressor` |
| `random_forest` | `sklearn.ensemble.RandomForestRegressor` |
| `extra_trees` | `sklearn.ensemble.ExtraTreesRegressor` |
| `gradient_boosting` | `sklearn.ensemble.GradientBoostingRegressor` |
| `knn` | `sklearn.neighbors.KNeighborsRegressor` |
| `svr` | `sklearn.svm.SVR` |

## bagging

```python
macroforecast.model_ensemble.bagging(
    X,
    y,
    *,
    base="ridge",
    n_estimators=50,
    max_samples=0.8,
    random_state=0,
    base_params=None,
    strategy="standard",
    block_length=4,
    replace=True,
    max_features=None,
)
```

**Input:** `X` is a pandas-like predictor matrix and `y` is a target series.
**Output:** `ModelFit` wrapping `BaggingRegressor`. `predict(X_new)` returns the
mean across member predictions. `predict_quantiles(X_new, levels=...)` returns
empirical quantiles across members.

The fitted object stores out-of-bag diagnostics when any observation is left out
by at least one member: `fit.diagnostics["model_ensemble"]["oob_predictions"]`,
`oob_residuals`, `oob_metrics`, and `n_members`. It also stores
`member_features`, even when `max_features=None`; this makes generic bagging,
random-subspace bagging, and CART-like tree bagging inspectable with the same
metadata contract.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `base` | `"ridge"` | Base estimator name. |
| `n_estimators` | `50` | Number of member fits. |
| `max_samples` | `0.8` | Row sample fraction per member. |
| `random_state` | `0` | Resampling seed. |
| `base_params` | `None` | Parameters passed to the base estimator. |
| `strategy` | `"standard"` | `standard` row draws or moving `block` draws. |
| `block_length` | `4` | Moving-block length when `strategy="block"`. |
| `replace` | `True` | Whether rows are sampled with replacement. |
| `max_features` | `None` | Optional member-level feature-subspace size. Accepts `None`/`"all"`, a fraction, an integer count, `"sqrt"`, or `"log2"`. |

## subagging

```python
macroforecast.model_ensemble.subagging(
    X,
    y,
    *,
    base="ridge",
    n_estimators=50,
    max_samples=0.632,
    random_state=0,
    base_params=None,
    max_features=None,
)
```

`subagging` is `bagging(..., replace=False, strategy="standard")`. It follows
the `ipredbagg(ns < n)` distinction where sampling fewer than `n` rows without
replacement creates subagging rather than bootstrap bagging.

**Input:** `X`, `y` as above.
**Output:** `ModelFit` wrapping `BaggingRegressor` with `fit.model ==
"subagging"`. `predict(X_new)` averages member predictions.

Diagnostics store `member_samples`, OOB diagnostics when available, and
`n_members`. `fit.metadata["replace"]` is always `False`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `base` | `"ridge"` | Base estimator name. |
| `n_estimators` | `50` | Number of subsampled member fits. |
| `max_samples` | `0.632` | Row sample fraction per member. |
| `random_state` | `0` | Resampling seed. |
| `base_params` | `None` | Parameters passed to the base estimator. |
| `max_features` | `None` | Optional member-level feature-subspace size, with the same accepted values as `bagging`. |

## random_subspace

```python
macroforecast.model_ensemble.random_subspace(
    X,
    y,
    *,
    base="ridge",
    n_estimators=100,
    max_features=0.5,
    max_samples=1.0,
    random_state=0,
    base_params=None,
)
```

Each member model sees a random subset of columns. `max_features` can be a
fraction of columns, an integer count, `"sqrt"`, or `"log2"`. This is useful
when `p` is large and the package user wants a fit-time model ensemble distinct
from random-forest split-level `mtry`.

`predict_quantiles(X_new, levels=...)` returns empirical quantiles across random
subspace members. The fitted diagnostics include `member_features`, which records
which columns each member saw.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `base` | `"ridge"` | Base estimator name. |
| `n_estimators` | `100` | Number of random feature-subspace fits. |
| `max_features` | `0.5` | Fraction, integer count, `"sqrt"`, or `"log2"` column count per member. |
| `max_samples` | `1.0` | Row subsample fraction per member. |
| `random_state` | `0` | Feature and row sampling seed. |
| `base_params` | `None` | Parameters passed to the base estimator. |

## stacking

```python
macroforecast.model_ensemble.stacking(
    X,
    y,
    *,
    models=("ridge", "lasso", "random_forest"),
    meta_model="ridge",
    n_splits=5,
    splitter="forward",
    random_state=0,
    model_params=None,
    meta_params=None,
    passthrough=False,
)
```

`stacking` creates out-of-fold base predictions, fits `meta_model` on those
predictions, then refits every base model on the full training sample. For macro
forecasting, the default `splitter="forward"` only validates on later blocks
after earlier training data. `splitter="blocked"` and `splitter="kfold"` are
available when the user wants R-style generic cross-validation behavior.

`models` must contain unique names because `model_params` is keyed by model
name. The fitted diagnostics include the OOF prediction matrix.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `models` | `("ridge", "lasso", "random_forest")` | Base model library. Names must be unique. |
| `meta_model` | `"ridge"` | Meta learner fit on OOF predictions. |
| `n_splits` | `5` | Number of OOF validation folds. |
| `splitter` | `"forward"` | OOF splitter: `forward`, `blocked`, or `kfold`. |
| `random_state` | `0` | Base/meta seed. |
| `model_params` | `None` | Per-base model parameter dictionary keyed by model name. |
| `meta_params` | `None` | Meta-model parameters. |
| `passthrough` | `False` | Whether the meta learner also receives original `X`. |

**Output:** `ModelFit` wrapping `StackingRegressor`. Diagnostics include
`folds` and `oof_predictions`.

## super_learner

```python
macroforecast.model_ensemble.super_learner(
    X,
    y,
    *,
    models=("ridge", "lasso", "random_forest"),
    n_splits=5,
    splitter="forward",
    weight_method="nnls",
    random_state=0,
    model_params=None,
)
```

`super_learner` uses the same OOF library matrix as stacking but fits a convex
weighted average instead of a general meta model. `weight_method="nnls"` uses
nonnegative least squares and normalizes weights to sum to one. `best` gives all
weight to the lowest OOF-MSE learner, matching the discrete Super Learner idea.
`equal` uses equal weights.

The fitted estimator exposes `weights_` and `oof_predictions_`.
`models` must contain unique names because weights and `model_params` are keyed
by model name. `fit.diagnostics["model_ensemble"]` stores `weights`, `oof_risk`,
and `oof_predictions`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `models` | `("ridge", "lasso", "random_forest")` | Base learner library. Names must be unique. |
| `n_splits` | `5` | Number of OOF validation folds. |
| `splitter` | `"forward"` | OOF splitter: `forward`, `blocked`, or `kfold`. |
| `weight_method` | `"nnls"` | `nnls`, `best`, or `equal`. |
| `random_state` | `0` | Base learner seed. |
| `model_params` | `None` | Per-base learner parameters keyed by model name. |

**Output:** `ModelFit` wrapping `SuperLearnerRegressor`. Diagnostics include
`folds`, `oof_predictions`, `oof_risk`, and `weights`.

## booging

```python
macroforecast.model_ensemble.booging(
    X,
    y,
    *,
    B=100,
    sampling_rate=0.75,
    mtry=0.8,
    data_aug=False,
    noise_level=0.3,
    shuffle_rate=0.2,
    n_trees=1000,
    tree_depth=3,
    nu=0.3,
    bf=0.5,
    n_augmented_copies=2,
    scale_continuous=True,
    fix_seeds=True,
    random_state=0,
)
```

`booging` fits many intentionally overfit stochastic gradient-boosting members.
It is the boosted-tree analogue of the randomized greedy ensemble logic behind
random forests: draw rows, draw features, fit an overfit boosted-tree path, and
average over many perturbed paths.

The callable accepts the paper/R-code names directly:

| R code name | macroforecast name | Meaning |
| --- | --- | --- |
| `B` | `B` | Number of overfit boosting members. |
| `sampling.rate` | `sampling_rate` | Row fraction sampled without replacement per member. |
| `mtry` | `mtry` | Feature-subspace size per member. Accepts fractions, integer counts, `"sqrt"`, or `"log2"`. |
| `data.aug` | `data_aug` | Whether to append fake perturbed feature copies. |
| `noise.level` | `noise_level` | Gaussian noise scale for continuous fake copies after standardization. |
| `shuffle.rate` | `shuffle_rate` | Share of rows shuffled inside binary fake copies. |
| `bf` | `bf` | Stochastic boosting subsample share inside each boosted member. |
| `n.trees` | `n_trees` | Boosting stages inside each member. |
| `tree.depth` | `tree_depth` | Inner boosting tree depth. |
| `nu` | `nu` | Inner boosting learning rate. |

Backward-compatible names are still accepted: `sample_frac`,
`inner_n_estimators`, `inner_learning_rate`, `inner_max_depth`,
`inner_subsample`, `max_features`, `da_noise_frac`, and `da_drop_rate`.

Backend parameter mapping:

| R `gbm` / Booging parameter | sklearn / macroforecast target |
| --- | --- |
| `gbm::gbm(n.trees=...)` | `GradientBoostingRegressor(n_estimators=...)` through `n_trees` |
| `gbm::gbm(shrinkage=...)` | `GradientBoostingRegressor(learning_rate=...)` through `nu` |
| `gbm::gbm(interaction.depth=...)` | `GradientBoostingRegressor(max_depth=...)` through `tree_depth` |
| `gbm::gbm(bag.fraction=...)` | `GradientBoostingRegressor(subsample=...)` through `bf` |
| `Booging(..., sampling.rate=...)` | outer row subsampling before each boosted member |
| `Booging(..., mtry=...)` | outer member-level feature subsampling before each boosted member |

When `data_aug=True`, macroforecast follows the R algorithm's fake-column
structure. Continuous variables are standardized and then copied with Gaussian
noise. Binary variables are copied after shuffling a `shuffle_rate` share of
rows. Two fake copies are used by default, matching the R source's `fake1_` and
`fake2_` construction. Prediction uses deterministic fake copies of `X_new`, so
calling `predict()` does not draw new perturbations.

Algorithm:

1. Align `X` and `y` under the standard `ModelFit` contract.
2. Detect binary predictors as columns with exactly two finite unique values.
3. Standardize continuous predictors using the training sample.
4. If `data_aug=True`, append `n_augmented_copies` fake copies:
   continuous fake columns receive Gaussian perturbations, and binary fake
   columns receive row-shuffle perturbations.
5. For each member `b = 1, ..., B`, draw rows at `sampling_rate`, draw columns at
   `mtry`, fit an overfit stochastic gradient-boosting member, and store the row
   and feature ledgers.
6. Predict by averaging member predictions. Quantile forecasts use empirical
   quantiles across member predictions.

**Input:** `X`, `y` as above.
**Output:** `ModelFit` wrapping `BoogingRegressor`. `predict(X_new)` averages
overfit boosting members, and `predict_quantiles(X_new, levels=...)` returns
member-forecast empirical quantiles.

Diagnostics include `member_samples`, `member_features`, `augmentation_summary`,
and `n_members`. `member_features` names original and fake columns; the
augmentation summary reports the number of binary and continuous features, the
number of fake copies, and the leakage boundary for continuous scaling.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `B` | `100` | Number of overfit boosting members. |
| `sampling_rate` | `0.75` | Row sample fraction per member. |
| `mtry` | `0.8` | Feature-subspace size per member. |
| `data_aug` | `False` | Whether to append perturbed fake feature copies. |
| `noise_level` | `0.3` | Gaussian noise scale for continuous fake copies. |
| `shuffle_rate` | `0.2` | Binary-feature row-shuffle share for fake copies. |
| `n_trees` | `1000` | Boosting stages inside each member. |
| `tree_depth` | `3` | Inner boosting tree depth. |
| `nu` | `0.3` | Inner boosting learning rate. |
| `bf` | `0.5` | Stochastic gradient boosting subsample share. For samples below 100 rows, this is floored at `0.4`, matching the R code. |
| `n_augmented_copies` | `2` | Number of fake feature copies when `data_aug=True`. |
| `scale_continuous` | `True` | Standardize continuous variables before fake-copy perturbation. |
| `fix_seeds` | `True` | Use deterministic member seeds analogous to the R source's `set.seed(2020+b)`. |
| `random_state` | `0` | Member, row, column, and augmentation seed. |

Important implementation note: the R script scales continuous `X` and `X.new`
jointly because it receives the training and new matrices in the same function
call. The macroforecast estimator uses train-only scaling, because a standard
fit/predict object must not use future `X_new` information during `fit()`. This
is the deliberate leakage-safe difference; the member sampling, fake-copy
perturbation, boosting backend role, and averaging logic follow the R algorithm.

## Custom Extensions

Use `custom_model_ensemble()` when the user wants a fit-time ensemble that
behaves like a model spec:

```python
spec = macroforecast.model_ensemble.custom_model_ensemble(
    "my_ensemble",
    my_fit_function,
    default_params={"B": 20},
)
```

Use `macroforecast.forecasting.custom_combination()` instead when the custom
logic combines forecast rows after runner execution.
