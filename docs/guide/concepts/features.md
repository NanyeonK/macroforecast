# Features

[Back to User Guide](../index.md)

`macroforecast.feature_engineering` is the direct pandas surface for building
forecast targets and model-ready feature matrices. For strict windowed
forecasting, use `feature_spec(...)`. The spec is fitted by
`macroforecast.forecasting.run(...)` inside each train window and then
transformed for the matching test rows, so stateful operations such as PCA are
estimated only on estimation-window data.

The package organizes feature types into five families used across replication
papers:

- **F (factors)**: principal component or sparse-PCA factors extracted from the
  full predictor set.
- **X (raw lags)**: lagged columns of individual series without dimension
  reduction.
- **MARX (moving-average lag cross)**: mixed lags and moving averages of each
  series, the standard macro predictor design used in McCracken-Ng style papers.
- **MAF (maximum autocorrelation factors)**: rotation of PC factors to maximize
  autocorrelation, useful for persistent macro series.
- **Level**: raw (untransformed) level columns joined alongside the stationary
  predictors; assigned t-code 1 (identity) so official preprocessing passes them
  through unchanged.

## Key Callables

`mf.feature_engineering.feature_spec` stores feature construction choices for
runner-fitted execution. The returned `FeatureSpec` is passed to `forecasting.run`
or to an `Arm`.

```{warning}
**`lags` defaults to `(0, 1)`, so every predictor enters at both `t` and `t-1`.**
It is worth setting explicitly, because it changes the *width* of the predictor
block and therefore what any downstream dimension reduction is reducing:

| `lags` | columns from a 132-series panel | what a PCA over them means |
|---|---|---|
| `0` | 132 | components of the cross-section `X_t` |
| `(0, 1)` (default) | **264** | components of the stacked `[X_t, X_{t-1}]` panel |

For a plain regression the extra lags are just more regressors. For anything that
reduces the block — `pca_step`, `maf_step`, or a model like `far` that runs PCA
*internally* on whatever design it is handed — the factors themselves become a
different object, silently and with no error.

The Stock-Watson diffusion-index model that most macro papers specify is
`X_t = Lambda F_t + u_t`: factors of the cross-section at **one** time index. To
reproduce it, pass `lags=0` and add factor lags deliberately with `lag_step` if
the paper's specification calls for them.

A related asymmetry worth knowing: `pca_step` standardizes before the PCA
(`scale=True`), while `far` centers only (`scale=False`, covariance PCA). Both
conventions are defensible and papers differ, so state which one you want rather
than inheriting a default.
```

```python
import macroforecast as mf

# MARX lags: the default macro predictor design. The marx_step builds the
# increasing-average lag ladder over every predictor column.
marx_features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    predictors="all",
    lags=None,
    feature_steps=[
        mf.feature_engineering.marx_step(name="MARX_X", max_lag=12),
    ],
)

# Pure lag features (no moving averages): lag every predictor 1..12.
lag_features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    predictors="all",
    lags=range(1, 13),
)

# Factor features: extract the first k principal components, then lag them.
factor_features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    predictors="all",
    lags=None,
    feature_steps=[
        mf.feature_engineering.pca_step(name="F", n_components=8, include=False),
        mf.feature_engineering.lag_step(name="F_lag", input="F", lags=range(0, 3)),
    ],
)

# MAF features: maximum autocorrelation factors.
maf_features = mf.feature_engineering.feature_spec(
    target="INDPRO",
    predictors="all",
    lags=None,
    feature_steps=[
        mf.feature_engineering.maf_step(name="MAF", n_components=8, max_lag=12),
    ],
)
```

Target-aware feature steps can also screen predictors inside each runner fit
window before downstream projections:

```python
screened_factors = mf.feature_engineering.feature_spec(
    target="INDPRO",
    horizon=1,
    predictors="all",
    feature_steps=[
        mf.feature_engineering.predictor_screen(
            method="t_stat",       # "delta_r2", "lasso", and "elastic_net" also work
            top_k=40,
            min_k=5,
            controls=["INDPRO"],
            include=False,
        ),
        mf.feature_engineering.pca_step(name="F_screen", input="screen", n_components=4),
    ],
)
```

`predictor_screen(...)` is fitted only on the feature-stage training panel and
the resolved direct target. `controls` are partialled out for scoring and always
retained in the transformed feature set; `min_k` provides a deterministic
fallback when the threshold is too strict.

Direct-average targets include `transform="log_average_value"` for cases where
the forecast object is `log(mean(y[t+1], ..., y[t+h]))` rather than the mean of
log changes.

## Executed walkthrough

For exploration, `mf.build_features` materializes a feature matrix immediately.
Building the MARX ladder over all predictors on a panel slice:

```python
sl = bundle.panel.loc["1960-01":"2000-12"]
fs = mf.build_features(
    sl, target="INDPRO", predictors="all", lags=None,
    feature_steps=[mf.feature_engineering.marx_step(name="MARX_X", max_lag=12)],
)
print(type(fs).__name__, fs.X.shape)
print(list(fs.X.columns[:6]))
```

```text
FeatureSet (94, 1524)
['RPI_ma1_lag1', 'RPI_ma2_lag1', 'RPI_ma3_lag1', 'RPI_ma4_lag1', 'RPI_ma5_lag1', 'RPI_ma6_lag1']
```

The 1524 columns are the 127 predictors each expanded into a 12-step
moving-average lag ladder. The row count is reduced because `drop_missing=True`
removes rows with any gap in the raw slice. Feature engineering works best on a
`PreprocessedData` panel from `reprocess`, which fills those gaps before the
ladder is built. Inside a runner, `feature_spec` fits these same steps on each
train window so that stateful operations such as PCA never see test rows.

## Available steps

`feature_steps` accepts any of these; each links to its signature in the
reference page. Target-aware steps (marked †) are fitted against the resolved
direct target inside the training window, never the test rows.

| step | what it builds |
|---|---|
| `lag_step` | lags of a named block |
| `seasonal_lag_step` | seasonal lags |
| `moving_average_step` | a moving-average ladder |
| `rolling_step` | a rolling mean |
| `marx_step` | the MARX mixed lag/moving-average ladder |
| `pca_step` | principal components |
| `group_pca_step` | principal components within groups |
| `sparse_pca_chen_rohe_step` | Chen-Rohe sparse components |
| `maf_step` | maximum-autocorrelation factors |
| `varimax_step` | an orthogonal varimax rotation |
| `partial_least_squares_step` † | PLS components |
| `sliced_inverse_regression_step` † | SIR directions |
| `predictor_screen` † | a screened predictor subset |
| `scale_step` | standardized / rescaled columns |
| `transform_step` | a deterministic column transform |
| `polynomial_step` | polynomial expansions |
| `interaction_step` | pairwise interactions |
| `nystroem_step` | a Nystroem kernel approximation |
| `random_projection_step` | a Gaussian random projection |
| `hamilton_step` | the Hamilton filter |
| `fourier_step` | Fourier seasonal terms |
| `season_dummy_step` | seasonal dummies |
| `time_step` | deterministic trend / month / quarter / year |
| `custom_step` | your own callable |

## Reference

- [Feature Engineering reference page](../../reference/feature_engineering.md) — full function list including `lag`, `rolling_mean`, `pca_features`, `build_features`, `direct_target`, `average_target`, and `path_targets`.
