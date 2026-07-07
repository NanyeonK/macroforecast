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

## Reference

- [Feature Engineering reference page](../../reference/feature_engineering.md) — full function list including `lag`, `rolling_mean`, `pca_features`, `build_features`, `direct_target`, `average_target`, and `path_targets`.
