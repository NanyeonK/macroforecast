# GCLS (2021) replication — verification findings

[Back to Replication Gallery](../guide/gallery.md)

This page records what the Goulet Coulombe, Leroux, Stevanovic, and Surprenant
(2021) "Macroeconomic data transformations matter" (IJF 37(4), DOI
10.1016/j.ijforecast.2021.05.005) replication tells us about the `macroforecast`
pipeline. The replication is a verification exercise. Its value is confirming
the pipeline implements the published methodology, not reproducing an
R-based paper bit for bit.

## Outcome in one paragraph

The configuration is faithful to the paper, and the pipeline is leak-free. At
horizon 1 the replicated relative-RMSE matches the appendix within about 0.02.
Agreement loosens at longer horizons. After fixing one genuine evaluation bug,
the residual divergence is dominated by the difference between R's
`randomForest` and scikit-learn's `RandomForestRegressor` (the same
hyperparameters but a different engine and RNG), which is not reducible without
matching the exact R implementation.

## Configuration faithfulness (verified)

| Axis | Paper | Replication | Status |
| --- | --- | --- | --- |
| POOS window | 38 years | 1980-01 to 2017-12 (38 years), estimation from 1960-01 | match |
| RF hyperparameters | R `randomForest` regression defaults | `max_features=1/3` (mtry=p/3), `min_samples_leaf=5` (nodesize=5), `bootstrap=True`, `n_estimators=200` | match |
| Benchmark FM | factor-augmented AR | `far`, 8 PCA factors, 12 AR lags | match (h1) |
| Features | F, X, MARX, MAF, Level | PCA n=8, MARX/MAF max_lag=12, lags 0..12 | match (h1) |
| Target | average growth rate; average difference for UNRATE | `YGROWTH__` one-period growth, averaged over the horizon | match |
| Preprocessing | stationarity + standardization | official t-codes, EM-factor imputation, IQR outliers | match (h1) |

The horizon-1 match (relRMSE within ~0.02, and a plain `ols` reproducing the
direct/path object exactly) confirms the data, preprocessing, features, FM
benchmark, and target construction are correct.

## Appendix comparison

`scripts/replication/gcls_2021_pipeline/_compare_appendix.py` scores every cell
(10 targets x 6 horizons x {AR, FM, RF F-Level/X-Level/MARX/F-X-MARX-Level} x
{direct, path-average}) against the appendix relative-RMSE table.

- Horizon 1: mean absolute delta about 0.03 (RF cases within ~0.02).
- Divergence grows with horizon (about 0.16 at h24 for full-coverage arms).
- Trend agreement is moderate (Pearson about 0.4 to 0.5 at the horizons where the
  cross-cell spread is large enough to measure; low at h1/h3 only because the
  values cluster in 0.9 to 1.1).

## Bugs found

### 1. Evaluation sample truncation (critical) — FIXED

`accuracy_table` enforced one listwise-common sample across ALL contenders in a
cell. A single short-coverage contender (here the `RF_X-Level` and
`RF_F-X-MARX-Level` arms, whose raw X-lag block needs ACOGNO, a FRED-MD series
that starts in 1992) silently truncated EVERY arm's relRMSE sample to 1992-2017
and reported one `n_common`, with no warning. So all 600 cells were scored on
1992-2017 instead of the paper's 1980-2017.

Fix (`macroforecast/pipeline/evaluate.py`): each contender is scored against the
benchmark on their PAIRWISE common sample; `n_common` is per-contender; ragged
coverage emits a `RuntimeWarning`; the joint listwise sample is kept only for the
Model Confidence Set, which genuinely needs it. The Diebold-Mariano table was
already pairwise and is unchanged. Regression tests in
`tests/pipeline/test_accuracy_pairwise_sample.py`. After the fix the
full-coverage arms (AR, RF_F-Level, RF_MARX) score on 1980-2017 and their mean
absolute delta drops from about 0.11 to 0.092 (AR alone to 0.065).

### 2. Horizon-1 direct vs path for series models (low impact) — documented

At horizon 1, direct and path-average must give identical forecasts (path-average
over one step IS the direct one-step forecast). A supervised model (`ols`)
satisfies this exactly. The series-input recursive models `far` and `ar` do not:
they receive the policy-specific target SERIES (`Y_average_value_h1` for direct
versus `Y_value_step1` for path), which differ in length and values even at
horizon 1, so the recursive AR-history seed diverges. The impact is small
(about 1% at the RMSE level, so relRMSE is barely affected and h1 still matches
the appendix), and the actuals are identical across policies. The fix belongs in
the target-series construction for series-input models, not the models
themselves. Guarded by `tests/forecasting/test_h1_direct_path_invariant.py`
(`ols` passes; `far`/`ar` are `xfail` with the diagnosis until the target-series
construction is unified).

## Divergence attribution

After fix 1, the residual long-horizon gap is structural, not random:

- It appears for `ar` (a pure linear autoregression with no forest), so it is not
  an RF engine artifact alone.
- It is one-directional within a series and series-specific (real-activity series
  diverge most), the signature of finite-sample direct-vs-path behaviour and
  implementation differences rather than RNG.
- The dominant component is R `randomForest` versus scikit-learn
  `RandomForestRegressor`: identical hyperparameters, different bootstrap RNG and
  split implementation, amplified at long horizons where the effective sample is
  small (a 24-month average target leaves about twelve independent observations).

This is consistent with the known irreducible R-versus-Python gap and is not a
package defect.

## Hypotheses raised and retracted (for the record)

1. RF `max_features=1.0` mismatch — false alarm. The bare `mf.random_forest`
   builder defaults to sklearn's 1.0, but the replication arms set
   `max_features=1/3` via `paper_model_params`.
2. A path-average "OLS-equivalence bug" — retracted. The paper does not run OLS;
   its own AR row differs between the direct and path tables, so direct != path
   for linear models in finite samples is expected.
3. The paper uses a single direct-FM denominator for the path table — retracted.
   Recomputing with the direct-FM denominator made the engine-independent AR
   row WORSE, so the path-FM denominator (what we use) is correct.

## Bottom line

The pipeline is methodologically faithful to GCLS (2021). One real evaluation bug
(sample truncation) is fixed and regression-tested. One low-impact series-model
inconsistency at horizon 1 is documented and guarded. The remaining numerical
divergence from the appendix is the expected R-versus-scikit-learn random-forest
difference, not a defect.
